"""Tests for `app.backend.jobs`, the upload -> preprocess -> segment job wiring.

Everything here is synthetic and small: tiny (6, 6, 6) in-memory NIfTI
volumes built with nibabel, never real BraTS data. Real inference
(`app.backend.inference.segment_case`) is monkeypatched in every `run_job`
test except the "no checkpoint configured" one, which deliberately runs the
real (fast, tiny) preprocessing step to prove the checkpoint check fires
with an actionable message rather than a stack trace.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from app.backend import jobs
from app.backend.config import Settings

_SHAPE = (6, 6, 6)
_AFFINE_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
_AFFINE_RAS = np.diag([1.0, 1.0, 1.0, 1.0])


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_job_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Points NVX_JOB_DIR at a fresh tmp_path and clears the module's job dict.

    Both matter: without the env var, `job_root`'s repo-relative default
    would write into the real repo's `outputs/demo_jobs`; without clearing
    `_JOBS`/`_LABEL_CONVENTIONS` (module-level, process-lifetime state),
    jobs from an earlier test in this file would leak into `list_jobs()`.
    """
    monkeypatch.setenv("NVX_JOB_DIR", str(tmp_path / "jobs"))
    jobs._JOBS.clear()
    jobs._LABEL_CONVENTIONS.clear()
    yield
    jobs._JOBS.clear()
    jobs._LABEL_CONVENTIONS.clear()


def _settings(tmp_path: Path, *, checkpoint: Path | None = None) -> Settings:
    """A `Settings` pointed at `tmp_path`, bypassing env vars / caching."""
    return Settings(
        prep_dir=tmp_path / "prep",
        eval_dir=tmp_path / "eval",
        checkpoint=checkpoint if checkpoint is not None else tmp_path / "no_such_checkpoint.pt",
        experiment="baseline_unet3d",
        cache_dir=tmp_path / "cache",
        max_cases=10,
        demo_overlap=0.25,
        report_dir=tmp_path / "reports",
    )


def _nifti_bytes(
    shape: tuple[int, int, int] = _SHAPE,
    affine: np.ndarray = _AFFINE_LPS,
    seed: int = 0,
) -> bytes:
    """Builds a tiny, valid, UNCOMPRESSED NIfTI volume's raw bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    img = nib.Nifti1Image(arr, affine)
    return img.to_bytes()


def _valid_uploads() -> dict[str, bytes]:
    """Four consistent (same shape, same affine) synthetic modality uploads."""
    return {
        "t1": _nifti_bytes(seed=1),
        "t1ce": _nifti_bytes(seed=2),
        "t2": _nifti_bytes(seed=3),
        "flair": _nifti_bytes(seed=4),
    }


def _checkpoint(tmp_path: Path) -> Path:
    """A placeholder checkpoint file -- only its existence is ever checked
    before `segment_case` is monkeypatched away."""
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"placeholder -- segment_case is monkeypatched in these tests")
    return ckpt


# --- 1-9: create_job / validation -------------------------------------------


def test_create_job_with_all_four_roles_succeeds(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = jobs.create_job(settings, _valid_uploads())
    assert job.state == "queued"
    assert job.job_id
    assert job.progress == 0.0
    assert job.case_id == job.job_id
    assert job.error is None
    assert jobs.get_job(job.job_id) is job


def test_missing_role_raises_naming_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    del uploads["t2"]
    with pytest.raises(ValueError, match="t2"):
        jobs.create_job(settings, uploads)


def test_extra_role_raises_naming_it(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["bonus"] = b"unexpected role"
    with pytest.raises(ValueError, match="bonus"):
        jobs.create_job(settings, uploads)


def test_non_nifti_bytes_raise_naming_role(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["t1"] = b"this is definitely not a NIfTI file"
    with pytest.raises(ValueError, match="t1"):
        jobs.create_job(settings, uploads)


def test_mismatched_shapes_raise_naming_both_roles(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["t2"] = _nifti_bytes(shape=(7, 7, 7), seed=99)
    with pytest.raises(ValueError) as exc_info:
        jobs.create_job(settings, uploads)
    message = str(exc_info.value)
    assert "t1" in message
    assert "t2" in message


def test_mismatched_affines_raise_naming_both_roles(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["flair"] = _nifti_bytes(affine=_AFFINE_RAS, seed=99)
    with pytest.raises(ValueError) as exc_info:
        jobs.create_job(settings, uploads)
    message = str(exc_info.value)
    assert "t1" in message
    assert "flair" in message


def test_oversized_payload_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Shrink the cap instead of allocating a real 400 MB+ payload, so this
    # stays fast.
    monkeypatch.setattr(jobs, "MAX_UPLOAD_BYTES", 100)
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["t1"] = b"0" * 101
    with pytest.raises(ValueError, match="t1"):
        jobs.create_job(settings, uploads)


def test_empty_payload_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    uploads = _valid_uploads()
    uploads["flair"] = b""
    with pytest.raises(ValueError, match="flair"):
        jobs.create_job(settings, uploads)


def test_stored_files_use_fixed_role_names_under_job_root(tmp_path: Path) -> None:
    # There is no filename parameter anywhere in create_job's signature --
    # this test just pins the resulting on-disk layout, which is the only
    # place a "malicious filename" concern could even show up.
    settings = _settings(tmp_path)
    job = jobs.create_job(settings, _valid_uploads())

    root = jobs.job_root(settings).resolve()
    raw_dir = (root / job.job_id / "raw").resolve()
    assert root in raw_dir.parents

    names = sorted(p.name for p in raw_dir.iterdir())
    assert names == ["flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz"]

    # Each stored file is a real, loadable, gzip-compressed NIfTI.
    for name in names:
        path = raw_dir / name
        assert path.read_bytes()[:2] == b"\x1f\x8b"
        img = nib.load(str(path))
        assert img.shape == _SHAPE


# --- 10-12: run_job ----------------------------------------------------------


def test_run_job_reaches_done_with_monotonic_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, checkpoint=_checkpoint(tmp_path))
    job = jobs.create_job(settings, _valid_uploads())

    recorded: list[float] = []
    original_update = jobs._update_job

    def _spy_update(job_obj: jobs.Job, **fields: object) -> None:
        original_update(job_obj, **fields)
        recorded.append(job_obj.progress)

    monkeypatch.setattr(jobs, "_update_job", _spy_update)

    def _fake_segment_case(settings_arg, case_id, *, force=False, progress=None):
        if progress is not None:
            progress("loading model", 0.1)
            progress("running inference", 0.3)
            progress("post-processing", 0.85)
            progress("done", 1.0)
        return settings_arg.prep_dir / case_id / "prediction.npy"

    monkeypatch.setattr(jobs.inference, "segment_case", _fake_segment_case)

    result = jobs.run_job(settings, job.job_id)

    assert result.state == "done"
    assert result.progress == 1.0
    assert result.error is None
    assert len(recorded) >= 2
    assert recorded == sorted(recorded)


def test_run_job_records_failure_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, checkpoint=_checkpoint(tmp_path))
    job = jobs.create_job(settings, _valid_uploads())

    def _boom(settings_arg, case_id, *, force=False, progress=None):
        raise RuntimeError("synthetic segmentation failure")

    monkeypatch.setattr(jobs.inference, "segment_case", _boom)

    result = jobs.run_job(settings, job.job_id)  # must not raise

    assert result.state == "failed"
    assert result.error
    assert "synthetic segmentation failure" in result.error


def test_run_job_fails_cleanly_with_no_checkpoint(tmp_path: Path) -> None:
    # No monkeypatching here: preprocessing genuinely runs (tiny, fast), and
    # the checkpoint gate must fire before segment_case is ever reached.
    settings = _settings(tmp_path)  # checkpoint deliberately does not exist
    job = jobs.create_job(settings, _valid_uploads())

    result = jobs.run_job(settings, job.job_id)

    assert result.state == "failed"
    assert result.error
    assert "checkpoint" in result.error.lower()
    assert "Traceback" not in result.error


# --- 13: get_job / list_jobs --------------------------------------------------


def test_get_job_returns_none_for_unknown_id() -> None:
    assert jobs.get_job("no-such-job-id") is None


def test_list_jobs_is_newest_first(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job_a = jobs.create_job(settings, _valid_uploads())
    time.sleep(0.005)
    job_b = jobs.create_job(settings, _valid_uploads())

    all_jobs = jobs.list_jobs()
    ids = [j.job_id for j in all_jobs]
    assert job_a.job_id in ids and job_b.job_id in ids
    assert ids.index(job_b.job_id) < ids.index(job_a.job_id)


# --- 14-15: delete_job --------------------------------------------------------


def test_delete_job_removes_directory_then_second_call_returns_false(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = jobs.create_job(settings, _valid_uploads())
    job_dir = jobs.job_root(settings) / job.job_id
    assert job_dir.is_dir()

    assert jobs.delete_job(settings, job.job_id) is True
    assert not job_dir.exists()
    assert jobs.get_job(job.job_id) is None

    assert jobs.delete_job(settings, job.job_id) is False


def test_delete_job_refuses_path_outside_job_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    jobs.job_root(settings)  # ensure the root itself exists
    with pytest.raises(ValueError, match="job_root"):
        jobs.delete_job(settings, "../escaped")
