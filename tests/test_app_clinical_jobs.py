"""Tests for `app.backend.clinical_jobs`, the raw-DICOM-study clinical job wiring.

Everything here is synthetic, small, and CPU-only: no real DICOM files, no
ANTs/HD-BET, no trained checkpoint, no network access. Follows
`tests/test_app_jobs.py`'s mocking style -- an isolated, tmp_path-backed job
store per test, and heavy/real work (DICOM ingest, clinical preprocessing,
segmentation) monkeypatched away wherever a test is not specifically about
that step.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pytest
from app.backend import clinical_jobs
from app.backend.config import Settings
from scipy.special import logit as inverse_expit

from neurovision.data.dicom_ingest import IngestResult, RoleAssignment, SeriesOutcome

# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_job_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Points NVX_JOB_DIR at a fresh tmp_path and clears the module's job dict.

    Mirrors `tests/test_app_jobs.py`'s identical fixture, applied to
    `clinical_jobs._CLINICAL_JOBS` instead of `jobs._JOBS` -- the two stores
    are independent dicts (see the module docstring), but both resolve
    their on-disk root through the SAME `NVX_JOB_DIR` env var via
    `jobs.job_root`.
    """
    monkeypatch.setenv("NVX_JOB_DIR", str(tmp_path / "jobs"))
    clinical_jobs._CLINICAL_JOBS.clear()
    yield
    clinical_jobs._CLINICAL_JOBS.clear()


def _settings(tmp_path: Path) -> Settings:
    """A `Settings` pointed at `tmp_path`, bypassing env vars / caching.

    Only `jobs.job_root(settings)` resolution matters for this module (see
    its module docstring); the rest of these fields are never read by
    anything under test here.
    """
    return Settings(
        prep_dir=tmp_path / "prep",
        eval_dir=tmp_path / "eval",
        checkpoint=tmp_path / "no_such_checkpoint.pt",
        experiment="baseline_unet3d",
        cache_dir=tmp_path / "cache",
        max_cases=10,
        demo_overlap=0.25,
        report_dir=tmp_path / "reports",
    )


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    """Builds a small in-memory zip archive from filename -> content bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


def _valid_study_zip() -> bytes:
    """A tiny, arbitrarily nested "DICOM study" zip -- content is a placeholder.

    Nothing in these tests calls `ingest_study` for real on this archive's
    contents (it is either not reached, or monkeypatched away), so the
    "DICOM" files are just placeholder bytes; only the archive's own
    structure (nesting, member names) is exercised by `create_clinical_job`.
    """
    return _zip_bytes(
        {
            "STUDY/SERIES1/IM001.dcm": b"placeholder dicom bytes 1",
            "STUDY/SERIES1/IM002.dcm": b"placeholder dicom bytes 2",
            "STUDY/SERIES2/IM001.dcm": b"placeholder dicom bytes 3",
        }
    )


# --- create_clinical_job: valid zip -------------------------------------------


def test_create_clinical_job_with_valid_zip_extracts_and_queues(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())

    assert job.state == "queued"
    assert job.job_id
    assert job.case_id == job.job_id
    assert job.progress == 0.0
    assert job.error is None
    assert clinical_jobs.get_clinical_job(job.job_id) is job

    raw_dicom_dir = clinical_jobs.jobs.job_root(settings) / job.job_id / "raw_dicom"
    extracted = sorted(p.relative_to(raw_dicom_dir).as_posix() for p in raw_dicom_dir.rglob("*"))
    assert extracted == [
        "STUDY",
        "STUDY/SERIES1",
        "STUDY/SERIES1/IM001.dcm",
        "STUDY/SERIES1/IM002.dcm",
        "STUDY/SERIES2",
        "STUDY/SERIES2/IM001.dcm",
    ]
    assert (raw_dicom_dir / "STUDY/SERIES1/IM001.dcm").read_bytes() == b"placeholder dicom bytes 1"


# --- create_clinical_job: rejections ------------------------------------------


def test_create_clinical_job_rejects_empty_payload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        clinical_jobs.create_clinical_job(settings, b"")


def test_create_clinical_job_rejects_oversized_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    payload = _valid_study_zip()
    monkeypatch.setattr(clinical_jobs, "MAX_STUDY_ZIP_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        clinical_jobs.create_clinical_job(settings, payload)


def test_create_clinical_job_size_limit_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    payload = _valid_study_zip()

    # Exactly at the limit: allowed.
    monkeypatch.setattr(clinical_jobs, "MAX_STUDY_ZIP_BYTES", len(payload))
    job = clinical_jobs.create_clinical_job(settings, payload)
    assert job.state == "queued"

    # One byte over: rejected.
    monkeypatch.setattr(clinical_jobs, "MAX_STUDY_ZIP_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        clinical_jobs.create_clinical_job(settings, payload)


def test_create_clinical_job_rejects_non_zip_payload(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="not a valid zip"):
        clinical_jobs.create_clinical_job(settings, b"this is definitely not a zip archive")


def test_create_clinical_job_rejects_path_escaping_member_and_extracts_nothing(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    malicious = _zip_bytes(
        {
            "safe_first.dcm": b"this looks fine",
            "../../escape.dcm": b"zip-slip payload",
        }
    )

    with pytest.raises(ValueError, match=r"\.\./\.\./escape\.dcm"):
        clinical_jobs.create_clinical_job(settings, malicious)

    # Nothing was extracted -- not even the "safe" member that sorted before
    # the escaping one -- and no job was queued.
    assert clinical_jobs.list_clinical_jobs() == []
    job_root = clinical_jobs.jobs.job_root(settings)
    leaked = list(job_root.rglob("safe_first.dcm")) + list(job_root.rglob("escape.dcm"))
    assert leaked == []


# --- _live_conformal_band_width -----------------------------------------------


def _logits_with_region_probs(probs: list[float]) -> np.ndarray:
    """A `(1, D)`-shaped logits array (one region channel) with the given probabilities.

    `_live_conformal_band_width` only ever reads `logits[region_channel]`, so
    a single-channel, 1-D "volume" is enough to hand-compute the expected
    voxel counts.
    """
    arr = np.asarray(probs, dtype=np.float64)
    return np.expand_dims(inverse_expit(arr), axis=0)  # (1, D)


def test_live_conformal_band_width_matches_hand_computed_ratio() -> None:
    # Probabilities: 5 voxels, at the reference threshold (0.5) three are
    # above (0.9, 0.6, 0.51) and two are at/below (0.5, 0.1) -> ref_voxels=3.
    # At the fitted threshold (0.8), only one voxel (0.9) survives ->
    # fitted_voxels=1. Expected ratio: 1 / 3.
    logits = _logits_with_region_probs([0.9, 0.6, 0.51, 0.5, 0.1])
    ratio = clinical_jobs._live_conformal_band_width(
        logits, region_channel=0, fitted_threshold=0.8, reference_threshold=0.5
    )
    assert ratio == pytest.approx(1.0 / 3.0)


def test_live_conformal_band_width_empty_reference_mask_is_nan() -> None:
    # Every probability is below the reference threshold -> ref_voxels=0.
    logits = _logits_with_region_probs([0.1, 0.2, 0.05])
    ratio = clinical_jobs._live_conformal_band_width(
        logits, region_channel=0, fitted_threshold=0.05, reference_threshold=0.5
    )
    assert ratio != ratio  # NaN != NaN


# --- _load_conformal_fitted_thresholds ----------------------------------------


def _write_fit_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_load_conformal_fitted_thresholds_finds_alpha_0_1_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_path = tmp_path / "fit.json"
    _write_fit_json(
        fit_path,
        {
            "WT__alpha_0.1": {"region": "WT", "alpha": 0.1, "threshold": 0.42},
            "TC__alpha_0.1": {"region": "TC", "alpha": 0.1, "threshold": 0.37},
        },
    )
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: fit_path)

    result = clinical_jobs._load_conformal_fitted_thresholds(["WT", "TC"], alpha=0.1)
    assert result == {"WT": 0.42, "TC": 0.37}


def test_load_conformal_fitted_thresholds_missing_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_path = tmp_path / "fit.json"
    _write_fit_json(fit_path, {"WT__alpha_0.1": {"threshold": 0.42}})
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: fit_path)

    with pytest.raises(ValueError, match="TC__alpha_0.1"):
        clinical_jobs._load_conformal_fitted_thresholds(["WT", "TC"], alpha=0.1)


def test_load_conformal_fitted_thresholds_null_threshold_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit_path = tmp_path / "fit.json"
    _write_fit_json(fit_path, {"WT__alpha_0.1": {"threshold": None}})
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: fit_path)

    with pytest.raises(ValueError, match="threshold=null"):
        clinical_jobs._load_conformal_fitted_thresholds(["WT"], alpha=0.1)


def test_load_conformal_fitted_thresholds_missing_file_returns_empty_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    missing_path = tmp_path / "does_not_exist" / "fit.json"
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: missing_path)

    with caplog.at_level(logging.WARNING, logger=clinical_jobs.logger.name):
        result = clinical_jobs._load_conformal_fitted_thresholds(["WT"], alpha=0.1)

    assert result == {}
    assert any("no fit.json" in record.message for record in caplog.records)


# --- _ingest_result_to_dict ----------------------------------------------------


def test_ingest_result_to_dict_round_trips_to_json() -> None:
    result = IngestResult(
        paths={"t1": Path("/tmp/whatever/t1.nii.gz"), "t1ce": Path("/tmp/whatever/t1ce.nii.gz")},
        assignments={
            "1.2.3": RoleAssignment(
                role="t1",
                score=5.0,
                reasons=("description token(s) ['t1'] indicate T1-weighted",),
                outcome=SeriesOutcome.ASSIGNED,
            ),
            "1.2.4": RoleAssignment(
                role=None,
                score=0.0,
                reasons=("image_type contains ['LOCALIZER']",),
                outcome=SeriesOutcome.REJECTED,
            ),
        },
        missing_roles=("t2", "flair"),
        rejected=(("1.2.4", "image_type contains ['LOCALIZER']"),),
        warnings=("some warning",),
    )

    payload = clinical_jobs._ingest_result_to_dict(result)

    # Must not raise -- every value is JSON-serialisable.
    encoded = json.dumps(payload)
    assert encoded

    assert payload["paths"]["t1"] == "/tmp/whatever/t1.nii.gz"
    assert payload["assignments"]["1.2.3"]["role"] == "t1"
    assert payload["assignments"]["1.2.3"]["outcome"] == "assigned"
    assert payload["assignments"]["1.2.4"]["outcome"] == "rejected"
    assert payload["missing_roles"] == ["t2", "flair"]
    assert payload["rejected"] == [
        {"series_uid": "1.2.4", "reason": "image_type contains ['LOCALIZER']"}
    ]
    assert payload["warnings"] == ["some warning"]


# --- refusal-vs-failure state distinction -------------------------------------


def test_run_clinical_job_with_no_assignable_series_is_refused_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())

    def _fake_ingest_study(cfg, study_dir, out_dir):
        return IngestResult(
            paths={},
            assignments={},
            missing_roles=("t1", "t1ce", "t2", "flair"),
            rejected=(),
            warnings=("no DICOM series matched any role",),
        )

    monkeypatch.setattr(clinical_jobs, "ingest_study", _fake_ingest_study)

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "refused"
    assert result.state != "failed"
    assert result.error
    assert "t1" in result.error
    assert result.ingest_result is not None
    assert result.ingest_result["paths"] == {}


# --- get / list / delete / start: quick sanity, mirroring test_app_jobs.py ----


def test_get_clinical_job_returns_none_for_unknown_id() -> None:
    assert clinical_jobs.get_clinical_job("no-such-job-id") is None


def test_delete_clinical_job_removes_directory_then_second_call_returns_false(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    job_dir = clinical_jobs.jobs.job_root(settings) / job.job_id
    assert job_dir.is_dir()

    assert clinical_jobs.delete_clinical_job(settings, job.job_id) is True
    assert not job_dir.exists()
    assert clinical_jobs.get_clinical_job(job.job_id) is None
    assert clinical_jobs.delete_clinical_job(settings, job.job_id) is False


def test_delete_clinical_job_refuses_path_outside_job_root(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    clinical_jobs.jobs.job_root(settings)  # ensure the root itself exists
    with pytest.raises(ValueError, match="job_root"):
        clinical_jobs.delete_clinical_job(settings, "../escaped")
