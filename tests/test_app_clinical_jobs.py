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
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from app.backend import clinical_jobs, inference, volumes
from app.backend.config import Settings
from scipy.special import logit as inverse_expit

from neurovision.anatomy.atlas import Atlas, AtlasLabels, AtlasStructure
from neurovision.data.clinical_preprocess import PreprocessResult
from neurovision.data.dicom_ingest import ROLES, IngestResult, RoleAssignment, SeriesOutcome
from neurovision.inference.gatekeeper import Decision, GateDecision
from neurovision.inference.input_qc import InputQCReport, Severity

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


# --- clinical_conformal_band_mask ---------------------------------------------


def _region_logits(probs: list[float], region_channel: int, num_channels: int = 3) -> np.ndarray:
    """A `(num_channels, 1, 1, len(probs))` logits array with `probs` on one channel.

    The other channels are left at 0 (prob 0.5) -- `clinical_conformal_band_mask`
    only ever reads `logits[region_channel]`, so their value never matters.
    """
    region = inverse_expit(np.asarray(probs, dtype=np.float64))
    logits = np.zeros((num_channels, 1, 1, len(probs)), dtype=np.float64)
    logits[region_channel, 0, 0, :] = region
    return logits


def test_clinical_conformal_band_mask_matches_hand_computed_values() -> None:
    # 5 voxels at reference_threshold=0.5, fitted_threshold=0.2 (more
    # permissive -- lower threshold flags MORE voxels):
    #   0.90 -> in reference (>0.5) and in conservative (>0.2)      -> 255
    #   0.60 -> in reference and in conservative                    -> 255
    #   0.30 -> NOT in reference, but in conservative (>0.2)        -> 128
    #   0.10 -> not in reference, not in conservative (<0.2)        -> 0
    #   0.05 -> not in reference, not in conservative               -> 0
    logits = _region_logits([0.90, 0.60, 0.30, 0.10, 0.05], region_channel=1)

    band = clinical_jobs.clinical_conformal_band_mask(
        logits, region_channel=1, fitted_threshold=0.2, reference_threshold=0.5
    )

    assert band.dtype == np.uint8
    assert band.shape == (1, 1, 5)
    np.testing.assert_array_equal(band[0, 0, :], np.array([255, 255, 128, 0, 0], dtype=np.uint8))


def test_clinical_conformal_band_mask_raises_on_invariant_violation() -> None:
    # fitted_threshold (0.8) is on the WRONG side of reference_threshold
    # (0.5) for this project's conformal loss: a voxel at prob=0.6 is inside
    # the reference mask (0.6 > 0.5) but NOT inside the "conservative" mask
    # (0.6 is not > 0.8) -- the nesting the function promises is violated.
    logits = _region_logits([0.9, 0.6, 0.3], region_channel=2)  # WT channel

    with pytest.raises(ValueError) as exc_info:
        clinical_jobs.clinical_conformal_band_mask(
            logits, region_channel=2, fitted_threshold=0.8, reference_threshold=0.5
        )
    message = str(exc_info.value)
    assert "WT" in message
    assert "0.8" in message
    assert "0.5" in message


# --- load_clinical_uncertainty ------------------------------------------------


def test_load_clinical_uncertainty_matches_entropy_from_logits(tmp_path: Path) -> None:
    settings = clinical_jobs.clinical_segmentation_settings(tmp_path / "prep", tmp_path / "cache")
    case_id = "case_x"
    logits = np.random.default_rng(0).normal(size=(3, 4, 5, 6)).astype(np.float32)
    logits_path = inference.cached_logits_path(settings, case_id)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, logits)

    data = volumes.load_clinical_uncertainty(case_id, settings)

    expected = (volumes.entropy_from_logits(logits) * 255.0).astype(np.uint8)
    actual = np.frombuffer(data, dtype=np.uint8).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)


def test_load_clinical_uncertainty_raises_file_not_found_when_uncached(tmp_path: Path) -> None:
    settings = clinical_jobs.clinical_segmentation_settings(tmp_path / "prep", tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        volumes.load_clinical_uncertainty("no_such_case", settings)


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


# --- Grad-CAM wiring: failure isolation ---------------------------------------


def _fake_ingest_study_all_roles(cfg, study_dir, out_dir):
    """A fake E1 that reports every role assigned -- reaches past the E1 refusal."""
    return IngestResult(
        paths={role: Path(f"/fake/{role}.nii.gz") for role in ROLES},
        assignments={},
        missing_roles=(),
        rejected=(),
        warnings=(),
    )


def _fake_load_volume_infos(paths):
    """A fake E3 volume loader -- its return value only matters to `run_input_qc`,
    which is also faked below, so a placeholder pair is enough."""
    return None, None


def _fake_run_input_qc(cfg, volumes, brain_mask=None):
    """A fake E3 that always passes -- reaches past both input-QC refusal points."""
    return InputQCReport(verdict=Severity.OK, findings=())


def _fake_preprocess_clinical_study(cfg, inputs, out_dir=None):
    """A fake E2 that reports every role written -- reaches past the missing-role refusal."""
    return PreprocessResult(
        plan=None,
        outputs={role: Path(f"/fake/prep_{role}.nii.gz") for role in ROLES},
        brain_mask=Path("/fake/no_such_brain_mask.nii.gz"),
        stages_run=(),
        warnings=(),
    )


def _wire_full_pipeline_to_gatekeeper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_shape: tuple[int, int, int] = (8, 8, 8)
) -> None:
    """Fakes every clinical-job stage up to (and including) the gatekeeper decision.

    Everything here is a placeholder: this test is about the Grad-CAM wiring that
    runs AFTER the gatekeeper, not about E1/E2/E3/the gatekeeper's own logic (each
    already covered by their own test modules). The gatekeeper is forced to
    PROCEED regardless of the (fake, meaningless) signals computed along the way,
    so `run_clinical_job` reaches the new Grad-CAM block deterministically.
    """
    checkpoint = tmp_path / "clinical_checkpoint.pt"
    checkpoint.write_bytes(b"placeholder -- inference.segment_case is monkeypatched below")
    monkeypatch.setenv("NVX_CLINICAL_CHECKPOINT", str(checkpoint))

    monkeypatch.setattr(clinical_jobs, "ingest_study", _fake_ingest_study_all_roles)
    monkeypatch.setattr(clinical_jobs, "load_volume_infos", _fake_load_volume_infos)
    monkeypatch.setattr(clinical_jobs, "run_input_qc", _fake_run_input_qc)
    monkeypatch.setattr(clinical_jobs, "preprocess_clinical_study", _fake_preprocess_clinical_study)

    def _fake_preprocess_case(case, out_dir, **kwargs):
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "image.npy", np.zeros((4, *image_shape), dtype=np.float16))

    monkeypatch.setattr(clinical_jobs, "preprocess_case", _fake_preprocess_case)

    def _fake_segment_case(settings, case_id, *, save_logits=False, progress=None):
        pred_path = inference.cached_prediction_path(settings, case_id)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, np.zeros(image_shape, dtype=np.uint8))
        if save_logits:
            logits_path = inference.cached_logits_path(settings, case_id)
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(logits_path, np.zeros((3, *image_shape), dtype=np.float16))
        return pred_path

    monkeypatch.setattr(inference, "segment_case", _fake_segment_case)
    monkeypatch.setattr(clinical_jobs, "_load_qc_model_and_cfg", lambda qc_checkpoint: (None, None))
    monkeypatch.setattr(
        clinical_jobs, "_conformal_fit_path", lambda: tmp_path / "does_not_exist" / "fit.json"
    )
    monkeypatch.setattr(
        clinical_jobs,
        "run_gatekeeper",
        lambda cfg, signals: GateDecision(decision=Decision.PROCEED, verdicts=()),
    )


def test_run_clinical_job_gradcam_failure_isolated_other_region_still_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Grad-CAM failure for one region must not fail the job or skip the other region."""
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path)

    calls: list[str] = []

    def _fake_explain_case(job_settings, case_id, region, **kwargs):
        calls.append(region)
        if region == "WT":
            raise RuntimeError("sentinel Grad-CAM failure for WT")
        return Path("/fake/gradcam_tc.npy")

    monkeypatch.setattr(inference, "explain_case", _fake_explain_case)

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    # Both regions were genuinely attempted -- the WT failure did not also skip
    # TC, proving the two calls sit in their own, independent try/except blocks.
    assert set(calls) == {"WT", "TC"}


# --- DICOM-SEG export (E6) wiring: _export_dicom_seg unit-level -------------


def _write_nifti(path: Path, shape: tuple[int, int, int], affine: np.ndarray | None = None) -> None:
    """Writes a small all-zero float32 NIfTI -- content never matters to these tests,
    only shape/affine, since every mask-carrying value is asserted separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.zeros(shape, dtype=np.float32), affine if affine is not None else np.eye(4)
        ),
        str(path),
    )


def _build_export_dicom_seg_fixture(
    tmp_path: Path, shape: tuple[int, int, int] = (4, 5, 6)
) -> dict[str, object]:
    """Builds everything `_export_dicom_seg` needs directly, with NO cropping
    (bbox == full shape) so `uncrop_to_original` is a pure identity and the
    chain can be checked precisely without a real crop/uncrop story.
    """
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    job_dir = clinical_jobs.jobs.job_root(settings) / job.job_id

    job_prep_dir = job_dir / "prep"
    job_cache_dir = job_dir / "cache"
    clinical_settings = clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)

    case_dir = job_prep_dir / job.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "cropped_shape": list(shape),
        "original_shape": list(shape),
        "bbox": [[0, shape[0]], [0, shape[1]], [0, shape[2]]],
        "spacing": [1.0, 1.0, 1.0],
    }
    (case_dir / "meta.json").write_text(json.dumps(meta))

    cropped_class_map = np.zeros(shape, dtype=np.uint8)
    cropped_class_map[1, 1, 1] = 3  # one ET voxel
    pred_path = inference.cached_prediction_path(clinical_settings, job.case_id)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, cropped_class_map)

    atlas_t1ce_path = tmp_path / "atlas_t1ce.nii.gz"
    _write_nifti(atlas_t1ce_path, shape)

    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1ce").mkdir(parents=True, exist_ok=True)

    plan = SimpleNamespace(center_role="t1ce")
    preprocess_result = PreprocessResult(
        plan=plan,
        outputs={"t1ce": atlas_t1ce_path},
        brain_mask=tmp_path / "no_such_brain_mask.nii.gz",
        stages_run=(),
        warnings=(),
        transformations_dir=transformations_dir,
    )

    t1ce_native_path = tmp_path / "native_t1ce.nii.gz"
    _write_nifti(t1ce_native_path, shape)

    series_uid = "1.2.3.series.t1ce"
    ingest_result = IngestResult(
        paths={"t1ce": t1ce_native_path},
        assignments={
            series_uid: RoleAssignment(
                role="t1ce", score=5.0, reasons=(), outcome=SeriesOutcome.ASSIGNED
            )
        },
        missing_roles=(),
        rejected=(),
        warnings=(),
    )

    return {
        "job": job,
        "job_dir": job_dir,
        "clinical_settings": clinical_settings,
        "preprocess_result": preprocess_result,
        "ingest_result": ingest_result,
        "cropped_class_map": cropped_class_map,
        "series_uid": series_uid,
        "shape": shape,
    }


def test_class_map_to_regions_matches_hand_computed_values() -> None:
    class_map = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)  # shape (1, 1, 4)
    regions = clinical_jobs._class_map_to_regions(class_map)
    assert regions.shape == (3, 1, 1, 4)
    et, tc, wt = regions[0, 0, 0], regions[1, 0, 0], regions[2, 0, 0]
    np.testing.assert_array_equal(et, [0, 0, 0, 1])
    np.testing.assert_array_equal(tc, [0, 1, 0, 1])
    np.testing.assert_array_equal(wt, [0, 1, 1, 1])


def test_series_uid_for_role_finds_assigned_uid_and_returns_none_when_absent() -> None:
    ingest_result = IngestResult(
        paths={},
        assignments={
            "uid-1": RoleAssignment(
                role="t1", score=1.0, reasons=(), outcome=SeriesOutcome.ASSIGNED
            ),
            "uid-2": RoleAssignment(
                role="t1ce", score=1.0, reasons=(), outcome=SeriesOutcome.ASSIGNED
            ),
        },
        missing_roles=(),
        rejected=(),
        warnings=(),
    )
    assert clinical_jobs._series_uid_for_role(ingest_result, "t1ce") == "uid-2"
    assert clinical_jobs._series_uid_for_role(ingest_result, "flair") is None


def test_export_dicom_seg_chain_order_and_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_export_dicom_seg_fixture(tmp_path)
    job = fixture["job"]
    job_dir = fixture["job_dir"]
    shape = fixture["shape"]

    # Step 1: uncrop_to_original on a full-extent bbox is a no-op, so the
    # ATLAS-SPACE class map handed to resample_mask_to_source must be
    # bit-identical to the cached (cropped) prediction.
    expected_atlas_mask = fixture["cropped_class_map"]

    # A native-space class map with DIFFERENT values than the atlas one, so
    # the test can tell whether write_dicom_seg's regions came from the
    # RESAMPLED map (correct) or the pre-resample atlas map (a step-ordering
    # bug: resampling a class map, then converting to regions, not the
    # reverse).
    native_class_map = np.zeros(shape, dtype=np.uint8)
    native_class_map[2, 2, 2] = 1  # necrotic core -- a different class/voxel than the atlas map

    resample_calls: list[dict] = []

    def _fake_resample_mask_to_source(
        mask, atlas_affine, transformations_dir, target_role, target_native_path, out_dir
    ):
        resample_calls.append(
            {
                "mask": np.array(mask),
                "atlas_affine": np.array(atlas_affine),
                "transformations_dir": Path(transformations_dir),
                "target_role": target_role,
                "target_native_path": Path(target_native_path),
                "out_dir": Path(out_dir),
            }
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "resampled.nii.gz"
        nib.save(nib.Nifti1Image(native_class_map, np.eye(4)), str(out_path))
        return out_path

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _fake_resample_mask_to_source
    )

    write_calls: list[dict] = []

    def _fake_write_dicom_seg(cfg, regions, source_datasets, out_path):
        write_calls.append(
            {
                "cfg": cfg,
                "regions": np.array(regions),
                "source_datasets": source_datasets,
                "out_path": Path(out_path),
            }
        )
        return Path(out_path)

    monkeypatch.setattr("neurovision.reporting.dicom_seg.write_dicom_seg", _fake_write_dicom_seg)

    collect_calls: list[dict] = []

    def _fake_collect_source_datasets(raw_dicom_dir, series_uid):
        collect_calls.append({"raw_dicom_dir": Path(raw_dicom_dir), "series_uid": series_uid})
        return ["dataset_a", "dataset_b"]

    monkeypatch.setattr(clinical_jobs, "_collect_source_datasets", _fake_collect_source_datasets)

    sentinel_cfg = object()
    result = clinical_jobs._export_dicom_seg(
        job,
        fixture["preprocess_result"],
        fixture["ingest_result"],
        fixture["clinical_settings"],
        job_dir,
        sentinel_cfg,
    )

    assert result == job_dir / "dicom_seg" / f"{job.case_id}.dcm"

    # resample_mask_to_source got the FULL uncropped atlas-space CLASS MAP,
    # not region channels, before any region split.
    assert len(resample_calls) == 1
    call = resample_calls[0]
    np.testing.assert_array_equal(call["mask"], expected_atlas_mask)
    assert call["target_role"] == "t1ce"
    assert call["transformations_dir"] == fixture["preprocess_result"].transformations_dir
    assert call["target_native_path"] == fixture["ingest_result"].paths["t1ce"]
    assert call["out_dir"] == job_dir / "dicom_seg_work"

    # write_dicom_seg received regions derived from the RESAMPLED
    # (native-space) class map -- proving the region split happens AFTER
    # resampling, not before.
    assert len(write_calls) == 1
    write_call = write_calls[0]
    assert write_call["cfg"] is sentinel_cfg
    expected_regions = clinical_jobs._class_map_to_regions(native_class_map)
    np.testing.assert_array_equal(write_call["regions"], expected_regions)
    assert write_call["source_datasets"] == ["dataset_a", "dataset_b"]
    assert write_call["out_path"] == job_dir / "dicom_seg" / f"{job.case_id}.dcm"

    # _collect_source_datasets was asked for the series_uid E1 assigned to
    # the center role, under this job's own raw_dicom directory.
    assert len(collect_calls) == 1
    assert collect_calls[0]["raw_dicom_dir"] == job_dir / "raw_dicom"
    assert collect_calls[0]["series_uid"] == fixture["series_uid"]


def test_export_dicom_seg_write_dicom_seg_refusal_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_export_dicom_seg_fixture(tmp_path)
    shape = fixture["shape"]

    def _fake_resample_mask_to_source(
        mask, atlas_affine, transformations_dir, target_role, target_native_path, out_dir
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "resampled.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4)), str(out_path))
        return out_path

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _fake_resample_mask_to_source
    )
    monkeypatch.setattr(
        clinical_jobs, "_collect_source_datasets", lambda raw_dicom_dir, series_uid: ["dataset_a"]
    )

    def _refusing_write_dicom_seg(cfg, regions, source_datasets, out_path):
        raise ValueError("write_dicom_seg: refusing to write -- geometry mismatch")

    monkeypatch.setattr(
        "neurovision.reporting.dicom_seg.write_dicom_seg", _refusing_write_dicom_seg
    )

    result = clinical_jobs._export_dicom_seg(
        fixture["job"],
        fixture["preprocess_result"],
        fixture["ingest_result"],
        fixture["clinical_settings"],
        fixture["job_dir"],
        cfg=None,
    )

    assert result is None
    assert not (fixture["job_dir"] / "dicom_seg" / f"{fixture['job'].case_id}.dcm").exists()


def test_export_dicom_seg_unexpected_exception_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_export_dicom_seg_fixture(tmp_path)

    def _raising_resample(*args, **kwargs):
        raise RuntimeError("sentinel: resample blew up")

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _raising_resample
    )

    with pytest.raises(RuntimeError, match="sentinel"):
        clinical_jobs._export_dicom_seg(
            fixture["job"],
            fixture["preprocess_result"],
            fixture["ingest_result"],
            fixture["clinical_settings"],
            fixture["job_dir"],
            cfg=None,
        )


def test_export_dicom_seg_no_series_assigned_to_center_role_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_export_dicom_seg_fixture(tmp_path)
    shape = fixture["shape"]

    # The series_uid check (step 4) runs AFTER uncrop/resample/region-split
    # (steps 1-3), so resample_mask_to_source is still reached and must be
    # faked here -- this test is about the series_uid lookup, not about
    # re-proving resample_mask_to_source itself.
    def _fake_resample_mask_to_source(
        mask, atlas_affine, transformations_dir, target_role, target_native_path, out_dir
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "resampled.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4)), str(out_path))
        return out_path

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _fake_resample_mask_to_source
    )

    empty_ingest_result = IngestResult(
        paths=fixture["ingest_result"].paths,
        assignments={},
        missing_roles=(),
        rejected=(),
        warnings=(),
    )

    result = clinical_jobs._export_dicom_seg(
        fixture["job"],
        fixture["preprocess_result"],
        empty_ingest_result,
        fixture["clinical_settings"],
        fixture["job_dir"],
        cfg=None,
    )

    assert result is None


def test_export_dicom_seg_no_matching_raw_dicom_files_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_export_dicom_seg_fixture(tmp_path)
    shape = fixture["shape"]

    def _fake_resample_mask_to_source(
        mask, atlas_affine, transformations_dir, target_role, target_native_path, out_dir
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "resampled.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4)), str(out_path))
        return out_path

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _fake_resample_mask_to_source
    )
    monkeypatch.setattr(
        clinical_jobs, "_collect_source_datasets", lambda raw_dicom_dir, series_uid: []
    )

    result = clinical_jobs._export_dicom_seg(
        fixture["job"],
        fixture["preprocess_result"],
        fixture["ingest_result"],
        fixture["clinical_settings"],
        fixture["job_dir"],
        cfg=None,
    )

    assert result is None


# --- DICOM-SEG export (E6) wiring: run_clinical_job integration --------------


def _wire_dicom_seg_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_shape: tuple[int, int, int] = (8, 8, 8)
) -> str:
    """Extends `_wire_full_pipeline_to_gatekeeper` so `_export_dicom_seg`'s chain
    reaches all the way to `write_dicom_seg`: a real `meta.json`, a real
    atlas-space + native-space NIfTI for the center role, an E1 assignment
    naming a series_uid for it, and fakes for the two dependencies a caller of
    `write_dicom_seg` needs but does not itself own
    (`resample_mask_to_source`, `_collect_source_datasets`). Returns the
    series_uid used.
    """
    _wire_full_pipeline_to_gatekeeper(monkeypatch, tmp_path, image_shape=image_shape)

    def _fake_preprocess_case_with_meta(case, out_dir, **kwargs):
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "image.npy", np.zeros((4, *image_shape), dtype=np.float16))
        meta = {
            "cropped_shape": list(image_shape),
            "original_shape": list(image_shape),
            "bbox": [[0, image_shape[0]], [0, image_shape[1]], [0, image_shape[2]]],
            "spacing": [1.0, 1.0, 1.0],
        }
        (case_dir / "meta.json").write_text(json.dumps(meta))

    monkeypatch.setattr(clinical_jobs, "preprocess_case", _fake_preprocess_case_with_meta)

    series_uid = "1.2.840.dicom.series.t1ce"
    t1ce_native_path = tmp_path / "t1ce_native.nii.gz"
    _write_nifti(t1ce_native_path, image_shape)

    def _fake_ingest_study_with_assignment(cfg, study_dir, out_dir):
        return IngestResult(
            paths={
                role: (t1ce_native_path if role == "t1ce" else Path(f"/fake/{role}.nii.gz"))
                for role in ROLES
            },
            assignments={
                series_uid: RoleAssignment(
                    role="t1ce", score=5.0, reasons=(), outcome=SeriesOutcome.ASSIGNED
                )
            },
            missing_roles=(),
            rejected=(),
            warnings=(),
        )

    monkeypatch.setattr(clinical_jobs, "ingest_study", _fake_ingest_study_with_assignment)

    atlas_t1ce_path = tmp_path / "t1ce_atlas.nii.gz"
    _write_nifti(atlas_t1ce_path, image_shape)
    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1ce").mkdir(parents=True, exist_ok=True)
    plan = SimpleNamespace(center_role="t1ce")

    def _fake_preprocess_clinical_study_full(cfg, inputs, out_dir=None):
        outputs = {
            role: (atlas_t1ce_path if role == "t1ce" else Path(f"/fake/prep_{role}.nii.gz"))
            for role in ROLES
        }
        return PreprocessResult(
            plan=plan,
            outputs=outputs,
            brain_mask=Path("/fake/no_such_brain_mask.nii.gz"),
            stages_run=(),
            warnings=(),
            transformations_dir=transformations_dir,
        )

    monkeypatch.setattr(
        clinical_jobs, "preprocess_clinical_study", _fake_preprocess_clinical_study_full
    )

    def _fake_resample_mask_to_source(
        mask, atlas_affine, transformations_dir, target_role, target_native_path, out_dir
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "resampled.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(image_shape, dtype=np.uint8), np.eye(4)), str(out_path))
        return out_path

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _fake_resample_mask_to_source
    )
    monkeypatch.setattr(
        clinical_jobs, "_collect_source_datasets", lambda raw_dicom_dir, series_uid: ["dataset_a"]
    )

    return series_uid


def test_run_clinical_job_dicom_seg_refusal_still_reaches_done_with_no_cached_dcm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain(monkeypatch, tmp_path)

    def _refusing_write_dicom_seg(cfg, regions, source_datasets, out_path):
        raise ValueError("refused: geometry mismatch")

    monkeypatch.setattr(
        "neurovision.reporting.dicom_seg.write_dicom_seg", _refusing_write_dicom_seg
    )

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    dicom_seg_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"
    )
    assert not dicom_seg_path.exists()


def test_run_clinical_job_dicom_seg_unexpected_exception_isolated_still_reaches_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain(monkeypatch, tmp_path)

    def _raising_resample(*args, **kwargs):
        raise RuntimeError("sentinel: unexpected resample failure")

    monkeypatch.setattr(
        "neurovision.data.clinical_resample.resample_mask_to_source", _raising_resample
    )

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    dicom_seg_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"
    )
    assert not dicom_seg_path.exists()


def test_run_clinical_job_dicom_seg_success_caches_dcm_at_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain(monkeypatch, tmp_path)

    def _fake_write_dicom_seg(cfg, regions, source_datasets, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"fake dicom bytes")
        return Path(out_path)

    monkeypatch.setattr("neurovision.reporting.dicom_seg.write_dicom_seg", _fake_write_dicom_seg)

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    dicom_seg_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"
    )
    assert dicom_seg_path.is_file()
    assert dicom_seg_path.read_bytes() == b"fake dicom bytes"


# --- _validate_dicom_seg_cfg: static config bugs vs. per-case refusals ------


def _fake_dicom_seg_cfg(
    segmentation_type: str = "BINARY", series_description: str = "ok"
) -> SimpleNamespace:
    """A minimal cfg-like object exposing only what `_validate_dicom_seg_cfg` reads."""
    return SimpleNamespace(
        clinical=SimpleNamespace(
            dicom_seg=SimpleNamespace(
                segmentation_type=segmentation_type, series_description=series_description
            )
        )
    )


def test_validate_dicom_seg_cfg_accepts_binary_and_short_description() -> None:
    clinical_jobs._validate_dicom_seg_cfg(_fake_dicom_seg_cfg())  # must not raise


def test_validate_dicom_seg_cfg_accepts_exactly_64_character_description() -> None:
    cfg = _fake_dicom_seg_cfg(series_description="x" * 64)
    clinical_jobs._validate_dicom_seg_cfg(cfg)  # boundary: exactly 64 is allowed


def test_validate_dicom_seg_cfg_rejects_non_binary_segmentation_type() -> None:
    cfg = _fake_dicom_seg_cfg(segmentation_type="FRACTIONAL")
    with pytest.raises(ValueError, match="BINARY"):
        clinical_jobs._validate_dicom_seg_cfg(cfg)


def test_validate_dicom_seg_cfg_rejects_over_length_series_description() -> None:
    cfg = _fake_dicom_seg_cfg(series_description="x" * 65)
    with pytest.raises(ValueError, match="64"):
        clinical_jobs._validate_dicom_seg_cfg(cfg)


# --- run_clinical_job: a dicom_seg config bug surfaces distinguishably -------


def _wire_dicom_seg_chain_with_cfg_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **dicom_seg_overrides: object
) -> None:
    """Extends `_wire_dicom_seg_chain` with a STATIC `dicom_seg` config override.

    Composes the real config exactly the way `_compose_clinical_cfg` does,
    mutates `cfg.clinical.dicom_seg` with `dicom_seg_overrides`, then
    monkeypatches `_compose_clinical_cfg` to return that mutated copy --
    so `run_clinical_job` runs against a config that is broken in precisely
    the way `_validate_dicom_seg_cfg` exists to catch, everything else about
    the pipeline unchanged.
    """
    _wire_dicom_seg_chain(monkeypatch, tmp_path)
    real_cfg = clinical_jobs._compose_clinical_cfg()
    for key, value in dicom_seg_overrides.items():
        setattr(real_cfg.clinical.dicom_seg, key, value)
    monkeypatch.setattr(clinical_jobs, "_compose_clinical_cfg", lambda: real_cfg)


def test_run_clinical_job_bad_segmentation_type_logged_at_error_not_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain_with_cfg_override(monkeypatch, tmp_path, segmentation_type="FRACTIONAL")

    with caplog.at_level(logging.WARNING, logger=clinical_jobs.logger.name):
        result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    dicom_seg_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"
    )
    assert not dicom_seg_path.exists()

    # Distinguishable path #1: an ERROR record, with a traceback, from the
    # generic "unexpected chain failure" handler -- not a WARNING.
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("DICOM-SEG export failed unexpectedly" in r.message for r in error_records)
    assert all(r.exc_info is not None for r in error_records)

    # Distinguishable path #2: the routine per-case geometry-refusal WARNING
    # `_export_dicom_seg` logs for write_dicom_seg's own refusal was NEVER
    # logged for this job -- a config bug never reaches that code at all.
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("write_dicom_seg refused" in r.message for r in warning_records)


def test_run_clinical_job_over_length_series_description_logged_at_error_not_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain_with_cfg_override(monkeypatch, tmp_path, series_description="x" * 65)

    with caplog.at_level(logging.WARNING, logger=clinical_jobs.logger.name):
        result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    dicom_seg_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "dicom_seg" / f"{job.case_id}.dcm"
    )
    assert not dicom_seg_path.exists()

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("DICOM-SEG export failed unexpectedly" in r.message for r in error_records)
    assert all(r.exc_info is not None for r in error_records)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("write_dicom_seg refused" in r.message for r in warning_records)


# --- Structured report (Phase 4) wiring: _generate_report -------------------


class _FakeAtlas:
    """A minimal `Atlas`-shaped stand-in: only the attributes `_generate_report`
    and the REAL (unmocked) `atlas_for_case`/`region_mask` actually touch."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self.name = "FakeAtlas"
        self.version = "9.9"
        self.source = "unit-test atlas"
        self.parcellation = np.zeros(shape, dtype=np.int16)
        self.tissue = None


class _FakeKnowledge:
    """A minimal `KnowledgeBase`-shaped stand-in."""

    def __init__(self) -> None:
        self.eloquence = {"Structure_A": "eloquent", "Structure_B": "unclassified"}
        self.lobe = {"Structure_A": "Frontal"}
        self.evidence = "fake evidence sentence"
        self.citation = "fake citation"
        self.classification_name = "Fake Classification"
        self.coverage_gaps = ("gap1",)
        self.near_eloquent_mm = 10.0

    def coverage_line(self, n_structures: int) -> str:
        return f"fake coverage line for {n_structures} structures"


def _build_generate_report_fixture(
    tmp_path: Path, shape: tuple[int, int, int] = (4, 5, 6)
) -> dict[str, object]:
    """Builds everything `_generate_report` needs directly: a job, its own
    segmentation `Settings`, a real `meta.json`, and a real cached prediction --
    the same minimal shape `_build_export_dicom_seg_fixture` builds for
    `_export_dicom_seg`, with NO cropping (bbox == full shape).
    """
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    job_dir = clinical_jobs.jobs.job_root(settings) / job.job_id

    job_prep_dir = job_dir / "prep"
    job_cache_dir = job_dir / "cache"
    clinical_settings = clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)

    case_dir = job_prep_dir / job.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "cropped_shape": list(shape),
        "original_shape": list(shape),
        "bbox": [[0, shape[0]], [0, shape[1]], [0, shape[2]]],
        "spacing": [1.0, 1.0, 1.0],
        "affine": np.eye(4).tolist(),
    }
    (case_dir / "meta.json").write_text(json.dumps(meta))

    classes = np.zeros(shape, dtype=np.uint8)
    classes[1, 1, 1] = 3  # one ET voxel -> also WT (class 3 is in {1,2,3})
    pred_path = inference.cached_prediction_path(clinical_settings, job.case_id)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, classes)

    return {
        "job": job,
        "job_dir": job_dir,
        "clinical_settings": clinical_settings,
        "classes": classes,
        "meta": meta,
        "shape": shape,
    }


def _patch_anatomy_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    atlas: object,
    knowledge: object,
    classification_version: int = 3,
    groups: object = "fake-groups",
    involvement_notes: tuple[str, ...] = ("caveat one", "caveat two"),
) -> None:
    """Monkeypatches every atlas/knowledge-base LOADING call `_generate_report` makes.

    None of these touch real atlas/knowledge files on disk -- keeping the
    test hermetic and fast regardless of whether `data/atlas/sri24` (fetched
    separately, gitignored) happens to be present on the machine running it.
    """
    monkeypatch.setattr("neurovision.anatomy.atlas.load_atlas", lambda cfg: atlas)
    monkeypatch.setattr(
        "neurovision.anatomy.localize.load_knowledge",
        lambda eloquence_path, lobe_path, atlas_: knowledge,
    )
    monkeypatch.setattr(
        "neurovision.anatomy.localize.load_classification",
        lambda eloquence_path: SimpleNamespace(version=classification_version),
    )
    monkeypatch.setattr(
        "neurovision.anatomy.involvement.load_involvement_groups",
        lambda path, atlas_: groups,
    )
    monkeypatch.setattr(
        "neurovision.anatomy.involvement.load_involvement_notes",
        lambda path: involvement_notes,
    )


def test_generate_report_calls_underlying_functions_with_expected_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves `_generate_report` wires `localize_case`/`summarize_case`/`burden_profile`/
    `involvement_profile`/`build_report` together correctly -- not that those functions
    are themselves correct (covered by their own modules' tests)."""
    fixture = _build_generate_report_fixture(tmp_path)
    job = fixture["job"]
    classes = fixture["classes"]
    meta = fixture["meta"]

    cfg = clinical_jobs._compose_clinical_cfg()
    fake_atlas = _FakeAtlas(fixture["shape"])
    fake_knowledge = _FakeKnowledge()
    _patch_anatomy_loaders(monkeypatch, atlas=fake_atlas, knowledge=fake_knowledge)

    localize_calls: list[dict] = []
    # frac_of_structure/frac_of_tumour both above the configured min_frac, so
    # _generate_report's own min_frac filter (Bug 2 fix) is a content-
    # preserving no-op here -- that filter gets its own dedicated test below.
    fake_table = pd.DataFrame(
        {
            "region": ["WT"],
            "structure": ["Structure_A"],
            "frac_of_structure": [1.0],
            "frac_of_tumour": [1.0],
        }
    )

    def _fake_localize_case(classes_, atlas_, meta_, *, cropped, regions, knowledge):
        localize_calls.append(
            {
                "classes": classes_,
                "atlas": atlas_,
                "meta": meta_,
                "cropped": cropped,
                "regions": regions,
                "knowledge": knowledge,
            }
        )
        return fake_table

    monkeypatch.setattr("neurovision.anatomy.localize.localize_case", _fake_localize_case)

    summarize_calls: list[dict] = []
    fake_summary = {"n_structures_involved": 1, "distance_to_eloquent_mm": float("nan")}

    def _fake_summarize_case(table, knowledge):
        summarize_calls.append({"table": table, "knowledge": knowledge})
        return fake_summary

    monkeypatch.setattr("neurovision.anatomy.localize.summarize_case", _fake_summarize_case)

    # Bug 1 fix wiring: _generate_report must call eloquent_union_mask /
    # distance_to_eloquent itself and overwrite summarize_case's own
    # (geometry-blind) distance -- not trust summarize_case's NaN/0.0 alone.
    eloquent_union_mask_calls: list[dict] = []
    fake_eloquent_mask = np.zeros(fixture["shape"], dtype=bool)

    def _fake_eloquent_union_mask(atlas_, knowledge_):
        eloquent_union_mask_calls.append({"atlas": atlas_, "knowledge": knowledge_})
        return fake_eloquent_mask

    monkeypatch.setattr(
        "neurovision.anatomy.localize.eloquent_union_mask", _fake_eloquent_union_mask
    )

    distance_to_eloquent_calls: list[dict] = []

    def _fake_distance_to_eloquent(mask, eloquent_mask, *, spacing):
        distance_to_eloquent_calls.append(
            {"mask": mask, "eloquent_mask": eloquent_mask, "spacing": spacing}
        )
        return 7.5

    monkeypatch.setattr(
        "neurovision.anatomy.localize.distance_to_eloquent", _fake_distance_to_eloquent
    )

    burden_calls: list[dict] = []
    fake_burden = {"vol_WT_mm3": 42.0}

    def _fake_burden_profile(classes_, geom, *, min_volume_mm3, connectivity):
        burden_calls.append(
            {
                "classes": classes_,
                "geom": geom,
                "min_volume_mm3": min_volume_mm3,
                "connectivity": connectivity,
            }
        )
        return fake_burden

    monkeypatch.setattr("neurovision.anatomy.burden.burden_profile", _fake_burden_profile)

    involvement_calls: list[dict] = []
    fake_involvement = {"ventricle_contact": False}

    def _fake_involvement_profile(
        mask, parcellation, tissue, atlas_, groups_, geom, *, min_overlap_mm3, lobe
    ):
        involvement_calls.append(
            {
                "mask": mask,
                "parcellation": parcellation,
                "tissue": tissue,
                "atlas": atlas_,
                "groups": groups_,
                "geom": geom,
                "min_overlap_mm3": min_overlap_mm3,
                "lobe": lobe,
            }
        )
        return fake_involvement

    monkeypatch.setattr(
        "neurovision.anatomy.involvement.involvement_profile", _fake_involvement_profile
    )

    build_report_calls: list[dict] = []
    fake_report = {"case_id": job.case_id, "report_version": 1, "fake": True}

    def _fake_build_report(case_id, burden, anatomy_table, anatomy_summary, provenance, **kwargs):
        build_report_calls.append(
            {
                "case_id": case_id,
                "burden": burden,
                "anatomy_table": anatomy_table,
                "anatomy_summary": anatomy_summary,
                "provenance": provenance,
                **kwargs,
            }
        )
        return fake_report

    monkeypatch.setattr("neurovision.reporting.report.build_report", _fake_build_report)

    result = clinical_jobs._generate_report(
        job, fixture["clinical_settings"], fixture["job_dir"], cfg
    )

    expected_path = fixture["job_dir"] / "report" / f"{job.case_id}.json"
    assert result == expected_path
    assert expected_path.is_file()
    assert json.loads(expected_path.read_text()) == fake_report

    # localize_case got THIS job's own cached prediction and meta.json, the
    # loaded atlas/knowledge, cropped=True (a clinical job's cached
    # prediction is always in the cropped research frame), and the
    # configured region list.
    assert len(localize_calls) == 1
    call = localize_calls[0]
    np.testing.assert_array_equal(call["classes"], classes)
    assert call["atlas"] is fake_atlas
    assert call["meta"] == meta
    assert call["cropped"] is True
    assert call["regions"] == [str(r) for r in cfg.analysis.localize.regions]
    assert call["knowledge"] is fake_knowledge

    # summarize_case got localize_case's OWN returned table, run through
    # _generate_report's own min_frac filter (Bug 2 fix) -- every row here is
    # above threshold, so the filter is a content-preserving no-op, but it
    # always returns a NEW DataFrame (never the same object, even when it
    # drops nothing), so this checks content, not identity.
    assert len(summarize_calls) == 1
    pd.testing.assert_frame_equal(
        summarize_calls[0]["table"].reset_index(drop=True), fake_table.reset_index(drop=True)
    )
    assert summarize_calls[0]["knowledge"] is fake_knowledge

    # Bug 1 fix: eloquent_union_mask got the loaded atlas/knowledge;
    # distance_to_eloquent got THIS job's WT mask, that eloquent mask, and
    # this case's spacing -- and its result overwrote summarize_case's own
    # (here NaN) distance, with near_eloquent recomputed from it.
    assert len(eloquent_union_mask_calls) == 1
    assert eloquent_union_mask_calls[0]["atlas"] is fake_atlas
    assert eloquent_union_mask_calls[0]["knowledge"] is fake_knowledge
    assert len(distance_to_eloquent_calls) == 1
    np.testing.assert_array_equal(distance_to_eloquent_calls[0]["mask"], classes > 0)
    np.testing.assert_array_equal(
        distance_to_eloquent_calls[0]["eloquent_mask"], fake_eloquent_mask
    )
    assert distance_to_eloquent_calls[0]["spacing"] == (1.0, 1.0, 1.0)
    assert fake_summary["distance_to_eloquent_mm"] == 7.5
    assert fake_summary["near_eloquent"] is True  # 7.5 <= fake_knowledge.near_eloquent_mm (10.0)

    # burden_profile got this job's classes and the configured thresholds.
    assert len(burden_calls) == 1
    np.testing.assert_array_equal(burden_calls[0]["classes"], classes)
    assert burden_calls[0]["min_volume_mm3"] == float(cfg.analysis.burden.min_volume_mm3)
    assert burden_calls[0]["connectivity"] == int(cfg.analysis.burden.connectivity)

    # involvement_profile got the WT mask derived from THIS job's classes,
    # the atlas parcellation/tissue in the SAME (cropped) frame, the loaded
    # groups, and the knowledge base's own lobe map.
    assert len(involvement_calls) == 1
    inv_call = involvement_calls[0]
    np.testing.assert_array_equal(inv_call["mask"], classes > 0)  # WT: only class 3 present here
    np.testing.assert_array_equal(inv_call["parcellation"], fake_atlas.parcellation)
    assert inv_call["tissue"] is None
    assert inv_call["atlas"] is fake_atlas
    assert inv_call["groups"] == "fake-groups"
    assert inv_call["lobe"] is fake_knowledge.lobe
    assert inv_call["min_overlap_mm3"] == float(cfg.analysis.localize.involvement.min_overlap_mm3)

    # build_report got localize_case/summarize_case/burden_profile/
    # involvement_profile's own outputs, unchanged, plus the knowledge base's
    # eloquence/citation/coverage fields and this job's own provenance.
    assert len(build_report_calls) == 1
    br = build_report_calls[0]
    assert br["case_id"] == job.case_id
    assert br["burden"] is fake_burden
    pd.testing.assert_frame_equal(
        br["anatomy_table"].reset_index(drop=True), fake_table.reset_index(drop=True)
    )
    assert br["anatomy_summary"] is fake_summary
    assert br["evidence"] == fake_knowledge.evidence
    assert br["citation"] == fake_knowledge.citation
    assert br["classification_name"] == fake_knowledge.classification_name
    assert br["coverage_line"] == fake_knowledge.coverage_line(len(fake_knowledge.eloquence))
    assert br["coverage_gaps"] == fake_knowledge.coverage_gaps
    assert br["near_eloquent_mm"] == fake_knowledge.near_eloquent_mm
    assert br["top_n"] == int(cfg.analysis.report.top_n)
    assert br["involvement"] is fake_involvement
    assert br["involvement_caveats"] == ("caveat one", "caveat two")

    from neurovision.utils.io import read_yaml

    expected_aal_version = int(read_yaml(cfg.analysis.localize.lobe_map)["version"])
    provenance = br["provenance"]
    assert provenance.atlas_name == "FakeAtlas"
    assert provenance.atlas_version == "9.9"
    assert provenance.atlas_source == "unit-test atlas"
    assert provenance.atlas_licence == str(cfg.anatomy.licence)
    assert provenance.knowledge_versions == {"eloquence_map": 3, "aal_lobes": expected_aal_version}
    assert provenance.segmentation_source == "prediction"
    assert job.job_id in provenance.segmentation_dir
    assert provenance.code_revision is None


def test_generate_report_involvement_disabled_skips_involvement_and_passes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_generate_report_fixture(tmp_path)
    job = fixture["job"]

    cfg = clinical_jobs._compose_clinical_cfg()
    cfg.analysis.localize.involvement.enabled = False  # mutate the composed DictConfig directly

    fake_atlas = _FakeAtlas(fixture["shape"])
    fake_knowledge = _FakeKnowledge()
    _patch_anatomy_loaders(monkeypatch, atlas=fake_atlas, knowledge=fake_knowledge)

    involvement_groups_calls: list[object] = []
    monkeypatch.setattr(
        "neurovision.anatomy.involvement.load_involvement_groups",
        lambda *a: involvement_groups_calls.append(a),
    )
    involvement_profile_calls: list[object] = []
    monkeypatch.setattr(
        "neurovision.anatomy.involvement.involvement_profile",
        lambda *a, **kw: involvement_profile_calls.append((a, kw)),
    )

    monkeypatch.setattr(
        "neurovision.anatomy.localize.localize_case",
        lambda *a, **kw: pd.DataFrame(
            {
                "region": ["WT"],
                "structure": ["Structure_A"],
                "frac_of_structure": [1.0],
                "frac_of_tumour": [1.0],
            }
        ),
    )
    monkeypatch.setattr("neurovision.anatomy.localize.summarize_case", lambda *a, **kw: {})
    monkeypatch.setattr("neurovision.anatomy.burden.burden_profile", lambda *a, **kw: {})
    # _FakeAtlas has no `structure_mask` (it is not a real Atlas) -- Bug 1's
    # fix calls eloquent_union_mask unconditionally, so it must be mocked
    # here too, same as every other _generate_report test using _FakeAtlas.
    monkeypatch.setattr(
        "neurovision.anatomy.localize.eloquent_union_mask",
        lambda atlas_, knowledge_: np.zeros(fixture["shape"], dtype=bool),
    )
    monkeypatch.setattr(
        "neurovision.anatomy.localize.distance_to_eloquent",
        lambda mask, eloquent_mask, *, spacing: float("nan"),
    )

    build_report_calls: list[dict] = []

    def _fake_build_report(case_id, burden, anatomy_table, anatomy_summary, provenance, **kwargs):
        build_report_calls.append(kwargs)
        return {"case_id": case_id}

    monkeypatch.setattr("neurovision.reporting.report.build_report", _fake_build_report)

    clinical_jobs._generate_report(job, fixture["clinical_settings"], fixture["job_dir"], cfg)

    # Involvement is disabled: neither loading it nor computing it ever ran.
    assert involvement_groups_calls == []
    assert involvement_profile_calls == []
    assert build_report_calls[0]["involvement"] is None
    assert build_report_calls[0]["involvement_caveats"] == ()


def _real_atlas_with_one_structure_at(
    shape: tuple[int, int, int],
    structure_voxel: tuple[int, int, int],
    structure_name: str = "Structure_A",
) -> Atlas:
    """A REAL `Atlas` (not `_FakeAtlas`) with exactly one structure, at one voxel.

    Needed for `test_generate_report_near_eloquent_gets_real_distance_not_null`,
    which exercises the REAL (unmocked) `eloquent_union_mask` /
    `distance_to_eloquent` -- unlike every other `_generate_report` test in this
    file, which mocks them because `_FakeAtlas` has no `structure_mask` method.
    `Atlas.structure_mask` (`np.isin(self.parcellation, structure.label_ids)`)
    is real, geometric code; giving it a real, tiny parcellation is simpler and
    more honest than re-mocking that method too.
    """
    parcellation = np.zeros(shape, dtype=np.int16)
    parcellation[structure_voxel] = 7
    labels = AtlasLabels(
        structures=(AtlasStructure(name=structure_name, label_ids=(7,), laterality="midline"),),
        unmapped_name="unmapped",
    )
    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=None,
        tissue_codes={},
        name="RealAtlasForDistanceTest",
        version="1.0",
        source="unit-test",
        unmapped_ids=(),
    )


def test_generate_report_near_eloquent_gets_real_distance_not_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 1, the exact false negative it produced: a mask that does NOT overlap
    an eloquent structure, but sits well within the configured near-eloquent
    threshold, must get the real measured mm distance and near_eloquent=True --
    not summarize_case's own geometry-blind NaN/False.
    """
    shape = (10, 10, 10)
    fixture = _build_generate_report_fixture(tmp_path, shape=shape)
    job = fixture["job"]

    # One WT voxel at (2, 2, 2). The eloquent structure (below) sits at
    # (2, 2, 5) -- 3 voxels away along one axis only, so at this fixture's
    # isotropic 1mm spacing the Euclidean distance is EXACTLY 3.0mm: inside
    # _FakeKnowledge's near_eloquent_mm=10.0 threshold, but not overlapping.
    classes = np.zeros(shape, dtype=np.uint8)
    classes[2, 2, 2] = 2  # oedema -- WT only, no ET/TC
    pred_path = inference.cached_prediction_path(fixture["clinical_settings"], job.case_id)
    np.save(pred_path, classes)

    cfg = clinical_jobs._compose_clinical_cfg()
    cfg.analysis.localize.involvement.enabled = False  # not under test here

    real_atlas = _real_atlas_with_one_structure_at(shape, structure_voxel=(2, 2, 5))
    fake_knowledge = _FakeKnowledge()  # near_eloquent_mm=10.0, "Structure_A" eloquent
    _patch_anatomy_loaders(monkeypatch, atlas=real_atlas, knowledge=fake_knowledge)

    monkeypatch.setattr(
        "neurovision.anatomy.localize.localize_case",
        lambda *a, **kw: pd.DataFrame(
            {
                "region": ["WT"],
                "structure": ["Structure_A"],
                "frac_of_structure": [1.0],
                "frac_of_tumour": [1.0],
            }
        ),
    )
    monkeypatch.setattr(
        "neurovision.anatomy.localize.summarize_case",
        # The exact shape of the OLD bug: summarize_case alone can only ever
        # report 0.0 (overlap) or NaN (no overlap) here -- NaN, since this
        # case does not overlap.
        lambda table, knowledge: {"distance_to_eloquent_mm": float("nan")},
    )
    monkeypatch.setattr("neurovision.anatomy.burden.burden_profile", lambda *a, **kw: {})

    build_report_calls: list[dict] = []

    def _fake_build_report(case_id, burden, anatomy_table, anatomy_summary, provenance, **kwargs):
        build_report_calls.append({"anatomy_summary": anatomy_summary})
        return {"case_id": case_id}

    monkeypatch.setattr("neurovision.reporting.report.build_report", _fake_build_report)

    clinical_jobs._generate_report(job, fixture["clinical_settings"], fixture["job_dir"], cfg)

    summary = build_report_calls[0]["anatomy_summary"]
    assert summary["distance_to_eloquent_mm"] == pytest.approx(3.0)
    assert summary["near_eloquent"] is True


def test_generate_report_near_eloquent_distance_survives_non_trivial_bbox_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression trap for the crop arithmetic itself, not just the fix's happy path.

    Every other `_generate_report` test uses an IDENTITY bbox
    (`[[0, shape[0]], [0, shape[1]], [0, shape[2]]]`), which slices the
    eloquent mask unchanged regardless of axis order or off-by-one errors --
    none of them would catch a regression in
    `eloquent_mask[tuple(slice(start, end) for start, end in bbox)]` itself.

    This test uses a DIFFERENT, asymmetric offset per axis (2, 1, 0) cropping
    a (14, 14, 14) full atlas down to a (6, 8, 10) cropped frame, with the
    eloquent structure placed well inside the crop (not at a boundary, so an
    off-by-one at the edges would not accidentally cancel out). The WT voxel
    and the structure are 3 voxels apart on axis 0 only in CROPPED
    coordinates, so the correct answer is exactly 3.0mm at 1mm isotropic
    spacing -- but only if the crop offset (2, 1, 0) is subtracted correctly,
    per axis, from the structure's FULL-frame position (7, 6, 5) to land it
    at cropped position (5, 5, 5).

    Two plausible bugs this specifically catches, by hand:
      - Swapped axis-0/axis-1 offsets (using 1 for axis 0, 2 for axis 1):
        the structure would land at cropped (6, 4, 5) instead of (5, 5, 5),
        giving distance sqrt(4^2 + 1^2) = sqrt(17) ~= 4.123mm, not 3.0mm.
      - An off-by-one on the axis-0 start (1 instead of 2, or 3 instead of
        2): the structure lands at cropped (6, 5, 5) or (4, 5, 5), giving
        distance 4.0mm or 2.0mm, not 3.0mm.
    Either bug produces a WRONG but well-formed number, which is exactly the
    failure mode `pytest.approx(3.0)` below is positioned to catch.
    """
    full_shape = (14, 14, 14)
    bbox = [[2, 8], [1, 9], [0, 10]]
    cropped_shape = (6, 8, 10)

    fixture = _build_generate_report_fixture(tmp_path, shape=cropped_shape)
    job = fixture["job"]

    # Overwrite the fixture's default identity-bbox meta.json with the real,
    # asymmetric crop under test.
    meta = {
        "cropped_shape": list(cropped_shape),
        "original_shape": list(full_shape),
        "bbox": bbox,
        "spacing": [1.0, 1.0, 1.0],
        "affine": np.eye(4).tolist(),
    }
    case_dir = fixture["clinical_settings"].prep_dir / job.case_id
    (case_dir / "meta.json").write_text(json.dumps(meta))

    # WT voxel at cropped-frame (2, 5, 5).
    classes = np.zeros(cropped_shape, dtype=np.uint8)
    classes[2, 5, 5] = 2  # oedema -- WT only
    pred_path = inference.cached_prediction_path(fixture["clinical_settings"], job.case_id)
    np.save(pred_path, classes)

    cfg = clinical_jobs._compose_clinical_cfg()
    cfg.analysis.localize.involvement.enabled = False  # not under test here

    # Structure at FULL-frame (7, 6, 5) -> cropped (7-2, 6-1, 5-0) = (5, 5, 5)
    # when the bbox offset is subtracted correctly, per axis.
    real_atlas = _real_atlas_with_one_structure_at(full_shape, structure_voxel=(7, 6, 5))
    fake_knowledge = _FakeKnowledge()  # near_eloquent_mm=10.0, "Structure_A" eloquent
    _patch_anatomy_loaders(monkeypatch, atlas=real_atlas, knowledge=fake_knowledge)

    monkeypatch.setattr(
        "neurovision.anatomy.localize.localize_case",
        lambda *a, **kw: pd.DataFrame(
            {
                "region": ["WT"],
                "structure": ["Structure_A"],
                "frac_of_structure": [1.0],
                "frac_of_tumour": [1.0],
            }
        ),
    )
    monkeypatch.setattr(
        "neurovision.anatomy.localize.summarize_case",
        lambda table, knowledge: {"distance_to_eloquent_mm": float("nan")},
    )
    monkeypatch.setattr("neurovision.anatomy.burden.burden_profile", lambda *a, **kw: {})

    build_report_calls: list[dict] = []

    def _fake_build_report(case_id, burden, anatomy_table, anatomy_summary, provenance, **kwargs):
        build_report_calls.append({"anatomy_summary": anatomy_summary})
        return {"case_id": case_id}

    monkeypatch.setattr("neurovision.reporting.report.build_report", _fake_build_report)

    clinical_jobs._generate_report(job, fixture["clinical_settings"], fixture["job_dir"], cfg)

    summary = build_report_calls[0]["anatomy_summary"]
    assert summary["distance_to_eloquent_mm"] == pytest.approx(3.0)
    assert summary["near_eloquent"] is True


def test_generate_report_min_frac_drops_low_overlap_row_from_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 2: `_generate_report` used to pass `localize_case`'s raw table straight
    through with no `min_frac` filtering at all. A row below the configured
    threshold on BOTH `frac_of_structure` and `frac_of_tumour` must not survive
    into the final report's `eloquence.involved` list or `n_structures_involved`
    -- exactly what `scripts/localize.py::localize_one`'s own filter prevents.
    Runs the REAL `summarize_case`/`build_report` (unmocked) so the assertions
    are about the actual report, not about what got passed to a mock.
    """
    fixture = _build_generate_report_fixture(tmp_path)
    job = fixture["job"]

    cfg = clinical_jobs._compose_clinical_cfg()
    cfg.analysis.localize.involvement.enabled = False  # not under test here
    min_frac = float(cfg.analysis.localize.min_frac)
    assert min_frac > 0.0  # sanity: the config actually sets a positive threshold

    fake_atlas = _FakeAtlas(fixture["shape"])
    fake_knowledge = _FakeKnowledge()
    _patch_anatomy_loaders(monkeypatch, atlas=fake_atlas, knowledge=fake_knowledge)
    monkeypatch.setattr(
        "neurovision.anatomy.localize.eloquent_union_mask",
        lambda atlas_, knowledge_: np.zeros(fixture["shape"], dtype=bool),
    )
    monkeypatch.setattr(
        "neurovision.anatomy.localize.distance_to_eloquent",
        lambda mask, eloquent_mask, *, spacing: float("nan"),
    )
    monkeypatch.setattr("neurovision.anatomy.burden.burden_profile", lambda *a, **kw: {})

    # "Structure_A" is well above min_frac on both fractions (survives).
    # "Structure_B" is below min_frac on BOTH fractions (must be dropped) --
    # both rows are marked "eloquent", so an unfiltered table would inflate
    # n_structures_involved to 2 and list both in eloquence.involved.
    below_threshold = min_frac / 10.0
    raw_table = pd.DataFrame(
        {
            "region": ["WT", "WT"],
            "structure": ["Structure_A", "Structure_B"],
            "laterality": ["L", "R"],
            "lobe": ["Frontal", "Parietal"],
            "eloquence": ["eloquent", "eloquent"],
            "matched_term": ["", ""],
            "n_voxels": [100, 1],
            "volume_mm3": [100.0, 1.0],
            "frac_of_tumour": [0.9, below_threshold],
            "frac_of_structure": [0.5, below_threshold],
        }
    )
    monkeypatch.setattr("neurovision.anatomy.localize.localize_case", lambda *a, **kw: raw_table)

    result_path = clinical_jobs._generate_report(
        job, fixture["clinical_settings"], fixture["job_dir"], cfg
    )
    report = json.loads(result_path.read_text())

    involved_names = {row["structure"] for row in report["eloquence"]["involved"]}
    assert involved_names == {"Structure_A"}
    assert "Structure_B" not in involved_names
    assert report["anatomy"]["n_structures_involved"] == 1


def test_generate_report_missing_meta_json_raises(tmp_path: Path) -> None:
    """No `_generate_report`-internal try/except absorbs this -- see its own
    docstring: `run_clinical_job`'s wrapping try/except is the only layer."""
    fixture = _build_generate_report_fixture(tmp_path)
    (fixture["job_dir"] / "prep" / fixture["job"].case_id / "meta.json").unlink()
    cfg = clinical_jobs._compose_clinical_cfg()

    with pytest.raises(FileNotFoundError):
        clinical_jobs._generate_report(
            fixture["job"], fixture["clinical_settings"], fixture["job_dir"], cfg
        )


# --- run_clinical_job: report-generation wiring and failure isolation -------


def test_run_clinical_job_calls_generate_report_and_reaches_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "neurovision.reporting.dicom_seg.write_dicom_seg",
        lambda cfg, regions, source_datasets, out_path: None,
    )

    calls: list[dict] = []

    def _fake_generate_report(job_, clinical_settings, job_dir, cfg):
        calls.append({"job": job_, "clinical_settings": clinical_settings, "job_dir": job_dir})
        out_dir = job_dir / "report"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job_.case_id}.json"
        out_path.write_text(json.dumps({"case_id": job_.case_id}))
        return out_path

    monkeypatch.setattr(clinical_jobs, "_generate_report", _fake_generate_report)

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert len(calls) == 1
    assert calls[0]["job"].job_id == job.job_id
    assert calls[0]["job_dir"] == clinical_jobs.jobs.job_root(settings) / job.job_id
    report_path = calls[0]["job_dir"] / "report" / f"{job.case_id}.json"
    assert report_path.is_file()


def test_run_clinical_job_report_generation_failure_isolated_still_reaches_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report-generation failure must never turn a good segmentation into `"failed"`."""
    settings = _settings(tmp_path)
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    _wire_dicom_seg_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "neurovision.reporting.dicom_seg.write_dicom_seg",
        lambda cfg, regions, source_datasets, out_path: None,
    )

    def _raising_generate_report(job_, clinical_settings, job_dir, cfg):
        raise RuntimeError("sentinel: report generation blew up")

    monkeypatch.setattr(clinical_jobs, "_generate_report", _raising_generate_report)

    result = clinical_jobs.run_clinical_job(settings, job.job_id)

    assert result.state == "done"
    assert result.state != "failed"
    report_path = (
        clinical_jobs.jobs.job_root(settings) / job.job_id / "report" / f"{job.case_id}.json"
    )
    assert not report_path.exists()


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
