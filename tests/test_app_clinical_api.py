"""Tests for the `/api/clinical/*` routes in `app.backend.api`.

Mirrors `tests/test_app_job_routes.py`'s structure exactly, applied to the
clinical (raw DICOM study) pipeline instead of the demo upload pipeline: a
synthetic zip archive stands in for a real DICOM study (nothing here calls
`neurovision.data.dicom_ingest.ingest_study` for real), and
`clinical_jobs.start_clinical_job` is monkeypatched to a no-op so
`POST /api/clinical/upload` never spawns a background thread that runs the
real (torch-requiring, ANTs/HD-BET-requiring) pipeline -- the routes under
test are the HTTP wiring onto `clinical_jobs.py`, which
`tests/test_app_clinical_jobs.py` already covers at the function level.

A `"done"` or `"refused"` job is fabricated directly, the same way
`test_app_job_routes.py._fabricate_done_job` does: create a real (queued) job
through `clinical_jobs.create_clinical_job`, then write the files a finished
pipeline would have written and mutate the `ClinicalJob` dataclass fields in
place (safe because `get_clinical_job` always returns the SAME in-memory
instance).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from app.backend import api, clinical_jobs, config, inference, jobs
from fastapi.testclient import TestClient


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    """Builds a small in-memory zip archive from filename -> content bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buf.getvalue()


def _valid_study_zip() -> bytes:
    """A tiny, arbitrarily nested placeholder "DICOM study" zip.

    Nothing in these tests calls `ingest_study` for real on this archive's
    contents (either not reached, or the job is monkeypatched away before it
    runs), so the "DICOM" files are just placeholder bytes -- only the
    archive's own structure is exercised by `create_clinical_job`.
    """
    return _zip_bytes(
        {
            "STUDY/SERIES1/IM001.dcm": b"placeholder dicom bytes 1",
            "STUDY/SERIES1/IM002.dcm": b"placeholder dicom bytes 2",
        }
    )


def _clear_caches() -> None:
    """Clears the process-global caches other test files also clear (see test_app_api.py)."""
    config.get_settings.cache_clear()


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points settings and the clinical job store at fresh directories under `tmp_path`."""
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    cache_dir = tmp_path / "cache"
    job_dir = tmp_path / "jobs"
    prep_dir.mkdir()
    eval_dir.mkdir()

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder -- clinical segmentation is never reached in these tests")

    monkeypatch.setenv("NVX_PREP_DIR", str(prep_dir))
    monkeypatch.setenv("NVX_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("NVX_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NVX_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NVX_JOB_DIR", str(job_dir))
    monkeypatch.setenv("NVX_MAX_CASES", "24")

    clinical_jobs._CLINICAL_JOBS.clear()
    _clear_caches()
    yield tmp_path
    clinical_jobs._CLINICAL_JOBS.clear()
    _clear_caches()


@pytest.fixture
def client(backend: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A `TestClient` with `clinical_jobs.start_clinical_job` monkeypatched to a no-op."""
    monkeypatch.setattr(api.clinical_jobs, "start_clinical_job", lambda settings, job_id: None)
    return TestClient(api.create_app())


def _upload(client: TestClient, payload: bytes):
    return client.post(
        "/api/clinical/upload",
        files={"dicom_zip": ("study.zip", payload, "application/zip")},
    )


def _fabricate_refused_clinical_job(settings: config.Settings) -> clinical_jobs.ClinicalJob:
    """Creates a clinical job and drives it directly to a `"refused"` terminal state.

    No real E1/E3/E5 gate runs here -- the fields below are exactly the shape
    those gates would have left behind (see `clinical_jobs.ClinicalJob`'s
    docstring), written by hand so the route under test is exercised without
    any of the heavy pipeline it normally follows.
    """
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())
    job.ingest_result = {
        "paths": {"t1": "/x/t1.nii.gz"},
        "assignments": {},
        "missing_roles": [],
        "rejected": [],
        "warnings": [],
    }
    job.input_qc_pre = {"verdict": "pass", "findings": []}
    job.input_qc_post = {"verdict": "pass", "findings": []}
    job.gatekeeper_decision = {"decision": "refuse", "signals": {}}
    job.state = "refused"
    job.stage = "gatekeeper"
    job.error = "Gatekeeper refused: predicted_dice: below threshold"
    return job


def _fabricate_done_clinical_job(settings: config.Settings) -> clinical_jobs.ClinicalJob:
    """Creates a clinical job and drives it directly to a `"done"` terminal state.

    Writes a preprocessed case (`image.npy` + `meta.json`) and a cached
    prediction directly to disk, under the SAME directories
    `clinical_jobs.clinical_segmentation_settings` resolves for this job --
    the same "fabricate the finished job's files by hand" approach
    `test_app_job_routes.py._fabricate_done_job` uses for the plain upload
    job.
    """
    job = clinical_jobs.create_clinical_job(settings, _valid_study_zip())

    job_prep_dir = jobs.job_root(settings) / job.job_id / "prep"
    job_cache_dir = jobs.job_root(settings) / job.job_id / "cache"
    clinical_settings = clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)

    case_dir = job_prep_dir / job.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    shape = (5, 6, 7)
    image = np.zeros((4, *shape), dtype=np.float16)
    image[:, 1:4, 1:4, 1:4] = 1.0
    np.save(case_dir / "image.npy", image)
    meta = {
        "cropped_shape": list(shape),
        "original_shape": list(shape),
        "bbox": [[0, shape[0]], [0, shape[1]], [0, shape[2]]],
        "spacing": [1.0, 1.0, 1.0],
    }
    (case_dir / "meta.json").write_text(json.dumps(meta))

    pred_path = inference.cached_prediction_path(clinical_settings, job.case_id)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.zeros(shape, dtype=np.uint8)
    prediction[1:3, 1:3, 1:3] = 3
    np.save(pred_path, prediction)

    job.state = "done"
    job.stage = "done"
    job.progress = 1.0
    return job


# --- POST /api/clinical/upload, happy path ----------------------------------


def test_clinical_upload_valid_returns_202_with_job_id_and_state(client: TestClient) -> None:
    response = _upload(client, _valid_study_zip())
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["state"] == "queued"
    assert body["case_id"] == body["job_id"]


# --- POST /api/clinical/upload, rejections ----------------------------------


def test_clinical_upload_empty_payload_is_400(client: TestClient) -> None:
    response = _upload(client, b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_clinical_upload_non_zip_payload_is_400(client: TestClient) -> None:
    response = _upload(client, b"this is definitely not a zip archive")
    assert response.status_code == 400
    assert "not a valid zip" in response.json()["detail"]


# --- GET /api/clinical/jobs --------------------------------------------------


def test_list_clinical_jobs_includes_created_job(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    body = client.get("/api/clinical/jobs").json()
    ids = [j["job_id"] for j in body["jobs"]]
    assert created["job_id"] in ids


# --- GET /api/clinical/jobs/{job_id} -----------------------------------------


def test_get_unknown_clinical_job_is_404(client: TestClient) -> None:
    response = client.get("/api/clinical/jobs/does-not-exist")
    assert response.status_code == 404


def test_refused_clinical_job_detail_is_200_with_full_diagnostic_payload(
    client: TestClient, backend: Path
) -> None:
    """A refused job is a normal, successful response -- not a 4xx/5xx."""
    settings = config.get_settings()
    job = _fabricate_refused_clinical_job(settings)

    response = client.get(f"/api/clinical/jobs/{job.job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "refused"
    assert body["error"]
    assert body["ingest_result"] is not None
    assert body["input_qc_pre"] is not None
    assert body["input_qc_post"] is not None
    assert body["gatekeeper_decision"] is not None


# --- DELETE /api/clinical/jobs/{job_id} --------------------------------------


def test_delete_clinical_job_then_second_delete_is_404(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    job_id = created["job_id"]

    first = client.delete(f"/api/clinical/jobs/{job_id}")
    assert first.status_code == 200
    assert first.json()["deleted"] is True

    assert client.get(f"/api/clinical/jobs/{job_id}").status_code == 404

    second = client.delete(f"/api/clinical/jobs/{job_id}")
    assert second.status_code == 404


# --- volume / mask: not-done jobs are 409, unknown ids/modalities are 404 ----


def test_clinical_volume_on_queued_job_is_409(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    response = client.get(f"/api/clinical/jobs/{created['job_id']}/volume/t1")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_clinical_mask_on_queued_job_is_409(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    response = client.get(f"/api/clinical/jobs/{created['job_id']}/mask/prediction")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_clinical_volume_on_refused_job_is_409_worded_neutrally(
    client: TestClient, backend: Path
) -> None:
    """A refused job is a legitimate terminal state -- the 409 must not read as a bug report."""
    settings = config.get_settings()
    job = _fabricate_refused_clinical_job(settings)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/volume/t1")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "not done yet" in detail
    assert "state='refused'" in detail
    # Must not phrase the state itself as a problem -- only report it.
    assert "bug" not in detail.lower()
    assert "invalid" not in detail.lower()


def test_clinical_mask_on_refused_job_is_409_worded_neutrally(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_refused_clinical_job(settings)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/mask/prediction")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "not done yet" in detail
    assert "state='refused'" in detail


def test_clinical_volume_and_mask_on_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/clinical/jobs/no-such-job/volume/t1").status_code == 404
    assert client.get("/api/clinical/jobs/no-such-job/mask/prediction").status_code == 404


# --- volume / mask: a done job serves bytes matching the /jobs/... contract -


def test_done_clinical_job_serves_volume_and_mask_matching_job_route_contract(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)

    vol_response = client.get(f"/api/clinical/jobs/{job.job_id}/volume/t1")
    assert vol_response.status_code == 200
    assert vol_response.headers["content-type"] == "application/octet-stream"
    assert vol_response.headers["x-volume-shape"] == "5,6,7"
    assert vol_response.headers["x-volume-dtype"] == "uint8"
    assert len(vol_response.content) == 5 * 6 * 7

    mask_response = client.get(f"/api/clinical/jobs/{job.job_id}/mask/prediction")
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "application/octet-stream"
    assert mask_response.headers["x-volume-shape"] == "5,6,7"
    assert mask_response.headers["x-volume-dtype"] == "uint8"
    mask_arr = np.frombuffer(mask_response.content, dtype=np.uint8).reshape((5, 6, 7))
    assert mask_arr[1, 1, 1] == 3
    assert mask_arr[0, 0, 0] == 0


def test_clinical_volume_unknown_modality_on_done_job_is_404(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    response = client.get(f"/api/clinical/jobs/{job.job_id}/volume/bogus")
    assert response.status_code == 404


# --- GET /api/clinical/jobs/{job_id}/uncertainty -----------------------------


def _write_clinical_logits(
    settings: config.Settings, job: clinical_jobs.ClinicalJob, logits: np.ndarray
) -> Path:
    """Writes a `(3, D, H, W)` logits array to this job's namespaced logits cache.

    Mirrors `_fabricate_done_clinical_job`'s own resolution of the job's
    dedicated clinical `Settings` -- the logits must land where
    `inference.cached_logits_path` (via `clinical_segmentation_settings`)
    actually looks, not under the generic `settings`.
    """
    job_prep_dir = jobs.job_root(settings) / job.job_id / "prep"
    job_cache_dir = jobs.job_root(settings) / job.job_id / "cache"
    clinical_settings = clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)
    logits_path = inference.cached_logits_path(clinical_settings, job.case_id)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(logits_path, logits.astype(np.float32))
    return logits_path


def test_clinical_job_uncertainty_done_with_logits_is_200(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    shape = (5, 6, 7)
    _write_clinical_logits(settings, job, np.zeros((3, *shape), dtype=np.float32))

    response = client.get(f"/api/clinical/jobs/{job.job_id}/uncertainty")
    assert response.status_code == 200
    assert response.headers["x-uncertainty-kind"] == "predictive-entropy-single-pass"
    assert response.headers["x-volume-shape"] == "5,6,7"
    assert len(response.content) == 5 * 6 * 7


def test_clinical_job_uncertainty_done_without_logits_is_404(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    response = client.get(f"/api/clinical/jobs/{job.job_id}/uncertainty")
    assert response.status_code == 404


# --- GET /api/clinical/jobs/{job_id}/conformal-band/{region} ----------------


def test_clinical_job_conformal_band_valid_region_with_threshold_is_200(
    client: TestClient, backend: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    shape = (5, 6, 7)
    _write_clinical_logits(settings, job, np.zeros((3, *shape), dtype=np.float32))

    # Real cfg.clinical.gatekeeper.conformal_alpha is 0.10; f"{0.10}" == "0.1",
    # so the fit.json key convention here must match that exactly (see
    # `clinical_jobs._load_conformal_fitted_thresholds`'s docstring).
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({"WT__alpha_0.1": {"threshold": 0.3}}))
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: fit_path)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/conformal-band/WT")
    assert response.status_code == 200
    assert response.headers["x-uncertainty-kind"] == "conformal-band"
    assert response.headers["x-volume-shape"] == "5,6,7"
    assert len(response.content) == 5 * 6 * 7


def test_clinical_job_conformal_band_invalid_region_is_404(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    # "ET" is not in the real cfg.clinical.gatekeeper.regions ([WT, TC]).
    response = client.get(f"/api/clinical/jobs/{job.job_id}/conformal-band/ET")
    assert response.status_code == 404


def test_clinical_job_conformal_band_no_fitted_threshold_is_404(
    client: TestClient, backend: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    shape = (5, 6, 7)
    _write_clinical_logits(settings, job, np.zeros((3, *shape), dtype=np.float32))

    missing_fit_path = tmp_path / "does_not_exist" / "fit.json"
    monkeypatch.setattr(clinical_jobs, "_conformal_fit_path", lambda: missing_fit_path)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/conformal-band/WT")
    assert response.status_code == 404


# --- GET /api/clinical/jobs/{job_id}/gradcam/{region} -----------------------


def _write_clinical_gradcam(
    settings: config.Settings,
    job: clinical_jobs.ClinicalJob,
    region: str,
    shape: tuple[int, int, int],
) -> Path:
    """Writes a cached Grad-CAM heatmap directly to disk for a fabricated 'done' job.

    Mirrors `_write_clinical_logits`'s own resolution of the job's dedicated
    clinical `Settings` -- the heatmap must land where
    `inference.cached_gradcam_path` (via `clinical_segmentation_settings`)
    actually looks, not under the generic `settings`.
    """
    job_prep_dir = jobs.job_root(settings) / job.job_id / "prep"
    job_cache_dir = jobs.job_root(settings) / job.job_id / "cache"
    clinical_settings = clinical_jobs.clinical_segmentation_settings(job_prep_dir, job_cache_dir)
    gradcam_path = inference.cached_gradcam_path(clinical_settings, job.case_id, region)
    gradcam_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(gradcam_path, np.zeros(shape, dtype=np.uint8))
    return gradcam_path


def test_clinical_job_gradcam_valid_region_with_cached_cam_is_200(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    shape = (5, 6, 7)
    _write_clinical_gradcam(settings, job, "WT", shape)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/gradcam/WT")
    assert response.status_code == 200
    assert response.headers["x-uncertainty-kind"] == "gradcam"
    assert response.headers["x-volume-shape"] == "5,6,7"
    assert response.headers["x-volume-dtype"] == "uint8"
    assert len(response.content) == 5 * 6 * 7


def test_clinical_job_gradcam_invalid_region_is_404(client: TestClient, backend: Path) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    # "ET" is out of scope for live Grad-CAM (see inference.explain_case's docstring).
    response = client.get(f"/api/clinical/jobs/{job.job_id}/gradcam/ET")
    assert response.status_code == 404


def test_clinical_job_gradcam_no_cached_cam_is_404(client: TestClient, backend: Path) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    # Valid region, but nothing was ever written for it (e.g. this job predates
    # the feature, or that region's Grad-CAM failed during the job's run).
    response = client.get(f"/api/clinical/jobs/{job.job_id}/gradcam/WT")
    assert response.status_code == 404


# --- GET /api/clinical/jobs/{job_id}/dicom-seg -------------------------------


def _write_clinical_dicom_seg(
    settings: config.Settings, job: clinical_jobs.ClinicalJob, content: bytes = b"fake dicom bytes"
) -> Path:
    """Writes a placeholder `.dcm` file at the path `clinical_jobs._export_dicom_seg`
    would have cached it at: `<job_dir>/dicom_seg/<case_id>.dcm`, NOT namespaced by
    experiment/checkpoint the way logits/Grad-CAM are -- see that function's own
    docstring for why (one clinical job produces at most one SEG object).
    """
    job_dir = jobs.job_root(settings) / job.job_id
    dicom_seg_path = job_dir / "dicom_seg" / f"{job.case_id}.dcm"
    dicom_seg_path.parent.mkdir(parents=True, exist_ok=True)
    dicom_seg_path.write_bytes(content)
    return dicom_seg_path


def test_clinical_job_dicom_seg_cached_is_200_binary_download(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    _write_clinical_dicom_seg(settings, job, b"fake dicom bytes")

    response = client.get(f"/api/clinical/jobs/{job.job_id}/dicom-seg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/dicom"
    assert response.content == b"fake dicom bytes"


def test_clinical_job_dicom_seg_no_cached_object_is_404(client: TestClient, backend: Path) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    # A done job that never produced a SEG object (export failed/refused, or
    # this job predates the feature) -- nothing to serve for THIS job.
    response = client.get(f"/api/clinical/jobs/{job.job_id}/dicom-seg")
    assert response.status_code == 404


def test_clinical_job_dicom_seg_on_queued_job_is_409(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    response = client.get(f"/api/clinical/jobs/{created['job_id']}/dicom-seg")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_clinical_job_dicom_seg_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/api/clinical/jobs/no-such-job/dicom-seg")
    assert response.status_code == 404


def _write_clinical_report(
    settings: config.Settings,
    job: clinical_jobs.ClinicalJob,
    payload: dict | None = None,
) -> Path:
    """Writes a placeholder report JSON at the path `clinical_jobs._generate_report`
    would have cached it at: `<job_dir>/report/<case_id>.json` -- mirroring
    `_write_clinical_dicom_seg`'s own convention exactly.
    """
    job_dir = jobs.job_root(settings) / job.job_id
    report_path = job_dir / "report" / f"{job.case_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload if payload is not None else {"case_id": job.case_id}))
    return report_path


def test_clinical_job_report_cached_is_200_with_json_payload(
    client: TestClient, backend: Path
) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    payload = {
        "case_id": job.case_id,
        "report_version": 1,
        "disclaimer": "research and educational decision-support artifact",
    }
    _write_clinical_report(settings, job, payload)

    response = client.get(f"/api/clinical/jobs/{job.job_id}/report")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == payload


def test_clinical_job_report_no_cached_report_is_404(client: TestClient, backend: Path) -> None:
    settings = config.get_settings()
    job = _fabricate_done_clinical_job(settings)
    # A done job that never produced a report (generation failed for this
    # job, or it predates the feature) -- nothing to serve for THIS job.
    response = client.get(f"/api/clinical/jobs/{job.job_id}/report")
    assert response.status_code == 404


def test_clinical_job_report_on_queued_job_is_409(client: TestClient) -> None:
    created = _upload(client, _valid_study_zip()).json()
    response = client.get(f"/api/clinical/jobs/{created['job_id']}/report")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_clinical_job_report_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/api/clinical/jobs/no-such-job/report")
    assert response.status_code == 404


# --- pre-existing routes still answer ----------------------------------------


def test_preexisting_routes_still_answer(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    cases = client.get("/api/cases")
    assert cases.status_code == 200
    assert cases.json() == {"cases": []}
