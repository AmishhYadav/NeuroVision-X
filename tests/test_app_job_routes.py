"""Tests for the upload/job routes in `app.backend.api`.

Everything here is synthetic: tiny in-memory NIfTI volumes built with
nibabel (never real BraTS data), and a fake preprocessed prep dir / eval dir
under `tmp_path` (the same fixture shape `tests/test_app_api.py` uses). Real
inference is monkeypatched via `jobs.start_job` so no torch model ever loads
in this file -- the job routes just have to wire HTTP onto `jobs.py`
correctly, which `tests/test_app_jobs.py` already covers at the function
level.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from app.backend import api, config, inference, jobs
from fastapi.testclient import TestClient

_SHAPE = (6, 6, 6)
_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])


def _nifti_bytes(shape: tuple[int, int, int] = _SHAPE, seed: int = 0) -> bytes:
    """Builds a tiny, valid, uncompressed NIfTI volume's raw bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=shape).astype(np.float32)
    img = nib.Nifti1Image(arr, _AFFINE)
    return img.to_bytes()


def _valid_files() -> dict[str, tuple[str, bytes, str]]:
    """Four consistent (same shape, same affine) synthetic uploads, as `httpx` files."""
    return {
        "t1": ("t1.nii", _nifti_bytes(seed=1), "application/octet-stream"),
        "t1ce": ("t1ce.nii", _nifti_bytes(seed=2), "application/octet-stream"),
        "t2": ("t2.nii", _nifti_bytes(seed=3), "application/octet-stream"),
        "flair": ("flair.nii", _nifti_bytes(seed=4), "application/octet-stream"),
    }


def _clear_caches() -> None:
    """Clears the process-global caches other test files also clear (see test_app_api.py)."""
    config.get_settings.cache_clear()


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points settings and the job store at fresh directories under `tmp_path`.

    No cases are written -- these tests never touch `/api/cases`, only the
    smoke check in test 11, which only needs `/api/cases` to answer (empty is
    fine).
    """
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    cache_dir = tmp_path / "cache"
    job_dir = tmp_path / "jobs"
    prep_dir.mkdir()
    eval_dir.mkdir()

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder -- start_job is monkeypatched in these tests")

    monkeypatch.setenv("NVX_PREP_DIR", str(prep_dir))
    monkeypatch.setenv("NVX_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("NVX_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NVX_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NVX_JOB_DIR", str(job_dir))
    monkeypatch.setenv("NVX_MAX_CASES", "24")

    jobs._JOBS.clear()
    jobs._LABEL_CONVENTIONS.clear()
    _clear_caches()
    yield tmp_path
    jobs._JOBS.clear()
    jobs._LABEL_CONVENTIONS.clear()
    _clear_caches()


@pytest.fixture
def client(backend: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A `TestClient` with `jobs.start_job` monkeypatched to a no-op.

    So `POST /api/upload` never spawns a background thread that runs real
    (torch-requiring) inference -- the route under test is the HTTP wiring,
    not `run_job` itself.
    """
    monkeypatch.setattr(api.jobs, "start_job", lambda settings, job_id: None)
    return TestClient(api.create_app())


def _fabricate_done_job(backend_root: Path) -> jobs.Job:
    """Creates a job and drives it to `"done"` by writing its outputs directly.

    Mirrors what `run_job` would have written, without running preprocessing
    or inference: a `meta.json` + `image.npy` under the job's own prep
    directory (`jobs.job_case_dir`), and a cached prediction `.npy` at
    `inference.cached_prediction_path` for the job's per-job settings (built
    the same way `api._job_settings` / `jobs.run_job` build it).
    """
    settings = config.get_settings()
    job = jobs.create_job(settings, {k: v[1] for k, v in _valid_files().items()})

    job_settings = api._job_settings(settings, job.job_id)
    case_dir = jobs.job_case_dir(settings, job.job_id)
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

    pred_path = inference.cached_prediction_path(job_settings, job.case_id)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.zeros(shape, dtype=np.uint8)
    prediction[1:3, 1:3, 1:3] = 3
    np.save(pred_path, prediction)

    job.state = "done"
    job.stage = "done"
    job.progress = 1.0
    return job


# --- 1: POST /api/upload, happy path ---------------------------------------


def test_upload_valid_returns_202_with_job_id_and_state(client: TestClient) -> None:
    response = client.post("/api/upload", files=_valid_files())
    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["state"] == "queued"


# --- 2: POST /api/upload, missing field -------------------------------------


def test_upload_missing_field_is_400_or_422_not_500(client: TestClient) -> None:
    files = _valid_files()
    del files["t2"]
    response = client.post("/api/upload", files=files)
    # A required multipart field entirely absent from the request fails
    # FastAPI's own request parsing (missing UploadFile parameter) before the
    # handler body ever runs, so this is FastAPI's 422, not jobs.create_job's
    # ValueError-derived 400. Both are asserted as acceptable; what matters is
    # that it is a client error, never a 500.
    assert response.status_code in (400, 422)


# --- 3: POST /api/upload, mismatched shapes ---------------------------------


def test_upload_mismatched_shapes_is_400_naming_roles(client: TestClient) -> None:
    files = _valid_files()
    files["t2"] = ("t2.nii", _nifti_bytes(shape=(7, 7, 7), seed=99), "application/octet-stream")
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "t1" in detail
    assert "t2" in detail


# --- 4: POST /api/upload, non-NIfTI bytes -----------------------------------


def test_upload_non_nifti_bytes_is_400_not_500(client: TestClient) -> None:
    files = _valid_files()
    files["flair"] = ("flair.nii", b"this is definitely not a NIfTI file", "text/plain")
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "flair" in response.json()["detail"]


# --- 5: GET /api/jobs lists a created job -----------------------------------


def test_list_jobs_includes_created_job(client: TestClient) -> None:
    created = client.post("/api/upload", files=_valid_files()).json()
    body = client.get("/api/jobs").json()
    ids = [j["job_id"] for j in body["jobs"]]
    assert created["job_id"] in ids


# --- 6: GET /api/jobs/{unknown} is 404 --------------------------------------


def test_get_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


# --- 7: DELETE removes, second DELETE is 404 --------------------------------


def test_delete_job_then_second_delete_is_404(client: TestClient) -> None:
    created = client.post("/api/upload", files=_valid_files()).json()
    job_id = created["job_id"]

    first = client.delete(f"/api/jobs/{job_id}")
    assert first.status_code == 200
    assert first.json()["deleted"] is True

    assert client.get(f"/api/jobs/{job_id}").status_code == 404

    second = client.delete(f"/api/jobs/{job_id}")
    assert second.status_code == 404


# --- 8/9: volume / mask on a not-done job are 409 ---------------------------


def test_volume_on_queued_job_is_409(client: TestClient) -> None:
    created = client.post("/api/upload", files=_valid_files()).json()
    response = client.get(f"/api/jobs/{created['job_id']}/volume/t1")
    assert response.status_code == 409


def test_mask_on_queued_job_is_409(client: TestClient) -> None:
    created = client.post("/api/upload", files=_valid_files()).json()
    response = client.get(f"/api/jobs/{created['job_id']}/mask/prediction")
    assert response.status_code == 409


def test_volume_and_mask_on_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/no-such-job/volume/t1").status_code == 404
    assert client.get("/api/jobs/no-such-job/mask/prediction").status_code == 404


# --- 10: a done job serves volume + mask under the existing header contract -


def test_done_job_serves_volume_and_mask_matching_case_route_contract(
    client: TestClient, backend: Path
) -> None:
    job = _fabricate_done_job(backend)

    vol_response = client.get(f"/api/jobs/{job.job_id}/volume/t1")
    assert vol_response.status_code == 200
    assert vol_response.headers["content-type"] == "application/octet-stream"
    assert vol_response.headers["x-volume-shape"] == "5,6,7"
    assert vol_response.headers["x-volume-dtype"] == "uint8"
    assert len(vol_response.content) == 5 * 6 * 7

    mask_response = client.get(f"/api/jobs/{job.job_id}/mask/prediction")
    assert mask_response.status_code == 200
    assert mask_response.headers["content-type"] == "application/octet-stream"
    assert mask_response.headers["x-volume-shape"] == "5,6,7"
    assert mask_response.headers["x-volume-dtype"] == "uint8"
    mask_arr = np.frombuffer(mask_response.content, dtype=np.uint8).reshape((5, 6, 7))
    assert mask_arr[1, 1, 1] == 3
    assert mask_arr[0, 0, 0] == 0


def test_volume_unknown_modality_on_done_job_is_404(client: TestClient, backend: Path) -> None:
    job = _fabricate_done_job(backend)
    response = client.get(f"/api/jobs/{job.job_id}/volume/bogus")
    assert response.status_code == 404


# --- 11: pre-existing routes still answer -----------------------------------


def test_preexisting_routes_still_answer(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    cases = client.get("/api/cases")
    assert cases.status_code == 200
    assert cases.json() == {"cases": []}
