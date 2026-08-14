"""Tests for `app.backend.api`, the demo's FastAPI HTTP layer.

Everything here is synthetic: a fake preprocessed case tree and a fake
`scripts/evaluate.py` output directory are built under `tmp_path`, never the
user's real `data/` or `outputs/`. The prediction is deliberately written in
ORIGINAL (uncropped) geometry and the fixture crops it by hand with the same
bbox `meta.json` carries -- exercising the geometry trap `volumes.py`'s
`_crop_to_meta` exists to catch, not just the happy path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from app.backend import api, config, volumes
from fastapi.testclient import TestClient

CROPPED_SHAPE = (8, 9, 10)
ORIGINAL_SHAPE = (12, 13, 14)
# Extents (10-2, 11-2, 12-2) == CROPPED_SHAPE, so a correctly-cropped
# prediction lands exactly on meta.shape.
BBOX = ((2, 10), (2, 11), (2, 12))


def _make_label(shape: tuple[int, int, int]) -> np.ndarray:
    """A small synthetic label with all three classes present."""
    label = np.zeros(shape, dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 2  # oedema -> part of WT only
    label[1:2, 1:2, 1:2] = 1  # necrotic core -> part of TC (and WT)
    label[4:6, 4:6, 4:6] = 3  # enhancing -> part of ET, TC and WT
    return label


def _write_case(
    prep_dir: Path,
    eval_dir: Path,
    case_id: str,
    *,
    include_label: bool = True,
    include_logits: bool = True,
) -> None:
    """Writes one synthetic case: preprocessed dir + prediction + logits."""
    case_root = prep_dir / case_id
    case_root.mkdir(parents=True)

    image = np.zeros((4, *CROPPED_SHAPE), dtype=np.float16)
    image[:, 2:6, 2:6, 2:6] = 1.0  # a nonzero "brain" block
    np.save(case_root / "image.npy", image)

    label = _make_label(CROPPED_SHAPE)
    if include_label:
        np.save(case_root / "label.npy", label)

    meta = {
        "cropped_shape": list(CROPPED_SHAPE),
        "original_shape": list(ORIGINAL_SHAPE),
        "bbox": [list(pair) for pair in BBOX],
        "spacing": [1.0, 1.0, 1.0],
    }
    (case_root / "meta.json").write_text(json.dumps(meta))

    predictions_dir = eval_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    # The prediction is a near-copy of the label with one voxel of
    # disagreement, saved in ORIGINAL geometry -- exactly how
    # scripts/evaluate.py writes it (see uncrop_to_original).
    pred_cropped = label.copy()
    pred_cropped[4, 4, 4] = 0
    pred_original = np.zeros(ORIGINAL_SHAPE, dtype=np.uint8)
    (d0, d1), (h0, h1), (w0, w1) = BBOX
    pred_original[d0:d1, h0:h1, w0:w1] = pred_cropped
    np.save(predictions_dir / f"{case_id}.npy", pred_original)

    if include_logits:
        logits_dir = eval_dir / "logits"
        logits_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
        logits = rng.normal(size=(3, *CROPPED_SHAPE)).astype(np.float16)
        np.save(logits_dir / f"{case_id}.npy", logits)


def _write_bad_geometry_case(prep_dir: Path, eval_dir: Path, case_id: str) -> None:
    """A case whose `meta.json` bbox extent disagrees with its own `cropped_shape`.

    Cropping any full-size prediction with this bbox cannot produce
    `cropped_shape`, so it exercises `_crop_to_meta`'s ValueError guard --
    the "prediction and meta.json come from different preprocessing runs"
    case, which must surface as a 500, not a 404. Deliberately has no
    `image.npy`, so `list_cases()` excludes it from every listing endpoint
    and it can only be reached by requesting it directly.
    """
    case_root = prep_dir / case_id
    case_root.mkdir(parents=True)
    meta = {
        "cropped_shape": list(CROPPED_SHAPE),
        "original_shape": list(ORIGINAL_SHAPE),
        "bbox": [[2, 9], [2, 11], [2, 12]],  # axis 0 extent is 7, not 8
        "spacing": [1.0, 1.0, 1.0],
    }
    (case_root / "meta.json").write_text(json.dumps(meta))

    predictions_dir = eval_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.save(predictions_dir / f"{case_id}.npy", np.zeros(ORIGINAL_SHAPE, dtype=np.uint8))


def _clear_caches() -> None:
    """Clears every process-global cache the API depends on.

    `get_settings`, `_metrics_table` and `_compute_profile` are all
    `lru_cache`-wrapped module globals -- left dirty across tests, whichever
    test ran first silently decides what every later test sees, which is
    exactly the failure mode this fixture exists to prevent.
    """
    config.get_settings.cache_clear()
    volumes._metrics_table.cache_clear()
    api._compute_profile.cache_clear()


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Builds a synthetic prep dir + eval dir and points settings at them."""
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"
    cache_dir = tmp_path / "cache"
    prep_dir.mkdir()
    eval_dir.mkdir()

    _write_case(prep_dir, eval_dir, "CaseHigh")
    _write_case(prep_dir, eval_dir, "CaseLow")
    _write_case(prep_dir, eval_dir, "CaseNoLabel", include_label=False)
    _write_case(prep_dir, eval_dir, "CaseNoLogits", include_logits=False)
    _write_bad_geometry_case(prep_dir, eval_dir, "CaseBadGeo")

    metrics_df = pd.DataFrame(
        [
            {
                "case_id": "CaseHigh",
                "dice_ET": 0.94999,
                "dice_TC": 0.96001,
                "dice_WT": 0.97,
                "dice_mean": 0.96,
                "hd95_ET": 1.2,
                "hd95_TC": 1.1,
                "hd95_WT": 1.0,
                "gt_empty_ET": 0,
                "gt_empty_TC": 0,
                "gt_empty_WT": 0,
            },
            {
                "case_id": "CaseLow",
                "dice_ET": 0.5,
                "dice_TC": 0.55,
                "dice_WT": 0.6,
                "dice_mean": 0.55,
                "hd95_ET": 5.2,
                "hd95_TC": 5.1,
                "hd95_WT": 5.0,
                "gt_empty_ET": 0,
                "gt_empty_TC": 0,
                "gt_empty_WT": 0,
            },
        ]
    )
    metrics_df.to_csv(eval_dir / "per_case_metrics.csv", index=False)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not a real checkpoint, existence is all that is checked")

    monkeypatch.setenv("NVX_PREP_DIR", str(prep_dir))
    monkeypatch.setenv("NVX_EVAL_DIR", str(eval_dir))
    monkeypatch.setenv("NVX_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NVX_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("NVX_MAX_CASES", "24")

    _clear_caches()
    yield prep_dir, eval_dir
    _clear_caches()


@pytest.fixture
def client(backend: tuple[Path, Path]) -> TestClient:
    return TestClient(api.create_app())


# --- /api/health -------------------------------------------------------


def test_health_everything_present(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["experiment"] == "baseline_unet3d"
    assert body["checkpoint_present"] is True
    assert body["has_metrics"] is True
    # CaseBadGeo has no image.npy, so list_cases() excludes it.
    assert body["case_count"] == 4


def test_health_degrades_when_eval_dir_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("NVX_EVAL_DIR", str(tmp_path / "does-not-exist"))
    _clear_caches()
    client = TestClient(api.create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["case_count"] == 0
    assert body["has_metrics"] is False


# --- /api/cases ----------------------------------------------------------


def test_cases_ordering_and_shape(client: TestClient) -> None:
    body = client.get("/api/cases").json()
    ids = [c["case_id"] for c in body["cases"]]
    assert "CaseBadGeo" not in ids
    assert ids.index("CaseHigh") < ids.index("CaseLow")

    high = next(c for c in body["cases"] if c["case_id"] == "CaseHigh")
    assert high["dice_mean"] == 0.96
    assert high["dice"] == {"ET": 0.95, "TC": 0.96, "WT": 0.97}  # rounded to 4dp
    assert high["has_label"] is True
    assert high["has_logits"] is True

    no_label = next(c for c in body["cases"] if c["case_id"] == "CaseNoLabel")
    assert no_label["has_label"] is False
    assert no_label["dice_mean"] is None
    assert no_label["dice"] is None


# --- /api/cases/{case_id} -------------------------------------------------


def test_case_detail_shape(client: TestClient) -> None:
    body = client.get("/api/cases/CaseHigh").json()
    assert body["meta"]["case_id"] == "CaseHigh"
    assert body["meta"]["shape"] == list(CROPPED_SHAPE)
    assert body["metrics"]["dice_mean"] == 0.96

    regions = body["regions"]
    assert regions["label"] is not None
    for side in ("prediction", "label"):
        assert set(regions[side].keys()) == {"ET", "TC", "WT"}
        for region in ("ET", "TC", "WT"):
            assert "voxels" in regions[side][region]
            assert "ml" in regions[side][region]


def test_case_detail_no_label_has_null_regions(client: TestClient) -> None:
    body = client.get("/api/cases/CaseNoLabel").json()
    assert body["regions"]["label"] is None
    assert body["regions"]["prediction"] is not None


def test_case_detail_unknown_case_is_404(client: TestClient) -> None:
    response = client.get("/api/cases/DoesNotExist")
    assert response.status_code == 404


# --- /api/cases/{case_id}/volume/{modality} -------------------------------


def test_volume_length_and_shape_header(client: TestClient) -> None:
    response = client.get("/api/cases/CaseHigh/volume/t1")
    assert response.status_code == 200
    assert response.headers["x-volume-shape"] == "8,9,10"
    assert response.headers["x-volume-dtype"] == "uint8"
    assert len(response.content) == 8 * 9 * 10


def test_volume_unknown_modality_is_404(client: TestClient) -> None:
    response = client.get("/api/cases/CaseHigh/volume/bogus")
    assert response.status_code == 404
    assert "t1" in response.json()["detail"]


# --- /api/cases/{case_id}/mask/{source} -----------------------------------


def test_mask_prediction_ok(client: TestClient) -> None:
    response = client.get("/api/cases/CaseHigh/mask/prediction")
    assert response.status_code == 200
    assert len(response.content) == 8 * 9 * 10
    assert response.headers["x-volume-shape"] == "8,9,10"


def test_mask_label_missing_is_404(client: TestClient) -> None:
    response = client.get("/api/cases/CaseNoLabel/mask/label")
    assert response.status_code == 404
    assert "label" in response.json()["detail"].lower()


def test_mask_unknown_source_is_404(client: TestClient) -> None:
    response = client.get("/api/cases/CaseHigh/mask/bogus")
    assert response.status_code == 404
    assert "prediction" in response.json()["detail"]


# --- /api/cases/{case_id}/uncertainty -------------------------------------


def test_uncertainty_ok(client: TestClient) -> None:
    response = client.get("/api/cases/CaseHigh/uncertainty")
    assert response.status_code == 200
    assert response.headers["x-uncertainty-kind"] == "predictive-entropy-single-pass"
    assert len(response.content) == 8 * 9 * 10


def test_uncertainty_404_names_save_logits(client: TestClient) -> None:
    response = client.get("/api/cases/CaseNoLogits/uncertainty")
    assert response.status_code == 404
    assert "save_logits" in response.json()["detail"]


# --- /api/cases/{case_id}/profile -----------------------------------------


def test_profile_lengths_and_range(client: TestClient) -> None:
    body = client.get("/api/cases/CaseHigh/profile").json()
    planes = body["planes"]
    assert planes["sagittal"]["n"] == 8
    assert planes["coronal"]["n"] == 9
    assert planes["axial"]["n"] == 10

    for name, n in (("sagittal", 8), ("coronal", 9), ("axial", 10)):
        plane = planes[name]
        for key in ("tumor", "error", "entropy"):
            values = plane[key]
            assert values is not None
            assert len(values) == n
            assert all(0.0 <= v <= 1.0 for v in values)


def test_profile_null_fields_when_label_or_logits_missing(client: TestClient) -> None:
    no_label = client.get("/api/cases/CaseNoLabel/profile").json()
    for plane in no_label["planes"].values():
        assert plane["error"] is None
        assert plane["tumor"] is not None  # tumor never depends on the label

    no_logits = client.get("/api/cases/CaseNoLogits/profile").json()
    for plane in no_logits["planes"].values():
        assert plane["entropy"] is None
        assert plane["error"] is not None  # this case does have a label


# --- error mapping ---------------------------------------------------------


def test_bad_geometry_is_500_not_404(client: TestClient) -> None:
    response = client.get("/api/cases/CaseBadGeo")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "meta.json" in detail
