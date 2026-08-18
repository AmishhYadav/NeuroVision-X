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


@pytest.fixture
def report_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: tuple[Path, Path]) -> Path:
    """Points `NVX_REPORT_DIR` at a not-yet-created directory under `tmp_path`.

    Deliberately not created here -- a demo with no reports generated yet is
    a valid configuration, and the "absent report dir" tests rely on this
    fixture NOT pre-creating it.
    """
    report_root = tmp_path / "reports"
    monkeypatch.setenv("NVX_REPORT_DIR", str(report_root))
    _clear_caches()
    return report_root


def _make_report(case_id: str, segmentation_source: str, segmentation_dir: Path) -> dict:
    """A minimal but schema-complete `build_report`-shaped dict, built by hand.

    Not imported from `neurovision.reporting.report` -- `api.py` deliberately
    imports nothing from the training package (see the module docstring), and
    this test mirrors that boundary rather than depending on the real
    schema module.
    """
    return {
        "report_version": 1,
        "case_id": case_id,
        "generated_utc": "2026-08-18T00:00:00+00:00",
        "disclaimer": (
            "This report is a research and educational decision-support artifact. It is "
            "not a diagnostic tool."
        ),
        "not_claimed": [
            ["cell type", "MRI resolves millimetre-scale tissue, not individual cells."],
        ],
        "burden": {"volumes": {"vol_WT_mm3": 1234.0}},
        "anatomy": {
            "atlas": {"name": "SRI24/TZO", "version": "1.0"},
            "region": "WT",
            "structures": [],
        },
        "eloquence": {
            "classification": "Sawaya eloquence grading",
            "involved": [],
            "distance_mm": 12.5,
        },
        "provenance": {
            "atlas_name": "SRI24/TZO",
            "atlas_version": "1.0",
            "atlas_source": "https://www.nitrc.org/projects/sri24",
            "atlas_licence": "CC-BY-SA",
            "knowledge_versions": {"eloquence_map": 1, "aal_lobes": 1},
            "segmentation_source": segmentation_source,
            "segmentation_dir": str(segmentation_dir),
            "code_revision": "deadbeef",
            "generated_utc": "2026-08-18T00:00:00+00:00",
        },
    }


def _write_report(
    report_dir: Path,
    case_id: str,
    *,
    segmentation_source: str,
    segmentation_dir: Path,
    markdown: bool = True,
) -> None:
    """Writes `<report_dir>/<case_id>.json` (and, optionally, `.md`) for one synthetic case."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report = _make_report(case_id, segmentation_source, segmentation_dir)
    (report_dir / f"{case_id}.json").write_text(json.dumps(report))
    if markdown:
        (report_dir / f"{case_id}.md").write_text(f"# Structured Report -- Case {case_id}\n")


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


# --- /api/report/{case_id} --------------------------------------------------


def test_report_ok_roundtrips_required_fields(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    prep_dir, eval_dir = backend
    _write_report(
        report_dir,
        "CaseHigh",
        segmentation_source="prediction",
        segmentation_dir=eval_dir / "predictions",
    )
    response = client.get("/api/report/CaseHigh")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "report_version",
        "case_id",
        "generated_utc",
        "disclaimer",
        "not_claimed",
        "burden",
        "anatomy",
        "eloquence",
        "provenance",
    }
    assert body["case_id"] == "CaseHigh"
    assert body["disclaimer"]
    assert body["not_claimed"]
    assert body["provenance"]["segmentation_source"] == "prediction"


def test_report_unknown_case_is_404(client: TestClient, report_dir: Path) -> None:
    response = client.get("/api/report/DoesNotExist")
    assert response.status_code == 404


def test_report_provenance_mismatch_is_500_and_names_both_dirs(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path, tmp_path: Path
) -> None:
    prep_dir, eval_dir = backend
    wrong_dir = tmp_path / "some-other-eval" / "predictions"
    _write_report(
        report_dir, "CaseHigh", segmentation_source="prediction", segmentation_dir=wrong_dir
    )

    response = client.get("/api/report/CaseHigh")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert str(wrong_dir.resolve()) in detail
    assert str((eval_dir / "predictions").resolve()) in detail


def test_report_from_ground_truth_labels_is_refused_even_though_it_is_self_consistent(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    """A ground-truth report is internally consistent with prep_dir and must STILL be refused.

    Checking a label-source report against prep_dir passes -- that is where its
    mask really came from. But every overlay this viewer draws comes from
    predictions_dir, so serving it would describe a mask that is not on screen,
    with every number in the panel correct about the wrong thing. Comparing
    ground truth against a prediction is Phase 5's experiment, not something an
    NVX_REPORT_DIR typo should turn on silently.
    """
    prep_dir, eval_dir = backend
    _write_report(report_dir, "CaseGTOk", segmentation_source="label", segmentation_dir=prep_dir)

    response = client.get("/api/report/CaseGTOk")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "ground-truth labels" in detail
    assert str((eval_dir / "predictions").resolve()) in detail
    assert "NVX_REPORT_DIR" in detail


def test_report_malformed_json_is_500(client: TestClient, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "CaseBadJson.json").write_text("{not valid json")

    response = client.get("/api/report/CaseBadJson")
    assert response.status_code == 500
    assert response.status_code != 200
    detail = response.json()["detail"]
    assert "CaseBadJson.json" in detail


# --- /api/report/{case_id}/markdown -----------------------------------------


def test_report_markdown_ok(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    prep_dir, eval_dir = backend
    _write_report(
        report_dir,
        "CaseHigh",
        segmentation_source="prediction",
        segmentation_dir=eval_dir / "predictions",
        markdown=True,
    )
    response = client.get("/api/report/CaseHigh/markdown")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "CaseHigh" in response.text


def test_report_markdown_missing_sibling_json_is_404(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "CaseOrphanMd.md").write_text("# orphan\n")

    response = client.get("/api/report/CaseOrphanMd/markdown")
    assert response.status_code == 404


def test_report_markdown_provenance_mismatch_is_500(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path, tmp_path: Path
) -> None:
    wrong_dir = tmp_path / "some-other-eval" / "predictions"
    _write_report(
        report_dir,
        "CaseBadProv",
        segmentation_source="prediction",
        segmentation_dir=wrong_dir,
        markdown=True,
    )
    response = client.get("/api/report/CaseBadProv/markdown")
    assert response.status_code == 500


# --- has_report / has_reports -----------------------------------------------


def test_has_report_flag_in_cases_list_and_detail(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    prep_dir, eval_dir = backend
    _write_report(
        report_dir,
        "CaseHigh",
        segmentation_source="prediction",
        segmentation_dir=eval_dir / "predictions",
    )

    cases = client.get("/api/cases").json()["cases"]
    high = next(c for c in cases if c["case_id"] == "CaseHigh")
    low = next(c for c in cases if c["case_id"] == "CaseLow")
    assert high["has_report"] is True
    assert low["has_report"] is False

    assert client.get("/api/cases/CaseHigh").json()["has_report"] is True
    assert client.get("/api/cases/CaseLow").json()["has_report"] is False


def test_health_has_reports_false_then_true(
    client: TestClient, backend: tuple[Path, Path], report_dir: Path
) -> None:
    prep_dir, eval_dir = backend
    before = client.get("/api/health").json()
    assert before["has_reports"] is False
    assert before["report_dir"] == str(report_dir)

    _write_report(
        report_dir,
        "CaseHigh",
        segmentation_source="prediction",
        segmentation_dir=eval_dir / "predictions",
    )
    after = client.get("/api/health").json()
    assert after["has_reports"] is True
