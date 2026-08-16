"""Tests for scripts/burden.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_evaluate_script.py` and `scripts/smoke_test.py`
use.

Every test composes the REAL Hydra config (via `hydra.compose`, exactly like
`scripts/smoke_test.py`) against tiny synthetic preprocessed cases written
under `tmp_path`. Nothing here touches real BraTS data or the real `outputs/`
tree.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pandas as pd
import pytest

from neurovision.anatomy.burden import CaseGeometry, burden_profile
from neurovision.utils.io import ensure_dir, read_yaml, write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "burden.py"
_spec = importlib.util.spec_from_file_location("burden_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
burden_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["burden_script"] = burden_script
_spec.loader.exec_module(burden_script)

BurdenSource = burden_script.BurdenSource
resolve_sources = burden_script.resolve_sources
load_case = burden_script.load_case
profile_case = burden_script.profile_case
run_burden = burden_script.run_burden

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# --- Fixed synthetic geometry, shared by most tests -------------------------
#
# original_shape / bbox / cropped_shape are consistent: bbox slices out of
# ORIGINAL_SHAPE give exactly CROPPED_SHAPE on every axis.
ORIGINAL_SHAPE: tuple[int, int, int] = (40, 40, 20)
BBOX: list[list[int]] = [[4, 36], [4, 36], [2, 18]]
CROPPED_SHAPE: tuple[int, int, int] = (32, 32, 16)
SPACING: list[float] = [1.0, 1.0, 1.0]

# BraTS-convention affine: diag(-1, -1, 1), so axis 0 runs right -> left and
# left_is_high_index is True. affine[0][0] must be non-zero or
# CaseGeometry.from_meta refuses to guess the left/right orientation.
AFFINE: list[list[float]] = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _nested_label(shape: tuple[int, int, int]) -> np.ndarray:
    """Concentric-sphere nested ET-subset-of-TC-subset-of-WT label, small and fast.

    Same recipe as scripts/smoke_test.py's `_build_synthetic_label`, so every
    region (ET, TC, WT) is non-empty and the nesting invariant holds.
    """
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[dist < min_edge * 0.45] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.28] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.12] = 3  # ET, innermost
    return label


def _embed(
    cropped: np.ndarray, bbox: list[list[int]], original_shape: tuple[int, int, int]
) -> np.ndarray:
    """Places a cropped array into a zero background at its bbox offset.

    Mirrors the real crop/uncrop convention `bbox` uses elsewhere in this
    project: `end` exclusive, so the slice matches exactly.
    """
    full = np.zeros(original_shape, dtype=cropped.dtype)
    slices = tuple(slice(lo, hi) for lo, hi in bbox)
    full[slices] = cropped
    return full


def _write_meta(
    case_dir: Path,
    case_id: str,
    *,
    original_shape: tuple[int, int, int] = ORIGINAL_SHAPE,
    cropped_shape: tuple[int, int, int] = CROPPED_SHAPE,
    bbox: list[list[int]] = BBOX,
    spacing: list[float] = SPACING,
    affine: list[list[float]] = AFFINE,
    has_label: bool = True,
) -> None:
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(original_shape),
            "cropped_shape": list(cropped_shape),
            "bbox": [list(b) for b in bbox],
            "affine": affine,
            "spacing": spacing,
            "has_label": has_label,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )


def _write_case(prep_dir: Path, case_id: str, seed: int = 0) -> None:
    """Writes meta.json + a cropped label.npy for one case, no prediction."""
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)
    label = _nested_label(CROPPED_SHAPE)
    np.save(case_dir / "label.npy", label)


def _write_prediction(eval_dir: Path, prep_dir: Path, case_id: str) -> None:
    """Writes an uncropped prediction for a case that already has meta.json."""
    label = _nested_label(CROPPED_SHAPE)
    uncropped = _embed(label, BBOX, ORIGINAL_SHAPE)
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", uncropped)


def _write_splits(path: Path, case_ids: list[str], split: str = "test") -> None:
    payload = {"train": [], "val": [], "test": []}
    payload[split] = list(case_ids)
    write_yaml(payload, path)


def _compose_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    output_dir: Path,
    *,
    source: str = "label",
    eval_dir: Path | None = None,
    split: str = "test",
):
    """Composes the real Hydra config with tmp_path-rooted overrides.

    Uses Hydra's programmatic API, exactly like `scripts/smoke_test.py`'s
    `_compose_config`, so the real config files (including
    `configs/analysis/default.yaml`'s interpolations) compose and resolve.
    """
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"output_dir={output_dir}",
        f"analysis.burden.split={split}",
        f"analysis.burden.source={source}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
    ]
    if eval_dir is not None:
        overrides.append(f"analysis.burden.eval_dir={eval_dir}")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


CASE_IDS = ["CASE_A", "CASE_B", "CASE_C"]


def _standard_split(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Writes 3 standard label-backed cases + a test split.

    Returns:
        `(prep_dir, splits_path, eval_dir)`.
    """
    prep_dir = tmp_path / "preprocessed"
    for case_id in CASE_IDS:
        _write_case(prep_dir, case_id)
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, CASE_IDS)
    eval_dir = tmp_path / "eval"
    for case_id in CASE_IDS:
        _write_prediction(eval_dir, prep_dir, case_id)
    return prep_dir, splits_path, eval_dir


# ---------------------------------------------------------------------------
# 1. Happy path, source=label
# ---------------------------------------------------------------------------


def test_happy_path_label_source(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out_label"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")

    csv_path = run_burden(cfg)
    assert csv_path == output_dir / "burden.csv"

    df = pd.read_csv(csv_path)
    assert len(df) == len(CASE_IDS)
    assert df.columns[0] == "case_id"
    assert set(df["case_id"]) == set(CASE_IDS)

    # Key set matches burden_profile's output plus case_id.
    label = _nested_label(CROPPED_SHAPE)
    meta = {
        "spacing": SPACING,
        "affine": AFFINE,
        "original_shape": ORIGINAL_SHAPE,
        "bbox": BBOX,
    }
    geom = CaseGeometry.from_meta(meta, cropped=True)
    expected_keys = {"case_id"} | set(burden_profile(label, geom).keys())
    assert set(df.columns) == expected_keys


# ---------------------------------------------------------------------------
# 2. Happy path, source=prediction
# ---------------------------------------------------------------------------


def test_happy_path_prediction_source(tmp_path: Path) -> None:
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out_pred"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, output_dir, source="prediction", eval_dir=eval_dir
    )

    csv_path = run_burden(cfg)
    df = pd.read_csv(csv_path)
    assert len(df) == len(CASE_IDS)
    assert df.columns[0] == "case_id"
    assert set(df["case_id"]) == set(CASE_IDS)


# ---------------------------------------------------------------------------
# 3. cropped follows source: regression guard for the crop-offset hemisphere bug
# ---------------------------------------------------------------------------


def test_cropped_follows_source_not_a_flag(tmp_path: Path) -> None:
    case_id = "SIDED"
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    # A lesion entirely on the LOW-index side of axis 0 in ORIGINAL geometry
    # (original midline = (40 - 1) / 2 = 19.5; block spans indices 6..15).
    original = np.zeros(ORIGINAL_SHAPE, dtype=np.uint8)
    original[6:16, 10:20, 5:12] = 2  # ED -> non-empty WT
    np.save(case_dir / "label.npy", original[tuple(slice(lo, hi) for lo, hi in BBOX)])

    eval_dir = tmp_path / "eval"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", original)

    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, [case_id])

    cfg_label = _compose_cfg(
        tmp_path, prep_dir, splits_path, tmp_path / "out_label", source="label"
    )
    cfg_pred = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        tmp_path / "out_pred",
        source="prediction",
        eval_dir=eval_dir,
    )

    df_label = pd.read_csv(run_burden(cfg_label)).set_index("case_id")
    df_pred = pd.read_csv(run_burden(cfg_pred)).set_index("case_id")

    side_label = df_label.loc[case_id, "dominant_side_WT"]
    side_pred = df_pred.loc[case_id, "dominant_side_WT"]
    assert side_label in ("left", "right")
    assert side_label == side_pred


# ---------------------------------------------------------------------------
# 4. Shape mismatch raises, naming the case
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    case_id = "BAD_SHAPE"
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    # Prediction is expected in ORIGINAL geometry; write it CROPPED instead.
    eval_dir = tmp_path / "eval"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", _nested_label(CROPPED_SHAPE))

    source = BurdenSource(
        case_id=case_id,
        array_path=predictions_dir / f"{case_id}.npy",
        meta_path=case_dir / "meta.json",
        cropped=False,
    )
    with pytest.raises(ValueError, match=case_id):
        load_case(source)


# ---------------------------------------------------------------------------
# 5. Missing files are reported and excluded
# ---------------------------------------------------------------------------


def test_missing_case_is_reported_and_excluded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    (prep_dir / "CASE_B" / "label.npy").unlink()

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")

    with caplog.at_level(logging.WARNING, logger="burden_script"):
        csv_path = run_burden(cfg)

    df = pd.read_csv(csv_path)
    assert len(df) == len(CASE_IDS) - 1
    assert "CASE_B" not in set(df["case_id"])
    assert any("CASE_B" in record.getMessage() for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


# ---------------------------------------------------------------------------
# 6. All files missing raises FileNotFoundError naming the resolved directory
# ---------------------------------------------------------------------------


def test_all_missing_raises_file_not_found(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    for case_id in CASE_IDS:
        (prep_dir / case_id / "label.npy").unlink()

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")

    with pytest.raises(FileNotFoundError) as excinfo:
        run_burden(cfg)
    assert str(prep_dir.resolve()) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 7. A case that raises does not kill the run
# ---------------------------------------------------------------------------


def test_one_bad_case_does_not_kill_the_run(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)

    # Overwrite CASE_B's label with an invalid class value (4 -- raw,
    # unremapped BraTS uses 4 for enhancing tumor; burden_profile rejects it).
    bad = _nested_label(CROPPED_SHAPE)
    bad[bad == 3] = 4
    np.save(prep_dir / "CASE_B" / "label.npy", bad)

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")

    csv_path = run_burden(cfg)
    df = pd.read_csv(csv_path)
    assert set(df["case_id"]) == {"CASE_A", "CASE_C"}


# ---------------------------------------------------------------------------
# 8. Zero successes raises RuntimeError
# ---------------------------------------------------------------------------


def test_zero_successes_raises_runtime_error(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    for case_id in CASE_IDS:
        bad = _nested_label(CROPPED_SHAPE)
        bad[bad == 3] = 4
        np.save(prep_dir / case_id / "label.npy", bad)

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")

    with pytest.raises(RuntimeError):
        run_burden(cfg)


# ---------------------------------------------------------------------------
# 9. Partial output survives and has no NaN case_id
# ---------------------------------------------------------------------------


def test_partial_output_is_readable_and_has_no_nan_case_id(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    bad = _nested_label(CROPPED_SHAPE)
    bad[bad == 3] = 4
    np.save(prep_dir / "CASE_B" / "label.npy", bad)

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="label")
    csv_path = run_burden(cfg)

    df = pd.read_csv(csv_path)
    assert not df["case_id"].isna().any()
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 10. burden_config.yaml is written and records the split and the source
# ---------------------------------------------------------------------------


def test_burden_config_yaml_written(tmp_path: Path) -> None:
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, prep_dir, splits_path, output_dir, source="prediction", eval_dir=eval_dir
    )
    run_burden(cfg)

    config_path = output_dir / "burden_config.yaml"
    assert config_path.is_file()
    record = read_yaml(config_path)
    assert record["split"] == "test"
    assert record["source"] == "prediction"
    assert "resolved_source_dir" in record
    assert str(eval_dir.resolve() / "predictions") == record["resolved_source_dir"]


# ---------------------------------------------------------------------------
# 11. eval_dir null with source=prediction raises ValueError naming the key
# ---------------------------------------------------------------------------


def test_eval_dir_null_with_prediction_source_raises(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    # source="prediction" is the config default; eval_dir defaults to null.
    cfg = _compose_cfg(tmp_path, prep_dir, splits_path, output_dir, source="prediction")

    with pytest.raises(ValueError, match="eval_dir"):
        resolve_sources(cfg)


# ---------------------------------------------------------------------------
# 12. Determinism: two runs over the same inputs produce byte-identical CSVs
# ---------------------------------------------------------------------------


def test_determinism(tmp_path: Path) -> None:
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)

    out_a = tmp_path / "out_a"
    cfg_a = _compose_cfg(tmp_path, prep_dir, splits_path, out_a, source="label")
    csv_a = run_burden(cfg_a)

    out_b = tmp_path / "out_b"
    cfg_b = _compose_cfg(tmp_path, prep_dir, splits_path, out_b, source="label")
    csv_b = run_burden(cfg_b)

    assert csv_a.read_bytes() == csv_b.read_bytes()
