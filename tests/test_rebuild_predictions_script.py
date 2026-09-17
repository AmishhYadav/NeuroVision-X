"""Tests for scripts/rebuild_predictions.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, following the exact same pattern as tests/test_replay_logits.py /
tests/test_evaluate_script.py.

Everything here is synthetic, tiny, and CPU-only: 2 hand-built cases (never
real BraTS data), cropped shape (3, 4, 5, 6), written to `tmp_path` in the
same on-disk layout `scripts/evaluate.py`
(`<eval_dir>/logits/<case_id>.npy`, `<eval_dir>/eval_config.yaml`,
`<eval_dir>/per_case_metrics.csv`) and
`neurovision.data.preprocessing.preprocess_case`
(`<prep_dir>/<case_id>/{label.npy,meta.json}`) actually produce.

Each case's ground-truth class map is designed so its three region classes
(necrotic core=1, edema=2, enhancing tumor=3) are all present and mutually
disjoint, and its logits are +/-10 exactly where `classes_to_regions` of that
same class map is 1/0 -- so, at the project-default threshold (0.5) and a
no-op post-processing chain (`min_component_size: 0`, `keep_largest_only:
false`, `et_min_volume: 0`), the rebuilt prediction reproduces the class map
EXACTLY. Using the same class map as the case's `label.npy` (a "perfect
prediction") means every region's Dice is exactly 1.0, so a hand-built
`per_case_metrics.csv` needs no separate Dice computation to be internally
consistent -- it is filled in by calling
`neurovision.metrics.segmentation.compute_case_metrics` on the identical
target regions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from neurovision.metrics.segmentation import classes_to_regions, compute_case_metrics
from neurovision.utils.io import read_json, write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_predictions.py"
_spec = importlib.util.spec_from_file_location("rebuild_predictions_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
rebuild_predictions_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["rebuild_predictions_script"] = rebuild_predictions_script
_spec.loader.exec_module(rebuild_predictions_script)

rebuild_predictions = rebuild_predictions_script.rebuild_predictions
run_rebuild = rebuild_predictions_script.run_rebuild

CROPPED_SHAPE = (4, 5, 6)
ORIGINAL_SHAPE = (8, 8, 8)
# [start, end) per axis, (D, H, W) order, extent matching CROPPED_SHAPE and
# fitting inside ORIGINAL_SHAPE.
BBOX = [[1, 5], [1, 6], [1, 7]]

# Project-default-shaped post-processing block, matching the keys of a real
# eval_config.yaml's inference.postprocess (see
# outputs/eval_test/eval_config.yaml). min_component_size/keep_largest_only/
# et_min_volume are all no-ops here so the rebuilt prediction reproduces the
# hand-built class map exactly.
POSTPROCESS_CFG = {
    "threshold": 0.5,
    "enforce_nesting": True,
    "min_component_size": 0,
    "connectivity": 1,
    "keep_largest_only": False,
    "et_min_volume": 0,
}


def _make_classes(shape: tuple[int, int, int]) -> np.ndarray:
    """A uint8 class map with disjoint, non-empty ET/TC/WT regions."""
    classes = np.zeros(shape, dtype=np.uint8)
    classes[0:2, 0:2, 0:2] = 1  # necrotic core -> TC only
    classes[2:4, 2:3, 2:4] = 2  # edema -> WT only
    classes[0:1, 3:5, 3:6] = 3  # enhancing tumor -> ET, TC and WT
    return classes


def _logits_for_classes(classes: np.ndarray) -> np.ndarray:
    """Builds (3, D, H, W) logits with a clear sign wherever a region is 1/0."""
    regions = classes_to_regions(torch.from_numpy(classes.astype(np.int64)))[0].numpy()
    return np.where(regions > 0.5, 10.0, -10.0).astype(np.float32)


def _write_case(eval_dir: Path, prep_dir: Path, case_id: str, classes: np.ndarray) -> np.ndarray:
    """Writes one case's logits/, meta.json and label.npy. Returns its uncropped class map."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", _logits_for_classes(classes).astype(np.float16))

    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", classes)
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(ORIGINAL_SHAPE),
            "cropped_shape": list(CROPPED_SHAPE),
            "bbox": BBOX,
            "spacing": [1.0, 1.0, 1.0],
            "has_label": True,
        },
        case_dir / "meta.json",
    )

    uncropped = np.zeros(ORIGINAL_SHAPE, dtype=np.uint8)
    (d0, d1), (h0, h1), (w0, w1) = BBOX
    uncropped[d0:d1, h0:h1, w0:w1] = classes
    return uncropped


def _expected_dice_row(classes: np.ndarray) -> dict[str, float]:
    """The Dice a perfect prediction (== the label itself) scores, per region."""
    label_t = torch.from_numpy(classes.astype(np.int64))
    regions = classes_to_regions(label_t)
    return compute_case_metrics(regions, regions, spacing=(1.0, 1.0, 1.0))


def _build_split(
    tmp_path: Path, *, corrupt_case_id: str | None = None
) -> tuple[Path, Path, dict[str, np.ndarray]]:
    """Builds a 2-case synthetic eval_dir/prep_dir pair plus its per_case_metrics.csv.

    Args:
        corrupt_case_id: If set, that case's dice_ET row in the written
            per_case_metrics.csv is perturbed far outside the verification
            tolerance, to test that rebuild_predictions raises.

    Returns:
        `(eval_dir, prep_dir, {case_id: expected_uncropped_class_map})`.
    """
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    write_yaml({"inference": {"postprocess": POSTPROCESS_CFG}}, eval_dir / "eval_config.yaml")

    expected: dict[str, np.ndarray] = {}
    rows: dict[str, dict[str, float]] = {}
    for case_id in ("case_a", "case_b"):
        classes = _make_classes(CROPPED_SHAPE)
        expected[case_id] = _write_case(eval_dir, prep_dir, case_id, classes)
        row = _expected_dice_row(classes)
        if case_id == corrupt_case_id:
            row["dice_ET"] = 0.0  # far outside the 1e-4 tolerance
        rows[case_id] = row

    per_case_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("case_id")
    per_case_df.to_csv(eval_dir / "per_case_metrics.csv")

    return eval_dir, prep_dir, expected


# ---------------------------------------------------------------------------
# rebuild_predictions
# ---------------------------------------------------------------------------


def test_rebuild_writes_correct_uncropped_predictions(tmp_path):
    eval_dir, prep_dir, expected = _build_split(tmp_path)

    summary = rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    assert summary["n_written"] == 2
    assert summary["n_skipped"] == 0
    for case_id, expected_array in expected.items():
        saved = np.load(eval_dir / "predictions" / f"{case_id}.npy")
        assert saved.dtype == np.uint8
        assert saved.shape == ORIGINAL_SHAPE
        np.testing.assert_array_equal(saved, expected_array)


def test_rebuild_writes_verify_json_with_near_zero_delta(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path)

    summary = rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    verify_path = eval_dir / "predictions_rebuild_verify.json"
    assert verify_path.is_file()
    verify = read_json(verify_path)
    assert verify["n_verified"] == 2
    assert verify["max_abs_delta"] < 1e-4
    assert summary["verify"]["n_verified"] == 2
    for case_id in ("case_a", "case_b"):
        for metric in ("dice_ET", "dice_TC", "dice_WT"):
            assert abs(verify["per_case"][case_id][metric]["delta"]) < 1e-4


def test_rebuild_second_run_skips_existing_files(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path)
    rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    second = rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    assert second["n_written"] == 0
    assert second["n_skipped"] == 2


def test_rebuild_overwrite_true_rewrites_existing_files(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path)
    rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    second = rebuild_predictions(eval_dir, prep_dir, overwrite=True, verify_n=5)

    assert second["n_written"] == 2
    assert second["n_skipped"] == 0


def test_rebuild_limit_only_rebuilds_first_n_cases(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path)

    summary = rebuild_predictions(eval_dir, prep_dir, limit=1, verify_n=5)

    assert summary["n_total"] == 1
    assert (eval_dir / "predictions" / "case_a.npy").is_file()
    assert not (eval_dir / "predictions" / "case_b.npy").is_file()


def test_rebuild_raises_on_wrong_per_case_metrics(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, corrupt_case_id="case_a")

    with pytest.raises(ValueError, match="self-verification FAILED"):
        rebuild_predictions(eval_dir, prep_dir, verify_n=5)


def test_rebuild_raises_when_no_logits_saved(tmp_path):
    eval_dir = tmp_path / "empty_eval"
    (eval_dir / "logits").mkdir(parents=True)
    prep_dir = tmp_path / "prep"

    with pytest.raises(FileNotFoundError, match="no saved logits"):
        rebuild_predictions(eval_dir, prep_dir)


def test_rebuild_raises_when_eval_config_missing(tmp_path):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    classes = _make_classes(CROPPED_SHAPE)
    _write_case(eval_dir, prep_dir, "case_a", classes)
    # No eval_config.yaml written.

    with pytest.raises(FileNotFoundError, match="eval_config.yaml"):
        rebuild_predictions(eval_dir, prep_dir)


def test_rebuild_verify_skipped_without_per_case_metrics(tmp_path, caplog):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    write_yaml({"inference": {"postprocess": POSTPROCESS_CFG}}, eval_dir / "eval_config.yaml")
    classes = _make_classes(CROPPED_SHAPE)
    _write_case(eval_dir, prep_dir, "case_a", classes)

    with caplog.at_level("WARNING"):
        summary = rebuild_predictions(eval_dir, prep_dir, verify_n=5)

    assert summary["verify"] is None
    assert not (eval_dir / "predictions_rebuild_verify.json").is_file()
    assert "UNVERIFIED" in caplog.text


# ---------------------------------------------------------------------------
# run_rebuild (the Hydra-cfg-shaped wrapper)
# ---------------------------------------------------------------------------


def test_run_rebuild_raises_when_eval_dir_unset():
    cfg = OmegaConf.create({"data": {"preprocessing": {"out_dir": "unused"}}})

    with pytest.raises(ValueError, match="rebuild.eval_dir is not set"):
        run_rebuild(cfg)


def test_run_rebuild_uses_cfg_data_preprocessing_out_dir_by_default(tmp_path):
    eval_dir, prep_dir, expected = _build_split(tmp_path)
    cfg = OmegaConf.create(
        {
            "rebuild": {"eval_dir": str(eval_dir)},
            "data": {"preprocessing": {"out_dir": str(prep_dir)}},
        }
    )

    summary = run_rebuild(cfg)

    assert summary["n_written"] == 2
    for case_id, expected_array in expected.items():
        saved = np.load(eval_dir / "predictions" / f"{case_id}.npy")
        np.testing.assert_array_equal(saved, expected_array)
