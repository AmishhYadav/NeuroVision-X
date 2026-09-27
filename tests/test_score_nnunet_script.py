"""Tests for scripts/score_nnunet.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, matching every other scripts/*.py test
file. Every test runs on CPU, on tiny synthetic preprocessed cases written
under `tmp_path`, well under a second. Nothing here touches real BraTS data
or a real nnU-Net export.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from neurovision.utils.io import write_json
from tests.script_loader import load_script

score_nnunet_script = load_script("score_nnunet")

score_predictions = score_nnunet_script.score_predictions
run_score_nnunet = score_nnunet_script.run_score_nnunet
resolve_lesionwise = score_nnunet_script.resolve_lesionwise

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# --------------------------------------------------------------------------- #
# Shared synthetic geometry -- a small grid, LPS orientation, WITH a
# translation (unlike test_nnunet_import.py's zero-translation fixture) so
# this test file also exercises the affine-equality check on a non-trivial
# offset, not just on flips.
# --------------------------------------------------------------------------- #

_ORIGINAL_SHAPE = (20, 20, 20)
_BBOX = [[2, 18], [2, 18], [2, 18]]  # cropped shape (16, 16, 16)
_CROPPED_SHAPE = (16, 16, 16)
_SPACING = [1.0, 1.0, 1.0]
_TARGET_AXCODES = ("L", "P", "S")

# LPS-convention affine (axis 0 flips for Left, axis 1 flips for Posterior,
# axis 2 does not for Superior) with a nonzero translation on every axis.
_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 7.0],
        [0.0, -1.0, 0.0, -3.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _nested_cubes_label() -> np.ndarray:
    """A (16, 16, 16) label with NCR/ED/ET cubes, each region >> 50 voxels.

    Same recipe as tests/test_nnunet_import.py's `_nested_cubes_label`
    (scaled to this file's cropped shape): WT = the 14^3 outer cube, TC =
    the 10^3 middle cube, ET = the 6^3 inner cube.
    """
    label = np.zeros(_CROPPED_SHAPE, dtype=np.uint8)
    label[1:15, 1:15, 1:15] = 2  # ED shell -> completes WT
    label[3:13, 3:13, 3:13] = 1  # NCR -> completes TC
    label[5:11, 5:11, 5:11] = 3  # ET, innermost
    return label


def _write_case(prep_dir: Path, case_id: str, label_cropped: np.ndarray) -> None:
    """Writes `<prep_dir>/<case_id>/{label.npy, meta.json}` in our convention."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label_cropped)
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(_ORIGINAL_SHAPE),
            "cropped_shape": list(label_cropped.shape),
            "bbox": [list(b) for b in _BBOX],
            "affine": _AFFINE.tolist(),
            "spacing": _SPACING,
            "has_label": True,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )


def _write_prediction(
    pred_dir: Path, case_id: str, label_cropped_our_convention: np.ndarray, affine: np.ndarray
) -> None:
    """Writes an nnU-Net-convention prediction NIfTI on the original grid.

    Args:
        pred_dir: Folder predictions are written into.
        case_id: Determines the output filename, `<case_id>.nii.gz`.
        label_cropped_our_convention: The cropped label in OUR label
            convention (0 background, 1 NCR, 2 ED, 3 ET) -- swapped to
            nnU-Net's convention (1 ED, 2 NCR) before writing, exactly the
            inverse of `nnunet_to_project_labels`.
        affine: Affine to write the NIfTI with.
    """
    pred_dir.mkdir(parents=True, exist_ok=True)
    nnunet_cropped = label_cropped_our_convention.copy()
    nnunet_cropped[label_cropped_our_convention == 1] = 2
    nnunet_cropped[label_cropped_our_convention == 2] = 1

    full = np.zeros(_ORIGINAL_SHAPE, dtype=np.int16)
    slices = tuple(slice(a, b) for a, b in _BBOX)
    full[slices] = nnunet_cropped
    nib.save(nib.Nifti1Image(full, affine), str(pred_dir / f"{case_id}.nii.gz"))


# --------------------------------------------------------------------------- #
# score_predictions -- happy path
# --------------------------------------------------------------------------- #


def test_score_predictions_perfect_predictions_give_dice_one(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "out"

    label = _nested_cubes_label()
    for case_id in ("case_a", "case_b"):
        _write_case(prep_dir, case_id, label)
        _write_prediction(pred_dir, case_id, label, _AFFINE)

    table = score_predictions(
        case_ids=["case_a", "case_b"],
        pred_dir=pred_dir,
        prep_dir=prep_dir,
        target_axcodes=_TARGET_AXCODES,
        lesionwise=None,
        selftest_case="case_a",
        out_dir=out_dir,
    )

    assert set(table.index) == {"case_a", "case_b"}
    for region in ("ET", "TC", "WT"):
        assert table[f"dice_{region}"].tolist() == pytest.approx([1.0, 1.0])

    per_case_path = out_dir / "per_case_metrics.csv"
    summary_path = out_dir / "summary.csv"
    assert per_case_path.is_file()
    assert summary_path.is_file()

    written = pd.read_csv(per_case_path)
    assert written.columns[0] == "case_id"
    assert set(written["case_id"]) == {"case_a", "case_b"}

    summary = pd.read_csv(summary_path)
    assert len(summary) == 1  # one row: the mean of every numeric column
    assert summary["dice_ET"].iloc[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Completeness check
# --------------------------------------------------------------------------- #


def test_missing_prediction_file_raises_and_writes_nothing(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "out"

    label = _nested_cubes_label()
    _write_case(prep_dir, "case_a", label)
    _write_case(prep_dir, "case_b", label)
    _write_prediction(pred_dir, "case_a", label, _AFFINE)
    # case_b's prediction is deliberately never written.

    with pytest.raises(ValueError, match="case_b"):
        score_predictions(
            case_ids=["case_a", "case_b"],
            pred_dir=pred_dir,
            prep_dir=prep_dir,
            target_axcodes=_TARGET_AXCODES,
            lesionwise=None,
            selftest_case="case_a",
            out_dir=out_dir,
        )

    assert not (out_dir / "per_case_metrics.csv").exists()


# --------------------------------------------------------------------------- #
# Round-trip self-test gate
# --------------------------------------------------------------------------- #


def test_selftest_case_lacking_ed_raises_before_any_scoring(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "out"

    # case_bad has only ET (label 3) -- no NCR (1) or ED (2) -- so the
    # round-trip self-test cannot expose a missed 1<->2 swap and must
    # refuse to run at all (see roundtrip_self_test's docstring).
    bad_label = np.zeros(_CROPPED_SHAPE, dtype=np.uint8)
    bad_label[5:11, 5:11, 5:11] = 3
    _write_case(prep_dir, "case_bad", bad_label)

    good_label = _nested_cubes_label()
    _write_case(prep_dir, "case_a", good_label)
    _write_prediction(pred_dir, "case_a", good_label, _AFFINE)

    with pytest.raises(ValueError):
        score_predictions(
            case_ids=["case_a"],
            pred_dir=pred_dir,
            prep_dir=prep_dir,
            target_axcodes=_TARGET_AXCODES,
            lesionwise=None,
            selftest_case="case_bad",
            out_dir=out_dir,
        )

    assert not out_dir.exists() or not (out_dir / "per_case_metrics.csv").exists()


# --------------------------------------------------------------------------- #
# Geometry gate (propagated from import_prediction)
# --------------------------------------------------------------------------- #


def test_prediction_with_wrong_affine_raises(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    pred_dir = tmp_path / "pred"
    out_dir = tmp_path / "out"

    label = _nested_cubes_label()
    _write_case(prep_dir, "case_a", label)

    wrong_affine = _AFFINE.copy()
    wrong_affine[0, 3] += 100.0  # a different grid entirely
    _write_prediction(pred_dir, "case_a", label, wrong_affine)

    with pytest.raises(ValueError):
        score_predictions(
            case_ids=["case_a"],
            pred_dir=pred_dir,
            prep_dir=prep_dir,
            target_axcodes=_TARGET_AXCODES,
            lesionwise=None,
            selftest_case="case_a",
            out_dir=out_dir,
        )


# --------------------------------------------------------------------------- #
# Hydra composition
# --------------------------------------------------------------------------- #


def test_config_block_is_reachable_and_null_pred_dir_raises() -> None:
    """The real project config must expose the block at `cfg.analysis.score_nnunet`.

    Same regression shape as `tests/test_train_qc.py`'s composed-path test:
    a hand-built fixture could pass every other test in this file while the
    real composed config never produces that shape. Also checks the
    null-forces-an-explicit-CLI-override convention `pred_dir` uses.
    """
    overrides = ["data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert "analysis" in cfg
    assert "score_nnunet" in cfg.analysis
    assert "score_nnunet" not in cfg  # NOT cfg.score_nnunet

    score_cfg = cfg.analysis.score_nnunet
    expected_keys = {
        "pred_dir",
        "prep_dir",
        "splits_path",
        "split",
        "out_dir",
        "selftest_case",
        "lesionwise",
    }
    assert expected_keys <= set(score_cfg.keys())
    assert score_cfg.pred_dir is None
    assert score_cfg.out_dir is None

    with pytest.raises(ValueError, match="pred_dir"):
        run_score_nnunet(cfg)


# --------------------------------------------------------------------------- #
# resolve_lesionwise
# --------------------------------------------------------------------------- #


def test_resolve_lesionwise_disabled_by_default() -> None:
    overrides = ["data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert resolve_lesionwise(cfg.analysis.score_nnunet) is None


def test_resolve_lesionwise_rejects_bad_matching_threshold() -> None:
    overrides = [
        "data.root_dir=/unused/for/this/test",
        "analysis.score_nnunet.lesionwise.enabled=true",
        "analysis.score_nnunet.lesionwise.matching_threshold=1.5",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    with pytest.raises(ValueError, match="matching_threshold"):
        resolve_lesionwise(cfg.analysis.score_nnunet)
