"""Tests for `neurovision.analysis.gatekeeper_calibration`.

CPU only, synthetic, no real checkpoint or BraTS data, whole file well under a few
seconds -- same spirit as `tests/test_qc_validate.py`. `build_gatekeeper_calibration_table`'s
join behaviour is tested through its factored-out private helper,
`_inner_join_calibration_tables`, so no real SegQC checkpoint or conformal artifact is
needed for that test either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis.detection import case_entropy_scalars
from neurovision.analysis.gatekeeper_calibration import (
    _inner_join_calibration_tables,
    case_conformal_band_widths,
    ood_score_table,
    resolve_fitted_thresholds,
)
from neurovision.uncertainty.conformal import CaseLossCurve, load_curves_npz

# ---------------------------------------------------------------------------
# resolve_fitted_thresholds
# ---------------------------------------------------------------------------


def test_resolve_fitted_thresholds_finds_default_float_formatted_key() -> None:
    # alpha=0.1 as a plain float f-strings to "0.1", not "0.10" -- the exact trap
    # described in the module's spec. A fixed-decimals lookup would miss this key.
    fit_payload = {
        "WT__alpha_0.1": {"threshold": 0.725, "feasible": True},
        "TC__alpha_0.1": {"threshold": 0.6, "feasible": True},
    }
    result = resolve_fitted_thresholds(fit_payload, ["WT", "TC"], alpha=0.1)
    assert result == {"WT": 0.725, "TC": 0.6}


def test_resolve_fitted_thresholds_raises_naming_missing_key() -> None:
    fit_payload = {"WT__alpha_0.1": {"threshold": 0.725, "feasible": True}}
    with pytest.raises(ValueError, match="TC__alpha_0.1"):
        resolve_fitted_thresholds(fit_payload, ["WT", "TC"], alpha=0.1)


def test_resolve_fitted_thresholds_raises_when_threshold_is_none() -> None:
    fit_payload = {"WT__alpha_0.05": {"threshold": None, "feasible": False}}
    with pytest.raises(ValueError, match="infeasible"):
        resolve_fitted_thresholds(fit_payload, ["WT"], alpha=0.05)


# ---------------------------------------------------------------------------
# case_conformal_band_widths
# ---------------------------------------------------------------------------


def test_case_conformal_band_widths_hand_computed() -> None:
    thresholds = (0.1, 0.5, 0.9)
    # case_a: mask(0.1)=20, mask(0.5)=10 -> inflation at tau=0.1 relative to 0.5 is 2.0.
    case_a = CaseLossCurve(
        case_id="case_a",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(0, 3, 9),
        mask_voxels=(20, 10, 2),
    )
    # case_b: mask(0.1)=30, mask(0.5)=15 -> inflation at tau=0.1 relative to 0.5 is 2.0.
    case_b = CaseLossCurve(
        case_id="case_b",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(1, 4, 10),
        mask_voxels=(30, 15, 3),
    )
    # A second region, different fitted threshold (0.9), different case set.
    case_c = CaseLossCurve(
        case_id="case_c",
        region="TC",
        gt_voxels=8,
        thresholds=thresholds,
        fn_voxels=(0, 2, 6),
        mask_voxels=(40, 20, 5),
    )

    curves_by_region = {"WT": [case_a, case_b], "TC": [case_c]}
    fitted_thresholds = {"WT": 0.1, "TC": 0.9}

    result = case_conformal_band_widths(curves_by_region, fitted_thresholds)

    assert result["WT"]["case_a"] == pytest.approx(20 / 10)
    assert result["WT"]["case_b"] == pytest.approx(30 / 15)
    # case_c at tau=0.9 relative to reference tau=0.5: mask(0.9)/mask(0.5) = 5/20.
    assert result["TC"]["case_c"] == pytest.approx(5 / 20)


def test_case_conformal_band_widths_raises_on_out_of_grid_threshold() -> None:
    thresholds = (0.1, 0.5, 0.9)
    curve = CaseLossCurve(
        case_id="c1",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(0, 3, 9),
        mask_voxels=(20, 10, 2),
    )
    with pytest.raises(ValueError):
        case_conformal_band_widths({"WT": [curve]}, {"WT": 0.42})


# ---------------------------------------------------------------------------
# conformal.load_curves_npz round-trip
# ---------------------------------------------------------------------------


def test_load_curves_npz_round_trips_write_curves_npz_layout(tmp_path: Path) -> None:
    # Builds the .npz in EXACTLY scripts/conformal.py::_write_curves_npz's layout,
    # by hand, rather than importing that private script function.
    thresholds = (0.1, 0.5, 0.9)
    payload: dict[str, np.ndarray] = {"thresholds": np.asarray(thresholds, dtype=np.float64)}

    case_ids = ["case_a", "case_b"]
    gt_voxels = [10, 20]
    fn_voxels = [(0, 3, 9), (1, 4, 10)]
    mask_voxels = [(20, 10, 2), (30, 15, 3)]

    payload["WT__case_ids"] = np.array(case_ids)
    payload["WT__gt_voxels"] = np.array(gt_voxels, dtype=np.int64)
    payload["WT__fn_voxels"] = np.array(fn_voxels, dtype=np.int64)
    payload["WT__mask_voxels"] = np.array(mask_voxels, dtype=np.int64)

    npz_path = tmp_path / "curves.npz"
    np.savez_compressed(npz_path, **payload)

    loaded = load_curves_npz(npz_path, ["WT"])

    assert set(loaded.keys()) == {"WT"}
    assert len(loaded["WT"]) == 2
    for i, curve in enumerate(loaded["WT"]):
        assert curve.case_id == case_ids[i]
        assert curve.region == "WT"
        assert curve.gt_voxels == gt_voxels[i]
        assert curve.thresholds == thresholds
        assert curve.fn_voxels == fn_voxels[i]
        assert curve.mask_voxels == mask_voxels[i]


# ---------------------------------------------------------------------------
# ood_score_table
# ---------------------------------------------------------------------------


def test_ood_score_table_matches_case_entropy_scalars_directly(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_val"
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 4, 4, 4)).astype(np.float32)
    np.save(logits_dir / "case_1.npy", logits)

    table = ood_score_table(eval_dir)

    expected = case_entropy_scalars(logits)["ent_mean_fg_mean"]
    assert list(table["case_id"]) == ["case_1"]
    assert table["ood_score"].iloc[0] == pytest.approx(expected)


def test_ood_score_table_respects_explicit_case_ids(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval_val"
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True)

    rng = np.random.default_rng(1)
    for name in ("case_1", "case_2"):
        np.save(logits_dir / f"{name}.npy", rng.normal(size=(3, 4, 4, 4)).astype(np.float32))

    table = ood_score_table(eval_dir, case_ids=["case_2"])
    assert list(table["case_id"]) == ["case_2"]


# ---------------------------------------------------------------------------
# _inner_join_calibration_tables (build_gatekeeper_calibration_table's join logic,
# tested via its factored-out private helper -- no real SegQC checkpoint or
# conformal artifact needed)
# ---------------------------------------------------------------------------


def test_inner_join_drops_cases_missing_from_any_table(caplog: pytest.LogCaptureFixture) -> None:
    dice_table = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "predicted_dice_WT": [0.9, 0.8, 0.7],
        }
    )
    # "b" is missing from the conformal table.
    conformal_table = pd.DataFrame(
        {
            "case_id": ["a", "c"],
            "conformal_band_WT": [1.1, 1.3],
        }
    )
    # "c" is missing from the ood table.
    ood_table = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "ood_score": [0.1, 0.2],
        }
    )

    with caplog.at_level("WARNING"):
        merged = _inner_join_calibration_tables(dice_table, conformal_table, ood_table)

    # Only "a" survives all three tables.
    assert list(merged["case_id"]) == ["a"]
    assert any("dropped" in record.message for record in caplog.records)


def test_inner_join_raises_when_nothing_survives() -> None:
    dice_table = pd.DataFrame({"case_id": ["a"], "predicted_dice_WT": [0.9]})
    conformal_table = pd.DataFrame({"case_id": ["b"], "conformal_band_WT": [1.1]})
    ood_table = pd.DataFrame({"case_id": ["c"], "ood_score": [0.1]})

    with pytest.raises(ValueError, match="no case_id"):
        _inner_join_calibration_tables(dice_table, conformal_table, ood_table)


def test_inner_join_keeps_every_case_when_all_tables_agree() -> None:
    dice_table = pd.DataFrame({"case_id": ["a", "b"], "predicted_dice_WT": [0.9, 0.8]})
    conformal_table = pd.DataFrame({"case_id": ["a", "b"], "conformal_band_WT": [1.1, 1.2]})
    ood_table = pd.DataFrame({"case_id": ["a", "b"], "ood_score": [0.1, 0.2]})

    merged = _inner_join_calibration_tables(dice_table, conformal_table, ood_table)
    assert sorted(merged["case_id"]) == ["a", "b"]
    assert len(merged) == 2
