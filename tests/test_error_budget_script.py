"""Tests for scripts/error_budget.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, the same pattern
`tests/test_calibrate_gatekeeper.py` and `tests/test_validate_qc_script.py` already
use for their own sibling scripts.

These tests only exercise the three pieces of plumbing that do not need real
artifacts (a saved QC checkpoint, saved logits, a conformal `curves.npz`, a frozen
`thresholds.json`) -- a full run against `outputs/` is a real analysis run, done
separately, never inside the test suite (`CLAUDE.md`'s testing rules; a run against
real BraTS-derived artifacts belongs to Phase G's own note in `docs/experiments.md`,
not to this file). Everything here is tiny, synthetic and CPU-only, well under a
second.
"""

from __future__ import annotations

import math

import pytest

from neurovision.uncertainty.conformal import CaseLossCurve
from tests.script_loader import load_script

error_budget_script = load_script("error_budget")

_miss_rate_at_threshold = error_budget_script._miss_rate_at_threshold
_stage_reliability_rows = error_budget_script._stage_reliability_rows
_ensure_bar_included = error_budget_script._ensure_bar_included


# ---------------------------------------------------------------------------
# 1. _miss_rate_at_threshold
# ---------------------------------------------------------------------------


def _curve(gt_voxels: int) -> CaseLossCurve:
    """A hand-built 3-point CaseLossCurve; fn_voxels chosen so miss_rate is easy by hand."""
    return CaseLossCurve(
        case_id="case_0",
        region="WT",
        gt_voxels=gt_voxels,
        thresholds=(0.1, 0.2, 0.3),
        fn_voxels=(50, 20, 5),
        mask_voxels=(80, 60, 40),
    )


def test_miss_rate_at_threshold_matches_hand_computed_value() -> None:
    curve = _curve(gt_voxels=100)
    # fn_voxels[1] / gt_voxels = 20 / 100 = 0.2, at threshold 0.2 (index 1).
    value = _miss_rate_at_threshold(curve, 0.2)
    assert value == pytest.approx(0.2)

    # Same check at the first and last grid points.
    assert _miss_rate_at_threshold(curve, 0.1) == pytest.approx(0.5)
    assert _miss_rate_at_threshold(curve, 0.3) == pytest.approx(0.05)


def test_miss_rate_at_threshold_nan_on_empty_gt() -> None:
    curve = _curve(gt_voxels=0)
    assert curve.empty_gt is True
    value = _miss_rate_at_threshold(curve, 0.2)
    assert math.isnan(value)


def test_miss_rate_at_threshold_raises_off_grid() -> None:
    curve = _curve(gt_voxels=100)
    with pytest.raises(ValueError, match="not in the threshold grid"):
        _miss_rate_at_threshold(curve, 0.15)


def test_miss_rate_at_threshold_tolerates_floating_point_noise() -> None:
    curve = _curve(gt_voxels=100)
    # Within the default 1e-9 tolerance -- still resolves to the 0.2 grid point.
    value = _miss_rate_at_threshold(curve, 0.2 + 1e-10)
    assert value == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2. _stage_reliability_rows
# ---------------------------------------------------------------------------


def _summary_row() -> dict[str, object]:
    """A hand-built summarise_cohort-shaped row for cohort 'test' at bar=0.7."""
    return {
        "cohort": "test",
        "bar": 0.7,
        "n": 10,
        "p_accepted": 0.6,
        "p_accepted_lo": 0.4,
        "p_accepted_hi": 0.8,
        "p_usable": 0.7,
        "p_usable_lo": 0.5,
        "p_usable_hi": 0.9,
        "p_usable_given_accepted": 0.8,
        "p_usable_given_accepted_lo": 0.6,
        "p_usable_given_accepted_hi": 0.95,
        "p_accepted_and_usable": 0.5,
        "p_accepted_and_usable_lo": 0.3,
        "p_accepted_and_usable_hi": 0.7,
        "n_correct_accept": 5,
        "n_silent_failure": 1,
        "n_over_refusal": 2,
        "n_correct_refusal": 2,
        "realised_risk_WT_all": 0.05,
        "realised_risk_WT_all_lo": 0.01,
        "realised_risk_WT_all_hi": 0.09,
        "realised_risk_WT_accepted": 0.02,
        "realised_risk_WT_accepted_lo": 0.0,
        "realised_risk_WT_accepted_hi": 0.04,
        "realised_risk_TC_all": 0.07,
        "realised_risk_TC_all_lo": 0.03,
        "realised_risk_TC_all_hi": 0.11,
    }


def test_stage_reliability_rows_every_stage_present() -> None:
    rows = _stage_reliability_rows(_summary_row(), regions=("WT", "TC"), alpha=0.1)

    stages = {row["stage"] for row in rows}
    assert stages == {"segmentation", "conformal", "gate", "end_to_end", "input_qc"}

    for row in rows:
        assert row["cohort"] == "test"
        assert row["n"] == 10


def test_stage_reliability_rows_alpha_row_present() -> None:
    rows = _stage_reliability_rows(_summary_row(), regions=("WT", "TC"), alpha=0.1)

    alpha_rows = [r for r in rows if r["stage"] == "conformal" and r["quantity"] == "alpha"]
    assert len(alpha_rows) == 1
    assert alpha_rows[0]["value"] == pytest.approx(0.1)
    assert math.isnan(alpha_rows[0]["lo"])
    assert math.isnan(alpha_rows[0]["hi"])


def test_stage_reliability_rows_conformal_regions_present_with_ci() -> None:
    rows = _stage_reliability_rows(_summary_row(), regions=("WT", "TC"), alpha=0.1)

    wt_row = next(
        r for r in rows if r["stage"] == "conformal" and r["quantity"] == "realised_risk_WT_all"
    )
    assert wt_row["value"] == pytest.approx(0.05)
    assert wt_row["lo"] == pytest.approx(0.01)
    assert wt_row["hi"] == pytest.approx(0.09)


def test_stage_reliability_rows_ci_nan_where_absent() -> None:
    rows = _stage_reliability_rows(_summary_row(), regions=("WT", "TC"), alpha=0.1)

    input_qc_rows = [r for r in rows if r["stage"] == "input_qc"]
    assert len(input_qc_rows) == 1
    assert input_qc_rows[0]["quantity"] == "not_applicable_curated_cohort"
    assert math.isnan(input_qc_rows[0]["value"])
    assert math.isnan(input_qc_rows[0]["lo"])
    assert math.isnan(input_qc_rows[0]["hi"])

    gate_row = next(r for r in rows if r["quantity"] == "p_accepted")
    assert gate_row["lo"] == pytest.approx(0.4)
    assert gate_row["hi"] == pytest.approx(0.8)


def test_stage_reliability_rows_missing_region_skipped_not_fabricated() -> None:
    # summary_row has no realised_risk_ET_* keys at all -- ET must simply not
    # appear, never show up as a fabricated NaN row.
    rows = _stage_reliability_rows(_summary_row(), regions=("WT", "TC", "ET"), alpha=0.1)
    quantities = {r["quantity"] for r in rows if r["stage"] == "conformal"}
    assert "realised_risk_ET_all" not in quantities


# ---------------------------------------------------------------------------
# 3. _ensure_bar_included
# ---------------------------------------------------------------------------


def test_ensure_bar_included_appends_missing_bar() -> None:
    result = _ensure_bar_included([0.5, 0.6, 0.8, 0.9], 0.7)
    assert result == [0.5, 0.6, 0.7, 0.8, 0.9]


def test_ensure_bar_included_no_duplicate_when_already_present() -> None:
    result = _ensure_bar_included([0.5, 0.6, 0.7, 0.8, 0.9], 0.7)
    assert result == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert result.count(0.7) == 1


def test_ensure_bar_included_sorts_unsorted_input() -> None:
    result = _ensure_bar_included([0.9, 0.5, 0.7], 0.6)
    assert result == [0.5, 0.6, 0.7, 0.9]
