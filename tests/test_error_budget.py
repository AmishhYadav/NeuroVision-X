"""Tests for `neurovision.analysis.error_budget`.

Phase G's end-to-end error budget (`docs/research/error_budget_protocol.md`). Every
table here is small and synthetic -- never real BraTS data -- and every test runs in
well under a second on CPU. Config is a minimal `OmegaConf.create` object exposing
exactly `cfg.clinical.gatekeeper.{enabled_signals,regions}`, the same path
`tests/test_gatekeeper.py` and `tests/test_calibrate_gatekeeper.py` already use.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.analysis.error_budget import (
    CELL_NAMES,
    bootstrap_rate_ci,
    coverage_curve,
    curated_input_qc_report,
    decide_cases,
    label_usable,
    outcome_cells,
    summarise_cohort,
    taxonomy_table,
)
from neurovision.inference.gatekeeper import Thresholds
from neurovision.inference.input_qc import Severity

REGIONS = ("WT", "TC")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(regions: tuple[str, ...] = REGIONS) -> OmegaConf:
    """A minimal composed config exposing exactly `cfg.clinical.gatekeeper`."""
    return OmegaConf.create(
        {
            "clinical": {
                "gatekeeper": {
                    "enabled_signals": ["input_qc", "predicted_dice", "conformal_band"],
                    "regions": list(regions),
                    "thresholds": None,
                }
            }
        }
    )


def _thresholds(regions: tuple[str, ...] = REGIONS) -> Thresholds:
    """A hand-built `Thresholds`: predicted_dice refuse<0.5, caution<0.7; conformal_band
    caution>0.1, refuse>0.3 -- same shape as `tests/test_gatekeeper.py`'s own fixture."""
    return Thresholds(
        predicted_dice={r: (0.5, 0.7) for r in regions},
        conformal_band={r: (0.1, 0.3) for r in regions},
        ood_score=(0.1, 0.3),
        calibration_n=187,
        caution_quantile=0.10,
        refuse_quantile=0.02,
    )


def _row(case_id: str, **overrides: float) -> dict[str, object]:
    """A clean, PROCEED-shaped signal row for `case_id`, with `overrides` applied."""
    row: dict[str, object] = {
        "case_id": case_id,
        "predicted_dice_WT": 0.95,
        "predicted_dice_TC": 0.95,
        "conformal_band_WT": 0.05,
        "conformal_band_TC": 0.05,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# 1. curated_input_qc_report + decide_cases on a clean case
# ---------------------------------------------------------------------------


def test_curated_report_is_ok_pass() -> None:
    report = curated_input_qc_report()
    assert report.verdict is Severity.OK
    assert report.findings == ()

    signals = pd.DataFrame([_row("case_0")])
    decided = decide_cases(_cfg(), signals, _thresholds(), regions=REGIONS)

    assert decided.loc[0, "decision"] == "proceed"
    assert bool(decided.loc[0, "accepted"]) is True
    assert decided.loc[0, "refusing_signals"] == ""
    assert decided.loc[0, "cautioning_signals"] == ""


# ---------------------------------------------------------------------------
# 2. Refuses on low predicted_dice
# ---------------------------------------------------------------------------


def test_decide_cases_refuses_low_predicted_dice() -> None:
    signals = pd.DataFrame([_row("case_0", predicted_dice_TC=0.3)])
    decided = decide_cases(_cfg(), signals, _thresholds(), regions=REGIONS)

    assert decided.loc[0, "decision"] == "refuse"
    assert bool(decided.loc[0, "accepted"]) is False
    assert decided.loc[0, "refusing_signals"] == "predicted_dice"


# ---------------------------------------------------------------------------
# 3. Cautions on a conformal band between the caution and refuse cuts
# ---------------------------------------------------------------------------


def test_decide_cases_cautions_on_band() -> None:
    signals = pd.DataFrame([_row("case_0", conformal_band_TC=0.2)])  # 0.1 < 0.2 < 0.3
    decided = decide_cases(_cfg(), signals, _thresholds(), regions=REGIONS)

    assert decided.loc[0, "decision"] == "proceed_with_caution"
    assert "conformal_band" in decided.loc[0, "cautioning_signals"].split(";")
    assert decided.loc[0, "refusing_signals"] == ""


# ---------------------------------------------------------------------------
# 4. label_usable
# ---------------------------------------------------------------------------


def test_label_usable_requires_all_regions_and_treats_nan_as_unusable() -> None:
    metrics = pd.DataFrame(
        {
            "dice_WT": [0.9, 0.9, float("nan"), 0.8],
            "dice_TC": [0.9, 0.5, 0.9, 0.75],
        }
    )
    usable = label_usable(metrics, regions=("WT", "TC"), bar=0.7)

    assert list(usable) == [True, False, False, True]

    with pytest.raises(ValueError, match="dice_TC"):
        label_usable(metrics[["dice_WT"]], regions=("WT", "TC"), bar=0.7)


# ---------------------------------------------------------------------------
# 5. outcome_cells
# ---------------------------------------------------------------------------


def test_outcome_cells_four_way() -> None:
    accepted = pd.Series([True, True, False, False])
    usable = pd.Series([True, False, True, False])

    cells = outcome_cells(accepted, usable)

    # correct_accept, silent_failure, over_refusal, correct_refusal, in that order.
    assert list(cells) == list(CELL_NAMES)

    mismatched = pd.Series([True, False, True, False], index=[0, 1, 2, 99])
    with pytest.raises(ValueError, match="index"):
        outcome_cells(accepted, mismatched)


# ---------------------------------------------------------------------------
# 6. bootstrap_rate_ci degenerate cases
# ---------------------------------------------------------------------------


def test_bootstrap_rate_ci_degenerate() -> None:
    ones = np.ones(5)
    point, lo, hi = bootstrap_rate_ci(ones, n_boot=200, ci=0.95, generator=np.random.default_rng(0))
    assert (point, lo, hi) == (1.0, 1.0, 1.0)

    empty = np.array([])
    point, lo, hi = bootstrap_rate_ci(
        empty, n_boot=200, ci=0.95, generator=np.random.default_rng(0)
    )
    assert math.isnan(point) and math.isnan(lo) and math.isnan(hi)

    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    r1 = bootstrap_rate_ci(values, n_boot=300, ci=0.95, generator=np.random.default_rng(42))
    r2 = bootstrap_rate_ci(values, n_boot=300, ci=0.95, generator=np.random.default_rng(42))
    assert r1 == r2  # seeded reproducibility


# ---------------------------------------------------------------------------
# 7-8. summarise_cohort
# ---------------------------------------------------------------------------


def _cohort_table() -> pd.DataFrame:
    """8 cases: 2 per outcome cell, plus dice and a miss_rate_WT column."""
    return pd.DataFrame(
        {
            "accepted": [True, True, True, True, False, False, False, False],
            "usable": [True, True, False, False, True, True, False, False],
            "dice_WT": [0.9] * 8,
            "dice_TC": [0.8] * 8,
            "dice_ET": [0.7] * 8,
            "miss_rate_WT": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_summarise_cohort_hand_computed() -> None:
    per_case = _cohort_table()
    generator = np.random.default_rng(0)

    summary = summarise_cohort(
        per_case, cohort="test_cohort", bar=0.7, n_boot=50, ci=0.95, generator=generator, alpha=0.1
    )

    assert summary["cohort"] == "test_cohort"
    assert summary["bar"] == 0.7
    assert summary["n"] == 8
    assert summary["p_accepted"] == pytest.approx(0.5)
    assert summary["p_usable"] == pytest.approx(0.5)
    assert summary["p_accepted_and_usable"] == pytest.approx(0.25)
    assert summary["p_usable_given_accepted"] == pytest.approx(0.5)

    assert summary["n_correct_accept"] == 2
    assert summary["n_silent_failure"] == 2
    assert summary["n_over_refusal"] == 2
    assert summary["n_correct_refusal"] == 2

    assert summary["dice_WT_all_mean"] == pytest.approx(0.9)
    assert summary["dice_TC_all_mean"] == pytest.approx(0.8)
    assert summary["dice_ET_all_mean"] == pytest.approx(0.7)
    assert summary["dice_TC_accepted_mean"] == pytest.approx(0.8)

    # miss_rate_WT: accepted rows are all 0.0, refused rows are all 1.0.
    assert summary["realised_risk_WT_all"] == pytest.approx(0.5)
    assert summary["realised_risk_WT_accepted"] == pytest.approx(0.0)
    assert summary["frac_case_risk_le_alpha_WT_all"] == pytest.approx(0.5)
    assert summary["frac_case_risk_le_alpha_WT_accepted"] == pytest.approx(1.0)

    # Bootstrap CIs must bracket their own point estimate.
    for prefix in ("p_accepted", "p_usable", "p_accepted_and_usable", "p_usable_given_accepted"):
        assert summary[f"{prefix}_lo"] <= summary[prefix] <= summary[f"{prefix}_hi"]

    # alpha=None must omit the frac_case_risk_le_alpha_* keys entirely.
    summary_no_alpha = summarise_cohort(
        per_case, cohort="test_cohort", bar=0.7, n_boot=50, ci=0.95, generator=generator
    )
    assert "frac_case_risk_le_alpha_WT_all" not in summary_no_alpha
    assert "realised_risk_WT_all" in summary_no_alpha  # always emitted, alpha or not


def test_summarise_cohort_nothing_accepted() -> None:
    per_case = pd.DataFrame(
        {
            "accepted": [False, False, False],
            "usable": [True, False, True],
            "dice_WT": [0.9, 0.5, 0.95],
            "dice_TC": [0.9, 0.5, 0.95],
            "dice_ET": [0.9, 0.5, 0.95],
        }
    )
    summary = summarise_cohort(
        per_case, cohort="empty", bar=0.7, n_boot=50, ci=0.95, generator=np.random.default_rng(0)
    )
    assert math.isnan(summary["p_usable_given_accepted"])
    assert math.isnan(summary["p_usable_given_accepted_lo"])
    assert math.isnan(summary["p_usable_given_accepted_hi"])
    assert summary["n_correct_accept"] == 0
    assert summary["p_accepted"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. coverage_curve
# ---------------------------------------------------------------------------


def test_coverage_curve_monotone_coverage() -> None:
    # 101 evenly spaced calibration values -> quantile(q) == q exactly, the same
    # construction tests/test_gatekeeper.py::test_calibrate_thresholds_quantiles uses.
    base = np.arange(101) / 100.0
    calibration_table = pd.DataFrame(
        {
            "predicted_dice_WT": base,
            "predicted_dice_TC": base,
            "conformal_band_WT": base,
            "conformal_band_TC": base,
            "ood_score": base,
        }
    )

    # One cohort case per calibration point. conformal_band held at 0.0 (always
    # well inside the safe side) so only predicted_dice drives refusals -- isolates
    # the variable this test is about.
    case_ids = [f"c{i}" for i in range(101)]
    signals = pd.DataFrame(
        {
            "case_id": case_ids,
            "predicted_dice_WT": base,
            "predicted_dice_TC": base,
            "conformal_band_WT": 0.0,
            "conformal_band_TC": 0.0,
        }
    )
    usable = pd.Series(True, index=case_ids)
    dice_tc = pd.Series(base, index=case_ids)

    # Includes quantiles below, at, and well past the deployed caution_quantile
    # (0.10) -- coverage_curve nudges caution_quantile_used past q in the latter
    # cases (see its docstring) purely so calibrate_thresholds' fit succeeds; that
    # nudge must never change coverage or any other reported number.
    refuse_quantiles = [0.01, 0.05, 0.09, 0.10, 0.30, 0.50]
    curve = coverage_curve(
        _cfg(),
        calibration_table,
        {"cohortA": signals},
        {"cohortA": usable},
        {"cohortA": dice_tc},
        regions=REGIONS,
        refuse_quantiles=refuse_quantiles,
        caution_quantile=0.10,
    )

    curve = curve.sort_values("refuse_quantile").reset_index(drop=True)
    coverages = curve["coverage"].to_numpy()

    # Non-increasing coverage as the refuse quantile rises.
    assert all(coverages[i] >= coverages[i + 1] - 1e-9 for i in range(len(coverages) - 1))
    # And strictly fewer accepted at the highest quantile than at the lowest.
    assert curve.loc[curve["refuse_quantile"] == 0.50, "n_accepted"].iloc[0] < (
        curve.loc[curve["refuse_quantile"] == 0.01, "n_accepted"].iloc[0]
    )
    assert set(curve["n"]) == {101}

    # q >= caution_quantile: no exception (this used to raise), and
    # caution_quantile_used is nudged to exactly q + 0.01, bookkeeping only.
    for q in (0.10, 0.30, 0.50):
        row = curve.loc[curve["refuse_quantile"] == q].iloc[0]
        assert row["caution_quantile_used"] == pytest.approx(q + 0.01)


# ---------------------------------------------------------------------------
# 10. taxonomy_table
# ---------------------------------------------------------------------------


def test_taxonomy_table_has_every_cell() -> None:
    cohort_a = pd.DataFrame(
        {
            "cell": ["correct_accept", "correct_accept", "silent_failure", "correct_refusal"],
            "refusing_signals": ["", "", "", "predicted_dice"],
        }
    )
    # cohort_b has NO over_refusal or correct_refusal cases at all.
    cohort_b = pd.DataFrame(
        {
            "cell": ["correct_accept", "silent_failure"],
            "refusing_signals": ["", ""],
        }
    )

    table = taxonomy_table({"cohort_a": cohort_a, "cohort_b": cohort_b})

    for cohort, n_total in (("cohort_a", 4), ("cohort_b", 2)):
        subset = table[table["cohort"] == cohort]
        assert set(CELL_NAMES) <= set(subset["cell"])
        assert subset["frac_of_cohort"].sum() == pytest.approx(1.0)
        # every present row's n sums back to n_total
        assert subset["n"].sum() == n_total

    # cohort_b's missing cells appear at n=0 with refusing_signals=''.
    b_over_refusal = table[(table["cohort"] == "cohort_b") & (table["cell"] == "over_refusal")]
    assert len(b_over_refusal) == 1
    assert b_over_refusal["n"].iloc[0] == 0
    assert b_over_refusal["refusing_signals"].iloc[0] == ""
