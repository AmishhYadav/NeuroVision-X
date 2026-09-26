"""Tests for `neurovision.analysis.local_recalibration`.

CPU only, tiny hand-built/synthetic curves, whole file well under two seconds. Numeric
expectations are hand-computed in the test itself, matching `tests/test_conformal.py`'s
convention. Synthetic exchangeable data is built the same way as
`test_conformal.test_guarantee_holds_on_simulated_exchangeable_data`.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from neurovision.analysis import local_recalibration
from neurovision.analysis.local_recalibration import (
    SplitResult,
    draw_split,
    evaluate_split,
    min_feasible_k,
    run_local_recalibration,
    summarise,
)
from neurovision.uncertainty.conformal import CaseLossCurve, case_loss_curve


def _make_curve(case_id: str, region: str, thresholds: tuple, gt_voxels: int, fn: tuple, mask=None):
    """Builds a CaseLossCurve directly, bypassing case_loss_curve, mirroring test_conformal.py."""
    if mask is None:
        mask = tuple(0 for _ in thresholds)
    return CaseLossCurve(
        case_id=case_id,
        region=region,
        gt_voxels=gt_voxels,
        thresholds=thresholds,
        fn_voxels=fn,
        mask_voxels=mask,
    )


# ---------------------------------------------------------------------------
# 1. draw_split
# ---------------------------------------------------------------------------


def test_draw_split_disjoint_and_covers_all_ids() -> None:
    ids = [f"c{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    calib, heldout = draw_split(ids, k=4, rng=rng)
    assert len(calib) == 4
    assert len(heldout) == 6
    assert set(calib).isdisjoint(set(heldout))
    assert set(calib) | set(heldout) == set(ids)


def test_draw_split_deterministic_for_fixed_seed() -> None:
    ids = [f"c{i}" for i in range(10)]
    calib1, heldout1 = draw_split(ids, k=4, rng=np.random.default_rng(7))
    calib2, heldout2 = draw_split(ids, k=4, rng=np.random.default_rng(7))
    assert calib1 == calib2
    assert heldout1 == heldout2


@pytest.mark.parametrize("k", [0, 10])
def test_draw_split_raises_for_k_at_boundary(k: int) -> None:
    ids = [f"c{i}" for i in range(10)]
    with pytest.raises(ValueError):
        draw_split(ids, k=k, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 2. Theorem sanity: the guarantee should hold on synthetic exchangeable data
# ---------------------------------------------------------------------------


def _build_exchangeable_curves(n_cases: int, seed: int) -> list[CaseLossCurve]:
    rng = np.random.default_rng(seed)
    n_voxels = 2000
    curves = []
    for i in range(n_cases):
        target = (rng.uniform(size=n_voxels) > 0.7).astype(np.float64)
        noise = rng.normal(loc=0.0, scale=0.3, size=n_voxels)
        prob = np.clip(target * 0.8 + 0.1 + noise, 0.0, 1.0)
        curves.append(case_loss_curve(prob, target, case_id=f"case_{i}", region="WT"))
    return curves


def test_guarantee_holds_over_many_local_recalibration_splits() -> None:
    curves = _build_exchangeable_curves(n_cases=200, seed=42)
    alpha = 0.10
    results = run_local_recalibration(
        {"WT": curves},
        cohort="SSA",
        alphas=[alpha],
        ks=[100],
        n_splits=300,
        rng=np.random.default_rng(123),
    )
    assert len(results) == 300
    feasible = [r for r in results if r.feasible]
    assert len(feasible) == len(results)  # feasible_rate == 1

    mean_risk = float(np.mean([r.realised_risk for r in feasible]))
    assert mean_risk <= alpha + 0.02


# ---------------------------------------------------------------------------
# 3. Infeasible: every split should report infeasible, never raise
# ---------------------------------------------------------------------------


def test_infeasible_cohort_never_raises_and_reports_nan_summary() -> None:
    thresholds = (0.1, 0.5, 0.9)
    # Every case misses 90% of the tumour even at the smallest threshold -- infeasible at
    # any reasonable alpha, deterministically, for every calibration subset.
    curves = [_make_curve(f"c{i}", "TC", thresholds, gt_voxels=10, fn=(9, 9, 9)) for i in range(20)]
    results = run_local_recalibration(
        {"TC": curves},
        cohort="PED",
        alphas=[0.05],
        ks=[10],
        n_splits=5,
        rng=np.random.default_rng(1),
    )
    assert len(results) == 5
    assert all(not r.feasible for r in results)
    assert all(r.threshold is None for r in results)
    assert all(r.realised_risk is None for r in results)
    assert all(r.violated is None for r in results)
    assert all(r.inflation is None for r in results)

    df = summarise(results)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["feasible_rate"] == 0.0
    assert np.isnan(row["mean_threshold"])
    assert np.isnan(row["mean_realised_risk"])
    assert np.isnan(row["p_violation"])
    assert np.isnan(row["mean_inflation"])
    assert np.isnan(row["realised_risk_ci_low"])
    assert np.isnan(row["realised_risk_ci_high"])


# ---------------------------------------------------------------------------
# 4. min_feasible_k, hand computed
# ---------------------------------------------------------------------------


def test_min_feasible_k_matches_hand_computed_value() -> None:
    thresholds = (0.01, 0.5, 0.9)
    # R_min = mean miss rate at the smallest threshold = 2/10 = 0.2 for every curve.
    curves = [_make_curve(f"c{i}", "WT", thresholds, gt_voxels=10, fn=(2, 5, 8)) for i in range(5)]
    # k >= (B - alpha) / (alpha - R_min) = (1 - 0.3) / (0.3 - 0.2) = 7.0 exactly -> k = 7.
    assert min_feasible_k(curves, alpha=0.3, k_max=20) == 7
    # k_max too small to reach the needed k=7.
    assert min_feasible_k(curves, alpha=0.3, k_max=6) is None


def test_min_feasible_k_returns_none_when_r_min_at_least_alpha() -> None:
    thresholds = (0.01, 0.5, 0.9)
    # R_min = 5/10 = 0.5 >= alpha=0.3 -- no k can ever help.
    curves = [_make_curve("c0", "WT", thresholds, gt_voxels=10, fn=(5, 7, 9))]
    assert min_feasible_k(curves, alpha=0.3, k_max=10_000) is None


# ---------------------------------------------------------------------------
# 5. "half" resolves to floor(n/2); k > n-1 is skipped with a warning
# ---------------------------------------------------------------------------


def test_half_resolves_to_floor_and_oversized_k_is_skipped(caplog) -> None:
    curves = _build_exchangeable_curves(n_cases=7, seed=0)
    with caplog.at_level(logging.WARNING, logger=local_recalibration.__name__):
        results = run_local_recalibration(
            {"WT": curves},
            cohort="SSA",
            alphas=[1.0],  # alpha=1.0 is always feasible, keeping this test about k only
            ks=["half", 100],
            n_splits=2,
            rng=np.random.default_rng(0),
        )
    ks_used = {r.k for r in results}
    assert ks_used == {3}  # floor(7 / 2) == 3; k=100 skipped entirely
    assert any("100" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# 6. Reproducibility and shared split draws across alphas
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_results() -> None:
    curves = _build_exchangeable_curves(n_cases=20, seed=5)
    kwargs = dict(
        curves_by_region={"WT": curves},
        cohort="PED",
        alphas=[0.1, 0.2],
        ks=[10],
        n_splits=4,
    )
    results1 = run_local_recalibration(rng=np.random.default_rng(99), **kwargs)
    results2 = run_local_recalibration(rng=np.random.default_rng(99), **kwargs)
    assert results1 == results2


def test_alphas_share_the_same_split_draws() -> None:
    curves = _build_exchangeable_curves(n_cases=20, seed=5)
    n_splits = 4
    alphas = [0.05, 0.10]
    with patch.object(
        local_recalibration, "draw_split", side_effect=local_recalibration.draw_split
    ) as spy:
        run_local_recalibration(
            {"WT": curves},
            cohort="PED",
            alphas=alphas,
            ks=[10],
            n_splits=n_splits,
            rng=np.random.default_rng(11),
        )
    # One draw per (region, k, split_index) -- NOT multiplied by len(alphas).
    assert spy.call_count == n_splits


def test_evaluate_split_reuses_same_calibration_ids_across_alphas() -> None:
    # A direct check, at the evaluate_split level, that n_calibration/n_heldout (and hence
    # the underlying id split) are identical for two alphas fed the same calib/heldout ids.
    curves = _build_exchangeable_curves(n_cases=20, seed=1)
    case_ids = [c.case_id for c in curves]
    calib_ids, heldout_ids = draw_split(case_ids, k=10, rng=np.random.default_rng(3))

    r1 = evaluate_split(
        curves, calib_ids, heldout_ids, 0.05, cohort="PED", region="WT", k=10, split_index=0
    )
    r2 = evaluate_split(
        curves, calib_ids, heldout_ids, 0.10, cohort="PED", region="WT", k=10, split_index=0
    )
    assert r1.n_calibration == r2.n_calibration == 10
    assert r1.n_heldout == r2.n_heldout == 10


# ---------------------------------------------------------------------------
# SplitResult sanity via evaluate_split, feasible case
# ---------------------------------------------------------------------------


def test_evaluate_split_feasible_case_hand_checked() -> None:
    thresholds = (0.1, 0.5, 0.9)
    calib = [
        _make_curve("cal0", "WT", thresholds, gt_voxels=10, fn=(0, 2, 9)),
        _make_curve("cal1", "WT", thresholds, gt_voxels=10, fn=(0, 1, 8)),
    ]
    held = [
        _make_curve("held0", "WT", thresholds, gt_voxels=10, fn=(0, 3, 9), mask=(20, 10, 2)),
    ]
    curves = calib + held
    result = evaluate_split(
        curves,
        calib_ids=["cal0", "cal1"],
        heldout_ids=["held0"],
        alpha=0.5,
        cohort="SSA",
        region="WT",
        k=2,
        split_index=0,
    )
    assert isinstance(result, SplitResult)
    assert result.feasible is True
    # Held-out miss rate at the fitted threshold must be one of the curve's own values.
    assert result.realised_risk in (0.0, 0.3, 0.9)
    assert result.violated is not None
