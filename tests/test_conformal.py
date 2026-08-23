"""Tests for `neurovision.uncertainty.conformal`.

CPU only, tiny hand-built/synthetic arrays, whole file well under a second.
Every expected numeric value below is hand-computed in the test itself
(never read back from the implementation) -- matching
`tests/test_calibration.py`'s convention.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    CaseLossCurve,
    band_inflation,
    case_loss_curve,
    fit_threshold,
    realised_risk,
)

# ---------------------------------------------------------------------------
# 1. DEFAULT_THRESHOLDS
# ---------------------------------------------------------------------------


def test_default_thresholds_are_sorted_unique_and_contain_half() -> None:
    arr = np.asarray(DEFAULT_THRESHOLDS, dtype=np.float64)
    assert np.all(np.diff(arr) > 0), "grid must be strictly increasing (no duplicates)"
    assert len(set(DEFAULT_THRESHOLDS)) == len(DEFAULT_THRESHOLDS)
    assert 0.5 in DEFAULT_THRESHOLDS


# ---------------------------------------------------------------------------
# 2. case_loss_curve: hand-computed
# ---------------------------------------------------------------------------


def test_case_loss_curve_hand_computed() -> None:
    # 8 voxels. target positive at indices {0, 1, 2, 3} (4 gt voxels).
    prob = np.array([0.9, 0.7, 0.4, 0.05, 0.6, 0.3, 0.02, 0.01])
    target = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    thresholds = (0.1, 0.5)

    curve = case_loss_curve(prob, target, case_id="c1", region="WT", thresholds=thresholds)

    assert curve.gt_voxels == 4
    assert curve.thresholds == thresholds

    # tau=0.1: mask = prob >= 0.1 -> indices {0,1,2,4,5} (0.9,0.7,0.4,0.6,0.3), 5 voxels.
    # gt voxels missed: gt={0,1,2,3}; masked in {0,1,2}; index 3 (0.05) missed. fn=1.
    # tau=0.5: mask = prob >= 0.5 -> indices {0,1,4} (0.9,0.7,0.6), 3 voxels.
    # gt={0,1,2,3}; masked in {0,1}; indices 2,3 missed. fn=2.
    assert curve.mask_voxels == (5, 3)
    assert curve.fn_voxels == (1, 2)


# ---------------------------------------------------------------------------
# 3. miss_rate is monotone non-decreasing along the increasing-tau grid
# ---------------------------------------------------------------------------


def test_miss_rate_is_monotone_non_increasing_in_threshold_order() -> None:
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.0, 1.0, size=(20, 20, 20))
    target = (rng.uniform(0.0, 1.0, size=(20, 20, 20)) > 0.7).astype(np.float64)
    curve = case_loss_curve(prob, target, case_id="c1", region="TC")
    miss = curve.miss_rate()
    assert np.all(np.diff(miss) >= -1e-12)


# ---------------------------------------------------------------------------
# 4. Empty ground truth
# ---------------------------------------------------------------------------


def test_empty_ground_truth_returns_nan_and_is_excluded() -> None:
    prob = np.array([0.9, 0.1, 0.5])
    target = np.array([0, 0, 0])
    thresholds = (0.2, 0.6)

    curve = case_loss_curve(prob, target, case_id="empty1", region="ET", thresholds=thresholds)
    assert curve.empty_gt is True
    assert np.all(np.isnan(curve.miss_rate()))

    nonempty = case_loss_curve(
        np.array([0.9, 0.1, 0.5]),
        np.array([1, 0, 0]),
        case_id="nonempty1",
        region="ET",
        thresholds=thresholds,
    )
    fit = fit_threshold([curve, nonempty], alpha=1.0)
    assert fit.n_excluded_empty == 1
    assert fit.n_calibration == 1


# ---------------------------------------------------------------------------
# 5. fit_threshold selection rule, hand computed
# ---------------------------------------------------------------------------


def _make_curve(
    case_id: str, region: str, thresholds: tuple, gt_voxels: int, fn: tuple
) -> CaseLossCurve:
    """Builds a CaseLossCurve directly, bypassing case_loss_curve, so risk values are exact."""
    mask_voxels = tuple(0 for _ in thresholds)  # unused by fit_threshold/realised_risk
    return CaseLossCurve(
        case_id=case_id,
        region=region,
        gt_voxels=gt_voxels,
        thresholds=thresholds,
        fn_voxels=fn,
        mask_voxels=mask_voxels,
    )


def test_fit_threshold_hand_computed_selection() -> None:
    thresholds = (0.1, 0.3, 0.5, 0.7)
    # 3 cases, gt_voxels=10 each, so miss_rate = fn/10.
    c1 = _make_curve("c1", "WT", thresholds, gt_voxels=10, fn=(0, 2, 5, 9))
    c2 = _make_curve("c2", "WT", thresholds, gt_voxels=10, fn=(0, 1, 4, 8))
    c3 = _make_curve("c3", "WT", thresholds, gt_voxels=10, fn=(1, 3, 6, 10))

    # risk_curve[t] = mean of miss rates:
    # tau=0.1: (0+0+1)/30 = 1/30 = 0.03333...
    # tau=0.3: (2+1+3)/30 = 6/30 = 0.2
    # tau=0.5: (5+4+6)/30 = 15/30 = 0.5
    # tau=0.7: (9+8+10)/30 = 27/30 = 0.9
    n = 3
    bound = 1.0
    risk_curve = np.array([1 / 30, 0.2, 0.5, 0.9])
    adjusted = (n * risk_curve + bound) / (n + 1)
    # adjusted = [(3*(1/30)+1)/4, (3*0.2+1)/4, (3*0.5+1)/4, (3*0.9+1)/4]
    # = [1.1/4, 1.6/4, 2.5/4, 3.7/4] = [0.275, 0.4, 0.625, 0.925]
    np.testing.assert_allclose(adjusted, [0.275, 0.4, 0.625, 0.925])

    # alpha=0.45: feasible at tau=0.1 (0.275) and tau=0.3 (0.4), infeasible at 0.5, 0.7.
    # largest feasible tau = 0.3.
    fit = fit_threshold([c1, c2, c3], alpha=0.45)
    assert fit.feasible is True
    assert fit.threshold == pytest.approx(0.3)
    assert fit.calibrated_risk == pytest.approx(0.2)
    np.testing.assert_allclose(fit.risk_curve, risk_curve)
    assert fit.min_achievable_risk == pytest.approx(0.275)


# ---------------------------------------------------------------------------
# 6. Infeasible case
# ---------------------------------------------------------------------------


def test_fit_threshold_infeasible_reports_min_achievable_risk() -> None:
    thresholds = (0.1, 0.5, 0.9)
    # Even at the smallest threshold, risk stays high.
    c1 = _make_curve("c1", "TC", thresholds, gt_voxels=10, fn=(5, 8, 10))
    c2 = _make_curve("c2", "TC", thresholds, gt_voxels=10, fn=(6, 9, 10))

    n = 2
    bound = 1.0
    risk_curve = np.array([(5 + 6) / 20, (8 + 9) / 20, (10 + 10) / 20])
    adjusted0 = (n * risk_curve[0] + bound) / (n + 1)  # (2*0.55+1)/3 = 2.1/3 = 0.7

    fit = fit_threshold([c1, c2], alpha=0.05)
    assert fit.feasible is False
    assert fit.threshold is None
    assert fit.calibrated_risk is None
    assert fit.min_achievable_risk == pytest.approx(float(adjusted0))
    assert fit.min_achievable_risk > 0.05


# ---------------------------------------------------------------------------
# 7. alpha=1.0 always feasible at the largest threshold
# ---------------------------------------------------------------------------


def test_fit_threshold_alpha_one_selects_largest_threshold() -> None:
    thresholds = (0.1, 0.5, 0.9)
    c1 = _make_curve("c1", "WT", thresholds, gt_voxels=10, fn=(0, 3, 9))
    c2 = _make_curve("c2", "WT", thresholds, gt_voxels=10, fn=(1, 4, 10))

    fit = fit_threshold([c1, c2], alpha=1.0)
    assert fit.feasible is True
    assert fit.threshold == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 8. Non-monotone risk curve raises
# ---------------------------------------------------------------------------


def test_fit_threshold_raises_on_non_monotone_risk() -> None:
    thresholds = (0.1, 0.5, 0.9)
    # fn decreases from tau=0.1 to tau=0.5, which is backwards (should be non-decreasing).
    c1 = _make_curve("bad", "WT", thresholds, gt_voxels=10, fn=(5, 1, 9))

    with pytest.raises(ValueError, match="monotonic"):
        fit_threshold([c1], alpha=0.5)


# ---------------------------------------------------------------------------
# 9. Mismatched grids or regions raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mismatch", ["thresholds", "region"])
def test_fit_threshold_raises_on_mismatched_grids_or_regions(mismatch: str) -> None:
    thresholds_a = (0.1, 0.5, 0.9)
    thresholds_b = (0.1, 0.4, 0.9)
    c1 = _make_curve("c1", "WT", thresholds_a, gt_voxels=10, fn=(0, 3, 9))
    if mismatch == "thresholds":
        c2 = _make_curve("c2", "WT", thresholds_b, gt_voxels=10, fn=(0, 3, 9))
    else:
        c2 = _make_curve("c2", "TC", thresholds_a, gt_voxels=10, fn=(0, 3, 9))

    with pytest.raises(ValueError):
        fit_threshold([c1, c2], alpha=0.5)


# ---------------------------------------------------------------------------
# 10. realised_risk hand computed
# ---------------------------------------------------------------------------


def test_realised_risk_matches_hand_computed_mean() -> None:
    thresholds = (0.1, 0.5, 0.9)
    c1 = _make_curve("c1", "WT", thresholds, gt_voxels=10, fn=(0, 3, 9))
    c2 = _make_curve("c2", "WT", thresholds, gt_voxels=10, fn=(1, 4, 10))
    empty = _make_curve("c3", "WT", thresholds, gt_voxels=0, fn=(0, 0, 0))

    result = realised_risk([c1, c2, empty], threshold=0.5)
    # miss_rate at tau=0.5: c1 -> 3/10=0.3, c2 -> 4/10=0.4; empty excluded.
    assert result["mean_miss_rate"] == pytest.approx((0.3 + 0.4) / 2)
    assert result["n"] == 2.0
    assert result["n_excluded_empty"] == 1.0


def test_realised_risk_raises_on_threshold_not_in_grid() -> None:
    thresholds = (0.1, 0.5, 0.9)
    c1 = _make_curve("c1", "WT", thresholds, gt_voxels=10, fn=(0, 3, 9))
    with pytest.raises(ValueError):
        realised_risk([c1], threshold=0.42)


# ---------------------------------------------------------------------------
# 12. band_inflation hand computed
# ---------------------------------------------------------------------------


def test_band_inflation_hand_computed() -> None:
    thresholds = (0.1, 0.5, 0.9)
    c1 = CaseLossCurve(
        case_id="c1",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(0, 3, 9),
        mask_voxels=(20, 10, 2),
    )
    c2 = CaseLossCurve(
        case_id="c2",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(1, 4, 10),
        mask_voxels=(30, 15, 0),
    )
    # Reference mask (tau=0.5) empty for a third case -> must be skipped, not NaN-propagated.
    c3 = CaseLossCurve(
        case_id="c3",
        region="WT",
        gt_voxels=10,
        thresholds=thresholds,
        fn_voxels=(0, 5, 10),
        mask_voxels=(5, 0, 0),
    )

    result = band_inflation([c1, c2, c3], threshold=0.1, reference_threshold=0.5)
    # c1: mask(0.1)/mask(0.5) = 20/10 = 2.0
    # c2: mask(0.1)/mask(0.5) = 30/15 = 2.0
    # c3: reference mask (tau=0.5) is 0 -> skipped
    assert result["mean_inflation"] == pytest.approx(2.0)
    assert result["median_inflation"] == pytest.approx(2.0)
    assert result["n"] == 2.0
    assert result["n_skipped"] == 1.0


# ---------------------------------------------------------------------------
# 13. torch and numpy inputs agree
# ---------------------------------------------------------------------------


def test_accepts_torch_and_numpy_inputs_identically() -> None:
    rng = np.random.default_rng(1)
    prob_np = rng.uniform(0.0, 1.0, size=(10, 10, 10)).astype(np.float32)
    target_np = (rng.uniform(0.0, 1.0, size=(10, 10, 10)) > 0.6).astype(np.float32)

    curve_np = case_loss_curve(prob_np, target_np, case_id="c", region="ET")
    curve_torch = case_loss_curve(
        torch.as_tensor(prob_np), torch.as_tensor(target_np), case_id="c", region="ET"
    )

    assert curve_np == curve_torch


# ---------------------------------------------------------------------------
# 14. End-to-end: the theorem itself, on simulated exchangeable data
# ---------------------------------------------------------------------------


def test_guarantee_holds_on_simulated_exchangeable_data() -> None:
    rng = np.random.default_rng(42)
    n_cases = 400
    n_voxels = 2000

    curves = []
    for i in range(n_cases):
        # A deliberately imperfect model: probability correlates with the label but
        # is noisy, so miss rate at any fixed threshold varies case to case.
        target = (rng.uniform(size=n_voxels) > 0.7).astype(np.float64)
        noise = rng.normal(loc=0.0, scale=0.3, size=n_voxels)
        prob = np.clip(target * 0.8 + 0.1 + noise, 0.0, 1.0)
        curves.append(case_loss_curve(prob, target, case_id=f"case_{i}", region="WT"))

    idx = rng.permutation(n_cases)
    cal_curves = [curves[i] for i in idx[: n_cases // 2]]
    eval_curves = [curves[i] for i in idx[n_cases // 2 :]]

    alpha = 0.1
    fit = fit_threshold(cal_curves, alpha=alpha)
    assert fit.feasible is True
    assert fit.threshold is not None

    realised = realised_risk(eval_curves, threshold=fit.threshold)
    # The theorem holds in expectation over the draw of (calibration, test) data; with
    # n=200 calibration cases and n=200 eval cases the realised risk should sit close to
    # (and, with high probability, at or below) alpha. Allow a small slack for sampling noise.
    assert realised["mean_miss_rate"] <= alpha + 0.03
