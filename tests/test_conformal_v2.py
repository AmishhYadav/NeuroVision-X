"""Tests for the v2 additions to `neurovision.uncertainty.conformal`.

CPU only, tiny synthetic curves. Expected values for the hand-checked cases are computed in
the test comments, not read back from the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    EXTENDED_THRESHOLDS,
    CaseLossCurve,
    fit_threshold,
    fit_threshold_crc,
    fit_threshold_rcps,
    mask_quality,
    realised_risk,
    realised_risk_with,
    wsr_upper_bound,
)

GRID3 = (0.1, 0.5, 0.9)


def _curve(case_id: str, gt: int, fn: tuple[int, ...], mask: tuple[int, ...], grid=GRID3):
    return CaseLossCurve(case_id, "WT", gt, grid, fn, mask)


def _random_curves(n: int, seed: int, empty_frac: float = 0.0, grid=GRID3, spread: float = 1.0):
    """Random curves whose miss rate is non-decreasing in tau (as it must be)."""
    rng = np.random.default_rng(seed)
    curves = []
    for i in range(n):
        if rng.uniform() < empty_frac:
            gt = 0
        else:
            gt = int(rng.integers(50, 200))
        frac = np.sort(rng.uniform(0, 1, len(grid))) * rng.uniform(0, spread)
        fn = tuple(int(x) for x in np.floor(frac * gt))
        mask = tuple(int(x) for x in np.sort(rng.integers(0, 300, len(grid)))[::-1])
        curves.append(_curve(f"c{i:03d}", gt, fn, mask, grid))
    return curves


# --- grid -------------------------------------------------------------------


def test_extended_grid() -> None:
    arr = np.asarray(EXTENDED_THRESHOLDS)
    assert set(DEFAULT_THRESHOLDS) <= set(EXTENDED_THRESHOLDS)
    assert np.all(np.diff(arr) > 0)
    assert 0.9999 in EXTENDED_THRESHOLDS
    assert len(EXTENDED_THRESHOLDS) == len(DEFAULT_THRESHOLDS) + 8


# --- empty conventions ------------------------------------------------------


def test_miss_rate_with() -> None:
    empty = _curve("e", 0, (0, 0, 0), (0, 0, 0))
    full = _curve("f", 10, (0, 5, 10), (10, 5, 0))
    assert np.all(np.isnan(empty.miss_rate_with("exclude")))
    assert np.array_equal(empty.miss_rate_with("zero"), np.zeros(3))
    assert np.array_equal(full.miss_rate_with("exclude"), full.miss_rate())
    assert np.array_equal(full.miss_rate_with("zero"), full.miss_rate())
    with pytest.raises(ValueError):
        full.miss_rate_with("one")


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_crc_exclude_equals_fit_threshold(alpha: float) -> None:
    curves = _random_curves(20, seed=1, empty_frac=0.2, spread=0.3)
    assert fit_threshold_crc(curves, alpha, empty="exclude") == fit_threshold(curves, alpha)


def test_crc_zero_changes_threshold() -> None:
    # Three non-empty cases, each miss rate (0, 0.1, 0.5) at taus (0.1, 0.5, 0.9); one empty case.
    real = [_curve(f"r{i}", 10, (0, 1, 5), (10, 9, 5)) for i in range(3)]
    curves = real + [_curve("e", 0, (0, 0, 0), (0, 0, 0))]
    # exclude: n=3, adjusted = (3R+1)/4 = 0.25, 0.325, 0.625 -> alpha 0.3 gives tau 0.1
    ex = fit_threshold_crc(curves, 0.3, empty="exclude")
    assert ex.threshold == 0.1 and ex.n_calibration == 3 and ex.n_excluded_empty == 1
    # zero: n=4, R = 0.75 * (0, 0.1, 0.5) -> adjusted (4R+1)/5 = 0.2, 0.26, 0.6 -> tau 0.5
    zr = fit_threshold_crc(curves, 0.3, empty="zero")
    assert zr.threshold == 0.5 and zr.n_calibration == 4 and zr.n_excluded_empty == 0
    assert zr.calibrated_risk == pytest.approx(0.075)


def test_realised_risk_with() -> None:
    curves = [_curve("a", 10, (0, 2, 4), (1, 1, 1)), _curve("e", 0, (0, 0, 0), (0, 0, 0))]
    assert realised_risk_with(curves, 0.5) == realised_risk(curves, 0.5)
    z = realised_risk_with(curves, 0.5, empty="zero")
    assert z == {"mean_miss_rate": pytest.approx(0.1), "n": 2.0, "n_excluded_empty": 0.0}


# --- WSR bound --------------------------------------------------------------


def test_wsr_all_zero_and_all_one() -> None:
    ucb0 = wsr_upper_bound(np.zeros(200), 0.1)
    assert 0.0 <= ucb0 < 0.05
    assert wsr_upper_bound(np.ones(50), 0.1) == 1.0


def test_wsr_is_upper_bound_and_shrinks_with_n() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.uniform(0, 1, 40)
        assert wsr_upper_bound(x, 0.1) >= x.mean()
    small = np.mean([wsr_upper_bound((rng.uniform(size=50) < 0.2) * 1.0, 0.1) for _ in range(20)])
    large = np.mean([wsr_upper_bound((rng.uniform(size=400) < 0.2) * 1.0, 0.1) for _ in range(20)])
    assert large < small


def test_wsr_coverage_simulation() -> None:
    rng = np.random.default_rng(1)
    delta = 0.1
    fail = sum(
        wsr_upper_bound((rng.uniform(size=100) < 0.3) * 1.0, delta) < 0.3 for _ in range(300)
    )
    assert fail / 300 <= delta + 0.05


def test_wsr_validation() -> None:
    with pytest.raises(ValueError):
        wsr_upper_bound(np.array([]), 0.1)
    with pytest.raises(ValueError):
        wsr_upper_bound(np.array([0.5, 1.5]), 0.1)
    with pytest.raises(ValueError):
        wsr_upper_bound(np.array([0.5, np.nan]), 0.1)
    with pytest.raises(ValueError):
        wsr_upper_bound(np.array([0.5]), 1.0)


# --- RCPS -------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.3])
def test_rcps_never_less_conservative_than_crc(alpha: float) -> None:
    curves = _random_curves(150, seed=3, spread=0.5, grid=(0.1, 0.3, 0.5, 0.7, 0.9))
    crc = fit_threshold_crc(curves, alpha)
    rcps = fit_threshold_rcps(curves, alpha, 0.1, order_seed=42)
    if rcps.feasible:
        assert crc.feasible and rcps.threshold <= crc.threshold
    assert rcps.n_calibration == 150 and rcps.order_seed == 42


def test_rcps_order_independent_and_seeded() -> None:
    curves = _random_curves(60, seed=4, spread=0.3)
    a = fit_threshold_rcps(curves, 0.3, 0.1, order_seed=42)
    b = fit_threshold_rcps(list(reversed(curves)), 0.3, 0.1, order_seed=42)
    assert a == b


def test_rcps_infeasible() -> None:
    curves = [_curve(f"c{i}", 10, (10, 10, 10), (0, 0, 0)) for i in range(20)]
    fit = fit_threshold_rcps(curves, 0.1, 0.1, order_seed=42)
    assert not fit.feasible and fit.threshold is None
    assert fit.ucb_at_threshold is None and fit.calibrated_risk is None
    assert fit.min_ucb == 1.0


def test_rcps_feasible_fields() -> None:
    curves = [_curve(f"c{i}", 10, (0, 0, 10), (10, 10, 0)) for i in range(200)]
    fit = fit_threshold_rcps(curves, 0.1, 0.1, order_seed=0)
    # losses are exactly 0 at taus 0.1 and 0.5 and exactly 1 at 0.9
    assert fit.feasible and fit.threshold == 0.5
    assert fit.calibrated_risk == 0.0 and fit.ucb_at_threshold < 0.1


# --- mask_quality -----------------------------------------------------------


def test_mask_quality_hand_computed() -> None:
    grid = (0.5,)
    curves = [
        _curve("a", 10, (2,), (12,), grid),  # TP=8: prec 8/12, dice 16/22, vol 1.2
        _curve("b", 0, (0,), (0,), grid),  # empty GT, empty mask
        _curve("c", 0, (0,), (5,), grid),  # empty GT, non-empty mask
        _curve("d", 10, (10,), (0,), grid),  # non-empty GT, empty mask
    ]
    q = mask_quality(curves, 0.5)
    assert list(q["case_id"]) == ["a", "b", "c", "d"]
    assert list(q["empty_gt"]) == [False, True, True, False]
    np.testing.assert_allclose(q["precision"], [8 / 12, 1.0, 0.0, np.nan])
    np.testing.assert_allclose(q["dice"], [16 / 22, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(q["volume_ratio"], [1.2, np.nan, np.nan, 0.0])
    with pytest.raises(ValueError):
        mask_quality(curves, 0.4)
