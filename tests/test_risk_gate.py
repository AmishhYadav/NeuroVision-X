"""Tests for the LTT selective-risk gate (hand-computed values)."""

import math

import numpy as np
import pytest
from scipy import stats

from neurovision.analysis.risk_gate import (
    COVERAGE_GRID,
    accepted_mask,
    apply_gate,
    binomial_pvalue,
    cell_verdict,
    clopper_pearson,
    fit_ltt,
    selective_summary,
    threshold_for_coverage,
)


def test_grid():
    assert len(COVERAGE_GRID) == 11
    assert COVERAGE_GRID[0] == 0.5 and COVERAGE_GRID[1] == 0.55
    assert COVERAGE_GRID[-1] == 1.0


def test_threshold_for_coverage():
    s = np.random.default_rng(0).permutation(np.arange(1.0, 11.0))
    assert threshold_for_coverage(s, 0.5) == 5
    assert threshold_for_coverage(s, 0.55) == 6  # ceil(5.5)
    assert threshold_for_coverage(s, 1.0) == 10
    with_inf = np.append(s, np.inf)
    assert threshold_for_coverage(with_inf, 1.0) == np.inf
    with pytest.raises(ValueError):
        threshold_for_coverage(np.array([1.0, np.nan]), 0.5)
    with pytest.raises(ValueError):
        threshold_for_coverage(s, 0.0)
    with pytest.raises(ValueError):
        threshold_for_coverage(np.array([]), 0.5)


def test_ties_accepted():
    s = np.array([1.0, 2, 2, 2, 5])
    lam = threshold_for_coverage(s, 0.4)
    assert lam == 2
    assert accepted_mask(s, lam).sum() == 4


def test_binomial_pvalue():
    for k, n, b in [(1, 94, 0.05), (3, 150, 0.05), (9, 150, 0.10)]:
        assert binomial_pvalue(k, n, b) == stats.binom.cdf(k, n, b)
    assert binomial_pvalue(0, 2, 0.5) == pytest.approx(0.25)
    assert binomial_pvalue(0, 0, 0.5) == 1.0


def test_fit_ltt_perfect_signal():
    scores = np.arange(100.0)
    unusable = scores >= 90
    fit = fit_ltt(scores, unusable, beta=0.10, delta=0.1)
    assert fit.feasible
    by_c = {s.coverage: s for s in fit.steps}
    assert by_c[0.9].k_unusable == 0 and by_c[0.9].rejected
    s95 = by_c[0.95]
    assert (s95.n_accepted, s95.k_unusable) == (95, 5)
    assert s95.p_value == stats.binom.cdf(5, 95, 0.1)
    assert s95.rejected == (s95.p_value <= 0.1)
    last = fit.steps[-1]
    assert (last.n_accepted, last.k_unusable) == (100, 10)
    assert not last.rejected  # p = cdf(10,100,.1) ~ 0.58
    # stops at first non-rejection; lam_hat is last rejected step's lambda
    assert [s.rejected for s in fit.steps].count(False) <= 1
    rejected = [s for s in fit.steps if s.rejected]
    assert fit.lam == rejected[-1].lam
    assert fit.target_coverage == rejected[-1].coverage
    assert fit.calib_selective_risk == rejected[-1].k_unusable / rejected[-1].n_accepted


def test_fit_ltt_useless_signal():
    scores = np.arange(100.0)
    unusable = scores < 20  # unusable cases have the lowest scores
    fit = fit_ltt(scores, unusable, beta=0.05)
    assert not fit.feasible
    assert math.isnan(fit.lam) and math.isnan(fit.target_coverage)
    assert len(fit.steps) == 1 and fit.steps[0].k_unusable == 20
    assert fit.n_accepted == 0 and fit.k_unusable == 0


def test_fst_stops_at_first_failure():
    # 5 unusable cases sit at ranks 50..54, just inside the 0.55 band: c=0.50
    # (k=0) is rejected, c=0.55 (n=55, k=5, beta=0.05) is not, so we stop there.
    # Later steps are never evaluated, even though the sequence continues.
    scores = np.arange(100.0)
    unusable = np.zeros(100, dtype=bool)
    unusable[50:55] = True
    fit = fit_ltt(scores, unusable, beta=0.05, delta=0.1)
    assert len(fit.steps) == 2
    assert fit.steps[0].rejected and not fit.steps[1].rejected
    assert fit.lam == 49.0  # 50th smallest score at c=0.50


def test_clopper_pearson():
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    lo, hi = clopper_pearson(5, 10)
    assert lo + hi == pytest.approx(1.0, abs=1e-9)
    lo, hi = clopper_pearson(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_cell_verdict():
    assert cell_verdict(0.0, 0.0, 0.1, 0) == "accepts_nothing"
    assert cell_verdict(0.2, 0.4, 0.1, 10) == "VIOLATED"
    assert cell_verdict(0.0, 0.08, 0.1, 10) == "holds"
    assert cell_verdict(0.02, 0.3, 0.1, 10) == "inconclusive"


def test_apply_gate_hand_example():
    scores = np.array([1.0, 2, 3, 4, 5, 6])
    unusable = np.array([False, False, True, False, True, False])
    g = apply_gate(scores, unusable, lam=4.0, beta=0.5)
    assert g.n == 6 and g.n_accepted == 4
    assert g.coverage == pytest.approx(4 / 6)
    assert g.k_unusable_accepted == 1
    assert g.selective_risk == pytest.approx(0.25)
    assert g.silent_failure == pytest.approx(1 / 6)
    assert g.over_refusal == pytest.approx(1 / 6)  # case 6 refused and usable
    with pytest.raises(ValueError):
        apply_gate(scores, unusable, lam=math.nan, beta=0.5)


def test_selective_summary_accepts_nothing():
    g = selective_summary(np.zeros(3, bool), np.array([True, False, False]), 0.1)
    assert g.verdict == "accepts_nothing"
    assert math.isnan(g.selective_risk)
    assert g.over_refusal == pytest.approx(2 / 3)


def test_input_validation():
    with pytest.raises(ValueError):
        fit_ltt(np.arange(5.0), np.zeros(4, bool), beta=0.1)
    with pytest.raises(ValueError):
        fit_ltt(np.zeros((2, 3)), np.zeros((2, 3), bool), beta=0.1)
    with pytest.raises(ValueError):
        fit_ltt(np.arange(5.0), np.zeros(5, bool), beta=0.1, coverages=(0.6, 0.5))
