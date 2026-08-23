"""Tests for neurovision.analysis.equivalence (paired TOST)."""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.analysis.equivalence import paired_tost


def test_tight_difference_inside_the_margin_is_equivalent() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.70, 0.02, size=60)
    b = a + rng.normal(0.001, 0.005, size=60)
    result = paired_tost(a, b, margin=0.03)
    assert result.equivalent
    assert abs(result.mean_difference) < 0.03


def test_a_real_difference_larger_than_the_margin_is_not_equivalent() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.70, 0.02, size=60)
    b = a - 0.08
    result = paired_tost(a, b, margin=0.03)
    assert not result.equivalent


def test_noisy_small_sample_is_NOT_declared_equivalent() -> None:
    """The whole point: an underpowered null must fail TOST, not pass it.

    A plain paired t-test on this data would be non-significant, and reading
    that as 'no difference' is the error TOST exists to prevent.
    """
    rng = np.random.default_rng(2)
    a = rng.normal(0.70, 0.15, size=8)
    b = rng.normal(0.70, 0.15, size=8)
    result = paired_tost(a, b, margin=0.03)
    assert not result.equivalent
    assert result.p_tost > 0.05


def test_interval_is_the_one_minus_two_alpha_interval() -> None:
    """TOST is algebraically the 1-2*alpha interval; reporting 95% would mislead."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.05, size=40)
    b = a + rng.normal(0.0, 0.01, size=40)
    result = paired_tost(a, b, margin=0.03, alpha=0.05)
    # Equivalence <=> the interval lies strictly inside the margin.
    assert result.equivalent == (result.ci_lo > -0.03 and result.ci_hi < 0.03)


def test_identical_arms_are_equivalent_without_dividing_by_zero() -> None:
    a = np.linspace(0.4, 0.9, 20)
    result = paired_tost(a, a.copy(), margin=0.03)
    assert result.equivalent
    assert result.mean_difference == 0.0


def test_constant_nonzero_difference_beyond_the_margin_is_not_equivalent() -> None:
    a = np.linspace(0.4, 0.9, 20)
    result = paired_tost(a, a - 0.10, margin=0.03)
    assert not result.equivalent


def test_nan_pairs_are_dropped_not_imputed() -> None:
    a = np.array([0.7, 0.72, np.nan, 0.69])
    b = np.array([0.70, 0.71, 0.5, np.nan])
    result = paired_tost(a, b, margin=0.03)
    assert result.n == 2


def test_rejects_a_nonpositive_margin() -> None:
    with pytest.raises(ValueError, match="margin"):
        paired_tost(np.arange(5.0), np.arange(5.0), margin=0.0)


def test_rejects_too_few_pairs() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        paired_tost(np.array([0.5]), np.array([0.4]), margin=0.03)
