"""Tests for neurovision.analysis.localisation (Gate 2's statistics).

Every test here is against a hand-constructed case with a known answer, not
against the module's own output on random data -- the failure mode this
module could have is producing a plausible number that is wrong, which only
an oracle catches.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.analysis.localisation import (
    LocalisationCombiner,
    case_auroc,
    fit_combiner,
    rank_transform,
    recall_at_budget,
)

# ---------------------------------------------------------------------------
# rank_transform
# ---------------------------------------------------------------------------


def test_rank_transform_maps_to_unit_interval_in_order() -> None:
    out = rank_transform(np.array([5.0, 1.0, 3.0]))
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[1] < out[2] < out[0]


def test_rank_transform_averages_ties() -> None:
    """Tied voxels must share a rank, not be ordered by array position."""
    out = rank_transform(np.array([2.0, 2.0, 9.0]))
    assert out[0] == out[1]


def test_rank_transform_constant_input_is_neutral_not_nan() -> None:
    """An entropy map can be flat over a sample; that must not divide by zero."""
    out = rank_transform(np.full(4, 7.0))
    assert np.allclose(out, 0.5)


# ---------------------------------------------------------------------------
# case_auroc -- against hand-computed values
# ---------------------------------------------------------------------------


def test_case_auroc_perfect_separation_is_one() -> None:
    assert case_auroc(np.array([0.1, 0.2, 0.9, 1.0]), np.array([0, 0, 1, 1])) == 1.0


def test_case_auroc_inverted_separation_is_zero() -> None:
    assert case_auroc(np.array([0.9, 1.0, 0.1, 0.2]), np.array([0, 0, 1, 1])) == 0.0


def test_case_auroc_all_ties_is_one_half() -> None:
    assert case_auroc(np.full(6, 3.0), np.array([0, 1, 0, 1, 0, 1])) == 0.5


def test_case_auroc_known_value() -> None:
    """Two positives ranked 2nd and 4th of 4: (2 + 4 - 3) / (2 * 2) = 0.75."""
    assert case_auroc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 1, 0, 1])) == 0.75


def test_case_auroc_is_nan_when_a_case_has_no_error() -> None:
    """Undefined, and must propagate as NaN rather than a plausible 0.5."""
    assert np.isnan(case_auroc(np.array([0.1, 0.2]), np.array([0, 0])))
    assert np.isnan(case_auroc(np.array([0.1, 0.2]), np.array([1, 1])))


# ---------------------------------------------------------------------------
# recall_at_budget
# ---------------------------------------------------------------------------


def test_recall_at_budget_catches_the_top_scoring_errors() -> None:
    scores = np.arange(100.0)
    positive = np.zeros(100, dtype=bool)
    positive[[97, 98, 99]] = True  # all three errors are the highest scores
    assert recall_at_budget(scores, positive, budget=0.05) == 1.0


def test_recall_at_budget_misses_errors_the_score_ranks_low() -> None:
    scores = np.arange(100.0)
    positive = np.zeros(100, dtype=bool)
    positive[[0, 1, 2]] = True
    assert recall_at_budget(scores, positive, budget=0.05) == 0.0


def test_recall_at_budget_flags_exactly_k_voxels_under_heavy_ties() -> None:
    """A tie at the boundary must not silently enlarge the budget.

    With 100 identical scores and a 5% budget, exactly 5 voxels are flagged,
    so a case where every voxel is an error recalls exactly 0.05. An
    'above the k-th value' rule would flag all 100 and report 1.0.
    """
    scores = np.full(100, 2.0)
    positive = np.ones(100, dtype=bool)
    assert recall_at_budget(scores, positive, budget=0.05) == pytest.approx(0.05)


def test_recall_at_budget_is_nan_without_errors() -> None:
    assert np.isnan(recall_at_budget(np.arange(10.0), np.zeros(10, dtype=bool)))


def test_recall_at_budget_rejects_a_budget_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="budget"):
        recall_at_budget(np.arange(10.0), np.ones(10, dtype=bool), budget=0.0)


# ---------------------------------------------------------------------------
# fit_combiner
# ---------------------------------------------------------------------------


def _separable_sample(n: int = 4000, seed: int = 0):
    """Errors driven by the DISAGREEMENT feature, with entropy pure noise."""
    rng = np.random.default_rng(seed)
    entropy = rng.uniform(0.0, 1.0, size=n)
    disagreement = rng.uniform(0.0, 1.0, size=n)
    positive = rng.uniform(size=n) < disagreement  # monotone in disagreement only
    return entropy, disagreement, positive


def test_fit_combiner_recovers_the_informative_feature() -> None:
    entropy, disagreement, positive = _separable_sample()
    fitted = fit_combiner(entropy, disagreement, positive, mode="both")
    assert fitted.converged
    # coefficients = [intercept, w_entropy, w_disagreement]
    assert fitted.coefficients[2] > 1.0
    assert abs(fitted.coefficients[1]) < 0.5


def test_entropy_only_mode_never_sees_the_disagreement_feature() -> None:
    """The baseline arm must be structurally blind to the extra feature."""
    entropy, disagreement, positive = _separable_sample()
    a = fit_combiner(entropy, disagreement, positive, mode="entropy")
    b = fit_combiner(entropy, disagreement * 0.0 + 0.123, positive, mode="entropy")
    assert a.coefficients.shape == (2,)
    assert np.allclose(a.coefficients, b.coefficients)


def test_fit_combiner_rejects_a_single_class_fit_set() -> None:
    with pytest.raises(ValueError, match="single class"):
        fit_combiner(np.arange(10.0), np.arange(10.0), np.ones(10, dtype=bool), mode="both")


def test_fit_combiner_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="feature rows"):
        fit_combiner(np.arange(10.0), np.arange(10.0), np.zeros(9, dtype=bool), mode="both")


def test_fit_is_finite_under_perfect_separation() -> None:
    """Saturation must not produce NaN -- the fp16-entropy lesson, restated."""
    entropy = np.concatenate([np.zeros(50), np.ones(50)])
    disagreement = entropy.copy()
    positive = entropy > 0.5
    fitted = fit_combiner(entropy, disagreement, positive, mode="both")
    assert np.all(np.isfinite(fitted.coefficients))


def test_score_is_monotone_and_uses_frozen_coefficients() -> None:
    combiner = LocalisationCombiner(
        mode="both", coefficients=np.array([0.0, 1.0, 2.0]), n_fit_voxels=10, converged=True
    )
    scores = combiner.score(np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert scores[1] > scores[0]
    assert np.allclose(scores, [0.0, 2.0])
