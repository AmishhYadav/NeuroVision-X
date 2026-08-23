"""Gate 2 — does entropy PLUS inter-branch disagreement localise error better than entropy alone?

Pre-registered in `docs/research/preregistration_gate2.md`; read that first.
This module holds the statistics, with no IO and no Hydra: the driver is
`scripts/gate2_localisation.py`.

## Why a fitted combiner, and why it is fitted on val

Gate 1 established that disagreement carries per-voxel error information
entropy does not have (residualised AUROC above 0.5 on all three cohorts).
It also established that entropy's OWN raw AUROC is higher than
disagreement's on every cohort — 0.888 vs 0.694 in distribution. So the
deployable question is not "which single map is better", it is whether the
two together beat entropy alone.

Answering that needs a fitted combination, and a fitted combination
reported on its own fit set is optimistic by construction. This project has
the rule already, from temperature scaling: fit on val, apply to test. The
combiner here is fitted once on the frozen 187-case validation split — which
backs no reported number anywhere — and applied **frozen** to BraTS test,
SSA and PED. A detector that needs the shifted cohort to be refitted is not
a detector that transfers, so there is deliberately no per-cohort refit.

## Two features, and no more

Per-case rank-transformed entropy and per-case rank-transformed
disagreement. Ranks rather than raw values because the two quantities live
on different scales (entropy in nats over three Bernoulli channels,
disagreement as a mean absolute probability difference) and because a
per-case rank makes cases comparable without a per-cohort normalisation
that would itself have to be fitted. No interaction term and no
regularisation search: both are fixed here so that nothing can be selected
on after seeing a p-value.

## The baseline is the same pipeline minus one feature

`ENTROPY_ONLY` fits the identical logistic model on the entropy rank alone,
on the identical voxels of the identical fit set. Both arms therefore share
the sampling mask, the rank transform, the optimiser and the fit split, and
the ONLY difference between them is whether the disagreement feature is
present. Anything else would confound the comparison with a preprocessing
difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

__all__ = [
    "CombinerMode",
    "LocalisationCombiner",
    "case_auroc",
    "fit_combiner",
    "rank_transform",
    "recall_at_budget",
]

CombinerMode = Literal["entropy", "both"]

# Fixed here rather than passed in: the flag budget is a pre-registered
# quantity, and a default that can be overridden per call is a default that
# can be tuned after seeing the answer.
DEFAULT_BUDGET = 0.05


def rank_transform(values: np.ndarray) -> np.ndarray:
    """Ranks `values` within one case and scales them to [0, 1].

    Average ranks for ties, so a map with large flat regions (an entropy
    field is nearly constant across confident background) does not have its
    tied voxels ordered arbitrarily by array position.

    Args:
        values: 1-D array of one case's sampled voxel values.

    Returns:
        Array of the same shape in [0, 1]. A constant input maps to all
        0.5 — the neutral value — rather than dividing by a zero range.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    ranks = rankdata(values, method="average")
    if values.size == 1:
        return np.full_like(values, 0.5)
    return (ranks - 1.0) / (values.size - 1.0)


def _design(entropy_rank: np.ndarray, disagreement_rank: np.ndarray, mode: CombinerMode):
    """Builds the design matrix for one mode, intercept column first."""
    entropy_rank = np.asarray(entropy_rank, dtype=np.float64).reshape(-1)
    ones = np.ones_like(entropy_rank)
    if mode == "entropy":
        return np.column_stack([ones, entropy_rank])
    disagreement_rank = np.asarray(disagreement_rank, dtype=np.float64).reshape(-1)
    return np.column_stack([ones, entropy_rank, disagreement_rank])


@dataclass(frozen=True)
class LocalisationCombiner:
    """A fitted logistic combiner, and the only thing carried from fit to apply.

    Frozen because applying a combiner must never be able to modify it —
    the whole design rests on the SAME coefficients reaching every cohort.

    Attributes:
        mode: `"entropy"` (baseline) or `"both"`.
        coefficients: `[intercept, w_entropy]` or
            `[intercept, w_entropy, w_disagreement]`.
        n_fit_voxels: How many voxels the fit saw, for the record.
        converged: Whether the optimiser reported success.
    """

    mode: CombinerMode
    coefficients: np.ndarray
    n_fit_voxels: int
    converged: bool

    def score(self, entropy_rank: np.ndarray, disagreement_rank: np.ndarray) -> np.ndarray:
        """Returns the linear score for one case's sampled voxels.

        The LINEAR predictor, not the sigmoid of it. Both endpoints (AUROC
        and recall at a budget) depend only on the ORDER of the scores, and
        the sigmoid is strictly monotone, so applying it would change no
        reported number while adding a saturation hazard at the extremes.
        """
        design = _design(entropy_rank, disagreement_rank, self.mode)
        return design @ self.coefficients


def _negative_log_likelihood(weights: np.ndarray, design: np.ndarray, y: np.ndarray):
    """Logistic NLL and its gradient, written to be finite at any saturation.

    `log(1 + exp(z))` overflows for large positive `z`, which is exactly
    where a confident voxel sits. `np.logaddexp(0, z)` is the stable form.
    This is the same lesson as the fp16 entropy clamp that cost this project
    10.5 GPU-hours: a numerical guard has to hold at the extremes the data
    actually reaches, not only in the middle.
    """
    z = design @ weights
    # log-likelihood = y*log(p) + (1-y)*log(1-p) = y*z - logaddexp(0, z)
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))
    p = 1.0 / (1.0 + np.exp(-z))
    grad = design.T @ (p - y)
    return nll, grad


def fit_combiner(
    entropy_rank: np.ndarray,
    disagreement_rank: np.ndarray,
    positive: np.ndarray,
    *,
    mode: CombinerMode,
) -> LocalisationCombiner:
    """Fits the logistic combiner on pooled fit-split voxels.

    Args:
        entropy_rank: Pooled per-case rank-transformed entropy.
        disagreement_rank: Pooled per-case rank-transformed disagreement.
            Ignored when `mode == "entropy"`, so the baseline cannot see it
            even by accident.
        positive: Boolean per-voxel error indicator, same length.
        mode: `"entropy"` or `"both"`.

    Returns:
        The fitted `LocalisationCombiner`.

    Raises:
        ValueError: If the inputs disagree in length, are empty, or the fit
            set contains only one class — a combiner fitted on all-positive
            or all-negative voxels is meaningless and must not be returned
            as if it were usable.
    """
    y = np.asarray(positive, dtype=np.float64).reshape(-1)
    design = _design(entropy_rank, disagreement_rank, mode)
    if design.shape[0] != y.size:
        raise ValueError(f"fit_combiner: {design.shape[0]} feature rows against {y.size} labels.")
    if y.size == 0:
        raise ValueError("fit_combiner: no voxels to fit on.")
    if y.min() == y.max():
        raise ValueError(
            "fit_combiner: the fit set has a single class "
            f"(all {'positive' if y.max() > 0 else 'negative'}); a combiner cannot be fitted."
        )

    result = minimize(
        _negative_log_likelihood,
        x0=np.zeros(design.shape[1]),
        args=(design, y),
        jac=True,
        method="L-BFGS-B",
    )
    return LocalisationCombiner(
        mode=mode,
        coefficients=np.asarray(result.x, dtype=np.float64),
        n_fit_voxels=int(y.size),
        converged=bool(result.success),
    )


def case_auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUROC for one case, by the rank formula, with ties handled.

    Returns NaN when the case has no errors or is entirely error — AUROC is
    undefined there, and a degenerate case must propagate as NaN rather than
    as a plausible 0.5 that would silently dilute a cohort mean.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(positive).reshape(-1).astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recall_at_budget(
    scores: np.ndarray, positive: np.ndarray, *, budget: float = DEFAULT_BUDGET
) -> float:
    """Fraction of a case's erroneous voxels inside the top-`budget` flagged voxels.

    The operational endpoint: flag a fixed share of the predicted foreground
    for review and ask how much of the error that catches.

    Ties at the budget boundary are broken by taking the highest-scoring
    `k` voxels via `argpartition`, so the flagged set is always exactly `k`
    voxels. An "all voxels above the k-th value" rule would flag more than
    the budget whenever the boundary value repeats — which for a rank
    feature with ties is common, and would quietly give one arm a larger
    budget than the other.

    Returns NaN when the case has no errors, for the same reason as
    `case_auroc`.
    """
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"recall_at_budget: budget must be in (0, 1], got {budget}.")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(positive).reshape(-1).astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = max(1, int(round(budget * scores.size)))
    k = min(k, scores.size)
    flagged = np.argpartition(-scores, k - 1)[:k]
    return float(y[flagged].sum() / n_pos)
