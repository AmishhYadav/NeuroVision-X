"""Two one-sided tests (TOST) for a paired equivalence claim.

## Why this exists

"Disagreement matches MC-dropout at one tenth the cost" is an **equivalence**
claim, and equivalence cannot be established by failing to reject a
difference. A non-significant paired test means "we did not detect a
difference", which at n=60 is exactly what an underpowered study looks like --
and this project has already produced six underpowered nulls. Asserting
equivalence from one of them would be the seventh and the worst.

TOST inverts the burden properly: it asks whether the confidence interval for
the difference lies ENTIRELY inside a margin fixed in advance. Rejecting both
one-sided nulls is positive evidence of equivalence, not an absence of
evidence of difference.

The margin here is **not** chosen in this module. It is passed in, and for the
MC-dropout comparison it was fixed at 0.03 AUROC in
`docs/research/execution_plan.md` §Phase 2 before any MC-dropout map for an
external cohort existed. A margin chosen after seeing the data is not a margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TOSTResult", "paired_tost"]


@dataclass(frozen=True)
class TOSTResult:
    """Outcome of a paired TOST.

    Attributes:
        mean_difference: Mean of `a - b`.
        margin: The equivalence margin, as supplied.
        ci_lo: Lower bound of the (1 - 2*alpha) interval TOST is equivalent to.
        ci_hi: Upper bound of that interval.
        p_lower: p-value for H0: difference <= -margin.
        p_upper: p-value for H0: difference >= +margin.
        p_tost: The larger of the two, which is the TOST p-value.
        equivalent: Whether equivalence is established at `alpha`.
        n: Number of paired observations.
    """

    mean_difference: float
    margin: float
    ci_lo: float
    ci_hi: float
    p_lower: float
    p_upper: float
    p_tost: float
    equivalent: bool
    n: int


def paired_tost(a: np.ndarray, b: np.ndarray, *, margin: float, alpha: float = 0.05) -> TOSTResult:
    """Paired TOST on `a - b` against a symmetric equivalence margin.

    Args:
        a: First arm, one value per case.
        b: Second arm, paired with `a`.
        margin: Equivalence margin, strictly positive, fixed BEFORE seeing
            the data. Equivalence means the true difference lies within
            `(-margin, +margin)`.
        alpha: Significance level for each one-sided test.

    Returns:
        A `TOSTResult`. The reported interval is the `1 - 2*alpha` interval,
        which is the interval TOST is algebraically equivalent to -- reporting
        a 95% interval next to a 5% TOST would invite the reader to check
        equivalence against the wrong bounds.

    Raises:
        ValueError: If the arms differ in length, fewer than two pairs
            survive, or `margin <= 0`.
    """
    if margin <= 0:
        raise ValueError(f"paired_tost: margin must be positive, got {margin}.")
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"paired_tost: {a.size} values against {b.size}.")

    diff = a - b
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 2:
        raise ValueError(f"paired_tost: need at least 2 finite pairs, got {n}.")

    # Student t on the paired differences. scipy is already a dependency; the
    # t distribution is imported lazily so this module stays importable in a
    # plotting-only environment, matching the figures.py convention.
    from scipy import stats

    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        # Every case has an identical difference. Equivalence is then decided
        # by the point value alone; a zero standard error would otherwise make
        # the t statistics infinite and the p-values 0 or 1 by accident.
        equivalent = abs(mean) < margin
        return TOSTResult(
            mean,
            margin,
            mean,
            mean,
            0.0 if equivalent else 1.0,
            0.0 if equivalent else 1.0,
            0.0 if equivalent else 1.0,
            equivalent,
            n,
        )

    df = n - 1
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = float(stats.t.sf(t_lower, df))  # H0: diff <= -margin
    p_upper = float(stats.t.cdf(t_upper, df))  # H0: diff >= +margin
    p_tost = max(p_lower, p_upper)

    crit = float(stats.t.ppf(1.0 - alpha, df))
    ci_lo = mean - crit * se
    ci_hi = mean + crit * se
    return TOSTResult(
        mean_difference=mean,
        margin=float(margin),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=bool(p_tost < alpha),
        n=n,
    )
