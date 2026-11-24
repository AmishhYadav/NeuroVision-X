"""Learn-then-Test (LTT) selective-risk gate.

A per-case score (higher = predicted worse mask) is thresholded: a case is
ACCEPTED iff ``score <= lambda``. LTT picks lambda so that, with probability
>= 1 - delta over the calibration draw, P(unusable | accepted) <= beta.

Protocol: ``docs/research_docs/preregistrations/preregistration_risk_gate.md``.
All functions are pure; scores are 1-D arrays with one entry per case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Target coverages tried in order, smallest first (11 points, 0.50 ... 1.00).
# Rounded so each value is the exact decimal float (0.55, not 0.55000000000000004).
COVERAGE_GRID: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(11))


@dataclass(frozen=True)
class LTTStep:
    """One fixed-sequence test at a single target coverage."""

    coverage: float
    lam: float
    n_accepted: int
    k_unusable: int
    p_value: float
    rejected: bool


@dataclass(frozen=True)
class LTTFit:
    """Result of :func:`fit_ltt`. ``lam`` / ``target_coverage`` are NaN if infeasible."""

    beta: float
    delta: float
    feasible: bool
    lam: float
    target_coverage: float
    n_calibration: int
    n_accepted: int
    k_unusable: int
    calib_selective_risk: float
    steps: tuple[LTTStep, ...]


@dataclass(frozen=True)
class GateApplication:
    """A gate scored on one cohort. Rates are fractions of all n cases."""

    n: int
    n_accepted: int
    coverage: float
    k_unusable_accepted: int
    selective_risk: float  # NaN if nothing accepted
    cp_lo: float
    cp_hi: float
    silent_failure: float  # accepted AND unusable, / n
    over_refusal: float  # refused AND usable, / n
    verdict: str


def _check_scores(scores: np.ndarray) -> np.ndarray:
    """Return scores as a float 1-D array; reject empty, non-1-D or NaN input."""
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    if s.size == 0:
        raise ValueError("scores is empty")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")  # +inf is allowed, NaN is not
    return s


def _check_bool(arr: np.ndarray, n: int, name: str) -> np.ndarray:
    """Return a bool 1-D array of length n."""
    a = np.asarray(arr, dtype=bool)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be 1-D with length {n}, got shape {a.shape}")
    return a


def threshold_for_coverage(scores: np.ndarray, coverage: float) -> float:
    """Return the ceil(coverage * n)-th smallest score (1-indexed).

    Args:
        scores: ``(n,)`` scores; may contain ``+inf`` but not NaN.
        coverage: Target accepted fraction in (0, 1].

    Returns:
        The threshold lambda. With coverage 1.0 this is the max score
        (``+inf`` if any case is always-refuse, meaning accept everything).
    """
    s = _check_scores(scores)
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    n = s.size
    # round() first: 0.55 * 20 = 11.000000000000002 would otherwise ceil to 12.
    k = math.ceil(round(coverage * n, 9))
    k = min(max(k, 1), n)
    return float(np.sort(s)[k - 1])


def accepted_mask(scores: np.ndarray, lam: float) -> np.ndarray:
    """Return a bool ``(n,)`` mask, True where ``score <= lam``.

    Ties are accepted (every case equal to lambda is in), so the accepted
    count can exceed ceil(coverage * n). ``+inf <= +inf`` is True on purpose:
    lambda = +inf means "accept everything".
    """
    return np.asarray(scores, dtype=float) <= lam


def binomial_pvalue(k: int, n: int, beta: float) -> float:
    """Return P(Binomial(n, beta) <= k); 1.0 if n == 0 (cannot reject).

    Small p means seeing only k unusable among n accepted is unlikely if the
    true selective risk were > beta, so we can reject "risk > beta".
    """
    if n == 0:
        return 1.0
    return float(stats.binom.cdf(k, n, beta))


def fit_ltt(
    scores: np.ndarray,
    unusable: np.ndarray,
    beta: float,
    delta: float = 0.1,
    coverages: tuple[float, ...] = COVERAGE_GRID,
) -> LTTFit:
    """Choose lambda by fixed-sequence testing over increasing coverage.

    Args:
        scores: ``(n,)`` calibration scores, higher = predicted worse.
        unusable: ``(n,)`` bool, True where the mask is unusable.
        beta: Selective-risk target in (0, 1).
        delta: Allowed failure probability in (0, 1).
        coverages: Strictly increasing target coverages in (0, 1].

    Returns:
        An :class:`LTTFit`. ``lam`` is the threshold of the last rejected
        step; the fit is infeasible if the first step is not rejected.
    """
    s = _check_scores(scores)
    n = s.size
    u = _check_bool(unusable, n, "unusable")
    if not (0.0 < beta < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("beta and delta must be in (0, 1)")
    cov = tuple(coverages)
    if (
        len(cov) == 0
        or any(not (0.0 < c <= 1.0) for c in cov)
        or any(b <= a for a, b in zip(cov, cov[1:]))
    ):
        raise ValueError("coverages must be strictly increasing and in (0, 1]")

    steps: list[LTTStep] = []
    last: LTTStep | None = None
    for c in cov:
        lam = threshold_for_coverage(s, c)
        acc = accepted_mask(s, lam)
        n_c = int(acc.sum())
        k_c = int(u[acc].sum())
        p = binomial_pvalue(k_c, n_c, beta)
        step = LTTStep(c, lam, n_c, k_c, p, p <= delta)
        steps.append(step)
        if not step.rejected:
            # Fixed-sequence testing: the guarantee only holds because we stop
            # at the first failure. Testing further would need a correction.
            break
        last = step

    if last is None:
        return LTTFit(beta, delta, False, math.nan, math.nan, n, 0, 0, math.nan, tuple(steps))
    risk = last.k_unusable / last.n_accepted if last.n_accepted else math.nan
    return LTTFit(
        beta,
        delta,
        True,
        last.lam,
        last.coverage,
        n,
        last.n_accepted,
        last.k_unusable,
        risk,
        tuple(steps),
    )


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for k successes in n trials.

    Returns:
        ``(lo, hi)``; ``(nan, nan)`` if n == 0.
    """
    if n == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def cell_verdict(cp_lo: float, cp_hi: float, beta: float, n_accepted: int) -> str:
    """Classify a cohort: accepts_nothing / VIOLATED / holds / inconclusive."""
    if n_accepted == 0:
        return "accepts_nothing"
    if cp_lo > beta:
        return "VIOLATED"  # even the optimistic end of the interval exceeds beta
    if cp_hi <= beta:
        return "holds"  # even the pessimistic end is within beta
    return "inconclusive"


def selective_summary(
    accepted: np.ndarray, unusable: np.ndarray, beta: float, conf: float = 0.95
) -> GateApplication:
    """Score an externally given accepted set.

    Args:
        accepted: ``(n,)`` bool, True where the gate accepted the case.
        unusable: ``(n,)`` bool, True where the mask is unusable.
        beta: Selective-risk target used for the verdict.
        conf: Clopper-Pearson confidence level.
    """
    acc = np.asarray(accepted, dtype=bool)
    if acc.ndim != 1 or acc.size == 0:
        raise ValueError("accepted must be a non-empty 1-D array")
    n = acc.size
    u = _check_bool(unusable, n, "unusable")
    n_acc = int(acc.sum())
    k = int((acc & u).sum())
    risk = k / n_acc if n_acc else math.nan
    lo, hi = clopper_pearson(k, n_acc, conf)
    return GateApplication(
        n=n,
        n_accepted=n_acc,
        coverage=n_acc / n,
        k_unusable_accepted=k,
        selective_risk=risk,
        cp_lo=lo,
        cp_hi=hi,
        silent_failure=k / n,
        over_refusal=int((~acc & ~u).sum()) / n,
        verdict=cell_verdict(lo, hi, beta, n_acc),
    )


def apply_gate(
    scores: np.ndarray,
    unusable: np.ndarray,
    lam: float,
    beta: float,
    conf: float = 0.95,
) -> GateApplication:
    """Apply a frozen threshold ``lam`` to a cohort and score it.

    Raises:
        ValueError: If ``lam`` is NaN (an infeasible gate must not be applied).
    """
    if math.isnan(lam):
        raise ValueError("lam is NaN: the gate is infeasible and cannot be applied")
    s = _check_scores(scores)
    return selective_summary(accepted_mask(s, lam), unusable, beta, conf)
