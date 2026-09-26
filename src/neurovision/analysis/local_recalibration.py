"""The Mondrian / per-cohort recalibration counterfactual for conformal risk control.

## What this asks

`neurovision.uncertainty.conformal.fit_threshold` picks a mask threshold `tau_hat` on a
calibration set so that, for a FRESH case EXCHANGEABLE with that calibration set, the
expected miss rate is bounded by `alpha`. The pre-registration's primary result (see
`docs/research/preregistration_conformal.md`) fits `tau_hat` on BraTS val and applies it
FROZEN to two external cohorts, SSA and PED -- and the bound breaks, because a case from a
different scanner population, age group or protocol is not exchangeable with val.

This module asks a different, narrower question: **if a new site labelled `k` of its OWN
cases and recalibrated `tau_hat` on exactly those, would the bound hold on that site's
REMAINING cases?** Exchangeability holds *within* a cohort even when it fails *across*
cohorts, so the pre-registered claim is that this restores the guarantee. `run_local_recalibration`
draws many random within-cohort splits (`draw_split`), refits on each calibration half
(`evaluate_split`, reusing `conformal.fit_threshold` verbatim -- this module never
reimplements the conformal inequality), and scores the fitted threshold on the held-out half.

**This is a counterfactual, not an external-validation result.** It answers "what would it
take to fix this", never "how well does the frozen-threshold pipeline do on unseen data" --
see the pre-registration's explicit warning at the point this arm is introduced. Every
caller of this module must keep that label attached to anything it prints or plots.

## Randomness

Every draw goes through the `numpy.random.Generator` the caller passes in (seed via
`neurovision.utils.seed.set_seed`, which returns a `torch.Generator` for torch code but
leaves NumPy seeded globally -- callers here should instead hold their own
`np.random.default_rng(seed)` for this module, since nothing here touches torch). This
module contains no other source of randomness and does no file I/O; the CLI driver that
loads curves from disk and writes `summarise`'s table out is a separate, later module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurovision.uncertainty.conformal import (
    CaseLossCurve,
    band_inflation,
    fit_threshold,
    realised_risk,
)

__all__ = [
    "SplitResult",
    "draw_split",
    "evaluate_split",
    "run_local_recalibration",
    "min_feasible_k",
    "summarise",
]

logger = logging.getLogger(__name__)

# The loss's known upper bound, `B` in conformal.fit_threshold's finite-sample correction
# `(n * R_hat(tau) + B) / (n + 1) <= alpha`. Matches fit_threshold's own default exactly
# (the miss rate is a fraction, always in [0, 1]) -- see min_feasible_k for where this is used.
_LOSS_BOUND = 1.0


@dataclass(frozen=True)
class SplitResult:
    """One random within-cohort calibration/held-out split, fitted and scored.

    Attributes:
        cohort: Cohort name this split was drawn from (e.g. `"SSA"`, `"PED"`).
        region: Region name (e.g. `"ET"`, `"TC"`, `"WT"`).
        alpha: Target risk level this split was fitted at.
        k: Calibration-set size used for this split.
        split_index: Which of the (independently drawn) random splits this is.
        n_calibration: Number of case ids in the calibration half (equals `k`).
        n_heldout: Number of case ids in the held-out half.
        feasible: Whether `fit_threshold` found a feasible threshold on the
            calibration half. `False` is a legitimate outcome, not an error --
            see `conformal.ConformalFit`.
        threshold: The fitted threshold, or `None` if infeasible.
        realised_risk: Mean miss rate on the held-out half at `threshold`, or
            `None` if infeasible, or if every held-out case had empty ground
            truth (the mean is then itself undefined, 0/0 -- see
            `conformal.realised_risk`'s own docstring for why that case is
            never silently scored as 0 or 1).
        violated: Whether `realised_risk > alpha`, or `None` wherever
            `realised_risk` itself is `None`.
        inflation: Mean mask-inflation ratio on the held-out half at
            `threshold` (`conformal.band_inflation`'s `mean_inflation`), or
            `None` under the same conditions as `realised_risk`.
    """

    cohort: str
    region: str
    alpha: float
    k: int
    split_index: int
    n_calibration: int
    n_heldout: int
    feasible: bool
    threshold: float | None
    realised_risk: float | None
    violated: bool | None
    inflation: float | None


def draw_split(
    case_ids: Sequence[str], k: int, rng: np.random.Generator
) -> tuple[list[str], list[str]]:
    """Randomly splits case ids into a calibration set of size `k` and a held-out remainder.

    Args:
        case_ids: All case ids to split; order does not matter going in.
        k: Calibration-set size. Must satisfy `1 <= k <= len(case_ids) - 1` so
            neither half is empty.
        rng: Generator the split is drawn from. Calling this twice with a
            freshly seeded generator of the same seed reproduces the same
            split exactly.

    Returns:
        `(calibration_ids, heldout_ids)`: disjoint lists that together cover
        every id in `case_ids`, in an order determined by `rng`.

    Raises:
        ValueError: `k` is not in `[1, len(case_ids) - 1]`.
    """
    n = len(case_ids)
    if not (1 <= k <= n - 1):
        raise ValueError(
            f"k must satisfy 1 <= k <= {n - 1} (len(case_ids) - 1 = {n} - 1), got k={k}."
        )
    ids = list(case_ids)
    order = rng.permutation(n)
    calibration_ids = [ids[i] for i in order[:k]]
    heldout_ids = [ids[i] for i in order[k:]]
    return calibration_ids, heldout_ids


def evaluate_split(
    curves: Sequence[CaseLossCurve],
    calib_ids: Sequence[str],
    heldout_ids: Sequence[str],
    alpha: float,
    *,
    cohort: str,
    region: str,
    k: int,
    split_index: int,
) -> SplitResult:
    """Fits a conformal threshold on one calibration half and scores it on the held-out half.

    Reuses `conformal.fit_threshold` / `realised_risk` / `band_inflation` exactly as they
    are -- this function only decides which curves feed which call.

    Args:
        curves: Every `CaseLossCurve` for one cohort and one region, keyed by
            `case_id`. `calib_ids` and `heldout_ids` must each be a subset of
            the case ids present here.
        calib_ids: Case ids to fit the threshold on (the "own labelled cases"
            in the module docstring's counterfactual).
        heldout_ids: Case ids to score the fitted threshold on.
        alpha: Target risk level passed through to `fit_threshold`.
        cohort: Cohort name, stamped onto the result.
        region: Region name, stamped onto the result.
        k: Calibration-set size, stamped onto the result.
        split_index: Which random split this is, stamped onto the result.

    Returns:
        A `SplitResult`. When the calibration fit is infeasible, `threshold`,
        `realised_risk`, `violated` and `inflation` are all `None` and
        `realised_risk`/`band_inflation` are never called -- mirroring
        `fit_threshold`'s own "infeasible is a registered outcome, not an
        error" contract rather than treating a missing threshold as 0.5.
    """
    by_id = {c.case_id: c for c in curves}
    calib_curves = [by_id[cid] for cid in calib_ids]
    heldout_curves = [by_id[cid] for cid in heldout_ids]

    fit = fit_threshold(calib_curves, alpha)

    if not fit.feasible:
        return SplitResult(
            cohort=cohort,
            region=region,
            alpha=alpha,
            k=k,
            split_index=split_index,
            n_calibration=len(calib_ids),
            n_heldout=len(heldout_ids),
            feasible=False,
            threshold=None,
            realised_risk=None,
            violated=None,
            inflation=None,
        )

    risk = realised_risk(heldout_curves, threshold=fit.threshold)
    band = band_inflation(heldout_curves, threshold=fit.threshold)

    # A held-out half that is entirely empty-ground-truth cases makes the mean miss rate
    # 0/0 (NaN) -- the same "undefined, not 0 and not 1" convention conformal.py applies to
    # a single case's miss_rate() -- so we do not paper over it by scoring 0 or False here.
    mean_miss = risk["mean_miss_rate"]
    realised_val: float | None
    violated_val: bool | None
    if np.isnan(mean_miss):
        realised_val = None
        violated_val = None
    else:
        realised_val = float(mean_miss)
        violated_val = bool(mean_miss > alpha)

    mean_inflation = band["mean_inflation"]
    inflation_val: float | None = None if np.isnan(mean_inflation) else float(mean_inflation)

    return SplitResult(
        cohort=cohort,
        region=region,
        alpha=alpha,
        k=k,
        split_index=split_index,
        n_calibration=len(calib_ids),
        n_heldout=len(heldout_ids),
        feasible=True,
        threshold=fit.threshold,
        realised_risk=realised_val,
        violated=violated_val,
        inflation=inflation_val,
    )


def run_local_recalibration(
    curves_by_region: Mapping[str, Sequence[CaseLossCurve]],
    *,
    cohort: str,
    alphas: Sequence[float],
    ks: Sequence[int | str],
    n_splits: int,
    rng: np.random.Generator,
) -> list[SplitResult]:
    """Runs the Mondrian recalibration counterfactual over every (region, alpha, k).

    For each region and each requested `k`, draws `n_splits` independent random splits
    (`draw_split`) and, for each split, fits and scores a threshold at every `alpha` --
    using the SAME split for every alpha at a given `(region, k, split_index)`, so results
    across alphas are directly comparable and are not confounded by different random draws.

    Args:
        curves_by_region: `{region: curves}`, one list of `CaseLossCurve` per
            region, covering every case in `cohort`.
        cohort: Cohort name (e.g. `"SSA"`, `"PED"`), stamped onto every result.
        alphas: Target risk levels to sweep.
        ks: Calibration-set sizes to sweep. Each entry is an `int`, or the
            literal string `"half"`, resolved to `floor(n / 2)` separately
            per region (region case counts can differ).
        n_splits: Number of independent random splits per (region, k).
        rng: Generator every random split is drawn from. Reusing a
            freshly-seeded generator across two calls reproduces the exact
            same list of results.

    Returns:
        One `SplitResult` per `(region, alpha, k, split_index)` actually run.

    Raises:
        ValueError: A `ks` entry is a string other than `"half"`.
    """
    results: list[SplitResult] = []
    for region, curves in curves_by_region.items():
        case_ids = [c.case_id for c in curves]
        n = len(case_ids)
        for k_spec in ks:
            if isinstance(k_spec, str):
                if k_spec != "half":
                    raise ValueError(f"unsupported k spec {k_spec!r}; only 'half' is a string k.")
                k = n // 2  # floor(n / 2), per this region's own case count
            else:
                k = int(k_spec)

            # A k outside [1, n-1] cannot be drawn at all (draw_split would raise) -- this
            # is a registered, logged skip rather than aborting the whole sweep over every
            # other (region, k) combination.
            if not (1 <= k <= n - 1):
                logger.warning(
                    "run_local_recalibration: skipping k=%d for region=%s cohort=%s (n=%d) -- "
                    "must satisfy 1 <= k <= n-1.",
                    k,
                    region,
                    cohort,
                    n,
                )
                continue

            for split_index in range(n_splits):
                # Drawn ONCE per (region, k, split_index) and reused for every alpha below.
                calib_ids, heldout_ids = draw_split(case_ids, k, rng)
                for alpha in alphas:
                    results.append(
                        evaluate_split(
                            curves,
                            calib_ids,
                            heldout_ids,
                            alpha,
                            cohort=cohort,
                            region=region,
                            k=k,
                            split_index=split_index,
                        )
                    )
    return results


def min_feasible_k(curves: Sequence[CaseLossCurve], alpha: float, k_max: int) -> int | None:
    """Smallest calibration size `k` that COULD satisfy conformal risk control at `alpha`.

    Not a guarantee for any specific draw of `k` cases -- it treats the whole cohort's own
    mean miss rate at the smallest grid threshold, `R_min`, as a stand-in for what a
    `k`-sized calibration subset would see on average, then asks the same finite-sample
    question `fit_threshold` asks at that one threshold. `conformal.fit_threshold`'s
    selection rule (`conformal.py` lines 534-538) accepts a threshold `tau` when
    `(n * R_hat(tau) + B) / (n + 1) <= alpha`, `B = 1.0` the loss's bound. Substituting
    `n = k` and `R_hat(tau) = R_min` and solving for `k` gives

        k >= (B - alpha) / (alpha - R_min),

    valid whenever `R_min < alpha` -- if `R_min >= alpha` the left-hand side of the
    original inequality only grows with `k` (since `R_min` does not shrink by
    definition of this stand-in), so no calibration size, however large, ever helps.

    Args:
        curves: `CaseLossCurve`s to read the cohort's own `R_min` from --
            typically every case in the cohort, for one region.
        alpha: Target risk level.
        k_max: Largest `k` to consider.

    Returns:
        The smallest integer `k` in `[1, k_max]` satisfying the condition
        above, or `None` if `R_min >= alpha` (no k works) or the smallest
        working `k` exceeds `k_max`.

    Raises:
        ValueError: Every curve is `empty_gt`, so `R_min` is undefined.
    """
    used = [c for c in curves if not c.empty_gt]
    if not used:
        raise ValueError("all curves have empty ground truth; R_min is undefined.")
    r_min = float(np.mean([c.miss_rate()[0] for c in used]))

    if r_min >= alpha:
        return None

    ratio = (_LOSS_BOUND - alpha) / (alpha - r_min)
    # ratio can land a hair above an exact integer purely from floating-point rounding
    # (e.g. 0.7 / 0.1 computing as 7.000000000000001) -- nudge down before ceiling so that
    # an exact-equality case is not bumped up to k+1 by rounding noise.
    k_needed = max(1, math.ceil(ratio - 1e-9))
    if k_needed > k_max:
        return None
    return k_needed


def summarise(results: Sequence[SplitResult]) -> pd.DataFrame:
    """Aggregates split results into one row per (cohort, region, alpha, k).

    Args:
        results: `SplitResult`s from one or more `run_local_recalibration` calls.

    Returns:
        A `pandas.DataFrame` with one row per distinct `(cohort, region, alpha, k)`
        and columns `n_splits`, `feasible_rate`, `mean_threshold`,
        `mean_realised_risk`, `p_violation`, `mean_inflation`,
        `realised_risk_ci_low`, `realised_risk_ci_high` (the 2.5/97.5
        percentiles of `realised_risk` across feasible splits). Every
        feasible-only column is `NaN` for a group with no feasible split;
        `feasible_rate` is `0.0` in that case, not `NaN`, since it is
        well-defined regardless.
    """
    groups: dict[tuple[str, str, float, int], list[SplitResult]] = {}
    for r in results:
        groups.setdefault((r.cohort, r.region, r.alpha, r.k), []).append(r)

    rows: list[dict[str, object]] = []
    for (cohort, region, alpha, k), group in groups.items():
        n_splits = len(group)
        feasible_rows = [r for r in group if r.feasible]
        feasible_rate = len(feasible_rows) / n_splits

        thresholds = [r.threshold for r in feasible_rows if r.threshold is not None]
        risks = [r.realised_risk for r in feasible_rows if r.realised_risk is not None]
        inflations = [r.inflation for r in feasible_rows if r.inflation is not None]

        mean_threshold = float(np.mean(thresholds)) if thresholds else float("nan")
        mean_realised_risk = float(np.mean(risks)) if risks else float("nan")
        mean_inflation = float(np.mean(inflations)) if inflations else float("nan")
        # Denominator is every FEASIBLE split (per spec), not just the ones with a defined
        # `violated` -- a feasible split whose held-out half happened to be all-empty-GT
        # (violated=None) counts toward the denominator but not the numerator.
        p_violation = (
            sum(1 for r in feasible_rows if r.violated is True) / len(feasible_rows)
            if feasible_rows
            else float("nan")
        )
        if risks:
            ci_low, ci_high = (float(x) for x in np.percentile(risks, [2.5, 97.5]))
        else:
            ci_low, ci_high = float("nan"), float("nan")

        rows.append(
            {
                "cohort": cohort,
                "region": region,
                "alpha": alpha,
                "k": k,
                "n_splits": n_splits,
                "feasible_rate": feasible_rate,
                "mean_threshold": mean_threshold,
                "mean_realised_risk": mean_realised_risk,
                "p_violation": p_violation,
                "mean_inflation": mean_inflation,
                "realised_risk_ci_low": ci_low,
                "realised_risk_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
