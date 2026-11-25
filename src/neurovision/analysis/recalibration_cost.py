"""The price of local recalibration: how many labelled cases, what mask, how many hours.

Implements `docs/research_docs/protocols/recalibration_cost_protocol.md` (MasterPlan W3.5,
DESCRIPTIVE / EXPLORATORY). It extends the CRC-only counterfactual in
`neurovision.analysis.local_recalibration` (C24) to both v2 fitters and also records what the
recalibrated mask looks like.

## The two guarantees being priced

Both fitters pick a probability threshold `tau`; a voxel is in the mask when its probability is
>= `tau`. A LOWER `tau` gives a BIGGER mask and so fewer missed tumour voxels. The miss rate is
therefore non-decreasing in `tau`, and both fitters return the LARGEST (tightest) `tau` that still
passes their test.

- **CRC** (conformal risk control): the EXPECTED miss rate on a fresh exchangeable case is <= alpha.
  An average-case promise, so it needs only a small finite-sample correction.
- **RCPS** (risk-controlling prediction sets, WSR bound): the miss rate is <= alpha with
  probability >= 1 - delta over the draw of the calibration set. A high-probability promise, so it
  needs an upper confidence bound on the risk and therefore MORE calibration cases, and for the
  same data it lands on a lower `tau` (bigger mask) or is infeasible.

Neither inequality is re-implemented here; this module calls `conformal.fit_threshold_crc` and
`conformal.fit_threshold_rcps` as the v2 driver (`scripts/conformal_v2.py`) does.

## Why splits are shared across methods

At each (cell, k) the random calibration/held-out splits are drawn once and reused for every alpha
and for both methods. So CRC vs RCPS (and alpha vs alpha) differ only in the rule applied to the
same cases, not in luck of the draw.

## Why the feasible rate matters

A split is "feasible" when the fitter finds any threshold at all. A site that recalibrates on k
cases and gets no threshold 40% of the time has not been served, even if the thresholds it does
get are well behaved. `k_needed` therefore requires a minimum feasible rate as well as RESTORED.

This is a counterfactual summarised over Monte Carlo splits, never a guarantee for a real site.

## Randomness and I/O

Every draw goes through the `numpy.random.Generator` the caller passes in. No file I/O: the
driver loads curves (`conformal.load_curves_npz`) and writes tables.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    CaseLossCurve,
    fit_threshold_crc,
    fit_threshold_rcps,
    mask_quality,
    realised_risk_with,
)

__all__ = [
    "CostSplitResult",
    "valid_ks",
    "restrict_to_grid",
    "evaluate_cost_split",
    "run_cost",
    "verdict",
    "summarise_cost",
    "k_needed",
    "annotation_hours",
]

logger = logging.getLogger(__name__)

_METHODS = ("crc", "rcps")
_QUALITY_KEYS = ("precision", "dice", "volume_ratio")
_CELL_KEYS = ["model", "cohort", "region", "alpha", "method"]


@dataclass(frozen=True)
class CostSplitResult:
    """One calibration/held-out split, fitted with one method and scored.

    Attributes:
        model: Model name.
        cohort: Cohort name (e.g. "test", "SSA", "PED").
        region: Region name ("WT", "TC", "ET").
        alpha: Target risk level.
        method: "crc" or "rcps".
        k: Number of calibration cases.
        split_index: Which of the shared random splits this is.
        feasible: Whether the fitter found any threshold on the calibration cases.
        threshold: Fitted threshold, or None if infeasible.
        realised_risk: Held-out mean miss rate at `threshold`; None if infeasible or undefined.
        violated: `realised_risk > alpha`; None wherever `realised_risk` is None.
        precision: Held-out median precision at `threshold`; None if infeasible or all NaN.
        dice: Held-out median Dice at `threshold`; None if infeasible or all NaN.
        volume_ratio: Held-out median predicted/true volume; None if infeasible or all NaN.
    """

    model: str
    cohort: str
    region: str
    alpha: float
    method: str
    k: int
    split_index: int
    feasible: bool
    threshold: float | None
    realised_risk: float | None
    violated: bool | None
    precision: float | None
    dice: float | None
    volume_ratio: float | None


def valid_ks(k_grid: Sequence[int | str], n: int, min_holdout: int) -> list[int]:
    """Expands the k grid for a cohort of `n` cases and drops sizes that leave too few held out.

    Args:
        k_grid: Ints, and/or the string "half" meaning `n // 2`.
        n: Number of cases in the cell.
        min_holdout: Minimum number of held-out cases; keeps only `k <= n - min_holdout`.

    Returns:
        Sorted, de-duplicated list of usable k (all >= 1).

    Raises:
        ValueError: A string entry other than "half".
    """
    ks: set[int] = set()
    for spec in k_grid:
        if isinstance(spec, str):
            if spec != "half":
                raise ValueError(f"unsupported k spec {spec!r}; only 'half' is a string k.")
            k = n // 2
        else:
            k = int(spec)
        if 1 <= k <= n - min_holdout:
            ks.add(k)
    return sorted(ks)


def restrict_to_grid(curves: Sequence[CaseLossCurve], grid: str) -> list[CaseLossCurve]:
    """Puts curves on the named threshold grid by picking columns (no recomputation).

    Args:
        curves: Curves on one shared grid (one region).
        grid: "extended" (keep the curves' own grid) or "v1" (restrict to
            `conformal.DEFAULT_THRESHOLDS`, as `scripts/conformal_v2.py` does for nnU-Net).

    Returns:
        Curves on the requested grid, same order.

    Raises:
        ValueError: Unknown `grid`, or a v1 threshold is missing from the curves' grid.
    """
    if grid == "extended":
        return list(curves)
    if grid != "v1":
        raise ValueError(f"grid must be 'extended' or 'v1', got {grid!r}.")
    if not curves or curves[0].thresholds == DEFAULT_THRESHOLDS:
        return list(curves)
    have = np.asarray(curves[0].thresholds, dtype=np.float64)
    idx = []
    for t in DEFAULT_THRESHOLDS:
        j = int(np.argmin(np.abs(have - t)))
        if abs(have[j] - t) > 1e-9:
            raise ValueError(f"v1 threshold {t} is not in the curves' grid.")
        idx.append(j)
    return [
        CaseLossCurve(
            case_id=c.case_id,
            region=c.region,
            gt_voxels=c.gt_voxels,
            thresholds=tuple(float(c.thresholds[i]) for i in idx),
            fn_voxels=tuple(c.fn_voxels[i] for i in idx),
            mask_voxels=tuple(c.mask_voxels[i] for i in idx),
        )
        for c in curves
    ]


def _median_or_none(values: np.ndarray) -> float | None:
    """NaN-aware median; None when every value is NaN."""
    finite = values[~np.isnan(values)]
    return float(np.median(finite)) if finite.size else None


def evaluate_cost_split(
    curves: Sequence[CaseLossCurve],
    cal_idx: Sequence[int],
    hold_idx: Sequence[int],
    *,
    alpha: float,
    method: str,
    delta: float,
    empty: str,
    grid: str = "extended",
    order_seed: int = 0,
    model: str = "",
    cohort: str = "",
    k: int | None = None,
    split_index: int = 0,
) -> CostSplitResult:
    """Fits a threshold on the calibration cases and scores it on the held-out cases.

    The held-out mask quality is the median over held-out cases of precision, Dice and volume
    ratio (`conformal.mask_quality`), ignoring NaN. NaN precision arises when the predicted mask
    is empty (0/0, no claim made); NaN volume ratio when the reference is empty. Dice is never NaN.
    Empty-reference cases are NOT dropped from the quality medians (same as the v2 driver).

    Args:
        curves: All cases of one cell (one region), one shared grid.
        cal_idx: Indices into `curves` used for fitting.
        hold_idx: Indices into `curves` used for scoring.
        alpha: Target risk level.
        method: "crc" (expected risk <= alpha) or "rcps" (risk <= alpha w.p. >= 1 - delta).
        delta: RCPS confidence parameter (ignored by CRC).
        empty: "zero" or "exclude" (empty-reference convention for this region).
        grid: "extended" or "v1"; curves are restricted to it if they are not already on it.
        order_seed: RCPS case-permutation seed (passed to `fit_threshold_rcps`).
        model: Stamped onto the result.
        cohort: Stamped onto the result.
        k: Calibration size stamped onto the result; defaults to `len(cal_idx)`.
        split_index: Stamped onto the result.

    Returns:
        A `CostSplitResult`; infeasible fits give None for every fitted/held-out field.

    Raises:
        ValueError: Unknown `method`.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    on_grid = restrict_to_grid(curves, grid)
    cal = [on_grid[i] for i in cal_idx]
    hold = [on_grid[i] for i in hold_idx]

    if method == "crc":
        fit = fit_threshold_crc(cal, alpha, empty=empty)
    else:
        fit = fit_threshold_rcps(cal, alpha, delta, order_seed=order_seed, empty=empty)

    base = dict(
        model=model,
        cohort=cohort,
        region=cal[0].region,
        alpha=alpha,
        method=method,
        k=len(cal_idx) if k is None else k,
        split_index=split_index,
    )
    if not fit.feasible or fit.threshold is None:
        return CostSplitResult(
            **base,
            feasible=False,
            threshold=None,
            realised_risk=None,
            violated=None,
            precision=None,
            dice=None,
            volume_ratio=None,
        )

    tau = fit.threshold
    mean_miss = realised_risk_with(hold, tau, empty=empty)["mean_miss_rate"]
    risk: float | None = None if np.isnan(mean_miss) else float(mean_miss)
    q = mask_quality(hold, tau)
    return CostSplitResult(
        **base,
        feasible=True,
        threshold=float(tau),
        realised_risk=risk,
        violated=None if risk is None else bool(risk > alpha),
        precision=_median_or_none(q["precision"]),
        dice=_median_or_none(q["dice"]),
        volume_ratio=_median_or_none(q["volume_ratio"]),
    )


def run_cost(
    curves_by_cell: Mapping[tuple[str, str, str], Sequence[CaseLossCurve]],
    *,
    alphas: Sequence[float],
    methods: Sequence[str],
    k_grid: Sequence[int | str],
    n_splits: int,
    min_holdout: int,
    delta: float,
    empty_by_region: Mapping[str, str],
    grid_by_model: Mapping[str, str],
    rng: np.random.Generator,
    rcps_order_seed: int = 0,
) -> list[CostSplitResult]:
    """Runs the recalibration-cost counterfactual over every cell, alpha, method and k.

    For each (cell, k), `n_splits` random calibration subsets of size k are drawn ONCE with
    `rng.permutation` and reused for every alpha and method (see the module docstring).

    Args:
        curves_by_cell: `{(model, cohort, region): curves}`, every case of that cell.
        alphas: Target risk levels.
        methods: Subset of ("crc", "rcps").
        k_grid: Calibration sizes (ints and/or "half").
        n_splits: Random splits per (cell, k).
        min_holdout: Minimum held-out cases per split (see `valid_ks`).
        delta: RCPS confidence parameter.
        empty_by_region: `{region: "zero" | "exclude"}`.
        grid_by_model: `{model: "extended" | "v1"}`.
        rng: The only source of randomness.
        rcps_order_seed: Seed for the RCPS case permutation inside the fitter.

    Returns:
        One `CostSplitResult` per (cell, alpha, method, k, split_index).
    """
    results: list[CostSplitResult] = []
    for (model, cohort, region), raw_curves in curves_by_cell.items():
        grid = grid_by_model[model]
        empty = empty_by_region[region]
        curves = restrict_to_grid(raw_curves, grid)  # once per cell, not once per split
        n = len(curves)
        ks = valid_ks(k_grid, n, min_holdout)
        logger.info("recalibration_cost: %s / %s / %s (n=%d, ks=%s)", model, cohort, region, n, ks)
        for k in ks:
            splits = [rng.permutation(n) for _ in range(n_splits)]
            for alpha in alphas:
                for method in methods:
                    for split_index, order in enumerate(splits):
                        results.append(
                            evaluate_cost_split(
                                curves,
                                order[:k].tolist(),
                                order[k:].tolist(),
                                alpha=alpha,
                                method=method,
                                delta=delta,
                                empty=empty,
                                grid=grid,
                                order_seed=rcps_order_seed,
                                model=model,
                                cohort=cohort,
                                k=k,
                                split_index=split_index,
                            )
                        )
    return results


def verdict(mean_risk: float, mc_se: float, alpha: float, n_feasible: int) -> str:
    """The C24 restoration rule (from `scripts/local_recalibration.classify_restoration`).

    Args:
        mean_risk: Mean held-out risk over feasible splits (NaN if none defined).
        mc_se: Monte Carlo standard error of that mean (NaN if < 2 defined risks).
        alpha: Target risk level.
        n_feasible: Number of feasible splits.

    Returns:
        "INFEASIBLE" if no split was feasible; "BORDERLINE" if fewer than 2 were; "RESTORED"
        if `mean_risk <= alpha` (equality counts); "NOT_RESTORED" if `mean_risk` exceeds alpha
        by more than 2 * `mc_se`; otherwise "BORDERLINE" (NaN also lands here).
    """
    if n_feasible == 0:
        return "INFEASIBLE"
    if n_feasible < 2:
        return "BORDERLINE"
    if mean_risk <= alpha:
        return "RESTORED"
    if mean_risk - alpha > 2.0 * mc_se:
        return "NOT_RESTORED"
    return "BORDERLINE"


def _nan_median(values: list[float | None]) -> float:
    """Median of the non-None values, NaN if there are none."""
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else float("nan")


def summarise_cost(results: Sequence[CostSplitResult]) -> pd.DataFrame:
    """Aggregates splits into one row per (model, cohort, region, alpha, method, k).

    Args:
        results: Output of `run_cost`.

    Returns:
        DataFrame with n_splits, feasible_rate, n_feasible, mean_threshold, mean_realised_risk,
        mc_se (sample std / sqrt(number of defined risks)), p_violation, verdict, and the median
        over feasible splits of the held-out median precision, dice and volume_ratio. Columns
        that need a feasible split are NaN without one; `feasible_rate` is then 0.0.
    """
    groups: dict[tuple, list[CostSplitResult]] = {}
    for r in results:
        groups.setdefault((r.model, r.cohort, r.region, r.alpha, r.method, r.k), []).append(r)

    rows = []
    for (model, cohort, region, alpha, method, k), grp in groups.items():
        feas = [r for r in grp if r.feasible]
        risks = np.asarray([r.realised_risk for r in feas if r.realised_risk is not None])
        thresholds = [r.threshold for r in feas if r.threshold is not None]
        mean_risk = float(risks.mean()) if risks.size else float("nan")
        mc_se = (
            float(np.std(risks, ddof=1) / math.sqrt(risks.size))
            if risks.size >= 2
            else float("nan")
        )
        rows.append(
            {
                "model": model,
                "cohort": cohort,
                "region": region,
                "alpha": alpha,
                "method": method,
                "k": k,
                "n_splits": len(grp),
                "feasible_rate": len(feas) / len(grp),
                "n_feasible": len(feas),
                "mean_threshold": float(np.mean(thresholds)) if thresholds else float("nan"),
                "mean_realised_risk": mean_risk,
                "mc_se": mc_se,
                "p_violation": (
                    sum(1 for r in feas if r.violated is True) / len(feas) if feas else float("nan")
                ),
                "verdict": verdict(mean_risk, mc_se, alpha, len(feas)),
                **{
                    f"{key}_median": _nan_median([getattr(r, key) for r in feas])
                    for key in _QUALITY_KEYS
                },
            }
        )
    return pd.DataFrame(rows)


def k_needed(summary: pd.DataFrame, *, min_feasible_rate: float) -> pd.DataFrame:
    """The smallest k that restores the bound, per (model, cohort, region, alpha, method).

    Args:
        summary: Output of `summarise_cost`.
        min_feasible_rate: Minimum feasible rate a k must reach (protocol: 0.90).

    Returns:
        One row per cell/alpha/method with `k_needed` (NaN if none), `restorable` (bool), and
        that row's feasible_rate, mean_realised_risk and precision/dice/volume_ratio medians
        (NaN when not restorable).
    """
    carry = ["feasible_rate", "mean_realised_risk"] + [f"{k}_median" for k in _QUALITY_KEYS]
    rows = []
    for key, grp in summary.groupby(_CELL_KEYS, sort=False):
        ok = grp[(grp["verdict"] == "RESTORED") & (grp["feasible_rate"] >= min_feasible_rate)]
        row = dict(zip(_CELL_KEYS, key, strict=True))
        if ok.empty:
            row.update(k_needed=float("nan"), restorable=False)
            row.update({c: float("nan") for c in carry})
        else:
            best = ok.sort_values("k").iloc[0]
            row.update(k_needed=float(best["k"]), restorable=True)
            row.update({c: float(best[c]) for c in carry})
        rows.append(row)
    return pd.DataFrame(rows)


def annotation_hours(k: float, minutes_per_case: float, speedup: float = 1.0) -> float:
    """Annotator hours to label `k` cases: `k * minutes_per_case / 60 / speedup`.

    Args:
        k: Number of cases (NaN propagates).
        minutes_per_case: Minutes to segment one case from scratch (protocol: 60).
        speedup: Factor by which correcting a pre-annotation is faster (protocol sensitivity: 2.21).

    Returns:
        Hours; NaN if `k` is NaN.
    """
    return float(k * minutes_per_case / 60.0 / speedup)
