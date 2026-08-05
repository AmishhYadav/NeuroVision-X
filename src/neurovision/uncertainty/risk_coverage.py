"""Risk-coverage analysis for selective prediction.

This module produces the clinical argument of the paper: if the model is
allowed to REFER its most uncertain cases to a human radiologist rather than
report on them, does quality on the cases it KEEPS improve? That is
"selective prediction", and the standard tool for showing it is a
risk-coverage curve -- retain the `k` most-confident cases (by whatever
uncertainty scalar the model produces), plot mean quality against the
fraction of cases retained ("coverage"), and compare against two reference
curves: an omniscient ORACLE (the best any referral policy could possibly
do) and a RANDOM referral policy (the null -- referral by an uncertainty
estimate that carries no information).

Unlike `neurovision.uncertainty.calibration` and `neurovision.inference.
mc_dropout`, which work on per-VOXEL probabilities and per-voxel uncertainty
maps, everything below works on PER-CASE SCALARS -- one uncertainty number
and one quality score (e.g. Dice, or HD95) per case. `case_uncertainty_
scalars` is the one bridge function that reduces a per-voxel uncertainty map
(such as `MCDropoutOutput.mutual_information`) down to that per-case scalar;
every other function in this module takes those scalars directly, typically
assembled by a caller looping `case_uncertainty_scalars` and
`neurovision.metrics.segmentation.compute_case_metrics` over an evaluation
split and joining the two per-case tables on `case_id`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import Tensor

from neurovision.data.transforms import REGION_NAMES

__all__ = [
    "case_uncertainty_scalars",
    "RiskCoverageCurve",
    "risk_coverage_curve",
    "oracle_curve",
    "random_curve",
    "bootstrap_curve_ci",
    "referral_table",
    "uncertainty_error_correlation",
]

logger = logging.getLogger(__name__)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of `y` over `x`, tolerant of the numpy trapz/trapezoid rename.

    `numpy.trapz` was removed in favour of `numpy.trapezoid` in recent numpy
    releases (the pinned version in this project's environment has
    `trapezoid` but not `trapz`), so this tries the new name first and falls
    back to the old one rather than hardcoding either.
    """
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


# ---------------------------------------------------------------------------
# Per-voxel uncertainty map -> per-case scalar
# ---------------------------------------------------------------------------


def _to_4d(x: Tensor | np.ndarray, name: str) -> Tensor:
    """Casts to a CPU float32 tensor and squeezes an optional leading batch axis of 1.

    Args:
        x: `(C, D, H, W)` or `(1, C, D, H, W)` array/tensor.
        name: Used in error messages.

    Returns:
        `(C, D, H, W)` float32 CPU tensor.

    Raises:
        ValueError: `x` has an ndim outside `{4, 5}`, or ndim 5 with a batch
            size other than 1.
    """
    t = torch.as_tensor(x).detach().to(dtype=torch.float32, device="cpu")
    if t.ndim == 5:
        if t.shape[0] != 1:
            raise ValueError(f"{name} is for a single case (batch size 1), got batch {t.shape[0]}.")
        t = t[0]
    if t.ndim != 4:
        raise ValueError(
            f"{name} expects shape (C, D, H, W) or (1, C, D, H, W), got {tuple(x.shape)}."  # type: ignore[union-attr]
        )
    return t


def case_uncertainty_scalars(
    mutual_information: Tensor | np.ndarray,
    predictive_entropy: Tensor | np.ndarray,
    mask: Tensor | np.ndarray | None = None,
    region_names: Sequence[str] = REGION_NAMES,
) -> dict[str, float]:
    """Reduces per-voxel MC-dropout uncertainty maps to per-case scalars.

    The mask is not optional in practice. The whole-volume mean of an
    uncertainty map is dominated by the ~99% of background voxels the model
    is trivially certain about, so an unmasked mean tracks TUMOR SIZE rather
    than case difficulty -- two cases that are equally hard for the model but
    have differently sized tumors would get different "uncertainty" purely
    from that size difference. Restricting the mean to the union of
    predicted-positive and ground-truth-positive voxels (the caller's
    responsibility to build and pass as `mask`) is what makes the resulting
    scalar mean "how unsure was the model where it actually mattered".

    Both `mutual_information` (epistemic -- uncertainty from the model's own
    parameters) and `predictive_entropy` (total -- epistemic + aleatoric) are
    reduced here because which one ranks cases better for referral is an
    empirical question, not something this module should quietly decide.
    Report in the paper which one `risk_coverage_curve` was actually sorted
    by (`unc_mi_<region>`/`unc_mi_mean` vs `unc_pe_<region>`/`unc_pe_mean`).

    Args:
        mutual_information: `MCDropoutOutput.mutual_information`-shaped,
            `(C, D, H, W)` or `(1, C, D, H, W)`. `C` must equal
            `len(region_names)`.
        predictive_entropy: Same shape convention as `mutual_information`.
        mask: Optional boolean-like mask. Either per-region
            (`(C, D, H, W)`/`(1, C, D, H, W)`, sliced alongside the maps) or
            shared across regions (e.g. `(D, H, W)`, broadcast to each
            region in turn) -- distinguished by whether its leading
            (post-batch) axis equals `len(region_names)`, matching
            `CalibrationAccumulator.add_case`'s convention. `None` means no
            masking (the whole-volume mean, which the paragraph above warns
            against using as the reported number).
        region_names: Region name per channel, in channel order. Defaults to
            `REGION_NAMES` (`("ET", "TC", "WT")`).

    Returns:
        A flat dict: `unc_mi_<REGION>`, `unc_pe_<REGION>` per region (mean
        over the masked voxels), plus NaN-skipping `unc_mi_mean`,
        `unc_pe_mean`. A region with zero masked voxels gets NaN for both of
        its entries (logged as a warning), never raises.

    Raises:
        ValueError: Shape mismatch between `mutual_information` and
            `predictive_entropy`, or channel count does not equal
            `len(region_names)`.
    """
    mi = _to_4d(mutual_information, "mutual_information")
    pe = _to_4d(predictive_entropy, "predictive_entropy")
    if mi.shape != pe.shape:
        raise ValueError(
            "case_uncertainty_scalars: mutual_information and predictive_entropy must have the "
            f"same shape, got {tuple(mi.shape)} and {tuple(pe.shape)}."
        )
    if mi.shape[0] != len(region_names):
        raise ValueError(
            f"case_uncertainty_scalars: maps have {mi.shape[0]} channels but region_names has "
            f"{len(region_names)} entries {tuple(region_names)}. One channel per region name is "
            "required."
        )

    mask_t: Tensor | None = None
    if mask is not None:
        mask_t = torch.as_tensor(mask).detach().to(dtype=torch.bool, device="cpu")
        if mask_t.ndim == 5:
            mask_t = mask_t[0]
    per_region_mask = (
        mask_t is not None and mask_t.ndim == 4 and mask_t.shape[0] == len(region_names)
    )

    scalars: dict[str, float] = {}
    mi_values: list[float] = []
    pe_values: list[float] = []
    for i, region in enumerate(region_names):
        region_mask = mask_t[i] if per_region_mask else mask_t
        if region_mask is None:
            mi_selected = mi[i].reshape(-1)
            pe_selected = pe[i].reshape(-1)
        else:
            region_mask = torch.broadcast_to(region_mask, mi[i].shape)
            mi_selected = mi[i][region_mask]
            pe_selected = pe[i][region_mask]

        if mi_selected.numel() == 0:
            logger.warning(
                "case_uncertainty_scalars: region %s has zero masked voxels; its scalars are NaN.",
                region,
            )
            scalars[f"unc_mi_{region}"] = float("nan")
            scalars[f"unc_pe_{region}"] = float("nan")
        else:
            scalars[f"unc_mi_{region}"] = float(mi_selected.mean())
            scalars[f"unc_pe_{region}"] = float(pe_selected.mean())
        mi_values.append(scalars[f"unc_mi_{region}"])
        pe_values.append(scalars[f"unc_pe_{region}"])

    scalars["unc_mi_mean"] = _nanmean_or_nan(mi_values)
    scalars["unc_pe_mean"] = _nanmean_or_nan(pe_values)
    return scalars


def _nanmean_or_nan(values: Sequence[float]) -> float:
    """`np.nanmean`, but returns NaN (silently) instead of warning on an all-NaN input."""
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


# ---------------------------------------------------------------------------
# Risk-coverage curve
# ---------------------------------------------------------------------------


@dataclass
class RiskCoverageCurve:
    """One risk-coverage curve, evaluated at every achievable coverage level.

    Attributes:
        coverage: `(N,)` float, `k / N` for `k = 1..N`, ascending.
        n_retained: `(N,)` int, the `k` behind each `coverage` entry.
        performance: `(N,)` float, mean `score` over the `k` retained cases
            (the `k` cases with the lowest uncertainty). This is what the
            paper's figure plots on the y-axis.
        risk: `(N,)` float. `1 - performance` when `higher_is_better=True`
            (so for a [0, 1]-bounded, mean-aggregated score like Dice this is
            an exact complement of `performance`, kept as a separate field
            purely for readability -- "risk" is the selective-prediction
            literature's term, "performance" is the paper's y-axis).
            `performance` itself when `higher_is_better=False` (e.g. HD95,
            where lower is already "risk").
        aurc: Area under the RISK-vs-coverage curve (`np.trapezoid(risk,
            coverage)`). LOWER is better. Only comparable between two curves
            computed over the SAME case set -- see `risk_coverage_curve`.
        n_cases: Number of usable cases (`N`, after dropping NaNs).
        n_dropped: Number of cases dropped for a NaN in `uncertainty` or
            `score`.
    """

    coverage: np.ndarray
    n_retained: np.ndarray
    performance: np.ndarray
    risk: np.ndarray
    aurc: float
    n_cases: int
    n_dropped: int


def _drop_nan_pairs(
    uncertainty: np.ndarray, score: np.ndarray, caller: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Shared NaN-dropping + length validation for the curve-building functions.

    Args:
        uncertainty: `(M,)` float array.
        score: `(M,)` float array.
        caller: Name used in error/log messages.

    Returns:
        `(uncertainty_clean, score_clean, n_dropped)`.

    Raises:
        ValueError: Length mismatch, empty input, or fewer than 2 usable
            cases after dropping NaNs.
    """
    if uncertainty.shape != score.shape:
        raise ValueError(
            f"{caller}: uncertainty and score must have the same shape, got "
            f"{uncertainty.shape} and {score.shape}."
        )
    if uncertainty.size == 0:
        raise ValueError(f"{caller}: uncertainty/score are empty; nothing to compute a curve over.")

    finite = ~(np.isnan(uncertainty) | np.isnan(score))
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        logger.warning(
            "%s: dropping %d/%d cases with a NaN uncertainty or score value.",
            caller,
            n_dropped,
            uncertainty.size,
        )
    unc_clean = uncertainty[finite]
    score_clean = score[finite]
    if unc_clean.size < 2:
        raise ValueError(
            f"{caller}: only {unc_clean.size} usable case(s) after dropping NaNs; need at "
            "least 2 to build a risk-coverage curve."
        )
    return unc_clean, score_clean, n_dropped


def _build_curve(
    ranking_signal: np.ndarray,
    score: np.ndarray,
    higher_is_better: bool,
    n_dropped: int,
) -> RiskCoverageCurve:
    """Shared curve construction: sort by `ranking_signal` ascending, then accumulate.

    Args:
        ranking_signal: `(N,)` float, already NaN-free. Cases are retained
            most-confident-by-this-signal first, i.e. ascending order.
        score: `(N,)` float, already NaN-free, same order as `ranking_signal`.
        higher_is_better: See `risk_coverage_curve`.
        n_dropped: Forwarded into the returned dataclass.

    Returns:
        A populated `RiskCoverageCurve`.
    """
    n = ranking_signal.size
    # kind="stable": ties in the ranking signal are common in this project
    # (e.g. many cases sitting at Dice exactly 1.0 when the oracle ranks by
    # score itself), and a stable sort keeps tied cases in their original
    # input order rather than in an order that depends on numpy's internal
    # sort implementation -- an unstable sort would make the curve silently
    # non-reproducible across numpy versions for any input with ties.
    order = np.argsort(ranking_signal, kind="stable")
    sorted_score = score[order]

    cumulative_sum = np.cumsum(sorted_score)
    n_retained = np.arange(1, n + 1)
    coverage = n_retained / n
    performance = cumulative_sum / n_retained

    risk = (1.0 - performance) if higher_is_better else performance
    aurc = _trapz(risk, coverage)

    return RiskCoverageCurve(
        coverage=coverage,
        n_retained=n_retained,
        performance=performance,
        risk=risk,
        aurc=aurc,
        n_cases=n,
        n_dropped=n_dropped,
    )


def risk_coverage_curve(
    uncertainty: Sequence[float] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    higher_is_better: bool = True,
) -> RiskCoverageCurve:
    """Builds the risk-coverage curve for a per-case uncertainty estimate.

    Sorts cases ascending by `uncertainty` (most confident first) and, for
    `k = 1..N`, retains the `k` most confident cases: `coverage[k-1] = k / N`,
    `performance[k-1] = mean(score of those k)`. See `RiskCoverageCurve` for
    the `risk` and `aurc` conventions.

    `aurc` is only meaningfully comparable between two curves computed over
    the SAME case set: it depends on both the coverage grid (which depends on
    `N`) and the score distribution, so a lower AURC on a different, easier
    evaluation split is not evidence of a better uncertainty estimate.

    Args:
        uncertainty: `(M,)` per-case uncertainty scalar. Any NaN entries (and
            the paired `score` entry) are dropped before building the curve.
        score: `(M,)` per-case quality score, e.g. Dice or HD95.
        higher_is_better: True for a score where higher is better (Dice,
            IoU) -- `risk = 1 - performance`. False for a score where lower
            is better (HD95, ECE) -- `risk = performance` directly. The score
            itself is NEVER negated; only which field (`risk` vs
            `performance`) is treated as "the thing to minimize" changes.

    Returns:
        A `RiskCoverageCurve`.

    Raises:
        ValueError: `uncertainty`/`score` length mismatch, empty input, or
            fewer than 2 usable cases after dropping NaNs.
    """
    unc = np.asarray(uncertainty, dtype=np.float64)
    sc = np.asarray(score, dtype=np.float64)
    unc_clean, score_clean, n_dropped = _drop_nan_pairs(unc, sc, "risk_coverage_curve")
    return _build_curve(unc_clean, score_clean, higher_is_better, n_dropped)


def oracle_curve(
    score: Sequence[float] | np.ndarray, higher_is_better: bool = True
) -> RiskCoverageCurve:
    """The referral ceiling: sorted by the TRUE score, not by any uncertainty estimate.

    No uncertainty estimate can beat this curve -- it is what an omniscient
    referral policy (one that already knows each case's true quality) would
    achieve. Implemented by delegating to `risk_coverage_curve` with the
    negated score as the ranking signal (best cases get the LOWEST ranking
    value, so they are the ones "most confidently" retained first) when
    `higher_is_better`, or the score itself otherwise (already "lower is
    better" ranking, matching a real uncertainty estimate's convention).
    Documented here as a deliberate reuse: both `oracle_curve` and
    `risk_coverage_curve` build the curve through the exact same
    `_build_curve` code path, so they cannot silently drift apart.

    Args:
        score: `(M,)` per-case quality score.
        higher_is_better: See `risk_coverage_curve`.

    Returns:
        A `RiskCoverageCurve`.

    Raises:
        ValueError: See `risk_coverage_curve`.
    """
    sc = np.asarray(score, dtype=np.float64)
    ranking_signal = -sc if higher_is_better else sc
    unc_clean, score_clean, n_dropped = _drop_nan_pairs(ranking_signal, sc, "oracle_curve")
    return _build_curve(unc_clean, score_clean, higher_is_better, n_dropped)


def random_curve(
    score: Sequence[float] | np.ndarray, higher_is_better: bool = True
) -> RiskCoverageCurve:
    """The referral null: a FLAT line at `mean(score)`, computed analytically.

    The expectation of the mean of a uniformly random subset of a finite
    population equals the population mean, at every subset size -- so a
    random referral policy's EXPECTED performance is exactly `mean(score)`
    at every coverage level, not just at coverage 1.0. This function returns
    that expectation directly rather than drawing one random permutation and
    reporting its (noisy) curve.

    This is loud on purpose: a plotted "random baseline" that is NOT flat
    means someone computed it by sampling a single random permutation
    instead of taking the expectation, and the resulting wiggle will be
    misread as a real difference in performance across coverage levels. The
    sampling variability random referral would actually show is exactly what
    `bootstrap_curve_ci` quantifies -- use that if a variance band around
    this flat line is wanted, do not replace this function's analytic value
    with one draw.

    Args:
        score: `(M,)` per-case quality score.
        higher_is_better: See `risk_coverage_curve`.

    Returns:
        A `RiskCoverageCurve` whose `performance` is `mean(score)` at every
        coverage level and whose `n_retained`/`coverage` grid matches
        `risk_coverage_curve`'s (`k / N` for `k = 1..N`).

    Raises:
        ValueError: See `risk_coverage_curve` (empty input, fewer than 2
            usable cases after NaN removal).
    """
    sc = np.asarray(score, dtype=np.float64)
    finite = ~np.isnan(sc)
    n_dropped = int((~finite).sum())
    if sc.size == 0:
        raise ValueError("random_curve: score is empty; nothing to compute a curve over.")
    if n_dropped > 0:
        logger.warning(
            "random_curve: dropping %d/%d cases with a NaN score value.", n_dropped, sc.size
        )
    sc_clean = sc[finite]
    n = sc_clean.size
    if n < 2:
        raise ValueError(
            f"random_curve: only {n} usable case(s) after dropping NaNs; need at least 2."
        )

    mean_score = float(np.mean(sc_clean))
    n_retained = np.arange(1, n + 1)
    coverage = n_retained / n
    performance = np.full(n, mean_score, dtype=np.float64)
    risk = np.full(n, 1.0 - mean_score if higher_is_better else mean_score, dtype=np.float64)
    aurc = _trapz(risk, coverage)

    return RiskCoverageCurve(
        coverage=coverage,
        n_retained=n_retained,
        performance=performance,
        risk=risk,
        aurc=aurc,
        n_cases=n,
        n_dropped=n_dropped,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence band
# ---------------------------------------------------------------------------


def bootstrap_curve_ci(
    uncertainty: Sequence[float] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    generator: np.random.Generator | None = None,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Bootstraps a confidence band around `risk_coverage_curve`.

    Exists because the curve is noisiest exactly where it looks most
    impressive: at low coverage the retained set is tiny (coverage 0.1 of a
    189-case split retains only 19 cases), so a curve plotted without a
    confidence band overstates how reliable the apparent gain over the
    random baseline is at that end.

    Resamples CASES with replacement `n_boot` times (each replicate has the
    same `N` as the input, so every replicate's coverage grid is `k / N` for
    `k = 1..N`, identical across replicates and identical to the
    un-resampled point estimate's grid), recomputes `risk_coverage_curve` on
    each replicate, and summarizes the `n_boot` resulting curves at each
    coverage level.

    Args:
        uncertainty: `(M,)` per-case uncertainty scalar.
        score: `(M,)` per-case quality score.
        n_boot: Number of bootstrap replicates.
        ci: Confidence level, e.g. `0.95` for a 95% interval (percentiles at
            `2.5` and `97.5`).
        generator: An explicit `np.random.Generator` (e.g.
            `np.random.default_rng(seed)`) -- REQUIRED, no default and no use
            of the global RNG, so two calls with independently, identically
            seeded generators reproduce exactly.
        higher_is_better: See `risk_coverage_curve`.

    Returns:
        A `DataFrame` indexed by `coverage` (the un-resampled point
        estimate's coverage grid), columns `performance, lo, hi, risk,
        risk_lo, risk_hi`. `performance`/`risk` are the point estimate from
        the un-resampled data (i.e. `risk_coverage_curve(uncertainty,
        score, ...)`'s own fields); `lo`/`hi`/`risk_lo`/`risk_hi` are the
        `(1 - ci) / 2` and `1 - (1 - ci) / 2` percentiles of `performance`/
        `risk` across the `n_boot` replicates, at each coverage level.

    Raises:
        ValueError: `generator` is `None`. Also propagated from
            `risk_coverage_curve` for the un-resampled point estimate (length
            mismatch, empty input, fewer than 2 usable cases after NaN
            removal).
    """
    if generator is None:
        raise ValueError(
            "bootstrap_curve_ci requires an explicit np.random.Generator (e.g. "
            "np.random.default_rng(seed)) -- no default and no use of the global RNG."
        )

    point = risk_coverage_curve(uncertainty, score, higher_is_better=higher_is_better)
    unc = np.asarray(uncertainty, dtype=np.float64)
    sc = np.asarray(score, dtype=np.float64)
    finite = ~(np.isnan(unc) | np.isnan(sc))
    unc_clean = unc[finite]
    sc_clean = sc[finite]
    n = unc_clean.size

    boot_performance = np.empty((n_boot, n), dtype=np.float64)
    boot_risk = np.empty((n_boot, n), dtype=np.float64)
    for b in range(n_boot):
        idx = generator.integers(0, n, size=n)  # resample CASES with replacement
        curve = _build_curve(unc_clean[idx], sc_clean[idx], higher_is_better, n_dropped=0)
        boot_performance[b] = curve.performance
        boot_risk[b] = curve.risk

    lower_pct = 100.0 * (1.0 - ci) / 2.0
    upper_pct = 100.0 - lower_pct
    perf_lo, perf_hi = np.percentile(boot_performance, [lower_pct, upper_pct], axis=0)
    risk_lo, risk_hi = np.percentile(boot_risk, [lower_pct, upper_pct], axis=0)

    return pd.DataFrame(
        {
            "performance": point.performance,
            "lo": perf_lo,
            "hi": perf_hi,
            "risk": point.risk,
            "risk_lo": risk_lo,
            "risk_hi": risk_hi,
        },
        index=pd.Index(point.coverage, name="coverage"),
    )


# ---------------------------------------------------------------------------
# The paper's referral table
# ---------------------------------------------------------------------------


def referral_table(
    uncertainty: Sequence[float] | np.ndarray,
    score: Sequence[float] | np.ndarray,
    coverage_points: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """The model-vs-oracle-vs-random comparison table for the paper.

    Each requested coverage point is snapped to the nearest ACHIEVABLE
    `k / N` (an `N`-case set cannot retain exactly 90% of 189 cases -- `0.9 *
    189 = 170.1`), by rounding to the nearest integer `k` in `[1, N]`. The
    achieved coverage and `n_retained` are reported alongside the requested
    one, not silently substituted for it.

    Args:
        uncertainty: `(M,)` per-case uncertainty scalar.
        score: `(M,)` per-case quality score.
        coverage_points: Requested coverage fractions in `(0, 1]`.
        higher_is_better: See `risk_coverage_curve`.

    Returns:
        A `DataFrame`, one row per requested coverage point, columns
        `coverage` (achieved), `n_retained`, `model`, `oracle`, `random`,
        `gain_over_random` (`model - random`). `model`/`oracle`/`random` are
        `performance` (not `risk`) at the achieved coverage. The row for
        coverage 1.0 has `model == oracle == random`, since retaining every
        case is the same set regardless of ranking -- a strong end-to-end
        sanity check on this table.

    Raises:
        ValueError: Propagated from `risk_coverage_curve`/`oracle_curve`/
            `random_curve` (length mismatch, empty input, fewer than 2
            usable cases after NaN removal). Also if any `coverage_points`
            entry is outside `(0, 1]`.
    """
    for c in coverage_points:
        if not (0.0 < c <= 1.0):
            raise ValueError(f"referral_table: coverage_points entries must be in (0, 1], got {c}.")

    model = risk_coverage_curve(uncertainty, score, higher_is_better=higher_is_better)
    oracle = oracle_curve(score, higher_is_better=higher_is_better)
    random_ = random_curve(score, higher_is_better=higher_is_better)

    n = model.n_cases
    rows = []
    for requested in coverage_points:
        k = int(round(requested * n))
        k = max(1, min(n, k))  # snap into [1, N]
        idx = k - 1  # coverage grid is k=1..N at index k-1
        rows.append(
            {
                "coverage": model.coverage[idx],
                "n_retained": model.n_retained[idx],
                "model": model.performance[idx],
                "oracle": oracle.performance[idx],
                "random": random_.performance[idx],
                "gain_over_random": model.performance[idx] - random_.performance[idx],
            }
        )

    return pd.DataFrame(rows, index=pd.Index(list(coverage_points), name="requested_coverage"))


# ---------------------------------------------------------------------------
# Uncertainty-error correlation
# ---------------------------------------------------------------------------


def uncertainty_error_correlation(
    uncertainty: Sequence[float] | np.ndarray, score: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Spearman and Pearson correlation between per-case uncertainty and score.

    Spearman is the number to report. The relationship a useful uncertainty
    estimate should have with quality is MONOTONE (more uncertain -> worse),
    not necessarily linear, and Pearson would be dragged around by the heavy
    left tail of the Dice distribution this project's real data has (mean
    0.903, min 0.322 on the 189-case unet3d test split) -- a handful of very
    low-Dice outliers can swing a linear correlation coefficient without
    reflecting the rank relationship the referral argument actually needs.

    Args:
        uncertainty: `(M,)` per-case uncertainty scalar. NaN entries (and the
            paired `score` entry) are dropped, same convention as
            `risk_coverage_curve`.
        score: `(M,)` per-case quality score.

    Returns:
        `{"spearman": float, "spearman_p": float, "pearson": float,
        "pearson_p": float, "n": int}`, from `scipy.stats.spearmanr` /
        `scipy.stats.pearsonr` on the NaN-dropped pair. `n` is the number of
        usable cases.

    Raises:
        ValueError: `uncertainty`/`score` length mismatch, empty input, or
            fewer than 2 usable cases after dropping NaNs.
    """
    unc = np.asarray(uncertainty, dtype=np.float64)
    sc = np.asarray(score, dtype=np.float64)
    unc_clean, score_clean, _ = _drop_nan_pairs(unc, sc, "uncertainty_error_correlation")

    spearman = stats.spearmanr(unc_clean, score_clean)
    pearson = stats.pearsonr(unc_clean, score_clean)

    return {
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n": int(unc_clean.size),
    }
