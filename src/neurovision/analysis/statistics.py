"""Statistical significance testing for paired per-case model comparisons.

Every claim of the form "model A is better than model B" that ends up in the
paper must go through this module rather than being read off a bare mean
difference. It operates on PER-CASE metric tables -- the `per_case_metrics.
csv` written by `scripts/evaluate.py`, whose columns come from
`neurovision.metrics.segmentation.compute_case_metrics` (`dice_ET`,
`dice_TC`, `dice_WT`, `iou_*`, `hd95_*`, `gt_empty_*`, `dice_mean`,
`iou_mean`, `hd95_mean`; indexed by `case_id`) -- so a "significant"
difference here means significant across CASES, the actual unit of
generalization the paper is arguing about, not across voxels.

## Why paired, not two-sample

Model A and model B are evaluated on the SAME test cases, so their per-case
scores are correlated (an easy case is easy for both models). A two-sample
test throws that correlation away and is needlessly conservative; every
comparison here works on the per-case DIFFERENCE (`a - b`), which is exactly
what a paired bootstrap, a Wilcoxon signed-rank test, and Cohen's `dz`
(rather than an unpaired `d`) are built for.

## Why three different tools, not one

`paired_bootstrap_ci` answers "how big is the difference, and how
uncertain are we about that size" -- an estimation question, and the one a
reader actually wants an answer to. `wilcoxon_signed_rank` answers "is
there a difference at all", robustly to the same violated-normality issues
that make Dice awkward (it is bounded in `[0, 1]`, and many BraTS cases sit
exactly at 1.0 under `ignore_empty=False` -- see the note on `n_zero` below).
`paired_effect_size` puts a magnitude on the difference in units that do not
depend on the sample size, so a reader can tell a real-but-tiny effect from a
real-and-large one. None of the three subsumes the other two, so all three
are reported together in `compare_models`.

## Direction handling

A metric is either "higher is better" (Dice, IoU) or "lower is better"
(HD95, ECE), and `diff = a - b` alone does not encode that -- a positive
`dice_ET` diff means A is better, but a positive `hd95_WT` diff means A is
WORSE. `metric_direction` resolves this from a small table keyed by the
metric's prefix, and refuses to guess for anything not in that table:
silently defaulting an unrecognized error-type metric to "higher is better"
would flip its sign in every downstream table with no visible failure --
exactly the kind of silent inversion that would corrupt a paper claim.

## Multiple comparisons

Reporting Dice, IoU and HD95 for three regions is up to nine simultaneous
hypothesis tests per baseline comparison; at `alpha=0.05` uncorrected, that
family has a false-positive rate well above 5%. `holm_bonferroni` corrects
for this, and `compare_models` applies it once, across every metric in the
table it returns -- that table IS the declared family. Running
`compare_models` separately per metric and Holm-correcting each call alone
would defeat the correction entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "DEFAULT_ALPHA",
    "KNOWN_METRIC_DIRECTIONS",
    "metric_direction",
    "BootstrapResult",
    "WilcoxonResult",
    "EffectSize",
    "paired_bootstrap_ci",
    "wilcoxon_signed_rank",
    "paired_effect_size",
    "holm_bonferroni",
    "load_per_case",
    "compare_models",
    "format_comparison",
]

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05

# Prefix (the part of a metric name before its first "_") -> higher_is_better.
# Looked up case-insensitively by `metric_direction`. Deliberately does not
# include a catch-all default -- see the module docstring.
KNOWN_METRIC_DIRECTIONS: dict[str, bool] = {
    "dice": True,
    "iou": True,
    "auc": True,
    "auroc": True,
    "auprc": True,
    "coverage": True,
    "accuracy": True,
    "hd95": False,
    "hausdorff": False,
    "ece": False,
    "mce": False,
    "brier": False,
    "nll": False,
    "loss": False,
    "aurc": False,
    "risk": False,
}


def metric_direction(metric: str, overrides: Mapping[str, bool] | None = None) -> bool:
    """Resolves whether higher values of `metric` are better.

    Args:
        metric: A metric column name, e.g. `"dice_ET"` or `"hd95_WT"`.
            Looked up by its prefix (the part before the first `"_"`,
            case-insensitive) in `KNOWN_METRIC_DIRECTIONS`.
        overrides: Optional mapping keyed by the FULL metric name (not the
            prefix), e.g. `{"my_score": True}`. Checked first and wins over
            the table.

    Returns:
        True if higher is better, False if lower is better.

    Raises:
        ValueError: `metric` has no entry in `overrides` and its prefix is
            not in `KNOWN_METRIC_DIRECTIONS`. This is deliberate -- guessing
            a direction for an unrecognized metric could silently invert a
            reported comparison.
    """
    if overrides is not None and metric in overrides:
        return overrides[metric]
    prefix = metric.split("_")[0].lower()
    if prefix in KNOWN_METRIC_DIRECTIONS:
        return KNOWN_METRIC_DIRECTIONS[prefix]
    raise ValueError(
        f"metric_direction: unknown metric {metric!r} (prefix {prefix!r} is not in the known "
        f"direction table). Known prefixes: {sorted(KNOWN_METRIC_DIRECTIONS)}. Pass an explicit "
        "override keyed by the full metric name rather than guessing a direction."
    )


@dataclass(frozen=True)
class BootstrapResult:
    """The result of one `paired_bootstrap_ci` call.

    Attributes:
        point: The statistic (mean or median) of the un-resampled paired
            differences `a - b`.
        lo: Lower confidence bound.
        hi: Upper confidence bound.
        ci: The confidence level used, e.g. `0.95`.
        n: Number of paired cases actually used (after dropping NaN pairs).
        n_boot: Number of bootstrap replicates drawn.
        method: `"percentile"` or `"bca"`, or `"percentile (bca degenerate)"`
            when a requested BCa interval fell back to percentile.
        se: Bootstrap standard error, `std(replicate statistics, ddof=1)`.
        contains_zero: Whether `[lo, hi]` contains 0 -- i.e. whether "no
            difference" is inside the confidence interval.
    """

    point: float
    lo: float
    hi: float
    ci: float
    n: int
    n_boot: int
    method: str
    se: float
    contains_zero: bool


@dataclass(frozen=True)
class WilcoxonResult:
    """The result of one `wilcoxon_signed_rank` call.

    Attributes:
        statistic: The Wilcoxon signed-rank test statistic.
        pvalue: Two-sided (or one-sided, per `alternative`) p-value.
        n: Number of paired cases used (after dropping NaN pairs).
        n_zero: Number of EXACT ties (`a - b == 0`) among those `n` pairs.
        n_effective: `n - n_zero` when `zero_method="wilcox"` (scipy
            discards exact zeros before ranking), else `n`.
        rank_biserial: `(W+ - W-) / (W+ + W-)`, the effect size that pairs
            with this test. Positive means A scored higher more often /
            more strongly than B.
        alternative: The `alternative` argument used.
        zero_method: The `zero_method` argument used.
    """

    statistic: float
    pvalue: float
    n: int
    n_zero: int
    n_effective: int
    rank_biserial: float
    alternative: str
    zero_method: str


@dataclass(frozen=True)
class EffectSize:
    """The result of one `paired_effect_size` call.

    Attributes:
        mean_diff: `mean(a - b)`.
        sd_diff: `std(a - b, ddof=1)`.
        cohens_dz: `mean_diff / sd_diff`, the paired-samples Cohen's d.
        hedges_g: `cohens_dz` with the small-sample bias correction applied.
        rank_biserial: Same quantity, same convention, as
            `WilcoxonResult.rank_biserial`.
        n: Number of paired cases used.
        magnitude: `"negligible"` / `"small"` / `"medium"` / `"large"`, from
            Cohen's CONVENTIONAL cutoffs on `abs(cohens_dz)`. These cutoffs
            are generic rules of thumb, not evidence about this dataset --
            state the raw `cohens_dz` in the paper, not just the label.
    """

    mean_diff: float
    sd_diff: float
    cohens_dz: float
    hedges_g: float
    rank_biserial: float
    n: int
    magnitude: str


ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _clean_pair(a: ArrayLike, b: ArrayLike, caller: str) -> tuple[np.ndarray, np.ndarray]:
    """Casts to float64, validates matching length, and drops NaN pairs.

    Args:
        a: First per-case array.
        b: Second per-case array, same case order as `a`.
        caller: Name used in error/log messages.

    Returns:
        `(a_clean, b_clean)`, 1-D float64 numpy arrays with any pair holding
        a NaN on either side removed.

    Raises:
        ValueError: `a`/`b` length mismatch, or fewer than 2 pairs survive
            NaN removal.
    """
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"{caller}: a and b must have the same length, got {a_arr.size} and {b_arr.size}."
        )
    finite = ~(np.isnan(a_arr) | np.isnan(b_arr))
    n_dropped = int((~finite).sum())
    if n_dropped > 0:
        logger.debug(
            "%s: dropping %d/%d pair(s) with a NaN on either side.", caller, n_dropped, a_arr.size
        )
    a_clean = a_arr[finite]
    b_clean = b_arr[finite]
    if a_clean.size < 2:
        raise ValueError(
            f"{caller}: only {a_clean.size} usable pair(s) after dropping NaNs; need at least 2."
        )
    return a_clean, b_clean


def _rank_biserial_from_diff(diff: np.ndarray) -> float:
    """The Wilcoxon signed-rank effect size, computed once and reused by both callers.

    Shared by `wilcoxon_signed_rank` and `paired_effect_size` so the two
    cannot silently drift apart -- see the class-level docstring notes on
    `rank_biserial` in `WilcoxonResult` and `EffectSize`.

    Args:
        diff: `(n,)` float64 paired differences (`a - b`), may include zeros.

    Returns:
        `(W+ - W-) / (W+ + W-)`, computed over the NONZERO entries of `diff`
        (ranks assigned to `abs(diff)` among those nonzero entries only,
        matching scipy's own `zero_method="wilcox"` convention). `0.0` if
        every entry is zero or the nonzero entries happen to cancel exactly.
    """
    nonzero = diff[diff != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos = float(np.sum(ranks[nonzero > 0]))
    w_neg = float(np.sum(ranks[nonzero < 0]))
    total = w_pos + w_neg
    return (w_pos - w_neg) / total if total > 0 else 0.0


def _bca_bounds(
    diff: np.ndarray, boot: np.ndarray, point: float, statistic: str, ci: float
) -> tuple[float, float] | None:
    """Bias-corrected-and-accelerated interval bounds, or `None` if degenerate.

    Args:
        diff: `(n,)` float64 un-resampled paired differences.
        boot: `(n_boot,)` float64 bootstrap replicate statistics.
        point: The statistic on `diff` itself.
        statistic: `"mean"` or `"median"`, used for the jackknife.
        ci: Confidence level.

    Returns:
        `(lo, hi)`, or `None` if any of the degenerate cases the BCa
        derivation cannot handle is hit (every replicate equal to `point`,
        a non-finite bias-correction term, a zero-variance jackknife, or an
        adjusted percentile outside `[0, 1]`). Callers must fall back to
        the percentile method and log a warning in that case.
    """
    stat_fn = np.mean if statistic == "mean" else np.median

    prop_less = float(np.mean(boot < point))
    z0 = stats.norm.ppf(prop_less)
    if not np.isfinite(z0):
        return None

    n = diff.size
    jack = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack[i] = stat_fn(np.delete(diff, i))
    d = jack.mean() - jack
    sum_d2 = float(np.sum(d**2))
    if sum_d2 == 0.0:
        return None
    a_hat = float(np.sum(d**3)) / (6.0 * sum_d2**1.5)

    alpha = 1.0 - ci
    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)

    def _adjusted_percentile(z_alpha: float) -> float | None:
        denom = 1.0 - a_hat * (z0 + z_alpha)
        if denom == 0.0:
            return None
        p = float(stats.norm.cdf(z0 + (z0 + z_alpha) / denom))
        if not (0.0 <= p <= 1.0):
            return None
        return p

    p_lo = _adjusted_percentile(z_lo)
    p_hi = _adjusted_percentile(z_hi)
    if p_lo is None or p_hi is None:
        return None

    lo, hi = np.percentile(boot, [100.0 * p_lo, 100.0 * p_hi])
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    generator: np.random.Generator | None = None,
    method: str = "percentile",
    statistic: str = "mean",
) -> BootstrapResult:
    """Bootstraps a confidence interval on the paired difference `a - b`.

    Resamples CASE INDICES with replacement (`idx = generator.integers(0, n,
    size=(n_boot, n))`), then indexes the paired-difference array with those
    same indices for every replicate. Because both sides of a pair are
    resampled together (by resampling one index array into `diff = a - b`
    rather than resampling `a` and `b` independently), the pairing between a
    case's A-score and B-score is preserved in every replicate -- this is
    what makes the procedure a PAIRED bootstrap rather than an unpaired one
    that happens to share a sample size.

    Args:
        a: `(n,)` per-case values for model A.
        b: `(n,)` per-case values for model B, same case order as `a`.
        n_boot: Number of bootstrap replicates.
        ci: Confidence level, e.g. `0.95` for a 95% interval.
        generator: An explicit `np.random.Generator` (e.g.
            `np.random.default_rng(seed)`) -- REQUIRED, no default and no
            use of the global RNG.
        method: `"percentile"` (default) or `"bca"` (bias-corrected and
            accelerated). A degenerate BCa case falls back to percentile
            with a warning; see `BootstrapResult.method`.
        statistic: `"mean"` (default) or `"median"`.

    Returns:
        A `BootstrapResult`.

    Raises:
        ValueError: `generator` is `None`; `statistic` or `method` is not a
            recognized value; `a`/`b` length mismatch; or fewer than 2
            pairs survive NaN removal.
    """
    if generator is None:
        raise ValueError(
            "paired_bootstrap_ci requires an explicit np.random.Generator (e.g. "
            "np.random.default_rng(seed)) -- no default and no use of the global RNG."
        )
    if statistic not in ("mean", "median"):
        raise ValueError(
            f"paired_bootstrap_ci: statistic must be 'mean' or 'median', got {statistic!r}."
        )
    if method not in ("percentile", "bca"):
        raise ValueError(
            f"paired_bootstrap_ci: method must be 'percentile' or 'bca', got {method!r}."
        )

    a_clean, b_clean = _clean_pair(a, b, "paired_bootstrap_ci")
    n = a_clean.size
    if n < 20:
        logger.warning(
            "paired_bootstrap_ci: only %d paired case(s) -- a bootstrap CI over this few cases "
            "is not reliable and must not be reported as if it were.",
            n,
        )

    diff = a_clean - b_clean
    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(diff))

    idx = generator.integers(0, n, size=(n_boot, n))  # resample CASE INDICES, preserves pairing
    resampled = diff[idx]  # (n_boot, n)
    boot = np.mean(resampled, axis=1) if statistic == "mean" else np.median(resampled, axis=1)
    se = float(np.std(boot, ddof=1))

    alpha = 1.0 - ci
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 - lower_pct

    method_used = method
    if method == "bca":
        bounds = _bca_bounds(diff, boot, point, statistic, ci)
        if bounds is None:
            logger.warning(
                "paired_bootstrap_ci: BCa interval degenerate (all replicates on one side of the "
                "point estimate, or a zero-variance jackknife); falling back to percentile."
            )
            method_used = "percentile (bca degenerate)"
            lo, hi = (float(x) for x in np.percentile(boot, [lower_pct, upper_pct]))
        else:
            lo, hi = bounds
    else:
        lo, hi = (float(x) for x in np.percentile(boot, [lower_pct, upper_pct]))

    return BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        ci=ci,
        n=n,
        n_boot=n_boot,
        method=method_used,
        se=se,
        contains_zero=(lo <= 0.0 <= hi),
    )


def wilcoxon_signed_rank(
    a: ArrayLike,
    b: ArrayLike,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Wilcoxon signed-rank test on the paired difference `a - b`.

    `n_zero` (exact ties, `a - b == 0`) matters concretely on this project's
    data: under the `ignore_empty=False` Dice convention (see CLAUDE.md), a
    case whose ground-truth ET region is empty scores Dice exactly 1.0 for
    ANY model that also predicts empty there, so two otherwise-different
    models tie exactly on those cases. A large `n_zero` means the test is
    effectively run on a smaller, non-obvious subset of cases.

    Args:
        a: `(n,)` per-case values for model A.
        b: `(n,)` per-case values for model B, same case order as `a`.
        alternative: `"two-sided"`, `"greater"`, or `"less"`, forwarded to
            `scipy.stats.wilcoxon`.
        zero_method: Forwarded to `scipy.stats.wilcoxon`. `"wilcox"` (the
            default) discards exact-zero differences before ranking.

    Returns:
        A `WilcoxonResult`. If every paired difference is exactly zero,
        there is nothing left to rank -- depending on sample size and scipy
        version, `scipy.stats.wilcoxon` either raises `ValueError` or
        returns a `NaN` p-value from a 0/0 division (both observed; neither
        is useful). Either way this is detected directly from `n_zero == n`
        BEFORE calling scipy, and reported as `statistic=0.0, pvalue=1.0`
        (two identical models is a legitimate input meaning "no
        difference", not a crash), with a warning logged.

    Raises:
        ValueError: `a`/`b` length mismatch, or fewer than 2 pairs survive
            NaN removal.
    """
    a_clean, b_clean = _clean_pair(a, b, "wilcoxon_signed_rank")
    n = a_clean.size
    diff = a_clean - b_clean
    n_zero = int(np.sum(diff == 0.0))

    if n_zero == n:
        logger.warning(
            "wilcoxon_signed_rank: all %d differences are exactly zero -- the two models produced "
            "identical per-case scores. Returning statistic=0.0, pvalue=1.0 instead of calling "
            "scipy (which either raises or returns NaN on this input, depending on sample size).",
            n,
        )
        statistic = 0.0
        pvalue = 1.0
    else:
        if n_zero / n > 0.5:
            logger.warning(
                "wilcoxon_signed_rank: %d/%d pairs are exact ties (a - b == 0) -- the test result "
                "is driven by a minority of cases.",
                n_zero,
                n,
            )
        result = stats.wilcoxon(a_clean, b_clean, alternative=alternative, zero_method=zero_method)
        statistic = float(result.statistic)
        pvalue = float(result.pvalue)

    rank_biserial = _rank_biserial_from_diff(diff)
    n_effective = (n - n_zero) if zero_method == "wilcox" else n

    return WilcoxonResult(
        statistic=statistic,
        pvalue=pvalue,
        n=n,
        n_zero=n_zero,
        n_effective=n_effective,
        rank_biserial=rank_biserial,
        alternative=alternative,
        zero_method=zero_method,
    )


def paired_effect_size(a: ArrayLike, b: ArrayLike) -> EffectSize:
    """Cohen's `dz`, Hedges' `g`, and the signed-rank effect size for `a - b`.

    Args:
        a: `(n,)` per-case values for model A.
        b: `(n,)` per-case values for model B, same case order as `a`.

    Returns:
        An `EffectSize`. `cohens_dz` is `+/-inf` (with a warning logged) if
        `sd_diff == 0` and `mean_diff != 0` -- every case has an identical,
        nonzero difference, which is an infinitely reliable but degenerate
        effect. `0.0` if `sd_diff == 0` and `mean_diff == 0` (the two
        models are identical on every case).

    Raises:
        ValueError: `a`/`b` length mismatch, or fewer than 2 pairs survive
            NaN removal.
    """
    a_clean, b_clean = _clean_pair(a, b, "paired_effect_size")
    n = a_clean.size
    diff = a_clean - b_clean
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    if sd_diff == 0.0:
        if mean_diff == 0.0:
            cohens_dz = 0.0
        else:
            cohens_dz = float("inf") if mean_diff > 0.0 else float("-inf")
            logger.warning(
                "paired_effect_size: sd_diff is exactly 0 with a nonzero mean_diff (%.6g) -- "
                "every case has an identical, nonzero difference. cohens_dz is +/-inf.",
                mean_diff,
            )
    else:
        cohens_dz = mean_diff / sd_diff

    # Small-sample bias correction (Hedges & Olkin). n >= 2 is guaranteed by
    # _clean_pair, so 4 * (n - 1) - 1 is never zero.
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    hedges_g = cohens_dz * correction

    rank_biserial = _rank_biserial_from_diff(diff)

    abs_dz = abs(cohens_dz)
    if abs_dz < 0.2:
        magnitude = "negligible"
    elif abs_dz < 0.5:
        magnitude = "small"
    elif abs_dz < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return EffectSize(
        mean_diff=mean_diff,
        sd_diff=sd_diff,
        cohens_dz=cohens_dz,
        hedges_g=hedges_g,
        rank_biserial=rank_biserial,
        n=n,
        magnitude=magnitude,
    )


def holm_bonferroni(
    pvalues: Sequence[float] | np.ndarray, alpha: float = DEFAULT_ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction for family-wise error rate.

    Holm's method controls the family-wise error rate (the probability of
    ANY false positive across the whole family of tests) at `alpha`, and is
    uniformly more powerful than plain Bonferroni (it multiplies the
    smallest p-value by `m`, not every p-value by `m`, and unwinds from
    there) while making no independence assumption about the tests. It
    requires that the family be declared BEFORE looking at the p-values --
    in this project the family is "one comparison per metric per baseline",
    which is exactly what `compare_models` passes through it in one call.

    Args:
        pvalues: The family's raw p-values, in some caller-meaningful order.
        alpha: Family-wise significance level.

    Returns:
        `(adjusted, reject)`: `adjusted` is each p-value's Holm-adjusted
        value (same order as the input, monotonically non-decreasing once
        sorted, clipped to `[0, 1]`); `reject` is `adjusted <= alpha`.

    Raises:
        ValueError: `pvalues` is empty, or any entry is NaN or outside
            `[0, 1]`.
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if p.size == 0:
        raise ValueError("holm_bonferroni: pvalues is empty; nothing to correct.")
    if np.any(np.isnan(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(
            f"holm_bonferroni: every p-value must be finite and in [0, 1], got {p.tolist()}."
        )

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)  # i-th smallest (0-based) multiplied by (m - i)
    adjusted_sorted = np.clip(np.maximum.accumulate(sorted_p * multipliers), 0.0, 1.0)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= alpha
    return adjusted, reject


def load_per_case(path: str | Path) -> pd.DataFrame:
    """Loads a `per_case_metrics.csv`-style table, indexed by `case_id`.

    Args:
        path: Path to the CSV file.

    Returns:
        A `DataFrame`. If a `case_id` column is present, it is set as the
        index.

    Raises:
        FileNotFoundError: `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"load_per_case: no file at {path}.")
    df = pd.read_csv(path)
    if "case_id" in df.columns:
        df = df.set_index("case_id")
    return df


def _with_case_id_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sets `case_id` as the index if the index is unnamed and that column exists."""
    if df.index.name is None and "case_id" in df.columns:
        df = df.set_index("case_id")
    return df


def _resolve_threshold(
    practical_threshold: Mapping[str, float] | float | None, metric: str
) -> float | None:
    """Resolves a per-metric practical-significance threshold.

    A float applies to every metric. A mapping is checked by the metric's
    FULL name first, then by its prefix (the part before the first `"_"`).
    `None` (no threshold given, or none matched) disables the
    `"negligible"` verdict for this metric.
    """
    if practical_threshold is None:
        return None
    if isinstance(practical_threshold, int | float):
        return float(practical_threshold)
    if metric in practical_threshold:
        return float(practical_threshold[metric])
    prefix = metric.split("_")[0]
    if prefix in practical_threshold:
        return float(practical_threshold[prefix])
    return None


_RESULT_COLUMNS_TEMPLATE = [
    "n",
    "n_missing",
    "mean_{name_a}",
    "mean_{name_b}",
    "mean_diff",
    "improvement",
    "improvement_lo",
    "improvement_hi",
    "ci_lo",
    "ci_hi",
    "higher_is_better",
    "p_wilcoxon",
    "p_holm",
    "reject_holm",
    "n_zero",
    "cohens_dz",
    "hedges_g",
    "rank_biserial",
    "magnitude",
    "verdict",
]


def compare_models(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    generator: np.random.Generator,
    metrics: Sequence[str] | None = None,
    name_a: str = "model",
    name_b: str = "baseline",
    n_boot: int = 10000,
    ci: float = 0.95,
    alpha: float = DEFAULT_ALPHA,
    method: str = "percentile",
    higher_is_better: Mapping[str, bool] | None = None,
    practical_threshold: Mapping[str, float] | float | None = None,
) -> pd.DataFrame:
    """Runs the full paired comparison (bootstrap CI + Wilcoxon + effect size + Holm) per metric.

    Cases are aligned by `case_id` INDEX INTERSECTION, never positionally --
    two evaluation runs can legitimately emit cases in different orders (a
    different dataloader shuffle, a different subset), and a positional
    comparison would silently compare unrelated cases while still producing
    entirely plausible-looking numbers.

    Args:
        a: Per-case metric table for model A, indexed by `case_id` (or with
            a `case_id` column and an unnamed index).
        b: Per-case metric table for model B, same convention.
        generator: An explicit `np.random.Generator`, forwarded to
            `paired_bootstrap_ci` for every metric -- REQUIRED, no default.
        metrics: Metric column names to compare. If `None`, uses every
            numeric column present in BOTH tables, excluding any column
            starting with `"gt_empty"`.
        name_a: Display name for `a`, used in the `mean_<name_a>` column.
        name_b: Display name for `b`, used in the `mean_<name_b>` column.
        n_boot: Forwarded to `paired_bootstrap_ci`.
        ci: Forwarded to `paired_bootstrap_ci`.
        alpha: Family-wise significance level for `holm_bonferroni`, and the
            threshold `p_holm` is compared against for the `verdict` column.
        method: Forwarded to `paired_bootstrap_ci` (`"percentile"`/`"bca"`).
        higher_is_better: Forwarded to `metric_direction` as `overrides`.
        practical_threshold: A single float applied to every metric, or a
            mapping keyed by full metric name or prefix. When given, a
            statistically-conclusive difference whose ENTIRE improvement CI
            sits inside `+/-threshold` is reported as `"negligible"` rather
            than `"better"`/`"worse"`.

    Returns:
        A `DataFrame` indexed by `metric`, one row per compared metric,
        columns (in order): `n, n_missing, mean_<name_a>, mean_<name_b>,
        mean_diff, improvement, improvement_lo, improvement_hi, ci_lo,
        ci_hi, higher_is_better, p_wilcoxon, p_holm, reject_holm, n_zero,
        cohens_dz, hedges_g, rank_biserial, magnitude, verdict`.

        `improvement` is `mean_diff` reoriented so POSITIVE ALWAYS MEANS "A
        IS BETTER", regardless of the metric's own direction (`ci_lo`/
        `ci_hi` stay in the metric's own, unre-oriented `a - b` units;
        `improvement_lo`/`improvement_hi` are the same bounds re-oriented
        the same way as `improvement`, and are swapped when the metric is
        lower-is-better so `improvement_lo <= improvement_hi` always holds).

        `verdict` is one of:
          - `"inconclusive"`: the bootstrap CI contains 0, or `p_holm >
            alpha`. Logged as a warning naming the metric -- this
            difference must not be claimed in the paper.
          - `"negligible"`: statistically conclusive but the entire
            improvement CI sits inside `+/-practical_threshold`. Logged as
            a warning.
          - `"better"` / `"worse"`: conclusive and (by `improvement`'s
            sign) practically meaningful.

    Raises:
        ValueError: The `case_id` intersection between `a` and `b` is
            empty; a requested metric in `metrics` is missing from either
            table; or propagated from `paired_bootstrap_ci` /
            `wilcoxon_signed_rank` / `paired_effect_size` / `holm_bonferroni`
            for an individual metric.
    """
    a_df = _with_case_id_index(a)
    b_df = _with_case_id_index(b)

    common = a_df.index.intersection(b_df.index)
    if len(common) == 0:
        raise ValueError("compare_models: a and b have no overlapping case_id; nothing to compare.")
    only_a = a_df.index.difference(b_df.index)
    only_b = b_df.index.difference(a_df.index)
    if len(only_a) > 0 or len(only_b) > 0:
        logger.warning(
            "compare_models: case sets differ -- %d case(s) only in %s, %d only in %s. "
            "Comparing on the %d-case intersection.",
            len(only_a),
            name_a,
            len(only_b),
            name_b,
            len(common),
        )

    a_aligned = a_df.loc[common]
    b_aligned = b_df.loc[common]

    if metrics is None:
        numeric_a = set(a_aligned.select_dtypes(include="number").columns)
        numeric_b = set(b_aligned.select_dtypes(include="number").columns)
        metric_list = sorted(m for m in (numeric_a & numeric_b) if not m.startswith("gt_empty"))
    else:
        metric_list = list(metrics)
        for m in metric_list:
            if m not in a_aligned.columns:
                raise ValueError(f"compare_models: metric {m!r} is missing from a.")
            if m not in b_aligned.columns:
                raise ValueError(f"compare_models: metric {m!r} is missing from b.")

    records: list[dict[str, object]] = []
    pvalues: list[float] = []
    for m in metric_list:
        a_vals = a_aligned[m].to_numpy(dtype=np.float64)
        b_vals = b_aligned[m].to_numpy(dtype=np.float64)
        n_before = a_vals.size

        boot = paired_bootstrap_ci(
            a_vals, b_vals, n_boot=n_boot, ci=ci, generator=generator, method=method
        )
        wil = wilcoxon_signed_rank(a_vals, b_vals)
        eff = paired_effect_size(a_vals, b_vals)
        higher = metric_direction(m, higher_is_better)

        if higher:
            improvement, improvement_lo, improvement_hi = eff.mean_diff, boot.lo, boot.hi
        else:
            improvement, improvement_lo, improvement_hi = -eff.mean_diff, -boot.hi, -boot.lo

        # The two reported means are taken over the PAIRED-COMPLETE subset (the
        # same cases the difference is computed on), not over each column's own
        # non-NaN values. HD95 is legitimately NaN when exactly one side of a
        # region is empty, and those NaNs need not land on the same cases in
        # both tables -- averaging each column independently would make
        # mean_a - mean_b disagree with mean_diff (measured: -1.173 vs -1.130
        # on a 30-case table with a single one-sided NaN), which reads as an
        # arithmetic error in the paper's table with nothing failing anywhere.
        paired = ~(np.isnan(a_vals) | np.isnan(b_vals))

        records.append(
            {
                "metric": m,
                "n": boot.n,
                "n_missing": n_before - boot.n,
                f"mean_{name_a}": float(np.mean(a_vals[paired])),
                f"mean_{name_b}": float(np.mean(b_vals[paired])),
                "mean_diff": eff.mean_diff,
                "improvement": improvement,
                "improvement_lo": improvement_lo,
                "improvement_hi": improvement_hi,
                "ci_lo": boot.lo,
                "ci_hi": boot.hi,
                "higher_is_better": higher,
                "p_wilcoxon": wil.pvalue,
                "n_zero": wil.n_zero,
                "cohens_dz": eff.cohens_dz,
                "hedges_g": eff.hedges_g,
                "rank_biserial": eff.rank_biserial,
                "magnitude": eff.magnitude,
                "_contains_zero": boot.contains_zero,
            }
        )
        pvalues.append(wil.pvalue)

    adjusted, reject = holm_bonferroni(pvalues, alpha=alpha)

    for record, p_holm, rej in zip(records, adjusted, reject, strict=True):
        record["p_holm"] = float(p_holm)
        record["reject_holm"] = bool(rej)
        contains_zero = record.pop("_contains_zero")

        if contains_zero or p_holm > alpha:
            record["verdict"] = "inconclusive"
            logger.warning(
                "compare_models: %s is WITHIN NOISE (inconclusive) for %s vs %s -- mean diff "
                "%.4g, CI [%.4g, %.4g], p_holm=%.4g. Do NOT claim this difference in the paper.",
                record["metric"],
                name_a,
                name_b,
                record["mean_diff"],
                record["ci_lo"],
                record["ci_hi"],
                p_holm,
            )
            continue

        threshold = _resolve_threshold(practical_threshold, str(record["metric"]))
        if (
            threshold is not None
            and abs(record["improvement_lo"]) < threshold
            and abs(record["improvement_hi"]) < threshold
        ):
            record["verdict"] = "negligible"
            logger.warning(
                "compare_models: %s is statistically detectable but practically negligible for "
                "%s vs %s -- the entire improvement CI [%.4g, %.4g] sits within +/-%.4g.",
                record["metric"],
                name_a,
                name_b,
                record["improvement_lo"],
                record["improvement_hi"],
                threshold,
            )
        elif record["improvement"] > 0:
            record["verdict"] = "better"
        else:
            record["verdict"] = "worse"

    result = pd.DataFrame(records).set_index("metric")
    columns = [c.format(name_a=name_a, name_b=name_b) for c in _RESULT_COLUMNS_TEMPLATE]
    return result[columns]


def format_comparison(table: pd.DataFrame, name_a: str = "model", name_b: str = "baseline") -> str:
    """Renders a `compare_models` table as a plain-text report.

    Args:
        table: The output of `compare_models`.
        name_a: Display name for A (must match what `compare_models` used).
        name_b: Display name for B.

    Returns:
        A multi-line plain-text string (no colour, no emoji, no markdown
        table) -- one line per metric with the improvement, its CI, the
        Holm-adjusted p-value and the verdict, followed by an explicit
        `WARNING` block listing every `"inconclusive"` or `"negligible"`
        row. States explicitly that every row is conclusive if none are.
    """
    lines = [f"Comparison: {name_a} vs {name_b} ({len(table)} metric(s))", ""]
    warn_lines: list[str] = []

    for metric, row in table.iterrows():
        direction = "higher is better" if row["higher_is_better"] else "lower is better"
        lines.append(
            f"{metric:20s} improvement={row['improvement']:+.4g} "
            f"CI=[{row['improvement_lo']:+.4g}, {row['improvement_hi']:+.4g}] "
            f"p_holm={row['p_holm']:.4g} ({direction}) -> {str(row['verdict']).upper()}"
        )
        if row["verdict"] in ("inconclusive", "negligible"):
            warn_lines.append(
                f"  - {metric}: {row['verdict']} (mean diff {row['mean_diff']:+.4g}, raw CI "
                f"[{row['ci_lo']:+.4g}, {row['ci_hi']:+.4g}], p_holm={row['p_holm']:.4g})"
            )

    lines.append("")
    if warn_lines:
        lines.append(
            "WARNING: the following differences are within noise or practically negligible and "
            "must NOT be claimed as real improvements in the paper:"
        )
        lines.extend(warn_lines)
    else:
        lines.append("All metrics are statistically and practically conclusive.")

    return "\n".join(lines)
