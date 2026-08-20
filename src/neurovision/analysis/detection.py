"""Does inter-branch disagreement carry signal entropy does not already have?

This backs the pre-registered decision in
`docs/research/preregistration_ambiguity.md`. Single-pass predictive
entropy is available for free from ANY model, including a plain U-Net, so a
disagreement score that merely reproduces entropy is worthless for the
claim this module exists to test. Every function here measures the
INCREMENTAL part: what survives once everything entropy already explains
is removed.

Three pieces, read top to bottom:

- `case_entropy_scalars` -- the entropy-only baseline. Computed from raw
  segmentation logits, no fusion internals, no label. Mirrors
  `summarize_case_ambiguity` in `scripts/extract_ambiguity.py` so the two
  scalar tables share one masking convention and can be joined on
  `case_id` without a mismatch.
- `spearman` / `partial_spearman` / `partial_spearman_ci` -- case-level
  correlation with a confound partialled out (e.g. "does disagreement
  correlate with Dice error, controlling for entropy").
- `auroc` / `residualised_auroc` -- voxel- or case-level detection power,
  with the same confound-removal idea applied via rank residualisation
  instead of partial correlation. `residualised_auroc` is the headline
  voxel-level quantity for the pre-registered decision.

No torch import here on purpose -- this is a numpy/scipy analysis module,
run on tables and arrays already saved to disk, and it should stay usable
without the training stack installed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from scipy import stats
from scipy.special import expit

from neurovision.analysis.statistics import BootstrapResult

__all__ = [
    "REGION_NAMES",
    "case_entropy_scalars",
    "spearman",
    "partial_spearman",
    "partial_spearman_ci",
    "auroc",
    "residualised_auroc",
]

logger = logging.getLogger(__name__)

# Deliberately NOT imported from neurovision.data.transforms -- that module
# imports torch and MONAI, which this analysis module (numpy/scipy only, run
# against already-saved arrays) should not need to pull in. Must match
# neurovision.data.transforms.REGION_NAMES exactly; both are frozen and this
# is the same channel order the model's sigmoid heads use.
REGION_NAMES: tuple[str, ...] = ("ET", "TC", "WT")

_LOG2 = float(np.log(2.0))

ArrayLike = Sequence[float] | np.ndarray


def _entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """Normalised Bernoulli entropy computed from raw logits, never from probabilities.

    `H(p) = p * softplus(-z) + (1 - p) * softplus(z)`, divided by `ln 2` so
    the result lies in `[0, 1]`. `softplus` is `np.logaddexp(0.0, x)`, which
    is finite for any finite `x` -- unlike `p.clamp(eps, 1-eps)` followed by
    `log(p)` / `log(1-p)`, whose clamp is a no-op in fp16 (fp16's epsilon is
    ~9.8e-4, so `1.0 - 1e-6` rounds to exactly `1.0`) and produced
    `0 * log(0) = NaN`, costing this project 10.5 GPU-hours. See
    `BranchAmbiguity._entropy_from_logits` in
    `neurovision.models.fusion.adaptive_fusion`, whose convention this
    mirrors exactly.

    Args:
        logits: Raw logits, any shape, float32.

    Returns:
        Normalised entropy, same shape as `logits`, values in `[0, 1]`.
    """
    p = expit(logits)
    softplus_neg = np.logaddexp(0.0, -logits)
    softplus_pos = np.logaddexp(0.0, logits)
    return (p * softplus_neg + (1.0 - p) * softplus_pos) / _LOG2


def case_entropy_scalars(
    logits: np.ndarray,
    region_names: Sequence[str] = REGION_NAMES,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Reduces one case's single-pass predictive entropy to per-case scalars.

    The entropy-only baseline this whole module exists to be compared
    against: computed from raw segmentation logits alone, with no fusion
    internals and no label. Mirrors `summarize_case_ambiguity` in
    `scripts/extract_ambiguity.py` -- same predicted-foreground masking
    convention, same NaN-for-empty-foreground semantics -- so the two
    scalar tables can be joined on `case_id` without their masks disagreeing.

    Args:
        logits: Raw segmentation logits (no sigmoid applied), shape
            `(C, D, H, W)`, typically loaded fp16 from a saved
            `logits/<case>.npy`. Cast to float32 before any math.
        region_names: Region channel names, in channel order. Defaults to
            `REGION_NAMES`.
        threshold: Probability threshold used to define each region's
            predicted foreground (`sigmoid(logits) > threshold`).

    Returns:
        One flat dict with, per region `R` in `region_names`:
        `ent_mean_R`, `ent_max_R`, `ent_mean_fg_R` -- plus
        `ent_mean_fg_mean`, the NaN-skipping mean of `ent_mean_fg_R` across
        regions. `ent_mean_fg_R` is NaN when that region's predicted
        foreground is empty: an empty prediction and a confidently certain
        prediction are different states and must not collapse to the same
        number.

    Note:
        Takes NO label argument, structurally -- not by convention. A
        scalar later correlated against per-case Dice must be computable
        with no access to the ground truth at all. This project already
        shipped a bug where a reporting mask was built from the label; it
        manufactured 41-57% of a reported ECE and passed 984 tests, because
        the code did exactly what it said.
    """
    logits_f32 = np.asarray(logits, dtype=np.float32)
    entropy = _entropy_from_logits(logits_f32)
    prob = expit(logits_f32)

    row: dict[str, float] = {}
    fg_means: list[float] = []
    for i, region in enumerate(region_names):
        ent_region = entropy[i]
        row[f"ent_mean_{region}"] = float(ent_region.mean())
        row[f"ent_max_{region}"] = float(ent_region.max())

        fg_mask = prob[i] > threshold
        if fg_mask.any():
            ent_fg_mean = float(ent_region[fg_mask].mean())
            row[f"ent_mean_fg_{region}"] = ent_fg_mean
            fg_means.append(ent_fg_mean)
        else:
            # NaN, not 0.0 -- an empty prediction is not the same state as a
            # perfectly-agreeing (zero-entropy) one. Same convention
            # summarize_case_ambiguity uses.
            row[f"ent_mean_fg_{region}"] = float("nan")
            logger.debug(
                "case_entropy_scalars: region %r has an empty predicted foreground.", region
            )

    row["ent_mean_fg_mean"] = float(np.mean(fg_means)) if fg_means else float("nan")
    return row


def _pairwise_complete(*arrays: ArrayLike) -> list[np.ndarray]:
    """Casts to float64 and drops any index where ANY input array is NaN.

    Args:
        *arrays: Two or more equal-length 1-D arrays.

    Returns:
        The same arrays, cast to float64 and filtered to the shared
        finite-in-all-arrays index set (pairwise-complete across all of
        them together, never per-variable -- masking independently would
        correlate different subsets of cases against each other).

    Raises:
        ValueError: The arrays do not all have the same length.
    """
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    lengths = {a.size for a in arrs}
    if len(lengths) > 1:
        raise ValueError(f"_pairwise_complete: arrays must have equal length, got {lengths}.")
    mask = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrs]


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation, dropping any pair with a NaN on either side.

    Args:
        x: `(n,)` values.
        y: `(n,)` values, same case order as `x`.

    Returns:
        The Spearman correlation coefficient, or NaN if fewer than 2 pairs
        survive NaN removal, or scipy itself returns NaN (e.g. one side is
        constant).
    """
    x_clean, y_clean = _pairwise_complete(x, y)
    if x_clean.size < 2:
        return float("nan")
    rho, _ = stats.spearmanr(x_clean, y_clean)
    return float(rho)


def partial_spearman(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> float:
    """Spearman correlation between `x` and `y`, controlling for `z`.

    `(r_xy - r_xz * r_yz) / sqrt((1 - r_xz**2) * (1 - r_yz**2))`, computed
    on Spearman correlations. This is the function that measures the
    INCREMENTAL part of a disagreement signal: whatever `x` (e.g. a
    disagreement scalar) shares with `y` (e.g. per-case Dice error) purely
    through `z` (e.g. entropy) is removed before the correlation is taken.

    Args:
        x: `(n,)` values.
        y: `(n,)` values, same case order as `x`.
        z: `(n,)` values to control for, same case order.

    Returns:
        The partial Spearman correlation, or NaN if the denominator is zero
        or non-finite (rather than raising), or if fewer than 2 cases
        survive NaN removal.

    Note:
        Cases are dropped pairwise-COMPLETE across `x`, `y` AND `z`
        together, never per-variable -- masking each pair independently
        would correlate different subsets of cases against each other.
    """
    x_clean, y_clean, z_clean = _pairwise_complete(x, y, z)
    if x_clean.size < 2:
        return float("nan")

    r_xy = spearman(x_clean, y_clean)
    r_xz = spearman(x_clean, z_clean)
    r_yz = spearman(y_clean, z_clean)

    denom = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    if not np.isfinite(denom) or denom == 0.0:
        return float("nan")
    result = (r_xy - r_xz * r_yz) / denom
    return float(result) if np.isfinite(result) else float("nan")


def partial_spearman_ci(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    generator: np.random.Generator,
    n_boot: int = 10000,
    ci: float = 0.95,
    return_replicates: bool = False,
) -> BootstrapResult | tuple[BootstrapResult, np.ndarray]:
    """Bootstraps a confidence interval on the partial Spearman correlation.

    Resamples CASE INDICES with replacement and recomputes
    `partial_spearman` on each resampled index set -- the same pairing
    discipline `neurovision.analysis.statistics.paired_bootstrap_ci` uses,
    so that whatever relationship exists between `x`, `y` and `z` within a
    case is preserved in every replicate.

    Args:
        x: `(n,)` values.
        y: `(n,)` values, same case order as `x`.
        z: `(n,)` values to control for, same case order.
        generator: An explicit `np.random.Generator` -- required, no
            default and no use of the global RNG.
        n_boot: Number of bootstrap replicates.
        ci: Confidence level, e.g. `0.95` for a 95% interval.
        return_replicates: When `True`, also return the raw, un-filtered
            array of `n_boot` bootstrap replicates (may contain NaN for
            degenerate resamples). Added for
            `scripts/detection_stats.py`'s Gate 1 driver, which needs the
            replicate distribution itself to compute a two-sided bootstrap
            p-value against a null of zero -- something a CI alone cannot
            give. Purely additive: default `False` preserves the exact
            prior return type and value for every existing caller.

    Returns:
        A `BootstrapResult` (imported from
        `neurovision.analysis.statistics`, so it composes unchanged with
        the rest of the analysis stack). `point` is `partial_spearman` on
        the un-resampled, pairwise-complete data. Replicates where the
        partial correlation is undefined (e.g. a degenerate resample makes
        one variable constant) are dropped before the interval and
        standard error are computed. When `return_replicates=True`, returns
        `(result, replicates)` instead, where `replicates` is the raw
        length-`n_boot` array used to build that `BootstrapResult` (NaN
        entries included -- the caller decides how to treat them).

    Raises:
        ValueError: Fewer than 4 cases survive pairwise-complete NaN
            removal, or every bootstrap replicate is undefined.
    """
    x_clean, y_clean, z_clean = _pairwise_complete(x, y, z)
    n = x_clean.size
    if n < 4:
        raise ValueError(
            f"partial_spearman_ci: only {n} case(s) survive pairwise-complete NaN removal; "
            "need at least 4."
        )
    if n < 20:
        logger.warning(
            "partial_spearman_ci: only %d case(s) -- a bootstrap CI over this few cases is not "
            "reliable and must not be reported as if it were.",
            n,
        )

    point = partial_spearman(x_clean, y_clean, z_clean)

    idx = generator.integers(0, n, size=(n_boot, n))  # resample CASE INDICES, preserves pairing
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        rows = idx[b]
        boot[b] = partial_spearman(x_clean[rows], y_clean[rows], z_clean[rows])

    valid = boot[np.isfinite(boot)]
    if valid.size == 0:
        raise ValueError(
            "partial_spearman_ci: every bootstrap replicate was undefined (denominator zero or "
            "non-finite in each resample)."
        )
    if valid.size < n_boot:
        logger.debug(
            "partial_spearman_ci: dropped %d/%d degenerate bootstrap replicate(s).",
            n_boot - valid.size,
            n_boot,
        )

    se = float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan")
    alpha = 1.0 - ci
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 - lower_pct
    lo, hi = (float(v) for v in np.percentile(valid, [lower_pct, upper_pct]))

    result = BootstrapResult(
        point=point,
        lo=lo,
        hi=hi,
        ci=ci,
        n=n,
        n_boot=n_boot,
        method="percentile",
        se=se,
        contains_zero=(lo <= 0.0 <= hi),
    )
    if return_replicates:
        return result, boot
    return result


def auroc(score: ArrayLike, positive: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U rank identity.

    `AUROC = U / (n_pos * n_neg)`, with `U` computed from
    `scipy.stats.rankdata`'s default average-rank tie handling -- so tied
    scores are handled correctly (a constant score gives exactly 0.5, never
    an off-by-tie-breaking value).

    Args:
        score: `(n,)` scores, higher meaning "more likely positive".
        positive: `(n,)` boolean array, True for the event being detected
            (e.g. a voxel is an error, or a case failed).

    Returns:
        AUROC in `[0, 1]`, or NaN if either class is empty.
    """
    score_arr = np.asarray(score, dtype=np.float64).reshape(-1)
    positive_arr = np.asarray(positive, dtype=bool).reshape(-1)

    n_pos = int(positive_arr.sum())
    n_neg = int(positive_arr.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = stats.rankdata(score_arr)
    rank_sum_pos = float(ranks[positive_arr].sum())
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def residualised_auroc(
    score: ArrayLike, control: ArrayLike, positive: ArrayLike
) -> dict[str, float]:
    """Detection power in `score` that survives removing what `control` explains.

    The headline voxel-level quantity for the pre-registered decision:
    `auroc_residual` says how much detection power a disagreement score
    carries after everything single-pass entropy (`control`) already
    explains is taken out. Both `score` and `control` are first converted
    to ranks and a linear fit of rank(score) on rank(control) is removed --
    a rank-based, monotone-invariant residual, consistent with using
    Spearman correlation elsewhere in this module. If `score` is a pure
    monotone function of `control`, the rank residual is identically zero
    and `auroc_residual` collapses to 0.5 (a constant "score" carries no
    detection information) even though `auroc_score` and `auroc_control`
    are both high -- exactly the "disagreement is just entropy again"
    outcome this function exists to catch.

    Args:
        score: `(n,)` the score being tested, e.g. inter-branch
            disagreement.
        control: `(n,)` the confound to remove, e.g. single-pass entropy.
        positive: `(n,)` boolean array, True for the event being detected.

    Returns:
        `{"auroc_score": ..., "auroc_control": ..., "auroc_residual": ...}`.
    """
    score_full = np.asarray(score, dtype=np.float64).reshape(-1)
    control_full = np.asarray(control, dtype=np.float64).reshape(-1)
    positive_full = np.asarray(positive, dtype=bool).reshape(-1)

    # positive is boolean, not NaN-droppable the same way score/control are,
    # but must share their finite mask so the residual and the two raw
    # AUROCs are all computed over the same cases.
    finite_mask = np.isfinite(score_full) & np.isfinite(control_full)
    score_arr = score_full[finite_mask]
    control_arr = control_full[finite_mask]
    positive_clean = positive_full[finite_mask]

    auroc_score = auroc(score_arr, positive_clean)
    auroc_control = auroc(control_arr, positive_clean)

    rank_score = stats.rankdata(score_arr)
    rank_control = stats.rankdata(control_arr)

    fit = stats.linregress(rank_control, rank_score)
    residual = rank_score - (fit.intercept + fit.slope * rank_control)

    auroc_residual = auroc(residual, positive_clean)

    return {
        "auroc_score": auroc_score,
        "auroc_control": auroc_control,
        "auroc_residual": auroc_residual,
    }
