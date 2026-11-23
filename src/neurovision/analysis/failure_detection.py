"""Case-level failure detection: which label-free signal finds the bad mask?

This backs `docs/research_docs/preregistrations/preregistration_failure_detection.md`.
Every case gets a score from a signal that never sees the label (higher = "this
mask is probably bad"). We then ask how well the score separates cases whose
mask turned out unusable from those that did not.

Two plain-language ideas used here:

- **AUROC**: pick one unusable case and one usable case at random; AUROC is the
  chance the unusable one has the higher score. 0.5 = coin flip, 1.0 = perfect.
- **AURC** (area under the risk-coverage curve): imagine a clinic that accepts
  cases from most-trusted to least-trusted score. After accepting the first k
  cases, "risk" is the fraction of them that are unusable. AURC is the mean
  risk over k = 1..n. LOWER is better. The perfect ordering (all usable cases
  first) gives the "oracle" AURC; a useless score gives the base rate.

Uncertainty on a difference of two signals comes from a **paired, stratified
bootstrap**: re-draw the cases with replacement (the same drawn cases are
scored by both signals, which is what "paired" means), and draw the unusable
and usable cases separately with their own counts ("stratified"), so every
re-draw still contains both classes and AUROC is always defined.

Pure numpy/pandas: no torch, no file I/O, no model.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from neurovision.analysis.detection import REGION_NAMES, auroc
from neurovision.analysis.statistics import holm_bonferroni

__all__ = [
    "pairwise_mask_disagreement",
    "ensemble_disagreement_score",
    "ensemble_entropy_score",
    "risk_coverage",
    "aurc",
    "oracle_aurc",
    "random_aurc",
    "paired_delta_bootstrap",
    "estimable",
    "detection_family",
]

logger = logging.getLogger(__name__)

_LOG2 = float(np.log(2.0))
# float64 is used for the entropy math, so 1e-12 is representable (it would
# round to exactly 1.0 in fp16 -- the project's known clamp trap).
_EPS = 1e-12


def _dice_np(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks; both empty -> 1.0, one empty -> 0.0."""
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def pairwise_mask_disagreement(masks: Sequence[np.ndarray]) -> float:
    """1 - mean Dice over all unordered pairs of masks (one region, one case).

    Args:
        masks: At least two boolean arrays of identical shape, e.g. `(D, H, W)`.

    Returns:
        Disagreement in `[0, 1]`; 0 means all members agree exactly.

    Raises:
        ValueError: Fewer than two masks, or shapes differ.
    """
    if len(masks) < 2:
        raise ValueError("need at least 2 masks to measure disagreement")
    arrs = [np.asarray(m, dtype=bool) for m in masks]
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError("all masks must have the same shape")
    dices = [_dice_np(a, b) for a, b in itertools.combinations(arrs, 2)]
    return float(1.0 - np.mean(dices))


def ensemble_disagreement_score(
    region_masks: Mapping[str, Sequence[np.ndarray]],
    regions: Sequence[str] = ("WT", "TC"),
) -> float:
    """Protocol ENS-PD: `1 - min_R mean_pairs Dice_R` (the worst region counts).

    Args:
        region_masks: Region name -> member masks (boolean, same shape per region).
        regions: Regions to take the worst case over.

    Returns:
        Max over `regions` of `pairwise_mask_disagreement`.

    Raises:
        ValueError: A requested region is missing.
    """
    missing = [r for r in regions if r not in region_masks]
    if missing:
        raise ValueError(f"missing regions in region_masks: {missing}")
    return float(max(pairwise_mask_disagreement(region_masks[r]) for r in regions))


def ensemble_entropy_score(
    member_probs: Sequence[np.ndarray],
    region_names: Sequence[str] = REGION_NAMES,
    fg_threshold: float = 0.5,
) -> float:
    """Protocol ENS-ENT: normalised entropy of the mean member probability.

    Mirrors `detection.case_entropy_scalars` (`ent_mean_fg_mean`) exactly, so a
    one-member ensemble equals ENT: entropy is Bernoulli entropy / ln 2 (range
    [0, 1]); for each region the foreground is that region's own mean
    probability `> fg_threshold` (strict); the score is the mean over regions of
    the per-region foreground-mean entropy, skipping regions with empty
    foreground. (This differs from "WT-only foreground, mean over channels";
    the existing function was followed so ENT and ENS-ENT are comparable.)

    Args:
        member_probs: Per-member SIGMOID probabilities, each `(C, D, H, W)`,
            channel order ET, TC, WT.
        region_names: Channel names, in channel order.
        fg_threshold: Probability above which a voxel is predicted foreground.

    Returns:
        The score, or NaN if no region has any predicted foreground. NaN (not
        0.0) because, as in `case_entropy_scalars`, an empty prediction is not
        the same state as a confidently certain one; the driver must decide.

    Raises:
        ValueError: No members.
    """
    if len(member_probs) == 0:
        raise ValueError("need at least one member")
    # float64 so the clamp below is meaningful (a 1e-6 clamp is a no-op in fp16).
    mean_p = np.mean([np.asarray(p, dtype=np.float64) for p in member_probs], axis=0)
    p = np.clip(mean_p, _EPS, 1.0 - _EPS)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / _LOG2

    fg_means = []
    for i in range(len(region_names)):
        fg = mean_p[i] > fg_threshold
        if fg.any():
            fg_means.append(float(entropy[i][fg].mean()))
    if not fg_means:
        logger.debug("ensemble_entropy_score: no region has predicted foreground.")
        return float("nan")
    return float(np.mean(fg_means))


def _prep(score: Sequence[float] | np.ndarray, unusable: Sequence[bool] | np.ndarray):
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    u = np.asarray(unusable, dtype=bool).reshape(-1)
    if s.shape != u.shape:
        raise ValueError("score and unusable must have the same length")
    if np.isnan(s).any():
        raise ValueError("score contains NaN; handle missing cases explicitly")
    return s, u


def risk_coverage(
    score: Sequence[float] | np.ndarray, unusable: Sequence[bool] | np.ndarray
) -> pd.DataFrame:
    """Risk-coverage curve: accept cases from lowest score (most trusted) up.

    Ties are NOT averaged: a stable sort keeps input order among equal scores,
    so the result is deterministic but depends on the order of the inputs.

    Args:
        score: `(n,)` scores, higher = predicted worse.
        unusable: `(n,)` booleans, True = the mask is unusable.

    Returns:
        DataFrame with `coverage` (k/n) and `risk` (unusable fraction among the
        first k accepted), for k = 1..n.
    """
    s, u = _prep(score, unusable)
    order = np.argsort(s, kind="stable")
    k = np.arange(1, len(s) + 1)
    risk = np.cumsum(u[order]) / k
    return pd.DataFrame({"coverage": k / len(s), "risk": risk})


def aurc(score: Sequence[float] | np.ndarray, unusable: Sequence[bool] | np.ndarray) -> float:
    """Discrete AURC: mean of `risk` over k = 1..n (lower is better)."""
    return float(risk_coverage(score, unusable)["risk"].mean())


def oracle_aurc(unusable: Sequence[bool] | np.ndarray) -> float:
    """AURC of the perfect ordering (all usable cases accepted first)."""
    u = np.asarray(unusable, dtype=bool).reshape(-1)
    return aurc(u.astype(np.float64), u)


def random_aurc(unusable: Sequence[bool] | np.ndarray) -> float:
    """AURC of an uninformative score in expectation: the base rate."""
    return float(np.asarray(unusable, dtype=bool).mean())


def estimable(unusable: Sequence[bool] | np.ndarray, min_each: int = 5) -> bool:
    """True iff there are at least `min_each` unusable and `min_each` usable cases."""
    u = np.asarray(unusable, dtype=bool).reshape(-1)
    n_pos = int(u.sum())
    return n_pos >= min_each and (u.size - n_pos) >= min_each


def paired_delta_bootstrap(
    score_a: Sequence[float] | np.ndarray,
    score_b: Sequence[float] | np.ndarray,
    unusable: Sequence[bool] | np.ndarray,
    metric: Literal["auroc", "aurc"],
    n_boot: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Paired, stratified case bootstrap of `metric(a) - metric(b)`.

    Args:
        score_a: `(n,)` scores of signal A.
        score_b: `(n,)` scores of signal B, same cases.
        unusable: `(n,)` booleans, True = unusable (the stratification key).
        metric: `"auroc"` (higher better) or `"aurc"` (lower better).
        n_boot: Number of resamples.
        seed: Seed for `np.random.default_rng`.
        ci: Central percentile interval mass.

    Returns:
        Dict with `delta` (full sample), `lo`, `hi`, `p` (two-sided,
        `2 * min(frac <= 0, frac >= 0)` capped at 1), `n`, `n_pos`,
        `n_boot_valid` (resamples with a finite delta).

    Raises:
        ValueError: NaN in a score, length mismatch, or unknown metric.
    """
    if metric == "auroc":

        def fn(s: np.ndarray, u: np.ndarray) -> float:
            return auroc(s, u)

    elif metric == "aurc":
        fn = aurc
    else:
        raise ValueError(f"unknown metric {metric!r}")
    a, u = _prep(score_a, unusable)
    b, u_b = _prep(score_b, unusable)
    if a.shape != b.shape:
        raise ValueError("score_a and score_b must have the same length")

    delta = fn(a, u) - fn(b, u)
    pos_idx = np.flatnonzero(u)
    neg_idx = np.flatnonzero(~u)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        # Positives and negatives are resampled separately, with their own
        # counts, so each resample keeps both classes.
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=pos_idx.size, replace=True),
                rng.choice(neg_idx, size=neg_idx.size, replace=True),
            ]
        )
        deltas[i] = fn(a[idx], u[idx]) - fn(b[idx], u[idx])
    valid = deltas[np.isfinite(deltas)]
    tail = (1.0 - ci) / 2.0
    lo, hi = np.quantile(valid, [tail, 1.0 - tail])
    p = min(1.0, 2.0 * min(float((valid <= 0).mean()), float((valid >= 0).mean())))
    return {
        "delta": float(delta),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(p),
        "n": int(u.size),
        "n_pos": int(u.sum()),
        "n_boot_valid": int(valid.size),
    }


def detection_family(
    rows: pd.DataFrame, alpha: float = 0.05, higher_is_better: bool = True
) -> pd.DataFrame:
    """Applies ONE Holm correction across all estimable comparisons.

    Args:
        rows: Columns `signal, cohort, target, delta, lo, hi, p, estimable`.
        alpha: Family-wise level.
        higher_is_better: True for AUROC deltas; False for AURC (lower is better),
            which flips the meaning of the sign of `delta`.

    Returns:
        Copy of `rows` plus `p_holm` (NaN if not estimable) and `verdict` in
        {"better", "worse", "inconclusive", "not_estimable"}. "better" needs the
        improving sign, a CI excluding 0 and `p_holm <= alpha`; "worse" mirrors it.
    """
    out = rows.copy().reset_index(drop=True)
    out["p_holm"] = np.nan
    est = out["estimable"].astype(bool).to_numpy()
    if est.any():
        adjusted, _ = holm_bonferroni(out.loc[est, "p"].to_numpy(dtype=float), alpha=alpha)
        out.loc[est, "p_holm"] = adjusted

    verdicts = []
    for r, is_est in zip(out.itertuples(), est):
        if not is_est:
            verdicts.append("not_estimable")
            continue
        sig = r.p_holm <= alpha
        pos = r.delta > 0 and r.lo > 0
        neg = r.delta < 0 and r.hi < 0
        if not higher_is_better:
            pos, neg = neg, pos
        verdicts.append("better" if pos and sig else "worse" if neg and sig else "inconclusive")
    out["verdict"] = verdicts
    return out
