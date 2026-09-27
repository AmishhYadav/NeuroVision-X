"""An input-statistics out-of-distribution score, pre-registered (P1.4).

`docs/research_docs/preregistrations/preregistration_ood.md` fixes this score BEFORE
any feature is extracted or any number exists: 46 label-free features per study, read
straight from the preprocessed 4-channel volume (the same volume the model sees), a
shrinkage-covariance Gaussian fitted on the `train` split only, Mahalanobis distance to
that fit as the score, and CAUTION/REFUSE cut points read off `val`-split quantiles
(never test/SSA/PED -- fitting or thresholding on anything but train/val respectively
is listed there as one of the things that would make this invalid).

This module is pure numpy/scipy/pandas: no file I/O, no Hydra, no torch. A driver
script (not written here) is the thing that walks `data/preprocessed/brats*/<case>/
image.npy`, calls `extract_features` on each array, and hands the resulting table to
`fit_ood_model` / `ood_scores` / `quantile_cuts`. Keeping the split here means every
function is testable on tiny synthetic arrays, on CPU, in well under a second -- the
project's own testing rule for anything below the driver-script layer.

**Why mean/SD are not features.** The preprocessing pipeline already z-scores every
modality over its own nonzero (brain) voxels, so a study's mean is fixed at 0 and its
SD at 1 by construction -- including them here would add two columns that are always
exactly the same value and would report zero information, not "the input is unusual".
The pre-registration says this explicitly; this module enforces it by simply never
computing them.

**Why every voxel array is cast to float32 before any reduction.** The volume on disk
is float16 (space; see `docs/reproducibility.md`). Summing ~1e6 float16 values (a
percentile sort is fine, but `np.std`, skew and kurtosis all sum squared deviations
internally) overflows float16's ~65504 max and was OBSERVED to return `inf` for `std`
during this module's own test-writing -- the "float16 overflow trap" the spec that
commissioned this module named directly. Casting to float32 immediately after loading,
before any arithmetic, is the fix; there is no way to enable it after the fact once a
sum has already overflowed.

**Why shrinkage.** 46 features fitted on `train` (875 cases) is not rank-deficient by
itself, but any subset used in testing or a future smaller split easily is: the sample
covariance of F features from n < F rows is exactly singular. Shrinking the sample
covariance toward its own diagonal (`Sigma_hat = (1 - lambda) * S + lambda * diag(S)`,
`lambda` fixed at 0.10 by the pre-registration, never tuned) guarantees a well-
conditioned, invertible matrix regardless of `n` vs `F`, at the cost of treating
features as slightly more independent than they measure. That is a deliberate,
pre-registered trade, not a tuning choice this module makes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from neurovision.inference.gatekeeper import Decision

logger = logging.getLogger(__name__)

# Channel order fixed by `cfg.data.modalities` and by the preprocessing pipeline that
# wrote `image.npy` -- changing this order would silently relabel every feature.
MODALITIES: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")

# The seven percentiles the pre-registration names, in the order they are read.
PERCENTILES: tuple[int, ...] = (1, 5, 25, 50, 75, 95, 99)

# The minimum number of nonzero voxels a modality must have for its percentiles,
# skewness and kurtosis to mean anything -- below this a "brain" is more likely a
# near-empty or corrupted volume, and `extract_features` raises rather than silently
# reporting statistics of a handful of stray voxels.
_MIN_NONZERO_VOXELS = 100


def _percentile_name(modality: str, percentile: int) -> str:
    """The feature name for one modality's one percentile, e.g. `"t1_p01"`.

    Args:
        modality: One of `MODALITIES`.
        percentile: One of `PERCENTILES`.

    Returns:
        `"{modality}_p{percentile:02d}"`.
    """
    return f"{modality}_p{percentile:02d}"


def _pairs(modalities: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Every unordered pair of `modalities`, in a fixed deterministic order.

    Args:
        modalities: The modality names to pair up, e.g. `MODALITIES`.

    Returns:
        A tuple of `(a, b)` pairs, `i < j` over `modalities`' own order, so the
        result does not depend on set/dict iteration order.
    """
    return tuple(
        (modalities[i], modalities[j])
        for i in range(len(modalities))
        for j in range(i + 1, len(modalities))
    )


def _feature_names(modalities: Sequence[str] = MODALITIES) -> tuple[str, ...]:
    """Build the full, ordered feature-name tuple for `modalities`.

    Order: per modality (percentiles, then skew, then kurtosis), then every
    pairwise correlation, then the three volume-shape features. This exact order
    is what `FEATURE_NAMES` freezes at import time -- `fit_ood_model` and
    `ood_scores` both index a `pd.DataFrame` by `FEATURE_NAMES`, so a caller's own
    column order in `features` never matters.

    Args:
        modalities: The modality names, e.g. `MODALITIES`.

    Returns:
        A tuple of feature names, `7*4 + 2*4 + 6 + 4 == 46` long for the default
        four modalities.
    """
    names: list[str] = []
    for modality in modalities:
        for percentile in PERCENTILES:
            names.append(_percentile_name(modality, percentile))
        names.append(f"{modality}_skew")
        names.append(f"{modality}_kurt")
    for a, b in _pairs(modalities):
        names.append(f"corr_{a}_{b}")
    names.append("log_brain_voxels")
    names.append("extent_d")
    names.append("extent_h")
    names.append("extent_w")
    return tuple(names)


# Frozen at import time for the default four modalities -- 46 names, fixed order.
FEATURE_NAMES: tuple[str, ...] = _feature_names(MODALITIES)


def extract_features(image: np.ndarray, modalities: Sequence[str] = MODALITIES) -> dict[str, float]:
    """Extract the 46 label-free OOD features from one preprocessed 4-channel volume.

    Args:
        image: `(C, D, H, W)` array, `C == len(modalities)`. Preprocessed and
            nonzero-z-scored, so background voxels are exactly 0 and brain voxels
            have (per-modality) mean ~0, SD ~1. Any float dtype; cast to float32
            internally before any reduction (see the module docstring's float16
            overflow note).
        modalities: The channel order, e.g. `MODALITIES`. Must match `image`'s
            channel axis length.

    Returns:
        A dict with exactly the keys `_feature_names(modalities)` names, each a
        Python `float`.

    Raises:
        ValueError: If `image` is not a 4-D array shaped `(len(modalities), D, H,
            W)`, or if any modality has fewer than `_MIN_NONZERO_VOXELS` nonzero
            voxels (too little brain to compute a meaningful statistic from).
    """
    if image.ndim != 4 or image.shape[0] != len(modalities):
        raise ValueError(
            f"extract_features: expected shape (C={len(modalities)}, D, H, W), "
            f"got {image.shape!r}."
        )

    # float32, immediately, before any sum/mean/std runs -- see module docstring.
    image = image.astype(np.float32, copy=False)

    features: dict[str, float] = {}
    nonzero_masks: list[np.ndarray] = []
    for c, modality in enumerate(modalities):
        volume = image[c]
        mask = volume != 0.0
        nonzero_masks.append(mask)
        count = int(mask.sum())
        if count < _MIN_NONZERO_VOXELS:
            raise ValueError(
                f"extract_features: modality {modality!r} has only {count} nonzero "
                f"voxels (< {_MIN_NONZERO_VOXELS}); too little brain to score."
            )
        brain_voxels = volume[mask]
        for percentile in PERCENTILES:
            features[_percentile_name(modality, percentile)] = float(
                np.percentile(brain_voxels, percentile)
            )
        # bias=True (the scipy default): a small, deliberate bias/variance choice,
        # not tuned per case -- the pre-registration names this explicitly as "ok".
        features[f"{modality}_skew"] = float(stats.skew(brain_voxels, bias=True))
        features[f"{modality}_kurt"] = float(stats.kurtosis(brain_voxels, fisher=True, bias=True))

    # The JOINT brain mask -- nonzero in every modality at once -- is the
    # denominator for the pairwise correlations: a voxel that is background in
    # even one modality (e.g. outside that sequence's field of view) should not
    # contribute to how two OTHER modalities correlate with each other.
    joint_mask = nonzero_masks[0]
    for mask in nonzero_masks[1:]:
        joint_mask = joint_mask & mask
    for a, b in _pairs(modalities):
        a_idx = modalities.index(a)
        b_idx = modalities.index(b)
        a_vals = image[a_idx][joint_mask]
        b_vals = image[b_idx][joint_mask]
        # np.corrcoef returns a 2x2 matrix; off-diagonal is the Pearson r.
        corr = float(np.corrcoef(a_vals, b_vals)[0, 1])
        features[f"corr_{a}_{b}"] = corr

    # "Brain" for the voxel-count feature is the UNION across modalities (any
    # modality nonzero there), unlike the joint mask above used for correlation --
    # the pre-registration draws this distinction explicitly ("brain = nonzero in
    # ANY modality" for the count, vs. "the JOINT brain mask" for correlation).
    union_mask = nonzero_masks[0]
    for mask in nonzero_masks[1:]:
        union_mask = union_mask | mask
    features["log_brain_voxels"] = float(np.log(float(union_mask.sum())))

    extent_d, extent_h, extent_w = image.shape[1:]
    features["extent_d"] = float(extent_d)
    features["extent_h"] = float(extent_h)
    features["extent_w"] = float(extent_w)

    return features


@dataclass(frozen=True)
class OODModel:
    """A fitted shrinkage-covariance Gaussian over standardised OOD features.

    Attributes:
        feature_names: The feature columns, in the order `center`/`scale`/
            `precision` index them. Always `FEATURE_NAMES` in practice, but kept
            explicit so `to_dict`/`from_dict` round-trip without relying on the
            module-level constant staying stable.
        center: `(F,)` train-split mean of each raw feature.
        scale: `(F,)` train-split standard deviation (`ddof=1`) of each raw
            feature.
        precision: `(F, F)` inverse of the shrunk covariance of the STANDARDISED
            train features.
        shrinkage: The `lambda` used to shrink the sample covariance toward its
            diagonal (0.10, fixed by the pre-registration).
        n_fit: The number of rows `fit_ood_model` fitted on.
    """

    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    shrinkage: float
    n_fit: int

    def to_dict(self) -> dict[str, Any]:
        """This model as a plain, JSON-safe dict (lists, not arrays)."""
        return {
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "precision": self.precision.tolist(),
            "shrinkage": float(self.shrinkage),
            "n_fit": int(self.n_fit),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OODModel:
        """Rebuild an `OODModel` from `to_dict`'s output (or equivalent JSON)."""
        return cls(
            feature_names=tuple(d["feature_names"]),
            center=np.asarray(d["center"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            precision=np.asarray(d["precision"], dtype=np.float64),
            shrinkage=float(d["shrinkage"]),
            n_fit=int(d["n_fit"]),
        )


def _check_finite(features: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise `ValueError` naming the first column of `features` holding NaN/inf.

    Args:
        features: The table to check.
        columns: The columns to check, e.g. `FEATURE_NAMES`.

    Raises:
        ValueError: If any of `columns` contains a NaN or infinite value.
    """
    for column in columns:
        values = features[column].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite value in column {column!r}.")


def fit_ood_model(features: pd.DataFrame, shrinkage: float = 0.1) -> OODModel:
    """Fit a shrinkage-covariance Gaussian OOD model on `train`-split features only.

    Standardises each of `FEATURE_NAMES` with its own train mean/SD, computes the
    sample covariance (`ddof=1`) of the standardised features, shrinks it toward
    its own diagonal by `shrinkage`, and inverts the result.

    Args:
        features: One row per training case, columns a superset of
            `FEATURE_NAMES` (extra columns, e.g. `case_id`, are ignored). Must be
            the `train` split only -- fitting on anything else invalidates the
            pre-registration this module implements.
        shrinkage: `lambda` in `(1 - lambda) * S + lambda * diag(S)`. The
            pre-registration fixes this at `0.10` and forbids revisiting it after
            any score exists; the default here matches that, but the argument
            stays open so a test can probe the shrinkage mechanism itself.

    Returns:
        The fitted `OODModel`.

    Raises:
        ValueError: If any `FEATURE_NAMES` column is missing, contains NaN/inf,
            or has zero standard deviation (a zero-SD feature would divide by
            zero when standardising); if `len(features) < 10`; or if the shrunk
            covariance is singular (checked via its condition number before
            inverting).
    """
    missing = [name for name in FEATURE_NAMES if name not in features.columns]
    if missing:
        raise ValueError(f"fit_ood_model: features is missing required columns: {missing}.")
    if len(features) < 10:
        raise ValueError(
            f"fit_ood_model: need at least 10 rows to fit, got {len(features)}. "
            "(Not `>= F + 1` -- shrinkage is exactly what makes a smaller n invertible.)"
        )
    _check_finite(features, FEATURE_NAMES)

    ordered = features.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    center = ordered.mean(axis=0)
    scale = ordered.std(axis=0, ddof=1)
    if np.any(scale == 0.0):
        zero_cols = [name for name, s in zip(FEATURE_NAMES, scale) if s == 0.0]
        raise ValueError(
            f"fit_ood_model: zero standard deviation in column(s) {zero_cols}; "
            "cannot standardise a constant feature."
        )

    standardised = (ordered - center) / scale
    # ddof=1: sample covariance, matching `scale`'s own ddof=1 SD.
    sample_cov = np.cov(standardised, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(sample_cov))
    shrunk_cov = (1.0 - shrinkage) * sample_cov + shrinkage * diagonal

    cond = np.linalg.cond(shrunk_cov)
    if not np.isfinite(cond) or cond > 1.0 / np.finfo(np.float64).eps:
        raise ValueError(
            f"fit_ood_model: shrunk covariance is singular or numerically unstable "
            f"(condition number {cond!r}); increase `shrinkage` or check for "
            "collinear features."
        )
    precision = np.linalg.inv(shrunk_cov)

    logger.info(
        "fit_ood_model: fitted on n=%d rows, %d features, shrinkage=%.3f.",
        len(features),
        len(FEATURE_NAMES),
        shrinkage,
    )
    return OODModel(
        feature_names=FEATURE_NAMES,
        center=center,
        scale=scale,
        precision=precision,
        shrinkage=float(shrinkage),
        n_fit=len(features),
    )


def ood_scores(model: OODModel, features: pd.DataFrame) -> np.ndarray:
    """Mahalanobis distance of each row of `features` to `model`'s fitted mean.

    Args:
        model: A fitted `OODModel` (from `fit_ood_model` or `OODModel.from_dict`).
        features: One row per case, columns a superset of `model.feature_names`.

    Returns:
        `(n,)` array of non-negative Mahalanobis distances,
        `sqrt(z @ precision @ z)` for each row's standardised feature vector `z`.

    Raises:
        ValueError: If a required column is missing or contains NaN/inf.
    """
    missing = [name for name in model.feature_names if name not in features.columns]
    if missing:
        raise ValueError(f"ood_scores: features is missing required columns: {missing}.")
    _check_finite(features, model.feature_names)

    ordered = features.loc[:, list(model.feature_names)].to_numpy(dtype=np.float64)
    standardised = (ordered - model.center) / model.scale
    # einsum computes, per row i, z_i @ precision @ z_i in one pass -- the
    # squared Mahalanobis distance -- without materialising an (n, n) matrix.
    squared = np.einsum("ij,jk,ik->i", standardised, model.precision, standardised)
    # A tiny negative value from floating-point round-off (precision is a numeric
    # inverse, not exact) would otherwise NaN under sqrt; clip at 0 since a
    # Mahalanobis distance is never actually negative.
    squared = np.clip(squared, 0.0, None)
    return np.sqrt(squared)


def quantile_cuts(
    val_scores: np.ndarray, caution_quantile: float, refuse_quantile: float
) -> tuple[float, float]:
    """CAUTION/REFUSE cut points from val-split score quantiles. HIGH is bad.

    Args:
        val_scores: `(n,)` OOD scores from the `val` split only -- never
            test/SSA/PED, per the pre-registration.
        caution_quantile: The fraction of `val` flagged CAUTION-or-worse, e.g.
            `0.10` (the pre-registration's frozen value).
        refuse_quantile: The fraction of `val` flagged REFUSE, e.g. `0.02`. Must
            be strictly less than `caution_quantile` (REFUSE is the rarer, more
            extreme tail).

    Returns:
        `(caution_cut, refuse_cut)`, with `caution_cut <= refuse_cut` since HIGH
        is bad and REFUSE is the more extreme tail.

    Raises:
        ValueError: If `refuse_quantile` and `caution_quantile` are not both in
            `(0, 1)` with `refuse_quantile < caution_quantile`.
    """
    if not 0.0 < refuse_quantile < caution_quantile < 1.0:
        raise ValueError(
            "quantile_cuts: require 0 < refuse_quantile < caution_quantile < 1, got "
            f"refuse_quantile={refuse_quantile!r}, caution_quantile={caution_quantile!r}."
        )
    # HIGH is bad, so the top `caution_quantile` fraction of scores is CAUTION-or-
    # worse: its cut is the (1 - caution_quantile) quantile, mirroring
    # `gatekeeper.calibrate_thresholds`'s own "high tail" convention for
    # conformal_band/ood_score.
    caution_cut = float(np.quantile(val_scores, 1.0 - caution_quantile))
    refuse_cut = float(np.quantile(val_scores, 1.0 - refuse_quantile))
    return caution_cut, refuse_cut


def flag(scores: np.ndarray, caution_cut: float, refuse_cut: float) -> np.ndarray:
    """Classify each score as `"proceed"`, `"proceed_with_caution"`, or `"refuse"`.

    Matches `neurovision.inference.gatekeeper.judge_ood_score`'s own equality
    rule exactly: a score strictly greater than a cut crosses it (equal to a cut
    does not), so a case sitting exactly at threshold PROCEEDs (or CAUTIONs, at
    the caution cut) rather than being pushed into the worse bucket.

    Args:
        scores: `(n,)` OOD scores.
        caution_cut: The CAUTION cut point, e.g. `quantile_cuts(...)[0]`.
        refuse_cut: The REFUSE cut point, e.g. `quantile_cuts(...)[1]`.

    Returns:
        `(n,)` array of dtype `object`, each entry one of
        `Decision.PROCEED.value`, `Decision.PROCEED_WITH_CAUTION.value`, or
        `Decision.REFUSE.value`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    out = np.full(scores.shape, Decision.PROCEED.value, dtype=object)
    out[scores > caution_cut] = Decision.PROCEED_WITH_CAUTION.value
    out[scores > refuse_cut] = Decision.REFUSE.value
    return out
