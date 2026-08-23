"""Lesion-wise segmentation metrics -- the official BraTS metric family since 2023.

`metrics/segmentation.py` scores each region (ET/TC/WT) as one big VOXEL-WISE overlap
problem: Dice counts how many voxels agree out of how many voxels there are in total.
That hides a specific, clinically important failure. Imagine the ground truth has a
60,000-voxel main tumour mass plus a 200-voxel satellite lesion a few centimetres away,
and the model finds the mass perfectly but never sees the satellite at all. Voxel Dice
barely notices -- 200 missing voxels out of 60,200 moves the score by well under a
percentage point -- but a radiologist reading that scan would say the model missed a
tumour. BraTS's 2023 lesion-wise metrics fix this by scoring each connected component
("lesion instance") of the mask as its own detection problem: was it found at all, and
if so, how well.

The pipeline, per region, independently:

1. **Instances.** Label the connected components of the reference mask and of the
   prediction mask separately (`connectivity=26`, the full 3-D voxel neighbourhood --
   the BraTS official convention).
2. **Small-lesion filter.** Drop every component smaller than `min_lesion_voxels` from
   BOTH masks before anything else happens. This is not just a prediction-side
   noise filter -- dropping a small REFERENCE lesion means it can no longer be counted
   as a missed lesion (a false negative) either. That is deliberate: BraTS does this
   because sub-50-voxel ground-truth annotations are not reliably drawn by human
   raters, so scoring a model for "missing" one would be scoring it against
   annotation noise, not against the model's own error.
3. **Matching.** Reference and prediction instances are matched one-to-one by IoU
   overlap (`panoptica.NaiveThresholdMatching`), giving true positives (tp, a matched
   pair), false positives (fp, a prediction instance nothing matched) and false
   negatives (fn, a reference instance nothing matched).
4. **Scoring.** For every matched (tp) pair, panoptica computes a per-instance Dice
   and a per-instance Normalized Surface Dice (NSD, a boundary-agreement metric with
   an explicit millimetre tolerance -- "close enough" counts as matched). Averaging
   those two numbers over the matched pairs only would be `sq_dsc`/`sq_nsd` in
   panoptica's own vocabulary ("s" for "segmentation", i.e. how good the matched
   ones are) -- but that number silently ignores every miss and every false alarm,
   because an unmatched lesion never enters the average. The BraTS lesion-wise Dice
   fixes that by folding tp/fp/fn back in as a detection penalty:

       lwdice = sq_dsc * tp / (tp + fp + fn)

   Every missed reference lesion and every spurious prediction lesion effectively
   scores 0 and is averaged in, alongside the matched-pair scores. `lwnsd` is the
   same construction with NSD in place of Dice. (Do not use panoptica's own `pq_dsc`
   -- its denominator is `tp + 0.5*fp + 0.5*fn`, a different, more forgiving metric
   that is not what BraTS reports.)

**A trap this module works around**: panoptica's per-instance NSD is normally reached
through `Panoptica_Evaluator.evaluate()`, whose internal reduction path hardcodes the
NSD distance threshold to `min(voxelspacing)` -- 1.0 by construction whenever spacing is
left at its 1mm-isotropic default, and silently 0.5 (with only a `warnings.warn`, no
error) whenever no spacing is given at all. There is no way to override that from
`Panoptica_Evaluator`'s high-level API: the threshold and the physical spacing used to
convert voxel offsets into millimetres are the same input, so rescaling spacing to hit a
different threshold would rescale the real distances by the same factor and cancel
itself out exactly. This module instead drives panoptica's public
`Metric.NSD(reference, prediction, voxelspacing=..., threshold=...)` call directly (it
forwards straight to `panoptica.metrics.normalized_surface_dice`, which genuinely
accepts `threshold` as an argument independent of `voxelspacing`), once per matched
lesion pair -- so `nsd_tolerance_mm` is under this module's control, not panoptica's
default.

**Environment.** `panoptica` cannot be installed in the project's main training `.venv`
(it pins `numpy<2.3, pandas<3.0`, incompatible with the training lockfile's
`numpy==2.4.6, pandas==3.0.5`). It lives in the separate `.venv-analysis` virtualenv
(`requirements-analysis.txt`). Importing this module from the training `.venv` works
fine (nothing above happens at import time); calling `lesionwise_case_metrics` from
there raises `ImportError` with instructions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from torch import Tensor

from neurovision.data.transforms import REGION_NAMES

__all__ = ["LESIONWISE_METRIC_PREFIXES", "lesionwise_case_metrics"]

logger = logging.getLogger(__name__)

# One entry per metric family this module returns, before the "_<region>" suffix.
LESIONWISE_METRIC_PREFIXES: tuple[str, ...] = (
    "lwdice",
    "lwnsd",
    "lwf1",
    "lwtp",
    "lwfp",
    "lwfn",
)

# Set to True the first time `_load_panoptica` succeeds, so the (cheap but not free)
# logger/citation-banner setup runs once per process rather than once per case.
_panoptica_configured = False


def _load_panoptica() -> Any:
    """Imports `panoptica` lazily and silences its stdout banners, once per process.

    Kept as a function (never `import panoptica` at module scope) so that
    `from neurovision.metrics import lesionwise_case_metrics` keeps working in the
    training `.venv`, where panoptica is deliberately not installed.

    Returns:
        The imported `panoptica` module.

    Raises:
        ImportError: If `panoptica` is not installed in the current interpreter, with
            a message pointing at `requirements-analysis.txt` and the
            `.venv-analysis` virtualenv.
    """
    global _panoptica_configured
    try:
        import panoptica
    except ImportError as exc:
        raise ImportError(
            "lesionwise_case_metrics() requires the 'panoptica' package, which is "
            "deliberately NOT installed in the project's main .venv (it pins "
            "numpy<2.3 and pandas<3.0, incompatible with the training lockfile). "
            "It lives in the separate '.venv-analysis' virtualenv described in "
            "requirements-analysis.txt. Run this with "
            "'.venv-analysis/bin/python', not the default interpreter."
        ) from exc

    if not _panoptica_configured:
        # panoptica prints a citation reminder banner and INFO-level logs to stdout
        # by default; library code in this project must never do that.
        panoptica.set_log_level("ERROR")
        try:
            panoptica.disable_citation_reminder()
        except AttributeError:
            # A different panoptica release might not expose this toggle. It only
            # controls a cosmetic banner, so a missing hook should not fail every
            # lesion-wise call over it.
            pass
        _panoptica_configured = True
    return panoptica


def _count_components(labels: np.ndarray) -> int:
    """Counts non-background connected components in a `cc3d`-labeled array.

    Args:
        labels: Integer label array, background is 0. Labels need not be
            contiguous (some may have been zeroed out by the small-lesion filter).

    Returns:
        Number of distinct non-zero labels present.
    """
    return int(np.count_nonzero(np.unique(labels)))


def _drop_small_components(labels: np.ndarray, min_voxels: int) -> np.ndarray:
    """Zeroes out every connected component smaller than `min_voxels`.

    Args:
        labels: Integer label array from `cc3d.connected_components`, background 0.
        min_voxels: Components with fewer voxels than this are removed. `0` (or
            negative) is a no-op -- nothing is ever big enough to fail `< 0`.

    Returns:
        A copy of `labels` with small components' voxels set back to 0. Remaining
        labels are NOT renumbered; gaps in the label sequence are fine, since only
        `label != 0` membership is ever tested downstream.
    """
    if min_voxels <= 0 or labels.max() == 0:
        return labels
    counts = np.bincount(labels.ravel())
    too_small = np.nonzero(counts < min_voxels)[0]
    too_small = too_small[too_small != 0]  # never touch the background label
    if too_small.size == 0:
        return labels
    out = labels.copy()
    out[np.isin(labels, too_small)] = 0
    return out


def _region_lesionwise_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    panoptica_module: Any,
    *,
    spacing: tuple[float, float, float],
    min_lesion_voxels: int,
    matching_threshold: float,
    nsd_tolerance_mm: float,
    connectivity: int,
) -> dict[str, float]:
    """Computes lesion-wise Dice/NSD/F1/tp/fp/fn for one region of one case.

    Args:
        pred_mask: Boolean prediction mask, shape `(D, H, W)`, ONE region.
        target_mask: Boolean reference mask, same shape.
        panoptica_module: The already-imported `panoptica` module (from
            `_load_panoptica`), passed in rather than re-imported per region.
        spacing: Voxel spacing in mm, e.g. `(1.0, 1.0, 1.0)`.
        min_lesion_voxels: Components smaller than this are dropped from both
            masks before matching (see the module docstring for why this also
            changes what counts as a false negative).
        matching_threshold: IoU threshold for `NaiveThresholdMatching`.
        nsd_tolerance_mm: Surface-distance tolerance passed explicitly to
            `Metric.NSD`, in millimetres.
        connectivity: Passed to `cc3d.connected_components` (26 = full 3-D
            neighbourhood).

    Returns:
        `{"lwdice": float, "lwnsd": float, "lwf1": float, "lwtp": float,
        "lwfp": float, "lwfn": float}`. Never NaN (see module docstring / spec:
        both-empty scores 1.0, one-sided-empty scores 0.0 with the appropriate
        count, and a tp==0 match with both masks non-empty also scores 0.0
        rather than propagating panoptica's NaN mean-of-nothing).
    """
    import cc3d  # panoptica's own instance labeller; only reached once panoptica

    # itself has already been confirmed importable by `_load_panoptica`.

    ref_labels = cc3d.connected_components(target_mask, connectivity=connectivity)
    pred_labels = cc3d.connected_components(pred_mask, connectivity=connectivity)

    ref_labels = _drop_small_components(ref_labels, min_lesion_voxels)
    pred_labels = _drop_small_components(pred_labels, min_lesion_voxels)

    n_ref = _count_components(ref_labels)
    n_pred = _count_components(pred_labels)

    # Short-circuit every degenerate case ourselves: panoptica may refuse to
    # evaluate an empty pair at all, and even where it does not, "0 matched
    # lesions" would otherwise produce a NaN mean-of-nothing exactly where this
    # function promises a definite number (see module docstring).
    if n_ref == 0 and n_pred == 0:
        return {
            "lwdice": 1.0,
            "lwnsd": 1.0,
            "lwf1": 1.0,
            "lwtp": 0.0,
            "lwfp": 0.0,
            "lwfn": 0.0,
        }
    if n_ref == 0:
        return {
            "lwdice": 0.0,
            "lwnsd": 0.0,
            "lwf1": 0.0,
            "lwtp": 0.0,
            "lwfp": float(n_pred),
            "lwfn": 0.0,
        }
    if n_pred == 0:
        return {
            "lwdice": 0.0,
            "lwnsd": 0.0,
            "lwf1": 0.0,
            "lwtp": 0.0,
            "lwfp": 0.0,
            "lwfn": float(n_ref),
        }

    Metric = panoptica_module.Metric
    pair = panoptica_module.UnmatchedInstancePair(
        prediction_arr=pred_labels.astype(np.uint32),
        reference_arr=ref_labels.astype(np.uint32),
    )
    matcher = panoptica_module.NaiveThresholdMatching(
        matching_metric=Metric.IOU, matching_threshold=matching_threshold
    )
    matched = matcher.match_instances(pair)

    tp = matched.n_matched_instances
    fp = len(matched.missed_prediction_labels)
    fn = len(matched.missed_reference_labels)

    if tp == 0:
        # tp / (tp + fp + fn) is genuinely 0 here, but the mean Dice/NSD over
        # zero matched pairs is NaN, and NaN * 0 is still NaN in floating point
        # -- so this has to be caught explicitly rather than trusted to the
        # formula below to come out to 0 on its own.
        return {
            "lwdice": 0.0,
            "lwnsd": 0.0,
            "lwf1": 0.0,
            "lwtp": 0.0,
            "lwfp": float(fp),
            "lwfn": float(fn),
        }

    # `matched.prediction_arr` has been relabeled by the matcher so a matched
    # pair shares the same label id in both arrays -- reference and prediction
    # for lesion `label` are directly comparable without re-deriving the match.
    dice_values = []
    nsd_values = []
    for label in matched.matched_instances:
        ref_instance = matched.reference_arr == label
        pred_instance = matched.prediction_arr == label
        dice_values.append(float(Metric.DSC(ref_instance, pred_instance)))
        nsd_values.append(
            float(
                Metric.NSD(
                    ref_instance,
                    pred_instance,
                    voxelspacing=spacing,
                    threshold=nsd_tolerance_mm,
                )
            )
        )
    sq_dsc = float(np.mean(dice_values))
    sq_nsd = float(np.mean(nsd_values))

    denom = tp + fp + fn
    lwf1 = 2 * tp / (2 * tp + fp + fn)

    return {
        "lwdice": sq_dsc * tp / denom,
        "lwnsd": sq_nsd * tp / denom,
        "lwf1": float(lwf1),
        "lwtp": float(tp),
        "lwfp": float(fp),
        "lwfn": float(fn),
    }


def lesionwise_case_metrics(
    pred: Tensor,
    target: Tensor,
    *,
    region_names: Sequence[str] = REGION_NAMES,
    spacing: Sequence[float] | None = None,
    min_lesion_voxels: int = 50,
    matching_threshold: float = 0.5,
    nsd_tolerance_mm: float = 1.0,
    connectivity: int = 26,
) -> dict[str, float]:
    """Computes lesion-wise Dice, NSD, F1 and instance counts for one case.

    Mirrors `neurovision.metrics.segmentation.compute_case_metrics`'s calling
    convention (same shapes, same batch-of-one restriction, same style of
    `ValueError`s) so the two can sit side by side in an evaluation script. See
    the module docstring for what "lesion-wise" means and why it needs its own
    metric family on top of voxel-wise Dice.

    Args:
        pred: Binary prediction for ONE case, shape `(C, D, H, W)` or
            `(1, C, D, H, W)`. Any device, any float dtype -- thresholded at
            `> 0.5` after moving to CPU numpy, so raw sigmoid probabilities work
            too (not just an already-hard 0/1 mask).
        target: Binary reference, same shape convention as `pred`.
        region_names: Region name per channel, in channel order. Defaults to
            `REGION_NAMES` (`("ET", "TC", "WT")`).
        spacing: Voxel spacing in mm, e.g. `(1.0, 1.0, 1.0)`. `None` means
            isotropic `(1.0, 1.0, 1.0)` -- correct for BraTS's native
            resolution, but a caller scoring a differently-spaced case must
            pass its real spacing explicitly (see `test_anisotropic_spacing_...`
            in the test file for why this matters for NSD specifically).
        min_lesion_voxels: Connected components smaller than this are dropped
            from BOTH masks before matching, in both directions (see the
            module docstring for why dropping small reference lesions is the
            official BraTS convention, not a permissive shortcut).
        matching_threshold: IoU threshold for matching a predicted lesion to a
            reference lesion (`panoptica.NaiveThresholdMatching`).
        nsd_tolerance_mm: Surface-distance tolerance for the Normalized
            Surface Dice, in millimetres. BraTS uses 1mm; passing spacing
            without also setting this leaves panoptica's own default in
            effect for anything computed outside this module (this module
            always overrides it explicitly -- see the module docstring).
        connectivity: Passed to `cc3d.connected_components` for BOTH masks
            (26 = full 3-D neighbourhood, the BraTS official convention).

    Returns:
        Flat dict with, for each region name `R` in `region_names`:
        `lwdice_R`, `lwnsd_R`, `lwf1_R`, `lwtp_R`, `lwfp_R`, `lwfn_R`; plus
        cross-region means `lwdice_mean`, `lwnsd_mean`, `lwf1_mean` computed
        with `float(np.nanmean(...))`. Every value is a `float`; none is ever
        NaN (see `_region_lesionwise_metrics` for the degenerate cases this
        is guaranteed for).

    Raises:
        ImportError: If `panoptica` is not installed in the current
            interpreter (see `_load_panoptica`).
        ValueError: If `pred`/`target` do not have a batch size of at most 1,
            their shapes disagree, their ndim is outside `{4, 5}`, or
            `len(region_names) != pred.shape[1]`.
    """
    panoptica_module = _load_panoptica()

    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
    if target.ndim == 4:
        target = target.unsqueeze(0)
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "lesionwise_case_metrics expects (C, D, H, W) or (1, C, D, H, W) inputs, "
            f"got pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)}."
        )
    if pred.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(
            f"lesionwise_case_metrics is for a single case (batch size 1), got pred "
            f"batch {pred.shape[0]} and target batch {target.shape[0]}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}."
        )
    # Without this check, too few names would silently drop a whole region from
    # the per-region keys while still counting it in the *_mean values.
    if len(region_names) != pred.shape[1]:
        raise ValueError(
            f"region_names has {len(region_names)} entries {tuple(region_names)} but "
            f"pred has {pred.shape[1]} channels. One name per channel is required."
        )

    spacing_t: tuple[float, float, float] = (
        (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        if spacing is not None
        else (1.0, 1.0, 1.0)
    )

    # Never assume CUDA is absent or present: move to CPU numpy before cc3d/panoptica
    # touch anything, exactly like metrics/boundary.py does for scipy.
    pred_np = (pred.detach().cpu().numpy() > 0.5)[0]  # (C, D, H, W) bool
    target_np = (target.detach().cpu().numpy() > 0.5)[0]  # (C, D, H, W) bool

    per_region: dict[str, dict[str, float]] = {}
    metrics: dict[str, float] = {}
    for c, region in enumerate(region_names):
        region_metrics = _region_lesionwise_metrics(
            pred_np[c],
            target_np[c],
            panoptica_module,
            spacing=spacing_t,
            min_lesion_voxels=min_lesion_voxels,
            matching_threshold=matching_threshold,
            nsd_tolerance_mm=nsd_tolerance_mm,
            connectivity=connectivity,
        )
        per_region[region] = region_metrics
        for key, value in region_metrics.items():
            metrics[f"{key}_{region}"] = value

    for prefix in ("lwdice", "lwnsd", "lwf1"):
        metrics[f"{prefix}_mean"] = float(np.nanmean([per_region[r][prefix] for r in region_names]))

    return metrics
