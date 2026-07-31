"""Segmentation metrics for the BraTS region-overlap setup.

Predictions and targets in this project are three *overlapping* regions —
ET (enhancing tumor), TC (tumor core), WT (whole tumor) — not four mutually
exclusive classes. This module scores that layout: per-region Dice, IoU and
95th-percentile Hausdorff distance (HD95), a `compute_case_metrics` helper
that packages all three into one flat dict per case, and a `MetricAggregator`
that collects those dicts across a validation/test set into per-case and
summary tables.

All metric maths is MONAI's (`monai.metrics.compute_dice`,
`compute_iou`, `compute_hausdorff_distance`) — this module only supplies the
BraTS-specific empty-region conventions on top, documented function by
function below.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from torch import Tensor

from neurovision.data.transforms import REGION_NAMES

__all__ = [
    "REGION_NAMES",
    "classes_to_regions",
    "binarize",
    "dice_score",
    "iou_score",
    "hd95",
    "compute_case_metrics",
    "MetricAggregator",
]

logger = logging.getLogger(__name__)

# Raw, contiguous class values written by scripts/preprocess.py. Must mirror
# neurovision.data.transforms.ConvertToRegionsd exactly: background=0,
# necrotic/non-enhancing core=1, edema=2, enhancing tumor=3 (raw BraTS value 4
# was remapped to 3 during preprocessing, so 4 is never a valid input here).
_NECROTIC_CORE = 1
_EDEMA = 2
_ENHANCING_TUMOR = 3


def classes_to_regions(label: Tensor) -> Tensor:
    """Expands an integer BraTS class label into the three nested regions.

    Mirrors `neurovision.data.transforms.ConvertToRegionsd` exactly (same
    class constants, same nesting), so a raw integer label and a
    `ConvertToRegionsd`-processed one produce identical region tensors. Kept
    as a separate function (rather than importing the transform here) because
    metrics code should not depend on the MONAI `MapTransform` machinery.

    Args:
        label: Integer-valued tensor with values in `{0, 1, 2, 3}`, shape
            `(B, 1, D, H, W)`, `(B, D, H, W)`, `(1, D, H, W)`, or `(D, H, W)`.
            Raw BraTS label value 4 (enhancing tumor before remapping) is
            NOT treated specially here — preprocessing already remapped it to
            3, so a 4 in the input is just an out-of-range value that will
            not match any region.

    Returns:
        Float32 binary tensor of shape `(B, 3, D, H, W)` (a batch axis of 1
        is added if the input had none), channel order `(ET, TC, WT)`.

    Raises:
        ValueError: If `label` has a channel axis with size != 1, or an
            ndim outside `{3, 4, 5}`.
    """
    label_t = label
    if label_t.ndim == 5:
        if label_t.shape[1] != 1:
            raise ValueError(
                "classes_to_regions expects a single-channel label, got shape "
                f"{tuple(label_t.shape)}."
            )
        label_t = label_t[:, 0]  # (B, D, H, W)
    elif label_t.ndim == 4:
        # Ambiguous between (B, D, H, W) and (1, D, H, W); treat as batched
        # since that is what every caller downstream (loss, dataloader)
        # produces. A genuine (1, D, H, W) single case still works: it is
        # just a batch of size 1.
        pass
    elif label_t.ndim == 3:
        label_t = label_t.unsqueeze(0)  # (1, D, H, W)
    else:
        raise ValueError(
            "classes_to_regions expects a (B, 1, D, H, W), (B, D, H, W), (1, D, H, W) "
            f"or (D, H, W) label, got shape {tuple(label.shape)}."
        )

    et = label_t == _ENHANCING_TUMOR
    tc = et | (label_t == _NECROTIC_CORE)
    wt = tc | (label_t == _EDEMA)

    # Stack in REGION_NAMES order (ET, TC, WT) on a new channel axis 1.
    regions = torch.stack([et, tc, wt], dim=1)
    return regions.to(dtype=torch.float32)


def binarize(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Turns raw sigmoid logits into a binary mask.

    Exists so callers are explicit about the logits-to-mask step: every
    metric function in this module expects an ALREADY-BINARY input and does
    not apply a sigmoid itself.

    Args:
        logits: Raw (pre-sigmoid) model output, any shape.
        threshold: Probability threshold applied after sigmoid.

    Returns:
        Float32 tensor of 0.0/1.0 values, same shape as `logits`.
    """
    return (torch.sigmoid(logits) >= threshold).to(dtype=torch.float32)


def _check_metric_inputs(pred: Tensor, target: Tensor) -> None:
    """Shared shape validation for dice_score/iou_score/hd95.

    Args:
        pred: Binary prediction, expected `(B, C, D, H, W)`.
        target: Binary target, expected same shape as `pred`.

    Raises:
        ValueError: If shapes differ or either input is not 5-D.
    """
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "dice_score/iou_score/hd95 expect 5-D (B, C, D, H, W) tensors, got "
            f"pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}."
        )


def dice_score(pred: Tensor, target: Tensor, ignore_empty: bool = False) -> Tensor:
    """Per-channel Dice score between two ALREADY-BINARY masks.

    Thin wrapper over `monai.metrics.compute_dice` with `include_background=True`
    (channel 0 is ET, a real foreground region here — this is a multi-label
    sigmoid layout, not a softmax one-hot label map where channel 0 would be
    background) and no smoothing, so results match hand-computed Dice exactly.

    Verified against MONAI 1.6.0 directly: with `ignore_empty=False`, an
    empty-ground-truth channel scores 1.0 if the prediction is also empty and
    0.0 if the prediction has anything in it. With `ignore_empty=True`, an
    empty-ground-truth channel always scores NaN, regardless of the
    prediction. `ignore_empty=False` is the BraTS reporting convention this
    project defaults to. Roughly 35% of BraTS cases have no enhancing tumor
    at all, so this flag materially moves the headline ET Dice: leaving it
    False rewards correctly predicting "no ET" and punishes hallucinating
    it; `True` drops those cases from the average entirely instead.

    Args:
        pred: Binary float tensor, shape `(B, C, D, H, W)`.
        target: Binary float tensor, same shape as `pred`.
        ignore_empty: If True, an empty-ground-truth channel scores NaN
            instead of 1.0/0.0, so it can be excluded from an average.

    Returns:
        Float tensor of shape `(B, C)`.

    Raises:
        ValueError: If shapes differ or either input is not 5-D.
    """
    _check_metric_inputs(pred, target)
    return compute_dice(pred, target, include_background=True, ignore_empty=ignore_empty)


def iou_score(pred: Tensor, target: Tensor, ignore_empty: bool = False) -> Tensor:
    """Per-channel IoU (Jaccard) score between two ALREADY-BINARY masks.

    Same empty-region conventions as `dice_score` (verified against MONAI
    1.6.0: `ignore_empty=False` -> 1.0 for empty/empty, 0.0 for empty
    ground truth with any prediction; `ignore_empty=True` -> NaN whenever the
    ground truth is empty). See `dice_score` for why the default matters for
    ET specifically.

    Args:
        pred: Binary float tensor, shape `(B, C, D, H, W)`.
        target: Binary float tensor, same shape as `pred`.
        ignore_empty: If True, an empty-ground-truth channel scores NaN
            instead of 1.0/0.0.

    Returns:
        Float tensor of shape `(B, C)`.

    Raises:
        ValueError: If shapes differ or either input is not 5-D.
    """
    _check_metric_inputs(pred, target)
    return compute_iou(pred, target, include_background=True, ignore_empty=ignore_empty)


def hd95(pred: Tensor, target: Tensor, spacing: Sequence[float] | None = None) -> Tensor:
    """95th-percentile Hausdorff distance between two ALREADY-BINARY masks.

    Wraps `monai.metrics.compute_hausdorff_distance(..., include_background=True,
    percentile=95)`. MONAI returns NaN for every degenerate case (verified
    against MONAI 1.6.0: empty/empty, empty-ground-truth-with-prediction, and
    non-empty-ground-truth-with-empty-prediction all come back NaN, with a
    UserWarning). NaN is left as-is when only one side is empty (distance is
    genuinely undefined, so it is treated as missing rather than assigned an
    arbitrary large penalty), but this function overrides the both-empty case
    to 0.0: a perfect prediction of "nothing here" has no boundary error and
    should not silently vanish from a NaN-skipping mean. Any other
    non-finite value MONAI might return for a non-degenerate case is also
    mapped to NaN so it cannot poison a downstream mean.

    Args:
        pred: Binary float tensor, shape `(B, C, D, H, W)`.
        target: Binary float tensor, same shape as `pred`.
        spacing: Voxel spacing in mm, e.g. `(1.0, 1.0, 1.0)`. BraTS ships at
            1mm isotropic resolution, so the default `None` (MONAI treats
            distances as unit voxel spacing) already gives distances in mm
            for this dataset.

    Returns:
        Float tensor of shape `(B, C)`.

    Raises:
        ValueError: If shapes differ or either input is not 5-D.
    """
    _check_metric_inputs(pred, target)

    pred_empty = pred.sum(dim=(2, 3, 4)) == 0  # (B, C)
    target_empty = target.sum(dim=(2, 3, 4)) == 0  # (B, C)
    both_empty = pred_empty & target_empty

    raw = compute_hausdorff_distance(
        pred, target, include_background=True, percentile=95, spacing=spacing
    )
    out = raw.clone()
    out[~torch.isfinite(out)] = float("nan")  # any other non-finite -> NaN
    out[both_empty] = 0.0  # both empty -> no boundary error
    return out


def compute_case_metrics(
    pred: Tensor,
    target: Tensor,
    region_names: Sequence[str] = REGION_NAMES,
    spacing: Sequence[float] | None = None,
) -> dict[str, float]:
    """Computes Dice, IoU and HD95 for every region of a single case.

    Args:
        pred: Binary float tensor for ONE case, shape `(C, D, H, W)` or
            `(1, C, D, H, W)`.
        target: Binary float tensor, same shape convention as `pred`.
        region_names: Region name per channel, in channel order. Defaults to
            `REGION_NAMES` (`("ET", "TC", "WT")`).
        spacing: Voxel spacing in mm passed through to `hd95`.

    Returns:
        Flat dict with, for each region name `R` in `region_names`:
        `dice_R`, `iou_R`, `hd95_R`, `gt_empty_R` (1.0 if that region is
        absent from `target`, else 0.0); plus NaN-skipping cross-region
        means `dice_mean`, `iou_mean`, `hd95_mean` (`hd95_mean` is NaN only
        if every region's HD95 is NaN). Dice and IoU use
        `ignore_empty=False` (the BraTS reporting convention).

    Raises:
        ValueError: If `pred`/`target` do not have a batch size of at most 1
            (use `dice_score`/`iou_score`/`hd95` directly for a real batch),
            or their shapes disagree.
    """
    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
    if target.ndim == 4:
        target = target.unsqueeze(0)
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "compute_case_metrics expects (C, D, H, W) or (1, C, D, H, W) inputs, got "
            f"pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)}."
        )
    if pred.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(
            f"compute_case_metrics is for a single case (batch size 1), got pred batch "
            f"{pred.shape[0]} and target batch {target.shape[0]}. Use dice_score/"
            "iou_score/hd95 directly for a batch of cases."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}."
        )
    # Without this check, too few names would silently drop a whole region
    # from the per-region keys while still counting it in the *_mean values.
    if len(region_names) != pred.shape[1]:
        raise ValueError(
            f"region_names has {len(region_names)} entries {tuple(region_names)} but "
            f"pred has {pred.shape[1]} channels. One name per channel is required."
        )

    dice = dice_score(pred, target, ignore_empty=False)[0]  # (C,)
    iou = iou_score(pred, target, ignore_empty=False)[0]  # (C,)
    hausdorff = hd95(pred, target, spacing=spacing)[0]  # (C,)
    gt_empty = (target.sum(dim=(2, 3, 4)) == 0)[0]  # (C,)

    metrics: dict[str, float] = {}
    for i, name in enumerate(region_names):
        metrics[f"dice_{name}"] = float(dice[i])
        metrics[f"iou_{name}"] = float(iou[i])
        metrics[f"hd95_{name}"] = float(hausdorff[i])
        metrics[f"gt_empty_{name}"] = float(gt_empty[i])

    metrics["dice_mean"] = float(dice.nanmean())
    metrics["iou_mean"] = float(iou.nanmean())
    metrics["hd95_mean"] = float(hausdorff.nanmean())
    return metrics


class MetricAggregator:
    """Collects per-case metric dicts across a validation/test set.

    Typical use: call `add_case` once per case during evaluation, then read
    `per_case()` for a full table (e.g. to save to CSV) and `summary()` for
    the headline mean/std/median a report or W&B log wants.
    """

    def __init__(self, region_names: Sequence[str] = REGION_NAMES) -> None:
        """Initializes an empty aggregator.

        Args:
            region_names: Region name per channel, forwarded to
                `compute_case_metrics` by `add_case`. Defaults to
                `REGION_NAMES`.
        """
        self.region_names = list(region_names)
        self._case_ids: list[str] = []
        self._records: list[dict[str, float]] = []

    def update(self, case_id: str, metrics: Mapping[str, float]) -> None:
        """Stores a precomputed metric dict for one case.

        Args:
            case_id: Unique identifier for the case (e.g. BraTS case name).
            metrics: Flat `{metric_name: value}` dict, e.g. the output of
                `compute_case_metrics`.

        Raises:
            ValueError: If `case_id` was already added. A silent overwrite
                would quietly shrink the evaluation set without any error.
        """
        if case_id in self._case_ids:
            raise ValueError(
                f"case_id {case_id!r} was already added to this MetricAggregator. "
                "A duplicate would silently overwrite the earlier case's metrics."
            )
        self._case_ids.append(case_id)
        self._records.append(dict(metrics))

    def add_case(
        self,
        case_id: str,
        pred: Tensor,
        target: Tensor,
        spacing: Sequence[float] | None = None,
    ) -> dict[str, float]:
        """Computes metrics for one case via `compute_case_metrics` and stores them.

        Args:
            case_id: Unique identifier for the case.
            pred: Binary float tensor, `(C, D, H, W)` or `(1, C, D, H, W)`.
            target: Binary float tensor, same shape convention as `pred`.
            spacing: Voxel spacing in mm, forwarded to `hd95`.

        Returns:
            The computed metric dict (same one that was stored).

        Raises:
            ValueError: Propagated from `compute_case_metrics` (bad shapes)
                or `update` (duplicate `case_id`).
        """
        metrics = compute_case_metrics(
            pred, target, region_names=self.region_names, spacing=spacing
        )
        self.update(case_id, metrics)
        return metrics

    def per_case(self) -> pd.DataFrame:
        """Returns every stored case's metrics as a table.

        Returns:
            A DataFrame indexed by `case_id`, one column per metric key, in
            insertion order. Empty (but valid) DataFrame if no cases were
            added.
        """
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame(self._records, index=pd.Index(self._case_ids, name="case_id"))

    def summary(self) -> pd.DataFrame:
        """Summarizes every metric across all stored cases.

        Returns:
            A DataFrame indexed by metric name with columns `mean`, `std`,
            `median`, `count`, `n_missing`. `mean`/`std`/`median` skip NaN
            values (`std` is the sample standard deviation, ddof=1). `count`
            is the number of non-NaN values contributing to those
            statistics; `n_missing` is the number of NaN values, which makes
            e.g. undefined HD95 cases visible instead of silently dropped.
            Empty (but valid) DataFrame if no cases were added.
        """
        table = self.per_case()
        if table.empty:
            return pd.DataFrame()

        summary = pd.DataFrame(
            {
                "mean": table.mean(skipna=True),
                "std": table.std(skipna=True, ddof=1),
                "median": table.median(skipna=True),
                "count": table.count(),
                "n_missing": table.isna().sum(),
            }
        )
        return summary

    def reset(self) -> None:
        """Clears all stored cases."""
        self._case_ids = []
        self._records = []

    def __len__(self) -> int:
        """Returns the number of cases stored so far."""
        return len(self._case_ids)
