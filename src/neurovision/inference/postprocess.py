"""Turns raw model logits into a BraTS-submittable segmentation.

The model predicts 3 OVERLAPPING sigmoid region channels, in the fixed order
`(ET, TC, WT)` (`neurovision.data.transforms.REGION_NAMES`), not a 4-way
softmax over mutually exclusive classes. The regions are strictly nested,
`ET subset-of TC subset-of WT`, because that is BraTS anatomy: enhancing
tumor is always inside the tumor core, which is always inside the whole
tumor extent.

There is no `argmax` anywhere in this module. Argmax over these channels
would be wrong twice over: the channels are not mutually exclusive (a voxel
can legitimately be 1 in all three), and argmax would force every background
voxel into one of the three foreground regions instead of allowing "none of
the above". Discretization here is always sigmoid + per-channel threshold
(`neurovision.metrics.segmentation.binarize`), with the nesting invariant
restored explicitly afterward when something has broken it.

Pipeline order (`postprocess_logits`), fixed and documented step by step:
threshold -> optional keep-largest-component -> small-component removal ->
optional ET-volume floor -> re-nesting. The component filters run per
channel independently, which is exactly what breaks nesting (a blob can
survive in ET while the corresponding TC blob is dropped), so nesting is
always restored LAST.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from monai.transforms import KeepLargestConnectedComponent
from monai.transforms.utils import remove_small_objects
from torch import Tensor

from neurovision.metrics.segmentation import binarize

__all__ = [
    "enforce_nesting",
    "remove_small_components",
    "keep_largest_component",
    "regions_to_classes",
    "uncrop_to_original",
    "postprocess_logits",
]

logger = logging.getLogger(__name__)

# Raw, contiguous class values written by scripts/preprocess.py -- must
# mirror neurovision.metrics.segmentation.classes_to_regions exactly, since
# regions_to_classes is that function's inverse.
_NECROTIC_CORE = 1
_EDEMA = 2
_ENHANCING_TUMOR = 3

# Region channel axis is always the 4th-from-last axis: index 0 of a
# (3, D, H, W) case, index 1 of a (B, 3, D, H, W) batch. Using a negative
# axis lets every region-channel function below handle both shapes with the
# same code instead of branching on ndim first.
_REGION_AXIS = -4


def _check_region_shape(regions: Tensor, fn_name: str) -> None:
    """Raises if `regions` is not a (3, D, H, W) or (B, 3, D, H, W) tensor."""
    if regions.ndim not in (4, 5):
        raise ValueError(
            f"{fn_name} expects a (3, D, H, W) or (B, 3, D, H, W) tensor, got shape "
            f"{tuple(regions.shape)}."
        )
    if regions.shape[_REGION_AXIS] != 3:
        raise ValueError(
            f"{fn_name} expects exactly 3 region channels (ET, TC, WT), got shape "
            f"{tuple(regions.shape)}."
        )


def enforce_nesting(regions: Tensor) -> Tensor:
    """Restores `ET subset-of TC subset-of WT` by unioning inner into outer.

    Component filters (`remove_small_components`, `keep_largest_component`)
    run per channel independently, which can drop a blob from TC while the
    same anatomical blob survives in ET -- silently breaking the nesting
    invariant. This function repairs that with a UNION, going inner to
    outer: `TC |= ET`, then `WT |= TC`.

    Union, not intersection, is the deliberate choice: an intersection would
    delete a confidently-predicted ET voxel just because a lower-confidence
    TC channel happened to miss it there, throwing away the model's
    strongest signal to satisfy the weaker one. Union keeps every voxel any
    channel predicted, and only ADDS voxels to the outer regions to restore
    the anatomical containment.

    Args:
        regions: Binary float tensor, channel order `(ET, TC, WT)`, shape
            `(3, D, H, W)` or `(B, 3, D, H, W)`.

    Returns:
        A new tensor of the same shape and dtype with nesting restored.
        A no-op (values unchanged) if `regions` was already nested.

    Raises:
        ValueError: If `regions` does not have exactly 3 region channels.
    """
    _check_region_shape(regions, "enforce_nesting")
    et, tc, wt = regions.unbind(dim=_REGION_AXIS)
    tc_nested = torch.maximum(tc, et)
    wt_nested = torch.maximum(wt, tc_nested)
    return torch.stack([et, tc_nested, wt_nested], dim=_REGION_AXIS)


def _filter_case(case_regions: Tensor, min_size: int, connectivity: int) -> Tensor:
    """Runs `remove_small_objects` on one case's (3, D, H, W) region tensor."""
    filtered = remove_small_objects(case_regions, min_size=min_size, connectivity=connectivity)
    return torch.as_tensor(filtered, dtype=case_regions.dtype)


def remove_small_components(regions: Tensor, min_size: int, connectivity: int = 1) -> Tensor:
    """Drops small connected components from each region channel independently.

    Thin wrapper over MONAI's `remove_small_objects` (itself
    `skimage.morphology.remove_small_objects`), run per channel independently
    so a speckle false positive in one region does not require a matching
    speckle in another to be removed.

    `connectivity` controls what counts as "connected" between two voxels in
    3D: `1` means only face-adjacent neighbours count (6-neighbourhood --
    the 6 voxels sharing a face), `3` means any voxel touching by a face,
    edge, or corner counts (26-neighbourhood). Higher connectivity merges
    more voxels into the same component, so fewer, larger components get
    removed at a given `min_size`.

    IMPORTANT: filtering channels independently BREAKS the ET subset-of TC
    subset-of WT nesting invariant -- a blob can be dropped from the TC
    channel while an overlapping blob survives in ET (or vice versa). This
    function does not repair that itself; every caller must run
    `enforce_nesting` afterward. `postprocess_logits` does this.

    Args:
        regions: Binary float tensor, channel order `(ET, TC, WT)`, shape
            `(3, D, H, W)` or `(B, 3, D, H, W)`.
        min_size: Connected components with fewer voxels than this are
            zeroed out. `min_size <= 0` is a no-op that returns `regions`
            unchanged (and must not raise).
        connectivity: `1` (6-neighbourhood, face-adjacent only) or `3`
            (26-neighbourhood). Passed straight through to
            `skimage.morphology.remove_small_objects`.

    Returns:
        A new tensor of the same shape and dtype as `regions` (unless
        `min_size <= 0`, in which case the input is returned unchanged).

    Raises:
        ValueError: If `regions` does not have exactly 3 region channels.
    """
    _check_region_shape(regions, "remove_small_components")
    if min_size <= 0:
        # Skip MONAI entirely rather than calling it with a no-op min_size:
        # keeps this branch dependency-free and guarantees bit-for-bit
        # identity with the input, which the "no-op" contract promises.
        return regions

    if regions.ndim == 5:
        cases = [_filter_case(regions[b], min_size, connectivity) for b in range(regions.shape[0])]
        return torch.stack(cases, dim=0)
    return _filter_case(regions, min_size, connectivity)


def keep_largest_component(regions: Tensor) -> Tensor:
    """Keeps only the single largest connected component in each region channel.

    Thin wrapper over MONAI's `KeepLargestConnectedComponent`, run with
    `is_onehot=True` (each channel is its own independent binary mask, not a
    mutually-exclusive one-hot label map) and `independent=True` (channels
    are filtered one at a time, so a large TC blob does not "protect" a
    smaller ET blob in a different location). An all-zero channel (no
    predicted voxels at all) stays all-zero rather than raising.

    Aggressive by nature: a real multifocal tumor (more than one lesion) is
    collapsed down to its single largest lesion. `postprocess_logits` gates
    this behind `cfg.inference.postprocess.keep_largest_only`, default off.

    Same nesting caveat as `remove_small_components`: filtering channels
    independently can break `ET subset-of TC subset-of WT`, so callers must
    run `enforce_nesting` afterward.

    Args:
        regions: Binary float tensor, channel order `(ET, TC, WT)`, shape
            `(3, D, H, W)` or `(B, 3, D, H, W)`.

    Returns:
        A new tensor of the same shape and dtype as `regions`.

    Raises:
        ValueError: If `regions` does not have exactly 3 region channels.
    """
    _check_region_shape(regions, "keep_largest_component")
    # applied_labels is passed explicitly (rather than left to MONAI's
    # auto-detection from which channels have any nonzero content) because
    # that auto-detection calls `get_unique_labels(img, is_onehot=True,
    # discard=0)`, which unconditionally discards channel INDEX 0 -- correct
    # for non-onehot label maps (where value 0 means background) but wrong
    # here, where channel 0 is ET, a real foreground region. Left to the
    # default, a case where ET (channel 0) is the only nonzero region
    # channel would silently skip filtering entirely. Naming all 3 channels
    # explicitly sidesteps that path.
    transform = KeepLargestConnectedComponent(
        applied_labels=(0, 1, 2), is_onehot=True, independent=True
    )

    if regions.ndim == 5:
        cases = [
            torch.as_tensor(transform(regions[b]), dtype=regions.dtype)
            for b in range(regions.shape[0])
        ]
        return torch.stack(cases, dim=0)
    return torch.as_tensor(transform(regions), dtype=regions.dtype)


def regions_to_classes(regions: Tensor) -> Tensor:
    """Collapses the 3 nested regions back into a single integer class map.

    This is the inverse of `neurovision.metrics.segmentation.classes_to_regions`
    and the step that REPLACES argmax for this model: it is only valid
    because ET/TC/WT are nested, not because they are mutually exclusive.

    Assignment is OUTER TO INNER, so the inner (more specific) region
    overwrites the outer one:

        out = zeros
        out[WT == 1] = 2   # edema fills the whole whole-tumor extent
        out[TC == 1] = 1   # necrotic/non-enhancing core overwrites
        out[ET == 1] = 3   # enhancing tumor overwrites, innermost

    The order cannot be reversed: assigning inner-to-outer would let the
    WT write LAST and paint class 2 over every ET and TC voxel too,
    silently erasing enhancing tumor from the class map entirely.

    Args:
        regions: Binary float tensor, channel order `(ET, TC, WT)`, shape
            `(3, D, H, W)` or `(B, 3, D, H, W)`. Assumed already nested --
            call `enforce_nesting` first if that is not guaranteed.

    Returns:
        Integer tensor (`uint8` values, held in a `uint8` dtype tensor) with
        values in `{0, 1, 2, 3}`, shape `(D, H, W)` if the input was
        unbatched or `(B, D, H, W)` if it was batched (channel axis dropped,
        batch presence preserved).

    Raises:
        ValueError: If `regions` does not have exactly 3 region channels.
    """
    _check_region_shape(regions, "regions_to_classes")
    et, tc, wt = regions.unbind(dim=_REGION_AXIS)

    out = torch.zeros_like(wt, dtype=torch.uint8)
    out[wt > 0.5] = _EDEMA
    out[tc > 0.5] = _NECROTIC_CORE
    out[et > 0.5] = _ENHANCING_TUMOR  # innermost, overwrites last
    return out


def uncrop_to_original(
    array: np.ndarray,
    bbox: Sequence[Sequence[int]],
    original_shape: Sequence[int],
) -> np.ndarray:
    """Places a cropped-frame prediction back into the original volume geometry.

    Preprocessing (`neurovision.data.preprocessing.compute_nonzero_bbox`)
    crops every case to its nonzero bounding box before training. A
    prediction is made in that cropped frame, but a BraTS submission (or any
    comparison against the original NIfTI) needs it in the ORIGINAL,
    uncropped geometry -- with zeros everywhere the crop removed, since
    those voxels were background (air) in the original scan.

    Args:
        array: The cropped-frame prediction. `(D, H, W)` for a class map, or
            `(C, D, H, W)` for region masks -- which shape it is is inferred
            from `array.ndim` (3 vs 4).
        bbox: Three `[start, end]` pairs (end EXCLUSIVE), in `(D, H, W)`
            axis order, exactly as stored in a case's `meta.json` by
            `neurovision.data.preprocessing.preprocess_case`.
        original_shape: The pre-crop `(D, H, W)` shape, also from
            `meta.json`.

    Returns:
        An array of shape `original_shape` (or `(C, *original_shape)`),
        same dtype as `array`, with `array`'s content placed at the `bbox`
        offset and zeros everywhere outside it.

    Raises:
        ValueError: If `array` is not 3-D or 4-D, if `bbox`/`original_shape`
            do not each have exactly 3 entries, or if the bbox's extent
            (`end - start` per axis) does not match `array`'s spatial shape.
            That mismatch means the prediction and the `meta.json` came from
            different preprocessing runs -- silently broadcasting anyway
            would place the content at the wrong offset in the output and
            produce a spatially shifted submission that looks plausible and
            is wrong.
    """
    array = np.asarray(array)
    bbox = [tuple(pair) for pair in bbox]
    original_shape = tuple(int(s) for s in original_shape)

    if len(bbox) != 3:
        raise ValueError(f"uncrop_to_original expects a 3-axis bbox, got {len(bbox)} entries.")
    if len(original_shape) != 3:
        raise ValueError(
            f"uncrop_to_original expects a 3-entry original_shape, got {original_shape}."
        )

    if array.ndim == 3:
        has_channels = False
        spatial_shape = array.shape
    elif array.ndim == 4:
        has_channels = True
        spatial_shape = array.shape[1:]
    else:
        raise ValueError(
            f"uncrop_to_original expects a (D, H, W) or (C, D, H, W) array, got shape "
            f"{array.shape}."
        )

    bbox_extent = tuple(int(end) - int(start) for start, end in bbox)
    if bbox_extent != tuple(spatial_shape):
        raise ValueError(
            f"bbox extent {bbox_extent} does not match array spatial shape "
            f"{tuple(spatial_shape)}. This means the prediction and meta.json came from "
            "different preprocessing runs; broadcasting anyway would silently produce a "
            "spatially shifted result."
        )

    out_shape = (array.shape[0], *original_shape) if has_channels else original_shape
    out = np.zeros(out_shape, dtype=array.dtype)
    spatial_slices = tuple(slice(int(start), int(end)) for start, end in bbox)
    if has_channels:
        out[(slice(None), *spatial_slices)] = array
    else:
        out[spatial_slices] = array
    return out


def _zero_small_et(regions: Tensor, et_min_volume: float) -> Tensor:
    """Zeros the ET channel for any batch element whose ET voxel count is too low.

    Per-case (per-batch-element), not across the whole batch: one case's ET
    prediction being small must not affect another case's ET channel.
    """
    regions = regions.clone()
    if regions.ndim == 5:
        et_counts = regions[:, 0].sum(dim=(1, 2, 3))  # (B,)
        zero_mask = et_counts < et_min_volume
        regions[zero_mask, 0] = 0.0
    else:
        if regions[0].sum() < et_min_volume:
            regions[0] = 0.0
    return regions


def postprocess_logits(logits: Tensor, cfg: Any) -> Tensor:
    """Runs the full, config-driven postprocessing pipeline on raw logits.

    Reads only `cfg.inference.postprocess.*`. The step order is fixed and
    must not be reordered:

    1. Sigmoid + threshold at `threshold`
       (`neurovision.metrics.segmentation.binarize`).
    2. `keep_largest_component`, if `keep_largest_only` is set.
    3. `remove_small_components` with `min_component_size`/`connectivity`
       (a no-op if `min_component_size <= 0`).
    4. `et_min_volume` floor: if `et_min_volume > 0`, any batch element
       whose ET channel has fewer than `et_min_volume` predicted voxels has
       its ET channel zeroed entirely.

       This is a KNOWN BraTS scoring trick: roughly 35% of cases have no
       enhancing tumor at all, so a model that "gives up" on small/uncertain
       ET predictions and reports nothing scores a Dice of 1.0 on all of
       them under the `ignore_empty=False` convention this project uses
       (see `neurovision.metrics.segmentation.dice_score`). It buys ET Dice
       for free without the model actually being any more accurate. It is
       DEFAULT OFF (`et_min_volume: 0`) in this project, because it makes
       the reported output systematically less honest about the model's own
       uncertainty on small enhancing-tumor cases -- exactly the failure
       mode this project's headline claim (competitive Dice with
       substantially better CALIBRATION) is supposed to catch, not launder.
    5. `enforce_nesting`, if `enforce_nesting` is set. Always LAST: steps
       2-4 each filter or zero a region channel independently, which can
       break `ET subset-of TC subset-of WT`.

    Args:
        logits: Raw (pre-sigmoid) model output, shape `(B, 3, D, H, W)`,
            channel order `(ET, TC, WT)`.
        cfg: The full composed Hydra config; only `cfg.inference.postprocess`
            is read.

    Returns:
        Binary float tensor, shape `(B, 3, D, H, W)`.
    """
    pp_cfg = cfg.inference.postprocess
    applied: list[str] = []

    regions = binarize(logits, threshold=pp_cfg.threshold)
    applied.append(f"binarize(threshold={pp_cfg.threshold})")

    if pp_cfg.keep_largest_only:
        regions = keep_largest_component(regions)
        applied.append("keep_largest_component")

    regions = remove_small_components(
        regions, min_size=pp_cfg.min_component_size, connectivity=pp_cfg.connectivity
    )
    if pp_cfg.min_component_size > 0:
        applied.append(
            f"remove_small_components(min_size={pp_cfg.min_component_size}, "
            f"connectivity={pp_cfg.connectivity})"
        )

    if pp_cfg.et_min_volume > 0:
        regions = _zero_small_et(regions, pp_cfg.et_min_volume)
        applied.append(f"et_min_volume({pp_cfg.et_min_volume})")

    if pp_cfg.enforce_nesting:
        regions = enforce_nesting(regions)
        applied.append("enforce_nesting")

    logger.info("postprocess_logits: steps applied -> %s", applied)
    return regions
