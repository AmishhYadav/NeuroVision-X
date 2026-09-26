"""Scoring for the real-DICOM validation protocol (P1.2).

Binding design: `docs/research/real_dicom_validation_protocol.md`, section
"Scoring -- fixed now". Read that file first; this module implements only
what it fixes, with no IO and no Hydra -- the driver that locates a job's
`meta.json`/prediction and the matching `data/preprocessed/brats/<case>/`
ground truth is a separate, later module.

## What "case dir format" means here

Both the research-path ground truth and the clinical job's own output live
as a case directory holding a `meta.json` (with `original_shape`,
`cropped_shape`, `bbox`, an `affine` describing the ORIGINAL full grid after
the project's own reorientation, and optionally `source_axcodes` -- older
`meta.json` files lack that key, and nothing here requires it) plus a
cropped `(D, H, W)` integer array in the project's label convention
`{0, 1, 2, 3}` (1 = NCR/NET, 2 = ED, 3 = ET; see
`neurovision.data.preprocessing.remap_labels`). Every function below is pure
numpy in, numpy out, CPU only -- HD95 is computed by
`neurovision.metrics.segmentation.hd95`, which must never see a CUDA tensor
(trap 8 in `docs/lessons.md`).

## Why the lateralisation check does not use brain-mask Dice

Trap 3 (`docs/lessons.md`): a brain is nearly left-right symmetric, so
brain-mask Dice scores *higher* on a left-right mirrored atlas (0.9416)
than on the correctly oriented one (0.9394) -- it is structurally blind to
the exact failure it would need to catch. `lateralisation_check` instead
uses the tumour's own WT mask, which is content, not geometry: a real
lesion is not symmetric, so its Dice against a flipped copy of itself is a
real signal a symmetric brain mask cannot give.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import nibabel as nib
import numpy as np
import torch

from neurovision.inference.postprocess import uncrop_to_original
from neurovision.metrics import REGION_NAMES, classes_to_regions, dice_score, hd95

__all__ = [
    "uncrop_to_full",
    "to_reference_grid",
    "region_masks",
    "score_case",
    "roundtrip_self_test",
    "lateralisation_check",
]

logger = logging.getLogger(__name__)


def uncrop_to_full(cropped: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """Places a cropped label/mask back into its full, uncropped geometry.

    Thin wrapper over `neurovision.inference.postprocess.uncrop_to_original`
    -- that function already does exactly this (zero-fill outside the bbox,
    raise on a shape mismatch), so it is reused rather than reimplemented.

    Args:
        cropped: `(D, H, W)` integer label or boolean mask, in the case
            dir's cropped frame.
        meta: The case's `meta.json`, read as a dict. Must have `bbox`
            (three `[start, end]` pairs, `end` exclusive) and
            `original_shape` (3 ints).

    Returns:
        `(D, H, W)` array shaped `meta["original_shape"]`, `cropped`'s
        dtype, zero everywhere the crop removed.

    Raises:
        ValueError: If `cropped.shape` does not match the extents implied
            by `meta["bbox"]` -- meaning `cropped` and `meta` came from
            different preprocessing runs.
    """
    return uncrop_to_original(cropped, meta["bbox"], meta["original_shape"])


def to_reference_grid(
    full: np.ndarray,
    affine: np.ndarray,
    reference_affine: np.ndarray,
    *,
    atol: float = 1e-3,
) -> np.ndarray:
    """Maps a full-grid integer array onto a reference grid, flip/permute only.

    Uses nibabel's orientation machinery (`io_orientation` / `ornt_transform`
    / `apply_orientation`) the same way
    `neurovision.data.preprocessing.reorient_to_axcodes` does, except the
    target is another AFFINE rather than a target axcode string -- both
    affines are read into nibabel's canonical-RAS orientation description
    and the transform between them is applied. This only flips and/or
    transposes voxel axes; it never resamples or interpolates, so integer
    label values are preserved exactly. That is deliberate: a real spacing
    or origin mismatch between the two grids is a registration problem to
    be reported, not a discrepancy to silently blur away with resampling.

    Args:
        full: `(D, H, W)` array already in `affine`'s full (uncropped) grid.
        affine: `full`'s current 4x4 affine.
        reference_affine: The 4x4 affine `full` should be expressed in.
        atol: Absolute tolerance for the affine-equality check performed
            after reorienting.

    Returns:
        `full` reoriented (flipped/transposed, values unchanged) so that its
        affine equals `reference_affine` within `atol`. Returned unchanged
        (a no-op) when `affine` already equals `reference_affine`.

    Raises:
        ValueError: If the reoriented affine does not equal
            `reference_affine` within `atol` -- meaning the two grids differ
            by more than an axis permutation/flip (e.g. a spacing or origin
            mismatch), which this function refuses to paper over by
            resampling. Both affines are included in the message.
    """
    full = np.asarray(full)
    affine = np.asarray(affine, dtype=np.float64)
    reference_affine = np.asarray(reference_affine, dtype=np.float64)

    current_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.io_orientation(reference_affine)
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)

    reoriented = nib.orientations.apply_orientation(full, transform)
    updated_affine = affine @ nib.orientations.inv_ornt_aff(transform, full.shape)

    # io_orientation only sees axis order and flip sign, never spacing or
    # origin -- so two affines that agree on orientation but disagree on
    # scale or translation pass the ornt_transform step above with an
    # identity transform, and only THIS check catches them.
    if not np.allclose(updated_affine, reference_affine, atol=atol):
        raise ValueError(
            "to_reference_grid: the reoriented affine does not match reference_affine "
            f"within atol={atol}. This function assumes the two grids differ only by an "
            "axis permutation/flip of the same voxel lattice, and refuses to resample a "
            "real spacing/origin mismatch away silently -- that is a registration problem "
            f"to report, not hide.\nreoriented affine:\n{updated_affine}\n"
            f"reference affine:\n{reference_affine}"
        )
    return reoriented


def region_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    """ET/TC/WT boolean masks from project-convention integer labels.

    Thin numpy wrapper over the project's own label-to-region conversion,
    `neurovision.metrics.classes_to_regions` (mirrors
    `neurovision.data.transforms.ConvertToRegionsd` exactly), so region
    nesting logic is defined in exactly one place.

    Args:
        labels: `(D, H, W)` integer array, values in `{0, 1, 2, 3}`.

    Returns:
        Dict with keys `"ET"`, `"TC"`, `"WT"` (from
        `neurovision.metrics.REGION_NAMES`), each a boolean `(D, H, W)`
        array.
    """
    label_t = torch.from_numpy(np.asarray(labels)).long()
    regions = classes_to_regions(label_t)  # (1, 3, D, H, W) float32
    regions_np = regions[0].numpy().astype(bool)  # (3, D, H, W)
    return {name: regions_np[i] for i, name in enumerate(REGION_NAMES)}


def _regions_to_tensor(labels: np.ndarray) -> torch.Tensor:
    """Labels -> the `(1, 3, D, H, W)` float32 region tensor the metrics module expects."""
    masks = region_masks(labels)
    stacked = np.stack([masks[name] for name in REGION_NAMES], axis=0)  # (3, D, H, W)
    return torch.from_numpy(stacked).unsqueeze(0).to(dtype=torch.float32)  # (1, 3, D, H, W)


def score_case(
    pred_labels_full: np.ndarray,
    gt_labels_full: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Voxel Dice and HD95 per region, on two full-grid label volumes.

    Reuses the project's own metric functions
    (`neurovision.metrics.dice_score`, `neurovision.metrics.hd95`) rather
    than reimplementing either -- both run on CPU tensors here, since
    `hd95` must never see a CUDA tensor (trap 8).

    Args:
        pred_labels_full: `(D, H, W)` integer prediction, values in
            `{0, 1, 2, 3}`, already on the same grid as `gt_labels_full`.
        gt_labels_full: `(D, H, W)` integer ground truth, same convention.
        spacing: Voxel spacing in mm, `(dz, dy, dx)`, passed to `hd95` so
            distances are reported in millimeters rather than voxels.

    Returns:
        Dict with `dice_ET`, `dice_TC`, `dice_WT` (Dice, `ignore_empty=False`:
        an empty ground-truth region scores 1.0 if the prediction is also
        empty for that region, else 0.0 -- see `neurovision.metrics.dice_score`)
        and `hd95_ET`, `hd95_TC`, `hd95_WT` (95th-percentile Hausdorff
        distance in mm; both masks empty -> 0.0, exactly one empty -> NaN,
        since that distance is genuinely undefined rather than zero -- see
        `neurovision.metrics.hd95`).

    Raises:
        ValueError: If `pred_labels_full.shape != gt_labels_full.shape`.
    """
    pred_labels_full = np.asarray(pred_labels_full)
    gt_labels_full = np.asarray(gt_labels_full)
    if pred_labels_full.shape != gt_labels_full.shape:
        raise ValueError(
            "score_case: pred and gt must have the same shape, got "
            f"{pred_labels_full.shape} and {gt_labels_full.shape}."
        )

    pred_t = _regions_to_tensor(pred_labels_full)
    gt_t = _regions_to_tensor(gt_labels_full)

    dice = dice_score(pred_t, gt_t, ignore_empty=False)[0]  # (3,)
    hausdorff = hd95(pred_t, gt_t, spacing=spacing)[0]  # (3,)

    metrics: dict[str, float] = {}
    for i, name in enumerate(REGION_NAMES):
        metrics[f"dice_{name}"] = float(dice[i])
        metrics[f"hd95_{name}"] = float(hausdorff[i])
    return metrics


def roundtrip_self_test(gt_cropped: np.ndarray, meta: Mapping[str, Any]) -> dict[str, float]:
    """Scores a ground-truth round trip against itself; every Dice must be 1.0.

    Uncrops `gt_cropped` and scores the result against itself, so this
    exercises exactly the uncrop path a real study will go through before
    any study is actually scored (per the protocol's "self-test first"
    rule). A prediction and its own ground truth being the same array makes
    every region's Dice exactly 1.0 by construction; anything else means
    the uncrop path itself is broken.

    Args:
        gt_cropped: `(D, H, W)` cropped ground-truth label array.
        meta: The matching case's `meta.json`, as a dict (needs `bbox`,
            `original_shape`, `spacing`).

    Returns:
        The `score_case` dict for the self-comparison (all `dice_*` values
        exactly 1.0).

    Raises:
        RuntimeError: If any `dice_*` value is not exactly 1.0 -- the
            round trip itself is broken, and no real study should be scored
            until it is fixed.
    """
    full = uncrop_to_full(np.asarray(gt_cropped), meta)
    spacing = tuple(float(s) for s in meta["spacing"])
    metrics = score_case(full, full, spacing)

    bad = {k: v for k, v in metrics.items() if k.startswith("dice_") and v != 1.0}
    if bad:
        raise RuntimeError(
            "roundtrip_self_test: ground-truth round trip did not score Dice 1.0 on every "
            f"region: {bad}. The uncrop path is broken; no study should be scored until this "
            "is fixed."
        )
    return metrics


def lateralisation_check(
    pred_labels_full: np.ndarray,
    gt_labels_full: np.ndarray,
    affine: np.ndarray,
    *,
    margin: float = 0.05,
) -> str:
    """Detects a left-right flip between prediction and ground truth, from tumour content.

    Finds the left-right voxel axis from `affine` (whichever axis
    `nibabel.aff2axcodes` labels `'L'` or `'R'`), then compares WT Dice of
    (pred, gt) against WT Dice of (pred, gt flipped along that axis). See
    the module docstring for why this uses the tumour's own WT mask and
    never brain-mask Dice (trap 3: brain-mask Dice is nearly symmetric and
    scores *higher* on a mirrored volume).

    Args:
        pred_labels_full: `(D, H, W)` integer prediction, full grid.
        gt_labels_full: `(D, H, W)` integer ground truth, same grid.
        affine: The shared 4x4 affine for both arrays.
        margin: How much higher the flipped Dice must be than the direct
            Dice before this is called `"mirrored"` rather than
            `"ambiguous"` -- guards against a near-symmetric (midline)
            lesion, where direct and flipped Dice are both plausible and
            neither reading should be trusted.

    Returns:
        `"mirrored"` if flipped Dice > direct Dice + `margin`; `"ok"` if
        direct Dice >= flipped Dice; `"ambiguous"` otherwise (flipped is
        higher, but by less than `margin` -- a midline tumour, where this
        check cannot tell).

    Raises:
        ValueError: If the two label arrays' shapes differ, or `affine` has
            no axis nibabel identifies as left-right.
    """
    pred_labels_full = np.asarray(pred_labels_full)
    gt_labels_full = np.asarray(gt_labels_full)
    if pred_labels_full.shape != gt_labels_full.shape:
        raise ValueError(
            "lateralisation_check: pred and gt must have the same shape, got "
            f"{pred_labels_full.shape} and {gt_labels_full.shape}."
        )

    axcodes = nib.aff2axcodes(np.asarray(affine, dtype=np.float64))
    lr_axes = [axis for axis, code in enumerate(axcodes) if code in ("L", "R")]
    if not lr_axes:
        raise ValueError(
            f"lateralisation_check: affine's axis codes {axcodes} have no L/R axis to flip."
        )
    lr_axis = lr_axes[0]

    pred_wt = region_masks(pred_labels_full)["WT"]
    gt_wt = region_masks(gt_labels_full)["WT"]
    # np.flip returns a negative-stride view; torch can't wrap that
    # directly, so copy it into contiguous memory before it reaches
    # _dice via region-mask-shaped tensors.
    gt_wt_flipped = np.ascontiguousarray(np.flip(gt_wt, axis=lr_axis))

    direct = _wt_dice(pred_wt, gt_wt)
    flipped = _wt_dice(pred_wt, gt_wt_flipped)

    if flipped > direct + margin:
        return "mirrored"
    if direct >= flipped:
        return "ok"
    return "ambiguous"


def _wt_dice(a: np.ndarray, b: np.ndarray) -> float:
    """WT-only Dice between two boolean `(D, H, W)` masks, via the project's own `dice_score`."""
    a_t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    b_t = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))[None, None]
    return float(dice_score(a_t, b_t, ignore_empty=False)[0, 0])
