"""Distance-to-boundary primitives and boundary-stratified error metrics.

`metrics/segmentation.py` scores whole regions (Dice, IoU, HD95) — HD95 is a
single scalar summary of boundary error. What the boundary-accuracy half of
this project's research claim needs on top of that is error as a FUNCTION of
distance to the true tumor margin: "the model errs less in the 0-2 mm shell
around the true boundary than the baseline does". This module supplies that,
plus the underlying signed-distance-transform and per-band reduction
primitives generic enough for a second use — correlating a fusion gate map or
an MC-dropout mutual-information map against distance to the boundary.

Every public function here does its own `.detach().cpu().numpy()`, because
`scipy.ndimage.distance_transform_edt` cannot consume a CUDA tensor and this
module has no other way to guard against that in a CPU-only test suite.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from neurovision.data.transforms import REGION_NAMES

__all__ = [
    "DEFAULT_BANDS",
    "signed_distance_to_boundary",
    "boundary_band_masks",
    "band_label",
    "boundary_stratified_errors",
    "distance_band_means",
]

logger = logging.getLogger(__name__)

# Millimetres when `spacing` is supplied to `signed_distance_to_boundary`,
# voxels otherwise. Half-open [lo, hi) bands, so they tile [0, inf) exactly
# once with no boundary-voxel double counting.
DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, float("inf")),
)


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """Normalizes a tensor-or-array entry point to a CPU numpy array.

    Args:
        x: Input, possibly a CUDA tensor. `scipy` can only consume CPU numpy
            arrays, so every public function in this module funnels its
            array-like inputs through this helper before touching scipy.

    Returns:
        A numpy array with the same values as `x`.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def signed_distance_to_boundary(
    mask: Tensor | np.ndarray,
    spacing: Sequence[float] | None = None,
    name: str | None = None,
) -> Tensor:
    """Computes the signed Euclidean distance transform of a binary mask.

    Convention: NEGATIVE inside the mask, POSITIVE outside. No voxel ever has
    distance exactly 0 under this convention — a voxel immediately inside the
    true surface gets `-1` (times spacing), one immediately outside gets
    `+1`; the surface itself lies between voxel centres. This is what makes
    half-open `[lo, hi)` bands over `abs(sdf)` in `boundary_band_masks` tile
    space with no "boundary voxel" that could be double counted.

    Args:
        mask: Binary mask, shape `(D, H, W)`. Any dtype; values `> 0` are
            treated as True.
        spacing: Physical voxel size per axis, e.g. `(1.0, 1.0, 1.0)` for
            BraTS's 1mm isotropic resolution. Passed straight through to
            `distance_transform_edt`'s `sampling`. `None` means distances are
            reported in voxels.
        name: Optional label used only to identify this input in the
            degenerate-case warning below.

    Returns:
        Float32 CPU tensor, shape `(D, H, W)`.

    Raises:
        ValueError: If `mask` is not 3-D, or `spacing` is given with a
            length other than 3.
    """
    m = _to_numpy(mask)
    if m.ndim != 3:
        raise ValueError(f"signed_distance_to_boundary expects a (D, H, W) mask, got {m.shape}.")
    if spacing is not None and len(spacing) != 3:
        raise ValueError(f"spacing must have length 3 to match a (D, H, W) mask, got {spacing!r}.")

    m = m > 0

    # distance_transform_edt of an array with no zero voxels (all-True) or no
    # nonzero voxels (all-False) does NOT raise -- it returns a meaningless
    # filled array (there is no boundary to measure distance to). Both
    # degeneracies must be caught up front rather than let scipy's silently
    # wrong output flow downstream.
    if not m.any():
        logger.warning(
            "signed_distance_to_boundary: mask %s is entirely empty (no foreground "
            "voxels) -- there is no boundary. Returning an all-NaN distance map.",
            name if name is not None else "<unnamed>",
        )
        return torch.full(m.shape, float("nan"), dtype=torch.float32)
    if m.all():
        logger.warning(
            "signed_distance_to_boundary: mask %s is entirely full (no background "
            "voxels) -- there is no boundary. Returning an all-NaN distance map.",
            name if name is not None else "<unnamed>",
        )
        return torch.full(m.shape, float("nan"), dtype=torch.float32)

    outside = distance_transform_edt(~m, sampling=spacing)
    inside = distance_transform_edt(m, sampling=spacing)
    sdf = outside - inside
    return torch.from_numpy(sdf.astype(np.float32))


def boundary_band_masks(
    sdf: Tensor,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    signed: bool = False,
) -> list[Tensor]:
    """Splits a signed distance field into per-band boolean masks.

    Args:
        sdf: Output of `signed_distance_to_boundary`, shape `(D, H, W)`.
        bands: Half-open `[lo, hi)` intervals, e.g. `DEFAULT_BANDS`. Must be
            pairwise non-overlapping. They need not be given in ascending
            order -- the overlap check sorts a copy, so it is
            order-independent, while the RETURNED list stays in the caller's
            own order so `zip(bands, masks)` is always correct.
        signed: If False (default), band `k` selects
            `lo <= abs(sdf) < hi` -- distance from the boundary regardless of
            side. If True, band `k` selects `lo <= sdf < hi` directly, so a
            caller can request the inside (negative `lo`/`hi`) and outside
            (positive `lo`/`hi`) shells separately.

    Returns:
        One bool CPU tensor per band, same shape as `sdf`. Every NaN voxel of
        `sdf` (i.e. a degenerate, boundary-less region) is False in every
        band, since a NaN comparison is always False.

    Raises:
        ValueError: If any band has `lo >= hi`, or two bands overlap.
    """
    sorted_bands = sorted(bands, key=lambda b: b[0])
    for lo, hi in sorted_bands:
        if not lo < hi:
            raise ValueError(f"boundary_band_masks: band ({lo}, {hi}) must have lo < hi.")
    for (lo_a, hi_a), (lo_b, hi_b) in zip(sorted_bands, sorted_bands[1:], strict=False):
        if hi_a > lo_b:
            raise ValueError(
                f"boundary_band_masks: bands ({lo_a}, {hi_a}) and ({lo_b}, {hi_b}) overlap."
            )

    reference = torch.abs(sdf) if not signed else sdf
    masks: list[Tensor] = []
    for lo, hi in bands:
        masks.append((reference >= lo) & (reference < hi))
    return masks


def band_label(lo: float, hi: float) -> str:
    """Formats one band as a stable, CSV-safe column-name suffix.

    Args:
        lo: Lower (inclusive) edge of the band.
        hi: Upper (exclusive) edge of the band. May be `float("inf")`.

    Returns:
        `f"{lo:g}-{hi:g}"`, with `inf`/`-inf` rendered literally, e.g.
        `(0.0, 2.0) -> "0-2"`, `(10.0, inf) -> "10-inf"`.
    """

    def _fmt(v: float) -> str:
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:g}"

    return f"{_fmt(lo)}-{_fmt(hi)}"


def boundary_stratified_errors(
    pred: Tensor,
    target: Tensor,
    region_names: Sequence[str] = REGION_NAMES,
    spacing: Sequence[float] | None = None,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
) -> dict[str, float]:
    """Computes per-region, per-distance-band error rates for a single case.

    The stratifying distance field is computed on `target` (the ground
    truth), never on `pred`. This is load-bearing: if each model's own
    prediction defined the bins, two models would be stratified by two
    different partitions of space and their per-band numbers would not be
    comparable, which is the entire purpose of this table.

    Args:
        pred: Binary prediction for ONE case, shape `(C, D, H, W)` or
            `(1, C, D, H, W)`.
        target: Binary ground truth, same shape convention as `pred`.
        region_names: Region name per channel, in channel order. Defaults to
            `REGION_NAMES` (`("ET", "TC", "WT")`).
        spacing: Voxel spacing in mm passed through to
            `signed_distance_to_boundary`.
        bands: Half-open `[lo, hi)` distance bands. Defaults to
            `DEFAULT_BANDS`.

    Returns:
        Flat dict of float values. For each region `R` and each band label
        `L`: `berr_R_L` (fraction of that band's voxels where
        `pred != target`), `bfnr_R_L` (fraction where `target == 1` and
        `pred == 0`, missed tumor), `bfpr_R_L` (fraction where `target == 0`
        and `pred == 1`, spurious tumor), `bn_R_L` (voxel count in that band,
        as a float). `berr == bfnr + bfpr` exactly, by construction. A band
        with zero voxels gets NaN rates and `bn = 0.0`. A region absent from
        `target` gets an all-NaN distance field, so every band for that
        region is NaN/0.0 -- no extra warning beyond the one
        `signed_distance_to_boundary` already logs for that region.

    Raises:
        ValueError: If `pred`/`target` do not have a batch size of at most 1,
            their shapes disagree, or `len(region_names) != pred.shape[1]`
            (an under-sized `region_names` would silently drop a region from
            the output).
    """
    if pred.ndim == 4:
        pred = pred.unsqueeze(0)
    if target.ndim == 4:
        target = target.unsqueeze(0)
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(
            "boundary_stratified_errors expects (C, D, H, W) or (1, C, D, H, W) inputs, got "
            f"pred shape {tuple(pred.shape)} and target shape {tuple(target.shape)}."
        )
    if pred.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError(
            f"boundary_stratified_errors is for a single case (batch size 1), got pred "
            f"batch {pred.shape[0]} and target batch {target.shape[0]}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {tuple(pred.shape)} "
            f"and {tuple(target.shape)}."
        )
    if len(region_names) != pred.shape[1]:
        raise ValueError(
            f"region_names has {len(region_names)} entries {tuple(region_names)} but "
            f"pred has {pred.shape[1]} channels. One name per channel is required."
        )

    pred_np = _to_numpy(pred)[0]  # (C, D, H, W)
    target_np = _to_numpy(target)[0]  # (C, D, H, W)

    metrics: dict[str, float] = {}
    for c, region in enumerate(region_names):
        sdf = signed_distance_to_boundary(
            torch.from_numpy(target_np[c]), spacing=spacing, name=f"{region} (target)"
        )
        band_masks = boundary_band_masks(sdf, bands=bands, signed=False)

        pred_c = torch.from_numpy(pred_np[c]) > 0
        target_c = torch.from_numpy(target_np[c]) > 0
        mismatch = pred_c != target_c
        fnr_voxels = target_c & ~pred_c
        fpr_voxels = ~target_c & pred_c

        for (lo, hi), band_mask in zip(bands, band_masks, strict=True):
            label = band_label(lo, hi)
            n = int(band_mask.sum())
            if n == 0:
                metrics[f"berr_{region}_{label}"] = float("nan")
                metrics[f"bfnr_{region}_{label}"] = float("nan")
                metrics[f"bfpr_{region}_{label}"] = float("nan")
                metrics[f"bn_{region}_{label}"] = 0.0
                continue
            metrics[f"berr_{region}_{label}"] = float(mismatch[band_mask].sum()) / n
            metrics[f"bfnr_{region}_{label}"] = float(fnr_voxels[band_mask].sum()) / n
            metrics[f"bfpr_{region}_{label}"] = float(fpr_voxels[band_mask].sum()) / n
            metrics[f"bn_{region}_{label}"] = float(n)

    return metrics


def distance_band_means(
    values: Tensor | np.ndarray,
    sdf: Tensor,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    signed: bool = False,
) -> dict[str, float]:
    """Reduces an arbitrary per-voxel quantity into per-distance-band means.

    Generic reducer shared by the gate-map and MC-dropout uncertainty-map
    analyses: both ask "how does quantity X behave as a function of distance
    to the true tumor boundary". Upsampling a coarser map (e.g. a stride-4
    fusion gate) to match `sdf`'s resolution is the caller's job -- the right
    interpolation depends on what the quantity means, which this function has
    no way to know.

    Args:
        values: Any per-voxel float quantity, shape `(D, H, W)`, same shape
            as `sdf`.
        sdf: Output of `signed_distance_to_boundary`, shape `(D, H, W)`.
        bands: Half-open `[lo, hi)` distance bands. Defaults to
            `DEFAULT_BANDS`.
        signed: Forwarded to `boundary_band_masks`.

    Returns:
        `{f"mean_{label}": float, f"n_{label}": float}` per band. NaN
        entries in `values` are skipped within a band (nan-aware mean) and
        are not counted in `n_`. An empty band (zero contributing voxels)
        gets NaN mean and `n = 0.0`.

    Raises:
        ValueError: If `values` and `sdf` have different shapes.
    """
    values_np = _to_numpy(values)
    sdf_np = _to_numpy(sdf)
    if values_np.shape != sdf_np.shape:
        raise ValueError(
            "distance_band_means: values and sdf must have the same shape, got "
            f"values {values_np.shape} and sdf {sdf_np.shape}."
        )

    values_t = torch.from_numpy(values_np.astype(np.float32))
    # Re-wrapped rather than forwarding `sdf` itself: this function accepts an
    # ndarray, and `boundary_band_masks` compares with `torch.abs`, which an
    # ndarray would not survive.
    sdf_t = torch.from_numpy(sdf_np.astype(np.float32))
    band_masks = boundary_band_masks(sdf_t, bands=bands, signed=signed)

    out: dict[str, float] = {}
    for (lo, hi), band_mask in zip(bands, band_masks, strict=True):
        label = band_label(lo, hi)
        band_values = values_t[band_mask]
        valid = band_values[~torch.isnan(band_values)]
        n = int(valid.numel())
        out[f"mean_{label}"] = float(valid.mean()) if n > 0 else float("nan")
        out[f"n_{label}"] = float(n)
    return out
