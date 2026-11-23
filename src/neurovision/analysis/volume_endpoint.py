"""A second definition of a "usable" segmentation: tumour-volume error (MasterPlan W2.5).

`error_budget.label_usable` calls a case usable when Dice >= 0.7 in WT and TC. Dice
is an overlap score; a clinician following a tumour over time cares about a
different question -- "is the VOLUME the model reports close to the true volume?"
(volumetric change is what response-assessment practice tracks). This module gives
that second definition so the two can be reported side by side.

Region channel order everywhere in this module is `(ET, TC, WT)`, verified from
`neurovision.data.transforms.REGION_NAMES` (transforms.py:47) and from
`ConvertToRegionsd._convert` (transforms.py:134-141, which stacks `[et, tc, wt]`).
Raw label classes are background 0 and contiguous 1..3 (necrotic core 1, edema 2,
enhancing tumour 3), so ET = {3}, TC = {1, 3}, WT = {1, 2, 3}.

**Pure numpy/pandas: no file I/O, no model, no label leakage into any gate.**
Volumes are in millilitres (1 ml = 1000 mm^3).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Order matches `neurovision.data.transforms.REGION_NAMES`; channel i of a region
# mask is REGIONS[i].
REGIONS: tuple[str, ...] = ("ET", "TC", "WT")

# Raw class sets for each region (transforms.py:134-136).
_REGION_CLASSES: dict[str, tuple[int, ...]] = {
    "ET": (3,),
    "TC": (1, 3),
    "WT": (1, 2, 3),
}

_MM3_PER_ML = 1000.0


def _voxel_ml(spacing_mm: Sequence[float]) -> float:
    """Volume of one voxel in ml: product of the three spacings (mm) / 1000."""
    if len(spacing_mm) != 3:
        raise ValueError(f"spacing_mm must have 3 entries, got {len(spacing_mm)}.")
    return float(np.prod([float(s) for s in spacing_mm])) / _MM3_PER_ML


def region_volumes(
    regions: np.ndarray | torch.Tensor, spacing_mm: Sequence[float]
) -> dict[str, float]:
    """Volume of each region mask, in ml.

    Args:
        regions: Boolean or 0/1 mask `(3, D, H, W)`, channel order `(ET, TC, WT)`.
        spacing_mm: Voxel spacing `(sD, sH, sW)` in mm, matching the spatial axes.

    Returns:
        `{"vol_ET_ml", "vol_TC_ml", "vol_WT_ml"}`: voxel count x voxel volume / 1000.

    Raises:
        ValueError: `regions` is not `(3, D, H, W)`, or `spacing_mm` is not length 3.
    """
    arr = (
        regions.detach().cpu().numpy() if isinstance(regions, torch.Tensor) else np.asarray(regions)
    )
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(f"region_volumes expects shape (3, D, H, W), got {tuple(arr.shape)}.")
    voxel_ml = _voxel_ml(spacing_mm)
    # `> 0.5` accepts both bool and 0.0/1.0 float masks.
    return {
        f"vol_{name}_ml": float((arr[i] > 0.5).sum()) * voxel_ml for i, name in enumerate(REGIONS)
    }


def label_volumes(label: np.ndarray, spacing_mm: Sequence[float]) -> dict[str, float]:
    """Volume of ET, TC and WT in ml from a raw class label.

    Args:
        label: Integer class label `(D, H, W)` with values in {0, 1, 2, 3}.
        spacing_mm: Voxel spacing `(sD, sH, sW)` in mm.

    Returns:
        Same keys as `region_volumes`. ET = class 3, TC = classes 1 and 3,
        WT = classes 1, 2 and 3, so ET <= TC <= WT by construction.

    Raises:
        ValueError: `label` is not 3-D, or `spacing_mm` is not length 3.
    """
    arr = np.asarray(label)
    if arr.ndim != 3:
        raise ValueError(f"label_volumes expects shape (D, H, W), got {tuple(arr.shape)}.")
    voxel_ml = _voxel_ml(spacing_mm)
    return {
        f"vol_{name}_ml": float(np.isin(arr, classes).sum()) * voxel_ml
        for name, classes in _REGION_CLASSES.items()
    }


def volume_error_table(pred: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per-case predicted vs reference volume and relative error.

    Args:
        pred: One row per case (index = case_id) with `vol_<R>_ml` columns, R in ET, TC, WT.
        ref: Same layout, from the ground-truth label, on the SAME index.

    Returns:
        A `DataFrame` on that index with, for each R: `vol_pred_<R>_ml`,
        `vol_ref_<R>_ml`, `vol_rel_err_<R>` = |pred - ref| / ref, and boolean
        `ref_empty_<R>`. When ref == 0 the relative error is undefined (division by
        zero), so it is NaN and `ref_empty_<R>` is True; `label_usable_volume`
        handles that case separately.

    Raises:
        ValueError: The indexes differ, or a `vol_<R>_ml` column is missing.
    """
    if not pred.index.equals(ref.index):
        raise ValueError("volume_error_table: `pred` and `ref` must share exactly the same index.")
    needed = [f"vol_{r}_ml" for r in REGIONS]
    for name, df in (("pred", pred), ("ref", ref)):
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"volume_error_table: missing column(s) {missing} in `{name}`.")

    out = pd.DataFrame(index=pred.index)
    for r in REGIONS:
        p = pred[f"vol_{r}_ml"].astype(float)
        t = ref[f"vol_{r}_ml"].astype(float)
        empty = t == 0
        out[f"vol_pred_{r}_ml"] = p
        out[f"vol_ref_{r}_ml"] = t
        # Replace 0 with NaN before dividing so we never emit inf.
        out[f"vol_rel_err_{r}"] = (p - t).abs() / t.where(~empty)
        out[f"ref_empty_{r}"] = empty
    return out


def label_usable_volume(
    vol_table: pd.DataFrame,
    *,
    regions: Sequence[str] = ("WT", "ET"),
    tol: float = 0.25,
    empty_pred_max_ml: float = 0.0,
) -> pd.Series:
    """The per-case volume-based usability label.

    A case is usable iff, for EVERY region in `regions`, either
    (reference non-empty AND `vol_rel_err_<R> <= tol`) or
    (reference empty AND `vol_pred_<R>_ml <= empty_pred_max_ml`).

    The default tolerance of 25% follows response-assessment practice for
    volumetric change: a change smaller than that is not treated as real, so a
    volume error inside it would not alter a clinical reading.

    Args:
        vol_table: Output of `volume_error_table`.
        regions: Regions the rule applies to. ET is included by default (unlike the
            Dice label) because empty-reference cases are handled explicitly.
        tol: Maximum relative volume error, > 0. The bound is inclusive.
        empty_pred_max_ml: When the reference is empty, the largest predicted volume
            (ml) still counted as correct. 0.0 means any predicted voxel is a failure.

    Returns:
        Boolean `Series` aligned to `vol_table.index`. NaN in any required value
        (other than the by-design NaN rel_err on an empty reference) -> False.

    Raises:
        ValueError: `tol <= 0`, or a required column is absent (named in the message).
    """
    # TODO(author): cite source for the 25% default tolerance (response-assessment practice).
    if not tol > 0:
        raise ValueError(f"label_usable_volume: `tol` must be > 0, got {tol}.")
    missing = [
        c
        for r in regions
        for c in (f"vol_pred_{r}_ml", f"vol_rel_err_{r}", f"ref_empty_{r}")
        if c not in vol_table.columns
    ]
    if missing:
        raise ValueError(f"label_usable_volume: missing column(s) {missing} in `vol_table`.")

    usable = pd.Series(True, index=vol_table.index)
    for r in regions:
        empty = vol_table[f"ref_empty_{r}"].astype(bool)
        # NaN <= x is False under IEEE comparison, so a NaN error or volume falls
        # out as not usable without a separate branch.
        nonempty_ok = ~empty & (vol_table[f"vol_rel_err_{r}"] <= tol)
        empty_ok = empty & (vol_table[f"vol_pred_{r}_ml"] <= empty_pred_max_ml)
        usable &= nonempty_ok | empty_ok
    return usable


def usable_both(dice_usable: pd.Series, volume_usable: pd.Series) -> pd.DataFrame:
    """Put the Dice-based and volume-based labels side by side.

    Args:
        dice_usable: Boolean `Series`, e.g. `error_budget.label_usable(...)`.
        volume_usable: Boolean `Series` from `label_usable_volume`, same index.

    Returns:
        `DataFrame` with `usable_dice`, `usable_volume`, `usable_both` (AND) and
        `usable_either` (OR).

    Raises:
        ValueError: The two indexes are not exactly equal.
    """
    if not dice_usable.index.equals(volume_usable.index):
        raise ValueError(
            "usable_both: `dice_usable` and `volume_usable` must share the same index."
        )
    d = dice_usable.astype(bool)
    v = volume_usable.astype(bool)
    return pd.DataFrame(
        {"usable_dice": d, "usable_volume": v, "usable_both": d & v, "usable_either": d | v}
    )
