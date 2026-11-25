"""Carried reference for the real-DICOM v2 protocol (MasterPlan W3.4, Amendment 1 step a).

BraTS registered each scan to SRI24 itself, and our clinical pipeline registers the
same scan to SRI24 on its own; the two registrations disagree by several mm. Scoring
our mask against BraTS's label in BraTS's alignment therefore mostly measures that
disagreement. This module removes it: the BraTS-preprocessed T1CE (same acquisition
as ours) is rigidly registered onto OUR atlas-space T1CE, and the same transform is
applied to the BraTS label. Our mask is then scored against the carried label on our
grid.

Why RIGID and not affine: both images are the same acquisition, so only the pose
differs; extra scaling/shear freedom would just let the optimiser deform anatomy.

Why NEAREST NEIGHBOUR for labels: interpolating label values (1, 2, 3) would invent
values such as 2.5; nearest neighbour only ever copies an existing label.

`ants` (antspyx) lives only in `.venv-clinical`, so it is imported lazily inside
`carry_label`; the main `.venv` can still import this module.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from neurovision.analysis.real_dicom_scoring import region_masks, uncrop_to_full
from neurovision.metrics import REGION_NAMES, dice_score

__all__ = [
    "CarryDivergedError",
    "rescale_brain_intensities",
    "CarryResult",
    "brats_reference_on_grid",
    "carry_label",
    "brain_overlap_dice",
    "carry_qc_passes",
    "score_against_carried",
]

logger = logging.getLogger(__name__)

_THREAD_ENV = "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"
# A real pose difference is a few mm; anything beyond this is a diverged optimiser.
_MAX_TRANSLATION_MM = 50.0


class CarryDivergedError(RuntimeError):
    """The registration diverged (absurd translation or empty carried label)."""


def rescale_brain_intensities(
    img: np.ndarray, *, lo_pct: float = 0.5, hi_pct: float = 99.5
) -> np.ndarray:
    """Maps the nonzero (brain) voxels of `img` to [0, 1]; zeros stay exactly 0.

    Mutual-information registration diverged on the first real case because the
    BraTS T1CE is z-scored (about -3.4 to 14) while ours is in [0, 1]. Both images
    are therefore rescaled before registering. Percentile clipping stops a few
    bright voxels from squashing the rest. The defaults were fixed 2026-11-25 after
    the first real case diverged; see the protocol implementation note.

    Args:
        img: `(D, H, W)` image, zero outside the brain.
        lo_pct: Lower percentile of the nonzero voxels, mapped to 0.
        hi_pct: Upper percentile of the nonzero voxels, mapped to 1.

    Returns:
        float32 `(D, H, W)` array; all zeros if `img` has no nonzero voxel.
    """
    img = np.asarray(img, dtype=np.float32)
    out = np.zeros_like(img)
    brain = img != 0
    if not brain.any():
        return out
    lo, hi = np.percentile(img[brain], [lo_pct, hi_pct])
    if hi <= lo:  # flat brain: nothing to stretch
        out[brain] = 1.0
        return out
    out[brain] = np.clip((img[brain] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _check_not_diverged(
    translation: tuple[float, ...], carried_label: np.ndarray, moving_label: np.ndarray
) -> None:
    """Raises `CarryDivergedError` for an absurd translation or a vanished label."""
    norm = float(np.linalg.norm(translation))
    if norm > _MAX_TRANSLATION_MM:
        raise CarryDivergedError(f"translation {norm:.1f} mm exceeds {_MAX_TRANSLATION_MM} mm")
    if not carried_label.any() and moving_label.any():
        raise CarryDivergedError("carried label is empty although the moving label is not")


@dataclass(frozen=True)
class CarryResult:
    """Outcome of carrying a label from the moving grid onto the fixed grid.

    Attributes:
        carried_label: `(D, H, W)` integer label on the fixed grid.
        carried_brain_mask: `(D, H, W)` bool, moving brain mask on the fixed grid.
        translation_mm: Displacement (array-axis order, mm) the transform applies at
            the fixed volume's centre (fixed point -> moving point).
        rotation_deg: Total rotation angle of the rigid matrix, in degrees.
        transform_params: ANTs' flat parameters (9 matrix entries, then 3 translation).
    """

    carried_label: np.ndarray
    carried_brain_mask: np.ndarray
    translation_mm: tuple[float, float, float]
    rotation_deg: float
    transform_params: list[float]


def brats_reference_on_grid(
    case_dir: Path, *, modality_index: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Loads a BraTS-preprocessed case's T1CE and label on the full reference grid.

    The image and the label go through the same `uncrop_to_full`, so they stay
    voxel-aligned. The full grid is already in the orientation of `meta["affine"]`,
    which is the reference grid the clinical pipeline's atlas-space masks are mapped
    onto (via `to_reference_grid(mask, nifti.affine, meta["affine"])`, as in
    `scripts/real_dicom_offsets.py`), so no further reorientation is needed here.

    Args:
        case_dir: Directory holding `image.npy` `(4, D, H, W)`, `label.npy`
            `(D, H, W)` and `meta.json`.
        modality_index: Channel of `image.npy` to return (1 = T1CE, per
            `configs/data/brats.yaml`).

    Returns:
        `(image, label)`: float32 `(D, H, W)` and int `(D, H, W)`, both shaped
        `meta["original_shape"]`.

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If image and label cropped shapes disagree, or the channel index
            is out of range.
    """
    case_dir = Path(case_dir)
    paths = {name: case_dir / name for name in ("image.npy", "label.npy", "meta.json")}
    missing = [name for name, p in paths.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{case_dir}: missing {missing}")

    meta = json.loads(paths["meta.json"].read_text())
    image = np.load(paths["image.npy"], mmap_mode="r")  # (C, D, H, W), cropped
    label = np.load(paths["label.npy"])  # (D, H, W), cropped
    if image.ndim != 4 or not 0 <= modality_index < image.shape[0]:
        raise ValueError(
            f"{case_dir}: image.npy shape {image.shape} has no channel {modality_index}"
        )
    if tuple(image.shape[1:]) != tuple(label.shape):
        raise ValueError(
            f"{case_dir}: image spatial shape {image.shape[1:]} != label shape {label.shape}"
        )

    channel = np.asarray(image[modality_index], dtype=np.float32)
    image_full = uncrop_to_full(channel, meta)
    label_full = uncrop_to_full(label, meta)
    return image_full, label_full


def _import_ants():  # noqa: ANN202 - returns a module
    try:
        import ants  # noqa: PLC0415 - lazy on purpose, see module docstring
    except ImportError as exc:
        raise ImportError(
            "carry_label needs `ants` (antspyx), which is installed only in "
            "`.venv-clinical`. Run this from `.venv-clinical`."
        ) from exc
    return ants


def carry_label(
    moving_image: np.ndarray,
    moving_label: np.ndarray,
    fixed_image: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int = 42,
) -> CarryResult:
    """Rigidly registers moving -> fixed (mutual information) and carries the label.

    Args:
        moving_image: `(D, H, W)` image to move (BraTS T1CE); zero outside the brain.
        moving_label: `(D, H, W)` integer label on the moving grid.
        fixed_image: `(D, H, W)` image defining the target grid (our atlas-space T1CE).
        spacing: Voxel size in mm, applied to both images.
        seed: Random seed for ANTs' metric sampling.

    Returns:
        A `CarryResult` on the fixed grid.

    Raises:
        ImportError: If `ants` is unavailable (it lives only in `.venv-clinical`).
        ValueError: If moving image and label shapes differ.
        CarryDivergedError: If the registration diverged.
    """
    ants = _import_ants()
    if moving_image.shape != moving_label.shape:
        raise ValueError(
            f"moving image {moving_image.shape} and label {moving_label.shape} differ in shape"
        )

    def to_ants(arr: np.ndarray):  # noqa: ANN202
        return ants.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), spacing=list(spacing))

    # Registration sees rescaled copies; the label and brain mask come from the
    # ORIGINAL moving image, and the output grid is the fixed image's.
    fixed = to_ants(fixed_image)
    fixed_reg = to_ants(rescale_brain_intensities(fixed_image))
    moving_reg = to_ants(rescale_brain_intensities(moving_image))
    label_img = to_ants(moving_label)
    brain_img = to_ants(moving_image != 0)

    # One ITK thread makes the optimiser's floating-point order reproducible.
    previous = os.environ.get(_THREAD_ENV)
    os.environ[_THREAD_ENV] = "1"
    try:
        reg = ants.registration(
            fixed=fixed_reg,
            moving=moving_reg,
            type_of_transform="Rigid",
            aff_metric="mattes",
            random_seed=seed,
        )
        transforms = reg["fwdtransforms"]
        # Nearest neighbour: labels must stay integers copied from the input.
        carried_label = ants.apply_transforms(
            fixed=fixed,
            moving=label_img,
            transformlist=transforms,
            interpolator="nearestNeighbor",
        ).numpy()
        carried_brain = ants.apply_transforms(
            fixed=fixed,
            moving=brain_img,
            transformlist=transforms,
            interpolator="nearestNeighbor",
        ).numpy()
        tx = ants.read_transform(transforms[0])
        params = [float(p) for p in tx.parameters]
        centre = [(n - 1) / 2.0 * s for n, s in zip(fixed_image.shape, spacing)]
        mapped = tx.apply_to_point(tuple(centre))
    finally:
        if previous is None:
            os.environ.pop(_THREAD_ENV, None)
        else:
            os.environ[_THREAD_ENV] = previous

    rotation = np.asarray(params[:9]).reshape(3, 3)
    # Angle of a rotation matrix: cos(theta) = (trace - 1) / 2.
    cos_theta = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    translation = tuple(float(m - c) for m, c in zip(mapped, centre))
    logger.info(
        "carry_label: translation_mm=%s rotation_deg=%.3f",
        translation,
        np.degrees(np.arccos(cos_theta)),
    )
    carried_label = np.rint(carried_label).astype(moving_label.dtype)
    _check_not_diverged(translation, carried_label, moving_label)
    return CarryResult(
        carried_label=carried_label,
        carried_brain_mask=carried_brain > 0.5,
        translation_mm=translation,  # type: ignore[arg-type]
        rotation_deg=float(np.degrees(np.arccos(cos_theta))),
        transform_params=params,
    )


def brain_overlap_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean brain masks.

    This is a registration-FAILURE screen only (did the carry land somewhere sane?).
    It is never an alignment or lateralisation check: brain-mask Dice scores higher
    on a left-right mirrored atlas (repo trap 3), so it is blind to a flip.

    Args:
        a: `(D, H, W)` boolean mask.
        b: `(D, H, W)` boolean mask, same shape.

    Returns:
        Dice in [0, 1]; 1.0 if both masks are empty.

    Raises:
        ValueError: If shapes differ.
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"brain_overlap_dice: shapes differ, {a.shape} vs {b.shape}")
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def carry_qc_passes(dice: float, threshold: float) -> bool:
    """True if the carry-QC Dice meets the threshold (`dice >= threshold`).

    Args:
        dice: Output of `brain_overlap_dice`.
        threshold: Minimum acceptable Dice, taken from config (no default here).
    """
    return bool(dice >= threshold)


def score_against_carried(pred_labels: np.ndarray, carried_label: np.ndarray) -> dict[str, float]:
    """Region Dice (ET/TC/WT) of our mask against the carried label, on our grid.

    Uses the same `region_masks` / `dice_score` path as the atlas-space column, with
    `ignore_empty=False` semantics (both empty scores 1.0), so numbers are comparable.

    Args:
        pred_labels: `(D, H, W)` integer prediction in `{0, 1, 2, 3}`.
        carried_label: `(D, H, W)` integer carried label, same grid.

    Returns:
        Dict with `dice_ET`, `dice_TC`, `dice_WT`.
    """
    if pred_labels.shape != carried_label.shape:
        raise ValueError(f"shape mismatch: {pred_labels.shape} vs {carried_label.shape}")
    # Only Dice is needed, so region_masks + dice_score are used directly; score_case
    # would also compute HD95 on a full 240x240x155 grid, which is slow and unused.
    pred_regions = region_masks(pred_labels)
    true_regions = region_masks(carried_label)
    result: dict[str, float] = {}
    for name in REGION_NAMES:
        p = torch.from_numpy(np.ascontiguousarray(pred_regions[name], dtype=np.float32))[None, None]
        t = torch.from_numpy(np.ascontiguousarray(true_regions[name], dtype=np.float32))[None, None]
        result[f"dice_{name}"] = float(dice_score(p, t, ignore_empty=False)[0, 0])
    return result
