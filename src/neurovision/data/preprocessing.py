"""Offline BraTS preprocessing: turn raw NIfTI volumes into `.npy` arrays.

This module runs once, locally, on CPU — never on Kaggle, never on a GPU. It
reads the four MRI modalities named by a `BratsCase`, normalizes and crops
them, remaps the segmentation labels to a contiguous range, and writes the
result to disk so that training later only ever reads small `.npy` files
instead of re-parsing NIfTI headers and re-normalizing on every epoch.

Deliberately dependency-light: only numpy, nibabel, and the project's own
`utils/io.py` helpers. No torch, no monai — this file must be importable and
runnable without either.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neurovision.data.brats import BratsCase
from neurovision.utils.io import ensure_dir, read_json, write_json

logger = logging.getLogger(__name__)

# BraTS 2020/2021 ships raw labels {0, 1, 2, 4} -- label 3 was retired from
# the BraTS labeling convention years ago but the gap was never closed, so a
# raw label array has a hole in it. Anything that indexes by class count (a
# one-hot encoding, a confusion matrix, a loss with num_classes=4) needs
# contiguous {0, 1, 2, 3} instead, hence this remap table.
BRATS_LABEL_MAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 4: 3}

# BraTS 2023+ (including BraTS-Africa) closed the gap and ships raw labels
# {0, 1, 2, 3} directly -- ET is already 3, so this remap is the identity.
BRATS2023_LABEL_MAP: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3}

# One label map per supported source-dataset convention. Keyed by the string
# a config or caller passes to `remap_labels`.
_LABEL_CONVENTIONS: dict[str, dict[int, int]] = {
    "brats2021": BRATS_LABEL_MAP,
    "brats2023": BRATS2023_LABEL_MAP,
}

# Fixed channel order, matching `BratsCase.modality_paths`.
_MODALITY_ROLES = ("t1", "t1ce", "t2", "flair")

# Valid single-character nibabel axis codes: one of each pair must appear,
# one per spatial axis, in `reorient_to_axcodes`'s `target_axcodes`.
_VALID_AXIS_CODES = {"L", "R", "P", "A", "I", "S"}


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score a single 3D volume using only its nonzero (brain) voxels.

    MRI background is air -- typically 60-70% of the volume -- and is
    exactly zero. Computing mean/std over the *whole* array would let that
    huge block of zeros drag the mean down and inflate the std, compressing
    the actual brain tissue into a narrow band. So statistics are computed
    only over `volume != 0`, the result is written back only at those
    voxels, and everything else is left at exactly 0.0. This matches
    MONAI's `NormalizeIntensity(nonzero=True)`.

    Args:
        volume: Array of shape `(D, H, W)`.

    Returns:
        Float32 array of shape `(D, H, W)`: z-scored at nonzero voxels,
        exactly 0.0 at background voxels. An all-zero or constant-valued
        (zero std) input returns all zeros instead of raising or producing
        NaNs from a divide-by-zero.
    """
    volume = np.asarray(volume)
    out = np.zeros_like(volume, dtype=np.float32)
    mask = volume != 0
    if not np.any(mask):
        logger.warning("normalize_nonzero: volume is entirely zero; returning zeros.")
        return out

    brain = volume[mask]
    mean = brain.mean()
    std = brain.std()
    if std == 0:
        # Constant-valued nonzero region -- dividing by zero would produce
        # inf/nan. There is no meaningful z-score for a constant, so we
        # return zeros and log rather than let NaNs propagate silently into
        # training.
        logger.warning(
            "normalize_nonzero: nonzero voxels have zero std (constant volume); returning zeros."
        )
        return out

    out[mask] = ((brain - mean) / std).astype(np.float32)
    return out


def compute_nonzero_bbox(image: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Compute the tight bounding box of nonzero voxels across all channels.

    A voxel counts as foreground if ANY modality is nonzero there (union
    across channels), since the four MRI sequences don't all light up the
    same tissue -- cropping to a single channel's support could clip real
    signal in another.

    Args:
        image: Stacked modalities, shape `(C, D, H, W)`.

    Returns:
        One `(start, end)` pair per spatial axis, in `(D, H, W)` order, with
        `end` exclusive so the result slices directly. If `image` is
        entirely zero, returns the full extent of each axis (rather than an
        empty box, which would crop to a zero-sized array and crash much
        later in a confusing place) and logs a warning.
    """
    mask = np.any(image != 0, axis=0)  # (D, H, W), union across channels
    if not np.any(mask):
        logger.warning("compute_nonzero_bbox: image is entirely zero; returning full extent.")
        return tuple((0, size) for size in mask.shape)

    bbox = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        projected = np.any(mask, axis=other_axes)  # 1D bool along this axis
        nonzero_idx = np.flatnonzero(projected)
        start, end = int(nonzero_idx[0]), int(nonzero_idx[-1]) + 1
        bbox.append((start, end))
    return tuple(bbox)


def crop_to_bbox(array: np.ndarray, bbox: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Crop the last three axes of an array to a bounding box.

    Works for both a 4D image `(C, D, H, W)` and a 3D label `(D, H, W)`
    without needing to know which one it got, because the box is always
    applied to the trailing three (spatial) axes via an `Ellipsis` slice.

    Args:
        array: Array whose last three axes are `(D, H, W)`.
        bbox: `((d0, d1), (h0, h1), (w0, w1))`, `end` exclusive, as returned
            by `compute_nonzero_bbox`.

    Returns:
        The cropped array, same number of dimensions as `array`.
    """
    slices = tuple(slice(start, end) for start, end in bbox)
    return array[(..., *slices)]


def remap_labels(seg: np.ndarray, convention: str = "brats2021") -> np.ndarray:
    """Remap raw source labels to contiguous {0,1,2,3}.

    0 = background, 1 = NCR/NET (necrotic/non-enhancing tumor core),
    2 = ED (peritumoral edema), 3 = ET (enhancing tumor). Which raw value
    means ET depends on which dataset the labels came from:

    - `"brats2021"` (BraTS 2020/2021 convention, default): raw values
      `{0, 1, 2, 4}`, ET was raw label 4. Label 3 was retired from BraTS
      years ago and never reused, so raw label 3 is invalid here.
    - `"brats2023"` (BraTS 2023+, including BraTS-Africa): raw values
      `{0, 1, 2, 3}` directly, ET is already 3 -- an identity map. Raw
      label 4 is invalid here.

    Note: the three BraTS *evaluation regions* (ET = {3}, TC = {1, 3},
    WT = {1, 2, 3}) are derived from these remapped labels, but that
    derivation belongs in the transform pipeline that builds training
    targets, not here -- this function's only job is the label remap.

    Args:
        seg: Raw label array of any shape.
        convention: Which source-dataset label convention `seg` follows.
            One of `"brats2021"` (default) or `"brats2023"`.

    Returns:
        `uint8` array of the same shape with values in `{0, 1, 2, 3}`.

    Raises:
        ValueError: If `convention` is not a recognized convention name, or
            if `seg` contains any value outside that convention's expected
            raw values. Silently passing an unexpected label through would
            corrupt training targets, so this is a hard failure.
    """
    label_map = _LABEL_CONVENTIONS.get(convention)
    if label_map is None:
        raise ValueError(
            f"remap_labels: unknown convention {convention!r}; "
            f"expected one of {sorted(_LABEL_CONVENTIONS)}."
        )

    unique_vals = np.unique(seg)
    valid = set(label_map.keys())
    bad_vals = sorted(int(v) for v in unique_vals if int(v) not in valid)
    if bad_vals:
        raise ValueError(
            f"remap_labels: unexpected label value(s) {bad_vals}; "
            f"expected values in {sorted(valid)}."
        )

    out = np.zeros_like(seg, dtype=np.uint8)
    for src_label, dst_label in label_map.items():
        out[seg == src_label] = dst_label
    return out


def _validate_axcodes(axcodes: Any) -> tuple[str, str, str]:
    """Normalize and validate a 3-axis-code sequence.

    Accepts a tuple or list (Hydra/OmegaConf configs deserialize a YAML
    list as a `ListConfig`, not a `tuple`), and always returns a plain
    `tuple[str, str, str]` so downstream nibabel calls get exactly what
    they expect.

    Raises:
        ValueError: If `axcodes` is not a 3-element sequence of valid
            single-character axis codes (one of L/R, P/A, I/S per axis).
    """
    is_sequence = not isinstance(axcodes, str) and hasattr(axcodes, "__len__")
    if not is_sequence or len(axcodes) != 3:
        raise ValueError(
            f"axcodes must be a 3-tuple of axis codes from {sorted(_VALID_AXIS_CODES)}, "
            f"got {axcodes!r}."
        )
    codes = tuple(str(c) for c in axcodes)
    bad = [c for c in codes if c not in _VALID_AXIS_CODES]
    if bad:
        raise ValueError(
            f"axcodes must be a 3-tuple of axis codes from {sorted(_VALID_AXIS_CODES)}, "
            f"got {axcodes!r} (invalid: {bad})."
        )
    return codes  # type: ignore[return-value]


def reorient_to_axcodes(
    array: np.ndarray,
    affine: np.ndarray,
    target_axcodes: tuple[str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a 3D volume so its voxel axes follow `target_axcodes`.

    Different BraTS releases ship different voxel-axis conventions -- BraTS
    2021's affine is `diag(-1, -1, 1)` (axis codes `LPS`), while BraTS-Africa
    (BraTS 2023 convention) ships `RAS`. This function reads the volume's
    *current* axis codes from `affine` and reorients it (flips and/or
    transposes, never resamples or interpolates) to match `target_axcodes`,
    using nibabel's own orientation machinery rather than hand-rolled flips
    -- the same logic nibabel itself uses to interpret NIfTI headers.

    Args:
        array: Volume of shape `(D, H, W)`, any dtype (including bool or
            integer label maps -- this only permutes/flips voxels, it never
            interpolates, so label values are preserved exactly).
        affine: The volume's current 4x4 affine (voxel index -> world mm).
        target_axcodes: Desired voxel axis codes, e.g. `("L", "P", "S")`.

    Returns:
        `(reoriented_array, updated_affine)`: `array` reordered/flipped to
        match `target_axcodes`, and `affine` updated so it still correctly
        describes the returned array (i.e.
        `nibabel.aff2axcodes(updated_affine) == target_axcodes`). An exact
        no-op -- identical array values, identical affine -- when `array`
        already matches `target_axcodes`.

    Raises:
        ValueError: If `target_axcodes` is not a 3-tuple of valid axis
            codes.
    """
    target = _validate_axcodes(target_axcodes)

    current_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(target)
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)

    reoriented = nib.orientations.apply_orientation(array, transform)
    updated_affine = np.asarray(affine) @ nib.orientations.inv_ornt_aff(transform, array.shape)
    return reoriented, updated_affine


def load_case_arrays(case: BratsCase) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load one case's four modalities (and optional label) from disk.

    Args:
        case: Resolved paths for one BraTS case.

    Returns:
        A tuple `(image, label, meta)`:
        - `image`: `(4, D, H, W)` float32, channels in `case.modality_paths`
          order (t1, t1ce, t2, flair).
        - `label`: `(D, H, W)` raw label array (NOT yet remapped), or
          `None` if `case.seg` is `None`.
        - `meta`: dict with `original_shape` (tuple of 3 ints), `affine`
          (4x4 nested list of floats, JSON-serializable) and `spacing`
          (tuple of 3 floats from the header).

    Raises:
        ValueError: If the four modalities do not all share the same shape,
            or if a label is present but its shape does not match.
    """
    volumes: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    affine: np.ndarray | None = None
    spacing: tuple[float, float, float] | None = None

    for role, path in zip(_MODALITY_ROLES, case.modality_paths):
        img = nib.load(str(path))
        # `img.get_fdata()` always upcasts to float64 and caches the result
        # on the proxy object -- fine for one file, wasteful across 1251
        # cases. `np.asarray(img.dataobj)` reads at the on-disk dtype with
        # no caching, and we cast to float32 ourselves right after.
        arr = np.asarray(img.dataobj).astype(np.float32)
        if shape is None:
            shape = arr.shape
            affine = img.affine
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        elif arr.shape != shape:
            raise ValueError(
                f"load_case_arrays: modality shape mismatch for case "
                f"{case.case_id!r}: {role!r} has shape {arr.shape}, expected {shape} "
                f"(from earlier modalities)."
            )
        volumes.append(arr)

    image = np.stack(volumes, axis=0)  # (4, D, H, W)

    label: np.ndarray | None = None
    if case.seg is not None:
        seg_img = nib.load(str(case.seg))
        label = np.asarray(seg_img.dataobj)
        if label.shape != shape:
            raise ValueError(
                f"load_case_arrays: label shape mismatch for case "
                f"{case.case_id!r}: seg has shape {label.shape}, expected {shape}."
            )

    assert shape is not None and affine is not None and spacing is not None
    meta: dict[str, Any] = {
        "original_shape": tuple(int(s) for s in shape),
        "affine": affine.tolist(),  # .tolist() so JSON can serialize it
        "spacing": spacing,
    }
    return image, label, meta


def is_case_processed(case_id: str, out_dir: str | Path) -> bool:
    """Check whether a case's outputs already exist on disk.

    A case counts as processed if `image.npy` and `meta.json` exist, and
    `label.npy` also exists whenever `meta.json` says the case had a label.
    Unlabeled validation cases legitimately have no `label.npy`, so its
    absence alone must not count as "not processed".

    Args:
        case_id: The case's identifier (matches the output subdirectory).
        out_dir: Root output directory (parent of the per-case directories).

    Returns:
        True if all expected outputs for this case are present.
    """
    case_dir = Path(out_dir) / case_id
    meta_path = case_dir / "meta.json"
    image_path = case_dir / "image.npy"
    if not meta_path.is_file() or not image_path.is_file():
        return False

    meta = read_json(meta_path)
    if meta.get("has_label"):
        return (case_dir / "label.npy").is_file()
    return True


def _summary_from_meta(meta: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    """Build the flat summary dict (one CSV row) from a case's `meta.json`.

    Shared by the fresh-processing path and the skip path so both return
    the exact same shape of summary regardless of whether work was done.
    """
    label_counts: dict[str, int] = meta.get("label_voxel_counts") or {}
    image_path = case_dir / "image.npy"
    label_path = case_dir / "label.npy"
    return {
        "case_id": meta["case_id"],
        "original_shape": tuple(meta["original_shape"]),
        "cropped_shape": tuple(meta["cropped_shape"]),
        "spacing": tuple(meta["spacing"]),
        "n_class_0": label_counts.get("0", 0),
        "n_class_1": label_counts.get("1", 0),
        "n_class_2": label_counts.get("2", 0),
        "n_class_3": label_counts.get("3", 0),
        "image_bytes": image_path.stat().st_size if image_path.is_file() else 0,
        "label_bytes": label_path.stat().st_size if label_path.is_file() else 0,
    }


def preprocess_case(
    case: BratsCase,
    out_dir: str | Path,
    overwrite: bool = False,
    label_convention: str = "brats2021",
    target_axcodes: tuple[str, str, str] = ("L", "P", "S"),
) -> dict[str, Any]:
    """Normalize, crop, and remap one case, writing results to `out_dir`.

    Pipeline: load -> reorient to `target_axcodes` -> per-channel z-score
    (nonzero only) -> crop to the union nonzero bounding box -> remap
    labels -> write `image.npy`, `label.npy`, `meta.json`.

    Reorientation happens first, before normalization and before the crop
    bbox is computed, so every downstream quantity (the bbox, the cropped
    shape, the saved arrays) describes the reoriented volume. For BraTS
    2021 -- whose own affine is already `diag(-1, -1, 1)`, i.e. `LPS` --
    reorienting to the default `target_axcodes=("L", "P", "S")` is an exact
    no-op (see `reorient_to_axcodes`), so this is purely additive: it does
    not change the output for any case already preprocessed under the old
    pipeline.

    Args:
        case: Resolved paths for one BraTS case.
        out_dir: Root output directory. Outputs go to `out_dir/<case_id>/`.
        overwrite: If False (default) and the case's outputs already exist,
            skip reprocessing and return the existing summary instead.
        label_convention: Which source-dataset label convention the case's
            segmentation follows. Passed straight to `remap_labels` --
            `"brats2021"` (default, raw ET = 4) or `"brats2023"` (raw
            ET = 3, e.g. BraTS-Africa).
        target_axcodes: Voxel axis codes every case is normalized to before
            cropping. Default `("L", "P", "S")` matches BraTS 2021's own
            convention.

    Returns:
        A summary dict with `case_id`, `original_shape`, `cropped_shape`,
        `spacing`, `n_class_0`..`n_class_3` (voxel counts), `image_bytes`,
        `label_bytes`, and `skipped` (True iff this call skipped work
        because the case was already processed).
    """
    out_dir = Path(out_dir)
    case_dir = out_dir / case.case_id

    if not overwrite and is_case_processed(case.case_id, out_dir):
        logger.debug("Case %s already processed; skipping.", case.case_id)
        meta = read_json(case_dir / "meta.json")
        summary = _summary_from_meta(meta, case_dir)
        summary["skipped"] = True
        return summary

    image, label, load_meta = load_case_arrays(case)

    # Reorient every modality (and the label, if present) to the target
    # axis convention BEFORE anything else -- the crop bbox and every
    # downstream quantity must describe the reoriented volume, not the raw
    # one. All four modalities and the label share one affine (BraTS ships
    # its modalities already co-registered), so the same transform applies
    # to each; reorient_to_axcodes recomputes it once per call, which is
    # cheap (nibabel orientation bookkeeping, no resampling) relative to
    # the rest of this pipeline.
    source_affine = np.asarray(load_meta["affine"], dtype=np.float64)
    source_axcodes = nib.aff2axcodes(source_affine)
    target = _validate_axcodes(target_axcodes)

    reoriented_channels = []
    updated_affine = source_affine
    for c in range(image.shape[0]):
        reoriented_channel, updated_affine = reorient_to_axcodes(image[c], source_affine, target)
        reoriented_channels.append(reoriented_channel)
    image = np.stack(reoriented_channels, axis=0)

    if label is not None:
        label, _ = reorient_to_axcodes(label, source_affine, target)

    reoriented_shape = tuple(int(s) for s in image.shape[1:])

    # Normalize each of the 4 channels independently, not globally: the four
    # MRI sequences (T1, T1CE, T2, FLAIR) have completely different
    # intensity scales, and a single shared mean/std would wash out
    # whichever sequence has the smaller dynamic range.
    normalized = np.stack([normalize_nonzero(image[c]) for c in range(image.shape[0])], axis=0)

    # Compute the bbox from the RAW image, not the normalized one. Usually
    # these agree, because normalize_nonzero only rescales nonzero voxels and
    # leaves background at exactly 0.0. But a channel whose foreground is
    # constant has std == 0, so normalize_nonzero returns all zeros for it --
    # and that channel's support would then vanish from the union, silently
    # cropping tighter than the actual brain. The raw image's nonzero support
    # is the ground truth for "where is there signal", so use it.
    bbox = compute_nonzero_bbox(image)
    cropped_image = crop_to_bbox(normalized, bbox)
    cropped_label = crop_to_bbox(label, bbox) if label is not None else None

    has_label = cropped_label is not None
    remapped_label = remap_labels(cropped_label, convention=label_convention) if has_label else None

    ensure_dir(case_dir)

    image_path = case_dir / "image.npy"
    # Z-scored values sit roughly in [-5, 5], comfortably inside float16's
    # precision range, and float16 halves the on-disk size of the dataset --
    # worth it across ~1251 cases. Labels stay integer: uint8 is exact
    # (values 0-3) and 8x smaller than a float label array would be.
    np.save(image_path, cropped_image.astype(np.float16))

    label_path = case_dir / "label.npy"
    if has_label:
        np.save(label_path, remapped_label.astype(np.uint8))

    label_voxel_counts: dict[str, int] | None = None
    if has_label:
        values, counts = np.unique(remapped_label, return_counts=True)
        label_voxel_counts = {str(int(v)): int(c) for v, c in zip(values, counts)}

    cropped_shape = tuple(int(s) for s in cropped_image.shape[1:])

    # `bbox` and `original_shape` are the load-bearing fields here:
    # predictions made on the cropped volume must be un-cropped back into
    # the original volume geometry before they mean anything as a BraTS
    # submission. Losing either makes the preprocessed data unusable for
    # anything that has to be submitted or compared against the raw NIfTI.
    # Both now describe the REORIENTED volume (post `target_axcodes`), which
    # is what `bbox` was actually computed against -- for BraTS 2021 this is
    # numerically identical to the raw volume's own shape, since reorienting
    # LPS to LPS is a no-op.
    meta: dict[str, Any] = {
        "case_id": case.case_id,
        "original_shape": reoriented_shape,
        "cropped_shape": cropped_shape,
        "bbox": [[start, end] for start, end in bbox],
        "affine": updated_affine.tolist(),
        "spacing": load_meta["spacing"],
        "has_label": has_label,
        "label_voxel_counts": label_voxel_counts,
        "source_axcodes": "".join(source_axcodes),
        "target_axcodes": "".join(target),
    }
    write_json(meta, case_dir / "meta.json")

    summary = _summary_from_meta(meta, case_dir)
    summary["skipped"] = False
    return summary
