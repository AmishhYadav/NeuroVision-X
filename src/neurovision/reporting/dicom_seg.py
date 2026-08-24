"""Exports a segmentation mask as a DICOM Segmentation (SEG) object.

Milestone 4, Phase E, task E6: so a case's mask loads in OHIF, or any other
PACS viewer, instead of only in this project's own viewer. Reuses
`neurovision.inference.postprocess.regions_to_classes` for the region ->
class-map collapse rather than reimplementing it -- see that function's
docstring for why the assignment order (outer to inner) cannot be reversed.

**The geometry trap, and why this module never resamples.** After E2
(`neurovision.data...` clinical preprocessing, not built yet at the time this
module was written) a mask lives in ATLAS space (SRI24, 240x240x155, 1 mm
isotropic -- see `configs/anatomy/sri24.yaml`), not in the pixel geometry of
the DICOM series it was derived from. A SEG object that references the
source series' SOP instances while carrying atlas-space voxels renders a
plausible-looking overlay in the WRONG place, and nothing raises -- this is
exactly the shape of the demo geometry re-crop bug this project already
caught, by recomputing an already-published Dice, not by looking at a
picture that "looked aligned" (see CLAUDE.md's ten traps, #9).

So `write_dicom_seg` VALIDATES the mask's shape and voxel spacing against the
source series (`check_geometry_against_source`) and REFUSES with every
mismatch reason named, before writing anything, when they disagree. It does
**not** resample the mask into the source series' geometry. Resampling a
mask back through E2's saved inverse transform is a separate, not-yet-built
step and is explicitly out of scope here.

Because the mask always arrives in atlas space, its spacing is a fixed,
documented physical constant of SRI24 (`_MASK_SPACING_MM`, 1 mm isotropic) --
not a caller-supplied parameter and not something read from config, since
there is nothing to configure: this project has committed to one atlas.

`highdicom` and `pydicom` live only in `.venv-clinical`
(`requirements-clinical.txt`), never in the project's main `.venv` -- see
`neurovision.data.dicom_ingest`'s module docstring for the same split. So,
following that module's convention exactly:

- `SegmentDefinition`, `SEGMENT_DEFINITIONS`, `GeometryCheck`,
  `check_geometry_against_source`, `classes_from_regions` and
  `segment_masks` use no DICOM library at all, and are fully tested in the
  main suite with synthetic numpy arrays.
- `read_source_geometry` and `write_dicom_seg` import `highdicom`/`pydicom`
  INSIDE their function bodies, never at module scope, so importing this
  module never requires `.venv-clinical`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurovision.utils.io import ensure_dir

__all__ = [
    "SegmentDefinition",
    "SEGMENT_DEFINITIONS",
    "GeometryCheck",
    "check_geometry_against_source",
    "classes_from_regions",
    "segment_masks",
    "read_source_geometry",
    "write_dicom_seg",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentDefinition:
    """One DICOM segment: its class-map value, its DICOM number, and its coding.

    Attributes:
        class_value: The value this segment has in the class map produced by
            `classes_from_regions` (equivalently,
            `neurovision.inference.postprocess.regions_to_classes`).
        number: The DICOM segment number, 1-based. Fixed and stable: the
            same class always gets the same `number`, regardless of which
            classes are present in a given case. `write_dicom_seg` may still
            have to write a DIFFERENT number into the actual DICOM object
            when a lower-valued class is absent -- highdicom (and the DICOM
            standard) require segment numbers in a written object to be
            contiguous starting at 1. When that happens it is logged; a
            segment's identity for cross-study comparison is its `label`
            and coding, not the raw integer.
        label: Human-readable segment name (DICOM SegmentLabel).
        category_code: `(value, designator, meaning)` for
            SegmentedPropertyCategoryCodeSequence.
        type_code: `(value, designator, meaning)` for
            SegmentedPropertyTypeCodeSequence.
    """

    class_value: int
    number: int
    label: str
    category_code: tuple[str, str, str]
    type_code: tuple[str, str, str]


# Fixed module constant, not config: a radiologist's viewer must see
# standard SNOMED-CT (SCT) codes for what a segment IS, and that is not a
# knob any experiment should be able to turn. `number` is always equal to
# `class_value` here -- the simplest possible stable numbering, and exactly
# what `postprocess.regions_to_classes` already assigns (1 = necrotic/
# non-enhancing core, 2 = edema, 3 = enhancing tumor).
SEGMENT_DEFINITIONS: tuple[SegmentDefinition, ...] = (
    SegmentDefinition(
        class_value=1,
        number=1,
        label="Necrotic and non-enhancing tumour core",
        category_code=("49755003", "SCT", "Morphologically Abnormal Structure"),
        type_code=("6574001", "SCT", "Necrosis"),
    ),
    SegmentDefinition(
        class_value=2,
        number=2,
        label="Peritumoral edema",
        category_code=("49755003", "SCT", "Morphologically Abnormal Structure"),
        type_code=("79654002", "SCT", "Edema"),
    ),
    SegmentDefinition(
        class_value=3,
        number=3,
        label="Enhancing tumour",
        category_code=("49755003", "SCT", "Morphologically Abnormal Structure"),
        type_code=("86049000", "SCT", "Neoplasm, Primary"),
    ),
)

# SRI24 is 1 mm isotropic (`configs/anatomy/sri24.yaml`'s `target.spacing`),
# and BraTS -- what the model was trained on -- is registered to it. A mask
# produced by this pipeline is therefore ALWAYS in this spacing; it is a
# physical fact of the atlas this project has committed to, not a per-call
# parameter.
_MASK_SPACING_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class GeometryCheck:
    """The result of comparing a mask's geometry against a source series'.

    Attributes:
        ok: True iff both shape and spacing agree (spacing within
            `spacing_atol`).
        reasons: Every reason `ok` is False, human-readable -- shape and
            spacing are reported independently, so a case wrong on both
            comes back with two reasons, not the first one found.
        detail: The raw shapes/spacings compared and the per-axis spacing
            difference, for logging or an audit trail.
    """

    ok: bool
    reasons: tuple[str, ...]
    detail: dict[str, Any]


def check_geometry_against_source(
    mask_shape: tuple[int, int, int],
    mask_spacing_mm: tuple[float, float, float],
    source_shape: tuple[int, int, int],
    source_spacing_mm: tuple[float, float, float],
    *,
    spacing_atol: float = 1e-3,
) -> GeometryCheck:
    """Checks whether a mask's geometry matches a DICOM source series.

    Shape must match exactly. Spacing must match within `spacing_atol` on
    every axis. This function only compares and reports -- it never raises
    and never resamples; `write_dicom_seg` is the caller that turns a failed
    check into a refusal.

    Args:
        mask_shape: The mask's `(D, H, W)` voxel shape.
        mask_spacing_mm: The mask's `(D, H, W)` voxel spacing, in mm.
        source_shape: The source DICOM series' `(D, H, W)` shape (see
            `read_source_geometry`).
        source_spacing_mm: The source DICOM series' `(D, H, W)` voxel
            spacing, in mm.
        spacing_atol: Absolute tolerance, in mm, for the spacing comparison
            on each axis independently.

    Returns:
        A `GeometryCheck` with every mismatch reason, never just the first.
    """
    mask_shape = tuple(int(v) for v in mask_shape)
    source_shape = tuple(int(v) for v in source_shape)
    mask_spacing_mm = tuple(float(v) for v in mask_spacing_mm)
    source_spacing_mm = tuple(float(v) for v in source_spacing_mm)

    reasons: list[str] = []
    spacing_diff_mm = tuple(abs(m - s) for m, s in zip(mask_spacing_mm, source_spacing_mm))
    detail: dict[str, Any] = {
        "mask_shape": mask_shape,
        "source_shape": source_shape,
        "mask_spacing_mm": mask_spacing_mm,
        "source_spacing_mm": source_spacing_mm,
        "spacing_diff_mm": spacing_diff_mm,
        "spacing_atol": spacing_atol,
    }

    if mask_shape != source_shape:
        reasons.append(
            f"shape mismatch: mask is {mask_shape}, source series is {source_shape}. The mask "
            "was not resampled back into the source series' geometry (out of scope for this "
            "module -- see its docstring)."
        )

    if any(diff > spacing_atol for diff in spacing_diff_mm):
        reasons.append(
            f"spacing mismatch: mask is {mask_spacing_mm} mm, source series is "
            f"{source_spacing_mm} mm (per-axis diff {spacing_diff_mm} mm, tolerance "
            f"{spacing_atol} mm)."
        )

    return GeometryCheck(ok=not reasons, reasons=tuple(reasons), detail=detail)


def classes_from_regions(regions: np.ndarray) -> np.ndarray:
    """Collapses nested region channels into a class map (thin numpy adapter).

    Converts `regions` to a `float32` torch tensor and calls
    `neurovision.inference.postprocess.regions_to_classes`, so the
    outer-to-inner collapse rule lives in exactly one place in the codebase.

    `regions_to_classes` itself thresholds at 0.5 and would silently
    binarize a non-{0, 1} input rather than raising, so this function
    rejects a non-binary input up front instead of relying on that.

    Args:
        regions: `(3, D, H, W)` numpy array, channel order `(ET, TC, WT)`,
            binary (`{0, 1}`) values.

    Returns:
        `uint8` numpy array, shape `(D, H, W)`, values in `{0, 1, 2, 3}`.

    Raises:
        ValueError: If `regions` is not `(3, D, H, W)`, or contains a value
            outside `{0, 1}` (names the shape / the offending values).
    """
    array = np.asarray(regions)
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError(
            "classes_from_regions expects a (3, D, H, W) array, channel order (ET, TC, WT); "
            f"got shape {array.shape}."
        )

    unique_values = np.unique(array)
    non_binary = unique_values[(unique_values != 0) & (unique_values != 1)]
    if non_binary.size > 0:
        raise ValueError(
            "classes_from_regions expects a binary {0, 1} array -- "
            "postprocess.regions_to_classes thresholds at 0.5 and would silently binarize a "
            f"non-binary input instead of raising; found non-binary value(s): "
            f"{non_binary.tolist()}."
        )

    # Imported here, not at module scope: neurovision.inference.postprocess
    # imports monai at ITS module top (for KeepLargestConnectedComponent,
    # which this function never touches), and monai is not in
    # requirements-clinical.txt / .venv-clinical (verified 2026-08-24 --
    # `import monai` raises ModuleNotFoundError there). regions_to_classes
    # itself is pure torch, no monai calls, so this keeps THIS module
    # importable in .venv-clinical; calling classes_from_regions there still
    # needs postprocess.py's import chain to succeed, i.e. still needs monai.
    # See this module's test file / delegation report for the resulting gap.
    from neurovision.inference.postprocess import regions_to_classes

    tensor = torch.as_tensor(array, dtype=torch.float32)
    class_map = regions_to_classes(tensor)
    return class_map.numpy().astype(np.uint8)


def segment_masks(class_map: np.ndarray) -> dict[int, np.ndarray]:
    """Splits a class map into one boolean mask per defined segment class.

    Args:
        class_map: `(D, H, W)` integer array with values in `{0, 1, 2, 3}`
            (as produced by `classes_from_regions`).

    Returns:
        `class_value -> boolean mask` for every class in `SEGMENT_DEFINITIONS`
        (1, 2, 3), whether or not that class actually has any voxels in
        `class_map`. Background (0) has no entry. The masks partition the
        nonzero voxels of `class_map`: their union equals `class_map != 0`
        and no two overlap, since each is an exact-value comparison against
        a single-valued map.
    """
    class_map = np.asarray(class_map)
    return {
        definition.class_value: (class_map == definition.class_value)
        for definition in SEGMENT_DEFINITIONS
    }


# ---------------------------------------------------------------------------
# .venv-clinical only. Every function below imports highdicom/pydicom INSIDE
# its body, so this module is importable in the main .venv without either
# library installed at all.
# ---------------------------------------------------------------------------


def _infer_slice_spacing_mm(source_datasets: Sequence[Any]) -> float:
    """Estimates through-plane spacing from a series' slice positions.

    Prefers the mean distance between consecutive slices' `ImagePositionPatient`
    (sorted by `InstanceNumber`), which reflects actual center-to-center
    spacing including any inter-slice gap. Falls back to the first dataset's
    `SliceThickness` -- less accurate (a slice's own thickness need not equal
    its spacing to the next one) -- when there are fewer than two slices or
    any dataset is missing `ImagePositionPatient`. Falls back to `1.0` mm if
    neither is available.
    """
    positions: list[tuple[float, float, float]] = []
    for dataset in source_datasets:
        pos = getattr(dataset, "ImagePositionPatient", None)
        if pos is None or len(pos) != 3:
            positions = []
            break
        positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

    if len(positions) >= 2:
        order = sorted(
            range(len(source_datasets)),
            key=lambda i: getattr(source_datasets[i], "InstanceNumber", i),
        )
        sorted_positions = [positions[i] for i in order]
        diffs = [
            math.dist(sorted_positions[i], sorted_positions[i + 1])
            for i in range(len(sorted_positions) - 1)
        ]
        return float(sum(diffs) / len(diffs))

    thickness = getattr(source_datasets[0], "SliceThickness", None)
    return float(thickness) if thickness is not None else 1.0


def read_source_geometry(
    source_datasets: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Reads `(D, H, W)` shape and voxel spacing off a DICOM source series.

    One `pydicom.Dataset` (header only -- pixel data need not be loaded) per
    slice, in any order; `D` is simply how many were given.

    Args:
        source_datasets: The source series' per-slice DICOM datasets.

    Returns:
        `(shape, spacing_mm)`, both `(D, H, W)`. `spacing_mm`'s in-plane
        components come from `PixelSpacing` (row spacing, column spacing);
        the through-plane component from `_infer_slice_spacing_mm`.

    Raises:
        ValueError: If `source_datasets` is empty, or the first dataset is
            missing `Rows`/`Columns`/`PixelSpacing`.
    """
    if not source_datasets:
        raise ValueError("read_source_geometry: source_datasets is empty.")

    first = source_datasets[0]
    rows = getattr(first, "Rows", None)
    cols = getattr(first, "Columns", None)
    if rows is None or cols is None:
        raise ValueError("read_source_geometry: the source dataset has no Rows/Columns.")

    pixel_spacing = getattr(first, "PixelSpacing", None)
    if pixel_spacing is None or len(pixel_spacing) != 2:
        raise ValueError("read_source_geometry: the source dataset has no (valid) PixelSpacing.")

    shape = (len(source_datasets), int(rows), int(cols))
    spacing_mm = (
        _infer_slice_spacing_mm(source_datasets),
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
    )
    return shape, spacing_mm


def write_dicom_seg(
    cfg: Any,
    regions: np.ndarray,
    source_datasets: Sequence[Any],
    out_path: Path,
) -> Path:
    """Writes a mask as a DICOM Segmentation object, or refuses.

    Reads `cfg.clinical.dicom_seg` (see `configs/clinical/default.yaml`).
    Order of operations: validate `segmentation_type` is `BINARY`; collapse
    `regions` to a class map (`classes_from_regions`); validate the class
    map's geometry against `source_datasets` (`check_geometry_against_source`)
    and refuse on any mismatch, BEFORE opening any output file; refuse an
    all-empty class map; then write one DICOM segment per non-empty class,
    omitting (not zero-filling) any class with no predicted voxels.

    highdicom -- and the DICOM standard -- require a written object's
    segment numbers to be contiguous starting at 1. When a lower-valued
    class (e.g. necrotic core) is entirely absent while a higher one (e.g.
    enhancing tumour) is present, the segments actually written are
    renumbered contiguously; this is logged, since the written integer can
    then differ from `SegmentDefinition.number`. A segment's identity for
    comparing across studies is its label and SNOMED coding, which never
    change.

    Args:
        cfg: The root config (or anything exposing `cfg.clinical.dicom_seg`
            with the fields in `configs/clinical/default.yaml`).
        regions: `(3, D, H, W)` numpy array, channel order `(ET, TC, WT)`,
            binary values -- see `classes_from_regions`.
        source_datasets: The source DICOM series' per-slice datasets (header
            data is enough; pixel data need not be loaded).
        out_path: Exact destination `.dcm` path. Its parent directory is
            created if missing.

    Returns:
        `out_path`, once the SEG object has been written there.

    Raises:
        ValueError: If `cfg.clinical.dicom_seg.segmentation_type` is not
            `"BINARY"`; if `regions` has the wrong shape or non-binary
            values (see `classes_from_regions`); if `source_datasets` is
            empty; if the mask's geometry does not match the source series
            (nothing is written in this case); or if the class map has no
            tumour voxels at all.
    """
    dicom_seg_cfg = cfg.clinical.dicom_seg

    if dicom_seg_cfg.segmentation_type != "BINARY":
        raise ValueError(
            "write_dicom_seg: cfg.clinical.dicom_seg.segmentation_type must be 'BINARY', got "
            f"{dicom_seg_cfg.segmentation_type!r}. The deployed output is a hard mask after "
            "postprocess_logits (threshold + component filtering); writing it as FRACTIONAL "
            "would imply a calibrated per-voxel probability the thresholded mask no longer "
            "carries."
        )

    if not source_datasets:
        raise ValueError("write_dicom_seg: source_datasets is empty; nothing to reference.")

    class_map = classes_from_regions(np.asarray(regions))

    source_shape, source_spacing_mm = read_source_geometry(source_datasets)
    geometry = check_geometry_against_source(
        mask_shape=class_map.shape,
        mask_spacing_mm=_MASK_SPACING_MM,
        source_shape=source_shape,
        source_spacing_mm=source_spacing_mm,
    )
    if not geometry.ok:
        raise ValueError(
            "write_dicom_seg: refusing to write -- the mask's geometry does not match the "
            "source series (see the module docstring's atlas-space trap). "
            + " ".join(geometry.reasons)
        )

    if not np.any(class_map):
        raise ValueError(
            "write_dicom_seg: the class map has no tumour voxels at all (every class is "
            "empty). A SEG object with zero segments is not a valid DICOM object, and there is "
            "nothing to export."
        )

    masks = segment_masks(class_map)
    present = [d for d in SEGMENT_DEFINITIONS if masks[d.class_value].any()]
    omitted = [d for d in SEGMENT_DEFINITIONS if d not in present]
    if omitted:
        logger.info(
            "write_dicom_seg: omitting empty segment(s) (no predicted voxels): %s",
            [d.label for d in omitted],
        )

    # SeriesDescription's DICOM value representation is LO (Long String),
    # capped at 64 characters. Fail loudly here rather than silently
    # truncating the configured string when passing it to highdicom's
    # constructor -- a truncated disclaimer that still looks plausible is
    # worse than a crash that tells the caller exactly what to fix.
    series_description = str(dicom_seg_cfg.series_description)
    if len(series_description) > 64:
        raise ValueError(
            "write_dicom_seg: cfg.clinical.dicom_seg.series_description is "
            f"{len(series_description)} characters, but DICOM's Long String (LO) value "
            "representation caps SeriesDescription at 64. Shorten series_description and put "
            "the long-form text in cfg.clinical.dicom_seg.disclaimer instead (written to "
            "ImageComments, VR LT, 10240 characters)."
        )

    import highdicom as hd
    from pydicom.sr.coding import Code
    from pydicom.uid import generate_uid

    algorithm = hd.content.AlgorithmIdentificationSequence(
        name=dicom_seg_cfg.manufacturer_model_name,
        # "99NVX": a private, locally-defined DICOM coding scheme designator
        # (the "99" prefix is the standard's own convention for one) -- there
        # is no standard code for "this specific research model", so this
        # names it honestly rather than borrowing an unrelated standard code.
        family=Code("1", "99NVX", "Dual-encoder CNN + Swin transformer segmentation model"),
        version=str(dicom_seg_cfg.software_versions),
    )

    segment_descriptions = []
    pixel_channels = []
    for i, definition in enumerate(present):
        dicom_number = i + 1  # highdicom requires contiguous numbering from 1.
        if dicom_number != definition.number:
            logger.info(
                "write_dicom_seg: segment %r written as DICOM SegmentNumber %d (logical number "
                "%d) because a lower-numbered segment is absent from this case -- highdicom and "
                "the DICOM standard require contiguous numbering starting at 1.",
                definition.label,
                dicom_number,
                definition.number,
            )
        segment_descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=dicom_number,
                segment_label=definition.label,
                segmented_property_category=Code(*definition.category_code),
                segmented_property_type=Code(*definition.type_code),
                algorithm_type="AUTOMATIC",
                algorithm_identification=algorithm,
            )
        )
        pixel_channels.append(masks[definition.class_value].astype(np.uint8))

    pixel_array = np.stack(pixel_channels, axis=-1)  # (D, H, W, num_present_segments)

    seg = hd.seg.Segmentation(
        source_images=list(source_datasets),
        pixel_array=pixel_array,
        segmentation_type=dicom_seg_cfg.segmentation_type,
        segment_descriptions=segment_descriptions,
        series_instance_uid=generate_uid(),
        series_number=int(dicom_seg_cfg.series_number),
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer=str(dicom_seg_cfg.manufacturer),
        manufacturer_model_name=str(dicom_seg_cfg.manufacturer_model_name),
        software_versions=str(dicom_seg_cfg.software_versions),
        device_serial_number=str(dicom_seg_cfg.device_serial_number),
        series_description=series_description,
    )
    # The full disclaimer statement (~340 characters -- far longer than
    # SeriesDescription's 64-character LO limit allows) goes into
    # ImageComments instead, whose VR is LT (Long Text, 10240 characters).
    # Set the ordinary way now that SeriesDescription itself is short enough
    # to pass through highdicom's own constructor validation.
    seg.ImageComments = str(dicom_seg_cfg.disclaimer)

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    seg.save_as(str(out_path), enforce_file_format=True)
    logger.info(
        "write_dicom_seg: wrote %d segment(s) to %s (%d omitted)",
        len(present),
        out_path,
        len(omitted),
    )
    return out_path
