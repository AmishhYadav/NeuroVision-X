"""Samples a mask onto a DICOM series' own pixel grid, and orders its slices.

Milestone 4, Phase E, task E6 follow-up (2026-09-18): `dicom_seg.write_dicom_seg`'s
own module docstring used to say the mask always arrives in ATLAS space, with a
fixed, hardcoded spacing (`_MASK_SPACING_MM = (1, 1, 1)`). That was true before
`app.backend.clinical_jobs._export_dicom_seg` existed. Since then, the live path
has resampled the mask into the CENTER modality's own native NIfTI grid
(`neurovision.data.clinical_resample.resample_mask_to_source`) before calling
`write_dicom_seg` at all -- so by the time a mask reaches that function, its
spacing is whatever the source series' spacing happens to be, not (1, 1, 1). A
real non-isotropic study (job `a37fcaad...`, measured 2026-09-18: source spacing
(1.0, 0.9765625, 0.9765625)) was refused on this stale assumption alone.

**A second, silent problem this module exists to fix.** The mask, once
resampled, lives in a NIfTI array with NIfTI's own axis order and orientation
(RAS, and `dcm2niix`-style axes that need not line up with DICOM's own row/
column/slice order at all, and are commonly FLIPPED relative to DICOM, which is
LPS). Passing that array's axes straight through as `(slice, row, col)` DICOM
frames -- as `_export_dicom_seg` used to -- can pass a shape check purely by
coincidence (e.g. columns == slices on a roughly cubic volume) while still
rendering the mask in the wrong physical place. This is exactly the trap
CLAUDE.md's ten traps names as #9: a check that runs is not a check that
actually reached the failure condition.

This module fixes both by never assuming axis correspondence at all. It reads
each DICOM pixel's own physical (patient-frame) position directly off that
DICOM slice's `ImagePositionPatient`/`ImageOrientationPatient`/`PixelSpacing`,
converts that position into the mask's own voxel space through the mask's own
NIfTI affine, and nearest-neighbour samples the mask there. If the two grids
are not, in fact, the same physical grid (any DICOM pixel centre lands more
than `max_offset_voxels` away from the nearest mask voxel centre), it refuses
loudly with the measured offset rather than silently picking a wrong voxel.

**The coordinate systems, spelled out because they are the entire point.**
DICOM's patient-based coordinate system (`ImagePositionPatient`,
`ImageOrientationPatient`) is LPS (+x = toward patient Left, +y = toward
Posterior, +z = toward Superior). NIfTI, and therefore every affine this
codebase reads via `nibabel`, is RAS (+x = Right, +y = Anterior, +z =
Superior). LPS -> RAS is exactly: negate x and y, leave z unchanged. Getting
this backwards silently mirrors left and right -- see CLAUDE.md trap #3, a
different mirroring bug with the same shape.

The per-pixel world-position formula follows the DICOM standard exactly
(PS3.3 C.7.6.2.1.1): for DICOM pixel `(row r, col c)` on a slice whose header
gives `ImagePositionPatient` (`IPP`), `ImageOrientationPatient` (`IOP`, six
values), and `PixelSpacing` (`(row_spacing, col_spacing)`),

    row_dir = IOP[3:6]   # direction cosines of the first COLUMN --
                         # i.e. the direction the physical position moves in
                         # as the ROW index increases.
    col_dir = IOP[0:3]   # direction cosines of the first ROW -- the
                         # direction position moves in as the COLUMN index
                         # increases.
    world_lps(r, c) = IPP + r * row_spacing * row_dir + c * col_spacing * col_dir

This module never imports `pydicom` or `highdicom`, and reads every DICOM
attribute with `getattr`, so it is fully testable in the main `.venv` with
plain `types.SimpleNamespace` fakes standing in for `pydicom.Dataset` objects
-- the same convention `dicom_seg.py`'s DICOM-library-free functions already
follow.

**Ordering, and why `write_dicom_seg` must receive the SAME sorted list this
module returns.** `sort_datasets_along_normal` orders single-frame datasets by
the signed projection of `ImagePositionPatient` onto the series' own slice
normal (`cross(IOP[0:3], IOP[3:6])` -- see that function's own docstring for
why this order, not the reverse), ascending -- the physically correct slice order,
independent of whatever order the files happened to be read in
(`InstanceNumber` is not trustworthy for this: it need not be monotonic with
physical position for every vendor). `mask_to_dicom_frames` returns its frames
in that exact order. `highdicom.seg.Segmentation` does **not** re-sort
`source_images` itself when `plane_positions` is not given -- reading
`highdicom/seg/sop.py`'s own `Segmentation.__init__` docstring in
`.venv-clinical` (the "Arrangement" section of the `pixel_array` parameter):
"there must be pixel-for-pixel correspondence between frame `pixel_array[i]`
and `source_images[i]`... It is the caller's responsibility to ensure correct
correspondences." So a caller that sorts the frames but not `source_images`
(or vice versa) gets a SEG object that passes every shape/geometry check and
is still silently wrong, frame by frame. The caller MUST pass the exact same
`sort_datasets_along_normal`-sorted list to both this module's
`mask_to_dicom_frames` and to `dicom_seg.write_dicom_seg`'s `source_datasets`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["sort_datasets_along_normal", "mask_to_dicom_frames"]

logger = logging.getLogger(__name__)

# How far (in mask VOXELS) a DICOM pixel centre is allowed to land from the
# nearest mask voxel centre before this module refuses to sample it. The mask
# was resampled onto this exact series' grid before this module ever runs
# (see the module docstring), so under a correct resample this offset should
# be at most ordinary floating-point round-trip noise -- a few hundredths of a
# voxel, not a fraction that would ever plausibly round to the wrong voxel.
_DEFAULT_MAX_OFFSET_VOXELS = 0.05


def sort_datasets_along_normal(source_datasets: Sequence[Any]) -> list[Any]:
    """Sorts single-frame DICOM datasets into physical slice order.

    Orders by the signed projection of each dataset's own
    `ImagePositionPatient` onto the series' slice normal, both read off the
    FIRST dataset's `ImageOrientationPatient` (a single series has one
    orientation), so the result is ascending along that normal regardless of
    the input order or the sign of the slice-to-slice step.

    The normal is `cross(IOP[0:3], IOP[3:6])` -- the standard DICOM formula
    (PS3.3's own "row cosines x column cosines", in ITS naming, which is the
    OPPOSITE of `col_dir`/`row_dir`'s naming used elsewhere in this module
    and in `mask_to_dicom_frames`'s world-position formula: there, `row_dir`
    is `IOP[3:6]` because it is the direction associated with an increasing
    ROW index, and `col_dir` is `IOP[0:3]` for the same reason with COLUMN.
    Both namings describe the exact same two vectors; only which one gets
    called "row" differs, and it does not change the geometry -- the sign of
    a cross product depends on argument order, not on what its arguments are
    called, and this specific order is what a standard axial series
    (`ImageOrientationPatient = [1, 0, 0, 0, 1, 0]`) needs to sort with
    increasing `ImagePositionPatient[2]` (i.e. increasing DICOM Z /
    superior) as "ascending".

    Args:
        source_datasets: One dataset per slice, in any order. Each must
            expose `ImageOrientationPatient` (the first one only) and
            `ImagePositionPatient` (every one).

    Returns:
        `source_datasets`, re-ordered ascending along the slice normal. This
        is the order `mask_to_dicom_frames` returns its frames in, and the
        order `dicom_seg.write_dicom_seg`'s `source_datasets` argument must
        also be given in -- see the module docstring's ordering trap.

    Raises:
        ValueError: If `source_datasets` is empty.
    """
    if not source_datasets:
        raise ValueError("sort_datasets_along_normal: source_datasets is empty.")

    iop = [float(v) for v in getattr(source_datasets[0], "ImageOrientationPatient")]
    normal = np.cross(np.array(iop[0:3], dtype=float), np.array(iop[3:6], dtype=float))

    def _projection(dataset: Any) -> float:
        ipp = np.array([float(v) for v in getattr(dataset, "ImagePositionPatient")], dtype=float)
        return float(np.dot(ipp, normal))

    return sorted(source_datasets, key=_projection)


def mask_to_dicom_frames(
    mask: np.ndarray,
    mask_affine: np.ndarray,
    source_datasets: Sequence[Any],
    *,
    max_offset_voxels: float = _DEFAULT_MAX_OFFSET_VOXELS,
) -> np.ndarray:
    """Nearest-neighbour samples a label volume onto a DICOM series' own pixel grid.

    For every DICOM pixel, computes that pixel's physical position directly
    from its own slice's `ImagePositionPatient` / `ImageOrientationPatient` /
    `PixelSpacing` (see the module docstring for the exact formula), converts
    LPS -> RAS, maps that RAS position into `mask`'s voxel space through
    `mask_affine`, and reads the nearest voxel. Slices are processed one at a
    time (rather than building one array for the whole volume at once) to
    keep peak memory to roughly one slice's worth of coordinates rather than
    the whole volume's.

    This function sorts `source_datasets` itself (via
    `sort_datasets_along_normal`) before sampling, so it does not matter what
    order they are given in here. That sort is repeated -- not assumed to
    have already happened -- because trusting a caller to have sorted first
    is exactly the kind of unchecked assumption this module exists to
    remove. It is still the CALLER's job to pass that *same* sorted order to
    `dicom_seg.write_dicom_seg`'s own `source_datasets` argument afterwards
    (e.g. `sorted_ds = sort_datasets_along_normal(source_datasets)`, then
    pass `sorted_ds` to both this function and `write_dicom_seg`), because
    `write_dicom_seg` does NOT sort -- see the module docstring's ordering
    trap.

    Args:
        mask: `(X, Y, Z)` label volume (as loaded by `nibabel` off a NIfTI --
            i.e. `mask[i, j, k]` corresponds to voxel index `(i, j, k)` in
            `mask_affine`'s own array-index convention), integer class
            values.
        mask_affine: The mask's own 4x4 NIfTI affine (voxel index -> RAS
            world position).
        source_datasets: The source series' per-slice datasets, in any
            order. Each must expose `Rows`, `Columns`, `PixelSpacing`,
            `ImageOrientationPatient`, and `ImagePositionPatient`.
        max_offset_voxels: How far, in mask voxels, a DICOM pixel centre may
            land from the nearest mask voxel centre before this function
            refuses (see `_DEFAULT_MAX_OFFSET_VOXELS`'s comment for why the
            default is what it is).

    Returns:
        `uint8` array, shape `(n_slices, Rows, Columns)`, in
        `sort_datasets_along_normal`'s order (ascending along the slice
        normal), regardless of `source_datasets`' input order.

    Raises:
        ValueError: If `source_datasets` is empty; if any DICOM pixel centre
            lands more than `max_offset_voxels` from the nearest mask voxel
            centre (names the measured maximum offset); or if any pixel's
            nearest voxel index falls outside `mask`'s shape (names the
            offending slice/row/col) -- either way, the mask and the source
            series are not actually sampled on the same physical grid, and a
            nearest-neighbour pick would silently misplace the mask (see the
            module docstring).
    """
    sorted_datasets = sort_datasets_along_normal(source_datasets)

    mask = np.asarray(mask)
    inv_affine = np.linalg.inv(np.asarray(mask_affine, dtype=float))

    n_slices = len(sorted_datasets)
    rows = int(getattr(sorted_datasets[0], "Rows"))
    cols = int(getattr(sorted_datasets[0], "Columns"))
    frames = np.zeros((n_slices, rows, cols), dtype=np.uint8)

    # Built once: (rows, cols) grids of row/column indices, reused every
    # slice (only IPP/IOP/PixelSpacing change slice to slice).
    row_idx, col_idx = np.meshgrid(
        np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij"
    )

    max_offset_voxels_found = 0.0

    for k, dataset in enumerate(sorted_datasets):
        slice_rows = int(getattr(dataset, "Rows"))
        slice_cols = int(getattr(dataset, "Columns"))
        if slice_rows != rows or slice_cols != cols:
            raise ValueError(
                f"mask_to_dicom_frames: slice {k} is {slice_rows}x{slice_cols}, but slice 0 is "
                f"{rows}x{cols} -- every slice in a series must share the same Rows/Columns."
            )

        iop = [float(v) for v in getattr(dataset, "ImageOrientationPatient")]
        row_dir = np.array(iop[3:6], dtype=float)  # direction of increasing ROW index
        col_dir = np.array(iop[0:3], dtype=float)  # direction of increasing COLUMN index
        ipp = np.array([float(v) for v in getattr(dataset, "ImagePositionPatient")], dtype=float)
        row_spacing, col_spacing = (float(v) for v in getattr(dataset, "PixelSpacing"))

        # Physical (LPS) position of every pixel on this slice -- (rows, cols, 3).
        world_lps = (
            ipp
            + row_idx[..., None] * row_spacing * row_dir
            + col_idx[..., None] * col_spacing * col_dir
        )
        # LPS -> RAS: negate x and y, z unchanged (see module docstring).
        world_ras = world_lps * np.array([-1.0, -1.0, 1.0])

        homogeneous = np.concatenate([world_ras, np.ones((rows, cols, 1))], axis=-1)
        voxel = homogeneous @ inv_affine.T  # (rows, cols, 4), in mask voxel space
        voxel = voxel[..., :3]

        rounded = np.round(voxel)
        slice_max_offset = float(np.abs(voxel - rounded).max())
        max_offset_voxels_found = max(max_offset_voxels_found, slice_max_offset)

        rounded_idx = rounded.astype(int)
        mask_shape = np.array(mask.shape)
        in_bounds = np.all((rounded_idx >= 0) & (rounded_idx < mask_shape), axis=-1)
        if not np.all(in_bounds):
            bad_row, bad_col = (int(v) for v in np.argwhere(~in_bounds)[0])
            raise ValueError(
                f"mask_to_dicom_frames: DICOM pixel (slice {k}, row {bad_row}, col {bad_col}) "
                f"maps to voxel index {tuple(int(v) for v in rounded_idx[bad_row, bad_col])}, "
                f"outside the mask's shape {mask.shape} -- the source series and the mask are "
                "not on the same physical grid."
            )

        frames[k] = mask[rounded_idx[..., 0], rounded_idx[..., 1], rounded_idx[..., 2]]

    if max_offset_voxels_found > max_offset_voxels:
        raise ValueError(
            f"mask_to_dicom_frames: the largest DICOM-pixel-to-mask-voxel offset is "
            f"{max_offset_voxels_found:.4f} voxels, exceeding the {max_offset_voxels} voxel "
            "tolerance -- the mask was not actually resampled onto this series' own grid (see "
            "the module docstring's alignment guard)."
        )

    return frames
