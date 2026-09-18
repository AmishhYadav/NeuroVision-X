"""Tests for neurovision.reporting.dicom_frames.

Pure numpy: `pydicom.Dataset` is stood in for with `types.SimpleNamespace`,
following the same convention `dicom_seg.py`'s DICOM-library-free functions
already use. No real DICOM data, no I/O -- every test builds a tiny synthetic
series (6 slices, 5 rows, 4 columns) directly in memory and runs in well
under a second.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neurovision.reporting.dicom_frames import mask_to_dicom_frames, sort_datasets_along_normal

# A small, deliberately non-cubic, non-isotropic series so a shape-only check
# could never pass by coincidence (see the module docstring's trap #9 note).
_ROWS = 5
_COLS = 4
_N_SLICES = 6
_ROW_SPACING = 1.0
_COL_SPACING = 2.0
_SLICE_SPACING = 1.5
_X0 = 10.0
_Y0 = 20.0
_Z0 = 30.0

# Standard axial orientation: IOP[0:3] (this module's "col_dir") = patient
# Left, IOP[3:6] ("row_dir") = patient Posterior.
_IOP_AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _make_dataset(
    ipp: tuple[float, float, float],
    iop: tuple[float, float, float, float, float, float],
    rows: int = _ROWS,
    cols: int = _COLS,
    pixel_spacing: tuple[float, float] = (_ROW_SPACING, _COL_SPACING),
) -> Any:
    """A `pydicom.Dataset`-shaped `SimpleNamespace` exposing only what this module reads."""
    return SimpleNamespace(
        Rows=rows,
        Columns=cols,
        PixelSpacing=list(pixel_spacing),
        ImageOrientationPatient=list(iop),
        ImagePositionPatient=list(ipp),
    )


def _build_gt() -> np.ndarray:
    """A small blob per slice, `gt[rank, r, c]`, value `rank + 1` -- distinct per slice so a
    slice-ordering bug (not just a within-slice sampling bug) would be caught."""
    gt = np.zeros((_N_SLICES, _ROWS, _COLS), dtype=np.uint8)
    for rank in range(_N_SLICES):
        gt[rank, 1:3, 1:3] = rank + 1
    return gt


def _gt_to_mask_nii(gt: np.ndarray) -> np.ndarray:
    """Re-lays `gt` (DICOM index order `(slice, row, col)`) onto NIfTI-style axes
    `(col, row, slice)` with x AND y flipped -- the exact kind of axis reorder +
    mirroring a real `dcm2niix` RAS mask can show up with (see the module
    docstring). `mask_nii[C-1-c, R-1-r, k] = gt[k, r, c]`.
    """
    return gt.transpose(2, 1, 0)[::-1, ::-1, :].copy()


def _build_affine(
    row_dir: tuple[float, float, float],
    col_dir: tuple[float, float, float],
    z0: float,
    dz: float,
    rows: int = _ROWS,
    cols: int = _COLS,
    row_spacing: float = _ROW_SPACING,
    col_spacing: float = _COL_SPACING,
    x0: float = _X0,
    y0: float = _Y0,
) -> np.ndarray:
    """The affine mapping `_gt_to_mask_nii`'s `(c', r', k)` voxel indices to RAS world
    position, given the SAME `row_dir`/`col_dir`/spacing/`IPP` formula used to build the
    fixture's source datasets. Derived by hand in the delegation report (substituting
    `c = C-1-c'`, `r = R-1-r'` into the DICOM standard's own per-pixel world-position
    formula, then negating x/y for the LPS -> RAS conversion) -- see that report for the
    derivation `mask_to_dicom_frames` is being checked against.
    """
    row_dir_arr = np.asarray(row_dir, dtype=float)
    col_dir_arr = np.asarray(col_dir, dtype=float)
    tx = -x0 - (rows - 1) * row_spacing * row_dir_arr[0] - (cols - 1) * col_spacing * col_dir_arr[0]
    ty = -y0 - (rows - 1) * row_spacing * row_dir_arr[1] - (cols - 1) * col_spacing * col_dir_arr[1]
    return np.array(
        [
            [col_spacing * col_dir_arr[0], row_spacing * row_dir_arr[0], 0.0, tx],
            [col_spacing * col_dir_arr[1], row_spacing * row_dir_arr[1], 0.0, ty],
            [0.0, 0.0, dz, z0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _build_fixture(
    iop: tuple[float, float, float, float, float, float] = _IOP_AXIAL,
    z0: float = _Z0,
    dz: float = _SLICE_SPACING,
    shuffle_order: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray]:
    """Builds `(mask_nii, affine, source_datasets, gt)` for one test scenario.

    `dz` may be negative (a "negative slice direction" series, where physical
    position DEcreases as the loop variable used to build `ImagePositionPatient`
    increases) -- `gt`'s slice axis always represents the PHYSICALLY ASCENDING
    order (ascending along whatever `sort_datasets_along_normal` computes as
    the normal for `iop`), never the raw loop variable, so `gt` is the correct
    expected output regardless of `dz`'s sign. The affine is built from the
    smallest z value and the (always positive) spacing between consecutive
    sorted z values, for the same reason.

    `shuffle_order`, if given, permutes the returned `source_datasets` list
    (indexed by the loop variable, BEFORE sorting) to simulate an arbitrary
    on-disk read order; it does not change which dataset is which.
    """
    col_dir = tuple(iop[0:3])
    row_dir = tuple(iop[3:6])

    z_values = [z0 + dz * loop_index for loop_index in range(_N_SLICES)]
    z0_ascending = min(z_values)
    dz_ascending = abs(dz)

    gt = _build_gt()  # gt[i] = blob value i + 1, i = 0..N-1, i = ascending-normal position
    mask_nii = _gt_to_mask_nii(gt)
    affine = _build_affine(row_dir, col_dir, z0_ascending, dz_ascending)

    datasets = [
        _make_dataset(ipp=(_X0, _Y0, z_values[loop_index]), iop=iop)
        for loop_index in range(_N_SLICES)
    ]
    if shuffle_order is not None:
        datasets = [datasets[i] for i in shuffle_order]
    return mask_nii, affine, datasets, gt


# ---------------------------------------------------------------------------
# 1. Shuffled input, positive slice spacing: sort_datasets_along_normal
#    recovers rank order; mask_to_dicom_frames recovers gt exactly.
# ---------------------------------------------------------------------------


def test_shuffled_positive_spacing_recovers_gt_and_order() -> None:
    shuffle_order = [3, 0, 5, 1, 4, 2]
    mask_nii, affine, datasets, gt = _build_fixture(shuffle_order=shuffle_order)

    ordered = sort_datasets_along_normal(datasets)
    assert [ds.ImagePositionPatient[2] for ds in ordered] == sorted(
        ds.ImagePositionPatient[2] for ds in datasets
    )

    frames = mask_to_dicom_frames(mask_nii, affine, datasets)
    np.testing.assert_array_equal(frames, gt)


# ---------------------------------------------------------------------------
# 2. Descending input order AND a negative slice-to-slice step: the general
#    formula does not care which physical direction "ascending" turns out to
#    be, only that it is consistent.
# ---------------------------------------------------------------------------


def test_descending_order_negative_spacing_recovers_gt() -> None:
    # Datasets already built rank-ascending by _build_fixture; reverse the
    # list order here so the INPUT the function sees is descending.
    mask_nii, affine, datasets, gt = _build_fixture(dz=-_SLICE_SPACING)
    descending_input = list(reversed(datasets))

    ordered = sort_datasets_along_normal(descending_input)
    z_values = [ds.ImagePositionPatient[2] for ds in ordered]
    assert z_values == sorted(z_values)  # ascending along the normal

    frames = mask_to_dicom_frames(mask_nii, affine, descending_input)
    np.testing.assert_array_equal(frames, gt)


# ---------------------------------------------------------------------------
# 3. Alignment guard: a sub-tolerance offset is fine; a 0.3-voxel offset (of
#    a diagonal affine, so 0.3 voxels along one axis is unambiguous) raises,
#    naming the measured offset.
# ---------------------------------------------------------------------------


def test_offset_within_tolerance_is_accepted() -> None:
    mask_nii, affine, datasets, gt = _build_fixture()
    nudged = affine.copy()
    # 0.01 voxels along the column axis (scale _COL_SPACING) -- well under
    # the default 0.05 voxel tolerance.
    nudged[0, 3] += 0.01 * _COL_SPACING

    frames = mask_to_dicom_frames(mask_nii, nudged, datasets)
    np.testing.assert_array_equal(frames, gt)


def test_offset_beyond_tolerance_raises_naming_the_offset() -> None:
    mask_nii, affine, datasets, _ = _build_fixture()
    nudged = affine.copy()
    nudged[0, 3] += 0.3 * _COL_SPACING  # exactly 0.3 voxels along the column axis

    with pytest.raises(ValueError, match="offset") as excinfo:
        mask_to_dicom_frames(mask_nii, nudged, datasets)

    match = re.search(r"([\d.]+) voxels", str(excinfo.value))
    assert match is not None, str(excinfo.value)
    assert abs(float(match.group(1)) - 0.3) < 1e-6


def test_index_outside_mask_raises() -> None:
    mask_nii, affine, datasets, _ = _build_fixture()
    shifted = affine.copy()
    shifted[0, 3] += 1000.0  # every pixel now maps far outside the mask's shape

    with pytest.raises(ValueError, match="outside the mask's shape"):
        mask_to_dicom_frames(mask_nii, shifted, datasets)


# ---------------------------------------------------------------------------
# 4. Oblique orientation (row/col directions rotated 30 degrees about z):
#    proves the general (non-axis-aligned) formula, not just the axial case.
# ---------------------------------------------------------------------------


def test_oblique_orientation_recovers_gt_exactly() -> None:
    theta = np.deg2rad(30.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # col_dir = R_z(theta) @ [1, 0, 0], row_dir = R_z(theta) @ [0, 1, 0].
    iop = (cos_t, sin_t, 0.0, -sin_t, cos_t, 0.0)

    shuffle_order = [4, 1, 3, 0, 5, 2]
    mask_nii, affine, datasets, gt = _build_fixture(iop=iop, shuffle_order=shuffle_order)

    frames = mask_to_dicom_frames(mask_nii, affine, datasets)
    np.testing.assert_array_equal(frames, gt)


# ---------------------------------------------------------------------------
# 5. sort_datasets_along_normal's own error behaviour.
# ---------------------------------------------------------------------------


def test_sort_datasets_along_normal_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        sort_datasets_along_normal([])


def test_mask_to_dicom_frames_rejects_empty_input() -> None:
    mask_nii, affine, _, _ = _build_fixture()
    with pytest.raises(ValueError, match="empty"):
        mask_to_dicom_frames(mask_nii, affine, [])
