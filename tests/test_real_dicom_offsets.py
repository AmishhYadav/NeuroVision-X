"""Tests for scripts/real_dicom_offsets.py on tiny synthetic volumes."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tests.script_loader import load_script

rdo = load_script("real_dicom_offsets")


def _cube(shape, lo, hi):
    m = np.zeros(shape, dtype=bool)
    m[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return m


def test_shift_mask_zero_fill_no_wrap():
    m = np.zeros((6, 6, 6), dtype=bool)
    m[5, 0, 0] = True
    out = rdo.shift_mask(m, (1, 0, 0))
    assert out.sum() == 0  # fell off the edge, did not wrap to index 0
    out = rdo.shift_mask(m, (-2, 1, 3))
    assert out[3, 1, 3] and out.sum() == 1
    assert rdo.shift_mask(m, (9, 0, 0)).sum() == 0


def test_best_shift_recovers_known_translation():
    fixed = _cube((20, 20, 20), (8, 8, 8), (12, 12, 12))
    moving = rdo.shift_mask(fixed, (-2, 3, -1))
    s = rdo.best_shift(moving, fixed, 5)
    assert s.tolist() == [2, -3, 1]
    assert np.array_equal(rdo.shift_mask(moving, s), fixed)


def test_best_shift_no_wraparound_artefact():
    # Moving block at the start of axis 0, fixed block at the end. The true
    # shift (+8) is outside the window of 3. On a length-10 axis, a circular
    # correlation aliases lag +8 to lag -2 and would report a perfect overlap
    # there. With padding the window has no overlap at all: peak 0, lag 0.
    moving = _cube((10, 3, 3), (0, 0, 0), (2, 3, 3))
    fixed = _cube((10, 3, 3), (8, 0, 0), (10, 3, 3))
    unpadded = np.fft.irfftn(
        np.conj(np.fft.rfftn(moving.astype(np.float32))) * np.fft.rfftn(fixed.astype(np.float32)),
        s=moving.shape,
        axes=(0, 1, 2),
    )
    assert np.round(unpadded[-2, 0, 0]) == moving.sum()  # the wrapped artefact exists
    s, peak = rdo.best_shift_with_peak(moving, fixed, 3)
    assert s.tolist() == [0, 0, 0] and peak == 0


def test_best_shift_at_window_edge():
    fixed = _cube((24, 24, 24), (10, 10, 10), (14, 14, 14))
    moving = rdo.shift_mask(fixed, (-5, 5, 0))
    assert rdo.best_shift(moving, fixed, 5).tolist() == [5, -5, 0]


def test_best_shift_empty_mask_not_found():
    z = np.zeros((8, 8, 8), dtype=bool)
    fixed = _cube((8, 8, 8), (2, 2, 2), (4, 4, 4))
    s, peak = rdo.best_shift_with_peak(z, fixed, 2)
    assert s.tolist() == [0, 0, 0] and peak == 0
    assert rdo.best_shift_with_peak(fixed, fixed, 2)[1] == 8


def test_shift_mask_axes_and_dropping():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[0, 1, 2] = True
    m[4, 4, 4] = True
    out = rdo.shift_mask(m, (0, 2, 1))
    assert out[0, 3, 3] and not out[4].any() and out.sum() == 1  # (4,4,4) dropped on axis 1
    out = rdo.shift_mask(m, (-1, 0, 0))
    assert out[3, 4, 4] and out.sum() == 1  # (0,1,2) dropped on axis 0
    with pytest.raises(ValueError):
        rdo.shift_mask(m, (1, 0))


def test_dice_values():
    a = _cube((4, 4, 4), (0, 0, 0), (2, 2, 2))  # 8 voxels
    b = _cube((4, 4, 4), (0, 0, 0), (2, 2, 4))  # 16 voxels, overlap 8
    assert rdo.dice(a, a) == 1.0
    assert rdo.dice(a, b) == pytest.approx(2 * 8 / 24)
    assert rdo.dice(a, ~a) == 0.0
    z = np.zeros((2, 2, 2), dtype=bool)
    assert rdo.dice(z, z) == 1.0


def test_all_job_ids(tmp_path):
    j1, j2, j3, j4 = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)
    log = tmp_path / "run.log"
    log.write_text(
        f"x run_clinical_study: created job {j1} for study_dir=/o/_extract/BraTS2021_00001/0\n"
        f"x run_clinical_study: created job {j2} for study_dir=/o/_extract/BraTS2021_00001/0\n"
        f"x run_clinical_study: created job {j3} for study_dir=/o/_extract/BraTS2021_00010/0\n"
        "x something else entirely\n"
        f"x created job {j4} for study_dir=/o/_extract/BraTS2021_0001/0\n"
        f"x created job {'z' * 32} for study_dir=/o/_extract/BraTS2021_00002/0\n"
    )
    assert rdo._all_job_ids(log) == {"BraTS2021_00001": [j1, j2], "BraTS2021_00010": [j3]}


def test_resolve_job_falls_back_to_job_with_mask(tmp_path):
    j1, j2 = "a" * 32, "b" * 32
    d = tmp_path / "jobs" / j1 / "dicom_seg_work"
    d.mkdir(parents=True)
    (d / "atlas_space_mask.nii.gz").write_bytes(b"")
    assert rdo.resolve_job("c", {"c": [j1, j2]}, tmp_path) == j1
    with pytest.raises(ValueError):
        rdo.resolve_job("c", {"c": [j2]}, tmp_path)


def test_analyse_case_and_self_check():
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    gt[8:12, 8:12, 8:12] = 1
    ours = rdo.shift_mask(gt, (0, 2, 0))
    atlas = _cube((20, 20, 20), (4, 4, 4), (16, 16, 16))
    brain = rdo.shift_mask(atlas, (0, -1, 0))
    row = rdo.analyse_case("c", ours, gt, brain, atlas, 0.5, 4, 0.5)
    assert (row["t0"], row["t1"], row["t2"]) == (0, -2, 0)
    assert (row["b0"], row["b1"], row["b2"]) == (0, 1, 0)
    assert row["wt_tumour_fit"] == 1.0
    assert row["wt_raw"] == pytest.approx(0.5)
    assert row["t_norm_mm"] == pytest.approx(2.0)
    row["logged_dice_WT"] = row["wt_raw"] + 1e-6
    rdo.check_dice_match([row], 1e-4)
    row["logged_dice_WT"] = 0.9
    with pytest.raises(ValueError, match="c"):
        rdo.check_dice_match([row], 1e-4)


def test_atlas_affine_mismatch_raises(tmp_path):
    data = np.ones((3, 3, 3), dtype=np.float32)
    path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    assert rdo.load_atlas_brain(path, np.eye(4)).all()
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    with pytest.raises(ValueError, match="affine"):
        rdo.load_atlas_brain(path, shifted)


def test_summarise_sign_counts_and_pairing():
    per = pd.DataFrame(
        {
            "t0": [1, 2, 3, 4],
            "t1": [3, -1, 0, 9],
            "t2": [0, 0, 0, 0],
            "b0": [-1, -2, -3, -4],
            "b1": [-3, 1, 0, 5],
            "b2": [1, 2, 3, 4],
            "t_found": [True, True, True, False],
        }
    )
    for name in ("raw", "brain_corrected", "tumour_fit"):
        per[f"wt_{name}"] = [0.5, 0.6, 0.7, 0.8]
        per[f"tc_{name}"] = [0.5, 0.6, 0.7, 0.8]
        per[f"usable_{name}"] = [True, False, True, False]
    per["wt_brain_corrected"] = [0.6, 0.7, 0.8, 0.9]
    cohort = pd.DataFrame({"norm": [1.0, 3.0, 5.0, 6.0]})
    out = rdo.summarise(per, cohort)
    assert out["n_t_excluded"] == 1
    assert out["t1_signs"] == {"positive": 1, "negative": 1, "zero": 1}
    assert out["t_vs_b"]["axis0"]["pearson_r"] == pytest.approx(-1.0)  # 4th row excluded
    assert out["t_vs_b"]["axis1"]["pearson_r"] == pytest.approx(-1.0)
    assert out["n_improved_wt_brain_corrected"] == 4
    assert out["usable_raw"] == 2
    assert out["cohort_brain_to_atlas"]["frac_ge_5mm"] == 0.5
