"""Tests for `neurovision.analysis.nnunet_import`.

Every test runs on CPU, on small synthetic arrays, well under a second, and
never touches real BraTS or nnU-Net output. `tmp_path` is the only place any
of these tests write a file.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurovision.analysis.nnunet_import import (
    import_prediction,
    labels_to_pseudo_logits,
    nnunet_to_project_labels,
    roundtrip_self_test,
    score_imported_case,
)

# --------------------------------------------------------------------------- #
# Shared synthetic-case fixtures
# --------------------------------------------------------------------------- #

_ORIGINAL_SHAPE = (12, 10, 8)
_BBOX = [[2, 8], [1, 7], [0, 5]]  # cropped shape (6, 6, 5)
# BraTS 2021's own convention (matches neurovision.data.preprocessing's
# default target_axcodes), used exactly as docs/research_docs/lessons.md and
# real_dicom_scoring's own tests do.
_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])
_TARGET_AXCODES = ("L", "P", "S")


def _nnunet_cropped_case() -> np.ndarray:
    """A (6, 6, 5) cropped array in nnU-Net's convention (1=ED, 2=NCR, 3=ET)."""
    arr = np.zeros((6, 6, 5), dtype=np.uint8)
    arr[0:2, 0:2, 0:2] = 1  # nnU-Net ED -> our label 2
    arr[3:5, 3:5, 3:5] = 2  # nnU-Net NCR -> our label 1
    arr[5, 5, 4] = 3  # ET, unchanged either way
    return arr


def _write_full_nifti(cropped: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Places `cropped` at `_BBOX` inside `_ORIGINAL_SHAPE` and writes it as a NIfTI."""
    full = np.zeros(_ORIGINAL_SHAPE, dtype=np.int16)
    slices = tuple(slice(a, b) for a, b in _BBOX)
    full[slices] = cropped
    nib.save(nib.Nifti1Image(full, affine), str(path))


def _meta() -> dict:
    return {
        "affine": _AFFINE.tolist(),
        "original_shape": list(_ORIGINAL_SHAPE),
        "bbox": _BBOX,
        "spacing": [1.0, 1.0, 1.0],
    }


# --------------------------------------------------------------------------- #
# nnunet_to_project_labels
# --------------------------------------------------------------------------- #


def test_nnunet_to_project_labels_swaps_1_and_2() -> None:
    pred = np.array([0, 1, 2, 3], dtype=np.uint8)

    out = nnunet_to_project_labels(pred)

    assert out.tolist() == [0, 2, 1, 3]
    assert out.dtype == np.uint8


def test_nnunet_to_project_labels_rejects_invalid_value() -> None:
    pred = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError):
        nnunet_to_project_labels(pred)


# --------------------------------------------------------------------------- #
# import_prediction
# --------------------------------------------------------------------------- #


def test_import_prediction_roundtrip_on_synthetic_grid(tmp_path: Path) -> None:
    cropped_nnunet = _nnunet_cropped_case()
    path = tmp_path / "pred.nii.gz"
    _write_full_nifti(cropped_nnunet, _AFFINE, path)

    result = import_prediction(path, _meta(), _TARGET_AXCODES)

    expected = nnunet_to_project_labels(cropped_nnunet)
    assert np.array_equal(result, expected)
    assert result.shape == (6, 6, 5)


def test_import_prediction_rejects_shape_mismatch(tmp_path: Path) -> None:
    cropped_nnunet = _nnunet_cropped_case()
    path = tmp_path / "pred.nii.gz"
    _write_full_nifti(cropped_nnunet, _AFFINE, path)

    bad_meta = _meta()
    bad_meta["original_shape"] = [12, 10, 9]  # deliberately wrong

    with pytest.raises(ValueError):
        import_prediction(path, bad_meta, _TARGET_AXCODES)


def test_import_prediction_rejects_mismatched_affine(tmp_path: Path) -> None:
    """A translated affine (not just flipped) must not pass the geometry gate."""
    cropped_nnunet = _nnunet_cropped_case()
    path = tmp_path / "pred.nii.gz"
    _write_full_nifti(cropped_nnunet, _AFFINE, path)

    bad_meta = _meta()
    translated = _AFFINE.copy()
    translated[0, 3] += 5.0
    bad_meta["affine"] = translated.tolist()

    with pytest.raises(ValueError):
        import_prediction(path, bad_meta, _TARGET_AXCODES)


def test_mirrored_header_does_not_silently_pass(tmp_path: Path) -> None:
    """A NIfTI whose header lies about orientation must not be scored silently.

    A voxel-content swap that survives a combined axis-0/axis-1 flip proves
    the mismatch: this fixture places different labels at (0, 0, 2) and
    (5, 5, 2), so a flip along both axes swaps which value lands at (0, 0,
    2). Either `import_prediction` raises on the affine check, or the
    decoded result differs from the correctly-written case -- it must never
    come out identical, which is what "catches it" means here (the self-test
    described in the spec is exactly this shape: it must not pass silently).
    """
    cropped_nnunet = np.zeros((6, 6, 5), dtype=np.uint8)
    cropped_nnunet[0, 0, 2] = 1  # ED
    cropped_nnunet[5, 5, 2] = 2  # NCR

    correct_path = tmp_path / "correct.nii.gz"
    _write_full_nifti(cropped_nnunet, _AFFINE, correct_path)
    correct_result = import_prediction(correct_path, _meta(), _TARGET_AXCODES)

    # Header lies: claims RAS (identity) when the true grid is LPS. This
    # coincidentally reorients to an affine equal to meta["affine"] here
    # (both have zero translation), so the affine check alone cannot catch
    # it -- exactly the sneaky failure mode docs/research_docs/lessons.md trap 3 warns
    # about for a symmetric geometric check.
    wrong_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    wrong_path = tmp_path / "wrong.nii.gz"
    _write_full_nifti(cropped_nnunet, wrong_affine, wrong_path)

    try:
        wrong_result = import_prediction(wrong_path, _meta(), _TARGET_AXCODES)
    except ValueError:
        return  # also an acceptable catch: the affine check raised directly.

    assert not np.array_equal(wrong_result, correct_result)


# --------------------------------------------------------------------------- #
# labels_to_pseudo_logits
# --------------------------------------------------------------------------- #


def test_labels_to_pseudo_logits_channel_order_and_signs() -> None:
    # One voxel per class, in class order: background, NCR, ED, ET.
    labels = np.array([0, 1, 2, 3], dtype=np.uint8).reshape(1, 1, 4)

    logits = labels_to_pseudo_logits(labels, magnitude=20.0)

    assert logits.shape == (3, 1, 1, 4)
    assert logits.dtype == np.float32
    # Channel order (ET, TC, WT).
    et, tc, wt = logits[0, 0, 0], logits[1, 0, 0], logits[2, 0, 0]
    assert et.tolist() == [-20.0, -20.0, -20.0, 20.0]  # ET = {3}
    assert tc.tolist() == [-20.0, 20.0, -20.0, 20.0]  # TC = {1, 3}
    assert wt.tolist() == [-20.0, 20.0, 20.0, 20.0]  # WT = {1, 2, 3}


# --------------------------------------------------------------------------- #
# score_imported_case
# --------------------------------------------------------------------------- #


def _nested_cubes_label() -> np.ndarray:
    """Three concentric cubes, each region mask well above min_component_size (50)."""
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[1:15, 1:15, 1:15] = 2  # ED shell -> WT mask is the whole 14^3 cube
    label[3:13, 3:13, 3:13] = 1  # NCR -> TC mask is the 10^3 cube
    label[5:11, 5:11, 5:11] = 3  # ET -> ET mask is the 6^3 cube
    return label


def test_score_imported_case_perfect_prediction_is_dice_one() -> None:
    label = _nested_cubes_label()

    metrics = score_imported_case(label, label, spacing=(1.0, 1.0, 1.0))

    assert metrics["dice_ET"] == pytest.approx(1.0)
    assert metrics["dice_TC"] == pytest.approx(1.0)
    assert metrics["dice_WT"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# roundtrip_self_test
# --------------------------------------------------------------------------- #


def test_roundtrip_self_test_passes_on_synthetic_case(tmp_path: Path) -> None:
    label_cropped = np.zeros((6, 6, 5), dtype=np.uint8)
    label_cropped[0:2, 0:2, 0:2] = 1  # NCR
    label_cropped[3:5, 3:5, 3:5] = 2  # ED
    label_cropped[5, 5, 4] = 3  # ET

    roundtrip_self_test(label_cropped, _meta(), _TARGET_AXCODES, tmp_path)  # must not raise


def test_roundtrip_self_test_raises_without_ncr_and_ed(tmp_path: Path) -> None:
    label_cropped = np.zeros((6, 6, 5), dtype=np.uint8)
    label_cropped[0, 0, 0] = 3  # ET only, no NCR (1) or ED (2)

    with pytest.raises(ValueError):
        roundtrip_self_test(label_cropped, _meta(), _TARGET_AXCODES, tmp_path)


def test_roundtrip_self_test_catches_a_missed_swap(tmp_path: Path, monkeypatch) -> None:
    """If the decode swap is skipped, the round trip must fail, not pass silently."""
    label_cropped = np.zeros((6, 6, 5), dtype=np.uint8)
    label_cropped[0:2, 0:2, 0:2] = 1  # NCR
    label_cropped[3:5, 3:5, 3:5] = 2  # ED
    label_cropped[5, 5, 4] = 3  # ET

    # Patches only the DECODE step (called inside import_prediction) to skip
    # the 1<->2 swap; roundtrip_self_test's own ENCODE step applies the swap
    # inline and is untouched by this patch, so the two are now inconsistent
    # -- exactly the "missed swap" bug this self-test exists to catch.
    monkeypatch.setattr(
        "neurovision.analysis.nnunet_import.nnunet_to_project_labels",
        lambda pred: np.asarray(pred, dtype=np.uint8),
    )

    with pytest.raises(AssertionError):
        roundtrip_self_test(label_cropped, _meta(), _TARGET_AXCODES, tmp_path)
