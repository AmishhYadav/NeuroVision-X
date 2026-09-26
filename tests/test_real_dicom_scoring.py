"""Tests for `neurovision.analysis.real_dicom_scoring`.

Every test runs on CPU, on small synthetic arrays, well under a second, and
never touches real BraTS or DICOM data. Values used for Dice/region checks
are hand-computed in comments so a failure is easy to diagnose.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.analysis.real_dicom_scoring import (
    lateralisation_check,
    region_masks,
    roundtrip_self_test,
    score_case,
    to_reference_grid,
    uncrop_to_full,
)
from neurovision.data.preprocessing import reorient_to_axcodes

# --------------------------------------------------------------------------- #
# uncrop_to_full
# --------------------------------------------------------------------------- #


def test_uncrop_to_full_places_bbox_correctly() -> None:
    cropped = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)  # (1, 2, 2)
    meta = {"bbox": [[1, 2], [0, 2], [1, 3]], "original_shape": [3, 3, 4]}

    full = uncrop_to_full(cropped, meta)

    assert full.shape == (3, 3, 4)
    assert full.dtype == cropped.dtype
    assert np.array_equal(full[1:2, 0:2, 1:3], cropped)
    # Everything outside the bbox is zero.
    full_minus_bbox = full.copy()
    full_minus_bbox[1:2, 0:2, 1:3] = 0
    assert np.all(full_minus_bbox == 0)


def test_uncrop_to_full_rejects_shape_mismatch() -> None:
    cropped = np.zeros((2, 2, 2), dtype=np.uint8)
    meta = {"bbox": [[0, 3], [0, 2], [0, 2]], "original_shape": [4, 4, 4]}
    with pytest.raises(ValueError):
        uncrop_to_full(cropped, meta)


def test_uncrop_to_full_works_without_source_axcodes() -> None:
    """A meta.json missing `source_axcodes` (older files) must still work."""
    cropped = np.ones((1, 1, 1), dtype=np.uint8)
    meta = {
        "bbox": [[0, 1], [0, 1], [0, 1]],
        "original_shape": [2, 2, 2],
        "spacing": [1.0, 1.0, 1.0],
        # deliberately no "source_axcodes" key
    }
    full = uncrop_to_full(cropped, meta)
    assert full.shape == (2, 2, 2)
    assert full[0, 0, 0] == 1


# --------------------------------------------------------------------------- #
# to_reference_grid
# --------------------------------------------------------------------------- #


def _random_full() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 4, size=(4, 5, 6)).astype(np.uint8)


def test_to_reference_grid_matches_reorient_oracle_on_a_flip() -> None:
    """A pure LPS -> RAS flip must match `reorient_to_axcodes`'s own output exactly."""
    full = _random_full()
    source_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected, reference_affine = reorient_to_axcodes(full, source_affine, ("R", "A", "S"))

    out = to_reference_grid(full, source_affine, reference_affine)

    assert np.array_equal(out, expected)


def test_to_reference_grid_matches_reorient_oracle_on_a_permutation() -> None:
    """A reorientation that also permutes axis order must match the same oracle."""
    full = _random_full()
    source_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected, reference_affine = reorient_to_axcodes(full, source_affine, ("A", "R", "S"))

    out = to_reference_grid(full, source_affine, reference_affine)

    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_to_reference_grid_is_identity_for_equal_affines() -> None:
    full = _random_full()
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    out = to_reference_grid(full, affine, affine)

    assert np.array_equal(out, full)


def test_to_reference_grid_raises_on_translated_affine() -> None:
    full = _random_full()
    source_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    _, reference_affine = reorient_to_axcodes(full, source_affine, ("R", "A", "S"))
    translated = reference_affine.copy()
    translated[0, 3] += 5.0  # break translation only

    with pytest.raises(ValueError):
        to_reference_grid(full, source_affine, translated)


def test_to_reference_grid_raises_on_rescaled_affine() -> None:
    full = _random_full()
    source_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    _, reference_affine = reorient_to_axcodes(full, source_affine, ("R", "A", "S"))
    rescaled = reference_affine.copy()
    rescaled[:3, :3] *= 2.0  # break spacing only

    with pytest.raises(ValueError):
        to_reference_grid(full, source_affine, rescaled)


# --------------------------------------------------------------------------- #
# region_masks
# --------------------------------------------------------------------------- #


def test_region_masks_hand_computed() -> None:
    # One voxel of each class, in class order: background, NCR, ED, ET.
    label = np.array([0, 1, 2, 3]).reshape(1, 1, 4)

    regions = region_masks(label)

    assert set(regions.keys()) == {"ET", "TC", "WT"}
    # ET = {3}: only the last voxel.
    assert regions["ET"].tolist() == [[[False, False, False, True]]]
    # TC = {1, 3}: voxels 1 and 3.
    assert regions["TC"].tolist() == [[[False, True, False, True]]]
    # WT = {1, 2, 3}: voxels 1, 2, 3.
    assert regions["WT"].tolist() == [[[False, True, True, True]]]
    for mask in regions.values():
        assert mask.dtype == np.bool_


# --------------------------------------------------------------------------- #
# score_case
# --------------------------------------------------------------------------- #


def test_score_case_perfect_prediction_is_dice_one() -> None:
    gt = np.zeros((2, 2, 4), dtype=np.uint8)
    gt[0, 0, :] = [1, 2, 3, 0]  # every region present and nonempty

    metrics = score_case(gt, gt, (1.0, 1.0, 1.0))

    assert metrics["dice_ET"] == pytest.approx(1.0)
    assert metrics["dice_TC"] == pytest.approx(1.0)
    assert metrics["dice_WT"] == pytest.approx(1.0)
    # Identical masks have no boundary error.
    assert metrics["hd95_ET"] == pytest.approx(0.0)
    assert metrics["hd95_TC"] == pytest.approx(0.0)
    assert metrics["hd95_WT"] == pytest.approx(0.0)


def test_score_case_hand_computed_partial_overlap() -> None:
    # WT-only labels (value 2 = ED) along one axis, so ET/TC stay empty on
    # both sides and only WT overlap has to be reasoned about by hand.
    gt = np.zeros((1, 1, 4), dtype=np.uint8)
    gt[0, 0, 0:3] = 2  # WT = {0, 1, 2}, size 3
    pred = np.zeros((1, 1, 4), dtype=np.uint8)
    pred[0, 0, 1:4] = 2  # WT = {1, 2, 3}, size 3

    metrics = score_case(pred, gt, (1.0, 1.0, 1.0))

    # intersection = {1, 2}, size 2 -> dice = 2*2 / (3+3) = 0.6667
    assert metrics["dice_WT"] == pytest.approx(2 * 2 / 6)
    # Both ET and TC are empty on both sides -> Dice 1.0 by convention.
    assert metrics["dice_ET"] == pytest.approx(1.0)
    assert metrics["dice_TC"] == pytest.approx(1.0)


def test_score_case_both_empty_et_is_dice_one() -> None:
    gt = np.zeros((1, 1, 4), dtype=np.uint8)
    gt[0, 0, :] = [1, 2, 0, 0]  # no ET (label 3) anywhere
    pred = np.zeros((1, 1, 4), dtype=np.uint8)
    pred[0, 0, :] = [1, 1, 2, 0]  # also no ET

    metrics = score_case(pred, gt, (1.0, 1.0, 1.0))

    assert metrics["dice_ET"] == pytest.approx(1.0)


def test_score_case_rejects_shape_mismatch() -> None:
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    pred = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        score_case(pred, gt, (1.0, 1.0, 1.0))


# --------------------------------------------------------------------------- #
# roundtrip_self_test
# --------------------------------------------------------------------------- #


def test_roundtrip_self_test_passes_on_synthetic_case() -> None:
    gt_cropped = np.zeros((2, 2, 4), dtype=np.uint8)
    gt_cropped[0, 0, :] = [1, 2, 3, 0]
    meta = {
        "bbox": [[0, 2], [0, 2], [0, 4]],
        "original_shape": [4, 4, 6],
        "spacing": [1.0, 1.0, 1.0],
    }

    metrics = roundtrip_self_test(gt_cropped, meta)

    assert metrics["dice_ET"] == 1.0
    assert metrics["dice_TC"] == 1.0
    assert metrics["dice_WT"] == 1.0


# --------------------------------------------------------------------------- #
# lateralisation_check
# --------------------------------------------------------------------------- #

_LR_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])  # LPS: axis 0 is left-right.


def _one_sided_lesion(indices: range) -> np.ndarray:
    labels = np.zeros((20, 1, 1), dtype=np.uint8)
    for i in indices:
        labels[i, 0, 0] = 2  # ED -> counts toward WT
    return labels


def test_lateralisation_check_ok_for_correct_prediction() -> None:
    gt = _one_sided_lesion(range(5, 14))  # indices 5..13, one-sided lesion
    pred = gt.copy()

    assert lateralisation_check(pred, gt, _LR_AFFINE) == "ok"


def test_lateralisation_check_mirrored_for_flipped_prediction() -> None:
    gt = _one_sided_lesion(range(5, 14))  # indices 5..13
    pred_flipped = _one_sided_lesion(range(6, 15))  # indices 6..14 = flip(gt)

    assert lateralisation_check(pred_flipped, gt, _LR_AFFINE) == "mirrored"


def test_lateralisation_check_ambiguous_for_symmetric_midline_lesion() -> None:
    # Same masks as the "mirrored" case, but with the discriminating margin
    # widened past the gap between direct and flipped Dice (0.111): a
    # near-midline lesion where this check genuinely cannot tell.
    gt = _one_sided_lesion(range(5, 14))
    pred_flipped = _one_sided_lesion(range(6, 15))

    assert lateralisation_check(pred_flipped, gt, _LR_AFFINE, margin=0.5) == "ambiguous"


def test_lateralisation_check_rejects_shape_mismatch() -> None:
    gt = np.zeros((20, 1, 1), dtype=np.uint8)
    pred = np.zeros((19, 1, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        lateralisation_check(pred, gt, _LR_AFFINE)
