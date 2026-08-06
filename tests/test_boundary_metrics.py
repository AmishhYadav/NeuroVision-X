"""Tests for neurovision.metrics.boundary.

Synthetic arrays only (nothing bigger than ~9x9x9), so the whole file runs
on CPU in well under a second.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
import torch

from neurovision.metrics.boundary import (
    DEFAULT_BANDS,
    band_label,
    boundary_band_masks,
    boundary_stratified_errors,
    distance_band_means,
    signed_distance_to_boundary,
)


def _cube_mask(shape: tuple[int, int, int] = (9, 9, 9)) -> torch.Tensor:
    """A 3x3x3 solid cube centred in `shape`."""
    m = torch.zeros(shape, dtype=torch.float32)
    m[3:6, 3:6, 3:6] = 1.0
    return m


# ---------------------------------------------------------------------------
# 1. Hand-computed SDF
# ---------------------------------------------------------------------------


def test_sdf_sign_convention_and_hand_computed_values() -> None:
    mask = _cube_mask()
    sdf = signed_distance_to_boundary(mask)

    assert sdf.shape == (9, 9, 9)
    assert sdf.dtype == torch.float32

    # Centre of the cube: negative (inside).
    assert sdf[4, 4, 4] < 0.0
    # Far corner of the volume: positive (outside), well away from the cube.
    assert sdf[0, 0, 0] > 0.0

    # Voxel just outside a face (index 2 along axis 0, cube starts at 3).
    assert sdf[2, 4, 4].item() == pytest.approx(1.0)
    # Voxel just inside that same face (index 3, the cube's first layer).
    assert sdf[3, 4, 4].item() == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 2. Spacing is honoured
# ---------------------------------------------------------------------------


def test_spacing_scales_distance_per_axis() -> None:
    mask = _cube_mask()
    sdf = signed_distance_to_boundary(mask, spacing=(2.0, 1.0, 1.0))

    # One step away along axis 0 (spacing 2.0): voxel (2, 4, 4) is outside,
    # its nearest True neighbour is (3, 4, 4), one voxel away along axis 0.
    assert sdf[2, 4, 4].item() == pytest.approx(2.0)
    # One step away along axis 2 (spacing 1.0): voxel (4, 4, 2).
    assert sdf[4, 4, 2].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3 & 4. Degenerate masks
# ---------------------------------------------------------------------------


def test_empty_mask_gives_all_nan_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    mask = torch.zeros((5, 5, 5), dtype=torch.float32)
    with caplog.at_level(logging.WARNING, logger="neurovision.metrics.boundary"):
        sdf = signed_distance_to_boundary(mask, name="empty-case")
    assert torch.isnan(sdf).all()
    assert sdf.shape == (5, 5, 5)
    assert any("empty-case" in r.message for r in caplog.records)
    assert any("empty" in r.message.lower() for r in caplog.records)


def test_full_mask_gives_all_nan_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    mask = torch.ones((5, 5, 5), dtype=torch.float32)
    with caplog.at_level(logging.WARNING, logger="neurovision.metrics.boundary"):
        sdf = signed_distance_to_boundary(mask, name="full-case")
    assert torch.isnan(sdf).all()
    assert any("full-case" in r.message for r in caplog.records)


def test_invalid_ndim_raises() -> None:
    with pytest.raises(ValueError):
        signed_distance_to_boundary(torch.zeros((5, 5)))


def test_invalid_spacing_length_raises() -> None:
    with pytest.raises(ValueError):
        signed_distance_to_boundary(_cube_mask(), spacing=(1.0, 1.0))


# ---------------------------------------------------------------------------
# 5. CUDA-safety proxy: numpy input vs torch input give identical results
# ---------------------------------------------------------------------------


def test_numpy_and_torch_inputs_agree() -> None:
    mask = _cube_mask()
    sdf_from_tensor = signed_distance_to_boundary(mask)
    sdf_from_numpy = signed_distance_to_boundary(mask.numpy())
    assert torch.equal(sdf_from_tensor, sdf_from_numpy)
    assert isinstance(sdf_from_numpy, torch.Tensor)


# ---------------------------------------------------------------------------
# 6. Bands are disjoint and complete
# ---------------------------------------------------------------------------


def test_bands_are_pairwise_disjoint() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    masks = boundary_band_masks(sdf, bands=DEFAULT_BANDS)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not (masks[i] & masks[j]).any()


def test_single_infinite_band_covers_every_non_nan_voxel() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    (mask,) = boundary_band_masks(sdf, bands=((0.0, float("inf")),))
    assert torch.equal(mask, ~torch.isnan(sdf))


# ---------------------------------------------------------------------------
# 7. NaN SDF is in no band
# ---------------------------------------------------------------------------


def test_nan_sdf_is_in_no_band() -> None:
    sdf = signed_distance_to_boundary(torch.zeros((5, 5, 5)))
    assert torch.isnan(sdf).all()
    masks = boundary_band_masks(sdf, bands=DEFAULT_BANDS)
    for m in masks:
        assert not m.any()


# ---------------------------------------------------------------------------
# 8. Band validation raises
# ---------------------------------------------------------------------------


def test_overlapping_bands_raise() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    with pytest.raises(ValueError, match=r"overlap"):
        boundary_band_masks(sdf, bands=((0.0, 3.0), (2.0, 5.0)))


def test_lo_ge_hi_raises() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    with pytest.raises(ValueError, match=r"lo < hi"):
        boundary_band_masks(sdf, bands=((2.0, 2.0),))


# ---------------------------------------------------------------------------
# 9. band_label
# ---------------------------------------------------------------------------


def test_band_label_formatting() -> None:
    assert band_label(0.0, 2.0) == "0-2"
    assert band_label(10.0, float("inf")) == "10-inf"
    assert band_label(float("-inf"), 0.0) == "-inf-0"


# ---------------------------------------------------------------------------
# 10. Perfect prediction
# ---------------------------------------------------------------------------


def test_perfect_prediction_has_zero_error_in_every_band() -> None:
    target = _cube_mask().unsqueeze(0)  # (1, D, H, W) -- single region channel
    pred = target.clone()
    result = boundary_stratified_errors(pred, target, region_names=("R",))
    for lo, hi in DEFAULT_BANDS:
        label = band_label(lo, hi)
        n = result[f"bn_R_{label}"]
        if n > 0:
            assert result[f"berr_R_{label}"] == pytest.approx(0.0)
            assert result[f"bfnr_R_{label}"] == pytest.approx(0.0)
            assert result[f"bfpr_R_{label}"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 11. One-voxel-thick over-segmentation
# ---------------------------------------------------------------------------


def test_over_segmentation_gives_fpr_not_fnr_near_boundary() -> None:
    target = _cube_mask()
    pred = target.clone()
    # Dilate by one voxel along axis 0 only: pred is a strict superset of
    # target, so every target voxel stays covered (fnr == 0 everywhere) and
    # the only errors are the added shell just outside the original cube.
    pred[2, 3:6, 3:6] = 1.0
    pred[6, 3:6, 3:6] = 1.0

    result = boundary_stratified_errors(pred.unsqueeze(0), target.unsqueeze(0), region_names=("R",))

    inner_label = band_label(*DEFAULT_BANDS[0])
    assert result[f"bfpr_R_{inner_label}"] > 0.0
    assert result[f"bfnr_R_{inner_label}"] == pytest.approx(0.0)

    for lo, hi in DEFAULT_BANDS:
        label = band_label(lo, hi)
        if result[f"bn_R_{label}"] > 0:
            assert result[f"berr_R_{label}"] == pytest.approx(
                result[f"bfnr_R_{label}"] + result[f"bfpr_R_{label}"]
            )


def test_berr_equals_bfnr_plus_bfpr_on_random_case() -> None:
    torch.manual_seed(0)
    target = (torch.rand(1, 9, 9, 9) > 0.5).to(torch.float32)
    target[0, 3:6, 3:6, 3:6] = 1.0  # guarantee a real interior/boundary
    pred = (torch.rand(1, 9, 9, 9) > 0.5).to(torch.float32)

    result = boundary_stratified_errors(pred, target, region_names=("R",))
    for lo, hi in DEFAULT_BANDS:
        label = band_label(lo, hi)
        if result[f"bn_R_{label}"] > 0:
            assert result[f"berr_R_{label}"] == pytest.approx(
                result[f"bfnr_R_{label}"] + result[f"bfpr_R_{label}"]
            )


# ---------------------------------------------------------------------------
# 12. Band counts sum to number of non-NaN voxels
# ---------------------------------------------------------------------------


def test_band_counts_sum_to_total_non_nan_voxels() -> None:
    target = _cube_mask()
    pred = target.clone()
    result = boundary_stratified_errors(pred.unsqueeze(0), target.unsqueeze(0), region_names=("R",))
    total_n = sum(result[f"bn_R_{band_label(lo, hi)}"] for lo, hi in DEFAULT_BANDS)
    assert total_n == pytest.approx(float(target.numel()))


# ---------------------------------------------------------------------------
# 13. Empty target region
# ---------------------------------------------------------------------------


def test_empty_target_region_gives_nan_rates_and_zero_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = torch.zeros(1, 9, 9, 9)
    pred = _cube_mask().unsqueeze(0)
    with caplog.at_level(logging.WARNING, logger="neurovision.metrics.boundary"):
        result = boundary_stratified_errors(pred, target, region_names=("R",))
    for lo, hi in DEFAULT_BANDS:
        label = band_label(lo, hi)
        assert math.isnan(result[f"berr_R_{label}"])
        assert math.isnan(result[f"bfnr_R_{label}"])
        assert math.isnan(result[f"bfpr_R_{label}"])
        assert result[f"bn_R_{label}"] == 0.0
    assert any("R (target)" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 14. distance_band_means
# ---------------------------------------------------------------------------


def test_distance_band_means_constant_field() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    values = torch.full(sdf.shape, 3.5)
    result = distance_band_means(values, sdf, bands=DEFAULT_BANDS)
    for lo, hi in DEFAULT_BANDS:
        label = band_label(lo, hi)
        if result[f"n_{label}"] > 0:
            assert result[f"mean_{label}"] == pytest.approx(3.5)
        else:
            assert math.isnan(result[f"mean_{label}"])


def test_distance_band_means_empty_band_is_nan_with_zero_count() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    values = torch.zeros(sdf.shape)
    # A band far beyond any distance present in a 9x9x9 volume is empty.
    result = distance_band_means(values, sdf, bands=((1000.0, 2000.0),))
    assert math.isnan(result["mean_1000-2000"])
    assert result["n_1000-2000"] == 0.0


def test_distance_band_means_skips_nan_values() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    values = torch.zeros(sdf.shape)
    values[0, 0, 0] = float("nan")  # inside the far-corner band
    result = distance_band_means(values, sdf, bands=((0.0, float("inf")),))
    assert result["mean_0-inf"] == pytest.approx(0.0)
    assert result["n_0-inf"] == pytest.approx(float(sdf.numel()) - 1.0)


# ---------------------------------------------------------------------------
# 15. distance_band_means shape mismatch
# ---------------------------------------------------------------------------


def test_distance_band_means_shape_mismatch_raises() -> None:
    sdf = signed_distance_to_boundary(_cube_mask())
    values = torch.zeros((3, 3, 3))
    with pytest.raises(ValueError, match=r"\(9, 9, 9\).*\(3, 3, 3\)|\(3, 3, 3\).*\(9, 9, 9\)"):
        distance_band_means(values, sdf)


# ---------------------------------------------------------------------------
# 16. Batch-of-1 accepted, batch of 2 raises
# ---------------------------------------------------------------------------


def test_batch_of_one_accepted_batch_of_two_raises() -> None:
    target = _cube_mask().unsqueeze(0)  # (1, D, H, W)
    pred = target.clone()
    result = boundary_stratified_errors(pred, target, region_names=("R",))
    assert isinstance(result, dict)

    target_b1 = target.unsqueeze(0)  # (1, 1, D, H, W)
    pred_b1 = pred.unsqueeze(0)
    result_b1 = boundary_stratified_errors(pred_b1, target_b1, region_names=("R",))
    assert result_b1.keys() == result.keys()
    for key, value in result.items():
        if math.isnan(value):
            assert math.isnan(result_b1[key])
        else:
            assert result_b1[key] == pytest.approx(value)

    target_b2 = torch.cat([target_b1, target_b1], dim=0)
    pred_b2 = torch.cat([pred_b1, pred_b1], dim=0)
    with pytest.raises(ValueError):
        boundary_stratified_errors(pred_b2, target_b2, region_names=("R",))


def test_region_names_length_mismatch_raises() -> None:
    target = _cube_mask().unsqueeze(0)
    pred = target.clone()
    with pytest.raises(ValueError):
        boundary_stratified_errors(pred, target, region_names=("A", "B"))


def test_numpy_array_ndarray_input_accepted() -> None:
    # np.ndarray is an accepted type for signed_distance_to_boundary per its
    # signature; make sure it actually flows through without error.
    mask = _cube_mask().numpy().astype(np.uint8)
    sdf = signed_distance_to_boundary(mask)
    assert sdf.shape == (9, 9, 9)
