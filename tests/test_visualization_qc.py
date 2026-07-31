"""Tests for neurovision.visualization.qc.

All figures are built from tiny synthetic arrays -- never real BraTS data --
and closed immediately after each assertion so the test suite does not trip
matplotlib's "more than 20 figures opened" warning. The Agg backend is forced
before pyplot is imported so nothing tries to open a window. See CLAUDE.md
for the project's testing rules.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from neurovision.visualization.qc import (  # noqa: E402
    extract_mid_slices,
    mid_slice_indices,
    plot_case_slices,
    plot_intensity_histograms,
)

_SHAPE = (12, 14, 16)


def _make_image(shape: tuple[int, int, int] = _SHAPE) -> np.ndarray:
    """A synthetic 4-channel volume: zero border, nonzero "brain" interior.

    The interior is sized relative to `shape` so this helper works for any
    shape passed in (needed to exercise raw-vs-normalized shape mismatch).
    """
    rng = np.random.default_rng(0)
    image = np.zeros((4, *shape), dtype=np.float32)
    d, h, w = shape
    interior = (slice(1, d - 1), slice(1, h - 1), slice(1, w - 1))
    interior_shape = (d - 2, h - 2, w - 2)
    image[(slice(None), *interior)] = rng.normal(
        loc=100.0, scale=10.0, size=(4, *interior_shape)
    ).astype(np.float32)
    return image


def _make_label(shape: tuple[int, int, int] = _SHAPE) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint8)
    label[2:5, 3:11, 4:12] = 1
    label[5:8, 3:11, 4:12] = 2
    label[8:10, 3:11, 4:12] = 3
    return label


# --- mid_slice_indices / extract_mid_slices --------------------------------


def test_mid_slice_indices_known_shape():
    idx = mid_slice_indices((12, 14, 16))
    assert idx == {"d": 6, "h": 7, "w": 8}


def test_mid_slice_indices_floor_division_on_odd_shape():
    idx = mid_slice_indices((11, 13, 15))
    assert idx == {"d": 5, "h": 6, "w": 7}


def test_extract_mid_slices_returns_expected_keys():
    volume = np.zeros(_SHAPE, dtype=np.float32)
    slices = extract_mid_slices(volume)
    assert set(slices.keys()) == {"sagittal", "coronal", "axial"}


def test_extract_mid_slices_shapes_after_rot90():
    d, h, w = _SHAPE
    volume = np.arange(d * h * w, dtype=np.float32).reshape(_SHAPE)
    slices = extract_mid_slices(volume)
    # np.rot90 swaps the two axes of a 2D array.
    assert slices["sagittal"].shape == (w, h)
    assert slices["coronal"].shape == (w, d)
    assert slices["axial"].shape == (h, d)


def test_extract_mid_slices_values_match_manual_slice_and_rotation():
    d, h, w = _SHAPE
    volume = np.arange(d * h * w, dtype=np.float32).reshape(_SHAPE)
    slices = extract_mid_slices(volume)
    expected_axial = np.rot90(volume[:, :, w // 2])
    np.testing.assert_array_equal(slices["axial"], expected_axial)


# --- plot_case_slices -------------------------------------------------------


def test_plot_case_slices_returns_figure_with_12_axes():
    image = _make_image()
    label = _make_label()
    fig = plot_case_slices(image, label, "case_001")
    assert len(fig.axes) == 12
    plt.close(fig)


def test_plot_case_slices_works_without_label():
    image = _make_image()
    fig = plot_case_slices(image, None, "case_002")
    assert len(fig.axes) == 12
    plt.close(fig)


def test_plot_case_slices_raises_on_3d_image():
    image = np.zeros(_SHAPE, dtype=np.float32)
    with pytest.raises(ValueError):
        plot_case_slices(image, None, "case_003")


def test_plot_case_slices_raises_on_wrong_channel_count():
    image = np.zeros((3, *_SHAPE), dtype=np.float32)
    with pytest.raises(ValueError):
        plot_case_slices(image, None, "case_004")


def test_plot_case_slices_raises_on_label_shape_mismatch():
    image = _make_image()
    bad_label = np.zeros((_SHAPE[0], _SHAPE[1], _SHAPE[2] + 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        plot_case_slices(image, bad_label, "case_005")


# --- plot_intensity_histograms ----------------------------------------------


def test_plot_intensity_histograms_returns_figure_with_8_axes():
    raw = _make_image()
    normalized = _make_image()
    fig = plot_intensity_histograms(raw, normalized, "case_006")
    assert len(fig.axes) == 8
    plt.close(fig)


def test_plot_intensity_histograms_handles_different_spatial_shapes():
    raw = _make_image(shape=(12, 14, 16))
    normalized = _make_image(shape=(8, 8, 8))
    fig = plot_intensity_histograms(raw, normalized, "case_007")
    assert len(fig.axes) == 8
    plt.close(fig)


def test_plot_intensity_histograms_raises_on_non_4d_input():
    raw = np.zeros(_SHAPE, dtype=np.float32)
    normalized = _make_image()
    with pytest.raises(ValueError):
        plot_intensity_histograms(raw, normalized, "case_008")


def test_plot_intensity_histograms_all_zero_channel_does_not_raise():
    raw = _make_image()
    raw[0] = 0.0  # entire T1 channel is background -- no nonzero voxels
    normalized = _make_image()
    normalized[0] = 0.0
    fig = plot_intensity_histograms(raw, normalized, "case_009")
    assert len(fig.axes) == 8
    plt.close(fig)
