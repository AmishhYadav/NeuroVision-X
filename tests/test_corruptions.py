"""Tests for neurovision.data.corruptions (synthetic tiny volumes, CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.data.corruptions import (
    KINDS,
    brain_mask,
    case_seed,
    corrupt_image,
    gamma_contrast,
    kspace_motion,
    motion_line_block,
    thick_slice,
)

SHAPE = (24, 24, 20)
SEVERITY = {"thick_slice": 3, "rician": 0.2, "bias": 0.3, "gamma": 0.5, "motion": 3.0}


def _zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = (x[mask] - x[mask].mean()) / x[mask].std()
    return out


def make_image(through_plane_structure: bool = True) -> np.ndarray:
    """Smooth ellipsoid 'brain', 4 channels, z-scored in mask, background 0."""
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    mask = (g[0] ** 2 / 0.8 + g[1] ** 2 / 0.8 + g[2] ** 2 / 0.9) < 1.0
    chans = []
    for c in range(4):
        v = 1.0 + 0.5 * np.sin(3 * g[0] + c) + 0.4 * np.cos(2 * g[1])
        if through_plane_structure:
            v = v + 0.8 * np.sin(6 * g[2] + c)
        v = v + 0.5 * (g[0] ** 2 + g[1] ** 2 < 0.1)  # a blob with edges
        chans.append(_zscore(np.where(mask, v, 0.0), mask))
    return np.stack(chans).astype(np.float32)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


@pytest.mark.parametrize("kind", [k for k in KINDS if k != "none"])
def test_shape_dtype_background_and_zscore(kind):
    img = make_image()
    mask = brain_mask(img)
    out = corrupt_image(img, kind, SEVERITY[kind], seed=1)
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.all(out[:, ~mask] == 0)
    for c in range(4):
        assert abs(out[c][mask].mean()) < 1e-4
        assert abs(out[c][mask].std() - 1.0) < 1e-4


def test_none_returns_float32_unchanged():
    img = make_image().astype(np.float16)
    out = corrupt_image(img, "none", 0, seed=0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, img.astype(np.float32))


@pytest.mark.parametrize(
    "kind,severity", [("thick_slice", 1), ("bias", 0), ("gamma", 0), ("motion", 0), ("none", 0)]
)
def test_identity_at_zero_severity(kind, severity):
    img = make_image()
    mask = brain_mask(img)
    out = corrupt_image(img, kind, severity, seed=5)
    expected = np.stack([_zscore(img[c], mask) for c in range(4)])
    np.testing.assert_allclose(out, expected, atol=1e-4)


@pytest.mark.parametrize("kind", ["rician", "bias", "gamma", "motion"])
def test_determinism_and_seed_sensitivity(kind):
    img = make_image()
    a = corrupt_image(img, kind, SEVERITY[kind], seed=7)
    b = corrupt_image(img, kind, SEVERITY[kind], seed=7)
    c = corrupt_image(img, kind, SEVERITY[kind], seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_channels_use_independent_noise():
    img = np.repeat(make_image()[:1], 4, axis=0)  # four identical channels
    out = corrupt_image(img, "rician", 0.3, seed=3)
    assert not np.allclose(out[0], out[1])


def test_case_seed():
    s = case_seed(42, "BraTS_001", "rician", 2)
    assert s == case_seed(42, "BraTS_001", "rician", 2)
    assert 0 <= s < 2**32
    others = [
        case_seed(43, "BraTS_001", "rician", 2),
        case_seed(42, "BraTS_002", "rician", 2),
        case_seed(42, "BraTS_001", "bias", 2),
        case_seed(42, "BraTS_001", "rician", 3),
    ]
    assert all(o != s for o in others)


@pytest.mark.parametrize(
    "kind,levels",
    [
        ("rician", (0.05, 0.1, 0.2, 0.3)),
        ("thick_slice", (2, 3, 5, 7)),
        ("bias", (0.1, 0.2, 0.3, 0.5)),
        ("motion", (1, 2, 4, 8)),
    ],
)
def test_error_increases_with_severity(kind, levels):
    img = make_image()
    mask = brain_mask(img)
    clean = np.stack([_zscore(img[c], mask) for c in range(4)])
    diffs = [mean_abs_diff(corrupt_image(img, kind, s, seed=11), clean) for s in levels]
    assert all(b > a for a, b in zip(diffs, diffs[1:])), diffs


def test_thick_slice_one_is_identity_and_constant_volume_unchanged():
    rng = np.random.default_rng(0)
    m = rng.random(SHAPE).astype(np.float32)
    np.testing.assert_array_equal(thick_slice(m, 1), m)
    const = np.repeat(rng.random(SHAPE[:2] + (1,)).astype(np.float32), SHAPE[2], axis=2)
    np.testing.assert_allclose(thick_slice(const, 4), const, atol=1e-6)


def test_thick_slice_hand_computed():
    # Slices 0,1,2,3,4 with slab 2: blocks [0,1] mean .5 (centre .5), [2,3] mean 2.5
    # (centre 2.5), [4] mean 4 (centre 4). Linear interpolation between the centres at voxels 0..4:
    m = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    out = thick_slice(m, 2)
    np.testing.assert_allclose(out[0, 0], [0.5, 1.0, 2.0, 3.0, 4.0], atol=1e-6)


def test_motion_block_excludes_dc_line():
    for n in (24, 25):
        for seed in range(50):
            start, stop = motion_line_block(n, 0.4, np.random.default_rng(seed))
            assert stop - start == round(0.4 * n)
            assert 0 <= start and stop <= n
            assert not (start <= n // 2 < stop)


def test_motion_block_too_large_raises():
    with pytest.raises(ValueError):
        motion_line_block(24, 0.95, np.random.default_rng(0))


def test_gamma_zero_unchanged_and_direction_varies():
    m = np.random.default_rng(0).random(SHAPE).astype(np.float32) + 0.1
    mask = np.ones(SHAPE, dtype=bool)
    np.testing.assert_allclose(gamma_contrast(m, 0.0, np.random.default_rng(1), mask), m, atol=1e-5)
    signs = {
        bool((gamma_contrast(m, 0.5, np.random.default_rng(s), mask) < m).mean() > 0.5)
        for s in range(20)
    }
    assert signs == {True, False}


def test_kspace_motion_zero_shift_identity():
    m = np.random.default_rng(0).random(SHAPE).astype(np.float32)
    np.testing.assert_allclose(kspace_motion(m, 0.0, np.random.default_rng(1)), m, atol=1e-5)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="thick_slice"):
        corrupt_image(make_image(), "blur", 1.0, seed=0)
