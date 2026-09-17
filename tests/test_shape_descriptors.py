"""Tests for `neurovision.anatomy.shape_descriptors`.

Every test runs on CPU on small, hand-built synthetic arrays (well under a
second each) and never touches real BraTS data.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest

from neurovision.anatomy import shape_descriptors
from neurovision.anatomy.burden import REGION_ORDER, CaseGeometry, burden_profile
from neurovision.anatomy.shape_descriptors import shape_profile, shape_profile_keys

try:
    # If test_burden.py exposes its pinned key set, reuse it directly instead
    # of duplicating the list here (see task 4 below).
    from tests.test_burden import _expected_burden_profile_keys
except ImportError:  # pragma: no cover - only hit if test_burden.py changes shape
    _expected_burden_profile_keys = None


def _sphere_mask(
    shape: tuple[int, int, int], center: tuple[float, float, float], r: float
) -> np.ndarray:
    """Boolean array, True inside a Euclidean ball of radius `r` about `center`."""
    ii, jj, kk = np.indices(shape)
    dist = np.sqrt((ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2)
    return dist <= r


# --------------------------------------------------------------------------- #
# 1. Sphere: near-isotropic PCA, bounding-box extent, solid-blob rim thickness
# --------------------------------------------------------------------------- #


def test_sphere_is_near_isotropic_with_expected_extent_and_rim_thickness() -> None:
    classes = np.zeros((32, 32, 32), dtype=np.uint8)
    classes[_sphere_mask((32, 32, 32), (15.5, 15.5, 15.5), 8.0)] = 3  # ET
    geom = CaseGeometry()

    profile = shape_profile(classes, geom)

    assert 0.9 < profile["elongation_ET"] < 1.1
    assert 0.9 < profile["flatness_ET"] < 1.1
    assert profile["extent_ET_i_mm"] == pytest.approx(17.0, abs=1.0)
    assert profile["extent_ET_j_mm"] == pytest.approx(17.0, abs=1.0)
    assert profile["extent_ET_k_mm"] == pytest.approx(17.0, abs=1.0)

    # Solid blob (no core): falls back to 2x the inradius, i.e. ~ the diameter.
    assert profile["rim_thickness_ET_has_core"] == 0.0
    assert profile["rim_thickness_ET_max_mm"] == pytest.approx(16.0, abs=2.0)


# --------------------------------------------------------------------------- #
# 2. Rod: strongly elongated, not flat, principal axis along k (sign fixed)
# --------------------------------------------------------------------------- #


def test_rod_is_elongated_with_principal_axis_along_k() -> None:
    classes = np.zeros((30, 30, 30), dtype=np.uint8)
    classes[10:12, 10:12, 5:25] = 3  # 2 x 2 x 20 rod, long axis = k
    geom = CaseGeometry()

    profile = shape_profile(classes, geom)

    assert profile["elongation_ET"] > 3.0
    assert profile["flatness_ET"] == pytest.approx(1.0, abs=0.2)
    assert profile["principal_axis_ET_i"] == pytest.approx(0.0, abs=1e-6)
    assert profile["principal_axis_ET_j"] == pytest.approx(0.0, abs=1e-6)
    # Sign-normalised so the largest-magnitude component (k) is positive.
    assert profile["principal_axis_ET_k"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. Slab: flat, not elongated
# --------------------------------------------------------------------------- #


def test_slab_is_flat_but_not_elongated() -> None:
    classes = np.zeros((30, 30, 30), dtype=np.uint8)
    classes[5:25, 5:25, 10:12] = 3  # 20 x 20 x 2 slab
    geom = CaseGeometry()

    profile = shape_profile(classes, geom)

    assert profile["flatness_ET"] < 0.2
    assert profile["elongation_ET"] == pytest.approx(1.0, abs=0.2)


# --------------------------------------------------------------------------- #
# 4. Hand-built shell: rim thickness responds to spacing (sampling=spacing used)
# --------------------------------------------------------------------------- #


def _build_cube_shell(core_side: int, shell_thickness: int, margin: int = 5) -> np.ndarray:
    """A `core_side`-cube of NCR (class 1) wrapped by a shell of ET (class 3).

    The shell is `shell_thickness` voxels wide on every face (built by carving
    the NCR cube out of the middle of a larger ET cube), with `margin` voxels
    of background on every side so the shell's outer surface does not touch
    the array edge (touching the edge would make `distance_transform_edt`
    treat "outside the array" as ET, silently understating distances -- the
    same edge-effect `burden.surface_area_mm2` pads against).

    `core_side=10` (rather than a smaller, more "illustrative" cube) is
    deliberate: on a cube shell, voxels near an edge or corner see their
    nearest non-tumour-core voxel diagonally, at up to `shell_thickness *
    sqrt(3)`, which pulls the corner-region thickness values above the
    flat-face value. With too small a core, those inflated corner voxels are
    a large enough fraction of the shell to shift the MEDIAN itself. At
    `core_side=10` the flat-face voxels dominate the vote and the median
    lands exactly on the flat-face analytic value (verified numerically);
    only the max (a handful of corner voxels) still carries the corner
    effect, which is the point of the separate max assertion below.
    """
    outer_side = core_side + 2 * shell_thickness
    size = outer_side + 2 * margin
    classes = np.zeros((size, size, size), dtype=np.uint8)
    o0 = margin
    o1 = margin + outer_side
    classes[o0:o1, o0:o1, o0:o1] = 3  # ET shell (outer cube)
    c0 = margin + shell_thickness
    c1 = c0 + core_side
    classes[c0:c1, c0:c1, c0:c1] = 1  # NCR core, carved out of the middle
    return classes


def test_hand_built_2voxel_shell_median_isotropic() -> None:
    classes = _build_cube_shell(core_side=10, shell_thickness=2)
    geom = CaseGeometry(spacing=(1.0, 1.0, 1.0))

    profile = shape_profile(classes, geom)

    assert profile["rim_thickness_ET_has_core"] == 1.0
    assert profile["rim_thickness_ET_median_mm"] == pytest.approx(2.0, abs=0.15)


def test_hand_built_3voxel_shell_median_and_max_isotropic() -> None:
    classes = _build_cube_shell(core_side=10, shell_thickness=3)
    geom = CaseGeometry(spacing=(1.0, 1.0, 1.0))

    profile = shape_profile(classes, geom)

    assert profile["rim_thickness_ET_has_core"] == 1.0
    # Two-sided estimator, exact on the flat faces: d_out + d_in - mean(spacing)
    # gives 3.0 at the inner voxel (1 + 3 - 1), the middle voxel (2 + 2 - 1),
    # and the outer voxel (3 + 1 - 1) alike.
    assert profile["rim_thickness_ET_median_mm"] == pytest.approx(3.0, abs=0.15)
    # The max comes from the cube's edges and corners, where the nearest
    # tumour-core/background voxel sits diagonally rather than face-on -- a
    # known corner artifact of a CUBE shell, not of the estimator. The
    # diagonal bound for a shell_thickness=3 corner is 3*sqrt(3) ~= 5.196; 10%
    # headroom on top of that comfortably covers it without hiding a real
    # regression.
    assert profile["rim_thickness_ET_max_mm"] <= 3.0 * math.sqrt(3.0) * 1.1


def test_hand_built_3voxel_shell_anisotropic_spacing_increases_median() -> None:
    classes = _build_cube_shell(core_side=10, shell_thickness=3)

    profile_iso = shape_profile(classes, CaseGeometry(spacing=(1.0, 1.0, 1.0)))
    profile_aniso = shape_profile(classes, CaseGeometry(spacing=(2.0, 1.0, 1.0)))

    # Doubling spacing along axis i can only lengthen distances measured
    # (partly) along that axis, so the median can only go up, never down --
    # this is the direct evidence that `sampling=spacing` is really used
    # (not left at its default of 1 in every axis) rather than reused from
    # the isotropic call.
    assert 3.0 <= profile_aniso["rim_thickness_ET_median_mm"] <= 6.0
    assert profile_aniso["rim_thickness_ET_median_mm"] > profile_iso["rim_thickness_ET_median_mm"]

    # The bounding box is exactly twice as wide along i; j and k are untouched.
    assert profile_aniso["extent_ET_i_mm"] == pytest.approx(2.0 * profile_iso["extent_ET_i_mm"])
    assert profile_aniso["extent_ET_j_mm"] == pytest.approx(profile_iso["extent_ET_j_mm"])
    assert profile_aniso["extent_ET_k_mm"] == pytest.approx(profile_iso["extent_ET_k_mm"])


# --------------------------------------------------------------------------- #
# 5. Empty region -> NaN for every key of that region, no warnings
# --------------------------------------------------------------------------- #


def test_empty_region_gives_nan_everywhere_with_no_warnings() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)  # no ET, TC, or WT at all
    geom = CaseGeometry()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = shape_profile(classes, geom)

    for region in REGION_ORDER:
        for key in (
            f"elongation_{region}",
            f"flatness_{region}",
            f"extent_{region}_i_mm",
            f"extent_{region}_j_mm",
            f"extent_{region}_k_mm",
            f"principal_axis_{region}_i",
            f"principal_axis_{region}_j",
            f"principal_axis_{region}_k",
        ):
            assert math.isnan(profile[key]), f"{key} should be NaN for an empty region"
        assert profile[f"n_voxels_{region}"] == 0.0

    assert math.isnan(profile["rim_thickness_ET_median_mm"])
    assert math.isnan(profile["rim_thickness_ET_max_mm"])
    assert math.isnan(profile["rim_thickness_ET_has_core"])


# --------------------------------------------------------------------------- #
# 6. Fewer than 4 voxels -> elongation/flatness NaN, everything else defined
# --------------------------------------------------------------------------- #


def test_fewer_than_four_voxels_gives_nan_ratios_but_defined_extent() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[2, 2, 2] = 3
    classes[2, 2, 3] = 3  # 2 ET voxels total, below the PCA-ratio floor of 4
    geom = CaseGeometry()

    profile = shape_profile(classes, geom)

    assert math.isnan(profile["elongation_ET"])
    assert math.isnan(profile["flatness_ET"])
    assert profile["n_voxels_ET"] == 2.0
    # Extent is still well defined for a tiny region.
    assert profile["extent_ET_k_mm"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 7. shape_profile_keys matches the actual dict order, with no data needed
# --------------------------------------------------------------------------- #


def test_shape_profile_keys_matches_actual_output_order() -> None:
    classes = np.zeros((16, 16, 16), dtype=np.uint8)
    classes[2:6, 2:6, 2:6] = 1
    classes[6:10, 6:10, 6:10] = 2
    classes[10:13, 10:13, 10:13] = 3
    geom = CaseGeometry()

    profile = shape_profile(classes, geom)

    assert list(profile.keys()) == list(shape_profile_keys())


# --------------------------------------------------------------------------- #
# 8. Forbidden-word scan: no clinical language anywhere in the module source
# --------------------------------------------------------------------------- #


def test_module_source_contains_no_forbidden_clinical_words() -> None:
    source = inspect.getsource(shape_descriptors).lower()
    forbidden = ["grade", "stage", "prognosis", "deficit", "impair", "invasi"]
    for word in forbidden:
        assert word not in source, f"forbidden word {word!r} found in shape_descriptors.py"


# --------------------------------------------------------------------------- #
# 9. burden_profile's key set is unaffected by importing this module
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    _expected_burden_profile_keys is None,
    reason="tests.test_burden._expected_burden_profile_keys is not importable",
)
def test_burden_profile_keys_unchanged_by_importing_shape_descriptors() -> None:
    classes = np.zeros((20, 20, 20), dtype=np.uint8)
    classes[0:3, 0:3, 0:3] = 1
    classes[3:8, 3:8, 3:8] = 2
    classes[8:10, 8:10, 8:10] = 3
    geom = CaseGeometry()

    profile = burden_profile(classes, geom)

    assert set(profile.keys()) == _expected_burden_profile_keys()
