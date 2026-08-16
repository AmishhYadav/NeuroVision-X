"""Tests for `neurovision.anatomy.burden`.

Every test runs on CPU on small synthetic arrays (well under a second each)
and never touches real BraTS data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from neurovision.anatomy import burden
from neurovision.anatomy.burden import (
    REGION_ORDER,
    CaseGeometry,
    burden_profile,
    centroid,
    compute_fractions,
    compute_volumes,
    connected_components,
    estimate_midline_index,
    laterality,
    region_mask,
    sphericity,
    volume_mm3,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sphere_mask(shape: tuple[int, int, int], center: tuple[float, float, float], r: float):
    """Boolean array, True inside a Euclidean ball of radius `r` about `center`."""
    ii, jj, kk = np.indices(shape)
    dist = np.sqrt((ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2)
    return dist <= r


def _assert_nan_aware_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb):
            continue
        assert va == vb, f"key {key!r}: {va!r} != {vb!r}"


# --------------------------------------------------------------------------- #
# 1. Empty class map
# --------------------------------------------------------------------------- #


def test_empty_class_map_gives_zeros_and_nans_everywhere() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    geom = CaseGeometry()
    profile = burden_profile(classes, geom)

    for key in (
        "vol_NCR_mm3",
        "vol_ED_mm3",
        "vol_ET_mm3",
        "vol_TC_mm3",
        "vol_WT_mm3",
    ):
        assert profile[key] == 0.0

    for key in (
        "frac_enhancing_of_wt",
        "frac_necrotic_of_wt",
        "frac_edema_of_wt",
        "frac_enhancing_of_tc",
        "frac_necrotic_of_tc",
        "ratio_edema_to_core",
    ):
        assert math.isnan(profile[key])

    for region in REGION_ORDER:
        assert profile[f"n_components_{region}"] == 0
        assert math.isnan(profile[f"sphericity_{region}"])
        assert math.isnan(profile[f"centroid_i_{region}"])
        assert math.isnan(profile[f"centroid_j_{region}"])
        assert math.isnan(profile[f"centroid_k_{region}"])
        assert profile[f"dominant_side_{region}"] == ""


# --------------------------------------------------------------------------- #
# 2. Hand-computed volume
# --------------------------------------------------------------------------- #


def test_hand_computed_volume_isotropic_and_anisotropic_spacing() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[0:4, 0:5, 0:6] = 3  # 4*5*6 = 120 voxels, all ET

    volumes_iso = compute_volumes(classes, CaseGeometry())
    assert volumes_iso["vol_ET_mm3"] == 120.0

    volumes_aniso = compute_volumes(classes, CaseGeometry(spacing=(2.0, 1.0, 1.0)))
    assert volumes_aniso["vol_ET_mm3"] == 240.0


# --------------------------------------------------------------------------- #
# 3. Nesting identity
# --------------------------------------------------------------------------- #


def test_nesting_identity_tc_and_wt_are_exact_sums_of_classes() -> None:
    classes = np.zeros((12, 12, 12), dtype=np.uint8)
    classes[0:3, 0:3, 0:3] = 1  # NCR
    classes[4:8, 4:8, 4:8] = 2  # ED
    classes[9:11, 9:11, 9:11] = 3  # ET
    geom = CaseGeometry()

    volumes = compute_volumes(classes, geom)
    assert volumes["vol_TC_mm3"] == volumes["vol_NCR_mm3"] + volumes["vol_ET_mm3"]
    assert volumes["vol_WT_mm3"] == (
        volumes["vol_NCR_mm3"] + volumes["vol_ED_mm3"] + volumes["vol_ET_mm3"]
    )


# --------------------------------------------------------------------------- #
# 4. Hand-computed fractions
# --------------------------------------------------------------------------- #


def test_hand_computed_fractions() -> None:
    flat = np.zeros(600, dtype=np.uint8)
    flat[0:100] = 1  # NCR
    flat[100:400] = 2  # ED
    flat[400:500] = 3  # ET
    classes = flat.reshape(10, 10, 6)
    geom = CaseGeometry()

    volumes = compute_volumes(classes, geom)
    fractions = compute_fractions(volumes)

    assert fractions["frac_enhancing_of_wt"] == pytest.approx(0.2)
    assert fractions["frac_necrotic_of_tc"] == pytest.approx(0.5)
    assert fractions["ratio_edema_to_core"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# 5. Zero denominator gives NaN, never a raise, never 0.0
# --------------------------------------------------------------------------- #


def test_zero_denominator_gives_nan_not_zero() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[0:5, 0:5, 0:5] = 2  # ED only -> TC (NCR+ET) is 0
    geom = CaseGeometry()

    volumes = compute_volumes(classes, geom)
    fractions = compute_fractions(volumes)

    assert math.isnan(fractions["frac_necrotic_of_tc"])


# --------------------------------------------------------------------------- #
# 6 & 7. Sphericity of known shapes
# --------------------------------------------------------------------------- #


def test_sphericity_of_a_digital_sphere_is_near_one() -> None:
    mask = _sphere_mask((40, 40, 40), (19.5, 19.5, 19.5), 15.0)
    geom = CaseGeometry()
    psi = sphericity(mask, geom)

    # Measured (not guessed): raw marching cubes on a binary mask, with no
    # signed-distance sub-voxel surface localisation, carries a small but
    # real systematic overestimate of curved surface area -- roughly 8-10%
    # for a sphere, essentially flat across radius (see the module
    # docstring of `surface_area_mm2`). So a perfect digital sphere lands
    # around 0.92, not 1.00.
    assert 0.88 < psi < 0.96
    # The range still clearly rules out the face-counting failure mode: a
    # digitised sphere's face-counted surface area converges to ~1.5x the
    # true area, which would put sphericity near 0.67, not near 0.92.
    assert abs(psi - 0.67) > 0.2


def test_sphericity_of_a_large_cube() -> None:
    mask = np.zeros((34, 34, 34), dtype=bool)
    mask[2:32, 2:32, 2:32] = True  # side 30
    geom = CaseGeometry()
    psi = sphericity(mask, geom)

    # Analytic value for a cube: (pi / 6) ** (1 / 3) ~= 0.806.
    assert 0.76 < psi < 0.84


def test_surface_area_zero_pad_is_load_bearing() -> None:
    # A sphere well inside the array (margin > radius) vs. the same sphere
    # cropped tightly to its own bounding box, so the mask touches every
    # face of the array. Without the zero-pad in surface_area_mm2, the
    # touching case produces an open surface and understated area -- this
    # test fails if the pad is removed.
    geom = CaseGeometry()
    margin_mask = _sphere_mask((40, 40, 40), (19.5, 19.5, 19.5), 10.0)
    sph_margin = sphericity(margin_mask, geom)

    idxs = np.argwhere(margin_mask)
    lo = idxs.min(axis=0)
    hi = idxs.max(axis=0) + 1
    touching_mask = margin_mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    # Confirm it really does touch every face of its own array.
    assert touching_mask[0, :, :].any()
    assert touching_mask[-1, :, :].any()
    sph_touching = sphericity(touching_mask, geom)

    assert abs(sph_margin - sph_touching) < 0.02


# --------------------------------------------------------------------------- #
# 9. Connectivity semantics
# --------------------------------------------------------------------------- #


def test_connectivity_corner_touching_blocks() -> None:
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[0:2, 0:2, 0:2] = True
    mask[2:4, 2:4, 2:4] = True  # touches the first block only at one corner voxel pair
    geom = CaseGeometry()

    assert connected_components(mask, geom, min_volume_mm3=0.0, connectivity=3).n == 1
    assert connected_components(mask, geom, min_volume_mm3=0.0, connectivity=1).n == 2


def test_connectivity_gap_separated_blocks_agree_at_both_connectivities() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[0:2, 0:2, 0:2] = True
    mask[4:6, 4:6, 4:6] = True  # clear gap on every axis
    geom = CaseGeometry()

    assert connected_components(mask, geom, min_volume_mm3=0.0, connectivity=3).n == 2
    assert connected_components(mask, geom, min_volume_mm3=0.0, connectivity=1).n == 2


# --------------------------------------------------------------------------- #
# 10. Min-volume floor
# --------------------------------------------------------------------------- #


def test_min_volume_floor_drops_speck_but_keeps_it_in_total() -> None:
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[0:10, 0:10, 0:10] = True  # 1000 voxels = 1000 mm^3 at unit spacing
    mask[15:16, 15:16, 0:10] = True  # 10 voxels = 10 mm^3, far from the big block
    geom = CaseGeometry()

    stats = connected_components(mask, geom, min_volume_mm3=50.0, connectivity=3)
    assert stats.n == 1
    assert stats.total_volume_mm3 == 1010.0
    assert stats.largest_frac < 1.0


# --------------------------------------------------------------------------- #
# 11. Laterality is not left/right swapped
# --------------------------------------------------------------------------- #


def test_laterality_low_index_block_is_patient_right() -> None:
    # Axis 0 runs right -> left under the BraTS diag(-1, -1, 1) affine, so a
    # block at LOW axis-0 indices is on the patient's RIGHT.
    mask = np.zeros((40, 10, 10), dtype=bool)
    mask[0:10, :, :] = True

    geom_right_high = CaseGeometry(midline_index=20.0, left_is_high_index=True)
    lat = laterality(mask, geom_right_high)
    assert lat["vol_right_mm3"] > 0.0
    assert lat["vol_left_mm3"] == 0.0
    assert lat["dominant_side"] == "right"

    geom_left_high = CaseGeometry(midline_index=20.0, left_is_high_index=False)
    lat_flipped = laterality(mask, geom_left_high)
    assert lat_flipped["vol_left_mm3"] > 0.0
    assert lat_flipped["vol_right_mm3"] == 0.0
    assert lat_flipped["dominant_side"] == "left"


# --------------------------------------------------------------------------- #
# 12. Laterality on a straddling block
# --------------------------------------------------------------------------- #


def test_laterality_straddling_block_sums_to_whole_tumor_volume() -> None:
    mask = np.zeros((140, 10, 10), dtype=bool)
    mask[110:131, :, :] = True  # spans the default midline of 119.5
    geom = CaseGeometry()  # midline_index=119.5, left_is_high_index=True

    lat = laterality(mask, geom)
    assert lat["vol_right_mm3"] > 0.0
    assert lat["vol_left_mm3"] > 0.0
    assert 0.0 < lat["frac_contralateral"] <= 0.5

    wt_vol = volume_mm3(mask, geom)
    assert lat["vol_right_mm3"] + lat["vol_left_mm3"] == pytest.approx(wt_vol)


# --------------------------------------------------------------------------- #
# 13. Centroid
# --------------------------------------------------------------------------- #


def test_centroid_of_symmetric_block_is_its_geometric_center() -> None:
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:8, 3:9, 4:10] = True
    geom = CaseGeometry()

    c = centroid(mask, geom)
    assert c == pytest.approx((4.5, 5.5, 6.5))


# --------------------------------------------------------------------------- #
# 14. CaseGeometry.from_meta
# --------------------------------------------------------------------------- #


def test_case_geometry_from_meta_orientation_and_midline() -> None:
    base_meta = {
        "spacing": [1.0, 1.0, 1.0],
        "original_shape": [240, 240, 155],
    }

    meta_neg = {**base_meta, "affine": [[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]]}
    geom_neg = CaseGeometry.from_meta(meta_neg)
    assert geom_neg.left_is_high_index is True
    assert geom_neg.midline_index == pytest.approx(119.5)

    meta_pos = {**base_meta, "affine": [[1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]]}
    geom_pos = CaseGeometry.from_meta(meta_pos)
    assert geom_pos.left_is_high_index is False

    meta_zero = {**base_meta, "affine": [[0, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]]}
    with pytest.raises(ValueError, match="orientation"):
        CaseGeometry.from_meta(meta_zero)

    meta_cropped = {**meta_neg, "bbox": [[53, 187], [0, 240], [0, 155]]}
    geom_cropped = CaseGeometry.from_meta(meta_cropped, cropped=True)
    assert geom_cropped.midline_index == pytest.approx(66.5)


def test_case_geometry_from_meta_cropped_without_bbox_raises() -> None:
    meta = {
        "spacing": [1.0, 1.0, 1.0],
        "original_shape": [240, 240, 155],
        "affine": [[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    with pytest.raises(ValueError, match="bbox"):
        CaseGeometry.from_meta(meta, cropped=True)


# --------------------------------------------------------------------------- #
# 15. estimate_midline_index
# --------------------------------------------------------------------------- #


def test_estimate_midline_index_recovers_a_planted_plane() -> None:
    mask = _sphere_mask((200, 40, 40), (100.0, 20.0, 20.0), 15.0)
    m = estimate_midline_index(mask, search_radius=15.0)
    assert abs(m - 100.0) < 0.5


def test_estimate_midline_index_raises_on_empty_mask() -> None:
    mask = np.zeros((40, 40, 40), dtype=bool)
    with pytest.raises(ValueError):
        estimate_midline_index(mask)


# --------------------------------------------------------------------------- #
# 16. Input validation
# --------------------------------------------------------------------------- #


def test_validation_rejects_raw_brats_label_four() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.uint8)
    classes[0, 0, 0] = 4
    with pytest.raises(ValueError, match="raw, unremapped BraTS"):
        region_mask(classes, "ET")


def test_validation_rejects_non_3d_array() -> None:
    classes = np.zeros((10, 10, 10, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="D, H, W"):
        region_mask(classes, "ET")


def test_validation_rejects_float_array() -> None:
    classes = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="probabilities|logits"):
        region_mask(classes, "ET")


# --------------------------------------------------------------------------- #
# 17. burden_profile output is CSV-safe
# --------------------------------------------------------------------------- #


def _expected_burden_profile_keys() -> set[str]:
    keys = {
        "vol_NCR_mm3",
        "vol_ED_mm3",
        "vol_ET_mm3",
        "vol_TC_mm3",
        "vol_WT_mm3",
        "frac_enhancing_of_wt",
        "frac_necrotic_of_wt",
        "frac_edema_of_wt",
        "frac_enhancing_of_tc",
        "frac_necrotic_of_tc",
        "ratio_edema_to_core",
    }
    per_region_templates = [
        "n_components_{}",
        "vol_largest_component_{}_mm3",
        "vol_second_component_{}_mm3",
        "largest_component_frac_{}",
        "surface_area_{}_mm2",
        "sphericity_{}",
        "surface_to_volume_{}",
        "vol_right_{}_mm3",
        "vol_left_{}_mm3",
        "frac_left_{}",
        "frac_contralateral_{}",
        "dominant_side_{}",
        "centroid_i_{}",
        "centroid_j_{}",
        "centroid_k_{}",
    ]
    for region in REGION_ORDER:
        for template in per_region_templates:
            keys.add(template.format(region))
    return keys


def test_burden_profile_output_is_csv_safe_and_has_exact_key_set() -> None:
    classes = np.zeros((20, 20, 20), dtype=np.uint8)
    classes[0:3, 0:3, 0:3] = 1
    classes[3:8, 3:8, 3:8] = 2
    classes[8:10, 8:10, 8:10] = 3
    geom = CaseGeometry()

    profile = burden_profile(classes, geom)

    assert profile.keys() == _expected_burden_profile_keys()
    for key, value in profile.items():
        assert isinstance(value, (float, int, str)), f"{key} has type {type(value)}"
        assert not isinstance(value, (list, tuple, np.ndarray))
        assert value is not None


# --------------------------------------------------------------------------- #
# 18. Determinism
# --------------------------------------------------------------------------- #


def test_burden_profile_is_deterministic() -> None:
    classes = np.zeros((20, 20, 20), dtype=np.uint8)
    classes[0:3, 0:3, 0:3] = 1
    classes[3:8, 3:8, 3:8] = 2
    classes[8:10, 8:10, 8:10] = 3
    geom = CaseGeometry()

    profile_a = burden_profile(classes, geom)
    profile_b = burden_profile(classes, geom)
    _assert_nan_aware_equal(profile_a, profile_b)


# --------------------------------------------------------------------------- #
# 19. No torch
# --------------------------------------------------------------------------- #


def test_burden_module_does_not_import_torch() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly imported torch already for some other test
    file in the suite -- see `test_figures.py`'s equivalent guard.
    """
    source = Path(burden.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
