"""Tests for `neurovision.anatomy.involvement`.

Every test runs on CPU against a small hand-built synthetic `Atlas` (never
the real SRI24 atlas) and hand-written knowledge YAML fixtures written to
`tmp_path` (never the real committed `knowledge/involvement_groups.yaml`).
Each test is well under a second.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from neurovision.anatomy import involvement
from neurovision.anatomy.atlas import Atlas, AtlasLabels, AtlasStructure
from neurovision.anatomy.burden import CaseGeometry
from neurovision.anatomy.involvement import (
    epicentre,
    group_overlap,
    involvement_profile,
    load_involvement_groups,
    tissue_overlap,
)

# --------------------------------------------------------------------------- #
# Synthetic atlas fixture
# --------------------------------------------------------------------------- #
#
# Shape (20, 20, 10). Five structures, non-overlapping:
#   Vent_L:    [0:5, 0:5, 0:5]     -> 125 voxels, label 1, laterality L
#   Vent_R:    [5:10, 0:5, 0:5]    -> 125 voxels, label 2, laterality R
#   DeepWM:    [10:12, 10:12, 0:2] -> 8 voxels, label 3, midline (small)
#   Other:     [12:20, 0:8, 0:8]   -> 512 voxels, label 4, midline (large,
#              unrelated to any group -- pure filler mass)
#   Cortex_L:  [0:3, 15:18, 0:3]   -> 27 voxels, label 5, laterality L, but
#              placed at LOW axis-0 index deliberately -- test 11 uses this
#              to prove epicentre_side and epicentre_laterality are two
#              independent measurements that can disagree.
#
# Tissue map (uint8, codes CSF=1 / GM=2 / WM=3): a 10-voxel strip at
# [0, 0, 0:10] is hand-assigned 3 GM / 4 WM / 2 CSF / 1 outside(0), used by
# the tissue_overlap sum-to-one test. Everywhere else is 0 (outside).

_ATLAS_SHAPE = (20, 20, 10)


def _make_atlas() -> Atlas:
    parcellation = np.zeros(_ATLAS_SHAPE, dtype=np.int16)
    parcellation[0:5, 0:5, 0:5] = 1  # Vent_L, 125
    parcellation[5:10, 0:5, 0:5] = 2  # Vent_R, 125
    parcellation[10:12, 10:12, 0:2] = 3  # DeepWM, 8
    parcellation[12:20, 0:8, 0:8] = 4  # Other, 512
    parcellation[0:3, 15:18, 0:3] = 5  # Cortex_L, 27

    tissue = np.zeros(_ATLAS_SHAPE, dtype=np.uint8)
    tissue[0, 0, 0:3] = 2  # GM, 3 voxels
    tissue[0, 0, 3:7] = 3  # WM, 4 voxels
    tissue[0, 0, 7:9] = 1  # CSF, 2 voxels
    # tissue[0, 0, 9] stays 0 (outside), 1 voxel

    structures = (
        AtlasStructure(name="Vent_L", label_ids=(1,), laterality="L"),
        AtlasStructure(name="Vent_R", label_ids=(2,), laterality="R"),
        AtlasStructure(name="DeepWM", label_ids=(3,), laterality="midline"),
        AtlasStructure(name="Other", label_ids=(4,), laterality="midline"),
        AtlasStructure(name="Cortex_L", label_ids=(5,), laterality="L"),
    )
    labels = AtlasLabels(structures=structures, unmapped_name="unclassified")
    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=tissue,
        tissue_codes={"CSF": 1, "GM": 2, "WM": 3},
        name="synthetic",
        version="0",
        source="test",
        unmapped_ids=(),
    )


_TISSUE_NAMES = {"cortical": "GM", "white_matter": "WM", "csf": "CSF"}


def _geom(
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    midline_index: float = 9.5,
    left_is_high_index: bool = True,
) -> CaseGeometry:
    return CaseGeometry(
        spacing=spacing, midline_index=midline_index, left_is_high_index=left_is_high_index
    )


# --------------------------------------------------------------------------- #
# Knowledge YAML fixture
# --------------------------------------------------------------------------- #


def _write_involvement_yaml(
    path: Path,
    *,
    ventricle_structures: list[str],
    deep_wm_structures: list[str],
    tissue: dict[str, str] | None = None,
    search_radius_mm: float = 10.0,
) -> None:
    doc: dict[str, Any] = {
        "version": 1,
        "atlas": "synthetic test atlas",
        "relationship_to_vasari": {
            "status": "approximate_and_unverified",
            "claim": "Test claim sentence about VASARI approximation.  \n",
        },
        "groups": {
            "ventricles": {
                "structures": ventricle_structures,
                "missing": [{"term": "fourth ventricle", "reason": "not delineated"}],
            },
            "deep_white_matter": {
                "structures": deep_wm_structures,
                "missing": [],
            },
        },
        "tissue": tissue if tissue is not None else dict(_TISSUE_NAMES),
        "epicentre": {
            "definition": "test definition",
            "search_radius_mm": search_radius_mm,
            "side_from_midline": True,
        },
    }
    path.write_text(yaml.safe_dump(doc))


# --------------------------------------------------------------------------- #
# 1. A mask filling exactly one structure of a two-structure group
# --------------------------------------------------------------------------- #


def test_group_overlap_mask_fills_one_structure_of_a_two_structure_group() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = atlas.parcellation == 1  # exactly Vent_L, 125 voxels

    result = group_overlap(
        mask,
        atlas.parcellation,
        atlas,
        ("Vent_L", "Vent_R"),
        geom,
        min_overlap_mm3=1.0,
        prefix="ventricle",
    )

    assert result["ventricle_frac_of_tumour"] == pytest.approx(1.0)
    # Group is Vent_L (125) + Vent_R (125) = 250; mask covers only Vent_L.
    assert result["ventricle_frac_of_group"] == pytest.approx(125 / 250)
    assert result["ventricle_overlap_mm3"] == pytest.approx(125.0)


# --------------------------------------------------------------------------- #
# 2. A mask half in a group, half outside -- both fractions hand-computed
# --------------------------------------------------------------------------- #


def test_group_overlap_mask_half_in_group_half_outside() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[0:3, 0:5, 0:4] = True  # 60 voxels, entirely inside Vent_L (125)
    mask[15:20, 10:14, 8:10] = True  # 40 voxels, entirely outside every structure

    result = group_overlap(
        mask, atlas.parcellation, atlas, ("Vent_L",), geom, min_overlap_mm3=1.0, prefix="ventricle"
    )

    assert result["ventricle_frac_of_tumour"] == pytest.approx(60 / 100)
    assert result["ventricle_frac_of_group"] == pytest.approx(60 / 125)


# --------------------------------------------------------------------------- #
# 3. frac_of_tumour and frac_of_group diverge on a small-structure fixture
# --------------------------------------------------------------------------- #


def test_group_overlap_frac_of_tumour_and_frac_of_group_diverge() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[12:17, 0:4, 0:5] = True  # 100 voxels inside "Other" (unrelated to the group)
    mask[10:12, 10:12, 0:2] = True  # all 8 voxels of DeepWM

    result = group_overlap(
        mask, atlas.parcellation, atlas, ("DeepWM",), geom, min_overlap_mm3=1.0, prefix="deep_wm"
    )

    # Small contribution to the tumour, but the small structure is wholly consumed.
    assert result["deep_wm_frac_of_tumour"] == pytest.approx(8 / 108)
    assert result["deep_wm_frac_of_group"] == pytest.approx(1.0)
    assert result["deep_wm_frac_of_tumour"] < result["deep_wm_frac_of_group"]


# --------------------------------------------------------------------------- #
# 4. contact flips exactly at min_overlap_mm3
# --------------------------------------------------------------------------- #


def test_group_overlap_contact_flips_at_threshold() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = atlas.parcellation == 3  # exactly DeepWM, 8 voxels -> overlap_mm3 = 8.0

    at_threshold = group_overlap(
        mask, atlas.parcellation, atlas, ("DeepWM",), geom, min_overlap_mm3=8.0, prefix="deep_wm"
    )
    above_threshold = group_overlap(
        mask, atlas.parcellation, atlas, ("DeepWM",), geom, min_overlap_mm3=8.5, prefix="deep_wm"
    )

    assert at_threshold["deep_wm_contact"] is True
    assert above_threshold["deep_wm_contact"] is False


# --------------------------------------------------------------------------- #
# 5. Empty mask -> NaN fractions, 0.0 overlap, False contact, unlabelled epicentre
# --------------------------------------------------------------------------- #


def test_empty_mask_gives_nan_fractions_and_never_raises() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)

    overlap = group_overlap(
        mask, atlas.parcellation, atlas, ("Vent_L",), geom, min_overlap_mm3=1.0, prefix="ventricle"
    )
    assert math.isnan(overlap["ventricle_frac_of_tumour"])
    assert math.isnan(overlap["ventricle_frac_of_group"])
    assert overlap["ventricle_overlap_mm3"] == 0.0
    assert overlap["ventricle_contact"] is False

    epi = epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=10.0)
    assert epi["epicentre_structure"] == "unlabelled"
    assert epi["epicentre_exact"] is False
    assert math.isnan(epi["epicentre_distance_mm"])
    assert epi["epicentre_laterality"] == ""
    assert epi["epicentre_side"] == ""
    assert epi["epicentre_lobe"] == ""


# --------------------------------------------------------------------------- #
# 6. Tissue fractions sum to 1.0
# --------------------------------------------------------------------------- #


def test_tissue_overlap_fractions_sum_to_one() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[0, 0, 0:10] = True  # matches the hand-assigned tissue strip: 3 GM/4 WM/2 CSF/1 outside

    result = tissue_overlap(mask, atlas.tissue, atlas, _TISSUE_NAMES, geom)

    assert result["cortical_frac_of_tumour"] == pytest.approx(3 / 10)
    assert result["white_matter_frac_of_tumour"] == pytest.approx(4 / 10)
    assert result["csf_frac_of_tumour"] == pytest.approx(2 / 10)
    assert result["outside_tissue_frac_of_tumour"] == pytest.approx(1 / 10)
    total = (
        result["cortical_frac_of_tumour"]
        + result["white_matter_frac_of_tumour"]
        + result["csf_frac_of_tumour"]
        + result["outside_tissue_frac_of_tumour"]
    )
    assert math.isclose(total, 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# 7. tissue is None -> all NaN, no raise
# --------------------------------------------------------------------------- #


def test_tissue_overlap_with_no_tissue_map_gives_all_nan() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = atlas.parcellation == 1

    result = tissue_overlap(mask, None, atlas, _TISSUE_NAMES, geom)

    assert set(result) == {
        "cortical_frac_of_tumour",
        "white_matter_frac_of_tumour",
        "csf_frac_of_tumour",
        "outside_tissue_frac_of_tumour",
    }
    assert all(math.isnan(v) for v in result.values())


# --------------------------------------------------------------------------- #
# 8. epicentre_exact True for a mask centred inside a labelled structure
# --------------------------------------------------------------------------- #


def test_epicentre_exact_when_centroid_lands_on_a_labelled_voxel() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[2:4, 2:4, 2:4] = True  # 8 voxels, entirely inside Vent_L [0:5,0:5,0:5]

    result = epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=10.0)

    assert result["epicentre_exact"] is True
    assert result["epicentre_structure"] == "Vent_L"
    assert result["epicentre_distance_mm"] == 0.0


# --------------------------------------------------------------------------- #
# 9. epicentre_exact False, correct nearest structure, hand-computed distance
# --------------------------------------------------------------------------- #


def test_epicentre_inexact_finds_nearest_structure_with_anisotropic_spacing() -> None:
    atlas = _make_atlas()
    geom = _geom(spacing=(2.0, 1.0, 1.0))
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[12, 10, 0] = True  # single unlabelled voxel, 1 voxel past DeepWM along axis 0

    result = epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=10.0)

    assert result["epicentre_exact"] is False
    assert result["epicentre_structure"] == "DeepWM"
    # 1 voxel of axis-0 spacing 2.0mm to the nearest DeepWM voxel at (11, 10, 0).
    assert result["epicentre_distance_mm"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 10. Nothing labelled within search_radius_mm -> unlabelled, NaN distance
# --------------------------------------------------------------------------- #


def test_epicentre_beyond_search_radius_gives_unlabelled_and_nan() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[19, 19, 9] = True  # far corner, well clear of every structure

    result = epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=3.0)

    assert result["epicentre_exact"] is False
    assert result["epicentre_structure"] == "unlabelled"
    assert math.isnan(result["epicentre_distance_mm"])


# --------------------------------------------------------------------------- #
# 11. epicentre_side disagrees with epicentre_laterality, on purpose
# --------------------------------------------------------------------------- #


def test_epicentre_side_and_laterality_are_independent_and_can_disagree() -> None:
    atlas = _make_atlas()
    # midline_index=9.5, left_is_high_index=True -> axis-0 index 1 (Cortex_L's
    # location) is LOW, hence geometric "right" under this project's
    # convention -- but the structure is named (and merged-labelled) "_L".
    geom = _geom(midline_index=9.5, left_is_high_index=True)
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[1, 16, 1] = True  # inside Cortex_L's [0:3, 15:18, 0:3]

    result = epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=10.0)

    assert result["epicentre_structure"] == "Cortex_L"
    assert result["epicentre_laterality"] == "L"
    assert result["epicentre_side"] == "right"
    # Neither field was silently overwritten to agree with the other.
    assert result["epicentre_laterality"] != result["epicentre_side"]


# --------------------------------------------------------------------------- #
# 12. Anisotropic spacing scales group-overlap volume correctly
# --------------------------------------------------------------------------- #


def test_group_overlap_volume_scales_with_anisotropic_spacing() -> None:
    atlas = _make_atlas()
    geom = _geom(spacing=(2.0, 1.0, 1.0))
    mask = atlas.parcellation == 3  # DeepWM, 8 voxels

    result = group_overlap(
        mask, atlas.parcellation, atlas, ("DeepWM",), geom, min_overlap_mm3=1.0, prefix="deep_wm"
    )

    # voxel volume = 2.0 * 1.0 * 1.0 = 2.0 mm^3, not the isotropic 1.0.
    assert result["deep_wm_overlap_mm3"] == pytest.approx(8 * 2.0)


# --------------------------------------------------------------------------- #
# 13. Shape mismatches raise, naming both shapes
# --------------------------------------------------------------------------- #


def test_group_overlap_raises_on_mask_parcellation_shape_mismatch() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros((5, 5, 5), dtype=bool)  # wrong shape entirely

    with pytest.raises(ValueError, match=r"\(5, 5, 5\)"):
        group_overlap(
            mask,
            atlas.parcellation,
            atlas,
            ("Vent_L",),
            geom,
            min_overlap_mm3=1.0,
            prefix="ventricle",
        )


def test_epicentre_raises_on_mask_parcellation_shape_mismatch() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match=str(_ATLAS_SHAPE)):
        epicentre(mask, atlas.parcellation, atlas, geom, search_radius_mm=10.0)


def test_tissue_overlap_raises_on_mask_tissue_shape_mismatch() -> None:
    atlas = _make_atlas()
    geom = _geom()
    mask = np.zeros((5, 5, 5), dtype=bool)

    with pytest.raises(ValueError, match=r"\(5, 5, 5\)"):
        tissue_overlap(mask, atlas.tissue, atlas, _TISSUE_NAMES, geom)


# --------------------------------------------------------------------------- #
# 14. load_involvement_groups validation
# --------------------------------------------------------------------------- #


def test_load_involvement_groups_raises_on_unknown_structure_and_names_it(
    tmp_path: Path,
) -> None:
    atlas = _make_atlas()
    path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        path, ventricle_structures=["NotARealStructure"], deep_wm_structures=["DeepWM"]
    )

    with pytest.raises(ValueError, match="NotARealStructure"):
        load_involvement_groups(path, atlas)


def test_load_involvement_groups_raises_on_unknown_tissue_and_names_it(tmp_path: Path) -> None:
    atlas = _make_atlas()
    path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        path,
        ventricle_structures=["Vent_L", "Vent_R"],
        deep_wm_structures=["DeepWM"],
        tissue={"cortical": "NOPE", "white_matter": "WM", "csf": "CSF"},
    )

    with pytest.raises(ValueError, match="NOPE"):
        load_involvement_groups(path, atlas)


def test_load_involvement_groups_reads_every_field(tmp_path: Path) -> None:
    atlas = _make_atlas()
    path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        path,
        ventricle_structures=["Vent_L", "Vent_R"],
        deep_wm_structures=["DeepWM"],
        search_radius_mm=12.5,
    )

    groups = load_involvement_groups(path, atlas)

    assert groups.version == 1
    assert groups.ventricle_structures == ("Vent_L", "Vent_R")
    assert groups.deep_wm_structures == ("DeepWM",)
    assert groups.ventricle_missing == ("fourth ventricle",)
    assert groups.deep_wm_missing == ()
    assert groups.tissue_names == _TISSUE_NAMES
    assert groups.epicentre_search_radius_mm == pytest.approx(12.5)
    assert groups.vasari_status == "approximate_and_unverified"
    assert groups.vasari_claim.startswith("Test claim sentence")
    assert not groups.vasari_claim.endswith("\n")


# --------------------------------------------------------------------------- #
# 15. involvement_profile merges with no duplicate keys
# --------------------------------------------------------------------------- #


def test_involvement_profile_has_no_duplicate_keys(tmp_path: Path) -> None:
    atlas = _make_atlas()
    geom = _geom()
    path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        path, ventricle_structures=["Vent_L", "Vent_R"], deep_wm_structures=["DeepWM"]
    )
    groups = load_involvement_groups(path, atlas)

    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[2:4, 2:4, 2:4] = True  # inside Vent_L

    profile = involvement_profile(
        mask, atlas.parcellation, atlas.tissue, atlas, groups, geom, min_overlap_mm3=1.0
    )

    expected_keys = {
        "ventricle_overlap_mm3",
        "ventricle_frac_of_tumour",
        "ventricle_frac_of_group",
        "ventricle_contact",
        "deep_wm_overlap_mm3",
        "deep_wm_frac_of_tumour",
        "deep_wm_frac_of_group",
        "deep_wm_contact",
        "cortical_frac_of_tumour",
        "white_matter_frac_of_tumour",
        "csf_frac_of_tumour",
        "outside_tissue_frac_of_tumour",
        "epicentre_structure",
        "epicentre_exact",
        "epicentre_distance_mm",
        "epicentre_laterality",
        "epicentre_side",
        "epicentre_lobe",
    }
    assert set(profile) == expected_keys
    assert len(profile) == 18


# --------------------------------------------------------------------------- #
# 16. No dependency on the deep-learning stack
# --------------------------------------------------------------------------- #


def test_involvement_module_does_not_import_the_deep_learning_stack() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly already imported that stack for some other
    test file in the suite -- see `test_localize.py`'s equivalent guard.
    """
    source = Path(involvement.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
