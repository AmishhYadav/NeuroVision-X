"""Tests for `neurovision.anatomy.alignment`.

Every test runs on CPU on small synthetic arrays (well under a second each)
and never touches the real SRI24 atlas or real BraTS data. The synthetic
`Atlas` is built directly as the frozen dataclass, exactly as
`src/neurovision/anatomy/atlas.py` defines it.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neurovision.anatomy import alignment
from neurovision.anatomy.alignment import (
    AlignmentReport,
    atlas_brain_mask,
    brain_mask_check,
    dice,
    laterality_check,
    load_lobe_map,
    lobe_distribution_check,
    run_checks,
    uncrop,
)
from neurovision.anatomy.atlas import Atlas, AtlasLabels, AtlasStructure

REPO_ROOT = Path(__file__).resolve().parents[1]

SHAPE = (40, 12, 12)  # (D, H, W) -- axis 0 is left-right, axis 1 is A-P.
MIDLINE = 19.5


# --------------------------------------------------------------------------- #
# Synthetic atlas builders
# --------------------------------------------------------------------------- #


def _make_labels() -> AtlasLabels:
    structures = (
        AtlasStructure(name="Foo_L", label_ids=(10,), laterality="L"),
        AtlasStructure(name="Foo_R", label_ids=(11,), laterality="R"),
        # Tiny pair -- below the 50-voxel floor, must be skipped.
        AtlasStructure(name="Bar_L", label_ids=(20,), laterality="L"),
        AtlasStructure(name="Bar_R", label_ids=(21,), laterality="R"),
        # Midline, infratentorial-style structure for the lobe check.
        AtlasStructure(name="Cereb", label_ids=(30,), laterality="midline"),
    )
    return AtlasLabels(structures=structures, unmapped_name="unclassified")


def _make_parcellation() -> np.ndarray:
    parcellation = np.zeros(SHAPE, dtype=np.int16)
    # Foo_L: high axis-0 indices -> patient LEFT (centroid 29.5).
    parcellation[25:35, :, :] = 10
    # Foo_R: low axis-0 indices -> patient RIGHT (centroid 9.5).
    parcellation[5:15, :, :] = 11
    # Bar_L / Bar_R: 1 voxel each, below the 50-voxel floor.
    parcellation[39:40, 0:1, 0:1] = 20
    parcellation[0:1, 0:1, 0:1] = 21
    # Cereb: near the midline, 2 * 12 * 12 = 288 voxels.
    parcellation[18:20, :, :] = 30
    return parcellation


def _make_tissue() -> np.ndarray:
    # Nonzero (GM) across ALL of axis 0 (so an axis-0 / L-R flip is a no-op --
    # this is the point of test 8) but restricted to j in [0, 4) of axis 1
    # (so an axis-1 / A-P flip is NOT a no-op -- the point of tests 4/5).
    tissue = np.zeros(SHAPE, dtype=np.uint8)
    tissue[:, 0:4, :] = 2  # GM
    return tissue


def make_atlas() -> Atlas:
    """The 'correct' synthetic atlas."""
    return Atlas(
        parcellation=_make_parcellation(),
        labels=_make_labels(),
        tissue=_make_tissue(),
        tissue_codes={"CSF": 1, "GM": 2, "WM": 3},
        name="synthetic",
        version="test",
        source="test",
        unmapped_ids=(),
    )


def make_lr_mirrored_atlas() -> Atlas:
    """An atlas whose PARCELLATION (and tissue, trivially, see `_make_tissue`) is L-R mirrored."""
    correct = make_atlas()
    mirrored_parcellation = np.ascontiguousarray(np.flip(correct.parcellation, axis=0))
    mirrored_tissue = np.ascontiguousarray(np.flip(correct.tissue, axis=0))
    return Atlas(
        parcellation=mirrored_parcellation,
        labels=correct.labels,
        tissue=mirrored_tissue,
        tissue_codes=correct.tissue_codes,
        name="synthetic-lr-mirrored",
        version="test",
        source="test",
        unmapped_ids=(),
    )


def make_no_pairs_atlas() -> Atlas:
    """An atlas with only midline structures -- no `_L`/`_R` pair at all."""
    structures = (AtlasStructure(name="Brainstem", label_ids=(1,), laterality="midline"),)
    labels = AtlasLabels(structures=structures, unmapped_name="unclassified")
    parcellation = np.zeros(SHAPE, dtype=np.int16)
    parcellation[15:25, :, :] = 1
    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=_make_tissue(),
        tissue_codes={"CSF": 1, "GM": 2, "WM": 3},
        name="synthetic-no-pairs",
        version="test",
        source="test",
        unmapped_ids=(),
    )


_LOBE_MAP = {
    "Foo": {"lobe": "frontal", "epidemiology_lobe": "frontal"},
    "Cereb": {"lobe": "cerebellum", "epidemiology_lobe": "excluded"},
}

_VALIDATION_CFG = SimpleNamespace(
    min_brain_dice=0.85,
    brain_mask_source="tissue",
    min_laterality_pairs_correct=1.0,
    midline_index=MIDLINE,
    max_midline_deviation=1.5,
)


# --------------------------------------------------------------------------- #
# 1. dice
# --------------------------------------------------------------------------- #


def test_dice_identical_disjoint_and_both_empty() -> None:
    a = np.zeros((10, 10, 10), dtype=bool)
    a[0:5, 0:5, 0:5] = True
    b = np.zeros((10, 10, 10), dtype=bool)
    b[5:10, 5:10, 5:10] = True

    assert dice(a, a) == pytest.approx(1.0)
    assert dice(a, b) == pytest.approx(0.0)

    empty = np.zeros((10, 10, 10), dtype=bool)
    result = dice(empty, empty)
    assert isinstance(result, float)
    assert math.isnan(result)
    assert result != 1.0  # explicit: NaN, never 1.0


# --------------------------------------------------------------------------- #
# 2. uncrop
# --------------------------------------------------------------------------- #


def test_uncrop_round_trips() -> None:
    original = np.zeros((10, 10, 10), dtype=np.uint8)
    original[3:7, 2:5, 1:9] = 1
    bbox = [[3, 7], [2, 5], [1, 9]]
    cropped = original[3:7, 2:5, 1:9]

    result = uncrop(cropped, bbox, original.shape)
    assert result.shape == original.shape
    assert result.dtype == cropped.dtype
    np.testing.assert_array_equal(result, original)


def test_uncrop_raises_on_shape_mismatch() -> None:
    wrong = np.zeros((3, 3, 3), dtype=np.uint8)
    bbox = [[3, 7], [2, 5], [1, 9]]  # extents (4, 3, 8) != (3, 3, 3)
    with pytest.raises(ValueError, match="does not match"):
        uncrop(wrong, bbox, (10, 10, 10))


# --------------------------------------------------------------------------- #
# 3. atlas_brain_mask
# --------------------------------------------------------------------------- #


def test_atlas_brain_mask_source_validation() -> None:
    atlas = make_atlas()

    mask = atlas_brain_mask(atlas, source="tissue")
    np.testing.assert_array_equal(mask, atlas.tissue > 0)

    with pytest.raises(ValueError, match="parcellation"):
        atlas_brain_mask(atlas, source="parcellation")

    with pytest.raises(ValueError, match="unsupported source"):
        atlas_brain_mask(atlas, source="spgr")

    no_tissue_atlas = Atlas(
        parcellation=atlas.parcellation,
        labels=atlas.labels,
        tissue=None,
        tissue_codes=atlas.tissue_codes,
        name="no-tissue",
        version="test",
        source="test",
        unmapped_ids=(),
    )
    with pytest.raises(ValueError, match="tissue is None"):
        atlas_brain_mask(no_tissue_atlas, source="tissue")


# --------------------------------------------------------------------------- #
# 4 & 5. brain_mask_check
# --------------------------------------------------------------------------- #


def test_brain_mask_check_passes_on_matching_and_fails_on_ap_mirrored() -> None:
    atlas = make_atlas()
    correct_mask = atlas_brain_mask(atlas)

    check_pass, df_pass = brain_mask_check(atlas, [("case_a", correct_mask.copy())], min_dice=0.85)
    assert check_pass.passed is True
    assert check_pass.value == pytest.approx(1.0)
    assert df_pass.loc[0, "case_id"] == "case_a"

    # A-P mirror: flip axis 1. The tissue mask lives only in j in [0, 4) of a
    # 12-wide axis, so this is a genuine, large mismatch -- not a no-op.
    ap_mirrored_mask = np.flip(correct_mask, axis=1)
    check_fail, _df_fail = brain_mask_check(atlas, [("case_b", ap_mirrored_mask)], min_dice=0.85)
    assert check_fail.passed is False
    assert check_fail.value < 0.85


def test_brain_mask_check_detail_reports_the_tail() -> None:
    atlas = make_atlas()
    correct_mask = atlas_brain_mask(atlas)
    bad_mask = np.flip(correct_mask, axis=1)

    cases = [(f"good_{i}", correct_mask.copy()) for i in range(4)] + [("bad_0", bad_mask)]
    check, df = brain_mask_check(atlas, cases, min_dice=0.85)

    # Median of [1, 1, 1, 1, 0] is 1.0 -- still passes.
    assert check.passed is True
    assert check.value == pytest.approx(1.0)
    assert len(df) == 5
    # But the tail is visible in the detail text, not hidden by the median.
    assert "1 case(s) below threshold" in check.detail


# --------------------------------------------------------------------------- #
# 6, 7 & 8. laterality_check
# --------------------------------------------------------------------------- #


def test_laterality_check_passes_on_correct_atlas() -> None:
    atlas = make_atlas()
    (pairs_check, midline_check), df = laterality_check(
        atlas, midline_index=MIDLINE, min_fraction_correct=1.0, max_midline_deviation=1.5
    )
    assert pairs_check.passed is True
    assert pairs_check.value == pytest.approx(1.0)
    assert midline_check.passed is True
    assert midline_check.value == pytest.approx(MIDLINE)
    assert set(df["base"]) == {"Foo"}


def test_laterality_check_fails_on_lr_mirrored_atlas() -> None:
    # The single most important test in this file: the only check in the
    # project that can catch a left/right swap.
    mirrored = make_lr_mirrored_atlas()
    (pairs_check, _midline_check), df = laterality_check(
        mirrored, midline_index=MIDLINE, min_fraction_correct=1.0, max_midline_deviation=1.5
    )
    assert pairs_check.passed is False
    assert pairs_check.value == pytest.approx(0.0)
    assert bool(df.loc[df["base"] == "Foo", "ok"].iloc[0]) is False


def test_brain_mask_dice_is_blind_to_the_same_lr_flip() -> None:
    # Same mirrored atlas as the test above: brain-mask Dice PASSES anyway,
    # because a brain is nearly left-right symmetric and the synthetic
    # tissue mask here is deliberately axis-0-symmetric (spans all of axis
    # 0). This is exactly why the laterality check exists as a SEPARATE,
    # independent gate -- do not "simplify" the two checks into one; brain
    # Dice alone would pass a left-right swapped atlas straight through.
    correct = make_atlas()
    mirrored = make_lr_mirrored_atlas()
    correct_case_mask = atlas_brain_mask(correct)

    check, _df = brain_mask_check(mirrored, [("case_a", correct_case_mask)], min_dice=0.85)
    assert check.passed is True
    assert check.value == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 9. midline_estimate
# --------------------------------------------------------------------------- #


def test_midline_estimate_recovers_planted_midline_and_fails_when_off_centre() -> None:
    atlas = make_atlas()

    (_pairs_check, midline_check), _df = laterality_check(
        atlas, midline_index=MIDLINE, min_fraction_correct=1.0, max_midline_deviation=1.5
    )
    assert midline_check.passed is True
    assert abs(midline_check.value - MIDLINE) < 1.5

    (_pairs_check2, midline_check_off), _df2 = laterality_check(
        atlas, midline_index=25.0, min_fraction_correct=1.0, max_midline_deviation=1.5
    )
    assert midline_check_off.passed is False
    # The estimate itself is unchanged -- only the assumed midline moved.
    assert midline_check_off.value == pytest.approx(MIDLINE)


# --------------------------------------------------------------------------- #
# 10. No _L/_R pairs raises
# --------------------------------------------------------------------------- #


def test_laterality_check_raises_when_no_pairs_exist() -> None:
    atlas = make_no_pairs_atlas()
    with pytest.raises(ValueError, match="_L/_R"):
        laterality_check(
            atlas, midline_index=MIDLINE, min_fraction_correct=1.0, max_midline_deviation=1.5
        )


# --------------------------------------------------------------------------- #
# 11. Structures below 50 voxels are skipped
# --------------------------------------------------------------------------- #


def test_small_structures_are_skipped_and_absent_from_laterality_pairs() -> None:
    atlas = make_atlas()
    (_pairs_check, _midline_check), df = laterality_check(
        atlas, midline_index=MIDLINE, min_fraction_correct=1.0, max_midline_deviation=1.5
    )
    assert "Bar" not in set(df["base"])
    assert "Foo" in set(df["base"])


# --------------------------------------------------------------------------- #
# 12. load_lobe_map
# --------------------------------------------------------------------------- #


def test_load_lobe_map_accepts_the_real_repo_file() -> None:
    lobe_map = load_lobe_map(REPO_ROOT / "knowledge" / "aal_lobes.yaml")
    assert "Frontal_Sup" in lobe_map
    assert lobe_map["Frontal_Sup"]["lobe"] == "frontal"
    assert lobe_map["Frontal_Sup"]["epidemiology_lobe"] == "frontal"
    assert lobe_map["Insula"]["epidemiology_lobe"] == "deep"


def test_load_lobe_map_raises_on_out_of_vocabulary_lobe(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad_lobes.yaml"
    bad_yaml.write_text("""
lobes: [frontal, temporal]
epidemiology_lobes: [frontal, temporal, excluded]
structures:
  Foo: {lobe: not_a_real_lobe, epidemiology_lobe: frontal}
""")
    with pytest.raises(ValueError, match="outside the declared vocabulary"):
        load_lobe_map(bad_yaml)


# --------------------------------------------------------------------------- #
# 13, 14 & 15. lobe_distribution_check
# --------------------------------------------------------------------------- #


def test_lobe_distribution_check_never_gates_even_with_terrible_distribution() -> None:
    atlas = make_atlas()
    foo_only_mask = atlas.parcellation == 10  # Foo_L only -> 100% frontal.
    cases = [(f"case_{i}", foo_only_mask.copy()) for i in range(3)]

    check, _df = lobe_distribution_check(atlas, _LOBE_MAP, cases)
    # Every case attributed to frontal (100%) against a 40% reference is a
    # terrible match -- and the check must still report passed=True.
    assert check.gating is False
    assert check.passed is True
    assert check.value > 0.0  # a real, nonzero deviation was measured

    report = AlignmentReport(
        checks=(check,), per_case_dice=_df, laterality_pairs=_df, lobe_distribution=_df
    )
    assert report.passed is True  # advisory-only report has nothing to fail on


def test_lobe_distribution_check_raises_on_structure_missing_from_map() -> None:
    atlas = make_atlas()
    foo_mask = atlas.parcellation == 10
    lobe_map_missing_foo = {"Cereb": {"lobe": "cerebellum", "epidemiology_lobe": "excluded"}}

    with pytest.raises(ValueError, match="no entry in lobe_map"):
        lobe_distribution_check(atlas, lobe_map_missing_foo, [("case_a", foo_mask)])


def test_lobe_distribution_check_excludes_excluded_lobes_from_tally() -> None:
    atlas = make_atlas()
    cereb_only_mask = atlas.parcellation == 30  # excluded epidemiology_lobe
    foo_only_mask = atlas.parcellation == 10  # frontal

    cases = [("excluded_case", cereb_only_mask), ("frontal_case", foo_only_mask)]
    check, df = lobe_distribution_check(atlas, _LOBE_MAP, cases, reference_pct={"frontal": 100.0})

    row = df.loc[df["lobe"] == "frontal"].iloc[0]
    # Only the frontal case was attributed; the excluded case contributed
    # nothing to any lobe's tally.
    assert row["n_cases"] == 1
    assert row["pct"] == pytest.approx(100.0)
    assert check.value == pytest.approx(0.0)
    assert "1/2" in check.detail


# --------------------------------------------------------------------------- #
# 16. AlignmentReport.passed / failures()
# --------------------------------------------------------------------------- #


def test_alignment_report_passed_false_when_a_gating_check_fails() -> None:
    mirrored = make_lr_mirrored_atlas()
    correct_case_mask = atlas_brain_mask(make_atlas())
    foo_mask = mirrored.parcellation == 10

    report = run_checks(
        atlas=mirrored,
        case_brain_masks=[("case_a", correct_case_mask)],
        case_tumour_masks=[("case_a", foo_mask)],
        cfg=_VALIDATION_CFG,
        lobe_map=_LOBE_MAP,
    )

    assert report.passed is False
    failure_names = {c.name for c in report.failures()}
    assert "laterality_pairs" in failure_names
    # Brain-mask Dice passed (it is blind to the L-R flip) and is therefore
    # not among the failures.
    assert "brain_mask_dice" not in failure_names
    # The advisory lobe check can never appear in failures(), regardless of
    # how bad its number is.
    assert "lobe_distribution" not in failure_names
    assert "FAIL" in report.summary()


def test_alignment_report_passed_true_when_everything_passes() -> None:
    atlas = make_atlas()
    correct_case_mask = atlas_brain_mask(atlas)
    foo_mask = atlas.parcellation == 10

    report = run_checks(
        atlas=atlas,
        case_brain_masks=[("case_a", correct_case_mask)],
        case_tumour_masks=[("case_a", foo_mask)],
        cfg=_VALIDATION_CFG,
        lobe_map=_LOBE_MAP,
    )
    assert report.passed is True
    assert report.failures() == ()
    assert "PASS" in report.summary()


# --------------------------------------------------------------------------- #
# 17. No torch
# --------------------------------------------------------------------------- #


def test_alignment_module_does_not_import_torch() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly imported torch already for some other test
    file in the suite -- see `test_burden.py`'s equivalent guard.
    """
    source = Path(alignment.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
