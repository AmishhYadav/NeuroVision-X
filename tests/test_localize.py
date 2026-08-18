"""Tests for `neurovision.anatomy.localize`.

Every test runs on CPU against a small hand-built synthetic `Atlas` (never
the real SRI24 atlas) and small hand-written knowledge YAML fixtures written
to `tmp_path` (never the real committed `knowledge/` files). Each test is
well under a second.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from neurovision.anatomy import localize
from neurovision.anatomy.atlas import Atlas, AtlasLabels, AtlasStructure
from neurovision.anatomy.localize import (
    atlas_for_case,
    distance_to_eloquent,
    eloquent_union_mask,
    load_classification,
    load_knowledge,
    localize_case,
    localize_mask,
    summarize_case,
)

_LOCALIZE_COLUMNS = (
    "structure",
    "laterality",
    "lobe",
    "eloquence",
    "matched_term",
    "n_voxels",
    "volume_mm3",
    "frac_of_tumour",
    "frac_of_structure",
)


# --------------------------------------------------------------------------- #
# Synthetic atlas fixture
# --------------------------------------------------------------------------- #
#
# Shape (20, 20, 10). Four structures, non-overlapping, laid out so their
# voxel counts are easy to reason about by hand:
#   StructA_L: [0:5, 0:5, 0:5]     -> 125 voxels, label 1
#   StructA_R: [5:10, 0:5, 0:5]    -> 125 voxels, label 2
#   Brainstem: [10:12, 10:12, 0:2] -> 8 voxels, label 3 (small, midline)
#   StructB_L: [15:20, 15:20, 5:10]-> 125 voxels, label 4
# Everything else is background (0).

_ATLAS_SHAPE = (20, 20, 10)


def _make_atlas() -> Atlas:
    parcellation = np.zeros(_ATLAS_SHAPE, dtype=np.int16)
    parcellation[0:5, 0:5, 0:5] = 1
    parcellation[5:10, 0:5, 0:5] = 2
    parcellation[10:12, 10:12, 0:2] = 3
    parcellation[15:20, 15:20, 5:10] = 4

    structures = (
        AtlasStructure(name="StructA_L", label_ids=(1,), laterality="L"),
        AtlasStructure(name="StructA_R", label_ids=(2,), laterality="R"),
        AtlasStructure(name="Brainstem", label_ids=(3,), laterality="midline"),
        AtlasStructure(name="StructB_L", label_ids=(4,), laterality="L"),
    )
    labels = AtlasLabels(structures=structures, unmapped_name="unclassified")
    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=None,
        tissue_codes={},
        name="synthetic",
        version="0",
        source="test",
        unmapped_ids=(),
    )


def _make_small_atlas() -> Atlas:
    """A second, smaller atlas (10, 10, 6) with one structure, for the crop-geometry test."""
    parcellation = np.zeros((10, 10, 6), dtype=np.int16)
    parcellation[2:5, 2:5, 1:4] = 1  # StructA_L, 27 voxels
    structures = (AtlasStructure(name="StructA_L", label_ids=(1,), laterality="L"),)
    labels = AtlasLabels(structures=structures, unmapped_name="unclassified")
    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=None,
        tissue_codes={},
        name="synthetic-small",
        version="0",
        source="test",
        unmapped_ids=(),
    )


# --------------------------------------------------------------------------- #
# Knowledge YAML fixtures
# --------------------------------------------------------------------------- #


def _write_eloquence_yaml(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    coverage_gaps: list[dict[str, str]] | None = None,
    vocabulary: list[str] | None = None,
    default: str = "unclassified",
    distance_mm: float = 10.0,
) -> None:
    doc = {
        "version": 1,
        "classification": {
            "name": "Test eloquence grading",
            "primary_citation": "Test R. A test classification. Test Journal. 1998.",
            "eloquent_structures_verbatim": "Eloquent locations are the motor and speech areas.",
            "read_via": "Secondary Source S. Open access review. 2013.",
        },
        "vocabulary": vocabulary or ["eloquent", "unclassified"],
        "default": default,
        "near_eloquent_rule": {"distance_mm": distance_mm},
        "coverage_gaps": coverage_gaps or [],
        "entries": entries,
    }
    path.write_text(yaml.safe_dump(doc))


def _write_lobe_yaml(path: Path, structures: dict[str, dict[str, str]]) -> None:
    path.write_text(yaml.safe_dump({"version": 1, "structures": structures}))


_FULL_LOBE_MAP = {
    "StructA": {"lobe": "frontal"},
    "StructB": {"lobe": "parietal"},
    "Brainstem": {"lobe": "brainstem"},
}


# --------------------------------------------------------------------------- #
# 1. A mask filling exactly one structure
# --------------------------------------------------------------------------- #


def test_mask_filling_one_structure_gives_frac_of_structure_one() -> None:
    atlas = _make_atlas()
    mask = atlas.parcellation == 1  # exactly StructA_L, 125 voxels

    table = localize_mask(mask, atlas.parcellation, atlas)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["structure"] == "StructA_L"
    assert row["frac_of_structure"] == pytest.approx(1.0)
    assert table["frac_of_tumour"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. frac_of_tumour sums to exactly 1.0, including the unlabelled row
# --------------------------------------------------------------------------- #


def test_frac_of_tumour_sums_to_one_including_unlabelled() -> None:
    atlas = _make_atlas()
    mask = atlas.parcellation == 1  # 125 voxels, StructA_L
    mask = mask.copy()
    mask[15:20, 0:2, 0:3] = True  # 30 voxels of pure background, outside every structure

    table = localize_mask(mask, atlas.parcellation, atlas)

    assert table["frac_of_tumour"].sum() == pytest.approx(1.0)
    assert set(table["structure"]) == {"StructA_L", "unlabelled"}
    unlabelled = table[table["structure"] == "unlabelled"].iloc[0]
    assert unlabelled["n_voxels"] == 30
    assert math.isnan(unlabelled["frac_of_structure"])


# --------------------------------------------------------------------------- #
# 3. The two fractions genuinely differ -- the brainstem scenario
# --------------------------------------------------------------------------- #


def test_frac_of_tumour_and_frac_of_structure_diverge_on_brainstem() -> None:
    atlas = _make_atlas()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[0:4, 0:5, 0:5] = True  # 100 of StructA_L's 125 voxels
    mask[10:12, 10:12, 0:2] = True  # all 8 of Brainstem's voxels

    table = localize_mask(mask, atlas.parcellation, atlas)

    brainstem = table[table["structure"] == "Brainstem"].iloc[0]
    struct_a = table[table["structure"] == "StructA_L"].iloc[0]

    assert brainstem["frac_of_tumour"] == pytest.approx(8 / 108)
    assert brainstem["frac_of_structure"] == pytest.approx(1.0)
    assert struct_a["frac_of_tumour"] == pytest.approx(100 / 108)
    assert struct_a["frac_of_structure"] == pytest.approx(100 / 125)

    # frac_of_tumour says StructA_L dominates; frac_of_structure says the
    # opposite -- the brainstem is the one FULLY destroyed.
    assert brainstem["frac_of_tumour"] < struct_a["frac_of_tumour"]
    assert brainstem["frac_of_structure"] > struct_a["frac_of_structure"]

    # Sorted by frac_of_structure first: brainstem must be row 0.
    assert table.iloc[0]["structure"] == "Brainstem"


# --------------------------------------------------------------------------- #
# 4. Empty mask -> empty table, right columns and dtypes, never a crash
# --------------------------------------------------------------------------- #


def test_empty_mask_gives_empty_table_with_right_columns_and_dtypes() -> None:
    atlas = _make_atlas()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)

    table = localize_mask(mask, atlas.parcellation, atlas)

    assert list(table.columns) == list(_LOCALIZE_COLUMNS)
    assert len(table) == 0
    assert table["n_voxels"].dtype == np.int64
    for col in ("volume_mm3", "frac_of_tumour", "frac_of_structure"):
        assert table[col].dtype == np.float64


# --------------------------------------------------------------------------- #
# 5. knowledge=None still yields the same column set
# --------------------------------------------------------------------------- #


def test_knowledge_none_still_yields_lobe_eloquence_matched_term_columns() -> None:
    atlas = _make_atlas()
    mask = atlas.parcellation == 1

    table = localize_mask(mask, atlas.parcellation, atlas, knowledge=None)

    assert list(table.columns) == list(_LOCALIZE_COLUMNS)
    assert (table["lobe"] == "").all()
    assert (table["eloquence"] == "").all()
    assert (table["matched_term"] == "").all()


# --------------------------------------------------------------------------- #
# 6. Unlabelled voxels are not dropped
# --------------------------------------------------------------------------- #


def test_mask_entirely_outside_any_structure_gives_one_unlabelled_row() -> None:
    atlas = _make_atlas()
    mask = np.zeros(_ATLAS_SHAPE, dtype=bool)
    mask[15:20, 0:2, 0:3] = True  # pure background, 30 voxels

    table = localize_mask(mask, atlas.parcellation, atlas)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["structure"] == "unlabelled"
    assert row["laterality"] == "midline"
    assert row["frac_of_tumour"] == pytest.approx(1.0)
    assert math.isnan(row["frac_of_structure"])


# --------------------------------------------------------------------------- #
# 7. Crop geometry: `cropped` is load-bearing, not cosmetic
# --------------------------------------------------------------------------- #


def test_cropped_and_uncropped_give_identical_tables_when_paired_correctly() -> None:
    atlas = _make_small_atlas()
    meta = {
        "bbox": [[1, 9], [1, 9], [0, 6]],
        "cropped_shape": (8, 8, 6),
        "original_shape": (10, 10, 6),
        "spacing": [1.0, 1.0, 1.0],
    }

    parcellation_cropped, tissue_cropped = atlas_for_case(atlas, meta, cropped=True)
    parcellation_full, tissue_full = atlas_for_case(atlas, meta, cropped=False)
    assert parcellation_cropped.shape == (8, 8, 6)
    assert parcellation_full.shape == (10, 10, 6)
    assert tissue_cropped is None
    assert tissue_full is None

    # The structure sits at global [2:5, 2:5, 1:4]; the crop offset is (1, 1, 0).
    mask_full = np.zeros((10, 10, 6), dtype=bool)
    mask_full[2:5, 2:5, 1:4] = True
    mask_local = np.zeros((8, 8, 6), dtype=bool)
    mask_local[1:4, 1:4, 1:4] = True

    table_cropped = localize_mask(mask_local, parcellation_cropped, atlas, spacing=(1.0, 1.0, 1.0))
    table_full = localize_mask(mask_full, parcellation_full, atlas, spacing=(1.0, 1.0, 1.0))
    pd.testing.assert_frame_equal(table_cropped, table_full)

    row = table_cropped.iloc[0]
    assert row["structure"] == "StructA_L"
    assert row["n_voxels"] == 27
    assert row["frac_of_structure"] == pytest.approx(1.0)


def test_wrong_cropped_pairing_raises_on_shape_mismatch() -> None:
    atlas = _make_small_atlas()
    meta = {
        "bbox": [[1, 9], [1, 9], [0, 6]],
        "cropped_shape": (8, 8, 6),
        "original_shape": (10, 10, 6),
        "spacing": [1.0, 1.0, 1.0],
    }
    parcellation_full, _ = atlas_for_case(atlas, meta, cropped=False)

    mask_local = np.zeros((8, 8, 6), dtype=bool)
    mask_local[1:4, 1:4, 1:4] = True

    # A cropped-frame mask against the full (uncropped) parcellation: shapes
    # disagree outright, so this must raise rather than silently misalign.
    with pytest.raises(ValueError, match="cropped"):
        localize_mask(mask_local, parcellation_full, atlas, spacing=(1.0, 1.0, 1.0))


def test_wrong_crop_offset_silently_shifts_every_assignment() -> None:
    """Same shape, wrong offset -> a different, wrong table, not an exception.

    Pins the hazard `atlas_for_case`'s docstring warns about: the crop
    offset (which `cropped` gates) is load-bearing rather than cosmetic. A
    shape-only check cannot catch a wrong offset that happens to produce an
    array of the right SIZE -- exactly the situation here.
    """
    atlas = _make_small_atlas()
    correct_offset_parcellation = atlas.parcellation[1:9, 1:9, 0:6]  # the true bbox
    wrong_offset_parcellation = atlas.parcellation[0:8, 0:8, 0:6]  # same shape, offset 0 not 1
    assert correct_offset_parcellation.shape == wrong_offset_parcellation.shape

    mask_local = np.zeros((8, 8, 6), dtype=bool)
    mask_local[1:4, 1:4, 1:4] = True  # correct under offset (1, 1, 0)

    table_correct = localize_mask(
        mask_local, correct_offset_parcellation, atlas, spacing=(1.0, 1.0, 1.0)
    )
    table_wrong = localize_mask(
        mask_local, wrong_offset_parcellation, atlas, spacing=(1.0, 1.0, 1.0)
    )

    assert not table_wrong.equals(table_correct)

    correct_row = table_correct[table_correct["structure"] == "StructA_L"].iloc[0]
    assert correct_row["n_voxels"] == 27
    assert correct_row["frac_of_structure"] == pytest.approx(1.0)

    wrong_row = table_wrong[table_wrong["structure"] == "StructA_L"].iloc[0]
    assert wrong_row["n_voxels"] == 12  # only the misaligned overlap
    assert wrong_row["frac_of_structure"] == pytest.approx(12 / 27)
    assert "unlabelled" in set(table_wrong["structure"])  # the rest fell outside the structure


# --------------------------------------------------------------------------- #
# 8. atlas_for_case raises when the cropped shape disagrees with meta
# --------------------------------------------------------------------------- #


def test_atlas_for_case_raises_on_cropped_shape_mismatch() -> None:
    atlas = _make_small_atlas()
    meta = {
        "bbox": [[1, 9], [1, 9], [0, 6]],  # true crop shape is (8, 8, 6)
        "cropped_shape": (7, 8, 6),  # deliberately wrong
        "original_shape": (10, 10, 6),
        "spacing": [1.0, 1.0, 1.0],
    }
    with pytest.raises(ValueError, match="cropped_shape"):
        atlas_for_case(atlas, meta, cropped=True)


# --------------------------------------------------------------------------- #
# load_classification -- the atlas-free half of the knowledge base
# --------------------------------------------------------------------------- #


def test_load_classification_needs_no_atlas_and_reads_every_field(tmp_path: Path) -> None:
    """scripts/report.py joins already-written CSVs; making it load an atlas for a citation
    string would cost minutes per run for metadata that is pure file content."""
    elo_path = tmp_path / "eloquence_map.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "Precentral_L", "eloquence": "eloquent", "matched_term": "motor"}],
        coverage_gaps=[{"term": "internal capsule"}, {"term": "dentate nucleus"}],
        distance_mm=7.5,
    )

    classification = load_classification(elo_path)

    assert classification.name == "Test eloquence grading"
    assert classification.evidence.startswith("Eloquent locations")
    # read_via is present in the fixture, so it must be folded into the citation --
    # this project did not read the primary source and the artifact has to say so.
    assert "read via" in classification.citation
    assert classification.coverage_gaps == ("internal capsule", "dentate nucleus")
    assert classification.near_eloquent_mm == 7.5
    assert classification.version == 1


def test_load_knowledge_and_load_classification_cannot_disagree(tmp_path: Path) -> None:
    """One parser, two callers. If these ever drift, a report's citation stops matching the
    one the localisation run validated against the atlas."""
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "StructA_L", "eloquence": "eloquent", "matched_term": "motor"}],
        distance_mm=12.0,
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    knowledge = load_knowledge(elo_path, lobe_path, atlas)
    classification = load_classification(elo_path)

    assert knowledge.classification_name == classification.name
    assert knowledge.evidence == classification.evidence
    assert knowledge.citation == classification.citation
    assert knowledge.coverage_gaps == classification.coverage_gaps
    assert knowledge.near_eloquent_mm == classification.near_eloquent_mm


@pytest.mark.parametrize("field", ["name", "eloquent_structures_verbatim", "primary_citation"])
def test_load_classification_raises_on_an_empty_required_field(tmp_path: Path, field: str) -> None:
    """The evidence sentence is this project's substitute for the expert review it does not
    have (see docs/research/interpretable_pipeline_plan.md Finding E), so an empty one is a
    hard failure rather than an empty string flowing into a report."""
    elo_path = tmp_path / "eloquence_map.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "Precentral_L", "eloquence": "eloquent", "matched_term": "motor"}],
    )
    doc = yaml.safe_load(elo_path.read_text())
    doc["classification"][field] = "   "
    elo_path.write_text(yaml.safe_dump(doc))

    with pytest.raises(ValueError, match=field if field != "name" else "classification.name"):
        load_classification(elo_path)


# --------------------------------------------------------------------------- #
# 9-11. load_knowledge validation
# --------------------------------------------------------------------------- #


def test_load_knowledge_raises_on_unknown_structure_and_names_it(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "NotARealStructure", "eloquence": "eloquent", "matched_term": "x"}],
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    with pytest.raises(ValueError, match="NotARealStructure"):
        load_knowledge(elo_path, lobe_path, atlas)


def test_load_knowledge_raises_on_out_of_vocabulary_value(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "StructA_L", "eloquence": "maybe", "matched_term": "x"}],
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    with pytest.raises(ValueError):
        load_knowledge(elo_path, lobe_path, atlas)


def test_default_is_unclassified_and_non_eloquent_never_appears(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "StructA_L", "eloquence": "eloquent", "matched_term": "motor"}],
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    kb = load_knowledge(elo_path, lobe_path, atlas)

    assert kb.eloquence["StructB_L"] == "unclassified"  # absent from entries
    assert "non-eloquent" not in kb.eloquence.values()
    assert all(v != "non-eloquent" for v in kb.eloquence.values())


# --------------------------------------------------------------------------- #
# 12. Lobe lookup strips _L/_R, raises when the base name is missing
# --------------------------------------------------------------------------- #


def test_lobe_lookup_strips_laterality_suffix(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(elo_path, [])
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    kb = load_knowledge(elo_path, lobe_path, atlas)

    assert kb.lobe["StructA_L"] == "frontal"
    assert kb.lobe["StructA_R"] == "frontal"
    assert kb.lobe["Brainstem"] == "brainstem"  # no suffix to strip


def test_lobe_lookup_raises_when_base_name_missing(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(elo_path, [])
    incomplete_lobe_map = {k: v for k, v in _FULL_LOBE_MAP.items() if k != "Brainstem"}
    _write_lobe_yaml(lobe_path, incomplete_lobe_map)

    with pytest.raises(ValueError, match="Brainstem"):
        load_knowledge(elo_path, lobe_path, atlas)


# --------------------------------------------------------------------------- #
# 13. coverage_line
# --------------------------------------------------------------------------- #


def test_coverage_line_reports_counts_and_gap_terms(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [
            {"structure_name": "StructA_L", "eloquence": "eloquent", "matched_term": "motor"},
            {"structure_name": "StructA_R", "eloquence": "eloquent", "matched_term": "motor"},
        ],
        coverage_gaps=[
            {"term": "internal capsule", "reason": "no white-matter label"},
            {"term": "dentate nucleus", "reason": "no deep cerebellar nuclei"},
        ],
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    kb = load_knowledge(elo_path, lobe_path, atlas)
    line = kb.coverage_line(4)

    assert "2 of 4" in line
    assert "2 unclassified" in line
    assert "internal capsule" in line
    assert "dentate nucleus" in line


# --------------------------------------------------------------------------- #
# 14. eloquent_union_mask raises when nothing is marked eloquent
# --------------------------------------------------------------------------- #


def test_eloquent_union_mask_raises_when_nothing_is_eloquent(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(elo_path, [])  # no entries at all -> everything unclassified
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    kb = load_knowledge(elo_path, lobe_path, atlas)

    with pytest.raises(ValueError, match="eloquent"):
        eloquent_union_mask(atlas, kb)


def test_eloquent_union_mask_unions_marked_structures(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "Brainstem", "eloquence": "eloquent", "matched_term": "brainstem"}],
    )
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)

    kb = load_knowledge(elo_path, lobe_path, atlas)
    mask = eloquent_union_mask(atlas, kb)

    assert mask.shape == atlas.shape
    assert np.array_equal(mask, atlas.parcellation == 3)


# --------------------------------------------------------------------------- #
# 15. distance_to_eloquent
# --------------------------------------------------------------------------- #


def test_distance_to_eloquent_zero_on_overlap() -> None:
    shape = (6, 3, 3)
    eloquent_mask = np.zeros(shape, dtype=bool)
    eloquent_mask[0, 1, 1] = True
    mask = np.zeros(shape, dtype=bool)
    mask[0, 1, 1] = True  # same voxel

    assert distance_to_eloquent(mask, eloquent_mask, spacing=(2.0, 1.0, 1.0)) == 0.0


def test_distance_to_eloquent_matches_hand_computed_value_with_anisotropic_spacing() -> None:
    shape = (6, 3, 3)
    eloquent_mask = np.zeros(shape, dtype=bool)
    eloquent_mask[0, 1, 1] = True
    mask = np.zeros(shape, dtype=bool)
    mask[3, 1, 1] = True  # 3 voxels away along axis 0

    dist = distance_to_eloquent(mask, eloquent_mask, spacing=(2.0, 1.0, 1.0))
    assert dist == pytest.approx(6.0)  # 3 voxels * 2.0 mm spacing


def test_distance_to_eloquent_nan_when_either_mask_empty() -> None:
    shape = (6, 3, 3)
    eloquent_mask = np.zeros(shape, dtype=bool)
    eloquent_mask[0, 1, 1] = True
    mask = np.zeros(shape, dtype=bool)
    mask[3, 1, 1] = True
    empty = np.zeros(shape, dtype=bool)

    assert math.isnan(distance_to_eloquent(empty, eloquent_mask))
    assert math.isnan(distance_to_eloquent(mask, empty))


# --------------------------------------------------------------------------- #
# 16. summarize_case
# --------------------------------------------------------------------------- #


def _hand_built_table() -> pd.DataFrame:
    rows = [
        {
            "structure": "Big",
            "laterality": "L",
            "lobe": "frontal",
            "eloquence": "unclassified",
            "matched_term": "",
            "n_voxels": 1000,
            "volume_mm3": 1000.0,
            "frac_of_tumour": 0.9,
            "frac_of_structure": 0.5,
        },
        {
            "structure": "Small",
            "laterality": "midline",
            "lobe": "deep",
            "eloquence": "eloquent",
            "matched_term": "basal ganglia",
            "n_voxels": 100,
            "volume_mm3": 100.0,
            "frac_of_tumour": 0.09,
            "frac_of_structure": 0.95,
        },
        {
            "structure": "unlabelled",
            "laterality": "midline",
            "lobe": "",
            "eloquence": "",
            "matched_term": "",
            "n_voxels": 10,
            "volume_mm3": 10.0,
            "frac_of_tumour": 0.01,
            "frac_of_structure": float("nan"),
        },
    ]
    return pd.DataFrame(rows, columns=list(_LOCALIZE_COLUMNS))


def test_summarize_case_returns_only_plain_python_scalars() -> None:
    table = _hand_built_table()
    summary = summarize_case(table, knowledge=None, top_n=5)
    for key, value in summary.items():
        assert type(value) in (int, float, str), f"{key} has type {type(value)}"


def test_summarize_case_most_displaced_is_max_frac_of_structure_not_max_volume() -> None:
    table = _hand_built_table()
    summary = summarize_case(table, knowledge=None, top_n=5)

    # "Big" has 10x the volume and the largest frac_of_tumour, but "Small" is
    # the one nearly wholly consumed (frac_of_structure 0.95 vs 0.5).
    assert summary["most_displaced_structure"] == "Small"
    assert summary["top_structure"] == "Big"


def test_summarize_case_with_knowledge_fills_eloquence_fields(tmp_path: Path) -> None:
    atlas = _make_atlas()
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(elo_path, [])
    _write_lobe_yaml(lobe_path, _FULL_LOBE_MAP)
    kb = load_knowledge(elo_path, lobe_path, atlas)

    table = _hand_built_table()
    summary = summarize_case(table, knowledge=kb, top_n=5)

    assert summary["n_eloquent_structures"] == 1  # "Small" row's eloquence == "eloquent"
    assert summary["eloquent_frac_of_tumour"] == pytest.approx(0.09)
    assert summary["distance_to_eloquent_mm"] == 0.0  # table already shows overlap
    assert summary["coverage_line"] == kb.coverage_line(len(kb.eloquence))


def test_summarize_case_without_knowledge_defaults_eloquence_fields() -> None:
    table = _hand_built_table()
    summary = summarize_case(table, knowledge=None, top_n=5)

    assert summary["n_eloquent_structures"] == 0
    assert summary["eloquent_frac_of_tumour"] == 0.0
    assert summary["coverage_line"] == ""
    assert math.isnan(summary["distance_to_eloquent_mm"])


def test_summarize_case_on_empty_table_gives_zeros_and_nans() -> None:
    atlas = _make_atlas()
    empty_table = localize_mask(np.zeros(_ATLAS_SHAPE, dtype=bool), atlas.parcellation, atlas)

    summary = summarize_case(empty_table, knowledge=None, top_n=5)

    assert summary["n_structures_involved"] == 0
    assert summary["top_structure"] == ""
    assert math.isnan(summary["top_frac_of_structure"])
    assert summary["most_displaced_structure"] == ""
    assert math.isnan(summary["frac_unlabelled"])
    assert summary["dominant_lobe"] == ""
    assert summary["n_eloquent_structures"] == 0
    assert summary["eloquent_frac_of_tumour"] == 0.0


# --------------------------------------------------------------------------- #
# 17. localize_case validates classes like burden
# --------------------------------------------------------------------------- #


def test_localize_case_raises_on_raw_brats_label_four() -> None:
    atlas = _make_small_atlas()
    classes = np.zeros((10, 10, 6), dtype=np.uint8)
    classes[0, 0, 0] = 4  # raw, unremapped BraTS enhancing-tumour label
    meta = {"spacing": [1.0, 1.0, 1.0]}

    with pytest.raises(ValueError, match="BraTS"):
        localize_case(classes, atlas, meta, cropped=False)


def test_localize_case_concatenates_regions_with_leading_region_column() -> None:
    atlas = _make_small_atlas()
    classes = np.zeros((10, 10, 6), dtype=np.uint8)
    classes[2:5, 2:5, 1:4] = 3  # ET, entirely inside StructA_L -> also TC, WT
    meta = {"spacing": [1.0, 1.0, 1.0]}

    table = localize_case(classes, atlas, meta, cropped=False)

    assert list(table.columns) == ["region"] + list(_LOCALIZE_COLUMNS)
    assert set(table["region"]) == {"ET", "TC", "WT"}
    for region in ("ET", "TC", "WT"):
        region_rows = table[table["region"] == region]
        assert len(region_rows) == 1
        assert region_rows.iloc[0]["structure"] == "StructA_L"
        assert region_rows.iloc[0]["n_voxels"] == 27


# --------------------------------------------------------------------------- #
# 18. No dependency on the deep-learning stack
# --------------------------------------------------------------------------- #


def test_localize_module_does_not_import_the_deep_learning_stack() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly already imported that stack for some other
    test file in the suite -- see `test_burden.py`'s equivalent guard.
    """
    source = Path(localize.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
