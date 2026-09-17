"""Tests for `neurovision.anatomy.atlas_export`.

Every test runs on CPU against a small hand-built synthetic `Atlas` (never the
real SRI24 atlas) and a `KnowledgeBase` built either directly or via
`load_knowledge` on tiny YAML fixtures written to `tmp_path` (never the real
committed `knowledge/` files). Each test is well under a second.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurovision.anatomy.atlas import Atlas, AtlasLabels, AtlasStructure
from neurovision.anatomy.atlas_export import (
    structure_index_for_name,
    structure_index_volume,
    structure_table,
)
from neurovision.anatomy.localize import KnowledgeBase, load_knowledge
from tests.test_localize import _write_eloquence_yaml, _write_lobe_yaml

# --------------------------------------------------------------------------- #
# Synthetic atlas fixture
# --------------------------------------------------------------------------- #
#
# Shape (6, 6, 4). Three structures:
#   StructA_L: [0:2, 0:2, 0:2] -> label 1
#   StructA_R: [2:4, 0:2, 0:2] -> label 2
#   Brainstem: merged from raw label ids 3 AND 5 (proves the merge), midline
#       label 3 at [4:5, 4:5, 0:1], label 5 at [5:6, 4:5, 0:1]
# Plus one unmapped raw id (9), present in the volume but claimed by no
# structure, at [0:1, 5:6, 0:1].
# Everything else is background (0).

_ATLAS_SHAPE = (6, 6, 4)


def _make_atlas() -> Atlas:
    parcellation = np.zeros(_ATLAS_SHAPE, dtype=np.int16)
    parcellation[0:2, 0:2, 0:2] = 1  # StructA_L
    parcellation[2:4, 0:2, 0:2] = 2  # StructA_R
    parcellation[4:5, 4:5, 0:1] = 3  # Brainstem, first sub-label
    parcellation[5:6, 4:5, 0:1] = 5  # Brainstem, second sub-label (the merge)
    parcellation[0:1, 5:6, 0:1] = 9  # unmapped raw id

    structures = (
        AtlasStructure(name="StructA_L", label_ids=(1,), laterality="L"),
        AtlasStructure(name="StructA_R", label_ids=(2,), laterality="R"),
        AtlasStructure(name="Brainstem", label_ids=(3, 5), laterality="midline"),
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
        unmapped_ids=(9,),
    )


def _make_knowledge(atlas: Atlas, tmp_path) -> KnowledgeBase:
    elo_path = tmp_path / "eloquence_map.yaml"
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_eloquence_yaml(
        elo_path,
        [{"structure_name": "Brainstem", "eloquence": "eloquent", "matched_term": "brainstem"}],
    )
    _write_lobe_yaml(
        lobe_path,
        {
            "StructA": {"lobe": "frontal"},
            "Brainstem": {"lobe": "brainstem"},
        },
    )
    return load_knowledge(elo_path, lobe_path, atlas)


# --------------------------------------------------------------------------- #
# structure_index_volume
# --------------------------------------------------------------------------- #


def test_volume_max_equals_number_of_structures():
    atlas = _make_atlas()
    volume = structure_index_volume(atlas)
    assert int(volume.max()) == len(atlas.labels.structures)


def test_volume_dtype_and_shape():
    atlas = _make_atlas()
    volume = structure_index_volume(atlas)
    assert volume.dtype == np.uint8
    assert volume.shape == atlas.parcellation.shape


def test_background_voxels_map_to_zero():
    atlas = _make_atlas()
    volume = structure_index_volume(atlas)
    # A background voxel far from every structure and the unmapped id.
    assert volume[3, 3, 3] == 0


def test_unmapped_voxel_maps_to_zero():
    atlas = _make_atlas()
    volume = structure_index_volume(atlas)
    assert volume[0, 5, 0] == 0


def test_merged_sub_label_maps_to_same_structure_index():
    atlas = _make_atlas()
    volume = structure_index_volume(atlas)
    brainstem_index = 1 + atlas.labels.names.index("Brainstem")
    # Raw label 3 (first sub-label).
    assert volume[4, 4, 0] == brainstem_index
    # Raw label 5 (second sub-label, merged into the same structure).
    assert volume[5, 4, 0] == brainstem_index


# --------------------------------------------------------------------------- #
# structure_table / structure_index_for_name round trip
# --------------------------------------------------------------------------- #


def test_round_trip_index_for_name(tmp_path):
    atlas = _make_atlas()
    knowledge = _make_knowledge(atlas, tmp_path)
    table = structure_table(atlas, knowledge)

    for structure in atlas.labels.structures:
        k = structure_index_for_name(table, structure.name)
        assert k is not None
        assert table[k - 1]["name"] == structure.name


def test_index_for_name_unknown_returns_none(tmp_path):
    atlas = _make_atlas()
    knowledge = _make_knowledge(atlas, tmp_path)
    table = structure_table(atlas, knowledge)
    assert structure_index_for_name(table, "NoSuchStructure") is None


def test_table_rows_carry_knowledge_fields(tmp_path):
    atlas = _make_atlas()
    knowledge = _make_knowledge(atlas, tmp_path)
    table = structure_table(atlas, knowledge)

    by_name = {row["name"]: row for row in table}
    assert by_name["Brainstem"]["eloquence"] == "eloquent"
    assert by_name["Brainstem"]["matched_term"] == "brainstem"
    assert by_name["Brainstem"]["lobe"] == "brainstem"
    assert by_name["StructA_L"]["laterality"] == "L"
    assert by_name["StructA_L"]["lobe"] == "frontal"
    assert by_name["StructA_L"]["eloquence"] == "unclassified"


def test_table_index_matches_volume_value(tmp_path):
    atlas = _make_atlas()
    knowledge = _make_knowledge(atlas, tmp_path)
    table = structure_table(atlas, knowledge)
    volume = structure_index_volume(atlas)

    struct_a_l_index = structure_index_for_name(table, "StructA_L")
    assert volume[0, 0, 0] == struct_a_l_index


# --------------------------------------------------------------------------- #
# 256-structure guard
# --------------------------------------------------------------------------- #


def test_too_many_structures_raises():
    n_structures = 256
    parcellation = np.zeros((2, 2, n_structures), dtype=np.int16)
    for i in range(n_structures):
        parcellation[0, 0, i] = i + 1

    structures = tuple(
        AtlasStructure(name=f"Struct_{i}", label_ids=(i + 1,), laterality="midline")
        for i in range(n_structures)
    )
    labels = AtlasLabels(structures=structures, unmapped_name="unclassified")
    atlas = Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=None,
        tissue_codes={},
        name="synthetic-too-big",
        version="0",
        source="test",
        unmapped_ids=(),
    )

    with pytest.raises(ValueError):
        structure_index_volume(atlas)
