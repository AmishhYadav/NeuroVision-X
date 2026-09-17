"""Wire-format export of the atlas parcellation for the 3D twin viewer.

The 3D twin renders a handful of selected anatomical structures as translucent
shells. It needs two things this module builds, both pure array/table
arithmetic with no model and no dependency on the deep-learning stack (same
family as `neurovision.anatomy.atlas` and `neurovision.anatomy.localize`):

1. A `uint8` structure-index volume it can ship to the browser like every
   other volume it already handles (predictions, uncertainty maps, ...).
   The raw SRI24/TZO parcellation carries `int16` per-plane sub-label ids up
   to 424, which does not fit a `uint8` wire volume and is not what a viewer
   wants anyway -- it wants one shell per merged structure, not one per
   sub-label. `structure_index_volume` collapses each voxel to its merged
   structure's position via `AtlasLabels.lookup_array` (the same lookup
   `localize.py` uses), so the two stay in agreement by construction.
2. A small per-structure table (name, laterality, lobe, eloquence,
   `matched_term`) the frontend can show alongside each shell, keyed by the
   same 1-based index the volume carries.

This module is NOT a report producer: nothing it computes backs any published
number, and it does not call `localize_mask` or touch a tumour mask at all.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from neurovision.anatomy.atlas import Atlas
from neurovision.anatomy.localize import KnowledgeBase

__all__ = ["structure_index_volume", "structure_table", "structure_index_for_name"]

logger = logging.getLogger(__name__)

# A uint8 wire volume has 256 distinct values; 0 is reserved for
# background/unmapped, leaving 255 usable structure indices.
_MAX_STRUCTURES = 255


def structure_index_volume(atlas: Atlas) -> np.ndarray:
    """Collapses the raw parcellation to a `uint8` merged-structure-index volume.

    Every raw label id belonging to the same merged structure (e.g. the
    per-plane sub-labels of a "plus" structure) collapses to the same output
    value, so the volume has one distinct value per structure rather than one
    per raw SRI24 label id.

    Voxel value is `0` for background/unmapped and `1 + position` of the
    structure in `atlas.labels.structures` otherwise. Index `0` is reserved
    for "no structure" -- deliberately, rather than using `-1` or the
    structure's own position starting at `0` -- for two reasons: the frontend
    can skip index `0` without a lookup, and every other volume this pipeline
    ships to the browser (predictions, uncertainty maps) is already `uint8`
    with `0` meaning "nothing here", so this stays consistent with that
    convention rather than introducing a new one.

    Args:
        atlas: The loaded `Atlas` to export.

    Returns:
        `(D, H, W)` `uint8`, C-contiguous, same shape as `atlas.parcellation`.

    Raises:
        ValueError: If `atlas.labels.structures` has more than 255 entries --
            a `uint8` volume can only distinguish 255 non-background values.
    """
    n_structures = len(atlas.labels.structures)
    if n_structures > _MAX_STRUCTURES:
        raise ValueError(
            f"structure_index_volume: atlas has {n_structures} structures, which exceeds the "
            f"{_MAX_STRUCTURES} a uint8 wire volume can represent (0 is reserved for "
            "background/unmapped)."
        )

    max_id = int(atlas.parcellation.max())
    # lookup_array already gives -1 for both background (label 0) and any
    # unmapped label id, and a structure's position (0-based) otherwise --
    # so "+1" alone turns that into exactly the 0/1-based convention above,
    # with no separate background/unmapped branch needed.
    lookup = atlas.labels.lookup_array(max_id)
    structure_index = lookup[atlas.parcellation]
    volume = np.ascontiguousarray((structure_index + 1).astype(np.uint8))

    n_unmapped = int(np.count_nonzero((atlas.parcellation != 0) & (volume == 0)))
    logger.debug(
        "structure_index_volume: %d structures, %d unmapped voxels", n_structures, n_unmapped
    )
    return volume


def structure_table(atlas: Atlas, knowledge: KnowledgeBase) -> list[dict[str, Any]]:
    """Builds one metadata row per structure, in the index order `structure_index_volume` uses.

    Args:
        atlas: The loaded `Atlas` to export.
        knowledge: A `KnowledgeBase` from `load_knowledge`, built against this
            same `atlas` -- it is guaranteed to hold an entry for every
            structure name in `atlas.labels.structures`.

    Returns:
        A list of dicts, one per structure, in the same order as
        `atlas.labels.structures`: `{"index": k, "name", "laterality",
        "lobe", "eloquence", "matched_term"}`, where `k` is the 1-based
        index matching the value that structure carries in
        `structure_index_volume`'s output.
    """
    rows: list[dict[str, Any]] = []
    for position, structure in enumerate(atlas.labels.structures):
        name = structure.name
        rows.append(
            {
                "index": position + 1,
                "name": name,
                "laterality": structure.laterality,
                "lobe": knowledge.lobe[name],
                "eloquence": knowledge.eloquence[name],
                "matched_term": knowledge.matched_term[name],
            }
        )
    return rows


def structure_index_for_name(table: list[dict[str, Any]], name: str) -> int | None:
    """Looks up a structure's 1-based volume index by name.

    Args:
        table: A `structure_table` output.
        name: A merged structure name, e.g. `"LateralVentricle_L"`.

    Returns:
        The matching row's `"index"`, or `None` if `name` is not present.
    """
    for row in table:
        if row["name"] == name:
            return int(row["index"])
    return None
