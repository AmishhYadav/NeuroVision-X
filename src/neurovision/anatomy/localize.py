"""Anatomical localisation: which SRI24 structures a tumour mask touches, and how badly.

Phase 1 of `docs/research/interpretable_pipeline_plan.md` §5. Given a binary
region mask (ET / TC / WT, from `neurovision.anatomy.burden.region_mask`) and
the loaded `Atlas`, this module answers two, deliberately separate, questions
per structure:

    frac_of_tumour   = overlap_voxels / tumour_voxels   -- "where is the tumour?"
    frac_of_structure = overlap_voxels / structure_voxels -- "how badly is this
                         structure affected?"

**Both ship, and they must never be conflated.** A lesion can place 5% of its
own volume in the brainstem while that 5% destroys 40% of the brainstem --
reporting only `frac_of_tumour` would demote that to a footnote, when it is
the most important line in a report about mass effect and eloquent structure
involvement.

This module also loads the two committed, versioned knowledge artifacts
(`knowledge/eloquence_map.yaml`, `knowledge/aal_lobes.yaml`) into a
`KnowledgeBase`, validated against the loaded atlas so a stale or mistyped
structure name is caught at load time rather than silently mapping nothing.

Geometry, and the rule that will otherwise go silently wrong: the atlas is in
original BraTS geometry (240x240x155, same as a saved prediction from
`scripts/evaluate.py`), while a preprocessed `label.npy` is CROPPED to the
case's nonzero bounding box. `atlas_for_case` crops the atlas to match --
cheaper than uncropping a mask back to original geometry, since it is a plain
array view. The caller must derive `cropped` from where the mask actually
came from, never guess: pairing a cropped mask with the uncropped atlas view
(or vice versa) either raises on a shape mismatch or, worse, silently shifts
every structure assignment by the crop offset and produces an entirely
plausible, entirely wrong table.

This is pure array and table arithmetic: no model and no dependency on the
deep-learning stack, matching `neurovision.anatomy.atlas` and
`neurovision.anatomy.burden` -- see
`tests/test_localize.py::test_localize_module_does_not_import_the_deep_learning_stack`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from neurovision.anatomy import burden
from neurovision.anatomy.atlas import Atlas

__all__ = [
    "KnowledgeBase",
    "load_knowledge",
    "atlas_for_case",
    "localize_mask",
    "localize_case",
    "eloquent_union_mask",
    "distance_to_eloquent",
    "summarize_case",
]

logger = logging.getLogger(__name__)

_LOCALIZE_COLUMNS: tuple[str, ...] = (
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

_UNLABELLED_NAME = "unlabelled"


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KnowledgeBase:
    """The committed knowledge artifacts, loaded and validated against an atlas.

    Every dict here is keyed on the atlas's MERGED structure names (i.e.
    `atlas.labels.names`) and holds one entry per structure -- a structure
    absent from the source YAML's `entries` still gets a key, filled with the
    declared default, so a lookup never silently falls through to a caller's
    own fallback.

    Attributes:
        eloquence: Structure name -> `"eloquent"` or the file's declared
            `default` (`"unclassified"`). Never `"non-eloquent"` -- absence of
            a source is not evidence of functional silence.
        matched_term: Structure name -> the source term it matched (e.g.
            `"motor/sensory cortices"`), or `""` for an unclassified
            structure.
        evidence: The one verbatim sentence backing every `eloquent` entry
            (`classification.eloquent_structures_verbatim`).
        citation: `classification.primary_citation` combined with
            `classification.read_via` when present.
        lobe: Structure name -> lobe, from the lobe map keyed on the
            structure's BASE name (its merged name with a trailing `_L`/`_R`
            stripped).
        coverage_gaps: The `term` field of every entry in the eloquence map's
            `coverage_gaps` list -- source terms with no representable
            structure in this parcellation.
        near_eloquent_mm: `near_eloquent_rule.distance_mm` from the eloquence
            map -- a configurable threshold, not computed here.
    """

    eloquence: dict[str, str]
    matched_term: dict[str, str]
    evidence: str
    citation: str
    lobe: dict[str, str]
    coverage_gaps: tuple[str, ...]
    near_eloquent_mm: float

    def coverage_line(self, n_structures: int) -> str:
        """A one-line coverage summary for the report, so a thin knowledge base is visible.

        Args:
            n_structures: The total number of structures in the atlas (the
                denominator). Usually `len(self.eloquence)`, since that dict
                holds one entry per atlas structure -- passed explicitly
                rather than inferred, so a caller reporting against a
                DIFFERENT atlas subset gets an honest denominator.

        Returns:
            E.g. `"23 of 122 structures classified eloquent, 99 unclassified;
            4 source terms have no structure in this parcellation: internal
            capsule, dentate nucleus, hypothalamus, brainstem (midbrain,
            medulla)"`.
        """
        n_eloquent = sum(1 for v in self.eloquence.values() if v == "eloquent")
        n_unclassified = n_structures - n_eloquent
        line = (
            f"{n_eloquent} of {n_structures} structures classified eloquent, "
            f"{n_unclassified} unclassified"
        )
        if self.coverage_gaps:
            gaps_text = ", ".join(self.coverage_gaps)
            line += (
                f"; {len(self.coverage_gaps)} source terms have no structure in this "
                f"parcellation: {gaps_text}"
            )
        return line


def _strip_laterality(name: str) -> str:
    """Strips a trailing `_L` or `_R` from a merged structure name to its base name."""
    return re.sub(r"_(L|R)$", "", name)


def load_knowledge(
    eloquence_path: str | Path, lobe_path: str | Path, atlas: Atlas
) -> KnowledgeBase:
    """Loads and validates the two committed knowledge YAML files against an atlas.

    Validation is deliberately strict, because these files are the artifacts
    a non-expert reviewer audits (`docs/research/interpretable_pipeline_plan.md`
    §5 Phase 2, Finding E): every eloquence entry must name a real atlas
    structure, every eloquence value must come from the file's own declared
    vocabulary, and every atlas structure's base name must resolve in the
    lobe map.

    Args:
        eloquence_path: Path to `knowledge/eloquence_map.yaml`.
        lobe_path: Path to `knowledge/aal_lobes.yaml`.
        atlas: The loaded `Atlas` to validate structure names against.

    Returns:
        A `KnowledgeBase` with one entry per `atlas.labels.names` structure in
        every dict.

    Raises:
        ValueError: If an eloquence entry names a structure absent from the
            atlas, if an eloquence value is outside the file's `vocabulary`,
            if `evidence` or `citation` is empty, or if an atlas structure's
            base name is absent from the lobe map.
    """
    with open(eloquence_path, encoding="utf-8") as f:
        elo_doc = yaml.safe_load(f)
    with open(lobe_path, encoding="utf-8") as f:
        lobe_doc = yaml.safe_load(f)

    vocabulary = set(elo_doc["vocabulary"])
    default = elo_doc["default"]
    if default not in vocabulary:
        raise ValueError(
            f"load_knowledge: {eloquence_path}'s default '{default}' is not in its own "
            f"declared vocabulary {sorted(vocabulary)}."
        )

    atlas_names = set(atlas.labels.names)

    eloquence: dict[str, str] = dict.fromkeys(atlas.labels.names, default)
    matched_term: dict[str, str] = dict.fromkeys(atlas.labels.names, "")

    for entry in elo_doc.get("entries", []):
        name = entry["structure_name"]
        if name not in atlas_names:
            raise ValueError(
                f"load_knowledge: eloquence entry names structure '{name}', which does not "
                "exist in the loaded atlas. A mapping against a structure that is not the one "
                "being measured is exactly the failure this check exists to catch."
            )
        value = entry["eloquence"]
        if value not in vocabulary:
            raise ValueError(
                f"load_knowledge: eloquence value '{value}' for structure '{name}' is not in "
                f"the declared vocabulary {sorted(vocabulary)}."
            )
        eloquence[name] = value
        matched_term[name] = entry.get("matched_term", "")

    classification = elo_doc["classification"]
    evidence = str(classification["eloquent_structures_verbatim"]).strip()
    if not evidence:
        raise ValueError(
            f"load_knowledge: {eloquence_path}'s classification.eloquent_structures_verbatim "
            "is empty. This sentence is the substitute for expert review and must be present."
        )
    primary_citation = str(classification["primary_citation"]).strip()
    if not primary_citation:
        raise ValueError(
            f"load_knowledge: {eloquence_path}'s classification.primary_citation is empty."
        )
    read_via = str(classification.get("read_via", "")).strip()
    citation = f"{primary_citation} (read via: {read_via})" if read_via else primary_citation

    coverage_gaps = tuple(str(gap["term"]) for gap in elo_doc.get("coverage_gaps", []))
    near_eloquent_mm = float(elo_doc["near_eloquent_rule"]["distance_mm"])

    lobe_by_base = lobe_doc["structures"]
    lobe: dict[str, str] = {}
    for name in atlas.labels.names:
        base = _strip_laterality(name)
        if base not in lobe_by_base:
            raise ValueError(
                f"load_knowledge: structure '{name}' has base name '{base}', which is absent "
                f"from {lobe_path}."
            )
        entry = lobe_by_base[base]
        lobe[name] = str(entry["lobe"]) if isinstance(entry, Mapping) else str(entry)

    logger.info(
        "load_knowledge: loaded %d eloquent / %d unclassified structures, %d coverage gaps.",
        sum(1 for v in eloquence.values() if v == "eloquent"),
        sum(1 for v in eloquence.values() if v != "eloquent"),
        len(coverage_gaps),
    )

    return KnowledgeBase(
        eloquence=eloquence,
        matched_term=matched_term,
        evidence=evidence,
        citation=citation,
        lobe=lobe,
        coverage_gaps=coverage_gaps,
        near_eloquent_mm=near_eloquent_mm,
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def atlas_for_case(
    atlas: Atlas, meta: Mapping[str, Any], *, cropped: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crops the atlas to one case's bounding box, or returns it unchanged.

    A cheap array VIEW, not a copy -- avoids materialising a full 240x240x155
    array per case. `cropped` must be derived by the caller from where the
    mask being localised actually came from (a cropped `label.npy` /
    `predictions/` saved by `scripts/evaluate.py` are in ORIGINAL geometry
    while the preprocessed training image and label are cropped), never
    guessed: passing the wrong value shifts every structure assignment by the
    crop offset and produces an entirely plausible, entirely wrong table.

    Args:
        atlas: The loaded, case-geometry-aligned `Atlas`.
        meta: A case's `meta.json` mapping, holding `bbox`
            (`[[d0, d1], [h0, h1], [w0, w1]]`, `end` exclusive) and
            `cropped_shape` when `cropped=True`.
        cropped: Whether to crop the atlas to `meta["bbox"]`.

    Returns:
        `(parcellation, tissue)`, `tissue` being `None` when the atlas has no
        tissue map loaded.

    Raises:
        KeyError: If `cropped=True` and `meta` has no `bbox` key.
        ValueError: If `cropped=True` and the cropped shape does not equal
            `meta["cropped_shape"]` -- meaning `meta` is internally
            inconsistent (e.g. from a different preprocessing run) rather
            than a rounding effect.
    """
    if not cropped:
        return atlas.parcellation, atlas.tissue

    bbox = tuple(tuple(int(v) for v in pair) for pair in meta["bbox"])
    slices = tuple(slice(start, end) for start, end in bbox)

    parcellation = atlas.parcellation[slices]
    tissue = atlas.tissue[slices] if atlas.tissue is not None else None

    expected_shape = tuple(int(s) for s in meta["cropped_shape"])
    if parcellation.shape != expected_shape:
        raise ValueError(
            f"atlas_for_case: cropped shape {parcellation.shape} != meta['cropped_shape'] "
            f"{expected_shape}. meta['bbox'] and meta['cropped_shape'] disagree, which usually "
            "means meta.json came from a different preprocessing run than the mask being "
            "localised."
        )
    return parcellation, tissue


# --------------------------------------------------------------------------- #
# Localisation table
# --------------------------------------------------------------------------- #


def _empty_localize_table() -> pd.DataFrame:
    """An empty, correctly-typed table with `_LOCALIZE_COLUMNS`, never a crash or a zero row."""
    return pd.DataFrame(
        {
            "structure": pd.Series(dtype="object"),
            "laterality": pd.Series(dtype="object"),
            "lobe": pd.Series(dtype="object"),
            "eloquence": pd.Series(dtype="object"),
            "matched_term": pd.Series(dtype="object"),
            "n_voxels": pd.Series(dtype="int64"),
            "volume_mm3": pd.Series(dtype="float64"),
            "frac_of_tumour": pd.Series(dtype="float64"),
            "frac_of_structure": pd.Series(dtype="float64"),
        }
    )


def _structure_index_map(parcellation: np.ndarray, atlas: Atlas) -> tuple[np.ndarray, int]:
    """Maps every voxel to a structure index; -1 means unlabelled.

    "Unlabelled" covers both plain background (raw value 0) and any raw
    value present with no LUT row (`atlas.unmapped_ids`) -- both are voxels
    with no known anatomical structure, and are reported together as the
    `"unlabelled"` row rather than silently dropped, since AAL parcellates
    grey matter only and this row is routinely large on real data.

    Args:
        parcellation: `(D, H, W)` raw label array, already cropped to match
            whatever mask this will be intersected with.
        atlas: The `Atlas` supplying the structure name table.

    Returns:
        `(structure_index, n_structures)`. `structure_index` is an `int32`
        array the same shape as `parcellation`.
    """
    names = atlas.labels.names
    name_to_index = {name: i for i, name in enumerate(names)}

    unique_vals = np.unique(parcellation)
    val_to_index = np.full(unique_vals.shape, -1, dtype=np.int32)
    for i, raw_value in enumerate(unique_vals):
        structure_name = atlas.labels.name_for_id(int(raw_value))
        if structure_name in name_to_index:
            val_to_index[i] = name_to_index[structure_name]

    positions = np.searchsorted(unique_vals, parcellation)
    structure_index = val_to_index[positions]
    return structure_index, len(names)


def localize_mask(
    mask: np.ndarray,
    parcellation: np.ndarray,
    atlas: Atlas,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    knowledge: KnowledgeBase | None = None,
) -> pd.DataFrame:
    """Intersects one binary mask with one (already-matched) parcellation array.

    Args:
        mask: `(D, H, W)` boolean (or 0/1) array, one tumour region.
        parcellation: `(D, H, W)` raw atlas label array, ALREADY in the same
            geometry as `mask` -- see `atlas_for_case`.
        atlas: The `Atlas` supplying structure names and laterality.
        spacing: Voxel spacing in mm, `(D, H, W)` axis order.
        knowledge: Optional `KnowledgeBase` supplying `lobe` / `eloquence` /
            `matched_term`. When `None`, those three columns are still
            emitted, filled with `""` -- a column that appears and
            disappears depending on an optional argument would break every
            downstream consumer.

    Returns:
        A tidy `DataFrame`, one row per structure with non-zero overlap plus
        (when any voxels fall outside every known structure) one
        `"unlabelled"` row, columns `_LOCALIZE_COLUMNS`. Sorted by
        `frac_of_structure` descending, then `frac_of_tumour` descending, NaN
        last -- sorting by raw volume would bury a small, badly-affected
        structure like the brainstem below large, lightly-touched ones. An
        empty mask gives an empty table with the right columns and dtypes.

    Raises:
        ValueError: If `mask` is not 3-D, or `mask.shape != parcellation.shape`
            (almost always means `cropped` was resolved the wrong way when
            building `parcellation` -- see `atlas_for_case`).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"localize_mask: expected a (D, H, W) mask, got shape {mask.shape}.")
    if mask.shape != parcellation.shape:
        raise ValueError(
            f"localize_mask: mask shape {mask.shape} != parcellation shape "
            f"{parcellation.shape}. This usually means `cropped` was resolved the wrong way "
            "when building `parcellation` -- see `atlas_for_case`."
        )

    total_tumour_voxels = int(mask.sum())
    if total_tumour_voxels == 0:
        return _empty_localize_table()

    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    structure_index, n_structures = _structure_index_map(parcellation, atlas)
    names = atlas.labels.names

    masked_index = structure_index[mask]
    # Bucket the -1 (unlabelled) voxels into one extra slot at the end so a
    # single bincount call handles both real structures and the unlabelled
    # count.
    overlap_counts = np.bincount(
        np.where(masked_index >= 0, masked_index, n_structures), minlength=n_structures + 1
    )
    n_unlabelled = int(overlap_counts[n_structures])
    overlap_counts = overlap_counts[:n_structures]

    total_counts = np.bincount(
        np.where(structure_index >= 0, structure_index, n_structures).ravel(),
        minlength=n_structures + 1,
    )[:n_structures]

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        n_voxels = int(overlap_counts[i])
        if n_voxels == 0:
            continue
        structure_total = int(total_counts[i])
        if knowledge is not None:
            lobe = knowledge.lobe.get(name, "")
            eloquence = knowledge.eloquence.get(name, "")
            matched_term = knowledge.matched_term.get(name, "")
        else:
            lobe = ""
            eloquence = ""
            matched_term = ""
        rows.append(
            {
                "structure": name,
                "laterality": atlas.labels.by_name(name).laterality,
                "lobe": lobe,
                "eloquence": eloquence,
                "matched_term": matched_term,
                "n_voxels": n_voxels,
                "volume_mm3": n_voxels * voxel_volume,
                "frac_of_tumour": n_voxels / total_tumour_voxels,
                # structure_total >= n_voxels always here, since total_counts
                # counts every occurrence of this structure in the whole
                # array and overlap_counts counts a subset of those voxels.
                "frac_of_structure": n_voxels / structure_total,
            }
        )

    if n_unlabelled > 0:
        rows.append(
            {
                "structure": _UNLABELLED_NAME,
                "laterality": "midline",
                "lobe": "",
                "eloquence": "",
                "matched_term": "",
                "n_voxels": n_unlabelled,
                "volume_mm3": n_unlabelled * voxel_volume,
                "frac_of_tumour": n_unlabelled / total_tumour_voxels,
                "frac_of_structure": float("nan"),
            }
        )

    table = pd.DataFrame(rows, columns=list(_LOCALIZE_COLUMNS))
    table["n_voxels"] = table["n_voxels"].astype("int64")
    for col in ("volume_mm3", "frac_of_tumour", "frac_of_structure"):
        table[col] = table[col].astype("float64")

    table = table.sort_values(
        by=["frac_of_structure", "frac_of_tumour"], ascending=False, na_position="last"
    ).reset_index(drop=True)
    return table


def localize_case(
    classes: np.ndarray,
    atlas: Atlas,
    meta: Mapping[str, Any],
    *,
    cropped: bool,
    regions: Sequence[str] = ("ET", "TC", "WT"),
    knowledge: KnowledgeBase | None = None,
) -> pd.DataFrame:
    """Runs `localize_mask` for every region of one case and concatenates the results.

    Args:
        classes: Integer class map, `(D, H, W)`, values in `{0, 1, 2, 3}` --
            validated the same way `neurovision.anatomy.burden` validates it
            (via `burden.region_mask`).
        atlas: The loaded `Atlas`.
        meta: The case's `meta.json` mapping (used for `spacing` and, via
            `atlas_for_case`, `bbox` / `cropped_shape`).
        cropped: Whether `classes` is in the cropped frame -- passed straight
            through to `atlas_for_case`; see its docstring for the hazard of
            getting this wrong.
        regions: Region names to localise, each passed to
            `burden.region_mask`.
        knowledge: Optional `KnowledgeBase`, passed through to `localize_mask`.

    Returns:
        One concatenated `DataFrame` with a leading `region` column, then
        `_LOCALIZE_COLUMNS`.
    """
    spacing = tuple(float(s) for s in meta["spacing"])
    parcellation, _tissue = atlas_for_case(atlas, meta, cropped=cropped)

    tables = []
    for region in regions:
        mask = burden.region_mask(classes, region)  # validates `classes`
        table = localize_mask(mask, parcellation, atlas, spacing=spacing, knowledge=knowledge)
        table.insert(0, "region", region)
        tables.append(table)

    if not tables:
        table = _empty_localize_table()
        table.insert(0, "region", pd.Series(dtype="object"))
        return table

    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------- #
# Eloquence distance
# --------------------------------------------------------------------------- #


def eloquent_union_mask(atlas: Atlas, knowledge: KnowledgeBase) -> np.ndarray:
    """Boolean union of every atlas structure the knowledge base marks eloquent.

    Args:
        atlas: The loaded `Atlas`.
        knowledge: A `KnowledgeBase` from `load_knowledge`.

    Returns:
        `(D, H, W)` boolean array, `atlas.shape`.

    Raises:
        ValueError: If no structure is marked `"eloquent"` -- that means the
            knowledge base failed to load rather than that nothing in this
            atlas is eloquent.
    """
    eloquent_names = [name for name, value in knowledge.eloquence.items() if value == "eloquent"]
    if not eloquent_names:
        raise ValueError(
            "eloquent_union_mask: no structure in this knowledge base is marked 'eloquent'. "
            "This indicates the knowledge base failed to load, not that nothing is eloquent."
        )
    mask = np.zeros(atlas.shape, dtype=bool)
    for name in eloquent_names:
        mask |= atlas.structure_mask(name)
    return mask


def distance_to_eloquent(
    mask: np.ndarray,
    eloquent_mask: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Millimetre distance from a tumour mask to the nearest eloquent voxel.

    A purely GEOMETRIC distance in a template, not an assessment of any
    patient. The near-eloquent threshold (`KnowledgeBase.near_eloquent_mm`)
    is applied by the caller, not here -- Sawaya's grade II ("near-eloquent")
    is documented in the literature as ambiguous, so this function reports
    the measured distance rather than reproducing that grade.

    Args:
        mask: `(D, H, W)` boolean tumour mask.
        eloquent_mask: `(D, H, W)` boolean mask, e.g. from
            `eloquent_union_mask`.
        spacing: Voxel spacing in mm, `(D, H, W)` axis order.

    Returns:
        `0.0` when the two masks overlap. Otherwise the minimum Euclidean
        distance, in mm, from any tumour voxel to the nearest eloquent voxel.
        `NaN` when either mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    eloquent_mask = np.asarray(eloquent_mask, dtype=bool)
    if not mask.any() or not eloquent_mask.any():
        return float("nan")
    if np.any(mask & eloquent_mask):
        return 0.0

    distance_field = ndimage.distance_transform_edt(~eloquent_mask, sampling=spacing)
    return float(distance_field[mask].min())


# --------------------------------------------------------------------------- #
# Case summary
# --------------------------------------------------------------------------- #


def summarize_case(
    table: pd.DataFrame, knowledge: KnowledgeBase | None, *, top_n: int = 5
) -> dict[str, object]:
    """Flattens a localisation table into one CSV-ready row for a case.

    If `table` carries a `region` column (i.e. it came from `localize_case`),
    only the `"WT"` rows are used -- whole tumour is the superset of ET and
    TC, so it is the right scope for a case-level "which structures does the
    tumour touch" summary. `table` without a `region` column is used as-is.

    `distance_to_eloquent_mm` cannot be computed exactly from `table` alone
    (a tidy structure table carries no voxel coordinates): it is `0.0` when
    the table already shows overlap with an eloquent structure, and `NaN`
    otherwise. A caller wanting the exact non-zero mm distance for a
    non-overlapping case must call `distance_to_eloquent` directly, with the
    mask, and may overwrite this field with that result.

    Args:
        table: A `localize_mask` or `localize_case` output table.
        knowledge: Optional `KnowledgeBase`; without it, eloquence-derived
            fields report as unknown (`0` / `0.0` / `""`) rather than raising.
        top_n: How many of the table's largest-`frac_of_tumour` real
            structures (`"unlabelled"` excluded) are pooled by lobe to decide
            `dominant_lobe`.

    Returns:
        A flat dict of `float`, `int`, or `str` values only -- never an
        array, tuple, or `None` -- with keys `n_structures_involved`,
        `top_structure`, `top_frac_of_structure`, `most_displaced_structure`,
        `frac_unlabelled`, `n_eloquent_structures`, `eloquent_frac_of_tumour`,
        `distance_to_eloquent_mm`, `dominant_lobe`, `coverage_line`. An empty
        table gives counts `0`, strings `""`, fractions `NaN`.
    """
    if "region" in table.columns:
        wt_only = table[table["region"] == "WT"]
        working = wt_only if not wt_only.empty else table.iloc[0:0]
    else:
        working = table

    real_rows = working[working["structure"] != _UNLABELLED_NAME]
    unlabelled_rows = working[working["structure"] == _UNLABELLED_NAME]

    if working.empty:
        frac_unlabelled = float("nan")
    else:
        frac_unlabelled = (
            float(unlabelled_rows["frac_of_tumour"].sum()) if not unlabelled_rows.empty else 0.0
        )

    if real_rows.empty:
        n_structures_involved = 0
        top_structure = ""
        top_frac_of_structure = float("nan")
        most_displaced_structure = ""
        dominant_lobe = ""
    else:
        n_structures_involved = int(len(real_rows))

        top_row = real_rows.loc[real_rows["frac_of_tumour"].idxmax()]
        top_structure = str(top_row["structure"])
        top_frac_of_structure = float(top_row["frac_of_structure"])

        displaced_row = real_rows.loc[real_rows["frac_of_structure"].idxmax()]
        most_displaced_structure = str(displaced_row["structure"])

        top_by_tumour = real_rows.sort_values("frac_of_tumour", ascending=False).head(top_n)
        lobe_weights: dict[str, float] = {}
        for _, row in top_by_tumour.iterrows():
            lobe_name = str(row["lobe"])
            if lobe_name:
                lobe_weights[lobe_name] = lobe_weights.get(lobe_name, 0.0) + float(
                    row["frac_of_tumour"]
                )
        dominant_lobe = max(lobe_weights, key=lobe_weights.get) if lobe_weights else ""

    if knowledge is not None and not real_rows.empty:
        eloquent_rows = real_rows[real_rows["eloquence"] == "eloquent"]
        n_eloquent_structures = int(len(eloquent_rows))
        eloquent_frac_of_tumour = float(eloquent_rows["frac_of_tumour"].sum())
        distance_to_eloquent_mm = 0.0 if n_eloquent_structures > 0 else float("nan")
        coverage_line = knowledge.coverage_line(len(knowledge.eloquence))
    else:
        n_eloquent_structures = 0
        eloquent_frac_of_tumour = 0.0
        distance_to_eloquent_mm = float("nan")
        coverage_line = knowledge.coverage_line(len(knowledge.eloquence)) if knowledge else ""

    return {
        "n_structures_involved": n_structures_involved,
        "top_structure": top_structure,
        "top_frac_of_structure": top_frac_of_structure,
        "most_displaced_structure": most_displaced_structure,
        "frac_unlabelled": frac_unlabelled,
        "n_eloquent_structures": n_eloquent_structures,
        "eloquent_frac_of_tumour": eloquent_frac_of_tumour,
        "distance_to_eloquent_mm": distance_to_eloquent_mm,
        "dominant_lobe": dominant_lobe,
        "coverage_line": coverage_line,
    }
