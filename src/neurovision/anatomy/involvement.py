"""Group-level and tissue-level tumour involvement, plus epicentre naming.

Phase 3b of `docs/research/interpretable_pipeline_plan.md` §5. This module is
`neurovision.anatomy.localize`'s sibling: where `localize.py` answers "which
individual atlas structure does the tumour touch, and how badly", this module
answers three coarser questions that a single structure cannot:

    "does the tumour touch the ventricular system at all?"        (group_overlap)
    "does it touch deep white matter?"                             (group_overlap)
    "what fraction sits in grey matter / white matter / CSF?"      (tissue_overlap)
    "what one structure is this tumour centred on?"                (epicentre)

Like `localize.py`'s `frac_of_tumour` / `frac_of_structure` split, this module
reports two overlap fractions per group and they must never be conflated:

    <prefix>_frac_of_tumour = overlap_voxels / tumour_voxels  -- "how much of
        the tumour sits in this group?"
    <prefix>_frac_of_group  = overlap_voxels / group_voxels   -- "how much of
        this group does the tumour occupy?"

A lesion can put 3% of itself in the ventricles while filling 60% of them --
the first number alone would bury the more clinically relevant one.

This module loads the committed, versioned `knowledge/involvement_groups.yaml`
into an `InvolvementGroups`, validated against the loaded atlas the same way
`localize.load_knowledge` validates `knowledge/eloquence_map.yaml`: a stale or
mistyped structure or tissue name is caught at load time, not silently mapped
to nothing.

WHAT THESE NUMBERS ARE NOT. Every quantity here is an overlap with a healthy
template's structure position -- never a claim that a structure is destroyed,
compressed, or displaced, and never a claim of ependymal invasion (ependyma is
a cell layer, unresolvable at 1mm). See `knowledge/involvement_groups.yaml`'s
own header for the full statement, including its relationship to VASARI
(approximate and unverified, not a VASARI score).

Geometry: exactly `localize.py`'s convention. All masks passed here are
assumed to already be in the SAME frame as `parcellation` (the caller has
already run `localize.atlas_for_case`), so this module never crops anything
-- it raises `ValueError` naming both shapes on any mismatch, because a
cropped mask paired with an uncropped atlas view silently shifts every
overlap and epicentre answer by the crop offset and looks entirely
plausible.

This is pure array arithmetic: no model and no dependency on the
deep-learning stack, matching `neurovision.anatomy.atlas`,
`neurovision.anatomy.burden` and `neurovision.anatomy.localize` -- see
`tests/test_involvement.py::test_involvement_module_does_not_import_the_deep_learning_stack`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from neurovision.anatomy import burden
from neurovision.anatomy.atlas import Atlas
from neurovision.anatomy.burden import CaseGeometry

__all__ = [
    "InvolvementGroups",
    "load_involvement_groups",
    "group_label_ids",
    "group_mask",
    "group_overlap",
    "tissue_overlap",
    "epicentre",
    "involvement_profile",
    "INVOLVEMENT_FIELDS",
    "load_involvement_notes",
]

logger = logging.getLogger(__name__)

_UNLABELLED_NAME = "unlabelled"

# Every key `involvement_profile` emits, in emission order.
#
# Exists so a CONSUMER can recognise these columns without calling the
# function. `scripts/report.py` has to split one anatomy_summary.csv row into
# the localisation summary and the involvement profile, and it has no volume
# to run the profile on -- it joins CSVs. Recognising them by prefix at the
# call site would duplicate `report._classify_involvement_key`'s rules in a
# third place and let the two drift; a field added here and not there would
# silently land in the report's localisation summary instead of its
# involvement block.
#
# `tests/test_involvement.py` asserts this equals the actual output keys, so
# the constant cannot fall behind the function.
INVOLVEMENT_FIELDS: tuple[str, ...] = (
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
)


# --------------------------------------------------------------------------- #
# Knowledge artifact
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class InvolvementGroups:
    """The committed group definitions from `knowledge/involvement_groups.yaml`, validated.

    Attributes:
        version: The file's own `version` field.
        ventricle_structures: Merged atlas structure names forming the
            ventricular group (`groups.ventricles.structures`).
        deep_wm_structures: Merged atlas structure names forming the deep
            white-matter group (`groups.deep_white_matter.structures`).
        ventricle_missing: The `term` of each entry in
            `groups.ventricles.missing` -- source terms with no
            representable structure (e.g. "fourth ventricle").
        deep_wm_missing: The `term` of each entry in
            `groups.deep_white_matter.missing`.
        tissue_names: `{"cortical": "GM", "white_matter": "WM", "csf": "CSF"}`
            -- the report-field name mapped to the atlas's own tissue-class
            name, from `tissue:`.
        epicentre_search_radius_mm: `epicentre.search_radius_mm` -- how far
            `epicentre` searches for the nearest labelled structure when the
            centroid voxel itself is unlabelled.
        vasari_status: `relationship_to_vasari.status`
            (`"approximate_and_unverified"`).
        vasari_claim: `relationship_to_vasari.claim`, verbatim -- the
            sentence a report must use rather than asserting equivalence
            with VASARI from feature names alone.
    """

    version: int
    ventricle_structures: tuple[str, ...]
    deep_wm_structures: tuple[str, ...]
    ventricle_missing: tuple[str, ...]
    deep_wm_missing: tuple[str, ...]
    tissue_names: dict[str, str]
    epicentre_search_radius_mm: float
    vasari_status: str
    vasari_claim: str


def load_involvement_notes(path: str | Path) -> tuple[str, ...]:
    """Reads the group definitions' `missing:` entries as lower-bound sentences, with no atlas.

    Every `missing:` entry names something the source concept covers that this
    parcellation cannot represent -- the internal capsule for deep white
    matter, the fourth ventricle for the ventricles. Each one makes the
    corresponding overlap a LOWER BOUND, and a reader who is not told that
    will read a small number as evidence of little involvement rather than as
    evidence of limited coverage.

    Split out of `load_involvement_groups` for the same reason
    `localize.load_classification` is split out of `load_knowledge`:
    `scripts/report.py` joins already-written CSVs and has no atlas to
    validate structure names against, but it still has to carry these
    sentences into the artifact.

    Args:
        path: Path to `knowledge/involvement_groups.yaml`.

    Returns:
        One sentence per `missing:` entry, in file order.
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    notes: list[str] = []
    for group_key, group in (doc.get("groups") or {}).items():
        label = str(group.get("label", group_key))
        for entry in group.get("missing") or []:
            term = str(entry["term"]).strip()
            reason = " ".join(str(entry.get("reason", "")).split())
            sentence = f"{label} is a lower bound: '{term}' has no structure in this parcellation."
            if reason:
                sentence = f"{sentence} {reason}"
            notes.append(sentence)
    return tuple(notes)


def load_involvement_groups(path: str | Path, atlas: Atlas) -> InvolvementGroups:
    """Loads and validates `knowledge/involvement_groups.yaml` against an atlas.

    Args:
        path: Path to `knowledge/involvement_groups.yaml`.
        atlas: The loaded `Atlas` to validate structure and tissue names
            against.

    Returns:
        The parsed, validated `InvolvementGroups`.

    Raises:
        ValueError: If a `groups.*.structures` entry names a structure absent
            from `atlas.labels.names`, or if a `tissue.*` value is absent
            from `atlas.tissue_codes` -- same reasoning as
            `localize.load_knowledge`: a mapping against a structure or
            tissue class that is not the one being measured is exactly the
            failure this check exists to catch.
    """
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    atlas_names = set(atlas.labels.names)

    def _validate_structures(names: Sequence[str], group_label: str) -> tuple[str, ...]:
        for name in names:
            if name not in atlas_names:
                raise ValueError(
                    f"load_involvement_groups: {group_label} names structure '{name}', which "
                    "does not exist in the loaded atlas. A mapping against a structure that is "
                    "not the one being measured is exactly the failure this check exists to "
                    "catch."
                )
        return tuple(str(n) for n in names)

    ventricles_doc = doc["groups"]["ventricles"]
    deep_wm_doc = doc["groups"]["deep_white_matter"]

    ventricle_structures = _validate_structures(
        ventricles_doc["structures"], "groups.ventricles.structures"
    )
    deep_wm_structures = _validate_structures(
        deep_wm_doc["structures"], "groups.deep_white_matter.structures"
    )

    ventricle_missing = tuple(str(m["term"]) for m in ventricles_doc.get("missing", []))
    deep_wm_missing = tuple(str(m["term"]) for m in deep_wm_doc.get("missing", []))

    tissue_names = {str(k): str(v) for k, v in doc["tissue"].items()}
    for field_name, tissue_name in tissue_names.items():
        if tissue_name not in atlas.tissue_codes:
            raise ValueError(
                f"load_involvement_groups: tissue.{field_name} names tissue '{tissue_name}', "
                f"which is not a key of atlas.tissue_codes ({sorted(atlas.tissue_codes)})."
            )

    epicentre_doc = doc["epicentre"]
    vasari_doc = doc["relationship_to_vasari"]

    logger.info(
        "load_involvement_groups: %d ventricle structures, %d deep-WM structures, "
        "%d tissue classes.",
        len(ventricle_structures),
        len(deep_wm_structures),
        len(tissue_names),
    )

    return InvolvementGroups(
        version=int(doc["version"]),
        ventricle_structures=ventricle_structures,
        deep_wm_structures=deep_wm_structures,
        ventricle_missing=ventricle_missing,
        deep_wm_missing=deep_wm_missing,
        tissue_names=tissue_names,
        epicentre_search_radius_mm=float(epicentre_doc["search_radius_mm"]),
        vasari_status=str(vasari_doc["status"]),
        vasari_claim=str(vasari_doc["claim"]).strip(),
    )


# --------------------------------------------------------------------------- #
# Group masks
# --------------------------------------------------------------------------- #


def group_label_ids(atlas: Atlas, structure_names: Sequence[str]) -> tuple[int, ...]:
    """Unions the raw label ids of every named structure into one sorted tuple.

    Args:
        atlas: The loaded `Atlas`.
        structure_names: Merged atlas structure names, e.g.
            `InvolvementGroups.ventricle_structures`.

    Returns:
        Sorted, de-duplicated raw label ids across every named structure.

    Raises:
        ValueError: From `AtlasLabels.by_name`, if a name is unknown.
    """
    ids: set[int] = set()
    for name in structure_names:
        ids.update(atlas.labels.by_name(name).label_ids)
    return tuple(sorted(ids))


def group_mask(parcellation: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    """Boolean union mask of every voxel whose raw label is in `label_ids`.

    Args:
        parcellation: `(D, H, W)` raw atlas label array.
        label_ids: Raw label ids to union, e.g. from `group_label_ids`.

    Returns:
        `(D, H, W)` boolean array. All-`False` when `label_ids` is empty --
        an empty group is a valid (if degenerate) input, not an error.
    """
    if not label_ids:
        return np.zeros(parcellation.shape, dtype=bool)
    return np.isin(parcellation, label_ids)


def _check_shape(mask: np.ndarray, reference: np.ndarray, fn_name: str, ref_name: str) -> None:
    """Raises `ValueError` naming both shapes when `mask` and `reference` disagree."""
    if mask.shape != reference.shape:
        raise ValueError(
            f"{fn_name}: mask shape {mask.shape} != {ref_name} shape {reference.shape}. This "
            "usually means a cropped mask was paired with an uncropped atlas view (or vice "
            "versa) -- see `localize.atlas_for_case`."
        )


# --------------------------------------------------------------------------- #
# Group and tissue overlap
# --------------------------------------------------------------------------- #


def group_overlap(
    mask: np.ndarray,
    parcellation: np.ndarray,
    atlas: Atlas,
    structure_names: Sequence[str],
    geom: CaseGeometry,
    *,
    min_overlap_mm3: float,
    prefix: str,
) -> dict[str, float | bool]:
    """Overlap between a tumour mask and the union of a named group of structures.

    `<prefix>_frac_of_tumour` and `<prefix>_frac_of_group` answer different
    questions and must never be conflated -- see the module docstring.

    Args:
        mask: `(D, H, W)` boolean tumour region mask.
        parcellation: `(D, H, W)` raw atlas label array, in the SAME frame as
            `mask`.
        atlas: The loaded `Atlas`.
        structure_names: The group's member structures, e.g.
            `InvolvementGroups.ventricle_structures`.
        geom: Geometry supplying voxel volume.
        min_overlap_mm3: The minimum overlap volume, in mm^3, for
            `<prefix>_contact` to be `True`.
        prefix: Field-name prefix, e.g. `"ventricle"` or `"deep_wm"`.

    Returns:
        Dict with keys `<prefix>_overlap_mm3`, `<prefix>_frac_of_tumour`,
        `<prefix>_frac_of_group`, `<prefix>_contact`. For an empty `mask`,
        every fraction is `NaN`, `overlap_mm3` is `0.0`, `contact` is
        `False` -- "no tumour" is not the same measurement as "tumour that
        touches nothing", and must never collapse to the same `0.0` a real
        observation would produce. For a non-empty `mask` overlapping an
        empty group (every member structure absent from this crop),
        `<prefix>_frac_of_group` is `NaN`; the rest are still computed.

    Raises:
        ValueError: If `mask.shape != parcellation.shape`, naming both.
    """
    mask = np.asarray(mask, dtype=bool)
    _check_shape(mask, parcellation, "group_overlap", "parcellation")

    mask_voxels = int(mask.sum())
    if mask_voxels == 0:
        return {
            f"{prefix}_overlap_mm3": 0.0,
            f"{prefix}_frac_of_tumour": float("nan"),
            f"{prefix}_frac_of_group": float("nan"),
            f"{prefix}_contact": False,
        }

    label_ids = group_label_ids(atlas, structure_names)
    group = group_mask(parcellation, label_ids)

    overlap = mask & group
    overlap_voxels = int(overlap.sum())
    overlap_mm3 = burden.volume_mm3(overlap, geom)
    group_voxels = int(group.sum())

    frac_of_tumour = overlap_voxels / mask_voxels
    frac_of_group = (overlap_voxels / group_voxels) if group_voxels > 0 else float("nan")
    contact = bool(overlap_mm3 >= min_overlap_mm3)

    return {
        f"{prefix}_overlap_mm3": overlap_mm3,
        f"{prefix}_frac_of_tumour": frac_of_tumour,
        f"{prefix}_frac_of_group": frac_of_group,
        f"{prefix}_contact": contact,
    }


def tissue_overlap(
    mask: np.ndarray,
    tissue: np.ndarray | None,
    atlas: Atlas,
    tissue_names: Mapping[str, str],
    geom: CaseGeometry,
) -> dict[str, float]:
    """Tissue-class composition of a tumour mask, from the atlas tissue map.

    The tissue-level complement to `group_overlap`'s structure-based groups:
    the parcellation covers grey matter only, so a structure-only view
    silently drops white matter -- roughly a third of a real glioma (see
    `localize.py`'s `"unlabelled"` row).

    Args:
        mask: `(D, H, W)` boolean tumour region mask.
        tissue: `(D, H, W)` `uint8` atlas tissue-code array (`Atlas.tissue`),
            in the SAME frame as `mask`, or `None` when the atlas has no
            tissue map loaded.
        atlas: The loaded `Atlas`, supplying `tissue_codes`.
        tissue_names: `InvolvementGroups.tissue_names`, e.g.
            `{"cortical": "GM", "white_matter": "WM", "csf": "CSF"}`.
        geom: Unused directly -- kept for API symmetry with the other
            per-mask functions, since every value here is a voxel-count
            ratio and the (identical) voxel volume cancels out of both
            numerator and denominator.

    Returns:
        Dict with one `<key>_frac_of_tumour` per entry of `tissue_names`,
        plus `outside_tissue_frac_of_tumour` (voxels where `tissue == 0`).
        These four sum to exactly `1.0` for a non-empty `mask` -- every
        tumour voxel is either one of the three named tissue classes or
        outside the tissue map entirely, with none double-counted or
        dropped. All `NaN` when `mask` is empty or when `tissue is None`
        (the latter also logs one WARNING; an atlas with no tissue map is a
        valid configuration, not an error).

    Raises:
        ValueError: If `tissue is not None` and `mask.shape != tissue.shape`,
            naming both.
    """
    del geom  # kept for API symmetry only; see docstring

    mask = np.asarray(mask, dtype=bool)
    keys = list(tissue_names) + ["outside_tissue"]

    if tissue is None:
        logger.warning(
            "tissue_overlap: no tissue map loaded for this atlas (tissue=None); reporting all "
            "tissue fractions as NaN."
        )
        return {f"{key}_frac_of_tumour": float("nan") for key in keys}

    _check_shape(mask, tissue, "tissue_overlap", "tissue")

    mask_voxels = int(mask.sum())
    if mask_voxels == 0:
        return {f"{key}_frac_of_tumour": float("nan") for key in keys}

    result: dict[str, float] = {}
    for field_name, tissue_class in tissue_names.items():
        code = atlas.tissue_codes[tissue_class]
        voxels = int(((tissue == code) & mask).sum())
        result[f"{field_name}_frac_of_tumour"] = voxels / mask_voxels

    outside_voxels = int(((tissue == 0) & mask).sum())
    result["outside_tissue_frac_of_tumour"] = outside_voxels / mask_voxels
    return result


# --------------------------------------------------------------------------- #
# Epicentre
# --------------------------------------------------------------------------- #


def _nearest_labelled_structure(
    parcellation: np.ndarray,
    atlas: Atlas,
    centroid_coords: tuple[float, float, float],
    center_idx: tuple[int, int, int],
    spacing: tuple[float, float, float],
    search_radius_mm: float,
) -> tuple[str, float]:
    """Local-window search for the nearest labelled structure to a centroid.

    Deliberately NOT a whole-volume distance transform:
    `scipy.ndimage.distance_transform_edt` over a full ~240x240x155
    parcellation costs seconds and this runs per case, per region. Instead a
    cube of half-width `ceil(search_radius_mm / min(spacing))` voxels around
    `center_idx` is sliced out -- sized so it fully contains the mm-radius
    sphere on every axis, even the coarsest -- and searched directly. The
    returned distance is exact within the radius and `NaN` beyond it.

    Args:
        parcellation: `(D, H, W)` raw atlas label array.
        atlas: The loaded `Atlas`.
        centroid_coords: The float `(i, j, k)` centroid, for the mm distance.
        center_idx: The rounded, clipped integer centroid voxel, for slicing.
        spacing: Voxel spacing in mm, `(D, H, W)` axis order.
        search_radius_mm: Search radius in mm.

    Returns:
        `(structure_name, distance_mm)`. `("unlabelled", nan)` when nothing
        labelled falls within the radius.
    """
    half_width = int(math.ceil(search_radius_mm / min(spacing)))
    shape = parcellation.shape
    lo = tuple(max(0, center_idx[a] - half_width) for a in range(3))
    hi = tuple(min(shape[a], center_idx[a] + half_width + 1) for a in range(3))

    sub = parcellation[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    if sub.size == 0:
        return _UNLABELLED_NAME, float("nan")

    unique_vals = np.unique(sub)
    unique_vals = unique_vals[unique_vals != 0]
    if unique_vals.size == 0:
        return _UNLABELLED_NAME, float("nan")

    # Only the raw values actually present in this small local cube are
    # looked up -- never `AtlasLabels.lookup_array`, which validates every
    # structure's label ids against a single max_id and would raise on any
    # structure elsewhere in the atlas with a larger id than this cube's max.
    val_to_name = {}
    for raw_value in unique_vals.tolist():
        name = atlas.labels.name_for_id(int(raw_value))
        if name and name != atlas.labels.unmapped_name:
            val_to_name[int(raw_value)] = name

    if not val_to_name:
        return _UNLABELLED_NAME, float("nan")

    valid = np.isin(sub, list(val_to_name))
    if not valid.any():
        return _UNLABELLED_NAME, float("nan")

    local_coords = np.argwhere(valid).astype(np.float64)
    offset = np.array(lo, dtype=np.float64)
    global_coords = local_coords + offset

    diffs = (global_coords - np.array(centroid_coords)) * np.array(spacing)
    dists = np.sqrt((diffs**2).sum(axis=1))
    best = int(np.argmin(dists))
    best_dist = float(dists[best])

    if best_dist > search_radius_mm:
        return _UNLABELLED_NAME, float("nan")

    best_local = tuple(int(v) for v in local_coords[best])
    raw_value = int(sub[best_local])
    return val_to_name[raw_value], best_dist


def epicentre(
    mask: np.ndarray,
    parcellation: np.ndarray,
    atlas: Atlas,
    geom: CaseGeometry,
    *,
    search_radius_mm: float,
    lobe: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The single atlas structure a tumour region is centred on -- a location, not an origin.

    A glioma's true origin is not recoverable from one timepoint, so this
    reports the structure containing the region's centroid voxel, and never
    claims more.

    Args:
        mask: `(D, H, W)` boolean tumour region mask.
        parcellation: `(D, H, W)` raw atlas label array, in the SAME frame as
            `mask`.
        atlas: The loaded `Atlas`.
        geom: Geometry supplying the centroid computation, `midline_index`,
            `left_is_high_index`, and `spacing`.
        search_radius_mm: How far to search for the nearest labelled
            structure when the centroid voxel itself is unlabelled --
            common, since AAL parcellates grey matter and a glioma centroid
            frequently sits in white matter.
        lobe: Optional structure name -> lobe mapping (e.g.
            `KnowledgeBase.lobe`). When absent, or when the resolved
            structure has no entry, `epicentre_lobe` is `""`.

    Returns:
        Dict with keys `epicentre_structure` (`str`), `epicentre_exact`
        (`bool`), `epicentre_distance_mm` (`float`), `epicentre_laterality`
        (`str`, the STRUCTURE's own `AtlasStructure.laterality`),
        `epicentre_side` (`str`, `"left"` / `"right"` / `"midline"`,
        computed INDEPENDENTLY from `geom` rather than from
        `epicentre_laterality` -- a disagreement between the two is
        diagnostic information, not a bug to paper over), and
        `epicentre_lobe` (`str`).

        For an empty `mask`: `epicentre_structure` is `"unlabelled"`,
        `epicentre_exact` is `False`, `epicentre_distance_mm` is `NaN`, and
        every string field is `""`.

    Raises:
        ValueError: If `mask.shape != parcellation.shape`, naming both.
    """
    mask = np.asarray(mask, dtype=bool)
    _check_shape(mask, parcellation, "epicentre", "parcellation")

    if not mask.any():
        return {
            "epicentre_structure": _UNLABELLED_NAME,
            "epicentre_exact": False,
            "epicentre_distance_mm": float("nan"),
            "epicentre_laterality": "",
            "epicentre_side": "",
            "epicentre_lobe": "",
        }

    centroid_coords = burden.centroid(mask, geom)

    shape = parcellation.shape
    center_idx = tuple(
        int(min(max(round(v), 0), shape[axis] - 1)) for axis, v in enumerate(centroid_coords)
    )

    center_raw = int(parcellation[center_idx])
    center_name = atlas.labels.name_for_id(center_raw)
    exact = bool(center_name) and center_name != atlas.labels.unmapped_name

    if exact:
        structure = center_name
        distance = 0.0
    else:
        structure, distance = _nearest_labelled_structure(
            parcellation, atlas, centroid_coords, center_idx, geom.spacing, search_radius_mm
        )

    if structure == _UNLABELLED_NAME:
        structure_laterality = ""
    else:
        structure_laterality = atlas.labels.by_name(structure).laterality

    axis0 = centroid_coords[0]
    diff = axis0 - geom.midline_index
    if abs(diff) <= 0.5:
        side = "midline"
    elif geom.left_is_high_index:
        side = "left" if diff > 0 else "right"
    else:
        side = "right" if diff > 0 else "left"

    lobe_name = ""
    if lobe is not None and structure in lobe:
        lobe_name = lobe[structure]

    return {
        "epicentre_structure": structure,
        "epicentre_exact": exact,
        "epicentre_distance_mm": distance,
        "epicentre_laterality": structure_laterality,
        "epicentre_side": side,
        "epicentre_lobe": lobe_name,
    }


# --------------------------------------------------------------------------- #
# Combined profile
# --------------------------------------------------------------------------- #


def involvement_profile(
    mask: np.ndarray,
    parcellation: np.ndarray,
    tissue: np.ndarray | None,
    atlas: Atlas,
    groups: InvolvementGroups,
    geom: CaseGeometry,
    *,
    min_overlap_mm3: float,
    lobe: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Runs `group_overlap` (ventricles, deep white matter), `tissue_overlap` and `epicentre`
    for one tumour region and merges the results. Adds nothing of its own.

    Args:
        mask: `(D, H, W)` boolean tumour region mask.
        parcellation: `(D, H, W)` raw atlas label array, in the SAME frame as
            `mask`.
        tissue: `(D, H, W)` atlas tissue-code array, in the SAME frame as
            `mask`, or `None`.
        atlas: The loaded `Atlas`.
        groups: The validated `InvolvementGroups` from
            `load_involvement_groups`.
        geom: Geometry for volumes, centroid, and the left/right convention.
        min_overlap_mm3: Passed through to both `group_overlap` calls.
        lobe: Optional structure name -> lobe mapping, passed through to
            `epicentre`.

    Returns:
        One flat, merged dict: 4 `ventricle_*` keys, 4 `deep_wm_*` keys, 4
        tissue keys, 6 `epicentre_*` keys -- 18 total, no overlap.

    Raises:
        ValueError: If `mask.shape != parcellation.shape`, naming both.
    """
    mask = np.asarray(mask, dtype=bool)
    _check_shape(mask, parcellation, "involvement_profile", "parcellation")

    ventricle = group_overlap(
        mask,
        parcellation,
        atlas,
        groups.ventricle_structures,
        geom,
        min_overlap_mm3=min_overlap_mm3,
        prefix="ventricle",
    )
    deep_wm = group_overlap(
        mask,
        parcellation,
        atlas,
        groups.deep_wm_structures,
        geom,
        min_overlap_mm3=min_overlap_mm3,
        prefix="deep_wm",
    )
    tissue_parts = tissue_overlap(mask, tissue, atlas, groups.tissue_names, geom)
    epi = epicentre(
        mask,
        parcellation,
        atlas,
        geom,
        search_radius_mm=groups.epicentre_search_radius_mm,
        lobe=lobe,
    )

    parts = (ventricle, deep_wm, tissue_parts, epi)
    merged: dict[str, object] = {}
    for part in parts:
        merged.update(part)
    assert len(merged) == sum(len(part) for part in parts), (
        "involvement_profile: key collision between group_overlap / tissue_overlap / "
        "epicentre outputs -- these must never share a key."
    )
    return merged
