"""Assembles and renders the per-case structured report (Phase 4).

This module runs no model and reads no NIfTI: it is pure assembly over the
already-computed artifacts earlier phases produce --
`neurovision.anatomy.burden.burden_profile`'s flat dict,
`neurovision.anatomy.localize.localize_case`'s structure table, and
`neurovision.anatomy.localize.summarize_case`'s summary dict -- plus a small
set of caller-supplied strings (evidence, citation, coverage line). It writes
one JSON file and, optionally, one Markdown rendering per case.

The reason this module exists is not formatting. It is
`docs/research/interpretable_pipeline_plan.md` section 2: the project has
exactly two kinds of claim it is allowed to ship (geometric, computed
deterministically from a mask and a label map; and referential, a lookup into
a named published classification where the source owns the claim), and this
is the one place every one of those claims is finally assembled into
something a reader sees. So the honesty rules below are not presentation
choices, they are the module's actual job:

    - a non-diagnostic disclaimer is a required field, not optional
    - a `not_claimed` block names what this artifact refuses to say, and why
    - every anatomical claim carries the atlas name, version, and the
      mass-effect caveat (a healthy-brain atlas mislabels displaced tissue
      exactly where the lesion is, so involvement is approximate)
    - the eloquence block carries its verbatim evidence sentence and
      citation, and states plainly that the source owns the classification
      and this project owns only the mapping onto it
    - the knowledge coverage line is a required field, so a thin knowledge
      base is visible in the output rather than hidden by it

No function or deficit text appears anywhere in this module or in anything
it renders. `tests/test_report.py` scans both output formats for the
forbidden vocabulary and fails if any of it appears outside the
`not_claimed` block itself.

This module has no dependency on the deep-learning stack and no dependency
on numpy -- see `json_safe`, which duck-types numpy-like scalars via
`.item()` instead of importing the library that produces them. The same
reasoning as `neurovision.visualization.tables`; see
`tests/test_report.py::test_report_module_has_no_deep_learning_or_array_dependency`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "REPORT_VERSION",
    "DISCLAIMER",
    "NOT_CLAIMED",
    "MASS_EFFECT_CAVEAT",
    "Provenance",
    "build_report",
    "render_markdown",
    "write_report",
    "json_safe",
]

logger = logging.getLogger(__name__)

REPORT_VERSION: int = 1

DISCLAIMER: str = (
    "This report is a research and educational decision-support artifact. It is not a "
    "diagnostic tool and must not be used, alone or together with any other information, to "
    "make or support a clinical decision about any patient."
)

MASS_EFFECT_CAVEAT: str = (
    "This atlas describes healthy-brain anatomy. A tumour physically displaces the tissue "
    "around it, so near a lesion the atlas can mislabel tissue that has moved out of its usual "
    "position -- structure involvement reported here is therefore approximate, not a direct "
    "measurement of this patient's own anatomy."
)

_SOURCE_OWNS_CLAIM: str = (
    "This eloquence classification is a lookup into a named, published source; our own "
    "contribution is only the mapping from an atlas structure to that source's list, not a "
    "claim about this patient's anatomy or function."
)

# (what we refuse to say, why) -- six items, one per row of
# `docs/research/interpretable_pipeline_plan.md` section 2's "Explicitly NOT
# in scope" table, minus the diagnostic-use item (that is what DISCLAIMER
# covers). Every reason below is allowed to use the forbidden vocabulary --
# this IS the block the forbidden-substring scan excludes -- and nowhere else
# in this module or its rendered output may.
NOT_CLAIMED: tuple[tuple[str, str], ...] = (
    (
        "cell type",
        "MRI resolves millimetre-scale tissue, not individual cells; the origin cell type of a "
        "glioma is a histopathology question this pipeline has no way to answer.",
    ),
    (
        "WHO grade",
        "WHO CNS5 grading is integrated: it needs histology plus molecular markers (IDH, "
        "1p/19q, ATRX, TERT, CDKN2A/B) that are not present anywhere in this dataset.",
    ),
    (
        "tumour stage",
        "Diffuse gliomas are not staged; there is no TNM staging system for them, only "
        "grading -- which this pipeline also does not attempt, for the same reason.",
    ),
    (
        "prognosis or outcome",
        "This dataset carries no clinical outcomes to validate a prognosis against, so none "
        "is computed or implied.",
    ),
    (
        "an eloquence assessment",
        "Only a geometric overlap with, and millimetre distance to, structures a named "
        "published classification calls eloquent is reported; the source owns that "
        "classification and this pipeline owns only the mapping onto it.",
    ),
    (
        "any deficit the patient has or will experience",
        "A deficit claim is unvalidatable against the outcomes data this project has, so no "
        "deficit or functional-loss text is generated anywhere in this artifact.",
    ),
)

_SEGMENTATION_SOURCES: frozenset[str] = frozenset({"prediction", "label"})


@dataclass(frozen=True)
class Provenance:
    """Where every value in a report came from, so a report can be traced back to its inputs.

    Attributes:
        atlas_name: E.g. `"SRI24/TZO"`.
        atlas_version: The atlas release/version string.
        atlas_source: Where the atlas was obtained from (e.g. a NITRC URL).
        atlas_licence: The atlas's licence (e.g. `"CC-BY-SA"`).
        knowledge_versions: Committed knowledge-file name -> its `version`
            field, e.g. `{"eloquence_map": 1, "aal_lobes": 1}`.
        segmentation_source: `"prediction"` or `"label"` -- whether the mask
            behind this report came from a model or from ground truth.
        segmentation_dir: The directory the segmentation was read from, or
            `None` when not applicable.
        code_revision: A git SHA, or `None` when not recorded.
        generated_utc: An ISO-8601 UTC timestamp string, supplied by the
            caller (this module does not read the clock itself).

    Raises:
        ValueError: If `segmentation_source` is not `"prediction"` or
            `"label"`.
    """

    atlas_name: str
    atlas_version: str
    atlas_source: str
    atlas_licence: str
    knowledge_versions: dict[str, int]
    segmentation_source: str
    segmentation_dir: str | None
    code_revision: str | None
    generated_utc: str

    def __post_init__(self) -> None:
        if self.segmentation_source not in _SEGMENTATION_SOURCES:
            raise ValueError(
                "Provenance.segmentation_source must be one of "
                f"{sorted(_SEGMENTATION_SOURCES)}, got {self.segmentation_source!r}."
            )


# --------------------------------------------------------------------------- #
# json_safe -- everything else in this module depends on it
# --------------------------------------------------------------------------- #


def json_safe(value: object) -> object:
    """Recursively converts `value` into something `json.dumps(..., allow_nan=False)` accepts.

    `burden_profile` and pandas both hand back numpy-like scalars
    (`np.float64`, `np.int64`, `np.bool_`) and non-finite floats, which
    `json.dump` either raises on or (for NaN/inf) writes as bare `NaN`/
    `Infinity` -- invalid JSON that many parsers reject. This function never
    imports numpy: it duck-types a numpy-like scalar via `hasattr(value,
    "item")` and detects non-finite floats via `math.isfinite`, so the
    module's import list stays stdlib plus pandas.

    `bool` is checked before `int` (a Python `bool` IS an `int` subclass) so
    a boolean never becomes a bare `0`/`1`.

    Args:
        value: Any value appearing inside a report dict -- a mapping, list,
            tuple, dataclass, `Path`, string, bool, int, float, numpy-like
            scalar, `pandas.NA`/`NaT`, or `None`.

    Returns:
        An equivalent value built only from `dict`, `list`, `str`, `bool`,
        `int`, finite `float`, and `None`.
    """
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None:
        return None
    # A numpy-like scalar (np.float64, np.int64, np.bool_, a 0-d pandas
    # value) or a stub exposing `.item()` -- unwrap and re-run through this
    # same function, since the unwrapped value still needs the finite-float
    # and bool checks above.
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    # `pandas.NA`, `pandas.NaT`, and any other pandas missing-value sentinel
    # that reaches here without an `.item()` method.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# --------------------------------------------------------------------------- #
# Burden regrouping
# --------------------------------------------------------------------------- #

# Reporting order: how the burden sub-blocks are grouped AND rendered.
_BURDEN_BLOCKS: tuple[str, ...] = (
    "volumes",
    "fractions",
    "shape",
    "multifocality",
    "laterality",
    "centroid",
    "other",
)

# Laterality prefixes are checked FIRST, ahead of both `vol_*` (volumes) and
# `frac_*` (fractions), because `vol_right_ET_mm3` and `frac_left_WT` both
# match those broader prefixes too. This is the documented precedence for
# `burden_profile`'s overlapping key names: laterality wins.
_LATERALITY_PREFIXES: tuple[str, ...] = (
    "vol_right_",
    "vol_left_",
    "frac_left_",
    "frac_contralateral_",
    "dominant_side_",
)
_SHAPE_PREFIXES: tuple[str, ...] = ("surface_", "sphericity_")


def _classify_burden_key(key: str) -> str:
    """Which report sub-block one `burden_profile` key belongs to.

    Checked in a fixed precedence order, because several of
    `burden_profile`'s own key prefixes overlap:
    `vol_right_{region}_mm3` / `frac_left_{region}` are volume- and
    fraction-shaped names that are actually laterality fields, and
    `vol_largest_component_{region}_mm3` / `largest_component_frac_{region}`
    are volume- and fraction-shaped names that are actually multifocality
    fields. The order below -- laterality, then multifocality, then shape,
    then centroid, then fractions, then volumes -- resolves every overlap the
    same way every time: the more specific block wins over the more generic
    one. Anything matching none of them lands in `"other"` rather than being
    dropped, because silently discarding a field the caller computed is
    worse than an untidy block.
    """
    if key.startswith(_LATERALITY_PREFIXES):
        return "laterality"
    if key.startswith("n_components_") or "_component_" in key:
        return "multifocality"
    if key.startswith(_SHAPE_PREFIXES):
        return "shape"
    if key.startswith("centroid_"):
        return "centroid"
    if key.startswith("frac_") or key.startswith("ratio_"):
        return "fractions"
    if key.startswith("vol_"):
        return "volumes"
    return "other"


def _group_burden(burden: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Regroups a flat `burden_profile` dict into the named sub-blocks a person can read.

    A flat 50-plus-key dict is not a report; grouping by what the key
    actually measures is. See `_classify_burden_key` for the precedence rule
    that resolves overlapping prefixes.
    """
    grouped: dict[str, dict[str, object]] = {name: {} for name in _BURDEN_BLOCKS}
    for key, value in burden.items():
        grouped[_classify_burden_key(key)][key] = value
    return grouped


# --------------------------------------------------------------------------- #
# Eloquence proximity (purely geometric, from distance + threshold -- no
# verdict is derived from these; the classification NAME is a caller-supplied
# string identifying the published source, see `classification_name` below)
# --------------------------------------------------------------------------- #


def _is_missing(value: object) -> bool:
    """True for `None` or a non-finite float -- the two "no measurement" shapes this module sees."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _near_eloquent(distance_mm: object, threshold_mm: float) -> bool:
    """`distance_mm <= threshold_mm`, and `False` -- never `None` -- when `distance_mm` is missing.

    A missing distance means "we do not know", which must not read as either
    "near" or "far" by accident; `False` is the conservative (non-alarming)
    default and is a definite boolean either way, never `None`.
    """
    if _is_missing(distance_mm):
        return False
    return float(distance_mm) <= float(threshold_mm)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# build_report
# --------------------------------------------------------------------------- #


def build_report(
    case_id: str,
    burden: Mapping[str, object],
    anatomy_table: pd.DataFrame,
    anatomy_summary: Mapping[str, object],
    provenance: Provenance,
    *,
    evidence: str,
    citation: str,
    classification_name: str,
    coverage_line: str,
    coverage_gaps: Sequence[str],
    near_eloquent_mm: float,
    top_n: int = 10,
) -> dict:
    """Assembles one case's report dict from already-computed artifacts.

    Args:
        case_id: The case identifier. Must be non-empty.
        burden: A `neurovision.anatomy.burden.burden_profile` output (or an
            equivalent flat mapping).
        anatomy_table: A `neurovision.anatomy.localize.localize_case` (or
            `localize_mask`) output. When it carries a `region` column, only
            the `WT` rows are used -- whole tumour is the right scope for a
            case-level anatomical summary. An empty table is valid input.
        anatomy_summary: A `neurovision.anatomy.localize.summarize_case`
            output (or an equivalent flat mapping); supplies
            `n_structures_involved`, `frac_unlabelled`, and
            `distance_to_eloquent_mm`.
        provenance: Where every value in this report came from.
        evidence: The verbatim sentence backing the eloquence classification.
            Must be non-empty.
        citation: The citation for that classification. Must be non-empty.
        classification_name: The NAME of the published classification system
            being looked up (e.g. `"Sawaya eloquence grading"`), not a
            computed verdict about this patient -- this project only maps
            atlas structures onto that source's list, it does not grade
            anyone. Must be non-empty.
        coverage_line: The knowledge-base coverage summary (e.g. from
            `KnowledgeBase.coverage_line`). Must be non-empty.
        coverage_gaps: Source terms with no representable structure in this
            parcellation.
        near_eloquent_mm: The configured near-eloquent distance threshold, in
            mm.
        top_n: How many `anatomy_table` rows (sorted by `frac_of_structure`
            descending) to keep in `anatomy.structures`.

    Returns:
        A plain dict, field order `report_version, case_id, generated_utc,
        disclaimer, not_claimed, burden, anatomy, eloquence, provenance`.
        Values may still include numpy-like scalars and non-finite floats
        pulled from `anatomy_table` / `burden` -- pass the result through
        `json_safe` before serialising.

    Raises:
        ValueError: If `case_id`, `evidence`, `citation`,
            `classification_name`, or `coverage_line` is empty.
    """
    if not case_id:
        raise ValueError("build_report: case_id must be non-empty.")
    if not evidence:
        raise ValueError(
            "build_report: evidence must be non-empty -- an eloquence block without its "
            "evidence is exactly the unauditable artifact this design forbids."
        )
    if not citation:
        raise ValueError("build_report: citation must be non-empty.")
    if not classification_name:
        raise ValueError("build_report: classification_name must be non-empty.")
    if not coverage_line:
        raise ValueError("build_report: coverage_line must be non-empty.")

    if "region" in anatomy_table.columns:
        region_used: str | None = "WT"
        working = anatomy_table[anatomy_table["region"] == "WT"]
    else:
        region_used = None
        working = anatomy_table

    if working.empty:
        structures: list[dict[str, object]] = []
    else:
        sorted_table = working.sort_values("frac_of_structure", ascending=False, na_position="last")
        structures = sorted_table.head(top_n).to_dict("records")

    anatomy_block = {
        "atlas": {"name": provenance.atlas_name, "version": provenance.atlas_version},
        "caveat": MASS_EFFECT_CAVEAT,
        "coverage_line": coverage_line,
        "region": region_used,
        "structures": structures,
        "n_structures_involved": anatomy_summary.get("n_structures_involved"),
        "frac_unlabelled": anatomy_summary.get("frac_unlabelled"),
    }

    distance_mm = anatomy_summary.get("distance_to_eloquent_mm")
    eloquent_involved: list[dict[str, object]] = []
    if not working.empty and "eloquence" in working.columns:
        for _, row in working[working["eloquence"] == "eloquent"].iterrows():
            eloquent_involved.append(
                {
                    "structure": row.get("structure"),
                    "laterality": row.get("laterality"),
                    "frac_of_tumour": row.get("frac_of_tumour"),
                    "frac_of_structure": row.get("frac_of_structure"),
                }
            )

    eloquence_block = {
        "classification": classification_name,
        "citation": citation,
        "evidence": evidence,
        "source_owns_claim": _SOURCE_OWNS_CLAIM,
        "involved": eloquent_involved,
        "distance_mm": distance_mm,
        "near_eloquent_threshold_mm": float(near_eloquent_mm),
        "near_eloquent": _near_eloquent(distance_mm, near_eloquent_mm),
        "coverage_gaps": list(coverage_gaps),
    }

    return {
        "report_version": REPORT_VERSION,
        "case_id": case_id,
        "generated_utc": provenance.generated_utc,
        "disclaimer": DISCLAIMER,
        "not_claimed": NOT_CLAIMED,
        "burden": _group_burden(burden),
        "anatomy": anatomy_block,
        "eloquence": eloquence_block,
        "provenance": asdict(provenance),
    }


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

_BURDEN_BLOCK_TITLES: dict[str, str] = {
    "volumes": "Volumes",
    "fractions": "Composition",
    "shape": "Shape",
    "multifocality": "Multifocality",
    "laterality": "Laterality",
    "centroid": "Centroid (voxel index)",
    "other": "Other",
}


def _cell(value: object) -> str:
    """Generic scalar-to-text renderer: `"n/a"` for missing or empty, else `str(value)`."""
    safe = json_safe(value)
    if safe is None:
        return "n/a"
    if isinstance(safe, str):
        return safe if safe else "n/a"
    if isinstance(safe, bool):
        return str(safe)
    if isinstance(safe, int):
        return str(safe)
    if isinstance(safe, float):
        return f"{safe:.3f}"
    return str(safe)


def _fmt_volume(value: object, unit: str = "mm³") -> str:
    """Volumes/areas as integers with a unit -- a third decimal on a voxel count is noise."""
    safe = json_safe(value)
    if safe is None:
        return "n/a"
    return f"{float(safe):,.0f} {unit}"


def _fmt_fraction(value: object) -> str:
    """Fractions as a percentage to one decimal place."""
    safe = json_safe(value)
    if safe is None:
        return "n/a"
    return f"{float(safe) * 100.0:.1f}%"


def _fmt_distance(value: object) -> str:
    """Distances in millimetres to one decimal place."""
    safe = json_safe(value)
    if safe is None:
        return "n/a"
    return f"{float(safe):.1f} mm"


def _format_burden_value(key: str, value: object) -> str:
    """Formats one burden value using its key name as the only formatting hint available here."""
    safe = json_safe(value)
    if safe is None:
        return "n/a"
    if isinstance(safe, str):
        return safe if safe else "n/a"
    if isinstance(safe, bool):
        return str(safe)
    if isinstance(safe, int):
        return str(safe)
    if key.startswith("frac_") or key.startswith("largest_component_frac") or "_frac_" in key:
        return f"{safe * 100.0:.1f}%"
    if key.startswith("ratio_"):
        return f"{safe:.2f}"
    if key.endswith("_mm3"):
        return f"{safe:,.0f} mm³"
    if key.endswith("_mm2"):
        return f"{safe:,.0f} mm²"
    if key.startswith("centroid_"):
        return f"{safe:.1f}"
    return f"{safe:.3f}"


def _pipe_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A hand-rolled Markdown pipe table.

    `pandas.DataFrame.to_markdown` needs the `tabulate` package, which is not
    in `requirements.txt`; this avoids adding a dependency for one table
    format.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: Mapping) -> str:
    """Renders a `build_report` dict as a readable Markdown document.

    Section order: title, the disclaimer (first, before any content), the
    tumour burden profile, anatomical involvement (atlas name/version and
    the mass-effect caveat immediately under the heading), the eloquence
    reference (verbatim evidence as a blockquote, plus citation), what this
    report refuses to claim, and provenance last.

    Args:
        report: A `build_report` output (or anything with the same shape).

    Returns:
        The Markdown text.
    """
    lines: list[str] = []

    lines.append(f"# Structured Report -- Case {_cell(report['case_id'])}")
    lines.append("")
    lines.append(
        f"*Report schema version {_cell(report['report_version'])}, generated "
        f"{_cell(report['generated_utc'])}.*"
    )
    lines.append("")
    lines.append("> " + str(report["disclaimer"]))
    lines.append("")

    # --- Burden -------------------------------------------------------- #
    lines.append("## Tumour Burden Profile")
    lines.append("")
    burden = report["burden"]
    for block_name in _BURDEN_BLOCKS:
        block = burden.get(block_name, {})
        if not block:
            continue
        lines.append(f"### {_BURDEN_BLOCK_TITLES[block_name]}")
        for key in sorted(block):
            lines.append(f"- **{key}**: {_format_burden_value(key, block[key])}")
        lines.append("")

    # --- Anatomy --------------------------------------------------------- #
    anatomy = report["anatomy"]
    lines.append("## Anatomical Involvement")
    lines.append("")
    atlas = anatomy["atlas"]
    lines.append(f"Atlas: **{_cell(atlas['name'])} {_cell(atlas['version'])}**.")
    lines.append("")
    lines.append(str(anatomy["caveat"]))
    lines.append("")
    lines.append(f"Knowledge coverage: {anatomy['coverage_line']}")
    lines.append("")
    if anatomy["region"]:
        lines.append(f"Region reported: {_cell(anatomy['region'])}.")
        lines.append("")
    lines.append(
        f"Structures involved: {_cell(anatomy['n_structures_involved'])}. "
        f"Unlabelled fraction: {_fmt_fraction(anatomy['frac_unlabelled'])}."
    )
    lines.append("")
    structures = anatomy["structures"]
    if structures:
        headers = [
            "structure",
            "laterality",
            "lobe",
            "eloquence",
            "frac_of_tumour",
            "frac_of_structure",
        ]
        rows = [
            [
                _cell(row.get("structure")),
                _cell(row.get("laterality")),
                _cell(row.get("lobe")),
                _cell(row.get("eloquence")),
                _fmt_fraction(row.get("frac_of_tumour")),
                _fmt_fraction(row.get("frac_of_structure")),
            ]
            for row in structures
        ]
        lines.append(_pipe_table(headers, rows))
    else:
        lines.append("_No structures recorded._")
    lines.append("")

    # --- Eloquence ------------------------------------------------------- #
    eloquence = report["eloquence"]
    lines.append("## Eloquence Reference")
    lines.append("")
    lines.append(f"Classification system: **{_cell(eloquence['classification'])}**.")
    lines.append("")
    lines.append(
        f"Distance to nearest listed structure: {_fmt_distance(eloquence['distance_mm'])}."
    )
    lines.append("")
    lines.append(
        f"Within {_fmt_distance(eloquence['near_eloquent_threshold_mm'])} of an eloquent "
        f"structure: {'yes' if eloquence['near_eloquent'] else 'no'}."
    )
    lines.append("")
    lines.append(f"> {eloquence['evidence']}")
    lines.append("")
    lines.append(f"Source: {eloquence['citation']}.")
    lines.append("")
    lines.append(str(eloquence["source_owns_claim"]))
    lines.append("")
    involved = eloquence["involved"]
    if involved:
        headers = ["structure", "laterality", "frac_of_tumour", "frac_of_structure"]
        rows = [
            [
                _cell(row.get("structure")),
                _cell(row.get("laterality")),
                _fmt_fraction(row.get("frac_of_tumour")),
                _fmt_fraction(row.get("frac_of_structure")),
            ]
            for row in involved
        ]
        lines.append(_pipe_table(headers, rows))
    else:
        lines.append("_No structure from this classification overlaps the reported region._")
    lines.append("")
    if eloquence["coverage_gaps"]:
        gaps = ", ".join(str(g) for g in eloquence["coverage_gaps"])
        lines.append(f"Coverage gaps (source terms with no matching structure here): {gaps}.")
        lines.append("")

    # --- Not claimed ------------------------------------------------------ #
    lines.append("## Not Claimed")
    lines.append("")
    for what, why in report["not_claimed"]:
        lines.append(f"- **{what}**: {why}")
    lines.append("")

    # --- Provenance (last) -------------------------------------------------- #
    lines.append("## Provenance")
    lines.append("")
    provenance = report["provenance"]
    for key in sorted(provenance):
        value = provenance[key]
        if isinstance(value, Mapping):
            body = ", ".join(f"{k}={v}" for k, v in value.items()) if value else "n/a"
            lines.append(f"- **{key}**: {body}")
        else:
            lines.append(f"- **{key}**: {_cell(value)}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def write_report(report: Mapping, out_dir: Path, *, markdown: bool = True) -> dict[str, Path]:
    """Writes `<out_dir>/<case_id>.json` and, optionally, `<out_dir>/<case_id>.md`.

    The module's only IO. JSON is written through `json_safe` so the file is
    strict JSON (no bare `NaN`) and field order survives (`sort_keys=False`).

    Args:
        report: A `build_report` output.
        out_dir: Destination directory, created if missing.
        markdown: Also render and write the Markdown document.

    Returns:
        Dict with key `"json"` (always) and `"markdown"` (when requested),
        mapping to the written paths.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    case_id = str(report["case_id"])

    paths: dict[str, Path] = {}

    json_path = directory / f"{case_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, allow_nan=False, sort_keys=False)
    paths["json"] = json_path
    logger.info("Wrote report %s", json_path)

    if markdown:
        md_path = directory / f"{case_id}.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        paths["markdown"] = md_path
        logger.info("Wrote report %s", md_path)

    return paths
