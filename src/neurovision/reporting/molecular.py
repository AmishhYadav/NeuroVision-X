"""Molecular pathology block: ENTERED marker values -> a WHO CNS5 diagnosis NAME.

T5.1 of `docs/research/tool_completion_plan.md`. This module is the code
counterpart of `knowledge/molecular_markers.yaml`: that file owns every claim
(the marker vocabulary, the CNS5 lookup rows, and the copy shown to a user),
this module owns only the mechanics of loading it, validating what a user
enters against it, and assembling the report block.

**NO PREDICTIVE CLAIM. NO DIAGNOSIS FROM IMAGING.** Every value that ends up
in the block returned by `empty_molecular_block` or `merge_pathology` is
either a value a user typed in (`entered`), or the literal string
`"Not entered"` / `"Not tested"` from the knowledge file's closed vocabulary.
The one derived field, `cns5["name"]`, is computed ONLY from `entered` values
via `cns5_lookup`, and the block labels it `"source": "from entered
pathology"` so a reader never mistakes it for something the model produced.
The IDH marker's `ai_estimate` is the single exception worth naming directly:
it is not `None` like every other marker's, but a fixed unavailability
string, because the tool completion plan once considered training an IDH
classifier and did not (Phase F / T7, gated on the author's go) -- the
field exists so that decision is visible in the schema rather than silently
absent.

This module is pure: no torch, no numpy, and its only I/O is reading the one
YAML file handed to `load_molecular_knowledge` (never a hardcoded path -- see
`neurovision.utils.io.read_yaml`, whose loader this module reuses).

Vocabulary rule, same discipline as `neurovision.reporting.report`: the
strings this module renders or returns must never contain "grade", "stage",
"prognosis", "deficit", "impair", or "will experience" (case-insensitive
substring), because `knowledge/molecular_markers.yaml`'s own header already
promises that and `tests/test_molecular.py` scans for it. This module has no
`not_claimed` block of its own -- that discipline lives in
`neurovision.reporting.report` and applies to the whole report, this block
included.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from neurovision.utils.io import read_yaml

__all__ = [
    "MarkerSpec",
    "LookupRow",
    "MolecularKnowledge",
    "MARKERS",
    "load_molecular_knowledge",
    "empty_molecular_block",
    "validate_entered",
    "merge_pathology",
    "cns5_lookup",
]

logger = logging.getLogger(__name__)

# The eight marker names, in the order `knowledge/molecular_markers.yaml`
# declares them. Hardcoded here (not derived from the loaded YAML) on
# purpose: `tests/test_molecular.py` asserts this tuple equals the loaded
# file's own key order, so an edit that adds/renames/reorders a marker in the
# YAML without updating this constant fails a test loudly, instead of the
# marker silently vanishing from every report built against the old tuple.
MARKERS: tuple[str, ...] = (
    "IDH",
    "1p/19q",
    "MGMT",
    "ATRX",
    "TP53",
    "TERT",
    "EGFR",
    "CDKN2A/B",
)

# "histology" is not a molecular marker but is looked up the same way (an
# entered value checked against a closed vocabulary), so it shares the
# "Not entered" sentinel with the eight markers above.
_HISTOLOGY_KEY = "histology"


@dataclass(frozen=True)
class MarkerSpec:
    """One marker's vocabulary and CNS5-facing text, straight from the YAML.

    Attributes:
        name: The marker's key, e.g. `"IDH"` or `"1p/19q"`.
        label: A human-readable label, e.g. `"IDH1/IDH2 mutation"`.
        allowed_values: The closed vocabulary a user may enter, always ending
            with `("Not tested", "Not entered")`.
        meaning: What the marker is (one sentence, source-derived).
        cns5_role: What CNS5 does with the marker (one sentence,
            source-derived).
    """

    name: str
    label: str
    allowed_values: tuple[str, ...]
    meaning: str
    cns5_role: str


@dataclass(frozen=True)
class LookupRow:
    """One row of the entered-pathology -> CNS5-name lookup table.

    Attributes:
        name: The CNS5 integrated diagnosis name this row produces when it
            matches.
        when: Marker name -> the value it must equal for this row to match.
        histology_in: The set of `histology` values this row matches.
        any_of: Marker name -> a value where AT LEAST ONE must match (empty
            when the row has no such clause -- e.g. the two IDH-mutant rows,
            which are fully decided by `when` and `histology_in` alone).
    """

    name: str
    when: dict[str, str]
    histology_in: tuple[str, ...]
    any_of: dict[str, str]


@dataclass(frozen=True)
class MolecularKnowledge:
    """The loaded, validated contents of `knowledge/molecular_markers.yaml`.

    Attributes:
        version: The file's own `version` field.
        citation: The WHO CNS5 summary citation, verbatim.
        scope: What tumour types this file's lookup covers (adult-type
            diffuse glioma only) and what it does not.
        markers: Marker name -> `MarkerSpec`, ordered as in the YAML (IDH,
            1p/19q, MGMT, ATRX, TP53, TERT, EGFR, CDKN2A/B).
        histology_values: The closed vocabulary for the `histology` field.
        lookup_rows: The CNS5 lookup table, in the order they are checked.
        lookup_requires: The marker/histology names `cns5_lookup` reports as
            still needed when no row has matched yet.
        copy: Verbatim report/panel text, keyed by the YAML's `copy` keys
            (`panel_title`, `ai_estimate_unavailable`, `cns5_line_suffix`,
            `cns5_line_label`).
        untested: The shared "not tested" sentinel value, `"Not tested"`.
        unentered: The shared "not entered" sentinel value, `"Not entered"`.
    """

    version: int
    citation: str
    scope: str
    markers: dict[str, MarkerSpec]
    histology_values: tuple[str, ...]
    lookup_rows: tuple[LookupRow, ...]
    lookup_requires: tuple[str, ...]
    copy: dict[str, str]
    untested: str
    unentered: str


def load_molecular_knowledge(path: Path) -> MolecularKnowledge:
    """Loads and validates `knowledge/molecular_markers.yaml`.

    Validation is strict, on purpose: this file's job is a closed vocabulary
    plus a small lookup table, and every failure mode below is exactly the
    kind of typo that would otherwise reach a report silently -- a marker
    whose `allowed_values` forgets one of the two shared sentinels, or a
    lookup row that names a marker or a value the file itself never declared.

    Args:
        path: Path to `knowledge/molecular_markers.yaml`.

    Returns:
        The parsed and validated `MolecularKnowledge`.

    Raises:
        ValueError: If a marker's `allowed_values` does not end with
            `(untested, unentered)`, if a lookup row's `when`/`any_of` names
            a marker outside `MARKERS`, if a lookup row's `when`/`any_of`
            value is outside that marker's own `allowed_values`, or if a
            lookup row's `histology_in` is not a subset of the declared
            histology vocabulary.
    """
    doc = read_yaml(path)

    shared = doc["shared_vocabulary"]
    untested = str(shared["untested"])
    unentered = str(shared["unentered"])

    raw_markers = doc["markers"]
    markers: dict[str, MarkerSpec] = {}
    for name in raw_markers:
        entry = raw_markers[name]
        allowed = tuple(str(v) for v in entry["allowed_values"])
        # Every marker's vocabulary must end in the two shared sentinels, in
        # that order -- "Not tested" (pathology looked and found nothing to
        # report) before "Not entered" (no one has typed anything in yet).
        # Checking the ORDER as well as membership catches a copy-paste that
        # reverses the pair, which `validate_entered` would otherwise accept
        # silently since both strings are still present.
        if allowed[-2:] != (untested, unentered):
            raise ValueError(
                f"load_molecular_knowledge: marker '{name}'s allowed_values must end with "
                f"({untested!r}, {unentered!r}), got {allowed!r}."
            )
        markers[name] = MarkerSpec(
            name=name,
            label=str(entry["label"]),
            allowed_values=allowed,
            meaning=str(entry["meaning"]).strip(),
            cns5_role=str(entry["cns5_role"]).strip(),
        )

    histology_values = tuple(str(v) for v in doc["histology"]["allowed_values"])

    lookup_doc = doc["cns5_lookup"]
    lookup_requires = tuple(str(v) for v in lookup_doc["requires"])
    for key in lookup_requires:
        if key != _HISTOLOGY_KEY and key not in markers:
            raise ValueError(
                f"load_molecular_knowledge: cns5_lookup.requires names '{key}', which is "
                "neither a declared marker nor 'histology'."
            )

    def _check_clause(row_name: str, clause_kind: str, clause: Mapping[str, str]) -> None:
        """Raises if a `when`/`any_of` clause names an unknown marker or value."""
        for marker_name, value in clause.items():
            if marker_name not in markers:
                raise ValueError(
                    f"load_molecular_knowledge: lookup row '{row_name}'s {clause_kind} names "
                    f"marker '{marker_name}', which is not declared under `markers`."
                )
            if value not in markers[marker_name].allowed_values:
                raise ValueError(
                    f"load_molecular_knowledge: lookup row '{row_name}'s {clause_kind} sets "
                    f"'{marker_name}' to '{value}', which is not in its allowed_values "
                    f"{markers[marker_name].allowed_values!r}."
                )

    lookup_rows: list[LookupRow] = []
    histology_set = set(histology_values)
    for raw_row in lookup_doc["rows"]:
        row_name = str(raw_row["name"])
        when = {str(k): str(v) for k, v in raw_row.get("when", {}).items()}
        any_of = {str(k): str(v) for k, v in raw_row.get("any_of", {}).items()}
        histology_in = tuple(str(v) for v in raw_row["histology_in"])

        _check_clause(row_name, "when", when)
        _check_clause(row_name, "any_of", any_of)

        if not set(histology_in).issubset(histology_set):
            raise ValueError(
                f"load_molecular_knowledge: lookup row '{row_name}'s histology_in "
                f"{histology_in!r} is not a subset of the declared histology values "
                f"{histology_values!r}."
            )

        lookup_rows.append(
            LookupRow(name=row_name, when=when, histology_in=histology_in, any_of=any_of)
        )

    copy = {str(k): str(v) for k, v in doc["copy"].items()}

    knowledge = MolecularKnowledge(
        version=int(doc["version"]),
        citation=str(doc["citation"]).strip(),
        scope=str(doc["scope"]).strip(),
        markers=markers,
        histology_values=histology_values,
        lookup_rows=tuple(lookup_rows),
        lookup_requires=lookup_requires,
        copy=copy,
        untested=untested,
        unentered=unentered,
    )
    logger.info(
        "load_molecular_knowledge: loaded %d markers, %d lookup rows from %s (version %d).",
        len(markers),
        len(lookup_rows),
        path,
        knowledge.version,
    )
    return knowledge


def empty_molecular_block(knowledge: MolecularKnowledge) -> dict:
    """Builds the molecular pathology report block with nothing entered yet.

    Every `confirmed_pathology` field starts as `"Not entered"` -- the
    default state before any user has typed a pathology result in -- and
    `cns5["name"]` starts `None`, since `cns5_lookup` never runs without
    `entered` values to look up.

    Args:
        knowledge: The loaded `MolecularKnowledge`.

    Returns:
        A plain dict with keys `caveat`, `citation`, `scope`, `histology`,
        `markers`, `cns5`. See the module's `empty_molecular_block` spec in
        `docs/research/tool_completion_plan.md` (T5.1) for the exact shape.
    """
    markers_block: dict[str, dict] = {}
    for name in MARKERS:
        spec = knowledge.markers[name]
        # Every marker gets ai_estimate=None -- no model was ever trained to
        # predict any of them -- EXCEPT IDH, whose slot is a fixed
        # "unavailable" string rather than None. This distinguishes "we never
        # planned to estimate this" (None, e.g. MGMT) from "we considered
        # estimating this and chose not to yet" (IDH) -- see the module
        # docstring and T7/Phase F in the tool completion plan.
        ai_estimate: dict[str, str] | None = None
        if name == "IDH":
            ai_estimate = {"status": knowledge.copy["ai_estimate_unavailable"]}
        markers_block[name] = {
            "label": spec.label,
            "meaning": spec.meaning,
            "cns5_role": spec.cns5_role,
            "allowed_values": list(spec.allowed_values),
            "confirmed_pathology": knowledge.unentered,
            "ai_estimate": ai_estimate,
        }

    return {
        "caveat": knowledge.copy["panel_title"],
        "citation": knowledge.citation,
        "scope": knowledge.scope,
        "histology": {
            "confirmed_pathology": knowledge.unentered,
            "allowed_values": list(knowledge.histology_values),
        },
        "markers": markers_block,
        "cns5": {
            "label": knowledge.copy["cns5_line_label"],
            "name": None,
            "requires": list(knowledge.lookup_requires),
            "source": None,
        },
    }


def validate_entered(knowledge: MolecularKnowledge, entered: Mapping[str, str]) -> dict[str, str]:
    """Checks a caller-supplied `{marker_or_"histology": value}` mapping against the knowledge base.

    Args:
        knowledge: The loaded `MolecularKnowledge`.
        entered: Marker names (from `MARKERS`) or `"histology"` mapped to a
            value a user entered.

    Returns:
        A plain `dict` copy of `entered` -- unchanged, since every value is
        either valid or this function has already raised.

    Raises:
        ValueError: If a key is not a known marker or `"histology"`, or if a
            value is outside that key's `allowed_values`.
    """
    result: dict[str, str] = {}
    for key, value in entered.items():
        if key == _HISTOLOGY_KEY:
            allowed = knowledge.histology_values
        elif key in knowledge.markers:
            allowed = knowledge.markers[key].allowed_values
        else:
            raise ValueError(
                f"validate_entered: '{key}' is not a known marker or 'histology'. Known keys: "
                f"{list(MARKERS) + [_HISTOLOGY_KEY]!r}."
            )
        if value not in allowed:
            raise ValueError(
                f"validate_entered: '{key}' was set to '{value}', which is not in its allowed "
                f"values {list(allowed)!r}."
            )
        result[key] = value
    return result


def _row_still_needed_keys(row: LookupRow, value_of, unentered: str) -> set[str] | None:
    """The keys `row` still needs to have a chance of matching, or `None` if it can never match.

    A constraint (one `when` marker, `histology_in`, or the `any_of` clause
    as a whole) is either already satisfied by the current values, still
    OPEN (its key(s) are `"Not entered"`, so a later entry could still
    satisfy it), or DEFINITELY BROKEN (its key holds some other value --
    including `"Not tested"`, which is a real fact that rules a constraint
    out just as much as a wrong mutation status would). One definitely
    broken constraint means this row can never match `entered`, no matter
    what else gets typed in later, so the row is dropped entirely (`None`).
    Otherwise the row is still "reachable", and this returns exactly the
    keys standing between the current `entered` and a match on this row.

    Args:
        row: The lookup row to test.
        value_of: `entered.get(key, unentered)`, passed in rather than
            re-derived so `cns5_lookup` builds it once.
        unentered: `knowledge.unentered`, the "Not entered" sentinel -- the
            only value that keeps a constraint open rather than broken.

    Returns:
        The set of still-open keys, or `None` if `row` is unreachable.
    """
    needed: set[str] = set()

    for marker, required_value in row.when.items():
        current = value_of(marker)
        if current == required_value:
            continue  # already satisfied -- nothing more needed for this marker
        if current == unentered:
            needed.add(marker)  # open: entering the right value could still satisfy it
            continue
        return None  # a definite, non-matching value (incl. "Not tested") -- dead end

    current_histology = value_of(_HISTOLOGY_KEY)
    if current_histology not in row.histology_in:
        if current_histology == unentered:
            needed.add(_HISTOLOGY_KEY)
        else:
            return None

    if row.any_of:
        satisfied = any(value_of(marker) == value for marker, value in row.any_of.items())
        if not satisfied:
            # At least one any_of marker must end up matching. Any marker
            # that already holds a definite non-matching value can never be
            # the one that does -- only the still-open ones can. If NONE are
            # open, this clause (and so this row) can never be satisfied.
            still_open = [marker for marker in row.any_of if value_of(marker) == unentered]
            if not still_open:
                return None
            needed.update(still_open)

    return needed


def cns5_lookup(knowledge: MolecularKnowledge, entered: Mapping[str, str]) -> dict:
    """Looks up the CNS5 integrated diagnosis name from entered pathology values only.

    Walks `knowledge.lookup_rows` in order and returns the first row whose
    `when` clause is fully satisfied, whose histology is in `histology_in`,
    and whose `any_of` clause (when non-empty) has at least one match. A
    missing key in `entered` is treated as `"Not entered"` -- never as a
    wildcard match -- so an unentered marker can never accidentally satisfy a
    row.

    When no row matches outright, `requires` is built from every row that is
    still REACHABLE -- one where nothing entered so far definitely rules it
    out (see `_row_still_needed_keys`) -- as the INTERSECTION, across those
    reachable rows, of each row's still-open keys. Intersection, not union:
    a key only belongs in `requires` if every remaining possible diagnosis
    still depends on it, so a marker that would only matter for ONE of
    several still-possible rows is not demanded up front. E.g. with nothing
    entered, all four rows are reachable but only IDH and histology are
    common to every one of them -- `1p/19q` only matters if IDH turns out
    "Mutant", so it is not required yet.

    Args:
        knowledge: The loaded `MolecularKnowledge`.
        entered: Marker names (or `"histology"`) mapped to their entered
            value. Values not supplied here are treated as `"Not entered"`.

    Returns:
        `{"name": ..., "requires": [...], "source": ...}`. `name` is the
        matching row's name, or `None` if no row matched. `source` is
        `"from entered pathology"` when `name` is not `None`, else `None`.
        `requires` is empty whenever `name` is not `None`, OR when `name` is
        `None` because no row is reachable at all (every row has already
        been definitely ruled out by what was entered). Otherwise `requires`
        is the sorted-by-relevance (`"histology"` first, then `MARKERS`
        order) list of keys every still-reachable row still needs.
    """
    unentered = knowledge.unentered

    def _value_of(key: str) -> str:
        return entered.get(key, unentered)

    for row in knowledge.lookup_rows:
        if any(_value_of(marker) != value for marker, value in row.when.items()):
            continue
        if _value_of(_HISTOLOGY_KEY) not in row.histology_in:
            continue
        if row.any_of and not any(
            _value_of(marker) == value for marker, value in row.any_of.items()
        ):
            continue
        return {"name": row.name, "requires": [], "source": "from entered pathology"}

    reachable_needed: list[set[str]] = []
    for row in knowledge.lookup_rows:
        needed = _row_still_needed_keys(row, _value_of, unentered)
        if needed is not None:
            reachable_needed.append(needed)

    if not reachable_needed:
        return {"name": None, "requires": [], "source": None}

    common = set.intersection(*reachable_needed)
    ordered: list[str] = [_HISTOLOGY_KEY] if _HISTOLOGY_KEY in common else []
    ordered += [marker for marker in MARKERS if marker in common]
    return {"name": None, "requires": ordered, "source": None}


def merge_pathology(
    block: Mapping, entered: Mapping[str, str], knowledge: MolecularKnowledge
) -> dict:
    """Returns a new molecular block with `entered` values merged in, never mutating `block`.

    Args:
        block: A molecular block, typically `empty_molecular_block`'s output
            or a previous `merge_pathology` result.
        entered: Marker names (or `"histology"`) mapped to a value to record
            as `confirmed_pathology`. Validated with `validate_entered`
            before anything is merged.
        knowledge: The loaded `MolecularKnowledge` `entered` is validated
            and looked up against.

    Returns:
        A new dict, deep-copied from `block`, with each entered key's
        `confirmed_pathology` set and `cns5` recomputed by `cns5_lookup` over
        the FULL merged set of confirmed values (not just the newly entered
        ones), so a second call correctly builds on a first.

    Raises:
        ValueError: Via `validate_entered`, if `entered` names an unknown key
            or an out-of-vocabulary value.
    """
    checked = validate_entered(knowledge, entered)
    merged = deepcopy(dict(block))

    for key, value in checked.items():
        if key == _HISTOLOGY_KEY:
            merged["histology"]["confirmed_pathology"] = value
        else:
            merged["markers"][key]["confirmed_pathology"] = value

    # Recompute CNS5 from every confirmed value currently on the block --
    # not just the ones passed in this call -- so calling merge_pathology
    # twice (e.g. IDH first, then histology) still reaches a name once both
    # are present.
    current: dict[str, str] = {_HISTOLOGY_KEY: merged["histology"]["confirmed_pathology"]}
    for name in MARKERS:
        current[name] = merged["markers"][name]["confirmed_pathology"]

    merged["cns5"] = {
        "label": merged["cns5"]["label"],
        **cns5_lookup(knowledge, current),
    }
    return merged
