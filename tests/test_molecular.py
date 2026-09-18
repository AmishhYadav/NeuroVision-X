"""Tests for `neurovision.reporting.molecular`.

Every test runs on CPU, well under a second each, and never touches the
deep-learning stack. Most tests load the real, committed
`knowledge/molecular_markers.yaml` -- it is the single source of truth this
module is built against -- except the loader-rejects-a-tampered-file test,
which writes a deliberately broken copy to `tmp_path`.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest
import yaml

from neurovision.reporting.molecular import (
    MARKERS,
    MolecularKnowledge,
    cns5_lookup,
    empty_molecular_block,
    load_molecular_knowledge,
    merge_pathology,
    validate_entered,
)
from neurovision.reporting.report import json_safe

_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "knowledge/molecular_markers.yaml"

_FORBIDDEN = ("grade", "stage", "prognosis", "deficit", "impair", "will experience")

_THREE_NAMES = frozenset(
    {
        "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted",
        "Astrocytoma, IDH-mutant",
        "Glioblastoma, IDH-wildtype",
    }
)


@pytest.fixture(scope="module")
def knowledge() -> MolecularKnowledge:
    return load_molecular_knowledge(_KNOWLEDGE_PATH)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _iter_strings(obj: object):
    """Yields every `str` reachable from `obj`, recursing through mappings/sequences/dataclasses."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from _iter_strings(key)
            yield from _iter_strings(value)
    elif is_dataclass(obj) and not isinstance(obj, type):
        yield from _iter_strings(asdict(obj))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            yield from _iter_strings(item)


def _scan_for_forbidden(strings) -> list[str]:
    hits: list[str] = []
    for text in strings:
        lowered = text.lower()
        for word in _FORBIDDEN:
            if word in lowered:
                hits.append(f"{word!r} in {text!r}")
    return hits


def _all_marker_value_combos(knowledge: MolecularKnowledge) -> Sequence[dict[str, str]]:
    """Every combination of IDH x 1p/19q x histology x TERT x EGFR allowed values."""
    idh_values = knowledge.markers["IDH"].allowed_values
    onep19q_values = knowledge.markers["1p/19q"].allowed_values
    histology_values = knowledge.histology_values
    tert_values = knowledge.markers["TERT"].allowed_values
    egfr_values = knowledge.markers["EGFR"].allowed_values

    combos = []
    for idh, onep19q, histology, tert, egfr in itertools.product(
        idh_values, onep19q_values, histology_values, tert_values, egfr_values
    ):
        combos.append(
            {
                "IDH": idh,
                "1p/19q": onep19q,
                "histology": histology,
                "TERT": tert,
                "EGFR": egfr,
            }
        )
    return combos


# --------------------------------------------------------------------------- #
# load_molecular_knowledge
# --------------------------------------------------------------------------- #


def test_markers_constant_matches_yaml_key_order(knowledge: MolecularKnowledge) -> None:
    """A YAML edit that adds/reorders a marker must fail this test, not disappear silently."""
    assert MARKERS == tuple(knowledge.markers.keys())


def test_load_molecular_knowledge_basic_shape(knowledge: MolecularKnowledge) -> None:
    assert knowledge.version == 1
    assert "2021 WHO Classification" in knowledge.citation
    assert knowledge.untested == "Not tested"
    assert knowledge.unentered == "Not entered"
    assert set(MARKERS) == set(knowledge.markers)
    for spec in knowledge.markers.values():
        assert spec.allowed_values[-2:] == ("Not tested", "Not entered")
    assert "Not entered" in knowledge.histology_values
    assert len(knowledge.lookup_rows) == 4
    assert knowledge.lookup_requires == ("IDH", "histology")


def test_load_molecular_knowledge_rejects_tampered_allowed_values(tmp_path: Path) -> None:
    """A marker's allowed_values that drops the shared sentinel pair must raise, not load."""
    with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["markers"]["IDH"]["allowed_values"] = ["Mutant", "Wildtype"]  # drops the sentinels
    bad_path = tmp_path / "molecular_markers_bad.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)

    with pytest.raises(ValueError, match="allowed_values"):
        load_molecular_knowledge(bad_path)


def test_load_molecular_knowledge_rejects_lookup_row_with_unknown_marker(tmp_path: Path) -> None:
    with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["cns5_lookup"]["rows"][0]["when"]["NOT_A_MARKER"] = "Whatever"
    bad_path = tmp_path / "molecular_markers_bad2.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)

    with pytest.raises(ValueError, match="NOT_A_MARKER"):
        load_molecular_knowledge(bad_path)


def test_load_molecular_knowledge_rejects_lookup_row_with_bad_value(tmp_path: Path) -> None:
    with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["cns5_lookup"]["rows"][0]["when"]["IDH"] = "Not-A-Real-Value"
    bad_path = tmp_path / "molecular_markers_bad3.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)

    with pytest.raises(ValueError, match="Not-A-Real-Value"):
        load_molecular_knowledge(bad_path)


def test_load_molecular_knowledge_rejects_lookup_row_histology_not_subset(tmp_path: Path) -> None:
    with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["cns5_lookup"]["rows"][0]["histology_in"] = ["Not a real histology pattern"]
    bad_path = tmp_path / "molecular_markers_bad4.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f)

    with pytest.raises(ValueError, match="histology_in"):
        load_molecular_knowledge(bad_path)


# --------------------------------------------------------------------------- #
# empty_molecular_block
# --------------------------------------------------------------------------- #


def test_empty_molecular_block_shape(knowledge: MolecularKnowledge) -> None:
    block = empty_molecular_block(knowledge)

    assert block["caveat"] == knowledge.copy["panel_title"]
    assert block["citation"] == knowledge.citation
    assert block["scope"] == knowledge.scope

    assert block["histology"]["confirmed_pathology"] == "Not entered"
    assert block["histology"]["allowed_values"] == list(knowledge.histology_values)

    assert set(block["markers"]) == set(MARKERS)
    for name in MARKERS:
        entry = block["markers"][name]
        spec = knowledge.markers[name]
        assert entry["label"] == spec.label
        assert entry["meaning"] == spec.meaning
        assert entry["cns5_role"] == spec.cns5_role
        assert entry["allowed_values"] == list(spec.allowed_values)
        assert entry["confirmed_pathology"] == "Not entered"
        if name == "IDH":
            assert entry["ai_estimate"] == {"status": "not available — model not trained"}
        else:
            assert entry["ai_estimate"] is None

    assert block["cns5"]["label"] == knowledge.copy["cns5_line_label"]
    assert block["cns5"]["name"] is None
    assert block["cns5"]["requires"] == list(knowledge.lookup_requires)
    assert block["cns5"]["source"] is None


# --------------------------------------------------------------------------- #
# validate_entered
# --------------------------------------------------------------------------- #


def test_validate_entered_accepts_good_values(knowledge: MolecularKnowledge) -> None:
    entered = {"IDH": "Mutant", "histology": "Astrocytic"}
    result = validate_entered(knowledge, entered)
    assert result == entered
    assert result is not entered  # returns a copy


def test_validate_entered_rejects_unknown_key(knowledge: MolecularKnowledge) -> None:
    with pytest.raises(ValueError, match="NOT_A_KEY"):
        validate_entered(knowledge, {"NOT_A_KEY": "Mutant"})


def test_validate_entered_rejects_bad_value(knowledge: MolecularKnowledge) -> None:
    with pytest.raises(ValueError, match="IDH"):
        validate_entered(knowledge, {"IDH": "Definitely Not A Value"})


# --------------------------------------------------------------------------- #
# merge_pathology
# --------------------------------------------------------------------------- #


def test_merge_pathology_sets_values_and_does_not_mutate(knowledge: MolecularKnowledge) -> None:
    block = empty_molecular_block(knowledge)
    entered = {"IDH": "Mutant", "histology": "Astrocytic", "1p/19q": "Intact"}

    merged = merge_pathology(block, entered, knowledge)

    # Original untouched.
    assert block["markers"]["IDH"]["confirmed_pathology"] == "Not entered"
    assert block["histology"]["confirmed_pathology"] == "Not entered"
    assert block["cns5"]["name"] is None

    assert merged["markers"]["IDH"]["confirmed_pathology"] == "Mutant"
    assert merged["markers"]["1p/19q"]["confirmed_pathology"] == "Intact"
    assert merged["histology"]["confirmed_pathology"] == "Astrocytic"
    assert merged["cns5"]["name"] == "Astrocytoma, IDH-mutant"
    assert merged["cns5"]["source"] == "from entered pathology"


def test_merge_pathology_builds_on_previous_merge(knowledge: MolecularKnowledge) -> None:
    block = empty_molecular_block(knowledge)
    step1 = merge_pathology(block, {"IDH": "Wildtype"}, knowledge)
    assert step1["cns5"]["name"] is None
    assert step1["cns5"]["requires"] == ["histology"]

    step2 = merge_pathology(step1, {"histology": "Glioblastoma pattern"}, knowledge)
    assert step2["cns5"]["name"] == "Glioblastoma, IDH-wildtype"
    assert step2["markers"]["IDH"]["confirmed_pathology"] == "Wildtype"  # carried over from step1


def test_merge_pathology_rejects_bad_entered(knowledge: MolecularKnowledge) -> None:
    block = empty_molecular_block(knowledge)
    with pytest.raises(ValueError):
        merge_pathology(block, {"IDH": "Nonsense"}, knowledge)


# --------------------------------------------------------------------------- #
# cns5_lookup -- exhaustiveness
# --------------------------------------------------------------------------- #


def test_cns5_lookup_never_raises_and_name_is_valid(knowledge: MolecularKnowledge) -> None:
    for combo in _all_marker_value_combos(knowledge):
        result = cns5_lookup(knowledge, combo)
        assert result["name"] is None or result["name"] in _THREE_NAMES
        assert isinstance(result["requires"], list)
        assert result["source"] in ("from entered pathology", None)
        assert (result["source"] is None) == (result["name"] is None)


@pytest.mark.parametrize("histology", ["Astrocytic", "Oligodendroglial", "Glioblastoma pattern"])
def test_cns5_lookup_oligodendroglioma_case(knowledge: MolecularKnowledge, histology: str) -> None:
    result = cns5_lookup(
        knowledge, {"IDH": "Mutant", "1p/19q": "Codeleted", "histology": histology}
    )
    assert result["name"] == "Oligodendroglioma, IDH-mutant and 1p/19q-codeleted"
    assert result["source"] == "from entered pathology"


def test_cns5_lookup_astrocytoma_idh_mutant_case(knowledge: MolecularKnowledge) -> None:
    result = cns5_lookup(
        knowledge, {"IDH": "Mutant", "1p/19q": "Intact", "histology": "Astrocytic"}
    )
    assert result["name"] == "Astrocytoma, IDH-mutant"


def test_cns5_lookup_glioblastoma_pattern_case(knowledge: MolecularKnowledge) -> None:
    result = cns5_lookup(knowledge, {"IDH": "Wildtype", "histology": "Glioblastoma pattern"})
    assert result["name"] == "Glioblastoma, IDH-wildtype"


@pytest.mark.parametrize("molecular_criterion", [{"TERT": "Mutant"}, {"EGFR": "Amplified"}])
def test_cns5_lookup_molecular_glioblastoma_case(
    knowledge: MolecularKnowledge, molecular_criterion: dict[str, str]
) -> None:
    entered = {"IDH": "Wildtype", "histology": "Astrocytic", **molecular_criterion}
    result = cns5_lookup(knowledge, entered)
    assert result["name"] == "Glioblastoma, IDH-wildtype"
    assert result["source"] == "from entered pathology"


def test_cns5_lookup_astrocytic_wildtype_no_molecular_criterion_gives_no_name(
    knowledge: MolecularKnowledge,
) -> None:
    """All four rows are ruled out by this combination, so no row is even reachable."""
    result = cns5_lookup(
        knowledge,
        {
            "IDH": "Wildtype",
            "histology": "Astrocytic",
            "TERT": "Wildtype",
            "EGFR": "Not amplified",
        },
    )
    assert result["name"] is None
    assert result["requires"] == []
    assert result["source"] is None


def test_cns5_lookup_idh_not_entered_gives_no_name_and_lists_idh(
    knowledge: MolecularKnowledge,
) -> None:
    result = cns5_lookup(knowledge, {})
    assert result["name"] is None
    assert "IDH" in result["requires"]


def test_cns5_lookup_verification_example(knowledge: MolecularKnowledge) -> None:
    """The plan's own worked example."""
    result = cns5_lookup(knowledge, {"IDH": "Wildtype", "histology": "Glioblastoma pattern"})
    assert result["name"] == "Glioblastoma, IDH-wildtype"
    assert result["source"] == "from entered pathology"


# --------------------------------------------------------------------------- #
# cns5_lookup -- the richer `requires` rule: intersection of still-reachable
# rows' still-open keys, not a static lookup_requires filter. Every example
# here is one the coordinator specified verbatim.
# --------------------------------------------------------------------------- #


def test_cns5_lookup_requires_nothing_entered(knowledge: MolecularKnowledge) -> None:
    """All four rows are reachable; only IDH and histology are common to every one of them."""
    result = cns5_lookup(knowledge, {})
    assert result["name"] is None
    assert result["requires"] == ["histology", "IDH"]
    assert result["source"] is None


def test_cns5_lookup_requires_idh_mutant_and_histology_narrows_to_1p19q(
    knowledge: MolecularKnowledge,
) -> None:
    """IDH=Mutant rules out both glioblastoma rows; the two remaining rows differ only on 1p/19q."""
    result = cns5_lookup(knowledge, {"IDH": "Mutant", "histology": "Astrocytic"})
    assert result["name"] is None
    assert result["requires"] == ["1p/19q"]
    assert result["source"] is None


def test_cns5_lookup_requires_idh_wildtype_astrocytic_narrows_to_tert_and_egfr(
    knowledge: MolecularKnowledge,
) -> None:
    """IDH=Wildtype + Astrocytic rules out every row except the molecular-GBM one."""
    result = cns5_lookup(knowledge, {"IDH": "Wildtype", "histology": "Astrocytic"})
    assert result["name"] is None
    assert result["requires"] == ["TERT", "EGFR"]
    assert result["source"] is None


def test_cns5_lookup_requires_empty_when_no_row_reachable_after_negative_criteria(
    knowledge: MolecularKnowledge,
) -> None:
    """TERT and EGFR both resolving negative closes off the one row that was still reachable."""
    result = cns5_lookup(
        knowledge,
        {
            "IDH": "Wildtype",
            "histology": "Astrocytic",
            "TERT": "Wildtype",
            "EGFR": "Not amplified",
        },
    )
    assert result["name"] is None
    assert result["requires"] == []
    assert result["source"] is None


def test_cns5_lookup_requires_empty_when_idh_not_tested(knowledge: MolecularKnowledge) -> None:
    """A "Not tested" value is a definite fact, not an open slot -- it rules every row out."""
    result = cns5_lookup(knowledge, {"IDH": "Not tested", "histology": "Astrocytic"})
    assert result["name"] is None
    assert result["requires"] == []
    assert result["source"] is None


# --------------------------------------------------------------------------- #
# Forbidden-word scan
# --------------------------------------------------------------------------- #


def test_no_forbidden_words_in_loaded_knowledge(knowledge: MolecularKnowledge) -> None:
    hits = _scan_for_forbidden(_iter_strings(knowledge))
    assert hits == [], f"forbidden words in loaded knowledge: {hits}"


def test_no_forbidden_words_in_empty_block(knowledge: MolecularKnowledge) -> None:
    block = empty_molecular_block(knowledge)
    hits = _scan_for_forbidden(_iter_strings(block))
    assert hits == [], f"forbidden words in empty_molecular_block: {hits}"

    # Also scan the JSON-serialised form, through the same json_safe used by
    # neurovision.reporting.report, in case serialisation surfaces a string
    # differently (e.g. via __str__ on a non-str value).
    text = json.dumps(json_safe(block))
    hits_json = _scan_for_forbidden([text])
    assert hits_json == [], f"forbidden words in json_safe(block): {hits_json}"


def test_no_forbidden_words_in_any_cns5_lookup_result(knowledge: MolecularKnowledge) -> None:
    for combo in _all_marker_value_combos(knowledge):
        result = cns5_lookup(knowledge, combo)
        hits = _scan_for_forbidden(_iter_strings(result))
        assert hits == [], f"forbidden words in cns5_lookup({combo}): {hits}"
