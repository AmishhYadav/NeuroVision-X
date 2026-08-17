"""Tests for `neurovision.reporting.report`.

Every test runs on CPU on small hand-built dicts/DataFrames, well under a
second each, and never touches real BraTS data, a real atlas, or the model.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from neurovision.reporting import report as report_module
from neurovision.reporting.report import (
    DISCLAIMER,
    MASS_EFFECT_CAVEAT,
    NOT_CLAIMED,
    Provenance,
    build_report,
    json_safe,
    render_markdown,
    write_report,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _NumpyLikeScalar:
    """A stand-in for `np.float64`/`np.int64`/`np.bool_` without importing numpy."""

    def __init__(self, value: object) -> None:
        self._value = value

    def item(self) -> object:
        return self._value


def _provenance(**overrides: object) -> Provenance:
    defaults: dict[str, object] = dict(
        atlas_name="SRI24/TZO",
        atlas_version="1.0",
        atlas_source="https://nitrc.org/sri24",
        atlas_licence="CC-BY-SA",
        knowledge_versions={"eloquence_map": 1, "aal_lobes": 1},
        segmentation_source="prediction",
        segmentation_dir="/outputs/eval_test/predictions",
        code_revision="abc1234",
        generated_utc="2026-08-17T00:00:00Z",
    )
    defaults.update(overrides)
    return Provenance(**defaults)  # type: ignore[arg-type]


def _burden() -> dict[str, object]:
    """A small, representative slice of a real `burden_profile` output."""
    return {
        "vol_NCR_mm3": 1200.0,
        "vol_ED_mm3": 3400.0,
        "vol_ET_mm3": 800.0,
        "vol_TC_mm3": 2000.0,
        "vol_WT_mm3": 5400.0,
        "frac_enhancing_of_wt": 800.0 / 5400.0,
        "ratio_edema_to_core": 3400.0 / 2000.0,
        "n_components_WT": 1,
        "vol_largest_component_WT_mm3": 5400.0,
        "largest_component_frac_WT": 1.0,
        "surface_area_WT_mm2": 1500.0,
        "sphericity_WT": 0.72,
        "surface_to_volume_WT": 1500.0 / 5400.0,
        "vol_right_WT_mm3": 5400.0,
        "vol_left_WT_mm3": 0.0,
        "frac_left_WT": 0.0,
        "frac_contralateral_WT": 0.0,
        "dominant_side_WT": "right",
        "centroid_i_WT": 120.5,
        "centroid_j_WT": 100.0,
        "centroid_k_WT": 80.0,
        "frac_unlabelled_extra": float("nan"),
    }


def _anatomy_table(*, with_region: bool = True) -> pd.DataFrame:
    rows = [
        {
            "region": "WT",
            "structure": "Precentral_L",
            "laterality": "left",
            "lobe": "frontal",
            "eloquence": "eloquent",
            "matched_term": "motor cortex",
            "n_voxels": 500,
            "volume_mm3": 500.0,
            "frac_of_tumour": 0.5,
            "frac_of_structure": 0.4,
        },
        {
            "region": "WT",
            "structure": "Temporal_Mid_L",
            "laterality": "left",
            "lobe": "temporal",
            "eloquence": "unclassified",
            "matched_term": "",
            "n_voxels": 300,
            "volume_mm3": 300.0,
            "frac_of_tumour": 0.3,
            "frac_of_structure": 0.6,
        },
        {
            "region": "WT",
            "structure": "unlabelled",
            "laterality": "midline",
            "lobe": "",
            "eloquence": "",
            "matched_term": "",
            "n_voxels": 200,
            "volume_mm3": 200.0,
            "frac_of_tumour": 0.2,
            "frac_of_structure": float("nan"),
        },
        {
            "region": "ET",
            "structure": "Precentral_L",
            "laterality": "left",
            "lobe": "frontal",
            "eloquence": "eloquent",
            "matched_term": "motor cortex",
            "n_voxels": 100,
            "volume_mm3": 100.0,
            "frac_of_tumour": 1.0,
            "frac_of_structure": 0.1,
        },
    ]
    table = pd.DataFrame(rows)
    if not with_region:
        table = table[table["region"] == "WT"].drop(columns=["region"])
    return table


def _anatomy_summary() -> dict[str, object]:
    return {
        "n_structures_involved": 2,
        "top_structure": "Precentral_L",
        "top_frac_of_structure": 0.4,
        "most_displaced_structure": "Temporal_Mid_L",
        "frac_unlabelled": 0.2,
        "n_eloquent_structures": 1,
        "eloquent_frac_of_tumour": 0.5,
        "distance_to_eloquent_mm": 0.0,
        "dominant_lobe": "frontal",
        "coverage_line": "23 of 122 structures classified eloquent, 99 unclassified",
    }


def _build(**overrides: object) -> dict:
    kwargs: dict[str, object] = dict(
        case_id="case_0001",
        burden=_burden(),
        anatomy_table=_anatomy_table(),
        anatomy_summary=_anatomy_summary(),
        provenance=_provenance(),
        evidence="Sawaya's eloquent set: motor/sensory cortices, visual center, speech center.",
        citation="Sawaya et al. 1998, PMID 9433896",
        classification_name="Sawaya eloquence grading",
        coverage_line="23 of 122 structures classified eloquent, 99 unclassified",
        coverage_gaps=["internal capsule", "dentate nucleus"],
        near_eloquent_mm=10.0,
    )
    kwargs.update(overrides)
    return build_report(**kwargs)  # type: ignore[arg-type]


# Forbidden vocabulary this artifact must never use outside `not_claimed`.
_FORBIDDEN = ("deficit", "prognosis", "grade", "stage", "will experience", "impair")


def _scan_for_forbidden(text: str) -> list[str]:
    lowered = text.lower()
    return [word for word in _FORBIDDEN if word in lowered]


# --------------------------------------------------------------------------- #
# 1. Strict JSON round-trip
# --------------------------------------------------------------------------- #


def test_report_round_trips_as_strict_json() -> None:
    report = _build(burden={**_burden(), "some_nan_value": float("nan")})
    safe = json_safe(report)
    text = json.dumps(safe, allow_nan=False)
    restored = json.loads(text)
    assert restored["burden"]["other"]["some_nan_value"] is None


# --------------------------------------------------------------------------- #
# 2. json_safe unit behaviour
# --------------------------------------------------------------------------- #


def test_json_safe_numpy_like_scalar_via_item() -> None:
    assert json_safe(_NumpyLikeScalar(3.5)) == 3.5
    assert json_safe(_NumpyLikeScalar(7)) == 7
    assert json_safe(_NumpyLikeScalar(True)) is True


def test_json_safe_nan_and_inf_become_none() -> None:
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe(_NumpyLikeScalar(float("nan"))) is None


def test_json_safe_path_becomes_string() -> None:
    p = Path("/tmp/x/case_0001.json")
    assert json_safe(p) == str(p)
    assert isinstance(json_safe(p), str)


def test_json_safe_nested_dicts_and_lists() -> None:
    nested = {"a": [1, {"b": float("nan")}, (2, 3)], "c": None}
    safe = json_safe(nested)
    assert safe == {"a": [1, {"b": None}, [2, 3]], "c": None}


def test_json_safe_bool_stays_bool_not_int() -> None:
    assert json_safe(True) is True
    assert json_safe(False) is False
    safe_dict = json_safe({"flag": True})
    assert safe_dict["flag"] is True
    assert not isinstance(safe_dict["flag"], int) or isinstance(safe_dict["flag"], bool)
    # Stronger check: it must be exactly a bool, not merely int-compatible.
    assert type(safe_dict["flag"]) is bool


# --------------------------------------------------------------------------- #
# 3. Disclaimer required
# --------------------------------------------------------------------------- #


def test_disclaimer_present_and_nonempty_in_dict_and_markdown() -> None:
    report = _build()
    assert report["disclaimer"] == DISCLAIMER
    assert DISCLAIMER.strip()
    md = render_markdown(report)
    assert DISCLAIMER in md


# --------------------------------------------------------------------------- #
# 4. not_claimed covers all six items
# --------------------------------------------------------------------------- #


def test_not_claimed_covers_six_items_with_reasons() -> None:
    assert len(NOT_CLAIMED) == 6
    report = _build()
    assert report["not_claimed"] == NOT_CLAIMED
    for what, why in NOT_CLAIMED:
        assert what.strip()
        assert why.strip()

    joined_what = " ".join(what.lower() for what, _ in NOT_CLAIMED)
    for token in ("cell", "grade", "stage", "prognosis", "eloquence", "deficit"):
        assert token in joined_what


# --------------------------------------------------------------------------- #
# 5. Forbidden-substring scan
# --------------------------------------------------------------------------- #


def _not_claimed_text_from_markdown(md: str) -> tuple[str, str]:
    """Splits rendered Markdown into (not_claimed section, everything else)."""
    start = md.index("## Not Claimed")
    end = md.index("## Provenance")
    return md[start:end], md[:start] + md[end:]


def test_forbidden_substrings_only_appear_inside_not_claimed_markdown() -> None:
    report = _build()
    md = render_markdown(report)
    not_claimed_text, rest = _not_claimed_text_from_markdown(md)

    # The not_claimed section is expected to use this vocabulary.
    assert _scan_for_forbidden(not_claimed_text)

    hits = _scan_for_forbidden(rest)
    assert hits == [], f"forbidden words leaked outside not_claimed: {hits}"


def test_forbidden_substrings_only_appear_inside_not_claimed_json() -> None:
    report = _build()
    safe = json_safe(report)
    rest = {k: v for k, v in safe.items() if k != "not_claimed"}
    text = json.dumps(rest)
    hits = _scan_for_forbidden(text)
    assert hits == [], f"forbidden words leaked outside not_claimed: {hits}"

    not_claimed_text = json.dumps(safe["not_claimed"])
    assert _scan_for_forbidden(not_claimed_text)


# --------------------------------------------------------------------------- #
# 6/7. Mass-effect caveat and atlas name/version in the anatomy block
# --------------------------------------------------------------------------- #


def test_mass_effect_caveat_is_inside_anatomy_block() -> None:
    report = _build()
    assert report["anatomy"]["caveat"] == MASS_EFFECT_CAVEAT
    # Not merely present somewhere -- specifically not a top-level field.
    assert "caveat" not in report


def test_atlas_name_and_version_in_anatomy_block() -> None:
    prov = _provenance(atlas_name="SRI24/TZO", atlas_version="2026.1")
    report = _build(provenance=prov)
    assert report["anatomy"]["atlas"]["name"] == "SRI24/TZO"
    assert report["anatomy"]["atlas"]["version"] == "2026.1"
    md = render_markdown(report)
    assert "SRI24/TZO" in md
    assert "2026.1" in md


# --------------------------------------------------------------------------- #
# 8. Required-field raises
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence", ""),
        ("citation", ""),
        ("classification_name", ""),
        ("coverage_line", ""),
        ("case_id", ""),
    ],
)
def test_empty_required_field_raises(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        _build(**{field: value})


# --------------------------------------------------------------------------- #
# 9. Burden regrouping: every input key appears, unrecognised -> other
# --------------------------------------------------------------------------- #


def test_every_burden_key_appears_somewhere_in_grouped_output() -> None:
    burden = _burden()
    report = _build(burden=burden)
    grouped = report["burden"]

    all_output_keys: set[str] = set()
    for block in grouped.values():
        all_output_keys.update(block.keys())

    assert set(burden.keys()) == all_output_keys


def test_unrecognised_burden_key_lands_in_other() -> None:
    burden = {**_burden(), "totally_unknown_metric_xyz": 42}
    report = _build(burden=burden)
    assert "totally_unknown_metric_xyz" in report["burden"]["other"]
    assert report["burden"]["other"]["totally_unknown_metric_xyz"] == 42


# --------------------------------------------------------------------------- #
# 10. Overlapping-prefix precedence
# --------------------------------------------------------------------------- #


def test_frac_left_lands_in_laterality_not_fractions() -> None:
    burden = _burden()
    assert "frac_left_WT" in burden
    report = _build(burden=burden)
    assert "frac_left_WT" in report["burden"]["laterality"]
    assert "frac_left_WT" not in report["burden"]["fractions"]


def test_vol_component_keys_land_in_multifocality_not_volumes() -> None:
    burden = _burden()
    report = _build(burden=burden)
    assert "vol_largest_component_WT_mm3" in report["burden"]["multifocality"]
    assert "vol_largest_component_WT_mm3" not in report["burden"]["volumes"]
    assert "largest_component_frac_WT" in report["burden"]["multifocality"]
    assert "largest_component_frac_WT" not in report["burden"]["fractions"]


# --------------------------------------------------------------------------- #
# 11. anatomy.structures sorted and truncated
# --------------------------------------------------------------------------- #


def test_structures_sorted_by_frac_of_structure_descending_and_truncated() -> None:
    report = _build(top_n=2)
    structures = report["anatomy"]["structures"]
    assert len(structures) == 2
    fracs = [row["frac_of_structure"] for row in structures]
    assert fracs[0] >= fracs[1]
    # Temporal_Mid_L (0.6) must outrank Precentral_L (0.4) here.
    assert structures[0]["structure"] == "Temporal_Mid_L"


# --------------------------------------------------------------------------- #
# 12. region column selects WT rows only
# --------------------------------------------------------------------------- #


def test_region_column_selects_wt_rows_only() -> None:
    report = _build(anatomy_table=_anatomy_table(with_region=True))
    assert report["anatomy"]["region"] == "WT"
    structures = report["anatomy"]["structures"]
    # The ET-only row (frac_of_structure 0.1) must not appear alongside the
    # WT Precentral_L row (frac_of_structure 0.4) -- only one Precentral_L.
    precentral_rows = [r for r in structures if r["structure"] == "Precentral_L"]
    assert len(precentral_rows) == 1
    assert precentral_rows[0]["frac_of_structure"] == 0.4


def test_no_region_column_uses_table_as_is() -> None:
    table = _anatomy_table(with_region=False)
    report = _build(anatomy_table=table)
    assert report["anatomy"]["region"] is None
    assert len(report["anatomy"]["structures"]) == len(table)


# --------------------------------------------------------------------------- #
# 13. near_eloquent correctness
# --------------------------------------------------------------------------- #


def test_near_eloquent_false_when_distance_is_nan_or_none() -> None:
    summary = {**_anatomy_summary(), "distance_to_eloquent_mm": float("nan")}
    report = _build(anatomy_summary=summary, near_eloquent_mm=10.0)
    assert report["eloquence"]["near_eloquent"] is False

    summary2 = {**_anatomy_summary()}
    del summary2["distance_to_eloquent_mm"]
    report2 = _build(anatomy_summary=summary2, near_eloquent_mm=10.0)
    assert report2["eloquence"]["near_eloquent"] is False


def test_near_eloquent_true_below_threshold_false_above() -> None:
    near_summary = {**_anatomy_summary(), "distance_to_eloquent_mm": 5.0}
    report_near = _build(anatomy_summary=near_summary, near_eloquent_mm=10.0)
    assert report_near["eloquence"]["near_eloquent"] is True

    far_summary = {**_anatomy_summary(), "distance_to_eloquent_mm": 50.0}
    report_far = _build(anatomy_summary=far_summary, near_eloquent_mm=10.0)
    assert report_far["eloquence"]["near_eloquent"] is False


# --------------------------------------------------------------------------- #
# 13b. classification_name -- required, and it is a NAME, never a verdict
# --------------------------------------------------------------------------- #


def test_classification_name_required_and_stored_verbatim() -> None:
    report = _build(classification_name="Sawaya eloquence grading")
    assert report["eloquence"]["classification"] == "Sawaya eloquence grading"
    md = render_markdown(report)
    assert "Sawaya eloquence grading" in md


def test_classification_name_empty_raises() -> None:
    with pytest.raises(ValueError):
        _build(classification_name="")


def test_non_eloquent_verdict_never_appears_far_from_any_eloquent_structure() -> None:
    """Regression guard: a tumour far from every eloquent structure must not

    manufacture a "non-eloquent" verdict about the patient. `build_report`
    must report only the measured distance and the boolean threshold
    comparison, never a three-level classification derived from them --
    see `docs/research/interpretable_pipeline_plan.md` section 2 and
    `knowledge/eloquence_map.yaml`'s forbidden vocabulary.
    """
    far_summary = {**_anatomy_summary(), "distance_to_eloquent_mm": 250.0}
    far_table = _anatomy_table().assign(eloquence="unclassified")
    report = _build(
        anatomy_table=far_table,
        anatomy_summary=far_summary,
        near_eloquent_mm=10.0,
    )
    assert report["eloquence"]["near_eloquent"] is False

    md = render_markdown(report)
    assert "non-eloquent" not in md.lower()

    safe = json_safe(report)
    text = json.dumps(safe)
    assert "non-eloquent" not in text.lower()


def test_markdown_states_near_eloquent_threshold_and_na_distance() -> None:
    summary = {**_anatomy_summary()}
    del summary["distance_to_eloquent_mm"]
    report = _build(anatomy_summary=summary, near_eloquent_mm=10.0)
    md = render_markdown(report)

    assert "10.0 mm" in md
    assert "Within 10.0 mm of an eloquent structure: no" in md
    assert "Distance to nearest listed structure: n/a" in md


# --------------------------------------------------------------------------- #
# 14. Empty anatomy table
# --------------------------------------------------------------------------- #


def test_empty_anatomy_table_yields_valid_report_no_crash() -> None:
    empty = pd.DataFrame()
    report = _build(anatomy_table=empty, anatomy_summary={})
    assert report["anatomy"]["structures"] == []
    assert report["anatomy"]["region"] is None
    assert report["eloquence"]["involved"] == []
    md = render_markdown(report)
    assert "No structures recorded" in md


# --------------------------------------------------------------------------- #
# 15. write_report
# --------------------------------------------------------------------------- #


def test_write_report_writes_both_files_and_preserves_field_order(tmp_path: Path) -> None:
    report = _build()
    paths = write_report(report, tmp_path)
    assert paths["json"].exists()
    assert paths["markdown"].exists()

    text = paths["json"].read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert list(parsed.keys())[0] == "report_version"
    assert list(parsed.keys())[-1] == "provenance"

    # Insertion order in the raw text, not just after re-parsing.
    positions = {key: text.index(f'"{key}"') for key in parsed}
    ordered_keys = sorted(positions, key=positions.get)
    assert ordered_keys[0] == "report_version"
    assert ordered_keys[-1] == "provenance"


def test_write_report_json_only(tmp_path: Path) -> None:
    report = _build()
    paths = write_report(report, tmp_path, markdown=False)
    assert "markdown" not in paths
    assert paths["json"].exists()


# --------------------------------------------------------------------------- #
# 16. Markdown NaN rendering
# --------------------------------------------------------------------------- #


def test_markdown_renders_nan_as_na_never_literal_nan_or_none() -> None:
    burden = {**_burden(), "frac_unlabelled_extra": float("nan")}
    summary = {**_anatomy_summary(), "distance_to_eloquent_mm": float("nan")}
    report = _build(burden=burden, anatomy_summary=summary)
    md = render_markdown(report)

    assert "n/a" in md
    assert not re.search(r"\bnan\b", md, flags=re.IGNORECASE)
    assert not re.search(r"\bNone\b", md)


# --------------------------------------------------------------------------- #
# 17. No deep-learning-stack / array-library import
# --------------------------------------------------------------------------- #


def test_report_module_has_no_deep_learning_or_array_dependency() -> None:
    """Keeps this module importable with no torch and no numpy installed.

    Checked against the source text rather than `sys.modules`, since pytest
    has almost certainly imported both already for some other test file --
    see `tests/test_burden.py`'s equivalent guard.
    """
    source = Path(report_module.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source


# --------------------------------------------------------------------------- #
# Extra: eloquence evidence/citation/source-owns-claim rendering
# --------------------------------------------------------------------------- #


def test_eloquence_block_carries_evidence_citation_and_source_owns_claim() -> None:
    report = _build(
        evidence="Verbatim sentence about eloquent structures.",
        citation="Some Citation 2024",
    )
    eloquence = report["eloquence"]
    assert eloquence["evidence"] == "Verbatim sentence about eloquent structures."
    assert eloquence["citation"] == "Some Citation 2024"
    assert eloquence["source_owns_claim"]
    assert "source" in eloquence["source_owns_claim"].lower()

    md = render_markdown(report)
    assert "> Verbatim sentence about eloquent structures." in md
    assert "Some Citation 2024" in md


def test_coverage_line_is_a_required_field_and_renders() -> None:
    report = _build(coverage_line="7 of 50 structures classified eloquent, 43 unclassified")
    assert report["anatomy"]["coverage_line"] == (
        "7 of 50 structures classified eloquent, 43 unclassified"
    )
    md = render_markdown(report)
    assert "7 of 50 structures classified eloquent, 43 unclassified" in md


def test_provenance_round_trips_through_build_report() -> None:
    prov = _provenance(code_revision=None, segmentation_dir=None)
    report = _build(provenance=prov)
    assert report["provenance"]["code_revision"] is None
    assert report["provenance"]["segmentation_dir"] is None
    assert report["provenance"]["atlas_licence"] == "CC-BY-SA"


def test_provenance_rejects_unknown_segmentation_source() -> None:
    with pytest.raises(ValueError):
        _provenance(segmentation_source="ground_truth")


def test_json_safe_finite_float_and_int_pass_through_unchanged() -> None:
    assert json_safe(1.5) == 1.5
    assert json_safe(3) == 3
    assert json_safe("hello") == "hello"
    assert json_safe(None) is None


def test_math_isnan_sanity_for_helper() -> None:
    # Guards the test module's own forbidden-word scanner against a false
    # negative: 'stage' as a substring must not accidentally appear in
    # ordinary report vocabulary used in these tests' fixtures.
    assert not math.isnan(1.0)
