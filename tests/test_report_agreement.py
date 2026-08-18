"""Tests for `neurovision.analysis.report_agreement`.

Every test runs on CPU against small hand-built report dicts (matching
`neurovision.reporting.report.build_report`'s schema), written to `tmp_path`
as JSON where a test needs `agreement_table`. Never reads a real report from
`outputs/`, never touches the deep-learning stack.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pytest

from neurovision.analysis import report_agreement
from neurovision.analysis.report_agreement import (
    REPORT_AGREEMENT_VERSION,
    agreement_table,
    compare_reports,
    jaccard,
    load_report,
    structure_set,
    top_structure,
)
from neurovision.analysis.statistics import metric_direction

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _structure_row(
    structure: str, frac_of_structure: float = 0.1, frac_of_tumour: float = 0.1
) -> dict[str, Any]:
    return {
        "structure": structure,
        "laterality": "left",
        "lobe": "frontal",
        "eloquence": "non_eloquent",
        "frac_of_tumour": frac_of_tumour,
        "frac_of_structure": frac_of_structure,
    }


def _eloquent_row(structure: str) -> dict[str, Any]:
    return {
        "structure": structure,
        "laterality": "left",
        "frac_of_tumour": 0.05,
        "frac_of_structure": 0.2,
    }


def _report(
    *,
    case_id: str = "BraTS2021_00002",
    report_version: int = 1,
    structures: list[dict[str, Any]] | None = None,
    n_structures_involved: int | None = None,
    eloquent_involved: list[dict[str, Any]] | None = None,
    near_eloquent: bool = False,
    distance_mm: float | None = 3.0,
    vol_et: float = 1000.0,
    vol_tc: float = 5000.0,
    vol_wt: float = 8000.0,
    frac_enhancing_of_wt: float = 0.125,
    n_components_wt: int = 1,
    dominant_side_wt: str = "left",
) -> dict[str, Any]:
    """Builds a small `build_report`-shaped dict for one case."""
    structures = [] if structures is None else structures
    eloquent_involved = [] if eloquent_involved is None else eloquent_involved
    if n_structures_involved is None:
        n_structures_involved = len(structures)
    return {
        "report_version": report_version,
        "case_id": case_id,
        "generated_utc": "2026-08-18T00:00:00Z",
        "disclaimer": "not a diagnostic tool",
        "not_claimed": [],
        "burden": {
            "volumes": {"vol_ET_mm3": vol_et, "vol_TC_mm3": vol_tc, "vol_WT_mm3": vol_wt},
            "fractions": {"frac_enhancing_of_wt": frac_enhancing_of_wt},
            "shape": {},
            "multifocality": {"n_components_WT": n_components_wt},
            "laterality": {"dominant_side_WT": dominant_side_wt},
            "centroid": {},
            "other": {},
        },
        "anatomy": {
            "atlas": {"name": "SRI24/TZO", "version": "1.0"},
            "caveat": "atlas caveat",
            "coverage_line": "coverage",
            "region": "WT",
            "structures": structures,
            "n_structures_involved": n_structures_involved,
            "frac_unlabelled": 0.3,
        },
        "eloquence": {
            "classification": "Sawaya eloquence grading",
            "citation": "Sawaya et al.",
            "evidence": "evidence sentence",
            "source_owns_claim": "source owns claim",
            "involved": eloquent_involved,
            "distance_mm": distance_mm,
            "near_eloquent_threshold_mm": 10.0,
            "near_eloquent": near_eloquent,
            "coverage_gaps": [],
        },
        "provenance": {},
    }


def _write(directory: Path, report: dict[str, Any]) -> Path:
    path = directory / f"{report['case_id']}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 1. Identical reports
# --------------------------------------------------------------------------- #


def test_identical_reports_agree_completely() -> None:
    structures = [_structure_row("Caudate_L", 0.9, 0.2), _structure_row("Insula_L", 0.5, 0.1)]
    eloquent = [_eloquent_row("Insula_L")]
    gt = _report(structures=structures, eloquent_involved=eloquent, near_eloquent=True)
    pred = _report(structures=structures, eloquent_involved=eloquent, near_eloquent=True)

    result = compare_reports(gt, pred)

    assert result["jaccard_structures"] == 1.0
    assert result["precision_structures"] == 1.0
    assert result["recall_structures"] == 1.0
    assert result["match_top_structure"] == 1.0
    assert result["agree_eloquent_any"] == 1.0
    assert result["jaccard_eloquent_structures"] == 1.0
    assert result["agree_near_eloquent"] == 1.0
    assert result["agree_multifocal_WT"] == 1.0
    assert result["agree_dominant_side_WT"] == 1.0

    for key, value in result.items():
        if key.startswith("abserr_") or key.startswith("relerr_"):
            assert value == 0.0, f"{key} expected 0.0, got {value}"


# --------------------------------------------------------------------------- #
# 2. Disjoint structure lists
# --------------------------------------------------------------------------- #


def test_disjoint_structure_lists() -> None:
    gt = _report(structures=[_structure_row("Caudate_L")])
    pred = _report(structures=[_structure_row("Putamen_R")])

    result = compare_reports(gt, pred)

    assert result["jaccard_structures"] == 0.0
    assert result["precision_structures"] == 0.0
    assert result["recall_structures"] == 0.0


# --------------------------------------------------------------------------- #
# 3. Hand-computed partial overlap
# --------------------------------------------------------------------------- #


def test_partial_overlap_exact_fractions() -> None:
    gt = _report(
        structures=[_structure_row("A"), _structure_row("B"), _structure_row("C")],
    )
    pred = _report(
        structures=[
            _structure_row("A"),
            _structure_row("B"),
            _structure_row("D"),
            _structure_row("E"),
        ],
    )

    result = compare_reports(gt, pred)

    assert result["jaccard_structures"] == pytest.approx(2 / 5)
    assert result["precision_structures"] == pytest.approx(2 / 4)
    assert result["recall_structures"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# 4. Both structure lists empty
# --------------------------------------------------------------------------- #


def test_both_structure_lists_empty() -> None:
    gt = _report(structures=[])
    pred = _report(structures=[])

    result = compare_reports(gt, pred)

    assert result["jaccard_structures"] == 1.0
    assert math.isnan(result["precision_structures"])
    assert math.isnan(result["recall_structures"])
    assert math.isnan(result["match_top_structure"])


# --------------------------------------------------------------------------- #
# 5. relerr_vol_ET is NaN, not inf, when gt volume is 0
# --------------------------------------------------------------------------- #


def test_relerr_vol_et_nan_not_inf_when_gt_zero() -> None:
    gt = _report(vol_et=0.0)
    pred = _report(vol_et=250.0)

    result = compare_reports(gt, pred)

    assert math.isnan(result["relerr_vol_ET"])
    assert not math.isinf(result["relerr_vol_ET"])


# --------------------------------------------------------------------------- #
# 6. abserr_distance_mm is NaN when either side missing/non-finite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_abserr_distance_mm_nan_when_missing(bad_value: float | None) -> None:
    gt = _report(distance_mm=bad_value)
    pred = _report(distance_mm=5.0)

    result = compare_reports(gt, pred)

    assert math.isnan(result["abserr_distance_mm"])

    gt2 = _report(distance_mm=5.0)
    pred2 = _report(distance_mm=bad_value)
    result2 = compare_reports(gt2, pred2)
    assert math.isnan(result2["abserr_distance_mm"])


# --------------------------------------------------------------------------- #
# 7. agree_multifocal_WT is about the multifocal call, not the count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("gt_n", "pred_n", "expected"),
    [
        (1, 1, 1.0),
        (3, 5, 1.0),
        (1, 2, 0.0),
    ],
)
def test_agree_multifocal_is_about_the_call_not_the_count(
    gt_n: int, pred_n: int, expected: float
) -> None:
    gt = _report(n_components_wt=gt_n)
    pred = _report(n_components_wt=pred_n)

    result = compare_reports(gt, pred)

    assert result["agree_multifocal_WT"] == expected


# --------------------------------------------------------------------------- #
# 8. Mismatched case_id raises
# --------------------------------------------------------------------------- #


def test_mismatched_case_id_raises() -> None:
    gt = _report(case_id="BraTS2021_00002")
    pred = _report(case_id="BraTS2021_00099")

    with pytest.raises(ValueError, match="BraTS2021_00002.*BraTS2021_00099|case_id mismatch"):
        compare_reports(gt, pred)


# --------------------------------------------------------------------------- #
# 9. Mismatched report_version raises
# --------------------------------------------------------------------------- #


def test_mismatched_report_version_raises() -> None:
    gt = _report(report_version=1)
    pred = _report(report_version=2)

    with pytest.raises(ValueError, match="report_version"):
        compare_reports(gt, pred)


# --------------------------------------------------------------------------- #
# 10. Missing required block raises rather than returning NaN row
# --------------------------------------------------------------------------- #


def test_missing_anatomy_block_raises() -> None:
    gt = _report()
    del gt["anatomy"]
    pred = _report()

    with pytest.raises(ValueError, match="anatomy"):
        compare_reports(gt, pred)


def test_missing_eloquence_block_raises() -> None:
    gt = _report()
    pred = _report()
    del pred["eloquence"]

    with pytest.raises(ValueError, match="eloquence"):
        compare_reports(gt, pred)


def test_missing_burden_block_raises() -> None:
    gt = _report()
    del gt["burden"]
    pred = _report()

    with pytest.raises(ValueError, match="burden"):
        compare_reports(gt, pred)


# --------------------------------------------------------------------------- #
# 11. agreement_table over 3 cases, one excluded
# --------------------------------------------------------------------------- #


def test_agreement_table_excludes_case_present_in_only_one_dir(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    for case_id in ("case_001", "case_002", "case_003"):
        _write(gt_dir, _report(case_id=case_id))
        _write(pred_dir, _report(case_id=case_id))
    # A fourth case only on the gt side.
    _write(gt_dir, _report(case_id="case_004_gt_only"))

    with caplog.at_level(logging.WARNING, logger=report_agreement.__name__):
        table = agreement_table(gt_dir, pred_dir)

    assert list(table.index) == ["case_001", "case_002", "case_003"]
    assert table.index.name == "case_id"
    assert any("only in gt_dir" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# 12. Every output column is recognised by metric_direction
# --------------------------------------------------------------------------- #


def test_every_column_has_a_known_metric_direction(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    structures = [_structure_row("Caudate_L", 0.9, 0.2)]
    for case_id in ("case_001", "case_002"):
        _write(gt_dir, _report(case_id=case_id, structures=structures))
        _write(pred_dir, _report(case_id=case_id, structures=structures))

    table = agreement_table(gt_dir, pred_dir)

    for column in table.columns:
        # Must not raise.
        metric_direction(column)


# --------------------------------------------------------------------------- #
# 13. A single unreadable/malformed case does not kill the run
# --------------------------------------------------------------------------- #


def test_one_malformed_case_does_not_kill_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    _write(gt_dir, _report(case_id="case_good"))
    _write(pred_dir, _report(case_id="case_good"))

    # Malformed JSON on the gt side for the second case.
    (gt_dir / "case_bad.json").write_text("{not valid json", encoding="utf-8")
    _write(pred_dir, _report(case_id="case_bad"))

    with caplog.at_level(logging.ERROR, logger=report_agreement.__name__):
        table = agreement_table(gt_dir, pred_dir)

    assert list(table.index) == ["case_good"]
    assert any("case_bad" in r.message for r in caplog.records)


def test_all_cases_failing_raises_runtime_error(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    (gt_dir / "case_bad.json").write_text("{not valid json", encoding="utf-8")
    (pred_dir / "case_bad.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="every one of"):
        agreement_table(gt_dir, pred_dir)


def test_agreement_table_raises_on_empty_case_set(tmp_path: Path) -> None:
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    with pytest.raises(ValueError, match="empty"):
        agreement_table(gt_dir, pred_dir)


# --------------------------------------------------------------------------- #
# 14. No deep-learning stack import
# --------------------------------------------------------------------------- #


def test_module_does_not_import_the_deep_learning_stack() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly already imported that stack for some other
    test file in the suite -- see `tests/test_localize.py`'s equivalent
    guard.
    """
    source = Path(report_agreement.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import monai" not in source


# --------------------------------------------------------------------------- #
# Small supporting-function tests
# --------------------------------------------------------------------------- #


def test_load_report_roundtrip(tmp_path: Path) -> None:
    report = _report()
    path = _write(tmp_path, report)

    loaded = load_report(path)

    assert loaded["case_id"] == report["case_id"]


def test_structure_set_and_top_structure() -> None:
    structures = [
        _structure_row("A", frac_of_structure=0.2),
        _structure_row("B", frac_of_structure=0.8),
    ]
    report = _report(structures=structures)

    assert structure_set(report) == {"A", "B"}
    assert top_structure(report) == "B"
    assert structure_set(report, eloquent_only=True) == set()
    assert top_structure(_report(structures=[])) is None


def test_jaccard_helper_directly() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_report_agreement_version_is_an_int() -> None:
    assert isinstance(REPORT_AGREEMENT_VERSION, int)
