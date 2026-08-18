"""Tests for scripts/report.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_localize_script.py` and `tests/test_burden_script.py`
use.

Every test composes the REAL Hydra config (via `hydra.compose`, exactly like
`scripts/smoke_test.py`) and points `analysis.report.{burden_dir,localize_dir,
eloquence_map,lobe_map}` at tiny SYNTHETIC CSVs and YAMLs hand-written under
`tmp_path`. This script loads no atlas, no checkpoint, and no volume -- it is
a pure CSV join -- so the fixtures below are hand-written tables rather than
anything built with an atlas or a preprocessed case. Nothing here touches
real BraTS data, the real SRI24 atlas, or the real `outputs/` tree, and each
test is well under a second.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import hydra
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from neurovision.utils.io import write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report.py"
_spec = importlib.util.spec_from_file_location("report_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
report_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["report_script"] = report_script
_spec.loader.exec_module(report_script)

load_inputs = report_script.load_inputs
resolve_cases = report_script.resolve_cases
report_one = report_script.report_one
run_report = report_script.run_report
git_revision = report_script.git_revision

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

CASE_IDS = ["CASE_A", "CASE_B", "CASE_C"]
NEAR_ELOQUENT_MM = 5.0


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def _write_eloquence_yaml(
    path: Path, *, near_eloquent_mm: float = NEAR_ELOQUENT_MM, coverage_gaps: tuple = ()
) -> None:
    doc = {
        "version": 1,
        "classification": {
            "name": "Sawaya eloquence grading",
            "primary_citation": "Test R. A test classification. Test Journal. 1998.",
            "eloquent_structures_verbatim": "Eloquent locations are the motor cortex.",
        },
        "near_eloquent_rule": {"distance_mm": near_eloquent_mm},
        "coverage_gaps": [{"term": g} for g in coverage_gaps],
    }
    path.write_text(yaml.safe_dump(doc))


def _write_lobe_yaml(path: Path) -> None:
    path.write_text(yaml.safe_dump({"version": 1, "structures": {}}))


def _write_burden_csv(path: Path, case_ids: list[str]) -> None:
    rows = [
        {
            "case_id": case_id,
            "vol_WT_mm3": 10_000.0 + i,
            "frac_left_WT": 0.6,
            "sphericity_WT": 0.7,
            "n_components_WT": 1,
        }
        for i, case_id in enumerate(case_ids)
    ]
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


# CASE_A: one WT row is the eloquent structure, plus the mandatory unlabelled
# row. CASE_B: one WT row, non-eloquent. CASE_C: only ET/TC rows -- no WT row
# at all, the fixture for test 6 (empty anatomy.structures, not a raise).
_ANATOMY_ROWS: list[dict] = [
    {
        "case_id": "CASE_A",
        "region": "WT",
        "structure": "Precentral_L",
        "laterality": "L",
        "lobe": "frontal",
        "eloquence": "eloquent",
        "matched_term": "motor cortex",
        "n_voxels": 100,
        "volume_mm3": 100.0,
        "frac_of_tumour": 0.5,
        "frac_of_structure": 0.1,
    },
    {
        "case_id": "CASE_A",
        "region": "WT",
        "structure": "unlabelled",
        "laterality": "",
        "lobe": "",
        "eloquence": "unclassified",
        "matched_term": "",
        "n_voxels": 100,
        "volume_mm3": 100.0,
        "frac_of_tumour": 0.5,
        "frac_of_structure": float("nan"),
    },
    {
        "case_id": "CASE_B",
        "region": "WT",
        "structure": "Frontal_L",
        "laterality": "L",
        "lobe": "frontal",
        "eloquence": "unclassified",
        "matched_term": "",
        "n_voxels": 200,
        "volume_mm3": 200.0,
        "frac_of_tumour": 1.0,
        "frac_of_structure": 0.2,
    },
    {
        "case_id": "CASE_C",
        "region": "ET",
        "structure": "Frontal_L",
        "laterality": "L",
        "lobe": "frontal",
        "eloquence": "unclassified",
        "matched_term": "",
        "n_voxels": 50,
        "volume_mm3": 50.0,
        "frac_of_tumour": 1.0,
        "frac_of_structure": 0.05,
    },
    {
        "case_id": "CASE_C",
        "region": "TC",
        "structure": "Frontal_L",
        "laterality": "L",
        "lobe": "frontal",
        "eloquence": "unclassified",
        "matched_term": "",
        "n_voxels": 50,
        "volume_mm3": 50.0,
        "frac_of_tumour": 1.0,
        "frac_of_structure": 0.05,
    },
]

# CASE_A is inside the near-eloquent threshold (3.0mm <= 5.0mm), CASE_B is
# outside it (20.0mm > 5.0mm) -- test 5 needs one of each.
_ANATOMY_SUMMARY_ROWS: list[dict] = [
    {
        "case_id": "CASE_A",
        "n_structures_involved": 1,
        "frac_unlabelled": 0.5,
        "distance_to_eloquent_mm": 3.0,
        "near_eloquent": True,
    },
    {
        "case_id": "CASE_B",
        "n_structures_involved": 1,
        "frac_unlabelled": 0.0,
        "distance_to_eloquent_mm": 20.0,
        "near_eloquent": False,
    },
    {
        "case_id": "CASE_C",
        "n_structures_involved": 0,
        "frac_unlabelled": float("nan"),
        "distance_to_eloquent_mm": float("nan"),
        "near_eloquent": False,
    },
]


def _write_anatomy_csvs(localize_dir: Path) -> None:
    pd.DataFrame.from_records(_ANATOMY_ROWS).to_csv(localize_dir / "anatomy.csv", index=False)
    pd.DataFrame.from_records(_ANATOMY_SUMMARY_ROWS).to_csv(
        localize_dir / "anatomy_summary.csv", index=False
    )


def _write_provenance_configs(
    burden_dir: Path,
    localize_dir: Path,
    *,
    burden_source: str = "prediction",
    localize_source: str = "prediction",
    burden_split: str = "test",
    localize_split: str = "test",
    burden_resolved_dir: str,
    localize_resolved_dir: str,
) -> None:
    write_yaml(
        {
            "source": burden_source,
            "split": burden_split,
            "preprocessed_dir": "preprocessed",
            "resolved_source_dir": burden_resolved_dir,
            "out_name": "burden.csv",
        },
        burden_dir / "burden_config.yaml",
    )
    write_yaml(
        {
            "source": localize_source,
            "split": localize_split,
            "preprocessed_dir": "preprocessed",
            "resolved_source_dir": localize_resolved_dir,
            "eloquence_map": "knowledge/eloquence_map.yaml",
            "lobe_map": "knowledge/aal_lobes.yaml",
            "atlas": {
                "name": "SRI24/TZO",
                "version": "2.0",
                "source": "NITRC group_id=214",
                "licence": "CC-BY-SA",
            },
            "coverage_line": "1 of 1 structures classified eloquent, 0 unclassified",
        },
        localize_dir / "localize_config.yaml",
    )


def _build_fixture_tree(
    tmp_path: Path,
    *,
    burden_source: str = "prediction",
    localize_source: str = "prediction",
    burden_split: str = "test",
    localize_split: str = "test",
    same_resolved_dir: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """Builds burden_dir + localize_dir + eloquence/lobe YAMLs under `tmp_path`.

    Returns:
        `(burden_dir, localize_dir, eloquence_path, lobe_path)`.
    """
    burden_dir = tmp_path / "burden"
    localize_dir = tmp_path / "localize"
    burden_dir.mkdir()
    localize_dir.mkdir()

    shared_resolved_dir = str((tmp_path / "eval" / "predictions").resolve())
    burden_resolved_dir = shared_resolved_dir
    localize_resolved_dir = (
        shared_resolved_dir if same_resolved_dir else str((tmp_path / "other_eval").resolve())
    )

    _write_burden_csv(burden_dir / "burden.csv", CASE_IDS)
    _write_anatomy_csvs(localize_dir)
    _write_provenance_configs(
        burden_dir,
        localize_dir,
        burden_source=burden_source,
        localize_source=localize_source,
        burden_split=burden_split,
        localize_split=localize_split,
        burden_resolved_dir=burden_resolved_dir,
        localize_resolved_dir=localize_resolved_dir,
    )

    eloquence_path = tmp_path / "eloquence_map.yaml"
    _write_eloquence_yaml(eloquence_path)
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_lobe_yaml(lobe_path)

    return burden_dir, localize_dir, eloquence_path, lobe_path


def _compose_cfg(
    tmp_path: Path,
    output_dir: Path,
    burden_dir: Path,
    localize_dir: Path,
    eloquence_path: Path,
    lobe_path: Path,
    *,
    cases: list[str] | None = None,
    markdown: bool | None = None,
    top_n: int | None = None,
):
    """Composes the real Hydra config, pointing analysis.report.* at tmp_path fixtures."""
    overrides = [
        f"data.root_dir={tmp_path}",
        f"output_dir={output_dir}",
        f"analysis.report.burden_dir={burden_dir}",
        f"analysis.report.localize_dir={localize_dir}",
        f"analysis.report.eloquence_map={eloquence_path}",
        f"analysis.report.lobe_map={lobe_path}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
    ]
    if markdown is not None:
        overrides.append(f"analysis.report.markdown={str(bool(markdown)).lower()}")
    if top_n is not None:
        overrides.append(f"analysis.report.top_n={int(top_n)}")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    if cases is not None:
        OmegaConf.set_struct(cfg, False)
        cfg.analysis.report.cases = list(cases)

    return cfg


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_writes_json_md_and_manifest(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    manifest_path = run_report(cfg)
    assert manifest_path == output_dir / "reports" / "report_manifest.csv"

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == len(CASE_IDS)
    assert set(manifest["case_id"]) == set(CASE_IDS)

    reports_dir = output_dir / "reports"
    for case_id in CASE_IDS:
        json_path = reports_dir / f"{case_id}.json"
        md_path = reports_dir / f"{case_id}.md"
        assert json_path.is_file()
        assert md_path.is_file()

        with open(json_path, encoding="utf-8") as f:
            report = json.load(f)  # raises if not strict JSON

        assert report["case_id"] == case_id
        assert "report_version" in report
        assert "disclaimer" in report
        assert "not_claimed" in report
        provenance = report["provenance"]
        assert provenance["segmentation_source"] == "prediction"

    config_record_path = reports_dir / "report_config.yaml"
    assert config_record_path.is_file()


# ---------------------------------------------------------------------------
# 2. The mismatch guard
# ---------------------------------------------------------------------------


def test_source_mismatch_raises_naming_both_dirs(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(
        tmp_path, burden_source="label", localize_source="prediction"
    )
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    with pytest.raises(ValueError) as excinfo:
        load_inputs(cfg)
    message = str(excinfo.value)
    assert str(burden_dir) in message
    assert str(localize_dir) in message
    assert "source" in message


def test_split_mismatch_raises_naming_both_dirs(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(
        tmp_path, burden_split="val", localize_split="test"
    )
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    with pytest.raises(ValueError) as excinfo:
        load_inputs(cfg)
    message = str(excinfo.value)
    assert str(burden_dir) in message
    assert str(localize_dir) in message
    assert "split" in message


def test_resolved_source_dir_mismatch_raises(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(
        tmp_path, same_resolved_dir=False
    )
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    with pytest.raises(ValueError) as excinfo:
        load_inputs(cfg)
    message = str(excinfo.value)
    assert str(burden_dir) in message
    assert str(localize_dir) in message
    assert "resolved_source_dir" in message


# ---------------------------------------------------------------------------
# 3. markdown: false writes JSON only
# ---------------------------------------------------------------------------


def test_markdown_disabled_writes_json_only(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path, markdown=False
    )

    manifest_path = run_report(cfg)
    manifest = pd.read_csv(manifest_path)

    reports_dir = output_dir / "reports"
    for case_id in CASE_IDS:
        assert (reports_dir / f"{case_id}.json").is_file()
        assert not (reports_dir / f"{case_id}.md").exists()

    md_paths = manifest.set_index("case_id")["markdown_path"]
    for case_id in CASE_IDS:
        # An empty CSV cell round-trips as NaN through pandas, not "".
        value = md_paths.loc[case_id]
        assert value == "" or pd.isna(value)


# ---------------------------------------------------------------------------
# 4. Explicit cases naming a missing one raises
# ---------------------------------------------------------------------------


def test_explicit_missing_case_raises(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        output_dir,
        burden_dir,
        localize_dir,
        eloq_path,
        lobe_path,
        cases=["CASE_A", "BraTS_does_not_exist"],
    )

    with pytest.raises(ValueError, match="BraTS_does_not_exist"):
        run_report(cfg)


# ---------------------------------------------------------------------------
# 5. near_eloquent traces back to the knowledge file's threshold
# ---------------------------------------------------------------------------


def test_near_eloquent_matches_summary_csv_and_uses_knowledge_threshold(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    run_report(cfg)
    reports_dir = output_dir / "reports"

    with open(reports_dir / "CASE_A.json", encoding="utf-8") as f:
        report_a = json.load(f)
    with open(reports_dir / "CASE_B.json", encoding="utf-8") as f:
        report_b = json.load(f)

    # CASE_A: distance 3.0mm <= threshold 5.0mm -> near. CASE_B: 20.0mm -> not.
    assert report_a["eloquence"]["near_eloquent"] is True
    assert report_b["eloquence"]["near_eloquent"] is False

    summary = pd.read_csv(localize_dir / "anatomy_summary.csv").set_index("case_id")
    assert report_a["eloquence"]["near_eloquent"] == bool(summary.loc["CASE_A", "near_eloquent"])
    assert report_b["eloquence"]["near_eloquent"] == bool(summary.loc["CASE_B", "near_eloquent"])

    assert report_a["eloquence"]["near_eloquent_threshold_mm"] == NEAR_ELOQUENT_MM


# ---------------------------------------------------------------------------
# 6. A case with no WT rows produces a valid report, not a raise
# ---------------------------------------------------------------------------


def test_case_with_no_wt_rows_gives_empty_structures(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    run_report(cfg)
    with open(output_dir / "reports" / "CASE_C.json", encoding="utf-8") as f:
        report_c = json.load(f)

    assert report_c["anatomy"]["structures"] == []
    assert report_c["anatomy"]["region"] == "WT"


# ---------------------------------------------------------------------------
# 7. Manifest row count and n_eloquent_involved
# ---------------------------------------------------------------------------


def test_manifest_row_count_and_eloquent_count(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    manifest_path = run_report(cfg)
    manifest = pd.read_csv(manifest_path).set_index("case_id")

    assert len(manifest) == len(CASE_IDS)
    assert int(manifest.loc["CASE_A", "n_eloquent_involved"]) == 1
    assert int(manifest.loc["CASE_B", "n_eloquent_involved"]) == 0
    assert int(manifest.loc["CASE_C", "n_eloquent_involved"]) == 0


# ---------------------------------------------------------------------------
# 8. git_revision on a non-repo directory returns None
# ---------------------------------------------------------------------------


def test_git_revision_on_non_repo_returns_none(tmp_path: Path) -> None:
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    assert git_revision(not_a_repo) is None


# ---------------------------------------------------------------------------
# Extra: resolve_cases default intersection and report_one directly
# ---------------------------------------------------------------------------


def test_resolve_cases_default_is_sorted_intersection(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    inputs = load_inputs(cfg)
    cases = resolve_cases(inputs, None)
    assert cases == sorted(CASE_IDS)


def test_resolve_cases_empty_requested_raises(tmp_path: Path) -> None:
    burden_dir, localize_dir, eloq_path, lobe_path = _build_fixture_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, burden_dir, localize_dir, eloq_path, lobe_path)

    inputs = load_inputs(cfg)
    with pytest.raises(ValueError):
        resolve_cases(inputs, [])
