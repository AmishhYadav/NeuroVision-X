"""Tests for scripts/report_agreement.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_report_script.py` and `scripts/smoke_test.py`
use.

Every test composes the REAL Hydra config (via `hydra.compose`, exactly like
`tests/test_report_script.py`) and points `analysis.report_agreement.{gt_dir,
pred_dirs}` at tiny SCHEMA-VALID report JSONs written under `tmp_path`. The
fixtures are built with `neurovision.reporting.report.build_report` /
`write_report`, never hand-written dicts, so they cannot drift from the real
report schema this script's library (`analysis/report_agreement.py`) reads.
Nothing here touches real BraTS data or the real `outputs/` tree, and the
whole file runs in a few seconds (every test overrides `n_boot` down from the
production 10000 except the reproducibility test, which needs a nonzero
`n_boot` to be a meaningful check at all).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import hydra
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.reporting.report import Provenance, build_report, write_report

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_agreement.py"
_spec = importlib.util.spec_from_file_location("report_agreement_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
report_agreement_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["report_agreement_script"] = report_agreement_script
_spec.loader.exec_module(report_agreement_script)

resolve_pred_dirs = report_agreement_script.resolve_pred_dirs
summarize = report_agreement_script.summarize
run_report_agreement = report_agreement_script.run_report_agreement

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

CASE_IDS = ["CASE_1", "CASE_2", "CASE_3", "CASE_4"]

# One structure gt keeps throughout, and the one model_b swaps in only for
# CASE_2 -- disjoint from the first, so jaccard/precision/recall/match_top all
# come out to exactly 0.0 at that case.
_STRUCT_PRECENTRAL = {
    "structure": "Precentral_L",
    "laterality": "L",
    "lobe": "frontal",
    "eloquence": "eloquent",
    "frac_of_tumour": 0.5,
    "frac_of_structure": 0.5,
}
_STRUCT_FRONTAL = {
    "structure": "Frontal_L",
    "laterality": "L",
    "lobe": "frontal",
    "eloquence": "unclassified",
    "frac_of_tumour": 0.5,
    "frac_of_structure": 0.5,
}

# Ground truth: vol_TC_mm3 is 1000.0 for the first three cases and a tiny
# 10.0 for CASE_4, which is what turns a plausible model_a error at CASE_4
# into a huge relerr_vol_TC (see the module docstring's measured max of 128.6
# on the real split -- this fixture reproduces the same shape on purpose).
_GT_VOL_TC = {"CASE_1": 1000.0, "CASE_2": 1000.0, "CASE_3": 1000.0, "CASE_4": 10.0}

# model_a: exact volume match on CASE_1/2, a modest +10% miss on CASE_3
# (relerr_vol_TC = 0.1, the "known ratio" case), and a wildly wrong absolute
# prediction on CASE_4 that -- because the ground truth there is tiny --
# scores relerr_vol_TC = |1000 - 10| / 10 = 99.0. Structures always match gt.
_MODEL_A_VOL_TC = {"CASE_1": 1000.0, "CASE_2": 1000.0, "CASE_3": 1100.0, "CASE_4": 1000.0}

# model_b: volumes always match gt exactly (relerr_vol_TC == 0.0 every case),
# but its structure list disagrees with gt at CASE_2.
_MODEL_B_VOL_TC = {"CASE_1": 1000.0, "CASE_2": 1000.0, "CASE_3": 1000.0, "CASE_4": 10.0}


def _make_report(
    case_id: str,
    *,
    structures: list[dict],
    vol_tc: float,
    segmentation_source: str,
) -> dict:
    """Builds one schema-valid `build_report` dict for the fixtures below.

    Every field not under test (eloquence distance, n_structures_involved,
    the other two volumes, multifocality, laterality) is held fixed across
    gt/model_a/model_b/every case, so only `structures` and `vol_tc` can move
    a comparison metric -- which is what makes the expected values in the
    tests below hand-computable.
    """
    burden = {
        "vol_ET_mm3": 100.0,
        "vol_TC_mm3": vol_tc,
        "vol_WT_mm3": 2000.0,
        "frac_enhancing_of_wt": 0.05,
        "n_components_WT": 1,
        "dominant_side_WT": "L",
    }
    anatomy_table = pd.DataFrame.from_records([{"region": "WT", **s} for s in structures])
    anatomy_summary = {
        "n_structures_involved": len(structures),
        "frac_unlabelled": 0.1,
        "distance_to_eloquent_mm": 3.0,  # inside the 5.0mm threshold on every case
    }
    provenance = Provenance(
        atlas_name="SRI24/TZO",
        atlas_version="2.0",
        atlas_source="NITRC group_id=214",
        atlas_licence="CC-BY-SA",
        knowledge_versions={"eloquence_map": 1, "aal_lobes": 1},
        segmentation_source=segmentation_source,
        segmentation_dir=None,
        code_revision=None,
        generated_utc="2026-08-01T00:00:00Z",
    )
    return build_report(
        case_id,
        burden,
        anatomy_table,
        anatomy_summary,
        provenance,
        evidence="Eloquent locations are the motor cortex.",
        citation="Test R. A test classification. Test Journal. 1998.",
        classification_name="Sawaya eloquence grading",
        coverage_line="1 of 1 structures classified eloquent, 0 unclassified",
        coverage_gaps=[],
        near_eloquent_mm=5.0,
        top_n=10,
    )


def _build_reports_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Writes gt/model_a/model_b report JSONs for `CASE_IDS` under `tmp_path`.

    Returns:
        `(gt_dir, model_a_dir, model_b_dir)`.
    """
    gt_dir = tmp_path / "report_gt" / "reports"
    model_a_dir = tmp_path / "report_model_a" / "reports"
    model_b_dir = tmp_path / "report_model_b" / "reports"

    for case_id in CASE_IDS:
        write_report(
            _make_report(
                case_id,
                structures=[_STRUCT_PRECENTRAL],
                vol_tc=_GT_VOL_TC[case_id],
                segmentation_source="label",
            ),
            gt_dir,
            markdown=False,
        )
        write_report(
            _make_report(
                case_id,
                structures=[_STRUCT_PRECENTRAL],
                vol_tc=_MODEL_A_VOL_TC[case_id],
                segmentation_source="prediction",
            ),
            model_a_dir,
            markdown=False,
        )
        model_b_structures = [_STRUCT_FRONTAL] if case_id == "CASE_2" else [_STRUCT_PRECENTRAL]
        write_report(
            _make_report(
                case_id,
                structures=model_b_structures,
                vol_tc=_MODEL_B_VOL_TC[case_id],
                segmentation_source="prediction",
            ),
            model_b_dir,
            markdown=False,
        )

    return gt_dir, model_a_dir, model_b_dir


def _compose_cfg(
    tmp_path: Path,
    output_dir: Path,
    gt_dir: Path | None,
    pred_dirs: dict[str, Path] | None,
    *,
    comparisons: list[list[str]] | None = None,
    cases: list[str] | None = None,
    n_boot: int = 200,
):
    """Composes the real Hydra config, pointing analysis.report_agreement.* at fixtures."""
    overrides = [
        f"data.root_dir={tmp_path}",
        f"output_dir={output_dir}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
        f"analysis.report_agreement.n_boot={int(n_boot)}",
    ]
    if gt_dir is not None:
        overrides.append(f"analysis.report_agreement.gt_dir={gt_dir}")
    if pred_dirs:
        for name, path in pred_dirs.items():
            overrides.append(f"+analysis.report_agreement.pred_dirs.{name}={path}")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    if comparisons is not None or cases is not None:
        OmegaConf.set_struct(cfg, False)
        if comparisons is not None:
            cfg.analysis.report_agreement.comparisons = [list(pair) for pair in comparisons]
        if cases is not None:
            cfg.analysis.report_agreement.cases = list(cases)

    return cfg


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_happy_path_writes_per_model_and_summary_and_config(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, output_dir, gt_dir, {"model_a": model_a_dir, "model_b": model_b_dir}
    )

    summary_path = run_report_agreement(cfg)
    assert summary_path == output_dir / "agreement_summary.csv"
    assert summary_path.is_file()

    for name in ("model_a", "model_b"):
        per_case_path = output_dir / f"agreement_{name}.csv"
        assert per_case_path.is_file()
        table = pd.read_csv(per_case_path)
        assert table["case_id"].tolist() == CASE_IDS
        assert len(table) == len(CASE_IDS)

    assert (output_dir / "report_agreement_config.yaml").is_file()


# ---------------------------------------------------------------------------
# 2. agreement_summary.csv is long-format
# ---------------------------------------------------------------------------


def test_agreement_summary_is_long_format(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, output_dir, gt_dir, {"model_a": model_a_dir, "model_b": model_b_dir}
    )

    run_report_agreement(cfg)
    summary = pd.read_csv(output_dir / "agreement_summary.csv")

    assert list(summary.columns) == ["model", "metric", "mean", "median", "std", "n", "n_missing"]
    assert set(summary["model"]) == {"model_a", "model_b"}
    # One row per (model, metric): every model reports the same metric set,
    # since compare_reports always emits the same 16 keys.
    counts = summary.groupby("model")["metric"].nunique()
    assert counts["model_a"] == counts["model_b"]
    assert (summary.groupby(["model", "metric"]).size() == 1).all()


# ---------------------------------------------------------------------------
# 3. summarize() emits both mean and median, and they diverge on a skewed
#    column -- pins the docstring's claim about relerr_* outliers.
# ---------------------------------------------------------------------------


def test_summarize_mean_and_median_diverge_on_skewed_relerr_column(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, output_dir, gt_dir, {"model_a": model_a_dir, "model_b": model_b_dir}
    )

    run_report_agreement(cfg)
    summary = pd.read_csv(output_dir / "agreement_summary.csv").set_index(["model", "metric"])

    # model_a: relerr_vol_TC = [0.0, 0.0, 0.1, 99.0] -- one huge outlier at
    # CASE_4 (tiny gt denominator). mean = 24.775, median = 0.05.
    row_a = summary.loc[("model_a", "relerr_vol_TC")]
    assert row_a["mean"] == pytest.approx(24.775, abs=1e-6)
    assert row_a["median"] == pytest.approx(0.05, abs=1e-6)
    assert row_a["mean"] > 10.0 * max(row_a["median"], 1e-9)

    # model_b: no outlier -- relerr_vol_TC is 0.0 on every case, so mean and
    # median agree exactly. The contrast against model_a is the point: the
    # divergence above is the outlier's effect, not an artifact of the metric.
    row_b = summary.loc[("model_b", "relerr_vol_TC")]
    assert row_b["mean"] == pytest.approx(0.0, abs=1e-9)
    assert row_b["median"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_function_directly_reports_n_missing() -> None:
    table = pd.DataFrame({"m": [1.0, float("nan"), 3.0]})
    summary = summarize({"only_model": table})
    row = summary.set_index(["model", "metric"]).loc[("only_model", "m")]
    assert row["n"] == 3
    assert row["n_missing"] == 1
    assert row["mean"] == pytest.approx(2.0)
    assert row["median"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. comparisons writes CSV + TXT, one row per metric with a verdict column
# ---------------------------------------------------------------------------


def test_comparisons_writes_csv_and_txt_with_verdict_column(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        output_dir,
        gt_dir,
        {"model_a": model_a_dir, "model_b": model_b_dir},
        comparisons=[["model_a", "model_b"]],
    )

    run_report_agreement(cfg)

    csv_path = output_dir / "comparison_model_a_vs_model_b.csv"
    txt_path = output_dir / "comparison_model_a_vs_model_b.txt"
    assert csv_path.is_file()
    assert txt_path.is_file()

    comparison = pd.read_csv(csv_path)
    assert "metric" in comparison.columns
    assert "verdict" in comparison.columns
    # One row per metric, no duplicate metric names.
    assert comparison["metric"].is_unique
    assert len(comparison) > 0

    assert txt_path.read_text(encoding="utf-8").strip() != ""


# ---------------------------------------------------------------------------
# 5. A comparison naming a model absent from pred_dirs raises
# ---------------------------------------------------------------------------


def test_comparison_naming_unknown_model_raises(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        output_dir,
        gt_dir,
        {"model_a": model_a_dir, "model_b": model_b_dir},
        comparisons=[["model_a", "does_not_exist"]],
    )

    with pytest.raises(ValueError, match="does_not_exist"):
        run_report_agreement(cfg)


# ---------------------------------------------------------------------------
# 6. gt_dir unset, or pointing at a non-directory, raises
# ---------------------------------------------------------------------------


def test_gt_dir_unset_raises_naming_label_source(tmp_path: Path) -> None:
    _, model_a_dir, _ = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, None, {"model_a": model_a_dir})

    with pytest.raises(ValueError, match="label"):
        run_report_agreement(cfg)


def test_gt_dir_not_a_directory_raises_naming_label_source(tmp_path: Path) -> None:
    _, model_a_dir, _ = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    not_a_dir = tmp_path / "not_a_directory_anywhere"
    cfg = _compose_cfg(tmp_path, output_dir, not_a_dir, {"model_a": model_a_dir})

    with pytest.raises(ValueError) as excinfo:
        run_report_agreement(cfg)
    message = str(excinfo.value)
    assert "label" in message
    assert "ground" in message.lower()


# ---------------------------------------------------------------------------
# 7. Empty pred_dirs raises, suggesting the override form
# ---------------------------------------------------------------------------


def test_empty_pred_dirs_raises_suggesting_override_form(tmp_path: Path) -> None:
    gt_dir, _, _ = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, gt_dir, None)

    with pytest.raises(ValueError, match=r"\+analysis\.report_agreement\.pred_dirs"):
        run_report_agreement(cfg)


def test_resolve_pred_dirs_empty_raises_directly(tmp_path: Path) -> None:
    gt_dir, _, _ = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, gt_dir, None)

    with pytest.raises(ValueError, match=r"\+analysis\.report_agreement\.pred_dirs"):
        resolve_pred_dirs(cfg)


# ---------------------------------------------------------------------------
# 8. A named pred_dirs path that does not exist raises before any table is
#    written
# ---------------------------------------------------------------------------


def test_missing_named_pred_dir_raises_before_writing_output(tmp_path: Path) -> None:
    gt_dir, model_a_dir, _ = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    missing_dir = tmp_path / "report_typo" / "reports"
    cfg = _compose_cfg(
        tmp_path, output_dir, gt_dir, {"model_a": model_a_dir, "typo_model": missing_dir}
    )

    with pytest.raises(ValueError, match="typo_model"):
        run_report_agreement(cfg)

    # resolve_pred_dirs raises before run_report_agreement ever calls
    # ensure_dir(output_dir) or writes a single agreement_*.csv.
    assert list(output_dir.glob("agreement_*.csv")) == []


# ---------------------------------------------------------------------------
# 9. report_agreement_config.yaml records provenance
# ---------------------------------------------------------------------------


def test_config_yaml_records_provenance(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, output_dir, gt_dir, {"model_a": model_a_dir, "model_b": model_b_dir}
    )

    run_report_agreement(cfg)

    with open(output_dir / "report_agreement_config.yaml", encoding="utf-8") as f:
        import yaml

        record = yaml.safe_load(f)

    assert record["resolved_gt_dir"] == str(gt_dir.resolve())
    assert record["resolved_pred_dirs"] == {
        "model_a": str(model_a_dir.resolve()),
        "model_b": str(model_b_dir.resolve()),
    }
    assert record["n_cases"] == {"model_a": len(CASE_IDS), "model_b": len(CASE_IDS)}
    assert record["seed"] == 42
    assert record["report_agreement_version"] == 1


# ---------------------------------------------------------------------------
# 10. Reproducibility: same seed -> byte-identical comparison CSV
# ---------------------------------------------------------------------------


def test_same_seed_gives_byte_identical_comparison_csv(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)

    output_dir_1 = tmp_path / "out1"
    cfg_1 = _compose_cfg(
        tmp_path,
        output_dir_1,
        gt_dir,
        {"model_a": model_a_dir, "model_b": model_b_dir},
        comparisons=[["model_a", "model_b"]],
        n_boot=500,
    )
    run_report_agreement(cfg_1)

    output_dir_2 = tmp_path / "out2"
    cfg_2 = _compose_cfg(
        tmp_path,
        output_dir_2,
        gt_dir,
        {"model_a": model_a_dir, "model_b": model_b_dir},
        comparisons=[["model_a", "model_b"]],
        n_boot=500,
    )
    run_report_agreement(cfg_2)

    csv_1 = (output_dir_1 / "comparison_model_a_vs_model_b.csv").read_bytes()
    csv_2 = (output_dir_2 / "comparison_model_a_vs_model_b.csv").read_bytes()
    assert csv_1 == csv_2


# ---------------------------------------------------------------------------
# 11. cases: [...] restricts the comparison to the named cases only
# ---------------------------------------------------------------------------


def test_cases_override_restricts_to_named_cases(tmp_path: Path) -> None:
    gt_dir, model_a_dir, model_b_dir = _build_reports_tree(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        output_dir,
        gt_dir,
        {"model_a": model_a_dir, "model_b": model_b_dir},
        cases=["CASE_1", "CASE_3"],
    )

    run_report_agreement(cfg)

    table = pd.read_csv(output_dir / "agreement_model_a.csv")
    assert sorted(table["case_id"]) == ["CASE_1", "CASE_3"]
