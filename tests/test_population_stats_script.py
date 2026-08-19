"""Tests for scripts/population_stats.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_report_script.py`, `tests/test_localize_script.py`
and `scripts/smoke_test.py` use.

Every test composes the REAL Hydra config (via `hydra.compose`, exactly like
`tests/test_report_script.py`) and points `analysis.population.localize_dirs`
at tiny SYNTHETIC `anatomy.csv` / `anatomy_summary.csv` / `localize_config.yaml`
triples hand-written under `tmp_path`. This script loads no atlas, no
checkpoint and no volume -- it is a pure pandas aggregation over CSVs another
script already wrote -- so the fixtures below are hand-built tables rather
than anything produced with a real atlas or a real preprocessed case. Nothing
here touches real BraTS data, the real SRI24 atlas, or the real `outputs/`
tree, and each test is well under a second.

matplotlib is switched to the Agg backend at import time, the same way
`tests/test_figures.py` does, since `analysis.population.figures=true` calls
straight into `neurovision.visualization.figures`.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import hydra
import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from neurovision.utils.io import read_yaml  # noqa: E402

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "population_stats.py"
_spec = importlib.util.spec_from_file_location("population_stats_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
population_stats_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["population_stats_script"] = population_stats_script
_spec.loader.exec_module(population_stats_script)

load_localize_runs = population_stats_script.load_localize_runs
run_population_stats = population_stats_script.run_population_stats

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


# --------------------------------------------------------------------------- #
# Fixture builders
#
# Every number below is hand-computable -- see the report for the arithmetic.
# Each case carries exactly two atlas rows (`Frontal_L`, `unlabelled`), plus
# `Parietal_R` for the two cases named in `parietal_cases`, so that raising
# `min_frac_of_structure` above `Parietal_R`'s 0.02 has a visible effect while
# `Frontal_L` (0.5) always clears it.
# --------------------------------------------------------------------------- #

_FRONTAL_FRAC_STRUCTURE = 0.5
_FRONTAL_FRAC_TUMOUR = 0.4
_UNLABELLED_FRAC_TUMOUR = 0.6
_PARIETAL_FRAC_STRUCTURE = 0.02
_PARIETAL_FRAC_TUMOUR = 0.05


def _anatomy_rows(case_ids: list[str], *, parietal_cases: tuple[str, ...] = ()) -> list[dict]:
    """`anatomy.csv` rows: `Frontal_L` + `unlabelled` per case, `Parietal_R` on `parietal_cases`."""
    rows: list[dict] = []
    for case_id in case_ids:
        rows.append(
            {
                "case_id": case_id,
                "region": "WT",
                "structure": "Frontal_L",
                "laterality": "L",
                "lobe": "frontal",
                "eloquence": "eloquent",
                "matched_term": "motor cortex",
                "n_voxels": 100,
                "volume_mm3": 100.0,
                "frac_of_tumour": _FRONTAL_FRAC_TUMOUR,
                "frac_of_structure": _FRONTAL_FRAC_STRUCTURE,
            }
        )
        rows.append(
            {
                "case_id": case_id,
                "region": "WT",
                "structure": "unlabelled",
                "laterality": "",
                "lobe": "",
                "eloquence": "unclassified",
                "matched_term": "",
                "n_voxels": 150,
                "volume_mm3": 150.0,
                "frac_of_tumour": _UNLABELLED_FRAC_TUMOUR,
                "frac_of_structure": float("nan"),
            }
        )
        if case_id in parietal_cases:
            rows.append(
                {
                    "case_id": case_id,
                    "region": "WT",
                    "structure": "Parietal_R",
                    "laterality": "R",
                    "lobe": "parietal",
                    "eloquence": "unclassified",
                    "matched_term": "",
                    "n_voxels": 5,
                    "volume_mm3": 5.0,
                    "frac_of_tumour": _PARIETAL_FRAC_TUMOUR,
                    "frac_of_structure": _PARIETAL_FRAC_STRUCTURE,
                }
            )
    return rows


def _summary_rows(case_ids: list[str], *, all_near_eloquent: bool = False) -> list[dict]:
    """One `anatomy_summary.csv`-shaped row per case.

    With `all_near_eloquent=False` (the default), the mix below deliberately
    keeps `frac_any_eloquent` / `frac_near_eloquent` / `frac_distance_zero`
    away from 0.0 and 1.0, so the happy-path tests do not accidentally
    exercise the degenerate-field warning that test 10 exists to check.
    """
    rows: list[dict] = []
    for i, case_id in enumerate(case_ids):
        if all_near_eloquent:
            near = True
            distance = 0.0
        else:
            # Cycle through near/far so the population is not degenerate.
            near = i % 2 == 0
            distance = 0.0 if near else 10.0 + i
        rows.append(
            {
                "case_id": case_id,
                "n_structures_involved": 2 if near else 1,
                "top_structure": "Frontal_L",
                "top_frac_of_structure": _FRONTAL_FRAC_STRUCTURE,
                "most_displaced_structure": "Frontal_L",
                "frac_unlabelled": _UNLABELLED_FRAC_TUMOUR,
                "n_eloquent_structures": 1 if (near or i == 0) else 0,
                "eloquent_frac_of_tumour": _FRONTAL_FRAC_TUMOUR if (near or i == 0) else 0.0,
                "distance_to_eloquent_mm": distance,
                "dominant_lobe": "frontal",
                "coverage_line": "",
                "near_eloquent": near,
                "frac_of_tumour_retained": 1.0,
            }
        )
    return rows


def _write_localize_config(path: Path, *, source: str = "prediction", split: str = "test") -> None:
    doc = {
        "source": source,
        "eval_dir": None,
        "preprocessed_dir": "preprocessed",
        "split": split,
        "regions": ["ET", "TC", "WT"],
        "eloquence_map": "knowledge/eloquence_map.yaml",
        "lobe_map": "knowledge/aal_lobes.yaml",
        "min_frac": 0.001,
        "out_name": "anatomy.csv",
        "summary_name": "anatomy_summary.csv",
        "resolved_source_dir": f"/synthetic/{split}",
        "atlas": {
            "name": "SRI24/TZO",
            "version": "2.0",
            "source": "NITRC group_id=214",
            "licence": "CC-BY-SA",
        },
        "coverage_line": "23 of 122 structures classified eloquent, 0 unclassified",
        "involvement": {"enabled": False},
    }
    path.write_text(yaml.safe_dump(doc))


def _build_localize_dir(
    tmp_path: Path,
    name: str,
    case_ids: list[str],
    *,
    split: str = "test",
    source: str = "prediction",
    parietal_cases: tuple[str, ...] = (),
    all_near_eloquent: bool = False,
    omit: str | None = None,
) -> Path:
    """Builds one `scripts/localize.py`-shaped output directory under `tmp_path`.

    Args:
        omit: One of `"anatomy.csv"` / `"anatomy_summary.csv"` /
            `"localize_config.yaml"` to leave that file out, for test 7.
    """
    directory = tmp_path / name
    directory.mkdir()

    if omit != "anatomy.csv":
        pd.DataFrame.from_records(_anatomy_rows(case_ids, parietal_cases=parietal_cases)).to_csv(
            directory / "anatomy.csv", index=False
        )
    if omit != "anatomy_summary.csv":
        pd.DataFrame.from_records(
            _summary_rows(case_ids, all_near_eloquent=all_near_eloquent)
        ).to_csv(directory / "anatomy_summary.csv", index=False)
    if omit != "localize_config.yaml":
        _write_localize_config(directory / "localize_config.yaml", source=source, split=split)

    return directory


def _compose_cfg(
    tmp_path: Path,
    output_dir: Path,
    *,
    figures: bool | None = None,
    min_frac_of_structure: float | None = None,
):
    """Composes the real Hydra config. `localize_dirs` is set AFTER compose (a list)."""
    overrides = [
        f"data.root_dir={tmp_path}",
        f"output_dir={output_dir}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
    ]
    if figures is not None:
        overrides.append(f"analysis.population.figures={str(bool(figures)).lower()}")
    if min_frac_of_structure is not None:
        overrides.append(f"analysis.population.min_frac_of_structure={min_frac_of_structure}")

    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _set_localize_dirs(cfg, dirs: list[Path]) -> None:
    OmegaConf.set_struct(cfg, False)
    cfg.analysis.population.localize_dirs = [str(d) for d in dirs]


# ---------------------------------------------------------------------------
# 1. Happy path over one directory
# ---------------------------------------------------------------------------


def test_happy_path_one_directory_writes_all_outputs(tmp_path: Path) -> None:
    case_ids = [f"CASE_{i}" for i in range(5)]
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids, parietal_cases=("CASE_0",))
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir)
    _set_localize_dirs(cfg, [localize_dir])

    structures_path = run_population_stats(cfg)
    assert structures_path == output_dir / "population_structures.csv"

    for name in (
        "population_structures.csv",
        "population_lobes.csv",
        "population_laterality.csv",
        "population_eloquence.json",
        "population_config.yaml",
    ):
        assert (output_dir / name).is_file(), name

    structures = pd.read_csv(structures_path)
    frontal = structures[structures["structure"] == "Frontal_L"].iloc[0]
    assert int(frontal["n_cases"]) == 5
    assert int(frontal["n_cases_involved"]) == 5
    assert frontal["frac_cases_involved"] == pytest.approx(1.0)
    assert frontal["median_frac_of_structure"] == pytest.approx(_FRONTAL_FRAC_STRUCTURE)
    assert frontal["median_frac_of_tumour"] == pytest.approx(_FRONTAL_FRAC_TUMOUR)
    # "unlabelled" is excluded from the structure table by default.
    assert "unlabelled" not in set(structures["structure"])

    lobes = pd.read_csv(output_dir / "population_lobes.csv")
    frontal_lobe = lobes[lobes["lobe"] == "frontal"].iloc[0]
    assert frontal_lobe["total_frac_of_tumour"] == pytest.approx(_FRONTAL_FRAC_TUMOUR)
    unlabelled_lobe = lobes[lobes["lobe"] == "unlabelled"].iloc[0]
    assert unlabelled_lobe["total_frac_of_tumour"] == pytest.approx(_UNLABELLED_FRAC_TUMOUR)

    laterality = pd.read_csv(output_dir / "population_laterality.csv")
    left = laterality[laterality["laterality"] == "L"].iloc[0]
    assert left["mean_frac_of_tumour"] == pytest.approx(_FRONTAL_FRAC_TUMOUR)


# ---------------------------------------------------------------------------
# 2. Two directories, different splits: counts sum, denominator is the cohort
# ---------------------------------------------------------------------------


def test_two_directories_different_splits_sums_case_counts(tmp_path: Path) -> None:
    case_ids_a = [f"CASE_A{i}" for i in range(5)]
    case_ids_b = [f"CASE_B{i}" for i in range(3)]
    dir_a = _build_localize_dir(tmp_path, "run_test", case_ids_a, split="test")
    dir_b = _build_localize_dir(tmp_path, "run_val", case_ids_b, split="val")
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir)
    _set_localize_dirs(cfg, [dir_a, dir_b])

    run_population_stats(cfg)

    structures = pd.read_csv(output_dir / "population_structures.csv")
    frontal = structures[structures["structure"] == "Frontal_L"].iloc[0]
    # Frontal_L is involved in every case of both directories: concatenation,
    # not just one split, decides both the numerator and the denominator.
    assert int(frontal["n_cases"]) == 8
    assert int(frontal["n_cases_involved"]) == 8
    assert frontal["frac_cases_involved"] == pytest.approx(1.0)

    config_record = read_yaml(output_dir / "population_config.yaml")
    assert config_record["n_cases"] == 8
    assert sorted(config_record["splits"]) == ["test", "val"]


# ---------------------------------------------------------------------------
# 3. Mixed-source guard
# ---------------------------------------------------------------------------


def test_mixed_source_raises_naming_both_sources(tmp_path: Path) -> None:
    """A cohort must not be half real anatomy (label) and half a model's opinion (prediction)."""
    dir_a = _build_localize_dir(tmp_path, "run_label", ["C1", "C2"], source="label", split="test")
    dir_b = _build_localize_dir(
        tmp_path, "run_pred", ["C3", "C4"], source="prediction", split="val"
    )

    with pytest.raises(ValueError) as excinfo:
        load_localize_runs([dir_a, dir_b])
    message = str(excinfo.value)
    assert "label" in message
    assert "prediction" in message


# ---------------------------------------------------------------------------
# 4. Repeated-split guard
# ---------------------------------------------------------------------------


def test_repeated_split_raises_naming_the_split(tmp_path: Path) -> None:
    """The same split named twice doubles every COUNT while leaving every FRACTION unchanged.

    frac_cases_involved and every mean/median in the population tables are
    ratios, so counting one split's cases twice would not show up as an
    implausible number anywhere in the output -- only the raw case count
    would be wrong, and nothing downstream compares it against an
    independent source of truth. The guard has to fire before that
    concatenation happens, which is why it is checked from the parsed
    configs alone rather than from anything derived from the data.
    """
    case_ids_a = ["C1", "C2"]
    case_ids_b = ["C3", "C4"]
    dir_a = _build_localize_dir(tmp_path, "run_a", case_ids_a, split="test")
    dir_b = _build_localize_dir(tmp_path, "run_b", case_ids_b, split="test")

    with pytest.raises(ValueError) as excinfo:
        load_localize_runs([dir_a, dir_b])
    message = str(excinfo.value)
    assert "test" in message


# ---------------------------------------------------------------------------
# 5. Duplicated case_id guard
# ---------------------------------------------------------------------------


def test_duplicated_case_id_raises_naming_count_and_offenders(tmp_path: Path) -> None:
    # Different splits (so the split guard does not fire first), but sharing
    # one case_id -- e.g. a case accidentally localised into two split runs.
    case_ids_a = ["SHARED_CASE", "C2"]
    case_ids_b = ["SHARED_CASE", "C4"]
    dir_a = _build_localize_dir(tmp_path, "run_a", case_ids_a, split="test")
    dir_b = _build_localize_dir(tmp_path, "run_b", case_ids_b, split="val")

    with pytest.raises(ValueError) as excinfo:
        load_localize_runs([dir_a, dir_b])
    message = str(excinfo.value)
    assert "1" in message
    assert "SHARED_CASE" in message


# ---------------------------------------------------------------------------
# 6. Empty localize_dirs
# ---------------------------------------------------------------------------


def test_empty_localize_dirs_raises() -> None:
    with pytest.raises(ValueError):
        load_localize_runs([])


# ---------------------------------------------------------------------------
# 7. A directory missing one of the three required files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omit", ["anatomy.csv", "anatomy_summary.csv", "localize_config.yaml"])
def test_missing_required_file_raises_naming_it(tmp_path: Path, omit: str) -> None:
    directory = _build_localize_dir(tmp_path, "run_a", ["C1", "C2"], omit=omit)

    with pytest.raises(ValueError, match=omit.replace(".", r"\.")):
        load_localize_runs([directory])


# ---------------------------------------------------------------------------
# 8. figures: true/false
# ---------------------------------------------------------------------------


def test_figures_true_writes_both_pdf_and_pairs(tmp_path: Path) -> None:
    case_ids = [f"CASE_{i}" for i in range(4)]
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, figures=True)
    _set_localize_dirs(cfg, [localize_dir])

    run_population_stats(cfg)

    figure_dir = output_dir / "figures"
    for stem in ("population_structure_involvement", "population_lobe_distribution"):
        for ext in ("pdf", "png"):
            assert (figure_dir / f"{stem}.{ext}").is_file(), f"{stem}.{ext}"

    # The tables are written regardless of the figures flag.
    assert (output_dir / "population_structures.csv").is_file()


def test_figures_false_writes_no_figures_but_tables_still_written(tmp_path: Path) -> None:
    case_ids = [f"CASE_{i}" for i in range(4)]
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir, figures=False)
    _set_localize_dirs(cfg, [localize_dir])

    run_population_stats(cfg)

    assert not (output_dir / "figures").exists()
    for name in (
        "population_structures.csv",
        "population_lobes.csv",
        "population_laterality.csv",
        "population_eloquence.json",
    ):
        assert (output_dir / name).is_file()


# ---------------------------------------------------------------------------
# 9. population_config.yaml records provenance
# ---------------------------------------------------------------------------


def test_population_config_records_provenance(tmp_path: Path) -> None:
    case_ids = [f"CASE_{i}" for i in range(3)]
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids, split="test")
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir)
    _set_localize_dirs(cfg, [localize_dir])

    run_population_stats(cfg)

    record = read_yaml(output_dir / "population_config.yaml")
    assert record["resolved_localize_dirs"] == [str(localize_dir.resolve())]
    assert record["source"] == "prediction"
    assert record["splits"] == ["test"]
    assert record["n_cases"] == 3
    assert record["atlas"]["name"] == "SRI24/TZO"
    assert record["coverage_line"] == "23 of 122 structures classified eloquent, 0 unclassified"


# ---------------------------------------------------------------------------
# 10. Degenerate-field warning
# ---------------------------------------------------------------------------


def test_degenerate_near_eloquent_field_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Measured on the real 189-case test split, `near_eloquent` is True for 100% of cases.

    That is what actually runs in production, so this path must not be
    silent -- a reader who only sees `frac_near_eloquent: 1.0` in the JSON
    would read it as agreement rather than as a field with zero per-case
    discriminating power.
    """
    case_ids = [f"CASE_{i}" for i in range(4)]
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids, all_near_eloquent=True)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, output_dir)
    _set_localize_dirs(cfg, [localize_dir])

    with caplog.at_level(logging.WARNING, logger="population_stats_script"):
        run_population_stats(cfg)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a WARNING-level log record about the degenerate field"
    assert any("frac_near_eloquent" in r.getMessage() for r in warnings)

    import json

    eloquence = json.loads((output_dir / "population_eloquence.json").read_text())
    assert eloquence["frac_near_eloquent"] == pytest.approx(1.0)
    assert "frac_near_eloquent" in eloquence["degenerate_fields"]


# ---------------------------------------------------------------------------
# 11. min_frac_of_structure is honoured
#
# `structure_involvement_frequency` (src/neurovision/analysis/population.py,
# out of scope for this file to change) keeps one row per structure that is
# PRESENT in the input regardless of the threshold -- it only zeroes out
# `n_cases_involved` / `frac_cases_involved` for a structure that never
# reaches it. So "drops a structure" is verified here as "drops it out of
# being counted as involved in any case", which is the observable effect on
# population_structures.csv: at the default threshold Parietal_R
# (frac_of_structure=0.02) never counts as involved, while a low threshold
# picks it up in both cases it appears in.
# ---------------------------------------------------------------------------


def test_min_frac_of_structure_is_honoured(tmp_path: Path) -> None:
    case_ids = [f"CASE_{i}" for i in range(5)]
    parietal_cases = ("CASE_0", "CASE_1")
    localize_dir = _build_localize_dir(tmp_path, "run_a", case_ids, parietal_cases=parietal_cases)

    out_low = tmp_path / "out_low"
    cfg_low = _compose_cfg(tmp_path, out_low, min_frac_of_structure=0.01)
    _set_localize_dirs(cfg_low, [localize_dir])
    run_population_stats(cfg_low)
    low = pd.read_csv(out_low / "population_structures.csv")
    parietal_low = low[low["structure"] == "Parietal_R"].iloc[0]
    assert int(parietal_low["n_cases_involved"]) == 2
    assert parietal_low["frac_cases_involved"] == pytest.approx(2 / 5)

    out_high = tmp_path / "out_high"
    cfg_high = _compose_cfg(tmp_path, out_high, min_frac_of_structure=0.05)
    _set_localize_dirs(cfg_high, [localize_dir])
    run_population_stats(cfg_high)
    high = pd.read_csv(out_high / "population_structures.csv")
    parietal_high = high[high["structure"] == "Parietal_R"].iloc[0]
    assert int(parietal_high["n_cases_involved"]) == 0
    assert parietal_high["frac_cases_involved"] == pytest.approx(0.0)

    # Frontal_L (0.5) clears both thresholds, so it stays fully involved --
    # this is what shows the threshold change is real and not an artifact of
    # e.g. the whole table going empty.
    frontal_high = high[high["structure"] == "Frontal_L"].iloc[0]
    assert int(frontal_high["n_cases_involved"]) == 5
