"""Tests for scripts/compare_family.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script` -- the same pattern as
tests/test_score_confidence.py / tests/test_conformal_script.py.

No real BraTS data anywhere here: tiny synthetic per-case CSVs written to
`tmp_path`, CPU only, whole file runs in well under two seconds (n_boot=200).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.script_loader import load_script

compare_family_script = load_script("compare_family")
run_compare_family = compare_family_script.run_compare_family

N_CASES = 12
N_BOOT = 200  # small, for test speed -- not a claim about a real result
SEED = 0


def _write_per_case(path: Path, case_ids: list[str], rng: np.random.Generator) -> None:
    """Writes a tiny synthetic per-case_metrics.csv with dice_ET and dice_TC columns."""
    df = pd.DataFrame(
        {
            "case_id": case_ids,
            "dice_ET": rng.uniform(0.5, 0.9, size=len(case_ids)),
            "dice_TC": rng.uniform(0.5, 0.9, size=len(case_ids)),
        }
    )
    df.to_csv(path, index=False)


def _make_cfg(tmp_path: Path, comparisons: list[dict]) -> OmegaConf:
    """Builds a minimal composed-looking cfg with only what run_compare_family reads."""
    return OmegaConf.create(
        {
            "seed": SEED,
            "analysis": {
                "compare_family": {
                    "out_dir": str(tmp_path / "out"),
                    "name": "test_family",
                    "name_a": "model_a",
                    "name_b": "model_b",
                    "n_boot": N_BOOT,
                    "ci": 0.95,
                    "alpha": 0.05,
                    "practical_threshold": None,
                    "comparisons": comparisons,
                }
            },
        }
    )


def _cohort_paths(tmp_path: Path, label: str, rng: np.random.Generator) -> tuple[Path, Path]:
    """Writes one cohort's a/b per-case CSVs and returns their paths."""
    case_ids = [f"{label}_{i:03d}" for i in range(N_CASES)]
    a_path = tmp_path / f"{label}_a.csv"
    b_path = tmp_path / f"{label}_b.csv"
    _write_per_case(a_path, case_ids, rng)
    _write_per_case(b_path, case_ids, rng)
    return a_path, b_path


# ---------------------------------------------------------------------------
# 1. Family shape + monotone conservatism of the family-wide correction.
# ---------------------------------------------------------------------------


def test_family_table_shape_and_holm_is_more_conservative(tmp_path: Path) -> None:
    """One row per (item, metric); p_holm_family >= p_holm_within_item everywhere.

    A larger family (2 items x 2 metrics = 4 p-values corrected together)
    can only make Holm's step-down correction more conservative, or equal,
    relative to correcting each item's 2 metrics alone -- never less.
    """
    rng = np.random.default_rng(1)
    test_a, test_b = _cohort_paths(tmp_path, "test", rng)
    ext_a, ext_b = _cohort_paths(tmp_path, "ext", rng)

    comparisons = [
        {"cohort": "test", "a": str(test_a), "b": str(test_b), "metrics": ["dice_ET", "dice_TC"]},
        {"cohort": "ext", "a": str(ext_a), "b": str(ext_b), "metrics": ["dice_ET", "dice_TC"]},
    ]
    cfg = _make_cfg(tmp_path, comparisons)

    paths = run_compare_family(cfg)
    table = pd.read_csv(paths["family_csv"])

    assert len(table) == 4  # 2 items x 2 metrics
    assert set(table["cohort"]) == {"test", "ext"}
    assert set(table["metric"]) == {"dice_ET", "dice_TC"}

    assert (table["p_holm_family"] >= table["p_holm_within_item"] - 1e-12).all()

    # Config file is written and round-trips the comparisons block.
    assert Path(paths["compare_family_config_yaml"]).is_file()


# ---------------------------------------------------------------------------
# 2. Pooling via a list-valued a/b, and the duplicate-case_id guard.
# ---------------------------------------------------------------------------


def test_list_valued_side_pools_rows_and_duplicate_case_id_raises(tmp_path: Path) -> None:
    """A list of paths for `a` (or `b`) concatenates row-wise; n == sum of the parts."""
    rng = np.random.default_rng(2)
    ssa_a, ssa_b = _cohort_paths(tmp_path, "ssa", rng)
    ped_a, ped_b = _cohort_paths(tmp_path, "ped", rng)

    comparisons = [
        {
            "cohort": "pooled",
            "a": [str(ssa_a), str(ped_a)],
            "b": [str(ssa_b), str(ped_b)],
            "metrics": ["dice_ET"],
        }
    ]
    cfg = _make_cfg(tmp_path, comparisons)
    paths = run_compare_family(cfg)
    table = pd.read_csv(paths["family_csv"])

    assert len(table) == 1
    assert int(table.loc[0, "n"]) == 2 * N_CASES

    # Now force a duplicate case_id across the pooled list (ssa and ped share
    # a case_id) and confirm the script refuses rather than silently
    # double-counting it.
    dup_ped_a = tmp_path / "ped_a_dup.csv"
    df = pd.read_csv(ped_a)
    df.loc[0, "case_id"] = pd.read_csv(ssa_a).loc[0, "case_id"]  # collide with an ssa id
    df.to_csv(dup_ped_a, index=False)

    comparisons_dup = [
        {
            "cohort": "pooled",
            "a": [str(ssa_a), str(dup_ped_a)],
            "b": [str(ssa_b), str(ped_b)],
            "metrics": ["dice_ET"],
        }
    ]
    cfg_dup = _make_cfg(tmp_path, comparisons_dup)
    with pytest.raises(ValueError, match="duplicate"):
        run_compare_family(cfg_dup)


# ---------------------------------------------------------------------------
# 3. Missing files: skip-with-warning for one item, raise if all are missing.
# ---------------------------------------------------------------------------


def test_missing_file_skips_item_with_warning_and_all_missing_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rng = np.random.default_rng(3)
    test_a, test_b = _cohort_paths(tmp_path, "test", rng)
    missing_a = tmp_path / "does_not_exist_a.csv"

    comparisons = [
        {"cohort": "missing", "a": str(missing_a), "b": str(test_b), "metrics": ["dice_ET"]},
        {"cohort": "test", "a": str(test_a), "b": str(test_b), "metrics": ["dice_ET"]},
    ]
    cfg = _make_cfg(tmp_path, comparisons)

    with caplog.at_level(logging.WARNING):
        paths = run_compare_family(cfg)
    assert any("missing" in rec.message and str(missing_a) in rec.message for rec in caplog.records)
    table = pd.read_csv(paths["family_csv"])
    assert len(table) == 1
    assert table.loc[0, "cohort"] == "test"

    # Every item missing -> nothing to correct -> raises.
    comparisons_all_missing = [
        {"cohort": "missing", "a": str(missing_a), "b": str(test_b), "metrics": ["dice_ET"]},
    ]
    cfg_all_missing = _make_cfg(tmp_path, comparisons_all_missing)
    with pytest.raises(ValueError, match="skipped"):
        run_compare_family(cfg_all_missing)
