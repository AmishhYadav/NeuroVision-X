"""Tests for scripts/recalibration_cost.py on tiny synthetic curves (no real data, CPU, fast)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.script_loader import load_script

script = load_script("recalibration_cost")

THRESHOLDS = np.array([0.1, 0.5, 0.9])
N_CASES = 30  # half = 15; with min_holdout 10 the usable k are <= 20


def _write_curves(path: Path, miss: float) -> None:
    """Writes a curves.npz (WT only) where every case misses `miss` of 100 voxels everywhere.

    Constant miss rate across thresholds: risk is `miss` for every tau, so CRC is feasible at
    alpha=0.2 exactly when `miss` is small.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = np.full((N_CASES, 3), int(round(miss * 100)), dtype=np.int64)
    np.savez(
        path,
        thresholds=THRESHOLDS,
        WT__case_ids=np.array([f"c{i:03d}" for i in range(N_CASES)]),
        WT__gt_voxels=np.full(N_CASES, 100, dtype=np.int64),
        WT__fn_voxels=fn,
        WT__mask_voxels=np.full((N_CASES, 3), 100, dtype=np.int64),
    )


def _make_block(tmp_path: Path, test_miss: float = 0.0, ped_miss: float = 0.9) -> dict:
    """Config block (plain dict) + files: test and ssa restorable, ped never restorable."""
    for cohort, miss in (("test", test_miss), ("ssa", 0.0), ("ped", ped_miss)):
        _write_curves(tmp_path / "in" / f"{cohort}.npz", miss)
    frozen = pd.DataFrame(
        [
            # (model, cohort, region, alpha, guarantee, ...) -- ssa/crc present and feasible,
            # ssa/rcps infeasible (NaN metrics), everything else absent.
            dict(model="m1", cohort="ssa", region="WT", alpha=0.2, guarantee="crc",
                 verdict="VIOLATED", mean_miss_rate=0.3, precision_median=0.5,
                 dice_median=0.6, volume_ratio_median=1.5),
            dict(model="m1", cohort="ssa", region="WT", alpha=0.2, guarantee="rcps",
                 verdict="infeasible", mean_miss_rate=np.nan, precision_median=np.nan,
                 dice_median=np.nan, volume_ratio_median=np.nan),
        ]
    )  # fmt: skip
    (tmp_path / "in").mkdir(exist_ok=True)
    frozen.to_csv(tmp_path / "in" / "frozen.csv", index=False)
    return {
        "out_dir": "out",
        "regions": ["WT"],
        "alphas": [0.2],
        "methods": ["crc", "rcps"],
        "delta": 0.1,
        "rcps_order_seed": 0,
        "empty_by_region": {"WT": "exclude"},
        "k_grid": [5, "half"],
        "min_holdout": 10,
        "n_splits": 6,
        "seed": 1,
        "min_feasible_rate": 0.9,
        "minutes_per_case": 60.0,
        "preannotation_speedup": 2.21,
        "models": {
            "m1": {
                "grid": "extended",
                "curves": {c: f"in/{c}.npz" for c in ("test", "ssa", "ped")},
            }
        },
        "frozen_realised": ["in/frozen.csv"],
    }


def test_outputs_and_annotation_hours(tmp_path: Path) -> None:
    paths = script.run_recalibration_cost(_make_block(tmp_path), tmp_path)
    for p in paths.values():
        assert p.is_file()
    assert (tmp_path / "out" / "cost_table.md").read_text().startswith("**COUNTERFACTUAL")

    k = pd.read_csv(paths["k_needed_csv"])
    row = k[(k.cohort == "ssa") & (k.method == "crc")].iloc[0]
    # Zero miss, 5 calibration cases: CRC adjusted risk 1/6 <= 0.2, so k=5 already restores.
    assert row["k_needed"] == 5
    assert row["annotation_hours_manual"] == pytest.approx(5.0)  # 5 * 60 / 60
    assert row["annotation_hours_preannotated"] == pytest.approx(5.0 / 2.21)
    ped = k[(k.cohort == "ped") & (k.method == "crc")].iloc[0]
    assert not ped["restorable"] and np.isnan(ped["annotation_hours_manual"])


def test_frozen_join_and_nan_where_absent(tmp_path: Path) -> None:
    paths = script.run_recalibration_cost(_make_block(tmp_path), tmp_path)
    k = pd.read_csv(paths["k_needed_csv"])

    def cell(cohort: str, method: str) -> pd.Series:
        return k[(k.cohort == cohort) & (k.method == method)].iloc[0]

    crc = cell("ssa", "crc")
    assert crc["frozen_verdict"] == "VIOLATED"
    assert crc["frozen_dice_median"] == pytest.approx(0.6)
    assert crc["frozen_volume_ratio_median"] == pytest.approx(1.5)
    assert cell("ssa", "rcps")["frozen_verdict"] == "infeasible"
    assert np.isnan(cell("ssa", "rcps")["frozen_dice_median"])
    assert pd.isna(cell("test", "crc")["frozen_verdict"])  # no frozen row at all
    assert len(k) == len(k.drop_duplicates(["model", "cohort", "region", "alpha", "method"]))


def test_overwrite_refusal(tmp_path: Path) -> None:
    block = _make_block(tmp_path)
    script.run_recalibration_cost(block, tmp_path)
    with pytest.raises(FileExistsError):
        script.run_recalibration_cost(block, tmp_path)
    script.run_recalibration_cost({**block, "overwrite": True}, tmp_path)  # allowed


def test_run_meta_has_hashes(tmp_path: Path) -> None:
    paths = script.run_recalibration_cost(_make_block(tmp_path), tmp_path)
    meta = json.loads(paths["run_meta_json"].read_text())
    assert meta["n_results"] > 0 and "git_sha" in meta and meta["timestamp_utc"]
    assert "COUNTERFACTUAL" in meta["status_label"]
    expected = hashlib.sha256((tmp_path / "in" / "ssa.npz").read_bytes()).hexdigest()
    assert meta["input_sha256"]["in/ssa.npz"] == expected
    assert "in/frozen.csv" in meta["input_sha256"]


def test_control_check_warns_not_aborts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Test cohort that misses 90% everywhere: CRC infeasible, so the control is not RESTORED.
    block = _make_block(tmp_path, test_miss=0.9)
    with caplog.at_level(logging.WARNING):
        paths = script.run_recalibration_cost(block, tmp_path)
    assert paths["k_needed_csv"].is_file()
    assert any("CONTROL CHECK" in r.message for r in caplog.records)
    meta = json.loads(paths["run_meta_json"].read_text())
    assert meta["control_failures"]


def test_missing_curves_file_named(tmp_path: Path) -> None:
    block = _make_block(tmp_path)
    (tmp_path / "in" / "ped.npz").unlink()
    with pytest.raises(FileNotFoundError, match="ped.npz"):
        script.run_recalibration_cost(block, tmp_path)


def test_hydra_config_resolves_and_n_splits_override() -> None:
    """The real config composes, and the documented n_splits override applies."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base="1.3", config_dir=script._CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["analysis.recalibration_cost.n_splits=3"])
    block = OmegaConf.to_container(cfg.analysis.recalibration_cost, resolve=True)
    assert block["n_splits"] == 3 and block["delta"] == cfg.analysis.conformal_v2.delta
