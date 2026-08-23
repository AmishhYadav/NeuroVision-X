"""Tests for scripts/mc_comparison.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mc_comparison.py"
_spec = importlib.util.spec_from_file_location("mc_comparison_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
mc_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["mc_comparison_script"] = mc_script
_spec.loader.exec_module(mc_script)

load_mc_map = mc_script.load_mc_map
compare = mc_script.compare

_STATS = OmegaConf.create({"n_boot": 500, "ci": 0.95, "seed": 0})


def test_load_mc_map_returns_the_region_mean(tmp_path: Path) -> None:
    """The ANY row of the other two signals is a region mean; MC must match it."""
    array = np.stack(
        [np.full((2, 2, 2), 0.0), np.full((2, 2, 2), 1.0), np.full((2, 2, 2), 2.0)]
    ).astype(np.float16)
    np.save(tmp_path / "CASE.npy", array)
    out = load_mc_map(tmp_path, "CASE")
    assert out.shape == (2, 2, 2)
    assert np.allclose(out, 1.0)


def test_load_mc_map_returns_none_when_absent(tmp_path: Path) -> None:
    """A missing map must be reported as missing, never imputed."""
    assert load_mc_map(tmp_path, "NOPE") is None


def _table(disagreement: np.ndarray, mc: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "auroc_entropy": mc,
            "auroc_disagreement": disagreement,
            "auroc_mc": mc,
        }
    )


def test_compare_declares_equivalence_for_a_tiny_paired_difference() -> None:
    rng = np.random.default_rng(0)
    mc = rng.normal(0.70, 0.03, size=60)
    disagreement = mc + rng.normal(0.0, 0.004, size=60)
    rows = compare(_table(disagreement, mc), "ssa", 0.03, _STATS)
    row = next(r for r in rows if r["arm"] == "disagreement")
    assert row["equivalent_to_mc"]


def test_compare_refuses_equivalence_for_a_real_gap() -> None:
    rng = np.random.default_rng(1)
    mc = rng.normal(0.75, 0.03, size=60)
    disagreement = mc - 0.09
    rows = compare(_table(disagreement, mc), "ssa", 0.03, _STATS)
    row = next(r for r in rows if r["arm"] == "disagreement")
    assert not row["equivalent_to_mc"]
    assert row["difference"] < 0


def test_compare_refuses_equivalence_on_a_noisy_small_sample() -> None:
    """An underpowered null must NOT be reported as 'as good as MC-dropout'."""
    rng = np.random.default_rng(2)
    mc = rng.normal(0.7, 0.2, size=10)
    disagreement = rng.normal(0.7, 0.2, size=10)
    rows = compare(_table(disagreement, mc), "ssa", 0.03, _STATS)
    assert not next(r for r in rows if r["arm"] == "disagreement")["equivalent_to_mc"]


def test_compare_scores_the_entropy_arm_too() -> None:
    """Entropy is the free comparator; without it 'as good as MC' has no context."""
    rng = np.random.default_rng(3)
    mc = rng.normal(0.7, 0.03, size=40)
    rows = compare(_table(mc + 0.001, mc), "ped", 0.03, _STATS)
    assert {r["arm"] for r in rows} == {"disagreement", "entropy"}


def test_compare_drops_nan_pairs_pairwise() -> None:
    mc = np.array([0.7, 0.8, np.nan, 0.6])
    disagreement = np.array([0.7, np.nan, 0.5, 0.61])
    rows = compare(_table(disagreement, mc), "ssa", 0.03, _STATS)
    assert next(r for r in rows if r["arm"] == "disagreement")["n"] == 2
