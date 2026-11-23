"""Tests for scripts/volume_endpoint.py. Synthetic 16^3 volumes, CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from neurovision.metrics.segmentation import classes_to_regions, dice_score
from tests.script_loader import load_script

ve = load_script("volume_endpoint")

SPACING = [1.0, 2.0, 5.0]  # voxel = 10 mm^3 = 0.01 ml
SHAPE = (16, 16, 16)


def _labels() -> dict[str, np.ndarray]:
    """case a: tumour; case b: tumour with no ET; case c: tumour with a big ET."""
    a = np.zeros(SHAPE, dtype=np.int64)
    a[2:10, 2:10, 2:10] = 2
    a[4:8, 4:8, 4:8] = 1
    a[5:7, 5:7, 5:7] = 3
    b = np.zeros(SHAPE, dtype=np.int64)
    b[2:10, 2:10, 2:10] = 2
    c = a.copy()
    c[4:8, 4:8, 4:8] = 3
    return {"c_a": a, "c_b": b, "c_c": c}


def _logits_from(pred_label: np.ndarray) -> np.ndarray:
    regions = classes_to_regions(torch.as_tensor(pred_label))[0].numpy()
    return (regions * 20.0 - 10.0).astype(np.float16)


def _world(root: Path, wrong_dice: bool = False) -> tuple[Path, Path]:
    """Writes eval dir + data dir. Case c_c's prediction misses the ET (it equals case a's)."""
    eval_dir, data_dir = root / "eval", root / "data"
    labels = _labels()
    preds = {"c_a": labels["c_a"], "c_b": labels["c_b"], "c_c": labels["c_a"]}
    (eval_dir / "logits").mkdir(parents=True)
    rows = []
    for cid, lab in labels.items():
        (data_dir / cid).mkdir(parents=True)
        np.save(data_dir / cid / "label.npy", lab)
        (data_dir / cid / "meta.json").write_text(json.dumps({"spacing": SPACING}))
        logits = _logits_from(preds[cid])
        np.save(eval_dir / "logits" / f"{cid}.npy", logits)
        d = dice_score(
            torch.from_numpy((logits > 0).astype(np.float32))[None],
            classes_to_regions(torch.as_tensor(lab)),
        )[0].numpy()
        rows.append({"case_id": cid, "dice_ET": d[0], "dice_TC": d[1], "dice_WT": d[2]})
    df = pd.DataFrame(rows)
    if wrong_dice:
        df.loc[0, "dice_WT"] -= 0.2
    df.to_csv(eval_dir / "per_case_metrics.csv", index=False)
    return eval_dir, data_dir


def _cfgs(tmp_path: Path, eval_dir: Path, data_dir: Path, **over: object):
    block = {
        "out_dir": "out",
        "eval_dirs": {"m_test": {"eval_dir": str(eval_dir), "data_dir": str(data_dir)}},
        "tol": 0.25,
        "regions": ["WT", "ET"],
        "empty_pred_max_ml": 0.0,
        "dice_bar": 0.7,
        "case_limit": None,
        "overwrite": False,
        "dice_check_tol": 1e-3,
        "fail_on_dice_mismatch": True,
        "n_boot": 50,
        "ci": 0.95,
        "boot_seed": 1,
    }
    block.update(over)
    full = {
        "inference": {
            "postprocess": {
                "threshold": 0.5,
                "enforce_nesting": True,
                "min_component_size": 0,
                "connectivity": 1,
                "keep_largest_only": False,
                "et_min_volume": 0,
            }
        }
    }
    return OmegaConf.create(block), OmegaConf.create(full)


def test_outputs_volumes_and_labels(tmp_path: Path) -> None:
    eval_dir, data_dir = _world(tmp_path)
    block, full = _cfgs(tmp_path, eval_dir, data_dir)
    # Relative paths are joined onto repo_root; absolute ones pass through unchanged.
    paths = ve.run_volume_endpoint(block, full, tmp_path)
    table = pd.read_csv(paths["m_test"], index_col="case_id")
    for col in (
        "vol_pred_WT_ml",
        "vol_ref_ET_ml",
        "vol_rel_err_ET",
        "ref_empty_ET",
        "usable_dice",
        "usable_volume",
        "usable_both",
        "usable_either",
    ):
        assert col in table.columns
    # case a: WT = 8^3 = 512 voxels * 0.01 ml; ET = 2^3 = 8 voxels.
    assert table.loc["c_a", "vol_ref_WT_ml"] == pytest.approx(5.12)
    assert table.loc["c_a", "vol_pred_ET_ml"] == pytest.approx(0.08)
    # case b has no ET, none predicted -> usable by volume; case c lost its ET (8 of 64 voxels).
    assert bool(table.loc["c_b", "ref_empty_ET"])
    assert table.loc["c_b", "usable_volume"]
    assert not table.loc["c_c", "usable_volume"]
    assert table.loc["c_a", "usable_both"]
    # The Dice label ignores ET (TC region is unchanged), so it still calls case c usable:
    # the two definitions disagree there.
    assert table.loc["c_c", "usable_dice"]
    assert table["dice_check_max_abs_diff"].max() < 1e-3

    summary = pd.read_csv(tmp_path / "out" / "m_test" / "summary.csv").iloc[0]
    assert summary["n"] == 3
    assert summary["n_ref_empty_ET"] == 1
    assert summary["p_usable_volume"] == pytest.approx(2 / 3)
    assert (
        summary["p_usable_volume_lo"] <= summary["p_usable_volume"] <= summary["p_usable_volume_hi"]
    )
    assert 0.0 <= summary["agreement_rate"] <= 1.0
    assert (tmp_path / "out" / "summary.csv").is_file()
    assert not list((tmp_path / "out").rglob("*.tmp"))


def test_resume_skips_then_overwrite_rewrites(tmp_path: Path) -> None:
    eval_dir, data_dir = _world(tmp_path)
    block, full = _cfgs(tmp_path, eval_dir, data_dir)
    path = ve.run_volume_endpoint(block, full, tmp_path)["m_test"]
    path.write_text("sentinel")
    ve.run_volume_endpoint(block, full, tmp_path)
    assert path.read_text() == "sentinel"  # skipped, not recomputed
    block, full = _cfgs(tmp_path, eval_dir, data_dir, overwrite=True)
    ve.run_volume_endpoint(block, full, tmp_path)
    assert path.read_text() != "sentinel"


def test_case_limit(tmp_path: Path) -> None:
    eval_dir, data_dir = _world(tmp_path)
    block, full = _cfgs(tmp_path, eval_dir, data_dir, case_limit=2)
    table = pd.read_csv(ve.run_volume_endpoint(block, full, tmp_path)["m_test"])
    assert len(table) == 2


def test_wrong_dice_raises_or_warns(tmp_path: Path) -> None:
    eval_dir, data_dir = _world(tmp_path, wrong_dice=True)
    block, full = _cfgs(tmp_path, eval_dir, data_dir)
    with pytest.raises(RuntimeError, match="not the deployed mask"):
        ve.run_volume_endpoint(block, full, tmp_path)
    assert not (tmp_path / "out" / "m_test" / "per_case_volumes.csv").exists()
    block, full = _cfgs(tmp_path, eval_dir, data_dir, fail_on_dice_mismatch=False)
    table = pd.read_csv(ve.run_volume_endpoint(block, full, tmp_path)["m_test"])
    assert table["dice_check_max_abs_diff"].max() > 0.15
