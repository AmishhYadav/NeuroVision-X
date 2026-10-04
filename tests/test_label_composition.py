"""Tests for scripts/label_composition.py. Synthetic tiny files only, CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.script_loader import load_script

script = load_script("label_composition")


def _write_meta(root: Path, case: str, counts: dict | None) -> None:
    (root / case).mkdir(parents=True)
    meta = {"case_id": case}
    if counts is not None:
        meta["label_voxel_counts"] = {str(k): v for k, v in counts.items()}
    (root / case / "meta.json").write_text(json.dumps(meta))


def _write_logits(eval_dir: Path, case: str, arr: np.ndarray) -> None:
    (eval_dir / "logits").mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "logits" / f"{case}.npy", arr.astype(np.float16))


def _logits(on: dict[int, int]) -> np.ndarray:
    """(3, 4, 4, 4) logits: channel c has its first on[c] voxels positive, rest negative."""
    arr = -np.ones((3, 64), dtype=np.float16)
    for ch, n in on.items():
        arr[ch, :n] = 1
    return arr.reshape(3, 4, 4, 4)


def test_reference_hand_values(tmp_path: Path) -> None:
    _write_meta(tmp_path, "a", {1: 8, 2: 0, 3: 2})
    _write_meta(tmp_path, "b", {1: 1, 2: 6, 3: 3})
    table, skipped = script.reference_counts(tmp_path, "x")
    assert skipped == 0
    a = table.set_index("case_id").loc["a"]
    b = table.set_index("case_id").loc["b"]
    assert a["tc_wt"] == pytest.approx(1.0)
    assert a["ed_share"] == pytest.approx(0.0)
    assert b["tc_wt"] == pytest.approx(0.4)
    assert b["ed_share"] == pytest.approx(0.6)


def test_reference_missing_key_and_skips(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "meta.json").write_text(json.dumps({"label_voxel_counts": {"1": 5, "3": 5}}))
    _write_meta(tmp_path, "zero", {1: 0, 2: 0, 3: 0})
    _write_meta(tmp_path, "nolabel", None)
    table, skipped = script.reference_counts(tmp_path, "x")
    assert skipped == 2
    assert list(table["case_id"]) == ["m"]
    assert table["ed_share"].iloc[0] == 0.0  # missing key '2' counts as 0


def test_reference_missing_root_names_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope"):
        script.reference_counts(tmp_path / "nope", "x")


def test_prediction_half_and_skip(tmp_path: Path) -> None:
    _write_logits(tmp_path, "half", _logits({1: 10, 2: 20}))  # TC 10, WT 20
    _write_logits(tmp_path, "empty", _logits({1: 5}))  # WT empty
    table, skipped = script.prediction_ratios([tmp_path], "s", ["ET", "TC", "WT"])
    assert skipped == 1
    assert table["tc_wt"].iloc[0] == pytest.approx(0.5)


def test_prediction_channel_order_from_config(tmp_path: Path) -> None:
    # Channel 0 = WT (20 voxels), channel 1 = TC (10 voxels) under [WT, TC, ET].
    _write_logits(tmp_path, "c", _logits({0: 20, 1: 10}))
    table, _ = script.prediction_ratios([tmp_path], "s", ["WT", "TC", "ET"])
    assert table["tc_wt"].iloc[0] == pytest.approx(0.5)
    # Same file read as [ET, TC, WT]: WT (ch 2) is empty -> skipped.
    table2, skipped2 = script.prediction_ratios([tmp_path], "s", ["ET", "TC", "WT"])
    assert skipped2 == 1 and table2.empty


def test_prediction_duplicate_case_raises(tmp_path: Path) -> None:
    for d in ("d0", "d1"):
        _write_logits(tmp_path / d, "same", _logits({1: 1, 2: 2}))
    with pytest.raises(ValueError, match="same"):
        script.prediction_ratios([tmp_path / "d0", tmp_path / "d1"], "s", ["ET", "TC", "WT"])


def test_prediction_missing_dir_names_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing_eval"):
        script.prediction_ratios([tmp_path / "missing_eval"], "s", ["ET", "TC", "WT"])


def test_run_writes_outputs(tmp_path: Path) -> None:
    _write_meta(tmp_path / "ref", "a", {1: 8, 2: 0, 3: 2})
    _write_logits(tmp_path / "ev", "p", _logits({1: 10, 2: 20}))
    cfg = OmegaConf.create(
        {
            "data": {"regions": ["ET", "TC", "WT"]},
            "analysis": {
                "label_composition": {
                    "out_dir": str(tmp_path / "out"),
                    "low_ratio": 0.5,
                    "low_ed_share": 0.05,
                    "references": {"r": str(tmp_path / "ref")},
                    "predictions": {"p": [str(tmp_path / "ev")]},
                }
            },
        }
    )
    paths = script.run(cfg)
    assert all(p.is_file() for p in paths.values())
    import pandas as pd

    summary = pd.read_csv(paths["summary"])
    assert list(summary["kind"]) == ["reference", "prediction"]
    assert summary["median_tc_wt"].tolist() == pytest.approx([1.0, 0.5])
    assert summary["frac_ed_share_below_low_ed_share"].iloc[0] == 1.0
