"""Tests for the deployed-model example-segmentation figure (review row R13).

Synthetic arrays only, CPU, tmp_path. Logits are written in the exact on-disk format
`load_case_logits` reads: `<eval_dir>/logits/<case_id>.npy`, fp16, (3, D, H, W).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import plot_semester_figures as psf  # noqa: E402

SHAPE = (12, 12, 6)  # (L, P, S): last axis is axial


def _table(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["case_id", "dice_ET", "dice_TC", "dice_WT"])


def _pp_cfg(min_size: int = 0):
    return OmegaConf.create(
        {
            "inference": {
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": min_size,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                }
            }
        }
    )


def _label() -> np.ndarray:
    label = np.zeros(SHAPE, dtype=np.uint8)
    label[3:9, 3:9, 2:5] = 2  # edema block (WT)
    label[4:8, 4:8, 2:5] = 1  # core
    label[5:7, 5:7, 3:4] = 3  # enhancing
    return label


def _write_case(root: Path, eval_dir: Path, case_id: str, label: np.ndarray) -> pd.Series:
    """Writes label, image and PERFECT logits for a case; returns its matching table row."""
    prep = root / case_id
    prep.mkdir(parents=True)
    np.save(prep / "label.npy", label)
    np.save(prep / "image.npy", np.random.default_rng(0).normal(size=(4, *SHAPE)).astype("float16"))
    et, tc, wt = label == 3, (label == 1) | (label == 3), label >= 1
    logits = np.stack([et, tc, wt]).astype(np.float32) * 20 - 10  # +10 inside, -10 outside
    (eval_dir / "logits").mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "logits" / f"{case_id}.npy", logits.astype("float16"))
    return pd.Series({"case_id": case_id, "dice_ET": 1.0, "dice_TC": 1.0, "dice_WT": 1.0})


def test_selection_median_failure_and_ties() -> None:
    table = _table(
        [
            ("c", 0.9, 0.9, 0.9),  # mean 0.9
            ("a", 0.5, 0.5, 0.5),  # mean 0.5
            ("b", 0.7, 0.7, 0.7),  # mean 0.7 = median
            ("d", 0.1, 0.1, 0.1),
            ("e", 1.0, 0.1, 1.0),  # ties d on dice_TC, later id
        ]
    )
    sel = psf.select_example_cases(table)
    assert sel == {"typical": "b", "failure": "d"}
    # Ties: two cases equally close to the median -> lowest case_id.
    tie = _table(
        [("z", 0.25, 0.25, 0.25), ("y", 0.75, 0.75, 0.75)]
    )  # median 0.5, both 0.25 away, exact in binary
    assert psf.select_example_cases(tie)["typical"] == "y"


def test_regions_to_classes() -> None:
    regions = np.zeros((3, 1, 1, 4), dtype=np.float32)
    regions[2, 0, 0, :3] = 1  # WT on voxels 0-2
    regions[1, 0, 0, :2] = 1  # TC on 0-1
    regions[0, 0, 0, 0] = 1  # ET on 0
    assert psf.regions_to_classes(regions)[0, 0].tolist() == [3, 1, 2, 0]


def test_slice_choice_and_fallback() -> None:
    ref = np.zeros((3, 4, 4, 5), dtype=np.float32)
    pred = np.zeros_like(ref)
    ref[2, :, :, 1] = 1
    ref[2, :2, :2, 3] = 1
    pred[2, :, :, 4] = 1  # would win if the prediction were used
    assert psf.choose_axial_slice(ref, pred) == 1
    ref[:] = 0
    assert psf.choose_axial_slice(ref, pred) == 4


def test_dice_assertion_fires_on_mismatch(tmp_path: Path) -> None:
    label = _label()
    eval_dir = tmp_path / "eval"
    row = _write_case(tmp_path / "prep", eval_dir, "c1", label)
    import torch

    from neurovision.analysis.replay import load_case_logits
    from neurovision.inference.postprocess import postprocess_logits
    from neurovision.metrics.segmentation import classes_to_regions

    pred = postprocess_logits(torch.from_numpy(load_case_logits(eval_dir, "c1"))[None], _pp_cfg())
    ref = classes_to_regions(torch.from_numpy(label.astype(np.int64)))[0].numpy()
    got = psf.check_dice_matches_table(pred[0].numpy(), ref, row, "c1")
    assert got["WT"] == pytest.approx(1.0)
    bad = row.copy()
    bad["dice_TC"] = 0.5
    with pytest.raises(AssertionError, match="Dice sanity check failed for c1 TC"):
        psf.check_dice_matches_table(pred[0].numpy(), ref, bad, "c1")


def test_end_to_end_writes_png_and_sidecar(tmp_path: Path) -> None:
    cohorts, rows_by_cohort = [], []
    for name in ("A", "B", "C"):
        eval_dir, prep = tmp_path / f"eval_{name}", tmp_path / f"prep_{name}"
        rows = []
        for i, case_id in enumerate((f"{name}1", f"{name}2")):
            label = _label()
            if i == 1:
                label[5:7, 5:7, 3:4] = 1  # no ET in the reference
            rows.append(_write_case(prep, eval_dir, case_id, label))
        table = pd.DataFrame(rows)
        table["dice_ET"] = [1.0, 1.0]
        table["dice_TC"] = [1.0, 1.0]
        csv = tmp_path / f"{name}.csv"
        table.to_csv(csv, index=False)
        cohorts.append(
            {"name": name, "per_case": str(csv), "eval_dir": str(eval_dir), "prep_dir": str(prep)}
        )
        rows_by_cohort.append(table)
    cfg = _pp_cfg()
    cfg.analysis = {
        "semester_figures": {
            "out_dir": str(tmp_path / "out"),
            "examples": {
                "enabled": True,
                "cohorts": cohorts,
                "out_name": "fig_examples_deployed.png",
            },
        }
    }
    png, txt = psf.run_examples_figure(cfg)
    assert png.name == "fig_examples_deployed.png" and png.stat().st_size > 1000
    text = txt.read_text(encoding="utf-8")
    assert "note 58" in text and "Command:" in text
    assert text.count("Dice ET") == 6
    for cohort in cohorts:
        assert cohort["per_case"] in text
