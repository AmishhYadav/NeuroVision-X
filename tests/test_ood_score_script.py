"""Tests for scripts/ood_score.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, the same pattern
`tests/test_local_recalibration_script.py` / `tests/test_error_budget_script.py`
already use for their own sibling scripts.

Everything here is synthetic and tiny -- (4, D, H, W) volumes with D, H, W in the
6-10 range, never real BraTS data. `neurovision.analysis.ood.extract_features`'s own
`_MIN_NONZERO_VOXELS` floor is 100, so every non-broken synthetic case gets a brain
mask of 100-170 voxels; a "broken" case gets exactly 10, deliberately below the floor,
to exercise the skip path. Whole file runs in well under 3 seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from neurovision.inference.gatekeeper import Thresholds
from neurovision.utils.io import write_yaml
from tests.script_loader import load_script

ood_score_script = load_script("ood_score")
error_budget_script = load_script("error_budget")

run = ood_score_script.run
list_case_ids = ood_score_script.list_case_ids
auroc_table = ood_score_script.auroc_table

_load_ood_table = error_budget_script._load_ood_table
_load_ood_thresholds = error_budget_script._load_ood_thresholds

MODALITIES = ["t1", "t1ce", "t2", "flair"]


# ---------------------------------------------------------------------------
# Synthetic preprocessed-volume builders
# ---------------------------------------------------------------------------


def _make_case(
    case_dir: Path,
    index: int,
    rng: np.random.Generator,
    shift: float = 0.0,
    broken: bool = False,
) -> None:
    """Writes one synthetic `<case_dir>/image.npy`, (4, D, H, W) float16.

    `D, H, W` vary with `index` (6-10 range) so extent_d/h/w -- computed as the
    array's own shape, not a bounding box -- have real variance across a cohort;
    a constant shape across every case would give `fit_ood_model` a zero-SD
    column and it would (correctly) refuse to fit.

    The SAME nonzero mask is reused for all four channels (mirroring
    `tests/test_ood.py`'s own `_synthetic_image`, where every modality shares
    one sphere's geometry), so the joint brain mask used for correlation
    features is never smaller than the per-modality mask.

    Args:
        case_dir: Directory to write `image.npy` into (created if needed).
        index: Used only to vary this case's shape deterministically.
        rng: Shared, already-seeded generator (mask location and values).
        shift: Added to every "brain" voxel's value -- used to make one
            cohort's features land far from another's.
        broken: If True, only 10 nonzero voxels (below the 100-voxel floor)
            are written, so `extract_features` raises `ValueError`.
    """
    d, h, w = 6 + index % 5, 7 + index % 4, 8 + index % 3
    flat_size = d * h * w
    if broken:
        n_brain = 10
    else:
        n_brain = max(100, min(flat_size - 4, 120 + (index * 13) % 50))
    chosen = rng.choice(flat_size, size=n_brain, replace=False)
    mask_flat = np.zeros(flat_size, dtype=bool)
    mask_flat[chosen] = True
    mask = mask_flat.reshape(d, h, w)

    image = np.zeros((4, d, h, w), dtype=np.float32)
    for c in range(4):
        image[c][mask] = rng.standard_normal(n_brain).astype(np.float32) + shift

    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "image.npy", image.astype(np.float16))


def _make_cohort(
    prep_dir: Path,
    case_ids: list[str],
    rng: np.random.Generator,
    shift: float = 0.0,
    broken_ids: set[str] | None = None,
) -> None:
    """Writes one `<prep_dir>/<case_id>/image.npy` per id, plus the decoy top-level
    files a real preprocessed tree also carries (`case_index.csv`, `metadata.csv`)."""
    prep_dir.mkdir(parents=True, exist_ok=True)
    (prep_dir / "case_index.csv").write_text("case_id\n")
    (prep_dir / "metadata.csv").write_text("case_id\n")
    broken_ids = broken_ids or set()
    for i, case_id in enumerate(case_ids):
        _make_case(prep_dir / case_id, i, rng, shift=shift, broken=case_id in broken_ids)


def _write_metrics(eval_dir: Path, case_ids: list[str], dice: list[float]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": case_ids, "dice_WT": dice, "dice_TC": dice}).to_csv(
        eval_dir / "per_case_metrics.csv", index=False
    )


def _build_cfg(
    root: Path,
    seed: int,
    test_shift: float = 0.0,
    ssa_broken: bool = False,
) -> DictConfig:
    """A full, minimal `cfg` for `run()`: brats (train/val/test) + ssa + ped.

    train has 12 cases (>= `fit_ood_model`'s 10-row floor), val 8, test 6, ssa 5
    (optionally with one broken case), ped 5. `test_shift` is added to every
    `test`-cohort voxel value -- used to check that shifting a `score_cohorts`
    entry never moves `model.json`/`thresholds.json` (both are train/val only).
    """
    rng = np.random.default_rng(seed)

    brats_dir = root / "brats"
    ssa_dir = root / "ssa"
    ped_dir = root / "ped"
    splits_path = root / "splits.yaml"
    out_dir = root / "out"

    train_ids = [f"train_{i:03d}" for i in range(12)]
    val_ids = [f"val_{i:03d}" for i in range(8)]
    test_ids = [f"test_{i:03d}" for i in range(6)]
    ssa_ids = [f"ssa_{i:03d}" for i in range(5)]
    ped_ids = [f"ped_{i:03d}" for i in range(5)]

    _make_cohort(brats_dir, train_ids, rng, shift=0.0)
    _make_cohort(brats_dir, val_ids, rng, shift=0.0)
    _make_cohort(brats_dir, test_ids, rng, shift=test_shift)

    broken = {"ssa_broken"} if ssa_broken else None
    ssa_all_ids = ssa_ids + (["ssa_broken"] if ssa_broken else [])
    _make_cohort(ssa_dir, ssa_all_ids, rng, shift=0.0, broken_ids=broken)
    _make_cohort(ped_dir, ped_ids, rng, shift=0.0)

    write_yaml({"train": train_ids, "val": val_ids, "test": test_ids}, splits_path)

    eval_test_dir = root / "eval_test"
    eval_ssa_dir = root / "eval_ssa"
    eval_ped_dir = root / "eval_ped"
    _write_metrics(eval_test_dir, test_ids, [0.9, 0.9, 0.9, 0.3, 0.3, 0.9])
    _write_metrics(eval_ssa_dir, ssa_ids, [0.9, 0.4, 0.9, 0.4, 0.9])
    _write_metrics(eval_ped_dir, ped_ids, [0.2, 0.2, 0.9, 0.9, 0.9])

    return OmegaConf.create(
        {
            "seed": seed,
            "data": {"modalities": MODALITIES, "splits": {"path": str(splits_path)}},
            "analysis": {
                "ood": {
                    "out_dir": str(out_dir),
                    "fit_split": "train",
                    "threshold_split": "val",
                    "shrinkage": 0.1,
                    "caution_quantile": 0.10,
                    "refuse_quantile": 0.02,
                    "score_cohorts": [
                        {
                            "name": "test",
                            "prep_dir": str(brats_dir),
                            "split": "test",
                            "eval_dir": str(eval_test_dir),
                        },
                        {
                            "name": "ssa",
                            "prep_dir": str(ssa_dir),
                            "split": None,
                            "eval_dir": str(eval_ssa_dir),
                        },
                        {
                            "name": "ped",
                            "prep_dir": str(ped_dir),
                            "split": None,
                            "eval_dir": str(eval_ped_dir),
                        },
                    ],
                    "brats_prep_dir": str(brats_dir),
                    "usable_bar": 0.7,
                    "usable_regions": ["WT", "TC"],
                    "n_boot": 30,
                    "ci": 0.95,
                    "seed": seed,
                }
            },
        }
    )


# ---------------------------------------------------------------------------
# 1. list_case_ids -- skips decoy files and non-image directories
# ---------------------------------------------------------------------------


def test_list_case_ids_skips_files_and_non_image_dirs(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    (prep_dir / "case_index.csv").write_text("case_id\n")
    (prep_dir / "metadata.csv").write_text("case_id\n")
    (prep_dir / "no_image_here").mkdir()  # a directory, but no image.npy inside
    for case_id in ["b_case", "a_case"]:
        case_dir = prep_dir / case_id
        case_dir.mkdir()
        np.save(case_dir / "image.npy", np.zeros((1,), dtype=np.float16))

    ids = list_case_ids(prep_dir, None)
    assert ids == ["a_case", "b_case"]


def test_list_case_ids_uses_split_list_verbatim(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    # prep_dir holds nothing that matches the split list -- list_case_ids must not
    # look at disk at all when a split list is given.
    ids = list_case_ids(prep_dir, ["z_case", "a_case", "a_case"])
    assert ids == ["a_case", "a_case", "z_case"]


# ---------------------------------------------------------------------------
# 2. Ordering: thresholds.json comes from val ONLY -- shifting the `test` score
#    cohort far away must not move it at all.
# ---------------------------------------------------------------------------


def test_thresholds_come_from_val_only(tmp_path: Path) -> None:
    cfg_normal = _build_cfg(tmp_path / "a", seed=1, test_shift=0.0)
    cfg_shifted = _build_cfg(tmp_path / "b", seed=1, test_shift=5000.0)

    run(cfg_normal)
    run(cfg_shifted)

    thr_normal = json.loads((Path(cfg_normal.analysis.ood.out_dir) / "thresholds.json").read_text())
    thr_shifted = json.loads(
        (Path(cfg_shifted.analysis.ood.out_dir) / "thresholds.json").read_text()
    )

    assert thr_normal["caution_cut"] == pytest.approx(thr_shifted["caution_cut"])
    assert thr_normal["refuse_cut"] == pytest.approx(thr_shifted["refuse_cut"])
    assert thr_normal["n_val"] == thr_shifted["n_val"] == 8

    model_normal = json.loads((Path(cfg_normal.analysis.ood.out_dir) / "model.json").read_text())
    model_shifted = json.loads((Path(cfg_shifted.analysis.ood.out_dir) / "model.json").read_text())
    assert model_normal["center"] == pytest.approx(model_shifted["center"])


# ---------------------------------------------------------------------------
# 3. Output files are exactly what scripts/error_budget.py's own readers expect.
# ---------------------------------------------------------------------------


def test_outputs_match_error_budget_readers(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, seed=2)
    paths = run(cfg)
    out_dir = Path(cfg.analysis.ood.out_dir)

    for name in ["train", "val", "test", "ssa", "ped"]:
        assert Path(paths[f"scores_{name}"]).is_file()

    table = _load_ood_table(out_dir, "test")
    assert list(table.columns) == ["case_id", "ood_score"]
    assert len(table) == 6  # all 6 test cases scored, none broken

    deployed = Thresholds(
        predicted_dice={},
        conformal_band={},
        ood_score=(0.0, 0.0),
        calibration_n=0,
        caution_quantile=0.1,
        refuse_quantile=0.02,
    )
    result = _load_ood_thresholds(out_dir, deployed)
    assert result.ood_score[0] <= result.ood_score[1]
    # Only ood_score changed -- everything else about `deployed` is untouched.
    assert result.calibration_n == deployed.calibration_n


# ---------------------------------------------------------------------------
# 4. A too-small-brain case is skipped, recorded, and never scored.
# ---------------------------------------------------------------------------


def test_skipped_case_not_in_score_file(tmp_path: Path) -> None:
    cfg = _build_cfg(tmp_path, seed=3, ssa_broken=True)
    run(cfg)
    out_dir = Path(cfg.analysis.ood.out_dir)

    skipped = pd.read_csv(out_dir / "skipped.csv")
    broken_rows = skipped[skipped["case_id"] == "ssa_broken"]
    assert len(broken_rows) == 1
    assert broken_rows.iloc[0]["cohort"] == "ssa"

    ssa_scores = pd.read_csv(out_dir / "ssa_ood_score.csv")
    assert "ssa_broken" not in set(ssa_scores["case_id"])
    assert len(ssa_scores) == 5  # the 5 good ssa cases, not the broken 6th

    run_meta = json.loads((out_dir / "run_meta.json").read_text())
    assert run_meta["n_skipped_total"] == 1
    assert run_meta["n_skipped_by_cohort"]["ssa"] == 1


# ---------------------------------------------------------------------------
# 5. AUROC: perfect separation -> 1.0; no unusable cases -> NaN.
# ---------------------------------------------------------------------------


def test_auroc_perfect_separation(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    case_ids = [f"c{i}" for i in range(10)]
    # First 5 usable (dice 0.9), last 5 unusable (dice 0.3).
    dice = [0.9] * 5 + [0.3] * 5
    _write_metrics(eval_dir, case_ids, dice)

    # ood_score perfectly separates: the 5 unusable cases score strictly higher.
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4]
    score_df = pd.DataFrame(
        {"case_id": case_ids, "ood_score": scores, "ood_flag": ["proceed"] * 10}
    )

    generator = np.random.default_rng(0)
    result = auroc_table(
        {"cohort": score_df},
        {"cohort": eval_dir},
        usable_regions=["WT", "TC"],
        usable_bar=0.7,
        n_boot=25,
        ci=0.95,
        generator=generator,
    )
    row = result.iloc[0]
    assert row["auroc"] == pytest.approx(1.0)
    assert int(row["n_unusable"]) == 5
    assert int(row["n"]) == 10


def test_auroc_no_unusable_cases_is_nan(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval2"
    case_ids = [f"c{i}" for i in range(6)]
    _write_metrics(eval_dir, case_ids, [0.9] * 6)  # every case usable
    score_df = pd.DataFrame(
        {"case_id": case_ids, "ood_score": np.arange(6.0), "ood_flag": ["proceed"] * 6}
    )

    generator = np.random.default_rng(1)
    result = auroc_table(
        {"cohort": score_df},
        {"cohort": eval_dir},
        usable_regions=["WT", "TC"],
        usable_bar=0.7,
        n_boot=25,
        ci=0.95,
        generator=generator,
    )
    row = result.iloc[0]
    assert np.isnan(row["auroc"])
    assert np.isnan(row["ci_lo"])
    assert int(row["n_unusable"]) == 0
    assert int(row["n_boot_valid"]) == 0


# ---------------------------------------------------------------------------
# 6. Determinism: same seed, same flag_rates.csv.
# ---------------------------------------------------------------------------


def test_deterministic_rerun(tmp_path: Path) -> None:
    cfg1 = _build_cfg(tmp_path / "r1", seed=4)
    cfg2 = _build_cfg(tmp_path / "r2", seed=4)

    run(cfg1)
    run(cfg2)

    flags1 = pd.read_csv(Path(cfg1.analysis.ood.out_dir) / "flag_rates.csv")
    flags2 = pd.read_csv(Path(cfg2.analysis.ood.out_dir) / "flag_rates.csv")
    pd.testing.assert_frame_equal(flags1, flags2)
