"""Tests for scripts/validate_qc.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` -- the same pattern
tests/test_train_qc.py and tests/test_conformal_script.py already use.

Everything here is synthetic, tiny (<=14^3 volumes), and CPU-only. All
"good" logits are strongly confident and nearly-correct, so every cell's
`n_positive` (cases with true Dice < bad_dice_threshold) is legitimately 0 --
that is fine for these tests, which check the PLUMBING (files written, the
falsification ordering guard, exclusion counting, determinism), not the Gate
C statistics themselves (already covered by tests/test_qc_validate.py).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from neurovision.analysis.qc_inference import load_case_arrays
from neurovision.data.qc_pairs import DegradationSpec, generate_pairs
from neurovision.models.qc import build_segqc
from neurovision.training.checkpoint import save_checkpoint

# Real configs/ directory, resolved relative to this file -- so the
# "reachable at the composed path" test composes the PROJECT's actual
# config, not a hand-built stand-in.
_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_qc.py"
_spec = importlib.util.spec_from_file_location("validate_qc_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
validate_qc_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["validate_qc_script"] = validate_qc_script
_spec.loader.exec_module(validate_qc_script)

run_validation = validate_qc_script.run_validation
resolve_checkpoint = validate_qc_script.resolve_checkpoint
resolve_cohorts = validate_qc_script.resolve_cohorts

SHAPE = (14, 14, 14)


# ---------------------------------------------------------------------------
# Synthetic data helpers -- same recipe as tests/test_train_qc.py
# ---------------------------------------------------------------------------


def _build_label(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    """A fixed nested ET-subset-of-TC-subset-of-WT sphere label."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.int64)
    label[dist < min_edge * 0.45] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.30] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.15] = 3  # ET, innermost
    return label


def _region_indicator(label: np.ndarray) -> np.ndarray:
    """(3, D, H, W) float32 array, channel order (ET, TC, WT)."""
    et = label == 3
    tc = et | (label == 1)
    wt = tc | (label == 2)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def _good_logits(label: np.ndarray, seed: int) -> np.ndarray:
    """Confident, mostly-correct logits: strongly positive inside each region."""
    rng = np.random.default_rng(seed)
    region = _region_indicator(label)
    logits = region * 12.0 - 6.0 + rng.normal(scale=0.3, size=region.shape)
    return logits.astype(np.float32)


def _write_case(
    prep_dir: Path,
    eval_dir: Path,
    case_id: str,
    label: np.ndarray,
    logits: np.ndarray,
    write_prep: bool = True,
) -> None:
    """Writes `eval_dir/logits/<case_id>.npy` and, unless `write_prep=False`,
    `prep_dir/<case_id>/{image.npy,label.npy}`."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))

    if not write_prep:
        return

    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))
    rng = np.random.default_rng(abs(hash(case_id)) % (2**32))
    image = rng.normal(size=(4, *label.shape)).astype(np.float32)
    np.save(case_dir / "image.npy", image.astype(np.float16))


def _write_cohort(
    tmp_path: Path,
    prep_dir: Path,
    tag: str,
    n_cases: int,
    *,
    seed_offset: int = 0,
) -> tuple[Path, list[str]]:
    """Writes a cohort's logits + preprocessed cases. Returns (eval_dir, case_ids)."""
    eval_dir = tmp_path / f"eval_{tag}"
    case_ids = []
    for i in range(n_cases):
        case_id = f"{tag}_{i:03d}"
        label = _build_label()
        logits = _good_logits(label, seed_offset + i)
        _write_case(prep_dir, eval_dir, case_id, label, logits)
        case_ids.append(case_id)
    return eval_dir, case_ids


def _true_identity_dice(
    cfg, eval_dir: Path, prep_dir: Path, case_id: str
) -> tuple[float, float, float]:
    """Independently reconstructs one case's true (ET, TC, WT) identity Dice --
    the SAME computation run_validation performs internally, used here to build a
    matching per_case_metrics.csv and, in test 2, to check the script's own output."""
    arrays = load_case_arrays(cfg, eval_dir, prep_dir, case_id)
    generator = np.random.default_rng(0)
    pairs = generate_pairs(
        arrays.pred_mask,
        arrays.label,
        specs=[DegradationSpec("identity", 0.0)],
        generator=generator,
        per_region=False,
    )
    return pairs[0].dice


def _write_per_case_metrics(
    cfg,
    eval_dir: Path,
    prep_dir: Path,
    case_ids: list[str],
    *,
    corrupt: tuple[str, str, float] | None = None,
) -> Path:
    """Writes `eval_dir/per_case_metrics.csv` with the TRUE identity Dice for every case
    (agreeing exactly with what run_validation reconstructs), optionally corrupting one
    (case_id, region) cell to a fixed ABSOLUTE value to trigger the falsification check.

    Args:
        corrupt: `(case_id, region, corrupted_value)` -- the published
            `dice_<region>` for that case is REPLACED (not shifted) by
            `corrupted_value`. Replacing rather than adding a delta avoids a
            near-perfect true Dice (this fixture's "good" logits score close
            to 1.0) silently clipping a large additive delta back down to a
            small, tolerance-passing difference.
    """
    rows = []
    for case_id in case_ids:
        et, tc, wt = _true_identity_dice(cfg, eval_dir, prep_dir, case_id)
        rows.append({"case_id": case_id, "dice_ET": et, "dice_TC": tc, "dice_WT": wt})

    if corrupt is not None:
        target_case, region, corrupted_value = corrupt
        row = next(r for r in rows if r["case_id"] == target_case)
        row[f"dice_{region}"] = float(corrupted_value)

    path = eval_dir / "per_case_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_checkpoint(tmp_path: Path, cfg) -> Path:
    """Builds a fresh SegQC and saves it as `best.pt`. Returns the checkpoint path."""
    ckpt_dir = tmp_path / "qc_ckpt"
    model = build_segqc(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    save_checkpoint(
        ckpt_dir,
        model,
        optimizer,
        epoch=3,
        global_step=10,
        best_metric=0.42,
        best_metric_name="spearman",
        best_metric_mode="max",
        is_best=True,
    )
    return ckpt_dir / "best.pt"


def _make_cfg(
    out_dir: Path,
    checkpoint: Path,
    cohorts_cfg: list[dict],
    *,
    regions: tuple[str, ...] = ("ET", "WT"),
    target_shape: tuple[int, int, int] = (6, 6, 6),
    modality_index: int = 1,
    min_component_size: int = 0,
    seed: int = 0,
    bad_dice_threshold: float = 0.7,
    min_positives: int = 1,
    n_boot: int = 200,
    ci: float = 0.95,
    alpha: float = 0.05,
    falsification_tol: float = 0.01,
    max_cases: int | None = None,
    in_distribution_cohort: str = "test",
):
    return OmegaConf.create(
        {
            "seed": seed,
            "device": "cpu",
            "analysis": {
                "qc": {
                    "modality_index": modality_index,
                    "target_shape": list(target_shape),
                    "regions": list(regions),
                },
                "qc_validate": {
                    "checkpoint": str(checkpoint),
                    "out_dir": str(out_dir),
                    "in_distribution_cohort": in_distribution_cohort,
                    "cohorts": cohorts_cfg,
                    "bad_dice_threshold": bad_dice_threshold,
                    "min_positives": min_positives,
                    "n_boot": n_boot,
                    "ci": ci,
                    "alpha": alpha,
                    "falsification_tol": falsification_tol,
                    "max_cases": max_cases,
                },
            },
            "inference": {
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": min_component_size,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                }
            },
            "model": {
                "name": "segqc",
                "in_channels": 3,
                "widths": [4, 8],
                "num_groups": 2,
                "dropout": 0.0,
            },
        }
    )


def _cohort_entry(name: str, eval_dir: Path, prep_dir: Path, entropy_cache: Path | None = None):
    return {
        "name": name,
        "eval_dir": str(eval_dir),
        "prep_dir": str(prep_dir),
        "entropy_cache": str(entropy_cache) if entropy_cache is not None else None,
    }


# ---------------------------------------------------------------------------
# 1. Every output is written
# ---------------------------------------------------------------------------


def test_run_validation_writes_every_output(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 3)
    eval_ssa, ids_ssa = _write_cohort(tmp_path, prep_dir, "ssa", 2, seed_offset=100)

    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out", tmp_path, []))

    cfg = _make_cfg(
        tmp_path / "out",
        checkpoint,
        [
            _cohort_entry("test", eval_test, prep_dir),
            _cohort_entry("ssa", eval_ssa, prep_dir),
        ],
        in_distribution_cohort="test",
    )
    _write_per_case_metrics(cfg, eval_test, prep_dir, ids_test)
    _write_per_case_metrics(cfg, eval_ssa, prep_dir, ids_ssa)

    result = run_validation(cfg)

    expected_keys = {
        "per_case_test",
        "per_case_ssa",
        "falsification_csv",
        "cells_csv",
        "gate_c_verdict_json",
        "silent_failure_csv",
        "qc_validation_config_yaml",
    }
    assert expected_keys <= set(result.keys())
    for path in result.values():
        assert path.is_file(), f"{path} was not written"

    out_dir = tmp_path / "out"
    assert (out_dir / "per_case_test.csv").is_file()
    assert (out_dir / "per_case_ssa.csv").is_file()
    assert (out_dir / "falsification.csv").is_file()
    assert (out_dir / "cells.csv").is_file()
    assert (out_dir / "gate_c_verdict.json").is_file()
    assert (out_dir / "silent_failure.csv").is_file()
    assert (out_dir / "qc_validation_config.yaml").is_file()


# ---------------------------------------------------------------------------
# 2. true_dice_<R> matches generate_pairs' identity pair exactly
# ---------------------------------------------------------------------------


def test_true_dice_matches_the_identity_pair(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 2)

    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out", tmp_path, []))
    cfg = _make_cfg(
        tmp_path / "out",
        checkpoint,
        [_cohort_entry("test", eval_test, prep_dir)],
        in_distribution_cohort="does_not_matter",
        regions=("ET", "TC", "WT"),
    )
    _write_per_case_metrics(cfg, eval_test, prep_dir, ids_test)

    result = run_validation(cfg)
    per_case = pd.read_csv(result["per_case_test"]).set_index("case_id")

    for case_id in ids_test:
        expected_et, expected_tc, expected_wt = _true_identity_dice(
            cfg, eval_test, prep_dir, case_id
        )
        assert per_case.loc[case_id, "true_dice_ET"] == pytest.approx(expected_et, abs=1e-9)
        assert per_case.loc[case_id, "true_dice_TC"] == pytest.approx(expected_tc, abs=1e-9)
        assert per_case.loc[case_id, "true_dice_WT"] == pytest.approx(expected_wt, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. Falsification runs before ANY endpoint is written
# ---------------------------------------------------------------------------


def test_falsification_check_runs_before_any_endpoint(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 2)
    eval_ssa, ids_ssa = _write_cohort(tmp_path, prep_dir, "ssa", 2, seed_offset=100)

    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out", tmp_path, []))
    cfg = _make_cfg(
        tmp_path / "out",
        checkpoint,
        [
            _cohort_entry("test", eval_test, prep_dir),
            _cohort_entry("ssa", eval_ssa, prep_dir),
        ],
        in_distribution_cohort="test",
    )
    # test's per_case_metrics.csv agrees; ssa's is planted to disagree by far
    # more than falsification_tol=0.01 on ET (a near-perfect true Dice,
    # forced down to 0.0 -- an absolute replacement, not an additive delta
    # that a near-1.0 true value could clip away).
    _write_per_case_metrics(cfg, eval_test, prep_dir, ids_test)
    _write_per_case_metrics(cfg, eval_ssa, prep_dir, ids_ssa, corrupt=(ids_ssa[0], "ET", 0.0))

    with pytest.raises(ValueError, match="falsification_check"):
        run_validation(cfg)

    out_dir = tmp_path / "out"
    # The point of this test: no endpoint file exists, not merely "it raised".
    assert not (out_dir / "cells.csv").exists()
    assert not (out_dir / "gate_c_verdict.json").exists()


# ---------------------------------------------------------------------------
# 4. Excluded case ids are counted and logged
# ---------------------------------------------------------------------------


def test_excluded_case_ids_are_counted(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 2)

    # One extra case with saved logits but NO prep_dir entry.
    orphan_label = _build_label()
    orphan_logits = _good_logits(orphan_label, seed=999)
    _write_case(prep_dir, eval_test, "test_orphan", orphan_label, orphan_logits, write_prep=False)

    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out", tmp_path, []))
    cfg = _make_cfg(
        tmp_path / "out",
        checkpoint,
        [_cohort_entry("test", eval_test, prep_dir)],
        in_distribution_cohort="does_not_matter",
    )
    _write_per_case_metrics(cfg, eval_test, prep_dir, ids_test)

    with caplog.at_level(logging.WARNING):
        result = run_validation(cfg)

    per_case = pd.read_csv(result["per_case_test"])
    assert "test_orphan" not in set(per_case["case_id"])
    assert set(per_case["case_id"]) == set(ids_test)

    excluded_records = [
        r for r in caplog.records if "excluded" in r.message and "test" in r.message
    ]
    assert excluded_records, "expected a warning naming an excluded case count"
    assert any("1 case id" in r.message for r in excluded_records)


# ---------------------------------------------------------------------------
# 5. Missing checkpoint
# ---------------------------------------------------------------------------


def test_missing_checkpoint_raises_with_a_useful_message(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 1)
    missing_checkpoint = tmp_path / "does_not_exist" / "best.pt"

    cfg = _make_cfg(
        tmp_path / "out",
        missing_checkpoint,
        [_cohort_entry("test", eval_test, prep_dir)],
    )

    with pytest.raises(FileNotFoundError, match="train_qc.py"):
        run_validation(cfg)


# ---------------------------------------------------------------------------
# 6. Missing logits dir
# ---------------------------------------------------------------------------


def test_missing_logits_dir_raises(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out", tmp_path, []))

    missing_eval_dir = tmp_path / "eval_does_not_exist"
    cfg = _make_cfg(
        tmp_path / "out",
        checkpoint,
        [_cohort_entry("test", missing_eval_dir, prep_dir)],
    )

    with pytest.raises(FileNotFoundError, match="logits"):
        run_validation(cfg)


# ---------------------------------------------------------------------------
# 7. Config block reachable at the composed path
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_at_the_composed_path() -> None:
    """The REAL project config, composed through Hydra, must expose the QC
    validation block at `cfg.analysis.qc_validate` -- the exact path
    `scripts/validate_qc.py` reads.

    Same regression shape as
    tests/test_train_qc.py::test_config_block_is_reachable_at_the_composed_path:
    a hand-built OmegaConf fixture that puts the block at the wrong nesting
    level would pass every other test in this file while the real composed
    config never produces that shape.
    """
    overrides = ["data.root_dir=/unused/for/this/test"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert "analysis" in cfg
    assert "qc_validate" in cfg.analysis
    assert "qc_validate" not in cfg  # NOT cfg.qc_validate

    qcv_cfg = cfg.analysis.qc_validate
    expected_keys = {
        "checkpoint",
        "out_dir",
        "in_distribution_cohort",
        "cohorts",
        "bad_dice_threshold",
        "min_positives",
        "n_boot",
        "ci",
        "alpha",
        "falsification_tol",
        "max_cases",
    }
    assert expected_keys <= set(qcv_cfg.keys())

    # Regions / target_shape / modality_index come from analysis.qc, not
    # analysis.qc_validate -- confirmed reachable here too, since
    # run_validation reads them from there.
    assert "qc" in cfg.analysis
    assert {"regions", "target_shape", "modality_index"} <= set(cfg.analysis.qc.keys())


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_is_deterministic(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_test, ids_test = _write_cohort(tmp_path, prep_dir, "test", 3)

    checkpoint = _write_checkpoint(tmp_path, _make_cfg(tmp_path / "out_a", tmp_path, []))

    cfg_a = _make_cfg(
        tmp_path / "out_a",
        checkpoint,
        [_cohort_entry("test", eval_test, prep_dir)],
        in_distribution_cohort="does_not_matter",
    )
    _write_per_case_metrics(cfg_a, eval_test, prep_dir, ids_test)
    result_a = run_validation(cfg_a)

    cfg_b = _make_cfg(
        tmp_path / "out_b",
        checkpoint,
        [_cohort_entry("test", eval_test, prep_dir)],
        in_distribution_cohort="does_not_matter",
    )
    result_b = run_validation(cfg_b)

    cells_a = pd.read_csv(result_a["cells_csv"])
    cells_b = pd.read_csv(result_b["cells_csv"])
    pd.testing.assert_frame_equal(cells_a, cells_b)
