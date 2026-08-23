"""Tests for scripts/replay_logits.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, following the exact same pattern as tests/test_evaluate_script.py /
tests/test_calibrate_script.py.

Everything here is synthetic, tiny, and CPU-only: 6 hand-built cases (never
real BraTS data), shape (3, 8, 8, 8), written to `tmp_path` in the same
on-disk layout `scripts/evaluate.py`
(`<eval_dir>/logits/<case_id>.npy`, `<eval_dir>/per_case_metrics.csv`) and
`neurovision.data.preprocessing.preprocess_case`
(`<prep_dir>/<case_id>/{label.npy,meta.json}`) actually produce. No config
composition through Hydra's CLI is needed -- `run_replay` takes a plain
`OmegaConf`-built `DictConfig`, exactly like every other analysis driver's
tests in this repo.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.analysis.replay import per_case_replay
from neurovision.utils.io import write_json

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_logits.py"
_spec = importlib.util.spec_from_file_location("replay_logits_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
replay_logits_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["replay_logits_script"] = replay_logits_script
_spec.loader.exec_module(replay_logits_script)

run_replay = replay_logits_script.run_replay
_resolve_out_dir = replay_logits_script._resolve_out_dir
_check_self_consistency = replay_logits_script._check_self_consistency
_compute_best_thresholds = replay_logits_script._compute_best_thresholds
resolve_lesionwise = replay_logits_script.resolve_lesionwise

# Lesion-wise scoring needs `panoptica`, which lives only in the separate
# `.venv-analysis` virtualenv -- not this project's main training `.venv`.
# `resolve_lesionwise` itself never imports panoptica (same reasoning as
# scripts/evaluate.py's own resolver), so its tests below run in both
# venvs; only a test that actually SCORES a case with lesionwise enabled
# needs to skip.
_PANOPTICA_MISSING = importlib.util.find_spec("panoptica") is None

SHAPE = (8, 8, 8)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _logit_for_prob(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _write_logits(eval_dir: Path, case_id: str, logits: np.ndarray) -> None:
    """Writes fp16 logits at `<eval_dir>/logits/<case_id>.npy`, mirroring evaluate.py."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))


def _write_case(prep_dir: Path, case_id: str, label: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    """Writes `<prep_dir>/<case_id>/{label.npy,meta.json}`, mirroring preprocess_case.

    Keys mirror `neurovision.data.preprocessing.preprocess_case`'s real
    `meta.json` (a subset -- only the keys any consumer here actually
    reads: `spacing`, via `neurovision.analysis.replay._load_label_and_spacing`).
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(label.shape),
            "cropped_shape": list(label.shape),
            "bbox": [[0, s] for s in label.shape],
            "spacing": list(spacing),
            "has_label": True,
        },
        case_dir / "meta.json",
    )


# Shape for `_graded_case`, distinct from the module-level `SHAPE` used by
# `_speckled_case`. Large enough that the inner (label-matching) cube's
# connected-component size clears `configs/inference/default.yaml`'s project
# default `min_component_size: 50` -- see this module's `_graded_case`
# docstring for why the size matters and not just the shape.
GRADED_SHAPE = (12, 12, 12)


def _graded_case(shape=GRADED_SHAPE):
    """A label/probability pair whose Dice genuinely depends on the threshold.

    Ground truth is a cube (label value 3, so ET == TC == WT). Predicted
    probability is high (0.8) inside that cube, medium (0.3) in a
    surrounding shell, and low (0.05) everywhere else -- so a low threshold
    over-predicts (includes the shell), the project default 0.5 matches
    exactly, and a high threshold under-predicts (misses everything).

    Sized deliberately, not just shaped: `neurovision.inference.postprocess
    .remove_small_components` (via MONAI's `remove_small_objects`) treats
    identical ET/TC/WT channels as ADJACENT along the channel axis under
    `connectivity=1`, so a same-shaped blob present in all 3 channels forms
    ONE connected component 3x the size of a single channel's blob -- not 3
    separate same-sized ones. The inner cube here is 4^3 = 64 voxels per
    channel (192 merged across channels), comfortably above the project
    default `min_component_size: 50`, so a sweep run at the project's
    default post-processing settings (as `scripts/replay_logits.py`'s
    `threshold_sweep` uses) does not have its exact-match case silently
    filtered away as if it were speckle.
    """
    label = np.zeros(shape, dtype=np.uint8)
    label[4:8, 4:8, 4:8] = 3  # 4^3 = 64 voxel inner cube

    probs = np.full(shape, 0.05, dtype=np.float64)
    probs[2:10, 2:10, 2:10] = 0.3  # 8^3 = 512 voxel shell
    probs[4:8, 4:8, 4:8] = 0.8  # inner cube overrides the shell value

    logit_field = np.vectorize(_logit_for_prob)(probs).astype(np.float32)
    logits = np.stack([logit_field, logit_field, logit_field], axis=0)
    return logits, label


def _speckled_case(shape=SHAPE):
    """A confident cube prediction plus a small disconnected false-positive speck.

    Used to give the post-processing ablation a variant (min_component_size)
    that visibly changes Dice. Per-channel the speck is 3 voxels and the
    cube is 64, but (see `_graded_case`'s docstring for why) identical
    per-channel blobs merge into one component 3x the size across the
    channel axis under `connectivity=1` -- speck 9, cube 192 -- so a
    `min_component_size` between those two values (this module's tests use
    50, the project default) removes the speck and keeps the cube.
    """
    label = np.zeros(shape, dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 3  # 4^3 = 64 voxel cube, matches the prediction exactly

    logits = np.full((3, *shape), -10.0, dtype=np.float32)
    for c in range(3):
        logits[c][2:6, 2:6, 2:6] = 10.0
    # A deliberate 3-voxel false-positive speck, far from and disconnected
    # from the real lesion, in every channel.
    for c in range(3):
        logits[c, 0, 0, 0] = 10.0
        logits[c, 0, 0, 1] = 10.0
        logits[c, 0, 0, 2] = 10.0
    return logits, label


def _build_split(tmp_path: Path, n_graded: int = 3, n_speckled: int = 3):
    """Builds a 6-case synthetic eval_dir/prep_dir pair.

    Returns:
        `(eval_dir, prep_dir, case_ids)`.
    """
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    case_ids: list[str] = []

    for i in range(n_graded):
        case_id = f"graded_{i:03d}"
        logits, label = _graded_case()
        _write_logits(eval_dir, case_id, logits)
        _write_case(prep_dir, case_id, label)
        case_ids.append(case_id)

    for i in range(n_speckled):
        case_id = f"speck_{i:03d}"
        logits, label = _speckled_case()
        _write_logits(eval_dir, case_id, logits)
        _write_case(prep_dir, case_id, label)
        case_ids.append(case_id)

    return eval_dir, prep_dir, case_ids


def _base_replay_cfg(eval_dir: Path, prep_dir: Path, out_dir: Path, **overrides) -> dict:
    cfg = {
        "seed": 0,
        "analysis": {
            "replay": {
                "eval_dir": str(eval_dir),
                "prep_dir": str(prep_dir),
                "out_dir": str(out_dir),
                "case_ids": None,
                "threshold_sweep": {"enabled": True, "thresholds": [0.3, 0.5, 0.7]},
                "postprocess_ablation": {
                    "enabled": True,
                    "variants": {
                        "raw": {
                            "min_component_size": 0,
                            "enforce_nesting": False,
                            "keep_largest_only": False,
                        },
                        "filtered": {
                            "min_component_size": 5,
                            "enforce_nesting": False,
                            "keep_largest_only": False,
                        },
                    },
                },
                "baseline_comparison": {"enabled": False, "eval_dir": None},
            }
        },
    }
    replay_block = cfg["analysis"]["replay"]
    for key, value in overrides.items():
        replay_block[key] = value
    return cfg


def _make_cfg(eval_dir: Path, prep_dir: Path, out_dir: Path, **overrides):
    return OmegaConf.create(_base_replay_cfg(eval_dir, prep_dir, out_dir, **overrides))


# ---------------------------------------------------------------------------
# threshold_sweep -> threshold_sweep.csv
# ---------------------------------------------------------------------------


def test_run_replay_writes_one_sweep_row_per_threshold_in_order(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=2, n_speckled=0)
    out_dir = tmp_path / "out"
    thresholds = [0.9, 0.1, 0.5]
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": True, "thresholds": thresholds},
        postprocess_ablation={"enabled": False, "variants": {}},
    )

    results = run_replay(cfg)

    sweep_path = out_dir / "eval" / "threshold_sweep.csv"
    assert sweep_path.is_file()
    sweep_df = pd.read_csv(sweep_path)

    assert list(sweep_df["threshold"]) == thresholds
    assert "threshold_sweep" in results
    assert list(results["threshold_sweep"]["threshold"]) == thresholds

    # Dice must genuinely differ across thresholds (see _graded_case). 0.8
    # (inner) < 0.9 -> nothing fires -> empty prediction vs non-empty label
    # -> Dice 0.0. 0.3 (shell) and 0.8 (inner) both >= 0.1 -> over-prediction
    # -> Dice < 1. Only the inner cube (0.8) clears 0.5 -> exact match -> 1.0.
    assert sweep_df.loc[sweep_df["threshold"] == 0.5, "dice_WT_mean"].iloc[0] == pytest.approx(1.0)
    assert sweep_df.loc[sweep_df["threshold"] == 0.9, "dice_WT_mean"].iloc[0] == pytest.approx(0.0)
    assert sweep_df.loc[sweep_df["threshold"] == 0.1, "dice_WT_mean"].iloc[0] < 1.0


# ---------------------------------------------------------------------------
# postprocess_ablation -> postprocess_ablation.csv
# ---------------------------------------------------------------------------


def test_run_replay_writes_one_ablation_row_per_variant_in_order(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=0, n_speckled=2)
    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": False, "thresholds": [0.5]},
        postprocess_ablation={
            "enabled": True,
            "variants": {
                "raw": {
                    "min_component_size": 0,
                    "enforce_nesting": False,
                    "keep_largest_only": False,
                },
                "filtered": {
                    # 50, not e.g. 5: see _speckled_case's docstring -- the
                    # merged (cross-channel) speck component is 9 voxels,
                    # not 3, so a small filtered value would not remove it.
                    "min_component_size": 50,
                    "enforce_nesting": False,
                    "keep_largest_only": False,
                },
            },
        },
    )

    results = run_replay(cfg)

    ablation_path = out_dir / "eval" / "postprocess_ablation.csv"
    assert ablation_path.is_file()
    ablation_df = pd.read_csv(ablation_path)

    assert list(ablation_df["variant"]) == ["raw", "filtered"]
    assert "postprocess_ablation" in results

    raw_dice = ablation_df.loc[ablation_df["variant"] == "raw", "dice_WT_mean"].iloc[0]
    filtered_dice = ablation_df.loc[ablation_df["variant"] == "filtered", "dice_WT_mean"].iloc[0]
    # The 3-voxel speck survives min_component_size=0 and is removed at 5.
    assert filtered_dice == pytest.approx(1.0)
    assert raw_dice < filtered_dice


# ---------------------------------------------------------------------------
# per_case_default.csv + self-consistency check
# ---------------------------------------------------------------------------


def test_self_consistency_passes_when_published_matches_replay(tmp_path, caplog):
    eval_dir, prep_dir, case_ids = _build_split(tmp_path, n_graded=2, n_speckled=2)
    out_dir = tmp_path / "out"

    # The published table IS this replay's own default-settings output --
    # the two must agree exactly.
    published = per_case_replay(eval_dir, prep_dir)
    published.to_csv(eval_dir / "per_case_metrics.csv")

    cfg = _make_cfg(eval_dir, prep_dir, out_dir)
    with caplog.at_level(logging.INFO):
        run_replay(cfg)

    assert any("self-consistency mean absolute deltas" in r.message for r in caplog.records)
    per_case_path = out_dir / "eval" / "per_case_default.csv"
    assert per_case_path.is_file()
    written = pd.read_csv(per_case_path)
    assert set(written["case_id"]) == set(case_ids)


def test_self_consistency_raises_when_published_is_perturbed(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=2, n_speckled=0)
    out_dir = tmp_path / "out"

    published = per_case_replay(eval_dir, prep_dir)
    # Perturb far beyond tolerance -- simulates a published table that came
    # from a different threshold/post-processing/checkpoint.
    published["dice_ET"] = published["dice_ET"] - 0.5
    published.to_csv(eval_dir / "per_case_metrics.csv")

    cfg = _make_cfg(eval_dir, prep_dir, out_dir)
    with pytest.raises(ValueError, match="disagrees with the published"):
        run_replay(cfg)


def test_self_consistency_missing_published_warns_and_does_not_raise(tmp_path, caplog):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=1, n_speckled=1)
    out_dir = tmp_path / "out"
    # Deliberately no per_case_metrics.csv written at eval_dir.

    cfg = _make_cfg(eval_dir, prep_dir, out_dir)
    with caplog.at_level(logging.WARNING):
        results = run_replay(cfg)

    assert any("skipping the self-consistency check" in r.message for r in caplog.records)
    assert "per_case_default" in results
    assert (out_dir / "eval" / "per_case_default.csv").is_file()


def test_check_self_consistency_directly_missing_file_returns_none(tmp_path, caplog):
    eval_dir = tmp_path / "eval_no_metrics"
    eval_dir.mkdir()
    replayed = pd.DataFrame(
        {"dice_ET": [1.0], "dice_TC": [1.0], "dice_WT": [1.0]},
        index=pd.Index(["c1"], name="case_id"),
    )
    with caplog.at_level(logging.WARNING):
        result = _check_self_consistency(replayed, eval_dir)
    assert result is None
    assert any("skipping the self-consistency check" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# best_threshold.json
# ---------------------------------------------------------------------------


def test_compute_best_thresholds_picks_true_argmax_and_has_caveat():
    sweep_df = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "dice_ET_mean": [0.5, 0.6, 0.4],
            "dice_TC_mean": [0.9, 0.6, 0.6],
            "dice_WT_mean": [0.6, 0.6, 0.6],
            "dice_mean_mean": [0.7, 0.6, 0.6],
        }
    )
    metrics = ("dice_ET", "dice_TC", "dice_WT", "dice_mean")

    result = _compute_best_thresholds(sweep_df, metrics)

    assert result["dice_ET"]["best_threshold"] == pytest.approx(0.5)
    assert result["dice_ET"]["best_value"] == pytest.approx(0.6)
    assert result["dice_ET"]["value_at_0.5"] == pytest.approx(0.6)
    assert result["dice_ET"]["delta_over_0.5"] == pytest.approx(0.0)

    assert result["dice_TC"]["best_threshold"] == pytest.approx(0.3)
    assert result["dice_TC"]["delta_over_0.5"] == pytest.approx(0.3)

    # A three-way tie: pandas' idxmax returns the FIRST occurrence, which is
    # threshold=0.3 here (row order [0.3, 0.5, 0.7]).
    assert result["dice_WT"]["best_threshold"] == pytest.approx(0.3)

    assert isinstance(result["caveat"], str) and len(result["caveat"]) > 0


def test_compute_best_thresholds_raises_without_half(tmp_path):
    sweep_df = pd.DataFrame({"threshold": [0.3, 0.7], "dice_ET_mean": [0.5, 0.4]})
    with pytest.raises(ValueError, match="0.5"):
        _compute_best_thresholds(sweep_df, ("dice_ET",))


def test_run_replay_writes_best_threshold_json_matching_direct_computation(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=3, n_speckled=0)
    out_dir = tmp_path / "out"
    thresholds = [0.3, 0.5, 0.7]
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": True, "thresholds": thresholds},
        postprocess_ablation={"enabled": False, "variants": {}},
    )

    run_replay(cfg)

    best_path = out_dir / "eval" / "best_threshold.json"
    assert best_path.is_file()
    best = json.loads(best_path.read_text())

    assert "dice_WT" in best
    assert best["dice_WT"]["best_threshold"] == pytest.approx(0.5)
    assert "caveat" in best and "val" in best["caveat"]


def test_run_replay_skips_best_threshold_when_sweep_disabled(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=1, n_speckled=0)
    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": False, "thresholds": [0.5]},
        postprocess_ablation={"enabled": False, "variants": {}},
    )

    run_replay(cfg)

    assert not (out_dir / "eval" / "best_threshold.json").exists()


# ---------------------------------------------------------------------------
# replay_config.yaml + _resolve_out_dir
# ---------------------------------------------------------------------------


def test_run_replay_writes_replay_config_yaml(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=1, n_speckled=0)
    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": False, "thresholds": [0.5]},
        postprocess_ablation={"enabled": False, "variants": {}},
    )

    run_replay(cfg)

    config_path = out_dir / "eval" / "replay_config.yaml"
    assert config_path.is_file()
    assert "eval_dir" in config_path.read_text()


def test_resolve_out_dir_uses_eval_dir_basename(tmp_path):
    eval_dir = tmp_path / "some_experiment" / "eval_test"
    prep_dir = tmp_path / "prep"
    out_parent = tmp_path / "out"
    cfg = _make_cfg(eval_dir, prep_dir, out_parent)

    resolved = _resolve_out_dir(cfg)

    assert resolved == out_parent / "eval_test"
    assert resolved.is_dir()


def test_resolve_out_dir_raises_when_eval_dir_unset(tmp_path):
    cfg = _make_cfg(tmp_path / "eval", tmp_path / "prep", tmp_path / "out")
    cfg.analysis.replay.eval_dir = None
    with pytest.raises(ValueError, match="eval_dir"):
        _resolve_out_dir(cfg)


# ---------------------------------------------------------------------------
# baseline_comparison -> comparison_default.csv
# ---------------------------------------------------------------------------


def test_run_replay_baseline_comparison_writes_comparison_csv(tmp_path):
    eval_dir, prep_dir, case_ids = _build_split(tmp_path, n_graded=2, n_speckled=0)
    baseline_dir = tmp_path / "baseline_eval"

    # Baseline predicts the exact opposite of the label everywhere -- a
    # clearly worse model, so the comparison table is not degenerate.
    for case_id in case_ids:
        _, label = _graded_case()
        bad_logits = np.full((3, *label.shape), 10.0, dtype=np.float32)
        for c in range(3):
            bad_logits[c][label > 0] = -10.0
        _write_logits(baseline_dir, case_id, bad_logits)

    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": False, "thresholds": [0.5]},
        postprocess_ablation={"enabled": False, "variants": {}},
        baseline_comparison={"enabled": True, "eval_dir": str(baseline_dir)},
    )

    results = run_replay(cfg)

    comparison_path = out_dir / "eval" / "comparison_default.csv"
    assert comparison_path.is_file()
    assert "comparison_default" in results
    comparison_df = results["comparison_default"]
    assert "dice_WT" in comparison_df.index
    mean_eval = comparison_df.loc["dice_WT", "mean_eval"]
    mean_baseline = comparison_df.loc["dice_WT", "mean_baseline_eval"]
    assert mean_eval > mean_baseline


def test_run_replay_skips_comparison_when_disabled(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=1, n_speckled=0)
    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": False, "thresholds": [0.5]},
        postprocess_ablation={"enabled": False, "variants": {}},
    )

    results = run_replay(cfg)

    assert "comparison_default" not in results
    assert not (out_dir / "eval" / "comparison_default.csv").exists()


# ---------------------------------------------------------------------------
# resolve_lesionwise (cfg.analysis.replay.lesionwise) -- mirrors
# scripts/evaluate.py's own resolve_lesionwise tests exactly, off the
# replay-specific config path. Never imports panoptica, so none of these
# are skipped.
# ---------------------------------------------------------------------------


def test_resolve_lesionwise_absent_key_returns_none():
    """Backward compatibility: a config composed before this key existed
    must still run, with lesion-wise scoring simply off."""
    replay_cfg = OmegaConf.create({"eval_dir": "x", "prep_dir": "y"})
    assert resolve_lesionwise(replay_cfg) is None


def test_resolve_lesionwise_disabled_returns_none():
    replay_cfg = OmegaConf.create({"lesionwise": {"enabled": False, "min_lesion_voxels": 10}})
    assert resolve_lesionwise(replay_cfg) is None

    # The whole block set to null is equally "off".
    replay_cfg_null = OmegaConf.create({"lesionwise": None})
    assert resolve_lesionwise(replay_cfg_null) is None


def test_resolve_lesionwise_returns_settings_when_enabled():
    replay_cfg = OmegaConf.create(
        {
            "lesionwise": {
                "enabled": True,
                "min_lesion_voxels": 25,
                "matching_threshold": 0.3,
                "nsd_tolerance_mm": 2.0,
                "connectivity": 6,
            }
        }
    )

    settings = resolve_lesionwise(replay_cfg)

    assert settings == {
        "min_lesion_voxels": 25,
        "matching_threshold": 0.3,
        "nsd_tolerance_mm": 2.0,
        "connectivity": 6,
    }
    assert isinstance(settings["min_lesion_voxels"], int)
    assert isinstance(settings["matching_threshold"], float)
    assert isinstance(settings["nsd_tolerance_mm"], float)
    assert isinstance(settings["connectivity"], int)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_lesion_voxels": -1},
        {"matching_threshold": 0.0},
        {"matching_threshold": 1.5},
        {"nsd_tolerance_mm": 0.0},
        {"nsd_tolerance_mm": -1.0},
        {"connectivity": 10},
    ],
    ids=[
        "negative_min_lesion_voxels",
        "matching_threshold_zero",
        "matching_threshold_above_one",
        "nsd_tolerance_zero",
        "nsd_tolerance_negative",
        "connectivity_not_6_18_26",
    ],
)
def test_resolve_lesionwise_raises_on_invalid_settings(overrides: dict[str, object]):
    settings = {
        "enabled": True,
        "min_lesion_voxels": 50,
        "matching_threshold": 0.5,
        "nsd_tolerance_mm": 1.0,
        "connectivity": 26,
    }
    settings.update(overrides)
    replay_cfg = OmegaConf.create({"lesionwise": settings})

    with pytest.raises(ValueError):
        resolve_lesionwise(replay_cfg)


# ---------------------------------------------------------------------------
# lesionwise wiring end-to-end -- per_case_default.csv only, never the
# sweep or the ablation.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    _PANOPTICA_MISSING,
    reason="panoptica is not installed in this venv (see requirements-analysis.txt); "
    "run from .venv-analysis to exercise lesion-wise scoring",
)
def test_run_replay_lesionwise_adds_columns_only_to_per_case_default(tmp_path):
    eval_dir, prep_dir, _ = _build_split(tmp_path, n_graded=2, n_speckled=0)
    out_dir = tmp_path / "out"
    cfg = _make_cfg(
        eval_dir,
        prep_dir,
        out_dir,
        threshold_sweep={"enabled": True, "thresholds": [0.3, 0.5]},
        postprocess_ablation={
            "enabled": True,
            "variants": {"raw": {"min_component_size": 0}},
        },
        lesionwise={"enabled": True, "min_lesion_voxels": 0},
    )

    results = run_replay(cfg)

    per_case_cols = results["per_case_default"].columns
    assert any(col.startswith("lw") for col in per_case_cols)
    # Neither the sweep nor the ablation ever gets lesion-wise columns --
    # they are per-threshold/per-variant summary tables (dice_*_mean /
    # dice_*_median), which lesionwise_case_metrics does not feed at all.
    assert not any(col.startswith("lw") for col in results["threshold_sweep"].columns)
    assert not any(col.startswith("lw") for col in results["postprocess_ablation"].columns)
