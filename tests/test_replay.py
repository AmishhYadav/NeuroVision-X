"""Tests for `neurovision.analysis.replay`.

Everything here is synthetic, tiny, and CPU-only: a handful of small hand-
built cases (never real BraTS data) with a known, exactly-computable ground
truth, written to `tmp_path` in the same on-disk layout `scripts/evaluate.py`
(`<eval_dir>/logits/<case_id>.npy`) and
`neurovision.data.preprocessing.preprocess_case`
(`<prep_dir>/<case_id>/{label.npy,meta.json}`) actually produce.
"""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis import replay
from neurovision.analysis.statistics import compare_models
from neurovision.utils.io import write_json

# Lesion-wise scoring needs `panoptica`, which lives only in the separate
# `.venv-analysis` virtualenv (see requirements-analysis.txt) -- not in this
# project's main training `.venv`. Any test that actually RUNS lesion-wise
# scoring must skip cleanly in the training .venv rather than fail. Tests
# that only prove the `lesionwise=None` default is unchanged, or that the
# module itself imports without panoptica, are NOT gated by this -- they
# must run (and pass) in both venvs. Mirrors tests/test_evaluate_script.py's
# own `_PANOPTICA_MISSING`.
_PANOPTICA_MISSING = importlib.util.find_spec("panoptica") is None

# A post-processing config with every optional step off, so a test can
# reason about `_binarize_regions` alone without component filtering or
# nesting repair interfering.
_RAW_PP_CFG = {
    "enforce_nesting": False,
    "min_component_size": 0,
    "connectivity": 1,
    "keep_largest_only": False,
    "et_min_volume": 0,
}

SPACING = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _write_logits(eval_dir, case_id: str, logits: np.ndarray) -> None:
    """Writes fp16 logits at `<eval_dir>/logits/<case_id>.npy`, mirroring evaluate.py."""
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))


def _write_case(prep_dir, case_id: str, label: np.ndarray, spacing=SPACING) -> None:
    """Writes `<prep_dir>/<case_id>/{label.npy,meta.json}`, mirroring preprocess_case."""
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", label.astype(np.uint8))
    write_json({"spacing": list(spacing), "case_id": case_id}, case_dir / "meta.json")


def _perfect_case(shape=(8, 8, 8)):
    """A label/logits pair that should score Dice 1.0 for ET, TC and WT.

    A single inner cube is enhancing tumor (label value 3), which under
    `classes_to_regions` makes ET == TC == WT == that cube exactly. Logits
    are +/-10 (sigmoid saturates far from 0.5) so the 0.5-threshold
    prediction matches the label exactly, voxel for voxel.
    """
    label = np.zeros(shape, dtype=np.uint8)
    inner = (slice(2, 6), slice(2, 6), slice(2, 6))
    label[inner] = 3

    logits = np.full((3, *shape), -10.0, dtype=np.float32)
    for c in range(3):
        logits[c][inner] = 10.0
    return logits, label


# ---------------------------------------------------------------------------
# load_case_logits / available_logit_cases
# ---------------------------------------------------------------------------


def test_load_case_logits_returns_float32(tmp_path):
    eval_dir = tmp_path / "eval"
    rng = np.random.default_rng(0)
    original = rng.normal(size=(3, 5, 6, 7)).astype(np.float32)
    _write_logits(eval_dir, "case_001", original)

    loaded = replay.load_case_logits(eval_dir, "case_001")

    assert loaded.dtype == np.float32
    assert loaded.shape == (3, 5, 6, 7)
    # fp16 round-trip loses precision but should stay close.
    assert np.allclose(loaded, original, atol=1e-2)


def test_load_case_logits_missing_raises_with_path(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        replay.load_case_logits(eval_dir, "does_not_exist")

    message = str(exc_info.value)
    assert "does_not_exist" in message
    assert str((eval_dir / "logits" / "does_not_exist.npy").resolve()) in message


def test_available_logit_cases(tmp_path):
    eval_dir = tmp_path / "eval"
    logits, _ = _perfect_case()
    _write_logits(eval_dir, "case_b", logits)
    _write_logits(eval_dir, "case_a", logits)

    assert replay.available_logit_cases(eval_dir) == ["case_a", "case_b"]


def test_available_logit_cases_no_logits_dir(tmp_path):
    assert replay.available_logit_cases(tmp_path / "nothing_here") == []


# ---------------------------------------------------------------------------
# replay_case
# ---------------------------------------------------------------------------


def test_replay_case_perfect_prediction_dice_one():
    logits, label = _perfect_case()

    metrics = replay.replay_case(logits, label, threshold=0.5, postprocess_cfg=_RAW_PP_CFG)

    assert metrics["dice_ET"] == pytest.approx(1.0)
    assert metrics["dice_TC"] == pytest.approx(1.0)
    assert metrics["dice_WT"] == pytest.approx(1.0)


def test_replay_case_disjoint_prediction_dice_zero():
    shape = (8, 8, 8)
    label = np.zeros(shape, dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 3  # ET == TC == WT == this cube

    # Prediction fires everywhere EXCEPT that cube: completely disjoint from
    # the label, for all three channels.
    logits = np.full((3, *shape), 10.0, dtype=np.float32)
    for c in range(3):
        logits[c][2:6, 2:6, 2:6] = -10.0

    metrics = replay.replay_case(logits, label, threshold=0.5, postprocess_cfg=_RAW_PP_CFG)

    assert metrics["dice_ET"] == pytest.approx(0.0)
    assert metrics["dice_TC"] == pytest.approx(0.0)
    assert metrics["dice_WT"] == pytest.approx(0.0)


def test_replay_case_per_channel_threshold_differs_from_scalar():
    """A length-3 threshold must NOT silently collapse to its first element."""
    shape = (6, 6, 6)
    # A single label value everywhere makes ET == TC == WT == the whole
    # volume, so any per-channel difference in the result can only come from
    # the threshold being applied per channel.
    label = np.full(shape, 3, dtype=np.uint8)

    def _logit_for_prob(p: float) -> float:
        return float(np.log(p / (1.0 - p)))

    # Constant probability per channel, chosen so a 0.5 scalar threshold and
    # the per-channel thresholds below disagree on every single channel.
    p_et, p_tc, p_wt = 0.6, 0.4, 0.9
    logits = np.empty((3, *shape), dtype=np.float32)
    logits[0] = _logit_for_prob(p_et)
    logits[1] = _logit_for_prob(p_tc)
    logits[2] = _logit_for_prob(p_wt)

    scalar_result = replay.replay_case(logits, label, threshold=0.5, postprocess_cfg=_RAW_PP_CFG)
    # ET fires (0.6 >= 0.5) -> full-volume match -> dice 1.0
    # TC does not fire (0.4 < 0.5) -> empty prediction vs full label -> 0.0
    # WT fires (0.9 >= 0.5) -> dice 1.0
    assert scalar_result["dice_ET"] == pytest.approx(1.0)
    assert scalar_result["dice_TC"] == pytest.approx(0.0)
    assert scalar_result["dice_WT"] == pytest.approx(1.0)

    per_channel_result = replay.replay_case(
        logits, label, threshold=[0.7, 0.3, 0.95], postprocess_cfg=_RAW_PP_CFG
    )
    # ET no longer fires (0.6 < 0.7) -> 0.0
    # TC now fires (0.4 >= 0.3) -> 1.0
    # WT no longer fires (0.9 < 0.95) -> 0.0
    assert per_channel_result["dice_ET"] == pytest.approx(0.0)
    assert per_channel_result["dice_TC"] == pytest.approx(1.0)
    assert per_channel_result["dice_WT"] == pytest.approx(0.0)

    assert per_channel_result != scalar_result


def test_replay_case_per_channel_threshold_wrong_length_raises():
    logits, label = _perfect_case()
    with pytest.raises(ValueError, match="threshold"):
        replay.replay_case(logits, label, threshold=[0.5, 0.5])


def test_replay_case_spacing_changes_hd95_not_dice():
    shape = (10, 10, 10)
    label = np.zeros(shape, dtype=np.uint8)
    label[3:7, 3:7, 3:7] = 3  # ET == TC == WT == this 4^3 cube

    # Prediction is a strictly LARGER cube (superset), so it is non-empty,
    # the label is non-empty, and the boundaries disagree -> finite nonzero
    # HD95, not one of the degenerate empty/empty or one-sided-empty cases.
    logits = np.full((3, *shape), -10.0, dtype=np.float32)
    for c in range(3):
        logits[c][2:8, 2:8, 2:8] = 10.0

    result_1mm = replay.replay_case(
        logits, label, threshold=0.5, postprocess_cfg=_RAW_PP_CFG, spacing=(1.0, 1.0, 1.0)
    )
    result_2mm = replay.replay_case(
        logits, label, threshold=0.5, postprocess_cfg=_RAW_PP_CFG, spacing=(2.0, 2.0, 2.0)
    )

    assert result_1mm["dice_WT"] == pytest.approx(result_2mm["dice_WT"])
    assert result_1mm["hd95_WT"] != pytest.approx(result_2mm["hd95_WT"])
    assert result_1mm["hd95_WT"] > 0.0
    # Doubling every axis' spacing doubles a Euclidean distance transform
    # exactly.
    assert result_2mm["hd95_WT"] == pytest.approx(2.0 * result_1mm["hd95_WT"])


def test_replay_case_default_postprocess_cfg_matches_project_default():
    """`postprocess_cfg=None` must read configs/inference/default.yaml, not invent defaults."""
    logits, label = _perfect_case()
    # Should not raise, and should score the perfect case at 1.0 exactly
    # like the explicit no-op config does (the project default's
    # min_component_size=50 does not remove a 4^3=64-voxel component).
    metrics = replay.replay_case(logits, label)
    assert metrics["dice_WT"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# threshold_sweep
# ---------------------------------------------------------------------------


def _build_graded_case(eval_dir, prep_dir, case_id: str, shape=(10, 10, 10)):
    """A case where Dice genuinely depends on the threshold.

    Ground truth is a small inner cube. Predicted probability is high (0.8)
    inside that cube, medium (0.3) in a surrounding shell, and low (0.05)
    everywhere else -- so a low threshold over-predicts (includes the
    shell), a mid threshold matches exactly, and a high threshold
    under-predicts (misses everything).
    """
    label = np.zeros(shape, dtype=np.uint8)
    label[4:6, 4:6, 4:6] = 3  # inner cube, 8 voxels

    def _logit_for_prob(p: float) -> float:
        return float(np.log(p / (1.0 - p)))

    probs = np.full(shape, 0.05, dtype=np.float64)
    probs[3:7, 3:7, 3:7] = 0.3  # shell
    probs[4:6, 4:6, 4:6] = 0.8  # inner cube overrides the shell value

    logit_field = np.vectorize(_logit_for_prob)(probs).astype(np.float32)
    logits = np.stack([logit_field, logit_field, logit_field], axis=0)

    _write_logits(eval_dir, case_id, logits)
    _write_case(prep_dir, case_id, label)


def test_threshold_sweep_rows_in_given_order_and_differ(tmp_path):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    _build_graded_case(eval_dir, prep_dir, "case_001")

    thresholds = [0.9, 0.1, 0.5]
    result = replay.threshold_sweep(
        eval_dir,
        prep_dir,
        thresholds,
        postprocess_cfg=_RAW_PP_CFG,
    )

    assert list(result["threshold"]) == thresholds
    assert list(result["n"]) == [1, 1, 1]

    row_09 = result[result["threshold"] == 0.9].iloc[0]
    row_01 = result[result["threshold"] == 0.1].iloc[0]
    row_05 = result[result["threshold"] == 0.5].iloc[0]

    # 0.8 (inner) < 0.9 -> nothing fires -> empty prediction vs non-empty
    # label -> Dice 0.0.
    assert row_09["dice_WT_mean"] == pytest.approx(0.0)
    # 0.3 (shell) and 0.8 (inner) both >= 0.1 -> over-prediction -> Dice < 1.
    assert row_01["dice_WT_mean"] < 1.0
    # Only the inner cube (0.8) clears 0.5 -> exact match -> Dice 1.0.
    assert row_05["dice_WT_mean"] == pytest.approx(1.0)

    # All three thresholds gave a different value -- the sweep is not a
    # constant column.
    assert len({row_09["dice_WT_mean"], row_01["dice_WT_mean"], row_05["dice_WT_mean"]}) == 3


# ---------------------------------------------------------------------------
# postprocess_ablation
# ---------------------------------------------------------------------------


def test_postprocess_ablation_speck_removed_by_min_component_size(tmp_path):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    shape = (12, 12, 12)

    inner = (slice(4, 8), slice(4, 8), slice(4, 8))  # 4^3 = 64 voxels
    label = np.zeros(shape, dtype=np.uint8)
    label[inner] = 3

    logits = np.full((3, *shape), -10.0, dtype=np.float32)
    for c in range(3):
        logits[c][inner] = 10.0
    # A deliberate 3-voxel false-positive speck in the WT channel only, far
    # from the real lesion and disconnected from it.
    logits[2, 0, 0, 0] = 10.0
    logits[2, 0, 0, 1] = 10.0
    logits[2, 0, 0, 2] = 10.0

    _write_logits(eval_dir, "case_001", logits)
    _write_case(prep_dir, "case_001", label)

    variants = {
        "raw": {**_RAW_PP_CFG, "min_component_size": 0},
        "filtered": {**_RAW_PP_CFG, "min_component_size": 5},
    }
    result = replay.postprocess_ablation(eval_dir, prep_dir, variants, threshold=0.5)

    assert list(result["variant"]) == ["raw", "filtered"]
    raw_dice = result[result["variant"] == "raw"].iloc[0]["dice_WT_mean"]
    filtered_dice = result[result["variant"] == "filtered"].iloc[0]["dice_WT_mean"]

    assert filtered_dice == pytest.approx(1.0)
    assert raw_dice < filtered_dice


# ---------------------------------------------------------------------------
# per_case_replay
# ---------------------------------------------------------------------------


def test_per_case_replay_columns_and_compare_models(tmp_path):
    prep_dir = tmp_path / "prep"
    eval_dir_a = tmp_path / "eval_a"
    eval_dir_b = tmp_path / "eval_b"

    logits_perfect, label = _perfect_case()
    _write_case(prep_dir, "case_001", label)
    _write_case(prep_dir, "case_002", label)

    # Model A predicts perfectly on both cases.
    _write_logits(eval_dir_a, "case_001", logits_perfect)
    _write_logits(eval_dir_a, "case_002", logits_perfect)

    # Model B is disjoint on both cases (worse).
    logits_bad = np.full((3, *label.shape), 10.0, dtype=np.float32)
    for c in range(3):
        logits_bad[c][2:6, 2:6, 2:6] = -10.0
    _write_logits(eval_dir_b, "case_001", logits_bad)
    _write_logits(eval_dir_b, "case_002", logits_bad)

    df_a = replay.per_case_replay(eval_dir_a, prep_dir, postprocess_cfg=_RAW_PP_CFG)
    df_b = replay.per_case_replay(eval_dir_b, prep_dir, postprocess_cfg=_RAW_PP_CFG)

    assert df_a.index.name == "case_id"
    assert set(df_a.index) == {"case_001", "case_002"}
    assert {"dice_ET", "dice_TC", "dice_WT"}.issubset(df_a.columns)

    comparison = compare_models(
        df_a,
        df_b,
        generator=np.random.default_rng(0),
        metrics=["dice_ET", "dice_TC", "dice_WT"],
        n_boot=200,
    )
    assert "dice_WT" in comparison.index
    assert comparison.loc["dice_WT", "mean_model"] == pytest.approx(1.0)
    assert comparison.loc["dice_WT", "mean_baseline"] == pytest.approx(0.0)


def test_per_case_replay_missing_label_is_skipped_and_warned(tmp_path, caplog):
    prep_dir = tmp_path / "prep"
    eval_dir = tmp_path / "eval"

    logits, label = _perfect_case()
    _write_case(prep_dir, "case_labeled", label)
    _write_logits(eval_dir, "case_labeled", logits)

    # "case_unlabeled" has saved logits but NO label.npy on disk at all.
    _write_logits(eval_dir, "case_unlabeled", logits)

    with caplog.at_level(logging.WARNING):
        result = replay.per_case_replay(eval_dir, prep_dir, postprocess_cfg=_RAW_PP_CFG)

    assert list(result.index) == ["case_labeled"]
    assert any("case_unlabeled" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# lesion-wise support (opt-in, additive, panoptica lazily imported)
# ---------------------------------------------------------------------------


def test_replay_case_lesionwise_none_is_unchanged():
    """`lesionwise=None` (the default) must be byte-identical to before this
    parameter existed -- no new keys, same values."""
    logits, label = _perfect_case()

    baseline = replay.replay_case(logits, label, postprocess_cfg=_RAW_PP_CFG)
    explicit_none = replay.replay_case(logits, label, postprocess_cfg=_RAW_PP_CFG, lesionwise=None)

    assert explicit_none == baseline
    assert not any(key.startswith("lw") for key in explicit_none)


@pytest.mark.skipif(
    _PANOPTICA_MISSING,
    reason="panoptica is not installed in this venv (see requirements-analysis.txt); "
    "run from .venv-analysis to exercise lesion-wise scoring",
)
def test_replay_case_lesionwise_adds_columns():
    """Lesion-wise scoring is ADDITIVE -- turning it on must not move an
    already-existing voxel-wise metric, only add new `lw*` keys."""
    logits, label = _perfect_case()

    without = replay.replay_case(logits, label, postprocess_cfg=_RAW_PP_CFG)
    with_lw = replay.replay_case(logits, label, postprocess_cfg=_RAW_PP_CFG, lesionwise={})

    # Every pre-existing key is bit-identical.
    for key, value in without.items():
        assert with_lw[key] == value

    # New lesion-wise keys were actually added.
    new_keys = set(with_lw) - set(without)
    assert new_keys
    assert all(key.startswith("lw") for key in new_keys)
    assert with_lw["lwdice_ET"] == pytest.approx(1.0)


@pytest.mark.skipif(
    _PANOPTICA_MISSING,
    reason="panoptica is not installed in this venv (see requirements-analysis.txt); "
    "run from .venv-analysis to exercise lesion-wise scoring",
)
def test_per_case_replay_lesionwise_additive(tmp_path):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    logits, label = _perfect_case()
    _write_case(prep_dir, "case_001", label)
    _write_logits(eval_dir, "case_001", logits)

    df_off = replay.per_case_replay(eval_dir, prep_dir, postprocess_cfg=_RAW_PP_CFG)
    df_on = replay.per_case_replay(eval_dir, prep_dir, postprocess_cfg=_RAW_PP_CFG, lesionwise={})

    shared_cols = list(df_off.columns)
    pd.testing.assert_frame_equal(df_on[shared_cols], df_off)

    lw_cols = [c for c in df_on.columns if c.startswith("lw")]
    assert lw_cols
    assert all(col not in df_off.columns for col in lw_cols)


def test_replay_module_imports_without_panoptica():
    """Proves the lazy-import discipline documented in `replay_case`: the
    module itself must never pull `lesionwise_case_metrics` (and therefore
    never `panoptica`) into its own namespace just by being imported."""
    import neurovision.analysis.replay as replay_module

    assert "lesionwise_case_metrics" not in vars(replay_module)


def test_require_panoptica_is_public_and_importable():
    """`require_panoptica` must be reachable from the public `neurovision.metrics`
    package, not just as a private helper inside `lesionwise.py`. Not called
    here -- calling it would raise in a venv without panoptica."""
    from neurovision.metrics import require_panoptica

    assert callable(require_panoptica)
