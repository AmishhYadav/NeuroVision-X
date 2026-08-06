"""Tests for scripts/calibrate.py's pure-ish, testable helper functions.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, following the exact same pattern as tests/test_evaluate_script.py.

Every case here is tiny (16^3 volumes, a handful of cases) and synthetic --
never real BraTS data, never a real model. `scripts/calibrate.py` itself
loads no checkpoint and runs no model, so these tests only ever write
`.npy` files and CSVs under `tmp_path` and call the script's helpers
directly.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import zlib
from pathlib import Path
from types import ModuleType

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"
_spec = importlib.util.spec_from_file_location("calibrate_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calibrate_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_script"] = calibrate_script
_spec.loader.exec_module(calibrate_script)

resolve_eval_dirs = calibrate_script.resolve_eval_dirs
resolve_source = calibrate_script.resolve_source
resolve_mask_mode = calibrate_script.resolve_mask_mode
load_case = calibrate_script.load_case
load_image = calibrate_script.load_image
build_risk_coverage = calibrate_script.build_risk_coverage
run_calibration = calibrate_script.run_calibration

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

SHAPE: tuple[int, int, int] = (16, 16, 16)
REGIONS = ("ET", "TC", "WT")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _build_synthetic_label(shape: tuple[int, int, int]) -> np.ndarray:
    """Nested ET-subset-of-TC-subset-of-WT spheres, same recipe as test_evaluate_script.py."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[dist < min_edge * 0.35] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.22] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.10] = 3  # ET, innermost
    return label


def _region_indicator(label: np.ndarray) -> np.ndarray:
    """(3, D, H, W) float32 array, channel order (ET, TC, WT), matching classes_to_regions."""
    et = label == 3
    tc = et | (label == 1)
    wt = tc | (label == 2)
    return np.stack([et, tc, wt], axis=0).astype(np.float32)


def _synthetic_logits(label: np.ndarray, seed: int) -> np.ndarray:
    """Plausible logits: strongly positive inside each region, negative outside, plus noise."""
    rng = np.random.default_rng(seed)
    region = _region_indicator(label)
    logits = region * 8.0 - 4.0 + rng.normal(scale=0.5, size=region.shape)
    return logits.astype(np.float32)


def _build_synthetic_image(
    shape: tuple[int, int, int], case_id: str, n_channels: int = 4
) -> np.ndarray:
    """A preprocessed-looking (C, D, H, W) MRI volume: negative interior, exact-zero corner.

    Mirrors real preprocessing (neurovision.data.preprocessing.normalize_nonzero):
    z-scored brain tissue is routinely NEGATIVE, and exact zero marks air, not a
    low-intensity voxel. The zeroed corner is the trap `brain_mask` exists to get
    right -- `image > 0` would select nothing at all from this array.
    """
    # Deterministic seed from case_id, independent of Python's (possibly
    # randomized) string hash -- so the same case_id always yields the same array.
    seed = zlib.crc32(case_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    image = (rng.normal(loc=-0.5, scale=0.3, size=(n_channels, *shape))).astype(np.float32)
    d, h, w = shape
    image[:, : max(d // 4, 1), : max(h // 4, 1), : max(w // 4, 1)] = 0.0
    return image.astype(np.float16)


def _write_prep_case(prep_dir: Path, case_id: str, shape: tuple[int, int, int] = SHAPE) -> None:
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "label.npy", _build_synthetic_label(shape))
    np.save(case_dir / "image.npy", _build_synthetic_image(shape, case_id))


def _write_eval_case(
    eval_dir: Path,
    source: str,
    case_id: str,
    seed: int,
    shape: tuple[int, int, int] = SHAPE,
) -> np.ndarray:
    """Writes one `<eval_dir>/<source>/<case_id>.npy`, returning the label used to build it."""
    label = _build_synthetic_label(shape)
    logits = _synthetic_logits(label, seed)
    source_dir = eval_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    if source == "logits":
        np.save(source_dir / f"{case_id}.npy", logits.astype(np.float16))
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))
        np.save(source_dir / f"{case_id}.npy", probs.astype(np.float16))
    return label


def _write_split(
    tmp_path: Path,
    tag: str,
    case_ids: list[str],
    source: str = "logits",
    seed_offset: int = 0,
) -> tuple[Path, Path]:
    """Writes a full synthetic split: eval_dir/<source>/*.npy and prep_dir/<case>/label.npy.

    Returns (eval_dir, prep_dir).
    """
    eval_dir = tmp_path / f"eval_{tag}"
    prep_dir = tmp_path / "prep"  # shared preprocessed root across fit/apply splits
    for i, case_id in enumerate(case_ids):
        _write_eval_case(eval_dir, source, case_id, seed=seed_offset + i)
        if not (prep_dir / case_id / "label.npy").is_file():
            _write_prep_case(prep_dir, case_id)
    return eval_dir, prep_dir


def _make_cfg(
    fit_dir: Path | None,
    apply_dir: Path | None,
    prep_dir: Path,
    out_dir: Path,
    **calibration_overrides: object,
) -> OmegaConf:
    calibration = {
        "fit_dir": str(fit_dir) if fit_dir is not None else None,
        "apply_dir": str(apply_dir) if apply_dir is not None else None,
        "out_dir": str(out_dir),
        "source": "auto",
        "n_bins": 10,
        "threshold": 0.5,
        "mask_mode": "predicted",
        "per_channel": True,
        "fit_voxels_per_case": 200,
        "seed": 0,
        "temperature": None,
        "risk_coverage": {
            "enabled": True,
            "uncertainty_column": "mi_mean_fg_WT",
            "score_column": "dice_mean",
            "score_higher_is_better": True,
            "coverage_points": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        },
    }
    calibration.update(calibration_overrides)
    base = {
        "seed": 0,
        "data": {"preprocessing": {"out_dir": str(prep_dir)}},
        "calibration": calibration,
    }
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# 1-2. resolve_eval_dirs
# ---------------------------------------------------------------------------


def test_resolve_eval_dirs_same_directory_raises_naming_the_leak(tmp_path: Path):
    shared = tmp_path / "eval_same"
    shared.mkdir()
    cfg = _make_cfg(shared, shared, tmp_path / "prep", tmp_path / "out")

    with pytest.raises(ValueError, match="same"):
        resolve_eval_dirs(cfg)


def test_resolve_eval_dirs_raises_on_unset_keys(tmp_path: Path):
    cfg = _make_cfg(None, None, tmp_path / "prep", tmp_path / "out")

    with pytest.raises(ValueError) as excinfo:
        resolve_eval_dirs(cfg)

    message = str(excinfo.value)
    assert "calibration.fit_dir" in message
    assert "calibration.apply_dir" in message


def test_resolve_eval_dirs_raises_when_only_apply_dir_unset(tmp_path: Path):
    fit_dir = tmp_path / "eval_val"
    fit_dir.mkdir()
    cfg = _make_cfg(fit_dir, None, tmp_path / "prep", tmp_path / "out")

    with pytest.raises(ValueError, match="calibration.apply_dir"):
        resolve_eval_dirs(cfg)


def test_resolve_eval_dirs_raises_on_missing_directory(tmp_path: Path):
    fit_dir = tmp_path / "eval_val"
    fit_dir.mkdir()
    apply_dir = tmp_path / "does_not_exist"
    cfg = _make_cfg(fit_dir, apply_dir, tmp_path / "prep", tmp_path / "out")

    with pytest.raises(FileNotFoundError):
        resolve_eval_dirs(cfg)


# ---------------------------------------------------------------------------
# 3. resolve_source
# ---------------------------------------------------------------------------


def test_resolve_source_auto_prefers_logits_over_probabilities(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    (eval_dir / "logits").mkdir(parents=True)
    (eval_dir / "probabilities").mkdir(parents=True)
    np.save(eval_dir / "logits" / "c0.npy", np.zeros((3, 4, 4, 4), dtype=np.float16))
    np.save(eval_dir / "probabilities" / "c0.npy", np.zeros((3, 4, 4, 4), dtype=np.float16))

    assert resolve_source(eval_dir, "auto") == "logits"


def test_resolve_source_auto_falls_back_to_probabilities(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    (eval_dir / "probabilities").mkdir(parents=True)
    np.save(eval_dir / "probabilities" / "c0.npy", np.zeros((3, 4, 4, 4), dtype=np.float16))

    assert resolve_source(eval_dir, "auto") == "probabilities"


def test_resolve_source_auto_raises_naming_both_paths_when_neither_exists(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_source(eval_dir, "auto")

    message = str(excinfo.value)
    assert "logits" in message
    assert "probabilities" in message


def test_resolve_source_explicit_logits_raises_rather_than_falling_back(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    (eval_dir / "probabilities").mkdir(parents=True)
    np.save(eval_dir / "probabilities" / "c0.npy", np.zeros((3, 4, 4, 4), dtype=np.float16))

    with pytest.raises(FileNotFoundError, match="logits"):
        resolve_source(eval_dir, "logits")


# ---------------------------------------------------------------------------
# 4. fit_dir with only probabilities -> refuse to fit, uncalibrated still written
# ---------------------------------------------------------------------------


def test_probabilities_only_fit_dir_produces_unconverged_temperature(tmp_path: Path):
    case_ids = ["c0", "c1", "c2"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="probabilities")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir)

    run_calibration(cfg)

    temp_payload = json.loads((out_dir / "temperature.json").read_text())
    assert temp_payload["converged"] is False
    assert "reason" in temp_payload and temp_payload["reason"]

    metrics = pd.read_csv(out_dir / "calibration_metrics.csv")
    assert set(metrics["variant"]) == {"uncalibrated"}
    assert set(metrics["mask"]) == {"predicted", "brain"}
    assert not metrics.empty


# ---------------------------------------------------------------------------
# 5. End-to-end happy path on logits
# ---------------------------------------------------------------------------


def test_end_to_end_logits_writes_expected_outputs(tmp_path: Path):
    case_ids = ["c0", "c1", "c2", "c3"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="logits")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir)

    written = run_calibration(cfg)

    assert (out_dir / "temperature.json").is_file()
    assert (out_dir / "calibration_metrics.csv").is_file()
    assert (out_dir / "per_case_calibration.csv").is_file()
    assert (out_dir / "calibration_config.yaml").is_file()

    metrics = pd.read_csv(out_dir / "calibration_metrics.csv")
    assert set(metrics["variant"]) == {"uncalibrated", "temperature_scaled"}
    assert set(metrics["mask"]) == {"predicted", "brain"}

    per_case = pd.read_csv(out_dir / "per_case_calibration.csv")
    assert set(per_case["variant"]) == {"uncalibrated", "temperature_scaled"}
    assert set(per_case["mask"]) == {"predicted", "brain"}
    assert set(per_case["case_id"]) == set(case_ids)

    for mask_name in ("predicted", "brain"):
        for variant in ("uncalibrated", "temperature_scaled"):
            for region in REGIONS:
                path = out_dir / f"reliability_{mask_name}_{variant}_{region}.csv"
                assert path.is_file(), path
                df = pd.read_csv(path)
                assert {
                    "bin_lower",
                    "bin_upper",
                    "count",
                    "mean_prob",
                    "mean_label",
                    "gap",
                } <= set(df.columns)

    assert "reliability_csvs" in written
    assert len(written["reliability_csvs"]) == 12  # 2 masks x 2 variants x 3 regions

    temp_payload = json.loads((out_dir / "temperature.json").read_text())
    assert temp_payload["converged"] is True
    assert temp_payload["mask_mode"] == "predicted"
    assert "circular_do_not_report" not in temp_payload
    assert set(temp_payload["temperature"].keys()) == set(REGIONS)


# ---------------------------------------------------------------------------
# 6. Threshold=0.5 invariant: same voxel counts uncalibrated vs temperature-scaled
# ---------------------------------------------------------------------------


def test_temperature_scaling_leaves_pooled_voxel_counts_unchanged_at_threshold_half(
    tmp_path: Path,
):
    case_ids = ["c0", "c1", "c2"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="logits")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir, threshold=0.5)

    run_calibration(cfg)

    for mask_name in ("predicted", "brain"):
        for region in REGIONS:
            uncalibrated_count = pd.read_csv(
                out_dir / f"reliability_{mask_name}_uncalibrated_{region}.csv"
            )["count"].sum()
            scaled_count = pd.read_csv(
                out_dir / f"reliability_{mask_name}_temperature_scaled_{region}.csv"
            )["count"].sum()
            assert uncalibrated_count == scaled_count, (mask_name, region)
            # sanity: the mask actually selected something
            assert uncalibrated_count > 0, (mask_name, region)


# ---------------------------------------------------------------------------
# 7. Shape mismatch -> load_case names both shapes
# ---------------------------------------------------------------------------


def test_load_case_raises_on_shape_mismatch_naming_both_shapes(tmp_path: Path):
    eval_dir = tmp_path / "eval"
    prep_dir = tmp_path / "prep"
    case_id = "mismatched"

    # Prediction at a DIFFERENT spatial shape than the label -- the trap
    # this guard exists to catch (e.g. predictions/ in original geometry
    # pointed at instead of logits/ / probabilities/ in cropped geometry).
    bigger_shape = (20, 20, 20)
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True)
    np.save(
        logits_dir / f"{case_id}.npy",
        np.zeros((3, *bigger_shape), dtype=np.float16),
    )
    _write_prep_case(prep_dir, case_id, shape=SHAPE)

    with pytest.raises(ValueError) as excinfo:
        load_case(eval_dir, "logits", prep_dir, case_id)

    message = str(excinfo.value)
    assert str((3, *bigger_shape)) in message
    assert str((3, *SHAPE)) in message


# ---------------------------------------------------------------------------
# 8. build_risk_coverage returns None rather than crashing
# ---------------------------------------------------------------------------


def test_build_risk_coverage_none_when_uncertainty_summary_missing(tmp_path: Path):
    apply_dir = tmp_path / "eval_test"
    apply_dir.mkdir()
    pd.DataFrame({"dice_mean": [0.5, 0.6]}, index=pd.Index(["c0", "c1"], name="case_id")).to_csv(
        apply_dir / "per_case_metrics.csv"
    )

    cfg = _make_cfg(tmp_path / "unused_fit", apply_dir, tmp_path / "prep", tmp_path / "out")

    assert build_risk_coverage(cfg, apply_dir) is None


def test_build_risk_coverage_none_when_configured_column_missing(tmp_path: Path):
    apply_dir = tmp_path / "eval_test"
    apply_dir.mkdir()
    pd.DataFrame(
        {"some_other_column": [0.1, 0.2]}, index=pd.Index(["c0", "c1"], name="case_id")
    ).to_csv(apply_dir / "uncertainty_summary.csv")
    pd.DataFrame({"dice_mean": [0.5, 0.6]}, index=pd.Index(["c0", "c1"], name="case_id")).to_csv(
        apply_dir / "per_case_metrics.csv"
    )

    cfg = _make_cfg(tmp_path / "unused_fit", apply_dir, tmp_path / "prep", tmp_path / "out")

    assert build_risk_coverage(cfg, apply_dir) is None


def test_build_risk_coverage_succeeds_with_matching_columns(tmp_path: Path):
    apply_dir = tmp_path / "eval_test"
    apply_dir.mkdir()
    case_ids = [f"c{i}" for i in range(6)]
    pd.DataFrame(
        {"mi_mean_fg_WT": [0.1, 0.2, 0.15, 0.3, 0.05, 0.4]},
        index=pd.Index(case_ids, name="case_id"),
    ).to_csv(apply_dir / "uncertainty_summary.csv")
    pd.DataFrame(
        {"dice_mean": [0.9, 0.8, 0.85, 0.7, 0.95, 0.6]},
        index=pd.Index(case_ids, name="case_id"),
    ).to_csv(apply_dir / "per_case_metrics.csv")

    cfg = _make_cfg(tmp_path / "unused_fit", apply_dir, tmp_path / "prep", tmp_path / "out")

    result = build_risk_coverage(cfg, apply_dir)
    assert result is not None
    assert "risk_coverage" in result
    assert "referral_table" in result
    assert "uncertainty_correlation" in result
    assert result["uncertainty_correlation"]["n_cases"] == len(case_ids)

    # The oracle ceiling and the random null must ship in the same table as the
    # model curve. Without them a risk-coverage figure cannot be read at all --
    # a model curve hugging the random line is the reportable negative result,
    # and nothing else in the saved output would reveal it.
    curve = result["risk_coverage"]
    assert {"oracle_performance", "random_performance"}.issubset(curve.columns)
    assert len(curve) == len(case_ids)
    # Retaining every case is the same set regardless of ranking, so all three
    # must agree at full coverage -- an end-to-end check on the alignment.
    last = curve.iloc[-1]
    assert last["performance"] == pytest.approx(last["oracle_performance"])
    assert last["performance"] == pytest.approx(last["random_performance"])
    # The oracle can never be beaten by a real ranking, at any coverage.
    assert (curve["oracle_performance"] >= curve["performance"] - 1e-9).all()


# ---------------------------------------------------------------------------
# 9. configs/config.yaml still composes with the new calibration group
# ---------------------------------------------------------------------------


def test_config_composes_with_calibration_group(tmp_path: Path):
    overrides = [f"data.root_dir={tmp_path}"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    assert "calibration" in cfg
    assert cfg.calibration.fit_dir is None
    assert cfg.calibration.apply_dir is None
    # threshold must interpolate cleanly against inference.postprocess.threshold
    assert cfg.calibration.threshold == cfg.inference.postprocess.threshold
    # predicted is the label-free, non-circular default -- see
    # union_foreground_mask's docstring for why union_gt must not be the default.
    assert cfg.calibration.mask_mode == "predicted"


# ---------------------------------------------------------------------------
# mask_mode: validation, the circular union_gt diagnostic path, and the
# always-both-predicted-and-brain reporting guarantee.
# ---------------------------------------------------------------------------


def test_resolve_mask_mode_raises_on_unknown_value_listing_allowed_ones():
    with pytest.raises(ValueError) as excinfo:
        resolve_mask_mode("not_a_real_mode")

    message = str(excinfo.value)
    assert "predicted" in message
    assert "brain" in message
    assert "union_gt" in message


def test_resolve_mask_mode_accepts_every_valid_value():
    for mode in ("predicted", "brain", "union_gt"):
        assert resolve_mask_mode(mode) == mode


def test_load_image_reads_preprocessed_channel_first_volume(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    _write_prep_case(prep_dir, "c0")

    image = load_image(prep_dir, "c0")

    assert image.shape == (4, *SHAPE)
    assert image.dtype == torch.float32


def test_load_image_raises_when_missing(tmp_path: Path):
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        load_image(prep_dir, "does_not_exist")


def test_mask_mode_union_gt_marks_temperature_json_circular_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    case_ids = ["c0", "c1", "c2"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="logits")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir, mask_mode="union_gt")

    with caplog.at_level(logging.WARNING):
        run_calibration(cfg)

    assert any(
        "union_gt" in record.message and "CIRCULAR" in record.message for record in caplog.records
    )

    temp_payload = json.loads((out_dir / "temperature.json").read_text())
    assert temp_payload["mask_mode"] == "union_gt"
    assert temp_payload["circular_do_not_report"] is True

    # The REPORTED metrics are still always predicted + brain, never union_gt --
    # a circular fit does not make the reported table circular too.
    metrics = pd.read_csv(out_dir / "calibration_metrics.csv")
    assert set(metrics["mask"]) == {"predicted", "brain"}


def test_mask_mode_predicted_does_not_mark_temperature_json_circular(tmp_path: Path):
    case_ids = ["c0", "c1", "c2"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="logits")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir, mask_mode="predicted")

    run_calibration(cfg)

    temp_payload = json.loads((out_dir / "temperature.json").read_text())
    assert temp_payload["mask_mode"] == "predicted"
    assert "circular_do_not_report" not in temp_payload


def test_mask_mode_brain_fits_successfully(tmp_path: Path):
    case_ids = ["c0", "c1", "c2"]
    fit_dir, prep_dir = _write_split(tmp_path, "val", case_ids, source="logits")
    apply_dir, _ = _write_split(tmp_path, "test", case_ids, source="logits", seed_offset=100)

    out_dir = tmp_path / "calib_out"
    cfg = _make_cfg(fit_dir, apply_dir, prep_dir, out_dir, mask_mode="brain")

    run_calibration(cfg)

    temp_payload = json.loads((out_dir / "temperature.json").read_text())
    assert temp_payload["mask_mode"] == "brain"
    assert temp_payload["converged"] is True


# ---------------------------------------------------------------------------
# Ragged per-channel sampling for the temperature fit
# ---------------------------------------------------------------------------


def test_subsample_masked_logits_is_ragged_not_truncated_to_the_smallest_channel():
    """Each channel keeps its own sample count, so ET's small mask cannot starve WT.

    Truncating every channel to the minimum would let the smallest region
    silently dictate how many voxels the whole temperature fit sees -- ET's
    reporting mask is routinely a fraction of WT's -- and would be invisible
    once the fitted temperature is in a table.
    """
    logits = torch.zeros(3, 4, 4, 4)
    label = torch.zeros(3, 4, 4, 4)
    mask = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    mask[0].reshape(-1)[:2] = True  # ET: 2 voxels
    mask[1].reshape(-1)[:8] = True  # TC: 8
    mask[2].reshape(-1)[:32] = True  # WT: 32

    generator = torch.Generator().manual_seed(0)
    sampled_logits, sampled_labels = calibrate_script.subsample_masked_logits(
        logits, label, mask, n_samples=1000, generator=generator
    )

    assert [t.numel() for t in sampled_logits] == [2, 8, 32]
    assert [t.numel() for t in sampled_labels] == [2, 8, 32]


def test_empty_et_case_still_contributes_its_tc_and_wt_voxels_to_the_fit(tmp_path: Path):
    """A case with no enhancing tumour must not drop out of the temperature fit entirely.

    2.6% of BraTS 2021 cases have zero enhancing tumour (measured). Under a
    rectangular (N, C) fit those cases contribute nothing at all -- including
    their perfectly usable TC and WT voxels -- because the ET channel's zero
    count would truncate the other two.
    """
    logits = torch.zeros(3, 4, 4, 4)
    label = torch.zeros(3, 4, 4, 4)
    mask = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    mask[1].reshape(-1)[:6] = True
    mask[2].reshape(-1)[:10] = True

    generator = torch.Generator().manual_seed(0)
    sampled_logits, _ = calibrate_script.subsample_masked_logits(
        logits, label, mask, n_samples=1000, generator=generator
    )

    assert sampled_logits[0].numel() == 0
    assert sampled_logits[1].numel() == 6
    assert sampled_logits[2].numel() == 10


def test_fit_per_channel_recovers_a_known_temperature_per_channel():
    """Each channel is fit independently, so a per-channel ground truth is recoverable.

    Channel 1's logits are the same evidence as channel 0's, scaled by 2 --
    so its fitted temperature must be ~2x channel 0's. A joint fit forced
    through one shared T could not express that, and a truncating
    rectangular fit would not even see the same number of voxels per channel.
    """
    generator = torch.Generator().manual_seed(0)
    n = 20_000
    base = torch.randn(n, generator=generator) * 2.0
    labels = (torch.rand(n, generator=generator) < torch.sigmoid(base)).float()

    # Channel 2 deliberately gets FEWER samples than the others -- the case
    # the ragged path exists for.
    channel_logits = [[base], [base * 2.0], [base[:500] * 0.5]]
    channel_labels = [[labels], [labels], [labels[:500]]]

    result = calibrate_script._fit_per_channel(channel_logits, channel_labels, per_channel=True)

    assert result.converged
    assert result.temperature.shape == (3,)
    t0, t1, t2 = (float(v) for v in result.temperature)
    assert t0 == pytest.approx(1.0, abs=0.15)
    assert t1 == pytest.approx(2.0, abs=0.3)
    assert t2 == pytest.approx(0.5, abs=0.2)


def test_fit_per_channel_channel_with_no_voxels_anywhere_stays_at_identity():
    """T = 1.0 is the identity -- never a fitted-looking number for an unfit channel."""
    generator = torch.Generator().manual_seed(0)
    n = 2_000
    base = torch.randn(n, generator=generator)
    labels = (torch.rand(n, generator=generator) < torch.sigmoid(base)).float()

    result = calibrate_script._fit_per_channel(
        [[], [base], [base]], [[], [labels], [labels]], per_channel=True
    )

    assert float(result.temperature[0]) == 1.0
    assert result.temperature.shape == (3,)
