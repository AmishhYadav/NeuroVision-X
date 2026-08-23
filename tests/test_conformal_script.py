"""Tests for scripts/conformal.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package
import, following the exact same pattern as tests/test_calibrate_script.py /
tests/test_replay_logits.py.

Everything here is synthetic, tiny (8^3 volumes), and CPU-only. All splits
share one preprocessed root (`prep_dir`, case ids tag-prefixed so they never
collide), mirroring the common in-project layout where val/test/external
cohorts read from one preprocessed tree -- this also exercises
`conformal.apply_prep_dirs`' default-fill behaviour "for free" in every test
that does not explicitly set it.
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

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "conformal.py"
_spec = importlib.util.spec_from_file_location("conformal_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
conformal_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["conformal_script"] = conformal_script
_spec.loader.exec_module(conformal_script)

extract_curves = conformal_script.extract_curves
resolve_dirs = conformal_script.resolve_dirs
resolve_prep_dirs = conformal_script.resolve_prep_dirs
run_conformal = conformal_script.run_conformal

SHAPE = (8, 8, 8)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _build_label(shape: tuple[int, int, int] = SHAPE) -> np.ndarray:
    """A fixed nested ET-subset-of-TC-subset-of-WT sphere label, same recipe used elsewhere
    in this project's tests (test_calibrate_script.py, test_replay_logits.py)."""
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
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
    """Confident, mostly-correct logits: strongly positive inside each region, negative
    outside, plus a little noise. sigmoid(+/-5) is ~0.993 / ~0.007."""
    rng = np.random.default_rng(seed)
    region = _region_indicator(label)
    logits = region * 10.0 - 5.0 + rng.normal(scale=0.3, size=region.shape)
    return logits.astype(np.float32)


def _catastrophic_logits(label: np.ndarray) -> np.ndarray:
    """Deep negative everywhere, regardless of ground truth.

    sigmoid(-50) ~ 1.9e-22, below DEFAULT_THRESHOLDS' smallest grid point
    (1e-4) -- so the predicted mask never includes a single ground-truth
    voxel at ANY threshold in the grid, and the miss rate is exactly 1.0
    everywhere. Used to force a conformal fit that is INFEASIBLE at any
    ordinary alpha, a registered outcome this script must report, not raise.
    """
    region = _region_indicator(label)
    return np.full_like(region, -50.0, dtype=np.float32)


def _write_split(
    tmp_path: Path,
    prep_dir: Path,
    tag: str,
    n_cases: int,
    *,
    catastrophic: bool = False,
    seed_offset: int = 0,
) -> Path:
    """Writes `eval_<tag>/logits/*.npy`, and `prep_dir/<case_id>/{label.npy,meta.json}`.

    Returns the eval_dir. Case ids are tag-prefixed so multiple splits can
    safely share one `prep_dir`.
    """
    eval_dir = tmp_path / f"eval_{tag}"
    logits_dir = eval_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_cases):
        case_id = f"{tag}_{i:03d}"
        label = _build_label()
        logits = (
            _catastrophic_logits(label) if catastrophic else _good_logits(label, seed_offset + i)
        )
        np.save(logits_dir / f"{case_id}.npy", logits.astype(np.float16))

        case_dir = prep_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "label.npy", label)
        write_json({"spacing": [1.0, 1.0, 1.0]}, case_dir / "meta.json")

    return eval_dir


def _make_cfg(
    calib_dir: Path | None,
    apply_dirs: list[Path],
    calib_prep_dir: Path,
    apply_prep_dirs: list[Path],
    out_dir: Path,
    *,
    alphas: tuple[float, ...] = (0.05, 0.10, 0.20),
    regions: tuple[str, ...] = ("WT", "TC"),
    check_consistency: bool = True,
    consistency_tol: float = 1e-6,
    seed: int = 0,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": seed,
            "conformal": {
                "calib_dir": str(calib_dir) if calib_dir is not None else None,
                "calib_prep_dir": str(calib_prep_dir),
                "apply_dirs": [str(p) for p in apply_dirs],
                "apply_prep_dirs": [str(p) for p in apply_prep_dirs],
                "out_dir": str(out_dir),
                "alphas": list(alphas),
                "regions": list(regions),
                "check_consistency": check_consistency,
                "consistency_tol": consistency_tol,
            },
        }
    )


# ---------------------------------------------------------------------------
# 1-2. extract_curves
# ---------------------------------------------------------------------------


def test_extract_writes_curves_and_manifest(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "calib", 3)
    out_dir = tmp_path / "conformal"
    thresholds = (0.01, 0.1, 0.5, 0.9)

    curves = extract_curves(
        eval_dir, prep_dir, out_dir, regions=["WT", "TC"], thresholds=thresholds
    )

    assert set(curves.keys()) == {"WT", "TC"}
    assert len(curves["WT"]) == 3
    assert curves["WT"][0].thresholds == thresholds

    curves_dir = out_dir / eval_dir.name
    assert (curves_dir / "curves.npz").is_file()
    manifest = json.loads((curves_dir / "curves_manifest.json").read_text())
    assert manifest["regions"] == ["WT", "TC"]
    assert manifest["n_cases"] == 3
    assert manifest["thresholds"] == [0.01, 0.1, 0.5, 0.9]
    assert set(manifest["gt_voxels"].keys()) == {"calib_000", "calib_001", "calib_002"}
    assert set(manifest["gt_voxels"]["calib_000"].keys()) == {"WT", "TC"}


def test_extract_reuses_cache_when_manifest_matches(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    prep_dir = tmp_path / "prep"
    eval_dir = _write_split(tmp_path, prep_dir, "calib", 2)
    out_dir = tmp_path / "conformal"
    thresholds = (0.1, 0.5)

    first = extract_curves(eval_dir, prep_dir, out_dir, regions=["WT"], thresholds=thresholds)
    curves_dir = out_dir / eval_dir.name
    manifest_before = json.loads((curves_dir / "curves_manifest.json").read_text())

    with caplog.at_level(logging.INFO):
        second = extract_curves(eval_dir, prep_dir, out_dir, regions=["WT"], thresholds=thresholds)
    assert "reusing cached curves" in caplog.text
    manifest_after = json.loads((curves_dir / "curves_manifest.json").read_text())
    assert manifest_after == manifest_before
    assert [c.case_id for c in second["WT"]] == [c.case_id for c in first["WT"]]
    assert [c.fn_voxels for c in second["WT"]] == [c.fn_voxels for c in first["WT"]]

    # A different threshold grid must NOT match the cached manifest -> re-extract.
    caplog.clear()
    different_thresholds = (0.2, 0.6, 0.8)
    with caplog.at_level(logging.INFO):
        third = extract_curves(
            eval_dir, prep_dir, out_dir, regions=["WT"], thresholds=different_thresholds
        )
    assert "re-extracting" in caplog.text
    assert third["WT"][0].thresholds == different_thresholds


# ---------------------------------------------------------------------------
# 3-4. fit / apply via the full pipeline
# ---------------------------------------------------------------------------


def test_fit_writes_fit_json_with_expected_keys(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 25)
    apply_dir = _write_split(tmp_path, prep_dir, "test", 5, seed_offset=1000)
    out_dir = tmp_path / "conformal"
    cfg = _make_cfg(
        calib_dir, [apply_dir], prep_dir, [], out_dir, alphas=(0.05, 0.2), regions=("WT", "TC")
    )

    run_conformal(cfg)

    fit_payload = json.loads((out_dir / "fit.json").read_text())
    assert len(fit_payload) == 2 * 2  # regions x alphas
    expected_keys = {
        "region",
        "alpha",
        "threshold",
        "feasible",
        "calibrated_risk",
        "min_achievable_risk",
        "n_calibration",
        "n_excluded_empty",
    }
    for entry in fit_payload.values():
        assert expected_keys <= set(entry.keys())


def test_apply_writes_realised_risk_and_inflation_csv(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 25)
    apply_dir = _write_split(tmp_path, prep_dir, "test", 4, seed_offset=2000)
    out_dir = tmp_path / "conformal"
    cfg = _make_cfg(
        calib_dir, [apply_dir], prep_dir, [], out_dir, alphas=(0.05, 0.2), regions=("WT", "TC")
    )

    run_conformal(cfg)

    risk_df = pd.read_csv(out_dir / "realised_risk.csv")
    inflation_df = pd.read_csv(out_dir / "inflation.csv")

    assert {
        "apply_dir",
        "region",
        "alpha",
        "threshold",
        "mean_miss_rate",
        "n",
        "n_excluded_empty",
    } <= set(risk_df.columns)
    # A good, well-separated 25-case calibration set is feasible at both alphas
    # for both regions -- so every (apply_dir x region x alpha) combo is applied.
    assert len(risk_df) == 1 * 2 * 2
    assert {
        "apply_dir",
        "region",
        "alpha",
        "mean_inflation",
        "median_inflation",
        "n",
        "n_skipped",
    } <= set(inflation_df.columns)
    assert len(inflation_df) == len(risk_df)


# ---------------------------------------------------------------------------
# 5. Guard 1 -- fit/apply separation
# ---------------------------------------------------------------------------


def test_raises_when_calib_dir_equals_apply_dir(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    shared = _write_split(tmp_path, prep_dir, "shared", 3)
    out_dir = tmp_path / "conformal"

    cfg = _make_cfg(shared, [shared], prep_dir, [], out_dir)
    with pytest.raises(ValueError, match="same"):
        run_conformal(cfg)

    # The "a/./b" form must be caught too -- Path.resolve() normalizes it to
    # the identical real directory.
    dotted_apply = shared.parent / "." / shared.name
    cfg_dotted = _make_cfg(shared, [dotted_apply], prep_dir, [], out_dir)
    with pytest.raises(ValueError, match="same"):
        run_conformal(cfg_dotted)


# ---------------------------------------------------------------------------
# 6. apply_prep_dirs length mismatch
# ---------------------------------------------------------------------------


def test_raises_on_apply_prep_dirs_length_mismatch(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path / "calib",
        [tmp_path / "test1", tmp_path / "test2"],
        tmp_path / "calib_prep",
        [tmp_path / "prep1"],
        tmp_path / "conformal",
    )
    with pytest.raises(ValueError, match="apply_prep_dirs"):
        resolve_prep_dirs(cfg)


# ---------------------------------------------------------------------------
# 7. Infeasible alpha -- reported, not raised
# ---------------------------------------------------------------------------


def test_infeasible_alpha_is_reported_not_raised(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 4, catastrophic=True)
    apply_dir = _write_split(tmp_path, prep_dir, "test", 3, catastrophic=True, seed_offset=50)
    out_dir = tmp_path / "conformal"
    cfg = _make_cfg(calib_dir, [apply_dir], prep_dir, [], out_dir, alphas=(0.05,), regions=("WT",))

    run_conformal(cfg)  # must NOT raise

    fit_payload = json.loads((out_dir / "fit.json").read_text())
    entry = next(iter(fit_payload.values()))
    assert entry["feasible"] is False
    assert entry["threshold"] is None
    assert entry["min_achievable_risk"] > 0.05
    assert (out_dir / "realised_risk.csv").is_file()
    assert (out_dir / "fit.json").is_file()


# ---------------------------------------------------------------------------
# 8-9. Guard 3 -- replay self-consistency
# ---------------------------------------------------------------------------


def test_consistency_check_raises_on_mismatch(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 5)
    apply_dir = _write_split(tmp_path, prep_dir, "test", 3, seed_offset=10)
    out_dir = tmp_path / "conformal"

    # A deliberately WRONG committed per_case_metrics.csv for calib_dir.
    case_ids = [f"calib_{i:03d}" for i in range(5)]
    replayed = per_case_replay(calib_dir, prep_dir, case_ids=case_ids)
    corrupted = replayed.copy()
    corrupted["dice_ET"] = (corrupted["dice_ET"] - 0.9).clip(lower=0.0)
    corrupted.to_csv(calib_dir / "per_case_metrics.csv")

    cfg = _make_cfg(calib_dir, [apply_dir], prep_dir, [], out_dir)

    with pytest.raises(ValueError, match="disagrees"):
        run_conformal(cfg)

    assert not (out_dir / "fit.json").exists()
    assert not (out_dir / "realised_risk.csv").exists()
    assert not (out_dir / "inflation.csv").exists()


def test_consistency_check_can_be_disabled(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 5)
    apply_dir = _write_split(tmp_path, prep_dir, "test", 3, seed_offset=20)
    out_dir = tmp_path / "conformal"

    case_ids = [f"calib_{i:03d}" for i in range(5)]
    replayed = per_case_replay(calib_dir, prep_dir, case_ids=case_ids)
    corrupted = replayed.copy()
    corrupted["dice_ET"] = (corrupted["dice_ET"] - 0.9).clip(lower=0.0)
    corrupted.to_csv(calib_dir / "per_case_metrics.csv")

    cfg = _make_cfg(calib_dir, [apply_dir], prep_dir, [], out_dir, check_consistency=False)

    run_conformal(cfg)  # must NOT raise despite the mismatched committed CSV

    assert (out_dir / "fit.json").is_file()


# ---------------------------------------------------------------------------
# 10. End-to-end smoke
# ---------------------------------------------------------------------------


def test_end_to_end_smoke(tmp_path: Path) -> None:
    prep_dir = tmp_path / "prep"
    calib_dir = _write_split(tmp_path, prep_dir, "calib", 25)
    apply_dir_test = _write_split(tmp_path, prep_dir, "test", 4, seed_offset=300)
    apply_dir_ssa = _write_split(tmp_path, prep_dir, "ssa", 3, seed_offset=400)
    out_dir = tmp_path / "conformal"

    cfg = _make_cfg(
        calib_dir,
        [apply_dir_test, apply_dir_ssa],
        prep_dir,
        [],
        out_dir,
        alphas=(0.05, 0.1, 0.2),
        regions=("WT", "TC"),
    )

    result = run_conformal(cfg)

    assert result["fit_json"].is_file()
    assert result["realised_risk_csv"].is_file()
    assert result["inflation_csv"].is_file()
    assert (out_dir / "conformal_config.yaml").is_file()

    risk_df = pd.read_csv(out_dir / "realised_risk.csv")
    # 2 apply_dirs x 2 regions x 3 alphas, all feasible for a good 25-case calibration set.
    assert len(risk_df) == 2 * 2 * 3
    assert set(risk_df["apply_dir"]) == {str(apply_dir_test), str(apply_dir_ssa)}

    # The extraction cache exists for calib and both apply dirs.
    assert (out_dir / calib_dir.name / "curves.npz").is_file()
    assert (out_dir / apply_dir_test.name / "curves.npz").is_file()
    assert (out_dir / apply_dir_ssa.name / "curves.npz").is_file()
