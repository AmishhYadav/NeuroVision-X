"""Tests for scripts/severity_ladder_analysis.py.

CPU only, a tiny synthetic shard tree under `tmp_path` (10 cases, 73-point threshold grid, 3
models, 2 kinds, levels 0/2/3/4) and tiny bootstrap counts. Never touches real BraTS output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

import neurovision.analysis.severity_ladder_analysis as sla
from neurovision.analysis.ood import OODModel
from tests.script_loader import load_script

script = load_script("severity_ladder_analysis")

N = 10
GRID = np.round(np.linspace(0.01, 0.99, 73), 12)
TAU = float(GRID[0])  # at the lowest grid point the miss rate is exactly fn[:, 0] / gt
GT = 1000
CASES = [f"BraTS2021_{i:05d}" for i in range(N)]
FEATURES = ["f1", "f2", "f3"]
# model -> kinds it has (the baseline is absent from `bias`, as after Amendment 3)
MODEL_KINDS = {
    "neurovision": ["rician", "bias"],
    "baseline_unet3d": ["rician"],
    "nnunet_gatea": ["rician", "bias"],
}
SEVERITY = {"rician": {2: 0.1, 3: 0.2, 4: 0.3}, "bias": {2: 0.2, 3: 0.3, 4: 0.5}}


def _write_shard(root: Path, model: str, kind: str, level: int, miss: np.ndarray) -> Path:
    d = root / model / kind / f"level{level}"
    d.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {"thresholds": GRID}
    base = np.round(miss * GT).astype(np.int64)
    fn = np.minimum(base[:, None] + np.arange(73)[None, :], GT)  # misses rise with threshold
    for region in ("ET", "TC", "WT"):
        arrays[f"{region}__case_ids"] = np.asarray(CASES)
        arrays[f"{region}__gt_voxels"] = np.full(N, GT, dtype=np.int64)
        arrays[f"{region}__fn_voxels"] = fn.astype(np.int64)
        arrays[f"{region}__mask_voxels"] = (GT - fn + 5).astype(np.int64)
    np.savez(d / "curves.npz", **arrays)
    dice = np.clip(0.95 - 3 * miss, 0, 1)
    pd.DataFrame(
        {
            "case_id": CASES,
            "model": model,
            "dice_ET": dice,
            "dice_TC": dice,
            "dice_WT": dice,
            "predicted_dice_WT": np.linspace(0.4, 0.95, N),
            "predicted_dice_TC": np.linspace(0.4, 0.95, N),
            "conformal_band_WT": 0.1,
            "conformal_band_TC": 0.1,
        }
    ).to_csv(d / "per_case.csv", index=False)
    rng = np.random.default_rng(level)
    feats = pd.DataFrame(rng.normal(size=(N, len(FEATURES))), columns=FEATURES)
    feats.insert(0, "case_id", CASES)
    feats.to_csv(d / "ood_features.csv", index=False)
    severity = 0.0 if kind == "none" else SEVERITY[kind][level]
    (d / "shard_meta.json").write_text(json.dumps({"severity": severity}))
    return d


def _build_tree(root: Path) -> None:
    rng = np.random.default_rng(0)
    for mi, (model, kinds) in enumerate(MODEL_KINDS.items()):
        base = 0.05 + 0.002 * np.arange(N)
        _write_shard(root, model, "none", 0, base)
        for kind in kinds:
            for lvl in (2, 3, 4):
                miss = np.clip(base + 0.02 * lvl + rng.normal(0, 0.003, N) + 0.001 * mi, 0, 1)
                _write_shard(root, model, kind, lvl, miss)


def _write_frozen(tmp: Path) -> dict[str, Path]:
    """Fake v1 fit.json (per model), v2 fits.csv, OOD model.json/thresholds.json, gate json."""
    fit = {
        f"{r}__alpha_{a}": {"threshold": TAU, "feasible": True}
        for r in ("WT", "TC")
        for a in (0.05, 0.1, 0.2)
    }
    paths: dict[str, Path] = {}
    for model in MODEL_KINDS:
        p = tmp / f"fit_{model}.json"
        p.write_text(json.dumps(fit))
        paths[model] = p
    rows = [
        {
            "model": m, "region": r, "alpha": a, "empty": "zero" if r == "ET" else "exclude",
            "grid": "extended", "guarantee": g, "feasible": True, "threshold": TAU,
        }
        for m in MODEL_KINDS
        for r in ("WT", "TC", "ET")
        for a in (0.05, 0.1, 0.2)
        for g in ("crc", "rcps")
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp / "fits.csv", index=False)
    ood_dir = tmp / "ood"
    ood_dir.mkdir()
    model = OODModel(
        feature_names=tuple(FEATURES),
        center=np.zeros(3),
        scale=np.ones(3),
        precision=np.eye(3),
        shrinkage=0.1,
        n_fit=100,
    )
    (ood_dir / "model.json").write_text(json.dumps(model.to_dict()))
    (ood_dir / "thresholds.json").write_text(json.dumps({"caution_cut": 1.5, "refuse_cut": 2.5}))
    return paths


def _cfg(tmp: Path, fits: dict[str, Path], **overrides: object) -> DictConfig:
    sl = {
        "out_dir": str(tmp / "out"),
        "shard_roots": [str(tmp / "shards")],
        "require_done": False,
        "expected_cases": N,
        "primary_alpha": 0.10,
        "regions": ["WT", "TC"],
        "alphas": [0.05, 0.10, 0.20],
        "n_boot": 50,
        "seed": 42,
        "v1_fits": {m: str(p) for m, p in fits.items()},
        "v2_fits": {m: {"csv": str(tmp / "fits.csv"), "model_key": m} for m in MODEL_KINDS},
        "v2_grid": "extended",
        "v2_empty_by_region": {"ET": "zero", "TC": "exclude", "WT": "exclude"},
        "gate_regions": ["WT", "TC"],
        "stand_in_age_years": 40,
        "ood_dir": str(tmp / "ood"),
    }
    sl.update(overrides)
    gate = {
        "enabled_signals": ["input_qc", "predicted_dice"],
        "regions": ["WT", "TC"],
        "thresholds": {
            "predicted_dice": {"WT": [0.5, 0.7], "TC": [0.5, 0.7]},
            "conformal_band": {"WT": [0.3, 0.6], "TC": [0.3, 0.6]},
            "ood_score": [1.0, 2.0],
            "calibration_n": 100,
            "caution_quantile": 0.9,
            "refuse_quantile": 0.99,
        },
    }
    return OmegaConf.create(
        {"seed": 42, "analysis": {"severity_ladder_analysis": sl}, "clinical": {"gatekeeper": gate}}
    )


@pytest.fixture()
def setup(tmp_path: Path) -> tuple[Path, DictConfig]:
    _build_tree(tmp_path / "shards")
    fits = _write_frozen(tmp_path)
    return tmp_path, _cfg(tmp_path, fits)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_passes_and_reports_table(setup: tuple[Path, DictConfig]) -> None:
    tmp, _ = setup
    shards = sla.discover_shards([tmp / "shards"], require_done=False)
    table = script.validate_shards(shards, N)
    assert set(table["model"]) == set(MODEL_KINDS)
    assert len(table) == 3 + 5  # (model, none) x3 plus five (model, kind) pairs


def test_validate_wrong_case_count(setup: tuple[Path, DictConfig]) -> None:
    tmp, _ = setup
    shards = sla.discover_shards([tmp / "shards"], require_done=False)
    with pytest.raises(ValueError, match="expected 9"):
        script.validate_shards(shards, N - 1)


def test_validate_duplicate_case_id(setup: tuple[Path, DictConfig]) -> None:
    tmp, _ = setup
    p = tmp / "shards" / "neurovision" / "rician" / "level2" / "per_case.csv"
    df = pd.read_csv(p)
    df.loc[1, "case_id"] = df.loc[0, "case_id"]  # same row count, one id twice
    df.to_csv(p, index=False)
    shards = sla.discover_shards([tmp / "shards"], require_done=False)
    with pytest.raises(ValueError, match="duplicate case_id"):
        script.validate_shards(shards, N)


def test_validate_missing_level0(setup: tuple[Path, DictConfig]) -> None:
    tmp, _ = setup
    import shutil

    shutil.rmtree(tmp / "shards" / "nnunet_gatea" / "none")
    shards = sla.discover_shards([tmp / "shards"], require_done=False)
    with pytest.raises(ValueError, match="nnunet_gatea: no none/level0"):
        script.validate_shards(shards, N)


# ---------------------------------------------------------------------------
# OOD columns
# ---------------------------------------------------------------------------


def _ood_model() -> OODModel:
    return OODModel(
        feature_names=tuple(FEATURES),
        center=np.zeros(3),
        scale=np.ones(3),
        precision=np.eye(3),
        shrinkage=0.1,
        n_fit=10,
    )


def test_check_ood_columns_exact_name_match() -> None:
    ok = pd.DataFrame({"case_id": ["a"], "f1": [0.0], "f2": [0.0], "f3": [0.0]})
    script.check_ood_columns(_ood_model(), ok, "ok")  # no raise
    with pytest.raises(ValueError, match="unexpected \\['f3x'\\]"):  # f3x is NOT f3
        script.check_ood_columns(_ood_model(), ok.rename(columns={"f3": "f3x"}), "bad")
    with pytest.raises(ValueError, match="missing \\['f3'\\]"):
        script.check_ood_columns(_ood_model(), ok.drop(columns="f3"), "bad")


def test_run_ood_raises_on_shard_with_extra_column(setup: tuple[Path, DictConfig]) -> None:
    tmp, _ = setup
    p = tmp / "shards" / "neurovision" / "none" / "level0" / "ood_features.csv"
    df = pd.read_csv(p)
    df["extra"] = 1.0
    df.to_csv(p, index=False)
    shards = sla.discover_shards([tmp / "shards"], require_done=False)
    model, cuts = script.load_ood(tmp / "ood")
    with pytest.raises(ValueError, match="OOD feature columns differ"):
        script.run_ood(model, cuts, shards)


def test_load_ood_rejects_inverted_cuts(tmp_path: Path) -> None:
    _write_frozen(tmp_path)
    (tmp_path / "ood" / "thresholds.json").write_text(
        json.dumps({"caution_cut": 3.0, "refuse_cut": 1.0})
    )
    with pytest.raises(ValueError, match="caution_cut"):
        script.load_ood(tmp_path / "ood")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_end_to_end_outputs(setup: tuple[Path, DictConfig]) -> None:
    tmp, cfg = setup
    paths = script.run_analysis(cfg, tmp)
    out = tmp / "out"
    for name in (
        "primary.csv", "kind_verdicts.csv", "secondary.csv", "descriptive.csv",
        "gate_per_case.csv", "ood_per_case.csv", "run_meta.json", "summary.md",
        "dose_response.png", "dose_response.pdf", "gate_vs_level.png", "gate_vs_level.pdf",
    ):  # fmt: skip
        assert (out / name).is_file(), name
    assert set(paths) >= {"primary", "descriptive", "run_meta", "summary"}

    # One primary row per present (kind, model, region): rician 3 models + bias 2, x 2 regions.
    primary = pd.read_csv(out / "primary.csv")
    assert len(primary) == (3 + 2) * 2
    assert set(primary["verdict"]) <= {"GRADED", "NOT GRADED"}
    verdicts = pd.read_csv(out / "kind_verdicts.csv")
    assert set(verdicts["kind"]) == {"rician", "bias"}

    # Descriptive carries v1 and v2 method labels; dice rows appear once (not duplicated by v2).
    desc = pd.read_csv(out / "descriptive.csv")
    methods = set(desc.loc[desc["quantity"] == "realised_risk", "method"])
    assert methods == {"crc_v1", "crc_v2", "rcps_v2"}
    n_dice = (desc["quantity"] == "dice").sum()
    assert n_dice == len(sla.discover_shards([tmp / "shards"], require_done=False)) * 3
    assert {"refusal_rate", "silent_failure_rate", "ood_flag_rate"} <= set(desc["quantity"])
    assert "ET" in set(desc.loc[desc["method"] == "crc_v2", "region"])  # v2 includes ET

    gate = pd.read_csv(out / "gate_per_case.csv")
    assert list(gate.columns) == [
        "model", "kind", "level", "case_id", "decision", "accepted", "usable",
        "silent_failure", "gate_label", "refusing_signals",
    ]  # fmt: skip
    assert set(gate.loc[gate["model"] == "neurovision", "gate_label"]) == {"deployed"}
    assert set(gate.loc[gate["model"] == "nnunet_gatea", "gate_label"]) == {
        "transferred, not refitted"
    }
    assert gate["accepted"].nunique() == 2  # synthetic predicted_dice spans the cut: some refused

    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["n_shards"] == 3 + 5 * 3
    hashes = meta["input_sha256"]
    assert any(k.endswith("fits.csv") for k in hashes)
    assert any(k.endswith("model.json") for k in hashes)
    assert sum(k.endswith("curves.npz") for k in hashes) == meta["n_shards"]
    assert all(len(v) == 64 for v in hashes.values())
    assert meta["library_constants"]["N_BOOT"] == 10_000
    assert "nothing_fitted" in meta and meta["taus_v1"]["neurovision"]
    assert "timestamp_utc" in meta and "git_head" in meta

    summary = (out / "summary.md").read_text()
    assert "## Primary" in summary and "## Kind verdicts" in summary


def test_missing_primary_tau_fails_loudly(setup: tuple[Path, DictConfig]) -> None:
    tmp, cfg = setup
    bad = json.loads(Path(cfg.analysis.severity_ladder_analysis.v1_fits.neurovision).read_text())
    bad["WT__alpha_0.1"] = {"threshold": None, "feasible": False}
    Path(cfg.analysis.severity_ladder_analysis.v1_fits.neurovision).write_text(json.dumps(bad))
    with pytest.raises(KeyError, match="no feasible v1 tau"):
        script.run_analysis(cfg, tmp)


def test_deterministic_given_seed(setup: tuple[Path, DictConfig]) -> None:
    tmp, cfg = setup
    script.run_analysis(cfg, tmp)
    first = pd.read_csv(tmp / "out" / "primary.csv")
    script.run_analysis(cfg, tmp)
    pd.testing.assert_frame_equal(first, pd.read_csv(tmp / "out" / "primary.csv"))
