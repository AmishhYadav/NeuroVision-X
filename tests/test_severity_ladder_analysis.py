"""Tests for `neurovision.analysis.severity_ladder_analysis`.

CPU only, tiny synthetic shards under `tmp_path` (12 cases, 73-point grid). Expected numbers are
hand-computed in the tests, never read back from the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import neurovision.analysis.severity_ladder_analysis as sla
from neurovision.analysis.ood import OODModel
from neurovision.inference.gatekeeper import Thresholds
from neurovision.uncertainty.conformal import (
    CaseLossCurve,
    load_curves_npz,
    realised_risk_with,
)

N = 12
GRID = np.round(np.linspace(0.01, 0.99, 73), 12)
TAU = float(GRID[0])  # at the lowest grid point the miss rate is exactly fn[:, 0] / gt
GT = 1000
CASES = [f"BraTS2021_{i:05d}" for i in range(N)]
NBOOT = 200  # tiny; the registered 10,000 is only needed for the real run


def _curves_npz(path: Path, miss: dict[str, np.ndarray], gt: np.ndarray | None = None) -> None:
    """Writes a `curves.npz` whose miss rate at GRID[0] is `miss[region]` (rounded to 1/GT)."""
    gt = np.full(N, GT) if gt is None else gt
    arrays: dict[str, np.ndarray] = {"thresholds": GRID}
    for region in ("ET", "TC", "WT"):
        m = miss.get(region, np.zeros(N))
        base = np.where(gt > 0, np.round(m * gt), 0).astype(np.int64)
        # fn rises (weakly) with threshold, as a real miss count must.
        fn = np.minimum(base[:, None] + np.arange(73)[None, :], gt[:, None])
        arrays[f"{region}__case_ids"] = np.asarray(CASES)
        arrays[f"{region}__gt_voxels"] = gt.astype(np.int64)
        arrays[f"{region}__fn_voxels"] = fn.astype(np.int64)
        arrays[f"{region}__mask_voxels"] = (gt[:, None] - fn + 5).astype(np.int64)
    np.savez(path, **arrays)


def _make_shard(
    root: Path,
    model: str,
    kind: str,
    level: int,
    miss: dict[str, np.ndarray],
    *,
    done: bool = True,
    gt: np.ndarray | None = None,
    per_case: pd.DataFrame | None = None,
) -> Path:
    d = root / model / kind / f"level{level}"
    d.mkdir(parents=True)
    _curves_npz(d / "curves.npz", miss, gt)
    if per_case is None:
        per_case = pd.DataFrame({"case_id": CASES, "model": model})
    per_case.to_csv(d / "per_case.csv", index=False)
    pd.DataFrame({"case_id": CASES, "f1": np.arange(N, dtype=float)}).to_csv(
        d / "ood_features.csv", index=False
    )
    (d / "shard_meta.json").write_text("{}")
    if done:
        (d / "DONE").write_text("done\n")
    return d


def _ladder(
    root: Path,
    model: str,
    kinds: dict[str, float],
    *,
    levels: tuple[int, ...] = (2, 3, 4),
    noise_seed: int = 0,
) -> None:
    """Clean level 0 plus `levels` per kind; `kinds[k]` is the miss-rate rise per level step."""
    rng = np.random.default_rng(noise_seed)
    base = 0.05 + 0.002 * np.arange(N)
    jitter = {lvl: rng.normal(0, 0.003, N) for lvl in (0, 1, 2, 3, 4)}
    _make_shard(
        root, model, "none", 0, {r: np.clip(base + jitter[0], 0, 1) for r in ("WT", "TC", "ET")}
    )
    for kind, slope in kinds.items():
        for lvl in levels:
            m = np.clip(base + slope * lvl + jitter[lvl], 0, 1)
            _make_shard(root, model, kind, lvl, {"WT": m, "TC": m, "ET": m})


def _taus(model_names: list[str]) -> dict[str, dict[tuple[str, float], float]]:
    return {m: {("WT", 0.10): TAU, ("TC", 0.10): TAU} for m in model_names}


# ---------------------------------------------------------------------------
# discover_shards / loaders
# ---------------------------------------------------------------------------


def test_discover_shards_and_done_required(tmp_path: Path) -> None:
    _ladder(tmp_path, "m1", {"rician": 0.01})
    shards = sla.discover_shards([tmp_path])
    assert list(shards.columns) == ["model", "kind", "level_index", "path"]
    assert sorted(shards["level_index"].unique()) == [0, 2, 3, 4]
    _make_shard(tmp_path, "m2", "rician", 2, {}, done=False)
    with pytest.raises(FileNotFoundError, match="DONE"):
        sla.discover_shards([tmp_path])
    assert len(sla.discover_shards([tmp_path], require_done=False)) == len(shards) + 1


def test_discover_logs_absent_pairs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _ladder(tmp_path, "full", {"rician": 0.01, "bias": 0.01})
    _ladder(tmp_path, "cut", {"rician": 0.01})
    with caplog.at_level("INFO"):
        shards = sla.discover_shards([tmp_path])
    assert "cut" in caplog.text and "bias" in caplog.text
    assert shards[(shards["model"] == "cut") & (shards["kind"] == "bias")].empty


def test_load_v1_taus(tmp_path: Path) -> None:
    fit = {
        "WT__alpha_0.1": {"threshold": 0.725, "feasible": True},
        "WT__alpha_0.2": {"threshold": 0.95, "feasible": True},
        "TC__alpha_0.1": {"threshold": None, "feasible": False},
        "TC__alpha_0.2": {"threshold": 0.5, "feasible": True},
    }
    p = tmp_path / "fit.json"
    p.write_text(json.dumps(fit))
    got = sla.load_v1_taus(p, ["WT", "TC"], [0.1, 0.2])
    assert got == {("WT", 0.1): 0.725, ("WT", 0.2): 0.95, ("TC", 0.2): 0.5}  # infeasible skipped
    with pytest.raises(KeyError):
        sla.load_v1_taus(p, ["ET"], [0.1])


def test_load_v2_taus(tmp_path: Path) -> None:
    cols = "model,region,alpha,empty,grid,guarantee,feasible,threshold"
    rows = [
        "m,WT,0.1,exclude,extended,crc,True,0.7",
        "m,WT,0.1,exclude,extended,rcps,True,0.5",
        "m,ET,0.05,zero,extended,crc,False,",
        "m,WT,0.1,exclude,v1,crc,True,0.9",
        "other,WT,0.1,exclude,extended,crc,True,0.1",
    ]
    p = tmp_path / "fits.csv"
    p.write_text("\n".join([cols, *rows]) + "\n")
    assert sla.load_v2_taus(p, "m") == {("WT", 0.1, "crc"): 0.7, ("WT", 0.1, "rcps"): 0.5}


# ---------------------------------------------------------------------------
# per_case_miss / miss_matrix
# ---------------------------------------------------------------------------


def _curve(case: str, gt: int, fn: list[int]) -> CaseLossCurve:
    return CaseLossCurve(
        case_id=case,
        region="WT",
        gt_voxels=gt,
        thresholds=(0.1, 0.5, 0.9),
        fn_voxels=tuple(fn),
        mask_voxels=(0, 0, 0),
    )


def test_per_case_miss_hand_computed_and_empty_conventions() -> None:
    curves = [_curve("a", 10, [1, 2, 5]), _curve("b", 4, [0, 1, 4]), _curve("c", 0, [0, 0, 0])]
    exclude = sla.per_case_miss(curves, 0.5, "exclude")
    assert list(exclude.index) == ["a", "b"]  # empty-GT case c dropped, not scored
    assert exclude["a"] == pytest.approx(0.2) and exclude["b"] == pytest.approx(0.25)
    zero = sla.per_case_miss(curves, 0.5, "zero")
    assert zero["c"] == 0.0 and len(zero) == 3
    # Agrees with the repo's own risk function under both conventions.
    for empty, s in (("exclude", exclude), ("zero", zero)):
        assert s.mean() == pytest.approx(
            realised_risk_with(curves, 0.5, empty=empty)["mean_miss_rate"]
        )


def test_per_case_miss_off_grid_raises() -> None:
    with pytest.raises(ValueError, match="not in the threshold grid"):
        sla.per_case_miss([_curve("a", 10, [1, 2, 5])], 0.3, "exclude")
    with pytest.raises(ValueError, match="empty"):
        sla.per_case_miss([_curve("a", 10, [1, 2, 5])], 0.5, "drop")


def test_shard_curves_load_with_repo_reader(tmp_path: Path) -> None:
    """The shard `curves.npz` layout is read directly by `load_curves_npz` (no adapter)."""
    _ladder(tmp_path, "m", {"rician": 0.01})
    path = tmp_path / "m" / "none" / "level0" / "curves.npz"
    curves = load_curves_npz(path, ("WT",))["WT"]
    assert len(curves) == N and len(curves[0].thresholds) == 73


def test_miss_matrix_levels_and_values(tmp_path: Path) -> None:
    _ladder(tmp_path, "m", {"rician": 0.01})
    shards = sla.discover_shards([tmp_path])
    mat = sla.miss_matrix(shards, "m", "rician", "WT", TAU, "exclude")
    assert list(mat.columns) == [0, 2, 3, 4]
    assert mat.shape == (N, 4)
    clean = _read_miss(tmp_path / "m" / "none" / "level0")
    np.testing.assert_allclose(mat[0].to_numpy(), clean)


def _read_miss(shard: Path) -> np.ndarray:
    z = np.load(shard / "curves.npz")
    return z["WT__fn_voxels"][:, 0] / z["WT__gt_voxels"]


def test_miss_matrix_mismatched_case_sets_raise(tmp_path: Path) -> None:
    _ladder(tmp_path, "m", {"rician": 0.01})
    gt = np.full(N, GT)
    gt[3] = 0  # level 4 sees case 3 as empty-GT, so "exclude" drops it only there
    shutil_dir = tmp_path / "m" / "rician" / "level4"
    _curves_npz(shutil_dir / "curves.npz", {"WT": np.full(N, 0.1)}, gt)
    shards = sla.discover_shards([tmp_path])
    with pytest.raises(ValueError, match="different case set"):
        sla.miss_matrix(shards, "m", "rician", "WT", TAU, "exclude")


def test_miss_matrix_consistent_empty_gt_drops_same_case(tmp_path: Path) -> None:
    gt = np.full(N, GT)
    gt[3] = 0
    for lvl, kind in ((0, "none"), (2, "rician"), (4, "rician")):
        _make_shard(tmp_path, "m", kind, lvl, {"WT": np.full(N, 0.1 * (1 + lvl))}, gt=gt)
    mat = sla.miss_matrix(sla.discover_shards([tmp_path]), "m", "rician", "WT", TAU, "exclude")
    assert mat.shape == (N - 1, 3) and CASES[3] not in mat.index


# ---------------------------------------------------------------------------
# Page direction (scipy)
# ---------------------------------------------------------------------------


def test_page_direction_increasing_small_p_decreasing_large_p() -> None:
    from scipy.stats import page_trend_test

    rng = np.random.default_rng(0)
    inc = np.arange(4)[None, :] + rng.normal(0, 0.1, (N, 4))  # strictly increasing columns
    ranks = np.arange(1, 5)
    assert page_trend_test(inc, predicted_ranks=ranks).pvalue < 1e-3
    assert page_trend_test(inc[:, ::-1], predicted_ranks=ranks).pvalue > 0.99
    flat = rng.normal(0, 1, (N, 4))
    assert page_trend_test(flat, predicted_ranks=ranks).pvalue > 0.01


# ---------------------------------------------------------------------------
# Primary family and kind verdicts
# ---------------------------------------------------------------------------


def _build_three_models(tmp_path: Path) -> pd.DataFrame:
    kinds_all = {"rician": 0.02, "motion": 0.0}  # rician rises strongly; motion is flat
    _ladder(tmp_path, "neurovision", {**kinds_all, "bias": 0.02}, noise_seed=1)
    _ladder(tmp_path, "baseline_unet3d", kinds_all, noise_seed=2)  # no bias: cut 2
    _ladder(tmp_path, "nnunet_gatea", {**kinds_all, "bias": 0.02}, noise_seed=3)
    return sla.discover_shards([tmp_path])


def test_primary_graded_flat_holm_count_and_absent_rows(tmp_path: Path) -> None:
    shards = _build_three_models(tmp_path)
    models = ["neurovision", "baseline_unet3d", "nnunet_gatea"]
    primary = sla.primary_trend_family(shards, _taus(models), n_boot=NBOOT)
    # 3 kinds present: rician (3 models) + motion (3) + bias (2) = 8 cells x 2 regions = 16.
    assert len(primary) == 16
    assert primary[(primary["kind"] == "bias") & (primary["model"] == "baseline_unet3d")].empty
    assert set(primary["n_levels"]) == {4}
    rician = primary[primary["kind"] == "rician"]
    assert (rician["verdict"] == "GRADED").all()
    assert (rician["ci_lo"] > 0).all() and (rician["p_holm"] < 0.05).all()
    motion = primary[primary["kind"] == "motion"]
    assert (motion["verdict"] == "NOT GRADED").all()
    # Holm over the whole family: adjusted p >= raw p, and the largest raw p is unchanged-or-up.
    assert (primary["p_holm"] >= primary["p"] - 1e-12).all()
    smallest = primary["p"].min()
    assert primary["p_holm"][primary["p"].idxmin()] == pytest.approx(min(1.0, smallest * 16))
    # diff is mean(level 4) - mean(level 0), hand-checked on one cell.
    cell = primary[(primary["kind"] == "rician") & (primary["model"] == "neurovision")].iloc[0]
    assert cell["diff"] == pytest.approx(cell["mean_l4"] - cell["mean_l0"])


def test_kind_verdicts_measured_models_only(tmp_path: Path) -> None:
    shards = _build_three_models(tmp_path)
    models = ["neurovision", "baseline_unet3d", "nnunet_gatea"]
    primary = sla.primary_trend_family(shards, _taus(models), n_boot=NBOOT)
    kv = sla.kind_verdicts(primary).set_index("kind")
    assert kv.loc["rician", "verdict"] == "GRADED" and kv.loc["rician", "models_measured"] == 3
    assert kv.loc["bias", "verdict"] == "GRADED_ON_MEASURED_MODELS_ONLY"
    assert kv.loc["bias", "models_measured"] == 2
    assert kv.loc["motion", "verdict"] == "NOT GRADED"


def test_kind_verdicts_needs_all_models_in_one_region() -> None:
    cells = pd.DataFrame(
        {
            "kind": ["k"] * 4,
            "model": ["a", "b", "a", "b"],
            "region": ["WT", "WT", "TC", "TC"],
            "verdict": ["GRADED", "NOT GRADED", "GRADED", "GRADED"],
        }
    )
    row = sla.kind_verdicts(cells, n_models_required=2).iloc[0]
    assert row["verdict"] == "GRADED" and row["graded_regions"] == "TC"


# ---------------------------------------------------------------------------
# Secondary slopes
# ---------------------------------------------------------------------------


def test_ols_slope_uses_actual_level_index() -> None:
    # y = 0.1 * x exactly at x = 0, 2, 3, 4 -> slope 0.1. Positions 0..3 would give 0.1 * 4/3.
    mat = pd.DataFrame({0: [0.0], 2: [0.2], 3: [0.3], 4: [0.4]}, index=["c"])
    assert sla._ols_slopes(mat)["c"] == pytest.approx(0.1)
    # Hand-computed: x - xbar = (-2.25, -0.25, 0.75, 1.75), Sxx = 8.75, y - ybar = (0, -1, 1, 0),
    # Sxy = 0.25 + 0.75 = 1.0, so slope = 1 / 8.75.
    got = sla._ols_slopes(pd.DataFrame([[1.0, 0.0, 2.0, 1.0]], columns=[0, 2, 3, 4]))
    assert got.iloc[0] == pytest.approx(1.0 / 8.75)


def test_slope_family_sign_pairs_and_holm(tmp_path: Path) -> None:
    # nnU-Net rises slowly (0.005/level), neurovision faster (0.02), baseline fastest (0.04).
    _ladder(tmp_path, "nnunet_gatea", {"rician": 0.005, "bias": 0.005}, noise_seed=1)
    _ladder(tmp_path, "neurovision", {"rician": 0.02, "bias": 0.02}, noise_seed=2)
    _ladder(tmp_path, "baseline_unet3d", {"rician": 0.04}, noise_seed=3)
    shards = sla.discover_shards([tmp_path])
    taus = _taus(["nnunet_gatea", "neurovision", "baseline_unet3d"])
    sec = sla.secondary_slope_family(shards, taus, n_boot=3000)  # min p = 2/3001, x6 Holm < 0.05
    # nn-vs-nv: rician+bias x 2 regions = 4; nv-vs-base: rician only x 2 = 2.
    assert len(sec) == 6
    assert sec[(sec["weaker"] == "baseline_unet3d") & (sec["kind"] == "bias")].empty
    assert (sec["diff"] < 0).all()  # stronger minus weaker, stronger is flatter
    assert (sec["verdict"] == "FLATTER").all()
    assert (sec["p_holm"] >= sec["p"] - 1e-12).all()
    # Reversed roles flip the sign and the verdict.
    rev = sla.secondary_slope_family(
        shards, taus, stronger_weaker=[("neurovision", "nnunet_gatea")], n_boot=3000
    )
    assert (rev["diff"] > 0).all() and (rev["verdict"] == "NOT FLATTER").all()


def test_bootstrap_p_uses_same_resamples_as_ci() -> None:
    from neurovision.analysis.statistics import paired_bootstrap_ci

    diff = np.random.default_rng(5).normal(0.5, 1.0, 40)
    ci = paired_bootstrap_ci(diff, np.zeros(40), n_boot=300, generator=np.random.default_rng(42))
    idx = np.random.default_rng(42).integers(0, 40, size=(300, 40))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert (ci.lo, ci.hi) == (pytest.approx(lo), pytest.approx(hi))  # identical stream
    p = sla._bootstrap_two_sided_p(diff, 300, 42)
    expected = min(1.0, 2 * min((np.sum(boot <= 0) + 1) / 301, (np.sum(boot >= 0) + 1) / 301))
    assert p == pytest.approx(expected)
    assert sla._bootstrap_two_sided_p(-np.ones(30), 300, 42) == pytest.approx(2 / 301)


# ---------------------------------------------------------------------------
# Gate and OOD
# ---------------------------------------------------------------------------


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "clinical": {
                "gatekeeper": {
                    "enabled_signals": ["input_qc", "predicted_dice"],
                    "regions": ["WT", "TC"],
                    "thresholds": None,
                }
            }
        }
    )


def _thresholds() -> Thresholds:
    return Thresholds(
        predicted_dice={r: (0.5, 0.7) for r in ("WT", "TC")},
        conformal_band={r: (0.1, 0.3) for r in ("WT", "TC")},
        ood_score=(0.1, 0.3),
        calibration_n=187,
        caution_quantile=0.10,
        refuse_quantile=0.02,
    )


def _gate_frame(model: str) -> pd.DataFrame:
    # case0: accepted+usable; case1: accepted, Dice low (silent failure);
    # case2: refused (predicted dice 0.2), usable; case3: refused, not usable.
    return pd.DataFrame(
        {
            "case_id": ["c0", "c1", "c2", "c3"],
            "model": model,
            "dice_WT": [0.9, 0.5, 0.9, 0.3],
            "dice_TC": [0.9, 0.9, 0.9, 0.3],
            "predicted_dice_WT": [0.95, 0.95, 0.2, 0.2],
            "predicted_dice_TC": [0.95, 0.95, 0.95, 0.95],
            "conformal_band_WT": 0.05,
            "conformal_band_TC": 0.05,
        }
    )


def test_gate_decisions_labels_and_silent_failure() -> None:
    out = sla.gate_decisions(_cfg(), _gate_frame("neurovision"), _thresholds(), ("WT", "TC"), 50.0)
    assert list(out["accepted"]) == [True, True, False, False]
    assert list(out["usable"]) == [True, False, True, False]
    assert list(out["silent_failure"]) == [False, True, False, False]  # accepted & ~usable
    assert set(out["gate_label"]) == {"deployed"}
    other = sla.gate_decisions(
        _cfg(), _gate_frame("nnunet_gatea"), _thresholds(), ("WT", "TC"), 50.0
    )
    assert set(other["gate_label"]) == {"transferred, not refitted"}


def test_gate_decisions_passes_frozen_thresholds_and_stand_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    real = sla.decide_cases

    def spy(cfg, signals, thresholds, *, regions, input_qc=None):  # type: ignore[no-untyped-def]
        seen.update(thresholds=thresholds, signals=signals, regions=regions)
        return real(cfg, signals, thresholds, regions=regions, input_qc=input_qc)

    monkeypatch.setattr(sla, "decide_cases", spy)
    thr = _thresholds()
    sla.gate_decisions(_cfg(), _gate_frame("neurovision"), thr, ("WT", "TC"), 42.0)
    assert seen["thresholds"] is thr  # the exact frozen object, nothing refitted
    signals = seen["signals"]
    assert isinstance(signals, pd.DataFrame)
    assert (signals["patient_age_years"] == 42.0).all()
    assert {"case_id", "predicted_dice_WT", "conformal_band_TC"} <= set(signals.columns)


def test_gate_decisions_rejects_mixed_models() -> None:
    frame = _gate_frame("neurovision")
    frame.loc[0, "model"] = "other"
    with pytest.raises(ValueError, match="one model"):
        sla.gate_decisions(_cfg(), frame, _thresholds(), ("WT", "TC"), 50.0)


def test_ood_flags_with_frozen_cuts() -> None:
    model = OODModel(
        feature_names=("f1",),
        center=np.array([0.0]),
        scale=np.array([1.0]),
        precision=np.array([[1.0]]),
        shrinkage=0.1,
        n_fit=10,
    )
    feats = pd.DataFrame({"case_id": ["a", "b", "c"], "f1": [0.5, 2.0, 5.0]})
    out = sla.ood_flags(model, (1.0, 3.0), feats)
    assert list(out["ood_score"]) == pytest.approx([0.5, 2.0, 5.0])
    assert list(out["ood_flag"]) == ["proceed", "proceed_with_caution", "refuse"]
    assert list(out["ood_not_ok"]) == [False, True, True]


# ---------------------------------------------------------------------------
# Descriptive table
# ---------------------------------------------------------------------------


def test_descriptive_table_schema_values_and_verdict(tmp_path: Path) -> None:
    pc = _gate_frame("m").drop(columns=["model"]).assign(model="m")
    pc = pd.concat([pc] * 3, ignore_index=True).iloc[:N].copy()
    pc["case_id"] = CASES[: len(pc)]
    pc["dice_ET"] = np.where(np.arange(len(pc)) == 0, np.nan, 0.8)
    for kind, lvl, level_miss in (("none", 0, 0.05), ("rician", 2, 0.30)):
        _make_shard(tmp_path, "m", kind, lvl, {"WT": np.full(N, level_miss)}, per_case=pc)
    shards = sla.discover_shards([tmp_path])
    taus = {"m": {("WT", 0.10, "crc"): TAU}}
    key = ("m", "rician", 2)
    dec = pd.DataFrame(
        {
            "accepted": [True, True, False, False] * 3,
            "silent_failure": [False, True, False, False] * 3,
            "usable": [True, False, True, False] * 3,
        }
    )
    ood = {key: pd.DataFrame({"ood_not_ok": [True, False, False, False] * 3})}
    out = sla.descriptive_table(shards, taus, decisions={key: dec}, ood=ood, n_boot=NBOOT)
    assert list(out.columns) == [
        "model", "kind", "level_index", "quantity", "region", "alpha", "method", "n",
        "value", "ci_lo", "ci_hi", "verdict",
    ]  # fmt: skip
    risk = out[out["quantity"] == "realised_risk"].set_index("level_index")
    assert risk.loc[0, "value"] == pytest.approx(0.05) and risk.loc[0, "verdict"] == "holds"
    # Constant 0.30 across cases -> CI is a point above alpha = 0.10 -> VIOLATED.
    assert risk.loc[2, "value"] == pytest.approx(0.30) and risk.loc[2, "verdict"] == "VIOLATED"
    et = out[(out["quantity"] == "dice") & (out["region"] == "ET") & (out["level_index"] == 0)]
    assert et["n"].iloc[0] == N - 1 and et["value"].iloc[0] == pytest.approx(0.8)  # NaN skipped
    rates = out[out["level_index"] == 2].set_index("quantity")["value"]
    assert rates["refusal_rate"] == pytest.approx(0.5)
    assert rates["silent_failure_rate"] == pytest.approx(0.25)
    assert rates["usable_rate"] == pytest.approx(0.5)
    assert rates["ood_flag_rate"] == pytest.approx(0.25)
    assert out[out["level_index"] == 0]["quantity"].isin(["refusal_rate"]).sum() == 0  # no gate
