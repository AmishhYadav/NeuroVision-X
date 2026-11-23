"""Tests for scripts/conformal_v2.py. Synthetic 8^3 volumes, CPU only.

The driver is loaded with `load_script` (scripts/ is not a package). The "v1 conformal dir"
that the additivity check compares against is produced by running the real v1 driver
(`scripts/conformal.py::run_conformal`) on the same synthetic logits, so the check is tested
against genuine v1 output, not a hand-written stand-in.

Logits here use ONE noise field shared by the three region channels, so the predicted masks
are exactly nested at every threshold (needed for the "post-processing is a no-op" test).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.uncertainty.conformal import (
    DEFAULT_THRESHOLDS,
    EXTENDED_THRESHOLDS,
    CaseLossCurve,
    fit_threshold_crc,
    fit_threshold_rcps,
    realised_risk_with,
)
from tests.script_loader import load_script
from tests.test_conformal_script import (
    _make_cfg,
    _region_indicator,
    _write_split,
    run_conformal,
)

v2 = load_script("conformal_v2")

N_VAL, N_TEST, N_SSA, N_PED = 60, 8, 6, 6
NN_COHORT_SIZES = {"val": N_VAL, "test": N_TEST, "ssa": N_SSA, "ped": N_PED}


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def _nested_logits(label: np.ndarray, seed: int) -> np.ndarray:
    """(3, D, H, W) float16 logits, nested at every threshold (one noise field, all channels)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=0.8, size=label.shape)
    return (_region_indicator(label) * 10.0 - 5.0 + noise[None]).astype(np.float16)


def _build_world(root: Path) -> dict[str, Any]:
    """Two logit models (m1, m2) over four cohorts, their v1 outputs, one curves-only model."""
    prep = root / "prep"
    sizes = NN_COHORT_SIZES
    eval_dirs: dict[str, dict[str, Path]] = {}
    for mi, model in enumerate(("m1", "m2")):
        eval_dirs[model] = {}
        for ci, (cohort, n) in enumerate(sizes.items()):
            d = _write_split(root / model, prep, cohort, n)
            for i in range(n):
                case = f"{cohort}_{i:03d}"
                label = np.load(prep / case / "label.npy")
                if i % 4 == 3:  # some cases with no enhancing tumour at all
                    label = np.where(label == 3, 1, label).astype(label.dtype)
                    np.save(prep / case / "label.npy", label)
                np.save(
                    d / "logits" / f"{case}.npy", _nested_logits(label, 1000 * mi + 100 * ci + i)
                )
            eval_dirs[model][cohort] = d

    v1_dirs: dict[str, Path] = {}
    for model in ("m1", "m2"):
        v1_dir = root / "v1" / model
        v1_cfg = _make_cfg(
            eval_dirs[model]["val"],
            [eval_dirs[model][c] for c in ("test", "ssa", "ped")],
            prep,
            [],
            v1_dir,
            check_consistency=False,
        )
        run_conformal(v1_cfg)
        v1_dirs[model] = v1_dir

    # A curves-only model: v1-grid WT/TC curves saved in the v1 npz format (like nnU-Net's).
    nn_curves: dict[str, str] = {}
    for cohort in sizes:
        out = root / "nn_curves"
        v2.extract_curves(
            eval_dirs["m1"][cohort], prep, out, regions=["WT", "TC"], thresholds=DEFAULT_THRESHOLDS
        )
        nn_curves[cohort] = str(out / eval_dirs["m1"][cohort].name / "curves.npz")
    return {"prep": prep, "eval_dirs": eval_dirs, "v1_dirs": v1_dirs, "nn_curves": nn_curves}


def _block(world: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    models: dict[str, Any] = {
        m: {
            "eval_dirs": {c: str(p) for c, p in world["eval_dirs"][m].items()},
            "v1_conformal_dir": str(world["v1_dirs"][m]),
        }
        for m in ("m1", "m2")
    }
    models["nn"] = {"curves": dict(world["nn_curves"])}
    return {
        "out_dir": str(out_dir),
        "alphas": [0.05, 0.10, 0.20],
        "delta": 0.1,
        "regions": ["ET", "TC", "WT"],
        "empty": {"ET": "zero", "TC": "exclude", "WT": "exclude"},
        "rcps_order_seed": 42,
        "n_boot": 200,
        "boot_seed": 42,
        "n_resplits": 20,
        "resplit_seed": 42,
        "prep_dirs": {c: str(world["prep"]) for c in ("val", "test", "ssa", "ped")},
        "models": models,
        "postprocess": True,
    }


def _full_cfg(min_component_size: int = 0) -> Any:
    return OmegaConf.create(
        {
            "inference": {
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": min_component_size,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                }
            }
        }
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _build_world(tmp_path_factory.mktemp("v2world"))


@pytest.fixture(scope="module")
def run(world: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("v2out")
    block = _block(world, out)
    paths = v2.run_v2(block, _full_cfg(0), Path("."))
    return {"paths": paths, "block": block, "out": out}


# ---------------------------------------------------------------------------
# Additivity
# ---------------------------------------------------------------------------


def test_additivity_passes_on_consistent_data(run: dict[str, Any]) -> None:
    data = json.loads(run["paths"]["additivity_json"].read_text())
    assert data["all_ok"] is True
    # 2 models x 2 regions x 3 alphas fits, each with up to 3 cohorts of realised risk.
    assert data["n_comparisons"] >= 2 * 2 * 3
    assert {r["kind"] for r in data["rows"]} == {"fit", "realised_risk"}


@pytest.mark.parametrize("what", ["fit_threshold", "realised_risk"])
def test_additivity_raises_when_v1_is_perturbed(
    world: dict[str, Any], tmp_path: Path, what: str
) -> None:
    v1_copy = tmp_path / "v1_m1"
    shutil.copytree(world["v1_dirs"]["m1"], v1_copy)
    if what == "fit_threshold":
        fit = json.loads((v1_copy / "fit.json").read_text())
        fit["WT__alpha_0.1"]["threshold"] += 0.001
        (v1_copy / "fit.json").write_text(json.dumps(fit))
    else:
        df = pd.read_csv(v1_copy / "realised_risk.csv")
        df.loc[0, "mean_miss_rate"] += 1e-6
        df.to_csv(v1_copy / "realised_risk.csv", index=False)
    block = _block(world, tmp_path / "out")
    block["models"]["m1"]["v1_conformal_dir"] = str(v1_copy)
    with pytest.raises(RuntimeError, match="m1"):
        v2.run_v2(block, _full_cfg(0), Path("."))
    saved = json.loads((tmp_path / "out" / "additivity.json").read_text())
    assert saved["all_ok"] is False  # the full comparison is on disk even though the run stopped
    assert not (tmp_path / "out" / "fits.csv").exists()  # and nothing after it was written


def test_restrict_curves_picks_v1_grid(world: dict[str, Any], run: dict[str, Any]) -> None:
    curves = v2.extract_curves(
        world["eval_dirs"]["m1"]["test"],
        world["prep"],
        run["out"] / "curves" / "m1",
        regions=["WT"],
        thresholds=EXTENDED_THRESHOLDS,
    )["WT"]
    sub = v2.restrict_curves(curves)
    assert sub[0].thresholds == DEFAULT_THRESHOLDS
    j = list(curves[0].thresholds).index(DEFAULT_THRESHOLDS[10])
    assert sub[0].fn_voxels[10] == curves[0].fn_voxels[j]


# ---------------------------------------------------------------------------
# fits.csv
# ---------------------------------------------------------------------------


def test_fits_rows_and_conventions(run: dict[str, Any]) -> None:
    fits = pd.read_csv(run["paths"]["fits_csv"])
    # m1, m2: 3 regions; nn: 2 regions -> 8 model-regions x 3 alphas x 2 guarantees.
    assert len(fits) == 8 * 3 * 2
    assert set(fits["guarantee"]) == {"crc", "rcps"}
    et = fits[fits["region"] == "ET"]
    assert set(et["empty"]) == {"zero"}
    assert (et["n_excluded_empty"] == 0).all()
    assert (et["n_calibration"] == N_VAL).all()  # empty-ET cases are kept with loss 0
    assert set(fits[fits["region"] != "ET"]["empty"]) == {"exclude"}
    assert set(fits[fits["model"] == "m1"]["grid"]) == {"extended"}


def test_rcps_threshold_not_above_crc_where_both_feasible(run: dict[str, Any]) -> None:
    fits = pd.read_csv(run["paths"]["fits_csv"])
    wide = fits.pivot_table(
        index=["model", "region", "alpha"], columns="guarantee", values="threshold"
    ).dropna()
    assert len(wide) > 0  # the comparison is not vacuous
    assert (wide["rcps"] <= wide["crc"] + 1e-12).all()


def test_at_grid_ceiling_flag(run: dict[str, Any]) -> None:
    fits = pd.read_csv(run["paths"]["fits_csv"])
    ext = fits[(fits["grid"] == "extended") & fits["threshold"].notna()]
    assert ((ext["threshold"] == max(EXTENDED_THRESHOLDS)) == ext["at_grid_ceiling"]).all()


# ---------------------------------------------------------------------------
# Curves-only model
# ---------------------------------------------------------------------------


def test_curves_only_model_has_no_et_and_is_marked(run: dict[str, Any]) -> None:
    fits = pd.read_csv(run["paths"]["fits_csv"])
    nn = fits[fits["model"] == "nn"]
    assert set(nn["region"]) == {"TC", "WT"}
    assert set(nn["grid"]) == {"v1"}
    nm = pd.read_csv(run["paths"]["not_measured_csv"])
    nn_items = set(nm[nm["model"] == "nn"]["item"])
    assert {"region:ET", "extended_grid", "postprocessed"} <= nn_items
    assert (nm["status"] == "not_measured_needs_W2.6").all()
    realised = pd.read_csv(run["paths"]["realised_csv"])
    assert "ET" not in set(realised[realised["model"] == "nn"]["region"])
    post = pd.read_csv(run["paths"]["postprocessed_csv"])
    assert "nn" not in set(post["model"])  # no logits -> skipped


# ---------------------------------------------------------------------------
# realised.csv and verdicts
# ---------------------------------------------------------------------------


def test_realised_table_shape_and_verdict_values(run: dict[str, Any]) -> None:
    r = pd.read_csv(run["paths"]["realised_csv"])
    assert set(r["cohort"]) == {"test", "ssa", "ped"}
    assert set(r["verdict"]) <= {"holds", "exceeds", "VIOLATED", "inconclusive", "infeasible"}
    inf = r[r["verdict"] == "infeasible"]
    assert inf["threshold"].isna().all()
    ok = r[r["verdict"] != "infeasible"]
    assert (ok["ci_lo"] <= ok["ci_hi"]).all()
    assert (ok["mean_miss_rate"].between(0, 1)).all()
    assert {"precision_mean", "dice_median", "volume_ratio_n_nan"} <= set(r.columns)


@pytest.mark.parametrize(
    ("cohort", "point", "lo", "hi", "alpha", "expected"),
    [
        ("test", 0.10, 0.0, 0.5, 0.10, "holds"),  # point <= alpha, CI ignored in distribution
        ("test", 0.1001, 0.0, 0.0, 0.10, "exceeds"),
        ("ssa", 0.20, 0.11, 0.30, 0.10, "VIOLATED"),  # lower bound above alpha
        ("ped", 0.05, 0.02, 0.10, 0.10, "holds"),  # upper bound <= alpha
        ("ssa", 0.10, 0.05, 0.15, 0.10, "inconclusive"),
        ("ped", 0.10, 0.10, 0.12, 0.10, "inconclusive"),  # lo == alpha is not > alpha
        ("ssa", 0.1, float("nan"), float("nan"), 0.10, "inconclusive"),
    ],
)
def test_cell_verdict_rules(
    cohort: str, point: float, lo: float, hi: float, alpha: float, expected: str
) -> None:
    assert v2.cell_verdict(cohort, point, lo, hi, alpha) == expected


def test_mask_cost_stats_counts_nans() -> None:
    grid = (0.5,)
    curves = [
        CaseLossCurve("a", "ET", 100, grid, (20,), (120,)),  # prec 80/120, dice 160/220, vol 1.2
        CaseLossCurve("b", "ET", 0, grid, (0,), (10,)),  # empty reference: vol NaN
        CaseLossCurve("c", "ET", 50, grid, (50,), (0,)),  # empty mask: precision NaN
    ]
    s = v2.mask_cost_stats(curves, 0.5)
    assert s["volume_ratio_n_nan"] == 1
    assert s["precision_n_nan"] == 1
    assert s["precision_mean"] == pytest.approx((80 / 120 + 0.0) / 2)  # 'b' precision is 0
    assert s["volume_ratio_mean"] == pytest.approx((1.2 + 0.0) / 2)


# ---------------------------------------------------------------------------
# Re-splits
# ---------------------------------------------------------------------------


def _synthetic_pool(n: int, seed: int, n_empty: int = 3) -> list[CaseLossCurve]:
    """Curves on a 6-point grid; fn non-decreasing in tau, a few empty-reference cases."""
    rng = np.random.default_rng(seed)
    grid = (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
    out = []
    for i in range(n):
        gt = 0 if i < n_empty else int(rng.integers(1000, 2000))
        fn = (
            (0,) * len(grid)
            if gt == 0
            else tuple(int(v) for v in np.sort(rng.integers(0, int(0.5 * gt), size=len(grid))))
        )
        out.append(CaseLossCurve(f"c{i:03d}", "ET", gt, grid, fn, (gt,) * len(grid)))
    return out


@pytest.mark.parametrize("empty", ["zero", "exclude"])
def test_resplit_cell_matches_library_fits(empty: str) -> None:
    pool = _synthetic_pool(40, 3)
    alphas = [0.05, 0.10, 0.20]
    perms = v2.make_perms(len(pool), 6, 7)
    result = v2.resplit_cell(v2.loss_matrix(pool, empty), perms, alphas, 0.1, 42)
    half = len(pool) // 2
    seen_feasible = {"crc": False, "rcps": False}
    seen_infeasible = {"crc": False, "rcps": False}
    for s in range(len(perms)):
        calib = [pool[i] for i in sorted(perms[s, :half])]
        held = [pool[i] for i in perms[s, half:]]
        for alpha in alphas:
            lib = {
                "crc": fit_threshold_crc(calib, alpha, empty=empty).threshold,
                "rcps": fit_threshold_rcps(calib, alpha, 0.1, order_seed=42, empty=empty).threshold,
            }
            for g in ("crc", "rcps"):
                idx, realised = result[(g, alpha)]
                if lib[g] is None:
                    seen_infeasible[g] = True
                    assert idx[s] == -1 and np.isnan(realised[s])
                else:
                    seen_feasible[g] = True
                    assert pool[0].thresholds[idx[s]] == lib[g]
                    expected = realised_risk_with(held, lib[g], empty=empty)["mean_miss_rate"]
                    assert realised[s] == pytest.approx(expected)
    assert all(seen_feasible.values()) and any(seen_infeasible.values())  # not vacuous


def test_resplits_deterministic_and_bounded(run: dict[str, Any]) -> None:
    summary = pd.read_csv(run["paths"]["resplits_csv"])
    assert len(summary) == 8 * 3 * 2
    ok = summary[summary["n_feasible"] > 0]
    assert ok["p_exceed"].between(0, 1).all()
    assert (ok["q05"] <= ok["q50"]).all() and (ok["q50"] <= ok["q95"]).all()
    assert (summary["n_feasible"] + summary["n_infeasible"] == 20).all()
    assert set(summary["verdict"]) <= {"holds", "FAILS", "infeasible"}

    per_split = pd.read_csv(run["paths"]["resplits_per_split"])
    assert len(per_split) == 8 * 3 * 2 * 20
    assert per_split["realised"].isna().sum() == summary["n_infeasible"].sum()


def test_resplits_same_seed_same_result(
    world: dict[str, Any], run: dict[str, Any], tmp_path: Path
) -> None:
    block = run["block"]
    models = [
        v2.load_model_curves(name, cfg, block, Path(block["out_dir"]), Path("."))
        for name, cfg in block["models"].items()
    ]
    a, a_split = v2.build_resplit_tables(models, block)
    b, b_split = v2.build_resplit_tables(models, block)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a_split, b_split)
    block2 = {**block, "resplit_seed": 43}
    _, c_split = v2.build_resplit_tables(models, block2)
    assert not a_split["realised"].equals(c_split["realised"])


def test_resplits_reject_overlapping_ids() -> None:
    pool = _synthetic_pool(10, 1)
    m = v2.ModelCurves(
        "x", {"val": {"ET": pool}, "test": {"ET": pool}}, "v1", False, {}, ["ET"], []
    )
    block = {
        "alphas": [0.1],
        "delta": 0.1,
        "empty": {"ET": "zero"},
        "n_resplits": 2,
        "resplit_seed": 1,
        "rcps_order_seed": 42,
    }
    with pytest.raises(AssertionError, match="disjoint"):
        v2.build_resplit_tables([m], block)


def test_all_infeasible_splits_are_counted_not_scored_as_zero() -> None:
    all_miss = np.ones((20, 4))  # every threshold loses everything
    perms = v2.make_perms(20, 5, 1)
    result = v2.resplit_cell(all_miss, perms, [0.1], 0.1, 42)
    for g in ("crc", "rcps"):
        idx, realised = result[(g, 0.1)]
        assert (idx == -1).all() and np.isnan(realised).all()
        s = v2.summarise_resplits(realised, 0.1, g, 0.1)
        assert s["n_feasible"] == 0 and s["n_infeasible"] == 5 and s["verdict"] == "infeasible"


def test_summarise_resplits_hand_values() -> None:
    realised = np.array([np.nan, 0.05, 0.15, 0.08])  # 3 feasible, 1 of them above alpha=0.1
    s = v2.summarise_resplits(realised, 0.1, "rcps", 0.1)
    assert (s["n_feasible"], s["n_infeasible"]) == (3, 1)
    assert s["p_exceed"] == pytest.approx(1 / 3)
    assert s["mcse_p"] == pytest.approx(np.sqrt((1 / 3) * (2 / 3) / 3))
    assert s["mean_realised"] == pytest.approx(0.28 / 3)
    assert s["verdict"] == "holds"  # 0.333 <= 0.1 + 2*0.272
    crc = v2.summarise_resplits(np.full(100, 0.3), 0.1, "crc", 0.1)
    assert crc["verdict"] == "FAILS"  # mean 0.3 > 0.1 + 2*0


# ---------------------------------------------------------------------------
# Deployed post-processing
# ---------------------------------------------------------------------------


def test_postprocess_noop_matches_raw_at_half(world: dict[str, Any], run: dict[str, Any]) -> None:
    eval_dir = world["eval_dirs"]["m1"]["test"]
    ids = [f"test_{i:03d}" for i in range(N_TEST)]
    counts = v2.postprocess_counts(
        eval_dir, world["prep"], ids, [0.5], ["ET", "TC", "WT"], _full_cfg(0)
    )
    raw = v2.extract_curves(
        eval_dir,
        world["prep"],
        run["out"] / "curves" / "m1",
        regions=["ET", "TC", "WT"],
        thresholds=EXTENDED_THRESHOLDS,
    )
    j = list(EXTENDED_THRESHOLDS).index(0.5)
    for region in ("ET", "TC", "WT"):
        for pp, rc in zip(counts[0.5][region], raw[region], strict=True):
            assert pp.gt_voxels == rc.gt_voxels
            assert pp.fn_voxels[0] == rc.fn_voxels[j]
            assert pp.mask_voxels[0] == rc.mask_voxels[j]


def test_postprocess_table_equals_raw_when_filter_off(run: dict[str, Any]) -> None:
    post = pd.read_csv(run["paths"]["postprocessed_csv"])
    assert len(post) > 0
    assert set(post["model"]) == {"m1", "m2"}
    # nested probabilities + min_component_size 0: post-processing changes nothing.
    assert np.allclose(post["delta_risk_vs_raw"], 0.0, atol=1e-12)


def test_postprocess_removing_everything_gives_risk_one(
    world: dict[str, Any], tmp_path: Path
) -> None:
    block = _block(world, tmp_path / "out")
    paths = v2.run_v2(block, _full_cfg(10**6), Path("."))  # no component can be this large
    post = pd.read_csv(paths["postprocessed_csv"])
    for region in ("TC", "WT"):  # every case has a tumour -> every voxel is a miss
        sub = post[post["region"] == region]
        assert len(sub) > 0
        assert np.allclose(sub["mean_miss_rate"], 1.0)
        assert np.allclose(sub["delta_risk_vs_raw"], 1.0 - sub["raw_mean_miss_rate"])
    et = post[post["region"] == "ET"]
    # ET: a missed tumour costs 1, an empty-reference case costs 0 ("zero" convention), so the
    # mean is the share of cases that have an enhancing tumour (6 of 8 etc.), strictly in (0, 1).
    assert len(et) > 0
    assert ((et["mean_miss_rate"] > 0.5) & (et["mean_miss_rate"] < 1.0)).all()


def test_postprocess_cache_is_reused(world: dict[str, Any], run: dict[str, Any]) -> None:
    cache = Path(run["block"]["out_dir"]) / "postprocessed_counts"
    files = sorted(cache.glob("*.json"))
    assert len(files) == 2 * 3  # 2 logit models x 3 apply cohorts
    before = [f.stat().st_mtime_ns for f in files]
    v2.run_v2(run["block"], _full_cfg(0), Path("."))
    assert [f.stat().st_mtime_ns for f in files] == before


def test_run_meta_records_timings_and_cases(run: dict[str, Any]) -> None:
    meta = json.loads(run["paths"]["run_meta_json"].read_text())
    assert meta["grid_sizes"] == {
        "v1": len(DEFAULT_THRESHOLDS),
        "extended": len(EXTENDED_THRESHOLDS),
    }
    assert {"curves", "additivity", "fits", "realised", "resplits", "postprocessed"} <= set(
        meta["wall_time_s"]
    )
    assert meta["n_cases"]["m1"]["val"] == N_VAL
