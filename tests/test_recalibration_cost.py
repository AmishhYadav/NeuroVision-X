"""Tests for `neurovision.analysis.recalibration_cost`.

CPU only, synthetic `CaseLossCurve`s, each test well under a second. Expected values for
`verdict`, `valid_ks`, `k_needed` and `annotation_hours` are worked out by hand.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from neurovision.analysis import recalibration_cost as rc
from neurovision.analysis.recalibration_cost import (
    CostSplitResult,
    annotation_hours,
    evaluate_cost_split,
    k_needed,
    run_cost,
    summarise_cost,
    valid_ks,
    verdict,
)
from neurovision.uncertainty.conformal import CaseLossCurve

GRID = tuple(float(t) for t in np.linspace(0.05, 0.95, 10))


def _cohort(n: int, seed: int, region: str = "WT") -> list[CaseLossCurve]:
    """Exchangeable cases: miss rate at tau is u * tau with u ~ U(0, 0.4); mask shrinks with tau."""
    rng = np.random.default_rng(seed)
    gt = 1000
    curves = []
    for i in range(n):
        u = rng.uniform(0.0, 0.4)
        fn = tuple(int(round(gt * u * t)) for t in GRID)
        mask = tuple(int(round(2 * gt * (1.0 - t))) + 10 for t in GRID)
        curves.append(CaseLossCurve(f"c{i:03d}", region, gt, GRID, fn, mask))
    return curves


def _run(curves, methods=("crc", "rcps"), ks=(60,), seed=0, alphas=(0.1,), n_splits=30):
    return run_cost(
        {("m", "test", "WT"): curves},
        alphas=alphas,
        methods=methods,
        k_grid=list(ks),
        n_splits=n_splits,
        min_holdout=10,
        delta=0.1,
        empty_by_region={"WT": "exclude"},
        grid_by_model={"m": "extended"},
        rng=np.random.default_rng(seed),
    )


def test_valid_ks_expands_half_caps_and_dedups() -> None:
    # n=40, min_holdout=10 -> k <= 30. "half" -> 20 duplicates the explicit 20.
    assert valid_ks([50, 20, 5, "half", 30, 31, 20], n=40, min_holdout=10) == [5, 20, 30]
    assert valid_ks([5], n=10, min_holdout=10) == []
    with pytest.raises(ValueError):
        valid_ks(["third"], n=40, min_holdout=10)


def test_verdict_boundaries() -> None:
    nan = float("nan")
    assert verdict(nan, nan, 0.1, 0) == "INFEASIBLE"
    assert verdict(0.01, nan, 0.1, 1) == "BORDERLINE"  # one draw is never decisive
    assert verdict(0.1, 0.01, 0.1, 5) == "RESTORED"  # exactly alpha counts as restored
    assert verdict(0.119, 0.01, 0.1, 5) == "BORDERLINE"  # excess 0.019 <= 2*0.01
    assert verdict(0.121, 0.01, 0.1, 5) == "NOT_RESTORED"  # excess 0.021 > 0.02
    assert verdict(nan, nan, 0.1, 5) == "BORDERLINE"


def test_crc_restored_and_rcps_more_conservative_on_exchangeable_data() -> None:
    results = _run(_cohort(200, seed=1))
    summ = summarise_cost(results).set_index("method")
    crc, rcps = summ.loc["crc"], summ.loc["rcps"]
    assert crc["verdict"] == "RESTORED"
    assert crc["feasible_rate"] == 1.0
    assert rcps["feasible_rate"] <= crc["feasible_rate"]
    # Lower tau = bigger mask = fewer misses, so "more conservative" means a LOWER threshold.
    assert rcps["mean_threshold"] <= crc["mean_threshold"]
    assert rcps["mean_realised_risk"] <= crc["mean_realised_risk"]


def test_splits_shared_between_methods_and_alphas() -> None:
    results = _run(_cohort(60, seed=2), alphas=(0.1, 0.2), ks=(20,), n_splits=5)
    # Re-run each split's fit directly: result keyed by (alpha, method) must share the split,
    # which we detect through identical held-out/calibration indices from the same rng stream.
    rng = np.random.default_rng(0)
    splits = [rng.permutation(60) for _ in range(5)]
    curves = _cohort(60, seed=2)
    for r in results:
        order = splits[r.split_index]
        direct = evaluate_cost_split(
            curves,
            order[:20].tolist(),
            order[20:].tolist(),
            alpha=r.alpha,
            method=r.method,
            delta=0.1,
            empty="exclude",
            model="m",
            cohort="test",
            split_index=r.split_index,
        )
        assert direct == r
    assert len(results) == 2 * 2 * 5


def test_deterministic_for_fixed_seed() -> None:
    curves = _cohort(60, seed=3)
    a = summarise_cost(_run(curves, ks=(10, 20), seed=7, n_splits=5))
    b = summarise_cost(_run(curves, ks=(10, 20), seed=7, n_splits=5))
    pd.testing.assert_frame_equal(a, b)


def test_infeasible_split_has_no_fitted_fields() -> None:
    # alpha so small that even the smallest tau cannot reach it with the CRC correction.
    r = evaluate_cost_split(
        _cohort(40, seed=4),
        list(range(10)),
        list(range(10, 40)),
        alpha=0.001,
        method="crc",
        delta=0.1,
        empty="exclude",
    )
    assert not r.feasible
    assert r.threshold is None and r.realised_risk is None and r.dice is None


def test_empty_prediction_gives_nan_precision_median_none() -> None:
    # Every mask empty (all GT voxels missed): precision is 0/0 -> median None; dice 0.
    curves = [
        CaseLossCurve(f"e{i}", "WT", 100, GRID, tuple([100] * 10), tuple([0] * 10))
        for i in range(12)
    ]
    r = evaluate_cost_split(
        curves,
        list(range(6)),
        list(range(6, 12)),
        alpha=1.0,
        method="crc",
        delta=0.1,
        empty="exclude",
    )
    assert r.feasible
    assert r.precision is None
    assert r.dice == 0.0
    assert r.realised_risk == 1.0 and r.violated is False  # every voxel missed; 1.0 > 1.0 is False


def _row(model, k, verdict_, rate, dice=0.5):
    return {
        "model": model,
        "cohort": "SSA",
        "region": "WT",
        "alpha": 0.1,
        "method": "crc",
        "k": k,
        "verdict": verdict_,
        "feasible_rate": rate,
        "mean_realised_risk": 0.05,
        "precision_median": 0.6,
        "dice_median": dice,
        "volume_ratio_median": 1.5,
    }


def test_k_needed_picks_smallest_restored_with_feasibility() -> None:
    summ = pd.DataFrame(
        [
            _row("a", 5, "RESTORED", 0.5),  # restored but infeasible too often
            _row("a", 10, "BORDERLINE", 1.0),
            _row("a", 20, "RESTORED", 0.9, dice=0.7),  # first that qualifies (rate == 0.90)
            _row("a", 30, "RESTORED", 1.0),
            _row("b", 5, "NOT_RESTORED", 1.0),
            _row("b", 10, "INFEASIBLE", 0.0),
        ]
    )
    out = k_needed(summ, min_feasible_rate=0.9).set_index("model")
    assert out.loc["a", "k_needed"] == 20 and out.loc["a", "restorable"]
    assert out.loc["a", "dice_median"] == 0.7
    assert math.isnan(out.loc["b", "k_needed"]) and not out.loc["b", "restorable"]
    assert math.isnan(out.loc["b", "dice_median"])


def test_annotation_hours() -> None:
    assert annotation_hours(30, 60.0) == 30.0
    assert annotation_hours(30, 60.0, 2.0) == 15.0
    assert annotation_hours(10, 90.0) == 15.0
    assert math.isnan(annotation_hours(float("nan"), 60.0))


def test_restrict_to_grid_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        rc.restrict_to_grid(_cohort(3, seed=0), "v3")


def test_result_is_frozen() -> None:
    r = _run(_cohort(30, seed=5), ks=(10,), n_splits=1)[0]
    assert isinstance(r, CostSplitResult)
    with pytest.raises(AttributeError):
        r.k = 3  # type: ignore[misc]
