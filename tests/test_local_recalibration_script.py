"""Tests for scripts/local_recalibration.py.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, the same pattern as
tests/test_conformal_script.py / tests/test_compare_family.py.

Everything here is synthetic and tiny: `CaseLossCurve`s are built by hand (no
BraTS data, no inference), `curves.npz` files are written directly with
`scripts/conformal.py`'s own `_write_curves_npz` (so the layout is guaranteed
to match what `load_curves_npz` reads), and `n_splits` is kept small (20-50)
purely for test speed. CPU only; whole file runs in well under three seconds.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from neurovision.analysis.local_recalibration import SplitResult
from neurovision.uncertainty.conformal import CaseLossCurve
from tests.script_loader import load_script

conformal_script = load_script("conformal")
local_recalibration_script = load_script("local_recalibration")

classify_restoration = local_recalibration_script.classify_restoration
run_local_recalibration = local_recalibration_script.run_local_recalibration
_write_curves_npz = conformal_script._write_curves_npz

THRESHOLDS = (0.01, 0.5, 0.9)


# ---------------------------------------------------------------------------
# Synthetic curve builders
# ---------------------------------------------------------------------------


def _make_curve(case_id: str, region: str, miss_rates: tuple[float, float, float]) -> CaseLossCurve:
    """One CaseLossCurve with gt_voxels=100 and a hand-picked, non-decreasing miss-rate curve
    over THRESHOLDS. mask_voxels is fixed and irrelevant to every quantity these tests check."""
    fn_voxels = tuple(int(round(m * 100)) for m in miss_rates)
    return CaseLossCurve(
        case_id=case_id,
        region=region,
        gt_voxels=100,
        thresholds=THRESHOLDS,
        fn_voxels=fn_voxels,
        mask_voxels=(100, 50, 10),
    )


def _make_cohort_curves(n_cases: int, region: str, tag: str, seed: int) -> list[CaseLossCurve]:
    """n_cases curves with mild per-case variability in miss rate, always non-decreasing."""
    rng = np.random.default_rng(seed)
    curves = []
    for i in range(n_cases):
        base = rng.uniform(0.0, 0.05)
        mid = base + rng.uniform(0.0, 0.05)
        high = mid + rng.uniform(0.0, 0.05)
        curves.append(_make_curve(f"{tag}_{i:03d}", region, (base, mid, high)))
    return curves


def _write_model_curves(
    conformal_dir: Path,
    subdir: str,
    n_cases: int,
    regions: tuple[str, ...],
    tag: str,
    seed: int,
) -> None:
    """Writes <conformal_dir>/<subdir>/curves.npz for the given regions, one call per
    (model, cohort) -- mirrors what scripts/conformal.py's extract_curves would have written."""
    out_dir = conformal_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_by_region = {
        region: _make_cohort_curves(n_cases, region, tag, seed + i)
        for i, region in enumerate(regions)
    }
    _write_curves_npz(curves_by_region, THRESHOLDS, out_dir / "curves.npz")


def _make_cfg(
    tmp_path: Path,
    models: dict,
    *,
    regions: list[str] | None = None,
    alphas: list[float] | None = None,
    ks: list | None = None,
    n_splits: int = 20,
    seed: int = 0,
) -> OmegaConf:
    """Minimal composed-looking cfg with only what run_local_recalibration reads."""
    return OmegaConf.create(
        {
            "analysis": {
                "local_recalibration": {
                    "out_dir": str(tmp_path / "out"),
                    "models": models,
                    "regions": regions or ["WT", "TC"],
                    "alphas": alphas or [0.2],
                    "ks": ks or [5, 10],
                    "n_splits": n_splits,
                    "seed": seed,
                }
            }
        }
    )


# ---------------------------------------------------------------------------
# 1. classify_restoration -- all four branches
# ---------------------------------------------------------------------------


def test_classify_restoration_all_branches() -> None:
    # INFEASIBLE: no feasible split at all, whatever the (irrelevant) risk/se say.
    assert classify_restoration(mean_risk=0.5, mc_se=0.01, alpha=0.2, n_feasible=0) == "INFEASIBLE"

    # RESTORED, including the boundary case mean_risk == alpha exactly.
    assert classify_restoration(mean_risk=0.2, mc_se=0.01, alpha=0.2, n_feasible=10) == "RESTORED"
    assert classify_restoration(mean_risk=0.1, mc_se=0.01, alpha=0.2, n_feasible=10) == "RESTORED"

    # NOT_RESTORED: exceeds alpha by (strictly) more than 2 * mc_se.
    assert (
        classify_restoration(mean_risk=0.3, mc_se=0.01, alpha=0.2, n_feasible=10) == "NOT_RESTORED"
    )

    # BORDERLINE: exceeds alpha, but not by more than 2 * mc_se.
    assert (
        classify_restoration(mean_risk=0.21, mc_se=0.01, alpha=0.2, n_feasible=10) == "BORDERLINE"
    )

    # BORDERLINE: n_feasible < 2 (mc_se undefined) must NEVER read as RESTORED or
    # NOT_RESTORED, however far mean_risk sits from alpha -- a single Monte Carlo draw
    # carries no usable estimate of the standard error.
    assert (
        classify_restoration(mean_risk=0.05, mc_se=float("nan"), alpha=0.2, n_feasible=1)
        == "BORDERLINE"
    )
    assert (
        classify_restoration(mean_risk=0.9, mc_se=float("nan"), alpha=0.2, n_feasible=1)
        == "BORDERLINE"
    )


# ---------------------------------------------------------------------------
# 2. Every output file is written, with the expected columns.
# ---------------------------------------------------------------------------


def test_run_writes_all_outputs(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=20, regions=("WT", "TC"), tag="test", seed=1)

    models = {
        "neurovision": {
            "conformal_dir": str(conf_dir),
            "cohorts": {"test": "eval_test"},
        }
    }
    cfg = _make_cfg(tmp_path, models, alphas=[0.2, 0.3], ks=[5, 10, "half"])

    paths = run_local_recalibration(cfg)

    assert Path(paths["floors_csv"]).is_file()
    assert Path(paths["splits_csv_gz"]).is_file()
    assert Path(paths["summary_csv"]).is_file()
    assert Path(paths["run_meta_json"]).is_file()

    floors = pd.read_csv(paths["floors_csv"])
    for col in [
        "model",
        "cohort",
        "region",
        "alpha",
        "n_cases",
        "min_feasible_k",
        "structural_floor",
    ]:
        assert col in floors.columns
    assert len(floors) == 2 * 2  # 2 regions x 2 alphas, one model one cohort

    with gzip.open(paths["splits_csv_gz"], "rt") as fh:
        splits = pd.read_csv(fh)
    for col in [
        "model",
        "cohort",
        "region",
        "alpha",
        "k",
        "split_index",
        "n_calibration",
        "n_heldout",
        "feasible",
        "threshold",
        "realised_risk",
        "violated",
        "inflation",
    ]:
        assert col in splits.columns
    assert (splits["model"] == "neurovision").all()

    summary = pd.read_csv(paths["summary_csv"])
    for col in [
        "model",
        "cohort",
        "region",
        "alpha",
        "k",
        "n_splits",
        "feasible_rate",
        "mean_realised_risk",
        "n_feasible",
        "mc_se",
        "is_half",
        "verdict",
    ]:
        assert col in summary.columns
    assert set(summary["verdict"]).issubset(
        {"RESTORED", "NOT_RESTORED", "BORDERLINE", "INFEASIBLE"}
    )

    meta = json.loads(Path(paths["run_meta_json"]).read_text())
    for key in [
        "git_sha",
        "seed",
        "n_splits",
        "config",
        "timestamp_utc",
        "status_label",
        "control_failed",
    ]:
        assert key in meta
    assert meta["status_label"].startswith("COUNTERFACTUAL")
    assert meta["seed"] == 0
    assert meta["n_splits"] == 20


# ---------------------------------------------------------------------------
# 3. Same seed -> same splits across two models on the same cohort (paired).
# ---------------------------------------------------------------------------


def test_same_seed_same_splits_across_models(tmp_path: Path) -> None:
    conf_dir_a = tmp_path / "conformal_a"
    conf_dir_b = tmp_path / "conformal_b"
    # Byte-identical curves.npz content under both roots, tag/seed matched exactly.
    _write_model_curves(conf_dir_a, "eval_test", n_cases=16, regions=("WT",), tag="test", seed=7)
    _write_model_curves(conf_dir_b, "eval_test", n_cases=16, regions=("WT",), tag="test", seed=7)

    models = {
        "model_a": {"conformal_dir": str(conf_dir_a), "cohorts": {"test": "eval_test"}},
        "model_b": {"conformal_dir": str(conf_dir_b), "cohorts": {"test": "eval_test"}},
    }
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[5, 8], n_splits=25, seed=3)

    paths = run_local_recalibration(cfg)
    with gzip.open(paths["splits_csv_gz"], "rt") as fh:
        splits = pd.read_csv(fh)

    key_cols = ["cohort", "region", "alpha", "k", "split_index"]
    a = splits[splits["model"] == "model_a"].sort_values(key_cols).reset_index(drop=True)
    b = splits[splits["model"] == "model_b"].sort_values(key_cols).reset_index(drop=True)

    assert len(a) == len(b) > 0
    pd.testing.assert_series_equal(
        a["realised_risk"], b["realised_risk"], check_names=False, check_exact=False
    )
    pd.testing.assert_series_equal(
        a["threshold"], b["threshold"], check_names=False, check_exact=False
    )
    pd.testing.assert_series_equal(
        a["n_calibration"], b["n_calibration"], check_names=False, check_exact=False
    )


# ---------------------------------------------------------------------------
# 4. A missing curves.npz skips only its own (model, cohort), not the run.
# ---------------------------------------------------------------------------


def test_missing_cohort_is_skipped_not_fatal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=14, regions=("WT",), tag="test", seed=2)
    # "eval_ssa" is configured but its curves.npz is never written.

    models = {
        "neurovision": {
            "conformal_dir": str(conf_dir),
            "cohorts": {"test": "eval_test", "ssa": "eval_ssa"},
        }
    }
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[5])

    missing_path = conf_dir / "eval_ssa" / "curves.npz"
    with caplog.at_level(logging.WARNING):
        paths = run_local_recalibration(cfg)
    assert any(
        "curves.npz" in rec.message and str(missing_path) in rec.message for rec in caplog.records
    )

    summary = pd.read_csv(paths["summary_csv"])
    assert set(summary["cohort"]) == {"test"}

    with gzip.open(paths["splits_csv_gz"], "rt") as fh:
        splits = pd.read_csv(fh)
    assert set(splits["cohort"]) == {"test"}


# ---------------------------------------------------------------------------
# 5. is_half is true exactly for k == n // 2.
# ---------------------------------------------------------------------------


def test_half_k_flag(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    n_cases = 20  # n // 2 == 10
    _write_model_curves(conf_dir, "eval_test", n_cases=n_cases, regions=("WT",), tag="test", seed=4)

    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[5, 10, 15])

    paths = run_local_recalibration(cfg)
    summary = pd.read_csv(paths["summary_csv"])

    is_half_by_k = summary.groupby("k")["is_half"].first()
    assert bool(is_half_by_k.loc[10]) is True
    assert bool(is_half_by_k.loc[5]) is False
    assert bool(is_half_by_k.loc[15]) is False


def test_half_string_dedupes_against_explicit_matching_k(tmp_path: Path) -> None:
    """ks=[10, "half"] on a cohort with n=20 (half=10) draws k=10's splits exactly once."""
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=20, regions=("WT",), tag="test", seed=5)

    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[10, "half"], n_splits=15)

    paths = run_local_recalibration(cfg)
    summary = pd.read_csv(paths["summary_csv"])

    rows_k10 = summary[summary["k"] == 10]
    assert len(rows_k10) == 1
    assert int(rows_k10.iloc[0]["n_splits"]) == 15  # not doubled to 30
    assert bool(rows_k10.iloc[0]["is_half"]) is True


# ---------------------------------------------------------------------------
# 6. The control-cohort falsifier: a NOT_RESTORED verdict on TEST at k=half
#    sets control_failed=true and logs an ERROR.
#
# A genuine violation of conformal risk control's finite-sample guarantee on a
# TRULY random within-cohort split could not be constructed by hand (an
# exhaustive search over bimodal populations, alphas and seeds never produced
# one) -- which is exactly the theorem doing its job, and exactly why Amendment
# 1 calls a control failure evidence of an IMPLEMENTATION bug. So this test
# exercises the driver's OWN detection/wiring directly, by monkeypatching
# classify_restoration to force the one verdict that must never legitimately
# occur for the control cohort, rather than relying on statistical chance.
# ---------------------------------------------------------------------------


def test_control_failure_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=20, regions=("WT",), tag="test", seed=6)

    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[10], n_splits=15)

    monkeypatch.setattr(
        local_recalibration_script, "classify_restoration", lambda *a, **k: "NOT_RESTORED"
    )

    with caplog.at_level(logging.ERROR):
        paths = run_local_recalibration(cfg)
    assert any("control failed" in rec.message for rec in caplog.records)

    meta = json.loads(Path(paths["run_meta_json"]).read_text())
    assert meta["control_failed"] is True


def test_control_not_failed_in_the_ordinary_case(tmp_path: Path) -> None:
    """Sanity companion to the failure test: well-behaved curves at a generous alpha do NOT
    trip the control-cohort falsifier."""
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=20, regions=("WT",), tag="test", seed=6)
    # (constructed with the same recipe as the failure test above, just without the
    # monkeypatch -- confirms the ordinary path stays control_failed=False.)
    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[10], n_splits=15)

    paths = run_local_recalibration(cfg)
    meta = json.loads(Path(paths["run_meta_json"]).read_text())
    assert meta["control_failed"] is False


# ---------------------------------------------------------------------------
# 7. Determinism: two runs from the same config produce an identical summary.csv.
# ---------------------------------------------------------------------------


def test_deterministic_rerun(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    _write_model_curves(conf_dir, "eval_test", n_cases=18, regions=("WT", "TC"), tag="test", seed=8)

    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg1 = _make_cfg(tmp_path / "run1", models, alphas=[0.1, 0.2], ks=[5, 9])
    cfg2 = _make_cfg(tmp_path / "run2", models, alphas=[0.1, 0.2], ks=[5, 9])

    paths1 = run_local_recalibration(cfg1)
    paths2 = run_local_recalibration(cfg2)

    summary1 = pd.read_csv(paths1["summary_csv"]).sort_values(["cohort", "region", "alpha", "k"])
    summary2 = pd.read_csv(paths2["summary_csv"]).sort_values(["cohort", "region", "alpha", "k"])
    pd.testing.assert_frame_equal(summary1.reset_index(drop=True), summary2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# 8. mc_se divides by the number of splits it was actually computed from, not
#    n_feasible -- a feasible split can still have realised_risk=None (its
#    held-out half was entirely empty-ground-truth).
# ---------------------------------------------------------------------------


def _split(*, split_index: int, feasible: bool, realised_risk: float | None) -> SplitResult:
    """A minimal SplitResult for one (cohort=test, region=WT, alpha=0.2, k=5) cell."""
    return SplitResult(
        cohort="test",
        region="WT",
        alpha=0.2,
        k=5,
        split_index=split_index,
        n_calibration=5,
        n_heldout=5,
        feasible=feasible,
        threshold=0.5 if feasible else None,
        realised_risk=realised_risk,
        violated=(realised_risk is not None and realised_risk > 0.2) if feasible else None,
        inflation=1.0 if feasible else None,
    )


def test_mc_se_divides_by_defined_risks_not_n_feasible() -> None:
    # 3 feasible splits; the third's held-out half was entirely empty-GT, so its
    # realised_risk is None even though the split itself was feasible.
    results = [
        _split(split_index=0, feasible=True, realised_risk=0.10),
        _split(split_index=1, feasible=True, realised_risk=0.30),
        _split(split_index=2, feasible=True, realised_risk=None),
    ]
    augmented = local_recalibration_script._augment_summary(results)
    assert len(augmented) == 1
    row = augmented.iloc[0]

    assert int(row["n_feasible"]) == 3  # all three splits found a threshold
    expected_se = float(np.std([0.10, 0.30], ddof=1) / np.sqrt(2))  # divide by 2, not 3
    assert row["mc_se"] == pytest.approx(expected_se)


def test_mc_se_nan_when_fewer_than_two_defined_risks() -> None:
    results = [
        _split(split_index=0, feasible=True, realised_risk=0.10),
        _split(split_index=1, feasible=True, realised_risk=None),
        _split(split_index=2, feasible=False, realised_risk=None),
    ]
    augmented = local_recalibration_script._augment_summary(results)
    row = augmented.iloc[0]
    assert int(row["n_feasible"]) == 2  # splits 0 and 1 were feasible; 2 was not
    assert np.isnan(row["mc_se"])  # only ONE defined realised_risk (split 0)


# ---------------------------------------------------------------------------
# 9. Cross-model case_id-order pairing is asserted before any split is drawn.
# ---------------------------------------------------------------------------


def test_case_id_order_mismatch_across_models_raises(tmp_path: Path) -> None:
    conf_dir_a = tmp_path / "conformal_a"
    conf_dir_b = tmp_path / "conformal_b"
    ids = [f"test_{i:03d}" for i in range(10)]

    curves_a = {"WT": [_make_curve(cid, "WT", (0.0, 0.01, 0.02)) for cid in ids]}
    # Same SET of case ids, but in a different order -- draw_split indexes into this
    # list positionally, so a reorder silently un-pairs the two models' splits.
    curves_b = {"WT": [_make_curve(cid, "WT", (0.0, 0.01, 0.02)) for cid in reversed(ids)]}

    (conf_dir_a / "eval_test").mkdir(parents=True)
    (conf_dir_b / "eval_test").mkdir(parents=True)
    _write_curves_npz(curves_a, THRESHOLDS, conf_dir_a / "eval_test" / "curves.npz")
    _write_curves_npz(curves_b, THRESHOLDS, conf_dir_b / "eval_test" / "curves.npz")

    models = {
        "model_a": {"conformal_dir": str(conf_dir_a), "cohorts": {"test": "eval_test"}},
        "model_b": {"conformal_dir": str(conf_dir_b), "cohorts": {"test": "eval_test"}},
    }
    cfg = _make_cfg(tmp_path, models, regions=["WT"], alphas=[0.2], ks=[5])

    with pytest.raises(ValueError, match="case_id order mismatch"):
        run_local_recalibration(cfg)


# ---------------------------------------------------------------------------
# 10. A cohort whose regions disagree on case count is rejected before any
#     split is drawn (min_feasible_k/"half" both assume one shared n).
# ---------------------------------------------------------------------------


def test_region_case_count_mismatch_raises(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conformal" / "neurovision"
    out_dir = conf_dir / "eval_test"
    out_dir.mkdir(parents=True)
    curves = {
        "WT": [_make_curve(f"wt_{i:03d}", "WT", (0.0, 0.01, 0.02)) for i in range(10)],
        "TC": [_make_curve(f"tc_{i:03d}", "TC", (0.0, 0.01, 0.02)) for i in range(8)],
    }
    _write_curves_npz(curves, THRESHOLDS, out_dir / "curves.npz")

    models = {"neurovision": {"conformal_dir": str(conf_dir), "cohorts": {"test": "eval_test"}}}
    cfg = _make_cfg(tmp_path, models, regions=["WT", "TC"], alphas=[0.2], ks=[5])

    with pytest.raises(ValueError, match="different case count per region"):
        run_local_recalibration(cfg)
