"""Tests for scripts/risk_gate.py. Tiny synthetic (3, 8, 8, 8) logits, CPU only, no real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.script_loader import load_script

rg = load_script("risk_gate")

N_VAL = 60
N_COH = 20
BAD_VAL = [f"c{i:02d}" for i in range(54, 60)]  # 6 of 60 unusable, highest entropy
EMPTY_VAL = "c59"  # predicts nothing -> NaN entropy -> +inf
BAD_COH = ["c18", "c19"]  # unusable in test / ssa / ped; their NV test ENT is NaN
COHORTS = ("test", "ssa", "ped")


def _ids(n: int) -> list[str]:
    return [f"c{i:02d}" for i in range(n)]


def _logits(i: int, empty: bool, weak: bool) -> np.ndarray:
    """A blob; weak blobs have logits near 0 (high entropy), strong ones are confident."""
    out = np.full((3, 8, 8, 8), -8.0, dtype=np.float32)
    if not empty:
        out[:, 2:6, 2:6, 2:6] = 0.3 if weak else 4.0 + 0.05 * i  # distinct entropy per case
    return out.astype(np.float16)


def _write_eval_dir(root: Path, rel: str, n: int, bad: list[str], with_logits: bool = True) -> str:
    """Writes `<rel>/logits/*.npy` (optional) and `<rel>/per_case_metrics.csv`."""
    d = root / rel
    d.mkdir(parents=True)
    ids = _ids(n)
    if with_logits:
        (d / "logits").mkdir()
        for i, cid in enumerate(ids):
            np.save(d / "logits" / f"{cid}.npy", _logits(i, cid == EMPTY_VAL, cid in bad))
    pd.DataFrame(
        {
            "case_id": ids,
            "dice_WT": [0.5 if c in bad else 0.9 for c in ids],
            "dice_TC": 0.9,
        }
    ).to_csv(d / "per_case_metrics.csv", index=False)
    return rel


def _deployed_table(root: Path, rel: str, bad: list[str], accepted: list[str]) -> str:
    ids = _ids(N_COH)
    pd.DataFrame(
        {
            "case_id": ids,
            "accepted": [c in accepted for c in ids],
            "usable": [c not in bad for c in ids],
        }
    ).to_csv(root / rel, index=False)
    return rel


def _world(root: Path) -> dict:
    """Builds every input of the `risk_gate` block under `root`; returns the block."""
    members = [
        _write_eval_dir(root, f"m{k}_val", N_VAL, BAD_VAL) for k in range(3)
    ]  # identical members -> ENS_PD == 0 everywhere -> NV-ENSPD infeasible
    nv_dirs = {"val": members[0]}
    nn_dirs = {"val": _write_eval_dir(root, "nn_val", N_VAL, BAD_VAL)}
    nv_sig, nv_dep, nn_dep = {}, {}, {}
    good = [c for c in _ids(N_COH) if c not in BAD_COH]
    for cohort in COHORTS:
        nv_dirs[cohort] = _write_eval_dir(root, f"nv_{cohort}", N_COH, BAD_COH, with_logits=False)
        nn_dirs[cohort] = _write_eval_dir(root, f"nn_{cohort}", N_COH, BAD_COH)
        ent = [np.nan if c in BAD_COH else 1e-4 * i for i, c in enumerate(_ids(N_COH))]
        pd.DataFrame(
            {
                "case_id": _ids(N_COH),
                "ENT": ent,
                "ENS_ENT": ent,
                "ENS_PD": 0.0,
                "n_empty_wt_members": 0,
            }
        ).to_csv(root / f"sig_{cohort}.csv", index=False)
        nv_sig[cohort] = f"sig_{cohort}.csv"
        # deployed neurovision gate: accepts 15 usable cases only (coverage 0.75, risk 0)
        nv_dep[cohort] = _deployed_table(root, f"dep_nv_{cohort}.csv", BAD_COH, good[:15])
        nn_dep[cohort] = _deployed_table(root, f"dep_nn_{cohort}.csv", BAD_COH, good)
    return {
        "out_dir": "out",
        "betas": [0.05, 0.10],
        "delta": 0.1,
        "dice_bar": 0.7,
        "conf": 0.95,
        "neurovision": {
            "val_members": members,
            "eval_dirs": nv_dirs,
            "signals_tables": nv_sig,
            "deployed_tables": nv_dep,
        },
        "nnunet_gatea": {"eval_dirs": nn_dirs, "deployed_tables": nn_dep},
        "pipelines": {
            "NV-ENT": {"model": "neurovision", "signal": "ENT"},
            "NV-ENSPD": {"model": "neurovision", "signal": "ENS_PD"},
            "NN-ENT": {"model": "nnunet_gatea", "signal": "ENT"},
        },
        "deployment_rule_betas": [0.10],
        "case_limit": None,
        "overwrite": False,
    }


def _full():
    return OmegaConf.create(
        {
            "inference": {
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": 0,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                }
            }
        }
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):
    """One end-to-end run shared by the read-only assertions below."""
    root = tmp_path_factory.mktemp("rg")
    block = _world(root)
    paths = rg.run_risk_gate(block, _full(), root)
    return root, block, paths


def test_label_rule_is_strictly_below_the_bar() -> None:
    m = pd.DataFrame(
        {"dice_WT": [0.69, 0.7, 0.9, 0.9], "dice_TC": [0.9, 0.7, 0.69, np.nan]},
        index=["a", "b", "c", "d"],
    )
    u = rg.unusable_labels(m, 0.7)
    assert u.to_dict() == {"a": True, "b": False, "c": True, "d": True}


def test_nan_to_inf_counts() -> None:
    t = pd.DataFrame({"ENT": [0.1, np.nan, np.nan], "ENS_PD": [0.0, 0.1, 0.2]})
    counts = rg.nan_to_inf(t, ["ENT", "ENS_PD"])
    assert counts == {"ENT": 2, "ENS_PD": 0}
    assert np.isposinf(t["ENT"]).sum() == 2


def test_end_to_end_outputs_and_inf_counts(run) -> None:
    root, _, paths = run
    for p in paths.values():
        assert p.is_file()
    assert not list((root / "out").rglob("*.tmp"))
    summary = json.loads(paths["run_summary"].read_text())
    inf = summary["inf_replacements"]
    assert inf["neurovision"]["val"]["ENT"] == 1
    assert inf["nnunet_gatea"]["val"]["ENT"] == 1
    assert inf["neurovision"]["test"]["ENT"] == 2
    assert inf["nnunet_gatea"]["test"]["ENT"] == 0
    assert summary["case_counts"]["neurovision/val"] == {"n": N_VAL, "n_unusable": 6}
    assert "git_head" in summary and "wall_times_s" in summary


def test_fit_uses_validation_cases_only(run) -> None:
    _, _, paths = run
    fits = pd.read_csv(paths["fits"])
    assert len(fits) == 6  # 3 pipelines x 2 betas
    assert (fits["n_calibration"] == N_VAL).all()  # test/ssa/ped have 20 cases; never fitted
    ent = fits[(fits["pipeline"] == "NV-ENT") & (fits["beta"] == 0.10)].iloc[0]
    assert bool(ent["feasible"]) and np.isfinite(ent["lam"]) and ent["k_unusable"] == 0
    steps = pd.read_csv(paths["ltt_steps"])
    assert set(steps.columns) >= {"pipeline", "beta", "coverage", "p_value", "rejected"}


def test_infeasible_gate_is_reported_not_applied(run) -> None:
    _, _, paths = run
    fits = pd.read_csv(paths["fits"])
    cells = pd.read_csv(paths["cells"])
    # identical members: ENS_PD == 0 for every case, so the first step accepts all and fails.
    assert not fits[fits["pipeline"] == "NV-ENSPD"]["feasible"].any()
    enspd = cells[cells["pipeline"] == "NV-ENSPD"]
    assert len(enspd) == 6  # 2 betas x 3 cohorts
    assert (enspd["verdict"] == "gate_infeasible").all()
    assert enspd["lam"].isna().all() and enspd["coverage"].isna().all()
    assert (enspd["n"] == N_COH).all()
    assert len(cells) == 18


def test_frozen_application_numbers(run) -> None:
    _, _, paths = run
    cells = pd.read_csv(paths["cells"])
    row = cells[
        (cells["pipeline"] == "NV-ENT") & (cells["beta"] == 0.10) & (cells["cohort"] == "test")
    ].iloc[0]
    # all 18 usable test cases score far below lambda; the 2 NaN-entropy cases became +inf.
    assert row["n_accepted"] == 18 and row["k_unusable_accepted"] == 0
    assert row["coverage"] == pytest.approx(0.9)
    assert row["over_refusal"] == 0.0 and row["silent_failure"] == 0.0


def test_deployed_rows_and_e3_winner(run) -> None:
    _, _, paths = run
    dep = pd.read_csv(paths["deployed"])
    assert (dep["pipeline"] == "deployed").all()
    d = dep[(dep["model"] == "neurovision") & (dep["cohort"] == "test") & (dep["beta"] == 0.10)]
    assert d.iloc[0]["coverage"] == pytest.approx(0.75)
    e3 = json.loads(paths["run_summary"].read_text())["deployment_rule"]["0.1"]
    assert e3["pipelines"]["NV-ENT"]["qualifies"] is True  # 0.9 >= 0.75 and 0.0 <= 0.0
    assert e3["pipelines"]["NV-ENSPD"]["qualifies"] is False  # infeasible
    assert e3["winner"] == "NV-ENT"
    assert e3["deployed"]["coverage"] == pytest.approx(0.75)


def _cells(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pipeline": p,
                "cohort": "test",
                "beta": 0.1,
                "coverage": c,
                "selective_risk": r,
                "verdict": "holds",
            }
            for p, c, r in rows
        ]
    )


def _dep(cov: float, risk: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "neurovision",
                "cohort": "test",
                "beta": 0.1,
                "coverage": cov,
                "selective_risk": risk,
            }
        ]
    )


def test_deployment_rule_tie_higher_coverage_and_none() -> None:
    both = _cells([("NV-ENT", 0.8, 0.05), ("NV-ENSPD", 0.8, 0.05)])
    assert rg.deployment_rule(both, _dep(0.7, 0.1), [0.1])["0.1"]["reason"] == "tie"
    assert rg.deployment_rule(both, _dep(0.7, 0.1), [0.1])["0.1"]["winner"] is None
    better = _cells([("NV-ENT", 0.8, 0.05), ("NV-ENSPD", 0.9, 0.05)])
    assert rg.deployment_rule(better, _dep(0.7, 0.1), [0.1])["0.1"]["winner"] == "NV-ENSPD"
    # higher selective risk than deployed disqualifies, even with more coverage
    worse = _cells([("NV-ENT", 0.95, 0.2), ("NV-ENSPD", 0.6, 0.0)])
    out = rg.deployment_rule(worse, _dep(0.7, 0.1), [0.1])["0.1"]
    assert out["winner"] is None and out["reason"] == "none_qualify"


def test_deployed_usable_disagreeing_with_label_raises(tmp_path: Path) -> None:
    block = _world(tmp_path)
    path = tmp_path / block["neurovision"]["deployed_tables"]["ssa"]
    df = pd.read_csv(path)
    df.loc[0, "usable"] = not df.loc[0, "usable"]  # c00 is usable by Dice; flip it
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagrees"):
        rg.run_risk_gate(block, _full(), tmp_path)


def test_missing_member_val_logits_raises_naming_the_member(tmp_path: Path) -> None:
    block = _world(tmp_path)
    bad = block["neurovision"]["val_members"][2]
    for f in (tmp_path / bad / "logits").glob("*.npy"):
        f.unlink()
    (tmp_path / bad / "logits").rmdir()
    with pytest.raises(FileNotFoundError, match="m2_val"):
        rg.run_risk_gate(block, _full(), tmp_path)
    assert not (tmp_path / "out" / "fits.csv").exists()


def test_partial_member_logits_raise(tmp_path: Path) -> None:
    block = _world(tmp_path)
    (tmp_path / block["neurovision"]["val_members"][1] / "logits" / "c10.npy").unlink()
    with pytest.raises(ValueError, match="m1_val"):
        rg.run_risk_gate(block, _full(), tmp_path)


def test_signal_label_case_mismatch_raises() -> None:
    sig = pd.DataFrame({"ENT": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="case_id"))
    lab = pd.Series([False], index=["a"])
    with pytest.raises(ValueError, match="differ"):
        rg.join_labels(sig, lab, "x", allow_partial=False)
    assert len(rg.join_labels(sig, lab, "x", allow_partial=True)) == 1
