"""Tests for scripts/failure_detection.py. Tiny synthetic (3, 8, 8, 8) logits, CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tests.script_loader import load_script

fd = load_script("failure_detection")

N_CASES = 12
CASES = [f"c{i:02d}" for i in range(N_CASES)]
EMPTY_CASE = "c11"  # every member predicts nothing -> NaN entropy -> +inf
UNUSABLE = {"c00", "c01", "c02", "c03", "c04", "c05"}  # 6 unusable / 6 usable


def _logits(case_idx: int, member: int, identical: bool) -> np.ndarray:
    """A small tumour blob; non-identical members get a different amount of noise."""
    rng = np.random.default_rng(case_idx + (0 if identical else 100 * member))
    logits = np.full((3, 8, 8, 8), -8.0, dtype=np.float32)
    logits[:, 2:6, 2:6, 2:6] = 4.0
    if CASES[case_idx] == EMPTY_CASE:
        return logits.astype(np.float16) * 0 - 8.0
    logits += rng.normal(0.0, 0.0 if identical else 3.0, logits.shape).astype(np.float32)
    return logits.astype(np.float16)


def _world(root: Path, identical: bool = True, drop_from_member: int | None = None) -> dict:
    """Writes 3 members, qc / ood / volume tables; returns a `failure_detection` block."""
    members = []
    for m in range(3):
        d = root / f"member{m}"
        (d / "logits").mkdir(parents=True)
        for i, cid in enumerate(CASES):
            if m == drop_from_member and cid == "c05":
                continue
            np.save(d / "logits" / f"{cid}.npy", _logits(i, m, identical))
        members.append(f"member{m}")
    metrics = pd.DataFrame(
        {
            "case_id": CASES,
            "dice_WT": [0.5 if c in UNUSABLE else 0.9 for c in CASES],
            "dice_TC": 0.9,
        }
    )
    metrics.to_csv(root / "member0" / "per_case_metrics.csv", index=False)
    rng = np.random.default_rng(0)
    qc = pd.DataFrame(
        {"case_id": CASES, "qc_pred_WT": rng.uniform(0.3, 0.9, N_CASES), "qc_pred_TC": 0.8}
    )
    qc.to_csv(root / "qc.csv", index=False)
    pd.DataFrame({"case_id": CASES, "ood_score": rng.uniform(0, 5, N_CASES)}).to_csv(
        root / "ood.csv", index=False
    )
    vol = pd.DataFrame({"case_id": CASES, "usable_volume": [c not in UNUSABLE for c in CASES]})
    vol.loc[[0, 7], "usable_volume"] = [True, False]  # make T-Vol differ from T-Dice
    vol.to_csv(root / "volumes.csv", index=False)
    return {
        "out_dir": "out",
        "cohorts": {
            "toy": {
                "members": members,
                "qc_table": "qc.csv",
                "ood_table": "ood.csv",
                "volume_table": "volumes.csv",
            }
        },
        "signals": ["QC", "ENS-PD", "ENS-ENT", "OOD"],
        "baseline_signal": "ENT",
        "dice_bar": 0.7,
        "min_each": 3,
        "n_boot": 30,
        "boot_seed": 1,
        "ci": 0.95,
        "alpha": 0.05,
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


def test_identical_members_have_zero_disagreement_and_ens_ent_equals_ent(tmp_path: Path) -> None:
    block = _world(tmp_path)
    paths = fd.run_failure_detection(block, _full(), tmp_path)
    sig = pd.read_csv(tmp_path / "out" / "signals_toy.csv").set_index("case_id")
    assert list(sig.columns) == ["ENT", "ENS_ENT", "ENS_PD", "n_empty_wt_members"]
    assert (sig["ENS_PD"] == 0.0).all()
    ok = sig.drop(index=EMPTY_CASE)
    np.testing.assert_allclose(ok["ENS_ENT"], ok["ENT"], rtol=1e-5)
    assert sig.loc[EMPTY_CASE, "n_empty_wt_members"] == 3
    assert sig.loc[[c for c in CASES if c != EMPTY_CASE], "n_empty_wt_members"].eq(0).all()
    for p in paths.values():
        assert p.is_file()
    assert not list((tmp_path / "out").rglob("*.tmp"))


def test_different_members_disagree(tmp_path: Path) -> None:
    block = _world(tmp_path, identical=False)
    fd.run_failure_detection(block, _full(), tmp_path)
    sig = pd.read_csv(tmp_path / "out" / "signals_toy.csv").set_index("case_id")
    assert (sig["ENS_PD"] > 0).any()
    assert ((sig["ENS_PD"] >= 0) & (sig["ENS_PD"] <= 1)).all()


def test_nan_entropy_becomes_inf_and_is_recorded(tmp_path: Path) -> None:
    block = _world(tmp_path)
    paths = fd.run_failure_detection(block, _full(), tmp_path)
    summary = json.loads(paths["run_summary"].read_text())
    toy = summary["cohorts"]["toy"]
    assert toy["n_cases"] == N_CASES
    assert toy["n_replaced_inf_ENT"] == 1
    assert toy["n_replaced_inf_ENS_ENT"] == 1
    assert "n_cells_not_estimable" in summary
    metrics = pd.read_csv(paths["metrics"])
    assert np.isfinite(metrics["auroc"]).all()  # +inf survived the rank-based metrics


def test_outputs_family_shape_and_not_estimable_marking(tmp_path: Path) -> None:
    block = _world(tmp_path)
    block["min_each"] = 7  # 6 unusable < 7 -> T-Dice cell not estimable
    paths = fd.run_failure_detection(block, _full(), tmp_path)
    family = pd.read_csv(paths["family"])
    assert len(family) <= 4 * 1 * 2
    assert len(family) == 4 * 2  # one row per signal x target, estimable or not
    dice = family[family["target"] == "T-Dice"]
    assert not dice["estimable"].any()
    assert (dice["verdict"] == "not_estimable").all()
    assert dice["p_holm"].isna().all()
    desc = pd.read_csv(paths["aurc_descriptive"])
    assert (desc["role"] == "descriptive").all()
    spearman = pd.read_csv(paths["spearman"])
    assert set(spearman["signal"]) == {"ENS-PD", "ENS-ENT"}

    block = _world(tmp_path / "b")
    paths = fd.run_failure_detection(block, _full(), tmp_path / "b")
    family = pd.read_csv(paths["family"])
    assert family["estimable"].all()
    assert set(family["verdict"]) <= {"better", "worse", "inconclusive"}
    metrics = pd.read_csv(paths["metrics"])
    assert len(metrics) == 5 * 2  # ENT baseline + 4 signals, 2 targets
    assert {"auroc", "aurc", "oracle_aurc", "random_aurc"} <= set(metrics.columns)


def test_resume_skips_existing_signals_and_overwrite_recomputes(tmp_path: Path) -> None:
    block = _world(tmp_path)
    fd.run_failure_detection(block, _full(), tmp_path)
    path = tmp_path / "out" / "signals_toy.csv"
    marker = 0.123456  # a valid edited value proves the file is kept / rewritten
    edited = pd.read_csv(path)
    edited.loc[0, "ENT"] = marker
    edited.to_csv(path, index=False)
    fd.run_failure_detection(block, _full(), tmp_path)
    assert pd.read_csv(path).loc[0, "ENT"] == pytest.approx(marker)
    block["overwrite"] = True
    fd.run_failure_detection(block, _full(), tmp_path)
    assert pd.read_csv(path).loc[0, "ENT"] != pytest.approx(marker)


def test_missing_volume_table_raises_naming_w25(tmp_path: Path) -> None:
    block = _world(tmp_path)
    (tmp_path / "volumes.csv").unlink()
    with pytest.raises(FileNotFoundError, match="W2.5"):
        fd.run_failure_detection(block, _full(), tmp_path)


def test_case_missing_from_non_deployed_member_raises(tmp_path: Path) -> None:
    block = _world(tmp_path, drop_from_member=2)
    with pytest.raises(ValueError, match="every case"):
        fd.run_failure_detection(block, _full(), tmp_path)


def test_missing_qc_value_raises(tmp_path: Path) -> None:
    block = _world(tmp_path)
    qc = pd.read_csv(tmp_path / "qc.csv")
    qc.loc[3, "qc_pred_WT"] = np.nan
    qc.to_csv(tmp_path / "qc.csv", index=False)
    with pytest.raises(ValueError, match="missing"):
        fd.run_failure_detection(block, _full(), tmp_path)


def test_case_limit_limits_cases(tmp_path: Path) -> None:
    _world(tmp_path)
    fd.run_stage1(
        "toy",
        [tmp_path / f"member{m}" for m in range(3)],
        tmp_path / "s.csv",
        0.5,
        fd._resolve_postprocess_cfg(
            {"enforce_nesting": True, "min_component_size": 0, "keep_largest_only": False}
        ),
        case_limit=4,
    )
    assert len(pd.read_csv(tmp_path / "s.csv")) == 4
