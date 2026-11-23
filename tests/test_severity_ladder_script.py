"""Tests for scripts/severity_ladder.py. Synthetic 16^3 volumes and a tiny U-Net, CPU only.

No real BraTS data, no GPU, no nnunetv2. The in-house backend runs a real (untrained) tiny
`unet3d` through the real sliding-window path; the nnU-Net backend is tested at the glue level
with `run_prediction` monkeypatched by a fake that writes what nnU-Net writes.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from neurovision.models.registry import build_model
from neurovision.training.checkpoint import save_checkpoint
from neurovision.uncertainty.conformal import EXTENDED_THRESHOLDS, load_curves_npz
from neurovision.utils.io import write_json
from tests.script_loader import load_script

ladder = load_script("severity_ladder")

SHAPE = (16, 16, 16)
N_CASES = 6  # probe (5) must be strictly fewer than the split
CASES = [f"case{i}" for i in range(N_CASES)]
TINY_MODEL = {
    "name": "unet3d",
    "in_channels": 4,
    "out_channels": 3,
    "channels": [4, 8],
    "strides": [2],
    "num_res_units": 1,
    "norm": "instance",
    "activation": "leakyrelu",
    "dropout": 0.0,
    "deep_supervision": False,
}
KINDS_SMALL = {"rician": [0.1, 0.2], "motion": [1, 2]}


def _write_cohort(prep: Path) -> None:
    """Writes image.npy (fp16, all-brain), label.npy and meta.json per case."""
    for i, case in enumerate(CASES):
        d = prep / case
        d.mkdir(parents=True)
        rng = np.random.default_rng(i)
        np.save(d / "image.npy", rng.standard_normal((4, *SHAPE)).astype(np.float16))
        label = np.zeros(SHAPE, dtype=np.uint8)
        label[3:13, 3:13, 3:13] = 2
        label[5:11, 5:11, 5:11] = 1
        label[7:9, 7:9, 7:9] = 3
        np.save(d / "label.npy", label)
        affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        write_json(
            {
                "spacing": [1.0, 1.0, 1.0],
                "affine": affine.tolist(),
                "bbox": [[0, 16], [0, 16], [0, 16]],
                "has_label": True,
            },
            d / "meta.json",
        )


def _make_cfg(tmp_path: Path, **overrides: Any) -> DictConfig:
    """A small config shaped like the composed one; `overrides` merge into severity_ladder."""
    # Like the real model YAMLs, in_channels is an interpolation into the rest of the config.
    model_yaml = {**TINY_MODEL, "in_channels": "${data.in_channels}"}
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(model_yaml), encoding="utf-8")
    (tmp_path / "splits.yaml").write_text(
        yaml.safe_dump({"train": ["x"], "val": ["y"], "test": CASES}), encoding="utf-8"
    )
    sl: dict[str, Any] = {
        "out_dir": str(tmp_path / "out"),
        "split": "test",
        "prep_dir": str(tmp_path / "prep"),
        "splits_path": str(tmp_path / "splits.yaml"),
        "base_seed": 42,
        "case_limit": 2,
        "probe": False,
        "probe_cases": 5,
        "models_to_run": ["m1"],
        "bridge_tol": 0.003,
        "max_hours": None,
        "kinds": KINDS_SMALL,
        "skip_levels": [],
        "skip_model_kinds": {},
        "gate": {"regions": ["WT", "TC"], "conformal_alpha": 0.1},
        "qc": {"enabled": False, "checkpoint": None, "model_config": None},
        "nnunet": {},
        "models": {
            "m1": {
                "backend": "inhouse",
                "model_config": str(tmp_path / "model.yaml"),
                "checkpoint": str(tmp_path / "ckpt" / "best.pt"),
                "reference_metrics": str(tmp_path / "ref.csv"),
                "conformal_fit": None,
            }
        },
    }
    sl.update(overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "data": {"in_channels": 4},
            "model": TINY_MODEL,
            "inference": {
                "sliding_window": {
                    "roi_size": [16, 16, 16],
                    "sw_batch_size": 1,
                    "overlap": 0.25,
                    "mode": "constant",
                    "sigma_scale": 0.125,
                    "output_device": None,
                    "padding_mode": "constant",
                },
                "postprocess": {
                    "threshold": 0.5,
                    "enforce_nesting": True,
                    "min_component_size": 0,
                    "connectivity": 1,
                    "keep_largest_only": False,
                    "et_min_volume": 0,
                },
                "evaluation": {"strict_arch_check": True},
            },
            "analysis": {
                "qc": {"modality_index": 1, "target_shape": [8, 8, 8]},
                "severity_ladder": sl,
            },
        }
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cohort + tiny checkpoint + a bridge reference made by a clean run of the same model."""
    root = tmp_path_factory.mktemp("ladder_world")
    _write_cohort(root / "prep")
    cfg = _make_cfg(root)
    model = build_model(cfg)
    save_checkpoint(
        root / "ckpt",
        model,
        torch.optim.Adam(model.parameters()),
        epoch=0,
        global_step=0,
        best_metric=0.5,
        best_metric_name="val/dice_mean",
        best_metric_mode="max",
        cfg=cfg,
        is_best=True,
    )
    # Reference = this model's own clean per-case metrics (what a saved clean eval would hold).
    pd.DataFrame({"case_id": CASES, "dice_ET": 0.0, "dice_TC": 0.0, "dice_WT": 0.0}).to_csv(
        root / "ref.csv", index=False
    )
    ref_cfg = _make_cfg(root, out_dir=str(root / "refrun"), case_limit=N_CASES, kinds={})
    ladder.run_severity_ladder(ref_cfg, root)
    clean = pd.read_csv(root / "refrun" / "m1" / "none" / "level0" / "per_case.csv")
    clean[["case_id", "dice_ET", "dice_TC", "dice_WT"]].to_csv(root / "ref.csv", index=False)
    return root


def _run(world: Path, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    cfg = _make_cfg(world, out_dir=str(tmp_path / "out"), **overrides)
    return ladder.run_severity_ladder(cfg, world)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_runs_level0_first_then_kind_level_model() -> None:
    sl = {"kinds": {"rician": [0.1, 0.2], "bias": [0.1]}, "skip_levels": [], "skip_model_kinds": {}}
    shards = ladder.plan_shards(sl, ["a", "b"])
    names = [(s.model, s.kind, s.level_index) for s in shards]
    assert names[:2] == [("a", "none", 0), ("b", "none", 0)]
    assert names[2:] == [
        ("a", "rician", 1),
        ("b", "rician", 1),
        ("a", "rician", 2),
        ("b", "rician", 2),
        ("a", "bias", 1),
        ("b", "bias", 1),
    ]


def test_plan_skips_keep_seed_level_index_stable() -> None:
    sl = {
        "kinds": {"rician": [0.1, 0.2], "bias": [0.1, 0.2]},
        "skip_levels": [1],
        "skip_model_kinds": {"b": ["bias"]},
    }
    shards = ladder.plan_shards(sl, ["a", "b"])
    corrupted = [(s.model, s.kind, s.level_index, s.severity) for s in shards if s.kind != "none"]
    assert corrupted == [("a", "rician", 2, 0.2), ("b", "rician", 2, 0.2), ("a", "bias", 2, 0.2)]


def test_plan_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        ladder.plan_shards({"kinds": {"blur": [1]}}, ["a"])


# ---------------------------------------------------------------------------
# Shard layout, columns, curves format
# ---------------------------------------------------------------------------


def test_shard_layout_columns_and_curves_loader(world: Path, tmp_path: Path) -> None:
    summary = _run(world, tmp_path, kinds={"rician": [0.1]})
    assert summary["bridge"] == {"m1": True}
    shard = tmp_path / "out" / "m1" / "rician" / "level1"
    for name in (
        "per_case.csv",
        "curves.npz",
        "curves_manifest.json",
        "ood_features.csv",
        "shard_meta.json",
        "DONE",
    ):
        assert (shard / name).is_file(), name
    assert (tmp_path / "out" / "m1" / "none" / "level0" / "DONE").is_file()
    assert not list((tmp_path / "out").rglob("*.partial"))
    assert not list((tmp_path / "out").rglob("*.npy"))  # never a volume

    df = pd.read_csv(shard / "per_case.csv")
    assert list(df["case_id"]) == CASES[:2]
    expected = {
        "model",
        "kind",
        "level_index",
        "severity",
        "seed",
        "seconds",
        "voxel_mm3",
        "ent_mean_fg_mean",
        "predicted_dice_WT",
        "conformal_band_TC",
    }
    for region in ("ET", "TC", "WT"):
        expected |= {
            f"dice_{region}",
            f"hd95_{region}",
            f"pred_voxels_{region}",
            f"gt_voxels_{region}",
            f"ent_mean_fg_{region}",
        }
    assert expected <= set(df.columns)
    assert (df["kind"] == "rician").all() and (df["level_index"] == 1).all()
    # QC and conformal fit are not configured here: their columns exist and are NaN, not absent.
    assert df["predicted_dice_WT"].isna().all() and df["conformal_band_TC"].isna().all()
    assert len(pd.read_csv(shard / "ood_features.csv").columns) == 47  # case_id + 46 features

    # The existing conformal loader reads the file unchanged, on the v2 grid.
    curves = load_curves_npz(shard / "curves.npz", ["ET", "TC", "WT"])
    assert [c.case_id for c in curves["WT"]] == CASES[:2]
    assert curves["WT"][0].thresholds == EXTENDED_THRESHOLDS
    wt_gt = int((np.load(world / "prep" / CASES[0] / "label.npy") > 0).sum())
    assert curves["WT"][0].gt_voxels == wt_gt
    assert curves["WT"][0].mask_voxels[0] >= curves["WT"][0].mask_voxels[-1]  # monotone in tau


def test_seeds_follow_case_seed(world: Path, tmp_path: Path) -> None:
    from neurovision.data.corruptions import case_seed

    _run(world, tmp_path, kinds={"motion": [1, 2]})
    df = pd.read_csv(tmp_path / "out" / "m1" / "motion" / "level2" / "per_case.csv")
    assert df["seed"][0] == case_seed(42, CASES[0], "motion", 2)
    clean = pd.read_csv(tmp_path / "out" / "m1" / "none" / "level0" / "per_case.csv")
    assert clean["seed"][0] == case_seed(42, CASES[0], "none", 0)


def test_gate_signals_computed_when_configured(world: Path, tmp_path: Path) -> None:
    """A tiny SegQC and a hand-made fit.json produce non-NaN predicted_dice and conformal_band."""
    from neurovision.models.qc import build_segqc

    qc_yaml = {
        "name": "segqc",
        "in_channels": 3,
        "widths": [8, 16],
        "num_groups": 4,
        "dropout": 0.0,
    }
    (tmp_path / "segqc.yaml").write_text(yaml.safe_dump(qc_yaml), encoding="utf-8")
    qc_cfg = OmegaConf.create({"model": qc_yaml})
    qc_model = build_segqc(qc_cfg)
    save_checkpoint(
        tmp_path / "qc",
        qc_model,
        torch.optim.Adam(qc_model.parameters()),
        epoch=0,
        global_step=0,
        best_metric=0.5,
        best_metric_name="x",
        best_metric_mode="max",
        cfg=qc_cfg,
        is_best=True,
    )
    fit = {f"{r}__alpha_0.1": {"threshold": 0.3, "feasible": True} for r in ("WT", "TC")}
    write_json(fit, tmp_path / "fit.json")
    models = copy.deepcopy(_make_cfg(world).analysis.severity_ladder.models)
    models = OmegaConf.to_container(models)
    models["m1"]["conformal_fit"] = str(tmp_path / "fit.json")
    _run(
        world,
        tmp_path,
        kinds={},
        models=models,
        qc={
            "enabled": True,
            "checkpoint": str(tmp_path / "qc" / "best.pt"),
            "model_config": str(tmp_path / "segqc.yaml"),
        },
    )
    df = pd.read_csv(tmp_path / "out" / "m1" / "none" / "level0" / "per_case.csv")
    for col in ("predicted_dice_WT", "predicted_dice_TC", "conformal_band_WT", "conformal_band_TC"):
        assert df[col].notna().all(), col
    assert df["predicted_dice_WT"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Resume, determinism
# ---------------------------------------------------------------------------


def test_done_shards_are_skipped_on_rerun(world: Path, tmp_path: Path) -> None:
    first = _run(world, tmp_path, kinds={"rician": [0.1]})
    assert len(first["completed"]) == 2 and not first["skipped"]
    csv = tmp_path / "out" / "m1" / "rician" / "level1" / "per_case.csv"
    stamp = csv.stat().st_mtime_ns
    second = _run(world, tmp_path, kinds={"rician": [0.1]})
    assert not second["completed"] and len(second["skipped"]) == 2
    assert (
        tmp_path / "out" / "m1" / "rician" / "level1" / "per_case.csv"
    ).stat().st_mtime_ns == stamp


def test_done_shard_for_a_different_case_list_is_refused(world: Path, tmp_path: Path) -> None:
    _run(world, tmp_path, kinds={})
    with pytest.raises(RuntimeError, match="different case list"):
        _run(world, tmp_path, kinds={}, case_limit=3)


def test_resume_discards_partial_debris(world: Path, tmp_path: Path) -> None:
    """A killed run leaves `<shard>.partial`; the next run must rebuild it, not trust it."""
    debris = tmp_path / "out" / "m1" / "rician" / "level1.partial"
    debris.mkdir(parents=True)
    (debris / "per_case.csv").write_text("garbage", encoding="utf-8")
    _run(world, tmp_path, kinds={"rician": [0.1]})
    assert not debris.exists()
    assert (tmp_path / "out" / "m1" / "rician" / "level1" / "DONE").is_file()


def test_two_runs_are_identical_except_timing(world: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        cfg = _make_cfg(world, out_dir=str(out), kinds={"motion": [2], "bias": [0.3]})
        ladder.run_severity_ladder(cfg, world)
    for rel in ("none/level0", "motion/level1", "bias/level1"):
        da = pd.read_csv(a / "m1" / rel / "per_case.csv").drop(columns="seconds")
        db = pd.read_csv(b / "m1" / rel / "per_case.csv").drop(columns="seconds")
        pd.testing.assert_frame_equal(da, db)
        pd.testing.assert_frame_equal(
            pd.read_csv(a / "m1" / rel / "ood_features.csv"),
            pd.read_csv(b / "m1" / rel / "ood_features.csv"),
        )
        za, zb = np.load(a / "m1" / rel / "curves.npz"), np.load(b / "m1" / rel / "curves.npz")
        for key in za.files:
            np.testing.assert_array_equal(za[key], zb[key])


def test_corruption_changes_the_numbers(world: Path, tmp_path: Path) -> None:
    """Guards against a driver that accepts a kind but feeds the clean image."""
    _run(world, tmp_path, kinds={"rician": [0.3]})
    clean = pd.read_csv(tmp_path / "out" / "m1" / "none" / "level0" / "per_case.csv")
    noisy = pd.read_csv(tmp_path / "out" / "m1" / "rician" / "level1" / "per_case.csv")
    assert not np.allclose(clean["ent_mean_fg_WT"], noisy["ent_mean_fg_WT"], equal_nan=True)
    f_clean = pd.read_csv(tmp_path / "out" / "m1" / "none" / "level0" / "ood_features.csv")
    f_noisy = pd.read_csv(tmp_path / "out" / "m1" / "rician" / "level1" / "ood_features.csv")
    assert not np.allclose(f_clean.drop(columns="case_id"), f_noisy.drop(columns="case_id"))


# ---------------------------------------------------------------------------
# Bridge, probe, budget
# ---------------------------------------------------------------------------


def test_bridge_fail_stops_the_model_before_any_corrupted_shard(
    world: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad = pd.read_csv(world / "ref.csv")
    bad["dice_WT"] = (bad["dice_WT"] + 0.05).clip(upper=1.0)
    bad["dice_ET"] = (bad["dice_ET"] - 0.05).clip(lower=0.0)
    bad_path = tmp_path / "bad_ref.csv"
    bad.to_csv(bad_path, index=False)
    models = OmegaConf.to_container(_make_cfg(world).analysis.severity_ladder.models)
    models["m1"]["reference_metrics"] = str(bad_path)
    with caplog.at_level(logging.INFO):
        summary = _run(world, tmp_path, models=models, kinds={"rician": [0.1]})
    assert summary["bridge"] == {"m1": False}
    assert summary["stopped_models"] == ["m1"]
    assert "NVX_BRIDGE: FAIL" in caplog.text
    assert (tmp_path / "out" / "m1" / "none" / "level0" / "DONE").is_file()
    assert not (tmp_path / "out" / "m1" / "rician").exists()


def test_bridge_pass_is_logged(
    world: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        _run(world, tmp_path, kinds={})
    assert "NVX_BRIDGE: PASS" in caplog.text
    assert "NVX_SHARD_DONE m1 none 0 2 " in caplog.text


def test_bridge_check_fails_on_missing_reference_case(tmp_path: Path) -> None:
    pd.DataFrame({"case_id": ["a", "b"], "dice_ET": 1.0, "dice_TC": 1.0, "dice_WT": 1.0}).to_csv(
        tmp_path / "ours.csv", index=False
    )
    pd.DataFrame({"case_id": ["a"], "dice_ET": 1.0, "dice_TC": 1.0, "dice_WT": 1.0}).to_csv(
        tmp_path / "ref.csv", index=False
    )
    passed, _ = ladder.bridge_check(tmp_path / "ours.csv", tmp_path / "ref.csv", 0.003)
    assert not passed


def test_probe_uses_first_five_cases_all_conditions_in_its_own_dir(
    world: Path, tmp_path: Path
) -> None:
    summary = _run(world, tmp_path, probe=True, case_limit=None, kinds={"rician": [0.1, 0.2]})
    probe_root = tmp_path / "out" / "probe"
    assert len(summary["completed"]) == 3  # level 0 + two rician levels
    df = pd.read_csv(probe_root / "m1" / "rician" / "level2" / "per_case.csv")
    assert list(df["case_id"]) == CASES[:5]
    assert not (tmp_path / "out" / "m1").exists()  # nothing a main run could mistake for DONE


def test_budget_exhausted_mid_shard_leaves_nothing_partial(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"n": 0}
    real = ladder._check_deadline

    def flaky(deadline: float | None) -> None:
        calls["n"] += 1
        if calls["n"] >= 7:  # let level 0 finish, then die inside the next shard
            raise ladder.BudgetExhausted()
        real(deadline)

    monkeypatch.setattr(ladder, "_check_deadline", flaky)
    with caplog.at_level(logging.INFO):
        summary = _run(world, tmp_path, kinds={"rician": [0.1]})
    assert summary["budget_exhausted"] is True
    assert calls["n"] == 7  # died on the first chunk of shard 2, i.e. mid-shard
    assert "NVX_BUDGET_EXHAUSTED" in caplog.text
    out = tmp_path / "out" / "m1"
    assert (out / "none" / "level0" / "DONE").is_file()
    assert not list((tmp_path / "out").rglob("*.partial"))
    assert not (out / "rician" / "level1").exists()


def test_real_deadline_in_the_past_runs_nothing(world: Path, tmp_path: Path) -> None:
    summary = _run(world, tmp_path, max_hours=0.0, kinds={"rician": [0.1]})
    assert summary["budget_exhausted"] is True and not summary["completed"]
    assert not list((tmp_path / "out").rglob("DONE"))


def test_missing_reference_fails_before_any_inference(world: Path, tmp_path: Path) -> None:
    models = OmegaConf.to_container(_make_cfg(world).analysis.severity_ladder.models)
    models["m1"]["reference_metrics"] = str(tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="reference_metrics"):
        _run(world, tmp_path, models=models)


# ---------------------------------------------------------------------------
# nnU-Net glue
# ---------------------------------------------------------------------------


def _fake_nnunet(state: dict[str, Any], file_ending: str = ".nii"):
    """A stand-in for `run_prediction` that writes what nnU-Net writes for each input case."""

    def fake(cmd: list[str]) -> float:
        in_dir, out_dir = Path(cmd[cmd.index("-i") + 1]), Path(cmd[cmd.index("-o") + 1])
        state["cmd"] = list(cmd)
        for first in sorted(in_dir.glob(f"*_0000{file_ending}")):
            case = first.name[: -len(f"_0000{file_ending}")]
            state.setdefault("inputs", {})[case] = np.asarray(
                nib.load(str(in_dir / f"{case}_0001{file_ending}")).dataobj
            )
            wt = np.zeros(SHAPE, bool)
            wt[3:13, 3:13, 3:13] = True
            tc = np.zeros(SHAPE, bool)
            tc[5:11, 5:11, 5:11] = True
            et = np.zeros(SHAPE, bool)
            et[7:9, 7:9, 7:9] = True
            # nnU-Net channel order (WT, TC, ET); axes (W, H, D) = reverse of ours.
            probs = np.stack([np.where(m, 0.9, 0.1) for m in (wt, tc, et)]).astype(np.float32)
            np.savez_compressed(out_dir / f"{case}.npz", probabilities=probs.transpose(0, 3, 2, 1))
            labels = np.zeros(SHAPE, np.uint8)
            labels[wt], labels[tc], labels[et] = 1, 2, 3
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(out_dir / f"{case}{file_ending}"))
        return 0.0

    return fake


def test_nnunet_backend_writes_converts_and_cleans_up(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, Any] = {}
    monkeypatch.setattr(ladder._NNUNET, "run_prediction", _fake_nnunet(state))
    work = tmp_path / "work"
    backend = ladder.NnunetBackend(
        tmp_path / "model_folder", ".nii", world / "prep", work, "cpu", 25, 1e-5
    )
    corrupted = {
        c: np.random.default_rng(i).standard_normal((4, *SHAPE)).astype(np.float32)
        for i, c in enumerate(CASES[:2])
    }
    out = backend.predict(corrupted)

    assert "--disable_tta" in state["cmd"] and "--save_probabilities" in state["cmd"]
    for case, image in corrupted.items():
        np.testing.assert_array_equal(state["inputs"][case], image[1])  # the CORRUPTED channel
        assert out[case].shape == (3, *SHAPE) and out[case].dtype == np.float32
        expected = np.log(0.9 / 0.1)
        assert out[case][2][8, 8, 8] == pytest.approx(expected, abs=0.01)  # WT inside
        assert out[case][2][0, 0, 0] == pytest.approx(-expected, abs=0.01)  # WT outside
    # Corrupted volumes were only transient.
    assert not list(work.rglob("*.npy")) and not list(work.rglob("*.nii"))
    assert not (work / "tmp_prep").exists()


def test_run_shard_with_nnunet_backend_chunks(
    world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state: dict[str, Any] = {}
    monkeypatch.setattr(ladder._NNUNET, "run_prediction", _fake_nnunet(state))
    cfg = _make_cfg(world)
    backend = ladder.NnunetBackend(
        tmp_path / "mf", ".nii", world / "prep", tmp_path / "work", "cpu", 2, 1e-5
    )
    gate = ladder.build_gate_signals(
        cfg.analysis.severity_ladder, cfg, {"conformal_fit": None}, world, torch.device("cpu")
    )
    sec = ladder.run_shard(
        ladder.Shard("nn", "none", 0, 0.0),
        out_dir=tmp_path / "out",
        case_ids=CASES[:3],  # chunks of 2 -> two nnU-Net calls
        prep_dir=world / "prep",
        cfg=cfg,
        backend=backend,
        gate=gate,
        base_seed=42,
        deadline=None,
    )
    assert sec is not None
    df = pd.read_csv(tmp_path / "out" / "nn" / "none" / "level0" / "per_case.csv")
    assert list(df["case_id"]) == CASES[:3]
    # The fake predicts the true regions exactly, so Dice is 1.
    assert df["dice_WT"].round(6).eq(1.0).all()


def test_resolve_nnunet_model_requires_search_root() -> None:
    with pytest.raises(ValueError, match="model_search_root"):
        ladder.resolve_nnunet_model({"model_search_root": None}, "cpu")


def test_build_inhouse_cfg_swaps_model_resolves_and_merges_overrides(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.model = {"name": "other"}
    out = ladder.build_inhouse_cfg(
        cfg, tmp_path / "model.yaml", {"training": {"loss": {"name": "multitask"}}}
    )
    assert out.model.name == "unet3d" and out.model.in_channels == 4  # interpolation resolved
    assert out.training.loss.name == "multitask"
    assert out.inference.sliding_window.roi_size == [16, 16, 16]  # the rest is untouched
