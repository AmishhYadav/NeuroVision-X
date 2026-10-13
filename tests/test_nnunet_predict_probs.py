"""Tests for scripts/nnunet_predict_probs.py.

No real nnU-Net, no GPU, no real BraTS. `subprocess.run` is monkeypatched with a
fake that writes what nnU-Net v2.8.1 writes with `--save_probabilities` (an
`.npz` with the key `probabilities`, shape `(3, W, H, D)` in channel order
WT, TC, ET, a `.pkl`, and a label map), built from a known synthetic label
volume. The synthetic volumes are deliberately asymmetric and the crop box
non-trivial, so a flip, a transpose or a channel swap cannot cancel out.
"""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from tests.script_loader import load_script

probs_script = load_script("nnunet_predict_probs")

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_TRAINER_DIR = "nnUNetTrainer__nnUNetPlans__3d_fullres"

# Cropped grid (D, H, W) = (16, 14, 12) inside a (20, 22, 18) full grid.
_FULL_SHAPE = (20, 22, 18)
_CROPPED = (16, 14, 12)
_BBOX = {
    "BraTS2021_00001": [[2, 18], [3, 17], [1, 13]],
    "BraTS2021_00002": [[0, 16], [4, 18], [2, 14]],
}
_CASES = list(_BBOX)
# BraTS 2021 raw affine (LPS) plus an offset, so the crop shift is visible.
_AFFINE = np.array([[-1.0, 0, 0, 5.0], [0, -1.0, 0, 30.0], [0, 0, 1.0, -4.0], [0, 0, 0, 1.0]])


def _label_a() -> np.ndarray:
    """Asymmetric nested label in OUR convention (1 NCR, 2 ED, 3 ET); ET has >= 50 voxels."""
    label = np.zeros(_CROPPED, dtype=np.uint8)
    label[1:14, 2:12, 1:10] = 2
    label[3:12, 4:10, 3:8] = 1
    label[4:11, 5:9, 3:8] = 3
    return label


def _labels() -> dict[str, np.ndarray]:
    return {_CASES[0]: _label_a(), _CASES[1]: np.flip(_label_a(), axis=1).copy()}


def _write_cohort(prep_dir: Path, cases: list[str] = _CASES) -> dict[str, np.ndarray]:
    """Writes image.npy (fp16, distinct per channel), label.npy, meta.json per case."""
    rng = np.random.default_rng(0)
    for cid in cases:
        case_dir = prep_dir / cid
        case_dir.mkdir(parents=True)
        image = (rng.normal(size=(4, *_CROPPED)) + np.arange(4).reshape(4, 1, 1, 1) * 10).astype(
            np.float16
        )
        np.save(case_dir / "image.npy", image)
        np.save(case_dir / "label.npy", _labels()[cid])
        meta = {
            "case_id": cid,
            "original_shape": list(_FULL_SHAPE),
            "cropped_shape": list(_CROPPED),
            "bbox": _BBOX[cid],
            "affine": _AFFINE.tolist(),
            "spacing": [1.0, 1.0, 1.0],
            "has_label": True,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta))
    return _labels()


def _write_model_folder(
    root: Path,
    labels: dict | None = None,
    class_order: list[int] | None = None,
    checkpoint: str = "checkpoint_final.pth",
) -> Path:
    model_folder = root / "Dataset901_X" / _TRAINER_DIR
    fold_dir = model_folder / "fold_all"
    fold_dir.mkdir(parents=True)
    torch.save(
        {
            "current_epoch": 1000,
            "trainer_name": "nnUNetTrainer",
            "init_args": {"plans": {}, "configuration": "3d_fullres", "fold": "all"},
            "logging": {"train_losses": [0.5, 0.4]},
        },
        fold_dir / checkpoint,
    )
    dataset_json = {
        "file_ending": ".nii",
        "labels": labels
        or {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3],
            "enhancing tumor": [3],
        },
        "regions_class_order": class_order or [1, 2, 3],
    }
    (model_folder / "dataset.json").write_text(json.dumps(dataset_json))
    (model_folder / "plans.json").write_text(json.dumps({"image_reader_writer": "SimpleITKIO"}))
    return model_folder


def _write_splits(path: Path, test: list[str]) -> None:
    path.write_text(yaml.safe_dump({"train": [], "val": [], "test": test}))


def _nnunet_outputs(
    label_our: np.ndarray, tweak: Callable[[np.ndarray], np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """What nnU-Net would write for a prediction of `label_our`.

    Args:
        label_our: The label the fake model "predicts", our convention.
        tweak: Optional edit of the `(3, D, H, W)` probabilities (channels WT, TC, ET)
            BEFORE nnU-Net's label map is built from them, i.e. legitimate model
            output such as a non-nested voxel.

    Returns:
        `(probabilities, label_map)`: probabilities `(3, W, H, D)` float32, channels
        (WT, TC, ET), the SimpleITK axis order; label map `(D, H, W)` in nnU-Net's
        convention (1 ED, 2 NCR, 3 ET), built from the probabilities with nnU-Net's
        overwrite rule (`p > 0.5`, later channels win).
    """
    wt = label_our > 0
    tc = (label_our == 1) | (label_our == 3)
    et = label_our == 3
    probs_dhw = np.where(np.stack([wt, tc, et]), 0.95, 0.02).astype(np.float32)
    if tweak is not None:
        probs_dhw = tweak(probs_dhw)
    label_map = np.zeros(label_our.shape, dtype=np.uint8)
    for channel, value in enumerate((1, 2, 3)):
        label_map[probs_dhw[channel] > 0.5] = value
    return probs_dhw.transpose(0, 3, 2, 1), label_map


def _make_fake_run(
    labels: dict[str, np.ndarray],
    corrupt: Callable[[np.ndarray], np.ndarray] | None = None,
    tweak: Callable[[np.ndarray], np.ndarray] | None = None,
    predict_from: dict[str, np.ndarray] | None = None,
):
    """A subprocess.run stand-in. Records the command and what it found in `-i`.

    Args:
        labels: Ground truth per case (not used to predict if `predict_from` is given).
        corrupt: Edit of the written `(3, W, H, D)` probabilities AFTER the label map is
            built, i.e. a bug in channel order or axes.
        tweak: See `_nnunet_outputs`.
        predict_from: Labels the fake model predicts instead of the ground truth.
    """
    state: dict = {"calls": [], "inputs": {}, "case_lists": []}
    source = predict_from or labels

    def fake_run(cmd: list[str], check: bool = True, **kwargs: object):
        state["calls"].append(list(cmd))
        in_dir, out_dir = Path(cmd[cmd.index("-i") + 1]), Path(cmd[cmd.index("-o") + 1])
        state["case_lists"].append([])
        for first in sorted(in_dir.glob("*_0000.nii")):
            cid = first.name[: -len("_0000.nii")]
            state["case_lists"][-1].append(cid)
            loaded = [nib.load(str(in_dir / f"{cid}_000{c}.nii")) for c in range(4)]
            state["inputs"][cid] = [(np.asarray(i.dataobj), i.affine) for i in loaded]
            probs, label_map = _nnunet_outputs(source[cid], tweak)
            if corrupt is not None:
                probs = corrupt(probs)
            np.savez_compressed(out_dir / f"{cid}.npz", probabilities=probs)
            (out_dir / f"{cid}.pkl").write_bytes(b"x")
            nib.save(nib.Nifti1Image(label_map, np.eye(4)), str(out_dir / f"{cid}.nii"))
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run, state


@pytest.fixture(scope="module")
def base_cfg() -> DictConfig:
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=["data.root_dir=/unused/for/this/test"])
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


@pytest.fixture()
def setup(tmp_path: Path, base_cfg: DictConfig):
    """A model folder, two cases, and a config pointing at them."""
    labels = _write_cohort(tmp_path / "prep")
    _write_model_folder(tmp_path / "models")
    _write_splits(tmp_path / "splits.yaml", _CASES)
    cfg = copy.deepcopy(base_cfg)
    pc = cfg.analysis.nnunet_probs
    pc.model_search_root = str(tmp_path / "models")
    pc.prep_dir = str(tmp_path / "prep")
    pc.splits_path = str(tmp_path / "splits.yaml")
    pc.out_eval_dir = str(tmp_path / "eval")
    pc.work_dir = str(tmp_path / "work")
    pc.device = "cpu"
    return cfg, labels, tmp_path


def _use_fake(monkeypatch, fake_run) -> None:
    monkeypatch.setattr(probs_script.subprocess, "run", fake_run)


# ---------------------------------------------------------------------------
# (a) End to end
# ---------------------------------------------------------------------------


def test_end_to_end_two_cases(setup, monkeypatch) -> None:
    cfg, labels, tmp_path = setup
    fake_run, state = _make_fake_run(labels)
    _use_fake(monkeypatch, fake_run)

    summary = probs_script.run_nnunet_probs(cfg)

    cmd = state["calls"][0]
    assert "--save_probabilities" in cmd and "--disable_tta" not in cmd
    assert cmd[cmd.index("-chk") + 1] == "checkpoint_final.pth"
    assert cmd[cmd.index("-f") + 1] == "all"

    out = tmp_path / "eval"
    for cid in _CASES:
        logits = np.load(out / "logits" / f"{cid}.npy")
        assert logits.dtype == np.float16 and logits.shape == (3, *_CROPPED)
        assert np.isfinite(logits).all()
        # Channel order (ET, TC, WT) and orientation: the sign matches the label exactly.
        label = labels[cid]
        assert np.array_equal(logits[0] > 0, label == 3)
        assert np.array_equal(logits[1] > 0, (label == 1) | (label == 3))
        assert np.array_equal(logits[2] > 0, label > 0)

        # Inputs: channel c is image.npy[c] unchanged, affine shifted by the crop start.
        image = np.load(tmp_path / "prep" / cid / "image.npy")
        start = [b[0] for b in _BBOX[cid]]
        for c, (array, affine) in enumerate(state["inputs"][cid]):
            assert array.dtype == np.float32 and np.array_equal(array, image[c].astype(np.float32))
            assert np.allclose(affine @ [0, 0, 0, 1], _AFFINE @ [*start, 1])
            assert np.allclose(affine[:3, :3], _AFFINE[:3, :3])

    per_case = pd.read_csv(out / "per_case_metrics.csv", index_col="case_id")
    for region in ("ET", "TC", "WT"):
        assert np.allclose(per_case[f"dice_{region}"], 1.0)
    assert (out / "summary.csv").is_file() and (out / "eval_config.yaml").is_file()
    assert json.loads((out / "nnunet_probs_summary.json").read_text())["n_cases"] == 2
    assert summary["checkpoint_epoch"] == 1000
    assert max(summary["label_check_max_mismatch"].values()) == 0.0
    # Scratch is cleaned up case by case.
    assert not list((tmp_path / "work").rglob("*.npz"))
    assert not list((tmp_path / "work" / "input").glob("*.nii"))
    assert (out / "COMPLETE").is_file()
    saved_cfg = yaml.safe_load((out / "eval_config.yaml").read_text())
    assert saved_cfg["inference"]["postprocess"]["min_component_size"] == 50
    prov = saved_cfg["nnunet_probs_provenance"]
    assert prov["model"] == "nnunet_v2_3dfullres_gatea" and prov["checkpoint_epoch"] == 1000
    assert prov["disable_tta"] is False and "do NOT describe" in prov["note"]


# ---------------------------------------------------------------------------
# (b), (c) The label check catches a wrong channel order and a wrong axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt",
    [lambda p: p[[1, 0, 2]], lambda p: p[[0, 2, 1]], lambda p: np.flip(p, axis=1).copy()],
    ids=["swapped_wt_tc_channels", "swapped_tc_et_channels", "flipped_axis"],
)
def test_label_check_raises(setup, monkeypatch, corrupt) -> None:
    cfg, labels, _ = setup
    fake_run, _ = _make_fake_run(labels, corrupt=corrupt)
    _use_fake(monkeypatch, fake_run)
    with pytest.raises(ValueError, match="BraTS2021_00001.*label map"):
        probs_script.run_nnunet_probs(cfg)


# ---------------------------------------------------------------------------
# (d), (e) Refusals before any work
# ---------------------------------------------------------------------------


def _forbid_run(monkeypatch) -> None:
    def boom(*args: object, **kwargs: object):
        raise AssertionError("nnU-Net must not be launched")

    monkeypatch.setattr(probs_script.subprocess, "run", boom)


def test_refuses_non_region_dataset(setup, monkeypatch, tmp_path: Path) -> None:
    cfg, _, root = setup
    (next((root / "models").rglob("dataset.json"))).write_text(
        json.dumps(
            {
                "file_ending": ".nii",
                "labels": {"background": 0, "edema": 1, "necrotic": 2, "enhancing": 3},
            }
        )
    )
    _forbid_run(monkeypatch)
    with pytest.raises(ValueError, match="region-mode"):
        probs_script.run_nnunet_probs(cfg)


@pytest.mark.parametrize("reader", ["SimpleITKIO", "NibabelIO"])
def test_accepts_non_reorienting_readers(tmp_path: Path, reader: str) -> None:
    (tmp_path / "plans.json").write_text(json.dumps({"image_reader_writer": reader}))
    probs_script.check_reader_writer(tmp_path)


@pytest.mark.parametrize("reader", ["NibabelIOWithReorient", "SimpleITKIOWithReorient"])
def test_refuses_reorienting_readers(tmp_path: Path, reader: str) -> None:
    (tmp_path / "plans.json").write_text(json.dumps({"image_reader_writer": reader}))
    with pytest.raises(ValueError, match=reader):
        probs_script.check_reader_writer(tmp_path)


def test_refuses_missing_case_before_predicting(setup, monkeypatch) -> None:
    cfg, _, root = setup
    (root / "prep" / "BraTS2021_00002" / "image.npy").unlink()
    _forbid_run(monkeypatch)
    with pytest.raises(FileNotFoundError, match="BraTS2021_00002"):
        probs_script.run_nnunet_probs(cfg)


def test_refuses_checkpoint_best(setup, monkeypatch, tmp_path: Path) -> None:
    cfg, _, _ = setup
    _forbid_run(monkeypatch)
    cfg.analysis.nnunet_probs.checkpoint_name = "checkpoint_best.pth"
    with pytest.raises(ValueError, match="checkpoint_final"):
        probs_script.run_nnunet_probs(cfg)

    # And a model folder that only has checkpoint_best is not found.
    only_best = tmp_path / "only_best"
    only_best.mkdir()
    _write_model_folder(only_best, checkpoint="checkpoint_best.pth")
    with pytest.raises(FileNotFoundError, match="checkpoint_final"):
        probs_script.find_model_folder(only_best)


# ---------------------------------------------------------------------------
# (f) fp16 trap
# ---------------------------------------------------------------------------


def test_logit_of_zero_and_one_is_finite_in_fp16() -> None:
    logits = probs_script.probabilities_to_logits(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert logits.dtype == np.float16 and np.isfinite(logits).all()
    assert logits[0] < -13 and logits[2] > 13 and logits[1] == 0
    with pytest.raises(ValueError, match="non-finite"):
        probs_script.probabilities_to_logits(np.array([np.nan], dtype=np.float32))


# ---------------------------------------------------------------------------
# (g) case_limit
# ---------------------------------------------------------------------------


def test_case_limit_predicts_only_the_first_cases(setup, monkeypatch) -> None:
    cfg, labels, root = setup
    cfg.analysis.nnunet_probs.case_limit = 1
    fake_run, state = _make_fake_run(labels)
    _use_fake(monkeypatch, fake_run)
    probs_script.run_nnunet_probs(cfg)
    assert list(state["inputs"]) == [_CASES[0]]
    assert [p.stem for p in (root / "eval" / "logits").glob("*.npy")] == [_CASES[0]]


def test_case_limit_still_checks_every_case(setup, monkeypatch) -> None:
    cfg, _, root = setup
    cfg.analysis.nnunet_probs.case_limit = 1
    (root / "prep" / "BraTS2021_00002" / "meta.json").unlink()
    _forbid_run(monkeypatch)
    with pytest.raises(FileNotFoundError, match="BraTS2021_00002"):
        probs_script.run_nnunet_probs(cfg)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_config_block_is_reachable_and_null_paths_raise(base_cfg: DictConfig) -> None:
    assert "nnunet_probs" in base_cfg.analysis and "nnunet_probs" not in base_cfg
    pc = base_cfg.analysis.nnunet_probs
    assert pc.checkpoint_name == "checkpoint_final.pth" and pc.expect_epoch == 1000
    assert pc.model_search_root is None and pc.out_eval_dir is None and pc.work_dir is None
    with pytest.raises(ValueError, match="model_search_root"):
        probs_script.run_nnunet_probs(base_cfg)


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_simpleitk_axis_premise(tmp_path: Path) -> None:
    """The axis reasoning the whole script rests on, on the real library."""
    import SimpleITK as sitk

    a = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)  # non-cubic (D, H, W)
    path = tmp_path / "x.nii"
    nib.save(nib.Nifti1Image(a, _AFFINE), str(path))
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(path))), a.transpose(2, 1, 0))


def test_non_nested_voxel_passes_the_label_check(setup, monkeypatch) -> None:
    """ET > 0.5 with TC <= 0.5 is legitimate nnU-Net output (label 3 -> TC true)."""
    cfg, labels, _ = setup

    def tweak(p: np.ndarray) -> np.ndarray:  # (3, D, H, W), channels WT, TC, ET
        p[2, 0, 0, 0] = 0.95  # a background voxel: ET on, TC and WT off
        return p

    fake_run, _ = _make_fake_run(labels, tweak=tweak)
    _use_fake(monkeypatch, fake_run)
    summary = probs_script.run_nnunet_probs(cfg)
    assert max(summary["label_check_max_mismatch"].values()) == 0.0


@pytest.mark.parametrize("k, passes", [(2, True), (3, False)])
def test_mismatch_tolerance_boundary(setup, monkeypatch, k: int, passes: bool) -> None:
    cfg, labels, _ = setup
    n_voxels = int(np.prod(_CROPPED))
    cfg.analysis.nnunet_probs.label_check_tol = 2.5 / n_voxels

    def corrupt(p: np.ndarray) -> np.ndarray:  # (3, W, H, D): k background voxels get ET on
        p[2, 0, 0, :k] = 0.95
        return p

    fake_run, _ = _make_fake_run(labels, corrupt=corrupt)
    _use_fake(monkeypatch, fake_run)
    if passes:
        probs_script.run_nnunet_probs(cfg)
    else:
        with pytest.raises(ValueError, match="label map"):
            probs_script.run_nnunet_probs(cfg)


def test_chunking_runs_nnunet_once_per_chunk(setup, monkeypatch) -> None:
    cfg, labels, root = setup
    cfg.analysis.nnunet_probs.chunk_size = 1
    fake_run, state = _make_fake_run(labels)
    _use_fake(monkeypatch, fake_run)
    summary = probs_script.run_nnunet_probs(cfg)
    assert state["case_lists"] == [[_CASES[0]], [_CASES[1]]]
    assert summary["n_cases"] == 2 and len(summary["chunk_wall_s"]) == 2
    assert len(list((root / "eval" / "logits").glob("*.npy"))) == 2


def test_stale_output_dir_is_removed(setup, monkeypatch) -> None:
    cfg, labels, root = setup
    stale = root / "eval" / "logits"
    stale.mkdir(parents=True)
    np.save(stale / "BraTS2021_99999.npy", np.zeros(1))
    (root / "eval" / "COMPLETE").write_text("old")
    fake_run, _ = _make_fake_run(labels)
    _use_fake(monkeypatch, fake_run)
    probs_script.run_nnunet_probs(cfg)
    assert sorted(p.stem for p in stale.glob("*.npy")) == sorted(_CASES)


def test_refuses_to_wipe_an_input_directory(setup, monkeypatch) -> None:
    cfg, _, root = setup
    cfg.analysis.nnunet_probs.out_eval_dir = str(root)  # contains prep, work, models
    _forbid_run(monkeypatch)
    with pytest.raises(ValueError, match="refusing to wipe"):
        probs_script.run_nnunet_probs(cfg)
    assert (root / "prep").is_dir()


def test_incomplete_logits_leave_no_summary_files(setup, monkeypatch) -> None:
    cfg, labels, root = setup
    fake_run, _ = _make_fake_run(labels)
    _use_fake(monkeypatch, fake_run)
    original = probs_script.replay_case

    def sabotage(*args: object, **kwargs: object):
        (root / "eval" / "logits" / f"{_CASES[0]}.npy").unlink(missing_ok=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(probs_script, "replay_case", sabotage)
    with pytest.raises(RuntimeError, match=_CASES[0]):
        probs_script.run_nnunet_probs(cfg)
    leftovers = {p.name for p in (root / "eval").iterdir()}
    assert leftovers == {"logits"}  # no csv, no yaml, no json, no COMPLETE marker


def test_dice_sanity_catches_misaligned_predictions(setup, monkeypatch) -> None:
    """Self-consistent with its own label map, but nowhere near label.npy."""
    cfg, labels, root = setup
    wrong = np.zeros(_CROPPED, dtype=np.uint8)
    wrong[12:16, 0:5, 0:5] = 3  # 100 voxels, off the tumour
    fake_run, _ = _make_fake_run(labels, predict_from={c: wrong for c in _CASES})
    _use_fake(monkeypatch, fake_run)
    with pytest.raises(ValueError, match="dice_WT"):
        probs_script.run_nnunet_probs(cfg)
    assert not (root / "eval" / "COMPLETE").exists()


def test_dice_sanity_fires_early_not_at_the_end(setup, monkeypatch) -> None:
    cfg, labels, _ = setup
    cfg.analysis.nnunet_probs.sanity_after_n = 1
    cfg.analysis.nnunet_probs.chunk_size = 1
    wrong = np.zeros(_CROPPED, dtype=np.uint8)
    wrong[12:16, 0:5, 0:5] = 3
    fake_run, state = _make_fake_run(labels, predict_from={c: wrong for c in _CASES})
    _use_fake(monkeypatch, fake_run)
    with pytest.raises(ValueError, match="dice_WT"):
        probs_script.run_nnunet_probs(cfg)
    assert len(state["case_lists"]) == 1  # the second chunk was never predicted


def test_dice_sanity_can_be_disabled(setup, monkeypatch) -> None:
    cfg, labels, _ = setup
    cfg.analysis.nnunet_probs.min_dice_wt_sanity = None
    wrong = np.zeros(_CROPPED, dtype=np.uint8)
    wrong[12:16, 0:5, 0:5] = 3
    fake_run, _ = _make_fake_run(labels, predict_from={c: wrong for c in _CASES})
    _use_fake(monkeypatch, fake_run)
    probs_script.run_nnunet_probs(cfg)


def test_refuses_non_default_postprocess_chain(setup, monkeypatch) -> None:
    cfg, _, _ = setup
    cfg.inference.postprocess.min_component_size = 0
    _forbid_run(monkeypatch)
    with pytest.raises(ValueError, match="min_component_size"):
        probs_script.run_nnunet_probs(cfg)


def test_cuda_preflight_runs_before_anything_is_written(setup, monkeypatch) -> None:
    cfg, _, root = setup
    cfg.analysis.nnunet_probs.device = "cuda"

    def no_cuda() -> None:
        raise RuntimeError("no usable CUDA")

    monkeypatch.setattr(probs_script, "_require_cuda_or_raise", no_cuda)
    _forbid_run(monkeypatch)
    with pytest.raises(RuntimeError, match="no usable CUDA"):
        probs_script.run_nnunet_probs(cfg)
    assert not (root / "work").exists() and not (root / "eval").exists()


def test_use_mask_for_norm_false_warns(tmp_path: Path, caplog) -> None:
    plans = {"configurations": {"3d_fullres": {"use_mask_for_norm": [True, False, True, True]}}}
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    with caplog.at_level("WARNING"):
        flags = probs_script.log_use_mask_for_norm(tmp_path)
    assert flags == [True, False, True, True] and "use_mask_for_norm" in caplog.text
    caplog.clear()
    plans["configurations"]["3d_fullres"]["use_mask_for_norm"] = [True] * 4
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    with caplog.at_level("WARNING"):
        probs_script.log_use_mask_for_norm(tmp_path)
    assert "use_mask_for_norm" not in caplog.text
