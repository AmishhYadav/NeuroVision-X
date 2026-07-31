"""Tests for scripts/train.py's pure-ish, testable helper functions.

`main()` is a Hydra entry point and is awkward (and unnecessary) to unit test
directly, so these tests import the plain helpers underneath it --
`build_dataloaders`, `init_wandb`, `select_resume_checkpoint`, `run_training`
-- and exercise them against tiny synthetic `.npy` trees under `tmp_path`,
never real BraTS data and never a real W&B call.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import,
following the exact same pattern as tests/test_preprocess_script.py. The path
is built relative to this test file, never hardcoded, so this works
regardless of where the repo is checked out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from omegaconf import OmegaConf

from neurovision.training.checkpoint import LAST_CHECKPOINT_NAME, save_checkpoint
from neurovision.utils.io import write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
train_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["train_script"] = train_script
_spec.loader.exec_module(train_script)

build_dataloaders = train_script.build_dataloaders
init_wandb = train_script.init_wandb
select_resume_checkpoint = train_script.select_resume_checkpoint
run_training = train_script.run_training

_IMAGE_SHAPE = (4, 16, 16, 16)
_LABEL_SHAPE = (16, 16, 16)


# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------


def _write_case(prep_dir: Path, case_id: str) -> None:
    """Writes one synthetic preprocessed case: image.npy + label.npy.

    Matches the on-disk layout `scripts/preprocess.py` produces: float16
    image, uint8 label, under `<prep_dir>/<case_id>/`. A small foreground
    blob is included so RandCropByPosNegLabeld's positive sampling has
    something real to find, rather than falling back to all-negative crops.
    """
    case_dir = prep_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    image = rng.random(_IMAGE_SHAPE, dtype=np.float32).astype(np.float16)
    label = np.zeros(_LABEL_SHAPE, dtype=np.uint8)
    label[4:8, 4:8, 4:8] = 1  # necrotic core -> in TC, WT
    label[9:12, 9:12, 9:12] = 3  # enhancing tumor -> in ET, TC, WT
    np.save(case_dir / "image.npy", image)
    np.save(case_dir / "label.npy", label)


def _write_splits(path: Path, train: list[str], val: list[str], test: list[str]) -> None:
    write_yaml({"train": train, "val": val, "test": test}, path)


def _make_cfg(tmp_path: Path, prep_dir: Path, splits_path: Path, **training_overrides) -> object:
    """Builds a full composed config mirroring configs/config.yaml + friends.

    Every value is kept tiny (16^3 volumes, a 2-level U-Net, 0 dataloader
    workers) so a whole run stays well under a second on CPU.
    """
    checkpoint_dir = training_overrides.pop("checkpoint_dir", tmp_path / "checkpoints")
    resume = training_overrides.pop("resume", None)
    epochs = training_overrides.pop("epochs", 1)
    batch_size = training_overrides.pop("batch_size", 2)
    wandb_mode = training_overrides.pop("wandb_mode", "disabled")

    base = {
        "seed": 0,
        "device": "cpu",
        "experiment_name": "test_exp",
        "output_dir": str(tmp_path / "outputs"),
        "wandb": {
            "project": "test-project",
            "entity": None,
            "mode": wandb_mode,
            "tags": [],
            "group": None,
            "run_id": None,
        },
        "data": {
            "in_channels": 4,
            "num_classes": 3,
            "regions": ["ET", "TC", "WT"],
            "patch_size": [16, 16, 16],
            "pos_neg_ratio": [1, 1],
            "samples_per_volume": 1,
            "augment": {
                "flip_prob": 0.0,
                "rot90_prob": 0.0,
                "scale_intensity_factor": 0.1,
                "shift_intensity_offset": 0.1,
                "noise_prob": 0.0,
                "noise_std": 0.01,
            },
            "splits": {"path": str(splits_path)},
            "preprocessing": {"out_dir": str(prep_dir)},
            "dataset_type": "dataset",
            "cache_dir": str(tmp_path / "cache"),
            "cache_rate": 0.0,
            "num_workers": 0,
        },
        "model": {
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
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum_steps": 1,
            "amp": False,
            "optimizer": {
                "name": "adamw",
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"name": "cosine", "warmup_epochs": 1, "min_lr": 1.0e-6},
            "loss": {
                "name": "dice_ce",
                "dice_weight": 1.0,
                "ce_weight": 1.0,
                "sigmoid": True,
                "softmax": False,
                "include_background": True,
                "squared_pred": False,
                "smooth_nr": 1.0e-5,
                "smooth_dr": 1.0e-5,
                "deep_supervision": {"enabled": False, "weights": None},
            },
            "grad_clip_norm": 1.0,
            "val_interval": 1,
            "log_interval": 1,
            "sliding_window": {"roi_size": [16, 16, 16], "sw_batch_size": 1, "overlap": 0.5},
            "checkpoint": {
                "dir": str(checkpoint_dir),
                "save_every_n_epochs": 1,
                "keep_last_n": 1,
                "monitor": "val/dice_mean",
                "mode": "max",
                "resume": str(resume) if resume is not None else None,
            },
            "max_hours": None,
        },
    }
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# 1. init_wandb(mode="disabled") never touches wandb
# ---------------------------------------------------------------------------


def test_init_wandb_disabled_returns_none_and_never_imports_wandb(monkeypatch):
    # sys.modules["wandb"] = None makes any `import wandb` raise ImportError,
    # per Python's import machinery -- so if init_wandb tried to import wandb
    # despite mode="disabled", this test would fail with that ImportError
    # instead of silently passing.
    monkeypatch.setitem(sys.modules, "wandb", None)

    cfg = OmegaConf.create({"wandb": {"mode": "disabled"}})
    result = init_wandb(cfg, resume_state=None)

    assert result is None


# ---------------------------------------------------------------------------
# 2. build_dataloaders: val batch_size == 1, train batch_size == configured
# ---------------------------------------------------------------------------


def test_build_dataloaders_train_and_val_batch_sizes(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    for case_id in ("case_000", "case_001", "case_002"):
        _write_case(prep_dir, case_id)
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, train=["case_000", "case_001"], val=["case_002"], test=["case_002"])

    cfg = _make_cfg(tmp_path, prep_dir, splits_path, batch_size=2)
    device = torch.device("cpu")

    train_loader, val_loader = build_dataloaders(cfg, device)

    assert train_loader.batch_size == cfg.training.batch_size == 2
    assert val_loader.batch_size == 1

    # Sanity: loaders actually produce a properly-collated dict batch (this
    # is what regresses if list_data_collate is ever dropped in favor of the
    # torch default -- RandCropByPosNegLabeld returns a list per case).
    train_batch = next(iter(train_loader))
    assert isinstance(train_batch, dict)
    assert train_batch["image"].shape[1:] == (4, 16, 16, 16)

    val_batch = next(iter(val_loader))
    assert isinstance(val_batch, dict)
    assert val_batch["image"].shape[0] == 1


# ---------------------------------------------------------------------------
# 3. End-to-end smoke test: one epoch, writes last.pt
# ---------------------------------------------------------------------------


def test_run_training_smoke_one_epoch_writes_last_checkpoint(tmp_path: Path):
    prep_dir = tmp_path / "preprocessed"
    for case_id in ("case_000", "case_001", "case_002"):
        _write_case(prep_dir, case_id)
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, train=["case_000", "case_001"], val=["case_002"], test=["case_002"])

    checkpoint_dir = tmp_path / "checkpoints"
    cfg = _make_cfg(
        tmp_path, prep_dir, splits_path, epochs=1, batch_size=2, checkpoint_dir=checkpoint_dir
    )

    final_metrics = run_training(cfg)

    assert isinstance(final_metrics, dict)
    assert (checkpoint_dir / LAST_CHECKPOINT_NAME).exists()


# ---------------------------------------------------------------------------
# 4. Resume path selection: empty dir -> fresh; explicit path wins
# ---------------------------------------------------------------------------


def test_select_resume_checkpoint_empty_dir_is_fresh_run(tmp_path: Path):
    cfg = OmegaConf.create(
        {"training": {"checkpoint": {"dir": str(tmp_path / "checkpoints"), "resume": None}}}
    )
    assert select_resume_checkpoint(cfg) is None


def test_select_resume_checkpoint_explicit_path_takes_precedence(tmp_path: Path):
    # A discoverable last.pt sits in the checkpoint dir...
    checkpoint_dir = tmp_path / "checkpoints"
    model = torch.nn.Conv3d(4, 3, kernel_size=3)
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(checkpoint_dir, model, optimizer, epoch=0, global_step=0)
    auto_discovered = checkpoint_dir / LAST_CHECKPOINT_NAME
    assert auto_discovered.is_file()

    # ...but a DIFFERENT checkpoint is named explicitly, elsewhere.
    explicit_dir = tmp_path / "elsewhere"
    save_checkpoint(explicit_dir, model, optimizer, epoch=5, global_step=0)
    explicit_path = explicit_dir / LAST_CHECKPOINT_NAME
    assert explicit_path.is_file()
    assert explicit_path != auto_discovered

    cfg = OmegaConf.create(
        {"training": {"checkpoint": {"dir": str(checkpoint_dir), "resume": str(explicit_path)}}}
    )

    selected = select_resume_checkpoint(cfg)

    assert selected == explicit_path
    assert selected != auto_discovered
