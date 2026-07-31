"""Tests for neurovision.training.trainer.Trainer.

CPU-only, tiny synthetic tensors, matching the config-construction style of
tests/test_losses.py and tests/test_checkpoint.py: an OmegaConf config built
inline to mirror configs/training/default.yaml's structure, and tmp_path for
every checkpoint directory.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovision.losses.segmentation import DiceBCELoss
from neurovision.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    load_checkpoint,
)
from neurovision.training.trainer import Trainer

CPU = torch.device("cpu")


class _SyntheticDataset(Dataset):
    """Tiny synthetic (image, label) pairs -- no real BraTS data needed."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.randn(4, 16, 16, 16),
            "label": (torch.rand(3, 16, 16, 16) > 0.5).float(),
        }


def _make_loaders(n: int = 4, batch_size: int = 2) -> tuple[DataLoader, DataLoader]:
    """4-item train loader (2 batches at batch_size=2) + a 2-case val loader."""
    train_loader = DataLoader(_SyntheticDataset(n), batch_size=batch_size, num_workers=0)
    # Validation always runs at batch size 1 over "whole volumes" per the spec.
    val_loader = DataLoader(_SyntheticDataset(2), batch_size=1, num_workers=0)
    return train_loader, val_loader


def _make_model() -> nn.Module:
    return nn.Conv3d(4, 3, 3, padding=1)


def _make_cfg(tmp_path, **overrides: object) -> object:
    """Builds a config mirroring configs/training/default.yaml's structure."""
    base = {
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "amp": True,
            "optimizer": {
                "name": "adamw",
                "lr": 1.0e-4,
                "weight_decay": 1.0e-5,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"name": "cosine", "warmup_epochs": 1, "min_lr": 1.0e-6},
            "grad_clip_norm": 1.0,
            "val_interval": 1,
            "log_interval": 1,
            "sliding_window": {"roi_size": [8, 8, 8], "sw_batch_size": 4, "overlap": 0.5},
            "checkpoint": {
                "dir": str(tmp_path / "checkpoints"),
                "save_every_n_epochs": 1,
                "keep_last_n": 2,
                "monitor": "val/dice_mean",
                "mode": "max",
                "resume": None,
            },
            "max_hours": 11.0,
        }
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.training, key, value, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# 1. Runs 2 epochs, writes last.pt
# ---------------------------------------------------------------------------


def test_train_runs_two_epochs_and_writes_last_checkpoint(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    trainer.train()

    assert (tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME).exists()


# ---------------------------------------------------------------------------
# 2. global_step counts optimizer steps, not batches, across a full train()
# ---------------------------------------------------------------------------


def test_global_step_counts_optimizer_steps_not_batches(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, grad_accum_steps=2)
    train_loader, val_loader = _make_loaders(n=4, batch_size=2)  # 2 batches/epoch
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    trainer.train()

    # 2 batches/epoch, grad_accum_steps=2 -> 1 optimizer step/epoch.
    # 2 epochs -> 2 steps total, NOT 4 (which is what counting batches would give).
    assert trainer.global_step == 2


# ---------------------------------------------------------------------------
# 3. Gradient accumulation: one optimizer step per 2 batches, not per batch
# ---------------------------------------------------------------------------


def test_grad_accumulation_steps_once_per_two_batches(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, grad_accum_steps=2)
    train_loader, val_loader = _make_loaders(n=4, batch_size=2)  # 2 batches this epoch
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    trainer.train_one_epoch(0)

    # If optimizer.zero_grad/step ran after every batch, global_step would be 2.
    # Accumulating over 2 batches means exactly 1 optimizer step this epoch.
    assert trainer.global_step == 1


# ---------------------------------------------------------------------------
# 4. Scheduler: warmup then cosine, computed from the formula
# ---------------------------------------------------------------------------


def test_scheduler_follows_warmup_then_cosine(tmp_path) -> None:
    lr = 1e-4
    min_lr = 1e-6
    warmup_epochs = 2
    epochs = 10

    cfg = _make_cfg(
        tmp_path,
        epochs=epochs,
        scheduler={"name": "cosine", "warmup_epochs": warmup_epochs, "min_lr": min_lr},
    )
    cfg.training.optimizer.lr = lr

    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    def current_lr() -> float:
        return trainer.optimizer.param_groups[0]["lr"]

    # Epoch 0: LR set at LambdaLR construction, before any .step().
    assert current_lr() == pytest.approx(lr * (0 + 1) / warmup_epochs)

    # Epoch 1: after one scheduler.step() (as train() does at the end of epoch 0).
    trainer.scheduler.step()
    assert current_lr() == pytest.approx(lr * (1 + 1) / warmup_epochs)

    lrs = [current_lr()]
    for _ in range(epochs - 2):
        trainer.scheduler.step()
        lrs.append(current_lr())

    # Post-warmup LR must decrease monotonically epoch over epoch. (lrs[1:] is
    # necessarily one element shorter than lrs -- this is a pairwise
    # comparison, not a strict zip.)
    for prev, curr in zip(lrs, lrs[1:]):
        assert curr <= prev + 1e-12

    # Cross-check one post-warmup point directly against the formula.
    min_ratio = min_lr / lr
    target_epoch = 5
    progress = (target_epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    expected = lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))

    trainer2 = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    for _ in range(target_epoch):
        trainer2.scheduler.step()
    assert trainer2.optimizer.param_groups[0]["lr"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Validation produces a finite val/dice_mean in [0, 1]
# ---------------------------------------------------------------------------


def test_validate_returns_finite_dice_mean_in_range(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    result = trainer.validate(0)

    assert "val/dice_mean" in result
    value = result["val/dice_mean"]
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# 6. Best-checkpoint tracking picks the better epoch, not the last one
# ---------------------------------------------------------------------------


def test_best_checkpoint_tracks_better_epoch_not_last(tmp_path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    # Epoch 0 gets the higher (better, mode="max") value; epoch 1 falls.
    forced_values = {0: 0.9, 1: 0.3}

    def fake_validate(epoch: int) -> dict[str, float]:
        return {"val/dice_mean": forced_values[epoch]}

    monkeypatch.setattr(trainer, "validate", fake_validate)
    trainer.train()

    assert trainer.best_metric == pytest.approx(0.9)

    best_path = tmp_path / "checkpoints" / BEST_CHECKPOINT_NAME
    assert best_path.exists()
    state = load_checkpoint(best_path, _make_model())
    assert state.best_metric == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 7. Resume restores epoch, global_step, and exact model weights
# ---------------------------------------------------------------------------


def test_resume_restores_epoch_step_and_weights(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, epochs=1)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME

    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    fresh_trainer.resume_from(checkpoint_path)

    assert fresh_trainer.start_epoch == 1
    assert fresh_trainer.global_step == trainer.global_step
    for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters(), strict=True):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# 8. max_hours stops the run before starting an epoch it cannot finish
# ---------------------------------------------------------------------------


def test_max_hours_stops_before_completing_all_epochs(tmp_path, caplog) -> None:
    cfg = _make_cfg(tmp_path, epochs=5, max_hours=1e-9)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    caplog.set_level(logging.WARNING)
    trainer.train()

    # An essentially-zero budget must stop training before any epoch's
    # periodic checkpoint is written.
    checkpoints_dir = tmp_path / "checkpoints"
    periodic_files = list(checkpoints_dir.glob("epoch_*.pt")) if checkpoints_dir.exists() else []
    assert len(periodic_files) < cfg.training.epochs

    assert any(
        "max_hours" in record.message or "budget" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# 9. wandb_run=None never touches wandb
# ---------------------------------------------------------------------------


def test_wandb_run_none_completes_training_without_error(tmp_path) -> None:
    cfg = _make_cfg(tmp_path)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(
        cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU, wandb_run=None
    )

    result = trainer.train()

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 10. AMP is off on CPU even when requested
# ---------------------------------------------------------------------------


def test_amp_disabled_on_cpu_even_when_requested(tmp_path) -> None:
    cfg = _make_cfg(tmp_path, amp=True)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    assert trainer.amp_is_enabled is False
    assert trainer.scaler.is_enabled() is False


def test_partial_accumulation_window_is_flushed_at_epoch_end(tmp_path) -> None:
    """A trailing partial accumulation window must still take an optimizer step.

    With 3 batches and grad_accum_steps=2, the first 2 batches trigger a step
    and the 3rd is left over. Without an end-of-epoch flush its gradients are
    neither applied nor zeroed, so they leak into the next epoch's first
    accumulation window -- stale gradients from a different LR silently mixed
    into a later step.
    """
    cfg = _make_cfg(tmp_path, grad_accum_steps=2, epochs=1)
    # 5 items at batch_size=2 -> 3 batches (2, 2, 1).
    train_loader = DataLoader(_SyntheticDataset(5), batch_size=2, num_workers=0)
    model = _make_model()
    trainer = Trainer(cfg, model, train_loader, None, DiceBCELoss(), CPU)

    trainer.train_one_epoch(0)

    # 3 batches / accum 2 -> 1 full window + 1 flushed partial window = 2 steps.
    assert trainer.global_step == 2

    # No gradient may survive the epoch boundary.
    for param in model.parameters():
        assert param.grad is None


def test_exact_accumulation_multiple_does_not_double_step(tmp_path) -> None:
    """The flush must not fire when batches divide evenly into the window."""
    cfg = _make_cfg(tmp_path, grad_accum_steps=2, epochs=1)
    train_loader = DataLoader(_SyntheticDataset(4), batch_size=2, num_workers=0)
    trainer = Trainer(cfg, _make_model(), train_loader, None, DiceBCELoss(), CPU)

    trainer.train_one_epoch(0)

    # 2 batches / accum 2 -> exactly 1 step, no extra flush step.
    assert trainer.global_step == 1
