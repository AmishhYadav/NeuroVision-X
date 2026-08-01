"""Regression guard for `Trainer.resume_from`: a killed Kaggle session must

resume into EXACTLY the state it left, not an approximately-similar one.

Each test below targets one specific way resume can go silently wrong --
"silently" meaning nothing raises and training appears to continue normally
while some piece of state has quietly been dropped or reset. CPU-only, tiny
synthetic tensors, tmp_path for every checkpoint directory.

Helper functions (`_SyntheticDataset`, `_make_loaders`, `_make_model`,
`_make_cfg`) are copied from `tests/test_trainer.py` rather than imported,
per that module being a separate test file with its own private helpers.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurovision.losses.segmentation import DiceBCELoss
from neurovision.training.checkpoint import LAST_CHECKPOINT_NAME, load_checkpoint
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
# 1. Core round trip -- the headline test
# ---------------------------------------------------------------------------


def test_resume_reproduces_exact_weights_epoch_and_step(tmp_path) -> None:
    """A resumed run must be byte-for-byte identical to the run it interrupted.

    If resume only approximately restored weights (e.g. because of a dtype
    cast, or because the wrong state_dict was loaded) training would look
    fine but silently diverge from the pre-resume trajectory. `torch.equal`,
    not `allclose`, is the point: this is a state *restore*, not a
    re-optimization that happens to land nearby.
    """
    cfg = _make_cfg(tmp_path, epochs=2)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    snapshot = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    global_step_before_resume = trainer.global_step

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    state = fresh_trainer.resume_from(checkpoint_path)

    for name, param in fresh_trainer.model.named_parameters():
        assert torch.equal(param, snapshot[name]), f"parameter {name!r} did not restore exactly"

    assert fresh_trainer.start_epoch == 2
    assert fresh_trainer.global_step == global_step_before_resume
    assert state.start_epoch == 2


# ---------------------------------------------------------------------------
# 2. Epoch counter continues -- only the remaining epochs run
# ---------------------------------------------------------------------------


def test_resume_runs_exactly_one_more_epoch_not_zero_or_two(tmp_path) -> None:
    """Catches an off-by-one in which epoch a resumed run starts at.

    Resuming one epoch too early re-trains the saved epoch, silently
    duplicating data and desynchronizing the epoch-indexed LR schedule.
    Resuming one too late makes a whole epoch of training vanish. Neither
    raises.
    """
    train_loader, val_loader = _make_loaders(n=4, batch_size=2)  # 2 batches/epoch
    cfg_two_epochs = _make_cfg(tmp_path, epochs=2)
    trainer = Trainer(cfg_two_epochs, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    # grad_accum_steps=1 -> 1 optimizer step per batch -> 2 steps/epoch.
    steps_per_epoch = 2
    assert trainer.global_step == 2 * steps_per_epoch

    # Fresh trainer built with epochs raised to 3, then resumed from the
    # checkpoint saved under the epochs=2 config.
    cfg_three_epochs = _make_cfg(tmp_path, epochs=3)
    fresh_train_loader, fresh_val_loader = _make_loaders(n=4, batch_size=2)
    fresh_trainer = Trainer(
        cfg_three_epochs, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_trainer.resume_from(checkpoint_path)
    assert fresh_trainer.start_epoch == 2

    global_step_before = fresh_trainer.global_step
    fresh_trainer.train()

    # Exactly ONE more epoch's worth of optimizer steps, not zero and not two.
    assert fresh_trainer.global_step == global_step_before + steps_per_epoch

    reloaded = load_checkpoint(checkpoint_path, _make_model())
    assert reloaded.start_epoch == 3


# ---------------------------------------------------------------------------
# 3. Optimizer momentum survives
# ---------------------------------------------------------------------------


def test_resume_restores_adam_moment_buffers_exactly(tmp_path) -> None:
    """Adam's two moment buffers must survive a resume, not restart cold.

    AdamW keeps `exp_avg` and `exp_avg_sq` per parameter. If the optimizer
    restarted cold on every resume, the first steps after each one would take
    badly-scaled updates while the moments re-warm -- across six Kaggle
    sessions that shows up as a sawtooth in the loss curve, with no error
    anywhere to point at it.
    """
    cfg = _make_cfg(tmp_path, epochs=2)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    first_param = trainer.optimizer.param_groups[0]["params"][0]
    opt_state = trainer.optimizer.state[first_param]
    assert "exp_avg" in opt_state and "exp_avg_sq" in opt_state
    exp_avg_before = opt_state["exp_avg"].clone()
    exp_avg_sq_before = opt_state["exp_avg_sq"].clone()

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    fresh_trainer.resume_from(checkpoint_path)

    fresh_first_param = fresh_trainer.optimizer.param_groups[0]["params"][0]
    fresh_opt_state = fresh_trainer.optimizer.state[fresh_first_param]

    assert torch.equal(fresh_opt_state["exp_avg"], exp_avg_before)
    assert torch.equal(fresh_opt_state["exp_avg_sq"], exp_avg_sq_before)


# ---------------------------------------------------------------------------
# 4. Scheduler continues on its curve, not a restarted warmup
# ---------------------------------------------------------------------------


def test_resume_continues_scheduler_curve_not_warmup_restart(tmp_path) -> None:
    """A scheduler that silently restarted on resume would send the LR back

    to warmup peak at the start of every Kaggle session, so the model would
    spend its whole schedule oscillating near peak LR and never actually
    anneal down to `min_lr`.
    """
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
    trainer.train()  # runs epochs 0 and 1, scheduler.step() called after each

    lr_after_two_epochs = trainer.optimizer.param_groups[0]["lr"]
    warmup_epoch_zero_lr = lr * (0 + 1) / warmup_epochs

    # Sanity: by epoch 2 the LR must already have moved on from the
    # epoch-0 warmup value, otherwise this test would not be exercising
    # anything interesting.
    assert lr_after_two_epochs != pytest.approx(warmup_epoch_zero_lr)

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    fresh_trainer.resume_from(checkpoint_path)

    resumed_lr = fresh_trainer.optimizer.param_groups[0]["lr"]
    assert resumed_lr == pytest.approx(lr_after_two_epochs)
    # Not reset back to the epoch-0 warmup LR.
    assert resumed_lr != pytest.approx(warmup_epoch_zero_lr)


# ---------------------------------------------------------------------------
# 5. best_metric carries over
# ---------------------------------------------------------------------------


def test_resume_restores_best_metric_not_sentinel(tmp_path, monkeypatch) -> None:
    """A reset `best_metric` (back to the -inf/+inf sentinel) means the very

    next validated epoch trivially "beats" it, silently overwriting `best.pt`
    with a worse model than the one actually recorded as best so far.
    """
    cfg = _make_cfg(tmp_path, epochs=2)  # mode="max" per _make_cfg's checkpoint block
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)

    # Epoch 0 gets the higher (better, mode="max") value; epoch 1 falls, so
    # best_metric at the end of training is 0.9, not the last epoch's 0.3.
    forced_values = {0: 0.9, 1: 0.3}

    def fake_validate(epoch: int) -> dict[str, float]:
        return {"val/dice_mean": forced_values[epoch]}

    monkeypatch.setattr(trainer, "validate", fake_validate)
    trainer.train()
    assert trainer.best_metric == pytest.approx(0.9)

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    fresh_trainer.resume_from(checkpoint_path)

    assert fresh_trainer.best_metric == pytest.approx(0.9)
    assert fresh_trainer.best_metric != float("-inf")


# ---------------------------------------------------------------------------
# 6. Resume does not depend on having trained in the same process
# ---------------------------------------------------------------------------


def test_load_checkpoint_into_independent_model_matches_trained_weights(tmp_path) -> None:
    """This is the property that makes a Kaggle session boundary work at all:

    a fresh process, with a fresh model object that never saw the training
    run, must be able to load the checkpoint and get identical weights. If
    resume secretly depended on in-process state (e.g. a closure over the
    original model), it would work in this test file but fail the moment
    training and resuming happen in genuinely separate processes.
    """
    cfg = _make_cfg(tmp_path, epochs=2)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    snapshot = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    independent_model = _make_model()  # built without any reference to `trainer`
    load_checkpoint(checkpoint_path, independent_model)

    for name, param in independent_model.named_parameters():
        assert torch.equal(param, snapshot[name])


# ---------------------------------------------------------------------------
# 7. RNG continuity through Trainer.resume_from
# ---------------------------------------------------------------------------


def test_resume_from_reproduces_rng_sequences(tmp_path) -> None:
    """`test_checkpoint.py` proves RNG round-trips at the `save_checkpoint`/

    `load_checkpoint` level; this asserts it through `Trainer.resume_from`,
    the path a real training run actually takes. Without this, augmentation
    and dropout draws would silently diverge from the pre-kill run on every
    resumed Kaggle session, making runs non-reproducible even with a fixed
    seed.
    """
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cfg = _make_cfg(tmp_path, epochs=1)
    train_loader, val_loader = _make_loaders()
    trainer = Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)
    trainer.train()

    # Draw sequences right after training/saving -- this is the state that
    # must be reproducible after a later resume.
    python_seq_1 = [random.random() for _ in range(5)]
    numpy_seq_1 = np.random.rand(5).tolist()
    torch_seq_1 = torch.rand(5).tolist()

    # Perturb all three generators so a resume that silently did nothing to
    # RNG state would still fail this test, not just pass by coincidence.
    for _ in range(50):
        random.random()
        np.random.rand()
        torch.rand(1)

    checkpoint_path = tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    fresh_train_loader, fresh_val_loader = _make_loaders()
    fresh_trainer = Trainer(
        cfg, _make_model(), fresh_train_loader, fresh_val_loader, DiceBCELoss(), CPU
    )
    # Building fresh_trainer's model above also draws from the torch RNG
    # (weight init), but resume_from restores RNG state afterwards, so that
    # draw must not leak into the sequences compared below.
    fresh_trainer.resume_from(checkpoint_path)

    python_seq_2 = [random.random() for _ in range(5)]
    numpy_seq_2 = np.random.rand(5).tolist()
    torch_seq_2 = torch.rand(5).tolist()

    assert python_seq_1 == python_seq_2
    assert numpy_seq_1 == numpy_seq_2
    assert torch_seq_1 == pytest.approx(torch_seq_2)
