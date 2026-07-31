"""Tests for neurovision.training.checkpoint.

All tests run on CPU with a tiny model (`nn.Conv3d(4, 3, 3)`), a real
`torch.optim.Adam`, and a real `torch.optim.lr_scheduler.StepLR`, and use
pytest's `tmp_path` fixture for every path -- checkpoint survival on Kaggle is
the whole point of this module, so these tests exercise the atomic write,
weights_only-safe load, defensive key handling, and periodic pruning paths
directly rather than trusting them by inspection.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from neurovision.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    find_resume_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


def _make_model() -> nn.Module:
    """A tiny 3D conv model -- big enough to have real state, small and fast."""
    return nn.Conv3d(4, 3, kernel_size=3)


def _make_training_objects() -> (
    tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return model, optimizer, scheduler


def _make_cfg() -> object:
    """A tiny OmegaConf config with an interpolation, mirroring test_losses.py style."""
    base = {
        "data": {"num_classes": 3},
        "model": {"out_channels": "${data.num_classes}"},
    }
    return OmegaConf.create(base)


def _train_step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """One tiny forward/backward/step so optimizer state is non-trivial."""
    x = torch.randn(1, 4, 8, 8, 8)
    out = model(x)
    loss = out.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_restores_model_optimizer_scheduler_and_metadata(tmp_path):
    model, optimizer, scheduler = _make_training_objects()
    _train_step(model, optimizer)
    scheduler.step()

    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        epoch=5,
        global_step=123,
        scheduler=scheduler,
        best_metric=0.75,
        best_metric_name="dice_wt",
        wandb_run_id="abc123",
    )

    new_model, new_optimizer, new_scheduler = _make_training_objects()
    state = load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME,
        new_model,
        new_optimizer,
        scheduler=new_scheduler,
    )

    for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p1, p2)

    assert optimizer.state_dict()["state"].keys() == new_optimizer.state_dict()["state"].keys()
    assert new_scheduler.last_epoch == scheduler.last_epoch

    assert state.global_step == 123
    assert state.best_metric == pytest.approx(0.75)
    assert state.best_metric_name == "dice_wt"
    assert state.wandb_run_id == "abc123"


def test_start_epoch_is_saved_epoch_plus_one(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=7, global_step=0)

    state = load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME, _make_model(), torch.optim.Adam(_make_model().parameters())
    )
    assert state.start_epoch == 8


# ---------------------------------------------------------------------------
# RNG determinism -- the test that proves resume actually works
# ---------------------------------------------------------------------------


def test_rng_state_round_trip_reproduces_identical_sequences(tmp_path):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0)

    # Draw sequences right after saving (state at save time is what must
    # be reproducible after a later load).
    python_seq_1 = [random.random() for _ in range(5)]
    numpy_seq_1 = np.random.rand(5).tolist()
    torch_seq_1 = torch.rand(5).tolist()

    # Perturb all three generators so a naive "load did nothing" bug would
    # still pass if we only checked "the load call didn't raise".
    for _ in range(50):
        random.random()
        np.random.rand()
        torch.rand(1)

    load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME, _make_model(), torch.optim.Adam(_make_model().parameters())
    )

    python_seq_2 = [random.random() for _ in range(5)]
    numpy_seq_2 = np.random.rand(5).tolist()
    torch_seq_2 = torch.rand(5).tolist()

    assert python_seq_1 == python_seq_2
    assert numpy_seq_1 == numpy_seq_2
    assert torch_seq_1 == pytest.approx(torch_seq_2)


# ---------------------------------------------------------------------------
# Config round trip
# ---------------------------------------------------------------------------


def test_config_round_trip_stores_resolved_interpolation(tmp_path):
    cfg = _make_cfg()
    assert cfg.model.out_channels == 3  # sanity: interpolation resolves to 3

    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0, cfg=cfg)

    state = load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME, _make_model(), torch.optim.Adam(_make_model().parameters())
    )

    assert state.config is not None
    assert state.config.model.out_channels == 3
    assert state.config.data.num_classes == 3
    # Returned object behaves like a normal config: attribute + item access.
    assert OmegaConf.select(state.config, "model.out_channels") == 3


# ---------------------------------------------------------------------------
# Defensive / missing-key handling
# ---------------------------------------------------------------------------


def test_missing_optional_keys_are_tolerated_with_warning(tmp_path, caplog):
    model, optimizer, scheduler = _make_training_objects()
    last_path = save_checkpoint(
        tmp_path, model, optimizer, epoch=0, global_step=0, scheduler=scheduler
    )

    checkpoint = torch.load(last_path, weights_only=True)
    for key in [
        "scheduler_state_dict",
        "scaler_state_dict",
        "wandb_run_id",
        "global_step",
        "random_state",
        "numpy_random_state",
        "torch_rng_state",
    ]:
        checkpoint.pop(key, None)
    torch.save(checkpoint, last_path)

    caplog.set_level(logging.WARNING)
    state = load_checkpoint(
        last_path,
        _make_model(),
        torch.optim.Adam(_make_model().parameters()),
        scheduler=torch.optim.lr_scheduler.StepLR(
            torch.optim.Adam(_make_model().parameters()), step_size=2
        ),
    )

    assert state.global_step == 0  # defaulted
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_missing_model_state_dict_raises_key_error(tmp_path):
    model, optimizer, _ = _make_training_objects()
    last_path = save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0)

    checkpoint = torch.load(last_path, weights_only=True)
    del checkpoint["model_state_dict"]
    torch.save(checkpoint, last_path)

    with pytest.raises(KeyError):
        load_checkpoint(last_path, _make_model())


# ---------------------------------------------------------------------------
# best.pt behaviour
# ---------------------------------------------------------------------------


def test_best_checkpoint_written_only_when_requested(tmp_path):
    model, optimizer, _ = _make_training_objects()

    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0, is_best=False)
    assert not (tmp_path / BEST_CHECKPOINT_NAME).exists()

    save_checkpoint(tmp_path, model, optimizer, epoch=1, global_step=0, is_best=True)
    assert (tmp_path / BEST_CHECKPOINT_NAME).exists()


# ---------------------------------------------------------------------------
# Periodic snapshots + pruning
# ---------------------------------------------------------------------------


def test_periodic_snapshots_pruned_to_keep_last_n_numerically(tmp_path):
    model, optimizer, _ = _make_training_objects()

    # Epochs 8..12 cross a digit boundary: a lexicographic sort of
    # "epoch_0009.pt" vs "epoch_0012.pt" still happens to be correct with
    # 4-digit padding, but we assert the *numeric* set survives regardless,
    # which is what actually proves the sort key is the parsed int.
    for epoch in range(8, 13):
        save_checkpoint(
            tmp_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=0,
            periodic=True,
            keep_last_n=3,
        )

    periodic_files = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
    assert periodic_files == ["epoch_0010.pt", "epoch_0011.pt", "epoch_0012.pt"]

    # last.pt / best.pt are never touched by pruning.
    assert (tmp_path / LAST_CHECKPOINT_NAME).exists()


def test_pruning_never_deletes_last_or_best(tmp_path):
    model, optimizer, _ = _make_training_objects()

    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0, is_best=True)
    for epoch in range(1, 6):
        save_checkpoint(
            tmp_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=0,
            periodic=True,
            keep_last_n=1,
        )

    assert (tmp_path / LAST_CHECKPOINT_NAME).exists()
    assert (tmp_path / BEST_CHECKPOINT_NAME).exists()
    periodic_files = list(tmp_path.glob("epoch_*.pt"))
    assert len(periodic_files) == 1


def test_keep_last_n_non_positive_keeps_all(tmp_path):
    model, optimizer, _ = _make_training_objects()
    for epoch in range(3):
        save_checkpoint(
            tmp_path, model, optimizer, epoch=epoch, global_step=0, periodic=True, keep_last_n=0
        )
    assert len(list(tmp_path.glob("epoch_*.pt"))) == 3


# ---------------------------------------------------------------------------
# Atomic write leaves no temp files
# ---------------------------------------------------------------------------


def test_atomic_write_leaves_no_temp_files(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0, is_best=True, periodic=True)

    all_files = sorted(p.name for p in tmp_path.iterdir())
    for name in all_files:
        assert not name.startswith(".")
        assert "tmp" not in name.lower()
    assert set(all_files) == {LAST_CHECKPOINT_NAME, BEST_CHECKPOINT_NAME, "epoch_0000.pt"}


# ---------------------------------------------------------------------------
# find_resume_checkpoint
# ---------------------------------------------------------------------------


def test_find_resume_checkpoint_none_when_missing(tmp_path):
    assert find_resume_checkpoint(tmp_path) is None
    assert find_resume_checkpoint(tmp_path / "does_not_exist") is None


def test_find_resume_checkpoint_returns_last_path(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0)

    found = find_resume_checkpoint(tmp_path)
    assert found == tmp_path / LAST_CHECKPOINT_NAME


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_load_nonexistent_path_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", _make_model())


def test_strict_false_allows_extra_unexpected_parameter(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0)

    class ModelWithExtraParam(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(4, 3, kernel_size=3)
            self.extra = nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    bigger_model = ModelWithExtraParam()
    with pytest.raises(RuntimeError):
        load_checkpoint(tmp_path / LAST_CHECKPOINT_NAME, bigger_model, strict=True)

    # strict=False tolerates the model having keys the checkpoint lacks.
    load_checkpoint(tmp_path / LAST_CHECKPOINT_NAME, bigger_model, strict=False)


# ---------------------------------------------------------------------------
# weights_only=True compatibility regression guard
# ---------------------------------------------------------------------------


def test_saved_checkpoint_loads_with_weights_only_true_directly(tmp_path):
    model, optimizer, scheduler = _make_training_objects()
    cfg = _make_cfg()
    last_path = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        scheduler=scheduler,
        cfg=cfg,
        wandb_run_id="run-xyz",
    )

    # This is the regression guard: if anyone later adds an unsafe object to
    # the payload (e.g. a raw DictConfig or a numpy ndarray), this call
    # raises instead of quietly working until the model changes on Kaggle.
    checkpoint = torch.load(last_path, weights_only=True)
    assert "model_state_dict" in checkpoint


# ---------------------------------------------------------------------------
# best_metric_mode
# ---------------------------------------------------------------------------


def test_best_metric_mode_round_trips(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(
        tmp_path,
        model,
        optimizer,
        epoch=0,
        global_step=0,
        best_metric=0.12,
        best_metric_name="ece",
        best_metric_mode="min",
    )

    state = load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME, _make_model(), torch.optim.Adam(_make_model().parameters())
    )
    assert state.best_metric_mode == "min"
    assert state.best_metric_name == "ece"
    assert state.best_metric == pytest.approx(0.12)


def test_best_metric_mode_defaults_to_max(tmp_path):
    model, optimizer, _ = _make_training_objects()
    save_checkpoint(tmp_path, model, optimizer, epoch=0, global_step=0, best_metric=0.9)

    state = load_checkpoint(
        tmp_path / LAST_CHECKPOINT_NAME, _make_model(), torch.optim.Adam(_make_model().parameters())
    )
    assert state.best_metric_mode == "max"


def test_invalid_best_metric_mode_raises(tmp_path):
    model, optimizer, _ = _make_training_objects()
    with pytest.raises(ValueError):
        save_checkpoint(
            tmp_path, model, optimizer, epoch=0, global_step=0, best_metric_mode="minimize"
        )


def test_missing_best_metric_falls_back_per_mode(tmp_path):
    """A minimized metric must fall back to +inf, not -inf.

    Falling back to -inf under mode 'min' would mean no later value ever
    compares as better, so best.pt would silently stop being written for the
    rest of the run -- with no error anywhere.
    """
    model, optimizer, _ = _make_training_objects()
    last_path = save_checkpoint(
        tmp_path, model, optimizer, epoch=0, global_step=0, best_metric_mode="min"
    )

    payload = torch.load(last_path, weights_only=True)
    del payload["best_metric"]
    torch.save(payload, last_path)

    state = load_checkpoint(last_path, _make_model(), torch.optim.Adam(_make_model().parameters()))
    assert state.best_metric == float("inf")
    assert state.best_metric_mode == "min"
