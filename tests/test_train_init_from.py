"""Tests for `training.checkpoint.init_from` -- fine-tuning from a foreign checkpoint.

Exercises `scripts/train.py`'s `resolve_init_from_checkpoint` and
`apply_resume_or_init` directly against small, real `Trainer` objects, reusing
the tiny-model/tiny-config helpers from `tests/test_trainer.py` (imported, not
duplicated) rather than building a synthetic BraTS dataset -- `init_from` is
purely a checkpoint-loading decision made before any data is touched, so no
dataloader is needed here at all.

The script lives under scripts/, not src/, so it is loaded via
`tests.script_loader.load_script`, exactly like tests/test_train_script.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from neurovision.losses.segmentation import DiceBCELoss
from neurovision.training.checkpoint import LAST_CHECKPOINT_NAME, save_checkpoint
from neurovision.training.trainer import Trainer
from tests.script_loader import load_script
from tests.test_trainer import CPU, _make_cfg, _make_loaders, _make_model

train_script = load_script("train")

resolve_init_from_checkpoint = train_script.resolve_init_from_checkpoint
apply_resume_or_init = train_script.apply_resume_or_init


def _write_source_checkpoint(
    source_dir: Path,
    model: nn.Module,
    *,
    epoch: int = 5,
    global_step: int = 100,
) -> Path:
    """Saves a full checkpoint (with REAL optimizer state) for `model`.

    Takes one real optimizer step first, so `optimizer_state_dict` in the
    saved payload is non-empty -- otherwise a test asserting "init_from does
    NOT restore optimizer state" would pass trivially, whether or not the
    restore code path is even reached.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dummy_input = torch.randn(1, 4, 8, 8, 8)
    output = model(dummy_input)
    dummy_label = torch.rand_like(output)  # matches output channels for any model width
    loss = DiceBCELoss()(output, dummy_label)
    loss.backward()
    optimizer.step()  # populates optimizer.state with real Adam moment buffers

    save_checkpoint(
        source_dir,
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        best_metric=0.77,
        best_metric_name="val/dice_mean",
        best_metric_mode="max",
        wandb_run_id="source-run-id",
    )
    return source_dir / LAST_CHECKPOINT_NAME


def _make_trainer(tmp_path: Path, **checkpoint_overrides: object) -> Trainer:
    """A fresh Trainer over the tiny synthetic loaders from test_trainer.py."""
    cfg = _make_cfg(tmp_path, checkpoint=checkpoint_overrides)
    train_loader, val_loader = _make_loaders()
    return Trainer(cfg, _make_model(), train_loader, val_loader, DiceBCELoss(), CPU)


# ---------------------------------------------------------------------------
# 1. init_from loads weights only: fresh optimizer, epoch 0, global_step 0
# ---------------------------------------------------------------------------


def test_init_from_loads_weights_with_fresh_optimizer_and_zeroed_state(tmp_path: Path) -> None:
    source_model = _make_model()
    source_path = _write_source_checkpoint(tmp_path / "source", source_model)

    trainer = _make_trainer(tmp_path, dir=str(tmp_path / "checkpoints"), init_from=str(source_path))

    resume_state = apply_resume_or_init(trainer.cfg, trainer)

    assert resume_state is None
    for src_param, tgt_param in zip(
        source_model.parameters(), trainer.model.parameters(), strict=True
    ):
        assert torch.equal(src_param, tgt_param)

    # Optimizer/scheduler/scaler are the fresh ones Trainer.__init__ built --
    # in particular, NO per-parameter Adam state (the source's optimizer had
    # real state from its own step; none of it should have crossed over).
    assert len(trainer.optimizer.state) == 0
    assert trainer.start_epoch == 0
    assert trainer.global_step == 0
    assert trainer.best_metric == float("-inf")  # fresh sentinel for mode="max"


# ---------------------------------------------------------------------------
# 2. Auto-found last.pt beats init_from
# ---------------------------------------------------------------------------


def test_last_pt_in_checkpoint_dir_wins_over_init_from(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    # A last.pt already sits in the run's own checkpoint dir (e.g. left by an
    # earlier, pre-empted attempt of this same fine-tune).
    resident_model = _make_model()
    _write_source_checkpoint(checkpoint_dir, resident_model, epoch=3, global_step=30)

    # A DIFFERENT checkpoint is also named as init_from, elsewhere.
    other_model = _make_model()
    init_from_path = _write_source_checkpoint(tmp_path / "elsewhere", other_model, epoch=5)

    trainer = _make_trainer(
        tmp_path, dir=str(checkpoint_dir), init_from=str(init_from_path), resume=None
    )

    resume_state = apply_resume_or_init(trainer.cfg, trainer)

    assert resume_state is not None
    assert resume_state.start_epoch == 4  # resident's epoch(3) + 1, NOT init_from's
    assert trainer.start_epoch == 4
    assert trainer.global_step == 30
    for resident_param, tgt_param in zip(
        resident_model.parameters(), trainer.model.parameters(), strict=True
    ):
        assert torch.equal(resident_param, tgt_param)


# ---------------------------------------------------------------------------
# 3. Explicit resume beats init_from
# ---------------------------------------------------------------------------


def test_explicit_resume_wins_over_init_from(tmp_path: Path) -> None:
    explicit_model = _make_model()
    explicit_path = _write_source_checkpoint(
        tmp_path / "explicit", explicit_model, epoch=7, global_step=70
    )

    init_from_model = _make_model()
    init_from_path = _write_source_checkpoint(
        tmp_path / "elsewhere", init_from_model, epoch=5, global_step=50
    )

    trainer = _make_trainer(
        tmp_path,
        dir=str(tmp_path / "checkpoints"),  # empty: no auto-discoverable last.pt
        resume=str(explicit_path),
        init_from=str(init_from_path),
    )

    resume_state = apply_resume_or_init(trainer.cfg, trainer)

    assert resume_state is not None
    assert resume_state.start_epoch == 8  # explicit's epoch(7) + 1
    assert trainer.global_step == 70


# ---------------------------------------------------------------------------
# 4. init_from inside training.checkpoint.dir -> ValueError, end-to-end,
#    BEFORE any resume happens (the exact hazard the guard exists for).
# ---------------------------------------------------------------------------


def test_init_from_inside_checkpoint_dir_raises_before_any_resume(tmp_path: Path) -> None:
    """Reusing the SOURCE run's own checkpoint.dir must raise, not resume.

    checkpoint.dir here ALREADY holds a resumable last.pt (as it would for
    the source run's own directory), and init_from points at a different
    file in that SAME directory. If init_from were only checked after
    confirming no resume is available, this config would silently RESUME
    the resident last.pt instead of ever raising -- apply_resume_or_init must
    validate init_from FIRST so this is caught regardless.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    resident_model = _make_model()
    _write_source_checkpoint(checkpoint_dir, resident_model, epoch=79, global_step=1000)

    # A second file in that SAME directory -- content doesn't matter, only
    # its existence and location, since the guard must fire before load.
    trap_path = checkpoint_dir / "some_other_checkpoint.pt"
    trap_path.write_bytes(b"not a real checkpoint, only existence is checked here")

    trainer = _make_trainer(tmp_path, dir=str(checkpoint_dir), init_from=str(trap_path))

    with pytest.raises(ValueError, match="init_from"):
        apply_resume_or_init(trainer.cfg, trainer)

    # No resume happened: the trainer is untouched, still at its fresh defaults.
    assert trainer.start_epoch == 0
    assert trainer.global_step == 0


# ---------------------------------------------------------------------------
# 5. Missing init_from file -> FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_init_from_file_raises_file_not_found_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.pt"
    cfg = _make_cfg(
        tmp_path,
        checkpoint={"dir": str(tmp_path / "checkpoints"), "init_from": str(missing_path)},
    )

    with pytest.raises(FileNotFoundError):
        resolve_init_from_checkpoint(cfg)


# ---------------------------------------------------------------------------
# 6. Architecture mismatch raises rather than silently partial-loading
# ---------------------------------------------------------------------------


def test_init_from_architecture_mismatch_raises(tmp_path: Path) -> None:
    # Wider output channels (5 instead of 3) -> same parameter name, different
    # tensor shape, so strict=True load_state_dict must raise, not silently
    # skip or partially load the mismatched weight.
    wider_model = nn.Conv3d(4, 5, 3, padding=1)
    source_path = _write_source_checkpoint(tmp_path / "source", wider_model)

    trainer = _make_trainer(tmp_path, dir=str(tmp_path / "checkpoints"), init_from=str(source_path))

    with pytest.raises(RuntimeError):
        apply_resume_or_init(trainer.cfg, trainer)


# ---------------------------------------------------------------------------
# 7. init_from=null (the default) leaves behaviour completely unchanged
# ---------------------------------------------------------------------------


def test_init_from_null_is_a_no_op(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, dir=str(tmp_path / "checkpoints"), init_from=None)

    resume_state = apply_resume_or_init(trainer.cfg, trainer)

    assert resume_state is None
    assert trainer.start_epoch == 0
    assert trainer.global_step == 0
