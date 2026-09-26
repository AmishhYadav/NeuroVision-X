"""Hydra entry point for training.

Wires together data, model, loss, and `neurovision.training.trainer.Trainer`,
then hands off to the trainer's own loop. This runs unmodified on the Mac's
CPU (for a smoke test) or on a Kaggle GPU (for a real run) -- device is
resolved once, from config, via `neurovision.utils.device.get_device`.

Example usage:

    python scripts/train.py data.root_dir=/path/to/preprocessed wandb.mode=disabled

The wiring is split into small functions (`build_dataloaders`, `init_wandb`,
`select_resume_checkpoint`, `resolve_init_from_checkpoint`,
`apply_resume_or_init`, `run_training`) rather than one long `main`, so each
piece can be unit tested without going through Hydra -- see
tests/test_train_script.py and tests/test_train_init_from.py (the last for
`training.checkpoint.init_from`, the fine-tune entry point).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import build_train_transforms, build_val_transforms

# Importing these registers the "unet3d"/"swinunetr" and "dice_ce" builders
# (the @register_model / @register_loss decorators run on import) before
# build_model / build_loss are ever called below. Importing the registry
# submodule alone would already trigger this indirectly, since Python first
# runs the parent package's __init__.py -- but spelling it out here makes the
# dependency obvious to a reader, matching the convention in test_models.py.
from neurovision.losses import segmentation  # noqa: F401
from neurovision.losses.registry import build_loss
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import ResumeState, find_resume_checkpoint, load_checkpoint
from neurovision.training.trainer import Trainer
from neurovision.utils.device import get_device
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/preprocess.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def build_dataloaders(cfg: DictConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    """Builds the train and validation `DataLoader`s from the frozen splits.

    Args:
        cfg: The full composed Hydra config.
        device: The resolved torch device, used only to decide whether
            `pin_memory` should be on (CUDA only -- pinning on CPU/MPS buys
            nothing and just costs host memory).

    Returns:
        `(train_loader, val_loader)`. The train loader yields patch-sized,
        augmented batches at `cfg.training.batch_size`. The validation
        loader yields *whole volumes* one at a time (`batch_size=1`):
        `Trainer.validate` runs sliding-window inference over full-size
        cases of differing shapes, and a batch size above 1 would fail to
        collate volumes that are not all the same size.
    """
    splits = load_splits(cfg.data.splits.path)
    prep_dir = cfg.data.preprocessing.out_dir

    train_ids = splits["train"]
    val_ids = splits["val"]

    # Overfit sanity check: train and validate on the same handful of cases.
    # Deliberately loud, because silently validating on training data would
    # invalidate any number reported from this run.
    overfit_n = cfg.data.get("overfit_n")
    if overfit_n:
        train_ids = list(train_ids[:overfit_n])
        val_ids = list(train_ids)
        logger.warning(
            "data.overfit_n=%d: training AND validating on the same %d case(s): %s. "
            "This is a pipeline sanity check -- the resulting metrics are memorization, "
            "not performance, and must never be reported.",
            overfit_n,
            len(train_ids),
            train_ids,
        )

    train_dicts = build_data_dicts(train_ids, prep_dir)
    val_dicts = build_data_dicts(val_ids, prep_dir)

    train_transform = build_train_transforms(cfg)
    val_transform = build_val_transforms(cfg)

    dataset_type = cfg.data.dataset_type
    cache_rate = cfg.data.cache_rate
    cache_dir = cfg.data.cache_dir
    num_workers = cfg.data.num_workers

    train_ds = build_dataset(
        train_dicts,
        train_transform,
        dataset_type=dataset_type,
        cache_rate=cache_rate,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )
    val_ds = build_dataset(
        val_dicts,
        val_transform,
        dataset_type=dataset_type,
        cache_rate=cache_rate,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )

    pin_memory = device.type == "cuda"

    # collate_fn=list_data_collate (not the plain torch default) is required
    # here, not optional: build_train_transforms' RandCropByPosNegLabeld
    # returns a LIST of `samples_per_volume` dicts per case, even when
    # samples_per_volume == 1. Plain default_collate would batch that as a
    # list of per-position dicts instead of one flat dict, and
    # Trainer.train_one_epoch's `batch["image"]` would raise a TypeError on
    # the very first step. list_data_collate flattens each case's sample
    # list before batching, so a `batch_size=B` loader actually yields
    # `B * samples_per_volume` patches per step -- this also correctly
    # collates build_val_transforms' plain (non-list) dicts.
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=list_data_collate,
    )
    # batch_size=1, shuffle=False: see docstring above -- sliding-window
    # validation needs whole, differently-shaped volumes one at a time.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader


def init_wandb(cfg: DictConfig, resume_state: ResumeState | None) -> Any | None:
    """Starts (or resumes) a W&B run, or skips W&B entirely.

    `wandb` is imported lazily, inside this function, so that code paths
    which never call this function (e.g. the test suite, or a CPU smoke test
    with `wandb.mode=disabled`) have no hard dependency on wandb being
    configured or even importable.

    Args:
        cfg: The full composed Hydra config. Reads `cfg.wandb.*` and
            `cfg.experiment_name`.
        resume_state: The `ResumeState` returned by `Trainer.resume_from`,
            or None for a fresh run. When not None, its `wandb_run_id` is
            used to resume logging into the SAME W&B run instead of starting
            a new, orphaned one.

    Returns:
        The active W&B run object, or None if `cfg.wandb.mode == "disabled"`.
    """
    if cfg.wandb.mode == "disabled":
        logger.info("wandb.mode=disabled: skipping W&B entirely, no wandb.init() call made.")
        return None

    import wandb

    init_kwargs: dict[str, Any] = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "mode": cfg.wandb.mode,
        "name": cfg.experiment_name,
        "tags": list(cfg.wandb.tags),
        "group": cfg.wandb.group,
        # resolve=True bakes in interpolations (e.g. ${data.num_classes}) so
        # W&B records the values that actually ran, not unresolved templates.
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if resume_state is not None:
        # Resuming into the SAME run requires the id that was saved in the
        # checkpoint; wandb.init(resume="allow") then continues that run's
        # history instead of creating a new one.
        init_kwargs["id"] = resume_state.wandb_run_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)


def select_resume_checkpoint(cfg: DictConfig) -> Path | None:
    """Decides which checkpoint (if any) a run should resume from.

    An explicit `cfg.training.checkpoint.resume` always takes precedence over
    auto-discovery: if a checkpoint is named explicitly, that is what the run
    resumes from, even when a different `last.pt` also happens to sit in the
    checkpoint directory (e.g. left over from an earlier, unrelated run).

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The path to resume from, or None to start a fresh run.
    """
    resume_path = cfg.training.checkpoint.resume
    if resume_path is not None:
        return Path(resume_path)
    return find_resume_checkpoint(cfg.training.checkpoint.dir)


def resolve_init_from_checkpoint(cfg: DictConfig) -> Path | None:
    """Validates and resolves `cfg.training.checkpoint.init_from`, if set.

    Called by `apply_resume_or_init` BEFORE `select_resume_checkpoint` is
    even consulted -- see that function's docstring for why the order
    matters: this validation must run whether or not a resume is about to
    happen, so a fine-tune config that accidentally reuses the SOURCE run's
    own `checkpoint.dir` is caught immediately, not only on runs where no
    `last.pt` happens to already be sitting there.

    `.get(...)` (not plain attribute access) is deliberate: `init_from` is a
    new key, and a config built without it (e.g. an older experiment config,
    or a test fixture that predates this key) must be read as "not set"
    rather than raise `ConfigAttributeError`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The resolved `init_from` path, or None if it is not set.

    Raises:
        FileNotFoundError: If `init_from` is set but no file exists there.
        ValueError: If `init_from` resolves to a path inside this run's own
            `training.checkpoint.dir`. That directory is exactly where this
            run auto-discovers a `last.pt` to resume from -- if `init_from`
            pointed into the SOURCE run's own checkpoint dir (which already
            has a `last.pt`), this run would silently RESUME the source run
            at its last saved epoch instead of fine-tuning from `init_from`'s
            weights at all.
    """
    init_from = cfg.training.checkpoint.get("init_from")
    if init_from is None:
        return None

    init_path = Path(init_from).resolve()
    if not init_path.is_file():
        raise FileNotFoundError(
            f"training.checkpoint.init_from={init_from!r} does not exist "
            f"(resolved to {init_path}). It must point at a checkpoint file "
            "written by neurovision.training.checkpoint.save_checkpoint."
        )

    checkpoint_dir = Path(cfg.training.checkpoint.dir).resolve()
    if init_path.is_relative_to(checkpoint_dir):
        raise ValueError(
            f"training.checkpoint.init_from={init_path} lies inside this run's own "
            f"training.checkpoint.dir={checkpoint_dir}. This run would auto-resume the "
            "source run's last.pt instead of fine-tuning -- give the fine-tune its own "
            "training.checkpoint.dir (e.g. a distinct experiment_name), separate from the "
            "source run's, and point init_from at a checkpoint in THAT (the source's) "
            "directory instead."
        )

    return init_path


def apply_resume_or_init(cfg: DictConfig, trainer: Trainer) -> ResumeState | None:
    """Resumes, fine-tune-initializes, or leaves `trainer` fresh, per config.

    Implements the full precedence in one place: explicit
    `training.checkpoint.resume` > an auto-found `<training.checkpoint.dir>/
    last.pt` > `training.checkpoint.init_from` > fresh init.

    `init_from` is validated FIRST, unconditionally, before
    `select_resume_checkpoint` ever runs -- even on a run that is about to
    resume anyway. This is deliberate, not an optimization: if `init_from`
    were only checked after confirming no resume is available, a fine-tune
    config that mistakenly points `training.checkpoint.dir` at the SOURCE
    run's own checkpoint directory would find that directory's `last.pt`,
    resume the source run silently, and never reach the `init_from` check at
    all -- exactly the failure the inside-`checkpoint.dir` guard exists to
    catch. Validating first closes that hole; `init_from` is still only
    *applied* below once resuming has been ruled out, so a legitimately
    pre-empted fine-tune (whose OWN checkpoint.dir, distinct from the
    source's, holds its own `last.pt`) still resumes correctly.

    `init_from` restores model weights ONLY, via `load_checkpoint` with
    `optimizer=None` and `restore_rng=False`: the optimizer, scheduler and
    AMP scaler stay the fresh ones `Trainer.__init__` already built, and
    `trainer.start_epoch` / `trainer.global_step` / `trainer.best_metric` are
    left at their fresh-run defaults. A fine-tune is a NEW run that borrows
    weights, not a continuation of the source run's schedule -- so the
    caller must also start a NEW W&B run for it (`init_wandb` is given None
    here, exactly as for a fresh run).

    Args:
        cfg: The full composed Hydra config.
        trainer: A freshly constructed `Trainer`, not yet trained.

    Returns:
        The `ResumeState` from `Trainer.resume_from`, if a resume happened;
        otherwise None, for both `init_from` and a fully fresh run.

    Raises:
        FileNotFoundError: See `resolve_init_from_checkpoint`.
        ValueError: See `resolve_init_from_checkpoint`.
    """
    # Validated up front, unconditionally -- see the docstring above for why
    # this must happen before select_resume_checkpoint, not only when it
    # returns None.
    init_path = resolve_init_from_checkpoint(cfg)

    resume_path = select_resume_checkpoint(cfg)
    if resume_path is not None:
        resume_state = trainer.resume_from(resume_path)
        logger.info(
            "RESUME: continuing training from epoch %d (checkpoint: %s)",
            resume_state.start_epoch,
            resume_path,
        )
        return resume_state

    if init_path is not None:
        load_checkpoint(
            init_path,
            trainer.model,
            map_location=str(trainer.device),
            restore_rng=False,
        )
        logger.info(
            "INIT_FROM: fine-tuning from %s (weights only -- fresh optimizer/scheduler/"
            "scaler, epoch 0, global_step 0, best_metric reset, new W&B run)",
            init_path,
        )
        return None

    logger.info("FRESH: starting a new training run from epoch 0")
    return None


def run_training(cfg: DictConfig) -> dict[str, float]:
    """Builds every training component and runs `Trainer.train()`.

    Order matters and mirrors the module-level spec:

    1. Logging, then `set_seed` -- seeding must happen before any dataset or
       model is built, since both draw random numbers (weight init, MONAI's
       random transforms) that a resumed run needs to reconstruct identically.
    2. Resolve the device, build the dataloaders, model, and loss.
    3. Construct the `Trainer` with `wandb_run=None`.
    4. `apply_resume_or_init` resumes from a checkpoint, or fine-tune-
       initializes from `training.checkpoint.init_from`, or leaves the
       Trainer fresh -- see that function's docstring for the precedence.
       This happens BEFORE initializing W&B: resuming into the same W&B run
       needs `id=resume_state.wandb_run_id`, which is only known after the
       checkpoint has been read (a fine-tune via `init_from`, like a fresh
       run, always starts a NEW W&B run). Initializing W&B first would
       create a fresh run and orphan the original one a resumed checkpoint
       belongs to.
    5. Run `trainer.train()`, log the final metrics, and finish the W&B run
       in a `finally` so an exception or a `max_hours` stop still closes it.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The final epoch's combined `train/*` and `val/*` metrics dict, as
        returned by `Trainer.train()`.
    """
    setup_logging(level="INFO")

    set_seed(cfg.seed)
    device = get_device(cfg)

    train_loader, val_loader = build_dataloaders(cfg, device)

    model = build_model(cfg)
    model = model.to(device)

    loss_fn = build_loss(cfg)

    # wandb_run=None deliberately: see the resume-then-wandb ordering note
    # in this function's docstring.
    trainer = Trainer(cfg, model, train_loader, val_loader, loss_fn, device, wandb_run=None)

    resume_state = apply_resume_or_init(cfg, trainer)

    # Only now, with resume_state known, is it safe to start (or resume) the
    # W&B run -- see the ordering note above.
    run = init_wandb(cfg, resume_state)
    trainer.wandb_run = run

    try:
        final_metrics = trainer.train()
        logger.info("Training finished. Final metrics: %s", final_metrics)
    finally:
        # finally, not just after trainer.train(), so a max_hours stop or an
        # exception mid-run still closes the W&B run cleanly instead of
        # leaving it "crashed" in the dashboard.
        if run is not None:
            run.finish()

    return final_metrics


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Train a model per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    run_training(cfg)


if __name__ == "__main__":
    main()
