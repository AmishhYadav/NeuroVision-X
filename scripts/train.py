"""Hydra entry point for training.

Wires together data, model, loss, and `neurovision.training.trainer.Trainer`,
then hands off to the trainer's own loop. This runs unmodified on the Mac's
CPU (for a smoke test) or on a Kaggle GPU (for a real run) -- device is
resolved once, from config, via `neurovision.utils.device.get_device`.

Example usage:

    python scripts/train.py data.root_dir=/path/to/preprocessed wandb.mode=disabled

The wiring is split into small functions (`build_dataloaders`, `init_wandb`,
`select_resume_checkpoint`, `run_training`) rather than one long `main`, so
each piece can be unit tested without going through Hydra -- see
tests/test_train_script.py.
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
from neurovision.training.checkpoint import ResumeState, find_resume_checkpoint
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


def run_training(cfg: DictConfig) -> dict[str, float]:
    """Builds every training component and runs `Trainer.train()`.

    Order matters and mirrors the module-level spec:

    1. Logging, then `set_seed` -- seeding must happen before any dataset or
       model is built, since both draw random numbers (weight init, MONAI's
       random transforms) that a resumed run needs to reconstruct identically.
    2. Resolve the device, build the dataloaders, model, and loss.
    3. Construct the `Trainer` with `wandb_run=None`.
    4. Resume from a checkpoint (if any) BEFORE initializing W&B. This order
       is inverted from what you might expect: resuming into the same W&B
       run needs `id=resume_state.wandb_run_id`, which is only known after
       the checkpoint has been read. Initializing W&B first would create a
       fresh run and orphan the original one the checkpoint belongs to.
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

    resume_path = select_resume_checkpoint(cfg)

    if resume_path is not None:
        resume_state = trainer.resume_from(resume_path)
        logger.info(
            "RESUME: continuing training from epoch %d (checkpoint: %s)",
            resume_state.start_epoch,
            resume_path,
        )
    else:
        resume_state = None
        logger.info("FRESH: starting a new training run from epoch 0")

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
