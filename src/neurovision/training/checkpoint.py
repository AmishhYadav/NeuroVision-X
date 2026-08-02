"""Checkpoint save/load for full training resume on Kaggle.

Kaggle sessions are killed by a 12-hour wall-clock limit with no warning, so a
run's survival depends entirely on being able to resume from the last saved
state: model weights, optimizer/scheduler/scaler state, the exact RNG state
of every generator in use, and the config that produced the run. This module
is the only place that reads or writes a training checkpoint.

Two safety properties matter more than convenience here:

1. **Writes are atomic.** A checkpoint is written to a temp file in the same
   directory as the destination, then swapped into place with `os.replace`,
   which is atomic on both macOS and Linux as long as source and destination
   are on the same filesystem (this is why the temp file is *not* placed in
   a different directory or in `/tmp`). A SIGKILL mid-write can therefore
   never leave a truncated `last.pt` that silently replaces a good one.
2. **Loading never executes arbitrary code.** Every checkpoint is saved so
   that it loads under `torch.load(..., weights_only=True)`, which restricts
   deserialization to safe primitive types. See the module-level note below
   for why this rules out saving an OmegaConf `DictConfig` or a raw NumPy RNG
   state tuple directly.
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"

# Bumped whenever the payload's key structure changes, so a future loader can
# detect and migrate an old checkpoint instead of failing on a missing key
# with no explanation.
_FORMAT_VERSION = 1

# Pattern used for both writing and globbing periodic snapshots. The epoch is
# zero-padded to 4 digits so that a *lexicographic* sort also happens to be
# numeric up to epoch 9999 -- but we still parse and sort numerically below,
# rather than relying on that coincidence, since nothing stops a run from
# training past epoch 9999.
_PERIODIC_PATTERN = "epoch_{epoch:04d}.pt"
_PERIODIC_GLOB = "epoch_*.pt"


@dataclass
class ResumeState:
    """Everything a training script needs to pick a run back up.

    Attributes:
        start_epoch: The epoch to resume training AT (saved epoch + 1). The
            saved epoch already finished, so re-running it would train on it
            twice and desynchronize any epoch-indexed LR schedule.
        global_step: The optimizer step count at save time.
        best_metric: The best validation metric seen so far.
        best_metric_name: Name of the metric `best_metric` refers to, or
            None if it was never recorded.
        best_metric_mode: `"max"` if a higher value of that metric is better
            (Dice, IoU), `"min"` if lower is better (loss, HD95, ECE). The
            trainer needs this to decide whether a new value beats
            `best_metric`, and storing it makes `best.pt` self-describing
            rather than dependent on the trainer remembering the convention.
        wandb_run_id: The W&B run ID to resume logging into, or None.
        config: The training config, rebuilt as a `DictConfig`, or None if
            none was saved.
    """

    start_epoch: int
    global_step: int
    best_metric: float
    best_metric_name: str | None
    best_metric_mode: str
    wandb_run_id: str | None
    config: Any | None


def _numpy_rng_state_to_safe_dict(state: tuple) -> dict[str, Any]:
    """Converts `np.random.get_state()` into a `weights_only=True`-safe dict.

    `np.random.get_state()` returns a 5-tuple `(str, ndarray[uint32], int,
    int, float)`. A raw `ndarray` inside a pickled tuple is exactly the kind
    of object `torch.load(weights_only=True)` refuses to construct, so the
    array is converted to a `torch.Tensor` (an allowed type) and the tuple is
    flattened into named dict entries instead of being pickled as a tuple of
    mixed types.

    Args:
        state: The 5-tuple returned by `np.random.get_state()`.

    Returns:
        A dict of primitives/tensors safe to include in a checkpoint payload.
    """
    bit_generator, state_array, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": bit_generator,
        "state": torch.from_numpy(state_array.copy()),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _safe_dict_to_numpy_rng_state(saved: dict[str, Any]) -> tuple:
    """Inverse of `_numpy_rng_state_to_safe_dict`.

    Args:
        saved: Dict previously produced by `_numpy_rng_state_to_safe_dict`.

    Returns:
        A 5-tuple usable with `np.random.set_state`.
    """
    # .cpu() before .numpy(), and it is load-bearing. The numpy RNG state is
    # stored as a torch tensor so the payload stays weights_only=True-loadable,
    # but load_checkpoint is called with map_location=str(device) -- so on a GPU
    # run EVERY tensor in the payload, this one included, is materialized on
    # CUDA. `.numpy()` then raises:
    #   TypeError: can't convert cuda:0 device type tensor to numpy.
    # Measured on Kaggle 2026-08-02, resuming a real run at epoch 130. Invisible
    # on the Mac: with map_location="cpu" the tensor is already on CPU, so every
    # CPU test passes either way. The tensor is 624 uint32s -- the copy is free.
    return (
        saved["bit_generator"],
        saved["state"].cpu().numpy().astype("uint32"),
        saved["pos"],
        saved["has_gauss"],
        saved["cached_gaussian"],
    )


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Writes `payload` to `destination` atomically.

    Saves to a temp file in `destination`'s own directory (so the later
    `os.replace` is guaranteed to be on the same filesystem, which is what
    makes it atomic), then swaps it into place. On any failure the temp file
    is removed and the exception re-raised, so a bad write never destroys a
    previously good checkpoint and never leaves a partial file behind.

    Args:
        payload: The checkpoint dict to serialize with `torch.save`.
        destination: Final path the checkpoint should end up at.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        # Close the low-level fd immediately; torch.save opens the path
        # itself and writes through its own file handle.
        os.close(fd)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _build_payload(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    scheduler: Any | None,
    scaler: Any | None,
    best_metric: float,
    best_metric_name: str | None,
    best_metric_mode: str,
    wandb_run_id: str | None,
    cfg: Any | None,
) -> dict[str, Any]:
    """Assembles the full checkpoint payload dict, built exactly once per save.

    Args:
        model: The model whose `state_dict` is saved.
        optimizer: The optimizer whose `state_dict` is saved.
        epoch: Epoch that just completed.
        global_step: Optimizer step count at save time.
        scheduler: LR scheduler, or None.
        scaler: AMP grad scaler, or None. Legitimately None on CPU/MPS runs,
            since AMP is CUDA-only in this project (see `utils/device.py`).
        best_metric: Best validation metric seen so far.
        best_metric_name: Name of that metric, or None.
        best_metric_mode: `"max"` or `"min"` for that metric.
        wandb_run_id: W&B run ID to resume into, or None.
        cfg: Full Hydra config, or None.

    Returns:
        The payload dict, containing only `weights_only=True`-safe types.
    """
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_metric_mode,
        "wandb_run_id": wandb_run_id,
        # resolve=True bakes in interpolations (e.g. `${data.num_classes}`)
        # so the stored config records what actually ran, not a template
        # that depends on other config files being present later.
        "config_yaml": OmegaConf.to_yaml(cfg, resolve=True) if cfg is not None else None,
        "random_state": random.getstate(),
        "numpy_random_state": _numpy_rng_state_to_safe_dict(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    return payload


def save_checkpoint(
    out_dir: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    best_metric: float = float("-inf"),
    best_metric_name: str | None = None,
    best_metric_mode: str = "max",
    wandb_run_id: str | None = None,
    cfg: Any | None = None,
    is_best: bool = False,
    periodic: bool = False,
    keep_last_n: int = 3,
) -> Path:
    """Saves a full training checkpoint, atomically, to `out_dir`.

    Always writes `last.pt`. Also writes `best.pt` when `is_best=True`, and
    an `epoch_{epoch:04d}.pt` snapshot when `periodic=True` (pruned to the
    newest `keep_last_n` afterwards). The same payload dict is built once and
    reused for every destination, so the (possibly large) model/optimizer
    state is only serialized once per call.

    Args:
        out_dir: Directory to write checkpoint files into. Created if it
            does not exist.
        model: Model to checkpoint.
        optimizer: Optimizer to checkpoint.
        epoch: The epoch that just completed (0-indexed).
        global_step: Optimizer step count at save time.
        scheduler: LR scheduler to checkpoint, or None.
        scaler: AMP `GradScaler` to checkpoint, or None. Pass None on
            CPU/MPS runs -- AMP is CUDA-only in this project.
        best_metric: Best validation metric seen so far in the run. The
            default `-inf` pairs with the default `best_metric_mode="max"`;
            pass `+inf` when tracking a metric that is minimized.
        best_metric_name: Name of that metric (e.g. `"dice_wt"`), or None.
        best_metric_mode: `"max"` if higher is better (Dice, IoU), `"min"` if
            lower is better (loss, HD95, ECE). Stored in the checkpoint so
            `best.pt` records which direction it was selected on, instead of
            relying on the trainer to remember.
        wandb_run_id: The active W&B run ID, so resume logs into the same
            run instead of starting a new one.

    Raises:
        ValueError: If `best_metric_mode` is not `"max"` or `"min"`.
        cfg: The full composed Hydra config for this run, or None.
        is_best: If True, also write `best.pt` (a full checkpoint, not just
            weights, so a run can be resumed from `best.pt` alone).
        periodic: If True, also write a numbered `epoch_NNNN.pt` snapshot.
        keep_last_n: Number of periodic snapshots to retain (newest by
            epoch number). `last.pt` and `best.pt` are never pruned, no
            matter what. `keep_last_n <= 0` disables pruning entirely (keep
            every periodic snapshot ever written) -- use with care given the
            ~150-750 MB size of a full checkpoint against Kaggle's disk quota.

    Returns:
        Path to the written `last.pt`.
    """
    if best_metric_mode not in ("max", "min"):
        raise ValueError(
            f"best_metric_mode must be 'max' or 'min', got {best_metric_mode!r}. "
            "Dice/IoU are 'max'; loss, HD95 and ECE are 'min'."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _build_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        best_metric_mode=best_metric_mode,
        wandb_run_id=wandb_run_id,
        cfg=cfg,
    )

    last_path = out_dir / LAST_CHECKPOINT_NAME
    _atomic_torch_save(payload, last_path)

    wrote_best = False
    if is_best:
        best_path = out_dir / BEST_CHECKPOINT_NAME
        _atomic_torch_save(payload, best_path)
        wrote_best = True

    wrote_periodic = False
    if periodic:
        periodic_path = out_dir / _PERIODIC_PATTERN.format(epoch=epoch)
        _atomic_torch_save(payload, periodic_path)
        wrote_periodic = True
        _prune_periodic_snapshots(out_dir, keep_last_n)

    logger.info(
        "Saved checkpoint to %s (epoch=%d, best=%s, periodic=%s)",
        last_path,
        epoch,
        wrote_best,
        wrote_periodic,
    )
    return last_path


def _parse_periodic_epoch(path: Path) -> int | None:
    """Extracts the epoch number from an `epoch_NNNN.pt` filename.

    Args:
        path: Candidate periodic-snapshot path.

    Returns:
        The parsed epoch number, or None if `path` does not match the
        expected `epoch_<digits>.pt` pattern.
    """
    stem = path.stem  # "epoch_0012"
    prefix = "epoch_"
    if not stem.startswith(prefix):
        return None
    digits = stem[len(prefix) :]
    if not digits.isdigit():
        return None
    return int(digits)


def _prune_periodic_snapshots(out_dir: Path, keep_last_n: int) -> None:
    """Deletes old `epoch_*.pt` snapshots, keeping the newest `keep_last_n`.

    Only ever globs the `epoch_*.pt` pattern, so `last.pt`/`best.pt` can
    never be touched here. Sorts by the parsed epoch number (not mtime, not
    filename string order) so results are correct even if snapshots were
    copied/touched out of order.

    Args:
        out_dir: Directory containing the periodic snapshots.
        keep_last_n: Number to keep. `<= 0` means keep all (no pruning).
    """
    if keep_last_n <= 0:
        return

    candidates = []
    for path in out_dir.glob(_PERIODIC_GLOB):
        parsed_epoch = _parse_periodic_epoch(path)
        if parsed_epoch is not None:
            candidates.append((parsed_epoch, path))

    candidates.sort(key=lambda item: item[0])  # oldest first, numeric

    n_to_delete = len(candidates) - keep_last_n
    if n_to_delete <= 0:
        return

    for _, path in candidates[:n_to_delete]:
        path.unlink()
        logger.info("Pruned periodic checkpoint %s (keep_last_n=%d)", path, keep_last_n)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """Loads a checkpoint written by `save_checkpoint` and restores state in place.

    Every restoration except the model is defensive: a missing or None key
    logs a WARNING and is skipped rather than raising, so a checkpoint saved
    by an older version of this module (missing a newer key) can still
    resume. The model is the one exception -- a checkpoint with no model
    weights is useless, so that case raises `KeyError` instead of silently
    producing an untrained model that looks like a successful resume.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into, in place.
        optimizer: Optimizer to restore state into, in place. Skipped if
            None or if the checkpoint has no optimizer state.
        scheduler: LR scheduler to restore state into, in place, or None.
        scaler: AMP `GradScaler` to restore state into, in place, or None.
        map_location: Passed to `torch.load`. Defaults to `"cpu"` so a
            checkpoint saved on a CUDA box loads on a CPU-only machine (and
            vice versa); `model.load_state_dict` then copies the restored
            tensors onto whatever device `model` already lives on.
        strict: Forwarded to `model.load_state_dict`. Defaults to True: a
            silently partial weight load produces a model that runs without
            error but scores garbage, which is worse than a loud failure.
        restore_rng: If False, skip restoring RNG state entirely (useful
            when resuming deliberately with a different seed).

    Returns:
        A `ResumeState` describing where training should pick back up.

    Raises:
        FileNotFoundError: If `path` does not exist.
        KeyError: If the checkpoint has no model state dict.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {path.resolve()}. Cannot resume from a missing file."
        )

    # weights_only=True is the whole point: it refuses to deserialize
    # anything but safe primitive types, so loading a checkpoint can never
    # execute arbitrary code. Every value in the payload was chosen at save
    # time to be compatible with this.
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    # The model is the one required key: everything else is recoverable
    # information about a run, but a checkpoint with no weights cannot be
    # resumed from at all, so it is treated as corrupt rather than partial.
    if "model_state_dict" not in checkpoint or checkpoint["model_state_dict"] is None:
        raise KeyError(
            f"Checkpoint at {path} has no 'model_state_dict'. It cannot be used to resume "
            "training or run inference."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        else:
            logger.warning("Checkpoint %s has no optimizer state; optimizer not restored.", path)

    if scheduler is not None:
        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)
        else:
            logger.warning("Checkpoint %s has no scheduler state; scheduler not restored.", path)

    if scaler is not None:
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        else:
            logger.warning("Checkpoint %s has no scaler state; scaler not restored.", path)

    if restore_rng:
        _restore_rng_state(checkpoint, path)

    epoch = checkpoint.get("epoch")
    if epoch is None:
        logger.warning("Checkpoint %s has no 'epoch'; assuming epoch -1 (fresh start).", path)
        epoch = -1

    global_step = checkpoint.get("global_step")
    if global_step is None:
        logger.warning("Checkpoint %s has no 'global_step'; defaulting to 0.", path)
        global_step = 0

    # Read the mode first: it decides which sentinel a missing best_metric
    # falls back to. Defaulting to -inf regardless of mode would be a silent
    # failure for a minimized metric -- no later value would ever compare as
    # better, so best.pt would never be written again for the rest of the run.
    best_metric_mode = checkpoint.get("best_metric_mode")
    if best_metric_mode not in ("max", "min"):
        logger.warning(
            "Checkpoint %s has no valid 'best_metric_mode' (got %r); assuming 'max'.",
            path,
            best_metric_mode,
        )
        best_metric_mode = "max"

    best_metric = checkpoint.get("best_metric")
    if best_metric is None:
        best_metric = float("-inf") if best_metric_mode == "max" else float("inf")
        logger.warning(
            "Checkpoint %s has no 'best_metric'; defaulting to %s for mode '%s'.",
            path,
            best_metric,
            best_metric_mode,
        )

    best_metric_name = checkpoint.get("best_metric_name")
    wandb_run_id = checkpoint.get("wandb_run_id")

    config_yaml = checkpoint.get("config_yaml")
    config = OmegaConf.create(config_yaml) if config_yaml is not None else None
    if config is None:
        logger.warning("Checkpoint %s has no stored config.", path)

    return ResumeState(
        # The saved epoch already completed, so resuming means starting the
        # NEXT one -- re-running the saved epoch would both double-count
        # training data and desynchronize an epoch-indexed LR schedule.
        start_epoch=epoch + 1,
        global_step=global_step,
        best_metric=best_metric,
        best_metric_name=best_metric_name,
        best_metric_mode=best_metric_mode,
        wandb_run_id=wandb_run_id,
        config=config,
    )


def _restore_rng_state(checkpoint: dict[str, Any], path: Path) -> None:
    """Restores Python/NumPy/torch (and CUDA, if available) RNG state.

    Each generator is guarded independently: a missing key logs a WARNING
    and is skipped, rather than aborting the whole restore.

    Args:
        checkpoint: The loaded checkpoint dict.
        path: Source path, used only for log messages.
    """
    random_state = checkpoint.get("random_state")
    if random_state is not None:
        random.setstate(random_state)
    else:
        logger.warning("Checkpoint %s has no 'random_state'; Python RNG not restored.", path)

    numpy_state = checkpoint.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(_safe_dict_to_numpy_rng_state(numpy_state))
    else:
        logger.warning("Checkpoint %s has no 'numpy_random_state'; NumPy RNG not restored.", path)

    torch_state = checkpoint.get("torch_rng_state")
    if torch_state is not None:
        # .cpu() for the same reason as the numpy state above: map_location
        # puts it on CUDA, and torch.set_rng_state requires a CPU ByteTensor.
        # Without this, resuming on GPU raises immediately after the numpy
        # state is fixed -- two separate faults on consecutive lines.
        torch.set_rng_state(torch_state.cpu())
    else:
        logger.warning("Checkpoint %s has no 'torch_rng_state'; torch RNG not restored.", path)

    cuda_state = checkpoint.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        # set_rng_state_all likewise wants CPU ByteTensors, one per device.
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
    elif cuda_state is not None:
        logger.debug(
            "Checkpoint %s has CUDA RNG state but no CUDA device is available; skipping.", path
        )
    else:
        logger.debug("Checkpoint %s has no CUDA RNG state to restore.", path)


def find_resume_checkpoint(out_dir: str | Path) -> Path | None:
    """Finds the checkpoint a training script should resume from, if any.

    Args:
        out_dir: Directory to look in. Does not need to exist.

    Returns:
        `out_dir / "last.pt"` if it exists, else None.
    """
    last_path = Path(out_dir) / LAST_CHECKPOINT_NAME
    return last_path if last_path.is_file() else None
