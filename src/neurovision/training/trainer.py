"""The training loop: optimizer, scheduler, AMP, validation, checkpointing.

`Trainer` owns everything that needs to survive a Kaggle 12-hour kill and be
resumed exactly: the optimizer, the LR scheduler, and the AMP `GradScaler`
are all built here (not passed in) so that `resume_from` can restore state
into the *same* objects that `train()` then keeps using. Checkpoint I/O
itself lives in `neurovision.training.checkpoint` -- this module only decides
*when* to save and what counts as "best".
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from monai.inferers import sliding_window_inference
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.metrics.segmentation import MetricAggregator, binarize
from neurovision.training.checkpoint import ResumeState, load_checkpoint, save_checkpoint
from neurovision.utils.device import amp_enabled

logger = logging.getLogger(__name__)


def _build_optimizer(cfg: Any, model: nn.Module) -> torch.optim.Optimizer:
    """Builds the optimizer named by `cfg.training.optimizer.name`.

    Args:
        cfg: The full composed Hydra config.
        model: Model whose parameters the optimizer updates.

    Returns:
        A constructed `torch.optim.Optimizer`.

    Raises:
        ValueError: If `cfg.training.optimizer.name` is not `"adamw"` or
            `"adam"`.
    """
    opt_cfg = cfg.training.optimizer
    name = opt_cfg.name.lower()
    # betas arrives from YAML as a list; torch's Adam family requires a tuple.
    betas = tuple(opt_cfg.betas)

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, betas=betas
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, betas=betas
        )
    raise ValueError(f"Unknown optimizer '{name}'. Supported optimizers: 'adamw', 'adam'.")


def _build_scheduler(
    cfg: Any, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LambdaLR:
    """Builds a linear-warmup-then-cosine schedule, stepped once per epoch.

    Epoch-indexed (not step-indexed) deliberately: resume is epoch-granular
    (see `neurovision.training.checkpoint`), so an epoch-indexed schedule
    restores to exactly the right LR just by calling `.step()` up to
    `start_epoch` -- a step-indexed schedule would need the exact batch
    count of every prior epoch, which is not recoverable across a resume.

    Args:
        cfg: The full composed Hydra config.
        optimizer: The optimizer whose LR is scheduled.

    Returns:
        A `torch.optim.lr_scheduler.LambdaLR` implementing the schedule.

    Raises:
        ValueError: If `cfg.training.scheduler.name` is not `"cosine"`.
    """
    sched_cfg = cfg.training.scheduler
    if sched_cfg.name != "cosine":
        raise ValueError(
            f"Unknown scheduler '{sched_cfg.name}'. Only 'cosine' (linear warmup + cosine "
            "decay) is supported."
        )

    warmup_epochs = sched_cfg.warmup_epochs
    total_epochs = cfg.training.epochs
    min_ratio = sched_cfg.min_lr / cfg.training.optimizer.lr

    def lr_lambda(epoch: int) -> float:
        # epoch < warmup_epochs is False for every epoch >= 0 when
        # warmup_epochs == 0, so the (epoch + 1) / warmup_epochs division
        # below is never reached in that case -- no explicit zero-guard
        # needed, but do not reorder this without re-checking that.
        if epoch < warmup_epochs:
            # +1 so the very first epoch does not train at LR exactly 0.
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Runs the training loop: forward/backward, validation, checkpointing.

    All config is read from `cfg.training` (see `configs/training/default.yaml`
    for the exact keys). Device is resolved by the caller via
    `neurovision.utils.device.get_device` and passed in -- this class never
    hardcodes `"cuda"` and runs unmodified on CPU.
    """

    def __init__(
        self,
        cfg: Any,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_fn: nn.Module,
        device: torch.device,
        wandb_run: Any | None = None,
    ) -> None:
        """Builds the optimizer, scheduler and AMP scaler for a fresh run.

        Args:
            cfg: The full composed Hydra config.
            model: The model to train. Moved to `device` here.
            train_loader: Yields dicts with `"image"`/`"label"` keys, patch-
                sized batches.
            val_loader: Yields whole-volume, batch-size-1 dicts with the same
                keys, or None to skip validation entirely.
            loss_fn: The training loss, called as `loss_fn(logits, labels)`.
            device: The resolved torch device. Never chosen by this class.
            wandb_run: An active W&B run to log into, or None. When None, no
                W&B call is ever made -- the `wandb` package is not imported
                here at all, so this class has no hard dependency on it.
        """
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.wandb_run = wandb_run

        # Built here, not passed in, so resume_from can restore state
        # directly into these same objects rather than reconstructing them.
        self.optimizer = _build_optimizer(cfg, self.model)
        self.scheduler = _build_scheduler(cfg, self.optimizer)

        # amp_enabled(device) is the final gate: AMP is requested by config
        # but only ever actually runs on CUDA. A GradScaler built with
        # enabled=False is a documented no-op passthrough, so there is no
        # separate "AMP off" code path anywhere below.
        self.amp_is_enabled = bool(cfg.training.amp) and amp_enabled(device)
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_is_enabled)

        self.start_epoch = 0
        self.global_step = 0

        # The sentinel must match best_metric_mode: -inf under "min" would
        # mean no later value ever compares as better, and best.pt would
        # silently stop updating for the rest of the run.
        mode = cfg.training.checkpoint.mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")

    def _current_lr(self) -> float:
        """Reads the optimizer's current learning rate (single param group)."""
        return self.optimizer.param_groups[0]["lr"]

    def resume_from(self, path: str | Path) -> ResumeState:
        """Restores model/optimizer/scheduler/scaler state from a checkpoint.

        Args:
            path: Path to a checkpoint written by `save_checkpoint`.

        Returns:
            The `ResumeState`, so the caller can read `wandb_run_id` to
            resume logging into the same W&B run.
        """
        state = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=str(self.device),
        )
        self.start_epoch = state.start_epoch
        self.global_step = state.global_step
        self.best_metric = state.best_metric
        return state

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """Runs one training epoch: forward, backward, optimizer steps.

        Implements gradient accumulation and gradient clipping in the order
        that actually matters: `scaler.unscale_` must run before
        `clip_grad_norm_`, because gradients are still multiplied by the
        scaler's scale factor until `unscale_` is called -- clipping before
        that clips a meaningless number.

        Args:
            epoch: Current epoch index (0-based), used only for the tqdm
                description and for resetting the CUDA peak-memory counter.

        Returns:
            A dict with `"train/loss_epoch"`, the mean of the (un-divided,
            un-accumulated) per-batch loss over the epoch.
        """
        cfg = self.cfg
        grad_accum_steps = cfg.training.grad_accum_steps
        grad_clip_norm = cfg.training.grad_clip_norm
        log_interval = cfg.training.log_interval

        # Reset here (epoch start) rather than in train(), so train_one_epoch
        # stays self-contained and callable on its own in tests.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.model.train()
        running_loss = 0.0
        n_batches = 0
        # Every pre-clip gradient norm this epoch. Kept so the epoch summary
        # below can reach the PYTHON LOG, not only W&B: whether grad_clip_norm
        # is set correctly has to be decided before a multi-run comparison
        # starts (clipping rescales the whole gradient, so two runs clipping at
        # different rates train at different effective segmentation learning
        # rates), and on Kaggle the log is retrievable from the API for free
        # while the W&B history means downloading gigabytes of checkpoints.
        grad_norms: list[float] = []

        progress = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_is_enabled):
                outputs = self.model(images)
                raw_loss = self.loss_fn(outputs, labels)

            # Divide by grad_accum_steps so N accumulated small-batch
            # gradients sum to the same magnitude as one gradient computed
            # on the full effective batch (batch_size * grad_accum_steps).
            loss = raw_loss / grad_accum_steps
            self.scaler.scale(loss).backward()

            # Logged/returned loss is the un-divided per-batch value -- the
            # division above is purely a backward-pass bookkeeping detail,
            # not something a human reading a loss curve should see.
            running_loss += raw_loss.item()
            n_batches += 1

            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_norm: float | None = None
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    # clip_grad_norm_ returns the total norm BEFORE clipping, so
                    # logging it is free. Worth watching under a multi-task loss:
                    # extra terms inflate the norm, and once it routinely exceeds
                    # grad_clip_norm the WHOLE gradient is scaled down -- which
                    # changes the effective LR of the segmentation term too, and
                    # silently confounds "did the auxiliary head help?" with "did
                    # the segmentation LR change?". If this runs consistently
                    # above the clip threshold where the seg-only baseline did
                    # not, raise grad_clip_norm rather than rescaling the loss
                    # weights.
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    )
                    grad_norms.append(grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # global_step counts OPTIMIZER steps, not batches. This is
                # the axis log_interval and the W&B x-axis are keyed on, so
                # accumulating e.g. 4 batches into 1 step must not make the
                # loss curve look 4x sparser than it should, or make the
                # step-vs-LR relationship silently wrong.
                self.global_step += 1

                if self.wandb_run is not None and self.global_step % log_interval == 0:
                    payload = {"train/loss": raw_loss.item(), "train/lr": self._current_lr()}
                    if grad_norm is not None:
                        payload["train/grad_norm"] = grad_norm
                    # A multi-task loss exposes its UNWEIGHTED per-term values as
                    # detached floats. Without them the total is uninterpretable:
                    # the terms converge to different magnitudes (region Dice+BCE
                    # settles ~0.25, the thin-shell boundary Dice stays ~0.7), so
                    # a flat total can hide the boundary term quietly taking over
                    # late in training. Any loss lacking this attribute -- every
                    # baseline -- logs exactly what it did before.
                    components = getattr(self.loss_fn, "last_components", None)
                    if components is not None:
                        payload.update({f"train/loss_{k}": v for k, v in components.items()})
                    self.wandb_run.log(payload, step=self.global_step)

            progress.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{self._current_lr():.2e}")

        # Flush a partial accumulation window at the end of the epoch.
        #
        # When the batch count is not a multiple of grad_accum_steps (e.g.
        # 1750 batches with grad_accum_steps=4), the trailing batches never
        # trigger the step above. Without this flush their gradients are
        # neither applied NOR zeroed, so they survive into the next epoch and
        # get summed into its first accumulation window -- mixing stale
        # gradients from the previous epoch's final batches, computed under a
        # different LR, into the next step. That is silent: nothing errors,
        # the loss curve just gets slightly wrong gradients at every epoch
        # boundary.
        if n_batches % grad_accum_steps != 0:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        avg_loss = running_loss / n_batches if n_batches > 0 else float("nan")
        metrics = {"train/loss_epoch": avg_loss}

        # Gradient-norm summary, to the python log so it survives without W&B.
        # `clipped_frac` is the number that actually decides grad_clip_norm: a
        # run clipping on most steps is training at an effective LR the config
        # does not describe, and two runs in one comparison clipping at
        # different rates are not a controlled comparison at all.
        if grad_norms:
            ordered = sorted(grad_norms)
            n = len(ordered)
            p50 = ordered[n // 2]
            p90 = ordered[min(n - 1, int(0.9 * n))]
            clipped = sum(1 for g in grad_norms if g > grad_clip_norm) / n
            logger.info(
                "epoch %d grad_norm: median %.3f | p90 %.3f | max %.3f | "
                "clipped %.1f%% of %d steps (grad_clip_norm=%.2f)",
                epoch,
                p50,
                p90,
                ordered[-1],
                100 * clipped,
                n,
                grad_clip_norm,
            )
            metrics.update(
                {
                    "train/grad_norm_median": p50,
                    "train/grad_norm_p90": p90,
                    "train/grad_norm_max": ordered[-1],
                    "train/grad_norm_clipped_frac": clipped,
                }
            )
        return metrics

    def validate(self, epoch: int) -> dict[str, float]:
        """Runs sliding-window inference and region metrics over `val_loader`.

        Validation loaders yield whole BraTS volumes at batch size 1 -- a
        full volume does not fit the 96^3 training patch size, which is
        exactly why sliding-window inference (not a single forward pass) is
        needed here at all.

        Args:
            epoch: Current epoch index. Unused in the computation itself;
                kept for a consistent signature and for future per-epoch
                logging (e.g. saving qualitative predictions).

        Returns:
            A flat dict of `val/<metric>` keys built from the aggregator's
            `summary()` `mean` column (e.g. `val/dice_ET`, `val/dice_mean`,
            `val/hd95_mean`, ...). Always includes `val/dice_mean` as long as
            at least one case was validated.
        """
        del epoch  # not used directly; see docstring
        sw_cfg = self.cfg.training.sliding_window

        self.model.eval()
        aggregator = MetricAggregator()
        try:
            with torch.no_grad():
                for case_idx, batch in enumerate(self.val_loader):
                    images = batch["image"].to(self.device, non_blocking=True)
                    labels = batch["label"].to(self.device, non_blocking=True)

                    with torch.amp.autocast(
                        device_type=self.device.type, enabled=self.amp_is_enabled
                    ):
                        logits = sliding_window_inference(
                            inputs=images,
                            roi_size=list(sw_cfg.roi_size),
                            sw_batch_size=sw_cfg.sw_batch_size,
                            predictor=self.model,
                            overlap=sw_cfg.overlap,
                        )

                    pred = binarize(logits)
                    case_id = batch.get("case_id", [str(case_idx)])[0]
                    # .cpu() on BOTH, and it is not optional. hd95() calls
                    # MONAI's compute_hausdorff_distance, which computes its
                    # distance transform via CuPy whenever the tensors are on
                    # CUDA. On Kaggle that CuPy JIT fails outright:
                    #   CompileException: Thrust requires at least C++17
                    # (measured 2026-08-01, T4 image). The CPU path uses scipy
                    # and has no such dependency. Moving to CPU also keeps
                    # HD95's intermediate buffers over a full ~240^3 volume off
                    # the 16 GB VRAM budget, which the model and the
                    # sliding-window output already share. scripts/evaluate.py
                    # does the same thing for the same reason.
                    aggregator.add_case(str(case_id), pred[0].cpu(), labels[0].cpu())
        finally:
            # Restore train mode even if inference raises, so a caller that
            # catches the exception does not end up with a model stuck in
            # eval mode for the next training epoch.
            self.model.train()

        summary = aggregator.summary()
        if summary.empty:
            logger.warning("validate() ran with an empty val_loader; no metrics computed.")
            return {"val/dice_mean": float("nan")}

        return {f"val/{name}": float(row["mean"]) for name, row in summary.iterrows()}

    def train(self) -> dict[str, float]:
        """Runs the full training loop from `self.start_epoch` to the configured end.

        Loops `for epoch in range(self.start_epoch, cfg.training.epochs)` so
        a resumed run picks up exactly where it left off. Before starting
        each new epoch, checks a predicted (not reactive) wall-clock budget:
        if the elapsed time plus the running mean epoch duration would
        exceed `cfg.training.max_hours`, the run stops WITHOUT starting that
        epoch, because a 12-hour Kaggle session gets killed mid-epoch with no
        chance to save -- stopping only after exceeding the budget would be
        too late.

        Returns:
            The last epoch's combined `train/*` and `val/*` metrics dict
            (empty if no epoch ran, e.g. `max_hours` was already exceeded
            before the first epoch).
        """
        cfg = self.cfg
        total_epochs = cfg.training.epochs
        max_hours = cfg.training.max_hours

        start_time = time.time()
        epoch_durations: list[float] = []
        final_metrics: dict[str, float] = {}

        for epoch in range(self.start_epoch, total_epochs):
            if max_hours is not None and max_hours > 0:
                elapsed = time.time() - start_time
                mean_epoch_seconds = (
                    sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0
                )
                if elapsed + mean_epoch_seconds > max_hours * 3600:
                    logger.warning(
                        "Stopping before epoch %d: elapsed %.4fh plus a predicted %.4fh for "
                        "the next epoch would exceed the max_hours=%.4fh budget. Run stopped "
                        "on the time budget, not by completing all %d configured epochs -- the "
                        "previous epoch's checkpoint is the final saved state.",
                        epoch,
                        elapsed / 3600,
                        mean_epoch_seconds / 3600,
                        max_hours,
                        total_epochs,
                    )
                    break

            epoch_start = time.time()
            train_metrics = self.train_one_epoch(epoch)
            # Stepped once per epoch (not per batch): resume is
            # epoch-granular, so an epoch-indexed schedule restores exactly
            # by stepping up to start_epoch, which is what LambdaLR gives us.
            self.scheduler.step()

            run_validation = (
                self.val_loader is not None and (epoch + 1) % cfg.training.val_interval == 0
            )
            val_metrics: dict[str, float] = {}
            if run_validation:
                val_metrics = self.validate(epoch)

            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)

            # --- best-checkpoint bookkeeping ---
            monitor_key = cfg.training.checkpoint.monitor
            checkpoint_mode = cfg.training.checkpoint.mode
            is_best = False
            if monitor_key in val_metrics:
                new_value = val_metrics[monitor_key]
                is_best = (
                    new_value > self.best_metric
                    if checkpoint_mode == "max"
                    else new_value < self.best_metric
                )
                if is_best:
                    self.best_metric = new_value
            # If this epoch had no validation, or the monitored key is
            # absent, best_metric is left untouched and is_best stays False
            # -- there is nothing this epoch to compare against "best" on.

            periodic = (epoch + 1) % cfg.training.checkpoint.save_every_n_epochs == 0

            save_checkpoint(
                out_dir=cfg.training.checkpoint.dir,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                global_step=self.global_step,
                scheduler=self.scheduler,
                scaler=self.scaler,
                best_metric=self.best_metric,
                best_metric_name=monitor_key,
                best_metric_mode=checkpoint_mode,
                wandb_run_id=getattr(self.wandb_run, "id", None),
                cfg=cfg,
                is_best=is_best,
                periodic=periodic,
                keep_last_n=cfg.training.checkpoint.keep_last_n,
            )

            if self.wandb_run is not None:
                log_payload: dict[str, float] = {
                    **train_metrics,
                    "train/epoch_time_seconds": epoch_time,
                    **val_metrics,
                }
                # Guarded so this never raises on CPU/MPS runs, where CUDA
                # peak-memory tracking simply does not apply.
                if torch.cuda.is_available():
                    log_payload["train/peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
                self.wandb_run.log(log_payload, step=self.global_step)

            final_metrics = {**train_metrics, **val_metrics}

        return final_metrics
