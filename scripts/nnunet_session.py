"""Runs one ~11 h Kaggle-T4 slice of nnU-Net v2's DEFAULT training recipe.

Implements Gate A Amendment 1 (`docs/research/preregistration_strong_baseline.md`):
`nnUNetTrainer`, `nnUNetPlans`, `3d_fullres`, fold `all`, 1000 epochs, unmodified
optimisation -- chained across several Kaggle sessions, each stopping cleanly
BETWEEN epochs when its wall-clock budget runs out, and resuming the next session
through nnU-Net's OWN resume path (`maybe_load_checkpoint(continue_training=True)`).
Nothing about the optimisation is touched: this script only decides *when* to
stop calling `nnUNetTrainer`'s own per-epoch hooks, exactly the way its own
`run_training()` calls them (nnunetv2 v2.8.1,
`training/nnUNetTrainer/nnUNetTrainer.py`, lines 1417-1438).

Two operational differences from a plain `nnUNetv2_train --c` session, neither of
which touches optimisation (Amendment 1, point 5):

    - `checkpoint_latest.pth` is written every epoch instead of every 50
      (`trainer.save_every = 1`), so a session that is killed mid-epoch loses at
      most one epoch of progress, not up to 49;
    - each session stops BETWEEN epochs once `should_stop` predicts the next
      epoch would blow the wall-clock budget, rather than running until an
      external kill.

Why `nnunetv2` is never imported at module scope: it lives only in
`.venv-analysis` (see `requirements-analysis.txt`), so importing it in a process
that only has the main `.venv` would crash this module before a single test
could run. (Its own `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` path
variables -- `nnunetv2/paths.py`'s `_EnvPath` -- are actually read lazily,
on-demand, each time a path is used, not once at import time in v2.8.1; this
module still sets all three env vars before its first `import nnunetv2` anyway,
as a safe, version-independent convention rather than something this exact
nnunetv2 release strictly requires.) This module's public functions therefore
import `nnunetv2` lazily, inside the function body, and `main()` sets those
three env vars from its CLI arguments before doing anything that could trigger
nnunetv2's first import. `torch` has no such constraint (it is a base
dependency, pinned in `requirements.txt`, and reads no path env vars at import
time), so it is imported normally at module scope -- same as every other script
in this project.

Example usage (chained across sessions; `--expect-start-epoch` is the epoch the
PREVIOUS session's summary said it ended at, 0 for the very first session):

    python scripts/nnunet_session.py \\
        --raw-root /kaggle/working/nnUNet_raw \\
        --preprocessed-root /kaggle/working/nnUNet_preprocessed \\
        --results-root /kaggle/working/nnUNet_results \\
        --search-root /kaggle/input --prev-results-search-root /kaggle/input \\
        --budget-hours 10.5 --expect-start-epoch 0 --require-cuda \\
        --summary-json /kaggle/working/nnunet_session_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch  # noqa: E402  -- torch has no env-var-at-import constraint, unlike nnunetv2.

from neurovision.utils.io import read_json, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# nnU-Net v2's own three environment variables, read once at nnunetv2 import
# time. Set by `main()` before the first lazy `import nnunetv2` in this
# process -- see the module docstring for why the ordering is load-bearing.
_ENV_RAW = "nnUNet_raw"
_ENV_PREPROCESSED = "nnUNet_preprocessed"
_ENV_RESULTS = "nnUNet_results"

# The two plans-identifier-INDEPENDENT JSONs nnU-Net keeps at the top level
# of a preprocessed dataset folder (sibling to the per-configuration case
# directories). The third required file, "<plans_identifier>.json", varies
# with `--plans` (default "nnUNetPlans"), so it is never hardcoded here.
_FIXED_DATASET_JSONS = ("dataset.json", "dataset_fingerprint.json")

# How many plausible nesting levels to search under a Kaggle input mount.
# Bounded rather than an unbounded "**" glob -- same reasoning, and the same
# depth, as `scripts/export_nnunet_dataset.py::_MAX_GLOB_DEPTH`: a Kaggle
# dataset does not reliably mount at a fixed depth, and an unbounded glob can
# silently traverse into symlinked directories.
_MAX_SEARCH_DEPTH = 6

# nnU-Net's own checkpoint fallback order inside `maybe_load_checkpoint`
# (nnunetv2 v2.8.1, `run/run_training.py`): final, then latest, then best.
_CHECKPOINT_FALLBACK_ORDER = (
    "checkpoint_final.pth",
    "checkpoint_latest.pth",
    "checkpoint_best.pth",
)

# The default nnU-Net plans identifier, used only as this module's own
# default when a caller does not name a different one (e.g. a ResEnc preset).
_DEFAULT_PLANS_IDENTIFIER = "nnUNetPlans"


def _copy_file_writable(src: Path, dst: Path) -> None:
    """Copies a file's CONTENT only, then makes the destination owner-writable.

    `shutil.copy2` (the obvious choice) also copies the source's permission
    bits, so copying out of a read-only source -- exactly what a Kaggle input
    mount is -- produces a read-only destination too. nnU-Net's own
    `on_train_start` then fails with `PermissionError` the moment it tries to
    overwrite `plans.json` there (and the same would happen to
    `checkpoint_latest.pth` on the next `torch.save`). `shutil.copyfile`
    copies bytes only, so the destination's mode comes from the process
    umask rather than the source; the explicit `chmod` below is a second,
    unconditional guarantee that holds regardless of umask.

    Args:
        src: Source file (may be read-only).
        dst: Destination file path. Its parent directory must already exist.
    """
    shutil.copyfile(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# 1. Linking a read-only preprocessed dataset mount into a writable tree
# ---------------------------------------------------------------------------


def find_preprocessed_dataset(
    search_root: Path,
    dataset_folder: str,
    plans_identifier: str = _DEFAULT_PLANS_IDENTIFIER,
    max_depth: int = _MAX_SEARCH_DEPTH,
) -> Path:
    """Finds the one `dataset_folder` dir holding `<plans_identifier>.json` under `search_root`.

    A Kaggle-mounted preprocessed-dataset input does not reliably land at a
    fixed depth, so this checks depths `0` through `max_depth` rather than
    assuming one (mirrors `scripts/export_nnunet_dataset.py::discover_case_dir`).

    Args:
        search_root: Root directory to search under.
        dataset_folder: Exact directory name to look for, e.g.
            `"Dataset901_NeuroVisionXBraTS21"`.
        plans_identifier: The nnU-Net plans identifier this dataset was
            preprocessed with, e.g. `"nnUNetPlans"` or a ResEnc preset name --
            the file checked for is `f"{plans_identifier}.json"`.
        max_depth: Deepest nesting level to try (0 = directly under `search_root`).

    Returns:
        The single matching directory.

    Raises:
        FileNotFoundError: If `search_root` does not exist, or zero or more
            than one qualifying directory is found; the error names every
            same-named directory found (with or without the plans file) so a
            near-miss is visible.
    """
    plans_filename = f"{plans_identifier}.json"
    if not search_root.is_dir():
        raise FileNotFoundError(f"--search-root {search_root} does not exist.")

    patterns = ["/".join(["*"] * depth + [dataset_folder]) for depth in range(max_depth + 1)]
    named_dirs = sorted(
        {p.resolve() for pattern in patterns for p in search_root.glob(pattern) if p.is_dir()}
    )
    hits = [d for d in named_dirs if (d / plans_filename).is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one directory named {dataset_folder!r} containing "
            f"{plans_filename} under {search_root} (within depth {max_depth}); found "
            f"{len(hits)} candidate(s) with the file: {[str(h) for h in hits]}. Every "
            f"directory named {dataset_folder!r} found at any depth <= {max_depth}: "
            f"{[str(d) for d in named_dirs]}."
        )
    return hits[0]


def link_preprocessed(
    src_dataset_dir: Path,
    dst_preprocessed_root: Path,
    dataset_folder: str,
    configuration_dir: str = "nnUNetPlans_3d_fullres",
    plans_identifier: str = _DEFAULT_PLANS_IDENTIFIER,
) -> Path:
    """Builds a writable preprocessed-dataset dir out of a read-only Kaggle input mount.

    `src_dataset_dir` is read-only (a Kaggle input mount), but nnU-Net's own
    `on_train_start` writes `plans.json`/`dataset.json`/`dataset_fingerprint.json`
    into its preprocessed dataset folder on every session (see
    `nnUNetTrainer.on_train_start`), so `dst_preprocessed_root/<dataset_folder>/`
    is built as a real, writable directory: the three top-level JSONs are
    copied (content only, then made owner-writable -- see
    `_copy_file_writable`, since a plain `shutil.copy2` would carry the
    source's read-only mode bits straight through and nnU-Net's own write
    would fail with `PermissionError`), and `<configuration_dir>/` is a real
    directory whose files are each an INDIVIDUAL symlink back into
    `src_dataset_dir` -- never a whole-directory symlink. A previous Kaggle
    attempt found nnU-Net's `os.listdir`-based case counting silently
    under-counting through a directory-level symlink, so per-file linking is
    the only form this project trusts. The per-case symlinks stay read-only:
    nnU-Net only ever reads them.

    Args:
        src_dataset_dir: Read-only source, e.g.
            `.../nnUNet_preprocessed/Dataset901_NeuroVisionXBraTS21`, holding
            `<plans_identifier>.json`, `dataset.json`, `dataset_fingerprint.json`,
            and `configuration_dir/` (per-case `.b2nd`/`.pkl`, maybe `_seg.b2nd`).
        dst_preprocessed_root: Writable root to build the dataset dir under
            (this is `os.environ["nnUNet_preprocessed"]`).
        dataset_folder: The dataset directory's name, e.g.
            `"Dataset901_NeuroVisionXBraTS21"`.
        configuration_dir: The per-configuration subdirectory name to link.
        plans_identifier: The nnU-Net plans identifier this dataset was
            preprocessed with -- the required top-level file is
            `f"{plans_identifier}.json"`.

    Returns:
        The built (writable) dataset directory,
        `dst_preprocessed_root/<dataset_folder>`.

    Raises:
        FileNotFoundError: If any of the three required JSONs, or
            `configuration_dir`, is missing from `src_dataset_dir`.
    """
    required_jsons = (f"{plans_identifier}.json", *_FIXED_DATASET_JSONS)
    for name in required_jsons:
        if not (src_dataset_dir / name).is_file():
            raise FileNotFoundError(f"{src_dataset_dir} is missing required file {name!r}.")
    src_config_dir = src_dataset_dir / configuration_dir
    if not src_config_dir.is_dir():
        raise FileNotFoundError(
            f"{src_dataset_dir} is missing required directory {configuration_dir!r}."
        )

    dst_dataset_dir = dst_preprocessed_root / dataset_folder
    dst_dataset_dir.mkdir(parents=True, exist_ok=True)

    for name in required_jsons:
        _copy_file_writable(src_dataset_dir / name, dst_dataset_dir / name)

    dst_config_dir = dst_dataset_dir / configuration_dir
    dst_config_dir.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(src_config_dir.iterdir()):
        if not src_file.is_file():
            continue
        dst_file = dst_config_dir / src_file.name
        resolved_src = src_file.resolve()
        # Idempotent: if a correct symlink is already there, leave it alone
        # rather than unlinking and relinking on every session's setup step.
        if dst_file.is_symlink() and dst_file.resolve() == resolved_src:
            continue
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        dst_file.symlink_to(resolved_src)

    return dst_dataset_dir


# ---------------------------------------------------------------------------
# 2. Restoring a previous session's results tree
# ---------------------------------------------------------------------------


def find_prev_fold_dir(
    search_root: Path, trainer_dir_name: str, fold_name: str, max_depth: int = _MAX_SEARCH_DEPTH
) -> Path | None:
    """Finds a prior session's fold output dir under a Kaggle input mount.

    Args:
        search_root: Root to search under (e.g. `/kaggle/input`).
        trainer_dir_name: e.g. `"nnUNetTrainer__nnUNetPlans__3d_fullres"`.
        fold_name: e.g. `"fold_all"`.
        max_depth: Deepest nesting level to try.

    Returns:
        The single matching directory, or None if none is found (a fresh
        first session has nothing to restore).

    Raises:
        FileNotFoundError: If more than one directory matches -- an ambiguous
            source is named rather than silently picked.
    """
    if not search_root.is_dir():
        return None

    patterns = [
        "/".join(["*"] * depth + [trainer_dir_name, fold_name]) for depth in range(max_depth + 1)
    ]
    hits = sorted(
        {p.resolve() for pattern in patterns for p in search_root.glob(pattern) if p.is_dir()}
    )
    if not hits:
        return None
    if len(hits) > 1:
        raise FileNotFoundError(
            f"Expected at most one {trainer_dir_name}/{fold_name} directory under "
            f"{search_root} (within depth {max_depth}); found {len(hits)}: "
            f"{[str(h) for h in hits]}."
        )
    return hits[0]


# Sibling, trainer-level JSONs nnU-Net writes once at `on_train_start`
# (nnunetv2 v2.8.1 `nnUNetTrainer.on_train_start`), one directory above the
# fold's own output folder.
_TRAINER_LEVEL_JSONS = ("plans.json", "dataset.json", "dataset_fingerprint.json")


def restore_results(prev_fold_dir: Path | None, dst_fold_dir: Path) -> None:
    """Copies a previous session's fold output dir into the writable results tree.

    Every copied file is made owner-writable (`_copy_file_writable`), not
    just readable: `prev_fold_dir` may be a read-only Kaggle input mount, and
    a plain `shutil.copy2` would carry that read-only mode straight through
    to `checkpoint_latest.pth`/`plans.json`/etc, so the very next
    `torch.save` or nnU-Net `on_train_start` write into this session's
    results tree would fail with `PermissionError`.

    Args:
        prev_fold_dir: A prior session's fold directory (e.g.
            `.../nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/`), or None
            to do nothing (a fresh first session has nothing to restore).
        dst_fold_dir: This session's (writable) fold directory, created if
            missing.
    """
    if prev_fold_dir is None:
        return
    if not prev_fold_dir.is_dir():
        raise FileNotFoundError(f"prev_fold_dir {prev_fold_dir} does not exist.")

    dst_fold_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(prev_fold_dir.iterdir()):
        if item.is_file():
            _copy_file_writable(item, dst_fold_dir / item.name)

    prev_trainer_dir = prev_fold_dir.parent
    dst_trainer_dir = dst_fold_dir.parent
    for name in _TRAINER_LEVEL_JSONS:
        src = prev_trainer_dir / name
        if src.is_file():
            dst_trainer_dir.mkdir(parents=True, exist_ok=True)
            _copy_file_writable(src, dst_trainer_dir / name)


# ---------------------------------------------------------------------------
# 3. Reading and checking a checkpoint's identity
# ---------------------------------------------------------------------------


def read_checkpoint_meta(path: Path) -> dict[str, Any]:
    """Reads the identity fields this project checks before resuming a checkpoint.

    Args:
        path: Path to an nnU-Net checkpoint (`checkpoint_latest.pth`, etc).

    Returns:
        `{"current_epoch", "trainer_name", "plans", "configuration", "fold",
        "train_loss_finite"}`. `plans`/`configuration`/`fold` come from
        `checkpoint["init_args"]` (nnU-Net's own `my_init_kwargs`, saved by
        `save_checkpoint`). `train_loss_finite` is True iff every value in
        `checkpoint["logging"]["train_losses"]` (nnU-Net's own per-epoch
        training-loss log, see `nnunetv2.training.logging.nnunet_logger`) is
        finite -- CLAUDE.md trap #2: an fp32-sized `eps` clamp is a no-op in
        fp16 and produces a NaN loss that trains on, undetected, for hours.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    init_args = checkpoint.get("init_args") or {}
    train_losses = (checkpoint.get("logging") or {}).get("train_losses") or []
    train_loss_finite = all(math.isfinite(float(v)) for v in train_losses)
    return {
        "current_epoch": checkpoint.get("current_epoch"),
        "trainer_name": checkpoint.get("trainer_name"),
        "plans": init_args.get("plans"),
        "configuration": init_args.get("configuration"),
        "fold": init_args.get("fold"),
        "train_loss_finite": train_loss_finite,
    }


def check_identity(
    meta: dict[str, Any] | None,
    *,
    expect_start_epoch: int,
    expect_trainer: str,
    plans_json: dict[str, Any],
    configuration: str,
    fold: str | int,
) -> None:
    """Raises if a session's checkpoint (or lack of one) does not match expectations.

    A checkpoint from the wrong run, the wrong plans, or a NaN training loss
    loads and trains perfectly -- the damage only shows up hours later. This
    is the cheap, load-bearing check that runs before any of that time is
    spent.

    Args:
        meta: `read_checkpoint_meta`'s output, or None if no checkpoint exists.
        expect_start_epoch: The epoch this session is expected to resume from
            (0 for a brand new training run).
        expect_trainer: Expected `trainer_name`, e.g. `"nnUNetTrainer"`.
        plans_json: The `nnUNetPlans.json` this session will train with,
            already loaded as a dict.
        configuration: Expected configuration, e.g. `"3d_fullres"`.
        fold: Expected fold, e.g. `"all"` or an int.

    Raises:
        RuntimeError: On any identity mismatch (see the module's per-check
            messages), or if the checkpoint's logged training loss carries a
            non-finite value.
    """
    if expect_start_epoch == 0:
        if meta is not None:
            raise RuntimeError(
                "expect_start_epoch is 0 (a fresh run) but a checkpoint already exists "
                f"(current_epoch={meta.get('current_epoch')}). Point --expect-start-epoch "
                "at that checkpoint's real epoch, or start from an empty results dir."
            )
        return

    if meta is None:
        raise RuntimeError(
            f"--expect-start-epoch={expect_start_epoch} but no checkpoint was found to "
            "resume from -- either the previous session's results did not carry over, or "
            "this session was told the wrong start epoch (0 is required for a fresh run)."
        )
    if meta.get("current_epoch") != expect_start_epoch:
        raise RuntimeError(
            f"checkpoint is at current_epoch={meta.get('current_epoch')}, expected "
            f"{expect_start_epoch}. Wrong session's checkpoint attached, or the previous "
            "session did not end where its summary says."
        )
    if meta.get("trainer_name") != expect_trainer:
        raise RuntimeError(
            f"checkpoint trainer_name={meta.get('trainer_name')!r}, expected "
            f"{expect_trainer!r}."
        )
    # Round-trip both dicts through JSON before comparing: the checkpoint's
    # "plans" came back through torch.load (which can carry numpy scalars or
    # tuples where the plain-JSON-loaded `plans_json` has floats and lists),
    # and this comparison must not fail on a difference that is only a
    # representation artifact.
    checkpoint_plans = json.loads(json.dumps(meta.get("plans"), default=str))
    expected_plans = json.loads(json.dumps(plans_json, default=str))
    if checkpoint_plans != expected_plans:
        raise RuntimeError(
            "checkpoint's plans dict does not match this session's nnUNetPlans.json -- this "
            "checkpoint was trained on a different (or since-edited) plan."
        )
    if str(meta.get("configuration")) != str(configuration):
        raise RuntimeError(
            f"checkpoint configuration={meta.get('configuration')!r}, expected "
            f"{configuration!r}."
        )
    if str(meta.get("fold")) != str(fold):
        raise RuntimeError(f"checkpoint fold={meta.get('fold')!r}, expected {fold!r}.")
    if not meta.get("train_loss_finite", False):
        raise RuntimeError(
            "checkpoint's logged train_losses contain a non-finite value -- refusing to "
            "resume it (CLAUDE.md trap #2: AMP's GradScaler can hide a NaN loss behind "
            "perfectly finite weights for an entire run)."
        )


# ---------------------------------------------------------------------------
# 4. Deciding when to stop, and running the budgeted epoch loop
# ---------------------------------------------------------------------------


def should_stop(
    elapsed_s: float,
    epoch_durations_s: Sequence[float],
    budget_s: float,
    *,
    first_epoch_estimate_s: float,
    safety_factor: float = 1.5,
) -> bool:
    """Decides, before starting the next epoch, whether this session should stop.

    Args:
        elapsed_s: Wall-clock time elapsed since this session's process
            started (setup included -- the caller's budget is the whole
            session's allowance, not just time spent training).
        epoch_durations_s: Durations of every epoch already run this session,
            in order.
        budget_s: This session's total wall-clock budget, in seconds.
        first_epoch_estimate_s: Assumed duration for the next epoch when no
            epoch has run yet this session.
        safety_factor: Multiplier applied to the duration estimate, so a
            session stops with margin rather than gambling on the next epoch
            running exactly as fast as the recent ones.

    Returns:
        True iff `elapsed_s + safety_factor * <predicted next-epoch duration>`
        would exceed `budget_s`. The predicted duration is the max of the last
        5 recorded epoch durations, or `first_epoch_estimate_s` if none yet.
    """
    recent = list(epoch_durations_s)[-5:]
    predicted_next_epoch_s = safety_factor * (max(recent) if recent else first_epoch_estimate_s)
    return (elapsed_s + predicted_next_epoch_s) > budget_s


def _last_logged(trainer: Any, key: str) -> Any:
    """Returns the last value nnU-Net's own logger has for `key`, or None if empty."""
    values = trainer.logger.get_value(key, step=None)
    return values[-1] if values else None


def _empty_cache_equivalent(device: Any) -> None:
    """Reimplements nnU-Net's own `empty_cache` helper using torch alone.

    Deliberately NOT `from nnunetv2.utilities.helpers import empty_cache`: the
    three-line body is trivial and torch-only, so reimplementing it here means
    the early-stop path (see `run_budgeted_training`) never needs an
    nnunetv2/batchgenerators import to run -- which matters because this exact
    code path is exercised by this module's own CPU tests, in the main `.venv`,
    against a FAKE trainer, where nnunetv2 is never installed.

    Args:
        device: A `torch.device` (or any object with a `.type` attribute).
    """
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        # Never actually reached in this project (CLAUDE.md: MPS is a
        # correctness harness only, never a training device), kept only so
        # this mirrors nnU-Net's own `empty_cache` exactly.
        torch.mps.empty_cache()
    # else: CPU -- nothing to empty, matching upstream's own no-op branch.


def _shutdown_dataloaders(trainer: Any) -> None:
    """Copies `nnUNetTrainer.on_train_end`'s dataloader-shutdown section verbatim.

    Only `MultiThreadedAugmenter`/`NonDetMultiThreadedAugmenter` dataloaders
    need an explicit `._finish()` call. Both classes live in `batchgenerators`,
    which is only ever installed alongside `nnunetv2` (`.venv-analysis`) --
    this project's main `.venv` runs this exact function against a FAKE
    trainer in tests, so an `ImportError` here just means "there is nothing of
    that type to shut down" (the FAKE dataloaders are not instances of these
    classes either), not a bug.

    Args:
        trainer: The (possibly real) `nnUNetTrainer` whose dataloaders should
            be shut down.
    """
    try:
        from batchgenerators.dataloading.multi_threaded_augmenter import (
            MultiThreadedAugmenter,
        )
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
            NonDetMultiThreadedAugmenter,
        )

        augmenter_types: tuple[type, ...] = (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)
    except ImportError:
        augmenter_types = ()

    old_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            if (
                trainer.dataloader_train is not None
                and augmenter_types
                and isinstance(trainer.dataloader_train, augmenter_types)
            ):
                trainer.dataloader_train._finish()
            if (
                trainer.dataloader_val is not None
                and augmenter_types
                and isinstance(trainer.dataloader_val, augmenter_types)
            ):
                trainer.dataloader_val._finish()
    finally:
        sys.stdout = old_stdout


def run_budgeted_training(
    trainer: Any,
    *,
    budget_s: float,
    first_epoch_estimate_s: float,
    clock: Callable[[], float] = time.monotonic,
    elapsed_before_s: float = 0.0,
) -> dict[str, Any]:
    """Runs `trainer`'s epoch loop, stopping between epochs at the wall-clock budget.

    Replicates `nnUNetTrainer.run_training` exactly (nnunetv2 v2.8.1,
    `training/nnUNetTrainer/nnUNetTrainer.py` lines 1417-1438: `on_train_start`;
    per epoch `on_epoch_start`, `on_train_epoch_start`, the
    `num_iterations_per_epoch` `train_step` loop, `on_train_epoch_end`, the
    `torch.no_grad()` validation block, `on_epoch_end`; then, only if every
    epoch ran, `on_train_end`) -- the only addition is a `should_stop` check
    immediately before each epoch's `on_epoch_start`. No other change is made
    to the trainer or the hook calls themselves.

    `trainer.save_every` is set to 1 before the loop, so `on_epoch_end` writes
    `checkpoint_latest.pth` every epoch (upstream's default, 50, would lose up
    to 49 epochs of progress to a session that stops or is killed between
    saves; upstream also skips that write on the very last epoch, relying on
    `on_train_end` to write `checkpoint_final.pth` instead -- unchanged here).

    Args:
        trainer: An (initialized) `nnUNetTrainer`-shaped object: needs
            `current_epoch`, `num_epochs`, `num_iterations_per_epoch`,
            `num_val_iterations_per_epoch`, `dataloader_train`,
            `dataloader_val`, `device`, `logger`, `save_every`, and the hook
            methods named above.
        budget_s: Wall-clock budget for this call, in seconds.
        first_epoch_estimate_s: Passed through to `should_stop` for the first
            epoch this call runs.
        clock: Zero-argument wall-clock function. Defaults to
            `time.monotonic`; tests pass a deterministic fake.
        elapsed_before_s: Wall-clock time already spent THIS SESSION before
            this call started (e.g. CLI parsing, preflight checks, linking
            the preprocessed dataset, building the trainer). The budget is
            the whole session's allowance measured from process start (see
            `main`), not just the time spent inside this function, so this
            is added to every `should_stop` check via the loop's own
            `start_time` rather than compared separately.

    Returns:
        `{"start_epoch", "end_epoch", "epochs_run", "finished",
        "epoch_durations_s", "mean_epoch_s", "last_pseudo_dice",
        "last_ema_fg_dice", "train_losses_finite"}`. `end_epoch` is
        `trainer.current_epoch` after the loop -- the next epoch a following
        session should resume from.
    """
    trainer.save_every = 1
    start_epoch = trainer.current_epoch
    epoch_durations_s: list[float] = []
    # Subtracting elapsed_before_s here (rather than adding it to every
    # should_stop call separately) means every `clock() - start_time` below
    # already reads as "time since the SESSION started", not just "time
    # since this function was entered" -- see the elapsed_before_s docstring.
    start_time = clock() - elapsed_before_s

    trainer.on_train_start()

    while trainer.current_epoch < trainer.num_epochs:
        elapsed_s = clock() - start_time
        if should_stop(
            elapsed_s,
            epoch_durations_s,
            budget_s,
            first_epoch_estimate_s=first_epoch_estimate_s,
        ):
            break

        epoch_start = clock()

        # --- nnUNetTrainer.run_training's own per-epoch body, unmodified ---
        trainer.on_epoch_start()

        trainer.on_train_epoch_start()
        train_outputs = [
            trainer.train_step(next(trainer.dataloader_train))
            for _ in range(trainer.num_iterations_per_epoch)
        ]
        trainer.on_train_epoch_end(train_outputs)

        with torch.no_grad():
            trainer.on_validation_epoch_start()
            val_outputs = [
                trainer.validation_step(next(trainer.dataloader_val))
                for _ in range(trainer.num_val_iterations_per_epoch)
            ]
            trainer.on_validation_epoch_end(val_outputs)

        trainer.on_epoch_end()
        # --- end of upstream's per-epoch body ---

        epoch_durations_s.append(clock() - epoch_start)

    finished = trainer.current_epoch >= trainer.num_epochs
    if finished:
        trainer.on_train_end()
    else:
        _shutdown_dataloaders(trainer)
        _empty_cache_equivalent(trainer.device)

    train_losses = trainer.logger.get_value("train_losses", step=None) or []
    train_losses_finite = all(math.isfinite(float(v)) for v in train_losses)

    return {
        "start_epoch": start_epoch,
        "end_epoch": trainer.current_epoch,
        "epochs_run": len(epoch_durations_s),
        "finished": finished,
        "epoch_durations_s": epoch_durations_s,
        "mean_epoch_s": (
            sum(epoch_durations_s) / len(epoch_durations_s) if epoch_durations_s else 0.0
        ),
        "last_pseudo_dice": _last_logged(trainer, "dice_per_class_or_region"),
        "last_ema_fg_dice": _last_logged(trainer, "ema_fg_dice"),
        "train_losses_finite": train_losses_finite,
    }


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses this script's CLI arguments.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        The parsed `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Sets env var nnUNet_raw.")
    parser.add_argument(
        "--preprocessed-root", required=True, help="Sets env var nnUNet_preprocessed."
    )
    parser.add_argument("--results-root", required=True, help="Sets env var nnUNet_results.")
    parser.add_argument("--dataset-id", type=int, default=901)
    parser.add_argument("--dataset-folder", default="Dataset901_NeuroVisionXBraTS21")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", default="all")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument(
        "--budget-hours",
        type=float,
        required=True,
        help="This session's whole wall-clock allowance, measured from process start.",
    )
    parser.add_argument("--first-epoch-estimate-s", type=float, default=310.0)
    parser.add_argument(
        "--expect-start-epoch",
        type=int,
        required=True,
        help="0 for a brand new training run; otherwise the epoch the prior session ended at.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast unless CUDA is available, runs a kernel, and is sm_70+ (no P100).",
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--preprocessed-src",
        default=None,
        help="A specific preprocessed dataset dir to link in via link_preprocessed.",
    )
    parser.add_argument(
        "--search-root",
        default=None,
        help="Search under this root for the preprocessed dataset dir (find_preprocessed_dataset)",
    )
    parser.add_argument(
        "--prev-results-search-root",
        default=None,
        help="Search under this root for a previous session's fold output dir to restore.",
    )
    return parser.parse_args(argv)


def _set_nnunet_env(raw_root: str, preprocessed_root: str, results_root: str) -> None:
    """Sets nnU-Net v2's own path env vars, before its first import in this process.

    Args:
        raw_root: Value for `nnUNet_raw`.
        preprocessed_root: Value for `nnUNet_preprocessed`.
        results_root: Value for `nnUNet_results`.
    """
    os.environ[_ENV_RAW] = raw_root
    os.environ[_ENV_PREPROCESSED] = preprocessed_root
    os.environ[_ENV_RESULTS] = results_root


def _require_cuda_or_raise() -> None:
    """Fails fast unless CUDA is present, can run a kernel, and is sm_70 or newer.

    Mirrors `scripts/gpu_session.py::run_cuda_smoke_check`, with the sm_70
    floor made explicit: CLAUDE.md names the Kaggle P100 (sm_60) as unusable
    for this project's pinned torch build, and Amendment 1's whole recipe
    assumes a Kaggle T4.

    Raises:
        RuntimeError: If CUDA is unavailable, cannot run a kernel, or reports
            a capability below sm_70.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--require-cuda given but torch.cuda.is_available() is False.")
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    try:
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    except Exception as exc:
        raise RuntimeError(
            f"{name} (sm_{major}{minor}) reports CUDA available but cannot run a kernel: {exc}"
        ) from exc
    if (major, minor) < (7, 0):
        raise RuntimeError(
            f"{name} is sm_{major}{minor}, below the sm_70 floor -- the P100 is unusable for "
            "this recipe. Request machine_shape=NvidiaTeslaT4 on Kaggle."
        )
    logger.info("CUDA smoke check passed: %s (sm_%d%d)", name, major, minor)


def _find_existing_checkpoint(output_folder: Path) -> Path | None:
    """Locates a checkpoint the same way `maybe_load_checkpoint(continue_training=True)` does.

    Args:
        output_folder: A trainer's `output_folder` (its `fold_<fold>` dir).

    Returns:
        The first of `checkpoint_final.pth`, `checkpoint_latest.pth`,
        `checkpoint_best.pth` that exists, in that order, or None if none do.
    """
    for name in _CHECKPOINT_FALLBACK_ORDER:
        candidate = output_folder / name
        if candidate.is_file():
            return candidate
    return None


def _nnunetv2_version() -> str | None:
    """Reads the installed `nnunetv2` package version, or None if unavailable.

    `nnunetv2` has no `__version__` attribute (checked against the pinned
    2.8.1 install), so this reads the distribution metadata instead.
    """
    try:
        from importlib.metadata import version

        return version("nnunetv2")
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """Parses args, runs (at most) one budgeted training session, and reports health.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        0 on success (including "already complete, nothing to do"), 1 on any
        failure (a preflight check, a non-finite training loss, or zero
        epochs run while not finished).
    """
    # The very first statement: --budget-hours is the WHOLE session's
    # allowance (CLAUDE.md / the module docstring), so the clock has to start
    # before argument parsing, not right before the training loop -- a slow
    # preflight (linking a large preprocessed dataset, restoring a big prior
    # results tree) must count against the budget too.
    process_start = time.monotonic()
    args = parse_args(argv)
    setup_logging(level="INFO")

    try:
        if args.require_cuda:
            _require_cuda_or_raise()

        # Must happen before the first `import nnunetv2` below -- see module
        # docstring.
        _set_nnunet_env(args.raw_root, args.preprocessed_root, args.results_root)

        # Derived from --plans/--configuration, never hardcoded: a non-default
        # plans identifier (e.g. a ResEnc preset) or configuration would
        # otherwise silently look for/build "nnUNetPlans_3d_fullres" and
        # "nnUNetPlans.json" regardless of what this session actually trains.
        configuration_dir = f"{args.plans}_{args.configuration}"

        if args.preprocessed_src is not None:
            src_dataset_dir: Path | None = Path(args.preprocessed_src)
        elif args.search_root is not None:
            src_dataset_dir = find_preprocessed_dataset(
                Path(args.search_root), args.dataset_folder, plans_identifier=args.plans
            )
        else:
            src_dataset_dir = None
        if src_dataset_dir is not None:
            link_preprocessed(
                src_dataset_dir,
                Path(args.preprocessed_root),
                args.dataset_folder,
                configuration_dir=configuration_dir,
                plans_identifier=args.plans,
            )

        trainer_dir_name = f"{args.trainer}__{args.plans}__{args.configuration}"
        fold_name = f"fold_{args.fold}"
        if args.prev_results_search_root is not None:
            prev_fold_dir = find_prev_fold_dir(
                Path(args.prev_results_search_root), trainer_dir_name, fold_name
            )
            dst_fold_dir = (
                Path(args.results_root) / args.dataset_folder / trainer_dir_name / fold_name
            )
            restore_results(prev_fold_dir, dst_fold_dir)

        # nnU-Net's own build-a-trainer entry point (nnunetv2 v2.8.1,
        # run/run_training.py). Imported here, lazily, and only after the env
        # vars above are set.
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
        from torch.backends import cudnn

        fold_arg: str | int = args.fold if args.fold == "all" else int(args.fold)
        device = torch.device(args.device)
        trainer = get_trainer_from_args(
            dataset_name_or_id=args.dataset_folder,
            configuration=args.configuration,
            fold=fold_arg,
            trainer_name=args.trainer,
            plans_identifier=args.plans,
            device=device,
        )

        output_folder = Path(trainer.output_folder)
        checkpoint_path = _find_existing_checkpoint(output_folder)
        meta = read_checkpoint_meta(checkpoint_path) if checkpoint_path is not None else None

        plans_path = Path(args.preprocessed_root) / args.dataset_folder / f"{args.plans}.json"
        plans_json = read_json(plans_path)

        check_identity(
            meta,
            expect_start_epoch=args.expect_start_epoch,
            expect_trainer=args.trainer,
            plans_json=plans_json,
            configuration=args.configuration,
            fold=args.fold,
        )

        if checkpoint_path is not None:
            # nnU-Net's own resume path: restores weights, optimizer, AMP
            # scaler, logger and epoch counter (see nnUNetTrainer.load_checkpoint).
            maybe_load_checkpoint(trainer, True, False, None)

        if checkpoint_path is not None and checkpoint_path.name == "checkpoint_final.pth":
            logger.info(
                "checkpoint_final.pth already exists at %s -- training is already complete.",
                output_folder,
            )
            print(
                f"NVX_NNUNET_HEALTH: OK start={meta['current_epoch']} "
                f"end={meta['current_epoch']} finished=True"
            )
            return 0

        # Matches nnU-Net's own top-level run_training() function exactly
        # (run/run_training.py): cudnn.benchmark only makes sense once CUDA
        # is actually present.
        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        # Everything above this point (CLI parsing, CUDA check, linking the
        # preprocessed dataset, restoring a prior results tree, building the
        # trainer, loading a checkpoint) already spent wall-clock time out of
        # this session's budget -- passed through so the FIRST should_stop
        # check inside run_budgeted_training already accounts for it.
        elapsed_before_s = time.monotonic() - process_start
        result = run_budgeted_training(
            trainer,
            budget_s=args.budget_hours * 3600.0,
            first_epoch_estimate_s=args.first_epoch_estimate_s,
            elapsed_before_s=elapsed_before_s,
        )
        elapsed_s = time.monotonic() - process_start

        summary = {
            **{k: v for k, v in result.items() if k != "epoch_durations_s"},
            "epoch_durations_s": result["epoch_durations_s"],
            "elapsed_s": elapsed_s,
            "budget_hours": args.budget_hours,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "nnunetv2_version": _nnunetv2_version(),
        }
        write_json(summary, Path(args.summary_json))

        if not result["train_losses_finite"]:
            print(
                f"NVX_NNUNET_HEALTH: FAIL non-finite train loss "
                f"(start={result['start_epoch']} end={result['end_epoch']})"
            )
            return 1
        if result["epochs_run"] == 0 and not result["finished"]:
            print(
                f"NVX_NNUNET_HEALTH: FAIL zero epochs run while not finished "
                f"(start={result['start_epoch']})"
            )
            return 1

        # A single `print`, not `logging`: Kaggle's log viewer is scraped by a
        # simple substring search for this exact prefix, regardless of what
        # logging handlers are (or are not) configured -- same reasoning as
        # `scripts/gpu_session.py::print_health_line`.
        print(
            f"NVX_NNUNET_HEALTH: OK start={result['start_epoch']} end={result['end_epoch']} "
            f"finished={result['finished']}"
        )
        return 0
    except Exception as exc:
        logger.exception("nnunet_session failed:")
        print(f"NVX_NNUNET_HEALTH: FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
