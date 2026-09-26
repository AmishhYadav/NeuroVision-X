"""One entry point for a safe training session on any GPU host.

`notebooks/kaggle_train.ipynb` is a *thin driver notebook*: no training logic
lives in its cells, only wiring. This script lifts that wiring -- cells 2, 5,
7, 9, 11, 13, 15 -- into a plain Python CLI so the exact same preflight and
post-training checks run whether the caller is a Kaggle notebook, an
SSH+tmux session on a box the author owns, or a SLURM job on the college
cluster. Cloning the repo and `pip install` happen *before* this script runs
(they are shell/CI concerns, not Python); this script starts from an already
checked-out tree and an already-installed environment, and reads secrets
(`WANDB_API_KEY`) from the environment only.

Every step is its own small, separately-testable function; `main()` just
sequences them. See CLAUDE.md and docs/gpu_session_checklist.md for the
constraints this file exists to enforce:

    - device is never assumed present at import time (get_device resolves it
      from config, inside `run_training`);
    - no hardcoded filesystem paths -- every path is a CLI argument or is
      derived from `__file__` (the config directory, the sibling train.py);
    - a checkpoint from the wrong run must be caught before it burns a whole
      session (`--expect-epoch` / `--expect-wandb-id` / non-finite scanning);
    - `--expect-sha` catches a pinned GIT_REF that silently drifted.

Example usage:

    python scripts/gpu_session.py --experiment neurovision \\
        --data-root /kaggle/input/neurovision-brats-prep \\
        --require-cuda --wandb-mode online

    # resuming a prior session's output, with identity checks:
    python scripts/gpu_session.py --experiment neurovision \\
        --search-root /kaggle/input \\
        --ckpt-src /kaggle/input/prior-session-output/checkpoints \\
        --expect-epoch 35 --expect-wandb-id abc123 --require-cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def set_cuda_allocator_env() -> None:
    """Sets the CUDA allocator env vars torch reads at import time.

    Must run before `import torch` anywhere in this process -- once torch's
    CUDA caching allocator has initialized, setting these afterward has no
    effect. `expandable_segments:True` lets the allocator grow a segment in
    place instead of fragmenting into many fixed-size blocks, which is the
    difference between a long run OOMing on fragmentation and not.
    `PYTORCH_ALLOC_CONF` is the newer, generic spelling torch is migrating
    towards; `PYTORCH_CUDA_ALLOC_CONF` is the one still checked by the
    pinned torch version here. Both are set so neither name matters.
    `setdefault` so an operator's own env value (e.g. a SLURM job template)
    is never clobbered.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# Runs at import time, before the `import torch` below -- see the function's
# own docstring for why this ordering is load-bearing. Every import after
# this point that might pull in torch (directly or transitively) is marked
# `noqa: E402`, matching the established pattern in scripts/validate_atlas.py
# (which does the same thing for matplotlib's backend selection).
set_cuda_allocator_env()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from neurovision.training.checkpoint import (  # noqa: E402
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    find_resume_checkpoint,
)
from neurovision.utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Relative to this file, so this script works from any working directory and
# on any machine -- no absolute paths. Same convention as scripts/train.py.
_REPO_DIR = Path(__file__).resolve().parent.parent
_CONFIG_DIR = str(_REPO_DIR / "configs")
_TRAIN_SCRIPT_PATH = _REPO_DIR / "scripts" / "train.py"

# The on-disk layout scripts/package_for_kaggle.py builds and
# kaggle_train.ipynb's cell 9 discovers: a directory holding both a
# `preprocessed/` tree and a `splits.yaml` file.
_SPLITS_FILENAME = "splits.yaml"
_PREPROCESSED_DIRNAME = "preprocessed"

# Matches kaggle_train.ipynb cell 9 exactly: it globs "splits.yaml",
# "*/splits.yaml", "*/*/splits.yaml", "*/*/*/splits.yaml" -- i.e. depths 0
# through 3 -- because a Kaggle dataset does not reliably mount at the
# documented depth.
_SEARCH_MAX_DEPTH = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses this script's CLI arguments.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        The parsed `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", required=True, help="Hydra +experiment= name, e.g. 'neurovision'."
    )

    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument(
        "--data-root",
        default=None,
        help="Directory holding preprocessed/ and splits.yaml directly (no search).",
    )
    root_group.add_argument(
        "--search-root",
        default=None,
        help=(
            "Root to search, bounded depth, for the one directory holding "
            "preprocessed/ and splits.yaml (mirrors kaggle_train.ipynb cell 9)."
        ),
    )

    parser.add_argument(
        "--ckpt-src",
        default=None,
        help="Prior session's output dir holding last.pt (and optionally best.pt).",
    )
    parser.add_argument(
        "--expect-epoch",
        type=int,
        default=None,
        help="Fail unless the carried-forward checkpoint is at exactly this epoch.",
    )
    parser.add_argument(
        "--expect-wandb-id",
        default=None,
        help="Fail unless the carried-forward checkpoint's wandb_run_id matches.",
    )
    parser.add_argument(
        "--expect-sha",
        default=None,
        help="Fail unless the repo's current HEAD commit SHA matches exactly.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless CUDA is available and can actually run a kernel.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="online needs WANDB_API_KEY in the environment; offline needs nothing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight only (no training), then exit 0.",
    )
    parser.add_argument(
        "--override",
        dest="override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override, repeatable, applied after every fixed override above.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Preflight: git SHA (notebook cell 2/4's pin, checked here rather than cloned)
# ---------------------------------------------------------------------------


def check_git_sha(repo_dir: Path, expect_sha: str | None) -> str:
    """Reads the repo's current HEAD commit SHA and optionally checks it.

    Cloning at a pinned SHA is out of scope for this script (it happens
    before Python runs) -- this only verifies the tree Python is *already*
    running from is the one the caller expects, catching the case where a
    `GIT_REF` pin drifted between "what was requested" and "what actually got
    checked out".

    Args:
        repo_dir: Path to the git working tree to inspect.
        expect_sha: Required SHA, or None to skip the check.

    Returns:
        The resolved HEAD SHA.

    Raises:
        RuntimeError: If `expect_sha` is given and does not match HEAD.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip()
    logger.info("git HEAD SHA: %s", sha)
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(
            f"HEAD is at {sha}, expected {expect_sha}. The wrong commit is checked out "
            "for this run -- verifying a branch name proves nothing about a pinned SHA."
        )
    return sha


# ---------------------------------------------------------------------------
# Preflight: CUDA smoke check (notebook cell 5)
# ---------------------------------------------------------------------------


def run_cuda_smoke_check() -> None:
    """Asserts CUDA is available AND can actually execute a kernel.

    `torch.cuda.is_available()` alone is not sufficient: on a Kaggle P100 it
    returns True while every kernel launch fails, because stock torch no
    longer targets sm_60 (min sm_70). Only executing something is honest, so
    this runs a tiny matmul the same way kaggle_train.ipynb's cell 5 does.

    Raises:
        RuntimeError: If CUDA is unavailable, or is available but a kernel
            launch fails.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Device 'cuda' was requested (--require-cuda) but torch.cuda.is_available() "
            "is False. Either the GPU accelerator is off, or pip reinstalled a CPU torch "
            "wheel over Kaggle's CUDA-matched build."
        )
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"sm_{major}{minor}"
    try:
        (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
    except Exception as exc:
        raise RuntimeError(
            f"{name} ({capability}) reports CUDA available but cannot run a kernel: {exc}. "
            "Request machine_shape=NvidiaTeslaT4 on Kaggle -- the P100 (sm_60) is unusable."
        ) from exc
    logger.info("CUDA smoke check passed: %s (%s)", name, capability)


# ---------------------------------------------------------------------------
# Preflight: data root discovery (notebook cell 9)
# ---------------------------------------------------------------------------


def _has_expected_data_layout(candidate: Path) -> bool:
    """True if `candidate` directly holds both `preprocessed/` and `splits.yaml`."""
    return (candidate / _SPLITS_FILENAME).is_file() and (candidate / _PREPROCESSED_DIRNAME).is_dir()


def discover_data_root(search_root: Path, max_depth: int = _SEARCH_MAX_DEPTH) -> Path:
    """Finds the one directory under `search_root` with the expected layout.

    Mirrors kaggle_train.ipynb cell 9's discovery exactly: a mounted dataset
    does not reliably land at a fixed depth, so this globs depths `0` through
    `max_depth` rather than assuming one. Raises loudly (rather than picking
    an arbitrary hit) if zero or more than one directory qualifies.

    Args:
        search_root: Root directory to search under (searched at every depth
            from 0 to `max_depth`, inclusive).
        max_depth: Maximum subdirectory depth to search. Defaults to the
            depth the Kaggle notebook searches.

    Returns:
        The single matching directory.

    Raises:
        FileNotFoundError: If `search_root` does not exist, or zero or more
            than one directory within depth qualifies.
    """
    if not search_root.is_dir():
        raise FileNotFoundError(f"--search-root {search_root} does not exist.")

    patterns = ["/".join(["*"] * depth + [_SPLITS_FILENAME]) for depth in range(max_depth + 1)]
    hits = sorted(
        {
            p.parent
            for pattern in patterns
            for p in search_root.glob(pattern)
            if _has_expected_data_layout(p.parent)
        }
    )
    if len(hits) != 1:
        present = sorted(str(p.relative_to(search_root)) for p in search_root.glob("*"))
        raise FileNotFoundError(
            f"Expected exactly one directory under {search_root} (within depth {max_depth}) "
            f"holding both {_PREPROCESSED_DIRNAME}/ and {_SPLITS_FILENAME}, found "
            f"{[str(h) for h in hits]}. Present at top level: {present[:20]}"
        )
    return hits[0]


def resolve_data_root(data_root: str | None, search_root: str | None) -> Path:
    """Resolves the directory holding `preprocessed/` and `splits.yaml`.

    Exactly one of `data_root` / `search_root` is expected to be set --
    `parse_args` enforces this with a mutually exclusive argparse group, so
    both being None here would be a caller bug, not a user error.

    Args:
        data_root: Direct path to the data directory, or None.
        search_root: Root to search under (bounded depth), or None.

    Returns:
        The resolved data root directory.

    Raises:
        FileNotFoundError: If `data_root` is given but lacks the expected
            layout, or if `discover_data_root` cannot find a unique match.
        ValueError: If neither argument is set.
    """
    if data_root is not None:
        root = Path(data_root)
        if not _has_expected_data_layout(root):
            raise FileNotFoundError(
                f"--data-root {root} does not hold both {_PREPROCESSED_DIRNAME}/ and "
                f"{_SPLITS_FILENAME}."
            )
        return root
    if search_root is not None:
        return discover_data_root(Path(search_root))
    raise ValueError("One of --data-root or --search-root is required.")


# ---------------------------------------------------------------------------
# Config composition (notebook cell 13, minus the actual training call)
# ---------------------------------------------------------------------------


def build_hydra_overrides(args: argparse.Namespace, data_root: Path) -> list[str]:
    """Builds the Hydra override strings for this session.

    Args:
        args: Parsed CLI arguments.
        data_root: Resolved directory holding `preprocessed/` and
            `splits.yaml`.

    Returns:
        Override strings in application order: the experiment, the three
        data-path overrides, the W&B mode, then every user `--override` --
        so a user override can always win over the fixed ones above it.
    """
    overrides = [
        f"+experiment={args.experiment}",
        f"data.root_dir={data_root}",
        f"data.preprocessing.out_dir={data_root / _PREPROCESSED_DIRNAME}",
        f"data.splits.path={data_root / _SPLITS_FILENAME}",
        f"wandb.mode={args.wandb_mode}",
    ]
    overrides.extend(args.override)
    return overrides


def compose_config(args: argparse.Namespace, data_root: Path) -> DictConfig:
    """Composes the real Hydra config for this session.

    Uses `hydra.compose` (not `@hydra.main`), exactly like
    `scripts/smoke_test.py` and `kaggle_train.ipynb` cell 13, so this script
    can pick its own overrides rather than taking them straight off `sys.argv`.

    Args:
        args: Parsed CLI arguments.
        data_root: Resolved directory holding `preprocessed/` and
            `splits.yaml`.

    Returns:
        The composed `DictConfig`.
    """
    overrides = build_hydra_overrides(args, data_root)
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    logger.info(
        "composed config: experiment_name=%s, epochs=%d, data.root_dir=%s",
        cfg.experiment_name,
        cfg.training.epochs,
        cfg.data.root_dir,
    )
    return cfg


# ---------------------------------------------------------------------------
# Preflight: checkpoint carry-forward (notebook cell 11)
# ---------------------------------------------------------------------------


def _read_checkpoint_epoch(path: Path) -> int | None:
    """Reads just the `epoch` field of a checkpoint file.

    Args:
        path: Path to a checkpoint written by `save_checkpoint`.

    Returns:
        The saved epoch, or None if the checkpoint has no `epoch` key.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("epoch")


def copy_checkpoints(ckpt_src: Path, checkpoint_dir: Path) -> tuple[Path, Path | None]:
    """Copies `last.pt` (required) and `best.pt` (optional) into the run dir.

    Never rolls progress back. If `checkpoint_dir` already holds a
    `last.pt`:

    - at a LATER epoch than `ckpt_src`'s -- refuse outright. A wrong
      `--ckpt-src` (e.g. an earlier session reattached by mistake) must fail
      loudly rather than silently erase further progress already sitting in
      `checkpoint_dir`.
    - at the SAME epoch -- skip the copy (already up to date) and log it.
    - at an EARLIER epoch (or no `last.pt` at all) -- copy as normal.

    `best.pt` is not required: `save_checkpoint` only writes it when
    validation improves, so a final session that never beats an earlier best
    legitimately has none -- see the "best.pt ABSENT" case in
    kaggle_train.ipynb cell 15's docstring. Only `last.pt` is load-bearing for
    resume, so it is the only file this rollback check applies to.

    Args:
        ckpt_src: Prior session's output directory.
        checkpoint_dir: This session's (writable) checkpoint directory,
            created if missing.

    Returns:
        `(last_dst, best_dst)`; `best_dst` is None if no `best.pt` existed
        at `ckpt_src`.

    Raises:
        FileNotFoundError: If `ckpt_src` has no `last.pt`.
        RuntimeError: If `checkpoint_dir` already holds a strictly LATER
            epoch than `ckpt_src` -- copying would roll progress back.
    """
    last_src = ckpt_src / LAST_CHECKPOINT_NAME
    if not last_src.is_file():
        raise FileNotFoundError(
            f"--ckpt-src {ckpt_src} has no {LAST_CHECKPOINT_NAME} -- nothing to resume from."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_dst = checkpoint_dir / LAST_CHECKPOINT_NAME

    if last_dst.is_file():
        src_epoch = _read_checkpoint_epoch(last_src)
        dst_epoch = _read_checkpoint_epoch(last_dst)
        if src_epoch is not None and dst_epoch is not None and dst_epoch > src_epoch:
            raise RuntimeError(
                f"{last_dst} is already at epoch {dst_epoch}, later than --ckpt-src "
                f"{last_src}'s epoch {src_epoch}. Refusing to copy over it -- that would "
                "roll progress back. Point --ckpt-src at a later checkpoint, or drop it "
                "to resume in place."
            )
        if src_epoch is not None and dst_epoch is not None and dst_epoch == src_epoch:
            logger.info(
                "%s already at epoch %d, matching --ckpt-src -- skipping the copy.",
                last_dst,
                dst_epoch,
            )
        else:
            shutil.copy2(last_src, last_dst)
    else:
        shutil.copy2(last_src, last_dst)

    best_src = ckpt_src / BEST_CHECKPOINT_NAME
    best_dst: Path | None = None
    if best_src.is_file():
        best_dst = checkpoint_dir / BEST_CHECKPOINT_NAME
        shutil.copy2(best_src, best_dst)
        logger.info("carried forward %s", best_src)
    else:
        logger.info(
            "no %s alongside %s -- fine if that run never beat an earlier best.",
            BEST_CHECKPOINT_NAME,
            last_src,
        )
    return last_dst, best_dst


def find_nonfinite_model_tensors(model_state_dict: dict[str, Any]) -> list[str]:
    """Names every floating-point tensor in a model state dict with a NaN/Inf.

    Args:
        model_state_dict: A `nn.Module.state_dict()`-shaped mapping.

    Returns:
        Sorted parameter names with at least one non-finite value. Empty if
        every tensor is finite.
    """
    return sorted(
        name
        for name, tensor in model_state_dict.items()
        if torch.is_tensor(tensor)
        and tensor.is_floating_point()
        and not torch.isfinite(tensor).all()
    )


def find_nonfinite_optimizer_tensors(optimizer_state_dict: dict[str, Any]) -> list[str]:
    """Names every floating-point optimizer-state tensor with a NaN/Inf.

    Args:
        optimizer_state_dict: An `Optimizer.state_dict()`-shaped mapping,
            i.e. `{"state": {param_idx: {key: tensor, ...}, ...}, ...}`.

    Returns:
        Sorted `"optimizer.state[<idx>].<key>"` names with a non-finite
        value. Empty if every tensor is finite.
    """
    bad = []
    for param_idx, state in optimizer_state_dict.get("state", {}).items():
        for key, tensor in state.items():
            if (
                torch.is_tensor(tensor)
                and tensor.is_floating_point()
                and not torch.isfinite(tensor).all()
            ):
                bad.append(f"optimizer.state[{param_idx}].{key}")
    return sorted(bad)


def verify_checkpoint_identity(
    payload: dict[str, Any], expect_epoch: int | None, expect_wandb_id: str | None
) -> None:
    """Checks a loaded checkpoint payload against expected epoch / W&B id.

    A checkpoint from the wrong run loads and trains perfectly; the damage
    only shows up hours later in a run that is not the run intended. These
    checks are the cheap, load-bearing alternative to noticing that late.

    Args:
        payload: A `torch.load`-ed checkpoint payload dict.
        expect_epoch: Required epoch, or None to skip this check.
        expect_wandb_id: Required `wandb_run_id`, or None to skip this check.

    Raises:
        RuntimeError: If either expectation is given and does not match.
    """
    epoch = payload.get("epoch")
    if expect_epoch is not None and epoch != expect_epoch:
        raise RuntimeError(
            f"checkpoint is at epoch {epoch}, expected {expect_epoch}. Wrong session's "
            "output attached, or the previous session did not end where its log says."
        )
    wandb_run_id = payload.get("wandb_run_id")
    if expect_wandb_id is not None and wandb_run_id != expect_wandb_id:
        raise RuntimeError(
            f"checkpoint carries wandb_run_id={wandb_run_id!r}, expected "
            f"{expect_wandb_id!r}. This is a DIFFERENT run -- resuming it would waste "
            "the whole session."
        )


def verify_checkpoint_finite(payload: dict[str, Any]) -> None:
    """Raises if a checkpoint's model or optimizer state carries a NaN/Inf.

    NaN weights train without raising anything (see CLAUDE.md trap #2), so
    the only honest check is to look, once, before spending a session on it.

    Args:
        payload: A `torch.load`-ed checkpoint payload dict.

    Raises:
        RuntimeError: If any model or optimizer tensor is non-finite.
    """
    bad = find_nonfinite_model_tensors(
        payload.get("model_state_dict") or {}
    ) + find_nonfinite_optimizer_tensors(payload.get("optimizer_state_dict") or {})
    if bad:
        raise RuntimeError(
            f"checkpoint contains non-finite values in {len(bad)} tensor(s): {bad[:5]}. "
            "Do NOT resume this checkpoint."
        )


def run_ckpt_preflight(
    checkpoint_dir: Path,
    ckpt_src: Path | None,
    expect_epoch: int | None,
    expect_wandb_id: str | None,
) -> None:
    """Validates the checkpoint this run will resume from, if any.

    Two carry-forward mechanisms both land here. `--ckpt-src` is the Kaggle
    shape: each session gets a fresh, empty `checkpoint_dir`, and a prior
    session's OUTPUT directory is copied in first. In-place resume is the
    college cluster's shape: its disk is persistent, so a pre-empted job's
    own `checkpoint_dir` already holds `last.pt` from an earlier attempt at
    the SAME job -- nothing to copy, the file is already where training will
    read it from. Either way, once a `last.pt` sits in `checkpoint_dir`, the
    SAME identity and finiteness checks apply before training ever starts.

    Args:
        checkpoint_dir: This session's checkpoint directory (from the
            composed config).
        ckpt_src: Prior session's output directory to copy `last.pt`/`best.pt`
            from, or None for in-place resume (or no resume at all).
        expect_epoch: Required epoch, or None to skip.
        expect_wandb_id: Required `wandb_run_id`, or None to skip.

    Raises:
        FileNotFoundError: If `ckpt_src` is given but has no `last.pt`; or if
            `expect_epoch`/`expect_wandb_id` is given but no checkpoint
            exists anywhere (neither `--ckpt-src` nor an in-place `last.pt`)
            to check them against.
        RuntimeError: If `copy_checkpoints` would roll progress back, or if
            the identity or finiteness check fails.
    """
    if ckpt_src is not None:
        copy_checkpoints(ckpt_src, checkpoint_dir)

    last_path = checkpoint_dir / LAST_CHECKPOINT_NAME
    if not last_path.is_file():
        if expect_epoch is not None or expect_wandb_id is not None:
            raise FileNotFoundError(
                f"--expect-epoch/--expect-wandb-id given but no {LAST_CHECKPOINT_NAME} "
                f"exists at {checkpoint_dir} (no --ckpt-src, and no in-place resume "
                "checkpoint there either) -- nothing to check them against."
            )
        logger.info(
            "no %s at %s -- fresh run, nothing to verify.", LAST_CHECKPOINT_NAME, checkpoint_dir
        )
        return

    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    verify_checkpoint_identity(payload, expect_epoch, expect_wandb_id)
    verify_checkpoint_finite(payload)
    logger.info(
        "checkpoint preflight OK (%s): resuming from epoch %s, wandb_run_id=%s, %d "
        "parameter tensors and their optimizer state all finite",
        "copied via --ckpt-src" if ckpt_src is not None else "in-place",
        payload.get("epoch"),
        payload.get("wandb_run_id"),
        len(payload.get("model_state_dict") or {}),
    )


# ---------------------------------------------------------------------------
# Preflight orchestration
# ---------------------------------------------------------------------------


def run_preflight(args: argparse.Namespace) -> DictConfig:
    """Runs every preflight check, in order, and returns the composed config.

    Order matters and mirrors the module docstring: git SHA, then the CUDA
    smoke check (if requested), then data-root resolution, then config
    composition, then checkpoint validation -- the last step needs the
    composed config to know the run's checkpoint directory, and runs
    unconditionally (not just when `--ckpt-src` is given): a checkpoint
    already sitting in that directory (the college cluster's in-place resume
    case) gets exactly the same identity and finiteness checks.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The composed Hydra config, ready for `run_training`.

    Raises:
        RuntimeError: On a git SHA mismatch, a failed CUDA check, a refused
            (rollback) checkpoint copy, or a checkpoint identity/finiteness
            failure.
        FileNotFoundError: On a data-root or checkpoint-source problem, or an
            `--expect-epoch`/`--expect-wandb-id` with no checkpoint anywhere
            to check it against.
    """
    check_git_sha(_REPO_DIR, args.expect_sha)

    if args.require_cuda:
        run_cuda_smoke_check()

    data_root = resolve_data_root(args.data_root, args.search_root)
    logger.info("resolved data root: %s", data_root)

    cfg = compose_config(args, data_root)

    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    run_ckpt_preflight(
        checkpoint_dir=checkpoint_dir,
        ckpt_src=Path(args.ckpt_src) if args.ckpt_src is not None else None,
        expect_epoch=args.expect_epoch,
        expect_wandb_id=args.expect_wandb_id,
    )

    return cfg


# ---------------------------------------------------------------------------
# Post-training verification (notebook cell 15)
# ---------------------------------------------------------------------------


def log_peak_vram() -> None:
    """Logs peak CUDA memory usage for this process, or a no-op note on CPU.

    `max_memory_allocated` / `max_memory_reserved` are process-lifetime peaks,
    so this is accurate wherever it is called after training, in the same
    process -- there is nothing to "estimate" here, only to report.
    """
    if not torch.cuda.is_available():
        logger.info("peak VRAM: n/a (no CUDA device in this process)")
        return
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    logger.info("peak VRAM: %.2f GiB allocated, %.2f GiB reserved", allocated, reserved)


def scan_final_checkpoint(checkpoint_dir: Path) -> tuple[int, list[str]]:
    """Loads the just-written `last.pt` and scans it for non-finite values.

    This alone is NOT sufficient to detect CLAUDE.md trap #2: under AMP,
    `GradScaler` silently skips any step whose gradients are inf/NaN, so the
    WEIGHTS can stay perfectly finite for an entire run whose LOSS was NaN
    throughout (the 10.5 GPU-h loss that trap describes). See
    `find_nonfinite_metrics`, which checks the other half -- the metrics
    `run_training` returns -- and must be combined with this scan's result
    before deciding OK vs NAN.

    Args:
        checkpoint_dir: This session's checkpoint directory.

    Returns:
        `(epoch, nonfinite)` -- the checkpoint's saved epoch, and the sorted
        names of every non-finite model or optimizer tensor (empty if none).

    Raises:
        FileNotFoundError: If `checkpoint_dir` has no `last.pt` -- training
            should always have written one; see checkpoint.py's module
            docstring on atomic writes.
    """
    last_path = find_resume_checkpoint(checkpoint_dir)
    if last_path is None:
        raise FileNotFoundError(
            f"{checkpoint_dir / LAST_CHECKPOINT_NAME} missing after training -- nothing "
            "to carry into the next session."
        )
    payload = torch.load(last_path, map_location="cpu", weights_only=True)
    nonfinite = find_nonfinite_model_tensors(
        payload.get("model_state_dict") or {}
    ) + find_nonfinite_optimizer_tensors(payload.get("optimizer_state_dict") or {})
    epoch = payload.get("epoch", -1)
    return epoch, sorted(nonfinite)


def find_nonfinite_metrics(metrics: dict[str, Any]) -> list[str]:
    """Names every non-finite float entry in a training metrics dict.

    This is `kaggle_train.ipynb` cell 15's exact rule (`_nan = [k for k, v
    in metrics.items() if isinstance(v, float) and not math.isfinite(v)]`),
    and it is the check CLAUDE.md trap #2 actually depends on: under AMP,
    `GradScaler` silently skips any step whose gradients are inf/NaN, so a
    checkpoint's WEIGHTS can stay finite for an entire run whose LOSS was
    NaN throughout -- 10.5 GPU-h were lost training on exactly that, with
    nothing raising. Scanning only the checkpoint's tensors (see
    `scan_final_checkpoint`) would print OK on that failure; the returned
    metrics are the only place it is visible.

    Args:
        metrics: The dict `run_training` returns.

    Returns:
        Sorted metric names with a non-finite float value. Empty if none.
    """
    return sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, float) and not math.isfinite(value)
    )


def read_best_epoch(checkpoint_dir: Path) -> int | None:
    """Reads `best.pt`'s saved epoch, mirroring `kaggle_train.ipynb` cell 15's `_be`.

    `best.pt` is written only when validation improves (see
    `save_checkpoint`), so a final session whose validation never beat an
    earlier best legitimately has none -- this is reported, never raised on.

    Args:
        checkpoint_dir: This session's checkpoint directory.

    Returns:
        The epoch saved in `best.pt`, or None if it does not exist.
    """
    best_path = checkpoint_dir / BEST_CHECKPOINT_NAME
    if not best_path.is_file():
        return None
    payload = torch.load(best_path, map_location="cpu", weights_only=True)
    return payload.get("epoch")


def print_health_line(
    epoch: int,
    best_epoch: int | None,
    loss: float | None,
    grad_norm_median: float | None,
    nonfinite: list[str],
) -> str:
    """Prints the one grep-able health line a resume watcher reads.

    Matches `kaggle_train.ipynb` cell 15's exact line format, so an existing
    log-watching script keeps working unchanged. Uses `print`, not
    `logging`, deliberately: this line must be found by a simple substring
    search in a raw session log regardless of what logging handlers are (or
    are not) configured.

    Args:
        epoch: The final checkpoint's saved epoch.
        best_epoch: `best.pt`'s saved epoch, or None if it does not exist.
        loss: `metrics.get("train/loss_epoch")`, or None if absent.
        grad_norm_median: `metrics.get("train/grad_norm_median")`, or None
            if absent.
        nonfinite: Names of non-finite tensors AND/OR metrics found (the
            union `find_nonfinite_model_tensors` / `find_nonfinite_optimizer_tensors`
            / `find_nonfinite_metrics` produce), or an empty list.

    Returns:
        The exact line printed, for tests to assert against.
    """
    status = "NAN" if nonfinite else "OK"
    line = (
        f"NVX_HEALTH: {status} | epoch={epoch} | best_epoch={best_epoch} | "
        f"loss={loss} | grad_norm_median={grad_norm_median} | nonfinite={nonfinite}"
    )
    print(line)
    return line


def load_run_training() -> Callable[[DictConfig], dict[str, float]]:
    """Loads `scripts/train.py::run_training` without importing `scripts` as a package.

    `scripts/` has no `__init__.py` (it holds CLI entry points, not library
    code), so this uses the same `importlib.util.spec_from_file_location`
    pattern as `scripts/smoke_test.py` and `tests/script_loader.py`, rather
    than reimplementing training here. This script never edits `train.py`;
    it only calls its public `run_training`.

    Returns:
        The `run_training` function from `scripts/train.py`.
    """
    spec = importlib.util.spec_from_file_location("train_script", _TRAIN_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = sys.modules.get("train_script")
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_script"] = module
        spec.loader.exec_module(module)
    return module.run_training


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Parses args, runs preflight, trains (unless --dry-run), and verifies.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        Process exit code: 0 on success (or a clean `--dry-run`), 1 if
        preflight failed or the final checkpoint has a non-finite value.
    """
    args = parse_args(argv)
    setup_logging(level="INFO")

    try:
        cfg = run_preflight(args)
    except Exception:
        logger.exception("Preflight failed -- exiting before training.")
        return 1

    if args.dry_run:
        logger.info("--dry-run: preflight passed, exiting without training.")
        return 0

    run_training = load_run_training()
    metrics = run_training(cfg)
    logger.info("run_training returned: %s", metrics)

    log_peak_vram()
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    epoch, bad_tensors = scan_final_checkpoint(checkpoint_dir)
    bad_metrics = find_nonfinite_metrics(metrics)
    # UNION, not just the tensor scan: a NaN loss with perfectly finite
    # weights is exactly CLAUDE.md trap #2 (AMP's GradScaler skips the step
    # instead of applying NaN gradients), and it must still print NAN.
    nonfinite = sorted(set(bad_tensors) | set(bad_metrics))
    best_epoch = read_best_epoch(checkpoint_dir)
    print_health_line(
        epoch,
        best_epoch,
        metrics.get("train/loss_epoch"),
        metrics.get("train/grad_norm_median"),
        nonfinite,
    )

    return 1 if nonfinite else 0


if __name__ == "__main__":
    sys.exit(main())
