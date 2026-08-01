"""End-to-end CPU smoke test for the whole training pipeline.

Runs the REAL pipeline -- real `Dataset`, real MONAI transforms, real
registry-built model and loss, real `Trainer`, real checkpointing, real
sliding-window validation -- against two tiny synthetic preprocessed cases.
Nothing here is mocked or stubbed: this script generates data shaped exactly
like `scripts/preprocess.py` writes it, composes the real Hydra configs, and
calls `scripts.train.run_training(cfg)` directly.

The point is to catch a broken wiring on the Mac's CPU in seconds, instead of
discovering it after burning a rationed Kaggle GPU session. Run this before
every Kaggle submission:

    python scripts/smoke_test.py

Exits 0 on success ("SMOKE TEST PASSED"), 1 on any failure ("SMOKE TEST
FAILED"), including an unexpected exception -- so it doubles as a pre-Kaggle
CI gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import numpy as np
import torch

# Importing these registers "unet3d" (@register_model) before build_model is
# called below, exactly like scripts/train.py does for the same reason.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.io import ensure_dir, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/preprocess.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# scripts/train.py has no __init__.py sibling, so `scripts` is not an
# importable package -- loading it via spec_from_file_location (rather than
# `from scripts.train import run_training`) is the same pattern
# tests/test_train_script.py uses, and it means run_training is called
# exactly as written, not reimplemented here.
_TRAIN_SCRIPT_PATH = Path(__file__).resolve().parent / "train.py"
_train_spec = importlib.util.spec_from_file_location("train_script", _TRAIN_SCRIPT_PATH)
assert _train_spec is not None and _train_spec.loader is not None
_train_module: ModuleType = importlib.util.module_from_spec(_train_spec)
sys.modules.setdefault("train_script", _train_module)
_train_spec.loader.exec_module(_train_module)
run_training = _train_module.run_training

# Two cases is the minimum that lets the frozen-split loader hand both a
# train and a val list without one of them being empty.
CASE_IDS: tuple[str, ...] = ("SMOKE_001", "SMOKE_002")

# 48^3 so a 32^3 patch fits with room for RandCropByPosNegLabeld to move
# around, and so sliding-window validation over the whole volume needs more
# than one window (roi_size=32 on a 48-voxel axis).
VOLUME_SHAPE: tuple[int, int, int] = (48, 48, 48)

SMOKE_SEED = 42


def _build_synthetic_label(shape: tuple[int, int, int]) -> np.ndarray:
    """Builds a deterministic label volume with BraTS-like nested regions.

    Three concentric spheres around the volume center, assigned outer-to-inner
    so the later (smaller) assignment overwrites the earlier (larger) one:
    class 2 (ED) fills the whole outer sphere, class 1 (NCR/NET) overwrites a
    smaller inner sphere, and class 3 (ET) overwrites the smallest, innermost
    sphere. The result is exactly the real BraTS region nesting -- ET (class
    3) is a subset of TC (classes 1 or 3), which is a subset of WT (classes
    1, 2, or 3) -- built geometrically rather than pulled from real data.

    Every class ends up with thousands of voxels. This matters for two
    reasons: `RandCropByPosNegLabeld` needs foreground voxels to sample
    positive crops from and behaves oddly against an all-background label,
    and a label with zero class-3 voxels would give an all-zero ET channel,
    silently skipping the exact ET path `ConvertToRegionsd` exists to get
    right.

    Args:
        shape: `(D, H, W)` volume shape.

    Returns:
        `uint8` array of shape `shape`, values in `{0, 1, 2, 3}`.
    """
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[dist < min_edge * 0.35] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.22] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.10] = 3  # ET, innermost
    return label


def _write_synthetic_case(case_dir: Path, seed: int) -> None:
    """Writes one synthetic case matching exactly what `preprocess_case` writes.

    Args:
        case_dir: `<preprocessed_dir>/<case_id>/`, created if missing.
        seed: Seed for the image RNG, so each case's intensities are
            reproducible but not identical across cases.
    """
    ensure_dir(case_dir)

    # float16, (4, D, H, W), roughly z-scored -- matches preprocess_case's
    # `cropped_image.astype(np.float16)` output (4 stacked, per-channel
    # nonzero-z-scored modalities).
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((4, *VOLUME_SHAPE)).astype(np.float16)
    np.save(case_dir / "image.npy", image)

    # uint8, (D, H, W), values in {0, 1, 2, 3} -- matches
    # `remapped_label.astype(np.uint8)`.
    label = _build_synthetic_label(VOLUME_SHAPE)
    np.save(case_dir / "label.npy", label)


def _write_splits(splits_path: Path, case_ids: tuple[str, ...]) -> None:
    """Writes a splits YAML with both cases in train AND val.

    With only 2 synthetic cases there is no point holding one out purely for
    validation -- and validation needs at least one case to produce metrics
    at all, so both cases go in both lists. `test` is left empty. This
    deliberate overlap is fine here because the smoke test asserts the
    pipeline RUNS end to end, never that the metrics are any good.

    Args:
        splits_path: Destination YAML path.
        case_ids: Case ids to put in both `train` and `val`.
    """
    write_yaml({"train": list(case_ids), "val": list(case_ids), "test": []}, splits_path)


def _compose_config(artifacts_dir: Path) -> Any:
    """Composes the real Hydra config with smoke-test-sized overrides.

    Uses Hydra's programmatic API (not `@hydra.main`) because this script
    picks its own fixed overrides rather than taking them from the CLI.
    Composing through Hydra -- not hand-building a `DictConfig` -- is the
    point: it verifies the real config files compose and their
    interpolations (e.g. `${data.num_classes}`) resolve.

    Args:
        artifacts_dir: Root directory holding the synthetic preprocessed
            data, splits file, and checkpoint output for this run.

    Returns:
        The composed `DictConfig`.
    """
    preprocessed_dir = artifacts_dir / "preprocessed"
    splits_path = artifacts_dir / "splits.yaml"
    checkpoint_dir = artifacts_dir / "checkpoints"

    overrides = [
        # Mandatory (???) in brats.yaml even though preprocessing is already
        # done -- must be supplied for the config to compose at all.
        f"data.root_dir={artifacts_dir}",
        f"data.preprocessing.out_dir={preprocessed_dir}",
        f"data.splits.path={splits_path}",
        "data.patch_size=[32,32,32]",
        "data.samples_per_volume=2",
        "data.num_workers=0",
        "data.dataset_type=dataset",
        "training.batch_size=1",
        "training.epochs=2",
        "training.val_interval=1",
        f"training.checkpoint.dir={checkpoint_dir}",
        "training.checkpoint.save_every_n_epochs=1",
        "training.sliding_window.roi_size=[32,32,32]",
        # Must be <= epochs, or the linear warmup ramp never completes.
        "training.scheduler.warmup_epochs=1",
        "wandb.mode=disabled",
        "device=cpu",
        "seed=42",
        # model=unet3d stays at production channel widths (the default) --
        # the DATA is what should be tiny here, not the architecture, since
        # the point is to exercise the real model.
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def _run_pipeline(artifacts_dir: Path) -> dict[str, float]:
    """Generates synthetic data, composes config, runs training, and asserts.

    Args:
        artifacts_dir: Root directory for all generated artifacts.

    Returns:
        The metrics dict returned by `run_training`.

    Raises:
        AssertionError: If any pipeline guarantee is violated. The message
            always names what was expected.
    """
    logger.info("STAGE 1/5: writing %d synthetic preprocessed case(s)", len(CASE_IDS))
    preprocessed_dir = artifacts_dir / "preprocessed"
    for i, case_id in enumerate(CASE_IDS):
        _write_synthetic_case(preprocessed_dir / case_id, seed=SMOKE_SEED + i)
    _write_splits(artifacts_dir / "splits.yaml", CASE_IDS)
    logger.info("  synthetic data written to %s", preprocessed_dir)

    logger.info("STAGE 2/5: composing the real Hydra config")
    cfg = _compose_config(artifacts_dir)
    logger.info(
        "  config composed: model=%s, patch_size=%s, epochs=%d",
        cfg.model.name,
        list(cfg.data.patch_size),
        cfg.training.epochs,
    )

    logger.info(
        "STAGE 3/5: calling run_training(cfg) -- dataset, model, loss, Trainer, "
        "training loop, sliding-window validation, checkpointing"
    )
    metrics = run_training(cfg)
    logger.info("  run_training returned metrics: %s", metrics)

    logger.info("STAGE 4/5: verifying checkpoints on disk")
    checkpoint_dir = Path(cfg.training.checkpoint.dir)
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"

    if not last_path.is_file():
        raise AssertionError(f"Expected last.pt to exist at {last_path}, but it does not.")
    if not best_path.is_file():
        raise AssertionError(
            f"Expected best.pt to exist at {best_path} (validation ran every epoch, so a "
            "best checkpoint should have been recorded), but it does not."
        )

    # Loading under weights_only=True is the checkpoint safety guarantee
    # itself -- see neurovision.training.checkpoint's module docstring. If
    # this raises, someone added an unsafe object to the payload.
    torch.load(last_path, weights_only=True)

    check_model = build_model(cfg)
    resume_state = load_checkpoint(last_path, check_model)
    if resume_state.start_epoch != 2:
        raise AssertionError(
            f"Expected start_epoch == 2 after {cfg.training.epochs} completed epoch(s) "
            f"(0-indexed: saved epoch 1, resume at 2), got {resume_state.start_epoch}."
        )

    logger.info("STAGE 5/5: verifying returned validation metrics")
    if "val/dice_mean" not in metrics:
        raise AssertionError(
            f"Expected 'val/dice_mean' in run_training's returned metrics, got keys: "
            f"{sorted(metrics)}."
        )
    dice_mean = metrics["val/dice_mean"]
    if not math.isfinite(dice_mean):
        raise AssertionError(f"Expected val/dice_mean to be finite, got {dice_mean}.")
    if not (0.0 <= dice_mean <= 1.0):
        raise AssertionError(f"Expected val/dice_mean in [0, 1], got {dice_mean}.")

    logger.info(
        "All stages exercised: synthetic data written, config composed, dataset built, "
        "model built, training ran (%d epoch(s)), validation ran, checkpoints written "
        "(last.pt + best.pt, weights_only=True-loadable, start_epoch=%d, val/dice_mean=%.4f).",
        cfg.training.epochs,
        resume_state.start_epoch,
        dice_mean,
    )
    return metrics


def main() -> int:
    """Parses CLI flags, runs the smoke test, and returns a process exit code.

    Returns:
        0 if the smoke test passed, 1 if it failed (assertion or exception).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the generated artifacts directory instead of deleting it.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Write artifacts here instead of a temp dir.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    if args.dir is not None:
        artifacts_dir = ensure_dir(args.dir)
    else:
        # mkdtemp (not the TemporaryDirectory context manager) so cleanup is
        # entirely under this script's control below -- a context manager's
        # finalizer would delete the directory regardless of --keep, since
        # it fires on garbage collection, not just on a `with` block exit.
        artifacts_dir = Path(tempfile.mkdtemp(prefix="nvx_smoke_"))

    success = False
    try:
        metrics = _run_pipeline(artifacts_dir)
        logger.info("=" * 70)
        logger.info("Final val/dice_mean: %.4f", metrics["val/dice_mean"])
        logger.info("SMOKE TEST PASSED")
        success = True
    except Exception:
        # logger.exception (inside an except block) includes the traceback
        # automatically, so a broken wiring is diagnosable from the log
        # without needing to reproduce it.
        logger.exception("Smoke test failed.")
        logger.error("SMOKE TEST FAILED")
        success = False
    finally:
        if args.keep:
            logger.info("--keep passed: artifacts retained at %s", artifacts_dir)
        else:
            shutil.rmtree(artifacts_dir, ignore_errors=True)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
