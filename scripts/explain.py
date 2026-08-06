"""Hydra entry point that produces attribution maps for a handful of cases.

`src/neurovision/explainability/` (`integrated_gradients.py`, `gradcam.py`,
`attention_rollout.py`, `faithfulness.py`) is fully implemented and unit
tested, but nothing calls it: no entry point writes an attribution map to
disk, so `notebooks/09_paper_figures.ipynb` cannot build the explainability
panel. This script is that producer.

Design, decided up front, not re-derived here:

- **Patch-based, not sliding-window or whole-volume**, for the same memory
  reason `scripts/extract_gates.py` is patch-based -- see that script's
  top-of-file docstring and every explainability module's own cost notes
  (Integrated Gradients needs `n_steps` forward+backward passes of the full
  network PER REGION; Grad-CAM needs a gradient-tracking forward pass;
  `compare_methods` needs `2 * n_points` forward passes per method). This is
  offline Mac tooling that produces a handful of published figures from a
  handful of hand-picked cases, never a full-split artifact.
- **Runs on CPU.** Device is resolved once from `cfg.device` via
  `neurovision.utils.device.get_device`, exactly like every other entry
  point in this project.
- **Everything model-touching happens here.** `notebooks/09_paper_figures.ipynb`
  must stay file-driven and CPU-only-by-default; only arrays and CSVs cross
  that boundary.

Mirrors `scripts/extract_gates.py`'s decomposition into small, individually
testable functions and its Hydra/`_CONFIG_DIR` idiom -- see
tests/test_explain_script.py.

Example usage:

    python scripts/explain.py explainability.attribution.split=test
    python scripts/explain.py explainability.attribution.num_cases=2 \\
        explainability.attribution.grad_cam.target_layer=decoder.stages.1.conv2
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms
from neurovision.explainability.attention_rollout import (
    attention_rollout,
    available_blocks,
    capture_attention,
)
from neurovision.explainability.faithfulness import compare_methods
from neurovision.explainability.gradcam import (
    available_layers,
    center_patch_on_mask,
    grad_cam,
    resolve_layer,
)
from neurovision.explainability.integrated_gradients import MODALITY_NAMES, integrated_gradients

# Importing this registers "unet3d"/"swinunetr" (from baseline.py) AND
# "neurovision" (models/__init__.py imports neurovision.py too -- see that
# file's own comment) before build_model is ever called below. Copied from
# scripts/extract_gates.py / scripts/evaluate.py.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/extract_gates.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Fixed lookup, not a re-declared constant -- REGION_NAMES is the single
# source of truth for channel order (see neurovision.data.transforms).
_WT_INDEX = REGION_NAMES.index("WT")

# The fixed column set for modality_attribution.csv, built from
# MODALITY_NAMES rather than hardcoded a second time. Used both to build
# every real row (see the per-region loop in run_explanation) AND to give
# an empty DataFrame real columns when Integrated Gradients never ran (e.g.
# integrated_gradients.enabled=false) -- an empty pd.DataFrame([]).to_csv()
# writes a genuinely empty (0-column) file that pandas.read_csv cannot read
# back, which would make "disable IG" a config a caller could not recover
# from without special-casing an unreadable file.
_MODALITY_ATTRIBUTION_COLUMNS: list[str] = (
    ["case_id", "region"]
    + [f"attr_{name}" for name in MODALITY_NAMES]
    + [f"attr_signed_{name}" for name in MODALITY_NAMES]
    + [
        "convergence_delta",
        "relative_delta",
        "target_score",
        "baseline_score",
        "n_target_voxels",
        "n_steps",
        "delta_exceeded_tolerance",
    ]
)


def select_cases(cfg: DictConfig) -> list[str]:
    """Chooses which cases to produce attribution maps for.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Case ids, in the order they should be processed:
        `cfg.explainability.attribution.case_ids` if set (validated against
        the split), else the first `cfg.explainability.attribution.num_cases`
        ids of the split in split-file order (`null` -> every case).

    Raises:
        ValueError: If `cfg.explainability.attribution.split` is not a key
            of the frozen splits file, if an explicit `case_ids` entry is
            not a member of that split (names the offending ids and the
            split), or if the resulting selection is empty.
    """
    attr_cfg = cfg.explainability.attribution
    splits = load_splits(cfg.data.splits.path)

    split = attr_cfg.split
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")
    available_ids = list(splits[split])

    if attr_cfg.case_ids is not None:
        case_ids = list(attr_cfg.case_ids)
        missing = [c for c in case_ids if c not in available_ids]
        if missing:
            raise ValueError(
                f"explainability.attribution.case_ids names case(s) not present in split "
                f"{split!r}: {missing}. That split has {len(available_ids)} case(s)."
            )
    else:
        num_cases = attr_cfg.num_cases
        case_ids = available_ids if num_cases is None else available_ids[:num_cases]

    if not case_ids:
        raise ValueError(
            f"select_cases selected 0 cases from split {split!r} "
            f"(case_ids={attr_cfg.case_ids}, num_cases={attr_cfg.num_cases})."
        )
    return case_ids


def _validate_cases(cfg: DictConfig, case_ids: list[str], prep_dir: Path) -> None:
    """Validates the selected cases have usable ground truth, before anything else runs.

    Called before the checkpoint is loaded or any output directory is
    created -- a data mismatch here should fail in the first second, not
    after minutes of Integrated Gradients on the first case.

    Unlike `scripts/extract_gates.py` (which has a `center_on="prediction"`
    escape hatch for unlabeled data), this script unconditionally needs the
    ground truth: it centres the crop on the WT label
    (`center_patch_on_mask`) and needs the label again for
    `compare_methods`' localization metrics.

    Args:
        cfg: The full composed Hydra config.
        case_ids: Cases selected by `select_cases`.
        prep_dir: Root of the preprocessed data (for reading each case's
            `meta.json`).

    Raises:
        ValueError: If any selected case has no `label.npy` on disk, or has
            `meta["has_label"] == False` (names the offending case ids in
            each instance).
    """
    missing_label_file = [
        case_id for case_id in case_ids if not (prep_dir / case_id / "label.npy").is_file()
    ]
    if missing_label_file:
        raise ValueError(
            f"The following selected case(s) have no label.npy on disk: {missing_label_file}. "
            "build_attribution_dataloader reuses build_val_transforms, whose LoadImaged is not "
            "built with allow_missing_keys=True, so such a case cannot be loaded at all -- and "
            "this script additionally needs the ground truth to centre the patch on the tumor "
            "and for compare_methods' localization metrics. Select cases from a labelled split."
        )

    unlabeled = [
        case_id
        for case_id in case_ids
        if not read_json(prep_dir / case_id / "meta.json")["has_label"]
    ]
    if unlabeled:
        raise ValueError(
            "The following selected case(s) have no ground-truth label (meta['has_label'] is "
            f"False): {unlabeled}. Unlike scripts/extract_gates.py's center_on='prediction' "
            "escape hatch, this script always needs the ground-truth WT mask (to centre the "
            "crop) and the label (for compare_methods' localization metrics) -- select cases "
            "from a labelled split instead."
        )


def _validate_regions(cfg: DictConfig) -> None:
    """Validates `cfg.explainability.attribution.regions` against the model's output width.

    Runs before the model or checkpoint is touched: the output channel
    count is already known from config (`model.out_channels` interpolates
    `${data.num_classes}` for every model this project builds), so this is
    a cheap, purely arithmetic check.

    Args:
        cfg: The full composed Hydra config.

    Raises:
        ValueError: If any entry of `cfg.explainability.attribution.regions`
            is outside `[0, cfg.model.out_channels)` (names the offending
            index/indices).
    """
    num_channels = int(cfg.model.out_channels)
    regions = [int(r) for r in cfg.explainability.attribution.regions]
    bad = [r for r in regions if not (0 <= r < num_channels)]
    if bad:
        raise ValueError(
            f"explainability.attribution.regions contains index/indices {bad} outside the "
            f"model's output channel count ({num_channels}). Valid range: "
            f"[0, {num_channels})."
        )


def resolve_attribution_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to load for attribution.

    Same convention as `scripts/evaluate.py`'s `resolve_checkpoint` and
    `scripts/extract_gates.py`'s `resolve_gates_checkpoint`, reading from
    `cfg.explainability.attribution.checkpoint` instead.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.explainability.attribution.checkpoint)` if set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message
            lists whatever `.pt` files ARE present in that directory.
    """
    explicit = cfg.explainability.attribution.checkpoint
    if explicit is not None:
        checkpoint_path = Path(explicit)
    else:
        checkpoint_path = Path(cfg.training.checkpoint.dir) / "best.pt"

    if not checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.is_dir():
            available = sorted(p.name for p in checkpoint_dir.glob("*.pt"))
        else:
            available = []
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path.resolve()}. "
            f".pt files present in {checkpoint_dir.resolve()}: "
            f"{available if available else '(none -- directory is empty or does not exist)'}."
        )
    return checkpoint_path


def load_attribution_model(
    cfg: DictConfig, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Builds the model from config, loads a checkpoint's weights, and puts it in eval mode.

    Every function in `neurovision.explainability` validates
    `model.training is False` and raises rather than flipping the mode
    itself -- this is the ONE place in this script that calls
    `model.eval()`, and nothing downstream ever calls `model.train()`.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by
            `resolve_attribution_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        The loaded model, in eval mode.
    """
    model = build_model(cfg)
    model = model.to(device)
    # restore_rng=False: attribution is deterministic-model inference (the
    # randomness in this script comes from an explicit torch.Generator, per
    # CLAUDE.md), with no reason to perturb the process's RNG state -- same
    # reasoning as scripts/evaluate.py's load_eval_model.
    load_checkpoint(checkpoint_path, model, map_location=str(device), restore_rng=False)
    model.eval()
    return model


def _validate_grad_cam_config(cfg: DictConfig, model: nn.Module) -> None:
    """Validates `cfg.explainability.attribution.grad_cam` against the loaded model.

    Called before any output directory is created -- a Grad-CAM figure
    without its target layer named is not reproducible, so a missing or
    invalid `target_layer` must fail in the first second.

    Args:
        cfg: The full composed Hydra config.
        model: The loaded model.

    Raises:
        ValueError: If `grad_cam.enabled` is True and `target_layer` is
            `null` (lists `available_layers(model)`), or names a module the
            model does not have (`resolve_layer`'s own near-miss error
            propagates unchanged).
    """
    gc_cfg = cfg.explainability.attribution.grad_cam
    if not gc_cfg.enabled:
        return

    if gc_cfg.target_layer is None:
        raise ValueError(
            "explainability.attribution.grad_cam.target_layer is null but grad_cam.enabled is "
            "True. There is no safe default -- the layer used must be reported alongside any "
            "Grad-CAM figure for it to be reproducible. Available layers on this model: "
            f"{available_layers(model)}. See neurovision.explainability.gradcam.grad_cam's "
            "docstring for guidance on which layer to pick."
        )

    # resolve_layer already produces a near-miss suggestion on a typo'd name;
    # let its ValueError propagate rather than wrapping it.
    resolve_layer(model, gc_cfg.target_layer)


def _validate_attention_config(cfg: DictConfig, model: nn.Module) -> None:
    """Validates `cfg.explainability.attribution.attention` against the loaded model.

    Args:
        cfg: The full composed Hydra config.
        model: The loaded model.

    Raises:
        ValueError: If `attention.enabled` is True and `model` has no
            `WindowAttention` submodules -- this usually means the model has
            no Swin branch (e.g. the `unet3d` baseline), and the message
            says so plainly rather than the script silently writing an
            empty attention map.
    """
    attn_cfg = cfg.explainability.attribution.attention
    if not attn_cfg.enabled:
        return

    if not available_blocks(model):
        raise ValueError(
            "explainability.attribution.attention.enabled=True but this model has no "
            "WindowAttention submodules -- this model has no Swin branch (e.g. a unet3d "
            "baseline). Attention rollout is only meaningful for a model with a Swin encoder. "
            "Set explainability.attribution.attention.enabled=false, or point cfg.model at a "
            "model with a Swin branch (e.g. 'neurovision')."
        )


def build_attribution_dataloader(cfg: DictConfig, case_ids: list[str]) -> DataLoader:
    """Builds a whole-volume `DataLoader` for the selected cases.

    Uses the same deterministic val transform pipeline as
    `scripts/evaluate.py` and `scripts/extract_gates.py` (no cropping, no
    randomness) -- this script does its own single tumor-centred crop
    afterward via `center_patch_on_mask`, so the dataset must hand back
    whole volumes to crop from.

    Args:
        cfg: The full composed Hydra config.
        case_ids: Case ids to load, in the order they should be yielded.

    Returns:
        A `DataLoader`, `batch_size=1` (whole volumes have per-case shapes
        and do not collate at any larger batch size), yielding batches in
        the SAME order as `case_ids`.
    """
    prep_dir = cfg.data.preprocessing.out_dir
    data_dicts = build_data_dicts(case_ids, prep_dir)
    transform = build_val_transforms(cfg)
    dataset = build_dataset(data_dicts, transform, dataset_type="dataset")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
    )


def _reduce_attribution_to_voxel_map(attribution: Tensor) -> Tensor:
    """Reduces a per-modality-channel signed attribution to one per-voxel importance map.

    Integrated Gradients returns a SIGNED value per INPUT MODALITY channel at every voxel,
    `(1, C, D, H, W)`. `compare_methods` (via `deletion_curve`/`insertion_curve`) ranks and
    perturbs LOCATIONS, not modality channels -- see `deletion_curve`'s own docstring: it
    replaces the top-ranked voxels "across ALL input channels at those spatial positions,
    since a heatmap ranks LOCATIONS, not individual modality channels". Summing the ABSOLUTE
    value over the channel axis answers "how much did this voxel matter, combining evidence
    from every modality regardless of sign (and regardless of whether it argued for or against
    the region)" -- exactly the quantity a location-ranking curve needs, and the reduction
    `deletion_curve`'s docstring implicitly assumes a caller has already done.

    Args:
        attribution: `(1, C, D, H, W)` or `(C, D, H, W)`.

    Returns:
        `(D, H, W)`.
    """
    attr = attribution
    if attr.ndim == 5:
        attr = attr[0]
    return attr.abs().sum(dim=0)


def _log_and_print_summary(
    modality_df: pd.DataFrame, manifest_df: pd.DataFrame, split: str, n_cases: int
) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        modality_df: The `modality_attribution.csv` DataFrame `run_explanation`
            is about to return / has just written.
        manifest_df: The manifest DataFrame, used to report how many cases
            ran each sub-analysis.
        split: The split name cases were drawn from.
        n_cases: Number of cases processed.
    """
    lines = [
        "=" * 70,
        f"Attribution summary -- split={split!r}, {n_cases} case(s)",
        "=" * 70,
    ]
    if manifest_df.empty:
        lines.append("No cases were processed.")
    else:
        for column in (
            "integrated_gradients_ran",
            "grad_cam_ran",
            "attention_ran",
            "faithfulness_ran",
        ):
            if column in manifest_df.columns:
                lines.append(f"  {column}: {int(manifest_df[column].sum())}/{n_cases} case(s)")
    if not modality_df.empty and "delta_exceeded_tolerance" in modality_df.columns:
        n_exceeded = int(modality_df["delta_exceeded_tolerance"].sum())
        lines.append(
            f"  IG convergence delta exceeded delta_tolerance on {n_exceeded}/"
            f"{len(modality_df)} (case, region) row(s) -- see modality_attribution.csv."
        )

    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/evaluate.py's / scripts/extract_gates.py's
    # _log_and_print_summary.
    print("\n".join(lines))


def run_explanation(cfg: DictConfig) -> pd.DataFrame:
    """Produces attribution maps and faithfulness scores for a handful of cases.

    For each selected case: load the whole preprocessed volume, crop one
    `patch_size` patch centred on the ground-truth WT region
    (`center_patch_on_mask`), then for each region index in
    `cfg.explainability.attribution.regions` optionally run Integrated
    Gradients, Grad-CAM, and (once per case, not per region) attention
    rollout, and optionally compare whichever attribution methods ran via
    `compare_methods`. Everything is written to disk as it goes -- see this
    module's "Outputs" section in the spec this implements, and
    `tests/test_explain_script.py`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The `modality_attribution.csv` DataFrame (also written to
        `<out_dir>/modality_attribution.csv`), one row per (case, region)
        that Integrated Gradients ran for.

    Raises:
        ValueError: See `select_cases`, `_validate_cases`, `_validate_regions`,
            `_validate_grad_cam_config`, `_validate_attention_config`.
        FileNotFoundError: See `resolve_attribution_checkpoint`.
    """
    device = get_device(cfg)
    attr_cfg = cfg.explainability.attribution
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    case_ids = select_cases(cfg)
    _validate_cases(cfg, case_ids, prep_dir)
    _validate_regions(cfg)

    checkpoint_path = resolve_attribution_checkpoint(cfg)
    model = load_attribution_model(cfg, checkpoint_path, device)
    # Checked once, before any output directory exists -- an unusable
    # grad_cam/attention config is a user error worth catching in the first
    # second, not partway through a multi-minute case loop.
    _validate_grad_cam_config(cfg, model)
    _validate_attention_config(cfg, model)

    out_dir = ensure_dir(attr_cfg.out_dir)
    patch_size = tuple(int(s) for s in attr_cfg.patch_size)
    regions = [int(r) for r in attr_cfg.regions]

    ig_cfg = attr_cfg.integrated_gradients
    gc_cfg = attr_cfg.grad_cam
    attn_cfg = attr_cfg.attention
    faith_cfg = attr_cfg.faithfulness

    # A single seeded generator for the whole run, mutated (its state
    # advances) across every draw -- CLAUDE.md: randomness only through an
    # explicitly seeded torch.Generator, never the global RNG.
    generator = torch.Generator().manual_seed(int(attr_cfg.seed))

    loader = build_attribution_dataloader(cfg, case_ids)

    manifest_rows: dict[str, dict[str, Any]] = {}
    modality_rows: list[dict[str, Any]] = []
    faithfulness_frames: list[pd.DataFrame] = []

    manifest_path = out_dir / "attribution_manifest.csv"
    modality_path = out_dir / "modality_attribution.csv"
    faithfulness_path = out_dir / "faithfulness.csv"

    progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc="Explaining")
    for case_id, batch in progress:
        meta = read_json(prep_dir / case_id / "meta.json")
        image = batch["image"][0].to(device)  # (4, D, H, W)
        label = batch["label"][0]  # (3, D, H, W)

        patch, slices = center_patch_on_mask(image, label[_WT_INDEX], patch_size)
        # Moved to `device` explicitly. `patch` is on `device` (it came from
        # `image`, which was), while `label` came straight off the DataLoader
        # on CPU -- and `label_patch` is handed to `compare_methods` as
        # `ground_truth`, where it meets model outputs. On a CPU box the two
        # agree by accident and every test passes, so no local test can catch
        # this; it is pinned by this comment, exactly as the same class of bug
        # is in scripts/evaluate.py.
        label_patch = label[:, slices[0], slices[1], slices[2]].to(device)

        save_arrays: dict[str, np.ndarray] = {"label": label_patch.cpu().numpy().astype(np.uint8)}
        if attr_cfg.save_image:
            save_arrays["image"] = patch.cpu().numpy().astype(np.float16)

        ig_ran = False
        grad_cam_ran = False
        attention_ran = False
        faithfulness_ran = False

        if attn_cfg.enabled:
            capture = capture_attention(model, patch.unsqueeze(0))
            attn_map = attention_rollout(
                capture,
                stage=attn_cfg.stage,
                residual_weight=attn_cfg.residual_weight,
                max_tokens=attn_cfg.max_tokens,
            )
            save_arrays["attention"] = attn_map[0, 0].cpu().numpy().astype(np.float16)
            attention_ran = True

        for region_index in regions:
            attributions_for_faithfulness: dict[str, Tensor] = {}
            native_resolution: dict[str, str] = {}

            if ig_cfg.enabled:
                ig_out = integrated_gradients(
                    model,
                    patch,
                    region_index=region_index,
                    generator=generator,
                    noise_scale=ig_cfg.noise_scale,
                    n_steps=ig_cfg.n_steps,
                    internal_batch_size=ig_cfg.internal_batch_size,
                    delta_tolerance=ig_cfg.delta_tolerance,
                )
                ig_ran = True

                save_arrays[f"ig_region_{region_index}"] = (
                    ig_out.attributions[0].cpu().numpy().astype(np.float16)
                )
                ig_abs = _reduce_attribution_to_voxel_map(ig_out.attributions)
                save_arrays[f"ig_abs_region_{region_index}"] = (
                    ig_abs.cpu().numpy().astype(np.float16)
                )

                exceeded = ig_out.relative_delta > ig_cfg.delta_tolerance
                if exceeded:
                    # integrated_gradients() already logs a WARNING internally;
                    # this one adds the case/region context that log line
                    # cannot know, and delta_exceeded_tolerance below is what
                    # makes it visible from the CSV alone, not just the log.
                    logger.warning(
                        "run_explanation: case=%s region=%d IG relative_delta=%.4f exceeds "
                        "delta_tolerance=%.4f -- see modality_attribution.csv's "
                        "delta_exceeded_tolerance column.",
                        case_id,
                        region_index,
                        ig_out.relative_delta,
                        ig_cfg.delta_tolerance,
                    )

                row: dict[str, Any] = {"case_id": case_id, "region": region_index}
                for name, value in ig_out.modality_attribution.items():
                    row[f"attr_{name}"] = value
                for name, value in ig_out.modality_attribution_signed.items():
                    row[f"attr_signed_{name}"] = value
                row["convergence_delta"] = ig_out.convergence_delta
                row["relative_delta"] = ig_out.relative_delta
                row["target_score"] = ig_out.target_score
                row["baseline_score"] = ig_out.baseline_score
                row["n_target_voxels"] = ig_out.n_target_voxels
                row["n_steps"] = ig_out.n_steps
                row["delta_exceeded_tolerance"] = exceeded
                modality_rows.append(row)

                attributions_for_faithfulness["integrated_gradients"] = ig_abs
                native_resolution["integrated_gradients"] = "voxel"

            if gc_cfg.enabled:
                cam_out = grad_cam(
                    model,
                    patch,
                    target_layer=gc_cfg.target_layer,
                    region_index=region_index,
                    relu=gc_cfg.relu,
                )
                grad_cam_ran = True
                save_arrays[f"cam_region_{region_index}"] = (
                    cam_out.cam[0, 0].cpu().numpy().astype(np.float16)
                )
                logger.info(
                    "case=%s region=%d grad_cam target_layer=%s target_score=%.4f "
                    "n_target_voxels=%d",
                    case_id,
                    region_index,
                    gc_cfg.target_layer,
                    cam_out.target_score,
                    cam_out.n_target_voxels,
                )
                attributions_for_faithfulness["grad_cam"] = cam_out.cam[0, 0]
                native_resolution["grad_cam"] = f"{gc_cfg.target_layer} (upsampled)"

            if faith_cfg.enabled:
                if not attributions_for_faithfulness:
                    logger.warning(
                        "run_explanation: case=%s region=%d faithfulness.enabled=True but no "
                        "attribution method ran (integrated_gradients and grad_cam are both "
                        "disabled); skipping compare_methods for this (case, region).",
                        case_id,
                        region_index,
                    )
                else:
                    df = compare_methods(
                        model,
                        patch,
                        attributions_for_faithfulness,
                        ground_truth=label_patch,
                        region_index=region_index,
                        n_points=faith_cfg.n_points,
                        fill=faith_cfg.fill,
                        generator=generator,
                        native_resolution=native_resolution,
                    )
                    df = df.rename_axis("method").reset_index()
                    df.insert(0, "region", region_index)
                    df.insert(0, "case_id", case_id)
                    faithfulness_frames.append(df)
                    faithfulness_ran = True

        (d0, d1), (h0, h1), (w0, w1) = (
            (s.start, s.stop) for s in (slices[0], slices[1], slices[2])
        )
        manifest_rows[case_id] = {
            "slice_d_start": d0,
            "slice_d_stop": d1,
            "slice_h_start": h0,
            "slice_h_stop": h1,
            "slice_w_start": w0,
            "slice_w_stop": w1,
            "patch_d": patch_size[0],
            "patch_h": patch_size[1],
            "patch_w": patch_size[2],
            "has_label": bool(meta["has_label"]),
            "integrated_gradients_ran": ig_ran,
            "grad_cam_ran": grad_cam_ran,
            "attention_ran": attention_ran,
            "faithfulness_ran": faithfulness_ran,
            "grad_cam_target_layer": gc_cfg.target_layer if gc_cfg.enabled else None,
        }

        np.savez_compressed(out_dir / f"{case_id}.npz", **save_arrays)

        # Rewritten after EVERY case, not only at the end -- this is a long
        # job (see every explainability module's own measured per-case cost)
        # and a killed run must keep the cases it already finished. Same
        # reasoning as scripts/evaluate.py's per-case CSV.
        pd.DataFrame.from_dict(manifest_rows, orient="index").rename_axis("case_id").to_csv(
            manifest_path
        )
        _modality_df_to_write = (
            pd.DataFrame(modality_rows)
            if modality_rows
            else pd.DataFrame(columns=_MODALITY_ATTRIBUTION_COLUMNS)
        )
        _modality_df_to_write.to_csv(modality_path, index=False)
        if faithfulness_frames:
            pd.concat(faithfulness_frames, ignore_index=True).to_csv(faithfulness_path, index=False)
        else:
            pd.DataFrame().to_csv(faithfulness_path, index=False)

    manifest_df = pd.DataFrame.from_dict(manifest_rows, orient="index").rename_axis("case_id")
    modality_df = (
        pd.DataFrame(modality_rows)
        if modality_rows
        else pd.DataFrame(columns=_MODALITY_ATTRIBUTION_COLUMNS)
    )

    config_path = out_dir / "explain_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(modality_df, manifest_df, attr_cfg.split, len(case_ids))

    return modality_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Produces attribution maps for a handful of cases, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_explanation(cfg)


if __name__ == "__main__":
    main()
