"""Hydra entry point that saves fusion gate maps for a handful of cases.

`NeuroVisionX.forward_with_gates(x) -> (logits, gates)` (see
`neurovision.models.neurovision`) exists and is tested, but nothing saves its
output. The fusion gate map is the primary evidence for this project's
central claim -- that the adaptive gated fusion module actually fires, and
fires where it should (near tumor margins), rather than being a decorative
extra parameter block. This script is that producer.

Design, decided up front, not re-derived here:

- **Patch-based, not sliding-window.** `forward_with_gates` returns gates at
  four different strides (2/4/8/16), one per fusion block, and MONAI's
  sliding-window inferer only knows how to stitch ONE output. Reimplementing
  four-level stitching is out of scope for what this produces -- a
  qualitative figure plus a per-case correlation over a few dozen cases, not
  a full-split artifact. So this extracts exactly ONE `patch_size` crop per
  case, centred on the tumor, and runs it through `forward_with_gates` in a
  single forward pass (two, under `center_on="prediction"` -- see
  `run_extraction`'s docstring).
- **Runs on CPU.** This is intended for the author's Mac, not Kaggle -- see
  `neurovision.utils.device.get_device`, resolved once from
  `cfg.device` exactly like every other entry point in this project.

Mirrors `scripts/evaluate.py`'s decomposition into small, individually
testable functions (`select_cases`, `tumor_centroid`, `crop_patch`,
`extract_case_gates`, `run_extraction`) and its Hydra/`_CONFIG_DIR` idiom --
see tests/test_extract_gates.py.

Example usage:

    python scripts/extract_gates.py explainability.gates.split=test
    python scripts/extract_gates.py explainability.gates.num_cases=8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms

# Importing this registers "unet3d"/"swinunetr" (from baseline.py) AND
# "neurovision" (models/__init__.py imports neurovision.py too -- see that
# file's own comment) before build_model is ever called below. Copied from
# scripts/evaluate.py.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Fixed lookup, not a re-declared constant -- REGION_NAMES is the single
# source of truth for channel order (see neurovision.data.transforms).
_WT_INDEX = REGION_NAMES.index("WT")

_VALID_CENTER_ON = ("label", "prediction")


def select_cases(cfg: DictConfig) -> list[str]:
    """Chooses which cases to extract gate maps for.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Case ids, in the order they should be processed:
        `cfg.explainability.gates.case_ids` if set (validated against the
        split), else the first `cfg.explainability.gates.num_cases` ids of
        the split in split-file order (`null` -> every case).

    Raises:
        ValueError: If `cfg.explainability.gates.split` is not a key of the
            frozen splits file, if an explicit `case_ids` entry is not a
            member of that split (names the offending ids and the split), or
            if the resulting selection is empty.
    """
    gates_cfg = cfg.explainability.gates
    splits = load_splits(cfg.data.splits.path)

    split = gates_cfg.split
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")
    available_ids = list(splits[split])

    if gates_cfg.case_ids is not None:
        case_ids = list(gates_cfg.case_ids)
        missing = [c for c in case_ids if c not in available_ids]
        if missing:
            raise ValueError(
                f"explainability.gates.case_ids names case(s) not present in split {split!r}: "
                f"{missing}. That split has {len(available_ids)} case(s)."
            )
    else:
        num_cases = gates_cfg.num_cases
        case_ids = available_ids if num_cases is None else available_ids[:num_cases]

    if not case_ids:
        raise ValueError(
            f"select_cases selected 0 cases from split {split!r} "
            f"(case_ids={gates_cfg.case_ids}, num_cases={gates_cfg.num_cases})."
        )
    return case_ids


def tumor_centroid(label: Tensor, region_index: int, case_id: str | None = None) -> tuple[
    int, int, int
]:
    """Finds the centroid of one region's nonzero voxels.

    Args:
        label: Binary region tensor, shape `(C, D, H, W)`.
        region_index: Which channel to use. Callers pass `_WT_INDEX` for a
            real label, or `0` after wrapping a single predicted mask in a
            size-1 channel axis (see `run_extraction`'s `center_on ==
            "prediction"` branch, which reuses this function for exactly
            that reason).
        case_id: Case identifier, used only to name the case in the fallback
            warning below. `None` is accepted so this function has no hard
            dependency on a caller always having one.

    Returns:
        `(d, h, w)`, the rounded centroid of `label[region_index]`'s nonzero
        voxels, or the volume's geometric centre if that channel is entirely
        empty.

    Raises:
        ValueError: If `label` is not 4-dimensional.
    """
    if label.ndim != 4:
        raise ValueError(f"tumor_centroid expects a (C, D, H, W) tensor, got shape {tuple(label.shape)}.")

    channel = label[region_index]
    nonzero = torch.nonzero(channel > 0, as_tuple=False)  # (N, 3)

    if nonzero.shape[0] == 0:
        center = tuple(int(s // 2) for s in channel.shape)
        logger.warning(
            "tumor_centroid: region index %d is entirely empty for case %r; falling back to "
            "the volume's geometric centre %s.",
            region_index,
            case_id,
            center,
        )
        return center

    centroid = nonzero.to(torch.float32).mean(dim=0)
    return tuple(int(round(v.item())) for v in centroid)


def crop_patch(
    volume: Tensor, center: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[Tensor, tuple[int, int, int]]:
    """Extracts a `patch_size` crop centred on `center` from a whole volume.

    The crop window is clamped to the volume bounds -- shifted, never
    shrunk -- so the returned patch is always exactly `patch_size` on every
    axis whose extent is at least `patch_size`. An axis SHORTER than
    `patch_size` is cropped in full and zero-padded at the end to reach it
    (both encoders in the full model downsample 5 times, and an axis under
    32 voxels reaches a degenerate bottleneck).

    Args:
        volume: `(C, D, H, W)` tensor to crop from.
        center: `(d, h, w)` voxel coordinates the crop is centred on, in
            `volume`'s own coordinate frame.
        patch_size: `(D, H, W)` size of the returned patch.

    Returns:
        `(patch, origin)`: `patch` has shape `(C, *patch_size)`; `origin` is
        the `(d0, h0, w0)` voxel `patch[:, 0, 0, 0]` (before any padding)
        corresponds to in `volume`'s coordinate frame, so the crop can be
        located back in the source volume later.

    Raises:
        ValueError: If `volume` is not 4-dimensional.
    """
    if volume.ndim != 4:
        raise ValueError(f"crop_patch expects a (C, D, H, W) tensor, got shape {tuple(volume.shape)}.")

    spatial_shape = tuple(volume.shape[1:])

    origin: list[int] = []
    crop_lens: list[int] = []
    pad_after: list[int] = []
    for axis_len, size, ctr in zip(spatial_shape, patch_size, center):
        if axis_len >= size:
            start = int(ctr) - size // 2
            # Shift the window to stay in bounds -- never shrink it.
            start = max(0, min(start, axis_len - size))
            origin.append(start)
            crop_lens.append(size)
            pad_after.append(0)
        else:
            # The whole (short) axis is kept, then padded at the end.
            origin.append(0)
            crop_lens.append(axis_len)
            pad_after.append(size - axis_len)

    d0, h0, w0 = origin
    dl, hl, wl = crop_lens
    cropped = volume[:, d0 : d0 + dl, h0 : h0 + hl, w0 : w0 + wl]

    pad_d, pad_h, pad_w = pad_after
    if pad_d or pad_h or pad_w:
        # F.pad's argument order is reversed (last spatial dim first) and
        # each axis takes a (before, after) pair -- padding only "after"
        # matches this function's contract of never moving the origin.
        cropped = F.pad(cropped, (0, pad_w, 0, pad_h, 0, pad_d))

    return cropped, (d0, h0, w0)


def _check_forward_with_gates(model: nn.Module, cfg: DictConfig) -> None:
    """Raises before any output is produced if `model` has no gate maps to give.

    Args:
        model: The loaded model.
        cfg: The full composed Hydra config, used only to name the
            offending model in the error message.

    Raises:
        TypeError: If `model` has no `forward_with_gates` method. Only
            `neurovision.models.neurovision.NeuroVisionX` has fusion blocks
            -- evaluating a `unet3d` or `swinunetr` checkpoint here is a
            configuration mistake worth catching in the first second, not on
            the first forward pass.
    """
    if not hasattr(model, "forward_with_gates"):
        raise TypeError(
            f"model.name={cfg.model.name!r} has no forward_with_gates method. Only the "
            "'neurovision' model has fusion blocks with gate maps to extract -- point "
            "cfg.model.name (and cfg.explainability.gates.checkpoint, if set) at a "
            "'neurovision' checkpoint instead."
        )


def extract_case_gates(model: nn.Module, patch: Tensor) -> tuple[Tensor, list[Tensor | None]]:
    """Runs `model.forward_with_gates` on one patch and drops the batch axis.

    Args:
        model: A model exposing `forward_with_gates` (see
            `_check_forward_with_gates`), already on the correct device and
            in eval mode.
        patch: One MRI patch, shape `(C, D, H, W)`, already on the model's
            device.

    Returns:
        `(logits, gates)`:

        - `logits`: full-resolution segmentation logits, shape
          `(out_channels, D, H, W)`.
        - `gates`: fine-to-coarse, one entry per fusion block (see
          `NeuroVisionX.forward_with_gates`'s docstring for the full
          contract). `None` entries are real (a fusion variant with no gate
          concept); the list itself is empty when the model has no fusion
          blocks at all. Each present entry has its batch axis dropped:
          shape `(1, D_i, H_i, W_i)` or `(C_i, D_i, H_i, W_i)`.
    """
    with torch.no_grad():
        logits, gates = model.forward_with_gates(patch.unsqueeze(0))
    logits_no_batch = logits[0]
    gates_no_batch: list[Tensor | None] = [g[0] if g is not None else None for g in gates]
    return logits_no_batch, gates_no_batch


def resolve_gates_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to load for gate extraction.

    Same convention as `scripts/evaluate.py`'s `resolve_checkpoint`, reading
    from `cfg.explainability.gates.checkpoint` instead of
    `cfg.inference.evaluation.checkpoint`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.explainability.gates.checkpoint)` if set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message
            lists whatever `.pt` files ARE present in that directory.
    """
    explicit = cfg.explainability.gates.checkpoint
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


def load_gates_model(cfg: DictConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Builds the model from config and loads a checkpoint's weights into it.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by
            `resolve_gates_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        The loaded model, in eval mode.
    """
    model = build_model(cfg)
    model = model.to(device)
    # restore_rng=False: extraction is deterministic inference, with no
    # reason to perturb the process's RNG state -- same reasoning as
    # scripts/evaluate.py's load_eval_model.
    load_checkpoint(checkpoint_path, model, map_location=str(device), restore_rng=False)
    model.eval()
    return model


def build_gates_dataloader(cfg: DictConfig, case_ids: list[str]) -> DataLoader:
    """Builds a whole-volume `DataLoader` for the selected cases.

    Uses the same deterministic val transform pipeline as
    `scripts/evaluate.py` (no cropping, no randomness) -- this script does
    its own single tumor-centred crop afterward via `crop_patch`, so the
    dataset must hand back whole volumes to crop from.

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


def _validate_center_on(cfg: DictConfig, case_ids: list[str], prep_dir: Path) -> None:
    """Validates `cfg.explainability.gates.center_on` against the selected cases.

    Called before the checkpoint is loaded or any output is written -- a
    data/config mismatch here should fail in the first second, not after
    minutes of forward passes.

    Args:
        cfg: The full composed Hydra config.
        case_ids: Cases selected by `select_cases`.
        prep_dir: Root of the preprocessed data (for reading each case's
            `meta.json`).

    Raises:
        ValueError: If `center_on` is not one of `_VALID_CENTER_ON`, or if
            it is `"label"` and any selected case has `meta["has_label"] ==
            False` (names the offending case ids).
    """
    center_on = cfg.explainability.gates.center_on
    if center_on not in _VALID_CENTER_ON:
        raise ValueError(
            f"explainability.gates.center_on must be one of {_VALID_CENTER_ON}, got "
            f"{center_on!r}."
        )

    if center_on == "label":
        unlabeled = [
            case_id
            for case_id in case_ids
            if not read_json(prep_dir / case_id / "meta.json")["has_label"]
        ]
        if unlabeled:
            raise ValueError(
                "explainability.gates.center_on='label' but the following selected case(s) "
                f"have no ground-truth label (meta['has_label'] is False): {unlabeled}. The "
                "BraTS validation set ships without segmentations -- use "
                "center_on='prediction' for such cases."
            )


def _log_and_print_summary(manifest_df: pd.DataFrame, split: str, n_cases: int) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        manifest_df: The manifest DataFrame `run_extraction` is about to
            return / has just written.
        split: The split name gate maps were extracted from.
        n_cases: Number of cases processed.
    """
    lines = [
        "=" * 70,
        f"Gate extraction summary -- split={split!r}, {n_cases} case(s)",
        "=" * 70,
    ]
    if manifest_df.empty:
        lines.append("No cases were processed.")
    else:
        n_with_gates = int((manifest_df["n_gate_levels"] > 0).sum())
        n_wt_empty = int(manifest_df["wt_empty"].sum())
        n_levels = int(manifest_df["n_levels"].iloc[0])
        lines.append(f"  fusion levels configured: {n_levels}")
        lines.append(f"  cases with at least one saved gate level: {n_with_gates}/{n_cases}")
        lines.append(
            f"  cases where the WT centroid fell back to the geometric centre: "
            f"{n_wt_empty}/{n_cases}"
        )

    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/evaluate.py's _log_and_print_summary.
    print("\n".join(lines))


def run_extraction(cfg: DictConfig) -> pd.DataFrame:
    """Extracts fusion gate maps for a handful of cases and writes them to disk.

    For each selected case: load the whole preprocessed volume, find a crop
    centre (the ground-truth WT centroid under `center_on="label"`, or the
    model's own predicted WT centroid under `center_on="prediction"` -- see
    below), crop one `patch_size` patch, run it through
    `model.forward_with_gates`, and write one compressed `.npz` per case
    plus one manifest row.

    Under `center_on="prediction"` two ordinary patch forward passes run per
    case, never a sliding-window stitch: an initial patch centred at the
    volume's geometric centre locates the tumor (its predicted WT centroid,
    translated back into the volume's coordinate frame), and a second patch
    centred there is what actually gets saved.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The manifest DataFrame (also written to
        `<out_dir>/gates_manifest.csv`), indexed by `case_id`.

    Raises:
        ValueError: See `select_cases` and `_validate_center_on`.
        FileNotFoundError: See `resolve_gates_checkpoint`.
        TypeError: See `_check_forward_with_gates`.
    """
    device = get_device(cfg)
    gates_cfg = cfg.explainability.gates
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    case_ids = select_cases(cfg)
    _validate_center_on(cfg, case_ids, prep_dir)

    checkpoint_path = resolve_gates_checkpoint(cfg)
    model = load_gates_model(cfg, checkpoint_path, device)
    # Checked once, before the loop and before any output directory exists --
    # evaluating a unet3d/swinunetr checkpoint here is a user error worth
    # catching in the first second, not on the first forward pass.
    _check_forward_with_gates(model, cfg)

    out_dir = ensure_dir(gates_cfg.out_dir)
    patch_size = tuple(int(s) for s in gates_cfg.patch_size)

    loader = build_gates_dataloader(cfg, case_ids)
    manifest_rows: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc="Extracting gates")
        for case_id, batch in progress:
            meta = read_json(prep_dir / case_id / "meta.json")
            image = batch["image"][0].to(device)  # (4, D, H, W)
            label = batch["label"][0]  # (3, D, H, W); real only if meta["has_label"]

            if gates_cfg.center_on == "label":
                wt_empty = not bool(label[_WT_INDEX].any())
                center = tumor_centroid(label, _WT_INDEX, case_id=case_id)
            else:
                # "prediction": locate the tumor from an initial pass at the
                # volume's geometric centre, then re-crop around its
                # predicted WT centroid. Two ordinary patch passes -- see
                # this function's docstring.
                spatial_shape = tuple(image.shape[1:])
                geometric_center = tuple(s // 2 for s in spatial_shape)
                initial_patch, initial_origin = crop_patch(image, geometric_center, patch_size)
                initial_logits, _ = extract_case_gates(model, initial_patch)
                predicted_wt = (torch.sigmoid(initial_logits[_WT_INDEX]) > 0.5).unsqueeze(0)
                wt_empty = not bool(predicted_wt.any())
                local_center = tumor_centroid(predicted_wt, 0, case_id=case_id)
                center = tuple(o + c for o, c in zip(initial_origin, local_center))

            patch, origin = crop_patch(image, center, patch_size)
            logits, gates = extract_case_gates(model, patch)

            save_arrays: dict[str, np.ndarray] = {}
            gate_level_indices: list[int] = []
            for level, gate in enumerate(gates):
                if gate is not None:
                    save_arrays[f"gate_level_{level}"] = gate.cpu().numpy().astype(np.float16)
                    gate_level_indices.append(level)

            if meta["has_label"]:
                label_patch, _ = crop_patch(label, center, patch_size)
                save_arrays["label"] = label_patch.cpu().numpy().astype(np.uint8)

            if gates_cfg.save_image:
                save_arrays["image"] = patch.cpu().numpy().astype(np.float16)

            save_arrays["logits"] = logits.cpu().numpy().astype(np.float16)

            np.savez_compressed(out_dir / f"{case_id}.npz", **save_arrays)

            manifest_rows[case_id] = {
                "center_d": center[0],
                "center_h": center[1],
                "center_w": center[2],
                "origin_d": origin[0],
                "origin_h": origin[1],
                "origin_w": origin[2],
                "patch_d": patch_size[0],
                "patch_h": patch_size[1],
                "patch_w": patch_size[2],
                "n_levels": len(gates),
                "n_gate_levels": len(gate_level_indices),
                "gate_levels": ";".join(str(i) for i in gate_level_indices),
                "has_label": bool(meta["has_label"]),
                "wt_empty": wt_empty,
            }

    manifest_df = pd.DataFrame.from_dict(manifest_rows, orient="index").rename_axis("case_id")
    manifest_df.to_csv(out_dir / "gates_manifest.csv")

    config_path = out_dir / "gates_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(manifest_df, gates_cfg.split, len(case_ids))

    return manifest_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Extracts fusion gate maps for a handful of cases, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_extraction(cfg)


if __name__ == "__main__":
    main()
