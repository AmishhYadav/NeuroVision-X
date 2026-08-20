"""Hydra entry point: an inference-time intervention on the fusion gate's ambiguity input.

## Why this exists

`docs/research/contribution.md` records prediction P2 -- "the ambiguity conditioning is
necessary, not decorative" -- as **NOT RUN**. The registered way to test it is a full
retraining ablation (`configs/experiment/ablation_content_only_gate.yaml`, ~23 GPU-hours),
which is not currently available.

This script answers a strictly WEAKER, zero-GPU question instead, on an ALREADY TRAINED
checkpoint: does the fusion gate actually use its ambiguity input at inference, or would it
behave identically without it? For each selected case it runs the model FOUR times, swapping
in a different `AdaptiveGatedFusion.ambiguity_transform` hook
(`neurovision.models.fusion.adaptive_fusion`) before each pass:

- `baseline` -- the hook left unset (`None`). The trained model exactly as it is.
- `zero`     -- the ambiguity tensor replaced with zeros before it reaches the gate
                (`zero_ambiguity`).
- `mean`     -- each channel collapsed to its own per-case spatial mean, keeping overall
                magnitude but destroying spatial pattern (`mean_ambiguity`).
- `shuffle`  -- each channel's values randomly permuted in space: same marginal distribution
                as `baseline`, destroyed spatial correspondence (`shuffle_ambiguity`).

For every (case, condition) pair it records how much the fusion gate maps diverge from
`baseline`'s, at every fusion level, and what happens to segmentation Dice/IoU/HD95.

## THIS IS NOT A SUBSTITUTE FOR THE `ablation_content_only_gate` RETRAINING ABLATION

A null result here (the gate barely moves under intervention, Dice barely changes) is WEAKER
evidence than a null from the retraining ablation would be. A checkpoint trained WITH the
ambiguity signal available could have needed it during training to reach the weights it has,
while having learned to mostly ignore it once trained -- "was the signal load-bearing for
getting here" and "is the signal load-bearing right now" are different questions, and only the
retraining ablation answers the first. What this script CAN show is whether the ambiguity
channels are load-bearing for the gate's behaviour and the final segmentation IN THE TRAINED
MODEL AS IT STANDS -- a real, falsifiable, zero-GPU data point, just not the pre-registered
one. State this caveat alongside any number this script produces.

## Config -- no new block, no new file under configs/

Reads the EXISTING `explainability.ambiguity` block: `split`, `checkpoint`, `num_cases`,
`case_ids`, `out_dir` -- plus `inference.sliding_window` / `inference.postprocess` / `data.*`
/ the root `seed`. `explainability.ambiguity.level` is NOT read here: gate divergence is
computed at EVERY fusion level in one pass (see `_GatesAndLogits` below), not one chosen
level, unlike `scripts/extract_ambiguity.py`. `out_dir` gets `_intervention` appended, so a
run here can never overwrite an extraction run's output in the same directory.

`explainability.ambiguity.num_cases: null` means something DIFFERENT here than it does for
`scripts/extract_ambiguity.py`: that script treats `null` as "the whole split", because it
costs one sliding-window pass per case. This script costs FOUR (one per condition), so `null`
here defaults to `_DEFAULT_NUM_CASES` (40) instead, and logs loudly that it is doing so. Set
`explainability.ambiguity.num_cases` explicitly on the CLI to run a different number, or to
the split's full size to run the whole thing.

Two further knobs are Python constants below (`_SPEARMAN_SUBSAMPLE_SIZE`,
`_BOOTSTRAP_N`) rather than config keys, because this script must not add a config block or
file. Edit the constants directly if you need a different value.

Example usage:

    python scripts/ambiguity_intervention.py explainability.ambiguity.split=test
    python scripts/ambiguity_intervention.py explainability.ambiguity.num_cases=10
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import list_data_collate
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.analysis.statistics import paired_bootstrap_ci
from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms
from neurovision.inference.postprocess import postprocess_logits
from neurovision.inference.sliding_window import sliding_window_predict
from neurovision.metrics.segmentation import compute_case_metrics

# Importing this registers "unet3d"/"swinunetr" (from baseline.py) AND "neurovision" (models/
# __init__.py imports neurovision.py too) before build_model is ever called below. Copied from
# scripts/evaluate.py / scripts/extract_ambiguity.py.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.fusion.adaptive_fusion import (
    AdaptiveGatedFusion,
    mean_ambiguity,
    shuffle_ambiguity,
    zero_ambiguity,
)
from neurovision.models.registry import build_model
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on any machine --
# no absolute paths. Copied from scripts/evaluate.py / scripts/extract_ambiguity.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# See the module docstring's "Config" section for why this differs from
# scripts/extract_ambiguity.py's "null -> whole split" convention.
_DEFAULT_NUM_CASES = 40

# The four conditions, in the fixed order every per-case pass runs them. "baseline" MUST come
# first: run_intervention reuses its pass as the reference every other condition's gate
# divergence is measured against, and resets every ambiguity_transform to it before intervening.
_CONDITIONS: tuple[str, ...] = ("baseline", "zero", "mean", "shuffle")

# Number of (channel, voxel) positions to draw for the Spearman correlation between an
# intervened gate map and baseline's. A full-volume rank correlation sorts several million
# values per level per case for no real gain in precision -- Spearman's rank statistic is
# stable under subsampling, and this keeps the correlation cheap enough to compute at every
# level for every case.
_SPEARMAN_SUBSAMPLE_SIZE = 2000

# Bootstrap replicate count for the paired Dice-difference CI in intervention_summary.csv --
# neurovision.analysis.statistics.paired_bootstrap_ci's own default.
_BOOTSTRAP_N = 10000
_BOOTSTRAP_CI = 0.95


class _GatesAndLogits(nn.Module):
    """Wraps a model so ONE sliding-window pass yields deterministic seg logits AND every
    fusion level's gate map, channel-concatenated into one tensor.

    Follows the same single-output-wrapper idea `scripts/extract_ambiguity.py`'s
    `_AmbiguityAtLevel` uses for `NeuroVisionX.forward_with_ambiguity` -- MONAI's
    `SlidingWindowInferer` only knows how to stitch one tensor per call, and
    `forward_with_gates` returns a pyramid. Unlike `_AmbiguityAtLevel` (which selects ONE
    chosen level), this wrapper carries EVERY fusion level's gate at once: gate divergence is
    reported per level here, and `forward_with_gates` already computes every level's gate in a
    single forward call, so packing them all into one pass costs nothing extra and keeps the
    total cost at exactly one sliding-window pass per condition (see the module docstring).

    `channel_groups`, `(name, size)` pairs in concatenation order -- `("logits", num_regions)`
    followed by one `("gate_<i>", gate_channels)` entry per fusion level that actually returns
    a gate (a `ConcatFusion`/`AddFusion` block contributes `None` and is skipped) -- is
    recorded on the FIRST forward call of a sliding-window run and reused for every later call
    in the SAME run (window shape varies call to call; the channel layout, fixed by the
    model's own fusion-block configuration, does not). `run_intervention`'s slicing reads the
    layout from this attribute rather than recomputing offsets independently, so the two
    cannot silently drift apart -- the same reasoning `_AmbiguityAtLevel.channel_groups`
    documents.
    """

    def __init__(self, model: nn.Module) -> None:
        """Wraps `model`.

        Args:
            model: A `NeuroVisionX` instance (or anything exposing
                `forward_with_gates(x) -> (logits, gates)` with the same contract).
        """
        super().__init__()
        self.model = model
        self.channel_groups: tuple[tuple[str, int], ...] | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Runs `forward_with_gates` and channel-concatenates logits with every real gate map.

        Args:
            x: Input MRI volume/patch, shape `(B, in_channels, D, H, W)`.

        Returns:
            `(B, total_channels, D, H, W)`, `total_channels` the sum of `self.channel_groups`'
            sizes.

        Raises:
            ValueError: If the set of levels reporting a real (non-`None`) gate changes
                between calls within one sliding-window run -- this would mean the model's own
                fusion-block configuration is not fixed, which should be impossible for a
                loaded checkpoint.
        """
        target_shape = tuple(x.shape[2:])
        logits, gates = self.model.forward_with_gates(x)

        pieces: list[Tensor] = [logits]
        groups: list[tuple[str, int]] = [("logits", logits.shape[1])]
        for i, gate in enumerate(gates):
            if gate is None:
                continue
            if tuple(gate.shape[2:]) != target_shape:
                gate = F.interpolate(gate, size=target_shape, mode="trilinear", align_corners=False)
            pieces.append(gate)
            groups.append((f"gate_{i}", gate.shape[1]))

        groups_t = tuple(groups)
        if self.channel_groups is None:
            self.channel_groups = groups_t
        elif self.channel_groups != groups_t:
            raise ValueError(
                "The gate channel layout changed between sliding-window calls on the same "
                f"model: first call produced {self.channel_groups}, this call produced "
                f"{groups_t}. This should be impossible for a fixed, already-loaded model."
            )
        return torch.cat(pieces, dim=1)


def _split_channel_groups(
    tensor: Tensor, channel_groups: Sequence[tuple[str, int]]
) -> dict[str, Tensor]:
    """Slices `_GatesAndLogits`' channel-concatenated output back into named groups.

    Args:
        tensor: Shape `(B, total_channels, D, H, W)`.
        channel_groups: `_GatesAndLogits.channel_groups` -- `(name, size)` pairs in the SAME
            order the wrapper concatenated them. Slicing from this attribute (rather than
            recomputing offsets independently here) is what keeps the two from ever
            disagreeing -- see `_GatesAndLogits`'s class docstring.

    Returns:
        A dict from group name (`"logits"`, `"gate_0"`, `"gate_1"`, ...) to that group's
        channel slice, each shape `(B, size, D, H, W)`.
    """
    out: dict[str, Tensor] = {}
    offset = 0
    for name, size in channel_groups:
        out[name] = tensor[:, offset : offset + size]
        offset += size
    return out


def select_intervention_cases(cfg: DictConfig) -> list[str]:
    """Chooses which cases to run the ambiguity intervention over.

    Same convention as `scripts/extract_ambiguity.py`'s `select_cases`, EXCEPT for the
    `num_cases: null` default -- see the module docstring's "Config" section for why.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Case ids, in the order they should be processed:
        `explainability.ambiguity.case_ids` if set (validated against the split), else the
        first N ids of the split in split-file order, where N is
        `explainability.ambiguity.num_cases` if set, else `_DEFAULT_NUM_CASES`.

    Raises:
        ValueError: If `explainability.ambiguity.split` is not a key of the frozen splits
            file, if an explicit `case_ids` entry is not a member of that split (names the
            offending ids and the split), or if the resulting selection is empty.
    """
    amb_cfg = cfg.explainability.ambiguity
    splits = load_splits(cfg.data.splits.path)

    split = amb_cfg.split
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}. Available splits: {sorted(splits.keys())}.")
    available_ids = list(splits[split])

    if amb_cfg.case_ids is not None:
        case_ids = list(amb_cfg.case_ids)
        missing = [c for c in case_ids if c not in available_ids]
        if missing:
            raise ValueError(
                f"explainability.ambiguity.case_ids names case(s) not present in split "
                f"{split!r}: {missing}. That split has {len(available_ids)} case(s)."
            )
    else:
        num_cases = amb_cfg.num_cases
        if num_cases is None:
            logger.warning(
                "explainability.ambiguity.num_cases is null -- ambiguity_intervention.py "
                "defaults that to %d case(s) here, NOT the whole %d-case split=%r, because "
                "this script runs FOUR sliding-window passes per case (one per condition), "
                "four times the cost of a plain evaluation run. Set "
                "explainability.ambiguity.num_cases explicitly to run a different number.",
                _DEFAULT_NUM_CASES,
                len(available_ids),
                split,
            )
            num_cases = _DEFAULT_NUM_CASES
        case_ids = available_ids[:num_cases]

    if not case_ids:
        raise ValueError(
            f"select_intervention_cases selected 0 cases from split {split!r} "
            f"(case_ids={amb_cfg.case_ids}, num_cases={amb_cfg.num_cases})."
        )
    return case_ids


def _validate_labeled_cases(case_ids: Sequence[str], prep_dir: Path) -> None:
    """Checks every selected case has a `label.npy` on disk.

    Segmentation effect needs ground truth to score Dice/IoU/HD95 against, and
    `build_intervention_dataloader` reuses the shared `build_val_transforms` pipeline, whose
    `LoadImaged` is built without `allow_missing_keys=True` -- a case with no `label.npy`
    would otherwise die inside MONAI's loader with a message several frames from anything
    that names the case.

    Args:
        case_ids: Cases selected by `select_intervention_cases`.
        prep_dir: Root of the preprocessed data.

    Raises:
        ValueError: If any selected case has no `label.npy` on disk. Names every offending
            case id.
    """
    missing = [case_id for case_id in case_ids if not (prep_dir / case_id / "label.npy").is_file()]
    if missing:
        raise ValueError(
            f"The following selected case(s) have no label.npy on disk: {missing}. This "
            "script scores Dice/IoU/HD95 under every condition, which needs ground truth. "
            "Select cases from a labelled split."
        )


def resolve_intervention_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to load for the ambiguity intervention.

    Same convention as `scripts/evaluate.py`'s `resolve_checkpoint` /
    `scripts/extract_ambiguity.py`'s `resolve_ambiguity_checkpoint`, reading from
    `cfg.explainability.ambiguity.checkpoint`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.explainability.ambiguity.checkpoint)` if set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message lists whatever
            `.pt` files ARE present in that directory.
    """
    explicit = cfg.explainability.ambiguity.checkpoint
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


def load_intervention_model(
    cfg: DictConfig, checkpoint_path: Path, device: torch.device
) -> nn.Module:
    """Builds the model from config and loads a checkpoint's weights into it.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by
            `resolve_intervention_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        The loaded model, in eval mode.
    """
    model = build_model(cfg)
    model = model.to(device)
    # restore_rng=False: this is deterministic inference tooling, with no reason to perturb
    # the process's RNG state -- same reasoning as scripts/evaluate.py / extract_ambiguity.py.
    load_checkpoint(checkpoint_path, model, map_location=str(device), restore_rng=False)
    model.eval()
    return model


def _find_ambiguity_blocks(model: nn.Module, cfg: DictConfig) -> list[AdaptiveGatedFusion]:
    """Finds every fusion block whose gate the intervention can act on.

    Args:
        model: The loaded model.
        cfg: The full composed Hydra config, used only to name the offending model in error
            messages.

    Returns:
        Every `AdaptiveGatedFusion` block in `model.fusion_blocks` built with
        `use_ambiguity=True` (i.e. `block.ambiguity is not None`) -- the only blocks that have
        an `ambiguity_transform` hook and an ambiguity signal for it to intervene on.

    Raises:
        TypeError: If `model` has no `forward_with_gates` method. Only
            `neurovision.models.neurovision.NeuroVisionX` has fusion blocks with gate maps to
            extract -- evaluating a `unet3d`/`swinunetr` checkpoint here is a configuration
            mistake worth catching in the first second, not on the first forward pass.
        ValueError: If `model` has `forward_with_gates` but no fusion block has a real
            ambiguity mechanism to intervene on -- e.g. every block was built with
            `use_ambiguity=False` (the content-only ablation checkpoint), or a non-adaptive
            fusion variant (`concat`/`add`) was used throughout, or the Swin branch is
            disabled entirely. There is nothing for this script to measure against such a
            checkpoint.
    """
    if not hasattr(model, "forward_with_gates"):
        raise TypeError(
            f"model.name={cfg.model.name!r} has no forward_with_gates method. Only the "
            "'neurovision' model has fusion blocks with gate maps -- point cfg.model.name "
            "(and cfg.explainability.ambiguity.checkpoint, if set) at a 'neurovision' "
            "checkpoint instead."
        )

    fusion_blocks = getattr(model, "fusion_blocks", [])
    ambiguity_blocks = [
        block
        for block in fusion_blocks
        if isinstance(block, AdaptiveGatedFusion) and block.ambiguity is not None
    ]
    if not ambiguity_blocks:
        raise ValueError(
            f"model.name={cfg.model.name!r} has no fusion block built with "
            "use_ambiguity=True -- there is no ambiguity signal for this script to "
            "intervene on. This checkpoint is either the content-only ablation "
            "(model.fusion.use_ambiguity=false), was trained with a non-adaptive fusion "
            "variant (concat/add), or has the Swin branch disabled entirely."
        )
    return ambiguity_blocks


def build_intervention_dataloader(cfg: DictConfig, case_ids: Sequence[str]) -> DataLoader:
    """Builds a whole-volume `DataLoader` for the selected cases.

    Uses the same deterministic val transform pipeline as `scripts/evaluate.py` /
    `scripts/extract_ambiguity.py` (no cropping, no randomness): sliding-window inference does
    the patching itself, so the dataset must hand back whole volumes.

    Args:
        cfg: The full composed Hydra config.
        case_ids: Case ids to load, in the order they should be yielded.

    Returns:
        A `DataLoader`, `batch_size=1`, yielding batches in the SAME order as `case_ids`.
    """
    prep_dir = cfg.data.preprocessing.out_dir
    data_dicts = build_data_dicts(list(case_ids), prep_dir)
    transform = build_val_transforms(cfg)
    dataset = build_dataset(data_dicts, transform, dataset_type="dataset")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
    )


def _gate_divergence(
    gate_cond: Tensor,
    gate_baseline: Tensor,
    rng: np.random.Generator,
    subsample_size: int = _SPEARMAN_SUBSAMPLE_SIZE,
) -> tuple[float, float]:
    """Mean absolute difference and Spearman rank correlation between two gate maps.

    For `condition == "baseline"` compared against itself this is exactly `(0.0, 1.0)` by
    construction -- the two tensors are identical, so their difference is zero everywhere and
    their ranks agree perfectly.

    Args:
        gate_cond: One condition's gate map at one fusion level, shape `(1, C, D, H, W)`.
        gate_baseline: `baseline`'s gate map at the SAME level, same shape.
        rng: A seeded `np.random.Generator` drawing the voxel subsample the correlation is
            computed over (CLAUDE.md: randomness only through an explicitly seeded
            generator). Required, no default and no use of the global RNG.
        subsample_size: Number of `(channel, voxel)` positions to draw, capped at the
            tensor's actual element count.

    Returns:
        `(mean_abs_diff, spearman_rho)`. `spearman_rho` is `nan` when scipy cannot compute a
        rank correlation (e.g. every sampled value on both sides is identical).
    """
    mean_abs_diff = (gate_cond - gate_baseline).abs().mean().item()

    flat_cond = gate_cond.reshape(-1).cpu().numpy()
    flat_base = gate_baseline.reshape(-1).cpu().numpy()
    n = flat_cond.size
    size = min(subsample_size, n)
    idx = rng.choice(n, size=size, replace=False)
    rho, _pvalue = stats.spearmanr(flat_cond[idx], flat_base[idx])
    return mean_abs_diff, float(rho)


def _log_and_print_summary(summary_df: pd.DataFrame, split: str, n_cases: int) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        summary_df: `intervention_summary.csv`'s DataFrame, indexed by `condition`.
        split: The split name cases were drawn from.
        n_cases: Number of cases processed.
    """
    lines = [
        "=" * 70,
        f"Ambiguity intervention summary -- split={split!r}, {n_cases} case(s)",
        "REMINDER: this is an inference-time intervention on an already-trained checkpoint, "
        "NOT a substitute for the ablation_content_only_gate retraining ablation. See this "
        "script's module docstring.",
        "=" * 70,
    ]
    for condition in _CONDITIONS:
        if condition not in summary_df.index:
            continue
        row = summary_df.loc[condition]
        lines.append(
            f"  {condition:8s}  dice_mean={row.get('dice_mean_mean', float('nan')):.4f}  "
            f"diff_vs_baseline={row.get('dice_mean_diff_vs_baseline', float('nan')):+.4f} "
            f"[{row.get('dice_mean_diff_lo', float('nan')):+.4f}, "
            f"{row.get('dice_mean_diff_hi', float('nan')):+.4f}]"
        )
    # print only, not logger.info as well -- setup_logging's StreamHandler already targets
    # stdout, so doing both would print this block twice. Matches evaluate.py /
    # extract_ambiguity.py.
    print("\n".join(lines))


def run_intervention(cfg: DictConfig) -> pd.DataFrame:
    """Runs the ambiguity intervention over a split and writes every result to disk.

    For each selected case: run the deterministic `baseline` sliding-window pass once (hook
    unset on every ambiguity-carrying fusion block), then one further pass per remaining
    condition (`zero`, `mean`, `shuffle`) with that condition's `ambiguity_transform` set on
    every such block -- exactly four sliding-window passes per case in total. Every pass
    reuses the SAME `_GatesAndLogits` wrapper, so segmentation logits and every fusion level's
    gate map both come from one forward call per condition; no separate pass is needed to get
    the deterministic logits for scoring Dice.

    See the module docstring for what the four conditions are and why this is NOT a
    substitute for the `ablation_content_only_gate` retraining ablation.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `intervention_per_case.csv`'s DataFrame (also written to disk), one row per
        `(case_id, condition)` pair.

    Raises:
        ValueError: See `select_intervention_cases`, `_validate_labeled_cases`, and
            `_find_ambiguity_blocks`.
        FileNotFoundError: See `resolve_intervention_checkpoint`.
        TypeError: See `_find_ambiguity_blocks`.
    """
    device = get_device(cfg)
    amb_cfg = cfg.explainability.ambiguity
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    case_ids = select_intervention_cases(cfg)
    _validate_labeled_cases(case_ids, prep_dir)

    checkpoint_path = resolve_intervention_checkpoint(cfg)
    model = load_intervention_model(cfg, checkpoint_path, device)
    # Checked before any output directory exists -- a misconfigured checkpoint should fail in
    # the first second, not after several minutes of sliding-window inference.
    ambiguity_blocks = _find_ambiguity_blocks(model, cfg)

    wrapped_model = _GatesAndLogits(model).to(device)
    wrapped_model.eval()

    # _intervention suffix: never overwrite scripts/extract_ambiguity.py's own out_dir, even
    # when both scripts are pointed at the same checkpoint/split.
    out_dir = ensure_dir(f"{amb_cfg.out_dir}_intervention")
    per_case_csv_path = out_dir / "intervention_per_case.csv"
    summary_csv_path = out_dir / "intervention_summary.csv"

    loader = build_intervention_dataloader(cfg, case_ids)

    # One CPU generator for shuffle_ambiguity (CLAUDE.md: no global-RNG randomness), one NumPy
    # generator for the gate-divergence voxel subsample, one for the bootstrap CI -- all seeded
    # from the run's own seed so the whole run is reproducible from cfg.seed alone.
    shuffle_generator = torch.Generator().manual_seed(int(cfg.seed))
    spearman_rng = np.random.default_rng(int(cfg.seed))
    bootstrap_rng = np.random.default_rng(int(cfg.seed))

    transforms: dict[str, Any] = {
        "baseline": None,
        "zero": zero_ambiguity,
        "mean": mean_ambiguity,
        "shuffle": functools.partial(shuffle_ambiguity, generator=shuffle_generator),
    }

    all_rows: list[dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc="Ambiguity intervention")
        for case_id, batch in progress:
            meta = read_json(prep_dir / case_id / "meta.json")
            image = batch["image"]
            label = batch["label"]

            # baseline first, always: every other condition's gate divergence is measured
            # against this pass, and it establishes wrapped_model.channel_groups.
            for block in ambiguity_blocks:
                block.ambiguity_transform = None
            baseline_combined = sliding_window_predict(
                wrapped_model, image, cfg, device, set_eval=True
            )
            baseline_groups = _split_channel_groups(baseline_combined, wrapped_model.channel_groups)
            gate_names = [
                name for name, _ in wrapped_model.channel_groups if name.startswith("gate_")
            ]

            for condition in _CONDITIONS:
                if condition == "baseline":
                    groups = baseline_groups
                else:
                    for block in ambiguity_blocks:
                        block.ambiguity_transform = transforms[condition]
                    combined = sliding_window_predict(
                        wrapped_model, image, cfg, device, set_eval=True
                    )
                    groups = _split_channel_groups(combined, wrapped_model.channel_groups)
                    for block in ambiguity_blocks:
                        block.ambiguity_transform = None

                logits = groups["logits"]
                regions = postprocess_logits(logits, cfg)
                case_metrics = compute_case_metrics(
                    regions.cpu(), label.cpu(), spacing=meta["spacing"]
                )

                row: dict[str, Any] = {"case_id": case_id, "condition": condition, **case_metrics}
                for name in gate_names:
                    level = int(name.split("_")[1])
                    diff, rho = _gate_divergence(
                        groups[name].cpu(), baseline_groups[name].cpu(), spearman_rng
                    )
                    row[f"gate_absdiff_level{level}"] = diff
                    row[f"gate_spearman_level{level}"] = rho
                all_rows.append(row)

            # Rewritten every CASE (all 4 of its condition rows together), not just at the
            # end: a killed run keeps every already-processed case instead of losing all of
            # them. Same reasoning as scripts/evaluate.py's per-case CSV.
            pd.DataFrame(all_rows).to_csv(per_case_csv_path, index=False)

    # Leave the model in a clean, un-intervened state for anything reusing it afterward.
    for block in ambiguity_blocks:
        block.ambiguity_transform = None

    per_case_df = pd.DataFrame(all_rows)
    gate_level_names = [
        name for name, _ in wrapped_model.channel_groups if name.startswith("gate_")
    ]
    gate_levels = sorted(int(name.split("_")[1]) for name in gate_level_names)
    summary_df = _build_summary(per_case_df, gate_levels, bootstrap_rng)
    summary_df.to_csv(summary_csv_path)

    config_path = out_dir / "intervention_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(summary_df, amb_cfg.split, len(case_ids))

    return per_case_df


def _build_summary(
    per_case_df: pd.DataFrame, gate_levels: Sequence[int], bootstrap_rng: np.random.Generator
) -> pd.DataFrame:
    """Reduces `intervention_per_case.csv` to one summary row per condition.

    Args:
        per_case_df: One row per `(case_id, condition)`, as `run_intervention` builds it.
        gate_levels: Fusion-level indices that have a `gate_absdiff_level<i>` /
            `gate_spearman_level<i>` column.
        bootstrap_rng: Seeded `np.random.Generator` for `paired_bootstrap_ci`'s resampling.
            Required, no default (CLAUDE.md: no global-RNG randomness) -- reused across every
            call in this function, so its state advances deterministically call to call.

    Returns:
        One row per condition (`baseline`, `zero`, `mean`, `shuffle`), indexed by `condition`:
        `dice_<region>_mean` (mean absolute Dice under that condition, for each of ET/TC/WT
        and the cross-region `mean`), and `dice_<region>_diff_vs_baseline` /
        `_diff_lo` / `_diff_hi` / `_diff_n` (the PAIRED bootstrap difference against
        `baseline`, from `neurovision.analysis.statistics.paired_bootstrap_ci`, which
        resamples CASE INDICES into the difference array so the pairing survives -- see that
        function's docstring). The `baseline` row's own diff-vs-baseline columns are exactly
        `0.0` / `[0.0, 0.0]` by construction: every per-case difference against itself is
        zero. Also `gate_absdiff_level<i>_mean` / `gate_spearman_level<i>_mean`, the mean over
        cases of each per-case gate-divergence column.
    """
    baseline_df = per_case_df[per_case_df["condition"] == "baseline"].set_index("case_id")
    region_cols = [f"dice_{region}" for region in REGION_NAMES] + ["dice_mean"]

    rows: dict[str, dict[str, float]] = {}
    for condition in _CONDITIONS:
        cond_df = per_case_df[per_case_df["condition"] == condition].set_index("case_id")
        # Same case order as baseline_df -- both are built from the SAME case_ids loop, but
        # aligning explicitly is what makes the pairing correct even if that ever changes.
        aligned = cond_df.loc[baseline_df.index]

        row: dict[str, float] = {}
        for col in region_cols:
            row[f"{col}_mean"] = float(cond_df[col].mean())
            result = paired_bootstrap_ci(
                aligned[col].to_numpy(),
                baseline_df[col].to_numpy(),
                n_boot=_BOOTSTRAP_N,
                ci=_BOOTSTRAP_CI,
                generator=bootstrap_rng,
            )
            row[f"{col}_diff_vs_baseline"] = result.point
            row[f"{col}_diff_lo"] = result.lo
            row[f"{col}_diff_hi"] = result.hi
            row[f"{col}_diff_n"] = float(result.n)

        for level in gate_levels:
            row[f"gate_absdiff_level{level}_mean"] = float(
                cond_df[f"gate_absdiff_level{level}"].mean()
            )
            row[f"gate_spearman_level{level}_mean"] = float(
                cond_df[f"gate_spearman_level{level}"].mean()
            )
        rows[condition] = row

    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("condition")


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the ambiguity intervention over a split, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_intervention(cfg)


if __name__ == "__main__":
    main()
