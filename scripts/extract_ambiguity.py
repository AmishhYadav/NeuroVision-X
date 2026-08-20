"""Hydra entry point that extracts fusion-ambiguity maps over a whole split.

`NeuroVisionX.forward_with_ambiguity(x) -> (logits, ambiguity_maps)` (see
`neurovision.models.neurovision`) exists and is tested, but nothing saves its
output. This script is that producer, and it is deliberately NOT built like
`scripts/extract_gates.py`.

## Why this is sliding-window, not patch-based, unlike extract_gates.py

`extract_gates.py` takes exactly one tumor-centred `patch_size` crop per
case, which is correct for a qualitative figure but wrong here: this
script's output (`ambiguity_summary.csv`) feeds per-case failure detection
-- correlating a per-case ambiguity scalar against per-case Dice, boundary
error, etc. -- which needs a scalar computed over the WHOLE predicted
volume, not one hand-centred crop that silently excludes anything outside
it. So this script runs full sliding-window inference
(`neurovision.inference.sliding_window.sliding_window_predict`, driven by
`cfg.inference.sliding_window`, exactly like `scripts/evaluate.py`) via a
thin per-level wrapper (`_AmbiguityAtLevel`) that adapts the model's
multi-scale ambiguity pyramid into the single input-shaped output MONAI's
`SlidingWindowInferer` knows how to stitch.

## Masking convention -- read before touching the reducer

The per-case reducer (`summarize_case_ambiguity`) uses the PREDICTED-
foreground mask (`regions > 0.5`, from the model's own deterministic
prediction), matching `scripts/evaluate.py`'s `uncertainty_summary.csv`
convention EXACTLY -- not the union-of-predicted-and-ground-truth mask that
`neurovision.uncertainty.risk_coverage.case_uncertainty_scalars`
recommends and uses. That function is tested but UNUSED; do not call it and
do not introduce a third masking convention here, or the several
conventions could not be compared against each other, only against
themselves.

The predicted-foreground convention is chosen over the union convention for
one hard reason: this scalar must be computable with **no access to the
ground-truth label at all**. This project has already shipped a bug where a
reporting mask was built from the label
(`union_foreground_mask` in `src/neurovision/uncertainty/calibration.py`,
see CLAUDE.md's "the calibration reporting mask must never be defined using
the ground-truth label" entry) -- it manufactured 41-57% of a reported ECE
and passed 984 tests, because the code did exactly what it said. A per-case
ambiguity scalar that will later be correlated against per-case Dice is the
same shape of hazard: if the mask that defines the scalar were built from
the label, the correlation would partly measure "how much of the label did
we use to build the mask" rather than anything about the model. So
`summarize_case_ambiguity` takes no label argument at all -- not `None` by
convention, but structurally absent from its signature.

Example usage:

    python scripts/extract_ambiguity.py explainability.ambiguity.split=test
    python scripts/extract_ambiguity.py explainability.ambiguity.level=1
"""

from __future__ import annotations

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
from monai.data.utils import dense_patch_slices, fall_back_tuple

try:  # Guarded on purpose -- see _count_sliding_windows.
    from monai.inferers.utils import _get_scan_interval
except ImportError:  # pragma: no cover -- only on a MONAI version that renamed it.
    _get_scan_interval = None  # type: ignore[assignment]
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neurovision.data.dataset import build_data_dicts, build_dataset, load_splits
from neurovision.data.transforms import REGION_NAMES, build_val_transforms
from neurovision.inference.postprocess import postprocess_logits
from neurovision.inference.sliding_window import sliding_window_predict

# Importing this registers "unet3d"/"swinunetr" (from baseline.py) AND
# "neurovision" (models/__init__.py imports neurovision.py too -- see that
# file's own comment) before build_model is ever called below. Copied from
# scripts/evaluate.py / scripts/extract_gates.py.
from neurovision.models import baseline  # noqa: F401
from neurovision.models.neurovision import NeuroVisionX
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

# The channel-group layout of one ambiguity map's dim-1 axis. Duplicated here
# (not imported from adaptive_fusion.py) only as documentation -- the actual
# split below is computed from `num_regions`, never a hardcoded index, so
# this stays correct even if the constant's own definition moves.
_NUM_AMBIGUITY_GROUPS = 3


class _AmbiguityAtLevel(nn.Module):
    """Adapts `NeuroVisionX` to emit exactly one fusion level's ambiguity map.

    MONAI's `SlidingWindowInferer` stitches a single output tensor per call.
    `NeuroVisionX.forward_with_ambiguity` returns a pyramid -- one ambiguity
    map per fusion block, at that block's own stride -- so a single level is
    selected here and trilinearly upsampled back to the input's spatial
    resolution, giving the inferer the one input-shaped output it requires.

    When `include_gates` and/or `include_auxiliary` are True, this wrapper's
    single output is instead the channel-wise concatenation, in this FIXED
    order: the chosen ambiguity level (`3 * num_regions` channels), then --
    if `include_gates` -- one gate channel per fusion block
    (`len(model.fusion_blocks)` channels, e.g. 4 at the production config),
    then -- if `include_auxiliary` -- the confidence head's full-resolution
    logits (`num_regions` channels) and the boundary head's full-resolution
    logits (`num_regions` channels). Every one of these reads the SAME
    encoder/fusion/decoder pass this wrapper runs (see `forward`'s inlined
    encode/fuse/decode below, mirroring `NeuroVisionX.forward_with_ambiguity`
    / `forward_with_gates`'s own bodies), so turning either flag on costs a
    handful of extra lightweight per-block/per-head computations inside a
    sliding-window pass that is already running over the whole split -- not
    a second (or third) whole-split extraction script, which would cost
    hours of CPU on its own.

    `channel_groups` records the exact `(name, size)` layout of that
    concatenation, in order, computed ONCE at construction time -- so
    `run_extraction`'s slicing can read offsets from this attribute instead
    of recomputing them independently, which is what keeps the two from
    ever disagreeing (see CLAUDE.md's "any glob/offset computed twice can
    silently drift" family of lessons).
    """

    def __init__(
        self,
        model: NeuroVisionX,
        level: int,
        include_auxiliary: bool = False,
        include_gates: bool = False,
        num_regions: int = len(REGION_NAMES),
    ) -> None:
        """Wraps `model` to expose one ambiguity level (plus, optionally, the fusion gates
        and/or the auxiliary heads) as its whole output.

        Args:
            model: A `NeuroVisionX` instance (or anything exposing
                `forward_with_ambiguity` with the same contract). When
                `include_gates` or `include_auxiliary` is True, `model` must
                additionally expose the same attribute surface as
                `NeuroVisionX` (`cnn_encoder`, `use_swin`, `swin_encoder`,
                `fusion_blocks`, `decoder`, `heads`) -- checked below.
            level: Which entry of the fine-to-coarse ambiguity pyramid to
                return. Validated lazily, on the first `forward` call (see
                `_select_level`), since the pyramid's length and which
                entries are real ambiguity tensors vs. `None` are properties
                of the model that are cheapest to check against the model's
                own return value rather than duplicated here.
            include_auxiliary: If True, also emit the confidence and
                boundary heads' logits (see the class docstring). Requires
                `model.heads.confidence` and `model.heads.boundary` to both
                be real heads (not `None`) -- checked immediately, so a
                misconfigured run fails before any output directory exists
                rather than after several minutes of sliding-window
                inference.
            include_gates: If True, also emit one channel per fusion block's
                gate map (see the class docstring). Requires
                `model.fusion_blocks` to be non-empty -- checked
                immediately, same reasoning as `include_auxiliary` above.
                Whether each individual block actually HAS a real gate to
                give (as opposed to a `ConcatFusion`/`AddFusion` block, which
                reports `None`) can only be observed by running the block, so
                that check happens per call inside `forward` instead -- see
                its comment there.
            num_regions: Number of region channels (ET/TC/WT -> 3). Used
                only to size `channel_groups` -- the ambiguity map's own
                channel count is still read from the model at `forward`
                time, never assumed from this value.

        Raises:
            ValueError: If `include_auxiliary` is True and `model` has no
                `heads` attribute, or `model.heads.confidence` is `None`, or
                `model.heads.boundary` is `None`. Also if `include_gates` is
                True and `model.fusion_blocks` is empty (or `model` has no
                such attribute at all). Each case is checked independently
                so the message names exactly what is missing.
        """
        super().__init__()
        self.model = model
        self.level = level
        self.include_auxiliary = include_auxiliary
        self.include_gates = include_gates
        self.num_regions = num_regions

        groups: list[tuple[str, int]] = [("ambiguity", 3 * num_regions)]

        if include_gates:
            fusion_blocks = getattr(model, "fusion_blocks", None)
            num_fusion_levels = len(fusion_blocks) if fusion_blocks is not None else 0
            if num_fusion_levels == 0:
                raise ValueError(
                    "explainability.ambiguity.include_gates is True but the loaded model has "
                    "no fusion blocks (model.fusion_blocks is empty, or the model has no such "
                    "attribute at all -- likely the cnn-only ablation, or a non-neurovision "
                    "checkpoint). Set explainability.ambiguity.include_gates: false, or point "
                    "this run at a checkpoint with a Swin branch and fusion blocks."
                )
            groups.append(("gate", num_fusion_levels))

        if include_auxiliary:
            heads = getattr(model, "heads", None)

            if heads is None or getattr(heads, "confidence", None) is None:
                raise ValueError(
                    "explainability.ambiguity.include_auxiliary is True but the loaded "
                    "model has no confidence head (model.heads.confidence is None, or the "
                    "model has no 'heads' attribute at all). Set "
                    "explainability.ambiguity.include_auxiliary: false, or point this run "
                    "at a checkpoint whose model.head.confidence.enabled was True at "
                    "training time."
                )
            groups.append(("confidence", num_regions))

            if getattr(heads, "boundary", None) is None:
                raise ValueError(
                    "explainability.ambiguity.include_auxiliary is True but the loaded "
                    "model has no boundary head (model.heads.boundary is None). Set "
                    "explainability.ambiguity.include_auxiliary: false, or point this run "
                    "at a checkpoint whose model.head.boundary.enabled was True at "
                    "training time."
                )
            groups.append(("boundary", num_regions))

        # Public and fixed at construction so run_extraction's slicing can never disagree
        # with what forward() actually concatenates -- see the class docstring.
        self.channel_groups: tuple[tuple[str, int], ...] = tuple(groups)

    def _select_level(
        self, ambiguity_maps: list[Tensor | None], target_shape: tuple[int, ...]
    ) -> Tensor:
        """Picks `self.level` out of a fine-to-coarse ambiguity pyramid and upsamples it.

        Args:
            ambiguity_maps: `NeuroVisionX.forward_with_ambiguity`'s second return value (or
                the equivalent collected inline in `forward`'s combined-pass branch).
            target_shape: `(D, H, W)` to upsample to if the chosen level is coarser.

        Returns:
            The chosen level's ambiguity map, shape `(B, 3 * num_regions, *target_shape)`.

        Raises:
            ValueError: If `self.level` is out of range, or that level's entry is `None`
                (the fusion block there has no ambiguity signal to give). Names the level and
                how many maps exist in either case -- never silently falls back to a
                different level.
        """
        if not (0 <= self.level < len(ambiguity_maps)):
            raise ValueError(
                f"_AmbiguityAtLevel: level={self.level} is out of range for a model whose "
                f"forward_with_ambiguity returns {len(ambiguity_maps)} map(s) (valid range "
                f"0..{len(ambiguity_maps) - 1})."
            )

        ambiguity = ambiguity_maps[self.level]
        if ambiguity is None:
            raise ValueError(
                f"_AmbiguityAtLevel: level={self.level} has no ambiguity map (that fusion "
                "block was built with use_ambiguity=False, or is a fusion variant such as "
                f"ConcatFusion/AddFusion with no ambiguity concept). {len(ambiguity_maps)} "
                "map(s) total; choose a level whose entry is a real tensor."
            )

        if tuple(ambiguity.shape[2:]) != target_shape:
            ambiguity = F.interpolate(
                ambiguity, size=target_shape, mode="trilinear", align_corners=False
            )
        return ambiguity

    def forward(self, x: Tensor) -> Tensor:
        """Returns the chosen ambiguity level, optionally concatenated with the fusion
        gates and/or the auxiliary heads' logits.

        Args:
            x: Input MRI volume/patch, shape `(B, in_channels, D, H, W)`.

        Returns:
            `(B, total_channels, D, H, W)`, `total_channels` the sum of `self.channel_groups`'
            sizes -- `3 * num_regions` when both flags are False, up to
            `3 * num_regions + len(fusion_blocks) + 2 * num_regions` when both are True.

        Raises:
            ValueError: See `_select_level`. Also raised (naming
                `explainability.ambiguity.include_gates`) if `include_gates` is True and a
                fusion block reports `None` for its gate at call time, or a gate map has more
                than the single channel this wrapper's fixed layout assumes.
        """
        target_shape = tuple(x.shape[2:])

        if not self.include_gates and not self.include_auxiliary:
            _logits, ambiguity_maps = self.model.forward_with_ambiguity(x)
            return self._select_level(ambiguity_maps, target_shape)

        # include_gates and/or include_auxiliary: __init__ already confirmed the model has
        # the attribute surface (and, for include_gates, the non-empty fusion_blocks; for
        # include_auxiliary, both auxiliary heads) this needs. Walk the encode -> fuse ->
        # decode pyramid ONCE here (mirroring NeuroVisionX.forward_with_ambiguity /
        # forward_with_gates's own bodies) so every requested group reads from the SAME
        # cnn/swin/decode pass rather than a separate full sliding-window pass per group.
        cnn_pyramid = self.model.cnn_encoder(x)
        if self.model.use_swin:
            swin_pyramid = self.model.swin_encoder(x)
            skips = [cnn_pyramid[0]]
            ambiguity_maps: list[Tensor | None] = []
            gate_maps: list[Tensor | None] = []
            for i, block in enumerate(self.model.fusion_blocks):
                if self.include_gates and hasattr(block, "_fuse"):
                    # One call, four outputs. `_fuse` is the single place that
                    # computes fused/gate/branch_logits/ambiguity together; both
                    # `forward(return_gate=True)` and `forward_with_ambiguity`
                    # are thin wrappers that each call it and discard what they
                    # do not return. Going through the two public wrappers
                    # instead would run this block's windowed cross-attention
                    # TWICE per sliding-window position -- measurable on a
                    # whole-split CPU extraction, which is the only thing this
                    # wrapper is ever used for. Reaching for the private name
                    # is deliberate and is guarded by hasattr, because
                    # ConcatFusion/AddFusion have no `_fuse` (and no gate
                    # either, so the branch below raises for them anyway).
                    fused, gate, _branch_logits, ambiguity = block._fuse(
                        cnn_pyramid[i + 1], swin_pyramid[i]
                    )
                    gate_maps.append(gate)
                elif self.include_gates:
                    fused, gate = block(cnn_pyramid[i + 1], swin_pyramid[i], return_gate=True)
                    gate_maps.append(gate)
                    _fused2, ambiguity = block.forward_with_ambiguity(
                        cnn_pyramid[i + 1], swin_pyramid[i]
                    )
                else:
                    fused, ambiguity = block.forward_with_ambiguity(
                        cnn_pyramid[i + 1], swin_pyramid[i]
                    )
                skips.append(fused)
                ambiguity_maps.append(ambiguity)
        else:
            skips = cnn_pyramid
            ambiguity_maps = []
            gate_maps = []

        # Gate validation runs BEFORE selecting the ambiguity level: a fusion variant with
        # no gate concept (ConcatFusion/AddFusion) also has no ambiguity concept, and
        # _select_level's error names the wrong thing (a missing ambiguity map, not a
        # missing gate) if it runs first -- checking include_gates's own precondition first
        # keeps the raised message pointing at the flag that is actually misconfigured.
        gate_group: Tensor | None = None
        if self.include_gates:
            upsampled_gates: list[Tensor] = []
            for i, gate in enumerate(gate_maps):
                if gate is None:
                    raise ValueError(
                        f"explainability.ambiguity.include_gates is True but fusion block "
                        f"{i} reports no gate map (return_gate=True gave None -- this fusion "
                        "variant, e.g. ConcatFusion or AddFusion, has no gating concept to "
                        "report). Set explainability.ambiguity.include_gates: false, or point "
                        "this run at a checkpoint built with model.fusion.name: "
                        "adaptive_gated at every fused level."
                    )
                if gate.shape[1] != 1:
                    raise ValueError(
                        f"explainability.ambiguity.include_gates is True but fusion block "
                        f"{i}'s gate map has {gate.shape[1]} channels, not the single "
                        "(scalar-gate) channel this wrapper's fixed channel layout assumes. "
                        "Set model.fusion.gate_channels: 'scalar' for this checkpoint, or set "
                        "explainability.ambiguity.include_gates: false."
                    )
                if tuple(gate.shape[2:]) != target_shape:
                    gate = F.interpolate(
                        gate, size=target_shape, mode="trilinear", align_corners=False
                    )
                upsampled_gates.append(gate)
            gate_group = torch.cat(upsampled_gates, dim=1)

        ambiguity = self._select_level(ambiguity_maps, target_shape)
        parts: list[Tensor] = [ambiguity]

        if gate_group is not None:
            parts.append(gate_group)

        if self.include_auxiliary:
            feats = self.model.decoder(skips)
            confidence_logits = self.model.heads.confidence(feats[0])
            boundary_logits = self.model.heads.boundary(feats[0])
            if tuple(confidence_logits.shape[2:]) != target_shape:
                confidence_logits = F.interpolate(
                    confidence_logits, size=target_shape, mode="trilinear", align_corners=False
                )
            if tuple(boundary_logits.shape[2:]) != target_shape:
                boundary_logits = F.interpolate(
                    boundary_logits, size=target_shape, mode="trilinear", align_corners=False
                )
            parts.append(confidence_logits)
            parts.append(boundary_logits)

        return torch.cat(parts, dim=1)


def select_cases(cfg: DictConfig) -> list[str]:
    """Chooses which cases to extract ambiguity maps for.

    Same convention as `scripts/extract_gates.py`'s `select_cases`, reading
    from `cfg.explainability.ambiguity` instead of `cfg.explainability.gates`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Case ids, in the order they should be processed:
        `cfg.explainability.ambiguity.case_ids` if set (validated against
        the split), else the first `cfg.explainability.ambiguity.num_cases`
        ids of the split in split-file order (`null` -> every case).

    Raises:
        ValueError: If `cfg.explainability.ambiguity.split` is not a key of
            the frozen splits file, if an explicit `case_ids` entry is not a
            member of that split (names the offending ids and the split), or
            if the resulting selection is empty.
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
        case_ids = available_ids if num_cases is None else available_ids[:num_cases]

    if not case_ids:
        raise ValueError(
            f"select_cases selected 0 cases from split {split!r} "
            f"(case_ids={amb_cfg.case_ids}, num_cases={amb_cfg.num_cases})."
        )
    return case_ids


def _validate_labeled_cases(case_ids: Sequence[str], prep_dir: Path) -> None:
    """Checks every selected case has a `label.npy` on disk.

    `build_ambiguity_dataloader` reuses the shared `build_val_transforms`
    pipeline, whose `LoadImaged` is built without `allow_missing_keys=True`
    (see that function's docstring) -- a case with no `label.npy` therefore
    dies inside MONAI's loader with a message several frames from anything
    that names the case. Checked up front so that failure is immediate and
    legible instead.

    Args:
        case_ids: Cases selected by `select_cases`.
        prep_dir: Root of the preprocessed data.

    Raises:
        ValueError: If any selected case has no `label.npy` on disk. Names
            every offending case id.
    """
    missing = [case_id for case_id in case_ids if not (prep_dir / case_id / "label.npy").is_file()]
    if missing:
        raise ValueError(
            f"The following selected case(s) have no label.npy on disk: {missing}. "
            "build_ambiguity_dataloader reuses build_val_transforms, whose LoadImaged is not "
            "built with allow_missing_keys=True, so such a case cannot be loaded at all. "
            "Select cases from a labelled split. Note the reported summary itself never reads "
            "the label -- this check exists only because the shared loading transform needs "
            "the file to be present."
        )


def _validate_logits_dir(logits_dir: Path, case_ids: Sequence[str]) -> None:
    """Checks a reused logits directory exists and covers every selected case.

    Called once, before the extraction loop starts, so a misconfigured
    `explainability.ambiguity.logits_dir` fails immediately instead of after
    several minutes of sliding-window inference on the ambiguity pass alone.

    Args:
        logits_dir: The directory `run_extraction` will read
            `<case_id>.npy` files from instead of running a second
            sliding-window pass through the plain model.
        case_ids: Cases selected by `select_cases`, in processing order.

    Raises:
        FileNotFoundError: If `logits_dir` does not exist, or if any
            selected case has no `<case_id>.npy` inside it. The second case
            names how many cases are missing and lists the first few --
            silently skipping them would shrink the sample the pre-registered
            statistics are computed over.
    """
    if not logits_dir.is_dir():
        raise FileNotFoundError(
            f"explainability.ambiguity.logits_dir does not exist: {logits_dir.resolve()}."
        )

    missing = [case_id for case_id in case_ids if not (logits_dir / f"{case_id}.npy").is_file()]
    if missing:
        preview = missing[:5]
        suffix = ", ..." if len(missing) > len(preview) else ""
        raise FileNotFoundError(
            f"{len(missing)} of {len(case_ids)} selected case(s) have no "
            f"<case_id>.npy in explainability.ambiguity.logits_dir={logits_dir.resolve()}: "
            f"{preview}{suffix}. Refusing to silently skip cases -- a shortened split would "
            "change the sample the pre-registered statistics are computed over. Point "
            "logits_dir at a logits/ directory produced by evaluating this SAME checkpoint, "
            "at this SAME split, with this SAME inference.sliding_window config."
        )


def _load_case_logits(logits_dir: Path, case_id: str) -> Tensor:
    """Loads one case's precomputed deterministic segmentation logits.

    Args:
        logits_dir: Directory holding one `<case_id>.npy` per case, as
            written by `scripts/evaluate.py`'s `save_logits` option --
            float16, shape `(out_channels, D, H, W)`.
        case_id: The case to load.

    Returns:
        The logits as a float32 tensor, shape `(out_channels, D, H, W)`
        (no batch dimension -- the caller adds one to match
        `sliding_window_predict`'s `(B, out_channels, D, H, W)` contract).
    """
    array = np.load(logits_dir / f"{case_id}.npy").astype(np.float32)
    return torch.from_numpy(array)


def resolve_ambiguity_checkpoint(cfg: DictConfig) -> Path:
    """Decides which checkpoint file to load for ambiguity extraction.

    Same convention as `scripts/evaluate.py`'s `resolve_checkpoint` /
    `scripts/extract_gates.py`'s `resolve_gates_checkpoint`, reading from
    `cfg.explainability.ambiguity.checkpoint` instead.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `Path(cfg.explainability.ambiguity.checkpoint)` if set, else
        `Path(cfg.training.checkpoint.dir) / "best.pt"`.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message
            lists whatever `.pt` files ARE present in that directory.
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


def load_ambiguity_model(cfg: DictConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Builds the model from config and loads a checkpoint's weights into it.

    Args:
        cfg: The full composed Hydra config.
        checkpoint_path: Path to the checkpoint, as returned by
            `resolve_ambiguity_checkpoint`.
        device: The resolved torch device to move the model to.

    Returns:
        The loaded model, in eval mode.
    """
    model = build_model(cfg)
    model = model.to(device)
    # restore_rng=False: extraction is deterministic inference, with no
    # reason to perturb the process's RNG state -- same reasoning as
    # scripts/evaluate.py's load_eval_model / extract_gates.py's
    # load_gates_model.
    load_checkpoint(checkpoint_path, model, map_location=str(device), restore_rng=False)
    model.eval()
    return model


def _check_forward_with_ambiguity(model: nn.Module, cfg: DictConfig) -> None:
    """Raises before any output is produced if `model` has no ambiguity maps to give.

    Args:
        model: The loaded model.
        cfg: The full composed Hydra config, used only to name the
            offending model in the error message.

    Raises:
        TypeError: If `model` has no `forward_with_ambiguity` method. Only
            `neurovision.models.neurovision.NeuroVisionX` has fusion blocks
            -- evaluating a `unet3d` or `swinunetr` checkpoint here is a
            configuration mistake worth catching in the first second, not on
            the first forward pass.
    """
    if not hasattr(model, "forward_with_ambiguity"):
        raise TypeError(
            f"model.name={cfg.model.name!r} has no forward_with_ambiguity method. Only the "
            "'neurovision' model has fusion blocks with ambiguity maps to extract -- point "
            "cfg.model.name (and cfg.explainability.ambiguity.checkpoint, if set) at a "
            "'neurovision' checkpoint instead."
        )


def build_ambiguity_dataloader(cfg: DictConfig, case_ids: Sequence[str]) -> DataLoader:
    """Builds a whole-volume `DataLoader` for the selected cases.

    Uses the same deterministic val transform pipeline as
    `scripts/evaluate.py` / `scripts/extract_gates.py` (no cropping, no
    randomness): sliding-window inference does the patching itself, so the
    dataset must hand back whole volumes.

    Args:
        cfg: The full composed Hydra config.
        case_ids: Case ids to load, in the order they should be yielded.

    Returns:
        A `DataLoader`, `batch_size=1` (whole volumes have per-case shapes
        and do not collate at any larger batch size), yielding batches in
        the SAME order as `case_ids`.
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


def _count_sliding_windows(spatial_shape: tuple[int, int, int], cfg: Any) -> int:
    """Counts the sliding-window inferer's window positions for one volume.

    Diagnostic only, recorded in `ambiguity_summary.csv`'s `n_windows`
    column so a reader can tell a whole-volume statistic from a
    single-window one at a glance. Recomputes what
    `monai.inferers.SlidingWindowInferer` visits internally, via the same
    (private, but stable and the only source of truth for this) MONAI
    helpers `sliding_window_inference` itself calls -- an independently
    reimplemented formula could silently drift from MONAI's own rounding.

    Args:
        spatial_shape: `(D, H, W)` of the volume sliding-window inference
            will run over.
        cfg: The full composed Hydra config, exposing
            `cfg.inference.sliding_window`.

    Returns:
        The number of window positions MONAI's inferer would visit for one
        volume of this shape.
    """
    if _get_scan_interval is None:
        # `_get_scan_interval` is MONAI-private and could be renamed by an
        # upgrade. It backs ONLY this diagnostic column, so a rename must
        # degrade that column to -1 rather than break extraction itself --
        # the ambiguity maps and every reported per-case scalar are
        # unaffected by it. Deps are pinned, so this is dormant; check it
        # before bumping MONAI, exactly as with skimage's `min_size`.
        logger.warning(
            "monai.inferers.utils._get_scan_interval is unavailable on this MONAI "
            "version; n_windows will be recorded as -1. Extraction is unaffected."
        )
        return -1

    sw_cfg = cfg.inference.sliding_window
    roi_size = fall_back_tuple(list(sw_cfg.roi_size), spatial_shape)
    num_spatial_dims = len(roi_size)
    overlap = [float(sw_cfg.overlap)] * num_spatial_dims

    # MONAI pads the input up to at least roi_size on every axis before
    # scanning (see sliding_window_inference's own padding step) -- matched
    # here so a volume shorter than roi_size on some axis is not undercounted.
    padded_shape = tuple(max(s, r) for s, r in zip(spatial_shape, roi_size))

    scan_interval = _get_scan_interval(padded_shape, roi_size, num_spatial_dims, overlap)
    slices = dense_patch_slices(padded_shape, roi_size, scan_interval)
    return len(slices)


def _split_channel_groups(
    tensor: Tensor, channel_groups: Sequence[tuple[str, int]]
) -> dict[str, Tensor]:
    """Slices a wrapper's channel-concatenated output back into named groups.

    Args:
        tensor: Shape `(total_channels, D, H, W)` -- the wrapper's per-case
            output with the batch dimension already removed.
        channel_groups: `_AmbiguityAtLevel.channel_groups` -- `(name, size)`
            pairs, in the SAME order the wrapper concatenated them in.
            Slicing from this attribute (rather than recomputing offsets
            independently here) is what keeps the two from ever disagreeing
            -- see `_AmbiguityAtLevel`'s class docstring.

    Returns:
        A dict from group name to that group's channel slice, e.g.
        `{"ambiguity": ..., "confidence": ..., "boundary": ...}` when
        `include_auxiliary` was True, or just `{"ambiguity": ...}` when it
        was False.
    """
    groups: dict[str, Tensor] = {}
    offset = 0
    for name, size in channel_groups:
        groups[name] = tensor[offset : offset + size]
        offset += size
    return groups


def summarize_case_ambiguity(
    disagreement: Tensor,
    entropy_cnn: Tensor,
    entropy_swin: Tensor,
    regions: Tensor,
    region_names: Sequence[str] = REGION_NAMES,
    confidence_logits: Tensor | None = None,
    gate: Tensor | None = None,
) -> dict[str, float]:
    """Reduces one case's per-voxel ambiguity (and, optionally, confidence and gate) maps to
    per-case scalar columns.

    Uses the PREDICTED-foreground mask (`regions > 0.5`) to define each
    region's foreground -- matching `scripts/evaluate.py`'s
    `uncertainty_summary.csv` convention exactly. See this module's
    top-of-file docstring for why that convention (and not the union mask
    `neurovision.uncertainty.risk_coverage.case_uncertainty_scalars`
    recommends) is used, and why this function takes no label argument at
    all: the scalar it returns must be computable with no access to the
    ground truth.

    Args:
        disagreement: `|p_cnn - p_swin|`, shape `(num_regions, D, H, W)`.
        entropy_cnn: CNN branch normalised Bernoulli entropy, same shape.
        entropy_swin: Swin branch normalised Bernoulli entropy, same shape.
        regions: Predicted region mask (already thresholded, e.g. via
            `neurovision.inference.postprocess.postprocess_logits`), shape
            `(num_regions, D, H, W)`, values in `{0, 1}`.
        region_names: Region channel names, in channel order. Defaults to
            `neurovision.data.transforms.REGION_NAMES`.
        confidence_logits: Optional confidence-head raw logits, shape
            `(num_regions, D, H, W)`. `None` (the default, and the only
            value ever passed when `explainability.ambiguity.
            include_auxiliary` is False) omits every `conf_*` column
            entirely, keeping the returned column set byte-for-byte
            identical to a run with no confidence head. The summarised
            quantity is `1 - sigmoid(confidence_logits)`, i.e. the head's
            PREDICTED ERROR PROBABILITY rather than its predicted
            correctness -- deliberately, so that HIGHER always means WORSE
            here, matching every other score column this function returns
            (`amb_dis_*` is also higher-is-worse). Contrast with
            `neurovision.models.neurovision.NeuroVisionX.
            forward_with_auxiliary`, whose documented convention for
            `sigmoid(confidence_logits)` itself is the OPPOSITE polarity
            (probability of being correct) -- getting this flipped would
            silently invert every failure-detection reading built on this
            column.
        gate: Optional fusion-gate maps, shape `(num_fusion_levels, D, H,
            W)`, one channel per fusion block, fine to coarse. `None` (the
            default, and the only value ever passed when `explainability.
            ambiguity.include_gates` is False) omits every `gate_*` column
            entirely. The foreground mask used for `gate_mean_fg_L` is the
            predicted WHOLE-TUMOR mask (`regions[region_names.index("WT")] >
            0.5`) for every level -- a gate map is a property of a spatial
            LOCATION relative to the tumor as a whole, not of any one
            ET/TC/WT region, so there is no natural per-region split to
            report it against the way `amb_dis_*` / `conf_*` do.

    Returns:
        One flat dict with, per region `R` in `region_names`:
        `amb_dis_mean_R`, `amb_dis_max_R`, `amb_dis_mean_fg_R`,
        `amb_hcnn_mean_fg_R`, `amb_hswin_mean_fg_R` -- plus
        `amb_dis_mean_fg_mean`, the NaN-skipping mean of `amb_dis_mean_fg_R`
        across regions. `*_fg_*` entries are NaN when that region's
        predicted foreground is empty: an empty prediction and a
        confidently certain prediction are different states and must not
        collapse to the same number (same convention `scripts/evaluate.py`
        uses for its `mi_mean_fg_*` columns). When `confidence_logits` is
        given, ALSO includes `conf_mean_R` and `conf_mean_fg_R` per region
        (same NaN-on-empty-foreground convention) and `conf_mean_fg_mean`
        (the NaN-skipping mean of `conf_mean_fg_R` across regions). When
        `gate` is given, ALSO includes `gate_mean_L` and `gate_mean_fg_L`
        (NaN when the whole-tumor prediction is empty) for each fusion level
        `L` in `range(gate.shape[0])`. Deliberately NO cross-level aggregate
        column for the gate: measured on the real model, level 1's mean gate
        runs 0.98 deep inside the tumor down to 0.33 in surrounding tissue,
        while level 2 runs the OPPOSITE direction -- the levels have
        opposite polarity, so averaging them together would cancel a real
        effect into noise rather than summarize it.
    """
    row: dict[str, float] = {}
    fg_means: list[float] = []
    conf_fg_means: list[float] = []

    # 1 - sigmoid(...): predicted ERROR probability, not predicted correctness -- see the
    # confidence_logits arg docstring above for why the polarity is flipped here.
    error_prob = 1.0 - torch.sigmoid(confidence_logits) if confidence_logits is not None else None

    for i, region in enumerate(region_names):
        dis = disagreement[i]
        hcnn = entropy_cnn[i]
        hswin = entropy_swin[i]
        fg_mask = regions[i] > 0.5

        row[f"amb_dis_mean_{region}"] = dis.mean().item()
        row[f"amb_dis_max_{region}"] = dis.max().item()

        if fg_mask.any():
            dis_fg_mean = dis[fg_mask].mean().item()
            row[f"amb_dis_mean_fg_{region}"] = dis_fg_mean
            row[f"amb_hcnn_mean_fg_{region}"] = hcnn[fg_mask].mean().item()
            row[f"amb_hswin_mean_fg_{region}"] = hswin[fg_mask].mean().item()
            fg_means.append(dis_fg_mean)
        else:
            # NaN, not 0.0 -- an empty prediction is not the same state as a
            # perfectly-agreeing one, and must not collapse to the same
            # number. Same convention scripts/evaluate.py uses.
            row[f"amb_dis_mean_fg_{region}"] = float("nan")
            row[f"amb_hcnn_mean_fg_{region}"] = float("nan")
            row[f"amb_hswin_mean_fg_{region}"] = float("nan")

        if error_prob is not None:
            err = error_prob[i]
            row[f"conf_mean_{region}"] = err.mean().item()
            if fg_mask.any():
                conf_fg_mean = err[fg_mask].mean().item()
                row[f"conf_mean_fg_{region}"] = conf_fg_mean
                conf_fg_means.append(conf_fg_mean)
            else:
                row[f"conf_mean_fg_{region}"] = float("nan")

    row["amb_dis_mean_fg_mean"] = float(np.mean(fg_means)) if fg_means else float("nan")
    if error_prob is not None:
        row["conf_mean_fg_mean"] = float(np.mean(conf_fg_means)) if conf_fg_means else float("nan")

    if gate is not None:
        # The whole-tumor mask, not a per-region one -- see the gate arg docstring above.
        wt_index = list(region_names).index("WT")
        fg_mask_wt = regions[wt_index] > 0.5
        for level_idx in range(gate.shape[0]):
            g = gate[level_idx]
            row[f"gate_mean_{level_idx}"] = g.mean().item()
            row[f"gate_mean_fg_{level_idx}"] = (
                g[fg_mask_wt].mean().item() if fg_mask_wt.any() else float("nan")
            )

    return row


def _log_and_print_summary(
    manifest_df: pd.DataFrame, summary_df: pd.DataFrame, split: str, n_cases: int
) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        manifest_df: The manifest DataFrame `run_extraction` is about to
            return / has just written.
        summary_df: The `ambiguity_summary.csv` DataFrame -- `n_windows`
            lives there, not in the manifest (see this module's
            `summarize_case_ambiguity` / `run_extraction` docstrings).
        split: The split name ambiguity maps were extracted from.
        n_cases: Number of cases processed.
    """
    lines = [
        "=" * 70,
        f"Ambiguity extraction summary -- split={split!r}, {n_cases} case(s)",
        "=" * 70,
    ]
    if manifest_df.empty:
        lines.append("No cases were processed.")
    else:
        level = int(manifest_df["level"].iloc[0])
        n_saved = int(manifest_df["maps_saved"].sum())
        mean_windows = summary_df["n_windows"].mean()
        lines.append(f"  level: {level}")
        lines.append(f"  cases with saved voxel maps: {n_saved}/{n_cases}")
        lines.append(f"  mean sliding-window positions per case: {mean_windows:.1f}")

    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/evaluate.py's / scripts/extract_gates.py's summary.
    print("\n".join(lines))


def run_extraction(cfg: DictConfig) -> pd.DataFrame:
    """Extracts fusion-ambiguity maps for a split and writes them to disk.

    For each selected case: get the deterministic segmentation logits --
    used only to build the predicted-foreground mask, exactly like
    `scripts/evaluate.py` -- and run whole-volume sliding-window inference
    through `_AmbiguityAtLevel` to get one fusion level's ambiguity map,
    upsampled to full resolution and stitched. By default the logits come
    from a second sliding-window pass through the plain model, deterministic
    (`set_eval=True`) and using the same weights as the ambiguity pass, which
    costs roughly double the inference time of a plain evaluation run in
    exchange for `_AmbiguityAtLevel`'s single-tensor contract staying exactly
    as simple as the spec: no packed/concatenated output to unpack
    downstream. When `explainability.ambiguity.logits_dir` is set, that
    second pass is skipped entirely and the logits are instead loaded from
    `<logits_dir>/<case_id>.npy` -- valid only when that directory was
    produced by evaluating this SAME checkpoint at this SAME
    `inference.sliding_window` settings, which is checked per case (see
    `_validate_logits_dir` for the up-front check and the per-case spatial
    shape assertion below).

    When `explainability.ambiguity.include_auxiliary` is True, the confidence and boundary
    heads' full-resolution logits are ALSO extracted, from the SAME sliding-window pass as
    the ambiguity map (see `_AmbiguityAtLevel`) -- `ambiguity_summary.csv` gains `conf_*`
    columns (predicted error probability, higher-is-worse -- see `summarize_case_ambiguity`)
    and each case's `.npz` gains `confidence` / `boundary` arrays. Requires the loaded model
    to have both auxiliary heads; raises immediately (before `out_dir` is created) otherwise.

    When `explainability.ambiguity.include_gates` is True, the fusion gate maps are ALSO
    extracted, from the SAME pass -- `ambiguity_summary.csv` gains `gate_mean_L` /
    `gate_mean_fg_L` columns per fusion level and each case's `.npz` gains a `gate` array.
    Requires the loaded model to have at least one fusion block (a Swin branch); raises
    immediately (before `out_dir` is created) if it does not, or per case if any fusion
    block reports no gate map at all (a non-`adaptive_gated` fusion variant).

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The summary DataFrame (also written to
        `<out_dir>/ambiguity_summary.csv`), indexed by `case_id`.

    Raises:
        ValueError: See `select_cases`, `_validate_labeled_cases`, and
            `_AmbiguityAtLevel.forward`. Also raised per case when
            `logits_dir` is set and a loaded case's spatial shape disagrees
            with that case's ambiguity map.
        FileNotFoundError: See `resolve_ambiguity_checkpoint` and
            `_validate_logits_dir`.
        TypeError: See `_check_forward_with_ambiguity`.
    """
    device = get_device(cfg)
    amb_cfg = cfg.explainability.ambiguity
    prep_dir = Path(cfg.data.preprocessing.out_dir)

    case_ids = select_cases(cfg)
    _validate_labeled_cases(case_ids, prep_dir)

    logits_dir_cfg = amb_cfg.logits_dir
    logits_dir = Path(logits_dir_cfg) if logits_dir_cfg is not None else None
    if logits_dir is not None:
        _validate_logits_dir(logits_dir, case_ids)
        logger.info(
            "Reusing deterministic segmentation logits from %s instead of running a second "
            "sliding-window pass through the plain model.",
            logits_dir.resolve(),
        )

    checkpoint_path = resolve_ambiguity_checkpoint(cfg)
    model = load_ambiguity_model(cfg, checkpoint_path, device)
    # Checked once, before the loop and before any output directory exists --
    # evaluating a unet3d/swinunetr checkpoint here is a user error worth
    # catching in the first second, not on the first forward pass.
    _check_forward_with_ambiguity(model, cfg)

    level = int(amb_cfg.level)
    num_regions = len(REGION_NAMES)
    # .get() with a False default, not amb_cfg.include_auxiliary / .include_gates: a config
    # composed before these keys existed must still build (same pattern as every other
    # .get()-guarded key in this project) -- the shipped configs/explainability/default.yaml
    # sets both True.
    include_auxiliary = bool(amb_cfg.get("include_auxiliary", False))
    include_gates = bool(amb_cfg.get("include_gates", False))
    # Raises here (before out_dir is created) if include_auxiliary/include_gates is True and
    # the model is missing what it needs -- see _AmbiguityAtLevel.__init__.
    wrapped_model = _AmbiguityAtLevel(
        model,
        level,
        include_auxiliary=include_auxiliary,
        include_gates=include_gates,
        num_regions=num_regions,
    ).to(device)
    wrapped_model.eval()

    out_dir = ensure_dir(amb_cfg.out_dir)
    save_maps = bool(amb_cfg.save_maps)
    save_image = bool(amb_cfg.save_image)

    loader = build_ambiguity_dataloader(cfg, case_ids)

    summary_rows: dict[str, dict[str, float]] = {}
    manifest_rows: dict[str, dict[str, Any]] = {}

    summary_csv_path = out_dir / "ambiguity_summary.csv"
    manifest_csv_path = out_dir / "ambiguity_manifest.csv"

    model.eval()
    with torch.no_grad():
        progress = tqdm(zip(case_ids, loader), total=len(case_ids), desc="Extracting ambiguity")
        for case_id, batch in progress:
            meta = read_json(prep_dir / case_id / "meta.json")
            image = batch["image"]  # (1, 4, D, H, W)
            spatial_shape = tuple(image.shape[2:])

            # Deterministic segmentation logits -- used only to build the
            # predicted-foreground mask, never saved as "the" prediction of
            # this script (scripts/evaluate.py already owns that artifact).
            # Either loaded from a prior evaluation run's saved logits, or
            # (the default) a second sliding-window pass through the plain
            # model.
            if logits_dir is not None:
                seg_logits = _load_case_logits(logits_dir, case_id).unsqueeze(0)
            else:
                seg_logits = sliding_window_predict(model, image, cfg, device)
            regions = postprocess_logits(seg_logits, cfg)[0]  # (num_regions, D, H, W)

            # Ambiguity pass, through the per-level wrapper -- also carries the fusion gates
            # and/or the confidence / boundary heads' logits, channel-concatenated, when
            # include_gates / include_auxiliary are True.
            combined_full = sliding_window_predict(wrapped_model, image, cfg, device, set_eval=True)
            combined = combined_full[0]  # (total_channels, D, H, W)

            # Sliced from wrapped_model's OWN recorded layout, never recomputed offsets --
            # see _AmbiguityAtLevel's class docstring for why that is load-bearing.
            groups = _split_channel_groups(combined, wrapped_model.channel_groups)
            ambiguity = groups["ambiguity"]  # (3 * num_regions, D, H, W)
            gate = groups.get("gate")  # (num_fusion_levels, D, H, W) or absent
            confidence_logits = groups.get("confidence")  # (num_regions, D, H, W) or absent
            boundary_logits = groups.get("boundary")  # (num_regions, D, H, W) or absent

            if logits_dir is not None:
                loaded_shape = tuple(seg_logits.shape[2:])
                ambiguity_shape = tuple(ambiguity.shape[1:])
                if loaded_shape != ambiguity_shape:
                    raise ValueError(
                        f"Loaded logits for case {case_id!r} from {logits_dir.resolve()} have "
                        f"spatial shape {loaded_shape}, which does not match this case's "
                        f"ambiguity map spatial shape {ambiguity_shape}. This means logits_dir "
                        "was produced from a different preprocessing run or cohort than the "
                        "one being extracted here -- the arrays would still load and mask "
                        "'successfully' against the wrong geometry, so this is checked "
                        "explicitly rather than left to produce a silently misaligned result."
                    )

            disagreement = ambiguity[0:num_regions]
            entropy_cnn = ambiguity[num_regions : 2 * num_regions]
            entropy_swin = ambiguity[2 * num_regions : 3 * num_regions]

            row = summarize_case_ambiguity(
                disagreement.cpu(),
                entropy_cnn.cpu(),
                entropy_swin.cpu(),
                regions.cpu(),
                confidence_logits=(
                    confidence_logits.cpu() if confidence_logits is not None else None
                ),
                gate=gate.cpu() if gate is not None else None,
            )
            row["level"] = float(level)
            row["n_windows"] = float(_count_sliding_windows(spatial_shape, cfg))
            summary_rows[case_id] = row

            if save_maps:
                save_arrays: dict[str, np.ndarray] = {
                    "disagreement": disagreement.cpu().numpy().astype(np.float16),
                    "entropy_cnn": entropy_cnn.cpu().numpy().astype(np.float16),
                    "entropy_swin": entropy_swin.cpu().numpy().astype(np.float16),
                    "logits": seg_logits[0].cpu().numpy().astype(np.float16),
                }
                if gate is not None:
                    save_arrays["gate"] = gate.cpu().numpy().astype(np.float16)
                if confidence_logits is not None:
                    save_arrays["confidence"] = confidence_logits.cpu().numpy().astype(np.float16)
                if boundary_logits is not None:
                    # Saved for a figure only -- the boundary head predicts a morphological
                    # shell, not correctness, so it gets no per-case summary scalar (see
                    # summarize_case_ambiguity's docstring and this script's config comment).
                    save_arrays["boundary"] = boundary_logits.cpu().numpy().astype(np.float16)
                if save_image:
                    save_arrays["image"] = image[0].cpu().numpy().astype(np.float16)
                np.savez_compressed(out_dir / f"{case_id}.npz", **save_arrays)

            manifest_rows[case_id] = {
                "shape_d": spatial_shape[0],
                "shape_h": spatial_shape[1],
                "shape_w": spatial_shape[2],
                "level": level,
                "has_label": bool(meta["has_label"]),
                "maps_saved": save_maps,
                "include_auxiliary": include_auxiliary,
                "include_gates": include_gates,
                # Provenance of the segmentation logits used for the
                # predicted-foreground mask and (if save_maps) the saved
                # "logits" array -- a reader must be able to tell from the
                # artifact alone whether that came from a second model pass
                # or a reused prior evaluation run.
                "logits_source": str(logits_dir.resolve()) if logits_dir is not None else "model",
            }

            # Rewritten every iteration, not just at the end: a full-split
            # sliding-window run can take minutes, and a killed run should
            # keep every already-processed case instead of losing all of
            # them. Same reasoning as scripts/evaluate.py's per-case CSVs.
            pd.DataFrame.from_dict(summary_rows, orient="index").rename_axis("case_id").to_csv(
                summary_csv_path
            )
            pd.DataFrame.from_dict(manifest_rows, orient="index").rename_axis("case_id").to_csv(
                manifest_csv_path
            )

    summary_df = pd.DataFrame.from_dict(summary_rows, orient="index").rename_axis("case_id")
    manifest_df = pd.DataFrame.from_dict(manifest_rows, orient="index").rename_axis("case_id")

    config_path = out_dir / "ambiguity_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    _log_and_print_summary(manifest_df, summary_df, amb_cfg.split, len(case_ids))

    return summary_df


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Extracts fusion-ambiguity maps over a split, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_extraction(cfg)


if __name__ == "__main__":
    main()
