"""Segmentation QC model: predicts a mask's own Dice score, with no label.

Phase C of the master plan asks "does the pipeline know when it is wrong" as
a deployable component, not a property claimed about one architecture. The
segmentation model (`neurovision.models.neurovision` / `baseline`) produces a
mask; THIS module is a second, independently trained network that looks at
one case and that mask and estimates the Dice score it would get against a
ground-truth label that is never available at inference time.

This file builds and registers the network only (`SegQC`, `build_segqc`).
Training (`neurovision.data.qc_pairs` generates the (degraded mask, true
Dice) pairs it learns from) is Phase C3, a separate module.

Intended input packing (C3's decision, not enforced here):
    The model runs ONCE PER REGION (ET, TC, WT), each time seeing 3 channels
    stacked as `[image modality, predicted mask for that region, uncertainty
    map for that region]`, and produces one Dice estimate per region per
    case. `in_channels` is a config value rather than hard-coded specifically
    so a later variant -- e.g. stacking all 4 MRI modalities alongside the
    mask and uncertainty map, giving 6 channels -- needs no code change here,
    only a config change and a different packing decision made by whatever
    calls this model.

Output convention -- read this before touching either function below:
    `SegQC.forward` returns a raw scalar per sample, a LOGIT, not a Dice
    value. Dice is bounded in [0, 1]; a logit is not. This mirrors every
    other model in this project (`neurovision.inference` applies sigmoid or
    softmax to raw model output, never has it baked into `forward`), so the
    QC model's caller applies the same nonlinearity explicitly rather than
    the model doing it silently. Use `predicted_dice()` below to get the
    actual, human-readable Dice estimate in [0, 1]. Plotting `forward`'s
    output directly as "estimated Dice" produces a plausible-looking, wrong
    figure -- this project has already shipped that exact class of bug once
    (see CLAUDE.md's calibration-mask trap), so the split is deliberate and
    both sites below repeat the warning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from neurovision.models.registry import register_model

logger = logging.getLogger(__name__)


class SegQC(nn.Module):
    """Small 3D CNN regressor: (image, predicted mask, uncertainty) -> Dice logit.

    Architecture: a stack of strided `Conv3d -> GroupNorm -> LeakyReLU`
    blocks (one block per entry of `widths`, each halving the spatial size
    after the first), then a global average pool, then a small MLP down to
    one scalar.

    Why GroupNorm, not BatchNorm3d: this model is trained on individual
    cases with a small batch size, on volumes that vary in size (see below),
    so a batch's statistics are noisy and unreliable as a normalisation
    signal -- the same reasoning `neurovision.models.baseline`'s U-Net
    applies (it uses `norm="instance"`, an even smaller-group case of the
    same idea). GroupNorm normalises within each sample, independent of
    batch size, so training stays stable regardless of batch composition.

    Why global average pooling, not a flatten-then-linear head: this
    project's cases are cropped to their own nonzero bounding box before
    training (see `neurovision.data.transforms`), so input volumes are NOT
    all the same size -- one case might be `(3, 140, 172, 138)` and the next
    `(3, 96, 150, 121)`. A flattened feature map's length depends on the
    input's spatial size, so a `Linear` layer built for one shape would
    raise (or silently mismatch) on the next case entirely. Global average
    pooling collapses the spatial dimensions to a single value per channel
    regardless of their size, so the exact same trained weights work on any
    input shape -- this is what `test_handles_variable_input_sizes` in
    `tests/test_qc_model.py` exists to prove.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (16, 32, 64, 128),
        num_groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Builds the encoder stack and the scalar regression head.

        Args:
            in_channels: Number of input channels. Default 3, matching the
                intended `[image modality, predicted mask, uncertainty map]`
                packing for one region (see module docstring). Not
                hard-coded to 3 so a later, differently-packed variant needs
                only a config change.
            widths: Output channel count at each encoder level, applied in
                order. Each level halves the spatial size (stride-2 conv)
                except the first, which only projects `in_channels` up to
                `widths[0]` at the input resolution.
            num_groups: Number of groups for every `GroupNorm` in the
                encoder. Every entry of `widths` must be evenly divisible by
                this, since `GroupNorm` splits its channels into
                `num_groups` equal-sized groups.
            dropout: Dropout probability applied right before the final
                linear layer of the regression head.

        Raises:
            ValueError: If any entry of `widths` is not evenly divisible by
                `num_groups` -- letting this reach `nn.GroupNorm` directly
                would raise there instead, with a message that does not
                name which of the two numbers is the problem.
        """
        super().__init__()

        for width in widths:
            if width % num_groups != 0:
                raise ValueError(
                    f"Every entry of widths must be evenly divisible by num_groups, so "
                    f"GroupNorm can split each level's channels into num_groups equal "
                    f"groups. Got width={width}, num_groups={num_groups} "
                    f"(width % num_groups = {width % num_groups} != 0)."
                )

        blocks: list[nn.Module] = []
        prev_channels = in_channels
        for level, width in enumerate(widths):
            # Level 0 only projects the input's channel count up to widths[0]
            # at the input's own resolution (stride 1). Every later level
            # both changes channel count and halves the spatial size in the
            # same conv (stride 2) -- this is the "stride-2 downsample
            # between levels" the spec asks for, done inline rather than as
            # a separate pooling layer.
            stride = 1 if level == 0 else 2
            blocks.append(
                nn.Sequential(
                    nn.Conv3d(prev_channels, width, kernel_size=3, stride=stride, padding=1),
                    nn.GroupNorm(num_groups=num_groups, num_channels=width),
                    nn.LeakyReLU(inplace=True),
                )
            )
            prev_channels = width
        self.encoder = nn.Sequential(*blocks)

        # Collapses (B, widths[-1], D, H, W) -> (B, widths[-1], 1, 1, 1)
        # regardless of D/H/W, which is what lets the same weights run on
        # any case's cropped-to-bounding-box volume size. See class
        # docstring for why this is load-bearing, not stylistic.
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

        hidden = max(1, widths[-1] // 2)
        self.head = nn.Sequential(
            nn.Linear(widths[-1], hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs the QC model on a packed volume.

        Args:
            x: `(B, in_channels, D, H, W)`. `D`, `H`, `W` may differ between
                calls (see class docstring) -- the global pool makes this
                safe.

        Returns:
            `(B,)`, a raw LOGIT per sample, NOT a Dice value. Apply
            `predicted_dice()` to get a number in [0, 1].
        """
        features = self.encoder(x)  # (B, widths[-1], D', H', W')
        pooled = self.pool(features).flatten(start_dim=1)  # (B, widths[-1])
        logits = self.head(pooled)  # (B, 1)
        return logits.squeeze(-1)  # (B,)


def predicted_dice(logits: Tensor) -> Tensor:
    """Converts `SegQC.forward`'s raw logits into Dice estimates.

    `forward` deliberately returns a logit, not a probability-like Dice
    value (see module docstring). This is the one place that nonlinearity
    should be applied -- callers that want the number a human reads (e.g.
    "estimated Dice: 0.81") call this, never `torch.sigmoid` inline
    somewhere else, so there is exactly one place the convention could
    silently drift.

    Args:
        logits: Raw output of `SegQC.forward`, any shape.

    Returns:
        Same shape as `logits`, every value in `[0, 1]`.
    """
    return torch.sigmoid(logits)


@register_model("segqc")
def build_segqc(cfg: Any) -> nn.Module:
    """Builds the segmentation QC model from `cfg.model`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.model` with keys
            matching `configs/model/segqc.yaml`: `in_channels`, `widths`,
            `num_groups`, `dropout`.

    Returns:
        A `SegQC` instance.
    """
    model_cfg = cfg.model
    # Hydra hands sequence fields over as OmegaConf ListConfig objects, not
    # plain lists; iterating one already yields plain Python ints, so
    # tuple(...) is all the normalisation needed (same approach as
    # neurovision.models.baseline's _to_tuple).
    widths = tuple(model_cfg.widths)
    logger.info(
        "Building segqc: in_channels=%d, widths=%s, num_groups=%d, dropout=%.3f",
        model_cfg.in_channels,
        widths,
        model_cfg.num_groups,
        model_cfg.dropout,
    )
    return SegQC(
        in_channels=model_cfg.in_channels,
        widths=widths,
        num_groups=model_cfg.num_groups,
        dropout=model_cfg.dropout,
    )
