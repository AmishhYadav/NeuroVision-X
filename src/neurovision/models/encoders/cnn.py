"""3D residual CNN encoder producing a multi-scale feature pyramid.

This is one half of the dual encoder (the other is the Swin Transformer
branch); its pyramid feeds the adaptive gated cross-attention fusion module.
It is deliberately NOT registered with `neurovision.models.registry` — that
registry holds complete segmentation models (`build_model` -> a full
network), while this file builds one reusable component of one.

## Why GroupNorm and never BatchNorm

Every norm layer in this file is `nn.GroupNorm`. That is a deliberate choice,
not an oversight, for reasons specific to this project:

1. The real per-step batch is small and correlated. Training uses
   `training.batch_size (1) x samples_per_volume (4)` = 4 patches per step,
   and MONAI's `RandCropByPosNegLabeld` draws all 4 crops from the SAME
   volume. BatchNorm estimates each channel's mean/variance from the batch,
   but 4 crops of one brain are not 4 independent samples — effectively it
   would be computing statistics from 1-2 independent draws, which is noisy
   enough to destabilize training.
2. BatchNorm behaves differently in train mode (uses the current batch's
   statistics) versus eval mode (uses running statistics accumulated during
   training). That train/eval gap is a known source of miscalibrated output
   probabilities — the model's confidence at eval time does not match what
   it learned from at train time. This project's headline claim is
   calibration, so a norm layer that quietly degrades it would undercut the
   whole point of the paper.
3. Gradient accumulation does not fix either problem above. Accumulation
   only changes when `optimizer.step()` is called; each individual forward
   pass still only ever sees its own small, correlated micro-batch, so
   BatchNorm's statistics are exactly as noisy as they would be without
   accumulation.
4. GroupNorm sidesteps all of this by never looking across the batch at all.
   It splits each sample's channels into groups and normalizes each group
   using only that one sample's own activations. No running statistics are
   kept, so train and eval mode compute the exact same thing — no gap to
   miscalibrate.
5. Why GroupNorm rather than InstanceNorm (which the `unet3d` baseline
   uses): InstanceNorm is the special case of GroupNorm where every group
   contains exactly one channel (`num_groups == num_channels`). Normalizing
   channels in small groups rather than completely alone lets the layer
   retain some cross-channel scale information, which tends to help at the
   channel widths used here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


def _to_tuple(value: Any) -> tuple:
    """Converts a sequence-valued config field to a plain tuple.

    Hydra hands sequence fields over as OmegaConf `ListConfig` objects, which
    are not `list`/`tuple` instances. This normalizes any sequence to a plain
    tuple before it is used to build the encoder.

    Args:
        value: A `ListConfig`, list, tuple, or other sequence.

    Returns:
        A plain `tuple` with the same elements.
    """
    return tuple(value)


class ResidualBlock(nn.Module):
    """A single post-activation 3D residual block.

    `Conv3d -> GroupNorm -> LeakyReLU -> Dropout3d -> Conv3d -> GroupNorm`,
    added to a shortcut path, then a final `LeakyReLU`. The first conv
    carries the block's stride and any channel-width change; the second conv
    is always stride 1 at constant width.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Spatial stride of the first conv (and of the shortcut, if
            the shortcut is not identity).
        num_groups: Number of GroupNorm groups. `out_channels` must be
            divisible by this.
        dropout: `Dropout3d` probability applied between the two convs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # bias=False on both convs: GroupNorm immediately follows each one
        # and applies its own learned per-channel shift, so a conv bias term
        # would be redundant (and immediately absorbed/cancelled by norm).
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # Dropout3d zeros whole channels rather than individual voxels, the
        # correct variant for convolutional feature maps: neighbouring
        # voxels in a feature map are highly correlated, so ordinary
        # per-voxel dropout barely perturbs the signal.
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )

        self.act_out = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the residual block.

        Args:
            x: Input, shape `(B, in_channels, D, H, W)`.

        Returns:
            Output, shape `(B, out_channels, D', H', W')` where the spatial
            dims are downsampled by `stride`.
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        return self.act_out(out)


class CNNEncoder(nn.Module):
    """3D residual CNN encoder that outputs a fine-to-coarse feature pyramid.

    The encoder has `len(channels)` levels. Level 0 is a full-resolution
    stem (stride 1); each subsequent level halves every spatial dimension
    (stride 2) via the first residual block of its stage, then applies the
    remaining blocks of that stage at stride 1.

    Args:
        in_channels: Number of input channels (MRI modalities).
        channels: Output channel width at each pyramid level, fine to
            coarse. Its length sets the number of levels.
        blocks_per_stage: Number of residual blocks at each level. Must be
            the same length as `channels`.
        num_groups: GroupNorm group count, shared by every block. Every
            entry of `channels` must be divisible by this.
        dropout: `Dropout3d` probability used in every residual block.
        use_checkpoint: If True, wrap each stage in gradient checkpointing
            during training to trade compute for activation memory.

    Raises:
        ValueError: If `blocks_per_stage` and `channels` have different
            lengths, or if any entry of `channels` is not divisible by
            `num_groups`.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        num_groups: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        channels = list(channels)
        blocks_per_stage = list(blocks_per_stage)

        if len(blocks_per_stage) != len(channels):
            raise ValueError(
                f"blocks_per_stage has {len(blocks_per_stage)} entries but channels has "
                f"{len(channels)} entries; they must be the same length, one entry per "
                f"pyramid level."
            )

        for level, width in enumerate(channels):
            if width % num_groups != 0:
                raise ValueError(
                    f"channels[{level}] = {width} is not divisible by num_groups = "
                    f"{num_groups}. GroupNorm requires num_channels % num_groups == 0."
                )

        self.out_channels: list[int] = list(channels)
        self.strides: list[int] = [2**i for i in range(len(channels))]
        self.num_levels: int = len(channels)
        self.use_checkpoint = use_checkpoint

        stages = []
        prev_width = in_channels
        for level, (width, n_blocks) in enumerate(zip(channels, blocks_per_stage, strict=True)):
            # Level 0 is the stem: stride 1 throughout. Level i > 0: the
            # first block of the stage carries stride 2 and the width
            # change from the previous level; the rest are stride-1 blocks
            # at constant width.
            stage_stride = 1 if level == 0 else 2
            blocks = [ResidualBlock(prev_width, width, stage_stride, num_groups, dropout)]
            for _ in range(n_blocks - 1):
                blocks.append(ResidualBlock(width, width, 1, num_groups, dropout))
            stages.append(nn.Sequential(*blocks))
            prev_width = width

        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> list[Tensor]:
        """Runs the encoder, returning a fine-to-coarse feature pyramid.

        Args:
            x: Input volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            A list of length `num_levels`, ordered level 0 (finest, stride
            1) to level `num_levels - 1` (coarsest). Element `i` has shape
            `(B, channels[i], ceil(D / 2**i), ceil(H / 2**i), ceil(W / 2**i))`.
        """
        pyramid: list[Tensor] = []
        out = x
        for stage in self.stages:
            # Checkpointing trades ~20-30% step time for a large activation
            # memory saving, which matters against the 16 GB T4 VRAM budget.
            # use_reentrant=False is required for correct gradients when a
            # checkpointed module's input does not itself require grad
            # (true for level 0, whose input is the raw image, not a
            # tensor with grad tracking). Only take this path in training
            # with grad enabled: torch.utils.checkpoint warns and silently
            # returns no gradient at all when nothing in the input requires
            # grad, which is exactly the eval/no_grad case.
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                out = checkpoint(stage, out, use_reentrant=False)
            else:
                out = stage(out)
            pyramid.append(out)
        return pyramid


def build_cnn_encoder(cfg: Any) -> CNNEncoder:
    """Builds the CNN encoder from `cfg.model.encoder.cnn`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.data.in_channels`
            and `cfg.model.encoder.cnn` with keys `channels`,
            `blocks_per_stage`, `num_groups`, `dropout`, `use_checkpoint`.

    Returns:
        A constructed `CNNEncoder`.
    """
    cnn_cfg = cfg.model.encoder.cnn

    encoder = CNNEncoder(
        in_channels=cfg.data.in_channels,
        channels=_to_tuple(cnn_cfg.channels),
        blocks_per_stage=_to_tuple(cnn_cfg.blocks_per_stage),
        num_groups=cnn_cfg.num_groups,
        dropout=cnn_cfg.dropout,
        use_checkpoint=cnn_cfg.use_checkpoint,
    )

    logger.info(
        "Built CNNEncoder: %d levels, channels=%s, strides=%s",
        encoder.num_levels,
        encoder.out_channels,
        encoder.strides,
    )

    return encoder
