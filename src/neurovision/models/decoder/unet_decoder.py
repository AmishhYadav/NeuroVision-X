"""Coarse-to-fine U-Net decoder that turns a fused skip pyramid into features.

`neurovision.models.encoders.cnn.CNNEncoder`, `neurovision.models.encoders.swin.SwinEncoder`
and `neurovision.models.fusion.adaptive_fusion` together produce a fine-to-coarse pyramid of
skip connections: CNN level 0 (full resolution, stride 1, CNN-only — see the fusion module
docstring for why the transformer branch has no stride-1 feature) followed by one fused
feature per level where both branches exist (strides 2, 4, 8, 16 in the production config).
This decoder is the piece that walks that pyramid back up from the bottleneck to full
resolution, merging one skip per step, the standard U-Net shape.

## This module owns no segmentation heads

`forward` returns decoder FEATURES, not logits — one tensor per stage, fine to coarse,
NOT including the bottleneck itself (the bottleneck is an input to this decoder, not
something it produces). Attaching a `Conv3d(decoder_channels[i], num_classes, 1)` head to
element 0 gives the main segmentation output; attaching heads to some or all of the other
elements gives deep supervision. Whether to do either of those, and how many heads, is the
caller's decision — this file only has to guarantee every intermediate feature is exposed at
a known, documented width (`self.out_channels`) so the caller never has to re-derive it.

## Upsampling: `deconv` vs `interp`, and why `deconv` is the default

`upsample="deconv"` uses a learnable `nn.ConvTranspose3d(kernel_size=2, stride=2)`. This is
what nnU-Net and SwinUNETR both use, and it is the default here. Its downside: a transposed
convolution's kernel footprints overlap, and depending on the learned weights that overlap
can produce a faint periodic "checkerboard" texture in the output. That matters more in this
project than in most, because the headline research claim is boundary accuracy — a
checkerboard artifact would show up directly as periodic noise in the predicted tumor margin.

`upsample="interp"` instead does `F.interpolate(..., mode="trilinear")` (a fixed, non-learned
resampling that cannot checkerboard by construction) followed by a `Conv3d(kernel_size=1)` to
change channel width. Its downside is memory, not artifacts: the interpolation happens at the
INPUT channel width, before the 1x1x1 conv narrows it, so the intermediate activation is
wider than the deconv path's ever gets. `interp` exists so a checkerboard hypothesis can be
ruled out by ablation if the boundary metrics ever look periodic — it is not expected to be
the better default, just the control.

## Attention gates: off by default, on purpose

`use_attention_gates=True` adds a second, independent attention mechanism (Oktay et al.
style) at every skip connection. It defaults to **off**. This project's one novel
contribution is the fusion module between the two encoder branches; if the decoder also ran
attention by default, any improvement over a baseline could not be cleanly attributed to
fusion versus decoder-side attention. The flag exists to be ablated (e.g. "does adding
decoder attention on top of fusion help further"), not to be left on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from neurovision.models.encoders.cnn import ResidualBlock

logger = logging.getLogger(__name__)


def _match_spatial(x: Tensor, target_shape: tuple[int, int, int]) -> Tensor:
    """Crops or zero-pads `x`'s trailing 3 spatial dims to exactly `target_shape`.

    Both encoder branches halve odd axes with a `ceil(size / 2)` convention (padded
    stride-2 convs on the CNN side, pad-then-merge on the Swin side), and both always add
    that padding at the END of the axis. `2 * ceil(n / 2)` is either `n` or `n + 1`, so an
    upsampled decoder feature is at most ONE voxel too large per axis relative to the skip
    it needs to merge with — never more, and never too small, in the intended use of this
    function. Both directions are still handled defensively, and anything past a 1-voxel
    gap raises, because that size of a gap can only mean the skip list came from a
    differently-shaped network, not an odd-size rounding effect this function is meant to
    absorb silently.

    Args:
        x: Decoder feature after upsampling, shape `(B, C, D, H, W)`.
        target_shape: The `(D, H, W)` spatial shape to match, taken from the skip
            connection this feature is about to be concatenated with.

    Returns:
        `x` with spatial shape exactly `target_shape`.

    Raises:
        ValueError: If any axis differs from `target_shape` by more than 1 voxel.
    """
    current = tuple(x.shape[2:])
    for axis, (cur, tgt) in enumerate(zip(current, target_shape, strict=True)):
        if abs(tgt - cur) > 1:
            raise ValueError(
                f"Spatial mismatch on axis {axis} is {tgt - cur} voxels, which exceeds the "
                f"1-voxel tolerance expected from ceil(size/2) rounding: upsampled feature "
                f"has shape {current}, skip has shape {target_shape}. This means the skip "
                f"list came from a different network, not an odd-size rounding effect."
            )

    # Crop first, from the END of each axis, never the centre. Both encoders pad at the
    # end when halving an odd axis, so the one extra voxel an upsample can introduce
    # corresponds exactly to that trailing pad. A centre crop would instead shift every
    # decoder feature by half a voxel relative to its skip -- a spatial-offset bug that
    # produces entirely plausible-looking output (nothing about a shape test would catch
    # it) while progressively misaligning the predicted mask against the underlying image.
    d, h, w = target_shape
    x = x[..., :d, :h, :w]

    # Then pad (zero) any axis that came out smaller than the target. Padding after
    # cropping, rather than combining the two, keeps each axis's adjustment independent
    # and lets the same call handle a mix of "too large on D" and "too small on H".
    cropped = tuple(x.shape[2:])
    pad_d = max(0, target_shape[0] - cropped[0])
    pad_h = max(0, target_shape[1] - cropped[1])
    pad_w = max(0, target_shape[2] - cropped[2])
    if pad_d or pad_h or pad_w:
        # F.pad on a 5D tensor pads from the LAST dim backward:
        # (W_left, W_right, H_left, H_right, D_left, D_right).
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    return x


class AttentionGate(nn.Module):
    """Oktay-style attention gate applied to a skip connection before concatenation.

    `attn = sigmoid(psi(leaky_relu(theta(skip) + phi(gate))))`, `return skip * attn`. The
    gating signal here is the decoder feature AFTER it has already been upsampled to the
    skip's resolution (see `UNetDecoder.forward`), so both `theta(skip)` and `phi(gate)` are
    computed at the same spatial resolution and no resampling happens inside this module.
    That differs from Oktay et al.'s original formulation, which gates at the skip's
    (finer) resolution using a coarser gating signal and resamples the gate map inside the
    attention block itself — that extra resampling step is unnecessary here because the
    caller already aligned the two resolutions upstream.

    Args:
        skip_channels: Channel width of the skip connection being gated.
        gate_channels: Channel width of the gating signal (the upsampled decoder feature).
    """

    def __init__(self, skip_channels: int, gate_channels: int) -> None:
        super().__init__()
        inter_channels = max(1, skip_channels // 2)
        self.theta = nn.Conv3d(skip_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv3d(gate_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Zero the final conv's bias so the gate starts at sigmoid(0) = 0.5 everywhere:
        # uniform, with no prior yet on which skip regions matter. Weight is left at
        # PyTorch's default random init so the gate is centred but not perfectly flat,
        # matching the same reasoning GateGenerator uses in adaptive_fusion.py.
        nn.init.zeros_(self.psi.bias)

    def forward(self, skip: Tensor, gate: Tensor) -> Tensor:
        """Gates a skip connection.

        Args:
            skip: Skip connection feature, shape `(B, skip_channels, D, H, W)`.
            gate: Gating signal (upsampled decoder feature), shape
                `(B, gate_channels, D, H, W)`, same spatial shape as `skip`.

        Returns:
            Gated skip, shape `(B, skip_channels, D, H, W)`.
        """
        attn = self.sigmoid(self.psi(self.act(self.theta(skip) + self.phi(gate))))
        return skip * attn


class UNetDecoder(nn.Module):
    """Coarse-to-fine decoder that merges a fine-to-coarse skip pyramid.

    See the module docstring for the upsample-mode trade-off and why attention gates
    default to off. This class works for any pyramid depth >= 2; production wires it to a
    5-entry pyramid (`[cnn_level0, fused_s2, fused_s4, fused_s8, fused_s16]`) but nothing
    here assumes that specific length.

    Args:
        skip_channels: Channel width of each skip connection, fine to coarse. Element 0 is
            the full-resolution skip; the LAST element is the bottleneck (an input to this
            decoder, not one of its outputs). Length must be >= 2.
        decoder_channels: Output width of each decoder stage, fine to coarse. Length must
            be `len(skip_channels) - 1`. Defaults to `list(skip_channels[:-1])`, i.e. each
            stage comes back out at the width of the skip it merged with.
        blocks_per_stage: Number of `ResidualBlock`s run at each stage after the
            skip is concatenated in. Must be >= 1.
        num_groups: GroupNorm group count, shared by every norm layer in this module. Every
            entry of `skip_channels` and `decoder_channels` must be divisible by this.
        dropout: `Dropout3d` probability, passed through to every `ResidualBlock`.
        upsample: `"deconv"` (learnable, default) or `"interp"` (fixed trilinear + 1x1x1
            conv). See the module docstring.
        use_attention_gates: If True, gate every skip connection with an `AttentionGate`
            before concatenation. Defaults to False — see the module docstring.
        use_checkpoint: If True, wrap each stage's residual-block sequence in gradient
            checkpointing during training, trading compute for activation memory.

    Raises:
        ValueError: If `len(skip_channels) < 2`; `decoder_channels` is given with a length
            other than `len(skip_channels) - 1`; any entry of `skip_channels` or
            `decoder_channels` is not divisible by `num_groups`; `upsample` is not
            `"deconv"`/`"interp"`; or `blocks_per_stage < 1`.
    """

    def __init__(
        self,
        skip_channels: Sequence[int],
        decoder_channels: Sequence[int] | None = None,
        blocks_per_stage: int = 2,
        num_groups: int = 8,
        dropout: float = 0.0,
        upsample: str = "deconv",
        use_attention_gates: bool = False,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        skip_channels = list(skip_channels)
        if len(skip_channels) < 2:
            raise ValueError(
                f"skip_channels must have at least 2 entries (one bottleneck plus at least "
                f"one skip to merge with), got {len(skip_channels)}."
            )

        if decoder_channels is None:
            decoder_channels = list(skip_channels[:-1])
        else:
            decoder_channels = list(decoder_channels)
            expected_len = len(skip_channels) - 1
            if len(decoder_channels) != expected_len:
                raise ValueError(
                    f"decoder_channels has {len(decoder_channels)} entries, expected "
                    f"{expected_len} (= len(skip_channels) - 1 = {len(skip_channels)} - 1)."
                )

        for level, width in enumerate(decoder_channels):
            if width % num_groups != 0:
                raise ValueError(
                    f"decoder_channels[{level}] = {width} is not divisible by num_groups = "
                    f"{num_groups}. GroupNorm requires num_channels % num_groups == 0."
                )
        for level, width in enumerate(skip_channels):
            if width % num_groups != 0:
                raise ValueError(
                    f"skip_channels[{level}] = {width} is not divisible by num_groups = "
                    f"{num_groups}. GroupNorm requires num_channels % num_groups == 0."
                )

        if upsample not in ("deconv", "interp"):
            raise ValueError(f"upsample must be 'deconv' or 'interp', got {upsample!r}.")

        if blocks_per_stage < 1:
            raise ValueError(f"blocks_per_stage must be >= 1, got {blocks_per_stage}.")

        self.skip_channels: list[int] = skip_channels
        self.out_channels: list[int] = decoder_channels
        self.num_stages: int = len(decoder_channels)
        self.upsample_mode = upsample
        self.use_attention_gates = use_attention_gates
        self.use_checkpoint = use_checkpoint

        up_convs = []
        up_norms = []
        gates = []
        conv_blocks = []

        for i in range(self.num_stages):
            # Stage i merges with skips[i]. Its INPUT width (before upsampling) is the
            # bottleneck's width on the coarsest stage, or the previous (coarser) stage's
            # output width otherwise -- "previous" in coarse-to-fine walk order, i.e.
            # decoder_channels[i + 1].
            in_channels = skip_channels[-1] if i == self.num_stages - 1 else decoder_channels[i + 1]
            out_channels = decoder_channels[i]

            if upsample == "deconv":
                up_convs.append(
                    nn.ConvTranspose3d(
                        in_channels, out_channels, kernel_size=2, stride=2, bias=False
                    )
                )
            else:
                # Kernel 1x1x1: the trilinear interpolation above already did the spatial
                # 2x upsampling; this conv only changes channel width.
                up_convs.append(nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False))
            up_norms.append(nn.GroupNorm(num_groups, out_channels))

            if use_attention_gates:
                gates.append(AttentionGate(skip_channels[i], out_channels))

            merge_channels = out_channels + skip_channels[i]
            blocks = [ResidualBlock(merge_channels, out_channels, 1, num_groups, dropout)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResidualBlock(out_channels, out_channels, 1, num_groups, dropout))
            conv_blocks.append(nn.Sequential(*blocks))

        self.up_convs = nn.ModuleList(up_convs)
        self.up_norms = nn.ModuleList(up_norms)
        self.attention_gates = nn.ModuleList(gates) if use_attention_gates else None
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, skips: list[Tensor]) -> list[Tensor]:
        """Runs the decoder, returning fine-to-coarse features (not logits, not heads).

        Args:
            skips: Fine-to-coarse skip pyramid, length `len(skip_channels)`. Element 0 is
                the full-resolution skip; the last element is the bottleneck. Channel
                counts must match `self.skip_channels` exactly.

        Returns:
            A list of length `self.num_stages` (= `len(skips) - 1`), ordered fine (index 0,
            full resolution) to coarse. Element `i` has shape
            `(B, out_channels[i], D_i, H_i, W_i)` where `(D_i, H_i, W_i)` is `skips[i]`'s
            spatial shape. The bottleneck itself is NOT included — it was an input, not a
            decoder output. Attach segmentation / deep-supervision heads to as many of
            these as wanted; this module does not do that itself.

        Raises:
            ValueError: If `len(skips) != len(self.skip_channels)`, or if any skip's
                channel count disagrees with what this decoder was built for.
        """
        if len(skips) != len(self.skip_channels):
            raise ValueError(
                f"forward() got {len(skips)} skips, but this decoder was built for "
                f"{len(self.skip_channels)} (skip_channels={self.skip_channels})."
            )
        for i, (skip, expected_c) in enumerate(zip(skips, self.skip_channels, strict=True)):
            if skip.shape[1] != expected_c:
                raise ValueError(
                    f"skips[{i}] has {skip.shape[1]} channels, expected {expected_c} "
                    f"(this decoder was built for skip_channels={self.skip_channels})."
                )

        x = skips[-1]  # start from the bottleneck
        outputs: list[Tensor] = []

        # Coarse-to-fine walk: stage num_stages-1 first (nearest the bottleneck), stage 0
        # last (full resolution). Collected in this order, then reversed once at the end.
        for i in range(self.num_stages - 1, -1, -1):
            if self.upsample_mode == "interp":
                x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
            x = self.up_convs[i](x)
            x = self.up_norms[i](x)

            target_shape = tuple(skips[i].shape[2:])
            x = _match_spatial(x, target_shape)

            skip = skips[i]
            if self.use_attention_gates:
                skip = self.attention_gates[i](skip, x)

            merged = torch.cat([x, skip], dim=1)

            # Same checkpointing guard as CNNEncoder.forward and AdaptiveGatedFusion._fuse:
            # torch.utils.checkpoint warns and silently returns no gradient at all when
            # nothing in its input requires grad, which is exactly the eval/no_grad case.
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(self.conv_blocks[i], merged, use_reentrant=False)
            else:
                x = self.conv_blocks[i](merged)

            outputs.append(x)

        return list(reversed(outputs))
