"""Swin Transformer encoder producing a multi-scale feature pyramid.

This is the other half of the dual encoder (the sibling is the residual CNN
in `neurovision.models.encoders.cnn`); its pyramid feeds the adaptive gated
cross-attention fusion module.

## Why this wraps `SwinTransformer`, not `SwinUNETR`

MONAI's `SwinUNETR` is a complete encoder-decoder segmentation network: a
Swin Transformer backbone (`swinViT`) plus its own convolutional decoder and
skip-connection blocks. We only want the backbone's feature pyramid — this
project has its own fusion module and its own decoder, so pulling in
SwinUNETR's decoder would mean building and immediately throwing away half of
it. `monai.networks.nets.swin_unetr.SwinTransformer` is exactly the encoder
half, so that is what this file wraps.

## The stride offset — the most important thing in this file

`SwinTransformer.patch_embed` merges every 2x2x2 block of input voxels into
one token BEFORE any attention runs, so the Swin pyramid starts at stride 2.
`CNNEncoder`'s pyramid, by contrast, starts at stride 1 (its level 0 is a
full-resolution stem). Swin cannot produce a stride-1 feature without
`patch_size=1`, and at a 96^3 patch that means self-attention over
96 * 96 * 96 = 884,736 tokens per window computation — an immediate OOM —
and it would also discard any ImageNet/SSL pretrained weights, which are
defined for patch_size=2.

The fix is on the CNN side, not here: `CNNEncoder` is configured with FIVE
levels (strides 1, 2, 4, 8, 16). Its levels 1..4 (strides 2, 4, 8, 16) align
with `SwinEncoder`'s levels 0..3 (also strides 2, 4, 8, 16) and are what the
fusion module actually fuses; CNN level 0 (stride 1) has no Swin counterpart
and passes straight to the decoder unfused. This is exactly what MONAI's own
SwinUNETR does internally — its `encoder1` is a plain convolution on the raw
input, never routed through the transformer, for the same reason.

## Why `num_levels=4` is the default rather than 5

At `feature_size=48`, the fifth stage (`layers4`, stride 32) alone accounts
for 6.02M of the branch's 8.06M parameters (75%). At a 96^3 input patch it
operates on a 3x3x3 = 27-token feature map, with `window_size=7` (343
positions) already larger than the map — so windowed attention there
degenerates to full attention over all 27 tokens, while the previous stage
(stride 16, 6x6x6 = 216 tokens) is already small enough to be effectively
global. The fifth stage buys little for its cost. Set `num_levels=5` to
build it back (e.g. to match a pretrained SwinUNETR checkpoint exactly).

## The shared `ceil` downsampling convention

Both branches downsample using `ceil(size / stride)`, not `floor`. On the
CNN side this falls out of "same"-padded stride-2 convolutions. On the Swin
side, MONAI's patch-merging downsample module pads any odd spatial
dimension up to even before merging 2x2x2 blocks, which has the same
`ceil` effect. Because both branches share this convention, level alignment
(see above) holds not just at the 96^3 training patch but at any input
shape, including odd and anisotropic ones.

## `use_checkpoint=True` by default

Consistent with `configs/model/swinunetr.yaml`: gradient checkpointing
recomputes attention activations during backward instead of storing them
from the forward pass, trading ~20-30% extra step time for a large
reduction in activation memory. That trade is worth it against the 16 GB T4
VRAM budget.

## Minimum input size

The wrapper downsamples by `2 ** num_levels` overall (patch embed 2x, then
`num_levels - 1` more 2x merging stages), so every spatial axis of the input
must be at least `2 ** num_levels` voxels: 16 at the default `num_levels=4`,
32 at `num_levels=5`. Tests here use 64^3, which clears both. The 96^3
training patch clears both by a wide margin.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import monai
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def _to_tuple(value: Any) -> tuple:
    """Converts a sequence-valued config field to a plain tuple.

    Hydra hands sequence fields over as OmegaConf `ListConfig` objects, which
    are not `list`/`tuple` instances. This normalizes any sequence to a plain
    tuple before it is used to build the encoder. Local copy of the same
    helper in `neurovision.models.encoders.cnn` — kept separate rather than
    shared so each encoder file stays self-contained and independently
    reviewable.

    Args:
        value: A `ListConfig`, list, tuple, or other sequence.

    Returns:
        A plain `tuple` with the same elements.
    """
    return tuple(value)


class SwinEncoder(nn.Module):
    """Swin Transformer encoder that outputs a fine-to-coarse feature pyramid.

    Wraps `monai.networks.nets.swin_unetr.SwinTransformer` (the "SwinViT"
    backbone used inside SwinUNETR) and reimplements its forward pass with an
    early exit, so that stages beyond `num_levels` are never built or run.
    See the module docstring for why this wraps `SwinTransformer` rather than
    the full `SwinUNETR`, and for the stride-offset contract with
    `CNNEncoder` that callers (the fusion module) depend on.

    Args:
        in_channels: Number of input channels (MRI modalities).
        feature_size: Base embedding width. Level `i`'s output has
            `feature_size * 2**i` channels. Must be divisible by the largest
            entry of `num_heads` for attention head widths to be integral
            (MONAI does not check this at construction, it fails inside
            attention; keep `feature_size` a multiple of 12 as SwinUNETR's
            own docs recommend).
        depths: Number of Swin Transformer blocks in each of MONAI's four
            hardcoded stages. Must have length 4.
        num_heads: Attention heads in each of the four stages. Must have
            length 4.
        window_size: Local attention window size, one scalar shared by every
            spatial axis. Expanded to a 3-tuple internally.
        patch_size: Patch embedding size, one scalar shared by every spatial
            axis. Expanded to a 3-tuple internally. `patch_size=2` is what
            gives the stride-2 first level described in the module
            docstring; do not lower it to 1 (see there for why).
        num_levels: Number of pyramid levels to build and return, 1 to 5.
            Levels beyond this are never constructed. See the module
            docstring for why the default omits the fifth level.
        drop_rate: Standard dropout after the MLP block in each Swin block.
        attn_drop_rate: Dropout applied to attention weights.
        dropout_path_rate: Stochastic depth rate (randomly skips whole Swin
            blocks during training). Named `dropout_path_rate` here to match
            `configs/model/swinunetr.yaml`'s existing key, even though
            `SwinTransformer` itself calls the identical argument
            `drop_path_rate` — mapped explicitly in `__init__`, see the
            comment there.
        use_checkpoint: If True, wrap each Swin stage in gradient
            checkpointing during training to trade compute for activation
            memory.
        normalize: If True, apply a `LayerNorm` to each pyramid level's
            output before returning it (MONAI's `proj_out`). This does not
            affect the internal computation, only the returned features.

    Raises:
        ValueError: If `num_levels` is not in `[1, 5]`, or if `depths` or
            `num_heads` does not have length 4.
        RuntimeError: If the installed MONAI version's `SwinTransformer`
            does not expose the internal attributes this wrapper's `forward`
            depends on (see the defensive guard at the end of `__init__`).
    """

    def __init__(
        self,
        in_channels: int,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        patch_size: int = 2,
        num_levels: int = 4,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        depths = list(depths)
        num_heads = list(num_heads)

        if not (1 <= num_levels <= 5):
            raise ValueError(
                f"num_levels = {num_levels} is out of range; SwinTransformer produces at "
                f"most 5 pyramid levels (one from patch_embed, one each from layers1..4), "
                f"so num_levels must be between 1 and 5."
            )

        # MONAI's SwinTransformer hardcodes exactly four stages -- self.layers1
        # through self.layers4 -- in both __init__ and forward. Any other
        # length either crashes deep inside MONAI (index error) or silently
        # mis-builds a stage with a mismatched depths/num_heads entry, so this
        # is checked here with a message that says why, rather than letting
        # MONAI's stack trace be the only signal.
        if len(depths) != 4:
            raise ValueError(
                f"depths has length {len(depths)}, but MONAI's SwinTransformer hardcodes "
                f"exactly four stages (layers1..layers4); depths must have length 4."
            )
        if len(num_heads) != 4:
            raise ValueError(
                f"num_heads has length {len(num_heads)}, but MONAI's SwinTransformer "
                f"hardcodes exactly four stages (layers1..layers4); num_heads must have "
                f"length 4."
            )

        self.num_levels: int = num_levels
        self.normalize = normalize
        # feature_size * 2**i / 2**(i+1): level i's channel width / stride,
        # for i in [0, num_levels). Matches CNNEncoder's out_channels/strides
        # attribute names so the fusion module can treat both branches the
        # same way.
        self.out_channels: list[int] = [feature_size * 2**i for i in range(num_levels)]
        self.strides: list[int] = [2 ** (i + 1) for i in range(num_levels)]

        self.swin = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            # window_size and patch_size are scalars in our API but
            # SwinTransformer wants a per-axis tuple; ensure_tuple_rep is the
            # same helper MONAI's own SwinUNETR uses to do this expansion.
            window_size=ensure_tuple_rep(window_size, 3),
            patch_size=ensure_tuple_rep(patch_size, 3),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            # Name mismatch, deliberate: our constructor parameter is
            # `dropout_path_rate` (matching the existing
            # configs/model/swinunetr.yaml key), but SwinTransformer's
            # parameter for the identical thing (stochastic depth rate) is
            # spelled `drop_path_rate`. Mapped explicitly here rather than
            # relying on keyword-argument order.
            drop_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            # Everything below mirrors SwinUNETR.__init__'s own construction
            # of self.swinViT (monai/networks/nets/swin_unetr.py), for
            # arguments this wrapper does not expose:
            mlp_ratio=4.0,  # SwinUNETR's default, never overridden there either
            qkv_bias=True,  # SwinUNETR's default
            norm_layer=nn.LayerNorm,  # SwinUNETR's default
            patch_norm=False,  # SwinUNETR's default (no norm after patch embed)
            spatial_dims=3,  # this project is 3D-only, never configurable
            downsample="merging",  # SwinUNETR's default downsample module string
            use_v2=False,  # forced: this wrapper does not support SwinUNETR v2
        )

        # Defensive attribute guard: this wrapper reimplements
        # SwinTransformer.forward (see below) rather than calling it, so it
        # depends on MONAI's internal attribute names. If a MONAI upgrade
        # renames or restructures any of these, fail loudly at construction
        # time with a message that says exactly what to go re-check, instead
        # of failing confusingly (or silently returning wrong features) the
        # first time forward() runs.
        #
        # This MUST run BEFORE the stage reclamation below. Reclamation uses
        # setattr to install empty ModuleLists over layers{num_levels}..4, so
        # running the guard afterwards would find those attributes present
        # because WE just created them -- masking their removal by a future
        # MONAI version for exactly the stages we care least about noticing.
        required_attrs = (
            "patch_embed",
            "pos_drop",
            "proj_out",
            "layers1",
            "layers2",
            "layers3",
            "layers4",
        )
        missing = [attr for attr in required_attrs if not hasattr(self.swin, attr)]
        if missing:
            raise RuntimeError(
                f"monai.networks.nets.swin_unetr.SwinTransformer (installed MONAI version "
                f"{monai.__version__}) is missing attribute(s) {missing}, which "
                f"neurovision.models.encoders.swin.SwinEncoder.forward depends on. This "
                f"wrapper reimplements SwinTransformer.forward with an early exit rather "
                f"than calling it directly (see the module docstring for why); re-check it "
                f"against the installed monai/networks/nets/swin_unetr.py before using this "
                f"encoder."
            )

        # Reclaim unused stages. Level 0 comes from patch_embed (no `layers`
        # module involved); level i > 0 comes from self.swin.layers{i}. So
        # producing num_levels outputs only needs layers1..layers{num_levels
        # - 1}; anything from layers{num_levels} onward is never called by
        # this wrapper's forward and would just sit there holding parameters
        # (and, once training starts, Adam's two moment buffers for each).
        # At feature_size=48, layers4 alone is 75% of the branch's
        # parameters, so at num_levels=4 this discards the majority of what
        # SwinTransformer would otherwise allocate.
        for level in range(num_levels, 5):
            setattr(self.swin, f"layers{level}", nn.ModuleList())

    def forward(self, x: Tensor) -> list[Tensor]:
        """Runs the encoder, returning a fine-to-coarse feature pyramid.

        This is a deliberate reimplementation of
        `SwinTransformer.forward`, stopping after `num_levels` outputs
        instead of unconditionally computing all five. Calling
        `self.swin(x, normalize)` directly is not an option: MONAI's own
        forward always runs every stage through `layers4`, which would
        defeat the point of reclaiming that stage above. The `.contiguous()`
        calls are kept in exactly the same places MONAI has them -- they
        matter for the window-partition reshape inside each stage -- and the
        `use_v2` branches from MONAI's forward are omitted since `use_v2` is
        always False here. `tests/test_swin_encoder.py` has a direct
        equivalence test against MONAI's own forward, which is what keeps
        this in sync across MONAI upgrades.

        Args:
            x: Input volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            A list of length `num_levels`, ordered level 0 (finest, stride
            2) to level `num_levels - 1` (coarsest). Element `i` has shape
            `(B, feature_size * 2**i, ceil(D / 2**(i+1)), ceil(H / 2**(i+1)),
            ceil(W / 2**(i+1)))`.
        """
        outputs: list[Tensor] = []

        x0 = self.swin.patch_embed(x)
        x0 = self.swin.pos_drop(x0)
        outputs.append(self.swin.proj_out(x0, self.normalize))
        if self.num_levels == 1:
            return outputs

        x1 = self.swin.layers1[0](x0.contiguous())
        outputs.append(self.swin.proj_out(x1, self.normalize))
        if self.num_levels == 2:
            return outputs

        x2 = self.swin.layers2[0](x1.contiguous())
        outputs.append(self.swin.proj_out(x2, self.normalize))
        if self.num_levels == 3:
            return outputs

        x3 = self.swin.layers3[0](x2.contiguous())
        outputs.append(self.swin.proj_out(x3, self.normalize))
        if self.num_levels == 4:
            return outputs

        x4 = self.swin.layers4[0](x3.contiguous())
        outputs.append(self.swin.proj_out(x4, self.normalize))
        return outputs


def build_swin_encoder(cfg: Any) -> SwinEncoder:
    """Builds the Swin encoder from `cfg.model.encoder.swin`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.data.in_channels`
            and `cfg.model.encoder.swin` with keys `feature_size`, `depths`,
            `num_heads`, `window_size`, `patch_size`, `num_levels`,
            `drop_rate`, `attn_drop_rate`, `dropout_path_rate`,
            `use_checkpoint`, `normalize`.

    Returns:
        A constructed `SwinEncoder`.
    """
    swin_cfg = cfg.model.encoder.swin

    encoder = SwinEncoder(
        in_channels=cfg.data.in_channels,
        feature_size=swin_cfg.feature_size,
        depths=_to_tuple(swin_cfg.depths),
        num_heads=_to_tuple(swin_cfg.num_heads),
        window_size=swin_cfg.window_size,
        patch_size=swin_cfg.patch_size,
        num_levels=swin_cfg.num_levels,
        drop_rate=swin_cfg.drop_rate,
        attn_drop_rate=swin_cfg.attn_drop_rate,
        dropout_path_rate=swin_cfg.dropout_path_rate,
        use_checkpoint=swin_cfg.use_checkpoint,
        normalize=swin_cfg.normalize,
    )

    logger.info(
        "Built SwinEncoder: %d levels, channels=%s, strides=%s",
        encoder.num_levels,
        encoder.out_channels,
        encoder.strides,
    )

    return encoder
