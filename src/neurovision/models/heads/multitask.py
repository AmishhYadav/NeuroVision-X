"""Bundles all of NeuroVision-X's output heads (segmentation + optional auxiliary heads)
behind one module and one return type.

`MultiTaskHead` owns:

- One `neurovision.models.heads.segmentation.SegmentationHead` per deep-supervision level,
  attached fine-to-coarse to the decoder's own fine-to-coarse features.
- Zero, one, or two `neurovision.models.heads.auxiliary.AuxiliaryHead` instances (confidence,
  boundary), each attached ONLY to the full-resolution decoder feature -- see
  `auxiliary.py`'s module docstring for why.

## Activation-memory cost of the auxiliary heads

Each `AuxiliaryHead` keeps roughly 4 feature-map-sized tensors alive for backward (the two
conv inputs, the norm output, the activation output) at the decoder's FULL-RESOLUTION width
-- the widest, most expensive resolution in the whole network. At a 96^3 patch, 4 patches per
step (the project default: `batch_size: 1` x `samples_per_volume: 4`), fp16, with
`decoder_channels[0] = 32` (production width) and hidden width 16: that is roughly 0.2 GB per
auxiliary head, ~0.45 GB for both together. Against the 16 GB T4 VRAM budget, where the full
model already measures ~4.5 GB of fp32 activations (see `neurovision.py`'s measured-memory
note), this is real but affordable -- it does not change the `batch_size: 1` recommendation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn

from neurovision.models.heads.auxiliary import AuxiliaryHead
from neurovision.models.heads.segmentation import SegmentationHead

logger = logging.getLogger(__name__)


@dataclass
class MultiTaskOutput:
    """The result of one `MultiTaskHead.forward` call.

    A plain `@dataclass`, deliberately NOT a `NamedTuple`. A `NamedTuple` is iterable and
    would silently unpack -- `seg, confidence, boundary = model(x)` -- into code that expects
    a plain list of logits tensors (e.g. `DeepSupervisionLoss`, which iterates its input
    expecting each entry to be a tensor of logits, not a 3-item output bundle). A dataclass
    has no such iteration protocol, so a caller written for the wrong return type raises
    immediately instead of silently computing something wrong.

    Attributes:
        seg: Segmentation logits, fine-to-coarse, length `deep_supervision_levels`. Element
            `i` has shape `(B, out_channels, D_i, H_i, W_i)` at decoder stage `i`'s
            resolution.
        confidence: Full-resolution confidence logits, shape `(B, out_channels, D, H, W)`,
            or `None` if the confidence head is disabled.
        boundary: Full-resolution boundary logits, shape `(B, out_channels, D, H, W)`, or
            `None` if the boundary head is disabled.
    """

    seg: list[Tensor]
    confidence: Tensor | None
    boundary: Tensor | None


class MultiTaskHead(nn.Module):
    """Owns every output head NeuroVisionX attaches to the decoder.

    Args:
        decoder_channels: The decoder's `out_channels`, fine-to-coarse (see
            `neurovision.models.decoder.unet_decoder.UNetDecoder.out_channels`).
        out_channels: Number of region channels for every head (segmentation and auxiliary
            alike). Always 3 for this project -- see `segmentation.py`'s module docstring.
        deep_supervision_levels: Number of decoder stages, starting from full resolution
            (index 0), that get a `SegmentationHead`. Must be in `1 ..
            len(decoder_channels)`.
        seg_dropout: `Dropout3d` probability inside every `SegmentationHead`.
        confidence: If True, build a confidence `AuxiliaryHead`.
        boundary: If True, build a boundary `AuxiliaryHead`.
        confidence_hidden_channels: Hidden width of the confidence head (`None` -> the head's
            own default; see `AuxiliaryHead`).
        confidence_num_groups: GroupNorm group count inside the confidence head.
        confidence_dropout: `Dropout3d` probability inside the confidence head.
        boundary_hidden_channels: Hidden width of the boundary head.
        boundary_num_groups: GroupNorm group count inside the boundary head.
        boundary_dropout: `Dropout3d` probability inside the boundary head.

    The two auxiliary heads are configured INDEPENDENTLY, even though `configs/model/
    neurovision.yaml` currently gives them identical values. They are separate sub-blocks in
    the config, so a shared parameter triple here would silently ignore one of them -- setting
    `model.head.boundary.hidden_channels: 64` would build a head of some other width with no
    error anywhere. Config keys that are read but not honoured are the exact failure mode
    CLAUDE.md's "config objects passed in, never global state" rule exists to prevent.

    Raises:
        ValueError: If `deep_supervision_levels` is not in `1 .. len(decoder_channels)`.
    """

    def __init__(
        self,
        decoder_channels: Sequence[int],
        out_channels: int,
        deep_supervision_levels: int = 1,
        seg_dropout: float = 0.0,
        confidence: bool = False,
        boundary: bool = False,
        confidence_hidden_channels: int | None = None,
        confidence_num_groups: int = 8,
        confidence_dropout: float = 0.0,
        boundary_hidden_channels: int | None = None,
        boundary_num_groups: int = 8,
        boundary_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        decoder_channels = list(decoder_channels)

        if not (1 <= deep_supervision_levels <= len(decoder_channels)):
            raise ValueError(
                f"deep_supervision_levels ({deep_supervision_levels}) must be between 1 and "
                f"len(decoder_channels) ({len(decoder_channels)}) inclusive -- one "
                f"segmentation head per supervised decoder stage, starting from the "
                f"full-resolution stage (index 0)."
            )

        self.deep_supervision_levels = deep_supervision_levels

        # Head i reads decoder_channels[i] -- decoder features are fine-to-coarse, so head 0
        # is the full-resolution head and heads 1.. read progressively coarser features.
        self.seg_heads = nn.ModuleList(
            [
                SegmentationHead(decoder_channels[i], out_channels, seg_dropout)
                for i in range(deep_supervision_levels)
            ]
        )

        # Both auxiliary heads, when present, read ONLY the full-resolution feature
        # (decoder_channels[0]) -- see this module's and auxiliary.py's docstrings for why.
        self.confidence = (
            AuxiliaryHead(
                decoder_channels[0],
                out_channels,
                hidden_channels=confidence_hidden_channels,
                num_groups=confidence_num_groups,
                dropout=confidence_dropout,
            )
            if confidence
            else None
        )
        self.boundary = (
            AuxiliaryHead(
                decoder_channels[0],
                out_channels,
                hidden_channels=boundary_hidden_channels,
                num_groups=boundary_num_groups,
                dropout=boundary_dropout,
            )
            if boundary
            else None
        )

    @property
    def has_auxiliary(self) -> bool:
        """True if either the confidence or the boundary head is enabled."""
        return self.confidence is not None or self.boundary is not None

    def forward(self, feats: Sequence[Tensor]) -> MultiTaskOutput:
        """Applies every head to the decoder's fine-to-coarse feature list.

        Args:
            feats: Decoder features, fine-to-coarse (index 0 is full resolution), length at
                least `deep_supervision_levels`. Typically
                `neurovision.models.decoder.unet_decoder.UNetDecoder.forward`'s output.

        Returns:
            A `MultiTaskOutput` with `seg[i] = self.seg_heads[i](feats[i])` for `i in
            range(deep_supervision_levels)`, and `confidence` / `boundary` computed from
            `feats[0]` only (or `None` if disabled).

        Raises:
            ValueError: If `len(feats) < deep_supervision_levels`.
        """
        if len(feats) < self.deep_supervision_levels:
            raise ValueError(
                f"len(feats) ({len(feats)}) is smaller than deep_supervision_levels "
                f"({self.deep_supervision_levels}) -- not enough decoder features to attach "
                f"every segmentation head."
            )

        seg = [head(feats[i]) for i, head in enumerate(self.seg_heads)]
        confidence = self.confidence(feats[0]) if self.confidence is not None else None
        boundary = self.boundary(feats[0]) if self.boundary is not None else None

        return MultiTaskOutput(seg=seg, confidence=confidence, boundary=boundary)
