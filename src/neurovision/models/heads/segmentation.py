"""Segmentation output head.

The first of the three heads the project's architecture calls for
(segmentation, confidence, boundary); the other two do not exist yet. Kept in
its own module rather than folded into the decoder because deep supervision
needs one head instance per supervised resolution, and because the confidence
and boundary heads will attach to the same decoder features later.

## Why the output is 3 channels and never 4

The heads predict the three *overlapping* BraTS regions (ET, TC, WT), one
sigmoid channel each. 4 is a tempting wrong answer twice over: it is the raw
class count `{0, 1, 2, 3}` after preprocessing's label remap, and it is also
the modality count (`in_channels`). A 4-channel head makes
`BCEWithLogitsLoss` raise against the 3-channel target on the very first
step. `out_channels` is always `${data.num_classes}` = 3.

## Why there is no activation here

The head emits raw logits. Sigmoid is applied inside the loss
(`BCEWithLogitsLoss` / `DiceLoss(sigmoid=True)`) for numerical stability, and
inside `neurovision.inference.postprocess` at inference time. A sigmoid here
would double-apply it.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

logger = logging.getLogger(__name__)


class SegmentationHead(nn.Module):
    """Maps decoder features to per-region segmentation logits.

    A single 1x1x1 convolution, optionally preceded by `Dropout3d`. There is
    deliberately no normalization or activation: this is the last layer of the
    network and its output goes straight into the loss.

    Args:
        in_channels: Width of the decoder feature this head reads.
        out_channels: Number of region channels. Always 3 for this project —
            see the module docstring for why 4 is a trap.
        dropout: `Dropout3d` probability applied before the convolution. Not
            only regularization: MC-dropout uncertainty estimation at
            inference re-runs the network with dropout left active, so a
            model trained at `p=0.0` has no predictive spread to measure
            later. Keep it non-zero if the uncertainty claim matters.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Dropout3d zeros whole channels rather than individual voxels — the
        # correct variant for convolutional feature maps, same reasoning as
        # in neurovision.models.encoders.cnn.ResidualBlock.
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)

        # Zero bias: every region starts at sigmoid(0) = 0.5, i.e. maximal
        # uncertainty and no prior on tumor extent. A negative bias prior
        # (RetinaNet-style, matching the true foreground fraction) would cut
        # the initial loss, but it also hands the model a calibrated-looking
        # starting point it did not learn — an unhelpful confound for a
        # project whose headline claim is calibration.
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Projects features to region logits.

        Args:
            x: Decoder feature, shape `(B, in_channels, D, H, W)`.

        Returns:
            Raw logits (no sigmoid), shape `(B, out_channels, D, H, W)`.
        """
        return self.conv(self.dropout(x))
