"""Auxiliary output heads: confidence and boundary.

One class, `AuxiliaryHead`, used for BOTH the confidence head and the boundary head.
Deliberately not split into two identical classes -- they are structurally identical (same
conv -> norm -> activation -> dropout -> conv body, same output shape); what makes one a
"confidence head" and the other a "boundary head" is entirely what it is trained against,
and that target construction lives in the loss (`neurovision.losses.multitask`), not here.
Do not "fix" this by writing `ConfidenceHead` and `BoundaryHead` subclasses -- there is
nothing left to differentiate once the target is factored out.

## What a confidence head is for

It predicts, per voxel and per region (so `out_channels == 3`, matching `SegmentationHead`'s
ET/TC/WT layout), whether the segmentation head's OWN prediction at that voxel is correct.
Its target is built at loss time from the segmentation output compared against the ground
truth, and that target is fully DETACHED -- the confidence head never receives a gradient
that flows back through the segmentation logits it is grading. It is a read-out of the
network's own correctness, not a second, independent segmenter competing with the first.

## What a boundary head is for

It predicts the 1-voxel shell at each region's surface, derived from the ground-truth label
by a morphological gradient (dilate minus erode, or similar) computed inside the loss -- no
change to the dataloader or preprocessing is needed. This is a standard auxiliary task: it
forces the shared decoder features to stay discriminative right at the margin between tumor
and healthy tissue, which is exactly what this project's boundary-accuracy (HD95) claim rests
on. The segmentation head can get high Dice while being sloppy about the exact boundary
voxel; a head that is explicitly graded on the boundary shell cannot.

## Why both attach only to the full-resolution decoder feature

Neither aux head is duplicated across deep-supervision levels the way `SegmentationHead` is.
Both concepts are only meaningful at full resolution: a "correctness map" downsampled to
stride 4 is not a correctness map (it has been pooled across voxels that disagree), and a
morphological 1-voxel boundary shell is mostly aliased away once the image has already been
downsampled 4x -- there may be no boundary voxels left to predict. So `MultiTaskHead` below
reads `feats[0]` (the decoder's full-resolution output) for both, regardless of how many
segmentation heads deep supervision attaches elsewhere.
"""

from __future__ import annotations

import logging

from torch import Tensor, nn

logger = logging.getLogger(__name__)


class AuxiliaryHead(nn.Module):
    """Maps a decoder feature to per-region auxiliary logits (confidence or boundary).

    Body: `Conv3d(3x3x3) -> GroupNorm -> LeakyReLU -> Dropout3d -> Conv3d(1x1x1)`. Unlike
    `SegmentationHead`'s single 1x1x1 convolution, this head has one hidden 3x3x3 layer --
    both auxiliary tasks (per-voxel correctness, boundary shells) depend on local spatial
    context beyond a single voxel, which a 1x1x1 projection cannot see.

    Args:
        in_channels: Width of the decoder feature this head reads.
        out_channels: Number of region channels. Always 3 for this project, matching
            `SegmentationHead` (ET/TC/WT) -- see that module's docstring for why 4 is a trap.
        hidden_channels: Width of the intermediate 3x3x3 conv. `None` (the default) picks
            `max(in_channels // 2, out_channels, 8)`: narrower than the input (this is a
            lightweight read-out head, not another encoder stage), never narrower than
            `out_channels` (would create an information bottleneck below the output width),
            and never below 8 (keeps GroupNorm meaningful at the default `num_groups=8`).
        num_groups: GroupNorm group count. `hidden_channels` must be divisible by this.
        dropout: `Dropout3d` probability applied before the final 1x1x1 conv.

    Raises:
        ValueError: If the resolved hidden width is not divisible by `num_groups`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        num_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = max(in_channels // 2, out_channels, 8)

        if hidden_channels % num_groups != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) is not divisible by num_groups "
                f"({num_groups}); GroupNorm requires num_channels % num_groups == 0. Either "
                f"change num_groups to a divisor of {hidden_channels}, or set "
                f"hidden_channels explicitly to a multiple of num_groups."
            )

        # bias=False: GroupNorm immediately follows and applies its own learned per-channel
        # shift, so a conv bias here would be redundant -- same reasoning as
        # neurovision.models.encoders.cnn.ResidualBlock.
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, hidden_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # Dropout3d zeros whole channels rather than individual voxels -- the correct variant
        # for convolutional feature maps, same reasoning as in cnn.py and segmentation.py.
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes the two convs and the GroupNorm.

        Kaiming init matched to LeakyReLU on the 3x3x3 conv, same as
        `neurovision.models.encoders.cnn.CNNEncoder._init_weights`. The final 1x1x1 conv's
        bias is zero-initialized so every output channel starts at `sigmoid(0) = 0.5`, i.e.
        maximal uncertainty and no prior baked in -- mirrors `SegmentationHead`'s zero-bias
        init, and for the same reason: an informative prior here would be a starting point
        the model did not learn, which is an unhelpful confound for a project whose headline
        claim is calibration.
        """
        nn.init.kaiming_normal_(
            self.conv1.weight, a=0.01, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Projects a decoder feature to auxiliary logits.

        Args:
            x: Decoder feature, shape `(B, in_channels, D, H, W)`.

        Returns:
            Raw logits (no sigmoid -- applied by the loss and by
            `neurovision.inference.postprocess`, never here), shape
            `(B, out_channels, D, H, W)`, spatially unchanged.
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        return self.conv2(out)
