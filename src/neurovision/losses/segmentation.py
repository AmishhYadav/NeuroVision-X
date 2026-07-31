"""Segmentation losses for the BraTS region-overlap setup.

BraTS predictions are three *overlapping* regions, one channel each:
channel 0 = ET (enhancing tumor), channel 1 = TC (tumor core),
channel 2 = WT (whole tumor). A voxel inside the enhancing tumor is
legitimately in ET, TC, and WT at once, so the three channels cannot compete
for a shared softmax probability budget the way ordinary multi-class
segmentation channels do.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch.nn.functional as F
from monai.losses import DiceLoss
from torch import Tensor, nn

from neurovision.losses.registry import register_loss

logger = logging.getLogger(__name__)


class DiceBCELoss(nn.Module):
    """Per-channel Dice + binary cross-entropy for overlapping regions.

    Deliberately NOT ``monai.losses.DiceCELoss``. That class picks its
    cross-entropy term by channel count, not by the ``sigmoid`` flag::

        ce_loss = self.ce(input, target) if input.shape[1] != 1 else self.bce(...)

    With our 3 region channels it therefore always applies
    ``nn.CrossEntropyLoss`` — a softmax across channels — even when
    constructed with ``sigmoid=True``. Softmax makes the channels compete for
    one shared probability budget, which is wrong here: every ET voxel is by
    definition also TC and WT, so the regions are nested, not mutually
    exclusive. The softmax term actively pushes them apart.

    Composing ``DiceLoss(sigmoid=True)`` with ``BCEWithLogitsLoss`` treats each
    channel independently, which is what multi-label region segmentation
    needs. Both halves are stock MONAI / PyTorch — no hand-rolled maths.
    A perfect prediction scores exactly 0.0 here; through ``DiceCELoss`` it
    does not.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = True,
        squared_pred: bool = False,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
    ) -> None:
        """Initializes the composed loss.

        Args:
            dice_weight: Multiplier on the Dice term.
            ce_weight: Multiplier on the binary cross-entropy term.
            include_background: Whether to include channel 0 in the Dice term.
                Must stay True for region targets — channel 0 is ET, a real
                foreground region, not background.
            squared_pred: Square the prediction in the Dice denominator.
            smooth_nr: Numerator smoothing. Keep non-zero: a correctly
                predicted *empty* channel would otherwise score Dice 0 and be
                punished as hard as a wrong one.
            smooth_dr: Denominator smoothing.
        """
        super().__init__()
        self.dice = DiceLoss(
            sigmoid=True,
            include_background=include_background,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Computes the weighted Dice + BCE loss.

        Args:
            input: Raw logits, shape `(B, C, D, H, W)`.
            target: Binary region masks, same shape.

        Returns:
            Scalar loss.
        """
        return self.dice_weight * self.dice(input, target) + self.ce_weight * self.bce(
            input, target.float()
        )


class DeepSupervisionLoss(nn.Module):
    """Applies a base loss to several decoder outputs and sums them, weighted.

    A decoder with deep supervision emits several outputs: the final
    full-resolution one, plus lower-resolution ones from earlier decoder
    stages. Supervising the intermediate outputs gives gradient signal
    directly to early decoder layers instead of only through backprop from
    the final layer.
    """

    def __init__(self, base_loss: nn.Module, weights: Sequence[float] | None = None) -> None:
        """Initializes the deep-supervision wrapper.

        Args:
            base_loss: The loss applied at every resolution (e.g. a
                ``DiceCELoss``). Stored as a submodule so ``.to(device)`` and
                ``.train()``/``.eval()`` propagate through it.
            weights: Per-output weights, highest resolution first. If None,
                halving weights ``1, 0.5, 0.25, ...`` are generated to match
                the number of outputs seen at ``forward`` time. Whatever is
                supplied (explicit or generated) is normalized to sum to 1.0
                inside ``forward``, so a single output always reduces
                exactly to the plain loss (weight becomes 1.0) and the loss
                magnitude stays comparable whether deep supervision is on or
                off — a learning rate tuned with it off does not need
                retuning when it is turned on.

        Raises:
            ValueError: If any explicit weight is negative, or if the
                explicit weights sum to zero (would make every weight zero
                after normalization, i.e. no signal at all).
        """
        super().__init__()
        self.base_loss = base_loss
        if weights is not None:
            if any(w < 0 for w in weights):
                raise ValueError(f"Deep supervision weights must be non-negative, got {weights}.")
            if sum(weights) == 0:
                raise ValueError(
                    f"Deep supervision weights sum to zero, got {weights}. "
                    "Normalizing would divide by zero and leave no signal."
                )
        self._explicit_weights = list(weights) if weights is not None else None

    def forward(self, preds: Tensor | Sequence[Tensor], target: Tensor) -> Tensor:
        """Computes the weighted sum of the base loss over all outputs.

        Args:
            preds: Either a single logits ``Tensor`` of shape
                ``(B, C, D, H, W)``, or a sequence of such tensors ordered
                highest resolution first. Lower-resolution tensors do not
                need to match the target's spatial size.
            target: Binary float target of shape ``(B, C, D, H, W)``, at
                full resolution.

        Returns:
            A 0-dim (scalar) weighted-sum loss tensor.

        Raises:
            ValueError: If explicit weights were given at construction and
                their length does not match the number of outputs.
        """
        if isinstance(preds, Tensor):
            preds = [preds]

        n = len(preds)
        if self._explicit_weights is not None:
            if len(self._explicit_weights) != n:
                raise ValueError(
                    f"Deep supervision got {n} outputs but {len(self._explicit_weights)} "
                    "explicit weights were configured. Lengths must match."
                )
            weights = self._explicit_weights
        else:
            # Halving weights: the full-resolution output matters most, each
            # earlier (lower-resolution, coarser) stage matters half as much
            # as the one after it.
            weights = [0.5**i for i in range(n)]

        total_weight = sum(weights)
        normalized = [w / total_weight for w in weights]

        loss = preds[0].new_zeros(())
        for pred, weight in zip(preds, normalized):
            if pred.shape[2:] != target.shape[2:]:
                # Upsample the prediction to the target's resolution rather
                # than downsampling the target to the prediction's
                # resolution (as nnU-Net does). Downsampling the target
                # would degrade an exact binary mask through interpolation;
                # upsampling the prediction keeps the target exact, at the
                # cost of extra memory for the interpolated tensor — worth
                # watching against the 16 GB VRAM budget if deep supervision
                # is enabled with many low-resolution outputs.
                pred = F.interpolate(
                    pred, size=target.shape[2:], mode="trilinear", align_corners=False
                )
            loss = loss + weight * self.base_loss(pred, target)
        return loss


@register_loss("dice_ce")
def build_dice_ce(cfg: Any) -> nn.Module:
    """Builds the combined Dice + cross-entropy loss from config.

    Reads ``cfg.training.loss``. Optionally wraps the result in
    ``DeepSupervisionLoss`` when ``cfg.training.loss.deep_supervision.enabled``
    is true.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A ``DiceBCELoss``, or a ``DeepSupervisionLoss`` wrapping one if deep
        supervision is enabled.

    Raises:
        ValueError: If both ``sigmoid`` and ``softmax`` are requested, or if
            ``softmax`` is requested at all — the overlapping-region target
            makes softmax semantically wrong here, so it fails loudly rather
            than training something subtly incorrect.
    """
    loss_cfg = cfg.training.loss

    if loss_cfg.sigmoid and loss_cfg.softmax:
        raise ValueError(
            "training.loss.sigmoid and training.loss.softmax cannot both be true. "
            "BraTS regions overlap, so this config must use sigmoid=True, softmax=False."
        )
    if loss_cfg.softmax:
        raise ValueError(
            "training.loss.softmax=true is not supported for BraTS region targets. "
            "ET, TC and WT are nested (every ET voxel is also TC and WT), so the "
            "channels must not compete for a shared softmax budget. Use sigmoid=true."
        )

    base_loss = DiceBCELoss(
        dice_weight=loss_cfg.dice_weight,
        ce_weight=loss_cfg.ce_weight,
        # `include_background` means "include channel 0 in the Dice term".
        # Here channel 0 is ET (enhancing tumor), not background — this is
        # region targets, not a softmax one-hot label map. Leaving this
        # False (the habit from softmax setups) would silently drop the
        # hardest, most clinically important region from the Dice loss.
        include_background=loss_cfg.include_background,
        squared_pred=loss_cfg.squared_pred,
        smooth_nr=loss_cfg.smooth_nr,
        smooth_dr=loss_cfg.smooth_dr,
    )

    ds_cfg = loss_cfg.deep_supervision
    if ds_cfg.enabled:
        logger.info("Wrapping dice_ce in DeepSupervisionLoss (weights=%s)", ds_cfg.weights)
        return DeepSupervisionLoss(base_loss, weights=ds_cfg.weights)

    return base_loss
