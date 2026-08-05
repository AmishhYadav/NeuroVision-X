"""Multi-task loss combining segmentation, boundary, confidence and (optional) surface terms.

`neurovision.models.heads.multitask.MultiTaskHead` bundles a segmentation head (one per
deep-supervision level) with two optional auxiliary heads: a **confidence** head that
predicts, per voxel, whether the segmentation head's own answer is right, and a **boundary**
head that predicts the tumor surface directly. `MultiTaskLoss` is the objective that trains
all of them together, weighted-summed into one scalar so `Trainer` can call `loss_fn(preds,
target).backward()` exactly as it does for the plain `dice_ce` loss.

Two of the four terms have a load-bearing `torch.no_grad()` around a derived target:

- The boundary target is a hard morphological shell of the ground-truth mask (see
  `morphological_boundary` below) -- it is a fixed function of the label, never of a
  prediction, so there is nothing to detach a gradient FROM.
- The confidence target is "did the segmentation head get this voxel right", which IS a
  function of a prediction (`preds.seg[0]`). That one is deliberately detached -- see
  `MultiTaskLoss.forward`'s confidence-term comment for why the detach is not optional.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neurovision.losses.registry import register_loss
from neurovision.losses.segmentation import DiceBCELoss, build_dice_ce

if TYPE_CHECKING:
    # Import ONLY for the type checker. neurovision.models.neurovision imports nothing from
    # neurovision.losses, and a runtime import here would create the reverse dependency and
    # make losses <-> models circular. Code against the attribute names documented on
    # MultiTaskOutput (seg / confidence / boundary), never against the class itself.
    from neurovision.models.heads.multitask import MultiTaskOutput

logger = logging.getLogger(__name__)


def morphological_boundary(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Extracts the surface shell of a binary mask by 3D morphological gradient.

    The shell is `dilate(mask) - erode(mask)`: a voxel is on the shell if it is inside the
    dilated mask but outside the eroded one, i.e. it sits within one dilation/erosion radius
    of the mask boundary. Both operations are implemented with plain `max_pool3d` --
    dilation is a max-pool directly, and erosion is a max-pool of the *inverted* mask,
    inverted back (`erode(x) = -maxpool(-x)`, the standard duality). This is deliberately NOT
    built from a distance transform: `monai`'s distance-transform utilities go through
    scipy on CPU and cuCIM/CuPy on CUDA, and that CuPy path is the exact backend that already
    failed on the Kaggle T4 image for HD95 (`CompileException: Thrust requires at least
    C++17`, see CLAUDE.md). Two `max_pool3d` calls are a couple of GPU kernels with no host
    round-trip, so this function is safe to call every training step.

    Args:
        mask: Binary float mask, values in `{0, 1}`, shape `(B, C, D, H, W)`.
        kernel_size: Side length of the cubic structuring element. Must be odd -- an even
            kernel needs asymmetric padding to keep the output the same size as the input,
            which would shift the shell by half a voxel relative to the mask it was derived
            from, silently offsetting every boundary training target from the image.

    Returns:
        A float tensor of the same shape as `mask`, 1.0 on shell voxels and 0.0 elsewhere.

    Raises:
        ValueError: If `kernel_size` is even.

    Note:
        `max_pool3d` pads with `-inf`. Measured, not assumed: this means erosion --
        `-max_pool3d(-mask, ...)` -- does NOT treat "outside the patch" as background. An
        `-inf` contribution can never win a max against a real neighbor value (0 for
        background, -1 for foreground in the negated erosion computation), so a padded
        position is always ignored when at least one real neighbor exists, which is every
        position here. Concretely: an all-ones `(1, 1, 8, 8, 8)` mask produces an EXACT
        all-zero shell (no border rim at all), and a mask filling `x in [0, 4)` of an 8-voxel
        axis produces a shell only at the real internal `x=3/4` transition, not at the `x=0`
        patch edge -- verified by hand, not by reading the docs. This is actually the
        behavior this project wants: a tumor cropped by the 96^3 patch edge is not really
        bounded there, it simply continues outside what this patch can see, so staying silent
        at a cut edge is correct rather than inventing a boundary that is really just where
        the crop stopped.
    """
    if kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd, got {kernel_size}. An even kernel needs asymmetric "
            "padding to preserve spatial size, which shifts the shell by half a voxel "
            "against the mask it was derived from."
        )
    padding = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool3d(-mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


class MultiTaskLoss(nn.Module):
    """Weighted sum of segmentation, boundary, confidence and surface losses.

    Accepts either a `neurovision.models.heads.multitask.MultiTaskOutput` (segmentation plus
    optional confidence/boundary logits) or a bare `Tensor` / `list[Tensor]` from a model with
    no auxiliary heads -- in the latter case only the segmentation term is computed and every
    other component is exactly 0.0, so this loss is a drop-in replacement for `dice_ce` /
    `DeepSupervisionLoss` even when the auxiliary heads are turned off.
    """

    def __init__(
        self,
        seg_loss: nn.Module,
        boundary_loss: nn.Module | None = None,
        surface_loss: nn.Module | None = None,
        seg_weight: float = 1.0,
        boundary_weight: float = 0.3,
        confidence_weight: float = 0.05,
        surface_weight: float = 0.0,
        boundary_kernel_size: int = 3,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initializes the multi-task loss.

        Args:
            seg_loss: Loss applied to `preds.seg`. Typically a `DiceBCELoss` (single output)
                or a `DeepSupervisionLoss` wrapping one (multiple deep-supervision outputs).
            boundary_loss: Loss applied to the boundary head's logits against a morphological
                shell of the ground truth. `None` disables the term regardless of
                `boundary_weight`. Typically a `DiceBCELoss` -- see `forward`'s comment on why
                Dice, not plain BCE.
            surface_loss: MONAI `HausdorffDTLoss` / `LogHausdorffDTLoss` applied to the
                full-resolution segmentation logits. `None` disables the term.
            seg_weight: Multiplier on the segmentation term.
            boundary_weight: Multiplier on the boundary term.
            confidence_weight: Multiplier on the confidence term. Kept small (0.05 by
                default) -- see `forward`'s comment on why this is a fixed weight rather than
                a warmup schedule.
            surface_weight: Multiplier on the surface term.
            boundary_kernel_size: Structuring-element size passed to
                `morphological_boundary` when deriving the boundary target.
            confidence_threshold: Sigmoid threshold used to decide "predicted positive" when
                building the confidence target.
        """
        super().__init__()
        self.seg_loss = seg_loss
        self.boundary_loss = boundary_loss
        self.surface_loss = surface_loss
        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.confidence_weight = confidence_weight
        self.surface_weight = surface_weight
        self.boundary_kernel_size = boundary_kernel_size
        self.confidence_threshold = confidence_threshold
        self._last_components: dict[str, float] = {
            "seg": 0.0,
            "boundary": 0.0,
            "confidence": 0.0,
            "surface": 0.0,
            "total": 0.0,
        }

    def forward(self, preds: MultiTaskOutput | Tensor | list[Tensor], target: Tensor) -> Tensor:
        """Computes the weighted multi-task loss.

        Args:
            preds: Either a `MultiTaskOutput` (accessed by attribute only: `.seg`,
                `.confidence`, `.boundary`), or a bare `Tensor` / `list[Tensor]` of
                segmentation logits from a model with no auxiliary heads. In the latter case
                only the segmentation term is computed.
            target: Binary region masks, shape `(B, C, D, H, W)`, full resolution.

        Returns:
            A 0-dim (scalar) weighted-sum loss tensor.
        """
        if isinstance(preds, (Tensor, list)):
            # No auxiliary heads: wrap in a throwaway namespace so the rest of this method
            # doesn't need a second code path. mypy/pyright would flag this as an ad-hoc
            # attribute bag, which is exactly why the type hint above is a Union rather than
            # `Any` -- callers with a real MultiTaskOutput still get real checking.
            seg_preds = preds
            confidence_logits = None
            boundary_logits = None
        else:
            seg_preds = preds.seg
            confidence_logits = preds.confidence
            boundary_logits = preds.boundary

        seg_input = (
            seg_preds[0] if isinstance(seg_preds, list) and len(seg_preds) == 1 else (seg_preds)
        )
        seg = self.seg_loss(seg_input, target)

        # --- Boundary term -------------------------------------------------------------
        # The boundary target is a hard function of the ground truth only, so it is derived
        # under no_grad and every gradient in this term flows through boundary_logits alone.
        if boundary_logits is not None and self.boundary_loss is not None:
            with torch.no_grad():
                boundary_target = morphological_boundary(target.float(), self.boundary_kernel_size)
            # A DiceBCELoss, not plain BCE: the shell is a thin structure (measured ~2.2% of
            # voxels for a 40^3 cube inside a 96^3 volume). An unweighted BCE loss averages
            # over all voxels, so it is dominated by the ~98% background and the model can
            # cut it almost to zero by predicting "no boundary anywhere" -- the Dice term is
            # what keeps the thin positive class from being ignored.
            boundary = self.boundary_loss(boundary_logits, boundary_target)
        else:
            boundary = target.new_zeros(())

        # --- Confidence term -------------------------------------------------------------
        if confidence_logits is not None:
            with torch.no_grad():
                # "Correct" is computed from the segmentation head's CURRENT prediction, so
                # this target moves as training progresses -- especially early on, when the
                # segmentation head is still flipping predictions from step to step. The
                # detach here is load-bearing, not a micro-optimization: if gradient reached
                # preds.seg[0] through this term, the model could cheat by making its
                # segmentation logits uniformly extreme (very confident everywhere) rather
                # than uniformly CORRECT -- exactly the overconfidence this project's
                # calibration claim exists to expose. With the detach, gradient reaches the
                # confidence head only through the decoder features it shares with the
                # segmentation head, never through the segmentation logits themselves.
                #
                # The project's answer to the moving-target problem is a small fixed weight
                # (0.05 by default) rather than a warmup ramp on confidence_weight. A warmup
                # ramp would need a step or epoch counter as loss state, and this loss module
                # is not part of the checkpoint payload (see training/checkpoint.py) -- a
                # 12-hour Kaggle session that gets killed and resumed would silently restart
                # the ramp from zero every time, making the effective objective depend on how
                # many times the session happened to die. CLAUDE.md's resume constraint is
                # about the checkpoint, but a stateful loss violates its spirit the same way.
                prob = torch.sigmoid(seg_preds[0] if isinstance(seg_preds, list) else seg_preds)
                correct = ((prob > self.confidence_threshold).float() == target.float()).float()
            confidence = F.binary_cross_entropy_with_logits(confidence_logits, correct)
        else:
            confidence = target.new_zeros(())

        # --- Surface term ------------------------------------------------------------------
        if self.surface_loss is not None:
            surface_input = seg_preds[0] if isinstance(seg_preds, list) else seg_preds
            surface = self.surface_loss(surface_input, target)
        else:
            surface = target.new_zeros(())

        total = (
            self.seg_weight * seg
            + self.boundary_weight * boundary
            + self.confidence_weight * confidence
            + self.surface_weight * surface
        )

        # For logging only -- plain detached Python floats, never used in any computation.
        self._last_components = {
            "seg": float(seg.detach()),
            "boundary": float(boundary.detach()),
            "confidence": float(confidence.detach()),
            "surface": float(surface.detach()),
            "total": float(total.detach()),
        }

        return total

    @property
    def last_components(self) -> dict[str, float]:
        """The unweighted per-term values (plus the weighted total) from the last `forward`.

        Overwritten on every call to `forward`. For logging only -- these are detached Python
        floats and carry no gradient, so they must never be used in any further computation.

        Returns:
            A dict with keys `"seg"`, `"boundary"`, `"confidence"`, `"surface"`, `"total"`.
        """
        return self._last_components


@register_loss("multitask")
def build_multitask(cfg: Any) -> nn.Module:
    """Builds `MultiTaskLoss` from config.

    Reads `cfg.training.loss.multitask` for the auxiliary-term settings and reuses
    `neurovision.losses.segmentation.build_dice_ce(cfg)` for the segmentation term, so the
    Dice/BCE settings and deep-supervision wrapping stay defined in exactly one place.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A configured `MultiTaskLoss`.
    """
    loss_cfg = cfg.training.loss
    mt_cfg = loss_cfg.multitask

    seg_loss = build_dice_ce(cfg)

    boundary_loss: nn.Module | None = None
    boundary_weight = 0.0
    if mt_cfg.boundary.enabled:
        boundary_loss = DiceBCELoss(
            dice_weight=loss_cfg.dice_weight,
            ce_weight=loss_cfg.ce_weight,
            # MUST stay True here for the same reason as the segmentation term: channel 0 of
            # the boundary target is ET's shell, a real foreground region, not background.
            include_background=True,
            squared_pred=mt_cfg.boundary.squared_pred,
            smooth_nr=loss_cfg.smooth_nr,
            smooth_dr=loss_cfg.smooth_dr,
        )
        boundary_weight = mt_cfg.boundary.weight

    confidence_weight = mt_cfg.confidence.weight if mt_cfg.confidence.enabled else 0.0

    surface_loss: nn.Module | None = None
    surface_weight = 0.0
    if mt_cfg.surface.enabled:
        # MONAI's HausdorffDTLoss computes a Euclidean distance transform for BOTH prediction
        # and target on every forward call. On CUDA that goes through cuCIM/CuPy -- the exact
        # backend that already failed to compile on the Kaggle T4 image for HD95
        # (CompileException: Thrust requires at least C++17, see CLAUDE.md). Without cuCIM it
        # falls back to scipy, i.e. a host round-trip every step; measured ~0.72s per step for
        # a 4-patch, 3-channel, 96^3 batch, which roughly doubles step time on rationed Kaggle
        # hours. The boundary head above buys the same signal for two max_pool3d kernels.
        logger.warning(
            "training.loss.multitask.surface.enabled=true: MONAI's HausdorffDTLoss recomputes "
            "a distance transform every forward pass. Measured ~0.72s/step extra on CPU for a "
            "4x3x96^3 batch, and its CUDA path (cuCIM/CuPy) previously failed outright on the "
            "Kaggle T4 image. Prefer the boundary head (multitask.boundary) unless you "
            "specifically need this term."
        )
        from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss

        surface_cls = LogHausdorffDTLoss if mt_cfg.surface.log else HausdorffDTLoss
        surface_loss = surface_cls(
            sigmoid=True,
            include_background=True,
            alpha=mt_cfg.surface.alpha,
            reduction="mean",
        )
        surface_weight = mt_cfg.surface.weight

    logger.info(
        "Building multitask loss: seg_weight=%s boundary_weight=%s confidence_weight=%s "
        "surface_weight=%s",
        mt_cfg.seg_weight,
        boundary_weight,
        confidence_weight,
        surface_weight,
    )

    return MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=boundary_loss,
        surface_loss=surface_loss,
        seg_weight=mt_cfg.seg_weight,
        boundary_weight=boundary_weight,
        confidence_weight=confidence_weight,
        surface_weight=surface_weight,
        boundary_kernel_size=mt_cfg.boundary.kernel_size,
        confidence_threshold=mt_cfg.confidence.threshold,
    )
