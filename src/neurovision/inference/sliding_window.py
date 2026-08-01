"""Evaluation-time sliding-window inference.

Standalone equivalent of the sliding-window call inside
`neurovision.training.trainer.Trainer.validate`, driven by a separate config
block (`cfg.inference.sliding_window`, not `cfg.training.sliding_window`).
The two stay separate on purpose: training's block is tuned for speed inside
the epoch loop (lower overlap, no Gaussian blending), and a final evaluation
should not silently inherit that trade-off. `Trainer.validate` is untouched
by this module.

Returns raw logits only. No sigmoid, no threshold, no argmax --
discretization into a binary segmentation lives in
`neurovision.inference.postprocess`, a separate module.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from monai.inferers import SlidingWindowInferer
from torch import Tensor, nn

from neurovision.utils.device import amp_enabled

logger = logging.getLogger(__name__)


def build_inferer(cfg: Any) -> SlidingWindowInferer:
    """Builds a `SlidingWindowInferer` from `cfg.inference.sliding_window`.

    Args:
        cfg: The full composed Hydra config, exposing
            `cfg.inference.sliding_window`.

    Returns:
        A configured MONAI `SlidingWindowInferer`. It does not run inference
        by itself; call the returned object as `inferer(inputs, network)`.
    """
    sw_cfg = cfg.inference.sliding_window

    # OmegaConf hands back a ListConfig, not a plain list, for roi_size.
    # MONAI does shape arithmetic (zip against the image shape, min(), ...)
    # on roi_size, which only works reliably on a plain list.
    roi_size = list(sw_cfg.roi_size)

    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_cfg.sw_batch_size,
        overlap=sw_cfg.overlap,
        mode=sw_cfg.mode,
        sigma_scale=sw_cfg.sigma_scale,
        padding_mode=sw_cfg.padding_mode,
        # Our `output_device` maps to MONAI's `device` kwarg -- the buffer
        # overlapping windows get blended into -- NOT to MONAI's identically
        # tempting `sw_device` kwarg, which is where each cropped window is
        # run through the model. Those are different parameters and confusing
        # them is why the config field is not called `sw_device`. The model
        # already lives on the target device, so window computation is left
        # at MONAI's default. `null` passes straight through: MONAI already
        # reads `None` as "same device as the input".
        device=sw_cfg.output_device,
    )


def sliding_window_predict(
    model: nn.Module,
    image: Tensor,
    cfg: Any,
    device: torch.device,
    use_amp: bool | None = None,
) -> Tensor:
    """Runs evaluation-time sliding-window inference and returns raw logits.

    Puts `model` in eval mode and runs the forward passes under
    `torch.no_grad()` itself, so callers do not need to do either. Applies
    no sigmoid, threshold, or argmax -- discretizing the output into a
    binary segmentation is `neurovision.inference.postprocess`'s job.

    Args:
        model: The segmentation model. Assumed to already be on `device`;
            its train/eval mode is not assumed, this function sets it.
        image: Input volume, shape `(B, C, D, H, W)`. Moved to `device`
            inside this function if it is not already there.
        cfg: The full composed Hydra config, exposing
            `cfg.inference.sliding_window`.
        device: Device to run inference on, resolved once via
            `neurovision.utils.device.get_device`. The autocast device type
            comes from `device.type`, never a hardcoded `"cuda"`.
        use_amp: Whether to run the forward passes under autocast. `None`
            (the default) decides from `device` via `amp_enabled`; an
            explicit `True`/`False` overrides that decision.

    Returns:
        Raw logits, shape `(B, out_channels, D, H, W)` -- the same spatial
        shape as `image` regardless of how `roi_size` compares to it, with
        `out_channels` coming from `model`. Always float32, even when AMP ran
        the forward passes in half precision, so callers never have to think
        about dtype.

        The result sits on `device` under the default
        `inference.sliding_window.output_device: null`. Setting that config
        field to `"cpu"` deliberately returns a CPU tensor instead -- that is
        the whole point of the option (keeping a full-volume output buffer
        off a 16 GB card), so this function does not move it back.
    """
    amp_on = amp_enabled(device) if use_amp is None else use_amp
    sw_cfg = cfg.inference.sliding_window

    logger.info(
        "Sliding-window inference: roi_size=%s overlap=%s mode=%s amp=%s input_shape=%s",
        list(sw_cfg.roi_size),
        sw_cfg.overlap,
        sw_cfg.mode,
        amp_on,
        tuple(image.shape[2:]),
    )

    image = image.to(device)
    inferer = build_inferer(cfg)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
            logits = inferer(inputs=image, network=model)

    # Cast back from AMP's half precision (a no-op when AMP did not run) so
    # the caller always gets float32 regardless of device/AMP settings.
    return logits.float()
