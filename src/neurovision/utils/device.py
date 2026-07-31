"""Single source of truth for device selection.

Every other module resolves its device by calling ``get_device`` on a config
object (or a plain string) instead of writing ``"cuda"`` / ``.cuda()``
directly. This keeps the same code runnable on both a CPU-only laptop and a
CUDA GPU on Kaggle.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_VALID_DEVICES = ("auto", "cuda", "cpu", "mps")


def get_device(cfg: Any | str) -> torch.device:
    """Resolve the torch device to use, from a config object or a string.

    Args:
        cfg: Either a plain string (``"auto"``, ``"cuda"``, ``"cpu"``, or
            ``"mps"``) or a config object (Hydra ``DictConfig``, dataclass,
            or ``SimpleNamespace``) exposing a ``.device`` attribute holding
            one of those strings.

    Returns:
        The resolved ``torch.device``.

    Raises:
        AttributeError: If a config object is passed without a ``device``
            attribute.
        ValueError: If the requested value is not a string, or is a string
            that does not match one of the valid options.
        RuntimeError: If ``"cuda"`` is requested but CUDA is not available,
            or ``"mps"`` is requested but MPS is not available.
    """
    if isinstance(cfg, str):
        requested = cfg
    else:
        if not hasattr(cfg, "device"):
            raise AttributeError(
                "Config object has no 'device' attribute. Add a 'device' key "
                "to the config (e.g. device: auto)."
            )
        requested = cfg.device

    if not isinstance(requested, str):
        raise ValueError(
            f"cfg.device must be a string, got {requested!r} ({type(requested).__name__}). "
            f"Valid options: {_VALID_DEVICES}"
        )

    normalized = requested.strip().lower()

    if normalized == "auto":
        # "auto" only ever resolves to cuda or cpu. MPS is deliberately
        # excluded here: 3D conv support on MPS is incomplete and can fail
        # silently, so auto-selection must never hand a user MPS by surprise.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif normalized == "cuda":
        # An explicit "cuda" request that silently degraded to CPU would
        # waste a rationed 12-hour Kaggle GPU session, so fail loudly
        # instead of falling back.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was requested but torch.cuda.is_available() is False. "
                "Refusing to silently fall back to CPU."
            )
        device = torch.device("cuda")
    elif normalized == "cpu":
        device = torch.device("cpu")
    elif normalized == "mps":
        logger.warning(
            "MPS support for 3D convolutions is incomplete and may fail silently or "
            "obscurely. CPU is the supported local test device; use 'mps' at your own risk."
        )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Device 'mps' was requested but torch.backends.mps.is_available() is False."
            )
        device = torch.device("mps")
    else:
        raise ValueError(
            f"Unknown device '{requested}'. Valid options (case-insensitive): {_VALID_DEVICES}"
        )

    logger.info("Resolved device: %s", device)
    return device


def amp_enabled(device: torch.device) -> bool:
    """Whether automatic mixed precision should be used for this device.

    AMP is on by default for CUDA; it is off for CPU (unsupported for our
    3D conv workload) and MPS (autocast support is incomplete there too).

    Args:
        device: The resolved torch device.

    Returns:
        True if AMP should be enabled, False otherwise.
    """
    return device.type == "cuda"
