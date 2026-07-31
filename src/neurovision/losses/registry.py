"""Registry mapping config strings to loss constructors.

Experiments select a loss by name from Hydra config (``training.loss.name``)
instead of by editing code. A builder function is registered once, under a
lowercase name, and ``build_loss`` looks it up at run time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)

# A builder takes the *full* composed config and returns a ready-to-use loss
# module. Full config in (not just the loss sub-config) because some losses
# may eventually need e.g. `cfg.data` (class weights, region list, ...).
LossBuilder = Callable[[Any], nn.Module]

_LOSS_REGISTRY: dict[str, LossBuilder] = {}


def register_loss(name: str) -> Callable[[LossBuilder], LossBuilder]:
    """Register a loss builder under ``name``.

    Args:
        name: Lookup key used in ``cfg.training.loss.name``. Stored lowercase.

    Returns:
        A decorator that registers the wrapped builder and returns it
        unchanged.

    Raises:
        ValueError: If ``name`` is already registered. A silent overwrite
            would make it ambiguous which loss actually ran.
    """

    def decorator(builder: LossBuilder) -> LossBuilder:
        key = name.lower()
        if key in _LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{key}' is already registered. Choose a different name "
                f"or remove the duplicate registration."
            )
        _LOSS_REGISTRY[key] = builder
        return builder

    return decorator


def build_loss(cfg: Any) -> nn.Module:
    """Build the loss module selected by ``cfg.training.loss.name``.

    Args:
        cfg: The full composed Hydra config, exposing ``cfg.training.loss``.

    Returns:
        The constructed loss module.

    Raises:
        ValueError: If ``cfg.training.loss.name`` is not a registered loss.
    """
    name = cfg.training.loss.name.lower()
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available_losses()}")
    logger.info("Building loss '%s'", name)
    return _LOSS_REGISTRY[name](cfg)


def available_losses() -> list[str]:
    """List the names of all currently registered losses.

    Returns:
        Registered loss names, in registration order.
    """
    return list(_LOSS_REGISTRY.keys())
