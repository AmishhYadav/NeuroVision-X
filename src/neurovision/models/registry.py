"""Registry mapping config strings to model constructors.

Experiments select a model by name from Hydra config (``model.name``) instead
of by editing code. A builder function is registered once, under a lowercase
name, and ``build_model`` looks it up at run time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)

# A builder takes the *full* composed config and returns a ready-to-use model
# module. Full config in (not just the model sub-config) because a builder
# needs e.g. `cfg.data.in_channels` / `cfg.data.num_classes`, not just
# `cfg.model`.
ModelBuilder = Callable[[Any], nn.Module]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder under ``name``.

    Args:
        name: Lookup key used in ``cfg.model.name``. Stored lowercase.

    Returns:
        A decorator that registers the wrapped builder and returns it
        unchanged.

    Raises:
        ValueError: If ``name`` is already registered. A silent overwrite
            would make it ambiguous which model actually ran.
    """

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{key}' is already registered. Choose a different name "
                f"or remove the duplicate registration."
            )
        _MODEL_REGISTRY[key] = builder
        return builder

    return decorator


def build_model(cfg: Any) -> nn.Module:
    """Build the model module selected by ``cfg.model.name``.

    Args:
        cfg: The full composed Hydra config, exposing ``cfg.model``.

    Returns:
        The constructed model module.

    Raises:
        ValueError: If ``cfg.model.name`` is not a registered model.
    """
    name = cfg.model.name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available models: {available_models()}")
    logger.info("Building model '%s'", name)
    return _MODEL_REGISTRY[name](cfg)


def available_models() -> list[str]:
    """List the names of all currently registered models.

    Returns:
        Registered model names, in registration order.
    """
    return list(_MODEL_REGISTRY.keys())
