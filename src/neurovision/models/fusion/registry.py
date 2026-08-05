"""Registry mapping config strings to fusion-block constructors.

Ablations select a fusion variant by name from Hydra config
(``model.fusion.name``) instead of by editing code — the novel gated
cross-attention block and its two baselines (concat, add) are all reached
through this one lookup.

Unlike ``neurovision.models.registry`` (which holds whole networks) and
``neurovision.losses.registry`` (whole losses), a fusion builder cannot work
from the config alone: one block is built per pyramid level, and each level
has its own channel widths. So a builder takes the full config *plus* the
two branch widths and the level index.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)

# (cfg, cnn_channels, swin_channels, level) -> a ready-to-use fusion block.
#
# `level` is the index into the FUSED pyramid, i.e. 0 is the finest fused
# level (stride 2), not CNN level 0 (stride 1, which is never fused — see the
# stride-offset note in neurovision.models.encoders.swin). It is passed so a
# block can size itself per level (e.g. choosing full attention on a coarse
# level whose feature map is small enough to afford it).
FusionBuilder = Callable[[Any, int, int, int], nn.Module]

_FUSION_REGISTRY: dict[str, FusionBuilder] = {}


def register_fusion(name: str) -> Callable[[FusionBuilder], FusionBuilder]:
    """Register a fusion-block builder under ``name``.

    Args:
        name: Lookup key used in ``cfg.model.fusion.name``. Stored lowercase.

    Returns:
        A decorator that registers the wrapped builder and returns it
        unchanged.

    Raises:
        ValueError: If ``name`` is already registered. A silent overwrite
            would make it ambiguous which fusion variant actually ran, which
            is exactly what the ablation table depends on.
    """

    def decorator(builder: FusionBuilder) -> FusionBuilder:
        key = name.lower()
        if key in _FUSION_REGISTRY:
            raise ValueError(
                f"Fusion '{key}' is already registered. Choose a different name "
                f"or remove the duplicate registration."
            )
        _FUSION_REGISTRY[key] = builder
        return builder

    return decorator


def build_fusion(cfg: Any, cnn_channels: int, swin_channels: int, level: int) -> nn.Module:
    """Build one fusion block, selected by ``cfg.model.fusion.name``.

    Args:
        cfg: The full composed Hydra config, exposing ``cfg.model.fusion``.
        cnn_channels: Channel width of the CNN branch at this level. Also the
            output width of every fusion variant, so the decoder sees CNN
            widths regardless of which variant is selected.
        swin_channels: Channel width of the Swin branch at this level.
        level: Index into the fused pyramid, 0 = finest (stride 2).

    Returns:
        The constructed fusion block.

    Raises:
        ValueError: If ``cfg.model.fusion.name`` is not a registered fusion.
    """
    name = cfg.model.fusion.name.lower()
    if name not in _FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion '{name}'. Available fusions: {available_fusions()}")
    logger.info(
        "Building fusion '%s' for level %d (cnn=%d ch, swin=%d ch)",
        name,
        level,
        cnn_channels,
        swin_channels,
    )
    return _FUSION_REGISTRY[name](cfg, cnn_channels, swin_channels, level)


def available_fusions() -> list[str]:
    """List the names of all currently registered fusion variants.

    Returns:
        Registered fusion names, in registration order.
    """
    return list(_FUSION_REGISTRY.keys())
