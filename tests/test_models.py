"""Tests for neurovision.models: registry and unet3d / swinunetr builders.

CPU only, whole file runs in about a second. Configs are built with
OmegaConf.create mirroring the real configs/model/*.yaml files, following the
same `_cfg`-style helper pattern as tests/test_losses.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.models import baseline  # noqa: F401  (registers unet3d/swinunetr)
from neurovision.models.baseline import build_swinunetr, build_unet3d
from neurovision.models.registry import (
    _MODEL_REGISTRY,
    available_models,
    build_model,
    register_model,
)


def _unet3d_cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring configs/model/unet3d.yaml."""
    base = {
        "data": {"in_channels": 4, "num_classes": 3},
        "model": {
            "name": "unet3d",
            "in_channels": 4,
            "out_channels": 3,
            "channels": [32, 64, 128, 256, 320],
            "strides": [2, 2, 2, 2],
            "num_res_units": 2,
            "norm": "instance",
            "activation": "leakyrelu",
            "dropout": 0.1,
            "deep_supervision": False,
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model, key, value, merge=True)
    return cfg


def _swinunetr_cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring configs/model/swinunetr.yaml."""
    base = {
        "data": {"in_channels": 4, "num_classes": 3},
        "model": {
            "name": "swinunetr",
            "in_channels": 4,
            "out_channels": 3,
            "feature_size": 48,
            "depths": [2, 2, 2, 2],
            "num_heads": [3, 6, 12, 24],
            "norm_name": "instance",
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "use_checkpoint": True,
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model, key, value, merge=True)
    return cfg


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# unet3d
# ---------------------------------------------------------------------------


def test_unet3d_shape() -> None:
    # Production channels, 32^3 input: hits the target output shape and the
    # bracket in test_unet3d_param_count below in well under a second.
    cfg = _unet3d_cfg()
    model = build_model(cfg)
    model.eval()

    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 32, 32, 32)


def test_unet3d_param_count_in_plausible_range() -> None:
    # The bracket exists to catch a config that never reached the model (e.g.
    # MONAI's UNet defaults would land far outside this range), not to pin an
    # exact parameter count.
    cfg = _unet3d_cfg()
    model = build_model(cfg)
    n_params = _n_params(model)
    assert 10e6 < n_params < 16e6


def test_unet3d_deep_supervision_raises() -> None:
    cfg = _unet3d_cfg(deep_supervision=True)
    with pytest.raises(ValueError, match="deep_supervision"):
        build_unet3d(cfg)


# ---------------------------------------------------------------------------
# swinunetr
# ---------------------------------------------------------------------------


def test_swinunetr_shape_small_feature_size() -> None:
    # 64^3 input and feature_size=12 (rather than 32^3 / production
    # feature_size=48, as used for unet3d above): SwinUNETR downsamples 32x,
    # so a 32^3 input reaches the bottleneck at 1x1x1 and InstanceNorm3d
    # raises there. feature_size=48 forward-passes in ~1.84s, which alone
    # would blow the suite's speed budget, so this shape test only proves the
    # forward pass and channel plumbing work at a small, fast width.
    cfg = _swinunetr_cfg(feature_size=12)
    model = build_model(cfg)
    model.eval()

    x = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 64, 64, 64)


def test_swinunetr_param_count_small_feature_size() -> None:
    cfg = _swinunetr_cfg(feature_size=12)
    model = build_model(cfg)
    n_params = _n_params(model)
    assert 3e6 < n_params < 5.5e6


def test_swinunetr_param_count_production_feature_size_build_only() -> None:
    # Build only, no forward pass (too slow for this suite at feature_size=48
    # -- see test_swinunetr_shape_small_feature_size). This is what proves
    # feature_size from config actually reaches the model: MONAI's own
    # default (24) would land outside this bracket.
    cfg = _swinunetr_cfg()  # feature_size=48, the production config
    model = build_model(cfg)
    n_params = _n_params(model)
    assert 55e6 < n_params < 70e6


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_build_model_unknown_name_raises_and_names_it() -> None:
    cfg = _unet3d_cfg()
    cfg.model.name = "not_a_real_model"
    with pytest.raises(ValueError, match="not_a_real_model"):
        build_model(cfg)


def test_register_model_duplicate_name_raises() -> None:
    @register_model("__test_dummy_model__")
    def _dummy(cfg: object) -> object:
        return object()

    try:
        with pytest.raises(ValueError, match="__test_dummy_model__"):

            @register_model("__test_dummy_model__")
            def _dummy2(cfg: object) -> object:
                return object()

    finally:
        # Clean up so this test is repeatable and doesn't pollute other tests.
        _MODEL_REGISTRY.pop("__test_dummy_model__", None)


def test_available_models_lists_both_baselines() -> None:
    names = available_models()
    assert "unet3d" in names
    assert "swinunetr" in names


def test_build_swinunetr_directly_matches_registry() -> None:
    cfg = _swinunetr_cfg(feature_size=12)
    model = build_swinunetr(cfg)
    assert _n_params(model) == _n_params(build_model(cfg))
