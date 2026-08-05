"""Tests for neurovision.models.encoders.swin: SwinEncoder and build_swin_encoder.

CPU only, whole file runs in a few seconds. feature_size=12 and
use_checkpoint=False are used throughout (rather than the production
feature_size=48) purely for speed -- see the same note in
tests/test_models.py for the SwinUNETR baseline. Configs are built with
OmegaConf.create, following the same `_cfg`-style helper pattern as
tests/test_cnn_encoder.py.
"""

from __future__ import annotations

import pytest
import torch
from monai.networks.nets.swin_unetr import SwinTransformer
from omegaconf import OmegaConf

from neurovision.models.encoders.cnn import CNNEncoder
from neurovision.models.encoders.swin import SwinEncoder, build_swin_encoder


def _cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring cfg.model.encoder.swin."""
    base = {
        "data": {"in_channels": 4, "num_classes": 3},
        "model": {
            "encoder": {
                "swin": {
                    "feature_size": 16,  # distinctive: not the SwinEncoder default (48)
                    "depths": [2, 2, 2, 2],
                    "num_heads": [3, 6, 12, 24],
                    "window_size": 7,
                    "patch_size": 2,
                    "num_levels": 4,
                    "drop_rate": 0.0,
                    "attn_drop_rate": 0.0,
                    "dropout_path_rate": 0.0,
                    "use_checkpoint": False,
                    "normalize": True,
                }
            }
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model.encoder.swin, key, value, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# pyramid shapes
# ---------------------------------------------------------------------------


def test_pyramid_shapes_default() -> None:
    model = SwinEncoder(in_channels=4, feature_size=12, use_checkpoint=False)
    x = torch.randn(1, 4, 64, 64, 64)
    pyramid = model(x)

    assert len(pyramid) == 4
    # Each shape asserted explicitly and separately -- a loop recomputing the
    # expectation with the same formula the implementation uses would pass
    # even if that formula were wrong.
    assert pyramid[0].shape == (1, 12, 32, 32, 32)
    assert pyramid[1].shape == (1, 24, 16, 16, 16)
    assert pyramid[2].shape == (1, 48, 8, 8, 8)
    assert pyramid[3].shape == (1, 96, 4, 4, 4)


def test_public_attributes_match_config() -> None:
    model = SwinEncoder(in_channels=4, feature_size=12, use_checkpoint=False)
    assert model.num_levels == 4
    assert model.out_channels == [12, 24, 48, 96]
    assert model.strides == [2, 4, 8, 16]


# ---------------------------------------------------------------------------
# THE ALIGNMENT TEST -- headline test of this file.
#
# CNNEncoder's pyramid starts at stride 1 (level 0, a full-resolution stem);
# SwinEncoder's starts at stride 2 (patch_embed downsamples before any
# attention runs). A 5-level CNNEncoder's levels 1..4 must therefore line up
# spatially, level for level, with a 4-level SwinEncoder's levels 0..3 -- that
# alignment is exactly what the fusion module depends on to fuse same-shape
# feature maps from the two branches. This is the load-bearing contract of
# the whole dual-encoder design, so it is checked both statically (the
# stride lists agree) and at runtime (the actual tensors agree), and across
# input shapes including an odd, anisotropic one that cannot pass by
# divisibility coincidence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 64, 64), (96, 96, 96), (33, 35, 37)])
def test_cnn_swin_pyramid_alignment(shape: tuple[int, int, int]) -> None:
    cnn = CNNEncoder(
        in_channels=4,
        channels=[8, 16, 32, 64, 128],
        blocks_per_stage=[1, 1, 1, 1, 1],
        num_groups=8,
    )
    swin = SwinEncoder(in_channels=4, feature_size=12, use_checkpoint=False)

    # Static contract: CNN's levels 1..4 strides equal Swin's levels 0..3
    # strides, i.e. [2, 4, 8, 16] == [2, 4, 8, 16].
    assert cnn.strides[1:] == swin.strides

    x = torch.randn(1, 4, *shape)
    cnn_pyramid = cnn(x)
    swin_pyramid = swin(x)

    # Runtime contract: the actual spatial shapes agree, level for level.
    for i in range(swin.num_levels):
        assert cnn_pyramid[i + 1].shape[2:] == swin_pyramid[i].shape[2:], (
            f"level {i}: cnn shape {cnn_pyramid[i + 1].shape} vs swin shape "
            f"{swin_pyramid[i].shape}, input shape {shape}"
        )

    # CNN's level 0 (stride 1) has no Swin counterpart at all.
    assert len(cnn_pyramid) == len(swin_pyramid) + 1


# ---------------------------------------------------------------------------
# MONAI equivalence -- regression guard on the reimplemented forward.
# ---------------------------------------------------------------------------


def test_forward_matches_monai_swin_transformer() -> None:
    kwargs = dict(
        in_channels=4,
        feature_size=12,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        patch_size=2,
        num_levels=5,
        use_checkpoint=False,
        normalize=True,
    )
    ours = SwinEncoder(**kwargs)

    reference = SwinTransformer(
        in_chans=4,
        embed_dim=12,
        window_size=(7, 7, 7),
        patch_size=(2, 2, 2),
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=torch.nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
    )
    # Copy weights so the two modules are numerically identical, then compare
    # forward outputs -- this is what catches a MONAI upgrade silently
    # changing SwinTransformer.forward under our reimplementation.
    reference.load_state_dict(ours.swin.state_dict())

    ours.eval()
    reference.eval()

    x = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        ours_out = ours(x)
        reference_out = reference(x, True)

    assert len(ours_out) == len(reference_out) == 5
    for level, (a, b) in enumerate(zip(ours_out, reference_out, strict=True)):
        assert torch.allclose(a, b, atol=1e-6), f"level {level} mismatch"


# ---------------------------------------------------------------------------
# stage reclamation
# ---------------------------------------------------------------------------


def test_num_levels_4_has_fewer_params_than_5_and_empty_layers4() -> None:
    enc4 = SwinEncoder(in_channels=4, feature_size=12, num_levels=4, use_checkpoint=False)
    enc5 = SwinEncoder(in_channels=4, feature_size=12, num_levels=5, use_checkpoint=False)

    params4 = sum(p.numel() for p in enc4.parameters())
    params5 = sum(p.numel() for p in enc5.parameters())

    assert params4 < params5
    assert len(enc4.swin.layers4) == 0
    assert len(enc5.swin.layers4) == 1


def test_num_levels_5_returns_5_levels_last_at_stride_32() -> None:
    model = SwinEncoder(in_channels=4, feature_size=12, num_levels=5, use_checkpoint=False)
    x = torch.randn(1, 4, 64, 64, 64)
    pyramid = model(x)

    assert len(pyramid) == 5
    assert model.out_channels == [12, 24, 48, 96, 192]
    assert model.strides == [2, 4, 8, 16, 32]
    assert pyramid[4].shape == (1, 192, 2, 2, 2)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_num_levels_zero_raises() -> None:
    with pytest.raises(ValueError, match="0"):
        SwinEncoder(in_channels=4, feature_size=12, num_levels=0)


def test_num_levels_six_raises() -> None:
    with pytest.raises(ValueError, match="6"):
        SwinEncoder(in_channels=4, feature_size=12, num_levels=6)


def test_depths_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match=r"3.*four stages|four stages.*3"):
        SwinEncoder(in_channels=4, feature_size=12, depths=(2, 2, 2))


def test_num_heads_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match=r"3.*four stages|four stages.*3"):
        SwinEncoder(in_channels=4, feature_size=12, num_heads=(3, 6, 12))


# ---------------------------------------------------------------------------
# backward pass
# ---------------------------------------------------------------------------


def test_backward_pass_produces_finite_grads_and_no_dead_stage_params() -> None:
    model = SwinEncoder(in_channels=4, feature_size=12, num_levels=4, use_checkpoint=False)
    x = torch.randn(1, 4, 64, 64, 64)
    pyramid = model(x)

    loss = sum(level.mean() for level in pyramid)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no grad"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"
        # The reclaimed stage must contribute NO parameters at all.
        assert "layers4" not in name, f"unexpected parameter from reclaimed stage: {name}"


# ---------------------------------------------------------------------------
# eval-mode determinism
# ---------------------------------------------------------------------------


def test_eval_mode_is_deterministic() -> None:
    model = SwinEncoder(in_channels=4, feature_size=12, use_checkpoint=False)
    model.eval()
    x = torch.randn(1, 4, 64, 64, 64)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    for level1, level2 in zip(out1, out2, strict=True):
        assert torch.equal(level1, level2)


# ---------------------------------------------------------------------------
# build_swin_encoder
# ---------------------------------------------------------------------------


def test_build_swin_encoder_reads_config_values() -> None:
    cfg = _cfg()
    encoder = build_swin_encoder(cfg)

    # feature_size=16 is distinctive -- the SwinEncoder default is 48, so a
    # fallback to the default could not coincidentally match these.
    assert encoder.out_channels == [16, 32, 64, 128]
    assert encoder.strides == [2, 4, 8, 16]
    assert encoder.num_levels == 4
