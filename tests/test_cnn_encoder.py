"""Tests for neurovision.models.encoders.cnn: CNNEncoder and build_cnn_encoder.

CPU only, whole file runs in well under a second. Configs are built with
OmegaConf.create, following the same `_cfg`-style helper pattern as
tests/test_models.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.models.encoders.cnn import CNNEncoder, build_cnn_encoder


def _cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring cfg.model.encoder.cnn."""
    base = {
        "data": {"in_channels": 4, "num_classes": 3},
        "model": {
            "encoder": {
                "cnn": {
                    "channels": [16, 48, 96, 192],
                    "blocks_per_stage": [1, 1, 1, 1],
                    "num_groups": 8,
                    "dropout": 0.1,
                    "use_checkpoint": False,
                }
            }
        },
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.model.encoder.cnn, key, value, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# pyramid shapes
# ---------------------------------------------------------------------------


def test_pyramid_shapes_production_like() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[1, 2, 2, 2],
        num_groups=8,
    )
    x = torch.randn(1, 4, 32, 32, 32)
    pyramid = model(x)

    assert len(pyramid) == 4
    # Each shape asserted explicitly and separately -- a loop recomputing
    # the expectation with the same 2**i formula the code uses would pass
    # even if that formula were wrong.
    assert pyramid[0].shape == (1, 32, 32, 32, 32)
    assert pyramid[1].shape == (1, 64, 16, 16, 16)
    assert pyramid[2].shape == (1, 128, 8, 8, 8)
    assert pyramid[3].shape == (1, 256, 4, 4, 4)


def test_public_attributes_match_config() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[1, 1, 1, 1],
        num_groups=8,
    )
    assert model.num_levels == 4
    assert model.out_channels == [32, 64, 128, 256]
    assert model.strides == [1, 2, 4, 8]


@pytest.mark.parametrize(
    "channels,expected_last_spatial",
    [
        ([8, 16, 24], 4),  # 3 levels: 16 -> 16, 8, 4
        ([8, 16, 24, 32, 40], 1),  # 5 levels: 16 -> 16, 8, 4, 2, 1
    ],
)
def test_level_count_follows_channels_length(
    channels: list[int], expected_last_spatial: int
) -> None:
    n_levels = len(channels)
    model = CNNEncoder(
        in_channels=4,
        channels=channels,
        blocks_per_stage=[1] * n_levels,
        num_groups=8,
    )
    x = torch.randn(1, 4, 16, 16, 16)
    pyramid = model(x)

    assert len(pyramid) == n_levels
    assert model.num_levels == n_levels
    last = pyramid[-1]
    assert last.shape == (
        1,
        channels[-1],
        expected_last_spatial,
        expected_last_spatial,
        expected_last_spatial,
    )


def test_odd_input_size_uses_ceil_division() -> None:
    # (20, 24, 28) pins the downsampling convention: each level's spatial
    # size is ceil(input / 2**level), not floor.
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16, 24],
        blocks_per_stage=[1, 1, 1],
        num_groups=8,
    )
    x = torch.randn(1, 4, 20, 24, 28)
    pyramid = model(x)

    assert pyramid[0].shape == (1, 8, 20, 24, 28)
    # ceil(20/2)=10, ceil(24/2)=12, ceil(28/2)=14
    assert pyramid[1].shape == (1, 16, 10, 12, 14)
    # ceil(20/4)=5, ceil(24/4)=6, ceil(28/4)=7
    assert pyramid[2].shape == (1, 24, 5, 6, 7)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_channels_not_divisible_by_num_groups_raises() -> None:
    with pytest.raises(ValueError, match="num_groups"):
        CNNEncoder(
            in_channels=4,
            channels=[16, 20],  # 20 is not divisible by num_groups=8
            blocks_per_stage=[1, 1],
            num_groups=8,
        )


def test_channels_not_divisible_by_num_groups_names_level_index() -> None:
    with pytest.raises(ValueError, match=r"channels\[1\]"):
        CNNEncoder(
            in_channels=4,
            channels=[16, 20],
            blocks_per_stage=[1, 1],
            num_groups=8,
        )


def test_mismatched_blocks_per_stage_length_raises() -> None:
    with pytest.raises(ValueError, match="blocks_per_stage"):
        CNNEncoder(
            in_channels=4,
            channels=[16, 32, 48],
            blocks_per_stage=[1, 1],  # length 2 vs channels length 3
            num_groups=8,
        )


def test_mismatched_blocks_per_stage_names_both_lengths() -> None:
    with pytest.raises(ValueError, match=r"2.*3|3.*2"):
        CNNEncoder(
            in_channels=4,
            channels=[16, 32, 48],
            blocks_per_stage=[1, 1],
            num_groups=8,
        )


# ---------------------------------------------------------------------------
# backward pass
# ---------------------------------------------------------------------------


def test_backward_pass_produces_finite_grads() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[1, 1],
        num_groups=8,
        dropout=0.1,
    )
    x = torch.randn(1, 4, 16, 16, 16)
    pyramid = model(x)

    loss = sum(level.mean() for level in pyramid)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no grad"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


# ---------------------------------------------------------------------------
# gradient checkpointing
# ---------------------------------------------------------------------------


def test_checkpointing_matches_no_checkpointing_output() -> None:
    torch.manual_seed(0)
    model_plain = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[1, 1],
        num_groups=8,
        dropout=0.0,  # deterministic: dropout would make outputs flaky
        use_checkpoint=False,
    )
    model_ckpt = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[1, 1],
        num_groups=8,
        dropout=0.0,
        use_checkpoint=True,
    )
    model_ckpt.load_state_dict(model_plain.state_dict())

    model_plain.train()
    model_ckpt.train()

    x = torch.randn(1, 4, 16, 16, 16)
    out_plain = model_plain(x)
    out_ckpt = model_ckpt(x)

    assert len(out_plain) == len(out_ckpt)
    for level_plain, level_ckpt in zip(out_plain, out_ckpt, strict=True):
        assert level_plain.shape == level_ckpt.shape
        assert torch.allclose(level_plain, level_ckpt, atol=1e-6)

    loss = sum(level.mean() for level in out_ckpt)
    loss.backward()
    for name, param in model_ckpt.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no grad"
            assert torch.isfinite(param.grad).all(), f"{name} has non-finite grad"


# ---------------------------------------------------------------------------
# eval-mode determinism
# ---------------------------------------------------------------------------


def test_eval_mode_is_deterministic() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[1, 1],
        num_groups=8,
        dropout=0.1,
    )
    model.eval()
    x = torch.randn(1, 4, 16, 16, 16)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    for level1, level2 in zip(out1, out2, strict=True):
        assert torch.equal(level1, level2)


# ---------------------------------------------------------------------------
# build_cnn_encoder
# ---------------------------------------------------------------------------


def test_build_cnn_encoder_reads_config_values() -> None:
    cfg = _cfg()
    encoder = build_cnn_encoder(cfg)

    # Distinctive widths [16, 48, 96, 192]: a default-value fallback could
    # not coincidentally match these.
    assert encoder.out_channels == [16, 48, 96, 192]
    assert encoder.num_levels == 4
