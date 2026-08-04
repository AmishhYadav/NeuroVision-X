"""Tests for neurovision.models.encoders.cnn: CNNEncoder and build_cnn_encoder.

CPU only, whole file runs in well under a second. Configs are built with
OmegaConf.create, following the same `_cfg`-style helper pattern as
tests/test_models.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.models.encoders.cnn import CNNEncoder, ResidualBlock, build_cnn_encoder


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
                    "zero_init_residual": True,
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
    # blocks_per_stage=[2, 2], not [1, 1]: with one block per stage every
    # block is the first of its stage and so always takes the projection
    # (nn.Sequential) shortcut. Only a SECOND block in a stage has the
    # nn.Identity shortcut, where `identity` aliases the block's own input
    # `x` -- the case where the in-place `out += identity` and
    # `LeakyReLU(inplace=True)` could in principle corrupt a tensor autograd
    # still needs. That path has to be backpropped through here or nothing
    # in the suite covers it.
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[2, 2],
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
    # blocks_per_stage=[2, 2] so the nn.Identity-shortcut path is inside the
    # checkpointed region too -- see test_backward_pass_produces_finite_grads.
    model_plain = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[2, 2],
        num_groups=8,
        dropout=0.0,  # deterministic: dropout would make outputs flaky
        use_checkpoint=False,
    )
    model_ckpt = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[2, 2],
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


def test_build_cnn_encoder_reads_zero_init_residual() -> None:
    encoder = build_cnn_encoder(_cfg(zero_init_residual=False))
    blocks = [m for m in encoder.modules() if isinstance(m, ResidualBlock)]
    assert all(torch.all(b.norm2.weight == 1.0) for b in blocks)


# ---------------------------------------------------------------------------
# weight initialization
# ---------------------------------------------------------------------------


def _blocks(model: CNNEncoder) -> list[ResidualBlock]:
    return [m for m in model.modules() if isinstance(m, ResidualBlock)]


def test_zero_init_residual_zeroes_only_norm2() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[2, 2],
        num_groups=8,
        zero_init_residual=True,
    )
    blocks = _blocks(model)
    assert len(blocks) == 4

    for block in blocks:
        # The residual branch's LAST norm is zeroed...
        assert torch.all(block.norm2.weight == 0.0)
        # ...but never norm1, and never the shortcut's norm. Zeroing either
        # of those would stop signal reaching the next stage at all rather
        # than merely making the block an identity.
        assert torch.all(block.norm1.weight == 1.0)
        if isinstance(block.shortcut, torch.nn.Sequential):
            assert torch.all(block.shortcut[1].weight == 1.0)
        # Every norm bias starts at zero regardless.
        assert torch.all(block.norm2.bias == 0.0)


def test_zero_init_residual_disabled_leaves_norm2_at_one() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[8, 16],
        blocks_per_stage=[2, 2],
        num_groups=8,
        zero_init_residual=False,
    )
    for block in _blocks(model):
        assert torch.all(block.norm2.weight == 1.0)


def test_zero_init_makes_identity_shortcut_block_an_exact_identity() -> None:
    # stages[0][1] is the SECOND block of the stem: stride 1, same width in
    # and out, so its shortcut is nn.Identity. With norm2.weight == 0 the
    # residual branch contributes exactly 0, leaving LeakyReLU(x).
    model = CNNEncoder(
        in_channels=4,
        channels=[8],
        blocks_per_stage=[2],
        num_groups=8,
        dropout=0.0,
        zero_init_residual=True,
    )
    model.eval()
    block = model.stages[0][1]
    assert isinstance(block.shortcut, torch.nn.Identity)

    x = torch.randn(1, 8, 8, 8, 8)
    reference = torch.nn.functional.leaky_relu(x.clone(), negative_slope=0.01)
    with torch.no_grad():
        out = block(x)

    assert torch.allclose(out, reference, atol=1e-6)


def test_conv_weights_use_kaiming_fan_out_not_torch_default() -> None:
    model = CNNEncoder(
        in_channels=4,
        channels=[32, 64],
        blocks_per_stage=[1, 1],
        num_groups=8,
    )
    # stages[1][0].conv1: in 32, out 64, kernel 3. Kaiming normal with
    # mode="fan_out" gives std = gain / sqrt(fan_out), where
    # fan_out = out_channels * 3**3 = 1728 and
    # gain = sqrt(2 / (1 + 0.01**2)) for a LeakyReLU of slope 0.01.
    weight = model.stages[1][0].conv1.weight
    fan_out = 64 * 27
    expected_std = (2.0 / (1.0 + 0.01**2)) ** 0.5 / fan_out**0.5

    measured = weight.std().item()
    assert measured == pytest.approx(expected_std, rel=0.10)

    # PyTorch's Conv3d default is kaiming_uniform_(a=sqrt(5)) over fan_in,
    # whose std is ~0.0196 here versus ~0.0340 for ours. Asserting they
    # differ is what proves _init_weights actually ran -- without it the
    # tolerance check above could pass on a coincidence.
    torch_default_std = (6.0 / ((1.0 + 5.0) * (32 * 27))) ** 0.5 / 3.0**0.5
    assert abs(measured - torch_default_std) > 0.5 * abs(expected_std - torch_default_std)


def test_zero_init_gives_conv2_zero_grad_on_first_step_only() -> None:
    # Documents a consequence that looks like a dead layer but is not: the
    # gradient reaching conv2 is scaled by norm2.weight, which is 0 at init,
    # so conv2's first-step gradient is exactly zero. norm2.weight itself
    # gets a non-zero gradient, moves off zero, and conv2 trains thereafter.
    model = CNNEncoder(
        in_channels=4,
        channels=[8],
        blocks_per_stage=[2],
        num_groups=8,
        dropout=0.0,
        zero_init_residual=True,
    )
    block = model.stages[0][1]
    x = torch.randn(1, 4, 8, 8, 8)

    sum(level.mean() for level in model(x)).backward()
    assert torch.all(block.conv2.weight.grad == 0.0)
    assert torch.any(block.norm2.weight.grad != 0.0)

    # One optimizer step moves norm2.weight off zero; conv2 then gets a
    # real gradient.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    assert torch.any(block.norm2.weight != 0.0)

    sum(level.mean() for level in model(x)).backward()
    assert torch.any(block.conv2.weight.grad != 0.0)
