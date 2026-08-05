"""Tests for neurovision.models.heads.auxiliary.AuxiliaryHead and
neurovision.models.heads.multitask.MultiTaskHead / MultiTaskOutput.

CPU only, synthetic tensors, tiny channel widths -- whole file runs in a couple of seconds.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from neurovision.models.heads.auxiliary import AuxiliaryHead
from neurovision.models.heads.multitask import MultiTaskHead, MultiTaskOutput

# ---------------------------------------------------------------------------
# AuxiliaryHead
# ---------------------------------------------------------------------------


def test_auxiliary_head_shape() -> None:
    head = AuxiliaryHead(in_channels=32, out_channels=3, num_groups=8)
    x = torch.randn(1, 32, 16, 16, 16)

    out = head(x)

    assert out.shape == (1, 3, 16, 16, 16)


def test_auxiliary_head_preserves_odd_anisotropic_shape() -> None:
    # in_channels=32 -> default hidden = max(16, 3, 8) = 16, divisible by num_groups=8.
    head = AuxiliaryHead(in_channels=32, out_channels=3, num_groups=8)
    x = torch.randn(2, 32, 9, 11, 13)

    out = head(x)

    assert out.shape == (2, 3, 9, 11, 13)


def test_auxiliary_head_output_is_unbounded_logits() -> None:
    # Regression guard against someone adding a sigmoid: feed a large-magnitude input and
    # check some output value falls outside [0, 1] (which sigmoid output never would).
    head = AuxiliaryHead(in_channels=16, out_channels=3, num_groups=8)
    x = torch.randn(1, 16, 8, 8, 8) * 50.0

    out = head(x)

    assert torch.any((out < 0.0) | (out > 1.0))


def test_auxiliary_head_final_bias_is_zero_at_init() -> None:
    head = AuxiliaryHead(in_channels=16, out_channels=3, num_groups=8)

    assert torch.all(head.conv2.bias == 0.0)


def test_auxiliary_head_raises_on_hidden_not_divisible_by_num_groups() -> None:
    # hidden defaults to max(16 // 2, 3, 8) = 8, which IS divisible by 8 -- force a mismatch
    # via an explicit hidden_channels instead.
    with pytest.raises(ValueError, match="num_groups"):
        AuxiliaryHead(in_channels=16, out_channels=3, hidden_channels=10, num_groups=8)


def test_auxiliary_head_dropout_module_type() -> None:
    head_no_dropout = AuxiliaryHead(in_channels=16, out_channels=3, num_groups=8, dropout=0.0)
    head_with_dropout = AuxiliaryHead(in_channels=16, out_channels=3, num_groups=8, dropout=0.3)

    assert isinstance(head_no_dropout.dropout, nn.Identity)
    assert isinstance(head_with_dropout.dropout, nn.Dropout3d)


# ---------------------------------------------------------------------------
# MultiTaskHead
# ---------------------------------------------------------------------------

DECODER_CHANNELS = [16, 24, 32]  # fine to coarse, matches a 3-stage decoder


def _feats(batch: int = 1) -> list[Tensor]:
    """Synthetic fine-to-coarse decoder features at halving spatial resolution."""
    return [
        torch.randn(batch, DECODER_CHANNELS[0], 16, 16, 16),
        torch.randn(batch, DECODER_CHANNELS[1], 8, 8, 8),
        torch.randn(batch, DECODER_CHANNELS[2], 4, 4, 4),
    ]


def test_multitask_head_no_auxiliary_heads() -> None:
    head = MultiTaskHead(
        decoder_channels=DECODER_CHANNELS,
        out_channels=3,
        deep_supervision_levels=3,
        confidence=False,
        boundary=False,
    )

    out = head(_feats())

    assert isinstance(out, MultiTaskOutput)
    assert head.has_auxiliary is False
    assert out.confidence is None
    assert out.boundary is None
    assert len(out.seg) == 3
    for i, seg_i in enumerate(out.seg):
        expected_size = 16 // (2**i)
        assert seg_i.shape == (1, 3, expected_size, expected_size, expected_size)


def test_multitask_head_both_auxiliary_heads_enabled() -> None:
    head = MultiTaskHead(
        decoder_channels=DECODER_CHANNELS,
        out_channels=3,
        deep_supervision_levels=2,
        confidence=True,
        boundary=True,
        confidence_num_groups=8,
        boundary_num_groups=8,
    )

    feats = _feats()
    out = head(feats)

    assert head.has_auxiliary is True
    # Full-resolution shape (feats[0]'s), NOT feats[1]'s -- catches an aux head accidentally
    # wired to the wrong decoder stage.
    assert out.confidence is not None
    assert out.boundary is not None
    assert out.confidence.shape == (1, 3, 16, 16, 16)
    assert out.boundary.shape == (1, 3, 16, 16, 16)
    assert out.confidence.shape != feats[1].shape
    assert len(out.seg) == 2


def test_multitask_head_raises_on_zero_deep_supervision_levels() -> None:
    with pytest.raises(ValueError, match="deep_supervision_levels"):
        MultiTaskHead(decoder_channels=DECODER_CHANNELS, out_channels=3, deep_supervision_levels=0)


def test_multitask_head_raises_on_too_many_deep_supervision_levels() -> None:
    with pytest.raises(ValueError, match="deep_supervision_levels"):
        MultiTaskHead(decoder_channels=DECODER_CHANNELS, out_channels=3, deep_supervision_levels=4)


def test_multitask_head_forward_raises_on_too_few_feats() -> None:
    head = MultiTaskHead(
        decoder_channels=DECODER_CHANNELS, out_channels=3, deep_supervision_levels=3
    )

    with pytest.raises(ValueError, match="deep_supervision_levels"):
        head(_feats()[:2])
