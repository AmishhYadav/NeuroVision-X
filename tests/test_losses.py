"""Tests for neurovision.losses: registry and dice_ce / DeepSupervisionLoss.

All tensors are tiny (B=2, C=3, 8^3) and everything runs on CPU, matching the
BraTS region-overlap setup: channel 0 = ET, channel 1 = TC, channel 2 = WT,
binary float targets, raw logit predictions.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.losses import segmentation  # noqa: F401  (registers "dice_ce")
from neurovision.losses.registry import (
    _LOSS_REGISTRY,
    available_losses,
    build_loss,
    register_loss,
)
from neurovision.losses.segmentation import DeepSupervisionLoss, DiceBCELoss, build_dice_ce

SHAPE = (2, 3, 8, 8, 8)


def _loss_cfg(**overrides: object) -> object:
    """Builds a full composed config mirroring configs/training/default.yaml."""
    base = {
        "training": {
            "loss": {
                "name": "dice_ce",
                "dice_weight": 1.0,
                "ce_weight": 1.0,
                "sigmoid": True,
                "softmax": False,
                "include_background": True,
                "squared_pred": False,
                "smooth_nr": 1.0e-5,
                "smooth_dr": 1.0e-5,
                "deep_supervision": {"enabled": False, "weights": None},
            }
        }
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.training.loss, key, value, merge=True)
    return cfg


def _binary_target(shape: tuple[int, ...] = SHAPE) -> torch.Tensor:
    """A reproducible-ish binary float target of the given shape."""
    torch.manual_seed(0)
    return (torch.rand(shape) > 0.5).float()


# ---------------------------------------------------------------------------
# build_dice_ce / build_loss
# ---------------------------------------------------------------------------


def test_build_loss_returns_dice_ce_without_deep_supervision() -> None:
    cfg = _loss_cfg()
    loss = build_loss(cfg)
    assert isinstance(loss, DiceBCELoss)


def test_build_loss_returns_deep_supervision_when_enabled() -> None:
    cfg = _loss_cfg()
    cfg.training.loss.deep_supervision.enabled = True
    loss = build_loss(cfg)
    assert isinstance(loss, DeepSupervisionLoss)


def test_build_loss_unknown_name_raises_and_names_it() -> None:
    cfg = _loss_cfg(name="not_a_real_loss")
    with pytest.raises(ValueError, match="not_a_real_loss"):
        build_loss(cfg)


def test_register_loss_duplicate_name_raises() -> None:
    @register_loss("__test_dummy_loss__")
    def _dummy(cfg: object) -> object:
        return object()

    try:
        with pytest.raises(ValueError, match="__test_dummy_loss__"):

            @register_loss("__test_dummy_loss__")
            def _dummy2(cfg: object) -> object:
                return object()

    finally:
        # Clean up so this test is repeatable and doesn't pollute other tests.
        _LOSS_REGISTRY.pop("__test_dummy_loss__", None)


def test_available_losses_lists_dice_ce() -> None:
    assert "dice_ce" in available_losses()


def test_both_sigmoid_and_softmax_raises() -> None:
    cfg = _loss_cfg(sigmoid=True, softmax=True)
    with pytest.raises(ValueError, match="sigmoid"):
        build_dice_ce(cfg)


# ---------------------------------------------------------------------------
# Perfect vs. random prediction
# ---------------------------------------------------------------------------


def test_perfect_prediction_gives_near_zero_loss() -> None:
    # DiceBCELoss composes DiceLoss(sigmoid=True) with BCEWithLogitsLoss, so
    # both terms are strictly per-channel and a saturated correct-sign
    # prediction drives the whole loss to ~0. This is the property that
    # monai.losses.DiceCELoss does NOT have for C > 1: it selects its CE by
    # channel count (`self.ce(...) if input.shape[1] != 1 else self.bce(...)`),
    # so it applies softmax cross-entropy across our three overlapping region
    # channels and leaves a residual cost even on a perfect prediction.
    # 1e-4 is loose enough for the sigmoid at logit 20 (~2e-9) plus Dice
    # smoothing, and tight enough that reintroducing a softmax CE term fails.
    cfg = _loss_cfg()
    loss_fn = build_loss(cfg)
    target = _binary_target()
    logits = (target * 2 - 1) * 20
    loss = loss_fn(logits, target)
    assert loss.item() < 1e-4


def test_random_prediction_gives_much_higher_loss_than_perfect() -> None:
    cfg = _loss_cfg()
    loss_fn = build_loss(cfg)
    target = _binary_target()

    perfect_logits = (target * 2 - 1) * 20
    perfect_loss = loss_fn(perfect_logits, target).item()

    torch.manual_seed(1)
    random_logits = torch.randn(SHAPE)
    random_loss = loss_fn(random_logits, target).item()

    # With a genuinely per-channel loss the separation is enormous (perfect is
    # ~1e-9), so an absolute floor is a more meaningful assertion than a ratio.
    assert random_loss > 0.5
    assert random_loss > 100 * perfect_loss


def test_include_background_regression() -> None:
    """channel 0 (ET) must count in the Dice term, not be silently dropped.

    This is the trap the spec warns about: with include_background=False
    (copied from a softmax setup where channel 0 really is background), a
    completely wrong ET channel would not raise the loss at all. Here we
    assert the opposite behaviour: getting ET completely wrong, while TC and
    WT are perfect, must produce a clearly higher loss than a fully perfect
    prediction. The margin is very large now that a perfect prediction scores
    ~0, so an absolute floor is asserted rather than only a ratio.
    """
    cfg = _loss_cfg()
    loss_fn = build_loss(cfg)
    target = _binary_target()

    fully_perfect = (target * 2 - 1) * 20
    perfect_loss = loss_fn(fully_perfect, target).item()

    wrong_et = fully_perfect.clone()
    wrong_et[:, 0] = -fully_perfect[:, 0]  # flip the sign on ET only
    wrong_et_loss = loss_fn(wrong_et, target).item()

    assert wrong_et_loss > 0.1
    assert wrong_et_loss > perfect_loss * 100


def test_correctly_predicted_empty_channel_is_not_punished() -> None:
    """An empty region, correctly predicted empty, must cost ~0 Dice.

    Many BraTS cases have no enhancing tumor at all, and at 96^3 patch level a
    large fraction of patches contain none of a given region. With
    smooth_nr=0 the Dice of an empty channel is 0/(0+smooth_dr) = 0, i.e. the
    maximum possible loss for the correct answer -- which would push the model
    to hallucinate tumor. Equal non-zero smoothing gives 1e-5/1e-5 = 1.
    """
    cfg = _loss_cfg()
    loss_fn = build_loss(cfg)

    target = torch.zeros(SHAPE)  # every region empty
    logits = torch.full(SHAPE, -20.0)  # confidently predicts empty

    smoothed = loss_fn(logits, target).item()
    unsmoothed = build_loss(_loss_cfg(smooth_nr=0.0))(logits, target).item()

    # smooth_nr=0 gives the WORST possible Dice (1.0) for the correct answer.
    assert unsmoothed == pytest.approx(1.0, abs=1e-3)
    # Equal smoothing pulls it far down. It is not exactly 0 because summed
    # sigmoid(-20) over all voxels is the same order as the smoothing constant,
    # so the ratio does not reach 1.0 -- but the gradient no longer rewards
    # inventing tumor, which is the property that matters.
    assert smoothed < 0.2
    assert smoothed < unsmoothed / 5


# ---------------------------------------------------------------------------
# DeepSupervisionLoss
# ---------------------------------------------------------------------------


def test_deep_supervision_single_output_matches_plain_loss() -> None:
    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base)
    target = _binary_target()
    logits = torch.randn(SHAPE)

    plain = base(logits, target)
    wrapped = ds_loss(logits, target)
    assert torch.allclose(plain, wrapped)

    # Also confirm passing a single-element list behaves identically.
    wrapped_list = ds_loss([logits], target)
    assert torch.allclose(plain, wrapped_list)


def test_deep_supervision_same_resolution_bounded_by_min_max() -> None:
    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base)
    target = _binary_target()

    torch.manual_seed(2)
    preds = [torch.randn(SHAPE) for _ in range(3)]
    individual = [base(p, target).item() for p in preds]
    combined = ds_loss(preds, target).item()

    # A weighted average of the individual losses must lie within their range.
    assert min(individual) <= combined <= max(individual)


def test_deep_supervision_mixed_resolutions_no_error() -> None:
    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base)
    target = _binary_target((2, 3, 8, 8, 8))

    full = torch.randn(2, 3, 8, 8, 8)
    half = torch.randn(2, 3, 4, 4, 4)
    quarter = torch.randn(2, 3, 2, 2, 2)

    loss = ds_loss([full, half, quarter], target)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_deep_supervision_wrong_length_explicit_weights_raises() -> None:
    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base, weights=[1.0, 0.5])
    target = _binary_target()
    preds = [torch.randn(SHAPE) for _ in range(3)]

    with pytest.raises(ValueError, match="3") as exc_info:
        ds_loss(preds, target)
    assert "2" in str(exc_info.value)


def test_deep_supervision_negative_weight_raises_at_construction() -> None:
    base = DiceBCELoss()
    with pytest.raises(ValueError):
        DeepSupervisionLoss(base, weights=[1.0, -0.5])


def test_deep_supervision_all_zero_weights_raises_at_construction() -> None:
    base = DiceBCELoss()
    with pytest.raises(ValueError):
        DeepSupervisionLoss(base, weights=[0.0, 0.0])


def test_loss_output_is_scalar_and_gradients_propagate() -> None:
    cfg = _loss_cfg()
    loss_fn = build_loss(cfg)
    target = _binary_target()
    logits = torch.randn(SHAPE, requires_grad=True)

    loss = loss_fn(logits, target)
    assert loss.dim() == 0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_deep_supervision_gradients_propagate() -> None:
    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base)
    target = _binary_target((2, 3, 8, 8, 8))

    full = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    half = torch.randn(2, 3, 4, 4, 4, requires_grad=True)

    loss = ds_loss([full, half], target)
    loss.backward()

    assert full.grad is not None and torch.isfinite(full.grad).all()
    assert half.grad is not None and torch.isfinite(half.grad).all()
