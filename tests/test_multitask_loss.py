"""Tests for neurovision.losses.multitask: morphological_boundary and MultiTaskLoss.

Uses a small local stand-in for `neurovision.models.heads.multitask.MultiTaskOutput`
(a `types.SimpleNamespace` with `.seg` / `.confidence` / `.boundary`) so this file does not
depend on the parallel agent's module -- `MultiTaskLoss` accesses those attributes only, it
never imports the real dataclass at runtime.

All tensors are tiny and everything runs on CPU, well under a second.
"""

from __future__ import annotations

import types

import pytest
import torch
from omegaconf import OmegaConf

from neurovision.losses import multitask  # noqa: F401  (registers "multitask")
from neurovision.losses.multitask import MultiTaskLoss, build_multitask, morphological_boundary
from neurovision.losses.registry import build_loss
from neurovision.losses.segmentation import DiceBCELoss

SHAPE = (2, 3, 16, 16, 16)


def _output(seg=None, confidence=None, boundary=None):
    """A stand-in for MultiTaskOutput, accessed by attribute only."""
    return types.SimpleNamespace(seg=seg, confidence=confidence, boundary=boundary)


def _binary_target(shape: tuple[int, ...] = SHAPE, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g) > 0.5).float()


def _loss_cfg(**overrides: object) -> object:
    """A full composed config mirroring configs/training/default.yaml's multitask block."""
    base = {
        "training": {
            "loss": {
                "name": "multitask",
                "dice_weight": 1.0,
                "ce_weight": 1.0,
                "sigmoid": True,
                "softmax": False,
                "include_background": True,
                "squared_pred": False,
                "smooth_nr": 1.0e-5,
                "smooth_dr": 1.0e-5,
                "deep_supervision": {"enabled": False, "weights": None},
                "multitask": {
                    "seg_weight": 1.0,
                    "boundary": {
                        "enabled": True,
                        "weight": 0.3,
                        "kernel_size": 3,
                        "squared_pred": False,
                    },
                    "confidence": {"enabled": True, "weight": 0.05, "threshold": 0.5},
                    "surface": {
                        "enabled": False,
                        "weight": 0.1,
                        "alpha": 2.0,
                        "log": False,
                    },
                },
            }
        }
    }
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg.training.loss, key, value, merge=True)
    return cfg


# ---------------------------------------------------------------------------
# morphological_boundary
# ---------------------------------------------------------------------------


def _cube_mask(shape=(1, 1, 16, 16, 16), lo=4, hi=12) -> torch.Tensor:
    mask = torch.zeros(shape)
    mask[..., lo:hi, lo:hi, lo:hi] = 1.0
    return mask


def test_boundary_hollow_shell_on_solid_cube() -> None:
    mask = _cube_mask()
    shell = morphological_boundary(mask, kernel_size=3)

    assert shell.shape == mask.shape
    # Deep interior voxel (well away from the cube's own surface) must be 0.
    assert shell[0, 0, 8, 8, 8].item() == 0.0
    # A voxel exactly on the cube's face must be 1.
    assert shell[0, 0, 4, 8, 8].item() == 1.0
    assert shell.sum().item() > 0.0


def test_boundary_shell_smaller_than_cube_when_cube_is_large_relative_to_shell_thickness() -> None:
    """shell volume < cube volume, using the exact case measured in the module docstring.

    A 40^3 cube inside a 96^3 volume: this is not true for every cube (a small cube's
    surface-to-volume ratio can make the two-voxel-thick shell bigger than the cube itself),
    but it is true here, and this is the shape this project actually cares about (shell
    fraction ~2.2%, cited in MultiTaskLoss's boundary-term comment for why plain BCE would be
    dominated by background).
    """
    mask = torch.zeros((1, 1, 96, 96, 96))
    mask[..., 28:68, 28:68, 28:68] = 1.0
    shell = morphological_boundary(mask, kernel_size=3)

    cube_volume = mask.sum().item()
    shell_volume = shell.sum().item()
    assert 0.0 < shell_volume < cube_volume
    assert shell_volume / mask.numel() == pytest.approx(0.0217, abs=1e-3)


def test_boundary_all_zero_mask_is_all_zero() -> None:
    mask = torch.zeros((1, 1, 8, 8, 8))
    shell = morphological_boundary(mask)
    assert torch.equal(shell, torch.zeros_like(shell))


def test_boundary_all_ones_mask_is_all_zero_everywhere_including_border() -> None:
    """No spurious rim at the volume border.

    `max_pool3d` pads with `-inf`, and an `-inf` contribution never wins a max against a real
    neighbor value -- so erosion (`-max_pool3d(-mask, ...)`) ignores missing (out-of-patch)
    neighbors rather than treating them as background. A mask that fills the entire volume
    therefore has NO shell anywhere, including at its own edge: verified here (not merely
    assumed from reading `max_pool3d`'s docs), and this is the behavior the project wants --
    a tumor that continues past the training patch's edge is not really bounded there.
    """
    mask = torch.ones((1, 1, 8, 8, 8))
    shell = morphological_boundary(mask, kernel_size=3)
    assert torch.equal(shell, torch.zeros_like(shell))


def test_boundary_foreground_touching_patch_edge_is_silent_there() -> None:
    """A region abutting the patch border shows a shell only at its real internal transition.

    Foreground fills x in [0, 4) of an 8-voxel axis, i.e. it touches the x=0 patch edge. The
    shell must appear only at the genuine x=3/4 transition, not at x=0 -- confirming the
    "outside the patch is background" reading of max_pool3d's -inf padding does NOT hold.
    """
    mask = torch.zeros((1, 1, 8, 8, 8))
    mask[..., 0:4, :, :] = 1.0
    shell = morphological_boundary(mask, kernel_size=3)

    profile = shell[0, 0, :, 4, 4]
    assert profile[0].item() == 0.0  # patch edge: silent
    assert profile[3].item() == 1.0  # real transition
    assert profile[4].item() == 1.0


def test_boundary_even_kernel_raises() -> None:
    mask = _cube_mask()
    with pytest.raises(ValueError, match="odd"):
        morphological_boundary(mask, kernel_size=4)


def test_boundary_shape_preserving_and_channel_independent() -> None:
    mask = torch.zeros((1, 3, 16, 16, 16))
    mask[:, 0, 4:12, 4:12, 4:12] = 1.0  # cube in channel 0 only
    shell = morphological_boundary(mask)

    assert shell.shape == mask.shape
    assert shell[:, 0].sum().item() > 0.0
    assert torch.equal(shell[:, 1], torch.zeros_like(shell[:, 1]))
    assert torch.equal(shell[:, 2], torch.zeros_like(shell[:, 2]))


# ---------------------------------------------------------------------------
# MultiTaskLoss -- seg-only no-op regression
# ---------------------------------------------------------------------------


def test_seg_only_matches_bare_seg_loss_exactly() -> None:
    """Aux weights 0 and aux preds None must be a true no-op vs. the bare seg loss."""
    seg_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=None,
        surface_loss=None,
        seg_weight=1.0,
        boundary_weight=0.0,
        confidence_weight=0.0,
        surface_weight=0.0,
    )

    target = _binary_target()
    logits = torch.randn(SHAPE)

    plain = seg_loss(logits, target)
    out = _output(seg=[logits], confidence=None, boundary=None)
    wrapped = mt_loss(out, target)

    assert torch.allclose(plain, wrapped)


# ---------------------------------------------------------------------------
# Perfect prediction -> ~0 total loss
# ---------------------------------------------------------------------------


def test_perfect_prediction_near_zero_total_loss() -> None:
    seg_loss = DiceBCELoss()
    boundary_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=boundary_loss,
        surface_loss=None,
        seg_weight=1.0,
        boundary_weight=0.3,
        confidence_weight=0.05,
        surface_weight=0.0,
    )

    target = _binary_target()
    # Saturated logits: strongly positive where target is 1, strongly negative where 0. Both
    # DiceLoss(sigmoid=True) and BCEWithLogitsLoss go to ~0 for a saturated correct-sign
    # prediction (sigmoid(20) ~ 1 - 2e-9).
    seg_logits = (target * 2 - 1) * 20

    boundary_target = morphological_boundary(target)
    boundary_logits = (boundary_target * 2 - 1) * 20

    # Confidence head predicts "always correct" with a saturated logit; since seg_logits is
    # itself a perfect prediction, the (detached) correctness target really is all-ones, so a
    # saturated positive logit is exactly right and BCE goes to ~0 too.
    confidence_logits = torch.full(SHAPE, 20.0)

    out = _output(seg=[seg_logits], confidence=confidence_logits, boundary=boundary_logits)
    loss = mt_loss(out, target)
    assert loss.item() < 0.05


# ---------------------------------------------------------------------------
# last_components
# ---------------------------------------------------------------------------


def test_last_components_keys_types_and_total() -> None:
    seg_loss = DiceBCELoss()
    boundary_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=boundary_loss,
        surface_loss=None,
        seg_weight=1.0,
        boundary_weight=0.3,
        confidence_weight=0.05,
        surface_weight=0.0,
    )

    target = _binary_target()
    torch.manual_seed(3)
    seg_logits = torch.randn(SHAPE)
    boundary_logits = torch.randn(SHAPE)
    confidence_logits = torch.randn(SHAPE)

    out = _output(seg=[seg_logits], confidence=confidence_logits, boundary=boundary_logits)
    mt_loss(out, target)

    components = mt_loss.last_components
    assert set(components.keys()) == {"seg", "boundary", "confidence", "surface", "total"}
    assert all(isinstance(v, float) for v in components.values())

    expected_total = (
        1.0 * components["seg"]
        + 0.3 * components["boundary"]
        + 0.05 * components["confidence"]
        + 0.0 * components["surface"]
    )
    assert components["total"] == pytest.approx(expected_total, abs=1e-5)


def test_last_components_no_gradient() -> None:
    seg_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(seg_loss=seg_loss)

    target = _binary_target()
    logits = torch.randn(SHAPE, requires_grad=True)
    out = _output(seg=[logits], confidence=None, boundary=None)
    mt_loss(out, target)

    for v in mt_loss.last_components.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Confidence term sends no gradient to segmentation logits (THE important test)
# ---------------------------------------------------------------------------


def test_confidence_term_sends_no_gradient_to_seg_logits() -> None:
    """Pins the detach the calibration claim depends on.

    If the confidence objective's target were not detached from the segmentation logits, the
    model could lower this loss by making seg predictions uniformly extreme (maximally
    "confident-looking") rather than actually correct -- exactly the overconfidence the
    project's calibration claim exists to expose.
    """
    seg_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=None,
        surface_loss=None,
        seg_weight=0.0,
        boundary_weight=0.0,
        confidence_weight=1.0,
        surface_weight=0.0,
    )

    target = _binary_target()
    seg_logits = torch.randn(SHAPE, requires_grad=True)
    confidence_logits = torch.randn(SHAPE, requires_grad=True)

    out = _output(seg=[seg_logits], confidence=confidence_logits, boundary=None)
    loss = mt_loss(out, target)
    loss.backward()

    assert seg_logits.grad is None or torch.equal(
        seg_logits.grad, torch.zeros_like(seg_logits.grad)
    )
    # Sanity: the confidence head itself DOES get gradient, so the detach is specific to
    # seg_logits and not an accidental "nothing computed" loss.
    assert confidence_logits.grad is not None
    assert not torch.equal(confidence_logits.grad, torch.zeros_like(confidence_logits.grad))


# ---------------------------------------------------------------------------
# Bare Tensor / list[Tensor] preds (no auxiliary heads)
# ---------------------------------------------------------------------------


def test_forward_accepts_bare_tensor() -> None:
    seg_loss = DiceBCELoss()
    mt_loss = MultiTaskLoss(seg_loss=seg_loss)

    target = _binary_target()
    logits = torch.randn(SHAPE)

    plain = seg_loss(logits, target)
    wrapped = mt_loss(logits, target)
    assert torch.allclose(plain, wrapped)
    assert mt_loss.last_components["boundary"] == 0.0
    assert mt_loss.last_components["confidence"] == 0.0
    assert mt_loss.last_components["surface"] == 0.0


def test_forward_accepts_list_of_tensors_deep_supervision_no_aux() -> None:
    from neurovision.losses.segmentation import DeepSupervisionLoss

    base = DiceBCELoss()
    ds_loss = DeepSupervisionLoss(base)
    mt_loss = MultiTaskLoss(seg_loss=ds_loss)

    target = _binary_target()
    torch.manual_seed(4)
    full = torch.randn(SHAPE)
    half = torch.randn(2, 3, 8, 8, 8)

    preds = [full, half]
    plain = ds_loss(preds, target)
    wrapped = mt_loss(preds, target)
    assert torch.allclose(plain, wrapped)


# ---------------------------------------------------------------------------
# Boundary skipped without error when preds.boundary is None
# ---------------------------------------------------------------------------


def test_boundary_skipped_when_preds_boundary_none_even_if_loss_configured() -> None:
    seg_loss = DiceBCELoss()
    boundary_loss = DiceBCELoss()  # configured, but preds won't provide boundary logits
    mt_loss = MultiTaskLoss(
        seg_loss=seg_loss,
        boundary_loss=boundary_loss,
        boundary_weight=0.3,
    )

    target = _binary_target()
    logits = torch.randn(SHAPE)
    out = _output(seg=[logits], confidence=None, boundary=None)
    mt_loss(out, target)

    assert mt_loss.last_components["boundary"] == 0.0


# ---------------------------------------------------------------------------
# build_multitask from config
# ---------------------------------------------------------------------------


def test_build_multitask_weights_from_config() -> None:
    cfg = _loss_cfg()
    loss = build_loss(cfg)
    assert isinstance(loss, MultiTaskLoss)
    assert loss.seg_weight == pytest.approx(1.0)
    assert loss.boundary_weight == pytest.approx(0.3)
    assert loss.confidence_weight == pytest.approx(0.05)
    assert loss.surface_weight == pytest.approx(0.0)
    assert loss.boundary_loss is not None
    assert loss.surface_loss is None


def test_build_multitask_boundary_disabled_zeroes_weight() -> None:
    cfg = _loss_cfg()
    cfg.training.loss.multitask.boundary.enabled = False
    loss = build_multitask(cfg)
    assert loss.boundary_weight == 0.0
    assert loss.boundary_loss is None


# ---------------------------------------------------------------------------
# Surface term (HausdorffDTLoss) actually composes -- slow path, one tiny tensor
# ---------------------------------------------------------------------------


def test_build_multitask_surface_enabled_composes_and_is_nonzero() -> None:
    cfg = _loss_cfg()
    cfg.training.loss.multitask.surface.enabled = True
    loss = build_multitask(cfg)
    assert loss.surface_loss is not None
    assert loss.surface_weight == pytest.approx(0.1)

    shape = (1, 3, 8, 8, 8)
    target = _binary_target(shape, seed=5)
    torch.manual_seed(6)
    logits = torch.randn(shape)

    out = _output(seg=[logits], confidence=None, boundary=None)
    total = loss(out, target)

    assert torch.isfinite(total)
    assert loss.last_components["surface"] != 0.0
