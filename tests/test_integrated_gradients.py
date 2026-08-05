"""Tests for `neurovision.explainability.integrated_gradients`.

CPU only (never MPS -- see CLAUDE.md's machine split). Every test uses a tiny hand-built stub
network, never a real registry model or `NeuroVisionX` -- CLAUDE.md's spec for this module
warns that a single forward+backward of the real model at 64^3 is ~5.5s on CPU, and
`integrated_gradients` runs `n_steps` of those, which would blow the suite's time budget.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import Tensor, nn

from neurovision.explainability.integrated_gradients import (
    MODALITY_NAMES,
    IntegratedGradientsOutput,
    build_region_score_fn,
    integrated_gradients,
    modality_ranking,
)

# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class TinyIGNet(nn.Module):
    """A tiny 2-layer conv net with a nonlinearity, emitting 3 region logit channels.

    The `Tanh` in the middle is what makes the network genuinely NONLINEAR along the IG
    integration path -- a purely linear network would converge to zero delta at any n_steps
    >= 1 (its gradient is constant along the path), which would make the convergence-tracks-
    n_steps test (test 4 below) meaningless.
    """

    def __init__(self, in_channels: int = 4, width: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, width, kernel_size=3, padding=1)
        self.act = nn.Tanh()
        self.head = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.act(self.stem(x)))


class AllNegativeNet(TinyIGNet):
    """Same as `TinyIGNet` but with a large negative head bias.

    Every sigmoid output sits far below 0.5, so the default predicted-positive mask comes back
    empty and the whole-volume fallback path is exercised.
    """

    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.head.bias.fill_(-50.0)


def _make_model(cls: type[nn.Module] = TinyIGNet) -> nn.Module:
    torch.manual_seed(0)
    model = cls()
    model.eval()
    return model


def _make_image(size: int = 8, channels: int = 4, seed: int = 1) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, channels, size, size, size, generator=generator)


# ---------------------------------------------------------------------------
# 1-2. Shapes and normalization of modality_attribution
# ---------------------------------------------------------------------------


def test_shapes_and_modality_keys() -> None:
    model = _make_model()
    image = _make_image()

    out = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert isinstance(out, IntegratedGradientsOutput)
    assert out.attributions.shape == image.shape
    assert set(out.modality_attribution.keys()) == set(MODALITY_NAMES)
    assert set(out.modality_attribution_signed.keys()) == set(MODALITY_NAMES)


def test_modality_attribution_sums_to_one() -> None:
    model = _make_model()
    image = _make_image()

    out = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert sum(out.modality_attribution.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. Completeness axiom -- THE correctness test for this whole module
# ---------------------------------------------------------------------------


def test_completeness_axiom_holds() -> None:
    """attributions.sum() ~= target_score - baseline_score, at a reasonable n_steps.

    This is the direct statement of IG's completeness axiom. If the scalar wrapper reduced
    over the wrong axes, or the mask were recomputed per integration step (see this module's
    top-of-file docstring, hazard 2), this identity would NOT hold and this is the test that
    would catch it. Tolerance: 1e-3 relative to the score's own magnitude -- generous enough to
    absorb the Riemann sum's residual error at n_steps=32 on a nonlinear stub (measured ~1e-4
    in practice), tight enough to fail hard if the mask or reduction were wrong (which produces
    errors many orders of magnitude larger, not a marginally worse approximation).
    """
    model = _make_model()
    image = _make_image()

    out = integrated_gradients(model, image, region_index=0, n_steps=32)

    expected = out.target_score - out.baseline_score
    assert out.attributions.sum().item() == pytest.approx(expected, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# 4-5. Convergence delta tracks n_steps, and the warning fires
# ---------------------------------------------------------------------------


def test_relative_delta_is_small_when_converged() -> None:
    model = _make_model()
    image = _make_image()

    out = integrated_gradients(model, image, region_index=0, n_steps=32)

    assert out.relative_delta < 0.05


def test_relative_delta_shrinks_with_more_steps() -> None:
    model = _make_model()
    image = _make_image()

    coarse = integrated_gradients(model, image, region_index=0, n_steps=2)
    fine = integrated_gradients(model, image, region_index=0, n_steps=64)

    # Pins that relative_delta genuinely tracks convergence quality rather than being a
    # constant -- measured directly: ~1.3e-3 at n_steps=2 vs ~0.0 at n_steps=64 on this stub.
    assert coarse.relative_delta > fine.relative_delta


def test_convergence_warning_fires_with_tiny_tolerance(caplog: pytest.LogCaptureFixture) -> None:
    model = _make_model()
    image = _make_image()

    with caplog.at_level(logging.WARNING):
        integrated_gradients(model, image, region_index=0, n_steps=8, delta_tolerance=1e-12)

    assert "relative convergence delta" in caplog.text


# ---------------------------------------------------------------------------
# 6. Fixed-mask regression test
# ---------------------------------------------------------------------------


def test_completeness_holds_when_baseline_and_input_masks_genuinely_differ() -> None:
    """Pins that the mask is fixed across the whole integration path (hazard 2).

    Builds a stub whose predicted-positive mask at the ALL-ZEROS BASELINE is empty (every
    sigmoid output sits at ~0.046, far below threshold) while the mask at the real INPUT is a
    genuine non-empty subset (~6% of voxels cross threshold) -- measured directly on this exact
    setup. If a future edit recomputed the target mask from `model(x)` at each point along the
    path instead of closing over the one mask built from the real input, the scalar function
    being integrated would not be a single well-defined function of x any more (it would sum
    over a different, mostly-empty voxel set near the baseline end of the path and a real
    subset near the input end), and the completeness identity below would not hold. Asserting
    it still holds here is what catches that regression.
    """
    model = _make_model()
    with torch.no_grad():
        model.head.bias[0] = -3.0
        model.head.weight[0] *= 5.0
    image = _make_image(seed=7) * 2  # larger magnitude so some voxels cross threshold

    baseline_logits = model(torch.zeros_like(image))
    image_logits = model(image)
    baseline_mask_would_be_empty = not torch.any(torch.sigmoid(baseline_logits[0, 0]) >= 0.5)
    image_mask_is_nonempty = torch.any(torch.sigmoid(image_logits[0, 0]) >= 0.5)
    # Sanity-check the stub actually produces the differing-mask situation this test needs --
    # if this assertion ever fails, the stub no longer exercises the hazard and must be fixed.
    assert baseline_mask_would_be_empty
    assert image_mask_is_nonempty

    out = integrated_gradients(model, image, region_index=0, n_steps=32)

    assert out.n_target_voxels > 0
    expected = out.target_score - out.baseline_score
    assert out.attributions.sum().item() == pytest.approx(expected, rel=1e-3, abs=1e-3)


# ---------------------------------------------------------------------------
# 7. build_region_score_fn's Captum-batching contract
# ---------------------------------------------------------------------------


def test_build_region_score_fn_returns_batch_shaped_output() -> None:
    model = _make_model()
    mask = torch.ones(8, 8, 8, dtype=torch.bool)
    score_fn = build_region_score_fn(model, region_index=0, target_mask=mask)

    batched_image = torch.randn(5, 4, 8, 8, 8, generator=torch.Generator().manual_seed(2))
    scores = score_fn(batched_image)

    assert scores.shape == (5,)


# ---------------------------------------------------------------------------
# 8-9. Target mask behaviour
# ---------------------------------------------------------------------------


def test_explicit_target_mask_is_honoured() -> None:
    model = _make_model()
    image = _make_image()

    default_out = integrated_gradients(model, image, region_index=0, n_steps=8)

    mask = torch.zeros(8, 8, 8, dtype=torch.bool)
    mask[:2, :2, :2] = True
    masked_out = integrated_gradients(model, image, region_index=0, target_mask=mask, n_steps=8)

    assert masked_out.n_target_voxels == 8
    assert not torch.equal(default_out.attributions, masked_out.attributions)


def test_empty_predicted_mask_falls_back_to_whole_volume(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _make_model(AllNegativeNet)
    image = _make_image()

    with caplog.at_level(logging.WARNING):
        out = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert out.n_target_voxels == 8 * 8 * 8
    assert "falling back to the whole spatial extent" in caplog.text


# ---------------------------------------------------------------------------
# 10. Custom baseline
# ---------------------------------------------------------------------------


def test_custom_baseline_is_honoured() -> None:
    model = _make_model()
    image = _make_image()

    zero_out = integrated_gradients(model, image, region_index=0, n_steps=8)
    custom_baseline = torch.full_like(image, 0.5)
    custom_out = integrated_gradients(
        model, image, region_index=0, baseline=custom_baseline, n_steps=8
    )

    assert custom_out.baseline_score != zero_out.baseline_score
    assert not torch.equal(custom_out.attributions, zero_out.attributions)


def test_wrong_shaped_baseline_raises() -> None:
    model = _make_model()
    image = _make_image()
    wrong_baseline = torch.zeros(1, 4, 4, 4, 4)  # wrong spatial size

    with pytest.raises(ValueError, match="baseline shape"):
        integrated_gradients(model, image, region_index=0, baseline=wrong_baseline, n_steps=8)


# ---------------------------------------------------------------------------
# 11. Validation
# ---------------------------------------------------------------------------


def test_raises_when_model_is_in_train_mode() -> None:
    model = _make_model()
    model.train()
    image = _make_image()

    with pytest.raises(ValueError, match="eval"):
        integrated_gradients(model, image, region_index=0, n_steps=8)


def test_raises_on_out_of_range_region_index() -> None:
    model = _make_model()
    image = _make_image()

    with pytest.raises(ValueError, match="region_index"):
        integrated_gradients(model, image, region_index=7, n_steps=8)


# ---------------------------------------------------------------------------
# 12. Accepts unbatched input
# ---------------------------------------------------------------------------


def test_accepts_unbatched_input() -> None:
    model = _make_model()
    image = _make_image()[0]  # (C, D, H, W)

    out = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert out.attributions.shape == (1, 4, 8, 8, 8)


# ---------------------------------------------------------------------------
# 13. Works under an enclosing no_grad context
# ---------------------------------------------------------------------------


def test_works_under_enclosing_no_grad() -> None:
    model = _make_model()
    image = _make_image()

    with torch.no_grad():
        out = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert torch.isfinite(out.attributions).all()


# ---------------------------------------------------------------------------
# 14. No parameter-gradient pollution
# ---------------------------------------------------------------------------


def test_does_not_pollute_parameter_grads() -> None:
    model = _make_model()
    image = _make_image()
    assert all(p.grad is None for p in model.parameters())

    integrated_gradients(model, image, region_index=0, n_steps=8)

    # Measured directly against captum 0.9.0: IntegratedGradients uses torch.autograd.grad
    # internally, never .backward(), so no parameter's .grad should be populated. A future
    # Captum version that changed this would be caught here.
    assert all(p.grad is None for p in model.parameters())


# ---------------------------------------------------------------------------
# 15. modality_ranking
# ---------------------------------------------------------------------------


def test_modality_ranking_sorted_descending_and_matches_argmax() -> None:
    model = _make_model()
    image = _make_image()

    out = integrated_gradients(model, image, region_index=0, n_steps=8)
    ranking = modality_ranking(out)

    assert len(ranking) == 4
    assert [name for name, _ in ranking] == sorted(
        MODALITY_NAMES, key=lambda name: out.modality_attribution[name], reverse=True
    )
    fractions = [frac for _, frac in ranking]
    assert fractions == sorted(fractions, reverse=True)
    top_name = max(out.modality_attribution, key=out.modality_attribution.get)
    assert ranking[0][0] == top_name


# ---------------------------------------------------------------------------
# 16. Determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_are_bitwise_identical() -> None:
    model = _make_model()
    image = _make_image()

    first = integrated_gradients(model, image, region_index=0, n_steps=8)
    second = integrated_gradients(model, image, region_index=0, n_steps=8)

    assert torch.equal(first.attributions, second.attributions)
    assert first.target_score == second.target_score
