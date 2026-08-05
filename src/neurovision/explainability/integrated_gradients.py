"""Integrated Gradients (Sundararajan et al. 2017), adapted to 3D segmentation via Captum.

Plain Integrated Gradients (IG) attributes a single scalar output back to the input, one
attribution value per input feature, by integrating the gradient of that scalar along the
straight-line path from a `baseline` input to the real `image`. Segmentation has no single
scalar output -- like `neurovision.explainability.gradcam`, this module builds one by SUMMING
one region's logits over a chosen set of target voxels (by default, the model's own
predicted-positive voxels for that region), then runs ordinary IG against that sum. The result
is attributed back to the four input MRI MODALITY channels (T1, T1CE, T2, FLAIR), answering
"which modality did this prediction rely on" rather than "which voxel".

## Why the all-zeros baseline is defensible here, unlike in natural-image IG

IG needs a baseline representing "the absence of information" -- the point the integration path
starts from. In natural-image work an all-zeros baseline is a degenerate black image and is
routinely criticized: black is not "no information", it is a specific, recognizable image the
network may respond to in its own right. This project's preprocessing (`neurovision.data.
preprocessing.normalize_nonzero`) z-scores every modality over its own nonzero (brain-tissue)
region, so a voxel value of exactly 0 is, by construction, the MEAN intensity of that tissue.
The all-zeros baseline used here is therefore "average brain tissue", a genuinely meaningful
reference point -- not a degenerate corner case. This distinction is worth one sentence in the
paper wherever this module's figures are used.

## Hazards this module exists to navigate

1. Same three-return-type hazard as `neurovision.explainability.gradcam`:
   `neurovision.models.neurovision.NeuroVisionX.forward` only guarantees a plain `Tensor` back
   in eval mode. `integrated_gradients` therefore RAISES if `model.training` is True, and never
   calls `model.eval()` on the caller's behalf -- see that module's top-of-file docstring, point
   1, for the full reasoning (unchanged here).

2. **THE MASK MUST BE FIXED ACROSS THE INTEGRATION PATH -- this is the correctness bug that
   makes this module non-trivial, and the single thing most likely to be "fixed" wrongly by a
   later editor.** IG integrates the gradient of a scalar function `F` along the straight path
   from `baseline` to `image`. If `F(x)` were defined as "sum of region logits over whatever
   voxels x itself currently predicts positive", `F` would be a DIFFERENT function at every
   point on the path -- the predicted mask at `0.2 * image` (mostly baseline, i.e. mostly
   "average tissue") is not the mask at `image`. Integrating a quantity whose very definition
   changes underneath it is not what IG's completeness axiom (attributions sum to
   `F(image) - F(baseline)`) is proved for; the integral would be of a discontinuous,
   path-dependent quantity, the axiom would silently stop holding, the reported convergence
   delta would be large, but the attribution MAP would still look entirely plausible -- nothing
   about it visibly signals the bug. The fix: the target mask is computed EXACTLY ONCE, from the
   real `image`, in a `torch.no_grad()` pre-pass, and CLOSED OVER by the scalar wrapper
   (`build_region_score_fn`) so every point on the path reduces over the identical voxel set.
   `tests/test_integrated_gradients.py` has a regression test pinning this directly.

3. **Captum's `forward_func` must return a `(B,)` tensor, one scalar per batch element, never a
   0-D scalar.** Captum internally batches multiple points along the integration path into one
   call when `internal_batch_size > 1` (and always batches the `n_steps` interpolation points
   together when it is `None`), so the wrapper reduces over the spatial axes ONLY and keeps the
   batch axis. Reducing all the way to a 0-D scalar would silently sum across interpolation
   steps that have nothing to do with each other -- a shape error at best, silently wrong
   attributions at worst.

4. **`internal_batch_size` is a memory control, not a speed knob, and defaults to 1.** Measured
   on this project's real model (34,904,981 params, CPU, per CLAUDE.md): a single
   forward+backward is ~5.5s at 64^3 and ~17.9s at 96^3, and one 96^3 forward at batch 1 saves
   ~4.53 GB of activations in fp32. `internal_batch_size` controls how many of the `n_steps`
   interpolation points Captum evaluates in one batched forward+backward -- raising it multiplies
   that activation memory directly and can exceed this project's ~13 GB Kaggle RAM budget for no
   speed benefit worth the risk on a CPU-bound host anyway. Rough figure-generation cost at
   `n_steps=32`, 64^3: ~3 minutes per case per region. Captum's own default `n_steps=50` at 96^3:
   ~15 minutes. This is a tool for producing a handful of published figures, not something to
   run over a 189-case evaluation split -- say so wherever it is used.

5. **Always request `return_convergence_delta=True` and act on it.** IG's completeness axiom
   says the attributions should sum to `F(image) - F(baseline)`; `delta` is the Riemann-sum
   approximation's error against that identity. A figure published without checking it may be
   showing an unconverged integral with no visible sign of it. This module computes a RELATIVE
   delta (`|delta| / max(|F(image) - F(baseline)|, eps)`) and logs a WARNING above
   `delta_tolerance` (default 0.05), telling the caller to raise `n_steps`. Both the absolute and
   relative delta are returned on `IntegratedGradientsOutput` so a caller can report them
   alongside any figure.

6. **Parameter gradient pollution -- verified empirically, not assumed.** Measured directly
   (a small stub `Conv3d`, `IntegratedGradients(...).attribute(...)`, then checked every
   `p.grad`): Captum's IG implementation calls `torch.autograd.grad` internally, the same as
   `neurovision.explainability.gradcam.grad_cam` does deliberately, and leaves every parameter's
   `.grad` as `None` -- it does NOT call `.backward()` anywhere in its path. No save/restore was
   needed. `tests/test_integrated_gradients.py` pins this with a direct check (mirroring
   `test_gradcam.py`'s equivalent test) so a future Captum version that changed this behaviour
   would be caught rather than silently pollute a caller's model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from captum.attr import IntegratedGradients
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "MODALITY_NAMES",
    "IntegratedGradientsOutput",
    "build_region_score_fn",
    "integrated_gradients",
    "modality_ranking",
]

# Fixed input channel order, per neurovision.data.brats._MODALITY_ROLES. Do not reorder --
# every caller of this module (and its output's dict keys) relies on this exact mapping.
MODALITY_NAMES: tuple[str, ...] = ("T1", "T1CE", "T2", "FLAIR")

# Floor for the relative-delta denominator, so a target score that happens to equal the
# baseline score (F(image) == F(baseline), a real if unusual case) does not divide by zero.
_DELTA_EPS = 1e-8


@dataclass
class IntegratedGradientsOutput:
    """The result of one `integrated_gradients` call.

    Attributes:
        attributions: Signed per-voxel attributions, same shape as the (batched) input image,
            `(1, C, D, H, W)`. Positive values pushed the target score UP (evidence FOR the
            region), negative values pushed it down (evidence against).
        modality_attribution: Per modality (keys `MODALITY_NAMES`), the sum of `|attributions|`
            over that channel, normalized so the four values sum to 1.0. This is "how much did
            each modality drive this prediction, in either direction" -- the headline number
            for the radiological sanity check described in `integrated_gradients`'s docstring.
        modality_attribution_signed: Per modality, the raw SIGNED sum over that channel, NOT
            normalized. A large negative value means that modality argued AGAINST this region
            being present -- information the absolute-value version above erases by construction.
        convergence_delta: The absolute Riemann-sum error against IG's completeness axiom
            (`attributions.sum()` should equal `target_score - baseline_score`).
        relative_delta: `|convergence_delta| / max(|target_score - baseline_score|, eps)`. The
            number actually compared against `delta_tolerance` -- see `integrated_gradients`'s
            docstring, hazard 5.
        target_score: `F(image)`, the target scalar evaluated at the real input.
        baseline_score: `F(baseline)`, the target scalar evaluated at the baseline.
        region_index: The output channel index the attribution explains.
        n_steps: The Riemann-sum step count actually used.
        n_target_voxels: The number of voxels the target score was summed over.
    """

    attributions: Tensor
    modality_attribution: dict[str, float]
    modality_attribution_signed: dict[str, float]
    convergence_delta: float
    relative_delta: float
    target_score: float
    baseline_score: float
    region_index: int
    n_steps: int
    n_target_voxels: int


def build_region_score_fn(
    model: nn.Module, region_index: int, target_mask: Tensor
) -> Callable[[Tensor], Tensor]:
    """Builds the scalar `forward_func` Captum's `IntegratedGradients` needs.

    The returned callable evaluates `model`, picks out `region_index`'s logits, and sums them
    over `target_mask` -- reducing over the spatial axes ONLY, so a batched call (Captum's
    `internal_batch_size > 1` path) still gets one scalar per batch element. `target_mask` is
    captured by the closure at BUILD time and used unchanged for every input the returned
    function is ever called on -- see this module's top-of-file docstring, hazard 2, for why
    that fixedness is the entire point of this function existing as a separate, reusable piece
    rather than being inlined into `integrated_gradients`.

    Args:
        model: The segmentation model. Must already be in eval mode; not checked here (checked
            once by `integrated_gradients` before this function is built).
        region_index: Which output channel to sum. Not re-validated here; validated once by
            `integrated_gradients` before this function is built.
        target_mask: The FIXED voxel set to sum over, boolean or float, broadcastable to
            `model(x)[:, region_index]`'s spatial shape `(D, H, W)`.

    Returns:
        A function `f(x) -> Tensor` where `x` is `(B, C, D, H, W)` and the result is `(B,)` --
        the shape Captum's `forward_func` contract requires (this module's top-of-file
        docstring, hazard 3).

    Raises:
        ValueError: If `model(x)` does not return a plain `Tensor` when the returned function is
            called -- see `NeuroVisionX.forward`'s three possible return types; this function
            requires the eval-mode single-tensor case.
    """

    def _score(x: Tensor) -> Tensor:
        logits = model(x)
        if not isinstance(logits, Tensor):
            type_name = type(logits).__name__
            raise ValueError(
                f"build_region_score_fn's forward_func: model(x) returned a {type_name}, not a "
                "Tensor. This means the model was not actually on the plain-Tensor eval path -- "
                "see NeuroVisionX.forward's docstring for its three possible return types "
                "(Tensor / list[Tensor] / MultiTaskOutput)."
            )
        region_logits = logits[:, region_index]  # (B, D, H, W)
        mask = target_mask.to(device=region_logits.device, dtype=region_logits.dtype)
        # Reduce over spatial axes only (flatten(1) then sum(dim=1)) -- keeping the batch axis
        # is what makes this safe under Captum's internal_batch_size batching, see hazard 3.
        return (region_logits * mask).flatten(1).sum(dim=1)

    return _score


def integrated_gradients(
    model: nn.Module,
    image: Tensor,
    region_index: int = 0,
    target_mask: Tensor | None = None,
    baseline: Tensor | None = None,
    n_steps: int = 32,
    internal_batch_size: int = 1,
    threshold: float = 0.5,
    delta_tolerance: float = 0.05,
) -> IntegratedGradientsOutput:
    """Computes Integrated Gradients for one region channel of `model`'s output on `image`.

    Algorithm:

    1. Validate `model.training is False`; accept `image` as `(1, C, D, H, W)` or
       `(C, D, H, W)` (unsqueezed automatically), batch size 1.
    2. Pre-pass under `torch.no_grad()`: `logits = model(image)`, validated as a plain `Tensor`
       with `region_index` in range. Build the FIXED target mask -- `target_mask` if given, else
       `sigmoid(logits[0, region_index]) >= threshold` (the model's own predicted positives). If
       that predicted set is empty, fall back to the whole spatial extent and log a warning
       (normal in BraTS -- many cases have no enhancing tumor, see CLAUDE.md's measured 2.6%
       figure). Record `n_target_voxels`.
    3. Baseline: `torch.zeros_like(image)` when `baseline is None` -- see this module's
       top-of-file docstring for why an all-zeros baseline is a meaningful "average tissue"
       reference here, unlike in natural-image IG. A supplied baseline must match `image`'s
       shape exactly.
    4. Build the scalar wrapper via `build_region_score_fn`, closing over the FIXED mask from
       step 2 -- see this module's top-of-file docstring, hazard 2, for why recomputing the mask
       per integration step would break IG's completeness axiom.
    5. `captum.attr.IntegratedGradients(forward_func).attribute(image, baselines=baseline,
       n_steps=n_steps, internal_batch_size=internal_batch_size,
       return_convergence_delta=True)`.
    6. `target_score = F(image)`, `baseline_score = F(baseline)` (under `no_grad`, via the same
       fixed-mask scalar wrapper), and the relative delta; warns per hazard 5 above
       `delta_tolerance`.
    7. `modality_attribution`: for each of the 4 input channels, `attributions[0,
       c].abs().sum()`, normalized so the four values sum to 1.0 -- "which modality drove this
       prediction, in either direction". `modality_attribution_signed`: the raw signed sum per
       channel, NOT normalized -- a large negative value means that modality argued AGAINST the
       region, information the absolute-value version erases.

    RADIOLOGICAL SANITY CHECK -- the main intended scientific use of this module. Enhancing
    tumor (ET) is *defined* clinically by contrast uptake, which is a T1CE finding; whole-tumor
    (WT) includes peritumoral edema, which is chiefly a FLAIR finding. A correctly-trained model
    should therefore show T1CE dominating `modality_attribution` for `region_index` ET (0) and
    FLAIR dominating it for WT (2). If it does not, the model may be reaching its Dice score
    through features that are not the radiologically meaningful ones for that region -- a
    reportable finding about the model, not a bug in this function. Checking this is the
    headline use `integrated_gradients` exists for.

    Args:
        model: The segmentation model. Must already be in eval mode (`model.eval()`) -- see the
            `Raises` section below. This function never flips the mode itself, for the same
            reason `neurovision.explainability.gradcam.grad_cam` does not (see that module's
            top-of-file docstring, point 1).
        image: Input volume, shape `(1, C, D, H, W)` or `(C, D, H, W)`.
        region_index: Which output channel to explain (0/1/2 for ET/TC/WT in this project's
            convention). Must be within the model's actual output channel count.
        target_mask: The region of interest to sum the target score over, `(D, H, W)` or
            broadcastable to it. `None` (the default) uses the model's own predicted positives
            for `region_index` -- see algorithm step 2.
        baseline: The IG baseline, same shape as `image`. `None` (the default) uses an
            all-zeros image -- see this module's top-of-file docstring for why that is
            meaningful here rather than degenerate.
        n_steps: Riemann-sum step count for the path integral. See hazard 4 above for measured
            per-step cost; this function's default (32) trades some convergence accuracy for
            keeping a single figure-generation call in the low minutes.
        internal_batch_size: How many of the `n_steps` interpolation points Captum evaluates in
            one batched forward+backward. Defaults to 1 -- see hazard 4 above for why raising it
            is a memory decision, not a speed one, on this project's model sizes.
        threshold: Probability threshold used to build the default predicted-positive mask.
            Unused when `target_mask` is given explicitly.
        delta_tolerance: Relative-delta threshold above which a convergence warning is logged
            (see hazard 5 above). Does not raise -- an unconverged integral is still returned,
            just flagged, so a caller can decide whether to re-run with more steps.

    Returns:
        An `IntegratedGradientsOutput`. See that dataclass's docstring for each field.

    Raises:
        ValueError: If `model.training` is True (call `model.eval()` yourself first); if `image`
            is not `(1, C, D, H, W)` or `(C, D, H, W)`, or has a batch size other than 1; if
            `model(image)` does not return a plain `Tensor` (see `NeuroVisionX.forward`'s three
            possible return types); if `region_index` is outside the model's actual output
            channel count; or if a supplied `baseline` does not match `image`'s shape.
    """
    if model.training:
        raise ValueError(
            "integrated_gradients requires model.training is False (eval mode). "
            "NeuroVisionX.forward (and most segmentation models generally) only guarantee a "
            "single logits Tensor back in eval mode -- in training mode NeuroVisionX may "
            "return a list[Tensor] (deep supervision) or a MultiTaskOutput (auxiliary heads), "
            "neither of which this function knows how to pick a single region's logits out of. "
            "Call model.eval() yourself before calling integrated_gradients -- this function "
            "deliberately does not flip your model's mode for you, see "
            "neurovision.inference.mc_dropout's top-of-file docstring for why silently doing "
            "that is exactly the class of bug this project avoids."
        )

    if image.ndim == 4:
        image = image.unsqueeze(0)
    elif image.ndim != 5:
        raise ValueError(f"image must be (1, C, D, H, W) or (C, D, H, W), got ndim={image.ndim}.")
    if image.shape[0] != 1:
        raise ValueError(
            f"integrated_gradients supports a single-volume batch only, got batch size "
            f"{image.shape[0]}."
        )

    if baseline is None:
        # See this module's top-of-file docstring: preprocessing z-scores each modality over
        # its nonzero region, so 0 is "average brain tissue", not a degenerate black image.
        baseline = torch.zeros_like(image)
    else:
        if baseline.ndim == 4:
            baseline = baseline.unsqueeze(0)
        if tuple(baseline.shape) != tuple(image.shape):
            raise ValueError(
                f"baseline shape {tuple(baseline.shape)} does not match image shape "
                f"{tuple(image.shape)}."
            )

    # Pre-pass, no_grad: this is where the FIXED target mask is built -- see this module's
    # top-of-file docstring, hazard 2. Everything downstream (the scalar wrapper, target_score,
    # baseline_score) reduces over this exact voxel set and never recomputes it.
    with torch.no_grad():
        logits = model(image)
        if not isinstance(logits, Tensor):
            type_name = type(logits).__name__
            raise ValueError(
                f"integrated_gradients: model(image) returned a {type_name}, not a Tensor. This "
                "means the model was not actually on the plain-Tensor eval path -- see "
                "NeuroVisionX.forward's docstring for its three possible return types (Tensor / "
                "list[Tensor] / MultiTaskOutput). model.training was checked False above, so a "
                "Tensor was expected; a non-standard model or a model that ignores eval-mode "
                "conventions is producing something integrated_gradients cannot pick a single "
                "region's logits out of."
            )

        num_channels = logits.shape[1]
        if not (0 <= region_index < num_channels):
            raise ValueError(
                f"region_index ({region_index}) is out of range for model output with "
                f"{num_channels} channels."
            )

        region_logits = logits[0, region_index]  # (D, H, W)

        if target_mask is None:
            positive_mask = torch.sigmoid(region_logits) >= threshold
            if not torch.any(positive_mask):
                logger.warning(
                    "integrated_gradients: the model's own predicted-positive mask for "
                    "region_index=%d at threshold=%.3f is empty; falling back to the whole "
                    "spatial extent. Normal when this volume has no predicted foreground for "
                    "this region (e.g. no enhancing tumor).",
                    region_index,
                    threshold,
                )
                positive_mask = torch.ones_like(region_logits, dtype=torch.bool)
            mask_t = positive_mask
        else:
            mask_t = torch.as_tensor(target_mask).to(device=region_logits.device, dtype=torch.bool)
            mask_t = torch.broadcast_to(mask_t, region_logits.shape)

        n_target_voxels = int(mask_t.sum().item())

    score_fn = build_region_score_fn(model, region_index, mask_t)

    ig = IntegratedGradients(score_fn)
    # Captum enables autograd internally (verified empirically -- see this module's top-of-file
    # docstring, hazard 6), so this call works even if the caller has this whole function
    # wrapped in an enclosing torch.no_grad(), same as neurovision.explainability.gradcam
    # guarantees for grad_cam.
    attributions, delta = ig.attribute(
        image,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    attributions = attributions.detach()

    with torch.no_grad():
        target_score = float(score_fn(image).item())
        baseline_score = float(score_fn(baseline).item())

    convergence_delta = float(delta.reshape(-1)[0].item())
    relative_delta = abs(convergence_delta) / max(abs(target_score - baseline_score), _DELTA_EPS)

    if relative_delta > delta_tolerance:
        logger.warning(
            "integrated_gradients: relative convergence delta %.4f exceeds delta_tolerance "
            "%.4f (n_steps=%d). The attributions may not have converged -- IG's completeness "
            "axiom (attributions should sum to target_score - baseline_score) is only "
            "approximately satisfied. Consider raising n_steps.",
            relative_delta,
            delta_tolerance,
            n_steps,
        )

    num_input_channels = attributions.shape[1]
    if num_input_channels != len(MODALITY_NAMES):
        raise ValueError(
            f"integrated_gradients: image has {num_input_channels} channels but "
            f"MODALITY_NAMES has {len(MODALITY_NAMES)} entries {MODALITY_NAMES}. This module "
            "assumes the project's fixed 4-modality (T1, T1CE, T2, FLAIR) channel order."
        )

    abs_sums = [float(attributions[0, c].abs().sum().item()) for c in range(num_input_channels)]
    total_abs = sum(abs_sums)
    if total_abs == 0.0:
        # A real and meaningful outcome (e.g. the target score is exactly flat along the whole
        # path), not a bug -- return an even split rather than dividing by zero.
        logger.warning(
            "integrated_gradients: total |attribution| is exactly zero; returning an even split "
            "across modalities instead of dividing by zero."
        )
        modality_attribution = {name: 1.0 / len(MODALITY_NAMES) for name in MODALITY_NAMES}
    else:
        modality_attribution = {
            name: abs_sums[c] / total_abs for c, name in enumerate(MODALITY_NAMES)
        }
    modality_attribution_signed = {
        name: float(attributions[0, c].sum().item()) for c, name in enumerate(MODALITY_NAMES)
    }

    return IntegratedGradientsOutput(
        attributions=attributions,
        modality_attribution=modality_attribution,
        modality_attribution_signed=modality_attribution_signed,
        convergence_delta=convergence_delta,
        relative_delta=relative_delta,
        target_score=target_score,
        baseline_score=baseline_score,
        region_index=region_index,
        n_steps=n_steps,
        n_target_voxels=n_target_voxels,
    )


def modality_ranking(output: IntegratedGradientsOutput) -> list[tuple[str, float]]:
    """Sorts `output.modality_attribution` descending -- what a caller actually wants to read.

    Trivial by design: keeps the one place this project decides "most important modality first"
    in one function, rather than every caller re-sorting the dict itself.

    Args:
        output: The result of an `integrated_gradients` call.

    Returns:
        `[(modality_name, fraction), ...]`, length 4, sorted descending by fraction.
    """
    return sorted(output.modality_attribution.items(), key=lambda item: item[1], reverse=True)
