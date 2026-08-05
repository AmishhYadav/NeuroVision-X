"""Seg-Grad-CAM: Grad-CAM adapted to 3D semantic segmentation (Vinogradova et al. 2020).

Plain Grad-CAM differentiates a single class logit. Segmentation has no such thing -- the
network emits a whole VOLUME of logits per output channel, not one score. Seg-Grad-CAM's fix
is to build the target score by SUMMING the logits of one region over a chosen region of
interest (by default, the model's own predicted-positive voxels for that region), then run
ordinary Grad-CAM against that sum: global-average-pool the gradient of the sum with respect
to a chosen layer's activation into one weight per channel, and use those weights to combine
the activation channels into a class activation map. See `grad_cam`'s docstring for the exact
steps and for guidance on which layer to pick.

## Hazards this module exists to navigate

1. `neurovision.models.neurovision.NeuroVisionX.forward` returns one of THREE Python types
   depending on `self.training`: a `MultiTaskOutput` dataclass, a `list[Tensor]` (deep
   supervision), or a plain `Tensor` -- and the `Tensor` case is the ONLY one guaranteed in
   eval mode. `grad_cam` therefore RAISES if `model.training` is True, and does NOT silently
   call `model.eval()` on the caller's behalf. Flipping a caller's model mode behind their back
   is exactly the class of bug `neurovision.inference.mc_dropout`'s top-of-file docstring warns
   about: the caller may be mid-way through something that depends on the model's current mode
   (e.g. a training loop that will resume after this call), and this function has no way to
   know it is safe to change that. Call `model.eval()` yourself first -- the requirement is
   visible in the raised error rather than silently patched over.

2. Gradients are computed with `torch.autograd.grad(score, activation)`, NEVER
   `tensor.backward()`. `.backward()` accumulates into every parameter's `.grad` field as a
   side effect, silently polluting the model -- if the caller later resumes training from this
   same model object, those stale gradients would be summed into the first optimizer step
   without anyone asking for that. `torch.autograd.grad` returns exactly the gradient asked
   for and touches no `.grad` field. `tests/test_gradcam.py` pins this with a direct check that
   every parameter's `.grad` is still `None` after a `grad_cam` call.

3. This function needs a forward pass that BUILDS a gradient graph, even though the model is
   in eval mode and the caller may well have the whole surrounding evaluation loop wrapped in
   `torch.no_grad()` (a normal and expected thing to do around inference code, and NOT
   something this module can assume has been undone for it). So the forward pass, target-mask
   construction, and gradient call are all wrapped in an explicit `with torch.enable_grad():`
   rather than assuming grad mode is already on.

4. Memory forces PATCH-based operation, never a whole preprocessed BraTS volume. Measured
   elsewhere in this project (CLAUDE.md): a single 96^3 forward pass at batch 1, fp32, needs
   roughly 4.53 GB of saved activations with only the Swin branch checkpointed. The median
   preprocessed BraTS volume is (137, 171, 140) -- about 3.7x more voxels than a 96^3 patch --
   so a whole-volume backward pass would need on the order of 17 GB, which does not fit this
   project's ~13 GB Kaggle RAM budget (let alone a laptop). `center_patch_on_mask` below is the
   intended workflow: crop a patch around whatever region a saved prediction says is
   interesting, and run `grad_cam` on that patch alone. This module does NOT attempt any
   sliding-window aggregation of CAMs across patches -- stitching together gradients computed
   from independent, disjoint receptive fields would be a different and dubious quantity, not
   a whole-volume Grad-CAM, and is out of scope here.

5. Hooks are removed in a `finally` block. A forward hook left attached after an exception
   would silently change every subsequent forward pass of that model -- an easy way to corrupt
   an otherwise-unrelated evaluation run days later with no visible cause.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "GradCAMOutput",
    "available_layers",
    "resolve_layer",
    "center_patch_on_mask",
    "grad_cam",
]

# Parameterless activation/dropout modules produce no new feature-map information of their
# own (they are elementwise functions of their input), so `available_layers` filters them out
# -- picking one as a Grad-CAM target would just be picking its predecessor's map, relu'd or
# masked. Everything else (conv, norm, container modules such as nn.Sequential, ...) is left
# in: this project's models nest feature-producing blocks under container modules, and a user
# discovering valid `target_layer` strings needs those names too.
_NON_FEATURE_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.GELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@dataclass
class GradCAMOutput:
    """The result of one `grad_cam` call.

    Attributes:
        cam: The class activation map, upsampled to `image`'s spatial shape (or left at the
            target layer's own resolution if `upsample=False` -- see `grad_cam`'s `upsample`
            argument). Shape `(1, 1, D, H, W)`. min-max normalized to `[0, 1]` when
            `normalize=True` (the default).
        raw_cam: The same map BEFORE upsampling, at the target layer's own spatial
            resolution. Shape `(1, 1, d, h, w)`. Kept alongside `cam` so a caller can inspect
            the map's true resolution without re-running `grad_cam` -- see the "resolution"
            warning in `grad_cam`'s docstring, point 10.
        channel_weights: The Grad-CAM weights `alpha_k`, one per target-layer feature
            channel -- the global-average-pooled gradient of the target score with respect
            to that channel. Shape `(K,)`.
        target_layer: The dotted module name the CAM was computed against. Stored on the
            output (not just passed as an argument the caller already has) because a figure
            built from this result should be able to report its own provenance -- a Grad-CAM
            figure without its target layer named is not reproducible, see `grad_cam`'s
            docstring.
        target_score: The scalar target score `S` (a Python float) the gradient was taken
            of -- the sum of `region_index`'s logits over the target voxels.
        region_index: The output channel index the CAM explains.
        n_target_voxels: The number of voxels the target score was summed over.
    """

    cam: Tensor
    raw_cam: Tensor
    channel_weights: Tensor
    target_layer: str
    target_score: float
    region_index: int
    n_target_voxels: int


def available_layers(model: nn.Module, max_depth: int | None = None) -> list[str]:
    """Lists dotted module names that are plausible Grad-CAM targets.

    Every named submodule of `model` is a candidate EXCEPT the root module itself (name `""`,
    which is `model` -- not a meaningful "layer" to target) and parameterless
    activation/dropout modules (see `_NON_FEATURE_TYPES`), which are elementwise functions of
    their input and contribute no new feature-map information of their own.

    This exists so a user can discover valid `target_layer` strings for `grad_cam` without
    reading the model's source.

    Args:
        model: Any `nn.Module`.
        max_depth: If given, only names with at most this many dot-separated components are
            returned (a name with no dots has depth 1). `None` (the default) returns every
            depth.

    Returns:
        Dotted module names, in `model.named_modules()` order.
    """
    names: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, _NON_FEATURE_TYPES):
            continue
        depth = name.count(".") + 1
        if max_depth is not None and depth > max_depth:
            continue
        names.append(name)
    return names


def _near_miss_candidates(name: str, all_names: Sequence[str], limit: int = 5) -> list[str]:
    """Finds existing module names sharing the longest dotted prefix with `name`.

    Walks the requested name's dot-separated components from most-specific to least-specific,
    stopping at the first prefix that matches (or is the prefix of) a real module name. Falls
    back to an arbitrary handful of real names if not even the first component matches
    anything, so the caller always gets at least one concrete, correct example to look at.

    Args:
        name: The (invalid) dotted name that was requested.
        all_names: Every real dotted module name on the model (excluding the root).
        limit: Maximum number of candidates to return.

    Returns:
        Up to `limit` real dotted names, sorted.
    """
    parts = name.split(".")
    for k in range(len(parts), 0, -1):
        prefix = ".".join(parts[:k])
        matches = [n for n in all_names if n == prefix or n.startswith(prefix + ".")]
        if matches:
            return sorted(matches)[:limit]
    return sorted(all_names)[:limit]


def resolve_layer(model: nn.Module, name: str) -> nn.Module:
    """Looks up a dotted module name, raising a helpful error on a typo.

    Args:
        model: Any `nn.Module`.
        name: A dotted submodule name, e.g. `"decoder.stages.1.conv2"`.

    Returns:
        `model.get_submodule(name)`.

    Raises:
        ValueError: If no such submodule exists. The message includes a short list of real
            module names sharing the longest matching dotted prefix with `name` -- a bare
            `AttributeError` from `get_submodule` is not useful when only one component of a
            five-component path was mistyped.
    """
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        all_names = [n for n, _ in model.named_modules() if n != ""]
        candidates = _near_miss_candidates(name, all_names)
        raise ValueError(
            f"No submodule named {name!r} on this model. Closest real names (by longest "
            f"shared dotted prefix): {candidates}. Call "
            "neurovision.explainability.gradcam.available_layers(model) to list every valid "
            "target_layer name."
        ) from exc


def center_patch_on_mask(
    image: Tensor, mask: Tensor, patch_size: Sequence[int]
) -> tuple[Tensor, tuple[slice, ...]]:
    """Crops a `patch_size` patch centred on `mask`'s centre of mass, clamped inside the volume.

    This is the intended workflow for running `grad_cam` on a real BraTS volume -- see point 4
    of this module's top-of-file docstring for why a whole-volume backward pass does not fit
    the project's memory budget. Typical use: run inference, take one region channel of the
    (thresholded) prediction as `mask`, crop a patch around it, and run `grad_cam` on the
    crop -- the returned slices let the caller place the resulting CAM back into full-volume
    geometry if needed.

    Args:
        image: The volume to crop, shape `(B, C, D, H, W)` with `B == 1`, or `(C, D, H, W)`.
            The returned patch keeps whichever of these two layouts was passed in.
        mask: Boolean or float, `(D, H, W)`, or a tensor broadcastable to that shape (e.g. a
            leading size-1 batch/channel axis is squeezed away) -- typically one region
            channel of a saved, thresholded prediction.
        patch_size: `(patch_D, patch_H, patch_W)`.

    Returns:
        A tuple `(patch, slices)`:

        - `patch`: The cropped volume, same number of dimensions as `image`, spatial shape
          exactly `patch_size`.
        - `slices`: A 3-tuple of `slice` objects for the `(D, H, W)` axes, such that indexing
          `image`'s spatial axes with them reproduces `patch`'s spatial content exactly.

    Raises:
        ValueError: If `image` is not `(B, C, D, H, W)` with `B == 1`, or `(C, D, H, W)`; if
            `len(patch_size) != 3`; or if `patch_size` exceeds the volume on some axis (named
            in the message) -- padding instead would change what the network actually sees,
            so this is a hard error rather than a silent pad.
    """
    if image.ndim == 5:
        if image.shape[0] != 1:
            raise ValueError(
                f"center_patch_on_mask supports a single-volume batch only, got batch size "
                f"{image.shape[0]}."
            )
        was_5d = True
        spatial_image = image
    elif image.ndim == 4:
        was_5d = False
        spatial_image = image.unsqueeze(0)
    else:
        raise ValueError(
            "image must be (B, C, D, H, W) with B == 1, or (C, D, H, W); got ndim=" f"{image.ndim}."
        )

    patch_size = tuple(int(p) for p in patch_size)
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must have exactly 3 entries (D, H, W), got {patch_size}.")

    spatial_shape = tuple(spatial_image.shape[2:])  # (D, H, W)
    axis_names = ("D", "H", "W")
    for axis, (p, s) in enumerate(zip(patch_size, spatial_shape, strict=True)):
        if p > s:
            raise ValueError(
                f"patch_size axis {axis_names[axis]} ({p}) exceeds the volume's axis "
                f"{axis_names[axis]} ({s}). Padding instead of raising would change what the "
                "network actually sees, so this is a hard error rather than a silent pad."
            )

    mask_t = torch.as_tensor(mask).detach()
    while mask_t.ndim > 3:
        if mask_t.shape[0] != 1:
            raise ValueError(
                f"mask has unexpected shape {tuple(mask_t.shape)}; expected (D, H, W) or a "
                "shape broadcastable to it (leading size-1 axes only)."
            )
        mask_t = mask_t.squeeze(0)
    mask_bool = torch.broadcast_to(mask_t.to(torch.bool), spatial_shape)

    nonzero = torch.nonzero(mask_bool, as_tuple=False)
    if nonzero.numel() == 0:
        logger.warning(
            "center_patch_on_mask: mask is entirely empty; falling back to the volume centre. "
            "Normal when a region is entirely absent from a case (e.g. no predicted enhancing "
            "tumor), not an error."
        )
        center = [s // 2 for s in spatial_shape]
    else:
        center = [int(nonzero[:, axis].float().mean().round().item()) for axis in range(3)]

    slices: list[slice] = []
    for c, p, s in zip(center, patch_size, spatial_shape, strict=True):
        start = c - p // 2
        start = max(0, min(start, s - p))  # clamp so the patch stays inside the volume
        slices.append(slice(start, start + p))
    slices_tuple = tuple(slices)

    patch_5d = spatial_image[:, :, slices_tuple[0], slices_tuple[1], slices_tuple[2]]
    patch = patch_5d if was_5d else patch_5d.squeeze(0)
    return patch, slices_tuple


def grad_cam(
    model: nn.Module,
    image: Tensor,
    target_layer: str,
    region_index: int = 0,
    target_mask: Tensor | None = None,
    threshold: float = 0.5,
    relu: bool = True,
    upsample: bool = True,
    normalize: bool = True,
) -> GradCAMOutput:
    """Computes Seg-Grad-CAM for one region channel of `model`'s output on `image`.

    Which layer to pick (`target_layer`). This is an empirical choice, and the layer used
    MUST be reported alongside any Grad-CAM figure -- a figure without its target layer named
    is not reproducible.

    - The final decoder stage (full resolution) produces a CAM that is nearly the
      segmentation mask itself: sharp, but not very informative, since it sits only one
      1x1x1 convolution away from the output logits.
    - The bottleneck, or the coarsest fusion block, gives the most SEMANTIC map, but at a
      steep cost in resolution -- at a 96^3 patch that is stride 16, i.e. a 6^3 = 216-voxel
      map upsampled 16x, which is very blurry.
    - A MIDDLE decoder stage is usually the informative compromise between the two.

    Algorithm (Seg-Grad-CAM, Vinogradova et al. 2020):

    1. Hook the target layer's forward output, `A`, shape `(1, K, d, h, w)`.
    2. Run `model(image)` with gradients enabled.
    3. Build a target mask over `logits[0, region_index]` (the model's own predicted
       positives by default -- see `target_mask` below).
    4. Target score `S = logits[0, region_index][mask].sum()`. Summed over the target
       region rather than a single voxel's logit, deliberately: a single voxel's gradient is
       dominated by that one voxel's own receptive field and produces a near-delta CAM that
       says almost nothing; summing over the whole predicted region instead asks "what
       evidence supported this region as a whole", which is the question a Grad-CAM figure is
       meant to answer.
    5. `grads = d(S)/d(A)`, computed with `torch.autograd.grad` (never `.backward()` -- see
       this module's top-of-file docstring, point 2).
    6. `alpha_k = mean(grads[:, k])` over the spatial dims -- global average pooling of the
       gradient, one weight per channel of `A`.
    7. `raw_cam = relu(sum_k alpha_k * A[:, k])` (the `relu` argument controls whether the
       ReLU is applied).

    Args:
        model: The segmentation model. Must already be in eval mode (`model.eval()`) --
            see the `Raises` section below.
        image: Input volume, shape `(1, C, D, H, W)` or `(C, D, H, W)` (unsqueezed
            automatically). See point 4 of this module's top-of-file docstring for why this
            should be a small PATCH, not a whole preprocessed BraTS volume --
            `center_patch_on_mask` is the intended way to produce one.
        target_layer: A dotted module name, e.g. `"decoder.stages.1.conv2"`. Use
            `available_layers(model)` to list valid names, and see the layer-choice guidance
            above.
        region_index: Which output channel to explain (e.g. 0/1/2 for ET/TC/WT in this
            project's convention). Must be within the model's actual output channel count.
        target_mask: The region of interest to sum the target score over, shape `(D, H, W)`
            or broadcastable to it. `None` (the default) uses the model's OWN predicted
            positives for `region_index`, i.e. `sigmoid(logits[0, region_index]) >= threshold`.
            If that predicted-positive set is empty (a real and normal event -- e.g. no
            enhancing tumor predicted in this patch), this falls back to the WHOLE spatial
            extent and logs a warning rather than raising.
        threshold: Probability threshold used to build the default predicted-positive mask.
            Unused when `target_mask` is given explicitly.
        relu: If True (the default, matching the original Grad-CAM convention), the CAM is
            clipped to `>= 0`, keeping only evidence FOR the region and discarding evidence
            against it. If False, the returned map is SIGNED and strictly more informative --
            worth showing in a supplementary figure, since it also shows which voxels argued
            against this prediction.
        upsample: If True (the default), `raw_cam` is trilinear-interpolated
            (`align_corners=False`) up to `image`'s spatial shape and stored as `cam`. If
            False, `cam` is left at the target layer's own (coarser) resolution.
            IMPORTANT: the CAM's TRUE resolution is always the target layer's, not the
            upsampled shape -- an upsampled CAM must never be presented as if it had
            voxel-level precision.
        normalize: If True (the default), `cam` is min-max scaled to `[0, 1]`. A CAM that is
            perfectly constant (max == min -- e.g. an all-zero map after ReLU, meaning no
            positive evidence was found at this layer, which is a real and meaningful outcome
            rather than a bug) is returned as all zeros instead of dividing by zero, with a
            warning logged.

    Returns:
        A `GradCAMOutput`. See that dataclass's docstring for each field.

    Raises:
        ValueError: If `model.training` is True (call `model.eval()` yourself first -- this
            function will not do it for you, see this module's top-of-file docstring, point
            1); if `image` is not `(1, C, D, H, W)` or `(C, D, H, W)`; if `target_layer` does
            not name a real submodule (see `resolve_layer`); if `model(image)` does not
            return a plain `Tensor` (i.e. the model was not actually on the eval-mode
            single-tensor path -- see `NeuroVisionX.forward`'s docstring for its three
            possible return types); or if `region_index` is outside the model's actual output
            channel count.
    """
    if model.training:
        raise ValueError(
            "grad_cam requires model.training is False (eval mode). "
            "NeuroVisionX.forward (and most segmentation models generally) only guarantee a "
            "single logits Tensor back in eval mode -- in training mode NeuroVisionX may "
            "return a list[Tensor] (deep supervision) or a MultiTaskOutput (auxiliary "
            "heads), neither of which this function knows how to pick a single region's "
            "logits out of. Call model.eval() yourself before calling grad_cam -- this "
            "function deliberately does not flip your model's mode for you, see "
            "neurovision.inference.mc_dropout's top-of-file docstring for why silently doing "
            "that is exactly the class of bug this project avoids."
        )

    if image.ndim == 4:
        image = image.unsqueeze(0)
    elif image.ndim != 5:
        raise ValueError(f"image must be (1, C, D, H, W) or (C, D, H, W), got ndim={image.ndim}.")
    if image.shape[0] != 1:
        raise ValueError(
            f"grad_cam supports a single-volume batch only, got batch size {image.shape[0]}."
        )

    layer = resolve_layer(model, target_layer)

    activation_holder: dict[str, Tensor] = {}

    def _capture_activation(_module: nn.Module, _inputs: tuple, output: Tensor) -> None:
        activation_holder["activation"] = output

    handle = layer.register_forward_hook(_capture_activation)

    try:
        # Point 3 of this module's top-of-file docstring: gradients are required even though
        # the model is in eval mode and a caller may well have this whole call wrapped in an
        # enclosing torch.no_grad() (a normal thing to do around inference code).
        with torch.enable_grad():
            logits = model(image)

            if not isinstance(logits, Tensor):
                type_name = type(logits).__name__
                raise ValueError(
                    f"grad_cam: model(image) returned a {type_name}, not a Tensor. This means "
                    "the model was not actually on the plain-Tensor eval path -- see "
                    "NeuroVisionX.forward's docstring for its three possible return types "
                    "(Tensor / list[Tensor] / MultiTaskOutput). model.training was checked "
                    "False above, so a Tensor was expected; a non-standard model or a model "
                    "that ignores eval-mode conventions is producing something grad_cam "
                    "cannot pick a single region's logits out of."
                )

            num_channels = logits.shape[1]
            if not (0 <= region_index < num_channels):
                raise ValueError(
                    f"region_index ({region_index}) is out of range for model output with "
                    f"{num_channels} channels."
                )

            activation = activation_holder.get("activation")
            if activation is None:
                raise ValueError(
                    f"Target layer {target_layer!r} never ran during this forward pass "
                    "(its forward hook never fired), so there is no activation to build a CAM "
                    "from. Check the layer is actually on the path the model executed."
                )

            region_logits = logits[0, region_index]  # (D, H, W)

            if target_mask is None:
                with torch.no_grad():
                    positive_mask = torch.sigmoid(region_logits) >= threshold
                if not torch.any(positive_mask):
                    logger.warning(
                        "grad_cam: the model's own predicted-positive mask for "
                        "region_index=%d at threshold=%.3f is empty; falling back to the "
                        "whole spatial extent. Normal when this patch has no predicted "
                        "foreground for this region.",
                        region_index,
                        threshold,
                    )
                    positive_mask = torch.ones_like(region_logits, dtype=torch.bool)
                mask_t = positive_mask
            else:
                mask_t = torch.as_tensor(target_mask).to(
                    device=region_logits.device, dtype=torch.bool
                )
                mask_t = torch.broadcast_to(mask_t, region_logits.shape)

            n_target_voxels = int(mask_t.sum().item())
            # Sum over the target region, not a single voxel -- see this function's docstring,
            # algorithm step 4, for why.
            score = region_logits[mask_t].sum()

            # torch.autograd.grad, never score.backward() -- see this module's top-of-file
            # docstring, point 2: .backward() would accumulate into every parameter's .grad,
            # silently polluting the model for any later training step.
            grads = torch.autograd.grad(score, activation)[0]  # (1, K, d, h, w)
    finally:
        # Removed even if the block above raised -- a hook left attached after an exception
        # would silently change every later forward pass of this model.
        handle.remove()

    num_target_channels = grads.shape[1]
    alpha = grads.mean(dim=(2, 3, 4))  # (1, K) -- global average pool of the gradient
    raw_cam = (alpha.view(1, num_target_channels, 1, 1, 1) * activation).sum(dim=1, keepdim=True)

    if relu:
        # Keeps only evidence FOR the region, discarding evidence against it -- the original
        # Grad-CAM convention. relu=False returns the signed map instead (see the docstring).
        raw_cam = F.relu(raw_cam)

    cam = raw_cam
    if upsample:
        # The CAM's TRUE resolution is the target layer's; upsampling makes it presentable
        # alongside the input volume but must never be read as voxel-level precision.
        cam = F.interpolate(
            raw_cam, size=tuple(image.shape[2:]), mode="trilinear", align_corners=False
        )

    if normalize:
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min).item() == 0.0:
            logger.warning(
                "grad_cam: the CAM is constant (max == min); returning it as all zeros instead "
                "of dividing by zero. A flat, all-zero map after ReLU is a real and meaningful "
                "outcome (no positive evidence found at this layer), not a bug to hide."
            )
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

    return GradCAMOutput(
        cam=cam.detach(),
        raw_cam=raw_cam.detach(),
        channel_weights=alpha.detach().squeeze(0),
        target_layer=target_layer,
        target_score=float(score.detach().item()),
        region_index=region_index,
        n_target_voxels=n_target_voxels,
    )
