"""Monte Carlo dropout uncertainty estimation.

Runs several stochastic sliding-window forward passes over the same volume
and decomposes the spread across those passes into two uncertainty
components: how much of the model's uncertainty comes from noisy/ambiguous
input (aleatoric, does not shrink with more data) versus how much comes from
the model itself not being sure what it learned (epistemic, shrinks as the
model sees more training data or MC samples). See "The uncertainty maths"
in `mc_dropout_predict`'s docstring for the exact formulas.

## The central hazard this module exists to navigate

`neurovision.models.neurovision.NeuroVisionX.forward` returns one of THREE
different Python types depending on `self.training` and which heads are
enabled: a `MultiTaskOutput` dataclass (training, an auxiliary head
enabled), a `list[Tensor]` (training, `deep_supervision_levels > 1`, no
auxiliary head), or a plain `Tensor` (every other case, including ALWAYS in
eval mode). MC-dropout wants dropout's stochastic masks active, and the
naive way to get that is `model.train()` -- but that flips the return-type
switch and hands `neurovision.inference.sliding_window.sliding_window_predict`
(and MONAI's `SlidingWindowInferer` underneath it, which calls `model(patch)`
and assumes a `Tensor` back) a dataclass or a list instead. There is no shape
error anywhere; it just silently breaks.

So `dropout_enabled` below never touches `model.training` at all. It walks
`model.modules()` and puts only the dropout submodules into train mode
individually, leaving every other submodule -- and the model itself -- in
eval. That is sufficient: `nn.Dropout3d.forward` only applies its random mask
when `self.training` is True, and that flag is checked on the dropout module
itself, not inherited from a parent flag at call time (`nn.Module.eval()`
sets it recursively at CALL time, but nothing re-checks the parent
afterward).

A second, easy-to-miss version of the same hazard lives inside
`sliding_window_predict` itself: it calls `model.eval()` on every invocation
(so a stray caller can't accidentally run inference in train mode). Because
`nn.Module.eval()` recurses into every child, calling it AFTER
`dropout_enabled` has switched the target dropout modules into train mode
would silently switch them back off -- turning every one of the N "stochastic"
passes below into the identical, deterministic forward pass, and producing a
mutual-information map that is exactly zero everywhere with no error raised
anywhere. `mc_dropout_predict` avoids this by passing `set_eval=False` to
`sliding_window_predict` -- see that function's docstring for the other half
of this story. `tests/test_mc_dropout.py`'s N=5-on-a-stochastic-model test is
the one test in this file that would silently pass even if this fix were
missing (every OTHER test here only checks internal consistency of the
returned numbers, not that they came from genuinely different passes) --
do not delete it.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from neurovision.inference.sliding_window import build_inferer, sliding_window_predict
from neurovision.utils.device import amp_enabled

logger = logging.getLogger(__name__)

# The full family of dropout layers torch ships. isinstance against this
# tuple, rather than a hand-rolled name check, is what makes the set both
# greppable (one place lists every targeted type) and trivially extendable
# if torch ever adds another dropout variant.
_DROPOUT_TYPES: tuple[type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


@dataclass
class MCDropoutOutput:
    """The result of one `mc_dropout_predict` call.

    All four tensor fields share the input volume's spatial shape and are
    reported PER CHANNEL (ET, TC, WT), never summed or averaged across
    channels -- see `mc_dropout_predict`'s docstring, point 5.

    Attributes:
        mean_prob: The Monte Carlo estimate of the predictive mean,
            `(B, C, D, H, W)` float32, values in `[0, 1]`. These are
            PROBABILITIES, not logits -- see `mc_dropout_predict`'s
            docstring, point 2, before feeding this anywhere that expects
            logits.
        predictive_entropy: TOTAL uncertainty, `(B, C, D, H, W)` float32,
            units nats, range `[0, ln 2]` (`ln 2 ≈ 0.693`, the maximum
            entropy of a Bernoulli variable, reached at `mean_prob = 0.5`).
        expected_entropy: The ALEATORIC component -- uncertainty inherent to
            the input itself, which more MC samples cannot resolve. Same
            shape, units, and range as `predictive_entropy`.
        mutual_information: The EPISTEMIC component -- uncertainty that
            comes from the model's own parameters being uncertain, which
            more training data (or, within one call, more MC samples) could
            in principle resolve. Same shape and units;
            mathematically non-negative (clamped at 0 here; see
            `mc_dropout_predict`'s docstring for the negative-value warning
            it can still log before clamping).
        num_samples: The number of stochastic forward passes averaged into
            the fields above. Echoed back because mutual information is
            biased low at small N (docstring point 3) -- callers comparing
            uncertainty across runs need to know this matched.
    """

    mean_prob: Tensor
    predictive_entropy: Tensor
    expected_entropy: Tensor
    mutual_information: Tensor
    num_samples: int


@contextlib.contextmanager
def dropout_enabled(model: nn.Module) -> Iterator[int]:
    """Puts only `model`'s dropout submodules into train mode.

    Walks `model.modules()`, records each dropout submodule's PRIOR
    `.training` flag, switches every one of them (including any with
    `p == 0`, which are harmless no-ops either way) into train mode, and
    restores each one's EXACT prior flag on exit -- in a `finally`, so a
    restore happens even if the body raises. Restoring the recorded flags
    individually (rather than blanket-calling `.eval()` on exit) matters
    because a caller who had already put the whole model, or one of these
    submodules, into `train()` before calling this must get that back
    exactly, not silently downgraded to `eval()`.

    Deliberately does NOT touch `model.training` itself (i.e. never calls
    `model.train()` or `model.eval()`). `NeuroVisionX.forward`'s return type
    depends on `self.training`: switching it would silently swap a plain
    logits `Tensor` for a `list[Tensor]` or a `MultiTaskOutput`, breaking
    sliding-window inference, which assumes a tensor. See this module's
    top-of-file docstring for the full story.

    Args:
        model: Any `nn.Module`. Works on a bare stub network in tests and on
            the full `NeuroVisionX` alike -- it only looks at module types,
            never at model-specific structure.

    Yields:
        The number of dropout submodules put into train mode (all matching
        submodules, regardless of their `p`).
    """
    targets = [m for m in model.modules() if isinstance(m, _DROPOUT_TYPES)]
    prior_modes = [m.training for m in targets]
    try:
        for module in targets:
            module.train()
        yield len(targets)
    finally:
        for module, was_training in zip(targets, prior_modes, strict=True):
            module.train(was_training)


def count_active_dropout(model: nn.Module) -> int:
    """Counts `model`'s dropout submodules with a nonzero drop probability.

    Args:
        model: Any `nn.Module`.

    Returns:
        The number of submodules matching `_DROPOUT_TYPES` with `p > 0`.
        A dropout submodule constructed at `p=0` (or an exotic dropout
        variant with no `.p` attribute at all -- `getattr(..., "p", 0.0)`
        treats that as inactive rather than raising) does not count: it is
        an identity function regardless of train/eval mode, so it
        contributes no stochasticity for MC-dropout to exploit.
    """
    return sum(
        1
        for module in model.modules()
        if isinstance(module, _DROPOUT_TYPES) and getattr(module, "p", 0.0) > 0.0
    )


def _bernoulli_entropy(p: Tensor, eps: float = 1e-7) -> Tensor:
    """Elementwise Bernoulli entropy, in nats.

    `H(p) = -(p*log(p) + (1-p)*log(1-p))`, clamping `p` to `[eps, 1-eps]`
    first so `log(0)` never happens. Written out explicitly rather than as
    `F.binary_cross_entropy(p, p)` (which computes the same quantity): the
    explicit clamp-then-log form is what it says it is, and this project
    prefers clear code over clever code.

    Args:
        p: Probabilities, any shape, values expected in `[0, 1]`.
        eps: Clamp margin away from the 0/1 boundary.

    Returns:
        Entropy, same shape as `p`, values in `[0, ln 2]`
        (`ln 2 ≈ 0.6931`, the maximum, reached at `p = 0.5`).
    """
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def mc_dropout_predict(
    model: nn.Module,
    image: Tensor,
    cfg: Any,
    device: torch.device,
    num_samples: int | None = None,
    seed: int | None = None,
    use_amp: bool | None = None,
) -> MCDropoutOutput:
    """Runs N stochastic sliding-window passes and decomposes the uncertainty.

    Cost. Exactly N times a single `sliding_window_predict` call -- dropout
    changes nothing about the sliding-window schedule (window count, batch
    grouping), only what each window's forward pass computes. Measured
    window arithmetic on this project's real data: at `roi_size: 96`,
    `overlap: 0.5`, the median case needs 12 sliding windows (max 36 across
    all 1251 preprocessed BraTS 2021 cases), which at `sw_batch_size: 4` is
    3 batched forward passes -- so the project default `num_samples: 10`
    means roughly 120 windows / 30 batched forward passes per case. Measure
    one plain `sliding_window_predict` call on your own hardware and
    multiply by N rather than trusting a wall-clock number quoted from a
    different GPU.

    `mean_prob` is PROBABILITIES, not logits -- a deliberate break from every
    other function in `neurovision.inference`, which all return raw logits.
    The break is necessary, not stylistic: the correct Monte Carlo estimate
    of the predictive mean is the mean of the per-pass PROBABILITIES,
    `mean(sigmoid(logits_i))`; `sigmoid(mean(logits_i))` is a different,
    generally smaller-in-magnitude quantity because `sigmoid` is nonlinear
    (Jensen's inequality again). The hazard this creates:
    `neurovision.inference.postprocess.postprocess_logits` applies its OWN
    sigmoid internally, so handing it `mean_prob` directly double-sigmoids
    every value and silently shrinks it toward 0.5. Use
    `logits_from_mean_prob` to convert `mean_prob` back into something
    `postprocess_logits` can consume correctly.

    Mutual information is biased LOW at small N. `mean_prob` estimated from
    only N samples is a noisier estimate of the TRUE predictive mean than
    the population value would be, and entropy is a concave function, so by
    Jensen's inequality `E[H(mean_prob_N)] <= H(mean_prob_true)` -- the
    predictive (total uncertainty) term is systematically underestimated at
    finite N, while the expected (aleatoric) term, being an average of N
    already-unbiased per-pass entropies, is not. Since
    `mutual_information = predictive_entropy - expected_entropy`, the
    epistemic term comes out too small, and the bias shrinks roughly as
    `1/N`. Practical consequence for this project: NEVER compare mutual
    information values between two runs that used a different `num_samples`
    -- a "lower epistemic uncertainty" reading could just mean a smaller N.

    Why `num_samples` defaults to 10: this is the standard choice in the
    MC-dropout segmentation literature (rather than the N=50+ some papers
    use), and a 10x evaluation-time cost multiplier is affordable within
    this project's rationed ~30 Kaggle GPU-hours/week, where a 50x
    multiplier would not be.

    Uncertainty is reported PER CHANNEL (ET, TC, WT), never summed or
    averaged across them. The three BraTS regions are separately
    interesting for this project -- ET is the smallest and hardest region
    and the one the calibration claim leans on most -- so collapsing them
    into one number here would throw away exactly the breakdown a reader
    would want. A caller that does want one combined map can take
    `mean_prob.mean(dim=1, keepdim=True)` (or the corresponding entropy
    mean) itself.

    Everything runs inside `dropout_enabled(model)`, so `model` itself stays
    in whatever mode it was already in (eval or train) except for its
    dropout submodules, which are switched to train for the duration of
    this call and restored after. `sliding_window_predict` is called with
    `set_eval=False` on every pass -- it would otherwise call `model.eval()`
    on every invocation and silently switch the just-enabled dropout
    submodules back off, since `nn.Module.eval()` recurses into every
    child. See this module's top-of-file docstring for the full story.

    Args:
        model: The segmentation model. Assumed already on `device`; its
            overall train/eval mode is left untouched (only its dropout
            submodules are toggled, and only for the duration of this call).
        image: Input volume, shape `(B, in_channels, D, H, W)`.
        cfg: The full composed Hydra config, exposing
            `cfg.inference.mc_dropout` and `cfg.inference.sliding_window`
            (read by `sliding_window_predict`/`build_inferer`).
        device: Device to run inference on, resolved once via
            `neurovision.utils.device.get_device`.
        num_samples: Number of stochastic forward passes. `None` (the
            default) reads `cfg.inference.mc_dropout.num_samples`.
        seed: `None` (the default) reads `cfg.inference.mc_dropout.seed`,
            which itself defaults to `null` (leave the global torch RNG
            alone). An explicit int here overrides the config either way.
            See the "Seeding" section below for what setting one does.
        use_amp: Whether to run each pass's forward under autocast. `None`
            (the default) decides from `device` via `amp_enabled`.

    Seeding. Dropout masks are drawn from the global torch RNG --
    `nn.Dropout3d.forward` takes no `Generator` argument, so seeding that
    global RNG is the only lever available. When `seed is not None`, this
    function saves `torch.get_rng_state()` (and, if CUDA is available,
    `torch.cuda.get_rng_state_all()`), calls `torch.manual_seed(seed)`,
    runs the N passes, and restores both saved states in a `finally` before
    returning. The save/restore matters because `scripts/evaluate.py` may
    call this once per case inside a loop over an entire split: without
    restoring, the RNG stream after case 5 would depend on exactly how many
    random draws case 1-4 each happened to make, so the same case could get
    a different uncertainty map depending on where it sits in the split.
    Deliberately NOT `neurovision.utils.seed.set_seed` -- that also flips
    MONAI's global determinism flags and `cudnn.benchmark`, which is far
    heavier machinery than a single dropout-mask seed needs for a
    potentially per-case call.

    Returns:
        An `MCDropoutOutput`. See that dataclass's docstring for each
        field's shape, units, and range.

    Raises:
        ValueError: If `num_samples < 1`. Also if
            `cfg.inference.mc_dropout.require_dropout` is True (the
            default) and `count_active_dropout(model) == 0` -- with no
            active dropout module, every one of the N passes computes the
            IDENTICAL forward pass, so `mutual_information` would be
            exactly 0 everywhere and `predictive_entropy` would collapse
            onto `expected_entropy` (the aleatoric term alone). Nothing
            about the returned tensors would look wrong -- they would just
            be a plausible-looking, entirely meaningless uncertainty map.
            The message names the config keys that control dropout
            (`model.encoder.cnn.dropout`, `model.head.dropout`,
            `model.decoder.dropout`) -- the production config sets
            `cnn.dropout: 0.1`, so this error firing on a production model
            means dropout was turned off somewhere upstream.
    """
    mc_cfg = cfg.inference.mc_dropout
    if num_samples is None:
        num_samples = mc_cfg.num_samples
    if seed is None:
        seed = mc_cfg.seed

    if num_samples < 1:
        raise ValueError(f"mc_dropout_predict: num_samples must be >= 1, got {num_samples}.")

    n_active = count_active_dropout(model)
    if mc_cfg.require_dropout and n_active == 0:
        raise ValueError(
            "mc_dropout_predict was called on a model with no active dropout module "
            "(a Dropout3d/Dropout/etc. submodule with p > 0). With every module's dropout "
            "inert, all num_samples forward passes below would be numerically identical, "
            "so mutual_information would be exactly 0.0 everywhere and predictive_entropy "
            "would collapse onto expected_entropy (the aleatoric term alone) -- this "
            "function would return a plausible-looking, entirely meaningless uncertainty "
            "map with no error anywhere else in the pipeline. Check "
            "model.encoder.cnn.dropout, model.head.dropout, and model.decoder.dropout are "
            "non-zero (the production config sets cnn.dropout: 0.1, so this error firing "
            "there means dropout was disabled somewhere upstream). Set "
            "cfg.inference.mc_dropout.require_dropout=False only if a degenerate, "
            "all-zero uncertainty map is genuinely what you want."
        )

    amp_on = amp_enabled(device) if use_amp is None else use_amp
    inferer = build_inferer(cfg)  # built once, reused across all N passes below

    logger.info(
        "mc_dropout_predict: num_samples=%d active_dropout_modules=%d input_shape=%s amp=%s",
        num_samples,
        n_active,
        tuple(image.shape[2:]),
        amp_on,
    )

    saved_cpu_rng: Tensor | None = None
    saved_cuda_rng: list[Tensor] | None = None
    if seed is not None:
        saved_cpu_rng = torch.get_rng_state()
        if torch.cuda.is_available():
            saved_cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(seed)

    # Two running float32 accumulators, shaped like ONE pass's output --
    # never a stack of all N passes. Measured on this project's actual 1251
    # preprocessed cases: the median cropped volume is (137, 171, 140), so a
    # 3-channel fp32 full-volume buffer is ~39 MB; two accumulators cost
    # ~78 MB total REGARDLESS of N, against stacking every pass (~390 MB at
    # N=10, growing linearly with N for no benefit -- nothing downstream
    # needs the individual passes once each has been folded into the sums).
    sum_prob: Tensor | None = None
    sum_entropy: Tensor | None = None

    try:
        with dropout_enabled(model):
            for _ in range(num_samples):
                logits = sliding_window_predict(
                    model,
                    image,
                    cfg,
                    device,
                    use_amp=amp_on,
                    inferer=inferer,
                    # See this module's top-of-file docstring: sliding_window_predict
                    # would otherwise call model.eval() on every pass and silently
                    # switch the dropout submodules dropout_enabled just turned on
                    # back off, since nn.Module.eval() recurses into every child.
                    set_eval=False,
                )
                prob = torch.sigmoid(logits)  # per-channel Bernoulli, never softmax
                entropy = _bernoulli_entropy(prob)

                if sum_prob is None:
                    # Allocated lazily from the first pass's own output, so shape
                    # AND device are inherited rather than guessed --
                    # cfg.inference.sliding_window.output_device: "cpu" legitimately
                    # returns a CPU tensor from sliding_window_predict, and this
                    # function must not silently move it back onto `device`.
                    sum_prob = torch.zeros_like(prob)
                    sum_entropy = torch.zeros_like(entropy)
                sum_prob += prob
                sum_entropy += entropy
    finally:
        if seed is not None:
            torch.set_rng_state(saved_cpu_rng)
            if saved_cuda_rng is not None:
                torch.cuda.set_rng_state_all(saved_cuda_rng)

    assert sum_prob is not None and sum_entropy is not None  # num_samples >= 1 was checked above

    mean_prob = sum_prob / num_samples
    expected_entropy = sum_entropy / num_samples
    predictive_entropy = _bernoulli_entropy(mean_prob)
    mutual_information = predictive_entropy - expected_entropy

    # Mathematically non-negative (entropy is concave, so Jensen's inequality
    # guarantees predictive_entropy >= expected_entropy). A value meaningfully
    # below 0 is not float rounding -- rounding error lives many orders of
    # magnitude below 1e-4 for quantities in [0, ln 2] -- it means an
    # accumulator bug in this function.
    mi_min = mutual_information.min().item()
    if mi_min < -1e-4:
        logger.warning(
            "mc_dropout_predict: mutual_information.min()=%.6f is meaningfully negative "
            "before clamping. Mutual information is mathematically non-negative (entropy "
            "is concave); a value this far below 0 indicates an accumulator bug in this "
            "function, not floating-point rounding error.",
            mi_min,
        )
    mutual_information = mutual_information.clamp(min=0.0)

    # The structural `require_dropout` check above proves the model CONTAINS
    # active dropout modules. It does not prove the N passes actually differed,
    # and those are not the same thing. Measured instance, on this project's own
    # production config: `model.encoder.cnn.zero_init_residual: true` zeroes
    # `norm2.weight` in every `ResidualBlock`, so at INITIALIZATION the residual
    # branch outputs exactly 0 and the block is an exact identity. `Dropout3d`
    # sits before `conv2 -> norm2` in that block, so its perturbation is
    # multiplied by that zero and annihilated -- 10 active dropout modules, and
    # every pass still numerically identical. (`norm2.weight` receives gradient
    # and moves off zero after one optimizer step, so a TRAINED checkpoint is
    # unaffected; see CLAUDE.md's zero_init_residual note.) Other routes to the
    # same place: a checkpoint saved before dropout was configured, or dropout
    # modules sitting on a branch a `layer_scale`-style parameter has driven to
    # zero. All of them yield a plausible-looking, entirely meaningless
    # uncertainty map, so warn rather than return one silently.
    #
    # A warning and not an exception: at N == 1 mutual information is 0 by
    # construction (nothing to warn about), and a genuinely, uniformly confident
    # model on an easy volume can legitimately produce a near-zero epistemic map.
    # Raising would block a real result to catch a misconfiguration.
    if num_samples > 1 and mutual_information.max().item() <= 0.0:
        logger.warning(
            "mc_dropout_predict: mutual_information is EXACTLY 0 at every voxel across "
            "%d passes, despite %d active dropout module(s). The passes were therefore "
            "deterministic and this uncertainty map is meaningless -- the epistemic term "
            "is empty and predictive_entropy has collapsed onto the aleatoric term. Most "
            "likely cause on an UNTRAINED or barely-trained model: "
            "model.encoder.cnn.zero_init_residual=true zeroes norm2.weight in every "
            "ResidualBlock, which annihilates the dropout perturbation that block applies "
            "before conv2. Verify you loaded a trained checkpoint.",
            num_samples,
            n_active,
        )

    return MCDropoutOutput(
        mean_prob=mean_prob,
        predictive_entropy=predictive_entropy,
        expected_entropy=expected_entropy,
        mutual_information=mutual_information,
        num_samples=num_samples,
    )


def logits_from_mean_prob(mean_prob: Tensor, eps: float = 1e-6) -> Tensor:
    """Converts MC-dropout averaged probabilities back into logits.

    The bridge from `MCDropoutOutput.mean_prob` (probabilities) into
    `neurovision.inference.postprocess.postprocess_logits` (which expects
    raw logits and applies its own sigmoid). Implemented as
    `torch.logit(mean_prob.clamp(eps, 1 - eps))`.

    The clamp is mandatory, not defensive: `logit(0) == -inf` and
    `logit(1) == +inf`, and an infinite value reaching
    `postprocess_logits`'s component filters (`remove_small_objects`,
    `KeepLargestConnectedComponent`) produces NaN there, silently poisoning
    the whole case.

    Why this is safe to route through `postprocess_logits` at all: `logit`
    is a strictly monotone increasing function, so thresholding
    `mean_prob >= 0.5` and thresholding
    `sigmoid(logits_from_mean_prob(mean_prob)) >= 0.5` produce EXACTLY the
    same binary mask (`logit(0.5) == 0`, and monotonicity preserves the
    ordering on either side of it). Converting through this function before
    calling `postprocess_logits` therefore does not change which voxels get
    discretized as foreground -- it only makes the existing component
    filtering, small-object removal, and nesting-repair logic reusable on
    MC-dropout's output instead of needing a second, probability-space copy
    of that whole pipeline.

    Args:
        mean_prob: Probabilities, any shape, values expected in `[0, 1]`
            (typically `MCDropoutOutput.mean_prob`).
        eps: Clamp margin away from the 0/1 boundary.

    Returns:
        Logits, same shape as `mean_prob`, always finite.
    """
    return torch.logit(mean_prob.clamp(eps, 1.0 - eps))
