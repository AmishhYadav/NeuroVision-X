"""The full assembled NeuroVision-X network: dual encoder, fusion, decoder, heads.

This file wires together every piece built elsewhere in `neurovision.models` into one
`nn.Module`:

- `neurovision.models.encoders.cnn.CNNEncoder` — 5 levels, strides 1/2/4/8/16.
- `neurovision.models.encoders.swin.SwinEncoder` — 4 levels, strides 2/4/8/16. It cannot
  produce a stride-1 feature (its patch embedding merges 2x2x2 voxels before any attention
  runs), so it is always one level short of the CNN branch. See that module's docstring for
  the full reasoning.
- `neurovision.models.fusion.registry.build_fusion` — one fusion block per level where both
  branches exist (4 of them in production), selected by `model.fusion.name` so the ablation
  (adaptive gated cross-attention vs. concat vs. add) is a config string, not a code change.
- `neurovision.models.decoder.unet_decoder.UNetDecoder` — walks the fused pyramid back up to
  full resolution.
- `neurovision.models.heads.multitask.MultiTaskHead` — one `SegmentationHead` per
  deep-supervision level, plus optional confidence / boundary `AuxiliaryHead`s attached only
  to the full-resolution decoder feature. See that module's docstring.

## The skip pyramid this model builds

CNN level 0 (stride 1, full resolution) has no Swin counterpart, so it passes to the decoder
UNFUSED. CNN level `i + 1` fuses with Swin level `i`, for `i` in `0 .. swin.num_levels - 1`.
So the decoder's skip pyramid, fine to coarse, is::

    [cnn_pyramid[0], fused[0], fused[1], ..., fused[swin.num_levels - 1]]

which has `cnn_encoder.out_channels` as its per-level channel widths — every fusion variant
outputs the CNN branch's width at that level (see `neurovision.models.fusion.adaptive_fusion`
module docstring), so the decoder always sees CNN-shaped skips regardless of which fusion
variant is selected.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import OmegaConf
from torch import Tensor, nn

from neurovision.models.decoder.unet_decoder import UNetDecoder
from neurovision.models.encoders.cnn import build_cnn_encoder
from neurovision.models.encoders.swin import build_swin_encoder
from neurovision.models.fusion.registry import build_fusion
from neurovision.models.heads.multitask import MultiTaskHead, MultiTaskOutput
from neurovision.models.registry import register_model

logger = logging.getLogger(__name__)


def _to_tuple(value: Any) -> tuple:
    """Converts a sequence-valued config field to a plain tuple.

    Hydra hands sequence fields over as OmegaConf `ListConfig` objects, which are not
    `list`/`tuple` instances. Local copy of the same helper used in
    `neurovision.models.encoders.cnn` / `.swin` / `baseline` — kept separate rather than
    shared so each file stays self-contained and independently reviewable.

    Args:
        value: A `ListConfig`, list, tuple, or other sequence.

    Returns:
        A plain `tuple` with the same elements.
    """
    return tuple(value)


class NeuroVisionX(nn.Module):
    """The full dual-encoder / fusion / decoder / multi-head segmentation network.

    Submodules are constructed elsewhere and injected here (see `build_neurovision`), so
    tests can assemble a tiny network directly without a full Hydra config.

    Args:
        cnn_encoder: The CNN branch. `num_levels` must be exactly
            `swin_encoder.num_levels + 1` (see `__init__`'s validation).
        swin_encoder: The Swin branch, or `None` to disable it entirely — the cnn-only
            ablation (`configs/experiment/ablation_cnn_only.yaml`). When `None`, the Swin
            encoder is neither constructed nor run, and every CNN level passes to the
            decoder unfused. This is a real removal of the branch (and its ~2M parameters
            and ~1/3 of step time), not `AdaptiveGatedFusion`'s `layer_scale` turned down —
            `layer_scale` is a learnable parameter that would train back away from zero.
        fusion_blocks: One fusion block per Swin level, `len(fusion_blocks) ==
            swin_encoder.num_levels`. Block `i` fuses `cnn_encoder`'s level `i + 1` with
            `swin_encoder`'s level `i`. Must be an empty `nn.ModuleList()` when
            `swin_encoder is None` — with no Swin branch there is nothing to fuse.
        decoder: The U-Net decoder. `decoder.skip_channels` must equal
            `cnn_encoder.out_channels` exactly.
        out_channels: Number of segmentation output channels. Always 3 for this project (the
            overlapping ET/TC/WT regions) — see `neurovision.models.heads.segmentation` for
            why 4 (the raw class count, also the modality count) is a trap here.
        deep_supervision_levels: Number of decoder stages to attach a `SegmentationHead` to,
            counted fine to coarse starting at the full-resolution stage. Must be in
            `1 .. decoder.num_stages`. `1` disables deep supervision (a single head, the
            full-resolution one).
        head_dropout: `Dropout3d` probability inside every `SegmentationHead`. Unread when
            `heads` is supplied directly.
        heads: A pre-built `MultiTaskHead`, or `None` (the default) to build a
            segmentation-only one from `out_channels` / `deep_supervision_levels` /
            `head_dropout`. Keyword-only and optional so every existing direct construction
            of `NeuroVisionX` (throughout `tests/`) keeps working unchanged. When supplied,
            `len(heads.seg_heads)` must equal `deep_supervision_levels`.

    Raises:
        ValueError: If any of the construction-time consistency checks below fails. Each
            message says how to fix it.
    """

    def __init__(
        self,
        cnn_encoder: nn.Module,
        swin_encoder: nn.Module | None,
        fusion_blocks: nn.ModuleList,
        decoder: nn.Module,
        out_channels: int,
        deep_supervision_levels: int = 3,
        head_dropout: float = 0.0,
        *,
        heads: MultiTaskHead | None = None,
    ) -> None:
        super().__init__()

        self.use_swin = swin_encoder is not None

        if self.use_swin:
            # --- 1. Stride-offset contract between the two encoders. ---
            if cnn_encoder.num_levels != swin_encoder.num_levels + 1:
                raise ValueError(
                    f"cnn_encoder.num_levels ({cnn_encoder.num_levels}) must equal "
                    f"swin_encoder.num_levels + 1 ({swin_encoder.num_levels} + 1 = "
                    f"{swin_encoder.num_levels + 1}). The Swin branch starts at stride 2 (its "
                    f"patch embedding merges 2x2x2 voxels before any attention runs and cannot "
                    f"produce a stride-1 feature), while the CNN branch starts at stride 1 (a "
                    f"full-resolution stem). So the CNN encoder needs exactly one more level "
                    f"than the Swin encoder — its level 0 passes to the decoder unfused, and "
                    f"its levels 1.. line up with the Swin encoder's levels 0.. . Fix by "
                    f"adding one more entry to cnn.channels (and cnn.blocks_per_stage), or "
                    f"removing one from swin.num_levels."
                )

            # --- 2. The two pyramids are spatially aligned, level for level. ---
            if list(cnn_encoder.strides[1:]) != list(swin_encoder.strides):
                raise ValueError(
                    f"cnn_encoder.strides[1:] ({list(cnn_encoder.strides[1:])}) does not match "
                    f"swin_encoder.strides ({list(swin_encoder.strides)}); the two encoder "
                    f"branches are not aligned, so fusion would merge feature maps from "
                    f"different spatial resolutions. Check that both encoders halve spatial "
                    f"dimensions the same number of times at the same levels."
                )

            # --- 3. One fusion block per level where both branches exist. ---
            if len(fusion_blocks) != swin_encoder.num_levels:
                raise ValueError(
                    f"len(fusion_blocks) ({len(fusion_blocks)}) must equal "
                    f"swin_encoder.num_levels ({swin_encoder.num_levels}) — one fusion block "
                    f"per level where both branches produce a feature map. Build fusion_blocks "
                    f"with one entry per Swin level, in the same order."
                )
        else:
            # --- No Swin branch: fusion_blocks must be empty, or it is a wiring bug. ---
            if len(fusion_blocks) != 0:
                raise ValueError(
                    f"swin_encoder is None but fusion_blocks has {len(fusion_blocks)} "
                    f"entries. With no Swin branch there is nothing to fuse — every CNN level "
                    f"passes to the decoder unfused. Pass an empty nn.ModuleList() for "
                    f"fusion_blocks, or pass a real swin_encoder if fusion was intended."
                )

        # --- 4. The decoder was built for this exact skip pyramid. ---
        if list(decoder.skip_channels) != list(cnn_encoder.out_channels):
            raise ValueError(
                f"decoder.skip_channels ({list(decoder.skip_channels)}) does not match "
                f"cnn_encoder.out_channels ({list(cnn_encoder.out_channels)}). Every fusion "
                f"variant outputs the CNN branch's channel width at each level, so the "
                f"decoder's skip pyramid is always CNN-widths ([cnn_encoder.out_channels[0], "
                f"fused[0], ..., fused[-1]]) regardless of which fusion variant is selected. "
                f"This decoder was built for a different pyramid — rebuild it with "
                f"skip_channels=cnn_encoder.out_channels."
            )

        # --- 5. deep_supervision_levels is a valid number of decoder stages. ---
        if not (1 <= deep_supervision_levels <= decoder.num_stages):
            raise ValueError(
                f"deep_supervision_levels ({deep_supervision_levels}) must be between 1 and "
                f"decoder.num_stages ({decoder.num_stages}) inclusive — one head per "
                f"supervised decoder stage, starting from the full-resolution stage (index "
                f"0). Set it to 1 to disable deep supervision (a single, full-resolution "
                f"head)."
            )

        # --- 6. A supplied MultiTaskHead was built for this exact deep_supervision_levels. ---
        if heads is not None and len(heads.seg_heads) != deep_supervision_levels:
            raise ValueError(
                f"heads.seg_heads has {len(heads.seg_heads)} entries but "
                f"deep_supervision_levels is {deep_supervision_levels} -- they must agree. "
                f"Build heads with the same deep_supervision_levels passed here, or omit "
                f"heads and let NeuroVisionX build a segmentation-only MultiTaskHead itself."
            )

        self.cnn_encoder = cnn_encoder
        self.swin_encoder = swin_encoder
        self.fusion_blocks = fusion_blocks
        self.decoder = decoder
        self.deep_supervision_levels = deep_supervision_levels

        # self.heads is a MultiTaskHead (owns the segmentation heads AND the optional
        # confidence / boundary heads), not a bare nn.ModuleList of segmentation heads --
        # see neurovision.models.heads.multitask's module docstring. Head i reads
        # decoder.out_channels[i] -- decoder features are fine-to-coarse, so head 0 is the
        # full-resolution head and heads 1.. read progressively coarser decoder features.
        self.heads = (
            heads
            if heads is not None
            else MultiTaskHead(
                decoder_channels=decoder.out_channels,
                out_channels=out_channels,
                deep_supervision_levels=deep_supervision_levels,
                seg_dropout=head_dropout,
            )
        )

    def _encode_decode(self, x: Tensor) -> list[Tensor]:
        """Shared encoder -> fusion -> decoder body used by every forward path.

        Factored out so `forward`, `forward_multitask`, and `forward_with_gates` do not each
        duplicate the fusion loop; `forward_with_gates` additionally needs the per-block gate
        maps, which this method does not return, so it has its own near-identical body
        instead of reusing this one (see that method's docstring).

        Args:
            x: Input MRI volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            Fine-to-coarse decoder features (no heads applied), length
            `decoder.num_stages`.
        """
        # Fine-to-coarse pyramid from the CNN branch (always present).
        cnn_pyramid = self.cnn_encoder(x)

        if self.use_swin:
            swin_pyramid = self.swin_encoder(x)
            # CNN level 0 (stride 1) has no Swin counterpart and passes through unfused;
            # levels 1.. are fused one at a time with the aligned Swin level.
            skips = [cnn_pyramid[0]] + [
                block(cnn_pyramid[i + 1], swin_pyramid[i])
                for i, block in enumerate(self.fusion_blocks)
            ]
        else:
            # cnn-only ablation: the Swin encoder is never called, and every CNN level
            # passes to the decoder unfused.
            skips = cnn_pyramid

        return self.decoder(skips)  # fine-to-coarse decoder features, no heads applied yet

    def forward(self, x: Tensor) -> Tensor | list[Tensor] | MultiTaskOutput:
        """Runs the full network.

        Args:
            x: Input MRI volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            One of THREE possible types, chosen by `self.training` and `self.heads`:

            1. A single logits `Tensor`, shape `(B, out_channels, D, H, W)`, matching the
               input's spatial shape. Returned in eval mode always, and in training mode
               when there is exactly one segmentation head and no auxiliary head enabled.
            2. A `list[Tensor]`, ordered **highest resolution first** (index 0 is full
               resolution, index 1 is stride 2, and so on) — element `i` has spatial shape
               `(D // 2**i, H // 2**i, W // 2**i)` and `out_channels` channels. Returned only
               in training mode, with `deep_supervision_levels > 1` and no auxiliary head
               enabled.
            3. A `MultiTaskOutput` (see `neurovision.models.heads.multitask`). Returned only
               in training mode, whenever either the confidence or the boundary head is
               enabled (`self.heads.has_auxiliary`) — regardless of
               `deep_supervision_levels`, since `MultiTaskOutput.seg` already carries
               whatever the segmentation case above would have returned.

            THE RETURN-TYPE SWITCH IS DELIBERATE AND LOAD-BEARING, for three independent
            reasons:

            1. `neurovision.inference.sliding_window` and MONAI's sliding-window inferer call
               `model(patch)` and expect a plain `Tensor`. Returning anything else in eval
               mode would silently break every evaluation path — there is no shape error,
               just a wrong Python type flowing into code that assumes a tensor.
            2. `neurovision.losses.segmentation.DeepSupervisionLoss.forward` takes a sequence
               ordered highest-resolution-first and internally upsamples each entry to the
               target's resolution to match. `self.heads.seg_heads[0]` reads `feats[0]` (the
               decoder's full-resolution output, since `UNetDecoder.forward` returns
               fine-to-coarse features), so `logits[0]` is already full resolution and
               `logits[1:]` are already the coarser ones — the natural construction order
               here IS the order the loss wants. Do not reverse it.
            3. When either auxiliary head is enabled, training returns a `MultiTaskOutput`,
               which only `neurovision.losses.multitask.MultiTaskLoss` knows how to consume.
               No other loss should be pointed at a model built this way — see
               `build_neurovision`'s auxiliary-head/loss-name guard.

            MC-DROPOUT HAZARD: uncertainty estimation re-runs this network with dropout
            active to measure predictive spread. Do NOT do that by calling `model.train()` —
            that flips this method's return type (to a list, or to a `MultiTaskOutput`) and
            breaks sliding-window inference, which expects a tensor. Instead, enable the
            `Dropout3d` modules individually (e.g. `module.train()` on each `Dropout3d` found
            via `self.modules()`) while leaving the rest of the model, and this method, in
            `eval()` mode. The auxiliary heads contain `Dropout3d` too (see
            `AuxiliaryHead`), so this applies to them exactly as it does to the segmentation
            heads.
        """
        feats = self._encode_decode(x)
        out = self.heads(feats)

        # See the docstring above: MultiTaskOutput only when an aux head is enabled and only
        # in training; list only in training with more than one segmentation head and no aux
        # head; a single tensor otherwise. This is the one place the switch happens.
        if self.training and self.heads.has_auxiliary:
            return out
        if self.training and len(out.seg) > 1:
            return out.seg
        return out.seg[0]

    def forward_multitask(self, x: Tensor) -> MultiTaskOutput:
        """Runs the network and always returns a `MultiTaskOutput`, regardless of mode.

        Unlike `forward`, this method never switches return type — it is the hook for
        uncertainty and calibration code (not yet written) that always wants segmentation
        logits alongside whatever auxiliary heads are enabled, whether the model is in train
        or eval mode. Nothing consumes this yet.

        Args:
            x: Input MRI volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            A `MultiTaskOutput` with `seg` of length `deep_supervision_levels` and
            `confidence` / `boundary` populated according to which auxiliary heads are
            enabled (`None` for a disabled one).
        """
        feats = self._encode_decode(x)
        return self.heads(feats)

    def forward_with_gates(self, x: Tensor) -> tuple[Tensor, list[Tensor | None]]:
        """Runs the network and also returns each fusion block's gate map.

        This is the explainability / visualization path (e.g. "does the model lean on
        transformer context near the tumor margin"), not a training path — it always returns
        a single full-resolution logits tensor, regardless of `self.training`.

        Args:
            x: Input MRI volume, shape `(B, in_channels, D, H, W)`.

        Returns:
            A tuple `(logits, gates)`:

            - `logits`: full-resolution logits, shape `(B, out_channels, D, H, W)`.
            - `gates`: one entry per fusion block, fine to coarse. `AdaptiveGatedFusion`
              blocks contribute a real spatially-varying gate map, shape `(B, 1, D_i, H_i,
              W_i)` (scalar gate) or `(B, C_i, D_i, H_i, W_i)` (per-channel gate).
              `ConcatFusion` and `AddFusion` blocks have no such concept and contribute
              `None` — so this list may contain a mix of tensors and `None`, and callers must
              check each entry before using it. When the Swin branch is disabled
              (`swin_encoder is None`), there are no fusion blocks at all, so `gates` is the
              EMPTY list `[]` — different from a list of `None`s, since there is nothing to
              report rather than several blocks each reporting nothing.
        """
        cnn_pyramid = self.cnn_encoder(x)

        if self.use_swin:
            swin_pyramid = self.swin_encoder(x)
            skips = [cnn_pyramid[0]]
            gates: list[Tensor | None] = []
            for i, block in enumerate(self.fusion_blocks):
                fused, gate = block(cnn_pyramid[i + 1], swin_pyramid[i], return_gate=True)
                skips.append(fused)
                gates.append(gate)
        else:
            skips = cnn_pyramid
            gates = []

        feats = self.decoder(skips)
        logits = self.heads.seg_heads[0](feats[0])
        return logits, gates


@register_model("neurovision")
def build_neurovision(cfg: Any) -> nn.Module:
    """Builds the full `NeuroVisionX` network from config.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.data`, `cfg.model` (with
            sub-groups `encoder.cnn`, `encoder.swin`, `fusion`, `decoder`, `head`, and scalar
            keys `out_channels`, `deep_supervision_levels` — see
            `configs/model/neurovision.yaml`), and optionally `cfg.training.loss` (see the
            deep-supervision guard below).

    Returns:
        A constructed `NeuroVisionX`.

    Raises:
        ValueError: If `cfg.model.deep_supervision_levels > 1` disagrees with
            `cfg.training.loss.deep_supervision.enabled` (when the latter is present at all —
            see below), naming both config keys. Also propagates any `ValueError` raised by
            `NeuroVisionX.__init__`'s own consistency checks.
    """
    cnn_encoder = build_cnn_encoder(cfg)

    # The cnn-only ablation switch (configs/experiment/ablation_cnn_only.yaml). Defaults to
    # True via .get() rather than hasattr, so older/model-only test configs that predate this
    # key still build the full dual-encoder model unchanged.
    swin_enabled = cfg.model.encoder.swin.get("enabled", True)

    if swin_enabled:
        swin_encoder = build_swin_encoder(cfg)
        fusion_blocks = nn.ModuleList(
            [
                build_fusion(cfg, cnn_encoder.out_channels[i + 1], swin_encoder.out_channels[i], i)
                for i in range(swin_encoder.num_levels)
            ]
        )
    else:
        # Do NOT call build_swin_encoder or build_fusion here: constructing the Swin branch
        # and then ignoring it would waste ~2.04M parameters and ~1/3 of step time, which
        # defeats the entire point of this ablation.
        swin_encoder = None
        fusion_blocks = nn.ModuleList()
        logger.info(
            "model.encoder.swin.enabled is False: omitting the Swin branch and all fusion "
            "blocks entirely (the cnn-only ablation). Every CNN level passes to the decoder "
            "unfused."
        )

    decoder_cfg = cfg.model.decoder
    # null in YAML -> None via OmegaConf, meaning "mirror the skip widths" (UNetDecoder's own
    # default). A real ListConfig must be converted to a plain tuple first -- Hydra hands
    # sequences over as ListConfig, which UNetDecoder's isinstance-free length checks accept,
    # but downstream code (and consistency with the other builders' _to_tuple convention)
    # expects a plain sequence.
    decoder_channels = decoder_cfg.channels
    if decoder_channels is not None:
        decoder_channels = _to_tuple(decoder_channels)

    decoder = UNetDecoder(
        skip_channels=cnn_encoder.out_channels,
        decoder_channels=decoder_channels,
        blocks_per_stage=decoder_cfg.blocks_per_stage,
        num_groups=decoder_cfg.num_groups,
        dropout=decoder_cfg.dropout,
        upsample=decoder_cfg.upsample,
        use_attention_gates=decoder_cfg.use_attention_gates,
        use_checkpoint=decoder_cfg.use_checkpoint,
    )

    deep_supervision_levels = cfg.model.deep_supervision_levels

    # Guard: deep_supervision_levels > 1 only makes sense if the loss is wrapped in
    # DeepSupervisionLoss (decided by training.loss.deep_supervision.enabled). If those two
    # disagree, the model hands a list to a loss that expects a tensor (or emits one output
    # where the loss expects several) -- a crash at best, a silently mis-weighted objective
    # at worst. Skip the check entirely when the training group is absent (e.g. a
    # model-only test config), rather than requiring every caller to fabricate one.
    loss_ds_enabled = OmegaConf.select(cfg, "training.loss.deep_supervision.enabled", default=None)
    if loss_ds_enabled is not None:
        model_wants_ds = deep_supervision_levels > 1
        if model_wants_ds != loss_ds_enabled:
            raise ValueError(
                f"model.deep_supervision_levels ({deep_supervision_levels}) and "
                f"training.loss.deep_supervision.enabled ({loss_ds_enabled}) disagree. With "
                f"deep_supervision_levels > 1 the model returns a LIST of logits tensors in "
                f"training mode, which only DeepSupervisionLoss knows how to consume; with "
                f"deep_supervision_levels == 1 it returns a single tensor, which "
                f"DeepSupervisionLoss (or any other loss) also accepts. Set "
                f"model.deep_supervision_levels to 1 (or set "
                f"training.loss.deep_supervision.enabled to true), or the reverse, so the "
                f"two agree."
            )

    # head_cfg.get(...) with defaults (rather than head_cfg.confidence, which would raise
    # on a config predating these keys) so existing model-only test configs -- built before
    # the auxiliary heads existed -- still build a segmentation-only model.
    head_cfg = cfg.model.head
    conf_cfg = head_cfg.get("confidence", None)
    bnd_cfg = head_cfg.get("boundary", None)
    confidence_enabled = bool(conf_cfg.get("enabled", False)) if conf_cfg is not None else False
    boundary_enabled = bool(bnd_cfg.get("enabled", False)) if bnd_cfg is not None else False

    # Guard: an auxiliary head is enabled only if training.loss.name == "multitask", and vice
    # versa. Modelled on the deep-supervision guard directly above -- same reasoning, a
    # different silent-mismatch shape. Skipped entirely when there is no training group at
    # all (e.g. a model-only test config), same as the deep-supervision guard.
    aux_enabled = confidence_enabled or boundary_enabled
    loss_name = OmegaConf.select(cfg, "training.loss.name", default=None)
    if loss_name is not None:
        if aux_enabled and loss_name != "multitask":
            raise ValueError(
                f"model.head.confidence.enabled ({confidence_enabled}) or "
                f"model.head.boundary.enabled ({boundary_enabled}) is True, but "
                f"training.loss.name is '{loss_name}', not 'multitask'. With an auxiliary "
                f"head enabled the model returns a MultiTaskOutput in training mode, which "
                f"only MultiTaskLoss knows how to consume. Set training.loss.name to "
                f"'multitask', or disable both model.head.confidence.enabled and "
                f"model.head.boundary.enabled."
            )
        if loss_name == "multitask" and not aux_enabled:
            raise ValueError(
                "training.loss.name is 'multitask' but both model.head.confidence.enabled "
                "and model.head.boundary.enabled are False. MultiTaskLoss would be "
                "configured with auxiliary loss terms that have no head to supervise, "
                "silently reducing to the plain segmentation loss while the config claims "
                "otherwise. Enable at least one of model.head.confidence.enabled / "
                "model.head.boundary.enabled, or set training.loss.name to a "
                "non-multitask loss."
            )

    # Each auxiliary head reads its OWN config sub-block. They currently carry identical
    # values in configs/model/neurovision.yaml, which makes it tempting to collapse them into
    # one shared parameter set -- do not. `model.head.boundary.hidden_channels` would then be
    # read and thrown away, building a head of a different width with no error anywhere.
    def _aux_kwargs(block: Any, prefix: str) -> dict[str, Any]:
        """Pulls one auxiliary head's hyperparameters out of its config sub-block.

        Args:
            block: The `model.head.confidence` / `model.head.boundary` sub-block, or `None`
                when the key is absent (older model-only test configs predate it).
            prefix: `"confidence"` or `"boundary"` -- the MultiTaskHead argument prefix.

        Returns:
            Keyword arguments for `MultiTaskHead`, falling back to that class's own defaults
            when the sub-block is absent.
        """
        if block is None:
            return {}
        return {
            f"{prefix}_hidden_channels": block.get("hidden_channels", None),
            f"{prefix}_num_groups": block.get("num_groups", 8),
            f"{prefix}_dropout": block.get("dropout", 0.0),
        }

    heads = MultiTaskHead(
        decoder_channels=decoder.out_channels,
        out_channels=cfg.model.out_channels,
        deep_supervision_levels=deep_supervision_levels,
        seg_dropout=head_cfg.dropout,
        confidence=confidence_enabled,
        boundary=boundary_enabled,
        **_aux_kwargs(conf_cfg if confidence_enabled else None, "confidence"),
        **_aux_kwargs(bnd_cfg if boundary_enabled else None, "boundary"),
    )

    model = NeuroVisionX(
        cnn_encoder=cnn_encoder,
        swin_encoder=swin_encoder,
        fusion_blocks=fusion_blocks,
        decoder=decoder,
        out_channels=cfg.model.out_channels,
        deep_supervision_levels=deep_supervision_levels,
        head_dropout=head_cfg.dropout,
        heads=heads,
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built NeuroVisionX: %d parameters, fusion='%s', deep_supervision_levels=%d, "
        "confidence_head=%s, boundary_head=%s",
        n_params,
        cfg.model.fusion.name,
        deep_supervision_levels,
        confidence_enabled,
        boundary_enabled,
    )

    return model
