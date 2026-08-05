"""Swin attention visualization for the fusion branch's transformer encoder.

Captures the post-softmax attention weights MONAI's `SwinTransformer` computes internally
(inside `monai.networks.nets.swin_unetr.WindowAttention`) and turns them into voxel-space maps
a figure can show alongside the input volume. This is an INSPECTION tool, not a training-time
component -- nothing here is differentiated or optimized against.

## There are only THREE attention stages in this project's production model, not four

`configs/model/neurovision.yaml` sets `model.encoder.swin.num_levels=4`, which -- per
`neurovision.models.encoders.swin.SwinEncoder` -- drops MONAI's `layers4` entirely (see that
module's docstring for why: `layers4` alone is 75% of the branch's parameters for little
benefit at 96^3). Fusion happens at strides 2/4/8/16, but Swin attention exists only at
strides 2/4/8 (`layers1`, `layers2`, `layers3`) -- the coarsest fused level (stride 16) has NO
attention map at all, because it is CNN-fused directly against the raw stride-16 Swin feature
with no further transformer stage beyond it. A figure built from this module must show three
panels and say so explicitly, not silently show three where a reader expects four.

## The central hazard: shifted vs. unshifted windows

Each Swin stage runs its blocks in pairs: `blocks.0` is unshifted (W-MSA), `blocks.1` is
shifted (SW-MSA) with `shift = window_size // 2` (e.g. shift 3 for the standard window size 7).
The two blocks partition the SAME token grid into DIFFERENT, non-overlapping sets of windows --
`blocks.1`'s windows are `blocks.0`'s windows shifted by `torch.roll`. Getting this wrong (e.g.
forgetting the roll, or applying it in the wrong direction) displaces every attention value by
`shift` tokens uniformly. Every shape check still passes -- the output is still `(nW, N, N)`,
still row-sums to 1, still upsamples and "looks like" a plausible attention map -- and only the
actual spatial content is wrong. Nothing downstream would notice.

## The fix: an index grid pushed through MONAI's own pad/roll/partition functions

This module does NOT hand-derive the inverse mapping from window-and-token index back to a
`(d, h, w)` voxel coordinate. Instead it builds a tensor whose value at each position IS that
position's flat index into the padded token grid (`torch.arange`), and pushes that index grid
through the EXACT SAME `torch.roll(-shift)` and
`monai.networks.nets.swin_unetr.window_partition(..., window_size)` calls MONAI's own
`SwinTransformerBlock.forward_part1` applies to the real feature tensor. The result tells us,
by construction and not by derivation, which padded-grid position each `(window, token)` slot
came from -- exactly the same technique this project's fusion module uses to build its
windowed-attention key-padding mask on a batch-invariant validity grid rather than deriving
indices by hand (`neurovision.models.fusion.adaptive_fusion`, see CLAUDE.md). Padding is
inverted by simply discarding index-grid entries whose decoded `(d, h, w)` falls outside the
real (unpadded) token grid -- MONAI pads only at the END of each axis (see
`SwinTransformerBlock.forward_part1`: `pad_l = pad_t = pad_d0 = 0`), so a decoded coordinate
`>= (d, h, w)` is unambiguously a pad position, never real data.

## Memory: reduce over heads INSIDE the hook, never store the raw tensor

At 96^3 the first Swin stage alone produces attention of shape `(343, 3, 343, 343)` -- 343
windows, 3 heads, 343x343 = 117,649 query/key pairs per head -- which is 343 * 3 * 343 * 343 *
4 bytes = ~483 MB for that ONE stage, and roughly 1 GB summed across all six blocks (three
stages x two blocks) if kept raw. `capture_attention`'s hook therefore averages over the head
axis (`output.mean(dim=1)`) and detaches BEFORE storing -- every value this module ever holds
onto is `(nW, N, N)`, never `(nW, heads, N, N)`.

## Which reduction is meaningful: RECEIVED attention, not a row mean

Attention rows sum to 1 by construction (that is what softmax does), so a per-QUERY-token row
mean is the constant `1 / N` for every token and carries zero information -- plotting it
produces a uniformly grey map that LOOKS like a legitimate (if boring) result rather than an
obvious error. The informative reduction is attention RECEIVED: the column mean,
`attn.mean(dim=-2)`, i.e. how much attention each KEY token is given, averaged over the queries
in its window. Every function in this module that reduces a `(nW, N, N)` (or composed `(T, T)`)
matrix to a per-token map uses this column-mean convention; none of them ever averages over the
query axis to read out a value.

## Resolution is honest, not voxel-level

`attention_to_voxel_map` and `attention_rollout` both optionally upsample their raw
token-resolution map up to the input volume's shape for display. That upsampling is
cosmetic -- the map's TRUE resolution is the token grid it came from, which is 2x / 4x / 8x
coarser than the input at strides 2 / 4 / 8 respectively. An upsampled map must never be
presented as having voxel-level precision. Padded token positions are simply ABSENT from the
unpadded output (not present as zeros to be averaged in) -- see the index-grid mapping above.

## `attention_rollout` is honest about NOT being Abnar & Zuidema's rollout

The published attention-rollout algorithm (Abnar & Zuidema 2020) assumes (a) global attention
over one fixed token set shared across every layer, and (b) a CLS token to read the final
result out of. Swin satisfies NEITHER assumption: attention is strictly local to non-overlapping
7^3 windows, the window PARTITION itself changes between the unshifted and shifted block within
a stage (so a naive matrix multiply of the two blocks' `(N, N)` matrices would not even be in
the same basis), and patch merging halves the token grid between stages, so there is no shared
token set to roll attention across stage boundaries at all. What `attention_rollout` actually
computes is much narrower: rollout WITHIN one stage only, composing that stage's two blocks
after first expanding each block's windowed `(N, N)` matrix into a full token-to-token
`(T, T)` matrix over the WHOLE stage's token grid (via the same index-grid technique), using
the standard rollout step `A_hat = residual_weight * I + (1 - residual_weight) * A`, row
re-normalized, multiplied in forward order. This is a legitimate composition of two blocks that
really do share a token set, but it is not, and must never be described as, "the" attention
rollout for the whole network. `T = d * h * w` grows fast (`48**3 = 110,592` for stage 1 at a
96^3 input) and a dense `(T, T)` matrix is `T**2` entries, so `max_tokens` is a HARD guard, not
a hint -- see that function's docstring.

## `combine_stage_maps` is a presentation heuristic, not attention

There is no attention connecting one stage's tokens to the next stage's tokens -- patch merging
between stages is a strided LINEAR projection, not an attention operation, so there is no
mathematically defined way to propagate an attention value across a stage boundary the way
rollout propagates it within a stage. `combine_stage_maps` averaging (or multiplying) several
already-upsampled per-stage maps together is a PRESENTATION choice for putting one figure on
screen, not a derived quantity, and must be labelled as such in any figure caption that uses it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import WindowAttention, get_window_size, window_partition
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "AttentionCapture",
    "capture_attention",
    "attention_to_voxel_map",
    "attention_rollout",
    "combine_stage_maps",
    "available_blocks",
]


@dataclass
class AttentionCapture:
    """The result of one `capture_attention` call.

    Attributes:
        stage_names: Dotted block names (e.g. `"layers1.0.blocks.0"`), one per
            `WindowAttention`-owning `SwinTransformerBlock`, in forward order. The name is the
            BLOCK's dotted path (the `WindowAttention` submodule's own name minus its trailing
            `.attn`), so it is stable regardless of how deeply the Swin branch is nested inside
            a larger model (e.g. `"swin_encoder.swin.layers1.0.blocks.0"` inside `NeuroVisionX`).
        attention: Dotted block name -> post-softmax attention, shape `(nW, N, N)`, ALREADY
            averaged over the head axis and detached (see this module's top-of-file docstring,
            memory hazard). `nW` is the number of windows the token grid was partitioned into at
            this block; `N` is the (possibly window-shrunk, see `get_window_size`) window volume.
        token_grids: Dotted block name -> the block's own UNPADDED token grid `(d, h, w)`, i.e.
            the spatial shape of the tensor entering that block, before MONAI pads it up to a
            multiple of the window size.
        window_sizes: Dotted block name -> the ACTUAL window size used by this block on this
            input, after `get_window_size` has (if necessary) shrunk it to fit an axis smaller
            than the configured window size.
        shift_sizes: Dotted block name -> the ACTUAL shift used, `(0, 0, 0)` for an unshifted
            block, or for a shifted block whose axis was shrunk to fit (`get_window_size` zeroes
            the shift together with the window in that case).
        input_shape: The `(D, H, W)` spatial shape of the volume that produced this capture --
            what `attention_to_voxel_map` and `attention_rollout` upsample their token-grid maps
            up to.
    """

    stage_names: list[str]
    attention: dict[str, Tensor]
    token_grids: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    window_sizes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    shift_sizes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    input_shape: tuple[int, int, int] = (0, 0, 0)


def _block_name_from_attn_name(attn_name: str) -> str:
    """Strips a `WindowAttention` submodule's dotted name down to its owning block's name.

    MONAI's `SwinTransformerBlock.__init__` always attaches its `WindowAttention` as
    `self.attn`, so the owning block's dotted name is simply everything before the trailing
    `.attn` component.

    Args:
        attn_name: A `WindowAttention` submodule's dotted name from `named_modules()`.

    Returns:
        The owning block's dotted name, or `""` if `attn_name` has no dot (the `WindowAttention`
        module was itself the root -- not a real configuration, but handled rather than crashing).
    """
    return attn_name.rsplit(".", 1)[0] if "." in attn_name else ""


def available_blocks(model: nn.Module) -> list[str]:
    """Lists dotted block names owning a `WindowAttention` submodule, in forward order.

    Mirrors `neurovision.explainability.gradcam.available_layers`'s purpose: lets a caller
    discover valid `block_name` arguments for `attention_to_voxel_map` (and valid stage
    prefixes for `attention_rollout`) without reading the model's source.

    Args:
        model: Any `nn.Module`. Models without a Swin branch (e.g. the `unet3d` baseline)
            simply return an empty list.

    Returns:
        Dotted block names, in `model.named_modules()` order (which for MONAI's
        `SwinTransformer` matches actual forward execution order: `layers1` before `layers2`
        before `layers3` before `layers4`, and within a stage, `blocks.0` before `blocks.1`).
    """
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, WindowAttention):
            names.append(_block_name_from_attn_name(name))
    return names


def _padded_index_grid(
    token_grid: tuple[int, int, int], window_size: tuple[int, int, int]
) -> tuple[Tensor, tuple[int, int, int]]:
    """Builds an index grid whose value at each position is its own flat index.

    The grid is padded up to a multiple of `window_size` on the END of each axis only --
    matching `SwinTransformerBlock.forward_part1` exactly (`pad_l = pad_t = pad_d0 = 0`, only
    `pad_d1` / `pad_b` / `pad_r` are non-zero). Because the low side is never padded, a padded
    position `i < token_grid[axis]` on any axis is, by construction, the SAME position as the
    corresponding unpadded token -- which is what lets `_window_index_mapping`'s callers invert
    padding by simple coordinate comparison rather than tracking a separate padding mask.

    Args:
        token_grid: The block's unpadded `(d, h, w)` token grid.
        window_size: The ACTUAL window size for this block on this grid (i.e. already passed
            through `get_window_size`).

    Returns:
        `(idx, padded_shape)`: `idx` has shape `(1, dp, hp, wp, 1)`, dtype `torch.long`, values
        `0 .. dp*hp*wp - 1`; `padded_shape` is `(dp, hp, wp)`.
    """
    d, h, w = token_grid
    wd, wh, ww = window_size
    pad_d = (wd - d % wd) % wd
    pad_h = (wh - h % wh) % wh
    pad_w = (ww - w % ww) % ww
    dp, hp, wp = d + pad_d, h + pad_h, w + pad_w
    idx = torch.arange(dp * hp * wp, dtype=torch.long).view(1, dp, hp, wp, 1)
    return idx, (dp, hp, wp)


def _window_index_mapping(
    token_grid: tuple[int, int, int],
    window_size: tuple[int, int, int],
    shift_size: tuple[int, int, int],
) -> tuple[Tensor, tuple[int, int, int]]:
    """Maps every `(window, token)` slot to its padded-grid flat index, by construction.

    Pushes the index grid from `_padded_index_grid` through the EXACT SAME
    `torch.roll(-shift)` + `monai.networks.nets.swin_unetr.window_partition` calls MONAI applies
    to the real feature tensor inside `SwinTransformerBlock.forward_part1` -- see this module's
    top-of-file docstring for why this construction, rather than hand-derived index arithmetic,
    is what makes the mapping immune to the shifted/unshifted ordering mistake described there.
    `torch.roll` with an all-zero shift is a no-op, so this is safe to call unconditionally even
    for an unshifted block (`shift_size == (0, 0, 0)`).

    Args:
        token_grid: The block's unpadded `(d, h, w)` token grid.
        window_size: The ACTUAL window size for this block (post `get_window_size`).
        shift_size: The ACTUAL shift for this block (post `get_window_size`); `(0, 0, 0)` for an
            unshifted block.

    Returns:
        `(idx_windows, padded_shape)`: `idx_windows` has shape `(nW, N)` (`N` = the window
        volume), dtype `torch.long`, each entry a flat index into the padded, PRE-roll grid of
        shape `padded_shape`. Decoding an entry with `padded_shape` and comparing against
        `token_grid` tells you whether that `(window, token)` slot corresponds to real data or a
        padded position.
    """
    idx, padded_shape = _padded_index_grid(token_grid, window_size)
    idx = torch.roll(idx, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
    idx_windows = window_partition(idx, window_size)  # (nW, N, 1)
    return idx_windows.squeeze(-1), padded_shape


def _decode_padded_indices(
    flat_idx: Tensor, padded_shape: tuple[int, int, int]
) -> tuple[Tensor, Tensor, Tensor]:
    """Unravels flat padded-grid indices back to `(d, h, w)` coordinates.

    Args:
        flat_idx: Any shape, dtype `torch.long`, values in `[0, dp*hp*wp)`.
        padded_shape: `(dp, hp, wp)`, the padded grid `flat_idx` indexes into.

    Returns:
        `(d_pad, h_pad, w_pad)`, each the same shape as `flat_idx`.
    """
    dp, hp, wp = padded_shape
    d_pad = flat_idx // (hp * wp)
    rem = flat_idx % (hp * wp)
    h_pad = rem // wp
    w_pad = rem % wp
    return d_pad, h_pad, w_pad


def _scatter_received_to_grid(
    received: Tensor,
    idx_windows: Tensor,
    padded_shape: tuple[int, int, int],
    token_grid: tuple[int, int, int],
) -> Tensor:
    """Scatters a per-`(window, token)` value onto the unpadded `(d, h, w)` token grid.

    Every valid (non-padded) `(window, token)` slot maps to a UNIQUE unpadded grid position --
    `_window_index_mapping` is a bijection built from a roll (a permutation) and a partition (a
    reshape), so this is a plain assignment, never an average or an accumulation.

    Args:
        received: Shape `(nW, N)`, one value per `(window, token)` slot (e.g. the received-
            attention reduction, `attn.mean(dim=-2)`).
        idx_windows: Shape `(nW, N)`, from `_window_index_mapping`.
        padded_shape: `(dp, hp, wp)`, from `_window_index_mapping`.
        token_grid: The block's unpadded `(d, h, w)` token grid.

    Returns:
        Shape `(d, h, w)`. Every position is written exactly once; there is no "leftover" or
        default-valued position, because padding removal and the bijection together cover the
        whole unpadded grid.
    """
    d, h, w = token_grid
    d_pad, h_pad, w_pad = _decode_padded_indices(idx_windows, padded_shape)
    valid = (d_pad < d) & (h_pad < h) & (w_pad < w)
    unpadded_flat = d_pad * h * w + h_pad * w + w_pad

    grid = torch.zeros(d * h * w, dtype=received.dtype)
    grid[unpadded_flat[valid]] = received[valid]
    return grid.view(d, h, w)


def _normalize_map(voxel_map: Tensor, context: str) -> Tensor:
    """Min-max normalizes `voxel_map` to `[0, 1]`, guarding the constant-map case.

    Mirrors `neurovision.explainability.gradcam.grad_cam`'s `normalize` handling: a perfectly
    constant map (max == min) is returned as all zeros instead of dividing by zero, with a
    warning logged -- a real and meaningful outcome (e.g. `attention_rollout` with
    `residual_weight=1.0`, which is uniform by construction, see that function's docstring) not
    a bug to hide.

    Args:
        voxel_map: Any shape.
        context: A short string naming the caller, used in the warning message.

    Returns:
        `voxel_map` min-max scaled to `[0, 1]`, or all zeros if it was constant.
    """
    vmin = voxel_map.min()
    vmax = voxel_map.max()
    if (vmax - vmin).item() == 0.0:
        logger.warning(
            "%s: the map is constant (max == min); returning it as all zeros instead of "
            "dividing by zero.",
            context,
        )
        return torch.zeros_like(voxel_map)
    return (voxel_map - vmin) / (vmax - vmin)


def capture_attention(model: nn.Module, image: Tensor) -> AttentionCapture:
    """Runs `model(image)` once and captures every `WindowAttention` block's attention.

    Requires the model to already be in eval mode -- mirroring every other function in
    `neurovision.explainability` (see `gradcam.grad_cam` and
    `integrated_gradients.integrated_gradients`), this function does not flip the mode itself.
    Here the reason is that Swin's `DropPath` (stochastic depth) and any attention dropout are
    active in training mode, so a training-mode capture would reflect training-time
    stochasticity rather than the model's genuine inference-time attention.

    Runs entirely under `torch.no_grad()` -- this is a pure observation of an existing forward
    pass, no gradient is ever needed.

    Hooks each `WindowAttention` module's `.softmax` child (a forward hook capturing its
    output) and each OWNING BLOCK's forward-PRE-hook (capturing its input shape, to recover the
    block's own unpadded token grid `(d, h, w)`) -- hooking the block's actual input is the more
    robust way to learn the token grid than deriving it arithmetically from `nW` and `N`, since
    it works unchanged regardless of how MONAI's internals happen to be structured. Both kinds
    of hook are removed in a `finally`, whether or not the forward pass raises -- a hook left
    attached after an exception would silently change every later forward pass of this model.

    Args:
        model: Any `nn.Module` containing zero or more `WindowAttention` submodules (typically
            this project's `NeuroVisionX`, or a bare `SwinEncoder` / MONAI `SwinTransformer`).
            Must already be in eval mode.
        image: Input volume, shape `(B, C, D, H, W)`.

    Returns:
        An `AttentionCapture`. See that dataclass's docstring for each field.

    Raises:
        ValueError: If `model.training` is True; or if `model` has no `WindowAttention`
            submodules at all (the likely cause: a model with no Swin branch, e.g. the
            `unet3d` baseline -- attention rollout is only meaningful for a model with a Swin
            encoder).
    """
    if model.training:
        raise ValueError(
            "capture_attention requires model.training is False (eval mode), matching the "
            "convention of every other function in neurovision.explainability (see "
            "gradcam.grad_cam and integrated_gradients.integrated_gradients). In training mode "
            "Swin's DropPath (stochastic depth) and attention dropout are active, so a "
            "training-mode capture would reflect training-time stochasticity rather than the "
            "model's genuine inference-time attention. Call model.eval() yourself first -- this "
            "function does not flip your model's mode for you."
        )

    window_attn_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, WindowAttention)
    ]
    if not window_attn_modules:
        raise ValueError(
            "capture_attention: model has no WindowAttention submodules "
            "(monai.networks.nets.swin_unetr.WindowAttention). This usually means the model has "
            "no Swin branch at all -- e.g. a CNN-only baseline (unet3d) -- attention capture is "
            "only meaningful for a model with a Swin encoder."
        )

    handles: list = []
    block_input_shapes: dict[str, tuple[int, int, int]] = {}
    attention: dict[str, Tensor] = {}
    stage_names: list[str] = []

    try:
        for attn_name, attn_module in window_attn_modules:
            block_name = _block_name_from_attn_name(attn_name)
            stage_names.append(block_name)
            block_module = model.get_submodule(block_name)

            # Default-argument trick to bind `block_name` at hook-CREATION time rather than
            # call time -- the classic Python closure-in-a-loop pitfall, which here would
            # otherwise make every hook capture under the LAST block's name.
            def _capture_input_shape(
                _module: nn.Module, inputs: tuple, _name: str = block_name
            ) -> None:
                block_input_shapes[_name] = tuple(inputs[0].shape[1:4])

            def _capture_attention(
                _module: nn.Module, _inputs: tuple, output: Tensor, _name: str = block_name
            ) -> None:
                # Reduce over the head axis INSIDE the hook and detach immediately -- see this
                # module's top-of-file docstring, memory hazard: storing the raw
                # (nW, heads, N, N) tensor costs ~1 GB summed across stages at 96^3.
                attention[_name] = output.mean(dim=1).detach()

            handles.append(block_module.register_forward_pre_hook(_capture_input_shape))
            handles.append(attn_module.softmax.register_forward_hook(_capture_attention))

        with torch.no_grad():
            model(image)

        token_grids: dict[str, tuple[int, int, int]] = {}
        window_sizes: dict[str, tuple[int, int, int]] = {}
        shift_sizes: dict[str, tuple[int, int, int]] = {}
        for block_name in stage_names:
            token_grid = block_input_shapes.get(block_name)
            if token_grid is None:
                raise ValueError(
                    f"Block {block_name!r} never ran during this forward pass (its input hook "
                    "never fired), so no attention was captured for it. Check the block is "
                    "actually on the path model(image) executed."
                )
            block_module = model.get_submodule(block_name)
            window_size, shift_size = get_window_size(
                token_grid, tuple(block_module.window_size), tuple(block_module.shift_size)
            )
            token_grids[block_name] = token_grid
            window_sizes[block_name] = window_size
            shift_sizes[block_name] = shift_size
    finally:
        for handle in handles:
            handle.remove()

    return AttentionCapture(
        stage_names=stage_names,
        attention=attention,
        token_grids=token_grids,
        window_sizes=window_sizes,
        shift_sizes=shift_sizes,
        input_shape=tuple(image.shape[2:5]),
    )


def attention_to_voxel_map(
    capture: AttentionCapture, block_name: str, upsample: bool = True, normalize: bool = True
) -> Tensor:
    """Turns one block's captured attention into a voxel-space map.

    Reduces `(nW, N, N)` to received attention `(nW, N)` via the COLUMN mean (see this module's
    top-of-file docstring for why the row mean is uninformative), scatters it onto the block's
    unpadded token grid via the index-grid mapping (`_window_index_mapping` /
    `_scatter_received_to_grid`), then optionally upsamples and normalizes.

    Args:
        capture: The result of a `capture_attention` call.
        block_name: A dotted block name from `capture.stage_names`.
        upsample: If True (default), trilinear-interpolates (`align_corners=False`) up to
            `capture.input_shape`. IMPORTANT: the map's TRUE resolution is always the token
            grid's, coarser than the input by the block's stride -- an upsampled map must never
            be presented as having voxel-level precision.
        normalize: If True (default), min-max scales to `[0, 1]`. A constant map (e.g. uniform
            attention) is returned as all zeros with a warning logged, rather than dividing by
            zero.

    Returns:
        `(1, 1, D, H, W)` when `upsample`, else `(1, 1, d, h, w)` at the token grid's own
        resolution.

    Raises:
        ValueError: If `block_name` is not a key of `capture.attention`.
    """
    if block_name not in capture.attention:
        raise ValueError(
            f"attention_to_voxel_map: no captured attention for block_name={block_name!r}. "
            f"Available block names: {capture.stage_names}."
        )

    attn = capture.attention[block_name]  # (nW, N, N)
    token_grid = capture.token_grids[block_name]
    window_size = capture.window_sizes[block_name]
    shift_size = capture.shift_sizes[block_name]

    # Received attention: the COLUMN mean, not the row mean -- see this module's top-of-file
    # docstring for why a row mean is the uninformative constant 1/N.
    received = attn.mean(dim=-2)  # (nW, N)

    idx_windows, padded_shape = _window_index_mapping(token_grid, window_size, shift_size)
    grid = _scatter_received_to_grid(received, idx_windows, padded_shape, token_grid)

    voxel_map = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, d, h, w)
    if upsample:
        voxel_map = F.interpolate(
            voxel_map, size=capture.input_shape, mode="trilinear", align_corners=False
        )
    if normalize:
        voxel_map = _normalize_map(voxel_map, "attention_to_voxel_map")
    return voxel_map.detach()


def _blocks_in_stage(capture: AttentionCapture, stage: str) -> list[str]:
    """Finds `stage`'s two blocks (e.g. `"layers1"` -> `blocks.0`, `blocks.1`), in forward order.

    Args:
        capture: The result of a `capture_attention` call.
        stage: A stage name like `"layers1"`.

    Returns:
        Dotted block names belonging to `stage`, sorted by block index (so `blocks.0` -
        unshifted - always comes before `blocks.1` - shifted).
    """
    matches: list[tuple[int, str]] = []
    for name in capture.stage_names:
        parts = name.split(".")
        for j in range(len(parts) - 3):
            if parts[j] == stage and parts[j + 1] == "0" and parts[j + 2] == "blocks":
                matches.append((int(parts[j + 3]), name))
                break
    matches.sort(key=lambda item: item[0])
    return [name for _, name in matches]


def _full_attention_matrix(capture: AttentionCapture, block_name: str, num_tokens: int) -> Tensor:
    """Expands one block's windowed `(nW, N, N)` attention into a full `(T, T)` matrix.

    `T = num_tokens` is the stage's WHOLE unpadded token count. Uses the same index-grid mapping
    as `attention_to_voxel_map`, applied to BOTH the query and key axes at once (since a window
    slot's query and key tokens share the same window structure) -- this is what puts the two
    blocks of a stage, which partition into windows differently, into the same `(T, T)` basis so
    they can be composed by ordinary matrix multiplication.

    Args:
        capture: The result of a `capture_attention` call.
        block_name: A dotted block name from `capture.stage_names`.
        num_tokens: `d * h * w` for this stage's (shared, pre-any-block) token grid.

    Returns:
        `(T, T)`. Padded-position entries are simply absent (never written), consistent with
        every other reduction in this module.
    """
    attn = capture.attention[block_name]  # (nW, N, N)
    token_grid = capture.token_grids[block_name]
    window_size = capture.window_sizes[block_name]
    shift_size = capture.shift_sizes[block_name]
    d, h, w = token_grid

    idx_windows, padded_shape = _window_index_mapping(token_grid, window_size, shift_size)
    d_pad, h_pad, w_pad = _decode_padded_indices(idx_windows, padded_shape)
    valid = (d_pad < d) & (h_pad < h) & (w_pad < w)  # (nW, N)
    unpadded_flat = d_pad * h * w + h_pad * w + w_pad  # (nW, N)

    num_windows, window_tokens, _ = attn.shape
    row_idx = unpadded_flat.unsqueeze(-1).expand(num_windows, window_tokens, window_tokens)
    col_idx = unpadded_flat.unsqueeze(-2).expand(num_windows, window_tokens, window_tokens)
    valid_mat = valid.unsqueeze(-1) & valid.unsqueeze(-2)

    full = torch.zeros(num_tokens, num_tokens, dtype=attn.dtype)
    full[row_idx[valid_mat], col_idx[valid_mat]] = attn[valid_mat]
    return full


def attention_rollout(
    capture: AttentionCapture, stage: str, residual_weight: float = 0.5, max_tokens: int = 4096
) -> Tensor:
    """Composes one Swin stage's two blocks into a single attention map.

    See this module's top-of-file docstring for why this is a stage-LOCAL composition, not
    Abnar & Zuidema's published rollout algorithm (which assumes global attention over a token
    set fixed across every layer, plus a CLS token -- Swin has neither).

    Algorithm: each block's windowed attention is first expanded to a full `(T, T)` matrix over
    the stage's whole token grid (`_full_attention_matrix`, needed because the two blocks
    partition into windows differently and their raw `(N, N)` matrices are not in the same
    basis). Each expanded matrix gets the standard rollout step,
    `A_hat = residual_weight * I + (1 - residual_weight) * A`, row-normalized, then the two
    blocks are composed by ordinary matrix multiplication in forward order (`A_hat_1` then
    `A_hat_2`, i.e. `composed = A_hat_2 @ A_hat_1`). The composed matrix is reduced to a
    per-token map with the same received-attention (column mean) convention as
    `attention_to_voxel_map`, then always upsampled to `capture.input_shape` and normalized to
    `[0, 1]` -- the same return-shape convention as that function.

    Args:
        capture: The result of a `capture_attention` call.
        stage: A stage name like `"layers1"` -- see `_blocks_in_stage`.
        residual_weight: The rollout residual mixing weight, `[0, 1]`. At `1.0` each `A_hat`
            reduces to the identity matrix alone (row-normalized identity is still identity),
            so the composed matrix is exactly identity and the output map is perfectly uniform
            -- a useful sanity check that the residual mixing is applied in the direction this
            docstring describes.
        max_tokens: HARD guard on `T = d * h * w` for the stage's token grid, not a hint. A
            dense `(T, T)` matrix needs `T**2` entries -- `48**3 = 110,592` tokens at stage 1 of
            a 96^3 input would need `1.2e10` entries, which is simply impossible to materialize.
            Raises rather than attempting it. Default `4096`: at a 96^3 input only stage 3
            (`12**3 = 1728`) fits; at a 64^3 input stage 3 is `8**3 = 512`.

    Returns:
        `(1, 1, D, H, W)`, upsampled to `capture.input_shape` and normalized to `[0, 1]`.

    Raises:
        ValueError: If `stage` names no captured blocks; or if the stage's token count `T`
            exceeds `max_tokens` (the message states the actual `T`, the limit, and suggests
            `attention_to_voxel_map` on individual blocks or a coarser stage instead).
    """
    block_names = _blocks_in_stage(capture, stage)
    if not block_names:
        raise ValueError(
            f"attention_rollout: no blocks found for stage={stage!r}. Available block names: "
            f"{capture.stage_names}. `stage` should be a component like 'layers1', not a full "
            "block name."
        )

    d, h, w = capture.token_grids[block_names[0]]
    num_tokens = d * h * w
    if num_tokens > max_tokens:
        raise ValueError(
            f"attention_rollout: stage {stage!r} has T = {d} * {h} * {w} = {num_tokens} tokens, "
            f"which exceeds max_tokens={max_tokens}. A dense T x T rollout matrix would need "
            f"{num_tokens} * {num_tokens} = {num_tokens * num_tokens} entries, which cannot be "
            "materialized. Use attention_to_voxel_map on the individual blocks instead, or pick "
            "a coarser stage."
        )

    eye = torch.eye(num_tokens)
    composed: Tensor | None = None
    for block_name in block_names:
        full = _full_attention_matrix(capture, block_name, num_tokens)
        a_hat = residual_weight * eye + (1.0 - residual_weight) * full
        row_sum = a_hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        a_hat = a_hat / row_sum
        composed = a_hat if composed is None else a_hat @ composed

    assert composed is not None  # block_names is non-empty, so the loop ran at least once
    received = composed.mean(dim=0)  # (T,) -- received attention, same convention as elsewhere

    voxel_map = received.view(1, 1, d, h, w)
    voxel_map = F.interpolate(
        voxel_map, size=capture.input_shape, mode="trilinear", align_corners=False
    )
    voxel_map = _normalize_map(voxel_map, "attention_rollout")
    return voxel_map.detach()


def combine_stage_maps(maps: Sequence[Tensor], method: str = "mean") -> Tensor:
    """Combines several already-upsampled per-stage voxel maps into one figure-ready map.

    A PRESENTATION heuristic, not attention rollout across stages -- see this module's
    top-of-file docstring for why there is no mathematically defined way to propagate attention
    across a patch-merging boundary. Any figure using this must caption it as such.

    Args:
        maps: Voxel maps to combine, all the SAME shape (typically each already produced by
            `attention_to_voxel_map` or `attention_rollout` with `upsample=True`, so they share
            `capture.input_shape`).
        method: `"mean"` (elementwise average) or `"product"` (elementwise product).

    Returns:
        One tensor, the same shape as every entry of `maps`.

    Raises:
        ValueError: If `maps` is empty; if the maps do not all share the same shape; or if
            `method` is not `"mean"` or `"product"`.
    """
    if len(maps) == 0:
        raise ValueError("combine_stage_maps: maps is empty; nothing to combine.")

    reference_shape = maps[0].shape
    for i, m in enumerate(maps):
        if m.shape != reference_shape:
            raise ValueError(
                f"combine_stage_maps: maps[{i}] has shape {tuple(m.shape)}, expected "
                f"{tuple(reference_shape)} (maps[0]'s shape). All maps must share the same "
                "shape -- typically by upsampling each to the same capture.input_shape first."
            )

    stacked = torch.stack(list(maps), dim=0)
    if method == "mean":
        return stacked.mean(dim=0)
    elif method == "product":
        return stacked.prod(dim=0)
    else:
        raise ValueError(
            f"combine_stage_maps: unknown method {method!r}; expected 'mean' or 'product'."
        )
