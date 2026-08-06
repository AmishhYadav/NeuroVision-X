"""Fusion blocks that merge the CNN and Swin encoder branches.

`neurovision.models.encoders.cnn.CNNEncoder` and
`neurovision.models.encoders.swin.SwinEncoder` each produce a fine-to-coarse
feature pyramid. CNN level 0 (stride 1) has no Swin counterpart and is never
fused (see the stride-offset note in `neurovision.models.encoders.swin`); the
remaining four levels line up spatially and are fused one level at a time by
one of the blocks in this file. Every variant shares the same public
interface (`FusionBlock.forward`) so an ablation can swap the fusion variant
by config string (`neurovision.models.fusion.registry`) without touching the
decoder, which always sees `cnn_channels`-wide output regardless of which
variant ran.

## Three variants, one contract

- `AdaptiveGatedFusion` — the novel module. A learned adapter projects the
  Swin branch into the CNN branch's channel width, an explicit local
  ambiguity signal (`BranchAmbiguity`: per-branch region logits, their
  disagreement, and their entropy — see `docs/research/contribution.md` for
  why this, not branch content alone, is the actual contribution) feeds a
  small conv-net that predicts a spatially-varying gate from both branches
  plus that signal, windowed cross-attention lets every CNN voxel query the
  (projected) Swin context, and the gate controls how much of that attention
  output is admitted at each voxel: `out = cnn + layer_scale * gate *
  attn_out`. `use_ambiguity=False` drops the ambiguity signal entirely (the
  content-only ablation, rung 2 of the P2 ablation ladder).
- `ConcatFusion` — the standard "what everyone does" baseline: concatenate,
  1x1x1 conv back down to `cnn_channels`, norm, activation. No gating, no
  attention.
- `AddFusion` — the floor of the ablation: a 1x1x1 conv to fix the channel
  mismatch (unavoidable — the widths differ, so the tensors cannot be added
  directly) then a plain residual add. No norm, no nonlinearity, no gating.
  Present so the ablation table can show that `ConcatFusion`'s conv is doing
  real work, not just an artifact of concatenation itself.

## The memory trade-off this file is built around

Cross-attention between two full-resolution 3D feature maps is quadratic in
token count. At the finest fused level (stride 2, a 96^3 patch gives
48^3 = 110,592 tokens) a naive N x N attention matrix would need
110,592^2 ~= 1.2e10 entries — far outside the 16 GB Kaggle T4 budget, per
attention call, before even considering batch size or head count. Two
choices keep this tractable:

1. `AdaptiveGatedFusion` never materializes the N x N score matrix at all.
   Attention runs through `torch.nn.functional.scaled_dot_product_attention`
   (SDPA), whose memory-efficient/flash backends compute the same result as
   `softmax(q @ k.T / sqrt(d)) @ v` without ever forming the full score
   matrix in memory.
2. At coarser levels the token count is naturally small enough (stride 16 on
   a 96^3 patch: 6^3 = 216 tokens) that full global attention is affordable;
   at finer levels it is not, so attention is restricted to non-overlapping
   local windows of `window_size^3` tokens each. Which regime applies is
   decided at *runtime* from the actual token count
   (`full_attention_max_tokens`), not hardcoded per level — the same block
   class handles both, "full attention" being the degenerate case of a
   single window equal to the whole feature map.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from neurovision.models.fusion.registry import register_fusion

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# base class
# -----------------------------------------------------------------------------


class FusionBlock(nn.Module):
    """Shared contract for every fusion variant in this file.

    All three subclasses (`AdaptiveGatedFusion`, `ConcatFusion`, `AddFusion`)
    implement the same `forward` signature so the decoder and the ablation
    machinery can treat them interchangeably. Subclasses must NOT change this
    signature.

    Args:
        cnn_channels: Channel width of the CNN branch at this level. Also
            the output width of every fusion variant — the decoder always
            sees CNN widths, regardless of which variant is selected.
        swin_channels: Channel width of the Swin branch at this level.
    """

    def __init__(self, cnn_channels: int, swin_channels: int) -> None:
        super().__init__()
        self.cnn_channels = cnn_channels
        self.swin_channels = swin_channels

    def _validate_inputs(self, cnn_feat: Tensor, swin_feat: Tensor) -> None:
        """Checks the two branch features are alignable before fusing them.

        Args:
            cnn_feat: CNN branch feature, shape `(B, cnn_channels, D, H, W)`.
            swin_feat: Swin branch feature, shape `(B, swin_channels, D, H, W)`.

        Raises:
            ValueError: If the spatial shapes disagree, or either tensor's
                channel count does not match what this block was built for.
        """
        if cnn_feat.shape[2:] != swin_feat.shape[2:]:
            raise ValueError(
                f"cnn_feat spatial shape {tuple(cnn_feat.shape[2:])} does not match "
                f"swin_feat spatial shape {tuple(swin_feat.shape[2:])}; the two encoder "
                f"branches are misaligned."
            )
        if cnn_feat.shape[1] != self.cnn_channels:
            raise ValueError(
                f"cnn_feat has {cnn_feat.shape[1]} channels, expected {self.cnn_channels} "
                f"(this block was built for cnn_channels={self.cnn_channels})."
            )
        if swin_feat.shape[1] != self.swin_channels:
            raise ValueError(
                f"swin_feat has {swin_feat.shape[1]} channels, expected {self.swin_channels} "
                f"(this block was built for swin_channels={self.swin_channels})."
            )

    def forward(
        self, cnn_feat: Tensor, swin_feat: Tensor, return_gate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Fuses one pyramid level of the two branches.

        Args:
            cnn_feat: CNN branch feature, shape `(B, cnn_channels, D, H, W)`.
            swin_feat: Swin branch feature, shape
                `(B, swin_channels, D, H, W)`, same spatial shape as
                `cnn_feat`.
            return_gate: If True, also return the gate map used to blend the
                two branches. `AdaptiveGatedFusion` returns a real
                spatially-varying gate; `ConcatFusion` and `AddFusion` have
                no such concept and return `None` rather than fabricate one.

        Returns:
            `fused`, shape `(B, cnn_channels, D, H, W)`, when
            `return_gate=False`. Otherwise `(fused, gate)`, where `gate` is
            `(B, 1 or cnn_channels, D, H, W)` for `AdaptiveGatedFusion` or
            `None` for the two baselines.
        """
        raise NotImplementedError

    def forward_with_branch_logits(
        self, cnn_feat: Tensor, swin_feat: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Fuses one level and also returns the per-branch ambiguity logits.

        Default implementation for variants with no ambiguity mechanism
        (`ConcatFusion`, `AddFusion`): just runs `forward` and reports no
        branch logits. `AdaptiveGatedFusion` overrides this to actually
        return them, so an ablation can swap fusion variants without the
        caller (`NeuroVisionX`, which supervises the branch logits when
        present) needing to special-case which variant is active.

        Args:
            cnn_feat: CNN branch feature, shape `(B, cnn_channels, D, H, W)`.
            swin_feat: Swin branch feature, shape
                `(B, swin_channels, D, H, W)`.

        Returns:
            `(fused, branch_logits)`, `branch_logits` always `None` here.
        """
        return self.forward(cnn_feat, swin_feat), None


# -----------------------------------------------------------------------------
# windowed cross-attention helpers
# -----------------------------------------------------------------------------


def _window_partition(x: Tensor, window: tuple[int, int, int]) -> Tensor:
    """Splits a token grid into non-overlapping windows.

    Args:
        x: Tokens, shape `(B, D, H, W, C)`. `D`, `H`, `W` must already be
            exact multiples of `window` (pad before calling this).
        window: `(win_d, win_h, win_w)` window extent along each axis.

    Returns:
        Windows, shape `(B * n_windows, win_d * win_h * win_w, C)`, where
        `n_windows = (D // win_d) * (H // win_h) * (W // win_w)`. Windows for
        batch element `b` occupy `n_windows` consecutive rows starting at
        `b * n_windows` — this batch-major ordering is what
        `_window_reverse` (and the padding-mask tiling in
        `WindowedCrossAttention`) assumes.
    """
    B, D, H, W, C = x.shape
    win_d, win_h, win_w = window
    x = x.view(B, D // win_d, win_d, H // win_h, win_h, W // win_w, win_w, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, win_d * win_h * win_w, C)


def _window_reverse(
    windows: Tensor, window: tuple[int, int, int], batch: int, D: int, H: int, W: int
) -> Tensor:
    """Inverse of `_window_partition`.

    Args:
        windows: Shape `(batch * n_windows, win_d * win_h * win_w, C)`.
        window: `(win_d, win_h, win_w)` window extent used to partition.
        batch: Original batch size `B`.
        D: Padded depth the windows tile exactly.
        H: Padded height the windows tile exactly.
        W: Padded width the windows tile exactly.

    Returns:
        Tokens, shape `(B, D, H, W, C)`.
    """
    win_d, win_h, win_w = window
    C = windows.shape[-1]
    x = windows.view(batch, D // win_d, H // win_h, W // win_w, win_d, win_h, win_w, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch, D, H, W, C)


class WindowedCrossAttention(nn.Module):
    """CNN-queries-Swin cross-attention, full or windowed depending on size.

    Queries come from the CNN branch, keys/values from the (already
    channel-projected) Swin branch. See the module docstring for why the
    full-vs-windowed choice exists and why it is decided at runtime.

    Args:
        cnn_channels: Shared channel width of both streams (the Swin stream
            has already been projected to this width by the caller).
        num_heads: Number of attention heads. `cnn_channels % num_heads`
            must be 0 (checked by the caller, `AdaptiveGatedFusion`).
        window_size: Window extent along each spatial axis when the
            windowed path is taken.
        full_attention_max_tokens: If the feature map has at most this many
            voxels, attend globally over the whole map (one window equal to
            the whole feature map) instead of partitioning into local
            windows.
        attn_dropout: Dropout probability inside the attention itself
            (SDPA's `dropout_p`), applied only in training mode.
        proj_dropout: Dropout applied after the output projection.
    """

    def __init__(
        self,
        cnn_channels: int,
        num_heads: int,
        window_size: int,
        full_attention_max_tokens: int,
        attn_dropout: float,
        proj_dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.full_attention_max_tokens = full_attention_max_tokens
        self.attn_dropout = attn_dropout
        self.head_dim = cnn_channels // num_heads

        # Pre-norm, one LayerNorm per stream. Both are applied in TOKEN
        # layout (B, D, H, W, C) since LayerNorm normalizes over the last
        # dim — the (B, C, D, H, W) -> (B, D, H, W, C) permute is done once
        # and the resulting token tensor is reused for windowing/attention
        # itself, rather than permuting twice.
        self.q_norm = nn.LayerNorm(cnn_channels)
        self.kv_norm = nn.LayerNorm(cnn_channels)

        self.q_proj = nn.Linear(cnn_channels, cnn_channels, bias=True)
        self.k_proj = nn.Linear(cnn_channels, cnn_channels, bias=True)
        self.v_proj = nn.Linear(cnn_channels, cnn_channels, bias=True)
        self.out_proj = nn.Linear(cnn_channels, cnn_channels, bias=True)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, cnn_feat: Tensor, swin_proj: Tensor) -> Tensor:
        """Runs cross-attention: CNN queries attend to (projected) Swin.

        Args:
            cnn_feat: Query source, shape `(B, C, D, H, W)`.
            swin_proj: Key/value source, already projected to `C` channels,
                shape `(B, C, D, H, W)`, same spatial shape as `cnn_feat`.

        Returns:
            Attention output, shape `(B, C, D, H, W)`.
        """
        B, C, D, H, W = cnn_feat.shape
        N = D * H * W

        # Runtime decision, not a per-level hardcode: a single window equal
        # to the whole feature map IS full global attention, so both regimes
        # share the exact same code path below.
        if N <= self.full_attention_max_tokens:
            window = (D, H, W)
        else:
            window = (self.window_size, self.window_size, self.window_size)

        pad_d = (window[0] - D % window[0]) % window[0]
        pad_h = (window[1] - H % window[1]) % window[1]
        pad_w = (window[2] - W % window[2]) % window[2]
        needs_padding = pad_d > 0 or pad_h > 0 or pad_w > 0

        if needs_padding:
            # F.pad on a 5D tensor pads from the LAST dim backward:
            # (W_left, W_right, H_left, H_right, D_left, D_right). Padding
            # happens here, in feature layout, before the permute to token
            # layout below -- padding after the permute would pad the
            # channel dim instead of the spatial ones.
            cnn_feat = F.pad(cnn_feat, (0, pad_w, 0, pad_h, 0, pad_d))
            swin_proj = F.pad(swin_proj, (0, pad_w, 0, pad_h, 0, pad_d))

        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        # Token layout (B, Dp, Hp, Wp, C). LayerNorm is per-token (normalizes
        # over the channel dim only), so padding these extra zero-tokens has
        # no effect on the normalized value of any REAL token.
        q_tokens = self.q_norm(cnn_feat.permute(0, 2, 3, 4, 1))
        kv_tokens = self.kv_norm(swin_proj.permute(0, 2, 3, 4, 1))

        q_windows = _window_partition(q_tokens, window)  # (B*nW, L, C)
        kv_windows = _window_partition(kv_tokens, window)

        q = self.q_proj(q_windows)
        k = self.k_proj(kv_windows)
        v = self.v_proj(kv_windows)

        BW, L, _ = q.shape

        def _split_heads(t: Tensor) -> Tensor:
            # (BW, L, C) -> (BW, num_heads, L, head_dim), the layout SDPA
            # expects.
            return t.view(BW, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        attn_mask = None
        if needs_padding:
            # Key-padding mask, built once from geometry (batch-invariant)
            # and tiled across batch. True = real voxel (attend), False =
            # padding (exclude). Padding is always strictly less than one
            # window per axis and windows tile from the origin, so no window
            # can be ENTIRELY padding -- every query row keeps at least one
            # real key, so softmax never sees an all -inf row and no NaN can
            # arise.
            valid = torch.ones(1, Dp, Hp, Wp, 1, dtype=torch.bool, device=cnn_feat.device)
            valid[:, D:, :, :, :] = False
            valid[:, :, H:, :, :] = False
            valid[:, :, :, W:, :] = False
            valid_windows = _window_partition(valid, window)  # (nW, L, 1)
            # Tile across batch: _window_partition's batch-major ordering
            # (see its docstring) makes plain repeat() line up with the
            # q/kv windows' batch ordering.
            valid_windows = valid_windows.repeat(B, 1, 1)  # (B*nW, L, 1)
            # (B*nW, L, 1) -> (B*nW, 1, 1, L): broadcasts the same key mask
            # over every head and every query row.
            attn_mask = valid_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)

        # SDPA, not a hand-rolled softmax(q @ k.T / sqrt(d)) @ v: its
        # memory-efficient/flash backends never materialize the full L x L
        # score matrix, which is the whole reason this module fits the VRAM
        # budget at the finer, larger-token-count pyramid levels (see the
        # module docstring).
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(BW, L, C)
        attn_out = _window_reverse(attn_out, window, B, Dp, Hp, Wp)  # (B, Dp, Hp, Wp, C)
        attn_out = attn_out[:, :D, :H, :W, :]  # crop off padding

        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out.permute(0, 4, 1, 2, 3)  # back to (B, C, D, H, W)


class BranchAmbiguity(nn.Module):
    """Computes the explicit local ambiguity signal the gate conditions on.

    This is the project's actual contribution (see `docs/research/contribution.md`):
    prior gated fusion derives its mixing weight from branch *content* alone: the
    gate here additionally sees how much the CNN branch and the Swin branch
    *disagree* about this voxel's region labels. Each branch gets a lightweight
    1x1x1 conv that reads out region logits independently, and per-voxel
    disagreement plus each branch's own predictive entropy are concatenated
    onto the gate's input.

    Args:
        channels: Channel width both `cnn_feat` and `swin_proj` share. Note
            `swin_proj` is already the channel-projected Swin feature (see
            `AdaptiveGatedFusion._fuse`), so both convs read the same width.
        num_regions: Number of region channels (ET/TC/WT -> 3) each branch's
            auxiliary projection predicts.
    """

    def __init__(self, channels: int, num_regions: int) -> None:
        super().__init__()
        self.cnn_logits = nn.Conv3d(channels, num_regions, kernel_size=1, bias=True)
        self.swin_logits = nn.Conv3d(channels, num_regions, kernel_size=1, bias=True)

        # Zero-init the BIAS only (same reasoning as GateGenerator.conv_out):
        # centres each branch's predicted probability on sigmoid(0) = 0.5, i.e.
        # no prior toward either region or background before training has seen
        # any data. The WEIGHT is left at PyTorch's default random init so the
        # disagreement/entropy signal has spatial spread from the start --
        # zeroing it too would make disagreement identically 0 everywhere,
        # which is exactly the "gate can't see anything useful yet" state this
        # module exists to avoid.
        nn.init.zeros_(self.cnn_logits.bias)
        nn.init.zeros_(self.swin_logits.bias)

        # 3 * num_regions: disagreement (num_regions channels) + CNN entropy
        # (num_regions) + Swin entropy (num_regions). Exposed so
        # AdaptiveGatedFusion does not re-derive it when sizing GateGenerator.
        self.out_channels = 3 * num_regions

    def forward(self, cnn_feat: Tensor, swin_proj: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Computes per-branch region logits and the ambiguity signal.

        Args:
            cnn_feat: CNN branch feature, shape `(B, channels, D, H, W)`.
            swin_proj: Channel-projected Swin feature, shape
                `(B, channels, D, H, W)`.

        Returns:
            `(ambiguity, l_c, l_s)`:
                `ambiguity`: shape `(B, 3 * num_regions, D, H, W)`, every
                    channel in `[0, 1]` -- `[disagreement, h_cnn, h_swin]`
                    concatenated on the channel dim.
                `l_c`, `l_s`: RAW region logits from the CNN and Swin
                    branches respectively, each shape
                    `(B, num_regions, D, H, W)`. No sigmoid applied, matching
                    the project-wide convention that heads emit logits and the
                    loss applies its own.
        """
        # The features are DETACHED before the probes, and this is load-bearing
        # for the research claim rather than an optimization.
        #
        # These two convs are LINEAR PROBES: "what does this branch, exactly as it
        # currently is, think the label is here?" The gate then conditions on how
        # much the two probes disagree. If gradient flowed back through them into
        # the encoders, two things would go wrong, neither of which would fail a
        # test or show up as a bad number:
        #
        #   1. The branch-supervision term (losses/multitask.py) would push BOTH
        #      encoders toward predicting the label well, hence toward agreeing --
        #      collapsing the disagreement signal the gate exists to read. The
        #      objective would be quietly destroying its own input.
        #   2. The segmentation loss could reach the ambiguity map through the
        #      encoders and shape it into whatever makes segmentation easiest. The
        #      disagreement would stop being a measurement and become a free latent
        #      quantity -- which is exactly the thing docs/research/contribution.md
        #      says prior content-only gates do and that this module is claimed to
        #      improve on.
        #
        # Detaching makes disagreement an honest read-out of two branches that were
        # trained independently by the main objective. The probe weights themselves
        # still train normally: they receive gradient from the branch-supervision
        # term and from the gate path, just not at the cost of perturbing what they
        # measure. Same reasoning as the confidence head's no-grad target.
        l_c = self.cnn_logits(cnn_feat.detach())
        l_s = self.swin_logits(swin_proj.detach())

        p_c = torch.sigmoid(l_c)
        p_s = torch.sigmoid(l_s)

        disagreement = (p_c - p_s).abs()

        # eps clamp is LOAD-BEARING, not defensive style: a saturated logit
        # gives p exactly 0.0 or 1.0 in floating point, log(0) is -inf, and
        # 0 * -inf is NaN. A trained (or even just confident, early-training)
        # branch saturates routinely, and a NaN here would propagate silently
        # into the gate and then into every downstream feature with no error
        # anywhere.
        eps = 1e-6
        p_c_clamped = p_c.clamp(eps, 1.0 - eps)
        p_s_clamped = p_s.clamp(eps, 1.0 - eps)

        # Per-channel Bernoulli entropy, normalized to [0, 1] by dividing by
        # ln 2 (its maximum, at p=0.5). cnn_feat/swin_proj are GroupNorm'd and
        # roughly unit-scale; an un-normalized entropy (max 0.693 nats) and a
        # disagreement already in [0, 1] would otherwise enter the gate's
        # first conv at two arbitrarily different scales for no reason.
        log2 = torch.log(torch.tensor(2.0, dtype=p_c.dtype, device=p_c.device))

        def _entropy(p: Tensor) -> Tensor:
            return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)) / log2

        h_c = _entropy(p_c_clamped)
        h_s = _entropy(p_s_clamped)

        # Per-region, not summed to one channel: ET is the region the
        # calibration claim leans on, and summing would make an ET-only
        # disagreement indistinguishable from a WT-only one -- exactly the
        # distinction the gate needs to condition on.
        #
        # Memory: at the finest fused level (stride 2, 48^3 tokens from a 96^3
        # patch) this is 3 * num_regions = 9 channels, ~1 MB/sample in fp16 --
        # negligible against the ~790 MB that level's attention already costs.
        ambiguity = torch.cat([disagreement, h_c, h_s], dim=1)
        return ambiguity, l_c, l_s


class GateGenerator(nn.Module):
    """Predicts a spatially-varying gate from the concatenated branch features.

    `Conv3d(2C [+ extra], hidden, 1) -> GroupNorm -> LeakyReLU ->
    Conv3d(hidden, out, 3) -> Sigmoid`. See `AdaptiveGatedFusion`'s docstring
    for why the gate exists and how it is used in the merge.

    Args:
        cnn_channels: CNN branch channel width `C`. The input to this module
            is `2 * C` wide (CNN feature concatenated with projected Swin
            feature) plus `extra_channels` when ambiguity conditioning is on.
        gate_channels: `"scalar"` for a single gate value per voxel
            (broadcasts over channels) or `"channel"` for one gate value per
            channel per voxel.
        gate_reduction: Bottleneck reduction factor for the hidden width.
        num_groups: GroupNorm group count.
        extra_channels: Width of an additional signal concatenated onto the
            gate's input alongside the two branch features -- the
            `BranchAmbiguity` output when ambiguity conditioning is enabled,
            0 otherwise (the content-only ablation).
    """

    def __init__(
        self,
        cnn_channels: int,
        gate_channels: str,
        gate_reduction: int,
        num_groups: int,
        extra_channels: int = 0,
    ) -> None:
        super().__init__()
        # Guarantees hidden % num_groups == 0 (GroupNorm's requirement) and a
        # floor of num_groups, so an aggressive reduction factor can never
        # shrink the hidden width below what GroupNorm needs.
        hidden = max(num_groups, (cnn_channels // gate_reduction // num_groups) * num_groups)
        gate_out = 1 if gate_channels == "scalar" else cnn_channels

        self.extra_channels = extra_channels
        self.conv_in = nn.Conv3d(
            2 * cnn_channels + extra_channels, hidden, kernel_size=1, bias=False
        )
        self.norm = nn.GroupNorm(num_groups, hidden)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # kernel_size=3, not 1: the gate decides WHERE transformer context is
        # admitted, so it should see a local neighbourhood rather than
        # deciding voxel-by-voxel with no spatial context at all.
        self.conv_out = nn.Conv3d(hidden, gate_out, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # Zero-initializing the final conv's BIAS centres the gate on
        # sigmoid(0) = 0.5 -- no prior favouring either branch before training
        # has seen any data. Its WEIGHT is deliberately left at PyTorch's
        # default random init, so the gate is centred on 0.5 but not flat at
        # it: measured on random inputs the initial gate spreads roughly
        # 0.27-0.73 (mean ~0.50, std ~0.07). That spread is wanted. A
        # zero-initialized weight would make the gate exactly 0.5 at every
        # voxel, i.e. spatially symmetric, and the block already starts as a
        # near-identity via `layer_scale` -- killing the gate's spatial
        # variation too would leave nothing to break the symmetry the gate
        # exists to learn.
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self, cnn_feat: Tensor, swin_proj: Tensor, ambiguity: Tensor | None = None
    ) -> Tensor:
        """Computes the gate map.

        Args:
            cnn_feat: CNN branch feature, shape `(B, C, D, H, W)`.
            swin_proj: Channel-projected Swin feature, shape
                `(B, C, D, H, W)`.
            ambiguity: The `BranchAmbiguity` output, shape
                `(B, extra_channels, D, H, W)`, or `None` when
                `extra_channels == 0`.

        Returns:
            Gate, shape `(B, 1, D, H, W)` (scalar) or `(B, C, D, H, W)`
            (channel), values in `(0, 1)`.

        Raises:
            ValueError: If `ambiguity` is supplied but this instance has
                `extra_channels == 0`, or omitted while `extra_channels > 0`.
                A silently-ignored (or silently-missing) ambiguity tensor
                would make the content-only ablation and the full model
                numerically identical while the config claims they differ.
        """
        if self.extra_channels == 0:
            if ambiguity is not None:
                raise ValueError(
                    "GateGenerator was built with extra_channels=0 (no ambiguity "
                    "conditioning) but an ambiguity tensor was passed in."
                )
            x = torch.cat([cnn_feat, swin_proj], dim=1)
        else:
            if ambiguity is None:
                raise ValueError(
                    f"GateGenerator was built with extra_channels={self.extra_channels} "
                    "and requires an ambiguity tensor of that width, but none was passed."
                )
            x = torch.cat([cnn_feat, swin_proj, ambiguity], dim=1)
        x = self.act(self.norm(self.conv_in(x)))
        return self.sigmoid(self.conv_out(x))


# -----------------------------------------------------------------------------
# the three fusion variants
# -----------------------------------------------------------------------------


class AdaptiveGatedFusion(FusionBlock):
    """The novel fusion module: gated windowed cross-attention.

    Forward, in order:

    1. Project the Swin feature into the CNN branch's channel width with a
       learned `Conv3d(1x1x1) -> GroupNorm` adapter, applied unconditionally
       even when the widths already match by coincidence — it is adapting
       between two differently-scaled feature spaces, not just fixing a
       width mismatch.
    2. Predict a gate map from `[cnn_feat, swin_proj]` (`GateGenerator`).
    3. Run windowed cross-attention, CNN queries against projected-Swin
       keys/values (`WindowedCrossAttention`).
    4. Merge: `out = cnn_feat + layer_scale * gate * attn_out`.

    ## Why the merge is `cnn + gate * attn`, not `gate * cnn + (1-gate) * swin`

    The CNN branch carries the full-resolution spatial detail this project's
    boundary-accuracy claim depends on and must never be attenuated by the
    gate. The gate instead controls *how much transformer context is
    admitted* at each voxel — a quantity with a direct, single-purpose
    meaning that is also directly plottable as a per-voxel figure for the
    paper (e.g. "the model leans on global context near the tumor margin,
    not in healthy tissue").

    ## Why `layer_scale_init=1e-4`, not 0 or 1

    At init, `layer_scale` (an `nn.Parameter` of shape `(1, C, 1, 1, 1)`)
    scales the attention branch's contribution to almost nothing, so the
    block starts as a near-identity pass-through of the CNN branch — the
    same reasoning `zero_init_residual` uses in `cnn.py`: a randomly
    initialized attention stack should not be allowed to destabilize early
    training. At exactly `layer_scale_init=0.0` the block is an EXACT
    identity (pinned by a test).

    Args:
        cnn_channels: CNN branch channel width. Also this block's output
            width.
        swin_channels: Swin branch channel width.
        num_heads: Attention heads. `cnn_channels % num_heads` must be 0.
        window_size: Local attention window extent per axis.
        full_attention_max_tokens: Token-count threshold below which
            attention runs globally instead of windowed. See the module
            docstring.
        gate_channels: `"scalar"` or `"channel"` gate granularity.
        gate_reduction: Gate MLP bottleneck reduction factor.
        num_groups: GroupNorm group count, shared by the projection and gate
            generator. `cnn_channels % num_groups` must be 0.
        attn_dropout: Dropout inside attention (training mode only).
        proj_dropout: Dropout after the attention output projection.
        layer_scale_init: Initial value of the residual scale. See above.
        use_checkpoint: If True, checkpoint the fusion computation during
            training to trade compute for activation memory. Ignored when
            `return_gate=True` — see `forward`.
        use_ambiguity: If True (the default), condition the gate on the
            explicit inter-branch ambiguity signal (`BranchAmbiguity`) --
            this is the project's actual contribution, see
            `docs/research/contribution.md`. If False, the gate sees only
            `[cnn_feat, swin_proj]`, i.e. the content-only ablation
            (rung 2 of the P2 ablation ladder); `BranchAmbiguity` is not
            constructed at all in that case, so the ablation is genuinely
            parameter-matched-minus-the-mechanism rather than a
            built-but-unused module inflating the parameter count.
        num_regions: Number of region channels each branch's auxiliary
            projection predicts (ET/TC/WT -> 3). Unused when
            `use_ambiguity=False`.

    Raises:
        ValueError: If `cnn_channels % num_heads != 0`,
            `cnn_channels % num_groups != 0`, or `gate_channels` is not
            `"scalar"` or `"channel"`.
    """

    def __init__(
        self,
        cnn_channels: int,
        swin_channels: int,
        num_heads: int = 4,
        window_size: int = 4,
        full_attention_max_tokens: int = 512,
        gate_channels: str = "scalar",
        gate_reduction: int = 4,
        num_groups: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        layer_scale_init: float = 1e-4,
        use_checkpoint: bool = False,
        use_ambiguity: bool = True,
        num_regions: int = 3,
    ) -> None:
        super().__init__(cnn_channels, swin_channels)

        if cnn_channels % num_heads != 0:
            raise ValueError(
                f"cnn_channels ({cnn_channels}) must be divisible by num_heads "
                f"({num_heads}) for per-head attention widths to be integral."
            )
        if cnn_channels % num_groups != 0:
            raise ValueError(
                f"cnn_channels ({cnn_channels}) must be divisible by num_groups "
                f"({num_groups}). GroupNorm requires num_channels % num_groups == 0."
            )
        if gate_channels not in ("scalar", "channel"):
            raise ValueError(f"gate_channels must be 'scalar' or 'channel', got {gate_channels!r}.")

        self.use_checkpoint = use_checkpoint
        self.use_ambiguity = use_ambiguity

        # bias=False: GroupNorm immediately follows and applies its own
        # learned per-channel shift, matching the convention in cnn.py.
        self.swin_proj_conv = nn.Conv3d(swin_channels, cnn_channels, kernel_size=1, bias=False)
        self.swin_proj_norm = nn.GroupNorm(num_groups, cnn_channels)

        # BranchAmbiguity is only constructed when the ablation asks for it --
        # not built-but-unused, so the content-only ablation is genuinely
        # parameter-free in this respect (see the arg docstring above).
        if use_ambiguity:
            self.ambiguity = BranchAmbiguity(cnn_channels, num_regions)
            extra_channels = self.ambiguity.out_channels
        else:
            self.ambiguity = None
            extra_channels = 0

        self.gate_generator = GateGenerator(
            cnn_channels, gate_channels, gate_reduction, num_groups, extra_channels=extra_channels
        )
        self.cross_attn = WindowedCrossAttention(
            cnn_channels,
            num_heads,
            window_size,
            full_attention_max_tokens,
            attn_dropout,
            proj_dropout,
        )

        self.layer_scale = nn.Parameter(torch.full((1, cnn_channels, 1, 1, 1), layer_scale_init))

    def _fuse(
        self, cnn_feat: Tensor, swin_feat: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """The actual fusion computation, factored out so it can be
        optionally wrapped in `torch.utils.checkpoint.checkpoint`.

        Args:
            cnn_feat: shape `(B, cnn_channels, D, H, W)`.
            swin_feat: shape `(B, swin_channels, D, H, W)`.

        Returns:
            `(fused, gate, branch_logits)`. `fused` shape
            `(B, cnn_channels, D, H, W)`, `gate` shape
            `(B, 1 or cnn_channels, D, H, W)`. `branch_logits` is the
            `(l_c, l_s)` pair from `BranchAmbiguity` when `use_ambiguity` is
            True, else `None`.
        """
        swin_proj = self.swin_proj_norm(self.swin_proj_conv(swin_feat))

        if self.ambiguity is not None:
            ambiguity, l_c, l_s = self.ambiguity(cnn_feat, swin_proj)
            gate = self.gate_generator(cnn_feat, swin_proj, ambiguity)
            branch_logits: tuple[Tensor, Tensor] | None = (l_c, l_s)
        else:
            gate = self.gate_generator(cnn_feat, swin_proj)
            branch_logits = None

        attn_out = self.cross_attn(cnn_feat, swin_proj)
        # Broadcasting handles both the scalar gate (1 channel) and the
        # per-channel gate (cnn_channels channels) with the same expression.
        fused = cnn_feat + self.layer_scale * gate * attn_out
        return fused, gate, branch_logits

    def forward(
        self, cnn_feat: Tensor, swin_feat: Tensor, return_gate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Fuses one pyramid level. See `FusionBlock.forward` for the shared contract.

        Args:
            cnn_feat: shape `(B, cnn_channels, D, H, W)`.
            swin_feat: shape `(B, swin_channels, D, H, W)`.
            return_gate: If True, also return the gate map (see `_fuse`).

        Returns:
            `fused` alone, or `(fused, gate)` when `return_gate=True`.
        """
        self._validate_inputs(cnn_feat, swin_feat)

        if return_gate:
            # Checkpointing a multi-output forward is avoidable complexity
            # for what is only a debug/visualization path (plotting the gate
            # map for the paper) -- always take the plain, non-checkpointed
            # path when the caller wants the gate.
            fused, gate, _branch_logits = self._fuse(cnn_feat, swin_feat)
            return fused, gate

        # Same guard as CNNEncoder.forward: only checkpoint in training with
        # grad enabled. torch.utils.checkpoint warns and silently returns no
        # gradient at all when nothing in the input requires grad, which is
        # exactly the eval/no_grad case.
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            fused, _gate, _branch_logits = checkpoint(
                self._fuse, cnn_feat, swin_feat, use_reentrant=False
            )
        else:
            fused, _gate, _branch_logits = self._fuse(cnn_feat, swin_feat)
        return fused

    def forward_with_branch_logits(
        self, cnn_feat: Tensor, swin_feat: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Fuses one level and also returns the per-branch ambiguity logits.

        The branch logits are what `BranchAmbiguity` reads out from each
        encoder branch independently -- exposed here so a caller (e.g. a
        deep-supervision-style loss on the ambiguity mechanism itself) can
        supervise them directly. Uses the SAME checkpointing guard as
        `forward`: checkpointing is what keeps this block inside the 16 GB
        VRAM budget, and the supervised-training path is exactly where that
        matters.

        Args:
            cnn_feat: shape `(B, cnn_channels, D, H, W)`.
            swin_feat: shape `(B, swin_channels, D, H, W)`.

        Returns:
            `(fused, branch_logits)`. `branch_logits` is `(l_c, l_s)`, each
            shape `(B, num_regions, D, H, W)`, when `use_ambiguity` is True;
            `None` otherwise.
        """
        self._validate_inputs(cnn_feat, swin_feat)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            fused, _gate, branch_logits = checkpoint(
                self._fuse, cnn_feat, swin_feat, use_reentrant=False
            )
        else:
            fused, _gate, branch_logits = self._fuse(cnn_feat, swin_feat)
        return fused, branch_logits


class ConcatFusion(FusionBlock):
    """Ablation baseline: concatenate, then conv back down to CNN width.

    `Conv3d(cnn_channels + swin_channels, cnn_channels, 1) -> GroupNorm ->
    LeakyReLU`. The standard "what everyone does" fusion — no gating, no
    attention — and the baseline the novel module has to beat.

    Args:
        cnn_channels: CNN branch channel width. Also this block's output
            width.
        swin_channels: Swin branch channel width.
        num_groups: GroupNorm group count. `cnn_channels % num_groups` must
            be 0.
    """

    def __init__(self, cnn_channels: int, swin_channels: int, num_groups: int = 8) -> None:
        super().__init__(cnn_channels, swin_channels)
        # Same check (and same message) as AdaptiveGatedFusion, so a bad
        # config fails identically whichever variant the ablation selected
        # rather than falling through to PyTorch's generic GroupNorm error.
        if cnn_channels % num_groups != 0:
            raise ValueError(
                f"cnn_channels ({cnn_channels}) must be divisible by num_groups "
                f"({num_groups}). GroupNorm requires num_channels % num_groups == 0."
            )
        self.conv = nn.Conv3d(cnn_channels + swin_channels, cnn_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups, cnn_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(
        self, cnn_feat: Tensor, swin_feat: Tensor, return_gate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Fuses one pyramid level. See `FusionBlock.forward` for the shared contract.

        Args:
            cnn_feat: shape `(B, cnn_channels, D, H, W)`.
            swin_feat: shape `(B, swin_channels, D, H, W)`.
            return_gate: If True, return `(fused, None)` — this variant has
                no gate.

        Returns:
            `fused` alone, or `(fused, None)` when `return_gate=True`.
        """
        self._validate_inputs(cnn_feat, swin_feat)
        x = torch.cat([cnn_feat, swin_feat], dim=1)
        fused = self.act(self.norm(self.conv(x)))
        if return_gate:
            return fused, None
        return fused


class AddFusion(FusionBlock):
    """Ablation baseline: the floor. A width-matching conv, then a plain add.

    `out = cnn_feat + Conv3d(swin_channels, cnn_channels, 1)(swin_feat)`. The
    1x1x1 conv is unavoidable (the widths differ, so the tensors cannot be
    added directly) but this is otherwise the weakest possible fusion — no
    gating, no norm, no nonlinearity. Present to show that `ConcatFusion`'s
    conv is doing real work, not merely benefiting from concatenation itself.

    Args:
        cnn_channels: CNN branch channel width. Also this block's output
            width.
        swin_channels: Swin branch channel width.
    """

    def __init__(self, cnn_channels: int, swin_channels: int) -> None:
        super().__init__(cnn_channels, swin_channels)
        self.conv = nn.Conv3d(swin_channels, cnn_channels, kernel_size=1, bias=False)

    def forward(
        self, cnn_feat: Tensor, swin_feat: Tensor, return_gate: bool = False
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Fuses one pyramid level. See `FusionBlock.forward` for the shared contract.

        Args:
            cnn_feat: shape `(B, cnn_channels, D, H, W)`.
            swin_feat: shape `(B, swin_channels, D, H, W)`.
            return_gate: If True, return `(fused, None)` — this variant has
                no gate.

        Returns:
            `fused` alone, or `(fused, None)` when `return_gate=True`.
        """
        self._validate_inputs(cnn_feat, swin_feat)
        fused = cnn_feat + self.conv(swin_feat)
        if return_gate:
            return fused, None
        return fused


# -----------------------------------------------------------------------------
# registry builders
# -----------------------------------------------------------------------------


@register_fusion("adaptive_gated")
def build_adaptive_gated_fusion(
    cfg: Any, cnn_channels: int, swin_channels: int, level: int
) -> nn.Module:
    """Builds `AdaptiveGatedFusion` from `cfg.model.fusion`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.model.fusion` with
            keys `num_heads`, `window_size`, `full_attention_max_tokens`,
            `gate_channels`, `gate_reduction`, `num_groups`, `attn_dropout`,
            `proj_dropout`, `layer_scale_init`, `use_checkpoint`, and
            optionally `use_ambiguity` (default True when absent, so an
            older config composed before this key existed still builds).
            `num_regions` is read from `cfg.model.out_channels`.
        cnn_channels: CNN branch channel width at this level.
        swin_channels: Swin branch channel width at this level.
        level: Index into the fused pyramid (0 = finest). Unused by this
            variant today — `AdaptiveGatedFusion` picks full-vs-windowed
            attention from the runtime token count in `forward`, not from
            which level it was built for — but kept in the signature because
            it is part of the shared `FusionBuilder` registry contract
            (`neurovision.models.fusion.registry`), which a future variant
            may need.

    Returns:
        A constructed `AdaptiveGatedFusion`.
    """
    fusion_cfg = cfg.model.fusion
    block = AdaptiveGatedFusion(
        cnn_channels=cnn_channels,
        swin_channels=swin_channels,
        num_heads=fusion_cfg.num_heads,
        window_size=fusion_cfg.window_size,
        full_attention_max_tokens=fusion_cfg.full_attention_max_tokens,
        gate_channels=fusion_cfg.gate_channels,
        gate_reduction=fusion_cfg.gate_reduction,
        num_groups=fusion_cfg.num_groups,
        attn_dropout=fusion_cfg.attn_dropout,
        proj_dropout=fusion_cfg.proj_dropout,
        layer_scale_init=fusion_cfg.layer_scale_init,
        use_checkpoint=fusion_cfg.use_checkpoint,
        use_ambiguity=fusion_cfg.get("use_ambiguity", True),
        num_regions=cfg.model.out_channels,
    )
    logger.debug(
        "Built AdaptiveGatedFusion for level %d: cnn=%d ch, swin=%d ch",
        level,
        cnn_channels,
        swin_channels,
    )
    return block


@register_fusion("concat")
def build_concat_fusion(cfg: Any, cnn_channels: int, swin_channels: int, level: int) -> nn.Module:
    """Builds `ConcatFusion` from `cfg.model.fusion`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.model.fusion.num_groups`.
        cnn_channels: CNN branch channel width at this level.
        swin_channels: Swin branch channel width at this level.
        level: Index into the fused pyramid (0 = finest). Unused — see
            `build_adaptive_gated_fusion` for why the signature still carries
            it.

    Returns:
        A constructed `ConcatFusion`.
    """
    block = ConcatFusion(
        cnn_channels=cnn_channels,
        swin_channels=swin_channels,
        num_groups=cfg.model.fusion.num_groups,
    )
    logger.debug(
        "Built ConcatFusion for level %d: cnn=%d ch, swin=%d ch", level, cnn_channels, swin_channels
    )
    return block


@register_fusion("add")
def build_add_fusion(cfg: Any, cnn_channels: int, swin_channels: int, level: int) -> nn.Module:
    """Builds `AddFusion`. Reads no sub-keys from `cfg.model.fusion`.

    Args:
        cfg: The full composed Hydra config. Not read by this builder —
            `AddFusion` has no configurable sub-keys — but accepted to match
            the shared `FusionBuilder` registry contract.
        cnn_channels: CNN branch channel width at this level.
        swin_channels: Swin branch channel width at this level.
        level: Index into the fused pyramid (0 = finest). Unused — see
            `build_adaptive_gated_fusion` for why the signature still carries
            it.

    Returns:
        A constructed `AddFusion`.
    """
    block = AddFusion(cnn_channels=cnn_channels, swin_channels=swin_channels)
    logger.debug(
        "Built AddFusion for level %d: cnn=%d ch, swin=%d ch", level, cnn_channels, swin_channels
    )
    return block
