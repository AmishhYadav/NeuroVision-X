"""Baseline segmentation model builders: a plain 3D U-Net and SwinUNETR.

Both models end in exactly 3 output channels, not 4. BraTS is scored on
three NESTED regions — ET (enhancing tumor), TC (tumor core), WT (whole
tumor) — not on the four raw, mutually exclusive class labels
`{background, necrotic core, edema, enhancing tumor}`. A voxel inside the
enhancing tumor is legitimately ET, TC, AND WT at once, so a 4-way softmax
over the raw classes is the wrong output layout entirely: softmax forces the
channels to compete for one shared probability budget, but the regions
overlap by definition. The heads here are 3 independent sigmoid channels,
matching `neurovision.losses.segmentation.DiceBCELoss` and
`neurovision.metrics.segmentation`, which are both written against this same
3-channel region layout. See `configs/data/brats.yaml` for the region
definitions.

`out_channels` therefore resolves to `cfg.data.num_classes` (== 3, the region
count), while `in_channels` resolves to `cfg.data.in_channels` (== 4, the
four MRI modalities T1/T1CE/T2/FLAIR stacked as input channels). The two
numbers happen to be close (3 vs 4) and easy to transpose by accident; both
`configs/model/unet3d.yaml` and `configs/model/swinunetr.yaml` already
interpolate them correctly from `cfg.data`, so a builder that reads
`cfg.model.in_channels` / `cfg.model.out_channels` (rather than reaching into
`cfg.data` directly) gets the right values without having to re-derive them.
"""

from __future__ import annotations

from typing import Any

from monai.networks.nets import SwinUNETR, UNet
from torch import nn

from neurovision.models.registry import register_model


def _to_tuple(value: Any) -> tuple:
    """Converts a sequence-valued config field to a plain tuple.

    Hydra hands sequence fields over as OmegaConf `ListConfig` objects, which
    are not `list`/`tuple` instances. MONAI's networks branch on `isinstance`
    checks in a few places, so every sequence read from config is normalized
    here before being passed on. Iterating a `ListConfig` already yields plain
    Python scalars, so `tuple()` is all that is needed.

    Args:
        value: A `ListConfig`, list, tuple, or other sequence.

    Returns:
        A plain `tuple` with the same elements.
    """
    return tuple(value)


@register_model("unet3d")
def build_unet3d(cfg: Any) -> nn.Module:
    """Builds the plain 3D U-Net baseline from `cfg.model`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.model` with keys
            matching `configs/model/unet3d.yaml`: `in_channels`,
            `out_channels`, `channels`, `strides`, `num_res_units`, `norm`,
            `activation`, `dropout`, `deep_supervision`.

    Returns:
        A `monai.networks.nets.UNet` with `spatial_dims=3`.

    Raises:
        ValueError: If `cfg.model.deep_supervision` is true. MONAI's `UNet`
            has a single output head and cannot emit the extra
            lower-resolution outputs deep supervision needs, so silently
            ignoring the flag would train a model the config claims has deep
            supervision but does not.
    """
    model_cfg = cfg.model

    if model_cfg.deep_supervision:
        raise ValueError(
            "model.deep_supervision=true is not supported by 'unet3d': MONAI's UNet "
            "has a single output head. Set model.deep_supervision=false; deep "
            "supervision for this architecture would need training.loss.deep_supervision "
            "on a model with multiple decoder outputs, which UNet does not have."
        )

    return UNet(
        spatial_dims=3,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        channels=_to_tuple(model_cfg.channels),
        strides=_to_tuple(model_cfg.strides),
        num_res_units=model_cfg.num_res_units,
        norm=model_cfg.norm,
        act=model_cfg.activation,
        dropout=model_cfg.dropout,
    )


@register_model("swinunetr")
def build_swinunetr(cfg: Any) -> nn.Module:
    """Builds the SwinUNETR baseline from `cfg.model`.

    Args:
        cfg: The full composed Hydra config, exposing `cfg.model` with keys
            matching `configs/model/swinunetr.yaml`: `in_channels`,
            `out_channels`, `feature_size`, `depths`, `num_heads`,
            `norm_name`, `drop_rate`, `attn_drop_rate`,
            `dropout_path_rate`, `use_checkpoint`.

    Returns:
        A `monai.networks.nets.SwinUNETR` with `spatial_dims=3`.
    """
    model_cfg = cfg.model

    # MONAI 1.6.0's SwinUNETR.__init__ has NO `img_size` parameter (removed
    # from earlier MONAI versions) — do not pass one.
    #
    # Measured constraint: SwinUNETR downsamples 32x (patch embed 2x, then 4
    # more 2x stages), so an input smaller than 64 voxels on any axis reaches
    # the bottleneck at 1x1x1, and InstanceNorm3d raises `ValueError:
    # Expected more than 1 spatial element when training` there. Calling
    # `.eval()` does not avoid this: InstanceNorm3d has
    # `track_running_stats=False`, so it computes per-instance statistics
    # from the batch in eval mode too, and a 1x1x1 spatial map has no
    # variance to compute. The 96^3 training patch is fine (bottleneck is
    # 3x3x3); this only constrains small inputs used in tests.
    return SwinUNETR(
        spatial_dims=3,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        feature_size=model_cfg.feature_size,
        depths=_to_tuple(model_cfg.depths),
        num_heads=_to_tuple(model_cfg.num_heads),
        norm_name=model_cfg.norm_name,
        drop_rate=model_cfg.drop_rate,
        attn_drop_rate=model_cfg.attn_drop_rate,
        dropout_path_rate=model_cfg.dropout_path_rate,
        use_checkpoint=model_cfg.use_checkpoint,
    )
