"""MONAI transform pipelines for BraTS training and validation.

The core job of this module is `ConvertToRegionsd`: turning the single-channel
integer label produced by `neurovision.data.preprocessing` into the three
overlapping binary regions (ET, TC, WT) that BraTS is actually scored on and
that the model's sigmoid heads predict.

`build_train_transforms` and `build_val_transforms` assemble the full MONAI
`Compose` pipelines that `neurovision.data`'s dataset classes apply to each
loaded case, reading every tunable value from `cfg.data` (Hydra config) so
nothing here is hardcoded.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from typing import Any

import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

logger = logging.getLogger(__name__)

# Fixed region order. Must match `cfg.data.regions` in configs/data/brats.yaml
# — everything downstream (loss channels, metric names, decoder head outputs)
# assumes this exact order.
REGION_NAMES: tuple[str, ...] = ("ET", "TC", "WT")

# Raw, contiguous class values written by scripts/preprocess.py: background=0,
# necrotic/non-enhancing core=1, edema=2, enhancing tumor=3 (raw BraTS value 4
# was remapped to 3 during preprocessing so the label tensor is dense).
_NECROTIC_CORE = 1
_EDEMA = 2
_ENHANCING_TUMOR = 3


class ConvertToRegionsd(MapTransform):
    """Expand an integer BraTS label into the three nested BraTS regions.

    BraTS is scored on three OVERLAPPING regions, not the four raw classes:
    ET (enhancing tumor, class 3), TC (tumor core, classes 1 or 3), and WT
    (whole tumor, classes 1, 2 or 3). Because the regions overlap the model
    has 3 sigmoid output channels and per-channel Dice, not a 4-way softmax,
    so the label must become a `(3, D, H, W)` binary tensor before it can be
    compared against the model's output.

    IMPORTANT — do NOT replace this with MONAI's
    `ConvertToMultiChannelBasedOnBratsClassesd`. That transform is written for
    *raw* BraTS labels, whose enhancing-tumor voxels are value 4. This
    project's offline preprocessing (`neurovision.data.preprocessing`) already
    remaps labels to the contiguous set {0, 1, 2, 3} (raw 4 becomes 3). Fed a
    remapped label, MONAI's transform would look for voxels equal to 4, find
    none, and silently produce an all-zero ET channel: a model that never
    learns to predict enhancing tumor, with no error or warning anywhere.
    This class checks for value 3 instead, which is correct for our labels.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """Init the transform.

        Args:
            keys: Key(s) in the data dict holding the integer label, e.g.
                `["label"]`.
            allow_missing_keys: If True, skip keys that are absent from the
                data dict instead of raising.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        """Convert each labeled key from `(1, D, H, W)`/`(D, H, W)` classes to regions.

        Args:
            data: Mapping containing at least the configured keys, each an
                integer-valued array/tensor of shape `(1, D, H, W)` or
                `(D, H, W)` with values in `{0, 1, 2, 3}`.

        Returns:
            A shallow-copied dict where each configured key now maps to a
            float32 tensor of shape `(3, D, H, W)`, channel order
            `(ET, TC, WT)`, with binary 0.0/1.0 values.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            # `key_iterator` already skips missing keys when
            # allow_missing_keys=True, so no extra check is needed here.
            d[key] = self._convert(label)
        return d

    @staticmethod
    def _convert(label: Any) -> torch.Tensor:
        """Turn one integer label array into a `(3, D, H, W)` float32 region tensor."""
        # Accept both numpy arrays and tensor-likes (including MONAI's
        # MetaTensor, which is what LoadImaged actually returns in the
        # pipeline) by funneling everything through torch.as_tensor.
        label_t = torch.as_tensor(label)

        # Squeeze a leading singleton channel so both (1, D, H, W) and
        # (D, H, W) inputs land at the same (D, H, W) starting point.
        if label_t.ndim == 4:
            if label_t.shape[0] != 1:
                raise ValueError(
                    f"ConvertToRegionsd expects a single-channel label, got shape "
                    f"{tuple(label_t.shape)}."
                )
            label_t = label_t[0]
        elif label_t.ndim != 3:
            raise ValueError(
                f"ConvertToRegionsd expects a (1, D, H, W) or (D, H, W) label, got "
                f"shape {tuple(label_t.shape)}."
            )

        et = label_t == _ENHANCING_TUMOR
        tc = et | (label_t == _NECROTIC_CORE)
        wt = tc | (label_t == _EDEMA)

        # Stack in REGION_NAMES order (ET, TC, WT) -- this is the channel
        # order every downstream loss/metric/head assumes.
        regions = torch.stack([et, tc, wt], dim=0)
        return regions.to(dtype=torch.float32)


def build_train_transforms(cfg: Any) -> Compose:
    """Build the training-time MONAI transform pipeline.

    Reads `cfg.data.patch_size`, `cfg.data.pos_neg_ratio`,
    `cfg.data.samples_per_volume`, and `cfg.data.augment.*`.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A `Compose` that maps a `{"image": path, "label": path}` dict to a
        list of `cfg.data.samples_per_volume` cropped, augmented samples,
        each `{"image": (4, *patch_size), "label": (3, *patch_size)}`.
    """
    patch_size = cfg.data.patch_size
    augment = cfg.data.augment

    transforms = [
        # Items are dicts of .npy file paths written by scripts/preprocess.py.
        LoadImaged(keys=["image", "label"], reader="NumpyReader", image_only=True),
        # Asymmetric on purpose: the label on disk is (D, H, W) with no
        # channel axis and needs one added, but the image on disk is already
        # channel-first (4, D, H, W). Passing "image" here too would add a
        # second leading axis and produce (1, 4, D, H, W).
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        # Image is stored as float16 on disk to halve dataset size; training
        # (and augmentation math below) needs float32.
        EnsureTyped(keys=["image"], dtype=torch.float32),
        # Crop BEFORE converting the label to regions: RandCropByPosNegLabeld
        # samples foreground/background using the label's integer class
        # values (it looks for any nonzero voxel as "positive"), which only
        # makes sense while the label is still single-channel integers. Doing
        # this after ConvertToRegionsd would hand it a multi-channel binary
        # tensor it isn't designed to sample from.
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=cfg.data.pos_neg_ratio[0],
            neg=cfg.data.pos_neg_ratio[1],
            num_samples=cfg.data.samples_per_volume,
            image_key="image",
            # A few BraTS brains are smaller than 96 voxels on some axis
            # after foreground cropping in preprocessing; without
            # allow_smaller=True those cases raise instead of just cropping
            # to whatever extent is available.
            allow_smaller=True,
        ),
        # allow_smaller=True means the crop can come back smaller than
        # patch_size on some axis, so pad back up to the exact patch size
        # before the fixed-shape region conversion and augmentations run.
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ConvertToRegionsd(keys=["label"]),
        # Three independent per-axis flips, not one flip transform with a
        # spatial_axis list -- that would flip all three axes together
        # whenever the single probability check passed, instead of each axis
        # getting its own independent coin flip.
        RandFlipd(keys=["image", "label"], prob=augment.flip_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=augment.flip_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=augment.flip_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=augment.rot90_prob, spatial_axes=(0, 1)),
        # Intensity augmentations below apply to "image" only, never
        # "label". The label is a binary region mask; running intensity
        # jitter on it would silently corrupt the training target.
        RandScaleIntensityd(keys=["image"], factors=augment.scale_intensity_factor, prob=1.0),
        RandShiftIntensityd(keys=["image"], offsets=augment.shift_intensity_offset, prob=1.0),
        RandGaussianNoised(keys=["image"], prob=augment.noise_prob, std=augment.noise_std),
    ]
    return Compose(transforms)


def build_val_transforms(cfg: Any) -> Compose:
    """Build the deterministic validation/test-time MONAI transform pipeline.

    No cropping and no randomness: sliding-window inference does the
    patching at evaluation time, and the Dice/HD95 metrics should be measured
    against whole volumes, not training-style crops.

    Args:
        cfg: The full composed Hydra config. Not read from directly today
            (the val pipeline has no tunables), but kept as a parameter so
            the signature matches `build_train_transforms` and future
            val-only options (e.g. an intensity-only TTA flag) can be added
            without an API change.

    Returns:
        A `Compose` mapping a `{"image": path, "label": path}` dict to
        `{"image": (4, D, H, W) float32, "label": (3, D, H, W) float32}` for
        the whole volume.
    """
    transforms = [
        LoadImaged(keys=["image", "label"], reader="NumpyReader", image_only=True),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        ConvertToRegionsd(keys=["label"]),
    ]
    return Compose(transforms)
