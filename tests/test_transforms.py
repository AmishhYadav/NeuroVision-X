"""Tests for neurovision.data.transforms.

All synthetic volumes are tiny (32^3 or smaller) numpy arrays written to
`tmp_path` as `.npy` files -- the same format scripts/preprocess.py writes
-- never real BraTS data, so the whole suite stays well under a second. See
CLAUDE.md for the project's testing rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from monai.transforms import Compose
from monai.utils import set_determinism
from omegaconf import OmegaConf

from neurovision.data.transforms import (
    REGION_NAMES,
    ConvertToRegionsd,
    build_train_transforms,
    build_val_transforms,
)

# --- helpers -----------------------------------------------------------


def _make_label(shape: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    """A small uint8 label with a few voxels of each foreground class."""
    label = np.zeros(shape, dtype=np.uint8)
    label[5:8, 5:8, 5:8] = 1  # necrotic/non-enhancing core -> in TC and WT
    label[10:13, 10:13, 10:13] = 2  # edema -> in WT only
    label[20:23, 20:23, 20:23] = 3  # enhancing tumor -> in ET, TC and WT
    return label


def _write_case(tmp_path: Path, label: np.ndarray | None = None) -> dict[str, str]:
    """Write a synthetic case (image + label .npy) and return the data dict."""
    if label is None:
        label = _make_label()
    image = np.random.rand(4, *label.shape).astype(np.float16)
    image_path = tmp_path / "image.npy"
    label_path = tmp_path / "label.npy"
    np.save(image_path, image)
    np.save(label_path, label)
    return {"image": str(image_path), "label": str(label_path)}


def _make_cfg(patch_size: tuple[int, int, int] = (16, 16, 16)) -> OmegaConf:
    """A minimal cfg matching the real configs/data/brats.yaml key structure."""
    return OmegaConf.create(
        {
            "data": {
                "regions": list(REGION_NAMES),
                "patch_size": list(patch_size),
                "pos_neg_ratio": [1, 1],
                "samples_per_volume": 4,
                "augment": {
                    "flip_prob": 0.5,
                    "rot90_prob": 0.5,
                    "scale_intensity_factor": 0.1,
                    "shift_intensity_offset": 0.1,
                    "noise_prob": 0.15,
                    "noise_std": 0.01,
                },
            }
        }
    )


# --- ConvertToRegionsd ---------------------------------------------------


def test_convert_to_regions_voxel_counts_match_hand_computed():
    label = _make_label()
    out = ConvertToRegionsd(keys=["label"])({"label": torch.as_tensor(label)})
    regions = out["label"]

    et, tc, wt = regions[0], regions[1], regions[2]
    assert et.sum().item() == 27  # only the 3x3x3 class-3 block
    assert tc.sum().item() == 27 + 27  # class 1 block + class 3 block
    assert wt.sum().item() == 27 * 3  # classes 1, 2, and 3 blocks


def test_convert_to_regions_et_nonzero_when_class_3_present():
    # Regression test for the MONAI-transform trap: our preprocessing remaps
    # raw BraTS enhancing-tumor voxels (originally value 4) down to value 3.
    # MONAI's ConvertToMultiChannelBasedOnBratsClassesd checks for value 4
    # and would find none here, silently producing an all-zero ET channel.
    # This assertion fails immediately if that transform is swapped in.
    label = _make_label()
    out = ConvertToRegionsd(keys=["label"])({"label": torch.as_tensor(label)})
    et = out["label"][0]
    assert et.sum().item() > 0


def test_convert_to_regions_handles_channel_first_input():
    label = _make_label()
    label_1chw = torch.as_tensor(label).unsqueeze(0)  # (1, D, H, W)
    out = ConvertToRegionsd(keys=["label"])({"label": label_1chw})
    assert out["label"].shape == (3, *label.shape)


def test_convert_to_regions_handles_no_channel_input():
    label = _make_label()
    label_dhw = torch.as_tensor(label)  # (D, H, W)
    out = ConvertToRegionsd(keys=["label"])({"label": label_dhw})
    assert out["label"].shape == (3, *label.shape)


def test_convert_to_regions_output_dtype_and_binary_values():
    label = _make_label()
    out = ConvertToRegionsd(keys=["label"])({"label": torch.as_tensor(label)})
    regions = out["label"]
    assert regions.dtype == torch.float32
    uniques = torch.unique(regions)
    assert torch.all((uniques == 0.0) | (uniques == 1.0))


def test_convert_to_regions_nesting_wt_ge_tc_ge_et():
    label = _make_label()
    out = ConvertToRegionsd(keys=["label"])({"label": torch.as_tensor(label)})
    et, tc, wt = out["label"][0], out["label"][1], out["label"][2]
    assert torch.all(wt >= tc)
    assert torch.all(tc >= et)


def test_convert_to_regions_all_background_gives_all_zero_channels():
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    out = ConvertToRegionsd(keys=["label"])({"label": torch.as_tensor(label)})
    regions = out["label"]
    assert regions.shape == (3, 16, 16, 16)
    assert torch.all(regions == 0.0)


def test_convert_to_regions_raises_on_bad_channel_count():
    bad = torch.zeros(2, 4, 4, 4)  # 2 channels is not a valid single-channel label
    with pytest.raises(ValueError):
        ConvertToRegionsd(keys=["label"])({"label": bad})


# --- build_train_transforms ---------------------------------------------


def test_build_train_transforms_returns_compose():
    cfg = _make_cfg()
    assert isinstance(build_train_transforms(cfg), Compose)


def test_build_train_transforms_returns_list_of_samples(tmp_path: Path):
    cfg = _make_cfg(patch_size=(16, 16, 16))
    data = _write_case(tmp_path)
    out = build_train_transforms(cfg)(data)
    assert isinstance(out, list)
    assert len(out) == cfg.data.samples_per_volume


def test_build_train_transforms_sample_shapes(tmp_path: Path):
    cfg = _make_cfg(patch_size=(16, 16, 16))
    data = _write_case(tmp_path)
    out = build_train_transforms(cfg)(data)
    for sample in out:
        assert sample["image"].shape == (4, 16, 16, 16)
        assert sample["label"].shape == (3, 16, 16, 16)


def test_build_train_transforms_image_dtype_is_float32(tmp_path: Path):
    # Confirms the float16 (on-disk) -> float32 (training) cast happened.
    cfg = _make_cfg(patch_size=(16, 16, 16))
    data = _write_case(tmp_path)
    out = build_train_transforms(cfg)(data)
    for sample in out:
        assert sample["image"].dtype == torch.float32


def test_build_train_transforms_label_only_binary_after_augmentation(tmp_path: Path):
    # Intensity augmentations must never touch the label -- if they did, the
    # label would no longer be a clean 0.0/1.0 binary mask.
    cfg = _make_cfg(patch_size=(16, 16, 16))
    data = _write_case(tmp_path)
    out = build_train_transforms(cfg)(data)
    for sample in out:
        uniques = torch.unique(sample["label"])
        assert torch.all((uniques == 0.0) | (uniques == 1.0))


def test_build_train_transforms_is_stochastic(tmp_path: Path):
    cfg = _make_cfg(patch_size=(16, 16, 16))
    data = _write_case(tmp_path)
    train_tf = build_train_transforms(cfg)

    set_determinism(seed=0)
    out_a = train_tf(data)
    set_determinism(seed=1)
    out_b = train_tf(data)

    # Compare several samples' images; guard against flakiness by requiring
    # at least one sample pair to differ rather than all of them.
    any_different = any(
        not torch.equal(a["image"], b["image"]) for a, b in zip(out_a, out_b, strict=True)
    )
    assert any_different
    set_determinism(seed=None)


# --- build_val_transforms -------------------------------------------------


def test_build_val_transforms_returns_compose():
    cfg = _make_cfg()
    assert isinstance(build_val_transforms(cfg), Compose)


def test_build_val_transforms_full_volume_shapes(tmp_path: Path):
    cfg = _make_cfg()
    data = _write_case(tmp_path)  # 32^3 volume, no cropping expected
    out = build_val_transforms(cfg)(data)
    assert out["image"].shape == (4, 32, 32, 32)
    assert out["label"].shape == (3, 32, 32, 32)


def test_build_val_transforms_is_deterministic(tmp_path: Path):
    cfg = _make_cfg()
    data = _write_case(tmp_path)
    val_tf = build_val_transforms(cfg)
    out_a = val_tf(data)
    out_b = val_tf(data)
    assert torch.equal(torch.as_tensor(out_a["image"]), torch.as_tensor(out_b["image"]))
    assert torch.equal(torch.as_tensor(out_a["label"]), torch.as_tensor(out_b["label"]))
