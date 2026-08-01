"""Tests for `neurovision.inference.postprocess`.

Everything here runs on tiny (<=16^3), hand-built tensors with exact,
hand-computed expected voxel counts -- no real BraTS data, no randomness
where an exact expectation is possible. The whole file must run on CPU in
under ~2 seconds.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch

from neurovision.inference.postprocess import (
    enforce_nesting,
    keep_largest_component,
    postprocess_logits,
    regions_to_classes,
    remove_small_components,
    uncrop_to_original,
)
from neurovision.metrics.segmentation import classes_to_regions

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


# ---------------------------------------------------------------------------
# enforce_nesting
# ---------------------------------------------------------------------------


def test_enforce_nesting_unions_upward_with_exact_counts() -> None:
    """A violating ET voxel (outside TC/WT) gets unioned into both outer regions."""
    et = torch.zeros(4, 4, 4)
    tc = torch.zeros(4, 4, 4)
    wt = torch.zeros(4, 4, 4)

    et[0, 0, 0] = 1.0  # violates nesting: not present in tc or wt
    tc[1, 1, 1] = 1.0  # tc's own, independent voxel
    wt[2, 2, 2] = 1.0  # wt's own, independent voxel

    regions = torch.stack([et, tc, wt], dim=0)  # (3, 4, 4, 4)
    out = enforce_nesting(regions)

    assert out.shape == regions.shape
    # ET is never modified -- union only ever ADDS to the outer regions.
    assert out[0].sum().item() == 1
    # TC = its own voxel + ET's voxel unioned in.
    assert out[1].sum().item() == 2
    assert out[1, 0, 0, 0].item() == 1.0
    assert out[1, 1, 1, 1].item() == 1.0
    # WT = its own voxel + the (already-updated) TC voxels unioned in.
    assert out[2].sum().item() == 3
    assert out[2, 0, 0, 0].item() == 1.0
    assert out[2, 1, 1, 1].item() == 1.0
    assert out[2, 2, 2, 2].item() == 1.0


def test_enforce_nesting_is_idempotent_and_noop_on_nested_input() -> None:
    """Already-nested input is unchanged by one or two applications."""
    et = torch.zeros(4, 4, 4)
    et[0, 0, 0] = 1.0
    tc = et.clone()
    tc[1, 1, 1] = 1.0
    wt = tc.clone()
    wt[2, 2, 2] = 1.0
    regions = torch.stack([et, tc, wt], dim=0)

    once = enforce_nesting(regions)
    twice = enforce_nesting(once)

    assert torch.equal(once, regions)
    assert torch.equal(twice, once)


# ---------------------------------------------------------------------------
# regions_to_classes
# ---------------------------------------------------------------------------


def _build_nested_cubes(size: int = 8) -> tuple[torch.Tensor, dict[str, int]]:
    """Builds a hand-computable nested (ET, TC, WT) region tensor.

    WT is a solid (size-2)^3 cube centered in a `size`^3 volume, TC a
    concentric (size-4)^3 cube, ET a concentric (size-6)^3 cube -- so
    ET subset-of TC subset-of WT by construction, with exactly countable
    voxel totals.
    """
    assert size >= 8 and size % 2 == 0
    wt = torch.zeros(size, size, size)
    tc = torch.zeros(size, size, size)
    et = torch.zeros(size, size, size)

    wt[1 : size - 1, 1 : size - 1, 1 : size - 1] = 1.0  # (size-2)^3
    tc[2 : size - 2, 2 : size - 2, 2 : size - 2] = 1.0  # (size-4)^3
    et[3 : size - 3, 3 : size - 3, 3 : size - 3] = 1.0  # (size-6)^3

    wt_count = (size - 2) ** 3
    tc_count = (size - 4) ** 3
    et_count = (size - 6) ** 3
    counts = {
        "total": size**3,
        "wt": wt_count,
        "tc": tc_count,
        "et": et_count,
        "background": size**3 - wt_count,
        "edema": wt_count - tc_count,
        "necrotic": tc_count - et_count,
        "et_class": et_count,
    }
    regions = torch.stack([et, tc, wt], dim=0)
    return regions, counts


def test_regions_to_classes_hand_built_nested_cubes() -> None:
    """Per-class voxel counts match exactly the hand-computed cube volumes."""
    regions, counts = _build_nested_cubes(size=8)
    classes = regions_to_classes(regions)

    assert classes.shape == (8, 8, 8)
    assert classes.dtype == torch.uint8
    assert (classes == 0).sum().item() == counts["background"]
    assert (classes == 1).sum().item() == counts["necrotic"]
    assert (classes == 2).sum().item() == counts["edema"]
    assert (classes == 3).sum().item() == counts["et_class"]


def test_regions_to_classes_assignment_order_et_wins_over_wt() -> None:
    """An ET voxel that is also inside WT (but not TC) must classify as 3, not 2.

    This is the test that catches a reversed inner/outer assignment loop: if
    the loop ran inner-to-outer instead of outer-to-inner, WT would be
    written last and would overwrite this voxel back to 2.
    """
    et = torch.zeros(4, 4, 4)
    tc = torch.zeros(4, 4, 4)
    wt = torch.zeros(4, 4, 4)

    et[1, 1, 1] = 1.0  # ET voxel...
    wt[1, 1, 1] = 1.0  # ...that is also inside WT, but NOT inside TC.

    regions = torch.stack([et, tc, wt], dim=0)
    classes = regions_to_classes(regions)

    assert classes[1, 1, 1].item() == 3


# ---------------------------------------------------------------------------
# Round trip against neurovision.metrics.segmentation.classes_to_regions
# ---------------------------------------------------------------------------


def test_round_trip_regions_to_classes_to_regions() -> None:
    """classes_to_regions(regions_to_classes(r)) == r for nested r."""
    regions, _ = _build_nested_cubes(size=8)
    batched_regions = regions.unsqueeze(0)  # (1, 3, 8, 8, 8)

    classes = regions_to_classes(batched_regions)  # (1, 8, 8, 8)
    round_tripped = classes_to_regions(classes)  # (1, 3, 8, 8, 8)

    assert torch.equal(round_tripped, batched_regions)


def test_round_trip_classes_to_regions_to_classes() -> None:
    """regions_to_classes(classes_to_regions(c)) == c for a class map c."""
    size = 8
    classes = torch.zeros(size, size, size, dtype=torch.uint8)
    classes[1 : size - 1, 1 : size - 1, 1 : size - 1] = 2  # edema (WT extent)
    classes[2 : size - 2, 2 : size - 2, 2 : size - 2] = 1  # necrotic core (TC extent)
    classes[3 : size - 3, 3 : size - 3, 3 : size - 3] = 3  # enhancing tumor (ET, innermost)

    regions = classes_to_regions(classes)  # (1, 3, 8, 8, 8), batch dim added
    round_tripped = regions_to_classes(regions)  # (1, 8, 8, 8)

    assert torch.equal(round_tripped, classes.unsqueeze(0).to(torch.uint8))


# ---------------------------------------------------------------------------
# remove_small_components
# ---------------------------------------------------------------------------


def test_remove_small_components_drops_small_keeps_large() -> None:
    """A 3-voxel blob is removed and a 30-voxel blob is kept at min_size=10."""
    et = torch.zeros(10, 10, 10)
    et[0, 0, 0:3] = 1.0  # 3-voxel line, one connected component
    et[5:8, 5:8, 5:8] = 1.0  # a solid 3x3x3 cube = 27 voxels
    et[5, 5, 8] = 1.0  # + 3 more face-adjacent voxels to reach 30, connected
    et[5, 6, 8] = 1.0
    et[5, 7, 8] = 1.0

    assert et.sum().item() == 33  # 3 (small blob) + 30 (large blob)
    regions = torch.stack([et, torch.zeros(10, 10, 10), torch.zeros(10, 10, 10)], dim=0)

    out = remove_small_components(regions, min_size=10, connectivity=1)

    assert out[0].sum().item() == 30  # only the large blob survives
    assert out[0, 0, 0, 0].item() == 0.0  # small blob's voxels are gone
    assert out[0, 5, 5, 5].item() == 1.0  # large blob's voxels remain


def test_remove_small_components_min_size_zero_is_noop() -> None:
    """min_size <= 0 returns the input completely unchanged, without raising."""
    regions = torch.zeros(3, 6, 6, 6)
    regions[0, 0, 0, 0] = 1.0  # a single isolated voxel, would be removed by any real filter

    out = remove_small_components(regions, min_size=0, connectivity=1)
    assert torch.equal(out, regions)

    out_negative = remove_small_components(regions, min_size=-5, connectivity=1)
    assert torch.equal(out_negative, regions)


def test_remove_small_components_can_leave_nesting_broken_then_repaired() -> None:
    """Documents, executably, why `enforce_nesting` must run after filtering.

    Raw thresholded model output is never guaranteed nested to begin with
    (ET/TC/WT come from three independent sigmoid channels). Here ET has a
    real, surviving blob at a location TC does not cover at all -- already a
    nesting violation before any filtering runs. `remove_small_components`
    filters each channel independently and has no notion of the other
    channels, so it does not repair this: the violation survives filtering
    untouched. Only `enforce_nesting`, run afterward, fixes it.
    """
    et = torch.zeros(10, 10, 10)
    et[2:5, 2:5, 2:5] = 1.0  # 27-voxel cube, well above any min_size used below

    tc = torch.zeros(10, 10, 10)
    tc[8, 8, 8] = 1.0  # a tiny, unrelated 1-voxel blob elsewhere

    wt = tc.clone()

    regions = torch.stack([et, tc, wt], dim=0)

    def _is_nested(r: torch.Tensor) -> bool:
        return bool((r[0] <= r[1]).all()) and bool((r[1] <= r[2]).all())

    assert not _is_nested(regions), "test setup should start already non-nested"

    filtered = remove_small_components(regions, min_size=10, connectivity=1)
    # TC's tiny blob is gone (1 voxel < 10), ET's big blob is untouched --
    # still not nested, because remove_small_components never looks across
    # channels.
    assert filtered[1].sum().item() == 0
    assert filtered[0].sum().item() == 27
    assert not _is_nested(filtered), "remove_small_components does not repair nesting"

    repaired = enforce_nesting(filtered)
    assert _is_nested(repaired), "enforce_nesting must repair it afterward"
    assert repaired[1].sum().item() == 27  # TC now includes ET's surviving blob
    assert repaired[2].sum().item() == 27


# ---------------------------------------------------------------------------
# keep_largest_component
# ---------------------------------------------------------------------------


def test_keep_largest_component_keeps_only_the_bigger_blob() -> None:
    """Of two disjoint blobs in one channel, only the larger one survives."""
    et = torch.zeros(10, 10, 10)
    et[0, 0, 0:2] = 1.0  # 2-voxel blob
    et[5:8, 5:8, 5:8] = 1.0  # 27-voxel blob

    regions = torch.stack([et, torch.zeros(10, 10, 10), torch.zeros(10, 10, 10)], dim=0)
    out = keep_largest_component(regions)

    assert out[0].sum().item() == 27
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 5, 5, 5].item() == 1.0


def test_keep_largest_component_empty_channel_stays_empty() -> None:
    """An all-zero channel must return all-zero, not raise."""
    regions = torch.zeros(3, 6, 6, 6)
    out = keep_largest_component(regions)
    assert out.sum().item() == 0
    assert out.shape == regions.shape


# ---------------------------------------------------------------------------
# uncrop_to_original
# ---------------------------------------------------------------------------


def test_uncrop_to_original_places_content_at_exact_offset() -> None:
    """A (4,4,4) array lands at the right offset in a (10,10,10) volume."""
    array = np.arange(1, 4 * 4 * 4 + 1, dtype=np.float32).reshape(4, 4, 4)
    bbox = [[2, 6], [3, 7], [1, 5]]
    original_shape = [10, 10, 10]

    out = uncrop_to_original(array, bbox, original_shape)

    assert out.shape == (10, 10, 10)
    assert np.array_equal(out[2:6, 3:7, 1:5], array)
    assert out.sum() == array.sum()
    # Nothing placed outside the bbox.
    out_copy = out.copy()
    out_copy[2:6, 3:7, 1:5] = 0
    assert out_copy.sum() == 0


def test_uncrop_to_original_handles_channel_first_array() -> None:
    """A (C, D, H, W) region-mask array is un-cropped per channel."""
    array = np.ones((3, 4, 4, 4), dtype=np.uint8)
    bbox = [[2, 6], [3, 7], [1, 5]]
    original_shape = [10, 10, 10]

    out = uncrop_to_original(array, bbox, original_shape)

    assert out.shape == (3, 10, 10, 10)
    assert np.array_equal(out[:, 2:6, 3:7, 1:5], array)
    assert out.sum() == array.sum()


def test_uncrop_to_original_raises_on_bbox_shape_mismatch() -> None:
    """A bbox extent that disagrees with the array's spatial shape must raise."""
    array = np.ones((4, 4, 4), dtype=np.float32)
    # Extent here is (3, 4, 4), not (4, 4, 4) -- disagrees with array.shape.
    bad_bbox = [[2, 5], [3, 7], [1, 5]]
    with pytest.raises(ValueError):
        uncrop_to_original(array, bad_bbox, [10, 10, 10])


def test_uncrop_to_original_preserves_dtype() -> None:
    """uint8 in, uint8 out."""
    array = np.ones((4, 4, 4), dtype=np.uint8)
    bbox = [[0, 4], [0, 4], [0, 4]]
    out = uncrop_to_original(array, bbox, [10, 10, 10])
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# postprocess_logits, end-to-end against the real composed Hydra config
# ---------------------------------------------------------------------------


def _compose_real_config(tmp_path: Path):
    """Composes the real configs/ tree with the one mandatory override."""
    overrides = [f"data.root_dir={tmp_path}"]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        return hydra.compose(config_name="config", overrides=overrides)


def test_postprocess_logits_end_to_end_binary_nested_and_shaped(tmp_path: Path) -> None:
    """Output of the full pipeline is binary, nested, and the right shape."""
    cfg = _compose_real_config(tmp_path)

    torch.manual_seed(0)
    logits = torch.randn(2, 3, 12, 12, 12)

    out = postprocess_logits(logits, cfg)

    assert out.shape == logits.shape
    unique_vals = set(torch.unique(out).tolist())
    assert unique_vals <= {0.0, 1.0}
    assert bool((out[:, 0] <= out[:, 1]).all())  # ET <= TC
    assert bool((out[:, 1] <= out[:, 2]).all())  # TC <= WT


# ---------------------------------------------------------------------------
# et_min_volume
# ---------------------------------------------------------------------------


def test_et_min_volume_zeroes_small_et_per_batch_element(tmp_path: Path) -> None:
    """et_min_volume zeroes a small ET and leaves a large one, per batch element."""
    cfg = _compose_real_config(tmp_path)
    cfg.inference.postprocess.et_min_volume = 20
    cfg.inference.postprocess.min_component_size = 0  # isolate the et_min_volume effect
    cfg.inference.postprocess.enforce_nesting = False  # isolate the et_min_volume effect

    logits = torch.full((2, 3, 8, 8, 8), -10.0)  # sigmoid ~ 0 everywhere by default

    # Batch element 0: a small ET blob (5 voxels), below the et_min_volume=20 floor.
    logits[0, 0, 0:5, 0, 0] = 10.0
    # Batch element 1: a large ET blob (30 voxels), at/above the floor.
    logits[1, 0, 0:6, 0:5, 0] = 10.0  # 6*5 = 30 voxels

    out = postprocess_logits(logits, cfg)

    assert out[0, 0].sum().item() == 0  # small ET zeroed for case 0
    assert out[1, 0].sum().item() == 30  # large ET untouched for case 1
