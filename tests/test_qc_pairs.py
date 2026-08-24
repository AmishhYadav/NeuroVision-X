"""Tests for `neurovision.data.qc_pairs`.

Tiny synthetic (<=24^3) hand-built region masks, no real BraTS data. Every
random draw goes through an explicit `np.random.Generator` instance so tests
stay deterministic. The whole file runs on CPU in well under a second.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from neurovision.data.qc_pairs import (
    DEFAULT_SPECS,
    DEGRADATION_KINDS,
    DegradationSpec,
    degrade_mask,
    generate_one_pair,
    generate_pairs,
)
from neurovision.metrics.segmentation import REGION_NAMES, dice_score

_SHAPE = (24, 24, 24)


def _sphere(shape: tuple[int, int, int], center: tuple[int, int, int], radius: int) -> np.ndarray:
    """Builds a uint8 solid-sphere mask, shape `shape`."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return (dist2 <= radius**2).astype(np.uint8)


def _nested_case(
    shape: tuple[int, int, int] = _SHAPE,
    center: tuple[int, int, int] = (12, 12, 12),
    radii: tuple[int, int, int] = (2, 4, 6),
) -> np.ndarray:
    """Builds a (3, D, H, W) nested ET/TC/WT mask of concentric spheres."""
    et = _sphere(shape, center, radii[0])
    tc = _sphere(shape, center, radii[1])
    wt = _sphere(shape, center, radii[2])
    return np.stack([et, tc, wt], axis=0)


def _four_cluster_nested_case(shape: tuple[int, int, int] = _SHAPE) -> np.ndarray:
    """Four well-separated, equal-sized nested clusters (4 components per channel).

    Needed for the `drop_component` determinism tests: with `magnitude=0.5`
    this drops 2 of the 3 non-largest components, and there must be more
    than 2 non-largest candidates (3, here) for which 2 get chosen to
    actually depend on the generator's seed -- with only 1 or 2 candidates
    available, "choose N of them" has only one possible outcome regardless
    of seed.
    """
    centers = [(5, 5, 5), (5, 19, 19), (19, 5, 19), (19, 19, 5)]
    clusters = [_nested_case(shape, center=c, radii=(1, 2, 3)) for c in centers]
    total = np.clip(sum(clusters), 0, 1).astype(np.uint8)
    return total


def _dice(pred: np.ndarray, label: np.ndarray) -> tuple[float, ...]:
    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
    label_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)
    return tuple(float(v) for v in dice_score(pred_t, label_t, ignore_empty=False)[0].tolist())


# ---------------------------------------------------------------------------
# Structural guarantee: degrade_mask cannot see the label.
# ---------------------------------------------------------------------------


def test_degrade_mask_takes_no_label_argument() -> None:
    """degrade_mask's signature has no parameter that could smuggle a label in."""
    params = list(inspect.signature(degrade_mask).parameters)
    for name in params:
        lowered = name.lower()
        assert "label" not in lowered, params
        assert "target" not in lowered, params
        assert "gt" not in lowered, params


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_identity_returns_mask_unchanged_and_dice_matches_prediction() -> None:
    rng = np.random.default_rng(0)
    label = _nested_case()
    pred = label.copy()  # perfectly matched prediction

    out = degrade_mask(pred, DegradationSpec("identity", 0.0), generator=rng)

    assert np.array_equal(out, pred)
    assert _dice(out, label) == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# erode / dilate
# ---------------------------------------------------------------------------


def test_erode_lowers_dice_and_dilate_changes_it() -> None:
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    label = _nested_case()
    pred = label.copy()  # starts perfectly matched

    eroded = degrade_mask(pred, DegradationSpec("erode", 3.0), generator=rng_a)
    dilated = degrade_mask(pred, DegradationSpec("dilate", 3.0), generator=rng_b)

    dice_eroded = _dice(eroded, label)
    dice_dilated = _dice(dilated, label)
    original_dice = _dice(pred, label)

    # ET (radius 2) is fully consumed by 3 erosion iterations -> Dice drops.
    assert dice_eroded[0] < original_dice[0]
    assert dice_dilated[0] < original_dice[0]
    # Erosion and dilation damage the mask differently, so their resulting
    # Dice values differ (erosion empties ET entirely -> 0.0; dilation only
    # adds a false-positive shell around it -> some positive Dice).
    assert dice_eroded[0] != pytest.approx(dice_dilated[0])


# ---------------------------------------------------------------------------
# drop_component
# ---------------------------------------------------------------------------


def test_drop_component_removes_a_whole_lesion() -> None:
    rng = np.random.default_rng(3)
    tc = _sphere(_SHAPE, (5, 5, 5), 2) + _sphere(_SHAPE, (18, 18, 18), 2)
    mask = np.zeros((3, *_SHAPE), dtype=np.uint8)
    mask[1] = tc  # only TC has content; region_index=1 targets it directly

    before_components = _label_count(mask[1])
    assert before_components == 2

    out = degrade_mask(mask, DegradationSpec("drop_component", 0.5), generator=rng, region_index=1)

    after_components = _label_count(out[1])
    assert after_components == 1
    assert out[1].sum() < mask[1].sum()


def _label_count(channel: np.ndarray) -> int:
    from scipy import ndimage

    _, n = ndimage.label(channel)
    return int(n)


# ---------------------------------------------------------------------------
# shift
# ---------------------------------------------------------------------------


def test_shift_is_a_translation_not_a_reshape() -> None:
    rng = np.random.default_rng(4)
    mask = np.zeros((3, *_SHAPE), dtype=np.uint8)
    mask[0] = _sphere(_SHAPE, (12, 12, 12), 3)  # far from every boundary

    out = degrade_mask(mask, DegradationSpec("shift", 3.0), generator=rng, region_index=0)

    assert out[0].sum() == mask[0].sum()  # same voxel count, no clipping
    assert not np.array_equal(out[0], mask[0])  # but a different position


# ---------------------------------------------------------------------------
# speckle
# ---------------------------------------------------------------------------


def test_speckle_adds_isolated_voxels_outside_the_mask() -> None:
    rng = np.random.default_rng(5)
    mask = np.zeros((3, *_SHAPE), dtype=np.uint8)
    mask[2] = _sphere(_SHAPE, (12, 12, 12), 3)

    out = degrade_mask(mask, DegradationSpec("speckle", 0.5), generator=rng, region_index=2)

    assert out[2].sum() > mask[2].sum()
    # Speckle only ADDS voxels -- everything originally on stays on.
    assert np.all(out[2][mask[2] > 0] == 1)


# ---------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------


def test_nesting_is_preserved_when_all_regions_degraded_together() -> None:
    rng = np.random.default_rng(6)
    mask = _nested_case()

    out = degrade_mask(mask, DegradationSpec("drop_component", 0.5), generator=rng)

    et, tc, wt = out[0], out[1], out[2]
    assert np.all(tc[et > 0] == 1)  # ET subset-of TC
    assert np.all(wt[tc > 0] == 1)  # TC subset-of WT


def test_nesting_may_break_when_a_single_region_is_degraded() -> None:
    rng = np.random.default_rng(7)
    mask = _nested_case()

    # Dilate ET (radius 2) hard, while TC/WT (radius 4/6) stay untouched --
    # ET grows past TC's boundary, breaking the nesting invariant on purpose.
    out = degrade_mask(mask, DegradationSpec("dilate", 5.0), generator=rng, region_index=0)

    et, tc = out[0], mask[1]
    assert not np.all(tc[et > 0] == 1)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,magnitude", [("drop_component", 0.5), ("shift", 3.0), ("speckle", 0.5)]
)
def test_same_seed_gives_identical_output(kind: str, magnitude: float) -> None:
    mask = _four_cluster_nested_case()
    spec = DegradationSpec(kind, magnitude)

    out_a = degrade_mask(mask, spec, generator=np.random.default_rng(42))
    out_b = degrade_mask(mask, spec, generator=np.random.default_rng(42))

    assert np.array_equal(out_a, out_b)


@pytest.mark.parametrize(
    "kind,magnitude", [("drop_component", 0.5), ("shift", 3.0), ("speckle", 0.5)]
)
def test_different_seeds_differ(kind: str, magnitude: float) -> None:
    mask = _four_cluster_nested_case()
    spec = DegradationSpec(kind, magnitude)

    out_a = degrade_mask(mask, spec, generator=np.random.default_rng(1))
    out_b = degrade_mask(mask, spec, generator=np.random.default_rng(2))

    assert not np.array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# the Dice convention
# ---------------------------------------------------------------------------


def test_dice_target_uses_project_dice_convention() -> None:
    """Both pred and label empty in a region: naive Dice is 0/0 (undefined);
    the project convention (`dice_score(ignore_empty=False)`) says 1.0."""
    rng = np.random.default_rng(8)
    label = np.zeros((3, *_SHAPE), dtype=np.uint8)
    label[1] = _sphere(_SHAPE, (12, 12, 12), 4)
    label[2] = _sphere(_SHAPE, (12, 12, 12), 6)
    # ET is empty in both pred and label.
    pred = label.copy()

    pairs = generate_pairs(
        pred, label, generator=rng, specs=(DegradationSpec("identity", 0.0),), per_region=False
    )

    assert len(pairs) == 1
    et_index = REGION_NAMES.index("ET")
    assert pairs[0].dice[et_index] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# generate_pairs
# ---------------------------------------------------------------------------


def test_generate_pairs_spans_a_range_of_dice() -> None:
    rng = np.random.default_rng(9)
    # A perfectly-matched prediction: dice(pred, label) is already 1.0, the
    # global maximum a Dice score can take. This is the most robust way to
    # verify "degrading cannot beat the prediction's own Dice" -- with any
    # partially-overlapping pred/label pair, a large enough dilate/shift can
    # occasionally IMPROVE alignment by accident (pure geometry, nothing to
    # do with this module), so only a perfect starting point guarantees the
    # ceiling holds for every spec without relying on fragile geometry.
    label = _nested_case(center=(12, 12, 12), radii=(2, 4, 6))
    pred = label.copy()

    ceiling = _dice(pred, label)
    assert ceiling == pytest.approx((1.0, 1.0, 1.0))

    pairs = generate_pairs(pred, label, generator=rng, specs=DEFAULT_SPECS, per_region=False)

    assert len(pairs) == len(DEFAULT_SPECS)
    for pair in pairs:
        for i in range(3):
            assert pair.dice[i] <= ceiling[i] + 1e-6

    identity_pair = next(p for p in pairs if p.spec.kind == "identity")
    assert identity_pair.dice == pytest.approx(ceiling)

    wt_index = REGION_NAMES.index("WT")
    wt_values = [p.dice[wt_index] for p in pairs]
    assert max(wt_values) - min(wt_values) > 0.1  # a meaningful spread


def test_generate_pairs_per_region_multiplies_pair_count() -> None:
    rng = np.random.default_rng(10)
    label = _nested_case()
    pred = label.copy()
    specs = (DegradationSpec("erode", 2.0),)

    pairs = generate_pairs(pred, label, generator=rng, specs=specs, per_region=True)

    assert len(pairs) == len(specs) * 4  # 1 all-regions pair + 3 per-region pairs
    assert sum(1 for p in pairs if p.region_index is None) == 1
    assert sum(1 for p in pairs if p.region_index == 0) == 1
    assert sum(1 for p in pairs if p.region_index == 1) == 1
    assert sum(1 for p in pairs if p.region_index == 2) == 1


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,magnitude",
    [
        ("bogus_kind", 0.0),
        ("identity", 1.0),
        ("drop_component", 0.0),
        ("drop_component", 1.5),
        ("erode", 0.0),
        ("erode", -1.0),
        ("dilate", 0.0),
        ("shift", 0.0),
        ("speckle", 0.0),
        ("erode", float("nan")),
    ],
)
def test_invalid_spec_raises(kind: str, magnitude: float) -> None:
    with pytest.raises(ValueError):
        DegradationSpec(kind, magnitude)


def test_valid_kinds_cover_declared_degradation_kinds() -> None:
    for kind in DEGRADATION_KINDS:
        magnitude = 0.0 if kind == "identity" else (0.5 if kind == "drop_component" else 1.0)
        DegradationSpec(kind, magnitude)  # must not raise


# ---------------------------------------------------------------------------
# label input flexibility
# ---------------------------------------------------------------------------


def test_label_accepts_class_map_or_region_stack() -> None:
    class_map = np.zeros(_SHAPE, dtype=np.int64)
    core = _sphere(_SHAPE, (12, 12, 12), 4)
    enhancing = _sphere(_SHAPE, (12, 12, 12), 2)
    edema = _sphere(_SHAPE, (12, 12, 12), 6)
    class_map[edema > 0] = 2  # edema
    class_map[core > 0] = 1  # necrotic/non-enhancing core
    class_map[enhancing > 0] = 3  # enhancing tumor, innermost, overwrites

    region_stack = _nested_case(center=(12, 12, 12), radii=(2, 4, 6))
    pred = region_stack.copy()

    pairs_from_classmap = generate_pairs(
        pred, class_map, generator=np.random.default_rng(11), specs=DEFAULT_SPECS[:3]
    )
    pairs_from_regions = generate_pairs(
        pred, region_stack, generator=np.random.default_rng(11), specs=DEFAULT_SPECS[:3]
    )

    assert len(pairs_from_classmap) == len(pairs_from_regions)
    for a, b in zip(pairs_from_classmap, pairs_from_regions, strict=True):
        assert np.array_equal(a.mask, b.mask)
        assert a.dice == pytest.approx(b.dice)


# ---------------------------------------------------------------------------
# generate_one_pair -- must be bit-for-bit identical to generate_pairs, for
# every region, including the STOCHASTIC spec kinds where this is only true
# if the generator's random draws are consumed in the same order.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("region_index", [0, 1, 2])
@pytest.mark.parametrize(
    "kind,magnitude",
    [
        ("identity", 0.0),
        ("erode", 2.0),
        ("dilate", 2.0),
        ("drop_component", 0.5),
        ("shift", 3.0),
        ("speckle", 0.5),
    ],
)
def test_generate_one_pair_matches_generate_pairs_exactly(
    kind: str, magnitude: float, region_index: int
) -> None:
    """For every degradation kind (stochastic or not) and every region, the fast
    single-pair path must return exactly what generate_pairs would have, at the
    same generator index -- proving generate_one_pair advances `generator`
    identically to generate_pairs even though it skips scoring 3 of the 4 pairs
    generate_pairs would compute."""
    mask = _four_cluster_nested_case()
    label = mask.copy()
    spec = DegradationSpec(kind, magnitude)

    full_pairs = generate_pairs(
        mask, label, generator=np.random.default_rng(123), specs=[spec], per_region=True
    )
    expected = next(p for p in full_pairs if p.region_index == region_index)

    got = generate_one_pair(mask, label, spec, region_index, generator=np.random.default_rng(123))

    assert np.array_equal(got.mask, expected.mask)
    assert got.dice == pytest.approx(expected.dice)
    assert got.spec == expected.spec
    assert got.region_index == expected.region_index


def test_generate_one_pair_rejects_none_region_index() -> None:
    mask = _nested_case()
    with pytest.raises(ValueError, match="region_index"):
        generate_one_pair(
            mask,
            mask,
            DegradationSpec("identity", 0.0),
            None,  # type: ignore[arg-type]
            generator=np.random.default_rng(0),
        )
