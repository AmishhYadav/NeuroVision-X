"""Tests for neurovision.data.preprocessing.

All synthetic volumes are built inline with numpy + nibabel and saved under
pytest's `tmp_path` -- never real BraTS data -- and kept tiny (16x16x16) so
the whole suite stays well under a second. See CLAUDE.md for the project's
testing rules.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from neurovision.data.brats import BratsCase
from neurovision.data.preprocessing import (
    BRATS_LABEL_MAP,
    compute_nonzero_bbox,
    crop_to_bbox,
    is_case_processed,
    load_case_arrays,
    normalize_nonzero,
    preprocess_case,
    remap_labels,
    reorient_to_axcodes,
)

# --- helpers -----------------------------------------------------------

_SHAPE = (16, 16, 16)
_BRAIN = (slice(2, 10), slice(3, 11), slice(4, 12))  # an 8x8x8 known cuboid


def _write_nifti(path: Path, arr: np.ndarray, affine: np.ndarray, zooms) -> None:
    img = nib.Nifti1Image(arr, affine)
    img.header.set_zooms(zooms)
    nib.save(img, str(path))


def _make_case(
    tmp_path: Path,
    case_id: str = "BraTS20_Training_001",
    modality_arrays: dict[str, np.ndarray] | None = None,
    seg_array: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    shapes: dict[str, tuple[int, ...]] | None = None,
) -> BratsCase:
    """Build a synthetic 2020-style BraTS case directory and return it.

    `modality_arrays` defaults to a brain-only-nonzero volume of `_SHAPE`
    for every modality if not given. `shapes` allows overriding individual
    modality shapes (used to test the mismatch error).
    """
    case_dir = tmp_path / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)

    if modality_arrays is None:
        # Non-constant brain values: a constant fill has zero std, which
        # normalize_nonzero deliberately zeros out (see its own tests) --
        # unrealistic for real MRI and not what these tests want to probe.
        rng = np.random.default_rng(42)
        base = np.zeros(_SHAPE, dtype=np.float32)
        base[_BRAIN] = rng.normal(loc=100.0, scale=10.0, size=(8, 8, 8)).astype(np.float32)
        modality_arrays = {role: base.copy() for role in ("t1", "t1ce", "t2", "flair")}

    suffixes = {"t1": "_t1", "t1ce": "_t1ce", "t2": "_t2", "flair": "_flair"}
    paths: dict[str, Path] = {}
    for role, suffix in suffixes.items():
        arr = modality_arrays[role]
        if shapes is not None and role in shapes:
            arr = np.zeros(shapes[role], dtype=np.float32)
        p = case_dir / f"{case_id}{suffix}.nii.gz"
        _write_nifti(p, arr.astype(np.float32), affine, zooms)
        paths[role] = p

    seg_path = None
    if seg_array is not None:
        seg_path = case_dir / f"{case_id}_seg.nii.gz"
        _write_nifti(seg_path, seg_array.astype(np.uint8), affine, zooms)

    return BratsCase(
        case_id=case_id,
        t1=paths["t1"],
        t1ce=paths["t1ce"],
        t2=paths["t2"],
        flair=paths["flair"],
        seg=seg_path,
    )


def _known_label() -> np.ndarray:
    """A label volume with a hand-countable mix of raw BraTS values."""
    seg = np.zeros(_SHAPE, dtype=np.uint8)
    seg[_BRAIN] = 1  # 512 voxels of NCR/NET
    seg[2:6, 3:11, 4:12] = 2  # overwrite a 4x8x8=256 sub-block -> ED
    seg[2:4, 3:11, 4:12] = 4  # overwrite a 2x8x8=128 sub-block -> ET
    return seg


# --- normalize_nonzero ---------------------------------------------------


def test_normalize_nonzero_brain_has_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    volume = np.zeros(_SHAPE, dtype=np.float32)
    volume[_BRAIN] = rng.normal(loc=500.0, scale=50.0, size=(8, 8, 8)).astype(np.float32)

    out = normalize_nonzero(volume)

    brain_values = out[_BRAIN]
    assert brain_values.mean() == pytest.approx(0.0, abs=1e-4)
    assert brain_values.std() == pytest.approx(1.0, abs=1e-4)


def test_normalize_nonzero_background_stays_exactly_zero():
    volume = np.zeros(_SHAPE, dtype=np.float32)
    volume[_BRAIN] = 42.0
    out = normalize_nonzero(volume)
    background_mask = np.ones(_SHAPE, dtype=bool)
    background_mask[_BRAIN] = False
    assert np.all(out[background_mask] == 0.0)


def test_normalize_nonzero_constant_volume_returns_zeros_no_raise():
    volume = np.zeros(_SHAPE, dtype=np.float32)
    volume[_BRAIN] = 7.0  # constant nonzero value -> std == 0
    out = normalize_nonzero(volume)
    assert np.all(out == 0.0)


def test_normalize_nonzero_all_zero_volume_returns_zeros():
    volume = np.zeros(_SHAPE, dtype=np.float32)
    out = normalize_nonzero(volume)
    assert np.all(out == 0.0)


def test_normalize_nonzero_ignores_background_in_statistics():
    # If background zeros were included in the mean/std, the brain values
    # would come out shifted away from zero-mean/unit-std once padded into
    # a much larger array of zeros.
    small_brain = np.full((8, 8, 8), 10.0, dtype=np.float32)
    small_brain += np.array([-1.0, 1.0] * 256, dtype=np.float32).reshape(8, 8, 8)
    volume = np.zeros(_SHAPE, dtype=np.float32)
    volume[_BRAIN] = small_brain

    out = normalize_nonzero(volume)

    expected_mean = small_brain.mean()
    expected_std = small_brain.std()
    expected = (small_brain - expected_mean) / expected_std
    np.testing.assert_allclose(out[_BRAIN], expected, atol=1e-5)

    # A naive whole-array normalization would give a visibly different
    # result because it would include ~93% zeros in its statistics.
    naive_mean = volume.mean()
    naive_std = volume.std()
    naive = (volume - naive_mean) / naive_std
    assert not np.allclose(out[_BRAIN], naive[_BRAIN], atol=1e-3)


def test_normalize_nonzero_returns_float32():
    volume = np.zeros(_SHAPE, dtype=np.float64)
    volume[_BRAIN] = 3.0
    out = normalize_nonzero(volume)
    assert out.dtype == np.float32


# --- compute_nonzero_bbox -------------------------------------------------


def test_compute_nonzero_bbox_known_cuboid():
    image = np.zeros((1, *_SHAPE), dtype=np.float32)
    image[0][_BRAIN] = 1.0
    bbox = compute_nonzero_bbox(image)
    assert bbox == ((2, 10), (3, 11), (4, 12))


def test_compute_nonzero_bbox_all_zero_returns_full_extent():
    image = np.zeros((4, *_SHAPE), dtype=np.float32)
    bbox = compute_nonzero_bbox(image)
    assert bbox == ((0, 16), (0, 16), (0, 16))


def test_compute_nonzero_bbox_is_union_across_channels():
    image = np.zeros((2, *_SHAPE), dtype=np.float32)
    image[0, 0:2, 0:2, 0:2] = 1.0  # channel 0: one corner
    image[1, 14:16, 14:16, 14:16] = 1.0  # channel 1: opposite corner
    bbox = compute_nonzero_bbox(image)
    assert bbox == ((0, 16), (0, 16), (0, 16))


# --- crop_to_bbox ---------------------------------------------------------


def test_crop_to_bbox_4d_image():
    image = np.zeros((4, *_SHAPE), dtype=np.float32)
    bbox = ((2, 10), (3, 11), (4, 12))
    cropped = crop_to_bbox(image, bbox)
    assert cropped.shape == (4, 8, 8, 8)


def test_crop_to_bbox_3d_label():
    label = np.zeros(_SHAPE, dtype=np.uint8)
    bbox = ((2, 10), (3, 11), (4, 12))
    cropped = crop_to_bbox(label, bbox)
    assert cropped.shape == (8, 8, 8)


# --- remap_labels ----------------------------------------------------------


def test_remap_labels_maps_all_raw_values():
    seg = np.array([0, 1, 2, 4], dtype=np.uint8)
    out = remap_labels(seg)
    assert out.dtype == np.uint8
    assert out.tolist() == [BRATS_LABEL_MAP[v] for v in [0, 1, 2, 4]]
    assert out.tolist() == [0, 1, 2, 3]


def test_remap_labels_raises_on_label_3():
    seg = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        remap_labels(seg)
    assert "3" in str(exc_info.value)


def test_remap_labels_raises_on_unexpected_value_5():
    seg = np.array([0, 5], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        remap_labels(seg)
    assert "5" in str(exc_info.value)


def test_remap_labels_brats2023_convention_is_identity():
    seg = np.array([0, 1, 2, 3], dtype=np.uint8)
    out = remap_labels(seg, convention="brats2023")
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 1, 2, 3]


def test_remap_labels_brats2023_raises_on_raw_label_4():
    # brats2023 labels are already {0,1,2,3} (ET=3) -- raw 4 is invalid here,
    # the mirror image of brats2021 rejecting raw 3.
    seg = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        remap_labels(seg, convention="brats2023")
    assert "4" in str(exc_info.value)


def test_remap_labels_unknown_convention_raises_naming_valid_ones():
    seg = np.array([0, 1], dtype=np.uint8)
    with pytest.raises(ValueError) as exc_info:
        remap_labels(seg, convention="nonsense")
    message = str(exc_info.value)
    assert "nonsense" in message
    assert "brats2021" in message
    assert "brats2023" in message


# --- reorient_to_axcodes --------------------------------------------------


def _lps_affine() -> np.ndarray:
    """BraTS 2021's own affine: diag(-1, -1, 1), axis codes LPS."""
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [239.0, 239.0, 0.0]
    return affine


def _ras_affine() -> np.ndarray:
    """Identity affine, axis codes RAS (BraTS-Africa's convention)."""
    return np.eye(4)


def test_reorient_to_axcodes_is_exact_noop_when_already_matching():
    rng = np.random.default_rng(0)
    array = rng.normal(size=(4, 5, 6)).astype(np.float32)
    affine = _lps_affine()

    out_array, out_affine = reorient_to_axcodes(array, affine, ("L", "P", "S"))

    assert np.array_equal(out_array, array)
    np.testing.assert_array_equal(out_affine, affine)


def test_reorient_to_axcodes_round_trip_recovers_original():
    rng = np.random.default_rng(1)
    array = rng.normal(size=(4, 5, 6)).astype(np.float32)
    affine = _lps_affine()

    ras_array, ras_affine = reorient_to_axcodes(array, affine, ("R", "A", "S"))
    lps_array, lps_affine = reorient_to_axcodes(ras_array, ras_affine, ("L", "P", "S"))

    assert np.array_equal(lps_array, array)
    np.testing.assert_allclose(lps_affine, affine)


def test_reorient_to_axcodes_ras_to_lps_flips_first_two_axes():
    # A known asymmetric array pins the actual geometry of the flip, not
    # just its shape -- RAS -> LPS reverses R<->L and A<->P, both axis 0
    # and axis 1, while axis 2 (S) is unchanged.
    array = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(np.float32)
    affine = _ras_affine()

    out_array, _ = reorient_to_axcodes(array, affine, ("L", "P", "S"))

    np.testing.assert_array_equal(out_array, np.flip(array, (0, 1)))


def test_reorient_to_axcodes_updated_affine_matches_target_axcodes():
    array = np.zeros((4, 5, 6), dtype=np.float32)
    affine = _ras_affine()

    _, out_affine = reorient_to_axcodes(array, affine, ("L", "P", "S"))

    assert nib.aff2axcodes(out_affine) == ("L", "P", "S")


def test_reorient_to_axcodes_preserves_label_values():
    label = np.zeros((4, 5, 6), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 3
    affine = _ras_affine()

    out_label, _ = reorient_to_axcodes(label, affine, ("L", "P", "S"))

    assert out_label.dtype == np.uint8
    assert set(np.unique(out_label).tolist()) == set(np.unique(label).tolist())


def test_reorient_to_axcodes_raises_on_bad_target_axcodes():
    array = np.zeros((4, 5, 6), dtype=np.float32)
    affine = _ras_affine()
    with pytest.raises(ValueError):
        reorient_to_axcodes(array, affine, ("L", "P"))  # wrong length
    with pytest.raises(ValueError):
        reorient_to_axcodes(array, affine, ("X", "P", "S"))  # invalid code


# --- load_case_arrays --------------------------------------------------


def test_load_case_arrays_image_shape_and_channel_order(tmp_path: Path):
    modality_arrays = {}
    for i, role in enumerate(("t1", "t1ce", "t2", "flair")):
        arr = np.zeros(_SHAPE, dtype=np.float32)
        arr[_BRAIN] = float(i + 1)  # distinguishes channels by value
        modality_arrays[role] = arr
    case = _make_case(tmp_path, modality_arrays=modality_arrays)

    image, label, meta = load_case_arrays(case)

    assert image.shape == (4, *_SHAPE)
    assert image.dtype == np.float32
    for i, role in enumerate(("t1", "t1ce", "t2", "flair")):
        assert image[i][_BRAIN][0, 0, 0] == pytest.approx(float(i + 1))
    assert label is None
    assert meta["original_shape"] == _SHAPE


def test_load_case_arrays_meta_spacing_and_affine_serializable(tmp_path: Path):
    zooms = (1.5, 1.5, 2.0)
    case = _make_case(tmp_path, zooms=zooms)
    _, _, meta = load_case_arrays(case)
    assert meta["spacing"] == pytest.approx(zooms)
    assert isinstance(meta["affine"], list)
    assert meta["affine"] == np.eye(4).tolist()


def test_load_case_arrays_with_label(tmp_path: Path):
    seg = _known_label()
    case = _make_case(tmp_path, seg_array=seg)
    image, label, meta = load_case_arrays(case)
    assert label is not None
    assert label.shape == _SHAPE
    assert np.array_equal(label, seg)


def test_load_case_arrays_raises_on_modality_shape_mismatch(tmp_path: Path):
    case = _make_case(tmp_path, shapes={"t1ce": (16, 16, 8)})
    with pytest.raises(ValueError) as exc_info:
        load_case_arrays(case)
    assert "t1ce" in str(exc_info.value)


# --- preprocess_case --------------------------------------------------


def test_preprocess_case_writes_expected_files(tmp_path: Path):
    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    summary = preprocess_case(case, out_dir)

    case_dir = out_dir / case.case_id
    image_path = case_dir / "image.npy"
    label_path = case_dir / "label.npy"
    meta_path = case_dir / "meta.json"
    assert image_path.is_file()
    assert label_path.is_file()
    assert meta_path.is_file()

    image = np.load(image_path)
    label = np.load(label_path)
    assert image.dtype == np.float16
    assert label.dtype == np.uint8
    assert summary["skipped"] is False


def test_preprocess_case_cropped_smaller_than_original(tmp_path: Path):
    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    summary = preprocess_case(case, out_dir)
    assert summary["cropped_shape"] == (8, 8, 8)
    assert summary["original_shape"] == _SHAPE
    original_voxels = np.prod(summary["original_shape"])
    cropped_voxels = np.prod(summary["cropped_shape"])
    assert cropped_voxels < original_voxels


def test_preprocess_case_bbox_roundtrips_cropped_shape(tmp_path: Path):
    import json

    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir)

    meta = json.loads((out_dir / case.case_id / "meta.json").read_text())
    bbox = tuple(tuple(pair) for pair in meta["bbox"])
    # Cropping the ORIGINAL raw t1 volume with the stored bbox must
    # reproduce the stored cropped_shape -- that's what makes the bbox
    # useful for un-cropping predictions back into original geometry.
    raw_t1 = np.asarray(nib.load(str(case.t1)).dataobj)
    reproduced = crop_to_bbox(raw_t1, bbox)
    assert reproduced.shape == tuple(meta["cropped_shape"])


def test_preprocess_case_no_seg_no_label_file(tmp_path: Path):
    case = _make_case(tmp_path / "raw")  # seg_array=None
    out_dir = tmp_path / "out"
    summary = preprocess_case(case, out_dir)

    case_dir = out_dir / case.case_id
    assert not (case_dir / "label.npy").is_file()

    import json

    meta = json.loads((case_dir / "meta.json").read_text())
    assert meta["has_label"] is False
    assert summary["n_class_0"] == 0
    assert summary["n_class_1"] == 0


def test_preprocess_case_skip_when_already_processed(tmp_path: Path):
    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir)

    image_path = out_dir / case.case_id / "image.npy"
    mtime_before = image_path.stat().st_mtime_ns

    summary = preprocess_case(case, out_dir, overwrite=False)
    assert summary["skipped"] is True
    assert image_path.stat().st_mtime_ns == mtime_before


def test_preprocess_case_overwrite_true_rewrites(tmp_path: Path):
    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir)

    image_path = out_dir / case.case_id / "image.npy"
    mtime_before = image_path.stat().st_mtime_ns

    # ensure the filesystem mtime clock can register a difference
    import time

    time.sleep(0.01)

    summary = preprocess_case(case, out_dir, overwrite=True)
    assert summary["skipped"] is False
    assert image_path.stat().st_mtime_ns != mtime_before


def test_is_case_processed_false_then_true(tmp_path: Path):
    case = _make_case(tmp_path / "raw", seg_array=_known_label())
    out_dir = tmp_path / "out"
    assert is_case_processed(case.case_id, out_dir) is False
    preprocess_case(case, out_dir)
    assert is_case_processed(case.case_id, out_dir) is True


def test_is_case_processed_true_without_label_when_unlabeled(tmp_path: Path):
    case = _make_case(tmp_path / "raw")  # no seg
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir)
    assert is_case_processed(case.case_id, out_dir) is True


def test_label_voxel_counts_matches_hand_computed(tmp_path: Path):
    seg = _known_label()
    case = _make_case(tmp_path / "raw", seg_array=seg)
    out_dir = tmp_path / "out"

    # Hand-compute the remapped counts within the crop bbox: the image's
    # nonzero support (hence the bbox) is the `_BRAIN` cuboid, so count only
    # inside it -- the full array also includes background outside the crop.
    remapped = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in BRATS_LABEL_MAP.items():
        remapped[seg == src] = dst
    cropped_remapped = remapped[_BRAIN]
    expected_counts = {
        str(int(v)): int(c) for v, c in zip(*np.unique(cropped_remapped, return_counts=True))
    }

    summary = preprocess_case(case, out_dir)
    for cls in range(4):
        assert summary[f"n_class_{cls}"] == expected_counts.get(str(cls), 0)


# --- reorientation wiring is additive: BraTS 2021 output must not change ---


def test_preprocess_case_lps_source_is_bitwise_identical_to_no_reorientation(tmp_path: Path):
    """BraTS 2021's own affine is LPS, so reorienting to the default
    target_axcodes=("L", "P", "S") must be an exact no-op end to end.

    Proves this by reconstructing the OLD pipeline by hand (load -> per-
    channel z-score -> bbox from raw -> crop -> remap, with NO
    reorientation step anywhere) and asserting the saved arrays from
    `preprocess_case` match it exactly. This is what guarantees the 1251
    already-preprocessed BraTS 2021 cases stay valid after this change.
    """
    affine = _lps_affine()
    case = _make_case(tmp_path / "raw", seg_array=_known_label(), affine=affine)
    out_dir = tmp_path / "out"

    preprocess_case(case, out_dir, target_axcodes=("L", "P", "S"))

    image = np.load(out_dir / case.case_id / "image.npy")
    label = np.load(out_dir / case.case_id / "label.npy")

    raw_image, raw_label, _ = load_case_arrays(case)
    normalized = np.stack(
        [normalize_nonzero(raw_image[c]) for c in range(raw_image.shape[0])], axis=0
    )
    bbox = compute_nonzero_bbox(raw_image)
    expected_image = crop_to_bbox(normalized, bbox).astype(np.float16)
    expected_label = remap_labels(crop_to_bbox(raw_label, bbox)).astype(np.uint8)

    np.testing.assert_array_equal(image, expected_image)
    np.testing.assert_array_equal(label, expected_label)


def test_preprocess_case_meta_gains_axcodes_and_updates_affine(tmp_path: Path):
    """`meta.json` records source/target axcodes and stores the UPDATED
    affine, even in the no-op LPS case."""
    import json

    affine = _lps_affine()
    case = _make_case(tmp_path / "raw", seg_array=_known_label(), affine=affine)
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir, target_axcodes=("L", "P", "S"))

    meta = json.loads((out_dir / case.case_id / "meta.json").read_text())
    assert meta["source_axcodes"] == "LPS"
    assert meta["target_axcodes"] == "LPS"
    np.testing.assert_allclose(np.array(meta["affine"]), affine)


def test_preprocess_case_reorients_ras_source_to_lps_target(tmp_path: Path):
    """A RAS-convention case (BraTS-Africa's orientation) gets flipped to
    LPS before cropping, and `meta.json` records the real transform."""
    import json

    affine = _ras_affine()
    case = _make_case(tmp_path / "raw", seg_array=_known_label(), affine=affine)
    out_dir = tmp_path / "out"
    preprocess_case(case, out_dir, target_axcodes=("L", "P", "S"))

    meta = json.loads((out_dir / case.case_id / "meta.json").read_text())
    assert meta["source_axcodes"] == "RAS"
    assert meta["target_axcodes"] == "LPS"
    assert nib.aff2axcodes(np.array(meta["affine"])) == ("L", "P", "S")

    # Tumor voxel histogram is invariant under a pure flip/permutation --
    # same voxels, same values, just relabeled positions.
    label = np.load(out_dir / case.case_id / "label.npy")
    values, counts = np.unique(label, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(values, counts)}
    assert hist.get(3, 0) == 128  # ET voxel count from _known_label()


def test_preprocess_case_brats2023_label_convention(tmp_path: Path):
    """`label_convention="brats2023"` remaps raw {0,1,2,3} (ET=3) as an
    identity, not through the brats2021 {0,1,2,4} table."""
    seg = np.zeros(_SHAPE, dtype=np.uint8)
    seg[_BRAIN] = 1
    seg[2:6, 3:11, 4:12] = 2
    seg[2:4, 3:11, 4:12] = 3  # ET is raw label 3 under brats2023

    case = _make_case(tmp_path / "raw", seg_array=seg, affine=_lps_affine())
    out_dir = tmp_path / "out"
    summary = preprocess_case(case, out_dir, label_convention="brats2023")

    assert summary["n_class_3"] == 128  # same 2x8x8 sub-block as _known_label()
