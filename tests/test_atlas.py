"""Tests for `neurovision.anatomy.atlas`.

Every test runs on CPU on small synthetic arrays (well under a second each),
using tiny NIfTI files written to `tmp_path`. None of them touch the real
SRI24 atlas or real BraTS data.
"""

from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from omegaconf import OmegaConf

from neurovision.anatomy import atlas
from neurovision.anatomy.atlas import (
    Atlas,
    apply_index_transform,
    load_atlas,
    load_nifti,
    parse_lut,
    reorient_to_target,
    solve_index_transform,
)

MERGE_PATTERNS = [r"(_[xyz][0-9]+)$", r"(_AP_[0-9]+)$"]

# Shared synthetic target geometry: axis 0 and axis 1 both size N=8 (so a
# single translation constant works for both, exactly like the real BraTS
# affine where both axes happen to be 240), axis 2 size 4.
N = 8
D, H, W = N, N, 4
DST_SHAPE = (D, H, W)
DST_AFFINE = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, N - 1],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# The two real orientations measured in docs/research/phase0_atlas_findings.md.
AP_MIRROR_AFFINE = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
PBMAP_AFFINE = [
    [1.0, 0.0, 0.0, -(N - 1)],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _write_nifti(path: Path, array: np.ndarray, affine) -> None:
    img = nib.Nifti1Image(np.asarray(array), np.asarray(affine, dtype=np.float64))
    nib.save(img, str(path))


# --------------------------------------------------------------------------- #
# 1-4, 9. solve_index_transform / apply_index_transform on the real-world cases
# --------------------------------------------------------------------------- #


def test_ap_mirror_case_matches_measured_transform() -> None:
    transform = solve_index_transform(AP_MIRROR_AFFINE, DST_AFFINE, DST_SHAPE, DST_SHAPE)
    assert transform.perm == (0, 1, 2)
    assert transform.flip == (False, True, False)

    arr = np.arange(D * H * W).reshape(D, H, W)
    out = apply_index_transform(arr, transform)
    assert np.array_equal(out, arr[:, ::-1])


def test_pbmap_case_matches_measured_transform() -> None:
    transform = solve_index_transform(PBMAP_AFFINE, DST_AFFINE, DST_SHAPE, DST_SHAPE)
    assert transform.flip == (True, True, False)

    arr = np.arange(D * H * W).reshape(D, H, W)
    out = apply_index_transform(arr, transform)
    assert np.array_equal(out, arr[::-1, ::-1])


def test_identity_affine_gives_identity_transform() -> None:
    transform = solve_index_transform(DST_AFFINE, DST_AFFINE, DST_SHAPE, DST_SHAPE)
    assert transform.perm == (0, 1, 2)
    assert transform.flip == (False, False, False)

    arr = np.arange(D * H * W).reshape(D, H, W)
    out = apply_index_transform(arr, transform)
    assert np.array_equal(out, arr)


def test_round_trip_recovers_original_array_exactly() -> None:
    arr = np.arange(D * H * W).reshape(D, H, W).astype(np.int32)
    forward = reorient_to_target(arr, AP_MIRROR_AFFINE, DST_AFFINE, DST_SHAPE)
    backward = reorient_to_target(forward, DST_AFFINE, AP_MIRROR_AFFINE, DST_SHAPE)
    assert np.array_equal(backward, arr)


def test_apply_index_transform_result_is_c_contiguous() -> None:
    transform = solve_index_transform(AP_MIRROR_AFFINE, DST_AFFINE, DST_SHAPE, DST_SHAPE)
    arr = np.arange(D * H * W).reshape(D, H, W)
    out = apply_index_transform(arr, transform)
    assert out.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------- #
# 5-8. Rejected transforms
# --------------------------------------------------------------------------- #


def test_rotation_raises_and_mentions_resampling() -> None:
    theta = math.pi / 4
    rotated = np.eye(4)
    rotated[:3, :3] = [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="resampling"):
        solve_index_transform(rotated, np.eye(4), DST_SHAPE, DST_SHAPE)


def test_non_unit_scale_raises() -> None:
    scaled = np.diag([2.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        solve_index_transform(np.eye(4), scaled, DST_SHAPE, DST_SHAPE)


def test_wrong_translation_offset_raises() -> None:
    wrong_dst = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, N - 2],  # should be N - 1
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValueError, match="offset"):
        solve_index_transform(AP_MIRROR_AFFINE, wrong_dst, DST_SHAPE, DST_SHAPE)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="disagree"):
        solve_index_transform(AP_MIRROR_AFFINE, DST_AFFINE, (D, H, W), (D, H + 1, W))


# --------------------------------------------------------------------------- #
# 10. LUT parsing handles mixed separators
# --------------------------------------------------------------------------- #


def test_lut_parsing_handles_mixed_separators(tmp_path: Path) -> None:
    lut_text = (
        "1\tPrecentral_L\t100\t50\t50\t0\n"
        "2\tPrecentral_R\t100\t50\t50\t0\n"
        "201 LateralVentricle_L_y48 10 10 10 0\n"
        "203 LateralVentricle_L_y49 10 10 10 0\n"
    )
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text(lut_text)

    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")
    all_ids = {label_id for s in labels.structures for label_id in s.label_ids}
    assert all_ids == {1, 2, 201, 203}

    # A tab-only split would have missed every space-separated row entirely.
    tab_only_ids = set()
    for line in lut_text.splitlines():
        tokens = line.split("\t")
        if tokens and tokens[0].isdigit():
            tab_only_ids.add(int(tokens[0]))
    assert tab_only_ids == {1, 2}
    assert len(all_ids) > len(tab_only_ids)


# --------------------------------------------------------------------------- #
# 11-12. Merging and laterality
# --------------------------------------------------------------------------- #


def _write_lut(path: Path, rows: list[tuple[int, str]]) -> None:
    lines = [f"{label_id} {name} 0 0 0 0" for label_id, name in rows]
    path.write_text("\n".join(lines) + "\n")


def test_merging_yields_expected_parents_and_label_ids(tmp_path: Path) -> None:
    rows = [
        (201, "LateralVentricle_L_y48"),
        (203, "LateralVentricle_L_y49"),
        (591, "Pons_x111"),
        (592, "Pons_x112"),
        (581, "CorpusCallosum_AP_0"),
        (582, "CorpusCallosum_AP_1"),
        (1, "Precentral_L"),
    ]
    lut_path = tmp_path / "lut.txt"
    _write_lut(lut_path, rows)

    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")
    assert set(labels.names) == {
        "LateralVentricle_L",
        "Pons",
        "CorpusCallosum",
        "Precentral_L",
    }
    assert labels.by_name("LateralVentricle_L").label_ids == (201, 203)
    assert labels.by_name("Pons").label_ids == (591, 592)
    assert labels.by_name("CorpusCallosum").label_ids == (581, 582)
    assert labels.by_name("Precentral_L").label_ids == (1,)


def test_laterality_from_merged_name(tmp_path: Path) -> None:
    rows = [
        (1, "Precentral_L"),
        (2, "Precentral_R"),
        (3, "Vermis_8"),
        (201, "LateralVentricle_L_y48"),
    ]
    lut_path = tmp_path / "lut.txt"
    _write_lut(lut_path, rows)

    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")
    assert labels.by_name("Precentral_L").laterality == "L"
    assert labels.by_name("Precentral_R").laterality == "R"
    assert labels.by_name("Vermis_8").laterality == "midline"
    assert labels.by_name("LateralVentricle_L").laterality == "L"


def test_by_name_raises_on_unknown_structure(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.txt"
    _write_lut(lut_path, [(1, "Precentral_L")])
    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")
    with pytest.raises(ValueError, match="unknown structure"):
        labels.by_name("NotARealStructure")


def test_parse_lut_raises_on_conflicting_parent_for_one_id(tmp_path: Path) -> None:
    # Same id appearing twice with names that merge to different parents.
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text("5 Pons_x111 0 0 0 0\n5 Precentral_L 0 0 0 0\n")
    with pytest.raises(ValueError):
        parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")


def test_parse_lut_raises_on_zero_structures(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text("0 Background 0 0 0 0\n")
    with pytest.raises(ValueError, match="zero structures"):
        parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")


# --------------------------------------------------------------------------- #
# 13. lookup_array
# --------------------------------------------------------------------------- #


def test_lookup_array_maps_ids_and_leaves_others_at_minus_one(tmp_path: Path) -> None:
    # StructA is built from two per-plane sub-labels merged to one parent.
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text("1 StructA_x1 0 0 0 0\n2 StructA_x2 0 0 0 0\n5 StructB 0 0 0 0\n")
    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")
    assert set(labels.names) == {"StructA", "StructB"}

    table = labels.lookup_array(max_id=10)
    assert table.shape == (11,)
    assert table.dtype == np.int32
    assert table[0] == -1  # background
    struct_a_index = labels.names.index("StructA")
    struct_b_index = labels.names.index("StructB")
    assert table[1] == struct_a_index
    assert table[2] == struct_a_index
    assert table[5] == struct_b_index
    # Unknown ids (present in neither structure) stay -1.
    assert table[3] == -1
    assert table[10] == -1

    with pytest.raises(ValueError, match="exceeds max_id"):
        labels.lookup_array(max_id=3)  # id 5 exceeds this


# --------------------------------------------------------------------------- #
# 14. Unmapped ids surface, and are not counted as background
# --------------------------------------------------------------------------- #


def test_unmapped_ids_surface_and_are_not_background(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text("1 StructA 0 0 0 0\n2 StructB 0 0 0 0\n")
    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")

    parcellation = np.zeros((4, 4, 2), dtype=np.int16)
    parcellation[0, 0, 0] = 1  # StructA
    parcellation[1, 1, 0] = 2  # StructB
    parcellation[2, 2, 0] = 99  # no LUT row
    parcellation[2, 2, 1] = 99  # no LUT row

    unmapped_ids = (99,)
    case_atlas = Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=None,
        tissue_codes={},
        name="test",
        version="0",
        source="test",
        unmapped_ids=unmapped_ids,
    )

    assert case_atlas.unmapped_ids == (99,)
    assert case_atlas.labels.name_for_id(99) == "unclassified"
    assert case_atlas.labels.name_for_id(0) == ""  # background stays distinct

    coverage = case_atlas.coverage()
    assert coverage["n_unmapped_voxels"] == 2
    assert coverage["n_unmapped_ids"] == 1
    assert coverage["n_structures"] == 2
    # The 2 unmapped voxels must not have been folded into the 2 labelled ones.
    assert coverage["n_labelled_voxels"] == 2


def test_tissue_mask_raises_when_no_tissue_loaded(tmp_path: Path) -> None:
    lut_path = tmp_path / "lut.txt"
    lut_path.write_text("1 StructA 0 0 0 0\n")
    labels = parse_lut(lut_path, MERGE_PATTERNS, unmapped_name="unclassified")

    case_atlas = Atlas(
        parcellation=np.zeros((2, 2, 2), dtype=np.int16),
        labels=labels,
        tissue=None,
        tissue_codes={"GM": 2},
        name="t",
        version="0",
        source="t",
        unmapped_ids=(),
    )
    with pytest.raises(ValueError, match="no tissue map"):
        case_atlas.tissue_mask("GM")


# --------------------------------------------------------------------------- #
# 15. 4-D squeeze
# --------------------------------------------------------------------------- #


def test_load_nifti_squeezes_singleton_trailing_axis(tmp_path: Path) -> None:
    path = tmp_path / "vol.nii"
    arr = np.arange(D * H * W).reshape(D, H, W, 1).astype(np.int16)
    _write_nifti(path, arr, AP_MIRROR_AFFINE)

    loaded, affine = load_nifti(path)
    assert loaded.shape == (D, H, W)
    assert np.array_equal(loaded, arr[..., 0])
    assert affine.shape == (4, 4)


def test_load_nifti_raises_on_non_singleton_trailing_axis(tmp_path: Path) -> None:
    path = tmp_path / "vol.nii"
    arr = np.zeros((D, H, W, 3), dtype=np.int16)
    _write_nifti(path, arr, AP_MIRROR_AFFINE)

    with pytest.raises(ValueError, match="non-singleton"):
        load_nifti(path)


# --------------------------------------------------------------------------- #
# 16. load_atlas end to end (tissues source)
# --------------------------------------------------------------------------- #


def _base_cfg(atlas_root: Path, tissue_source: str = "tissues") -> OmegaConf:
    return OmegaConf.create(
        {
            "dir": str(atlas_root.parent),
            "subdir": atlas_root.name,
            "version": "test-1.0",
            "source": "synthetic test fixture",
            "parcellation": {
                "name": "tzo116plus",
                "image": "parc.nii",
                "lut": "lut.txt",
                "merge_patterns": MERGE_PATTERNS,
                "unmapped_name": "unclassified",
            },
            "tissue": {
                "source": tissue_source,
                "image": "tissues.nii",
                "codes": {"CSF": 1, "GM": 2, "WM": 3},
                "pbmap": {"GM": "pbmap_GM.nii", "WM": "pbmap_WM.nii", "CSF": "pbmap_CSF.nii"},
            },
            "target": {"shape": list(DST_SHAPE), "spacing": [1.0, 1.0, 1.0], "affine": DST_AFFINE},
            "reorient": {"per_file_from_affine": True, "require_pure_flip": True},
        }
    )


def test_load_atlas_end_to_end_with_tissues_source(tmp_path: Path) -> None:
    root = tmp_path / "sri24"
    root.mkdir()

    raw_parc = np.zeros((D, H, W), dtype=np.int16)
    raw_parc[2, 3, 1] = 1  # StructA
    raw_parc[5, 6, 2] = 2  # StructB
    _write_nifti(root / "parc.nii", raw_parc, AP_MIRROR_AFFINE)

    lut_path = root / "lut.txt"
    lut_path.write_text("1 StructA 0 0 0 0\n2 StructB 0 0 0 0\n")

    raw_tissue = np.zeros((D, H, W), dtype=np.uint8)
    raw_tissue[0, 0, 0] = 1  # CSF
    raw_tissue[1, 1, 1] = 2  # GM
    raw_tissue[3, 3, 3] = 3  # WM
    _write_nifti(root / "tissues.nii", raw_tissue, AP_MIRROR_AFFINE)

    cfg = _base_cfg(root, tissue_source="tissues")
    result = load_atlas(cfg)

    expected_parc = raw_parc[:, ::-1, :]
    assert np.array_equal(result.parcellation, expected_parc)

    expected_tissue = raw_tissue[:, ::-1, :]
    assert np.array_equal(result.tissue, expected_tissue)

    assert np.array_equal(result.structure_mask("StructA"), expected_parc == 1)
    assert np.array_equal(result.structure_mask("StructB"), expected_parc == 2)
    assert np.array_equal(result.tissue_mask("GM"), expected_tissue == 2)

    coverage = result.coverage()
    assert coverage["n_structures"] == 2
    assert coverage["n_unmapped_voxels"] == 0
    assert coverage["n_unmapped_ids"] == 0
    assert coverage["n_labelled_voxels"] == 2

    assert result.shape == DST_SHAPE
    assert result.name == "tzo116plus"
    assert result.version == "test-1.0"


# --------------------------------------------------------------------------- #
# 17. load_atlas with pbmap tissue source agrees with the tissues source
# --------------------------------------------------------------------------- #


def test_load_atlas_pbmap_source_matches_tissues_source(tmp_path: Path) -> None:
    root = tmp_path / "sri24"
    root.mkdir()

    raw_parc = np.zeros((D, H, W), dtype=np.int16)
    _write_nifti(root / "parc.nii", raw_parc, AP_MIRROR_AFFINE)
    lut_path = root / "lut.txt"
    lut_path.write_text("1 StructA 0 0 0 0\n")

    # The hard tissue map this test's pbmap variant must reproduce exactly.
    raw_tissue = np.zeros((D, H, W), dtype=np.uint8)
    raw_tissue[0, 0, 0] = 1  # CSF
    raw_tissue[1, 1, 1] = 2  # GM
    raw_tissue[3, 3, 3] = 3  # WM
    _write_nifti(root / "tissues.nii", raw_tissue, AP_MIRROR_AFFINE)

    expected_tissue = raw_tissue[:, ::-1, :]  # what the "tissues" path produces

    codes = {"CSF": 1, "GM": 2, "WM": 3}
    # pbmap files are flipped on BOTH axis 0 and axis 1 relative to BraTS
    # (Finding H). Build each probability map so that, after that transform,
    # it reproduces `expected_tissue`'s one-hot code exactly.
    for name, code in codes.items():
        prob_dst = (expected_tissue == code).astype(np.float32)
        raw_prob = prob_dst[::-1, ::-1, :]  # undo the pbmap transform to get the raw frame
        _write_nifti(root / f"pbmap_{name}.nii", raw_prob, PBMAP_AFFINE)

    cfg = _base_cfg(root, tissue_source="pbmap")
    result = load_atlas(cfg)

    assert np.array_equal(result.tissue, expected_tissue)


# --------------------------------------------------------------------------- #
# 18. Missing atlas directory
# --------------------------------------------------------------------------- #


def test_missing_atlas_directory_raises_and_names_fetch_script(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="fetch_atlas"):
        load_atlas(cfg)


def test_missing_required_file_raises_and_names_fetch_script(tmp_path: Path) -> None:
    root = tmp_path / "sri24"
    root.mkdir()
    # No files written at all -- the directory exists but parc.nii does not.
    cfg = _base_cfg(root)
    with pytest.raises(FileNotFoundError, match="fetch_atlas"):
        load_atlas(cfg)


# --------------------------------------------------------------------------- #
# 19. No deep-learning-stack import
# --------------------------------------------------------------------------- #


def test_atlas_module_does_not_import_the_deep_learning_stack() -> None:
    """Keeps this module importable with no deep-learning stack installed.

    Checked against the source text rather than `sys.modules`, because
    pytest has almost certainly imported torch already for some other test
    file in the suite -- see `test_burden.py`'s equivalent guard.
    """
    source = Path(atlas.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
