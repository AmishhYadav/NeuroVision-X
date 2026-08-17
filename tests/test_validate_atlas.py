"""Tests for scripts/validate_atlas.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_burden_script.py` and `scripts/smoke_test.py`
use.

Every test builds a small synthetic SRI24-shaped atlas on disk (with
`nibabel`, exactly like `tests/test_atlas.py`) and a handful of synthetic
preprocessed cases under `tmp_path`. Nothing here touches the real atlas or
real BraTS data, and every test runs well under a second.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import hydra
import nibabel as nib
import numpy as np
import pytest
from omegaconf import OmegaConf

from neurovision.utils.io import ensure_dir, read_json, write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_atlas.py"
_spec = importlib.util.spec_from_file_location("validate_atlas_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
validate_atlas_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["validate_atlas_script"] = validate_atlas_script
_spec.loader.exec_module(validate_atlas_script)

CaseSample = validate_atlas_script.CaseSample
sample_cases = validate_atlas_script.sample_cases
brain_mask_iter = validate_atlas_script.brain_mask_iter
tumour_mask_iter = validate_atlas_script.tumour_mask_iter
axial_display_slice = validate_atlas_script.axial_display_slice
qc_overlay_figure = validate_atlas_script.qc_overlay_figure
write_report = validate_atlas_script.write_report
run_validation = validate_atlas_script.run_validation
main = validate_atlas_script.main

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# --------------------------------------------------------------------------- #
# Synthetic geometry shared across the full-pipeline tests.
#
# Small on purpose (well under a second per test): 20x20x10, identity-frame
# atlas files (src affine == dst affine, so load_atlas needs no flip at all)
# with the BraTS-convention diag(-1, -1, 1)-style affine so axis 0 low index
# is patient right, matching the laterality check's own assumption.
# --------------------------------------------------------------------------- #

D, H, W = 20, 20, 10
SHAPE = (D, H, W)
AFFINE = [
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, float(H - 1)],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
MIDLINE_INDEX = (D - 1) / 2.0  # 9.5

# StructA_R (patient right, LOW d) and StructA_L (patient left, HIGH d).
R_D_SLICE = slice(2, 7)
L_D_SLICE = slice(13, 18)

# "Brain" region: everything but a 1-voxel border, shared by the atlas
# tissue map and every case's channel-0 nonzero support, so brain-mask Dice
# is exactly 1.0 by construction -- deterministic and easy to force-fail
# (any threshold above 1.0) without touching real data.
BRAIN_BOX = (slice(1, D - 1), slice(1, H - 1), slice(1, W - 1))

# A small tumour, entirely inside StructA_R (so the lobe check has something
# to attribute) and entirely inside BRAIN_BOX.
TUMOUR_BOX = (slice(3, 5), slice(8, 12), slice(3, 7))


def _write_nifti(path: Path, array: np.ndarray, affine) -> None:
    img = nib.Nifti1Image(np.asarray(array), np.asarray(affine, dtype=np.float64))
    nib.save(img, str(path))


def _build_synthetic_atlas(atlas_root: Path) -> None:
    """Writes a tiny SRI24-shaped atlas under `atlas_root/sri24/`, default filenames."""
    sri24_dir = atlas_root / "sri24"
    sri24_dir.mkdir(parents=True)

    parcellation = np.zeros(SHAPE, dtype=np.int16)
    parcellation[R_D_SLICE, :, :] = 1  # StructA_R
    parcellation[L_D_SLICE, :, :] = 2  # StructA_L
    _write_nifti(sri24_dir / "tzo116plus.nii", parcellation, AFFINE)

    lut_path = sri24_dir / "SRI24-tzo116plus.txt"
    lut_path.write_text("1 StructA_R 0 0 0 0\n2 StructA_L 0 0 0 0\n")

    tissue = np.zeros(SHAPE, dtype=np.uint8)
    tissue[BRAIN_BOX] = 2  # GM, everywhere inside the "brain" box
    _write_nifti(sri24_dir / "tissues.nii", tissue, AFFINE)


def _write_lobe_map(path: Path) -> None:
    write_yaml(
        {
            "lobes": ["frontal", "temporal", "parietal", "deep", "occipital"],
            "epidemiology_lobes": [
                "frontal",
                "temporal",
                "parietal",
                "deep",
                "occipital",
                "excluded",
            ],
            "structures": {
                "StructA": {"lobe": "frontal", "epidemiology_lobe": "frontal"},
            },
        },
        path,
    )


def _write_case(prep_dir: Path, case_id: str) -> None:
    """Writes one preprocessed case: full-extent bbox (cropped == original)."""
    case_dir = ensure_dir(prep_dir / case_id)

    bbox = [[0, D], [0, H], [0, W]]
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(SHAPE),
            "cropped_shape": list(SHAPE),
            "bbox": bbox,
            "affine": AFFINE,
            "spacing": [1.0, 1.0, 1.0],
            "has_label": True,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )

    image = np.zeros((4, D, H, W), dtype=np.float16)
    # Negative on purpose: exercises brain_mask_iter's abs() requirement for
    # EVERY case in the end-to-end tests, not just the dedicated unit test.
    image[0][BRAIN_BOX] = -1.0
    np.save(case_dir / "image.npy", image)

    label = np.zeros(SHAPE, dtype=np.uint8)
    label[TUMOUR_BOX] = 2  # ED -> non-empty, inside StructA_R
    np.save(case_dir / "label.npy", label)


def _build_standard_setup(tmp_path: Path, n_cases: int = 5) -> tuple[Path, Path, Path]:
    """Builds an atlas root, a lobe map, and `n_cases` preprocessed cases.

    Returns:
        `(atlas_root, lobe_map_path, prep_dir)`.
    """
    atlas_root = tmp_path / "atlas_root"
    _build_synthetic_atlas(atlas_root)

    lobe_map_path = tmp_path / "aal_lobes.yaml"
    _write_lobe_map(lobe_map_path)

    prep_dir = tmp_path / "preprocessed"
    for i in range(n_cases):
        _write_case(prep_dir, f"CASE_{i:02d}")

    return atlas_root, lobe_map_path, prep_dir


def _compose_cfg(
    tmp_path: Path,
    atlas_root: Path,
    lobe_map_path: Path,
    prep_dir: Path,
    output_dir: Path,
    *,
    n_cases: int = 5,
    seed: int = 0,
    min_brain_dice: float = 0.85,
    qc_cases: int = 2,
    max_midline_deviation: float = 1.0,
):
    """Composes the real Hydra config, then patches in tmp_path-rooted synthetic geometry.

    Uses `hydra.compose` (the real config files) for everything except the
    values that must point at THIS test's synthetic fixtures -- assigned
    directly on the composed `DictConfig` afterwards, since every one of
    them is an EXISTING key (only `n_brain_cases`, deliberately never
    exercised through the full config here, is absent from the YAML).
    """
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"output_dir={output_dir}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
    ]
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    cfg.anatomy.dir = str(atlas_root)
    cfg.anatomy.version = "test-1.0"
    cfg.anatomy.source = "synthetic test fixture"
    cfg.anatomy.target.shape = list(SHAPE)
    cfg.anatomy.target.affine = AFFINE

    cfg.anatomy.validation.n_cases = n_cases
    cfg.anatomy.validation.seed = seed
    cfg.anatomy.validation.lobe_map = str(lobe_map_path)
    cfg.anatomy.validation.min_brain_dice = min_brain_dice
    cfg.anatomy.validation.min_laterality_pairs_correct = 1.0
    cfg.anatomy.validation.midline_index = MIDLINE_INDEX
    cfg.anatomy.validation.max_midline_deviation = max_midline_deviation
    cfg.anatomy.validation.qc_cases = qc_cases

    return cfg


# --------------------------------------------------------------------------- #
# 1. Sampling is reproducible; brain_case_ids is a prefix.
# --------------------------------------------------------------------------- #


def test_sampling_is_reproducible_and_brain_ids_are_a_prefix(tmp_path: Path) -> None:
    prep_dir = tmp_path / "preprocessed"
    for i in range(10):
        _write_case(prep_dir, f"CASE_{i:02d}")

    cfg = OmegaConf.create({"n_cases": 6, "seed": 7})
    sample_a = sample_cases(cfg, prep_dir)
    sample_b = sample_cases(cfg, prep_dir)

    assert sample_a.case_ids == sample_b.case_ids
    assert isinstance(sample_a, CaseSample)
    assert len(sample_a.case_ids) == 6
    # n_brain_cases absent -> default 60, clamped down to n_cases.
    assert sample_a.brain_case_ids == sample_a.case_ids
    assert sample_a.case_ids[: len(sample_a.brain_case_ids)] == sample_a.brain_case_ids


# --------------------------------------------------------------------------- #
# 2. n_brain_cases defaults to 60, honoured when present.
# --------------------------------------------------------------------------- #


def test_n_brain_cases_defaults_to_60_and_is_honoured_when_present(tmp_path: Path) -> None:
    prep_dir = tmp_path / "preprocessed"
    for i in range(70):
        _write_case(prep_dir, f"CASE_{i:03d}")

    cfg_default = OmegaConf.create({"n_cases": 65, "seed": 1})
    sample_default = sample_cases(cfg_default, prep_dir)
    assert len(sample_default.brain_case_ids) == 60
    assert sample_default.brain_case_ids == sample_default.case_ids[:60]

    cfg_explicit = OmegaConf.create({"n_cases": 65, "seed": 1, "n_brain_cases": 5})
    sample_explicit = sample_cases(cfg_explicit, prep_dir)
    assert len(sample_explicit.brain_case_ids) == 5
    assert sample_explicit.brain_case_ids == sample_explicit.case_ids[:5]
    # Same seed and n_cases -> same full draw regardless of n_brain_cases.
    assert sample_explicit.case_ids == sample_default.case_ids


# --------------------------------------------------------------------------- #
# 3. Too few cases raises ValueError naming the found count.
# --------------------------------------------------------------------------- #


def test_too_few_cases_raises_value_error(tmp_path: Path) -> None:
    prep_dir = tmp_path / "preprocessed"
    for i in range(3):
        _write_case(prep_dir, f"CASE_{i}")

    cfg = OmegaConf.create({"n_cases": 10, "seed": 0})
    with pytest.raises(ValueError, match="3"):
        sample_cases(cfg, prep_dir)


# --------------------------------------------------------------------------- #
# 4. Empty directory raises FileNotFoundError naming it.
# --------------------------------------------------------------------------- #


def test_empty_directory_raises_file_not_found(tmp_path: Path) -> None:
    prep_dir = tmp_path / "empty_preprocessed"
    prep_dir.mkdir()
    cfg = OmegaConf.create({"n_cases": 1, "seed": 0})
    with pytest.raises(FileNotFoundError, match=str(prep_dir.resolve())):
        sample_cases(cfg, prep_dir)


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    prep_dir = tmp_path / "does_not_exist"
    cfg = OmegaConf.create({"n_cases": 1, "seed": 0})
    with pytest.raises(FileNotFoundError, match=str(prep_dir.resolve())):
        sample_cases(cfg, prep_dir)


# --------------------------------------------------------------------------- #
# 5. brain_mask_iter uses abs(): a case with all-negative channel-0 brain
#    voxels must still yield a mask covering them.
# --------------------------------------------------------------------------- #


def test_brain_mask_iter_uses_abs_not_greater_than_zero(tmp_path: Path) -> None:
    prep_dir = tmp_path / "preprocessed"
    _write_case(prep_dir, "NEG_CASE")  # image[0] is -1.0 inside BRAIN_BOX, by construction

    case_id, mask = next(brain_mask_iter(["NEG_CASE"], prep_dir))
    assert case_id == "NEG_CASE"

    expected = np.zeros(SHAPE, dtype=bool)
    expected[BRAIN_BOX] = True
    assert np.array_equal(mask, expected)

    # A `> 0` implementation would give an EMPTY mask here (every brain voxel
    # is exactly -1.0) -- this is the assertion that would fail against that
    # regression.
    assert mask.any()


# --------------------------------------------------------------------------- #
# 6. Masks come back in original geometry.
# --------------------------------------------------------------------------- #


def test_masks_are_in_original_geometry(tmp_path: Path) -> None:
    prep_dir = tmp_path / "preprocessed"
    _write_case(prep_dir, "CASE_0")

    _, brain_mask = next(brain_mask_iter(["CASE_0"], prep_dir))
    assert brain_mask.shape == SHAPE

    _, tumour_mask = next(tumour_mask_iter(["CASE_0"], prep_dir))
    assert tumour_mask.shape == SHAPE
    expected_tumour = np.zeros(SHAPE, dtype=bool)
    expected_tumour[TUMOUR_BOX] = True
    assert np.array_equal(tumour_mask, expected_tumour)


# --------------------------------------------------------------------------- #
# 7. Iterators are lazy.
# --------------------------------------------------------------------------- #


def test_brain_mask_iter_is_lazy(tmp_path: Path) -> None:
    missing_root = tmp_path / "nowhere"
    gen = brain_mask_iter(["does_not_exist"], missing_root)
    assert inspect.isgenerator(gen)
    # Constructing the generator must not have touched the filesystem yet.
    with pytest.raises(FileNotFoundError):
        next(gen)


def test_tumour_mask_iter_is_lazy(tmp_path: Path) -> None:
    missing_root = tmp_path / "nowhere"
    gen = tumour_mask_iter(["does_not_exist"], missing_root)
    assert inspect.isgenerator(gen)
    with pytest.raises(FileNotFoundError):
        next(gen)


# --------------------------------------------------------------------------- #
# 8. run_validation end to end writes all five write_report artifacts.
# --------------------------------------------------------------------------- #


def test_run_validation_end_to_end_writes_all_artifacts(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir)

    report = run_validation(cfg)
    assert report.passed is True

    for name in (
        "alignment_report.json",
        "alignment_per_case_dice.csv",
        "alignment_laterality_pairs.csv",
        "alignment_lobe_distribution.csv",
        "alignment_summary.txt",
    ):
        path = output_dir / name
        assert path.is_file(), f"missing artifact: {name}"

    payload = read_json(output_dir / "alignment_report.json")
    assert "passed" in payload
    assert payload["passed"] is True
    assert payload["atlas"]["name"] == "tzo116plus"
    assert {c["name"] for c in payload["checks"]} == {
        "brain_mask_dice",
        "laterality_pairs",
        "midline_estimate",
        "lobe_distribution",
    }


# --------------------------------------------------------------------------- #
# 9. A failed gate still writes artifacts, and the failing check is named.
# --------------------------------------------------------------------------- #


def test_failed_gate_still_writes_artifacts(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    # Actual brain-mask Dice is exactly 1.0 by construction (see BRAIN_BOX) --
    # 1.01 is unreachable and forces a deterministic failure.
    cfg = _compose_cfg(
        tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir, min_brain_dice=1.01
    )

    report = run_validation(cfg)
    assert report.passed is False
    failing_names = {c.name for c in report.failures()}
    assert "brain_mask_dice" in failing_names

    payload = read_json(output_dir / "alignment_report.json")
    assert payload["passed"] is False
    failing_in_json = {c["name"] for c in payload["checks"] if c["gating"] and not c["passed"]}
    assert "brain_mask_dice" in failing_in_json
    # Every artifact must still exist -- a failed gate is diagnosed from disk.
    assert (output_dir / "alignment_per_case_dice.csv").is_file()
    assert (output_dir / "alignment_summary.txt").is_file()


# --------------------------------------------------------------------------- #
# 10. main() exits non-zero on failure, zero on success.
# --------------------------------------------------------------------------- #


def test_main_exits_nonzero_on_failed_gate(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir, min_brain_dice=1.01
    )

    with pytest.raises(SystemExit) as excinfo:
        main(cfg)
    assert excinfo.value.code != 0


def test_main_exits_zero_on_passing_gate(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir)

    # main() only raises SystemExit on FAILURE; a passing run returns
    # normally (falls off the end of the function), so there is nothing to
    # catch here -- calling it directly and letting it return is the
    # "exit code 0" case.
    main(cfg)


# --------------------------------------------------------------------------- #
# 11. Advisory failure never fails the gate.
# --------------------------------------------------------------------------- #


def test_advisory_lobe_distribution_never_fails_the_gate(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir)

    # By construction every case's tumour sits entirely in StructA_R ->
    # "frontal", i.e. 100% frontal against a 40% reference -- about as
    # atrocious a lobe distribution as a two-lobe-map can produce. The
    # module's own contract is that lobe_distribution.gating is False and
    # .passed is unconditionally True; this test exercises that through the
    # real driver rather than asserting the contract in the abstract.
    report = run_validation(cfg)

    lobe_check = next(c for c in report.checks if c.name == "lobe_distribution")
    assert lobe_check.gating is False
    assert lobe_check.passed is True
    assert report.passed is True


def test_main_exit_code_zero_despite_atrocious_lobe_distribution(tmp_path: Path) -> None:
    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(tmp_path, atlas_root, lobe_map_path, prep_dir, output_dir)
    main(cfg)  # must return normally (no SystemExit) despite the skewed lobe distribution


# --------------------------------------------------------------------------- #
# 12. qc_overlay_figure writes a non-empty PNG with the expected grid, no window.
# --------------------------------------------------------------------------- #


def test_qc_overlay_figure_writes_nonempty_png_with_expected_grid(tmp_path: Path) -> None:
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"

    atlas_root, lobe_map_path, prep_dir = _build_standard_setup(tmp_path, n_cases=2)
    from neurovision.anatomy.atlas import load_atlas

    cfg = OmegaConf.create(
        {
            "dir": str(atlas_root),
            "subdir": "sri24",
            "version": "test-1.0",
            "source": "synthetic test fixture",
            "parcellation": {
                "name": "tzo116plus",
                "image": "tzo116plus.nii",
                "lut": "SRI24-tzo116plus.txt",
                "merge_patterns": [r"(_[xyz][0-9]+)$", r"(_AP_[0-9]+)$"],
                "unmapped_name": "unclassified",
            },
            "tissue": {
                "source": "tissues",
                "image": "tissues.nii",
                "codes": {"CSF": 1, "GM": 2, "WM": 3},
                "pbmap": {"GM": "pbmap_GM.nii", "WM": "pbmap_WM.nii", "CSF": "pbmap_CSF.nii"},
            },
            "target": {"shape": list(SHAPE), "spacing": [1.0, 1.0, 1.0], "affine": AFFINE},
        }
    )
    atlas = load_atlas(cfg)

    out_path = tmp_path / "qc" / "overlay.png"
    n_slices = 2
    case_ids = ["CASE_00", "CASE_01"]

    import matplotlib.pyplot as plt

    with patch.object(plt, "subplots", wraps=plt.subplots) as spy:
        result = qc_overlay_figure(atlas, case_ids, prep_dir, out_path, n_slices=n_slices)
        args, kwargs = spy.call_args
        assert args[:2] == (len(case_ids), n_slices) or (
            kwargs.get("nrows") == len(case_ids) and kwargs.get("ncols") == n_slices
        )

    assert result == out_path
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 13. QC orientation: anterior at TOP, patient's left on the RIGHT.
# --------------------------------------------------------------------------- #


def test_axial_display_slice_puts_anterior_top_and_left_on_the_right() -> None:
    # d: axis 0, right(low) -> left(high). h: axis 1, anterior(low) ->
    # posterior(high). w: axis 2, the fixed axial slice axis.
    volume = np.zeros((D, H, W), dtype=np.float32)

    anterior_d, anterior_h, slice_w = 10, 1, 5  # near-minimal h -> anterior
    volume[anterior_d, anterior_h, slice_w] = 1.0  # "anterior" marker

    left_d, left_h = D - 2, 10  # near-maximal d -> patient's LEFT
    volume[left_d, left_h, slice_w] = 2.0  # "patient's left" marker

    display = axial_display_slice(volume, slice_w)
    assert display.shape == (H, D)

    # The anterior marker's row must be nearer the TOP (row index 0) than the
    # left marker's row -- i.e. displayed higher on screen.
    anterior_row, anterior_col = np.argwhere(display == 1.0)[0]
    left_row, left_col = np.argwhere(display == 2.0)[0]

    assert anterior_row < H // 2  # anterior marker sits in the top half
    assert anterior_row == anterior_h  # row = h directly, no reversal
    assert anterior_col == anterior_d  # col = d directly, no reversal

    # The "patient's left" marker (high d) must land in the RIGHT half of
    # the displayed image (high column index).
    assert left_col > D // 2
    assert left_col == left_d
    assert left_row == left_h


def test_qc_overlay_figure_orientation_end_to_end(tmp_path: Path) -> None:
    """Same claim as the unit test above, but through the real QC pipeline.

    Builds a case whose T1 channel carries an unambiguous anterior marker
    (bright, near h=0) and an unambiguous "patient's left" marker (bright,
    near d=D-1), and asserts the array actually handed to `imshow` -- not
    the saved PNG file -- places them at the top and right respectively.
    This is the test that would have caught the demo's sideways-head bug.
    """
    prep_dir = tmp_path / "preprocessed"
    case_id = "ORIENTED"
    case_dir = ensure_dir(prep_dir / case_id)

    write_json(
        {
            "case_id": case_id,
            "original_shape": list(SHAPE),
            "cropped_shape": list(SHAPE),
            "bbox": [[0, D], [0, H], [0, W]],
            "affine": AFFINE,
            "spacing": [1.0, 1.0, 1.0],
            "has_label": True,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )

    image = np.zeros((4, D, H, W), dtype=np.float16)
    slice_w = 5
    anterior_d, anterior_h = 10, 1
    left_d, left_h = D - 2, 10
    image[0, anterior_d, anterior_h, slice_w] = 5.0
    image[0, left_d, left_h, slice_w] = 6.0
    np.save(case_dir / "image.npy", image)

    # Tumour centred exactly at slice_w so the QC figure picks it.
    label = np.zeros(SHAPE, dtype=np.uint8)
    label[9:11, 9:11, slice_w : slice_w + 1] = 2  # exactly one w-slice -> unambiguous centroid
    np.save(case_dir / "label.npy", label)

    atlas_root = tmp_path / "atlas_root"
    _build_synthetic_atlas(atlas_root)
    from neurovision.anatomy.atlas import load_atlas

    cfg = OmegaConf.create(
        {
            "dir": str(atlas_root),
            "subdir": "sri24",
            "version": "test-1.0",
            "source": "synthetic test fixture",
            "parcellation": {
                "name": "tzo116plus",
                "image": "tzo116plus.nii",
                "lut": "SRI24-tzo116plus.txt",
                "merge_patterns": [r"(_[xyz][0-9]+)$", r"(_AP_[0-9]+)$"],
                "unmapped_name": "unclassified",
            },
            "tissue": {
                "source": "tissues",
                "image": "tissues.nii",
                "codes": {"CSF": 1, "GM": 2, "WM": 3},
                "pbmap": {"GM": "pbmap_GM.nii", "WM": "pbmap_WM.nii", "CSF": "pbmap_CSF.nii"},
            },
            "target": {"shape": list(SHAPE), "spacing": [1.0, 1.0, 1.0], "affine": AFFINE},
        }
    )
    atlas = load_atlas(cfg)

    captured: list[np.ndarray] = []
    import matplotlib.pyplot as plt

    original_imshow = plt.Axes.imshow

    def _spy_imshow(self, array, *args, **kwargs):
        captured.append(np.asarray(array))
        return original_imshow(self, array, *args, **kwargs)

    with patch.object(plt.Axes, "imshow", _spy_imshow):
        out_path = qc_overlay_figure(atlas, [case_id], prep_dir, tmp_path / "qc.png", n_slices=1)

    assert out_path.is_file()
    # The first imshow call in the single panel is the T1 grayscale slice.
    t1_drawn = captured[0]
    assert t1_drawn.shape == (H, D)

    anterior_row, anterior_col = np.argwhere(t1_drawn == 5.0)[0]
    left_row, left_col = np.argwhere(t1_drawn == 6.0)[0]

    assert anterior_row < H // 2
    assert left_col > D // 2
