"""Tests for scripts/localize.py.

The script lives under scripts/, not src/, so it is loaded via
`importlib.util.spec_from_file_location` rather than a normal package import
-- the same pattern `tests/test_burden_script.py` and `scripts/smoke_test.py`
use.

Every test composes the REAL Hydra config (via `hydra.compose`, exactly like
`scripts/smoke_test.py`) and then points the atlas / knowledge-base fields at
a tiny SYNTHETIC atlas and synthetic knowledge YAML files written under
`tmp_path`, built with `nibabel` the way `tests/test_atlas.py` does. Nothing
here touches the real SRI24 atlas or real BraTS data, and each test is well
under a second.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path
from types import ModuleType

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml
from omegaconf import OmegaConf

from neurovision.utils.io import ensure_dir, read_yaml, write_json, write_yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "localize.py"
_spec = importlib.util.spec_from_file_location("localize_script", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
localize_script: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["localize_script"] = localize_script
_spec.loader.exec_module(localize_script)

LocalizeSource = localize_script.LocalizeSource
resolve_sources = localize_script.resolve_sources
load_case = localize_script.load_case
localize_one = localize_script.localize_one
run_localize = localize_script.run_localize
load_atlas = localize_script.load_atlas
load_knowledge = localize_script.load_knowledge
eloquent_union_mask = localize_script.eloquent_union_mask
resolve_involvement = localize_script.resolve_involvement
load_involvement_groups = localize_script.load_involvement_groups

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# --- Shared synthetic geometry ----------------------------------------------
#
# ATLAS_SHAPE doubles as ORIGINAL_SHAPE for every case: a saved prediction
# lives at exactly this shape (uncropped BraTS geometry), and the atlas is
# reoriented into this same grid by load_atlas.
ATLAS_SHAPE: tuple[int, int, int] = (60, 60, 30)
BBOX: list[list[int]] = [[20, 60], [20, 60], [10, 30]]
CROPPED_SHAPE: tuple[int, int, int] = (40, 40, 20)
SPACING: list[float] = [1.0, 1.0, 1.0]

# Identity affine for both the raw atlas NIfTI files and cfg.anatomy.target.affine:
# solve_index_transform solves M = inv(src_affine) @ dst_affine, and identity
# affines on both sides give M = I regardless of the affine's actual values,
# so this yields a no-op reorientation and keeps the geometry easy to reason
# about by hand.
IDENTITY_AFFINE: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Synthetic atlas structures, in ATLAS_SHAPE (=raw, since the affine is
# identity) coordinates. Chosen so the eloquent structures (Precentral_L/R,
# combining into one contiguous D[0:19) H[0:10) W[0:20) box) sit entirely
# OUTSIDE BBOX, while Frontal_L sits entirely INSIDE it -- so a cropped case's
# lesion can touch Frontal_L but never the eloquent structures, and the
# cropped eloquent view is genuinely empty (the NaN-distance case).
_PRECENTRAL_L_SLICE = (slice(0, 19), slice(0, 10), slice(0, 10))
_PRECENTRAL_R_SLICE = (slice(0, 19), slice(0, 10), slice(10, 20))
_FRONTAL_L_SLICE = (slice(30, 60), slice(30, 60), slice(15, 30))
_ELOQUENT_BOX = ((0, 18), (0, 9), (0, 19))  # inclusive bounds of the union above


def _write_nifti(path: Path, array: np.ndarray, affine) -> None:
    img = nib.Nifti1Image(np.asarray(array), np.asarray(affine, dtype=np.float64))
    nib.save(img, str(path))


def _build_atlas_dir(root: Path) -> None:
    """Writes a tiny synthetic SRI24-shaped atlas (parcellation + LUT + tissue) to `root`."""
    root.mkdir(parents=True, exist_ok=True)

    parc = np.zeros(ATLAS_SHAPE, dtype=np.int16)
    parc[_PRECENTRAL_L_SLICE] = 1
    parc[_PRECENTRAL_R_SLICE] = 2
    parc[_FRONTAL_L_SLICE] = 3
    _write_nifti(root / "parc.nii", parc, IDENTITY_AFFINE)

    (root / "lut.txt").write_text(
        "1 Precentral_L 0 0 0 0\n2 Precentral_R 0 0 0 0\n3 Frontal_L 0 0 0 0\n"
    )

    # Real tissue codes (CSF=1, GM=2, WM=3, matching configs/anatomy/sri24.yaml's
    # defaults, which _compose_cfg does not override), placed under the
    # structures they anatomically belong to: grey matter under the two
    # Precentral (motor cortex) boxes, white matter under Frontal_L. Left
    # all-zero ("outside_tissue") everywhere else. Was previously all-zero,
    # which made every tissue_overlap fraction trivially NaN/0 -- no existing
    # test in this file asserts anything about the tissue array's content.
    tissue = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
    tissue[_PRECENTRAL_L_SLICE] = 2  # GM
    tissue[_PRECENTRAL_R_SLICE] = 2  # GM
    tissue[_FRONTAL_L_SLICE] = 3  # WM
    _write_nifti(root / "tissues.nii", tissue, IDENTITY_AFFINE)


def _write_eloquence_yaml(path: Path, *, near_eloquent_mm: float = 10.0) -> None:
    doc = {
        "version": 1,
        "classification": {
            "name": "Test eloquence grading",
            "primary_citation": "Test R. A test classification. Test Journal. 1998.",
            "eloquent_structures_verbatim": "Eloquent locations are the motor cortex.",
        },
        "vocabulary": ["eloquent", "unclassified"],
        "default": "unclassified",
        "near_eloquent_rule": {"distance_mm": near_eloquent_mm},
        "coverage_gaps": [],
        "entries": [
            {
                "structure_name": "Precentral_L",
                "eloquence": "eloquent",
                "matched_term": "motor cortex",
            },
            {
                "structure_name": "Precentral_R",
                "eloquence": "eloquent",
                "matched_term": "motor cortex",
            },
        ],
    }
    path.write_text(yaml.safe_dump(doc))


def _write_involvement_yaml(
    path: Path,
    *,
    ventricle_structures: list[str],
    deep_wm_structures: list[str],
    search_radius_mm: float = 10.0,
    version: int = 1,
) -> None:
    """A synthetic, schema-valid stand-in for knowledge/involvement_groups.yaml.

    Names ONLY structures that exist in the synthetic atlas built by
    `_build_atlas_dir` (Precentral_L, Precentral_R, Frontal_L) -- never the
    real committed knowledge file and never the real SRI24 structure names.
    """
    doc = {
        "version": version,
        "groups": {
            "ventricles": {"structures": ventricle_structures, "missing": []},
            "deep_white_matter": {"structures": deep_wm_structures, "missing": []},
        },
        "tissue": {"cortical": "GM", "white_matter": "WM", "csf": "CSF"},
        "epicentre": {"search_radius_mm": search_radius_mm},
        "relationship_to_vasari": {
            "status": "approximate_and_unverified",
            "claim": "Test-only synthetic involvement groups; not validated against VASARI.",
        },
    }
    path.write_text(yaml.safe_dump(doc))


def _write_lobe_yaml(path: Path) -> None:
    doc = {
        "version": 1,
        "structures": {
            "Precentral": {"lobe": "frontal"},
            "Frontal": {"lobe": "frontal"},
        },
    }
    path.write_text(yaml.safe_dump(doc))


def _build_knowledge_fixtures(
    tmp_path: Path, *, near_eloquent_mm: float = 10.0
) -> tuple[Path, Path, Path]:
    """Builds the atlas dir + eloquence yaml + lobe yaml. Returns their paths."""
    atlas_root = tmp_path / "sri24"
    _build_atlas_dir(atlas_root)
    eloquence_path = tmp_path / "eloquence_map.yaml"
    _write_eloquence_yaml(eloquence_path, near_eloquent_mm=near_eloquent_mm)
    lobe_path = tmp_path / "aal_lobes.yaml"
    _write_lobe_yaml(lobe_path)
    return atlas_root, eloquence_path, lobe_path


def _nested_block(shape: tuple[int, int, int]) -> np.ndarray:
    """Concentric-sphere nested ET-subset-of-TC-subset-of-WT label, small and fast.

    Same recipe as scripts/smoke_test.py's `_build_synthetic_label` and
    tests/test_burden_script.py's `_nested_label`.
    """
    d, h, w = shape
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    cz, cy, cx = d / 2, h / 2, w / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    min_edge = min(shape)
    label = np.zeros(shape, dtype=np.uint8)
    label[dist < min_edge * 0.45] = 2  # ED shell -> completes WT
    label[dist < min_edge * 0.28] = 1  # NCR/NET core -> completes TC
    label[dist < min_edge * 0.12] = 3  # ET, innermost
    return label


# A small nested lesion, placed inside CROPPED_SHAPE so that -- once embedded
# at BBOX's offset into ATLAS_SHAPE -- it lands entirely inside Frontal_L
# ([30:50, 30:50, 15:25], well within [30:60, 30:60, 15:30]) and nowhere near
# the eloquent structures (which live at D < 20).
_LESION_BLOCK_SHAPE: tuple[int, int, int] = (20, 20, 10)
_LESION_OFFSET_IN_CROPPED: tuple[int, int, int] = (10, 10, 5)


def _cropped_lesion_label() -> np.ndarray:
    small = _nested_block(_LESION_BLOCK_SHAPE)
    label = np.zeros(CROPPED_SHAPE, dtype=np.uint8)
    d0, h0, w0 = _LESION_OFFSET_IN_CROPPED
    bd, bh, bw = _LESION_BLOCK_SHAPE
    label[d0 : d0 + bd, h0 : h0 + bh, w0 : w0 + bw] = small
    return label


def _embed(
    cropped: np.ndarray, bbox: list[list[int]], original_shape: tuple[int, int, int]
) -> np.ndarray:
    """Places a cropped array into a zero background at its bbox offset."""
    full = np.zeros(original_shape, dtype=cropped.dtype)
    slices = tuple(slice(lo, hi) for lo, hi in bbox)
    full[slices] = cropped
    return full


def _write_meta(
    case_dir: Path,
    case_id: str,
    *,
    original_shape: tuple[int, int, int] = ATLAS_SHAPE,
    cropped_shape: tuple[int, int, int] = CROPPED_SHAPE,
    bbox: list[list[int]] = BBOX,
    spacing: list[float] = SPACING,
) -> None:
    write_json(
        {
            "case_id": case_id,
            "original_shape": list(original_shape),
            "cropped_shape": list(cropped_shape),
            "bbox": [list(b) for b in bbox],
            "affine": IDENTITY_AFFINE,
            "spacing": spacing,
            "has_label": True,
            "label_voxel_counts": None,
        },
        case_dir / "meta.json",
    )


def _write_case(prep_dir: Path, case_id: str) -> None:
    """Writes meta.json + a cropped label.npy for one case, no prediction."""
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)
    np.save(case_dir / "label.npy", _cropped_lesion_label())


def _write_prediction(eval_dir: Path, case_id: str) -> None:
    """Writes an uncropped prediction for a case that already has meta.json."""
    uncropped = _embed(_cropped_lesion_label(), BBOX, ATLAS_SHAPE)
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", uncropped)


def _write_splits(path: Path, case_ids: list[str], split: str = "test") -> None:
    payload = {"train": [], "val": [], "test": []}
    payload[split] = list(case_ids)
    write_yaml(payload, path)


def _compose_cfg(
    tmp_path: Path,
    prep_dir: Path,
    splits_path: Path,
    output_dir: Path,
    atlas_root: Path,
    eloquence_path: Path,
    lobe_path: Path,
    *,
    source: str = "label",
    eval_dir: Path | None = None,
    split: str = "test",
    min_frac: float | None = None,
    involvement_path: Path | None = None,
    omit_involvement_key: bool = False,
):
    """Composes the real Hydra config, then points atlas/knowledge fields at tmp_path fixtures.

    The atlas geometry (`anatomy.target.shape` / `.affine`) and the atlas
    file paths are set by direct attribute assignment on the composed
    `DictConfig` rather than through Hydra's CLI override grammar, since a
    4x4 affine matrix is awkward to express as an override dotlist string.
    Every key touched already exists in `configs/anatomy/sri24.yaml`, so this
    is a plain reassignment, not a schema change.
    """
    overrides = [
        f"data.root_dir={tmp_path}",
        f"data.preprocessing.out_dir={prep_dir}",
        f"data.splits.path={splits_path}",
        f"output_dir={output_dir}",
        f"analysis.localize.split={split}",
        f"analysis.localize.source={source}",
        "seed=42",
        "device=cpu",
        "wandb.mode=disabled",
    ]
    if eval_dir is not None:
        overrides.append(f"analysis.localize.eval_dir={eval_dir}")
    with hydra.initialize_config_dir(version_base="1.3", config_dir=_CONFIG_DIR):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    OmegaConf.set_struct(cfg, False)
    cfg.anatomy.dir = str(atlas_root.parent)
    cfg.anatomy.subdir = atlas_root.name
    cfg.anatomy.parcellation.image = "parc.nii"
    cfg.anatomy.parcellation.lut = "lut.txt"
    cfg.anatomy.tissue.image = "tissues.nii"
    cfg.anatomy.target.shape = list(ATLAS_SHAPE)
    cfg.anatomy.target.affine = IDENTITY_AFFINE
    cfg.analysis.localize.eloquence_map = str(eloquence_path)
    cfg.analysis.localize.lobe_map = str(lobe_path)
    if min_frac is not None:
        cfg.analysis.localize.min_frac = min_frac

    # The real composed default (configs/analysis/default.yaml) has
    # involvement.enabled: true with groups_map pointing at the real,
    # committed knowledge/involvement_groups.yaml, which names real SRI24
    # structures absent from this synthetic 3-structure atlas -- so every
    # test must explicitly say what it wants here, never inherit the
    # production default. `omit_involvement_key` simulates an OLDER config
    # composed before the `involvement` key existed at all.
    if omit_involvement_key:
        del cfg.analysis.localize["involvement"]
    elif involvement_path is not None:
        cfg.analysis.localize.involvement.enabled = True
        cfg.analysis.localize.involvement.groups_map = str(involvement_path)
    else:
        cfg.analysis.localize.involvement.enabled = False

    return cfg


CASE_IDS = ["CASE_A", "CASE_B", "CASE_C"]


def _standard_split(
    tmp_path: Path, *, prep_subdir: str = "preprocessed"
) -> tuple[Path, Path, Path]:
    """Writes 3 standard label-backed cases + a test split.

    Returns:
        `(prep_dir, splits_path, eval_dir)`.
    """
    prep_dir = tmp_path / prep_subdir
    for case_id in CASE_IDS:
        _write_case(prep_dir, case_id)
    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, CASE_IDS)
    eval_dir = tmp_path / "eval"
    for case_id in CASE_IDS:
        _write_prediction(eval_dir, case_id)
    return prep_dir, splits_path, eval_dir


# ---------------------------------------------------------------------------
# 1. Happy path, source=label
# ---------------------------------------------------------------------------


def test_happy_path_label_source(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out_label"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    anatomy_csv, summary_csv = run_localize(cfg)
    assert anatomy_csv == output_dir / "anatomy.csv"
    assert summary_csv == output_dir / "anatomy_summary.csv"

    table = pd.read_csv(anatomy_csv)
    summary = pd.read_csv(summary_csv)

    assert table.columns[0] == "case_id"
    assert summary.columns[0] == "case_id"
    assert set(table["case_id"]) == set(CASE_IDS)
    assert set(summary["case_id"]) == set(CASE_IDS)
    assert len(summary) == len(CASE_IDS)


# ---------------------------------------------------------------------------
# 2. Happy path, source=prediction
# ---------------------------------------------------------------------------


def test_happy_path_prediction_source(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out_pred"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )

    anatomy_csv, summary_csv = run_localize(cfg)
    table = pd.read_csv(anatomy_csv)
    summary = pd.read_csv(summary_csv)
    assert table.columns[0] == "case_id"
    assert summary.columns[0] == "case_id"
    assert set(summary["case_id"]) == set(CASE_IDS)
    assert len(summary) == len(CASE_IDS)


# ---------------------------------------------------------------------------
# 3. cropped follows source: regression guard for the crop-offset bug
# ---------------------------------------------------------------------------


def test_cropped_follows_source_gives_same_involved_structures(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    case_id = "SIDED"
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)
    np.save(case_dir / "label.npy", _cropped_lesion_label())

    eval_dir = tmp_path / "eval"
    _write_prediction(eval_dir, case_id)

    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, [case_id])

    cfg_label = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        tmp_path / "out_label",
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )
    cfg_pred = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        tmp_path / "out_pred",
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )

    df_label = pd.read_csv(run_localize(cfg_label)[0])
    df_pred = pd.read_csv(run_localize(cfg_pred)[0])

    # A mismatched `cropped` flag would shift every structure assignment by
    # the crop offset -- this asserts the two geometries agree exactly on
    # which structures the SAME lesion involves.
    for region in ("ET", "TC", "WT"):
        structures_label = set(df_label.loc[df_label["region"] == region, "structure"])
        structures_pred = set(df_pred.loc[df_pred["region"] == region, "structure"])
        assert structures_label == structures_pred
    assert "Frontal_L" in set(df_label["structure"])
    assert "Precentral_L" not in set(df_label["structure"])
    assert "Precentral_R" not in set(df_label["structure"])


# ---------------------------------------------------------------------------
# 4. Shape mismatch raises, naming the case
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    case_id = "BAD_SHAPE"
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    # Prediction is expected in ORIGINAL geometry; write it CROPPED instead.
    eval_dir = tmp_path / "eval"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", _cropped_lesion_label())

    source = LocalizeSource(
        case_id=case_id,
        array_path=predictions_dir / f"{case_id}.npy",
        meta_path=case_dir / "meta.json",
        cropped=False,
    )
    with pytest.raises(ValueError, match=case_id):
        load_case(source)


# ---------------------------------------------------------------------------
# 5. Missing files are reported and excluded; all missing raises
# ---------------------------------------------------------------------------


def test_missing_case_is_reported_and_excluded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    (prep_dir / "CASE_B" / "label.npy").unlink()

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    with caplog.at_level(logging.WARNING, logger="localize_script"):
        anatomy_csv, summary_csv = run_localize(cfg)

    summary = pd.read_csv(summary_csv)
    assert len(summary) == len(CASE_IDS) - 1
    assert "CASE_B" not in set(summary["case_id"])
    assert any("CASE_B" in record.getMessage() for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_all_missing_raises_file_not_found(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    for case_id in CASE_IDS:
        (prep_dir / case_id / "label.npy").unlink()

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        run_localize(cfg)
    assert str(prep_dir.resolve()) in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. A bad case does not kill the run; zero successes raises RuntimeError
# ---------------------------------------------------------------------------


def test_one_bad_case_does_not_kill_the_run(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)

    # Invalid class value 4 -- raw, unremapped BraTS uses 4 for enhancing
    # tumor; region_mask's validation rejects it.
    bad = _cropped_lesion_label()
    bad[bad == 3] = 4
    np.save(prep_dir / "CASE_B" / "label.npy", bad)

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    anatomy_csv, summary_csv = run_localize(cfg)
    summary = pd.read_csv(summary_csv)
    assert set(summary["case_id"]) == {"CASE_A", "CASE_C"}


def test_zero_successes_raises_runtime_error(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    for case_id in CASE_IDS:
        bad = _cropped_lesion_label()
        bad[bad == 3] = 4
        np.save(prep_dir / case_id / "label.npy", bad)

    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    with pytest.raises(RuntimeError):
        run_localize(cfg)


# ---------------------------------------------------------------------------
# 7. min_frac filtering drops a tiny row, never drops 'unlabelled', a normal
#    filtering run logs no warning, and frac_of_tumour_retained is reported.
# ---------------------------------------------------------------------------


def _filter_case_paths(
    tmp_path: Path, case_id: str = "FILTER_CASE"
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Writes the shared min_frac fixture: one dominant WT structure (Frontal_L,
    13500 voxels), one tiny non-'unlabelled' structure (1 voxel inside
    Precentral_L), and one tiny 'unlabelled' voxel -- so the default min_frac
    genuinely drops the non-unlabelled row while keeping the dominant one and
    the (never-dropped) unlabelled one.

    Returns:
        `(atlas_root, eloq_path, lobe_path, prep_dir, splits_path, eval_dir)`.
    """
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    array = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
    array[_FRONTAL_L_SLICE] = 2  # 30*30*15 = 13500 voxels, WT (ED)
    array[0, 0, 0] = 2  # 1 voxel inside Precentral_L (size 1900) -> tiny both ways
    array[0, 0, 25] = 2  # 1 voxel background -> 'unlabelled', also tiny

    eval_dir = tmp_path / "eval"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", array)

    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, [case_id])

    return atlas_root, eloq_path, lobe_path, prep_dir, splits_path, eval_dir


def test_min_frac_filtering_drops_tiny_row_but_never_unlabelled_and_warns_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    atlas_root, eloq_path, lobe_path, prep_dir, splits_path, eval_dir = _filter_case_paths(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )

    with caplog.at_level(logging.WARNING, logger="localize_script"):
        anatomy_csv, _summary_csv = run_localize(cfg)

    table = pd.read_csv(anatomy_csv)
    wt_rows = table[table["region"] == "WT"]

    structures = set(wt_rows["structure"])
    assert "Frontal_L" in structures
    assert "Precentral_L" not in structures  # dropped: both fractions < min_frac
    assert "unlabelled" in structures  # never dropped, even though also tiny

    # Regression guard for the fixed defect: min_frac genuinely dropped a row
    # here, and that is BY DESIGN -- a normal run must not warn about the
    # (correctly-unfiltered) sum identity or about the retained fraction
    # (which is still high). This is exactly the case that made the old
    # post-filter check fire on essentially every case of every run.
    messages = [record.getMessage() for record in caplog.records]
    assert not any("sums to" in m for m in messages)
    assert not any("retained" in m for m in messages)


def test_frac_of_tumour_retained_present_and_less_than_one_when_dropped(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path, prep_dir, splits_path, eval_dir = _filter_case_paths(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )

    _anatomy_csv, summary_csv = run_localize(cfg)
    summary = pd.read_csv(summary_csv)

    assert "frac_of_tumour_retained" in summary.columns
    retained = float(summary.loc[0, "frac_of_tumour_retained"])
    assert 0.0 <= retained <= 1.0
    assert retained < 1.0  # the tiny Precentral_L row was dropped


def test_frac_of_tumour_retained_equals_one_when_nothing_dropped(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path, prep_dir, splits_path, eval_dir = _filter_case_paths(tmp_path)
    output_dir = tmp_path / "out"
    # min_frac=0.0: a fraction is never strictly less than 0.0, so nothing is
    # dropped and the reported retention must be exactly 1.0.
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        min_frac=0.0,
    )

    _anatomy_csv, summary_csv = run_localize(cfg)
    summary = pd.read_csv(summary_csv)
    retained = float(summary.loc[0, "frac_of_tumour_retained"])
    assert retained == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 8. distance_to_eloquent_mm is a real measured distance, not NaN
# ---------------------------------------------------------------------------


def _clamped_distance(point: tuple[int, int, int], box: tuple[tuple[int, int], ...]) -> float:
    """Euclidean distance from `point` to the nearest point of an axis-aligned box (inclusive)."""
    nearest = tuple(min(max(p, lo), hi) for p, (lo, hi) in zip(point, box, strict=True))
    return math.dist(point, nearest)


def test_distance_to_eloquent_is_real_measured_value_when_not_touching(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path, near_eloquent_mm=10.0)
    case_id = "FAR_CASE"
    prep_dir = tmp_path / "preprocessed"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    point = (55, 55, 25)
    array = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
    array[point] = 2  # single WT voxel, far from the eloquent box

    eval_dir = tmp_path / "eval"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", array)

    splits_path = tmp_path / "splits.yaml"
    _write_splits(splits_path, [case_id])
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )

    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(
        cfg.analysis.localize.eloquence_map, cfg.analysis.localize.lobe_map, atlas
    )
    eloquent_mask = eloquent_union_mask(atlas, knowledge)
    source = LocalizeSource(
        case_id=case_id,
        array_path=predictions_dir / f"{case_id}.npy",
        meta_path=case_dir / "meta.json",
        cropped=False,
    )
    _table, summary = localize_one(source, atlas, knowledge, eloquent_mask, cfg)

    expected = _clamped_distance(point, _ELOQUENT_BOX)
    assert not math.isnan(summary["distance_to_eloquent_mm"])
    assert summary["distance_to_eloquent_mm"] == pytest.approx(expected, abs=1e-6)
    assert expected > 10.0  # sanity: this case genuinely does not touch eloquence


# ---------------------------------------------------------------------------
# 9. near_eloquent: True below threshold, False above, False when NaN
# ---------------------------------------------------------------------------


def _run_single_voxel_case(
    tmp_path: Path,
    case_id: str,
    point: tuple[int, int, int],
    *,
    cropped: bool,
    near_eloquent_mm: float = 10.0,
) -> dict:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(
        tmp_path, near_eloquent_mm=near_eloquent_mm
    )
    prep_dir = tmp_path / f"prep_{case_id}"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    if cropped:
        array = np.zeros(CROPPED_SHAPE, dtype=np.uint8)
        array[point] = 2
        array_path = case_dir / "label.npy"
        np.save(array_path, array)
        source_kind = "label"
        eval_dir = None
    else:
        array = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
        array[point] = 2
        eval_dir = tmp_path / f"eval_{case_id}"
        predictions_dir = ensure_dir(eval_dir / "predictions")
        array_path = predictions_dir / f"{case_id}.npy"
        np.save(array_path, array)
        source_kind = "prediction"

    splits_path = tmp_path / f"splits_{case_id}.yaml"
    _write_splits(splits_path, [case_id])
    output_dir = tmp_path / f"out_{case_id}"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source=source_kind,
        eval_dir=eval_dir,
    )

    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(
        cfg.analysis.localize.eloquence_map, cfg.analysis.localize.lobe_map, atlas
    )
    eloquent_mask = eloquent_union_mask(atlas, knowledge)
    source = LocalizeSource(
        case_id=case_id, array_path=array_path, meta_path=case_dir / "meta.json", cropped=cropped
    )
    _table, summary = localize_one(source, atlas, knowledge, eloquent_mask, cfg)
    return summary


def test_near_eloquent_true_below_threshold(tmp_path: Path) -> None:
    # Overlaps Precentral_L directly -> distance 0.0 < 10.0.
    summary = _run_single_voxel_case(tmp_path, "NEAR", (2, 2, 2), cropped=False)
    assert summary["distance_to_eloquent_mm"] == 0.0
    assert summary["near_eloquent"] is True


def test_near_eloquent_false_above_threshold(tmp_path: Path) -> None:
    # Far from the eloquent box -> distance well above 10.0 (see test 8).
    summary = _run_single_voxel_case(tmp_path, "FAR", (55, 55, 25), cropped=False)
    assert summary["distance_to_eloquent_mm"] > 10.0
    assert summary["near_eloquent"] is False


def test_near_eloquent_false_when_distance_is_nan(tmp_path: Path) -> None:
    # Cropped view: BBOX excludes the eloquent structures entirely, so the
    # cropped eloquent mask is empty and the distance is NaN by construction.
    point_in_cropped_frontal = (12, 12, 6)  # inside Frontal_L's cropped block
    summary = _run_single_voxel_case(tmp_path, "NANCASE", point_in_cropped_frontal, cropped=True)
    assert math.isnan(summary["distance_to_eloquent_mm"])
    assert summary["near_eloquent"] is False


# ---------------------------------------------------------------------------
# 10. localize_config.yaml carries the coverage line, atlas version, source
# ---------------------------------------------------------------------------


def test_localize_config_yaml_written(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )
    run_localize(cfg)

    config_path = output_dir / "localize_config.yaml"
    assert config_path.is_file()
    record = read_yaml(config_path)

    assert record["split"] == "test"
    assert record["source"] == "prediction"
    assert "resolved_source_dir" in record
    assert str(eval_dir.resolve() / "predictions") == record["resolved_source_dir"]

    assert "coverage_line" in record
    assert isinstance(record["coverage_line"], str)
    assert "structures classified eloquent" in record["coverage_line"]

    assert record["atlas"]["version"] == str(cfg.anatomy.version)
    assert record["atlas"]["name"] or True  # atlas.name is present (tzo116plus by default)


# ---------------------------------------------------------------------------
# 11. Atlas, knowledge base, and eloquent mask are built exactly once
# ---------------------------------------------------------------------------


def test_atlas_knowledge_and_eloquent_mask_built_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )

    calls = {"atlas": 0, "knowledge": 0, "eloquent": 0}

    original_load_atlas = localize_script.load_atlas
    original_load_knowledge = localize_script.load_knowledge
    original_eloquent_union_mask = localize_script.eloquent_union_mask

    def counting_load_atlas(anatomy_cfg):
        calls["atlas"] += 1
        return original_load_atlas(anatomy_cfg)

    def counting_load_knowledge(*args, **kwargs):
        calls["knowledge"] += 1
        return original_load_knowledge(*args, **kwargs)

    def counting_eloquent_union_mask(*args, **kwargs):
        calls["eloquent"] += 1
        return original_eloquent_union_mask(*args, **kwargs)

    monkeypatch.setattr(localize_script, "load_atlas", counting_load_atlas)
    monkeypatch.setattr(localize_script, "load_knowledge", counting_load_knowledge)
    monkeypatch.setattr(localize_script, "eloquent_union_mask", counting_eloquent_union_mask)

    run_localize(cfg)  # 3 cases in CASE_IDS

    assert calls == {"atlas": 1, "knowledge": 1, "eloquent": 1}


# ---------------------------------------------------------------------------
# 12. Determinism: two runs over the same inputs produce byte-identical CSVs
# ---------------------------------------------------------------------------


def test_determinism(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)

    out_a = tmp_path / "out_a"
    cfg_a = _compose_cfg(
        tmp_path, prep_dir, splits_path, out_a, atlas_root, eloq_path, lobe_path, source="label"
    )
    anatomy_a, summary_a = run_localize(cfg_a)

    out_b = tmp_path / "out_b"
    cfg_b = _compose_cfg(
        tmp_path, prep_dir, splits_path, out_b, atlas_root, eloq_path, lobe_path, source="label"
    )
    anatomy_b, summary_b = run_localize(cfg_b)

    assert anatomy_a.read_bytes() == anatomy_b.read_bytes()
    assert summary_a.read_bytes() == summary_b.read_bytes()


# ---------------------------------------------------------------------------
# 13. Unfiltered identity check, retained-fraction scoping and warning,
#     monkeypatching localize_case for full control over the table.
# ---------------------------------------------------------------------------


def _prepare_localize_one_env(
    tmp_path: Path, case_id: str, *, min_frac: float | None = None
) -> tuple:
    """Builds everything `localize_one` needs for one case, so a test can
    monkeypatch `localize_case` (the thing `localize_one` calls) before
    invoking it directly, and drive the returned table exactly.

    Returns:
        `(atlas, knowledge, eloquent_mask, source, cfg)`.
    """
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir = tmp_path / f"prep_{case_id}"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    array = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
    array[0, 0, 0] = 2  # a single WT voxel so region_mask('WT') is non-empty
    eval_dir = tmp_path / f"eval_{case_id}"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    array_path = predictions_dir / f"{case_id}.npy"
    np.save(array_path, array)

    splits_path = tmp_path / f"splits_{case_id}.yaml"
    _write_splits(splits_path, [case_id])
    output_dir = tmp_path / f"out_{case_id}"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        min_frac=min_frac,
    )

    atlas = load_atlas(cfg.anatomy)
    knowledge = load_knowledge(
        cfg.analysis.localize.eloquence_map, cfg.analysis.localize.lobe_map, atlas
    )
    eloquent_mask = eloquent_union_mask(atlas, knowledge)
    source = LocalizeSource(
        case_id=case_id, array_path=array_path, meta_path=case_dir / "meta.json", cropped=False
    )
    return atlas, knowledge, eloquent_mask, source, cfg


def _synthetic_localize_table(rows: list[dict]) -> pd.DataFrame:
    """A minimal, correctly-columned stand-in for `localize_case`'s output.

    Used to monkeypatch `localize_case` so a test can drive
    `frac_of_tumour` / `frac_of_structure` exactly, rather than deriving them
    from a real mask/atlas intersection.
    """
    records = []
    for row in rows:
        records.append(
            {
                "region": row["region"],
                "structure": row["structure"],
                "laterality": row.get("laterality", "unknown"),
                "lobe": row.get("lobe", ""),
                "eloquence": row.get("eloquence", "unclassified"),
                "matched_term": row.get("matched_term", ""),
                "n_voxels": row.get("n_voxels", 1),
                "volume_mm3": row.get("volume_mm3", 1.0),
                "frac_of_tumour": row["frac_of_tumour"],
                "frac_of_structure": row["frac_of_structure"],
            }
        )
    return pd.DataFrame.from_records(records)


def test_unfiltered_identity_check_fires_before_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    atlas, knowledge, eloquent_mask, source, cfg = _prepare_localize_one_env(tmp_path, "BAD_SUM")
    # frac_of_tumour sums to 0.5, not 1.0 -- a bug in localize_case's own
    # output, not something min_frac filtering could ever cause (min_frac
    # only removes rows, which can only make a filtered sum smaller than 1.0,
    # never larger, and this check runs BEFORE filtering anyway).
    bad_table = _synthetic_localize_table(
        [{"region": "WT", "structure": "X", "frac_of_tumour": 0.5, "frac_of_structure": 0.5}]
    )
    monkeypatch.setattr(localize_script, "localize_case", lambda *a, **k: bad_table.copy())

    with caplog.at_level(logging.WARNING, logger="localize_script"):
        localize_one(source, atlas, knowledge, eloquent_mask, cfg)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("localize_case" in r.getMessage() for r in warnings)


def test_retained_fraction_scoped_to_wt_like_summarize_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas, knowledge, eloquent_mask, source, cfg = _prepare_localize_one_env(
        tmp_path, "SCOPE", min_frac=0.5
    )
    # WT: B (frac 0.1) is dropped at min_frac=0.5 -> WT retains 0.9.
    # ET: D (frac 0.4) is dropped at min_frac=0.5 -> ET retains 0.6.
    # summarize_case (and therefore frac_of_tumour_retained) must use WT.
    table = _synthetic_localize_table(
        [
            {"region": "WT", "structure": "A", "frac_of_tumour": 0.9, "frac_of_structure": 0.9},
            {"region": "WT", "structure": "B", "frac_of_tumour": 0.1, "frac_of_structure": 0.1},
            {"region": "ET", "structure": "C", "frac_of_tumour": 0.6, "frac_of_structure": 0.6},
            {"region": "ET", "structure": "D", "frac_of_tumour": 0.4, "frac_of_structure": 0.4},
        ]
    )
    monkeypatch.setattr(localize_script, "localize_case", lambda *a, **k: table.copy())

    _table, summary = localize_one(source, atlas, knowledge, eloquent_mask, cfg)

    assert summary["frac_of_tumour_retained"] == pytest.approx(0.9)


def test_low_retained_fraction_warns_once_naming_case_and_min_frac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    case_id = "LOWRET"
    atlas, knowledge, eloquent_mask, source, cfg = _prepare_localize_one_env(
        tmp_path, case_id, min_frac=1.5
    )
    # The lone row's fractions (1.0) are both below min_frac=1.5, so it is
    # entirely dropped -- retention is 0.0, well under the 0.5 warn threshold.
    table = _synthetic_localize_table(
        [{"region": "WT", "structure": "A", "frac_of_tumour": 1.0, "frac_of_structure": 1.0}]
    )
    monkeypatch.setattr(localize_script, "localize_case", lambda *a, **k: table.copy())

    with caplog.at_level(logging.WARNING, logger="localize_script"):
        _table, summary = localize_one(source, atlas, knowledge, eloquent_mask, cfg)

    assert summary["frac_of_tumour_retained"] == pytest.approx(0.0, abs=1e-9)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert case_id in message
    assert "1.5" in message


# ---------------------------------------------------------------------------
# 14. _log_sanity_summary surfaces the retained fraction at the run level
# ---------------------------------------------------------------------------


def test_log_sanity_summary_reports_retained_fraction_and_low_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    summary_rows = [
        {
            "n_structures_involved": 2,
            "frac_unlabelled": 0.1,
            "n_eloquent_structures": 1,
            "frac_of_tumour_retained": 0.95,
        },
        {
            "n_structures_involved": 3,
            "frac_unlabelled": 0.05,
            "n_eloquent_structures": 0,
            "frac_of_tumour_retained": 0.2,  # below the 0.5 warn threshold
        },
    ]

    with caplog.at_level(logging.INFO, logger="localize_script"):
        localize_script._log_sanity_summary(summary_rows)

    message = " ".join(record.getMessage() for record in caplog.records)
    # median([0.95, 0.2]) == 0.575; exactly one row (0.2) is below 0.5.
    assert "frac_of_tumour_retained=0.575" in message
    assert "1 case(s) below" in message


# ---------------------------------------------------------------------------
# 15. Phase 3b involvement layer, wired into localize_one / run_localize.
#
# The real composed default has involvement.enabled: true pointing at the
# real, committed knowledge/involvement_groups.yaml, whose structure names
# (LateralVentricle_L, CorpusCallosum, ...) do not exist in the tiny
# synthetic atlas every other test in this file uses. `_compose_cfg` was
# extended (not replaced) so every earlier test explicitly disables the
# layer via the new `involvement_path=None` default -- see its docstring.
# ---------------------------------------------------------------------------

_INVOLVEMENT_BOX = (slice(32, 38), slice(32, 38), slice(16, 22))  # entirely inside Frontal_L


def _write_involvement_case(tmp_path: Path, case_id: str) -> tuple[Path, Path, Path]:
    """A single WT box entirely inside Frontal_L (tissue=WM there, see `_build_atlas_dir`),
    nowhere near the ventricle group (Precentral_L) -- chosen so every
    `involvement_profile` field is hand-computable: the box contributes its
    full voxel count to deep-white-matter overlap and zero to ventricle
    overlap, its tissue is entirely white matter, and its centroid lands
    (after round-half-to-even) exactly inside Frontal_L, giving an EXACT
    epicentre at 0.0 mm.

    Returns:
        `(prep_dir, splits_path, eval_dir)`.
    """
    prep_dir = tmp_path / f"prep_{case_id}"
    case_dir = ensure_dir(prep_dir / case_id)
    _write_meta(case_dir, case_id)

    array = np.zeros(ATLAS_SHAPE, dtype=np.uint8)
    array[_INVOLVEMENT_BOX] = 2  # ED -> completes WT via region_mask
    eval_dir = tmp_path / f"eval_{case_id}"
    predictions_dir = ensure_dir(eval_dir / "predictions")
    np.save(predictions_dir / f"{case_id}.npy", array)

    splits_path = tmp_path / f"splits_{case_id}.yaml"
    _write_splits(splits_path, [case_id])
    return prep_dir, splits_path, eval_dir


def test_involvement_columns_hand_computed_when_enabled(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    involvement_path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        involvement_path,
        ventricle_structures=["Precentral_L"],
        deep_wm_structures=["Frontal_L"],
    )
    case_id = "INVOLVE"
    prep_dir, splits_path, eval_dir = _write_involvement_case(tmp_path, case_id)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        involvement_path=involvement_path,
    )

    _anatomy_csv, summary_csv = run_localize(cfg)
    summary = pd.read_csv(summary_csv)
    assert len(summary) == 1
    row = summary.iloc[0]

    n_voxels = 6 * 6 * 6  # the involvement box, 216 voxels at 1mm^3 spacing
    frontal_l_voxels = 30 * 30 * 15  # _FRONTAL_L_SLICE, 13500
    precentral_l_voxels = 19 * 10 * 10  # _PRECENTRAL_L_SLICE, 1900

    assert row["deep_wm_overlap_mm3"] == pytest.approx(float(n_voxels))
    assert row["deep_wm_frac_of_tumour"] == pytest.approx(1.0)
    assert row["deep_wm_frac_of_group"] == pytest.approx(n_voxels / frontal_l_voxels)
    assert bool(row["deep_wm_contact"]) is True  # 216 mm3 >= min_overlap_mm3 (50.0)

    assert row["ventricle_overlap_mm3"] == pytest.approx(0.0)
    assert row["ventricle_frac_of_tumour"] == pytest.approx(0.0)
    assert row["ventricle_frac_of_group"] == pytest.approx(0.0 / precentral_l_voxels)
    assert bool(row["ventricle_contact"]) is False

    # The box's tissue is entirely WM (see _build_atlas_dir), so the four
    # tissue fractions sum to 1.0 with all of it in white_matter.
    assert row["white_matter_frac_of_tumour"] == pytest.approx(1.0)
    assert row["cortical_frac_of_tumour"] == pytest.approx(0.0)
    assert row["csf_frac_of_tumour"] == pytest.approx(0.0)
    assert row["outside_tissue_frac_of_tumour"] == pytest.approx(0.0)

    assert row["epicentre_structure"] == "Frontal_L"
    assert bool(row["epicentre_exact"]) is True
    assert row["epicentre_distance_mm"] == pytest.approx(0.0)
    assert row["epicentre_laterality"] == "L"
    assert row["epicentre_side"] == "right"
    assert row["epicentre_lobe"] == "frontal"


def test_involvement_disabled_no_columns_and_enabling_is_additive_only(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    involvement_path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        involvement_path,
        ventricle_structures=["Precentral_L"],
        deep_wm_structures=["Frontal_L"],
    )
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)

    out_off = tmp_path / "out_off"
    cfg_off = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        out_off,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
    )
    anatomy_off, summary_off = run_localize(cfg_off)

    out_on = tmp_path / "out_on"
    cfg_on = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        out_on,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        involvement_path=involvement_path,
    )
    anatomy_on, summary_on = run_localize(cfg_on)

    df_off = pd.read_csv(summary_off)
    df_on = pd.read_csv(summary_on)

    involvement_columns = {
        "ventricle_overlap_mm3",
        "ventricle_frac_of_tumour",
        "ventricle_frac_of_group",
        "ventricle_contact",
        "deep_wm_overlap_mm3",
        "deep_wm_frac_of_tumour",
        "deep_wm_frac_of_group",
        "deep_wm_contact",
        "cortical_frac_of_tumour",
        "white_matter_frac_of_tumour",
        "csf_frac_of_tumour",
        "outside_tissue_frac_of_tumour",
        "epicentre_structure",
        "epicentre_exact",
        "epicentre_distance_mm",
        "epicentre_laterality",
        "epicentre_side",
        "epicentre_lobe",
    }
    assert involvement_columns.isdisjoint(df_off.columns)
    assert involvement_columns.issubset(df_on.columns)

    # Additive-only: every column the disabled run wrote has an IDENTICAL
    # value in the enabled run, aligned by case_id -- same approach as
    # tests/test_evaluate_script.py's boundary-metrics additive-only check.
    shared_columns = [c for c in df_off.columns if c in df_on.columns]
    left = df_off[shared_columns].sort_values("case_id").reset_index(drop=True)
    right = df_on[shared_columns].sort_values("case_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

    # anatomy.csv (the per-structure table) is untouched by this layer either way.
    assert anatomy_off.read_bytes() == anatomy_on.read_bytes()


def test_missing_involvement_key_runs_with_layer_off(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
        omit_involvement_key=True,
    )
    assert "involvement" not in cfg.analysis.localize
    assert resolve_involvement(cfg.analysis.localize) is None

    _anatomy_csv, summary_csv = run_localize(cfg)
    summary = pd.read_csv(summary_csv)
    assert "ventricle_contact" not in summary.columns


def test_localize_config_yaml_records_involvement_settings_and_version(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    involvement_path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        involvement_path,
        ventricle_structures=["Precentral_L"],
        deep_wm_structures=["Frontal_L"],
        version=7,
    )
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        involvement_path=involvement_path,
    )
    run_localize(cfg)

    record = read_yaml(output_dir / "localize_config.yaml")
    assert record["involvement"]["enabled"] is True
    assert record["involvement"]["groups_map"] == str(involvement_path)
    assert record["involvement"]["min_overlap_mm3"] == pytest.approx(50.0)
    assert record["involvement"]["version"] == 7


def test_localize_config_yaml_records_involvement_disabled(tmp_path: Path) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    prep_dir, splits_path, _eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="label",
    )
    run_localize(cfg)

    record = read_yaml(output_dir / "localize_config.yaml")
    assert record["involvement"] == {"enabled": False}


def test_involvement_groups_built_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atlas_root, eloq_path, lobe_path = _build_knowledge_fixtures(tmp_path)
    involvement_path = tmp_path / "involvement_groups.yaml"
    _write_involvement_yaml(
        involvement_path,
        ventricle_structures=["Precentral_L"],
        deep_wm_structures=["Frontal_L"],
    )
    prep_dir, splits_path, eval_dir = _standard_split(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _compose_cfg(
        tmp_path,
        prep_dir,
        splits_path,
        output_dir,
        atlas_root,
        eloq_path,
        lobe_path,
        source="prediction",
        eval_dir=eval_dir,
        involvement_path=involvement_path,
    )

    calls = {"n": 0}
    original_load_involvement_groups = localize_script.load_involvement_groups

    def counting_load_involvement_groups(*args, **kwargs):
        calls["n"] += 1
        return original_load_involvement_groups(*args, **kwargs)

    monkeypatch.setattr(
        localize_script, "load_involvement_groups", counting_load_involvement_groups
    )

    run_localize(cfg)  # 3 cases in CASE_IDS

    assert calls["n"] == 1
