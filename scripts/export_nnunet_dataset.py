"""Convert our frozen BraTS2021 train/test split into nnU-Net v2's raw dataset layout.

Task G1 (`docs/research/master_plan.md` Section 4.3): the Gate A strong-baseline
comparator trains `nnunet_v2_3dfullres` on the SAME cases as `neurovision`, and is
scored through OUR metric path -- never nnU-Net's own internal split or metrics. That
means nnU-Net's `imagesTr`/`labelsTr`/`imagesTs`/`dataset.json`/`splits_final.json`
must be built from our own frozen `configs/data/splits.yaml`, not from nnU-Net's usual
"drop everything in imagesTr and let it pick its own folds" workflow.

This is a "default nnU-Net recipe, unmodified" comparator, not our own variant: the
label remap (`remap_brats_label`) is a faithful port of nnU-Net v2.8.1's own
`Dataset137_BraTS21.copy_BraTS_segmentation_and_convert_labels_to_nnUNet`, and
`build_splits_final` calls nnU-Net's own `generate_crossval_split` for real rather than
re-implementing scikit-learn's KFold by hand.

Raw BraTS2021 case directories (once downloaded from Kaggle
`dschettler8845/brats-2021-task1`) hold five **uncompressed** `.nii` files per case,
2020-style suffixes (`<case_id>_t1.nii`, `_t1ce`, `_t2`, `_flair`, `_seg` -- confirmed
in `docs/lessons.md`), and the exact nesting depth under `--raw-dir` is not something to
assume: this project's own `notebooks/kaggle_train.ipynb` cell 9 already had to stop
assuming a fixed `/kaggle/input/<slug>` depth after a real Kaggle download put
everything one level deeper than expected. `discover_case_dir` does the equivalent
defensive discovery here -- a handful of bounded nesting depths, never an unbounded
`**`.

`--out`'s basename is the single source of truth for the dataset's identity (id + name):
validated against nnU-Net's own `Dataset<3 digits>_<name>` convention, and used verbatim
as `dataset.json`'s `"name"` field. There is no separate `--dataset-id`/`--dataset-name`
flag on purpose.

Plain `argparse`, not Hydra: this is a one-shot operational conversion tool (raw dir in,
nnU-Net-formatted dir out) with no config group to select, no sweep, and nothing to log
to W&B -- exactly `scripts/package_for_kaggle.py`'s own reasoning for the same choice.

Example usage:

    python scripts/export_nnunet_dataset.py \\
        --raw-dir /path/to/brats2021_raw \\
        --splits configs/data/splits.yaml \\
        --out outputs/nnunet_raw/Dataset901_NeuroVisionXBraTS21

Dry run first (checks every case's raw files exist, writes nothing):

    python scripts/export_nnunet_dataset.py \\
        --raw-dir /path/to/brats2021_raw --splits configs/data/splits.yaml \\
        --out outputs/nnunet_raw/Dataset901_NeuroVisionXBraTS21 --dry-run

After a real (non-dry-run) export, and after running `nnUNetv2_plan_and_preprocess`,
copy the `splits_final.json` this script writes at the top level of `--out` into
`nnUNet_preprocessed/<the same Dataset folder name>/splits_final.json` BEFORE running
`nnUNetv2_train` -- see `build_splits_final`'s docstring for why nnU-Net needs it there
and not in `nnUNet_raw`.

This is a CLI entry point, so the terminal summary (case counts, output path, the
`splits_final.json` reminder) is logged via `logging` at INFO level, same as
`package_for_kaggle.py`'s own `logger.info(...)` calls -- a human-facing report through
`logging`, not a bare `print`.

This script only ever runs in `.venv-analysis` (`nnunetv2` is not, and must never become,
a dependency of the training lockfile `requirements.txt`) -- same environment
`scripts/evaluate.py` uses when lesion-wise scoring is enabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from neurovision.data.dataset import load_splits
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# nnU-Net's own required dataset-folder naming convention: "Dataset<3-digit
# id>_<name, no whitespace>". --out's basename is the single source of truth
# for the dataset's identity -- there is deliberately no separate
# --dataset-id/--dataset-name flag.
_DATASET_NAME_RE = re.compile(r"^Dataset(\d{3})_(\S+)$")

# Raw BraTS2021 filenames use 2020-style suffixes ("<case_id>_t1.nii", not
# "_0000.nii.gz", and uncompressed -- see docs/lessons.md). Every pattern
# below is anchored with the "." that follows the modality token specifically
# so "<case_id>_t1.nii*" can never match "<case_id>_t1ce.nii*" as a substring
# -- CLAUDE.md's own trap: a short token matched against a path is a
# substring of some longer word there ("_t1" in name is true for "_t1ce").
_MODALITY_SUFFIXES = {
    "t1": "_t1.nii",
    "t1ce": "_t1ce.nii",
    "t2": "_t2.nii",
    "flair": "_flair.nii",
    "seg": "_seg.nii",
}

# nnU-Net's own channel order for THIS dataset (its official BraTS21
# converter's convention). Independent of, and not required to match, this
# project's own channel order elsewhere -- nnU-Net only needs internal
# self-consistency between imagesTr and imagesTs, which building both from
# this same list provides by construction.
_CHANNEL_ORDER = ("t1", "t1ce", "t2", "flair")

# BraTS's raw segmentation label values. nnU-Net wants a continuous 0..3
# remap -- see remap_brats_label.
_VALID_RAW_LABELS = frozenset({0, 1, 2, 4})

# How many plausible nesting levels under --raw-dir to search for each case's
# files. Bounded rather than an unbounded "**": this project's own
# notebooks/kaggle_train.ipynb cell 9 already had to stop assuming a fixed
# /kaggle/input/<slug> depth after a real Kaggle download landed everything
# one level deeper than expected. A handful of explicit depths covers that
# same class of surprise without an unbounded recursive glob's cost (or its
# silent traversal into symlinked directories).
_MAX_GLOB_DEPTH = 6


def _glob_at_depths(
    root: Path, filename_pattern: str, max_depth: int = _MAX_GLOB_DEPTH
) -> list[Path]:
    """Globs for a filename pattern at a handful of plausible nesting depths.

    Args:
        root: Directory to search under.
        filename_pattern: A glob pattern for the filename itself, e.g.
            `"BraTS2021_01417_t1.nii*"`.
        max_depth: Deepest nesting level to try (0 = directly under `root`).

    Returns:
        Every distinct match (deduped by resolved path), sorted for
        determinism. Empty if `root` does not exist.
    """
    if not root.is_dir():
        return []
    matches: dict[str, Path] = {}
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + [filename_pattern]) if depth else filename_pattern
        for p in root.glob(pattern):
            if p.is_file():
                matches.setdefault(str(p.resolve()), p)
    return sorted(matches.values(), key=str)


def discover_case_dir(raw_dir: Path, case_id: str) -> Path:
    """Finds the raw source directory holding one case's NIfTI files.

    Located by searching for the case's own `_t1.nii*` file at a handful of
    plausible nesting depths under `raw_dir` (see module docstring) -- a
    Kaggle-downloaded BraTS archive's exact folder depth is not something to
    hardcode.

    Args:
        raw_dir: Root to search under.
        case_id: BraTS case id, e.g. `"BraTS2021_01417"`.

    Returns:
        The directory containing that case's `_t1.nii*` file.

    Raises:
        FileNotFoundError: If no `_t1.nii*` file for `case_id` is found under
            `raw_dir`.
        ValueError: If `_t1.nii*` files for `case_id` are found under more
            than one directory -- an ambiguous source, named rather than
            silently guessed at.
    """
    pattern = f"{case_id}_t1.nii*"
    hits = _glob_at_depths(raw_dir, pattern)
    if not hits:
        raise FileNotFoundError(
            f"Case '{case_id}': no file matching '{pattern}' found under "
            f"{raw_dir.resolve()} (searched up to {_MAX_GLOB_DEPTH} nesting levels deep)."
        )
    case_dirs = sorted({hit.parent.resolve() for hit in hits}, key=str)
    if len(case_dirs) > 1:
        raise ValueError(
            f"Case '{case_id}' is ambiguous: '{pattern}' was found under "
            f"{len(case_dirs)} different directories: {[str(d) for d in case_dirs]}. "
            "Resolve the duplication under --raw-dir before exporting."
        )
    return case_dirs[0]


def discover_case_files(raw_dir: Path, case_id: str, require_seg: bool) -> dict[str, Path]:
    """Finds every required raw NIfTI file for one case.

    Args:
        raw_dir: Root to search under.
        case_id: BraTS case id.
        require_seg: If True, also require and return the `_seg` file (train
            cases). If False, only the four image modalities are required
            (test cases -- their ground truth is deliberately never copied
            into the nnU-Net export; see module docstring).

    Returns:
        Dict mapping modality key (`"t1"`, `"t1ce"`, `"t2"`, `"flair"`, and
        `"seg"` if `require_seg`) to that file's path.

    Raises:
        FileNotFoundError: If the case directory, or any required file
            within it, cannot be found. Names the case id and the missing
            modality.
        ValueError: If the case's `_t1.nii*` file is ambiguous (see
            `discover_case_dir`).
    """
    case_dir = discover_case_dir(raw_dir, case_id)
    modalities = list(_CHANNEL_ORDER) + (["seg"] if require_seg else [])
    files: dict[str, Path] = {}
    for key in modalities:
        pattern = f"{case_id}{_MODALITY_SUFFIXES[key]}*"
        hits = sorted(p for p in case_dir.glob(pattern) if p.is_file())
        if not hits:
            raise FileNotFoundError(
                f"Case '{case_id}': missing required file matching '{pattern}' "
                f"(modality '{key}') in {case_dir.resolve()}."
            )
        files[key] = hits[0]
    return files


def remap_brats_label(src: Path, dst: Path, case_id: str) -> None:
    """Reproduces nnU-Net's own BraTS label remap: `{0, 1, 2, 4} -> {0, 1, 2, 3}`.

    Faithful port of nnU-Net v2.8.1's
    `Dataset137_BraTS21.copy_BraTS_segmentation_and_convert_labels_to_nnUNet`.
    nnU-Net wants continuous label values; BraTS ships 0 (background), 1
    (necrotic core), 2 (edema), 4 (enhancing tumor). The remap is exactly:
    4 -> 3, 2 -> 1, 1 -> 2, 0 -> 0. This is a "default nnU-Net recipe,
    unmodified" comparator, so the remap must match nnU-Net's own script
    exactly, not be reinvented.

    Args:
        src: Path to the raw `_seg.nii` (or `.nii.gz`) file.
        dst: Destination path for the remapped, compressed label NIfTI.
            Parent directory is created if missing.
        case_id: Used only to name the case in a raised error.

    Raises:
        RuntimeError: If `src` contains any value outside `{0, 1, 2, 4}`.
            Preserves nnU-Net's own check verbatim -- does not loosen it.
    """
    img = sitk.ReadImage(str(src))
    img_npy = sitk.GetArrayFromImage(img)
    uniques = np.unique(img_npy)
    for u in uniques:
        if int(u) not in _VALID_RAW_LABELS:
            raise RuntimeError(
                f"Case '{case_id}': segmentation at {src} contains unexpected label "
                f"value {int(u)} (expected only {sorted(_VALID_RAW_LABELS)})."
            )
    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    ensure_dir(dst.parent)
    sitk.WriteImage(img_corr, str(dst))


def _write_image_channel(src: Path, dst: Path) -> None:
    """Reads one raw NIfTI and writes it back out gzip-compressed at `dst`.

    A uniform code path regardless of whether `src` is already `.nii.gz`:
    this both performs the `.nii` -> `.nii.gz` compression nnU-Net expects,
    and avoids a special case for a source that happens to already be
    gzipped -- `sitk.ReadImage`/`WriteImage` handle both transparently.

    Args:
        src: Raw source NIfTI (`.nii` or `.nii.gz`).
        dst: Destination `.nii.gz` path. Parent directory is created if
            missing.
    """
    img = sitk.ReadImage(str(src))
    ensure_dir(dst.parent)
    sitk.WriteImage(img, str(dst))


def _convert_train_case(args: tuple[str, dict[str, Path], Path, Path]) -> str:
    """Converts one train case: four image channels plus the remapped label.

    A plain module-level function (not a lambda or closure) so it can be
    pickled and shipped to a worker process by `ProcessPoolExecutor` --
    `scripts/preprocess.py` uses the same shape for the same reason.

    Args:
        args: `(case_id, files, images_dir, labels_dir)`, bundled into one
            tuple since `executor.map` takes one argument per call.

    Returns:
        `case_id`, for the caller to confirm completion.
    """
    case_id, files, images_dir, labels_dir = args
    for i, key in enumerate(_CHANNEL_ORDER):
        _write_image_channel(files[key], images_dir / f"{case_id}_{i:04d}.nii.gz")
    remap_brats_label(files["seg"], labels_dir / f"{case_id}.nii.gz", case_id)
    return case_id


def _convert_test_case(args: tuple[str, dict[str, Path], Path]) -> str:
    """Converts one test case: four image channels only, NEVER a label.

    BraTS2021's public release ships `_seg` for every case, including the
    ones our split holds out as `test`, but nnU-Net must not see it here --
    copying it would defeat the point of a held-out inference set (see
    module docstring). `scripts/evaluate.py` scores nnU-Net's predictions
    against the ground truth this project already holds elsewhere.

    Args:
        args: `(case_id, files, images_dir)`.

    Returns:
        `case_id`, for the caller to confirm completion.
    """
    case_id, files, images_dir = args
    for i, key in enumerate(_CHANNEL_ORDER):
        _write_image_channel(files[key], images_dir / f"{case_id}_{i:04d}.nii.gz")
    return case_id


def build_dataset_json(dataset_folder_name: str, num_training: int) -> dict[str, Any]:
    """Builds nnU-Net v2's `dataset.json`, matching `generate_dataset_json`'s output shape.

    Reproduces nnU-Net v2.8.1's own BraTS21 converter's `dataset.json` schema
    exactly (channel order, label groupings, region class order), without
    importing nnU-Net's `generate_dataset_json` helper -- not needed for a
    five-key dict, and this keeps the schema pinned to what is documented
    here rather than to whatever that helper's internals do next release.

    Args:
        dataset_folder_name: `--out`'s own basename, e.g.
            `"Dataset901_NeuroVisionXBraTS21"` -- used verbatim as the
            `"name"` field, so the dataset's identity has exactly one source
            of truth.
        num_training: Number of train cases (`len(train_ids)`).

    Returns:
        The `dataset.json` object.
    """
    return {
        "channel_names": {"0": "T1", "1": "T1ce", "2": "T2", "3": "Flair"},
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3],
            "enhancing tumor": [3],
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3],
        "name": dataset_folder_name,
    }


def build_splits_final(train_ids: list[str]) -> list[dict[str, list[str]]]:
    """Builds nnU-Net's cross-validation `splits_final.json`, via nnU-Net's own algorithm.

    Calls `nnunetv2.utilities.crossval_split.generate_crossval_split` for
    real (imported inside this function, not at module scope, since
    `nnunetv2` lives only in `.venv-analysis` -- following the exact
    deferred-import idiom `neurovision.data.clinical_resample` uses for
    `brainles_preprocessing`, so importing this module never requires
    `nnunetv2` to be installed) rather than hand-reimplementing scikit-learn's
    `KFold` -- nnU-Net's own `nnUNetTrainer.do_split` calls this exact
    function, with these exact arguments, when generating a fresh split.

    Writes all 5 folds: nnU-Net's own docstring warns that requesting fewer
    folds than 5 silently falls back to a random 80:20 split instead, so
    under-writing this file would be a correctness bug, not a simplification.

    IMPORTANT (see module docstring): nnU-Net reads `splits_final.json` from
    its *preprocessed* dataset folder (`nnUNet_preprocessed/<Dataset
    folder>/`), created later by `nnUNetv2_plan_and_preprocess`, which does
    not exist yet when this script runs. This function's caller therefore
    writes the file at the top level of `--out` (`nnUNet_raw/<Dataset
    folder>/splits_final.json`) as a sibling to `dataset.json` -- copy it into
    the preprocessed folder yourself, after `nnUNetv2_plan_and_preprocess` and
    before `nnUNetv2_train`.

    Args:
        train_ids: Train case ids. Sorted by the caller before this is
            called, matching how `do_split` calls `generate_crossval_split`
            on `sorted(...)`.

    Returns:
        List of 5 dicts, each `{"train": [...], "val": [...]}`, plain
        `str`-valued lists (nnU-Net's own function returns numpy string
        arrays internally).

    Raises:
        ValueError: Propagated from scikit-learn's `KFold` if `train_ids`
            has fewer than 5 entries (5-fold cross-validation needs at least
            5 samples) -- this module never catches or works around it, the
            same as nnU-Net's own script would not.
    """
    from nnunetv2.utilities.crossval_split import generate_crossval_split

    folds = generate_crossval_split(train_ids, seed=12345, n_splits=5)
    return [
        {"train": [str(cid) for cid in fold["train"]], "val": [str(cid) for cid in fold["val"]]}
        for fold in folds
    ]


def validate_out_name(out_dir: Path) -> str:
    """Validates `--out`'s basename against nnU-Net's own dataset-folder naming convention.

    Args:
        out_dir: The `--out` path.

    Returns:
        The validated basename, e.g. `"Dataset901_NeuroVisionXBraTS21"`.

    Raises:
        ValueError: If the basename does not match `Dataset<3 digits>_<name>`.
    """
    basename = out_dir.name
    if not _DATASET_NAME_RE.match(basename):
        raise ValueError(
            "--out's basename must match nnU-Net's own naming convention "
            "'Dataset<3 digits>_<name>' (e.g. 'Dataset901_NeuroVisionXBraTS21'), got "
            f"{basename!r} (--out={out_dir})."
        )
    return basename


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Root to search for raw BraTS2021 case directories (nesting depth not assumed).",
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="Path to the split YAML (load_splits format)."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="The exact Dataset<ID>_<Name> folder to build, "
        "e.g. outputs/nnunet_raw/Dataset901_NeuroVisionXBraTS21.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Confirm every case's raw files exist under --raw-dir; write nothing.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Allow reusing/overwriting a non-empty --out."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel case-conversion workers via ProcessPoolExecutor. 1 (default) = serial.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validates, discovers, converts, and reports -- the whole script, minus CLI/exit plumbing.

    Args:
        args: Parsed CLI arguments (see `build_arg_parser`).

    Returns:
        A summary dict: `{"train": int, "test": int, "dry_run": bool}` for a
        dry run, or additionally `"out_dir"`, `"dataset_json_path"`,
        `"splits_final_path"` for a real export.

    Raises:
        ValueError: If `--out`'s basename does not match nnU-Net's dataset
            naming convention (checked before anything else, so this never
            touches the filesystem first).
        FileNotFoundError: If a case's raw files cannot be found (dry run or
            real run alike) -- names the case id and the missing modality.
        FileExistsError: If `--out` already exists and is non-empty, and
            `--force` was not given.
        RuntimeError: If a train case's segmentation contains an unexpected
            label value (propagated from `remap_brats_label`).
    """
    out_dir = Path(args.out)
    # Validated FIRST, before touching the filesystem at all: --out's
    # basename is the one source of truth for this dataset's identity.
    dataset_folder_name = validate_out_name(out_dir)

    raw_dir = Path(args.raw_dir)
    splits_path = Path(args.splits)
    splits = load_splits(splits_path)
    train_ids = list(splits["train"])
    test_ids = list(splits["test"])
    logger.info(
        "Loaded split %s: %d train, %d val (NEVER exported -- not part of this comparison), "
        "%d test.",
        splits_path,
        len(train_ids),
        len(splits["val"]),
        len(test_ids),
    )

    if args.dry_run:
        logger.info(
            "--dry-run: checking that every case's raw files exist under %s ...",
            raw_dir.resolve(),
        )
        # Fail loud on the FIRST missing case -- mirrors package_for_kaggle.py's
        # own "confirm everything is present before spending anything"
        # philosophy. Not a collect-and-report-all-misses pass.
        for case_id in train_ids:
            discover_case_files(raw_dir, case_id, require_seg=True)
        for case_id in test_ids:
            discover_case_files(raw_dir, case_id, require_seg=False)
        logger.info(
            "--dry-run OK: found all raw files for %d train + %d test case(s) under %s. "
            "Nothing written.",
            len(train_ids),
            len(test_ids),
            raw_dir.resolve(),
        )
        return {"train": len(train_ids), "test": len(test_ids), "dry_run": True}

    # Discover every case's files BEFORE touching --out at all -- a missing
    # or ambiguous case must fail loud before the previous export (which
    # --force is about to rmtree) is destroyed. Ordering this before the
    # out-dir check below is what keeps a --force re-run against a typo'd or
    # incomplete --raw-dir from deleting a valid, possibly expensive-to-
    # rebuild prior export and then raising with nothing left to show for it.
    logger.info(
        "Discovering raw files for %d train + %d test case(s) under %s ...",
        len(train_ids),
        len(test_ids),
        raw_dir.resolve(),
    )
    train_files = {cid: discover_case_files(raw_dir, cid, require_seg=True) for cid in train_ids}
    test_files = {cid: discover_case_files(raw_dir, cid, require_seg=False) for cid in test_ids}

    out_is_nonempty = out_dir.is_dir() and any(out_dir.iterdir())
    if out_is_nonempty:
        if not args.force:
            raise FileExistsError(
                f"--out {out_dir.resolve()} already exists and is not empty. "
                "Pass --force to overwrite it, or choose a different --out."
            )
        logger.warning("Overwriting non-empty --out %s (--force given).", out_dir.resolve())
        shutil.rmtree(out_dir)

    images_tr = ensure_dir(out_dir / "imagesTr")
    labels_tr = ensure_dir(out_dir / "labelsTr")
    images_ts = ensure_dir(out_dir / "imagesTs")

    train_work = [(cid, train_files[cid], images_tr, labels_tr) for cid in train_ids]
    test_work = [(cid, test_files[cid], images_ts) for cid in test_ids]

    workers = int(args.workers)
    if workers > 1:
        # ProcessPoolExecutor, matching scripts/preprocess.py's own choice
        # over multiprocessing.Pool. Kept as a plain serial loop at
        # workers=1 -- simple and obviously correct, since that is the path
        # the test suite exercises.
        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_convert_train_case, train_work))
            list(executor.map(_convert_test_case, test_work))
    else:
        for item in train_work:
            _convert_train_case(item)
        for item in test_work:
            _convert_test_case(item)

    dataset_json = build_dataset_json(dataset_folder_name, len(train_ids))
    write_json(dataset_json, out_dir / "dataset.json")

    splits_final_path = out_dir / "splits_final.json"
    splits_final = build_splits_final(sorted(train_ids))
    write_json(splits_final, splits_final_path)

    logger.info("=" * 70)
    logger.info("nnU-Net export summary")
    logger.info("  Dataset:      %s", dataset_folder_name)
    logger.info("  Train cases:  %d  (imagesTr/ + labelsTr/)", len(train_ids))
    logger.info(
        "  Test cases:   %d  (imagesTs/ only -- no labels; held out for scripts/evaluate.py)",
        len(test_ids),
    )
    logger.info("  Output:       %s", out_dir.resolve())
    logger.info(
        "  IMPORTANT: after `nnUNetv2_plan_and_preprocess -d <id>`, copy %s into "
        "nnUNet_preprocessed/%s/splits_final.json BEFORE running nnUNetv2_train -- "
        "nnU-Net only auto-generates a splits_final.json when one is not already "
        "present there.",
        splits_final_path,
        dataset_folder_name,
    )
    logger.info("=" * 70)

    return {
        "train": len(train_ids),
        "test": len(test_ids),
        "out_dir": out_dir,
        "dataset_json_path": out_dir / "dataset.json",
        "splits_final_path": splits_final_path,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list, or None to use `sys.argv[1:]`.

    Returns:
        0 on success, 1 on any failure.
    """
    setup_logging(level="INFO")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception:
        logger.exception("nnU-Net dataset export failed:")
        return 1
    return 0


if __name__ == "__main__":
    # macOS's default multiprocessing start method is "spawn": each worker
    # process re-imports this module from scratch rather than inheriting the
    # parent's memory via fork. Keeping all real work behind this guard (and
    # inside main()) is what makes --workers > 1 behave the same on the Mac
    # as on Linux -- see scripts/preprocess.py for the identical reasoning.
    sys.exit(main())
