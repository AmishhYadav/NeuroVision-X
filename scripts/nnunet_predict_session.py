"""Runs the Gate A PREDICTION session: nnU-Net inference on the 189 held-out test cases.

Implements Gate A Amendment 1 (`docs/research_docs/preregistrations/
preregistration_strong_baseline.md`), items 6-8:

  - **Checkpoint used: `checkpoint_final.pth`, never `checkpoint_best.pth`.**
    `checkpoint_best` is selected on training cases in fold "all" -- it is not
    a fair validation signal, so predicting from it would bias the comparison
    in nnU-Net's favour. This script refuses to run if only `checkpoint_best`
    exists.
  - **Two arms.** nnU-Net's own defaults (Gaussian sliding window, mirroring
    ON) are the PRIMARY, pre-registered arm. A SECOND, purely descriptive arm
    runs with `--disable_tta` (mirroring off), so a reader can see how much of
    any Dice gap is test-time augmentation -- it never enters the Holm family.
    `--arms` preserves order, primary first, so the primary arm's cost and
    correctness are never gated on the secondary arm having run at all.
  - **Why gzip.** `scripts/score_nnunet.py::_check_completeness` RAISES if any
    case in the requested split has no matching `<case_id>.nii.gz` (an extra,
    unrequested file is only logged and ignored, never raised on) -- so every
    case this script hands off must exist as a `.nii.gz`. But Kaggle
    DECOMPRESSES `.nii.gz` to `.nii` on ingest (confirmed against
    `outputs/kaggle_kernels/gatea-prep/gatea_prep.ipynb`, which had to rewrite
    `dataset.json`'s `file_ending` to match), so both the raw test images AND
    nnU-Net's own predictions land on disk as `.nii`, not `.nii.gz`. This
    script gzips nnU-Net's raw output bytes with the stdlib `gzip` module
    (never re-encoding through nibabel,
    which could silently change the header or data layout) and verifies, via
    nibabel, that the gzipped file loads to the same array as the `.nii` it
    came from -- for the first case of each arm only (cheap sanity, not a
    189-case re-verification).

Every check below runs, and can raise, BEFORE a single prediction is made:
locating the one trained model folder, the checkpoint's identity (epoch,
trainer name, finite training loss), locating the one `imagesTs` directory,
and confirming its case ids match `configs/data/splits.yaml`'s `test` list
exactly. A missing or extra case, or the wrong file ending, is refused rather
than silently scored on a partial set -- CLAUDE.md's own trap #9: a probe (or
a preflight check) must be built to reach the failure condition, not merely
run.

`nnunetv2` is never imported here, at module scope or otherwise -- it lives
only in `.venv-analysis` (see `requirements-analysis.txt`). `nnUNetv2_predict_
from_modelfolder` is invoked as a subprocess (the console script an
`.venv-analysis`-equipped Kaggle session installs via `pip install --no-deps
nnunetv2==2.8.1`, matching `scripts/nnunet_session.py`'s own
"the training/prediction venv is not this process's venv" reasoning), so this
whole module -- including its tests -- runs unmodified in the main `.venv`, on
CPU, with no nnunetv2 install at all.

Its CLI flags (`-i`, `-o`, `-m`, `-f`, `-chk`, `-device`, `--disable_tta`) were
read directly out of the installed nnunetv2==2.8.1 source in `.venv-analysis`
(`nnunetv2/inference/predict_from_raw_data.py::predict_entry_point_modelfolder`,
the function `dist-info/entry_points.txt` names as the
`nnUNetv2_predict_from_modelfolder` console script's entry point) -- not
guessed at or taken from documentation that could be stale.

Example usage (primary + secondary arm, both against the 189 test cases):

    python scripts/nnunet_predict_session.py \\
        --model-search-root /kaggle/input --images-search-root /kaggle/input \\
        --out-root /kaggle/working/nnunet_predictions \\
        --splits configs/data/splits.yaml --device cuda --require-cuda \\
        --summary-json /kaggle/working/gatea_predict_summary.json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import nibabel as nib
import numpy as np
import torch

from neurovision.data.dataset import load_splits
from neurovision.utils.io import read_json, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# nnU-Net's own results-tree naming for this recipe (Amendment 1: trainer
# nnUNetTrainer, plans nnUNetPlans, configuration 3d_fullres, fold all).
_TRAINER_DIR_NAME = "nnUNetTrainer__nnUNetPlans__3d_fullres"
_FOLD_NAME = "fold_all"
_CHECKPOINT_NAME = "checkpoint_final.pth"

_IMAGES_TS_DIR_NAME = "imagesTs"

# How many plausible nesting levels to search under a Kaggle input mount.
# Bounded rather than an unbounded "**" glob -- mirrors
# `scripts/nnunet_session.py::_MAX_SEARCH_DEPTH` exactly, same reasoning: a
# Kaggle dataset does not reliably mount at a fixed depth.
_MAX_SEARCH_DEPTH = 6

# nnU-Net's own per-case channel suffixes for a 4-modality dataset
# (`scripts/export_nnunet_dataset.py::_CHANNEL_ORDER`, t1/t1ce/t2/flair).
_CHANNEL_SUFFIXES = ("0000", "0001", "0002", "0003")

# nnU-Net's own label convention for this dataset (module docstring / this
# project's remap): 1 = ED, 2 = NCR, 3 = ET. Background is 0. The REMAP to our
# own convention happens later, in `analysis/nnunet_import.py` -- never here.
_VALID_NNUNET_LABELS = frozenset({0, 1, 2, 3})

_ARM_TTA_ON = "tta_on"
_ARM_TTA_OFF = "tta_off"
_VALID_ARMS = (_ARM_TTA_ON, _ARM_TTA_OFF)


def _load_nnunet_session() -> ModuleType:
    """Loads `scripts/nnunet_session.py` so this module can reuse three helpers.

    `scripts/` is not an importable package (it holds CLI entry points, not
    library code -- see CLAUDE.md's repository layout), so there is no
    `from nnunet_session import ...` available. This project's own tests
    already load a sibling script this exact way (`tests/script_loader.py`);
    this production module does the same thing, at runtime, so it never has
    to re-implement `find_prev_fold_dir` (bounded, dedup-by-realpath
    directory search), `read_checkpoint_meta` (checkpoint identity fields),
    or `_require_cuda_or_raise` (the sm_70+ CUDA smoke check).

    Returns:
        The executed `nnunet_session` module.
    """
    path = Path(__file__).resolve().parent / "nnunet_session.py"
    spec = importlib.util.spec_from_file_location("nnunet_session", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nnunet_session = _load_nnunet_session()
find_prev_fold_dir = _nnunet_session.find_prev_fold_dir
read_checkpoint_meta = _nnunet_session.read_checkpoint_meta
_require_cuda_or_raise = _nnunet_session._require_cuda_or_raise
_nnunetv2_version = _nnunet_session._nnunetv2_version


# ---------------------------------------------------------------------------
# 1. Locating the trained model folder
# ---------------------------------------------------------------------------


def find_model_folder(search_root: Path, max_depth: int = _MAX_SEARCH_DEPTH) -> Path:
    """Finds the one trained model folder holding `fold_all/checkpoint_final.pth`.

    Reuses `nnunet_session.find_prev_fold_dir`'s bounded, dedup-by-realpath
    search for a `<trainer_dir_name>/<fold_name>` directory -- the same
    ambiguity check ("more than one candidate" raises, naming every hit)
    already tested and trusted in the training driver -- then adds the one
    requirement prediction needs on top: the fold directory found must
    actually hold `checkpoint_final.pth`. Amendment 1 item 6 forbids ever
    predicting from `checkpoint_best.pth` (selected on training cases in fold
    "all", so it is not a fair validation signal).

    Args:
        search_root: Root to search under (e.g. `/kaggle/input`).
        max_depth: Deepest nesting level to try.

    Returns:
        The model folder (the trainer dir, i.e. `fold_dir.parent`) -- the
        `-m` argument `nnUNetv2_predict_from_modelfolder` expects.

    Raises:
        FileNotFoundError: If zero, or more than one,
            `<trainer_dir_name>/<fold_name>` directory is found under
            `search_root`, or if the one found has no `checkpoint_final.pth`
            (names the checkpoint file(s) that DO exist there instead, e.g.
            `checkpoint_best.pth` only).
    """
    fold_dir = find_prev_fold_dir(search_root, _TRAINER_DIR_NAME, _FOLD_NAME, max_depth=max_depth)
    if fold_dir is None:
        raise FileNotFoundError(
            f"No directory named '{_TRAINER_DIR_NAME}/{_FOLD_NAME}' found under {search_root} "
            f"(within depth {max_depth}). Point --model-search-root at the mount holding the "
            "trained nnU-Net model folder."
        )
    checkpoint_path = fold_dir / _CHECKPOINT_NAME
    if not checkpoint_path.is_file():
        existing = sorted(p.name for p in fold_dir.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"{fold_dir} has no {_CHECKPOINT_NAME} (found: {existing}). Prediction requires "
            f"{_CHECKPOINT_NAME} -- checkpoint_best.pth is never used for prediction (it is "
            "selected on training cases in fold 'all', Amendment 1 item 6)."
        )
    return fold_dir.parent


def check_checkpoint_identity(meta: dict[str, Any], expect_epoch: int) -> None:
    """Raises if `checkpoint_final.pth`'s identity does not match what predict expects.

    Args:
        meta: `read_checkpoint_meta`'s output for `checkpoint_final.pth`.
        expect_epoch: The epoch training was expected to finish at
            (`--expect-epoch`).

    Raises:
        RuntimeError: If `current_epoch != expect_epoch`, `trainer_name` is
            not `"nnUNetTrainer"`, or the checkpoint's logged training losses
            carry a non-finite value (CLAUDE.md trap #2: an fp32-sized `eps`
            clamp is a no-op in fp16 and trains on a NaN loss for hours,
            undetected).
    """
    if meta.get("current_epoch") != expect_epoch:
        raise RuntimeError(
            f"checkpoint_final.pth is at current_epoch={meta.get('current_epoch')}, expected "
            f"{expect_epoch} (--expect-epoch). Training may not be complete, or the wrong "
            "checkpoint was attached."
        )
    if meta.get("trainer_name") != "nnUNetTrainer":
        raise RuntimeError(
            f"checkpoint_final.pth trainer_name={meta.get('trainer_name')!r}, expected "
            "'nnUNetTrainer'."
        )
    if not meta.get("train_loss_finite", False):
        raise RuntimeError(
            "checkpoint_final.pth's logged train_losses contain a non-finite value -- refusing "
            "to predict from it (CLAUDE.md trap #2)."
        )


# ---------------------------------------------------------------------------
# 2. Locating and validating the test images
# ---------------------------------------------------------------------------


def find_images_ts_dir(search_root: Path, max_depth: int = _MAX_SEARCH_DEPTH) -> Path:
    """Finds the one `imagesTs` directory under `search_root`.

    Mirrors `scripts/nnunet_session.py::find_preprocessed_dataset`'s bounded,
    dedup-by-realpath glob pattern (a handful of explicit nesting depths,
    never an unbounded `**`).

    Args:
        search_root: Root to search under (e.g. `/kaggle/input`).
        max_depth: Deepest nesting level to try.

    Returns:
        The single matching `imagesTs` directory.

    Raises:
        FileNotFoundError: If `search_root` does not exist, or zero or more
            than one `imagesTs` directory is found.
    """
    if not search_root.is_dir():
        raise FileNotFoundError(f"--images-search-root {search_root} does not exist.")
    patterns = ["/".join(["*"] * depth + [_IMAGES_TS_DIR_NAME]) for depth in range(max_depth + 1)]
    hits = sorted(
        {p.resolve() for pattern in patterns for p in search_root.glob(pattern) if p.is_dir()}
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '{_IMAGES_TS_DIR_NAME}' directory under {search_root} "
            f"(within depth {max_depth}); found {len(hits)}: {[str(h) for h in hits]}."
        )
    return hits[0]


def read_file_ending(model_folder: Path) -> str:
    """Reads the trained model's `dataset.json` `file_ending` field.

    Args:
        model_folder: The trainer dir, e.g.
            `.../nnUNetTrainer__nnUNetPlans__3d_fullres`, holding
            `dataset.json` as a sibling of `fold_all/` (nnU-Net writes it
            there once at `on_train_start`).

    Returns:
        The file ending, e.g. `".nii"` (Kaggle decompresses `.nii.gz` to
        `.nii` on ingest -- see module docstring) or `".nii.gz"`.

    Raises:
        FileNotFoundError: If `model_folder/dataset.json` does not exist.
        ValueError: If it has no (or an empty) `file_ending` field.
    """
    dataset_json_path = model_folder / "dataset.json"
    dataset_json = read_json(dataset_json_path)
    ending = dataset_json.get("file_ending")
    if not ending:
        raise ValueError(f"{dataset_json_path} has no 'file_ending' field.")
    return str(ending)


def discover_test_cases(images_ts_dir: Path, file_ending: str) -> dict[str, dict[str, Path]]:
    """Groups `imagesTs` files into per-case channel dicts, enforcing the file ending.

    Args:
        images_ts_dir: The `imagesTs` directory.
        file_ending: The model's expected file ending (e.g. `".nii"`), read
            via `read_file_ending`.

    Returns:
        `{case_id: {"0000": path, "0001": path, "0002": path, "0003": path}}`.

    Raises:
        ValueError: If `images_ts_dir` has no files, if any file in it does
            not end with `file_ending` (names the offending file(s) -- this
            is the "Kaggle decompressed .nii.gz to .nii but the model folder
            still says .nii.gz" failure mode named in the module docstring),
            if a file does not match the `<case_id>_000{0-3}<file_ending>`
            naming convention, or if any case does not have exactly channels
            0000-0003 (names the case and what was found instead).
    """
    all_files = sorted(p for p in images_ts_dir.iterdir() if p.is_file())
    if not all_files:
        raise ValueError(f"{images_ts_dir} has no files.")

    bad_ending = [p.name for p in all_files if not p.name.endswith(file_ending)]
    if bad_ending:
        raise ValueError(
            f"{len(bad_ending)} file(s) under {images_ts_dir} do not end with the model's "
            f"file_ending {file_ending!r} (from dataset.json): {bad_ending}."
        )

    # Anchored ("^...$") so a case id can never accidentally swallow another
    # case's channel token -- CLAUDE.md's own trap: a short token matched
    # against a path is a substring of some longer word there.
    pattern = re.compile(r"^(?P<case_id>.+)_(?P<channel>000[0-3])" + re.escape(file_ending) + r"$")
    cases: dict[str, dict[str, Path]] = {}
    for p in all_files:
        m = pattern.match(p.name)
        if m is None:
            raise ValueError(
                f"File {p.name!r} under {images_ts_dir} does not match the expected "
                f"'<case_id>_000{{0-3}}{file_ending}' naming convention."
            )
        cases.setdefault(m.group("case_id"), {})[m.group("channel")] = p

    for case_id, channels in cases.items():
        missing = sorted(set(_CHANNEL_SUFFIXES) - set(channels))
        if missing:
            raise ValueError(
                f"Case {case_id!r} under {images_ts_dir} is missing channel(s) {missing} "
                f"(found: {sorted(channels)})."
            )
    return cases


def check_case_ids_match_split(found_case_ids: set[str], expected_case_ids: Sequence[str]) -> None:
    """Requires `imagesTs`'s case ids to equal `configs/data/splits.yaml`'s `test` list exactly.

    Args:
        found_case_ids: Case ids `discover_test_cases` found under `imagesTs`.
        expected_case_ids: The split file's `test` list.

    Raises:
        ValueError: If any case is missing or extra (names every offending
            id -- exact set membership, never a substring match).
    """
    expected_set = set(expected_case_ids)
    missing = sorted(expected_set - found_case_ids)
    extra = sorted(found_case_ids - expected_set)
    if missing or extra:
        raise ValueError(
            "imagesTs's case ids do not match configs/data/splits.yaml's 'test' list exactly. "
            f"missing ({len(missing)}): {missing}. extra ({len(extra)}): {extra}."
        )


# ---------------------------------------------------------------------------
# 3. Building and running one arm's prediction command
# ---------------------------------------------------------------------------


def build_predict_command(
    images_ts_dir: Path,
    out_dir: Path,
    model_folder: Path,
    device: str,
    disable_tta: bool,
) -> list[str]:
    """Builds one `nnUNetv2_predict_from_modelfolder` command line.

    Flags verified against the installed nnunetv2==2.8.1 source in
    `.venv-analysis` (`nnunetv2/inference/predict_from_raw_data.py::
    predict_entry_point_modelfolder` -- the function
    `nnunetv2-2.8.1.dist-info/entry_points.txt` names as the
    `nnUNetv2_predict_from_modelfolder` console script's entry point): `-i`,
    `-o`, `-m`, `-f` (fold list -- `"all"` parses there as the literal string
    `"all"`, matching this project's fold `all`), `-chk` (its own default is
    already `checkpoint_final.pth`; named explicitly here anyway so a future
    upstream default change can never silently switch checkpoints), `-device`,
    and `--disable_tta` (mirroring is ON by default -- Amendment 1 item 7's
    primary arm -- and this flag alone selects the secondary, mirroring-off
    arm).

    Args:
        images_ts_dir: `imagesTs` directory (`-i`).
        out_dir: This arm's output directory (`-o`).
        model_folder: The trained model folder (`-m`).
        device: `"cuda"` or `"cpu"` (`-device`).
        disable_tta: True for the secondary (mirroring-off) arm.

    Returns:
        The full argv list for `subprocess.run`.
    """
    cmd = [
        "nnUNetv2_predict_from_modelfolder",
        "-i",
        str(images_ts_dir),
        "-o",
        str(out_dir),
        "-m",
        str(model_folder),
        "-f",
        "all",
        "-chk",
        _CHECKPOINT_NAME,
        "-device",
        device,
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    return cmd


def run_arm(
    arm: str,
    images_ts_dir: Path,
    out_dir: Path,
    model_folder: Path,
    device: str,
    dry_run: bool,
) -> tuple[list[str], float]:
    """Builds and (unless `dry_run`) runs one arm's prediction command.

    Args:
        arm: `"tta_on"` (primary) or `"tta_off"` (secondary).
        images_ts_dir: `imagesTs` directory.
        out_dir: This arm's output directory. Created if missing.
        model_folder: The trained model folder.
        device: `"cuda"` or `"cpu"`.
        dry_run: If True, log the command and return without running it.

    Returns:
        `(cmd, wall_s)` -- the command run (or that would have run), and the
        wall-clock seconds it took (`0.0` for a dry run).

    Raises:
        subprocess.CalledProcessError: If nnU-Net's own process exits
            non-zero (propagated from `subprocess.run(check=True)`).
    """
    disable_tta = arm == _ARM_TTA_OFF
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_predict_command(images_ts_dir, out_dir, model_folder, device, disable_tta)
    logger.info("nnunet_predict_session: arm=%s command: %s", arm, " ".join(cmd))
    if dry_run:
        return cmd, 0.0
    start = time.monotonic()
    # check=True (not captured): output streams straight to this process's
    # own stdout/stderr, so a Kaggle log shows every nnU-Net progress line as
    # it happens -- the same "let it inherit the parent's streams" choice
    # `scripts/nnunet_session.py` itself relies on for its own subprocess-free
    # in-process training loop's ordinary print/log output.
    subprocess.run(cmd, check=True)
    wall_s = time.monotonic() - start
    return cmd, wall_s


# ---------------------------------------------------------------------------
# 4. Normalising nnU-Net's output into score_nnunet.py's required format
# ---------------------------------------------------------------------------


def normalize_arm_output(out_dir: Path, file_ending: str, case_ids: Sequence[str]) -> None:
    """Converts every `<case_id><file_ending>` nnU-Net wrote into `<case_id>.nii.gz`.

    See the module docstring for why: `scripts/score_nnunet.py`'s
    completeness check requires exactly `<case_id>.nii.gz` per case, but
    Kaggle's `.nii.gz` -> `.nii` decompression on ingest means nnU-Net itself
    writes `.nii` here. A no-op if `file_ending` is already `.nii.gz`.

    Args:
        out_dir: One arm's prediction output directory.
        file_ending: The model's file ending (`".nii"` or `".nii.gz"`).
        case_ids: Every case id expected in `out_dir`. Sorted before
            iterating, so the array-verified "first case" is deterministic
            across a re-run.

    Raises:
        FileNotFoundError: If nnU-Net did not write `<case_id><file_ending>`
            for some case (names the case and the expected path).
        AssertionError: If the first (sorted) case's gzip round trip does not
            load, via nibabel, to the same array as the `.nii` it came from.
    """
    if file_ending == ".nii.gz":
        return  # Already in the format score_nnunet.py requires -- nothing to do.

    sorted_ids = sorted(case_ids)
    for i, case_id in enumerate(sorted_ids):
        nii_path = out_dir / f"{case_id}{file_ending}"
        if not nii_path.is_file():
            raise FileNotFoundError(
                f"Expected nnU-Net to have written {nii_path} for case {case_id!r}, but it is "
                f"missing from {out_dir}."
            )
        gz_path = out_dir / f"{case_id}.nii.gz"
        with nii_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            # Byte-for-byte gzip of the raw file -- never re-encoded through
            # nibabel, which could silently change the header or data layout.
            shutil.copyfileobj(f_in, f_out)
        if i == 0:
            # Cheap sanity, first case of the arm only -- not a 189-case
            # re-verification.
            arr_nii = np.asarray(nib.load(str(nii_path)).dataobj)
            arr_gz = np.asarray(nib.load(str(gz_path)).dataobj)
            if not np.array_equal(arr_nii, arr_gz):
                raise AssertionError(
                    f"Gzip round trip for {case_id!r} does not match: {gz_path} does not load "
                    f"to the same array as {nii_path}."
                )
        nii_path.unlink()


def check_arm_completeness(out_dir: Path, case_ids: Sequence[str]) -> None:
    """Requires exactly the expected `<case_id>.nii.gz` files, no more, no fewer.

    Args:
        out_dir: One arm's (normalized) prediction output directory.
        case_ids: Every case id expected.

    Raises:
        ValueError: If any case's `.nii.gz` is missing, or an extra
            `.nii.gz` file (not in `case_ids`) is present.
    """
    expected = {f"{cid}.nii.gz" for cid in case_ids}
    found = {p.name for p in out_dir.glob("*.nii.gz") if p.is_file()}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise ValueError(
            f"{out_dir} does not hold exactly the expected {len(expected)} case(s): "
            f"missing {missing}, extra {extra}."
        )


def check_first_case_labels(out_dir: Path, case_ids: Sequence[str]) -> None:
    """Checks the first (sorted) case's predicted labels are a subset of `{0, 1, 2, 3}`.

    A raw sanity check on nnU-Net's OWN label convention (1 = ED, 2 = NCR,
    3 = ET) -- the remap to this project's convention happens later, in
    `analysis/nnunet_import.py`, never here.

    Args:
        out_dir: One arm's (normalized) prediction output directory.
        case_ids: Every case id expected; the first, sorted, is checked.

    Raises:
        ValueError: If any label outside `{0, 1, 2, 3}` is found.
    """
    first_case = sorted(case_ids)[0]
    path = out_dir / f"{first_case}.nii.gz"
    labels = {int(v) for v in np.unique(np.asarray(nib.load(str(path)).dataobj))}
    bad = labels - _VALID_NNUNET_LABELS
    if bad:
        raise ValueError(f"{path} contains label value(s) outside {{0, 1, 2, 3}}: {sorted(bad)}.")


# ---------------------------------------------------------------------------
# 5. Orchestration and CLI
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Runs every preflight check, then (unless `--dry-run`) every arm's prediction.

    Args:
        args: Parsed CLI arguments (see `parse_args`).

    Returns:
        The summary dict written to `--summary-json` on success.

    Raises:
        ValueError: If `--arms` is empty or names an unknown arm, or any
            check in `find_images_ts_dir` / `discover_test_cases` /
            `check_case_ids_match_split` fails.
        FileNotFoundError: See `find_model_folder` / `find_images_ts_dir`.
        RuntimeError: See `check_checkpoint_identity`, or if `--require-cuda`
            is given but CUDA is unavailable (propagated from
            `nnunet_session._require_cuda_or_raise`).
        subprocess.CalledProcessError: If any arm's nnU-Net process exits
            non-zero.
        AssertionError: See `normalize_arm_output`.
    """
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise ValueError("--arms resolved to an empty list.")
    unknown = [a for a in arms if a not in _VALID_ARMS]
    if unknown:
        raise ValueError(f"--arms names unknown arm(s) {unknown}; valid arms are {_VALID_ARMS}.")

    # --- 1. Locate the model folder + checkpoint identity ---
    model_folder = find_model_folder(Path(args.model_search_root))
    checkpoint_path = model_folder / _FOLD_NAME / _CHECKPOINT_NAME
    meta = read_checkpoint_meta(checkpoint_path)
    check_checkpoint_identity(meta, args.expect_epoch)

    # --- 2. Locate + validate the test images ---
    images_ts_dir = find_images_ts_dir(Path(args.images_search_root))
    file_ending = read_file_ending(model_folder)
    cases = discover_test_cases(images_ts_dir, file_ending)
    splits = load_splits(args.splits)
    expected_case_ids = list(splits["test"])
    check_case_ids_match_split(set(cases), expected_case_ids)
    case_ids = sorted(cases)

    # --- 3. Device ---
    if args.require_cuda:
        _require_cuda_or_raise()
    device = args.device

    logger.info(
        "nnunet_predict_session: model_folder=%s checkpoint_epoch=%s device=%s arms=%s "
        "n_cases=%d",
        model_folder,
        meta["current_epoch"],
        device,
        arms,
        len(case_ids),
    )

    # --- 4. Run every arm, primary first ---
    out_root = Path(args.out_root)
    arm_summaries: dict[str, Any] = {}
    for arm in arms:
        arm_out_dir = out_root / arm
        cmd, wall_s = run_arm(arm, images_ts_dir, arm_out_dir, model_folder, device, args.dry_run)
        if args.dry_run:
            arm_summaries[arm] = {"command": cmd}
            continue
        normalize_arm_output(arm_out_dir, file_ending, case_ids)
        check_arm_completeness(arm_out_dir, case_ids)
        check_first_case_labels(arm_out_dir, case_ids)
        arm_summaries[arm] = {
            "command": cmd,
            "wall_s": wall_s,
            "s_per_case": wall_s / len(case_ids) if case_ids else 0.0,
            "n_cases": len(case_ids),
            "out_dir": str(arm_out_dir),
        }

    return {
        "model_folder": str(model_folder),
        "checkpoint_epoch": meta["current_epoch"],
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nnunetv2_version": _nnunetv2_version(),
        "arms": arm_summaries,
        "dry_run": bool(args.dry_run),
        "health": "OK",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses this script's CLI arguments.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        The parsed `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-search-root",
        required=True,
        help="Root to search for the trained model folder (e.g. /kaggle/input).",
    )
    parser.add_argument(
        "--images-search-root",
        required=True,
        help="Root to search for the one imagesTs directory (e.g. /kaggle/input).",
    )
    parser.add_argument(
        "--out-root", required=True, help="Writes <out-root>/<arm>/ for each arm in --arms."
    )
    parser.add_argument(
        "--splits", required=True, help="Path to configs/data/splits.yaml (its 'test' list)."
    )
    parser.add_argument(
        "--expect-epoch",
        type=int,
        default=1000,
        help="Epoch checkpoint_final.pth must report (Amendment 1: 1000).",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast unless CUDA is available, runs a kernel, and is sm_70+ (no P100).",
    )
    parser.add_argument(
        "--arms",
        default=f"{_ARM_TTA_ON},{_ARM_TTA_OFF}",
        help="Comma-separated arms to run, order preserved (primary first). "
        f"Valid values: {_VALID_ARMS}.",
    )
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and print the commands that would run; run no prediction.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parses args, runs every check and (unless `--dry-run`) every arm, and reports health.

    Args:
        argv: Argument list, or None to read `sys.argv[1:]`.

    Returns:
        0 on success, 1 on any failure (a preflight check, an nnU-Net
        subprocess failure, or a post-prediction sanity check).
    """
    args = parse_args(argv)
    setup_logging(level="INFO")
    try:
        summary = run(args)
        write_json(summary, Path(args.summary_json))
        logger.info("NVX_PREDICT: OK")
        return 0
    except Exception as exc:
        logger.exception("nnunet_predict_session failed:")
        logger.error("NVX_PREDICT: FAIL %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
