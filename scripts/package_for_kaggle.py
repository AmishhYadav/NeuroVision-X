"""Assemble an upload-ready folder for a Kaggle dataset, and size-check it.

A Kaggle training session needs the preprocessed BraTS cache plus the frozen
train/val/test split as INPUT data. This script gathers both into one folder
under `--out`, then reports its size against Kaggle's per-dataset upload
guideline *before* you spend 40 minutes uploading something too large or
missing a case.

Plain `argparse`, not Hydra: this is an operational tool you run once per
dataset refresh (source dir, output dir, dataset slug), not an experiment
whose parameters get logged and swept. Hydra's config composition and
mandatory-value ("???") machinery would add ceremony without buying anything
here -- there is no config group to select, no sweep, nothing to log to W&B.

Example usage:

    python scripts/package_for_kaggle.py \\
        --prep-dir data/preprocessed/brats \\
        --splits configs/data/splits.yaml \\
        --out outputs/kaggle_upload \\
        --slug myuser/neurovision-brats-prep

This is a CLI entry point, so it prints a human-facing summary via `logging`
at INFO level rather than a silent library call -- the whole point of this
script is the terminal report.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from neurovision.data.dataset import load_splits
from neurovision.utils.io import (
    KAGGLE_DATASET_WARN_GB,
    directory_size_bytes,
    ensure_dir,
    format_size,
    write_json,
)
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Files/directories that only exist because macOS Finder, Spotlight, or a
# Python import writes them into any directory it touches. They carry no
# training-relevant data and would otherwise ride along into the uploaded
# Kaggle dataset -- this matters specifically because packaging runs on a
# Mac, never on Kaggle's Linux image.
_JUNK_FILE_NAMES = {".DS_Store"}
_JUNK_FILE_PREFIXES = ("._",)  # macOS AppleDouble shadow files
_JUNK_FILE_SUFFIXES = (".pyc",)
_JUNK_DIR_NAMES = {"__pycache__"}


def is_junk_file(path: Path) -> bool:
    """Return True if `path` is a macOS/Python artifact to exclude from packaging.

    Args:
        path: File path to check (need not exist).

    Returns:
        True if the file should be skipped when assembling the upload folder.
    """
    name = path.name
    if name in _JUNK_FILE_NAMES:
        return True
    if name.startswith(_JUNK_FILE_PREFIXES):
        return True
    if name.endswith(_JUNK_FILE_SUFFIXES):
        return True
    return False


def _place_file(src: Path, dst: Path, force_copy: bool, warned: list[bool]) -> None:
    """Put one file at `dst`, hardlinking from `src` unless that is impossible.

    Args:
        src: Source file.
        dst: Destination path. Parent directories are created as needed.
        force_copy: If True, always use a real copy (`--copy`), skipping the
            hardlink attempt entirely.
        warned: One-element mutable list used as an in/out flag so the
            "falling back to copies" message is logged once per run, not
            once per file -- a cache with tens of thousands of `.npy` files
            would otherwise flood the log.
    """
    ensure_dir(dst.parent)
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        # Hardlinking is why this is fast and cheap: the preprocessed cache
        # is tens of GB, and a real copy would double disk use and wall time
        # for no benefit when --prep-dir and --out sit on the same volume.
        os.link(src, dst)
    except OSError:
        # Cross-filesystem moves (e.g. an external drive) or filesystems
        # without hardlink support (some network shares) raise here. Fall
        # back to a real copy and say so exactly once.
        if not warned[0]:
            logger.warning(
                "Hardlinking failed (likely a cross-filesystem destination or a "
                "filesystem without hardlink support); falling back to real copies "
                "for the rest of this run. This uses more disk and takes longer."
            )
            warned[0] = True
        shutil.copy2(src, dst)


def copy_tree_excluding_junk(
    src_dir: Path, dst_dir: Path, force_copy: bool, warned: list[bool]
) -> None:
    """Recursively place every non-junk file from `src_dir` under `dst_dir`.

    Args:
        src_dir: Source directory to walk.
        dst_dir: Destination root. Created if missing.
        force_copy: Passed through to `_place_file`.
        warned: Passed through to `_place_file`.
    """
    ensure_dir(dst_dir)
    for root, dirnames, filenames in os.walk(src_dir):
        # Prune junk directories in place so os.walk never descends into
        # them -- cheaper than filtering their contents file by file.
        dirnames[:] = [d for d in dirnames if d not in _JUNK_DIR_NAMES]
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        for filename in filenames:
            src_file = root_path / filename
            if is_junk_file(src_file):
                continue
            dst_file = dst_dir / rel_root / filename
            _place_file(src_file, dst_file, force_copy, warned)


def validate_inputs(prep_dir: Path, splits_path: Path) -> dict[str, list[str]]:
    """Check that the preprocessed cache and split file are consistent before packaging.

    Args:
        prep_dir: Root of the preprocessed cache (`data.preprocessing.out_dir`).
        splits_path: Path to the split YAML (`data.splits.path`).

    Returns:
        The loaded split dict (`"train"`, `"val"`, `"test"` -> case id lists).

    Raises:
        FileNotFoundError: If `prep_dir` does not exist or is empty, or if
            `splits_path` does not exist.
        ValueError: If the split file is malformed, or if any case id in the
            split has no corresponding directory under `prep_dir`.
    """
    if not prep_dir.is_dir():
        raise FileNotFoundError(f"--prep-dir does not exist: {prep_dir.resolve()}")
    if not any(prep_dir.iterdir()):
        raise FileNotFoundError(f"--prep-dir is empty: {prep_dir.resolve()}")

    # load_splits raises FileNotFoundError/ValueError on a missing or
    # malformed file, so a broken split fails right here -- not after a
    # 40-minute upload and a queued Kaggle GPU session, which is the
    # expensive failure mode this whole function exists to prevent.
    splits = load_splits(splits_path)

    all_case_ids = [*splits["train"], *splits["val"], *splits["test"]]
    missing = [cid for cid in all_case_ids if not (prep_dir / cid).is_dir()]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"{len(missing)} case id(s) from {splits_path} have no directory under "
            f"{prep_dir.resolve()}. First few missing: {preview}. "
            "Re-run preprocessing for these cases (or regenerate the split) before "
            "packaging -- discovering this after uploading is far more expensive "
            "than checking it now."
        )

    # The whole prep_dir is packaged, not just the cases the split names, so
    # any extra preprocessed case is dead weight in the upload. This is easy to
    # end up with: splits are frozen once written (adding cases later raises
    # rather than reshuffling), so preprocessing more cases afterwards leaves
    # them on disk but outside the split. Warn rather than silently dropping
    # them -- uploading a few spare cases is recoverable, quietly omitting a
    # case the split does reference would not be.
    on_disk = {p.name for p in prep_dir.iterdir() if p.is_dir()}
    extra = sorted(on_disk - set(all_case_ids))
    if extra:
        logger.warning(
            "%d case director(ies) under %s are not referenced by %s and will still be "
            "uploaded as dead weight. First few: %s. Regenerate the split, or package a "
            "directory containing only the split's cases, to avoid paying for them.",
            len(extra),
            prep_dir.resolve(),
            splits_path,
            extra[:5],
        )

    return splits


def measure_components(prep_dir: Path, splits_path: Path) -> dict[str, int]:
    """Measure the size, in bytes, of each piece that will go into the package.

    Measured from the SOURCE paths (not the assembled `--out` folder), via
    `directory_size_bytes` -- the same size arithmetic `preprocess.py` uses
    -- so this gives a real number whether or not anything has actually been
    written yet. That is what makes `--dry-run` able to report a real size:
    it measures `prep_dir` directly instead of the (unwritten) output.

    Args:
        prep_dir: Root of the preprocessed cache.
        splits_path: Path to the split YAML.

    Returns:
        Dict with `"preprocessed_bytes"`, `"metadata_bytes"` (0 if
        `metadata.csv` is absent), and `"splits_bytes"`.
    """
    preprocessed_bytes = directory_size_bytes(prep_dir)
    metadata_path = prep_dir / "metadata.csv"
    metadata_bytes = metadata_path.stat().st_size if metadata_path.is_file() else 0
    splits_bytes = splits_path.stat().st_size if splits_path.is_file() else 0
    return {
        "preprocessed_bytes": preprocessed_bytes,
        "metadata_bytes": metadata_bytes,
        "splits_bytes": splits_bytes,
    }


def report_size(components: dict[str, int], case_count: int, limit_gb: float) -> dict[str, Any]:
    """Log a size breakdown and escalate clearly if it is close to or over the limit.

    Note `metadata_bytes` is NOT added into the total: `metadata.csv` lives
    inside `prep_dir` and is therefore already counted within
    `preprocessed_bytes`. It is reported separately purely as an informative
    breakdown line, not double-counted into the total.

    Args:
        components: Output of `measure_components`.
        case_count: Number of cases in the split (train + val + test).
        limit_gb: Kaggle dataset size guideline, in GB.

    Returns:
        Dict with `"total_bytes"`, `"total_str"`, `"pct_of_limit"`, and
        `"mean_per_case_bytes"`.
    """
    total_bytes = components["preprocessed_bytes"] + components["splits_bytes"]
    total_str = format_size(total_bytes)
    mean_per_case = total_bytes / case_count if case_count else 0.0
    limit_bytes = limit_gb * (1024**3)
    pct = (total_bytes / limit_bytes * 100.0) if limit_bytes > 0 else float("inf")

    logger.info("Package size breakdown:")
    logger.info("  preprocessed/  %s", format_size(components["preprocessed_bytes"]))
    # Labelled as already-counted so the breakdown lines visibly do not sum to
    # TOTAL on purpose -- metadata.csv lives inside prep_dir, so it is already
    # inside preprocessed_bytes. Without the label this reads as an arithmetic
    # bug in the report.
    logger.info(
        "  metadata.csv   %s  (already inside preprocessed/)",
        format_size(components["metadata_bytes"]),
    )
    logger.info("  splits.yaml    %s", format_size(components["splits_bytes"]))
    logger.info(
        "  TOTAL          %s  (%d case(s), %s/case)",
        total_str,
        case_count,
        format_size(int(mean_per_case)),
    )
    logger.info("  %.1f%% of the ~%.0f GB Kaggle dataset guideline", pct, limit_gb)

    if pct > 100.0:
        excess = format_size(int(total_bytes - limit_bytes))
        logger.warning("!" * 70)
        logger.warning(
            "PACKAGE EXCEEDS the ~%.0f GB Kaggle dataset guideline by %s (%.1f%% of the limit). "
            "Upload may be slow or rejected. Consider a smaller subset via "
            "data.preprocessing.limit=N, or splitting the cache across multiple "
            "Kaggle datasets.",
            limit_gb,
            excess,
            pct,
        )
        logger.warning("!" * 70)
    elif pct >= 75.0:
        logger.warning(
            "Package is at %.1f%% of the ~%.0f GB Kaggle dataset guideline -- getting close.",
            pct,
            limit_gb,
        )
    else:
        logger.info("Well under the ~%.0f GB Kaggle dataset guideline.", limit_gb)

    return {
        "total_bytes": total_bytes,
        "total_str": total_str,
        "pct_of_limit": pct,
        "mean_per_case_bytes": mean_per_case,
    }


def print_next_commands(out_dir: Path, slug: str | None) -> None:
    """Log the exact `kaggle` CLI commands to run next.

    Args:
        out_dir: The assembled upload folder.
        slug: Kaggle dataset slug, or None if `--slug` was not given.
    """
    # --dir-mode zip matters because the preprocessed cache is thousands of
    # small .npy files: Kaggle's API uploads a directory file-by-file unless
    # told to zip it first, and doing that individually is dramatically
    # slower (and more failure-prone) than one zipped upload.
    logger.info("Next steps (uploads a single zip, not thousands of individual files):")
    logger.info("  kaggle datasets create -p %s --dir-mode zip        # first upload", out_dir)
    logger.info(
        '  kaggle datasets version -p %s -m "<message>" --dir-mode zip   # subsequent',
        out_dir,
    )
    if slug is None:
        logger.info(
            "  (no --slug was given, so dataset-metadata.json was not written -- "
            "`kaggle datasets create` needs one; re-run with --slug first.)"
        )


def assemble_package(
    prep_dir: Path,
    splits_path: Path,
    out_dir: Path,
    slug: str | None,
    title: str | None,
    force_copy: bool,
    dry_run: bool,
) -> None:
    """Build the upload folder at `out_dir` (no-op if `dry_run`).

    Args:
        prep_dir: Root of the preprocessed cache.
        splits_path: Path to the split YAML (copied into the package).
        out_dir: Destination upload folder.
        slug: Kaggle dataset slug, or None to skip `dataset-metadata.json`.
        title: Dataset title; defaults to the slug's last path segment.
        force_copy: Force real copies instead of hardlinks.
        dry_run: If True, measure only -- write nothing.
    """
    if dry_run:
        logger.info("--dry-run: not writing anything to %s", out_dir)
        return

    ensure_dir(out_dir)
    warned = [False]  # shared "already logged the hardlink fallback" flag

    copy_tree_excluding_junk(prep_dir, out_dir / "preprocessed", force_copy, warned)

    metadata_src = prep_dir / "metadata.csv"
    if metadata_src.is_file():
        _place_file(metadata_src, out_dir / "metadata.csv", force_copy, warned)
    else:
        logger.warning(
            "No metadata.csv found under %s -- skipping (informational only, not "
            "required for training).",
            prep_dir,
        )

    # Hardlinked/copied verbatim (not re-serialized from the loaded dict) so
    # the packaged copy is byte-identical to the frozen split file, including
    # its "meta" block (seed, fractions) that `load_splits` does not surface.
    _place_file(splits_path, out_dir / "splits.yaml", force_copy, warned)

    if slug is not None:
        resolved_title = title or slug.rsplit("/", 1)[-1]
        # License is "other", NOT CC0-1.0. CC0 means "public domain, no rights
        # reserved", which is neither BraTS's license nor ours to grant: BraTS
        # is distributed under its own data use agreement, and that agreement
        # does not permit public redistribution of the imaging data. Declaring
        # CC0 on a derived copy is a false license statement even when the
        # Kaggle dataset is private. Keep the dataset private (which is what
        # `kaggle datasets create` does by default) and leave the real terms
        # to the BraTS DUA.
        write_json(
            {"title": resolved_title, "id": slug, "licenses": [{"name": "other"}]},
            out_dir / "dataset-metadata.json",
        )
        logger.info(
            "Wrote dataset-metadata.json (id=%s, title=%s, license=other -- BraTS DUA applies, "
            "keep this dataset PRIVATE)",
            slug,
            resolved_title,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prep-dir",
        type=str,
        required=True,
        help="Preprocessed cache root (what data.preprocessing.out_dir points at).",
    )
    parser.add_argument("--splits", type=str, required=True, help="Path to the split YAML.")
    parser.add_argument("--out", type=str, required=True, help="Upload folder to build.")
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Kaggle dataset slug, e.g. username/neurovision-brats-prep.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Dataset title; defaults to the slug's last segment.",
    )
    parser.add_argument(
        "--limit-gb",
        type=float,
        default=KAGGLE_DATASET_WARN_GB,
        help="Warn above this size, in GB.",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Force real copies instead of hardlinks."
    )
    parser.add_argument("--dry-run", action="store_true", help="Measure and report; write nothing.")
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty --out.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate, assemble, and report -- the whole script, minus CLI/exit-code plumbing.

    Args:
        args: Parsed CLI arguments (see `build_arg_parser`).

    Returns:
        The size report dict from `report_size`, for callers (tests) that
        want to assert on it directly instead of scraping logs.

    Raises:
        FileNotFoundError: Invalid `--prep-dir` or `--splits`.
        ValueError: Malformed split file, missing cases, or a non-empty
            `--out` without `--force`.
    """
    prep_dir = Path(args.prep_dir)
    splits_path = Path(args.splits)
    out_dir = Path(args.out)

    splits = validate_inputs(prep_dir, splits_path)

    out_is_nonempty = out_dir.is_dir() and any(out_dir.iterdir())
    if not args.dry_run and out_is_nonempty:
        if not args.force:
            raise ValueError(
                f"--out {out_dir.resolve()} already exists and is not empty. "
                "Pass --force to overwrite it, or choose a different --out."
            )
        logger.warning("Overwriting non-empty --out %s (--force given).", out_dir.resolve())
        shutil.rmtree(out_dir)

    assemble_package(
        prep_dir=prep_dir,
        splits_path=splits_path,
        out_dir=out_dir,
        slug=args.slug,
        title=args.title,
        force_copy=args.copy,
        dry_run=args.dry_run,
    )

    components = measure_components(prep_dir, splits_path)
    case_count = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    report = report_size(components, case_count, args.limit_gb)

    print_next_commands(out_dir, args.slug)
    return report


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
        logger.exception("Packaging failed:")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
