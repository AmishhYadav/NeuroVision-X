"""Hydra entry point: mirror SMALL result files from outputs/ into the tracked results/ tree.

`outputs/` is gitignored, so a laptop loss would destroy every measured result. This script
copies per-case CSVs, summaries and verdict JSONs to `results/` (same relative paths) so they
can be committed. It never copies volume-sized artifacts: a deny list in this file (not in
config) blocks logits, predictions, checkpoints, NIfTI, DICOM and archives even if a config
edit allows their extension.

Run a preview first:  python scripts/sync_results.py dry_run=true
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Hard deny list: matched against the END of the lowercase file name, so the double suffix
# ".nii.gz" is caught. Deliberately NOT configurable.
DENIED_SUFFIXES: tuple[str, ...] = (
    ".npy", ".nii", ".nii.gz", ".pt", ".pth", ".ckpt", ".dcm", ".mha",
    ".nrrd", ".h5", ".f32", ".u32", ".zip", ".tar", ".gz",
)  # fmt: skip


@dataclass
class SelectionResult:
    """Outcome of scanning the source tree. All paths are relative to the source dir."""

    selected: list[Path] = field(default_factory=list)
    skipped_oversize: list[tuple[Path, int]] = field(default_factory=list)
    skipped_denied: list[Path] = field(default_factory=list)


@dataclass
class SyncReport:
    """Counts and lists from one sync pass. Paths are relative."""

    copied: int = 0
    unchanged: int = 0
    copied_files: list[Path] = field(default_factory=list)
    orphaned: list[Path] = field(default_factory=list)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(
    source_dir: Path,
    allowed_extensions: list[str],
    max_bytes: int,
    exclude_dir_names: list[str],
    exclude_globs: list[str],
) -> SelectionResult:
    """Pick the files under `source_dir` that are safe and small enough to mirror.

    Args:
        source_dir: Directory to scan.
        allowed_extensions: Allowed `Path.suffix` values (case-insensitive).
        max_bytes: Files strictly larger than this are skipped as oversize.
        exclude_dir_names: Directory names; a file is dropped if ANY path component equals one.
        exclude_globs: fnmatch patterns matched against the POSIX relative path.

    Returns:
        A SelectionResult with sorted relative paths.
    """
    allowed = {ext.lower() for ext in allowed_extensions}
    excluded_names = set(exclude_dir_names)
    result = SelectionResult()

    for root, dirs, files in os.walk(source_dir, followlinks=False):
        # Prune excluded directories in place so os.walk never descends into them.
        dirs[:] = sorted(d for d in dirs if d not in excluded_names)
        for name in files:
            full = Path(root) / name
            rel = full.relative_to(source_dir)
            if full.is_symlink():
                logger.debug("Skipping symlinked file: %s", rel)
                continue
            if any(fnmatch.fnmatchcase(rel.as_posix(), pat) for pat in exclude_globs):
                continue
            lower_name = name.lower()
            suffix_allowed = full.suffix.lower() in allowed
            if lower_name.endswith(DENIED_SUFFIXES):
                # Only list denied files whose suffix was otherwise allowed (avoids listing
                # thousands of DICOMs).
                if suffix_allowed:
                    result.skipped_denied.append(rel)
                continue
            if not suffix_allowed:
                continue
            size = full.stat().st_size
            if size > max_bytes:
                logger.warning("Skipping oversize file (%.2f MB): %s", size / 1024**2, rel)
                result.skipped_oversize.append((rel, size))
                continue
            result.selected.append(rel)

    result.selected.sort()
    result.skipped_oversize.sort()
    result.skipped_denied.sort()
    return result


def sync(source_dir: Path, dest_dir: Path, selection: SelectionResult, dry_run: bool) -> SyncReport:
    """Copy selected files into `dest_dir`, only when missing or different. Never deletes.

    Args:
        source_dir: Where the files come from.
        dest_dir: Where they go (created on demand; untouched in a dry run).
        selection: Result of `select_files`.
        dry_run: If True, only count what would be copied; write nothing.

    Returns:
        A SyncReport. In a dry run, `copied` is the number that WOULD be copied.
    """
    report = SyncReport()
    for rel in selection.selected:
        src = source_dir / rel
        dst = dest_dir / rel
        if dst.is_file() and _sha256(dst) == _sha256(src):
            report.unchanged += 1
            continue
        report.copied += 1
        report.copied_files.append(rel)
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if dest_dir.is_dir():
        selected = set(selection.selected)
        existing = (p.relative_to(dest_dir) for p in dest_dir.rglob("*") if p.is_file())
        report.orphaned = sorted(p for p in existing if p not in selected)
    return report


def write_manifest(dest_dir: Path, manifest_name: str) -> Path:
    """Write `<sha256>  <posix relpath>` lines, sorted, for every file under `dest_dir`.

    The manifest itself is excluded, so two runs over the same tree are byte-identical.

    Returns:
        Path of the manifest file.
    """
    lines = sorted(
        (p.relative_to(dest_dir).as_posix(), _sha256(p))
        for p in dest_dir.rglob("*")
        if p.is_file() and p.relative_to(dest_dir).as_posix() != manifest_name
    )
    manifest = dest_dir / manifest_name
    manifest.write_text("".join(f"{digest}  {rel}\n" for rel, digest in lines))
    return manifest


def _log_dry_run(source_dir: Path, report: SyncReport) -> None:
    """Log per-top-level-directory counts and total bytes of what would be copied."""
    per_top: Counter[str] = Counter()
    total = 0
    for rel in report.copied_files:
        per_top[rel.parts[0] if len(rel.parts) > 1 else "."] += 1
        total += (source_dir / rel).stat().st_size
    for top, count in sorted(per_top.items()):
        logger.info("DRY RUN would copy %5d files under %s", count, top)
    logger.info("DRY RUN would copy %d files, %.2f MB", report.copied, total / 1024**2)


@hydra.main(config_path="../configs/results", config_name="sync", version_base=None)
def main(cfg: DictConfig) -> None:
    """Resolve paths against the original cwd, then select, sync and write the manifest."""
    base = Path(hydra.utils.get_original_cwd())
    source_dir = (base / cfg.source_dir).resolve()
    dest_dir = (base / cfg.dest_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    selection = select_files(
        source_dir,
        list(cfg.allowed_extensions),
        int(cfg.max_file_mb * 1024 * 1024),
        list(cfg.exclude_dir_names),
        list(cfg.exclude_globs),
    )
    report = sync(source_dir, dest_dir, selection, bool(cfg.dry_run))

    if cfg.dry_run:
        _log_dry_run(source_dir, report)
    else:
        write_manifest(dest_dir, cfg.manifest_name)

    if report.orphaned:
        # Orphans are informational: files in dest that no longer qualify in source.
        names = [p.as_posix() for p in report.orphaned if p.name != cfg.manifest_name]
        if names:
            logger.warning("%d orphaned files in dest (first 10): %s", len(names), names[:10])

    total = sum((source_dir / p).stat().st_size for p in selection.selected)
    logger.info(
        "selected %d files, %.2f MB total; copied %d, unchanged %d; "
        "skipped oversize %d; denied %d",
        len(selection.selected),
        total / 1024**2,
        report.copied,
        report.unchanged,
        len(selection.skipped_oversize),
        len(selection.skipped_denied),
    )
    for rel, size in selection.skipped_oversize:
        logger.info("oversize: %s (%.2f MB)", rel, size / 1024**2)


if __name__ == "__main__":
    main()
