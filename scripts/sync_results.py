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
from pathlib import Path, PurePosixPath

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


def _normalise_subtrees(include_subtrees: list[str] | None) -> list[tuple[str, ...]]:
    """Turn include_subtrees into tuples of path components (never strings; see trap note).

    Raises:
        ValueError: If an entry is absolute, contains "..", or is empty.
    """
    parsed: list[tuple[str, ...]] = []
    for raw in include_subtrees or []:
        # PurePosixPath drops trailing slashes and "." parts, so "a/b/" == "a/b".
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(
                f"include_subtrees entry must be a relative path without '..': {raw!r}"
            )
        parsed.append(path.parts)
    return parsed


def _is_prefix(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    """True if `short` is a prefix of `long`, compared component by component."""
    return long[: len(short)] == short


def select_files(
    source_dir: Path,
    allowed_extensions: list[str],
    max_bytes: int,
    exclude_dir_names: list[str],
    exclude_globs: list[str],
    include_subtrees: list[str] | None = None,
) -> SelectionResult:
    """Pick the files under `source_dir` that are safe and small enough to mirror.

    Args:
        source_dir: Directory to scan.
        allowed_extensions: Allowed `Path.suffix` values (case-insensitive).
        max_bytes: Files strictly larger than this are skipped as oversize.
        exclude_dir_names: Directory names; a file is dropped if ANY path component equals one.
        exclude_globs: fnmatch patterns matched against the POSIX relative path.
        include_subtrees: POSIX paths relative to `source_dir` that are re-admitted even
            though they sit inside an excluded directory. Only the excluded names on the way
            TO the subtree (its ancestors, or itself) are forgiven; an excluded name BELOW
            the subtree root stays excluded. Deny list, globs, extensions and size limits
            still apply. Matching is by path component, not string prefix.

    Returns:
        A SelectionResult with sorted relative paths.

    Raises:
        ValueError: If an include_subtrees entry is absolute or contains "..".
    """
    allowed = {ext.lower() for ext in allowed_extensions}
    excluded_names = set(exclude_dir_names)
    includes = _normalise_subtrees(include_subtrees)
    for inc in includes:
        if not (source_dir / Path(*inc)).exists():
            logger.warning("include_subtrees entry not found under source_dir: %s", "/".join(inc))
    result = SelectionResult()

    def dir_is_allowed(dir_parts: tuple[str, ...], for_walking: bool) -> bool:
        """Decide whether a directory (as component tuple) may be walked / hold files."""
        for inc in includes:
            if _is_prefix(inc, dir_parts):
                # Inside the include subtree: only excluded names BELOW its root count.
                return not any(part in excluded_names for part in dir_parts[len(inc) :])
            if for_walking and _is_prefix(dir_parts, inc):
                # An ancestor of the subtree (e.g. kaggle_kernels/): walk through it so we
                # can reach the subtree. Its own files are rejected by the file check.
                return True
        return not any(part in excluded_names for part in dir_parts)

    for root, dirs, files in os.walk(source_dir, followlinks=False):
        root_parts = Path(root).relative_to(source_dir).parts
        # Prune excluded directories in place so os.walk never descends into them
        # (unless they lead to an include subtree).
        dirs[:] = sorted(d for d in dirs if dir_is_allowed((*root_parts, d), for_walking=True))
        # Files here count only if this directory is allowed as a file location, which is
        # stricter than "may be walked" for ancestors of an include subtree.
        if not dir_is_allowed(root_parts, for_walking=False):
            continue
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
        list(cfg.get("include_subtrees", [])),
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
