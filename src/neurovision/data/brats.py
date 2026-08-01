"""BraTS directory reader — pure path logic, no volume I/O.

Resolves a BraTS root directory (2020/2021 underscore-style or 2023+
hyphen-style case naming, or a mix of both) into a list of `BratsCase`
records that point at the four MRI modalities and, optionally, the ground
truth segmentation mask.

This module never opens a `.nii`/`.nii.gz` file — it only builds and checks
paths — so it does not import torch, monai, nibabel, or SimpleITK. That keeps
it fast and dependency-free, and keeps this file testable with empty
placeholder files.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from neurovision.utils.io import ensure_dir

logger = logging.getLogger(__name__)

# Canonical internal role names, in the fixed channel order the rest of the
# pipeline expects (t1, t1ce, t2, flair). "seg" is handled separately because
# it is optional (see `require_seg`).
_MODALITY_ROLES = ("t1", "t1ce", "t2", "flair")
_ALL_ROLES = (*_MODALITY_ROLES, "seg")

# Per-convention suffix for each canonical role. Suffixes are matched exactly
# against a filename built from the case directory name — never with a
# substring/glob check — because "_t1" is a substring of "_t1ce" and a naive
# `"_t1" in name` (or `*_t1*` glob) would silently resolve t1 and t1ce to the
# same file. That would train the model on a duplicated channel with no
# error anywhere, which is why this dict-based exact-suffix approach exists.
_SUFFIXES_2020 = {
    "t1": "_t1",
    "t1ce": "_t1ce",
    "t2": "_t2",
    "flair": "_flair",
    "seg": "_seg",
}
_SUFFIXES_2023 = {
    "t1": "-t1n",
    "t1ce": "-t1c",
    "t2": "-t2w",
    "flair": "-t2f",
    "seg": "-seg",
}
_CONVENTIONS = {
    "BraTS 2020 (_t1/_t1ce/_t2/_flair/_seg)": _SUFFIXES_2020,
    "BraTS 2023+ (-t1n/-t1c/-t2w/-t2f/-seg)": _SUFFIXES_2023,
}


@dataclass(frozen=True)
class BratsCase:
    """One resolved BraTS case: the four modalities plus an optional mask.

    Attributes:
        case_id: Name of the case directory (e.g. "BraTS20_Training_001").
        t1: Path to the T1 volume.
        t1ce: Path to the contrast-enhanced T1 volume.
        t2: Path to the T2 volume.
        flair: Path to the FLAIR volume.
        seg: Path to the ground-truth segmentation mask, or `None` if the
            case has no mask (BraTS validation/test sets ship without one).
    """

    case_id: str
    t1: Path
    t1ce: Path
    t2: Path
    flair: Path
    seg: Path | None

    @property
    def modality_paths(self) -> list[Path]:
        """Return the four MRI modality paths in a fixed, model-facing order.

        Returns:
            `[t1, t1ce, t2, flair]`, always in this order, matching the
            channel order the encoders expect.
        """
        return [self.t1, self.t1ce, self.t2, self.flair]


def _resolve_role(case_dir: Path, case_id: str, suffix: str) -> Path | None:
    """Find the file for one role under one naming convention.

    Builds the expected filename explicitly from `case_id` + `suffix` and
    checks it with `.is_file()` rather than globbing, so "t1" can never
    accidentally match a "t1ce" file (see module docstring).

    Args:
        case_dir: Directory the case's files live in.
        case_id: Name of the case directory (used as the filename prefix).
        suffix: Exact suffix for this role under this convention, e.g. "_t1".

    Returns:
        Path to the `.nii.gz` file if present; else the plain `.nii` file if
        present (fallback for decompressed copies); else `None`.
    """
    gz_path = case_dir / f"{case_id}{suffix}.nii.gz"
    if gz_path.is_file():
        return gz_path
    plain_path = case_dir / f"{case_id}{suffix}.nii"
    if plain_path.is_file():
        return plain_path
    return None


def _resolve_case(
    case_dir: Path, require_seg: bool
) -> tuple[BratsCase | None, str | None, dict[str, str] | None]:
    """Try to resolve one case directory against both naming conventions.

    Args:
        case_dir: The case's directory.
        require_seg: Whether a missing `seg` file makes the case incomplete.

    Returns:
        A tuple `(case, matched_convention_name, missing_roles_by_convention)`.
        Exactly one of `case` or `missing_roles_by_convention` is not None:
        - On success, `case` is the resolved `BratsCase` and
          `matched_convention_name` names which convention matched.
        - On failure, `case` is `None` and `missing_roles_by_convention` maps
          each convention's display name to the list of roles it could not
          find, so the caller can report the convention that matched most.
    """
    case_id = case_dir.name
    roles_required = _ALL_ROLES if require_seg else _MODALITY_ROLES

    best_convention: str | None = None
    best_found: dict[str, Path] = {}
    best_missing: list[str] = list(roles_required)

    for convention_name, suffixes in _CONVENTIONS.items():
        found: dict[str, Path] = {}
        missing: list[str] = []
        for role in roles_required:
            resolved = _resolve_role(case_dir, case_id, suffixes[role])
            if resolved is not None:
                found[role] = resolved
            else:
                missing.append(role)

        if not missing:
            # Complete match under this convention -> done.
            seg_path = (
                found.get("seg")
                if require_seg
                else _resolve_role(case_dir, case_id, suffixes["seg"])
            )
            logger.debug("Case %s matched convention: %s", case_id, convention_name)
            case = BratsCase(
                case_id=case_id,
                t1=found["t1"],
                t1ce=found["t1ce"],
                t2=found["t2"],
                flair=found["flair"],
                seg=seg_path,
            )
            return case, convention_name, None

        # Track whichever convention found the most files so far, so a
        # totally-missed case reports against the most-likely-intended layout
        # rather than an arbitrary one.
        if len(found) > len(best_found) or best_convention is None:
            best_convention = convention_name
            best_found = found
            best_missing = missing

    return None, None, {best_convention: best_missing} if best_convention else {}


def scan_brats_root(root_dir: str | Path, require_seg: bool = True) -> list[BratsCase]:
    """Discover and resolve every BraTS case under a root directory.

    Case directories are read as immediate subdirectories of `root_dir`,
    sorted by name. Sorting is required for determinism: train/val splits are
    derived from this list's order downstream, and the raw order returned by
    `Path.iterdir()` is filesystem-dependent, so an unsorted scan would
    silently produce different splits on different machines.

    Each case is checked against both the BraTS 2020/2021 (underscore) and
    BraTS 2023+ (hyphen) naming conventions independently, so a directory
    that mixes both styles across cases still resolves correctly.

    Args:
        root_dir: Directory containing one subdirectory per case.
        require_seg: If True (default), every case must have a segmentation
            mask or it is reported as incomplete. If False, `seg` may be
            absent (`BratsCase.seg` is then `None`) — needed because the
            BraTS validation and test sets are distributed without masks, and
            inference still needs to read those cases.

    Returns:
        One `BratsCase` per case directory, sorted by `case_id`.

    Raises:
        FileNotFoundError: If `root_dir` does not exist or is not a directory.
        ValueError: If `root_dir` has no case subdirectories, or if one or
            more cases are missing required files.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"BraTS root directory not found: {root.resolve()}")

    # Sorted, immediate subdirectories only. Hidden (".") and private ("_")
    # prefixed entries are skipped (e.g. ".DS_Store", "__pycache__"), as are
    # any non-directory entries that happen to live at the root.
    case_dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    )

    if not case_dirs:
        raise ValueError(
            f"No case subdirectories found under {root.resolve()}. "
            "Expected one directory per BraTS case."
        )

    cases: list[BratsCase] = []
    problems: dict[str, list[str]] = {}

    for case_dir in case_dirs:
        case, _convention_name, missing_by_convention = _resolve_case(case_dir, require_seg)
        if case is not None:
            cases.append(case)
            continue

        # Incomplete: report the missing roles for whichever convention
        # matched the most files, so the error points at the likely intended
        # layout rather than an arbitrary one.
        if missing_by_convention:
            missing_roles = next(iter(missing_by_convention.values()))
        else:
            missing_roles = list(_ALL_ROLES if require_seg else _MODALITY_ROLES)
        problems[case_dir.name] = missing_roles

    if problems:
        lines = [f"Found {len(problems)} incomplete case(s) under {root.resolve()}:"]
        for case_id in sorted(problems):
            lines.append(f"  {case_id}: missing {', '.join(problems[case_id])}")
        lines.append(
            "Checked both BraTS 2020 (_t1/_t1ce/_t2/_flair/_seg) and\n"
            "BraTS 2023+ (-t1n/-t1c/-t2w/-t2f/-seg) naming conventions."
        )
        raise ValueError("\n".join(lines))

    logger.info(
        "Found %d complete case(s) under %s (require_seg=%s)",
        len(cases),
        root.resolve(),
        require_seg,
    )
    return cases


def write_case_index(cases: Sequence[BratsCase], path: str | Path) -> Path:
    """Write a case index CSV with one row per case.

    Args:
        cases: Cases to write, in the order given.
        path: Destination `.csv` file path. Parent directories are created
            if missing.

    Returns:
        The path written to.
    """
    path = Path(path)
    ensure_dir(path.parent)

    # newline="" is required by the csv module: without it, on Windows (and
    # sometimes elsewhere), the writer's own line terminators get combined
    # with the platform's text-mode newline translation, producing blank
    # rows. Passing newline="" hands newline handling entirely to `csv`.
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "t1", "t1ce", "t2", "flair", "seg"])
        for case in cases:
            writer.writerow(
                [
                    case.case_id,
                    str(case.t1),
                    str(case.t1ce),
                    str(case.t2),
                    str(case.flair),
                    str(case.seg) if case.seg is not None else "",
                ]
            )

    logger.info("Wrote %d case(s) to %s", len(cases), path)
    return path
