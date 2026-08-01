"""Generate (or verify) the frozen train/val/test split file.

Splits are load-bearing: every reported number is measured against the cases
in `configs/data/splits.yaml`, and that file is frozen once written (see
`neurovision.data.dataset.make_splits`'s docstring). This script is the
reproducible entry point for producing it, so nobody has to remember an
ad-hoc one-off command to (re)create it.

This is a Hydra script, so `data.root_dir` is a mandatory `???` in
`configs/data/brats.yaml` and must be supplied on the command line for the
config to compose at all -- even though this script never reads it. The
split is computed from what has ACTUALLY been preprocessed on disk
(`cfg.data.preprocessing.out_dir`), not from the raw BraTS directory.

Example usage:

    python scripts/make_splits.py data.root_dir=data/raw/BraTS2021_Training_Data

To deliberately regenerate an existing split (invalidates any result already
measured against the old one):

    python scripts/make_splits.py data.root_dir=... data.splits.overwrite=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from neurovision.data.dataset import make_splits
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/preprocess.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def discover_case_ids(prep_dir: str | Path) -> list[str]:
    """Find case ids that have been successfully preprocessed on disk.

    A case counts as valid only if `<prep_dir>/<case_id>/` contains BOTH
    `image.npy` and `meta.json`. `build_data_dicts` raises `FileNotFoundError`
    on a case missing `image.npy`, but only later -- at the start of a
    training run, potentially a Kaggle run that has already paid to install
    dependencies and load data. Checking here, before a split is ever
    written, means a split can never name a case with no usable output.

    The returned list is sorted. `make_splits` seeds its own shuffle, but
    feeding it an unsorted, filesystem-order list would make the resulting
    split depend on directory iteration order -- which is not guaranteed
    stable across machines or filesystems. Sorting first removes that
    silent nondeterminism.

    Args:
        prep_dir: Root directory of preprocessed cases, as written by
            `scripts/preprocess.py` (`cfg.data.preprocessing.out_dir`).

    Returns:
        Sorted list of valid case ids.

    Raises:
        FileNotFoundError: If `prep_dir` does not exist.
        ValueError: If `prep_dir` exists but contains zero valid cases.
    """
    prep_dir = Path(prep_dir)
    if not prep_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {prep_dir.resolve()}. "
            "Has scripts/preprocess.py been run yet?"
        )

    case_ids: list[str] = []
    for entry in sorted(prep_dir.iterdir()):
        if not entry.is_dir():
            continue
        has_image = (entry / "image.npy").is_file()
        has_meta = (entry / "meta.json").is_file()
        if has_image and has_meta:
            case_ids.append(entry.name)
        else:
            missing = [
                name
                for name, present in (("image.npy", has_image), ("meta.json", has_meta))
                if not present
            ]
            logger.warning("Skipping '%s': missing %s.", entry.name, ", ".join(missing))

    if not case_ids:
        raise ValueError(
            f"No valid preprocessed cases found under {prep_dir.resolve()} "
            "(each case needs both image.npy and meta.json). "
            "Has scripts/preprocess.py been run yet?"
        )

    return sorted(case_ids)


def run_make_splits(cfg: DictConfig) -> dict[str, list[str]]:
    """Discover preprocessed cases and produce (or reuse) the frozen split.

    Args:
        cfg: Composed Hydra config. Reads `cfg.data.preprocessing.out_dir` and
            `cfg.data.splits.{path,fractions,seed,overwrite}`.

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"`, each a list of case ids.
    """
    out_dir = cfg.data.preprocessing.out_dir
    case_ids = discover_case_ids(out_dir)

    splits_path = Path(cfg.data.splits.path)
    # Checked BEFORE calling make_splits, which may write the file as a side
    # effect -- this is the only way to know afterwards whether the file was
    # just written or was already there and simply reloaded.
    existed_before = splits_path.is_file()

    fractions = tuple(cfg.data.splits.fractions)
    seed = int(cfg.data.splits.seed)
    overwrite = bool(cfg.data.splits.get("overwrite", False))

    splits = make_splits(
        case_ids,
        splits_path,
        fractions=fractions,
        seed=seed,
        overwrite=overwrite,
    )

    was_reused = existed_before and not overwrite
    n_total = len(case_ids)

    lines = ["=" * 70, "Split summary", "=" * 70]
    if was_reused:
        lines.append(f"FROZEN: loaded existing split from {splits_path} (nothing regenerated)")
    else:
        lines.append(f"WROTE new split to {splits_path}")
    lines.append(f"Total cases discovered: {n_total}")
    for split_name in ("train", "val", "test"):
        n = len(splits[split_name])
        pct = 100.0 * n / n_total if n_total else 0.0
        lines.append(f"  {split_name:5s}: {n:5d} ({pct:.1f}%)")
    lines.append(f"Fractions: {fractions}")
    lines.append(f"Seed: {seed}")
    lines.append("=" * 70)
    summary = "\n".join(lines)

    # print only, not logger.info as well: setup_logging's StreamHandler
    # already targets stdout, so doing both emits this block twice and reads
    # like the script ran twice. Matches scripts/preprocess.py's summary.
    print(summary)

    return splits


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: discover preprocessed cases and write the split file.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    run_make_splits(cfg)


if __name__ == "__main__":
    main()
