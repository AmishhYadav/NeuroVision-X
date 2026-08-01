"""Turn preprocessed BraTS cases into MONAI datasets.

This module sits between offline preprocessing (`preprocessing.py`, which
writes one `image.npy` + `label.npy` pair per case) and the transform
pipeline (`neurovision.data.transforms`, written separately). It knows
nothing about *what* a transform does -- it only accepts one as an argument
-- so it stays usable regardless of how the transforms module evolves.

Three responsibilities:

1. `build_data_dicts` -- turn a list of case ids into the list of dicts
   MONAI transforms expect (`{"image": ..., "label": ..., "case_id": ...}`).
2. `make_splits` / `load_splits` -- a deterministic, seeded, *frozen*
   train/val/test split, written once to YAML and reused forever after so
   that preprocessing more cases later can never silently reshuffle val/test
   under a result that has already been reported.
3. `build_dataset` -- pick the MONAI dataset class (plain / cached /
   persistent) by string, so experiments switch caching strategy from
   config without touching code.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from monai.data import CacheDataset, Dataset, PersistentDataset

from neurovision.utils.io import ensure_dir, read_yaml, write_yaml

logger = logging.getLogger(__name__)

# Small float tolerance for validating that split fractions sum to 1.0 --
# avoids rejecting e.g. 0.7 + 0.15 + 0.15 for ordinary floating point noise.
_FRACTION_TOL = 1e-6


def build_data_dicts(case_ids: Sequence[str], prep_dir: str | Path) -> list[dict[str, str]]:
    """Build MONAI-style data dicts for a list of preprocessed case ids.

    Each case is expected to live at `<prep_dir>/<case_id>/` and contain an
    `image.npy` (required) and a `label.npy` (optional -- unlabeled BraTS
    validation/test cases legitimately lack one).

    Args:
        case_ids: Case identifiers to build dicts for.
        prep_dir: Root directory of preprocessed cases (as written by
            `preprocessing.preprocess_case`).

    Returns:
        One dict per case, with string (not `Path`) values under
        `"image"`, optionally `"label"`, and `"case_id"`. Strings are used
        because MONAI's `LoadImaged` expects string paths and strings
        pickle cleanly across dataloader worker processes.

    Raises:
        FileNotFoundError: If a case's `image.npy` is missing.
    """
    prep_dir = Path(prep_dir)
    data_dicts: list[dict[str, str]] = []
    for case_id in case_ids:
        case_dir = prep_dir / case_id
        image_path = case_dir / "image.npy"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image.npy for case '{case_id}': expected at {image_path.resolve()}"
            )
        entry: dict[str, str] = {"image": str(image_path), "case_id": case_id}
        label_path = case_dir / "label.npy"
        if label_path.is_file():
            entry["label"] = str(label_path)
        data_dicts.append(entry)
    logger.info("Built %d data dicts from %s", len(data_dicts), prep_dir)
    return data_dicts


def make_splits(
    case_ids: Sequence[str],
    out_path: str | Path,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Compute (or reuse) a deterministic train/val/test split.

    The split is written once to `out_path` and then treated as frozen: if
    the file already exists and `overwrite` is False, it is loaded and
    validated against `case_ids` rather than regenerated. This is the whole
    point of the function -- preprocessing more cases later must not
    silently reshuffle val/test underneath a result that was already
    measured against the old split.

    Args:
        case_ids: All case ids to split. Must be non-empty and unique.
        out_path: YAML file the split is read from / written to.
        fractions: (train, val, test) fractions. Must sum to 1.0.
        seed: Seed for the local shuffling RNG. Recorded in the output file.
        overwrite: If True, always regenerate and overwrite `out_path`.

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"`, each a list of case ids.

    Raises:
        ValueError: If `case_ids` is empty or has duplicates, if `fractions`
            is not length 3 or does not sum to 1.0, or if an existing split
            file's case set does not match `case_ids` (see point 2 below).
    """
    if len(case_ids) == 0:
        raise ValueError("make_splits received an empty list of case_ids.")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("make_splits received duplicate case_ids; ids must be unique.")
    if len(fractions) != 3:
        raise ValueError(f"fractions must have length 3 (train, val, test); got {fractions!r}.")
    if abs(sum(fractions) - 1.0) > _FRACTION_TOL:
        raise ValueError(f"fractions must sum to 1.0; got {fractions!r} (sum={sum(fractions)}).")

    out_path = Path(out_path)
    input_ids = set(case_ids)

    if out_path.is_file() and not overwrite:
        existing = load_splits(out_path)
        existing_ids = set(existing["train"]) | set(existing["val"]) | set(existing["test"])
        if existing_ids != input_ids:
            only_in_file = sorted(existing_ids - input_ids)[:5]
            only_in_input = sorted(input_ids - existing_ids)[:5]
            raise ValueError(
                f"Existing split at {out_path.resolve()} has {len(existing_ids)} cases but "
                f"{len(input_ids)} were passed in -- the case sets do not match. "
                f"Examples only in the file: {only_in_file}. "
                f"Examples only in the input: {only_in_input}. "
                "Pass overwrite=True to regenerate the split, but note that this will "
                "invalidate any results already measured against the old split."
            )
        logger.info("Reusing existing split at %s (%d cases).", out_path, len(existing_ids))
        return {"train": existing["train"], "val": existing["val"], "test": existing["test"]}

    # Sort first so the caller's input order can never affect the outcome,
    # then shuffle with a *local* seeded RNG -- never the global `random`
    # module, which any other code touching randomness would perturb.
    shuffled = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    # The test split gets the remainder rather than int(n * fractions[2]) so
    # that every case is assigned exactly once -- rounding down twice could
    # otherwise drop one or two cases on the floor.
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    payload = {
        **splits,
        "meta": {"seed": seed, "fractions": list(fractions), "n_cases": n},
    }
    ensure_dir(out_path.parent)
    write_yaml(payload, out_path)
    logger.info(
        "Wrote new split to %s: %d train / %d val / %d test (seed=%d).",
        out_path,
        len(train_ids),
        len(val_ids),
        len(test_ids),
        seed,
    )
    return splits


def load_splits(path: str | Path) -> dict[str, list[str]]:
    """Load a train/val/test split written by `make_splits`.

    Args:
        path: Path to the split YAML file.

    Returns:
        Dict with keys `"train"`, `"val"`, `"test"`, each a list of case ids.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If any of `"train"`, `"val"`, `"test"` is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path.resolve()}")
    obj = read_yaml(path)
    required = ("train", "val", "test")
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError(f"Split file {path.resolve()} is missing key(s): {missing}")
    return {key: obj[key] for key in required}


_VALID_DATASET_TYPES = ("dataset", "cache", "persistent")


def build_dataset(
    data_dicts: Sequence[dict[str, str]],
    transform: Any,
    dataset_type: str = "dataset",
    cache_rate: float = 0.0,
    cache_dir: str | Path | None = None,
    num_workers: int = 0,
) -> Dataset:
    """Build the MONAI dataset selected by `dataset_type`.

    Selecting the class by string lets experiments switch caching strategy
    purely from config (`data.dataset_type=cache`, etc.) without code
    changes.

    Args:
        data_dicts: List of data dicts, e.g. from `build_data_dicts`.
        transform: A MONAI (or MONAI-compatible) transform applied per item.
        dataset_type: One of `"dataset"` (no caching), `"cache"`
            (`CacheDataset`, RAM), or `"persistent"` (`PersistentDataset`,
            disk).
        cache_rate: Fraction of items cached in RAM. Only used when
            `dataset_type == "cache"`.
        cache_dir: Directory for on-disk cache. Required when
            `dataset_type == "persistent"`.
        num_workers: Worker processes used to build the cache. Only used
            when `dataset_type == "cache"`.

    Returns:
        A `monai.data.Dataset`, `CacheDataset`, or `PersistentDataset`.

    Raises:
        ValueError: If `dataset_type` is not one of the valid strings, or if
            `dataset_type == "persistent"` and `cache_dir` is None.
    """
    if dataset_type == "dataset":
        ds = Dataset(data=data_dicts, transform=transform)
    elif dataset_type == "cache":
        ds = CacheDataset(
            data=data_dicts,
            transform=transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    elif dataset_type == "persistent":
        if cache_dir is None:
            raise ValueError("dataset_type='persistent' requires cache_dir, but got None.")
        cache_dir = ensure_dir(cache_dir)
        ds = PersistentDataset(data=data_dicts, transform=transform, cache_dir=cache_dir)
    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type!r}. Valid options: {_VALID_DATASET_TYPES}."
        )

    logger.info(
        "Built %s with %d items.",
        type(ds).__name__,
        len(data_dicts),
    )
    return ds
