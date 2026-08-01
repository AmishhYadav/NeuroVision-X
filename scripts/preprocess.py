"""Offline BraTS preprocessing CLI.

Scans a raw BraTS directory, normalizes/crops/remaps every case with
`neurovision.data.preprocessing.preprocess_case`, and writes the results to
`.npy` files under a configured output directory. This runs once, locally, on
the Mac's CPU -- Kaggle training later reads only the small `.npy` files this
script produces, never the raw NIfTI volumes.

Example usage (see the bottom of this file for the full smoke-test command):

    python scripts/preprocess.py data.root_dir=/path/to/brats2021

This is a CLI entry point, so it prints a human-facing summary to stdout on
purpose -- the no-bare-print rule applies to library code under src/, not to
scripts whose whole job is producing terminal output. Progress and warnings
still go through `logging`, via `setup_logging`.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from neurovision.data.brats import BratsCase, scan_brats_root, write_case_index
from neurovision.data.preprocessing import preprocess_case
from neurovision.utils.io import (
    KAGGLE_DATASET_WARN_GB,
    directory_size_bytes,
    ensure_dir,
    format_size,
)
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/show_config.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# directory_size_bytes / format_size / KAGGLE_DATASET_WARN_GB now live in
# neurovision.utils.io (imported above) so that scripts/package_for_kaggle.py
# can share the exact same size arithmetic instead of a second, possibly
# drifting, implementation. Re-exported here as module attributes (via the
# import above) so tests/test_preprocess_script.py keeps working unchanged.


def _process_one(args: tuple[BratsCase, str, bool]) -> dict[str, Any]:
    """Preprocess a single case inside a worker process.

    Must be a plain module-level function (not a lambda or a closure) so it
    can be pickled and shipped to the child process -- that pickling is what
    makes `ProcessPoolExecutor` work at all. Any exception raised while
    processing this one case is caught here and turned into a failure record
    instead of propagating, because losing the entire run at case 1200 of
    1251 over one corrupt volume is unacceptable.

    Args:
        args: `(case, out_dir, overwrite)`, bundled into one tuple since
            `executor.submit` takes a single argument list per call.

    Returns:
        The summary dict from `preprocess_case` on success, or
        `{"case_id": ..., "error": ...}` on failure.
    """
    case, out_dir, overwrite = args
    try:
        return preprocess_case(case, out_dir, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - any failure must become a record, not crash the pool
        logger.warning("Case %s failed: %s", case.case_id, exc)
        return {"case_id": case.case_id, "error": str(exc)}


def summarize(summaries: Sequence[dict[str, Any]], out_dir: str | Path) -> dict[str, Any]:
    """Aggregate per-case summary dicts into one run-level report.

    Args:
        summaries: One dict per case, either a success summary from
            `preprocess_case` (has `"skipped"`, `"cropped_shape"`,
            `"n_class_0"`..`"n_class_3"`, etc.) or a failure record
            (`{"case_id": ..., "error": ...}`).
        out_dir: Output root directory, walked to measure total size on disk.

    Returns:
        A dict with:
        - `n_found`: number of cases in `summaries`.
        - `n_processed`: successes with `skipped=False`.
        - `n_skipped`: successes with `skipped=True` (already present).
        - `n_failed`: failure records.
        - `failures`: list of `{"case_id", "error"}` for every failure.
        - `total_size_bytes` / `total_size_str`: size of `out_dir` on disk.
        - `mean_cropped_shape` / `median_cropped_shape`: per-axis `(D, H, W)`
          tuples of floats across successful cases, or `None` if there were
          no successes (avoids a division-by-zero from an empty `statistics`
          call).
        - `voxel_totals`: dict with `n_class_1`, `n_class_2`, `n_class_3`
          summed across successful cases. Class 0 (background) is omitted --
          it dwarfs the tumor classes and isn't informative here.
    """
    successes = [s for s in summaries if "error" not in s]
    failures = [s for s in summaries if "error" in s]

    n_processed = sum(1 for s in successes if not s.get("skipped", False))
    n_skipped = sum(1 for s in successes if s.get("skipped", False))

    voxel_totals = {"n_class_1": 0, "n_class_2": 0, "n_class_3": 0}
    for s in successes:
        for cls in (1, 2, 3):
            voxel_totals[f"n_class_{cls}"] += s.get(f"n_class_{cls}", 0)

    cropped_shapes = [s["cropped_shape"] for s in successes if "cropped_shape" in s]
    if cropped_shapes:
        # zip(*shapes) transposes a list of (D, H, W) tuples into three
        # per-axis sequences, so mean/median are computed per spatial axis
        # rather than pooling D, H, W values together.
        mean_shape: tuple[float, ...] | None = tuple(
            statistics.mean(axis_values) for axis_values in zip(*cropped_shapes)
        )
        median_shape: tuple[float, ...] | None = tuple(
            statistics.median(axis_values) for axis_values in zip(*cropped_shapes)
        )
    else:
        mean_shape = None
        median_shape = None

    total_size_bytes = directory_size_bytes(out_dir)

    return {
        "n_found": len(summaries),
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "n_failed": len(failures),
        "failures": [
            {"case_id": f.get("case_id", "<unknown>"), "error": f.get("error", "")}
            for f in failures
        ],
        "total_size_bytes": total_size_bytes,
        "total_size_str": format_size(total_size_bytes),
        "mean_cropped_shape": mean_shape,
        "median_cropped_shape": median_shape,
        "voxel_totals": voxel_totals,
    }


def _print_summary(stats: dict[str, Any]) -> None:
    """Print the final human-facing summary block for a preprocessing run."""
    print("=" * 70)
    print("Preprocessing summary")
    print("=" * 70)
    print(f"Cases found:     {stats['n_found']}")
    print(f"Cases processed: {stats['n_processed']}")
    print(f"Cases skipped:   {stats['n_skipped']}  (already present)")
    print(f"Cases failed:    {stats['n_failed']}")
    print()
    print(f"Total output size: {stats['total_size_str']}")

    mean_shape = stats["mean_cropped_shape"]
    median_shape = stats["median_cropped_shape"]
    if mean_shape is not None:
        mean_str = ", ".join(f"{v:.1f}" for v in mean_shape)
        median_str = ", ".join(f"{v:.1f}" for v in median_shape)
        print(f"Cropped shape (D, H, W) - mean:   ({mean_str})")
        print(f"Cropped shape (D, H, W) - median: ({median_str})")
    else:
        print("Cropped shape - no successful cases to summarize.")

    print()
    print("Tumor voxel counts (background/class 0 omitted):")
    voxel_totals = stats["voxel_totals"]
    print(f"  class_1 (NCR/NET): {voxel_totals['n_class_1']}")
    print(f"  class_2 (ED):      {voxel_totals['n_class_2']}")
    print(f"  class_3 (ET):      {voxel_totals['n_class_3']}")

    if stats["n_failed"]:
        print()
        print(f"FAILURES ({stats['n_failed']}):")
        for failure in stats["failures"]:
            print(f"  {failure['case_id']}: {failure['error']}")

    # Not a hard ceiling -- Kaggle allows 200 GB per dataset. This is a
    # "that is a long upload, confirm you meant it" nudge, so it says so
    # rather than claiming the upload will fail.
    total_gb = stats["total_size_bytes"] / (1024**3)
    if total_gb > KAGGLE_DATASET_WARN_GB:
        print()
        print("!" * 70)
        print(
            f"NOTE: total output size ({stats['total_size_str']}) is over the "
            f"{KAGGLE_DATASET_WARN_GB:.0f} GB review threshold. Kaggle's per-dataset "
            "limit is 200 GB, so this should still upload -- but budget the time, "
            "and use data.preprocessing.limit=N if you only meant to process a subset."
        )
        print("!" * 70)

    print("=" * 70)


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Preprocess every BraTS case under `cfg.data.root_dir`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
            Reads `cfg.data.root_dir` and the `cfg.data.preprocessing.*`
            settings described in `configs/data/brats.yaml`.
    """
    setup_logging(level="INFO")

    cases = scan_brats_root(cfg.data.root_dir, require_seg=cfg.data.preprocessing.require_seg)

    limit = cfg.data.preprocessing.limit
    if limit is not None:
        # scan_brats_root already returns cases sorted by case_id, so taking
        # the first N here is deterministic -- the same limit always selects
        # the same subset, which is what makes a "preprocess 5 cases" smoke
        # test reproducible.
        cases = cases[: int(limit)]

    out_dir = ensure_dir(cfg.data.preprocessing.out_dir)
    write_case_index(cases, out_dir / "case_index.csv")

    overwrite = bool(cfg.data.preprocessing.overwrite)
    num_workers = int(cfg.data.preprocessing.num_workers)

    summaries: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_one, (case, str(out_dir), overwrite)): case for case in cases
        }
        # as_completed (not executor.map) so the progress bar advances as
        # each case actually finishes, in whatever order that happens to be,
        # rather than waiting to report results in submission order.
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Preprocessing"):
            summaries.append(future.result())

    metadata_path = out_dir / cfg.data.preprocessing.metadata_csv
    successes = [s for s in summaries if "error" not in s]
    pd.DataFrame(successes).to_csv(metadata_path, index=False)
    logger.info("Wrote metadata for %d case(s) to %s", len(successes), metadata_path)

    stats = summarize(summaries, out_dir)
    _print_summary(stats)


if __name__ == "__main__":
    # macOS's default multiprocessing start method is "spawn": every worker
    # process re-imports this module from scratch, rather than inheriting the
    # parent's memory via fork (Linux/Kaggle's default). If the real work sat
    # at module level instead of behind this guard, each spawned worker would
    # re-run the whole scan-and-dispatch pipeline recursively. Keeping
    # everything behind `if __name__ == "__main__":` and the `main()` call is
    # what makes this script behave the same on the Mac as it will on Kaggle.
    main()
