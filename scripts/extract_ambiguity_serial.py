"""Serial, resumable driver for `scripts/extract_ambiguity.py`.

Why this exists
---------------
Whole-volume ambiguity extraction is a sliding-window job over the full
34.9M-parameter dual-encoder network, on a 16 GiB MacBook. Two earlier
attempts ran several extraction workers in parallel and drove the machine
into swap: at `sw_batch_size=4` one worker peaks at 12.39 GiB, so two cannot
coexist, and the config default `data.num_workers=4` forks four dataloader
subprocesses *per* worker. Both failures are recorded in CLAUDE.md.

So this driver does exactly one thing: it runs ONE extraction process at a
time, with the memory-safe overrides pinned here rather than retyped per
invocation, and it skips every case that already has a `.npz` on disk. A
killed run therefore costs at most the chunk in flight.

Chunking (`--chunk`) exists only so that peak RSS is released back to the OS
periodically and so a crash loses a few cases rather than the whole cohort.
Each chunk writes its own output directory; `scripts/detection_stats.py`
already reads a LIST of shard directories and refuses duplicate case ids, so
no merge step is needed or wanted.

Usage
-----
    python scripts/extract_ambiguity_serial.py --cohort ssa
    python scripts/extract_ambiguity_serial.py --cohort test --chunk 6
    python scripts/extract_ambiguity_serial.py --cohort ped --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Threads are capped well below the core count on purpose: this job runs on the
# author's daily-driver laptop, and a torch process that grabs every core makes
# the machine unusable for the hours the cohort takes.
THREAD_CAP = "4"

# One process at a time, `sw_batch_size=1`, `num_workers=0`. Measured: 5.42 GiB
# peak RSS and ~99 s/case, versus 12.39 GiB and 157 s/case at sw_batch_size=4.
MEMORY_SAFE_OVERRIDES = [
    "data.num_workers=0",
    "inference.sliding_window.sw_batch_size=1",
]


@dataclass(frozen=True)
class Cohort:
    """One evaluation cohort and everything that differs about it."""

    name: str
    splits_path: str
    prep_dir: str
    logits_dir: str
    out_prefix: str
    # Which split inside `splits_path` to read. The external cohorts put all
    # their cases in "test"; "val" exists so the Gate 2 combiner can be fitted
    # on data no reported number comes from (see
    # docs/research/preregistration_gate2.md).
    split: str = "test"
    extra: list[str] = field(default_factory=list)


COHORTS = {
    "test": Cohort(
        name="test",
        splits_path="configs/data/splits.yaml",
        prep_dir="data/preprocessed/brats",
        logits_dir="outputs/neurovision/eval_test/logits",
        out_prefix="outputs/ambiguity_test",
    ),
    "val": Cohort(
        name="val",
        splits_path="configs/data/splits.yaml",
        prep_dir="data/preprocessed/brats",
        logits_dir="outputs/neurovision/eval_val/logits",
        out_prefix="outputs/ambiguity_val",
        split="val",
    ),
    "ssa": Cohort(
        name="ssa",
        splits_path="configs/data/splits_ssa.yaml",
        prep_dir="data/preprocessed/brats_ssa",
        logits_dir="outputs/eval_ssa_neurovision/logits",
        out_prefix="outputs/ambiguity_ssa",
    ),
    "ped": Cohort(
        name="ped",
        splits_path="configs/data/splits_ped.yaml",
        prep_dir="data/preprocessed/brats_ped",
        logits_dir="outputs/eval_ped_neurovision/logits",
        out_prefix="outputs/ambiguity_ped",
    ),
}


def split_case_ids(cohort: Cohort) -> list[str]:
    """Case ids of the cohort's frozen split, in split-file order."""
    with (REPO_ROOT / cohort.splits_path).open(encoding="utf-8") as handle:
        splits = yaml.safe_load(handle)
    return [str(c) for c in splits[cohort.split]]


def done_case_ids(cohort: Cohort) -> dict[str, str]:
    """Map case_id -> shard directory name, over every existing shard dir.

    Any directory whose name starts with the cohort's output prefix counts,
    including the hand-made `_s0`/`_w0` shards from the parallel attempts --
    their `.npz` files are perfectly good and must not be recomputed.
    """
    done: dict[str, str] = {}
    for shard in sorted((REPO_ROOT / "outputs").glob(f"{Path(cohort.out_prefix).name}*")):
        if not shard.is_dir():
            continue
        for npz in shard.glob("*.npz"):
            done[npz.stem] = shard.name
    return done


def next_out_dir(cohort: Cohort, reserved: set[Path]) -> Path:
    """First unused `<prefix>_pNN` directory, so chunks never collide.

    `reserved` holds the directories this process has already handed out --
    a --dry-run creates nothing, so existence on disk alone would return the
    same name for every chunk.
    """
    index = 0
    while True:
        candidate = REPO_ROOT / f"{cohort.out_prefix}_p{index:02d}"
        if not candidate.exists() and candidate not in reserved:
            reserved.add(candidate)
            return candidate
        index += 1


def build_command(cohort: Cohort, case_ids: list[str], out_dir: Path) -> list[str]:
    """Hydra command line for one chunk."""
    joined = ",".join(case_ids)
    return [
        sys.executable,
        "scripts/extract_ambiguity.py",
        "+experiment=neurovision",
        f"data.splits.path={cohort.splits_path}",
        f"data.preprocessing.out_dir={cohort.prep_dir}",
        *MEMORY_SAFE_OVERRIDES,
        *cohort.extra,
        f"explainability.ambiguity.split={cohort.split}",
        "explainability.ambiguity.checkpoint=outputs/neurovision/checkpoints/best.pt",
        f"explainability.ambiguity.logits_dir={cohort.logits_dir}",
        f"explainability.ambiguity.out_dir={out_dir}",
        f"explainability.ambiguity.case_ids=[{joined}]",
    ]


def run_chunk(command: list[str]) -> int:
    """Run one extraction process to completion and return its exit code."""
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=THREAD_CAP,
        MKL_NUM_THREADS=THREAD_CAP,
        VECLIB_MAXIMUM_THREADS=THREAD_CAP,
        NUMEXPR_NUM_THREADS=THREAD_CAP,
        # Hydra would otherwise write a run directory per invocation.
        HYDRA_FULL_ERROR="1",
    )
    # `nice` keeps the laptop usable while the cohort runs; this job is never
    # the interactive foreground task.
    return subprocess.run(["nice", "-n", "10", *command], cwd=REPO_ROOT, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=sorted(COHORTS))
    parser.add_argument("--chunk", type=int, default=6, help="Cases per process (default 6).")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N cases this run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cohort = COHORTS[args.cohort]
    wanted = split_case_ids(cohort)
    done = done_case_ids(cohort)
    remaining = [c for c in wanted if c not in done]
    if args.limit is not None:
        remaining = remaining[: args.limit]

    print(f"[{cohort.name}] split={len(wanted)} done={len(done)} remaining={len(remaining)}")
    if not remaining:
        print(f"[{cohort.name}] nothing to do.")
        return 0

    chunks = [remaining[i : i + args.chunk] for i in range(0, len(remaining), args.chunk)]
    reserved: set[Path] = set()
    for position, chunk in enumerate(chunks, start=1):
        out_dir = next_out_dir(cohort, reserved)
        command = build_command(cohort, chunk, out_dir)
        print(f"\n[{cohort.name}] chunk {position}/{len(chunks)} -> {out_dir.name}: {chunk}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        started = time.time()
        code = run_chunk(command)
        elapsed = time.time() - started
        print(f"[{cohort.name}] chunk {position} exit={code} in {elapsed / 60:.1f} min")
        if code != 0:
            print(f"[{cohort.name}] STOPPING: chunk {position} failed. Rerun to resume.")
            return code

    print(f"\n[{cohort.name}] all chunks done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
