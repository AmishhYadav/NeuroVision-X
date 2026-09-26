"""Generate the frozen cross-fit split files for the D3 external-cohort fine-tune.

D3 (`docs/research/preregistration_finetune.md`, "Design -> Cross-fitting") fine-tunes
on each external cohort (SSA, PED) using two-fold cross-fitting: each cohort is split
once into two halves, a model is fine-tuned on each half in turn, and scored on the
OTHER half. Pooling both folds' held-out predictions scores every case in the cohort
exactly once, by a model that never trained on it.

The two source files this reads (`configs/data/splits_ssa.yaml`,
`configs/data/splits_ped.yaml`) put every case in `test` and keep `train`/`val` empty
-- see their own header comments -- because nothing may ever be fitted on them. This
script is the ONLY place that is allowed to carve those cohorts into train/val/test,
and it does so into brand-new files (`configs/data/splits_{cohort}_cf{fold}.yaml`); the
two source files are read-only here and are never touched.

Like `scripts/make_splits.py`, this is a Hydra script composed off `configs/config.yaml`,
so `data.root_dir` is a mandatory `???` in `configs/data/brats.yaml` and must be supplied
on the command line for the config to compose at all -- even though this script never
reads it (it only reads `cfg.analysis.crossfit_splits`).

Per `docs/research/preregistration_finetune.md`, every split file written here is FROZEN
the moment it is committed: never regenerate it once a fine-tune has used it. Re-running
this script requires the explicit `analysis.crossfit_splits.overwrite=true` flag.

Example usage:

    python scripts/make_crossfit_splits.py data.root_dir=data/raw/BraTS2021_Training_Data

To deliberately regenerate (invalidates any D3 number already measured against the old
split files):

    python scripts/make_crossfit_splits.py data.root_dir=... analysis.crossfit_splits.overwrite=true
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from neurovision.data.dataset import load_splits
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/make_splits.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# The pre-registered fine-tune budget: ~3,000 optimizer steps regardless of
# cohort size, at one step per training case per epoch
# (batch_size=1 x samples_per_volume=4). See preregistration_finetune.md,
# "Fine-tune recipe, fixed now".
_STEP_BUDGET = 3000


def load_cohort_test_ids(source_path: str | Path) -> list[str]:
    """Loads and validates a cohort's frozen, unsplit source file.

    An external-cohort source file such as `configs/data/splits_ssa.yaml` puts
    every case in `test` and keeps `train`/`val` empty -- that emptiness is the
    guarantee that nothing has ever been fitted on the cohort. This function
    checks that guarantee still holds before this script is allowed to carve
    the cohort into cross-fit folds.

    Args:
        source_path: Path to the cohort's source split YAML.

    Returns:
        Sorted list of case ids from the source file's `test` list.

    Raises:
        ValueError: If `train` or `val` is non-empty, or `test` is empty.
    """
    source_path = Path(source_path)
    splits = load_splits(source_path)
    if splits["train"] or splits["val"]:
        raise ValueError(
            f"Source split {source_path} must have an empty train/val (found "
            f"{len(splits['train'])} train, {len(splits['val'])} val case(s)) -- "
            "cross-fit splits are only defined for a cohort whose source file "
            "puts every case in test. Nothing may already be fitted on it."
        )
    if not splits["test"]:
        raise ValueError(f"Source split {source_path} has an empty test list.")
    return sorted(splits["test"])


def build_cohort_folds(
    case_ids: list[str],
    n_folds: int,
    val_n: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Splits one cohort's case ids into `n_folds` cross-fit folds.

    The cohort is sorted, then permuted with a single seeded
    `np.random.default_rng(seed)` and cut into `n_folds` contiguous,
    near-equal halves (`np.array_split`). Fold `f` trains on half `f` (minus
    `val_n` monitoring cases carved from the SAME rng, drawn in fold order
    right after the permutation) and is scored on every case NOT in half `f`.

    Building one fresh generator per call is what makes a cohort's fold
    assignment independent of which OTHER cohorts happen to be configured --
    two cohorts never share rng state.

    Args:
        case_ids: All case ids in the cohort (need not be pre-sorted).
        n_folds: Number of cross-fit folds (halves).
        val_n: Monitoring-only cases carved from each fold's training half.
        seed: Seed for this cohort's `np.random.default_rng`.

    Returns:
        List of length `n_folds`, each a dict with sorted `"train"`,
        `"val"`, `"test"` case-id lists.

    Raises:
        ValueError: If `val_n` is larger than any fold's half.
    """
    sorted_ids = sorted(case_ids)
    rng = np.random.default_rng(seed)
    # Permute first, THEN carve val off each half in fold order -- both steps
    # share this one rng, so the whole cohort's assignment is one deterministic
    # sequence of draws, never something that depends on iteration order
    # elsewhere.
    permuted = rng.permutation(np.array(sorted_ids))
    halves = np.array_split(permuted, n_folds)

    folds: list[dict[str, list[str]]] = []
    for fold in range(n_folds):
        half = halves[fold]
        if val_n > len(half):
            raise ValueError(f"val_n={val_n} exceeds fold {fold}'s half size ({len(half)} cases).")
        val_ids = rng.choice(half, size=val_n, replace=False)
        val_set = {str(case_id) for case_id in val_ids}
        train_ids = sorted(str(case_id) for case_id in half if str(case_id) not in val_set)
        val_list = sorted(str(case_id) for case_id in val_ids)

        # test = every case NOT in this fold's half, i.e. every other half --
        # for n_folds=2 that is simply "the other half".
        other_halves = [halves[k] for k in range(n_folds) if k != fold]
        other = np.concatenate(other_halves) if other_halves else np.array([], dtype=permuted.dtype)
        test_list = sorted(str(case_id) for case_id in other)

        folds.append({"train": train_ids, "val": val_list, "test": test_list})
    return folds


def _fold_header(
    *, cohort: str, fold: int, n_folds: int, seed: int, val_n: int, source_path: Path
) -> str:
    """Builds the comment header block written above each fold's YAML body."""
    return (
        f"# Cross-fit split -- cohort '{cohort}', fold {fold} of {n_folds}.\n"
        "# Generated by scripts/make_crossfit_splits.py "
        f"(seed={seed}, n_folds={n_folds}, val_n={val_n}),\n"
        f"# from source {source_path.as_posix()}.\n"
        "#\n"
        "# FROZEN -- committed before any D3 fine-tune; never regenerate\n"
        "# (docs/research/preregistration_finetune.md).\n"
        "#\n"
        "# These numbers are cross-fitted, NOT external validation: this\n"
        "# fold's model is fine-tuned on cases from the SAME cohort it is\n"
        "# later scored on (just not these particular ones), so a result on\n"
        "# `test` here must always be reported as cross-fitted.\n"
        "#\n"
        "# `val` is monitoring-only -- the pre-registration scores last.pt,\n"
        "# so val never selects a checkpoint.\n"
    )


def _write_fold_split(out_path: Path, fold_data: dict[str, list[str]], header: str) -> None:
    """Writes one fold's split YAML: the header comment block, then train/val/test."""
    ensure_dir(out_path.parent)
    # Not neurovision.utils.io.write_yaml: that helper has no way to attach a
    # leading comment block, and the header is exactly what makes this file
    # self-documenting as FROZEN once it lands in the repo (matches the
    # hand-authored style of splits_ssa.yaml / splits_ped.yaml).
    body = yaml.safe_dump(
        {"train": fold_data["train"], "val": fold_data["val"], "test": fold_data["test"]},
        sort_keys=False,
        default_flow_style=False,
    )
    out_path.write_text(header + body, encoding="utf-8")


def _write_summary(
    summary_path: Path, rows: list[dict[str, Any]], *, seed: int, n_folds: int, val_n: int
) -> None:
    """Writes the sidecar summary (n_train/n_val/n_test/epochs per cohort x fold)."""
    header = (
        "# Cross-fit split summary, generated by scripts/make_crossfit_splits.py\n"
        f"# (seed={seed}, n_folds={n_folds}, val_n={val_n}).\n"
        "#\n"
        "# FROZEN -- committed before any D3 fine-tune; never regenerate\n"
        "# (docs/research/preregistration_finetune.md).\n"
        "#\n"
        "# `epochs` is the pre-registered budget rule round(3000 / n_train)\n"
        "# (preregistration_finetune.md, 'Fine-tune recipe, fixed now') -- the\n"
        "# D3 launch reads this fixed number directly, it is not recomputed at\n"
        "# launch time.\n"
    )
    ensure_dir(summary_path.parent)
    body = yaml.safe_dump(rows, sort_keys=False, default_flow_style=False)
    summary_path.write_text(header + body, encoding="utf-8")


def run_make_crossfit_splits(cfg: DictConfig) -> dict[str, Any]:
    """Builds and writes every cohort's cross-fit split files, plus the summary sidecar.

    Args:
        cfg: Composed config. Reads `cfg.analysis.crossfit_splits.{n_folds,
            seed, val_n, cohorts, out_pattern, overwrite}`. `overwrite`
            defaults to False when absent.

    Returns:
        Dict with `"folds"` (cohort -> fold index -> {"train","val","test"})
        and `"summary"` (the list of per-fold row dicts written to the
        sidecar file).

    Raises:
        ValueError: Via `load_cohort_test_ids` / `build_cohort_folds`, if a
            source file is malformed.
        FileExistsError: If any target file already exists and
            `analysis.crossfit_splits.overwrite` is not True.
    """
    cf_cfg = cfg.analysis.crossfit_splits
    n_folds = int(cf_cfg.n_folds)
    seed = int(cf_cfg.seed)
    val_n = int(cf_cfg.val_n)
    out_pattern = str(cf_cfg.out_pattern)
    overwrite = bool(cf_cfg.get("overwrite", False))

    # The sidecar lives alongside the per-fold files rather than at a second,
    # independently hardcoded config path -- deriving it from out_pattern's
    # own directory is what keeps this "from config", per CLAUDE.md's no-
    # hardcoded-paths rule, while still landing at the exact path
    # (configs/data/splits_crossfit_summary.yaml) the pre-registration names.
    summary_path = Path(out_pattern).parent / "splits_crossfit_summary.yaml"

    # Pass 1: compute every fold for every cohort, and every target path, but
    # write NOTHING yet. This is what lets the overwrite check below refuse
    # cleanly -- one cohort's files are never partially written before a
    # conflict on a later cohort is discovered.
    folds_by_cohort: dict[str, list[dict[str, list[str]]]] = {}
    fold_paths: dict[tuple[str, int], Path] = {}
    for cohort, source_rel in cf_cfg.cohorts.items():
        source_path = Path(str(source_rel))
        case_ids = load_cohort_test_ids(source_path)
        folds = build_cohort_folds(case_ids, n_folds=n_folds, val_n=val_n, seed=seed)
        for fold, fold_data in enumerate(folds):
            if len(fold_data["train"]) <= 0:
                # Caught here, before pass 2 writes a single byte: val_n equal
                # to (not just greater than) a half's size passes
                # build_cohort_folds's own check but leaves nothing to train
                # on, which the round(3000 / n_train) budget rule below
                # cannot divide by.
                raise ValueError(
                    f"Cohort '{cohort}' fold {fold} has 0 training cases after removing "
                    f"val_n={val_n} monitoring cases -- val_n is too large for this "
                    "cohort's fold size."
                )
        folds_by_cohort[cohort] = folds
        for fold in range(n_folds):
            fold_paths[(cohort, fold)] = Path(out_pattern.format(cohort=cohort, fold=fold))

    all_targets = list(fold_paths.values()) + [summary_path]
    if not overwrite:
        for target in all_targets:
            if target.is_file():
                raise FileExistsError(
                    f"Refusing to overwrite existing cross-fit split file: {target}. "
                    "Pass analysis.crossfit_splits.overwrite=true to regenerate, but "
                    "note this invalidates any D3 result already measured against it."
                )

    # Pass 2: write every fold file, then the summary sidecar.
    rows: list[dict[str, Any]] = []
    for cohort, source_rel in cf_cfg.cohorts.items():
        source_path = Path(str(source_rel))
        for fold in range(n_folds):
            fold_data = folds_by_cohort[cohort][fold]
            header = _fold_header(
                cohort=cohort,
                fold=fold,
                n_folds=n_folds,
                seed=seed,
                val_n=val_n,
                source_path=source_path,
            )
            _write_fold_split(fold_paths[(cohort, fold)], fold_data, header)

            n_train = len(fold_data["train"])
            n_val = len(fold_data["val"])
            n_test = len(fold_data["test"])
            # round(), not int(): the budget rule in preregistration_finetune.md
            # is stated as round(3000 / n_train), and rounding down would
            # silently under-train every fold by up to half an epoch's worth
            # of steps.
            epochs = round(_STEP_BUDGET / n_train)
            rows.append(
                {
                    "cohort": cohort,
                    "fold": fold,
                    "n_train": n_train,
                    "n_val": n_val,
                    "n_test": n_test,
                    "epochs": epochs,
                }
            )

    _write_summary(summary_path, rows, seed=seed, n_folds=n_folds, val_n=val_n)

    lines = ["=" * 78, "Cross-fit split summary", "=" * 78]
    lines.append(
        f"{'cohort':8s} {'fold':5s} {'n_train':8s} {'n_val':6s} {'n_test':7s} {'epochs':7s}"
    )
    for row in rows:
        lines.append(
            f"{row['cohort']:8s} {row['fold']:<5d} {row['n_train']:<8d} "
            f"{row['n_val']:<6d} {row['n_test']:<7d} {row['epochs']:<7d}"
        )
    lines.append(f"Wrote {len(fold_paths)} split file(s) and 1 summary to {summary_path}.")
    lines.append("=" * 78)
    for line in lines:
        logger.info(line)

    return {"folds": folds_by_cohort, "summary": rows}


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: build and write every cohort's cross-fit split files.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    run_make_crossfit_splits(cfg)


if __name__ == "__main__":
    main()
