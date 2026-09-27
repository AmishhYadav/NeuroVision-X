"""Hydra entry point for P1.4, the input-statistics OOD score.

`docs/research_docs/preregistrations/preregistration_ood.md` fixes everything this
script does BEFORE any feature is extracted or any score exists -- read it first.
`neurovision.analysis.ood` is the pure-statistics half (no file I/O, no Hydra, no
torch; `extract_features`, `fit_ood_model`, `ood_scores`, `quantile_cuts`, `flag`);
this script is exactly the other half: it walks the preprocessed BraTS trees, calls
those functions, and writes
the pre-registration's output files, IN THE ORDER THE PRE-REGISTRATION FIXES. That
order is not a style choice -- it is the thing that makes "fit on train only,
threshold on val only" checkable from the file layout alone:

    1. Extract features for every cohort (train, val, and every `score_cohorts`
       entry) and write `features_<name>.csv` + `skipped.csv`.
    2. Fit the model on the `train` features ONLY -> `model.json`.
    3. Score `val`, take quantile cuts from `val` ONLY -> `thresholds.json`.
    4. Only once `model.json` and `thresholds.json` are already on disk: score
       every cohort (train, val, test, ssa, ped) -> `<name>_ood_score.csv`.
    5. `flag_rates.csv` -- CAUTION-or-worse / REFUSE rate per cohort, bootstrap CIs.
    6. `auroc.csv` -- descriptive, case-level AUROC for "unusable" (note 47's bar),
       only for cohorts with ground truth (`eval_dir` set).
    7. `run_meta.json` -- git SHA, config, per-cohort counts, a fixed status label.

Steps 5-6 are read by `scripts/error_budget.py` (`_load_ood_table` /
`_load_ood_thresholds`) when `cfg.analysis.error_budget.ood_dir` is pointed at this
script's `out_dir` -- see that script's own docstring. This script never runs
`scripts/error_budget.py` itself and never touches its config.

## CPU only, one volume in memory at a time

Every `image.npy` is loaded, passed to `extract_features`, and deleted before the
next one is loaded -- never a whole cohort of volumes held at once (CLAUDE.md's "one
heavy local job at a time" / 16 GB Mac budget). No multiprocessing: a single process
reading ~1,410 volumes serially, at roughly a quarter second each, is the pre-
registration's own stated cost. No `torch` import anywhere in this file.

## No hardcoded paths

Every input and output path comes from `cfg.analysis.ood` / `cfg.data`, exactly as
every other `scripts/*.py` driver in this project already requires.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import detection, error_budget, ood
from neurovision.data.dataset import load_splits
from neurovision.inference.gatekeeper import Decision
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Fixed cohort processing order for flag_rates.csv (step 5) -- the pre-registration
# names this exact order, and it also fixes what "one shared generator, consumed
# deterministically" means: the bootstrap draws happen in this order, every run.
_FLAG_RATE_COHORT_ORDER: tuple[str, ...] = ("train", "val", "test", "ssa", "ped")

# Recorded verbatim into run_meta.json.
_STATUS_LABEL = (
    "PRE-REGISTERED P1.4 run: fitted on train, cuts on val, frozen before "
    "test/SSA/PED were scored"
)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    """Returns the current commit SHA, or `None` if it cannot be determined.

    Never raises -- `git` missing, this not being a repository, or any other
    failure all fall back to `None` rather than aborting a CPU-only analysis run
    over a provenance nicety. Mirrors `scripts/local_recalibration.py::_git_sha`.
    """
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root
        )
    except (OSError, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


# ---------------------------------------------------------------------------
# Channel order -- must match neurovision.analysis.ood.MODALITIES exactly
# ---------------------------------------------------------------------------


def resolve_modalities(cfg: DictConfig) -> tuple[str, ...]:
    """The channel order every feature name assumes: `cfg.data.modalities`, checked.

    `ood.extract_features` names every column from the modality order it is
    called with (e.g. `"t1_p01"`), and `ood.FEATURE_NAMES` freezes that order at
    import time for `MODALITIES`. This project's preprocessing pipeline writes
    `image.npy` in exactly that order, so `cfg.data.modalities` (when present)
    must equal `ood.MODALITIES` exactly -- a config that silently reordered
    channels would relabel every feature without raising anywhere else.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        `ood.MODALITIES`.

    Raises:
        ValueError: `cfg.data.modalities` exists and differs from `ood.MODALITIES`.
    """
    configured = cfg.data.get("modalities", None)
    modalities = tuple(str(m) for m in configured) if configured is not None else ood.MODALITIES
    if modalities != ood.MODALITIES:
        raise ValueError(
            f"ood_score: cfg.data.modalities={modalities!r} does not match "
            f"neurovision.analysis.ood.MODALITIES={ood.MODALITIES!r} -- every OOD "
            "feature name is fixed to that order; a mismatch would silently relabel "
            "every feature."
        )
    return modalities


# ---------------------------------------------------------------------------
# Step 1: case listing and feature extraction
# ---------------------------------------------------------------------------


def list_case_ids(prep_dir: Path, split_case_ids: Sequence[str] | None) -> list[str]:
    """Case ids to score for one cohort.

    Args:
        prep_dir: Root of a preprocessed tree. Its top level also holds
            non-case files (`case_index.csv`, `metadata.csv`) and, for every
            real case, a `<case_id>/` directory containing `image.npy` -- both
            are skipped when `split_case_ids` is `None`.
        split_case_ids: If not `None`, this exact list of ids (sorted) is
            used and `prep_dir` is not listed at all. If `None`, every
            subdirectory of `prep_dir` that contains an `image.npy` file is
            used, sorted.

    Returns:
        A sorted list of case ids.
    """
    if split_case_ids is not None:
        return sorted(str(c) for c in split_case_ids)
    ids: list[str] = []
    for entry in sorted(prep_dir.iterdir()):
        if entry.is_dir() and (entry / "image.npy").is_file():
            ids.append(entry.name)
    return ids


def extract_cohort_features(
    prep_dir: Path,
    case_ids: Sequence[str],
    modalities: Sequence[str],
    cohort_name: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Extracts `ood.extract_features` for every case, one volume in memory at a time.

    Args:
        prep_dir: Root of the preprocessed tree holding `<case_id>/image.npy`.
        case_ids: Case ids to load, in the order processed (also the row order
            of the returned table).
        modalities: Channel order passed straight through to `extract_features`.
        cohort_name: Used only in log messages and `skipped.csv` rows.

    Returns:
        `(features_df, skipped_rows)`. `features_df` has a `case_id` column
        plus every `ood.FEATURE_NAMES` column, one row per successfully scored
        case. `skipped_rows` is a list of `{"cohort", "case_id", "reason"}`
        dicts, one per case whose `extract_features` call raised `ValueError`
        (too little brain to score) -- never dropped without a record.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    n = len(case_ids)
    for i, case_id in enumerate(case_ids):
        image = np.load(prep_dir / case_id / "image.npy", mmap_mode=None)
        try:
            features = ood.extract_features(image, modalities)
        except ValueError as exc:
            logger.warning("ood_score: cohort %s case %s skipped -- %s", cohort_name, case_id, exc)
            skipped.append({"cohort": cohort_name, "case_id": case_id, "reason": str(exc)})
            del image
            continue
        del image  # one volume in memory at a time -- see module docstring.

        row: dict[str, Any] = {"case_id": case_id}
        row.update(features)
        rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info("ood_score: cohort %s -- %d/%d volume(s) processed.", cohort_name, i + 1, n)

    columns = ["case_id", *ood.FEATURE_NAMES]
    features_df = pd.DataFrame(rows, columns=columns)
    logger.info(
        "ood_score: cohort %s -- extracted %d/%d, skipped %d.",
        cohort_name,
        len(features_df),
        n,
        len(skipped),
    )
    return features_df, skipped


# ---------------------------------------------------------------------------
# Step 4: scoring
# ---------------------------------------------------------------------------


def score_table(
    model: ood.OODModel, features_df: pd.DataFrame, caution_cut: float, refuse_cut: float
) -> pd.DataFrame:
    """One cohort's `case_id, ood_score, ood_flag` table.

    Args:
        model: The fitted `OODModel` (train-only).
        features_df: A table with a `case_id` column plus `model.feature_names`.
        caution_cut: The CAUTION cut point (from `quantile_cuts` on `val`).
        refuse_cut: The REFUSE cut point (from `quantile_cuts` on `val`).

    Returns:
        Columns exactly `case_id`, `ood_score`, `ood_flag` -- the format
        `scripts/error_budget.py::_load_ood_table` reads.
    """
    scores = ood.ood_scores(model, features_df)
    flags = ood.flag(scores, caution_cut, refuse_cut)
    return pd.DataFrame(
        {"case_id": features_df["case_id"].to_numpy(), "ood_score": scores, "ood_flag": flags}
    )


def write_scores(table: pd.DataFrame, out_path: Path) -> None:
    """Writes one cohort's score table to `out_path`, logging how many cases it covers."""
    table.to_csv(out_path, index=False)
    logger.info("ood_score: wrote %s (%d case(s)).", out_path, len(table))


# ---------------------------------------------------------------------------
# Step 5: flag_rates.csv
# ---------------------------------------------------------------------------


def flag_rate_table(
    flags_by_cohort: Mapping[str, np.ndarray],
    cohort_order: Sequence[str],
    n_boot: int,
    ci: float,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """CAUTION-or-worse and REFUSE rates per cohort, with bootstrap CIs.

    Args:
        flags_by_cohort: `{cohort_name: (n,) array of ood.flag's string output}`.
        cohort_order: The fixed order to process cohorts in -- both the row
            order of the returned table and the order the shared `generator`
            is drawn from, so the whole run is deterministic given `generator`'s
            seed.
        n_boot: Number of bootstrap resamples per rate.
        ci: Central confidence level, e.g. `0.95`.
        generator: ONE seeded `np.random.Generator`, shared across every
            cohort and rate computed here (and, by the caller, with `auroc_table`
            afterwards) -- never re-seeded per cohort.

    Returns:
        One row per cohort: `cohort`, `n`, `caution_or_worse` (+`_lo`/`_hi`),
        `refuse` (+`_lo`/`_hi`).
    """
    rows: list[dict[str, Any]] = []
    for cohort in cohort_order:
        flags = np.asarray(flags_by_cohort[cohort], dtype=object)
        # HIGH is bad (ood.flag's own convention): "caution or worse" is anything
        # that did not PROCEED; "refuse" is the more extreme subset of that.
        caution_or_worse = (flags != Decision.PROCEED.value).astype(float)
        refuse = (flags == Decision.REFUSE.value).astype(float)
        c_point, c_lo, c_hi = error_budget.bootstrap_rate_ci(
            caution_or_worse, n_boot=n_boot, ci=ci, generator=generator
        )
        r_point, r_lo, r_hi = error_budget.bootstrap_rate_ci(
            refuse, n_boot=n_boot, ci=ci, generator=generator
        )
        rows.append(
            {
                "cohort": cohort,
                "n": len(flags),
                "caution_or_worse": c_point,
                "caution_or_worse_lo": c_lo,
                "caution_or_worse_hi": c_hi,
                "refuse": r_point,
                "refuse_lo": r_lo,
                "refuse_hi": r_hi,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6: auroc.csv -- DESCRIPTIVE ONLY (see the pre-registration's Endpoint 2)
# ---------------------------------------------------------------------------


def auroc_table(
    score_tables: Mapping[str, pd.DataFrame],
    eval_dirs: Mapping[str, Path],
    usable_regions: Sequence[str],
    usable_bar: float,
    n_boot: int,
    ci: float,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Descriptive, case-level AUROC of `ood_score` for "unusable" (WT or TC Dice < bar).

    This is DESCRIPTIVE, per the pre-registration's Endpoint 2 -- an input-shift
    score is not expected to rank failed masks (prediction (d): AUROC <= 0.65 in
    SSA/PED), and this table is reported as evidence either way, never tuned
    against.

    Args:
        score_tables: `{cohort_name: DataFrame with case_id, ood_score, ...}`,
            for every cohort that has ground truth (i.e. every `score_cohorts`
            entry with `eval_dir` set).
        eval_dirs: `{cohort_name: eval_dir}` for the SAME cohorts as
            `score_tables` -- each must hold a `per_case_metrics.csv`.
        usable_regions: Regions `error_budget.label_usable` checks, e.g. `[WT, TC]`.
        usable_bar: The Dice bar, e.g. `0.7`.
        n_boot: Number of case-resampled bootstrap draws attempted per cohort.
        ci: Central confidence level, e.g. `0.95`.
        generator: The SAME shared `np.random.Generator` `flag_rate_table` drew
            from -- cohorts are processed in `eval_dirs`' own iteration order.

    Returns:
        One row per cohort: `cohort`, `n`, `n_unusable`, `auroc`, `ci_lo`, `ci_hi`,
        `n_boot_valid` (resamples that had both classes present; the ones that did
        not are skipped, never counted or padded). `auroc`/`ci_lo`/`ci_hi` are NaN
        when `n_unusable` is `0` or `n` -- there is no "unusable" class to detect.

    Raises:
        ValueError: A cohort's score table and `per_case_metrics.csv` share no
            `case_id` at all.
    """
    rows: list[dict[str, Any]] = []
    for cohort_name, eval_dir in eval_dirs.items():
        metrics = pd.read_csv(eval_dir / "per_case_metrics.csv")
        usable = error_budget.label_usable(metrics, regions=usable_regions, bar=usable_bar)
        labels = pd.DataFrame(
            {"case_id": metrics["case_id"].to_numpy(), "unusable": (~usable).to_numpy()}
        )
        scores = score_tables[cohort_name][["case_id", "ood_score"]]

        merged = scores.merge(labels, on="case_id", how="inner")
        n_dropped_scores = len(scores) - len(merged)
        n_dropped_labels = len(labels) - len(merged)
        logger.info(
            "ood_score: cohort %s AUROC join -- dropped %d score row(s), %d metrics "
            "row(s); %d case(s) survive.",
            cohort_name,
            n_dropped_scores,
            n_dropped_labels,
            len(merged),
        )
        if merged.empty:
            raise ValueError(
                f"ood_score: cohort {cohort_name!r}'s ood-score table and "
                "per_case_metrics.csv share no case_id at all; nothing to score for AUROC."
            )

        n = len(merged)
        unusable = merged["unusable"].to_numpy()
        scores_arr = merged["ood_score"].to_numpy()
        n_unusable = int(unusable.sum())

        if n_unusable == 0 or n_unusable == n:
            rows.append(
                {
                    "cohort": cohort_name,
                    "n": n,
                    "n_unusable": n_unusable,
                    "auroc": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n_boot_valid": 0,
                }
            )
            continue

        point = detection.auroc(scores_arr, unusable)

        boot_values: list[float] = []
        for _ in range(n_boot):
            resample = generator.integers(0, n, size=n)
            resampled_unusable = unusable[resample]
            # A resample can, by chance, land all-usable or all-unusable -- AUROC
            # is undefined there, so that draw is skipped rather than counted as a
            # spurious 0.5 or padded with a fabricated value.
            if resampled_unusable.all() or not resampled_unusable.any():
                continue
            boot_values.append(detection.auroc(scores_arr[resample], resampled_unusable))

        n_boot_valid = len(boot_values)
        if n_boot_valid >= 2:
            lo = float(np.percentile(boot_values, 100.0 * (1.0 - ci) / 2.0))
            hi = float(np.percentile(boot_values, 100.0 * (1.0 + ci) / 2.0))
        else:
            lo = hi = float("nan")

        rows.append(
            {
                "cohort": cohort_name,
                "n": n,
                "n_unusable": n_unusable,
                "auroc": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_boot_valid": n_boot_valid,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _log_summary(flag_rates: pd.DataFrame, auroc: pd.DataFrame) -> None:
    """Logs a short summary table of flag rates and AUROC, at the end of the run."""
    lines = ["=" * 78, "ood_score: P1.4 summary", "=" * 78]
    for _, row in flag_rates.iterrows():
        lines.append(
            f"  {row['cohort']:>6s} | n={int(row['n']):4d} | "
            f"caution_or_worse={row['caution_or_worse']:.4f} "
            f"[{row['caution_or_worse_lo']:.4f}, {row['caution_or_worse_hi']:.4f}] | "
            f"refuse={row['refuse']:.4f} [{row['refuse_lo']:.4f}, {row['refuse_hi']:.4f}]"
        )
    lines.append("-" * 78)
    for _, row in auroc.iterrows():
        lines.append(
            f"  {row['cohort']:>6s} | n={int(row['n']):4d} | "
            f"n_unusable={int(row['n_unusable']):4d} | "
            f"auroc={row['auroc']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] "
            f"(descriptive only)"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> dict[str, Path]:
    """Runs the full P1.4 OOD score pipeline, in the pre-registration's fixed order.

    Args:
        cfg: The full composed Hydra config. Reads `cfg.analysis.ood` (this
            driver's own config block, see `configs/analysis/default.yaml`) and
            `cfg.data.modalities`/`cfg.data.splits.path`.

    Returns:
        A dict mapping a short name to the `Path` each output file was written
        to: `"model_json"`, `"thresholds_json"`, `"skipped_csv"`,
        `"flag_rates_csv"`, `"auroc_csv"`, `"run_meta_json"`, plus one
        `"features_<name>"` and one `"scores_<name>"` entry per cohort.

    Raises:
        ValueError: `cfg.data.modalities` disagrees with `ood.MODALITIES`; the
            `train` split has fewer than 10 usable cases (`fit_ood_model`); or
            an AUROC cohort's score table shares no case with its
            `per_case_metrics.csv`.
    """
    ood_cfg = cfg.analysis.ood
    modalities = resolve_modalities(cfg)

    out_dir = ensure_dir(str(ood_cfg.out_dir))
    output_paths: dict[str, Path] = {}

    fit_split = str(ood_cfg.fit_split)
    threshold_split = str(ood_cfg.threshold_split)
    brats_prep_dir = Path(str(ood_cfg.brats_prep_dir))

    splits_path = Path(str(cfg.data.splits.path))
    splits = load_splits(splits_path)

    # --- Step 1: feature extraction, every cohort, before anything is fitted ---
    # (name, prep_dir, case_ids) for train, val, and every score_cohorts entry, in
    # that fixed order -- fitting/thresholding below only ever reads the train/val
    # entries, so a score_cohorts entry's own statistics can never leak into them.
    cohort_specs: list[tuple[str, Path, list[str]]] = [
        (fit_split, brats_prep_dir, list_case_ids(brats_prep_dir, splits[fit_split])),
        (threshold_split, brats_prep_dir, list_case_ids(brats_prep_dir, splits[threshold_split])),
    ]
    score_cohort_eval_dirs: dict[str, Path] = {}
    for entry in ood_cfg.score_cohorts:
        name = str(entry.name)
        prep_dir = Path(str(entry.prep_dir))
        split_name = entry.get("split", None)
        split_case_ids = splits[str(split_name)] if split_name is not None else None
        cohort_specs.append((name, prep_dir, list_case_ids(prep_dir, split_case_ids)))
        eval_dir = entry.get("eval_dir", None)
        if eval_dir is not None:
            score_cohort_eval_dirs[name] = Path(str(eval_dir))

    features_by_cohort: dict[str, pd.DataFrame] = {}
    all_skipped: list[dict[str, str]] = []
    n_by_cohort: dict[str, int] = {}
    for name, prep_dir, case_ids in cohort_specs:
        features_df, skipped = extract_cohort_features(prep_dir, case_ids, modalities, name)
        features_by_cohort[name] = features_df
        all_skipped.extend(skipped)
        n_by_cohort[name] = len(features_df)

        features_path = out_dir / f"features_{name}.csv"
        features_df.to_csv(features_path, index=False)
        output_paths[f"features_{name}"] = features_path
        logger.info("ood_score: wrote %s.", features_path)

    skipped_df = pd.DataFrame(all_skipped, columns=["cohort", "case_id", "reason"])
    skipped_path = out_dir / "skipped.csv"
    skipped_df.to_csv(skipped_path, index=False)
    output_paths["skipped_csv"] = skipped_path
    logger.info("ood_score: wrote %s (%d row(s)).", skipped_path, len(skipped_df))

    # --- Step 2: fit on train ONLY -------------------------------------------
    train_features = features_by_cohort[fit_split]
    model = ood.fit_ood_model(train_features, shrinkage=float(ood_cfg.shrinkage))
    model_path = out_dir / "model.json"
    write_json(model.to_dict(), model_path)
    output_paths["model_json"] = model_path
    logger.info("ood_score: wrote %s.", model_path)

    # --- Step 3: score val, cuts from val ONLY -------------------------------
    val_features = features_by_cohort[threshold_split]
    val_scores = ood.ood_scores(model, val_features)
    caution_quantile = float(ood_cfg.caution_quantile)
    refuse_quantile = float(ood_cfg.refuse_quantile)
    caution_cut, refuse_cut = ood.quantile_cuts(val_scores, caution_quantile, refuse_quantile)
    thresholds_payload = {
        "caution_cut": caution_cut,
        "refuse_cut": refuse_cut,
        "caution_quantile": caution_quantile,
        "refuse_quantile": refuse_quantile,
        "threshold_split": threshold_split,
        "n_val": len(val_features),
    }
    thresholds_path = out_dir / "thresholds.json"
    write_json(thresholds_payload, thresholds_path)
    output_paths["thresholds_json"] = thresholds_path
    logger.info("ood_score: wrote %s -- %s.", thresholds_path, thresholds_payload)

    # --- Step 4: score every cohort, only now that 2-3 are on disk ----------
    score_tables: dict[str, pd.DataFrame] = {}
    for name, _prep_dir, _case_ids in cohort_specs:
        table = score_table(model, features_by_cohort[name], caution_cut, refuse_cut)
        score_tables[name] = table
        scores_path = out_dir / f"{name}_ood_score.csv"
        write_scores(table, scores_path)
        output_paths[f"scores_{name}"] = scores_path

    # --- Step 5: flag_rates.csv, one shared generator, fixed cohort order ----
    n_boot = int(ood_cfg.n_boot)
    ci = float(ood_cfg.ci)
    generator = np.random.default_rng(int(ood_cfg.seed))

    flags_by_cohort = {name: score_tables[name]["ood_flag"].to_numpy() for name in score_tables}
    missing_flag_cohorts = [c for c in _FLAG_RATE_COHORT_ORDER if c not in flags_by_cohort]
    if missing_flag_cohorts:
        raise ValueError(
            f"ood_score: flag_rates.csv expects cohorts {_FLAG_RATE_COHORT_ORDER}, missing "
            f"{missing_flag_cohorts}; check analysis.ood.score_cohorts names."
        )
    flag_rates = flag_rate_table(flags_by_cohort, _FLAG_RATE_COHORT_ORDER, n_boot, ci, generator)
    flag_rates_path = out_dir / "flag_rates.csv"
    flag_rates.to_csv(flag_rates_path, index=False)
    output_paths["flag_rates_csv"] = flag_rates_path
    logger.info("ood_score: wrote %s.", flag_rates_path)

    # --- Step 6: auroc.csv, descriptive, same shared generator ---------------
    usable_regions = [str(r) for r in ood_cfg.usable_regions]
    usable_bar = float(ood_cfg.usable_bar)
    auroc = auroc_table(
        score_tables, score_cohort_eval_dirs, usable_regions, usable_bar, n_boot, ci, generator
    )
    auroc_path = out_dir / "auroc.csv"
    auroc.to_csv(auroc_path, index=False)
    output_paths["auroc_csv"] = auroc_path
    logger.info("ood_score: wrote %s.", auroc_path)

    # --- Step 7: run_meta.json ------------------------------------------------
    n_skipped_by_cohort: dict[str, int] = {}
    for row in all_skipped:
        n_skipped_by_cohort[row["cohort"]] = n_skipped_by_cohort.get(row["cohort"], 0) + 1

    run_meta = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": OmegaConf.to_container(ood_cfg, resolve=True),
        "n_by_cohort": n_by_cohort,
        "n_skipped_total": len(all_skipped),
        "n_skipped_by_cohort": n_skipped_by_cohort,
        "status_label": _STATUS_LABEL,
    }
    run_meta_path = out_dir / "run_meta.json"
    write_json(run_meta, run_meta_path)
    output_paths["run_meta_json"] = run_meta_path
    logger.info("ood_score: wrote %s.", run_meta_path)

    _log_summary(flag_rates, auroc)

    return output_paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the P1.4 OOD score pipeline, per the composed config.

    Example:

        python scripts/ood_score.py

    (every knob it needs lives at `cfg.analysis.ood` -- see
    `configs/analysis/default.yaml`.)

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
