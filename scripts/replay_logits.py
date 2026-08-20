"""Hydra entry point that turns `neurovision.analysis.replay` into committed result files.

`src/neurovision/analysis/replay.py` re-scores a checkpoint's SAVED fp16
logits (`<eval_dir>/logits/<case_id>.npy`, written by `scripts/evaluate.py`
when `cfg.inference.evaluation.save_logits=true`) at a different
discretization threshold or post-processing chain, with no model run and no
GPU -- already verified once by hand to reproduce a published test Dice
exactly (ET 0.870859 vs 0.870859, delta 0). That module is fully implemented
and tested, but nothing calls it. This script is the missing driver.

It answers two questions a reviewer will ask about any reported Dice
advantage:

1. Is it an artifact of the 0.5 threshold, or does it hold across operating
   points? -- `analysis.replay.threshold_sweep`.
2. How much of it comes from the model versus from post-processing
   (small-component removal, largest-component, nesting)? --
   `analysis.replay.postprocess_ablation`.

## Why every run re-verifies itself against the published metrics first

`per_case_default.csv` -- the replay at the project's own default threshold
and post-processing chain -- is checked against the eval directory's own
`per_case_metrics.csv` (written by `scripts/evaluate.py` for that same run)
before anything else in this file is trusted. If the two disagree, the
saved logits and the published numbers came from a different configuration
(a different threshold, post-processing chain, or even checkpoint) and
every other file this script writes -- the sweep, the ablation, the
best-threshold pick -- would be replaying the wrong thing while looking
entirely plausible. This is the same shape of rule CLAUDE.md states for
`union_foreground_mask`: a fix (or, here, a whole downstream analysis) to an
analysis pipeline is only verified by re-running the real analysis and
checking the output against something independently known, never by a unit
test on the machinery alone.

## Why `out_dir` is not a single OmegaConf interpolation

`configs/analysis/default.yaml`'s `replay.out_dir` names only a parent
directory (e.g. `${output_dir}/replay`); `_resolve_out_dir` appends
`eval_dir`'s own basename in Python. No "basename of another config value"
resolver is registered anywhere in this project (checked: nothing calls
`OmegaConf.register_new_resolver`), and adding one is out of scope for a
single driver script. Doing it in Python instead keeps two replays of
different eval directories (e.g. `eval_test` vs
`eval_test_baseline_unet3d`) from overwriting each other's
`threshold_sweep.csv` / `postprocess_ablation.csv` under one shared folder.

Example usage:

    python scripts/replay_logits.py +analysis.replay.eval_dir=outputs/neurovision/eval_test
    python scripts/replay_logits.py +analysis.replay.eval_dir=outputs/neurovision/eval_test \\
        analysis.replay.baseline_comparison.enabled=true \\
        +analysis.replay.baseline_comparison.eval_dir=outputs/eval_test_baseline_unet3d
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.replay import per_case_replay, postprocess_ablation, threshold_sweep
from neurovision.analysis.statistics import compare_models, format_comparison, load_per_case
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/evaluate.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Metrics summarized by both the threshold sweep and the post-processing
# ablation, and the metrics `best_threshold.json` reports on. dice_mean is
# included alongside the three per-region Dice scores because
# `compute_case_metrics` already returns it for free, and a single "did the
# operating point help overall" number is worth having next to the
# per-region breakdown.
_METRICS: tuple[str, ...] = ("dice_ET", "dice_TC", "dice_WT", "dice_mean")

# Metrics the self-consistency check compares against the eval directory's
# own published per_case_metrics.csv. Deliberately just the three per-region
# Dice scores -- HD95 can legitimately be NaN on one side and not the other
# depending on pandas' NaN handling through a join, which would make an
# "agree" check noisy for no reason; Dice alone is already dispositive.
_CONSISTENCY_METRICS: tuple[str, ...] = ("dice_ET", "dice_TC", "dice_WT")

# Mean absolute difference, in Dice units, above which the self-consistency
# check raises rather than merely logging. `replay.py`'s own docstring
# reports an exact replay reproducing a published Dice to zero delta; 1e-6 is
# generous enough to absorb float32/float64 summation order differences
# while catching any real configuration mismatch.
_CONSISTENCY_TOLERANCE = 1e-6


def _resolve_out_dir(cfg: DictConfig) -> Path:
    """Derives this run's output directory from `eval_dir`'s own basename.

    See this module's docstring for why this is computed in Python rather
    than as a single OmegaConf interpolation.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The created output directory, `<analysis.replay.out_dir>/<eval_dir basename>`.

    Raises:
        ValueError: If `analysis.replay.eval_dir` is not set. Matches
            `scripts/burden.py` / `scripts/localize.py`'s convention of
            refusing to guess an experiment's eval directory from a default.
    """
    replay_cfg = cfg.analysis.replay
    if replay_cfg.eval_dir is None:
        raise ValueError(
            "analysis.replay.eval_dir is not set. Point it at an evaluation output directory "
            "written by scripts/evaluate.py with save_logits=true, e.g. "
            "'+analysis.replay.eval_dir=outputs/neurovision/eval_test'."
        )
    leaf = Path(str(replay_cfg.eval_dir)).name
    return ensure_dir(Path(str(replay_cfg.out_dir)) / leaf)


def _check_self_consistency(replayed: pd.DataFrame, eval_dir: Path) -> dict[str, float] | None:
    """Verifies a default-settings replay against the eval directory's own published metrics.

    Args:
        replayed: The `per_case_default` table (project-default threshold
            and post-processing, indexed by `case_id`).
        eval_dir: The evaluation output directory being replayed, expected
            to hold `per_case_metrics.csv` if `scripts/evaluate.py` has
            already been run on it.

    Returns:
        The measured `{metric: mean_absolute_delta}` dict on success, or
        `None` if the check was skipped because no published table exists
        or the two tables share no case ids.

    Raises:
        ValueError: If the published table exists, shares at least one case
            id with the replay, and any metric's mean absolute delta is at
            or above `_CONSISTENCY_TOLERANCE`. Names the per-region deltas.
    """
    published_path = Path(eval_dir) / "per_case_metrics.csv"
    if not published_path.is_file():
        logger.warning(
            "replay_logits: no per_case_metrics.csv at %s; skipping the self-consistency "
            "check. Every other file this run writes is therefore UNVERIFIED against a prior "
            "evaluation run.",
            published_path,
        )
        return None

    published = load_per_case(published_path)
    metrics = list(_CONSISTENCY_METRICS)
    joined = replayed[metrics].join(
        published[metrics], how="inner", lsuffix="_replay", rsuffix="_published"
    )
    if joined.empty:
        logger.warning(
            "replay_logits: the replayed table and %s share no case ids; skipping the "
            "self-consistency check.",
            published_path,
        )
        return None

    deltas = {
        metric: float((joined[f"{metric}_replay"] - joined[f"{metric}_published"]).abs().mean())
        for metric in metrics
    }
    logger.info(
        "replay_logits: self-consistency mean absolute deltas vs %s (n=%d shared case(s)): %s",
        published_path,
        len(joined),
        deltas,
    )

    if max(deltas.values()) >= _CONSISTENCY_TOLERANCE:
        raise ValueError(
            "replay_logits: the replayed per-case Dice disagrees with the published "
            f"{published_path} by more than {_CONSISTENCY_TOLERANCE} (mean absolute deltas: "
            f"{deltas}). This means the eval directory's saved logits and its published "
            "per_case_metrics.csv came from a different configuration (a different threshold, "
            "post-processing chain, or checkpoint) than this replay used -- nothing downstream "
            "of this replay can be trusted until that discrepancy is resolved."
        )
    return deltas


def _compute_best_thresholds(sweep_df: pd.DataFrame, metrics: Sequence[str]) -> dict[str, Any]:
    """Picks, per metric, the swept threshold that maximises its mean value.

    Args:
        sweep_df: The output of `neurovision.analysis.replay.threshold_sweep`,
            with a `<metric>_mean` column for every entry of `metrics`.
        metrics: Metric names (without the `_mean` suffix) to report on.

    Returns:
        `{metric: {"best_threshold", "best_value", "value_at_0.5",
        "delta_over_0.5"}, ...}` for each entry of `metrics`, plus a
        `"caveat"` string warning that a threshold chosen this way is
        optimistically biased.

    Raises:
        ValueError: If `threshold=0.5` is not one of the swept thresholds --
            every reported delta is measured against it, so its absence
            makes the whole file meaningless rather than merely incomplete.
    """
    threshold_values = sweep_df["threshold"].to_numpy(dtype=float)
    half_matches = np.nonzero(np.isclose(threshold_values, 0.5))[0]
    if len(half_matches) == 0:
        raise ValueError(
            "replay_logits: best_threshold.json requires threshold=0.5 to be one of the swept "
            f"thresholds so every other threshold's delta can be measured against it; got "
            f"{list(threshold_values)}. Add 0.5 to analysis.replay.threshold_sweep.thresholds."
        )
    row_at_half = sweep_df.iloc[int(half_matches[0])]

    result: dict[str, Any] = {}
    for metric in metrics:
        col = f"{metric}_mean"
        best_idx = sweep_df[col].idxmax()
        best_row = sweep_df.loc[best_idx]
        value_at_half = float(row_at_half[col])
        best_value = float(best_row[col])
        result[metric] = {
            "best_threshold": float(best_row["threshold"]),
            "best_value": best_value,
            "value_at_0.5": value_at_half,
            "delta_over_0.5": best_value - value_at_half,
        }

    result["caveat"] = (
        "This threshold was selected on the SAME split it is reported on, which is "
        "optimistically biased (selection on the test data itself). It must be re-selected "
        "on the val split and shown to transfer to test before it can support any claim in "
        "the paper -- this project evaluates every checkpoint on both val and test "
        "specifically so that distinction can be made."
    )
    return result


def _log_and_print_summary(
    cfg: DictConfig, results: Mapping[str, pd.DataFrame], out_dir: Path
) -> None:
    """Logs and prints a compact end-of-run summary.

    Args:
        cfg: The full composed Hydra config.
        results: The dict `run_replay` is about to return.
        out_dir: The resolved output directory everything was written to.
    """
    lines = [
        "=" * 70,
        f"Logit replay summary -- eval_dir={cfg.analysis.replay.eval_dir}",
        "=" * 70,
        f"  out_dir: {out_dir}",
    ]
    artifact_names = (
        "threshold_sweep",
        "postprocess_ablation",
        "per_case_default",
        "comparison_default",
    )
    for name in artifact_names:
        if name in results:
            lines.append(f"  {name}.csv: {len(results[name])} row(s)")
        else:
            lines.append(f"  {name}.csv: skipped (disabled in config)")
    # print only, not logger.info as well -- setup_logging's StreamHandler
    # already targets stdout, so doing both would print this block twice.
    # Matches scripts/evaluate.py's / scripts/extract_ambiguity.py's summary.
    print("\n".join(lines))


def run_replay(cfg: DictConfig) -> dict[str, pd.DataFrame]:
    """Replays a checkpoint's saved logits at swept thresholds and post-processing variants.

    Writes into the resolved output directory (see `_resolve_out_dir`):

    - `threshold_sweep.csv` when `analysis.replay.threshold_sweep.enabled`.
    - `postprocess_ablation.csv` when `analysis.replay.postprocess_ablation.enabled`.
    - `per_case_default.csv` -- always, the replay at the project default
      threshold and post-processing chain. Verified against the eval
      directory's own published `per_case_metrics.csv` (see
      `_check_self_consistency`) before anything else in the run is trusted.
    - `best_threshold.json` when the threshold sweep ran -- the per-metric
      argmax threshold, its delta over 0.5, and an explicit selection-bias
      caveat.
    - `comparison_default.csv` when `analysis.replay.baseline_comparison.enabled`
      -- `per_case_default` compared against a second eval directory replayed
      at the same project-default settings.
    - `replay_config.yaml` -- always, the fully resolved config this run used.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A dict of the DataFrame artifacts actually produced -- a subset of
        `{"threshold_sweep", "postprocess_ablation", "per_case_default",
        "comparison_default"}` depending on which stages were enabled.
        `per_case_default` is always present.

    Raises:
        ValueError: See `_resolve_out_dir`, `_check_self_consistency`, and
            `_compute_best_thresholds`.
    """
    replay_cfg = cfg.analysis.replay
    out_dir = _resolve_out_dir(cfg)
    eval_dir = Path(str(replay_cfg.eval_dir))
    prep_dir = Path(str(replay_cfg.prep_dir))
    case_ids = list(replay_cfg.case_ids) if replay_cfg.case_ids is not None else None

    logger.info(
        "replay_logits: eval_dir=%s prep_dir=%s out_dir=%s case_ids=%s",
        eval_dir,
        prep_dir,
        out_dir,
        "every case with saved logits" if case_ids is None else f"{len(case_ids)} explicit case(s)",
    )

    results: dict[str, pd.DataFrame] = {}
    sweep_df: pd.DataFrame | None = None

    if replay_cfg.threshold_sweep.enabled:
        thresholds = [float(t) for t in replay_cfg.threshold_sweep.thresholds]
        logger.info("replay_logits: running threshold sweep over %d threshold(s).", len(thresholds))
        sweep_df = threshold_sweep(
            eval_dir, prep_dir, thresholds, case_ids=case_ids, metrics=_METRICS
        )
        sweep_path = out_dir / "threshold_sweep.csv"
        sweep_df.to_csv(sweep_path, index=False)
        logger.info("replay_logits: wrote %s", sweep_path)
        results["threshold_sweep"] = sweep_df

    if replay_cfg.postprocess_ablation.enabled:
        variants = OmegaConf.to_container(replay_cfg.postprocess_ablation.variants, resolve=True)
        logger.info(
            "replay_logits: running post-processing ablation over %d variant(s): %s.",
            len(variants),
            list(variants.keys()),
        )
        ablation_df = postprocess_ablation(
            eval_dir, prep_dir, variants, case_ids=case_ids, metrics=_METRICS
        )
        ablation_path = out_dir / "postprocess_ablation.csv"
        ablation_df.to_csv(ablation_path, index=False)
        logger.info("replay_logits: wrote %s", ablation_path)
        results["postprocess_ablation"] = ablation_df

    logger.info("replay_logits: replaying every selected case at project default settings.")
    per_case_default = per_case_replay(eval_dir, prep_dir, case_ids=case_ids)
    per_case_default_path = out_dir / "per_case_default.csv"
    per_case_default.to_csv(per_case_default_path)
    logger.info("replay_logits: wrote %s", per_case_default_path)
    results["per_case_default"] = per_case_default

    _check_self_consistency(per_case_default, eval_dir)

    if sweep_df is not None:
        best = _compute_best_thresholds(sweep_df, _METRICS)
        best_path = out_dir / "best_threshold.json"
        write_json(best, best_path)
        logger.info("replay_logits: wrote %s", best_path)
    else:
        logger.info(
            "replay_logits: threshold_sweep is disabled, so best_threshold.json is not written."
        )

    if replay_cfg.baseline_comparison.enabled:
        baseline_eval_dir = Path(str(replay_cfg.baseline_comparison.eval_dir))
        logger.info(
            "replay_logits: replaying baseline_comparison.eval_dir=%s at project default "
            "settings for comparison.",
            baseline_eval_dir,
        )
        baseline_per_case = per_case_replay(baseline_eval_dir, prep_dir, case_ids=case_ids)
        generator = np.random.default_rng(int(cfg.seed))
        comparison = compare_models(
            per_case_default,
            baseline_per_case,
            generator=generator,
            metrics=list(_METRICS),
            name_a=eval_dir.name,
            name_b=baseline_eval_dir.name,
        )
        comparison_path = out_dir / "comparison_default.csv"
        comparison.to_csv(comparison_path)
        logger.info("replay_logits: wrote %s", comparison_path)
        comparison_report = format_comparison(
            comparison, name_a=eval_dir.name, name_b=baseline_eval_dir.name
        )
        logger.info("\n%s", comparison_report)
        results["comparison_default"] = comparison

    config_path = out_dir / "replay_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    logger.info("replay_logits: wrote %s", config_path)

    _log_and_print_summary(cfg, results, out_dir)

    return results


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Replays a checkpoint's saved logits per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_replay(cfg)


if __name__ == "__main__":
    main()
