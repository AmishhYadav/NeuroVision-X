"""Hydra entry point for Gate C: does the trained SegQC model beat free entropy?

This is the driver half of Phase C4 (per-cohort validation) and C5 (the
silent-failure test) -- the statistics are already built and tested in
`neurovision.analysis.qc_validate`; this script only wires them to the
project's saved artifacts. Read `docs/research/preregistration_qc.md`
FIRST -- it fixes the endpoints, the family and the decision rule, and
nothing here may reinterpret them.

## What this script does, end to end

1. Loads the trained `SegQC` checkpoint (`analysis.qc_validate.checkpoint`).
2. For every configured cohort, for every case with both saved logits and a
   preprocessed label: reconstructs the TRUE Dice of the model's own,
   undegraded prediction (via `neurovision.data.qc_pairs.generate_pairs` with
   an identity degradation -- this is a second, independent path to the same
   number `per_case_metrics.csv` already reports); runs the QC model once per
   region to get a predicted Dice; and recomputes the free entropy baseline
   from the case's saved logits. Writes one `per_case_<cohort>.csv` per
   cohort.
3. Runs the falsification check (`qc_validate.falsification_check`) for
   EVERY cohort before computing a single endpoint -- see that function's
   docstring and the pre-registration's "falsification check" section for
   why: a disagreement here means one of the two paths to "true Dice" is
   scoring a mask nobody ever evaluated, and no result may be reported until
   that is resolved.
4. Builds `CellEndpoints` for every cohort x region cell, marks the
   pre-registered gate family, applies the fixed decision rule
   (`gate_c_verdict`), and writes `cells.csv`, `gate_c_verdict.json`,
   `silent_failure.csv`.

## Why the geometry cannot drift from training

Regions, `target_shape` and `modality_index` are read from
`cfg.analysis.qc` -- the SAME block `scripts/train_qc.py` reads -- rather
than being re-specified under `analysis.qc_validate`. A validation run that
silently used a different resize target or a different modality than the
run that trained the checkpoint would score the model on an input
distribution it never saw, and nothing about that mismatch would look wrong
in isolation.

## Why the raw fp32 logits are loaded once per case, directly

`case_entropy_scalars` (the free baseline) needs the case's raw saved
logits, cast to float32, exactly once -- it computes all three regions'
entropy scalars from a single `(3, D, H, W)` array in one call. This script
loads that array itself with one `np.load`, rather than asking
`neurovision.analysis.qc_inference.load_case_arrays` (called separately, for
the deployed prediction / label / image / frozen-entropy-in-nats bundle) to
expose its own internal logits load -- the two loads serve genuinely
different purposes (nats for the QC model's input channel vs. the
normalised `[0, 1]` scale `case_entropy_scalars` reports) and are documented
here so a future reader does not "simplify" this into one shared load and
silently conflate the two conventions.

## Why the falsification loop runs to completion before any endpoint

Per the pre-registration: a mismatch discovered on the THIRD cohort must not
be preceded by two cohorts' worth of already-written endpoint files. This
script therefore processes every cohort's per-case table FIRST (writing
`per_case_<cohort>.csv` as it goes -- these are raw reconstructions, not
endpoints, so writing them early is fine), THEN runs
`falsification_check` for every cohort in one loop, and only builds a single
`CellEndpoints` after that whole loop has completed without raising. If any
cohort fails the check, the exception propagates before `cells.csv`,
`gate_c_verdict.json` or `silent_failure.csv` is touched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from neurovision.analysis.detection import case_entropy_scalars
from neurovision.analysis.qc_inference import load_case_arrays, pack_sample
from neurovision.analysis.qc_validate import (
    cell_endpoints,
    endpoints_table,
    falsification_check,
    gate_c_verdict,
    mark_family,
    silent_failure_table,
)
from neurovision.analysis.statistics import load_per_case
from neurovision.data.qc_pairs import DegradationSpec, generate_pairs
from neurovision.data.transforms import REGION_NAMES
from neurovision.models.qc import build_segqc, predicted_dice
from neurovision.training.checkpoint import load_checkpoint
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# How often (in cases) per-cohort progress is logged. Matches
# scripts/conformal.py's own _LOG_EVERY -- a cohort can hold up to 189 cases,
# so logging every case would flood the console.
_LOG_EVERY = 25

# The entropy-cache cross-check (step 5 of the module docstring) is a
# recomputation of the SAME function against the SAME saved logits, so any
# real disagreement should be many orders of magnitude above this -- it
# exists to catch a genuine drift (a different threshold, a different
# logits file), not floating-point noise.
_ENTROPY_CACHE_TOL = 1e-6


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def resolve_checkpoint(cfg: DictConfig) -> Path:
    """Validates `analysis.qc_validate.checkpoint` exists.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The checkpoint path.

    Raises:
        FileNotFoundError: No file exists at the configured path.
    """
    path = Path(str(cfg.analysis.qc_validate.checkpoint))
    if not path.is_file():
        raise FileNotFoundError(
            f"validate_qc: no checkpoint at {path.resolve()}. Run scripts/train_qc.py first to "
            "train a SegQC checkpoint (see docs/research/preregistration_qc.md) before scoring "
            "Gate C."
        )
    return path


def load_qc_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Builds `SegQC` and loads the configured checkpoint's weights, in eval mode.

    Logs which epoch and which recorded best-metric value the checkpoint
    carries, so a run's numbers can always be traced back to the checkpoint
    that produced them.

    Args:
        cfg: The full composed Hydra config.
        device: Where the model should live.

    Returns:
        The loaded model, on `device`, in `.eval()` mode.

    Raises:
        FileNotFoundError: See `resolve_checkpoint`.
    """
    checkpoint_path = resolve_checkpoint(cfg)
    model = build_segqc(cfg).to(device)
    # No optimizer to restore (this script never trains) and RNG restore is
    # deliberately skipped -- this run's own randomness (the bootstrap) is
    # seeded independently from cfg.seed, never from whatever RNG state a
    # training run happened to save.
    resume_state = load_checkpoint(
        checkpoint_path, model, optimizer=None, map_location=str(device), restore_rng=False
    )
    # ResumeState.start_epoch is "saved epoch + 1" (see checkpoint.py) -- the
    # epoch that actually PRODUCED this checkpoint is one less.
    epoch = resume_state.start_epoch - 1
    logger.info(
        "validate_qc: loaded checkpoint %s -- epoch=%d, best_metric_name=%s, best_metric=%.6g.",
        checkpoint_path,
        epoch,
        resume_state.best_metric_name,
        resume_state.best_metric,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Cohort resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortSpec:
    """One cohort's resolved paths.

    Attributes:
        name: Cohort name, e.g. `"test"`, `"ssa"`, `"ped"`.
        eval_dir: A `scripts/evaluate.py` output directory holding
            `logits/*.npy` and `per_case_metrics.csv`.
        prep_dir: Root of the preprocessed BraTS data for this cohort.
        entropy_cache: Optional path to a previously-written
            `entropy_cache_*.csv` to cross-check the recomputed entropy
            against, or `None` to skip the check.
    """

    name: str
    eval_dir: Path
    prep_dir: Path
    entropy_cache: Path | None


def resolve_cohorts(cfg: DictConfig) -> list[CohortSpec]:
    """Resolves and validates every configured cohort, before any case is processed.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        One `CohortSpec` per entry of `analysis.qc_validate.cohorts`, in
        the configured order.

    Raises:
        FileNotFoundError: A cohort's `eval_dir` has no `logits/`
            subdirectory. Raised before any per-case work starts, for any
            cohort -- a misconfigured cohort listed later must not be
            discovered only after the earlier cohorts were already scored.
    """
    cohorts: list[CohortSpec] = []
    for entry in cfg.analysis.qc_validate.cohorts:
        name = str(entry.name)
        eval_dir = Path(str(entry.eval_dir))
        prep_dir = Path(str(entry.prep_dir))
        entropy_cache_raw = entry.entropy_cache
        entropy_cache = Path(str(entropy_cache_raw)) if entropy_cache_raw is not None else None

        logits_dir = eval_dir / "logits"
        if not logits_dir.is_dir():
            raise FileNotFoundError(
                f"validate_qc: cohort {name!r}'s eval_dir has no logits/ directory: "
                f"{logits_dir}. Re-run scripts/evaluate.py with "
                "inference.evaluation.save_logits=true against this eval_dir."
            )
        cohorts.append(
            CohortSpec(name=name, eval_dir=eval_dir, prep_dir=prep_dir, entropy_cache=entropy_cache)
        )
    return cohorts


def _shared_case_ids(
    eval_dir: Path, prep_dir: Path, max_cases: int | None, cohort_name: str
) -> list[str]:
    """Case ids with both saved logits and a preprocessed case directory.

    Logs the number of ids present in only one of the two sources --
    `logits/` without a matching `prep_dir` entry, or vice versa -- so a
    cohort that silently shrank is never invisible (see the "excluded case
    ids" edge case in the spec this driver implements).

    Args:
        eval_dir: A `scripts/evaluate.py` output directory.
        prep_dir: Root of the preprocessed BraTS data.
        max_cases: If not `None`, truncates the sorted id list to this many
            entries -- a deterministic subsetting knob, not a random
            subsample.
        cohort_name: Used only in log messages.

    Returns:
        Sorted, deduplicated list of shared case ids.

    Raises:
        ValueError: No case id is shared between `eval_dir/logits` and
            `prep_dir`.
    """
    logits_dir = eval_dir / "logits"
    logits_ids = {p.stem for p in logits_dir.glob("*.npy")}
    prep_ids = {p.name for p in prep_dir.iterdir() if p.is_dir()} if prep_dir.is_dir() else set()
    shared = sorted(logits_ids & prep_ids)

    excluded = logits_ids ^ prep_ids  # present in exactly one of the two sources
    if excluded:
        logger.warning(
            "validate_qc: cohort %r excluded %d case id(s) present in only one of "
            "<eval_dir>/logits and prep_dir -- not scored.",
            cohort_name,
            len(excluded),
        )

    if not shared:
        raise ValueError(
            f"validate_qc: cohort {cohort_name!r} has no case id shared between {logits_dir} "
            f"and {prep_dir}; nothing to score."
        )

    if max_cases is not None:
        shared = shared[: int(max_cases)]
    return shared


# ---------------------------------------------------------------------------
# Per-cohort scoring
# ---------------------------------------------------------------------------


def _score_case(
    cfg: DictConfig,
    cohort: CohortSpec,
    case_id: str,
    regions: Sequence[str],
    target_shape: tuple[int, int, int],
    model: nn.Module,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Scores one case: true Dice, predicted Dice per region, entropy baseline per region.

    Args:
        cfg: The full composed Hydra config.
        cohort: The cohort `case_id` belongs to.
        case_id: The case identifier.
        regions: Region names to score, e.g. `("ET", "TC", "WT")`.
        target_shape: `(D', H', W')` the packed volume is resized to before
            the QC model sees it.
        model: The loaded `SegQC`, in eval mode.
        device: Where `model` lives.
        seed: `cfg.seed` -- see the docstring note on `generate_pairs`'s
            `generator` below for why its exact value does not affect the
            result here.

    Returns:
        One row: `case_id`, and per region `qc_pred_<R>`, `true_dice_<R>`,
        `ent_mean_fg_<R>`.
    """
    arrays = load_case_arrays(cfg, cohort.eval_dir, cohort.prep_dir, case_id)

    # True Dice of the REAL, undegraded prediction -- a second, independent
    # path to the same number per_case_metrics.csv already reports (see
    # falsification_check). An "identity" spec never touches `generator` (see
    # DegradationSpec/degrade_mask), so this generator's seed has no effect
    # on the result -- constructed explicitly anyway, never a bare/implicit
    # RNG, per this project's randomness convention.
    generator = torch.Generator().manual_seed(seed)
    pairs = generate_pairs(
        arrays.pred_mask,
        arrays.label,
        specs=[DegradationSpec("identity", 0.0)],
        generator=generator,
        per_region=False,
    )
    true_dice = pairs[0].dice  # tuple in REGION_NAMES (ET, TC, WT) order, always

    # Entropy baseline: ONE load of the case's raw logits, ONE call scoring
    # all three regions -- see the module docstring's "why loaded once" note.
    logits_path = cohort.eval_dir / "logits" / f"{case_id}.npy"
    raw_logits = np.load(logits_path).astype(np.float32)
    entropy_scalars = case_entropy_scalars(raw_logits)

    row: dict[str, Any] = {"case_id": case_id}
    with torch.no_grad():
        for region in regions:
            region_channel = REGION_NAMES.index(region)

            sample = pack_sample(arrays, arrays.pred_mask, region_channel, target_shape)
            sample = sample.unsqueeze(0).to(device)  # (1, 3, D', H', W')
            logit = model(sample)
            qc_pred = float(predicted_dice(logit).item())

            row[f"qc_pred_{region}"] = qc_pred
            row[f"true_dice_{region}"] = float(true_dice[region_channel])
            row[f"ent_mean_fg_{region}"] = entropy_scalars[f"ent_mean_fg_{region}"]

    return row


def process_cohort(
    cfg: DictConfig,
    cohort: CohortSpec,
    regions: Sequence[str],
    target_shape: tuple[int, int, int],
    model: nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    """Scores every shared case in one cohort.

    Args:
        cfg: The full composed Hydra config.
        cohort: The cohort to score.
        regions: Region names to score.
        target_shape: `(D', H', W')` passed to `pack_sample`.
        model: The loaded `SegQC`, in eval mode.
        device: Where `model` lives.

    Returns:
        A `DataFrame` indexed by `case_id`, with `qc_pred_<R>`,
        `true_dice_<R>`, `ent_mean_fg_<R>` columns for every entry of
        `regions`.
    """
    max_cases = cfg.analysis.qc_validate.max_cases
    case_ids = _shared_case_ids(cohort.eval_dir, cohort.prep_dir, max_cases, cohort.name)
    seed = int(cfg.seed)

    rows: list[dict[str, Any]] = []
    for i, case_id in enumerate(case_ids, start=1):
        rows.append(_score_case(cfg, cohort, case_id, regions, target_shape, model, device, seed))
        if i % _LOG_EVERY == 0:
            logger.info(
                "validate_qc: cohort %s -- scored %d/%d case(s).", cohort.name, i, len(case_ids)
            )

    logger.info("validate_qc: cohort %s -- scored %d case(s) total.", cohort.name, len(case_ids))
    return pd.DataFrame(rows).set_index("case_id")


def check_entropy_cache(df: pd.DataFrame, cohort: CohortSpec, regions: Sequence[str]) -> None:
    """Cross-checks recomputed entropy against a previously-cached table, if configured.

    A null `entropy_cache`, or one pointing at a file that does not exist,
    skips the check silently at INFO level. A mismatch above
    `_ENTROPY_CACHE_TOL` is logged as a WARNING, never raised -- this is an
    optional cross-check, not a falsifier.

    Args:
        df: This cohort's per-case table (indexed by `case_id`), with
            `ent_mean_fg_<R>` columns already populated.
        cohort: The cohort `df` belongs to.
        regions: Region names to check.
    """
    if cohort.entropy_cache is None:
        logger.info(
            "validate_qc: cohort %r has no entropy_cache configured; skipping the cross-check.",
            cohort.name,
        )
        return
    if not cohort.entropy_cache.is_file():
        logger.info(
            "validate_qc: entropy_cache %s does not exist; skipping the cross-check for "
            "cohort %r.",
            cohort.entropy_cache,
            cohort.name,
        )
        return

    cached = load_per_case(cohort.entropy_cache)
    common = df.index.intersection(cached.index)
    if len(common) == 0:
        logger.info(
            "validate_qc: entropy_cache %s and the recomputed table share no case ids for "
            "cohort %r; skipping the cross-check.",
            cohort.entropy_cache,
            cohort.name,
        )
        return

    max_diff = 0.0
    for region in regions:
        col = f"ent_mean_fg_{region}"
        if col not in cached.columns:
            continue
        diff = (df.loc[common, col] - cached.loc[common, col]).abs().to_numpy()
        if diff.size:
            region_max = float(np.nanmax(diff))
            max_diff = max(max_diff, region_max)

    if max_diff > _ENTROPY_CACHE_TOL:
        logger.warning(
            "validate_qc: cohort %r's recomputed ent_mean_fg differs from cached %s by up to "
            "%.6g (n_shared=%d) -- above the %.1e cross-check tolerance.",
            cohort.name,
            cohort.entropy_cache,
            max_diff,
            len(common),
            _ENTROPY_CACHE_TOL,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _print_summary(table: pd.DataFrame, verdict: dict[str, Any], out_dir: Path) -> None:
    """Prints (not logs -- see `scripts/conformal.py`'s identical convention) a compact summary."""
    lines = ["=" * 78, f"Gate C (QC validation) summary -- out_dir={out_dir}", "=" * 78]
    for _, row in table.iterrows():
        ci_lo, ci_hi = row["delta_ci_lo"], row["delta_ci_hi"]
        ci_str = (
            f"[{ci_lo:.4f}, {ci_hi:.4f}]" if pd.notna(ci_lo) and pd.notna(ci_hi) else "[nan, nan]"
        )
        p_holm_str = f"{row['p_holm']:.4g}" if pd.notna(row["p_holm"]) else "nan"
        marker = "IN_FAMILY" if bool(row["in_family"]) else "-"
        lines.append(
            f"  {row['cohort']:>10s} | {row['region']:>3s} | n={int(row['n']):4d} "
            f"n_pos={int(row['n_positive']):3d} | auroc_qc={row['auroc_qc']:.4f} "
            f"auroc_ent={row['auroc_ent']:.4f} | delta_auroc={row['delta_auroc']:+.4f} "
            f"CI={ci_str} p_holm={p_holm_str} | spearman_qc={row['spearman_qc']:.4f} "
            f"bias={row['bias']:+.4f} | {marker}"
        )
    lines.append("-" * 78)
    lines.append(
        f"  Gate C verdict: {verdict['verdict']} (alpha={verdict['alpha']}, "
        f"family_size={verdict['family_size']}, "
        f"n_family_cells_with_positive_delta={verdict['n_family_cells_with_positive_delta']})"
    )
    # print only, not logger.info as well -- matches scripts/conformal.py's
    # / scripts/calibrate.py's summary convention.
    print("\n".join(lines))


def run_validation(cfg: DictConfig) -> dict[str, Path]:
    """Runs Gate C end to end: load model -> score every cohort -> falsify -> endpoints.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A dict mapping a short name to the `Path` each output file was
        written to (one `per_case_<cohort>` entry per cohort, plus
        `falsification_csv`, `cells_csv`, `gate_c_verdict_json`,
        `silent_failure_csv`, `qc_validation_config_yaml`).

    Raises:
        FileNotFoundError: See `resolve_checkpoint`, `resolve_cohorts`.
        ValueError: See `_shared_case_ids`, and `falsification_check`
            (propagated unmodified -- no endpoint is written once any
            cohort fails its falsification check).
    """
    qcv_cfg = cfg.analysis.qc_validate
    device = get_device(cfg)
    model = load_qc_model(cfg, device)

    cohorts = resolve_cohorts(cfg)  # validates every cohort's logits/ up front
    regions = [str(r) for r in cfg.analysis.qc.regions]
    target_shape = tuple(int(v) for v in cfg.analysis.qc.target_shape)
    out_dir = ensure_dir(str(qcv_cfg.out_dir))

    # --- Step 2/3: score every cohort, write per_case_<cohort>.csv -------
    per_cohort_tables: dict[str, pd.DataFrame] = {}
    output_paths: dict[str, Path] = {}
    for cohort in cohorts:
        df = process_cohort(cfg, cohort, regions, target_shape, model, device)
        per_cohort_tables[cohort.name] = df
        check_entropy_cache(df, cohort, regions)

        per_case_path = out_dir / f"per_case_{cohort.name}.csv"
        df.reset_index().to_csv(per_case_path, index=False)
        output_paths[f"per_case_{cohort.name}"] = per_case_path
        logger.info("validate_qc: wrote %s", per_case_path)

    # --- Step 4: falsification, EVERY cohort, before ANY endpoint --------
    falsification_tol = float(qcv_cfg.falsification_tol)
    falsification_rows: list[pd.DataFrame] = []
    for cohort in cohorts:
        published = load_per_case(cohort.eval_dir / "per_case_metrics.csv")
        result = falsification_check(
            per_cohort_tables[cohort.name], published, regions, tol=falsification_tol
        )
        result = result.copy()
        result.insert(0, "cohort", cohort.name)
        falsification_rows.append(result)
    falsification_table = pd.concat(falsification_rows, ignore_index=True)
    falsification_path = out_dir / "falsification.csv"
    falsification_table.to_csv(falsification_path, index=False)
    output_paths["falsification_csv"] = falsification_path
    logger.info(
        "validate_qc: falsification check passed for all %d cohort(s); wrote %s",
        len(cohorts),
        falsification_path,
    )

    # --- Steps 6-8: endpoints, family, verdict, silent-failure -----------
    bad_dice_threshold = float(qcv_cfg.bad_dice_threshold)
    n_boot = int(qcv_cfg.n_boot)
    ci = float(qcv_cfg.ci)
    seed = int(cfg.seed)  # the ONLY source of the bootstrap's randomness

    cells = []
    for cohort in cohorts:
        df = per_cohort_tables[cohort.name]
        for region in regions:
            cells.append(
                cell_endpoints(
                    df[f"true_dice_{region}"].to_numpy(),
                    df[f"qc_pred_{region}"].to_numpy(),
                    df[f"ent_mean_fg_{region}"].to_numpy(),
                    cohort=cohort.name,
                    region=region,
                    bad_dice_threshold=bad_dice_threshold,
                    n_boot=n_boot,
                    ci=ci,
                    seed=seed,
                )
            )

    table = endpoints_table(cells)
    table = mark_family(
        table,
        in_distribution_cohort=str(qcv_cfg.in_distribution_cohort),
        min_positives=int(qcv_cfg.min_positives),
    )
    cells_path = out_dir / "cells.csv"
    table.to_csv(cells_path, index=False)
    output_paths["cells_csv"] = cells_path
    logger.info("validate_qc: wrote %s", cells_path)

    verdict = gate_c_verdict(table, alpha=float(qcv_cfg.alpha))
    verdict_path = out_dir / "gate_c_verdict.json"
    write_json(verdict, verdict_path, indent=2)
    output_paths["gate_c_verdict_json"] = verdict_path
    logger.info("validate_qc: Gate C verdict=%s. Wrote %s", verdict["verdict"], verdict_path)

    silent_table = silent_failure_table(
        table, in_distribution_cohort=str(qcv_cfg.in_distribution_cohort)
    )
    silent_path = out_dir / "silent_failure.csv"
    silent_table.to_csv(silent_path, index=False)
    output_paths["silent_failure_csv"] = silent_path
    logger.info("validate_qc: wrote %s", silent_path)

    config_path = out_dir / "qc_validation_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    output_paths["qc_validation_config_yaml"] = config_path
    logger.info("validate_qc: wrote %s", config_path)

    _print_summary(table, verdict, out_dir)

    return output_paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs Gate C validation, per the composed config.

    Example:

        python scripts/validate_qc.py

    (every knob it needs lives at `cfg.analysis.qc_validate` /
    `cfg.analysis.qc` already -- see `configs/analysis/default.yaml`.)

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_validation(cfg)


if __name__ == "__main__":
    main()
