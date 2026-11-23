"""Hydra entry point for MasterPlan W2.3: which label-free signal finds the unusable mask?

Pre-registration: `docs/research_docs/preregistrations/preregistration_failure_detection.md`.
Do NOT run this on real data before that protocol has an external registration ID.

Example usage:

    python scripts/failure_detection.py
    python scripts/failure_detection.py analysis.failure_detection.case_limit=3   # smoke run

## The signals (every one is computed WITHOUT the label; higher score = "probably a bad mask")

- ENT: mean entropy of the deployed model's probabilities, averaged over the voxels it
  predicts as tumour. Entropy is "how unsure is the model". This is the free baseline.
- QC: minus the smaller of the SegQC model's predicted Dice for WT and TC (from a saved table).
- ENS-PD: three trained models (members) each make a mask; we measure how much the masks
  disagree (1 - Dice between pairs), worst of regions WT and TC.
- ENS-ENT: the same entropy as ENT, but on the AVERAGE probability of the three members.
- OOD: the input-statistics out-of-distribution score (from a saved table).

## The targets (computed from the label, only to score the signals)

- T-Dice: a case is unusable when dice_WT < bar or dice_TC < bar (deployed member's metrics).
- T-Vol: a case is unusable when the volume table (W2.5, `scripts/volume_endpoint.py`) says the
  mask is not usable by volume.

## Two stages

Stage 1 (heavy, per cohort, resumable): replay the saved fp16 logits of the three members, one
case at a time, and write `signals_<cohort>.csv` (ENT, ENS_ENT, ENS_PD, n_empty_wt_members).
Stage 2 (light): join with the QC / OOD / volume tables, apply the empty-foreground rule
(NaN entropy -> +inf, i.e. "most suspicious"), and write AUROC / AURC, the paired bootstrap
family (one Holm correction) and the Spearman correlations with ENT.

+inf is safe here: AUROC ranks scores (scipy `rankdata`) and AURC sorts them, and +inf simply
ranks last; the bootstrap only rejects NaN. So no value is substituted.

Every path comes from `cfg.analysis.failure_detection`; no model is run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import detection, error_budget, failure_detection
from neurovision.analysis.replay import (
    _apply_postprocess_steps,
    _binarize_regions,
    _resolve_postprocess_cfg,
    available_logit_cases,
    load_case_logits,
)
from neurovision.analysis.statistics import load_per_case
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_LOG_EVERY = 10
_CHANNEL = {"ET": 0, "TC": 1, "WT": 2}  # channel order of logits and masks
_SIGNAL_COLUMN = {  # signal id (config / paper) -> column name in the joined table
    "ENT": "ENT",
    "QC": "QC",
    "ENS-PD": "ENS_PD",
    "ENS-ENT": "ENS_ENT",
    "OOD": "OOD",
}
_TARGETS = ("T-Dice", "T-Vol")
_SIGNALS_COLUMNS = ["case_id", "ENT", "ENS_ENT", "ENS_PD", "n_empty_wt_members"]


def _atomic_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Writes a CSV to `<path>.tmp` then renames it, so a kill never leaves a half file."""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def _member_mask(logits: np.ndarray, threshold: float, pp_cfg: dict[str, Any]) -> torch.Tensor:
    """Sigmoid + threshold + deployed post-processing for one member.

    Args:
        logits: `(3, D, H, W)` float32 logits, channel order (ET, TC, WT).
        threshold: Probability threshold shared by all channels.
        pp_cfg: Resolved post-processing config.

    Returns:
        Binary float mask `(1, 3, D, H, W)`.
    """
    batch = torch.from_numpy(logits).unsqueeze(0)  # add the batch axis
    return _apply_postprocess_steps(_binarize_regions(batch, threshold), pp_cfg)


def case_signals(
    member_dirs: list[Path], case_id: str, threshold: float, pp_cfg: dict[str, Any]
) -> dict[str, float]:
    """Computes the three heavy signals for one case from the members' saved logits.

    Members are loaded one at a time; only probabilities (for ENS-ENT) and small boolean
    masks (for ENS-PD) are kept, never all three logit arrays at once.

    Args:
        member_dirs: Eval dirs of the members, DEPLOYED FIRST.
        case_id: Case identifier.
        threshold: Probability threshold for the masks.
        pp_cfg: Resolved deployed post-processing config.

    Returns:
        Dict with `ENT` (deployed member, may be NaN), `ENS_ENT` (may be NaN),
        `ENS_PD`, and `n_empty_wt_members` (members whose WT mask is empty).
    """
    probs: list[np.ndarray] = []
    masks: dict[str, list[np.ndarray]] = {"WT": [], "TC": []}
    ent = float("nan")
    n_empty_wt = 0
    for k, eval_dir in enumerate(member_dirs):
        logits = load_case_logits(eval_dir, case_id)  # float32 (3, D, H, W)
        if k == 0:
            # ENT is the deployed model's own entropy scalar (the free baseline).
            ent = float(detection.case_entropy_scalars(logits)["ent_mean_fg_mean"])
        probs.append(torch.sigmoid(torch.from_numpy(logits)).numpy())  # float32 probabilities
        mask = _member_mask(logits, threshold, pp_cfg)[0].numpy().astype(bool)
        for region in masks:
            masks[region].append(mask[_CHANNEL[region]])
        n_empty_wt += int(not mask[_CHANNEL["WT"]].any())
        del logits, mask  # free the big arrays before the next member loads
    ens_ent = failure_detection.ensemble_entropy_score(probs)
    ens_pd = failure_detection.ensemble_disagreement_score(masks, regions=("WT", "TC"))
    return {"ENT": ent, "ENS_ENT": ens_ent, "ENS_PD": ens_pd, "n_empty_wt_members": n_empty_wt}


def run_stage1(
    cohort: str,
    member_dirs: list[Path],
    out_path: Path,
    threshold: float,
    pp_cfg: dict[str, Any],
    case_limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Stage 1 for one cohort: writes `signals_<cohort>.csv` (skipped if it exists).

    The deployed (first) member's logits define the cohort's case set.

    Args:
        cohort: Cohort name (for logging).
        member_dirs: Eval dirs of the members, DEPLOYED FIRST.
        out_path: Where `signals_<cohort>.csv` goes.
        threshold: Probability threshold for the masks.
        pp_cfg: Resolved deployed post-processing config.
        case_limit: If set, only the first N cases (smoke runs).
        overwrite: Recompute even if `out_path` exists.

    Returns:
        `out_path`.

    Raises:
        FileNotFoundError: The deployed member has no logits.
        ValueError: A case of the deployed member is missing from another member.
    """
    if out_path.is_file() and not overwrite:
        logger.info("%s: %s exists, skipping (overwrite=false).", cohort, out_path)
        return out_path
    case_ids = available_logit_cases(member_dirs[0])
    if not case_ids:
        raise FileNotFoundError(f"{cohort}: no logits under {member_dirs[0] / 'logits'}.")
    # The protocol says every case is scored by every member: a gap is an error, not a drop.
    for other in member_dirs[1:]:
        have = set(available_logit_cases(other))
        missing = [c for c in case_ids if c not in have]
        if missing:
            logger.error("%s: %d cases missing from member %s", cohort, len(missing), other)
            raise ValueError(
                f"{cohort}: member {other} lacks logits for {len(missing)} deployed cases, "
                f"e.g. {missing[:5]}. The protocol requires every case from every member."
            )
    if case_limit is not None:
        case_ids = case_ids[: int(case_limit)]
    rows = []
    for i, case_id in enumerate(case_ids, start=1):
        rows.append({"case_id": case_id, **case_signals(member_dirs, case_id, threshold, pp_cfg)})
        if i % _LOG_EVERY == 0:
            logger.info("%s: %d/%d cases", cohort, i, len(case_ids))
    _atomic_csv(pd.DataFrame(rows, columns=_SIGNALS_COLUMNS), out_path)
    return out_path


def _lookup(table: pd.DataFrame, column: str, case_ids: pd.Index, what: str) -> pd.Series:
    """Column of a per-case table for exactly `case_ids`; raises on a missing case or value."""
    absent = [c for c in case_ids if c not in table.index]
    if absent:
        raise ValueError(f"{what}: missing {len(absent)} cases, e.g. {absent[:5]}")
    values = table.loc[case_ids, column]
    if values.isna().any():
        raise ValueError(f"{what}: column {column!r} has missing values.")
    return values


def build_cohort_table(
    signals: pd.DataFrame,
    qc: pd.DataFrame,
    ood: pd.DataFrame,
    volume: pd.DataFrame,
    metrics: pd.DataFrame,
    dice_bar: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Joins signals, targets and the other signals for one cohort (the empty-foreground rule).

    Args:
        signals: Stage-1 table indexed by `case_id`.
        qc: Table with `qc_pred_WT`, `qc_pred_TC`, indexed by `case_id`.
        ood: Table with `ood_score`, indexed by `case_id`.
        volume: W2.5 table with boolean `usable_volume`, indexed by `case_id`.
        metrics: Deployed member's `per_case_metrics.csv` (`dice_WT`, `dice_TC`), by `case_id`.
        dice_bar: Dice bar of the T-Dice target.

    Returns:
        `(table, replaced)`: table with columns ENT, QC, ENS_PD, ENS_ENT, OOD, `unusable_T-Dice`,
        `unusable_T-Vol`, n_empty_wt_members; `replaced` counts NaN -> +inf per signal.

    Raises:
        ValueError: A case is missing from a side table, or a signal has a missing value other
            than the NaN entropy of an empty prediction.
    """
    ids = signals.index
    table = pd.DataFrame(index=ids)
    table["ENT"] = signals["ENT"]
    table["ENS_ENT"] = signals["ENS_ENT"]
    table["ENS_PD"] = signals["ENS_PD"]
    table["n_empty_wt_members"] = signals["n_empty_wt_members"]
    # Higher score = predicted worse, so the predicted Dice enters with a minus sign.
    qc_pred = pd.concat(
        [
            _lookup(qc, "qc_pred_WT", ids, "qc table"),
            _lookup(qc, "qc_pred_TC", ids, "qc table"),
        ],
        axis=1,
    )
    table["QC"] = -qc_pred.min(axis=1)
    table["OOD"] = _lookup(ood, "ood_score", ids, "ood table")

    if table["ENS_PD"].isna().any() or table[["QC", "OOD"]].isna().any().any():
        raise ValueError("ENS_PD, QC and OOD must have no missing values.")
    replaced = {}
    for col in ("ENT", "ENS_ENT"):
        nan = table[col].isna()
        replaced[col] = int(nan.sum())
        # Empty foreground -> +inf: a study with no predicted tumour is "most suspicious".
        table.loc[nan, col] = np.inf

    for col in ("dice_WT", "dice_TC"):
        _lookup(metrics, col, ids, "per_case_metrics")
    table["unusable_T-Dice"] = ~error_budget.label_usable(
        metrics.loc[ids], regions=("WT", "TC"), bar=dice_bar
    ).astype(bool)
    table["unusable_T-Vol"] = ~_lookup(volume, "usable_volume", ids, "volume table").astype(bool)
    return table, replaced


def _cell_rows(
    table: pd.DataFrame,
    cohort: str,
    target: str,
    block: dict[str, Any],
    metric: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Bootstrap rows `signal - baseline` for one (cohort, target) cell."""
    unusable = table[f"unusable_{target}"].to_numpy(dtype=bool)
    is_est = failure_detection.estimable(unusable, int(block["min_each"]))
    base = table[_SIGNAL_COLUMN[block["baseline_signal"]]].to_numpy(dtype=float)
    rows = []
    for sig in block["signals"]:
        row: dict[str, Any] = {
            "signal": sig,
            "cohort": cohort,
            "target": target,
            "metric": metric,
            "estimable": is_est,
        }
        if is_est:
            res = failure_detection.paired_delta_bootstrap(
                table[_SIGNAL_COLUMN[sig]].to_numpy(dtype=float),
                base,
                unusable,
                metric=metric,  # type: ignore[arg-type]
                n_boot=int(block["n_boot"]),
                seed=int(block["boot_seed"]),
                ci=float(block["ci"]),
            )
            row.update(res)
        else:
            # Not estimable: leaves the family, never replaced (and a one-class cell cannot
            # be bootstrapped anyway).
            row.update({k: float("nan") for k in ("delta", "lo", "hi", "p")})
            row.update({"n": len(unusable), "n_pos": int(unusable.sum())})
        rows.append(row)
    return rows, is_est


def run_stage2(
    block: dict[str, Any],
    repo_root: Path,
    signals_paths: dict[str, Path],
) -> dict[str, Path]:
    """Stage 2: joins tables and writes metrics, family, AURC, Spearman and the run summary.

    Args:
        block: The resolved `failure_detection` config block.
        repo_root: Root against which relative config paths resolve.
        signals_paths: Cohort name -> stage-1 `signals_<cohort>.csv`.

    Returns:
        Output name -> path (`metrics`, `family`, `aurc_descriptive`, `spearman`, `run_summary`).

    Raises:
        FileNotFoundError: A volume table is missing (it comes from W2.5).
    """
    bar = float(block["dice_bar"])
    out_root = ensure_dir(repo_root / block["out_dir"])
    all_signals = [block["baseline_signal"], *block["signals"]]
    metric_rows: list[dict[str, Any]] = []
    auroc_rows: list[dict[str, Any]] = []
    aurc_rows: list[dict[str, Any]] = []
    spearman_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"cohorts": {}, "n_cells_not_estimable": 0}

    for cohort, spec in block["cohorts"].items():
        volume_path = repo_root / spec["volume_table"]
        if not volume_path.is_file():
            raise FileNotFoundError(
                f"{cohort}: volume table {volume_path} not found. T-Vol needs the W2.5 output of "
                "scripts/volume_endpoint.py; run W2.5 first (the target is never silently dropped)."
            )
        signals = pd.read_csv(signals_paths[cohort]).set_index("case_id")
        table, replaced = build_cohort_table(
            signals,
            load_per_case(repo_root / spec["qc_table"]),
            load_per_case(repo_root / spec["ood_table"]),
            load_per_case(volume_path),
            load_per_case(repo_root / spec["members"][0] / "per_case_metrics.csv"),
            bar,
        )
        summary["cohorts"][cohort] = {
            "n_cases": len(table),
            "n_replaced_inf_ENT": replaced["ENT"],
            "n_replaced_inf_ENS_ENT": replaced["ENS_ENT"],
            "n_unusable": {t: int(table[f"unusable_{t}"].sum()) for t in _TARGETS},
        }
        for sig in ("ENS-PD", "ENS-ENT"):
            spearman_rows.append(
                {
                    "cohort": cohort,
                    "signal": sig,
                    "spearman_with_ENT": detection.spearman(
                        table[_SIGNAL_COLUMN[sig]], table["ENT"]
                    ),
                }
            )
        for target in _TARGETS:
            u = table[f"unusable_{target}"].to_numpy(dtype=bool)
            for sig in all_signals:
                score = table[_SIGNAL_COLUMN[sig]].to_numpy(dtype=float)
                metric_rows.append(
                    {
                        "cohort": cohort,
                        "target": target,
                        "signal": sig,
                        "estimable": failure_detection.estimable(u, int(block["min_each"])),
                        "n": len(u),
                        "n_unusable": int(u.sum()),
                        "auroc": detection.auroc(score, u),
                        "aurc": failure_detection.aurc(score, u),
                        "oracle_aurc": failure_detection.oracle_aurc(u),
                        "random_aurc": failure_detection.random_aurc(u),
                    }
                )
            rows, is_est = _cell_rows(table, cohort, target, block, "auroc")
            auroc_rows += rows
            aurc_rows += _cell_rows(table, cohort, target, block, "aurc")[0]
            summary["n_cells_not_estimable"] += int(not is_est)

    alpha = float(block["alpha"])
    family = failure_detection.detection_family(
        pd.DataFrame(auroc_rows), alpha=alpha, higher_is_better=True
    )
    aurc_desc = failure_detection.detection_family(
        pd.DataFrame(aurc_rows), alpha=alpha, higher_is_better=False
    )
    aurc_desc["role"] = "descriptive"  # not part of the registered family
    paths = {
        "metrics": out_root / "metrics.csv",
        "family": out_root / "family.csv",
        "aurc_descriptive": out_root / "aurc_descriptive.csv",
        "spearman": out_root / "spearman.csv",
        "run_summary": out_root / "run_summary.json",
    }
    _atomic_csv(pd.DataFrame(metric_rows), paths["metrics"])
    _atomic_csv(family, paths["family"])
    _atomic_csv(aurc_desc, paths["aurc_descriptive"])
    _atomic_csv(pd.DataFrame(spearman_rows), paths["spearman"])
    paths["run_summary"].write_text(json.dumps(summary, indent=2))
    return paths


def run_failure_detection(block_cfg: Any, full_cfg: Any, repo_root: Path) -> dict[str, Path]:
    """Runs stage 1 for every cohort, then stage 2 (Hydra-free core).

    Args:
        block_cfg: The `analysis.failure_detection` block (DictConfig or mapping).
        full_cfg: Full config; only `inference.postprocess` is read (the deployed chain).
        repo_root: Root against which relative config paths resolve.

    Returns:
        The stage-2 output paths.
    """
    block: dict[str, Any] = (
        OmegaConf.to_container(block_cfg, resolve=True)
        if isinstance(block_cfg, DictConfig)
        else dict(block_cfg)
    )
    pp_full = (
        OmegaConf.to_container(full_cfg.inference.postprocess, resolve=True)
        if isinstance(full_cfg.inference.postprocess, DictConfig)
        else dict(full_cfg.inference.postprocess)
    )
    threshold = float(pp_full.pop("threshold"))
    pp_cfg = _resolve_postprocess_cfg(pp_full)

    out_root = repo_root / block["out_dir"]
    signals_paths: dict[str, Path] = {}
    for cohort, spec in block["cohorts"].items():
        signals_paths[cohort] = run_stage1(
            cohort,
            [repo_root / m for m in spec["members"]],
            out_root / f"signals_{cohort}.csv",
            threshold,
            pp_cfg,
            case_limit=block["case_limit"],
            overwrite=bool(block["overwrite"]),
        )
    return run_stage2(block, repo_root, signals_paths)


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs W2.3 failure detection per `cfg.analysis.failure_detection`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_failure_detection(
        cfg.analysis.failure_detection, cfg, Path(__file__).resolve().parent.parent
    )


if __name__ == "__main__":
    main()
