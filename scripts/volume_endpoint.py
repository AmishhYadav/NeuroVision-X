"""Hydra entry point for MasterPlan W2.5: a volume-based "usable" label next to the Dice one.

For every case of a saved evaluation this computes the predicted and the reference tumour
volume (ml), the volume-based usable label and the Dice-based usable label, from the SAVED
fp16 logits. It runs no model: logits -> sigmoid -> deployed post-processing -> mask.

Example usage:

    python scripts/volume_endpoint.py
    python scripts/volume_endpoint.py analysis.volume_endpoint.case_limit=2   # smoke run

## Why the Dice cross-check matters

The logits are only trustworthy as "the deployed mask" if re-binarising them reproduces the
Dice that `scripts/evaluate.py` wrote into `per_case_metrics.csv`. So for every case we
recompute Dice (ET, TC, WT) from the replayed mask and the label, and compare it with the CSV.
A mismatch above `dice_check_tol` means the replay is NOT the deployed mask (wrong threshold,
wrong post-processing, wrong label), and every volume below would be about a different mask.

Pure driver: all definitions live in `neurovision.analysis.volume_endpoint` and
`neurovision.analysis.error_budget`. Every path comes from `cfg.analysis.volume_endpoint`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis import error_budget, volume_endpoint
from neurovision.analysis.replay import (
    _apply_postprocess_steps,
    _binarize_regions,
    _load_label_and_spacing,
    _resolve_postprocess_cfg,
    available_logit_cases,
    load_case_logits,
)
from neurovision.analysis.statistics import load_per_case
from neurovision.metrics.segmentation import classes_to_regions, dice_score
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_LOG_EVERY = 10
_REGION_ORDER = ("ET", "TC", "WT")  # channel order of logits and masks


def _atomic_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Writes a CSV to `<path>.tmp` then renames it, so a kill never leaves a half file."""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def _replayed_mask(logits: np.ndarray, threshold: float, pp_cfg: dict[str, Any]) -> torch.Tensor:
    """Sigmoid + threshold + the deployed post-processing chain.

    Args:
        logits: `(3, D, H, W)` float32 logits, channel order (ET, TC, WT).
        threshold: Probability threshold shared by all channels.
        pp_cfg: Resolved post-processing config (replay's `_resolve_postprocess_cfg`).

    Returns:
        Binary float mask `(1, 3, D, H, W)`.
    """
    batch = torch.from_numpy(logits).unsqueeze(0)  # add the batch axis: (1, 3, D, H, W)
    return _apply_postprocess_steps(_binarize_regions(batch, threshold), pp_cfg)


def _summarise(
    table: pd.DataFrame,
    name: str,
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> pd.DataFrame:
    """One-row summary of an eval dir: usable rates with bootstrap CIs and agreement.

    Args:
        table: The per-case table (needs the usable_* and vol_rel_err_* columns).
        name: Eval dir name.
        n_boot: Bootstrap resamples.
        ci: Central confidence level, e.g. 0.95.
        seed: Seed for the bootstrap generator (so the CI is reproducible).

    Returns:
        A one-row `DataFrame`.
    """
    rng = np.random.default_rng(seed)
    row: dict[str, Any] = {"name": name, "n": len(table)}
    for key in ("dice", "volume", "both"):
        point, lo, hi = error_budget.bootstrap_rate_ci(
            table[f"usable_{key}"].astype(float).to_numpy(), n_boot=n_boot, ci=ci, generator=rng
        )
        row[f"p_usable_{key}"] = point
        row[f"p_usable_{key}_lo"] = lo
        row[f"p_usable_{key}_hi"] = hi
    # Fraction of cases where the two definitions give the same verdict.
    row["agreement_rate"] = float((table["usable_dice"] == table["usable_volume"]).mean())
    row["n_ref_empty_ET"] = int(table["ref_empty_ET"].sum())
    for region in ("WT", "ET"):
        # NaN (empty reference) is skipped by nanmedian; all-NaN gives NaN.
        vals = table[f"vol_rel_err_{region}"].to_numpy(dtype=float)
        row[f"median_vol_rel_err_{region}"] = (
            float("nan") if np.all(np.isnan(vals)) else float(np.nanmedian(vals))
        )
    return pd.DataFrame([row])


def _process_eval_dir(
    name: str,
    eval_dir: Path,
    data_dir: Path,
    block: dict[str, Any],
    threshold: float,
    pp_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the per-case table and the one-row summary for one eval dir.

    Cases are handled one at a time: whole-volume logits are large, so nothing is kept
    except small per-case numbers.

    Raises:
        FileNotFoundError: No logits, a case missing from `per_case_metrics.csv`, or a
            case with no label.
        RuntimeError: The replayed Dice disagrees with `per_case_metrics.csv` by more than
            `dice_check_tol` and `fail_on_dice_mismatch` is true.
    """
    case_ids = available_logit_cases(eval_dir)
    if not case_ids:
        raise FileNotFoundError(f"{name}: no logits under {eval_dir / 'logits'}.")
    if block["case_limit"] is not None:
        case_ids = case_ids[: int(block["case_limit"])]
    metrics = load_per_case(eval_dir / "per_case_metrics.csv")
    absent = [c for c in case_ids if c not in metrics.index]
    if absent:
        raise FileNotFoundError(f"{name}: cases missing from per_case_metrics.csv: {absent[:5]}")

    pred_rows: dict[str, dict[str, float]] = {}
    ref_rows: dict[str, dict[str, float]] = {}
    max_diffs: dict[str, float] = {}
    tol = float(block["dice_check_tol"])
    for i, case_id in enumerate(case_ids, start=1):
        loaded = _load_label_and_spacing(data_dir, case_id)
        if loaded is None:
            raise FileNotFoundError(f"{name}: no label for case {case_id} under {data_dir}.")
        label, spacing = loaded
        # Logits and label are both in the CROPPED space (what replay_case scores in).
        mask = _replayed_mask(load_case_logits(eval_dir, case_id), threshold, pp_cfg)
        target = classes_to_regions(torch.as_tensor(label))  # (1, 3, D, H, W)

        # Cross-check against the Dice evaluate.py recorded for this very case.
        dice = dice_score(mask, target)[0].numpy()
        diffs = [
            abs(float(dice[k]) - float(metrics.at[case_id, f"dice_{r}"]))
            for k, r in enumerate(_REGION_ORDER)
        ]
        max_diffs[case_id] = float(np.nanmax(diffs))
        if not max_diffs[case_id] <= tol:  # also catches NaN
            msg = f"{name} {case_id}: replayed Dice differs from per_case_metrics.csv by {diffs}"
            if block["fail_on_dice_mismatch"]:
                raise RuntimeError(msg + f" (tol {tol}). The replay is not the deployed mask.")
            logger.warning(msg)

        pred_rows[case_id] = volume_endpoint.region_volumes(mask[0], spacing)
        ref_rows[case_id] = volume_endpoint.label_volumes(label, spacing)
        if i % _LOG_EVERY == 0:
            logger.info("%s: %d/%d cases", name, i, len(case_ids))

    pred = pd.DataFrame.from_dict(pred_rows, orient="index").rename_axis("case_id")
    ref = pd.DataFrame.from_dict(ref_rows, orient="index").rename_axis("case_id")
    vol = volume_endpoint.volume_error_table(pred, ref)
    usable_vol = volume_endpoint.label_usable_volume(
        vol,
        regions=tuple(block["regions"]),
        tol=float(block["tol"]),
        empty_pred_max_ml=float(block["empty_pred_max_ml"]),
    )
    usable_dice = error_budget.label_usable(
        metrics.loc[vol.index], regions=("WT", "TC"), bar=float(block["dice_bar"])
    )
    table = pd.concat([vol, volume_endpoint.usable_both(usable_dice, usable_vol)], axis=1)
    table["dice_check_max_abs_diff"] = pd.Series(max_diffs)
    summary = _summarise(
        table,
        name,
        n_boot=int(block["n_boot"]),
        ci=float(block["ci"]),
        seed=int(block["boot_seed"]),
    )
    return table, summary


def run_volume_endpoint(block_cfg: Any, full_cfg: Any, repo_root: Path) -> dict[str, Path]:
    """Runs the volume endpoint over every configured eval dir (Hydra-free core).

    Args:
        block_cfg: The `analysis.volume_endpoint` block (DictConfig or mapping).
        full_cfg: Full config; only `inference.postprocess` is read (the deployed chain).
        repo_root: Root against which relative config paths resolve.

    Returns:
        `{name: per_case_volumes.csv path}` for every eval dir (written or skipped).
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
    written: dict[str, Path] = {}
    summaries: list[pd.DataFrame] = []
    for name, paths in block["eval_dirs"].items():
        out_dir = out_root / name
        per_case_path = out_dir / "per_case_volumes.csv"
        summary_path = out_dir / "summary.csv"
        written[name] = per_case_path
        if per_case_path.is_file() and not block["overwrite"]:
            logger.info("%s: %s exists, skipping (overwrite=false).", name, per_case_path)
            if summary_path.is_file():
                summaries.append(pd.read_csv(summary_path))
            continue
        table, summary = _process_eval_dir(
            name,
            repo_root / paths["eval_dir"],
            repo_root / paths["data_dir"],
            block,
            threshold,
            pp_cfg,
        )
        # Summary first: per_case_volumes.csv is the "done" marker the resume check reads.
        _atomic_csv(summary, summary_path)
        _atomic_csv(table, per_case_path, index=True)
        summaries.append(summary)
        row = summary.iloc[0]
        logger.info(
            "%s: n=%d usable dice %.3f volume %.3f both %.3f agreement %.3f",
            name,
            row["n"],
            row["p_usable_dice"],
            row["p_usable_volume"],
            row["p_usable_both"],
            row["agreement_rate"],
        )
    if summaries:
        _atomic_csv(pd.concat(summaries, ignore_index=True), out_root / "summary.csv")
    return written


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the W2.5 volume endpoint per `cfg.analysis.volume_endpoint`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_volume_endpoint(cfg.analysis.volume_endpoint, cfg, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
