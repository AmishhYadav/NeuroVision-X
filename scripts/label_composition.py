"""Hydra entry point: tumour sub-region COMPOSITION in reference labels and predictions.

D3 note 58, post-hoc mechanism. The question: does the paediatric cohort label nearly
the whole tumour as "core"? Two measurements, both CPU-only and light:

1. Reference labels. Each case's `meta.json` already holds `label_voxel_counts`
   (internal labels after remap: 1 = NCR/NET, 2 = ED, 3 = ET), so `label.npy` is
   never loaded. WT = 1 + 2 + 3, TC = 1 + 3. Per case: each label's share of WT, and
   `tc_wt` = TC / WT.
2. Predictions. Saved fp16 logits `<eval_dir>/logits/<case>.npy`, shape
   (3, D, H, W), channels ordered like `cfg.data.regions`. A region is "on" where its
   logit > 0 (sigmoid 0.5). Per case: `tc_wt` = predicted TC voxels / predicted WT voxels.

All paths and thresholds come from `cfg.analysis.label_composition`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def reference_counts(root: Path, cohort: str) -> tuple[pd.DataFrame, int]:
    """Per-case label composition for one cohort, read from `<root>/<case>/meta.json`.

    Args:
        root: Preprocessed root holding one directory per case.
        cohort: Cohort name, written into the `cohort` column.

    Returns:
        `(table, n_skipped)`. Columns: cohort, case_id, ncr_share, ed_share,
        et_share, tc_wt. Cases without `label_voxel_counts`, or with WT = 0, are
        skipped and counted.

    Raises:
        FileNotFoundError: `root` does not exist.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"label_composition: reference root not found: {root}")
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for meta_path in sorted(root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        counts = meta.get("label_voxel_counts")
        if not counts:
            n_skipped += 1
            continue
        # A label absent from the dict simply has no voxels.
        ncr, ed, et = (float(counts.get(k, 0)) for k in ("1", "2", "3"))
        wt = ncr + ed + et
        if wt == 0:
            n_skipped += 1
            continue
        rows.append(
            {
                "cohort": cohort,
                "case_id": meta_path.parent.name,
                "ncr_share": ncr / wt,
                "ed_share": ed / wt,
                "et_share": et / wt,
                "tc_wt": (ncr + et) / wt,
            }
        )
    columns = ["cohort", "case_id", "ncr_share", "ed_share", "et_share", "tc_wt"]
    return pd.DataFrame(rows, columns=columns), n_skipped


def prediction_ratios(
    eval_dirs: Sequence[Path], set_name: str, regions: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    """Per-case predicted TC/WT voxel ratio for one prediction set.

    Args:
        eval_dirs: Eval directories, each holding `logits/<case>.npy` of shape
            (3, D, H, W) -- one case, channels ordered as `regions`.
        set_name: Name written into the `set` column.
        regions: Channel order, i.e. `cfg.data.regions`.

    Returns:
        `(table, n_skipped)` with columns set, case_id, tc_wt. Cases whose predicted
        WT is empty are skipped and counted.

    Raises:
        FileNotFoundError: An eval dir or its `logits/` folder is missing.
        ValueError: A case_id appears in more than one eval dir, or a logit file
            does not have 3 channels.
    """
    regions = list(regions)
    tc_idx, wt_idx = regions.index("TC"), regions.index("WT")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    n_skipped = 0
    for eval_dir in eval_dirs:
        logits_dir = eval_dir / "logits"
        if not logits_dir.is_dir():
            raise FileNotFoundError(f"label_composition: logits dir not found: {logits_dir}")
        for path in sorted(logits_dir.glob("*.npy")):
            case_id = path.stem
            if case_id in seen:
                raise ValueError(
                    f"label_composition: case {case_id!r} appears in more than one eval dir "
                    f"of prediction set {set_name!r} (again in {eval_dir})"
                )
            seen.add(case_id)
            logits = np.load(path, mmap_mode="r")
            if logits.shape[0] != 3:
                raise ValueError(f"label_composition: {path} has {logits.shape[0]} channels, not 3")
            tc = int((logits[tc_idx] > 0).sum())
            wt = int((logits[wt_idx] > 0).sum())
            if wt == 0:
                n_skipped += 1
                continue
            rows.append({"set": set_name, "case_id": case_id, "tc_wt": tc / wt})
    return pd.DataFrame(rows, columns=["set", "case_id", "tc_wt"]), n_skipped


def _summary_row(
    kind: str, name: str, table: pd.DataFrame, n_skipped: int, cfg_lc: DictConfig
) -> dict[str, Any]:
    """One summary.csv row; columns that do not apply to `kind` are NaN."""
    nan = float("nan")
    row: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "n": len(table),
        "n_skipped": n_skipped,
        "median_ncr_share": nan,
        "median_ed_share": nan,
        "median_et_share": nan,
        "median_tc_wt": float(table["tc_wt"].median()) if len(table) else nan,
        "frac_tc_wt_below_low_ratio": nan,
        "frac_ed_share_below_low_ed_share": nan,
    }
    if len(table):
        row["frac_tc_wt_below_low_ratio"] = float((table["tc_wt"] < float(cfg_lc.low_ratio)).mean())
    if kind == "reference" and len(table):
        row["median_ncr_share"] = float(table["ncr_share"].median())
        row["median_ed_share"] = float(table["ed_share"].median())
        row["median_et_share"] = float(table["et_share"].median())
        row["frac_ed_share_below_low_ed_share"] = float(
            (table["ed_share"] < float(cfg_lc.low_ed_share)).mean()
        )
    return row


def _log_summary(summary: pd.DataFrame) -> None:
    """Logs the summary table as readable text."""
    logger.info(
        "label_composition summary:\n%s", summary.to_string(index=False, float_format="%.4f")
    )


def run(cfg: DictConfig) -> dict[str, Path]:
    """Runs both measurements and writes the three CSVs.

    Args:
        cfg: Full composed config; reads `cfg.analysis.label_composition` and
            `cfg.data.regions`.

    Returns:
        Mapping of short name to written path.
    """
    lc = cfg.analysis.label_composition
    regions = [str(r) for r in cfg.data.regions]
    out_dir = ensure_dir(str(lc.out_dir))

    summary_rows: list[dict[str, Any]] = []
    ref_tables: list[pd.DataFrame] = []
    for name, root in lc.references.items():
        table, n_skipped = reference_counts(Path(str(root)), str(name))
        logger.info(
            "label_composition: reference %s: n=%d, skipped %d", name, len(table), n_skipped
        )
        ref_tables.append(table)
        summary_rows.append(_summary_row("reference", str(name), table, n_skipped, lc))

    pred_tables: list[pd.DataFrame] = []
    predictions: Mapping[str, Sequence[str]] = lc.predictions
    for name, dirs in predictions.items():
        table, n_skipped = prediction_ratios([Path(str(d)) for d in dirs], str(name), regions)
        logger.info(
            "label_composition: prediction %s: n=%d, skipped %d", name, len(table), n_skipped
        )
        pred_tables.append(table)
        summary_rows.append(_summary_row("prediction", str(name), table, n_skipped, lc))

    paths = {
        "reference": out_dir / "reference_per_case.csv",
        "prediction": out_dir / "prediction_per_case.csv",
        "summary": out_dir / "summary.csv",
    }
    pd.concat(ref_tables, ignore_index=True).to_csv(paths["reference"], index=False)
    pd.concat(pred_tables, ignore_index=True).to_csv(paths["prediction"], index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["summary"], index=False)
    _log_summary(summary)
    return paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Measures label composition, per `cfg.analysis.label_composition`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run(cfg)


if __name__ == "__main__":
    main()
