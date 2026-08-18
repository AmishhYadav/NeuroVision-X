"""Hydra entry point for the Phase 5 report-agreement experiment.

The question this script exists to answer: **does a more accurate segmentation
produce a report that agrees more with the report generated from ground
truth?** That is the one place the model's contribution to report quality can
be measured, and it is what connects the interpretable pipeline
(`scripts/localize.py`, `scripts/burden.py`, `scripts/report.py`) to the
+0.0267 ET Dice result in `docs/experiments.md` note 12.

It runs no model, loads no checkpoint, no atlas and no volume: every input is
a directory of report JSONs that `scripts/report.py` already wrote. CPU-only,
seconds per model.

Example usage:

    python scripts/report_agreement.py \\
        analysis.report_agreement.gt_dir=outputs/report_gt/reports \\
        '+analysis.report_agreement.pred_dirs.neurovision=outputs/report_neurovision/reports' \\
        '+analysis.report_agreement.pred_dirs.baseline=outputs/report_baseline/reports' \\
        output_dir=outputs/report_agreement

**Patch size is a controlled variable here, not an incidental one.**
`docs/experiments.md` note 18 measured that report agreement is NOT monotonic
in Dice: the superseded 96^3/200-epoch U-Net has lower ET Dice than
`neurovision` and still produced the better report on ET-volume agreement. The
candidate mechanism is patch size -- the failure being measured is
fragmentation of one lesion into several, a context failure rather than a
boundary one. Every model in the primary comparison
(`neurovision`, `baseline_unet3d`, `capacity_control_unet3d`) trained AND
evaluated at 64^3, verified from each run's own `eval_config.yaml`, so patch
size is held fixed across them. The 96^3 run may be included as a deliberate,
explicitly-confounded probe (it differs in epochs too) and must never be
reported inside the same Holm family as the controlled three.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.report_agreement import REPORT_AGREEMENT_VERSION, agreement_table
from neurovision.analysis.statistics import compare_models, format_comparison
from neurovision.utils.io import ensure_dir, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/localize.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def resolve_pred_dirs(cfg: DictConfig) -> dict[str, Path]:
    """Reads the model-name -> report-directory mapping, checking every path exists.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Model name -> resolved report directory, in config order.

    Raises:
        ValueError: If `pred_dirs` is empty, or any named directory is
            missing. Checked up front, before any work starts, so a typo
            costs a second rather than a full run.
    """
    raw = cfg.analysis.report_agreement.pred_dirs
    pred_dirs = {str(name): Path(str(path)) for name, path in (raw or {}).items()}
    if not pred_dirs:
        raise ValueError(
            "resolve_pred_dirs: analysis.report_agreement.pred_dirs is empty. Name at least one "
            "model, e.g. '+analysis.report_agreement.pred_dirs.neurovision="
            "outputs/report_neurovision/reports'."
        )
    missing = {name: str(path) for name, path in pred_dirs.items() if not path.is_dir()}
    if missing:
        raise ValueError(f"resolve_pred_dirs: report directory does not exist: {missing}.")
    return pred_dirs


def summarize(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reduces each model's per-case agreement table to one row per (model, metric).

    Both `mean` and `median` are emitted, and neither is a substitute for the
    other here. The `relerr_*` columns are ratios with a volume in the
    denominator, so a case whose ground-truth region is a handful of voxels
    produces a legitimately enormous value -- measured max 128.6 for
    `relerr_vol_TC` on the real test split. The mean of such a column is
    dominated by two or three cases; the median is what describes the typical
    report. `n_missing` is carried alongside for the same reason
    `metrics.segmentation.MetricAggregator` carries it: a NaN here means a
    metric was undefined for that case (an empty ground-truth region, an
    absent distance), not that it scored zero.

    Args:
        tables: Model name -> per-case agreement table.

    Returns:
        Long-format `DataFrame` with columns
        `model, metric, mean, median, std, n, n_missing`.
    """
    rows: list[dict[str, object]] = []
    for model, table in tables.items():
        for metric in table.columns:
            column = table[metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": float(column.mean()),
                    "median": float(column.median()),
                    "std": float(column.std(ddof=1)),
                    "n": int(len(column)),
                    "n_missing": int(column.isna().sum()),
                }
            )
    return pd.DataFrame.from_records(rows)


def run_report_agreement(cfg: DictConfig) -> Path:
    """Scores every named model's reports against the ground-truth reports, and compares them.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The path of `agreement_summary.csv`.

    Raises:
        ValueError: From `resolve_pred_dirs`, or when `gt_dir` is unset or
            missing, or when a `comparisons` entry names a model with no
            report directory.
    """
    agreement_cfg = cfg.analysis.report_agreement

    gt_dir = Path(str(agreement_cfg.gt_dir or ""))
    if not gt_dir.is_dir():
        raise ValueError(
            f"run_report_agreement: analysis.report_agreement.gt_dir={str(gt_dir)!r} is not a "
            "directory. It must point at reports generated with analysis.report.source=label -- "
            "the ground-truth side of the comparison."
        )

    pred_dirs = resolve_pred_dirs(cfg)
    cases = list(agreement_cfg.cases) if agreement_cfg.cases else None
    out_dir = ensure_dir(cfg.output_dir)

    tables: dict[str, pd.DataFrame] = {}
    for model, path in pred_dirs.items():
        logger.info("Scoring %s against %s", path, gt_dir)
        table = agreement_table(gt_dir, path, cases=cases)
        table.to_csv(out_dir / f"agreement_{model}.csv")
        tables[model] = table

    summary = summarize(tables)
    summary_path = out_dir / "agreement_summary.csv"
    summary.to_csv(summary_path, index=False)

    # One generator for the whole run, seeded from config: two comparisons run
    # from the same seed are reproducible, and re-running the script gives the
    # same intervals rather than intervals that wander by a few 1e-4 and make
    # a reader wonder which table is the real one.
    generator = np.random.default_rng(int(cfg.seed))

    for pair in agreement_cfg.comparisons or []:
        name_a, name_b = str(pair[0]), str(pair[1])
        for name in (name_a, name_b):
            if name not in tables:
                raise ValueError(
                    f"run_report_agreement: comparison names {name!r}, which is not in "
                    f"pred_dirs {sorted(tables)}."
                )
        # Holm is applied once across the whole returned table, and THAT table
        # is the declared family -- see the statistics module's docstring.
        comparison = compare_models(
            tables[name_a],
            tables[name_b],
            generator=generator,
            name_a=name_a,
            name_b=name_b,
            n_boot=int(agreement_cfg.n_boot),
            alpha=float(agreement_cfg.alpha),
        )
        stem = f"comparison_{name_a}_vs_{name_b}"
        comparison.to_csv(out_dir / f"{stem}.csv")
        (out_dir / f"{stem}.txt").write_text(
            format_comparison(comparison, name_a=name_a, name_b=name_b), encoding="utf-8"
        )
        logger.info("Wrote %s.csv and %s.txt", stem, stem)

    # Provenance that only ever lived in a terminal log cannot be traced months
    # later -- the same reason localize_config.yaml and burden_config.yaml exist.
    record = OmegaConf.to_container(agreement_cfg, resolve=True)
    record["report_agreement_version"] = REPORT_AGREEMENT_VERSION
    record["resolved_gt_dir"] = str(gt_dir.resolve())
    record["resolved_pred_dirs"] = {name: str(p.resolve()) for name, p in pred_dirs.items()}
    record["n_cases"] = {name: int(len(t)) for name, t in tables.items()}
    record["seed"] = int(cfg.seed)
    write_yaml(record, out_dir / "report_agreement_config.yaml")

    return summary_path


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Score prediction-derived reports against ground-truth-derived ones, per the config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    summary_path = run_report_agreement(cfg)
    print(f"Agreement summary written to {summary_path}")


if __name__ == "__main__":
    main()
