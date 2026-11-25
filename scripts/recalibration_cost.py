"""Hydra driver for W3.5: the price of local recalibration (CRC and RCPS).

Specification: `docs/research_docs/protocols/recalibration_cost_protocol.md` (DESCRIPTIVE /
EXPLORATORY). All statistics live in `neurovision.analysis.recalibration_cost`; this script only
loads the tracked conformal v2 curves, calls the library, joins the frozen v2 rows, and writes
tables.

Background for a reader new to conformal prediction: a model's probability map is turned into a
mask by a threshold `tau`. "Conformal" fitting picks `tau` on labelled calibration cases so that
the fraction of tumour voxels the mask misses stays below `alpha`. If a new hospital's data differ
from the original calibration data the promise breaks; "local recalibration" re-fits `tau` on `k`
labelled cases from the new hospital. This script asks: how big must `k` be, what does the
recalibrated mask look like, and how many annotator-hours is `k`?

Everything written is a COUNTERFACTUAL: a literature annotation rate times `k`, not a measurement.

Run (CPU only, reads tracked result files, no inference):

    python scripts/recalibration_cost.py
    python scripts/recalibration_cost.py analysis.recalibration_cost.n_splits=20
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.recalibration_cost import (
    annotation_hours,
    k_needed,
    run_cost,
    summarise_cost,
)
from neurovision.uncertainty.conformal import CaseLossCurve, load_curves_npz
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
_REPO_ROOT = Path(__file__).resolve().parent.parent

COHORTS: tuple[str, ...] = ("test", "ssa", "ped")
CONTROL_COHORT = "test"
CONTROL_REGIONS: tuple[str, ...] = ("WT", "TC")
COST_COHORTS: tuple[str, ...] = ("ssa", "ped")  # cohorts shown in cost_table.md
STATUS_LABEL = "COUNTERFACTUAL / DESCRIPTIVE -- literature annotation rate x k, not measured"

# Frozen v2 column -> name it gets in k_needed.csv.
_FROZEN_COLUMNS = {
    "verdict": "frozen_verdict",
    "mean_miss_rate": "frozen_mean_miss_rate",
    "precision_median": "frozen_precision_median",
    "dice_median": "frozen_dice_median",
    "volume_ratio_median": "frozen_volume_ratio_median",
}


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Resolves a config path against `repo_root` (absolute paths pass through unchanged)."""
    return Path(repo_root) / Path(str(path))


def _sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    """Current commit SHA, or None if it cannot be determined (never raises)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root
        )
    except OSError:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def check_out_dir(out_dir: Path, overwrite: bool) -> None:
    """Refuses to write into a non-empty `out_dir` unless `overwrite` is true.

    Raises:
        FileExistsError: `out_dir` has contents and `overwrite` is false.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty; set analysis.recalibration_cost.overwrite=true to replace it."
        )


def load_all_curves(
    block: dict[str, Any], repo_root: Path
) -> tuple[dict[tuple[str, str, str], list[CaseLossCurve]], list[Path]]:
    """Loads curves for every model x cohort x region named in the config.

    Returns:
        `({(model, cohort, region): curves}, [every curves.npz path read])`.

    Raises:
        FileNotFoundError: A configured curves file is missing (the message names it).
        KeyError: A configured region is absent from a file (the message names it).
    """
    regions = [str(r) for r in block["regions"]]
    by_cell: dict[tuple[str, str, str], list[CaseLossCurve]] = {}
    paths: list[Path] = []
    for model, mcfg in block["models"].items():
        for cohort in COHORTS:
            path = _resolve(repo_root, mcfg["curves"][cohort])
            if not path.is_file():
                raise FileNotFoundError(
                    f"recalibration_cost: curves file for {model}/{cohort} not found: {path}"
                )
            try:
                loaded = load_curves_npz(path, regions)
            except KeyError as err:
                raise KeyError(f"{path} lacks a configured region ({err})") from err
            paths.append(path)
            for region in regions:
                by_cell[(str(model), cohort, region)] = loaded[region]
    return by_cell, paths


def join_frozen(
    k_table: pd.DataFrame, frozen: pd.DataFrame, grid_by_model: dict[str, str]
) -> pd.DataFrame:
    """Left-joins the frozen v2 realised rows onto the k_needed table.

    Matches on model, cohort, region, alpha and `guarantee == method`. Only frozen rows whose
    `grid` equals the model's configured grid are kept: a model can appear in two frozen CSVs
    fitted on different threshold grids, and joining both would duplicate its rows. Cells with no
    frozen row, or whose frozen fit was infeasible, get NaN in the metric columns.

    Raises:
        ValueError: The merge produced duplicate (model, cohort, region, alpha, method) keys.
    """
    frozen = frozen[frozen["grid"] == frozen["model"].map(grid_by_model)]
    keep = frozen[["model", "cohort", "region", "alpha", "guarantee", *_FROZEN_COLUMNS]].copy()
    keep = keep.rename(columns={**_FROZEN_COLUMNS, "guarantee": "method"})
    # Round alpha so 0.1 read from CSV and 0.1 from config compare equal.
    keep["alpha"] = keep["alpha"].astype(float).round(9)
    left = k_table.copy()
    left["_alpha_key"] = left["alpha"].astype(float).round(9)
    merged = left.merge(
        keep.rename(columns={"alpha": "_alpha_key"}),
        on=["model", "cohort", "region", "_alpha_key", "method"],
        how="left",
    )
    key = ["model", "cohort", "region", "_alpha_key", "method"]
    dup = merged[merged.duplicated(key, keep=False)]
    if not dup.empty:
        raise ValueError(
            "join_frozen: duplicate keys after merge: "
            + "; ".join(sorted({str(tuple(r)) for r in dup[key].itertuples(index=False)}))
        )
    return merged.drop(columns="_alpha_key")


def control_failures(summary: pd.DataFrame) -> list[str]:
    """Cells on the control cohort where CRC at k = half is not RESTORED (WT and TC).

    The control cohort is in-distribution, so within-cohort splits are exchangeable and CRC's
    guarantee should hold; a failure here means the machinery, not the shift, is suspect.
    """
    failures = []
    rows = summary[
        (summary["cohort"] == CONTROL_COHORT)
        & (summary["method"] == "crc")
        & (summary["region"].isin(CONTROL_REGIONS))
    ]
    for (model, region, alpha), grp in rows.groupby(["model", "region", "alpha"]):
        row = grp.sort_values("k").iloc[-1]  # largest k is "half"
        if row["verdict"] != "RESTORED":
            failures.append(f"{model}/{region}/alpha={alpha}/k={int(row['k'])}: {row['verdict']}")
    return failures


def build_cost_table(k_table: pd.DataFrame) -> str:
    """Markdown table of restorable SSA/PED cells, headed by the counterfactual label."""
    rows = k_table[k_table["cohort"].isin(COST_COHORTS) & k_table["restorable"]]
    lines = [
        f"**{STATUS_LABEL}**",
        "",
        "| model | cohort | region | alpha | method | k | manual h | "
        "Dice recal/frozen | precision recal/frozen | vol ratio recal/frozen |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def pair(a: float, b: float) -> str:
        return f"{a:.3f} / {b:.3f}"

    for r in rows.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.cohort} | {r.region} | {r.alpha:g} | {r.method} | "
            f"{int(r.k_needed)} | {r.annotation_hours_manual:.1f} | "
            f"{pair(r.dice_median, r.frozen_dice_median)} | "
            f"{pair(r.precision_median, r.frozen_precision_median)} | "
            f"{pair(r.volume_ratio_median, r.frozen_volume_ratio_median)} |"
        )
    return "\n".join(lines) + "\n"


def _run_cost_quietly(
    block: dict[str, Any],
    by_cell: dict[tuple[str, str, str], list[CaseLossCurve]],
    seed: int,
) -> list[Any]:
    """Calls the library's `run_cost` with every setting taken from the config block."""
    return run_cost(
        by_cell,
        alphas=[float(a) for a in block["alphas"]],
        methods=[str(m) for m in block["methods"]],
        k_grid=block["k_grid"],
        n_splits=int(block["n_splits"]),
        min_holdout=int(block["min_holdout"]),
        delta=float(block["delta"]),
        empty_by_region={str(k): str(v) for k, v in block["empty_by_region"].items()},
        grid_by_model={str(m): str(c["grid"]) for m, c in block["models"].items()},
        rng=np.random.default_rng(seed),
        rcps_order_seed=int(block["rcps_order_seed"]),
    )


def run_recalibration_cost(block_cfg: Any, repo_root: Path) -> dict[str, Path]:
    """Runs the whole W3.5 analysis (Hydra-free core).

    Args:
        block_cfg: The `analysis.recalibration_cost` block (DictConfig or mapping).
        repo_root: Root against which relative config paths resolve.

    Returns:
        `{short name: Path}` of each file written.

    Raises:
        FileExistsError: `out_dir` is non-empty and `overwrite` is not true.
        FileNotFoundError: A configured input file is missing.
    """
    t0 = time.perf_counter()
    block: dict[str, Any] = (
        OmegaConf.to_container(block_cfg, resolve=True)
        if isinstance(block_cfg, DictConfig)
        else dict(block_cfg)
    )
    repo_root = Path(repo_root)
    out_dir = _resolve(repo_root, block["out_dir"])
    check_out_dir(out_dir, bool(block.get("overwrite", False)))

    frozen_paths = [_resolve(repo_root, p) for p in block["frozen_realised"]]
    for p in frozen_paths:
        if not p.is_file():
            raise FileNotFoundError(f"recalibration_cost: frozen realised CSV not found: {p}")
    by_cell, curve_paths = load_all_curves(block, repo_root)

    # The fitters log one WARNING per infeasible fit; at n_splits x cells that is millions of
    # lines. Infeasibility is recorded in feasible_rate instead, so silence the logger here.
    conformal_logger = logging.getLogger("neurovision.uncertainty.conformal")
    old_level = conformal_logger.level
    conformal_logger.setLevel(logging.ERROR)
    try:
        results = _run_cost_quietly(block, by_cell, seed=int(block["seed"]))
    finally:
        conformal_logger.setLevel(old_level)
    summary = summarise_cost(results)

    # Control check: warn, never abort (the tables are still informative).
    failures = control_failures(summary)
    for line in failures:
        logger.warning("CONTROL CHECK: CRC at k=half not RESTORED on test: %s", line)
    if not failures:
        logger.info("control check passed: CRC at k=half RESTORED on test for WT/TC, all alphas")

    k_table = k_needed(summary, min_feasible_rate=float(block["min_feasible_rate"]))
    minutes = float(block["minutes_per_case"])
    k_table["annotation_hours_manual"] = [
        annotation_hours(k, minutes, 1.0) for k in k_table["k_needed"]
    ]
    k_table["annotation_hours_preannotated"] = [
        annotation_hours(k, minutes, float(block["preannotation_speedup"]))
        for k in k_table["k_needed"]
    ]
    frozen = pd.concat([pd.read_csv(p) for p in frozen_paths], ignore_index=True)
    k_table = join_frozen(
        k_table, frozen, {str(m): str(c["grid"]) for m, c in block["models"].items()}
    )

    ensure_dir(out_dir)
    paths = {
        "splits_summary_csv": out_dir / "splits_summary.csv",
        "k_needed_csv": out_dir / "k_needed.csv",
        "cost_table_md": out_dir / "cost_table.md",
        "run_meta_json": out_dir / "run_meta.json",
    }
    summary.to_csv(paths["splits_summary_csv"], index=False)
    k_table.to_csv(paths["k_needed_csv"], index=False)
    paths["cost_table_md"].write_text(build_cost_table(k_table), encoding="utf-8")
    write_json(
        {
            "status_label": STATUS_LABEL,
            "git_sha": _git_sha(repo_root),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "config": block,
            "input_sha256": {
                str(p.relative_to(repo_root) if p.is_relative_to(repo_root) else p): _sha256(p)
                for p in [*curve_paths, *frozen_paths]
            },
            "n_results": len(results),
            "control_failures": failures,
            "runtime_s": round(time.perf_counter() - t0, 3),
        },
        paths["run_meta_json"],
    )
    logger.info("recalibration_cost: wrote %d results to %s", len(results), out_dir)
    return paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs W3.5 per `cfg.analysis.recalibration_cost`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    # `overwrite` is not in the YAML; pass it as `+analysis.recalibration_cost.overwrite=true`.
    # The core reads it with `.get("overwrite", False)`.
    run_recalibration_cost(cfg.analysis.recalibration_cost, _REPO_ROOT)


if __name__ == "__main__":
    main()
