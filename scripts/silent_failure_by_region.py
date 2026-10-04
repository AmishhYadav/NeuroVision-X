"""Splits each cohort's silent failures by WHICH region made them unusable.

## Why this script exists

The registered error budget counts a study as a "silent failure" when the
deployed gate ACCEPTED it but it is UNUSABLE, where usable means Dice >= the bar
for BOTH whole tumour (WT) and tumour core (TC). The paediatric cohort's
tumour-core label means something different from the adult one, so a large PED
silent-failure rate may partly measure that label change. This script says, for
every silent failure, which region(s) fell below the bar, and also reports the
silent-failure rate if "usable" meant each single region alone. CPU only,
seconds, no model and no GPU. Exploratory, not pre-registered.

## Outputs (in `cfg.analysis.silent_failure_by_region.out_dir`)

- `silent_failure_by_region.csv`: one row per cohort (counts and rates).
- `silent_failure_cases.csv`: one row per silent-failure case.
- `silent_failure_by_region_config.yaml`: the resolved config block.
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

from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_TRUE_STRINGS = {"True"}
_FALSE_STRINGS = {"False"}
_MAX_IDS_IN_ERROR = 10


def parse_bool(series: pd.Series, column: str, cohort: str) -> pd.Series:
    """Parses a boolean CSV column by exact equality.

    Args:
        series: The raw column (bool dtype, or strings "True"/"False").
        column: Column name, for error messages.
        cohort: Cohort name, for error messages.

    Returns:
        A bool Series with the same index.

    Raises:
        ValueError: A value is neither a bool nor exactly "True"/"False".
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    def one(value: object) -> bool:
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, str):
            if value in _TRUE_STRINGS:
                return True
            if value in _FALSE_STRINGS:
                return False
        raise ValueError(f"cohort {cohort!r}: cannot parse {column!r} value {value!r}.")

    return series.map(one).astype(bool)


def _raise_mismatch(what: str, cohort: str, ids: pd.Series) -> None:
    shown = list(ids[:_MAX_IDS_IN_ERROR])
    raise ValueError(
        f"cohort {cohort!r}: recomputed {what} disagrees with the file for "
        f"{len(ids)} case(s): {shown}."
    )


def analyse_cohort(
    df: pd.DataFrame, cohort: str, regions: Sequence[str], bar: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Splits one cohort's silent failures by failing region.

    Args:
        df: Per-case table with `case_id`, `accepted`, `usable`, `cell` and
            `dice_<R>` for every region R.
        cohort: Cohort name, for messages and output rows.
        regions: Regions that must all reach the bar (config order).
        bar: Dice threshold; Dice >= bar counts as passing.

    Returns:
        (summary row dict, DataFrame of silent-failure cases).

    Raises:
        ValueError: Missing column, NaN Dice, unparseable boolean, or a
            `usable` / `cell` column that disagrees with the recomputation.
    """
    # Exact column names only: `predicted_dice_*` columns must never match.
    needed = ["case_id", "accepted", "usable", "cell"] + [f"dice_{r}" for r in regions]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"cohort {cohort!r}: missing column(s) {missing}.")
    if len(df) == 0:
        raise ValueError(f"cohort {cohort!r}: per-case CSV is empty.")

    dice = {r: df[f"dice_{r}"].astype(float) for r in regions}
    for r, values in dice.items():
        if values.isna().any():
            raise ValueError(f"cohort {cohort!r}: NaN in dice_{r}.")

    accepted = parse_bool(df["accepted"], "accepted", cohort)
    file_usable = parse_bool(df["usable"], "usable", cohort)
    case_ids = df["case_id"].astype(str)

    # Guard against the bar or regions drifting from the registered error budget.
    passes = {r: dice[r] >= bar for r in regions}
    usable = pd.Series(True, index=df.index)
    for r in regions:
        usable &= passes[r]
    if (usable != file_usable).any():
        _raise_mismatch("usable", cohort, case_ids[usable != file_usable])

    silent = accepted & ~usable
    file_silent = df["cell"] == "silent_failure"
    if (silent != file_silent).any():
        _raise_mismatch("silent_failure", cohort, case_ids[silent != file_silent])

    # Which regions are below the bar, for each silent-failure case.
    cases = pd.DataFrame({"cohort": cohort, "case_id": case_ids[silent]})
    for r in regions:
        cases[f"dice_{r}"] = dice[r][silent]
    failing = pd.DataFrame({r: ~passes[r][silent] for r in regions})
    # List comprehension (not .apply) so the column is always str, even when empty.
    cases["failing_regions"] = pd.Series(
        ["+".join(r for r in regions if failing.at[i, r]) for i in failing.index],
        index=failing.index,
        dtype=object,
    )
    cases = cases.reset_index(drop=True)

    n = len(df)
    n_silent = int(silent.sum())
    n_fail = failing.sum(axis=1)  # number of failing regions per silent case
    row: dict[str, Any] = {
        "cohort": cohort,
        "n": n,
        "n_accepted": int(accepted.sum()),
        "n_silent": n_silent,
        "silent_rate": n_silent / n,
    }
    for r in regions:
        row[f"only_{r}"] = int((failing[r] & (n_fail == 1)).sum())
    row["multiple"] = int((n_fail > 1).sum())
    for r in regions:
        # Silent-failure count if "usable" meant only this one region.
        count = int((accepted & ~passes[r]).sum())
        row[f"silent_if_{r}_only"] = count
        row[f"silent_rate_if_{r}_only"] = count / n
    return row, cases


def run(cfg_block: Mapping[str, Any] | DictConfig) -> pd.DataFrame:
    """Analyses every configured cohort and writes the output files.

    Args:
        cfg_block: `cfg.analysis.silent_failure_by_region`.

    Returns:
        The per-cohort table written to `silent_failure_by_region.csv`.
    """
    regions = [str(r) for r in cfg_block["usable_regions"]]
    bar = float(cfg_block["usable_bar"])
    per_case_dir = Path(str(cfg_block["per_case_dir"]))
    out_dir = ensure_dir(str(cfg_block["out_dir"]))

    rows: list[dict[str, Any]] = []
    case_tables: list[pd.DataFrame] = []
    for cohort in (str(c) for c in cfg_block["cohorts"]):
        path = per_case_dir / f"per_case_{cohort}.csv"
        row, cases = analyse_cohort(pd.read_csv(path), cohort, regions, bar)
        rows.append(row)
        case_tables.append(cases)
        parts = ", ".join(f"only_{r}={row[f'only_{r}']}" for r in regions)
        logger.info(
            "silent_failure_by_region %s: n=%d accepted=%d silent=%d (%.1f%%) %s multiple=%d",
            cohort,
            row["n"],
            row["n_accepted"],
            row["n_silent"],
            100 * row["silent_rate"],
            parts,
            row["multiple"],
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "silent_failure_by_region.csv", index=False)
    pd.concat(case_tables, ignore_index=True).to_csv(
        out_dir / "silent_failure_cases.csv", index=False
    )

    cfg_yaml = (
        OmegaConf.to_yaml(cfg_block, resolve=True)
        if isinstance(cfg_block, DictConfig)
        else OmegaConf.to_yaml(OmegaConf.create(dict(cfg_block)), resolve=True)
    )
    (out_dir / "silent_failure_by_region_config.yaml").write_text(cfg_yaml)
    logger.info("silent_failure_by_region: wrote outputs to %s", out_dir)
    return table


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: `python scripts/silent_failure_by_region.py`.

    Args:
        cfg: Composed config; reads `cfg.analysis.silent_failure_by_region`.
    """
    setup_logging(level="INFO")
    run(cfg.analysis.silent_failure_by_region)


if __name__ == "__main__":
    main()
