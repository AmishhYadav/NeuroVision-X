"""Regenerates the paper's "reference-ET" column (Table IV) from saved per-case CSVs.

## Why this script exists

Five test cases have an EMPTY reference enhancing tumour (ET). Under the BraTS
convention an empty reference with an empty prediction scores Dice 1, and any
false-positive voxel scores 0. Two of those five cases score 1 for the proposed
model and 0 for the baseline, which carries roughly 40% of the headline mean
ET difference. So the paper reports a second, exploratory column: the same
comparison restricted to test cases WITH reference ET (n = 184). This script
makes that number reproducible from files on disk -- CPU only, seconds, no
model and no GPU. It also writes the empty-case breakdown.

## Why a FRESH generator per (comparison, subset)

`scripts/compare_family.py` shares ONE `np.random.Generator` across all items,
so each item's bootstrap CI depends on how many rows ran before it. Here we
build a new `np.random.default_rng(seed)` for EVERY (comparison, subset) call
instead. Each row's CI is then independent of row order and of which other
comparisons are configured, and this is what reproduces the published values.

## Outputs (in `cfg.analysis.ref_et.out_dir`)

- `ref_et_comparisons.csv`: one row per comparison x subset ("all", "ref_et").
- `empty_et_cases.csv`: one row per empty-flag case, each arm's metric value.
- `ref_et_config.yaml`: the resolved config block, for provenance.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.statistics import (
    load_per_case,
    paired_bootstrap_ci,
    wilcoxon_signed_rank,
)
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_SUBSETS = ("all", "ref_et")
_TRUE_STRINGS = {"true", "1", "1.0"}
_FALSE_STRINGS = {"false", "0", "0.0"}


def _parse_flag(series: pd.Series, arm: str, path: str | Path) -> pd.Series:
    """Parses an empty-flag column into real booleans.

    Accepts bool dtype, numeric 0/1 (how the CSVs store it), or the exact
    strings "True"/"False" (a CSV round-trip artefact). Strings are compared
    for equality, never by substring.

    Args:
        series: The raw flag column.
        arm: Arm name, for error messages.
        path: CSV path, for error messages.

    Returns:
        A bool Series with the same index.

    Raises:
        ValueError: A value cannot be read as a boolean.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        if series.isna().any():
            raise ValueError(f"arm {arm!r} ({path}): empty-flag column has NaN.")
        if not series.isin([0, 1]).all():
            raise ValueError(f"arm {arm!r} ({path}): numeric empty flag not in {{0,1}}.")
        return series == 1

    def one(value: object) -> bool:
        if isinstance(value, bool | np.bool_):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        raise ValueError(f"arm {arm!r} ({path}): cannot parse empty flag value {value!r}.")

    return series.map(one).astype(bool)


def load_arms(
    arms: Mapping[str, str | Path], metric: str, empty_flag_column: str
) -> dict[str, pd.DataFrame]:
    """Loads and validates every arm's per-case table.

    Args:
        arms: Mapping arm name -> `per_case_metrics.csv` path.
        metric: Metric column, e.g. `"dice_ET"`.
        empty_flag_column: Boolean "reference is empty" column.

    Returns:
        Arm name -> DataFrame indexed by `case_id`, with exactly two columns:
        `metric` (float) and `empty_flag_column` (bool).

    Raises:
        ValueError: Missing column, NaN metric, differing case-id sets, or an
            empty flag that disagrees between arms (it comes from the ground
            truth, so a disagreement means the files do not match).
    """
    tables: dict[str, pd.DataFrame] = {}
    for arm, path in arms.items():
        df = load_per_case(path)
        if df.index.name != "case_id":
            # Without a case_id index, pairing would silently become positional.
            raise ValueError(f"arm {arm!r} ({path}): no 'case_id' column.")
        for col in (metric, empty_flag_column):
            if col not in df.columns:
                raise ValueError(f"arm {arm!r} ({path}): missing column {col!r}.")
        if df.index.duplicated().any():
            raise ValueError(f"arm {arm!r} ({path}): duplicate case_id values.")
        values = df[metric].astype(float)
        if values.isna().any():
            raise ValueError(f"arm {arm!r} ({path}): NaN in metric column {metric!r}.")
        flag = _parse_flag(df[empty_flag_column], arm, path)
        tables[arm] = pd.DataFrame({metric: values, empty_flag_column: flag})

    if not tables:
        raise ValueError("load_arms: no arms given.")

    names = list(tables)
    ref_name = names[0]
    ref_ids = set(tables[ref_name].index)
    for arm in names[1:]:
        ids = set(tables[arm].index)
        if ids != ref_ids:
            raise ValueError(
                f"arm {arm!r} ({arms[arm]}) has a different case_id set than {ref_name!r} "
                f"({arms[ref_name]}): {len(ids ^ ref_ids)} case(s) differ."
            )
    ref_flag = tables[ref_name][empty_flag_column]
    for arm in names[1:]:
        flag = tables[arm][empty_flag_column].reindex(ref_flag.index)
        if not (flag == ref_flag).all():
            bad = list(ref_flag.index[flag != ref_flag][:5])
            raise ValueError(
                f"arm {arm!r} ({arms[arm]}) disagrees with {ref_name!r} on the empty flag "
                f"{empty_flag_column!r} for case(s) {bad}; the files do not match."
            )
    return tables


def hodges_lehmann(diff: np.ndarray) -> float:
    """Hodges-Lehmann estimate: median of the Walsh averages (d_i + d_j) / 2, i <= j.

    Args:
        diff: 1-D array of paired differences.

    Returns:
        The pseudo-median of `diff`.
    """
    diff = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(diff.size)  # all pairs with i <= j, including i == j
    return float(np.median((diff[i] + diff[j]) / 2.0))


def compare_pair(
    a_name: str,
    b_name: str,
    tables: Mapping[str, pd.DataFrame],
    metric: str,
    empty_flag_column: str,
    n_boot: int,
    ci: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Compares arm `a` against arm `b` (a - b) on the "all" and "ref_et" subsets.

    Args:
        a_name: First arm.
        b_name: Second arm.
        tables: Output of `load_arms`.
        metric: Metric column.
        empty_flag_column: Empty-flag column.
        n_boot: Bootstrap replicates.
        ci: Confidence level.
        seed: Seed for a fresh generator per subset (see module docstring).

    Returns:
        Two row dicts, one per subset, in the order ("all", "ref_et").
    """
    ta = tables[a_name]
    tb = tables[b_name].reindex(ta.index)  # same case order for pairing
    empty = ta[empty_flag_column].to_numpy()
    a_all = ta[metric].to_numpy()
    b_all = tb[metric].to_numpy()
    n_all = a_all.size
    empty_sum = float((a_all - b_all)[empty].sum())

    rows: list[dict[str, Any]] = []
    for subset in _SUBSETS:
        keep = np.ones(n_all, dtype=bool) if subset == "all" else ~empty
        a, b = a_all[keep], b_all[keep]
        diff = a - b
        boot = paired_bootstrap_ci(
            a, b, n_boot=n_boot, ci=ci, generator=np.random.default_rng(seed)
        )
        wil = wilcoxon_signed_rank(a, b)
        rows.append(
            {
                "a": a_name,
                "b": b_name,
                "subset": subset,
                "n": int(diff.size),
                "mean_diff": float(diff.mean()),
                "ci_lo": boot.lo,
                "ci_hi": boot.hi,
                "p_wilcoxon": wil.pvalue,
                "wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()),
                "hodges_lehmann": hodges_lehmann(diff),
                # Share of the all-cases mean carried by empty-reference cases.
                "empty_contribution": empty_sum / n_all if subset == "all" else float("nan"),
            }
        )
    return rows


def run(cfg_block: Mapping[str, Any] | DictConfig) -> pd.DataFrame:
    """Runs every configured comparison and writes the three output files.

    Args:
        cfg_block: `cfg.analysis.ref_et` (see `configs/analysis/default.yaml`).

    Returns:
        The comparisons table that was written to `ref_et_comparisons.csv`.
    """
    metric = str(cfg_block["metric"])
    flag_col = str(cfg_block["empty_flag_column"])
    arms = {str(k): str(v) for k, v in cfg_block["arms"].items()}
    out_dir = ensure_dir(str(cfg_block["out_dir"]))

    tables = load_arms(arms, metric, flag_col)

    rows: list[dict[str, Any]] = []
    for pair in cfg_block["comparisons"]:
        a_name, b_name = str(pair[0]), str(pair[1])
        for name in (a_name, b_name):
            if name not in tables:
                raise ValueError(f"comparison names unknown arm {name!r}; arms: {list(tables)}.")
        rows.extend(
            compare_pair(
                a_name,
                b_name,
                tables,
                metric,
                flag_col,
                int(cfg_block["n_boot"]),
                float(cfg_block["ci"]),
                int(cfg_block["seed"]),
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ref_et_comparisons.csv", index=False)

    first = next(iter(tables.values()))
    empty_ids = first.index[first[flag_col]]
    empty_table = pd.DataFrame({arm: t.loc[empty_ids, metric] for arm, t in tables.items()})
    empty_table.index.name = "case_id"
    empty_table.reset_index().to_csv(out_dir / "empty_et_cases.csv", index=False)

    cfg_yaml = (
        OmegaConf.to_yaml(cfg_block, resolve=True)
        if isinstance(cfg_block, DictConfig)
        else OmegaConf.to_yaml(OmegaConf.create(dict(cfg_block)), resolve=True)
    )
    (out_dir / "ref_et_config.yaml").write_text(cfg_yaml)

    _log_summary(table, len(empty_ids))
    logger.info("ref_et: wrote outputs to %s", out_dir)
    return table


def _log_summary(table: pd.DataFrame, n_empty: int) -> None:
    """Logs one compact line per comparison: all-cases diff vs ref-ET diff with CI."""
    lines = [f"ref_et summary ({n_empty} empty-reference case(s)):"]
    for (a, b), grp in table.groupby(["a", "b"], sort=False):
        allr = grp[grp["subset"] == "all"].iloc[0]
        ref = grp[grp["subset"] == "ref_et"].iloc[0]
        lines.append(
            f"  {a} - {b}: all(n={allr['n']}) {allr['mean_diff']:+.4f} | "
            f"ref_et(n={ref['n']}) {ref['mean_diff']:+.4f} "
            f"[{ref['ci_lo']:+.4f}, {ref['ci_hi']:+.4f}]"
        )
    logger.info("\n".join(lines))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: `python scripts/ref_et_comparison.py`.

    Args:
        cfg: Composed config; reads `cfg.analysis.ref_et`.
    """
    setup_logging(level="INFO")
    run(cfg.analysis.ref_et)


if __name__ == "__main__":
    main()
