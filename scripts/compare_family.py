"""Hydra entry point that Holm-corrects one comparison across several cohorts/tables at once.

`neurovision.analysis.statistics.compare_models` Holm-corrects the p-values it
sees in ONE call -- i.e. one pair of per-case tables, one set of metrics. But
a pre-registration such as `docs/research/preregistration_augmentation.md` or
`preregistration_multiseed.md` typically declares a SINGLE family spanning
several cohorts and metrics at once, e.g. {dice_ET, dice_TC, dice_WT} on
BraTS test (n=189) AND on pooled SSA+PED (n=159). Running `compare_models`
once per cohort and treating each call's own Holm correction as the final
answer would under-correct the family: the true family size is the sum
across every call, not the size of any one call.

## What this script does

For every entry of `cfg.analysis.compare_family.comparisons` (one cohort/table
pair each), this script calls `compare_models` once -- so each item still
gets its own (necessarily too-liberal) within-item Holm correction, kept in
the output for comparison. It then collects every item's `p_wilcoxon`
column into ONE list, re-runs `holm_bonferroni` across that whole list, and
recomputes `verdict` under the corrected `p_holm_family` -- using exactly
the decision rule `compare_models` itself uses (see that function's
docstring): inconclusive if the raw bootstrap CI contains 0 or
`p_holm_family > alpha`; else negligible if a practical threshold is given
and the whole improvement CI sits inside `+/-threshold`; else better/worse
by the sign of `improvement`.

## Why one shared `np.random.Generator`

Every `compare_models` call needs an explicit generator for its bootstrap
(no default, no global RNG -- see that function's docstring). This script
constructs exactly one, from `cfg.seed`, and passes the SAME object into
every item's call, so each item draws the next slice of one seeded stream
rather than restarting from the same state -- matching this project's
"randomness only through the seeded generator" convention
(`neurovision/utils/seed.py`) rather than re-seeding per item.

## Pooling cohorts

An item's `a` (or `b`) may be a single path or a list of paths. A list is
loaded with `neurovision.analysis.statistics.load_per_case` (case_id index)
and concatenated row-wise -- e.g. SSA and PED per-case tables pooled into
one 159-case table for a single "external" comparison. A duplicate
`case_id` surviving that concatenation is refused with `ValueError`: it
would silently double-count a case in every downstream statistic.

## Why a missing file skips only its own item

An eval directory that has not been produced yet (e.g. a Kaggle run still in
flight) must not crash the whole family -- it is logged as a WARNING naming
the exact missing path, and that one item is left out of the family. If
every item ends up skipped, there is nothing to correct, so the script
raises rather than writing an empty, meaningless `family.csv`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.statistics import compare_models, holm_bonferroni, load_per_case
from neurovision.utils.io import ensure_dir, write_yaml
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Same pattern as every other scripts/*.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# Columns copied verbatim from a compare_models() call before its own
# p_holm/reject_holm/verdict are renamed to the "_within_item" variants (see
# module docstring) -- kept so a reader can see both the (too liberal)
# within-item correction and the (correct) family-wide one side by side.
_WITHIN_ITEM_RENAME = {
    "p_holm": "p_holm_within_item",
    "reject_holm": "reject_within_item",
    "verdict": "verdict_within_item",
}


def _resolve_paths(spec: str | Sequence[str]) -> list[Path]:
    """Normalizes an item's `a`/`b` config value to a list of one or more paths.

    Args:
        spec: A single path string, or a list of path strings (pooling).

    Returns:
        `[Path(spec)]` for a single path, or one `Path` per list entry, in
        the configured order.
    """
    if isinstance(spec, str):
        return [Path(spec)]
    return [Path(str(p)) for p in spec]


def _missing_paths(paths: Sequence[Path]) -> list[Path]:
    """Returns the subset of `paths` that do not exist as files."""
    return [p for p in paths if not p.is_file()]


def _load_pooled(paths: Sequence[Path], side: str, cohort: str) -> pd.DataFrame:
    """Loads one or more per-case tables and concatenates them row-wise.

    Args:
        paths: One or more `per_case_metrics.csv`-style paths, already
            confirmed to exist by the caller.
        side: `"a"` or `"b"`, used only in the error message.
        cohort: The item's cohort name, used only in the error message.

    Returns:
        A single `DataFrame` indexed by `case_id`.

    Raises:
        ValueError: A `case_id` appears more than once after concatenation
            -- pooling e.g. SSA and PED must never silently double-count a
            case that happens to share an id across the two source tables.
    """
    tables = [load_per_case(p) for p in paths]
    pooled = pd.concat(tables, axis=0) if len(tables) > 1 else tables[0]
    dupes = pooled.index[pooled.index.duplicated()].unique()
    if len(dupes) > 0:
        raise ValueError(
            f"compare_family: cohort {cohort!r} side {side!r} has {len(dupes)} duplicate "
            f"case_id(s) after pooling {[str(p) for p in paths]}: {list(dupes)[:10]}. Pooling "
            "must never double-count a case."
        )
    return pooled


def _threshold_for(
    practical_threshold: Mapping[str, float] | float | None, metric: str
) -> float | None:
    """Resolves a per-metric practical-significance threshold.

    Mirrors `neurovision.analysis.statistics._resolve_threshold` exactly (a
    float applies to every metric; a mapping is checked by the metric's
    full name, then by its prefix; `None` disables the "negligible"
    verdict) -- reimplemented locally rather than imported, since that
    helper is a private, unexported name and this script's family-level
    verdict must apply the identical rule `compare_models` used per item.

    Args:
        practical_threshold: `cfg.analysis.compare_family.practical_threshold`.
        metric: Full metric column name, e.g. `"dice_ET"`.

    Returns:
        The threshold, or `None` if none applies.
    """
    if practical_threshold is None:
        return None
    if isinstance(practical_threshold, int | float):
        return float(practical_threshold)
    if metric in practical_threshold:
        return float(practical_threshold[metric])
    prefix = metric.split("_")[0]
    if prefix in practical_threshold:
        return float(practical_threshold[prefix])
    return None


def _family_verdict(
    row: pd.Series, alpha: float, practical_threshold: Mapping[str, float] | float | None
) -> str:
    """Recomputes one row's verdict under the family-wide Holm correction.

    Same rule `compare_models` applies per item (see that function's
    docstring), but reading `p_holm_family` and the raw `ci_lo`/`ci_hi` (for
    the "does the CI contain 0" check) instead of the within-item values.

    Args:
        row: One row of the family table, already carrying `ci_lo`, `ci_hi`,
            `p_holm_family`, `improvement`, `improvement_lo`,
            `improvement_hi`, `metric`.
        alpha: Family-wise significance level.
        practical_threshold: `cfg.analysis.compare_family.practical_threshold`.

    Returns:
        `"inconclusive"`, `"negligible"`, `"better"`, or `"worse"`.
    """
    contains_zero = bool(row["ci_lo"] <= 0.0 <= row["ci_hi"])
    if contains_zero or row["p_holm_family"] > alpha:
        return "inconclusive"

    threshold = _threshold_for(practical_threshold, str(row["metric"]))
    if (
        threshold is not None
        and abs(row["improvement_lo"]) < threshold
        and abs(row["improvement_hi"]) < threshold
    ):
        return "negligible"
    return "better" if row["improvement"] > 0 else "worse"


def run_compare_family(cfg: DictConfig) -> dict[str, Path]:
    """Runs every configured item, then Holm-corrects across the whole family.

    Args:
        cfg: The full composed Hydra config. Reads `cfg.seed` and
            `cfg.analysis.compare_family` (see the module docstring for the
            block's shape).

    Returns:
        A dict with `"family_csv"` and `"compare_family_config_yaml"`
        mapped to the paths written.

    Raises:
        ValueError: No item ran (every item's `a` or `b` file was missing),
            or a pooled `a`/`b` had a duplicate `case_id` (see
            `_load_pooled`), or `compare_models` itself raised (e.g. a
            requested metric missing from a table).
    """
    fam_cfg = cfg.analysis.compare_family
    out_dir = ensure_dir(str(fam_cfg.out_dir))
    name = str(fam_cfg.name)
    name_a = str(fam_cfg.name_a)
    name_b = str(fam_cfg.name_b)
    n_boot = int(fam_cfg.n_boot)
    ci = float(fam_cfg.ci)
    alpha = float(fam_cfg.alpha)
    practical_threshold = fam_cfg.practical_threshold

    # ONE generator, shared across every item's compare_models call -- see
    # module docstring.
    generator = np.random.default_rng(int(cfg.seed))

    item_tables: list[pd.DataFrame] = []
    for item in fam_cfg.comparisons:
        cohort = str(item.cohort)
        a_paths = _resolve_paths(item.a)
        b_paths = _resolve_paths(item.b)

        missing = _missing_paths(a_paths) + _missing_paths(b_paths)
        if missing:
            for path in missing:
                logger.warning(
                    "compare_family: cohort %r skipped -- missing file %s.", cohort, path
                )
            continue

        a_table = _load_pooled(a_paths, "a", cohort)
        b_table = _load_pooled(b_paths, "b", cohort)
        metrics = [str(m) for m in item.metrics]

        comparison = compare_models(
            a_table,
            b_table,
            generator=generator,
            metrics=metrics,
            name_a=name_a,
            name_b=name_b,
            n_boot=n_boot,
            ci=ci,
            alpha=alpha,
            practical_threshold=practical_threshold,
        )
        comparison = comparison.reset_index()  # "metric" becomes a column
        comparison.insert(0, "cohort", cohort)
        item_tables.append(comparison)
        logger.info("compare_family: cohort %r -- scored %d metric(s).", cohort, len(comparison))

    if not item_tables:
        raise ValueError(
            "compare_family: every item was skipped (missing a/b file) -- nothing to correct. "
            f"Configured comparisons: {[str(item.cohort) for item in fam_cfg.comparisons]}."
        )

    table = pd.concat(item_tables, axis=0, ignore_index=True)
    table = table.rename(columns=_WITHIN_ITEM_RENAME)

    p_holm_family, reject_family = holm_bonferroni(table["p_wilcoxon"].to_numpy(), alpha=alpha)
    table["p_holm_family"] = p_holm_family
    table["reject_family"] = reject_family
    table["verdict_family"] = table.apply(
        lambda row: _family_verdict(row, alpha, practical_threshold), axis=1
    )

    n_inconclusive = int((table["verdict_family"] == "inconclusive").sum())
    if n_inconclusive > 0:
        logger.warning(
            "compare_family: %d/%d row(s) are inconclusive under the family-wide Holm "
            "correction (m=%d) -- do NOT claim these as real differences.",
            n_inconclusive,
            len(table),
            len(table),
        )

    family_csv_path = out_dir / "family.csv"
    table.to_csv(family_csv_path, index=False)
    logger.info("compare_family: wrote %s", family_csv_path)

    config_path = out_dir / "compare_family_config.yaml"
    write_yaml(OmegaConf.to_container(fam_cfg, resolve=True), config_path)
    logger.info("compare_family: wrote %s", config_path)

    _print_summary(table, name, name_a, name_b, out_dir)

    return {"family_csv": family_csv_path, "compare_family_config_yaml": config_path}


def _print_summary(table: pd.DataFrame, name: str, name_a: str, name_b: str, out_dir: Path) -> None:
    """Prints (not logs -- matches `scripts/validate_qc.py`'s summary convention) a compact table.

    Args:
        table: The family table written to `family.csv`.
        name: `cfg.analysis.compare_family.name`, printed in the header.
        name_a: Display name for A.
        name_b: Display name for B.
        out_dir: Where `family.csv` was written, printed in the header.
    """
    m = len(table)
    lines = [
        "=" * 88,
        f"compare_family: {name} ({name_a} vs {name_b}) -- family size m={m} -- out_dir={out_dir}",
        "=" * 88,
    ]
    mean_a_col = f"mean_{name_a}"
    mean_b_col = f"mean_{name_b}"
    for _, row in table.iterrows():
        ci_str = f"[{row['improvement_lo']:+.4f}, {row['improvement_hi']:+.4f}]"
        lines.append(
            f"  {row['cohort']:>16s} | {row['metric']:>10s} | n={int(row['n']):4d} "
            f"mean_{name_a}={row[mean_a_col]:.4f} mean_{name_b}={row[mean_b_col]:.4f} "
            f"improvement={row['improvement']:+.4f} CI={ci_str} "
            f"p_holm_family={row['p_holm_family']:.4g} -> {str(row['verdict_family']).upper()}"
        )
    lines.append("-" * 88)
    print("\n".join(lines))  # print, not logger -- see convention note above


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the declared comparison family and writes a family-wide Holm correction.

    Example:

        python scripts/compare_family.py analysis/compare_family=d0_heavy_aug

    (every knob it needs lives at `cfg.analysis.compare_family` -- see the
    module docstring for its shape.)

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_compare_family(cfg)


if __name__ == "__main__":
    main()
