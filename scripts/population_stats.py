"""Hydra entry point for the Phase 5 population-anatomy statistics.

Aggregates one or more `scripts/localize.py` runs into cohort-level anatomy:
how often each atlas structure is involved, how tumour volume distributes over
lobes, the left/right balance, and the eloquence rates. Writes the tables and
the two figures `neurovision.visualization.figures` provides for them.

Runs no model and loads no atlas -- every input is a CSV a localisation run
already wrote. CPU-only, seconds.

Example usage, over the full 1251-case cohort (three frozen splits, all
ground truth):

    python scripts/population_stats.py output_dir=outputs/population_gt \\
        '+analysis.population.localize_dirs=[outputs/localize_gt,\\
         outputs/localize_gt_train,outputs/localize_gt_val]'

The two guards this script exists to enforce, both of which produce an
entirely plausible wrong answer if skipped:

1. **Every input directory must have the same `source`.** Concatenating a
   ground-truth localisation with a prediction-derived one produces a
   "population" that is partly the cohort's real anatomy and partly a model's
   opinion of it, with no column marking which is which.
2. **No two input directories may name the same split.** The splits are
   disjoint by construction, so distinct splits concatenate to a cohort; the
   same split twice doubles every count while leaving every FRACTION
   unchanged, which is exactly the failure that would survive a plausibility
   check on the output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.population import (
    eloquence_rates,
    laterality_distribution,
    lobe_burden_distribution,
    structure_involvement_frequency,
)
from neurovision.utils.io import ensure_dir, read_yaml, write_json, write_yaml
from neurovision.utils.logging import setup_logging
from neurovision.visualization.figures import (
    paper_style,
    plot_lobe_distribution,
    plot_structure_involvement,
    save_figure,
)

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/localize.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def load_localize_runs(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Loads and concatenates several localisation runs, refusing to mix incompatible ones.

    Args:
        dirs: Directories, each holding `anatomy.csv`, `anatomy_summary.csv`
            and `localize_config.yaml`.

    Returns:
        `(anatomy, anatomy_summary, configs)` -- the two concatenated frames
        and each run's parsed config record, in input order.

    Raises:
        ValueError: If `dirs` is empty, a directory is missing one of the
            three files, two runs disagree on `source`, two runs name the same
            `split`, or the concatenated frames contain a duplicated
            `case_id`.
    """
    if not dirs:
        raise ValueError(
            "load_localize_runs: analysis.population.localize_dirs is empty. Name at least one "
            "scripts/localize.py output directory."
        )

    anatomy_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    configs: list[dict] = []

    for directory in dirs:
        for name in ("anatomy.csv", "anatomy_summary.csv", "localize_config.yaml"):
            if not (directory / name).exists():
                raise ValueError(f"load_localize_runs: {directory} has no {name}.")
        anatomy_parts.append(pd.read_csv(directory / "anatomy.csv"))
        summary_parts.append(pd.read_csv(directory / "anatomy_summary.csv"))
        configs.append(dict(read_yaml(directory / "localize_config.yaml")))

    sources = {str(c.get("source")) for c in configs}
    if len(sources) > 1:
        raise ValueError(
            f"load_localize_runs: input runs disagree on source {sorted(sources)}. Concatenating "
            "a ground-truth localisation with a prediction-derived one produces a population "
            "that is partly the cohort's real anatomy and partly a model's opinion of it, with "
            "no column marking which is which."
        )

    splits = [str(c.get("split")) for c in configs]
    if len(set(splits)) != len(splits):
        raise ValueError(
            f"load_localize_runs: two input runs name the same split {splits}. The frozen splits "
            "are disjoint, so distinct splits concatenate to a cohort -- but the same split "
            "twice doubles every count while leaving every FRACTION unchanged, which no "
            "plausibility check on the output would catch."
        )

    anatomy = pd.concat(anatomy_parts, ignore_index=True)
    summary = pd.concat(summary_parts, ignore_index=True)

    duplicated = summary["case_id"].duplicated()
    if duplicated.any():
        offenders = sorted(summary.loc[duplicated, "case_id"].unique())[:5]
        raise ValueError(
            f"load_localize_runs: {int(duplicated.sum())} duplicated case_id(s) across the input "
            f"runs, e.g. {offenders}. Every case must appear exactly once or it is weighted twice."
        )

    logger.info(
        "Loaded %d localisation run(s): %d case(s), splits %s, source %s.",
        len(dirs),
        len(summary),
        splits,
        sorted(sources)[0],
    )
    return anatomy, summary, configs


def run_population_stats(cfg: DictConfig) -> Path:
    """Computes the population statistics, writes the tables, and renders the figures.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        The path of `population_structures.csv`.

    Raises:
        Whatever `load_localize_runs` raises (all config problems, checked
        before any work starts).
    """
    population_cfg = cfg.analysis.population
    dirs = [Path(str(d)) for d in population_cfg.localize_dirs]
    anatomy, summary, configs = load_localize_runs(dirs)

    region = str(population_cfg.region)
    out_dir = ensure_dir(cfg.output_dir)

    structures = structure_involvement_frequency(
        anatomy,
        region=region,
        min_frac_of_structure=float(population_cfg.min_frac_of_structure),
    )
    lobes = lobe_burden_distribution(anatomy, region=region)
    laterality = laterality_distribution(anatomy, region=region)
    eloquence = eloquence_rates(summary)

    structures_path = out_dir / "population_structures.csv"
    structures.to_csv(structures_path, index=False)
    lobes.to_csv(out_dir / "population_lobes.csv", index=False)
    laterality.to_csv(out_dir / "population_laterality.csv", index=False)
    write_json(eloquence, out_dir / "population_eloquence.json")

    # Surfaced at WARNING, not buried in the JSON: a field that is constant
    # across the whole cohort carries no per-case information and cannot
    # discriminate between anything, and a reader who sees 100% without that
    # context will read saturation as agreement.
    degenerate = eloquence.get("degenerate_fields") or []
    if degenerate:
        logger.warning(
            "Degenerate eloquence field(s) across this cohort: %s. These are constant for every "
            "case, so they carry no per-case information -- report them as saturated, never as "
            "a success rate.",
            ", ".join(str(f) for f in degenerate),
        )

    if population_cfg.figures:
        figure_dir = ensure_dir(Path(str(cfg.output_dir)) / "figures")
        with paper_style():
            save_figure(
                plot_structure_involvement(structures, top_n=int(population_cfg.top_n_structures)),
                figure_dir,
                "population_structure_involvement",
                close=True,
            )
            save_figure(
                plot_lobe_distribution(lobes),
                figure_dir,
                "population_lobe_distribution",
                close=True,
            )
        logger.info("Figures written to %s", figure_dir)

    record = OmegaConf.to_container(population_cfg, resolve=True)
    record["resolved_localize_dirs"] = [str(d.resolve()) for d in dirs]
    record["source"] = str(configs[0].get("source"))
    record["splits"] = [str(c.get("split")) for c in configs]
    record["n_cases"] = int(len(summary))
    record["atlas"] = configs[0].get("atlas")
    record["coverage_line"] = configs[0].get("coverage_line")
    write_yaml(record, out_dir / "population_config.yaml")

    return structures_path


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Compute cohort-level anatomy statistics from one or more localisation runs.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    structures_path = run_population_stats(cfg)
    print(f"Population structure table written to {structures_path}")


if __name__ == "__main__":
    main()
