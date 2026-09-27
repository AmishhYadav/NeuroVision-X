"""Hydra entry point: the two figures the semester report cites from Milestone 5.

Both figures are drawn from artifacts a prior, already-committed analysis run
wrote to disk -- this script runs no model and needs no GPU. It never
recomputes a conformal fit or a local-recalibration split; doing so would
silently re-derive a number the semester report is supposed to be quoting
verbatim.

1. `fig_local_recalibration.png` -- from
   `scripts/local_recalibration.py`'s `summary.csv` (docs/research_docs/experiments.md note
   52). A **counterfactual** ("what would a new site's own labelled cases
   buy back?"), never external validation -- see that note before reading
   the figure.
2. `fig_realised_vs_nominal.png` -- from `scripts/conformal.py`'s
   `realised_risk.csv` (docs/research_docs/experiments.md note 42): the conformal bound's
   realised miss rate at the frozen BraTS-val threshold, applied to each
   cohort, against the nominal alpha it was supposed to control.

Style and the `save_figure` idiom (PNG, `out_dir` created on demand) are
reused from `neurovision.visualization.figures` rather than reinvented here.

Example usage:

    python scripts/plot_semester_figures.py
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

# Must happen before pyplot is imported anywhere in this process -- the
# default backend tries to open a GUI window, which has no display in a
# headless test run or a batch script and either hangs or raises.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from neurovision.utils.logging import setup_logging  # noqa: E402
from neurovision.visualization.figures import REGION_COLORS, save_figure  # noqa: E402

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and
# on any machine -- no absolute paths. Same convention as scripts/replay_logits.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# The four (cohort, region) series fig_local_recalibration draws, in legend
# order. Test is deliberately excluded here -- the counterfactual question
# ("what would a new site need") is about the two cohorts the bound actually
# breaks on (note 42/52); test's own k=half control point is reported in the
# note's prose, not in this figure.
_LOCAL_RECAL_SERIES: tuple[tuple[str, str], ...] = (
    ("ssa", "WT"),
    ("ssa", "TC"),
    ("ped", "WT"),
    ("ped", "TC"),
)

# Linestyle AND marker per cohort; colour per region (REGION_COLORS, imported
# above) -- same "colour and linestyle both carry independent meaning" rule
# figures.py's module docstring states for MODEL_COLOR_CYCLE /
# MODEL_LINESTYLE_CYCLE, applied here to (cohort, region) instead of models.
# A marker is needed IN ADDITION to linestyle: SSA·WT and PED·WT share a
# colour (both WT) and differ only by linestyle, which is invisible on a
# series with a single point (nothing to draw a dash between).
_COHORT_LINESTYLE: dict[str, str] = {"ssa": "-", "ped": "--"}
_COHORT_MARKER_LOCAL_RECAL: dict[str, str] = {"ssa": "o", "ped": "s"}

_CAPTION_LOCAL_RECALIBRATION = (
    "Counterfactual, not external validation: each cohort is split at random 1000 times, "
    "k of its own cases refit the conformal threshold, and the realised miss rate is read on "
    "the held-out remainder. It answers 'what would a new site need to restore the bound', "
    "never 'this cohort's coverage'. A point marked with a black x has verdict NOT_RESTORED: "
    "the held-out risk misses alpha even though a local fit was feasible at that k. "
    "Source: docs/research_docs/experiments.md note 52."
)

# apply_dir -> cohort, keyed by the exact directory basename `scripts/conformal.py`
# wrote into outputs/conformal/neurovision/realised_risk.csv (note 42). An exact
# match, not a substring check -- CLAUDE.md's "any short token you match against
# a path is a substring of some longer word there" trap names this exact
# hazard (eval_test vs eval_test_baseline_unet3d, etc.).
_COHORT_BY_APPLY_DIR_BASENAME: dict[str, str] = {
    "eval_test": "test",
    "eval_ssa_neurovision": "ssa",
    "eval_ped_neurovision": "ped",
}

# Marker per cohort for fig_realised_vs_nominal, one region per panel so
# colour there is free to encode the region via REGION_COLORS instead.
_COHORT_MARKER: dict[str, str] = {"test": "o", "ssa": "s", "ped": "^"}


# --------------------------------------------------------------------------- #
# Figure 1 -- local recalibration
# --------------------------------------------------------------------------- #
def plot_local_recalibration(
    df: pd.DataFrame,
    *,
    model: str = "neurovision",
    alphas: Sequence[float] = (0.05, 0.10, 0.20),
    series: Sequence[tuple[str, str]] = _LOCAL_RECAL_SERIES,
) -> Figure:
    """Held-out miss rate vs local-calibration budget `k`, one panel per alpha.

    Reads `scripts/local_recalibration.py`'s `summary.csv` directly (see that
    script and `docs/research_docs/experiments.md` note 52). Filters to `model` and the
    requested `series` of `(cohort, region)` pairs -- deliberately NOT
    `is_half == False`: SSA's half point is k=30 and PED's is k=49, both of
    which sit exactly on the swept-k grid's high end and both of which note
    52 quotes by name (SSA·TC k=30 NOT_RESTORED; PED·WT k=49 RESTORED). An
    `is_half` filter would silently drop those two rows, and for SSA it drops
    the single point the note highlights. `series` already excludes the
    `"test"` cohort, whose `is_half` row is a control point reported in the
    note's prose rather than here, so no cohort needs an `is_half` filter at
    all.

    A series with a real, finite value at at least one `k` in a panel is
    drawn as a line with a shaded 95% CI band (`realised_risk_ci_low/high`),
    a marker per cohort, and a legend entry. Any point whose `verdict` is
    `NOT_RESTORED` gets a black 'x' overlay -- the local fit was feasible at
    that k but the held-out risk still missed alpha. A series that is
    `INFEASIBLE` at every swept `k` in that panel draws nothing but still
    gets a legend entry, suffixed `"(infeasible at every k)"`, so a reader
    can see it was considered.

    Args:
        df: The `summary.csv` table -- columns `cohort, region, alpha, k,
            mean_realised_risk, realised_risk_ci_low, realised_risk_ci_high,
            model, verdict` (and others, unused here).
        model: Which model's rows to draw.
        alphas: The alpha values to give one panel each, left to right.
        series: `(cohort, region)` pairs to draw, in legend order.

    Returns:
        A figure with `len(alphas)` panels.

    Raises:
        ValueError: `df` is empty after filtering to `model`, or a requested
            alpha has no rows at all (not even an infeasible one) for `model`.
    """
    filtered = df[df["model"] == model]
    if filtered.empty:
        raise ValueError(
            f"plot_local_recalibration: no rows for model={model!r} in the supplied summary "
            "table."
        )

    fig, axes = plt.subplots(1, len(alphas), figsize=(3.6 * len(alphas), 3.4), squeeze=False)
    axes = axes[0]

    for ax, alpha in zip(axes, alphas, strict=True):
        panel = filtered[np.isclose(filtered["alpha"].astype(float), alpha)]
        if panel.empty:
            raise ValueError(
                f"plot_local_recalibration: alpha={alpha} has no rows at all for model={model!r}."
            )

        for cohort, region in series:
            rows = panel[(panel["cohort"] == cohort) & (panel["region"] == region)].sort_values("k")
            duplicated = rows.duplicated(subset="k", keep="first")
            if duplicated.any():
                logger.warning(
                    "plot_local_recalibration: %s·%s at alpha=%s has %d duplicate k value(s) "
                    "(%s); keeping the first occurrence of each.",
                    cohort,
                    region,
                    alpha,
                    int(duplicated.sum()),
                    sorted(rows.loc[duplicated, "k"].unique().tolist()),
                )
                rows = rows.loc[~duplicated]

            base_label = f"{cohort.upper()}·{region}"
            risk = rows["mean_realised_risk"].to_numpy(dtype=float)
            if rows.empty or np.all(np.isnan(risk)):
                # Nothing to draw -- an invisible handle keeps the legend
                # entry so the reader can see this series was considered and
                # never restored the bound at any swept k, rather than
                # silently vanishing from the figure.
                ax.plot([], [], color=REGION_COLORS[region], linestyle=_COHORT_LINESTYLE[cohort])[
                    0
                ].set_label(f"{base_label} (infeasible at every k)")
                continue

            k_values = rows["k"].to_numpy(dtype=float)
            lo = rows["realised_risk_ci_low"].to_numpy(dtype=float)
            hi = rows["realised_risk_ci_high"].to_numpy(dtype=float)
            verdict = rows["verdict"].to_numpy(dtype=object)
            valid = np.isfinite(risk)
            ax.plot(
                k_values[valid],
                risk[valid],
                marker=_COHORT_MARKER_LOCAL_RECAL[cohort],
                markersize=4,
                color=REGION_COLORS[region],
                linestyle=_COHORT_LINESTYLE[cohort],
                label=base_label,
            )
            not_restored = valid & (verdict == "NOT_RESTORED")
            if not_restored.any():
                ax.scatter(
                    k_values[not_restored],
                    risk[not_restored],
                    marker="x",
                    s=45,
                    color="black",
                    linewidths=1.3,
                    zorder=5,
                )
            fill_valid = valid & np.isfinite(lo) & np.isfinite(hi)
            if fill_valid.any():
                ax.fill_between(
                    k_values[fill_valid],
                    lo[fill_valid],
                    hi[fill_valid],
                    color=REGION_COLORS[region],
                    alpha=0.15,
                    linewidth=0.0,
                )

        ax.axhline(alpha, color="#333333", linewidth=0.9, linestyle=(0, (2, 2)))
        ax.set_title(f"α = {alpha:g}")
        ax.set_xlabel("local calibration cases k")
        if ax is axes[0]:
            ax.set_ylabel("held-out miss rate (mean over splits)")
        ax.legend(loc="best", fontsize=6)

    fig.tight_layout()
    return fig


def save_local_recalibration_figure(fig: Figure, out_dir: str | Path, dpi: int) -> list[Path]:
    """Writes `fig_local_recalibration.png` and its caption `.txt` sidecar.

    Split out from `main` so it is testable without composing a Hydra config
    (the spec's "main-less save function").

    Args:
        fig: The figure built by `plot_local_recalibration`.
        out_dir: Destination directory, created if missing.
        dpi: Raster resolution for the PNG.

    Returns:
        The written paths: the PNG, then the caption `.txt`.
    """
    written = save_figure(
        fig, out_dir, "fig_local_recalibration", formats=("png",), dpi=dpi, close=True
    )
    caption_path = Path(out_dir) / "fig_local_recalibration.txt"
    caption_path.write_text(_CAPTION_LOCAL_RECALIBRATION + "\n", encoding="utf-8")
    logger.info("plot_semester_figures: wrote %s", caption_path)
    written.append(caption_path)
    return written


# --------------------------------------------------------------------------- #
# Figure 2 -- realised vs nominal conformal risk
# --------------------------------------------------------------------------- #
def cohort_from_apply_dir(apply_dir: str) -> str | None:
    """Maps a `realised_risk.csv` `apply_dir` value to its cohort name.

    Matches the directory's exact basename against
    `_COHORT_BY_APPLY_DIR_BASENAME` -- never a substring check. CLAUDE.md
    records that `"eval_test" in name` is also true for
    `"eval_test_baseline_unet3d"`, which is a different run entirely; an
    exact-basename match cannot make that mistake.

    Args:
        apply_dir: A path such as `outputs/neurovision/eval_test` or
            `outputs/eval_ssa_neurovision`.

    Returns:
        The cohort name (`"test"`, `"ssa"` or `"ped"`), or `None` if the
        basename is not recognized.
    """
    basename = Path(apply_dir).name
    cohort = _COHORT_BY_APPLY_DIR_BASENAME.get(basename)
    if cohort is None:
        logger.warning(
            "plot_semester_figures: apply_dir %r has unrecognized basename %r; dropping its "
            "row(s) from fig_realised_vs_nominal.",
            apply_dir,
            basename,
        )
    return cohort


def plot_realised_vs_nominal(
    df: pd.DataFrame,
    *,
    regions: Sequence[str] = ("WT", "TC"),
    cohort_order: Sequence[str] = ("test", "ssa", "ped"),
) -> Figure:
    """Realised conformal miss rate vs the nominal alpha it was meant to control.

    Reads a table already carrying a `cohort` column (see `cohort_from_apply_dir`
    for how `main` derives it from `realised_risk.csv`'s `apply_dir`). One panel
    per region; a point above the `y = x` reference line is a violation of the
    nominal guarantee (docs/research_docs/experiments.md note 42).

    Args:
        df: Columns `cohort, region, alpha, mean_miss_rate` (realised risk).
        regions: Regions to give one panel each, left to right.
        cohort_order: Legend/marker order.

    Returns:
        A figure with `len(regions)` panels.

    Raises:
        ValueError: `df` is empty, or a requested region has no rows.
    """
    if df.empty:
        raise ValueError("plot_realised_vs_nominal: the supplied table is empty.")

    fig, axes = plt.subplots(1, len(regions), figsize=(3.6 * len(regions), 3.4), squeeze=False)
    axes = axes[0]

    for ax, region in zip(axes, regions, strict=True):
        panel = df[df["region"] == region]
        if panel.empty:
            raise ValueError(f"plot_realised_vs_nominal: region={region!r} has no rows.")

        alpha_max = float(panel["alpha"].max())
        risk_max = float(panel["mean_miss_rate"].max())
        span_hi = max(alpha_max, risk_max) * 1.1
        ax.plot(
            [0.0, span_hi],
            [0.0, span_hi],
            color="#666666",
            linewidth=0.8,
            linestyle=(0, (2, 2)),
            label="y = x (nominal met exactly)",
            zorder=1,
        )

        for cohort in cohort_order:
            rows = panel[panel["cohort"] == cohort].sort_values("alpha")
            if rows.empty:
                continue
            ax.plot(
                rows["alpha"].to_numpy(dtype=float),
                rows["mean_miss_rate"].to_numpy(dtype=float),
                marker=_COHORT_MARKER.get(cohort, "o"),
                markersize=6,
                linestyle="none",
                color=REGION_COLORS[region],
                label=cohort.upper(),
                zorder=2,
            )

        ax.set_title(region)
        ax.set_xlabel("nominal α")
        if ax is axes[0]:
            ax.set_ylabel("realised risk (mean miss rate)")
        ax.set_xlim(0.0, span_hi)
        ax.set_ylim(0.0, span_hi)
        ax.legend(loc="best", fontsize=6)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Hydra driver
# --------------------------------------------------------------------------- #
def _load_realised_risk_with_cohort(path: Path) -> pd.DataFrame | None:
    """Reads `realised_risk.csv` and attaches a `cohort` column, or `None` if unreadable.

    Args:
        path: Expected location of `realised_risk.csv` (note 42).

    Returns:
        The table with an added `cohort` column, rows with an unrecognized
        `apply_dir` dropped -- or `None` if the file does not exist.
    """
    if not path.is_file():
        logger.warning(
            "plot_semester_figures: %s not found; skipping fig_realised_vs_nominal. This "
            "figure is only produced when scripts/conformal.py has already been run for the "
            "model named in cfg.analysis.semester_figures.conformal_dir.",
            path,
        )
        return None

    table = pd.read_csv(path)
    table["cohort"] = table["apply_dir"].map(cohort_from_apply_dir)
    n_dropped = int(table["cohort"].isna().sum())
    if n_dropped:
        logger.warning(
            "plot_semester_figures: dropped %d row(s) of %s with an unrecognized apply_dir.",
            n_dropped,
            path,
        )
    return table.dropna(subset=["cohort"])


def run_plot_semester_figures(cfg: DictConfig) -> dict[str, list[Path]]:
    """Builds and saves the semester report's Milestone 5 figures.

    Args:
        cfg: The full composed Hydra config; reads `cfg.analysis.semester_figures`.

    Returns:
        Figure stem -> list of paths written, for whichever figures had a
        readable input table. `fig_realised_vs_nominal` is absent from the
        dict (with a logged warning, not an exception) when
        `realised_risk.csv` is not on disk.
    """
    figures_cfg = cfg.analysis.semester_figures
    out_dir = Path(str(figures_cfg.out_dir))
    dpi = int(figures_cfg.dpi)

    written: dict[str, list[Path]] = {}

    local_recal_path = Path(str(figures_cfg.local_recalibration_summary))
    logger.info("plot_semester_figures: reading %s", local_recal_path)
    summary = pd.read_csv(local_recal_path)
    fig1 = plot_local_recalibration(summary)
    written["fig_local_recalibration"] = save_local_recalibration_figure(fig1, out_dir, dpi)

    realised_risk_path = Path(str(figures_cfg.conformal_dir)) / "realised_risk.csv"
    realised_risk = _load_realised_risk_with_cohort(realised_risk_path)
    if realised_risk is not None:
        fig2 = plot_realised_vs_nominal(realised_risk)
        written["fig_realised_vs_nominal"] = save_figure(
            fig2, out_dir, "fig_realised_vs_nominal", formats=("png",), dpi=dpi, close=True
        )

    _log_summary(written)
    return written


def _log_summary(written: Mapping[str, list[Path]]) -> None:
    """Logs and prints a compact end-of-run summary, matching the other analysis drivers."""
    lines = ["=" * 70, "Semester figures summary", "=" * 70]
    for stem in ("fig_local_recalibration", "fig_realised_vs_nominal"):
        if stem in written:
            lines.append(f"  {stem}: {[str(p) for p in written[stem]]}")
        else:
            lines.append(f"  {stem}: skipped (input table not found)")
    print("\n".join(lines))


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Builds the semester report figures per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    run_plot_semester_figures(cfg)


if __name__ == "__main__":
    main()
