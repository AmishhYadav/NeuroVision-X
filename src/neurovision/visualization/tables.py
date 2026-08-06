"""Paper tables, built from result files and rendered to Markdown and LaTeX.

Companion to `figures.py`: same job, different medium. Everything the paper
prints as a table is produced here, so that regenerating the paper is one
notebook run and no number is ever retyped. A number typed by hand into a LaTeX
table is a number nobody can trace back to a run.

Two stages, deliberately separate:

1. `build_results_table` / `compare_models` (in `analysis.statistics`) produce a
   tidy `DataFrame` -- the DATA.
2. `format_*` turn a table into a string -- the PRESENTATION.

Keeping them apart means the rounding, the bolding and the LaTeX escaping are all
testable without touching a metric, and the same data can be rendered twice
without recomputing anything.

Dependency-light: numpy, pandas and `analysis.statistics` only. No torch, no
matplotlib.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neurovision.analysis.statistics import metric_direction

logger = logging.getLogger(__name__)

# Reporting order: outermost to innermost, as BraTS papers write it. Duplicated
# from `figures.REGION_ORDER` on purpose -- importing it here would drag
# matplotlib into a module that is otherwise pure pandas, and a table should be
# renderable in an environment with no plotting stack at all.
REGION_ORDER: tuple[str, str, str] = ("WT", "TC", "ET")

# Decimal places per metric prefix. Dice differences that matter in this
# literature live in the third decimal, so four places is the honest minimum;
# HD95 is in millimetres, where a third decimal is noise dressed up as
# precision.
DEFAULT_PRECISION: dict[str, int] = {
    "dice": 4,
    "iou": 4,
    "ece": 4,
    "mce": 4,
    "brier": 4,
    "aurc": 4,
    "hd95": 2,
}
FALLBACK_PRECISION: int = 3

_TIDY_COLUMNS: tuple[str, ...] = (
    "model",
    "region",
    "metric",
    "mean",
    "std",
    "median",
    "n",
    "n_missing",
    "gt_empty_frac",
)

_LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX's special characters in a piece of display text.

    Model labels routinely contain an underscore (`baseline_unet3d`), which
    LaTeX reads as subscript-start and which then either errors or silently
    renders the rest of the cell as a subscript. This is the single most common
    way a generated table breaks a build.

    Args:
        text: Raw display text.

    Returns:
        The escaped text, safe to place in a LaTeX cell.
    """
    # ONE pass over the characters, not a sequence of `str.replace` calls. The
    # replacements themselves contain special characters -- `\` becomes
    # `\textbackslash{}` -- so a second pass would escape the braces that the
    # first pass just emitted, turning a lone backslash into
    # `\textbackslash\{\}`. Pinned by a test.
    return "".join(_LATEX_ESCAPES.get(char, char) for char in text)


def _precision_for(metric: str, precision: Mapping[str, int] | int | None) -> int:
    """Decimal places for a metric, from an override, the table, or the fallback."""
    if isinstance(precision, int):
        return precision
    if precision is not None and metric in precision:
        return precision[metric]
    return DEFAULT_PRECISION.get(metric.lower(), FALLBACK_PRECISION)


def _fmt(value: float, places: int) -> str:
    """Format one number, rendering NaN as an em dash rather than the string 'nan'."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{places}f}"


def build_results_table(
    per_case: Mapping[str, pd.DataFrame],
    *,
    regions: Sequence[str] = REGION_ORDER,
    metrics: Sequence[str] = ("dice", "hd95"),
    strict: bool = True,
) -> pd.DataFrame:
    """Aggregate per-case metric tables into one tidy results table.

    Reports mean, std AND median. The median is not decoration: per-case Dice
    distributions here are strongly left-skewed, so the mean alone understates
    the typical case while hiding how bad the failures are. A paper that prints
    only the mean is describing a distribution it never showed the reader.

    `n_missing` counts cases where the metric was NaN. For HD95 that is a real
    quantity -- the distance is genuinely undefined when exactly one side of a
    region is empty -- and those cases are excluded from the mean rather than
    given an arbitrary penalty, so the count has to travel with the number.

    Args:
        per_case: Model display label -> the `per_case_metrics.csv` table from
            `scripts/evaluate.py`, indexed by `case_id`. Insertion order is
            preserved and becomes the report order.
        regions: Regions to report, in reporting order.
        metrics: Metric prefixes; the column read is `f"{metric}_{region}"`.
        strict: Raise when a required column is missing. With `strict=False` the
            row is emitted with NaN statistics and a warning instead -- useful
            while a run is still in progress, never for a table going in a paper.

    Returns:
        A tidy `DataFrame` with columns `model, region, metric, mean, std,
        median, n, n_missing, gt_empty_frac`. `gt_empty_frac` is NaN when the
        source table has no `gt_empty_<region>` column.

    Raises:
        ValueError: `per_case`, `regions` or `metrics` is empty, or (under
            `strict`) a required column is missing from one of the tables.
    """
    if not per_case:
        raise ValueError("build_results_table: `per_case` must contain at least one model.")
    if not regions:
        raise ValueError("build_results_table: `regions` must contain at least one region.")
    if not metrics:
        raise ValueError("build_results_table: `metrics` must contain at least one metric.")

    rows: list[dict[str, object]] = []
    for model, table in per_case.items():
        for region in regions:
            empty_column = f"gt_empty_{region}"
            gt_empty = (
                float(table[empty_column].mean()) if empty_column in table.columns else float("nan")
            )
            for metric in metrics:
                column = f"{metric}_{region}"
                if column not in table.columns:
                    message = (
                        f"build_results_table: model {model!r} has no column {column!r}. "
                        f"Available: {sorted(table.columns)}."
                    )
                    if strict:
                        raise ValueError(message)
                    logger.warning("%s Emitting NaN statistics for this cell.", message)
                    values = np.array([], dtype=float)
                    total = 0
                else:
                    raw = table[column].to_numpy(dtype=float)
                    total = int(raw.size)
                    values = raw[np.isfinite(raw)]

                rows.append(
                    {
                        "model": model,
                        "region": region,
                        "metric": metric,
                        # ddof=1: the sample standard deviation, matching what
                        # pandas' `.std()` reports and therefore what the
                        # existing baseline notebook printed. A silent switch to
                        # the population std would shift every +/- in the paper.
                        "mean": float(values.mean()) if values.size else float("nan"),
                        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                        "median": float(np.median(values)) if values.size else float("nan"),
                        "n": total,
                        "n_missing": total - int(values.size),
                        "gt_empty_frac": gt_empty,
                    }
                )

    return pd.DataFrame(rows, columns=list(_TIDY_COLUMNS))


def _validate_tidy(table: pd.DataFrame, func_name: str) -> None:
    """Raise unless `table` looks like `build_results_table` output."""
    missing = sorted(set(_TIDY_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(
            f"{func_name}: table is missing column(s) {missing}; expected the output of "
            "build_results_table."
        )
    if table.empty:
        raise ValueError(f"{func_name}: table is empty.")


def _best_models(table: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Which model wins each (region, metric) cell, by the metric's own direction.

    Direction comes from `analysis.statistics.metric_direction`, which RAISES on
    an unrecognized metric rather than assuming higher-is-better. Bolding the
    wrong end of an HD95 or ECE column would invert a paper claim with nothing
    failing anywhere, so guessing is not an option.
    """
    winners: dict[tuple[str, str], str] = {}
    if table["model"].nunique() < 2:
        # "Best per column" with one model bolds every cell, which reads as an
        # emphasis the table has not earned. Nothing to compare, nothing to bold.
        return winners
    for (region, metric), group in table.groupby(["region", "metric"], sort=False):
        higher_is_better = metric_direction(f"{metric}_{region}")
        means = group.set_index("model")["mean"]
        means = means[np.isfinite(means)]
        if means.empty:
            continue
        winners[(region, metric)] = str(means.idxmax() if higher_is_better else means.idxmin())
    return winners


def _cell(
    row: pd.Series,
    *,
    places: int,
    show_median: bool,
    bold: bool,
    emphasis: str,
) -> str:
    """Render one `mean +/- std` cell, optionally with the median and emphasis."""
    body = f"{_fmt(float(row['mean']), places)} ± {_fmt(float(row['std']), places)}"
    if show_median:
        body = f"{body} ({_fmt(float(row['median']), places)})"
    if bold and emphasis:
        return f"{emphasis}{body}{emphasis}"
    return body


def format_results_markdown(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    precision: Mapping[str, int] | int | None = None,
    show_median: bool = True,
    highlight_best: bool = True,
) -> str:
    """Render a tidy results table as Markdown, rows = models, columns = region x metric.

    Args:
        table: Output of `build_results_table`.
        caption: Optional line printed above the table.
        precision: Decimal places, as an int for every metric or a mapping keyed
            by metric name. `None` uses `DEFAULT_PRECISION`.
        show_median: Print the median in parentheses after `mean ± std`.
        highlight_best: Bold the winning model in each column.

    Returns:
        The Markdown text, ending without a trailing newline.

    Raises:
        ValueError: `table` is not a `build_results_table` output, or is empty.
    """
    _validate_tidy(table, "format_results_markdown")

    models = list(dict.fromkeys(table["model"]))
    regions = list(dict.fromkeys(table["region"]))
    metrics = list(dict.fromkeys(table["metric"]))
    winners = _best_models(table) if highlight_best else {}

    header = ["Model"]
    for region in regions:
        for metric in metrics:
            header.append(f"{region} {metric}")

    lines: list[str] = []
    if caption:
        lines.extend([caption, ""])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    indexed = table.set_index(["model", "region", "metric"])
    for model in models:
        cells = [model]
        for region in regions:
            for metric in metrics:
                key = (model, region, metric)
                if key not in indexed.index:
                    cells.append("--")
                    continue
                cells.append(
                    _cell(
                        indexed.loc[key],
                        places=_precision_for(metric, precision),
                        show_median=show_median,
                        bold=winners.get((region, metric)) == model,
                        emphasis="**",
                    )
                )
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(_footnote(table, show_median))
    return "\n".join(lines)


def _footnote(table: pd.DataFrame, show_median: bool) -> str:
    """The caveats that must travel with every results table in this project.

    Built once and rendered by BOTH the Markdown and the LaTeX formatter. The
    LaTeX one is the version that goes in the paper, so it is the one that most
    needs the caveats -- an earlier version attached them only to the Markdown
    output, which meant the compiled PDF silently dropped the `ignore_empty`
    convention and the excluded-case counts.
    """
    parts = ["Reported as mean +/- std over the held-out cases"]
    if show_median:
        parts[0] += ", median in parentheses"
    parts[0] += "."

    missing = table[table["n_missing"] > 0]
    if not missing.empty:
        detail = ", ".join(
            f"{row['model']} {row['region']} {row['metric']}: "
            f"{int(row['n_missing'])}/{int(row['n'])}"
            for _, row in missing.iterrows()
        )
        parts.append(
            "Cases where the metric was undefined are excluded from the mean rather than given "
            f"an arbitrary penalty ({detail})."
        )

    # Surfaces `gt_empty_frac`, which is otherwise carried in the tidy table and
    # never rendered. The size of the ignore_empty effect depends entirely on
    # this fraction, so stating the convention without it is half a caveat.
    empty = table[table["gt_empty_frac"] > 0]
    if not empty.empty:
        detail = ", ".join(
            f"{region} {100.0 * frac:.1f}%"
            for region, frac in empty.groupby("region", sort=False)["gt_empty_frac"].first().items()
        )
        parts.append(f"Cases with an empty ground-truth region: {detail}.")

    parts.append(
        "Metrics use ignore_empty=False (the BraTS convention): a region absent from the "
        "ground truth scores Dice 1.0 if the prediction is also empty."
    )
    return " ".join(parts)


def _latex_note(text: str) -> list[str]:
    """Render a caveat below a LaTeX table, wrapped to the table's width.

    A `\\multicolumn` cell does not line-break, so a two-sentence caveat placed
    in one would run off the page. `\\parbox` wraps. Deliberately NOT
    `threeparttable`, which would add a package to the paper's preamble that
    this project has no other reason to require.
    """
    return [
        "\\vspace{2pt}",
        f"\\parbox{{\\linewidth}}{{\\footnotesize {escape_latex(text)}}}",
    ]


def format_results_latex(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    precision: Mapping[str, int] | int | None = None,
    show_median: bool = False,
    highlight_best: bool = True,
) -> str:
    """Render a tidy results table as a booktabs LaTeX table.

    Requires `\\usepackage{booktabs}` in the document preamble.

    `show_median` defaults to False here but True in the Markdown renderer, on
    purpose: the Markdown version is for reading during analysis, where more
    numbers help, while the LaTeX version goes in a page-limited paper, where
    `mean ± std (median)` across six columns does not fit.

    Args:
        table: Output of `build_results_table`.
        caption: Table caption.
        label: LaTeX label, used as `\\label{<label>}`.
        precision: See `format_results_markdown`.
        show_median: Print the median in parentheses.
        highlight_best: Wrap the winning model's cell in `\\textbf{}`.

    Returns:
        The LaTeX source, ending without a trailing newline.

    Raises:
        ValueError: `table` is not a `build_results_table` output, or is empty.
    """
    _validate_tidy(table, "format_results_latex")

    models = list(dict.fromkeys(table["model"]))
    regions = list(dict.fromkeys(table["region"]))
    metrics = list(dict.fromkeys(table["metric"]))
    winners = _best_models(table) if highlight_best else {}
    n_metric_columns = len(regions) * len(metrics)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * n_metric_columns + "}",
        "\\toprule",
    ]

    # Region group header, then a metric header underneath it.
    group_header = ["Model"]
    rules: list[str] = []
    for index, region in enumerate(regions):
        group_header.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{escape_latex(region)}}}")
        start = 2 + index * len(metrics)
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(metrics) - 1}}}")
    lines.append(" & ".join(group_header) + " \\\\")
    lines.append(" ".join(rules))
    lines.append(" & ".join([""] + [escape_latex(m) for m in metrics] * len(regions)) + " \\\\")
    lines.append("\\midrule")

    indexed = table.set_index(["model", "region", "metric"])
    for model in models:
        cells = [escape_latex(model)]
        for region in regions:
            for metric in metrics:
                key = (model, region, metric)
                if key not in indexed.index:
                    cells.append("--")
                    continue
                body = _cell(
                    indexed.loc[key],
                    places=_precision_for(metric, precision),
                    show_median=show_median,
                    bold=False,
                    emphasis="",
                )
                # `$\pm$` rather than the literal character: a raw U+00B1 in a
                # .tex file only compiles under a UTF-8-aware engine, and the
                # failure is an unhelpful "Package inputenc Error".
                body = body.replace("±", "$\\pm$")
                if winners.get((region, metric)) == model:
                    body = f"\\textbf{{{body}}}"
                cells.append(body)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend(_latex_note(_footnote(table, show_median)))
    lines.append("\\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Statistical comparison
# --------------------------------------------------------------------------- #
_COMPARISON_REQUIRED: frozenset[str] = frozenset(
    {"n", "mean_diff", "improvement", "improvement_lo", "improvement_hi", "p_holm", "verdict"}
)


def _validate_comparison(table: pd.DataFrame, func_name: str) -> None:
    """Raise unless `table` looks like `compare_models` output."""
    missing = sorted(_COMPARISON_REQUIRED - set(table.columns))
    if missing:
        raise ValueError(
            f"{func_name}: table is missing column(s) {missing}; expected the output of "
            "analysis.statistics.compare_models."
        )
    if table.empty:
        raise ValueError(f"{func_name}: table is empty.")


def _comparison_footnote(name_a: str) -> str:
    """The caveats that must travel with every comparison table.

    Shared by the Markdown and LaTeX renderers so the two cannot drift. Both
    `inconclusive` AND `negligible` are named: `compare_models` treats them as
    two different reasons not to claim a difference, and a footnote that warns
    about only one of them reads as permission to claim the other.
    """
    return (
        f"Positive improvement always means {name_a} is better, regardless of whether the "
        "metric itself is higher- or lower-is-better. Neither an inconclusive verdict (the "
        "bootstrap CI contains zero, or the Holm-adjusted p exceeds alpha) nor a negligible "
        "one (conclusive, but the entire CI sits inside the practical-equivalence margin) may "
        "be claimed as a difference. n is the paired-complete case count and n_missing the "
        "cases dropped for a NaN on either side. The Holm family is the whole table -- "
        "re-running on a subset after seeing the p-values would destroy the error-rate "
        "guarantee."
    )


def _p_value(value: float) -> str:
    """Format a p-value, floored rather than rounded to zero.

    `p = 0.000` is a claim no finite sample supports, and a reviewer will say so.
    """
    if not np.isfinite(value):
        return "--"
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def format_comparison_markdown(
    table: pd.DataFrame,
    *,
    name_a: str = "model",
    name_b: str = "baseline",
    caption: str | None = None,
    precision: Mapping[str, int] | int | None = None,
) -> str:
    """Render a `compare_models` table as Markdown.

    Prints the effect, its confidence interval, the Holm-adjusted p-value and the
    verdict together. All four are needed: a CI that excludes zero and a
    non-significant p-value disagree, and that disagreement is itself a finding
    -- which is why `verdict` resolves it conservatively rather than letting the
    reader pick whichever column suits.

    Args:
        table: Output of `compare_models`, indexed by metric.
        name_a: Display name of model A. Must match what `compare_models` used,
            since the per-model mean columns are named `mean_<name>`.
        name_b: Display name of model B.
        caption: Optional line printed above the table.
        precision: See `format_results_markdown`; applied to the mean and effect
            columns, keyed by the metric's prefix.

    Returns:
        The Markdown text.

    Raises:
        ValueError: `table` is not a `compare_models` output, or is empty.
    """
    _validate_comparison(table, "format_comparison_markdown")

    mean_a_column = f"mean_{name_a}"
    mean_b_column = f"mean_{name_b}"
    has_means = mean_a_column in table.columns and mean_b_column in table.columns
    if not has_means:
        logger.warning(
            "format_comparison_markdown: no %r/%r columns; `name_a`/`name_b` must match the names "
            "passed to compare_models. Omitting the per-model mean columns.",
            mean_a_column,
            mean_b_column,
        )

    header = ["Metric"]
    if has_means:
        header += [name_a, name_b]
    header += ["Improvement", "95% CI", "p (Holm)", "Effect (g)", "n", "n missing", "Verdict"]

    lines: list[str] = []
    if caption:
        lines.extend([caption, ""])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for metric, row in table.iterrows():
        places = _precision_for(str(metric).split("_")[0], precision)
        cells = [str(metric)]
        if has_means:
            cells += [
                _fmt(float(row[mean_a_column]), places),
                _fmt(float(row[mean_b_column]), places),
            ]
        cells.append(_fmt(float(row["improvement"]), places))
        cells.append(
            f"[{_fmt(float(row['improvement_lo']), places)}, "
            f"{_fmt(float(row['improvement_hi']), places)}]"
        )
        cells.append(_p_value(float(row["p_holm"])))
        cells.append(_fmt(float(row["hedges_g"]), 2) if "hedges_g" in table.columns else "--")
        cells.append(str(int(row["n"])))
        cells.append(str(int(row["n_missing"])) if "n_missing" in table.columns else "--")
        cells.append(str(row["verdict"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(_comparison_footnote(f"**{name_a}**"))
    return "\n".join(lines)


def format_comparison_latex(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    name_a: str = "model",
    name_b: str = "baseline",
    precision: Mapping[str, int] | int | None = None,
) -> str:
    """Render a `compare_models` table as a booktabs LaTeX table.

    Args:
        table: Output of `compare_models`, indexed by metric.
        caption: Table caption.
        label: LaTeX label.
        name_a: Display name of model A (see `format_comparison_markdown`).
        name_b: Display name of model B.
        precision: See `format_results_markdown`.

    Returns:
        The LaTeX source.

    Raises:
        ValueError: `table` is not a `compare_models` output, or is empty.
    """
    _validate_comparison(table, "format_comparison_latex")

    mean_a_column = f"mean_{name_a}"
    mean_b_column = f"mean_{name_b}"
    has_means = mean_a_column in table.columns and mean_b_column in table.columns

    header = ["Metric"]
    if has_means:
        header += [escape_latex(name_a), escape_latex(name_b)]
    header += [
        "$\\Delta$",
        "95\\% CI",
        "$p_{\\mathrm{Holm}}$",
        "$g$",
        "$n$",
        "$n_{\\mathrm{miss}}$",
        "Verdict",
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]

    for metric, row in table.iterrows():
        places = _precision_for(str(metric).split("_")[0], precision)
        cells = [escape_latex(str(metric))]
        if has_means:
            cells += [
                _fmt(float(row[mean_a_column]), places),
                _fmt(float(row[mean_b_column]), places),
            ]
        cells.append(_fmt(float(row["improvement"]), places))
        cells.append(
            f"[{_fmt(float(row['improvement_lo']), places)}, "
            f"{_fmt(float(row['improvement_hi']), places)}]"
        )
        cells.append(_p_value(float(row["p_holm"])).replace("<", "$<$"))
        cells.append(_fmt(float(row["hedges_g"]), 2) if "hedges_g" in table.columns else "--")
        cells.append(str(int(row["n"])))
        cells.append(str(int(row["n_missing"])) if "n_missing" in table.columns else "--")
        cells.append(escape_latex(str(row["verdict"])))
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend(_latex_note(_comparison_footnote(name_a)))
    lines.append("\\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Boundary-stratified tables
# --------------------------------------------------------------------------- #
# `scripts/evaluate.py` writes columns named `f"{metric}_{region}_{band}"`
# (e.g. `berr_ET_0-2`) where `band` is produced by
# `neurovision.metrics.boundary.band_label`. That module is NOT imported here
# -- it pulls in scipy and torch, and this module's whole reason to exist is
# to stay renderable with no plotting stack or torch installed. Band labels
# are instead re-derived from the column names themselves.
_VALID_BOUNDARY_METRICS: frozenset[str] = frozenset({"berr", "bfnr", "bfpr"})

_BOUNDARY_TIDY_COLUMNS: tuple[str, ...] = (
    "model",
    "region",
    "band",
    "metric",
    "mean",
    "std",
    "median",
    "n",
    "n_missing",
    "mean_voxels",
)


def _band_lower_edge(label: str) -> float:
    """Parses the lower (inclusive) edge out of a `band_label` string.

    Splits on the LAST `-`, not the first: a signed band such as `"-inf-0"`
    has its own leading `-`, and splitting on the first `-` would cut the
    label in the wrong place (`"-inf-0"` -> `("", "inf-0")` instead of
    `("-inf", "0")`). `float()` parses `"inf"` / `"-inf"` natively, so no
    special-casing of infinities is needed beyond that.

    Args:
        label: A `band_label` output, e.g. `"0-2"`, `"10-inf"`, `"-inf-0"`.

    Returns:
        The lower edge, as a float (`-inf` is a legal result).

    Raises:
        ValueError: `label` does not split into two floats on its last `-`.
    """
    parts = label.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"build_boundary_table: cannot parse band label {label!r} as 'LO-HI'; found no '-' "
            "separating the two edges."
        )
    try:
        lo = float(parts[0])
        float(parts[1])  # validated but unused -- only the lower edge sorts the table
    except ValueError as exc:
        raise ValueError(
            f"build_boundary_table: cannot parse band label {label!r} as 'LO-HI' -- "
            f"{parts[0]!r} and/or {parts[1]!r} is not a number."
        ) from exc
    return lo


def build_boundary_table(
    per_case: Mapping[str, pd.DataFrame],
    *,
    metric: str = "berr",
    regions: Sequence[str] | None = None,
    bands: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregates per-case boundary-stratified error columns into a tidy table.

    This is the quantitative form of the project's boundary-accuracy claim --
    not "our HD95 is lower" but "our error rate in the 0-2 mm shell around the
    true margin is lower". Companion to `build_results_table`, one level more
    specific: instead of one number per (model, region, metric) it reports one
    per (model, region, distance-from-boundary band).

    Args:
        per_case: Model display label -> that model's `per_case_metrics.csv`
            (as written by `scripts/evaluate.py`), indexed by case id.
            Insertion order is preserved and becomes the column order in the
            rendered table.
        metric: One of `"berr"` (total error rate), `"bfnr"` (under-
            segmentation / missed tumour) or `"bfpr"` (over-segmentation /
            spurious tumour). The column read is `f"{metric}_{region}_{band}"`.
        regions: Regions to report, in reporting order. Defaults to
            `REGION_ORDER`.
        bands: Explicit band labels, in the order they should appear. `None`
            discovers them from the column names present across `per_case`
            and sorts by numeric lower edge -- a plain string sort would put
            `"10-inf"` before `"2-5"` and silently reverse the table's
            reading order.

    Returns:
        A tidy `DataFrame`, one row per (model, region, band), with columns
        `model, region, band, metric, mean, std, median, n, n_missing,
        mean_voxels`. `mean`/`std` (sample std, ddof=1)/`median` are computed
        over cases, skipping any case where that band contained zero voxels
        (rate is NaN by construction there). `n` is the TOTAL case count and
        `n_missing` the number of those skipped as NaN -- the same convention
        as `build_results_table`, deliberately, since one column name meaning
        two different things across two builders in one module is a silently
        wrong number in a paper table. A band routinely has zero voxels -- the
        outermost band on a small tumour, or every band of a region absent
        from the ground truth -- so `n_missing` is not decoration; a rate
        averaged over 140 of 189 cases is a different number from the one a
        reader assumes. `mean_voxels` is the mean of the
        matching `bn_<region>_<band>` column when present, NaN otherwise; it
        is what tells a reader whether a band's rate was averaged over
        thousands of voxels per case or a handful.

    Raises:
        ValueError: `metric` is not one of `"berr"`/`"bfnr"`/`"bfpr"`,
            `per_case` is empty, a band label cannot be parsed as `"LO-HI"`,
            or no column in any table matches the requested metric for any
            requested region.
    """
    if metric not in _VALID_BOUNDARY_METRICS:
        raise ValueError(
            f"build_boundary_table: metric must be one of {sorted(_VALID_BOUNDARY_METRICS)}, "
            f"got {metric!r}."
        )
    if not per_case:
        raise ValueError("build_boundary_table: `per_case` must contain at least one model.")

    resolved_regions = list(regions) if regions is not None else list(REGION_ORDER)

    # Discover which `f"{metric}_{region}_{band}"` columns actually exist,
    # across every model's table, regardless of whether `bands` was given
    # explicitly -- a caller-supplied band that matches nothing anywhere
    # would otherwise silently produce an all-NaN row rather than a raise.
    discovered: set[str] = set()
    sample_columns: list[str] = []
    for table in per_case.values():
        sample_columns.extend(table.columns)
        for region in resolved_regions:
            prefix = f"{metric}_{region}_"
            for column in table.columns:
                if column.startswith(prefix):
                    discovered.add(column[len(prefix) :])

    if not discovered:
        raise ValueError(
            f"build_boundary_table: no column matched metric {metric!r} for region(s) "
            f"{resolved_regions}. Columns present (sample): "
            f"{sorted(set(sample_columns))[:10]}."
        )

    resolved_bands = sorted(discovered, key=_band_lower_edge) if bands is None else list(bands)

    rows: list[dict[str, object]] = []
    for model, table in per_case.items():
        for region in resolved_regions:
            for band in resolved_bands:
                column = f"{metric}_{region}_{band}"
                if column in table.columns:
                    raw = table[column].to_numpy(dtype=float)
                else:
                    raw = np.array([], dtype=float)
                values = raw[np.isfinite(raw)]

                voxel_column = f"bn_{region}_{band}"
                if voxel_column in table.columns:
                    voxels = table[voxel_column].to_numpy(dtype=float)
                    voxels = voxels[np.isfinite(voxels)]
                    mean_voxels = float(voxels.mean()) if voxels.size else float("nan")
                else:
                    mean_voxels = float("nan")

                rows.append(
                    {
                        "model": model,
                        "region": region,
                        "band": band,
                        "metric": metric,
                        "mean": float(values.mean()) if values.size else float("nan"),
                        "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                        "median": float(np.median(values)) if values.size else float("nan"),
                        # `n` is the TOTAL case count and `n_missing` the NaN
                        # count within it, exactly matching build_results_table
                        # above. The two builders live in one module and their
                        # output lands in one paper; `n` meaning "total" in one
                        # table and "contributing" in the other is a silently
                        # wrong number waiting to happen.
                        "n": int(raw.size),
                        "n_missing": int(raw.size - values.size),
                        "mean_voxels": mean_voxels,
                    }
                )

    return pd.DataFrame(rows, columns=list(_BOUNDARY_TIDY_COLUMNS))


def _validate_boundary_tidy(table: pd.DataFrame, func_name: str) -> None:
    """Raise unless `table` looks like `build_boundary_table` output."""
    missing = sorted(set(_BOUNDARY_TIDY_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(
            f"{func_name}: table is missing column(s) {missing}; expected the output of "
            "build_boundary_table."
        )
    if table.empty:
        raise ValueError(f"{func_name}: table is empty.")


def _boundary_best_models(table: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Which model wins each (region, band) cell.

    Unlike `_best_models`, the direction is hardcoded rather than looked up
    through `metric_direction` -- `berr`/`bfnr`/`bfpr` are error RATES, and
    every one of them is lower-is-better, by definition, for the whole
    family. `metric_direction` does not know these column names and would
    raise on them.
    """
    winners: dict[tuple[str, str], str] = {}
    if table["model"].nunique() < 2:
        # "Best per row" with one model bolds every cell, which reads as an
        # emphasis the table has not earned. Nothing to compare, nothing to bold.
        return winners
    for (region, band), group in table.groupby(["region", "band"], sort=False):
        means = group.set_index("model")["mean"]
        means = means[np.isfinite(means)]
        if means.empty:
            continue
        winners[(region, band)] = str(means.idxmin())  # lower error rate wins, always
    return winners


def _boundary_footnote(table: pd.DataFrame, metric: str, *, show_voxels: bool) -> str:
    """The caveats that must travel with every boundary-stratified table.

    Built once and rendered by BOTH the Markdown and the LaTeX formatter, the
    same pattern as `_footnote` / `_comparison_footnote` above and for the
    same reason: an earlier version of this codebase attached caveats only to
    the Markdown output, so the compiled PDF -- the artifact that actually
    goes in the paper -- silently dropped them.
    """
    parts = [
        "Bands are distance in millimetres from the GROUND-TRUTH boundary, so every model is "
        "stratified by the same partition of space -- a per-model stratification would not be "
        "comparable."
    ]
    parts.append(f"Rates ({metric}) are per-case means over the held-out cases.")

    missing = table[table["n_missing"] > 0]
    if not missing.empty:
        detail = ", ".join(
            f"{row['model']} {row['region']} {row['band']}: "
            f"{int(row['n_missing'])}/{int(row['n'])}"
            for _, row in missing.iterrows()
        )
        parts.append(
            "Cases where a band contained zero voxels are excluded from the mean rather than "
            f"given an arbitrary penalty ({detail})."
        )

    # `metric` is always one of berr/bfnr/bfpr -- build_boundary_table raises
    # otherwise -- so this fires unconditionally, but the check is left
    # explicit rather than assumed in case a future metric joins the family.
    if metric in ("berr", "bfnr", "bfpr"):
        parts.append(
            "berr = bfnr + bfpr exactly, by construction (missed tumour plus spurious tumour)."
        )

    if show_voxels:
        bands_in_order = list(dict.fromkeys(table["band"]))
        per_band = table.groupby("band", sort=False)["mean_voxels"].mean()
        detail = ", ".join(
            f"{band}: {per_band[band]:.0f}"
            for band in bands_in_order
            if np.isfinite(per_band[band])
        )
        if detail:
            parts.append(f"Mean voxel count per band, averaged over models and regions: {detail}.")

    return " ".join(parts)


def _boundary_cell(row: pd.Series, *, places: int, bold: bool, emphasis: str) -> str:
    """Render one `mean +/- std` cell for the boundary table, optionally bolded."""
    body = f"{_fmt(float(row['mean']), places)} ± {_fmt(float(row['std']), places)}"
    if bold and emphasis:
        return f"{emphasis}{body}{emphasis}"
    return body


def format_boundary_markdown(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    precision: int = 3,
    show_voxels: bool = True,
) -> str:
    """Render a `build_boundary_table` output as Markdown.

    Rows are grouped by region then band; columns are models; cells are
    `mean +/- std`. The winning (lowest-error) model per row is bolded.

    Args:
        table: Output of `build_boundary_table`.
        caption: Optional line printed above the table.
        precision: Decimal places for the error rates.
        show_voxels: Include the mean voxel count per band in the footnote.

    Returns:
        The Markdown text, ending without a trailing newline.

    Raises:
        ValueError: `table` is not a `build_boundary_table` output, or is empty.
    """
    _validate_boundary_tidy(table, "format_boundary_markdown")

    models = list(dict.fromkeys(table["model"]))
    regions = list(dict.fromkeys(table["region"]))
    metric = str(table["metric"].iloc[0])
    winners = _boundary_best_models(table)

    header = ["Region", "Band", *models]
    lines: list[str] = []
    if caption:
        lines.extend([caption, ""])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    indexed = table.set_index(["model", "region", "band"])
    for region in regions:
        region_bands = list(dict.fromkeys(table.loc[table["region"] == region, "band"]))
        for band in region_bands:
            cells = [region, band]
            for model in models:
                key = (model, region, band)
                if key not in indexed.index:
                    cells.append("--")
                    continue
                cells.append(
                    _boundary_cell(
                        indexed.loc[key],
                        places=precision,
                        bold=winners.get((region, band)) == model,
                        emphasis="**",
                    )
                )
            lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(_boundary_footnote(table, metric, show_voxels=show_voxels))
    return "\n".join(lines)


def format_boundary_latex(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
    show_voxels: bool = True,
) -> str:
    """Render a `build_boundary_table` output as a booktabs LaTeX table.

    Requires `\\usepackage{booktabs}` in the document preamble.

    Args:
        table: Output of `build_boundary_table`.
        caption: Optional table caption. Omitted from the output when `None`.
        label: Optional LaTeX label, used as `\\label{<label>}`. Omitted when
            `None`.
        precision: Decimal places for the error rates.
        show_voxels: Include the mean voxel count per band in the footnote.

    Returns:
        The LaTeX source, ending without a trailing newline.

    Raises:
        ValueError: `table` is not a `build_boundary_table` output, or is empty.
    """
    _validate_boundary_tidy(table, "format_boundary_latex")

    models = list(dict.fromkeys(table["model"]))
    regions = list(dict.fromkeys(table["region"]))
    metric = str(table["metric"].iloc[0])
    winners = _boundary_best_models(table)

    lines = ["\\begin{table}[t]", "\\centering"]
    if caption:
        lines.append(f"\\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{ll" + "c" * len(models) + "}")
    lines.append("\\toprule")
    header = ["Region", "Band", *(escape_latex(m) for m in models)]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    indexed = table.set_index(["model", "region", "band"])
    for region in regions:
        region_bands = list(dict.fromkeys(table.loc[table["region"] == region, "band"]))
        for band in region_bands:
            cells = [escape_latex(region), escape_latex(band)]
            for model in models:
                key = (model, region, band)
                if key not in indexed.index:
                    cells.append("--")
                    continue
                body = _boundary_cell(indexed.loc[key], places=precision, bold=False, emphasis="")
                # `$\pm$` rather than the literal character: a raw U+00B1 in a
                # .tex file only compiles under a UTF-8-aware engine.
                body = body.replace("±", "$\\pm$")
                if winners.get((region, band)) == model:
                    body = f"\\textbf{{{body}}}"
                cells.append(body)
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    lines.extend(_latex_note(_boundary_footnote(table, metric, show_voxels=show_voxels)))
    lines.append("\\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
_ALLOWED_TABLE_EXTENSIONS: frozenset[str] = frozenset({"md", "tex", "txt", "csv"})


def write_table(text: str, out_dir: str | Path, stem: str, extension: str = "md") -> Path:
    """Write rendered table text to `<out_dir>/<stem>.<extension>`.

    The only function in this module that touches the filesystem, and it only
    writes -- mirroring `figures.save_figure`.

    Args:
        text: Rendered table text.
        out_dir: Destination directory, created if missing.
        stem: Filename without extension.
        extension: One of `md`, `tex`, `txt`, `csv`.

    Returns:
        The written path.

    Raises:
        ValueError: `stem` is empty, contains a separator or a `"."`, or
            `extension` is not allowed.
    """
    if not stem:
        raise ValueError("write_table: `stem` must be a non-empty filename without extension.")
    if "/" in stem or "\\" in stem:
        raise ValueError(f"write_table: `stem` must be a bare filename, got {stem!r}.")
    if "." in stem:
        raise ValueError(f"write_table: `stem` must not contain '.', got {stem!r}.")
    ext = extension.lower().lstrip(".")
    if ext not in _ALLOWED_TABLE_EXTENSIONS:
        raise ValueError(
            f"write_table: unsupported extension {extension!r}; allowed: "
            f"{sorted(_ALLOWED_TABLE_EXTENSIONS)}."
        )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.{ext}"
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    logger.info("Wrote table %s", path)
    return path
