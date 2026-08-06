"""Tests for `neurovision.visualization.tables`.

Table formatting fails silently in a way plotting does not: a wrong bold, an
unescaped underscore or a rounded-to-zero p-value all produce output that looks
fine right up until it is in a submitted PDF. So the tests here check the exact
rendered strings, not just that a string came back.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from neurovision.visualization.tables import (
    build_results_table,
    escape_latex,
    format_comparison_latex,
    format_comparison_markdown,
    format_results_latex,
    format_results_markdown,
    write_table,
)


def _per_case(seed: int, *, hd95_nan: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 12
    data: dict[str, np.ndarray] = {}
    for region in ("WT", "TC", "ET"):
        data[f"dice_{region}"] = rng.uniform(0.7, 0.99, n)
        data[f"hd95_{region}"] = rng.uniform(1.0, 12.0, n)
        data[f"gt_empty_{region}"] = np.zeros(n)
    table = pd.DataFrame(data, index=[f"case_{i:03d}" for i in range(n)])
    if hd95_nan:
        table.loc[table.index[:hd95_nan], "hd95_ET"] = np.nan
    return table


def _results_table() -> pd.DataFrame:
    return build_results_table({"U-Net": _per_case(0), "NeuroVision-X": _per_case(1)})


# --------------------------------------------------------------------------- #
# build_results_table
# --------------------------------------------------------------------------- #
def test_build_results_table_shape_and_columns() -> None:
    table = _results_table()
    assert list(table.columns) == [
        "model",
        "region",
        "metric",
        "mean",
        "std",
        "median",
        "n",
        "n_missing",
        "gt_empty_frac",
    ]
    # 2 models x 3 regions x 2 metrics
    assert len(table) == 12
    assert list(dict.fromkeys(table["region"])) == ["WT", "TC", "ET"]


def test_build_results_table_statistics_match_pandas() -> None:
    per_case = _per_case(0)
    table = build_results_table({"U-Net": per_case}, regions=["WT"], metrics=["dice"])
    row = table.iloc[0]
    assert row["mean"] == pytest.approx(per_case["dice_WT"].mean())
    # ddof=1 -- the sample std, matching pandas' default. A silent switch to the
    # population std would shift every +/- in the paper.
    assert row["std"] == pytest.approx(per_case["dice_WT"].std())
    assert row["median"] == pytest.approx(per_case["dice_WT"].median())
    assert row["n"] == 12
    assert row["n_missing"] == 0


def test_build_results_table_counts_nan_as_missing_and_excludes_it_from_the_mean() -> None:
    """HD95 is legitimately NaN when exactly one side of a region is empty."""
    per_case = _per_case(0, hd95_nan=3)
    table = build_results_table({"U-Net": per_case}, regions=["ET"], metrics=["hd95"])
    row = table.iloc[0]
    assert row["n"] == 12
    assert row["n_missing"] == 3
    assert row["mean"] == pytest.approx(per_case["hd95_ET"].mean())  # pandas also skips NaN


def test_build_results_table_is_strict_about_a_missing_column() -> None:
    with pytest.raises(ValueError, match="ece_WT"):
        build_results_table({"U-Net": _per_case(0)}, regions=["WT"], metrics=["ece"])


def test_build_results_table_can_degrade_instead_of_raising(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        table = build_results_table(
            {"U-Net": _per_case(0)}, regions=["WT"], metrics=["ece"], strict=False
        )
    assert "ece_WT" in caplog.text
    assert np.isnan(table.iloc[0]["mean"])


def test_build_results_table_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        build_results_table({})


def test_build_results_table_records_gt_empty_fraction() -> None:
    per_case = _per_case(0)
    per_case.loc[per_case.index[:3], "gt_empty_ET"] = 1.0
    table = build_results_table({"U-Net": per_case}, regions=["ET"], metrics=["dice"])
    assert table.iloc[0]["gt_empty_frac"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Markdown results
# --------------------------------------------------------------------------- #
def test_results_markdown_header_and_row_count() -> None:
    text = format_results_markdown(_results_table(), caption="Test caption")
    lines = text.splitlines()
    assert lines[0] == "Test caption"
    header = lines[2]
    assert header.startswith("| Model |")
    assert "WT dice" in header and "ET hd95" in header
    assert any(line.startswith("| U-Net |") for line in lines)
    assert any(line.startswith("| NeuroVision-X |") for line in lines)


def test_results_markdown_respects_per_metric_precision() -> None:
    """Dice gets 4 places, HD95 gets 2 -- a third HD95 decimal is noise."""
    text = format_results_markdown(_results_table(), show_median=False)
    row = next(line for line in text.splitlines() if line.startswith("| U-Net |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    dice_cell, hd95_cell = cells[1], cells[2]
    assert len(dice_cell.split(" ± ")[0].split(".")[1]) == 4
    assert len(hd95_cell.split(" ± ")[0].split(".")[1]) == 2


def test_results_markdown_bolds_the_better_model_per_direction() -> None:
    """Higher Dice wins; LOWER HD95 wins. Bolding the wrong end inverts a claim."""
    per_case_a = _per_case(0)
    per_case_b = _per_case(0)
    per_case_a["dice_WT"] = 0.90
    per_case_b["dice_WT"] = 0.80
    per_case_a["hd95_WT"] = 8.0
    per_case_b["hd95_WT"] = 3.0
    table = build_results_table({"A": per_case_a, "B": per_case_b}, regions=["WT"])
    text = format_results_markdown(table, show_median=False)
    row_a = next(line for line in text.splitlines() if line.startswith("| A |"))
    row_b = next(line for line in text.splitlines() if line.startswith("| B |"))
    assert "**0.9000" in row_a  # A wins Dice
    assert "**3.00" in row_b  # B wins HD95 (lower is better)
    assert "**8.00" not in row_a


def test_results_markdown_highlight_can_be_disabled() -> None:
    text = format_results_markdown(_results_table(), highlight_best=False)
    assert "**" not in text


def test_results_markdown_footnote_names_the_missing_cases() -> None:
    table = build_results_table({"U-Net": _per_case(0, hd95_nan=3)}, regions=["ET"])
    text = format_results_markdown(table)
    assert "3/12" in text
    assert "ignore_empty=False" in text


def test_results_markdown_renders_nan_as_a_dash() -> None:
    table = build_results_table(
        {"U-Net": _per_case(0)}, regions=["WT"], metrics=["ece"], strict=False
    )
    assert "--" in format_results_markdown(table)


def test_results_markdown_rejects_a_foreign_table() -> None:
    with pytest.raises(ValueError, match="missing column"):
        format_results_markdown(pd.DataFrame({"model": ["a"]}))


# --------------------------------------------------------------------------- #
# LaTeX results
# --------------------------------------------------------------------------- #
def test_results_latex_is_booktabs_and_balanced() -> None:
    text = format_results_latex(_results_table(), caption="Main results", label="tab:main")
    assert "\\begin{table}" in text and "\\end{table}" in text
    assert text.count("\\toprule") == 1
    assert text.count("\\midrule") == 1
    assert text.count("\\bottomrule") == 1
    assert "\\label{tab:main}" in text
    # 1 model column + 3 regions x 2 metrics
    assert "\\begin{tabular}{lcccccc}" in text


def test_results_latex_escapes_underscores_in_model_names() -> None:
    """`baseline_unet3d` is the realistic label, and a bare `_` breaks the build."""
    table = build_results_table({"baseline_unet3d": _per_case(0)}, regions=["WT"])
    text = format_results_latex(table, caption="c", label="tab:x")
    assert "baseline\\_unet3d" in text
    assert "baseline_unet3d" not in text


def test_results_latex_uses_a_math_pm_not_a_literal_character() -> None:
    """A raw U+00B1 only compiles under a UTF-8-aware engine."""
    text = format_results_latex(_results_table(), caption="c", label="tab:x")
    assert "$\\pm$" in text
    assert "±" not in text


def test_results_latex_bolds_the_winner() -> None:
    per_case_a = _per_case(0)
    per_case_b = _per_case(0)
    per_case_a["dice_WT"] = 0.90
    per_case_b["dice_WT"] = 0.80
    table = build_results_table(
        {"A": per_case_a, "B": per_case_b}, regions=["WT"], metrics=["dice"]
    )
    text = format_results_latex(table, caption="c", label="tab:x")
    assert "\\textbf{0.9000" in text


def test_results_latex_emits_a_cmidrule_per_region_group() -> None:
    text = format_results_latex(_results_table(), caption="c", label="tab:x")
    assert text.count("\\cmidrule(lr)") == 3
    assert "\\cmidrule(lr){2-3}" in text
    assert "\\cmidrule(lr){4-5}" in text


# --------------------------------------------------------------------------- #
# escape_latex
# --------------------------------------------------------------------------- #
def test_escape_latex_handles_every_special_character() -> None:
    assert escape_latex("a_b") == "a\\_b"
    assert escape_latex("50%") == "50\\%"
    assert escape_latex("a&b") == "a\\&b"
    assert escape_latex("#1") == "\\#1"


def test_escape_latex_does_not_double_escape_its_own_backslash() -> None:
    """Escaping the backslash first is what stops the later rules re-escaping it."""
    assert escape_latex("\\") == "\\textbackslash{}"


# --------------------------------------------------------------------------- #
# Comparison tables
# --------------------------------------------------------------------------- #
def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n": [180, 180, 180],
            "mean_Ours": [0.9200, 0.8600, 3.10],
            "mean_U-Net": [0.9000, 0.8580, 5.20],
            "mean_diff": [0.02, 0.002, -2.10],
            "improvement": [0.02, 0.002, 2.10],
            "improvement_lo": [0.011, -0.001, 0.90],
            "improvement_hi": [0.029, 0.005, 3.30],
            "p_holm": [0.0002, 0.41, 0.004],
            "hedges_g": [0.62, 0.05, 0.44],
            "verdict": ["better", "inconclusive", "better"],
        },
        index=["dice_WT", "dice_ET", "hd95_WT"],
    )


def test_comparison_markdown_includes_both_model_means() -> None:
    text = format_comparison_markdown(_comparison(), name_a="Ours", name_b="U-Net")
    header = text.splitlines()[0]
    assert "| Ours | U-Net |" in header
    assert "0.9200" in text and "0.9000" in text


def test_comparison_markdown_floors_tiny_p_values() -> None:
    """`p = 0.000` is a claim no finite sample supports."""
    text = format_comparison_markdown(_comparison(), name_a="Ours", name_b="U-Net")
    assert "< 0.001" in text
    assert "| 0.000 |" not in text


def test_comparison_markdown_prints_the_ci_as_an_interval() -> None:
    text = format_comparison_markdown(_comparison(), name_a="Ours", name_b="U-Net")
    assert "[0.0110, 0.0290]" in text


def test_comparison_markdown_carries_the_verdict_caveat() -> None:
    text = format_comparison_markdown(_comparison(), name_a="Ours", name_b="U-Net")
    assert "inconclusive" in text
    assert "Holm family is the whole table" in text
    assert "Ours is better" in text


def test_comparison_markdown_warns_when_the_names_do_not_match(caplog) -> None:
    """`mean_<name>` columns are named by compare_models; a mismatch is a real slip."""
    with caplog.at_level(logging.WARNING):
        text = format_comparison_markdown(_comparison(), name_a="wrong", name_b="also-wrong")
    assert "mean_wrong" in caplog.text
    assert "| Metric | Improvement |" in text.splitlines()[0]


def test_comparison_markdown_rejects_a_foreign_table() -> None:
    with pytest.raises(ValueError, match="missing column"):
        format_comparison_markdown(pd.DataFrame({"improvement": [0.1]}, index=["dice_WT"]))


def test_comparison_latex_escapes_and_is_balanced() -> None:
    text = format_comparison_latex(
        _comparison(), caption="Comparison", label="tab:cmp", name_a="Ours", name_b="U-Net"
    )
    assert "\\toprule" in text and "\\bottomrule" in text
    assert "dice\\_WT" in text
    assert "$<$ 0.001" in text
    assert "95\\% CI" in text


# --------------------------------------------------------------------------- #
# write_table
# --------------------------------------------------------------------------- #
def test_write_table_creates_the_directory_and_appends_a_newline(tmp_path) -> None:
    path = write_table("hello", tmp_path / "nested", "results", "md")
    assert path == tmp_path / "nested" / "results.md"
    assert path.read_text(encoding="utf-8") == "hello\n"


def test_write_table_does_not_double_a_trailing_newline(tmp_path) -> None:
    path = write_table("hello\n", tmp_path, "results", "md")
    assert path.read_text(encoding="utf-8") == "hello\n"


@pytest.mark.parametrize("stem", ["", "a/b", "results.v2"])
def test_write_table_rejects_bad_stems(tmp_path, stem: str) -> None:
    with pytest.raises(ValueError):
        write_table("x", tmp_path, stem)


def test_write_table_rejects_an_unknown_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported extension"):
        write_table("x", tmp_path, "results", "docx")
