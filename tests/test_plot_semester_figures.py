"""Tests for `scripts.plot_semester_figures`.

Synthetic DataFrames only, CPU, fast -- no real `outputs/` artifact is read.
Matches the pattern in `tests/test_figures.py`: force the Agg backend before
any pyplot import, and close every figure a test builds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# scripts/ is not a package (no __init__.py), so it is imported by adding the
# scripts directory to sys.path -- the same trick scripts/*.py itself uses to
# reach configs/ by a path relative to __file__, applied here to reach the
# script's module from a test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import plot_semester_figures as psf  # noqa: E402


def _local_recalibration_summary() -> pd.DataFrame:
    """A small synthetic `summary.csv`-shaped table.

    Covers: two models (only "neurovision" should be plotted), a `k="half"`
    (is_half=True) row that must be excluded, and one series -- PED.TC -- that
    is INFEASIBLE (NaN risk) at every swept k, which must end up labelled
    "(infeasible at every k)" rather than silently dropped.
    """
    rows = []
    alphas = [0.05, 0.10, 0.20]
    ks = [5, 10, 15, 20, 30]
    for model in ("neurovision", "baseline_unet3d"):
        for cohort in ("ssa", "ped"):
            for region in ("WT", "TC"):
                for alpha in alphas:
                    for k in ks:
                        infeasible = cohort == "ped" and region == "TC"
                        risk = float("nan") if infeasible else 0.5 * alpha
                        lo = float("nan") if infeasible else risk * 0.8
                        hi = float("nan") if infeasible else risk * 1.2
                        rows.append(
                            {
                                "cohort": cohort,
                                "region": region,
                                "alpha": alpha,
                                "k": k,
                                "n_splits": 1000,
                                "feasible_rate": 0.0 if infeasible else 1.0,
                                "mean_threshold": float("nan"),
                                "mean_realised_risk": risk,
                                "p_violation": 0.0,
                                "mean_inflation": 1.0,
                                "realised_risk_ci_low": lo,
                                "realised_risk_ci_high": hi,
                                "n_feasible": 0 if infeasible else 1000,
                                "mc_se": 0.001,
                                "is_half": False,
                                "model": model,
                                "verdict": "INFEASIBLE" if infeasible else "RESTORED",
                            }
                        )
        # One "test"-cohort row per model, is_half=True -- must never appear
        # in the figure, but because "test" is not in `_LOCAL_RECAL_SERIES`,
        # not because of any is_half filtering (there is none any more).
        rows.append(
            {
                "cohort": "test",
                "region": "WT",
                "alpha": 0.05,
                "k": 94,
                "n_splits": 1000,
                "feasible_rate": 1.0,
                "mean_threshold": 0.1,
                "mean_realised_risk": 999.0,
                "p_violation": 0.0,
                "mean_inflation": 1.0,
                "realised_risk_ci_low": 999.0,
                "realised_risk_ci_high": 999.0,
                "n_feasible": 1000,
                "mc_se": 0.001,
                "is_half": True,
                "model": model,
                "verdict": "RESTORED",
            }
        )
    return pd.DataFrame(rows)


def _realised_vs_nominal_table() -> pd.DataFrame:
    """A `realised_risk.csv`-shaped table with a `cohort` column already attached."""
    rows = []
    for cohort, region, scale in (
        ("test", "WT", 0.8),
        ("test", "TC", 0.9),
        ("ssa", "WT", 1.1),
        ("ssa", "TC", 1.5),
        ("ped", "WT", 1.4),
        ("ped", "TC", 4.0),
    ):
        for alpha in (0.05, 0.10, 0.20):
            rows.append(
                {
                    "apply_dir": f"outputs/{cohort}",
                    "cohort": cohort,
                    "region": region,
                    "alpha": alpha,
                    "threshold": 0.5,
                    "mean_miss_rate": alpha * scale,
                    "n": 100,
                    "n_excluded_empty": 0,
                    "ci_lo": alpha * scale * 0.8,
                    "ci_hi": alpha * scale * 1.2,
                }
            )
    return pd.DataFrame(rows)


class TestPlotLocalRecalibration:
    def test_three_panels_and_infeasible_label(self) -> None:
        df = _local_recalibration_summary()
        fig = psf.plot_local_recalibration(df)
        try:
            assert len(fig.axes) == 3
            # alpha panel titles, in order.
            titles = [ax.get_title() for ax in fig.axes]
            assert any("0.05" in t for t in titles)
            assert any("0.1" in t for t in titles)
            assert any("0.2" in t for t in titles)

            # Every panel's legend must contain the infeasible PED.TC series,
            # labelled as infeasible, and must NOT contain the huge is_half
            # sentinel value (999.0) anywhere in the panel's y data.
            for ax in fig.axes:
                labels = [line.get_label() for line in ax.get_lines()]
                assert any("infeasible at every k" in label for label in labels)
                for line in ax.get_lines():
                    ydata = line.get_ydata()
                    assert all(abs(y) < 900 for y in ydata if y == y)  # skip NaN
        finally:
            plt.close(fig)

    def test_alpha_reference_line_present(self) -> None:
        df = _local_recalibration_summary()
        fig = psf.plot_local_recalibration(df, alphas=(0.05,))
        try:
            assert len(fig.axes) == 1
            ax = fig.axes[0]
            hlines = [line for line in ax.get_lines() if len(set(line.get_ydata())) == 1]
            assert any(abs(line.get_ydata()[0] - 0.05) < 1e-9 for line in hlines)
        finally:
            plt.close(fig)

    def test_raises_on_unknown_model(self) -> None:
        df = _local_recalibration_summary()
        with pytest.raises(ValueError):
            psf.plot_local_recalibration(df, model="does_not_exist")

    def test_is_half_point_is_still_plotted(self) -> None:
        """Regression test: a cohort's own half-point (e.g. SSA's k=30) must not be dropped.

        note 52 quotes SSA·TC's k=30 row by name (feasible 0.387, NOT_RESTORED)
        and PED·WT's k=49 row by name (RESTORED); both are is_half=True in the
        real summary.csv and must still be drawn. Here the ONLY row at k=30
        for SSA·WT is marked is_half=True, with no separate is_half=False k=30
        row -- exactly the shape that made the old `is_half == False` filter
        silently drop it.
        """
        rows = [
            {
                "cohort": "ssa",
                "region": "WT",
                "alpha": 0.05,
                "k": k,
                "mean_realised_risk": 0.01 * k,
                "realised_risk_ci_low": 0.008 * k,
                "realised_risk_ci_high": 0.012 * k,
                "is_half": k == 30,
                "model": "neurovision",
                "verdict": "RESTORED",
            }
            for k in (5, 10, 30)
        ]
        df = pd.DataFrame(rows)
        fig = psf.plot_local_recalibration(df, alphas=(0.05,), series=(("ssa", "WT"),))
        try:
            ax = fig.axes[0]
            (line,) = [line for line in ax.get_lines() if line.get_label() == "SSA·WT"]
            assert 30.0 in line.get_xdata()
        finally:
            plt.close(fig)

    def test_not_restored_points_get_an_x_marker_overlay(self) -> None:
        rows = [
            {
                "cohort": "ssa",
                "region": "TC",
                "alpha": 0.05,
                "k": k,
                "mean_realised_risk": 0.05,
                "realised_risk_ci_low": 0.04,
                "realised_risk_ci_high": 0.06,
                "is_half": False,
                "model": "neurovision",
                "verdict": "NOT_RESTORED" if k == 30 else "RESTORED",
            }
            for k in (5, 10, 30)
        ]
        df = pd.DataFrame(rows)
        fig = psf.plot_local_recalibration(df, alphas=(0.05,), series=(("ssa", "TC"),))
        try:
            from matplotlib.collections import PathCollection

            ax = fig.axes[0]
            x_marker_collections = [c for c in ax.collections if isinstance(c, PathCollection)]
            assert len(x_marker_collections) == 1
            (offsets,) = [c.get_offsets() for c in x_marker_collections]
            assert list(offsets[0]) == [30.0, 0.05]
        finally:
            plt.close(fig)

    def test_duplicate_k_is_deduped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        rows = [
            {
                "cohort": "ssa",
                "region": "WT",
                "alpha": 0.05,
                "k": 30,
                "mean_realised_risk": 0.05,
                "realised_risk_ci_low": 0.04,
                "realised_risk_ci_high": 0.06,
                "is_half": is_half,
                "model": "neurovision",
                "verdict": "RESTORED",
            }
            for is_half in (False, True)
        ]
        df = pd.DataFrame(rows)
        with caplog.at_level("WARNING"):
            fig = psf.plot_local_recalibration(df, alphas=(0.05,), series=(("ssa", "WT"),))
        try:
            assert "duplicate" in caplog.text.lower()
            ax = fig.axes[0]
            (line,) = [line for line in ax.get_lines() if line.get_label() == "SSA·WT"]
            assert list(line.get_xdata()) == [30.0]
        finally:
            plt.close(fig)


class TestPlotRealisedVsNominal:
    def test_two_panels_one_per_region(self) -> None:
        df = _realised_vs_nominal_table()
        fig = psf.plot_realised_vs_nominal(df)
        try:
            assert len(fig.axes) == 2
            titles = {ax.get_title() for ax in fig.axes}
            assert titles == {"WT", "TC"}
        finally:
            plt.close(fig)

    def test_raises_on_empty_table(self) -> None:
        empty = pd.DataFrame(columns=["cohort", "region", "alpha", "mean_miss_rate"])
        with pytest.raises(ValueError):
            psf.plot_realised_vs_nominal(empty)


class TestCohortFromApplyDir:
    def test_recognized_basenames(self) -> None:
        assert psf.cohort_from_apply_dir("outputs/neurovision/eval_test") == "test"
        assert psf.cohort_from_apply_dir("outputs/eval_ssa_neurovision") == "ssa"
        assert psf.cohort_from_apply_dir("outputs/eval_ped_neurovision") == "ped"

    def test_unrecognized_basename_returns_none(self) -> None:
        # Deliberately a near-miss (contains "test" as a substring) to check
        # this is an exact basename match, not a substring check.
        assert psf.cohort_from_apply_dir("outputs/eval_test_baseline_unet3d") is None


class TestSaveLocalRecalibrationFigure:
    def test_writes_png_and_caption(self, tmp_path: Path) -> None:
        df = _local_recalibration_summary()
        fig = psf.plot_local_recalibration(df)
        written = psf.save_local_recalibration_figure(fig, tmp_path, dpi=100)

        png_path = tmp_path / "fig_local_recalibration.png"
        caption_path = tmp_path / "fig_local_recalibration.txt"
        assert png_path in written
        assert caption_path in written
        assert png_path.is_file()
        assert png_path.stat().st_size > 0
        assert caption_path.is_file()
        caption_text = caption_path.read_text(encoding="utf-8")
        assert "counterfactual" in caption_text.lower()
        assert "note 52" in caption_text
        assert "not_restored" in caption_text.lower()
