"""Tests for `neurovision.visualization.figures`.

Every test runs on CPU on tiny synthetic arrays and closes its figures, so the
suite neither slows down nor triggers matplotlib's "more than 20 figures open"
warning. Style-touching tests run inside the `paper_style` context manager so no
test can leak rcParams into the next one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from neurovision.visualization import figures  # noqa: E402
from neurovision.visualization.figures import (  # noqa: E402
    MODEL_COLOR_CYCLE,
    QualitativeCase,
    TrainingPanel,
    model_style,
    paper_rc,
    paper_style,
    pick_slice,
    plot_comparison_forest,
    plot_metric_distributions,
    plot_qualitative_panel,
    plot_reliability_diagram,
    plot_risk_coverage,
    plot_training_curves,
    save_figure,
    take_slice,
)

SHAPE = (12, 14, 16)


def _synthetic_image() -> np.ndarray:
    """`(4, D, H, W)` with a distinguishable constant offset per modality."""
    base = np.zeros((4, *SHAPE), dtype=np.float32)
    for channel in range(4):
        base[channel] = channel + 1
    # A nonzero "brain" region so `crop_to_brain` has something to find, and a
    # zero rim so the crop is a real crop rather than the whole slice.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[2:10, 3:11, 4:12] = True
    base *= mask[None]
    return base


def _synthetic_label() -> np.ndarray:
    """`(D, H, W)` int label with one small cube of each class 1, 2, 3."""
    label = np.zeros(SHAPE, dtype=np.uint8)
    label[3:6, 4:7, 5:8] = 2  # ED, the outermost / biggest
    label[4:6, 5:7, 6:8] = 1  # NCR
    label[4:5, 5:6, 6:7] = 3  # ET, the smallest
    return label


def _case(case_id: str = "CASE_A", *, with_uncertainty: bool = False) -> QualitativeCase:
    label = _synthetic_label()
    pred_a = label.copy()
    pred_b = np.zeros_like(label)
    pred_b[3:5, 4:6, 5:7] = 2
    return QualitativeCase(
        case_id=case_id,
        image=_synthetic_image(),
        ground_truth=label,
        predictions={"Baseline": pred_a, "NeuroVision-X": pred_b},
        uncertainty=(
            np.abs(np.linspace(0, 1, int(np.prod(SHAPE))).reshape(SHAPE))
            if with_uncertainty
            else None
        ),
        uncertainty_label="MC-dropout MI",
    )


def _grid_axes(fig) -> list:
    """Axes that actually hold an image -- excludes colorbar axes."""
    return [ax for ax in fig.axes if ax.get_images()]


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def test_paper_rc_uses_type42_fonts() -> None:
    """Regression guard: type 3 fonts are rejected by many publishers.

    Invisible until a submission bounces, so it is pinned here rather than
    trusted to stay correct.
    """
    rc = paper_rc()
    assert rc["pdf.fonttype"] == 42
    assert rc["ps.fonttype"] == 42


def test_paper_rc_scales_every_size_from_the_base() -> None:
    rc = paper_rc(base_font_size=10.0)
    assert rc["font.size"] == 10.0
    assert rc["axes.titlesize"] == 11.0
    assert rc["xtick.labelsize"] == 9.0


def test_paper_style_restores_rcparams_even_on_exception() -> None:
    before = plt.rcParams["pdf.fonttype"]
    plt.rcParams["pdf.fonttype"] = 3
    try:
        with pytest.raises(RuntimeError):
            with paper_style():
                assert plt.rcParams["pdf.fonttype"] == 42
                raise RuntimeError("boom")
        assert plt.rcParams["pdf.fonttype"] == 3
    finally:
        plt.rcParams["pdf.fonttype"] = before


def test_model_style_cycles_and_wraps() -> None:
    first = model_style(0)
    wrapped = model_style(len(MODEL_COLOR_CYCLE))
    assert first["color"] == wrapped["color"]
    assert model_style(0)["color"] != model_style(1)["color"]
    assert model_style(0)["linestyle"] != model_style(1)["linestyle"]


def test_model_style_rejects_a_negative_index() -> None:
    """A negative index would wrap to another model's colour instead of erroring."""
    with pytest.raises(ValueError, match="index must be >= 0"):
        model_style(-1)


# --------------------------------------------------------------------------- #
# save_figure
# --------------------------------------------------------------------------- #
def test_save_figure_writes_vector_and_raster_and_creates_the_directory(tmp_path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    target = tmp_path / "nested" / "figs"
    paths = save_figure(fig, target, "fig1", close=True)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert target.is_dir()


def test_save_figure_close_actually_closes(tmp_path) -> None:
    fig, _ = plt.subplots()
    number = fig.number
    save_figure(fig, tmp_path, "fig", formats=("png",), close=True)
    assert not plt.fignum_exists(number)


@pytest.mark.parametrize(
    "stem",
    ["", f"a{os.sep}b", "fig1.v2"],
)
def test_save_figure_rejects_bad_stems(tmp_path, stem: str) -> None:
    fig, _ = plt.subplots()
    try:
        with pytest.raises(ValueError):
            save_figure(fig, tmp_path, stem)
    finally:
        plt.close(fig)


def test_save_figure_rejects_an_unknown_format(tmp_path) -> None:
    fig, _ = plt.subplots()
    try:
        with pytest.raises(ValueError, match="unsupported format"):
            save_figure(fig, tmp_path, "fig", formats=("tiff",))
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Slice selection
# --------------------------------------------------------------------------- #
def test_pick_slice_returns_the_index_with_the_most_foreground() -> None:
    label = np.zeros((4, 4, 5), dtype=np.uint8)
    label[:, :, 1] = 1  # 16 voxels
    label[:2, :2, 3] = 1  # 4 voxels
    assert pick_slice(label, "axial") == 1


def test_pick_slice_breaks_ties_toward_the_lowest_index() -> None:
    """Determinism: the same inputs must always produce the same figure."""
    label = np.zeros((4, 4, 5), dtype=np.uint8)
    label[:2, :2, 1] = 1
    label[:2, :2, 3] = 1
    assert pick_slice(label, "axial") == 1


def test_pick_slice_uses_the_midpoint_and_warns_on_an_empty_volume(caplog) -> None:
    label = np.zeros((4, 4, 5), dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        assert pick_slice(label, "axial") == 2
    assert "zero foreground" in caplog.text


def test_pick_slice_honours_the_plane_axis() -> None:
    label = np.zeros((4, 6, 5), dtype=np.uint8)
    label[3, :, :] = 1
    assert pick_slice(label, "sagittal") == 3


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros((2, 2, 2, 2))])
def test_pick_slice_rejects_non_3d(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"\(D, H, W\)"):
        pick_slice(bad)


def test_pick_slice_rejects_an_unknown_plane() -> None:
    with pytest.raises(ValueError, match="unknown plane"):
        pick_slice(np.zeros((2, 2, 2)), "coronal-ish")


def test_take_slice_matches_qc_orientation() -> None:
    """Pins the rot90 against `qc.extract_mid_slices` so the two never diverge."""
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    got = take_slice(volume, 2, "axial")
    expected = np.rot90(volume[:, :, 2])
    np.testing.assert_array_equal(got, expected)
    # And an asymmetric marker really does move: a bare slice would not equal
    # the rotated one, which is what makes this test meaningful.
    assert not np.array_equal(got, volume[:, :, 2])


def test_take_slice_rejects_an_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        take_slice(np.zeros((2, 2, 2)), 5, "axial")


# --------------------------------------------------------------------------- #
# QualitativeCase.validate
# --------------------------------------------------------------------------- #
def test_validate_accepts_a_well_formed_case() -> None:
    _case().validate()  # must not raise


def test_validate_rejects_a_wrong_channel_count() -> None:
    case = _case()
    case.image = case.image[:3]
    with pytest.raises(ValueError, match="CASE_A.*shape \\(4, D, H, W\\)"):
        case.validate()


def test_validate_rejects_a_ground_truth_shape_mismatch() -> None:
    case = _case()
    case.ground_truth = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="CASE_A.*ground_truth"):
        case.validate()


def test_validate_names_the_offending_prediction() -> None:
    """The geometry trap: an un-recropped prediction is the realistic mistake."""
    case = _case()
    case.predictions["NeuroVision-X"] = np.zeros((240, 240, 155), dtype=np.uint8)
    with pytest.raises(ValueError, match="NeuroVision-X"):
        case.validate()


def test_validate_rejects_an_uncertainty_shape_mismatch() -> None:
    case = _case()
    case.uncertainty = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="uncertainty"):
        case.validate()


def test_validate_rejects_empty_predictions() -> None:
    case = _case()
    case.predictions = {}
    with pytest.raises(ValueError, match="predictions.* is empty"):
        case.validate()


# --------------------------------------------------------------------------- #
# Qualitative panel
# --------------------------------------------------------------------------- #
def test_qualitative_panel_has_one_axes_per_grid_cell() -> None:
    with paper_style():
        fig = plot_qualitative_panel([_case("A"), _case("B")])
    try:
        # 2 rows x (modality + GT + 2 models) = 8 image-bearing axes.
        assert len(_grid_axes(fig)) == 8
    finally:
        plt.close(fig)


def test_qualitative_panel_adds_an_uncertainty_column_when_present() -> None:
    with paper_style():
        fig = plot_qualitative_panel([_case("A", with_uncertainty=True)])
    try:
        # 1 row x (modality + GT + 2 models + uncertainty) = 5.
        assert len(_grid_axes(fig)) == 5
    finally:
        plt.close(fig)


def test_qualitative_panel_tolerates_a_case_without_uncertainty() -> None:
    with paper_style():
        fig = plot_qualitative_panel([_case("A", with_uncertainty=True), _case("B")])
    try:
        # Row B's uncertainty cell holds text, not an image, so it is not counted.
        assert len(_grid_axes(fig)) == 5 + 4
        assert any("n/a" in t.get_text() for ax in fig.axes for t in ax.texts)
    finally:
        plt.close(fig)


def test_qualitative_panel_rejects_differing_prediction_keys() -> None:
    a = _case("A")
    b = _case("B")
    b.predictions = {"NeuroVision-X": b.predictions["NeuroVision-X"]}
    with pytest.raises(ValueError, match="'A'"):
        plot_qualitative_panel([a, b])


def test_qualitative_panel_rejects_reordered_prediction_keys() -> None:
    """Columns are positional; a reorder would attribute one model's output to another."""
    a = _case("A")
    b = _case("B")
    b.predictions = {
        "NeuroVision-X": b.predictions["NeuroVision-X"],
        "Baseline": b.predictions["Baseline"],
    }
    with pytest.raises(ValueError, match="prediction keys"):
        plot_qualitative_panel([a, b])


def test_qualitative_panel_rejects_an_unknown_modality() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        plot_qualitative_panel([_case()], modality="PET")


def test_qualitative_panel_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        plot_qualitative_panel([])


def test_qualitative_panel_rasterizes_every_image() -> None:
    """Keeps text and contours vector while image layers rasterize.

    Without this the panel's PDF balloons; with it the same figure stays small
    and the outlines stay crisp at any zoom.
    """
    with paper_style():
        fig = plot_qualitative_panel([_case("A", with_uncertainty=True)])
    try:
        images = [im for ax in fig.axes for im in ax.get_images()]
        assert images
        assert all(im.get_rasterized() for im in images)
    finally:
        plt.close(fig)


def _roomy_case() -> QualitativeCase:
    """A case whose brain is small relative to the volume.

    The default `_case` cannot exercise cropping: its brain plus the 4-voxel pad
    covers the whole 12x14x16 volume, so the crop is a legitimate no-op.
    """
    shape = (30, 32, 34)
    image = np.zeros((4, *shape), dtype=np.float32)
    image[:, 10:16, 12:18, 14:20] = 1.0
    label = np.zeros(shape, dtype=np.uint8)
    label[11:14, 13:16, 15:18] = 2
    return QualitativeCase(
        case_id="ROOMY",
        image=image,
        ground_truth=label,
        predictions={"Baseline": label.copy()},
    )


def test_qualitative_panel_crop_shrinks_the_displayed_extent() -> None:
    with paper_style():
        cropped = plot_qualitative_panel([_roomy_case()], crop_to_brain=True)
        full = plot_qualitative_panel([_roomy_case()], crop_to_brain=False)
    try:
        cropped_shape = _grid_axes(cropped)[0].get_images()[0].get_array().shape
        full_shape = _grid_axes(full)[0].get_images()[0].get_array().shape
        assert cropped_shape[0] < full_shape[0] or cropped_shape[1] < full_shape[1]
    finally:
        plt.close(cropped)
        plt.close(full)


def test_qualitative_panel_renders_background_as_black() -> None:
    """Preprocessing z-scores over nonzero voxels, so brain interiors go negative.

    A plain min-max would map the zero-valued air outside the head to mid-grey and
    render every brain on a grey card. Exact zeros are the preprocessing's own
    background marker and must come out at 0.0.
    """
    case = _roomy_case()
    # Negative interior values, exactly as z-scoring produces.
    case.image[:, 10:16, 12:18, 14:20] = -2.0
    case.image[:, 12:14, 14:16, 16:18] = 3.0
    with paper_style():
        fig = plot_qualitative_panel([case], crop_to_brain=False)
    try:
        displayed = _grid_axes(fig)[0].get_images()[0].get_array()
        assert float(displayed[0, 0]) == 0.0  # a corner is outside the head
        assert float(displayed.max()) > 0.0  # and the brain is not black too
    finally:
        plt.close(fig)


def test_qualitative_panel_crop_survives_negative_intensities() -> None:
    """Regression: the brain bbox is found on the DISPLAY array, not the raw one.

    With a min-max normalization the background is nonzero, so the bounding box
    covered the whole slice and `crop_to_brain` silently did nothing.
    """
    case = _roomy_case()
    case.image[:, 10:16, 12:18, 14:20] = -2.0
    with paper_style():
        cropped = plot_qualitative_panel([case], crop_to_brain=True)
        full = plot_qualitative_panel([_roomy_case()], crop_to_brain=False)
    try:
        cropped_shape = _grid_axes(cropped)[0].get_images()[0].get_array().shape
        full_shape = _grid_axes(full)[0].get_images()[0].get_array().shape
        assert cropped_shape[0] < full_shape[0]
        assert cropped_shape[1] < full_shape[1]
    finally:
        plt.close(cropped)
        plt.close(full)


def test_qualitative_panel_tolerates_an_all_zero_slice() -> None:
    """A slice past the end of the head is all background, not an error."""
    case = _roomy_case()
    case.image[:] = 0.0
    with paper_style():
        fig = plot_qualitative_panel([case])
    plt.close(fig)


def test_qualitative_panel_validates_before_drawing_anything() -> None:
    """A bad input must not leave a half-drawn figure the caller might save."""
    bad = _case("BAD")
    bad.ground_truth = np.zeros((2, 2, 2), dtype=np.uint8)
    open_before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="BAD"):
        plot_qualitative_panel([_case("GOOD"), bad])
    assert len(plt.get_fignums()) == open_before


# --------------------------------------------------------------------------- #
# Metric distributions
# --------------------------------------------------------------------------- #
def _per_case_table(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 20
    data = {}
    for region in ("WT", "TC", "ET"):
        data[f"dice_{region}"] = rng.uniform(0.5, 1.0, n)
        data[f"hd95_{region}"] = rng.uniform(1.0, 20.0, n)
    return pd.DataFrame(data, index=[f"case_{i:03d}" for i in range(n)])


def test_metric_distributions_draws_one_box_per_model_and_region() -> None:
    tables = {"U-Net": _per_case_table(0), "NeuroVision-X": _per_case_table(1)}
    with paper_style():
        fig = plot_metric_distributions(tables, metric="dice", ylim=(0.0, 1.0))
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == 2 * 3  # 2 models x 3 regions
        assert [t.get_text() for t in ax.get_xticklabels()] == ["WT", "TC", "ET"]
    finally:
        plt.close(fig)


def test_metric_distributions_names_a_missing_column() -> None:
    tables = {"U-Net": _per_case_table(0).drop(columns=["dice_ET"])}
    with pytest.raises(ValueError, match="dice_ET"):
        plot_metric_distributions(tables, metric="dice")


def test_metric_distributions_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        plot_metric_distributions({})


def test_metric_distributions_tolerates_all_nan_hd95() -> None:
    """HD95 is legitimately NaN when exactly one side of a region is empty."""
    table = _per_case_table(0)
    table["hd95_ET"] = np.nan
    with paper_style():
        fig = plot_metric_distributions({"U-Net": table}, metric="hd95")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Reliability diagram
# --------------------------------------------------------------------------- #
def _reliability_table(overconfident: bool) -> pd.DataFrame:
    lower = np.linspace(0.0, 0.9, 10)
    upper = lower + 0.1
    mean_prob = (lower + upper) / 2
    mean_label = mean_prob * (0.8 if overconfident else 1.0)
    return pd.DataFrame(
        {
            "bin_lower": lower,
            "bin_upper": upper,
            "count": np.full(10, 100.0),
            "mean_prob": mean_prob,
            "mean_label": mean_label,
            "gap": mean_label - mean_prob,
        }
    )


def test_reliability_diagram_builds_with_counts_and_ece() -> None:
    curves = {"U-Net": _reliability_table(True), "NeuroVision-X": _reliability_table(False)}
    with paper_style():
        fig = plot_reliability_diagram(curves, ece={"U-Net": 0.031, "NeuroVision-X": 0.008})
    try:
        assert len(fig.axes) == 2  # curve panel + count panel
        legend_text = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
        assert "ECE 0.0310" in legend_text
    finally:
        plt.close(fig)


def test_reliability_diagram_without_counts_is_one_axes() -> None:
    with paper_style():
        fig = plot_reliability_diagram({"m": _reliability_table(True)}, show_counts=False)
    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_reliability_diagram_rejects_a_wrong_shaped_table() -> None:
    with pytest.raises(ValueError, match="missing column"):
        plot_reliability_diagram({"m": pd.DataFrame({"mean_prob": [0.5]})})


def test_reliability_diagram_tolerates_an_all_empty_curve() -> None:
    """`reliability_curve` returns all-NaN bins when a region is absent from a case."""
    table = _reliability_table(True)
    table[["count", "mean_prob", "mean_label", "gap"]] = np.nan
    table["count"] = 0.0
    with paper_style():
        fig = plot_reliability_diagram({"m": table})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Risk-coverage
# --------------------------------------------------------------------------- #
@dataclass
class _FakeCurve:
    coverage: np.ndarray
    performance: np.ndarray
    aurc: float


def _fake_curve(offset: float) -> _FakeCurve:
    coverage = np.linspace(0.1, 1.0, 10)
    performance = 0.95 - offset * coverage
    return _FakeCurve(coverage, performance, aurc=float(offset))


def test_risk_coverage_plots_models_oracle_and_random() -> None:
    with paper_style():
        fig = plot_risk_coverage(
            {"U-Net": _fake_curve(0.1), "NeuroVision-X": _fake_curve(0.05)},
            oracle=_fake_curve(0.02),
            random=_fake_curve(0.0),
        )
    try:
        assert len(fig.axes[0].lines) == 4
        legend_text = " ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
        assert "Oracle" in legend_text and "Random" in legend_text
        assert "AURC 0.1000" in legend_text
    finally:
        plt.close(fig)


def test_risk_coverage_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        plot_risk_coverage({})


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def _history() -> pd.DataFrame:
    steps = np.arange(20)
    loss = np.linspace(1.0, 0.2, 20)
    # Validation logged every 4th step -- the sparse-NaN case that a naive plot
    # would connect straight across.
    dice = np.full(20, np.nan)
    dice[::4] = np.linspace(0.4, 0.9, 5)
    return pd.DataFrame({"_step": steps, "train/loss_epoch": loss, "val/dice_mean": dice})


def test_training_curves_drops_nan_rows_per_column() -> None:
    panels = [
        TrainingPanel("Loss", ["train/loss_epoch"], "DiceBCE loss"),
        TrainingPanel("Validation Dice", ["val/dice_mean"], "Dice", ylim=(0.0, 1.0)),
    ]
    with paper_style():
        fig = plot_training_curves({"U-Net": _history()}, panels)
    try:
        assert len(fig.axes) == 2
        assert len(fig.axes[0].lines[0].get_xdata()) == 20
        # 5 real validation points, not 20 with interpolated gaps.
        assert len(fig.axes[1].lines[0].get_xdata()) == 5
    finally:
        plt.close(fig)


def test_training_curves_warns_and_continues_on_a_missing_column(caplog) -> None:
    panels = [TrainingPanel("Loss", ["train/loss_epoch", "train/absent"], "loss")]
    with caplog.at_level(logging.WARNING), paper_style():
        fig = plot_training_curves({"U-Net": _history()}, panels)
    try:
        assert "train/absent" in caplog.text
        assert len(fig.axes[0].lines) == 1
    finally:
        plt.close(fig)


def test_training_curves_rejects_a_missing_x_column() -> None:
    panels = [TrainingPanel("Loss", ["train/loss_epoch"], "loss")]
    with pytest.raises(ValueError, match="no '_step' column"):
        plot_training_curves({"U-Net": _history().drop(columns=["_step"])}, panels)


def test_training_curves_rejects_mismatched_labels() -> None:
    panels = [TrainingPanel("Loss", ["a", "b"], "loss", labels=["only one"])]
    with pytest.raises(ValueError, match="but 1 labels"):
        plot_training_curves({"U-Net": _history()}, panels)


def test_training_curves_prefixes_the_run_name_when_comparing_runs() -> None:
    panels = [TrainingPanel("Loss", ["train/loss_epoch"], "loss")]
    with paper_style():
        fig = plot_training_curves({"A": _history(), "B": _history()}, panels)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any(label.startswith("A ") for label in labels)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Comparison forest
# --------------------------------------------------------------------------- #
def _comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "improvement": [0.02, -0.01, 0.004],
            "improvement_lo": [0.01, -0.03, -0.002],
            "improvement_hi": [0.03, -0.002, 0.010],
            "verdict": ["better", "worse", "inconclusive"],
        },
        index=["dice_WT", "hd95_ET", "dice_ET"],
    )


def test_comparison_forest_draws_a_row_per_metric_in_table_order() -> None:
    with paper_style():
        fig = plot_comparison_forest(_comparison_table(), name_a="Ours", name_b="U-Net")
    try:
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["dice_WT", "hd95_ET", "dice_ET"]
        assert "Ours" in ax.get_xlabel()
    finally:
        plt.close(fig)


def test_comparison_forest_greys_out_inconclusive_rows() -> None:
    """Inconclusive differences must not be claimed, so they must not look like results."""
    with paper_style():
        fig = plot_comparison_forest(_comparison_table())
    try:
        colors = {line.get_color() for line in fig.axes[0].lines}
        assert "#999999" in colors
    finally:
        plt.close(fig)


def test_comparison_forest_can_subset_metrics() -> None:
    with paper_style():
        fig = plot_comparison_forest(_comparison_table(), metrics=["dice_ET", "dice_WT"])
    try:
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["dice_ET", "dice_WT"]
    finally:
        plt.close(fig)


def test_comparison_forest_rejects_an_unknown_metric() -> None:
    with pytest.raises(ValueError, match="not in the table index"):
        plot_comparison_forest(_comparison_table(), metrics=["dice_NOPE"])


def test_comparison_forest_rejects_a_foreign_table() -> None:
    with pytest.raises(ValueError, match="missing column"):
        plot_comparison_forest(pd.DataFrame({"mean_diff": [0.1]}, index=["dice_WT"]))


# --------------------------------------------------------------------------- #
# Dependency guard
# --------------------------------------------------------------------------- #
def test_figures_module_does_not_import_torch() -> None:
    """Keeps figure tests fast and the module importable without a DL stack.

    Checked against the source text rather than `sys.modules`, because pytest
    has almost certainly imported torch already for some other test file.
    """
    source = Path(figures.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import monai" not in source
