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

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from neurovision.visualization import figures  # noqa: E402
from neurovision.visualization.figures import (  # noqa: E402
    MODEL_COLOR_CYCLE,
    AttributionCase,
    GateCase,
    QualitativeCase,
    TrainingPanel,
    model_style,
    paper_rc,
    paper_style,
    pick_slice,
    plot_attribution_panel,
    plot_band_profile,
    plot_comparison_forest,
    plot_gate_maps,
    plot_metric_distributions,
    plot_modality_attribution,
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


def test_region_colors_are_keyed_in_reporting_order() -> None:
    """Pins the WT/TC/ET reporting order against the ET/TC/WT channel order.

    `REGION_COLORS` is consumed by notebooks rather than by this module, so a
    swapped key order would recolour every per-region figure with nothing in the
    package failing.
    """
    assert tuple(figures.REGION_COLORS) == figures.REGION_ORDER
    assert figures.REGION_ORDER == ("WT", "TC", "ET")
    assert len(set(figures.REGION_COLORS.values())) == 3


def test_palettes_exclude_okabe_ito_yellow() -> None:
    """#F0E442 is a fine fill but illegible as a 1pt line on white."""
    assert "#F0E442" not in set(figures.REGION_COLORS.values()) | set(MODEL_COLOR_CYCLE)


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


def test_qualitative_panel_rejects_disagreeing_uncertainty_labels() -> None:
    """One column header cannot describe two different quantities.

    MC-dropout mutual information is epistemic; the predictive entropy of a
    single deterministic pass is not. Taking the header from whichever case came
    first would label the other row as a measurement that was never made — the
    same positional-mislabeling bug the prediction-key check exists to stop.
    """
    a = _case("A", with_uncertainty=True)
    b = _case("B", with_uncertainty=True)
    a.uncertainty_label = "MC-dropout MI"
    b.uncertainty_label = "Entropy (1 pass)"
    with pytest.raises(ValueError, match="uncertainty_label"):
        plot_qualitative_panel([a, b])


def test_qualitative_panel_accepts_matching_uncertainty_labels() -> None:
    a = _case("A", with_uncertainty=True)
    b = _case("B", with_uncertainty=True)
    with paper_style():
        fig = plot_qualitative_panel([a, b])
    try:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert "MC-dropout MI" in titles
    finally:
        plt.close(fig)


def test_qualitative_panel_warns_about_an_unmatched_annotation_key(caplog) -> None:
    """A typo'd model label would otherwise just be a missing caption in the PDF."""
    case = _case("A")
    case.annotations = {"Basline": "Dice 0.91"}  # codespell:ignore
    with caplog.at_level(logging.WARNING), paper_style():
        fig = plot_qualitative_panel([case])
    try:
        assert "Basline" in caplog.text  # codespell:ignore
        assert "matching no column" in caplog.text
    finally:
        plt.close(fig)


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


def test_comparison_forest_warns_on_an_unrecognized_verdict(caplog) -> None:
    """Grey is safe, but silently greying would hide an upstream compare_models bug."""
    table = _comparison_table()
    table.loc["dice_ET", "verdict"] = "maybe"
    with caplog.at_level(logging.WARNING), paper_style():
        fig = plot_comparison_forest(table)
    try:
        assert "maybe" in caplog.text
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


# --------------------------------------------------------------------------- #
# Gate maps
# --------------------------------------------------------------------------- #
def _gate_case(case_id: str = "G_A", *, gates: dict | None = None) -> GateCase:
    if gates is None:
        rng = np.random.default_rng(0)
        gates = {
            "stride2": rng.uniform(0.4, 0.6, size=(6, 7, 8)).astype(np.float32),
            "stride4": rng.uniform(0.4, 0.6, size=(3, 4, 4)).astype(np.float32),
        }
    return GateCase(
        case_id=case_id,
        image=_synthetic_image(),
        ground_truth=_synthetic_label(),
        gates=gates,
    )


def test_gate_case_validate_accepts_a_well_formed_case() -> None:
    _gate_case().validate()  # must not raise


def test_gate_case_validate_rejects_an_oversized_gate_map() -> None:
    case = _gate_case()
    case.gates["stride2"] = np.zeros((13, 7, 8), dtype=np.float32)  # SHAPE is (12, 14, 16)
    with pytest.raises(ValueError, match="G_A.*stride2"):
        case.validate()


def test_gate_case_validate_rejects_empty_gates() -> None:
    case = _gate_case()
    case.gates = {}
    with pytest.raises(ValueError, match="gates.* is empty"):
        case.validate()


def test_gate_maps_has_one_axes_per_grid_cell() -> None:
    with paper_style():
        fig = plot_gate_maps([_gate_case("A"), _gate_case("B")])
    try:
        # 2 rows x (modality + 2 gate keys) = 6 image-bearing axes.
        assert len(_grid_axes(fig)) == 6
    finally:
        plt.close(fig)


def test_gate_maps_rejects_differing_gate_keys() -> None:
    a = _gate_case("A")
    b = _gate_case("B")
    b.gates = {"stride2": b.gates["stride2"]}
    with pytest.raises(ValueError, match="gate keys"):
        plot_gate_maps([a, b])


def test_gate_maps_rejects_reordered_gate_keys() -> None:
    a = _gate_case("A")
    b = _gate_case("B")
    b.gates = {"stride4": b.gates["stride4"], "stride2": b.gates["stride2"]}
    with pytest.raises(ValueError, match="gate keys"):
        plot_gate_maps([a, b])


def test_gate_maps_upsample_is_nearest_neighbour_only() -> None:
    """Pins the "do not imply voxel precision" decision.

    A coarse gate split sharply into two halves must render as exactly two
    values -- any interpolated intermediate would be a value the data never
    had.
    """
    coarse = np.full((4, 4, 4), 0.2, dtype=np.float32)
    coarse[2:] = 0.8  # split along an axis orthogonal to the (axial) plane axis
    case = _gate_case("A", gates={"stride8": coarse})
    case.slice_index = 6
    with paper_style():
        fig = plot_gate_maps([case], crop_to_brain=False)
    try:
        images = _grid_axes(fig)
        drawn = images[1].get_images()[0].get_array()
        assert np.all(np.isclose(drawn, 0.2) | np.isclose(drawn, 0.8))
        assert np.any(np.isclose(drawn, 0.2))
        assert np.any(np.isclose(drawn, 0.8))
    finally:
        plt.close(fig)


def test_gate_maps_colour_limits_are_fixed_to_zero_one() -> None:
    """A gate hovering near 0.5 must not be rescaled to look dramatic."""
    rng = np.random.default_rng(1)
    gates = {"stride8": rng.uniform(0.49, 0.51, size=(3, 4, 4)).astype(np.float32)}
    with paper_style():
        fig = plot_gate_maps([_gate_case("A", gates=gates)])
    try:
        for ax in _grid_axes(fig)[1:]:
            image = ax.get_images()[0]
            assert image.get_clim() == (0.0, 1.0)
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Band profile
# --------------------------------------------------------------------------- #
_BANDS = ("0-2", "2-5", "5-10", "10-inf")


def _band_table(values: list, errors: list | None = None) -> pd.DataFrame:
    data = {"band": list(_BANDS), "mean": values}
    if errors is not None:
        data["std"] = errors
    return pd.DataFrame(data)


def test_band_profile_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        plot_band_profile({})


def test_band_profile_names_a_missing_column() -> None:
    table = _band_table([0.1, 0.2, 0.3, 0.4])  # no "std" column: errors=None
    with pytest.raises(ValueError, match="'std'"):
        plot_band_profile({"m": table})


def test_band_profile_rejects_mismatched_band_labels() -> None:
    a = _band_table([0.1, 0.2, 0.3, 0.4], errors=[0.0] * 4)
    b = a.copy()
    b["band"] = ["0-2", "2-5", "5-10", "10-99"]
    with pytest.raises(ValueError, match="has bands"):
        plot_band_profile({"A": a, "B": b})


def test_band_profile_rejects_reordered_band_labels() -> None:
    a = _band_table([0.1, 0.2, 0.3, 0.4], errors=[0.0] * 4)
    b = a.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="has bands"):
        plot_band_profile({"A": a, "B": b})


def test_band_profile_xticks_match_band_labels_in_order() -> None:
    table = _band_table([0.1, 0.2, 0.3, 0.4], errors=[0.02] * 4)
    with paper_style():
        fig = plot_band_profile({"m": table})
    try:
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == list(_BANDS)
    finally:
        plt.close(fig)


def test_band_profile_draws_no_fill_without_an_error_column() -> None:
    table = _band_table([0.1, 0.2, 0.3, 0.4])
    with paper_style():
        fig = plot_band_profile({"m": table}, error_column=None)
    try:
        assert len(fig.axes[0].collections) == 0
    finally:
        plt.close(fig)


def test_band_profile_draws_a_fill_with_an_error_column() -> None:
    table = _band_table([0.1, 0.2, 0.3, 0.4], errors=[0.02] * 4)
    with paper_style():
        fig = plot_band_profile({"m": table})
    try:
        assert len(fig.axes[0].collections) == 1
    finally:
        plt.close(fig)


def test_band_profile_higher_is_better_annotates_ylabel() -> None:
    table = _band_table([0.1, 0.2, 0.3, 0.4])
    with paper_style():
        up = plot_band_profile(
            {"m": table}, ylabel="Gate value", higher_is_better=True, error_column=None
        )
        down = plot_band_profile(
            {"m": table}, ylabel="Error rate", higher_is_better=False, error_column=None
        )
    try:
        assert "better" in up.axes[0].get_ylabel()
        assert "better" in down.axes[0].get_ylabel()
    finally:
        plt.close(up)
        plt.close(down)


# --------------------------------------------------------------------------- #
# Modality attribution
# --------------------------------------------------------------------------- #
def _modality_attribution_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["ET", "TC", "WT"],
            "attr_T1": [0.1, 0.15, 0.2],
            "attr_T1CE": [0.5, 0.3, 0.2],
            "attr_T2": [0.1, 0.2, 0.2],
            "attr_FLAIR": [0.3, 0.35, 0.4],
        }
    )


def test_modality_attribution_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        plot_modality_attribution(pd.DataFrame())


def test_modality_attribution_names_a_missing_column() -> None:
    df = _modality_attribution_df().drop(columns=["attr_FLAIR"])
    with pytest.raises(ValueError, match="attr_FLAIR"):
        plot_modality_attribution(df)


def test_modality_attribution_averages_rows_sharing_a_region() -> None:
    df = pd.DataFrame(
        {
            "region": ["ET", "ET"],
            "attr_T1": [0.2, 0.6],
            "attr_T1CE": [0.5, 0.5],
            "attr_T2": [0.1, 0.1],
            "attr_FLAIR": [0.2, 0.2],
        }
    )
    with paper_style():
        fig = plot_modality_attribution(df)
    try:
        ax = fig.axes[0]
        # One region -> one bar per modality's `ax.bar()` call, so `ax.patches`
        # is in MODALITY_NAMES order; T1 is first.
        assert ax.patches[0].get_height() == pytest.approx(0.4)
    finally:
        plt.close(fig)


def test_modality_attribution_raises_on_integer_region_values() -> None:
    """Integer channel indices are ambiguous here and must never be guessed.

    This module's REGION_ORDER is ("WT", "TC", "ET") while the model's channel
    order -- what scripts/explain.py writes into modality_attribution.csv -- is
    ("ET", "TC", "WT"). Index 0 is therefore ET in the file and WT here, and
    either mapping renders a perfectly plausible chart with every bar
    relabelled. Same reasoning as analysis.statistics.metric_direction raising
    on an unknown metric.
    """
    df = pd.DataFrame(
        {
            "region": [0, 2],
            "attr_T1": [0.1, 0.1],
            "attr_T1CE": [0.2, 0.5],
            "attr_T2": [0.1, 0.1],
            "attr_FLAIR": [0.4, 0.2],
        }
    )
    with pytest.raises(ValueError, match="integer channel indices"):
        plot_modality_attribution(df)


def test_modality_attribution_uses_named_regions_verbatim() -> None:
    df = pd.DataFrame(
        {
            "region": ["ET", "WT"],
            "attr_T1": [0.1, 0.1],
            "attr_T1CE": [0.5, 0.2],
            "attr_T2": [0.1, 0.1],
            "attr_FLAIR": [0.2, 0.4],
        }
    )
    with paper_style():
        fig = plot_modality_attribution(df)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        # REGION_ORDER is the drawing order, so WT precedes ET.
        assert labels == ["WT", "ET"]
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Attribution panel
# --------------------------------------------------------------------------- #
def _attribution_case(case_id: str = "A", *, maps: dict | None = None) -> AttributionCase:
    if maps is None:
        rng = np.random.default_rng(0)
        maps = {
            "IG": rng.normal(0.0, 1.0, size=SHAPE).astype(np.float32),
            "GradCAM": np.abs(rng.normal(0.0, 1.0, size=SHAPE)).astype(np.float32),
        }
    return AttributionCase(
        case_id=case_id,
        image=_synthetic_image(),
        ground_truth=_synthetic_label(),
        maps=maps,
        region_label="ET",
    )


def test_attribution_case_validate_accepts_a_well_formed_case() -> None:
    _attribution_case().validate()  # must not raise


def test_attribution_case_validate_rejects_a_shape_mismatch() -> None:
    """The geometry trap: a map saved at patch resolution overlaid on the whole volume."""
    case = _attribution_case()
    case.maps["IG"] = np.zeros((32, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="A.*IG"):
        case.validate()


def test_attribution_case_validate_rejects_empty_maps() -> None:
    case = _attribution_case()
    case.maps = {}
    with pytest.raises(ValueError, match="maps.* is empty"):
        case.validate()


def test_attribution_panel_has_one_axes_per_grid_cell() -> None:
    with paper_style():
        fig = plot_attribution_panel([_attribution_case("A"), _attribution_case("B")])
    try:
        # 2 rows x (modality + 2 maps) = 6 image-bearing axes.
        assert len(_grid_axes(fig)) == 6
    finally:
        plt.close(fig)


def test_attribution_panel_rejects_differing_map_keys() -> None:
    a = _attribution_case("A")
    b = _attribution_case("B")
    b.maps = {"IG": b.maps["IG"]}
    with pytest.raises(ValueError, match="map keys"):
        plot_attribution_panel([a, b])


def test_attribution_panel_signed_key_gets_symmetric_limits() -> None:
    maps = {
        "IG": np.full(SHAPE, -2.0, dtype=np.float32),
        "GradCAM": np.full(SHAPE, 1.5, dtype=np.float32),
    }
    maps["IG"][0, 0, 0] = 3.0  # asymmetric raw range: min -2, max 3
    case = _attribution_case("A", maps=maps)
    case.slice_index = 0  # the axial slice that actually contains voxel (0, 0, 0)
    with paper_style():
        fig = plot_attribution_panel([case], signed_keys=("IG",), crop_to_brain=False)
    try:
        images = _grid_axes(fig)
        ig_vmin, ig_vmax = images[1].get_images()[0].get_clim()
        assert ig_vmin == -ig_vmax
        assert ig_vmax == pytest.approx(3.0)

        cam_vmin, cam_vmax = images[2].get_images()[0].get_clim()
        assert cam_vmin == 0.0
        assert cam_vmax != -cam_vmin
    finally:
        plt.close(fig)


def test_modality_attribution_hatch_is_visible_against_the_bar() -> None:
    """The expected-modality hatch must not be drawn in the bar's own face colour.

    Matplotlib renders hatching in the EDGE colour, so edge == face paints the
    marks invisibly -- the artist still reports a hatch (so a `get_hatch()`
    assertion passes) while the rendered figure shows nothing, and the legend
    goes on advertising a mark that is not there. Found by looking at the PNG,
    which is the only way this class of bug surfaces.
    """
    df = pd.DataFrame(
        {
            "region": ["ET", "WT"],
            "attr_T1": [0.1, 0.1],
            "attr_T1CE": [0.5, 0.2],
            "attr_T2": [0.1, 0.1],
            "attr_FLAIR": [0.3, 0.6],
        }
    )
    with paper_style():
        fig = plot_modality_attribution(df)
    try:
        hatched = [p for p in fig.axes[0].patches if p.get_hatch()]
        assert len(hatched) == 2, "one bar per region should carry the expected-modality hatch"
        for patch in hatched:
            face = mcolors.to_rgba(patch.get_facecolor())
            edge = mcolors.to_rgba(patch.get_edgecolor())
            assert (
                face[:3] != edge[:3]
            ), "hatch edge colour equals the bar face colour, so the hatch renders invisibly"
    finally:
        plt.close(fig)


def test_modality_attribution_reserves_headroom_for_its_legend() -> None:
    """The legend must not land on top of the tallest bar group.

    `loc="best"` can only avoid the data if there is somewhere to go, so the y
    limit is expanded before the legend is placed.
    """
    df = pd.DataFrame(
        {
            "region": ["ET"],
            "attr_T1": [0.1],
            "attr_T1CE": [0.6],
            "attr_T2": [0.1],
            "attr_FLAIR": [0.2],
        }
    )
    with paper_style():
        fig = plot_modality_attribution(df)
    try:
        ax = fig.axes[0]
        tallest = max(p.get_height() for p in ax.patches)
        assert ax.get_ylim()[1] > tallest * 1.15
    finally:
        plt.close(fig)
