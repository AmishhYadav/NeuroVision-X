"""Publication-quality figure helpers for the paper.

Companion to `qc.py`. The two modules have different jobs: `qc.py` makes figures
a human squints at to catch a broken pipeline, so it optimizes for legibility at
any cost. This module makes figures that go into a PDF submission, so it
optimizes for a consistent house style, a colour-vision-safe palette, and vector
output that survives a publisher's production pipeline.

**This module never READS files.** Every function takes arrays or DataFrames and
returns a `matplotlib.figure.Figure`. The single exception is `save_figure`,
which only WRITES a figure the caller already built. All input IO belongs to the
calling notebook (`notebooks/09_paper_figures.ipynb`), which is what makes every
function here testable on tiny synthetic arrays in milliseconds.

Dependency-light on purpose: numpy, pandas and matplotlib only. **No torch, no
monai.** Importing torch here would drag a ~2 s import and a CUDA probe into
every figure test and into any docs build. Where a function needs to accept an
object defined in a torch-importing module (`uncertainty.risk_coverage`), it is
typed against a structural `Protocol` instead of importing the real class.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from neurovision.visualization.qc import CLASS_COLORS, CLASS_NAMES, MODALITY_NAMES

logger = logging.getLogger(__name__)

# BraTS papers report the nested regions outermost-to-innermost (WT, TC, ET),
# which is the REVERSE of the (ET, TC, WT) channel order the model, the loss and
# `metrics.segmentation` use internally. Both orders are correct in their own
# context and neither is inherently "the" order -- which is exactly why getting
# it backwards silently mislabels every column of every table with no error
# anywhere. This constant is the reporting order and nothing else.
REGION_ORDER: tuple[str, str, str] = ("WT", "TC", "ET")

# Okabe-Ito blue / orange / bluish-green. Chosen because the three stay mutually
# distinguishable under deuteranopia, protanopia and tritanopia -- roughly 1 in
# 12 male readers, which for a reviewer pool is a near-certainty rather than an
# edge case. Okabe-Ito yellow (#F0E442) is deliberately excluded from every
# palette in this module: it is a fine fill colour but illegible as a 1pt line
# on white.
REGION_COLORS: dict[str, str] = {"WT": "#0072B2", "TC": "#E69F00", "ET": "#009E73"}

# Colour AND linestyle both vary across models. A figure distinguished by colour
# alone becomes unreadable the moment it is printed in greyscale or photocopied,
# which is still how a lot of people read papers.
MODEL_COLOR_CYCLE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
)
MODEL_LINESTYLE_CYCLE: tuple[Any, ...] = (
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2)),
)

# Verdict colours for the comparison forest plot. Grey for the two verdicts that
# must NOT be claimed in the paper, so an inconclusive row is visually inert
# rather than looking like a result.
_VERDICT_COLORS: dict[str, str] = {
    "better": "#009E73",
    "worse": "#D55E00",
    "negligible": "#999999",
    "inconclusive": "#999999",
}

_ALLOWED_SAVE_FORMATS: frozenset[str] = frozenset({"pdf", "svg", "png", "eps"})

_PLANE_AXES: dict[str, int] = {"sagittal": 0, "coronal": 1, "axial": 2}


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def paper_rc(base_font_size: float = 8.0) -> dict[str, Any]:
    """Build the project's matplotlib rcParams for paper figures.

    Does not mutate global state -- callers apply it via `use_paper_style` or
    the `paper_style` context manager.

    Args:
        base_font_size: Body font size in points. Every other size is derived
            from it, so one number rescales the whole figure's typography.

    Returns:
        A dict suitable for `plt.rcParams.update`.
    """
    return {
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Type 42 embeds TrueType outlines. Matplotlib's DEFAULT is type 3,
        # which many publishers reject outright and which cannot be edited in
        # Illustrator. This is invisible until a submission bounces, so it is
        # pinned here and regression-tested.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # No concrete font family is named on purpose. Naming one that exists on
        # the dev Mac but not on a Linux box makes the same script silently emit
        # two different-looking figures.
        "font.family": "sans-serif",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        # Label maps must never be smoothed by the renderer: interpolation
        # invents intermediate values, which for a class map means inventing
        # voxel classes that are not in the data.
        "image.interpolation": "nearest",
        "mathtext.fontset": "dejavusans",
    }


def use_paper_style(base_font_size: float = 8.0) -> None:
    """Apply `paper_rc` to the global rcParams. One-liner for a notebook.

    Args:
        base_font_size: See `paper_rc`.
    """
    plt.rcParams.update(paper_rc(base_font_size))


@contextlib.contextmanager
def paper_style(base_font_size: float = 8.0) -> Iterator[None]:
    """Temporarily apply the paper style, restoring rcParams on exit.

    Restores even when the body raises. Use this in tests so a test can never
    leak style state into the next one.

    Args:
        base_font_size: See `paper_rc`.

    Yields:
        None.
    """
    with mpl.rc_context(rc=paper_rc(base_font_size)):
        yield


def model_style(index: int) -> dict[str, Any]:
    """Return the colour and linestyle for the `index`-th model in a figure.

    Args:
        index: Zero-based position of the model in the figure's model order.
            Cycles modulo the palette length.

    Returns:
        `{"color": str, "linestyle": ...}`, ready to splat into a plot call.

    Raises:
        ValueError: `index` is negative. Python's negative indexing would wrap
            to a DIFFERENT model's colour rather than erroring, silently giving
            two models the same style in one legend.
    """
    if index < 0:
        raise ValueError(f"model_style: index must be >= 0, got {index}.")
    return {
        "color": MODEL_COLOR_CYCLE[index % len(MODEL_COLOR_CYCLE)],
        "linestyle": MODEL_LINESTYLE_CYCLE[index % len(MODEL_LINESTYLE_CYCLE)],
    }


def save_figure(
    fig: Figure,
    out_dir: str | Path,
    stem: str,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int | None = None,
    close: bool = False,
) -> list[Path]:
    """Write one figure to disk in several formats under a shared stem.

    PDF comes first and is the default because it is the vector format LaTeX
    wants -- text stays selectable and lines stay sharp at any zoom. PNG is the
    preview that renders in a notebook, a README or a GitHub diff.

    Args:
        fig: The figure to write.
        out_dir: Destination directory, created if missing.
        stem: Filename without extension.
        formats: Extensions to write, in order.
        dpi: Raster resolution. `None` uses rcParams `savefig.dpi`.
        close: Close `fig` afterwards. Useful in a loop that builds many
            figures, so the notebook does not accumulate open figures.

    Returns:
        The written paths, in the same order as `formats`.

    Raises:
        ValueError: `stem` is empty, contains a path separator, or contains a
            `"."` (a stem like `"fig1.v2"` would produce `fig1.v2.pdf`, which
            reads as a different file to anyone globbing for it); or `formats`
            is empty or names an unsupported extension.
    """
    if not stem:
        raise ValueError("save_figure: `stem` must be a non-empty filename without extension.")
    if os.sep in stem or (os.altsep and os.altsep in stem) or "/" in stem:
        raise ValueError(
            f"save_figure: `stem` must be a bare filename, got {stem!r}. Pass directories via "
            "`out_dir`."
        )
    if "." in stem:
        raise ValueError(
            f"save_figure: `stem` must not contain '.', got {stem!r} -- the extension is added "
            "from `formats`."
        )
    if not formats:
        raise ValueError("save_figure: `formats` must name at least one format.")

    unknown = [f for f in formats if f.lower().lstrip(".") not in _ALLOWED_SAVE_FORMATS]
    if unknown:
        raise ValueError(
            f"save_figure: unsupported format(s) {unknown}; allowed: "
            f"{sorted(_ALLOWED_SAVE_FORMATS)}."
        )

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        ext = fmt.lower().lstrip(".")
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi)
        written.append(path)

    logger.info("Wrote figure %r: %s", stem, ", ".join(str(p) for p in written))
    if close:
        plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Slice selection
# --------------------------------------------------------------------------- #
def _validate_plane(plane: str, func_name: str) -> int:
    """Return the array axis a named anatomical plane indexes along."""
    if plane not in _PLANE_AXES:
        raise ValueError(
            f"{func_name}: unknown plane {plane!r}; expected one of {sorted(_PLANE_AXES)}."
        )
    return _PLANE_AXES[plane]


def pick_slice(ground_truth: np.ndarray, plane: str = "axial") -> int:
    """Choose the display slice: the one with the most ground-truth foreground.

    Deterministic by construction -- ties resolve to the LOWEST index, so the
    same inputs always produce the same figure. A figure that silently changes
    between two runs of the notebook is not reproducible even if every number
    in it is.

    Axis convention matches `qc.extract_mid_slices`: sagittal indexes axis 0,
    coronal axis 1, axial axis 2, on a `(D, H, W)` array.

    Args:
        ground_truth: `(D, H, W)` integer class map; anything `> 0` counts as
            foreground.
        plane: `"sagittal"`, `"coronal"` or `"axial"`.

    Returns:
        The chosen index along `plane`'s axis.

    Raises:
        ValueError: `ground_truth` is not 3D, or `plane` is unknown.
    """
    if ground_truth.ndim != 3:
        raise ValueError(
            f"pick_slice: expected a (D, H, W) volume, got shape {ground_truth.shape}."
        )
    axis = _validate_plane(plane, "pick_slice")

    foreground = np.asarray(ground_truth) > 0
    other_axes = tuple(a for a in range(3) if a != axis)
    counts = foreground.sum(axis=other_axes)

    if int(counts.sum()) == 0:
        midpoint = ground_truth.shape[axis] // 2
        logger.warning(
            "pick_slice: volume has zero foreground on the %s plane; falling back to the "
            "midpoint slice %d. The figure will show anatomy with no label overlay.",
            plane,
            midpoint,
        )
        return midpoint

    # argmax returns the FIRST maximum, which is exactly the lowest-index
    # tie-break this function promises.
    return int(np.argmax(counts))


def take_slice(volume: np.ndarray, index: int, plane: str = "axial") -> np.ndarray:
    """Extract one 2D slice, rotated for upright display.

    The `np.rot90` is display-only and matches `qc.extract_mid_slices` exactly,
    so a QC figure and a paper figure of the same case are in the same
    orientation. Two modules disagreeing about this would make a reader think
    the preprocessing changed.

    Args:
        volume: `(D, H, W)` array.
        index: Index along `plane`'s axis.
        plane: `"sagittal"`, `"coronal"` or `"axial"`.

    Returns:
        A 2D array ready to hand to `imshow`.

    Raises:
        ValueError: `volume` is not 3D, `plane` is unknown, or `index` is out of
            range for that axis.
    """
    if volume.ndim != 3:
        raise ValueError(f"take_slice: expected a (D, H, W) volume, got shape {volume.shape}.")
    axis = _validate_plane(plane, "take_slice")

    extent = volume.shape[axis]
    if not 0 <= index < extent:
        raise ValueError(
            f"take_slice: index {index} is out of range for the {plane} axis of length {extent}."
        )

    if axis == 0:
        plane_slice = volume[index, :, :]
    elif axis == 1:
        plane_slice = volume[:, index, :]
    else:
        plane_slice = volume[:, :, index]
    return np.rot90(plane_slice)


# --------------------------------------------------------------------------- #
# Qualitative panel
# --------------------------------------------------------------------------- #
@dataclass
class QualitativeCase:
    """One row of the qualitative panel: a case and every model's take on it.

    Attributes:
        case_id: Identifier, drawn as the row label.
        image: `(4, D, H, W)` float, channel order `MODALITY_NAMES`.
        ground_truth: `(D, H, W)` integer class map in `{0, 1, 2, 3}`.
        predictions: Model display label -> `(D, H, W)` class map. Insertion
            order becomes the column order and must be identical across cases.
        uncertainty: Optional `(D, H, W)` non-negative float map.
        uncertainty_label: Column title for the uncertainty map. Say precisely
            what it is -- "MC-dropout mutual information" and "predictive
            entropy (single pass)" are different quantities and a paper that
            mixes them up is making a claim it did not measure.
        slice_index: Explicit display slice, or `None` to use `pick_slice`.
        annotations: Column title -> short caption drawn inside that cell
            (e.g. `{"NeuroVision-X": "Dice 0.91"}`).
    """

    case_id: str
    image: np.ndarray
    ground_truth: np.ndarray
    predictions: dict[str, np.ndarray]
    uncertainty: np.ndarray | None = None
    uncertainty_label: str = "Uncertainty"
    slice_index: int | None = None
    annotations: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Check every array agrees on geometry, raising with the case id if not.

        This is the only thing standing between a real, easy mistake and a
        figure in the paper. Saved predictions from `scripts/evaluate.py` are in
        ORIGINAL BraTS geometry (240x240x155) because that is what a submission
        requires, while the preprocessed image and label are CROPPED to the
        case's nonzero bounding box. Overlaying them without re-cropping
        produces a figure that is misaligned by the crop offset and still looks
        entirely plausible.

        Raises:
            ValueError: Any shape disagreement, or an empty `predictions`.
        """
        cid = self.case_id
        if self.image.ndim != 4 or self.image.shape[0] != len(MODALITY_NAMES):
            raise ValueError(
                f"{cid}: `image` must have shape (4, D, H, W) with channels {MODALITY_NAMES}, "
                f"got {self.image.shape}."
            )
        spatial = tuple(self.image.shape[1:])
        if tuple(self.ground_truth.shape) != spatial:
            raise ValueError(
                f"{cid}: `ground_truth` shape {self.ground_truth.shape} does not match the "
                f"image's spatial shape {spatial}."
            )
        if not self.predictions:
            raise ValueError(f"{cid}: `predictions` is empty; the panel needs at least one model.")
        for label, pred in self.predictions.items():
            if tuple(pred.shape) != spatial:
                raise ValueError(
                    f"{cid}: prediction {label!r} has shape {pred.shape}, expected {spatial}. A "
                    "prediction saved in ORIGINAL geometry must be re-cropped with the case's "
                    "meta.json bbox before it can be overlaid on the CROPPED image."
                )
        if self.uncertainty is not None and tuple(self.uncertainty.shape) != spatial:
            raise ValueError(
                f"{cid}: `uncertainty` has shape {self.uncertainty.shape}, expected {spatial}."
            )


def _normalize01(values: np.ndarray) -> np.ndarray:
    """Min-max a 2D array into [0, 1]; a constant array becomes all zeros."""
    finite = np.asarray(values, dtype=np.float64)
    lo = float(np.nanmin(finite)) if finite.size else 0.0
    hi = float(np.nanmax(finite)) if finite.size else 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(finite)
    return (finite - lo) / (hi - lo)


def _normalize_mri(
    values: np.ndarray, percentiles: tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """Scale an MRI slice for display: robust window, background forced to black.

    Two things a plain min-max gets wrong on this project's data, both visible in
    the figure rather than in any number:

    1. Preprocessing z-scores each modality over its NONZERO voxels, so brain
       interior values are routinely negative. A min-max therefore maps the
       zero-valued air OUTSIDE the head to some mid-grey, and the panel renders
       every brain on a grey card. Voxels that are exactly zero are the
       preprocessing's own background marker, so they are pinned to black after
       scaling.
    2. A single bright voxel (an enhancing rim, a scanner artefact) sets the
       maximum and washes out everything else. The window is therefore taken
       from percentiles of the nonzero voxels, not from the extremes.

    Args:
        values: A 2D slice.
        percentiles: Lower/upper percentile bounds of the display window,
            computed over nonzero voxels only.

    Returns:
        A float array in `[0, 1]`, same shape.
    """
    array = np.asarray(values, dtype=np.float64)
    foreground = array[array != 0]
    if foreground.size == 0:
        # An all-zero slice is legitimately all background (e.g. a slice past
        # the end of the head), not an error.
        return np.zeros_like(array)

    lo, hi = np.percentile(foreground, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Constant foreground: there is no window to compute. Render it as a
        # silhouette rather than falling back to a plain min-max, which would
        # put the MINIMUM at black -- and since the constant foreground can be
        # negative, that minimum is the brain, leaving the background WHITE.
        return (array != 0).astype(np.float64)

    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    scaled[array == 0] = 0.0
    return scaled


def _label_overlay(label_slice: np.ndarray) -> np.ma.MaskedArray:
    """Mask background (class 0) so it renders transparent over the MRI beneath."""
    return np.ma.masked_where(label_slice == 0, label_slice)


def _class_cmap() -> ListedColormap:
    """ListedColormap over classes 1..3, matching `qc.CLASS_COLORS` exactly."""
    return ListedColormap([CLASS_COLORS[k] for k in sorted(CLASS_COLORS)])


def _brain_bbox(background: np.ndarray, pad: int = 4) -> tuple[slice, slice] | None:
    """Bounding box of the nonzero part of a 2D slice, padded and clamped."""
    nonzero = np.nonzero(np.asarray(background) != 0)
    if nonzero[0].size == 0:
        return None
    r0, r1 = int(nonzero[0].min()), int(nonzero[0].max())
    c0, c1 = int(nonzero[1].min()), int(nonzero[1].max())
    rows = slice(max(r0 - pad, 0), min(r1 + 1 + pad, background.shape[0]))
    cols = slice(max(c0 - pad, 0), min(c1 + 1 + pad, background.shape[1]))
    return rows, cols


def _apply_crop(array: np.ndarray, crop: tuple[slice, slice] | None) -> np.ndarray:
    """Apply a shared row crop, or return the array untouched when there is none."""
    return array if crop is None else array[crop[0], crop[1]]


def _blank_axes(ax: plt.Axes) -> None:
    """Strip an axes down to a bare image frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _cell_annotation(ax: plt.Axes, text: str) -> None:
    """Draw a small caption in the lower-left of a cell."""
    ax.text(
        0.03,
        0.03,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=6,
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2, "edgecolor": "none"},
    )


def plot_qualitative_panel(
    cases: Sequence[QualitativeCase],
    *,
    modality: str = "FLAIR",
    plane: str = "axial",
    overlay_alpha: float = 0.45,
    show_gt_contour: bool = True,
    uncertainty_cmap: str = "magma",
    panel_size: float = 1.9,
    crop_to_brain: bool = True,
) -> Figure:
    """The paper's qualitative figure: rows are cases, columns are views.

    Columns, in order: the chosen modality, ground truth, one column per model
    in `predictions` order, and -- only if at least one case supplies it -- an
    uncertainty map.

    Each prediction cell draws the prediction as a filled overlay AND the
    ground-truth whole-tumour boundary as a thin contour. That combination is
    the point of the figure: over- and under-segmentation are readable in a
    single cell, where two side-by-side filled panels force the reader to
    mentally difference them.

    Args:
        cases: One `QualitativeCase` per row. Every case must expose the same
            prediction keys, in the same order.
        modality: Which of `MODALITY_NAMES` to draw as the greyscale backdrop.
        plane: `"sagittal"`, `"coronal"` or `"axial"`.
        overlay_alpha: Opacity of the class overlays.
        show_gt_contour: Draw the ground-truth outline on prediction cells.
        uncertainty_cmap: Colormap for the uncertainty column.
        panel_size: Side length in inches of one cell.
        crop_to_brain: Crop every cell in a row to that row's brain bounding
            box, computed once from the modality slice and applied identically
            across the row (cropping cells independently would misalign the
            columns against each other).

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: `cases` is empty, `modality` is not in `MODALITY_NAMES`,
            any case fails `validate()`, or two cases disagree on their
            prediction keys.
    """
    if not cases:
        raise ValueError("plot_qualitative_panel: `cases` must contain at least one case.")
    if modality not in MODALITY_NAMES:
        raise ValueError(
            f"plot_qualitative_panel: unknown modality {modality!r}; expected one of "
            f"{MODALITY_NAMES}."
        )

    # Validate everything BEFORE drawing anything, so a bad input never leaves a
    # half-drawn figure behind for the caller to mistake for a real one.
    for case in cases:
        case.validate()

    reference = cases[0]
    model_labels = list(reference.predictions)
    for case in cases[1:]:
        if list(case.predictions) != model_labels:
            raise ValueError(
                f"plot_qualitative_panel: case {case.case_id!r} has prediction keys "
                f"{list(case.predictions)} but case {reference.case_id!r} has {model_labels}. "
                "Columns are positional, so a reordered or differing key set would attribute one "
                "model's output to another."
            )

    modality_index = MODALITY_NAMES.index(modality)
    show_uncertainty = any(case.uncertainty is not None for case in cases)

    column_titles = [modality, "Ground truth", *model_labels]
    if show_uncertainty:
        # Take the label from the first case that actually has a map -- an
        # explicit label beats a generic one, and cases without a map do not get
        # to name the column.
        labelled = next(c for c in cases if c.uncertainty is not None)
        column_titles.append(labelled.uncertainty_label)

    n_rows = len(cases)
    n_cols = len(column_titles)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    cmap = _class_cmap()
    # vmin/vmax pinned to the class range so a case missing class 3 does not
    # get its remaining classes recoloured by matplotlib's autoscaling.
    class_lo, class_hi = min(CLASS_COLORS), max(CLASS_COLORS)

    for row, case in enumerate(cases):
        if case.slice_index is not None:
            index = case.slice_index
        else:
            index = pick_slice(case.ground_truth, plane)
        background = _normalize_mri(take_slice(case.image[modality_index], index, plane))
        gt_slice = take_slice(case.ground_truth, index, plane)

        crop: tuple[slice, slice] | None = None
        if crop_to_brain:
            crop = _brain_bbox(background)

        background_view = _apply_crop(background, crop)
        gt_view = _apply_crop(gt_slice, crop)

        # Column 0 -- the modality alone, no overlay. The reader needs one cell
        # per row showing what the model actually saw.
        ax = axes[row][0]
        _blank_axes(ax)
        ax.imshow(background_view, cmap="gray", vmin=0.0, vmax=1.0, rasterized=True)
        ax.set_ylabel(case.case_id, fontsize=7, rotation=90, labelpad=3)
        if column_titles[0] in case.annotations:
            _cell_annotation(ax, case.annotations[column_titles[0]])

        # Column 1 -- ground truth.
        ax = axes[row][1]
        _blank_axes(ax)
        ax.imshow(background_view, cmap="gray", vmin=0.0, vmax=1.0, rasterized=True)
        ax.imshow(
            _label_overlay(gt_view),
            cmap=cmap,
            vmin=class_lo,
            vmax=class_hi,
            alpha=overlay_alpha,
            rasterized=True,
        )
        if column_titles[1] in case.annotations:
            _cell_annotation(ax, case.annotations[column_titles[1]])

        # Model columns.
        for offset, label in enumerate(model_labels):
            ax = axes[row][2 + offset]
            _blank_axes(ax)
            pred_view = _apply_crop(take_slice(case.predictions[label], index, plane), crop)
            ax.imshow(background_view, cmap="gray", vmin=0.0, vmax=1.0, rasterized=True)
            ax.imshow(
                _label_overlay(pred_view),
                cmap=cmap,
                vmin=class_lo,
                vmax=class_hi,
                alpha=overlay_alpha,
                rasterized=True,
            )
            if show_gt_contour and np.any(gt_view > 0):
                # Contours stay VECTOR in the PDF (they are line artists, not
                # images), which is why the outline is still crisp at 400%.
                ax.contour(
                    (gt_view > 0).astype(float),
                    levels=[0.5],
                    colors="#FFFFFF",
                    linewidths=0.6,
                )
            if label in case.annotations:
                _cell_annotation(ax, case.annotations[label])

        # Uncertainty column.
        if show_uncertainty:
            ax = axes[row][-1]
            _blank_axes(ax)
            if case.uncertainty is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                unc_view = _apply_crop(take_slice(case.uncertainty, index, plane), crop)
                image = ax.imshow(unc_view, cmap=uncertainty_cmap, rasterized=True)
                # Normalization is PER CASE, so brightness is not comparable
                # between rows -- the colorbar makes that explicit. A shared
                # scale would be dominated by whichever case is worst, flattening
                # every other row to near-black and hiding its structure.
                bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
                bar.ax.tick_params(labelsize=5)
                bar.outline.set_linewidth(0.4)
            if column_titles[-1] in case.annotations:
                _cell_annotation(ax, case.annotations[column_titles[-1]])

    for col, title in enumerate(column_titles):
        axes[0][col].set_title(title, fontsize=8)

    handles = [Patch(facecolor=CLASS_COLORS[k], label=CLASS_NAMES[k]) for k in sorted(CLASS_NAMES)]
    # "outside lower center" makes constrained_layout RESERVE space for the
    # legend instead of drawing it on top of the bottom row. A plain
    # `loc="lower center"` with a negative bbox_to_anchor lands the class labels
    # across the last row's images, which is the kind of thing that survives all
    # the way into a submitted PDF because every test still passes.
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=7)
    return fig


# --------------------------------------------------------------------------- #
# Metric distributions
# --------------------------------------------------------------------------- #
def plot_metric_distributions(
    per_case: Mapping[str, pd.DataFrame],
    *,
    metric: str = "dice",
    regions: Sequence[str] = REGION_ORDER,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    show_points: bool = True,
    point_seed: int = 0,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Box-and-jitter of one metric, grouped by region, one box per model.

    The tail is the story, not the box. A Dice mean of 0.90 with a median of
    0.95 means the typical case is far better than the average and a minority
    fail badly enough to move the mean several points -- and for a claim about
    reliability those failures are the result. So outliers are drawn as
    individual jittered points rather than as boxplot fliers.

    Args:
        per_case: Model display label -> per-case metric table (the
            `per_case_metrics.csv` produced by `scripts/evaluate.py`). Insertion
            order is the plotting order and drives `model_style`.
        metric: Metric prefix; the column read is `f"{metric}_{region}"`.
        regions: Regions to draw, in reporting order.
        ylabel: Y-axis label. Defaults to `metric`.
        ylim: Explicit y limits; `None` autoscales.
        show_points: Overlay the per-case scatter.
        point_seed: Seed for the horizontal jitter, so the figure is
            reproducible.
        figsize: Explicit size; defaults to a width that scales with the number
            of boxes.

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: `per_case` is empty, `regions` is empty, or a required
            column is missing from one of the tables (named in the message).
    """
    if not per_case:
        raise ValueError("plot_metric_distributions: `per_case` must contain at least one model.")
    if not regions:
        raise ValueError("plot_metric_distributions: `regions` must contain at least one region.")

    labels = list(per_case)
    for label, table in per_case.items():
        missing = [f"{metric}_{r}" for r in regions if f"{metric}_{r}" not in table.columns]
        if missing:
            raise ValueError(
                f"plot_metric_distributions: model {label!r} is missing column(s) {missing}. "
                f"Available: {sorted(table.columns)}."
            )

    n_models = len(labels)
    if figsize is None:
        figsize = (max(4.0, 1.1 * n_models * len(regions) + 1.4), 3.2)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    rng = np.random.default_rng(point_seed)
    width = 0.8 / n_models
    positions: list[float] = []
    data: list[np.ndarray] = []
    colors: list[str] = []

    for region_index, region in enumerate(regions):
        for model_index, label in enumerate(labels):
            values = per_case[label][f"{metric}_{region}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            # Offset each model within its region group, centred on the group.
            offset = (model_index - (n_models - 1) / 2.0) * width
            positions.append(region_index + offset)
            data.append(values)
            colors.append(model_style(model_index)["color"])

    box = ax.boxplot(
        data,
        positions=positions,
        widths=width * 0.85,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        # `tick_labels`, not `labels`: matplotlib removed the `labels` kwarg.
        tick_labels=[""] * len(data),
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    if show_points:
        for position, values, color in zip(positions, data, colors, strict=True):
            if values.size == 0:
                continue
            jitter = rng.normal(position, width * 0.10, values.size)
            ax.scatter(jitter, values, s=5, alpha=0.35, color=color, edgecolors="none", zorder=3)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(list(regions))
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)

    handles = [
        Patch(facecolor=model_style(i)["color"], alpha=0.55, label=label)
        for i, label in enumerate(labels)
    ]
    ax.legend(handles=handles, loc="best", ncol=min(len(labels), 3))
    return fig


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def plot_reliability_diagram(
    curves: Mapping[str, pd.DataFrame],
    *,
    ece: Mapping[str, float] | None = None,
    title: str | None = None,
    show_counts: bool = True,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Reliability diagram: predicted probability against observed frequency.

    Consumes `uncertainty.calibration.reliability_curve` output directly. Note
    the axes are probability-vs-frequency, NOT the confidence-vs-accuracy form
    used for multi-class softmax models -- our three heads are independent
    sigmoids over overlapping regions, where "confidence" of the argmax class is
    undefined.

    A point BELOW the diagonal is overconfident (predicted more than observed),
    which is the failure mode this project's research claim is about.

    Args:
        curves: Model display label -> a `reliability_curve` DataFrame with
            columns `bin_lower, bin_upper, count, mean_prob, mean_label, gap`.
        ece: Optional model label -> ECE, appended to the legend entry. Reported
            from `expected_calibration_error`, never recomputed here from the
            binned means (that would be an approximation of an approximation).
        title: Axes title.
        show_counts: Draw a bin-population histogram underneath. Without it a
            reader cannot tell a real miscalibration from a bin holding four
            voxels.
        figsize: Explicit size.

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: `curves` is empty or a table is missing a required column.
    """
    if not curves:
        raise ValueError("plot_reliability_diagram: `curves` must contain at least one model.")
    required = {"bin_lower", "bin_upper", "count", "mean_prob", "mean_label"}
    for label, table in curves.items():
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"plot_reliability_diagram: curve {label!r} is missing column(s) {missing}; "
                "expected the output of uncertainty.calibration.reliability_curve."
            )

    if figsize is None:
        figsize = (3.6, 4.4 if show_counts else 3.4)

    if show_counts:
        fig, (ax, ax_hist) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        ax_hist = None

    ax.plot([0, 1], [0, 1], color="#666666", linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)

    for index, (label, table) in enumerate(curves.items()):
        style = model_style(index)
        legend_label = label
        if ece is not None and label in ece and np.isfinite(ece[label]):
            legend_label = f"{label} (ECE {ece[label]:.4f})"
        ax.plot(
            table["mean_prob"].to_numpy(dtype=float),
            table["mean_label"].to_numpy(dtype=float),
            marker="o",
            markersize=3,
            label=legend_label,
            zorder=2,
            **style,
        )
        if ax_hist is not None:
            centres = (
                table["bin_lower"].to_numpy(dtype=float) + table["bin_upper"].to_numpy(dtype=float)
            ) / 2.0
            counts = table["count"].to_numpy(dtype=float)
            total = counts.sum()
            # Fraction, not raw count: two models evaluated over different
            # voxel counts would otherwise produce incomparable bar heights.
            fraction = counts / total if total > 0 else counts
            ax_hist.step(centres, fraction, where="mid", color=style["color"], linewidth=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed frequency")
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left")

    if ax_hist is not None:
        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Bin share")
        # Log scale only when there is something positive to log. Bin
        # populations span several orders of magnitude in a real run (almost
        # every voxel lands in the first bin), but an all-empty curve -- what
        # `reliability_curve` returns for a region absent from a case -- has no
        # positive values and matplotlib warns rather than erroring.
        if any(np.nansum(t["count"].to_numpy(dtype=float)) > 0 for t in curves.values()):
            ax_hist.set_yscale("log")
    else:
        ax.set_xlabel("Predicted probability")
    return fig


# --------------------------------------------------------------------------- #
# Risk-coverage
# --------------------------------------------------------------------------- #
class CurveLike(Protocol):
    """Structural type of `uncertainty.risk_coverage.RiskCoverageCurve`.

    Typed as a Protocol rather than imported, because `risk_coverage` imports
    torch and this module must stay torch-free. Any object exposing these three
    attributes plots correctly, which also makes the tests trivial.
    """

    coverage: np.ndarray
    performance: np.ndarray
    aurc: float


def plot_risk_coverage(
    curves: Mapping[str, CurveLike],
    *,
    oracle: CurveLike | None = None,
    random: CurveLike | None = None,
    ylabel: str = "Mean Dice over retained cases",
    title: str | None = None,
    show_aurc: bool = True,
    figsize: tuple[float, float] = (3.8, 3.2),
) -> Figure:
    """Selective-prediction curve: performance against the fraction of cases kept.

    Reads left-to-right as "if we refer the least-confident cases to a human,
    how good is what remains?" The oracle is the ceiling no uncertainty estimate
    can beat; the random line is the null. A curve that hugs the random line
    means the uncertainty estimate carries no information about case difficulty,
    which is a real and reportable negative result.

    Args:
        curves: Model display label -> anything exposing `coverage`,
            `performance` and `aurc` (a `RiskCoverageCurve`).
        oracle: Optional ceiling curve from `risk_coverage.oracle_curve`.
        random: Optional null curve from `risk_coverage.random_curve`.
        ylabel: Y-axis label. Name the score AND its aggregation -- "Dice" alone
            is ambiguous about whether it is a mean over retained cases.
        title: Axes title.
        show_aurc: Append each curve's AURC to its legend entry. Only comparable
            between curves over the SAME case set.
        figsize: Figure size.

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: `curves` is empty.
    """
    if not curves:
        raise ValueError("plot_risk_coverage: `curves` must contain at least one model.")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if oracle is not None:
        ax.plot(
            oracle.coverage,
            oracle.performance,
            color="#000000",
            linewidth=0.9,
            linestyle=(0, (1, 1.5)),
            label="Oracle (ceiling)",
            zorder=1,
        )
    if random is not None:
        ax.plot(
            random.coverage,
            random.performance,
            color="#999999",
            linewidth=0.9,
            label="Random (null)",
            zorder=1,
        )

    for index, (label, curve) in enumerate(curves.items()):
        legend_label = f"{label} (AURC {curve.aurc:.4f})" if show_aurc else label
        ax.plot(
            curve.coverage, curve.performance, label=legend_label, zorder=2, **model_style(index)
        )

    ax.set_xlabel("Coverage (fraction of cases retained)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best")
    return fig


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
@dataclass
class TrainingPanel:
    """One axes of the training-curve figure.

    Attributes:
        title: Axes title.
        columns: History column names to draw in this panel.
        ylabel: Y-axis label.
        labels: Legend labels, one per column. Defaults to the column names.
        ylim: Explicit y limits.
        logy: Use a log y scale.
    """

    title: str
    columns: Sequence[str]
    ylabel: str
    labels: Sequence[str] | None = None
    ylim: tuple[float, float] | None = None
    logy: bool = False


def plot_training_curves(
    histories: Mapping[str, pd.DataFrame],
    panels: Sequence[TrainingPanel],
    *,
    x_column: str = "_step",
    xlabel: str = "optimizer step",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Training/validation curves, one axes per `TrainingPanel`.

    Style rule: **colour encodes the model, linestyle encodes the column.** With
    several models and several metrics per panel, letting both vary freely makes
    a legend nobody can read.

    Rows where a column is NaN are dropped per column, not per row. Train and
    validation metrics are logged at different cadences (validation every N
    epochs), so a W&B history is sparse; plotting it raw connects a line across
    the gaps and invents values that were never measured.

    Args:
        histories: Model display label -> a history DataFrame (e.g. the W&B
            API's `run.history(...)`).
        panels: The panels to draw, left to right.
        x_column: Column used as the x axis, present in every history.
        xlabel: X-axis label.
        figsize: Explicit size; defaults to scaling with the panel count.

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: `histories` or `panels` is empty, or `x_column` is missing
            from one of the histories.
    """
    if not histories:
        raise ValueError("plot_training_curves: `histories` must contain at least one run.")
    if not panels:
        raise ValueError("plot_training_curves: `panels` must contain at least one panel.")
    for label, history in histories.items():
        if x_column not in history.columns:
            raise ValueError(
                f"plot_training_curves: run {label!r} has no {x_column!r} column; available: "
                f"{sorted(history.columns)[:20]}."
            )

    if figsize is None:
        figsize = (3.9 * len(panels), 3.0)
    fig, axes = plt.subplots(
        1, len(panels), figsize=figsize, squeeze=False, constrained_layout=True
    )

    multi_model = len(histories) > 1
    for panel_index, panel in enumerate(panels):
        ax = axes[0][panel_index]
        legend_labels = list(panel.labels) if panel.labels is not None else list(panel.columns)
        if len(legend_labels) != len(panel.columns):
            raise ValueError(
                f"plot_training_curves: panel {panel.title!r} has {len(panel.columns)} columns "
                f"but {len(legend_labels)} labels."
            )

        for model_index, (model_label, history) in enumerate(histories.items()):
            color = model_style(model_index)["color"]
            for column_index, column in enumerate(panel.columns):
                if column not in history.columns:
                    logger.warning(
                        "plot_training_curves: run %r has no column %r; skipping it in panel %r.",
                        model_label,
                        column,
                        panel.title,
                    )
                    continue
                subset = history[[x_column, column]].dropna()
                if subset.empty:
                    continue
                name = legend_labels[column_index]
                entry = f"{model_label} — {name}" if multi_model else name
                ax.plot(
                    subset[x_column].to_numpy(),
                    subset[column].to_numpy(),
                    color=color,
                    linestyle=MODEL_LINESTYLE_CYCLE[column_index % len(MODEL_LINESTYLE_CYCLE)],
                    label=entry,
                )

        ax.set_title(panel.title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(panel.ylabel)
        if panel.ylim is not None:
            ax.set_ylim(*panel.ylim)
        if panel.logy:
            ax.set_yscale("log")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", ncol=1 if len(panel.columns) < 3 else 2)
    return fig


# --------------------------------------------------------------------------- #
# Statistical comparison
# --------------------------------------------------------------------------- #
def plot_comparison_forest(
    table: pd.DataFrame,
    *,
    name_a: str = "model",
    name_b: str = "baseline",
    metrics: Sequence[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Forest plot of a `analysis.statistics.compare_models` table.

    Plots the `improvement` column with its bootstrap CI, one row per metric.
    `improvement` is already re-oriented by `compare_models` so POSITIVE ALWAYS
    MEANS "A IS BETTER", regardless of whether the metric itself is higher- or
    lower-is-better -- so this figure needs no per-metric direction logic, and a
    reader never has to remember the convention per row.

    Rows whose `verdict` is `inconclusive` or `negligible` are drawn in grey.
    That is deliberate: those differences must not be claimed in the paper, and
    a figure that colours them like results invites exactly that claim.

    Args:
        table: The output of `compare_models`, indexed by metric.
        name_a: Display name of model A (the one positive values favour).
        name_b: Display name of model B.
        metrics: Subset of rows to draw, in order. `None` draws all of them in
            table order. NOTE: the Holm correction in `compare_models` was
            computed over the WHOLE table -- subsetting here changes what is
            drawn, never what was tested, which is the correct way round.
        title: Axes title.
        figsize: Explicit size; defaults to scaling with the row count.

    Returns:
        The assembled `Figure`.

    Raises:
        ValueError: A required column is missing, the table is empty, or a
            requested metric is not in the index.
    """
    required = {"improvement", "improvement_lo", "improvement_hi", "verdict"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            f"plot_comparison_forest: table is missing column(s) {missing}; expected the output "
            "of analysis.statistics.compare_models."
        )
    if table.empty:
        raise ValueError("plot_comparison_forest: table is empty.")

    if metrics is None:
        rows = table
    else:
        unknown = [m for m in metrics if m not in table.index]
        if unknown:
            raise ValueError(
                f"plot_comparison_forest: metric(s) {unknown} are not in the table index "
                f"{list(table.index)}."
            )
        rows = table.loc[list(metrics)]

    n_rows = len(rows)
    if figsize is None:
        figsize = (4.6, max(2.0, 0.32 * n_rows + 1.1))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Drawn top-to-bottom in table order, so the figure reads in the same order
    # as the table it accompanies.
    y_positions = np.arange(n_rows)[::-1]
    for y, (_metric, row) in zip(y_positions, rows.iterrows(), strict=True):
        verdict = str(row["verdict"])
        color = _VERDICT_COLORS.get(verdict, "#999999")
        centre = float(row["improvement"])
        lo = float(row["improvement_lo"])
        hi = float(row["improvement_hi"])
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.4, solid_capstyle="butt")
        ax.plot([centre], [y], marker="o", markersize=4.5, color=color, linestyle="none")

    ax.axvline(0.0, color="#333333", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(rows.index))
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel(f"improvement of {name_a} over {name_b}  (positive = {name_a} better)")
    ax.grid(axis="y", visible=False)
    if title is not None:
        ax.set_title(title)

    seen = list(dict.fromkeys(str(v) for v in rows["verdict"]))
    handles = [Patch(facecolor=_VERDICT_COLORS.get(v, "#999999"), label=v) for v in seen]
    ax.legend(handles=handles, loc="best", ncol=min(len(handles), 3))
    return fig
