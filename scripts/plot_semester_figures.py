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

3. `fig_examples_deployed.png` -- example segmentations from the DEPLOYED model
   (`neurovision`, seed 42): per cohort one typical and one failure case, drawn
   from the saved fp16 logits with the deployed post-processing (review row R13).
   This is the only figure that reads volumes; it needs no model and no GPU.
   Run it alone with `analysis.semester_figures.only=examples`.

Style and the `save_figure` idiom (PNG, `out_dir` created on demand) are
reused from `neurovision.visualization.figures` rather than reinvented here.

Example usage:

    python scripts/plot_semester_figures.py
    python scripts/plot_semester_figures.py analysis.semester_figures.only=examples
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import hydra
import matplotlib
import matplotlib.colors
import matplotlib.patches
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig

# Must happen before pyplot is imported anywhere in this process -- the
# default backend tries to open a GUI window, which has no display in a
# headless test run or a batch script and either hangs or raises.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from neurovision.analysis.replay import load_case_logits  # noqa: E402
from neurovision.inference.postprocess import postprocess_logits  # noqa: E402
from neurovision.metrics.segmentation import classes_to_regions, dice_score  # noqa: E402
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
# Figure 3 -- example segmentations from the deployed model (review row R13)
# --------------------------------------------------------------------------- #
_REGION_NAMES: tuple[str, str, str] = ("ET", "TC", "WT")

# Class index -> (legend name, colour). Okabe-Ito colour-blind-safe palette.
# Class 0 is background and is never drawn.
_EXAMPLE_CLASSES: dict[int, tuple[str, str]] = {
    1: ("NCR/NET", "#E69F00"),
    2: ("ED", "#56B4E9"),
    3: ("ET", "#D55E00"),
}
_OVERLAY_ALPHA = 0.45

# Channel order of image.npy (z-scored): t1, t1ce, t2, flair.
_FLAIR_CHANNEL = 3
_T1CE_CHANNEL = 1

_EXAMPLE_COLUMNS: tuple[str, ...] = ("FLAIR", "T1CE", "Reference", "Prediction")
_SERIF_FALLBACKS = ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"]
_DICE_TOLERANCE = 1e-3
_PED_NOTE = "PED reference labels count most non-enhancing tumour as core (note 58)"


def select_example_cases(per_case: pd.DataFrame) -> dict[str, str]:
    """Picks the typical and the failure case of one cohort from its per-case table.

    TYPICAL: the case whose mean of dice_ET, dice_TC, dice_WT is closest to the
    cohort median of that mean (ties broken by case_id). FAILURE: the lowest
    dice_TC (ties broken by case_id). Cases with a NaN Dice are ignored.

    Args:
        per_case: Table with columns `case_id, dice_ET, dice_TC, dice_WT`.

    Returns:
        `{"typical": case_id, "failure": case_id}`.

    Raises:
        ValueError: No case has all three Dice values.
    """
    cols = ["dice_ET", "dice_TC", "dice_WT"]
    table = per_case.dropna(subset=cols).copy()
    if table.empty:
        raise ValueError("select_example_cases: no case has finite dice_ET, dice_TC and dice_WT.")
    table["case_id"] = table["case_id"].astype(str)
    mean_dice = table[cols].mean(axis=1)
    distance = (mean_dice - mean_dice.median()).abs()
    # Sort by (distance, case_id): the id is only a tie-breaker.
    typical = table.assign(_d=distance).sort_values(["_d", "case_id"]).iloc[0]["case_id"]
    failure = table.sort_values(["dice_TC", "case_id"]).iloc[0]["case_id"]
    return {"typical": str(typical), "failure": str(failure)}


def regions_to_classes(regions: np.ndarray) -> np.ndarray:
    """Converts nested binary regions into the three BraTS display classes.

    ET = ET; NCR/NET = TC minus ET; ED = WT minus TC.

    Args:
        regions: Binary array `(3, D, H, W)`, channel order (ET, TC, WT).

    Returns:
        Integer array `(D, H, W)`: 0 background, 1 NCR/NET, 2 ED, 3 ET.
    """
    et, tc, wt = (regions[i] > 0.5 for i in range(3))
    classes = np.zeros(et.shape, dtype=np.uint8)
    # Written outermost first so inner regions overwrite the outer ones.
    classes[wt & ~tc & ~et] = 2
    classes[tc & ~et] = 1
    classes[et] = 3
    return classes


def choose_axial_slice(reference: np.ndarray, prediction: np.ndarray) -> int:
    """Index of the axial slice with the largest whole-tumour area.

    Uses the reference WT channel; if the reference WT is empty, the predicted WT.
    The axial axis is the LAST array axis (arrays are LPS, so axis 2 is S).

    Args:
        reference: Binary reference regions `(3, D, H, W)`.
        prediction: Binary predicted regions `(3, D, H, W)`.

    Returns:
        Slice index along the last axis (0 if both are empty).
    """
    for regions in (reference, prediction):
        area = (regions[2] > 0.5).sum(axis=(0, 1))
        if area.max() > 0:
            return int(np.argmax(area))
    return 0


def check_dice_matches_table(
    prediction: np.ndarray, reference: np.ndarray, row: pd.Series, case_id: str
) -> dict[str, float]:
    """Asserts our recomputed Dice equals the per-case table (guards wrong case/logits).

    Args:
        prediction: Post-processed binary regions `(3, D, H, W)`.
        reference: Reference binary regions `(3, D, H, W)`.
        row: The case's row of the per-case table (`dice_ET`, `dice_TC`, `dice_WT`).
        case_id: Used in the error message.

    Returns:
        Region name -> recomputed Dice.

    Raises:
        AssertionError: Any region differs from the table by more than 1e-3.
    """
    scores = dice_score(torch.from_numpy(prediction)[None], torch.from_numpy(reference)[None])[0]
    recomputed = {name: float(scores[i]) for i, name in enumerate(_REGION_NAMES)}
    for name, value in recomputed.items():
        expected = float(row[f"dice_{name}"])
        assert abs(value - expected) <= _DICE_TOLERANCE, (
            f"Dice sanity check failed for {case_id} {name}: recomputed {value:.4f} from the "
            f"logits and label, but the per-case table says {expected:.4f}. Wrong case, wrong "
            "logits directory, or different post-processing."
        )
    return recomputed


def _window(slice_2d: np.ndarray) -> tuple[float, float]:
    """Per-slice 1st to 99th percentile display window."""
    lo, hi = np.percentile(slice_2d, [1, 99])
    return float(lo), float(hi) if hi > lo else float(lo) + 1.0


def _show(ax: plt.Axes, slice_2d: np.ndarray, classes_2d: np.ndarray | None = None) -> None:
    """Draws one panel. `.T` puts anterior at the top and patient-left on the image right.

    Arrays are LPS (axis 0 grows toward patient left, axis 1 toward posterior), so after
    the transpose rows run anterior->posterior and columns run right->left-of-patient,
    which is the radiological convention.
    """
    lo, hi = _window(slice_2d)
    ax.imshow(slice_2d.T, cmap="gray", vmin=lo, vmax=hi, origin="upper", interpolation="nearest")
    if classes_2d is not None:
        overlay = np.zeros((*classes_2d.T.shape, 4), dtype=float)
        for index, (_, colour) in _EXAMPLE_CLASSES.items():
            overlay[classes_2d.T == index] = (*matplotlib.colors.to_rgb(colour), _OVERLAY_ALPHA)
        ax.imshow(overlay, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_example_row(
    axes_row: Sequence[plt.Axes],
    cohort: Mapping[str, str],
    kind: str,
    case_id: str,
    row: pd.Series,
    cfg: DictConfig,
) -> dict[str, float]:
    """Loads ONE case, runs the sanity check, draws four panels, returns its Dice.

    Only this case's arrays are alive; they are released when the call returns.
    """
    prep_case = Path(str(cohort["prep_dir"])) / case_id
    logits = load_case_logits(str(cohort["eval_dir"]), case_id)  # (3, D, H, W)
    prediction = postprocess_logits(torch.from_numpy(logits)[None], cfg)[0].numpy()
    label = np.load(prep_case / "label.npy")  # (D, H, W), values 0..3
    reference = classes_to_regions(torch.from_numpy(label.astype(np.int64)))[0].numpy()
    dice = check_dice_matches_table(prediction, reference, row, case_id)

    k = choose_axial_slice(reference, prediction)
    image = np.load(prep_case / "image.npy", mmap_mode="r")  # (4, D, H, W)
    flair = np.asarray(image[_FLAIR_CHANNEL][:, :, k], dtype=np.float32)
    t1ce = np.asarray(image[_T1CE_CHANNEL][:, :, k], dtype=np.float32)
    ref_classes = regions_to_classes(reference[..., k : k + 1])[:, :, 0]
    pred_classes = regions_to_classes(prediction[..., k : k + 1])[:, :, 0]

    _show(axes_row[0], flair)
    _show(axes_row[1], t1ce)
    _show(axes_row[2], flair, ref_classes)
    _show(axes_row[3], flair, pred_classes)
    axes_row[0].text(
        -0.04,
        0.5,
        f"{cohort['name']} \u00b7 {kind} \u00b7\n{case_id}\n"
        f"Dice ET {dice['ET']:.2f} \u00b7 TC {dice['TC']:.2f} \u00b7 WT {dice['WT']:.2f}",
        transform=axes_row[0].transAxes,
        ha="right",
        va="center",
    )
    return dice


def plot_example_segmentations(
    cohorts: Sequence[Mapping[str, str]],
    selections: Sequence[dict[str, str]],
    per_case_tables: Sequence[pd.DataFrame],
    cfg: DictConfig,
) -> tuple[Figure, list[dict[str, object]]]:
    """Draws the (2 x n_cohorts) x 4 grid of example segmentations.

    Args:
        cohorts: Per cohort `name, eval_dir, prep_dir` (and `per_case`).
        selections: Per cohort `{"typical": id, "failure": id}`.
        per_case_tables: Per cohort per-case table (for the Dice sanity check).
        cfg: Composed config; `inference.postprocess` drives the post-processing.

    Returns:
        The figure and one record per row (cohort, kind, case_id, dice).
    """
    n_rows = 2 * len(cohorts)
    records: list[dict[str, object]] = []
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": _SERIF_FALLBACKS,
            "font.size": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(
            n_rows, 4, figsize=(7.6, 1.85 * n_rows + 0.5), squeeze=False, facecolor="white"
        )
        fig.subplots_adjust(left=0.30, right=0.99, top=0.95, bottom=0.07, wspace=0.03, hspace=0.04)
        r = 0
        for cohort, selection, table in zip(cohorts, selections, per_case_tables, strict=True):
            lookup = table.astype({"case_id": str}).set_index("case_id", drop=False)
            for kind in ("typical", "failure"):
                case_id = selection[kind]
                dice = _draw_example_row(axes[r], cohort, kind, case_id, lookup.loc[case_id], cfg)
                records.append(
                    {"cohort": cohort["name"], "kind": kind, "case_id": case_id, "dice": dice}
                )
                r += 1
        for ax, title in zip(axes[0], _EXAMPLE_COLUMNS, strict=True):
            ax.set_title(title)
        handles = [
            matplotlib.patches.Patch(facecolor=c, alpha=_OVERLAY_ALPHA + 0.2, label=n)
            for n, c in _EXAMPLE_CLASSES.values()
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    return fig, records


def save_examples_sidecar(
    path: Path,
    records: Sequence[Mapping[str, object]],
    sources: Sequence[str],
) -> None:
    """Writes `fig_examples_deployed.txt`: command, sources, selections, caveats.

    Args:
        path: Destination `.txt`.
        records: Rows returned by `plot_example_segmentations`.
        sources: Every file path read.
    """
    rules = {
        "typical": "per-case mean of dice_ET/TC/WT closest to the cohort median, ties by case_id",
        "failure": "lowest dice_TC in the cohort, ties by case_id",
    }
    lines = [
        "Figure 15 (R13): example segmentations, deployed model (neurovision, seed 42).",
        "",
        "Command: " + " ".join([sys.executable, *sys.argv]),
        "",
        "Orientation: arrays are LPS; panels are radiological, anterior at top, "
        "patient LEFT is on the image RIGHT.",
        "Slice: axial slice with the largest reference WT area (predicted WT if reference empty).",
        "Overlay: NCR/NET = TC minus ET (orange), ED = WT minus TC (blue), ET (vermilion); "
        "alpha 0.45. Prediction = saved logits + deployed inference.postprocess.",
        "",
        "Selected cases:",
    ]
    for rec in records:
        d = rec["dice"]
        lines.append(
            f"  {rec['cohort']} | {rec['kind']} | {rec['case_id']} | "
            f"Dice ET {d['ET']:.4f} TC {d['TC']:.4f} WT {d['WT']:.4f} | "
            f"rule: {rules[str(rec['kind'])]}"
        )
    lines += ["", "Sources read:", *[f"  {s}" for s in sources], "", _PED_NOTE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("plot_semester_figures: wrote %s", path)


def run_examples_figure(cfg: DictConfig) -> list[Path]:
    """Builds and saves `fig_examples_deployed.png` and its sidecar.

    Args:
        cfg: Composed config; reads `analysis.semester_figures.examples` and
            `inference.postprocess`.

    Returns:
        The written paths: PNG, then sidecar.
    """
    figures_cfg = cfg.analysis.semester_figures
    ex_cfg = figures_cfg.examples
    out_dir = Path(str(figures_cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = [{k: str(v) for k, v in c.items()} for c in ex_cfg.cohorts]
    tables, selections, sources = [], [], []
    for cohort in cohorts:
        table = pd.read_csv(cohort["per_case"])
        tables.append(table)
        selections.append(select_example_cases(table))
        sources += [
            cohort["per_case"],
            str(Path(cohort["eval_dir"]) / "logits"),
            cohort["prep_dir"],
        ]
    fig, records = plot_example_segmentations(cohorts, selections, tables, cfg)
    png_path = out_dir / str(ex_cfg.out_name)
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    sidecar = png_path.with_suffix(".txt")
    save_examples_sidecar(sidecar, records, sources)
    logger.info("plot_semester_figures: wrote %s", png_path)
    return [png_path, sidecar]


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

    # `only=examples` runs just the example figure; the default ("all") keeps the
    # original two figures and adds the example figure when examples.enabled.
    only = str(figures_cfg.get("only", "all"))
    if only == "examples":
        written["fig_examples_deployed"] = run_examples_figure(cfg)
        _log_summary(written)
        return written

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

    if figures_cfg.get("examples") is not None and bool(figures_cfg.examples.enabled):
        written["fig_examples_deployed"] = run_examples_figure(cfg)

    _log_summary(written)
    return written


def _log_summary(written: Mapping[str, list[Path]]) -> None:
    """Logs and prints a compact end-of-run summary, matching the other analysis drivers."""
    lines = ["=" * 70, "Semester figures summary", "=" * 70]
    for stem in ("fig_local_recalibration", "fig_realised_vs_nominal", "fig_examples_deployed"):
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
