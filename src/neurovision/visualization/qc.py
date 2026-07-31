"""Quality-control figures for the offline preprocessing pipeline.

These figures exist so a human can catch a broken pipeline by eye: wrong
channel order, a label that does not line up with the anatomy, normalization
that flattened everything. Legibility is the point, so figures are large and
titles/labels are explicit rather than terse.

Dependency-light on purpose: only numpy and matplotlib. No torch, no monai --
this module must be importable on a machine that has neither installed, and
it must run on CPU. Nothing here reads or writes files; every function takes
arrays and returns a `Figure` that the caller saves or displays.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# Fixed channel order used throughout the project's preprocessing pipeline.
MODALITY_NAMES: tuple[str, ...] = ("T1", "T1CE", "T2", "FLAIR")

# Remapped (contiguous) BraTS label meanings -- see
# `neurovision.data.preprocessing.BRATS_LABEL_MAP`. Class 0 (background) is
# intentionally absent here: it is never drawn (see plot_case_slices).
CLASS_NAMES: dict[int, str] = {1: "NCR/NET", 2: "ED", 3: "ET"}

# Okabe-Ito palette (blue, green, vermillion). Chosen because it stays
# distinguishable under the most common forms of colour-vision deficiency --
# the whole point of this overlay is that a human can tell the three tumor
# sub-regions apart, so a palette that degrades under CVD would defeat the
# purpose.
CLASS_COLORS: dict[int, str] = {1: "#56B4E9", 2: "#009E73", 3: "#D55E00"}


def mid_slice_indices(shape: tuple[int, int, int]) -> dict[str, int]:
    """Compute the middle index along each axis of a 3D shape.

    Args:
        shape: Volume shape `(D, H, W)`.

    Returns:
        Dict with keys `"d"`, `"h"`, `"w"`, each the floor-division midpoint
        of the corresponding axis.
    """
    d, h, w = shape
    return {"d": d // 2, "h": h // 2, "w": w // 2}


def extract_mid_slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the three orthogonal mid-slices of a single 3D volume.

    Args:
        volume: Array of shape `(D, H, W)`.

    Returns:
        Dict with keys `"sagittal"`, `"coronal"`, `"axial"`, each a rotated
        2D slice ready for display.

    Note:
        This mapping (sagittal = mid-D plane, coronal = mid-H plane, axial =
        mid-W plane) assumes the standard BraTS array layout, where the last
        axis indexes axial slices. The notebook prints each case's affine so
        the orientation can be sanity-checked against real data; if a
        dataset turns out to be stored differently, this is the single place
        to fix.
    """
    idx = mid_slice_indices(volume.shape)
    sagittal = volume[idx["d"], :, :]
    coronal = volume[:, idx["h"], :]
    axial = volume[:, :, idx["w"]]
    # Rotate 90 degrees so brains appear upright in the figure rather than on
    # their side. Display-only -- this does not touch the underlying data,
    # only how these particular 2D arrays are drawn.
    return {
        "sagittal": np.rot90(sagittal),
        "coronal": np.rot90(coronal),
        "axial": np.rot90(axial),
    }


def _validate_image(image: np.ndarray, func_name: str) -> None:
    """Raise ValueError unless `image` is 4D with 4 channels."""
    if image.ndim != 4 or image.shape[0] != len(MODALITY_NAMES):
        raise ValueError(
            f"{func_name}: expected `image` with shape (4, D, H, W), got shape {image.shape}."
        )


def _label_overlay(label_slice: np.ndarray, overlay_alpha: float) -> np.ma.MaskedArray:
    """Build a masked array for use with a class-color ListedColormap.

    Background (class 0) is masked out so it renders fully transparent
    instead of drawing over the grayscale MRI panel underneath.
    """
    return np.ma.masked_where(label_slice == 0, label_slice)


def plot_case_slices(
    image: np.ndarray,
    label: np.ndarray | None,
    case_id: str,
    overlay_alpha: float = 0.45,
) -> Figure:
    """Plot mid-slices of all 4 modalities, in 3 anatomical planes, with an optional label overlay.

    Args:
        image: Stacked modalities, shape `(4, D, H, W)`, channel order
            matching `MODALITY_NAMES`.
        label: Optional remapped label volume, shape `(D, H, W)`, values in
            `{0, 1, 2, 3}`. If None, no overlay is drawn.
        case_id: Identifier shown in the figure title.
        overlay_alpha: Opacity of the label overlay (0 = invisible, 1 =
            opaque). Background (class 0) is always fully transparent
            regardless of this value.

    Returns:
        A 4x3 grid `Figure`: rows are modalities, columns are
        sagittal/coronal/axial mid-slices.

    Raises:
        ValueError: If `image` is not 4D with 4 channels, or `label` is not
            None and its spatial shape does not match `image`'s spatial
            shape. A silent mismatch here would produce a plausible-looking
            but wrong figure, which is worse than no figure.
    """
    _validate_image(image, "plot_case_slices")
    if label is not None and label.shape != image.shape[1:]:
        raise ValueError(
            f"plot_case_slices: label shape {label.shape} does not match "
            f"image spatial shape {image.shape[1:]}."
        )

    plane_names = ("sagittal", "coronal", "axial")
    plane_titles = ("Sagittal", "Coronal", "Axial")

    # A ListedColormap indexed 1..3 (class 0 is masked, so its color never
    # actually gets used) -- vmin/vmax pin the colormap to exactly the 3
    # known classes so colors do not shift if a panel happens to be missing
    # one of them.
    overlay_cmap = ListedColormap([CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]])

    # constrained_layout keeps the 4x3 grid aligned. The three planes have
    # different aspect ratios, and because each panel preserves its true voxel
    # aspect (anchored top-centre below, never stretched -- distorting anatomy
    # to fill a box would defeat the purpose of a QC figure), tight_layout
    # leaves rows ragged and column titles at different heights.
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)

    label_slices = extract_mid_slices(label) if label is not None else None

    for row, modality in enumerate(MODALITY_NAMES):
        image_slices = extract_mid_slices(image[row])
        for col, plane in enumerate(plane_names):
            ax = axes[row, col]
            ax.imshow(image_slices[plane], cmap="gray", interpolation="nearest")
            if label_slices is not None:
                masked = _label_overlay(label_slices[plane], overlay_alpha)
                ax.imshow(
                    masked,
                    cmap=overlay_cmap,
                    vmin=1,
                    vmax=3,
                    alpha=overlay_alpha,
                    interpolation="nearest",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            # Anchor panels to the top of their cell so rows read as rows even
            # when the three planes have different heights.
            ax.set_anchor("N")
            if row == 0:
                ax.set_title(plane_titles[col], fontsize=14)
            if col == 0:
                ax.set_ylabel(modality, fontsize=14)

    fig.suptitle(f"Case {case_id} -- volume shape {image.shape}", fontsize=16)

    legend_handles = [
        Patch(facecolor=CLASS_COLORS[cls], label=name) for cls, name in CLASS_NAMES.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=12,
    )
    return fig


def plot_intensity_histograms(
    raw_image: np.ndarray,
    normalized_image: np.ndarray,
    case_id: str,
    bins: int = 100,
) -> Figure:
    """Plot before/after intensity histograms for each modality.

    Args:
        raw_image: Un-normalized modalities, shape `(4, D, H, W)`.
        normalized_image: Z-scored modalities, shape `(4, D', H', W')`.
            May differ in spatial shape from `raw_image` (raw is typically
            uncropped, normalized is typically cropped) -- that mismatch is
            expected and is not validated.
        case_id: Identifier shown in the figure title.
        bins: Number of histogram bins.

    Returns:
        A 4x2 grid `Figure`: rows are modalities, columns are
        before (raw) / after (normalized).

    Raises:
        ValueError: If either input is not 4D with 4 channels.
    """
    _validate_image(raw_image, "plot_intensity_histograms")
    _validate_image(normalized_image, "plot_intensity_histograms")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

    for row, modality in enumerate(MODALITY_NAMES):
        # Background is ~60-70% of an MRI volume and sits at exactly 0.
        # Including it would produce one enormous spike at zero that
        # flattens everything else into invisibility, so only nonzero
        # voxels are histogrammed -- both here and on the "after" panel.
        raw_nonzero = raw_image[row][raw_image[row] != 0]
        norm_nonzero = normalized_image[row][normalized_image[row] != 0]

        ax_before = axes[row, 0]
        if raw_nonzero.size > 0:
            ax_before.hist(raw_nonzero.ravel(), bins=bins, color="gray")
        else:
            # An empty nonzero selection (e.g. an all-zero channel) must not
            # crash the figure -- just leave the panel blank and say why.
            ax_before.text(0.5, 0.5, "no nonzero voxels", ha="center", va="center")
        ax_before.set_title(f"{modality} -- before (raw)")
        ax_before.set_xlabel("Intensity (nonzero voxels only)")
        ax_before.set_ylabel("Voxel count")

        ax_after = axes[row, 1]
        if norm_nonzero.size > 0:
            ax_after.hist(norm_nonzero.ravel(), bins=bins, color="steelblue")
            mean = float(norm_nonzero.mean())
            std = float(norm_nonzero.std())
            ax_after.axvline(0.0, color="black", linestyle="--", linewidth=1)
            ax_after.set_title(f"{modality} -- after (z-scored)  mean={mean:.2f}  std={std:.2f}")
        else:
            ax_after.text(0.5, 0.5, "no nonzero voxels", ha="center", va="center")
            ax_after.set_title(f"{modality} -- after (z-scored)")
        ax_after.set_xlabel("Intensity (nonzero voxels only)")
        ax_after.set_ylabel("Voxel count")

    fig.suptitle(f"Case {case_id} -- intensity distributions", fontsize=16)
    return fig
