"""Deterministic, CPU-only tumour "burden profile" from a segmentation class map.

This module is pure arithmetic on numpy arrays: no file IO, no model, no
checkpoint. It is meant to run on artifacts that `scripts/evaluate.py` and
`neurovision.data.preprocessing` already write to disk (`predictions/<case>.npy`,
`label.npy`, `meta.json`) -- reading those files is the caller's job, not
this module's.

It deliberately has no dependency on torch, so it (and anything that
imports it) stays importable in an environment with no deep-learning stack
at all. The same reasoning as `neurovision.visualization.figures`; see
`tests/test_burden.py::test_burden_module_does_not_import_torch`.

Label convention (after preprocessing -- see `neurovision.data.preprocessing`):
the class map is an integer array with values in `{0, 1, 2, 3}` only.

    1 = NCR  (necrotic / non-enhancing core)
    2 = ED   (peritumoural oedema)
    3 = ET   (enhancing tumour)

Raw BraTS uses label 4 for enhancing tumour; preprocessing remaps it to 3.
A value of 4 reaching this module means unremapped raw data and every entry
point that accepts a class map raises on it.

Regions are UNIONS of classes, not three independently-predicted masks:

    ET = {3}
    TC = {1, 3}
    WT = {1, 2, 3}

Because the classes are disjoint by construction, building regions this way
guarantees `ET subset-of TC subset-of WT` -- the same nesting invariant
`neurovision.inference.postprocess` restores by hand for logit-derived
region masks. This module never accepts three independent binary masks as
input for exactly that reason: nothing here could re-verify the nesting.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import measure

__all__ = [
    "CLASS_IDS",
    "REGION_CLASSES",
    "REGION_ORDER",
    "CaseGeometry",
    "ComponentStats",
    "region_mask",
    "volume_mm3",
    "compute_volumes",
    "compute_fractions",
    "connected_components",
    "surface_area_mm2",
    "sphericity",
    "laterality",
    "centroid",
    "estimate_midline_index",
    "burden_profile",
]

logger = logging.getLogger(__name__)

CLASS_IDS: dict[str, int] = {"NCR": 1, "ED": 2, "ET": 3}
REGION_CLASSES: dict[str, tuple[int, ...]] = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")

_VALID_CLASS_VALUES = frozenset({0, 1, 2, 3})


def _validate_classes(classes: np.ndarray) -> None:
    """Raises if `classes` is not a valid `(D, H, W)` integer class map.

    Called by every public function in this module that accepts a raw class
    map (as opposed to an already-boolean region mask).

    Raises:
        ValueError: If `classes` is not 3-D, if its dtype is not an integer
            dtype (a float array usually means probabilities or logits were
            passed instead of a discretized class map), or if it contains
            any value outside `{0, 1, 2, 3}` -- in particular, value 4 means
            raw, unremapped BraTS labels reached this module (BraTS uses 4
            for enhancing tumour; preprocessing remaps it to 3).
    """
    if classes.ndim != 3:
        raise ValueError(
            f"burden: expected a (D, H, W) class map, got shape {classes.shape} "
            f"(ndim={classes.ndim})."
        )
    if not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(
            f"burden: expected an integer class map, got dtype {classes.dtype}. A float "
            "array usually means probabilities or logits were passed instead of a "
            "discretized class map."
        )
    values = set(int(v) for v in np.unique(classes))
    bad = sorted(values - _VALID_CLASS_VALUES)
    if bad:
        raise ValueError(
            f"burden: class map contains values outside {{0, 1, 2, 3}}: {bad}. Value 4 in "
            "particular means raw, unremapped BraTS labels reached this module -- BraTS uses "
            "4 for enhancing tumour, and preprocessing remaps it to 3."
        )


def _validate_mask(mask: np.ndarray, fn_name: str) -> None:
    """Raises if `mask` is not a `(D, H, W)` array. Shared by the mask-taking functions."""
    if mask.ndim != 3:
        raise ValueError(
            f"{fn_name}: expected a (D, H, W) mask, got shape {mask.shape} " f"(ndim={mask.ndim})."
        )


def _safe_div(numerator: float, denominator: float) -> float:
    """`numerator / denominator`, or NaN when the denominator is exactly 0.0.

    0/0 is undefined, not "no burden" -- this must never return 0.0 for a
    zero denominator, and must never raise.
    """
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


@dataclass(frozen=True)
class CaseGeometry:
    """Voxel geometry and the left/right convention for one case.

    Attributes:
        spacing: Voxel spacing in mm, `(D, H, W)` axis order.
        midline_index: The axis-0 index of the mid-sagittal plane, in voxel
            coordinates of whatever array this geometry describes (cropped
            or uncropped -- see `from_meta`). Defaults to `119.5`, the grid
            centre of the uncropped 240-voxel BraTS/SRI24 axis-0 extent.
            The affine's world origin sits at the grid EDGE (voxel index 0),
            not at the midline, so the midline cannot be read off the
            affine directly -- `119.5` is an assumption that the SRI24
            template is centred in its own grid. `estimate_midline_index`
            exists to check that assumption empirically.
        left_is_high_index: Whether patient LEFT corresponds to the HIGH end
            of axis 0. Under the BraTS affine `diag(-1, -1, 1)`, axis 0 runs
            right to left in world space, so low index = patient right and
            `left_is_high_index` is `True`.
    """

    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    midline_index: float = 119.5
    left_is_high_index: bool = True

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of one voxel in mm^3, the product of the three spacings."""
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2])

    @classmethod
    def from_meta(
        cls,
        meta: Mapping[str, Any],
        *,
        cropped: bool = False,
        midline_index: float | None = None,
    ) -> CaseGeometry:
        """Builds a `CaseGeometry` from a `meta.json` mapping.

        Args:
            meta: The mapping loaded from a case's `meta.json`, holding at
                least `spacing`, `affine`, and `original_shape` (and `bbox`
                when `cropped=True`).
            cropped: Whether the array this geometry will describe is in the
                CROPPED frame (`label.npy`, the preprocessed training image)
                rather than the original, uncropped geometry (saved
                predictions from `scripts/evaluate.py`). The midline index
                differs between the two by the crop offset.
            midline_index: Overrides the computed midline index entirely,
                e.g. from `estimate_midline_index`. When given, `cropped` is
                ignored for the midline computation (but still read for
                nothing else, since this is the only geometry field it
                affects).

        Returns:
            A new `CaseGeometry`.

        Raises:
            ValueError: If `meta["affine"][0][0]` is exactly 0 -- the
                left/right orientation cannot be inferred and this function
                refuses to guess, because a left/right swap in a burden
                report is the most damaging error this layer can make. Also
                raised if `cropped=True` and `meta` has no `bbox` key.
        """
        spacing = tuple(float(s) for s in meta["spacing"])

        a = meta["affine"][0][0]
        if a < 0:
            left_is_high_index = True
        elif a > 0:
            left_is_high_index = False
        else:
            raise ValueError(
                "CaseGeometry.from_meta: affine[0][0] == 0, so the left/right orientation "
                "cannot be inferred. Refusing to guess -- a left/right swap in a burden "
                "report is the most damaging error this layer can make."
            )

        if midline_index is not None:
            resolved_midline = float(midline_index)
        else:
            resolved_midline = (float(meta["original_shape"][0]) - 1.0) / 2.0
            if cropped:
                if "bbox" not in meta:
                    raise ValueError(
                        "CaseGeometry.from_meta: cropped=True requires meta['bbox'], which "
                        "is absent. The preprocessed label.npy is cropped to the nonzero "
                        "bbox while saved predictions are in original geometry -- the two "
                        "need different midline indices, and there is no safe default for "
                        "the cropped one without the bbox."
                    )
                resolved_midline -= float(meta["bbox"][0][0])

        return cls(
            spacing=spacing,
            midline_index=resolved_midline,
            left_is_high_index=left_is_high_index,
        )


@dataclass(frozen=True)
class ComponentStats:
    """Connected-component summary of one binary mask.

    Attributes:
        n: Number of components surviving the min-volume floor.
        volumes_mm3: Per-component volumes in mm^3, descending, AFTER the
            min-volume filter.
        total_volume_mm3: Volume in mm^3 of the UNFILTERED mask (i.e. all
            foreground voxels, including ones in components too small to
            survive the floor).
        largest_frac: `volumes_mm3[0] / total_volume_mm3` -- "how much of
            the tumour sits in its single biggest piece". NaN when the mask
            is empty, or when every component fell below the floor.
    """

    n: int
    volumes_mm3: tuple[float, ...]
    total_volume_mm3: float
    largest_frac: float


def region_mask(classes: np.ndarray, region: str) -> np.ndarray:
    """Builds the binary mask for one nested region from a class map.

    Args:
        classes: Integer class map, `(D, H, W)`, values in `{0, 1, 2, 3}`.
        region: One of `REGION_ORDER` (`"ET"`, `"TC"`, `"WT"`).

    Returns:
        Boolean array, `(D, H, W)`.

    Raises:
        ValueError: If `classes` fails validation, or `region` is not a
            known region name.
    """
    _validate_classes(classes)
    if region not in REGION_CLASSES:
        raise ValueError(
            f"region_mask: unknown region '{region}'; valid regions are " f"{list(REGION_CLASSES)}."
        )
    return np.isin(classes, REGION_CLASSES[region])


def volume_mm3(mask: np.ndarray, geom: CaseGeometry) -> float:
    """Volume of a binary mask in mm^3.

    Args:
        mask: Boolean (or 0/1) array, `(D, H, W)`.
        geom: Geometry supplying the per-voxel volume.

    Returns:
        `mask.sum() * geom.voxel_volume_mm3`, as a plain float.
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "volume_mm3")
    return float(mask.sum()) * geom.voxel_volume_mm3


def compute_volumes(classes: np.ndarray, geom: CaseGeometry) -> dict[str, float]:
    """Per-class and per-region volumes, in mm^3.

    Args:
        classes: Integer class map, `(D, H, W)`, values in `{0, 1, 2, 3}`.
        geom: Geometry supplying the per-voxel volume.

    Returns:
        Dict with keys `vol_NCR_mm3`, `vol_ED_mm3`, `vol_ET_mm3` (per-class)
        and `vol_TC_mm3`, `vol_WT_mm3` (per-region).
    """
    _validate_classes(classes)
    return {
        "vol_NCR_mm3": volume_mm3(classes == CLASS_IDS["NCR"], geom),
        "vol_ED_mm3": volume_mm3(classes == CLASS_IDS["ED"], geom),
        "vol_ET_mm3": volume_mm3(classes == CLASS_IDS["ET"], geom),
        "vol_TC_mm3": volume_mm3(np.isin(classes, REGION_CLASSES["TC"]), geom),
        "vol_WT_mm3": volume_mm3(np.isin(classes, REGION_CLASSES["WT"]), geom),
    }


def compute_fractions(volumes: Mapping[str, float]) -> dict[str, float]:
    """Composition ratios derived from a `compute_volumes` dict.

    Every ratio is NaN when its denominator is exactly 0.0 -- never 0.0 and
    never a raise. 0/0 is undefined, not "no necrosis" or "no oedema".

    Args:
        volumes: A mapping with (at least) the keys `compute_volumes`
            returns.

    Returns:
        Dict with keys `frac_enhancing_of_wt`, `frac_necrotic_of_wt`,
        `frac_edema_of_wt`, `frac_enhancing_of_tc`, `frac_necrotic_of_tc`,
        `ratio_edema_to_core`.
    """
    ncr = volumes["vol_NCR_mm3"]
    ed = volumes["vol_ED_mm3"]
    et = volumes["vol_ET_mm3"]
    tc = volumes["vol_TC_mm3"]
    wt = volumes["vol_WT_mm3"]
    return {
        "frac_enhancing_of_wt": _safe_div(et, wt),
        "frac_necrotic_of_wt": _safe_div(ncr, wt),
        "frac_edema_of_wt": _safe_div(ed, wt),
        "frac_enhancing_of_tc": _safe_div(et, tc),
        "frac_necrotic_of_tc": _safe_div(ncr, tc),
        "ratio_edema_to_core": _safe_div(ed, tc),
    }


def connected_components(
    mask: np.ndarray,
    geom: CaseGeometry,
    *,
    min_volume_mm3: float = 50.0,
    connectivity: int = 3,
) -> ComponentStats:
    """Connected-component (multifocality) summary of a binary mask.

    `connectivity` defaults to 3 (26-neighbourhood: any voxel touching by a
    face, edge, or corner counts as connected), which deliberately DIFFERS
    from `neurovision.inference.postprocess.remove_small_components`'s
    default of 1 (6-neighbourhood, face-adjacent only). Those two defaults
    serve opposite goals: postprocessing uses low connectivity to be
    aggressive about REMOVING speckle false positives, whereas counting
    tumour foci here must be conservative about CLAIMING multifocality --
    under 6-connectivity, a diagonal staircase ridge on an ordinary
    irregular tumour boundary can split into spurious separate components
    that a clinician reading "2 lesions" in a report would take literally.

    Args:
        mask: Boolean array, `(D, H, W)`.
        geom: Geometry supplying the per-voxel volume.
        min_volume_mm3: Components smaller than this (by volume, not voxel
            count) are dropped before counting and reporting.
        connectivity: Passed to `scipy.ndimage.generate_binary_structure(3, ...)`.
            `1` = 6-neighbourhood (face-adjacent only), `3` = 26-neighbourhood
            (face, edge, or corner adjacent).

    Returns:
        A `ComponentStats`. An empty mask gives `n=0`, `volumes_mm3=()`,
        `total_volume_mm3=0.0`, `largest_frac=nan`. A non-empty mask whose
        every component falls below `min_volume_mm3` gives `n=0`,
        `volumes_mm3=()`, a non-zero `total_volume_mm3`, and
        `largest_frac=nan`.
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "connected_components")

    total_volume_mm3 = volume_mm3(mask, geom)
    if not mask.any():
        return ComponentStats(n=0, volumes_mm3=(), total_volume_mm3=0.0, largest_frac=float("nan"))

    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled, num = ndimage.label(mask, structure=structure)

    counts = np.bincount(labeled.ravel(), minlength=num + 1)[1 : num + 1]
    component_volumes = counts.astype(np.float64) * geom.voxel_volume_mm3
    component_volumes = component_volumes[component_volumes >= min_volume_mm3]

    volumes_sorted = tuple(sorted((float(v) for v in component_volumes), reverse=True))
    n = len(volumes_sorted)
    largest_frac = volumes_sorted[0] / total_volume_mm3 if n > 0 else float("nan")

    return ComponentStats(
        n=n,
        volumes_mm3=volumes_sorted,
        total_volume_mm3=total_volume_mm3,
        largest_frac=largest_frac,
    )


def surface_area_mm2(mask: np.ndarray, geom: CaseGeometry) -> float:
    """Marching-cubes surface area of a binary mask, in mm^2.

    The mask is zero-padded by 1 voxel on every side BEFORE meshing.
    Without the pad, a lesion touching a volume face produces an OPEN
    surface (the face itself is not meshed) and the area comes out
    understated -- silently, with no error, and worse the closer the tumour
    sits to the array boundary.

    This is deliberately NOT face counting (summing exposed voxel faces).
    The face-counted surface area of a digitised sphere converges to about
    1.5x the true area as resolution increases, which would make a perfect
    sphere score `sphericity` around 0.67 instead of near 1.0 -- an artifact
    of the counting method, not a real shape measurement. Marching cubes on
    a raw binary mask (no signed-distance field, no sub-voxel surface
    localisation -- exactly what this function does) still carries a
    smaller, well-known systematic overestimate of its own: measured here at
    roughly 8-10% for a sphere, essentially flat across radius, because
    every crossing vertex sits at the fixed midpoint between a 0 voxel and a
    1 voxel rather than at the surface's true sub-voxel position. So a
    perfect digital sphere scores `sphericity` around 0.92, not 1.00 -- see
    `tests/test_burden.py::test_sphericity_of_a_digital_sphere_is_near_one`
    for the measured bound this module is pinned against.

    Args:
        mask: Boolean array, `(D, H, W)`.
        geom: Geometry supplying voxel spacing for the mesh.

    Returns:
        Surface area in mm^2, or NaN for an empty mask.
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "surface_area_mm2")
    if not mask.any():
        return float("nan")

    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    verts, faces, _normals, _values = measure.marching_cubes(
        padded.astype(np.float32), level=0.5, spacing=geom.spacing
    )
    return float(measure.mesh_surface_area(verts, faces))


def sphericity(mask: np.ndarray, geom: CaseGeometry) -> float:
    """Dimensionless shape compactness, 1.0 for a perfect sphere.

    `psi = pi^(1/3) * (6V)^(2/3) / A`, with `V` the mask volume and `A` its
    marching-cubes surface area.

    Args:
        mask: Boolean array, `(D, H, W)`.
        geom: Geometry supplying voxel spacing and volume.

    Returns:
        Sphericity, or NaN if the mask is empty or its surface area is 0 or
        NaN.
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "sphericity")
    if not mask.any():
        return float("nan")

    area = surface_area_mm2(mask, geom)
    if area == 0.0 or math.isnan(area):
        return float("nan")

    vol = volume_mm3(mask, geom)
    return float(math.pi ** (1.0 / 3.0) * (6.0 * vol) ** (2.0 / 3.0) / area)


def laterality(mask: np.ndarray, geom: CaseGeometry) -> dict[str, float | str]:
    """Hemispheric split of a binary mask about the mid-sagittal plane.

    Voxels at axis-0 index `i < geom.midline_index` are on the LOW-index
    side; `i > geom.midline_index` the HIGH-index side. `geom.midline_index`
    is `119.5` by default, so no voxel sits exactly on the plane. If it
    happens to be an exact integer, that plane's voxels are assigned to
    NEITHER side (logged at DEBUG) rather than silently folded into one
    hemisphere.

    Args:
        mask: Boolean array, `(D, H, W)`.
        geom: Geometry supplying the midline index, spacing, and left/right
            convention.

    Returns:
        Dict with `vol_right_mm3`, `vol_left_mm3` (floats),
        `frac_left` (= left / (left + right), NaN when the mask is empty),
        `frac_contralateral` (= `min(frac_left, 1 - frac_left)`, NaN when
        empty -- the direct "how much sits in the other hemisphere" signal),
        and `dominant_side` (`"right"`, `"left"`, or `""` when the mask is
        empty; an exact tie returns `"right"`).
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "laterality")

    n_axis0 = mask.shape[0]
    idx = np.arange(n_axis0).reshape(-1, 1, 1)

    if float(geom.midline_index).is_integer():
        logger.debug(
            "laterality: midline_index %.1f is an exact integer; that plane's voxels are "
            "assigned to neither hemisphere.",
            geom.midline_index,
        )

    lo_mask = mask & (idx < geom.midline_index)
    hi_mask = mask & (idx > geom.midline_index)

    lo_vol = volume_mm3(lo_mask, geom)
    hi_vol = volume_mm3(hi_mask, geom)

    if geom.left_is_high_index:
        vol_left, vol_right = hi_vol, lo_vol
    else:
        vol_left, vol_right = lo_vol, hi_vol

    total = vol_left + vol_right
    if total == 0.0:
        frac_left = float("nan")
        frac_contralateral = float("nan")
        dominant_side = ""
    else:
        frac_left = vol_left / total
        frac_contralateral = min(frac_left, 1.0 - frac_left)
        # Exact tie resolves to "right" -- documented, not incidental.
        dominant_side = "left" if vol_left > vol_right else "right"

    return {
        "vol_right_mm3": vol_right,
        "vol_left_mm3": vol_left,
        "frac_left": frac_left,
        "frac_contralateral": frac_contralateral,
        "dominant_side": dominant_side,
    }


def centroid(mask: np.ndarray, geom: CaseGeometry) -> tuple[float, float, float]:
    """Volume-weighted centroid of a binary mask, in voxel index coordinates.

    Args:
        mask: Boolean array, `(D, H, W)`.
        geom: Unused directly (kept for API symmetry with the other
            per-mask functions); the centroid is reported in voxel indices,
            not mm, since it is primarily used to locate a lesion within the
            array rather than to measure a physical distance.

    Returns:
        `(i, j, k)` as floats, or `(nan, nan, nan)` for an empty mask.
    """
    mask = np.asarray(mask, dtype=bool)
    _validate_mask(mask, "centroid")
    if not mask.any():
        return (float("nan"), float("nan"), float("nan"))

    coords = np.argwhere(mask)
    means = coords.mean(axis=0)
    return (float(means[0]), float(means[1]), float(means[2]))


def estimate_midline_index(brain_mask: np.ndarray, *, search_radius: float = 15.0) -> float:
    """Finds the axis-0 plane that maximises left/right mirror symmetry.

    A validation helper for the `midline_index = 119.5` assumption baked
    into `CaseGeometry`'s default -- NOT called by `burden_profile` itself.
    Intended use: run this on a real brain mask (e.g. a WT union across many
    cases, or a single case's nonzero-image mask) and check the result sits
    close to 119.5.

    Searched on a 0.5-voxel grid over `[c - search_radius, c + search_radius]`
    where `c = (N - 1) / 2` and `N = brain_mask.shape[0]`. For each candidate
    plane `m`, the array is reflected about `m` with explicit slicing (no
    Python loop over voxels) and scored against the original by Dice; the
    best-scoring `m` is returned.

    Args:
        brain_mask: Boolean array, `(D, H, W)`.
        search_radius: Half-width of the search window around the grid
            centre, in voxels.

    Returns:
        The candidate `m` (a multiple of 0.5) with the highest mirror-Dice
        score.

    Raises:
        ValueError: If `brain_mask` is empty.
    """
    mask = np.asarray(brain_mask, dtype=bool)
    _validate_mask(mask, "estimate_midline_index")
    if not mask.any():
        raise ValueError("estimate_midline_index: brain_mask is empty.")

    n_axis0 = mask.shape[0]
    c = (n_axis0 - 1) / 2.0
    candidates = np.arange(c - search_radius, c + search_radius + 1e-9, 0.5)

    mask_sum = float(mask.sum())
    best_m = float(candidates[0])
    best_score = -1.0

    for m in candidates:
        s = int(round(2 * m))
        lo = max(0, s - (n_axis0 - 1))
        hi = min(n_axis0 - 1, s)

        refl = np.zeros_like(mask)
        if lo <= hi:
            src_stop = s - hi - 1
            refl[lo : hi + 1] = mask[s - lo : (src_stop if src_stop >= 0 else None) : -1]

        intersection = float(np.logical_and(mask, refl).sum())
        denom = mask_sum + float(refl.sum())
        score = (2.0 * intersection / denom) if denom > 0.0 else 0.0

        if score > best_score:
            best_score = score
            best_m = float(m)

    return best_m


def burden_profile(
    classes: np.ndarray,
    geom: CaseGeometry,
    *,
    min_volume_mm3: float = 50.0,
    connectivity: int = 3,
) -> dict[str, float | int | str]:
    """Assembles one flat, CSV-ready burden-profile row for a single case.

    Deterministic: identical input produces an identical dict. Every value
    is a plain `float`, `int`, or `str` -- no arrays, tuples, or `None` --
    so the result can be written directly as one row of a per-case CSV.

    Args:
        classes: Integer class map, `(D, H, W)`, values in `{0, 1, 2, 3}`.
        geom: Geometry for volumes, surface areas, and laterality.
        min_volume_mm3: Passed through to `connected_components`.
        connectivity: Passed through to `connected_components`.

    Returns:
        A flat dict: the 5 `compute_volumes` keys, the 6 `compute_fractions`
        keys, and per-region (`ET`, `TC`, `WT`) component, shape, and
        laterality descriptors.
    """
    _validate_classes(classes)

    volumes = compute_volumes(classes, geom)
    fractions = compute_fractions(volumes)

    profile: dict[str, float | int | str] = {}
    profile.update(volumes)
    profile.update(fractions)

    for region in REGION_ORDER:
        mask = region_mask(classes, region)

        comp = connected_components(
            mask, geom, min_volume_mm3=min_volume_mm3, connectivity=connectivity
        )
        profile[f"n_components_{region}"] = comp.n
        profile[f"vol_largest_component_{region}_mm3"] = (
            comp.volumes_mm3[0] if comp.n >= 1 else float("nan")
        )
        profile[f"vol_second_component_{region}_mm3"] = (
            comp.volumes_mm3[1] if comp.n >= 2 else float("nan")
        )
        profile[f"largest_component_frac_{region}"] = comp.largest_frac

        vol = volume_mm3(mask, geom)
        area = surface_area_mm2(mask, geom)
        profile[f"surface_area_{region}_mm2"] = area
        profile[f"sphericity_{region}"] = sphericity(mask, geom)
        profile[f"surface_to_volume_{region}"] = _safe_div(area, vol)

        lat = laterality(mask, geom)
        profile[f"vol_right_{region}_mm3"] = lat["vol_right_mm3"]
        profile[f"vol_left_{region}_mm3"] = lat["vol_left_mm3"]
        profile[f"frac_left_{region}"] = lat["frac_left"]
        profile[f"frac_contralateral_{region}"] = lat["frac_contralateral"]
        profile[f"dominant_side_{region}"] = lat["dominant_side"]

        c_i, c_j, c_k = centroid(mask, geom)
        profile[f"centroid_i_{region}"] = c_i
        profile[f"centroid_j_{region}"] = c_j
        profile[f"centroid_k_{region}"] = c_k

    return profile
