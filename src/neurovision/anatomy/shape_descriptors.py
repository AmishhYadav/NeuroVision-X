"""Geometric shape descriptors for the tumour regions of a segmentation.

This is a NEW module, deliberately kept separate from `neurovision.anatomy.burden`.
`burden.py` is the producer of the published `burden.csv` -- every key it emits is
frozen and this module never touches it, only imports `CaseGeometry` and
`region_mask` from it. Everything in `shape_profile` below is a PURELY GEOMETRIC
description of the segmented shape (how elongated, how flat, how thick a wall) --
"tumour extent", "rim thickness" -- and never a clinical judgement about what that
shape means for a patient, its severity, or its likely course. A reporting layer
downstream scans rendered text for exactly that kind of language, and a key name or
comment leaking one in would defeat the scan.

Regions and label convention follow `burden.py` exactly (see that module's
docstring): classes are `{0, 1, 2, 3}` = background / NCR / ED / ET, and regions are
the nested unions `ET = {3}`, `TC = {1, 3}`, `WT = {1, 2, 3}`.

Every quantity below is computed from voxel coordinates converted to millimetres via
`geom.spacing` (so an anisotropic voxel grid does not distort the shape), and every
value is a plain `float`, using NaN where the quantity is undefined for an empty
region -- never a raise, never a warning, matching `burden.py`'s convention.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy import ndimage

from neurovision.anatomy.burden import CLASS_IDS, REGION_ORDER, CaseGeometry, region_mask

__all__ = ["shape_profile", "shape_profile_keys"]

logger = logging.getLogger(__name__)

# Below this many voxels, a covariance-based principal-component analysis is too
# noisy to report a meaningful elongation or flatness ratio -- see `shape_profile`.
_MIN_VOXELS_FOR_PCA = 4


def _pca(coords_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal-component analysis of a point cloud in millimetres.

    Args:
        coords_mm: `(N, 3)` array of point coordinates in mm, `N >= 1`.

    Returns:
        `(eigvals, eigvecs)`: eigenvalues sorted DESCENDING (`eigvals[0]` is the
        largest), and `eigvecs[:, 0]` the matching unit eigenvector for the
        largest eigenvalue. Uses the population covariance (divide by `N`, not
        `N - 1`) so a single point gives the well-defined zero matrix instead of
        `numpy.cov`'s divide-by-zero warning at `N == 1`.
    """
    n = coords_mm.shape[0]
    mean = coords_mm.mean(axis=0)
    centered = coords_mm - mean
    cov = (centered.T @ centered) / n
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def _sign_normalized_axis(v: np.ndarray) -> np.ndarray:
    """Flips `v` so its largest-magnitude component is positive.

    Makes the reported principal axis deterministic: an eigenvector and its
    negation describe the same axis, and without a fixed convention the sign
    returned by `numpy.linalg.eigh` is arbitrary.

    Args:
        v: A 3-element unit vector.

    Returns:
        `v` or `-v`, whichever has a positive value at the index of largest
        `abs(v)` (the first such index, on a tie).
    """
    idx = int(np.argmax(np.abs(v)))
    return -v if v[idx] < 0.0 else v


def _rim_thickness_et(
    et_mask: np.ndarray,
    core_mask: np.ndarray,
    tc_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Median and max wall thickness of the ET shell, in mm, plus a core flag.

    ET typically forms a shell wrapped around the necrotic core (class NCR).
    When a core is present this uses a TWO-SIDED estimator that is exact for a
    shell of uniform thickness: for every ET voxel,

    - `d_out = distance_transform_edt(tc_mask, sampling=spacing)`: distance to
      the nearest voxel OUTSIDE the tumour core region (TC = ET union core),
      i.e. how far the voxel sits from the shell's OUTER surface.
    - `d_in = distance_transform_edt(~core_mask, sampling=spacing)`: distance
      to the nearest necrotic-core voxel, i.e. how far the voxel sits from the
      shell's INNER surface.
    - `thickness = d_out + d_in - mean(spacing)`. Both distance transforms
      measure to voxel CENTRES, not faces, so each term individually
      overshoots the true distance to the surface by about half a voxel;
      subtracting the mean spacing (one "half voxel" from each side) corrects
      for that. For a uniform shell this is exact everywhere in the shell, not
      only on its medial axis (unlike doubling a single one-sided distance):
      e.g. a 3-voxel-thick isotropic shell gives thickness 3.0 at its inner,
      middle, and outer voxels alike.

    When there is NO core at all (a solid ET blob), `tc_mask` degenerates to
    `et_mask` and there is nothing for `d_in` to measure -- this falls back to
    `thickness = 2 * distance_transform_edt(et_mask, sampling=spacing)`, i.e.
    twice the local inradius (the diameter, for a ball). That fallback is
    exact only at the medial axis; it is documented as an approximation
    elsewhere in this module for exactly that reason.

    Args:
        et_mask: Boolean `(D, H, W)` mask of the ET region.
        core_mask: Boolean `(D, H, W)` mask of the necrotic-core region.
        tc_mask: Boolean `(D, H, W)` mask of the tumour-core region (ET union
            core).
        spacing: Voxel spacing in mm, `(D, H, W)` axis order.

    Returns:
        `(median_mm, max_mm, has_core)`. `has_core` is `1.0` when the two-sided
        estimator was used (a core is present), `0.0` when the single-sided
        fallback was used. All three are `nan` if `et_mask` is empty.
    """
    if not et_mask.any():
        return float("nan"), float("nan"), float("nan")

    if core_mask.any():
        d_out = ndimage.distance_transform_edt(tc_mask, sampling=spacing)
        d_in = ndimage.distance_transform_edt(~core_mask, sampling=spacing)
        mean_spacing = float(np.mean(spacing))
        thickness = d_out[et_mask] + d_in[et_mask] - mean_spacing
        has_core = 1.0
    else:
        edt = ndimage.distance_transform_edt(et_mask, sampling=spacing)
        thickness = 2.0 * edt[et_mask]
        has_core = 0.0

    return float(np.median(thickness)), float(np.max(thickness)), has_core


def shape_profile(classes: np.ndarray, geom: CaseGeometry) -> dict[str, float]:
    """Assembles one flat, geometric shape-profile row for a single case.

    Deterministic: identical input produces an identical dict. Every value is a
    plain `float` (NaN where undefined, e.g. an empty region) -- no arrays,
    tuples, or `None` -- so the result is safe to write as one row of a CSV, the
    same convention `burden.burden_profile` uses (this module does not touch
    that one; see the module docstring).

    For each region `R` in `burden.REGION_ORDER` (`ET`, `TC`, `WT`), voxel
    coordinates are converted to millimetres (`voxel index * geom.spacing`)
    and a principal-component analysis of that point cloud gives eigenvalues
    `lambda1 >= lambda2 >= lambda3`:

    - `elongation_R = sqrt(lambda1 / lambda2)`: near 1 for a sphere, much
      greater than 1 for a rod.
    - `flatness_R = sqrt(lambda3 / lambda2)`: near 1 for a sphere, much less
      than 1 for a slab.

    Both are NaN when the region has fewer than 4 voxels, or when `lambda2` is
    exactly 0 (a degenerate point cloud, e.g. every voxel collinear) -- dividing
    by either would be meaningless, never a real ratio.

    Also per region:

    - `extent_R_i_mm`, `extent_R_j_mm`, `extent_R_k_mm`: the bounding-box
      extent along each voxel axis, in mm (`(max_index - min_index + 1) *
      spacing`).
    - `principal_axis_R_i`, `principal_axis_R_j`, `principal_axis_R_k`: the
      unit eigenvector for `lambda1`, in `(i, j, k)` voxel-axis order and
      sign-normalised (see `_sign_normalized_axis`) so the result is the same
      regardless of which of the two opposite directions `eigh` happened to
      return.
    - `n_voxels_R`: the region's voxel count, as a float, so a reader can judge
      how much data the other statistics for that region rest on.

    Plus, ET only, `rim_thickness_ET_median_mm`, `rim_thickness_ET_max_mm`, and
    `rim_thickness_ET_has_core` (see `_rim_thickness_et`).

    Args:
        classes: Integer class map, `(D, H, W)`, values in `{0, 1, 2, 3}`.
        geom: Geometry supplying voxel spacing in mm.

    Returns:
        A flat dict of plain floats; see `shape_profile_keys` for the exact key
        order.
    """
    spacing = tuple(float(s) for s in geom.spacing)
    spacing_arr = np.array(spacing, dtype=np.float64)

    profile: dict[str, float] = {}

    for region in REGION_ORDER:
        mask = region_mask(classes, region)
        coords = np.argwhere(mask)  # (N, 3) voxel indices, (i, j, k) order
        n_voxels = coords.shape[0]

        if n_voxels == 0:
            elongation = float("nan")
            flatness = float("nan")
            extent = (float("nan"), float("nan"), float("nan"))
            axis = (float("nan"), float("nan"), float("nan"))
        else:
            coords_mm = coords.astype(np.float64) * spacing_arr
            eigvals, eigvecs = _pca(coords_mm)
            lam1, lam2, lam3 = (float(v) for v in eigvals)

            if n_voxels < _MIN_VOXELS_FOR_PCA or lam2 == 0.0:
                elongation = float("nan")
                flatness = float("nan")
            else:
                elongation = math.sqrt(lam1 / lam2)
                flatness = math.sqrt(lam3 / lam2)

            mins = coords.min(axis=0).astype(np.float64)
            maxs = coords.max(axis=0).astype(np.float64)
            extent = tuple(float(e) for e in (maxs - mins + 1.0) * spacing_arr)
            axis = tuple(float(a) for a in _sign_normalized_axis(eigvecs[:, 0]))

        profile[f"elongation_{region}"] = elongation
        profile[f"flatness_{region}"] = flatness
        profile[f"extent_{region}_i_mm"] = extent[0]
        profile[f"extent_{region}_j_mm"] = extent[1]
        profile[f"extent_{region}_k_mm"] = extent[2]
        profile[f"principal_axis_{region}_i"] = axis[0]
        profile[f"principal_axis_{region}_j"] = axis[1]
        profile[f"principal_axis_{region}_k"] = axis[2]
        profile[f"n_voxels_{region}"] = float(n_voxels)

        if region == "ET":
            core_mask = classes == CLASS_IDS["NCR"]
            tc_mask = region_mask(classes, "TC")
            median_mm, max_mm, has_core = _rim_thickness_et(mask, core_mask, tc_mask, spacing)
            profile["rim_thickness_ET_median_mm"] = median_mm
            profile["rim_thickness_ET_max_mm"] = max_mm
            profile["rim_thickness_ET_has_core"] = has_core

    return profile


def shape_profile_keys() -> tuple[str, ...]:
    """The exact key order `shape_profile` builds its dict in.

    Data-independent (derived only from `burden.REGION_ORDER`), so it can be
    used to pin the output shape for a report renderer without running the
    computation on real data.

    Returns:
        A tuple of key names in insertion order.
    """
    keys: list[str] = []
    for region in REGION_ORDER:
        keys.extend(
            [
                f"elongation_{region}",
                f"flatness_{region}",
                f"extent_{region}_i_mm",
                f"extent_{region}_j_mm",
                f"extent_{region}_k_mm",
                f"principal_axis_{region}_i",
                f"principal_axis_{region}_j",
                f"principal_axis_{region}_k",
                f"n_voxels_{region}",
            ]
        )
        if region == "ET":
            keys.extend(
                [
                    "rim_thickness_ET_median_mm",
                    "rim_thickness_ET_max_mm",
                    "rim_thickness_ET_has_core",
                ]
            )
    return tuple(keys)
