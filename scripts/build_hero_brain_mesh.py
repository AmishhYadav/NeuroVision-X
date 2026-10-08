"""Builds the landing-page hero brain mesh from the SRI24 T1 template.

The old hero brain was a marching-cubes surface of a skull-stripped *mask*:
a smooth blob with no folds. This script instead iso-surfaces the real T1
*intensities* of the healthy SRI24 average brain at the grey-matter / CSF
boundary, which gives the pial surface with visible gyri and sulci.

Pipeline: blur -> marching cubes -> keep largest piece -> vertex-clustering
decimation -> Taubin smoothing -> normals -> map to the twin's scene axes ->
write `{name}-position.f32 / -normal.f32 / -index.u32 / -meta.json`, the
format `app/frontend/src/lib/loadBinary.ts` `loadMesh` reads.

SRI24 is CC BY-SA (Rohlfing et al. 2010); the meta.json carries the licence
and citation.

Example usage:

    python scripts/build_hero_brain_mesh.py
    python scripts/build_hero_brain_mesh.py hero.target_faces=60000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import nibabel as nib
import numpy as np
from omegaconf import DictConfig
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from skimage import measure

from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = str(_REPO_ROOT / "configs")

SOURCE = "SRI24 atlas, spgr T1 template — healthy adult average, not a patient"
LICENCE = "CC BY-SA"
CITATION = (
    "Rohlfing T, Zahr NM, Sullivan EV, Pfefferbaum A. The SRI24 multichannel atlas "
    "of normal adult human brain structure. Hum Brain Mapp. 2010;31(5):798-819."
)

# ---------------------------------------------------------------------------
# Axis mapping, derived from the two affines (not guessed).
#
# SRI24 spgr.nii affine = diag(-1, 1, 1): in RAS+ world, index i0 increases
# toward -x = Left, i1 toward +y = Anterior, i2 toward +z = Superior.
# BraTS (the source of the existing twin meshes) affine = diag(-1, -1, 1):
# i0 increases toward Left, i1 toward -y = Posterior, i2 toward Superior.
# The twin uses scene X = i0 (R->L), Y = i2 (I->S, up), Z = i1 (A->P), i.e.
# +Z = Posterior (twin-meta.json coord_note). In SRI24 i1 runs the opposite
# way (A, not P), so to put anatomy in the SAME place we need
#   scene X = +i0,  scene Y = +i2,  scene Z = -i1.
# As a matrix acting on (i0, i1, i2) that has determinant +1 (swap = -1,
# one sign flip = -1), so it is a pure rotation: triangle winding and outward
# normals survive it unchanged. `to_scene_coords` asserts this.
# ---------------------------------------------------------------------------
_AXIS_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],  # scene X
        [0.0, 0.0, 1.0],  # scene Y
        [0.0, -1.0, 0.0],  # scene Z
    ],
    dtype=np.float64,
)
COORD_NOTE = (
    "X = voxel axis0 (R->L), Y = voxel axis2 (I->S, up), Z = -voxel axis1 (A->P); "
    "matches twin-meta.json (d->X, w->Y, h->Z). Centred on bbox centre and scaled "
    "uniformly so the largest half-extent equals the twin brain's."
)


def extract_surface(
    volume: np.ndarray, sigma: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts an iso-surface from a 3D intensity volume.

    An iso-surface is the set of points where the (blurred) intensity equals
    `level`; marching cubes turns it into a triangle mesh.

    Args:
        volume: Intensity volume, shape `(D, H, W)` (a trailing singleton axis
            is squeezed away).
        sigma: Gaussian blur width in voxels applied first (0 disables it).
        level: Intensity of the iso-surface. Voxels brighter than `level` are
            "inside".

    Returns:
        `(verts, faces)`: float32 `(N, 3)` voxel-index coordinates and int64
        `(M, 3)` triangle indices, wound so face normals point outward.
    """
    vol = np.squeeze(np.asarray(volume)).astype(np.float32)
    if sigma > 0:
        vol = ndimage.gaussian_filter(vol, sigma=sigma)
    verts, faces, _, _ = measure.marching_cubes(vol, level=level, step_size=1)
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)
    # marching_cubes' winding depends on its inside/outside convention. Check
    # the signed volume (positive = counter-clockwise faces point outward) and
    # flip the winding if the mesh is inside-out.
    if _signed_volume(verts, faces) < 0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


def _signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a mesh; positive when faces are wound outward."""
    v = verts.astype(np.float64)
    v = v - v.mean(axis=0)  # centring makes this meaningful for open meshes
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _compact(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drops unreferenced vertices and reindexes faces."""
    used, inverse = np.unique(faces.ravel(), return_inverse=True)
    return verts[used], inverse.reshape(-1, 3).astype(np.int64)


def largest_component(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keeps the connected component (by shared vertices) with the most faces.

    Args:
        verts: `(N, 3)` vertex positions.
        faces: `(M, 3)` triangle indices.

    Returns:
        `(verts, faces)` of the biggest component, vertices reindexed.
    """
    n = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = csgraph.connected_components(graph, directed=False)
    face_labels = labels[faces[:, 0]]  # all 3 corners of a face share a label
    keep = np.argmax(np.bincount(face_labels))
    return _compact(verts, faces[face_labels == keep])


def taubin_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Taubin (shrink-free) uniform-Laplacian smoothing.

    Each iteration moves every vertex toward the mean of its neighbours
    (factor `lam > 0`, smooths but shrinks) and then away from it
    (factor `mu < 0`, with |mu| slightly larger than lam, undoing the shrink).

    Args:
        verts: `(N, 3)` vertex positions.
        faces: `(M, 3)` triangle indices.
        iterations: Number of lam+mu pairs.
        lam: Positive smoothing step.
        mu: Negative inflating step.

    Returns:
        float32 `(N, 3)` smoothed positions.
    """
    n = len(verts)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adj = adj + adj.T
    adj.data[:] = 1.0  # binary adjacency: an edge shared by 2 faces counts once
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1), 0.0)
    mean_op = sparse.diags(inv_degree) @ adj  # row i: average of i's neighbours

    x = verts.astype(np.float64)
    for _ in range(iterations):
        for step in (lam, mu):
            laplacian = mean_op @ x - x
            # Isolated vertices have no neighbours: mean_op row is 0, so
            # laplacian = -x; mask them so they stay put.
            laplacian[degree == 0] = 0.0
            x = x + step * laplacian
    return x.astype(np.float32)


def _cluster(verts: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """One vertex-clustering pass at a given grid cell size."""
    origin = verts.min(axis=0)
    ijk = np.floor((verts - origin) / cell).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, cluster_id = np.unique(key, return_inverse=True)
    cluster_id = cluster_id.ravel()
    k = cluster_id.max() + 1
    counts = np.bincount(cluster_id, minlength=k).astype(np.float64)
    new_verts = np.stack(
        [np.bincount(cluster_id, weights=verts[:, a], minlength=k) / counts for a in range(3)],
        axis=1,
    )
    new_faces = cluster_id[faces]
    # Drop degenerate faces (two corners merged into one cell).
    ok = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[ok]
    # Drop duplicate faces (same vertex set), keeping the first's winding.
    _, first = np.unique(np.sort(new_faces, axis=1), axis=0, return_index=True)
    new_faces = new_faces[np.sort(first)]
    return _compact(new_verts.astype(np.float32), new_faces)


def decimate(
    verts: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduces face count by vertex clustering.

    Vertices are snapped to a uniform grid; all vertices in one cell merge
    into their mean position, faces are remapped, and faces that collapsed
    (or duplicated another) are dropped. The cell size is found by bisection
    so the result lands within +-15% of `target_faces`.

    Vertex clustering is approximate: it ignores curvature, so it can nick
    thin features and leave a few non-manifold edges, unlike quadric-error
    decimation. That is acceptable for a decorative hero visual, where we
    only need a lighter mesh that still shows the folds, and it needs no
    extra dependency.

    Args:
        verts: `(N, 3)` vertex positions.
        faces: `(M, 3)` triangle indices.
        target_faces: Desired number of faces.

    Returns:
        `(verts, faces)` with float32 / int64 dtypes. Returned unchanged
        (apart from dtype) if already at or below the target.
    """
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)
    if len(faces) <= target_faces:
        return verts, faces

    # Bisect on log(cell size): bigger cells -> fewer faces (monotone enough).
    lo = 1e-3  # tiny cell: nothing merges
    hi = float((verts.max(axis=0) - verts.min(axis=0)).max())  # one cell: everything merges
    best = (verts, faces)
    best_err = abs(len(faces) - target_faces)
    for _ in range(30):
        cell = float(np.sqrt(lo * hi))
        v, f = _cluster(verts, faces, cell)
        err = abs(len(f) - target_faces)
        if err < best_err:
            best, best_err = (v, f), err
        if err <= 0.03 * target_faces:
            break
        if len(f) > target_faces:
            lo = cell
        else:
            hi = cell
    if best_err > 0.15 * target_faces:
        logger.warning("decimate: %d faces is outside +-15%% of %d", len(best[1]), target_faces)
    return best


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals.

    For faces wound counter-clockwise when seen from outside, the cross
    product of two edges points outward and its length is twice the triangle
    area, so summing raw cross products per vertex area-weights for free.

    Args:
        verts: `(N, 3)` vertex positions.
        faces: `(M, 3)` triangle indices (outward winding).

    Returns:
        float32 `(N, 3)` unit normals (zero vector for unreferenced vertices).
    """
    v = verts.astype(np.float64)
    face_n = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    normals = np.zeros_like(v)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(length, 1e-12)
    return normals.astype(np.float32)


def to_scene_coords(
    verts: np.ndarray, shape: tuple[int, ...], half_extent: float = 0.9
) -> tuple[np.ndarray, dict[str, Any]]:
    """Maps voxel-index coordinates to the twin's scene axes, centred and scaled.

    See the derivation comment at `_AXIS_MATRIX`. The mapping is a proper
    rotation (determinant +1), so face winding and normals need no fixing.

    Args:
        verts: `(N, 3)` voxel-index coordinates `(i0, i1, i2)`.
        shape: Volume shape (kept for the record; centring uses the mesh's own
            bounding box, like the twin).
        half_extent: Target largest half-extent of the result, in scene units.

    Returns:
        `(scene_verts, record)`: float32 `(N, 3)` positions and a dict with
        `centre` (in scene axes before centring) and `scale`.
    """
    if np.linalg.det(_AXIS_MATRIX) <= 0:  # would reverse winding
        raise ValueError("axis mapping must be a proper rotation to keep winding")
    mapped = verts.astype(np.float64) @ _AXIS_MATRIX.T
    lo, hi = mapped.min(axis=0), mapped.max(axis=0)
    centre = (lo + hi) / 2.0
    scale = half_extent / float(((hi - lo) / 2.0).max())
    out = ((mapped - centre) * scale).astype(np.float32)
    record = {
        "centre": [float(c) for c in centre],
        "scale": scale,
        "half_extent": half_extent,
        "volume_shape": [int(s) for s in shape],
    }
    return out, record


def write_mesh(
    out_dir: Path,
    name: str,
    verts: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Writes the three binary buffers and `{name}-meta.json`.

    Args:
        out_dir: Output directory (created if missing).
        name: File prefix.
        verts: `(N, 3)` positions.
        normals: `(N, 3)` normals.
        faces: `(M, 3)` triangle indices.
        meta: Extra entries merged into the meta.json.

    Returns:
        Mapping of `position`, `normal`, `index`, `meta` to written paths.
    """
    out = ensure_dir(out_dir)
    paths = {
        "position": out / f"{name}-position.f32",
        "normal": out / f"{name}-normal.f32",
        "index": out / f"{name}-index.u32",
        "meta": out / f"{name}-meta.json",
    }
    np.ascontiguousarray(verts, dtype="<f4").tofile(paths["position"])
    np.ascontiguousarray(normals, dtype="<f4").tofile(paths["normal"])
    np.ascontiguousarray(faces, dtype="<u4").tofile(paths["index"])
    record = {
        "source": SOURCE,
        "licence": LICENCE,
        "citation": CITATION,
        "coord_note": COORD_NOTE,
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
    }
    record.update(meta or {})
    write_json(record, paths["meta"])
    return paths


def _resolve(path: str) -> Path:
    """Resolves a config path against the repo root unless it is absolute."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def build(cfg: DictConfig) -> dict[str, Path]:
    """Runs the full pipeline for a `cfg.hero` config block.

    Args:
        cfg: The `hero` config sub-tree (see `configs/hero/default.yaml`).

    Returns:
        Paths of the written files.
    """
    volume = np.asarray(nib.load(str(_resolve(cfg.template_path))).dataobj)
    shape = tuple(np.squeeze(volume).shape)
    verts, faces = extract_surface(volume, float(cfg.sigma), float(cfg.level))
    logger.info("marching cubes: %d verts, %d faces", len(verts), len(faces))
    if cfg.keep_largest_component:
        verts, faces = largest_component(verts, faces)
        logger.info("largest component: %d verts, %d faces", len(verts), len(faces))
    verts, faces = decimate(verts, faces, int(cfg.target_faces))
    logger.info("decimated: %d verts, %d faces", len(verts), len(faces))
    verts = taubin_smooth(verts, faces, int(cfg.smooth_iterations))
    logger.info("Taubin smoothing: %d iterations", int(cfg.smooth_iterations))
    verts, scene = to_scene_coords(verts, shape, float(cfg.reference_half_extent))
    normals = vertex_normals(verts, faces)
    params = {
        k: cfg[k] for k in ("template_path", "sigma", "level", "smooth_iterations", "target_faces")
    }
    params["keep_largest_component"] = bool(cfg.keep_largest_component)
    meta = {"parameters": params, **scene}
    paths = write_mesh(_resolve(cfg.out_dir), cfg.name, verts, normals, faces, meta)
    for key, path in paths.items():
        logger.info("wrote %s (%d bytes)", path, path.stat().st_size)
    return paths


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Builds the hero brain mesh described by `cfg.hero`."""
    setup_logging(level="INFO")
    build(cfg.hero)


if __name__ == "__main__":
    main()
