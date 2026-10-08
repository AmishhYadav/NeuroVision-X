"""Tests for scripts/build_hero_brain_mesh.py (synthetic volumes only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

_REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_hero", _REPO / "scripts/build_hero_brain_mesh.py"
)
hero = importlib.util.module_from_spec(_spec)
sys.modules["build_hero"] = hero
_spec.loader.exec_module(hero)


def _sphere_volume(shape=(40, 40, 40), centre=(20, 20, 20), radius=12.0) -> np.ndarray:
    grid = np.indices(shape).astype(np.float32)
    dist = np.sqrt(sum((grid[i] - centre[i]) ** 2 for i in range(3)))
    return np.where(dist <= radius, 1000.0, 0.0).astype(np.float32)


def _sphere_mesh():
    return hero.extract_surface(_sphere_volume(), sigma=1.0, level=500.0)


def _outward_fraction(verts, normals) -> float:
    radial = verts - verts.mean(axis=0)
    return float((np.einsum("ij,ij->i", radial, normals) > 0).mean())


def test_sphere_surface_closed_and_outward():
    verts, faces = _sphere_mesh()
    assert verts.dtype == np.float32 and faces.dtype == np.int64
    # Closed mesh: every edge is shared by exactly two faces.
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    normals = hero.vertex_normals(verts, faces)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    assert _outward_fraction(verts, normals) > 0.99


def test_largest_component_keeps_bigger_sphere():
    vol = _sphere_volume((40, 40, 70), (20, 20, 20), 12.0)
    vol = np.maximum(vol, _sphere_volume((40, 40, 70), (20, 20, 55), 6.0))
    verts, faces = hero.extract_surface(vol, 1.0, 500.0)
    v2, f2 = hero.largest_component(verts, faces)
    assert len(f2) < len(faces)
    assert v2[:, 2].max() < 40  # the small sphere (z around 55) is gone
    assert f2.max() == len(v2) - 1


def test_taubin_does_not_shrink():
    verts, faces = _sphere_mesh()
    centre = verts.mean(axis=0)
    before = np.linalg.norm(verts - centre, axis=1).mean()
    smooth = hero.taubin_smooth(verts, faces, iterations=10)
    after = np.linalg.norm(smooth - smooth.mean(axis=0), axis=1).mean()
    assert abs(after - before) / before < 0.03


def test_decimate_hits_target_and_is_valid():
    verts, faces = _sphere_mesh()
    target = len(faces) // 3
    v2, f2 = hero.decimate(verts, faces, target)
    assert abs(len(f2) - target) <= 0.15 * target
    assert f2.min() >= 0 and f2.max() < len(v2)
    assert ((f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])).all()
    assert len(np.unique(np.sort(f2, axis=1), axis=0)) == len(f2)
    assert len(np.unique(f2)) == len(v2)  # no unreferenced vertices
    # Winding survives: normals still point outward.
    assert _outward_fraction(v2, hero.vertex_normals(v2, f2)) > 0.95


def test_write_mesh_round_trip(tmp_path):
    verts, faces = _sphere_mesh()
    normals = hero.vertex_normals(verts, faces)
    paths = hero.write_mesh(tmp_path, "m", verts, normals, faces, {"extra": 1})
    np.testing.assert_array_equal(np.fromfile(paths["position"], "<f4").reshape(-1, 3), verts)
    np.testing.assert_array_equal(np.fromfile(paths["normal"], "<f4").reshape(-1, 3), normals)
    np.testing.assert_array_equal(np.fromfile(paths["index"], "<u4").reshape(-1, 3), faces)
    meta = json.loads(paths["meta"].read_text())
    assert meta["licence"] == "CC BY-SA" and "Rohlfing" in meta["citation"]
    assert meta["n_faces"] == len(faces) and meta["extra"] == 1


def test_scene_coords_centred_scaled_and_outward():
    # Off-centre ellipsoid-ish blob so axes are distinguishable.
    vol = _sphere_volume((50, 40, 60), (30, 15, 40), 10.0)
    verts, faces = hero.extract_surface(vol, 1.0, 500.0)
    scene, record = hero.to_scene_coords(verts, vol.shape, half_extent=0.9)
    lo, hi = scene.min(axis=0), scene.max(axis=0)
    np.testing.assert_allclose((lo + hi) / 2, 0.0, atol=1e-5)
    assert np.isclose(((hi - lo) / 2).max(), 0.9, atol=1e-5)
    assert record["scale"] > 0
    assert _outward_fraction(scene, hero.vertex_normals(scene, faces)) > 0.99


def test_scene_axes_follow_twin_convention():
    # Point further along voxel axis1 (anterior in SRI24) must get smaller scene Z
    # (+Z is posterior); axis2 (superior) -> +Y; axis0 (toward Left) -> +X.
    pts = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    scene, _ = hero.to_scene_coords(pts, (11, 11, 11))
    d = scene[1:] - scene[0]
    assert d[0, 0] > 0 and d[1, 2] < 0 and d[2, 1] > 0


def test_real_config_key_path_composes():
    with initialize_config_dir(version_base="1.3", config_dir=str(_REPO / "configs")):
        cfg = compose(config_name="config")
    hero_cfg = cfg.hero  # the path main() reads: Hydra puts the group under `hero`
    assert hero_cfg.template_path == "data/atlas/sri24/spgr.nii"
    assert hero_cfg.name == "sri24-cortex"
    assert hero_cfg.target_faces == 120000
    assert hero_cfg.keep_largest_component is True
    assert hero_cfg.reference_half_extent == 0.9
