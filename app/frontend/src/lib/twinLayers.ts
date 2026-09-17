// Pure vertex-scalar sampling for tumour meshes: given voxel-space geometry
// already produced by the worker's mesh pass (positions/normals BEFORE
// voxelToScene/normalToScene - see twinGeometry.ts) and one or more uint8
// uncertainty-style volumes, samples each layer at each tumour class's
// vertices. Kept out of the worker (twinMesh.worker.ts) so this logic gets
// its own unit test without spinning up a Worker, and so it can be reused
// unchanged by BOTH the initial mesh pass (which has fresh raw geometry on
// hand) and a later layer-only "sample" message (which replays cached
// geometry to add a new layer without re-meshing).
import type { ClassName } from "../workers/twinMesh.worker";
import { sampleScalarsAtVertices } from "./vertexScalars";
import type { VertexLayerKind } from "./vertexColors";

/**
 * For every (kind -> volume) entry in `layers` and every (class -> geometry)
 * entry in `voxelGeometry`, samples `volume` at that class's vertices one
 * voxel inward (see vertexScalars.ts: a surface vertex sits exactly on the
 * class boundary, where an entropy-like quantity is maximal by construction,
 * so sampling AT the vertex would paint every tumour surface uniformly
 * "maximally uncertain" and say nothing).
 *
 * @param layers uint8 (D,H,W) volumes keyed by the backend's X-Uncertainty-Kind value.
 * @param shape [D, H, W] dimensions shared by every volume in `layers`.
 * @param voxelGeometry per-tumour-class (d,h,w) voxel positions/normals, from
 *   the SAME mesh pass that produced the scene-space mesh (i.e. BEFORE
 *   voxelToScene/normalToScene) - sampling must happen in voxel space
 *   because the layer volumes are indexed by voxel, not scene, coordinates.
 * @returns layerScalars[kind][class] = Float32Array(N) in [0,1]. The output
 *   keys mirror `layers`' keys exactly (a layer never requested is never
 *   invented), and within each layer, the classes mirror `voxelGeometry`'s
 *   keys (a class with no mesh has nothing to sample).
 */
export function sampleLayersForClasses(
  layers: Partial<Record<VertexLayerKind, Uint8Array>>,
  shape: [number, number, number],
  voxelGeometry: Partial<Record<ClassName, { positions: Float32Array; normals: Float32Array }>>,
): Partial<Record<VertexLayerKind, Partial<Record<ClassName, Float32Array>>>> {
  const out: Partial<Record<VertexLayerKind, Partial<Record<ClassName, Float32Array>>>> = {};

  for (const kind of Object.keys(layers) as VertexLayerKind[]) {
    const volume = layers[kind];
    if (!volume) continue;
    const perClass: Partial<Record<ClassName, Float32Array>> = {};
    for (const className of Object.keys(voxelGeometry) as ClassName[]) {
      const geom = voxelGeometry[className];
      if (!geom) continue;
      perClass[className] = sampleScalarsAtVertices(volume, shape, geom.positions, geom.normals, 1);
    }
    out[kind] = perClass;
  }

  return out;
}
