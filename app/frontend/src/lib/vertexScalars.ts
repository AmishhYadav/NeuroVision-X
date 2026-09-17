// Samples a uint8 (D,H,W) uncertainty-style volume (predictive entropy,
// the conformal band, or Grad-CAM - see api.ts's X-Uncertainty-Kind values)
// at the vertices of a twin mesh, so the mesh can be coloured by the same
// per-voxel quantity the 2D slice view already shows. Pure and
// worker-agnostic: it only knows about flat (D,H,W) buffers and (d,h,w)
// positions/normals, the same conventions as twinGeometry.ts and slicing.ts.

/**
 * Samples a uint8 (D,H,W) w-fastest volume at mesh vertices, one voxel
 * INWARD along the normal, nearest-neighbour. Returns a `Float32Array` of
 * length N normalised to [0, 1] (byte/255).
 *
 * Why inset: a surface-nets vertex sits exactly on the class boundary,
 * where predictive entropy is maximal BY CONSTRUCTION (the softmax is
 * split there), so sampling at the vertex would paint every tumour surface
 * uniformly "maximally uncertain" and say nothing. Sampling one voxel
 * inward (position - inset*normal, normals point outward) reads the
 * interior just inside the boundary, which is the quantity the 2D slice
 * shows one voxel in. The sample point is clamped to the volume bounds and
 * rounded to the nearest voxel.
 *
 * @param volume uint8 (D,H,W) w-fastest buffer, flat index d*H*W + h*W + w.
 * @param shape [D, H, W] dimensions of `volume`.
 * @param voxelPositions (N,3) vertex positions in (d,h,w) voxel coordinates.
 * @param normals (N,3) per-vertex outward unit normals, same (d,h,w) axis order.
 * @param insetVoxels how many voxels to step inward along the normal before sampling.
 */
export function sampleScalarsAtVertices(
  volume: Uint8Array,
  shape: [number, number, number],
  voxelPositions: Float32Array,
  normals: Float32Array,
  insetVoxels = 1,
): Float32Array {
  const [D, H, W] = shape;
  const n = voxelPositions.length / 3;
  const out = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const base = i * 3;
    const d = voxelPositions[base] - insetVoxels * normals[base];
    const h = voxelPositions[base + 1] - insetVoxels * normals[base + 1];
    const w = voxelPositions[base + 2] - insetVoxels * normals[base + 2];

    // Nearest-neighbour, then clamp to the volume bounds - an inset vertex
    // near a face of the volume can land outside it (e.g. d < 0), which must
    // not read out of the buffer or produce NaN.
    const di = clampIndex(Math.round(d), D);
    const hi = clampIndex(Math.round(h), H);
    const wi = clampIndex(Math.round(w), W);

    out[i] = volume[di * H * W + hi * W + wi] / 255;
  }

  return out;
}

function clampIndex(value: number, dimSize: number): number {
  return Math.min(dimSize - 1, Math.max(0, value));
}
