// Pure geometry helpers for the digital twin mesh pipeline: turning a
// scalar field into a (d, h, w)-voxel-space mesh, and mapping voxel
// coordinates into the right-handed anatomical scene frame the twin
// renders in. Kept separate from the worker (twinMesh.worker.ts) so this
// axis bookkeeping - easy to get backwards and hard to spot once it's
// wrong, because a mirrored or wrong-stride brain still looks like a
// brain - has its own unit tests, independent of worker-message plumbing.
import { surfaceNets } from "./surfaceNets";

export interface VoxelMesh {
  /** (N, 3) vertex positions in (d, h, w) voxel coordinates. */
  positions: Float32Array;
  /** (N, 3) per-vertex unit normals, same (d, h, w) axis order as positions. */
  normals: Float32Array;
  indices: Uint32Array;
}

/**
 * Surface-nets a (D, H, W) w-fastest scalar field - the layout every volume
 * buffer in this codebase uses, see slicing.ts's flat index
 * d*H*W + h*W + w - and returns a mesh in (d, h, w) voxel coordinates.
 *
 * surfaceNets() itself reads its input x-fastest
 * (index(x,y,z) = x + dx*(y + dy*z), see surfaceNets.ts). To read our
 * w-fastest buffer with the right strides, x must be w, y must be h, and z
 * must be d - i.e. we must call it with dims=[W,H,D], NOT [D,H,W]. Passing
 * [D,H,W] on a w-fastest buffer silently reinterprets the strides whenever
 * D !== W (every real BraTS case, e.g. 136x171x146) and produces a
 * smeared, wrong-shaped mesh with no error. The vertices surfaceNets hands
 * back are then in (w,h,d) axis order (its x,y,z), so we reorder every
 * position and every normal component into (d,h,w) here, once, so every
 * caller downstream (the worker, voxelToScene, tests) can think in
 * (d,h,w) like the rest of the codebase. Measured: surfaceNets emits
 * normals that point INWARD for a `field > isovalue` solid, with winding
 * consistent with those inward normals. Swapping two axis components is a
 * reflection (determinant -1) - it flips triangle winding but keeps a
 * normal field's inward/outward sense (positions and normals reflect
 * together) - so after this reorder, with NO index change, the winding is
 * already outward-facing CCW (what three.js FrontSide needs), while the
 * reordered normals are still inward. So we leave indices untouched and
 * negate every reordered normal component here, making both winding and
 * normals outward in (d,h,w).
 */
export function meshVoxelField(
  field: Float32Array | Uint8Array,
  shape: [number, number, number],
  isovalue: number,
): VoxelMesh {
  const [D, H, W] = shape;
  const raw = surfaceNets(field, [W, H, D], isovalue);
  return {
    positions: reorderWhdToDhw(raw.positions),
    normals: reorderAndNegateWhdToDhw(raw.normals),
    indices: raw.indices,
  };
}

// surfaceNets hands back (x,y,z) = (w,h,d) triplets; swap the first and
// third component of each to get (d,h,w). The middle component (h) never
// moves. Used for positions - a direct axis reorder, no sign change.
function reorderWhdToDhw(src: Float32Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    out[i] = src[i + 2]; // d
    out[i + 1] = src[i + 1]; // h
    out[i + 2] = src[i]; // w
  }
  return out;
}

// Same (w,h,d) -> (d,h,w) axis reorder as reorderWhdToDhw, but for normals:
// surfaceNets' normals point inward (measured), so on top of the reorder we
// negate every component to make them outward, matching the winding the
// axis-swap reflection already made outward-CCW for free (see
// meshVoxelField's docstring).
function reorderAndNegateWhdToDhw(src: Float32Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    out[i] = -src[i + 2]; // d
    out[i + 1] = -src[i + 1]; // h
    out[i + 2] = -src[i]; // w
  }
  return out;
}

/**
 * Bounding-box center and a uniform scale (so the mesh's longest axis maps
 * to a fixed scene extent) for a flat (N,3) position array. Axis-agnostic -
 * it only tracks min/max per component - so it works unchanged whether the
 * positions are in (d,h,w) voxel space or already in scene space.
 */
export function extentCenter(positions: Float32Array): {
  center: [number, number, number];
  scale: number;
} {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const center: [number, number, number] = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  return { center, scale: 1.8 / extent };
}

/**
 * (d, h, w) voxel coordinates -> right-handed anatomical scene coordinates:
 *
 *   X = (d - c_d) * scale    (+X = patient LEFT; d runs RIGHT -> LEFT)
 *   Y = (w - c_w) * scale    (+Y = SUPERIOR; w runs INFERIOR -> SUPERIOR)
 *   Z = -(h - c_h) * scale   (+Z = ANTERIOR; h runs ANTERIOR -> POSTERIOR,
 *                             hence the negation)
 *
 * (Left, Superior, Anterior) is a right-handed frame (L x S = A), so this
 * places the default +Z camera in front of the face looking at the back of
 * the head, with patient-left on screen-right - the same radiological
 * convention slicing.ts uses for the 2D views. The PREVIOUS mapping put
 * +Z = posterior (L x S = -Z there), which mirrored the whole brain; a
 * mirrored brain still looks like a brain, so nothing visual caught it.
 *
 * The linear part of this map (ignoring the center-subtract and uniform
 * scale) is the matrix taking (d,h,w) -> (d, w, -h). Its determinant is
 * +1: it is a rotation, not a reflection. So triangle winding computed in
 * voxel (d,h,w) space is preserved after this transform - callers do not
 * need to flip index order to compensate, unlike a genuinely mirrored
 * mapping would require.
 */
export function voxelToScene(
  positions: Float32Array,
  center: [number, number, number],
  scale: number,
): Float32Array {
  const out = new Float32Array(positions.length);
  const [cd, ch, cw] = center;
  for (let i = 0; i < positions.length; i += 3) {
    const d = positions[i];
    const h = positions[i + 1];
    const w = positions[i + 2];
    out[i] = (d - cd) * scale;
    out[i + 1] = (w - cw) * scale;
    out[i + 2] = -(h - ch) * scale;
  }
  return out;
}

/**
 * Transforms per-vertex normals from (d, h, w) axis order into scene axis
 * order, WITHOUT voxelToScene's center-subtract and uniform scale: a
 * normal is a direction, not a position, and voxelToScene's linear part is
 * an orthogonal matrix (a permutation plus one sign flip, see its
 * docstring), so the same matrix carries normals across correctly with no
 * inverse-transpose step (an orthogonal matrix's inverse transpose is
 * itself). Concretely:
 *
 *   normal_scene = (normal_d, normal_w, -normal_h)
 *
 * The negation on the h -> Z component matters on its own: dropping it
 * would leave every OTHER part of the mesh (positions, winding) correctly
 * right-handed while every normal's Z component still pointed the old,
 * mirrored way - i.e. lighting inside-out on exactly that one axis, with
 * no effect on silhouette or winding, so nothing but a shading artifact
 * would show it.
 */
export function normalToScene(normals: Float32Array): Float32Array {
  const out = new Float32Array(normals.length);
  for (let i = 0; i < normals.length; i += 3) {
    const nd = normals[i];
    const nh = normals[i + 1];
    const nw = normals[i + 2];
    out[i] = nd;
    out[i + 1] = nw;
    out[i + 2] = -nh;
  }
  return out;
}
