// Builds one Surface-Nets mesh per SELECTED atlas structure (T3.4), from the
// same uint8 atlas structure-index volume the report/atlas endpoints already
// serve (see api.ts's AtlasStructureRow / getCaseAtlas): value k at a voxel
// means "this voxel belongs to structure index k" (k from
// AtlasStructureRow.index), 0 = background/no structure. Same (D,H,W)
// w-fastest layout as tumorMask (flat index d*H*W + h*W + w, see
// slicing.ts / twinGeometry.ts).
//
// Kept separate from twinMesh.worker.ts, like twinGeometry.ts and
// twinLayers.ts, so this gets its own unit tests independent of worker
// message plumbing.
//
// Why crop-then-mesh instead of one full-volume mesh pass per structure:
// a report shows up to MAX_ATLAS_STRUCTURES (16, see atlasSelection.ts)
// structures, and a full-volume pass is ~3-4M voxels regardless of how
// small the structure is. Cropping to each structure's own bounding box
// (expanded by a small margin) makes the cost proportional to the
// structure's own size instead of the whole volume.
import { meshVoxelField } from "./twinGeometry";

export interface StructureBBox {
  /** Inclusive (d,h,w) lower corner of the structure's voxels. */
  min: [number, number, number];
  /** Inclusive (d,h,w) upper corner of the structure's voxels. */
  max: [number, number, number];
  /** Voxel count for the structure - meshStructure refuses anything under 20. */
  count: number;
}

/**
 * One pass over `atlas` collecting, for each index in `selection`, the
 * (d,h,w) bounding box and voxel count of that structure's voxels.
 *
 * Uses a flat 256-entry lookup (atlas is uint8, and `selection` is capped at
 * MAX_ATLAS_STRUCTURES = 16 by atlasSelection.ts) to turn "is this voxel's
 * value one we care about, and which one" into a single typed-array read per
 * voxel, rather than a Map lookup or `.includes()` scan in the hot loop.
 *
 * @param atlas uint8 (D,H,W) w-fastest structure-index volume, 0 = background.
 * @param shape [D, H, W] dimensions of `atlas`.
 * @param selection atlas indices to collect bboxes for (order does not matter here).
 * @returns a Map keyed by structure index, present only for indices that
 *   actually occur at least once in `atlas` - an index with zero voxels
 *   (e.g. a stale report reference, or version skew between report and
 *   atlas) has no bbox to speak of and is simply absent, not a zero-sized entry.
 */
export function structureBBoxes(
  atlas: Uint8Array,
  shape: [number, number, number],
  selection: number[],
): Map<number, StructureBBox> {
  const [D, H, W] = shape;
  const nSlots = selection.length;

  // atlas value (0-255) -> slot in the per-selection accumulator arrays
  // below, or -1 if that value was not requested.
  const slotForValue = new Int16Array(256).fill(-1);
  for (let s = 0; s < nSlots; s++) {
    const idx = selection[s];
    if (idx > 0 && idx < 256) slotForValue[idx] = s;
  }

  // Float64Array (not Int32Array) so the Infinity/-Infinity sentinels below
  // survive exactly - Int32Array coerces Infinity to 0 via ToInt32, which
  // would silently break the "never seen yet" check.
  const minD = new Float64Array(nSlots).fill(Infinity);
  const minH = new Float64Array(nSlots).fill(Infinity);
  const minW = new Float64Array(nSlots).fill(Infinity);
  const maxD = new Float64Array(nSlots).fill(-Infinity);
  const maxH = new Float64Array(nSlots).fill(-Infinity);
  const maxW = new Float64Array(nSlots).fill(-Infinity);
  const count = new Uint32Array(nSlots);

  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      const rowBase = d * H * W + h * W;
      for (let w = 0; w < W; w++) {
        const value = atlas[rowBase + w];
        if (value === 0) continue;
        const slot = slotForValue[value];
        if (slot < 0) continue;
        if (d < minD[slot]) minD[slot] = d;
        if (h < minH[slot]) minH[slot] = h;
        if (w < minW[slot]) minW[slot] = w;
        if (d > maxD[slot]) maxD[slot] = d;
        if (h > maxH[slot]) maxH[slot] = h;
        if (w > maxW[slot]) maxW[slot] = w;
        count[slot]++;
      }
    }
  }

  const out = new Map<number, StructureBBox>();
  for (let s = 0; s < nSlots; s++) {
    if (count[s] === 0) continue;
    out.set(selection[s], {
      min: [minD[s], minH[s], minW[s]],
      max: [maxD[s], maxH[s], maxW[s]],
      count: count[s],
    });
  }
  return out;
}

/**
 * Surface-Nets meshes ONE atlas structure, cropped to its bounding box
 * (expanded by `margin` on every side, clamped to the volume) so the pass
 * costs roughly the structure's own voxel footprint rather than the whole
 * volume.
 *
 * Why the margin: Surface Nets only emits a vertex for a CELL (a group of 8
 * neighbouring voxels) that has both inside and outside corners (see
 * surfaceNets.ts) - it never looks past the edge of the field it is given.
 * If the crop were exactly the structure's own bbox, every boundary voxel of
 * the structure would sit exactly on the crop's edge with no "outside" cell
 * beyond it to pair with, and Surface Nets would not emit a boundary face
 * there at all, leaving the mesh open. Expanding the crop by `margin` (1
 * voxel is enough) guarantees at least one outside layer around the
 * structure, so the surface closes on every face - EXCEPT where the
 * structure itself touches the true edge of `atlas` (min/max clamped to
 * [0, D-1] etc.), where there genuinely is no further voxel to look at; that
 * face stays open, same as any other mesh in this codebase touching the
 * volume boundary (e.g. the brain shell).
 *
 * @param atlas uint8 (D,H,W) w-fastest structure-index volume.
 * @param shape [D, H, W] dimensions of `atlas`.
 * @param index the structure's atlas value to isolate within the crop.
 * @param bbox this structure's bbox/count, from `structureBBoxes`.
 * @param margin voxels to expand the bbox by on every side before cropping.
 * @returns a mesh in FULL-VOLUME (d,h,w) voxel coordinates (the crop offset
 *   is added back onto every vertex), or `null` if `bbox.count < 20` (too
 *   small to surface meaningfully - same threshold the worker already uses
 *   for tumour classes) or the cropped field turns out to have no boundary
 *   at all.
 */
export function meshStructure(
  atlas: Uint8Array,
  shape: [number, number, number],
  index: number,
  bbox: StructureBBox,
  margin = 1,
): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } | null {
  if (bbox.count < 20) return null;

  const [D, H, W] = shape;
  const [minD, minH, minW] = bbox.min;
  const [maxD, maxH, maxW] = bbox.max;

  const lo0 = Math.max(0, minD - margin);
  const lo1 = Math.max(0, minH - margin);
  const lo2 = Math.max(0, minW - margin);
  const hi0 = Math.min(D - 1, maxD + margin);
  const hi1 = Math.min(H - 1, maxH + margin);
  const hi2 = Math.min(W - 1, maxW + margin);

  const cropD = hi0 - lo0 + 1;
  const cropH = hi1 - lo1 + 1;
  const cropW = hi2 - lo2 + 1;

  // Binary field, ONE voxel per crop cell: 1 where atlas equals this
  // structure's own index, 0 everywhere else (background AND any other
  // structure that happens to share the crop) - meshVoxelField/surfaceNets
  // isolate a single structure at a time this way, same pattern the worker
  // already uses per tumour class.
  const field = new Float32Array(cropD * cropH * cropW);
  for (let d = 0; d < cropD; d++) {
    for (let h = 0; h < cropH; h++) {
      const srcRowBase = (d + lo0) * H * W + (h + lo1) * W + lo2;
      const dstRowBase = d * cropH * cropW + h * cropW;
      for (let w = 0; w < cropW; w++) {
        field[dstRowBase + w] = atlas[srcRowBase + w] === index ? 1 : 0;
      }
    }
  }

  const cropped = meshVoxelField(field, [cropD, cropH, cropW], 0.5);
  if (cropped.positions.length === 0) return null;

  // meshVoxelField returns positions in the CROP's own (d,h,w) voxel space
  // (0..cropD-1 etc). Add the crop's lower corner back so callers see
  // full-volume (d,h,w) voxel coordinates, matching the brain/tumour meshes
  // before voxelToScene/normalToScene are applied.
  const positions = new Float32Array(cropped.positions.length);
  for (let i = 0; i < cropped.positions.length; i += 3) {
    positions[i] = cropped.positions[i] + lo0;
    positions[i + 1] = cropped.positions[i + 1] + lo1;
    positions[i + 2] = cropped.positions[i + 2] + lo2;
  }

  return { positions, normals: cropped.normals, indices: cropped.indices };
}
