import { describe, expect, it } from "vitest";
import { structureBBoxes, meshStructure } from "./structureMesh";

const SHAPE: [number, number, number] = [16, 16, 16];

// Fills a w-fastest (D,H,W) atlas volume (flat index d*H*W + h*W + w, same
// layout as tumorMask/slicing.ts) with an axis-aligned cube of the given
// atlas index, voxels [d0,d1] x [h0,h1] x [w0,w1] inclusive.
function fillCube(
  atlas: Uint8Array,
  shape: [number, number, number],
  index: number,
  d0: number,
  d1: number,
  h0: number,
  h1: number,
  w0: number,
  w1: number,
) {
  const [, H, W] = shape;
  for (let d = d0; d <= d1; d++) {
    for (let h = h0; h <= h1; h++) {
      for (let w = w0; w <= w1; w++) {
        atlas[d * H * W + h * W + w] = index;
      }
    }
  }
}

function buildAtlas(): Uint8Array {
  const [D, H, W] = SHAPE;
  const atlas = new Uint8Array(D * H * W);
  // Interior cube, index 5: 4x4x4 = 64 voxels, well clear of every face.
  fillCube(atlas, SHAPE, 5, 4, 7, 4, 7, 4, 7);
  // Cube touching the volume's d=0 face, index 9: 4x4x4 = 64 voxels.
  fillCube(atlas, SHAPE, 9, 0, 3, 10, 13, 10, 13);
  // A tiny cluster, index 7: 5 voxels - under the count>=20 threshold.
  fillCube(atlas, SHAPE, 7, 12, 12, 0, 0, 0, 0);
  atlas[12 * H * W + 0 * W + 1] = 7;
  atlas[12 * H * W + 0 * W + 2] = 7;
  atlas[12 * H * W + 0 * W + 3] = 7;
  atlas[12 * H * W + 1 * W + 0] = 7;
  // A structure that is never in `selection` below, so it must never show
  // up in bboxes or get meshed - index 3.
  fillCube(atlas, SHAPE, 3, 8, 9, 8, 9, 8, 9);
  return atlas;
}

describe("structureBBoxes", () => {
  it("computes exact bboxes and counts for selected indices", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [5, 9, 7]);

    expect(bboxes.get(5)).toEqual({ min: [4, 4, 4], max: [7, 7, 7], count: 64 });
    expect(bboxes.get(9)).toEqual({ min: [0, 10, 10], max: [3, 13, 13], count: 64 });
    expect(bboxes.get(7)).toEqual({ min: [12, 0, 0], max: [12, 1, 3], count: 5 });
  });

  it("omits an index that was never selected, even though it occurs in the atlas", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [5, 9]);

    expect(bboxes.has(3)).toBe(false);
    expect(bboxes.size).toBe(2);
  });

  it("omits a selected index that never occurs in the atlas", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [5, 42]);

    expect(bboxes.has(42)).toBe(false);
    expect(bboxes.get(5)).toBeDefined();
  });
});

describe("meshStructure", () => {
  it("meshes an interior structure with all vertices inside the margin-expanded bbox, in full-volume coords", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [5, 9]);
    const bbox = bboxes.get(5)!;

    const mesh = meshStructure(atlas, SHAPE, 5, bbox);
    expect(mesh).not.toBeNull();
    expect(mesh!.positions.length).toBeGreaterThan(0);
    expect(mesh!.normals.length).toBe(mesh!.positions.length);
    expect(mesh!.indices.length).toBeGreaterThan(0);

    // Every vertex must land inside [bbox.min - 1, bbox.max + 1] on every
    // axis (margin = 1 by default) - i.e. the crop offset was correctly
    // added back, giving FULL-VOLUME coordinates rather than 0-based crop
    // coordinates (a cube at d 4..7 must yield d in ~3..8, not ~0..4).
    for (let i = 0; i < mesh!.positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const p = mesh!.positions[i + axis];
        expect(p).toBeGreaterThanOrEqual(bbox.min[axis] - 1 - 1e-3);
        expect(p).toBeLessThanOrEqual(bbox.max[axis] + 1 + 1e-3);
      }
    }
  });

  it("meshes a structure touching the volume edge without erroring, still bounded by the (clamped) margin", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [5, 9]);
    const bbox = bboxes.get(9)!; // min d = 0, touches the true volume edge

    const mesh = meshStructure(atlas, SHAPE, 9, bbox);
    expect(mesh).not.toBeNull();
    expect(mesh!.positions.length).toBeGreaterThan(0);

    for (let i = 0; i < mesh!.positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const p = mesh!.positions[i + axis];
        // Clamped margin: never below 0 on the d axis (no voxel exists
        // there), never below bbox.min - 1 / above bbox.max + 1 elsewhere.
        expect(p).toBeGreaterThanOrEqual(Math.max(0, bbox.min[axis] - 1) - 1e-3);
        expect(p).toBeLessThanOrEqual(bbox.max[axis] + 1 + 1e-3);
      }
    }
  });

  it("returns null when the structure has fewer than 20 voxels", () => {
    const atlas = buildAtlas();
    const bboxes = structureBBoxes(atlas, SHAPE, [7]);
    const bbox = bboxes.get(7)!;
    expect(bbox.count).toBe(5);

    const mesh = meshStructure(atlas, SHAPE, 7, bbox);
    expect(mesh).toBeNull();
  });

  it("returns null for an all-zero field even when count happens to satisfy the threshold", () => {
    const atlas = buildAtlas();
    // A bbox that claims 64 voxels but points at the wrong index (3, which
    // was never meshed as 5) must never crash - meshVoxelField just returns
    // an empty mesh (isolating `index` finds nothing) -> null.
    const fakeBbox = { min: [8, 8, 8] as [number, number, number], max: [9, 9, 9] as [number, number, number], count: 64 };
    const mesh = meshStructure(atlas, SHAPE, 99, fakeBbox);
    expect(mesh).toBeNull();
  });
});
