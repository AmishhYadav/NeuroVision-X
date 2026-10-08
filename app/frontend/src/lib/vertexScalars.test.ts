import { describe, expect, it } from "vitest";
import { sampleScalarsAtVertices } from "./vertexScalars";

// Builds a flat (D,H,W) w-fastest uint8 buffer (index d*H*W + h*W + w,
// matching every volume buffer in this codebase - see slicing.ts) whose
// value depends only on d, so sampling along d is easy to reason about.
function rampAlongD(shape: [number, number, number], perD: number): Uint8Array {
  const [D, H, W] = shape;
  const volume = new Uint8Array(D * H * W);
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        volume[d * H * W + h * W + w] = d * perD;
      }
    }
  }
  return volume;
}

describe("sampleScalarsAtVertices", () => {
  it("reads a constant field as that constant/255 everywhere", () => {
    const shape: [number, number, number] = [6, 6, 6];
    const volume = new Uint8Array(6 * 6 * 6).fill(200);
    // Three arbitrary vertices with distinct outward normals.
    const positions = new Float32Array([3, 3, 3, 1, 4, 2, 5, 0, 5]);
    const normals = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

    const scalars = sampleScalarsAtVertices(volume, shape, positions, normals);

    expect(scalars.length).toBe(3);
    for (const value of scalars) {
      expect(value).toBeCloseTo(200 / 255, 6);
    }
  });

  it("samples one voxel INWARD along the normal by default (insetVoxels=1)", () => {
    const shape: [number, number, number] = [10, 3, 3];
    const volume = rampAlongD(shape, 10);
    // Vertex at d=5, outward normal pointing +d - inward is -d, so the
    // default inset of 1 must read d=4's value, not d=5's.
    const positions = new Float32Array([5, 1, 1]);
    const normals = new Float32Array([1, 0, 0]);

    const scalars = sampleScalarsAtVertices(volume, shape, positions, normals);

    expect(scalars[0]).toBeCloseTo((4 * 10) / 255, 6);
  });

  it("samples exactly at the vertex when insetVoxels=0", () => {
    const shape: [number, number, number] = [10, 3, 3];
    const volume = rampAlongD(shape, 10);
    const positions = new Float32Array([5, 1, 1]);
    const normals = new Float32Array([1, 0, 0]);

    const scalars = sampleScalarsAtVertices(volume, shape, positions, normals, 0);

    expect(scalars[0]).toBeCloseTo((5 * 10) / 255, 6);
  });

  it("clamps an inward sample that falls outside the volume, with no NaN and no throw", () => {
    const shape: [number, number, number] = [10, 3, 3];
    const volume = rampAlongD(shape, 10);
    // Vertex at d=0 with outward normal +d: inward (position - 1*normal)
    // lands at d=-1, which must clamp to d=0 rather than reading garbage or
    // producing NaN.
    const positions = new Float32Array([0, 1, 1]);
    const normals = new Float32Array([1, 0, 0]);

    let scalars: Float32Array = new Float32Array();
    expect(() => {
      scalars = sampleScalarsAtVertices(volume, shape, positions, normals);
    }).not.toThrow();

    expect(scalars[0]).toBeCloseTo(0, 6);
    expect(Number.isNaN(scalars[0])).toBe(false);
  });

  it("returns an array of length N for N vertices", () => {
    const shape: [number, number, number] = [4, 4, 4];
    const volume = new Uint8Array(4 * 4 * 4).fill(1);
    const nVerts = 5;
    const positions = new Float32Array(nVerts * 3).fill(1);
    const normals = new Float32Array(nVerts * 3);
    for (let i = 0; i < nVerts; i++) normals[i * 3] = 1; // +d normals

    const scalars = sampleScalarsAtVertices(volume, shape, positions, normals);

    expect(scalars.length).toBe(nVerts);
  });
});
