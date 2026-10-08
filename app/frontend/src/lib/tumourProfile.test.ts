import { describe, expect, it } from "vitest";
import type { Plane } from "../api";
import { sliceIndexer } from "./slicing";
import { tumourProfile } from "./tumourProfile";

const SHAPE: [number, number, number] = [3, 4, 5]; // D, H, W
const [D, H, W] = SHAPE;
const flat = (d: number, h: number, w: number) => d * H * W + h * W + w;

function handMask(): Uint8Array {
  const m = new Uint8Array(D * H * W);
  // Known voxels (d,h,w): (0,0,0) (0,1,0) (0,1,3) (2,3,4) (2,3,3) (1,2,3)
  for (const [d, h, w] of [
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 3],
    [2, 3, 4],
    [2, 3, 3],
    [1, 2, 3],
  ]) {
    m[flat(d, h, w)] = 2; // any value > 0 counts
  }
  return m;
}

describe("tumourProfile", () => {
  it("gives exact hand-counted fractions for all three planes", () => {
    const m = handMask();
    // Sagittal (along D, 20 voxels/slice): d0=3, d1=1, d2=2
    expect(tumourProfile(m, SHAPE, "sagittal")).toEqual([3 / 20, 1 / 20, 2 / 20]);
    // Coronal (along H, 15 voxels/slice): h0=1, h1=2, h2=1, h3=2
    expect(tumourProfile(m, SHAPE, "coronal")).toEqual([1 / 15, 2 / 15, 1 / 15, 2 / 15]);
    // Axial (along W, 12 voxels/slice): w0=2, w1=0, w2=0, w3=3, w4=1
    expect(tumourProfile(m, SHAPE, "axial")).toEqual([2 / 12, 0, 0, 3 / 12, 1 / 12]);
  });

  it("returns one entry per slice, matching sliceIndexer.count", () => {
    for (const plane of ["sagittal", "coronal", "axial"] as Plane[]) {
      expect(tumourProfile(handMask(), SHAPE, plane)).toHaveLength(sliceIndexer(plane, SHAPE).count);
    }
  });

  it("is all zeros for an empty mask", () => {
    const empty = new Uint8Array(D * H * W);
    for (const plane of ["sagittal", "coronal", "axial"] as Plane[]) {
      expect(tumourProfile(empty, SHAPE, plane).every((v) => v === 0)).toBe(true);
    }
  });

  it("agrees with summing the voxels the slice drawer reads", () => {
    // Deterministic pseudo-random mask (LCG) so the test is reproducible.
    let s = 12345;
    const m = new Uint8Array(D * H * W);
    for (let i = 0; i < m.length; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      m[i] = s % 3 === 0 ? 1 : 0;
    }
    for (const plane of ["sagittal", "coronal", "axial"] as Plane[]) {
      const ix = sliceIndexer(plane, SHAPE);
      const got = tumourProfile(m, SHAPE, plane);
      for (let i = 0; i < ix.count; i++) {
        let n = 0;
        for (let r = 0; r < ix.height; r++) {
          for (let c = 0; c < ix.width; c++) if (m[ix.at(r, c, i)] > 0) n++;
        }
        expect(got[i]).toBeCloseTo(n / (ix.width * ix.height), 12);
      }
    }
  });
});
