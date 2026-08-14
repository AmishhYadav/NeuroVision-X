// These tests pin the DISPLAY CONVENTION documented in slicing.ts's module
// comment, not merely "whatever sliceIndexer currently computes". A wrong
// axis permute or a dropped row-reversal still produces a full, bijective,
// plausible-looking slice through the volume - it is just the wrong picture
// (head on its side, or superior/inferior flipped). Content assertions below
// therefore use HAND-DERIVED literal expected grids, computed once by hand
// from the volume's fill rule, rather than re-deriving them from the same
// formula sliceIndexer uses - a copy-pasted formula bug would pass silently
// against a "test" that is really just the implementation restated.

import { describe, expect, it } from "vitest";
import { sliceIndexer } from "./slicing";
import type { Plane } from "../api";

// Deliberately asymmetric shape: D=2, H=3, W=4. If width/height were ever
// swapped between two axes, or two axes' extents were confused, a symmetric
// shape (e.g. a cube) would not catch it - the wrong mapping would still
// produce a same-sized image full of *some* valid voxel. Asymmetric dims
// mean a mismatched axis produces an out-of-range index or a wrong shape.
const D = 2;
const H = 3;
const W = 4;
const SHAPE: [number, number, number] = [D, H, W];

// value(d, h, w) = d*100 + h*10 + w - every voxel is uniquely identifiable
// by inspection of its value, so a wrong flat-index computation shows up
// immediately as "value implies a different (d,h,w) than intended".
function value(d: number, h: number, w: number): number {
  return d * 100 + h * 10 + w;
}

function buildVolume(): Uint32Array {
  const vol = new Uint32Array(D * H * W);
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        vol[d * H * W + h * W + w] = value(d, h, w);
      }
    }
  }
  return vol;
}

const volume = buildVolume();

// Hand-derived (d,h,w) lookup, independent of the flat-index arithmetic
// under test - used only to translate a hand-worked-out (d,h,w) triple into
// its expected value, never to compute row/col/i -> flat-index.
function vol(d: number, h: number, w: number): number {
  return value(d, h, w);
}

describe("sliceIndexer - plane dimensions", () => {
  it("axial: width=D, height=H, count=W", () => {
    const idx = sliceIndexer("axial", SHAPE);
    expect(idx.width).toBe(D);
    expect(idx.height).toBe(H);
    expect(idx.count).toBe(W);
  });

  it("coronal: width=D, height=W, count=H", () => {
    const idx = sliceIndexer("coronal", SHAPE);
    expect(idx.width).toBe(D);
    expect(idx.height).toBe(W);
    expect(idx.count).toBe(H);
  });

  it("sagittal: width=H, height=W, count=D", () => {
    const idx = sliceIndexer("sagittal", SHAPE);
    expect(idx.width).toBe(H);
    expect(idx.height).toBe(W);
    expect(idx.count).toBe(D);
  });
});

describe("sliceIndexer - axial content (rows=h, anterior at top; cols=d)", () => {
  const idx = sliceIndexer("axial", SHAPE);

  // Hand-derived: axial slice i is w=i. image[row=h][col=d] = vol(d, h, i).
  const expected: number[][][] = [
    // i = 0 (w=0)
    [
      [vol(0, 0, 0), vol(1, 0, 0)],
      [vol(0, 1, 0), vol(1, 1, 0)],
      [vol(0, 2, 0), vol(1, 2, 0)],
    ],
    // i = 1 (w=1)
    [
      [vol(0, 0, 1), vol(1, 0, 1)],
      [vol(0, 1, 1), vol(1, 1, 1)],
      [vol(0, 2, 1), vol(1, 2, 1)],
    ],
    // i = 2 (w=2)
    [
      [vol(0, 0, 2), vol(1, 0, 2)],
      [vol(0, 1, 2), vol(1, 1, 2)],
      [vol(0, 2, 2), vol(1, 2, 2)],
    ],
    // i = 3 (w=3)
    [
      [vol(0, 0, 3), vol(1, 0, 3)],
      [vol(0, 1, 3), vol(1, 1, 3)],
      [vol(0, 2, 3), vol(1, 2, 3)],
    ],
  ];

  it.each([0, 1, 2, 3])("slice i=%i matches the hand-derived grid", (i) => {
    for (let row = 0; row < idx.height; row++) {
      for (let col = 0; col < idx.width; col++) {
        expect(volume[idx.at(row, col, i)]).toBe(expected[i][row][col]);
      }
    }
  });
});

describe("sliceIndexer - coronal content (rows=reversed w, superior at top; cols=d)", () => {
  const idx = sliceIndexer("coronal", SHAPE);

  // Hand-derived: coronal slice i is h=i. image[row][col] = vol(d=col, h=i, w=W-1-row).
  const expected: number[][][] = [
    // i = 0 (h=0)
    [
      [vol(0, 0, 3), vol(1, 0, 3)], // row 0 -> w=3 (superior)
      [vol(0, 0, 2), vol(1, 0, 2)], // row 1 -> w=2
      [vol(0, 0, 1), vol(1, 0, 1)], // row 2 -> w=1
      [vol(0, 0, 0), vol(1, 0, 0)], // row 3 -> w=0 (inferior)
    ],
    // i = 1 (h=1)
    [
      [vol(0, 1, 3), vol(1, 1, 3)],
      [vol(0, 1, 2), vol(1, 1, 2)],
      [vol(0, 1, 1), vol(1, 1, 1)],
      [vol(0, 1, 0), vol(1, 1, 0)],
    ],
    // i = 2 (h=2)
    [
      [vol(0, 2, 3), vol(1, 2, 3)],
      [vol(0, 2, 2), vol(1, 2, 2)],
      [vol(0, 2, 1), vol(1, 2, 1)],
      [vol(0, 2, 0), vol(1, 2, 0)],
    ],
  ];

  it.each([0, 1, 2])("slice i=%i matches the hand-derived grid", (i) => {
    for (let row = 0; row < idx.height; row++) {
      for (let col = 0; col < idx.width; col++) {
        expect(volume[idx.at(row, col, i)]).toBe(expected[i][row][col]);
      }
    }
  });
});

describe("sliceIndexer - sagittal content (rows=reversed w, superior at top; cols=h, anterior at left)", () => {
  const idx = sliceIndexer("sagittal", SHAPE);

  // Hand-derived: sagittal slice i is d=i. image[row][col] = vol(d=i, h=col, w=W-1-row).
  const expected: number[][][] = [
    // i = 0 (d=0)
    [
      [vol(0, 0, 3), vol(0, 1, 3), vol(0, 2, 3)], // row 0 -> w=3 (superior)
      [vol(0, 0, 2), vol(0, 1, 2), vol(0, 2, 2)],
      [vol(0, 0, 1), vol(0, 1, 1), vol(0, 2, 1)],
      [vol(0, 0, 0), vol(0, 1, 0), vol(0, 2, 0)], // row 3 -> w=0 (inferior)
    ],
    // i = 1 (d=1)
    [
      [vol(1, 0, 3), vol(1, 1, 3), vol(1, 2, 3)],
      [vol(1, 0, 2), vol(1, 1, 2), vol(1, 2, 2)],
      [vol(1, 0, 1), vol(1, 1, 1), vol(1, 2, 1)],
      [vol(1, 0, 0), vol(1, 1, 0), vol(1, 2, 0)],
    ],
  ];

  it.each([0, 1])("slice i=%i matches the hand-derived grid", (i) => {
    for (let row = 0; row < idx.height; row++) {
      for (let col = 0; col < idx.width; col++) {
        expect(volume[idx.at(row, col, i)]).toBe(expected[i][row][col]);
      }
    }
  });
});

describe("sliceIndexer - superior is up (coronal, sagittal)", () => {
  // This is the assertion that fails if someone "simplifies away" the row
  // reversal in coronal/sagittal at(): without it, row 0 would resolve to
  // w=0 (inferior) and the head would render upside down while every other
  // test in this file (bijection, coverage, even most content checks done
  // sloppily) could still pass.
  const planes: Plane[] = ["coronal", "sagittal"];

  it.each(planes)("%s: row 0 is w=W-1 (superior), last row is w=0 (inferior)", (plane) => {
    const idx = sliceIndexer(plane, SHAPE);
    const i = 0;
    for (let col = 0; col < idx.width; col++) {
      const topIdx = idx.at(0, col, i);
      const bottomIdx = idx.at(idx.height - 1, col, i);
      const topW = topIdx % W;
      const bottomWFull = bottomIdx % W;
      expect(topW).toBe(W - 1);
      expect(bottomWFull).toBe(0);
    }
  });
});

describe("sliceIndexer - bijection per slice", () => {
  const planes: Plane[] = ["axial", "coronal", "sagittal"];

  it.each(planes)("%s: every (row,col) maps to a distinct, in-range flat index", (plane) => {
    const idx = sliceIndexer(plane, SHAPE);
    for (let i = 0; i < idx.count; i++) {
      const seen = new Set<number>();
      for (let row = 0; row < idx.height; row++) {
        for (let col = 0; col < idx.width; col++) {
          const flat = idx.at(row, col, i);
          expect(flat).toBeGreaterThanOrEqual(0);
          expect(flat).toBeLessThan(D * H * W);
          expect(seen.has(flat)).toBe(false);
          seen.add(flat);
        }
      }
      expect(seen.size).toBe(idx.width * idx.height);
    }
  });
});

describe("sliceIndexer - full slice coverage", () => {
  const planes: Plane[] = ["axial", "coronal", "sagittal"];

  it.each(planes)("%s: iterating every slice visits every voxel exactly once", (plane) => {
    const idx = sliceIndexer(plane, SHAPE);
    const seen = new Set<number>();
    for (let i = 0; i < idx.count; i++) {
      for (let row = 0; row < idx.height; row++) {
        for (let col = 0; col < idx.width; col++) {
          const flat = idx.at(row, col, i);
          expect(seen.has(flat)).toBe(false);
          seen.add(flat);
        }
      }
    }
    expect(seen.size).toBe(D * H * W);
  });
});
