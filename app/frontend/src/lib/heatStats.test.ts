import { describe, expect, it } from "vitest";
import type { Plane } from "../api";
import { sliceIndexer } from "./slicing";
import { bandStats, bandSummary, entropyProfile, entropyStats, gradcamStats } from "./heatStats";

describe("entropyStats", () => {
  // 8 voxels: labels 1,1,2,3,0,0,0,0
  const mask = Uint8Array.from([1, 1, 2, 3, 0, 0, 0, 0]);
  const ent = Uint8Array.from([0, 255, 51, 255, 0, 255, 0, 51]);
  it("computes per-compartment means, outside, histogram and fracAbove", () => {
    const s = entropyStats(ent, mask);
    expect(s.necrotic).toEqual({ n: 2, mean: 0.5 });
    expect(s.oedema.n).toBe(1);
    expect(s.oedema.mean).toBeCloseTo(0.2, 9);
    expect(s.enhancing).toEqual({ n: 1, mean: 1 });
    // outside with entropy > 0: values 255 and 51
    expect(s.outside.n).toBe(2);
    expect(s.outside.mean).toBeCloseTo(0.6, 9);
    expect(s.tumourVoxels).toBe(4);
    // tumour entropies 0, 1, 0.2, 1 -> bins 0, 9, 2, 9
    expect(s.histogram).toEqual([1, 0, 1, 0, 0, 0, 0, 0, 0, 2]);
    expect(s.fracAbove(0.5)).toBe(0.5);
  });
  it("handles empty mask", () => {
    const s = entropyStats(new Uint8Array(4), new Uint8Array(4));
    expect(s.necrotic.mean).toBeNull();
    expect(s.fracAbove(0.5)).toBeNull();
  });
});

describe("bandStats", () => {
  it("counts side-neutral both / one-mask voxels", () => {
    const s = bandStats(Uint8Array.from([255, 255, 255, 255, 128, 128, 0, 0]));
    expect(s).toEqual({ bothVoxels: 4, oneMaskVoxels: 2 });
  });
  it("handles a buffer with no 255 voxels", () => {
    expect(bandStats(Uint8Array.from([128, 0]))).toEqual({ bothVoxels: 0, oneMaskVoxels: 1 });
  });
});

describe("bandSummary", () => {
  const stats = { bothVoxels: 4, oneMaskVoxels: 2 };
  it("restrictive: point estimate = both + one, set = both", () => {
    expect(bandSummary(stats, "restrictive")).toEqual({ pointEstimateVoxels: 6, conformalSetVoxels: 4 });
  });
  it("permissive: point estimate = both, set = both + one", () => {
    expect(bandSummary(stats, "permissive")).toEqual({ pointEstimateVoxels: 4, conformalSetVoxels: 6 });
  });
  it("unknown side: both null", () => {
    expect(bandSummary(stats, null)).toEqual({ pointEstimateVoxels: null, conformalSetVoxels: null });
  });
});

describe("gradcamStats", () => {
  const mask = Uint8Array.from([1, 2, 3, 0]);
  const cam = Uint8Array.from([255, 255, 255, 255]);
  it("WT region", () => {
    const s = gradcamStats(cam, mask, "WT");
    expect(s.shareInside).toBe(0.75);
    expect(s.meanInside).toBe(1);
    expect(s.meanOutside).toBe(1);
  });
  it("TC region excludes oedema", () => {
    const s = gradcamStats(Uint8Array.from([255, 255, 255, 0]), mask, "TC");
    expect(s.shareInside).toBeCloseTo(2 / 3, 9);
    expect(s.meanOutside).toBeCloseTo(0.5, 9);
  });
  it("null share for an all-zero map", () => {
    expect(gradcamStats(new Uint8Array(4), mask, "WT").shareInside).toBeNull();
  });
});

describe("entropyProfile", () => {
  const shape: [number, number, number] = [3, 4, 5];
  const [D, H, W] = shape;
  let s = 7;
  const e = new Uint8Array(D * H * W);
  for (let i = 0; i < e.length; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    e[i] = s % 256;
  }
  it("matches the slice drawer's voxels for every plane", () => {
    for (const plane of ["sagittal", "coronal", "axial"] as Plane[]) {
      const ix = sliceIndexer(plane, shape);
      const got = entropyProfile(e, shape, plane);
      expect(got).toHaveLength(ix.count);
      for (let i = 0; i < ix.count; i++) {
        let sum = 0;
        for (let r = 0; r < ix.height; r++)
          for (let c = 0; c < ix.width; c++) sum += e[ix.at(r, c, i)];
        expect(got[i]).toBeCloseTo(sum / 255 / (ix.width * ix.height), 12);
      }
    }
  });
  it("hand case: one hot voxel", () => {
    const m = new Uint8Array(D * H * W);
    m[1 * H * W + 2 * W + 3] = 255; // d=1,h=2,w=3
    expect(entropyProfile(m, shape, "sagittal")).toEqual([0, 1 / 20, 0]);
    expect(entropyProfile(m, shape, "axial")[3]).toBe(1 / 12);
  });
});
