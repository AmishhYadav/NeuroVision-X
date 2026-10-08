// Summary statistics over the three heat buffers (entropy, conformal band,
// Grad-CAM), computed client-side from the served uint8 volumes. All buffers
// share the predicted mask's flat (D, H, W) row-major layout. Each function
// makes one pass and allocates nothing per voxel.
//
// Mask labels: 1 = necrotic core, 2 = oedema, 3 = enhancing, 0 = background.

import type { Plane } from "../api";

export interface CompartmentStat {
  /** Voxels in the compartment. */
  n: number;
  /** Mean normalised entropy in 0..1, or null when n === 0. */
  mean: number | null;
}

export interface EntropyStats {
  necrotic: CompartmentStat; // label 1
  oedema: CompartmentStat; // label 2
  enhancing: CompartmentStat; // label 3
  /** Voxels with mask === 0 AND entropy > 0. */
  outside: CompartmentStat;
  /** 10 bins over [0, 1] counting voxels with mask > 0. */
  histogram: number[];
  /** Voxels with mask > 0. */
  tumourVoxels: number;
  /** Fraction (0..1) of tumour voxels with entropy > threshold; null if no tumour. */
  fracAbove: (threshold: number) => number | null;
}

const mean255 = (sum: number, n: number): number | null => (n > 0 ? sum / 255 / n : null);

/** Entropy buffer values are normalised entropy x 255 (0 = certain, 1 = maximal). */
export function entropyStats(entropy: Uint8Array, mask: Uint8Array): EntropyStats {
  const n = Math.min(entropy.length, mask.length);
  const cnt = [0, 0, 0, 0]; // label 1, 2, 3, outside
  const sum = [0, 0, 0, 0];
  const histogram = new Array<number>(10).fill(0);
  let tumourVoxels = 0;
  // Per-byte count of tumour voxels, so fracAbove(t) is exact for any t
  // without keeping the volume around.
  const byteCounts = new Uint32Array(256);

  for (let i = 0; i < n; i++) {
    const m = mask[i];
    const e = entropy[i];
    if (m > 0) {
      tumourVoxels++;
      byteCounts[e]++;
      const bin = Math.min(9, Math.floor((e / 255) * 10));
      histogram[bin]++;
      if (m <= 3) {
        cnt[m - 1]++;
        sum[m - 1] += e;
      }
    } else if (e > 0) {
      cnt[3]++;
      sum[3] += e;
    }
  }

  return {
    necrotic: { n: cnt[0], mean: mean255(sum[0], cnt[0]) },
    oedema: { n: cnt[1], mean: mean255(sum[1], cnt[1]) },
    enhancing: { n: cnt[2], mean: mean255(sum[2], cnt[2]) },
    outside: { n: cnt[3], mean: mean255(sum[3], cnt[3]) },
    histogram,
    tumourVoxels,
    fracAbove: (threshold: number) => {
      if (tumourVoxels === 0) return null;
      let above = 0;
      for (let b = 0; b < 256; b++) if (b / 255 > threshold) above += byteCounts[b];
      return above / tumourVoxels;
    },
  };
}

export interface BandStats {
  /** Voxels in both the point-estimate mask and the conformal set (byte 255). */
  bothVoxels: number;
  /** Voxels in exactly one of the two masks (byte 128); which one depends on the band side. */
  oneMaskVoxels: number;
}

/** Side-neutral counts: 255 = in both masks, 128 = in exactly one, 0 = neither. */
export function bandStats(band: Uint8Array): BandStats {
  let bothVoxels = 0;
  let oneMaskVoxels = 0;
  for (let i = 0; i < band.length; i++) {
    const v = band[i];
    if (v === 255) bothVoxels++;
    else if (v === 128) oneMaskVoxels++;
  }
  return { bothVoxels, oneMaskVoxels };
}

export interface BandSummary {
  pointEstimateVoxels: number | null;
  conformalSetVoxels: number | null;
}

/**
 * Resolve the neutral counts into point-estimate and conformal-set sizes.
 * restrictive: 128 = point estimate only, so point = both + one, set = both.
 * permissive: 128 = conformal set only, so point = both, set = both + one.
 * Unknown side: cannot tell, so both are null.
 */
export function bandSummary(stats: BandStats, side: "permissive" | "restrictive" | null): BandSummary {
  const total = stats.bothVoxels + stats.oneMaskVoxels;
  if (side === "restrictive") return { pointEstimateVoxels: total, conformalSetVoxels: stats.bothVoxels };
  if (side === "permissive") return { pointEstimateVoxels: stats.bothVoxels, conformalSetVoxels: total };
  return { pointEstimateVoxels: null, conformalSetVoxels: null };
}

export interface GradcamStats {
  /** Share (0..1) of total Grad-CAM mass inside the region; null when the total is 0. */
  shareInside: number | null;
  meanInside: number | null;
  meanOutside: number | null;
}

/** Region WT = mask > 0; TC = mask in {1, 3}. */
export function gradcamStats(cam: Uint8Array, mask: Uint8Array, region: "WT" | "TC"): GradcamStats {
  const n = Math.min(cam.length, mask.length);
  let sumIn = 0;
  let sumOut = 0;
  let nIn = 0;
  let nOut = 0;
  for (let i = 0; i < n; i++) {
    const m = mask[i];
    const inside = region === "WT" ? m > 0 : m === 1 || m === 3;
    if (inside) {
      sumIn += cam[i];
      nIn++;
    } else {
      sumOut += cam[i];
      nOut++;
    }
  }
  const total = sumIn + sumOut;
  return {
    shareInside: total > 0 ? sumIn / total : null,
    meanInside: mean255(sumIn, nIn),
    meanOutside: mean255(sumOut, nOut),
  };
}

/**
 * Mean normalised entropy (0..1) per slice along `plane`. Axes match
 * tumourProfile / sliceIndexer: axial -> W, coronal -> H, sagittal -> D.
 */
export function entropyProfile(
  entropy: Uint8Array,
  shape: [number, number, number],
  plane: Plane,
): number[] {
  const [D, H, W] = shape;
  const count = plane === "sagittal" ? D : plane === "coronal" ? H : W;
  const perSlice = (D * H * W) / count;
  const sums = new Float64Array(count);
  let idx = 0;
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++, idx++) {
        sums[plane === "sagittal" ? d : plane === "coronal" ? h : w] += entropy[idx];
      }
    }
  }
  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) out[i] = perSlice > 0 ? sums[i] / 255 / perSlice : 0;
  return out;
}
