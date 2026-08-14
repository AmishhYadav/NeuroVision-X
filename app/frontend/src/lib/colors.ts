// Colour constants and small colour-math helpers shared by the render
// pipeline and the legend. Values are the project's fixed paper-figure
// palette - do not substitute or "improve" these.

export type RGB = [number, number, number];

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Segmentation class colours (classes 1-3). Index 0 (background) is unused.
export const CLASS_COLORS: Record<number, RGB> = {
  1: hexToRgb("#56B4E9"), // necrotic core
  2: hexToRgb("#009E73"), // oedema
  3: hexToRgb("#D55E00"), // enhancing tumour
};

export const CLASS_LABELS: Record<number, string> = {
  1: "Necrotic core",
  2: "Oedema",
  3: "Enhancing tumour",
};

export const DISAGREEMENT_COLORS = {
  falseNegative: hexToRgb("#F0E442"), // truth has tumour, prediction does not
  falsePositive: hexToRgb("#CC79A7"), // prediction has tumour, truth does not
};

export const WT_COLOR = hexToRgb("#0072B2");
export const ENTROPY_LINE_COLOR = hexToRgb("#E4693E");
export const ERROR_LANE_COLOR = hexToRgb("#F0E442");

// 3-stop magma-like ramp for the uncertainty (predictive entropy) overlay.
const ENTROPY_STOPS: RGB[] = [
  hexToRgb("#3B0F70"),
  hexToRgb("#E4693E"),
  hexToRgb("#FCFDBF"),
];

/**
 * Entropy at which the overlay reaches full opacity.
 *
 * The COLOUR of a voxel always encodes its true entropy on a fixed [0, 1]
 * scale, so the same colour means the same value in every case - per-case
 * rescaling would make a barely-uncertain case look dramatic and destroy any
 * comparison between two cases. Only the ALPHA is given a gain, because
 * without one the layer is invisible: measured on the test split, entropy is
 * concentrated in a thin boundary shell and only ~0.4% of voxels on a slice
 * exceed 0.3, so a straight `alpha = e` peaks near 0.2 exactly where the
 * model is genuinely uncertain.
 *
 * 0.35 rather than 1.0 because of how the field is normalized:
 * `entropy_from_logits` divides the summed per-channel Bernoulli entropy by
 * `3 * ln 2`, so a voxel where ONE of the three region channels is maximally
 * uncertain reads ~0.33, not 1.0. Values above that need two channels
 * uncertain at once and are rare.
 */
export const ENTROPY_ALPHA_FULL = 0.35;

/**
 * Entropy reached when exactly one of the three region channels is maximally
 * uncertain (1/3). Marked on the legend's colour bar so a reader can tell the
 * reachable part of the scale from the part that needs several channels to
 * disagree at once - the same reason gate maps mark 0.5.
 */
export const ENTROPY_ONE_CHANNEL = 1 / 3;

/** Linear interpolation across the 3-stop entropy ramp. `t` in [0, 1]. */
export function entropyColor(t: number): RGB {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (ENTROPY_STOPS.length - 1);
  const idx = Math.min(ENTROPY_STOPS.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const a = ENTROPY_STOPS[idx];
  const b = ENTROPY_STOPS[idx + 1];
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ];
}
