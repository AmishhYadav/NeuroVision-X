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
