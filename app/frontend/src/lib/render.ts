// Pure slice-compositing into an ImageData. No React, no DOM beyond the
// ImageData constructor itself, so this is unit-testable and reusable by
// every viewport without duplicating the layering logic.
//
// Layer order (bottom to top):
//   1. greyscale modality
//   2. class/disagreement overlay
//   3. ground-truth whole-tumour outline (prediction mode only)
//   4. predictive-entropy heat blend

import type { Plane } from "../api";
import { sliceIndexer } from "./slicing";
import { CLASS_COLORS, DISAGREEMENT_COLORS, ENTROPY_ALPHA_FULL, entropyColor } from "./colors";

export type OverlayMode = "prediction" | "truth" | "disagreement";

export interface RenderParams {
  plane: Plane;
  shape: [number, number, number];
  sliceIndex: number;
  /** (D,H,W) greyscale modality buffer, 0-255. Required. */
  image: Uint8Array;
  /** (D,H,W) class buffer {0,1,2,3}, or null if unavailable. */
  predictionMask: Uint8Array | null;
  /** (D,H,W) class buffer {0,1,2,3}, or null if no label for this case. */
  labelMask: Uint8Array | null;
  /** (D,H,W) uint8 buffer, value/255 = entropy in [0,1], or null if no logits. */
  uncertainty: Uint8Array | null;
  overlayMode: OverlayMode;
  overlayOpacity: number;
  showTruthOutline: boolean;
  showUncertainty: boolean;
  uncertaintyOpacity: number;
}

/** Render one slice into a freshly allocated ImageData at the slice's native resolution. */
export function renderSlice(params: RenderParams): ImageData {
  const {
    plane,
    shape,
    sliceIndex,
    image,
    predictionMask,
    labelMask,
    uncertainty,
    overlayMode,
    overlayOpacity,
    showTruthOutline,
    showUncertainty,
    uncertaintyOpacity,
  } = params;

  const indexer = sliceIndexer(plane, shape);
  const { width, height } = indexer;
  const out = new ImageData(width, height);
  const pixels = out.data;

  const overlayMask =
    overlayMode === "truth" ? labelMask : overlayMode === "prediction" ? predictionMask : null;
  const disagreementActive = overlayMode === "disagreement" && predictionMask && labelMask;

  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const srcIdx = indexer.at(row, col, sliceIndex);
      const v = image[srcIdx] ?? 0;
      let r = v;
      let g = v;
      let b = v;

      if (overlayMask) {
        const cls = overlayMask[srcIdx];
        const color = cls ? CLASS_COLORS[cls] : undefined;
        if (color) {
          r = v * (1 - overlayOpacity) + color[0] * overlayOpacity;
          g = v * (1 - overlayOpacity) + color[1] * overlayOpacity;
          b = v * (1 - overlayOpacity) + color[2] * overlayOpacity;
        }
      } else if (disagreementActive) {
        const predTumor = predictionMask![srcIdx] > 0;
        const truthTumor = labelMask![srcIdx] > 0;
        let color: readonly [number, number, number] | undefined;
        if (truthTumor && !predTumor) color = DISAGREEMENT_COLORS.falseNegative;
        else if (predTumor && !truthTumor) color = DISAGREEMENT_COLORS.falsePositive;
        if (color) {
          r = v * (1 - overlayOpacity) + color[0] * overlayOpacity;
          g = v * (1 - overlayOpacity) + color[1] * overlayOpacity;
          b = v * (1 - overlayOpacity) + color[2] * overlayOpacity;
        }
      }

      if (showUncertainty && uncertainty) {
        const e = uncertainty[srcIdx] / 255;
        if (e > 0) {
          // Colour carries the true entropy on a fixed [0, 1] scale; alpha
          // ramps to full by ENTROPY_ALPHA_FULL so the thin high-entropy
          // shell is actually visible. See the constant for the measurements.
          const alpha = Math.min(1, e / ENTROPY_ALPHA_FULL) * uncertaintyOpacity;
          const [er, eg, eb] = entropyColor(e);
          r = r * (1 - alpha) + er * alpha;
          g = g * (1 - alpha) + eg * alpha;
          b = b * (1 - alpha) + eb * alpha;
        }
      }

      const pixelOffset = (row * width + col) * 4;
      pixels[pixelOffset] = r;
      pixels[pixelOffset + 1] = g;
      pixels[pixelOffset + 2] = b;
      pixels[pixelOffset + 3] = 255;
    }
  }

  if (showTruthOutline && labelMask && overlayMode === "prediction") {
    drawTruthOutline(pixels, indexer, labelMask, sliceIndex, width, height);
  }

  return out;
}

function drawTruthOutline(
  pixels: Uint8ClampedArray,
  indexer: ReturnType<typeof sliceIndexer>,
  labelMask: Uint8Array,
  sliceIndex: number,
  width: number,
  height: number,
): void {
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const idx = indexer.at(row, col, sliceIndex);
      if (labelMask[idx] === 0) continue;

      const up = row > 0 ? labelMask[indexer.at(row - 1, col, sliceIndex)] : 0;
      const down = row < height - 1 ? labelMask[indexer.at(row + 1, col, sliceIndex)] : 0;
      const left = col > 0 ? labelMask[indexer.at(row, col - 1, sliceIndex)] : 0;
      const right = col < width - 1 ? labelMask[indexer.at(row, col + 1, sliceIndex)] : 0;

      if (up === 0 || down === 0 || left === 0 || right === 0) {
        const pixelOffset = (row * width + col) * 4;
        pixels[pixelOffset] = 255;
        pixels[pixelOffset + 1] = 255;
        pixels[pixelOffset + 2] = 255;
        pixels[pixelOffset + 3] = 255;
      }
    }
  }
}
