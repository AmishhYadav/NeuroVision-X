// Turns per-vertex scalars (from vertexScalars.ts) into per-vertex RGB for
// three.js's `color` vertex attribute, using the exact same colour mapping
// as the 2D slice view (render.ts) and its legend (Legend.tsx), so the twin
// and the 2D view can never show two different colours for the same
// measured value.

import { CONFORMAL_BAND, GRADCAM, PREDICTIVE_ENTROPY_SINGLE_PASS } from "../api";
import { entropyColor } from "./colors";

/**
 * The only vertex-layer kinds this module knows how to colour, named by the
 * exact `X-Uncertainty-Kind` header values (api.ts) rather than an invented
 * string - so a layer can never be mislabelled as something the backend did
 * not actually send.
 */
export type VertexLayerKind =
  | typeof PREDICTIVE_ENTROPY_SINGLE_PASS
  | typeof CONFORMAL_BAND
  | typeof GRADCAM;

// entropyColor's inputs at the conformal band's two non-zero byte classes
// (128 and 255 out of 255) - Legend.tsx colours its two swatches with these
// exact same calls (see its lines 55-75), so reusing them here rather than
// hardcoding hex is what keeps the legend and the twin from ever drifting
// apart.
const CONFORMAL_SAFETY_MARGIN_ONLY = 128 / 255;
const CONFORMAL_POINT_ESTIMATE_AND_MARGIN = 1;

// "Outside the band" (point estimate only). The point estimate's own
// tumour colour is a per-region CLASS_COLORS lookup this function does not
// have access to, so this is a neutral placeholder, chosen to be visually
// distinct from anything entropyColor's magma-like ramp can produce.
const NEUTRAL_GREY: [number, number, number] = [0.85, 0.85, 0.85];

/**
 * Converts N normalised-to-[0,1] scalars into a `Float32Array` of length
 * N*3, 0..1 RGB triples for three.js's `color` vertex attribute.
 *
 * `predictive-entropy-single-pass` and `gradcam` are both continuous [0,1]
 * fields, so each scalar goes straight through `entropyColor`.
 *
 * `conformal-band` is CATEGORICAL - the source buffer only ever contains
 * byte values 0, 128 or 255 (see Legend.tsx lines 55-75), so the normalised
 * scalar is bucketed back into those three classes: below 0.25 is "outside
 * the band" (neutral grey, since this function has no way to know the
 * point estimate's own class colour); 0.25 up to (not including) 0.75 is
 * the safety-margin-only class, coloured with `entropyColor` at the
 * legend's exact 128/255 input; 0.75 and above is the point-estimate-and-
 * margin class, coloured with `entropyColor` at the legend's exact 1
 * (255/255) input. Calling `entropyColor` at those same two inputs, rather
 * than hardcoding `#E46A3F` / `#FCFDBF`, is what guarantees the legend and
 * this mesh colouring can never show two different colours for the same
 * class.
 *
 * Any other `kind` throws rather than falling back to a default mapping -
 * the whole point of threading the header value this far is that a layer
 * must never be coloured as something it is not.
 */
export function scalarsToColors(scalars: Float32Array, kind: VertexLayerKind): Float32Array {
  const out = new Float32Array(scalars.length * 3);

  // Writes entropyColor(t)'s 0-255 RGB into out[i], normalised to 0..1.
  const writeEntropyColor = (i: number, t: number) => {
    const [r, g, b] = entropyColor(t);
    out[i * 3] = r / 255;
    out[i * 3 + 1] = g / 255;
    out[i * 3 + 2] = b / 255;
  };

  if (kind === PREDICTIVE_ENTROPY_SINGLE_PASS || kind === GRADCAM) {
    for (let i = 0; i < scalars.length; i++) {
      writeEntropyColor(i, scalars[i]);
    }
  } else if (kind === CONFORMAL_BAND) {
    for (let i = 0; i < scalars.length; i++) {
      const t = scalars[i];
      if (t < 0.25) {
        out[i * 3] = NEUTRAL_GREY[0];
        out[i * 3 + 1] = NEUTRAL_GREY[1];
        out[i * 3 + 2] = NEUTRAL_GREY[2];
      } else if (t < 0.75) {
        writeEntropyColor(i, CONFORMAL_SAFETY_MARGIN_ONLY);
      } else {
        writeEntropyColor(i, CONFORMAL_POINT_ESTIMATE_AND_MARGIN);
      }
    }
  } else {
    throw new Error(`scalarsToColors: unknown vertex layer kind "${kind}"`);
  }

  return out;
}
