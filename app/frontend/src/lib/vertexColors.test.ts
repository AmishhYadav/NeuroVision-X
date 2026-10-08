import { describe, expect, it } from "vitest";
import { CONFORMAL_BAND, GRADCAM, PREDICTIVE_ENTROPY_SINGLE_PASS } from "../api";
import { entropyColor, hexToRgb } from "./colors";
import { scalarsToColors, type VertexLayerKind } from "./vertexColors";

describe("scalarsToColors", () => {
  it("colours predictive-entropy-single-pass with entropyColor(t)/255", () => {
    const scalars = new Float32Array([0, 0.5, 1]);
    const colors = scalarsToColors(scalars, PREDICTIVE_ENTROPY_SINGLE_PASS);

    for (let i = 0; i < scalars.length; i++) {
      const [r, g, b] = entropyColor(scalars[i]);
      expect(colors[i * 3]).toBeCloseTo(r / 255, 6);
      expect(colors[i * 3 + 1]).toBeCloseTo(g / 255, 6);
      expect(colors[i * 3 + 2]).toBeCloseTo(b / 255, 6);
    }
  });

  it("colours gradcam the same way as entropy, since both are continuous [0,1] fields", () => {
    const t = 0.42;
    const [r, g, b] = entropyColor(t);
    const colors = scalarsToColors(new Float32Array([t]), GRADCAM);

    expect(colors[0]).toBeCloseTo(r / 255, 6);
    expect(colors[1]).toBeCloseTo(g / 255, 6);
    expect(colors[2]).toBeCloseTo(b / 255, 6);
  });

  it("colours the conformal band's two non-zero classes with exactly the legend's swatch colours", () => {
    // Legend.tsx (lines 55-75) draws these two swatches as "#E46A3F" and
    // "#FCFDBF" - the exact hex it says comes from entropyColor(128/255)
    // and entropyColor(255/255). This test locks the twin's colouring to
    // that same source so the two views can never drift apart.
    const [legendMarginR, legendMarginG, legendMarginB] = hexToRgb("#E46A3F");
    const [legendPointR, legendPointG, legendPointB] = hexToRgb("#FCFDBF");

    // A value inside [0.25, 0.75) is the "safety margin only" class.
    const marginColors = scalarsToColors(new Float32Array([0.5]), CONFORMAL_BAND);
    expect(marginColors[0]).toBeCloseTo(legendMarginR / 255, 2);
    expect(marginColors[1]).toBeCloseTo(legendMarginG / 255, 2);
    expect(marginColors[2]).toBeCloseTo(legendMarginB / 255, 2);

    // A value >= 0.75 is "point estimate and margin".
    const pointColors = scalarsToColors(new Float32Array([1]), CONFORMAL_BAND);
    expect(pointColors[0]).toBeCloseTo(legendPointR / 255, 2);
    expect(pointColors[1]).toBeCloseTo(legendPointG / 255, 2);
    expect(pointColors[2]).toBeCloseTo(legendPointB / 255, 2);
  });

  it("colours a conformal-band value below 0.25 as neutral grey (outside the band)", () => {
    const colors = scalarsToColors(new Float32Array([0.1]), CONFORMAL_BAND);
    expect(colors[0]).toBeCloseTo(0.85, 6);
    expect(colors[1]).toBeCloseTo(0.85, 6);
    expect(colors[2]).toBeCloseTo(0.85, 6);
  });

  it("throws on an unknown kind", () => {
    expect(() =>
      scalarsToColors(new Float32Array([0.5]), "not-a-real-kind" as VertexLayerKind),
    ).toThrow();
  });

  it("returns an array of length 3N for N scalars", () => {
    const scalars = new Float32Array([0, 0.2, 0.4, 0.6, 0.8]);
    const colors = scalarsToColors(scalars, PREDICTIVE_ENTROPY_SINGLE_PASS);
    expect(colors.length).toBe(scalars.length * 3);
  });
});
