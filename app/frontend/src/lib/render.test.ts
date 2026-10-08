// These tests pin the COMPOSITING CONVENTION documented in render.ts's
// module comment (layer order, which mask drives which mode, the gate
// values used by disagreement mode, the fixed-scale entropy colour, and the
// draw-order of the outline over the entropy blend) - not merely whatever
// renderSlice currently outputs. A wrong layer order or an accidentally
// per-case-normalized entropy colour still produces a plausible-looking
// image, so several assertions below use expected pixel values computed by
// hand from the documented arithmetic (`v*(1-o) + c*o`) rather than by
// calling back into render.ts's own helpers.

import { describe, expect, it } from "vitest";
import { renderSlice, type RenderParams } from "./render";
import { sliceIndexer } from "./slicing";
import { ENTROPY_ALPHA_FULL } from "./colors";

/** Read one RGBA pixel out of an ImageData-shaped object. */
function px(img: ImageData, row: number, col: number): [number, number, number, number] {
  const o = (row * img.width + col) * 4;
  return [img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3]];
}

// A 2x2x2 volume viewed on the "axial" plane, slice i=0, gives a 2x2 image
// (width=D=2, height=H=2). For this shape, axial's at(row,col,0) =
// col*4 + row*2 - the four pixels of the slice read flat indices
// (row,col) -> flat: (0,0)->0, (1,0)->2, (0,1)->4, (1,1)->6. The remaining
// flat indices (1,3,5,7) belong to slice i=1 and are irrelevant here.
const SHAPE: [number, number, number] = [2, 2, 2];
const PLANE = "axial";
const SLICE = 0;
const IDX_00 = 0; // row0, col0
const IDX_10 = 2; // row1, col0
const IDX_01 = 4; // row0, col1
const IDX_11 = 6; // row1, col1

function baseParams(overrides: Partial<RenderParams>): RenderParams {
  return {
    plane: PLANE,
    shape: SHAPE,
    sliceIndex: SLICE,
    image: new Uint8Array(8),
    predictionMask: null,
    labelMask: null,
    uncertainty: null,
    overlayMode: "prediction",
    overlayOpacity: 0.5,
    showTruthOutline: false,
    showUncertainty: false,
    uncertaintyOpacity: 1,
    ...overrides,
  };
}

describe("renderSlice - greyscale passthrough", () => {
  it("outputs [v,v,v,255] with no masks or uncertainty, at the indexer's own dimensions", () => {
    const image = new Uint8Array(8);
    image[IDX_00] = 10;
    image[IDX_10] = 50;
    image[IDX_01] = 200;
    image[IDX_11] = 255;

    const out = renderSlice(baseParams({ image }));
    const idx = sliceIndexer(PLANE, SHAPE);
    expect(out.width).toBe(idx.width);
    expect(out.height).toBe(idx.height);

    expect(px(out, 0, 0)).toEqual([10, 10, 10, 255]);
    expect(px(out, 1, 0)).toEqual([50, 50, 50, 255]);
    expect(px(out, 0, 1)).toEqual([200, 200, 200, 255]);
    expect(px(out, 1, 1)).toEqual([255, 255, 255, 255]);
  });
});

describe("renderSlice - class colours", () => {
  it("blends the source grey toward the class colour at overlayOpacity; class 0 is untouched", () => {
    const image = new Uint8Array(8).fill(100);
    const predictionMask = new Uint8Array(8);
    predictionMask[IDX_00] = 0; // background
    predictionMask[IDX_10] = 1; // necrotic core -> #56B4E9 (86,180,233)
    predictionMask[IDX_01] = 2; // oedema -> #009E73 (0,158,115)
    predictionMask[IDX_11] = 3; // enhancing tumour -> #D55E00 (213,94,0)

    const out = renderSlice(
      baseParams({ image, predictionMask, overlayMode: "prediction", overlayOpacity: 0.5 }),
    );

    // v=100, o=0.5. Hand computed: v*(1-o) + c*o = 50 + c*0.5, then rounded
    // by Uint8ClampedArray (round-half-to-even where the fraction is .5).
    expect(px(out, 0, 0)).toEqual([100, 100, 100, 255]); // class 0 -> untouched
    expect(px(out, 1, 0)).toEqual([93, 140, 166, 255]); // class 1
    expect(px(out, 0, 1)).toEqual([50, 129, 108, 255]); // class 2
    expect(px(out, 1, 1)).toEqual([156, 97, 50, 255]); // class 3
  });
});

describe("renderSlice - truth mode reads the label mask, not the prediction", () => {
  it("follows labelMask content even when predictionMask disagrees everywhere", () => {
    const image = new Uint8Array(8).fill(100);
    // Every voxel is class 3 in the prediction - if truth mode accidentally
    // read this mask, every pixel would come out as the class-3 colour.
    const predictionMask = new Uint8Array(8).fill(3);
    const labelMask = new Uint8Array(8);
    labelMask[IDX_00] = 0;
    labelMask[IDX_10] = 1;
    labelMask[IDX_01] = 2;
    labelMask[IDX_11] = 3;

    const out = renderSlice(
      baseParams({ image, predictionMask, labelMask, overlayMode: "truth", overlayOpacity: 0.5 }),
    );

    expect(px(out, 0, 0)).toEqual([100, 100, 100, 255]); // label class 0
    expect(px(out, 1, 0)).toEqual([93, 140, 166, 255]); // label class 1
    expect(px(out, 0, 1)).toEqual([50, 129, 108, 255]); // label class 2
    expect(px(out, 1, 1)).toEqual([156, 97, 50, 255]); // label class 3
  });
});

describe("renderSlice - disagreement mode", () => {
  it("colours false negative / false positive, leaves agreement and background untouched, ignoring class VALUES", () => {
    const image = new Uint8Array(8).fill(100);
    const predictionMask = new Uint8Array(8);
    const labelMask = new Uint8Array(8);

    // background: pred=0, label=0
    predictionMask[IDX_00] = 0;
    labelMask[IDX_00] = 0;
    // truth-only (false negative): pred=0, label>0
    predictionMask[IDX_10] = 0;
    labelMask[IDX_10] = 1;
    // prediction-only (false positive): pred>0, label=0
    predictionMask[IDX_01] = 2;
    labelMask[IDX_01] = 0;
    // agreement, but with DIFFERENT class values on each side - disagreement
    // mode must compare only ">0", not the class value, so this must NOT
    // read as an error.
    predictionMask[IDX_11] = 1;
    labelMask[IDX_11] = 3;

    const out = renderSlice(
      baseParams({
        image,
        predictionMask,
        labelMask,
        overlayMode: "disagreement",
        overlayOpacity: 0.5,
      }),
    );

    expect(px(out, 0, 0)).toEqual([100, 100, 100, 255]); // background: untouched
    expect(px(out, 1, 0)).toEqual([170, 164, 83, 255]); // false negative -> #F0E442
    expect(px(out, 0, 1)).toEqual([152, 110, 134, 255]); // false positive -> #CC79A7
    expect(px(out, 1, 1)).toEqual([100, 100, 100, 255]); // agreement (1 vs 3): untouched
  });
});

// A 3x3 axial slice (D=3, H=3, W=1) so the truth-outline tests have a
// genuine interior voxel. axial: width=D=3, height=H=3, count=W=1; with the
// single slice i=0, at(row,col,0) = col*3 + row.
const OUTLINE_SHAPE: [number, number, number] = [3, 3, 1];
const OUTLINE_BORDER: Array<[number, number]> = [
  [0, 0],
  [0, 1],
  [0, 2],
  [1, 0],
  [1, 2],
  [2, 0],
  [2, 1],
  [2, 2],
];
const OUTLINE_CENTER: [number, number] = [1, 1];

describe("renderSlice - truth outline", () => {
  it("outlines the label blob's border voxels white, leaves the interior voxel untouched", () => {
    const image = new Uint8Array(9).fill(50);
    // A solid 3x3 label blob: every border cell has an out-of-bounds (=> 0)
    // 4-neighbour, so all 8 border cells outline. The single center cell's
    // four neighbours are all label=1, so it is genuinely interior.
    const labelMask = new Uint8Array(9).fill(1);

    const out = renderSlice(
      baseParams({
        plane: PLANE,
        shape: OUTLINE_SHAPE,
        image,
        labelMask,
        overlayMode: "prediction",
        showTruthOutline: true,
      }),
    );

    for (const [row, col] of OUTLINE_BORDER) {
      expect(px(out, row, col)).toEqual([255, 255, 255, 255]);
    }
    expect(px(out, ...OUTLINE_CENTER)).toEqual([50, 50, 50, 255]);
  });

  it("is NOT drawn in truth mode or disagreement mode, even with showTruthOutline true", () => {
    const image = new Uint8Array(9).fill(50);
    const labelMask = new Uint8Array(9).fill(1);

    for (const overlayMode of ["truth", "disagreement"] as const) {
      const out = renderSlice(
        baseParams({
          plane: PLANE,
          shape: OUTLINE_SHAPE,
          image,
          labelMask,
          predictionMask: overlayMode === "disagreement" ? new Uint8Array(9) : null,
          overlayMode,
          showTruthOutline: true,
        }),
      );
      // No pixel should be the pure-white outline colour.
      for (const [row, col] of OUTLINE_BORDER) {
        expect(px(out, row, col)).not.toEqual([255, 255, 255, 255]);
      }
    }
  });

  it("is drawn ON TOP of the entropy blend - an outline pixel stays exactly white", () => {
    const image = new Uint8Array(9).fill(50);
    const labelMask = new Uint8Array(9).fill(1);
    const uncertainty = new Uint8Array(9).fill(255); // e=1 everywhere

    const out = renderSlice(
      baseParams({
        plane: PLANE,
        shape: OUTLINE_SHAPE,
        image,
        labelMask,
        uncertainty,
        overlayMode: "prediction",
        showTruthOutline: true,
        showUncertainty: true,
        uncertaintyOpacity: 1,
      }),
    );

    for (const [row, col] of OUTLINE_BORDER) {
      expect(px(out, row, col)).toEqual([255, 255, 255, 255]);
    }
    // Interior voxel is not outlined, so it shows the full-alpha entropy
    // colour instead: at alpha=1 the source grey is fully replaced, so
    // v=50 drops out and the pixel is exactly entropyColor(1).
    expect(px(out, ...OUTLINE_CENTER)).toEqual([252, 253, 191, 255]);
  });
});

describe("renderSlice - entropy alpha ramp", () => {
  const v = 100;
  const uncertaintyOpacity = 0.6;

  function withUncertaintyByte(byte: number): ImageData {
    const image = new Uint8Array(8).fill(v);
    const uncertainty = new Uint8Array(8);
    uncertainty[IDX_00] = byte;
    return renderSlice(
      baseParams({
        image,
        uncertainty,
        showUncertainty: true,
        uncertaintyOpacity,
        overlayMode: "prediction",
      }),
    );
  }

  it("entropy >= ENTROPY_ALPHA_FULL blends at exactly uncertaintyOpacity", () => {
    // byte=255 -> e=1.0, comfortably >= ENTROPY_ALPHA_FULL, so alpha is
    // capped at exactly 1 * uncertaintyOpacity.
    const out = withUncertaintyByte(255);
    // Hand-computed: alpha=0.6, colour=entropyColor(1)=(252,253,191).
    // r = 100*0.4 + 252*0.6 = 191.2 -> 191; g = 100*0.4 + 253*0.6 = 191.8 -> 192;
    // b = 100*0.4 + 191*0.6 = 154.6 -> 155.
    expect(px(out, 0, 0)).toEqual([191, 192, 155, 255]);
  });

  it("entropy 0 leaves the pixel untouched", () => {
    const out = withUncertaintyByte(0);
    expect(px(out, 0, 0)).toEqual([v, v, v, 255]);
  });

  it("entropy ~= ENTROPY_ALPHA_FULL/2 blends at ~= uncertaintyOpacity/2", () => {
    // Entropy is stored as a uint8 byte (value/255), so ENTROPY_ALPHA_FULL/2
    // (0.175) has no exact byte representation - 255 does not divide evenly
    // by 40. We use the nearest representable byte (45 -> e=0.17647) and
    // assert the EXACT pixel that byte produces (pinning the linear ramp
    // formula precisely for a real input), plus a close-to-half check on the
    // ratio to document the intent the byte value approximates.
    const nearestHalfByte = Math.round(255 * (ENTROPY_ALPHA_FULL / 2));
    expect(nearestHalfByte).toBe(45);

    const out = withUncertaintyByte(nearestHalfByte);
    // Hand-computed (see script-verified values): alpha = 0.30252..., which
    // is within ~1% of uncertaintyOpacity/2 = 0.3 - the residual is exactly
    // the 45-vs-44.625 byte quantization gap, not a ramp-shape error.
    expect(px(out, 0, 0)).toEqual([106, 84, 98, 255]);

    const impliedAlpha = Math.min(1, nearestHalfByte / 255 / ENTROPY_ALPHA_FULL) * uncertaintyOpacity;
    expect(impliedAlpha).toBeCloseTo(uncertaintyOpacity / 2, 1);
  });
});

describe("renderSlice - entropy colour is on a FIXED scale, independent of other voxels", () => {
  it("renders an identical pixel for the same entropy byte regardless of what other voxels in the volume contain", () => {
    const targetByte = 200;
    const targetImageValue = 120;

    const imageA = new Uint8Array(8).fill(10); // low "case scale"
    imageA[IDX_00] = targetImageValue;
    const uncertaintyA = new Uint8Array(8).fill(20); // low maxima elsewhere
    uncertaintyA[IDX_00] = targetByte;

    const imageB = new Uint8Array(8).fill(240); // high "case scale"
    imageB[IDX_00] = targetImageValue;
    const uncertaintyB = new Uint8Array(8).fill(255); // high maxima elsewhere
    uncertaintyB[IDX_00] = targetByte;

    const outA = renderSlice(
      baseParams({
        image: imageA,
        uncertainty: uncertaintyA,
        showUncertainty: true,
        uncertaintyOpacity: 1,
        overlayMode: "prediction",
      }),
    );
    const outB = renderSlice(
      baseParams({
        image: imageB,
        uncertainty: uncertaintyB,
        showUncertainty: true,
        uncertaintyOpacity: 1,
        overlayMode: "prediction",
      }),
    );

    // A future "helpful" per-case normalization of the entropy colour scale
    // (e.g. dividing by the slice's own max entropy) would make these
    // diverge. They must not: the colour is a fixed function of this one
    // voxel's own entropy value alone.
    expect(px(outA, 0, 0)).toEqual(px(outB, 0, 0));
  });
});

describe("renderSlice - null safety", () => {
  it("prediction mode with predictionMask=null renders greyscale without throwing", () => {
    const image = new Uint8Array(8).fill(77);
    expect(() =>
      renderSlice(
        baseParams({ image, predictionMask: null, overlayMode: "prediction" }),
      ),
    ).not.toThrow();
    const out = renderSlice(baseParams({ image, predictionMask: null, overlayMode: "prediction" }));
    expect(px(out, 0, 0)).toEqual([77, 77, 77, 255]);
  });

  it("truth mode with labelMask=null renders greyscale without throwing", () => {
    const image = new Uint8Array(8).fill(88);
    expect(() =>
      renderSlice(baseParams({ image, labelMask: null, overlayMode: "truth" })),
    ).not.toThrow();
    const out = renderSlice(baseParams({ image, labelMask: null, overlayMode: "truth" }));
    expect(px(out, 0, 0)).toEqual([88, 88, 88, 255]);
  });

  it("showUncertainty=true with uncertainty=null renders greyscale without throwing", () => {
    const image = new Uint8Array(8).fill(99);
    expect(() =>
      renderSlice(baseParams({ image, uncertainty: null, showUncertainty: true })),
    ).not.toThrow();
    const out = renderSlice(baseParams({ image, uncertainty: null, showUncertainty: true }));
    expect(px(out, 0, 0)).toEqual([99, 99, 99, 255]);
  });
});
