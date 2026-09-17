import { describe, expect, it } from "vitest";
import { PREDICTIVE_ENTROPY_SINGLE_PASS } from "../api";
import { sampleLayersForClasses } from "./twinLayers";

describe("sampleLayersForClasses", () => {
  it("samples one voxel inward for a single class/layer and mirrors the layer keys", () => {
    const shape: [number, number, number] = [4, 4, 4];
    const volume = new Uint8Array(4 * 4 * 4);
    // index = d*H*W + h*W + w, H=W=4 (see slicing.ts / vertexScalars.ts).
    // A vertex sits on the w=3 face with an outward +w normal, so the
    // INWARD sample (one voxel back along -normal) must land at w=2, not
    // at the vertex's own w=3 voxel.
    volume[1 * 16 + 1 * 4 + 2] = 100; // the voxel the inward sample should read
    volume[1 * 16 + 1 * 4 + 3] = 40; // the vertex's own voxel - must NOT be read

    const voxelGeometry = {
      necrotic: {
        positions: new Float32Array([1, 1, 3]), // (d,h,w) on the +w face
        normals: new Float32Array([0, 0, 1]), // outward, pointing +w
      },
    };

    const result = sampleLayersForClasses(
      { [PREDICTIVE_ENTROPY_SINGLE_PASS]: volume },
      shape,
      voxelGeometry,
    );

    expect(Object.keys(result)).toEqual([PREDICTIVE_ENTROPY_SINGLE_PASS]);
    const perClass = result[PREDICTIVE_ENTROPY_SINGLE_PASS];
    expect(perClass && Object.keys(perClass)).toEqual(["necrotic"]);
    expect(perClass?.necrotic?.[0]).toBeCloseTo(100 / 255, 6);
  });

  it("returns an empty object when there are no layers, and skips a class the layer has no geometry for", () => {
    const shape: [number, number, number] = [4, 4, 4];
    const voxelGeometry = {
      necrotic: {
        positions: new Float32Array([1, 1, 3]),
        normals: new Float32Array([0, 0, 1]),
      },
    };

    expect(sampleLayersForClasses({}, shape, voxelGeometry)).toEqual({});

    const volume = new Uint8Array(4 * 4 * 4).fill(10);
    const result = sampleLayersForClasses({ [PREDICTIVE_ENTROPY_SINGLE_PASS]: volume }, shape, {});
    expect(result[PREDICTIVE_ENTROPY_SINGLE_PASS]).toEqual({});
  });
});
