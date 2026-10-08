import { describe, expect, it } from "vitest";
import { extentCenter, meshVoxelField, normalToScene, voxelToScene } from "./twinGeometry";

// Builds a w-fastest (D,H,W) field (flat index d*H*W + h*W + w, matching
// every real volume buffer in this codebase - see slicing.ts) containing a
// filled sphere of the given radius centred at voxel (cd, ch, cw).
function sphereField(shape: [number, number, number], center: [number, number, number], radius: number): Float32Array {
  const [D, H, W] = shape;
  const [cd, ch, cw] = center;
  const field = new Float32Array(D * H * W);
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        const dist = Math.hypot(d - cd, h - ch, w - cw);
        field[d * H * W + h * W + w] = dist <= radius ? 1 : 0;
      }
    }
  }
  return field;
}

describe("meshVoxelField", () => {
  // This is the exact probe that exposed bug 1 (axis order): calling
  // surfaceNets with dims [D,H,W] on a w-fastest buffer reinterprets the
  // strides whenever D !== W, turning a sphere into a smear. A 30x40x50
  // field (D=30, H=40, W=50, all distinct) centred off-center guarantees
  // any axis mixup shows up as a wrong centroid or wrong extent.
  it("recovers a sphere's centroid and extent in (d,h,w) voxel order", () => {
    const shape: [number, number, number] = [30, 40, 50];
    const center: [number, number, number] = [8, 30, 40];
    const radius = 5;
    const field = sphereField(shape, center, radius);

    const mesh = meshVoxelField(field, shape, 0.5);
    expect(mesh.positions.length).toBeGreaterThan(0);

    let sumD = 0;
    let sumH = 0;
    let sumW = 0;
    let minD = Infinity;
    let maxD = -Infinity;
    let minH = Infinity;
    let maxH = -Infinity;
    let minW = Infinity;
    let maxW = -Infinity;
    const nVerts = mesh.positions.length / 3;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const d = mesh.positions[i];
      const h = mesh.positions[i + 1];
      const w = mesh.positions[i + 2];
      sumD += d;
      sumH += h;
      sumW += w;
      minD = Math.min(minD, d);
      maxD = Math.max(maxD, d);
      minH = Math.min(minH, h);
      maxH = Math.max(maxH, h);
      minW = Math.min(minW, w);
      maxW = Math.max(maxW, w);
    }

    // Centroid within 0.5 voxels on every axis - checked directly rather
    // than with toBeCloseTo, whose second argument is a decimal-digit
    // count, not a tolerance.
    expect(Math.abs(sumD / nVerts - center[0])).toBeLessThan(0.5);
    expect(Math.abs(sumH / nVerts - center[1])).toBeLessThan(0.5);
    expect(Math.abs(sumW / nVerts - center[2])).toBeLessThan(0.5);

    const extentD = maxD - minD;
    const extentH = maxH - minH;
    const extentW = maxW - minW;
    for (const extent of [extentD, extentH, extentW]) {
      expect(extent).toBeGreaterThan(9);
      expect(extent).toBeLessThan(11);
    }
  });
});

describe("voxelToScene", () => {
  const center: [number, number, number] = [10, 10, 10];
  const scale = 2;

  it("maps larger d to larger X", () => {
    const positions = new Float32Array([10, 10, 10, 15, 10, 10]);
    const out = voxelToScene(positions, center, scale);
    expect(out[3]).toBeGreaterThan(out[0]);
  });

  it("maps larger w to larger Y", () => {
    const positions = new Float32Array([10, 10, 10, 10, 10, 15]);
    const out = voxelToScene(positions, center, scale);
    expect(out[4]).toBeGreaterThan(out[1]);
  });

  it("maps smaller h to larger Z", () => {
    const positions = new Float32Array([10, 10, 10, 10, 5, 10]);
    const out = voxelToScene(positions, center, scale);
    expect(out[5]).toBeGreaterThan(out[2]);
  });

  it("is right-handed: X-hat cross Y-hat = Z-hat for unit voxel steps", () => {
    // Scene images of unit steps along d and w give the scene X and Y axes
    // directly. h runs ANTERIOR -> POSTERIOR, and scene +Z is anterior, so
    // the scene image of a unit step in -h (not +h) gives scene +Z; this is
    // exactly what a right-handed (Left, Superior, Anterior) frame means.
    const base = voxelToScene(new Float32Array([0, 0, 0]), [0, 0, 0], 1);
    const dStep = voxelToScene(new Float32Array([1, 0, 0]), [0, 0, 0], 1);
    const wStep = voxelToScene(new Float32Array([0, 0, 1]), [0, 0, 0], 1);
    const hStepNeg = voxelToScene(new Float32Array([0, -1, 0]), [0, 0, 0], 1);

    const sub = (a: Float32Array, b: Float32Array) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const xHat = sub(dStep, base);
    const yHat = sub(wStep, base);
    const zHat = sub(hStepNeg, base);

    const cross = (a: number[], b: number[]) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const xCrossY = cross(xHat, yHat);

    expect(xCrossY[0]).toBeCloseTo(zHat[0], 6);
    expect(xCrossY[1]).toBeCloseTo(zHat[1], 6);
    expect(xCrossY[2]).toBeCloseTo(zHat[2], 6);
  });
});

describe("normalToScene", () => {
  it("is a pure permutation/sign transform: length preserved", () => {
    const normals = new Float32Array([0.2, -0.6, 0.7745966692414834]); // unit-ish vector
    const len = Math.hypot(normals[0], normals[1], normals[2]);
    const out = normalToScene(normals);
    const outLen = Math.hypot(out[0], out[1], out[2]);
    expect(outLen).toBeCloseTo(len, 6);
    // Exact component mapping: (nd, nh, nw) -> (nd, nw, -nh).
    expect(out[0]).toBeCloseTo(normals[0], 10);
    expect(out[1]).toBeCloseTo(normals[2], 10);
    expect(out[2]).toBeCloseTo(-normals[1], 10);
  });
});

describe("winding preserved through voxelToScene", () => {
  // The voxel(d,h,w) -> scene linear map has determinant +1 (a rotation,
  // not a reflection - see voxelToScene's docstring), so every triangle's
  // outward-facing geometric normal (right-hand rule on its edges) should
  // still agree in direction with its own vertex normals after both go
  // through voxelToScene / normalToScene. If winding needed flipping (a
  // reflection) essentially every triangle's dot product would come out
  // negative. Uses a solid box (same hand-checkable shape as
  // surfaceNets.test.ts) rather than a sphere: flat faces give an
  // unambiguous face normal, whereas a sphere's naive-surface-nets mesh can
  // include a few tiny near-degenerate boundary triangles whose face
  // normal is dominated by rounding noise and isn't a meaningful winding
  // check either way - skipped here via the geomLen guard.
  it("keeps every box-mesh triangle's geometric normal aligned with its averaged vertex normal", () => {
    const shape: [number, number, number] = [10, 10, 10];
    const [D, H, W] = shape;
    const field = new Float32Array(D * H * W);
    for (let d = 0; d < D; d++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          const inside = d >= 2 && d < 8 && h >= 2 && h < 8 && w >= 2 && w < 8;
          field[d * H * W + h * W + w] = inside ? 1 : 0;
        }
      }
    }

    const mesh = meshVoxelField(field, shape, 0.5);
    const { center, scale } = extentCenter(mesh.positions);
    const scenePos = voxelToScene(mesh.positions, center, scale);
    const sceneNorm = normalToScene(mesh.normals);

    const nTris = mesh.indices.length / 3;
    let checked = 0;
    for (let t = 0; t < nTris; t++) {
      const i0 = mesh.indices[t * 3];
      const i1 = mesh.indices[t * 3 + 1];
      const i2 = mesh.indices[t * 3 + 2];

      const p0 = [scenePos[i0 * 3], scenePos[i0 * 3 + 1], scenePos[i0 * 3 + 2]];
      const p1 = [scenePos[i1 * 3], scenePos[i1 * 3 + 1], scenePos[i1 * 3 + 2]];
      const p2 = [scenePos[i2 * 3], scenePos[i2 * 3 + 1], scenePos[i2 * 3 + 2]];

      const edge1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
      const edge2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
      const geomNormal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
      ];
      const geomLen = Math.hypot(geomNormal[0], geomNormal[1], geomNormal[2]);
      if (geomLen < 1e-6) continue; // degenerate triangle, no meaningful face normal

      const avgNormal = [0, 1, 2].map(
        (c) => (sceneNorm[i0 * 3 + c] + sceneNorm[i1 * 3 + c] + sceneNorm[i2 * 3 + c]) / 3,
      );
      const dot = geomNormal[0] * avgNormal[0] + geomNormal[1] * avgNormal[1] + geomNormal[2] * avgNormal[2];
      expect(dot).toBeGreaterThan(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  // Consistency (above) only checks that geometric and vertex normals agree
  // with EACH OTHER - a mesh with every normal flipped inward would pass it
  // too. This checks they also agree with the one absolute reference a
  // centred, convex box gives for free: outward from its own centre. The
  // box is centred by extentCenter, so the scene origin sits at its centre,
  // and every vertex normal should point away from the origin.
  it("points every vertex normal outward, away from the box centre", () => {
    const shape: [number, number, number] = [10, 10, 10];
    const [D, H, W] = shape;
    const field = new Float32Array(D * H * W);
    for (let d = 0; d < D; d++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          const inside = d >= 2 && d < 8 && h >= 2 && h < 8 && w >= 2 && w < 8;
          field[d * H * W + h * W + w] = inside ? 1 : 0;
        }
      }
    }

    const mesh = meshVoxelField(field, shape, 0.5);
    const { center, scale } = extentCenter(mesh.positions);
    const scenePos = voxelToScene(mesh.positions, center, scale);
    const sceneNorm = normalToScene(mesh.normals);

    const nVerts = scenePos.length / 3;
    let checked = 0;
    for (let v = 0; v < nVerts; v++) {
      const p = [scenePos[v * 3], scenePos[v * 3 + 1], scenePos[v * 3 + 2]];
      const distFromOrigin = Math.hypot(p[0], p[1], p[2]);
      if (distFromOrigin < 1e-6) continue; // no meaningful outward direction at the origin

      const n = [sceneNorm[v * 3], sceneNorm[v * 3 + 1], sceneNorm[v * 3 + 2]];
      const dot = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
      expect(dot).toBeGreaterThan(0);
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });
});
