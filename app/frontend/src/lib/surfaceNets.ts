// Naive Surface Nets: turns a scalar field on a regular grid into a
// triangle mesh crossing one isovalue. Chosen over classic Marching Cubes
// for the digital twin because it needs no 256-entry triangulation table -
// one vertex per boundary cell, positioned at the average of that cell's
// edge crossings, connected into quads along the three axes wherever a
// neighbouring cell also crosses the isovalue. Runs inside a Web Worker
// (see workers/twinMesh.worker.ts) so a ~3M-voxel case never blocks the
// main thread.
//
// Reference: Gibson, "Constrained Elastic Surface Nets" (1998); this is the
// widely-taught unconstrained/naive variant.

export interface SurfaceNetsResult {
  positions: Float32Array; // (N, 3)
  normals: Float32Array; // (N, 3)
  indices: Uint32Array; // (M, 3)
}

const CUBE_EDGES: readonly [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 4],
  [1, 3],
  [1, 5],
  [2, 3],
  [2, 6],
  [3, 7],
  [4, 5],
  [4, 6],
  [5, 7],
  [6, 7],
];

// Corner offsets in the same bit order used above: bit0=x, bit1=y, bit2=z.
const CORNER_OFFSETS: readonly [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
];

/**
 * @param field flat (dx, dy, dz) scalar field, C order (x-fastest... see index())
 * @param dims [dx, dy, dz]
 * @param isovalue surface threshold; cells straddling it become boundary cells
 */
export function surfaceNets(
  field: Float32Array | Uint8Array,
  dims: readonly [number, number, number],
  isovalue: number,
): SurfaceNetsResult {
  const [dx, dy, dz] = dims;
  const index = (x: number, y: number, z: number) => x + dx * (y + dy * z);

  // One cell per (dx-1, dy-1, dz-1) grid of cubes. cellVertex maps a cell's
  // flat cell-index to its vertex index in `positions`, or -1 if inactive.
  const cdx = dx - 1;
  const cdy = dy - 1;
  const cdz = dz - 1;
  const cellIndex = (x: number, y: number, z: number) => x + cdx * (y + cdy * z);
  const cellVertex = new Int32Array(Math.max(1, cdx * cdy * cdz)).fill(-1);

  const positions: number[] = [];

  for (let z = 0; z < cdz; z++) {
    for (let y = 0; y < cdy; y++) {
      for (let x = 0; x < cdx; x++) {
        const corner = new Float32Array(8);
        let inside = 0;
        for (let c = 0; c < 8; c++) {
          const [ox, oy, oz] = CORNER_OFFSETS[c];
          const v = field[index(x + ox, y + oy, z + oz)];
          corner[c] = v;
          if (v > isovalue) inside |= 1 << c;
        }
        if (inside === 0 || inside === 255) continue; // uniformly outside or inside

        let sx = 0;
        let sy = 0;
        let sz = 0;
        let crossings = 0;
        for (const [a, b] of CUBE_EDGES) {
          const va = corner[a];
          const vb = corner[b];
          const aInside = va > isovalue;
          const bInside = vb > isovalue;
          if (aInside === bInside) continue;
          const t = (isovalue - va) / (vb - va);
          const [ax, ay, az] = CORNER_OFFSETS[a];
          const [bx, by, bz] = CORNER_OFFSETS[b];
          sx += ax + t * (bx - ax);
          sy += ay + t * (by - ay);
          sz += az + t * (bz - az);
          crossings++;
        }
        // crossings is always >= 2 here (inside is neither 0 nor 255).
        const vx = x + sx / crossings;
        const vy = y + sy / crossings;
        const vz = z + sz / crossings;
        cellVertex[cellIndex(x, y, z)] = positions.length / 3;
        positions.push(vx, vy, vz);
      }
    }
  }

  const indices: number[] = [];
  // For each axis, walk every edge of the GRID (not the cube list above) and,
  // where the two grid points straddle the isovalue, quad together the up to
  // four cells sharing that edge.
  const emitQuad = (v0: number, v1: number, v2: number, v3: number, flip: boolean) => {
    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) return;
    if (flip) {
      indices.push(v0, v2, v1, v0, v3, v2);
    } else {
      indices.push(v0, v1, v2, v0, v2, v3);
    }
  };

  for (let z = 0; z < dz; z++) {
    for (let y = 0; y < dy; y++) {
      for (let x = 0; x < dx; x++) {
        const v0 = field[index(x, y, z)];
        // Edge along X: needs y>0,z>0 to have all four neighbouring cells.
        if (x + 1 < dx && y > 0 && z > 0 && y < dy && z < dz) {
          const v1 = field[index(x + 1, y, z)];
          const a = v0 > isovalue;
          const b = v1 > isovalue;
          if (a !== b) {
            emitQuad(
              cellVertex[cellIndex(x, y - 1, z - 1)],
              cellVertex[cellIndex(x, y, z - 1)],
              cellVertex[cellIndex(x, y, z)],
              cellVertex[cellIndex(x, y - 1, z)],
              a,
            );
          }
        }
        // Edge along Y.
        if (y + 1 < dy && x > 0 && z > 0) {
          const v1 = field[index(x, y + 1, z)];
          const a = v0 > isovalue;
          const b = v1 > isovalue;
          if (a !== b) {
            emitQuad(
              cellVertex[cellIndex(x - 1, y, z - 1)],
              cellVertex[cellIndex(x, y, z - 1)],
              cellVertex[cellIndex(x, y, z)],
              cellVertex[cellIndex(x - 1, y, z)],
              !a,
            );
          }
        }
        // Edge along Z.
        if (z + 1 < dz && x > 0 && y > 0) {
          const v1 = field[index(x, y, z + 1)];
          const a = v0 > isovalue;
          const b = v1 > isovalue;
          if (a !== b) {
            emitQuad(
              cellVertex[cellIndex(x - 1, y - 1, z)],
              cellVertex[cellIndex(x, y - 1, z)],
              cellVertex[cellIndex(x, y, z)],
              cellVertex[cellIndex(x - 1, y, z)],
              a,
            );
          }
        }
      }
    }
  }

  const positionArr = new Float32Array(positions);
  const indexArr = new Uint32Array(indices);
  const normalArr = computeNormals(positionArr, indexArr);
  return { positions: positionArr, normals: normalArr, indices: indexArr };
}

function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);
  const nTris = indices.length / 3;
  for (let t = 0; t < nTris; t++) {
    const i0 = indices[t * 3] * 3;
    const i1 = indices[t * 3 + 1] * 3;
    const i2 = indices[t * 3 + 2] * 3;
    const ax = positions[i1] - positions[i0];
    const ay = positions[i1 + 1] - positions[i0 + 1];
    const az = positions[i1 + 2] - positions[i0 + 2];
    const bx = positions[i2] - positions[i0];
    const by = positions[i2 + 1] - positions[i0 + 1];
    const bz = positions[i2 + 2] - positions[i0 + 2];
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    for (const i of [i0, i1, i2]) {
      normals[i] += nx;
      normals[i + 1] += ny;
      normals[i + 2] += nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
    normals[i] /= len;
    normals[i + 1] /= len;
    normals[i + 2] /= len;
  }
  return normals;
}
