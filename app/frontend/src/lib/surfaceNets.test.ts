import { describe, expect, it } from "vitest";
import { surfaceNets } from "./surfaceNets";

// value(x,y,z) = 1 inside a solid box, 0 outside. A closed box has a known,
// hand-checkable bounding volume: every extracted vertex must fall on or
// inside [0, 6] on each axis (the field's own extent) and the surface must
// bound the solid region, not overshoot it - the failure mode a botched
// corner-offset or edge-crossing formula produces is a mesh with vertices
// outside the field's bounds entirely, which this catches directly.
function solidBoxField(dims: [number, number, number], lo: number, hi: number): Float32Array {
  const [dx, dy, dz] = dims;
  const field = new Float32Array(dx * dy * dz);
  for (let z = 0; z < dz; z++) {
    for (let y = 0; y < dy; y++) {
      for (let x = 0; x < dx; x++) {
        const inside = x >= lo && x < hi && y >= lo && y < hi && z >= lo && z < hi;
        field[x + dx * (y + dy * z)] = inside ? 1 : 0;
      }
    }
  }
  return field;
}

describe("surfaceNets", () => {
  it("returns an empty mesh for a uniform field (no isovalue crossing)", () => {
    const dims: [number, number, number] = [4, 4, 4];
    const field = new Float32Array(4 * 4 * 4); // all zero, isovalue 0.5 never crossed
    const result = surfaceNets(field, dims, 0.5);
    expect(result.positions.length).toBe(0);
    expect(result.indices.length).toBe(0);
  });

  it("extracts a closed, well-formed surface bounding a solid box", () => {
    const dims: [number, number, number] = [8, 8, 8];
    const field = solidBoxField(dims, 2, 6);
    const result = surfaceNets(field, dims, 0.5);

    expect(result.positions.length).toBeGreaterThan(0);
    expect(result.indices.length).toBeGreaterThan(0);
    expect(result.indices.length % 3).toBe(0);

    const nVerts = result.positions.length / 3;
    for (let i = 0; i < result.indices.length; i++) {
      expect(result.indices[i]).toBeGreaterThanOrEqual(0);
      expect(result.indices[i]).toBeLessThan(nVerts);
    }

    // Every vertex lies within the field's own coordinate extent, and (since
    // the box occupies [2, 6)) strictly inside the field's outer boundary -
    // a vertex at exactly x=0 or x=7 would mean the surface leaked to the
    // field's edge instead of tracking the box.
    for (let i = 0; i < result.positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const v = result.positions[i + axis];
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThan(0);
        expect(v).toBeLessThan(7);
      }
    }

    // Normals are unit length (or exactly zero only if genuinely degenerate,
    // which a solid box's flat faces never are).
    for (let i = 0; i < result.normals.length; i += 3) {
      const len = Math.hypot(result.normals[i], result.normals[i + 1], result.normals[i + 2]);
      expect(len).toBeGreaterThan(0.99);
      expect(len).toBeLessThan(1.01);
    }
  });

  it("is watertight: every edge is shared by exactly two triangles", () => {
    const dims: [number, number, number] = [8, 8, 8];
    const field = solidBoxField(dims, 2, 6);
    const { indices } = surfaceNets(field, dims, 0.5);

    const edgeCount = new Map<string, number>();
    for (let t = 0; t < indices.length / 3; t++) {
      const tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
      for (let e = 0; e < 3; e++) {
        const a = tri[e];
        const b = tri[(e + 1) % 3];
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
      }
    }
    for (const count of edgeCount.values()) {
      expect(count).toBe(2);
    }
  });

  // FNV-1a 32-bit hash, used only to compress a golden mesh into one constant
  // below so the fixture itself doesn't have to live in the repo.
  function fnv1a(str: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }

  it("matches the pre-optimisation golden checksum bit-for-bit", () => {
    // Same field the golden fixture was generated from: dims [40, 33, 29],
    // a sphere of radius 11 centred at the grid middle, plus a second,
    // off-centre blob of radius 5 at (8, 8, 8). 1 inside either sphere, 0
    // outside, isovalue 0.5.
    const dims: [number, number, number] = [40, 33, 29];
    const [dx, dy, dz] = dims;
    const field = new Float32Array(dx * dy * dz);
    const cx = dx / 2;
    const cy = dy / 2;
    const cz = dz / 2;
    const r1 = 11;
    const bx = 8;
    const by = 8;
    const bz = 8;
    const r2 = 5;
    for (let z = 0; z < dz; z++) {
      for (let y = 0; y < dy; y++) {
        for (let x = 0; x < dx; x++) {
          const d1 = Math.hypot(x - cx, y - cy, z - cz);
          const d2 = Math.hypot(x - bx, y - by, z - bz);
          const inside = d1 < r1 || d2 < r2;
          field[x + dx * (y + dy * z)] = inside ? 1 : 0;
        }
      }
    }

    const result = surfaceNets(field, dims, 0.5);

    expect(result.positions.length).toBeGreaterThan(0);
    expect(result.indices.length % 3).toBe(0);
    for (let i = 0; i < result.normals.length; i += 3) {
      const len = Math.hypot(result.normals[i], result.normals[i + 1], result.normals[i + 2]);
      expect(len).toBeGreaterThan(1 - 1e-4);
      expect(len).toBeLessThan(1 + 1e-4);
    }

    const round = (v: number) => Math.round(v * 1e6) / 1e6;
    const positions = Array.from(result.positions, round);
    const normals = Array.from(result.normals, round);
    const indices = Array.from(result.indices);
    const str = `${JSON.stringify(positions)}|${JSON.stringify(normals)}|${JSON.stringify(indices)}`;

    // Generated from the UNMODIFIED (pre-hot-loop-optimisation) surfaceNets
    // implementation, against the field described above. Any change to
    // vertex order, vertex position, index order/winding, or normals must
    // reproduce this exact value.
    const GOLDEN_CHECKSUM = 2228312546;
    expect(fnv1a(str)).toBe(GOLDEN_CHECKSUM);
  });
});
