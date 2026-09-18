// Tests for structureColor (see structureColors.ts).

import { describe, expect, it } from "vitest";
import { structureColor } from "./structureColors";

describe("structureColor", () => {
  it("is deterministic - same index always returns the same colour", () => {
    expect(structureColor(5)).toEqual(structureColor(5));
  });

  it("returns values in [0, 1] for a range of indices", () => {
    for (let i = 1; i <= 122; i++) {
      const [r, g, b] = structureColor(i);
      for (const c of [r, g, b]) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });

  it("gives visually distinct colours to consecutive indices", () => {
    for (let i = 1; i < 16; i++) {
      const a = structureColor(i);
      const b = structureColor(i + 1);
      const dist = Math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2,
      );
      expect(dist).toBeGreaterThan(0.15);
    }
  });

  it("differs between two arbitrary distinct indices", () => {
    expect(structureColor(1)).not.toEqual(structureColor(2));
    expect(structureColor(3)).not.toEqual(structureColor(4));
  });
});
