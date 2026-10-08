import { describe, expect, it } from "vitest";
import { Opaque, opaque } from "./opaque";

describe("opaque", () => {
  it("returns the same wrapper for the same value", () => {
    const a = new Uint8Array(10);
    expect(opaque(a)).toBe(opaque(a));
  });

  it("returns different wrappers for different values", () => {
    expect(opaque(new Uint8Array(10))).not.toBe(opaque(new Uint8Array(10)));
  });

  it("exposes no enumerable keys", () => {
    const w = opaque(new Uint8Array(1000));
    const keys: string[] = [];
    for (const k in w) keys.push(k);
    expect(keys).toEqual([]);
    expect(Object.keys(w)).toEqual([]);
  });

  it("unwraps to the original", () => {
    const a = new Uint8Array(4);
    expect(opaque(a).value).toBe(a);
    expect(opaque(a)).toBeInstanceOf(Opaque);
  });

  it("maps null and undefined to null", () => {
    expect(opaque(null)).toBeNull();
    expect(opaque(undefined)).toBeNull();
  });
});
