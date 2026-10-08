import { describe, expect, it, vi } from "vitest";
import { InflightMap, LruCache } from "./caseCache";

describe("LruCache", () => {
  it("evicts the least recently used entry over capacity", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    c.set("c", 3);
    expect(c.keys()).toEqual(["b", "c"]);
    expect(c.get("a")).toBeUndefined();
  });

  it("get refreshes recency", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    expect(c.get("a")).toBe(1);
    c.set("c", 3); // evicts b, not a
    expect(c.keys()).toEqual(["a", "c"]);
  });

  it("re-setting a key refreshes it and does not grow", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    c.set("a", 9);
    expect(c.size).toBe(2);
    expect(c.keys()).toEqual(["b", "a"]);
    expect(c.get("a")).toBe(9);
  });

  it("rejects a capacity below 1", () => {
    expect(() => new LruCache(0)).toThrow();
  });
});

describe("InflightMap", () => {
  it("returns the same promise for concurrent calls", async () => {
    const m = new InflightMap<number>();
    const start = vi.fn(() => Promise.resolve(1));
    const p1 = m.run("x", start);
    const p2 = m.run("x", start);
    expect(p1).toBe(p2);
    expect(start).toHaveBeenCalledTimes(1);
    await p1;
  });

  it("starts fresh work after a success settles", async () => {
    const m = new InflightMap<number>();
    const start = vi.fn(() => Promise.resolve(1));
    await m.run("x", start);
    await Promise.resolve();
    await m.run("x", start);
    expect(start).toHaveBeenCalledTimes(2);
  });

  it("evicts a failed promise so a retry refetches", async () => {
    const m = new InflightMap<number>();
    const start = vi
      .fn<() => Promise<number>>()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(7);
    await expect(m.run("x", start)).rejects.toThrow("boom");
    await Promise.resolve();
    expect(m.has("x")).toBe(false);
    await expect(m.run("x", start)).resolves.toBe(7);
    expect(start).toHaveBeenCalledTimes(2);
  });
});
