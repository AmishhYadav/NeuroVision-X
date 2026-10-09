// Hand-computed checks for the landing page's scroll maths (scrollScene.ts).

import { describe, expect, it } from "vitest";
import { poseAt, progressBetween, smoothstep, tickValue, type PoseKey } from "./scrollScene";

const A = { x: 0, y: 0, size: 1, opacity: 1 };
const B = { x: 1, y: 0.5, size: 3, opacity: 0 };
const KEYS: PoseKey[] = [
  { at: 100, pose: A },
  { at: 300, pose: B },
];

describe("progressBetween", () => {
  it("clamps below and above the range", () => {
    expect(progressBetween(100, 200, 50)).toBe(0);
    expect(progressBetween(100, 200, 250)).toBe(1);
  });
  it("is linear inside the range", () => {
    expect(progressBetween(100, 200, 125)).toBe(0.25);
  });
  it("treats an empty range as a step at its end", () => {
    expect(progressBetween(200, 200, 199)).toBe(0);
    expect(progressBetween(200, 200, 200)).toBe(1);
  });
});

describe("smoothstep", () => {
  it("is 0, 0.5 and 1 at the start, middle and end", () => {
    expect(smoothstep(0, 10, 0)).toBe(0);
    expect(smoothstep(0, 10, 5)).toBe(0.5);
    expect(smoothstep(0, 10, 10)).toBe(1);
  });
});

describe("poseAt", () => {
  it("holds the first pose before the first key", () => {
    expect(poseAt(KEYS, 0)).toEqual(A);
  });
  it("holds the last pose after the last key", () => {
    expect(poseAt(KEYS, 10_000)).toEqual(B);
  });
  it("returns the exact pose at a key", () => {
    expect(poseAt(KEYS, 300)).toEqual(B);
  });
  it("blends halfway at the midpoint (smoothstep(0.5) = 0.5)", () => {
    expect(poseAt(KEYS, 200)).toEqual({ x: 0.5, y: 0.25, size: 2, opacity: 0.5 });
  });
  it("picks the right pair among three keys", () => {
    const keys: PoseKey[] = [...KEYS, { at: 500, pose: A }];
    expect(poseAt(keys, 400)).toEqual({ x: 0.5, y: 0.25, size: 2, opacity: 0.5 });
  });
  it("is invisible, not a crash, with no keys", () => {
    expect(poseAt([], 50).opacity).toBe(0);
  });
});

describe("tickValue", () => {
  it("returns the published string verbatim when settled", () => {
    expect(tickValue("+0.0267", 1)).toBe("+0.0267");
    expect(tickValue("0.64–0.96×", 1.5)).toBe("0.64–0.96×");
  });
  it("keeps sign, unit and decimals while counting", () => {
    expect(tickValue("+0.0267", 0)).toBe("+0.0000");
    expect(tickValue("85.2%", 0.5)).toBe("42.6%");
  });
  it("counts both ends of a range", () => {
    expect(tickValue("0.64–0.96×", 0.5)).toBe("0.32–0.48×");
  });
});
