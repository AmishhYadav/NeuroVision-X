// Boundary tests for formatRelativeTime (see relativeTime.ts) - one case
// just inside and one just outside each bucket edge (minute/hour/day), plus
// the UTC-date fallback for anything a day or older.

import { describe, expect, it } from "vitest";
import { formatRelativeTime } from "./relativeTime";

const NOW = Date.parse("2026-09-18T12:00:00.000Z");
const nowSec = NOW / 1000;

describe("formatRelativeTime", () => {
  it("renders 0 seconds ago as just now", () => {
    expect(formatRelativeTime(nowSec, NOW)).toBe("just now");
  });

  it("renders 59 seconds ago as just now", () => {
    expect(formatRelativeTime(nowSec - 59, NOW)).toBe("just now");
  });

  it("renders exactly 60 seconds ago as 1 min ago", () => {
    expect(formatRelativeTime(nowSec - 60, NOW)).toBe("1 min ago");
  });

  it("renders 3 minutes ago as 3 min ago", () => {
    expect(formatRelativeTime(nowSec - 3 * 60, NOW)).toBe("3 min ago");
  });

  it("renders 59 minutes ago as 59 min ago", () => {
    expect(formatRelativeTime(nowSec - 59 * 60, NOW)).toBe("59 min ago");
  });

  it("renders exactly 60 minutes ago as 1 h ago", () => {
    expect(formatRelativeTime(nowSec - 60 * 60, NOW)).toBe("1 h ago");
  });

  it("renders 2 hours ago as 2 h ago", () => {
    expect(formatRelativeTime(nowSec - 2 * 60 * 60, NOW)).toBe("2 h ago");
  });

  it("renders 23 hours ago as 23 h ago", () => {
    expect(formatRelativeTime(nowSec - 23 * 60 * 60, NOW)).toBe("23 h ago");
  });

  it("renders exactly 24 hours ago as a UTC calendar date", () => {
    expect(formatRelativeTime(nowSec - 24 * 60 * 60, NOW)).toBe("2026-09-17");
  });

  it("renders several days ago as a UTC calendar date", () => {
    expect(formatRelativeTime(nowSec - 5 * 24 * 60 * 60, NOW)).toBe("2026-09-13");
  });
});
