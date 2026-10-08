import { describe, expect, it } from "vitest";
import type { HealthResponse } from "../api";
import { NAV_ITEMS, apiStatus, isPlainLeftClick, navKeyForPath } from "./shellStatus";

function health(over: Partial<HealthResponse> = {}): HealthResponse {
  return {
    status: "ok",
    experiment: "exp_a",
    eval_dir: "e",
    prep_dir: "p",
    checkpoint_present: true,
    case_count: 5,
    has_metrics: true,
    report_dir: "r",
    has_reports: true,
    ...over,
  };
}

describe("navKeyForPath", () => {
  it("maps exact routes", () => {
    expect(navKeyForPath("/")).toBe("home");
    expect(navKeyForPath("/clinical")).toBe("clinical");
    expect(navKeyForPath("/app")).toBe("viewer");
    expect(navKeyForPath("/report/abc")).toBe("report");
  });
  it("does not match substrings", () => {
    expect(navKeyForPath("/application")).toBe("home");
    expect(navKeyForPath("/clinicalx")).toBe("home");
    expect(navKeyForPath("/report/app-case")).toBe("report");
    expect(navKeyForPath("/reports")).toBe("home");
  });
  it("falls back to home", () => {
    expect(navKeyForPath("/nope")).toBe("home");
  });
});

describe("NAV_ITEMS", () => {
  it("has three items and none for report", () => {
    expect(NAV_ITEMS.map((i) => i.href)).toEqual(["/", "/clinical", "/app"]);
  });
});

describe("apiStatus", () => {
  it("connecting while unknown", () => {
    expect(apiStatus(null, null)).toEqual({ tone: "connecting", label: "Connecting…", detail: null });
  });
  it("offline when unreachable", () => {
    expect(apiStatus(null, false)).toEqual({ tone: "offline", label: "API offline", detail: null });
    expect(apiStatus(health(), false).tone).toBe("offline");
  });
  it("offline if reachable but no health body", () => {
    expect(apiStatus(null, true).tone).toBe("offline");
  });
  it("degraded without checkpoint", () => {
    expect(apiStatus(health({ checkpoint_present: false }), true)).toEqual({
      tone: "degraded",
      label: "API online · no checkpoint",
      detail: "exp_a",
    });
  });
  it("online with plural cases", () => {
    expect(apiStatus(health(), true)).toEqual({
      tone: "online",
      label: "API online",
      detail: "exp_a · 5 cases",
    });
  });
  it("singular case", () => {
    expect(apiStatus(health({ case_count: 1 }), true).detail).toBe("exp_a · 1 case");
  });
  it("zero cases is plural", () => {
    expect(apiStatus(health({ case_count: 0 }), true).detail).toBe("exp_a · 0 cases");
  });
});

describe("isPlainLeftClick", () => {
  const base = { button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false };
  it("accepts a plain left click", () => expect(isPlainLeftClick(base)).toBe(true));
  it("rejects other buttons", () => {
    expect(isPlainLeftClick({ ...base, button: 1 })).toBe(false);
    expect(isPlainLeftClick({ ...base, button: 2 })).toBe(false);
  });
  it("rejects each modifier", () => {
    for (const k of ["metaKey", "ctrlKey", "shiftKey", "altKey"] as const) {
      expect(isPlainLeftClick({ ...base, [k]: true })).toBe(false);
    }
  });
});
