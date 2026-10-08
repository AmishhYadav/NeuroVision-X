import { describe, expect, it } from "vitest";

import { ApiError, ApiUnreachableError } from "../api";
import { settleSupplementary } from "./supplementaryFetch";

/**
 * The bug this guards against: `/conformal-band/WT` 500'd on the first real
 * done clinical job, and because the hook awaited every supplementary fetch
 * in one `Promise.all`, that single rejection blanked the entire viewer -
 * slices, twin and report included - even though the four volumes, the mask,
 * entropy, Grad-CAM, geometry and the report had all loaded fine. A
 * supplementary artifact must fail on its own, not take the study with it.
 */
describe("settleSupplementary", () => {
  it("resolves 404 to null silently - an expected absence, not a warning", async () => {
    const warnings: string[] = [];
    const result = await settleSupplementary(
      "Grad-CAM (WT)",
      Promise.reject(new ApiError(404, "404 Not Found on /clinical/jobs/x/gradcam/WT")),
      warnings,
    );
    expect(result).toBeNull();
    expect(warnings).toEqual([]);
  });

  it("resolves a non-404 ApiError to null and appends a warning line", async () => {
    const warnings: string[] = [];
    const result = await settleSupplementary(
      "Conformal band (WT)",
      Promise.reject(
        new ApiError(500, "500 Internal Server Error on /clinical/jobs/x/conformal-band/WT"),
      ),
      warnings,
    );
    expect(result).toBeNull();
    expect(warnings).toEqual([
      "Conformal band (WT) unavailable: 500 Internal Server Error on /clinical/jobs/x/conformal-band/WT",
    ]);
  });

  it("resolves any other Error to null and appends a warning too", async () => {
    const warnings: string[] = [];
    const result = await settleSupplementary(
      "Geometry",
      Promise.reject(new TypeError("unexpected token < in JSON")),
      warnings,
    );
    expect(result).toBeNull();
    expect(warnings).toEqual(["Geometry unavailable: unexpected token < in JSON"]);
  });

  it("rethrows ApiUnreachableError - nothing else will work if the API is gone", async () => {
    const warnings: string[] = [];
    await expect(
      settleSupplementary("Uncertainty", Promise.reject(new ApiUnreachableError()), warnings),
    ).rejects.toBeInstanceOf(ApiUnreachableError);
    expect(warnings).toEqual([]);
  });

  it("rethrows AbortError - a cancelled switch, not a failed fetch", async () => {
    const warnings: string[] = [];
    const abortError = new DOMException("The operation was aborted.", "AbortError");
    await expect(
      settleSupplementary("Uncertainty", Promise.reject(abortError), warnings),
    ).rejects.toBe(abortError);
    expect(warnings).toEqual([]);
  });

  it("resolves the value through untouched on success", async () => {
    const warnings: string[] = [];
    const result = await settleSupplementary("Uncertainty", Promise.resolve(42), warnings);
    expect(result).toBe(42);
    expect(warnings).toEqual([]);
  });
});
