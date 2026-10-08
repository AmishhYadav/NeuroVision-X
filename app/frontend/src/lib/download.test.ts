import { describe, expect, it } from "vitest";

import { filenameFromContentDisposition } from "./download";

describe("filenameFromContentDisposition", () => {
  it("reads a quoted filename", () => {
    expect(
      filenameFromContentDisposition('attachment; filename="neurovision-abc123.zip"', "fallback.zip"),
    ).toBe("neurovision-abc123.zip");
  });

  it("reads a bare (unquoted) filename", () => {
    expect(
      filenameFromContentDisposition("attachment; filename=neurovision-abc123.zip", "fallback.zip"),
    ).toBe("neurovision-abc123.zip");
  });

  it("falls back when the header is null", () => {
    expect(filenameFromContentDisposition(null, "fallback.zip")).toBe("fallback.zip");
  });

  it("falls back when the header has no filename parameter", () => {
    expect(filenameFromContentDisposition("attachment", "fallback.zip")).toBe("fallback.zip");
  });
});
