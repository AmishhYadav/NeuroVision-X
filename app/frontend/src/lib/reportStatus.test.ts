// Tests for classifyReportError's four branches (see reportStatus.ts) plus
// the non-Error-thrown edge case (a rejection that isn't even an Error, e.g.
// a raw string thrown across a boundary that doesn't normalize it).

import { describe, expect, it } from "vitest";
import { ApiError, ApiUnreachableError } from "../api";
import { classifyReportError } from "./reportStatus";

describe("classifyReportError", () => {
  it("maps ApiUnreachableError to unreachable with no message", () => {
    expect(classifyReportError(new ApiUnreachableError())).toEqual({
      status: "unreachable",
      message: null,
    });
  });

  it("maps a 404 ApiError to not_found, carrying the server's message", () => {
    const err = new ApiError(404, "report not found for case X");
    expect(classifyReportError(err)).toEqual({
      status: "not_found",
      message: "report not found for case X",
    });
  });

  it("maps any other ApiError status to server_error, carrying the server's message", () => {
    const err = new ApiError(500, "internal error building report");
    expect(classifyReportError(err)).toEqual({
      status: "server_error",
      message: "internal error building report",
    });
  });

  it("maps a validateReport-style Error to invalid, carrying its message", () => {
    const err = new Error("report.burden is required.");
    expect(classifyReportError(err)).toEqual({
      status: "invalid",
      message: "report.burden is required.",
    });
  });

  it("maps a non-Error thrown value to invalid with the generic fallback message", () => {
    expect(classifyReportError("some string thrown, not an Error")).toEqual({
      status: "invalid",
      message: "Failed to load report.",
    });
  });
});
