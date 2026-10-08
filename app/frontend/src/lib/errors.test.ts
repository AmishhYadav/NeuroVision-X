import { describe, expect, it } from "vitest";

import { ApiError, ApiUnreachableError, responseError } from "../api";

/**
 * The distinction these tests protect is "what should the user DO about it",
 * not "what did HTTP say". A dead backend behind the Vite dev proxy arrives as
 * an HTTP 502 rather than as a failed fetch, and showing "502 Bad Gateway on
 * /health" tells a user nothing; "start uvicorn" tells them everything. That
 * exact bug shipped once and was only caught by loading the page with the
 * backend stopped.
 */
describe("responseError", () => {
  const res = (status: number, statusText = "") => ({ status, statusText });

  it("treats a 502 from the dev proxy as the API being unreachable", () => {
    expect(responseError(res(502, "Bad Gateway"), "/health")).toBeInstanceOf(ApiUnreachableError);
  });

  it("treats 503 and 504 the same way", () => {
    expect(responseError(res(503, "Service Unavailable"), "/health")).toBeInstanceOf(
      ApiUnreachableError,
    );
    expect(responseError(res(504, "Gateway Timeout"), "/health")).toBeInstanceOf(
      ApiUnreachableError,
    );
  });

  it("keeps a 404 as a normal API error, since the server DID answer", () => {
    const err = responseError(res(404, "Not Found"), "/cases/NOPE");
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
  });

  it("keeps a 500 as a normal API error", () => {
    // A 500 from our own API is a real server-side fault with a message worth
    // showing verbatim -- e.g. the geometry guard firing because a prediction
    // and its meta.json came from different preprocessing runs. Collapsing it
    // into "unreachable" would hide that.
    const err = responseError(res(500, "Internal Server Error"), "/cases/X/mask/prediction");
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
  });

  it("puts the status and path in the message so a report is actionable", () => {
    expect(responseError(res(404, "Not Found"), "/cases/NOPE").message).toContain("404");
    expect(responseError(res(404, "Not Found"), "/cases/NOPE").message).toContain("/cases/NOPE");
  });
});
