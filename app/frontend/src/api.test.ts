import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  ApiUnreachableError,
  createClinicalJob,
  getClinicalJob,
  getClinicalJobMask,
  getClinicalJobVolume,
} from "./api";

/**
 * Minimal mocked-`fetch` coverage for the clinical-job client functions,
 * following the same "what should the caller DO about it" framing
 * `lib/errors.test.ts` already uses for `responseError` - these go one layer
 * up, to the functions built on top of it (and on `fetchReport`'s
 * detail-reading pattern, for `createClinicalJob`).
 */
describe("createClinicalJob", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts multipart form data and returns the created job on success", async () => {
    const job = {
      job_id: "abc123",
      state: "queued",
      stage: "queued",
      progress: 0,
      case_id: "abc123",
      error: null,
      ingest_result: null,
      input_qc_pre: null,
      input_qc_post: null,
      preprocess_warnings: null,
      gatekeeper_decision: null,
      created_at: 0,
      updated_at: 0,
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 202,
      statusText: "Accepted",
      json: async () => job,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await createClinicalJob(new Blob(["fake zip bytes"]));
    expect(result).toEqual(job);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/upload");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    // The browser must set its own multipart boundary - a manually-set
    // Content-Type here would omit it and the server could not parse the body.
    expect(init.headers).toBeUndefined();
  });

  it("reads the detail field off a 400 response and throws ApiError with it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: async () => ({ detail: "create_clinical_job: dicom_zip is empty" }),
      }),
    );

    await expect(createClinicalJob(new Blob([]))).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      message: "create_clinical_job: dicom_zip is empty",
    });
  });

  it("falls back to the generic status message when the error body has no detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => {
          throw new SyntaxError("not json");
        },
      }),
    );

    await expect(createClinicalJob(new Blob(["x"]))).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "500 Internal Server Error on /clinical/upload",
    });
  });

  it("treats a 502 from the dev proxy as unreachable, not a normal API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: "Bad Gateway" }),
    );

    await expect(createClinicalJob(new Blob(["x"]))).rejects.toBeInstanceOf(ApiUnreachableError);
  });
});

describe("getClinicalJob", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("throws ApiError with status 404 for an unknown job id", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const err = await getClinicalJob("nope").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
  });
});

describe("getClinicalJobVolume / getClinicalJobMask", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("surfaces a 409 (job not done yet) as an ApiError, same as any other binary route", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        statusText: "Conflict",
      }),
    );

    const err = await getClinicalJobVolume("job1", "t1ce", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
  });

  it("getClinicalJobMask also surfaces a 409 as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        statusText: "Conflict",
      }),
    );

    const err = await getClinicalJobMask("job1", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
  });

  it("reads the shape off X-Volume-Shape and the bytes off the body on success", async () => {
    const headers = new Map([["X-Volume-Shape", "2,3,4"]]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: "OK",
        headers: { get: (k: string) => headers.get(k) ?? null },
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      }),
    );

    const buf = await getClinicalJobVolume("job1", "t1ce", [1, 1, 1]);
    expect(buf.shape).toEqual([2, 3, 4]);
    expect(Array.from(buf.data)).toEqual([1, 2, 3]);
  });
});
