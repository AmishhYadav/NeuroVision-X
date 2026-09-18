import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  ApiUnreachableError,
  createClinicalJob,
  fetchClinicalReport,
  getAtlasStructures,
  getCaseAtlas,
  getClinicalJob,
  getClinicalJobAtlas,
  getClinicalJobConformalBand,
  getClinicalJobGeometry,
  getClinicalJobGradcam,
  getClinicalJobMask,
  getClinicalJobUncertainty,
  getClinicalJobVolume,
  getClinicalPathology,
  listClinicalJobs,
  putClinicalPathology,
} from "./api";
import type { AtlasStructuresResponse, ReportResponse } from "./api";

// A minimal but STRUCTURALLY COMPLETE report, satisfying validateReport's
// required-field guard - copied from `lib/report.test.ts`'s `makeReport`
// base object (not exported from there, and re-deriving an "invalid" fixture
// by hand risks asserting the wrong thing about validateReport rather than
// about fetchClinicalReport, which is what this file actually tests).
const VALID_REPORT: ReportResponse = {
  report_version: 1,
  case_id: "BraTS2021_00002",
  generated_utc: "2026-08-18T15:08:03.013692+00:00",
  disclaimer:
    "This report is a research and educational decision-support artifact. It is not a " +
    "diagnostic tool.",
  not_claimed: [
    ["cell type", "MRI resolves millimetre-scale tissue, not individual cells."],
    ["WHO grade", "WHO CNS5 grading needs histology and molecular markers."],
  ],
  burden: {
    volumes: { vol_ET_mm3: 23651.0, vol_WT_mm3: 190594.0 },
    fractions: { frac_enhancing_of_wt: 0.1241, ratio_edema_to_core: 4.4613 },
    shape: { sphericity_ET: 0.3116, surface_area_ET_mm2: 12786.996 },
    multifocality: { n_components_ET: 1, largest_component_frac_ET: 0.9999 },
    laterality: { dominant_side_ET: "left", frac_left_ET: 0.9984 },
    centroid: { centroid_i_ET: 87.1017 },
    other: {},
  },
  anatomy: {
    atlas: { name: "tzo116plus", version: "2.0" },
    caveat:
      "This atlas describes healthy-brain anatomy. A tumour physically displaces the " +
      "tissue around it.",
    coverage_line: "23 of 122 structures classified eloquent, 99 unclassified.",
    region: "WT",
    structures: [
      {
        region: "WT",
        structure: "Caudate_L",
        laterality: "L",
        lobe: "deep",
        eloquence: "eloquent",
        matched_term: "basal ganglia",
        n_voxels: 4864,
        volume_mm3: 4864.0,
        frac_of_tumour: 0.02552,
        frac_of_structure: 0.98501,
      },
    ],
    n_structures_involved: 46,
    frac_unlabelled: 0.3097,
  },
  eloquence: {
    classification: "Sawaya eloquence grading",
    citation: "Sawaya R, et al. Neurosurgery. 1998.",
    evidence: "Eloquent locations in the Sawaya study are the motor/sensory cortices.",
    source_owns_claim:
      "This eloquence classification is a lookup into a named, published source.",
    involved: [
      { structure: "Caudate_L", laterality: "L", frac_of_tumour: 0.02552, frac_of_structure: 0.98501 },
    ],
    distance_mm: 0.0,
    near_eloquent_threshold_mm: 10.0,
    near_eloquent: true,
    coverage_gaps: ["internal capsule", "dentate nucleus"],
  },
  provenance: {
    atlas_name: "tzo116plus",
    atlas_version: "2.0",
    atlas_source: "NITRC group_id=214",
    atlas_licence: "CC-BY-SA",
    knowledge_versions: { eloquence_map: 1, aal_lobes: 1 },
    segmentation_source: "label",
    segmentation_dir: "/data/preprocessed/brats",
    code_revision: "b918b35-dirty",
    generated_utc: "2026-08-18T15:08:03.013692+00:00",
  },
};

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

describe("listClinicalJobs", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits /api/clinical/jobs and returns the list on success", async () => {
    const jobs = [
      {
        job_id: "abc123",
        state: "done",
        stage: "done",
        progress: 1,
        case_id: "abc123",
        error: null,
        ingest_result: null,
        input_qc_pre: null,
        input_qc_post: null,
        preprocess_warnings: null,
        gatekeeper_decision: null,
        created_at: 0,
        updated_at: 0,
      },
    ];
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({ jobs }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await listClinicalJobs();
    expect(result).toEqual({ jobs });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/clinical/jobs");
  });

  it("treats a 502 from the dev proxy as unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: "Bad Gateway" }),
    );

    await expect(listClinicalJobs()).rejects.toBeInstanceOf(ApiUnreachableError);
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

describe("getClinicalJobUncertainty", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads the shape and kind headers and the bytes off the body on success", async () => {
    const headers = new Map([
      ["X-Volume-Shape", "2,3,4"],
      ["X-Uncertainty-Kind", "predictive-entropy-single-pass"],
    ]);
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

    const buf = await getClinicalJobUncertainty("job1", [1, 1, 1]);
    expect(buf).not.toBeNull();
    expect(buf?.shape).toEqual([2, 3, 4]);
    expect(buf?.kind).toBe("predictive-entropy-single-pass");
    expect(Array.from(buf?.data ?? [])).toEqual([1, 2, 3]);
  });

  // Different from getClinicalJobVolume/getClinicalJobMask, which throw an
  // ApiError on any non-ok status (see the 409 tests above): a 404 here means
  // "no cached logits for this job", a normal outcome, not an error - same
  // as the demo viewer's own getUncertainty.
  it("resolves to null on 404 rather than throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const result = await getClinicalJobUncertainty("job1", [1, 1, 1]);
    expect(result).toBeNull();
  });

  it("still surfaces a non-404 error status (500) as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      }),
    );

    const err = await getClinicalJobUncertainty("job1", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
  });
});

describe("getClinicalJobConformalBand", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads the shape and kind headers and the bytes off the body on success", async () => {
    const headers = new Map([
      ["X-Volume-Shape", "2,3,4"],
      ["X-Uncertainty-Kind", "conformal-band"],
    ]);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: (k: string) => headers.get(k) ?? null },
      arrayBuffer: async () => new Uint8Array([0, 128, 255]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    const buf = await getClinicalJobConformalBand("job1", "WT", [1, 1, 1]);
    expect(buf).not.toBeNull();
    expect(buf?.shape).toEqual([2, 3, 4]);
    expect(buf?.kind).toBe("conformal-band");
    expect(Array.from(buf?.data ?? [])).toEqual([0, 128, 255]);

    // The region segment must land in the URL exactly, distinguishing WT
    // from TC - a wrong path here would silently fetch the wrong region's
    // band with no other symptom.
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/conformal-band/WT");
  });

  it("includes the region segment for TC too", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: () => null },
      arrayBuffer: async () => new Uint8Array([]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    await getClinicalJobConformalBand("job1", "TC", [1, 1, 1]);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/conformal-band/TC");
  });

  // Same convention as getClinicalJobUncertainty: a 404 means no fitted
  // threshold is available for this region yet, a normal outcome, not an
  // error.
  it("resolves to null on 404 rather than throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const result = await getClinicalJobConformalBand("job1", "WT", [1, 1, 1]);
    expect(result).toBeNull();
  });

  it("still surfaces a non-404 error status (500) as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      }),
    );

    const err = await getClinicalJobConformalBand("job1", "TC", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
  });
});

describe("getClinicalJobGradcam", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads the shape and kind headers and the bytes off the body on success", async () => {
    const headers = new Map([
      ["X-Volume-Shape", "2,3,4"],
      ["X-Uncertainty-Kind", "gradcam"],
    ]);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: (k: string) => headers.get(k) ?? null },
      arrayBuffer: async () => new Uint8Array([0, 64, 255]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    const buf = await getClinicalJobGradcam("job1", "WT", [1, 1, 1]);
    expect(buf).not.toBeNull();
    expect(buf?.shape).toEqual([2, 3, 4]);
    expect(buf?.kind).toBe("gradcam");
    expect(Array.from(buf?.data ?? [])).toEqual([0, 64, 255]);

    // The region segment must land in the URL exactly, distinguishing WT
    // from TC - a wrong path here would silently fetch the wrong region's
    // heatmap with no other symptom.
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/gradcam/WT");
  });

  it("includes the region segment for TC too", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: () => null },
      arrayBuffer: async () => new Uint8Array([]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    await getClinicalJobGradcam("job1", "TC", [1, 1, 1]);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/gradcam/TC");
  });

  // Same convention as getClinicalJobConformalBand: a 404 means either the
  // job predates this feature or that region's Grad-CAM computation failed
  // and was skipped, a normal outcome, not an error.
  it("resolves to null on 404 rather than throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const result = await getClinicalJobGradcam("job1", "WT", [1, 1, 1]);
    expect(result).toBeNull();
  });

  it("still surfaces a non-404 error status (500) as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
      }),
    );

    const err = await getClinicalJobGradcam("job1", "TC", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
  });
});

describe("getClinicalJobGeometry", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits the geometry route and returns the parsed CaseMeta on success", async () => {
    const meta = {
      case_id: "job1",
      shape: [128, 128, 96],
      original_shape: [240, 240, 155],
      bbox: [10, 20, 30, 100, 110, 90],
      spacing: [1.0, 1.0, 1.0],
      has_label: false,
      has_prediction: true,
      has_logits: true,
      planes: { sagittal: 128, coronal: 128, axial: 96 },
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => meta,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getClinicalJobGeometry("job1");
    expect(result).toEqual(meta);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/geometry");
  });

  it("throws ApiError with status 409 when the job isn't done yet", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 409, statusText: "Conflict" }),
    );

    const err = await getClinicalJobGeometry("job1").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
  });

  it("throws ApiError with status 404 for an unknown job id", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const err = await getClinicalJobGeometry("nope").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(404);
  });
});

describe("fetchClinicalReport", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits the clinical report route and returns the validated report on success", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => VALID_REPORT,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await fetchClinicalReport("job1");
    expect(result).toEqual(VALID_REPORT);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/report");
  });

  it("reads the detail field off a 404 response and throws ApiError with it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: async () => ({ detail: "no report for job1" }),
      }),
    );

    await expect(fetchClinicalReport("job1")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "no report for job1",
    });
  });

  it("treats a 502 from the dev proxy as unreachable, not a normal API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: "Bad Gateway" }),
    );

    await expect(fetchClinicalReport("job1")).rejects.toBeInstanceOf(ApiUnreachableError);
  });

  it("throws whatever validateReport throws when the body is structurally invalid", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        statusText: "OK",
        // Missing every required field validateReport checks for.
        json: async () => ({}),
      }),
    );

    await expect(fetchClinicalReport("job1")).rejects.toThrow();
  });
});

describe("getAtlasStructures", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits /api/atlas/structures and returns the parsed table on success", async () => {
    const body: AtlasStructuresResponse = {
      atlas: "tzo116plus",
      version: "2.0",
      n_structures: 1,
      structures: [
        {
          index: 5,
          name: "Caudate_L",
          laterality: "L",
          lobe: "deep",
          eloquence: "eloquent",
          matched_term: "basal ganglia",
        },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => body,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getAtlasStructures();
    expect(result).toEqual(body);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/atlas/structures");
  });

  it("treats a 502 from the dev proxy as unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: "Bad Gateway" }),
    );

    await expect(getAtlasStructures()).rejects.toBeInstanceOf(ApiUnreachableError);
  });
});

describe("getClinicalJobAtlas / getCaseAtlas", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("getClinicalJobAtlas reads the shape and kind headers and the bytes off the body on success", async () => {
    const headers = new Map([
      ["X-Volume-Shape", "2,3,4"],
      ["X-Uncertainty-Kind", "atlas-structure-index"],
    ]);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: (k: string) => headers.get(k) ?? null },
      arrayBuffer: async () => new Uint8Array([0, 5, 12]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    const buf = await getClinicalJobAtlas("job1", [1, 1, 1]);
    expect(buf).not.toBeNull();
    expect(buf?.shape).toEqual([2, 3, 4]);
    expect(buf?.kind).toBe("atlas-structure-index");
    expect(Array.from(buf?.data ?? [])).toEqual([0, 5, 12]);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/atlas");
  });

  // Same convention as getClinicalJobGradcam: a 404 means no saved case
  // meta for this job, a normal outcome, not an error.
  it("getClinicalJobAtlas resolves to null on 404 rather than throwing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const result = await getClinicalJobAtlas("job1", [1, 1, 1]);
    expect(result).toBeNull();
  });

  // A job that hasn't finished yet 409s, same as every other clinical
  // volume route - the gate is "the job is done", not "the atlas exists".
  it("getClinicalJobAtlas still surfaces a non-404 error status (409) as an ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 409, statusText: "Conflict" }),
    );

    const err = await getClinicalJobAtlas("job1", [1, 1, 1]).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(409);
  });

  it("getCaseAtlas hits the demo case atlas route and resolves to null on 404", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 404, statusText: "Not Found" }),
    );

    const result = await getCaseAtlas("case1", [1, 1, 1]);
    expect(result).toBeNull();
  });

  it("getCaseAtlas reads the shape and kind headers on success", async () => {
    const headers = new Map([
      ["X-Volume-Shape", "5,6,7"],
      ["X-Uncertainty-Kind", "atlas-structure-index"],
    ]);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      headers: { get: (k: string) => headers.get(k) ?? null },
      arrayBuffer: async () => new Uint8Array([1]).buffer,
    });
    vi.stubGlobal("fetch", fetchMock);

    const buf = await getCaseAtlas("case1", [1, 1, 1]);
    expect(buf?.shape).toEqual([5, 6, 7]);
    expect(buf?.kind).toBe("atlas-structure-index");

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/cases/case1/atlas");
  });
});

// T5.6: the pathology PUT/GET pair `MolecularPanel` is built on. Same
// mocked-fetch, "what should the caller DO about it" framing as
// `createClinicalJob`'s describe block above - `putClinicalPathology` reuses
// that function's detail-reading pattern on a non-2xx body, so the 400 case
// is the one worth asserting in detail here.
describe("putClinicalPathology", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends a PUT with a JSON body and returns the parsed response on success", async () => {
    const body = {
      job_id: "job1",
      pathology: { IDH: "Wildtype" },
      cns5: { name: null, requires: ["histology"], source: null },
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => body,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await putClinicalPathology("job1", { IDH: "Wildtype" });
    expect(result).toEqual(body);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/pathology");
    expect(init.method).toBe("PUT");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(init.body).toBe(JSON.stringify({ IDH: "Wildtype" }));
  });

  it("reads the detail field off a 400 response and throws ApiError with it", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: async () => ({
          detail: "validate_entered: 'IDH' was set to 'Positive', which is not in its allowed values ['Mutant', 'Wildtype', 'Not tested', 'Not entered'].",
        }),
      }),
    );

    const err = await putClinicalPathology("job1", { IDH: "Positive" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(400);
    expect((err as ApiError).message).toContain("Mutant");
    expect((err as ApiError).message).toContain("Wildtype");
  });

  it("treats a 502 from the dev proxy as unreachable, not a normal API error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 502, statusText: "Bad Gateway" }),
    );

    await expect(putClinicalPathology("job1", { IDH: "Wildtype" })).rejects.toBeInstanceOf(
      ApiUnreachableError,
    );
  });
});

describe("getClinicalPathology", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("hits the pathology route and returns the currently entered values on success", async () => {
    const body = { job_id: "job1", pathology: { IDH: "Mutant" } };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => body,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await getClinicalPathology("job1");
    expect(result).toEqual(body);

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/clinical/jobs/job1/pathology");
  });
});
