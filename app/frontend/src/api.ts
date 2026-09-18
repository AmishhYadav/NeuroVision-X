// Typed client for the NeuroVision-X demo backend.
//
// Every function here mirrors one endpoint of the API contract exactly - no
// reshaping, no re-sorting, no invented fields. Binary endpoints return the
// raw bytes plus the shape read from the `X-Volume-Shape` header, falling
// back to a caller-supplied shape when the header is missing (a proxy can
// strip custom headers).

// `validateReport` is a runtime import; `lib/report.ts` only imports
// `ReportResponse` (and friends) from this file as a `type`, which is erased
// at compile time - so this is not a runtime circular dependency.
import { validateReport } from "./lib/report";

export type Modality = "t1" | "t1ce" | "t2" | "flair";
export type MaskSource = "prediction" | "label";
export type Plane = "sagittal" | "coronal" | "axial";
export type RegionKey = "ET" | "TC" | "WT";

export interface HealthResponse {
  status: string;
  experiment: string;
  eval_dir: string;
  prep_dir: string;
  checkpoint_present: boolean;
  case_count: number;
  has_metrics: boolean;
  report_dir: string;
  has_reports: boolean;
}

export interface CaseSummary {
  case_id: string;
  dice_mean: number | null;
  dice: Record<RegionKey, number> | null;
  has_label: boolean;
  has_logits: boolean;
  has_report: boolean;
}

export interface CasesResponse {
  cases: CaseSummary[];
}

export interface CaseMeta {
  case_id: string;
  shape: [number, number, number];
  original_shape: [number, number, number];
  bbox: number[];
  spacing: [number, number, number];
  has_label: boolean;
  has_prediction: boolean;
  has_logits: boolean;
  planes: { sagittal: number; coronal: number; axial: number };
}

export interface CaseMetrics {
  dice: Record<RegionKey, number | null>;
  hd95: Record<RegionKey, number | null>;
  dice_mean: number | null;
  gt_empty: Record<RegionKey, boolean | null>;
}

export interface RegionStat {
  voxels: number;
  ml: number;
}

export type RegionStats = Record<RegionKey, RegionStat>;

export interface CaseRegions {
  prediction: RegionStats;
  label: RegionStats | null;
}

export interface CaseDetail {
  meta: CaseMeta;
  metrics: CaseMetrics | null;
  regions: CaseRegions;
  has_report: boolean;
}

export interface VolumeBuffer {
  data: Uint8Array;
  shape: [number, number, number];
}

/** The only uncertainty kind the client is allowed to present as such - see UncertaintyBuffer. */
export const PREDICTIVE_ENTROPY_SINGLE_PASS = "predictive-entropy-single-pass";

/** The conformal risk-control band - see `getClinicalJobConformalBand`. */
export const CONFORMAL_BAND = "conformal-band";

/** Seg-Grad-CAM explainability evidence - see `getClinicalJobGradcam`. */
export const GRADCAM = "gradcam";

/** The atlas structure-index volume - see `getClinicalJobAtlas` / `getCaseAtlas`. */
export const ATLAS_STRUCTURE_INDEX = "atlas-structure-index";

/**
 * `VolumeBuffer` plus the `X-Uncertainty-Kind` response header, verbatim.
 * The backend is CORS-exposing that header on purpose so the client cannot
 * mislabel what this quantity is (e.g. present a single-pass entropy map as
 * epistemic/MC-dropout uncertainty). `kind` is the raw header value, or null
 * if the header was absent - callers must not assume a default.
 */
export interface UncertaintyBuffer extends VolumeBuffer {
  kind: string | null;
}

/**
 * One row of the atlas's structure table - name, laterality, lobe,
 * eloquence, per index. `index` is the 1-based value that structure carries
 * in the `uint8` volume served by `getClinicalJobAtlas` / `getCaseAtlas`
 * (0 is background, never a real structure) - see `getAtlasStructures`.
 */
export interface AtlasStructureRow {
  index: number;
  name: string;
  laterality: string | null;
  lobe: string | null;
  eloquence: string | null;
  matched_term: string | null;
}

/** `GET /api/atlas/structures` - mirrors `app.backend.api.get_atlas_structures` field-for-field. */
export interface AtlasStructuresResponse {
  atlas: string;
  version: string;
  n_structures: number;
  structures: AtlasStructureRow[];
}

/**
 * Wire-identical to `UncertaintyBuffer` (same `{data, shape, kind}` shape) -
 * a distinct name only so callers reading atlas code don't have to reason
 * about entropy/conformal/Grad-CAM semantics. `kind` must equal
 * `ATLAS_STRUCTURE_INDEX`; a caller must check that from the header before
 * treating the buffer as atlas data, never assume it (lesson: label layers
 * only from the header, never assume what a volume is).
 */
export type AtlasBuffer = UncertaintyBuffer;

export interface ProfilePlaneData {
  n: number;
  tumor: number[];
  error: number[] | null;
  entropy: number[] | null;
}

export interface CaseProfile {
  case_id: string;
  planes: Record<Plane, ProfilePlaneData>;
}

// --------------------------------------------------------------------- //
// Phase 4 structured report - mirrors neurovision.reporting.report.build_report
// field-for-field. See src/lib/report.ts for formatting/validation logic;
// this file only names the shape the server actually sends.
// --------------------------------------------------------------------- //

/** A flat burden sub-block: raw measurement name -> value, exactly as `burden_profile` emits it. */
export type BurdenValue = number | string | boolean | null;
export type BurdenBlock = Record<string, BurdenValue>;

export interface ReportBurden {
  volumes: BurdenBlock;
  fractions: BurdenBlock;
  shape: BurdenBlock;
  multifocality: BurdenBlock;
  laterality: BurdenBlock;
  centroid: BurdenBlock;
  other: BurdenBlock;
}

export interface AnatomyStructureRow {
  region: string | null;
  structure: string;
  laterality: string | null;
  lobe: string | null;
  eloquence: string | null;
  matched_term: string | null;
  n_voxels: number | null;
  volume_mm3: number | null;
  frac_of_tumour: number | null;
  frac_of_structure: number | null;
}

export interface ReportAnatomy {
  atlas: { name: string; version: string };
  caveat: string;
  coverage_line: string;
  region: string | null;
  /** Already sorted by frac_of_structure descending and truncated to top_n server-side - render in order received. */
  structures: AnatomyStructureRow[];
  n_structures_involved: number | null;
  frac_unlabelled: number | null;
}

export interface EloquenceInvolvedRow {
  structure: string;
  laterality: string | null;
  frac_of_tumour: number | null;
  frac_of_structure: number | null;
}

export interface ReportEloquence {
  classification: string;
  citation: string;
  evidence: string;
  source_owns_claim: string;
  involved: EloquenceInvolvedRow[];
  distance_mm: number | null;
  near_eloquent_threshold_mm: number;
  near_eloquent: boolean;
  coverage_gaps: string[];
}

/**
 * The optional "Shape and extent (geometric)" block - mirrors
 * `neurovision.reporting.report._build_geometry_block` field-for-field.
 * Regrouped from a flat `shape_profile` dict into four labelled sub-blocks
 * server-side; `caveat` is a required field of the block itself (not README
 * text), same convention as `anatomy.caveat`.
 *
 * `geometry` is OPTIONAL on `ReportResponse` because this block was added in
 * T4.2, after the report schema already shipped: a report file written
 * before that change, and any batch report generated with the geometry flag
 * off, has no `"geometry"` key at all - `validateReport` must not require
 * it, and the panel must render correctly with `report.geometry` absent.
 */
export interface ReportGeometry {
  caveat: string;
  shape: BurdenBlock;
  extent: BurdenBlock;
  rim: BurdenBlock;
  other: BurdenBlock;
}

/** Optional atlas-overlap block (ventricles, deep white matter, tissue class, epicentre). Present in reports built by scripts/report.py from 2026-08-19 on; absent on older reports, so every reader must null-check it. */
export interface ReportInvolvement {
  caveat: string;
  not_vasari: string;
  lower_bound_notes: string[];
  groups: BurdenBlock; // keys like ventricle_overlap_mm3, ventricle_frac_of_tumour, ventricle_frac_of_group, ventricle_contact, deep_wm_* (same four)
  tissue: BurdenBlock; // cortical_frac_of_tumour, white_matter_frac_of_tumour, csf_frac_of_tumour, outside_tissue_frac_of_tumour
  epicentre: BurdenBlock; // epicentre_structure, epicentre_exact, epicentre_distance_mm, epicentre_laterality, epicentre_side, epicentre_lobe
  other: BurdenBlock;
}

export interface ReportProvenance {
  atlas_name: string;
  atlas_version: string;
  atlas_source: string;
  atlas_licence: string;
  knowledge_versions: Record<string, number>;
  segmentation_source: "prediction" | "label";
  segmentation_dir: string | null;
  code_revision: string | null;
  generated_utc: string;
}

export interface ReportResponse {
  report_version: number;
  case_id: string;
  generated_utc: string;
  disclaimer: string;
  /** (what this artifact refuses to claim, why) pairs. */
  not_claimed: [string, string][];
  burden: ReportBurden;
  anatomy: ReportAnatomy;
  /** Present only when the server built this report with the geometry flag on - see `ReportGeometry`. */
  geometry?: ReportGeometry;
  /** Present only on reports built from 2026-08-19 on - see `ReportInvolvement`. */
  involvement?: ReportInvolvement;
  eloquence: ReportEloquence;
  provenance: ReportProvenance;
}

const API_BASE = "/api";

/** Thrown for a reachable-but-erroring response (4xx/5xx) with the status attached. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Thrown when the request never reached the server (network failure, refused connection). */
export class ApiUnreachableError extends Error {
  constructor(message = "No response from the API.") {
    super(message);
    this.name = "ApiUnreachableError";
  }
}

/**
 * Gateway statuses that mean the API itself is not running.
 *
 * In development the Vite dev server proxies `/api`, so a dead backend comes
 * back as an HTTP **502** rather than as a failed fetch -- the proxy answered,
 * the API did not. Treating that as an ordinary API error surfaces
 * "502 Bad Gateway on /health" to the user, when the actual problem is that
 * uvicorn was never started. These are classified as unreachable so the UI
 * can print the command that fixes it.
 */
const GATEWAY_DOWN = new Set([502, 503, 504]);

/**
 * Maps a non-ok response onto the error type that describes what to DO about
 * it. Exported only so the 502-is-unreachable rule can be tested directly;
 * it is not part of the client surface.
 */
export function responseError(res: Pick<Response, "status" | "statusText">, path: string): Error {
  if (GATEWAY_DOWN.has(res.status)) return new ApiUnreachableError();
  return new ApiError(res.status, `${res.status} ${res.statusText} on ${path}`);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    throw responseError(res, path);
  }
  return (await res.json()) as T;
}

async function getBinary(
  path: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    throw responseError(res, path);
  }
  const shapeHeader = res.headers.get("X-Volume-Shape");
  const shape: [number, number, number] = shapeHeader
    ? (shapeHeader.split(",").map((s) => parseInt(s.trim(), 10)) as [number, number, number])
    : fallbackShape;
  const buf = await res.arrayBuffer();
  return { data: new Uint8Array(buf), shape };
}

/**
 * Shared implementation behind every "optional, kinded binary" route -
 * uncertainty, the conformal band, Grad-CAM, and the atlas structure-index
 * volume. All four share one wire contract: a 404 is an expected absence (no
 * cached logits, no fitted threshold, a job predating the feature, no saved
 * atlas meta) and resolves to `null` rather than throwing; any other non-2xx
 * status is a real `ApiError`; on success the shape comes from
 * `X-Volume-Shape` (falling back to the caller-supplied shape when a proxy
 * strips the header) and `kind` comes verbatim from `X-Uncertainty-Kind` -
 * never assumed, so a caller can never mislabel what a volume actually is.
 *
 * Extracted because `getUncertainty`, `getClinicalJobUncertainty`,
 * `getClinicalJobConformalBand` and `getClinicalJobGradcam` were byte-for-
 * byte this same body, differing only in the path they built.
 */
async function getOptionalKindedBinary(
  path: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (res.status === 404) return null;
  if (!res.ok) {
    throw responseError(res, path);
  }
  const shapeHeader = res.headers.get("X-Volume-Shape");
  const shape: [number, number, number] = shapeHeader
    ? (shapeHeader.split(",").map((s) => parseInt(s.trim(), 10)) as [number, number, number])
    : fallbackShape;
  const kind = res.headers.get("X-Uncertainty-Kind");
  const buf = await res.arrayBuffer();
  return { data: new Uint8Array(buf), shape, kind };
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return getJson<HealthResponse>("/health", signal);
}

export function getCases(signal?: AbortSignal): Promise<CasesResponse> {
  return getJson<CasesResponse>("/cases", signal);
}

export function getCase(caseId: string, signal?: AbortSignal): Promise<CaseDetail> {
  return getJson<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`, signal);
}

export function getVolume(
  caseId: string,
  modality: Modality,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer> {
  return getBinary(
    `/cases/${encodeURIComponent(caseId)}/volume/${modality}`,
    fallbackShape,
    signal,
  );
}

export function getMask(
  caseId: string,
  source: MaskSource,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer> {
  return getBinary(`/cases/${encodeURIComponent(caseId)}/mask/${source}`, fallbackShape, signal);
}

/**
 * Returns null on 404 (no saved logits for this case) rather than throwing.
 * Reads `X-Uncertainty-Kind` itself (rather than going through `getBinary`,
 * which only surfaces `X-Volume-Shape`) so the label shown to the user is
 * always what the backend actually measured.
 */
export function getUncertainty(
  caseId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  return getOptionalKindedBinary(
    `/cases/${encodeURIComponent(caseId)}/uncertainty`,
    fallbackShape,
    signal,
  );
}

/** The atlas's structure table - name, laterality, lobe, eloquence, per index. */
export function getAtlasStructures(signal?: AbortSignal): Promise<AtlasStructuresResponse> {
  return getJson<AtlasStructuresResponse>("/atlas/structures", signal);
}

/**
 * This demo case's atlas structure-index volume, cropped to its own bbox -
 * same "optional, kinded binary" contract as `getUncertainty` (null on 404,
 * an `ApiError` on any other failure). `kind` must be checked against
 * `ATLAS_STRUCTURE_INDEX` by the caller before the buffer is treated as
 * atlas data.
 */
export function getCaseAtlas(
  caseId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<AtlasBuffer | null> {
  return getOptionalKindedBinary(`/cases/${encodeURIComponent(caseId)}/atlas`, fallbackShape, signal);
}

export function getProfile(caseId: string, signal?: AbortSignal): Promise<CaseProfile> {
  return getJson<CaseProfile>(`/cases/${encodeURIComponent(caseId)}/profile`, signal);
}

/**
 * Fetches a structured report from `path` and validates its shape. Shared by
 * `fetchReport` (the demo viewer's `/report/{case_id}`) and
 * `fetchClinicalReport` (a live clinical job's `/clinical/jobs/{id}/report`)
 * - both endpoints return the identical `ReportResponse` JSON shape, so the
 * only difference between the two public functions is the path.
 *
 * Deliberately does not reuse `getJson`: on a 404 or 500 the backend's body
 * carries a `detail` message worth showing verbatim - e.g. the provenance
 * guard's 500 names both the report's own segmentation directory and the one
 * this server is configured to display, which is exactly what a reader needs
 * to fix a misconfigured `NVX_REPORT_DIR`. `getJson` only ever builds a
 * generic "<status> <statusText> on <path>" message from the response line,
 * which would throw that detail away.
 */
async function fetchValidatedReport(path: string, signal?: AbortSignal): Promise<ReportResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    if (GATEWAY_DOWN.has(res.status)) throw new ApiUnreachableError();
    let detail: string | undefined;
    try {
      const body = (await res.json()) as unknown;
      if (body && typeof body === "object" && typeof (body as { detail?: unknown }).detail === "string") {
        detail = (body as { detail: string }).detail;
      }
    } catch {
      // Body wasn't JSON (or was empty) - fall through to the generic message.
    }
    throw new ApiError(res.status, detail ?? `${res.status} ${res.statusText} on ${path}`);
  }
  const raw = (await res.json()) as unknown;
  return validateReport(raw);
}

/** Fetches one case's Phase 4 structured report and validates its shape. See `fetchValidatedReport`. */
export async function fetchReport(caseId: string, signal?: AbortSignal): Promise<ReportResponse> {
  return fetchValidatedReport(`/report/${encodeURIComponent(caseId)}`, signal);
}

// --------------------------------------------------------------------- //
// Clinical pipeline (Milestone 4 Phase E) - a REAL DICOM study, uploaded
// live and run through ingest -> input QC -> clinical preprocessing ->
// segmentation -> the gatekeeper. Mirrors `app.backend.clinical_jobs
// .ClinicalJob` (a plain dataclass, serialised field-for-field via
// `dataclasses.asdict` in `app/backend/api.py`'s `_clinical_job_to_json`)
// and the nested report/decision shapes from
// `neurovision.inference.input_qc.InputQCReport.to_dict` and
// `neurovision.inference.gatekeeper.GateDecision.to_dict`. Every field name
// and nesting level below was checked against those three functions'
// current source, not assumed.
// --------------------------------------------------------------------- //

export type ClinicalJobState = "queued" | "running" | "done" | "refused" | "failed";

/** `neurovision.inference.input_qc.Severity` - ordered ok < warn < refuse. */
export type GateSeverity = "ok" | "warn" | "refuse";

/** `neurovision.inference.gatekeeper.Decision` - ordered proceed < proceed_with_caution < refuse. */
export type GateDecisionValue = "proceed" | "proceed_with_caution" | "refuse";

/** `neurovision.inference.input_qc.Finding.to_dict()`'s per-item shape. */
export interface InputQCFinding {
  check: string;
  severity: GateSeverity;
  message: string;
  detail: Record<string, unknown>;
}

/** `neurovision.inference.input_qc.InputQCReport.to_dict()`. */
export interface InputQCReportJson {
  verdict: GateSeverity;
  findings: InputQCFinding[];
}

/** `neurovision.inference.gatekeeper.SignalVerdict`, as serialised by `GateDecision.to_dict()`. */
export interface GatekeeperSignalVerdict {
  signal: string;
  decision: GateDecisionValue;
  available: boolean;
  enabled: boolean;
  message: string;
  detail: Record<string, unknown>;
}

/** `neurovision.inference.gatekeeper.GateDecision.to_dict()`. */
export interface GatekeeperDecisionJson {
  decision: GateDecisionValue;
  verdicts: GatekeeperSignalVerdict[];
}

/**
 * One DICOM series' role assignment, from `neurovision.data.dicom_ingest
 * .RoleAssignment` (`role` is `null` when a series matched none of ours, or
 * was rejected outright, or was too ambiguous to call).
 */
export interface ClinicalSeriesAssignment {
  role: string | null;
  score: number;
  reasons: string[];
  outcome: string;
}

/**
 * A series `assign_roles` could not place. `app/backend/clinical_jobs.py`'s
 * `_ingest_result_to_dict` emits this as an OBJECT, `{series_uid, reason}` -
 * not the `[uid, reason]` tuple pair this file's own docstring conventions
 * might otherwise suggest, so it is typed exactly as the server sends it.
 */
export interface ClinicalRejectedSeries {
  series_uid: string;
  reason: string;
}

/** `app/backend/clinical_jobs.py`'s `_ingest_result_to_dict` of E1's `IngestResult`. */
export interface ClinicalIngestResult {
  paths: Record<string, string>;
  assignments: Record<string, ClinicalSeriesAssignment>;
  missing_roles: string[];
  rejected: ClinicalRejectedSeries[];
  warnings: string[];
}

/**
 * `app.backend.clinical_jobs.ClinicalJob`, serialised verbatim via
 * `dataclasses.asdict`. See that dataclass's docstring for the field-by-
 * field meaning and the fixed pipeline order that populates them.
 *
 * `state="refused"` is a distinct, successful terminal state, not an error -
 * one of the label-free gates (input QC or the gatekeeper) looked at this
 * exact study and correctly declined it. `error` is set for both `"refused"`
 * and `"failed"`; only `state` tells them apart.
 */
export interface ClinicalJob {
  job_id: string;
  state: ClinicalJobState;
  stage: string;
  progress: number;
  case_id: string;
  error: string | null;
  ingest_result: ClinicalIngestResult | null;
  input_qc_pre: InputQCReportJson | null;
  input_qc_post: InputQCReportJson | null;
  preprocess_warnings: string[] | null;
  gatekeeper_decision: GatekeeperDecisionJson | null;
  created_at: number;
  updated_at: number;
}

/**
 * Uploads a raw DICOM study `.zip` and queues a clinical job.
 *
 * `FormData` sets its own multipart boundary via `Content-Type` - setting
 * that header manually here would omit the boundary parameter and the
 * server would fail to parse the body at all.
 *
 * Mirrors `fetchReport`'s error handling exactly: a non-2xx response's body
 * is read for a `detail` string (the server's actual reason - "dicom_zip is
 * empty", a zip-slip attempt, an oversized upload) and that becomes the
 * thrown `ApiError`'s message, falling back to the generic
 * status-line message when the body is not JSON or carries no `detail`.
 */
export async function createClinicalJob(
  dicomZip: Blob,
  signal?: AbortSignal,
): Promise<ClinicalJob> {
  const path = "/clinical/upload";
  const formData = new FormData();
  formData.append("dicom_zip", dicomZip);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "POST", body: formData, signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    if (GATEWAY_DOWN.has(res.status)) throw new ApiUnreachableError();
    let detail: string | undefined;
    try {
      const body = (await res.json()) as unknown;
      if (body && typeof body === "object" && typeof (body as { detail?: unknown }).detail === "string") {
        detail = (body as { detail: string }).detail;
      }
    } catch {
      // Body wasn't JSON (or was empty) - fall through to the generic message.
    }
    throw new ApiError(res.status, detail ?? `${res.status} ${res.statusText} on ${path}`);
  }
  return (await res.json()) as ClinicalJob;
}

export function getClinicalJob(jobId: string, signal?: AbortSignal): Promise<ClinicalJob> {
  return getJson<ClinicalJob>(`/clinical/jobs/${encodeURIComponent(jobId)}`, signal);
}

/**
 * Every clinical job the backend has on disk (`job.json` per job, rehydrated
 * at startup - see `app/backend/clinical_jobs.py`), newest first as the
 * server already orders them. This is what lets a finished study be reopened
 * after a reload: the page only ever learns a job's id from the upload that
 * created it, so without this list there is no way back to a job once its
 * id falls out of memory.
 */
export function listClinicalJobs(signal?: AbortSignal): Promise<{ jobs: ClinicalJob[] }> {
  return getJson<{ jobs: ClinicalJob[] }>("/clinical/jobs", signal);
}

/**
 * A `"done"` clinical job's case geometry - the one clinical route that
 * carries voxel `spacing` and `bbox`. The backend returns exactly
 * `volumes.CaseMeta.to_json()`, the same object `/cases/{id}` nests under
 * `meta`, so the existing `CaseMeta` type is reused verbatim rather than
 * declaring a parallel one. The 3D viewer needs `spacing` to convert voxel
 * counts to physical units (mL, real-world geometry) - the binary routes'
 * `X-Volume-Shape` response header carries only a voxel-count shape, never
 * spacing, so this JSON route is the only way to get it.
 */
export function getClinicalJobGeometry(jobId: string, signal?: AbortSignal): Promise<CaseMeta> {
  return getJson<CaseMeta>(`/clinical/jobs/${encodeURIComponent(jobId)}/geometry`, signal);
}

/**
 * A `"done"` clinical job's Phase 4 structured report and validates its
 * shape - identical semantics to `fetchReport`, just against
 * `/clinical/jobs/{id}/report` instead of `/report/{case_id}`. See
 * `fetchValidatedReport` for the shared error handling (404/500 `detail`
 * message, 502/503/504 as `ApiUnreachableError`, then `validateReport`).
 */
export async function fetchClinicalReport(jobId: string, signal?: AbortSignal): Promise<ReportResponse> {
  return fetchValidatedReport(`/clinical/jobs/${encodeURIComponent(jobId)}/report`, signal);
}

/**
 * Deletes a clinical job and everything it wrote to disk.
 *
 * Not built on `getJson` (GET-only) or `getBinary` (binary-only) - a small,
 * one-off `fetch` with `method: "DELETE"`, same non-2xx handling as every
 * other JSON route (`responseError`, so a dead backend behind the dev proxy
 * still classifies as unreachable rather than a bare 502 `ApiError`).
 */
export async function deleteClinicalJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<{ job_id: string; deleted: boolean }> {
  const path = `/clinical/jobs/${encodeURIComponent(jobId)}`;
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { method: "DELETE", signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    throw responseError(res, path);
  }
  return (await res.json()) as { job_id: string; deleted: boolean };
}

/**
 * One modality volume from a `"done"` clinical job - same wire shape as
 * `getVolume`, a 409 (job not done yet, including a `"refused"` job - there
 * is nothing to view) or 404 (unknown modality or job) surfacing as an
 * `ApiError` via `getBinary`'s existing handling.
 */
export function getClinicalJobVolume(
  jobId: string,
  modality: Modality,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer> {
  return getBinary(
    `/clinical/jobs/${encodeURIComponent(jobId)}/volume/${modality}`,
    fallbackShape,
    signal,
  );
}

/** The clinical job's segmentation as a `{0,1,2,3}` class map - same wire shape as `getMask`. */
export function getClinicalJobMask(
  jobId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer> {
  return getBinary(`/clinical/jobs/${encodeURIComponent(jobId)}/mask/prediction`, fallbackShape, signal);
}

/**
 * Same as `getUncertainty`, but for a clinical job's live-computed entropy.
 * Returns null on 404 (no cached logits for this job) rather than throwing.
 */
export function getClinicalJobUncertainty(
  jobId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  return getOptionalKindedBinary(
    `/clinical/jobs/${encodeURIComponent(jobId)}/uncertainty`,
    fallbackShape,
    signal,
  );
}

/**
 * The fitted conformal-risk-control band for one region (`WT` or `TC`) of a
 * clinical job's live segmentation - same wire format and same "null on 404"
 * convention as `getClinicalJobUncertainty` (a 404 here means no fitted
 * threshold is available for this region yet, a normal outcome, not an
 * error). Byte values: 0 outside the conservative mask, 128 inside the
 * conservative ("safety margin") mask but not the ordinary prediction, 255
 * inside the ordinary prediction.
 */
export function getClinicalJobConformalBand(
  jobId: string,
  region: "WT" | "TC",
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  return getOptionalKindedBinary(
    `/clinical/jobs/${encodeURIComponent(jobId)}/conformal-band/${region}`,
    fallbackShape,
    signal,
  );
}

/**
 * The Seg-Grad-CAM explainability heatmap for one region (`WT` or `TC`) of a
 * clinical job's live segmentation - same wire format and same "null on 404"
 * convention as `getClinicalJobConformalBand` (a 404 here means either the
 * job predates this feature or that region's Grad-CAM computation failed and
 * was skipped during the job run, per the backend's per-region failure
 * isolation - a normal outcome, not an error). Byte values are a `[0, 1]`-
 * normalized evidence score scaled to uint8, the same convention as entropy.
 */
export function getClinicalJobGradcam(
  jobId: string,
  region: "WT" | "TC",
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  return getOptionalKindedBinary(
    `/clinical/jobs/${encodeURIComponent(jobId)}/gradcam/${region}`,
    fallbackShape,
    signal,
  );
}

/**
 * This clinical job's atlas structure-index volume, cropped to its own bbox
 * - same "optional, kinded binary" contract as `getClinicalJobGradcam` (null
 * on 404 - no saved case meta for this job; an `ApiError`, including 409 for
 * a job that isn't `"done"` yet, on any other failure). `kind` must be
 * checked against `ATLAS_STRUCTURE_INDEX` by the caller before the buffer is
 * treated as atlas data - see `AtlasBuffer`.
 */
export function getClinicalJobAtlas(
  jobId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<AtlasBuffer | null> {
  return getOptionalKindedBinary(
    `/clinical/jobs/${encodeURIComponent(jobId)}/atlas`,
    fallbackShape,
    signal,
  );
}
