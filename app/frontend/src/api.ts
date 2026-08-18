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
export async function getUncertainty(
  caseId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<UncertaintyBuffer | null> {
  const path = `/cases/${encodeURIComponent(caseId)}/uncertainty`;
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

export function getProfile(caseId: string, signal?: AbortSignal): Promise<CaseProfile> {
  return getJson<CaseProfile>(`/cases/${encodeURIComponent(caseId)}/profile`, signal);
}

/**
 * Fetches one case's Phase 4 structured report and validates its shape.
 *
 * Deliberately does not reuse `getJson`: on a 404 or 500 the backend's body
 * carries a `detail` message worth showing verbatim - e.g. the provenance
 * guard's 500 names both the report's own segmentation directory and the one
 * this server is configured to display, which is exactly what a reader needs
 * to fix a misconfigured `NVX_REPORT_DIR`. `getJson` only ever builds a
 * generic "<status> <statusText> on <path>" message from the response line,
 * which would throw that detail away.
 */
export async function fetchReport(caseId: string, signal?: AbortSignal): Promise<ReportResponse> {
  const path = `/report/${encodeURIComponent(caseId)}`;
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
