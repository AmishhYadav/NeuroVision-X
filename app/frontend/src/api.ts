// Typed client for the NeuroVision-X demo backend.
//
// Every function here mirrors one endpoint of the API contract exactly - no
// reshaping, no re-sorting, no invented fields. Binary endpoints return the
// raw bytes plus the shape read from the `X-Volume-Shape` header, falling
// back to a caller-supplied shape when the header is missing (a proxy can
// strip custom headers).

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
}

export interface CaseSummary {
  case_id: string;
  dice_mean: number | null;
  dice: Record<RegionKey, number> | null;
  has_label: boolean;
  has_logits: boolean;
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
}

export interface VolumeBuffer {
  data: Uint8Array;
  shape: [number, number, number];
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

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiUnreachableError();
  }
  if (!res.ok) {
    throw new ApiError(res.status, `${res.status} ${res.statusText} on ${path}`);
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
    throw new ApiError(res.status, `${res.status} ${res.statusText} on ${path}`);
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

/** Returns null on 404 (no saved logits for this case) rather than throwing. */
export async function getUncertainty(
  caseId: string,
  fallbackShape: [number, number, number],
  signal?: AbortSignal,
): Promise<VolumeBuffer | null> {
  try {
    return await getBinary(
      `/cases/${encodeURIComponent(caseId)}/uncertainty`,
      fallbackShape,
      signal,
    );
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export function getProfile(caseId: string, signal?: AbortSignal): Promise<CaseProfile> {
  return getJson<CaseProfile>(`/cases/${encodeURIComponent(caseId)}/profile`, signal);
}
