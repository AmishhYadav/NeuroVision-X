import { useEffect, useRef, useState } from "react";
import { InflightMap, LruCache } from "../lib/caseCache";
import {
  ApiUnreachableError,
  getCase,
  getMask,
  getProfile,
  getUncertainty,
  getVolume,
  type CaseDetail,
  type CaseProfile,
  type Modality,
  type UncertaintyBuffer,
  type VolumeBuffer,
} from "../api";

const MODALITIES: Modality[] = ["t1", "t1ce", "t2", "flair"];

export interface CaseDataState {
  /** The case id this data belongs to (null when nothing is loaded). Compare it
   * with the selected id before drawing: while the next case loads, the state
   * deliberately still holds the previous case's data. */
  caseId: string | null;
  detail: CaseDetail | null;
  volumes: Partial<Record<Modality, VolumeBuffer>>;
  predictionMask: VolumeBuffer | null;
  labelMask: VolumeBuffer | null;
  uncertainty: UncertaintyBuffer | null;
  profile: CaseProfile | null;
  loading: boolean;
  error: string | null;
}

const EMPTY_STATE: CaseDataState = {
  caseId: null,
  detail: null,
  volumes: {},
  predictionMask: null,
  labelMask: null,
  uncertainty: null,
  profile: null,
  loading: false,
  error: null,
};

// Capacity = the five-case showcase, so every showcase case can stay resident.
// Memory budget: 5 cases x ~26 MB (four modality volumes + masks + entropy)
// ~= 130 MB of Uint8Arrays - acceptable for a demo machine; the oldest case is
// dropped first if a deep link pushes a sixth in.
const CASE_CACHE_CAPACITY = 5;
const caseCache = new LruCache<CaseDataState>(CASE_CACHE_CAPACITY);
const inflight = new InflightMap<CaseDataState>();

function isAbort(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** The part of a case needed to draw it: metadata, four volumes and both masks. */
type CoreState = Pick<CaseDataState, "caseId" | "detail" | "volumes" | "predictionMask" | "labelMask">;

/**
 * Fetch one whole case with every request in flight at once (after the
 * detail, which says which artefacts exist). `onCore` fires once the
 * drawable core is ready; uncertainty and the profile are optional extras
 * that only delay the final result, and a failed extra is simply absent.
 */
async function fetchCase(
  caseId: string,
  signal: AbortSignal | undefined,
  onCore?: (core: CoreState) => void,
): Promise<CaseDataState> {
  // The profile does not depend on the detail, so start it immediately.
  // .catch keeps an early failure from becoming an unhandled rejection.
  const profileP = getProfile(caseId, signal).catch(() => null);

  const detail = await getCase(caseId, signal);
  const shape = detail.meta.shape;

  const volumeP = Promise.all(
    MODALITIES.map(async (m) => [m, await getVolume(caseId, m, shape, signal)] as const),
  );
  const predictionP = detail.meta.has_prediction
    ? getMask(caseId, "prediction", shape, signal)
    : Promise.resolve(null);
  const labelP = detail.meta.has_label
    ? getMask(caseId, "label", shape, signal)
    : Promise.resolve(null);
  const uncertaintyP = detail.meta.has_logits
    ? getUncertainty(caseId, shape, signal).catch(() => null)
    : Promise.resolve(null);

  const [volumeEntries, predictionMask, labelMask] = await Promise.all([
    volumeP,
    predictionP,
    labelP,
  ]);
  const core: CoreState = {
    caseId: detail.meta.case_id,
    detail,
    volumes: Object.fromEntries(volumeEntries),
    predictionMask,
    labelMask,
  };
  onCore?.(core);

  const [uncertainty, profile] = await Promise.all([uncertaintyP, profileP]);
  if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
  return { ...core, uncertainty, profile, loading: false, error: null };
}

/** Cache hit, else join an in-flight load, else start one (and cache it when done). */
function loadCase(
  caseId: string,
  signal?: AbortSignal,
  onCore?: (core: CoreState) => void,
): Promise<CaseDataState> {
  const hit = caseCache.get(caseId);
  if (hit) return Promise.resolve(hit);
  return inflight.run(caseId, async () => {
    const full = await fetchCase(caseId, signal, onCore);
    caseCache.set(caseId, full);
    return full;
  });
}

/**
 * Load a case into the module cache without touching React state, so a later
 * selection is instant. Failures (and aborts) are swallowed: a prefetch is a
 * hint, and the real selection will retry and surface any error.
 */
export function prefetchCase(caseId: string): Promise<void> {
  return loadCase(caseId).then(
    () => undefined,
    () => undefined,
  );
}

/**
 * Loads everything needed to render one case: metadata, all four modality
 * volumes, the prediction mask, the label mask (if present), the
 * uncertainty volume (if present), and the slice-ribbon profile.
 *
 * All requests run in parallel and land in one setState (the core), with
 * the optional extras in a second. Fully-loaded cases are kept in an LRU
 * cache, so returning to one commits synchronously with no network.
 *
 * Every case switch gets its own AbortController so a fast switch cannot
 * land a stale buffer on top of the current case - in-flight requests from
 * the previous case are aborted rather than raced.
 */
export function useCaseData(caseId: string | null): CaseDataState {
  const [state, setState] = useState<CaseDataState>(EMPTY_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    controllerRef.current?.abort();

    if (!caseId) {
      setState(EMPTY_STATE);
      return;
    }

    const cached = caseCache.get(caseId);
    if (cached) {
      setState(cached);
      return;
    }

    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;

    setState({ ...EMPTY_STATE, loading: true });

    const onCore = (core: CoreState) => {
      if (!signal.aborted) setState({ ...EMPTY_STATE, ...core, loading: true });
    };

    (async () => {
      try {
        let full: CaseDataState;
        try {
          full = await loadCase(caseId, signal, onCore);
        } catch (err) {
          // Under StrictMode the effect runs twice; the second run can join the
          // first run's load just as its abort lands. If WE were not aborted,
          // the shared load has already been evicted, so simply start again.
          if (isAbort(err) && !signal.aborted) {
            full = await loadCase(caseId, signal, onCore);
          } else {
            throw err;
          }
        }
        if (!signal.aborted) setState(full);
      } catch (err) {
        if (signal.aborted) return;
        const message =
          err instanceof ApiUnreachableError
            ? "No response from the API. Start it with `uvicorn app.backend.main:app --reload`."
            : err instanceof Error
              ? err.message
              : "Failed to load case.";
        setState((prev) => ({ ...prev, loading: false, error: message }));
      }
    })();

    return () => {
      controller.abort();
    };
  }, [caseId]);

  return state;
}
