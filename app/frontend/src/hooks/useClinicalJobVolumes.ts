import { useEffect, useRef, useState } from "react";
import {
  ApiUnreachableError,
  getClinicalJobMask,
  getClinicalJobVolume,
  type Modality,
  type VolumeBuffer,
} from "../api";

const MODALITIES: Modality[] = ["t1", "t1ce", "t2", "flair"];

// `getBinary` only falls back to this when a response is missing its own
// `X-Volume-Shape` header (a proxy stripping custom headers) - every real
// clinical response carries the header, so the exact placeholder value here
// is inert.
const FALLBACK_SHAPE: [number, number, number] = [1, 1, 1];

export interface ClinicalJobVolumesState {
  volumes: Partial<Record<Modality, VolumeBuffer>>;
  predictionMask: VolumeBuffer | null;
  loading: boolean;
  error: string | null;
}

const EMPTY_STATE: ClinicalJobVolumesState = {
  volumes: {},
  predictionMask: null,
  loading: false,
  error: null,
};

/**
 * Loads a `"done"` clinical job's four modality volumes and its prediction
 * mask, in parallel.
 *
 * Mirrors `useCaseData`'s shape (a per-switch `AbortController`, partial
 * state filled in as each fetch resolves, so viewports light up one at a
 * time rather than waiting for everything) but scoped to what a clinical job
 * actually has once done: no label, no logits/uncertainty, no slice profile
 * - a live case has no ground truth, and this pipeline saves none of those
 * per-slice artifacts for it.
 *
 * Only fetches while `ready` is true (the caller passes
 * `job?.state === "done"`) - fetching against a job that is still running,
 * or was refused, would just 409.
 */
export function useClinicalJobVolumes(
  jobId: string | null,
  ready: boolean,
): ClinicalJobVolumesState {
  const [state, setState] = useState<ClinicalJobVolumesState>(EMPTY_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    controllerRef.current?.abort();

    if (!jobId || !ready) {
      setState(EMPTY_STATE);
      return;
    }

    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;

    setState({ ...EMPTY_STATE, loading: true });

    (async () => {
      try {
        const volumePromises = MODALITIES.map(async (modality) => {
          const vol = await getClinicalJobVolume(jobId, modality, FALLBACK_SHAPE, signal);
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, volumes: { ...prev.volumes, [modality]: vol } }));
        });

        const maskPromise = getClinicalJobMask(jobId, FALLBACK_SHAPE, signal).then((mask) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, predictionMask: mask }));
        });

        await Promise.all([...volumePromises, maskPromise]);

        if (!signal.aborted) {
          setState((prev) => ({ ...prev, loading: false }));
        }
      } catch (err) {
        if (signal.aborted) return;
        const message =
          err instanceof ApiUnreachableError
            ? "No response from the API. Start it with `uvicorn app.backend.main:app --reload`."
            : err instanceof Error
              ? err.message
              : "Failed to load this job's volumes.";
        setState((prev) => ({ ...prev, loading: false, error: message }));
      }
    })();

    return () => {
      controller.abort();
    };
  }, [jobId, ready]);

  return state;
}
