import { useEffect, useRef, useState } from "react";
import {
  ApiUnreachableError,
  getClinicalJobConformalBand,
  getClinicalJobGradcam,
  getClinicalJobMask,
  getClinicalJobUncertainty,
  getClinicalJobVolume,
  type Modality,
  type UncertaintyBuffer,
  type VolumeBuffer,
} from "../api";

const MODALITIES: Modality[] = ["t1", "t1ce", "t2", "flair"];

// The two fixed regions the backend's conformal-band route serves, per
// `configs/clinical/default.yaml`'s `clinical.gatekeeper.regions` - not
// derived at runtime, so this list needs updating here if that config ever
// changes.
const REGIONS: ("WT" | "TC")[] = ["WT", "TC"];

// `getBinary` only falls back to this when a response is missing its own
// `X-Volume-Shape` header (a proxy stripping custom headers) - every real
// clinical response carries the header, so the exact placeholder value here
// is inert.
const FALLBACK_SHAPE: [number, number, number] = [1, 1, 1];

export interface ClinicalJobVolumesState {
  volumes: Partial<Record<Modality, VolumeBuffer>>;
  predictionMask: VolumeBuffer | null;
  uncertainty: UncertaintyBuffer | null;
  conformalBand: Partial<Record<"WT" | "TC", UncertaintyBuffer | null>>;
  gradcam: Partial<Record<"WT" | "TC", UncertaintyBuffer | null>>;
  loading: boolean;
  error: string | null;
}

const EMPTY_STATE: ClinicalJobVolumesState = {
  volumes: {},
  predictionMask: null,
  uncertainty: null,
  conformalBand: {},
  gradcam: {},
  loading: false,
  error: null,
};

/**
 * Loads a `"done"` clinical job's four modality volumes, its prediction
 * mask, its live-computed entropy map, its fitted conformal band for both
 * regions (`WT`, `TC`), and its Grad-CAM explainability heatmap for both
 * regions, in parallel.
 *
 * Mirrors `useCaseData`'s shape (a per-switch `AbortController`, partial
 * state filled in as each fetch resolves, so viewports light up one at a
 * time rather than waiting for everything) but scoped to what a clinical job
 * actually has once done: no label, no slice profile - a live case has no
 * ground truth, and this pipeline saves no per-slice artifacts for it.
 * Uncertainty IS available (computed live from the job's segmentation
 * logits, same wire format as the demo viewer's `/cases/{id}/uncertainty`);
 * a `null` result just means no cached logits for this particular job, and
 * is a normal outcome, not an error - see `getClinicalJobUncertainty`. The
 * conformal band is fetched the same eager way, for both regions at once -
 * a `null` result there means no fitted threshold is available yet (or,
 * defensively, an unrecognised region), also a normal outcome, not an error.
 * The Grad-CAM heatmap is fetched the same way again, also per region - a
 * `null` result there means either the job predates this feature or that
 * region's Grad-CAM computation failed and was skipped, likewise a normal
 * outcome, not an error - see `getClinicalJobGradcam`. All of these are
 * fetched unconditionally on load; which one (if any) is actually displayed
 * is a UI-only decision made by the caller.
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

        const uncertaintyPromise = getClinicalJobUncertainty(jobId, FALLBACK_SHAPE, signal).then(
          (result) => {
            if (signal.aborted) return;
            setState((prev) => ({ ...prev, uncertainty: result }));
          },
        );

        const conformalBandPromises = REGIONS.map(async (region) => {
          const result = await getClinicalJobConformalBand(jobId, region, FALLBACK_SHAPE, signal);
          if (signal.aborted) return;
          setState((prev) => ({
            ...prev,
            conformalBand: { ...prev.conformalBand, [region]: result },
          }));
        });

        const gradcamPromises = REGIONS.map(async (region) => {
          const result = await getClinicalJobGradcam(jobId, region, FALLBACK_SHAPE, signal);
          if (signal.aborted) return;
          setState((prev) => ({
            ...prev,
            gradcam: { ...prev.gradcam, [region]: result },
          }));
        });

        await Promise.all([
          ...volumePromises,
          maskPromise,
          uncertaintyPromise,
          ...conformalBandPromises,
          ...gradcamPromises,
        ]);

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
