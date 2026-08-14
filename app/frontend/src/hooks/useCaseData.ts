import { useEffect, useRef, useState } from "react";
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
  type VolumeBuffer,
} from "../api";

const MODALITIES: Modality[] = ["t1", "t1ce", "t2", "flair"];

export interface CaseDataState {
  detail: CaseDetail | null;
  volumes: Partial<Record<Modality, VolumeBuffer>>;
  predictionMask: VolumeBuffer | null;
  labelMask: VolumeBuffer | null;
  uncertainty: VolumeBuffer | null;
  profile: CaseProfile | null;
  loading: boolean;
  error: string | null;
}

const EMPTY_STATE: CaseDataState = {
  detail: null,
  volumes: {},
  predictionMask: null,
  labelMask: null,
  uncertainty: null,
  profile: null,
  loading: false,
  error: null,
};

/**
 * Loads everything needed to render one case: metadata, all four modality
 * volumes, the prediction mask, the label mask (if present), the
 * uncertainty volume (if present), and the slice-ribbon profile.
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

    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;

    setState({ ...EMPTY_STATE, loading: true });

    (async () => {
      try {
        const detail = await getCase(caseId, signal);
        if (signal.aborted) return;
        const shape = detail.meta.shape;

        setState((prev) => (prev.loading ? { ...prev, detail } : prev));

        const volumePromises = MODALITIES.map(async (modality) => {
          const vol = await getVolume(caseId, modality, shape, signal);
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, volumes: { ...prev.volumes, [modality]: vol } }));
        });

        const predictionPromise = detail.meta.has_prediction
          ? getMask(caseId, "prediction", shape, signal).then((mask) => {
              if (signal.aborted) return;
              setState((prev) => ({ ...prev, predictionMask: mask }));
            })
          : Promise.resolve();

        const labelPromise = detail.meta.has_label
          ? getMask(caseId, "label", shape, signal).then((mask) => {
              if (signal.aborted) return;
              setState((prev) => ({ ...prev, labelMask: mask }));
            })
          : Promise.resolve();

        const uncertaintyPromise = detail.meta.has_logits
          ? getUncertainty(caseId, shape, signal).then((vol) => {
              if (signal.aborted) return;
              setState((prev) => ({ ...prev, uncertainty: vol }));
            })
          : Promise.resolve();

        const profilePromise = getProfile(caseId, signal).then((profile) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, profile }));
        });

        await Promise.all([
          ...volumePromises,
          predictionPromise,
          labelPromise,
          uncertaintyPromise,
          profilePromise,
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
