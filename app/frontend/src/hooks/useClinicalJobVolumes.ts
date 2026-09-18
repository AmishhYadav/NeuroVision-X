import { useEffect, useRef, useState } from "react";
import {
  ApiUnreachableError,
  ATLAS_STRUCTURE_INDEX,
  getAtlasStructures,
  getClinicalJobAtlas,
  getClinicalJobConformalBand,
  getClinicalJobGeometry,
  getClinicalJobGradcam,
  getClinicalJobMask,
  getClinicalJobUncertainty,
  getClinicalJobVolume,
  type AtlasBuffer,
  type AtlasStructureRow,
  type CaseMeta,
  type Modality,
  type UncertaintyBuffer,
  type VolumeBuffer,
} from "../api";
import { settleSupplementary } from "../lib/supplementaryFetch";

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
  /**
   * The atlas structure-index volume, cropped to this job's own bbox - which
   * structure (by index) occupies each voxel, for picking a shell in the 3D
   * twin (T3.4). `null` while loading, if the job has no saved case meta
   * (404), or if the response's `X-Uncertainty-Kind` was not
   * `ATLAS_STRUCTURE_INDEX` - a mislabelled volume is never painted (see
   * `docs/lessons.md` lesson 29: label layers only from the header, never
   * assume what a volume is).
   */
  atlas: AtlasBuffer | null;
  /**
   * The atlas's structure table (name, laterality, lobe, eloquence per
   * index) - what `atlas`'s raw index values are named against, and what
   * `selectStructures` (`src/lib/atlasSelection.ts`) maps the live report's
   * structure names onto. `null` while loading or on failure.
   */
  atlasTable: AtlasStructureRow[] | null;
  /**
   * Case geometry (`shape`, `spacing`, `bbox`) - what the 3D digital twin
   * needs to build real-world-scaled mesh geometry. Served by
   * `/clinical/jobs/{id}/geometry`, the one clinical route that carries
   * voxel spacing; the binary volume/mask routes' `X-Volume-Shape` header
   * gives only a voxel-count shape. `null` while loading, or if this job's
   * `meta.json` is missing (see the 404 handling below) - a normal outcome,
   * not an error.
   */
  geometry: CaseMeta | null;
  loading: boolean;
  error: string | null;
  /**
   * Human-readable lines for supplementary artifacts that failed to load
   * (anything other than an expected 404 - see the hook's doc comment for
   * the core/supplementary split). Empty when nothing went wrong. Rendered
   * by the caller as a small "Layer unavailable" strip; never blocks
   * rendering of the slices, twin or report the way `error` does.
   */
  warnings: string[];
}

const EMPTY_STATE: ClinicalJobVolumesState = {
  volumes: {},
  predictionMask: null,
  uncertainty: null,
  conformalBand: {},
  gradcam: {},
  atlas: null,
  atlasTable: null,
  geometry: null,
  loading: false,
  error: null,
  warnings: [],
};

/**
 * Loads a `"done"` clinical job's four modality volumes, its prediction
 * mask, its live-computed entropy map, its fitted conformal band for both
 * regions (`WT`, `TC`), its Grad-CAM explainability heatmap for both
 * regions, its atlas structure-index volume and the atlas's structure
 * table, and its case geometry, in parallel.
 *
 * **Core vs. supplementary.** The four volumes and the prediction mask are
 * the product - a done job with no error means at minimum those loaded, and
 * a failure fetching any of them is a real, hook-level `error` (see below).
 * Everything else - uncertainty, the conformal band (both regions), Grad-CAM
 * (both regions), the atlas volume, the atlas structure table, and geometry
 * - is supplementary: extras layered on top, each capable of naming its own
 * absence without taking the study down with it. Geometry is the one
 * exception that cuts both ways - the 3D twin treats it as required to
 * build scaled mesh geometry, but for fetch-failure purposes here it is
 * classified as supplementary, same as the rest.
 *
 * This split exists because the batch is awaited together in one
 * `Promise.all`: on the first real done clinical job, `/conformal-band/WT`
 * 500'd (a backend bug), and without this split that one rejection took the
 * whole batch down - the viewer rendered only its `error` line, with no
 * slices, no twin and no report, even though everything else had already
 * loaded successfully. Every supplementary fetch is routed through
 * `settleSupplementary` (`src/lib/supplementaryFetch.ts`): a 404 (an
 * expected absence - no cached logits yet, no fitted threshold, a job that
 * predates a feature, an unrecognised region) resolves to `null` silently;
 * any other error also resolves to `null` but appends a human-readable line
 * to `warnings` instead of throwing; `ApiUnreachableError` (the API itself
 * is down) still escalates to the hook-level `error`, since nothing else in
 * the batch is going to succeed either. Core fetches keep their original,
 * unwrapped behaviour - any error there is a hook-level `error`.
 *
 * Mirrors `useCaseData`'s shape (a per-switch `AbortController`, partial
 * state filled in as each fetch resolves, so viewports light up one at a
 * time rather than waiting for everything) but scoped to what a clinical job
 * actually has once done: no label, no slice profile - a live case has no
 * ground truth, and this pipeline saves no per-slice artifacts for it. All
 * of the above are fetched unconditionally on load; which one (if any) is
 * actually displayed is a UI-only decision made by the caller.
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
      // Collected by every supplementary fetch below (via `settleSupplementary`)
      // and written into state once as `warnings` after the batch settles -
      // see the hook's doc comment for why these are never a hook-level `error`.
      const warnings: string[] = [];

      try {
        // Core: the four volumes and the prediction mask. Any error here
        // propagates unwrapped to the outer catch below, same as before.
        const volumePromises = MODALITIES.map(async (modality) => {
          const vol = await getClinicalJobVolume(jobId, modality, FALLBACK_SHAPE, signal);
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, volumes: { ...prev.volumes, [modality]: vol } }));
        });

        const maskPromise = getClinicalJobMask(jobId, FALLBACK_SHAPE, signal).then((mask) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, predictionMask: mask }));
        });

        // Supplementary from here down: a non-404 failure resolves to `null`
        // and appends a line to `warnings` instead of failing the batch.
        const uncertaintyPromise = settleSupplementary(
          "Uncertainty",
          getClinicalJobUncertainty(jobId, FALLBACK_SHAPE, signal),
          warnings,
        ).then((result) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, uncertainty: result }));
        });

        const conformalBandPromises = REGIONS.map(async (region) => {
          const result = await settleSupplementary(
            `Conformal band (${region})`,
            getClinicalJobConformalBand(jobId, region, FALLBACK_SHAPE, signal),
            warnings,
          );
          if (signal.aborted) return;
          setState((prev) => ({
            ...prev,
            conformalBand: { ...prev.conformalBand, [region]: result },
          }));
        });

        const gradcamPromises = REGIONS.map(async (region) => {
          const result = await settleSupplementary(
            `Grad-CAM (${region})`,
            getClinicalJobGradcam(jobId, region, FALLBACK_SHAPE, signal),
            warnings,
          );
          if (signal.aborted) return;
          setState((prev) => ({
            ...prev,
            gradcam: { ...prev.gradcam, [region]: result },
          }));
        });

        // The atlas structure-index volume - what structure (by index)
        // occupies each voxel, for the twin's per-structure shells (T3.4).
        const atlasPromise = settleSupplementary(
          "Atlas",
          getClinicalJobAtlas(jobId, FALLBACK_SHAPE, signal),
          warnings,
        ).then((result) => {
          if (signal.aborted) return;
          // Guard against a mislabelled volume (docs/lessons.md lesson 29):
          // only ever store this buffer as atlas data when the backend's
          // own header confirms it - a 200 response is not, by itself,
          // proof of what it contains.
          if (result && result.kind !== ATLAS_STRUCTURE_INDEX) {
            warnings.push(`Atlas unavailable: unexpected X-Uncertainty-Kind ${result.kind}`);
            setState((prev) => ({ ...prev, atlas: null }));
            return;
          }
          setState((prev) => ({ ...prev, atlas: result }));
        });

        // The atlas's own structure table - what `atlas`'s raw index values
        // are named against. Not job-specific (same atlas for every job),
        // but fetched alongside everything else so the viewer never has to
        // juggle a second loading state for it.
        const atlasTablePromise = settleSupplementary(
          "Atlas structures",
          getAtlasStructures(signal),
          warnings,
        ).then((result) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, atlasTable: result ? result.structures : null }));
        });

        // Geometry feeds the 3D twin, but a failure to fetch it is still
        // classified as supplementary - it must not blank the slice viewer.
        const geometryPromise = settleSupplementary(
          "Geometry",
          getClinicalJobGeometry(jobId, signal),
          warnings,
        ).then((geometry) => {
          if (signal.aborted) return;
          setState((prev) => ({ ...prev, geometry }));
        });

        await Promise.all([
          ...volumePromises,
          maskPromise,
          uncertaintyPromise,
          ...conformalBandPromises,
          ...gradcamPromises,
          atlasPromise,
          atlasTablePromise,
          geometryPromise,
        ]);

        if (!signal.aborted) {
          setState((prev) => ({ ...prev, loading: false, warnings }));
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
