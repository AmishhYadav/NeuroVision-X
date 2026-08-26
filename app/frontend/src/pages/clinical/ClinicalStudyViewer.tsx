import { useEffect, useState } from "react";
import type { Modality, Plane } from "../../api";
import { MODALITY_ORDER } from "../../components/ControlBar";
import { Legend } from "../../components/Legend";
import { SliceRibbon } from "../../components/SliceRibbon";
import { ViewportGrid } from "../../components/ViewportGrid";
import { useClinicalJobVolumes } from "../../hooks/useClinicalJobVolumes";
import { useResponsiveLayout } from "../../hooks/useResponsiveLayout";
import { sliceIndexer } from "../../lib/slicing";

const ZERO_PLANES: Record<Plane, number> = { sagittal: 0, coronal: 0, axial: 0 };
// A live clinical case has no saved slice-ribbon profile (that is a
// precomputed evaluation artifact - see useCaseData) - the ribbon still
// needs SOME array for its tumour lane, and an empty one draws a flat,
// honest "nothing measured" line rather than fabricating one.
const EMPTY_PROFILE: number[] = [];

interface ClinicalStudyViewerProps {
  jobId: string;
}

/**
 * The segmentation viewer for a `"done"` clinical job.
 *
 * Deliberately much smaller than `App.tsx`: no truth outline (no ground
 * truth exists for a live case), no uncertainty toggle (no saved logits are
 * exposed to the client), no report panel, no case list - just the four
 * modalities and the predicted mask, scrollable per plane. `planeCounts` is
 * derived purely from `sliceIndexer(plane, shape).count` on whichever
 * volume's shape has arrived first; no backend "meta" endpoint is needed for
 * that, since every one of a job's binary responses already carries its own
 * `X-Volume-Shape` header (see `getBinary` in `api.ts`).
 */
export function ClinicalStudyViewer({ jobId }: ClinicalStudyViewerProps) {
  const { volumes, predictionMask, loading, error } = useClinicalJobVolumes(jobId, true);
  const { layout } = useResponsiveLayout();

  const [modality, setModality] = useState<Modality>("t1ce");
  const [expandedPlane, setExpandedPlane] = useState<Plane | null>(null);
  const [singlePlane, setSinglePlane] = useState<Plane>("axial");
  const [focusedPlane, setFocusedPlane] = useState<Plane>("axial");
  const [sliceIndices, setSliceIndices] = useState<Record<Plane, number>>(ZERO_PLANES);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);

  const shape =
    volumes.t1ce?.shape ??
    volumes.t1?.shape ??
    volumes.t2?.shape ??
    volumes.flair?.shape ??
    predictionMask?.shape ??
    null;

  const planeCounts: Record<Plane, number> = shape
    ? {
        sagittal: sliceIndexer("sagittal", shape).count,
        coronal: sliceIndexer("coronal", shape).count,
        axial: sliceIndexer("axial", shape).count,
      }
    : ZERO_PLANES;

  // Centre the view the first time this job's shape becomes known.
  useEffect(() => {
    if (!shape) return;
    setSliceIndices({
      sagittal: Math.floor(planeCounts.sagittal / 2),
      coronal: Math.floor(planeCounts.coronal / 2),
      axial: Math.floor(planeCounts.axial / 2),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shape]);

  // Global slice-stepping, scoped to the last-focused viewport - same
  // shortcut convention App.tsx uses, minus the 1-4 modality keys (this
  // viewer's four buttons are few enough to just click).
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (!shape) return;

      const activePlane = layout === "single" ? singlePlane : (expandedPlane ?? focusedPlane);
      const count = planeCounts[activePlane];

      let delta = 0;
      if (e.key === "ArrowRight") delta = 1;
      else if (e.key === "ArrowLeft") delta = -1;
      else if (e.key === "ArrowUp") delta = 10;
      else if (e.key === "ArrowDown") delta = -10;
      else return;

      e.preventDefault();
      setSliceIndices((prev) => ({
        ...prev,
        [activePlane]: Math.min(count - 1, Math.max(0, prev[activePlane] + delta)),
      }));
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [shape, layout, singlePlane, expandedPlane, focusedPlane, planeCounts]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center text-center">
        <p className="font-mono text-sm text-text-primary">{error}</p>
      </div>
    );
  }

  const ribbonPlane: Plane = layout === "single" ? singlePlane : (expandedPlane ?? "axial");
  const ribbonLabel = ribbonPlane.charAt(0).toUpperCase() + ribbonPlane.slice(1);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className={`relative flex min-h-0 flex-1 ${layout === "single" ? "flex-col" : "flex-row"}`}
      >
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-2">
          {loading && (
            <div
              className="flex shrink-0 items-center gap-2 border border-surface-seam bg-surface-panel px-3 py-1.5"
              role="status"
              aria-live="polite"
            >
              <span className="font-condensed text-[11px] tracking-[0.12em] text-text-dim uppercase">
                Loading segmentation…
              </span>
            </div>
          )}
          <div className="min-h-0 flex-1">
            <ViewportGrid
              layout={layout}
              expandedPlane={expandedPlane}
              onToggleExpand={(plane) =>
                setExpandedPlane((prev) => (prev === plane ? null : plane))
              }
              singlePlane={singlePlane}
              onChangeSinglePlane={setSinglePlane}
              onFocusPlane={setFocusedPlane}
              sliceIndices={sliceIndices}
              planeCounts={planeCounts}
              shape={shape}
              image={volumes[modality]?.data ?? null}
              predictionMask={predictionMask?.data ?? null}
              labelMask={null}
              uncertainty={null}
              overlayMode="prediction"
              overlayOpacity={overlayOpacity}
              showTruthOutline={false}
              showUncertainty={false}
              uncertaintyOpacity={0}
            />
          </div>
          <div className="shrink-0">
            <SliceRibbon
              planeLabel={ribbonLabel}
              sliceCount={planeCounts[ribbonPlane]}
              currentIndex={sliceIndices[ribbonPlane]}
              onScrub={(i) => setSliceIndices((prev) => ({ ...prev, [ribbonPlane]: i }))}
              tumor={EMPTY_PROFILE}
              error={null}
              entropy={null}
              onFocusRibbon={() => setFocusedPlane(ribbonPlane)}
            />
          </div>
        </div>

        <div
          className={`shrink-0 overflow-y-auto border-surface-seam bg-surface-panel ${
            layout === "single" ? "max-h-56 w-full border-t" : "w-56 border-l"
          }`}
        >
          <Legend overlayMode="prediction" showUncertainty={false} hasLabel={false} uncertaintyKind={null} />
        </div>
      </div>

      <div className="flex min-h-12 shrink-0 flex-wrap items-center gap-3 border-t border-surface-seam bg-surface-panel px-3 py-1.5">
        <div className="flex items-center gap-1" role="group" aria-label="Modality">
          {MODALITY_ORDER.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setModality(m)}
              aria-pressed={modality === m}
              className={`rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
                modality === m
                  ? "bg-surface-raised text-text-primary"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="eyebrow shrink-0">Opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            className="w-24 accent-[#E7EAEE]"
            aria-label="Overlay opacity"
          />
          <span className="tabular w-9 shrink-0 font-mono text-xs text-text-secondary">
            {overlayOpacity.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
