import { Box, Download, FileText } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  CONFORMAL_BAND,
  exportClinicalJob,
  GRADCAM,
  PREDICTIVE_ENTROPY_SINGLE_PASS,
  type GateDecisionValue,
  type Modality,
  type Plane,
  type UncertaintyBuffer,
} from "../../api";
import {
  BrainTwinScene,
  type BrainTwinInput,
  type TwinActiveLayer,
} from "../../components/BrainTwinScene";
import { MODALITY_ORDER } from "../../components/ControlBar";
import { Legend } from "../../components/Legend";
import { ReportPanel } from "../../components/ReportPanel";
import { SliceRibbon } from "../../components/SliceRibbon";
import { Viewport } from "../../components/Viewport";
import { ViewportGrid } from "../../components/ViewportGrid";
import { useClinicalJobVolumes } from "../../hooks/useClinicalJobVolumes";
import { useClinicalReport } from "../../hooks/useClinicalReport";
import { useResponsiveLayout } from "../../hooks/useResponsiveLayout";
import {
  MAX_ATLAS_STRUCTURES,
  selectStructures,
  structureIndexForName,
  structureRow,
} from "../../lib/atlasSelection";
import { triggerBlobDownload } from "../../lib/download";
import { sliceIndexer } from "../../lib/slicing";
import { bandStats, entropyProfile, entropyStats, gradcamStats } from "../../lib/heatStats";
import { tumourProfile } from "../../lib/tumourProfile";
import { reportRowForStructure } from "../../lib/structureDetail";
import { captureCanvas, findTwinCanvas, snapshotFilename } from "../../lib/twinSnapshot";
import { StudySidebar } from "./StudySidebar";
import { UncertaintyCard } from "./UncertaintyCard";

const ZERO_PLANES: Record<Plane, number> = { sagittal: 0, coronal: 0, axial: 0 };
// Fixed, not a slider - matches App.tsx's own choice exactly (same constant
// name and value) so the two viewers render entropy identically. Neither
// viewer exposes a user-adjustable uncertainty-opacity control.
const UNCERTAINTY_OPACITY = 0.6;
// The slice ribbon's tumour lane is computed from the predicted mask itself
// (lib/tumourProfile.ts) - a live clinical case has no saved, precomputed
// profile like the research viewer's evaluation cases do.

interface ClinicalStudyViewerProps {
  jobId: string;
  /** The gatekeeper's verdict for this job, or `null` before it has one (still running, refused). */
  decision: GateDecisionValue | null;
}

/** The six heat sources this viewer can show over the segmentation, one at a time. */
/** Stage layout: three planes + twin together, planes only, or the twin alone. */
type ViewMode = "tri" | "slices" | "twin";

type HeatOverlay = "none" | "entropy" | "band_wt" | "band_tc" | "gradcam_wt" | "gradcam_tc";

/**
 * The segmentation viewer for a `"done"` clinical job.
 *
 * A "Digital Twin Studio" layout: a toolbar strip on top (view mode,
 * modality, heat overlay, opacity, structures, report, export), the stage in
 * the middle and a study-summary sidebar (StudySidebar) on the right. Three
 * view modes: "Tri-plane + twin" (default once the twin's data is in: the
 * three slice planes plus the twin side by side), "Slices" (planes only) and
 * "3D twin" (twin only). The twin is `BrainTwinScene` - the SAME
 * digital-twin component the research viewer (`App.tsx`) uses, fed here by
 * this job's `/geometry` response plus its four volumes and predicted mask,
 * never forked - and only ONE instance is ever mounted. Unlike the research
 * viewer there is no truth outline and no ground truth for the twin to draw
 * (a live case has no label, so its `tumorSource` is always `"prediction"`),
 * and no case list. A "Report" button opens the same `ReportPanel` drawer
 * the research viewer uses, overlaying the stage.
 *
 * The heat overlay is a single-choice selector between predictive entropy,
 * the fitted conformal band, and the Grad-CAM explainability heatmap, each
 * of the latter two further split by region (WT, TC) - all six states reuse
 * the same generic `uncertainty`/`showUncertainty`/`uncertaintyOpacity`
 * props on `ViewportGrid` and `Legend`, which alpha-blend and label whatever
 * buffer is handed to them regardless of what it represents. `planeCounts`
 * is derived purely from `sliceIndexer(plane, shape).count` on whichever
 * volume's shape has arrived first; no backend "meta" endpoint is needed for
 * that, since every one of a job's binary responses already carries its own
 * `X-Volume-Shape` header (see `getBinary` in `api.ts`).
 */
export function ClinicalStudyViewer({ jobId, decision }: ClinicalStudyViewerProps) {
  const {
    volumes,
    predictionMask,
    uncertainty,
    conformalBand,
    gradcam,
    atlas,
    atlasTable,
    geometry,
    loading,
    error,
    warnings,
  } = useClinicalJobVolumes(jobId, true);
  const { layout } = useResponsiveLayout();
  // Only mounted for a `"done"` job (see ClinicalPage), so fetching the
  // report is always safe here - mirrors App.tsx's own report effect, which
  // only fires once its case detail has confirmed a report exists.
  const reportState = useClinicalReport(jobId, true);

  const [modality, setModality] = useState<Modality>("t1ce");
  const [expandedPlane, setExpandedPlane] = useState<Plane | null>(null);
  const [singlePlane, setSinglePlane] = useState<Plane>("axial");
  const [focusedPlane, setFocusedPlane] = useState<Plane>("axial");
  const [sliceIndices, setSliceIndices] = useState<Record<Plane, number>>(ZERO_PLANES);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [heatOverlay, setHeatOverlay] = useState<HeatOverlay>("none");
  // Default is the combined tri-plane + twin stage; `effectiveView` (below)
  // falls back to "slices" until the twin's data has arrived.
  const [view, setView] = useState<ViewMode>("tri");
  const [reportOpen, setReportOpen] = useState(false);
  // T6.4 export: whether a bundle download is in flight, and the last
  // failure to show next to the button. Both are local to this component -
  // the export itself carries no state on the server (see exportClinicalJob's
  // docstring), so there is nothing to lift or persist across a reload.
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  // Points at the wrapper div around BrainTwinScene (only mounted while
  // the twin is on screen, in "tri" or "twin" mode) so the export handler can find its live <canvas> via
  // findTwinCanvas - the same DOM query twinSnapshot.ts already centralises
  // in one place, reused rather than re-querying document here.
  const twinHostRef = useRef<HTMLDivElement>(null);
  // The atlas structure currently highlighted in the twin - lifted here (per
  // T3.5/T3.6) because both BrainTwinScene (a shell click/hover) and
  // ReportPanel (a table row hover/click) need to read AND write it, so
  // neither owns it alone.
  const [highlightedStructure, setHighlightedStructure] = useState<number | null>(null);
  // Structure indices the user added via the "Add a structure" checkboxes,
  // on top of whatever the report already selected. Kept separate from the
  // report-derived indices so unchecking one only ever removes a
  // user-added structure - the report's own selection is never touched by
  // this state.
  const [extraStructures, setExtraStructures] = useState<number[]>([]);

  // A new job has no relationship to the previous one's highlighted/added
  // structures - carrying them over would highlight or mesh a structure
  // index that may mean something entirely different (or nothing) for the
  // new case's report.
  useEffect(() => {
    setHighlightedStructure(null);
    setExtraStructures([]);
  }, [jobId]);

  // What actually feeds ViewportGrid/Legend's existing generic uncertainty
  // props - derived from the selector above plus whichever buffers are in.
  // `heatOverlay === "entropy"` reproduces the old single-toggle behavior
  // exactly: `uncertainty` is the same buffer, `showHeat` is the same boolean
  // `showUncertainty` used to be.
  const heatBuffer: UncertaintyBuffer | null =
    heatOverlay === "entropy"
      ? uncertainty
      : heatOverlay === "band_wt"
        ? (conformalBand.WT ?? null)
        : heatOverlay === "band_tc"
          ? (conformalBand.TC ?? null)
          : heatOverlay === "gradcam_wt"
            ? (gradcam.WT ?? null)
            : heatOverlay === "gradcam_tc"
              ? (gradcam.TC ?? null)
              : null;
  const showHeat = heatOverlay !== "none" && heatBuffer !== null;

  // What feeds BrainTwinScene's tumour-mesh painting - the SAME selected
  // buffer the 2D view is heat-overlaying, never a second, independent
  // choice. `key` is `heatOverlay` itself (e.g. "band_wt" vs "band_tc")
  // rather than `heatBuffer.kind` ("conformal-band" for both): those two
  // buffers share a `kind`, so keying the scene's scalar cache on `kind`
  // alone would let switching WT<->TC silently reuse the wrong region's
  // stale scalars (see TwinActiveLayer's docstring). `kind` is read off the
  // buffer itself, never hardcoded, and is validated against the three
  // kinds scalarsToColors actually knows how to paint - an unrecognised or
  // missing header degrades to "no layer" rather than mislabelling one.
  const activeLayer: TwinActiveLayer | null = useMemo(() => {
    if (heatOverlay === "none" || !heatBuffer) return null;
    const kind = heatBuffer.kind;
    if (
      kind !== PREDICTIVE_ENTROPY_SINGLE_PASS &&
      kind !== CONFORMAL_BAND &&
      kind !== GRADCAM
    ) {
      return null;
    }
    return { key: heatOverlay, kind, data: heatBuffer.data, conformal: heatBuffer.conformal ?? null };
  }, [heatOverlay, heatBuffer]);

  // Drives the button group below. "None" is never disabled; the other
  // five are disabled exactly when their backing buffer is unavailable -
  // entropy for a job with no saved logits, a band for a region with no
  // fitted conformal threshold yet, a Grad-CAM map for a job that predates
  // this feature or whose Grad-CAM computation failed for that region - and
  // each names why, honestly.
  const heatOptions: { value: HeatOverlay; label: string; disabled: boolean; title?: string }[] = [
    { value: "none", label: "None", disabled: false },
    {
      value: "entropy",
      label: "Entropy",
      disabled: !uncertainty,
      title: !uncertainty ? "No saved logits for this case." : undefined,
    },
    {
      value: "band_wt",
      label: "Band (WT)",
      disabled: !conformalBand.WT,
      title: !conformalBand.WT ? "No fitted threshold available." : undefined,
    },
    {
      value: "band_tc",
      label: "Band (TC)",
      disabled: !conformalBand.TC,
      title: !conformalBand.TC ? "No fitted threshold available." : undefined,
    },
    {
      value: "gradcam_wt",
      label: "Grad-CAM (WT)",
      disabled: !gradcam.WT,
      title: !gradcam.WT ? "No explainability heatmap available." : undefined,
    },
    {
      value: "gradcam_tc",
      label: "Grad-CAM (TC)",
      disabled: !gradcam.TC,
      title: !gradcam.TC ? "No explainability heatmap available." : undefined,
    },
  ];

  // Whether this job has everything BrainTwinScene needs: case geometry
  // (for real-world mesh scale, from /geometry - the only clinical route
  // that carries voxel spacing) plus all four modality volumes plus the
  // predicted mask (there is never a label on this path). Tracked
  // separately from `twinInput` below, which is deliberately null whenever
  // the twin view isn't showing (a worker mesh pass is not free - same
  // reasoning as App.tsx's `twinOpen` guard) - the "3D twin" button needs to
  // know whether switching TO that view is possible before the user does.
  const twinDataReady =
    geometry != null &&
    MODALITY_ORDER.every((m) => volumes[m]?.data != null) &&
    predictionMask?.data != null;

  // Built exactly as App.tsx's own `twinInput` memo, but from this hook's
  // data. On the clinical path there is never a label, so `tumorSource` is
  // always `"prediction"`.
  // The twin is mounted (and so meshed) in "tri" and "twin" modes only.
  // `effectiveView` demotes "tri"/"twin" to "slices" while its data is not
  // ready, so the default "tri" never mounts a twin with nothing to draw.
  const effectiveView: ViewMode = twinDataReady ? view : "slices";
  const twinOn = effectiveView !== "slices";

  const twinInput: BrainTwinInput | null = useMemo(() => {
    if (!twinOn || !twinDataReady || !geometry) return null;
    return {
      caseId: jobId,
      shape: geometry.shape,
      spacing: geometry.spacing,
      modalityVolumes: MODALITY_ORDER.map((m) => volumes[m]!.data),
      tumorMask: predictionMask!.data,
      tumorSource: "prediction",
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [twinOn, twinDataReady, geometry, volumes, predictionMask, jobId]);

  // Which atlas indices get a shell in the twin: the report's own top-N
  // involved structures (by name -> index, via the atlas table) plus
  // whatever the user has added via the checkboxes below, capped at
  // MAX_ATLAS_STRUCTURES - all the actual selection logic lives in
  // selectStructures (lib/atlasSelection.ts), this is just wiring this
  // job's report/table/extra state into it.
  const atlasSelection = useMemo(
    () => selectStructures({ report: reportState.report, table: atlasTable, extra: extraStructures }),
    [reportState.report, atlasTable, extraStructures],
  );

  // The subset of atlasSelection that came from the report itself (as
  // opposed to a user-added extra) - used only to label and lock those
  // checkboxes in the "Add a structure" control below, since a report-
  // derived structure isn't something extraStructures ever removes.
  const reportDerivedIndices = useMemo(() => {
    const set = new Set<number>();
    if (!reportState.report || !atlasTable) return set;
    for (const row of reportState.report.anatomy.structures) {
      const index = structureIndexForName(atlasTable, row.structure);
      if (index !== null) set.add(index);
    }
    return set;
  }, [reportState.report, atlasTable]);

  // Toggles one atlas structure in/out of extraStructures - never touches a
  // report-derived index (its checkbox is disabled in the control below, so
  // this never actually gets called for one, but the `<= 0` / duplicate
  // guards mirror selectStructures's own so this stays safe either way).
  function toggleExtraStructure(index: number, checked: boolean) {
    setExtraStructures((prev) =>
      checked ? (prev.includes(index) ? prev : [...prev, index]) : prev.filter((i) => i !== index),
    );
  }

  // T6.4: downloads this job's full export bundle (report.json, report.md,
  // dicom-seg.dcm, job.json and, when possible, one twin snapshot).
  //
  // A failed snapshot capture must never block the rest of the bundle - the
  // report and DICOM-SEG are the export's whole point, and a blank or
  // zero-size canvas (the twin hasn't finished its first paint yet, or the
  // browser dropped the WebGL context) is a real possibility captureCanvas
  // already guards against by throwing. So a capture failure here is caught,
  // logged, and treated as "zero snapshots", never surfaced as an export
  // error to the user.
  async function handleExport() {
    setExporting(true);
    setExportError(null);
    try {
      const snapshots: { blob: Blob; filename: string }[] = [];
      if (twinOn && twinHostRef.current) {
        const canvas = findTwinCanvas(twinHostRef.current);
        if (canvas) {
          try {
            snapshots.push({ blob: captureCanvas(canvas), filename: snapshotFilename(jobId, "twin") });
          } catch (captureErr) {
            console.warn(
              "export: twin snapshot capture failed, bundling without one",
              captureErr,
            );
          }
        }
      }
      // effectiveView === "slices": the 2D viewport draws to several plane canvases,
      // not one twin canvas - snapshotting those is out of scope for this
      // first version of export, so the bundle simply carries no snapshot.
      const { blob, filename } = await exportClinicalJob(jobId, snapshots);
      triggerBlobDownload(blob, filename);
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  }

  // The twin's badge mirrors the gatekeeper's decision for this job - the
  // one piece of information from the pipeline that belongs on the 3D view
  // even though it says nothing about the segmentation's shape.
  let twinBadge: string | null = null;
  let twinBadgeTone: "caution" | "neutral" = "neutral";
  if (decision === "proceed_with_caution") {
    twinBadge = "Proceed with caution";
    twinBadgeTone = "caution";
  } else if (decision === "proceed") {
    twinBadge = "Proceed";
    twinBadgeTone = "neutral";
  }

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

  // >= 1280px (Tailwind xl): tri mode becomes a true 2x2. Tracked in JS
  // (not just CSS) because the two arrangements render different trees and
  // the twin must still be mounted exactly once.
  const [isXl, setIsXl] = useState(() =>
    typeof window === "undefined" ? true : window.matchMedia("(min-width: 1280px)").matches,
  );
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1280px)");
    const onChange = () => setIsXl(mq.matches);
    mq.addEventListener("change", onChange);
    onChange();
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const ribbonPlane: Plane = layout === "single" ? singlePlane : (expandedPlane ?? "axial");
  const ribbonLabel = ribbonPlane.charAt(0).toUpperCase() + ribbonPlane.slice(1);

  // Real per-slice tumour fraction of the predicted mask along the ribbon's
  // plane (same definition as the backend's /profile). Recomputed only when
  // the mask, its shape or the ribbon's plane changes - one pass over the
  // volume, so scrubbing slices never re-runs it. Hook order: this sits
  // above the early `error` return below on purpose.
  const maskData = predictionMask?.data ?? null;
  const maskShape = predictionMask?.shape ?? null;
  const tumorProfile = useMemo(
    () => (maskData && maskShape ? tumourProfile(maskData, maskShape, ribbonPlane) : []),
    [maskData, maskShape, ribbonPlane],
  );

  // Heat-overlay statistics, computed once per buffer (memo keyed on buffer
  // identity). Only the stats for the ACTIVE overlay's kind are computed.
  const heatKind = heatBuffer?.kind ?? null;
  const heatData = heatOverlay !== "none" ? (heatBuffer?.data ?? null) : null;
  const heatRegion: "WT" | "TC" | null = heatOverlay.endsWith("_wt")
    ? "WT"
    : heatOverlay.endsWith("_tc")
      ? "TC"
      : null;
  const entropyStatsMemo = useMemo(
    () =>
      heatData && maskData && heatKind === PREDICTIVE_ENTROPY_SINGLE_PASS
        ? entropyStats(heatData, maskData)
        : null,
    [heatData, maskData, heatKind],
  );
  const bandStatsMemo = useMemo(
    () => (heatData && heatKind === CONFORMAL_BAND ? bandStats(heatData) : null),
    [heatData, heatKind],
  );
  const gradcamStatsMemo = useMemo(
    () =>
      heatData && maskData && heatKind === GRADCAM && heatRegion
        ? gradcamStats(heatData, maskData, heatRegion)
        : null,
    [heatData, maskData, heatKind, heatRegion],
  );
  // Per-slice mean entropy for the ribbon - entropy overlay only.
  const entropyRibbon = useMemo(
    () =>
      heatData && maskShape && heatKind === PREDICTIVE_ENTROPY_SINGLE_PASS
        ? entropyProfile(heatData, maskShape, ribbonPlane)
        : null,
    [heatData, maskShape, heatKind, ribbonPlane],
  );
  // Voxel volume from the job geometry; null (-> report voxels, never an
  // assumed 1 mm) until /geometry has arrived.
  const voxelMm3 = geometry ? geometry.spacing[0] * geometry.spacing[1] * geometry.spacing[2] : null;

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center text-center">
        <p className="text-sm text-text-primary">{error}</p>
      </div>
    );
  }

  const segBtn = (active: boolean, disabled = false) =>
    `flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-[13px] font-medium transition-colors duration-[120ms] ${
      disabled
        ? "cursor-not-allowed text-text-dim"
        : active
          ? "bg-brand-primary/20 text-text-primary ring-1 ring-brand-primary/40"
          : "text-text-secondary hover:text-text-primary"
    }`;
  const segGroup =
    "inline-flex shrink-0 flex-wrap min-[1800px]:flex-nowrap rounded-lg border border-surface-seam bg-surface-raised/50 p-0.5";

  const slicesGrid = (
    <ViewportGrid
      layout={layout}
      expandedPlane={expandedPlane}
      onToggleExpand={(plane) => setExpandedPlane((prev) => (prev === plane ? null : plane))}
      singlePlane={singlePlane}
      onChangeSinglePlane={setSinglePlane}
      onFocusPlane={setFocusedPlane}
      sliceIndices={sliceIndices}
      planeCounts={planeCounts}
      shape={shape}
      image={volumes[modality]?.data ?? null}
      predictionMask={predictionMask?.data ?? null}
      labelMask={null}
      uncertainty={heatBuffer?.data ?? null}
      overlayMode="prediction"
      overlayOpacity={overlayOpacity}
      showTruthOutline={false}
      showUncertainty={showHeat}
      uncertaintyOpacity={showHeat ? UNCERTAINTY_OPACITY : 0}
    />
  );

  // One plane tile of the 2x2: the same Viewport, with the same props
  // ViewportGrid hands it, so slicing/overlay behaviour is identical. The
  // expand button leaves tri mode for "Slices" with this plane expanded.
  const planeTile = (plane: Plane, label: string) => (
    <div key={plane} className="min-h-0">
      <Viewport
        plane={plane}
        planeLabel={label}
        sliceIndex={sliceIndices[plane]}
        sliceCount={planeCounts[plane]}
        expanded={false}
        expandable
        onToggleExpand={() => {
          setExpandedPlane(plane);
          setView("slices");
        }}
        onFocusPlane={() => setFocusedPlane(plane)}
        shape={shape}
        image={volumes[modality]?.data ?? null}
        predictionMask={predictionMask?.data ?? null}
        labelMask={null}
        uncertainty={heatBuffer?.data ?? null}
        overlayMode="prediction"
        overlayOpacity={overlayOpacity}
        showTruthOutline={false}
        showUncertainty={showHeat}
        uncertaintyOpacity={showHeat ? UNCERTAINTY_OPACITY : 0}
      />
    </div>
  );

  // The one and only BrainTwinScene. It is rendered in exactly one place per
  // mode (inside `twinTile` below), so there is never a second WebGL context.
  const twinTile = (
    <div
      ref={twinHostRef}
      className="h-full min-h-0 overflow-hidden rounded-[10px] border border-surface-seam bg-surface-panel"
    >
      <BrainTwinScene
        input={twinInput}
        badge={twinBadge}
        badgeTone={twinBadgeTone}
        activeLayer={activeLayer}
        atlas={
          atlas && atlasTable && atlasSelection.length > 0
            ? { volume: atlas.data, selection: atlasSelection, table: atlasTable }
            : null
        }
        highlightedStructure={highlightedStructure}
        onStructureSelect={setHighlightedStructure}
        structureDetail={(index) => reportRowForStructure(reportState.report, atlasTable, index)}
      />
    </div>
  );

  return (
    <div className="flex flex-none flex-col lg:min-h-0 lg:flex-1">
      <div className="glass-panel relative z-20 mx-2 mt-2 flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2 min-[1800px]:flex-nowrap">
        <div className={segGroup} role="group" aria-label="View">
          <button
            type="button"
            disabled={!twinDataReady}
            onClick={() => setView("tri")}
            aria-pressed={effectiveView === "tri"}
            title={!twinDataReady ? "Waiting for all four volumes and the mask…" : undefined}
            className={segBtn(effectiveView === "tri", !twinDataReady)}
          >
            Tri-plane + twin
          </button>
          <button
            type="button"
            onClick={() => setView("slices")}
            aria-pressed={effectiveView === "slices"}
            className={segBtn(effectiveView === "slices")}
          >
            Slices
          </button>
          <button
            type="button"
            data-testid="clinical-view-twin"
            disabled={!twinDataReady}
            onClick={() => setView("twin")}
            aria-pressed={effectiveView === "twin"}
            title={!twinDataReady ? "Waiting for all four volumes and the mask…" : undefined}
            className={segBtn(effectiveView === "twin", !twinDataReady)}
          >
            <Box size={13} aria-hidden="true" />
            3D twin
          </button>
        </div>

        <div className={segGroup} role="group" aria-label="Modality">
          {MODALITY_ORDER.map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setModality(m)}
              aria-pressed={modality === m}
              className={segBtn(modality === m)}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>

        <div className={segGroup} role="group" aria-label="Heat overlay">
          {heatOptions.map((opt) => (
            <button
              key={opt.value}
              type="button"
              disabled={opt.disabled}
              onClick={() => setHeatOverlay(opt.value)}
              aria-pressed={heatOverlay === opt.value}
              title={opt.title}
              className={segBtn(heatOverlay === opt.value, opt.disabled)}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="eyebrow shrink-0">Opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            className="w-24 accent-brand-primary"
            aria-label="Overlay opacity"
          />
          <span className="tabular w-9 shrink-0 font-mono text-xs text-text-secondary">
            {overlayOpacity.toFixed(2)}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 min-[1800px]:ml-auto min-[1800px]:flex-nowrap">
          {/* "Add a structure": only meaningful while the twin is on screen
              (the 2D slices never draw atlas shells), and only once the
              atlas table has loaded - a native <details> keeps this a
              zero-JS popover that closes itself on an outside click, same
              as a browser <select>. Copy stays strictly geometric
              (name/lobe/laterality/"from report"/a count) - never a grade,
              stage, prognosis, deficit or impairment word. */}
          {twinOn && atlasTable && (
            <details className="relative shrink-0">
              <summary
                data-testid="clinical-atlas-structures-toggle"
                className="cursor-pointer list-none rounded-lg border border-surface-seam bg-surface-raised/50 px-2.5 py-1 text-[13px] font-medium text-text-secondary transition-colors duration-[120ms] hover:text-text-primary"
              >
                Structures · {atlasSelection.length} shown
              </summary>
              <div className="liquid-glass absolute top-full right-0 z-30 mt-1 max-h-64 w-64 overflow-y-auto rounded-md border border-surface-seam p-2">
                <div className="flex flex-col gap-1">
                  {[...atlasTable]
                    .sort((a, b) => a.name.localeCompare(b.name))
                    .map((row) => {
                      const checked = atlasSelection.includes(row.index);
                      const fromReport = reportDerivedIndices.has(row.index);
                      const atCap = !checked && atlasSelection.length >= MAX_ATLAS_STRUCTURES;
                      const disabled = fromReport || atCap;
                      return (
                        <label
                          key={row.index}
                          title={atCap ? "16 structures max" : undefined}
                          className="flex items-center gap-1.5 text-xs text-text-secondary"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            disabled={disabled}
                            onChange={(e) => toggleExtraStructure(row.index, e.target.checked)}
                            className="accent-brand-primary"
                          />
                          <span className="truncate font-mono">{row.name}</span>
                          {fromReport && (
                            <span className="ml-auto shrink-0 text-text-dim">from report</span>
                          )}
                        </label>
                      );
                    })}
                </div>
              </div>
            </details>
          )}

          <button
            type="button"
            data-testid="clinical-report-toggle"
            disabled={reportState.status === "not_found"}
            onClick={() => setReportOpen((v) => !v)}
            aria-pressed={reportOpen}
            title={
              reportState.status === "not_found"
                ? "No report was generated for this job."
                : undefined
            }
            className="btn-secondary !px-3 !py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-40"
          >
            <FileText size={13} aria-hidden="true" />
            Report
          </button>

          <button
            type="button"
            data-testid="clinical-export"
            disabled={exporting}
            onClick={handleExport}
            title="Download report.json, report.md, DICOM-SEG and a twin snapshot as one zip"
            className="btn-secondary !px-3 !py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Download size={13} aria-hidden="true" />
            {exporting ? "Exporting…" : "Export"}
          </button>
          {exportError && (
            <span role="alert" className="text-xs text-gate-caution">
              {exportError}
            </span>
          )}
        </div>
      </div>

      <div className="relative flex flex-none flex-col lg:min-h-0 lg:flex-1 lg:flex-row">
        <div className="bg-grid flex flex-none flex-col gap-2 p-3 lg:min-h-0 lg:flex-1 lg:overflow-hidden">
          {decision === "proceed_with_caution" && (
            <div
              data-testid="clinical-caution-strip"
              role="status"
              className="flex shrink-0 items-center gap-2 rounded-lg border border-gate-caution/50 bg-gate-caution/10 px-3 py-1.5 text-gate-caution"
            >
              <span className="font-condensed text-xs">
                Gatekeeper: proceed with caution — read the report's not_claimed list before
                relying on it
              </span>
            </div>
          )}
          {loading && (
            <div
              className="flex shrink-0 items-center gap-2 rounded-lg border border-surface-seam bg-surface-panel px-3 py-1.5"
              role="status"
              aria-live="polite"
            >
              <span className="font-condensed text-xs text-text-dim">
                Loading segmentation…
              </span>
            </div>
          )}
          {warnings.length > 0 && (
            <div
              data-testid="clinical-layer-warnings"
              role="status"
              className="flex shrink-0 flex-col gap-0.5 rounded-lg border border-surface-seam bg-surface-panel px-3 py-1.5"
            >
              {warnings.map((warning) => (
                <span key={warning} className="text-xs text-text-dim">
                  Layer unavailable — {warning}
                </span>
              ))}
            </div>
          )}

          {effectiveView === "tri" && isXl ? (
            // True 2x2: axial | coronal over sagittal | twin.
            <div className="grid h-full min-h-0 flex-1 grid-cols-2 grid-rows-2 gap-3">
              {planeTile("axial", "Axial")}
              {planeTile("coronal", "Coronal")}
              {planeTile("sagittal", "Sagittal")}
              <div className="min-h-0">{twinTile}</div>
            </div>
          ) : effectiveView === "tri" ? (
            // Below xl: three plane tiles (ViewportGrid, unmodified: 3 columns on wide
            // screens, stacked below) on the left two thirds; twin as the
            // fourth tile on the right third. Stacks vertically below lg.
            <div className="grid flex-none grid-cols-1 gap-2 lg:min-h-0 lg:flex-1 lg:grid-cols-3">
              <div className="h-[50vh] min-h-[280px] md:h-[70vh] md:min-h-[360px] lg:col-span-2 lg:h-auto lg:min-h-0">{slicesGrid}</div>
              <div className="h-[50vh] min-h-[280px] lg:h-auto lg:min-h-0">{twinTile}</div>
            </div>
          ) : effectiveView === "twin" ? (
            <div className="h-[70vh] min-h-[360px] flex-none lg:h-auto lg:min-h-0 lg:flex-1">{twinTile}</div>
          ) : (
            <div className="h-[70vh] min-h-[360px] flex-none lg:h-auto lg:min-h-0 lg:flex-1">{slicesGrid}</div>
          )}

          <div className="shrink-0">
            <SliceRibbon
              planeLabel={ribbonLabel}
              sliceCount={planeCounts[ribbonPlane]}
              currentIndex={sliceIndices[ribbonPlane]}
              onScrub={(i) => setSliceIndices((prev) => ({ ...prev, [ribbonPlane]: i }))}
              tumor={tumorProfile}
              error={null}
              entropy={entropyRibbon}
              onFocusRibbon={() => setFocusedPlane(ribbonPlane)}
            />
          </div>
        </div>

        <aside
          aria-label="Study summary"
          className="flex w-full shrink-0 flex-col gap-3 overflow-auto border-t border-surface-seam p-3 lg:w-80 lg:border-t-0 lg:border-l"
        >
          <Legend
            overlayMode="prediction"
            showUncertainty={showHeat}
            hasLabel={false}
            uncertaintyKind={heatBuffer?.kind ?? null}
            conformal={heatBuffer?.conformal ?? null}
          />
          <StudySidebar
            uncertaintySlot={
              showHeat ? (
                <UncertaintyCard
                  kind={heatKind}
                  region={heatRegion}
                  entropy={entropyStatsMemo}
                  band={bandStatsMemo}
                  conformal={heatBuffer?.conformal ?? null}
                  gradcam={gradcamStatsMemo}
                  voxelMm3={voxelMm3}
                />
              ) : null
            }
            report={reportState.report}
            reportStatus={reportState.status}
            decision={decision}
            jobId={jobId}
            onHoverStructure={(name) =>
              setHighlightedStructure(name ? structureIndexForName(atlasTable, name) : null)
            }
            highlightedStructureName={
              highlightedStructure !== null
                ? (structureRow(atlasTable, highlightedStructure)?.name ?? null)
                : null
            }
            onOpenReport={() => setReportOpen(true)}
            onExport={handleExport}
            exporting={exporting}
            exportError={exportError}
          />
        </aside>

        <ReportPanel
          open={reportOpen}
          onClose={() => setReportOpen(false)}
          layout={layout}
          caseId={jobId}
          status={reportState.status}
          report={reportState.report}
          errorMessage={reportState.errorMessage}
          pathologyJobId={jobId}
          // The report drawer overlays the viewport, but highlightedStructure
          // is lifted to this component and fed to BrainTwinScene regardless
          // of which view is showing - so hovering a row here still lights
          // the shell even while the twin sits behind the drawer. A hovered
          // name with no match in atlasTable (version mismatch, or the
          // table hasn't loaded yet) clears the highlight rather than
          // guessing.
          onHoverStructure={(name) =>
            setHighlightedStructure(name ? structureIndexForName(atlasTable, name) : null)
          }
          highlightedStructureName={
            highlightedStructure !== null ? (structureRow(atlasTable, highlightedStructure)?.name ?? null) : null
          }
        />
      </div>
    </div>
  );
}
