import { useEffect, useMemo, useState } from "react";
import { Brain, Layers, PanelLeftOpen } from "lucide-react";
import {
  ApiUnreachableError,
  getCases,
  getHealth,
  type CaseSummary,
  type HealthResponse,
  type Modality,
  type Plane,
} from "./api";
import type { OverlayMode } from "./lib/render";
import { opaque } from "./lib/opaque";
import { navigateTo } from "./lib/navigate";
import { dataForSelection } from "./lib/caseConsistency";
import { splitLabel } from "./lib/splitLabel";
import { prefetchCase, useCaseData, type CaseDataState } from "./hooks/useCaseData";
import { useResponsiveLayout } from "./hooks/useResponsiveLayout";
import { AppShell } from "./components/AppShell";
import { CaseList } from "./components/CaseList";
import { ViewportGrid } from "./components/ViewportGrid";
import { SliceRibbon } from "./components/SliceRibbon";
import { MetricsPanel } from "./components/MetricsPanel";
import { Legend } from "./components/Legend";
import { ControlBar, MODALITY_ORDER } from "./components/ControlBar";
import { BrainTwinScene, type BrainTwinInput } from "./components/BrainTwinScene";

// The author asked for five cases with distinct tumour sizes; the server picks
// them (evenly spaced ranks by ground-truth whole-tumour volume).
const SHOWCASE_CASE_COUNT = 5;
const UNCERTAINTY_OPACITY = 0.6;
const ZERO_PLANES: Record<Plane, number> = { sagittal: 0, coronal: 0, axial: 0 };
// Module-level so this is the *same* array reference across renders - a
// fresh `[]` on every render (while the profile is still loading) would
// re-trigger the ribbon's draw effect on unrelated re-renders, e.g. dragging
// the opacity slider.
const EMPTY_PROFILE: number[] = [];
// What the UI sees while the loaded data does not belong to the selected case.
const EMPTY_SHOWN: Pick<
  CaseDataState,
  "detail" | "volumes" | "predictionMask" | "labelMask" | "uncertainty" | "profile"
> = {
  detail: null,
  volumes: {},
  predictionMask: null,
  labelMask: null,
  uncertainty: null,
  profile: null,
};

type BootState = "loading" | "ready" | "unreachable" | "error";

export default function App() {
  const [bootState, setBootState] = useState<BootState>("loading");
  const [bootError, setBootError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [casesTotal, setCasesTotal] = useState<number | undefined>(undefined);
  const [casesSelection, setCasesSelection] = useState<string | undefined>(undefined);
  // Initialised from the URL so a link like /app?case=BraTS2021_00123 (e.g.
  // the report page's "back to viewer" link) reopens on that case rather
  // than the empty "pick a case" state.
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get("case"),
  );
  const [caseListOpen, setCaseListOpen] = useState(false);
  const [caseListCollapsed, setCaseListCollapsed] = useState(false);

  const [modality, setModality] = useState<Modality>("t1ce");
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("prediction");
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [showTruthOutline, setShowTruthOutline] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(false);

  const [expandedPlane, setExpandedPlane] = useState<Plane | null>(null);
  const [singlePlane, setSinglePlane] = useState<Plane>("axial");
  const [focusedPlane, setFocusedPlane] = useState<Plane>("axial");
  const [sliceIndices, setSliceIndices] = useState<Record<Plane, number>>(ZERO_PLANES);

  // The 3D twin is the default view now - the flat viewport grid is one tab
  // switch away, not the initial state. Deliberately NOT reset on case
  // change: a user who switched to the scan view stays there while browsing
  // cases (see the view-switch toolbar below).
  const [twinOpen, setTwinOpen] = useState(true);

  const { layout, isPanelWidth } = useResponsiveLayout();
  const caseData = useCaseData(selectedCaseId);
  // useCaseData keeps the PREVIOUS case's data while the next one loads. Only
  // data that is literally the selected case's may be drawn or labelled with
  // its id; `shown` is the hook state when it matches, else empty.
  const { matches, stale } = dataForSelection(selectedCaseId, caseData);
  const shown = matches ? caseData : EMPTY_SHOWN;

  // Bootstrap: health + case list.
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const [healthRes, casesRes] = await Promise.all([
          getHealth(controller.signal),
          getCases(controller.signal, SHOWCASE_CASE_COUNT),
        ]);
        if (cancelled) return;
        setHealth(healthRes);
        setCases(casesRes.cases);
        setCasesTotal(casesRes.total);
        setCasesSelection(casesRes.selection);
        setBootState("ready");
      } catch (err) {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === "AbortError") return;
        if (err instanceof ApiUnreachableError) {
          setBootState("unreachable");
        } else {
          setBootError(err instanceof Error ? err.message : "Failed to load.");
          setBootState("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  // After the selected case has finished loading, warm the cache with the
  // other showcase cases, one at a time, whenever the browser is idle. Memory
  // budget: 5 cases x ~26 MB ~= 130 MB (see useCaseData's cache). Stops on
  // unmount; a prefetch already on the wire is left to finish (it only fills
  // the cache, never React state).
  const selectedLoaded = selectedCaseId !== null && !caseData.loading && matches;
  useEffect(() => {
    if (!selectedLoaded || cases.length === 0) return;
    let stopped = false;
    let handle: number | undefined;
    const usingIdle = typeof window.requestIdleCallback === "function";
    const schedule = (fn: () => void) => {
      handle = usingIdle
        ? window.requestIdleCallback(fn)
        : window.setTimeout(fn, 200);
    };
    const queue = cases.map((c) => c.case_id).filter((id) => id !== selectedCaseId);
    const next = () => {
      if (stopped) return;
      const id = queue.shift();
      if (id === undefined) return;
      prefetchCase(id).then(() => {
        if (!stopped) schedule(next);
      });
    };
    schedule(next);
    return () => {
      stopped = true;
      if (handle !== undefined) {
        if (usingIdle) window.cancelIdleCallback(handle);
        else window.clearTimeout(handle);
      }
    };
  }, [selectedLoaded, selectedCaseId, cases]);

  // A case opened by deep link that the showcase does not include: still
  // load it (fetch is by id) and list it as an extra "linked" row.
  const linkedCase: CaseSummary | null = useMemo(() => {
    if (!selectedCaseId || cases.length === 0) return null;
    if (cases.some((c) => c.case_id === selectedCaseId)) return null;
    const d = caseData.detail;
    const same = d !== null && d.meta.case_id === selectedCaseId;
    return {
      case_id: selectedCaseId,
      dice_mean: same ? (d.metrics?.dice_mean ?? null) : null,
      dice: null,
      has_label: same ? d.meta.has_label : false,
      has_logits: same ? d.meta.has_logits : false,
      has_report: same ? d.has_report : false,
      wt_volume_ml: same ? d.regions.label?.WT.ml : undefined,
    };
  }, [selectedCaseId, cases, caseData.detail]);

  // Reset per-case view state whenever a new case finishes loading its meta.
  useEffect(() => {
    if (!caseData.detail) return;
    const { sagittal, coronal, axial } = caseData.detail.meta.planes;
    setSliceIndices({
      sagittal: Math.floor(sagittal / 2),
      coronal: Math.floor(coronal / 2),
      axial: Math.floor(axial / 2),
    });
    setExpandedPlane(null);
    if (!caseData.detail.meta.has_label) setOverlayMode("prediction");
    // Prediction and Disagreement overlays both need a saved prediction;
    // fall back to Truth (if available) rather than leaving the mode on a
    // now-disabled control.
    if (!caseData.detail.meta.has_prediction && caseData.detail.meta.has_label) {
      setOverlayMode("truth");
    }
    if (!caseData.detail.meta.has_logits) setShowUncertainty(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseData.detail?.meta.case_id]);

  // Keep the URL in sync with the selected case so it is bookmarkable and so
  // the report page (a separate route now, not a panel) can link back to
  // this exact case via /app?case=<id>. replaceState, not pushState: browsing
  // cases is one continuous session, not a sequence of back-button stops.
  useEffect(() => {
    const next = selectedCaseId
      ? `/app?case=${encodeURIComponent(selectedCaseId)}`
      : "/app";
    window.history.replaceState({}, "", next);
  }, [selectedCaseId]);

  // Global slice-stepping and modality shortcuts, scoped to the last-focused viewport.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (!shown.detail) return;

      if (e.key >= "1" && e.key <= "4") {
        const m = MODALITY_ORDER[parseInt(e.key, 10) - 1];
        if (m) setModality(m);
        return;
      }

      const activePlane = layout === "single" ? singlePlane : (expandedPlane ?? focusedPlane);
      const count = shown.detail.meta.planes[activePlane];
      if (count === undefined) return;

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
  }, [shown.detail, layout, singlePlane, expandedPlane, focusedPlane]);

  if (bootState === "unreachable") {
    return (
      <div className="flex h-screen items-center justify-center bg-surface-page px-6 text-center">
        <div className="max-w-md">
          <p className="mb-2 font-condensed text-sm text-text-dim">
            NeuroVision-X
          </p>
          <p className="text-sm text-text-primary">
            No response from the API. Start it with{" "}
            <code className="text-data-oedema">uvicorn app.backend.main:app --reload</code>.
          </p>
        </div>
      </div>
    );
  }

  if (bootState === "error") {
    return (
      <div className="flex h-screen items-center justify-center bg-surface-page px-6 text-center">
        <div className="max-w-md">
          <p className="mb-2 font-condensed text-sm text-text-dim">
            NeuroVision-X
          </p>
          <p className="mb-2 text-sm text-text-primary">
            The API responded, but not with what the viewer expected.
          </p>
          <p className="text-xs text-text-secondary">{bootError}</p>
          <p className="mt-3 text-xs text-text-dim">
            Check the paths the server resolved at <code>/api/health</code>.
          </p>
        </div>
      </div>
    );
  }

  const detail = shown.detail;
  const planeCounts = detail?.meta.planes ?? ZERO_PLANES;
  const shape = detail?.meta.shape ?? null;
  const hasLabel = detail?.meta.has_label ?? false;
  const hasLogits = detail?.meta.has_logits ?? false;
  const hasPrediction = detail?.meta.has_prediction ?? false;
  const hasReport = detail?.has_report ?? false;
  const uncertaintyKind = shown.uncertainty?.kind ?? null;

  // Fraction of this case's artifacts that have arrived. Counted against what
  // the case ACTUALLY has -- a case with no label or no logits must still be
  // able to reach 100%, or the bar would stall short of the end and look like
  // a failed load.
  const expectedArtifacts =
    4 + 1 + (hasLabel ? 1 : 0) + (hasLogits ? 1 : 0); // modalities + profile + label + logits
  const loadedArtifacts =
    Object.keys(shown.volumes).length +
    (shown.profile ? 1 : 0) +
    (shown.labelMask ? 1 : 0) +
    (shown.uncertainty ? 1 : 0);
  const loadProgress = Math.min(1, loadedArtifacts / expectedArtifacts);
  const ribbonPlane: Plane =
    layout === "single" ? singlePlane : (expandedPlane ?? "axial");
  const ribbonLabel = ribbonPlane.charAt(0).toUpperCase() + ribbonPlane.slice(1);
  const profilePlane = shown.profile?.planes[ribbonPlane];
  const showCaseListInline = isPanelWidth;

  // Built only while the twin view is actually open (a worker pass over a
  // full volume is not free) and only once every modality has arrived, so
  // the brain shell it produces reflects the whole case, not a partial one.
  const twinInput: BrainTwinInput | null = useMemo(() => {
    if (!twinOpen || !matches || !detail) return null;
    const modalityVolumes = Object.values(shown.volumes)
      .map((v) => v?.data)
      .filter((d): d is Uint8Array => d != null);
    if (modalityVolumes.length < 4) return null;
    const tumorMask = shown.labelMask?.data ?? shown.predictionMask?.data ?? null;
    const tumorSource: "label" | "prediction" | null = shown.labelMask
      ? "label"
      : shown.predictionMask
        ? "prediction"
        : null;
    return {
      caseId: detail.meta.case_id, // the data's own id, never the selection
      shape: detail.meta.shape,
      spacing: detail.meta.spacing,
      modalityVolumes: opaque(modalityVolumes),
      tumorMask: opaque(tumorMask),
      tumorSource,
    };
  }, [
    twinOpen,
    matches,
    detail,
    shown.volumes,
    shown.labelMask,
    shown.predictionMask,
  ]);

  // Header's info now lives in AppShell's toolbar slot as chips. The split
  // label comes from the eval directory actually in use (see lib/splitLabel).
  const toolbar = (
    <>
      {!showCaseListInline && (
        <button
          type="button"
          onClick={() => setCaseListOpen((v) => !v)}
          className="btn-secondary !px-3 !py-1 text-xs"
        >
          Cases
        </button>
      )}
      {health ? (
        <>
          <span className="chip">{splitLabel(health.eval_dir)}</span>
        </>
      ) : null}
    </>
  );

  const viewBtn = (active: boolean) =>
    `flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-[13px] font-medium transition-colors duration-[120ms] ${
      active
        ? "bg-brand-primary/20 text-text-primary ring-1 ring-brand-primary/40"
        : "text-text-secondary hover:text-text-primary"
    }`;

  return (
    <AppShell fullHeight toolbar={toolbar}>
      {/* Below the single-viewport breakpoint the readout moves BELOW the
          image instead of beside it: at ~600px a 224px sidebar eats a third
          of the width, and the MRI is the thing worth the pixels. */}
      <div className="relative flex h-full min-h-0 flex-row">
        {showCaseListInline && !caseListCollapsed && (
          <div className="m-3 mr-0 w-64 shrink-0 overflow-hidden glass-panel">
            <CaseList
              cases={cases}
              total={casesTotal}
              selection={casesSelection}
              linkedCase={linkedCase}
              selectedCaseId={selectedCaseId}
              onSelect={(id) => {
                setSelectedCaseId(id);
              }}
              collapsible
              onCollapse={() => setCaseListCollapsed(true)}
            />
          </div>
        )}

        {showCaseListInline && caseListCollapsed && (
          <div className="m-3 mr-0 flex w-9 shrink-0 flex-col items-center self-start glass-panel py-2">
            <button
              type="button"
              onClick={() => setCaseListCollapsed(false)}
              aria-label="Show cases"
              title="Show cases"
              className="rounded-sm p-1 text-text-secondary transition-colors duration-[120ms] hover:text-text-primary"
            >
              <PanelLeftOpen size={14} aria-hidden="true" />
            </button>
          </div>
        )}

        {!showCaseListInline && caseListOpen && (
          <>
            <div
              className="absolute inset-0 z-10 bg-black/60"
              onClick={() => setCaseListOpen(false)}
              aria-hidden="true"
            />
            <div className="absolute inset-y-0 left-0 z-20 w-64 overflow-hidden border-r border-surface-seam glass-panel">
              <CaseList
                cases={cases}
                total={casesTotal}
                selection={casesSelection}
                linkedCase={linkedCase}
                selectedCaseId={selectedCaseId}
                onSelect={(id) => {
                  setSelectedCaseId(id);
                  setCaseListOpen(false);
                }}
              />
            </div>
          </>
        )}

        {/* Centre stage + right column. Side by side from lg; below lg the
            right column drops under the stage (capped at 40vh, scrolling). */}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
          <div className="flex min-w-0 flex-none flex-col gap-3 p-3 lg:min-h-0 lg:flex-1 lg:overflow-hidden">
            {!selectedCaseId ? (
              <div className="flex flex-1 items-center justify-center text-center">
                <div>
                  <p className="text-sm text-text-primary">Pick a case to begin.</p>
                  {health && (
                    <p className="mt-1 text-xs text-text-dim">
                      Evaluation directory: {health.eval_dir}
                    </p>
                  )}
                </div>
              </div>
            ) : caseData.error ? (
              <div className="flex flex-1 items-center justify-center text-center">
                <p className="text-sm text-text-primary">{caseData.error}</p>
              </div>
            ) : (
              <>
                {/* Toolbar strip on top: view switch + case id, then the
                    ControlBar (modality / overlay / opacity / entropy /
                    report) with its props unchanged. */}
                <div className="glass-panel relative z-20 flex shrink-0 flex-col">
                  <div className="flex flex-wrap items-center gap-2 px-3 py-2">
                    <span className="font-mono text-xs text-text-primary">
                      {matches ? shown.detail?.meta.case_id : "Loading…"}
                    </span>
                    <div
                      role="group"
                      aria-label="View"
                      className="inline-flex shrink-0 rounded-lg border border-surface-seam bg-surface-raised/50 p-0.5"
                    >
                      <button
                        type="button"
                        onClick={() => setTwinOpen(true)}
                        aria-pressed={twinOpen}
                        className={viewBtn(twinOpen)}
                      >
                        <Brain size={13} aria-hidden="true" />
                        3D twin
                      </button>
                      <button
                        type="button"
                        onClick={() => setTwinOpen(false)}
                        aria-pressed={!twinOpen}
                        className={viewBtn(!twinOpen)}
                      >
                        <Layers size={13} aria-hidden="true" />
                        Scan view
                      </button>
                    </div>
                  </div>
                  <div className="border-t border-surface-seam [&>div]:border-t-0 [&>div]:bg-transparent">
                    <ControlBar
                      modality={modality}
                      onChangeModality={setModality}
                      overlayMode={overlayMode}
                      onChangeOverlayMode={setOverlayMode}
                      hasLabel={hasLabel}
                      hasPrediction={hasPrediction}
                      showTruthOutline={showTruthOutline}
                      onToggleTruthOutline={() => setShowTruthOutline((v) => !v)}
                      overlayOpacity={overlayOpacity}
                      onChangeOverlayOpacity={setOverlayOpacity}
                      hasLogits={hasLogits}
                      showUncertainty={showUncertainty}
                      onToggleUncertainty={() => setShowUncertainty((v) => !v)}
                      hasReport={hasReport}
                      onOpenReport={() =>
                        matches && selectedCaseId && navigateTo(`/report/${encodeURIComponent(selectedCaseId)}`)
                      }
                    />
                  </div>
                </div>

                {/* A case pulls four modality volumes plus masks, entropy and the
                    profile - around 20 MB. Without this the viewports sit black
                    for several seconds and the app reads as frozen. Determinate,
                    because we know exactly how many artifacts are outstanding. */}
                {(caseData.loading || stale) && (
                  <div
                    className="flex shrink-0 items-center gap-3 glass-panel px-3 py-1.5"
                    role="status"
                    aria-live="polite"
                  >
                    <span className="font-condensed text-[11px] text-text-dim">
                      Loading {selectedCaseId}
                    </span>
                    <span className="h-px flex-1 bg-surface-seam">
                      <span
                        className="block h-px bg-text-secondary transition-[width] duration-[120ms]"
                        style={{ width: `${Math.round(loadProgress * 100)}%` }}
                      />
                    </span>
                    <span className="tabular font-mono text-[11px] text-text-dim">
                      {Math.round(loadProgress * 100)}%
                    </span>
                  </div>
                )}
                {twinOpen ? (
                  <div className="bg-grid relative h-[70vh] min-h-[360px] flex-none overflow-hidden glass-panel lg:h-auto lg:min-h-0 lg:flex-1">
                    <BrainTwinScene input={twinInput} />
                    {/* BrainTwinScene's own empty state ("Pick a case to build
                        its twin.") is meant for the no-case-selected moment,
                        which can no longer reach it now that the twin is the
                        default view - a case IS selected here, just still
                        pulling its volumes, so this overlay says that instead. */}
                    {twinInput === null && (
                      <div className="absolute inset-0 flex items-center justify-center glass-panel">
                        <span className="text-xs text-text-secondary">
                          Loading {selectedCaseId}…
                        </span>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <div className="bg-grid h-[70vh] min-h-[360px] flex-none overflow-hidden rounded-[10px] lg:h-auto lg:min-h-0 lg:flex-1">
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
                        image={opaque(shown.volumes[modality]?.data)}
                        predictionMask={opaque(shown.predictionMask?.data)}
                        labelMask={opaque(shown.labelMask?.data)}
                        uncertainty={opaque(shown.uncertainty?.data)}
                        overlayMode={overlayMode}
                        overlayOpacity={overlayOpacity}
                        showTruthOutline={showTruthOutline}
                        showUncertainty={showUncertainty}
                        uncertaintyOpacity={UNCERTAINTY_OPACITY}
                      />
                    </div>
                    <div className="shrink-0">
                      <SliceRibbon
                        planeLabel={ribbonLabel}
                        sliceCount={planeCounts[ribbonPlane]}
                        currentIndex={sliceIndices[ribbonPlane]}
                        onScrub={(i) =>
                          setSliceIndices((prev) => ({ ...prev, [ribbonPlane]: i }))
                        }
                        tumor={profilePlane?.tumor ?? EMPTY_PROFILE}
                        error={profilePlane?.error ?? null}
                        entropy={profilePlane?.entropy ?? null}
                        onFocusRibbon={() => setFocusedPlane(ribbonPlane)}
                      />
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          {selectedCaseId && !caseData.error && (
            <aside
              aria-label="Case summary"
              className="flex w-full shrink-0 flex-col gap-3 overflow-auto border-t border-surface-seam p-3 lg:w-80 lg:border-t-0 lg:border-l"
            >
              <div className="glass-panel shrink-0">
                <MetricsPanel
                  metrics={shown.detail?.metrics ?? null}
                  regions={shown.detail?.regions ?? null}
                />
              </div>
              <div className="glass-panel shrink-0">
                <Legend
                  overlayMode={overlayMode}
                  showUncertainty={showUncertainty}
                  hasLabel={hasLabel}
                  uncertaintyKind={uncertaintyKind}
                />
              </div>
            </aside>
          )}
        </div>
      </div>
    </AppShell>
  );
}
