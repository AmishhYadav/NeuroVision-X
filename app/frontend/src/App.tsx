import { useEffect, useState } from "react";
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
import { useCaseData } from "./hooks/useCaseData";
import { useResponsiveLayout } from "./hooks/useResponsiveLayout";
import { Header } from "./components/Header";
import { CaseList } from "./components/CaseList";
import { ViewportGrid } from "./components/ViewportGrid";
import { SliceRibbon } from "./components/SliceRibbon";
import { MetricsPanel } from "./components/MetricsPanel";
import { Legend } from "./components/Legend";
import { ControlBar, MODALITY_ORDER } from "./components/ControlBar";

const UNCERTAINTY_OPACITY = 0.6;
const ZERO_PLANES: Record<Plane, number> = { sagittal: 0, coronal: 0, axial: 0 };
// Module-level so this is the *same* array reference across renders - a
// fresh `[]` on every render (while the profile is still loading) would
// re-trigger the ribbon's draw effect on unrelated re-renders, e.g. dragging
// the opacity slider.
const EMPTY_PROFILE: number[] = [];

type BootState = "loading" | "ready" | "unreachable" | "error";

export default function App() {
  const [bootState, setBootState] = useState<BootState>("loading");
  const [bootError, setBootError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [caseListOpen, setCaseListOpen] = useState(false);

  const [modality, setModality] = useState<Modality>("t1ce");
  const [overlayMode, setOverlayMode] = useState<OverlayMode>("prediction");
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const [showTruthOutline, setShowTruthOutline] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(false);

  const [expandedPlane, setExpandedPlane] = useState<Plane | null>(null);
  const [singlePlane, setSinglePlane] = useState<Plane>("axial");
  const [focusedPlane, setFocusedPlane] = useState<Plane>("axial");
  const [sliceIndices, setSliceIndices] = useState<Record<Plane, number>>(ZERO_PLANES);

  const { layout, isPanelWidth } = useResponsiveLayout();
  const caseData = useCaseData(selectedCaseId);

  // Bootstrap: health + case list.
  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    (async () => {
      try {
        const [healthRes, casesRes] = await Promise.all([
          getHealth(controller.signal),
          getCases(controller.signal),
        ]);
        if (cancelled) return;
        setHealth(healthRes);
        setCases(casesRes.cases);
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

  // Global slice-stepping and modality shortcuts, scoped to the last-focused viewport.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (!caseData.detail) return;

      if (e.key >= "1" && e.key <= "4") {
        const m = MODALITY_ORDER[parseInt(e.key, 10) - 1];
        if (m) setModality(m);
        return;
      }

      const activePlane = layout === "single" ? singlePlane : (expandedPlane ?? focusedPlane);
      const count = caseData.detail.meta.planes[activePlane];
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
  }, [caseData.detail, layout, singlePlane, expandedPlane, focusedPlane]);

  if (bootState === "unreachable") {
    return (
      <div className="flex h-screen items-center justify-center bg-surface-page px-6 text-center">
        <div className="max-w-md">
          <p className="mb-2 font-condensed text-sm tracking-[0.12em] text-text-dim uppercase">
            NeuroVision-X
          </p>
          <p className="font-mono text-sm text-text-primary">
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
          <p className="mb-2 font-condensed text-sm tracking-[0.12em] text-text-dim uppercase">
            NeuroVision-X
          </p>
          <p className="mb-2 font-mono text-sm text-text-primary">
            The API responded, but not with what the viewer expected.
          </p>
          <p className="font-mono text-xs text-text-secondary">{bootError}</p>
          <p className="mt-3 font-mono text-xs text-text-dim">
            Check the paths the server resolved at <code>/api/health</code>.
          </p>
        </div>
      </div>
    );
  }

  const detail = caseData.detail;
  const planeCounts = detail?.meta.planes ?? ZERO_PLANES;
  const shape = detail?.meta.shape ?? null;
  const hasLabel = detail?.meta.has_label ?? false;
  const hasLogits = detail?.meta.has_logits ?? false;
  const hasPrediction = detail?.meta.has_prediction ?? false;
  const uncertaintyKind = caseData.uncertainty?.kind ?? null;

  // Fraction of this case's artifacts that have arrived. Counted against what
  // the case ACTUALLY has -- a case with no label or no logits must still be
  // able to reach 100%, or the bar would stall short of the end and look like
  // a failed load.
  const expectedArtifacts =
    4 + 1 + (hasLabel ? 1 : 0) + (hasLogits ? 1 : 0); // modalities + profile + label + logits
  const loadedArtifacts =
    Object.keys(caseData.volumes).length +
    (caseData.profile ? 1 : 0) +
    (caseData.labelMask ? 1 : 0) +
    (caseData.uncertainty ? 1 : 0);
  const loadProgress = Math.min(1, loadedArtifacts / expectedArtifacts);
  const ribbonPlane: Plane =
    layout === "single" ? singlePlane : (expandedPlane ?? "axial");
  const ribbonLabel = ribbonPlane.charAt(0).toUpperCase() + ribbonPlane.slice(1);
  const profilePlane = caseData.profile?.planes[ribbonPlane];
  const showCaseListInline = isPanelWidth;

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface-page text-text-primary">
      <Header
        health={health}
        reachable={bootState === "ready"}
        showCaseListToggle={!showCaseListInline}
        onToggleCaseList={() => setCaseListOpen((v) => !v)}
      />

      {/* Below the single-viewport breakpoint the readout moves BELOW the
          image instead of beside it: at ~600px a 224px sidebar eats a third
          of the width, and the MRI is the thing worth the pixels. */}
      <div
        className={`relative flex min-h-0 flex-1 ${layout === "single" ? "flex-col" : "flex-row"}`}
      >
        {showCaseListInline && (
          <div className="w-56 shrink-0 overflow-hidden border-r border-surface-seam bg-surface-panel">
            <CaseList
              cases={cases}
              selectedCaseId={selectedCaseId}
              onSelect={(id) => {
                setSelectedCaseId(id);
              }}
            />
          </div>
        )}

        {!showCaseListInline && caseListOpen && (
          <>
            <div
              className="absolute inset-0 z-10 bg-black/60"
              onClick={() => setCaseListOpen(false)}
              aria-hidden="true"
            />
            <div className="absolute inset-y-0 left-0 z-20 w-64 overflow-hidden border-r border-surface-seam bg-surface-panel">
              <CaseList
                cases={cases}
                selectedCaseId={selectedCaseId}
                onSelect={(id) => {
                  setSelectedCaseId(id);
                  setCaseListOpen(false);
                }}
              />
            </div>
          </>
        )}

        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-2">
          {!selectedCaseId ? (
            <div className="flex flex-1 items-center justify-center text-center">
              <div>
                <p className="font-mono text-sm text-text-primary">Pick a case to begin.</p>
                {health && (
                  <p className="mt-1 font-mono text-xs text-text-dim">
                    Evaluation directory: {health.eval_dir}
                  </p>
                )}
              </div>
            </div>
          ) : caseData.error ? (
            <div className="flex flex-1 items-center justify-center text-center">
              <p className="font-mono text-sm text-text-primary">{caseData.error}</p>
            </div>
          ) : (
            <>
              {/* A case pulls four modality volumes plus masks, entropy and the
                  profile - around 20 MB. Without this the viewports sit black
                  for several seconds and the app reads as frozen. Determinate,
                  because we know exactly how many artifacts are outstanding. */}
              {caseData.loading && (
                <div
                  className="flex shrink-0 items-center gap-3 border border-surface-seam bg-surface-panel px-3 py-1.5"
                  role="status"
                  aria-live="polite"
                >
                  <span className="font-condensed text-[11px] tracking-[0.12em] text-text-dim uppercase">
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
                  image={caseData.volumes[modality]?.data ?? null}
                  predictionMask={caseData.predictionMask?.data ?? null}
                  labelMask={caseData.labelMask?.data ?? null}
                  uncertainty={caseData.uncertainty?.data ?? null}
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
        </div>

        {selectedCaseId && !caseData.error && (
          <div
            className={`shrink-0 overflow-y-auto border-surface-seam bg-surface-panel ${
              layout === "single" ? "max-h-56 w-full border-t" : "w-56 border-l"
            }`}
          >
            <MetricsPanel metrics={caseData.detail?.metrics ?? null} regions={caseData.detail?.regions ?? null} />
            <Legend
              overlayMode={overlayMode}
              showUncertainty={showUncertainty}
              hasLabel={hasLabel}
              uncertaintyKind={uncertaintyKind}
            />
          </div>
        )}
      </div>

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
      />
    </div>
  );
}
