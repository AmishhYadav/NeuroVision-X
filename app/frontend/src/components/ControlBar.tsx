import { Brain, FileText, Power } from "lucide-react";
import type { Modality } from "../api";
import type { OverlayMode } from "../lib/render";

interface ControlBarProps {
  modality: Modality;
  onChangeModality: (m: Modality) => void;
  overlayMode: OverlayMode;
  onChangeOverlayMode: (m: OverlayMode) => void;
  hasLabel: boolean;
  hasPrediction: boolean;
  showTruthOutline: boolean;
  onToggleTruthOutline: () => void;
  overlayOpacity: number;
  onChangeOverlayOpacity: (v: number) => void;
  hasLogits: boolean;
  showUncertainty: boolean;
  onToggleUncertainty: () => void;
  hasReport: boolean;
  reportOpen: boolean;
  onToggleReport: () => void;
  twinOpen: boolean;
  onToggleTwin: () => void;
}

// Visual left-to-right order shown in the control bar; keys 1-4 map to this
// same order so the shortcut matches what the researcher sees on screen.
export const MODALITY_ORDER: Modality[] = ["t1ce", "t1", "t2", "flair"];

const MODALITIES: { key: Modality; label: string }[] = [
  { key: "t1ce", label: "T1CE" },
  { key: "t1", label: "T1" },
  { key: "t2", label: "T2" },
  { key: "flair", label: "FLAIR" },
];

const OVERLAY_MODES: { key: OverlayMode; label: string }[] = [
  { key: "prediction", label: "Prediction" },
  { key: "truth", label: "Truth" },
  { key: "disagreement", label: "Disagreement" },
];

function Divider() {
  return <div className="mx-3 h-6 w-px shrink-0 bg-surface-seam" aria-hidden="true" />;
}

export function ControlBar({
  modality,
  onChangeModality,
  overlayMode,
  onChangeOverlayMode,
  hasLabel,
  hasPrediction,
  showTruthOutline,
  onToggleTruthOutline,
  overlayOpacity,
  onChangeOverlayOpacity,
  hasLogits,
  showUncertainty,
  onToggleUncertainty,
  hasReport,
  reportOpen,
  onToggleReport,
  twinOpen,
  onToggleTwin,
}: ControlBarProps) {
  return (
    // Wraps to a second row on narrow screens rather than clipping. It used to
    // be a fixed-height row with overflow-x-auto, which technically scrolled
    // but gave no affordance -- the entropy toggle simply vanished off the
    // right edge at ~820px and looked like a missing feature.
    <div className="flex min-h-12 shrink-0 flex-wrap items-center gap-y-1 border-t border-surface-seam bg-surface-panel px-3 py-1 xl:flex-nowrap xl:gap-y-0 xl:py-0">
      <div className="flex shrink-0 items-center gap-1" role="group" aria-label="Modality">
        {MODALITIES.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            onClick={() => onChangeModality(key)}
            aria-pressed={modality === key}
            className={`rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
              modality === key
                ? "bg-surface-raised text-text-primary"
                : "text-text-secondary hover:text-text-primary"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <Divider />

      <div className="flex shrink-0 items-center gap-1" role="group" aria-label="Overlay">
        {OVERLAY_MODES.map(({ key, label }) => {
          // Prediction needs a saved prediction; Truth needs a label;
          // Disagreement compares the two, so it needs both.
          const missingPrediction = key !== "truth" && !hasPrediction;
          const missingLabel = key !== "prediction" && !hasLabel;
          const disabled = missingPrediction || missingLabel;
          const hint = missingPrediction
            ? "No saved prediction for this case."
            : missingLabel
              ? "No ground-truth label for this case."
              : undefined;
          return (
            <button
              key={key}
              type="button"
              disabled={disabled}
              onClick={() => onChangeOverlayMode(key)}
              aria-pressed={overlayMode === key}
              title={hint}
              className={`rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
                disabled
                  ? "cursor-not-allowed text-text-dim"
                  : overlayMode === key
                    ? "bg-surface-raised text-text-primary"
                    : "text-text-secondary hover:text-text-primary"
              }`}
            >
              {label}
            </button>
          );
        })}
        <button
          type="button"
          disabled={!hasLabel || overlayMode !== "prediction"}
          onClick={onToggleTruthOutline}
          aria-pressed={showTruthOutline}
          title="Outline the ground-truth whole tumour"
          className={`ml-1 rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
            !hasLabel || overlayMode !== "prediction"
              ? "cursor-not-allowed text-text-dim"
              : showTruthOutline
                ? "bg-surface-raised text-text-primary"
                : "text-text-secondary hover:text-text-primary"
          }`}
        >
          Outline
        </button>
      </div>

      <Divider />

      <div className="flex min-w-[9rem] shrink-0 items-center gap-2">
        <span className="eyebrow shrink-0">Opacity</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={overlayOpacity}
          onChange={(e) => onChangeOverlayOpacity(parseFloat(e.target.value))}
          className="w-24 accent-[#E7EAEE]"
          aria-label="Overlay opacity"
        />
        <span className="tabular w-9 shrink-0 font-mono text-xs text-text-secondary">
          {overlayOpacity.toFixed(2)}
        </span>
      </div>

      <Divider />

      <button
        type="button"
        disabled={!hasLogits}
        onClick={onToggleUncertainty}
        aria-pressed={showUncertainty}
        title={!hasLogits ? "No saved logits for this case." : undefined}
        className={`flex shrink-0 items-center gap-1.5 rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
          !hasLogits
            ? "cursor-not-allowed text-text-dim"
            : showUncertainty
              ? "bg-surface-raised text-text-primary"
              : "text-text-secondary hover:text-text-primary"
        }`}
      >
        <Power size={13} aria-hidden="true" />
        Predictive entropy
      </button>
      {!hasLogits && (
        <span className="ml-2 hidden shrink-0 font-mono text-[11px] text-text-dim sm:inline">
          No saved logits for this case.
        </span>
      )}

      <Divider />

      <button
        type="button"
        disabled={!hasReport}
        onClick={onToggleReport}
        aria-pressed={reportOpen}
        title={!hasReport ? "No report has been generated for this case." : undefined}
        className={`flex shrink-0 items-center gap-1.5 rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
          !hasReport
            ? "cursor-not-allowed text-text-dim"
            : reportOpen
              ? "bg-surface-raised text-text-primary"
              : "text-text-secondary hover:text-text-primary"
        }`}
      >
        <FileText size={13} aria-hidden="true" />
        Report
      </button>
      {!hasReport && (
        <span className="ml-2 hidden shrink-0 font-mono text-[11px] text-text-dim sm:inline">
          No report has been generated for this case.
        </span>
      )}

      <Divider />

      <button
        type="button"
        onClick={onToggleTwin}
        aria-pressed={twinOpen}
        title="Real 3D reconstruction of this case's own tumour"
        className={`flex shrink-0 items-center gap-1.5 rounded-sm px-2 py-1 font-mono text-xs transition-colors duration-[120ms] ${
          twinOpen
            ? "bg-surface-raised text-text-primary"
            : "text-text-secondary hover:text-text-primary"
        }`}
      >
        <Brain size={13} aria-hidden="true" />
        3D twin
      </button>
    </div>
  );
}
