import { Power } from "lucide-react";
import type { Modality } from "../api";
import type { OverlayMode } from "../lib/render";

interface ControlBarProps {
  modality: Modality;
  onChangeModality: (m: Modality) => void;
  overlayMode: OverlayMode;
  onChangeOverlayMode: (m: OverlayMode) => void;
  hasLabel: boolean;
  showTruthOutline: boolean;
  onToggleTruthOutline: () => void;
  overlayOpacity: number;
  onChangeOverlayOpacity: (v: number) => void;
  hasLogits: boolean;
  showUncertainty: boolean;
  onToggleUncertainty: () => void;
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

const OVERLAY_MODES: { key: OverlayMode; label: string; needsLabel: boolean }[] = [
  { key: "prediction", label: "Prediction", needsLabel: false },
  { key: "truth", label: "Truth", needsLabel: true },
  { key: "disagreement", label: "Disagreement", needsLabel: true },
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
  showTruthOutline,
  onToggleTruthOutline,
  overlayOpacity,
  onChangeOverlayOpacity,
  hasLogits,
  showUncertainty,
  onToggleUncertainty,
}: ControlBarProps) {
  return (
    <div className="flex h-12 shrink-0 items-center overflow-x-auto border-t border-surface-seam bg-surface-panel px-3">
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
        {OVERLAY_MODES.map(({ key, label, needsLabel }) => {
          const disabled = needsLabel && !hasLabel;
          return (
            <button
              key={key}
              type="button"
              disabled={disabled}
              onClick={() => onChangeOverlayMode(key)}
              aria-pressed={overlayMode === key}
              title={disabled ? "No ground-truth label for this case." : undefined}
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
        Entropy
      </button>
      {!hasLogits && (
        <span className="ml-2 hidden shrink-0 font-mono text-[11px] text-text-dim sm:inline">
          No saved logits for this case.
        </span>
      )}
    </div>
  );
}
