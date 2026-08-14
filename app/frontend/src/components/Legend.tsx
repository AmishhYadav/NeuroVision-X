import type { OverlayMode } from "../lib/render";

interface LegendProps {
  overlayMode: OverlayMode;
  showUncertainty: boolean;
  hasLabel: boolean;
}

function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
        style={{ backgroundColor: color }}
        aria-hidden="true"
      />
      <span className="font-mono text-[11px] text-text-secondary">{label}</span>
    </div>
  );
}

export function Legend({ overlayMode, showUncertainty, hasLabel }: LegendProps) {
  return (
    <div className="flex flex-col gap-2 border-t border-surface-seam px-3 py-3">
      <div className="eyebrow">Legend</div>

      {(overlayMode === "prediction" || overlayMode === "truth") && (
        <div className="flex flex-col gap-1.5">
          <Swatch color="#56B4E9" label="Necrotic core" />
          <Swatch color="#009E73" label="Oedema" />
          <Swatch color="#D55E00" label="Enhancing tumour" />
          {overlayMode === "prediction" && hasLabel && (
            <div className="mt-1 flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 shrink-0 rounded-[2px] border border-white"
                aria-hidden="true"
              />
              <span className="font-mono text-[11px] text-text-secondary">
                Ground-truth outline
              </span>
            </div>
          )}
        </div>
      )}

      {overlayMode === "disagreement" && (
        <div className="flex flex-col gap-1.5">
          <Swatch color="#F0E442" label="False negative" />
          <Swatch color="#CC79A7" label="False positive" />
        </div>
      )}

      {showUncertainty && (
        <div className="flex flex-col gap-1.5 pt-1">
          <div
            className="h-2 w-full rounded-[2px]"
            style={{
              background: "linear-gradient(90deg, #3B0F70, #E4693E, #FCFDBF)",
            }}
            aria-hidden="true"
          />
          <div className="flex justify-between font-mono text-[10px] text-text-dim">
            <span>0</span>
            <span>Predictive entropy</span>
            <span>1</span>
          </div>
        </div>
      )}
    </div>
  );
}
