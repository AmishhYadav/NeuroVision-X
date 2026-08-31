import { CONFORMAL_BAND, GRADCAM, PREDICTIVE_ENTROPY_SINGLE_PASS } from "../api";
import { ENTROPY_ONE_CHANNEL } from "../lib/colors";
import type { OverlayMode } from "../lib/render";

interface LegendProps {
  overlayMode: OverlayMode;
  showUncertainty: boolean;
  hasLabel: boolean;
  /** Raw `X-Uncertainty-Kind` header value from the last uncertainty fetch, or null if absent. */
  uncertaintyKind: string | null;
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

export function Legend({ overlayMode, showUncertainty, hasLabel, uncertaintyKind }: LegendProps) {
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

      {showUncertainty &&
        (uncertaintyKind === CONFORMAL_BAND ? (
          <div className="flex flex-col gap-1.5 pt-1">
            {/* This buffer is categorical - byte values 0, 128, 255 only - not
                a continuous [0, 1] entropy field, so no gradient bar or
                "N channel" tick mark here. render.ts clamps alpha to full for
                both non-zero values (they both sit above ENTROPY_ALPHA_FULL),
                so hue is the only thing distinguishing them; that's what these
                two swatches show, using entropyColor at the buffer's actual
                normalized values (128/255 and 255/255). */}
            <Swatch color="#E46A3F" label="Safety margin (not in point estimate)" />
            <Swatch color="#FCFDBF" label="Point estimate (and safety margin)" />
            <div className="text-center font-mono text-[10px] text-text-dim">
              Conformal band: guaranteed-coverage region
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-1.5 pt-1">
            {/* The bar spans the full [0, 1] scale the colours encode. The
                tick mark below (ONE region channel maximally uncertain, 1/3)
                is specific to the 3-channel Bernoulli entropy calculation -
                Grad-CAM has no such property, so it is only drawn for
                entropy. Everything to the right of it needs two of the three
                channels uncertain at once, so marking it tells the reader
                which part of the ramp entropy can realistically reach. */}
            <div className="relative h-2 w-full">
              <div
                className="h-2 w-full rounded-[2px]"
                style={{
                  background: "linear-gradient(90deg, #3B0F70, #E4693E, #FCFDBF)",
                }}
                aria-hidden="true"
              />
              {uncertaintyKind !== GRADCAM && (
                <div
                  className="absolute top-0 h-2 w-px bg-white/80"
                  style={{ left: `${ENTROPY_ONE_CHANNEL * 100}%` }}
                  aria-hidden="true"
                />
              )}
            </div>
            <div className="flex justify-between font-mono text-[10px] text-text-dim">
              <span>0</span>
              <span>{uncertaintyKind === GRADCAM ? "Grad-CAM evidence" : "Predictive entropy"}</span>
              <span>1</span>
            </div>
            {uncertaintyKind !== GRADCAM && (
              <div className="text-center font-mono text-[10px] text-text-dim">
                tick = 1 channel fully uncertain
              </div>
            )}
            {/* Sub-label states exactly what was measured. The backend header
                is the source of truth - never fall back to the reassuring
                "single pass" text for an unrecognized or missing header, since
                this layer must never be presented as epistemic/MC-dropout
                uncertainty when it isn't. */}
            <div className="text-center font-mono text-[10px] text-text-dim">
              {uncertaintyKind === PREDICTIVE_ENTROPY_SINGLE_PASS
                ? "single pass · aleatoric + epistemic combined"
                : uncertaintyKind === GRADCAM
                  ? "Seg-Grad-CAM · evidence for this region's prediction"
                  : `unknown source (${uncertaintyKind ?? "no header"})`}
            </div>
          </div>
        ))}
    </div>
  );
}
