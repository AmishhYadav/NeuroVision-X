// Base layer: the grayscale T1CE slice. Reveal layer: the SAME slice with the
// case's GROUND-TRUTH segmentation (the expert label shipped with BraTS, not a
// model prediction) composited on top, masked to a soft circle that follows
// the cursor. Both layers are real data from BraTS2021_00000.
import { useRef, useState } from "react";
import { useHeroSliceImages } from "./heroSliceImages";

const SPOTLIGHT_RADIUS_PX = 120;

export function CursorReveal() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { baseUrl, revealUrl, caseId } = useHeroSliceImages();
  const [pos, setPos] = useState({ x: -999, y: -999 });

  const handleMove = (clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ x: clientX - rect.left, y: clientY - rect.top });
  };
  const mask = `radial-gradient(circle ${SPOTLIGHT_RADIUS_PX}px at ${pos.x}px ${pos.y}px, black 0%, black 60%, transparent 100%)`;

  return (
    <figure className="m-0">
      <div
        ref={containerRef}
        className="relative aspect-square w-full overflow-hidden rounded-xl border border-surface-seam bg-surface-viewport"
        onMouseMove={(e) => handleMove(e.clientX, e.clientY)}
        onMouseLeave={() => setPos({ x: -999, y: -999 })}
        onTouchMove={(e) => {
          const t = e.touches[0];
          if (t) handleMove(t.clientX, t.clientY);
        }}
      >
        {baseUrl && (
          <img
            src={baseUrl}
            alt="Axial T1CE MRI slice from a BraTS case"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ imageRendering: "pixelated" }}
          />
        )}
        {revealUrl && (
          <img
            src={revealUrl}
            alt="The same slice with the ground-truth tumour label overlaid"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ imageRendering: "pixelated", WebkitMaskImage: mask, maskImage: mask }}
          />
        )}
        {!baseUrl && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-mono text-xs text-text-dim">Loading slice…</span>
          </div>
        )}
      </div>
      <figcaption className="mt-2 font-mono text-[11px] text-text-dim">
        {caseId ? `${caseId} · axial T1CE · ground-truth label, revealed under the cursor` : " "}
      </figcaption>
    </figure>
  );
}
