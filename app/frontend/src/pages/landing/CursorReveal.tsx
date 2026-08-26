// Base layer: the real grayscale MRI slice. Reveal layer: the SAME slice
// with the real ground-truth segmentation composited on top, masked to a
// soft circle that follows the cursor - drag across the scan and it
// re-enacts "here is the raw image, here is what the model finds," using
// real data on both layers, not a decorative reveal.
import { useRef, useState } from "react";
import { useHeroSliceImages } from "./heroSliceImages";

const SPOTLIGHT_RADIUS_PX = 140;

export function CursorReveal() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { baseUrl, revealUrl, caseId } = useHeroSliceImages();
  const [pos, setPos] = useState({ x: -999, y: -999 });

  const handleMove = (clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({ x: clientX - rect.left, y: clientY - rect.top });
  };

  return (
    <div
      ref={containerRef}
      className="relative aspect-square w-full max-w-md overflow-hidden rounded-lg border border-landing-seam bg-black shadow-xl"
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
          alt="Raw T1CE MRI slice, evaluation cohort"
          className="absolute inset-0 h-full w-full object-contain"
          style={{ imageRendering: "pixelated" }}
        />
      )}
      {revealUrl && (
        <img
          src={revealUrl}
          alt="Same slice with the real segmentation overlay"
          className="absolute inset-0 h-full w-full object-contain"
          style={{
            imageRendering: "pixelated",
            WebkitMaskImage: `radial-gradient(circle ${SPOTLIGHT_RADIUS_PX}px at ${pos.x}px ${pos.y}px, black 0%, black 60%, transparent 100%)`,
            maskImage: `radial-gradient(circle ${SPOTLIGHT_RADIUS_PX}px at ${pos.x}px ${pos.y}px, black 0%, black 60%, transparent 100%)`,
          }}
        />
      )}
      {!baseUrl && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xs text-white/40">Loading real slice…</span>
        </div>
      )}
      {baseUrl && (
        <div className="absolute right-2 bottom-2 font-mono text-[10px] text-white/40">
          {caseId}, real ground truth
        </div>
      )}
    </div>
  );
}
