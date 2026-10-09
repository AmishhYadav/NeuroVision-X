// Base layer: the grayscale T1CE slice. Reveal layer: the SAME slice with the
// case's GROUND-TRUTH segmentation (the expert label shipped with BraTS, not a
// model prediction) composited on top. Both layers are real data from
// BraTS2021_00000.
//
// The reveal layer shows through two masks at once (CSS masks add):
//  - a soft circle that follows the cursor or finger, as before;
//  - everything above a scan line that the page's scroll moves down the
//    slice (`scan`, 0 = top, 1 = bottom), so the label is uncovered as the
//    reader moves through the pipeline steps beside it.
// Both are motion values feeding one mask string, so moving the mouse or
// scrolling never re-renders React.
import { useRef } from "react";
import {
  motion,
  useMotionTemplate,
  useMotionValue,
  useReducedMotion,
  useTransform,
  type MotionValue,
} from "motion/react";
import { useHeroSliceImages } from "./heroSliceImages";

const SPOTLIGHT_RADIUS_PX = 120;

export function CursorReveal({ scan }: { scan?: MotionValue<number> }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { baseUrl, revealUrl, caseId } = useHeroSliceImages();
  const mx = useMotionValue(-999);
  const my = useMotionValue(-999);
  const noScan = useMotionValue(0);
  const s = scan ?? noScan;
  const scanPct = useTransform(s, (v) => v * 100);
  const lineY = useTransform(s, (v) => `${v * 100}%`);
  const lineOpacity = useTransform(s, [0, 0.03, 0.97, 1], [0, 1, 1, 0]);
  // Under reduced motion `scan` is a constant 1: the label is simply shown.
  const reduce = useReducedMotion();
  const captionTail = reduce
    ? "ground-truth label overlaid"
    : scan
      ? "ground-truth label, uncovered as you scroll and under the cursor"
      : "ground-truth label, revealed under the cursor";
  const mask = useMotionTemplate`radial-gradient(circle ${SPOTLIGHT_RADIUS_PX}px at ${mx}px ${my}px, black 0%, black 60%, transparent 100%), linear-gradient(to bottom, black calc(${scanPct}% - 8px), transparent ${scanPct}%)`;

  const handleMove = (clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    mx.set(clientX - rect.left);
    my.set(clientY - rect.top);
  };

  return (
    <figure className="m-0">
      <div
        ref={containerRef}
        className="relative aspect-square w-full overflow-hidden rounded-[10px] border border-surface-seam bg-surface-viewport"
        onMouseMove={(e) => handleMove(e.clientX, e.clientY)}
        onMouseLeave={() => {
          mx.set(-999);
          my.set(-999);
        }}
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
          <motion.img
            src={revealUrl}
            alt="The same slice with the ground-truth tumour label overlaid"
            className="absolute inset-0 h-full w-full object-contain"
            style={{ imageRendering: "pixelated", WebkitMaskImage: mask, maskImage: mask }}
          />
        )}
        {/* The scan line itself: a full-size layer translated by the scan
            progress, so only transform changes per frame. */}
        {revealUrl && scan && (
          <motion.div aria-hidden="true" className="pointer-events-none absolute inset-0" style={{ y: lineY, opacity: lineOpacity }}>
            <div className="h-px w-full bg-brand-primary shadow-[0_0_12px_1px_var(--color-brand-primary)]" />
          </motion.div>
        )}
        {!baseUrl && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-mono text-xs text-text-dim">Loading slice…</span>
          </div>
        )}
      </div>
      {/* Solid backing: the page's hologram glows behind this figure, and a
          dim mono caption over its bright contour lines lost its contrast. */}
      <figcaption className="bg-surface-page pt-2 pb-1 font-mono text-[11px] text-text-dim">
        {caseId ? `${caseId} · axial T1CE · ${captionTail}` : " "}
      </figcaption>
    </figure>
  );
}
