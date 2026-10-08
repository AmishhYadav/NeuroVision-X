import { useEffect, useRef, useState } from "react";
import { ERROR_LANE_COLOR, ENTROPY_LINE_COLOR, WT_COLOR } from "../lib/colors";

interface SliceRibbonProps {
  planeLabel: string;
  sliceCount: number;
  currentIndex: number;
  onScrub: (index: number) => void;
  tumor: number[];
  error: number[] | null;
  entropy: number[] | null;
  onFocusRibbon?: () => void;
}

const BG = "#0A0B0D";
const DIVIDER_FRAC = 0.6;

function rgba([r, g, b]: readonly [number, number, number], a: number): string {
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

export function SliceRibbon({
  planeLabel,
  sliceCount,
  currentIndex,
  onScrub,
  tumor,
  error,
  entropy,
  onFocusRibbon,
}: SliceRibbonProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const draggingRef = useRef(false);

  const maxTumor = Math.max(1e-9, ...tumor);
  const maxEntropy = entropy ? Math.max(1e-9, ...entropy) : 1;

  const indexFromX = (x: number, width: number): number => {
    const frac = Math.min(1, Math.max(0, x / width));
    return Math.min(sliceCount - 1, Math.max(0, Math.floor(frac * sliceCount)));
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || sliceCount === 0) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    const dividerY = height * DIVIDER_FRAC;
    const colWidth = width / sliceCount;

    // Lane 1: tumour cross-section, drawn upward from the divider.
    ctx.fillStyle = rgba(WT_COLOR, 0.9);
    for (let i = 0; i < sliceCount; i++) {
      const frac = tumor[i] / maxTumor;
      const barHeight = frac * dividerY;
      ctx.fillRect(i * colWidth, dividerY - barHeight, Math.ceil(colWidth), barHeight);
    }

    // Lane 2: disagreement, drawn downward from the divider, same denominator as lane 1.
    if (error) {
      ctx.fillStyle = rgba(ERROR_LANE_COLOR, 0.9);
      const laneHeight = height - dividerY;
      for (let i = 0; i < sliceCount; i++) {
        const frac = error[i] / maxTumor;
        const barHeight = Math.min(laneHeight, frac * dividerY);
        ctx.fillRect(i * colWidth, dividerY, Math.ceil(colWidth), barHeight);
      }
    }

    // Divider line.
    ctx.strokeStyle = "rgba(42, 47, 54, 0.8)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, dividerY + 0.5);
    ctx.lineTo(width, dividerY + 0.5);
    ctx.stroke();

    // Entropy polyline across the full height.
    if (entropy) {
      ctx.strokeStyle = rgba(ENTROPY_LINE_COLOR, 0.8);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < sliceCount; i++) {
        const frac = entropy[i] / maxEntropy;
        const x = (i + 0.5) * colWidth;
        const y = height - frac * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Hover marker.
    if (hoverIndex !== null && hoverIndex !== currentIndex) {
      const x = (hoverIndex + 0.5) * colWidth;
      ctx.strokeStyle = "rgba(231, 234, 238, 0.35)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Current slice marker.
    const cx = (currentIndex + 0.5) * colWidth;
    ctx.strokeStyle = "#E7EAEE";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, height);
    ctx.stroke();
  };

  // Kept current every render so the resize observer below - which only
  // subscribes once - always calls the freshest closure instead of the one
  // captured at mount (a stale draw() would repaint the ribbon with
  // mount-time tumor/error/entropy/index data on every window resize).
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sliceCount, currentIndex, hoverIndex, tumor, error, entropy]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => drawRef.current());
    observer.observe(container);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scrubFromEvent = (e: { clientX: number }) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    onScrub(indexFromX(e.clientX - rect.left, rect.width));
  };

  const hoverFromEvent = (e: { clientX: number }) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setHoverIndex(indexFromX(e.clientX - rect.left, rect.width));
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    draggingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    scrubFromEvent(e);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    hoverFromEvent(e);
    if (draggingRef.current) scrubFromEvent(e);
  };

  const handlePointerUp = () => {
    draggingRef.current = false;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    // stopPropagation on the keys this element owns so the global
    // ArrowLeft/ArrowRight viewport-stepping listener on window does not
    // also fire and double-step the same plane.
    if (e.key === "ArrowRight") {
      e.preventDefault();
      e.stopPropagation();
      onScrub(Math.min(sliceCount - 1, currentIndex + 1));
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      onScrub(Math.max(0, currentIndex - 1));
    } else if (e.key === "Home") {
      e.preventDefault();
      e.stopPropagation();
      onScrub(0);
    } else if (e.key === "End") {
      e.preventDefault();
      e.stopPropagation();
      onScrub(Math.max(0, sliceCount - 1));
    }
  };

  const captionParts = ["Tumour cross-section"];
  if (error) captionParts.push("disagreement");
  if (entropy) captionParts.push("mean entropy");
  const caption = `${captionParts.join(" · ")}, by slice`;

  const hoverInfo =
    hoverIndex !== null
      ? {
          slice: hoverIndex,
          tumor: tumor[hoverIndex] ?? 0,
          error: error ? error[hoverIndex] : null,
          entropy: entropy ? entropy[hoverIndex] : null,
        }
      : null;

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="eyebrow">{caption}</span>
        {hoverInfo && (
          <span className="tabular font-mono text-[11px] text-text-secondary">
            #{hoverInfo.slice} · tumour {(hoverInfo.tumor * 100).toFixed(1)}%
            {hoverInfo.error !== null && ` · error ${(hoverInfo.error * 100).toFixed(1)}%`}
            {hoverInfo.entropy !== null && ` · entropy ${hoverInfo.entropy.toFixed(2)}`}
          </span>
        )}
      </div>
      <div
        ref={containerRef}
        role="slider"
        tabIndex={0}
        aria-label={`${planeLabel} slice position`}
        aria-valuemin={0}
        aria-valuemax={Math.max(sliceCount - 1, 0)}
        aria-valuenow={currentIndex}
        className="h-16 w-full cursor-pointer touch-none select-none"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={() => setHoverIndex(null)}
        onKeyDown={handleKeyDown}
        onFocus={onFocusRibbon}
      >
        <canvas ref={canvasRef} className="h-full w-full" />
      </div>
      <div className="flex justify-end">
        <span className="tabular font-mono text-[11px] text-text-dim">
          {currentIndex} / {Math.max(sliceCount - 1, 0)}
        </span>
      </div>
    </div>
  );
}
