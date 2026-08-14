import { useEffect, useRef } from "react";
import type { Plane } from "../api";
import { renderSlice, type OverlayMode } from "../lib/render";

interface ViewportProps {
  plane: Plane;
  planeLabel: string;
  sliceIndex: number;
  sliceCount: number;
  shape: [number, number, number] | null;
  image: Uint8Array | null;
  predictionMask: Uint8Array | null;
  labelMask: Uint8Array | null;
  uncertainty: Uint8Array | null;
  overlayMode: OverlayMode;
  overlayOpacity: number;
  showTruthOutline: boolean;
  showUncertainty: boolean;
  uncertaintyOpacity: number;
  expanded: boolean;
  expandable: boolean;
  onToggleExpand: () => void;
  onFocusPlane: () => void;
}

export function Viewport({
  plane,
  planeLabel,
  sliceIndex,
  sliceCount,
  shape,
  image,
  predictionMask,
  labelMask,
  uncertainty,
  overlayMode,
  overlayOpacity,
  showTruthOutline,
  showUncertainty,
  uncertaintyOpacity,
  expanded,
  expandable,
  onToggleExpand,
  onFocusPlane,
}: ViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const draw = () => {
    const off = offscreenRef.current;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!off || !canvas || !container) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw === 0 || ch === 0) return;
    // Backing store scaled to device pixels, CSS size pinned to the layout
    // size, and the transform folds the DPR back in so the rest of this
    // function can keep working in CSS-pixel coordinates. Without this the
    // browser upsamples the canvas on HiDPI displays and blurs the slice -
    // imageSmoothingEnabled only controls drawImage *within* the canvas, it
    // cannot prevent that upsample.
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, cw, ch);
    const scale = Math.min(cw / off.width, ch / off.height);
    const dw = off.width * scale;
    const dh = off.height * scale;
    const dx = (cw - dw) / 2;
    const dy = (ch - dh) / 2;
    ctx.drawImage(off, 0, 0, off.width, off.height, dx, dy, dw, dh);
  };

  // Recompute the ImageData only when a rendering dependency actually changes.
  useEffect(() => {
    if (!image || !shape) return;
    const imgData = renderSlice({
      plane,
      shape,
      sliceIndex,
      image,
      predictionMask,
      labelMask,
      uncertainty,
      overlayMode,
      overlayOpacity,
      showTruthOutline,
      showUncertainty,
      uncertaintyOpacity,
    });
    if (!offscreenRef.current) offscreenRef.current = document.createElement("canvas");
    const off = offscreenRef.current;
    off.width = imgData.width;
    off.height = imgData.height;
    const offCtx = off.getContext("2d");
    if (!offCtx) return;
    offCtx.putImageData(imgData, 0, 0);
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    plane,
    shape,
    sliceIndex,
    image,
    predictionMask,
    labelMask,
    uncertainty,
    overlayMode,
    overlayOpacity,
    showTruthOutline,
    showUncertainty,
    uncertaintyOpacity,
  ]);

  // Redraw (no recompute) on container resize.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(container);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      className="flex h-full min-h-0 flex-col border border-surface-seam bg-surface-panel"
      onFocus={onFocusPlane}
      onMouseDown={onFocusPlane}
    >
      <div className="flex shrink-0 items-center gap-2 border-b border-surface-seam px-2 py-1">
        <button
          type="button"
          onClick={onToggleExpand}
          disabled={!expandable}
          aria-expanded={expanded}
          title={expandable ? "Click to expand this plane" : undefined}
          className="font-mono text-[11px] tracking-[0.02em] text-text-secondary uppercase transition-colors duration-[120ms] hover:text-text-primary disabled:cursor-default disabled:hover:text-text-secondary"
        >
          {planeLabel}
        </button>
        <span className="tabular ml-auto font-mono text-[11px] text-text-dim">
          {sliceIndex} / {Math.max(sliceCount - 1, 0)}
        </span>
      </div>
      <div ref={containerRef} className="min-h-0 flex-1 bg-surface-viewport">
        <canvas ref={canvasRef} tabIndex={0} aria-label={`${planeLabel} slice viewport`} />
      </div>
    </div>
  );
}
