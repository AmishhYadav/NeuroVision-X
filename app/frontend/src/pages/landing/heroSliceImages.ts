// Shared by CursorReveal and PipelineStory: loads BraTS2021_00000's real
// axial T1CE slice and composites it two ways - plain grayscale, and with
// the real ground-truth segmentation overlaid - so both components draw
// from one fetch instead of two.
import { useEffect, useState } from "react";
import { CLASS_COLORS } from "../../lib/colors";
import { loadUint8 } from "../../lib/loadBinary";

const HERO_BASE = "/hero";
const OVERLAY_OPACITY = 0.55;

interface HeroMeta {
  width: number;
  height: number;
  case_id: string;
}

export interface HeroSliceImages {
  baseUrl: string | null;
  revealUrl: string | null;
  caseId: string;
}

function compositeToDataUrl(
  gray: Uint8Array,
  label: Uint8Array | null,
  width: number,
  height: number,
): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = gray[i];
    let r = v;
    let g = v;
    let b = v;
    const cls = label ? label[i] : 0;
    const color = cls ? CLASS_COLORS[cls] : undefined;
    if (color) {
      r = v * (1 - OVERLAY_OPACITY) + color[0] * OVERLAY_OPACITY;
      g = v * (1 - OVERLAY_OPACITY) + color[1] * OVERLAY_OPACITY;
      b = v * (1 - OVERLAY_OPACITY) + color[2] * OVERLAY_OPACITY;
    }
    const o = i * 4;
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

export function useHeroSliceImages(): HeroSliceImages {
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [revealUrl, setRevealUrl] = useState<string | null>(null);
  const [caseId, setCaseId] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [gray, label, metaRes] = await Promise.all([
        loadUint8(`${HERO_BASE}/hero-image.bin`),
        loadUint8(`${HERO_BASE}/hero-label.bin`),
        fetch(`${HERO_BASE}/hero-meta.json`),
      ]);
      if (cancelled) return;
      const meta = (await metaRes.json()) as HeroMeta;
      setBaseUrl(compositeToDataUrl(gray, null, meta.width, meta.height));
      setRevealUrl(compositeToDataUrl(gray, label, meta.width, meta.height));
      setCaseId(meta.case_id);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return { baseUrl, revealUrl, caseId };
}
