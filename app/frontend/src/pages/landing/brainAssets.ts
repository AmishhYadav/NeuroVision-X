// Mesh loading and colours shared by the landing page's two brain views:
// HeroBrain (static, in the hero column; used under reduced motion) and
// ScrollBrain (fixed behind the whole page, driven by scroll).
//
// Mesh: the SRI24 healthy-adult T1 template pial surface (/hero/sri24-cortex-*,
// real gyri and sulci). If those files are missing we fall back to the older
// BraTS2021_00000 brain shell under /twin, and the caption says which one is
// on screen. The caption travels with the mesh so it can never disagree with it.
import { useEffect, useState } from "react";
import * as THREE from "three";
import { loadMesh, type MeshBuffers } from "../../lib/loadBinary";

const SRI24_BASE = "/hero";
const SRI24_NAME = "sri24-cortex";
const TWIN_BASE = "/twin";

const CAPTION_SRI24 = "SRI24 healthy-adult T1 template (Rohlfing et al., 2010), CC BY-SA · an atlas, not a patient";
const CAPTION_FALLBACK = "Brain surface reconstructed from one BraTS case (BraTS2021_00000) · no tumour shown";

// Hologram hues, converted to LINEAR sRGB (negatives clamped to 0):
//   --color-holo:      oklch(0.80 0.12 232) -> (0.124, 0.598, 0.987)
//   --color-holo-deep: oklch(0.45 0.13 250) -> (0.000, 0.095, 0.321)
export const HOLO = new THREE.Color().setRGB(0.124, 0.598, 0.987, THREE.LinearSRGBColorSpace);
export const HOLO_DEEP = new THREE.Color().setRGB(0.0, 0.095, 0.321, THREE.LinearSRGBColorSpace);

function toGeometry(buf: MeshBuffers): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(buf.position, 3));
  geom.setAttribute("normal", new THREE.BufferAttribute(buf.normal, 3));
  geom.setIndex(new THREE.BufferAttribute(buf.index, 1));
  return geom;
}

export interface HeroAssets {
  geometries: THREE.BufferGeometry[];
  caption: string;
}

/** Try the SRI24 mesh; on any failure use the /twin brain. The dev server
 * answers a missing file with index.html (status 200), so the meta must
 * parse as JSON and the buffer sizes must match it. */
async function loadSri24(): Promise<THREE.BufferGeometry[] | null> {
  try {
    const res = await fetch(`${SRI24_BASE}/${SRI24_NAME}-meta.json`);
    if (!res.ok) return null;
    const meta = (await res.json()) as { n_vertices?: number; n_faces?: number };
    const buf = await loadMesh(SRI24_BASE, SRI24_NAME);
    if (!meta.n_vertices || buf.position.length !== meta.n_vertices * 3) return null;
    if (buf.normal.length !== buf.position.length || buf.index.length === 0) return null;
    return [toGeometry(buf)];
  } catch {
    return null;
  }
}

/** Loads the hero mesh once; disposes its GPU buffers on unmount. */
export function useHeroAssets(): HeroAssets | null {
  const [assets, setAssets] = useState<HeroAssets | null>(null);
  useEffect(() => {
    let cancelled = false;
    let made: THREE.BufferGeometry[] = [];
    (async () => {
      let geometries = await loadSri24();
      let caption = CAPTION_SRI24;
      if (!geometries) {
        const [l, r] = await Promise.all([loadMesh(TWIN_BASE, "brain-left"), loadMesh(TWIN_BASE, "brain-right")]);
        geometries = [toGeometry(l), toGeometry(r)];
        caption = CAPTION_FALLBACK;
      }
      made = geometries;
      if (cancelled) {
        made.forEach((g) => g.dispose());
        return;
      }
      setAssets({ geometries, caption });
    })().catch(() => {
      /* leave the quiet loading text; the page still works without the brain */
    });
    return () => {
      cancelled = true;
      made.forEach((g) => g.dispose());
    };
  }, []);
  return assets;
}
