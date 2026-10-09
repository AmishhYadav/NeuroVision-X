// The landing page's scroll-driven background: one hologram of the SRI24
// cortex, fixed behind the whole page. Scrolling is the scan.
//
//  - At the top it sits exactly where the hero's brain used to (the hero
//    column holds an empty, measured "anchor" box for it) and looks the same.
//  - As the page scrolls it eases to a pose per section (`data-brain-pose`
//    on each <section>): it recedes, grows, dims behind text, and returns to
//    full brightness beside the closing call to action.
//  - A scan plane sweeps through it from crown to base between the first
//    section and the last. Above the plane the cortex reads as "scanned"
//    (full brightness), below it is held dimmer, and where the plane cuts the
//    surface an axial contour lights up. A gantry ring marks the plane.
//  - Scroll also turns it, scroll speed quickens its scanlines, the mouse
//    tilts it slightly, and dragging the hero anchor spins it.
//
// Nothing here re-renders React per frame: scroll is read in useFrame and
// eased there, and every change is a uniform write or a camera update.
//
// How the brain is placed: the camera never moves. Its view is shifted with
// setViewOffset (a lens shift) to put the brain's centre at the pose's screen
// point, and zoomed with camera.zoom. A lens shift keeps the 3/4 view angle
// identical wherever the brain sits, which moving the mesh would not.
import { useEffect, useMemo, useRef, type RefObject } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { HOLO, HOLO_DEEP, useHeroAssets, type HeroAssets } from "./brainAssets";
import {
  easeOutCubic,
  poseAt,
  progressBetween,
  smoothstep,
  type BrainPose,
  type PoseKey,
} from "../../lib/scrollScene";

// The hero's original 3/4 left-lateral view: camera on +X (patient left), a
// little anterior (-Z) and raised so the top of the brain tilts toward us.
const CAMERA_POS: [number, number, number] = [3.25, 1.63, -1.75];
const FOV = 40;

// Below this width the page is one column of text, so the brain stays
// centred and faint behind it instead of trading sides with the copy.
const WIDE_MIN_PX = 1024;

// Poses per section, named by each section's data-brain-pose attribute. See
// BrainPose in lib/scrollScene.ts for the units. Opacities behind running
// text stay at or under ~0.2 so body text keeps its contrast over the glow.
const WIDE_POSES: Record<string, BrainPose> = {
  measurements: { x: 0.8, y: 0.52, size: 1.45, opacity: 0.18 },
  architecture: { x: 0.5, y: 0.62, size: 1.9, opacity: 0.09 },
  // Sits behind the sticky MRI slice, so it glows around the image's edges.
  pipeline: { x: 0.74, y: 0.44, size: 1.35, opacity: 0.2 },
  notclaimed: { x: 0.1, y: 0.6, size: 1.25, opacity: 0.12 },
  closing: { x: 0.71, y: 0.5, size: 1.05, opacity: 1 },
};
const NARROW_POSES: Record<string, BrainPose> = {
  measurements: { x: 0.5, y: 0.5, size: 1.1, opacity: 0.1 },
  architecture: { x: 0.5, y: 0.55, size: 1.3, opacity: 0.07 },
  pipeline: { x: 0.5, y: 0.45, size: 1.1, opacity: 0.1 },
  notclaimed: { x: 0.5, y: 0.55, size: 1.1, opacity: 0.08 },
  closing: { x: 0.5, y: 0.7, size: 0.75, opacity: 0.75 },
};

// Radians of turn per px scrolled (~300 degrees over the whole page).
const SCROLL_YAW = 0.0011;
const AUTO_YAW = 0.1; // rad/s, the idle drift the hero always had
const DRAG_YAW = 0.008; // rad per px dragged
const POINTER_TILT = 0.07; // rad at the viewport edge
const INTRO_SECONDS = 0.9; // fade-in once the mesh has loaded

interface Timeline {
  keys: PoseKey[];
  scanStart: number; // scrollY where the scan plane leaves the crown
  scanEnd: number; // scrollY where it reaches the base
}

/** Reads the hero anchor and every [data-brain-pose] section into scroll-
 * keyed poses. Called on mount, resize and any layout change. */
function measureTimeline(anchor: HTMLElement | null): Timeline {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const sy = window.scrollY;
  const maxScroll = Math.max(1, document.documentElement.scrollHeight - vh);
  const poses = vw >= WIDE_MIN_PX ? WIDE_POSES : NARROW_POSES;
  const keys: PoseKey[] = [];
  const at: Record<string, number> = {};

  if (anchor) {
    const r = anchor.getBoundingClientRect();
    keys.push({
      at: 0,
      pose: {
        x: (r.left + r.width / 2) / vw,
        y: (r.top + sy + r.height / 2) / vh,
        size: r.height / vh,
        opacity: 1,
      },
    });
  }
  document.querySelectorAll<HTMLElement>("[data-brain-pose]").forEach((el) => {
    const name = el.dataset.brainPose ?? "";
    const pose = poses[name];
    if (!pose) return;
    const r = el.getBoundingClientRect();
    // The pose is fully reached when the section's centre is at the
    // viewport's centre (or at the end of the page, if it never gets there).
    const centred = r.top + sy + r.height / 2 - vh / 2;
    const key = Math.min(Math.max(1, centred), maxScroll - 1);
    at[name] = key;
    keys.push({ at: key, pose });
  });
  // On one-column layouts the brain cannot step aside from the text, so it
  // must already be dim when the first ledger rows come up: an extra key
  // reaches the measurements pose as soon as that section's top is a little
  // above the viewport's bottom.
  const first = document.querySelector<HTMLElement>('[data-brain-pose="measurements"]');
  if (vw < WIDE_MIN_PX && first && at.measurements !== undefined) {
    const early = first.getBoundingClientRect().top + sy - vh * 0.75;
    if (early > 1 && early < at.measurements) keys.push({ at: early, pose: poses.measurements });
  }
  keys.sort((a, b) => a.at - b.at);
  return {
    keys,
    scanStart: at.measurements ?? vh,
    scanEnd: at.closing ?? maxScroll,
  };
}

const BRAIN_VERT = /* glsl */ `
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  varying float vObjY;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormalView = normalize(normalMatrix * normal);
    vViewDir = -mv.xyz;
    vObjY = position.y;
    gl_Position = projectionMatrix * mv;
  }
`;

// HeroBrain's shader plus three uniforms: uOpacity (pose), uScanY (plane
// height, object space, +Y = superior) and uScanOn (0 in the hero, 1 once the
// scan has started). With uScanOn = 0 and uOpacity = 1 it draws exactly what
// HeroBrain draws.
const BRAIN_FRAG = /* glsl */ `
  uniform vec3 uHolo;
  uniform vec3 uDeep;
  uniform float uTime;
  uniform float uOpacity;
  uniform float uScanY;
  uniform float uScanOn;
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  varying float vObjY;
  void main() {
    vec3 n = normalize(vNormalView);
    float facing = abs(dot(n, normalize(vViewDir)));
    float rim = pow(1.0 - facing, 2.2);
    float key = max(dot(n, normalize(vec3(-0.35, 0.6, 0.7))), 0.0);
    float scan = 0.88 + 0.12 * sin(vObjY * 70.0 - uTime * 0.9);
    float side = gl_FrontFacing ? 1.0 : 0.55;
    // Not yet reached by the scan plane: held at half brightness.
    float pending = step(vObjY, uScanY) * uScanOn;
    float lum = (0.10 + 0.9 * rim + 0.22 * key) * scan * side * (1.0 - 0.5 * pending);
    // Where the plane cuts the surface: a thin axial contour. Its width is
    // set in screen pixels (fwidth = how much vObjY changes across one
    // pixel), so it stays a crisp ~1.5px line at every zoom, plus a soft
    // halo four times wider.
    float w = max(fwidth(vObjY) * 1.5, 1e-5);
    float d = (vObjY - uScanY) / w;
    float contour = (exp(-d * d) + 0.3 * exp(-d * d / 16.0)) * uScanOn;
    vec3 col = mix(uDeep, uHolo, clamp(lum * 1.2, 0.0, 1.0));
    col = mix(col, vec3(0.80, 0.93, 1.0), contour);
    // Alpha must track brightness: additive blending adds rgb*alpha on a
    // transparent canvas. The contour is boosted relative to the dimmed brain
    // so the scan still reads behind text, but it scales with the pose too.
    float a = clamp(lum * 1.6, 0.0, 1.0) * uOpacity + contour * min(1.0, uOpacity * 2.0);
    gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
  }
`;

const DISC_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Floor glow under the brain (as in HeroBrain), scaled by uAlpha.
const GLOW_FRAG = /* glsl */ `
  uniform vec3 uHolo;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    float fall = pow(clamp(1.0 - r, 0.0, 1.0), 2.0) * 0.14;
    float ring = smoothstep(0.03, 0.0, abs(r - 0.78)) * 0.12;
    gl_FragColor = vec4(uHolo, min(fall + ring, 0.25) * uAlpha);
  }
`;

// The gantry: a thin ring with 72 ticks around the scan plane, plus a very
// faint fill so the plane itself reads as a surface.
const GANTRY_FRAG = /* glsl */ `
  uniform vec3 uHolo;
  uniform float uAlpha;
  varying vec2 vUv;
  void main() {
    vec2 p = vUv - 0.5;
    float r = length(p) * 2.0;
    float line = smoothstep(0.012, 0.0, abs(r - 0.93));
    float tickBand = smoothstep(0.03, 0.0, abs(r - 0.965));
    float tick = step(0.92, abs(cos(atan(p.y, p.x) * 36.0))) * tickBand;
    float fill = step(r, 0.93) * 0.03;
    gl_FragColor = vec4(uHolo, (line * 0.6 + tick * 0.35 + fill) * uAlpha);
  }
`;

function additive(fragmentShader: string, vertexShader: string, uniforms: Record<string, THREE.IUniform>) {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
}

interface SceneProps {
  assets: HeroAssets;
  anchorRef: RefObject<HTMLElement | null>;
}

function Scene({ assets, anchorRef }: SceneProps) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const rigRef = useRef<THREE.Group>(null); // tilt (whole assembly)
  const brainRef = useRef<THREE.Group>(null); // yaw (brain only; rings are round)
  const gantryRef = useRef<THREE.Mesh>(null);

  // Per-frame state lives in refs: none of it should ever re-render React.
  const timeline = useRef<Timeline>({ keys: [], scanStart: 1, scanEnd: 2 });
  const smoothY = useRef<number | null>(null);
  const autoYaw = useRef(0);
  const dragYaw = useRef(0);
  const shownDragYaw = useRef(0);
  const pointer = useRef({ x: 0, y: 0 });
  const tilt = useRef({ x: 0, y: 0 });
  const intro = useRef(0);

  // The mesh's vertical extent, for the scan plane's travel.
  const [yMin, yMax] = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const g of assets.geometries) {
      g.computeBoundingBox();
      const b = g.boundingBox;
      if (b) {
        lo = Math.min(lo, b.min.y);
        hi = Math.max(hi, b.max.y);
      }
    }
    return Number.isFinite(lo) ? [lo, hi] : [-0.6, 0.6];
  }, [assets]);

  const brainMaterial = useMemo(
    () =>
      additive(BRAIN_FRAG, BRAIN_VERT, {
        uHolo: { value: HOLO },
        uDeep: { value: HOLO_DEEP },
        uTime: { value: 0 },
        uOpacity: { value: 0 },
        uScanY: { value: 10 },
        uScanOn: { value: 0 },
      }),
    [],
  );
  const glowMaterial = useMemo(() => additive(GLOW_FRAG, DISC_VERT, { uHolo: { value: HOLO }, uAlpha: { value: 0 } }), []);
  const gantryMaterial = useMemo(
    () => additive(GANTRY_FRAG, DISC_VERT, { uHolo: { value: HOLO }, uAlpha: { value: 0 } }),
    [],
  );
  useEffect(
    () => () => {
      brainMaterial.dispose();
      glowMaterial.dispose();
      gantryMaterial.dispose();
    },
    [brainMaterial, glowMaterial, gantryMaterial],
  );

  // The camera looks at the brain's centre once; after that only its view
  // offset and zoom change.
  useEffect(() => {
    camera.lookAt(0, 0, 0);
  }, [camera]);

  // Re-measure the scroll timeline whenever layout can have moved.
  useEffect(() => {
    const measure = () => {
      timeline.current = measureTimeline(anchorRef.current);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(document.body);
    window.addEventListener("resize", measure);
    void document.fonts?.ready.then(measure);
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [anchorRef]);

  // Mouse tilt (mouse only: on touch, a finger on the page is a scroll).
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      pointer.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      pointer.current.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  // Drag on the hero anchor spins the brain, as the old OrbitControls did.
  // The anchor has touch-action: pan-y, so vertical swipes still scroll.
  useEffect(() => {
    const el = anchorRef.current;
    if (!el) return;
    let lastX: number | null = null;
    const down = (e: PointerEvent) => {
      lastX = e.clientX;
      el.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (lastX === null) return;
      dragYaw.current += (e.clientX - lastX) * DRAG_YAW;
      lastX = e.clientX;
    };
    const up = (e: PointerEvent) => {
      lastX = null;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
    };
  }, [anchorRef]);

  useFrame((state, rawDelta) => {
    // A backgrounded tab resumes with a huge delta; cap it so nothing jumps.
    const delta = Math.min(rawDelta, 0.1);
    const W = state.size.width;
    const H = state.size.height;

    // Eased scroll: the scene trails the page a little, which reads as weight.
    const target = window.scrollY;
    if (smoothY.current === null) smoothY.current = target;
    const prev = smoothY.current;
    smoothY.current += (target - prev) * (1 - Math.exp(-delta * 7));
    const y = smoothY.current;
    const speed = Math.abs(y - prev) / Math.max(delta, 1e-3); // px/s

    const tl = timeline.current;
    const pose = poseAt(tl.keys, y);
    intro.current = Math.min(1, intro.current + delta / INTRO_SECONDS);
    const opacity = pose.opacity * easeOutCubic(intro.current);

    // Place: lens-shift the brain's centre to the pose's screen point, zoom.
    camera.zoom = Math.max(0.05, pose.size);
    camera.setViewOffset(W, H, W / 2 - pose.x * W, H / 2 - pose.y * H, W, H);
    camera.updateProjectionMatrix();

    // Scan plane: fades in after the hero, sweeps crown to base, and the
    // gantry fades out as it finishes so the closing brain is whole.
    const scanOn = smoothstep(0, tl.scanStart, y);
    const scanT = progressBetween(tl.scanStart, tl.scanEnd, y);
    const scanY = yMax + 0.02 - scanT * (yMax - yMin + 0.04);
    const done = 1 - smoothstep(0.9, 1, scanT);

    const u = brainMaterial.uniforms;
    u.uTime.value += delta * (1 + Math.min(speed / 600, 4));
    u.uOpacity.value = opacity;
    u.uScanY.value = scanY;
    u.uScanOn.value = scanOn * done;
    glowMaterial.uniforms.uAlpha.value = opacity;
    gantryMaterial.uniforms.uAlpha.value = scanOn * done * Math.min(1, opacity * 2.5);
    if (gantryRef.current) gantryRef.current.position.y = scanY;

    // Turn: idle drift + scroll + drag (eased) + a little toward the mouse.
    autoYaw.current += delta * AUTO_YAW;
    shownDragYaw.current += (dragYaw.current - shownDragYaw.current) * (1 - Math.exp(-delta * 10));
    const k = 1 - Math.exp(-delta * 3);
    tilt.current.x += (pointer.current.y * POINTER_TILT - tilt.current.x) * k;
    tilt.current.y += (pointer.current.x * POINTER_TILT * 1.6 - tilt.current.y) * k;
    if (brainRef.current) {
      brainRef.current.rotation.y = autoYaw.current + y * SCROLL_YAW + shownDragYaw.current + tilt.current.y;
    }
    if (rigRef.current) rigRef.current.rotation.x = tilt.current.x;
  });

  return (
    <group ref={rigRef}>
      <mesh rotation-x={-Math.PI / 2} position={[0, yMin - 0.12, 0]} material={glowMaterial}>
        <circleGeometry args={[1.05, 64]} />
      </mesh>
      <mesh ref={gantryRef} rotation-x={-Math.PI / 2} position={[0, yMax, 0]} material={gantryMaterial}>
        <circleGeometry args={[1.08, 128]} />
      </mesh>
      <group ref={brainRef}>
        {assets.geometries.map((g, i) => (
          <mesh key={i} geometry={g} material={brainMaterial} />
        ))}
      </group>
    </group>
  );
}

export interface ScrollBrainProps {
  /** The empty box in the hero column the brain starts in (and can be dragged on). */
  anchorRef: RefObject<HTMLElement | null>;
  /** Called with the on-screen mesh's caption once it has loaded. */
  onCaption: (caption: string) => void;
}

export function ScrollBrain({ anchorRef, onCaption }: ScrollBrainProps) {
  const assets = useHeroAssets();
  useEffect(() => {
    if (assets) onCaption(assets.caption);
  }, [assets, onCaption]);

  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0">
      <Canvas
        camera={{ position: CAMERA_POS, fov: FOV }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        style={{ pointerEvents: "none" }}
      >
        {assets && <Scene assets={assets} anchorRef={anchorRef} />}
      </Canvas>
    </div>
  );
}
