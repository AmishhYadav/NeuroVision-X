// The landing page's hero visual under prefers-reduced-motion: a still blue
// "hologram" of a cortical surface in the hero column. With motion allowed the
// page uses ScrollBrain instead (the same hologram, fixed behind the page and
// driven by scroll). Mesh loading and colours are shared via brainAssets.ts.
//
// Why the folds read clearly: the fragment shader is dominated by a Fresnel
// rim term. Sulcal walls are seen at grazing angles, so they glow, while the
// gyral crowns (facing the camera) only get a faint base fill. Additive
// blending with no depth writes makes the far side of the brain add faintly
// through the near side, which is the "projected light" look.
import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useReducedMotion } from "motion/react";
import * as THREE from "three";
import { HOLO, HOLO_DEEP, useHeroAssets, type HeroAssets } from "./brainAssets";

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

const BRAIN_FRAG = /* glsl */ `
  uniform vec3 uHolo;
  uniform vec3 uDeep;
  uniform float uTime;
  varying vec3 vNormalView;
  varying vec3 vViewDir;
  varying float vObjY;
  void main() {
    vec3 n = normalize(vNormalView);
    float facing = abs(dot(n, normalize(vViewDir)));
    // Grazing angles (sulcal walls, silhouette) light up.
    float rim = pow(1.0 - facing, 2.2);
    // Soft key light from upper-left-front so the 3D form reads.
    float key = max(dot(n, normalize(vec3(-0.35, 0.6, 0.7))), 0.0);
    // Scanlines in object Y, drifting upward.
    float scan = 0.88 + 0.12 * sin(vObjY * 70.0 - uTime * 0.9);
    // Back faces add less, so the far side hints through without clutter.
    float side = gl_FrontFacing ? 1.0 : 0.55;
    float lum = (0.10 + 0.9 * rim + 0.22 * key) * scan * side;
    vec3 col = mix(uDeep, uHolo, clamp(lum * 1.2, 0.0, 1.0));
    // Alpha must track brightness: additive blending adds rgb*alpha, and the
    // canvas is transparent, so alpha 1 on black would paint opaque black.
    gl_FragColor = vec4(col, clamp(lum * 1.6, 0.0, 1.0));
  }
`;

// A faint projector glow on the floor under the brain: radial falloff plus one
// thin ring. Additive, opacity-limited.
const GLOW_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const GLOW_FRAG = /* glsl */ `
  uniform vec3 uHolo;
  varying vec2 vUv;
  void main() {
    float r = length(vUv - 0.5) * 2.0;
    float fall = pow(clamp(1.0 - r, 0.0, 1.0), 2.0) * 0.14;
    float ring = smoothstep(0.03, 0.0, abs(r - 0.78)) * 0.12;
    float a = min(fall + ring, 0.25);
    // alpha = a so black/empty areas stay transparent on the alpha canvas.
    gl_FragColor = vec4(uHolo, a);
  }
`;

function Scene({ assets, animate }: { assets: HeroAssets; animate: boolean }) {
  const groupRef = useRef<THREE.Group>(null);

  const brainMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: BRAIN_VERT,
        fragmentShader: BRAIN_FRAG,
        uniforms: { uHolo: { value: HOLO }, uDeep: { value: HOLO_DEEP }, uTime: { value: 0 } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      }),
    [],
  );
  const glowMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: GLOW_VERT,
        fragmentShader: GLOW_FRAG,
        uniforms: { uHolo: { value: HOLO } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );
  useEffect(
    () => () => {
      brainMaterial.dispose();
      glowMaterial.dispose();
    },
    [brainMaterial, glowMaterial],
  );

  // No allocations per frame: one rotation add and one uniform write.
  useFrame((_, delta) => {
    if (!animate) return;
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.12;
    brainMaterial.uniforms.uTime.value += delta;
  });

  return (
    <>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.9, 0]} material={glowMaterial}>
        <circleGeometry args={[1.05, 64]} />
      </mesh>
      <group ref={groupRef}>
        {assets.geometries.map((g, i) => (
          <mesh key={i} geometry={g} material={brainMaterial} />
        ))}
      </group>
    </>
  );
}

export function HeroBrain() {
  const assets = useHeroAssets();
  const reduce = useReducedMotion() ?? false;
  return (
    <figure className="m-0">
      <div className="relative mx-auto aspect-square w-full max-w-[640px]">
        {/* 3/4 left-lateral view: camera on +X (patient left), a little
            anterior (-Z) and raised so the top of the brain tilts toward us. */}
        <Canvas
          camera={{ position: [3.25, 1.63, -1.75], fov: 40 }}
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }}
          frameloop={reduce ? "demand" : "always"}
        >
          {assets && <Scene assets={assets} animate={!reduce} />}
          <OrbitControls
            enablePan={false}
            enableZoom={false}
            enableDamping
            dampingFactor={0.08}
            rotateSpeed={0.4}
            minPolarAngle={Math.PI / 2 - 0.9}
            maxPolarAngle={Math.PI / 2 + 0.5}
          />
        </Canvas>
        {!assets && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm text-text-dim">Loading brain surface…</span>
          </div>
        )}
      </div>
      <figcaption className="mt-2 text-xs text-text-dim">{assets ? assets.caption : " "}</figcaption>
    </figure>
  );
}
