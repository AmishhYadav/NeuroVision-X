// The landing page's hero visual: a real reconstructed brain shell (marching
// cubes over BraTS2021_00000's own skull-stripped mask, extracted offline -
// see /public/twin) with a properly opaque, lit material instead of the
// thin translucent shell used for the in-tool digital twin, plus a
// procedural streamline layer wrapping the surface for visual texture.
//
// Two honesty notes, deliberate:
//   - This is a FIXED illustrative case, not case-aware. The real,
//     case-switching reconstruction lives in the tool itself
//     (components/BrainTwinScene.tsx) - this one exists only to give the
//     landing page a real, non-generic hero visual.
//   - The streamlines are a procedural visual motif, not real diffusion
//     tractography (BraTS is structural MRI only; this project has no DTI
//     data). They are never captioned as fibre-tract data anywhere in copy.
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { hexToRgb } from "../../lib/colors";
import { loadMesh, type MeshBuffers } from "../../lib/loadBinary";

const TWIN_BASE = "/twin";
const CLASS_HEX: Record<string, string> = {
  necrotic: "#56B4E9",
  oedema: "#009E73",
  enhancing: "#D55E00",
};

function toGeometry(buf: MeshBuffers): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(buf.position, 3));
  geom.setAttribute("normal", new THREE.BufferAttribute(buf.normal, 3));
  geom.setIndex(new THREE.BufferAttribute(buf.index, 1));
  return geom;
}

function rgbToThreeColor(hex: string): THREE.Color {
  const [r, g, b] = hexToRgb(hex);
  return new THREE.Color(r / 255, g / 255, b / 255);
}

interface Assets {
  brainLeft: THREE.BufferGeometry;
  brainRight: THREE.BufferGeometry;
  tumor: Record<string, THREE.BufferGeometry>;
}

function useHeroAssets(): Assets | null {
  const [assets, setAssets] = useState<Assets | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [brainLeft, brainRight, necrotic, oedema, enhancing] = await Promise.all([
        loadMesh(TWIN_BASE, "brain-left"),
        loadMesh(TWIN_BASE, "brain-right"),
        loadMesh(TWIN_BASE, "tumor-necrotic"),
        loadMesh(TWIN_BASE, "tumor-oedema"),
        loadMesh(TWIN_BASE, "tumor-enhancing"),
      ]);
      if (cancelled) return;
      setAssets({
        brainLeft: toGeometry(brainLeft),
        brainRight: toGeometry(brainRight),
        tumor: {
          necrotic: toGeometry(necrotic),
          oedema: toGeometry(oedema),
          enhancing: toGeometry(enhancing),
        },
      });
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return assets;
}

// Deterministic pseudo-random so the visual is stable across reloads.
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Procedural streamlines hugging just outside the brain surface - a visual motif, not tractography data (see file header). */
function Streamlines({ radius }: { radius: number }) {
  const rand = useMemo(() => mulberry32(7), []);
  const curves = useMemo(() => {
    const lines: THREE.Vector3[][] = [];
    for (let i = 0; i < 46; i++) {
      const latOffset = (rand() - 0.5) * Math.PI * 0.9;
      const tilt = rand() * Math.PI;
      const r = radius * (1.04 + rand() * 0.1);
      const points: THREE.Vector3[] = [];
      const segments = 48;
      for (let s = 0; s <= segments; s++) {
        const theta = (s / segments) * Math.PI * 2;
        const x = Math.cos(theta) * r;
        const z = Math.sin(theta) * r * Math.cos(latOffset);
        const y = Math.sin(theta) * r * Math.sin(latOffset) * 0.6 + latOffset * radius * 0.3;
        const v = new THREE.Vector3(x, y, z);
        v.applyAxisAngle(new THREE.Vector3(0, 1, 0), tilt);
        points.push(v);
      }
      lines.push(points);
    }
    return lines;
  }, [radius, rand]);

  return (
    <group>
      {curves.map((pts, i) => (
        <Line
          key={i}
          points={pts}
          color="#7fd4ff"
          transparent
          opacity={0.16}
          lineWidth={1}
        />
      ))}
    </group>
  );
}

function Scene({ assets }: { assets: Assets }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.09;
  });

  const shellMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: 0xc7ccd4,
        roughness: 0.42,
        metalness: 0.04,
        clearcoat: 0.35,
        clearcoatRoughness: 0.5,
        transmission: 0.18,
        thickness: 0.4,
        transparent: true,
        opacity: 0.92,
        side: THREE.DoubleSide,
      }),
    [],
  );

  return (
    <group ref={groupRef}>
      <mesh geometry={assets.brainLeft} material={shellMaterial} />
      <mesh geometry={assets.brainRight} material={shellMaterial} />
      {Object.entries(assets.tumor).map(([name, geom]) => (
        <mesh key={name} geometry={geom}>
          <meshStandardMaterial
            color={rgbToThreeColor(CLASS_HEX[name])}
            emissive={rgbToThreeColor(CLASS_HEX[name])}
            emissiveIntensity={0.5}
            roughness={0.3}
          />
        </mesh>
      ))}
      <Streamlines radius={1.05} />
    </group>
  );
}

export function HeroBrain() {
  const assets = useHeroAssets();
  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ position: [0, 0.2, 2.9], fov: 40 }} dpr={[1, 2]}>
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 4, 5]} intensity={1.6} color="#ffffff" />
        <directionalLight position={[-3, -2, -4]} intensity={0.5} color="#7fd4ff" />
        <pointLight position={[0, 0, 3]} intensity={0.4} color="#0b5fff" />
        {assets && <Scene assets={assets} />}
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          autoRotate={false}
          rotateSpeed={0.4}
          minPolarAngle={Math.PI / 2 - 0.6}
          maxPolarAngle={Math.PI / 2 + 0.6}
        />
      </Canvas>
      {!assets && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xs text-landing-text-dim">Loading real case geometry…</span>
        </div>
      )}
    </div>
  );
}
