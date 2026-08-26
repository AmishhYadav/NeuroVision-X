// The digital twin: a real 3D reconstruction of WHICHEVER case is currently
// selected in the tool, not a fixed demo case. The brain shell is a
// surface-nets isosurface of that case's own skull-stripped nonzero-voxel
// mask; the tumour sub-structures are the same treatment applied to its
// real label (or, if this case has no ground truth, its real saved
// prediction) - both computed in a Web Worker (workers/twinMesh.worker.ts)
// from the exact volumes useCaseData already fetched, so a different case
// selection produces a genuinely different tumour, sized and shaped as it
// really is.
//
// Interaction: drag to orbit, scroll/pinch to zoom (drei OrbitControls),
// click a hemisphere to slide the brain apart, click a tumour
// sub-structure to dolly in and read its real volume for this case.
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { hexToRgb } from "../lib/colors";
import type { TwinMeshRequest, TwinMeshResult } from "../workers/twinMesh.worker";

export type ClassName = "necrotic" | "oedema" | "enhancing";

const CLASS_HEX: Record<ClassName, string> = {
  necrotic: "#56B4E9",
  oedema: "#009E73",
  enhancing: "#D55E00",
};
const CLASS_LABEL: Record<ClassName, string> = {
  necrotic: "Necrotic core",
  oedema: "Oedema",
  enhancing: "Enhancing tumour",
};

interface MeshBuf {
  position: Float32Array;
  normal: Float32Array;
  index: Uint32Array;
}

function toGeometry(buf: MeshBuf): THREE.BufferGeometry {
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

export interface BrainTwinInput {
  caseId: string;
  shape: [number, number, number];
  spacing: [number, number, number];
  modalityVolumes: Uint8Array[];
  tumorMask: Uint8Array | null;
  tumorSource: "label" | "prediction" | null;
}

/** Owns the worker, dispatches one mesh job per case, and caches results by case id. */
function useTwinMesh(input: BrainTwinInput | null): {
  result: TwinMeshResult | null;
  geometries: {
    brainLeft: THREE.BufferGeometry;
    brainRight: THREE.BufferGeometry;
    tumor: Partial<Record<ClassName, THREE.BufferGeometry>>;
  } | null;
  computing: boolean;
} {
  const workerRef = useRef<Worker | null>(null);
  const cacheRef = useRef<Map<string, TwinMeshResult>>(new Map());
  const requestIdRef = useRef(0);
  const [result, setResult] = useState<TwinMeshResult | null>(null);
  const [computing, setComputing] = useState(false);

  useEffect(() => {
    workerRef.current = new Worker(new URL("../workers/twinMesh.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current.onmessage = (e: MessageEvent<TwinMeshResult>) => {
      if (e.data.requestId !== requestIdRef.current) return; // stale response from a superseded case
      cacheRef.current.set(e.data.caseId, e.data);
      setResult(e.data);
      setComputing(false);
    };
    return () => workerRef.current?.terminate();
  }, []);

  useEffect(() => {
    if (!input) {
      setResult(null);
      return;
    }
    const cached = cacheRef.current.get(input.caseId);
    if (cached) {
      setResult(cached);
      setComputing(false);
      return;
    }
    requestIdRef.current += 1;
    setComputing(true);
    const request: TwinMeshRequest = {
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      spacing: input.spacing,
      modalityVolumes: input.modalityVolumes,
      tumorMask: input.tumorMask,
      tumorSource: input.tumorSource,
    };
    // Deliberately NOT transferred: these are the SAME ArrayBuffers
    // useCaseData handed to the 2D viewport (Viewport.tsx / SliceRibbon.tsx
    // read caseData.volumes[modality].data directly). Transferring them
    // would detach them from the main thread and blank out the 2D view the
    // moment the twin computes. A structured-clone copy costs more, but the
    // 2D viewport must keep working while the twin is open.
    workerRef.current?.postMessage(request);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input?.caseId]);

  const geometries = useMemo(() => {
    if (!result) return null;
    const tumor: Partial<Record<ClassName, THREE.BufferGeometry>> = {};
    for (const [name, buf] of Object.entries(result.tumor)) {
      if (buf) tumor[name as ClassName] = toGeometry(buf);
    }
    return {
      brainLeft: toGeometry(result.brainLeft),
      brainRight: toGeometry(result.brainRight),
      tumor,
    };
  }, [result]);

  return { result, geometries, computing };
}

interface TwinModelProps {
  geometries: NonNullable<ReturnType<typeof useTwinMesh>["geometries"]>;
  separated: boolean;
  onToggleSeparate: () => void;
  selected: ClassName | null;
  onSelect: (c: ClassName | null) => void;
}

const HEMISPHERE_GAP = 0.55;

function TwinModel({ geometries, separated, onToggleSeparate, selected, onSelect }: TwinModelProps) {
  const groupRef = useRef<THREE.Group>(null);
  const leftRef = useRef<THREE.Group>(null);
  const rightRef = useRef<THREE.Group>(null);
  const userInteractedRef = useRef(false);

  useFrame((_, delta) => {
    if (groupRef.current && !userInteractedRef.current) {
      groupRef.current.rotation.y += delta * 0.15;
    }
    const targetShift = separated ? HEMISPHERE_GAP : 0;
    if (leftRef.current) {
      leftRef.current.position.x = THREE.MathUtils.damp(leftRef.current.position.x, -targetShift, 4, delta);
    }
    if (rightRef.current) {
      rightRef.current.position.x = THREE.MathUtils.damp(rightRef.current.position.x, targetShift, 4, delta);
    }
  });

  const shellMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xe7eaee,
        transparent: true,
        opacity: 0.16,
        roughness: 0.6,
        metalness: 0,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );

  const tumorClasses = Object.keys(geometries.tumor) as ClassName[];

  return (
    <group
      ref={groupRef}
      onPointerDown={() => {
        userInteractedRef.current = true;
      }}
    >
      <group ref={leftRef}>
        <mesh
          geometry={geometries.brainLeft}
          material={shellMaterial}
          onClick={(e: ThreeEvent<MouseEvent>) => {
            e.stopPropagation();
            onToggleSeparate();
          }}
        />
      </group>
      <group ref={rightRef}>
        <mesh
          geometry={geometries.brainRight}
          material={shellMaterial}
          onClick={(e: ThreeEvent<MouseEvent>) => {
            e.stopPropagation();
            onToggleSeparate();
          }}
        />
      </group>

      {tumorClasses.map((cls) => (
        <mesh
          key={cls}
          geometry={geometries.tumor[cls]}
          onClick={(e: ThreeEvent<MouseEvent>) => {
            e.stopPropagation();
            onSelect(selected === cls ? null : cls);
          }}
        >
          <meshStandardMaterial
            color={rgbToThreeColor(CLASS_HEX[cls])}
            roughness={0.35}
            metalness={0.05}
            emissive={rgbToThreeColor(CLASS_HEX[cls])}
            emissiveIntensity={selected === cls ? 0.35 : 0.08}
          />
        </mesh>
      ))}
    </group>
  );
}

function CameraDolly({
  controlsRef,
  target,
}: {
  controlsRef: React.RefObject<import("three-stdlib").OrbitControls | null>;
  target: THREE.Vector3;
}) {
  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;
    controls.target.x = THREE.MathUtils.damp(controls.target.x, target.x, 4, delta);
    controls.target.y = THREE.MathUtils.damp(controls.target.y, target.y, 4, delta);
    controls.target.z = THREE.MathUtils.damp(controls.target.z, target.z, 4, delta);
    controls.update();
  });
  return null;
}

const DEFAULT_TARGET = new THREE.Vector3(0, 0, 0);

export function BrainTwinScene({ input }: { input: BrainTwinInput | null }) {
  const { result, geometries, computing } = useTwinMesh(input);
  const [separated, setSeparated] = useState(false);
  const [selected, setSelected] = useState<ClassName | null>(null);
  const controlsRef = useRef<import("three-stdlib").OrbitControls | null>(null);

  // A new case resets the view - a previous case's separated/selected state
  // has no meaning for a different tumour.
  useEffect(() => {
    setSeparated(false);
    setSelected(null);
  }, [input?.caseId]);

  const dollyTarget = useMemo(() => {
    if (selected && result?.tumorCentroidScene) {
      const [x, y, z] = result.tumorCentroidScene;
      return new THREE.Vector3(x, y, z);
    }
    return DEFAULT_TARGET;
  }, [selected, result]);

  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [0, 0.3, 2.6], fov: 42 }}
        onPointerMissed={() => setSelected(null)}
        dpr={[1, 2]}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[2, 3, 4]} intensity={1.1} />
        <directionalLight position={[-2, -1, -3]} intensity={0.3} />
        <Suspense fallback={null}>
          {geometries && (
            <>
              <TwinModel
                geometries={geometries}
                separated={separated}
                onToggleSeparate={() => setSeparated((v) => !v)}
                selected={selected}
                onSelect={setSelected}
              />
              <CameraDolly controlsRef={controlsRef} target={dollyTarget} />
            </>
          )}
        </Suspense>
        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom
          minDistance={1.1}
          maxDistance={4.5}
          rotateSpeed={0.6}
          zoomSpeed={0.7}
        />
      </Canvas>

      {!geometries && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xs text-text-dim">
            {computing ? "Reconstructing this case's geometry…" : "Pick a case to build its twin."}
          </span>
        </div>
      )}

      {geometries && (
        <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col gap-1 font-mono text-[11px] text-text-dim">
          <span>Drag to orbit · scroll to zoom · click a hemisphere to separate</span>
        </div>
      )}

      {geometries && selected && result && (
        <div className="liquid-glass absolute top-3 right-3 w-56 rounded-md border border-surface-seam px-3 py-3">
          <div className="mb-1 flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
              style={{ backgroundColor: CLASS_HEX[selected] }}
              aria-hidden="true"
            />
            <span className="font-mono text-xs text-text-primary">{CLASS_LABEL[selected]}</span>
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="ml-auto text-text-dim hover:text-text-primary"
              aria-label="Close tumour detail"
            >
              ×
            </button>
          </div>
          <p className="tabular font-mono text-[11px] text-text-secondary">
            {(result.classVolumesMl[selected] ?? 0).toFixed(2)} ml, this case
          </p>
          <p className="mt-1 font-mono text-[10px] leading-relaxed text-text-dim">
            {result.tumorSource === "label" ? "Real ground-truth label" : "Real saved model prediction"},{" "}
            {result.caseId}.
          </p>
        </div>
      )}
    </div>
  );
}
