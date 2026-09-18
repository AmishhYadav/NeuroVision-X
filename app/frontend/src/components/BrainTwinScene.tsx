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
// The twin can also be PAINTED with the same per-voxel layers the 2D slice
// view shows (predictive entropy, the conformal band, Grad-CAM) - each
// tumour class's mesh vertices are coloured by that layer's value sampled
// one voxel inward from the surface (see lib/vertexScalars.ts for why: a
// vertex sits exactly on the class boundary, where an entropy-like quantity
// is maximal by construction, so sampling AT the surface would paint every
// tumour uniformly "maximally uncertain" and say nothing). Switching which
// layer is active never re-runs the surface-nets mesh pass - only a cheap
// resample against geometry the worker already computed - because the mesh
// shape does not depend on which layer is being displayed.
//
// Interaction: drag to orbit, scroll/pinch to zoom (drei OrbitControls),
// click a hemisphere to slide the brain apart, click a tumour
// sub-structure to dolly in and read its real volume for this case.
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { AtlasStructureRow } from "../api";
import { hexToRgb } from "../lib/colors";
import { scalarsToColors, type VertexLayerKind } from "../lib/vertexColors";
import { structureRow } from "../lib/atlasSelection";
import { structureColor } from "../lib/structureColors";
import type {
  TwinMeshRequest,
  TwinMeshResult,
  TwinSampleRequest,
  TwinSampleResult,
  TwinStructuresRequest,
  TwinStructuresResult,
} from "../workers/twinMesh.worker";
import { Legend } from "./Legend";

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

/** structureColor's [0,1] RGB triple as a CSS hex string, for the legend/detail-card swatches (three.js meshes use structureColor directly - see TwinModel.renderStructureMesh). */
function structureColorHex(index: number): string {
  const [r, g, b] = structureColor(index);
  const toHex = (c: number) => Math.round(c * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export interface BrainTwinInput {
  caseId: string;
  shape: [number, number, number];
  spacing: [number, number, number];
  modalityVolumes: Uint8Array[];
  tumorMask: Uint8Array | null;
  tumorSource: "label" | "prediction" | null;
}

/**
 * The one scalar layer currently selected to paint the tumour meshes.
 * `kind` is the exact backend `X-Uncertainty-Kind` value and decides how
 * `scalarsToColors` colours it. `key` is a separate cache key because two
 * buffers can share a `kind` (the conformal band and Grad-CAM are each
 * fetched for both the WT and TC regions) - keying the scene's scalar cache
 * by `kind` alone would let switching WT<->TC silently reuse the wrong
 * region's stale scalars, so the caller (ClinicalStudyViewer) passes its own
 * selector value (e.g. `"band_wt"`) as `key` to force a fresh sample on
 * every region switch.
 */
export interface TwinActiveLayer {
  key: string;
  kind: VertexLayerKind;
  data: Uint8Array;
}

interface TwinGeometries {
  brainLeft: THREE.BufferGeometry;
  brainRight: THREE.BufferGeometry;
  tumor: Partial<Record<ClassName, THREE.BufferGeometry>>;
}

/** Atlas structure shells to draw - see BrainTwinSceneProps.atlas for the field meanings. */
export interface TwinAtlasInput {
  volume: Uint8Array;
  selection: number[];
  table: AtlasStructureRow[];
}

/**
 * Owns the worker, dispatches one mesh job per case, and caches results by
 * case id. Also resolves `activeLayer`'s scalars for the current case: if
 * they were not already included in the case's mesh response (e.g. the
 * layer was picked after the mesh finished), it sends a cheap "sample"
 * message that reuses the mesh's own voxel geometry instead of re-meshing.
 * And, when `atlas` names a non-empty selection, meshes those atlas
 * structures against the case's own scene frame (see TwinMeshResult.frame),
 * again without re-meshing the brain or tumour.
 */
function useTwinMesh(
  input: BrainTwinInput | null,
  activeLayer: TwinActiveLayer | null,
  atlas: TwinAtlasInput | null,
): {
  result: TwinMeshResult | null;
  geometries: TwinGeometries | null;
  computing: boolean;
  /** activeLayer's per-class scalars for the CURRENT case, or null if there is no active layer or its scalars have not arrived yet. */
  activeLayerScalars: Partial<Record<ClassName, Float32Array>> | null;
  /** Meshed shells for the CURRENT case + atlas.selection, or null if atlas is absent/empty or not yet meshed. */
  structureGeometries: Map<number, THREE.BufferGeometry> | null;
  structuresComputing: boolean;
} {
  const workerRef = useRef<Worker | null>(null);
  // Base mesh geometry per case - independent of which layer is active, so
  // switching layers never touches this cache.
  const meshCacheRef = useRef<Map<string, TwinMeshResult>>(new Map());
  // Sampled scalars per case, then per the caller's `key` (see
  // TwinActiveLayer's docstring on why `key` and not `kind`).
  const layerCacheRef = useRef<Map<string, Map<string, Partial<Record<ClassName, Float32Array>>>>>(
    new Map(),
  );
  // Meshed atlas structures per case, then per selection key
  // (`selection.join(",")` - a different selection is a different set of
  // shells, but revisiting a previously-seen selection, e.g. switching case
  // and back, is free).
  const structuresCacheRef = useRef<Map<string, Map<string, TwinStructuresResult["structures"]>>>(
    new Map(),
  );
  const requestIdRef = useRef(0);
  // A "structures" request shares this same requestId counter with "mesh"
  // and "sample" requests (see the pendingRef type below), so the two
  // request kinds serialise against each other rather than racing - a
  // structures request fired just after a sample request will simply wait
  // its turn. Both are cheap compared to the initial mesh pass, so this is
  // acceptable rather than needing its own independent id space.
  const pendingRef = useRef<
    { type: "mesh" | "sample" | "structures"; caseId: string; key?: string } | null
  >(null);
  // Read inside the mesh-request effect without adding activeLayer as a
  // dependency - that effect must stay keyed on caseId alone (a case switch
  // is the only thing that should trigger a re-mesh).
  const activeLayerRef = useRef(activeLayer);
  const [result, setResult] = useState<TwinMeshResult | null>(null);
  const [computing, setComputing] = useState(false);
  // layerCacheRef is a plain ref (mutated off the React render cycle by the
  // worker's onmessage), so bump this to force activeLayerScalars to
  // recompute after a sample response lands.
  const [layerVersion, setLayerVersion] = useState(0);
  // Same idea as layerVersion, but for structuresCacheRef - forces
  // structureGeometries to recompute after a "structures" response lands.
  const [structuresVersion, setStructuresVersion] = useState(0);
  const [structuresComputing, setStructuresComputing] = useState(false);

  useEffect(() => {
    activeLayerRef.current = activeLayer;
  }, [activeLayer]);

  useEffect(() => {
    workerRef.current = new Worker(new URL("../workers/twinMesh.worker.ts", import.meta.url), {
      type: "module",
    });
    workerRef.current.onmessage = (
      e: MessageEvent<TwinMeshResult | TwinSampleResult | TwinStructuresResult>,
    ) => {
      const data = e.data;
      if (data.requestId !== requestIdRef.current) return; // stale response from a superseded request
      const pending = pendingRef.current;

      if ("kind" in data && data.kind === "sample") {
        // A sample request only ever carries ONE layer, so its single
        // entry is the one being resolved.
        const perClass = Object.values(data.layerScalars)[0];
        if (perClass && pending?.type === "sample" && pending.key) {
          let caseLayers = layerCacheRef.current.get(data.caseId);
          if (!caseLayers) {
            caseLayers = new Map();
            layerCacheRef.current.set(data.caseId, caseLayers);
          }
          caseLayers.set(pending.key, perClass);
          setLayerVersion((v) => v + 1);
        }
        setComputing(false);
        return;
      }

      if ("kind" in data && data.kind === "structures") {
        if (pending?.type === "structures" && pending.key) {
          let caseStructures = structuresCacheRef.current.get(data.caseId);
          if (!caseStructures) {
            caseStructures = new Map();
            structuresCacheRef.current.set(data.caseId, caseStructures);
          }
          caseStructures.set(pending.key, data.structures);
          setStructuresVersion((v) => v + 1);
        }
        setStructuresComputing(false);
        return;
      }

      meshCacheRef.current.set(data.caseId, data);
      // The mesh request may itself have carried the active layer (see the
      // request-building effect below) - fold its result into the same
      // per-key cache a later "sample" response would use, so the two paths
      // are indistinguishable to activeLayerScalars.
      if (data.layerScalars && pending?.type === "mesh" && pending.key) {
        const perClass = Object.values(data.layerScalars)[0];
        if (perClass) {
          let caseLayers = layerCacheRef.current.get(data.caseId);
          if (!caseLayers) {
            caseLayers = new Map();
            layerCacheRef.current.set(data.caseId, caseLayers);
          }
          caseLayers.set(pending.key, perClass);
        }
      }
      setResult(data);
      setComputing(false);
    };
    return () => workerRef.current?.terminate();
  }, []);

  // Mesh request: fires only on a case switch. Always asks the worker to
  // keep voxel geometry (`keepVoxelGeometry: true`) so a later layer switch
  // can be served by a "sample" message instead of a re-mesh, and includes
  // whichever layer is active right now so the common case (a layer is
  // already selected when a new case loads) costs zero extra round trips.
  useEffect(() => {
    if (!input) {
      setResult(null);
      return;
    }
    const cached = meshCacheRef.current.get(input.caseId);
    if (cached) {
      setResult(cached);
      setComputing(false);
      return;
    }
    requestIdRef.current += 1;
    setComputing(true);
    const layer = activeLayerRef.current;
    pendingRef.current = { type: "mesh", caseId: input.caseId, key: layer?.key };
    const request: TwinMeshRequest = {
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      // The worker uses `spacing` only to convert voxel counts to mL for
      // `classVolumesMl` - the mesh geometry itself assumes every case sits
      // on the 1 mm SRI24 grid, which holds for both the research (`/cases`)
      // and clinical (`/clinical/jobs`) paths, so spacing never rescales the
      // isosurface itself.
      spacing: input.spacing,
      modalityVolumes: input.modalityVolumes,
      tumorMask: input.tumorMask,
      tumorSource: input.tumorSource,
      scalarLayers: layer ? { [layer.kind]: layer.data } : undefined,
      keepVoxelGeometry: true,
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

  // Layer request: fires whenever the active layer changes (or a cached
  // case turns out not to have this layer's scalars yet) - never re-meshes,
  // only resamples the case's already-computed voxel geometry.
  useEffect(() => {
    if (!input || !activeLayer) return;
    const cachedMesh = meshCacheRef.current.get(input.caseId);
    if (!cachedMesh || !cachedMesh.voxelGeometry) return; // mesh not ready yet; the mesh effect above will carry this layer once it lands
    const caseLayers = layerCacheRef.current.get(input.caseId);
    if (caseLayers?.has(activeLayer.key)) return; // already sampled

    requestIdRef.current += 1;
    pendingRef.current = { type: "sample", caseId: input.caseId, key: activeLayer.key };
    const request: TwinSampleRequest = {
      kind: "sample",
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      layers: { [activeLayer.kind]: activeLayer.data },
      voxelGeometry: cachedMesh.voxelGeometry,
    };
    workerRef.current?.postMessage(request);
  }, [input, activeLayer, result]);

  // Structures request: fires whenever the case, the atlas volume, or the
  // selection changes (or a cached case turns out not to have THIS selection
  // meshed yet) - never re-meshes the brain/tumour, only meshes the newly
  // selected structures against the case's own scene frame (result.frame).
  useEffect(() => {
    if (!input || !atlas || atlas.selection.length === 0) return;
    const cachedMesh = meshCacheRef.current.get(input.caseId);
    if (!cachedMesh) return; // mesh (and its frame) not ready yet
    const key = atlas.selection.join(",");
    const caseStructures = structuresCacheRef.current.get(input.caseId);
    if (caseStructures?.has(key)) return; // already meshed

    requestIdRef.current += 1;
    setStructuresComputing(true);
    pendingRef.current = { type: "structures", caseId: input.caseId, key };
    const request: TwinStructuresRequest = {
      kind: "structures",
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      // Deliberately NOT transferred (see TwinStructuresRequest's docstring)
      // - the caller (the report/atlas panel a later unit wires up) keeps
      // its own copy of this volume for the next selection change.
      atlas: atlas.volume,
      selection: atlas.selection,
      frame: cachedMesh.frame,
    };
    workerRef.current?.postMessage(request);
  }, [input, atlas, result]);

  const geometries = useMemo<TwinGeometries | null>(() => {
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

  const activeLayerScalars = useMemo(() => {
    if (!input || !activeLayer) return null;
    return layerCacheRef.current.get(input.caseId)?.get(activeLayer.key) ?? null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input?.caseId, activeLayer, layerVersion, result]);

  const structureGeometries = useMemo(() => {
    if (!input || !atlas || atlas.selection.length === 0) return null;
    const key = atlas.selection.join(",");
    const structures = structuresCacheRef.current.get(input.caseId)?.get(key);
    if (!structures) return null;
    const out = new Map<number, THREE.BufferGeometry>();
    for (const [indexStr, buf] of Object.entries(structures)) {
      out.set(Number(indexStr), toGeometry(buf));
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input?.caseId, atlas, structuresVersion]);

  return {
    result,
    geometries,
    computing,
    activeLayerScalars,
    structureGeometries,
    structuresComputing,
  };
}

interface TwinModelProps {
  geometries: NonNullable<ReturnType<typeof useTwinMesh>["geometries"]>;
  separated: boolean;
  onToggleSeparate: () => void;
  selected: ClassName | null;
  onSelect: (c: ClassName | null) => void;
  /** Per-class vertex RGB from scalarsToColors, or absent/no-entry -> that class keeps its flat class colour. */
  colorsByClass: Partial<Record<ClassName, Float32Array>> | null;
  /** Meshed atlas structure shells to render, keyed by atlas index, or null when none are selected/ready. */
  structureGeometries: Map<number, THREE.BufferGeometry> | null;
  /** Atlas index to draw more opaque/emissive, or null/absent for none. */
  highlightedStructure?: number | null;
  /** Fired on a shell click (its index) or on empty space (null, via the canvas's onPointerMissed). */
  onStructureSelect?: (index: number | null) => void;
}

const HEMISPHERE_GAP = 0.55;
// Selection highlight when a class is vertex-painted: white rather than the
// class hue, so the glow reads as "selected" without fighting the layer's
// own colour ramp (which the class hue would, since it's what painting
// replaces).
const PAINTED_EMISSIVE = new THREE.Color(0xffffff);

function TwinModel({
  geometries,
  separated,
  onToggleSeparate,
  selected,
  onSelect,
  colorsByClass,
  structureGeometries,
  highlightedStructure = null,
  onStructureSelect,
}: TwinModelProps) {
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

  // Which hemisphere group each structure shell belongs to, so it slides
  // apart WITH that hemisphere when the brain is separated (same rule
  // splitMesh above uses for the brain itself: scene X < 0 is patient LEFT -
  // see twinMesh.worker.ts's axis-convention comment). Computed once per
  // structureGeometries change via computeBoundingBox() rather than on every
  // frame - a structure shell's geometry never moves once meshed, only the
  // group it sits in does.
  const structureSides = useMemo(() => {
    const sides = new Map<number, "left" | "right">();
    if (!structureGeometries) return sides;
    for (const [index, geom] of structureGeometries) {
      geom.computeBoundingBox();
      const bbox = geom.boundingBox;
      const centerX = bbox ? (bbox.min.x + bbox.max.x) / 2 : 0;
      sides.set(index, centerX < 0 ? "left" : "right");
    }
    return sides;
  }, [structureGeometries]);

  const leftStructures: [number, THREE.BufferGeometry][] = [];
  const rightStructures: [number, THREE.BufferGeometry][] = [];
  if (structureGeometries) {
    for (const [index, geom] of structureGeometries) {
      (structureSides.get(index) === "left" ? leftStructures : rightStructures).push([index, geom]);
    }
  }

  // Translucent atlas structure shell: opacity/emissive step up when this
  // index is the highlighted one (report-table hover/click, controlled by
  // the parent), same pattern as the tumour meshes' own selected-vs-not
  // emissiveIntensity step below. `stopPropagation` keeps a shell click from
  // also toggling hemisphere separation (the click would otherwise bubble to
  // the hemisphere group's own onClick).
  const renderStructureMesh = (index: number, geom: THREE.BufferGeometry) => {
    const isHighlighted = highlightedStructure === index;
    const [r, g, b] = structureColor(index);
    const color = new THREE.Color(r, g, b);
    return (
      <mesh
        key={`structure-${index}`}
        geometry={geom}
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation();
          onStructureSelect?.(index);
        }}
      >
        <meshStandardMaterial
          transparent
          opacity={isHighlighted ? 0.55 : 0.28}
          roughness={0.5}
          depthWrite={false}
          side={THREE.DoubleSide}
          color={color}
          emissive={color}
          emissiveIntensity={isHighlighted ? 0.35 : 0.08}
        />
      </mesh>
    );
  };

  // Imperative, not a JSX attribute prop: BufferGeometry attributes are
  // mutated directly (same pattern toGeometry already uses for
  // position/normal/index), and this needs to re-run whenever EITHER the
  // mesh geometry OR the active layer's colours change, independently of
  // each other. `deleteAttribute` (rather than leaving stale colours
  // attached) is what lets a class fall back to its flat colour the moment
  // no layer is active, or the moment this class has no scalars for the
  // active layer.
  useEffect(() => {
    for (const cls of Object.keys(geometries.tumor) as ClassName[]) {
      const geom = geometries.tumor[cls];
      if (!geom) continue;
      const colors = colorsByClass?.[cls];
      if (colors) {
        geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      } else if (geom.hasAttribute("color")) {
        geom.deleteAttribute("color");
      }
    }
  }, [geometries, colorsByClass]);

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
        {leftStructures.map(([index, geom]) => renderStructureMesh(index, geom))}
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
        {rightStructures.map(([index, geom]) => renderStructureMesh(index, geom))}
      </group>

      {/* Tumour meshes render AFTER the atlas shells above (JSX/paint order),
          so a tumour surface is never hidden behind a translucent structure
          shell occupying the same space. */}
      {tumorClasses.map((cls) => {
        const painted = Boolean(colorsByClass?.[cls]);
        return (
          <mesh
            // `painted` in the key forces React to remount the material
            // when painting toggles on/off, rather than mutating an
            // existing THREE.MeshStandardMaterial's `vertexColors` in
            // place - `vertexColors` changes the compiled shader
            // (USE_COLOR), which three.js only picks up on a fresh
            // material, not via a prop update flagged `needsUpdate`-free by
            // r3f's usual reconciliation.
            key={`${cls}-${painted}`}
            geometry={geometries.tumor[cls]}
            onClick={(e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation();
              onSelect(selected === cls ? null : cls);
            }}
          >
            <meshStandardMaterial
              vertexColors={painted}
              color={painted ? 0xffffff : rgbToThreeColor(CLASS_HEX[cls])}
              roughness={0.35}
              metalness={0.05}
              emissive={painted ? PAINTED_EMISSIVE : rgbToThreeColor(CLASS_HEX[cls])}
              emissiveIntensity={selected === cls ? 0.35 : 0.08}
            />
          </mesh>
        );
      })}
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

export interface BrainTwinSceneProps {
  input: BrainTwinInput | null;
  /** Short text rendered top-left over the canvas (e.g. "PROCEED WITH CAUTION"). Absent → nothing rendered. */
  badge?: string | null;
  /** Visual tone of the badge. */
  badgeTone?: "caution" | "neutral";
  /** The scalar layer currently painting the tumour meshes, or null/absent -> flat class colours (today's look). */
  activeLayer?: TwinActiveLayer | null;
  /** Atlas structure shells to draw. `volume` is the uint8 structure-index volume in the same (D,H,W) layout as input.tumorMask; `selection` the indices to mesh (already capped by selectStructures); `table` the structure metadata. */
  atlas?: TwinAtlasInput | null;
  /** Structure index to emphasise (hover/click from the report table), or null. Controlled by the parent. */
  highlightedStructure?: number | null;
  /** Fired when the user clicks a shell (index) or clicks empty space (null). */
  onStructureSelect?: (index: number | null) => void;
  /** Per-structure detail the parent computes from the report (e.g. overlap fractions). Absent → the card shows name/laterality/lobe/eloquence only. */
  structureDetail?: (index: number) => { fracOfStructure: number | null; fracOfTumour: number | null } | null;
}

export function BrainTwinScene({
  input,
  badge,
  badgeTone = "neutral",
  activeLayer = null,
  atlas = null,
  highlightedStructure = null,
  onStructureSelect,
  structureDetail,
}: BrainTwinSceneProps) {
  const { result, geometries, computing, activeLayerScalars, structureGeometries } = useTwinMesh(
    input,
    activeLayer,
    atlas,
  );
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

  // scalarsToColors throws on an unrecognised kind (by design - see
  // vertexColors.ts), so this only ever runs it against a real,
  // caller-provided kind; a class with no scalars for this layer (too small
  // to have been sampled, or not yet resampled) is simply absent from the
  // result and keeps its flat colour (see TwinModel's colorsByClass usage).
  const colorsByClass = useMemo(() => {
    if (!activeLayer || !activeLayerScalars) return null;
    const out: Partial<Record<ClassName, Float32Array>> = {};
    for (const cls of Object.keys(activeLayerScalars) as ClassName[]) {
      const scalars = activeLayerScalars[cls];
      if (scalars) out[cls] = scalarsToColors(scalars, activeLayer.kind);
    }
    return out;
  }, [activeLayer, activeLayerScalars]);

  // Geometric mean, not a claim about accuracy - see the detail card below.
  const selectedLayerMean =
    selected && activeLayer && activeLayerScalars?.[selected]
      ? activeLayerScalars[selected]!.reduce((sum, v) => sum + v, 0) / activeLayerScalars[selected]!.length
      : null;

  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [0, 0.3, 2.6], fov: 42 }}
        onPointerMissed={() => {
          setSelected(null);
          onStructureSelect?.(null);
        }}
        dpr={[1, 2]}
        // `preserveDrawingBuffer` keeps the rendered frame in the backbuffer
        // after compositing, instead of the default WebGL clear - required
        // for `canvas.toDataURL()` to return the actual picture rather than
        // black (both the E2E harness's pixel assertions and T6's export
        // snapshot read the canvas this way). Costs one extra buffer copy
        // per frame, which is acceptable at this scene's size.
        gl={{ preserveDrawingBuffer: true }}
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
                colorsByClass={colorsByClass}
                structureGeometries={structureGeometries}
                highlightedStructure={highlightedStructure}
                onStructureSelect={onStructureSelect}
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

      {badge && (
        <div
          data-testid="twin-badge"
          role="status"
          className={`liquid-glass pointer-events-none absolute top-3 left-3 rounded-md border px-2.5 py-1 font-condensed text-[11px] font-semibold tracking-[0.12em] uppercase ${
            badgeTone === "caution"
              ? "border-data-amber/60 text-data-amber"
              : "border-surface-seam text-text-secondary"
          }`}
        >
          {badge}
        </div>
      )}

      {!geometries && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xs text-text-dim">
            {computing ? "Reconstructing this case's geometry…" : "Pick a case to build its twin."}
          </span>
        </div>
      )}

      {geometries && (
        <div className="pointer-events-none absolute bottom-3 left-3 flex flex-col gap-1 font-mono text-[11px] text-text-dim">
          <span>
            Drag to orbit · scroll to zoom · click a hemisphere to separate
            {structureGeometries && structureGeometries.size > 0 ? " · click a structure" : ""}
          </span>
        </div>
      )}

      {/* Tumour detail (top) and structure detail (below it, when both are
          open at once) stack in one flex column rather than two independently
          top-positioned cards, so a tall tumour card never overlaps the
          structure card beneath it. */}
      {geometries && (selected || (highlightedStructure !== null && atlas)) && (
        <div className="absolute top-3 right-3 flex w-56 flex-col gap-2">
          {selected && result && (
            <div className="liquid-glass rounded-md border border-surface-seam px-3 py-3">
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
              {selectedLayerMean !== null && (
                <p className="mt-1 font-mono text-[10px] leading-relaxed text-text-dim">
                  mean {activeLayer!.kind} at surface−1 voxel: {selectedLayerMean.toFixed(2)}
                </p>
              )}
            </div>
          )}

          {highlightedStructure !== null &&
            atlas &&
            (() => {
              const row = structureRow(atlas.table, highlightedStructure);
              if (!row) return null; // stale index (a report/atlas version mismatch) - nothing to show
              const detail = structureDetail?.(highlightedStructure) ?? null;
              return (
                <div
                  data-testid="twin-structure-detail"
                  className="liquid-glass rounded-md border border-surface-seam px-3 py-3"
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                      style={{ backgroundColor: structureColorHex(highlightedStructure) }}
                      aria-hidden="true"
                    />
                    <span className="font-mono text-xs text-text-primary">{row.name}</span>
                    <button
                      type="button"
                      onClick={() => onStructureSelect?.(null)}
                      className="ml-auto text-text-dim hover:text-text-primary"
                      aria-label="Close structure detail"
                    >
                      ×
                    </button>
                  </div>
                  <p className="font-mono text-[11px] text-text-secondary">
                    {row.laterality ?? "—"} · {row.lobe ?? "—"}
                  </p>
                  {/* Raw table value, verbatim - never invented (e.g. never
                      substitute a friendlier word than what the atlas's own
                      eloquence field actually says). */}
                  <p className="mt-1 font-mono text-[11px] text-text-secondary">{row.eloquence ?? "—"}</p>
                  {detail && (detail.fracOfStructure !== null || detail.fracOfTumour !== null) && (
                    <p className="mt-1 font-mono text-[10px] leading-relaxed text-text-dim">
                      {detail.fracOfStructure !== null &&
                        `${(detail.fracOfStructure * 100).toFixed(0)}% of ${row.name} overlaps the mask`}
                      {detail.fracOfStructure !== null && detail.fracOfTumour !== null && " · "}
                      {detail.fracOfTumour !== null && `${(detail.fracOfTumour * 100).toFixed(0)}% of the tumour`}
                    </p>
                  )}
                </div>
              );
            })()}
        </div>
      )}

      {/* Bottom-right stack: the structure legend (when shells are drawn)
          above the layer legend (when a scalar layer is active) - a
          flex-col-reverse column anchored at its bottom edge, so either can
          be present alone without leaving a gap where the other would have
          been. */}
      {(activeLayer || (structureGeometries && structureGeometries.size > 0)) && (
        <div className="absolute right-3 bottom-3 flex w-48 flex-col-reverse gap-2">
          {/* Label the layer ONLY by the header value passed in via
              activeLayer.kind - never an invented string - so the twin can
              never show a name for a quantity the backend did not actually
              send. */}
          {activeLayer && (
            <div data-testid="twin-layer-legend" className="liquid-glass rounded-md border border-surface-seam">
              <Legend overlayMode="prediction" showUncertainty hasLabel={false} uncertaintyKind={activeLayer.kind} />
            </div>
          )}

          {structureGeometries && structureGeometries.size > 0 && atlas && (
            <div
              data-testid="twin-structure-legend"
              className="liquid-glass max-h-48 overflow-y-auto rounded-md border border-surface-seam p-2"
            >
              <div className="eyebrow px-1 pb-1">Structures</div>
              <div className="flex flex-col gap-1">
                {/* atlas.selection's own order (report order), not the Map's
                    - Object.entries on the worker's numeric-keyed structures
                    object always comes back in ascending index order in JS,
                    which would silently re-sort this list away from the
                    report's involvement ranking. */}
                {atlas.selection
                  .filter((index) => structureGeometries.has(index))
                  .map((index) => {
                    const row = structureRow(atlas.table, index);
                    return (
                      <button
                        key={index}
                        type="button"
                        onClick={() => onStructureSelect?.(index)}
                        className="flex items-center gap-2 text-left"
                      >
                        <span
                          className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                          style={{ backgroundColor: structureColorHex(index) }}
                          aria-hidden="true"
                        />
                        <span className="truncate font-mono text-[11px] text-text-secondary">
                          {row?.name ?? `#${index}`}
                        </span>
                      </button>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
