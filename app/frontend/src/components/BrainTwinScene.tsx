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
import type { AtlasStructureRow, ConformalMeta } from "../api";
import { hexToRgb } from "../lib/colors";
import { scalarsToColors, type VertexLayerKind } from "../lib/vertexColors";
import { structureRow } from "../lib/atlasSelection";
import { Opaque, opaque } from "../lib/opaque";
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
  // Opaque-wrapped (lib/opaque.ts): React dev enumerates every index of a raw
  // typed array in a changed prop, which cost seconds per render.
  modalityVolumes: Opaque<Uint8Array[]>;
  tumorMask: Opaque<Uint8Array> | null;
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
  data: Opaque<Uint8Array>;
  /** Conformal operating point, only used to label the legend. */
  conformal?: ConformalMeta | null;
}

interface TwinGeometries {
  brainLeft: THREE.BufferGeometry;
  brainRight: THREE.BufferGeometry;
  tumor: Partial<Record<ClassName, THREE.BufferGeometry>>;
}

/** Atlas structure shells to draw - see BrainTwinSceneProps.atlas for the field meanings. */
export interface TwinAtlasInput {
  volume: Opaque<Uint8Array>;
  selection: number[];
  table: AtlasStructureRow[];
}

// Module-level LRU of worker mesh results, so remounting the scene (view
// switch) or revisiting a recent case does not re-mesh. The buffers in a
// result were transferred to the main thread by the worker, so the main
// thread owns them; they are never posted back to the worker with a
// transfer list (the "sample" request structured-clones voxelGeometry).
const MESH_LRU_SIZE = 5;
const meshLru = new Map<string, TwinMeshResult>();

/** Everything that defines the meshed input: case/job id, which mask fed the tumour surfaces, and the grid shape. */
function meshKey(input: BrainTwinInput): string {
  return `${input.caseId}|${input.tumorSource ?? "none"}|${input.shape.join("x")}`;
}
function lruGet(key: string): TwinMeshResult | undefined {
  const hit = meshLru.get(key);
  if (hit) {
    meshLru.delete(key);
    meshLru.set(key, hit); // refresh recency
  }
  return hit;
}
function lruSet(key: string, value: TwinMeshResult): void {
  meshLru.delete(key);
  meshLru.set(key, value);
  while (meshLru.size > MESH_LRU_SIZE) {
    const oldest = meshLru.keys().next().value;
    if (oldest === undefined) break;
    meshLru.delete(oldest);
  }
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
  // Base mesh geometry lives in the module-level meshLru above - independent
  // of which layer is active, so switching layers never touches it.
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
    { type: "mesh" | "sample" | "structures"; caseId: string; key?: string; meshKey?: string } | null
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
        if (perClass && pending?.type === "sample" && pending.key && pending.meshKey) {
          let caseLayers = layerCacheRef.current.get(pending.meshKey);
          if (!caseLayers) {
            caseLayers = new Map();
            layerCacheRef.current.set(pending.meshKey, caseLayers);
          }
          caseLayers.set(pending.key, perClass);
          setLayerVersion((v) => v + 1);
        }
        setComputing(false);
        return;
      }

      if ("kind" in data && data.kind === "structures") {
        if (pending?.type === "structures" && pending.key && pending.meshKey) {
          let caseStructures = structuresCacheRef.current.get(pending.meshKey);
          if (!caseStructures) {
            caseStructures = new Map();
            structuresCacheRef.current.set(pending.meshKey, caseStructures);
          }
          caseStructures.set(pending.key, data.structures);
          setStructuresVersion((v) => v + 1);
        }
        setStructuresComputing(false);
        return;
      }

      if (pending?.type === "mesh" && pending.meshKey) lruSet(pending.meshKey, data);
      // The mesh request may itself have carried the active layer (see the
      // request-building effect below) - fold its result into the same
      // per-key cache a later "sample" response would use, so the two paths
      // are indistinguishable to activeLayerScalars.
      if (data.layerScalars && pending?.type === "mesh" && pending.key && pending.meshKey) {
        const perClass = Object.values(data.layerScalars)[0];
        if (perClass) {
          let caseLayers = layerCacheRef.current.get(pending.meshKey);
          if (!caseLayers) {
            caseLayers = new Map();
            layerCacheRef.current.set(pending.meshKey, caseLayers);
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
    const key = meshKey(input);
    const cached = lruGet(key);
    if (cached) {
      setResult(cached);
      setComputing(false);
      return;
    }
    requestIdRef.current += 1;
    setComputing(true);
    const layer = activeLayerRef.current;
    pendingRef.current = { type: "mesh", caseId: input.caseId, key: layer?.key, meshKey: key };
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
      modalityVolumes: input.modalityVolumes.value,
      tumorMask: input.tumorMask?.value ?? null,
      tumorSource: input.tumorSource,
      scalarLayers: layer ? { [layer.kind]: layer.data.value } : undefined,
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
  }, [input?.caseId, input?.tumorSource]);

  // Layer request: fires whenever the active layer changes (or a cached
  // case turns out not to have this layer's scalars yet) - never re-meshes,
  // only resamples the case's already-computed voxel geometry.
  useEffect(() => {
    if (!input || !activeLayer) return;
    const cachedMesh = lruGet(meshKey(input));
    if (!cachedMesh || !cachedMesh.voxelGeometry) return; // mesh not ready yet; the mesh effect above will carry this layer once it lands
    const caseLayers = layerCacheRef.current.get(meshKey(input));
    if (caseLayers?.has(activeLayer.key)) return; // already sampled

    requestIdRef.current += 1;
    pendingRef.current = { type: "sample", caseId: input.caseId, key: activeLayer.key, meshKey: meshKey(input) };
    const request: TwinSampleRequest = {
      kind: "sample",
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      layers: { [activeLayer.kind]: activeLayer.data.value },
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
    const cachedMesh = lruGet(meshKey(input));
    if (!cachedMesh) return; // mesh (and its frame) not ready yet
    const key = atlas.selection.join(",");
    const caseStructures = structuresCacheRef.current.get(meshKey(input));
    if (caseStructures?.has(key)) return; // already meshed

    requestIdRef.current += 1;
    setStructuresComputing(true);
    pendingRef.current = { type: "structures", caseId: input.caseId, key, meshKey: meshKey(input) };
    const request: TwinStructuresRequest = {
      kind: "structures",
      requestId: requestIdRef.current,
      caseId: input.caseId,
      shape: input.shape,
      // Deliberately NOT transferred (see TwinStructuresRequest's docstring)
      // - the caller (the report/atlas panel a later unit wires up) keeps
      // its own copy of this volume for the next selection change.
      atlas: atlas.volume.value,
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

  // Free the GPU buffers of a replaced / unmounted geometry set. Keyed on the
  // geometries object itself, so the cleanup only runs for the set that is
  // being replaced, never for the one the next render uses.
  useEffect(() => {
    if (!geometries) return;
    return () => {
      geometries.brainLeft.dispose();
      geometries.brainRight.dispose();
      for (const g of Object.values(geometries.tumor)) g?.dispose();
    };
  }, [geometries]);

  const activeLayerScalars = useMemo(() => {
    if (!input || !activeLayer) return null;
    return layerCacheRef.current.get(meshKey(input))?.get(activeLayer.key) ?? null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input?.caseId, input?.tumorSource, activeLayer, layerVersion, result]);

  const structureGeometries = useMemo(() => {
    if (!input || !atlas || atlas.selection.length === 0) return null;
    const key = atlas.selection.join(",");
    const structures = structuresCacheRef.current.get(meshKey(input))?.get(key);
    if (!structures) return null;
    const out = new Map<number, THREE.BufferGeometry>();
    for (const [indexStr, buf] of Object.entries(structures)) {
      out.set(Number(indexStr), toGeometry(buf));
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [input?.caseId, input?.tumorSource, atlas, structuresVersion]);

  useEffect(() => {
    if (!structureGeometries) return;
    return () => {
      for (const g of structureGeometries.values()) g.dispose();
    };
  }, [structureGeometries]);

  return {
    result,
    geometries,
    computing,
    activeLayerScalars,
    structureGeometries,
    structuresComputing,
  };
}

// geometries / colorsByClass / structureGeometries are Opaque-wrapped: a
// BufferGeometry's own enumerable keys reach its typed arrays, which React's
// dev prop-diffing would walk (see lib/opaque.ts).
interface TwinModelProps {
  geometries: Opaque<NonNullable<ReturnType<typeof useTwinMesh>["geometries"]>>;
  separated: boolean;
  onToggleSeparate: () => void;
  selected: ClassName | null;
  onSelect: (c: ClassName | null) => void;
  /** Per-class vertex RGB from scalarsToColors, or absent/no-entry -> that class keeps its flat class colour. */
  colorsByClass: Opaque<Partial<Record<ClassName, Float32Array>>> | null;
  /** Meshed atlas structure shells to render, keyed by atlas index, or null when none are selected/ready. */
  structureGeometries: Opaque<Map<number, THREE.BufferGeometry>> | null;
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
  geometries: geometriesHandle,
  separated,
  onToggleSeparate,
  selected,
  onSelect,
  colorsByClass: colorsHandle,
  structureGeometries: structureHandle,
  highlightedStructure = null,
  onStructureSelect,
}: TwinModelProps) {
  const geometries = geometriesHandle.value;
  const colorsByClass = colorsHandle?.value ?? null;
  const structureGeometries = structureHandle?.value ?? null;
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

  // Two-pass transparency: rendering the shell with DoubleSide in a single
  // pass causes near-parallel triangles to overlap additively at glancing
  // angles, producing bright streaks (the white ray artifacts). Splitting
  // into a BackSide pass (drawn first, the far wall) and a FrontSide pass
  // (drawn second, the near wall) eliminates self-overlap because each pass
  // only draws one face per triangle. polygonOffset on the back pass pushes
  // it slightly behind, preventing z-fighting at shallow angles.
  const shellBackMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xe7eaee,
        transparent: true,
        opacity: 0.12,
        roughness: 0.6,
        metalness: 0,
        side: THREE.BackSide,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      }),
    [],
  );
  const shellFrontMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: 0xe7eaee,
        transparent: true,
        opacity: 0.16,
        roughness: 0.6,
        metalness: 0,
        side: THREE.FrontSide,
        depthWrite: false,
      }),
    [],
  );

  // Free the shell materials' GPU programs when replaced / on unmount.
  useEffect(() => () => shellBackMaterial.dispose(), [shellBackMaterial]);
  useEffect(() => () => shellFrontMaterial.dispose(), [shellFrontMaterial]);

  // Clipping plane at the mid-sagittal plane (scene X = 0): when the brain
  // is separated, this clips the tumour meshes so the cross-section of all
  // three classes is visible through the gap. The plane normal points +X,
  // removing the +X half of each tumour mesh, which exposes the interior
  // from the left-hemisphere gap (the camera's natural viewing angle when
  // orbiting a separated brain). When not separated, no clipping is applied.
  const tumorClipPlanes = useMemo(
    () => [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0)],
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
        {/* Back-side pass first (far wall), then front-side (near wall). */}
        <mesh
          geometry={geometries.brainLeft}
          material={shellBackMaterial}
          renderOrder={0}
        />
        <mesh
          geometry={geometries.brainLeft}
          material={shellFrontMaterial}
          renderOrder={1}
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
          material={shellBackMaterial}
          renderOrder={0}
        />
        <mesh
          geometry={geometries.brainRight}
          material={shellFrontMaterial}
          renderOrder={1}
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
            // `painted` and `separated` in the key force React to remount
            // the material when painting or clipping toggles, rather than
            // mutating an existing THREE.MeshStandardMaterial in place -
            // `vertexColors` changes the compiled shader (USE_COLOR) and
            // `clippingPlanes` is not reactive, so both need a fresh
            // material via remount.
            key={`${cls}-${painted}-${separated}`}
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
              side={separated ? THREE.DoubleSide : THREE.FrontSide}
              clippingPlanes={separated ? tumorClipPlanes : []}
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

  // In a short host (the clinical 2x2 tile) the Structures list starts
  // collapsed so it cannot crowd out the layer legend.
  const hostRef = useRef<HTMLDivElement>(null);
  const [compact, setCompact] = useState(false);
  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const h = entries[0]?.contentRect.height ?? el.clientHeight;
      setCompact(h < 420);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

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
    <div ref={hostRef} className="relative h-full w-full">
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
        gl={{ preserveDrawingBuffer: true, localClippingEnabled: true }}
      >
        <ambientLight intensity={0.55} />
        <directionalLight position={[2, 3, 4]} intensity={1.1} />
        <directionalLight position={[-2, -1, -3]} intensity={0.3} />
        <Suspense fallback={null}>
          {geometries && (
            <>
              <TwinModel
                geometries={opaque(geometries)}
                separated={separated}
                onToggleSeparate={() => setSeparated((v) => !v)}
                selected={selected}
                onSelect={setSelected}
                colorsByClass={opaque(colorsByClass)}
                structureGeometries={opaque(structureGeometries)}
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
          className={`chip pointer-events-none absolute top-3 left-3 ${
            badgeTone === "caution"
              ? "border-data-amber/60 text-data-amber"
              : ""
          }`}
        >
          {badge}
        </div>
      )}

      {!geometries && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs text-text-dim">
            {computing ? "Reconstructing this case's geometry…" : "Pick a case to build its twin."}
          </span>
        </div>
      )}

      {geometries && (
        <div className="pointer-events-none absolute bottom-3 left-3 max-w-[calc(100%-18rem)] truncate text-xs text-text-dim">
          Drag to orbit · scroll to zoom · click a hemisphere to separate
          {structureGeometries && structureGeometries.size > 0 ? " · click a structure" : ""}
        </div>
      )}

      {/* Tumour detail (top) and structure detail (below it, when both are
          open at once) stack in one flex column rather than two independently
          top-positioned cards, so a tall tumour card never overlaps the
          structure card beneath it. */}
      {geometries && (selected || (highlightedStructure !== null && atlas)) && (
        <div className="absolute top-3 right-3 flex w-56 flex-col gap-2">
          {selected && result && (
            <div className="glass-panel px-3 py-3">
              <div className="mb-1 flex items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                  style={{ backgroundColor: CLASS_HEX[selected] }}
                  aria-hidden="true"
                />
                <span className="text-xs text-text-primary">{CLASS_LABEL[selected]}</span>
                <button
                  type="button"
                  onClick={() => setSelected(null)}
                  className="ml-auto rounded-md px-1 text-text-dim transition-colors hover:text-text-primary"
                  aria-label="Close tumour detail"
                >
                  ×
                </button>
              </div>
              <p className="tabular text-xs text-text-secondary">
                <span className="font-mono tabular-nums">{(result.classVolumesMl[selected] ?? 0).toFixed(2)}</span> ml, this case
              </p>
              <p className="mt-1 text-xs leading-relaxed text-text-dim">
                {result.tumorSource === "label" ? "Real ground-truth label" : "Real saved model prediction"},{" "}
                {result.caseId}.
              </p>
              {selectedLayerMean !== null && (
                <p className="mt-1 text-xs leading-relaxed text-text-dim">
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
                  className="glass-panel px-3 py-3"
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
                      className="ml-auto rounded-md px-1 text-text-dim transition-colors hover:text-text-primary"
                      aria-label="Close structure detail"
                    >
                      ×
                    </button>
                  </div>
                  <p className="text-xs text-text-secondary">
                    {row.laterality ?? "—"} · {row.lobe ?? "—"}
                  </p>
                  {/* Raw table value, verbatim - never invented (e.g. never
                      substitute a friendlier word than what the atlas's own
                      eloquence field actually says). */}
                  <p className="mt-1 text-xs text-text-secondary">{row.eloquence ?? "—"}</p>
                  {detail && (detail.fracOfStructure !== null || detail.fracOfTumour !== null) && (
                    <p className="mt-1 text-xs leading-relaxed text-text-dim">
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

      {/* Bottom-right column holding BOTH overlays (structures above, layer
          legend below) so they can never overlap each other. */}
      {(activeLayer || (structureGeometries && structureGeometries.size > 0)) && (
        <div className="absolute right-3 bottom-3 flex max-h-[calc(100%-1.5rem)] flex-col items-end gap-2 overflow-hidden">
          {structureGeometries && structureGeometries.size > 0 && atlas && (
            <div
              data-testid="twin-structure-legend"
              className="glass-panel min-h-0 w-full max-w-[16rem] p-2"
            >
              <details open={!compact}>
                <summary className="eyebrow cursor-pointer px-1 pb-1">
                  Structures ({atlas.selection.filter((i) => structureGeometries.has(i)).length})
                </summary>
                <div className="flex max-h-48 flex-col gap-1 overflow-auto">
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
                          className="flex items-center gap-2 rounded-sm text-left transition-colors hover:bg-surface-raised/50"
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
              </details>
            </div>
          )}

          {/* Label the layer ONLY by the header value passed in via
              activeLayer.kind - never an invented string - so the twin can
              never show a name for a quantity the backend did not actually
              send. */}
          {activeLayer && (
            <div data-testid="twin-layer-legend" className="w-full max-w-[16rem] shrink-0 rounded-[10px]">
              <Legend overlayMode="prediction" showUncertainty hasLabel={false} uncertaintyKind={activeLayer.kind} conformal={activeLayer.conformal ?? null} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
