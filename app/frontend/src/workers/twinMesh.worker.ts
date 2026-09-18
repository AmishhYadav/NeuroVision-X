// Computes the digital twin's meshes off the main thread: a brain-shell
// isosurface (split at the mid-sagittal plane) plus one mesh per tumour
// class, all from the SAME case data the rest of the tool already fetches
// (see useCaseData) - never from a fixed, hardcoded case. A ~3-4M-voxel
// volume takes real time to surface; doing it here keeps the tab responsive
// while it runs.
import { extentCenter, meshVoxelField, normalToScene, voxelToScene } from "../lib/twinGeometry";
import { sampleLayersForClasses } from "../lib/twinLayers";
import { meshStructure, structureBBoxes } from "../lib/structureMesh";
import { MAX_ATLAS_STRUCTURES } from "../lib/atlasSelection";
import type { VertexLayerKind } from "../lib/vertexColors";

export type ClassName = "necrotic" | "oedema" | "enhancing";

export interface TwinMeshRequest {
  requestId: number;
  caseId: string;
  shape: [number, number, number]; // (D, H, W)
  spacing: [number, number, number];
  modalityVolumes: Uint8Array[]; // whichever modalities loaded, OR'd for the brain mask
  tumorMask: Uint8Array | null; // label (preferred) or prediction, {0,1,2,3}
  tumorSource: "label" | "prediction" | null;
  /** Optional per-voxel uint8 scalar volumes in the same (D,H,W) layout as tumorMask, keyed by the backend's X-Uncertainty-Kind value. */
  scalarLayers?: Partial<Record<VertexLayerKind, Uint8Array>>;
  /** Keep each surfaced tumour class's (d,h,w) voxel positions/normals in the result even with no scalarLayers, so a later "sample" message can add a layer without re-meshing. Implied by a non-empty scalarLayers. */
  keepVoxelGeometry?: boolean;
}

export interface TwinMeshResult {
  kind?: "mesh";
  requestId: number;
  caseId: string;
  brainLeft: { position: Float32Array; normal: Float32Array; index: Uint32Array };
  brainRight: { position: Float32Array; normal: Float32Array; index: Uint32Array };
  tumor: Partial<Record<ClassName, { position: Float32Array; normal: Float32Array; index: Uint32Array }>>;
  tumorSource: "label" | "prediction" | null;
  tumorCentroidScene: [number, number, number] | null;
  classVolumesMl: Partial<Record<ClassName, number>>;
  /** Per tumour class, vertex positions in (d,h,w) voxel space and outward normals (d,h,w) - kept so a NEW layer can be sampled later without re-meshing. Present only when the request carried scalarLayers or `keepVoxelGeometry: true`. */
  voxelGeometry?: Partial<Record<ClassName, { positions: Float32Array; normals: Float32Array }>>;
  /** layerScalars[kind][class] = Float32Array (N) in [0,1], sampled one voxel inward. Present only when the request carried scalarLayers. */
  layerScalars?: Partial<Record<VertexLayerKind, Partial<Record<ClassName, Float32Array>>>>;
  /**
   * The brain mesh's own extentCenter() output (center + scale), i.e. the
   * exact voxelToScene() arguments used for the brain/tumour meshes above.
   * ALWAYS present (additive field - the existing demo path in App.tsx just
   * ignores it). A later "structures" request needs this to place atlas
   * structure shells in the SAME scene frame without re-deriving it (and
   * without re-meshing the brain to get it).
   */
  frame: { center: [number, number, number]; scale: number };
}

// A second message the worker understands: resample an ALREADY-MESHED case
// with a new (or additional) scalar layer, using the voxel geometry a prior
// mesh request returned (see TwinMeshResult.voxelGeometry) - avoids paying
// for surface-nets again just to switch which uncertainty layer is shown.
// Discriminated from TwinMeshRequest by the `kind` field: a mesh request has
// none, so `"kind" in e.data` is false for it.
export interface TwinSampleRequest {
  kind: "sample";
  requestId: number;
  caseId: string;
  shape: [number, number, number]; // (D, H, W)
  layers: Partial<Record<VertexLayerKind, Uint8Array>>;
  voxelGeometry: Partial<Record<ClassName, { positions: Float32Array; normals: Float32Array }>>;
}

export interface TwinSampleResult {
  kind: "sample";
  requestId: number;
  caseId: string;
  layerScalars: Partial<Record<VertexLayerKind, Partial<Record<ClassName, Float32Array>>>>;
}

// A third message the worker understands: mesh a SELECTION of atlas
// structures (T3.4) - one Surface-Nets shell per selected index, in the
// SAME scene frame as an earlier mesh request's brain/tumour meshes (see
// TwinMeshResult.frame), without re-meshing the brain or tumour. Like
// TwinSampleRequest, discriminated from a plain TwinMeshRequest by `kind`.
export interface TwinStructuresRequest {
  kind: "structures";
  requestId: number;
  caseId: string;
  shape: [number, number, number]; // (D, H, W)
  /**
   * uint8 (D,H,W) atlas structure-index volume, 0 = background, value k =
   * AtlasStructureRow.index. The CALLER keeps its own copy of this volume
   * (a later selection change needs it again) - it must be sent with
   * `.slice()` or structured-clone semantics, never `postMessage(..., [atlas.buffer])`,
   * which would detach the caller's only copy.
   */
  atlas: Uint8Array;
  /** Atlas indices to mesh, in no particular order - capped at MAX_ATLAS_STRUCTURES here regardless of what the caller sends. */
  selection: number[];
  /** The scene frame from a prior TwinMeshResult, so shells line up with the brain/tumour meshes already on screen. */
  frame: TwinMeshResult["frame"];
}

export interface TwinStructuresResult {
  kind: "structures";
  requestId: number;
  caseId: string;
  /** Keyed by atlas structure index. An index absent here was either not found in `atlas`, or had fewer than 20 voxels (see meshStructure) - never thrown as an error. */
  structures: Record<number, { position: Float32Array; normal: Float32Array; index: Uint32Array }>;
}

const CLASS_NAMES: Record<number, ClassName> = { 1: "necrotic", 2: "oedema", 3: "enhancing" };

// Axis convention (see src/lib/twinGeometry.ts for the derivation and the
// tests that pin it down): voxel (d,h,w) -> scene X = d (R->L, +X patient
// LEFT), Y = w (+Y SUPERIOR), Z = -h (+Z ANTERIOR) - a right-handed
// (Left, Superior, Anterior) frame, matching the radiological convention
// slicing.ts uses for the 2D views. This REPLACES a previous mapping that
// was wrong two ways at once: it called surfaceNets with dims [D,H,W] on a
// w-fastest buffer, which reinterprets the strides (wrong whenever
// D !== W, i.e. always, for a real case); and its scene mapping put
// +Z = posterior, which is left-handed and mirrors the whole brain left
// for right - a mirrored brain still looks like a brain, so neither bug
// was visible without a numeric probe.
self.onmessage = (e: MessageEvent<TwinMeshRequest | TwinSampleRequest | TwinStructuresRequest>) => {
  const data = e.data;
  // A mesh request carries no `kind` field at all (see TwinSampleRequest's
  // docstring) - this keeps the existing App.tsx demo path, which only ever
  // sends TwinMeshRequest, completely untouched. TwinMeshRequest does not
  // declare `kind`, so the cast below is only needed because TS cannot
  // prove the negative case from a plain `in` check on a non-discriminated
  // field; the runtime check itself is exact.
  if ("kind" in data && data.kind === "sample") {
    handleSampleRequest(data);
    return;
  }
  if ("kind" in data && data.kind === "structures") {
    handleStructuresRequest(data);
    return;
  }
  handleMeshRequest(data as TwinMeshRequest);
};

function handleMeshRequest(request: TwinMeshRequest) {
  const { requestId, caseId, shape, spacing, modalityVolumes, tumorMask, tumorSource, scalarLayers, keepVoxelGeometry } =
    request;
  const [D, H, W] = shape;
  const n = D * H * W;

  // Perf instrumentation: this project verifies a performance change by
  // re-running the real workload and checking the number moved, not by
  // unit tests (there are none for this worker) - this is that number.
  const tStart = performance.now();

  const brainField = new Float32Array(n);
  for (const vol of modalityVolumes) {
    for (let i = 0; i < n; i++) {
      if (vol[i] !== 0) brainField[i] = 1;
    }
  }

  const brainRaw = meshVoxelField(brainField, [D, H, W], 0.5);
  const { center, scale } = extentCenter(brainRaw.positions);
  const brainScenePos = voxelToScene(brainRaw.positions, center, scale);
  const brainSceneNorm = normalToScene(brainRaw.normals);

  // Split at the mid-sagittal plane (scene X = 0) into two hemisphere
  // sub-meshes, same rationale as the landing page's original build: an
  // interaction the geometry itself grounds, not an arbitrary animation.
  const splitMesh = (
    positions: Float32Array,
    normals: Float32Array,
    indices: Uint32Array,
    keepLeft: boolean,
  ) => {
    const nTris = indices.length / 3;
    const nVerts = positions.length / 3;
    // Vertex remap: -1 = not kept (yet), else the vertex's NEW index,
    // assigned in first-seen order while walking kept triangles in
    // triangle order - same order the old Map<number, number> produced,
    // but as one fixed-size typed array instead of a Map plus two
    // number[] arrays growing via push().
    const remap = new Int32Array(nVerts).fill(-1);
    let vertCount = 0;
    let triCount = 0;
    for (let t = 0; t < nTris; t++) {
      const a = indices[t * 3];
      const b = indices[t * 3 + 1];
      const c = indices[t * 3 + 2];
      const meanX = (positions[a * 3] + positions[b * 3] + positions[c * 3]) / 3;
      if ((meanX < 0) !== keepLeft) continue;
      triCount++;
      if (remap[a] === -1) remap[a] = vertCount++;
      if (remap[b] === -1) remap[b] = vertCount++;
      if (remap[c] === -1) remap[c] = vertCount++;
    }

    // Exact-sized output arrays now that vertCount/triCount are known -
    // no push(), no growth, no re-copy.
    const outPos = new Float32Array(vertCount * 3);
    const outNorm = new Float32Array(vertCount * 3);
    const outIdx = new Uint32Array(triCount * 3);

    // Place each kept vertex's data at its assigned slot. Iteration order
    // here does not affect correctness (or the "first-seen order"
    // guarantee) because the WRITE target is remap[v], not the loop
    // position - remap[v] already encodes the first-seen order from the
    // pass above.
    for (let v = 0; v < nVerts; v++) {
      const mapped = remap[v];
      if (mapped === -1) continue;
      outPos[mapped * 3] = positions[v * 3];
      outPos[mapped * 3 + 1] = positions[v * 3 + 1];
      outPos[mapped * 3 + 2] = positions[v * 3 + 2];
      outNorm[mapped * 3] = normals[v * 3];
      outNorm[mapped * 3 + 1] = normals[v * 3 + 1];
      outNorm[mapped * 3 + 2] = normals[v * 3 + 2];
    }

    // Second walk over triangles, in the SAME order as the counting pass
    // above, so triangle order (and therefore winding) is unchanged.
    let triOut = 0;
    for (let t = 0; t < nTris; t++) {
      const a = indices[t * 3];
      const b = indices[t * 3 + 1];
      const c = indices[t * 3 + 2];
      const meanX = (positions[a * 3] + positions[b * 3] + positions[c * 3]) / 3;
      if ((meanX < 0) !== keepLeft) continue;
      outIdx[triOut * 3] = remap[a];
      outIdx[triOut * 3 + 1] = remap[b];
      outIdx[triOut * 3 + 2] = remap[c];
      triOut++;
    }

    return { position: outPos, normal: outNorm, index: outIdx };
  };

  const brainLeft = splitMesh(brainScenePos, brainSceneNorm, brainRaw.indices, true);
  const brainRight = splitMesh(brainScenePos, brainSceneNorm, brainRaw.indices, false);
  const tBrainDone = performance.now();

  const tumor: TwinMeshResult["tumor"] = {};
  const classVolumesMl: TwinMeshResult["classVolumesMl"] = {};
  const voxelMl = (spacing[0] * spacing[1] * spacing[2]) / 1000;
  let tumorCentroidScene: [number, number, number] | null = null;
  // A non-empty scalarLayers always needs the raw voxel geometry to sample
  // it against; keepVoxelGeometry lets a caller ask for it up front too, so
  // a LATER layer-only "sample" message never has to re-mesh.
  const keepGeometry = Boolean(keepVoxelGeometry) || Boolean(scalarLayers && Object.keys(scalarLayers).length > 0);
  const voxelGeometry: NonNullable<TwinMeshResult["voxelGeometry"]> = {};

  if (tumorMask) {
    let sumD = 0;
    let sumH = 0;
    let sumW = 0;
    let count = 0;
    for (let d = 0; d < D; d++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          const cls = tumorMask[d * H * W + h * W + w];
          if (cls > 0) {
            sumD += d;
            sumH += h;
            sumW += w;
            count++;
          }
        }
      }
    }
    if (count > 0) {
      const centroidVoxel = new Float32Array([sumD / count, sumH / count, sumW / count]);
      const centroidScene = voxelToScene(centroidVoxel, center, scale);
      tumorCentroidScene = [centroidScene[0], centroidScene[1], centroidScene[2]];
    }

    // Mesh each tumour class inside its own bounding box instead of the
    // full (D,H,W) volume: a tumour class occupies a few percent of the
    // volume's voxels, so one full-volume structureBBoxes pass (done ONCE
    // for all three classes) plus meshStructure's per-class crop costs a
    // fraction of three separate full-volume meshVoxelField calls.
    // meshStructure is the exact same crop-then-mesh routine the atlas
    // structure shells already use (see structureMesh.ts) - same field
    // construction (`=== index ? 1 : 0`), same isovalue (0.5), same
    // margin (1), just cropped to a bounding box first.
    const bboxes = structureBBoxes(tumorMask, shape, [1, 2, 3]);
    for (const [clsStr, name] of Object.entries(CLASS_NAMES)) {
      const cls = Number(clsStr);
      const bbox = bboxes.get(cls);
      const voxelCount = bbox?.count ?? 0;
      classVolumesMl[name] = Math.round(voxelCount * voxelMl * 100) / 100;
      if (!bbox || voxelCount < 20) continue; // too small to surface meaningfully
      const raw = meshStructure(tumorMask, shape, cls, bbox);
      if (!raw || raw.positions.length === 0) continue;
      const scenePos = voxelToScene(raw.positions, center, scale);
      const sceneNorm = normalToScene(raw.normals);
      tumor[name] = { position: scenePos, normal: sceneNorm, index: raw.indices };
      // Keep the PRE-voxelToScene positions/normals - scalar layers are
      // sampled in voxel space (see twinLayers.ts), not scene space.
      if (keepGeometry) {
        voxelGeometry[name] = { positions: raw.positions, normals: raw.normals };
      }
    }
  }
  const tTumorDone = performance.now();

  const layerScalars = scalarLayers ? sampleLayersForClasses(scalarLayers, shape, voxelGeometry) : undefined;

  const result: TwinMeshResult = {
    requestId,
    caseId,
    brainLeft,
    brainRight,
    tumor,
    tumorSource,
    tumorCentroidScene,
    classVolumesMl,
    frame: { center, scale },
    ...(keepGeometry ? { voxelGeometry } : {}),
    ...(layerScalars ? { layerScalars } : {}),
  };

  const transferables: Transferable[] = [
    brainLeft.position.buffer,
    brainLeft.normal.buffer,
    brainLeft.index.buffer,
    brainRight.position.buffer,
    brainRight.normal.buffer,
    brainRight.index.buffer,
  ];
  for (const m of Object.values(tumor)) {
    if (m) transferables.push(m.position.buffer, m.normal.buffer, m.index.buffer);
  }
  if (keepGeometry) {
    for (const g of Object.values(voxelGeometry)) {
      if (g) transferables.push(g.positions.buffer, g.normals.buffer);
    }
  }
  if (layerScalars) {
    for (const perClass of Object.values(layerScalars)) {
      if (!perClass) continue;
      for (const arr of Object.values(perClass)) {
        if (arr) transferables.push(arr.buffer);
      }
    }
  }

  // One line, logged just before the buffers below get transferred away
  // (and so still readable here) - see the tStart comment above for why
  // this exists: the author's rule is that a perf change is verified by
  // the real run moving this number, not by a passing test suite.
  const totalMs = performance.now() - tStart;
  const brainVerts = brainLeft.position.length / 3 + brainRight.position.length / 3;
  let tumourVerts = 0;
  for (const m of Object.values(tumor)) {
    if (m) tumourVerts += m.position.length / 3;
  }
  console.debug(
    `[twin] mesh ${caseId}: brain ${Math.round(tBrainDone - tStart)} ms, tumour ${Math.round(tTumorDone - tBrainDone)} ms, total ${Math.round(totalMs)} ms, verts brain=${brainVerts} tumour=${tumourVerts}`,
  );

  (self as unknown as Worker).postMessage(result, transferables);
}

// Resamples ALREADY-MESHED voxel geometry (from a prior TwinMeshResult's
// voxelGeometry) against one or more new/changed scalar layers, with no
// surface-nets work at all - the whole point of this message type.
function handleSampleRequest(request: TwinSampleRequest) {
  const { requestId, caseId, shape, layers, voxelGeometry } = request;
  const layerScalars = sampleLayersForClasses(layers, shape, voxelGeometry);

  const result: TwinSampleResult = { kind: "sample", requestId, caseId, layerScalars };

  const transferables: Transferable[] = [];
  for (const perClass of Object.values(layerScalars)) {
    if (!perClass) continue;
    for (const arr of Object.values(perClass)) {
      if (arr) transferables.push(arr.buffer);
    }
  }
  (self as unknown as Worker).postMessage(result, transferables);
}

// Meshes a SELECTION of atlas structures against an already-known scene
// frame (see TwinStructuresRequest.frame) - no brain/tumour re-mesh at all.
// structureBBoxes does the one full-volume pass this needs; meshStructure
// then crops per structure (see structureMesh.ts for why cropping matters).
function handleStructuresRequest(request: TwinStructuresRequest) {
  const { requestId, caseId, shape, atlas, selection, frame } = request;

  // Defensive cap: selectStructures() (atlasSelection.ts) already caps its
  // own output at MAX_ATLAS_STRUCTURES, but the worker cannot assume every
  // caller goes through that function.
  const capped = selection.slice(0, MAX_ATLAS_STRUCTURES);
  const bboxes = structureBBoxes(atlas, shape, capped);

  const structures: TwinStructuresResult["structures"] = {};
  const transferables: Transferable[] = [];

  for (const structureIndex of capped) {
    const bbox = bboxes.get(structureIndex);
    if (!bbox) continue; // never occurs in this atlas volume (e.g. stale report reference)
    const raw = meshStructure(atlas, shape, structureIndex, bbox);
    if (!raw) continue; // <20 voxels, or an empty mesh

    // Same voxel->scene transform as the brain/tumour meshes, using the
    // CALLER-SUPPLIED frame rather than re-deriving one from this
    // structure's own extent - that is exactly what keeps the shell
    // aligned with everything already on screen.
    const position = voxelToScene(raw.positions, frame.center, frame.scale);
    const normal = normalToScene(raw.normals);
    structures[structureIndex] = { position, normal, index: raw.indices };
    transferables.push(position.buffer, normal.buffer, raw.indices.buffer);
  }

  const result: TwinStructuresResult = { kind: "structures", requestId, caseId, structures };
  (self as unknown as Worker).postMessage(result, transferables);
}
