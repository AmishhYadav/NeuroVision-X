// Computes the digital twin's meshes off the main thread: a brain-shell
// isosurface (split at the mid-sagittal plane) plus one mesh per tumour
// class, all from the SAME case data the rest of the tool already fetches
// (see useCaseData) - never from a fixed, hardcoded case. A ~3-4M-voxel
// volume takes real time to surface; doing it here keeps the tab responsive
// while it runs.
import { extentCenter, meshVoxelField, normalToScene, voxelToScene } from "../lib/twinGeometry";
import { sampleLayersForClasses } from "../lib/twinLayers";
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
self.onmessage = (e: MessageEvent<TwinMeshRequest | TwinSampleRequest>) => {
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
  handleMeshRequest(data as TwinMeshRequest);
};

function handleMeshRequest(request: TwinMeshRequest) {
  const { requestId, caseId, shape, spacing, modalityVolumes, tumorMask, tumorSource, scalarLayers, keepVoxelGeometry } =
    request;
  const [D, H, W] = shape;
  const n = D * H * W;

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
    const outIdx: number[] = [];
    const remap = new Map<number, number>();
    const outPos: number[] = [];
    const outNorm: number[] = [];
    for (let t = 0; t < nTris; t++) {
      const a = indices[t * 3];
      const b = indices[t * 3 + 1];
      const c = indices[t * 3 + 2];
      const meanX = (positions[a * 3] + positions[b * 3] + positions[c * 3]) / 3;
      const isLeft = meanX < 0;
      if (isLeft !== keepLeft) continue;
      for (const idx of [a, b, c]) {
        let mapped = remap.get(idx);
        if (mapped === undefined) {
          mapped = outPos.length / 3;
          remap.set(idx, mapped);
          outPos.push(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
          outNorm.push(normals[idx * 3], normals[idx * 3 + 1], normals[idx * 3 + 2]);
        }
        outIdx.push(mapped);
      }
    }
    return {
      position: new Float32Array(outPos),
      normal: new Float32Array(outNorm),
      index: new Uint32Array(outIdx),
    };
  };

  const brainLeft = splitMesh(brainScenePos, brainSceneNorm, brainRaw.indices, true);
  const brainRight = splitMesh(brainScenePos, brainSceneNorm, brainRaw.indices, false);

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

    for (const [clsStr, name] of Object.entries(CLASS_NAMES)) {
      const cls = Number(clsStr);
      const field = new Float32Array(n);
      let voxelCount = 0;
      for (let i = 0; i < n; i++) {
        if (tumorMask[i] === cls) {
          field[i] = 1;
          voxelCount++;
        }
      }
      classVolumesMl[name] = Math.round(voxelCount * voxelMl * 100) / 100;
      if (voxelCount < 20) continue; // too small to surface meaningfully
      const raw = meshVoxelField(field, [D, H, W], 0.5);
      if (raw.positions.length === 0) continue;
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
