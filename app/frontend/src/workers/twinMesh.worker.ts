// Computes the digital twin's meshes off the main thread: a brain-shell
// isosurface (split at the mid-sagittal plane) plus one mesh per tumour
// class, all from the SAME case data the rest of the tool already fetches
// (see useCaseData) - never from a fixed, hardcoded case. A ~3-4M-voxel
// volume takes real time to surface; doing it here keeps the tab responsive
// while it runs.
import { surfaceNets } from "../lib/surfaceNets";

export type ClassName = "necrotic" | "oedema" | "enhancing";

export interface TwinMeshRequest {
  requestId: number;
  caseId: string;
  shape: [number, number, number]; // (D, H, W)
  spacing: [number, number, number];
  modalityVolumes: Uint8Array[]; // whichever modalities loaded, OR'd for the brain mask
  tumorMask: Uint8Array | null; // label (preferred) or prediction, {0,1,2,3}
  tumorSource: "label" | "prediction" | null;
}

export interface TwinMeshResult {
  requestId: number;
  caseId: string;
  brainLeft: { position: Float32Array; normal: Float32Array; index: Uint32Array };
  brainRight: { position: Float32Array; normal: Float32Array; index: Uint32Array };
  tumor: Partial<Record<ClassName, { position: Float32Array; normal: Float32Array; index: Uint32Array }>>;
  tumorSource: "label" | "prediction" | null;
  tumorCentroidScene: [number, number, number] | null;
  classVolumesMl: Partial<Record<ClassName, number>>;
}

const CLASS_NAMES: Record<number, ClassName> = { 1: "necrotic", 2: "oedema", 3: "enhancing" };

function toSceneTransform() {
  // Same convention as slicing.ts documents and the offline extraction
  // script used: d -> X (R<->L), w -> Y (I->S, up), h -> Z (A->P). Center
  // and scale are established by the caller from the brain shell's own
  // extent and reused for the tumour meshes so the two stay aligned.
  return (positions: Float32Array, center: [number, number, number], scale: number) => {
    const out = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i += 3) {
      const d = positions[i];
      const h = positions[i + 1];
      const w = positions[i + 2];
      out[i] = (d - center[0]) * scale;
      out[i + 1] = (w - center[2]) * scale;
      out[i + 2] = (h - center[1]) * scale;
    }
    return out;
  };
}

function extentCenter(positions: Float32Array): { center: [number, number, number]; scale: number } {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const center: [number, number, number] = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
  return { center, scale: 1.8 / extent };
}

self.onmessage = (e: MessageEvent<TwinMeshRequest>) => {
  const { requestId, caseId, shape, spacing, modalityVolumes, tumorMask, tumorSource } = e.data;
  const [D, H, W] = shape;
  const n = D * H * W;

  const brainField = new Float32Array(n);
  for (const vol of modalityVolumes) {
    for (let i = 0; i < n; i++) {
      if (vol[i] !== 0) brainField[i] = 1;
    }
  }

  const brainRaw = surfaceNets(brainField, [D, H, W], 0.5);
  const { center, scale } = extentCenter(brainRaw.positions);
  const transform = toSceneTransform();
  const brainScenePos = transform(brainRaw.positions, center, scale);

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

  const brainLeft = splitMesh(brainScenePos, brainRaw.normals, brainRaw.indices, true);
  const brainRight = splitMesh(brainScenePos, brainRaw.normals, brainRaw.indices, false);

  const tumor: TwinMeshResult["tumor"] = {};
  const classVolumesMl: TwinMeshResult["classVolumesMl"] = {};
  const voxelMl = (spacing[0] * spacing[1] * spacing[2]) / 1000;
  let tumorCentroidScene: [number, number, number] | null = null;

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
      const centroidScene = transform(centroidVoxel, center, scale);
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
      const raw = surfaceNets(field, [D, H, W], 0.5);
      if (raw.positions.length === 0) continue;
      const scenePos = transform(raw.positions, center, scale);
      tumor[name] = { position: scenePos, normal: raw.normals, index: raw.indices };
    }
  }

  const result: TwinMeshResult = {
    requestId,
    caseId,
    brainLeft,
    brainRight,
    tumor,
    tumorSource,
    tumorCentroidScene,
    classVolumesMl,
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
  (self as unknown as Worker).postMessage(result, transferables);
};
