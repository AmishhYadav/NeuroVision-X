// Per-slice tumour fraction of a predicted mask, for the slice ribbon.
//
// Matches the backend's /cases/{id}/profile definition: for each slice,
// (voxels with mask > 0 in that slice) / (voxels in that slice).
// Plane axes follow slicing.ts's `sliceIndexer` exactly, so entry i of the
// result describes the very slice the viewport draws at index i:
//   sagittal -> slices along axis 0 (D), axial -> axis 2 (W), coronal -> axis 1 (H).
// The buffer is a flat (D, H, W) row-major Uint8Array: index d*H*W + h*W + w.

import type { Plane } from "../api";

export function tumourProfile(
  mask: Uint8Array,
  shape: [number, number, number],
  plane: Plane,
): number[] {
  const [D, H, W] = shape;
  const count = plane === "sagittal" ? D : plane === "coronal" ? H : W;
  const perSlice = (D * H * W) / count; // voxels in one slice of this plane
  const counts = new Float64Array(count);

  // One pass over the buffer; the slice a voxel belongs to is just one of
  // its three coordinates, so no per-voxel allocation is needed.
  let idx = 0;
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++, idx++) {
        if (mask[idx] > 0) {
          counts[plane === "sagittal" ? d : plane === "coronal" ? h : w]++;
        }
      }
    }
  }

  const out = new Array<number>(count);
  for (let i = 0; i < count; i++) out[i] = perSlice > 0 ? counts[i] / perSlice : 0;
  return out;
}
