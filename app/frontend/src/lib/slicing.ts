// Geometry for slicing a flat (D, H, W) C-order volume buffer into a 2D
// plane image. This is the ONLY place index arithmetic for volume access
// lives - the image layer, mask layer and uncertainty layer all go through
// the same `at()` so the three cannot drift apart.
//
// Index into the flat buffer for voxel (d, h, w): d*H*W + h*W + w.

import type { Plane } from "../api";

export interface SliceIndexer {
  /** Number of columns in the slice image. */
  width: number;
  /** Number of rows in the slice image. */
  height: number;
  /** Flat index into the source (D,H,W) buffer for pixel (row, col) at slice `i`. */
  at(row: number, col: number, i: number): number;
  /** Number of slices available along this plane's axis. */
  count: number;
}

/**
 * Build a plane-specific indexer over a (D, H, W) volume.
 *
 * ANATOMICAL AXES. BraTS volumes carry the affine diag(-1, -1, 1), so for a
 * voxel index (d, h, w):
 *   - axis 0 (d) runs RIGHT -> LEFT
 *   - axis 1 (h) runs ANTERIOR -> POSTERIOR
 *   - axis 2 (w) runs INFERIOR -> SUPERIOR
 *
 * The naive mapping (rows = the first remaining axis, columns = the second)
 * is what a shape test would accept, and it is wrong on screen: it puts the
 * inferior-superior axis HORIZONTAL on the coronal and sagittal planes, so
 * the head is displayed lying on its side. Every voxel is correct and the
 * picture is still a brain, which is exactly why this has to be pinned here
 * rather than noticed later.
 *
 * The display convention this encodes:
 *   - superior is UP on coronal and sagittal (hence the reversed row for w)
 *   - anterior is UP on axial, and LEFT on sagittal
 *   - the patient's left is on the RIGHT of the image (radiological
 *     convention, i.e. viewed from the feet), which follows from axis 0
 *     running right -> left as the column axis
 *
 * Concretely:
 *   - axial, index i in [0, W): rows = h (anterior at top), cols = d,
 *     width = D, height = H, pixel -> col*H*W + row*W + i
 *   - coronal, index i in [0, H): rows = reversed w (superior at top),
 *     cols = d, width = D, height = W, pixel -> col*H*W + i*W + (W-1-row)
 *   - sagittal, index i in [0, D): rows = reversed w, cols = h (anterior at
 *     left), width = H, height = W, pixel -> i*H*W + col*W + (W-1-row)
 */
export function sliceIndexer(plane: Plane, shape: [number, number, number]): SliceIndexer {
  const [D, H, W] = shape;
  switch (plane) {
    case "sagittal":
      return {
        width: H,
        height: W,
        count: D,
        at: (row, col, i) => i * H * W + col * W + (W - 1 - row),
      };
    case "coronal":
      return {
        width: D,
        height: W,
        count: H,
        at: (row, col, i) => col * H * W + i * W + (W - 1 - row),
      };
    case "axial":
      return {
        width: D,
        height: H,
        count: W,
        at: (row, col, i) => col * H * W + row * W + i,
      };
  }
}
