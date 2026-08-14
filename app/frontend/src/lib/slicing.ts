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
 * - sagittal, index i in [0, D): image = arr[i, :, :], width=W, height=H,
 *   pixel (row=h, col=w) -> i*H*W + h*W + w
 * - coronal, index i in [0, H): image = arr[:, i, :], width=W, height=D,
 *   pixel (row=d, col=w) -> d*H*W + i*W + w
 * - axial, index i in [0, W): image = arr[:, :, i], width=H, height=D,
 *   pixel (row=d, col=h) -> d*H*W + h*W + i
 */
export function sliceIndexer(plane: Plane, shape: [number, number, number]): SliceIndexer {
  const [D, H, W] = shape;
  switch (plane) {
    case "sagittal":
      return {
        width: W,
        height: H,
        count: D,
        at: (row, col, i) => i * H * W + row * W + col,
      };
    case "coronal":
      return {
        width: W,
        height: D,
        count: H,
        at: (row, col, i) => row * H * W + i * W + col,
      };
    case "axial":
      return {
        width: H,
        height: D,
        count: W,
        at: (row, col, i) => row * H * W + col * W + i,
      };
  }
}
