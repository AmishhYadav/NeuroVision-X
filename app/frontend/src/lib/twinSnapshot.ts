// Captures the 3D twin's WebGL canvas as a downloadable PNG.
//
// `BrainTwinScene.tsx` creates its canvas with `gl={{ preserveDrawingBuffer:
// true }}` (see its comment there) precisely so that `toDataURL()` returns
// the actually-rendered picture instead of a black frame - WebGL clears its
// backbuffer after each paint by default, and without that flag the browser
// would read back whatever was left over (usually nothing). Everything here
// is a pure function over that already-rendered canvas: no DOM access beyond
// what's passed in, so it can be unit-tested with a stub in the "node"
// vitest environment rather than a real HTMLCanvasElement.

/**
 * Minimal canvas surface this module needs - lets tests pass a stub instead
 * of a real HTMLCanvasElement.
 */
export interface SnapshotCanvas {
  toDataURL(type?: string, quality?: number): string;
  width: number;
  height: number;
}

// A `data:image/png;base64,` PNG shorter than this cannot possibly hold a
// real image - a canvas that failed to render (or was never drawn to)
// typically comes back either empty or as a single-pixel/blank encoding
// that's far shorter than any genuine capture. This is a cheap sanity floor,
// not a full image decode.
const MIN_SANE_DATA_URL_LENGTH = 100;
const PNG_DATA_URL_PREFIX = "data:image/png;base64,";

/**
 * Reads the canvas's current backbuffer as a PNG data URL.
 *
 * Throws if the canvas has zero width or height (nothing was ever rendered
 * into it), or if the resulting data URL is too short to plausibly contain a
 * real PNG (the capture failed or returned a blank image).
 */
export function captureCanvasDataUrl(canvas: SnapshotCanvas): string {
  if (canvas.width === 0 || canvas.height === 0) {
    throw new Error("snapshot: canvas has zero size");
  }

  const dataUrl = canvas.toDataURL("image/png");

  if (dataUrl.length < MIN_SANE_DATA_URL_LENGTH || !dataUrl.startsWith(PNG_DATA_URL_PREFIX)) {
    throw new Error("snapshot: canvas returned an empty image");
  }

  return dataUrl;
}

/**
 * Decodes a `data:<mime>;base64,<...>` URL into a `Blob` of that same mime
 * type. Pure - uses `atob`, which is available in both browsers and Node
 * (>=16), so this needs no DOM.
 */
export function dataUrlToBlob(dataUrl: string): Blob {
  const commaIndex = dataUrl.indexOf(",");
  const header = dataUrl.slice(0, commaIndex);
  const base64 = dataUrl.slice(commaIndex + 1);

  const mimeMatch = /^data:([^;]+);base64$/.exec(header);
  const mimeType = mimeMatch ? mimeMatch[1] : "application/octet-stream";

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return new Blob([bytes], { type: mimeType });
}

/** Captures the canvas and decodes it straight to a PNG `Blob`. */
export function captureCanvas(canvas: SnapshotCanvas): Blob {
  return dataUrlToBlob(captureCanvasDataUrl(canvas));
}

// Anything that isn't a lowercase letter, digit, or hyphen - used to turn a
// free-text view label (e.g. "3D twin") into a filename-safe token.
const UNSAFE_FILENAME_CHARS = /[^a-z0-9-]+/g;

/**
 * Builds a snapshot filename: `twin_<jobId8>_<view>_<YYYYMMDD-HHMMSS>.png`.
 *
 * `jobId` is truncated to its first 8 characters; `view` is lowercased and
 * anything outside `[a-z0-9-]` is collapsed to a single hyphen. Uses UTC
 * getters (not local-time ones) so the same instant always produces the same
 * filename regardless of the machine's timezone - this is what makes the
 * test below timezone-independent.
 */
export function snapshotFilename(jobId: string, view: string, now: Date = new Date()): string {
  const shortJobId = jobId.slice(0, 8);
  const safeView = view.toLowerCase().trim().replace(UNSAFE_FILENAME_CHARS, "-");

  const pad = (n: number, width = 2) => String(n).padStart(width, "0");
  const stamp =
    `${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1)}${pad(now.getUTCDate())}` +
    `-${pad(now.getUTCHours())}${pad(now.getUTCMinutes())}${pad(now.getUTCSeconds())}`;

  return `twin_${shortJobId}_${safeView}_${stamp}.png`;
}

/**
 * Finds the twin's `<canvas>` element under `root`. Kept as a one-line
 * wrapper so the DOM query lives in exactly one place in the codebase.
 */
export function findTwinCanvas(root: ParentNode): HTMLCanvasElement | null {
  return root.querySelector("canvas");
}
