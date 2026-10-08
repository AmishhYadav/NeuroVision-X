import { describe, expect, it } from "vitest";
import {
  captureCanvas,
  captureCanvasDataUrl,
  dataUrlToBlob,
  findTwinCanvas,
  snapshotFilename,
  type SnapshotCanvas,
} from "./twinSnapshot";

// A real, valid 1x1 transparent PNG, base64-encoded. Used (rather than an
// arbitrary string) so tests exercise actual byte-for-byte decoding through
// dataUrlToBlob/atob, not just length checks.
const TINY_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
const TINY_PNG_DATA_URL = `data:image/png;base64,${TINY_PNG_BASE64}`;

function makeStubCanvas(overrides: Partial<SnapshotCanvas> = {}): SnapshotCanvas {
  return {
    width: 4,
    height: 4,
    toDataURL: () => TINY_PNG_DATA_URL,
    ...overrides,
  };
}

describe("captureCanvasDataUrl", () => {
  it("returns the canvas's PNG data URL", () => {
    const canvas = makeStubCanvas();
    expect(captureCanvasDataUrl(canvas)).toBe(TINY_PNG_DATA_URL);
  });

  it("throws when the canvas has zero width", () => {
    const canvas = makeStubCanvas({ width: 0 });
    expect(() => captureCanvasDataUrl(canvas)).toThrow("snapshot: canvas has zero size");
  });

  it("throws when the canvas has zero height", () => {
    const canvas = makeStubCanvas({ height: 0 });
    expect(() => captureCanvasDataUrl(canvas)).toThrow("snapshot: canvas has zero size");
  });

  it("throws when the canvas returns an empty/blank image", () => {
    const canvas = makeStubCanvas({ toDataURL: () => "data:," });
    expect(() => captureCanvasDataUrl(canvas)).toThrow("snapshot: canvas returned an empty image");
  });
});

describe("dataUrlToBlob", () => {
  it("round-trips the exact PNG bytes", async () => {
    const blob = dataUrlToBlob(TINY_PNG_DATA_URL);
    expect(blob.type).toBe("image/png");

    const bytes = new Uint8Array(await blob.arrayBuffer());
    const expectedBytes = new Uint8Array(Buffer.from(TINY_PNG_BASE64, "base64"));
    expect(bytes).toEqual(expectedBytes);
  });
});

describe("captureCanvas", () => {
  it("captures and decodes the canvas to a PNG blob of the right byte length", async () => {
    const canvas = makeStubCanvas();
    const blob = captureCanvas(canvas);

    expect(blob.type).toBe("image/png");
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const expectedBytes = new Uint8Array(Buffer.from(TINY_PNG_BASE64, "base64"));
    expect(bytes.length).toBe(expectedBytes.length);
    expect(bytes).toEqual(expectedBytes);
  });

  it("throws for a zero-size stub canvas", () => {
    const canvas = makeStubCanvas({ width: 0, height: 0 });
    expect(() => captureCanvas(canvas)).toThrow("snapshot: canvas has zero size");
  });

  it("throws for a stub returning a blank data URL", () => {
    const canvas = makeStubCanvas({ toDataURL: () => "data:," });
    expect(() => captureCanvas(canvas)).toThrow("snapshot: canvas returned an empty image");
  });
});

describe("snapshotFilename", () => {
  it("builds the expected filename, truncating jobId and sanitising view", () => {
    // Uses Date.UTC + the function's UTC getters so this assertion holds
    // regardless of the machine's local timezone.
    const now = new Date(Date.UTC(2026, 8, 19, 14, 0, 0));
    const filename = snapshotFilename("a37fcaad8b324fc5...", "3D twin", now);
    expect(filename).toBe("twin_a37fcaad_3d-twin_20260919-140000.png");
  });

  it("pads single-digit month/day/hour/minute/second components", () => {
    const now = new Date(Date.UTC(2026, 0, 5, 3, 4, 5));
    const filename = snapshotFilename("shortid", "axial", now);
    expect(filename).toBe("twin_shortid_axial_20260105-030405.png");
  });
});

describe("findTwinCanvas", () => {
  it("returns the first canvas element under root", () => {
    const canvas = { tagName: "CANVAS" };
    const root = {
      querySelector: (selector: string) => (selector === "canvas" ? canvas : null),
    } as unknown as ParentNode;

    expect(findTwinCanvas(root)).toBe(canvas);
  });

  it("returns null when there is no canvas", () => {
    const root = { querySelector: () => null } as unknown as ParentNode;
    expect(findTwinCanvas(root)).toBeNull();
  });
});
