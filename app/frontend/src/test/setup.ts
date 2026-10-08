// Minimal ImageData polyfill for the "node" vitest environment.
//
// render.ts's only DOM dependency is the `ImageData` constructor. Pulling in
// jsdom/happy-dom for that one class would be a heavyweight, slow way to get
// a feature we can stub in five lines - so instead of a browser environment,
// vitest runs under plain "node" and this file supplies just enough of the
// ImageData shape (`data` / `width` / `height`) for render.ts to work and for
// tests to read pixels back out.
class FakeImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.data = new Uint8ClampedArray(width * height * 4);
  }
}

// @ts-expect-error - intentionally partial polyfill, sufficient for render.ts.
globalThis.ImageData = FakeImageData;
