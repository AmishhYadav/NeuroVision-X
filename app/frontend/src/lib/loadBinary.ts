// Fetches static-asset flat binary buffers bundled under /public. Used by the
// landing page's real-data visuals (hero slice, digital-twin meshes) - the
// same "fetch raw bytes, no reshaping" discipline api.ts documents for the
// live API, applied to build-time assets instead.

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  return res.arrayBuffer();
}

export async function loadFloat32(url: string): Promise<Float32Array> {
  return new Float32Array(await fetchBuffer(url));
}

export async function loadUint32(url: string): Promise<Uint32Array> {
  return new Uint32Array(await fetchBuffer(url));
}

export async function loadUint8(url: string): Promise<Uint8Array> {
  return new Uint8Array(await fetchBuffer(url));
}

export interface MeshBuffers {
  position: Float32Array;
  normal: Float32Array;
  index: Uint32Array;
}

/** Loads one {name}-position.f32 / {name}-normal.f32 / {name}-index.u32 triple. */
export async function loadMesh(baseUrl: string, name: string): Promise<MeshBuffers> {
  const [position, normal, index] = await Promise.all([
    loadFloat32(`${baseUrl}/${name}-position.f32`),
    loadFloat32(`${baseUrl}/${name}-normal.f32`),
    loadUint32(`${baseUrl}/${name}-index.u32`),
  ]);
  return { position, normal, index };
}
