// Two tiny pure helpers behind useCaseData, kept free of React and fetch so
// they can be unit-tested in plain Node.

/**
 * Least-recently-used cache with a fixed capacity. `get` refreshes recency;
 * `set` evicts the oldest entry once over capacity. A Map iterates in
 * insertion order, so the first key is always the least recently used.
 */
export class LruCache<V> {
  private readonly map = new Map<string, V>();

  constructor(private readonly capacity: number) {
    if (capacity < 1) throw new Error("LruCache capacity must be >= 1");
  }

  get size(): number {
    return this.map.size;
  }

  has(key: string): boolean {
    return this.map.has(key);
  }

  get(key: string): V | undefined {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key) as V;
    this.map.delete(key);
    this.map.set(key, value); // re-insert = most recently used
    return value;
  }

  set(key: string, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    while (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as string;
      this.map.delete(oldest);
    }
  }

  keys(): string[] {
    return [...this.map.keys()];
  }

  clear(): void {
    this.map.clear();
  }
}

/**
 * Dedupes concurrent async work per key: while a promise for `key` is in
 * flight, `run` returns that same promise instead of starting another. The
 * entry is removed when the promise settles, so a failed load can be retried
 * by simply calling `run` again.
 */
export class InflightMap<V> {
  private readonly map = new Map<string, Promise<V>>();

  has(key: string): boolean {
    return this.map.has(key);
  }

  run(key: string, start: () => Promise<V>): Promise<V> {
    const existing = this.map.get(key);
    if (existing) return existing;
    const p = start();
    this.map.set(key, p);
    const clear = () => {
      if (this.map.get(key) === p) this.map.delete(key);
    };
    p.then(clear, clear);
    return p;
  }
}
