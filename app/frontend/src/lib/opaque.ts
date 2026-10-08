/**
 * A handle that hides a large value from React's dev-mode prop diffing.
 *
 * React (dev) enumerates every own enumerable key of changed props when it
 * logs component renders; a typed array of millions of voxels makes that take
 * seconds per render. A private field is not enumerable, so React sees an
 * empty object. Identity is cached per wrapped value so memoisation (useMemo /
 * useEffect deps / React.memo) keeps working: wrapping the same array twice
 * gives the same wrapper.
 */
export class Opaque<T> {
  readonly #value: T;

  constructor(value: T) {
    this.#value = value;
  }

  get value(): T {
    return this.#value;
  }
}

const cache = new WeakMap<object, Opaque<object>>();

export function opaque<T extends object>(value: T): Opaque<T>;
export function opaque<T extends object>(value: T | null | undefined): Opaque<T> | null;
export function opaque<T extends object>(value: T | null | undefined): Opaque<T> | null {
  if (value == null) return null;
  let wrapper = cache.get(value) as Opaque<T> | undefined;
  if (!wrapper) {
    wrapper = new Opaque(value);
    cache.set(value, wrapper);
  }
  return wrapper;
}
