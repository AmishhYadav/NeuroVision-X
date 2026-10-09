// Pure maths behind the landing page's scroll-driven motion. Kept free of
// React, three.js and the DOM so it is unit-testable under plain Node (see
// scrollScene.test.ts).

/** Clamp to [0, 1]. */
export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Where `v` sits between `a` and `b`, clamped to [0, 1]. Returns 1 when the
 * range is empty or inverted and `v` has reached `b`, so a degenerate range
 * never leaves an animation stuck half-way. */
export function progressBetween(a: number, b: number, v: number): number {
  if (b <= a) return v >= b ? 1 : 0;
  return clamp01((v - a) / (b - a));
}

/** Hermite smoothstep: zero slope at both ends, so motion eases in and out. */
export function smoothstep(edge0: number, edge1: number, v: number): number {
  const t = progressBetween(edge0, edge1, v);
  return t * t * (3 - 2 * t);
}

export function easeOutCubic(t: number): number {
  const k = 1 - clamp01(t);
  return 1 - k * k * k;
}

/** Where the background brain sits on screen.
 * x, y   - its centre, as fractions of the viewport width and height.
 * size   - zoom, as "height of the old 640px hero canvas / viewport height":
 *          1 draws the brain as the hero did, but filling the whole viewport.
 * opacity - 0..1, multiplies every layer of the hologram. */
export interface BrainPose {
  x: number;
  y: number;
  size: number;
  opacity: number;
}

/** A pose pinned to one scroll position (window.scrollY, in px). */
export interface PoseKey {
  at: number;
  pose: BrainPose;
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** The pose at `scrollY`, smoothstep-blended between the two keys around it.
 * Before the first key the first pose holds; after the last, the last holds.
 * `keys` must be sorted by `at`. */
export function poseAt(keys: readonly PoseKey[], scrollY: number): BrainPose {
  if (keys.length === 0) return { x: 0.5, y: 0.5, size: 1, opacity: 0 };
  if (scrollY <= keys[0].at) return keys[0].pose;
  const last = keys[keys.length - 1];
  if (scrollY >= last.at) return last.pose;
  let i = 0;
  while (i < keys.length - 2 && scrollY > keys[i + 1].at) i++;
  const a = keys[i];
  const b = keys[i + 1];
  const t = smoothstep(a.at, b.at, scrollY);
  return {
    x: mix(a.pose.x, b.pose.x, t),
    y: mix(a.pose.y, b.pose.y, t),
    size: mix(a.pose.size, b.pose.size, t),
    opacity: mix(a.pose.opacity, b.pose.opacity, t),
  };
}

/** A count-up frame of a displayed figure: every plain decimal number inside
 * `target` is scaled by `t` and printed with the same number of decimals, and
 * everything else (sign, %, ×, en dash) is kept. At t >= 1 the target string
 * comes back verbatim, so the settled value is always exactly the published
 * one. Only plain decimals are handled: "1.4e-21" would scale both parts. */
export function tickValue(target: string, t: number): string {
  const k = clamp01(t);
  if (k >= 1) return target;
  return target.replace(/\d+(?:\.\d+)?/g, (m) => {
    const dot = m.indexOf(".");
    const decimals = dot === -1 ? 0 : m.length - dot - 1;
    return (parseFloat(m) * k).toFixed(decimals);
  });
}
