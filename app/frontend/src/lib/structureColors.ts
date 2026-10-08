// Deterministic per-structure colour for the twin's atlas shells (T3.5).
//
// There is no fixed palette for "structure #k" the way there is for the
// three tumour classes (CLASS_COLORS in colors.ts) - up to MAX_ATLAS_STRUCTURES
// (16) structures can be on screen at once, picked at runtime from a ~122-row
// atlas table, so the colour has to be DERIVED from the index rather than
// looked up. The golden-angle hue step (~137.5 degrees) is the standard trick
// for this: repeatedly adding it mod 360 spreads any number of hues around
// the wheel so that neighbouring indices (which is what "index k" and
// "index k+1" are, once selectStructures has ordered them) land far apart in
// hue, rather than a naive `(index / n) * 360` which clusters nearby indices
// together whenever the selection is a small subset of the full range.

const GOLDEN_ANGLE_DEG = 137.508;

/**
 * HSL -> RGB, all channels in [0, 1]. `h` in degrees (any real number - taken
 * mod 360), `s` and `l` in [0, 1]. A small self-contained implementation
 * rather than pulling in three.js here - this module has to stay pure and
 * framework-free so it can be unit-tested in vitest's node environment.
 */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const hue = ((h % 360) + 360) % 360;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0;
  let g = 0;
  let b = 0;
  if (hue < 60) {
    [r, g, b] = [c, x, 0];
  } else if (hue < 120) {
    [r, g, b] = [x, c, 0];
  } else if (hue < 180) {
    [r, g, b] = [0, c, x];
  } else if (hue < 240) {
    [r, g, b] = [0, x, c];
  } else if (hue < 300) {
    [r, g, b] = [x, 0, c];
  } else {
    [r, g, b] = [c, 0, x];
  }
  return [r + m, g + m, b + m];
}

/**
 * Deterministic RGB in [0, 1] for atlas structure `index` (the 1-based value
 * from AtlasStructureRow.index - 0 is background and never passed here).
 *
 * Fixed saturation/lightness (0.55/0.6, matching TwinModel's use of
 * THREE.Color().setHSL(h, 0.55, 0.6) in the spec) so every structure reads at
 * the same "translucent shell" brightness regardless of index - only hue
 * varies, walked around the wheel by the golden angle per index so that two
 * structures selected next to each other in display order get visually
 * distinct colours.
 */
export function structureColor(index: number): [number, number, number] {
  const hue = index * GOLDEN_ANGLE_DEG;
  return hslToRgb(hue, 0.55, 0.6);
}
