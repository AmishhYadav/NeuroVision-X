// Shared classifier for "supplementary" clinical-job fetches - uncertainty,
// the conformal band, Grad-CAM, geometry. These artifacts are extras layered
// on top of the four volumes and the prediction mask (the actual product of
// a clinical job): a study a clinician can still read is not nothing just
// because one overlay's backend route 500'd. `useClinicalJobVolumes` awaits
// several of these fetches in one `Promise.all`, so without this classifier
// a single throwing overlay would reject the whole batch and blank the
// study - slices, twin and report included - even though everything else
// had already loaded. See that hook's doc comment for the full core vs.
// supplementary split.

import { ApiError, ApiUnreachableError } from "../api";

/**
 * Awaits a supplementary fetch and turns "this one artifact is missing" into
 * data instead of a thrown error that would take the whole batch down with
 * it.
 *
 * - A 404 (`ApiError` with `status === 404`) is an expected, silent absence -
 *   resolves to `null`, no warning appended.
 * - Any other error (`ApiError` with a different status, or any other
 *   `Error`) resolves to `null` and appends a human-readable line to
 *   `warnings`, e.g. `"Conformal band (WT) unavailable: 500 Internal Server
 *   Error on /clinical/jobs/.../conformal-band/WT"`.
 * - `ApiUnreachableError` and an aborted fetch (`AbortError`) are NOT
 *   downgraded - they rethrow, because if the API itself is gone or the
 *   request was cancelled, nothing else in the batch is going to succeed
 *   either, and the caller's own abort/unreachable handling needs to see it.
 *
 * @param label - Human-readable name for the artifact, used to build the
 *   warning line (e.g. `"Conformal band (WT)"`, `"Grad-CAM (TC)"`).
 * @param p - The in-flight fetch promise to settle.
 * @param warnings - Mutated in place: a warning line is pushed onto it when
 *   this fetch fails with anything other than a 404.
 */
export async function settleSupplementary<T>(
  label: string,
  p: Promise<T>,
  warnings: string[],
): Promise<T | null> {
  try {
    return await p;
  } catch (err) {
    if (err instanceof ApiUnreachableError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    if (err instanceof ApiError && err.status === 404) return null;
    const message = err instanceof Error ? err.message : String(err);
    warnings.push(`${label} unavailable: ${message}`);
    return null;
  }
}
