// Tiny, pure-where-possible helpers for turning a fetched `Blob` into a
// browser download. Split out of `api.ts` (which only builds requests and
// parses responses) and `ClinicalStudyViewer.tsx` (which only wires state)
// so the string-parsing half of "download this zip" has its own unit test
// with no DOM at all, and the DOM-touching half is one small function
// whoever calls it can trust without re-reading `api.ts`.
//
// Nothing here runs at import time - `triggerBlobDownload` only touches
// `document`/`URL` when it is actually CALLED, so this module imports
// cleanly under vitest's "node" environment (no jsdom), same as
// `twinSnapshot.ts`.

// Matches a `Content-Disposition` header's `filename` parameter, quoted or
// bare: `attachment; filename="x.zip"` or `attachment; filename=x.zip`. The
// server (`export_clinical_job`) always sends the quoted form, but the bare
// form is legal HTTP and cheap to accept too. Deliberately does not attempt
// the RFC 5987 `filename*=UTF-8''...` extended form - this endpoint never
// sends one, and a job id is ASCII-safe by construction (see
// `clinical_jobs.py`'s id generation), so there is nothing here that would
// ever need it.
const FILENAME_PARAM = /filename="([^"]+)"|filename=([^;]+)/i;

/**
 * Reads the `filename` a `Content-Disposition` header names, or `fallback`
 * if the header is missing or names none.
 *
 * Pure - takes the header value as a plain string (or `null`) rather than a
 * `Response`, so it needs no `fetch`/DOM machinery to test.
 */
export function filenameFromContentDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = FILENAME_PARAM.exec(header);
  if (!match) return fallback;
  // Group 1 is the quoted form, group 2 the bare form - exactly one of the
  // two is set depending on which alternative matched.
  const name = (match[1] ?? match[2])?.trim();
  return name ? name : fallback;
}

/**
 * Prompts the browser to save `blob` as `filename`.
 *
 * The standard "fetch a blob, then click a synthetic `<a download>`" dance:
 * an object URL is the only way to give an `<a>` tag something to point at
 * that isn't already a same-origin file, and the URL is revoked right after
 * the click so the browser can free the blob's memory instead of holding it
 * for the rest of the page's life. `doc` defaults to the real `document` but
 * is a parameter (not read from the global at call time), so a caller could
 * substitute a fake one in a test without this file importing `jsdom`.
 */
export function triggerBlobDownload(blob: Blob, filename: string, doc: Document = document): void {
  const url = URL.createObjectURL(blob);
  const link = doc.createElement("a");
  link.href = url;
  link.download = filename;
  doc.body.appendChild(link);
  link.click();
  doc.body.removeChild(link);
  URL.revokeObjectURL(url);
}
