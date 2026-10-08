// Pure classification of a report-fetch failure into the status/message pair
// `ReportPanel` renders. Extracted verbatim from the if/else chain that used
// to live inline in `App.tsx`'s report-fetch effect, so `useClinicalReport`
// (the clinical pipeline's equivalent fetch) can share the exact same rules
// instead of re-deriving them - divergence here would mean the demo viewer
// and the clinical page explain the same kind of failure differently.

import { ApiError, ApiUnreachableError } from "../api";
import type { ReportPanelStatus } from "../components/ReportPanel";

export interface ClassifiedReportError {
  status: Exclude<ReportPanelStatus, "loading" | "loaded">;
  /** The server's `detail` message (server_error / not_found) or the client validation message (invalid). Always null for `unreachable` - there is no server response to quote. */
  message: string | null;
}

/**
 * Maps whatever a report fetch (`fetchReport` / `fetchClinicalReport`) threw
 * onto the `ReportPanelStatus` that describes it and the message worth
 * showing, if any.
 *
 * Order matters: `ApiUnreachableError` is checked before `ApiError` because
 * the two are unrelated classes (see `api.ts`), not a subtype relationship -
 * either could in principle be checked first, but `ApiUnreachableError` never
 * carries a `status`, so it is handled as its own case rather than folded
 * into the `ApiError` branch below.
 */
export function classifyReportError(err: unknown): ClassifiedReportError {
  if (err instanceof ApiUnreachableError) {
    return { status: "unreachable", message: null };
  }
  if (err instanceof ApiError && err.status === 404) {
    return { status: "not_found", message: err.message };
  }
  if (err instanceof ApiError) {
    return { status: "server_error", message: err.message };
  }
  return {
    status: "invalid",
    message: err instanceof Error ? err.message : "Failed to load report.",
  };
}
