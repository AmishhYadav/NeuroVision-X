import { useEffect, useRef, useState } from "react";
import { fetchClinicalReport, type ReportResponse } from "../api";
import { classifyReportError } from "../lib/reportStatus";
import type { ReportPanelStatus } from "../components/ReportPanel";

export interface ClinicalReportState {
  status: ReportPanelStatus;
  report: ReportResponse | null;
  /** The server's `detail` message (server_error / not_found) or the client validation message (invalid). */
  errorMessage: string | null;
}

const LOADING_STATE: ClinicalReportState = { status: "loading", report: null, errorMessage: null };

/**
 * Fetches a `"done"` clinical job's Phase 4 structured report, once, when
 * `ready` becomes true.
 *
 * Mirrors `useClinicalJobVolumes`'s per-jobId `AbortController` shape (abort
 * whatever was in flight before starting a new fetch, and again on unmount)
 * but for the single report request rather than several volumes fetched in
 * parallel. Error classification is delegated to `classifyReportError` - the
 * exact same rules `App.tsx`'s demo-viewer report effect uses for
 * `fetchReport`, so the clinical page and the demo viewer describe the same
 * kind of failure the same way.
 *
 * While `!ready || !jobId` this fetches nothing and reports `"loading"`
 * (never `"loaded"` or an error) - the caller passes
 * `job?.state === "done"` for `ready`, same convention as
 * `useClinicalJobVolumes`, so a still-running or refused job is simply never
 * queried (there is nothing to fetch yet, not an error).
 */
export function useClinicalReport(jobId: string | null, ready: boolean): ClinicalReportState {
  const [state, setState] = useState<ClinicalReportState>(LOADING_STATE);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    controllerRef.current?.abort();

    if (!jobId || !ready) {
      setState(LOADING_STATE);
      return;
    }

    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;

    setState(LOADING_STATE);

    (async () => {
      try {
        const report = await fetchClinicalReport(jobId, signal);
        if (signal.aborted) return;
        setState({ status: "loaded", report, errorMessage: null });
      } catch (err) {
        if (signal.aborted) return;
        if (err instanceof DOMException && err.name === "AbortError") return;
        const c = classifyReportError(err);
        setState({ status: c.status, report: null, errorMessage: c.message });
      }
    })();

    return () => {
      controller.abort();
    };
  }, [jobId, ready]);

  return state;
}
