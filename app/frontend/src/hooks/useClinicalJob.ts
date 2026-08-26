import { useEffect, useState } from "react";
import { ApiUnreachableError, getClinicalJob, type ClinicalJob } from "../api";

const POLL_INTERVAL_MS = 1500;

export interface ClinicalJobHookState {
  job: ClinicalJob | null;
  error: string | null;
}

const EMPTY_STATE: ClinicalJobHookState = { job: null, error: null };

/**
 * Polls one clinical job's status until it reaches a terminal state.
 *
 * Polls on an interval while `state` is `"queued"` or `"running"`, and stops
 * (no more requests scheduled) once it reaches `"done"`, `"refused"` or
 * `"failed"` - or once a fetch itself fails, the same fail-stop discipline
 * `useCaseData` uses rather than retrying an unreachable API silently
 * forever. Every jobId change gets its own `AbortController` and cancels any
 * in-flight poll, mirroring `useCaseData`'s per-case-switch guard so a fast
 * switch between jobs can never land a stale response on top of a newer one.
 */
export function useClinicalJob(jobId: string | null): ClinicalJobHookState {
  const [state, setState] = useState<ClinicalJobHookState>(EMPTY_STATE);

  useEffect(() => {
    if (!jobId) {
      setState(EMPTY_STATE);
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let controller: AbortController | null = null;

    async function poll() {
      controller = new AbortController();
      try {
        const job = await getClinicalJob(jobId as string, controller.signal);
        if (cancelled) return;
        setState({ job, error: null });
        if (job.state === "queued" || job.state === "running") {
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === "AbortError") return;
        const message =
          err instanceof ApiUnreachableError
            ? "No response from the API. Start it with `uvicorn app.backend.main:app --reload`."
            : err instanceof Error
              ? err.message
              : "Failed to load job status.";
        setState((prev) => ({ job: prev.job, error: message }));
      }
    }

    setState(EMPTY_STATE);
    poll();

    return () => {
      cancelled = true;
      controller?.abort();
      if (timer !== null) clearTimeout(timer);
    };
  }, [jobId]);

  return state;
}
