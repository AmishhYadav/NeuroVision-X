import { useEffect, useState } from "react";
import { ApiUnreachableError, listClinicalJobs, type ClinicalJob, type ClinicalJobState } from "../../api";
import { formatRelativeTime } from "../../lib/relativeTime";

interface PreviousStudiesProps {
  onOpen: (jobId: string) => void;
}

const STATE_LABEL: Record<ClinicalJobState, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  refused: "Declined",
  failed: "Failed",
};

// Same "refused is not a fault" framing as ClinicalJobStatus's dot colours -
// neutral for a normal completion, amber for a correct refusal, dim for a
// genuine failure so it doesn't read as more urgent than a demo warrants.
const STATE_TAG_CLASS: Record<ClinicalJobState, string> = {
  queued: "text-text-secondary",
  running: "text-text-secondary",
  done: "text-text-primary",
  refused: "text-data-amber",
  failed: "text-text-dim",
};

function decisionLabel(job: ClinicalJob): string | null {
  if (job.state !== "done" || !job.gatekeeper_decision) return null;
  return job.gatekeeper_decision.decision.replace(/_/g, " ");
}

/**
 * Lists every clinical job the backend still has on disk, so a study that
 * finished (or was declined) in an earlier session can be reopened without
 * its job id - see `listClinicalJobs` in `api.ts` for why this exists.
 *
 * Fetches once per mount, on an `AbortController` cleaned up on unmount.
 * `ClinicalPage` re-mounts this component whenever the upload panel comes
 * back into view (going from a selected job to `jobId === null`), which is
 * enough to keep it fresh - no shared state or manual refetch trigger needed.
 */
export function PreviousStudies({ onOpen }: PreviousStudiesProps) {
  const [jobs, setJobs] = useState<ClinicalJob[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listClinicalJobs(controller.signal)
      .then((res) => setJobs(res.jobs))
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(
          err instanceof ApiUnreachableError
            ? "No response from the API."
            : "Could not load previous studies.",
        );
      });
    return () => controller.abort();
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-lg flex-col gap-2">
      <p className="eyebrow">Previous studies</p>

      {error && <p className="font-mono text-xs text-text-secondary">{error}</p>}

      {!error && jobs === null && (
        <p className="font-mono text-xs text-text-dim">Loading…</p>
      )}

      {!error && jobs !== null && jobs.length === 0 && (
        <p className="font-mono text-xs text-text-dim">No previous studies on this server.</p>
      )}

      {jobs && jobs.length > 0 && (
        <ul className="flex flex-col gap-1">
          {jobs.map((job) => (
            <li key={job.job_id}>
              <button
                type="button"
                data-testid="previous-study-row"
                onClick={() => onOpen(job.job_id)}
                className="flex w-full items-center gap-3 border border-surface-seam bg-surface-panel px-3 py-2 text-left font-mono text-xs transition-colors duration-[120ms] hover:border-text-dim"
              >
                <span className="shrink-0 text-text-dim">{job.job_id.slice(0, 8)}</span>
                <span
                  className={`shrink-0 font-condensed text-[11px] tracking-[0.1em] uppercase ${STATE_TAG_CLASS[job.state]}`}
                >
                  {STATE_LABEL[job.state]}
                </span>
                <span className="truncate text-text-secondary">{job.stage}</span>
                {decisionLabel(job) && (
                  <span className="shrink-0 text-text-secondary">{decisionLabel(job)}</span>
                )}
                <span className="tabular ml-auto shrink-0 text-text-dim">
                  {formatRelativeTime(job.created_at)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
