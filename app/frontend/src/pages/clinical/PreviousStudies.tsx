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
// teal for a normal completion, amber for a correct refusal, gate-refuse only
// for a genuine failure.
const STATE_TAG_CLASS: Record<ClinicalJobState, string> = {
  queued: "text-text-secondary",
  running: "text-brand-primary",
  done: "text-brand-teal",
  refused: "text-gate-caution",
  failed: "text-gate-refuse",
};

// Shown whenever the gatekeeper ran, including for refused jobs. Refuse is
// amber like caution: a refusal is a correct outcome, not an alarm.
const DECISION_DISPLAY = {
  proceed: { text: "proceed", cls: "text-gate-proceed" },
  proceed_with_caution: { text: "caution", cls: "text-gate-caution" },
  refuse: { text: "refuse", cls: "text-gate-caution" },
} as const;

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
    <section className="glass-panel p-5">
      <h2 className="font-heading text-base font-semibold">Previous studies</h2>

      {error && <p className="mt-3 text-xs text-text-secondary">{error}</p>}

      {!error && jobs === null && <p className="mt-3 text-xs text-text-dim">Loading…</p>}

      {!error && jobs !== null && jobs.length === 0 && (
        <p className="mt-3 text-xs text-text-dim">No previous studies on this server.</p>
      )}

      {jobs && jobs.length > 0 && (
        <ul className="mt-3 flex flex-col">
          <li className="eyebrow hidden grid-cols-[88px_96px_minmax(0,1fr)_140px_90px] gap-3 px-3 pb-2 sm:grid">
            <span>Job</span>
            <span>State</span>
            <span>Stage</span>
            <span>Decision</span>
            <span className="text-right">Created</span>
          </li>
          {jobs.map((job) => (
            <li key={job.job_id} className="border-t border-surface-seam">
              <button
                type="button"
                data-testid="previous-study-row"
                onClick={() => onOpen(job.job_id)}
                className="grid w-full grid-cols-[88px_96px_minmax(0,1fr)] items-center gap-3 px-3 py-2.5 text-left text-xs transition-colors duration-[120ms] hover:bg-surface-raised/60 sm:grid-cols-[88px_96px_minmax(0,1fr)_140px_90px]"
              >
                <span className="font-mono text-text-secondary">{job.job_id.slice(0, 8)}</span>
                <span className={`font-heading text-xs font-semibold ${STATE_TAG_CLASS[job.state]}`}>
                  {STATE_LABEL[job.state]}
                </span>
                <span className="truncate text-text-secondary">{job.stage}</span>
                <span
                  className={`hidden truncate sm:block ${
                    job.gatekeeper_decision
                      ? DECISION_DISPLAY[job.gatekeeper_decision.decision].cls
                      : "text-text-dim"
                  }`}
                >
                  {job.gatekeeper_decision
                    ? DECISION_DISPLAY[job.gatekeeper_decision.decision].text
                    : "—"}
                </span>
                <span className="tabular hidden text-right text-text-dim sm:block">
                  {formatRelativeTime(job.created_at)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
