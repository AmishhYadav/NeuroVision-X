import { useState } from "react";
import { useClinicalJob } from "../../hooks/useClinicalJob";
import { ClinicalJobStatus } from "./ClinicalJobStatus";
import { ClinicalStudyViewer } from "./ClinicalStudyViewer";
import { ClinicalUploadPanel } from "./ClinicalUploadPanel";
import { GatekeeperPanel } from "./GatekeeperPanel";
import { RefusalBanner } from "./RefusalBanner";

// "/app" (main.tsx) remains the separate viewer for PRECOMPUTED evaluation
// cases - this page is a second, independent entry point that puts a REAL
// uploaded DICOM study through the live clinical pipeline
// (app/backend/clinical_jobs.py). Same pushState-based navigation Landing.tsx's
// ViewerLink already uses, no router dependency.
function goToLanding() {
  window.history.pushState({}, "", "/");
  window.dispatchEvent(new PopStateEvent("popstate"));
}

/**
 * Top-level clinical-upload page: pick a file, watch the job move through
 * the pipeline, then either a refusal explanation or the segmented study.
 *
 * Tracks only the active `job_id` in local state; `useClinicalJob` does all
 * the polling. "New upload" simply drops that id, discarding this component's
 * view of the old job - it does not call `deleteClinicalJob`, so a job a
 * user has moved on from still exists server-side (and can be revisited by
 * its id) until whatever housekeeping the backend does for `NVX_JOB_DIR`
 * reclaims it.
 */
export function ClinicalPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const { job, error } = useClinicalJob(jobId);

  const isDone = job?.state === "done";

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface-page text-text-primary">
      <header className="flex h-12 shrink-0 items-center gap-4 border-b border-surface-seam bg-surface-panel px-4">
        <a
          href="/"
          onClick={(e) => {
            e.preventDefault();
            goToLanding();
          }}
          className="shrink-0 font-condensed text-[11px] tracking-[0.12em] text-text-secondary uppercase transition-colors duration-[120ms] hover:text-text-primary"
        >
          ← NeuroVision-X
        </a>
        <h1 className="font-condensed shrink-0 text-sm font-semibold tracking-[0.12em] whitespace-nowrap text-text-primary uppercase">
          Clinical upload
        </h1>
        {job && (
          <button
            type="button"
            onClick={() => setJobId(null)}
            className="ml-auto shrink-0 rounded-sm border border-surface-seam px-2 py-1 font-condensed text-[11px] tracking-[0.12em] text-text-secondary uppercase transition-colors duration-[120ms] hover:border-text-dim hover:text-text-primary"
          >
            New upload
          </button>
        )}
      </header>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          className={`flex flex-col gap-3 overflow-y-auto p-4 ${
            isDone ? "max-h-[45vh] shrink-0" : "min-h-0 flex-1"
          }`}
        >
          {!jobId && <ClinicalUploadPanel onJobCreated={setJobId} />}

          {error && <p className="font-mono text-xs text-text-secondary">{error}</p>}

          {job && (
            <>
              <ClinicalJobStatus job={job} />
              {job.state === "refused" && <RefusalBanner job={job} />}
              {job.state === "failed" && (
                <div className="border border-surface-seam bg-surface-panel px-4 py-4">
                  <p className="font-mono text-xs leading-relaxed text-text-primary">
                    {job.error ?? "The job failed for an unspecified reason."}
                  </p>
                </div>
              )}
              {job.state === "done" && job.gatekeeper_decision && <GatekeeperPanel job={job} />}
            </>
          )}
        </div>

        {isDone && job && (
          <div className="min-h-0 flex-1">
            <ClinicalStudyViewer jobId={job.job_id} />
          </div>
        )}
      </div>
    </div>
  );
}
