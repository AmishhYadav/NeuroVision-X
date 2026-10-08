import type { ClinicalJob, ClinicalJobState } from "../../api";

interface ClinicalJobStatusProps {
  job: ClinicalJob;
}

const STATE_LABEL: Record<ClinicalJobState, string> = {
  queued: "Queued",
  running: "Running",
  done: "Done",
  refused: "Declined",
  failed: "Failed",
};

// Deliberately NOT alarm-red for "refused" - a refusal is the pipeline
// correctly declining a study, not a system fault (see
// app/backend/clinical_jobs.py's module docstring). "failed" is a genuine
// fault (a bad checkpoint path, a bug) and is the only state that gets the
// same red/orange data colour Header.tsx already uses for "offline".
const STATE_DOT_CLASS: Record<ClinicalJobState, string> = {
  queued: "bg-text-dim",
  running: "bg-text-secondary animate-pulse",
  done: "bg-data-oedema",
  refused: "bg-data-amber",
  failed: "bg-data-enhancing",
};

/** State badge + stage label + a determinate progress bar (`job.progress` is already `0..1`). */
export function ClinicalJobStatus({ job }: ClinicalJobStatusProps) {
  const pct = Math.round(Math.min(1, Math.max(0, job.progress)) * 100);

  return (
    <div className="flex flex-col gap-2 border border-surface-seam bg-surface-panel px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${STATE_DOT_CLASS[job.state]}`}
          aria-hidden="true"
        />
        <span className="font-condensed text-xs font-semibold tracking-[0.1em] text-text-primary uppercase">
          {STATE_LABEL[job.state]}
        </span>
        <span className="truncate font-mono text-xs text-text-secondary">{job.stage}</span>
        <span className="tabular ml-auto shrink-0 font-mono text-[11px] text-text-dim">
          job {job.job_id.slice(0, 8)}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="h-px flex-1 bg-surface-seam">
          <span
            className="block h-px bg-text-secondary transition-[width] duration-[120ms]"
            style={{ width: `${pct}%` }}
          />
        </span>
        <span className="tabular w-9 shrink-0 font-mono text-[11px] text-text-dim">{pct}%</span>
      </div>
    </div>
  );
}
