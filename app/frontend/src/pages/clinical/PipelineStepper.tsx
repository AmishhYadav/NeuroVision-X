import type { ClinicalJob } from "../../api";
import { PIPELINE_STEPS, stepStatuses, type StepStatus } from "../../lib/pipelineSteps";

const BADGE_CLASS: Record<StepStatus, string> = {
  pending: "border-surface-seam text-text-dim",
  active: "border-brand-primary text-brand-primary shadow-[0_0_0_4px_rgb(108_92_231/0.15)] animate-pulse",
  done: "border-brand-teal text-brand-teal",
  refused: "border-gate-caution text-gate-caution",
  failed: "border-gate-refuse text-gate-refuse",
};

// A refusal is a correct outcome, so its step reads "declined", not "error".
const STATUS_WORD: Record<StepStatus, string> = {
  pending: "pending",
  active: "in progress",
  done: "done",
  refused: "declined here",
  failed: "failed",
};

/** Eight fixed steps, coloured by `stepStatuses(job)`. Progress bar only while running. */
export function PipelineStepper({ job }: { job: ClinicalJob | null }) {
  const statuses = stepStatuses(job);
  const running = job?.state === "running";
  const pct = job ? Math.round(Math.min(1, Math.max(0, job.progress)) * 100) : 0;

  return (
    <section className="glass-panel p-5" aria-label="Pipeline">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-heading text-base font-semibold">Pipeline</h2>
          <p className="text-xs text-text-dim">Fixed order, from upload to report.</p>
        </div>
        {running && job && (
          <div className="flex items-center gap-2">
            <span className="chip">{job.stage}</span>
            <span className="font-mono text-xs text-text-secondary">{pct}%</span>
          </div>
        )}
      </div>

      <ol className="mt-5 flex items-start overflow-x-auto pb-2">
        {PIPELINE_STEPS.map((step, i) => {
          const status = statuses[i];
          const isLast = i === PIPELINE_STEPS.length - 1;
          const num = String(i + 1).padStart(2, "0");
          return (
            <li
              key={step.key}
              aria-label={`Step ${i + 1}, ${step.label}: ${status}`}
              className="flex min-w-[120px] flex-1 flex-col"
            >
              <div className="flex items-center">
                <span
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border bg-surface-panel font-mono text-xs ${BADGE_CLASS[status]}`}
                >
                  {num}
                </span>
                {!isLast && <span className="h-px flex-1 bg-surface-seam" aria-hidden="true" />}
              </div>
              <div className="mt-2 pr-3">
                <p className="font-heading text-sm text-text-primary">{step.label}</p>
                <p className="text-xs text-text-dim">{step.detail}</p>
                {(status === "failed" || status === "refused") && (
                  <p
                    className={`mt-1 text-xs ${status === "failed" ? "text-gate-refuse" : "text-gate-caution"}`}
                  >
                    {STATUS_WORD[status]}
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      {running && (
        <div className="mt-3 h-1 w-full overflow-hidden rounded bg-surface-seam">
          <div
            className="h-1 rounded bg-brand-primary transition-[width] duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
    </section>
  );
}
