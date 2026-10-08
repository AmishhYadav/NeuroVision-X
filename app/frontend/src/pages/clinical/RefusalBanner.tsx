import type { ClinicalJob } from "../../api";

interface RefusalBannerProps {
  job: ClinicalJob;
  /**
   * Skip the input-QC findings. ClinicalPage renders every input-QC finding
   * in its own InputQCPanel, so listing the refuse ones here too would
   * duplicate them. Defaults to false so the banner still works alone.
   */
  hideInputQc?: boolean;
}

interface RefusalReason {
  source: string;
  message: string;
}

/**
 * Pulls every REFUSE-severity finding/verdict out of whichever gate actually
 * refused this job - a study can be refused at any of three points in the
 * pipeline: E1 ingest (no structured findings, only `job.error`), either E3
 * input-QC pass (`input_qc_pre` / `input_qc_post`), or the E5 gatekeeper
 * (`gatekeeper_decision`). All three are checked; only the ones that fired
 * contribute anything, so this reads correctly no matter which stage refused.
 */
function collectReasons(job: ClinicalJob, hideInputQc: boolean): RefusalReason[] {
  const reasons: RefusalReason[] = [];

  if (!hideInputQc && job.input_qc_pre) {
    for (const finding of job.input_qc_pre.findings) {
      if (finding.severity === "refuse") {
        reasons.push({
          source: `Input QC (pre-preprocessing) · ${finding.check}`,
          message: finding.message,
        });
      }
    }
  }
  if (!hideInputQc && job.input_qc_post) {
    for (const finding of job.input_qc_post.findings) {
      if (finding.severity === "refuse") {
        reasons.push({
          source: `Input QC (post-preprocessing) · ${finding.check}`,
          message: finding.message,
        });
      }
    }
  }
  if (job.gatekeeper_decision) {
    for (const verdict of job.gatekeeper_decision.verdicts) {
      if (verdict.decision === "refuse") {
        reasons.push({ source: `Gatekeeper · ${verdict.signal}`, message: verdict.message });
      }
    }
  }
  return reasons;
}

/**
 * Shown when `job.state === "refused"`.
 *
 * Worded neutrally on purpose: a refusal here is the pipeline correctly
 * declining a study it determined it could not safely segment, not a bug or
 * a system failure (see `app/backend/clinical_jobs.py`'s module docstring
 * and `app/README.md`'s clinical-upload section, both explicit about this).
 * No alarm-red styling - gate-caution (amber) is the only colour accent,
 * never gate-refuse, which is reserved for genuine failures.
 */
export function RefusalBanner({ job, hideInputQc = false }: RefusalBannerProps) {
  const reasons = collectReasons(job, hideInputQc);

  return (
    <div role="status" className="glass-panel flex flex-col gap-3 p-5">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 shrink-0 rounded-full bg-gate-caution" aria-hidden="true" />
        <span className="font-heading text-lg font-semibold text-gate-caution">
          Declined — not segmented
        </span>
      </div>
      <p className="text-sm leading-relaxed text-text-secondary">
        The pipeline determined it could not safely segment this study and stopped before
        producing a result. This is the intended, correct outcome for a study a gate cannot
        clear - not a system error.
      </p>
      {job.error && (
        <p className="text-xs leading-relaxed text-text-primary">{job.error}</p>
      )}
      {reasons.length > 0 && (
        <ul className="flex flex-col gap-1.5 border-t border-surface-seam pt-3">
          {reasons.map((r, i) => (
            <li key={i} className="text-xs leading-relaxed text-text-secondary">
              <span className="text-text-primary">{r.source}</span>: {r.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
