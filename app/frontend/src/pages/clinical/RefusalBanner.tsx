import type { ClinicalJob } from "../../api";

interface RefusalBannerProps {
  job: ClinicalJob;
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
function collectReasons(job: ClinicalJob): RefusalReason[] {
  const reasons: RefusalReason[] = [];

  if (job.input_qc_pre) {
    for (const finding of job.input_qc_pre.findings) {
      if (finding.severity === "refuse") {
        reasons.push({
          source: `Input QC (pre-preprocessing) · ${finding.check}`,
          message: finding.message,
        });
      }
    }
  }
  if (job.input_qc_post) {
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
 * No alarm-red styling - a small amber dot is the only colour accent, the
 * same restrained treatment `Header.tsx`'s reachability dot and
 * `ReportPanel.tsx`'s segmentation-source badge already use for status,
 * never colouring the surrounding text or border.
 */
export function RefusalBanner({ job }: RefusalBannerProps) {
  const reasons = collectReasons(job);

  return (
    <div
      role="status"
      className="flex flex-col gap-3 border border-surface-seam bg-surface-panel px-4 py-4"
    >
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 shrink-0 rounded-full bg-data-amber" aria-hidden="true" />
        <span className="font-condensed text-xs font-semibold tracking-[0.12em] text-text-primary uppercase">
          Declined — not segmented
        </span>
      </div>
      <p className="font-mono text-xs leading-relaxed text-text-secondary">
        The pipeline determined it could not safely segment this study and stopped before
        producing a result. This is the intended, correct outcome for a study a gate cannot
        clear - not a system error.
      </p>
      {job.error && (
        <p className="font-mono text-xs leading-relaxed text-text-primary">{job.error}</p>
      )}
      {reasons.length > 0 && (
        <ul className="flex flex-col gap-1.5 border-t border-surface-seam pt-3">
          {reasons.map((r, i) => (
            <li key={i} className="font-mono text-[11px] leading-relaxed text-text-secondary">
              <span className="text-text-primary">{r.source}</span>: {r.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
