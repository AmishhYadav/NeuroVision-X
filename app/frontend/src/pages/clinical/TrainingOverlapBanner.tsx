import type { ClinicalJob } from "../../api";

interface TrainingOverlapBannerProps {
  job: ClinicalJob;
}

/**
 * Honest-label notice for a study that is a patient the deployed model was
 * trained on. A PROCEED on such a study shows the pipeline runs on scanner
 * DICOM; it is not evidence of accuracy on unseen patients.
 *
 * Informational/caution tone, never gate-refuse red: nothing went wrong, the
 * reader just needs to know what the result can and cannot show. Renders
 * nothing when the job carries no overlap (including an older backend that
 * omits the field). Rendered above the verdict in every layout, so it is
 * never inside the collapsible details panel.
 */
export function TrainingOverlapBanner({ job }: TrainingOverlapBannerProps) {
  const overlap = job.training_overlap;
  if (!overlap) return null;

  const identity = overlap.status === "probable" ? "probably" : "";
  return (
    <div
      role="note"
      data-testid="training-overlap-banner"
      className="glass-panel flex flex-col gap-1.5 border-gate-caution/40 px-4 py-3"
    >
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 shrink-0 rounded-full bg-gate-caution" aria-hidden="true" />
        <span className="font-heading text-sm font-semibold text-gate-caution">
          Training-set patient — illustration only
        </span>
      </div>
      <p className="text-xs leading-relaxed text-text-primary">
        {`This study is ${identity ? `${identity} ` : ""}${overlap.brats_id} in the model's training data (${overlap.source_id}). The result shows that the pipeline runs on scanner DICOM; it is not evidence of accuracy on unseen patients.`}
      </p>
      <p className="text-xs leading-relaxed text-text-dim">Source: {overlap.evidence}</p>
    </div>
  );
}
