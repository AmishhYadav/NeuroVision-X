/**
 * The fixed clinical pipeline as eight display steps, and the logic that maps
 * a job's backend `stage` string onto them. Pure functions, no React, so it
 * is unit-tested directly.
 *
 * The stage strings are set by app/backend/clinical_jobs.py and must match
 * exactly. A refused or failed job keeps the stage that stopped it.
 */
import type { ClinicalJobState } from "../api";

export type StepStatus = "pending" | "active" | "done" | "refused" | "failed";

export interface PipelineStep {
  key: string;
  label: string;
  detail: string;
  /** Backend `job.stage` values that belong to this step. */
  stages: string[];
}

export const PIPELINE_STEPS: PipelineStep[] = [
  { key: "ingest", label: "DICOM ingest", detail: "Assign series to T1 / T1CE / T2 / FLAIR", stages: ["ingest"] },
  { key: "qc_pre", label: "Input QC", detail: "Checks before registration", stages: ["input_qc (pre-preprocessing)"] },
  { key: "register", label: "Registration", detail: "Co-registration · SRI24 atlas · HD-BET", stages: ["clinical_preprocessing"] },
  { key: "qc_post", label: "Input QC", detail: "Checks after registration", stages: ["input_qc (post-preprocessing)"] },
  { key: "prep", label: "Preprocessing", detail: "Reorient, normalise, crop", stages: ["research_preprocessing"] },
  { key: "segment", label: "Segmentation", detail: "Dual-encoder network · ET / TC / WT", stages: ["segmenting"] },
  { key: "gate", label: "Gatekeeper", detail: "PROCEED / CAUTION / REFUSE", stages: ["gatekeeper"] },
  {
    key: "outputs",
    label: "Outputs",
    detail: "Grad-CAM · DICOM-SEG · report",
    stages: ["explaining", "exporting_dicom_seg", "generating_report"],
  },
];

/** Index into PIPELINE_STEPS for a stage string; -1 for queued, done or unknown. */
export function stepIndexForStage(stage: string): number {
  // "segmenting: <substage>" belongs to the segmentation step. Exact match or
  // the "segmenting: " prefix only, so "segmentingX" does not match.
  if (stage === "segmenting" || stage.startsWith("segmenting: ")) {
    return PIPELINE_STEPS.findIndex((s) => s.key === "segment");
  }
  return PIPELINE_STEPS.findIndex((s) => s.stages.includes(stage));
}

/** One status per step, in step order. */
export function stepStatuses(job: { state: ClinicalJobState; stage: string } | null): StepStatus[] {
  const n = PIPELINE_STEPS.length;
  const allPending = (): StepStatus[] => Array<StepStatus>(n).fill("pending");
  if (!job || job.state === "queued") return allPending();
  if (job.state === "done") return Array<StepStatus>(n).fill("done");

  const idx = stepIndexForStage(job.stage);
  if (idx < 0) return allPending();

  const current: StepStatus =
    job.state === "running" ? "active" : job.state === "refused" ? "refused" : "failed";
  return PIPELINE_STEPS.map((_, i) => (i < idx ? "done" : i === idx ? current : "pending"));
}
