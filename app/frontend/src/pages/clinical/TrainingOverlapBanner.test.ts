// Tests for the W6.2 "honest labels" banner: a study that is a patient the
// deployed model trained on must say so, and a study that is not (or an older
// backend that omits the field) must render nothing. Same server-render
// approach as GatekeeperPanel.test.ts (no jsdom in this project).

import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { ClinicalJob, TrainingOverlap } from "../../api";
import { TrainingOverlapBanner } from "./TrainingOverlapBanner";

const OVERLAP: TrainingOverlap = {
  source_id: "UPENN-GBM-00002",
  brats_id: "BraTS2021_01202",
  split: "train",
  status: "confirmed",
  evidence: "experiments.md note 50 (series UIDs match)",
  matched_series: ["1.3.6.1.4.1.1"],
};

function job(extra: Partial<ClinicalJob>): ClinicalJob {
  return {
    job_id: "job-1",
    state: "done",
    stage: "done",
    progress: 1,
    case_id: "case-1",
    error: null,
    ingest_result: null,
    input_qc_pre: null,
    input_qc_post: null,
    preprocess_warnings: null,
    gatekeeper_decision: null,
    created_at: 0,
    updated_at: 0,
    ...extra,
  };
}

function render(j: ClinicalJob): string {
  return renderToStaticMarkup(React.createElement(TrainingOverlapBanner, { job: j }));
}

describe("TrainingOverlapBanner", () => {
  it("renders nothing when training_overlap is null", () => {
    expect(render(job({ training_overlap: null }))).toBe("");
  });

  it("renders nothing when the field is absent (older backend)", () => {
    expect(render(job({}))).toBe("");
  });

  it("states the headline, the BraTS id, the source id and the evidence for a confirmed match", () => {
    const html = render(job({ training_overlap: OVERLAP }));
    expect(html).toContain("Training-set patient — illustration only");
    expect(html).toContain('data-testid="training-overlap-banner"');
    expect(html).toContain('role="note"');
    expect(html).toContain("This study is BraTS2021_01202 in the model&#x27;s training data (UPENN-GBM-00002).");
    expect(html).toContain("not evidence of accuracy on unseen patients");
    expect(html).toContain("Source: experiments.md note 50 (series UIDs match)");
    expect(html).not.toContain("probably");
  });

  it("says 'probably' for a probable match", () => {
    const html = render(job({ training_overlap: { ...OVERLAP, status: "probable" } }));
    expect(html).toContain("This study is probably BraTS2021_01202 in the model&#x27;s training data");
  });

  it("also renders for a refused job", () => {
    const html = render(job({ state: "refused", training_overlap: OVERLAP }));
    expect(html).toContain("Training-set patient — illustration only");
  });
});
