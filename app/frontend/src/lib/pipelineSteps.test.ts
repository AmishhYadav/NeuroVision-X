import { describe, expect, it } from "vitest";
import { PIPELINE_STEPS, stepIndexForStage, stepStatuses } from "./pipelineSteps";

describe("stepIndexForStage", () => {
  it("maps every backend stage string to its step", () => {
    const expected: [string, number][] = [
      ["ingest", 0],
      ["input_qc (pre-preprocessing)", 1],
      ["clinical_preprocessing", 2],
      ["input_qc (post-preprocessing)", 3],
      ["research_preprocessing", 4],
      ["segmenting", 5],
      ["gatekeeper", 6],
      ["explaining", 7],
      ["exporting_dicom_seg", 7],
      ["generating_report", 7],
    ];
    for (const [stage, idx] of expected) expect(stepIndexForStage(stage)).toBe(idx);
    expect(PIPELINE_STEPS).toHaveLength(8);
  });
  it("matches segmenting substages by prefix only", () => {
    expect(stepIndexForStage("segmenting: sliding window")).toBe(5);
    expect(stepIndexForStage("segmentingX")).toBe(-1);
  });
  it("returns -1 for queued, done and unknown", () => {
    for (const s of ["queued", "done", "", "nonsense"]) expect(stepIndexForStage(s)).toBe(-1);
  });
});

describe("stepStatuses", () => {
  it("is all pending for null or queued", () => {
    expect(stepStatuses(null).every((s) => s === "pending")).toBe(true);
    expect(stepStatuses({ state: "queued", stage: "queued" }).every((s) => s === "pending")).toBe(true);
  });
  it("marks earlier steps done and the current one active while running", () => {
    const s = stepStatuses({ state: "running", stage: "segmenting: tta" });
    expect(s).toEqual(["done", "done", "done", "done", "done", "active", "pending", "pending"]);
  });
  it("is all pending for a running job with an unknown stage", () => {
    expect(stepStatuses({ state: "running", stage: "???" }).every((s) => s === "pending")).toBe(true);
  });
  it("is all done when done", () => {
    expect(stepStatuses({ state: "done", stage: "done" }).every((s) => s === "done")).toBe(true);
  });
  it("marks the refusing step refused: at ingest and at the gatekeeper", () => {
    expect(stepStatuses({ state: "refused", stage: "ingest" })).toEqual([
      "refused", "pending", "pending", "pending", "pending", "pending", "pending", "pending",
    ]);
    expect(stepStatuses({ state: "refused", stage: "gatekeeper" })).toEqual([
      "done", "done", "done", "done", "done", "done", "refused", "pending",
    ]);
  });
  it("marks the failing step failed mid-run", () => {
    expect(stepStatuses({ state: "failed", stage: "clinical_preprocessing" })).toEqual([
      "done", "done", "failed", "pending", "pending", "pending", "pending", "pending",
    ]);
  });
  it("is all pending for a refused job with an unknown stage", () => {
    expect(stepStatuses({ state: "refused", stage: "weird" }).every((s) => s === "pending")).toBe(true);
  });
});
