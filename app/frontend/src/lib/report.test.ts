// Tests for the Phase 4 report panel's pure logic: formatters,
// burdenLabel's fallback rule, validateReport's required-field guard, and
// segmentationLabel's refusal to guess an unrecognised source. See
// slicing.test.ts / render.test.ts for the sibling pattern this follows -
// all real logic lives here so ReportPanel.tsx stays a thin renderer.

import { describe, expect, it } from "vitest";
import {
  burdenLabel,
  formatBurdenValue,
  formatDistanceMm,
  formatNumber,
  formatPercent,
  formatVolumeMl,
  segmentationLabel,
  validateReport,
} from "./report";
import type { ReportResponse } from "../api";

// A minimal but STRUCTURALLY COMPLETE report, matching the real shape
// produced by neurovision.reporting.report.build_report (verified against
// outputs/report_gt/reports/*.json while writing this panel). Built as a
// literal object rather than read from outputs/, per the task spec.
function makeReport(overrides: Partial<ReportResponse> = {}): ReportResponse {
  const base: ReportResponse = {
    report_version: 1,
    case_id: "BraTS2021_00002",
    generated_utc: "2026-08-18T15:08:03.013692+00:00",
    disclaimer:
      "This report is a research and educational decision-support artifact. It is not a " +
      "diagnostic tool.",
    not_claimed: [
      ["cell type", "MRI resolves millimetre-scale tissue, not individual cells."],
      ["WHO grade", "WHO CNS5 grading needs histology and molecular markers."],
    ],
    burden: {
      volumes: { vol_ET_mm3: 23651.0, vol_WT_mm3: 190594.0 },
      fractions: { frac_enhancing_of_wt: 0.1241, ratio_edema_to_core: 4.4613 },
      shape: { sphericity_ET: 0.3116, surface_area_ET_mm2: 12786.996 },
      multifocality: { n_components_ET: 1, largest_component_frac_ET: 0.9999 },
      laterality: { dominant_side_ET: "left", frac_left_ET: 0.9984 },
      centroid: { centroid_i_ET: 87.1017 },
      other: {},
    },
    anatomy: {
      atlas: { name: "tzo116plus", version: "2.0" },
      caveat:
        "This atlas describes healthy-brain anatomy. A tumour physically displaces the " +
        "tissue around it.",
      coverage_line: "23 of 122 structures classified eloquent, 99 unclassified.",
      region: "WT",
      structures: [
        {
          region: "WT",
          structure: "Caudate_L",
          laterality: "L",
          lobe: "deep",
          eloquence: "eloquent",
          matched_term: "basal ganglia",
          n_voxels: 4864,
          volume_mm3: 4864.0,
          frac_of_tumour: 0.02552,
          frac_of_structure: 0.98501,
        },
      ],
      n_structures_involved: 46,
      frac_unlabelled: 0.30970,
    },
    eloquence: {
      classification: "Sawaya eloquence grading",
      citation: "Sawaya R, et al. Neurosurgery. 1998.",
      evidence: "Eloquent locations in the Sawaya study are the motor/sensory cortices.",
      source_owns_claim:
        "This eloquence classification is a lookup into a named, published source.",
      involved: [
        { structure: "Caudate_L", laterality: "L", frac_of_tumour: 0.02552, frac_of_structure: 0.98501 },
      ],
      distance_mm: 0.0,
      near_eloquent_threshold_mm: 10.0,
      near_eloquent: true,
      coverage_gaps: ["internal capsule", "dentate nucleus"],
    },
    provenance: {
      atlas_name: "tzo116plus",
      atlas_version: "2.0",
      atlas_source: "NITRC group_id=214",
      atlas_licence: "CC-BY-SA",
      knowledge_versions: { eloquence_map: 1, aal_lobes: 1 },
      segmentation_source: "label",
      segmentation_dir: "/data/preprocessed/brats",
      code_revision: "b918b35-dirty",
      generated_utc: "2026-08-18T15:08:03.013692+00:00",
    },
  };
  return { ...base, ...overrides };
}

describe("formatVolumeMl", () => {
  it("converts mm3 to mL at one decimal", () => {
    expect(formatVolumeMl(23651)).toBe("23.7 mL");
    expect(formatVolumeMl(1000)).toBe("1.0 mL");
    expect(formatVolumeMl(0)).toBe("0.0 mL");
  });

  it("renders the missing marker for null, undefined and non-finite input", () => {
    expect(formatVolumeMl(null)).toBe("—");
    expect(formatVolumeMl(undefined)).toBe("—");
    expect(formatVolumeMl(NaN)).toBe("—");
    expect(formatVolumeMl(Infinity)).toBe("—");
  });
});

describe("formatPercent", () => {
  it("converts a fraction to a percentage string", () => {
    expect(formatPercent(0.817)).toBe("81.7%");
    expect(formatPercent(1)).toBe("100.0%");
    expect(formatPercent(0)).toBe("0.0%");
  });

  it("honours a custom digit count", () => {
    expect(formatPercent(0.98501, 2)).toBe("98.50%");
  });

  it("renders the missing marker for null, undefined and non-finite input", () => {
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
    expect(formatPercent(NaN)).toBe("—");
  });
});

describe("formatNumber", () => {
  it("fixes to the requested digit count", () => {
    expect(formatNumber(4.4613, 2)).toBe("4.46");
    expect(formatNumber(1, 0)).toBe("1");
  });

  it("renders the missing marker for null, undefined and non-finite input", () => {
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(undefined)).toBe("—");
    expect(formatNumber(NaN)).toBe("—");
  });
});

describe("formatDistanceMm", () => {
  it("formats a millimetre distance to one decimal with a unit", () => {
    expect(formatDistanceMm(10)).toBe("10.0 mm");
    expect(formatDistanceMm(0)).toBe("0.0 mm");
  });

  it("renders the missing marker for null, undefined and non-finite input", () => {
    expect(formatDistanceMm(null)).toBe("—");
    expect(formatDistanceMm(undefined)).toBe("—");
  });
});

describe("formatBurdenValue", () => {
  it("formats a volume key (_mm3) as mL", () => {
    expect(formatBurdenValue("vol_ET_mm3", 23651)).toBe("23.7 mL");
  });

  it("formats an area key (_mm2) with a mm2 unit", () => {
    expect(formatBurdenValue("surface_area_ET_mm2", 12786.996)).toBe("12787 mm²");
  });

  it("formats a frac_ key as a percentage", () => {
    expect(formatBurdenValue("frac_enhancing_of_wt", 0.1241)).toBe("12.4%");
  });

  it("formats a largest_component_frac_ key as a percentage", () => {
    expect(formatBurdenValue("largest_component_frac_ET", 0.9999)).toBe("100.0%");
  });

  it("formats a ratio_ key to two decimals with no unit", () => {
    expect(formatBurdenValue("ratio_edema_to_core", 4.4613)).toBe("4.46");
  });

  it("formats an n_components_ key as a bare integer", () => {
    expect(formatBurdenValue("n_components_ET", 1)).toBe("1");
  });

  it("formats a centroid_ key to one decimal", () => {
    expect(formatBurdenValue("centroid_i_ET", 87.1017)).toBe("87.1");
  });

  it("renders a dominant_side_ (string) key verbatim", () => {
    expect(formatBurdenValue("dominant_side_ET", "left")).toBe("left");
  });

  it("falls back to three decimals for an unrecognised numeric shape", () => {
    expect(formatBurdenValue("sphericity_ET", 0.3116127822951931)).toBe("0.312");
  });

  it("renders the missing marker for null and undefined", () => {
    expect(formatBurdenValue("vol_ET_mm3", null)).toBe("—");
    expect(formatBurdenValue("vol_second_component_ET_mm3", undefined)).toBe("—");
  });

  it("renders a boolean verbatim", () => {
    expect(formatBurdenValue("some_flag", true)).toBe("true");
  });
});

describe("burdenLabel", () => {
  it("labels a known volume key", () => {
    expect(burdenLabel("vol_ET_mm3")).toBe("Enhancing tumour");
  });

  it("labels a known fraction key", () => {
    expect(burdenLabel("frac_edema_of_wt")).toBe("Oedema fraction of WT");
  });

  it("labels a regionally-generated multifocality key", () => {
    expect(burdenLabel("n_components_WT")).toBe("WT components");
  });

  it("falls back to the raw key, unchanged, for an unknown key", () => {
    expect(burdenLabel("some_future_metric_nobody_labelled_yet")).toBe(
      "some_future_metric_nobody_labelled_yet",
    );
  });
});

describe("validateReport", () => {
  it("accepts a structurally complete report", () => {
    expect(() => validateReport(makeReport())).not.toThrow();
  });

  it("accepts a report with optional values missing (nulls, empty lists)", () => {
    const report = makeReport();
    report.anatomy.structures = [];
    report.anatomy.n_structures_involved = null;
    report.anatomy.frac_unlabelled = null;
    report.eloquence.distance_mm = null;
    report.eloquence.involved = [];
    report.eloquence.coverage_gaps = [];
    report.provenance.segmentation_dir = null;
    report.provenance.code_revision = null;
    expect(() => validateReport(report)).not.toThrow();
  });

  it("throws when disclaimer is missing", () => {
    const report = makeReport() as unknown as Record<string, unknown>;
    delete report.disclaimer;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when not_claimed is missing", () => {
    const report = makeReport() as unknown as Record<string, unknown>;
    delete report.not_claimed;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when not_claimed is an empty array", () => {
    const report = makeReport({ not_claimed: [] });
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when anatomy.caveat is missing", () => {
    const report = makeReport();
    const anatomy = report.anatomy as unknown as Record<string, unknown>;
    delete anatomy.caveat;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when eloquence.evidence is missing", () => {
    const report = makeReport();
    const eloquence = report.eloquence as unknown as Record<string, unknown>;
    delete eloquence.evidence;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when eloquence.citation is missing", () => {
    const report = makeReport();
    const eloquence = report.eloquence as unknown as Record<string, unknown>;
    delete eloquence.citation;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws when provenance is missing", () => {
    const report = makeReport() as unknown as Record<string, unknown>;
    delete report.provenance;
    expect(() => validateReport(report)).toThrow();
  });

  it("throws on a non-object input", () => {
    expect(() => validateReport(null)).toThrow();
    expect(() => validateReport("not a report")).toThrow();
    expect(() => validateReport(42)).toThrow();
  });
});

describe("segmentationLabel", () => {
  it("labels a prediction-sourced report", () => {
    const label = segmentationLabel({ segmentation_source: "prediction" });
    expect(label.text).toBe("Model prediction");
    expect(label.tone).toBe("prediction");
  });

  it("labels a label-sourced (ground truth) report", () => {
    const label = segmentationLabel({ segmentation_source: "label" });
    expect(label.text).toBe("Ground truth");
    expect(label.tone).toBe("truth");
  });

  it("throws on an unrecognised segmentation_source rather than guessing", () => {
    // @ts-expect-error - deliberately passing a value outside the union to
    // exercise the runtime guard a malformed or future server response hits.
    expect(() => segmentationLabel({ segmentation_source: "simulation" })).toThrow();
  });
});
