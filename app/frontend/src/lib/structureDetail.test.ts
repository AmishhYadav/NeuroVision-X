// Tests for reportRowForStructure (see structureDetail.ts).

import { describe, expect, it } from "vitest";
import type { AtlasStructureRow, ReportResponse } from "../api";
import { reportRowForStructure } from "./structureDetail";

function makeTable(): AtlasStructureRow[] {
  return [
    { index: 1, name: "Caudate_L", laterality: "L", lobe: "deep", eloquence: "eloquent", matched_term: "basal ganglia" },
    { index: 2, name: "Precentral_L", laterality: "L", lobe: "frontal", eloquence: "eloquent", matched_term: "motor cortex" },
  ];
}

// Only the fields reportRowForStructure actually reads (report.anatomy.structures)
// are filled with real values; everything else is a minimal placeholder so
// this compiles against the full ReportResponse shape.
function makeReport(): ReportResponse {
  return {
    report_version: 1,
    case_id: "case-1",
    generated_utc: "2026-09-18T00:00:00Z",
    disclaimer: "",
    not_claimed: [],
    burden: {
      volumes: {},
      fractions: {},
      shape: {},
      multifocality: {},
      laterality: {},
      centroid: {},
      other: {},
    },
    anatomy: {
      atlas: { name: "tzo116plus", version: "2.0" },
      caveat: "",
      coverage_line: "",
      region: "WT",
      structures: [
        {
          region: "WT",
          structure: "Caudate_L",
          laterality: "L",
          lobe: "deep",
          eloquence: "eloquent",
          matched_term: "basal ganglia",
          n_voxels: 100,
          volume_mm3: 100.0,
          frac_of_tumour: 0.12,
          frac_of_structure: 0.34,
        },
      ],
      n_structures_involved: 1,
      frac_unlabelled: 0.1,
    },
    eloquence: {
      classification: "",
      citation: "",
      evidence: "",
      source_owns_claim: "",
      involved: [],
      distance_mm: null,
      near_eloquent_threshold_mm: 10,
      near_eloquent: false,
      coverage_gaps: [],
    },
    provenance: {
      atlas_name: "tzo116plus",
      atlas_version: "2.0",
      atlas_source: "",
      atlas_licence: "",
      knowledge_versions: {},
      segmentation_source: "prediction",
      segmentation_dir: null,
      code_revision: null,
      generated_utc: "2026-09-18T00:00:00Z",
    },
  };
}

describe("reportRowForStructure", () => {
  it("found: returns the fractions for a structure the report and table both have", () => {
    const result = reportRowForStructure(makeReport(), makeTable(), 1);
    expect(result).toEqual({ fracOfStructure: 0.34, fracOfTumour: 0.12 });
  });

  it("not found: returns null when the index is in the table but not in the report", () => {
    // index 2 ("Precentral_L") is in the table but the report only has "Caudate_L".
    const result = reportRowForStructure(makeReport(), makeTable(), 2);
    expect(result).toBeNull();
  });

  it("not found: returns null for an index absent from the table entirely", () => {
    const result = reportRowForStructure(makeReport(), makeTable(), 999);
    expect(result).toBeNull();
  });

  it("null inputs: returns null when report is null", () => {
    expect(reportRowForStructure(null, makeTable(), 1)).toBeNull();
  });

  it("null inputs: returns null when table is null", () => {
    expect(reportRowForStructure(makeReport(), null, 1)).toBeNull();
  });
});
