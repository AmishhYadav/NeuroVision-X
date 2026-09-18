// Tests for selectStructures/structureName/structureRow (see atlasSelection.ts).

import { describe, expect, it } from "vitest";
import type { AtlasStructureRow } from "../api";
import {
  MAX_ATLAS_STRUCTURES,
  selectStructures,
  structureIndexForName,
  structureName,
  structureRow,
} from "./atlasSelection";

function makeTable(): AtlasStructureRow[] {
  return [
    { index: 1, name: "Caudate_L", laterality: "L", lobe: "deep", eloquence: "eloquent", matched_term: "basal ganglia" },
    { index: 2, name: "Caudate_R", laterality: "R", lobe: "deep", eloquence: "eloquent", matched_term: "basal ganglia" },
    { index: 3, name: "Precentral_L", laterality: "L", lobe: "frontal", eloquence: "eloquent", matched_term: "motor cortex" },
    { index: 4, name: "Thalamus_L", laterality: "L", lobe: "deep", eloquence: "eloquent", matched_term: null },
  ];
}

function makeReport(names: string[]) {
  return { anatomy: { structures: names.map((structure) => ({ structure })) } };
}

describe("selectStructures", () => {
  it("returns [] when report is null", () => {
    expect(selectStructures({ report: null, table: makeTable() })).toEqual([]);
  });

  it("returns [] when table is null", () => {
    expect(selectStructures({ report: makeReport(["Caudate_L"]), table: null })).toEqual([]);
  });

  it("maps report structures to atlas indices, preserving the order received", () => {
    const result = selectStructures({
      report: makeReport(["Precentral_L", "Caudate_L", "Thalamus_L"]),
      table: makeTable(),
    });
    // Order must follow the report's own order (already sorted server-side
    // by frac_of_structure), not the table's index order.
    expect(result).toEqual([3, 1, 4]);
  });

  it("silently skips a structure name absent from the table (atlas/knowledge version mismatch)", () => {
    const result = selectStructures({
      report: makeReport(["Caudate_L", "Unknown_Structure_Z", "Caudate_R"]),
      table: makeTable(),
    });
    expect(result).toEqual([1, 2]);
  });

  it("appends extra indices not already present, after the report-derived ones", () => {
    const result = selectStructures({
      report: makeReport(["Caudate_L"]),
      table: makeTable(),
      extra: [4, 2],
    });
    expect(result).toEqual([1, 4, 2]);
  });

  it("de-duplicates an extra index already produced by the report, preserving first-seen order", () => {
    const result = selectStructures({
      report: makeReport(["Caudate_L", "Caudate_R"]),
      table: makeTable(),
      extra: [1, 4],
    });
    expect(result).toEqual([1, 2, 4]);
  });

  it("drops non-positive extra indices - 0 is the atlas background value, never a real structure", () => {
    const result = selectStructures({
      report: makeReport([]),
      table: makeTable(),
      extra: [0, -1, 2],
    });
    expect(result).toEqual([2]);
  });

  it("caps the result at MAX_ATLAS_STRUCTURES", () => {
    const bigTable: AtlasStructureRow[] = Array.from({ length: 20 }, (_, i) => ({
      index: i + 1,
      name: `Structure_${i + 1}`,
      laterality: null,
      lobe: null,
      eloquence: null,
      matched_term: null,
    }));
    const report = makeReport(bigTable.map((row) => row.name));
    const result = selectStructures({ report, table: bigTable });
    expect(result).toHaveLength(MAX_ATLAS_STRUCTURES);
    expect(result).toEqual(bigTable.slice(0, MAX_ATLAS_STRUCTURES).map((row) => row.index));
  });

  it("caps after merging report-derived and extra indices, not before", () => {
    const bigTable: AtlasStructureRow[] = Array.from({ length: 20 }, (_, i) => ({
      index: i + 1,
      name: `Structure_${i + 1}`,
      laterality: null,
      lobe: null,
      eloquence: null,
      matched_term: null,
    }));
    // 15 from the report, 5 more via extra -> 20 total candidates, capped to 16.
    const report = makeReport(bigTable.slice(0, 15).map((row) => row.name));
    const result = selectStructures({
      report,
      table: bigTable,
      extra: [16, 17, 18, 19, 20],
    });
    expect(result).toHaveLength(MAX_ATLAS_STRUCTURES);
    expect(result[0]).toBe(1);
    expect(result[MAX_ATLAS_STRUCTURES - 1]).toBe(16);
  });

  it("never returns index 0 even if a (malformed) table row claims it", () => {
    const table: AtlasStructureRow[] = [
      { index: 0, name: "Background", laterality: null, lobe: null, eloquence: null, matched_term: null },
      { index: 1, name: "Caudate_L", laterality: "L", lobe: "deep", eloquence: "eloquent", matched_term: null },
    ];
    const result = selectStructures({ report: makeReport(["Background", "Caudate_L"]), table });
    // "Background" resolves to index 0 via the table, and the report path
    // has no `> 0` filter (unlike `extra`) - so this documents that a real
    // atlas table never contains a 0-indexed row for the client to look up;
    // a caller must not construct one. Given a correct table, 0 never appears.
    expect(result).not.toContain(0);
    expect(result).toEqual([1]);
  });
});

describe("structureName", () => {
  it("returns the name for a known index", () => {
    expect(structureName(makeTable(), 3)).toBe("Precentral_L");
  });

  it("returns null when table is null", () => {
    expect(structureName(null, 1)).toBeNull();
  });

  it("returns null for an index absent from the table", () => {
    expect(structureName(makeTable(), 999)).toBeNull();
  });
});

describe("structureRow", () => {
  it("returns the full row for a known index", () => {
    expect(structureRow(makeTable(), 2)).toEqual({
      index: 2,
      name: "Caudate_R",
      laterality: "R",
      lobe: "deep",
      eloquence: "eloquent",
      matched_term: "basal ganglia",
    });
  });

  it("returns null when table is null", () => {
    expect(structureRow(null, 1)).toBeNull();
  });

  it("returns null for an index absent from the table", () => {
    expect(structureRow(makeTable(), 999)).toBeNull();
  });
});

describe("structureIndexForName", () => {
  it("returns the index for a known name", () => {
    expect(structureIndexForName(makeTable(), "Precentral_L")).toBe(3);
  });

  it("returns null when table is null", () => {
    expect(structureIndexForName(null, "Precentral_L")).toBeNull();
  });

  it("returns null for a name absent from the table", () => {
    expect(structureIndexForName(makeTable(), "Unknown_Structure_Z")).toBeNull();
  });
});
