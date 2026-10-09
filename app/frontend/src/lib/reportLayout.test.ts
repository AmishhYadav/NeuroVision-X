// Checks for the report page's presentation helpers (reportLayout.ts).

import { describe, expect, it } from "vitest";
import { isFigure, regionTable, splitCode } from "./reportLayout";

const f = (label: string, value: string) => ({ label, value });

describe("regionTable", () => {
  it("builds a region x measure table and keeps the other facts in order", () => {
    const { table, rest } = regionTable([
      f("WT sphericity", "0.71"),
      f("TC sphericity", "0.87"),
      f("Elongation", "1.2"),
      f("ET sphericity", "0.38"),
      f("WT surface area", "186.1 cm²"),
    ]);
    expect(table).toEqual({
      columns: ["Sphericity", "Surface area"],
      rows: [
        { region: "WT", cells: ["0.71", "186.1 cm²"] },
        { region: "TC", cells: ["0.87", null] },
        { region: "ET", cells: ["0.38", null] },
      ],
    });
    expect(rest).toEqual([f("Elongation", "1.2")]);
  });

  it("orders rows WT, TC, ET whatever order the facts arrive in", () => {
    const { table } = regionTable([f("ET components", "1"), f("WT components", "2")]);
    expect(table?.rows.map((r) => r.region)).toEqual(["WT", "ET"]);
  });

  it("does not build a table from a single region", () => {
    const facts = [f("WT components", "1"), f("WT largest component volume", "143.8 mL")];
    const { table, rest } = regionTable(facts);
    expect(table).toBeNull();
    expect(rest).toEqual(facts);
  });

  it("does not match a label that merely contains a region code", () => {
    const facts = [f("Oedema fraction of WT", "57.0%"), f("Enhancing fraction of TC", "66.3%")];
    expect(regionTable(facts).table).toBeNull();
  });

  it("does not treat a longer word starting with a region code as a region", () => {
    // "WTX" or "ETA" must not match: the code has to be followed by a space.
    expect(regionTable([f("ETA estimate", "1"), f("WTX value", "2")]).table).toBeNull();
  });
});

describe("isFigure", () => {
  it("accepts numbers with units and signs", () => {
    for (const v of ["143.8 mL", "100.0%", "0.71", "-2.0 mm", "−2.0 mm", "+0.03"]) {
      expect(isFigure(v)).toBe(true);
    }
  });
  it("rejects words and placeholders", () => {
    for (const v of ["left", "yes (10.0 mm)", "—", "Sawaya eloquence grading"]) {
      expect(isFigure(v)).toBe(false);
    }
  });
});

describe("splitCode", () => {
  it("puts code spans at odd indices", () => {
    expect(splitCode("run `scripts/report.py` first")).toEqual(["run ", "scripts/report.py", " first"]);
  });
});
