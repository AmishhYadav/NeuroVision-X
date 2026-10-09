// Presentation-only helpers for the report page (pages/report). They never
// format a number: every string they place comes verbatim from an
// InterpretedFact (lib/reportInterpretation.ts). They only decide WHERE a
// fact goes - a region table, a ledger row - and how its value is typeset.

import type { InterpretedFact } from "./reportInterpretation";

export type RegionKey = "WT" | "TC" | "ET";

const REGION_ORDER: RegionKey[] = ["WT", "TC", "ET"];

export interface RegionTable {
  /** Column headers, in first-seen order, e.g. ["Sphericity", "Surface area"]. */
  columns: string[];
  /** One row per region present, in WT, TC, ET order; null where a region lacks a column. */
  rows: { region: RegionKey; cells: (string | null)[] }[];
}

const REGION_FACT = /^(WT|TC|ET) (.+)$/;

/** Splits facts labelled "<WT|TC|ET> <measure>" (e.g. "TC sphericity") into
 * a region x measure table, and returns every other fact untouched, in order.
 * A table needs at least two such facts across at least two regions;
 * otherwise `table` is null and every fact stays in `rest`. */
export function regionTable(facts: readonly InterpretedFact[]): {
  table: RegionTable | null;
  rest: InterpretedFact[];
} {
  const columns: string[] = [];
  const cells = new Map<RegionKey, Map<string, string>>();
  const rest: InterpretedFact[] = [];

  for (const f of facts) {
    const m = REGION_FACT.exec(f.label);
    if (!m) {
      rest.push(f);
      continue;
    }
    const region = m[1] as RegionKey;
    const measure = m[2].charAt(0).toUpperCase() + m[2].slice(1);
    if (!columns.includes(measure)) columns.push(measure);
    if (!cells.has(region)) cells.set(region, new Map());
    cells.get(region)!.set(measure, f.value);
  }

  const matched = [...cells.values()].reduce((n, row) => n + row.size, 0);
  if (matched < 2 || cells.size < 2) return { table: null, rest: [...facts] };

  const rows = REGION_ORDER.filter((r) => cells.has(r)).map((region) => ({
    region,
    cells: columns.map((c) => cells.get(region)!.get(c) ?? null),
  }));
  return { table: { columns, rows }, rest };
}

/** True when a displayed value is a figure ("143.8 mL", "−2.0 mm", "100.0%")
 * rather than a word ("left", "yes", "Sawaya eloquence grading"). Figures are
 * set in the mono face so columns line up; words stay in the text face. */
export function isFigure(value: string): boolean {
  return /^[+\-−]?\d/.test(value.trim());
}

/** Splits text on `backtick` spans so a status message can render its
 * commands as <code>. Even indices are plain text, odd indices are code. */
export function splitCode(text: string): string[] {
  return text.split("`");
}
