// Pure lookup: given a live report and the atlas structure table, find the
// report's row for one atlas structure index, and pull out the two numbers
// the twin's structure-detail card (BrainTwinScene) wants to show. No React,
// no fetch - see structureDetail.test.ts for coverage.

import type { AtlasStructureRow, ReportResponse } from "../api";
import { structureRow } from "./atlasSelection";

export interface StructureReportDetail {
  fracOfStructure: number | null;
  fracOfTumour: number | null;
}

/**
 * Resolves atlas structure `index` to its name via `table`, then finds the
 * matching row in `report.anatomy.structures` by an exact `structure ===
 * name` match - the same exact-match convention `selectStructures` uses to
 * go the other direction (name -> index).
 *
 * Returns `null` when `report` or `table` is null, `index` has no row in
 * `table`, or the report has no row for that name (it fell outside the
 * server's top_n truncation, or an atlas/knowledge version mismatch renamed
 * the structure) - never throws. A `null` result is what tells the caller
 * (BrainTwinScene's detail card) to show name/laterality/lobe/eloquence only,
 * with no overlap numbers.
 */
export function reportRowForStructure(
  report: ReportResponse | null,
  table: AtlasStructureRow[] | null,
  index: number,
): StructureReportDetail | null {
  if (!report || !table) return null;
  const name = structureRow(table, index)?.name;
  if (!name) return null;
  const row = report.anatomy.structures.find((r) => r.structure === name);
  if (!row) return null;
  return { fracOfStructure: row.frac_of_structure, fracOfTumour: row.frac_of_tumour };
}
