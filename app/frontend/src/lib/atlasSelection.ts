// Pure selection logic for which atlas structures get a 3D shell in the
// digital twin (T3.4 builds one Surface-Nets mesh per selected structure).
// No React, no fetch, no worker - this module only decides WHICH indices to
// mesh, from the report already on hand and the atlas structure table
// already on hand. See atlasSelection.test.ts for coverage.

import type { AtlasStructureRow } from "../api";

/**
 * Hard cap on how many structures ever get their own mesh in the twin.
 *
 * The worker builds one Surface-Nets mesh per structure (T3.4) - meshing
 * all 122 atlas structures at once would be both visually useless (the
 * report already limits itself to the top 10 by involvement) and slow, so
 * this cap is enforced here, once, rather than trusted to every caller.
 */
export const MAX_ATLAS_STRUCTURES = 16;

/**
 * What `selectStructures` needs. `report` and `table` mirror exactly the
 * shapes `useClinicalJobVolumes`-adjacent code already has in hand - the
 * live report's `anatomy.structures` rows (only the `structure` name field
 * is used here) and the atlas's own structure table (`getAtlasStructures`).
 * `report` is typed as a narrowed subset of `ReportResponse`, not the full
 * type, so a caller can pass a partial/mock report in tests without having
 * to construct an entire valid one.
 */
export interface AtlasSelectionInput {
  report: { anatomy: { structures: { structure: string }[] } } | null;
  table: AtlasStructureRow[] | null;
  /**
   * User-added structure indices (T3.5's checkboxes), appended after the
   * report-derived ones. Only positive indices are honoured - 0 is the
   * atlas's background value and can never name a real structure.
   */
  extra?: number[];
}

/**
 * Picks which atlas structure indices get meshed in the twin, in display
 * order.
 *
 * 1. Start from `report.anatomy.structures`, taken in the order the server
 *    sent them (already sorted by `frac_of_structure` descending and
 *    truncated server-side to `top_n` = 10 - this function does not re-sort
 *    or re-truncate that part).
 * 2. Map each row's `structure` name to its atlas `index` via an exact
 *    string match against `table`'s `name` field. A name with no match in
 *    the table (an atlas/knowledge version mismatch - the report was built
 *    against a different atlas version than the one now loaded) is silently
 *    dropped rather than thrown: a stale label is not a reason to break the
 *    whole selection.
 * 3. Append `extra` (user-added indices), skipping any already present and
 *    any `<= 0`.
 * 4. De-duplicate, preserving first-seen order.
 * 5. Cap the result at `MAX_ATLAS_STRUCTURES`.
 *
 * Returns `[]` when `report` or `table` is `null` (nothing loaded yet, or a
 * job/case with no report) - never throws.
 */
export function selectStructures(input: AtlasSelectionInput): number[] {
  const { report, table, extra = [] } = input;
  if (!report || !table) return [];

  // Exact-match name -> index lookup, built once per call rather than
  // scanning `table` per structure - `table` has up to 122 rows and this
  // runs on every report/table change, not just once.
  const indexByName = new Map<string, number>();
  for (const row of table) {
    indexByName.set(row.name, row.index);
  }

  const ordered: number[] = [];
  const seen = new Set<number>();

  // 0 is the atlas's background value, never a real structure - guarded
  // here, once, so neither the report-derived nor the `extra` path can ever
  // emit it, regardless of what a (malformed) table or caller supplies.
  const pushIfNew = (index: number) => {
    if (index <= 0 || seen.has(index)) return;
    seen.add(index);
    ordered.push(index);
  };

  for (const structure of report.anatomy.structures) {
    const index = indexByName.get(structure.structure);
    if (index === undefined) continue; // Version mismatch - skip, don't throw.
    pushIfNew(index);
  }

  for (const index of extra) {
    pushIfNew(index);
  }

  return ordered.slice(0, MAX_ATLAS_STRUCTURES);
}

/** The atlas structure name for `index`, or `null` if `table` is null/has no such index. */
export function structureName(table: AtlasStructureRow[] | null, index: number): string | null {
  return structureRow(table, index)?.name ?? null;
}

/**
 * The atlas index for structure `name` (exact match), or `null` if `table`
 * is null or has no row with that name. Inverse of `structureName` - used to
 * turn a hovered/clicked `ReportPanel` row (which only carries a name) back
 * into the index `BrainTwinScene`'s `highlightedStructure` prop expects.
 */
export function structureIndexForName(table: AtlasStructureRow[] | null, name: string): number | null {
  if (!table) return null;
  return table.find((row) => row.name === name)?.index ?? null;
}

/**
 * The full atlas structure table row for `index`, or `null` if `table` is
 * null/has no such index. A linear scan - `table` has at most ~122 rows
 * (one full atlas), so this is not worth indexing.
 */
export function structureRow(
  table: AtlasStructureRow[] | null,
  index: number,
): AtlasStructureRow | null {
  if (!table) return null;
  return table.find((row) => row.index === index) ?? null;
}
