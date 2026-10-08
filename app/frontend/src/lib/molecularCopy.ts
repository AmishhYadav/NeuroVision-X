// The two strings `MolecularPanel` authors itself, rather than rendering
// verbatim off `ReportMolecular` (see `api.ts`). Pulled into their own pure
// module, same reasoning as `reportStatus.ts`: a module this small still
// gets its own test, and the forbidden-word scan lives here once rather than
// duplicated at every call site.
//
// Copy discipline (see CLAUDE.md and `knowledge/molecular_markers.yaml`'s own
// header): neither string may contain "grade", "stage", "prognosis",
// "deficit", "impair", "will experience", "malignant"/"malignancy", or
// "aggressive" - `molecularCopy.test.ts` scans both against the same regex
// `tests/test_molecular.py` and `tests/test_report.py` already use
// server-side.

/**
 * Shown in place of the AI slot for a marker whose `ai_estimate` is `null` -
 * i.e. no model was ever trained to predict it (every marker except IDH; see
 * `neurovision.reporting.molecular.empty_molecular_block`'s docstring). IDH's
 * slot is never `null`, so this string never applies to it - IDH instead
 * renders `ai_estimate.status` verbatim from the server.
 */
export const NO_AI_ESTIMATE_TEXT = "No AI estimate — never trained";

/**
 * Builds the dim hint shown under the CNS5 line while `cns5.name` is still
 * `null` and `cns5.requires` is non-empty - a factual statement about which
 * entries are still missing, not a diagnosis or a claim about what they will
 * show. `requires` is rendered verbatim (marker names / `"histology"`, in
 * the server's own relevance order - see `cns5_lookup`'s docstring), never
 * reordered or relabelled here.
 *
 * Args:
 *   requires: The `cns5.requires` array from a `ReportMolecular` block.
 *
 * Returns:
 *   `""` when `requires` is empty (the caller should render nothing in that
 *   case - see `MolecularPanel`), otherwise the full hint sentence.
 */
export function requiresHint(requires: string[]): string {
  if (requires.length === 0) return "";
  return `No integrated classification yet — needs: ${requires.join(", ")}`;
}
