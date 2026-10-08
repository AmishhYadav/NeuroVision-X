/**
 * The loaded data only counts as "the selected case" when it is literally that
 * case's data. `useCaseData` keeps showing the previous case while the next
 * one loads, so the selected id alone cannot be trusted to label what is in
 * memory: pair it with the id the data itself carries.
 */
export function dataForSelection<
  T extends { caseId: string | null; detail: { meta: { case_id: string } } | null },
>(selectedCaseId: string | null, state: T): { matches: boolean; stale: boolean } {
  const matches =
    selectedCaseId !== null &&
    state.caseId === selectedCaseId &&
    state.detail?.meta.case_id === selectedCaseId;
  return { matches, stale: state.detail !== null && !matches };
}
