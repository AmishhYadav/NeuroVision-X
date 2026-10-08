import { describe, expect, it } from "vitest";
import { dataForSelection } from "./caseConsistency";

const st = (caseId: string | null, detailId: string | null) => ({
  caseId,
  detail: detailId === null ? null : { meta: { case_id: detailId } },
});

describe("dataForSelection", () => {
  it("null selection never matches; loaded data is stale", () => {
    expect(dataForSelection(null, st(null, null))).toEqual({ matches: false, stale: false });
    expect(dataForSelection(null, st("A", "A"))).toEqual({ matches: false, stale: true });
  });
  it("matches when state id and detail id both equal the selection", () => {
    expect(dataForSelection("A", st("A", "A"))).toEqual({ matches: true, stale: false });
  });
  it("previous case's data is stale for the new selection", () => {
    expect(dataForSelection("B", st("A", "A"))).toEqual({ matches: false, stale: true });
  });
  it("nothing loaded yet is neither matching nor stale", () => {
    expect(dataForSelection("B", st(null, null))).toEqual({ matches: false, stale: false });
  });
  it("detail/caseId disagreement does not match", () => {
    expect(dataForSelection("B", st("B", "A")).matches).toBe(false);
    expect(dataForSelection("B", st("A", "B")).matches).toBe(false);
    expect(dataForSelection("B", st("B", "A")).stale).toBe(true);
  });
});
