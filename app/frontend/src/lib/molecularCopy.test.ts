// Tests for the panel's two self-authored strings (see molecularCopy.ts):
// requiresHint's construction, and the forbidden-word scan the house rule
// (CLAUDE.md) requires of any copy this frontend writes itself, rather than
// rendering verbatim off the server's molecular block.

import { describe, expect, it } from "vitest";
import { NO_AI_ESTIMATE_TEXT, requiresHint } from "./molecularCopy";

// Same forbidden-word list the spec names, case-insensitive substring match.
const FORBIDDEN = /\bgrade\b|\bstage\b|prognos|deficit|impair|will experience|malignan|aggressiv/i;

describe("requiresHint", () => {
  it("builds the expected hint for a list of still-needed keys", () => {
    expect(requiresHint(["IDH", "histology"])).toBe(
      "No integrated classification yet — needs: IDH, histology",
    );
  });

  it("returns the empty string when nothing is still needed", () => {
    expect(requiresHint([])).toBe("");
  });
});

describe("forbidden-word discipline", () => {
  it("NO_AI_ESTIMATE_TEXT contains none of the forbidden words", () => {
    expect(NO_AI_ESTIMATE_TEXT).not.toMatch(FORBIDDEN);
  });

  it("requiresHint's output contains none of the forbidden words", () => {
    expect(requiresHint(["IDH", "1p/19q", "MGMT", "histology"])).not.toMatch(FORBIDDEN);
  });
});
