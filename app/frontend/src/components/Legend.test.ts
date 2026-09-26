// Tests for the conformal-band caveat text in Legend.tsx. This overlay is a
// distribution-free *average-case* guarantee over in-distribution studies
// (deployed alpha = 0.10), never a per-patient one, and it is measured to
// fail under distribution shift (SSA, paediatric cohorts) - see
// docs/experiments.md and CLAUDE.md's "external validation on BraTS-Africa
// came back negative" note. The short on-screen line must not overclaim, and
// the full caveat must still be reachable via the native `title` tooltip.
//
// No React renderer (testing-library, jsdom) is set up in this project's
// vitest config (see vitest.config.ts's comment - the "node" environment was
// chosen deliberately to avoid that weight), so this renders with
// `react-dom/server`'s `renderToStaticMarkup`, which needs no DOM at all, and
// asserts on the resulting HTML string. `.ts`, not `.tsx`, and
// `React.createElement` rather than JSX, so this file needs no JSX transform
// and stays covered by tsconfig.app.json's `src/**/*.test.ts` exclude (test
// files are intentionally left out of `tsc -b`/`npm run build`'s typecheck).

import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { CONFORMAL_BAND } from "../api";
import { Legend } from "./Legend";

const CAVEAT_TOOLTIP =
  "Calibrated so that, averaged over in-distribution studies, mask + band miss at most 10% of tumour voxels (α = 0.10). Not a guarantee for this patient, and it does not hold for scans unlike the training data.";

function renderConformalBandLegend(): string {
  return renderToStaticMarkup(
    React.createElement(Legend, {
      overlayMode: "prediction",
      showUncertainty: true,
      hasLabel: false,
      uncertaintyKind: CONFORMAL_BAND,
    }),
  );
}

describe("Legend conformal band caveat", () => {
  it("shows the average-case, in-distribution, not-per-patient line", () => {
    const html = renderConformalBandLegend();
    expect(html).toContain(
      "Conformal band: average-case bound (in-distribution only, not per patient)",
    );
  });

  it("never claims a bare guarantee in the visible line", () => {
    const html = renderConformalBandLegend();
    expect(html).not.toContain("guaranteed-coverage");
  });

  it("carries the full caveat, including the 10% figure and the shift warning, in a title tooltip", () => {
    const html = renderConformalBandLegend();
    expect(html).toContain(`title="${CAVEAT_TOOLTIP}"`);
  });
});
