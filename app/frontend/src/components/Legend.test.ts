// Tests for the conformal-band caveat text in Legend.tsx. This overlay is a
// distribution-free *average-case* guarantee over in-distribution studies
// (deployed alpha = 0.10), never a per-patient one, and it is measured to
// fail under distribution shift (SSA, paediatric cohorts) - see
// docs/research_docs/experiments.md and CLAUDE.md's "external validation on BraTS-Africa
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
import { CONFORMAL_BAND, type ConformalMeta } from "../api";
import { Legend } from "./Legend";

const CAVEAT_TOOLTIP =
  "Calibrated so that, averaged over in-distribution studies, mask + band miss at most 10% of tumour voxels (α = 0.10). Not a guarantee for this patient, and it does not hold for scans unlike the training data.";

function renderConformalBandLegend(conformal?: ConformalMeta | null): string {
  return renderToStaticMarkup(
    React.createElement(Legend, {
      overlayMode: "prediction",
      showUncertainty: true,
      hasLabel: false,
      uncertaintyKind: CONFORMAL_BAND,
      conformal,
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

describe("Legend conformal band labels by side", () => {
  const meta = (side: ConformalMeta["side"], threshold = 0.725): ConformalMeta => ({
    threshold,
    reference: 0.5,
    side,
    alpha: 0.1,
  });

  it("restrictive: 128 is point estimate outside the conformal set", () => {
    const html = renderConformalBandLegend(meta("restrictive"));
    expect(html).toContain("Point estimate and conformal set");
    expect(html).toContain("Point estimate only (outside the conformal set)");
    expect(html).not.toContain("Safety margin");
    expect(html).toContain("fitted 0.725 vs 0.5 · α = 0.1");
  });

  it("permissive: 128 is a margin beyond the point estimate", () => {
    const html = renderConformalBandLegend(meta("permissive", 0.3));
    expect(html).toContain("Point estimate and conformal set");
    expect(html).toContain("Conformal set only (margin beyond the point estimate)");
    expect(html).toContain("fitted 0.300 vs 0.5");
  });

  it("unknown or absent side: neutral labels", () => {
    for (const c of [undefined, null, meta(null)]) {
      const html = renderConformalBandLegend(c);
      expect(html).toContain("In both masks");
      expect(html).toContain("In one mask only (side unknown)");
    }
  });
});
