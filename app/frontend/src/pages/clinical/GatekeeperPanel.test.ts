// Tests for GatekeeperPanel's rendering of a *disabled* signal verdict
// (backend `enabled: false` - see neurovision.inference.gatekeeper.
// GateDecision.to_dict). A disabled signal was computed and is shown for
// information only; the refusal decision never consulted it (2026-09-26,
// conformal_band was removed from configs/clinical/default.yaml's
// `gatekeeper.enabled_signals`). It must never render like an error, a
// missing value, or a step towards refusal - and it must be visually
// distinct from an enabled signal, not just differently worded.
//
// Rendered via `react-dom/server`'s `renderToStaticMarkup` (no DOM needed),
// same approach as Legend.test.ts - see that file's header comment for why
// there is no testing-library/jsdom here. `.ts`, not `.tsx`, so this stays
// covered by tsconfig.app.json's `src/**/*.test.ts` exclude from `tsc -b`.

import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { ClinicalJob, GatekeeperDecisionJson, GatekeeperSignalVerdict } from "../../api";
import { GatekeeperPanel } from "./GatekeeperPanel";

function verdict(overrides: Partial<GatekeeperSignalVerdict>): GatekeeperSignalVerdict {
  return {
    signal: "input_qc",
    decision: "proceed",
    available: true,
    enabled: true,
    message: "",
    detail: {},
    ...overrides,
  };
}

function jobWithDecision(decision: GatekeeperDecisionJson): ClinicalJob {
  return {
    job_id: "job-1",
    state: "done",
    stage: "done",
    progress: 1,
    case_id: "case-1",
    error: null,
    ingest_result: null,
    input_qc_pre: null,
    input_qc_post: null,
    preprocess_warnings: null,
    gatekeeper_decision: decision,
    created_at: 0,
    updated_at: 0,
  };
}

function renderPanel(decision: GatekeeperDecisionJson): string {
  return renderToStaticMarkup(React.createElement(GatekeeperPanel, { job: jobWithDecision(decision) }));
}

describe("GatekeeperPanel disabled-signal rendering", () => {
  it("labels a disabled conformal_band as not used for refusal, not as unavailable or a refusal", () => {
    const html = renderPanel({
      decision: "proceed",
      verdicts: [
        verdict({
          signal: "conformal_band",
          enabled: false,
          message: "Band width computed but not consulted by the gate.",
        }),
      ],
    });

    expect(html).toContain("Conformal band width (not used for refusal)");
    expect(html).toContain("informational");
    expect(html).not.toContain("unavailable");
    expect(html).not.toMatch(/>refuse</);
  });

  it("mutes a disabled signal's label and message so it is visually distinct from an enabled one", () => {
    const html = renderPanel({
      decision: "proceed",
      verdicts: [verdict({ signal: "conformal_band", enabled: false, message: "not consulted" })],
    });

    // The label span picks up the same dim tone used for de-emphasised text
    // elsewhere in the app, in place of the normal enabled-row primary tone.
    expect(html).toContain('text-text-dim">Conformal band width (not used for refusal)</span>');
    expect(html).not.toContain('text-text-primary">Conformal band width');
    expect(html).toContain('text-text-dim">not consulted</p>');
  });

  it("renders an enabled signal without the disabled styling or suffix", () => {
    const html = renderPanel({
      decision: "proceed",
      verdicts: [
        verdict({ signal: "input_qc", enabled: true, decision: "proceed", message: "looks fine" }),
      ],
    });

    expect(html).toContain('text-text-primary">Input QC</span>');
    expect(html).not.toContain("(not used for refusal)");
    expect(html).not.toContain("informational");
    expect(html).toContain("proceed");
  });
});
