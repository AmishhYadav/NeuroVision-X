import type { ClinicalJob, GatekeeperSignalVerdict } from "../../api";

interface GatekeeperPanelProps {
  job: ClinicalJob;
}

// `neurovision.inference.gatekeeper.SIGNAL_NAMES`, in the same order that
// module always emits them - not every deployment necessarily enables every
// signal (see `configs/clinical/default.yaml`'s `gatekeeper.enabled_signals`
// - `ood_score` is measured but not yet trusted), so a label is supplied for
// every signal regardless of which are enabled here.
const SIGNAL_LABEL: Record<string, string> = {
  input_qc: "Input QC",
  predicted_dice: "Predicted Dice (QC estimate)",
  conformal_band: "Conformal band width",
  ood_score: "Out-of-distribution score",
  intended_use: "Intended use (adults only)",
};

function formatDetailValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function VerdictRow({ verdict }: { verdict: GatekeeperSignalVerdict }) {
  const baseLabel = SIGNAL_LABEL[verdict.signal] ?? verdict.signal;
  // `enabled: false` means this signal is computed and shown, but the
  // backend's gatekeeper never looks at it when deciding proceed/caution/
  // refuse (see `configs/clinical/default.yaml`'s `gatekeeper.enabled_signals`
  // - conformal_band was removed from that list 2026-09-26). That is
  // informational, not an error, a missing value, or a step towards refusal,
  // so it gets its own label suffix and muted styling below rather than
  // sharing any treatment that could read as a problem with this case.
  const label = verdict.enabled ? baseLabel : `${baseLabel} (not used for refusal)`;
  // Only flat (non-object) detail entries are worth a one-line readout here;
  // anything nested stays available in the raw JSON below rather than being
  // recursively flattened into a summary that was never meant to hold it.
  const detailEntries = Object.entries(verdict.detail).filter(
    ([, v]) => typeof v !== "object" || v === null,
  );

  return (
    <div className="flex flex-col gap-1 border-t border-surface-seam pt-3 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span
          className={`text-xs ${verdict.enabled ? "text-text-primary" : "text-text-dim"}`}
        >
          {label}
        </span>
        <span className="eyebrow">
          {!verdict.enabled
            ? "informational"
            : !verdict.available
              ? "unavailable"
              : verdict.decision.replace(/_/g, " ")}
        </span>
      </div>
      <p
        className={`text-xs leading-relaxed ${verdict.enabled ? "text-text-secondary" : "text-text-dim"}`}
      >
        {verdict.message}
      </p>
      {detailEntries.length > 0 && (
        <dl className="flex flex-wrap gap-x-4 gap-y-0.5">
          {detailEntries.map(([k, v]) => (
            <div key={k} className="flex items-baseline gap-1.5">
              <dt className="font-mono text-[10px] text-text-dim">{k}</dt>
              <dd className="tabular font-mono text-[10px] text-text-secondary">
                {formatDetailValue(v)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/**
 * Shown when `job.gatekeeper_decision` is present and the job did NOT end up
 * `"refused"` (i.e. `"done"`, decision `proceed` or `proceed_with_caution`).
 *
 * This is the master plan's "QC estimate" (`predicted_dice`) and conformal
 * band width surfaced as an actual product feature rather than left in a log
 * file - every signal the gatekeeper judged is listed, including ones that
 * are not enabled or were unavailable for this case, so the panel never
 * implies more was checked than actually was.
 */
export function GatekeeperPanel({ job }: GatekeeperPanelProps) {
  const decision = job.gatekeeper_decision;
  if (!decision) return null;

  const isCaution = decision.decision === "proceed_with_caution";
  const isRefuse = decision.decision === "refuse";
  // Refusal is amber (a correct outcome) drawn as a hollow ring so it differs
  // from CAUTION's filled dot.
  const pillClass = isRefuse || isCaution
    ? "border-gate-caution/50 bg-gate-caution/10 text-gate-caution"
    : "border-gate-proceed/50 bg-gate-proceed/10 text-gate-proceed";
  const dotClass = isRefuse
    ? "border-2 border-gate-caution"
    : isCaution
      ? "bg-gate-caution"
      : "bg-gate-proceed";
  const pillText = isRefuse ? "REFUSE" : isCaution ? "PROCEED WITH CAUTION" : "PROCEED";

  return (
    <div className="glass-panel flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-center gap-3">
        <span className="eyebrow">Gatekeeper decision</span>
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-4 py-1.5 font-heading text-base font-semibold ${pillClass}`}
        >
          <span className={`h-2 w-2 shrink-0 rounded-full ${dotClass}`} aria-hidden="true" />
          {pillText}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {decision.verdicts.map((v) => (
          <VerdictRow key={v.signal} verdict={v} />
        ))}
      </div>

      <details>
        <summary className="eyebrow cursor-pointer select-none">Raw gatekeeper decision</summary>
        <pre className="mt-2 max-h-64 overflow-auto font-mono text-[10px] leading-relaxed text-text-secondary">
          {JSON.stringify(decision, null, 2)}
        </pre>
      </details>
    </div>
  );
}
