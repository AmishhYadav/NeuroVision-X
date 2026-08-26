import type { ClinicalJob, GatekeeperSignalVerdict } from "../../api";

interface GatekeeperPanelProps {
  job: ClinicalJob;
}

// `neurovision.inference.gatekeeper.SIGNAL_NAMES`, in the same order that
// module always emits them - not every deployment necessarily enables every
// signal (see `configs/clinical/default.yaml`'s `gatekeeper.enabled_signals`
// - `ood_score` is measured but not yet trusted), so a label is supplied for
// all four regardless of which are enabled here.
const SIGNAL_LABEL: Record<string, string> = {
  input_qc: "Input QC",
  predicted_dice: "Predicted Dice (QC estimate)",
  conformal_band: "Conformal band width",
  ood_score: "Out-of-distribution score",
};

function formatDetailValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function VerdictRow({ verdict }: { verdict: GatekeeperSignalVerdict }) {
  const label = SIGNAL_LABEL[verdict.signal] ?? verdict.signal;
  // Only flat (non-object) detail entries are worth a one-line readout here;
  // anything nested stays available in the raw JSON below rather than being
  // recursively flattened into a summary that was never meant to hold it.
  const detailEntries = Object.entries(verdict.detail).filter(
    ([, v]) => typeof v !== "object" || v === null,
  );

  return (
    <div className="flex flex-col gap-1 border-t border-surface-seam pt-2 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-mono text-xs text-text-primary">{label}</span>
        <span className="font-mono text-[10px] tracking-[0.04em] text-text-dim uppercase">
          {!verdict.enabled
            ? "not enabled"
            : !verdict.available
              ? "unavailable"
              : verdict.decision.replace(/_/g, " ")}
        </span>
      </div>
      <p className="font-mono text-[11px] leading-relaxed text-text-secondary">
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

  return (
    <div className="flex flex-col gap-3 border border-surface-seam bg-surface-panel px-4 py-4">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${isCaution ? "bg-data-amber" : "bg-data-oedema"}`}
          aria-hidden="true"
        />
        <span className="font-condensed text-xs font-semibold tracking-[0.12em] text-text-primary uppercase">
          {isCaution ? "Proceed with caution" : "Proceed"}
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
