// T5.6 of docs/research/tool_completion_plan.md - the "Confirmed pathology"
// panel `ReportPanel` mounts on a clinical job's report. This is the most
// diagnosis-shaped surface in the tool, so the copy discipline in CLAUDE.md
// is load-bearing here specifically:
//
//   - every string the server's molecular block already carries (labels,
//     meanings, cns5_role, caveat, ai_estimate.status, cns5.label,
//     cns5.source) is rendered VERBATIM from that block, never re-typed;
//   - the two strings this file DOES author itself live in
//     `lib/molecularCopy.ts`, scanned there against the forbidden-word list;
//   - the CNS5 line appears only once the lookup resolves (`cns5.name !==
//     null`), and always cites `cns5.source` next to the name so a reader
//     never mistakes it for something derived from imaging.
//
// NOTHING here is derived from imaging: every select offers exactly the
// server's own `allowed_values`, and every write goes through
// `putClinicalPathology`, which validates against the same knowledge base
// the report itself was built from.

import { useEffect, useState } from "react";
import { putClinicalPathology, type ReportMolecular } from "../../api";
import { NO_AI_ESTIMATE_TEXT, requiresHint } from "../../lib/molecularCopy";

interface MolecularPanelProps {
  jobId: string;
  molecular: ReportMolecular;
}

const SELECT_CLASSES =
  "bg-surface-panel border border-surface-seam rounded-sm px-2 py-1 font-mono text-xs text-text-primary disabled:cursor-not-allowed disabled:text-text-dim";

/** The AI-estimate slot for one marker - greyed, and always the server's own words (see the module docstring). */
function AiEstimateLine({ aiEstimate }: { aiEstimate: { status: string } | null }) {
  const text = aiEstimate ? aiEstimate.status : NO_AI_ESTIMATE_TEXT;
  return <p className="mt-1 font-mono text-[11px] text-text-dim">AI estimate: {text}</p>;
}

/**
 * The select + label shared by every row (a marker or histology). Extracted
 * because it is identical in both cases; what differs (the AI slot, the
 * "what this means" detail) is only present for a marker, whose block
 * carries `meaning` / `cns5_role` / `ai_estimate` - histology's block does
 * not, so `HistologyRow` below renders nothing invented in their place.
 */
function LabelledSelect({
  fieldKey,
  label,
  allowedValues,
  value,
  saving,
  onChange,
}: {
  fieldKey: string;
  label: string;
  allowedValues: string[];
  value: string;
  saving: boolean;
  onChange: (key: string, value: string) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-sm text-text-primary">{label}</span>
      <select
        aria-label={label}
        value={value}
        disabled={saving}
        onChange={(e) => onChange(fieldKey, e.target.value)}
        className={SELECT_CLASSES}
      >
        {allowedValues.map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </div>
  );
}

/** One marker row: label + select, its AI slot, and a collapsed detail of `meaning` / `cns5_role` (both verbatim from the block). */
function MarkerRow({
  markerKey,
  label,
  meaning,
  cnsRole,
  allowedValues,
  value,
  aiEstimate,
  saving,
  onChange,
}: {
  markerKey: string;
  label: string;
  meaning: string;
  cnsRole: string;
  allowedValues: string[];
  value: string;
  aiEstimate: { status: string } | null;
  saving: boolean;
  onChange: (key: string, value: string) => void;
}) {
  return (
    <div className="border-t border-surface-seam py-3 first:border-t-0">
      <LabelledSelect
        fieldKey={markerKey}
        label={label}
        allowedValues={allowedValues}
        value={value}
        saving={saving}
        onChange={onChange}
      />
      <AiEstimateLine aiEstimate={aiEstimate} />
      <details className="mt-1">
        <summary className="cursor-pointer select-none font-mono text-[10px] uppercase tracking-wide text-text-secondary">
          What this means
        </summary>
        <p className="mt-1 text-xs leading-relaxed text-text-secondary">{meaning}</p>
        <p className="mt-1 text-xs leading-relaxed text-text-secondary">{cnsRole}</p>
      </details>
    </div>
  );
}

/** The histology row - a select only. Histology's block carries no `meaning` / `cns5_role` / `ai_estimate` of its own, so there is nothing verbatim to show beneath it - inventing that text here would be exactly the re-typed copy CLAUDE.md forbids. */
function HistologyRow({
  allowedValues,
  value,
  saving,
  onChange,
}: {
  allowedValues: string[];
  value: string;
  saving: boolean;
  onChange: (key: string, value: string) => void;
}) {
  return (
    <div className="border-t border-surface-seam py-3">
      <LabelledSelect
        fieldKey="histology"
        label="Histology"
        allowedValues={allowedValues}
        value={value}
        saving={saving}
        onChange={onChange}
      />
    </div>
  );
}

export function MolecularPanel({ jobId, molecular }: MolecularPanelProps) {
  // `entered` mirrors every marker's and histology's `confirmed_pathology`
  // - the value a select shows. Initialised from the block and re-derived
  // whenever a new `molecular` prop arrives (e.g. the report was refetched
  // after this job's pathology changed elsewhere), never mutated in place.
  const [entered, setEntered] = useState<Record<string, string>>({});
  const [cns5, setCns5] = useState(molecular.cns5);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initial: Record<string, string> = {};
    for (const key of Object.keys(molecular.markers)) {
      initial[key] = molecular.markers[key].confirmed_pathology;
    }
    initial.histology = molecular.histology.confirmed_pathology;
    setEntered(initial);
    setCns5(molecular.cns5);
  }, [molecular]);

  async function handleChange(key: string, value: string) {
    const previous = entered[key];
    setError(null);
    setSaving(key);
    // Optimistic: show the pick immediately, revert only if the server
    // rejects it (e.g. a stale allowed_values list after a knowledge-base
    // change underneath this page).
    setEntered((prev) => ({ ...prev, [key]: value }));
    try {
      const res = await putClinicalPathology(jobId, { [key]: value });
      setEntered((prev) => ({ ...prev, ...res.pathology }));
      // `PathologyPutResponse.cns5` carries no `label` - it is fixed,
      // knowledge-base copy that never changes after the block first loads,
      // so it is kept from the current state rather than re-sent by the
      // server on every write.
      setCns5((prev) => ({ ...prev, ...res.cns5 }));
    } catch (err) {
      setEntered((prev) => ({ ...prev, [key]: previous }));
      setError(err instanceof Error ? err.message : "Failed to save pathology.");
    } finally {
      setSaving(null);
    }
  }

  const hint = requiresHint(cns5.requires);

  return (
    <div>
      <div className="font-condensed text-xl text-text-primary">Confirmed pathology</div>

      {/* The block's own disclaimer, verbatim - this is not frontend copy. */}
      <div className="mt-3 border border-surface-seam bg-surface-raised px-3 py-2.5">
        <p className="font-mono text-xs leading-relaxed text-text-primary">{molecular.caveat}</p>
      </div>

      <div className="mt-4">
        {Object.keys(molecular.markers).map((key) => {
          const marker = molecular.markers[key];
          return (
            <MarkerRow
              key={key}
              markerKey={key}
              label={marker.label}
              meaning={marker.meaning}
              cnsRole={marker.cns5_role}
              allowedValues={marker.allowed_values}
              value={entered[key] ?? marker.confirmed_pathology}
              aiEstimate={marker.ai_estimate}
              saving={saving === key}
              onChange={handleChange}
            />
          );
        })}
        <HistologyRow
          allowedValues={molecular.histology.allowed_values}
          value={entered.histology ?? molecular.histology.confirmed_pathology}
          saving={saving === "histology"}
          onChange={handleChange}
        />
      </div>

      {cns5.name !== null && (
        <p data-testid="cns5-line" className="mt-4 text-sm leading-relaxed text-text-primary">
          {cns5.label}: {cns5.name} — {cns5.source}
        </p>
      )}
      {cns5.name === null && hint && (
        <p className="mt-4 font-mono text-xs text-text-dim">{hint}</p>
      )}

      {error && (
        <p
          role="alert"
          className="mt-4 border border-surface-seam bg-surface-raised px-3 py-2 font-mono text-xs leading-relaxed text-text-primary"
        >
          {error}
        </p>
      )}
    </div>
  );
}
