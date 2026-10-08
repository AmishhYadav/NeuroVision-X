import { Check, Minus, X } from "lucide-react";
import type { ClinicalJob } from "../../api";

// Backend ROLES = ("t1", "t1ce", "t2", "flair") in app/backend/clinical_jobs.py.
const ROLES: { key: string; label: string }[] = [
  { key: "t1", label: "T1" },
  { key: "t1ce", label: "T1CE" },
  { key: "t2", label: "T2" },
  { key: "flair", label: "FLAIR" },
];

type RoleState = "unknown" | "missing" | "assigned";

/** Everything shown here comes from `job.ingest_result`; with none, every row is "—". */
export function SeriesMatrix({ job }: { job: ClinicalJob | null }) {
  const ingest = job?.ingest_result ?? null;

  function roleState(key: string): RoleState {
    if (!ingest) return "unknown";
    if (ingest.missing_roles.includes(key)) return "missing";
    if (Object.values(ingest.assignments).some((a) => a.role === key)) return "assigned";
    return "unknown";
  }

  const chipText = !job
    ? "awaiting upload"
    : !ingest
      ? "awaiting ingest"
      : `${4 - ingest.missing_roles.length}/4 assigned`;

  return (
    <section className="glass-panel p-5" aria-label="Required series">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-heading text-base font-semibold">Required series</h2>
        <span className="chip">{chipText}</span>
      </div>

      <ul className="mt-4 flex flex-col gap-2">
        {ROLES.map(({ key, label }) => {
          const s = roleState(key);
          return (
            <li
              key={key}
              className="flex items-center justify-between rounded-lg border border-surface-seam bg-surface-raised/40 px-3 py-2"
            >
              <span className="font-mono text-sm text-text-primary">{label}</span>
              {s === "assigned" && (
                <span className="flex items-center gap-1.5 text-xs text-brand-teal">
                  <Check className="h-4 w-4" aria-hidden="true" /> assigned
                </span>
              )}
              {s === "missing" && (
                <span className="flex items-center gap-1.5 text-xs text-gate-refuse">
                  <X className="h-4 w-4" aria-hidden="true" /> missing
                </span>
              )}
              {s === "unknown" && (
                <span className="flex items-center gap-1.5 text-xs text-text-dim">
                  <Minus className="h-4 w-4" aria-hidden="true" />
                  <span className="sr-only">not yet known</span>
                </span>
              )}
            </li>
          );
        })}
      </ul>

      {ingest && ingest.rejected.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-text-secondary select-none">
            {ingest.rejected.length} series not used
          </summary>
          <ul className="mt-2 flex flex-col gap-1">
            {ingest.rejected.map((r, i) => (
              <li key={`${r.series_uid}-${i}`} className="text-xs text-text-dim">
                <span className="font-mono text-text-secondary">…{r.series_uid.slice(-12)}</span>: {r.reason}
              </li>
            ))}
          </ul>
        </details>
      )}

      {ingest && ingest.warnings.length > 0 && (
        <ul className="mt-3 flex flex-col gap-1">
          {ingest.warnings.map((w, i) => (
            <li key={i} className="text-xs text-gate-caution">
              {w}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
