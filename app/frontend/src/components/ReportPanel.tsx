import type { ReactNode } from "react";
import { X } from "lucide-react";
import type { LayoutMode } from "./ViewportGrid";
import type { AnatomyStructureRow, EloquenceInvolvedRow, ReportBurden, ReportResponse } from "../api";
import { burdenLabel, formatBurdenValue, formatDistanceMm, formatPercent, segmentationLabel } from "../lib/report";

export type ReportPanelStatus =
  | "loading"
  | "loaded"
  | "not_found"
  | "server_error"
  | "unreachable"
  | "invalid";

interface ReportPanelProps {
  open: boolean;
  onClose: () => void;
  layout: LayoutMode;
  caseId: string | null;
  status: ReportPanelStatus;
  report: ReportResponse | null;
  /** The server's `detail` message (server_error / not_found) or the client validation message (invalid). */
  errorMessage: string | null;
}

// Rendering order for the burden sub-blocks - deliberately NOT the same order
// report.py uses for its own Markdown (volumes, fractions, shape,
// multifocality, laterality, centroid, other): this panel leads with
// multifocality and laterality ahead of shape, per the panel spec, because
// those two read as the more clinically load-bearing numbers at a glance.
const BURDEN_BLOCK_ORDER: (keyof ReportBurden)[] = [
  "volumes",
  "fractions",
  "multifocality",
  "laterality",
  "shape",
  "centroid",
  "other",
];

const BURDEN_BLOCK_TITLES: Record<keyof ReportBurden, string> = {
  volumes: "Volumes",
  fractions: "Composition",
  multifocality: "Multifocality",
  laterality: "Laterality",
  shape: "Shape",
  centroid: "Centroid (voxel index)",
  other: "Other",
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-t border-surface-seam px-4 py-4">
      <div className="eyebrow">{title}</div>
      {children}
    </div>
  );
}

function BurdenBlock({ title, block }: { title: string; block: Record<string, unknown> }) {
  const keys = Object.keys(block).sort();
  if (keys.length === 0) return null;
  return (
    <div>
      <div className="mb-1 font-condensed text-[11px] font-semibold tracking-[0.08em] text-text-secondary uppercase">
        {title}
      </div>
      <dl className="flex flex-col gap-0.5">
        {keys.map((key) => (
          <div key={key} className="flex items-baseline justify-between gap-3">
            <dt className="font-mono text-xs text-text-secondary">{burdenLabel(key)}</dt>
            <dd className="tabular shrink-0 font-mono text-xs text-text-primary">
              {formatBurdenValue(key, block[key])}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function StructureTable({ rows }: { rows: AnatomyStructureRow[] }) {
  if (rows.length === 0) {
    return <p className="font-mono text-xs text-text-dim">No structures recorded.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse font-mono text-xs">
        <thead>
          <tr className="text-left text-text-dim">
            <th className="py-1 pr-2 font-normal">Structure</th>
            <th className="py-1 pr-2 font-normal">Lat.</th>
            <th className="py-1 pr-2 font-normal">Lobe</th>
            <th className="py-1 pr-2 font-normal">Eloquence</th>
            <th className="py-1 pr-2 text-right font-normal">% of tumour</th>
            <th className="py-1 text-right font-normal">% of structure</th>
          </tr>
        </thead>
        <tbody>
          {/* Order and truncation are the server's - frac_of_structure
              descending, top_n rows. Re-sorting here would bury exactly the
              row report.py's docstring calls out: a structure that holds a
              small share of the tumour but has itself been mostly destroyed. */}
          {rows.map((row, i) => (
            <tr key={`${row.structure}-${i}`} className="border-t border-surface-seam">
              <td className="py-1 pr-2 text-text-primary">{row.structure}</td>
              <td className="py-1 pr-2 text-text-secondary">{row.laterality ?? "—"}</td>
              <td className="py-1 pr-2 text-text-secondary">{row.lobe ?? "—"}</td>
              <td className="py-1 pr-2 text-text-secondary">{row.eloquence ?? "—"}</td>
              <td className="tabular py-1 pr-2 text-right text-text-primary">
                {formatPercent(row.frac_of_tumour)}
              </td>
              <td className="tabular py-1 text-right text-text-primary">
                {formatPercent(row.frac_of_structure)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InvolvedTable({ rows }: { rows: EloquenceInvolvedRow[] }) {
  if (rows.length === 0) {
    return (
      <p className="font-mono text-xs text-text-dim">
        No structure from this classification overlaps the reported region.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse font-mono text-xs">
        <thead>
          <tr className="text-left text-text-dim">
            <th className="py-1 pr-2 font-normal">Structure</th>
            <th className="py-1 pr-2 font-normal">Lat.</th>
            <th className="py-1 pr-2 text-right font-normal">% of tumour</th>
            <th className="py-1 text-right font-normal">% of structure</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={`${row.structure}-${i}`} className="border-t border-surface-seam">
              <td className="py-1 pr-2 text-text-primary">{row.structure}</td>
              <td className="py-1 pr-2 text-text-secondary">{row.laterality ?? "—"}</td>
              <td className="tabular py-1 pr-2 text-right text-text-primary">
                {formatPercent(row.frac_of_tumour)}
              </td>
              <td className="tabular py-1 text-right text-text-primary">
                {formatPercent(row.frac_of_structure)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CenteredMessage({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center px-6 text-center">
      <p className="font-mono text-xs text-text-secondary">{children}</p>
    </div>
  );
}

export function ReportPanel({ open, onClose, layout, caseId, status, report, errorMessage }: ReportPanelProps) {
  if (!open) return null;

  const widthClass = layout === "single" ? "w-full" : "w-[420px]";

  // Resolved once per render rather than inside JSX so the fallback path
  // (an unrecognised segmentation_source) is computed exactly once - same
  // reasoning as Legend.tsx's inline handling of an unrecognised
  // X-Uncertainty-Kind header, which this mirrors deliberately: never fall
  // back to the reassuring label for a value this client does not recognise.
  let segBadge: { text: string; tone: "prediction" | "truth" } | null = null;
  if (report) {
    try {
      segBadge = segmentationLabel(report.provenance);
    } catch {
      segBadge = {
        text: `unrecognised source (${String(report.provenance.segmentation_source)})`,
        tone: "prediction",
      };
    }
  }

  return (
    <>
      <div
        className="absolute inset-0 z-30 bg-black/60"
        onClick={onClose}
        aria-hidden="true"
      />
      <div
        className={`absolute inset-y-0 right-0 z-40 flex flex-col overflow-hidden border-l border-surface-seam bg-surface-panel shadow-2xl ${widthClass}`}
        role="dialog"
        aria-label="Structured report"
      >
        <div className="flex shrink-0 items-center gap-2 border-b border-surface-seam px-4 py-2.5">
          <span className="font-condensed text-[11px] tracking-[0.12em] text-text-dim uppercase">
            Report
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close report"
            className="ml-auto rounded-sm p-1 text-text-secondary transition-colors duration-[120ms] hover:text-text-primary"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {status === "loading" && <CenteredMessage>Loading report for {caseId}…</CenteredMessage>}

          {status === "unreachable" && (
            <CenteredMessage>
              No response from the API. Start it with{" "}
              <code className="text-data-oedema">uvicorn app.backend.main:app --reload</code>.
            </CenteredMessage>
          )}

          {status === "not_found" && (
            <CenteredMessage>
              No report has been generated for this case — run{" "}
              <code className="text-data-oedema">scripts/report.py</code>.
            </CenteredMessage>
          )}

          {status === "server_error" && (
            <CenteredMessage>
              The server refused to serve this report.
              <br />
              <span className="mt-2 block text-text-dim">{errorMessage}</span>
            </CenteredMessage>
          )}

          {status === "invalid" && (
            <CenteredMessage>
              The report the server returned could not be read.
              <br />
              <span className="mt-2 block text-text-dim">{errorMessage}</span>
            </CenteredMessage>
          )}

          {status === "loaded" && report && segBadge && (
            <>
              {/* --- Header ------------------------------------------------ */}
              <div className="flex flex-col gap-1.5 px-4 py-4">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-sm text-text-primary">
                    {report.case_id}
                  </span>
                  {/* Deliberately no data colour here (index.css: "Data
                      colours - never used for chrome") - the distinction is
                      carried by the text itself, not by colour coding that
                      could be read as matching a tissue-class swatch. */}
                  <span className="shrink-0 rounded-sm border border-surface-seam bg-surface-raised px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wide text-text-primary uppercase">
                    {segBadge.text}
                  </span>
                </div>
                <p className="font-mono text-[11px] text-text-dim">
                  Report schema v{report.report_version} · generated {report.generated_utc}
                </p>
              </div>

              {/* --- Disclaimer --------------------------------------------- */}
              <div className="mx-4 mb-1 border border-surface-seam bg-surface-raised px-3 py-2.5">
                <p className="font-mono text-xs leading-relaxed text-text-primary">
                  {report.disclaimer}
                </p>
              </div>

              {/* --- Burden --------------------------------------------------- */}
              <Section title="Tumour burden profile">
                <div className="flex flex-col gap-3">
                  {BURDEN_BLOCK_ORDER.map((blockKey) => (
                    <BurdenBlock
                      key={blockKey}
                      title={BURDEN_BLOCK_TITLES[blockKey]}
                      block={report.burden[blockKey]}
                    />
                  ))}
                </div>
              </Section>

              {/* --- Anatomy ---------------------------------------------------- */}
              <Section title="Anatomical involvement">
                <p className="font-mono text-xs text-text-secondary">
                  Atlas: <span className="text-text-primary">{report.anatomy.atlas.name}</span>{" "}
                  {report.anatomy.atlas.version}
                </p>
                <p className="font-mono text-[11px] leading-relaxed text-text-dim">
                  {report.anatomy.caveat}
                </p>
                <p className="font-mono text-[11px] leading-relaxed text-text-dim">
                  {report.anatomy.coverage_line}
                </p>
                <p className="font-mono text-xs text-text-secondary">
                  Structures involved:{" "}
                  <span className="text-text-primary">
                    {report.anatomy.n_structures_involved ?? "—"}
                  </span>
                  {" · "}Unlabelled fraction:{" "}
                  <span className="text-text-primary">
                    {formatPercent(report.anatomy.frac_unlabelled)}
                  </span>
                </p>
                <StructureTable rows={report.anatomy.structures} />
                <p className="font-mono text-[10px] leading-relaxed text-text-dim">
                  % of tumour is this structure's share of the whole tumour; % of structure is how
                  much of THIS structure the tumour has overtaken. A lesion can hold a small share
                  of the tumour while destroying nearly all of one structure — the table is sorted
                  by the second column.
                </p>
              </Section>

              {/* --- Eloquence -------------------------------------------------- */}
              <Section title="Eloquence reference">
                <p className="font-mono text-xs text-text-secondary">
                  Classification:{" "}
                  <span className="text-text-primary">{report.eloquence.classification}</span>
                </p>
                <p className="font-mono text-xs text-text-secondary">
                  Distance to nearest listed structure:{" "}
                  <span className="text-text-primary">
                    {formatDistanceMm(report.eloquence.distance_mm)}
                  </span>
                </p>
                <p className="font-mono text-xs text-text-secondary">
                  Within {formatDistanceMm(report.eloquence.near_eloquent_threshold_mm)} of an
                  eloquent structure:{" "}
                  <span className="text-text-primary">
                    {report.eloquence.near_eloquent ? "yes" : "no"}
                  </span>
                </p>
                <blockquote className="border-l-2 border-surface-seam pl-3 font-mono text-xs leading-relaxed text-text-primary italic">
                  {report.eloquence.evidence}
                </blockquote>
                <p className="font-mono text-[11px] leading-relaxed text-text-dim">
                  Source: {report.eloquence.citation}
                </p>
                <p className="font-mono text-[11px] leading-relaxed text-text-dim">
                  {report.eloquence.source_owns_claim}
                </p>
                <InvolvedTable rows={report.eloquence.involved} />
                {report.eloquence.coverage_gaps.length > 0 && (
                  <p className="font-mono text-[11px] leading-relaxed text-text-dim">
                    Coverage gaps (source terms with no matching structure here):{" "}
                    {report.eloquence.coverage_gaps.join(", ")}
                  </p>
                )}
              </Section>

              {/* --- Not claimed -------------------------------------------------- */}
              <Section title="Not claimed">
                <ul className="flex flex-col gap-2">
                  {report.not_claimed.map(([what, why]) => (
                    <li key={what} className="font-mono text-xs leading-relaxed text-text-secondary">
                      <span className="text-text-primary">{what}</span>: {why}
                    </li>
                  ))}
                </ul>
              </Section>

              {/* --- Provenance ----------------------------------------------------- */}
              <div className="border-t border-surface-seam px-4 py-4">
                <details>
                  <summary className="eyebrow cursor-pointer select-none">Provenance</summary>
                  <dl className="mt-2 flex flex-col gap-0.5">
                    {Object.entries(report.provenance)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([key, value]) => (
                        <div key={key} className="flex items-baseline justify-between gap-3">
                          <dt className="shrink-0 font-mono text-xs text-text-secondary">{key}</dt>
                          <dd className="tabular text-right font-mono text-xs break-all text-text-primary">
                            {value === null || value === undefined
                              ? "—"
                              : typeof value === "object"
                                ? Object.entries(value as Record<string, unknown>)
                                    .map(([k, v]) => `${k}=${v}`)
                                    .join(", ") || "—"
                                : String(value)}
                          </dd>
                        </div>
                      ))}
                  </dl>
                </details>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
