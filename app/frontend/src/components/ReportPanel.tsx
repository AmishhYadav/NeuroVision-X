import { Fragment, useMemo, type ReactNode } from "react";
import { X } from "lucide-react";
import type { LayoutMode } from "./ViewportGrid";
import type {
  AnatomyStructureRow,
  BurdenBlock,
  ReportProvenance,
  ReportResponse,
} from "../api";
import {
  burdenLabel,
  formatBurdenValue,
  formatGeometryValue,
  formatPercent,
  geometryLabel,
  segmentationLabel,
} from "../lib/report";
import {
  type InterpretedFact,
  type InterpretedSection,
  interpretReport,
} from "../lib/reportInterpretation";
import { MolecularPanel } from "../pages/clinical/MolecularPanel";

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
  /**
   * Fired when the pointer enters/leaves a structure-table row (`name`), or
   * the row is clicked (same handler, so a touch device - which never fires
   * `onMouseEnter` - can still light the twin's shell). Optional so
   * `App.tsx`'s demo path, which has no 3D twin to highlight, needs no
   * change.
   */
  onHoverStructure?: (name: string | null) => void;
  /** The structure name to highlight in the table, or null/absent for none - set by the caller from whichever atlas index the twin currently has highlighted. */
  highlightedStructureName?: string | null;
  /**
   * The clinical job id this report belongs to, if any - passed straight
   * through to `MolecularPanel` so it knows which job's pathology to
   * PUT/GET. Absent on the demo path (`App.tsx`, no clinical job), where
   * `report.molecular` is also always absent - see that field's doc comment
   * in `api.ts`. Both conditions gate the panel below, so neither one alone
   * is enough to render it.
   */
  pathologyJobId?: string | null;
}

function CenteredMessage({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center px-6 text-center">
      <p className="font-mono text-xs text-text-secondary">{children}</p>
    </div>
  );
}

function FactsGrid({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-6 gap-y-2">
      {facts.map((f, i) => (
        <Fragment key={i}>
          <dt className="text-sm text-text-primary">
            {f.label}
            {f.note && <div className="mt-0.5 text-xs text-text-dim">{f.note}</div>}
          </dt>
          <dd className="tabular text-right font-mono text-sm text-text-primary">{f.value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function CompositionBar({ fractions }: { fractions: BurdenBlock }) {
  const segments: { key: string; value: unknown; color: string; label: string }[] = [
    { key: "edema", value: fractions.frac_edema_of_wt, color: "bg-data-oedema", label: "Swelling" },
    {
      key: "enhancing",
      value: fractions.frac_enhancing_of_wt,
      color: "bg-data-enhancing",
      label: "Enhancing",
    },
    {
      key: "necrotic",
      value: fractions.frac_necrotic_of_wt,
      color: "bg-data-necrotic",
      label: "Necrotic",
    },
  ];
  const allFinite = segments.every((s) => typeof s.value === "number" && Number.isFinite(s.value));
  if (!allFinite) return null;

  return (
    <div className="mb-4">
      <div className="flex h-3 w-full overflow-hidden rounded-sm">
        {segments.map((s) => (
          <div
            key={s.key}
            className={s.color}
            style={{ width: `${(s.value as number) * 100}%` }}
            title={`${s.label}: ${formatPercent(s.value as number)}`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {segments.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-text-secondary">
            <span className={`inline-block h-2 w-2 rounded-sm ${s.color}`} />
            {s.label} {formatPercent(s.value as number)}
          </div>
        ))}
      </div>
    </div>
  );
}

function RegionsFacts({
  facts,
  structures,
  onHoverStructure,
  highlightedStructureName,
}: {
  facts: InterpretedFact[];
  structures: AnatomyStructureRow[];
  onHoverStructure?: (name: string | null) => void;
  highlightedStructureName?: string | null;
}) {
  const summaryFacts = facts.slice(0, 2);
  const rowFacts = facts.slice(2);
  const n = Math.min(rowFacts.length, structures.length);

  return (
    <>
      <FactsGrid facts={summaryFacts} />
      <div className="mt-4 flex flex-col gap-1">
        {Array.from({ length: n }, (_, i) => {
          const fact = rowFacts[i];
          const structure = structures[i];
          const rawWidth =
            typeof structure.frac_of_structure === "number" ? structure.frac_of_structure * 100 : 0;
          const width = Math.min(100, Math.max(0, rawWidth));
          const isHighlighted = structure.structure === highlightedStructureName;
          
          return (
            <div 
              key={i}
              onMouseEnter={() => onHoverStructure?.(structure.structure)}
              onMouseLeave={() => onHoverStructure?.(null)}
              onClick={() => onHoverStructure?.(structure.structure)}
              className={`-mx-3 cursor-default rounded-sm p-3 transition-colors ${
                isHighlighted ? "bg-surface-raised" : "hover:bg-surface-raised/50"
              }`}
            >
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm text-text-primary">{fact.label}</span>
                <span className="font-mono text-xs text-text-dim">
                  {formatPercent(structure.frac_of_structure)} of region &middot;{" "}
                  {formatPercent(structure.frac_of_tumour)} of tumour
                </span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-sm bg-surface-seam">
                <div className="h-1.5 bg-text-secondary" style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

function LimitsFacts({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <ol className="flex flex-col gap-3">
      {facts.map((f, i) => (
        <li key={i} className="text-sm leading-relaxed text-text-primary">
          <strong>{f.label}</strong>
          {f.note && <div className="mt-1 text-text-secondary">{f.note}</div>}
        </li>
      ))}
    </ol>
  );
}

function SectionFacts({ 
  section, 
  report, 
  onHoverStructure, 
  highlightedStructureName 
}: { 
  section: InterpretedSection; 
  report: ReportResponse;
  onHoverStructure?: (name: string | null) => void;
  highlightedStructureName?: string | null;
}) {
  if (section.id === "composition") {
    return (
      <>
        <CompositionBar fractions={report.burden.fractions} />
        <FactsGrid facts={section.facts} />
      </>
    );
  }
  if (section.id === "regions") {
    return (
      <RegionsFacts 
        facts={section.facts} 
        structures={report.anatomy.structures} 
        onHoverStructure={onHoverStructure}
        highlightedStructureName={highlightedStructureName}
      />
    );
  }
  if (section.id === "limits") {
    return <LimitsFacts facts={section.facts} />;
  }
  return <FactsGrid facts={section.facts} />;
}

function isEmptyBlock(block: BurdenBlock | undefined): block is undefined {
  return !block || Object.keys(block).length === 0;
}

function BurdenBlockDl({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <div className="mt-4">
      <div className="font-condensed text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
        {title}
      </div>
      <dl className="mt-1 flex flex-col gap-0.5">
        {Object.entries(block).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-3">
            <dt className="font-mono text-xs text-text-secondary">{burdenLabel(k)}</dt>
            <dd className="tabular shrink-0 text-right font-mono text-xs text-text-primary">
              {formatBurdenValue(k, v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function GeometryBlockDl({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <div className="mt-4">
      <div className="font-condensed text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
        {title}
      </div>
      <dl className="mt-1 flex flex-col gap-0.5">
        {Object.entries(block).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-3">
            <dt className="font-mono text-xs text-text-secondary">{geometryLabel(k)}</dt>
            <dd className="tabular shrink-0 text-right font-mono text-xs text-text-primary">
              {formatGeometryValue(k, typeof v === "number" ? v : null)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function flattenProvenanceValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, val]) => `${k}=${String(val)}`)
      .join(", ");
  }
  return String(v);
}

function ProvenanceDl({ provenance }: { provenance: ReportProvenance }) {
  const entries = Object.entries(provenance as unknown as Record<string, unknown>);
  return (
    <div className="mt-4">
      <div className="font-condensed text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
        Provenance
      </div>
      <dl className="mt-1 flex flex-col gap-0.5">
        {entries.sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-3">
            <dt className="shrink-0 font-mono text-xs text-text-secondary">{k}</dt>
            <dd className="tabular text-right font-mono text-xs break-all text-text-primary">
              {flattenProvenanceValue(v)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function TechnicalData({ report }: { report: ReportResponse }) {
  return (
    <details className="border-t border-surface-seam px-4 py-4 mb-4">
      <summary className="eyebrow cursor-pointer select-none">
        Full technical data
      </summary>
      <div className="mt-2 flex flex-col gap-2">
        <BurdenBlockDl title="Volumes" block={report.burden.volumes} />
        <BurdenBlockDl title="Fractions" block={report.burden.fractions} />
        <BurdenBlockDl title="Shape" block={report.burden.shape} />
        <BurdenBlockDl title="Multifocality" block={report.burden.multifocality} />
        <BurdenBlockDl title="Laterality" block={report.burden.laterality} />
        <BurdenBlockDl title="Centroid" block={report.burden.centroid} />
        <BurdenBlockDl title="Other (burden)" block={report.burden.other} />
        {report.geometry && (
          <>
            <GeometryBlockDl title="Shape (geometric)" block={report.geometry.shape} />
            <GeometryBlockDl title="Extent" block={report.geometry.extent} />
            <GeometryBlockDl title="Rim" block={report.geometry.rim} />
            <GeometryBlockDl title="Other (geometry)" block={report.geometry.other} />
          </>
        )}
        <div className="mt-4">
          <div className="font-condensed text-[11px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
            Eloquence citation
          </div>
          <p className="mt-1 font-mono text-[11px] leading-relaxed text-text-dim">
            {report.eloquence.citation}
          </p>
        </div>
        <ProvenanceDl provenance={report.provenance} />
      </div>
    </details>
  );
}

export function ReportPanel({
  open,
  onClose,
  layout,
  caseId,
  status,
  report,
  errorMessage,
  onHoverStructure,
  highlightedStructureName,
  pathologyJobId,
}: ReportPanelProps) {
  const interpreted = useMemo(() => (report ? interpretReport(report) : null), [report]);

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

  const overview = interpreted?.sections.find((s) => s.id === "overview") ?? null;
  const remainingSections = interpreted?.sections.filter((s) => s.id !== "overview") ?? [];

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

          {status === "loaded" && report && interpreted && segBadge && (
            <>
              {/* --- Header ------------------------------------------------ */}
              <div className="flex flex-col gap-1.5 px-4 py-4">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-sm text-text-primary">
                    {report.case_id}
                  </span>
                  <span className="shrink-0 rounded-sm border border-surface-seam bg-surface-raised px-1.5 py-0.5 font-mono text-[10px] font-semibold tracking-wide text-text-primary uppercase">
                    {segBadge.text}
                  </span>
                </div>
                <p className="font-mono text-[11px] text-text-dim">
                  Report schema v{report.report_version} · generated {report.generated_utc}
                </p>
              </div>

              {/* --- Disclaimer --------------------------------------------- */}
              <div className="mx-4 mb-4 border border-surface-seam bg-surface-raised px-3 py-2.5">
                <p className="font-mono text-xs leading-relaxed text-text-primary">
                  {report.disclaimer}
                </p>
              </div>

              {/* --- Overview --------------------------------------------- */}
              {overview && (
                <div className="border-t border-surface-seam px-4 py-6">
                  <p className="text-sm leading-relaxed text-text-primary">
                    {overview.headline}
                  </p>
                  {overview.facts.length > 0 && (
                    <div className="mt-6 grid grid-cols-2 gap-4">
                      {overview.facts.map((f, i) => (
                        <div key={i}>
                          <div className="font-mono text-[10px] uppercase tracking-wide text-text-secondary">
                            {f.label}
                          </div>
                          <div className="tabular font-condensed text-xl text-text-primary mt-0.5">
                            {f.value}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="mt-5 text-xs text-text-secondary leading-relaxed">
                    {overview.explanation}
                  </p>
                </div>
              )}

              {/* --- Remaining Sections ----------------------------------- */}
              {remainingSections.map((section) => (
                <div key={section.id} className="border-t border-surface-seam px-4 py-6">
                  <h3 className="font-condensed text-xl text-text-primary">
                    {section.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-text-primary">
                    {section.headline}
                  </p>
                  <p className="mt-2 mb-5 text-xs leading-relaxed text-text-secondary">
                    {section.explanation}
                  </p>
                  
                  <SectionFacts 
                    section={section} 
                    report={report}
                    onHoverStructure={onHoverStructure}
                    highlightedStructureName={highlightedStructureName}
                  />

                  {section.caveat && (
                    <div className="mt-5 border-l-2 border-surface-seam pl-3 text-xs leading-relaxed text-text-dim">
                      <div className="font-mono text-[10px] uppercase tracking-wide mb-1">Caveat</div>
                      {section.caveat.split("\n").map((line, i) => (
                        <p key={i} className={i > 0 ? "mt-1" : ""}>
                          {line}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {/* --- Confirmed pathology (T5.6) ---------------------------- */}
              {report.molecular && pathologyJobId && (
                <div className="border-t border-surface-seam px-4 py-6">
                  <MolecularPanel jobId={pathologyJobId} molecular={report.molecular} />
                </div>
              )}

              {/* --- Full technical data ---------------------------------- */}
              <TechnicalData report={report} />
            </>
          )}
        </div>
      </div>
    </>
  );
}
