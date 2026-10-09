// "Full technical data": every raw key the report carries, collapsed by
// default. Formatting goes through lib/report (burdenLabel / formatBurdenValue
// / geometryLabel / formatGeometryValue), as before.
//
// Print: the <details> stays closed, so it prints as nothing but its
// summary line, and the summary itself is print:hidden ("expand nothing").
import type { ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import type { BurdenBlock, ReportProvenance, ReportResponse } from "../../api";
import { burdenLabel, formatBurdenValue, formatGeometryValue, geometryLabel } from "../../lib/report";

function isEmptyBlock(block: BurdenBlock | undefined): block is undefined {
  return !block || Object.keys(block).length === 0;
}

function KvTable({ title, rows }: { title: string; rows: [string, ReactNode][] }) {
  return (
    <div className="min-w-0 break-inside-avoid">
      <h3 className="text-xs font-medium text-text-dim">{title}</h3>
      <dl className="mt-2 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-baseline justify-between gap-4 border-t border-surface-seam/70 py-1.5">
            <dt className="min-w-0 text-text-secondary">{k}</dt>
            <dd className="max-w-[60%] text-right font-mono tabular break-all text-text-primary">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function BurdenTable({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <KvTable title={title} rows={Object.entries(block).map(([k, v]) => [burdenLabel(k), formatBurdenValue(k, v)])} />
  );
}

function GeometryTable({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <KvTable
      title={title}
      rows={Object.entries(block).map(([k, v]) => [geometryLabel(k), formatGeometryValue(k, typeof v === "number" ? v : null)])}
    />
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

export function TechnicalData({ report, number }: { report: ReportResponse; number: string }) {
  const provenance = Object.entries(report.provenance as ReportProvenance as unknown as Record<string, unknown>);
  return (
    <details id="section-technical" className="group scroll-mt-6 border-t border-surface-seam py-8">
      <summary className="flex cursor-pointer list-none items-baseline gap-3 rounded-md print:hidden [&::-webkit-details-marker]:hidden">
        <span className="font-mono tabular text-sm text-text-dim">{number}</span>
        <span className="font-heading text-xl font-semibold text-text-primary">Full technical data</span>
        <span className="hidden text-sm text-text-dim sm:inline">raw values and provenance</span>
        <ChevronDown
          size={18}
          aria-hidden="true"
          className="ml-auto self-center text-text-dim transition-transform duration-200 group-open:rotate-180"
        />
      </summary>
      <div className="mt-8 grid gap-x-10 gap-y-8 sm:grid-cols-2">
        <BurdenTable title="Volumes" block={report.burden.volumes} />
        <BurdenTable title="Fractions" block={report.burden.fractions} />
        <BurdenTable title="Shape" block={report.burden.shape} />
        <BurdenTable title="Multifocality" block={report.burden.multifocality} />
        <BurdenTable title="Laterality" block={report.burden.laterality} />
        <BurdenTable title="Centroid" block={report.burden.centroid} />
        <BurdenTable title="Other (burden)" block={report.burden.other} />
        {report.geometry && (
          <>
            <GeometryTable title="Shape (geometric)" block={report.geometry.shape} />
            <GeometryTable title="Extent" block={report.geometry.extent} />
            <GeometryTable title="Rim" block={report.geometry.rim} />
            <GeometryTable title="Other (geometry)" block={report.geometry.other} />
          </>
        )}
        <div className="min-w-0">
          <h3 className="text-xs font-medium text-text-dim">Eloquence citation</h3>
          <p className="mt-2 text-xs leading-relaxed text-text-secondary">{report.eloquence.citation}</p>
        </div>
        <KvTable title="Provenance" rows={provenance.map(([k, v]) => [k, flattenProvenanceValue(v)])} />
      </div>
    </details>
  );
}
