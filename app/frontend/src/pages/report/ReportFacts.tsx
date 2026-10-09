// How each report section's facts are laid out. Every label and value shown
// is an InterpretedFact string from lib/reportInterpretation.ts, which owns
// how a number becomes plain language. Three places read raw report numbers,
// and only to SIZE something (a bar width), never to print a new figure:
// the composition bars (report.burden.fractions) and the regions list
// (report.anatomy.structures).
//
// Layout vocabulary, matching the rest of the "reading room" system:
// hairline rows instead of boxes; figures in the mono face, words in the
// text face; tables where the data really is a table (WT/TC/ET x measure).
import { useState } from "react";
import type { AnatomyStructureRow, BurdenBlock, ReportResponse } from "../../api";
import { formatPercent } from "../../lib/report";
import { isFigure, regionTable, type RegionKey, type RegionTable } from "../../lib/reportLayout";
import type { InterpretedFact, InterpretedSection } from "../../lib/reportInterpretation";

const REGION_NAME: Record<RegionKey, string> = {
  WT: "Whole tumour",
  TC: "Tumour core",
  ET: "Enhancing tumour",
};

/** Mono for figures, the text face for words. */
function valueClass(value: string): string {
  return isFigure(value) ? "font-mono tabular" : "";
}

// --------------------------------------------------------------------- //
// Generic: a ledger of label / value rows
// --------------------------------------------------------------------- //

/** Label left, value right, hairline between rows. One column: the facts
 * are ordered in related runs (ventricles, then deep white matter, ...),
 * and two columns read across rows and split those runs apart. */
export function FactLedger({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl>
      {facts.map((f, i) => (
        <div
          key={i}
          className="flex items-baseline justify-between gap-6 border-t border-surface-seam py-2.5 break-inside-avoid"
        >
          <dt className="min-w-0 text-sm text-text-secondary">
            {f.label}
            {f.note && <span className="mt-0.5 block text-xs leading-snug text-text-dim">{f.note}</span>}
          </dt>
          <dd className={`shrink-0 text-right text-sm text-text-primary ${valueClass(f.value)}`}>{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}

// --------------------------------------------------------------------- //
// WT / TC / ET x measure
// --------------------------------------------------------------------- //

function RegionTableView({ table, caption }: { table: RegionTable; caption: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-surface-seam">
            <th scope="col" className="py-2 pr-3 text-left align-bottom text-xs font-medium text-text-dim sm:pr-4">
              Region
            </th>
            {table.columns.map((c) => (
              <th key={c} scope="col" className="py-2 pl-3 text-right align-bottom text-xs font-medium text-text-dim sm:pl-4">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r) => (
            <tr key={r.region} className="border-b border-surface-seam/60">
              <th scope="row" className="py-2.5 pr-3 text-left font-normal text-text-secondary sm:pr-4">
                {REGION_NAME[r.region]} <span className="font-mono text-xs text-text-dim">{r.region}</span>
              </th>
              {r.cells.map((v, i) => (
                <td
                  key={i}
                  className={`py-2.5 pl-3 text-right whitespace-nowrap sm:pl-4 ${v === null ? "text-text-dim" : `text-text-primary ${valueClass(v)}`}`}
                >
                  {v ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A region table for the WT/TC/ET facts, then a ledger for the rest. */
function TabledFacts({ facts, caption }: { facts: InterpretedFact[]; caption: string }) {
  const { table, rest } = regionTable(facts);
  return (
    <div className="flex flex-col gap-6">
      {table && <RegionTableView table={table} caption={caption} />}
      <FactLedger facts={rest} />
    </div>
  );
}

// --------------------------------------------------------------------- //
// Overview: the key figures strip
// --------------------------------------------------------------------- //

export function KeyFigures({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-5 border-y border-surface-seam py-5 sm:grid-cols-3 md:grid-cols-5">
      {facts.map((f, i) => (
        <div key={i} className="min-w-0">
          <dt className="eyebrow">{f.label}</dt>
          <dd className={`mt-1 text-xl text-text-primary ${valueClass(f.value)}`}>{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}

// --------------------------------------------------------------------- //
// Composition: two share bars (of the whole tumour; within the core)
// --------------------------------------------------------------------- //

interface Segment {
  value: unknown;
  color: string;
  label: string;
}

function finite(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

/** One stacked share bar with its legend. Null if any share is missing. */
function ShareBar({ title, segments }: { title: string; segments: Segment[] }) {
  if (!segments.every((s) => finite(s.value))) return null;
  return (
    <figure className="m-0 break-inside-avoid">
      <figcaption className="text-xs font-medium text-text-dim">{title}</figcaption>
      <div className="mt-2 flex h-3 w-full gap-px overflow-hidden rounded-sm bg-surface-raised" aria-hidden="true">
        {segments.map((s) => (
          <div key={s.label} className={s.color} style={{ width: `${(s.value as number) * 100}%` }} />
        ))}
      </div>
      <ul className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1 p-0 text-sm text-text-secondary">
        {segments.map((s) => (
          <li key={s.label} className="flex list-none items-center gap-2">
            <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${s.color}`} aria-hidden="true" />
            {s.label}
            <span className="font-mono tabular text-text-primary">{formatPercent(s.value as number)}</span>
          </li>
        ))}
      </ul>
    </figure>
  );
}

function CompositionFacts({ facts, fractions }: { facts: InterpretedFact[]; fractions: BurdenBlock }) {
  const wt: Segment[] = [
    { value: fractions.frac_edema_of_wt, color: "bg-data-oedema", label: "Swelling" },
    { value: fractions.frac_enhancing_of_wt, color: "bg-data-enhancing", label: "Enhancing" },
    { value: fractions.frac_necrotic_of_wt, color: "bg-data-necrotic", label: "Necrotic" },
  ];
  const tc: Segment[] = [
    { value: fractions.frac_enhancing_of_tc, color: "bg-data-enhancing", label: "Enhancing" },
    { value: fractions.frac_necrotic_of_tc, color: "bg-data-necrotic", label: "Necrotic" },
  ];
  const wtDrawn = wt.every((s) => finite(s.value));
  const tcDrawn = tc.every((s) => finite(s.value));
  // A fraction a bar already shows is not repeated as a row; anything a bar
  // could not draw (missing data) stays in the ledger.
  const rest = facts.filter(
    (f) => !((wtDrawn && f.label.endsWith(" of WT")) || (tcDrawn && f.label.endsWith(" of TC"))),
  );
  return (
    <div className="flex flex-col gap-6">
      <ShareBar title="Share of the whole marked region" segments={wt} />
      <ShareBar title="Within the solid core" segments={tc} />
      <FactLedger facts={rest} />
    </div>
  );
}

// --------------------------------------------------------------------- //
// Regions: two summary rows, then one row per atlas structure
// --------------------------------------------------------------------- //

// Collapse the list only when it is long enough for that to help.
const REGIONS_SHOWN = 8;
const REGIONS_COLLAPSE_ABOVE = 12;

/** `regions` facts are, in order: two summary facts, then one fact per
 * `report.anatomy.structures` row (same order). The name comes from the
 * fact's label; the bar width and the two percentages from the raw row. */
function RegionsFacts({
  facts,
  structures,
  nInvolved,
}: {
  facts: InterpretedFact[];
  structures: AnatomyStructureRow[];
  nInvolved: number | null;
}) {
  const [showAll, setShowAll] = useState(false);
  const summary = facts.slice(0, 2);
  const rowFacts = facts.slice(2);
  const n = Math.min(rowFacts.length, structures.length);
  const collapsible = n > REGIONS_COLLAPSE_ABOVE;

  return (
    <div className="flex flex-col gap-6">
      <FactLedger facts={summary} />
      {n > 0 && (
        <div>
          {/* report.py lists the top rows by share of the region covered. */}
          {nInvolved !== null && nInvolved > n && (
            <p className="mb-3 text-sm text-text-secondary">
              The {n} regions the tumour covers most, of the {nInvolved} it touches.
            </p>
          )}
          <div className="grid grid-cols-[minmax(0,1fr)_4rem_4.5rem] items-end gap-x-4 border-b sm:grid-cols-[minmax(0,1fr)_5rem_6.5rem] border-surface-seam pb-2 text-xs font-medium text-text-dim">
            <span>Atlas region</span>
            <span className="text-right">% of it</span>
            <span className="text-right">% of the tumour</span>
          </div>
          <ol className="m-0 list-none p-0">
            {Array.from({ length: n }, (_, i) => {
              const s = structures[i];
              const width = finite(s.frac_of_structure) ? Math.min(100, Math.max(0, s.frac_of_structure * 100)) : 0;
              const collapsed = collapsible && !showAll && i >= REGIONS_SHOWN;
              return (
                <li
                  key={i}
                  className={`${collapsed ? "hidden print:grid" : "grid"} grid-cols-[minmax(0,1fr)_4rem_4.5rem] items-center sm:grid-cols-[minmax(0,1fr)_5rem_6.5rem] gap-x-4 border-b border-surface-seam/60 py-2 break-inside-avoid`}
                >
                  <span className="min-w-0">
                    <span className="block text-sm leading-snug text-text-primary">
                      {rowFacts[i].label}
                    </span>
                    <span className="mt-1.5 block h-1 w-full overflow-hidden rounded-full bg-surface-raised" aria-hidden="true">
                      <span className="block h-full rounded-full bg-brand-primary/80" style={{ width: `${width}%` }} />
                    </span>
                  </span>
                  <span className="text-right font-mono tabular text-sm text-text-primary">
                    {formatPercent(s.frac_of_structure)}
                  </span>
                  <span className="text-right font-mono tabular text-sm text-text-secondary">
                    {formatPercent(s.frac_of_tumour)}
                  </span>
                </li>
              );
            })}
          </ol>
          {collapsible && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              aria-expanded={showAll}
              className="btn-secondary mt-3 !px-3 !py-1.5 text-sm print:hidden"
            >
              {showAll ? `Show the first ${REGIONS_SHOWN} only` : `Show all ${n} listed regions`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------- //
// Limits: what the report does not say
// --------------------------------------------------------------------- //

/** One hairline row per limit: what is not claimed, then why. Rendered as
 * `ol > li` (the e2e smoke test counts them against the API). */
function LimitsFacts({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <ol className="m-0 grid list-none gap-x-10 p-0 sm:grid-cols-2">
      {facts.map((f, i) => (
        <li key={i} className="border-t border-surface-seam py-4 break-inside-avoid">
          <p className="text-sm font-semibold text-text-primary first-letter:uppercase">{f.label}</p>
          {f.note && <p className="mt-1 text-sm leading-relaxed text-text-secondary">{f.note}</p>}
        </li>
      ))}
    </ol>
  );
}

// --------------------------------------------------------------------- //
// Dispatcher
// --------------------------------------------------------------------- //

export function SectionFacts({ section, report }: { section: InterpretedSection; report: ReportResponse }) {
  switch (section.id) {
    case "overview":
      return <KeyFigures facts={section.facts} />;
    case "composition":
      return <CompositionFacts facts={section.facts} fractions={report.burden.fractions} />;
    case "regions":
      return (
        <RegionsFacts
          facts={section.facts}
          structures={report.anatomy.structures}
          nInvolved={report.anatomy.n_structures_involved}
        />
      );
    case "limits":
      return <LimitsFacts facts={section.facts} />;
    case "shape":
      return <TabledFacts facts={section.facts} caption="Shape measures by tumour region" />;
    case "connectivity":
      return <TabledFacts facts={section.facts} caption="Connected pieces by tumour region" />;
    default:
      return <FactLedger facts={section.facts} />;
  }
}
