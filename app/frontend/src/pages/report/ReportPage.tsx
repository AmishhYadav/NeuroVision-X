// A plain-language, printable report document.
//
// Stitch "Structured Report" look: the app's dark glass-panel tokens inside
// the shared AppShell, a two-column layout (report + sticky section nav).
// A small @media print block below swaps the tokens to a white page so a
// printed copy stays readable.
//
// Almost every value rendered here comes from `interpretReport` as an
// already-formatted STRING (InterpretedFact.value), never a raw number this
// component reformats itself - `reportInterpretation.ts` is the single place
// that decides how a number becomes a plain-language fact, and it is owned
// by another agent concurrently. The two exceptions are the composition bar
// and the regions bar list, which need the underlying fraction as a *number*
// to size a `width` style - those two read `report.burden.fractions` /
// `report.anatomy.structures` directly (never re-deriving a label from them;
// labels still come from the matching InterpretedFact).
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { AppShell } from "../../components/AppShell";
import {
  type AnatomyStructureRow,
  type BurdenBlock,
  type ReportProvenance,
  type ReportResponse,
  fetchReport,
} from "../../api";
import {
  burdenLabel,
  formatBurdenValue,
  formatGeometryValue,
  formatPercent,
  geometryLabel,
} from "../../lib/report";
import { classifyReportError } from "../../lib/reportStatus";
import { navigateTo } from "../../lib/navigate";
import {
  type InterpretedFact,
  type InterpretedSection,
  interpretReport,
} from "../../lib/reportInterpretation";

type LoadStatus =
  | "loading"
  | "loaded"
  | "not_found"
  | "server_error"
  | "unreachable"
  | "invalid";

/** The message shown for each non-loaded status. `loading` needs the case id; the rest fall back to the classified message, which is null only for `unreachable`. */
function statusMessage(
  status: LoadStatus,
  caseId: string,
  classifiedMessage: string | null,
): string {
  switch (status) {
    case "loading":
      return `Loading report for ${caseId}…`;
    case "not_found":
      return "No report has been generated for this case — run `scripts/report.py`.";
    case "unreachable":
      return "No response from the API. Start it with `uvicorn app.backend.main:app --reload`.";
    default:
      return classifiedMessage ?? "Failed to load report.";
  }
}

/** `generated_utc` -> a locale date string; falls back to the raw string if it does not parse (an older report, a clock skew, anything that is not valid ISO). */
function formatGeneratedDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function isEmptyBlock(block: BurdenBlock | undefined): block is undefined {
  return !block || Object.keys(block).length === 0;
}

// --------------------------------------------------------------------- //
// Print: re-point the colour tokens at a white page. Print-only, so the
// on-screen theme is untouched. (Raw colours are unavoidable here: paper.)
// --------------------------------------------------------------------- //

const PRINT_CSS = `
@media print {
  :root {
    --color-surface-page: #ffffff;
    --color-surface-panel: #ffffff;
    --color-surface-raised: #f3f4f6;
    --color-surface-seam: #d1d5db;
    --color-text-primary: #111111;
    --color-text-secondary: #374151;
    --color-text-dim: #4b5563;
  }
  html, body, #root { background: #ffffff !important; }
  body header { display: none !important; }
  .glass-panel { background: #ffffff !important; backdrop-filter: none !important; border-color: #d1d5db !important; }
}
`;

// --------------------------------------------------------------------- //
// Actions (back / print)
// --------------------------------------------------------------------- //

function BackLink({
  caseId,
  className,
}: {
  caseId: string;
  className?: string;
}) {
  const href = `/app?case=${encodeURIComponent(caseId)}`;
  return (
    <a
      href={href}
      onClick={(e) => {
        e.preventDefault();
        navigateTo(href);
      }}
      className={className}
    >
      &larr; Back to viewer
    </a>
  );
}

const SMALL_BTN = "btn-secondary !px-3 !py-1.5 text-xs justify-center";

// --------------------------------------------------------------------- //
// Facts rendering - one dispatcher per section id. Every branch renders
// strings from InterpretedFact; only the two "bar" branches below reach
// past that into raw report numbers, and only to size a width.
// --------------------------------------------------------------------- //

/** KPI tiles: eyebrow label, mono value, optional note underneath. */
function FactsGrid({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {facts.map((f, i) => (
        <div
          key={i}
          className="rounded-lg border border-surface-seam bg-surface-raised/40 p-3"
        >
          <dt className="eyebrow">{f.label}</dt>
          <dd className="tabular mt-1 break-words font-mono text-xl text-text-primary">
            {f.value}
          </dd>
          {f.note && <div className="mt-1 text-xs text-text-dim">{f.note}</div>}
        </div>
      ))}
    </dl>
  );
}

/** The three shares of WT, read raw so the bar segment widths are exact percentages rather than re-parsed from a formatted string. Renders nothing if any of the three is missing or non-finite. */
function CompositionBar({ fractions }: { fractions: BurdenBlock }) {
  const segments: {
    key: string;
    value: unknown;
    color: string;
    label: string;
  }[] = [
    {
      key: "edema",
      value: fractions.frac_edema_of_wt,
      color: "bg-data-oedema",
      label: "Swelling",
    },
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
  const allFinite = segments.every(
    (s) => typeof s.value === "number" && Number.isFinite(s.value),
  );
  if (!allFinite) return null;

  return (
    <div className="mb-4">
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-seam">
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
          <div
            key={s.key}
            className="flex items-center gap-1.5 font-mono text-xs text-text-secondary"
          >
            <span className={`inline-block h-2 w-2 rounded-full ${s.color}`} />
            {s.label} {formatPercent(s.value as number)}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * `regions` facts are, in order: two summary facts ("Regions touched",
 * "Unlabelled share"), then one fact per `report.anatomy.structures` row -
 * same order, offset by those two. The bar width and the two percentages
 * come from the matching raw structure row (frac_of_structure,
 * frac_of_tumour); the structure NAME comes from the fact's own label so
 * this component never re-derives a name from the raw atlas row itself.
 */
function RegionsFacts({
  facts,
  structures,
}: {
  facts: InterpretedFact[];
  structures: AnatomyStructureRow[];
}) {
  const summaryFacts = facts.slice(0, 2);
  const rowFacts = facts.slice(2);
  const n = Math.min(rowFacts.length, structures.length);

  return (
    <>
      <FactsGrid facts={summaryFacts} />
      <div className="mt-4 flex flex-col gap-3">
        {Array.from({ length: n }, (_, i) => {
          const fact = rowFacts[i];
          const structure = structures[i];
          const rawWidth =
            typeof structure.frac_of_structure === "number"
              ? structure.frac_of_structure * 100
              : 0;
          const width = Math.min(100, Math.max(0, rawWidth));
          return (
            <div key={i}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm text-text-primary">{fact.label}</span>
                <span className="font-mono text-xs text-text-dim">
                  {formatPercent(structure.frac_of_structure)} of region
                  &middot; {formatPercent(structure.frac_of_tumour)} of tumour
                </span>
              </div>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-surface-seam">
                <div
                  className="h-2 rounded-full bg-brand-primary/70"
                  style={{ width: `${width}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

/** "What this report does not claim": each limit is a small card, label as mono title and its note as the reason - `value` carries nothing for this section. */
function LimitsFacts({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <ol className="grid gap-3 sm:grid-cols-2">
      {facts.map((f, i) => (
        <li key={i} className="glass-panel p-3">
          <strong className="font-mono text-sm font-normal text-text-primary">
            {f.label}
          </strong>
          {f.note && (
            <div className="mt-1 text-xs leading-relaxed text-text-secondary">
              {f.note}
            </div>
          )}
        </li>
      ))}
    </ol>
  );
}

function SectionFacts({
  section,
  report,
}: {
  section: InterpretedSection;
  report: ReportResponse;
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
      />
    );
  }
  if (section.id === "limits") {
    return <LimitsFacts facts={section.facts} />;
  }
  return <FactsGrid facts={section.facts} />;
}

// --------------------------------------------------------------------- //
// Full technical data - the raw-key panel, collapsed by default. Kept
// `print:hidden` on just the `<summary>` (not the whole `<details>`): the
// element defaults to closed, so on print it contributes nothing either way
// - hiding only the summary chrome removes the toggle affordance from the
// printed page without needing to force the section open ("expand nothing").
// --------------------------------------------------------------------- //

/** One zebra-striped mono row of a key/value table. */
function KvRow({
  k,
  children,
  breakAll,
}: {
  k: string;
  children: ReactNode;
  breakAll?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 px-2 py-1 odd:bg-surface-raised/30">
      <dt className="text-text-secondary">{k}</dt>
      <dd
        className={`tabular text-right text-text-primary ${breakAll ? "break-all" : ""}`}
      >
        {children}
      </dd>
    </div>
  );
}

function BlockTitle({ children }: { children: ReactNode }) {
  return <div className="eyebrow">{children}</div>;
}

function BurdenBlockDl({
  title,
  block,
}: {
  title: string;
  block: BurdenBlock | undefined;
}) {
  if (isEmptyBlock(block)) return null;
  return (
    <div>
      <BlockTitle>{title}</BlockTitle>
      <dl className="mt-1 rounded-lg border border-surface-seam py-1 font-mono text-xs">
        {Object.entries(block).map(([k, v]) => (
          <KvRow key={k} k={burdenLabel(k)}>
            {formatBurdenValue(k, v)}
          </KvRow>
        ))}
      </dl>
    </div>
  );
}

function GeometryBlockDl({
  title,
  block,
}: {
  title: string;
  block: BurdenBlock | undefined;
}) {
  if (isEmptyBlock(block)) return null;
  return (
    <div>
      <BlockTitle>{title}</BlockTitle>
      <dl className="mt-1 rounded-lg border border-surface-seam py-1 font-mono text-xs">
        {Object.entries(block).map(([k, v]) => (
          <KvRow key={k} k={geometryLabel(k)}>
            {formatGeometryValue(k, typeof v === "number" ? v : null)}
          </KvRow>
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
  const entries = Object.entries(
    provenance as unknown as Record<string, unknown>,
  );
  return (
    <dl className="mt-1 rounded-lg border border-surface-seam py-1 font-mono text-xs">
      {entries.map(([k, v]) => (
        <KvRow key={k} k={k} breakAll>
          {flattenProvenanceValue(v)}
        </KvRow>
      ))}
    </dl>
  );
}

function TechnicalData({ report }: { report: ReportResponse }) {
  return (
    <details id="section-technical" className="glass-panel scroll-mt-24 p-6">
      <summary className="eyebrow cursor-pointer select-none print:hidden">
        Full technical data
      </summary>
      <div className="mt-6 grid gap-6 sm:grid-cols-2">
        <BurdenBlockDl title="Volumes" block={report.burden.volumes} />
        <BurdenBlockDl title="Fractions" block={report.burden.fractions} />
        <BurdenBlockDl title="Shape" block={report.burden.shape} />
        <BurdenBlockDl
          title="Multifocality"
          block={report.burden.multifocality}
        />
        <BurdenBlockDl title="Laterality" block={report.burden.laterality} />
        <BurdenBlockDl title="Centroid" block={report.burden.centroid} />
        <BurdenBlockDl title="Other (burden)" block={report.burden.other} />
        {report.geometry && (
          <>
            <GeometryBlockDl
              title="Shape (geometric)"
              block={report.geometry.shape}
            />
            <GeometryBlockDl title="Extent" block={report.geometry.extent} />
            <GeometryBlockDl title="Rim" block={report.geometry.rim} />
            <GeometryBlockDl
              title="Other (geometry)"
              block={report.geometry.other}
            />
          </>
        )}
        <div>
          <BlockTitle>Eloquence citation</BlockTitle>
          <p className="mt-1 font-mono text-xs leading-relaxed text-text-secondary">
            {report.eloquence.citation}
          </p>
        </div>
        <div>
          <BlockTitle>Provenance</BlockTitle>
          <ProvenanceDl provenance={report.provenance} />
        </div>
      </div>
    </details>
  );
}

// --------------------------------------------------------------------- //
// One report section as a glass card: teal dot + numbered title, then the
// headline / explanation / facts / caveat the interpretation supplies.
// --------------------------------------------------------------------- //

function SectionCard({
  section,
  index,
  report,
}: {
  section: InterpretedSection;
  index: number;
  report: ReportResponse;
}) {
  const isOverview = section.id === "overview";
  return (
    <section
      id={`section-${section.id}`}
      className="glass-panel scroll-mt-24 p-6"
    >
      <div className="flex items-center gap-3">
        <span
          className="h-2 w-2 shrink-0 rounded-full bg-brand-teal"
          aria-hidden="true"
        />
        <h2 className="font-heading text-xl">
          {index + 1}. {section.title}
        </h2>
      </div>
      <p
        className={`mt-3 leading-relaxed text-text-primary ${isOverview ? "font-heading text-lg" : "text-base"}`}
      >
        {section.headline}
      </p>
      {!isOverview && (
        <p className="mt-2 text-sm leading-relaxed text-text-secondary">
          {section.explanation}
        </p>
      )}
      <div className="mt-4">
        <SectionFacts section={section} report={report} />
      </div>
      {isOverview && (
        <p className="mt-4 text-sm leading-relaxed text-text-secondary">
          {section.explanation}
        </p>
      )}
      {section.caveat && (
        <div className="mt-4 border-l-2 border-surface-seam pl-3 text-xs leading-relaxed text-text-dim">
          <div className="eyebrow">Caveat</div>
          {section.caveat.split("\n").map((line, i) => (
            <p key={i} className="mt-1">
              {line}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}

// --------------------------------------------------------------------- //
// Page
// --------------------------------------------------------------------- //

export function ReportPage({ caseId }: { caseId: string }) {
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);

  // Restore the previous document title on unmount so navigating away (this
  // is a no-router SPA - see main.tsx) doesn't leave a stale "Report · ..."
  // title behind on whatever renders next.
  useEffect(() => {
    const previousTitle = document.title;
    document.title = `Report · ${caseId}`;
    return () => {
      document.title = previousTitle;
    };
  }, [caseId]);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setMessage(null);
    setReport(null);
    fetchReport(caseId, controller.signal)
      .then((r) => {
        setReport(r);
        setStatus("loaded");
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const classified = classifyReportError(err);
        setStatus(classified.status);
        setMessage(classified.message);
      });
    return () => controller.abort();
  }, [caseId]);

  const interpreted = useMemo(
    () => (report ? interpretReport(report) : null),
    [report],
  );

  // Every section the page renders, in order: the overview first, then the
  // rest, then the collapsed technical-data panel. Drives the numbered nav.
  const navItems = useMemo(
    () =>
      interpreted
        ? [
            ...interpreted.sections.map((s) => ({
              id: `section-${s.id}`,
              title: s.title,
            })),
            { id: "section-technical", title: "Full technical data" },
          ]
        : [],
    [interpreted],
  );

  // Highlight the nav entry for whichever section is nearest the top of the
  // viewport. Skipped where IntersectionObserver does not exist (tests).
  useEffect(() => {
    if (navItems.length === 0 || typeof IntersectionObserver === "undefined")
      return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        visible.sort(
          (a, b) => a.boundingClientRect.top - b.boundingClientRect.top,
        );
        setActiveId(visible[0].target.id);
      },
      { rootMargin: "-80px 0px -60% 0px" },
    );
    for (const item of navItems) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [navItems]);

  const basisChip = interpreted
    ? interpreted.basis === "prediction"
      ? "Model segmentation"
      : "Reference label"
    : null;

  // Mobile: horizontal chip row.
  const mobileNav = (
    <nav
      aria-label="Report structure"
      className="-mx-4 mb-6 flex gap-2 overflow-x-auto px-4 pb-1 sm:-mx-6 sm:px-6 lg:hidden print:hidden"
    >
      {navItems.map((n, i) => (
        <a
          key={n.id}
          href={`#${n.id}`}
          className={`chip shrink-0 ${activeId === n.id ? "border-brand-teal/60 text-brand-teal" : ""}`}
        >
          {i + 1}. {n.title}
        </a>
      ))}
    </nav>
  );

  // Desktop: sticky right column.
  const sideNav = (
    <aside className="hidden lg:block print:hidden">
      <div className="sticky top-20 flex flex-col gap-4">
        <div className="glass-panel p-4">
          <div className="eyebrow">Report structure</div>
          <nav
            aria-label="Report structure"
            className="mt-3 flex flex-col gap-1"
          >
            {navItems.map((n, i) => (
              <a
                key={n.id}
                href={`#${n.id}`}
                aria-current={activeId === n.id ? "true" : undefined}
                className={`rounded-lg px-2 py-1.5 text-sm transition-colors ${
                  activeId === n.id
                    ? "bg-brand-teal/10 text-brand-teal"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                <span className="mr-2 font-mono text-xs">{i + 1}.</span>
                {n.title}
              </a>
            ))}
          </nav>
        </div>
        <div className="flex flex-col gap-2">
          <button
            type="button"
            onClick={() => window.print()}
            className={SMALL_BTN}
          >
            Print
          </button>
          <BackLink caseId={caseId} className={SMALL_BTN} />
        </div>
      </div>
    </aside>
  );

  return (
    <AppShell>
      <style>{PRINT_CSS}</style>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        {status !== "loaded" && (
          <div className="mx-auto max-w-3xl">
            <div className="glass-panel p-6 text-center">
              <p className="text-base leading-relaxed text-text-secondary">
                {statusMessage(status, caseId, message)}
              </p>
            </div>
            <div className="mt-4 text-center">
              <BackLink
                caseId={caseId}
                className="text-sm text-text-secondary hover:text-text-primary"
              />
            </div>
          </div>
        )}

        {status === "loaded" && report && interpreted && (
          <>
            {mobileNav}
            <div className="lg:grid lg:grid-cols-[minmax(0,860px)_18rem] lg:justify-center lg:gap-8">
              <div className="flex min-w-0 flex-col gap-6">
                {/* Header block */}
                <div className="glass-panel p-6">
                  <div className="eyebrow">Structured report</div>
                  <h1 className="mt-2 break-words font-heading text-3xl">
                    {interpreted.caseId}
                  </h1>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <span className="chip">
                      Generated {formatGeneratedDate(report.generated_utc)}
                    </span>
                    <span className="chip">
                      {report.anatomy.atlas.name} {report.anatomy.atlas.version}
                    </span>
                    {basisChip && <span className="chip">{basisChip}</span>}
                    <span className="chip">
                      schema v{report.report_version}
                    </span>
                  </div>
                  <p className="mt-4 text-sm text-text-secondary">
                    {interpreted.basisSentence}
                  </p>
                </div>

                {/* Disclaimer: always visible, directly under the header. */}
                <div className="rounded-lg border border-gate-caution/40 bg-gate-caution/10 p-3 text-xs text-gate-caution">
                  {report.disclaimer}
                </div>

                {interpreted.sections.map((section, idx) => (
                  <SectionCard
                    key={section.id}
                    section={section}
                    index={idx}
                    report={report}
                  />
                ))}

                <TechnicalData report={report} />

                {/* Footer - the disclaimer again, plus a second way back (on
                    print, the only one, since the nav is print:hidden). */}
                <footer className="flex flex-col items-center gap-3 py-4 text-center">
                  <p className="font-mono text-xs text-text-dim">
                    {report.disclaimer}
                  </p>
                  <BackLink
                    caseId={caseId}
                    className="text-sm text-text-secondary hover:text-text-primary"
                  />
                </footer>
              </div>
              {sideNav}
            </div>
          </>
        )}
      </div>
    </AppShell>
  );
}
