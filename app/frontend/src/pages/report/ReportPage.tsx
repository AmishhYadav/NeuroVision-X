// A plain-language, printable report document.
//
// Light "paper" register, deliberately the SAME tokens as the landing page
// (landing-bg / landing-text / etc.) rather than the dark clinical viewer
// chrome (App.tsx). A report is something a person reads and possibly
// prints and hands to someone else - a document, not an instrument - so it
// gets the document register, same as CLAUDE.md's Page Theme Lock allows one
// deliberately separate register per surface, not a random mix within one.
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
import { Fragment, useEffect, useMemo, useState } from "react";
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
  type InterpretedReport,
  type InterpretedSection,
  interpretReport,
} from "../../lib/reportInterpretation";

type LoadStatus = "loading" | "loaded" | "not_found" | "server_error" | "unreachable" | "invalid";

/** The message shown for each non-loaded status. `loading` needs the case id; the rest fall back to the classified message, which is null only for `unreachable`. */
function statusMessage(status: LoadStatus, caseId: string, classifiedMessage: string | null): string {
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
// Nav
// --------------------------------------------------------------------- //

function BackLink({ caseId, className }: { caseId: string; className?: string }) {
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

function Nav({ caseId, basis }: { caseId: string; basis: InterpretedReport["basis"] | null }) {
  return (
    <nav className="sticky top-0 z-10 border-b border-landing-seam bg-landing-bg-raised/90 backdrop-blur print:hidden">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-3">
        <BackLink
          caseId={caseId}
          className="text-sm text-landing-text-secondary hover:text-landing-text"
        />
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-landing-text-dim">{caseId}</span>
          {basis && (
            <span className="rounded-full border border-landing-seam px-2.5 py-0.5 font-mono text-[11px] uppercase tracking-wide text-landing-text-secondary">
              {basis === "prediction" ? "Model segmentation" : "Reference label"}
            </span>
          )}
          <button
            type="button"
            onClick={() => window.print()}
            className="rounded-full border border-landing-seam px-3 py-1 font-mono text-xs text-landing-text hover:border-landing-text-dim"
          >
            Print
          </button>
        </div>
      </div>
    </nav>
  );
}

// --------------------------------------------------------------------- //
// Facts rendering - one dispatcher per section id (see spec in the task
// that produced this file). Every branch renders strings from
// InterpretedFact; only the two "bar" branches below reach past that into
// raw report numbers, and only to size a width.
// --------------------------------------------------------------------- //

function FactsGrid({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-6 gap-y-2">
      {facts.map((f, i) => (
        <Fragment key={i}>
          <dt className="text-sm">
            {f.label}
            {f.note && <div className="mt-0.5 text-xs text-landing-text-dim">{f.note}</div>}
          </dt>
          <dd className="tabular text-right font-mono text-sm">{f.value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

/** The three shares of WT, read raw so the bar segment widths are exact percentages rather than re-parsed from a formatted string. Renders nothing if any of the three is missing or non-finite. */
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
      <div className="flex h-4 w-full overflow-hidden rounded-sm">
        {segments.map((s) => (
          <div
            key={s.key}
            className={s.color}
            style={{ width: `${(s.value as number) * 100}%` }}
            title={`${s.label}: ${formatPercent(s.value as number)}`}
          />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-4">
        {segments.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5 text-xs text-landing-text-secondary">
            <span className={`inline-block h-2.5 w-2.5 rounded-sm ${s.color}`} />
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
            typeof structure.frac_of_structure === "number" ? structure.frac_of_structure * 100 : 0;
          const width = Math.min(100, Math.max(0, rawWidth));
          return (
            <div key={i}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm">{fact.label}</span>
                <span className="font-mono text-xs text-landing-text-dim">
                  {formatPercent(structure.frac_of_structure)} of region &middot;{" "}
                  {formatPercent(structure.frac_of_tumour)} of tumour
                </span>
              </div>
              <div className="mt-1 h-2 w-full overflow-hidden rounded-sm bg-landing-seam">
                <div className="h-2 bg-landing-accent/70" style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

/** Each limit is `<strong>label</strong>` followed by its note as the body - `value` carries nothing for this section. */
function LimitsFacts({ facts }: { facts: InterpretedFact[] }) {
  if (facts.length === 0) return null;
  return (
    <ol className="flex flex-col gap-3">
      {facts.map((f, i) => (
        <li key={i} className="text-sm leading-relaxed">
          <strong>{f.label}</strong>
          {f.note && <div className="mt-1 text-landing-text-secondary">{f.note}</div>}
        </li>
      ))}
    </ol>
  );
}

function SectionFacts({ section, report }: { section: InterpretedSection; report: ReportResponse }) {
  if (section.id === "composition") {
    return (
      <>
        <CompositionBar fractions={report.burden.fractions} />
        <FactsGrid facts={section.facts} />
      </>
    );
  }
  if (section.id === "regions") {
    return <RegionsFacts facts={section.facts} structures={report.anatomy.structures} />;
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

function BurdenBlockDl({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <div>
      <div className="font-condensed text-xs uppercase tracking-wide text-landing-text-dim">{title}</div>
      <dl className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 font-mono text-xs">
        {Object.entries(block).map(([k, v]) => (
          <Fragment key={k}>
            <dt>{burdenLabel(k)}</dt>
            <dd className="text-right">{formatBurdenValue(k, v)}</dd>
          </Fragment>
        ))}
      </dl>
    </div>
  );
}

function GeometryBlockDl({ title, block }: { title: string; block: BurdenBlock | undefined }) {
  if (isEmptyBlock(block)) return null;
  return (
    <div>
      <div className="font-condensed text-xs uppercase tracking-wide text-landing-text-dim">{title}</div>
      <dl className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 font-mono text-xs">
        {Object.entries(block).map(([k, v]) => (
          <Fragment key={k}>
            <dt>{geometryLabel(k)}</dt>
            <dd className="text-right">{formatGeometryValue(k, typeof v === "number" ? v : null)}</dd>
          </Fragment>
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
    <dl className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 font-mono text-xs">
      {entries.map(([k, v]) => (
        <Fragment key={k}>
          <dt>{k}</dt>
          <dd className="text-right break-all">{flattenProvenanceValue(v)}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function TechnicalData({ report }: { report: ReportResponse }) {
  return (
    <details className="mx-auto mt-4 max-w-5xl px-6 pb-16">
      <summary className="print:hidden cursor-pointer font-mono text-xs uppercase tracking-wide text-landing-text-dim">
        Full technical data
      </summary>
      <div className="mt-6 grid gap-6 sm:grid-cols-2">
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
        <div>
          <div className="font-condensed text-xs uppercase tracking-wide text-landing-text-dim">
            Eloquence citation
          </div>
          <p className="mt-1 font-mono text-xs leading-relaxed">{report.eloquence.citation}</p>
        </div>
        <div>
          <div className="font-condensed text-xs uppercase tracking-wide text-landing-text-dim">
            Provenance
          </div>
          <ProvenanceDl provenance={report.provenance} />
        </div>
      </div>
    </details>
  );
}

// --------------------------------------------------------------------- //
// Page
// --------------------------------------------------------------------- //

export function ReportPage({ caseId }: { caseId: string }) {
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [message, setMessage] = useState<string | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);

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

  const interpreted = useMemo(() => (report ? interpretReport(report) : null), [report]);

  const overview = interpreted?.sections.find((s) => s.id === "overview") ?? null;
  const remainingSections = interpreted?.sections.filter((s) => s.id !== "overview") ?? [];

  return (
    <div className="min-h-screen bg-landing-bg font-sans text-landing-text">
      <Nav caseId={caseId} basis={interpreted?.basis ?? null} />

      {status !== "loaded" && (
        <div className="mx-auto max-w-3xl px-6 py-24 text-center">
          <p className="text-base leading-relaxed text-landing-text-secondary">
            {statusMessage(status, caseId, message)}
          </p>
        </div>
      )}

      {status === "loaded" && report && interpreted && (
        <>
          {/* Masthead */}
          <div className="mx-auto max-w-3xl px-6 pt-10">
            <span className="font-mono text-xs uppercase tracking-[0.14em] text-landing-accent">
              Plain-language report
            </span>
            <h1 className="mt-2 font-condensed text-4xl tracking-tight md:text-5xl">
              {interpreted.caseId}
            </h1>
            <p className="mt-2 font-mono text-xs text-landing-text-dim">
              Generated {formatGeneratedDate(report.generated_utc)} &middot; report schema v
              {report.report_version}
            </p>
            <p className="mt-4 text-landing-text-secondary">{interpreted.basisSentence}</p>
            <div className="mt-4 border border-landing-seam bg-landing-bg-raised p-4 text-sm leading-relaxed">
              {report.disclaimer}
            </div>
          </div>

          {/* At a glance - the overview section as a hero card */}
          {overview && (
            <div className="mx-auto max-w-3xl px-6 py-10">
              <p className="font-condensed text-2xl leading-snug md:text-3xl">{overview.headline}</p>
              {overview.facts.length > 0 && (
                <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-5">
                  {overview.facts.map((f, i) => (
                    <div key={i}>
                      <div className="font-mono text-[11px] uppercase tracking-wide text-landing-text-dim">
                        {f.label}
                      </div>
                      <div className="tabular font-condensed text-2xl">{f.value}</div>
                    </div>
                  ))}
                </div>
              )}
              <p className="mt-6 text-sm text-landing-text-secondary">{overview.explanation}</p>
            </div>
          )}

          {/* Body: a sticky contents rail (anchors into the sections below,
              offset under the sticky nav via scroll-mt) plus the sections
              themselves. */}
          <div className="mx-auto grid max-w-5xl gap-10 px-6 lg:grid-cols-[200px_minmax(0,1fr)]">
            <div className="hidden lg:block print:hidden">
              <nav className="sticky top-20 flex flex-col gap-2 text-sm">
                {remainingSections.map((s) => (
                  <a
                    key={s.id}
                    href={`#section-${s.id}`}
                    className="text-landing-text-secondary hover:text-landing-text"
                  >
                    {s.title}
                  </a>
                ))}
              </nav>
            </div>

            <div>
              {remainingSections.map((section, idx) => (
                <section
                  id={`section-${section.id}`}
                  key={section.id}
                  className={`scroll-mt-24 ${idx === 0 ? "" : "mt-8 border-t border-landing-seam pt-8"}`}
                >
                  <h2 className="font-condensed text-2xl">{section.title}</h2>
                  <p className="mt-2 text-lg leading-relaxed">{section.headline}</p>
                  <p className="mt-2 text-sm leading-relaxed text-landing-text-secondary">
                    {section.explanation}
                  </p>
                  <div className="mt-4">
                    <SectionFacts section={section} report={report} />
                  </div>
                  {section.caveat && (
                    <div className="mt-4 border-l-2 border-landing-seam pl-4 text-xs leading-relaxed text-landing-text-dim">
                      <div className="font-mono text-[11px] uppercase tracking-wide">Caveat</div>
                      {section.caveat.split("\n").map((line, i) => (
                        <p key={i} className="mt-1">
                          {line}
                        </p>
                      ))}
                    </div>
                  )}
                </section>
              ))}
            </div>
          </div>

          <TechnicalData report={report} />

          {/* Footer - the disclaimer again, plus a second way back for anyone
              who scrolled past the sticky nav (and, on print, the only way
              back, since the nav itself is print:hidden). */}
          <footer className="border-t border-landing-seam px-6 py-10">
            <div className="mx-auto flex max-w-3xl flex-col items-center gap-3 text-center">
              <p className="font-mono text-xs text-landing-text-dim">{report.disclaimer}</p>
              <BackLink caseId={caseId} className="text-sm text-landing-text-secondary hover:text-landing-text" />
            </div>
          </footer>
        </>
      )}
    </div>
  );
}
