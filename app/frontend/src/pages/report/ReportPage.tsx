// A plain-language, printable report document for one case.
//
// Layout ("reading room" system, see index.css): a document, not a stack of
// cards. A title block (case id, metadata, the basis sentence, the
// disclaimer once), then numbered sections separated by hairlines, a sticky
// contents rail on wide screens, and the raw technical data collapsed at the
// end. Section bodies are laid out by ReportFacts.tsx; the raw panel lives
// in TechnicalData.tsx.
//
// Every sentence and figure comes from `interpretReport`
// (lib/reportInterpretation.ts) as an already-formatted string - this page
// decides placement and typesetting only, never how a number is worded.
//
// Contract with e2e/smoke.mjs (section 10), keep it when restyling:
//  - each section is <section id="section-{id}"> whose FIRST direct <p> is
//    the section's headline (checked on the overview);
//  - the limits section lists its items as `ol > li`;
//  - an <a> whose text includes "Back to viewer" returns to /app?case=<id>.
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ArrowLeft, Info, Printer } from "lucide-react";
import { AppShell } from "../../components/AppShell";
import { fetchReport, type ReportResponse } from "../../api";
import { classifyReportError } from "../../lib/reportStatus";
import { navigateTo } from "../../lib/navigate";
import { splitCode } from "../../lib/reportLayout";
import { type InterpretedSection, interpretReport } from "../../lib/reportInterpretation";
import { SectionFacts } from "./ReportFacts";
import { TechnicalData } from "./TechnicalData";

type LoadStatus = "loading" | "loaded" | "not_found" | "server_error" | "unreachable" | "invalid";

/** The message for each non-loaded status. `loading` needs the case id; the rest fall back to the classified message, which is null only for `unreachable`. */
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

const STATUS_TITLE: Record<Exclude<LoadStatus, "loading" | "loaded">, string> = {
  not_found: "No report for this case yet",
  unreachable: "Can't reach the API",
  server_error: "The report couldn't be loaded",
  invalid: "The report couldn't be read",
};

/** `generated_utc` -> a locale date string; falls back to the raw string if it does not parse. */
function formatGeneratedDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

/** Two-digit section numbers: the report is an ordered document, cited by section. */
function sectionNumber(i: number): string {
  return String(i + 1).padStart(2, "0");
}

// --------------------------------------------------------------------- //
// Print: re-point the colour tokens at a white page. Print-only, so the
// on-screen theme is untouched. (Raw colours are unavoidable here: paper.)
// --------------------------------------------------------------------- //

const PRINT_CSS = `
@media print {
  @page { margin: 14mm; }
  :root {
    --color-surface-page: #ffffff;
    --color-surface-panel: #ffffff;
    --color-surface-raised: #eceef1;
    --color-surface-seam: #d1d5db;
    --color-text-primary: #111111;
    --color-text-secondary: #374151;
    --color-text-dim: #4b5563;
    --color-brand-primary: #1f6f8b;
  }
  html, body, #root { background: #ffffff !important; }
  #root > div > header { display: none !important; }
  #report-doc { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  #report-doc h2 { break-after: avoid-page; }
  #section-technical:not([open]) { display: none; }
}
`;

// --------------------------------------------------------------------- //
// Small pieces
// --------------------------------------------------------------------- //

function BackLink({ caseId, className, children }: { caseId: string; className?: string; children: ReactNode }) {
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
      {children}
    </a>
  );
}

/** Renders `code` spans in a status message as <code>. */
function WithCode({ text }: { text: string }) {
  return (
    <>
      {splitCode(text).map((part, i) =>
        i % 2 === 1 ? (
          <code key={i} className="rounded bg-surface-raised px-1.5 py-0.5 font-mono text-[0.9em] text-text-primary">
            {part}
          </code>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="eyebrow">{label}</dt>
      <dd className={`mt-1 text-sm whitespace-nowrap text-text-primary ${mono ? "font-mono tabular" : ""}`}>{value}</dd>
    </div>
  );
}

function Caveat({ text }: { text: string }) {
  return (
    <aside className="mt-6 rounded-lg border border-surface-seam bg-surface-panel px-4 py-3 break-inside-avoid">
      <p className="flex items-center gap-2 text-xs font-semibold text-text-secondary">
        <Info size={14} aria-hidden="true" className="text-text-dim" />
        Caveat
      </p>
      {text.split("\n").map((line, i) => (
        <p key={i} className="mt-1.5 text-sm leading-relaxed text-text-secondary">
          {line}
        </p>
      ))}
    </aside>
  );
}

// --------------------------------------------------------------------- //
// One section: number + title, the headline (first direct <p>), the
// explanation, the facts, then any caveat.
// --------------------------------------------------------------------- //

function ReportSection({
  section,
  number,
  report,
}: {
  section: InterpretedSection;
  number: string;
  report: ReportResponse;
}) {
  const isOverview = section.id === "overview";
  // The limits section's explanation is the disclaimer, already shown once
  // under the title block; don't print it a second time.
  const explanation = section.explanation && section.explanation !== report.disclaimer ? section.explanation : null;
  const headingId = `heading-${section.id}`;
  return (
    <section id={`section-${section.id}`} aria-labelledby={headingId} className="scroll-mt-6 border-t border-surface-seam py-10">
      <div className="flex items-baseline gap-3">
        <span className="font-mono tabular text-sm text-text-dim">{number}</span>
        <h2 id={headingId} className="font-heading text-xl font-semibold text-text-primary">
          {section.title}
        </h2>
      </div>
      <p
        className={
          isOverview
            ? "mt-4 max-w-[60ch] text-xl leading-snug text-text-primary sm:text-[1.375rem]"
            : "mt-3 max-w-[68ch] text-base leading-relaxed text-text-primary"
        }
      >
        {section.headline}
      </p>
      {!isOverview && explanation && (
        <p className="mt-2 max-w-[68ch] text-sm leading-relaxed text-text-secondary">{explanation}</p>
      )}
      <div className="mt-6">
        <SectionFacts section={section} report={report} />
      </div>
      {isOverview && explanation && (
        <p className="mt-4 max-w-[68ch] text-sm leading-relaxed text-text-secondary">{explanation}</p>
      )}
      {section.caveat && <Caveat text={section.caveat} />}
    </section>
  );
}

// --------------------------------------------------------------------- //
// Contents: a sticky rail on wide screens, a scrolling chip row below.
// --------------------------------------------------------------------- //

interface NavItem {
  id: string;
  number: string;
  title: string;
}

function ContentsRail({ items, activeId, caseId }: { items: NavItem[]; activeId: string | null; caseId: string }) {
  return (
    <div className="sticky top-6 flex flex-col gap-8">
      <nav aria-label="Report contents">
        <p className="eyebrow">Contents</p>
        <ol className="m-0 mt-3 list-none border-l border-surface-seam p-0">
          {items.map((n) => {
            const active = activeId === n.id;
            return (
              <li key={n.id}>
                <a
                  href={`#${n.id}`}
                  aria-current={active ? "location" : undefined}
                  className={`-ml-px flex gap-2.5 border-l py-1.5 pl-3.5 text-[13px] leading-snug transition-colors duration-150 ${
                    active
                      ? "border-brand-primary text-text-primary"
                      : "border-transparent text-text-secondary hover:border-text-dim hover:text-text-primary"
                  }`}
                >
                  <span className={`font-mono tabular text-xs leading-[1.2rem] ${active ? "text-brand-primary" : "text-text-dim"}`}>
                    {n.number}
                  </span>
                  <span>{n.title}</span>
                </a>
              </li>
            );
          })}
        </ol>
      </nav>
      <div className="flex flex-col items-start gap-1 border-t border-surface-seam pt-4 text-[13px]">
        <button
          type="button"
          onClick={() => window.print()}
          className="flex items-center gap-2 rounded-md py-1 text-text-secondary transition-colors duration-150 hover:text-text-primary"
        >
          <Printer size={14} aria-hidden="true" />
          Print report
        </button>
        <BackLink
          caseId={caseId}
          className="flex items-center gap-2 rounded-md py-1 text-text-secondary transition-colors duration-150 hover:text-text-primary"
        >
          <ArrowLeft size={14} aria-hidden="true" />
          Back to viewer
        </BackLink>
      </div>
    </div>
  );
}

function ContentsChips({ items, activeId }: { items: NavItem[]; activeId: string | null }) {
  return (
    <nav aria-label="Report contents" className="-mx-4 mb-2 flex gap-2 overflow-x-auto px-4 pb-3 sm:-mx-6 sm:px-6 lg:hidden print:hidden">
      {items.map((n) => {
        const active = activeId === n.id;
        return (
          <a
            key={n.id}
            href={`#${n.id}`}
            aria-current={active ? "location" : undefined}
            className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1.5 text-[13px] whitespace-nowrap transition-colors duration-150 ${
              active ? "border-brand-primary text-text-primary" : "border-surface-seam text-text-secondary"
            }`}
          >
            <span className="font-mono tabular text-xs text-text-dim">{n.number}</span>
            {n.title}
          </a>
        );
      })}
    </nav>
  );
}

// --------------------------------------------------------------------- //
// Loading and error states
// --------------------------------------------------------------------- //

function Skeleton() {
  const bar = "animate-pulse rounded bg-surface-raised";
  return (
    <div className="mx-auto max-w-[47rem]" aria-hidden="true">
      <div className={`${bar} h-3 w-28`} />
      <div className={`${bar} mt-3 h-8 w-72`} />
      <div className="mt-6 grid grid-cols-2 gap-6 sm:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i}>
            <div className={`${bar} h-2.5 w-16`} />
            <div className={`${bar} mt-2 h-4 w-24`} />
          </div>
        ))}
      </div>
      <div className={`${bar} mt-8 h-14 w-full`} />
      <div className="mt-10 border-t border-surface-seam pt-10">
        <div className={`${bar} h-5 w-40`} />
        <div className={`${bar} mt-4 h-5 w-full`} />
        <div className={`${bar} mt-2 h-5 w-4/5`} />
      </div>
    </div>
  );
}

function StatusView({ status, caseId, message }: { status: LoadStatus; caseId: string; message: string | null }) {
  if (status === "loading") {
    return (
      <div role="status" aria-live="polite">
        <p className="sr-only">{statusMessage(status, caseId, message)}</p>
        <Skeleton />
      </div>
    );
  }
  if (status === "loaded") return null;
  return (
    <div className="mx-auto max-w-[36rem] py-12" role="alert">
      <p className="eyebrow">
        Structured report · <span className="font-mono">{caseId}</span>
      </p>
      <h1 className="mt-2 font-heading text-2xl font-semibold text-text-primary">{STATUS_TITLE[status]}</h1>
      <p className="mt-3 leading-relaxed text-text-secondary">
        <WithCode text={statusMessage(status, caseId, message)} />
      </p>
      <BackLink caseId={caseId} className="btn-secondary mt-6">
        <ArrowLeft size={16} aria-hidden="true" />
        Back to viewer
      </BackLink>
    </div>
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

  // Restore the previous document title on unmount so navigating away (a
  // no-router SPA, see main.tsx) doesn't leave a stale title behind.
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

  // Every section in page order, then the technical-data panel.
  const navItems: NavItem[] = useMemo(() => {
    if (!interpreted) return [];
    const items = interpreted.sections.map((s, i) => ({ id: `section-${s.id}`, number: sectionNumber(i), title: s.title }));
    items.push({ id: "section-technical", number: sectionNumber(items.length), title: "Full technical data" });
    return items;
  }, [interpreted]);

  // Mark the contents entry for the section nearest the top of the viewport.
  // Skipped where IntersectionObserver does not exist (tests).
  useEffect(() => {
    if (navItems.length === 0 || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        visible.sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        setActiveId(visible[0].target.id);
      },
      { rootMargin: "0px 0px -65% 0px" },
    );
    for (const item of navItems) {
      const el = document.getElementById(item.id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [navItems]);

  return (
    <AppShell>
      <style>{PRINT_CSS}</style>
      <div className="mx-auto max-w-[1180px] px-4 py-10 sm:px-6 md:py-14">
        {status !== "loaded" && <StatusView status={status} caseId={caseId} message={message} />}

        {status === "loaded" && report && interpreted && (
          <div className="lg:grid lg:grid-cols-[12.5rem_minmax(0,47rem)] lg:justify-center lg:gap-16 print:block">
            <aside className="hidden lg:block print:hidden">
              <ContentsRail items={navItems} activeId={activeId} caseId={caseId} />
            </aside>

            <article id="report-doc" className="min-w-0">
              {/* Title block */}
              <div className="pb-8">
                <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-4">
                  <div className="min-w-0">
                    <p className="eyebrow">Structured report</p>
                    <h1 className="mt-1.5 font-mono text-[1.75rem] leading-tight font-semibold break-all text-text-primary sm:text-3xl">
                      {interpreted.caseId}
                    </h1>
                  </div>
                  <div className="flex shrink-0 gap-2 print:hidden">
                    <BackLink caseId={caseId} className="btn-secondary !px-3 !py-1.5 text-sm">
                      <ArrowLeft size={15} aria-hidden="true" />
                      Back to viewer
                    </BackLink>
                    <button type="button" onClick={() => window.print()} className="btn-secondary !px-3 !py-1.5 text-sm">
                      <Printer size={15} aria-hidden="true" />
                      Print
                    </button>
                  </div>
                </div>
                <dl className="mt-6 flex flex-wrap gap-x-10 gap-y-4">
                  <Meta label="Generated" value={formatGeneratedDate(report.generated_utc)} mono />
                  <Meta label="Measured on" value={interpreted.basis === "prediction" ? "Model segmentation" : "Reference label"} />
                  <Meta label="Atlas" value={`${report.anatomy.atlas.name} ${report.anatomy.atlas.version}`} mono />
                  <Meta label="Report schema" value={`v${report.report_version}`} mono />
                </dl>
                <p className="mt-5 max-w-[68ch] text-sm leading-relaxed text-text-secondary">{interpreted.basisSentence}</p>
                <p className="mt-5 flex gap-2.5 rounded-lg border border-surface-seam bg-surface-panel px-4 py-3 text-sm leading-relaxed text-text-secondary">
                  <Info size={16} aria-hidden="true" className="mt-0.5 shrink-0 text-text-dim" />
                  <span>{report.disclaimer}</span>
                </p>
              </div>

              <ContentsChips items={navItems} activeId={activeId} />

              {interpreted.sections.map((section, i) => (
                <ReportSection key={section.id} section={section} number={sectionNumber(i)} report={report} />
              ))}

              <TechnicalData report={report} number={sectionNumber(interpreted.sections.length)} />
            </article>
          </div>
        )}
      </div>
    </AppShell>
  );
}
