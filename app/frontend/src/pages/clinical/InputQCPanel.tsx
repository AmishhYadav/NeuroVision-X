import type { GateSeverity, InputQCFinding, InputQCReportJson } from "../../api";

const VERDICT_CLASS: Record<GateSeverity, string> = {
  ok: "text-brand-teal",
  warn: "text-gate-caution",
  refuse: "text-gate-refuse",
};
const DOT_CLASS: Record<GateSeverity, string> = {
  ok: "bg-brand-teal",
  warn: "bg-gate-caution",
  refuse: "bg-gate-refuse",
};

/** One input-QC report (before or after registration): verdict chip plus every finding. */
export function InputQCPanel({ title, report }: { title: string; report: InputQCReportJson }) {
  const notOk = report.findings.filter((f) => f.severity !== "ok");
  const ok = report.findings.filter((f) => f.severity === "ok");
  const renderFinding = (f: InputQCFinding, i: number) => (
    <li key={i} className="flex items-start gap-3">
      <span
        className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${DOT_CLASS[f.severity]}`}
        role="img"
        aria-label={f.severity}
      />
      <div className="min-w-0">
        <p className="font-mono text-xs text-text-primary">{f.check}</p>
        <p className="text-xs leading-relaxed text-text-secondary">{f.message}</p>
      </div>
    </li>
  );

  return (
    <section className="glass-panel p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-heading text-base font-semibold">{title}</h2>
        <span className={`chip ${VERDICT_CLASS[report.verdict]}`}>{report.verdict}</span>
      </div>
      {report.findings.length === 0 ? (
        <p className="mt-3 text-xs text-text-dim">No findings.</p>
      ) : (
        <>
          {notOk.length > 0 && <ul className="mt-3 flex flex-col gap-2">{notOk.map(renderFinding)}</ul>}
          {ok.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-text-secondary select-none">
                {ok.length} {ok.length === 1 ? "check" : "checks"} passed
              </summary>
              <ul className="mt-2 flex flex-col gap-2">{ok.map(renderFinding)}</ul>
            </details>
          )}
        </>
      )}
    </section>
  );
}
