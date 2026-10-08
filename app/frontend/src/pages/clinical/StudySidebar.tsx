import type { BurdenBlock, GateDecisionValue, ReportResponse } from "../../api";
import type { ReportPanelStatus } from "../../components/ReportPanel";
import { formatPercent, formatVolumeMl } from "../../lib/report";

interface StudySidebarProps {
  report: ReportResponse | null;
  reportStatus: ReportPanelStatus;
  decision: GateDecisionValue | null;
  jobId: string;
  onHoverStructure: (name: string | null) => void;
  highlightedStructureName: string | null;
  onOpenReport: () => void;
  onExport: () => void;
  exporting: boolean;
  exportError: string | null;
  /** Optional card rendered first (the viewer passes the active heat overlay's statistics). */
  uncertaintySlot?: React.ReactNode;
}

const EXPORT_TITLE = "Download report.json, report.md, DICOM-SEG and a twin snapshot as one zip";
const MAX_ATLAS_ROWS = 8;

/** A burden value as a finite number, or null (strings, booleans, null, NaN all become null). */
function num(block: BurdenBlock | undefined, key: string): number | null {
  const v = block?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function has(block: BurdenBlock | undefined, key: string): boolean {
  return block !== undefined && Object.prototype.hasOwnProperty.call(block, key);
}

function Skeleton({ rows }: { rows: number }) {
  return (
    <div className="flex flex-col gap-2" aria-hidden="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-4 animate-pulse rounded bg-surface-raised" />
      ))}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="glass-panel p-4">
      <p className="eyebrow mb-3">{title}</p>
      {children}
    </section>
  );
}

/** One horizontal stacked bar; segments are fractions (any finite, non-negative), drawn proportionally. */
function StackedBar({ segments, label }: { segments: { color: string; value: number }[]; label: string }) {
  return (
    <div
      className="flex h-2 w-full overflow-hidden rounded-full bg-surface-raised"
      role="img"
      aria-label={label}
    >
      {segments.map((s, i) => (
        <div key={i} className={s.color} style={{ width: `${Math.max(0, s.value) * 100}%` }} />
      ))}
    </div>
  );
}

function VolumeRow({
  swatch,
  label,
  mm3,
  fraction,
}: {
  swatch?: string;
  label: string;
  mm3: number | null;
  fraction?: number | null;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      {swatch ? (
        <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${swatch}`} aria-hidden="true" />
      ) : (
        <span className="h-2.5 w-2.5 shrink-0" aria-hidden="true" />
      )}
      <span className="text-text-secondary">{label}</span>
      <span className="ml-auto font-mono text-text-primary">{formatVolumeMl(mm3)}</span>
      {fraction !== undefined && (
        <span className="w-14 text-right font-mono text-text-dim">{formatPercent(fraction)}</span>
      )}
    </div>
  );
}

function capitalise(s: string): string {
  return s.length > 0 ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function plural(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

/**
 * Right-hand sidebar of the clinical study viewer. Every number comes from
 * the job's structured report (`report.burden`, `report.anatomy`,
 * `report.provenance`); a missing or non-numeric value renders as an em
 * dash, never a guess. Volumes in the report are mm^3; shown here as mL.
 */
export function StudySidebar(props: StudySidebarProps) {
  const {
    report,
    reportStatus,
    decision,
    onHoverStructure,
    highlightedStructureName,
    onOpenReport,
    onExport,
    exporting,
    exportError,
    uncertaintySlot,
  } = props;

  const loading = reportStatus === "loading" && !report;
  const unavailable = !report && !loading;
  const unavailableText =
    reportStatus === "not_found"
      ? "Report not available for this job."
      : "Report could not be loaded.";

  const volumes = report?.burden.volumes;
  const fractions = report?.burden.fractions;
  const laterality = report?.burden.laterality;
  const multifocality = report?.burden.multifocality;

  const fracEt = num(fractions, "frac_enhancing_of_wt");
  const fracNcr = num(fractions, "frac_necrotic_of_wt");
  const fracEd = num(fractions, "frac_edema_of_wt");

  const volLeft = num(laterality, "vol_left_WT_mm3");
  const volRight = num(laterality, "vol_right_WT_mm3");
  const dominant = laterality?.dominant_side_WT;
  const showLaterality =
    has(laterality, "dominant_side_WT") ||
    has(laterality, "vol_left_WT_mm3") ||
    has(laterality, "vol_right_WT_mm3");

  const components = (["WT", "TC", "ET"] as const)
    .filter((r) => has(multifocality, `n_components_${r}`))
    .map((r) => ({ region: r, n: num(multifocality, `n_components_${r}`) }));

  const atlasRows = report
    ? report.anatomy.structures
        .filter((r) => r.region === "WT")
        .sort((a, b) => (b.volume_mm3 ?? -Infinity) - (a.volume_mm3 ?? -Infinity))
        .slice(0, MAX_ATLAS_ROWS)
    : [];

  const atlasFooter =
    report?.provenance?.atlas_name && report.provenance.atlas_version
      ? `Atlas: ${report.provenance.atlas_name} v${report.provenance.atlas_version}`
      : null;

  const body = (rows: number, content: React.ReactNode) =>
    loading ? <Skeleton rows={rows} /> : unavailable ? (
      <p className="text-xs text-text-secondary">{unavailableText}</p>
    ) : (
      content
    );

  return (
    <>
      {uncertaintySlot}
      <Card title="Tumour volume">
        {body(
          5,
          <>
            <p className="eyebrow">Whole tumour (WT)</p>
            <p className="mt-1 font-mono text-3xl text-text-primary">
              {formatVolumeMl(num(volumes, "vol_WT_mm3"))}
            </p>
            <div className="mt-3 flex flex-col gap-1.5">
              <VolumeRow
                swatch="bg-data-enhancing"
                label="Enhancing (ET)"
                mm3={num(volumes, "vol_ET_mm3")}
                fraction={fracEt}
              />
              <VolumeRow
                swatch="bg-data-necrotic"
                label="Necrotic core"
                mm3={num(volumes, "vol_NCR_mm3")}
                fraction={fracNcr}
              />
              <VolumeRow
                swatch="bg-data-oedema"
                label="Oedema"
                mm3={num(volumes, "vol_ED_mm3")}
                fraction={fracEd}
              />
              <VolumeRow label="Tumour core (TC)" mm3={num(volumes, "vol_TC_mm3")} />
            </div>
            {fracEt !== null && fracNcr !== null && fracEd !== null && (
              <div className="mt-3">
                <StackedBar
                  label="Composition of whole tumour: enhancing, necrotic core, oedema"
                  segments={[
                    { color: "bg-data-enhancing", value: fracEt },
                    { color: "bg-data-necrotic", value: fracNcr },
                    { color: "bg-data-oedema", value: fracEd },
                  ]}
                />
              </div>
            )}
          </>,
        )}
      </Card>

      {(loading || (report && showLaterality)) && (
        <Card title="Laterality">
          {body(
            2,
            <>
              {typeof dominant === "string" && dominant.length > 0 && (
                <p className="font-mono text-sm text-text-primary">
                  Dominant side (WT): {capitalise(dominant)}
                </p>
              )}
              {(has(laterality, "vol_left_WT_mm3") || has(laterality, "vol_right_WT_mm3")) && (
                <div className="mt-2 flex flex-col gap-2">
                  {volLeft !== null && volRight !== null && volLeft + volRight > 0 && (
                    <StackedBar
                      label="Whole-tumour volume, left versus right"
                      segments={[
                        { color: "bg-brand-primary", value: volLeft / (volLeft + volRight) },
                        { color: "bg-brand-teal", value: volRight / (volLeft + volRight) },
                      ]}
                    />
                  )}
                  <div className="flex justify-between font-mono text-xs text-text-secondary">
                    <span>L {formatVolumeMl(volLeft)}</span>
                    <span>R {formatVolumeMl(volRight)}</span>
                  </div>
                </div>
              )}
            </>,
          )}
        </Card>
      )}

      {report && components.length > 0 && (
        <Card title="Multifocality">
          <ul className="flex flex-col gap-1">
            {components.map((c) => (
              <li key={c.region} className="flex justify-between text-xs">
                <span className="text-text-secondary">{c.region}</span>
                <span className="font-mono text-text-primary">
                  {c.n === null ? "—" : plural(c.n, "component")}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card title="Atlas overlap">
        {body(
          4,
          <>
            {atlasRows.length === 0 ? (
              <p className="text-xs text-text-secondary">No whole-tumour structure overlap in the report.</p>
            ) : (
              <ul className="flex flex-col gap-0.5" onMouseLeave={() => onHoverStructure(null)}>
                {atlasRows.map((row) => (
                  <li
                    key={`${row.structure}-${row.laterality ?? ""}`}
                    onMouseEnter={() => onHoverStructure(row.structure)}
                    onMouseLeave={() => onHoverStructure(null)}
                    className={`rounded-md px-2 py-1.5 ${
                      highlightedStructureName === row.structure ? "bg-brand-primary/15" : ""
                    }`}
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="truncate font-mono text-xs text-text-primary">{row.structure}</span>
                      {row.laterality && (
                        <span className="shrink-0 text-[11px] text-text-dim">{row.laterality}</span>
                      )}
                    </div>
                    <div className="flex justify-between font-mono text-[11px] text-text-secondary">
                      <span>{formatVolumeMl(row.volume_mm3)}</span>
                      <span>{formatPercent(row.frac_of_structure)} of structure</span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
            {typeof report?.anatomy.caveat === "string" && report.anatomy.caveat.length > 0 && (
              <p className="mt-3 text-[11px] text-text-dim">{report.anatomy.caveat}</p>
            )}
            {atlasFooter && <p className="mt-2 font-mono text-[11px] text-text-dim">{atlasFooter}</p>}
          </>,
        )}
      </Card>

      {(decision === "proceed" || decision === "proceed_with_caution") && (
        <Card title="Gate">
          <span
            className={`inline-flex items-center gap-2 rounded-md border px-2.5 py-1 font-mono text-xs ${
              decision === "proceed"
                ? "border-gate-proceed/50 bg-gate-proceed/10 text-gate-proceed"
                : "border-gate-caution/50 bg-gate-caution/10 text-gate-caution"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${decision === "proceed" ? "bg-gate-proceed" : "bg-gate-caution"}`}
              aria-hidden="true"
            />
            {decision === "proceed" ? "PROCEED" : "PROCEED WITH CAUTION"}
          </span>
        </Card>
      )}

      <Card title="Export">
        <div className="flex flex-col gap-2">
          <button
            type="button"
            className="btn-primary w-full justify-center text-sm"
            disabled={exporting}
            onClick={onExport}
            title={EXPORT_TITLE}
          >
            {exporting ? "Exporting…" : "Export study (.zip)"}
          </button>
          <button
            type="button"
            className="btn-secondary w-full justify-center !px-3 !py-1.5 text-xs"
            disabled={reportStatus === "not_found"}
            onClick={onOpenReport}
          >
            Open report
          </button>
          {exportError && (
            <p role="alert" className="font-mono text-[11px] text-gate-caution">
              {exportError}
            </p>
          )}
        </div>
      </Card>
    </>
  );
}
