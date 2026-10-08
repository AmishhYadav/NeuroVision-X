import type { CaseMetrics, CaseRegions, RegionKey } from "../api";

interface MetricsPanelProps {
  metrics: CaseMetrics | null;
  regions: CaseRegions | null;
}

const REGION_ORDER: RegionKey[] = ["ET", "TC", "WT"];

function fmt(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined ? "—" : v.toFixed(digits);
}

export function MetricsPanel({ metrics, regions }: MetricsPanelProps) {
  const anyGtEmpty = metrics ? REGION_ORDER.some((r) => metrics.gt_empty[r]) : false;
  const anyHd95Missing = metrics
    ? REGION_ORDER.some((r) => metrics.hd95[r] === null || metrics.hd95[r] === undefined)
    : false;
  const hasTruthVolume = Boolean(regions?.label);

  return (
    <div className="flex flex-col gap-4 px-3 py-3">
      <div>
        <div className="eyebrow mb-1.5">Dice</div>
        {metrics ? (
          <dl className="flex flex-col gap-0.5">
            {REGION_ORDER.map((region) => {
              const empty = metrics.gt_empty[region];
              return (
                <div key={region} className="flex items-baseline justify-between">
                  <dt className="font-mono text-xs text-text-secondary">{region}</dt>
                  <dd
                    className="tabular font-mono text-xs text-text-primary"
                    title={
                      empty
                        ? "Ground truth is empty for this region; Dice is scored empty-vs-empty."
                        : undefined
                    }
                  >
                    {fmt(metrics.dice[region])}
                    {empty && <span className="ml-0.5 text-text-dim">*</span>}
                  </dd>
                </div>
              );
            })}
          </dl>
        ) : (
          <p className="font-mono text-xs text-text-dim">No ground truth for this case.</p>
        )}
      </div>

      <div>
        <div className="eyebrow mb-1.5">HD95 (mm)</div>
        {metrics ? (
          <dl className="flex flex-col gap-0.5">
            {REGION_ORDER.map((region) => (
              <div key={region} className="flex items-baseline justify-between">
                <dt className="font-mono text-xs text-text-secondary">{region}</dt>
                <dd className="tabular font-mono text-xs text-text-primary">
                  {fmt(metrics.hd95[region], 2)}
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="font-mono text-xs text-text-dim">—</p>
        )}
      </div>

      <div>
        <div className="eyebrow mb-1.5">Volume (ml)</div>
        {/* Predicted against ground-truth volume side by side. Dice says how
            well two masks overlap; it does not say whether the model
            over- or under-segments, and the pair of numbers does. */}
        {hasTruthVolume && (
          <div className="mb-0.5 flex items-baseline justify-between font-mono text-[10px] text-text-dim">
            <span />
            <span className="flex gap-3">
              <span className="w-12 text-right">pred</span>
              <span className="w-12 text-right">truth</span>
            </span>
          </div>
        )}
        <dl className="flex flex-col gap-0.5">
          {REGION_ORDER.map((region) => (
            <div key={region} className="flex items-baseline justify-between">
              <dt className="font-mono text-xs text-text-secondary">{region}</dt>
              <dd className="flex gap-3 font-mono text-xs">
                <span className="tabular w-12 text-right text-text-primary">
                  {fmt(regions?.prediction[region]?.ml, 1)}
                </span>
                {hasTruthVolume && (
                  <span className="tabular w-12 text-right text-text-secondary">
                    {fmt(regions?.label?.[region]?.ml, 1)}
                  </span>
                )}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Footnotes, not tooltips. A tooltip is invisible in a screenshot and
          unreachable when this is on a projector, and both marks below change
          how a number must be read. */}
      <div className="flex flex-col gap-1 border-t border-surface-seam pt-2 font-mono text-[10px] leading-snug text-text-dim">
        <p>Dice and HD95 come from scripts/evaluate.py at overlap 0.5. HD95 is in millimetres.</p>
        {anyGtEmpty && (
          <p>
            <span className="text-text-secondary">*</span> ground truth is empty for this region, so
            Dice is scored empty-vs-empty rather than by real overlap.
          </p>
        )}
        {anyHd95Missing && (
          <p>
            <span className="text-text-secondary">—</span> HD95 is undefined when exactly one of
            prediction and ground truth is empty; there is no surface to measure against.
          </p>
        )}
      </div>
    </div>
  );
}
