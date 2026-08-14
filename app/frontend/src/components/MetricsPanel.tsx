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
        <dl className="flex flex-col gap-0.5">
          {REGION_ORDER.map((region) => (
            <div key={region} className="flex items-baseline justify-between">
              <dt className="font-mono text-xs text-text-secondary">{region}</dt>
              <dd className="tabular font-mono text-xs text-text-primary">
                {fmt(regions?.prediction[region]?.ml, 1)}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <p className="border-t border-surface-seam pt-2 font-mono text-[10px] leading-snug text-text-dim">
        Dice and HD95 come from scripts/evaluate.py at overlap 0.5. HD95 is in millimetres.
      </p>
    </div>
  );
}
