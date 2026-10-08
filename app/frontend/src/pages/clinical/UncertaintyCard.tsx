import { CONFORMAL_BAND, GRADCAM, PREDICTIVE_ENTROPY_SINGLE_PASS, type ConformalMeta } from "../../api";
import { bandSummary, type BandStats, type EntropyStats, type GradcamStats } from "../../lib/heatStats";
import { formatPercent, formatVolumeMl } from "../../lib/report";

/** What the viewer hands the card: the active buffer's kind, region and precomputed stats. */
export interface UncertaintyCardProps {
  kind: string | null;
  /** "WT" or "TC" for band / Grad-CAM overlays. */
  region: "WT" | "TC" | null;
  entropy: EntropyStats | null;
  band: BandStats | null;
  /** Operating point of the conformal band (decides what the one-mask voxels mean); null/absent if not reported. */
  conformal?: ConformalMeta | null;
  gradcam: GradcamStats | null;
  /** Millimetres cubed per voxel from the job geometry; null when geometry is not loaded (volumes then show voxels). */
  voxelMm3: number | null;
}

const BAND_FOOTER =
  "Calibrated so that, averaged over in-distribution studies, mask + band miss at most 10% of tumour voxels (α = 0.10). Not a guarantee for this patient, and it does not hold for scans unlike the training data.";

function Row({ swatch, label, value }: { swatch?: string; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${swatch ?? ""}`} aria-hidden="true" />
      <span className="text-text-secondary">{label}</span>
      <span className="ml-auto font-mono text-text-primary">{value}</span>
    </div>
  );
}

const dec2 = (v: number | null) => (v === null ? "—" : v.toFixed(2));

function volume(voxels: number, voxelMm3: number | null): string {
  return voxelMm3 === null ? `${voxels.toLocaleString()} voxels` : formatVolumeMl(voxels * voxelMm3);
}

function Shell({ title, subtitle, children, footer }: { title: string; subtitle?: string; children: React.ReactNode; footer: string }) {
  return (
    <section className="glass-panel p-4" data-testid="clinical-uncertainty-card">
      <p className="eyebrow">{title}</p>
      {subtitle && <p className="mt-0.5 text-xs text-text-dim">{subtitle}</p>}
      <div className="mt-3 flex flex-col gap-1.5">{children}</div>
      <p className="mt-3 text-[11px] text-text-dim">{footer}</p>
    </section>
  );
}

/** Statistics for whichever heat overlay is active. Renders nothing for an unknown kind. */
export function UncertaintyCard({ kind, region, entropy, band, gradcam, voxelMm3, conformal }: UncertaintyCardProps) {
  if (kind === PREDICTIVE_ENTROPY_SINGLE_PASS && entropy) {
    const max = Math.max(1, ...entropy.histogram);
    const above = entropy.fracAbove(0.5);
    return (
      <Shell
        title="Predictive entropy"
        subtitle="single pass · aleatoric + epistemic combined"
        footer="Normalised to [0, 1]. A localisation aid, not a probability of error."
      >
        <Row swatch="bg-data-necrotic" label="Necrotic core" value={dec2(entropy.necrotic.mean)} />
        <Row swatch="bg-data-oedema" label="Oedema" value={dec2(entropy.oedema.mean)} />
        <Row swatch="bg-data-enhancing" label="Enhancing" value={dec2(entropy.enhancing.mean)} />
        <Row label="Outside tumour (where entropy > 0)" value={dec2(entropy.outside.mean)} />
        <div className="mt-2 flex h-12 items-end gap-0.5" role="img" aria-label="Entropy histogram within predicted tumour">
          {entropy.histogram.map((c, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm bg-brand-teal"
              style={{ height: `${Math.max(c > 0 ? 4 : 0, (c / max) * 100)}%` }}
              title={`${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}: ${c.toLocaleString()} voxels`}
            />
          ))}
        </div>
        <p className="text-xs text-text-dim">
          Within predicted tumour, 0 = certain → 1 = maximal
        </p>
        <p className="text-xs text-text-primary">
          {above === null ? "—" : formatPercent(above, 0)} of tumour voxels above 0.5
        </p>
      </Shell>
    );
  }

  if (kind === CONFORMAL_BAND && band) {
    const side = conformal?.side ?? null;
    const { pointEstimateVoxels, conformalSetVoxels } = bandSummary(band, side);
    const threshold = conformal?.threshold != null ? String(conformal.threshold) : "—";
    const alpha = conformal?.alpha != null ? String(conformal.alpha) : "—";
    const pct = (frac: number) => formatPercent(frac, 0);
    let sentence: string;
    if (side === "restrictive" && pointEstimateVoxels) {
      sentence = `The conformal set is ${pct(band.oneMaskVoxels / pointEstimateVoxels)} smaller than the point estimate: at α = ${alpha} ${volume(band.oneMaskVoxels, voxelMm3)} of the point estimate lies outside the conformal set.`;
    } else if (side === "permissive" && pointEstimateVoxels) {
      sentence = `The conformal set adds ${pct(band.oneMaskVoxels / pointEstimateVoxels)} beyond the point estimate.`;
    } else if (side === null) {
      sentence = "Band side not reported by the server.";
    } else {
      sentence = "—";
    }
    return (
      <Shell title={`Conformal band · ${region ?? "—"}`} footer={BAND_FOOTER}>
        {side === null ? (
          <>
            <Row label="In both masks" value={volume(band.bothVoxels, voxelMm3)} />
            <Row label="In one mask only" value={volume(band.oneMaskVoxels, voxelMm3)} />
          </>
        ) : (
          <>
            <Row label="Point estimate (p > 0.5)" value={volume(pointEstimateVoxels ?? 0, voxelMm3)} />
            <Row label={`Conformal set (p ≥ ${threshold})`} value={volume(conformalSetVoxels ?? 0, voxelMm3)} />
          </>
        )}
        <p className="text-xs text-text-primary">{sentence}</p>
      </Shell>
    );
  }

  if (kind === GRADCAM && gradcam) {
    return (
      <Shell
        title={`Grad-CAM · ${region ?? "—"}`}
        footer="Evidence for this region's prediction. It shows where the network looked, not whether it is right."
      >
        <p className="text-xs text-text-primary">
          Evidence inside predicted region:{" "}
          {gradcam.shareInside === null ? "—" : formatPercent(gradcam.shareInside, 0)}
        </p>
        <Row label="Mean inside" value={dec2(gradcam.meanInside)} />
        <Row label="Mean outside" value={dec2(gradcam.meanOutside)} />
      </Shell>
    );
  }

  return null;
}
