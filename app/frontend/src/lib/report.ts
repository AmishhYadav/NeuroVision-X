// Pure formatting, labelling and validation logic for the Phase 4 structured
// report panel. No React, no fetch - `ReportPanel.tsx` is a thin renderer
// that maps over what this module returns, and every branch here is unit
// tested directly (see report.test.ts), the same split as render.ts /
// slicing.ts already use for the viewer.

import type { ReportProvenance, ReportResponse } from "../api";

// --------------------------------------------------------------------- //
// Scalar formatters
// --------------------------------------------------------------------- //

const MISSING = "—";

function isMissingNumber(value: unknown): value is null | undefined {
  if (value === null || value === undefined) return true;
  return typeof value === "number" && !Number.isFinite(value);
}

/** mm³ -> mL at one decimal (1 mL = 1000 mm³). `"—"` for null/undefined/non-finite. */
export function formatVolumeMl(mm3: number | null | undefined): string {
  if (isMissingNumber(mm3)) return MISSING;
  return `${(mm3 / 1000).toFixed(1)} mL`;
}

/** A fraction (e.g. `0.817`) to a percentage string (e.g. `"81.7%"`). `"—"` for missing. */
export function formatPercent(fraction: number | null | undefined, digits = 1): string {
  if (isMissingNumber(fraction)) return MISSING;
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** A bare number to a fixed number of decimals. `"—"` for missing. */
export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (isMissingNumber(value)) return MISSING;
  return value.toFixed(digits);
}

/** A millimetre distance to one decimal, with unit. `"—"` for missing. */
export function formatDistanceMm(mm: number | null | undefined): string {
  if (isMissingNumber(mm)) return MISSING;
  return `${mm.toFixed(1)} mm`;
}

// --------------------------------------------------------------------- //
// burdenLabel - one lookup table, built once, never guessed at runtime
// --------------------------------------------------------------------- //

const REGIONS = ["ET", "TC", "WT"] as const;

/**
 * Builds the per-region label entries (surface area, shape, multifocality,
 * laterality, centroid) for ET/TC/WT. This is still an explicit,
 * enumerated table - built once at module load from the fixed key patterns
 * `burden_profile` actually emits (see report.py's `_classify_burden_key`)
 * - not a runtime guess applied to an arbitrary unknown key. A key this
 * function does not produce falls through to `burdenLabel`'s raw-key
 * fallback, same as any other unrecognised key.
 */
function buildRegionalBurdenLabels(): Record<string, string> {
  const labels: Record<string, string> = {};
  for (const r of REGIONS) {
    labels[`surface_area_${r}_mm2`] = `${r} surface area`;
    labels[`sphericity_${r}`] = `${r} sphericity`;
    labels[`surface_to_volume_${r}`] = `${r} surface-to-volume ratio`;
    labels[`n_components_${r}`] = `${r} components`;
    labels[`vol_largest_component_${r}_mm3`] = `${r} largest component volume`;
    labels[`vol_second_component_${r}_mm3`] = `${r} second component volume`;
    labels[`largest_component_frac_${r}`] = `${r} largest component fraction`;
    labels[`vol_right_${r}_mm3`] = `${r} volume, right hemisphere`;
    labels[`vol_left_${r}_mm3`] = `${r} volume, left hemisphere`;
    labels[`frac_left_${r}`] = `${r} fraction left`;
    labels[`frac_contralateral_${r}`] = `${r} contralateral fraction`;
    labels[`dominant_side_${r}`] = `${r} dominant side`;
    labels[`centroid_i_${r}`] = `${r} centroid (i)`;
    labels[`centroid_j_${r}`] = `${r} centroid (j)`;
    labels[`centroid_k_${r}`] = `${r} centroid (k)`;
  }
  return labels;
}

const BURDEN_LABELS: Record<string, string> = {
  vol_NCR_mm3: "Necrotic core",
  vol_ED_mm3: "Oedema",
  vol_ET_mm3: "Enhancing tumour",
  vol_TC_mm3: "Tumour core",
  vol_WT_mm3: "Whole tumour",
  frac_enhancing_of_wt: "Enhancing fraction of WT",
  frac_necrotic_of_wt: "Necrotic fraction of WT",
  frac_edema_of_wt: "Oedema fraction of WT",
  frac_enhancing_of_tc: "Enhancing fraction of TC",
  frac_necrotic_of_tc: "Necrotic fraction of TC",
  ratio_edema_to_core: "Oedema-to-core ratio",
  ...buildRegionalBurdenLabels(),
};

/**
 * A human label for a raw `burden_profile` key. Falls back to the raw key,
 * unchanged, for anything not in the table - never drops a field the caller
 * computed, and never guesses a label from a pattern.
 */
export function burdenLabel(key: string): string {
  return BURDEN_LABELS[key] ?? key;
}

// --------------------------------------------------------------------- //
// formatBurdenValue - dispatches on the key's shape, mirroring
// report.py's `_format_burden_value`
// --------------------------------------------------------------------- //

/**
 * Formats one burden value using its key name as the only formatting hint
 * available here - the same convention `report.py::_format_burden_value`
 * uses server-side, so a reader sees the same numbers shaped the same way
 * in the Markdown report and in this panel. Volumes render in mL rather
 * than raw mm³, per this panel's own convention (formatVolumeMl).
 */
export function formatBurdenValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return MISSING;
  if (typeof value === "boolean") return String(value);
  if (typeof value === "string") return value.length > 0 ? value : MISSING;
  if (typeof value !== "number" || !Number.isFinite(value)) return MISSING;

  if (
    key.startsWith("frac_") ||
    key.startsWith("largest_component_frac") ||
    key.includes("_frac_")
  ) {
    return formatPercent(value);
  }
  if (key.startsWith("ratio_")) {
    return formatNumber(value, 2);
  }
  if (key.endsWith("_mm3")) {
    return formatVolumeMl(value);
  }
  if (key.endsWith("_mm2")) {
    return `${formatNumber(value, 0)} mm²`;
  }
  if (key.startsWith("centroid_")) {
    return formatNumber(value, 1);
  }
  if (key.startsWith("n_components_")) {
    return formatNumber(value, 0);
  }
  return formatNumber(value, 3);
}

// --------------------------------------------------------------------- //
// validateReport
// --------------------------------------------------------------------- //

function fail(message: string): never {
  throw new Error(`Invalid report: ${message}`);
}

function asRecord(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(`${path} is missing or not an object.`);
  }
  return value as Record<string, unknown>;
}

function requireNonEmptyString(value: unknown, path: string): void {
  if (typeof value !== "string" || value.length === 0) {
    fail(`${path} is required and must be a non-empty string.`);
  }
}

/**
 * Validates that `raw` carries every field this artifact's design treats as
 * required (see `report.py`'s module docstring: the disclaimer, the
 * `not_claimed` block, the mass-effect caveat, and the eloquence evidence +
 * citation are required fields, not optional presentation choices), and
 * returns it typed as `ReportResponse`.
 *
 * Throws a descriptive `Error` when a required block is missing or
 * malformed. Missing OPTIONAL values inside an otherwise-present block
 * (e.g. a null `distance_mm`, an empty `structures` list) are valid input
 * and must not throw here - they render as `"—"` or an empty-state message
 * in the panel instead.
 */
export function validateReport(raw: unknown): ReportResponse {
  const r = asRecord(raw, "report");

  requireNonEmptyString(r.case_id, "case_id");
  requireNonEmptyString(r.disclaimer, "disclaimer");

  if (!Array.isArray(r.not_claimed) || r.not_claimed.length === 0) {
    fail("not_claimed is required and must be a non-empty array.");
  }

  const anatomy = asRecord(r.anatomy, "anatomy");
  requireNonEmptyString(anatomy.caveat, "anatomy.caveat");

  const eloquence = asRecord(r.eloquence, "eloquence");
  requireNonEmptyString(eloquence.evidence, "eloquence.evidence");
  requireNonEmptyString(eloquence.citation, "eloquence.citation");

  asRecord(r.provenance, "provenance");

  if (!r.burden || typeof r.burden !== "object") {
    fail("burden is required.");
  }

  return r as unknown as ReportResponse;
}

// --------------------------------------------------------------------- //
// segmentationLabel
// --------------------------------------------------------------------- //

export interface SegmentationLabel {
  text: string;
  tone: "prediction" | "truth";
}

/**
 * `"Model prediction"` vs `"Ground truth"`, read from
 * `provenance.segmentation_source`. Throws on any other value rather than
 * defaulting to one - the same rule the uncertainty layer already follows
 * for an unrecognised `X-Uncertainty-Kind` header (see CLAUDE.md): a reader
 * must never be told this report describes a prediction when the server
 * sent something this client does not recognise.
 */
export function segmentationLabel(
  provenance: Pick<ReportProvenance, "segmentation_source">,
): SegmentationLabel {
  if (provenance.segmentation_source === "prediction") {
    return { text: "Model prediction", tone: "prediction" };
  }
  if (provenance.segmentation_source === "label") {
    return { text: "Ground truth", tone: "truth" };
  }
  throw new Error(
    `Unrecognised segmentation_source ${JSON.stringify(
      provenance.segmentation_source,
    )} - refusing to guess whether this report describes a prediction or ground truth.`,
  );
}
