// Turns a Phase 4 structured `ReportResponse` into plain-language,
// categorised sections for a non-specialist reader. No React, no fetch, no
// DOM - a future `ReportInterpretationPanel.tsx` will just map over what
// `interpretReport` returns, the same split `report.ts` already uses for the
// raw-key panel.
//
// This is a RESEARCH ARTIFACT under a strict claims gate (see CLAUDE.md and
// docs/paper/claims_and_evidence.md): every string built here is purely
// DESCRIPTIVE. Nothing here may say or imply grade, prognosis, malignancy,
// aggressiveness, treatability, symptoms, deficits, invasion, compression,
// mass effect, midline shift, or that anything is dangerous/concerning/
// good/bad. "Detects" and "diagnoses" are never used - the segmentation is
// "what the model marked" (or "what the reference label marks").
//
// Every caveat string the report itself carries (anatomy.caveat,
// involvement.caveat, geometry.caveat, eloquence.source_owns_claim) is
// threaded through VERBATIM, never paraphrased - that text is the artifact's
// own honesty about its limits and must survive unedited.

import type { BurdenBlock, ReportResponse } from "../api";
import {
  burdenLabel,
  formatDistanceMm,
  formatGeometryValue,
  formatNumber,
  formatPercent,
  formatVolumeMl,
  geometryLabel,
} from "./report";

// --------------------------------------------------------------------- //
// Public types
// --------------------------------------------------------------------- //

export interface InterpretedFact {
  label: string;
  value: string;
  note?: string;
}

export interface InterpretedSection {
  id:
    | "overview"
    | "size"
    | "composition"
    | "location"
    | "regions"
    | "shape"
    | "connectivity"
    | "surroundings"
    | "eloquence"
    | "limits";
  title: string;
  headline: string;
  facts: InterpretedFact[];
  explanation: string;
  caveat?: string;
}

export interface InterpretedReport {
  caseId: string;
  basis: "prediction" | "label";
  basisSentence: string;
  sections: InterpretedSection[];
}

// --------------------------------------------------------------------- //
// Safe burden-block readers - every section below must survive any key
// being missing, so nothing reads `block[key]` directly.
// --------------------------------------------------------------------- //

function num(block: BurdenBlock | undefined | null, key: string): number | null {
  if (!block) return null;
  const v = block[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function str(block: BurdenBlock | undefined | null, key: string): string | null {
  if (!block) return null;
  const v = block[key];
  return typeof v === "string" && v.length > 0 ? v : null;
}

function bool(block: BurdenBlock | undefined | null, key: string): boolean | null {
  if (!block) return null;
  const v = block[key];
  return typeof v === "boolean" ? v : null;
}

function fact(label: string, value: string, note?: string): InterpretedFact {
  return note !== undefined ? { label, value, note } : { label, value };
}

function joinWithAnd(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

// --------------------------------------------------------------------- //
// volumeComparison - a nearest-on-log-scale "roughly the size of" phrase.
// Fixed reference table, not a formula - chosen so the numbers read as
// everyday objects rather than requiring the reader to picture a cm^3.
// --------------------------------------------------------------------- //

const VOLUME_TABLE: [number, string][] = [
  [1, "sugar cube"],
  [5, "teaspoon"],
  [15, "tablespoon"],
  [40, "golf ball"],
  [60, "hen's egg"],
  [150, "tennis ball"],
  [250, "cup of water"],
  [500, "pint of water"],
];

/** "about the volume of a …" for a volume already in mL, nearest on a log scale. `""` for non-finite input. */
export function volumeComparison(ml: number): string {
  if (!Number.isFinite(ml)) return "";
  // These two are returned bare (no "about the volume of" prefix) - that
  // prefix reads ungrammatically in front of a comparative ("about the
  // volume of less than..."). Callers insert the result verbatim, so both
  // this bare form and the "about the volume of a ..." form below must
  // read correctly dropped straight into a sentence.
  if (ml < 0.5) return "less than half a sugar cube";
  if (ml > 750) return "more than a pint of water";
  let bestName = VOLUME_TABLE[0][1];
  let bestDist = Infinity;
  for (const [refMl, name] of VOLUME_TABLE) {
    const dist = Math.abs(Math.log(ml) - Math.log(refMl));
    if (dist < bestDist) {
      bestDist = dist;
      bestName = name;
    }
  }
  return `about the volume of a ${bestName}`;
}

// --------------------------------------------------------------------- //
// humanStructureName - tzo116plus / AAL atlas token -> plain English.
// See the module-level rule table in the spec this was written against;
// side (_L/_R) is stripped first and re-applied as a "left "/"right "
// prefix around whichever name rule matches the remaining token.
// --------------------------------------------------------------------- //

const WHOLE_NAMES: Record<string, string> = {
  Precentral: "precentral gyrus (motor strip)",
  Postcentral: "postcentral gyrus (sensory strip)",
  Supp_Motor_Area: "supplementary motor area",
  Rolandic_Oper: "rolandic operculum",
  Paracentral_Lobule: "paracentral lobule",
  Insula: "insula",
  Caudate: "caudate nucleus",
  Putamen: "putamen",
  Pallidum: "globus pallidus",
  Thalamus: "thalamus",
  Hippocampus: "hippocampus",
  ParaHippocampal: "parahippocampal gyrus",
  Amygdala: "amygdala",
  Cingulum_Ant: "anterior cingulate",
  Cingulum_Mid: "middle cingulate",
  Cingulum_Post: "posterior cingulate",
  Heschl: "Heschl's gyrus (primary auditory cortex)",
  Angular: "angular gyrus",
  SupraMarginal: "supramarginal gyrus",
  Precuneus: "precuneus",
  Cuneus: "cuneus",
  Lingual: "lingual gyrus",
  Fusiform: "fusiform gyrus",
  Calcarine: "calcarine cortex (primary visual)",
  Olfactory: "olfactory cortex",
  Rectus: "gyrus rectus",
  CorpusCallosum: "corpus callosum",
  LateralVentricle: "lateral ventricle",
  ThirdVentricle: "third ventricle",
  Pons: "pons",
  Frontal_Sup_Medial: "superior frontal gyrus, medial part",
  Frontal_Med_Orb: "medial orbitofrontal cortex",
  Temporal_Pole_Sup: "superior temporal pole",
  Temporal_Pole_Mid: "middle temporal pole",
  Frontal_Inf_Orb: "inferior frontal gyrus, orbital part",
  Frontal_Inf_Tri: "inferior frontal gyrus, triangular part",
  Frontal_Inf_Oper: "inferior frontal gyrus, opercular part",
  Frontal_Mid_Orb: "middle frontal gyrus, orbital part",
  Frontal_Sup_Orb: "superior frontal gyrus, orbital part",
};

const LOBE_WORD: Record<string, string> = {
  Frontal: "frontal",
  Temporal: "temporal",
  Parietal: "parietal",
  Occipital: "occipital",
};

const POS_WORD: Record<string, string> = {
  Inf: "inferior",
  Mid: "middle",
  Sup: "superior",
};

const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

function toRoman(n: number): string {
  return ROMAN[n] ?? String(n);
}

/** The cerebellum/vermis "lobule X" suffix, after the `Cerebelum_`/`Vermis_` prefix is stripped. */
function cerebellumLobule(rest: string): string {
  if (rest === "Crus1") return "crus I";
  if (rest === "Crus2") return "crus II";
  if (rest === "4_5") return "IV–V";
  if (rest === "7b") return "VIIb";
  if (/^\d+$/.test(rest)) return toRoman(parseInt(rest, 10));
  return rest.replace(/_/g, " ");
}

/** A plain-English name for a tzo116plus/AAL atlas structure token. Unrecognised input falls back to underscores replaced with spaces. */
export function humanStructureName(atlasName: string): string {
  let side: "left" | "right" | null = null;
  let base = atlasName;
  if (base.endsWith("_L")) {
    side = "left";
    base = base.slice(0, -2);
  } else if (base.endsWith("_R")) {
    side = "right";
    base = base.slice(0, -2);
  }

  const withSide = (name: string): string => (side ? `${side} ${name}` : name);

  const whole = WHOLE_NAMES[base];
  if (whole) return withSide(whole);

  if (base.startsWith("Cerebelum_")) {
    const rest = base.slice("Cerebelum_".length);
    return withSide(`cerebellum, lobule ${cerebellumLobule(rest)}`);
  }
  if (base === "Vermis" || base.startsWith("Vermis_")) {
    const rest = base === "Vermis" ? "" : base.slice("Vermis_".length);
    return withSide(`cerebellar vermis, lobule ${cerebellumLobule(rest)}`);
  }

  const m = /^(Frontal|Temporal|Parietal|Occipital)_(Inf|Mid|Sup)$/.exec(base);
  if (m) {
    const lobe = m[1];
    const pos = m[2];
    if (lobe === "Parietal" && pos !== "Mid") {
      return withSide(`${POS_WORD[pos]} parietal lobule`);
    }
    return withSide(`${POS_WORD[pos]} ${LOBE_WORD[lobe]} gyrus`);
  }

  return atlasName.replace(/_/g, " ");
}

// --------------------------------------------------------------------- //
// Section builders - one per InterpretedSection id, in the fixed order
// interpretReport assembles them.
// --------------------------------------------------------------------- //

/**
 * Shared threshold: WT reads as "one main mass plus fragment(s)" once its
 * largest connected component holds at least this share of the whole
 * tumour's volume. Both the overview and connectivity sections branch on
 * this same constant so they can never disagree about which of the three
 * connectivity regimes (single mass / dominant mass with fragments / truly
 * separate pieces) a case falls into.
 */
const FRAGMENT_DOMINANCE_THRESHOLD = 0.95;

function buildOverview(report: ReportResponse): InterpretedSection {
  const { volumes, fractions, laterality, multifocality } = report.burden;

  const volWT = num(volumes, "vol_WT_mm3");
  const volTC = num(volumes, "vol_TC_mm3");
  const volET = num(volumes, "vol_ET_mm3");
  const side = str(laterality, "dominant_side_WT");
  const nWT = num(multifocality, "n_components_WT");

  const edema = num(fractions, "frac_edema_of_wt");
  const enh = num(fractions, "frac_enhancing_of_wt");
  const nec = num(fractions, "frac_necrotic_of_wt");
  let dominantTissue = "a mix of tissue types";
  if (edema !== null || enh !== null || nec !== null) {
    const ranked: [string, number][] = [
      ["mostly swelling", edema ?? -Infinity],
      ["mostly enhancing tissue", enh ?? -Infinity],
      ["mostly necrotic tissue", nec ?? -Infinity],
    ];
    ranked.sort((a, b) => b[1] - a[1]);
    dominantTissue = ranked[0][0];
  }

  let epicentrePhrase: string | null = null;
  const epicentre = report.involvement?.epicentre;
  if (epicentre) {
    const lobe = str(epicentre, "epicentre_lobe");
    const eside = str(epicentre, "epicentre_side");
    if (lobe && eside) epicentrePhrase = `the ${eside} ${lobe}`;
  }
  if (!epicentrePhrase) {
    const first = report.anatomy.structures[0];
    if (first?.lobe) epicentrePhrase = `the ${first.lobe}`;
  }

  // Same three-way branch, and the same threshold, that buildConnectivity
  // uses for its WT headline - see FRAGMENT_DOMINANCE_THRESHOLD's docstring.
  // Previously this only checked n === 1 vs n > 1, so a case like the real
  // BraTS2021_00002 (n=2, largest_component_frac_WT=0.996) read as "in 2
  // separate pieces" here while the connectivity section correctly called
  // it "one main mass plus a small separate fragment" - contradictory.
  let connectivityPhrase = "in a pattern that could not be summarised";
  if (nWT !== null) {
    const largestFracWT = num(multifocality, "largest_component_frac_WT");
    if (nWT === 1) {
      connectivityPhrase = "in one connected mass";
    } else if (largestFracWT !== null && largestFracWT >= FRAGMENT_DOMINANCE_THRESHOLD) {
      connectivityPhrase =
        nWT > 2
          ? "in one main mass with small separate fragments"
          : "in one main mass with a small separate fragment";
    } else {
      connectivityPhrase = `in ${nWT} separate pieces`;
    }
  }

  const pieces: string[] = [`A ${formatVolumeMl(volWT)} abnormal region`];
  if (volWT !== null) {
    const cmp = volumeComparison(volWT / 1000);
    if (cmp) pieces.push(`— ${cmp} —`);
  }
  pieces.push(
    `${dominantTissue}${epicentrePhrase ? `, centred in ${epicentrePhrase}` : ""}, ${connectivityPhrase}.`,
  );
  const headline = pieces.join(" ");

  const facts: InterpretedFact[] = [
    fact("Whole tumour", formatVolumeMl(volWT)),
    fact("Tumour core", formatVolumeMl(volTC)),
    fact("Enhancing tumour", formatVolumeMl(volET)),
    fact("Side", side ?? "—"),
    fact("Pieces", nWT !== null ? formatNumber(nWT, 0) : "—"),
  ];

  return {
    id: "overview",
    title: "At a glance",
    headline,
    facts,
    explanation:
      '"Abnormal region" means every voxel the model marked as tumour tissue of any kind ' +
      "— swelling, solid core, or enhancing; the sections below break that total down by " +
      "size, composition, location and shape.",
  };
}

const VOLUME_KEYS = ["vol_WT_mm3", "vol_TC_mm3", "vol_ET_mm3", "vol_NCR_mm3", "vol_ED_mm3"] as const;
const VOLUME_NOTES: Record<string, string> = {
  vol_WT_mm3: "whole tumour (everything marked abnormal, including swelling)",
  vol_TC_mm3: "tumour core (the solid part: enhancing + necrotic)",
  vol_ET_mm3: "enhancing tumour (tissue that takes up contrast dye)",
  vol_NCR_mm3: "necrotic core (dead tissue, usually at the centre)",
  vol_ED_mm3: "oedema (swelling in the surrounding brain)",
};

function buildSize(report: ReportResponse): InterpretedSection {
  const volumes = report.burden.volumes;
  const volWT = num(volumes, "vol_WT_mm3");
  const volTC = num(volumes, "vol_TC_mm3");
  const volET = num(volumes, "vol_ET_mm3");

  const facts = VOLUME_KEYS.map((key) =>
    fact(burdenLabel(key), formatVolumeMl(num(volumes, key)), VOLUME_NOTES[key]),
  );

  const headline =
    `The whole marked region is ${formatVolumeMl(volWT)}; the solid core is ` +
    `${formatVolumeMl(volTC)} of that, and ${formatVolumeMl(volET)} is enhancing.`;

  let explanation =
    "1 mL is one cubic centimetre — a sugar cube. Volumes come from counting marked " +
    "voxels times voxel size.";
  if (volWT !== null) {
    const cmp = volumeComparison(volWT / 1000);
    if (cmp) explanation += ` This region is ${cmp}.`;
  }

  return { id: "size", title: "How big it is", headline, facts, explanation };
}

function buildComposition(report: ReportResponse): InterpretedSection {
  const fractions = report.burden.fractions;
  const edemaWt = num(fractions, "frac_edema_of_wt");
  const enhWt = num(fractions, "frac_enhancing_of_wt");
  const necWt = num(fractions, "frac_necrotic_of_wt");
  const enhTc = num(fractions, "frac_enhancing_of_tc");
  const necTc = num(fractions, "frac_necrotic_of_tc");
  const ratio = num(fractions, "ratio_edema_to_core");

  let headline: string;
  if (edemaWt !== null && edemaWt >= 0.6) {
    headline =
      `Swelling makes up most of the marked region (${formatPercent(edemaWt)}); the solid ` +
      `core is the remaining ${formatPercent(1 - edemaWt)}.`;
  } else if (enhWt !== null && enhWt >= 0.4) {
    headline = `Enhancing tissue makes up the largest share (${formatPercent(enhWt)}) of the marked region.`;
  } else if (necWt !== null && necWt >= 0.4) {
    headline = `Necrotic tissue makes up the largest share (${formatPercent(necWt)}) of the marked region.`;
  } else {
    headline =
      `The marked region is a mix: ${formatPercent(edemaWt)} swelling, ` +
      `${formatPercent(enhWt)} enhancing, ${formatPercent(necWt)} necrotic.`;
  }

  if (enhTc !== null && enhTc >= 0.66) {
    headline += ` Within the core, enhancing tissue dominates (${formatPercent(enhTc)}).`;
  } else if (necTc !== null && necTc >= 0.66) {
    headline += ` Within the core, most is necrotic (${formatPercent(necTc)}).`;
  } else {
    headline += " The core is a mix of enhancing and necrotic tissue.";
  }

  const facts: InterpretedFact[] = [
    fact(burdenLabel("frac_edema_of_wt"), formatPercent(edemaWt)),
    fact(burdenLabel("frac_enhancing_of_wt"), formatPercent(enhWt)),
    fact(burdenLabel("frac_necrotic_of_wt"), formatPercent(necWt)),
    fact(burdenLabel("frac_enhancing_of_tc"), formatPercent(enhTc)),
    fact(burdenLabel("frac_necrotic_of_tc"), formatPercent(necTc)),
    fact(burdenLabel("ratio_edema_to_core"), formatNumber(ratio, 2), "how many mL of swelling per mL of solid core"),
  ];

  const explanation =
    "Oedema is swelling in the brain around the tumour; enhancing tissue is the part that " +
    "takes up contrast dye; necrotic tissue is dead tissue, usually at the centre of the " +
    "solid core.";

  return { id: "composition", title: "What it is made of", headline, facts, explanation };
}

function buildLocation(report: ReportResponse): InterpretedSection {
  const laterality = report.burden.laterality;
  const side = str(laterality, "dominant_side_WT");
  const fracLeft = num(laterality, "frac_left_WT");
  const contra = num(laterality, "frac_contralateral_WT");
  const epicentre = report.involvement?.epicentre;

  let headline: string;
  if (side === null || contra === null) {
    headline = "Location could not be summarised.";
  } else {
    if (contra < 0.05) {
      headline = `Confined to the ${side} hemisphere`;
    } else if (contra < 0.25) {
      headline = `Mostly in the ${side} hemisphere, with ${formatPercent(contra)} crossing the midline`;
    } else {
      const other = side === "left" ? "right" : "left";
      headline = `Spans both hemispheres (${formatPercent(contra)} on the ${other} side)`;
    }
    if (epicentre) {
      const structureName = str(epicentre, "epicentre_structure");
      if (structureName) headline += `, centred near the ${humanStructureName(structureName)}`;
    }
    headline += ".";
  }

  const facts: InterpretedFact[] = [
    fact("Dominant side", side ?? "—"),
    fact(
      side === "right" ? "Share on the right" : "Share on the left",
      fracLeft !== null ? formatPercent(side === "right" ? 1 - fracLeft : fracLeft) : "—",
    ),
    fact("Crossing the midline", formatPercent(contra)),
  ];

  if (epicentre) {
    const structureName = str(epicentre, "epicentre_structure");
    facts.push(fact("Epicentre structure", structureName ? humanStructureName(structureName) : "—"));
    facts.push(fact("Epicentre lobe", str(epicentre, "epicentre_lobe") ?? "—"));
    facts.push(fact("Epicentre side", str(epicentre, "epicentre_side") ?? "—"));
    facts.push(
      fact(
        "Distance from epicentre to that structure",
        formatDistanceMm(num(epicentre, "epicentre_distance_mm")),
        "0 mm means the tumour's centre point falls inside the structure",
      ),
    );
  }

  const explanation =
    "Side is measured against the atlas midline, so a tumour that pushes the midline over " +
    "can read as slightly more bilateral than it is.";

  return { id: "location", title: "Where it is", headline, facts, explanation };
}

function buildRegions(report: ReportResponse): InterpretedSection {
  const anatomy = report.anatomy;
  const nInvolved = anatomy.n_structures_involved;

  const facts: InterpretedFact[] = [
    fact("Regions touched", nInvolved !== null ? formatNumber(nInvolved, 0) : "—"),
    fact(
      "Unlabelled share",
      formatPercent(anatomy.frac_unlabelled),
      "part of the tumour lies where the atlas has no label, e.g. white matter not in the parcellation",
    ),
  ];

  for (const row of anatomy.structures) {
    facts.push(
      fact(
        humanStructureName(row.structure),
        `${formatPercent(row.frac_of_structure)} of it is inside the tumour`,
        `${formatPercent(row.frac_of_tumour)} of the tumour`,
      ),
    );
  }

  const top3 = anatomy.structures
    .slice(0, 3)
    .map((r) => `${humanStructureName(r.structure)} (${formatPercent(r.frac_of_structure)})`);

  const headline =
    top3.length > 0
      ? `Overlaps ${nInvolved ?? "an unknown number of"} atlas regions; the most affected are ${joinWithAnd(top3)}.`
      : `Overlaps ${nInvolved ?? "an unknown number of"} atlas regions.`;

  const explanation =
    '"% of it" says how much of that region the tumour covers; "% of the tumour" says how ' +
    "much of the tumour that region holds — a small region can be almost entirely " +
    "covered while holding little of the tumour.";

  return {
    id: "regions",
    title: "Which brain regions it touches",
    headline,
    facts,
    explanation,
    caveat: anatomy.caveat,
  };
}

function buildShape(report: ReportResponse): InterpretedSection {
  const shape = report.burden.shape;
  const sphWT = num(shape, "sphericity_WT");
  const sphTC = num(shape, "sphericity_TC");
  const sphET = num(shape, "sphericity_ET");
  const surfWT = num(shape, "surface_area_WT_mm2");

  const facts: InterpretedFact[] = [
    fact(burdenLabel("sphericity_WT"), formatNumber(sphWT, 2)),
    fact(burdenLabel("sphericity_TC"), formatNumber(sphTC, 2)),
    fact(burdenLabel("sphericity_ET"), formatNumber(sphET, 2)),
    fact(burdenLabel("surface_area_WT_mm2"), surfWT !== null ? `${(surfWT / 100).toFixed(1)} cm²` : "—"),
  ];

  if (report.geometry) {
    for (const [key, value] of Object.entries(report.geometry.shape)) {
      facts.push(fact(geometryLabel(key), formatGeometryValue(key, value as number | null | undefined)));
    }
    for (const [key, value] of Object.entries(report.geometry.extent)) {
      facts.push(fact(geometryLabel(key), formatGeometryValue(key, value as number | null | undefined)));
    }
  }

  let headline: string;
  if (sphWT === null) {
    headline = "Shape could not be summarised.";
  } else if (sphWT >= 0.7) {
    headline = `Compact and fairly round (sphericity ${formatNumber(sphWT, 2)})`;
  } else if (sphWT >= 0.45) {
    headline = `Moderately irregular outline (sphericity ${formatNumber(sphWT, 2)})`;
  } else {
    headline = `Highly irregular, spread-out outline (sphericity ${formatNumber(sphWT, 2)})`;
  }

  const explanation =
    "Sphericity is 1.0 for a perfect ball and falls toward 0 as the outline becomes more " +
    "finger-like or spread out. Surface area grows with irregularity for the same volume.";

  const section: InterpretedSection = { id: "shape", title: "Its shape", headline, facts, explanation };
  if (report.geometry?.caveat) section.caveat = report.geometry.caveat;
  return section;
}

const CONNECTIVITY_REGIONS = ["WT", "TC", "ET"] as const;

function buildConnectivity(report: ReportResponse): InterpretedSection {
  const multi = report.burden.multifocality;
  const facts: InterpretedFact[] = [];

  for (const region of CONNECTIVITY_REGIONS) {
    const n = num(multi, `n_components_${region}`);
    const largestVol = num(multi, `vol_largest_component_${region}_mm3`);
    const largestFrac = num(multi, `largest_component_frac_${region}`);
    const secondVol = num(multi, `vol_second_component_${region}_mm3`);

    facts.push(fact(burdenLabel(`n_components_${region}`), n !== null ? formatNumber(n, 0) : "—"));
    facts.push(fact(burdenLabel(`vol_largest_component_${region}_mm3`), formatVolumeMl(largestVol)));
    facts.push(fact(burdenLabel(`largest_component_frac_${region}`), formatPercent(largestFrac)));
    if (secondVol !== null) {
      facts.push(fact(burdenLabel(`vol_second_component_${region}_mm3`), formatVolumeMl(secondVol)));
    }
  }

  const nWT = num(multi, "n_components_WT");
  const largestFracWT = num(multi, "largest_component_frac_WT");
  const secondVolWT = num(multi, "vol_second_component_WT_mm3");

  let headline: string;
  if (nWT === null) {
    headline = "Connectivity could not be summarised.";
  } else if (nWT === 1) {
    headline = "One connected mass.";
  } else if (largestFracWT !== null && largestFracWT >= FRAGMENT_DOMINANCE_THRESHOLD) {
    const nFragments = nWT - 1;
    const word = nFragments === 1 ? "fragment" : "fragments";
    const fragVol = secondVolWT !== null ? ` (largest fragment ${formatVolumeMl(secondVolWT)})` : "";
    headline =
      `One main mass plus ${nFragments} small separate ${word}${fragVol} — at this size a ` +
      "fragment is often stray marked voxels rather than a separate lesion.";
  } else {
    headline = `${nWT} separate pieces; the largest holds ${formatPercent(largestFracWT)} of the marked volume.`;
  }

  const explanation =
    'A "piece" is a group of marked voxels that all touch each other, directly or through a ' +
    "shared face, edge or corner; two areas with no touching voxels between them count as " +
    "separate pieces even if they sit close together.";

  return { id: "connectivity", title: "One mass or several", headline, facts, explanation };
}

function buildSurroundings(report: ReportResponse): InterpretedSection | null {
  const involvement = report.involvement;
  if (!involvement) return null;

  const { groups, tissue } = involvement;

  const ventOverlap = num(groups, "ventricle_overlap_mm3");
  const ventContact = bool(groups, "ventricle_contact");
  const ventShare = num(groups, "ventricle_frac_of_group");
  const dwmOverlap = num(groups, "deep_wm_overlap_mm3");
  const dwmContact = bool(groups, "deep_wm_contact");
  const dwmShare = num(groups, "deep_wm_frac_of_group");

  const cortical = num(tissue, "cortical_frac_of_tumour");
  const wm = num(tissue, "white_matter_frac_of_tumour");
  const csf = num(tissue, "csf_frac_of_tumour");

  const facts: InterpretedFact[] = [
    fact("Ventricle overlap", formatVolumeMl(ventOverlap)),
    fact("Ventricle contact", ventContact === null ? "—" : ventContact ? "yes" : "no"),
    fact("Share of ventricle group", formatPercent(ventShare)),
    fact("Deep white matter overlap", formatVolumeMl(dwmOverlap)),
    fact("Deep white matter contact", dwmContact === null ? "—" : dwmContact ? "yes" : "no"),
    fact("Share of deep white matter group", formatPercent(dwmShare)),
    fact("Cortical (grey matter)", formatPercent(cortical)),
    fact("White matter", formatPercent(wm)),
    fact("CSF / fluid spaces", formatPercent(csf)),
  ];

  let overlapPhrase: string;
  if (ventContact && dwmContact) {
    overlapPhrase =
      `Overlaps both the ventricles (${formatVolumeMl(ventOverlap)}) and deep white matter ` +
      `(${formatVolumeMl(dwmOverlap)}) on the atlas.`;
  } else if (ventContact) {
    overlapPhrase = `Overlaps the ventricles (${formatVolumeMl(ventOverlap)}) on the atlas.`;
  } else if (dwmContact) {
    overlapPhrase = `Overlaps deep white matter (${formatVolumeMl(dwmOverlap)}) on the atlas.`;
  } else {
    overlapPhrase = "Does not overlap the ventricles or deep white matter on the atlas.";
  }

  const headline =
    `${overlapPhrase} By atlas tissue class it is ${formatPercent(cortical)} grey matter, ` +
    `${formatPercent(wm)} white matter, ${formatPercent(csf)} fluid spaces.`;

  const explanation =
    "The ventricles are the brain's fluid-filled cavities that hold cerebrospinal fluid; " +
    "deep white matter is the nerve-fibre tissue beneath the cortex that connects brain " +
    "regions to each other.";

  return {
    id: "surroundings",
    title: "Ventricles, white matter and tissue type",
    headline,
    facts,
    explanation,
    caveat: `${involvement.caveat}\n${involvement.not_vasari}`,
  };
}

function buildEloquence(report: ReportResponse): InterpretedSection {
  const el = report.eloquence;
  const T = formatDistanceMm(el.near_eloquent_threshold_mm);

  const headline = el.near_eloquent
    ? `This report lists the tumour as within ${T} of a region the Sawaya (1998) study calls ` +
      "eloquent. In this dataset that flag is true for essentially every case, so it does " +
      "not distinguish this tumour from any other — read it as a reference lookup, not a finding."
    : `This report lists the tumour as more than ${T} from every region the Sawaya (1998) ` +
      "study calls eloquent. In this dataset that flag is true for essentially every case, " +
      "so it does not distinguish this tumour from any other — read it as a reference " +
      "lookup, not a finding.";

  const facts: InterpretedFact[] = [
    fact("Classification", el.classification),
    fact("Within threshold", `${el.near_eloquent ? "yes" : "no"} (${T})`),
    fact("Distance to nearest listed structure", formatDistanceMm(el.distance_mm)),
  ];

  for (const row of el.involved) {
    facts.push(fact(humanStructureName(row.structure), formatPercent(row.frac_of_structure)));
  }

  let caveat = el.source_owns_claim;
  if (el.coverage_gaps.length > 0) {
    caveat += ` Not covered by this atlas: ${el.coverage_gaps.join(", ")}.`;
  }

  return {
    id: "eloquence",
    title: "Nearness to 'eloquent' regions",
    headline,
    facts,
    explanation: el.evidence,
    caveat,
  };
}

function buildLimits(report: ReportResponse): InterpretedSection {
  const facts: InterpretedFact[] = report.not_claimed.map(([what, why]) => fact(what, "", why));
  return {
    id: "limits",
    title: "What this report does not say",
    headline:
      "No grade, stage, prognosis, deficit, or mass-effect statement is made — those " +
      "need information this pipeline does not have.",
    facts,
    explanation: report.disclaimer,
  };
}

// --------------------------------------------------------------------- //
// interpretReport
// --------------------------------------------------------------------- //

/**
 * Builds the plain-language interpretation of a Phase 4 structured report.
 * Sections are returned in a fixed order; `surroundings` is omitted when
 * the report carries no `involvement` block (older reports, see
 * `ReportInvolvement`'s docstring in api.ts). `overview` and `limits` are
 * always present.
 */
export function interpretReport(report: ReportResponse): InterpretedReport {
  const sections: InterpretedSection[] = [
    buildOverview(report),
    buildSize(report),
    buildComposition(report),
    buildLocation(report),
    buildRegions(report),
    buildShape(report),
    buildConnectivity(report),
  ];

  const surroundings = buildSurroundings(report);
  if (surroundings) sections.push(surroundings);

  sections.push(buildEloquence(report));
  sections.push(buildLimits(report));

  const basis = report.provenance.segmentation_source;
  const basisSentence =
    basis === "prediction"
      ? "Every number below is measured on the model's own segmentation, not on a human annotation."
      : "Every number below is measured on the reference (human) label for this case, not on the model's segmentation.";

  return { caseId: report.case_id, basis, basisSentence, sections };
}
