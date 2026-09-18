// Tests for the plain-language report interpretation layer. Pure logic only
// - no React, no fetch, no DOM. The fixture's multifocality block matches
// the real outputs/report_neurovision/reports/BraTS2021_00002.json exactly
// (n_components_WT=2, largest_component_frac_WT=0.996, one small second
// fragment) - this is the case that exposed the overview/connectivity
// disagreement (overview said "in 2 separate pieces", connectivity said
// "one main mass plus a small separate fragment"); separate small
// overrides exercise the other connectivity/location/composition branches.
// `geometry` is deliberately omitted (older-report shape); `involvement`
// is included (present on reports from 2026-08-19 on).

import { describe, expect, it } from "vitest";
import type { ReportResponse } from "../api";
import { humanStructureName, interpretReport, volumeComparison } from "./reportInterpretation";

function makeReport(overrides: Partial<ReportResponse> = {}): ReportResponse {
  const base: ReportResponse = {
    report_version: 1,
    case_id: "BraTS2021_00002",
    generated_utc: "2026-08-19T04:58:27.180912+00:00",
    disclaimer:
      "This report is a research and educational decision-support artifact. It is not a " +
      "diagnostic tool and must not be used, alone or together with any other information, " +
      "to make or support a clinical decision about any patient.",
    not_claimed: [
      ["cell type", "MRI resolves millimetre-scale tissue, not individual cells."],
      ["WHO grade", "WHO CNS5 grading needs histology and molecular markers."],
      [
        "mass effect, midline shift, or ventricular compression",
        "The atlas encodes where a healthy midline and healthy ventricles sit, not where " +
          "this patient's own are, so none is computed.",
      ],
    ],
    burden: {
      volumes: {
        vol_NCR_mm3: 11159.0,
        vol_ED_mm3: 112505.0,
        vol_ET_mm3: 21795.0,
        vol_TC_mm3: 32954.0,
        vol_WT_mm3: 145459.0,
      },
      fractions: {
        frac_enhancing_of_wt: 0.1498360362713892,
        frac_necrotic_of_wt: 0.0767157755793728,
        frac_edema_of_wt: 0.7734481881492379,
        frac_enhancing_of_tc: 0.6613764641621654,
        frac_necrotic_of_tc: 0.3386235358378345,
        ratio_edema_to_core: 3.4140013351945138,
      },
      shape: {
        surface_area_ET_mm2: 11444.8779296875,
        sphericity_ET: 0.3296938721247329,
        surface_area_TC_mm2: 6770.275390625,
        sphericity_TC: 0.7342036650893259,
        surface_area_WT_mm2: 23285.5078125,
        sphericity_WT: 0.5744146080768832,
      },
      multifocality: {
        n_components_ET: 1,
        vol_largest_component_ET_mm3: 21791.0,
        vol_second_component_ET_mm3: null,
        largest_component_frac_ET: 0.9998164716678136,
        n_components_TC: 1,
        vol_largest_component_TC_mm3: 32951.0,
        vol_second_component_TC_mm3: null,
        largest_component_frac_TC: 0.9999089640104388,
        n_components_WT: 2,
        vol_largest_component_WT_mm3: 144878.0,
        vol_second_component_WT_mm3: 581.0,
        largest_component_frac_WT: 0.9960057473239882,
      },
      laterality: {
        dominant_side_ET: "left",
        frac_left_ET: 0.995824730442762,
        dominant_side_TC: "left",
        frac_left_TC: 0.9969654670146264,
        vol_right_WT_mm3: 7775.0,
        vol_left_WT_mm3: 137684.0,
        frac_left_WT: 0.9465485119518214,
        frac_contralateral_WT: 0.0534514880481785,
        dominant_side_WT: "left",
      },
      centroid: {
        centroid_i_WT: 143.24415814765672,
        centroid_j_WT: 94.34447507545082,
        centroid_k_WT: 83.06070439092802,
      },
      other: {},
    },
    anatomy: {
      atlas: { name: "tzo116plus", version: "2.0" },
      caveat:
        "This atlas describes healthy-brain anatomy. A tumour physically displaces the " +
        "tissue around it, so structure involvement reported here is approximate.",
      coverage_line: "23 of 122 structures classified eloquent, 99 unclassified.",
      region: "WT",
      structures: [
        {
          region: "WT",
          structure: "Frontal_Inf_Orb_L",
          laterality: "L",
          lobe: "frontal",
          eloquence: "unclassified",
          matched_term: null,
          n_voxels: 9254,
          volume_mm3: 9254.0,
          frac_of_tumour: 0.0636193016588867,
          frac_of_structure: 0.8741734366143964,
        },
        {
          region: "WT",
          structure: "Caudate_L",
          laterality: "L",
          lobe: "deep",
          eloquence: "eloquent",
          matched_term: "basal ganglia",
          n_voxels: 4288,
          volume_mm3: 4288.0,
          frac_of_tumour: 0.0294790972026481,
          frac_of_structure: 0.8683677602268125,
        },
        {
          region: "WT",
          structure: "Frontal_Mid_Orb_L",
          laterality: "L",
          lobe: "frontal",
          eloquence: "unclassified",
          matched_term: null,
          n_voxels: 5611,
          volume_mm3: 5611.0,
          frac_of_tumour: 0.0385744436576629,
          frac_of_structure: 0.861904761904762,
        },
      ],
      n_structures_involved: 41,
      frac_unlabelled: 0.3095305206278058,
    },
    involvement: {
      caveat:
        "Every value in this block is an overlap between the tumour mask and where a " +
        "healthy-brain atlas places a structure — never a measurement of this patient's " +
        "own anatomy.",
      not_vasari:
        "These fields are geometrically named quantities; their relationship to VASARI " +
        "features F19-F21 is approximate and has not been verified against the primary " +
        "documentation.",
      lower_bound_notes: ["ventricular overlap is a lower bound."],
      groups: {
        ventricle_overlap_mm3: 5089.0,
        ventricle_frac_of_tumour: 0.0349858035597659,
        ventricle_frac_of_group: 0.2399339933993399,
        ventricle_contact: true,
        deep_wm_overlap_mm3: 616.0,
        deep_wm_frac_of_tumour: 0.0042348703070968,
        deep_wm_frac_of_group: 0.0842220399234345,
        deep_wm_contact: true,
      },
      tissue: {
        cortical_frac_of_tumour: 0.3923648588261984,
        white_matter_frac_of_tumour: 0.4795921874892581,
        csf_frac_of_tumour: 0.1261592613726204,
        outside_tissue_frac_of_tumour: 0.0018836923119229,
      },
      epicentre: {
        epicentre_structure: "Insula_L",
        epicentre_exact: false,
        epicentre_distance_mm: 2.7779511183265324,
        epicentre_laterality: "L",
        epicentre_side: "left",
        epicentre_lobe: "insula",
      },
      other: {},
    },
    eloquence: {
      classification: "Sawaya eloquence grading",
      citation: "Sawaya R, et al. Neurosurgery. 1998.",
      evidence:
        "Eloquent locations in the Sawaya study are the motor/sensory cortices, visual " +
        "center, speech center, internal capsule, basal ganglia, hypothalamus/thalamus, " +
        "brainstem, and dentate nucleus.",
      source_owns_claim:
        "This eloquence classification is a lookup into a named, published source; our own " +
        "contribution is only the mapping from an atlas structure to that source's list.",
      involved: [
        { structure: "Caudate_L", laterality: "L", frac_of_tumour: 0.0294790972026481, frac_of_structure: 0.8683677602268125 },
        { structure: "Putamen_L", laterality: "L", frac_of_tumour: 0.0254229714215002, frac_of_structure: 0.7770540029417945 },
      ],
      distance_mm: 0.0,
      near_eloquent_threshold_mm: 10.0,
      near_eloquent: true,
      coverage_gaps: ["internal capsule", "dentate nucleus", "hypothalamus"],
    },
    provenance: {
      atlas_name: "tzo116plus",
      atlas_version: "2.0",
      atlas_source: "NITRC group_id=214",
      atlas_licence: "CC-BY-SA",
      knowledge_versions: { eloquence_map: 1, aal_lobes: 1 },
      segmentation_source: "prediction",
      segmentation_dir: "/Users/amish/NeuroVision-X/outputs/neurovision/eval_test/predictions",
      code_revision: "3e74e239ccb47ce0186065481c74c8d6f3237a7c-dirty",
      generated_utc: "2026-08-19T04:58:27.180912+00:00",
    },
  };
  return { ...base, ...overrides };
}

const SECTION_ORDER = [
  "overview",
  "size",
  "composition",
  "location",
  "regions",
  "shape",
  "connectivity",
  "surroundings",
  "eloquence",
  "limits",
];

describe("interpretReport - section structure", () => {
  it("returns every section id in the fixed order when involvement is present", () => {
    const result = interpretReport(makeReport());
    expect(result.sections.map((s) => s.id)).toEqual(SECTION_ORDER);
  });

  it("omits surroundings when involvement is undefined, but keeps overview and limits", () => {
    const result = interpretReport(makeReport({ involvement: undefined }));
    const ids = result.sections.map((s) => s.id);
    expect(ids).not.toContain("surroundings");
    expect(ids).toEqual(SECTION_ORDER.filter((id) => id !== "surroundings"));
    expect(ids).toContain("overview");
    expect(ids).toContain("limits");
  });
});

describe("interpretReport - overview", () => {
  function withOverviewMultifocality(overrides: Record<string, number | null>): ReportResponse {
    const base = makeReport();
    return {
      ...base,
      burden: {
        ...base.burden,
        multifocality: { ...base.burden.multifocality, ...overrides },
      },
    };
  }

  it("builds the at-a-glance headline from volume, tissue, side, epicentre and connectivity", () => {
    const result = interpretReport(makeReport());
    const overview = result.sections.find((s) => s.id === "overview")!;
    expect(overview.headline).toContain("145.5 mL");
    expect(overview.headline).toContain("tennis ball");
    expect(overview.headline).toContain("mostly swelling");
    expect(overview.headline).toContain("left");
    // Base fixture matches the real case exactly: n=2, largest_component_frac_WT=0.996 -
    // this must read the SAME regime as the connectivity section's headline for the
    // identical numbers (see the connectivity describe block below), not "2 separate pieces".
    expect(overview.headline).toContain("in one main mass with a small separate fragment");
  });

  it("agrees with the connectivity section on all three WT connectivity regimes", () => {
    // n === 1 -> one connected mass.
    const single = interpretReport(withOverviewMultifocality({ n_components_WT: 1, largest_component_frac_WT: 1.0 }));
    expect(single.sections.find((s) => s.id === "overview")!.headline).toContain("in one connected mass");

    // n > 1, largest_component_frac_WT >= 0.95 -> one main mass with fragment(s).
    // n=2 -> singular "fragment".
    const twoDominant = interpretReport(
      withOverviewMultifocality({ n_components_WT: 2, largest_component_frac_WT: 0.996 }),
    );
    expect(twoDominant.sections.find((s) => s.id === "overview")!.headline).toContain(
      "in one main mass with a small separate fragment",
    );

    // n > 2 -> plural "fragments".
    const manyDominant = interpretReport(
      withOverviewMultifocality({ n_components_WT: 4, largest_component_frac_WT: 0.97 }),
    );
    expect(manyDominant.sections.find((s) => s.id === "overview")!.headline).toContain(
      "in one main mass with small separate fragments",
    );

    // Otherwise -> N separate pieces (no single piece dominates).
    const trulySeparate = interpretReport(
      withOverviewMultifocality({ n_components_WT: 3, largest_component_frac_WT: 0.6 }),
    );
    expect(trulySeparate.sections.find((s) => s.id === "overview")!.headline).toContain("in 3 separate pieces");
  });
});

describe("interpretReport - composition", () => {
  it("uses the oedema-dominant branch when frac_edema_of_wt >= 0.6", () => {
    const result = interpretReport(makeReport());
    const composition = result.sections.find((s) => s.id === "composition")!;
    expect(composition.headline).toContain("Swelling makes up most of the marked region");
  });

  it("falls back to the mix branch when no tissue clears its threshold", () => {
    const report = makeReport({
      burden: {
        ...makeReport().burden,
        fractions: {
          frac_enhancing_of_wt: 0.3,
          frac_necrotic_of_wt: 0.2,
          frac_edema_of_wt: 0.5,
          frac_enhancing_of_tc: 0.5,
          frac_necrotic_of_tc: 0.5,
          ratio_edema_to_core: 1.0,
        },
      },
    });
    const composition = interpretReport(report).sections.find((s) => s.id === "composition")!;
    expect(composition.headline).toContain("The marked region is a mix");
    expect(composition.headline).toContain("The core is a mix of enhancing and necrotic tissue.");
  });
});

describe("interpretReport - location", () => {
  it("reads 'mostly in' with the crossing percentage for a mid-range contralateral fraction", () => {
    const result = interpretReport(makeReport());
    const location = result.sections.find((s) => s.id === "location")!;
    expect(location.headline).toContain("Mostly in the left hemisphere, with 5.3% crossing the midline");
  });

  it("reads 'confined to' below 0.05", () => {
    const report = makeReport({
      burden: {
        ...makeReport().burden,
        laterality: { ...makeReport().burden.laterality, frac_contralateral_WT: 0.02, dominant_side_WT: "left" },
      },
    });
    const location = interpretReport(report).sections.find((s) => s.id === "location")!;
    expect(location.headline).toContain("Confined to");
  });

  it("reads 'spans both hemispheres' at or above 0.25", () => {
    const report = makeReport({
      burden: {
        ...makeReport().burden,
        laterality: { ...makeReport().burden.laterality, frac_contralateral_WT: 0.3, dominant_side_WT: "left" },
      },
    });
    const location = interpretReport(report).sections.find((s) => s.id === "location")!;
    expect(location.headline).toContain("Spans both hemispheres");
  });
});

describe("interpretReport - connectivity", () => {
  function withMultifocality(overrides: Record<string, number | null>): ReportResponse {
    const base = makeReport();
    return {
      ...base,
      burden: {
        ...base.burden,
        multifocality: { ...base.burden.multifocality, ...overrides },
      },
    };
  }

  it("reads 'one main mass plus fragments' when the largest piece dominates", () => {
    const report = withMultifocality({
      n_components_WT: 2,
      largest_component_frac_WT: 0.996,
      vol_second_component_WT_mm3: 581.0,
    });
    const connectivity = interpretReport(report).sections.find((s) => s.id === "connectivity")!;
    expect(connectivity.headline).toContain("One main mass plus 1 small separate fragment");
  });

  it("reads 'one connected mass' for a single component", () => {
    const report = withMultifocality({ n_components_WT: 1, largest_component_frac_WT: 1.0 });
    const connectivity = interpretReport(report).sections.find((s) => s.id === "connectivity")!;
    expect(connectivity.headline).toBe("One connected mass.");
  });

  it("reads 'N separate pieces' when no single piece dominates", () => {
    const report = withMultifocality({ n_components_WT: 3, largest_component_frac_WT: 0.6 });
    const connectivity = interpretReport(report).sections.find((s) => s.id === "connectivity")!;
    expect(connectivity.headline).toContain("3 separate pieces");
  });
});

describe("interpretReport - shape", () => {
  it("reads 'moderately irregular' for a mid-range sphericity", () => {
    const result = interpretReport(makeReport());
    const shape = result.sections.find((s) => s.id === "shape")!;
    expect(shape.headline).toContain("Moderately irregular");
    expect(shape.headline).toContain("0.57");
  });
});

describe("interpretReport - eloquence", () => {
  it("flags near_eloquent as a non-discriminating reference lookup", () => {
    const result = interpretReport(makeReport());
    const eloquence = result.sections.find((s) => s.id === "eloquence")!;
    expect(eloquence.headline).toContain("does not distinguish this tumour from any other");
    expect(eloquence.headline).toContain("within");
  });

  it("uses the 'more than' phrasing when not near eloquent", () => {
    const report = makeReport({
      eloquence: { ...makeReport().eloquence, near_eloquent: false },
    });
    const eloquence = interpretReport(report).sections.find((s) => s.id === "eloquence")!;
    expect(eloquence.headline).toContain("more than");
    expect(eloquence.headline).toContain("does not distinguish this tumour from any other");
  });
});

describe("interpretReport - basisSentence", () => {
  it("names the model's own segmentation for a prediction-sourced report", () => {
    const result = interpretReport(makeReport());
    expect(result.basis).toBe("prediction");
    expect(result.basisSentence).toContain("model's own segmentation");
  });

  it("names the reference label for a label-sourced report", () => {
    const report = makeReport({ provenance: { ...makeReport().provenance, segmentation_source: "label" } });
    const result = interpretReport(report);
    expect(result.basis).toBe("label");
    expect(result.basisSentence).toContain("reference (human) label");
  });
});

describe("interpretReport - caveats carried verbatim", () => {
  it("threads anatomy.caveat, involvement.caveat/not_vasari and eloquence.source_owns_claim through unedited", () => {
    const report = makeReport();
    const result = interpretReport(report);
    const regions = result.sections.find((s) => s.id === "regions")!;
    const surroundings = result.sections.find((s) => s.id === "surroundings")!;
    const eloquence = result.sections.find((s) => s.id === "eloquence")!;
    expect(regions.caveat).toBe(report.anatomy.caveat);
    expect(surroundings.caveat).toContain(report.involvement!.caveat);
    expect(surroundings.caveat).toContain(report.involvement!.not_vasari);
    expect(eloquence.caveat).toContain(report.eloquence.source_owns_claim);
  });
});

describe("interpretReport - fact values", () => {
  it("gives every fact a non-empty value string, except in limits where value may be empty", () => {
    const result = interpretReport(makeReport());
    for (const section of result.sections) {
      for (const f of section.facts) {
        if (section.id === "limits") {
          expect(typeof f.value).toBe("string");
        } else {
          expect(f.value.length).toBeGreaterThan(0);
        }
      }
    }
  });
});

describe("interpretReport - wording gate", () => {
  const BANNED = [
    "grade",
    "prognos",
    "malignan",
    "aggressiv",
    "invasion",
    "compression",
    "mass effect",
    "midline shift",
    "danger",
    "concerning",
    "diagnos",
  ];

  it("never uses a banned clinical-implication word in any headline or explanation", () => {
    const result = interpretReport(makeReport());
    const text = result.sections
      .filter((s) => s.id !== "limits")
      .map((s) => `${s.headline} ${s.explanation}`)
      .join(" ")
      .toLowerCase();
    for (const word of BANNED) {
      expect(text).not.toContain(word);
    }
  });
});

describe("humanStructureName", () => {
  it("maps generic lobe-gyrus tokens with side and sub-part", () => {
    expect(humanStructureName("Frontal_Inf_Orb_L")).toBe("left inferior frontal gyrus, orbital part");
  });

  it("maps a named whole with side", () => {
    expect(humanStructureName("Caudate_L")).toBe("left caudate nucleus");
    expect(humanStructureName("Precentral_R")).toBe("right precentral gyrus (motor strip)");
  });

  it("maps cerebellum crus lobules", () => {
    expect(humanStructureName("Cerebelum_Crus1_R")).toBe("right cerebellum, lobule crus I");
  });

  it("maps vermis compound lobule numbers", () => {
    expect(humanStructureName("Vermis_4_5")).toBe("cerebellar vermis, lobule IV–V");
  });

  it("falls back to underscore-to-space for unrecognised tokens", () => {
    expect(humanStructureName("Foo_Bar")).toBe("Foo Bar");
  });
});

describe("volumeComparison", () => {
  it("picks the nearest-on-log-scale reference object", () => {
    expect(volumeComparison(1.2)).toContain("sugar cube");
    expect(volumeComparison(145)).toContain("tennis ball");
  });

  it("returns the bare comparative phrase (no 'about the volume of' prefix) outside the table", () => {
    // "about the volume of less than half a sugar cube" is ungrammatical, so these two
    // are returned bare - unlike the in-range case above, which keeps the "about the
    // volume of a ..." framing. Callers insert the result verbatim either way.
    expect(volumeComparison(0.2)).toBe("less than half a sugar cube");
    expect(volumeComparison(2000)).toBe("more than a pint of water");
  });

  it("returns an empty string for non-finite input", () => {
    expect(volumeComparison(NaN)).toBe("");
    expect(volumeComparison(Infinity)).toBe("");
  });
});
