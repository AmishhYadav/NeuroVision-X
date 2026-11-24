# Lockbox candidates: external adult glioma cohort (2026-11-24)

Web reading only. Nothing downloaded, registered or requested. "UNKNOWN" means the primary page fetched did not state it. Some facts come from a fetch tool's summary of a page, not a verbatim read; verify the load-bearing ones (marked **VERIFY**) before relying on them.

## Summary

| Candidate | n usable w/ labels | Sub-regions | Overlap with BraTS 2021 | Licence | Access | 4 modalities | Format | Size | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1. BraTS-Africa extra cases | 95 adult glioma in total (60 train + 35 val), so **35 new** beyond our 60. TCIA lists labels for all 146 (51 non-glioma). Whether the 35 are in the open package: **VERIFY** | ET / NETC / SNFH (WT/TC/ET) | None expected (Nigerian cohort); our 60 are inside it, so exclude by ID | CC BY 4.0 (processed images + segs); unprocessed NIfTI under NIH Controlled Access | TCIA (Aspera) for the BraTS 2023 package; Synapse for challenge phases | Yes | Skull-stripped, co-registered to a common template (SRI24 per BraTS convention, not stated on the TCIA page) | 1.6 GB package; 3.7 GB collection | **NEEDS-AUTHOR-ACTION** (confirm the 35 are labelled and open). n=35 is small |
| 2. UCSF-PDGM | 495 patients (501 IDs; some are follow-ups). Non-BraTS-2021 count: at most 495-387 = ~108, **derived, not stated** | ET / NETC / SNFH | 387 of 501 reviewed in BraTS 2021. A crosswalk exists in metadata v2 | CC BY 4.0 | TCIA open download (Aspera), no application | Yes (plus DWI, SWI, ASL) | Skull-stripped, ANTs-registered to T2-FLAIR space, 1 mm. **Not SRI24**, so needs re-registration | 142 GB (v5) | **PASS on access and labels, FAIL on size of the clean pool (~108)**; NEEDS-AUTHOR-ACTION for the crosswalk |
| 3. EGD | 374 manual | **Whole tumour only** (binary) | Not mentioned. UNKNOWN | UNKNOWN (DUA referenced, terms not on the page) | Health-RI XNAT, DUA | Yes | Defaced, MNI152-registered, 1 mm; HD-BET brain mask supplied | UNKNOWN | **FAIL** for sub-region analysis (WT only); NEEDS-AUTHOR-ACTION |
| 4a. BraTS-MEN 2023 | 1,141 public (1,000 train + 141 val) | ET / NETC / SNFH (meningioma definitions) | None (different tumour) | CC BY 4.0 (article); dataset licence not separately stated | Synapse, doi 10.7303/syn51514106 | Yes | SRI24, 1 mm, skull-stripped (BraTS format) | UNKNOWN | **PASS as fallback** for WT only. NEEDS-AUTHOR-ACTION (Synapse login) |
| 4b. BraTS-METS 2023 | 402 studies on Synapse per abstract (train/val split UNKNOWN) | UNKNOWN (not read) | None | UNKNOWN | Synapse | Yes per eligibility | UNKNOWN | UNKNOWN | Not assessed in depth |

## 1. BraTS-Africa extra cases

- 2024 release: 95 individuals, 60 train, 35 validation. Test split not mentioned. https://arxiv.org/html/2508.10905v1
- BraTS-Africa 2025 challenge: 95 preoperative glioma cases (60 train, 35 val). Validation predictions went to Synapse. https://arxiv.org/html/2510.03568v1
- Whole dataset: 146 annotated cases, of which 95 adult diffuse glioma and 51 other CNS neoplasms. https://arxiv.org/html/2510.03568v1 (secondary source; the dataset paper is Adewole et al., Radiol AI 2025;7(4):e240528).
- TCIA collection: 146 patients, expert-annotated sub-regions "for each of the 146 cases", CC BY 4.0 for images and segmentations (BraTS 2023 package, 1.6 GB, Aspera); unprocessed NIfTI limited access; T1, T1CE, T2, FLAIR at 1.5T, six Nigerian centres; skull-stripped and rigidly co-registered to the resampled T1CE. nnU-Net pre-annotation, ITK-SNAP review. https://www.cancerimagingarchive.net/collection/brats-africa/
- Label protocol: ET/TC/WT "as defined in the 2024 BraTS challenge" (https://arxiv.org/html/2508.10905v1). Whether it is identical to 2023 is **UNKNOWN**.
- Not found: whether the 35 validation cases have labels in the open package, or whether any hidden test set exists. TCIA gives no ID list and no 2024/2025 mention. The Synapse wiki would not render.
- Risk: the 51 non-glioma cases share the folder. Filter to the 95 gliomas and drop our 60 by ID.

## 2. UCSF-PDGM

- 495 subjects (501 IDs listed, some follow-ups); CC BY 4.0; public Aspera download; 142 GB (v5); 3T GE; segmentation done for BraTS 2021 (ensemble of prior BraTS winners, manually corrected, two neuroradiologists approved); ET, necrotic/non-enhancing core, FLAIR abnormality. https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- Version 2 added BraTS 2021 IDs to the metadata CSV (UCSF-PDGM-metadata_v2.csv); BraTS organisers confirmed the mapping. Whether v5's CSV still carries it is UNKNOWN (same page).
- 387 of 501 reviewed in BraTS 2021; skull-stripped; ANTs-registered to T2-FLAIR space at 1 mm. No licence stated in the paper. https://pmc.ncbi.nlm.nih.gov/articles/PMC9748624
- "387 reviewed" is not necessarily "387 in the BraTS 2021 release". The clean pool needs the crosswalk. It is probably about 100 cases at most.
- Labels for the non-BraTS cases are the same BraTS-style segmentations (per the TCIA text). Not confirmed per case.

## 3. Erasmus Glioma Database

- 774 patients, 2008-2018; T1, T1c, T2, FLAIR; manual segmentations for 374, CNN-generated for 400; **whole-tumour binary masks only**, drawn on T2 or FLAIR by one of four observers. https://pmc.ncbi.nlm.nih.gov/articles/PMC8203723
- Registered to MNI152 2009a at 1 mm, defaced, HD-BET mask provided. Hosted on Health-RI XNAT (project EGD); a DUA is referenced as a supplementary file, but its terms are not on the page. No dataset-specific licence stated. BraTS overlap not mentioned. Size not stated.
- Fails the sub-region requirement outright.

## 4. Fallbacks

- BraTS-MEN: 1,424 exams from 1,344 patients; 1,141 public (1,000 train, 141 val), 283 private test; four sequences; SRI24, 1 mm, skull-stripped via FeTS; ET / NETC / SNFH with meningioma-specific definitions (dural tail is ET); Synapse; article CC BY 4.0. https://pmc.ncbi.nlm.nih.gov/articles/PMC11096318/ A TCIA mirror URL I guessed returned 404, so none is confirmed.
- BraTS-METS 2023: 1,303 studies annotated, 402 released on Synapse (public train + val split not read). https://arxiv.org/abs/2306.00838v3 Everything else UNKNOWN.
- Both are not gliomas, so they test a different failure (tumour type shift), not "adult glioma external". Whole tumour only is the fair comparison.

## UCSF-PDGM crosswalk result (2026-11-24, metadata only — no image or label opened)

Fetched directly from TCIA (`https://www.cancerimagingarchive.net/wp-content/uploads/<file>`, no login or
agreement step): `UCSF-PDGM-metadata_v5.csv` (SHA-256 `afc1c23a…6fd78`) and `UCSF-PDGM-metadata_v2.csv`
(`c83b2783…1ba1df6`). Local copies in `data/external/ucsf_pdgm/` (gitignored; re-fetch with the URL above).

- v5 carries the `BraTS21 ID`, `BraTS21 Segmentation Cohort` and `BraTS21 MGMT Cohort` columns. 501 scans,
  495 patients; 298 scans have a BraTS 2021 ID (262 segmentation Training, 36 Validation), 203 have none.
- All 262 Training IDs are among our 1,251 preprocessed BraTS cases; none of the 36 Validation IDs is.
- v5 renamed six scans as follow-ups (`UCSF-PDGM-0NNN_FUxxxd`). Five of them are the scan BraTS used, so the
  same patient's baseline (396, 409, 431, and 391's follow-up) would leak if the crosswalk were done per scan.
  **Exclusion must be per patient** (strip `_FU…` and leading zeros from the ID).
- **Patient-level clean pool: 235 patients** (199 never in BraTS 2021 + 36 BraTS 2021 Validation, never
  in our training data). Never-BraTS grades: 100 grade 4, 43 grade 3, 56 grade 2 — far more lower-grade
  glioma than BraTS 2021, which is itself a population shift to declare in the protocol.
- The earlier "~108" was derived from 387 reviewed cases and is superseded.
- Side finding: patients 429 and 433 each appear twice inside BraTS 2021 (00557/00639, 00758/00626). Both
  pairs sit in our `train` split, so no train/test leak; recorded for W2.7.

## BraTS-Africa extra 35 — confirmed (2026-11-24, metadata only — no image or label opened)

Source: TCIA "Scanner and Diagnosis Information" sheet,
`https://www.cancerimagingarchive.net/wp-content/uploads/BraTS-Africa_TCIA_datainfo_v2.xlsx` (SHA-256 `74ac283b…`;
local copy `data/external/brats_africa/`, gitignored). Read with the standard library (no openpyxl).

- Sheet 1 lists **95 adult glioma** IDs, sheet 2 the **51 other CNS neoplasms** (with a "Neoplasm Category"
  column: metastases, ventricular masses …). No ID is on both sheets.
- All 60 of our `brats_ssa` cases are among the 95 glioma IDs. **35 glioma IDs are not ours:** 125, 126, 129,
  130, 132, 134, 136–140, 143, 148, 154, 155, 157, 158, 163, 169, 171, 174, 175, 179, 180, 188, 192, 198, 206,
  207, 210, 218, 225–228 (as `BraTS-SSA-00NNN-000`).
- Labels: TCIA states "expert-annotated tumor sub-regions for each of the 146 cases"; the paper (Adewole et al.,
  Radiol AI 2025, Data Availability) says the full 146 with "tumor subregion labels is freely available" under
  CC BY 4.0, 20 of the 95 having been withheld as the 2023 challenge test set. The sheet gives a sub-region
  count (1–3) for every glioma row.
- Download: the 1.6 GB "BraTS 2023 Challenge" package (Aspera) holds all 146 — fits on disk, same format as
  our SSA cases, so the existing preprocessing applies.
- The 51 non-glioma cases are a label-definition change (other tumour types), not a glioma lockbox.

## Author actions needed

1. Confirm on the TCIA BraTS-Africa page, or from the Radiol AI paper's data-availability section, that the 35 non-2023 adult glioma cases are in the open package with labels. Then match IDs against our 60.
2. Download UCSF-PDGM metadata v5 (the CSV is small, no Aspera needed) and check whether the BraTS 2021 ID column is present. If not, get v2. The count of non-BraTS cases follows from it. Needs the TCIA data usage agreement checkbox, which only the author can accept.
3. Create or use a Synapse account for BraTS-MEN or BraTS-METS (author, personally; click-through terms).
4. EGD: do not pursue unless a whole-tumour-only analysis is acceptable and the author is willing to file the DUA on Health-RI.

## Recommendation

UCSF-PDGM non-BraTS cases are the best true adult-glioma lockbox if the crosswalk yields roughly 100 cases. It is open, CC BY 4.0, and has BraTS-protocol labels, but needs SRI24 re-registration and a 142 GB download. BraTS-Africa's 35 extra cases are the cleanest format match, but they are small, and the label release is unconfirmed. Both can be used together if each stays above the power threshold.

## Orchestrator notes (2026-11-24)

- **Disk blocks UCSF-PDGM as a whole download:** 142 GB vs ~51 GiB free. Either W6.4 disk hygiene
  first (target ≥ 100 GiB free), or fetch only the non-BraTS-2021 subjects (~108 × ~0.28 GB ≈ 30 GB)
  once the crosswalk names them. Do not start any download before the crosswalk is read.
- **n vs power:** at ~35 (SSA extras) or ~100 (UCSF-PDGM clean pool) the lockbox resolves only large
  conformal violations; the protocol must state that before registration (MasterPlan §B3 rule 6).
- **Lockbox rule:** nothing in this file is a result; no case has been downloaded or looked at.
  The choice between candidates is made on these access/label/overlap facts only.

## Label definition check (2026-11-25, text only — no image or label opened)

**BraTS-Africa, 2023 vs 2024.** Source: Adewole et al., arXiv 2305.19369 (the BraTS-Africa 2023 challenge
paper; PDF text read directly). It states the task is "to automatically segment intracranial gliomas into
three distinct classes using a 3-label system": ET (enhancing tumour), NETC (non-enhancing tumour core,
"includes necrosis, cystic change, calcification"), and the surrounding non-enhancing FLAIR hyperintensity
(SNFH), generated with "the BraTS tumor annotation protocol" from nnU-Net pre-segmentations trained on BraTS
2021, then reviewed by neuroradiologists. No resection-cavity label (pre-operative cohort). That is the
definition our 60 SSA cases already use (TC = ET + NETC, WT = TC + SNFH).

The "2024 BraTS challenge" wording in Musah et al. (arXiv 2508.10905) names only the evaluated regions
(TC, ET, WT) and gives no new definition. It is a modelling paper on the same data, not a re-annotation.
TCIA ships all 146 cases in one package labelled "BraTS 2023 Challenge", so the 35 extra gliomas come
in the same release and format as our 60. **Verdict: same label definition. The SSA 2023-vs-2024 question
is closed on the text.** Still to check per case after download: the label values are exactly {0, 1, 2, 3},
and every case among the 35 has a non-empty mask (20 of the 95 were the withheld 2023 test set, and TCIA
says labels exist for all 146).

**UCSF-PDGM.** TCIA says the segmentations were made for BraTS 2021 by the same pipeline (ensemble
pre-segmentation, manual correction, two neuroradiologists). These are BraTS 2021 conventions: labels 1
(NCR/NET), 2 (ED), 4 (ET), the same raw format as our BraTS 2021 data before remapping, so the existing
preprocessing applies. Still to check per case after download: label values {0, 1, 2, 4} (not 3), and
that the 199 never-BraTS patients carry segmentations at all. The TCIA text implies they do but does not
list them per case.

**Open, needs the author:** the download decision (BraTS-Africa 1.6 GB fits; UCSF-PDGM's 235 clean
patients ≈ 66 GB at ~0.28 GB each, against the current free disk). The per-case checks above are the first
thing run on the downloaded labels, before any image is read. They are label-format checks, not results.
The lockbox rule still holds: no model output on these cases before W3.1 registration.
