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
