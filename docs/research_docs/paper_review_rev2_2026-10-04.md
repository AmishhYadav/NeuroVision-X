# Review of rev2: `NeuroVision-X_IEEE_Paper_rev2.docx` (2026-10-04)

This checks rev2 against the rev1 review (`paper_review_rev1_2026-10-03.md`, in git history at
`8b0ad9d`). Each "Find" string matches rev2 verbatim.

**Verdict: the content is ready. What remains is submission logistics and the analyses the paper
already declares as not run.** Every edit requested in the rev1 review was applied correctly, with
the exceptions listed in Part A. The new numbers check out against the data:
- Table IV's new ref-ET column (all 8 values, and all 8 CIs in its footnote).
- The empty-ET paragraph.
- The R3 and R8 text.
- The bar-sensitivity rates.

Structure is intact: 16 equations, 18 tables, 7 figures. The abstract is 248 words.

---

## Part A — Still to do in the text

| # | Location | Find | Replace with / action |
|---|---|---|---|
| A1 | §III-A | `are listed as outstanding in Section XI rather than implied.` | `are listed as outstanding in Section XI rather than implied; those run before submission (the enhancing-tumour comparison on cases with reference enhancing tumour, the registration-offset diagnosis and the confidence-head check) are labelled exploratory where they are quoted.` Without this, the "post-review" wording in §VII and §IX-E has no definition. |
| A2 | §V-G | `46 and 23 cases at α = 0.05 and 0.10 as computed in [33]` | Still unverified (rev1 review B1). Open [33] and confirm it prints 46 and 23. If it does not: `about 46 and 23 cases at α = 0.05 and 0.10 for the betting bound, reached only if every calibration loss were zero,` |
| A3 | §XI and ref. [61] | `needs the later BraTS-PEDs release [61] that distributes the four sub-region labels separately.` | [61] is real: arXiv:2404.15009, Kazerooni et al., and the title matches. Confirm that its data description lists four separate labels (enhancing, non-enhancing, cystic, oedema) before relying on it. |
| A4 | §III-A | `[Zenodo DOI, to be minted at submission]` | Mint the DOI at submission |
| A5 | Data availability | `[Author to confirm: whether trained weights and saved logits are released, and where.]` | Decide |
| A6 | Acknowledgment | `[Author to add: faculty guide, department and any funding.]` | Fill in |

## Part B — Check by eye in the exported PDF

- **Table IV now has 6 columns.** At IEEE single-column width it will probably overflow or wrap
  badly. Either make it a two-column-spanning table, or move the ref-ET column into a separate
  small table.
- Look over the 16 equations once more in the PDF. The XML is well-formed, but the extracted text
  cannot show how fractions and hats actually render.

## Part C — Before submission, not text edits

1. **Venue.** rev2 is 16,553 words (16 pages per the file metadata). That suits MELBA; it is
   far over IEEE TMI or JBHI.
2. **Reproducibility.** The empty-ET, R3 and R8 numbers now in the paper come from hand-run
   diagnostics. Their code sits in `outputs/` (gitignored), so it is **not** in the public repo that
   the Data-availability statement points to. Before submission, turn each one into a committed
   script via `py-implementer`, one at a time:
   - (a) the ref-ET comparison table
   - (b) the R3 brain/tumour offset analysis
   - (c) the R8 confidence-channel diagnostic
3. **Still declared "not run" in §XI.** All of these are honest as written. In order of value per
   hour:
   - R5: Dice and precision of the recalibrated masks, and RCPS floors (saved logits, CPU).
   - R6: SSA TC at α = 0.05, k ≥ 39 (minutes).
   - R7: pairwise-Dice detector and AURC (needs the MC passes; check whether they still exist).
   - nnU-Net (Gate A, training in progress), then R2 (nnU-Net as a second audited model).

## Verified in rev2 (no action)

- rev1-review edits A0a–A0j, A1a–A1h, A2a–A2b and B2–B5: all applied as written.
- Table IV ref-ET column: +0.0165 / +0.0057 / +0.0109 / +0.0140 / −0.0032 / +0.0145 / +0.0022 /
  +0.0042, with the CIs in the footnote. All match `statistics.py` recomputed from the per-case
  CSVs.
- No `p_holm`, caret, `PENDING`, "pre-registered TOST", "withheld" or R-number labels remain.
- The "below chance" wording is gone; WT 0.477 is now restored and explained.
- Reference [61] exists (see A3).

Source: [arXiv 2404.15009](https://arxiv.org/abs/2404.15009v4)
