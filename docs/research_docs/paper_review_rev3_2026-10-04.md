# Review of rev3: `NeuroVision-X_IEEE_Paper_rev3.docx` (2026-10-04)

Checks rev3 against the rev2 edit list (`paper_review_rev2_2026-10-04.md`), then rereads every
sentence that changed between rev2 and rev3.

**Verdict.** Every Part A and Part B edit has landed, and the old text is gone. There are no new
factual errors. Three small items are left, plus the PDF and submission checks, which carry over
from the rev2 review.

How it was checked:
- Text was extracted from both .docx files, including the math runs.
- Each Find string had to be absent from rev3, and each Replace string had to be present.
  Whitespace was normalised before matching.
- Every sentence that differs between rev2 and rev3 was read in full.
- Stale phrases were searched for: "170 GPU", "46 cases", "twice as many", "65%", "toolkit",
  "not yet run" and "about 30%".

## One correction to the rev2 review

The rev2 review's A2 said that the abstract of [33] gives no CRC comparison. **That was wrong.**
The abstract says: "WSR … re-certifies six Tier-1 organs with 25 local cases, versus 30–40 for
Hoeffding–Bentkus. CRC needs 10–15". Your rewrite of A2b, "against 10–15 for CRC", is supported,
and it is better than the sentence the rev2 review proposed. Keep it.

## Remaining edits

| # | Location | Find | Replace with | Why |
|---|---|---|---|---|
| R1 | §VIII-C, second caution | `[33] re-certified six organs with 25 local cases under a betting-based RCPS bound, against 10–15 for CRC.` | `[33] re-certified six organs at α = 0.10 with 25 local cases under a betting-based RCPS bound, against 10–15 for CRC.` | Without α, "25" sits beside "45 cases at α = 0.05" in the next sentence and looks like it breaks the model-independent minimum. At α = 0.10 the minimum is 22, so 25 is consistent. |
| R2 | §III-A | `are labelled post-review and exploratory where they are quoted.` | `are labelled post-review or exploratory where they are quoted.` | The ref-ET analysis (§VI-A, Table IV †) is labelled only "exploratory", and the §VII confidence check only "post-review". Changing "and" to "or" makes the promise true with one edit. |
| R3 | Abstract (only if the venue caps it at 250 words) | `(0.70–0.91× α; two intervals include α)` | delete | The abstract is now 251 words. MELBA does not need this cut. |

## Read and confirmed (no action)

- **§X:** "all but one evaluated cell" is correct. The sentence's own clause, "where the model is
  not hopeless", already excludes PED TC, so the one remaining failure is SSA TC at α = 0.05.
- **§IX-E:** "rotational and non-rigid disagreement" does not conflict with "our rigid one".
- **§V-G bound:** ln 0.1 / ln 0.95 = 44.9, which rounds up to 45. ln 0.1 / ln 0.90 = 21.9, which
  rounds up to 22. The CRC values of 19 and 9 come from k ≥ 1/α − 1.
- **R6 numbers:** Table XI, §VIII-C and Fig. 5 agree with each other and with note 57.
- **Word count:** about 16,900.

## Still open from the rev2 review

- **Part C, PDF by eye:** Table IV width (6 columns), equation rendering, page breaks in
  Tables XI and XIII, and the legibility of Fig. 5.
- **Part D, at submission:** three placeholders remain: the Zenodo DOI (§III-A), weights and logits
  (Data availability), and the Acknowledgment.
- **Part D:** the GPU-hour total (186) needs updating at submission, once D3 and nnU-Net add hours.
- **Part E:** where to slot nnU-Net, D3 and TTA as each one lands. The paper is internally
  consistent without them.
