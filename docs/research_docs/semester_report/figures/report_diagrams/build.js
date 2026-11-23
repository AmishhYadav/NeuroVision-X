// Builds the NeuroVision-X final project report (.docx).
// Formatting: Times New Roman, black text only, A4, 1-inch margins, 12 pt body, 1.5 line spacing.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ImageRun, PageBreak, TableOfContents, Footer, PageNumber, LevelFormat,
  SequentialIdentifier, TabStopType, VerticalAlign,
} = require("docx");
const { estimatePageNumbersWith } = require("docx/layout");

const FONT = "Times New Roman";
const ROOT = "/Users/amish/NeuroVision-X";
const FIG = path.join(__dirname, "figs");
const SEM = path.join(ROOT, "docs/research_docs/semester_report/figures");
const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins

// ---------------------------------------------------------------- helpers
function runs(text, opts = {}) {
  // Inline **bold** and *italic* markup.
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: runs(text), ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
let numListId = 0;
function NL(items) {
  const ref = `num${numListId++}`;
  numberingConfigs.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) }));
}
const numberingConfigs = [];

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption, widthIn = 6.2) {
  const { w, h } = pngSize(file);
  const wpx = Math.round(widthIn * 96);
  const hpx = Math.round((wpx * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ style: "Caption", alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Figure ", bold: true }), new SequentialIdentifier("Figure"),
        new TextRun({ text: ". ", bold: true }), ...runs(caption)] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, bold) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { line: 240, before: 0, after: 0 }, alignment: (i > 0 && /^[\d.,~+\u2212\-\u2013\s%\u00d7()\/*M]*$|^\u2014$|^default$/.test(t)) ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(t, { size: 20, bold: bold || undefined }) })],
  });
  return [
    new Paragraph({ style: "Caption", alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: "Table ", bold: true }), new SequentialIdentifier("Table"),
        new TextRun({ text: ". ", bold: true }), ...runs(caption)] }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
    }),
    new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }),
  ];
}
const blank = (n = 1) => Array.from({ length: n }, () => new Paragraph({ children: [] }));

// ---------------------------------------------------------------- title page
const titlePage = [
  ...blank(3),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "FINAL PROJECT REPORT", bold: true, size: 28 })] }),
  ...blank(1),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "NeuroVision-X", bold: true, size: 44 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120, line: 360 }, children: [new TextRun({ text: "Three-Dimensional Brain Tumour Segmentation with a Distribution-Free Error Bound and a Refusal Gate: Where It Holds, Where It Breaks, and What a New Site Needs", size: 30 })] }),
  ...blank(3),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Submitted by", size: 24, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: "Amish Yadav", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Course Project, Semester 2026", size: 24 })] }),
  ...blank(4),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "November 2026", size: 24 })] }),
  ...blank(2),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Research software. Not a medical device. Not for clinical use.", italics: true, size: 22 })] }),
];

// ---------------------------------------------------------------- abstract
const abstract = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Abstract")] }),
  P("Deep-learning models for brain tumour segmentation are usually validated on clean, curated data and reported as a single accuracy number. In use they meet scans from other hospitals, scanners and patient populations, where they can fail without any warning. This project, NeuroVision-X, builds a complete three-dimensional brain tumour segmentation system on multi-modal MRI (T1, T1CE, T2 and FLAIR) and then measures, under pre-registered rules, how far its reliability machinery can be trusted when the data changes."),
  P("The segmentation model is a dual encoder, a 3D convolutional network and a Swin Transformer, joined by adaptive gated cross-attention fusion at four scales, a U-Net decoder and three output heads (34.9 million parameters). It was trained on a frozen random split of the BraTS 2021 training set (875 / 187 / 189 cases) and tested on two external cohorts, BraTS-Africa (60 cases) and paediatric BraTS (99 cases), and on real clinical DICOM of 40 patients. Around the model sits a full clinical pipeline: DICOM ingest, co-registration to the SRI24 atlas, skull stripping, label-free input quality control, a quality-control model, conformal risk control, and a refusal gate that returns PROCEED, CAUTION or REFUSE, served through a web application with a 3D viewer and an atlas-based anatomical report."),
  P("Against a U-Net trained with the same recipe, the model improves enhancing-tumour Dice by +0.0267 (95% CI 0.0143 to 0.0432, Holm-adjusted p = 1.4 × 10⁻²¹, n = 189), replicated at a second seed pair and about 80% attributable to architecture rather than parameter count. A properly configured nnU-Net, however, beats it (0.8879 vs 0.8709), so the architecture claim was withdrawn. The project's founding hypothesis, that conditioning the fusion gate on branch disagreement produces the gain, returned a pre-registered null. A distribution-free bound on the fraction of tumour the mask misses holds in distribution in all six tested cells, and breaks under distribution shift by an amount ordered by the size of the shift: about 1.1 times the nominal level on BraTS-Africa whole tumour, up to 11.5 times on paediatric tumour core. End to end, the pipeline returns a usable mask for 85%, 78% and 24% of in-distribution, African and paediatric studies, with silent failure rates of 4.2%, 18.3% and 49.5%; the refusal gate cannot see cohort-level shift. A counterfactual analysis shows that 10 to 40 labelled local cases restore the bound at a new site, except for paediatric tumour core, which cross-fitted fine-tuning recovers."),
  P("**Keywords:** brain tumour segmentation, BraTS, Swin Transformer, conformal risk control, distribution shift, refusal gate, pre-registration, clinical imaging pipeline."),
];

// ---------------------------------------------------------------- front matter lists
const frontLists = [
  new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("Table of Contents")] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Figures")] }),
  new TableOfContents("List of Figures", { hyperlink: true, captionLabelIncludingNumbers: "Figure" }),
  new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Tables")] }),
  new TableOfContents("List of Tables", { hyperlink: true, captionLabelIncludingNumbers: "Table" }),
  new Paragraph({ style: "FrontTitle", pageBreakBefore: true, children: [new TextRun("List of Abbreviations")] }),
  ...table("Abbreviations used in this report.", ["Abbreviation", "Meaning"], [
    ["BraTS", "Brain Tumor Segmentation challenge and dataset"],
    ["SSA / PED", "BraTS-Africa (sub-Saharan Africa) cohort / paediatric BraTS cohort"],
    ["ET / TC / WT", "Enhancing tumour / tumour core / whole tumour sub-regions"],
    ["NCR, ED", "Necrotic and non-enhancing tumour core; peritumoural oedema"],
    ["CNN", "Convolutional neural network"],
    ["Swin", "Shifted-window vision Transformer"],
    ["CRC", "Conformal risk control"],
    ["CI", "Confidence interval (95%, paired bootstrap unless stated)"],
    ["HD95", "95th-percentile Hausdorff distance (mm)"],
    ["QC", "Quality control"],
    ["OOD", "Out of distribution"],
    ["TTA", "Test-time augmentation"],
    ["AMP", "Automatic mixed precision"],
    ["AUROC", "Area under the receiver operating characteristic curve"],
    ["DICOM, DICOM-SEG", "Digital Imaging and Communications in Medicine; its segmentation object"],
    ["SRI24", "SRI24 multichannel brain atlas"],
    ["GPU-h", "GPU hours (Kaggle Tesla T4)"],
  ], [2.2, 6.8]),
];

// ---------------------------------------------------------------- chapter 1
const ch1 = [
  H1("1. Introduction"),
  H2("1.1 Background"),
  P("Gliomas are the most common primary malignant brain tumours in adults. Their treatment, from surgical planning to radiotherapy and follow-up, depends on knowing where the tumour is and how large each of its parts is. On magnetic resonance imaging (MRI) a glioma is not one object but three nested sub-regions: the **enhancing tumour (ET)**, which lights up after contrast injection on T1CE; the **tumour core (TC)**, which adds necrotic and non-enhancing tissue; and the **whole tumour (WT)**, which adds the surrounding oedema visible on FLAIR. Outlining these regions by hand in three dimensions takes an expert a long time and different experts disagree, which is why automatic segmentation has been studied for more than a decade."),
  P("The Brain Tumor Segmentation (BraTS) challenge supplies the standard benchmark: skull-stripped, co-registered multi-modal MRI with expert labels. Modern deep networks reach Dice overlaps above 0.9 on whole tumour. But these numbers are measured on data that looks like the training data. Real hospitals use different scanners, field strengths, slice thicknesses and patient populations, and a network that scores well on a benchmark can return a confident and wrong mask on such a scan with nothing in the system raising an alarm."),
  H2("1.2 Problem statement"),
  P("Most published segmentation work reports one accuracy number on a held-out split and stops. That number says nothing about three questions a deployment actually depends on:"),
  ...NL([
    "How large is the error on a particular new study, and can that error be bounded with a guarantee rather than an estimate?",
    "Does that guarantee still hold when the new study comes from a different site or a different population?",
    "Can the system recognise the studies it should not answer, and refuse them, instead of failing silently?",
  ]),
  P("This project builds a full brain tumour segmentation system and answers those questions by measurement, on one in-distribution test set, two external cohorts at increasing distance from the training data, and real clinical DICOM."),
  H2("1.3 Motivation and how the project's aim changed"),
  P("The project started with an architectural idea: a dual-encoder network whose fusion gate reads how much its two branches disagree at each voxel, expected to give better calibrated and better bounded segmentations. Every claim was pre-registered, that is, written down with its decision rule before the data existed. The evidence then reshaped the project three times (Figure 1). The model turned out to be more accurate but not more reliable; its claimed novelty was not what produced the gain; and in the end a properly configured nnU-Net beat it. Instead of hiding these results, the project rebuilt the reliability question as a measurable property of a wrapper around *any* model. The final thesis is:"),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 720, right: 720 }, spacing: { before: 120, after: 160 },
    children: runs("*A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the bound holds in distribution, fails under distribution shift in proportion to the shift, and the gate cannot see cohort-level shift. We measure where it breaks, and what a new site needs in order to restore it.*") }),
  ...figure(path.join(FIG, "fig_timeline.png"), "Project timeline and its three identities. Each change was forced by a measured result."),
  H2("1.4 Objectives"),
  ...NL([
    "Build and train a dual-encoder (CNN + Swin Transformer) segmentation network with gated cross-attention fusion, and compare it fairly against a matched U-Net, a width-matched capacity control, an ablation of its key mechanism, and nnU-Net.",
    "Evaluate it in distribution and on two external cohorts (BraTS-Africa and paediatric BraTS) with paired statistics and multiple-comparison correction.",
    "Wrap it in conformal risk control to obtain a distribution-free bound on the fraction of tumour the mask misses, and measure where that bound holds and where it breaks.",
    "Build a clinical pipeline from a raw DICOM study to a gated result (PROCEED / CAUTION / REFUSE), with input quality control, a quality-control model and an end-to-end error budget.",
    "Determine what a new site would need to restore the bound: local recalibration data, fine-tuning, or both.",
    "Deliver a usable web application with a 3D viewer and an atlas-based anatomical report, and document every result, positive or negative.",
  ]),
  H2("1.5 Scope"),
  P("The project is a college course project (semester demo and report, November 2026) that continues as a capstone (summer 2027) and a journal submission to MELBA. It is research software. It is **not** a medical device, it is not deployed, and nothing it outputs is intended for diagnosis or treatment. The guarantee it studies is an average over studies in distribution, never a promise about an individual patient."),
  H2("1.6 Contributions"),
  B("A controlled study of what a dual-encoder fusion architecture does and does not buy, including a width-matched capacity control, a parameter-matched ablation of the founding mechanism, a two-seed noise floor and a full-recipe nnU-Net comparison, all pre-registered."),
  B("A measurement of conformal risk control under a graded distribution shift (adult glioma → African adult glioma → paediatric tumours) showing that the violation grows with the size of the shift."),
  B("An end-to-end DICOM-in pipeline with a refusal gate, and an honest error budget of what it returns per cohort, including its silent-failure rate."),
  B("A counterfactual estimate of how many labelled local cases restore the bound, a sub-region shown before the run to be unrecoverable by recalibration, and evidence that fine-tuning recovers it."),
  B("A reproducible, tested code base (about 2,670 automated CPU tests), eleven pre-registrations and a public record of every negative result."),
  H2("1.7 Organisation of the report"),
  P("Chapter 2 reviews related work. Chapter 3 describes the data. Chapter 4 gives the methodology: preprocessing, the model, baselines, training, evaluation and the reliability methods. Chapter 5 describes the system design and implementation. Chapter 6 presents the results and Chapter 7 discusses them. Chapter 8 lists limitations, Chapter 9 the engineering lessons, and Chapter 10 concludes with future work."),
];

// ---------------------------------------------------------------- chapter 2
const ch2 = [
  H1("2. Literature Review"),
  H2("2.1 Brain tumour segmentation"),
  P("The BraTS benchmark (Menze et al., 2015; Baid et al., 2021) established multi-modal MRI glioma segmentation as a standard task. The U-Net (Ronneberger et al., 2015) and its 3D form (Çiçek et al., 2016) remain the backbone of most solutions. nnU-Net (Isensee et al., 2021) showed that a carefully self-configuring U-Net, with a long training schedule, heavy augmentation and a large patch, wins most medical segmentation benchmarks without architectural novelty. Its 2024 follow-up, *nnU-Net Revisited*, found that many claimed architectural gains, including those of transformer hybrids such as SwinUNETR, disappear against a properly configured CNN baseline. This project's Gate A experiment reproduces exactly that finding for its own model."),
  H2("2.2 CNN–Transformer hybrids and fusion"),
  P("Transformers capture long-range context through self-attention; the Swin Transformer (Liu et al., 2021) makes this affordable through shifted local windows, and SwinUNETR (Hatamizadeh et al., 2022) applies it to 3D brain tumour segmentation. Hybrid models combine a CNN for local texture with a transformer for global context and merge the two streams by concatenation, addition or gated attention. Gated fusion, where a learned gate weights one stream against the other, is published territory. The novelty this project proposed was narrower: a gate that reads an explicit per-voxel ambiguity signal (branch disagreement and branch entropy) rather than only the branch content."),
  H2("2.3 Uncertainty estimation"),
  P("Common approaches are single-pass predictive entropy, Monte-Carlo (MC) dropout (Gal and Ghahramani, 2016), deep ensembles, learned confidence heads and temperature scaling for calibration. In brain tumour segmentation, ensembles generally beat MC dropout. A recurring finding, confirmed in this project, is that free single-pass entropy is a strong baseline that elaborate mechanisms rarely beat."),
  H2("2.4 Conformal prediction and risk control"),
  P("Conformal prediction gives distribution-free guarantees from a held-out calibration set, assuming only that new data are exchangeable with it. Risk-controlling prediction sets (Bates et al., 2021) and conformal risk control (Angelopoulos et al., 2024) extend this from coverage to any bounded monotone loss, such as the fraction of tumour voxels a mask misses. Under covariate shift the exchangeability assumption fails; weighted conformal prediction (Tibshirani et al., 2019) and Mondrian (group-conditional) calibration (Vovk et al., 2005) are the standard responses."),
  H2("2.5 Closest prior work"),
  P("Three 2026 papers are direct neighbours and limit what this project may claim as new (Table 2). Coverage failure under shift for brain tumour segmentation is therefore already reported; what this project adds is a graded shift across populations, a sub-region shown in advance to be unrecoverable, and an end-to-end pipeline measured against the bound."),
  ...table("Closest prior work and how this project differs.", ["Work", "What it shows", "Difference from this project"], [
    ["Shahid, arXiv 2606.20115 (2026)", "Pooled conformal risk control on FeTS-2022 brain tumours violates the false-negative target at 40% of individual hospitals", "Their shift is between sites within one adult population; ours is across populations, ordered by distance"],
    ["Adhikary et al., arXiv 2608.18193 (2026)", "Per-organ recall control in CT fails after transfer; ~10–40 local cases re-certify, some organs never", "Same shape of answer in CT organs; ours is MRI tumour sub-regions with the infeasible case derived before the run"],
    ["Liu et al., arXiv 2608.10893 (2026)", "Theory for certified selective risk control under bounded covariate shift (text data)", "Their refusal is certified; our gate is not, and we measure that it cannot see cohort shift"],
  ], [2.4, 3.6, 3.6]),
  H2("2.6 Clinical pipelines and structured reporting"),
  P("Turning research segmentation into a clinical tool requires DICOM conversion (dcm2niix), co-registration, atlas registration and skull stripping (for example HD-BET and the BrainLes preprocessing toolkit), and standard output formats such as DICOM-SEG (highdicom). Automated structured reporting of glioma features has been explored by VASARI-auto and BTReport; this project's report layer is related to them, and its measured negative result (a better segmenter does not produce a better report) is reported in Chapter 6."),
];

// ---------------------------------------------------------------- chapter 3
const ch3 = [
  H1("3. Datasets"),
  H2("3.1 BraTS 2021"),
  P("The training data is the BraTS 2021 training set: 1,251 adult glioma cases, each with four co-registered, skull-stripped MRI sequences (T1, T1CE, T2, FLAIR) at 1 mm isotropic resolution on a 240 × 240 × 155 grid, and an expert label map with three classes (necrotic/non-enhancing core, oedema, enhancing tumour). The cases were split once, with a seeded random generator, into **875 training, 187 validation and 189 test** cases, and the split was frozen in configuration (Figure 2). Nothing outside the training split was ever fitted: not the networks, not the calibration temperature, not the conformal threshold and not the refusal gate thresholds, which were all set on validation."),
  P("Because the split is a random split of the BraTS *training* set, the numbers in this report are not comparable to published BraTS leaderboard numbers, which are measured on the official validation set."),
  H2("3.2 External cohorts"),
  B("**BraTS-Africa 2023 (SSA), 60 cases.** Adult gliomas scanned in sub-Saharan Africa, typically with lower image quality and more varied scanners. The label definition matches BraTS 2021, so it is the cleanest test of imaging shift."),
  B("**BraTS-PEDs 2023 (PED), 99 cases.** Paediatric brain tumours: a different disease population, and, as later measured, a different use of the label classes (the paediatric reference counts most of the non-enhancing tumour as core)."),
  B("**RSNA-MICCAI real DICOM, 40 patients.** Original clinical DICOM of patients whose BraTS case lies in the test split, used to measure what the clinical front end costs on real scans. These are the same patients, so this is not external validation."),
  ...figure(path.join(FIG, "fig_data.png"), "Data used in the project: the frozen BraTS 2021 split and the external evaluation sets."),
  ...table("Datasets and their role.", ["Dataset", "Cases", "Role", "Used for fitting?"], [
    ["BraTS 2021 train split", "875", "Train every model", "Yes"],
    ["BraTS 2021 validation split", "187", "Checkpoint selection, calibration, conformal threshold, gate thresholds", "Selection only"],
    ["BraTS 2021 test split", "189", "In-distribution evaluation", "No"],
    ["BraTS-Africa 2023", "60", "External evaluation (imaging shift)", "No (except cross-fitted fine-tune, §6.8)"],
    ["BraTS-PEDs 2023", "99", "External evaluation (population and label shift)", "No (except cross-fitted fine-tune, §6.8)"],
    ["RSNA-MICCAI DICOM", "40", "Real-DICOM front-end validation", "No"],
  ], [2.6, 0.8, 3.8, 2.2]),
  H2("3.3 Preprocessing"),
  P("Each case is reoriented from its own affine, kept at 1 mm isotropic spacing, normalised with a z-score computed over non-zero (brain) voxels for each sequence, and cropped to the bounding box of the brain. Images are stored as float16 and labels as uint8 NumPy arrays with a per-case metadata file recording the crop box and orientation, so predictions can be restored exactly to the original grid. The raw labels are converted to the three nested region channels (ET ⊂ TC ⊂ WT) used for training and scoring. All 1,251 cases were preprocessed without failure (34.2 GB). SHA-256 manifests are committed for the external cohorts."),
];

// ---------------------------------------------------------------- chapter 4
const ch4 = [
  H1("4. Methodology"),
  H2("4.1 Proposed model: NeuroVision-X"),
  P("The model (Figure 3) has five parts. A **3D CNN encoder** of residual blocks produces five feature levels, from a full-resolution stem of 32 channels down to 320 channels at stride 16. A **Swin Transformer encoder** with four stages processes the same input through patch embedding and shifted-window self-attention, providing long-range context. At each of the four Swin levels an **adaptive gated cross-attention fusion block** merges the two streams. A **U-Net decoder** with attention gates upsamples the fused features. Three **heads** produce the outputs: a segmentation head with deep supervision emitting ET, TC and WT logits, plus a confidence head and a boundary head used as auxiliary training signals. The model has **34,911,341 parameters**; most of them are in the CNN encoder (18.9 M) and decoder (12.9 M), while the Swin branch has only 2.0 M and the fusion blocks 1.0 M."),
  ...figure(path.join(FIG, "fig_architecture.png"), "NeuroVision-X architecture: dual encoder, gated cross-attention fusion at four scales, U-Net decoder and three heads."),
  H3("4.1.1 The fusion block and the founding hypothesis"),
  P("Inside each fusion block (Figure 4), the Swin feature is first projected to the CNN width. Windowed cross-attention lets each CNN location query the Swin features around it. A gate generator produces a per-voxel weight *g* between 0 and 1, and the output is the CNN feature plus a learned layer scale times *g* times the attention output. The proposed novelty was the gate's input: besides the two feature maps, it receives an **ambiguity signal** computed by two small probe layers that predict region probabilities from each branch separately, namely their absolute disagreement and the entropy of each. The hypothesis (P2 in the pre-registration) was that this disagreement conditioning is what makes the architecture work. A parameter-matched ablation, the *content-only gate*, removes exactly these inputs (a difference of 6,360 parameters, 0.018%)."),
  ...figure(path.join(FIG, "fig_fusion.png"), "Adaptive gated cross-attention fusion block. The grey disagreement inputs are what the content-only ablation removes.", 5.8),
  H2("4.2 Baselines and controls"),
  P("A claim that one architecture beats another is only as good as its comparison. The project therefore trained a ladder of controls, each differing from the proposed model in one controlled way (Table 4)."),
  ...table("Models trained and compared.", ["Model", "Params", "Purpose"], [
    ["Matched 3D U-Net (MONAI)", "12.9 M", "Same recipe, same data, same seed: the comparison row"],
    ["Capacity control U-Net", "34.8 M", "U-Net widened to the proposed model's size, to separate width from architecture (trained twice)"],
    ["Content-only gate ablation", "34.9 M", "Proposed model without disagreement conditioning: tests the founding hypothesis"],
    ["NeuroVision-X seed 43", "34.9 M", "Second seed: measures the seed-to-seed noise floor"],
    ["U-Net seed 43", "12.9 M", "Second seed of the baseline: replication of the headline"],
    ["NeuroVision-X, heavy augmentation", "34.9 M", "Rotation, gamma, bias field, elastic: does augmentation close the shift gap?"],
    ["nnU-Net v2 3d_fullres", "default", "The field's standard: full 1000-epoch recipe on the same 875 cases"],
  ], [3.0, 1.1, 5.0]),
  H2("4.3 Training"),
  P("All project models share one recipe, fixed after a timing probe showed the proposed model was 10.7 times slower per epoch than the U-Net at 96³ patches: **64³ patches, 80 epochs**, AdamW (learning rate 1 × 10⁻⁴, weight decay 1 × 10⁻⁵), five warm-up epochs followed by a cosine schedule, batch size 1 with four patches per step, automatic mixed precision, and gradient clipping at norm 5. Augmentation is random flips, 90° rotations, ±10% intensity scale and shift, and light Gaussian noise. The loss is a multitask loss: Dice plus binary cross-entropy with deep supervision for segmentation, plus a boundary term (weight 0.3), a confidence term (0.05) and a branch-consistency term on the probe layers (0.1). The checkpoint with the best mean validation Dice is kept; for the deployed model this was epoch 69 of 80."),
  P("Training ran on Kaggle Tesla T4 GPUs (14.56 GiB usable) in sessions capped at 12 hours. The trainer therefore supports **full resume** from any point: model, optimiser, scheduler, mixed-precision scaler, epoch, global step, every random-number-generator state and the experiment-tracking run ID are checkpointed atomically. The deployed model took 23.1 GPU-hours over three chained sessions."),
  H2("4.4 Inference and post-processing"),
  P("Whole volumes are segmented with a sliding window of 64³, overlap 0.5 and Gaussian blending. Post-processing enforces the nesting ET ⊂ TC ⊂ WT and removes connected components smaller than 50 voxels, and predictions are restored to the original 240 × 240 × 155 grid with the saved crop box. Inference takes about 4 s per case on a T4 and about 1.4 minutes per case on the author's laptop CPU. Logits for every evaluated split are saved in float16, so all later re-scoring (new metrics, conformal analysis, recalibration) needs no new inference."),
  H2("4.5 Evaluation metrics"),
  B("**Voxel-wise Dice** per region, the BraTS convention up to 2022: 2|A ∩ B| / (|A| + |B|). An empty prediction of an empty reference scores 1."),
  B("**Lesion-wise Dice**, the BraTS convention since 2023: each connected lesion is matched separately, so missed small lesions and spurious false-positive lesions are penalised. It is reported alongside voxel Dice."),
  B("**HD95**, the 95th-percentile Hausdorff distance in millimetres, measuring boundary error."),
  B("**Usable mask**: a study counts as usable when both WT and TC Dice are at least 0.7; **silent failure** is a study that the gate accepts but whose mask is not usable."),
  B("**Realised risk**: the mean fraction of reference tumour voxels that the conservative mask misses, compared with the nominal level α."),
  H2("4.6 Statistical protocol and pre-registration"),
  P("Every comparison is **paired** over the same cases. Confidence intervals come from a paired bootstrap that resamples case indices (10,000 resamples), p-values from the Wilcoxon signed-rank test, and every family of comparisons is corrected with the Holm–Bonferroni method. A difference is called better or worse only if its CI excludes zero and its Holm-adjusted p is below 0.05; otherwise it is *inconclusive*, which is never read as equivalence."),
  P("Each experiment that could change a claim was **pre-registered**: a document stating the hypothesis, the endpoints, the decision rule and an abort budget was committed to version control, with a timestamp, before the first number existed. Eleven such documents govern the results in this report. This discipline is why the negative results below can be trusted as much as the positive ones."),
  H2("4.7 Uncertainty and quality signals"),
  P("Four per-voxel uncertainty signals were compared as error localisers: single-pass predictive entropy (free), 10-sample MC dropout, the disagreement between the two branches' probe predictions, and the model's trained confidence head. For per-study quality, a separate 3D CNN, the **QC model**, was trained to predict a mask's own Dice from the image and the mask, using synthetically degraded masks so that it never needed the ground-truth label at inference."),
  H2("4.8 Conformal risk control"),
  P("Conformal risk control converts the network's voxel probabilities into a mask with a guarantee (Figure 5). A threshold λ is chosen on the calibration set (BraTS validation, 187 cases) as the largest value whose adjusted miss rate, (n·R(λ) + 1)/(n + 1), stays at or below the target α. Frozen and applied to new studies, it guarantees that the *expected* fraction of tumour voxels missed is at most α, provided the new studies are exchangeable with the calibration set. The project used α = 0.05, 0.10 and 0.20 for WT and TC. The guarantee is a theorem and holds for any model, even a poor one; the interesting question is what happens when exchangeability is broken by distribution shift."),
  ...figure(path.join(FIG, "fig_conformal.png"), "Conformal risk control applied to the mask's miss rate."),
  H2("4.9 Local recalibration and fine-tuning"),
  P("To ask what a new site would need, each shifted cohort was split 1,000 times with a seeded generator; the threshold was refitted on k of the cohort's own labelled cases (k = 5 to 49) and the miss rate measured on the rest. This is a **counterfactual** analysis on public data, not a site study. Separately, the deployed model was **fine-tuned** on half of each external cohort and scored on the other half, then the halves were swapped (two-fold cross-fitting), so that every case was scored by a model that never saw it."),
  H2("4.10 Refusal gate"),
  P("The gatekeeper combines signals into PROCEED, PROCEED WITH CAUTION or REFUSE. Live signals are label-free input quality control (twelve checks on intensity, geometry, slice thickness and registration), the QC model's predicted Dice, and an intended-use rule (adults only, read from the DICOM PatientAge field). The conformal band and an input-statistics out-of-distribution score are shown but are display-only, because each failed its pre-registered rule for refusing studies. All thresholds were fitted on validation only. A refused job is treated as a successful outcome of the pipeline, never as an error."),
];

// ---------------------------------------------------------------- chapter 5
const ch5 = [
  H1("5. System Design and Implementation"),
  H2("5.1 Software architecture"),
  P("The code base (Figure 6) is a Python 3.11 package with four parts in separate dependency environments so that none can break another: the research path (training, inference, metrics, statistics), the clinical path (DICOM ingest, registration, skull stripping, gating, export), the interpretable layer (atlas, localisation, reports) and the web application. Every script is driven by Hydra configuration; models, losses and fusion blocks are selected by name from registries, so an experiment is a configuration change rather than a code change. No path is hard-coded and the same code runs on macOS and Linux, on CPU for tests and on CUDA for training."),
  ...figure(path.join(FIG, "fig_software.png"), "Software architecture of NeuroVision-X."),
  ...table("Technology stack.", ["Area", "Tools"], [
    ["Language and deep learning", "Python 3.11, PyTorch, MONAI (transforms, sliding-window inference, metrics)"],
    ["Configuration and tracking", "Hydra, Weights & Biases"],
    ["Medical imaging I/O", "NiBabel, SimpleITK, dcm2niix, highdicom (DICOM-SEG)"],
    ["Clinical preprocessing", "BrainLes preprocessing (ANTs registration, SRI24 atlas), HD-BET skull stripping"],
    ["Analysis", "NumPy, Pandas, SciPy, scikit-image, Captum (explainability)"],
    ["Baseline", "nnU-Net v2 (separate environment, Kaggle)"],
    ["Web application", "FastAPI backend; React, TypeScript, Vite and Tailwind frontend; 3D rendering in the browser"],
    ["Quality", "pytest (about 2,670 CPU tests in about 70 s), Ruff, Black, vitest, headless-Chrome end-to-end tests"],
    ["Compute", "Kaggle Tesla T4 for training; Apple M4 laptop (CPU) for everything else"],
  ], [2.6, 6.4]),
  H2("5.2 The clinical pipeline"),
  P("The clinical pipeline (Figure 7) accepts a zipped DICOM study through the web page at /clinical. **Ingest** converts each series to NIfTI with dcm2niix, assigns the four sequence roles from a rule table and reads the patient's age. **Input QC** runs before registration, then the study is co-registered to its T1CE series, registered to the SRI24 atlas that BraTS itself uses, and skull-stripped with HD-BET, so that it lands in the same space as the training data. Input QC runs again, the model segments the study, the QC model and conformal band are computed, and the **gatekeeper** decides. Outputs include a 3D digital twin of the brain with tumour and atlas structure shells and uncertainty and Grad-CAM layers, a structured report, a DICOM-SEG object that loads in standard viewers, and a downloadable bundle. A single clinical job peaks at about 12 GB of memory on the laptop."),
  ...figure(path.join(FIG, "fig_pipeline.png"), "Clinical pipeline from a DICOM study to a gated result."),
  H2("5.3 Interpretable report layer"),
  P("The report layer reads a mask in SRI24 atlas space (122 merged anatomical structures) and computes, for each case, which structures the tumour involves and what fraction of each structure is affected, its lobe, laterality and midline crossing, sub-region volumes and shape descriptors, and its distance to functionally eloquent structures from a source-derived knowledge base. The report carries its non-diagnostic disclaimer, atlas version, provenance and a list of what it does not claim as required fields. Midline shift was deliberately not implemented, because no public MRI ground truth exists to validate it. Two atlas traps were found and fixed during this work: the SRI24 atlas is mirrored front-to-back relative to BraTS indexing, and a brain-mask overlap check cannot detect a left–right mirrored atlas (it scores higher on the wrong orientation), so laterality is verified from 56 labelled structure pairs instead."),
  H2("5.4 Web application"),
  P("The FastAPI backend serves precomputed research results and runs live clinical jobs; the React frontend provides a research slice viewer with ground-truth, prediction and uncertainty overlays, the /clinical upload page with a gatekeeper panel and refusal banner, the 3D digital twin (Figure 8), the report pages and export. The interface wording was checked so that no string says \"guaranteed\" without \"on average, in distribution, not per patient\"."),
  ...figure(path.join(ROOT, "docs/archive/figures/twin_clinical_9c2cc294_front_separated.png"), "The 3D digital twin of a real DICOM study that the gate passed (PROCEED): separated hemispheres, predicted oedema and involved atlas structures."),
  H2("5.5 Engineering practices"),
  B("**Tests first for every component.** Every model part has a CPU shape test on tiny tensors that runs in under a second; losses and metrics are tested against hand-computed values; data tests use synthetic volumes. The full suite runs on a laptop CPU in about 70 seconds."),
  B("**Resume and provenance.** Every GPU run is pinned to a specific git commit and checked for resume correctness by seven dedicated tests; a four-second end-to-end smoke test runs before every GPU session."),
  B("**Analysis fixes are verified on real data.** A fix is accepted only when re-running the real analysis moves the output in the predicted direction, after a calibration bug once survived a green test suite."),
  B("**Caches versus results.** Volume-sized arrays (logits, predictions) are treated as rebuildable caches while their checkpoint exists; small CSV results are kept forever."),
  H2("5.6 Compute"),
  P("All training ran on the Kaggle free tier (Tesla T4, about 30 GPU-hours per week), roughly 210 GPU-hours in total across 18 tracked training runs, including about 59 GPU-hours for the nnU-Net baseline. Everything else, including evaluation, calibration, statistics, the clinical pipeline and the web application, ran on an Apple M4 laptop CPU."),
];

// ---------------------------------------------------------------- chapter 6
const ch6 = [
  H1("6. Results"),
  P("Every number in this chapter is taken from the project's claims-and-evidence table, in which each claim is tied to the artifact that produced it. Results are reported whether they support the original hypothesis or not."),
  H2("6.1 Segmentation accuracy in distribution"),
  P("On the 189-case test split, NeuroVision-X improves enhancing-tumour Dice over the matched U-Net by **+0.0267** (95% CI 0.0143 to 0.0432; Holm-adjusted p = 1.4 × 10⁻²¹) and tumour-core Dice by +0.0103; the whole-tumour difference is inconclusive (Table 6, Figure 9). The result replicates at a second seed of both models (+0.0247, CI 0.0116 to 0.0420). A second seed of NeuroVision-X alone moves test ET Dice by only +0.0021, so the margin is about thirteen times the seed noise floor. Flip test-time augmentation changes ET Dice by +0.0007, not resolvable at n = 189, so the margin is not an artefact of inference settings."),
  ...table("Test-set results (BraTS 2021 test split, n = 189). Higher Dice and lower HD95 are better.", ["Metric", "Matched U-Net", "NeuroVision-X", "nnU-Net v2"], [
    ["Dice ET", "0.8442", "0.8709", "0.8879"],
    ["Dice TC", "0.9058", "0.9161", "0.9247"],
    ["Dice WT", "0.9276", "0.9321", "0.9424"],
    ["Lesion-wise Dice ET", "—", "0.7550", "0.7859"],
    ["HD95 ET (mm)", "4.91", "4.20", "—"],
  ], [3, 2, 2, 2]),
  ...figure(path.join(SEM, "fig_dice_vs_baselines.png"), "Mean test Dice per region for the matched U-Net, NeuroVision-X and nnU-Net v2."),
  H2("6.2 Where the gain comes from"),
  P("The width-matched capacity control (34.83 M vs 34.91 M parameters) recovers only +0.0055 of the ET gain, so about **80%** of it is architecture rather than capacity (0.79 and 0.80 over two independent training draws). Within the architecture, the content-only gate ablation scores within noise of the full model (+0.0022, CI −0.0067 to +0.0152): the gain comes from **gated cross-attention fusion itself**, and the disagreement conditioning, the project's founding hypothesis, adds nothing measurable (Figure 10). This is a pre-registered null and is reported as one."),
  ...figure(path.join(FIG, "fig_decomposition.png"), "Decomposition of the enhancing-tumour Dice gain over the matched U-Net, with nnU-Net for reference.", 5.6),
  H2("6.3 The strong baseline: nnU-Net"),
  P("nnU-Net v2, trained with its default plans and full 1000-epoch recipe on the same 875 cases, beats NeuroVision-X on both pre-registered co-primary endpoints: voxel ET Dice 0.8879 vs 0.8709 (−0.0170, CI −0.0309 to −0.0069, p_holm = 1.0 × 10⁻⁷) and lesion-wise ET Dice 0.7859 vs 0.7550 (−0.0309, CI −0.0605 to −0.0020). It is better on every Dice row. Mirroring at test time does not explain the gap (without it nnU-Net still leads by 0.0113). nnU-Net beats the matched U-Net by +0.0437, so the weak link is the project's shorter training recipe (80 epochs, 64³ patches), not the U-Net topology. nnU-Net also used about 2.7 times the GPU time (about 59 vs 22 T4-hours). By the rule fixed in advance, **the architecture claim is withdrawn**: the +0.0267 gain is real but holds only against a matched-recipe U-Net."),
  H2("6.4 Lesion-wise scoring"),
  P("Under lesion-wise scoring the in-distribution gain over the matched U-Net is larger (ET +0.0508, TC +0.0371; exploratory, not pre-registered) and comes from fewer spurious lesions (0.32 vs 0.47 false-positive ET lesions per case) rather than better recall. Voxel Dice overstates absolute performance for both models by 0.10 to 0.32; NeuroVision-X whole tumour reads 0.9321 voxel-wise and only 0.7183 lesion-wise, because one large lesion dominates the voxel count."),
  H2("6.5 Out of distribution"),
  P("Accuracy falls with distance from the training data (Table 7, Figure 11), and the ET gain over the U-Net does not transfer: on the external cohorts it disappears, and tumour-core Dice is significantly worse than the U-Net's on pooled shifted data (−0.0333, p_holm = 0.0132). Heavier simulated augmentation does not close the gap: 12 of 12 pre-registered comparisons are inconclusive, and the pooled tumour-core change (+0.0117) is the same size as seed noise."),
  ...table("NeuroVision-X voxel Dice by cohort.", ["Cohort", "n", "Dice ET", "Dice TC", "Dice WT"], [
    ["BraTS test (in distribution)", "189", "0.8709", "0.9161", "0.9321"],
    ["BraTS-Africa (SSA)", "60", "0.7784", "0.7846", "0.8959"],
    ["Paediatric BraTS (PED)", "99", "0.5634", "0.4394", "0.8490"],
  ], [3.4, 0.8, 1.6, 1.6, 1.6]),
  ...figure(path.join(FIG, "fig_ood.png"), "NeuroVision-X accuracy on the in-distribution test set and the two external cohorts.", 5.6),
  H2("6.6 Uncertainty mechanisms"),
  P("Single-pass predictive entropy was never beaten as a per-voxel error localiser. Ten-sample MC dropout is statistically equivalent to it (paired equivalence test at a margin of 0.03 AUROC fixed in advance), branch disagreement is worse (by 0.157 AUROC on SSA and 0.056 on PED against MC dropout), and the trained confidence head is worse on every region (ET 0.855 vs 0.912; uninformative on WT). Branch disagreement does carry some per-voxel information that entropy lacks (residualised AUROC 0.578, 0.569 and 0.677 on test, SSA and PED), but adding it to a fitted detector did not improve it. Calibration, boundary accuracy and MC-dropout risk-coverage were all within noise of the U-Net against a temperature-scaled baseline: the model is more accurate, not more reliable."),
  H2("6.7 The error bound: where it holds and where it breaks"),
  P("Calibrated on validation and applied frozen, the conformal bound **holds in distribution in all six cells** (WT and TC at three α levels), with realised risk at 0.64 to 0.96 times nominal, for both NeuroVision-X and the U-Net. Its cost is small: at α = 0.05 the conservative mask grows 1.10 times (WT) and 1.18 times (TC), and at looser α it is even smaller than the default mask. Under shift the bound **fails, by an amount ordered by the size of the shift** (Figure 12): BraTS-Africa whole tumour about 1.1 times nominal, paediatric whole tumour 1.4 to 1.9 times, and paediatric tumour core 3.5 to 11.5 times. Seven of twelve shifted cells are violated for each model. Part of the paediatric tumour-core failure is a difference in label definition rather than imaging alone: the paediatric reference counts most non-enhancing tumour as core (median 84% of the core), which the adult convention calls oedema."),
  ...figure(path.join(SEM, "fig_realised_vs_nominal.png"), "Realised versus nominal risk for whole tumour and tumour core per cohort. Points above the diagonal violate the bound."),
  H2("6.8 What a new site would need"),
  P("In the counterfactual recalibration analysis, about **10 to 40 local labelled cases**, depending on α, restore the bound for whole tumour on BraTS-Africa and paediatric data and for tumour core on BraTS-Africa (Figure 13). **Paediatric tumour core cannot be restored at any k**: even at the lowest possible threshold the model misses 35.6% of the paediatric tumour core, above every α tested. This was derived and written down before the run."),
  ...figure(path.join(SEM, "fig_local_recalibration.png"), "Held-out miss rate against the number of local calibration cases k; crosses mark cells that do not meet the target."),
  P("Changing the model does what recalibration cannot. Cross-fitted fine-tuning on half of each cohort recovers tumour-core Dice on both: BraTS-Africa +0.0433 (0.785 → 0.828) and paediatric +0.3676 (0.439 → 0.807). After fine-tuning, paediatric tumour core becomes restorable by local recalibration at α = 0.10 and 0.20. Most of the paediatric gain is the model learning the paediatric label convention (its predicted core-to-whole-tumour ratio moved from 0.27 to 0.95), not better imaging generalisation. These results are cross-fitted, not external validation; the deployed model is unchanged."),
  H2("6.9 The end-to-end pipeline and its refusal gate"),
  P("Measured through every stage with the deployed, frozen gate (Figure 14), the pipeline returns a usable mask for **85.2%** of in-distribution studies, **78.3%** of BraTS-Africa studies and **24.2%** of paediatric studies, with silent-failure rates of **4.2%, 18.3% and 49.5%**. The gate has decent precision and poor recall, and it is blind to cohort-level shift: on BraTS-Africa it refuses 2 of 60 studies while passing 11 of 12 unusable masks; on paediatric data 23 of 26 refusals are correct but it catches only 32% of unusable masks. Gate acceptance does not repair the bound where it is broken (paediatric tumour-core risk 0.573 on accepted cases at a nominal 0.10). The QC model behind the gate beats entropy in one of five pre-registered cells, loses in two, and becomes more optimistic under shift in all six external cells. An adults-only intended-use rule removes every paediatric study by construction, which is scoping, not detection."),
  ...figure(path.join(SEM, "fig_error_budget.png"), "End-to-end outcome of each study with the frozen refusal gate, per cohort."),
  H2("6.10 Real clinical DICOM and the out-of-distribution score"),
  P("On original DICOM of 40 test-split patients, the clinical front end refuses 40% of studies for thick slices; of the 22 it accepts, only 36% are usable against the BraTS reference, and whole-tumour Dice falls by a median 0.23 relative to the research path. Most of that loss is a median 5.9 mm disagreement between the project's rigid registration to the atlas and BraTS's own, counted against the pipeline by design because the report's anatomy is read in atlas space. A pre-registered out-of-distribution score computed from input statistics flags 60% of BraTS-Africa studies against 9.5% of test studies, so it sees the shifted cohort, but it ranks failed masks barely above chance (AUROC 0.56 to 0.60) and failed its rule for becoming a refusal signal; it stays display-only."),
  H2("6.11 Hypothesis scoreboard"),
  P("Table 8 lists every hypothesis the project tested, in order, with its verdict."),
  ...table("Hypotheses tested and their outcome.", ["Hypothesis", "Verdict", "Key evidence"], [
    ["Dual encoder beats a matched U-Net on Dice", "Supported", "ET +0.0267, TC +0.0103; WT inconclusive"],
    ["…with better calibration, boundaries, risk-coverage", "Refuted", "All within noise"],
    ["The gain is architecture, not width", "Supported", "~80% architecture, two draws"],
    ["Disagreement conditioning produces the gain (founding hypothesis)", "Null", "+0.0022, CI −0.0067 to +0.0152"],
    ["A better segmentation gives a better report", "Refuted", "1 of 25 report metrics"],
    ["The gain transfers to external cohorts", "Refuted", "Gone on SSA and PED; TC worse"],
    ["Disagreement detects failure beyond entropy", "Partial", "Per voxel yes, per case no"],
    ["Disagreement matches MC dropout at a tenth of the cost", "Refuted", "Worse; entropy equals MC dropout"],
    ["Conformal bound holds in distribution", "Supported", "6 of 6 cells"],
    ["…and survives distribution shift", "Refuted", "Breaks, graded by shift size"],
    ["QC model beats entropy at flagging bad masks", "Mixed", "1 win, 1 loss, 1 borderline of 5"],
    ["The refusal gate makes the pipeline safe under shift", "Refuted", "Silent failure 4.2 / 18.3 / 49.5%"],
    ["Heavy augmentation closes the shift gap", "Null", "12 of 12 inconclusive"],
    ["Local labels restore the bound", "Supported*", "10–40 cases, except PED TC (counterfactual)"],
    ["Front end costs ≤ 0.05 WT Dice on real DICOM", "Refuted", "Median −0.23"],
    ["Input OOD score earns a refusal role", "Refuted", "Sees cohort, not failed study"],
    ["Fine-tuning recovers tumour core", "Supported*", "SSA +0.043, PED +0.368 (cross-fitted)"],
    ["The gain survives nnU-Net", "Refuted", "nnU-Net ET 0.8879 vs 0.8709"],
  ], [4.4, 1.3, 3.3]),
  P("*Counterfactual or cross-fitted, not external validation.", { alignment: AlignmentType.LEFT }),
];

// ---------------------------------------------------------------- chapter 7
const ch7 = [
  H1("7. Discussion"),
  P("The architecture result is modest, in distribution, and relative. Gated cross-attention fusion does produce a real, attributable enhancing-tumour gain over a matched U-Net, but a standard nnU-Net with a longer training recipe exceeds it, the gain does not transfer to new populations, and the mechanism the project was built around is not what produces it. This matches the broader finding of *nnU-Net Revisited* that architectural gains over weak baselines tend to evaporate against a properly configured CNN, and it is the reason the project's contribution moved away from the architecture."),
  P("The more useful result concerns the reliability machinery around any model. A distribution-free bound is only as good as the exchangeability behind it. It held in every in-distribution cell, at small cost, and it degraded in proportion to the shift, from barely noticeable on African adult gliomas to an order of magnitude on paediatric tumour core. A learned refusal gate did not see that shift coming, because each of its inputs is calibrated on the same distribution that the shift moves away from; the QC model even became more confident where the segmentation was worst. What restored the bound was local data: tens of labelled cases at a new site, except where the model's failure is so large that no threshold can meet the target. There, changing the model helped, largely by teaching it a different label convention, which is itself a finding about what \"shift\" means between datasets."),
  P("Two practical lessons follow. First, a single accuracy number, even with a confidence interval, is a weak description of a clinical segmentation tool; the silent-failure rate per population and the cost of the front end on real DICOM say much more. Second, free single-pass entropy was never beaten by the more elaborate uncertainty mechanisms built here, so any new mechanism should be compared against it first."),
  P("Finally, the negative results are the evidence for the final claim rather than an embarrassment within it. Because every experiment was pre-registered with an abort budget and a decision rule, a reader can check that the positive results were not selected after the fact."),
];

// ---------------------------------------------------------------- chapter 8
const ch8 = [
  H1("8. Limitations"),
  ...NL([
    "Most models are single-seed. The noise floor comes from one seed pair, which is a floor, not a variance estimate, so a margin below it is not claimable and every null means no detectable difference at the sample size, not equivalence.",
    "The headline gain holds only against a matched-recipe U-Net; nnU-Net beats the proposed model. nnU-Net was trained once (one fold, one seed) with about 2.7 times the GPU time.",
    "No transformer baseline such as SwinUNETR was trained on the same split, for budget reasons.",
    "The test split is a random split of the BraTS 2021 training set, so the numbers are not comparable to published challenge results; lesion-wise results are exploratory.",
    "The local-recalibration result is a counterfactual on public cohorts, not a site study. The fine-tuning result is cross-fitted, single-seed, and its effect on in-distribution accuracy (forgetting) was not measured.",
    "The real-DICOM study uses the same patients as the test split and measures loss in atlas space, where registration disagreement counts against the pipeline.",
    "Conformal thresholds at α = 0.20 lie at the edge of the search grid, and the calibration split was also used for checkpoint selection, which could in principle make the threshold optimistic (it did not in practice).",
    "The anatomical eloquence field is degenerate on this cohort (every case is near an eloquent structure), so it carries no per-case information.",
    "The demonstration PROCEED study is a BraTS training patient.",
    "The system is not for clinical use. The bound is an average over studies, in distribution, never a statement about one patient.",
  ]),
];

// ---------------------------------------------------------------- chapter 9
const ch9 = [
  H1("9. Engineering Lessons"),
  P("Several traps cost real GPU hours, wrong numbers or silent bugs. They are recorded here because each one passed a fully green test suite."),
  ...table("Engineering traps found during the project.", ["Trap", "Consequence", "Rule adopted"], [
    ["A calibration reporting mask selected voxels using the ground-truth label", "41–57% of reported calibration error was an artefact; a first \"fix\" changed no call site", "A reporting mask never touches the label; verify fixes by re-running the real analysis"],
    ["An fp32-sized clamp (1 − 10⁻⁶) rounds to exactly 1.0 in fp16", "10.5 GPU-hours of training on NaN with nothing raising", "Compute entropy from logits; probes must reach the failure condition"],
    ["Brain-mask Dice scores higher on a left–right mirrored atlas", "The widest-margin alignment check was blind to the worst error", "Verify laterality from labelled structure pairs"],
    ["Error shares computed from rates, not voxel counts", "Headline \"92% of error within 2 mm\" was really 74%", "Weight by voxel count"],
    ["Evaluation mode switch disabled MC dropout recursively", "A silently deterministic \"uncertainty\" map", "Re-enable only dropout layers explicitly, and test it"],
    ["Batching sliding windows on CPU", "12.4 GiB vs 5.4 GiB peak memory and 57% slower", "sw_batch_size = 1 on the laptop"],
    ["Short tokens matched against paths (\"val\" in \"eval\", \"_t1\" in \"_t1ce\")", "Mislabelled splits and sequences", "Match whole tokens"],
    ["Checkpoints written to temporary storage", "Two checkpoints lost forever, so those runs can never be re-scored", "Treat checkpoints as the only irreplaceable artifact"],
  ], [3.2, 3.0, 2.8]),
];

// ---------------------------------------------------------------- chapter 10
const ch10 = [
  H1("10. Conclusion and Future Work"),
  H2("10.1 Conclusion"),
  P("NeuroVision-X delivers a complete, tested brain tumour segmentation system, from a DICOM study to a gated, explained result in a web application, and a measured account of how far its reliability machinery can be trusted. Its dual-encoder model improves enhancing-tumour Dice over a matched U-Net by +0.0267, mostly through gated cross-attention fusion, but nnU-Net beats it and its founding hypothesis returned a null. A conformal bound on the mask's miss rate holds in distribution and breaks under shift in proportion to the shift; the refusal gate cannot see cohort-level shift; and a new site needs roughly 10 to 40 labelled cases to restore the bound, or a fine-tuned model where the failure is too large for any threshold. The project's lasting contribution is less a model than a method: pre-registered, honestly reported measurement of where a medical-imaging AI system holds and where it breaks."),
  H2("10.2 Future work"),
  B("Evaluate the pooled multi-cohort training run (BraTS + BraTS-Africa + paediatric, completed in October 2026) under its pre-registration."),
  B("Repeat the conformal analysis on nnU-Net to test whether the graded shift failure replicates with a stronger model (predictions already produced, interpretation pending)."),
  B("Run at least three seeds per model so that equivalence, not just difference, can be tested."),
  B("Replace the uncertified refusal gate with a certified selective risk-control approach, and test site-conditional recalibration on a real multi-site cohort."),
  B("Decide in January 2027 whether to add IDH molecular subtyping on UCSF-PDGM for the capstone, then write and submit the MELBA journal paper around April 2027."),
];

// ---------------------------------------------------------------- references
const refs = [
  "Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control. International Conference on Learning Representations (ICLR), 2024. arXiv:2208.02814.",
  "Bates, S., Angelopoulos, A., Lei, L., Malik, J., and Jordan, M. Distribution-free, risk-controlling prediction sets. Journal of the ACM, 68(6), 2021. arXiv:2101.02703.",
  "Tibshirani, R. J., Barber, R. F., Candès, E., and Ramdas, A. Conformal prediction under covariate shift. Advances in Neural Information Processing Systems (NeurIPS), 2019. arXiv:1904.06019.",
  "Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World. Springer, 2005.",
  "Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18, 203–211, 2021.",
  "Isensee, F., et al. nnU-Net Revisited: a call for rigorous validation in 3D medical image segmentation. Medical Image Computing and Computer Assisted Intervention (MICCAI), 2024.",
  "Baid, U., et al. The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv:2107.02314, 2021.",
  "Menze, B. H., et al. The multimodal brain tumor image segmentation benchmark (BRATS). IEEE Transactions on Medical Imaging, 34(10), 1993–2024, 2015.",
  "Adewole, M., et al. The Brain Tumor Segmentation (BraTS) challenge 2023: glioma segmentation in sub-Saharan Africa patient population (BraTS-Africa). arXiv:2305.19369, 2023.",
  "Kazerooni, A. F., et al. The Brain Tumor Segmentation (BraTS) challenge 2023: focus on pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs). arXiv:2305.17033, 2023.",
  "Ronneberger, O., Fischer, P., and Brox, T. U-Net: convolutional networks for biomedical image segmentation. MICCAI, 2015.",
  "Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O. 3D U-Net: learning dense volumetric segmentation from sparse annotation. MICCAI, 2016.",
  "Liu, Z., et al. Swin Transformer: hierarchical vision transformer using shifted windows. IEEE/CVF International Conference on Computer Vision (ICCV), 2021.",
  "Hatamizadeh, A., et al. Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images. BrainLes workshop, MICCAI, 2022.",
  "Gal, Y., and Ghahramani, Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. International Conference on Machine Learning (ICML), 2016.",
  "Shahid. When average calibration fails: site-conditional federated conformal risk control. arXiv:2606.20115, 2026.",
  "Adhikary, Chabi, and Mastmeyer. Bound-aware per-organ recall risk control for multi-organ CT segmentation under clinical domain shift. arXiv:2608.18193, 2026.",
  "Liu, Qiao, Zhang, and Chen. Certify or refuse: a cross-model map for selective risk control with coverage floors under covariate shift. arXiv:2608.10893, 2026.",
  "Rohlfing, T., Zahr, N. M., Sullivan, E. V., and Pfefferbaum, A. The SRI24 multichannel atlas of normal adult human brain structure. Human Brain Mapping, 31(5), 798–819, 2010.",
  "Isensee, F., et al. Automated brain extraction of multisequence MRI using artificial neural networks (HD-BET). Human Brain Mapping, 40(17), 4952–4964, 2019.",
  "Li, X., Morgan, P. S., Ashburner, J., Smith, J., and Rorden, C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion (dcm2niix). Journal of Neuroscience Methods, 264, 47–56, 2016.",
  "Cardoso, M. J., et al. MONAI: an open-source framework for deep learning in healthcare. arXiv:2211.02701, 2022.",
];
const refSection = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("References")] }),
  ...refs.map((r, i) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 567, hanging: 567 },
    spacing: { line: 276, after: 100 }, children: [new TextRun(`[${i + 1}]\t${r}`)],
    tabStops: [{ type: TabStopType.LEFT, position: 567 }] })),
];

// ---------------------------------------------------------------- appendix
const appendix = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Appendix A. Training runs and reproducibility")] }),
  P("All training ran on Kaggle Tesla T4 GPUs, each run pinned to a git commit. GPU-hours are session wall-clock."),
  ...table("Main training runs.", ["Run", "GPU-h", "Outcome"], [
    ["Matched U-Net, 80 epochs, 64³", "~3.2", "Test ET 0.8442, the comparison row"],
    ["NeuroVision-X (deployed), 3 sessions", "23.1", "Test ET 0.8709, best epoch 69"],
    ["Capacity control U-Net (two draws)", "~8 + 5.8", "Architecture share 0.79 / 0.80"],
    ["Content-only gate ablation", "24.2", "Within noise of the full model (P2 null)"],
    ["Heavy augmentation", "23.7", "12 of 12 inconclusive"],
    ["NeuroVision-X seed 43", "~23", "Noise floor +0.0021 ET"],
    ["U-Net seed 43", "~5", "Headline replicated (+0.0247)"],
    ["nnU-Net v2, 1000 epochs, 7 sessions", "~59", "Test ET 0.8879; architecture claim withdrawn"],
    ["Cross-fitted fine-tunes (4 folds)", "~5", "Tumour core recovered on SSA and PED"],
    ["Flip TTA measurement", "~3", "ET +0.0007, inconclusive"],
    ["Pooled multi-cohort training", "~25", "Completed; evaluation pending"],
  ], [4, 1.2, 4]),
  P("The repository records every result in docs/research_docs/experiments.md (numbered notes), every allowed claim in docs/research_docs/claims_and_evidence.md, the pre-registrations in docs/research_docs/preregistrations/, and the exact rebuild commands in docs/research_docs/reproducibility.md. Saved float16 logits for every evaluated split allow all metrics, conformal analyses and recalibrations to be recomputed without new inference."),
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("Appendix B. Example segmentations")] }),
  ...figure(path.join(ROOT, "outputs/paper/figures/qualitative_panel.png"), "Example BraTS cases (worst, median and best by Dice for a baseline 3D U-Net): FLAIR, ground truth and prediction, with the three label classes NCR/NET, ED and ET.", 5.4),
];

// ---------------------------------------------------------------- document
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] });

const doc = new Document({
  title: "NeuroVision-X Final Project Report",
  creator: "Amish Yadav",
  description: "Final project report for NeuroVision-X",
  pageNumbers: estimatePageNumbersWith({ guess: true }),
  styles: {
    default: { document: { run: { font: FONT, size: 24, color: "000000" },
      paragraph: { spacing: { line: 360, after: 120 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 240, line: 276 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 240, after: 120, line: 276 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: "000000" },
        paragraph: { spacing: { before: 200, after: 100, line: 276 }, outlineLevel: 2, keepNext: true } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 20, color: "000000" },
        paragraph: { spacing: { before: 60, after: 200, line: 260 } } },
      { id: "FrontTitle", name: "Front Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 0, after: 240 } } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 120, after: 40, line: 276 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", run: { font: FONT, size: 24 }, paragraph: { indent: { left: 440 }, spacing: { before: 0, after: 40, line: 276 } } },
      { id: "TableofFigures", name: "table of figures", basedOn: "Normal", run: { font: FONT, size: 24 }, paragraph: { spacing: { before: 0, after: 60, line: 276 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
    ] },
    ...numberingConfigs,
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      children: titlePage },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        pageNumbers: { start: 1, formatType: "lowerRoman" } } },
      footers: { default: footer },
      children: [...abstract, ...frontLists] },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        pageNumbers: { start: 1, formatType: "decimal" } } },
      footers: { default: footer },
      children: [...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...ch7, ...ch8, ...ch9, ...ch10, ...refSection, ...appendix] },
  ],
});

// numbering configs are collected while building the content, so they exist before the Document is constructed
Packer.toBuffer(doc).then((buf) => {
  const out = process.argv[2] || path.join(__dirname, "NeuroVision-X_Final_Project_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
