// Model 3 submission: final model, robustness, self-assessment and guarded deployment.
const L = require("./lib.js");
const { P, H1, H2, H3, B, NL, figure, table, REF, QUOTE, fig, sem, path, ROOT } = L;

const abstract = [
  P("This is the third of three submissions describing the deep-learning model behind NeuroVision-X. It covers **Model 3**, developed from 23 August to October 2026: the final model. Model 2 had shown that the dual-encoder network is more accurate than a matched U-Net in distribution, but not more reliable, and that its gain disappears on data from other populations. Model 3 keeps that network as the segmentation engine and builds around it the parts that let the system judge its own output, then measures exactly where those parts work and where they fail."),
  P("First, the Model 2 result was stress-tested. It replicates at a second seed pair (+0.0247), is about thirteen times the seed-to-seed noise floor, is unchanged by flip test-time augmentation, and is about 80% architectural over two capacity-control draws; under lesion-wise scoring it is larger (+0.0508) and comes from fewer spurious lesions. Heavier augmentation does not close the out-of-distribution gap. A properly configured nnU-Net, however, beats the model (ET Dice 0.8879 against 0.8709), so the architecture claim was withdrawn."),
  P("Second, three self-assessment mechanisms were added and tested: the trained confidence head (beaten by free entropy), a separate 3D quality-control network that predicts a mask's Dice without a label (better than entropy in one of five pre-registered cells, worse in two, and more optimistic under shift in all six), and **conformal risk control**, which turns the network's probabilities into a mask with a distribution-free bound on the fraction of tumour missed. The bound holds in distribution in all six cells, and breaks under shift by an amount ordered by the size of the shift, up to 11.5 times nominal on paediatric tumour core. About 10 to 40 local labelled cases restore it at a new site, except for paediatric tumour core, which cross-fitted fine-tuning recovers (tumour-core Dice 0.439 to 0.807)."),
  P("Finally, the model was deployed in a DICOM-in clinical pipeline with a refusal gate that answers PROCEED, CAUTION or REFUSE. End to end it returns a usable mask for 85%, 78% and 24% of in-distribution, African and paediatric studies, with silent failure of 4.2%, 18.3% and 49.5%; the gate cannot see cohort-level shift. These measurements are the project's final result."),
  P("**Keywords:** conformal risk control, quality control, refusal gate, distribution shift, lesion-wise Dice, nnU-Net, fine-tuning, clinical pipeline."),
];

const abbrevs = [
  ["BraTS", "Brain Tumor Segmentation challenge and dataset"],
  ["SSA / PED", "BraTS-Africa (sub-Saharan Africa) / paediatric BraTS cohorts"],
  ["ET / TC / WT", "Enhancing tumour / tumour core / whole tumour"],
  ["CRC", "Conformal risk control"],
  ["QC", "Quality control"],
  ["SegQC", "The project's segmentation quality-control network"],
  ["OOD", "Out of distribution"],
  ["TTA", "Test-time augmentation"],
  ["AUROC", "Area under the receiver operating characteristic curve"],
  ["α", "Target (nominal) risk level of the conformal bound"],
  ["DICOM, DICOM-SEG", "Clinical image standard; its segmentation object"],
  ["SRI24", "SRI24 multichannel brain atlas"],
  ["GPU-h", "GPU hours on a Kaggle Tesla T4"],
];

const body = [
  // 1
  H1("1. Introduction"),
  H2("1.1 Where Model 2 left off"),
  P("Model 2, the NeuroVision-X dual encoder, beat the matched Model 1 U-Net on enhancing tumour by +0.0267, mostly through gated cross-attention fusion. But it was not better calibrated, its uncertainty did not rank its failures better, its founding hypothesis was a null, and on BraTS-Africa and paediatric data the gain vanished and tumour core got worse. Neither model could tell when it was failing."),
  H2("1.2 The question for Model 3"),
  P("A segmentation model will always meet scans unlike its training data. The useful question is therefore not only *how accurate* the model is, but whether the system around it can **bound its error** and **refuse studies it should not answer**, and where that machinery itself breaks. The project's final thesis, adopted on 26 September 2026, states this:"),
  QUOTE("*A tumour segmentation model wrapped in a distribution-free error bound and a refusal gate: the bound holds in distribution, fails under distribution shift in proportion to the shift, and the gate cannot see cohort-level shift. We measure where it breaks, and what a new site needs in order to restore it.*"),
  H2("1.3 Objectives of Model 3"),
  ...NL([
    "Stress-test the Model 2 result: seed noise, replication, test-time augmentation, capacity, heavier augmentation, lesion-wise scoring, and a full-recipe nnU-Net.",
    "Give the system ways to judge its own output: the confidence head, a separate quality-control network, and conformal risk control.",
    "Measure where the error bound holds and where it breaks under distribution shift.",
    "Find what a new site needs to restore it: local recalibration, fine-tuning, or pooled training.",
    "Deploy the model in a DICOM-in pipeline with a refusal gate, and measure its end-to-end error budget, including on real clinical DICOM.",
  ]),
  ...figure(fig("fig_evolution"), "The three stages of model development. Model 3 keeps the Model 2 network and adds robustness checks, self-assessment and a guarded deployment."),
  H2("1.4 Method of working"),
  P("Every experiment that could change a claim was **pre-registered**: the hypothesis, endpoints, decision rule and GPU abort budget were committed to version control before the first number existed. All comparisons are paired over the same cases, with 95% paired-bootstrap confidence intervals and Holm-corrected Wilcoxon tests; a difference that does not survive correction is reported as *inconclusive*, never as *no difference*."),

  // 2
  H1("2. Stress-Testing the Model 2 Result"),
  H2("2.1 Seed noise and replication"),
  P("A single training run can be lucky. Model 2 was retrained with a second seed (43), and so was the U-Net (Figure 2, Table 1)."),
  ...table("Robustness of the enhancing-tumour gain on the test split (n = 189).", ["Comparison", "Change in ET Dice", "95% CI", "Meaning"], [
    ["Model 2 − U-Net, seed 42 (headline)", "+0.0267", "0.0143 to 0.0432", "The Model 2 result"],
    ["Model 2 − U-Net, seed 43", "+0.0247", "0.0116 to 0.0420", "Replicates"],
    ["Model 2 seed 43 − seed 42", "+0.0021", "−0.0018 to +0.0068", "Seed noise, model"],
    ["U-Net seed 43 − seed 42", "+0.0041", "0.0016 to 0.0072", "Seed noise, baseline"],
    ["Flip TTA (8 flips) − none", "+0.0007", "−0.0004 to +0.0017", "Inference setting"],
  ], [3.5, 1.6, 2, 2]),
  ...figure(fig("fig_m3_noise"), "The headline gain against seed noise and test-time augmentation."),
  P("The gain replicates and is about **thirteen times** the model's own seed noise. Flip test-time augmentation, averaging predictions over the eight mirror images of the input, does not measurably change ET Dice (it gives a tiny significant +0.0017 on whole tumour), so the gain is not an artefact of inference settings."),
  H2("2.2 Capacity, a second time"),
  P("Model 2's capacity control checkpoint had been lost on Kaggle. It was retrained under a new pre-registration. The second draw reproduces the first almost exactly (width alone +0.0053 against +0.0055), giving an architecture share of **0.80** against 0.79. Under lesion-wise scoring the architecture share is 0.64."),
  H2("2.3 Heavier augmentation"),
  P("If the model fails on other populations because their images look different, stronger simulated variation during training might help. Model 2 was retrained (23.7 GPU-h) with small-angle rotation, gamma, simulated bias field and elastic deformation added. **All 12 pre-registered comparisons were inconclusive**; the primary pooled external tumour-core change was +0.0117 (CI −0.0006 to +0.0245), the same size as a seed-to-seed difference measured on the same cases (+0.0111). The recipe was left unchanged."),
  H2("2.4 Lesion-wise scoring"),
  P("Since 2023 BraTS has scored segmentations **lesion by lesion**: each connected tumour is matched to a reference lesion, and unmatched predicted lesions count as zero. Re-scoring the saved logits this way changed the picture in two ways:"),
  B("The in-distribution gain over the U-Net is **larger**: ET +0.0508 (CI 0.0225 to 0.0798) and TC +0.0371. Its mechanism is **fewer spurious lesions** (0.32 against 0.47 false-positive ET lesions per case) rather than better recall, matching the far-field false-positive problem first seen in Model 1."),
  B("Voxel Dice **overstates** absolute performance by 0.10 to 0.32 for every model. Model 2's whole tumour reads 0.9321 voxel-wise but 0.7183 lesion-wise, because one large lesion dominates the voxel count while small missed or spurious lesions count fully lesion-wise."),
  H2("2.5 The strong baseline: nnU-Net"),
  P("A gain over a matched U-Net does not show that an architecture is good: a properly configured self-adapting U-Net may do better. **nnU-Net v2** (`3d_fullres`, default plans, its full 1000-epoch recipe) was trained on the same 875 cases, about 59 GPU-hours over seven Kaggle sessions."),
  ...table("Test results against nnU-Net v2 (n = 189).", ["Metric", "Matched U-Net", "NeuroVision-X", "nnU-Net v2"], [
    ["Dice ET", "0.8442", "0.8709", "0.8879"],
    ["Dice TC", "0.9058", "0.9161", "0.9247"],
    ["Dice WT", "0.9276", "0.9321", "0.9424"],
    ["Lesion-wise Dice ET", "—", "0.7550", "0.7859"],
    ["Training GPU-h", "3.2", "23.1", "about 59"],
  ], [3, 2, 2, 2]),
  ...figure(sem("fig_dice_vs_baselines"), "Mean test Dice per region for the matched U-Net, NeuroVision-X and nnU-Net v2."),
  P("nnU-Net beats NeuroVision-X on both pre-registered endpoints: voxel ET −0.0170 (CI −0.0309 to −0.0069, p_holm 1.0 × 10⁻⁷) and lesion-wise ET −0.0309. It is also +0.0437 above the matched U-Net, which shows the weak link was the project's short recipe (80 epochs, 64³), not the U-Net topology. By the rule fixed in advance, **the architecture claim was withdrawn**: the +0.0267 gain is real, but only against a matched-recipe U-Net. Figure 4 places all the steps on one scale."),
  ...figure(fig("fig_decomposition"), "Decomposition of the enhancing-tumour gain, with nnU-Net for reference.", 5.6),
  P("The deployed model was nevertheless kept as NeuroVision-X (author decision, 13 October), because the project's thesis is about the wrapper, which works with any model. A follow-up run tests whether the conformal bound's behaviour under shift replicates on nnU-Net; it has completed and its analysis is pending at the time of writing."),

  // 3
  H1("3. Self-Assessment Mechanisms"),
  H2("3.1 The confidence head"),
  P("Model 2's confidence head was trained to predict, voxel by voxel, whether the segmentation is correct. Scored against free single-pass predictive entropy, it **loses on every region** (AUROC ET 0.855 against 0.912, TC 0.871 against 0.908; on whole tumour its output is nearly constant and uninformative). Together with Model 2's results this makes single-pass entropy undefeated in the project: MC dropout is equivalent to it, branch disagreement is worse, and the trained head is worse."),
  H2("3.2 SegQC: a quality-control network"),
  P("Per-voxel uncertainty does not directly say whether a *whole* mask is usable. A separate network, **SegQC**, was trained to predict a mask's Dice from the image and the mask alone, so it can run on a new patient without a label (Figure 5)."),
  ...figure(fig("fig_segqc"), "The SegQC quality-control network."),
  B("**Input:** three channels at 64³: the T1CE image, the predicted region mask and the entropy map, cropped around the prediction."),
  B("**Network:** four 3D convolution stages (16, 32, 64, 128 channels, group normalisation, LeakyReLU), global average pooling and a small MLP that outputs the predicted Dice. One model serves all three regions."),
  B("**Training data:** real model predictions on the BraTS validation split, plus copies degraded in six ways (identity, erosion, dilation, dropping a component, shifting, speckle), each labelled with its true Dice. This teaches the network what bad masks look like without needing bad real cases."),
  P("It was tested on the external cohorts in a pre-registered five-cell family against free entropy (Gate C, Figure 6)."),
  ...figure(fig("fig_m3_qc"), "SegQC against free entropy at flagging bad segmentations, five pre-registered cells."),
  P("The verdict is **mixed**. SegQC wins clearly on paediatric tumour core (ΔAUROC +0.1686, p_holm 0.006), loses significantly on paediatric whole tumour (−0.1898), and loses on African tumour core at the edge of significance (p_holm 0.0504). More important, its error becomes **more optimistic under shift in all six external cells** (largest on paediatric tumour core, +0.298): a quality check that grows more confident exactly where the segmentation is least trustworthy. It is used in the gate, with that limitation stated."),
  H2("3.3 An input out-of-distribution score"),
  P("A pre-registered score built from input-image statistics flags 60% of African studies against 9.5% of test studies, so it does see the shifted cohort. But it ranks failed masks barely above chance (AUROC 0.56 to 0.60), and as a refusal rule it would remove 3 of 11 African silent failures (4 required) at the cost of 12 usable studies. It failed its switch-on rule and is **display-only**."),

  // 4
  H1("4. Conformal Risk Control"),
  H2("4.1 Method"),
  P("Conformal risk control converts the network's voxel probabilities into a mask with a guarantee (Figure 7). The risk is the **fraction of true tumour voxels the mask misses**. On a calibration set (BraTS validation, 187 cases) a threshold λ is chosen as the largest value whose adjusted miss rate, (n·R(λ) + 1)/(n + 1), stays at or below the target α. Frozen and applied to new studies, it guarantees that the *expected* miss rate is at most α, provided the new studies are **exchangeable** with the calibration set, roughly, drawn from the same distribution. The guarantee is a mathematical theorem that holds for any model; it is an average over studies, never a promise about one patient. It was applied to whole tumour and tumour core at α = 0.05, 0.10 and 0.20."),
  ...figure(fig("fig_conformal"), "Conformal risk control applied to the mask's miss rate."),
  H2("4.2 In distribution"),
  P("On the test split the bound **holds in all six cells**, with realised risk at 0.64 to 0.96 times nominal, for both NeuroVision-X and the U-Net. Its cost is small: at α = 0.05 the conservative mask is 1.10 times (WT) and 1.18 times (TC) the size of the default mask, and at looser α it is actually smaller than the default."),
  H2("4.3 Under distribution shift"),
  P("Applied frozen to the external cohorts, the bound **fails, by an amount ordered by how far the cohort is from the training data** (Figure 8): BraTS-Africa whole tumour about 1.1 times nominal, paediatric whole tumour 1.4 to 1.9 times, paediatric tumour core 3.5 to 11.5 times. Seven of twelve shifted cells are violated, for each model independently. Part of the paediatric tumour-core failure is a label convention rather than imaging: the paediatric reference counts most non-enhancing tumour as core (median 84% of the core), which the adult convention calls oedema."),
  ...figure(sem("fig_realised_vs_nominal"), "Realised against nominal risk per cohort. Points above the diagonal violate the bound."),

  // 5
  H1("5. Adapting to a New Site"),
  H2("5.1 Local recalibration"),
  P("If the bound breaks because a new site is not exchangeable with BraTS, the obvious repair is to recalibrate on the site's own labelled cases. This was simulated as a pre-registered counterfactual: each external cohort was split 1,000 times, the threshold refitted on k local cases (k = 5 to 49), and the miss rate measured on the rest."),
  ...figure(sem("fig_local_recalibration"), "Held-out miss rate against the number of local calibration cases k; crosses mark cells that do not meet the target."),
  P("About **10 to 40 local labelled cases**, depending on α, restore the bound for whole tumour on both cohorts and for tumour core on BraTS-Africa. **Paediatric tumour core cannot be restored at any k**: even at the lowest possible threshold the model misses 35.6% of it, above every α tested. This was derived and written down before the run."),
  H2("5.2 Fine-tuning"),
  P("Where recalibration cannot help, the model itself must change. The deployed model was fine-tuned on half of each external cohort and scored on the other half, then the halves were swapped (two-fold cross-fitting), so every case was scored by a model that never saw it (four fine-tunes, about 5 GPU-h)."),
  ...figure(fig("fig_m3_finetune"), "Tumour-core Dice before and after cross-fitted fine-tuning."),
  P("Fine-tuning **recovers tumour core on both cohorts**: BraTS-Africa +0.0433 (0.785 to 0.828) and paediatric +0.3676 (0.439 to 0.807). After fine-tuning, paediatric tumour core becomes restorable by local recalibration at α = 0.10 and 0.20. Most of the paediatric gain is the model learning the paediatric label convention (its predicted core-to-whole-tumour ratio moves from 0.27 to 0.95). These are cross-fitted results, not external validation, and the deployed model is unchanged."),
  H2("5.3 Pooled training"),
  P("The developer-side alternative is to include the shifted populations in training from the start. A pooled model (same recipe, trained from scratch on the 875 BraTS cases plus 25 African and 45 paediatric cases, held-out halves identical to the fine-tuning fold) was pre-registered and trained on Kaggle over three resumed sessions, finishing all 80 epochs (validation mean Dice 0.8855). Its evaluation is pending at the time of writing, so no result from it is reported here."),

  // 6
  H1("6. The Deployed System"),
  H2("6.1 Clinical pipeline"),
  P("The final model runs inside a pipeline that accepts a zipped DICOM study through the web page at /clinical (Figure 11). **Ingest** converts each series to NIfTI, assigns the four sequence roles from a rule table and reads the patient's age. **Input QC** (twelve label-free checks on intensity, geometry, slice thickness and registration) runs before and after the study is co-registered to its T1CE series, registered to the SRI24 atlas BraTS uses, and skull-stripped with HD-BET, so the study lands in the space the model was trained in. The model segments it; SegQC and the conformal band are computed; the **gatekeeper** decides."),
  ...figure(fig("fig_pipeline"), "The clinical pipeline from a DICOM study to a gated result."),
  H2("6.2 The refusal gate"),
  P("The gate combines label-free input QC, SegQC's predicted Dice and an **intended-use rule** (adults only, from the DICOM PatientAge field) into PROCEED, PROCEED WITH CAUTION or REFUSE. The conformal band and the OOD score are shown but do not decide. All thresholds were fitted on validation only. A refused study is treated as a successful outcome, never an error. Outputs include a 3D digital twin of the brain with tumour and atlas-structure shells (Figure 12), a structured anatomical report, a DICOM-SEG object for standard viewers and a downloadable bundle."),
  ...figure(path.join(ROOT, "docs/archive/figures/twin_clinical_9c2cc294_front_separated.png"), "The 3D digital twin of a real DICOM study that the gate passed (PROCEED)."),
  H2("6.3 End-to-end error budget"),
  P("Measured through every stage with the frozen gate, counting a study as **usable** when both WT and TC Dice are at least 0.7 and as a **silent failure** when the gate accepts an unusable mask (Figure 13):"),
  ...table("End-to-end outcome per cohort with the deployed, frozen gate.", ["Cohort", "n", "Usable mask returned", "Silent failure"], [
    ["BraTS test (in distribution)", "189", "85.2% [79.9, 90.0]", "4.2%"],
    ["BraTS-Africa (SSA)", "60", "78.3% [66.7, 88.3]", "18.3%"],
    ["Paediatric BraTS (PED)", "99", "24.2% [16.2, 33.3]", "49.5%"],
  ], [3.4, 0.8, 2.6, 1.8]),
  ...figure(sem("fig_error_budget"), "End-to-end outcome of each study with the frozen refusal gate, per cohort."),
  P("The gate has decent precision and poor recall, and it is **blind to cohort-level shift**: on BraTS-Africa it refuses 2 of 60 studies while passing 11 of 12 unusable masks; on paediatric data 23 of 26 refusals are correct, but it catches only 32% of unusable masks. Accepting a study does not repair a broken bound (paediatric tumour-core risk 0.573 on accepted cases at nominal 0.10). The adults-only rule removes every paediatric study by construction, which is **scoping, not detection**."),
  H2("6.4 Real clinical DICOM"),
  P("On original DICOM of 40 test-split patients (from the RSNA-MICCAI collection), the front end refuses 40% of studies for thick slices. On the rest, whole-tumour Dice falls by a median 0.23 against the research path, mostly a median 5.9 mm disagreement between the pipeline's rigid atlas registration and BraTS's own. The gate accepts 22 of 24 remaining studies, and 14 of those are unusable (silent failure 0.35). This is a measurement of the front end, not external validation of the model."),

  // 7
  H1("7. Final Model Summary"),
  ...table("The final deployed model and its wrapper.", ["Item", "Value"], [
    ["Segmentation network", "NeuroVision-X dual encoder (Model 2), seed 42, epoch 69"],
    ["Parameters / compute", "34.9 M; 267 GFLOPs per 64³ patch"],
    ["Training", "BraTS 2021, 875 cases, 64³ patches, 80 epochs, 23.1 T4-hours"],
    ["Inference", "Sliding window 64³, overlap 0.5, Gaussian blending, re-nesting, components < 50 voxels removed"],
    ["Test Dice (voxel)", "ET 0.8709, TC 0.9161, WT 0.9321"],
    ["Test Dice (lesion-wise)", "ET 0.7550; WT 0.7183"],
    ["Error bound", "Conformal risk control on miss rate, α 0.05 / 0.10 / 0.20, calibrated on BraTS validation"],
    ["Self-assessment", "SegQC predicted Dice; input QC; entropy map"],
    ["Gate", "PROCEED / CAUTION / REFUSE from input QC, SegQC and intended use (adults)"],
    ["Intended use", "Research only, adult glioma; not a medical device"],
  ], [2.8, 6.2]),
  ...table("What each stage contributed.", ["Stage", "Main result"], [
    ["Model 1", "Pipeline, resumable trainer, U-Net reference (ET 0.8442)"],
    ["Model 2", "Dual encoder, ET +0.0267 via gated fusion; founding hypothesis null; gain does not transfer"],
    ["Model 3", "Gain robust but beaten by nnU-Net; bound holds in distribution, breaks with shift; 10–40 local cases or fine-tuning restore it; gate blind to cohort shift"],
  ], [1.6, 7.4]),

  // 8
  H1("8. Limitations"),
  B("The test set is a random split of BraTS training data; external cohorts are small (60 and 99 cases) and the real-DICOM set is 40 patients of the same test split."),
  B("Local recalibration and fine-tuning are counterfactual and cross-fitted analyses on public data, not prospective site studies."),
  B("The conformal guarantee is an average over studies in distribution, never a statement about an individual patient."),
  B("The training recipe is shorter than state of the art; nnU-Net outperforms the deployed network."),
  B("The pooled-training run and the nnU-Net conformal run are complete but not yet analysed."),
  B("No clinician reviewed the outputs; the system is research software and not for clinical use."),

  // 9
  H1("9. Conclusion and Future Work"),
  P("Model 3 turned an accurate but unguarded network into a measured system. The Model 2 gain survived every robustness check but lost to nnU-Net, so it is stated only against a matched-recipe U-Net. Of the self-assessment mechanisms, free entropy remained the best voxel-level signal and SegQC helps in one setting while growing optimistic under shift. The conformal bound is the strongest piece: it holds in distribution, and its failure under shift is predictable and ordered by the size of the shift. A new site can restore it with 10–40 labelled cases, and where it cannot, fine-tuning can. The refusal gate catches some bad studies but cannot see when a whole cohort is different, so it must be paired with local calibration data."),
  P("Future work: analyse the pooled model and the nnU-Net conformal run; move the deployed wrapper onto nnU-Net; validate on a genuinely external, prospectively collected site; add a cohort-level shift detector; and submit the measured results to MELBA."),

  H1("References"),
  REF("Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L. and Schuster, T. (2024). Conformal risk control. *ICLR*."),
  REF("Bates, S., Angelopoulos, A., Lei, L., Malik, J. and Jordan, M. (2021). Distribution-free, risk-controlling prediction sets. *Journal of the ACM*, 68(6), 1–34."),
  REF("Isensee, F. et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203–211."),
  REF("Isensee, F. et al. (2024). nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation. *MICCAI*."),
  REF("Baid, U. et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. *arXiv:2107.02314*."),
  REF("Adewole, M. et al. (2023). The BraTS challenge 2023: Glioma segmentation in sub-Saharan Africa patient population (BraTS-Africa). *arXiv:2305.19369*."),
  REF("Kazerooni, A. F. et al. (2023). The BraTS challenge 2023: Focus on pediatrics (BraTS-PEDs). *arXiv:2305.17033*."),
  REF("Robinson, R. et al. (2019). Automated quality control in image segmentation: application to the UK Biobank cardiovascular magnetic resonance imaging study. *Journal of Cardiovascular Magnetic Resonance*, 21, 18."),
  REF("Isensee, F. et al. (2019). Automated brain extraction of multisequence MRI using artificial neural networks (HD-BET). *Human Brain Mapping*, 40(17), 4952–4964."),
  REF("Rohlfing, T., Zahr, N. M., Sullivan, E. V. and Pfefferbaum, A. (2010). The SRI24 multichannel atlas of normal adult human brain structure. *Human Brain Mapping*, 31(5), 798–819."),
];

const meta = { kicker: "MODEL 3 SUBMISSION", period: "Development period: 23 August – October 2026",
  subtitle: "Final Model: Robustness, Self-Assessment, a Distribution-Free Error Bound and a Guarded Clinical Deployment", date: "November 2026" };
const doc = L.makeDoc({ title: "NeuroVision-X Model 3: Final Model", titlePage: L.titlePage(meta),
  front: L.frontMatter(abstract, abbrevs), body });
L.write(doc, process.argv[2] || path.join(__dirname, "model3.docx"));
