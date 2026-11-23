// Model 2 submission: the NeuroVision-X dual-encoder network.
const L = require("./lib.js");
const { P, H1, H2, H3, B, NL, figure, table, REF, QUOTE, fig, path } = L;

const abstract = [
  P("This is the second of three submissions describing the deep-learning model behind NeuroVision-X. It covers **Model 2**, developed between 4 and 23 August 2026: the proposed dual-encoder segmentation network, its training, and a controlled set of experiments to find out what it actually improves and why."),
  P("Model 2 combines a 3D convolutional encoder, which captures local texture, with a Swin Transformer encoder, which captures wider context. At four scales the two streams are joined by **adaptive gated cross-attention fusion**: the CNN features attend to the Swin features, and a learned gate decides at each location how much of that information to add. The founding idea, pre-registered before training, was that the gate should also read how much the two branches *disagree* at each voxel. A U-Net decoder and three heads (segmentation, confidence and boundary) complete the model, 34.9 million parameters in total. It was trained with exactly the recipe of Model 1 (64³ patches, 80 epochs) on a Kaggle T4, 23.1 GPU-hours over three resumed sessions, after a first attempt silently trained on NaN for 10.5 GPU-hours because of a float16 bug."),
  P("Against the matched Model 1 U-Net, Model 2 improves enhancing-tumour Dice by **+0.0267** (95% CI 0.0143 to 0.0432, Holm p = 1.4 × 10⁻²¹, n = 189) and tumour-core Dice by +0.0103. A width-matched U-Net shows that about 79% of the gain comes from the architecture rather than the parameter count, and a parameter-matched ablation shows that the gain comes from the gated fusion itself: removing the disagreement inputs changes nothing measurable (+0.0022, not significant). The predicted reliability benefits did not appear: calibration, boundary error share and uncertainty ranking are all within noise of the baseline, and single-pass entropy localises errors as well as 10-pass Monte Carlo dropout while branch disagreement does worse. On two external cohorts (BraTS-Africa and paediatric BraTS) the accuracy gain disappears and tumour-core Dice is significantly worse. Model 2 is more accurate, not more reliable, at 23.7 times the computation. These results define the work of Model 3."),
  P("**Keywords:** dual encoder, Swin Transformer, gated cross-attention fusion, ablation, capacity control, external validation, uncertainty."),
];

const abbrevs = [
  ["BraTS", "Brain Tumor Segmentation challenge and dataset"],
  ["SSA / PED", "BraTS-Africa (sub-Saharan Africa) / paediatric BraTS cohorts"],
  ["ET / TC / WT", "Enhancing tumour / tumour core / whole tumour"],
  ["CNN", "Convolutional neural network"],
  ["Swin", "Shifted-window vision Transformer"],
  ["MC dropout", "Monte Carlo dropout: several stochastic forward passes"],
  ["AUROC", "Area under the receiver operating characteristic curve"],
  ["AURC", "Area under the risk-coverage curve"],
  ["ECE", "Expected calibration error"],
  ["GFLOPs", "Billions of floating-point operations"],
  ["TOST", "Two one-sided tests (equivalence test)"],
  ["n.s.", "Not statistically significant after Holm correction"],
];

const body = [
  // 1
  H1("1. Introduction"),
  H2("1.1 Where Model 1 left off"),
  P("Model 1 produced a 3D U-Net reference (test Dice ET 0.8442, TC 0.9058, WT 0.9276) and two observations: 73.7% of whole-tumour error lies within 2 mm of the boundary, with a second population of spurious lesions far from any tumour, and the network is over-confident. It also fixed the recipe, split, metrics and statistics that any new model must be judged by."),
  H2("1.2 Motivation"),
  P("A convolutional network builds its picture of a voxel from a small neighbourhood that grows slowly with depth. It is excellent at texture and edges but sees wider context only indirectly. A **transformer** relates every region of its input to every other through self-attention, so it captures global context directly, but it is weaker at fine local detail and expensive in 3D. The Swin Transformer makes attention affordable by computing it inside small shifted windows. A model that uses both should be able to combine local precision with global context, which is exactly what a tumour boundary seems to need."),
  H2("1.3 The founding hypothesis"),
  P("Hybrid CNN–transformer models with gated fusion already exist. The idea proposed here was narrower, and it was written down as a pre-registration on 4 August, before any model was trained:"),
  QUOTE("*Prior fusion gates derive their weight from the branch features alone. Our gate also reads an explicit local ambiguity signal: the per-voxel disagreement between the CNN branch's and the Swin branch's own region predictions, and each branch's entropy. It should open where local evidence is ambiguous, and give better calibration and boundary accuracy rather than a uniform Dice gain.*"),
  P("The pre-registration also fixed the test that could refute it: a version of the model whose gate sees only the features, not the disagreement, must lose. If it ties, the null result is reported."),
  H2("1.4 Objectives of Model 2"),
  ...NL([
    "Design and implement the dual-encoder network with gated cross-attention fusion and three output heads, each component with its own CPU shape test.",
    "Train it under exactly the Model 1 recipe within the GPU budget.",
    "Measure it against the matched U-Net, a width-matched capacity control and the content-only gate ablation, with paired statistics.",
    "Test the predicted reliability benefits: calibration, boundary error, and uncertainty.",
    "Test it outside the training distribution on two external cohorts.",
  ]),

  // 2
  H1("2. Architecture"),
  H2("2.1 Overview"),
  P("Figure 1 shows the full network. The input (four MRI sequences, a 64³ patch) goes into two encoders in parallel. At each of four scales a fusion block combines the CNN feature map with the Swin feature map of the same resolution; the fused maps become the skip connections of a U-Net decoder. Three heads read the decoder output."),
  ...figure(fig("fig_architecture"), "The NeuroVision-X network: CNN encoder, Swin Transformer encoder, gated cross-attention fusion at four scales, U-Net decoder and three heads.", 6.3),
  H2("2.2 CNN encoder"),
  P("The CNN branch has five levels with 32, 64, 128, 256 and 320 channels, the same widths as the Model 1 U-Net, each level made of two residual blocks with group normalisation and dropout 0.1. Each residual block's last normalisation is initialised to zero (`zero_init_residual`), so at the start of training every block behaves as an identity and the network trains stably. This branch holds most of the parameters (18.9 M)."),
  H2("2.3 Swin Transformer encoder"),
  P("The Swin branch is the encoder of MONAI's SwinUNETR with feature size 48, two transformer blocks per stage, 3, 6, 12 and 24 attention heads, window size 7 and patch size 2. Only four stages are used (the deepest stage of a full SwinUNETR, three quarters of its parameters, is dropped), so the branch is small, about 2.0 M parameters. Gradient checkpointing is on in this branch to save memory."),
  H2("2.4 Adaptive gated cross-attention fusion"),
  P("Figure 2 shows one fusion block. The Swin feature is first passed through a 1×1×1 convolution to match the CNN width. Then:"),
  ...figure(fig("fig_fusion"), "One adaptive gated fusion block. The grey branch probes are what the content-only ablation removes."),
  ...NL([
    "**Windowed cross-attention.** Queries come from the CNN feature, keys and values from the Swin feature, inside local 4³ windows (4 heads). Each CNN location therefore gathers the context the transformer has found around it.",
    "**Gate.** A small network produces a sigmoid gate g between 0 and 1 at each voxel. Its inputs are the two features and, in the full model, three ambiguity maps from small probe heads on each branch: the disagreement |p_CNN − p_Swin| and each branch's entropy.",
    "**Merge.** The output is F_CNN + layer_scale × g × attention. The `layer_scale` starts at 10⁻⁴, so at initialisation the block passes the CNN feature through unchanged and the model starts as a plain CNN U-Net, learning to add transformer context only where it helps.",
  ]),
  P("The entropy of the probes is computed from logits with a numerically stable softplus form. An earlier version computed it from probabilities in float16 and caused the NaN failure described in Section 3.3."),
  H2("2.5 Decoder and heads"),
  P("The decoder mirrors the CNN encoder with transposed-convolution upsampling and two residual blocks per stage, taking the fused maps as skip connections. Three heads read its output:"),
  B("**Segmentation head:** three sigmoid channels (ET, TC, WT), with deep supervision, i.e. extra auxiliary outputs at lower decoder resolutions that also receive a loss."),
  B("**Confidence head:** predicts, per voxel, whether the segmentation is correct, intended as a learned uncertainty map."),
  B("**Boundary head:** predicts a thin shell around each region's boundary, to push the shared features towards precise edges."),
  H2("2.6 Size"),
  ...table("Parameter breakdown of the NeuroVision-X network.", ["Component", "Parameters"], [
    ["CNN encoder", "18,854,208"],
    ["Swin Transformer encoder", "2,043,330"],
    ["Fusion blocks (4)", "1,049,284"],
    ["Decoder", "12,929,664"],
    ["Heads and branch probes", "34,855"],
    ["**Total (production model)**", "**34,911,341**"],
    ["Model 1 U-Net, for comparison", "12,870,489"],
  ], [5, 4]),

  // 3
  H1("3. Training"),
  H2("3.1 Loss"),
  P("The training loss is a weighted sum of several terms (Table 2). The segmentation term keeps a weight of exactly 1.0, so its gradient is the same as in the baseline, and the auxiliary terms are deliberately small so they cannot quietly take over late in training, when Dice is decided."),
  ...table("Multi-task loss terms.", ["Term", "Weight", "What it trains"], [
    ["Segmentation: Dice + BCE with deep supervision", "1.0", "The three region masks"],
    ["Boundary: Dice on a thin boundary shell", "0.3", "Edge-aware shared features"],
    ["Branch probes: Dice + BCE on each branch's own prediction", "0.1", "Meaningful disagreement maps"],
    ["Confidence: BCE against 'segmentation is correct'", "0.05", "A per-voxel correctness estimate"],
  ], [4.5, 1, 3.5]),
  H2("3.2 The timing probe and the re-plan"),
  P("Before committing GPU time, a short **timing probe** trained the model on 50 cases. Two versions ran out of memory; the third measured **3.6 seconds per step at 96³**, about 0.875 hours per epoch, meaning about 91 GPU-hours for 100 epochs, three times the whole budget. The probe also logged gradient norms and found 66–70% of steps clipped at the then-default threshold of 1.0. Two changes followed, applied to every model in the comparison: the recipe moved to **64³ patches and 80 epochs**, and gradient clipping moved to **5.0**. Clipping matters for fairness here: the full model and its ablation produce different gradient magnitudes, and at a threshold of 1.0 they would have trained at different effective learning rates."),
  H2("3.3 Ten GPU-hours on NaN"),
  P("The first full run looked perfectly healthy by every operational signal and stopped cleanly at epoch 38. But the loss had been `nan` since around epoch 10–19, and the saved best checkpoint was frozen at epoch 9. The cause: the probe entropy was computed from probabilities with a clamp of `1 − 10⁻⁶`, which rounds to exactly 1.0 in float16, so a saturated probe gave 0 × log(0) = NaN. **10.5 GPU-hours were lost.** The fix computes entropy from logits with softplus, which is exact in any precision, and the training loop now checks that the loss is finite."),
  H2("3.4 The real run"),
  P("The corrected run trained in **three resumed Kaggle sessions, 23.1 GPU-hours** in total, pinned to commit `7caacfa`. Peak memory was 6.2 GiB of the T4's 14.6 GiB; the median gradient norm was 0.71, and about 1.4% of steps were clipped, which is exactly the intended behaviour of the new threshold. The best validation checkpoint was at **epoch 69** (validation mean Dice 0.8938), and that checkpoint is the model scored and deployed."),

  // 4
  H1("4. Experimental Design"),
  H2("4.1 The comparison ladder"),
  P("A single comparison against the U-Net cannot say *why* a model is better. Model 2 was therefore compared through a ladder in which each step changes one thing (Table 3). All arms share the split, recipe, seed and evaluation."),
  ...table("The ablation ladder. Each row adds one change to the row above.", ["Arm", "Parameters", "What it isolates", "GPU-h"], [
    ["Matched U-Net (Model 1)", "12.87 M", "Reference", "3.2"],
    ["Capacity control: U-Net widened", "34.83 M", "Effect of parameter count alone", "about 8"],
    ["Content-only gate: full model without the disagreement inputs", "34.90 M", "Effect of gated fusion", "24.2"],
    ["Full NeuroVision-X", "34.91 M", "Effect of disagreement conditioning", "23.1"],
  ], [4.2, 1.5, 3.3, 1]),
  P("The capacity control matches the full model's parameter count to within 0.2%, so any remaining gain cannot be blamed on size. The content-only gate matches the full model to within 0.02% and differs only in what the gate can see, which is the precise test of the founding hypothesis."),
  H2("4.2 External cohorts"),
  P("Two cohorts never used for training test how the model behaves under distribution shift, at increasing distance from the training data:"),
  B("**BraTS-Africa (SSA), 60 cases:** adult glioma from sub-Saharan African centres, with lower image quality and larger tumours (median whole tumour about 1.7 times larger)."),
  B("**Paediatric BraTS (PED), 99 cases:** children's brain tumours, which differ in biology, appearance and in how the tumour core is labelled."),
  P("Preparing them exposed a real hazard: the African data is stored with two image axes reversed relative to BraTS 2021. Because Model 1's pipeline derives orientation from each file's own affine rather than assuming it, the data was handled correctly; this was verified by brain-mask overlap with an atlas, which rose from 0.68 to 0.91 after correct reorientation."),
  H2("4.3 Statistics"),
  P("Every comparison is paired over cases, with 95% paired-bootstrap confidence intervals and Wilcoxon tests corrected by Holm's method. A difference is called better or worse only if it survives correction; otherwise it is reported as inconclusive. Decision rules were written down before the results existed."),

  // 5
  H1("5. Results"),
  H2("5.1 In-distribution accuracy"),
  ...table("Test-set results, Model 2 against the matched Model 1 U-Net (n = 189, paired).", ["Metric", "Model 1 U-Net", "Model 2", "Difference", "Verdict"], [
    ["Dice ET", "0.8442", "0.8709", "+0.0267", "better (p 1.4e-21)"],
    ["Dice TC", "0.9058", "0.9161", "+0.0103", "better"],
    ["Dice WT", "0.9276", "0.9321", "+0.0045", "inconclusive"],
    ["HD95 ET (mm)", "4.91", "4.24", "−0.67", "inconclusive"],
    ["HD95 TC (mm)", "5.24", "4.98", "−0.27", "inconclusive"],
    ["HD95 WT (mm)", "6.72", "7.09", "+0.37", "inconclusive"],
  ], [2.2, 1.8, 1.5, 1.5, 2.3]),
  ...figure(fig("fig_m2_results"), "Mean test Dice per region, Model 1 against Model 2."),
  P("The model is clearly better on enhancing tumour, the smallest and hardest region, and on tumour core. Whole tumour and all boundary distances are within noise. Splitting the enhancing-tumour error by distance from the boundary shows where the gain comes from: false negatives within 2 mm of the boundary fall by 23% (enhancing tumour recovered at the margin), and far-field false positives fall by 17% (fewer spurious specks)."),
  H2("5.2 Where the gain comes from"),
  ...table("The ablation ladder on test enhancing-tumour Dice (n = 189).", ["Arm", "ET Dice", "Step", "Verdict"], [
    ["Matched U-Net", "0.8442", "—", "—"],
    ["Capacity control (34.8 M)", "0.8497", "+0.0055 (width)", "—"],
    ["Content-only gate", "0.8686", "+0.0189 (gated fusion)", "better"],
    ["Full NeuroVision-X", "0.8709", "+0.0022 (disagreement)", "inconclusive (CI −0.0067 to +0.0152)"],
  ], [3, 1.4, 2.4, 2.8]),
  ...figure(fig("fig_m2_ablation"), "The ablation ladder. Almost all of the gain arrives with gated fusion; the disagreement inputs add nothing measurable."),
  P("Two conclusions follow. First, **the gain is mostly architectural**: making the U-Net as large as Model 2 buys only +0.0055, so about 79% of the improvement over the U-Net is due to the architecture, not to parameter count (Model 3 repeats this control and gets 80%). Second, **the founding hypothesis is a null**. The content-only gate matches the full model on every metric; the gain comes from gated cross-attention fusion itself, which is already published. As pre-registered, the null is reported rather than hidden."),
  P("The pre-registration also predicted that the gate would open at ambiguous places such as the tumour margin. Measured on the trained model, the gate is strongly organised by anatomy (at the first level it averages 0.98 inside the tumour against 0.33 outside, with opposite polarity at adjacent scales) but it does not peak at the margin."),
  H2("5.3 Reliability: the predicted benefit did not appear"),
  P("The hypothesis predicted better calibration and boundary accuracy rather than a Dice gain. The opposite happened (Figure 5):"),
  B("**Calibration.** Uncalibrated mean ECE is 0.0446 for Model 2 and 0.0395 for the U-Net; after temperature scaling 0.0135 and 0.0175. Neither difference is conclusive. Both models need about the same temperature."),
  B("**Boundary.** Model 2 makes fewer errors at every distance from the boundary, in proportion. The share of error within 2 mm is essentially unchanged (ET 77.6% against 78.6%). It is \"fewer errors overall\", not \"better at the margin\"."),
  B("**Uncertainty ranking.** With 10-pass Monte Carlo dropout, Model 2's uncertainty captures 37.6% of the achievable gain in ranking its own failures, against 40.6% for the U-Net, within noise."),
  ...figure(fig("fig_m2_reliability"), "Calibration after temperature scaling and MC-dropout risk-coverage. Both differences are inconclusive."),
  P("**Model 2 is more accurate, not more reliable.** The reliability framing of the project was dropped on 16 August as a result."),
  H2("5.4 Uncertainty: disagreement against entropy"),
  P("If the gate does not need disagreement, the disagreement map might still be useful as a free uncertainty signal, available from a single forward pass of a dual-encoder model. This was tested as its own pre-registered question. Disagreement does carry some per-voxel error information that entropy lacks (residualised AUROC 0.578, 0.569 and 0.677 on test, SSA and PED). But as an error detector it is **worse** than both alternatives on both external cohorts (Figure 6), and adding it to entropy does not improve a working detector. A further finding concerns the baseline itself: **single-pass predictive entropy is statistically equivalent to 10-pass MC dropout** for locating errors (equivalence margin 0.03 AUROC, fixed in advance), so the expensive method buys nothing here."),
  ...figure(fig("fig_m2_uncertainty"), "How well three uncertainty maps locate voxel errors on the external cohorts."),
  H2("5.5 Out of distribution"),
  ...table("External cohorts, Model 2 against Model 1 (nothing fitted on these cohorts).", ["Cohort", "Region", "Model 1", "Model 2", "Difference", "Verdict"], [
    ["SSA (n = 60)", "ET", "0.7792", "0.7784", "−0.0008", "inconclusive"],
    ["SSA (n = 60)", "TC", "0.7745", "0.7846", "+0.0101", "inconclusive"],
    ["SSA (n = 60)", "WT", "0.8975", "0.8959", "−0.0016", "inconclusive"],
    ["PED (n = 99)", "ET", "0.5733", "0.5634", "−0.0099", "inconclusive"],
    ["PED (n = 99)", "TC", "0.4990", "0.4394", "−0.0595", "worse"],
    ["Pooled (n = 159)", "TC", "—", "—", "−0.0333", "worse"],
  ], [2, 1, 1.4, 1.4, 1.4, 1.8]),
  ...figure(fig("fig_ood"), "Model 2's Dice falls with distance from the training data: BraTS test, BraTS-Africa, paediatric."),
  P("**The accuracy gain does not transfer.** On BraTS-Africa the enhancing-tumour advantage vanishes; on paediatric data tumour core is significantly worse than the U-Net, and both models fail badly (paediatric ET Dice around 0.56–0.57, with many complete failures). Temperature scaling, which worked in distribution, barely helps out of distribution and on African enhancing tumour makes calibration about four times worse (ECE 0.0247 to 0.0685), because a temperature fitted on one population is wrong for another."),
  H2("5.6 Cost"),
  ...figure(fig("fig_m2_cost"), "Inference cost of Model 2 relative to the Model 1 U-Net, measured on one 64³ patch."),
  P("Model 2 has 2.7 times the parameters but needs **23.7 times the computation** (267.4 against 11.3 GFLOPs per patch, 1,086 against 46 ms on CPU), because windowed cross-attention at high resolution operates on a very large number of tokens. Against an in-distribution gain that does not survive distribution shift, this is an unfavourable trade, and it is stated plainly rather than left for a reader to compute."),

  // 6
  H1("6. Discussion"),
  P("Model 2 answered its research question, though not in the hoped-for direction. Its positive result is real and well-controlled: a +0.0267 enhancing-tumour gain over a matched U-Net, mostly architectural, coming from recovered enhancing tumour at the margin and fewer spurious lesions. Its proposed novelty, disagreement-conditioned gating, is a pre-registered null, and its predicted reliability benefits did not appear."),
  P("The out-of-distribution results reframed the project. A model that is more accurate on data like its training data, but no better, or worse, on data from other populations is not safer to use. Neither model could tell when it was failing on paediatric scans. This shifted the question from *which architecture is best* to *how can any segmentation model know, and say, when its output cannot be trusted*, which is what Model 3 addresses. Model 2 itself is kept as the segmentation network: it is the most accurate model trained in this project under the shared recipe, and it becomes the deployed model of Model 3."),

  // 7
  H1("7. Limitations"),
  B("One training seed per arm. The seed-to-seed noise floor, and a replication at a second seed pair, are measured in Model 3."),
  B("The capacity control's checkpoint was never retrieved from Kaggle, so it could not be evaluated on the external cohorts; Model 3 retrains it."),
  B("The 64³, 80-epoch recipe is shorter than state-of-the-art recipes; a properly configured nnU-Net is benchmarked in Model 3."),
  B("The test set is a random split of BraTS training data. External cohorts are small (60 and 99 cases)."),
  B("All Dice values here are voxel-wise; Model 3 adds lesion-wise scoring, which changes the picture."),

  // 8
  H1("8. Conclusion and Next Step"),
  P("Model 2, the NeuroVision-X dual encoder, is a well-tested, resumable 34.9-million-parameter network that beats the matched U-Net on enhancing tumour by +0.0267, mostly because of gated cross-attention fusion. It is not more reliable, its claimed novelty adds nothing, its gain does not transfer to other populations, and it costs 23.7 times the computation."),
  P("Model 3 therefore keeps this network and changes the goal: test how robust the gain is (seeds, augmentation, test-time augmentation, a strong nnU-Net baseline), give the system ways to assess its own output (a quality-control network and a distribution-free error bound), measure exactly where those guarantees fail, find what a new site needs to restore them, and deploy the result in a pipeline that refuses studies it cannot handle."),
  QUOTE("*Model 2 deployed checkpoint: epoch 69, seed 42. Test Dice ET 0.8709, TC 0.9161, WT 0.9321; 34.9 M parameters; 23.1 GPU-hours.*"),

  H1("References"),
  REF("Baid, U. et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. *arXiv:2107.02314*."),
  REF("Adewole, M. et al. (2023). The Brain Tumor Segmentation (BraTS) challenge 2023: Glioma segmentation in sub-Saharan Africa patient population (BraTS-Africa). *arXiv:2305.19369*."),
  REF("Kazerooni, A. F. et al. (2023). The Brain Tumor Segmentation (BraTS) challenge 2023: Focus on pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs). *arXiv:2305.17033*."),
  REF("Liu, Z. et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. *ICCV*, 10012–10022."),
  REF("Hatamizadeh, A. et al. (2022). Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images. *BrainLes, MICCAI Workshops*, 272–284."),
  REF("Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*, 5998–6008."),
  REF("Touvron, H. et al. (2021). Going deeper with image transformers (LayerScale). *ICCV*, 32–42."),
  REF("Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*, 1050–1059."),
  REF("Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, 1321–1330."),
  REF("Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70."),
  REF("Isensee, F. et al. (2024). nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation. *MICCAI*."),
];

const meta = { kicker: "MODEL 2 SUBMISSION", period: "Development period: 4 – 23 August 2026",
  subtitle: "The NeuroVision-X Dual-Encoder Network: CNN and Swin Transformer with Adaptive Gated Cross-Attention Fusion", date: "November 2026" };
const doc = L.makeDoc({ title: "NeuroVision-X Model 2: Dual-Encoder Network", titlePage: L.titlePage(meta),
  front: L.frontMatter(abstract, abbrevs), body });
L.write(doc, process.argv[2] || path.join(__dirname, "model2.docx"));
