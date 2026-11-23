// Model 1 submission: baseline 3D U-Net, data pipeline and training foundations.
const L = require("./lib.js");
const { P, H1, H2, H3, B, NL, figure, table, REF, QUOTE, fig, path, ROOT } = L;
const early = (n) => path.join(ROOT, "outputs/figures", n + ".png");

const abstract = [
  L.P("This document is the first of three submissions describing the development of the deep-learning model behind NeuroVision-X, a three-dimensional brain tumour segmentation system for multi-modal MRI. It covers **Model 1**, the period from 31 July to 9 August 2026, in which the data pipeline, the training system and the baseline network were built and the first trained models were measured."),
  L.P("All 1,251 cases of the BraTS 2021 training set were preprocessed (reorientation, 1 mm resampling, per-sequence z-score normalisation, cropping) without a single failure, and split once, with a fixed seed, into 875 training, 187 validation and 189 test cases. The model is a MONAI 3D U-Net with five resolution levels and residual units (12.87 million parameters), predicting the three nested BraTS regions, enhancing tumour (ET), tumour core (TC) and whole tumour (WT), with sigmoid outputs trained by Dice plus binary cross-entropy. Training runs on a free Kaggle Tesla T4 GPU with mixed precision and a resume system that restores every part of the training state, so a 12-hour session cap never loses work."),
  L.P("The first run (200 epochs, 96³ patches, 16.5 GPU-hours) reached test Dice ET 0.8587, TC 0.9157 and WT 0.9354. After a timing probe showed that the planned dual-encoder model would not fit the GPU budget at that recipe, the baseline was retrained under the shared 80-epoch, 64³ recipe used for every later comparison (3.2 GPU-hours), reaching ET 0.8442, TC 0.9058 and WT 0.9276. Two measurements on this model shaped the rest of the project: 73.7% of whole-tumour error lies within 2 mm of the true boundary, and the network is over-confident, with an honest expected calibration error of 0.0565 that one temperature per region reduces to 0.0158. On the way, a calibration bug that manufactured 41–57% of the reported error was found and removed. Model 1 is the reference that Model 2 must beat."),
  L.P("**Keywords:** brain tumour segmentation, BraTS 2021, 3D U-Net, MONAI, preprocessing, resumable training, calibration."),
];

const abbrevs = [
  ["BraTS", "Brain Tumor Segmentation challenge and dataset"],
  ["ET / TC / WT", "Enhancing tumour / tumour core / whole tumour"],
  ["NCR, ED", "Necrotic tumour core; peritumoural oedema"],
  ["T1, T1CE, T2, FLAIR", "The four MRI sequences: T1-weighted, contrast-enhanced T1, T2-weighted, fluid-attenuated inversion recovery"],
  ["CNN", "Convolutional neural network"],
  ["BCE", "Binary cross-entropy"],
  ["AMP", "Automatic mixed precision (float16 arithmetic)"],
  ["HD95", "95th-percentile Hausdorff distance (mm)"],
  ["ECE", "Expected calibration error"],
  ["GPU-h", "GPU hours on a Kaggle Tesla T4"],
  ["RNG", "Random number generator"],
  ["W&B", "Weights & Biases experiment tracker"],
];

const body = [
  // ------------------------------------------------------------ 1
  H1("1. Introduction"),
  H2("1.1 Background"),
  P("Gliomas are the most common primary malignant brain tumours in adults. Planning surgery or radiotherapy for one depends on knowing where the tumour is and how large each part of it is. On MRI a glioma appears as three nested sub-regions: the **enhancing tumour (ET)**, bright on contrast-enhanced T1; the **tumour core (TC)**, which adds necrotic and non-enhancing tissue; and the **whole tumour (WT)**, which adds the surrounding oedema seen on FLAIR. Outlining these by hand in 3D takes an expert a long time and experts disagree, so automatic segmentation is an active research area. The BraTS challenge provides the standard data and evaluation."),
  H2("1.2 Aim of the whole project"),
  P("NeuroVision-X builds a complete 3D brain tumour segmentation system and then measures how far its reliability can be trusted when the data changes. The deep-learning model was developed in three stages, each submitted separately (Figure 1):"),
  ...NL([
    "**Model 1 (this document):** the data pipeline, the training system and a strong, honest baseline, a 3D U-Net.",
    "**Model 2:** the proposed NeuroVision-X dual-encoder network (CNN + Swin Transformer with gated cross-attention fusion), compared against Model 1 with matched training.",
    "**Model 3:** the final model: robustness checks, a quality-control network, a distribution-free error bound, adaptation to new sites, a benchmark against nnU-Net, and the deployed clinical pipeline.",
  ]),
  ...figure(fig("fig_evolution"), "The three stages of model development and what each one delivered."),
  H2("1.3 Objectives of Model 1"),
  ...NL([
    "Prepare all 1,251 BraTS 2021 training cases into a fast, verified on-disk format and freeze one train / validation / test split that every later model uses.",
    "Build a training system that runs unmodified on any single 16 GB GPU, uses mixed precision, and survives being killed at any moment.",
    "Train a 3D U-Net baseline and measure it with Dice, HD95, per-case distributions, error location and calibration.",
    "Fix the training recipe that all later models will share, so that comparisons are fair.",
  ]),
  H2("1.4 Why start with a baseline"),
  P("A new architecture can only be judged against something. If the baseline is weak, trained for fewer epochs, or evaluated differently, any new model looks good. Model 1 therefore had two jobs: to be a reasonable model in its own right, and to define the exact recipe (patch size, epochs, optimiser, augmentation, evaluation) that Model 2 would be held to. Getting this right first is why the later comparisons in Model 2 can be trusted."),
  H2("1.5 Organisation"),
  P("Chapter 2 describes the dataset, Chapter 3 the preprocessing and data pipeline, Chapter 4 the network, Chapter 5 the training system, and Chapter 6 the evaluation method. Chapter 7 gives the results, Chapter 8 the problems found and fixed, Chapter 9 the limitations, and Chapter 10 the conclusion and the step to Model 2."),

  // ------------------------------------------------------------ 2
  H1("2. Dataset"),
  H2("2.1 BraTS 2021"),
  P("The training data is the BraTS 2021 training set: **1,251** adult glioma cases. Each case has four co-registered, skull-stripped MRI volumes of 240 × 240 × 155 voxels at 1 mm isotropic spacing, and an expert label map. The four sequences show different tissue properties, which is why the model receives all of them as input channels (Table 1)."),
  ...table("The four MRI sequences used as input channels.", ["Sequence", "What it shows best"], [
    ["T1", "Anatomy; grey and white matter structure"],
    ["T1CE", "Contrast-enhanced T1; enhancing tumour and necrotic core"],
    ["T2", "Fluid and oedema; tumour extent"],
    ["FLAIR", "Oedema with fluid signal suppressed; whole-tumour boundary"],
  ], [2, 7]),
  H2("2.2 Labels and target regions"),
  P("The label map has four values: 0 background, 1 necrotic core (NCR), 2 oedema (ED) and 4 enhancing tumour (ET). Clinical evaluation, and the BraTS metric, use three **nested** regions built from these labels (Table 2). Because the regions overlap (every ET voxel is also a TC and a WT voxel), the network predicts them as three independent binary channels with a sigmoid, not as mutually exclusive classes with a softmax."),
  ...table("Target regions. Each is a union of raw labels, so ET is inside TC which is inside WT.", ["Region", "Raw labels", "Clinical meaning"], [
    ["ET", "4", "Actively enhancing tumour"],
    ["TC", "1 + 4", "Tumour core, the usual surgical target"],
    ["WT", "1 + 2 + 4", "Whole tumour including oedema"],
  ], [1.4, 2, 5.6]),
  H2("2.3 The split"),
  P("The 1,251 cases were divided once, with a seeded random generator, into **875 training, 187 validation and 189 test** cases (70 / 15 / 15%). The split is written to a committed file, `configs/data/splits.yaml`, and never changes. The validation set chooses the best checkpoint and fits calibration temperatures; the test set is touched only for final numbers. Every model in all three submissions uses this same split, so their test numbers are paired case by case."),
  P("One limitation is stated now because it matters later: the test set is a random part of the BraTS *training* set, so it comes from the same hospitals and scanners as the training data. It measures in-distribution performance only. Model 2 and Model 3 add external cohorts to measure performance under distribution shift."),

  // ------------------------------------------------------------ 3
  H1("3. Preprocessing and Data Pipeline"),
  H2("3.1 Offline preprocessing"),
  P("Loading and normalising four NIfTI volumes on every training step would make the GPU wait for the CPU. All preprocessing is therefore done once, offline, on the laptop, and the result is saved as compact NumPy arrays (Figure 2)."),
  ...figure(fig("fig_data_pipeline"), "The data pipeline from a raw BraTS case to a training patch."),
  ...NL([
    "**Reorientation and resampling.** Each volume is reoriented using its own affine matrix and resampled to 1 mm isotropic. Deriving the orientation from each file rather than assuming it later proved essential: the African cohort used in Model 2 is stored with two axes reversed.",
    "**Intensity normalisation.** Each sequence is z-scored using only the non-zero (brain) voxels, so the large black background does not distort the mean and standard deviation. MRI intensities have no absolute unit, so this step is what makes scans from different machines comparable.",
    "**Cropping.** The volume is cropped to the bounding box of the brain, which removes most of the empty background and reduces memory and disk use.",
    "**Label remapping.** Labels are checked and remapped; a case with unexpected label values is rejected instead of silently passed through.",
    "**Storage.** Images are stored as float16 and labels as uint8 `.npy` files, with a `meta.json` holding the affine, the crop box and the original shape so a prediction can be placed back into the original scan.",
  ]),
  P("All **1,251 cases** were processed with **zero failures**, producing 34.2 GB. The preprocessed set was also uploaded as a private Kaggle dataset so the GPU sessions can read it directly."),
  H2("3.2 Online patch sampling and augmentation"),
  P("A whole cropped brain does not fit through a 3D network on a 16 GB GPU together with its gradients, so training uses random **patches**. Each step loads one volume and draws four patches from it; half are centred on tumour and half are random, so the network sees enough tumour without forgetting what healthy tissue looks like. The labels are converted to the three nested region channels after cropping. Light augmentation is then applied (Table 3)."),
  ...table("Training-time augmentation used by Model 1 (and kept for Model 2).", ["Transform", "Setting", "Purpose"], [
    ["Random flip", "p = 0.5 on each of 3 axes", "Left-right and other symmetry"],
    ["Random 90° rotation", "p = 0.5", "Orientation invariance"],
    ["Intensity scale", "±10%", "Scanner gain differences"],
    ["Intensity shift", "±0.1 (z-score units)", "Offset differences"],
    ["Gaussian noise", "p = 0.15, std 0.01", "Acquisition noise"],
    ["Patch sampling", "64³ (final recipe), 4 per volume, 1:1 tumour : random", "Memory and class balance"],
  ], [2.3, 3.5, 3.2]),
  P("Heavier augmentations (rotation by arbitrary angles, gamma, bias field, elastic deformation) are implemented but switched off. Model 3 tests whether turning them on helps under distribution shift."),

  // ------------------------------------------------------------ 4
  H1("4. Model Architecture"),
  H2("4.1 Why a 3D U-Net"),
  P("The U-Net is the standard architecture for medical image segmentation. An **encoder** repeatedly halves the resolution while increasing the number of feature channels, so deep layers see large regions of the brain and learn *what* is there. A **decoder** reverses this and restores full resolution. **Skip connections** copy the encoder's fine-resolution features directly to the decoder at the matching level, so the decoder can recover *where* the boundaries are. Working in 3D, with 3D convolutions, lets the network use context from neighbouring slices, which matters because a tumour is a 3D object."),
  ...figure(fig("fig_unet"), "The Model 1 network: a MONAI 3D U-Net with five resolution levels and residual units."),
  H2("4.2 Configuration"),
  P("The network is MONAI's `UNet`, configured as in Table 4. It is registered under the name `unet3d` in the project's model registry, so an experiment selects it by a string in its configuration file."),
  ...table("Model 1 network configuration.", ["Setting", "Value"], [
    ["Input", "4 channels (T1, T1CE, T2, FLAIR), patch 64³ (final) or 96³ (first run)"],
    ["Output", "3 channels (ET, TC, WT), sigmoid"],
    ["Channels per level", "32, 64, 128, 256, 320"],
    ["Downsampling", "Strided convolution, stride 2 at each level (total 16)"],
    ["Residual units per block", "2"],
    ["Normalisation", "Instance normalisation"],
    ["Activation", "LeakyReLU"],
    ["Dropout", "0.1"],
    ["Parameters", "12,870,489"],
  ], [3, 6]),
  P("Two choices deserve explanation. **Instance normalisation** normalises each sample on its own, which suits a batch size of one volume and MRI whose intensity varies from scan to scan. **Dropout of 0.1** regularises the model and also makes it possible to estimate uncertainty later by running the network several times with dropout switched on (Monte Carlo dropout)."),
  H2("4.3 Output and post-processing"),
  P("Each output voxel holds three probabilities, one per region, thresholded at 0.5. Connected components smaller than 50 voxels are removed, which deletes isolated false-positive specks. Because each region is predicted by its own channel, and removing a small component from one channel can leave it outside another, a final step re-imposes the nesting ET ⊂ TC ⊂ WT so the output is always anatomically consistent."),

  // ------------------------------------------------------------ 5
  H1("5. Training System"),
  H2("5.1 Loss function"),
  P("The loss is the sum of a soft **Dice loss** and **binary cross-entropy (BCE)**, each with weight 1, computed per region channel with a sigmoid. Dice directly optimises overlap and is robust to the heavy imbalance between tiny tumours and a large background; BCE gives smooth, well-behaved gradients early in training when Dice is still noisy. The project uses its own implementation rather than MONAI's `DiceCELoss`, because that one applies a softmax cross-entropy that is wrong for overlapping region channels."),
  H2("5.2 Optimisation"),
  ...table("Final shared training recipe, used by Model 1 and inherited unchanged by Model 2.", ["Setting", "Value", "Reason"], [
    ["Optimiser", "AdamW, lr 1e-4, weight decay 1e-5", "Standard for 3D segmentation"],
    ["Schedule", "5-epoch linear warm-up, then cosine to 1e-6", "Stable start, smooth decay"],
    ["Epochs", "80", "Set by the GPU budget (Section 7.1)"],
    ["Patches per step", "4 × 64³", "1 volume × 4 samples"],
    ["Mixed precision", "On (float16) for CUDA", "About 2× faster, half the memory"],
    ["Gradient clipping", "Norm 5.0", "Guards against rare spikes only"],
    ["Validation", "Every 10 epochs, sliding window", "Selects best.pt"],
    ["Seed", "42", "Reproducibility"],
  ], [2.4, 3.6, 3]),
  P("The gradient-clipping threshold was first set to 1.0. Logging the gradient norm in a short probe showed that this clipped 66–70% of all steps, meaning the model was really training at a smaller learning rate than the configuration stated. It was raised to 5.0, above the measured maximum, so clipping only catches genuine spikes; on the final baseline run only 1.6% of steps were clipped."),
  H2("5.3 Running on a free GPU that gets killed"),
  P("All training ran on Kaggle's free Tesla T4 (16 GB), which ends every session after 12 hours and limits weekly GPU time. A run longer than one session has to stop and continue in a new session without changing the result. The training system was built around this (Figure 4)."),
  ...figure(fig("fig_training_system"), "The training system. A checkpoint holds the full training state, so a killed session resumes exactly."),
  B("**Full resume.** Every checkpoint stores the model, optimiser, scheduler and AMP scaler state, the epoch and global step, the best metric so far, all random-number-generator states (Python, NumPy, PyTorch and CUDA) and the Weights & Biases run ID. Restarting the same command continues as if nothing happened."),
  B("**Atomic saving.** A checkpoint is written to a temporary file and renamed, so a kill during a save cannot corrupt `last.pt`."),
  B("**Time budget.** The trainer predicts how long the next epoch will take and stops cleanly before the session limit."),
  B("**Pinned code.** The Kaggle notebook clones the repository at a fixed commit, so the code that produced a number can always be recovered."),
  B("**Smoke test.** A 4-second end-to-end CPU test runs one tiny training and validation step before any GPU session is spent."),
  B("**Configuration.** Everything, including every file path, comes from Hydra configuration files; experiments are selected by name. The same code runs on the Mac (CPU, for tests) and on Linux (GPU)."),
  P("The code base is covered by automated CPU tests; every model component has a shape test on tiny tensors that runs in under a second, and the resume path has its own tests."),

  // ------------------------------------------------------------ 6
  H1("6. Evaluation Method"),
  H2("6.1 Inference"),
  P("Test volumes are much larger than a training patch, so prediction uses **sliding-window inference**: the volume is covered by overlapping patches of the training size (50% overlap), each is predicted, and overlapping predictions are blended with a Gaussian weighting that trusts each patch's centre more than its edges. This removes visible seams. The network's raw outputs (logits) are saved in float16 so every later metric can be recomputed without running the network again."),
  H2("6.2 Metrics"),
  B("**Dice coefficient**, 2|A ∩ B| / (|A| + |B|), the overlap between prediction and ground truth, from 0 (none) to 1 (perfect), per region."),
  B("**HD95**, the 95th-percentile distance in millimetres between the predicted and the true surfaces; it measures the worst boundary errors while ignoring the most extreme 5%."),
  B("**Per-case distributions**, because a mean hides the few cases where the model fails badly."),
  B("**Error location**, the share of wrong voxels at each distance from the true boundary."),
  B("**Expected calibration error (ECE)**, the average gap between the predicted probability and the actual frequency of tumour, measured on a mask that never uses the ground-truth label (Section 8.1)."),
  H2("6.3 Statistics"),
  P("Comparisons between models are **paired**: both models are scored on the same 189 cases and the per-case differences are analysed. Confidence intervals come from a paired bootstrap that resamples cases; significance uses the Wilcoxon signed-rank test with Holm correction across the family of comparisons. These tools were built and tested in Model 1 and used for every comparison in Models 2 and 3."),

  // ------------------------------------------------------------ 7
  H1("7. Results"),
  H2("7.1 Two training runs"),
  P("Model 1 was trained twice (Table 6, Figure 5). The first run used a long recipe: 200 epochs at 96³ patches over two Kaggle sessions (16.5 GPU-hours). Then a timing probe of the planned Model 2 network showed it would take 3.6 seconds per step at 96³, about 91 GPU-hours for 100 epochs, and it had to be trained at least twice. That was more than three times the entire budget. The shared recipe was therefore cut to **64³ patches and 80 epochs**, and the baseline was retrained under exactly that recipe (3.2 GPU-hours, one session) so that it would be a fair comparison for Model 2."),
  ...table("Test-set results of the two baseline runs (n = 189). HD95 in mm.", ["Run", "Dice ET", "Dice TC", "Dice WT", "HD95 ET", "HD95 TC", "HD95 WT", "GPU-h"], [
    ["Run 1: 200 epochs, 96³", "0.8587", "0.9157", "0.9354", "4.02", "5.21", "5.36", "16.5"],
    ["Run 2: 80 epochs, 64³ (reference)", "0.8442", "0.9058", "0.9276", "4.91", "5.24", "6.72", "3.2"],
  ], [3.2, 1, 1, 1, 1, 1, 1, 0.9]),
  ...figure(fig("fig_baseline_runs"), "Mean test Dice of the two baseline runs. The cheaper recipe costs about one Dice point."),
  P("The cheaper recipe costs about **one Dice point** and 0.9–1.4 mm of HD95, most on whole tumour, which is the largest structure and loses most from a smaller field of view. Because every later model uses the same recipe, this cost applies equally to all of them and does not affect comparisons, but it does mean the absolute numbers are not state of the art. **Run 2 is the reference row** that Model 2 is compared against. The baseline used only 0.50 GiB of the T4's memory, so the U-Net is far from the 16 GB limit."),
  H2("7.2 Training behaviour"),
  P("Figure 6 shows the first run's training curves. The loss falls quickly in the first few thousand steps and then slowly; validation Dice rises above 0.8 early and keeps improving slowly; the learning rate follows the warm-up and cosine schedule exactly as configured. No divergence or instability occurred."),
  ...figure(early("training_curves"), "Training loss, validation Dice and learning-rate schedule for the first baseline run (200 epochs, 96³).", 6.3),
  H2("7.3 Per-case results"),
  P("Mean Dice hides failures. Figure 7 shows the per-case distribution for the first run. Most cases score above 0.9 on whole tumour and tumour core, but a tail of cases scores far lower, and on enhancing tumour a handful of cases score zero; these are mostly cases with very little or no enhancing tumour (five test cases have none at all), where a few wrong voxels decide the score. The median ET Dice (0.929) is well above the mean (0.859) for this reason."),
  ...figure(early("per_case_dice"), "Per-case test Dice for the first baseline run, as box plots and as cumulative distributions.", 6.3),
  P("Figure 8 shows the worst, median and best cases by Dice. The median and best predictions are close to the expert labels; the worst case shows the model placing small core and enhancing regions inside an area the expert labelled as oedema only."),
  ...figure(early("qualitative_best_median_worst"), "Worst, median and best test cases for the first baseline run: FLAIR, ground truth, prediction and whole-tumour agreement (green correct, red false positive, blue false negative).", 5.6),
  H2("7.4 Where the errors are"),
  P("For the reference run, every wrongly labelled whole-tumour voxel was assigned to a distance band from the true boundary, and the shares were weighted by voxel count (Figure 9). **73.7% of error lies within 2 mm of the boundary.** Most of the remaining error is in a distinct second failure mode: 8.8% lies more than 10 mm away, almost entirely false-positive specks far from any tumour, concentrated in a small number of cases. The first version of this analysis averaged the band error *rates* and reported 92%, which treats a thin band as equal to a thick one; correcting it changed the conclusion from \"only the boundary matters\" to \"the boundary dominates, and there is also a spurious-lesion problem\". That second problem returns in Model 2."),
  ...figure(fig("fig_boundary_bands"), "Share of whole-tumour error voxels by distance from the true boundary, reference baseline run."),
  H2("7.5 Calibration"),
  P("A calibrated model's 80% predictions are right 80% of the time. Measured on the first run's epoch-130 checkpoint with a label-free mask, the baseline is **over-confident**: mean ECE 0.0565 (Figure 10). **Temperature scaling**, dividing the logits by one constant per region fitted on the validation set, reduces it 3.6 times to **0.0158**. This set an important bar: any later claim that a new model is \"better calibrated\" has to beat a temperature-scaled baseline, not an uncalibrated one, because temperature scaling costs seconds and can be applied to any model."),
  ...figure(fig("fig_calibration_m1"), "Expected calibration error of the baseline before and after temperature scaling."),

  // ------------------------------------------------------------ 8
  H1("8. Problems Found and Fixed"),
  P("Several faults were found while building Model 1. Each is recorded because each one would otherwise have produced a wrong number with every automated test passing."),
  H2("8.1 The calibration measurement used the answer"),
  P("The first calibration run looked wrong: the fitted temperatures were 3.7–4.8, far outside the normal 1.1–2.0, and whole-tumour calibration got *worse* after scaling. The cause was the mask that chose which voxels to measure: it included every voxel where the model predicted tumour *or* the ground truth was tumour. A low-probability voxel could only enter that mask by being true tumour, so by construction those voxels always looked under-confident. This **manufactured 41–57% of the reported ECE**, while all 984 tests passed, because the code did exactly what it said. The fix was to use only label-free masks (the predicted region, or the whole brain). The rule that came out of it, *a reporting mask must never be defined using the label*, is applied to every later analysis."),
  H2("8.2 Other faults"),
  ...table("Engineering faults found during Model 1 and how they were fixed.", ["Fault", "Effect", "Fix"], [
    ["RNG state restored to the GPU on resume", "Resume crashed on Kaggle", "Move state to CPU before restoring; regression tests"],
    ["Kaggle P100 GPU", "Stock PyTorch does not support it", "Always request the T4"],
    ["MONAI HD95 on GPU tensors", "Fails to compile on Kaggle", "Compute HD95 on the CPU"],
    ["Gradient clipping at 1.0", "66–70% of steps clipped", "Raised to 5.0 from measured norms"],
    ["Notebook cloned the latest code", "First run's exact source lost", "Pin every run to a commit"],
    ["Clipping before AMP unscaling", "Would clip every step to nearly nothing", "Unscale first, then clip; tested"],
  ], [3, 3, 3]),
  P("The lost commit of the first run is why that run is reported but never used as a comparison: its exact code cannot be recovered. From then on every run was pinned to a specific commit."),

  // ------------------------------------------------------------ 9
  H1("9. Limitations"),
  B("The test set is a random split of the BraTS training data, so it measures only in-distribution performance."),
  B("The final recipe (64³, 80 epochs) was chosen to fit the budget for Model 2, and costs about one Dice point against the longer recipe."),
  B("Only one training seed was run in this stage, so the run-to-run variation was not yet known. It is measured in Model 3."),
  B("Mean voxel Dice is the headline metric here; Model 3 shows it flatters performance compared with the lesion-wise metric BraTS now uses."),
  B("The first run's code version was not recorded."),

  // ------------------------------------------------------------ 10
  H1("10. Conclusion and Next Step"),
  P("Model 1 delivered a verified data pipeline, a robust training system that survives the free GPU's session limits, and a 3D U-Net baseline with test Dice ET 0.8442, TC 0.9058 and WT 0.9276 under the shared recipe. It also delivered two measurements that set the agenda: errors concentrate at the tumour boundary with a second population of spurious lesions, and the network is over-confident, though one temperature per region largely fixes that."),
  P("These motivated Model 2. A CNN sees local texture well but has a limited field of view; a transformer sees global context. The hypothesis for Model 2 is that combining the two, with a gate that decides at each location how much to trust each, will segment the boundary better and be better calibrated than this baseline. Model 2 is held to exactly the recipe, split, metrics and statistics defined here."),
  QUOTE("*Model 1 reference row (test, n = 189): ET 0.8442, TC 0.9058, WT 0.9276; 12.87 M parameters; 3.2 GPU-hours.*"),

  H1("References"),
  REF("Baid, U. et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. *arXiv:2107.02314*."),
  REF("Menze, B. H. et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024."),
  REF("Ronneberger, O., Fischer, P. and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 234–241."),
  REF("Çiçek, Ö. et al. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. *MICCAI*, 424–432."),
  REF("Kerfoot, E. et al. (2019). Left-ventricle quantification using residual U-Net. *STACOM, MICCAI Workshops*, 371–380."),
  REF("Cardoso, M. J. et al. (2022). MONAI: An open-source framework for deep learning in healthcare. *arXiv:2211.02701*."),
  REF("Milletari, F., Navab, N. and Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*, 565–571."),
  REF("Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*."),
  REF("Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, 1321–1330."),
  REF("Isensee, F. et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203–211."),
];

const meta = { kicker: "MODEL 1 SUBMISSION", period: "Development period: 31 July – 9 August 2026",
  subtitle: "Baseline 3D U-Net: Data Pipeline, Training System and Reference Model", date: "November 2026" };
const doc = L.makeDoc({ title: "NeuroVision-X Model 1: Baseline 3D U-Net", titlePage: L.titlePage(meta),
  front: L.frontMatter(abstract, abbrevs), body });
L.write(doc, process.argv[2] || path.join(__dirname, "model1.docx"));
