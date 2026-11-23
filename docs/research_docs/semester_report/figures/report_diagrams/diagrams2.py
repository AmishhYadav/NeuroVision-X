"""Extra diagrams for the three model submission documents (Model 1, 2, 3).

Numbers copied from docs/research_docs/experiments.md and claims_and_evidence.md.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from diagrams import (BLUE, GREEN, GREY, LBLUE, LGREEN, LGREY, LORANGE, LPURPLE, LRED, LYELLOW, ORANGE,
                      PURPLE, RED, YELLOW, arrow, box, save)

plt.rcParams["font.family"] = "Times New Roman"


def bar_labels(ax, bars, fmt="{:.4f}", dy=0.002, fs=8.5):
    for b in bars:
        h = b.get_height() + (b.get_y() if b.get_y() else 0)
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(b.get_height()), ha="center",
                va="bottom", fontsize=fs)


# ------------------------------------------------------------ evolution (all docs)
def evolution():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    items = [
        (0.1, "Model 1\nBaseline 3D U-Net", "31 Jul – 9 Aug 2026",
         "Data pipeline, trainer with\nfull resume, 12.9 M U-Net.\nTest ET Dice 0.8442\n(matched 80-epoch recipe)",
         LBLUE, BLUE),
        (3.45, "Model 2\nNeuroVision-X dual encoder", "4 Aug – 23 Aug 2026",
         "CNN + Swin, gated fusion,\n34.9 M params. ET +0.0267\nover Model 1; gain is gated\nfusion, not disagreement",
         LORANGE, ORANGE),
        (6.8, "Model 3\nFinal: robust + guarded", "23 Aug – Oct 2026",
         "Seeds, augmentation, TTA,\nQC model, conformal bound,\nfine-tuning, nnU-Net check,\nclinical deployment",
         LGREEN, GREEN),
    ]
    for x, title, date, body, fc, ec in items:
        box(ax, x, 1.95, 3.1, 0.95, title, fc, ec, fs=10.5, bold=True)
        ax.text(x + 1.55, 1.75, date, ha="center", fontsize=9, style="italic")
        box(ax, x, 0.15, 3.1, 1.4, body, "white", ec, fs=9)
    arrow(ax, 3.2, 2.42, 3.45, 2.42, lw=2)
    arrow(ax, 6.55, 2.42, 6.8, 2.42, lw=2)
    ax.text(5, 3.2, "Development of the deep-learning model in three stages", ha="center", fontsize=11,
            fontweight="bold")
    save(fig, "fig_evolution")


# ------------------------------------------------------------ Model 1 figures
def unet():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    enc = [("32", 4.0), ("64", 3.25), ("128", 2.5), ("256", 1.75)]
    for i, (c, y) in enumerate(enc):
        box(ax, 1.3 + i * 0.55, y, 1.2, 0.5, f"Enc {i + 1}\n{c} ch", LBLUE, BLUE, fs=8.5)
        box(ax, 7.5 - i * 0.55, y, 1.2, 0.5, f"Dec {i + 1}\n{c} ch", LGREEN, GREEN, fs=8.5)
        ax.annotate("", xy=(7.5 - i * 0.55, y + 0.25), xytext=(2.5 + i * 0.55, y + 0.25),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, ls=":", lw=1.1))
        if i:
            arrow(ax, 1.3 + (i - 1) * 0.55 + 0.6, enc[i - 1][1], 1.3 + i * 0.55 + 0.6, y + 0.5, color=BLUE)
            arrow(ax, 7.5 - i * 0.55 + 0.6, y + 0.5, 7.5 - (i - 1) * 0.55 + 0.6, enc[i - 1][1], color=GREEN)
    box(ax, 4.0, 0.85, 2.0, 0.55, "Bottleneck\n320 ch, stride 16", LPURPLE, PURPLE, fs=8.5)
    arrow(ax, 3.55, 1.75, 4.4, 1.4, color=BLUE)
    arrow(ax, 5.6, 1.4, 6.45, 1.75, color=GREEN)
    box(ax, 0.0, 3.95, 1.1, 0.6, "Input\n4 x 96³", LGREY, GREY, fs=8.5)
    arrow(ax, 1.1, 4.25, 1.3, 4.25)
    box(ax, 8.85, 3.95, 1.1, 0.6, "Output\n3 x 96³\nET/TC/WT", LRED, RED, fs=8)
    arrow(ax, 8.7, 4.25, 8.85, 4.25)
    ax.text(5, 4.62, "Skip connections (dotted) copy encoder features to the decoder", ha="center",
            fontsize=9, color=GREY)
    ax.text(5, 0.35, "Each block: 2 residual units of Conv3d, InstanceNorm, LeakyReLU, dropout 0.1. "
            "Strided convolutions downsample by 2.", ha="center", fontsize=9, style="italic")
    ax.text(5, 0.05, "MONAI UNet, channels [32, 64, 128, 256, 320], 12,870,489 parameters.", ha="center",
            fontsize=9, style="italic")
    save(fig, "fig_unet")


def data_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    steps = [("Raw BraTS case\nT1, T1CE, T2, FLAIR\n+ label (NIfTI)", LGREY, GREY),
             ("Reorient from\nown affine,\n1 mm isotropic", LBLUE, BLUE),
             ("Non-zero z-score\nper sequence", LBLUE, BLUE),
             ("Crop to brain\nbounding box,\nsave .npy + meta", LBLUE, BLUE),
             ("Labels to nested\nregions\nET $\\subset$ TC $\\subset$ WT", LPURPLE, PURPLE),
             ("Random patch +\naugmentation\n(training only)", LGREEN, GREEN)]
    w = 1.5
    for i, (t, fc, ec) in enumerate(steps):
        x = 0.05 + i * 1.66
        box(ax, x, 0.9, w, 1.1, t, fc, ec, fs=8.5)
        if i:
            arrow(ax, x - 0.16, 1.45, x, 1.45)
    ax.text(5, 0.4, "1,251 cases preprocessed with no failures (34.2 GB); split frozen 875 / 187 / 189 with a "
            "seeded generator.", ha="center", fontsize=9, style="italic")
    save(fig, "fig_data_pipeline")


def training_system():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    box(ax, 0.1, 2.3, 2.0, 0.9, "Hydra config\n(experiment chosen\nby name)", LGREY, GREY, fs=9)
    box(ax, 2.6, 2.3, 2.2, 0.9, "Trainer\nAMP, clipping,\nwarm-up + cosine LR", LBLUE, BLUE, fs=9)
    box(ax, 5.3, 2.3, 2.1, 0.9, "Sliding-window\nvalidation\n(every N epochs)", LGREEN, GREEN, fs=9)
    box(ax, 7.9, 2.3, 2.0, 0.9, "Atomic checkpoint\nlast.pt / best.pt", LORANGE, ORANGE, fs=9)
    for a, b in [(2.1, 2.6), (4.8, 5.3), (7.4, 7.9)]:
        arrow(ax, a, 2.75, b, 2.75)
    box(ax, 2.6, 0.3, 4.8, 1.2, "Checkpoint stores: model, optimiser, scheduler, AMP scaler, epoch,\n"
        "global step, best metric, every RNG state (Python, NumPy, PyTorch, CUDA),\n"
        "Weights & Biases run ID, resolved config", LYELLOW, YELLOW, fs=7.8)
    ax.annotate("", xy=(3.7, 3.2), xytext=(8.9, 3.2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4, ls="--", connectionstyle="arc3,rad=0.3"))
    ax.text(6.3, 4.15, "Kaggle session killed (12 h cap)? Re-run the same command: it resumes exactly from last.pt",
            ha="center", fontsize=8.5, color=RED)
    box(ax, 0.1, 0.45, 2.0, 0.9, "Kaggle T4 notebook\nclones repo at a\npinned commit", LGREY, GREY, fs=8.5)
    box(ax, 7.9, 0.45, 2.0, 0.9, "4-second CPU\nsmoke test before\nevery session", LGREY, GREY, fs=8.5)
    save(fig, "fig_training_system")


def boundary_bands():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bands = ["0–2 mm", "2–5 mm", "5–10 mm", "10 mm +"]
    vals = [73.7, 14.0, 3.5, 8.8]
    bars = ax.bar(bands, vals, color=[BLUE, ORANGE, GREEN, RED], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=9)
    ax.set_ylabel("Share of whole-tumour error voxels (%)")
    ax.set_xlabel("Distance from the true tumour boundary")
    ax.set_ylim(0, 85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Where the baseline U-Net makes its errors (WT, test split)", fontsize=10.5)
    save(fig, "fig_boundary_bands")


def baseline_runs():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    regions = ["ET", "TC", "WT"]
    a = [0.8587, 0.9157, 0.9354]
    b = [0.8442, 0.9058, 0.9276]
    x = range(3)
    b1 = ax.bar([i - 0.18 for i in x], a, 0.36, color=BLUE, label="Run 1: 200 epochs, 96³ patches (16.5 GPU-h)")
    b2 = ax.bar([i + 0.18 for i in x], b, 0.36, color=ORANGE, label="Run 2: 80 epochs, 64³ patches (3.2 GPU-h)")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(regions)
    ax.set_ylim(0.8, 0.97)
    ax.set_ylabel("Mean test Dice (n = 189)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_baseline_runs")


def calibration_m1():
    fig, ax = plt.subplots(figsize=(7, 3.3))
    regions = ["ET", "TC", "WT", "Mean"]
    before = [0.0864, 0.0504, 0.0327, 0.0565]
    after = [0.0197, 0.0157, 0.0121, 0.0158]
    x = range(4)
    b1 = ax.bar([i - 0.18 for i in x], before, 0.36, color=ORANGE, label="Uncalibrated")
    b2 = ax.bar([i + 0.18 for i in x], after, 0.36, color=GREEN, label="Temperature-scaled")
    bar_labels(ax, b1, dy=0.001)
    bar_labels(ax, b2, dy=0.001)
    ax.set_xticks(list(x))
    ax.set_xticklabels(regions)
    ax.set_ylim(0, 0.1)
    ax.set_ylabel("Expected calibration error")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Baseline calibration, label-free mask (96$^3$ run, epoch 130, test split)", fontsize=10.5)
    save(fig, "fig_calibration_m1")


# ------------------------------------------------------------ Model 2 figures
def m2_results():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    regions = ["ET", "TC", "WT"]
    a = [0.8442, 0.9058, 0.9276]
    b = [0.8709, 0.9161, 0.9321]
    x = range(3)
    b1 = ax.bar([i - 0.18 for i in x], a, 0.36, color="#AAAAAA", label="Model 1: matched U-Net (12.9 M)")
    b2 = ax.bar([i + 0.18 for i in x], b, 0.36, color=ORANGE, label="Model 2: NeuroVision-X (34.9 M)")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    for i, (t, sig) in enumerate([("+0.0267 *", True), ("+0.0103 *", True), ("+0.0045 n.s.", False)]):
        ax.text(i, 0.962, t, ha="center", fontsize=9, fontweight="bold" if sig else "normal")
    ax.set_xticks(list(x))
    ax.set_xticklabels(regions)
    ax.set_ylim(0.8, 0.975)
    ax.set_ylabel("Mean test Dice (n = 189)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("* Holm-significant, paired over the same 189 cases", fontsize=9, style="italic")
    save(fig, "fig_m2_results")


def m2_ablation():
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    names = ["Matched U-Net\n12.9 M", "Capacity control\nU-Net 34.8 M", "Content-only\ngate 34.9 M",
             "Full NeuroVision-X\n34.9 M"]
    vals = [0.8442, 0.8497, 0.8686, 0.8709]
    cols = ["#AAAAAA", BLUE, GREEN, ORANGE]
    bars = ax.bar(names, vals, color=cols, width=0.55)
    bar_labels(ax, bars, dy=0.0008)
    ax.set_ylim(0.83, 0.88)
    ax.set_ylabel("Test ET Dice (n = 189)")
    ax.spines[["top", "right"]].set_visible(False)
    steps = [("+0.0055\nwidth", 0.5), ("+0.0189\ngated fusion", 1.5), ("+0.0022\nn.s.", 2.5)]
    for t, x in steps:
        ax.text(x, 0.8765, t, ha="center", fontsize=8.5)
    ax.set_title("Ablation ladder: each step changes one thing", fontsize=10.5, pad=18)
    save(fig, "fig_m2_ablation")


def m2_uncertainty():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cohorts = ["BraTS-Africa (SSA)", "Paediatric (PED)"]
    ent = [0.8874, 0.8307]
    mc = [0.8661, 0.8437]
    dis = [0.7095, 0.7879]
    x = range(2)
    w = 0.25
    for j, (v, c, l) in enumerate([(ent, BLUE, "Single-pass entropy (free)"), (mc, GREEN, "MC dropout, 10 passes"),
                                   (dis, ORANGE, "Branch disagreement")]):
        bs = ax.bar([i + (j - 1) * w for i in x], v, w, color=c, label=l)
        bar_labels(ax, bs, dy=0.004, fs=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cohorts)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Mean per-case AUROC for voxel error")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_m2_uncertainty")


def m2_reliability():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    ax = axs[0]
    bs = ax.bar(["U-Net", "NeuroVision-X"], [0.0175, 0.0135], color=["#AAAAAA", ORANGE], width=0.5)
    bar_labels(ax, bs, dy=0.0003)
    ax.set_ylim(0, 0.024)
    ax.set_title("ECE after temperature scaling", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1]
    bs = ax.bar(["U-Net", "NeuroVision-X"], [40.6, 37.6], color=["#AAAAAA", ORANGE], width=0.5)
    bar_labels(ax, bs, fmt="{:.1f}%", dy=0.6)
    ax.set_ylim(0, 50)
    ax.set_title("MC-dropout risk-coverage\n(% of achievable gain)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Both differences inconclusive: more accurate, not more reliable", fontsize=10, y=1.03,
                 style="italic")
    save(fig, "fig_m2_reliability")


def m2_cost():
    fig, ax = plt.subplots(figsize=(7, 3.0))
    labels = ["Parameters (M)", "GFLOPs per 64³ patch", "CPU ms per patch"]
    u = [12.87, 11.3, 45.8]
    n = [34.91, 267.4, 1085.5]
    ratios = [n[i] / u[i] for i in range(3)]
    bs = ax.barh(labels, ratios, color=[BLUE, ORANGE, RED], height=0.5)
    for b, r, a, c in zip(bs, ratios, u, n):
        ax.text(r + 0.4, b.get_y() + b.get_height() / 2, f"{r:.1f}x  ({a:g} vs {c:g})", va="center", fontsize=9)
    ax.set_xlim(0, 32)
    ax.set_xlabel("NeuroVision-X relative to the U-Net")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    save(fig, "fig_m2_cost")


# ------------------------------------------------------------ Model 3 figures
def segqc():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    box(ax, 0.1, 1.0, 1.9, 1.3, "Inputs (3 ch, 64³)\nT1CE image\npredicted mask\nentropy map", LGREY, GREY, fs=9)
    ws = [16, 32, 64, 128]
    for i, w in enumerate(ws):
        box(ax, 2.4 + i * 1.15, 1.05, 0.95, 1.2, f"Conv3d\nGN, LReLU\n{w} ch", LBLUE, BLUE, fs=8.5)
        arrow(ax, 2.4 + i * 1.15 - 0.2, 1.65, 2.4 + i * 1.15, 1.65)
    box(ax, 7.05, 1.15, 1.1, 1.0, "Global\naverage\npool + MLP", LPURPLE, PURPLE, fs=8.5)
    arrow(ax, 6.85, 1.65, 7.05, 1.65)
    box(ax, 8.5, 1.15, 1.4, 1.0, "Predicted Dice\nfor this region\n(no label needed)", LGREEN, GREEN, fs=8.5)
    arrow(ax, 8.15, 1.65, 8.5, 1.65)
    ax.text(5, 0.45, "Trained on BraTS val masks degraded six ways (identity, erode, dilate, drop component, "
            "shift, speckle);", ha="center", fontsize=9, style="italic")
    ax.text(5, 0.12, "target = the degraded mask's true Dice. Run once per region (ET, TC, WT).",
            ha="center", fontsize=9, style="italic")
    ax.text(5, 2.75, "SegQC: a second network that estimates the segmentation's own quality", ha="center",
            fontsize=10.5, fontweight="bold")
    save(fig, "fig_segqc")


def m3_noise():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    labels = ["Headline:\nModel 2 − U-Net\n(seed 42)", "Replication:\nseed-43 pair", "Seed noise:\nModel 2 seed 43\n− seed 42",
              "Seed noise:\nU-Net seed 43\n− seed 42", "Flip TTA\n(8 flips)"]
    vals = [0.0267, 0.0247, 0.0021, 0.0041, 0.0007]
    lo = [0.0143, 0.0116, -0.0018, 0.0016, -0.0004]
    hi = [0.0432, 0.0420, 0.0068, 0.0072, 0.0017]
    cols = [ORANGE, ORANGE, GREY, GREY, BLUE]
    bs = ax.bar(labels, vals, color=cols, width=0.55)
    ax.errorbar(range(5), vals, yerr=[[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    for b, v in zip(bs, vals):
        ax.text(b.get_x() + b.get_width() * 0.82, v + 0.0008, f"+{v:.4f}", ha="left", fontsize=8.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Change in test ET Dice (95% CI)")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("The headline gain is far above seed noise and TTA effects", fontsize=10.5)
    save(fig, "fig_m3_noise")


def m3_finetune():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cohorts = ["BraTS-Africa (n = 60)", "Paediatric (n = 99)"]
    before = [0.7846, 0.4394]
    after = [0.8279, 0.8070]
    x = range(2)
    b1 = ax.bar([i - 0.18 for i in x], before, 0.36, color="#AAAAAA", label="Deployed model (frozen)")
    b2 = ax.bar([i + 0.18 for i in x], after, 0.36, color=GREEN, label="Cross-fitted fine-tune")
    bar_labels(ax, b1, dy=0.01)
    bar_labels(ax, b2, dy=0.01)
    for i, t in enumerate(["+0.0433", "+0.3676"]):
        ax.text(i, 0.93, t, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cohorts)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Tumour-core Dice")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_m3_finetune")


def m3_qc():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    cells = ["PED · TC", "SSA · ET", "PED · ET", "SSA · TC", "PED · WT"]
    d = [0.1686, 0.0375, 0.0290, -0.1951, -0.1898]
    lo = [0.0686, -0.1096, -0.0959, -0.3551, -0.3312]
    hi = [0.2747, 0.1758, 0.1556, -0.0429, -0.0667]
    cols = [GREEN, GREY, GREY, ORANGE, RED]
    ax.barh(cells, d, color=cols, height=0.5)
    ax.errorbar(d, range(5), xerr=[[v - l for v, l in zip(d, lo)], [h - v for v, h in zip(d, hi)]], fmt="none",
                ecolor="black", capsize=3, lw=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUROC, QC model minus free entropy (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Gate C: one win (green), one significant loss (red), one borderline (orange)", fontsize=10)
    save(fig, "fig_m3_qc")


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    for f in (evolution, unet, data_pipeline, training_system, boundary_bands, baseline_runs, calibration_m1,
              m2_results, m2_ablation, m2_uncertainty, m2_reliability, m2_cost, segqc, m3_noise, m3_finetune,
              m3_qc):
        f()
    print("done")
