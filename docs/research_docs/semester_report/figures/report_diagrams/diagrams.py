"""Coloured diagrams for the NeuroVision-X final project report.

All numbers are copied from docs/research_docs/claims_and_evidence.md and experiments.md.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
OUT = "figs"

BLUE = "#4C72B0"
LBLUE = "#DCE6F2"
ORANGE = "#DD8452"
LORANGE = "#FBE3D3"
GREEN = "#55A868"
LGREEN = "#DDEFE0"
RED = "#C44E52"
LRED = "#F6DADB"
PURPLE = "#8172B3"
LPURPLE = "#E6E2F3"
GREY = "#8C8C8C"
LGREY = "#EEEEEE"
YELLOW = "#CCB974"
LYELLOW = "#F6F0D6"


def box(ax, x, y, w, h, text, fc, ec, fs=10, bold=False, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.005,rounding_size={r}",
                       fc=fc, ec=ec, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)
    return p


def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.3, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1 architecture
def architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box(ax, 0.1, 2.5, 1.3, 1.2, "Input\n4 MRI sequences\nT1, T1CE,\nT2, FLAIR\n(4, 64, 64, 64)",
        LGREY, GREY, fs=9)

    # CNN encoder column
    cnn_levels = ["Stem  32 ch\nstride 1", "Level 1  64 ch\nstride 2", "Level 2  128 ch\nstride 4",
                  "Level 3  256 ch\nstride 8", "Level 4  320 ch\nstride 16"]
    swin_levels = ["Swin stage 1", "Swin stage 2", "Swin stage 3", "Swin stage 4"]
    ys = [5.1, 4.1, 3.1, 2.1, 1.1]
    ax.text(2.45, 5.95, "3D CNN encoder\n(18.9 M params)", ha="center", fontsize=10, fontweight="bold",
            color=BLUE)
    for i, (t, y) in enumerate(zip(cnn_levels, ys)):
        box(ax, 1.8, y, 1.3, 0.7, t, LBLUE, BLUE, fs=8.5)
        if i:
            arrow(ax, 2.45, ys[i - 1], 2.45, y + 0.7, color=BLUE)
    ax.text(5.15, 5.95, "Swin Transformer encoder\n(2.0 M params)", ha="center", fontsize=10,
            fontweight="bold", color=ORANGE)
    for i, (t, y) in enumerate(zip(swin_levels, ys[1:])):
        box(ax, 4.5, y, 1.3, 0.7, t, LORANGE, ORANGE, fs=8.5)
        if i:
            arrow(ax, 5.15, ys[i], 5.15, y + 0.7, color=ORANGE)

    arrow(ax, 1.4, 3.1, 1.8, 5.45)
    box(ax, 4.5, 4.97, 1.3, 0.33, "Patch embedding (input)", LORANGE, ORANGE, fs=7.5)
    arrow(ax, 5.15, 4.97, 5.15, 4.8, color=ORANGE)

    # fusion blocks
    ax.text(3.8, 0.35, "Adaptive gated cross-attention fusion\n(one block per Swin level, 1.0 M params)",
            ha="center", fontsize=9.5, color=PURPLE, fontweight="bold")
    for y in ys[1:]:
        box(ax, 3.3, y + 0.12, 0.95, 0.46, "Gate +\ncross-attn", LPURPLE, PURPLE, fs=7.5)
        arrow(ax, 3.1, y + 0.35, 3.3, y + 0.35, color=PURPLE)
        arrow(ax, 4.5, y + 0.35, 4.25, y + 0.35, color=PURPLE)

    # decoder
    ax.text(7.1, 5.95, "U-Net decoder\n(12.9 M params)", ha="center", fontsize=10, fontweight="bold",
            color=GREEN)
    for i, y in enumerate(ys):
        box(ax, 6.45, y, 1.3, 0.7, f"Decoder stage {4 - i}" if i < 4 else "Bottleneck", LGREEN, GREEN,
            fs=8.5)
        if i < 4:
            arrow(ax, 7.1, ys[i + 1] + 0.7, 7.1, y, color=GREEN)
    # skips
    for y in ys[:1]:
        arrow(ax, 3.1, y + 0.35, 6.45, y + 0.35, color=GREY, ls=":")
    for y in ys[1:]:
        ax.plot([3.78, 3.78, 6.25], [y + 0.12, y - 0.15, y - 0.15], color=GREY, ls=":", lw=1.1)
        arrow(ax, 6.25, y - 0.15, 6.45, y + 0.12, color=GREY)
    ax.text(4.6, 5.88, "", fontsize=8)
    ax.text(4.95, 5.48 - 0.05, "", fontsize=8)
    ax.text(5.6, 1.05 - 0.25, "dotted: fused skip connections", fontsize=8, color=GREY, ha="center")

    # heads
    heads = [("Segmentation head\nET / TC / WT logits\n(deep supervision)", LRED, RED, 4.6),
             ("Confidence head", LYELLOW, YELLOW, 3.5),
             ("Boundary head", LYELLOW, YELLOW, 2.6)]
    for t, fc, ec, y in heads:
        box(ax, 8.15, y, 1.5, 0.75 if "Seg" in t else 0.55, t, fc, ec, fs=8.5)
        arrow(ax, 7.75, 5.45, 8.15, y + 0.3, color="#555555")
    box(ax, 8.25, 1.2, 1.6, 0.9, "Output\n3 nested region masks\n+ voxel probabilities",
        LGREY, GREY, fs=8.5)
    ax.annotate("", xy=(9.6, 2.1), xytext=(9.65, 4.95),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, connectionstyle="arc3,rad=-0.35"))
    ax.text(5, -0.15, "Total: 34,911,341 parameters.  Gate input = [CNN feature, Swin feature, branch "
            "disagreement, entropy of each branch probe].", ha="center", fontsize=9, style="italic")
    save(fig, "fig_architecture")


# ---------------------------------------------------------------- 2 fusion block
def fusion():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    box(ax, 0.1, 2.5, 1.5, 0.7, "CNN feature\n$F_c$", LBLUE, BLUE, fs=10)
    box(ax, 0.1, 0.5, 1.5, 0.7, "Swin feature\n$F_s$", LORANGE, ORANGE, fs=10)
    box(ax, 2.1, 0.5, 1.5, 0.7, "1x1x1 width\nadapter", LORANGE, ORANGE, fs=9)
    box(ax, 1.9, 1.3, 1.5, 0.6, "Branch probes:\n|p_c - p_s|, H(p_c), H(p_s)", LGREY, GREY,
        fs=7.5)
    box(ax, 4.1, 2.4, 1.6, 0.8, "Windowed\ncross-attention\n(Q from CNN)", LPURPLE, PURPLE, fs=9)
    box(ax, 4.1, 1.0, 1.6, 0.8, "Gate generator\nsigmoid gate g", LPURPLE, PURPLE, fs=9)
    box(ax, 6.3, 1.7, 2.5, 0.9, "Merge\n$F_c$ + layer_scale * g * attn", LGREEN, GREEN, fs=10)
    arrow(ax, 1.6, 0.85, 2.1, 0.85)
    arrow(ax, 1.6, 2.85, 4.1, 2.85)
    arrow(ax, 3.6, 0.85, 4.1, 2.5)
    arrow(ax, 3.6, 0.85, 4.1, 1.2)
    arrow(ax, 1.6, 2.6, 4.1, 1.75)
    arrow(ax, 3.4, 1.6, 4.1, 1.35)
    arrow(ax, 5.7, 2.8, 6.3, 2.3)
    arrow(ax, 5.7, 1.4, 6.3, 2.0)
    ax.text(4.5, 0.1, "Ablation 'content-only gate' removes the grey disagreement inputs "
            "(parameter-matched to 0.018%).", ha="center", fontsize=9, style="italic")
    save(fig, "fig_fusion")


# ---------------------------------------------------------------- 3 clinical pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    steps = [("DICOM study\n(zip upload)", LGREY, GREY),
             ("E1 Ingest\ndcm2niix, series\nrole assignment,\nPatientAge", LBLUE, BLUE),
             ("E3 Input QC\n(pre-registration)\nlabel-free checks", LYELLOW, YELLOW),
             ("E2 Co-register to T1CE,\nSRI24 atlas,\nHD-BET skull strip", LBLUE, BLUE),
             ("E4 Input QC\n(post-registration)", LYELLOW, YELLOW)]
    x = 0.1
    xs = []
    for t, fc, ec in steps:
        box(ax, x, 3.6, 1.75, 1.1, t, fc, ec, fs=8.5)
        xs.append(x)
        x += 2.0
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + 1.75, 4.15, xs[i + 1], 4.15)
    # second row
    box(ax, 7.9, 1.9, 2.0, 1.1, "Segmentation\nNeuroVision-X seed 42\nsliding window 64³", LGREEN,
        GREEN, fs=8.5)
    arrow(ax, 8.9, 3.6, 8.9, 3.0)
    box(ax, 5.2, 1.9, 2.2, 1.1, "Signals\nQC model predicted Dice\nintended use (adults)\n"
        "conformal band, OOD score\n(display only)", LPURPLE, PURPLE, fs=8)
    arrow(ax, 7.9, 2.45, 7.4, 2.45)
    box(ax, 2.6, 1.9, 2.1, 1.1, "E5 Gatekeeper\nrefusal gate\n(thresholds fit on val)", LORANGE,
        ORANGE, fs=8.5, bold=True)
    arrow(ax, 5.2, 2.45, 4.7, 2.45)
    outs = [("PROCEED", LGREEN, GREEN), ("CAUTION", LYELLOW, YELLOW), ("REFUSE", LRED, RED)]
    for i, (t, fc, ec) in enumerate(outs):
        box(ax, 0.1, 2.85 - i * 0.75, 1.6, 0.55, t, fc, ec, fs=9.5, bold=True)
        arrow(ax, 2.6, 2.45, 1.7, 3.12 - i * 0.75)
    box(ax, 2.6, 0.2, 7.3, 1.0, "Outputs: 3D digital twin with atlas structure shells, uncertainty and "
        "Grad-CAM layers;\nstructured anatomical report (JSON + Markdown); DICOM-SEG export; "
        "zip bundle.\nA refused job is a successful outcome, never an error.", LGREY, GREY, fs=8.5)
    arrow(ax, 8.9, 1.9, 8.9, 1.2)
    ax.text(5.0, 3.42, "fails: REFUSE early (e.g. thick slices)", fontsize=8, color=RED, ha="center")
    ax.text(9.0, 3.42, "", fontsize=8)
    save(fig, "fig_pipeline")


# ---------------------------------------------------------------- 4 data
def data():
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    box(ax, 0.1, 1.4, 2.2, 1.1, "BraTS 2021\ntraining set\n1,251 adult glioma\ncases", LBLUE, BLUE,
        fs=10, bold=True)
    split = [("Train  875 (70%)\nfits every model", 2.75), ("Val  187 (15%)\ncheckpoint selection,\n"
             "calibration, gate thresholds", 1.5), ("Test  189 (15%)\nheld out, reported", 0.25)]
    for t, y in split:
        box(ax, 3.0, y, 2.4, 0.95, t, LGREEN, GREEN, fs=9)
        arrow(ax, 2.3, 1.95, 3.0, y + 0.47)
    ax.text(4.2, 3.78, "Frozen seeded split", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    ax.text(8.0, 3.78, "External cohorts (never trained on)", ha="center", fontsize=10,
            fontweight="bold", color=ORANGE)
    ext = [("BraTS-Africa 2023 (SSA)\n60 adult glioma cases\nlower image quality,\nvaried scanners", 2.75),
           ("BraTS-PEDs 2023 (PED)\n99 paediatric tumours\ndifferent disease + label use", 1.5),
           ("RSNA-MICCAI DICOM\n40 test-split patients\nreal clinical DICOM", 0.25)]
    for t, y in ext:
        box(ax, 6.6, y, 2.8, 0.95, t, LORANGE, ORANGE, fs=9)
    ax.annotate("", xy=(6.6, 0.72), xytext=(5.4, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=GREY, ls="--"))
    ax.text(6.0, 0.95, "same\npatients", fontsize=8, color=GREY, ha="center")
    save(fig, "fig_data")


# ---------------------------------------------------------------- 5 timeline / identities
def timeline():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.plot([0.3, 9.7], [1.9, 1.9], color="#444444", lw=2)
    events = [(0.4, "31 Jul", "Repo, data,\ntrainer"), (1.4, "16 Aug", "Model trained:\n+0.027 ET,\nnot more reliable"),
              (2.4, "19 Aug", "External cohorts:\ngain does not\ntransfer"), (3.4, "23 Aug", "P2 null;\npivot to safety\npipeline"),
              (4.4, "24 Aug", "Conformal bound:\nholds in dist.,\nbreaks under shift"),
              (5.4, "27 Aug", "Clinical DICOM\npipeline built"), (6.4, "19 Sep", "Demo; end-to-end\nerror budget"),
              (7.4, "27 Sep", "Local recalibration;\nreal DICOM test"), (8.4, "4 Oct", "Fine-tuning\nrecovers TC"),
              (9.4, "12 Oct", "nnU-Net wins:\narchitecture claim\nwithdrawn")]
    for i, (x, d, t) in enumerate(events):
        ax.plot(x, 1.9, "o", ms=9, color=[BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN, GREEN, PURPLE, PURPLE,
                                          RED][i])
        ax.text(x, 2.15, d, ha="center", fontsize=9, fontweight="bold")
        ax.text(x, 1.6 if i % 2 == 0 else 2.5, t, ha="center", va="top" if i % 2 == 0 else "bottom",
                fontsize=7.8)
    bands = [(0.3, 2.0, "Identity 1:\nmore reliable segmenter", LBLUE),
             (2.0, 3.9, "Identity 2: interpretable\npipeline, failure detector", LORANGE),
             (3.9, 9.7, "Identity 3: error bound + refusal gate, measured where it breaks", LGREEN)]
    for a, b, t, c in bands:
        ax.add_patch(FancyBboxPatch((a, 0.05), b - a - 0.05, 0.55, boxstyle="round,pad=0.01", fc=c,
                                    ec="#999999"))
        ax.text((a + b) / 2, 0.32, t, ha="center", va="center", fontsize=7.8)
    ax.text(5, 3.45, "Project timeline, 2026", ha="center", fontsize=11, fontweight="bold")
    save(fig, "fig_timeline")


# ---------------------------------------------------------------- 6 conformal concept
def conformal():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    box(ax, 0.1, 1.2, 2.0, 1.2, "Calibration set\n(BraTS val, 187 cases)\nvoxel probabilities\n+ labels",
        LBLUE, BLUE, fs=9)
    box(ax, 2.6, 1.2, 2.4, 1.2, "For each threshold λ:\nmiss rate = fraction of\ntumour voxels "
        "outside\nthe mask", LPURPLE, PURPLE, fs=9)
    box(ax, 5.5, 1.2, 2.0, 1.2, "Pick the largest λ\nwith adjusted risk\n(n R + 1)/(n + 1) ≤ α",
        LPURPLE, PURPLE, fs=9)
    box(ax, 8.0, 1.2, 1.9, 1.2, "Freeze λ.\nApply to test,\nSSA, PED.\nMeasure realised risk",
        LGREEN, GREEN, fs=9)
    for a, b in [(2.1, 2.6), (5.0, 5.5), (7.5, 8.0)]:
        arrow(ax, a, 1.8, b, 1.8)
    ax.text(5, 0.55, "Guarantee: E[miss rate] ≤ α on average over new studies, IF they are "
            "exchangeable with the calibration set.", ha="center", fontsize=9.5, style="italic")
    ax.text(5, 0.15, "Distribution shift breaks exchangeability, which is exactly what this project "
            "measures.", ha="center", fontsize=9.5, style="italic", color=RED)
    ax.text(5, 2.85, "Conformal risk control (Angelopoulos et al., ICLR 2024) applied to the mask's miss rate",
            ha="center", fontsize=10.5, fontweight="bold")
    save(fig, "fig_conformal")


# ---------------------------------------------------------------- 7 decomposition waterfall
def decomposition():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    steps = [("Matched U-Net\n(12.9 M)", 0.8442, None), ("+ width\n(capacity\ncontrol)", 0.0055, BLUE),
             ("+ gated\ncross-attention\nfusion", 0.0189, GREEN),
             ("+ disagreement\nconditioning", 0.0022, GREY)]
    base = 0.8442
    ax.bar(0, base, color="#BBBBBB", width=0.6)
    ax.text(0, base + 0.001, f"{base:.4f}", ha="center", va="bottom", fontsize=9)
    cur = base
    for i, (lab, v, c) in enumerate(steps[1:], start=1):
        ax.bar(i, v, bottom=cur, color=c, width=0.6)
        ax.text(i, cur + v + 0.001, f"+{v:.4f}", ha="center", va="bottom", fontsize=9)
        cur += v
    ax.bar(4, cur, color=RED, width=0.6)
    ax.text(4, cur + 0.001, "0.8709", ha="center", va="bottom", fontsize=9)
    ax.bar(5, 0.8879, color=ORANGE, width=0.6)
    ax.text(5, 0.8889, "0.8879", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(6))
    ax.set_xticklabels([s[0] for s in steps] + ["NeuroVision-X\n(34.9 M)", "nnU-Net v2\n(1000 ep.)"],
                       fontsize=8.5)
    ax.set_ylim(0.83, 0.895)
    ax.set_ylabel("Test Dice, enhancing tumour (n = 189)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Where the +0.0267 ET gain comes from (grey step: not significant)", fontsize=10.5)
    save(fig, "fig_decomposition")


# ---------------------------------------------------------------- 8 OOD dice
def ood():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cohorts = ["BraTS test\n(n = 189)", "BraTS-Africa\n(n = 60)", "Paediatric\n(n = 99)"]
    vals = {"ET": [0.8709, 0.7784, 0.5634], "TC": [0.9161, 0.7846, 0.4394], "WT": [0.9321, 0.8959, 0.8490]}
    cols = {"ET": RED, "TC": ORANGE, "WT": GREEN}
    w = 0.26
    for j, (k, v) in enumerate(vals.items()):
        xs = [i + (j - 1) * w for i in range(3)]
        ax.bar(xs, v, width=w, color=cols[k], label=k)
        for x, y in zip(xs, v):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cohorts)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean voxel Dice")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("NeuroVision-X accuracy falls with distance from the training data", fontsize=10.5)
    save(fig, "fig_ood")


# ---------------------------------------------------------------- 9 software architecture
def software():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    cols = [("Research path (.venv)", LBLUE, BLUE,
             ["data/  preprocessing, splits", "models/  registry: unet3d,\nswinunetr, neurovision, segqc",
              "training/  trainer, resume", "inference/  sliding window,\nMC-dropout, TTA",
              "metrics/ + analysis/\nstatistics, conformal, OOD"]),
            ("Clinical path (.venv-clinical)", LORANGE, ORANGE,
             ["DICOM ingest (dcm2niix)", "brainles-preprocessing:\nANTs + SRI24 + HD-BET",
              "input QC (12 checks)", "gatekeeper\nPROCEED / CAUTION / REFUSE", "DICOM-SEG export"]),
            ("Interpretable layer", LGREEN, GREEN,
             ["SRI24 atlas, 122 structures", "localisation, burden,\ninvolvement", "eloquence map\n(knowledge/)",
              "structured report\nJSON + Markdown", "population statistics"]),
            ("Web application (app/)", LPURPLE, PURPLE,
             ["FastAPI backend", "React + TypeScript\n+ Tailwind frontend", "research slice viewer",
              "/clinical upload page", "3D digital twin"])]
    for i, (title, fc, ec, items) in enumerate(cols):
        x = 0.1 + i * 2.5
        ax.add_patch(FancyBboxPatch((x, 0.1), 2.3, 3.95, boxstyle="round,pad=0.01", fc="white", ec=ec,
                                    lw=1.6))
        ax.text(x + 1.15, 4.2, title, ha="center", fontsize=9.5, fontweight="bold", color=ec)
        for j, t in enumerate(items):
            box(ax, x + 0.1, 3.3 - j * 0.78, 2.1, 0.66, t, fc, ec, fs=8)
    ax.text(5, -0.15, "Shared: Hydra configuration (configs/), registries selected by string, "
            "pytest suite (~2,670 CPU tests), Weights & Biases tracking.", ha="center", fontsize=9,
            style="italic")
    save(fig, "fig_software")


if __name__ == "__main__":
    import os

    os.makedirs(OUT, exist_ok=True)
    for f in (architecture, fusion, pipeline, data, timeline, conformal, decomposition, ood, software):
        f()
    print("done")
