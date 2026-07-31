# NeuroVision-X

3D brain tumor segmentation on BraTS multi-modal MRI (T1, T1CE, T2, FLAIR).

**The research claim is reliability, not raw accuracy** — competitive Dice with
substantially better calibration and boundary accuracy.

## Architecture

Dual encoder (3D CNN + Swin Transformer) → adaptive gated cross-attention fusion →
U-Net decoder → three heads (segmentation, confidence, boundary). Plus MC-dropout
uncertainty, calibration analysis, and explainability.

## Setup

Requires Python 3.11.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

On Kaggle, skip `torch` / `torchvision` — the notebook image already provides a
CUDA-matched build. See the comment at the top of `requirements.txt`.

## Development

```bash
pytest          # full suite: CPU only, under ~60s
ruff check .
black .
```

Tests run on CPU. The Mac is a correctness harness; gradient descent happens on Kaggle.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Hydra configs — data, model, training, experiment |
| `src/neurovision/` | Library code |
| `scripts/` | CLI entry points (preprocess, train, evaluate) |
| `notebooks/` | Analysis and the thin Kaggle driver notebook |
| `tests/` | pytest suite |
| `docs/` | MkDocs sources and the experiment log |
| `outputs/` | Run artifacts (gitignored) |

## Status

Phase 0 — repository scaffolding. No functional code yet.
