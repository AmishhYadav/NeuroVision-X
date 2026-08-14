"""Path and runtime settings for the demo backend.

Every path is resolved from an environment variable with a repo-relative
default, never hardcoded to one machine -- the same rule the training code
follows (`CLAUDE.md` constraint 2). Nothing here reaches into Hydra: the
Hydra config is composed lazily inside `inference.py`, so the API can serve
precomputed cases even on a machine with no checkpoint present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# app/backend/config.py -> app/backend -> app -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_env(name: str, default: str) -> Path:
    """Reads a path from the environment, resolving relative values to the repo root."""
    raw = os.environ.get(name, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for one server process.

    Attributes:
        prep_dir: Preprocessed case root, one directory per case holding
            `image.npy` / `label.npy` / `meta.json`.
        eval_dir: An evaluation output directory holding `predictions/`,
            `per_case_metrics.csv` and optionally `logits/`. Supplies the
            precomputed results the demo serves instantly.
        checkpoint: Checkpoint used for live inference.
        experiment: Hydra experiment name whose config the checkpoint was
            trained under.
        cache_dir: Where live-inference results are written so a case is
            only ever segmented once.
        max_cases: Upper bound on how many cases the case list exposes.
        demo_overlap: Sliding-window overlap for LIVE inference. Lower than
            the 0.5 used for reported metrics: measured on an M4 CPU, 0.5
            takes ~153 s per case against ~66 s at 0.25, and the visual
            difference is imperceptible. The demo is a viewer, not a
            measuring instrument -- reported numbers still come from
            `scripts/evaluate.py` at 0.5.
    """

    prep_dir: Path
    eval_dir: Path
    checkpoint: Path
    experiment: str
    cache_dir: Path
    max_cases: int
    demo_overlap: float

    @property
    def predictions_dir(self) -> Path:
        return self.eval_dir / "predictions"

    @property
    def metrics_csv(self) -> Path:
        return self.eval_dir / "per_case_metrics.csv"

    @property
    def logits_dir(self) -> Path:
        return self.eval_dir / "logits"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Builds the settings once per process."""
    settings = Settings(
        prep_dir=_path_env("NVX_PREP_DIR", "data/preprocessed/brats"),
        eval_dir=_path_env("NVX_EVAL_DIR", "outputs/eval_test_baseline_unet3d"),
        checkpoint=_path_env("NVX_CHECKPOINT", "outputs/checkpoints/baseline_unet3d/best.pt"),
        experiment=os.environ.get("NVX_EXPERIMENT", "baseline_unet3d"),
        cache_dir=_path_env("NVX_CACHE_DIR", "outputs/demo_cache"),
        # 200 covers the whole 189-case test split. Listing them costs one
        # meta.json read each (~90 ms measured), and the case list is the only
        # way to reach a hard case -- capping below the split size hides
        # exactly the failures the demo exists to show.
        max_cases=int(os.environ.get("NVX_MAX_CASES", "200")),
        demo_overlap=float(os.environ.get("NVX_DEMO_OVERLAP", "0.25")),
    )
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
