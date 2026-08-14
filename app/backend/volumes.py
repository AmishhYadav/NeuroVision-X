"""Loads cases and turns volumes into browser-transportable bytes.

Everything served to the browser is in the CROPPED preprocessed frame, so
image, ground-truth label, prediction and uncertainty are all indexed by the
same voxel coordinates. Saved predictions are stored in ORIGINAL BraTS
geometry (240x240x155), so they are re-cropped here with the same bbox
`meta.json` records. Skipping that step misaligns the overlay by the crop
offset and still looks entirely plausible -- the geometry trap described in
`CLAUDE.md`.

Volumes go over the wire as raw uint8 in C order `(D, H, W)`, so the frontend
can slice any plane instantly from a single fetch rather than making one
request per slice.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import Settings, get_settings

MODALITIES = ("t1", "t1ce", "t2", "flair")
MODALITY_INDEX = {name: i for i, name in enumerate(MODALITIES)}
REGION_NAMES = ("ET", "TC", "WT")

MaskSource = Literal["prediction", "label"]


@dataclass(frozen=True)
class CaseMeta:
    """Geometry and provenance for one case."""

    case_id: str
    shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    bbox: tuple[tuple[int, int], ...]
    spacing: tuple[float, float, float]
    has_label: bool
    has_prediction: bool
    has_logits: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "shape": list(self.shape),
            "original_shape": list(self.original_shape),
            "bbox": [list(b) for b in self.bbox],
            "spacing": list(self.spacing),
            "has_label": self.has_label,
            "has_prediction": self.has_prediction,
            "has_logits": self.has_logits,
            # Axis 2 is the axial (superior-inferior) axis in BraTS geometry.
            "planes": {
                "sagittal": self.shape[0],
                "coronal": self.shape[1],
                "axial": self.shape[2],
            },
        }


def case_dir(case_id: str, settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    return s.prep_dir / case_id


def read_meta(case_id: str, settings: Settings | None = None) -> CaseMeta:
    """Reads one case's `meta.json` and notes which artifacts exist for it."""
    s = settings or get_settings()
    meta_path = case_dir(case_id, s) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no preprocessed case at {meta_path}")
    raw = json.loads(meta_path.read_text())
    shape = tuple(int(v) for v in raw["cropped_shape"])
    return CaseMeta(
        case_id=case_id,
        shape=shape,  # type: ignore[arg-type]
        original_shape=tuple(int(v) for v in raw["original_shape"]),  # type: ignore[arg-type]
        bbox=tuple(tuple(int(v) for v in pair) for pair in raw["bbox"]),  # type: ignore[arg-type]
        spacing=tuple(float(v) for v in raw.get("spacing", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        has_label=(case_dir(case_id, s) / "label.npy").exists(),
        has_prediction=(s.predictions_dir / f"{case_id}.npy").exists(),
        has_logits=(s.logits_dir / f"{case_id}.npy").exists(),
    )


@lru_cache(maxsize=1)
def _metrics_table() -> pd.DataFrame | None:
    """Loads per-case metrics once, or returns None when no evaluation exists."""
    s = get_settings()
    if not s.metrics_csv.exists():
        return None
    df = pd.read_csv(s.metrics_csv)
    return df.set_index("case_id")


def case_metrics(case_id: str) -> dict[str, Any] | None:
    """Returns the reported Dice/HD95 row for a case, if it was evaluated.

    These come from `scripts/evaluate.py` at the reported overlap of 0.5, so
    they describe the SAVED prediction. A live-inference run at the demo's
    lower overlap is a different prediction and does not inherit them.
    """
    table = _metrics_table()
    if table is None or case_id not in table.index:
        return None
    row = table.loc[case_id]

    def _num(key: str) -> float | None:
        if key not in row:
            return None
        value = float(row[key])
        return None if math.isnan(value) else value

    return {
        "dice": {r: _num(f"dice_{r}") for r in REGION_NAMES},
        "hd95": {r: _num(f"hd95_{r}") for r in REGION_NAMES},
        "dice_mean": _num("dice_mean"),
        "gt_empty": {r: bool(row.get(f"gt_empty_{r}", 0)) for r in REGION_NAMES},
    }


def list_cases(settings: Settings | None = None) -> list[str]:
    """Lists cases that have both a preprocessed volume and a saved prediction.

    Sorted by descending mean Dice so the case picker opens on results worth
    looking at, with the harder cases still reachable further down.
    """
    s = settings or get_settings()
    if not s.predictions_dir.exists():
        return []
    ids = sorted(p.stem for p in s.predictions_dir.glob("*.npy"))
    ids = [c for c in ids if (s.prep_dir / c / "image.npy").exists()]

    table = _metrics_table()
    if table is not None:
        ranked = [c for c in ids if c in table.index]
        ranked.sort(key=lambda c: float(table.loc[c, "dice_mean"]), reverse=True)
        ids = ranked + [c for c in ids if c not in table.index]
    return ids[: s.max_cases]


def _window_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Maps a z-scored modality to 0-255 for display.

    Preprocessing z-scores each modality over its NONZERO voxels, so brain
    interiors are routinely negative and the air outside the head is exactly
    0. A plain min-max would therefore render air as mid-grey and put every
    brain on a grey card. Percentile windowing over the nonzero voxels fixes
    the contrast, and exact zeros are forced back to black afterwards.
    """
    out = np.zeros(volume.shape, dtype=np.uint8)
    brain = volume != 0
    if not brain.any():
        return out
    lo, hi = np.percentile(volume[brain], (1.0, 99.0))
    if hi <= lo:
        out[brain] = 255
        return out
    scaled = (volume - lo) / (hi - lo)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    out[:] = (scaled * 255.0).astype(np.uint8)
    out[~brain] = 0
    return out


def load_modality(case_id: str, modality: str, settings: Settings | None = None) -> bytes:
    """Returns one modality as display-windowed uint8 bytes, shape `(D, H, W)`."""
    if modality not in MODALITY_INDEX:
        raise KeyError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
    s = settings or get_settings()
    arr = np.load(case_dir(case_id, s) / "image.npy", mmap_mode="r")
    channel = np.asarray(arr[MODALITY_INDEX[modality]], dtype=np.float32)
    return _window_to_uint8(channel).tobytes()


def _crop_to_meta(volume: np.ndarray, meta: CaseMeta) -> np.ndarray:
    """Crops an original-geometry volume into the preprocessed frame."""
    (d0, d1), (h0, h1), (w0, w1) = meta.bbox
    cropped = volume[d0:d1, h0:h1, w0:w1]
    if cropped.shape != meta.shape:
        raise ValueError(
            f"crop produced {cropped.shape}, expected {meta.shape} -- prediction and "
            "meta.json come from different preprocessing runs"
        )
    return cropped


def load_mask(case_id: str, source: MaskSource, settings: Settings | None = None) -> bytes:
    """Returns a `{0,1,2,3}` class map in the cropped frame as uint8 bytes.

    `label` is already cropped on disk. `prediction` is stored in original
    BraTS geometry and is re-cropped here with the case's own bbox.
    """
    s = settings or get_settings()
    meta = read_meta(case_id, s)
    if source == "label":
        arr = np.load(case_dir(case_id, s) / "label.npy", mmap_mode="r")
        return np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    if source == "prediction":
        arr = np.asarray(
            np.load(s.predictions_dir / f"{case_id}.npy", mmap_mode="r"), dtype=np.uint8
        )
        return np.ascontiguousarray(_crop_to_meta(arr, meta)).tobytes()
    raise KeyError(f"unknown mask source {source!r}")


def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    """Per-voxel predictive entropy of a single deterministic pass, in [0, 1].

    The three heads are independent sigmoids over nested regions, so this is
    a per-channel BERNOULLI entropy summed over channels, never a softmax
    entropy. Computed from logits via softplus rather than from probabilities:
    `log(1 - p)` underflows the moment a confident probability rounds to 1.0,
    which is exactly what happens in fp16 (see the entropy lesson in
    `CLAUDE.md`).

    This is aleatoric + epistemic combined and contains NO separable epistemic
    term -- that needs MC-dropout. The UI labels it accordingly.
    """
    z = logits.astype(np.float32)
    # softplus(x) = log1p(exp(-|x|)) + max(x, 0), the overflow-safe form.
    sp_neg = np.log1p(np.exp(-np.abs(z))) + np.maximum(-z, 0.0)  # softplus(-z)
    sp_pos = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)  # softplus(z)
    p = 1.0 / (1.0 + np.exp(-z))
    per_channel = p * sp_neg + (1.0 - p) * sp_pos  # nats, max ln 2 per channel
    total = per_channel.sum(axis=0) / (logits.shape[0] * math.log(2.0))
    return np.clip(total, 0.0, 1.0)


def load_uncertainty(case_id: str, settings: Settings | None = None) -> bytes:
    """Returns per-voxel entropy scaled to uint8 in the cropped frame."""
    s = settings or get_settings()
    path = s.logits_dir / f"{case_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"no saved logits for {case_id}")
    logits = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    field = entropy_from_logits(logits)
    return (field * 255.0).astype(np.uint8).tobytes()


def region_voxel_counts(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """Counts voxels and millilitres per nested region from a class map.

    Classes are `{1: necrotic core, 2: oedema, 3: enhancing}`. Regions are
    nested, so WT is every non-zero class, TC is necrotic plus enhancing, and
    ET is enhancing alone -- the same definition `ConvertToRegionsd` uses.
    """
    voxel_ml = float(np.prod(spacing)) / 1000.0
    counts = {
        "ET": int((mask == 3).sum()),
        "TC": int(((mask == 1) | (mask == 3)).sum()),
        "WT": int((mask > 0).sum()),
    }
    return {region: {"voxels": n, "ml": round(n * voxel_ml, 2)} for region, n in counts.items()}
