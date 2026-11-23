"""Patient-level leakage check (MasterPlan W2.7).

Finds byte-identical and near-duplicate cases ACROSS the frozen train/val/test
splits, from the preprocessed arrays. It ranks candidate pairs for a human to look
at; it never decides on its own. CPU only, no model, no torch.

How near-duplicates are found: every case is placed back into the common atlas
frame (using the crop bbox in meta.json), block-averaged down to a tiny grid, and
turned into a mean-centred unit vector. The dot product of two such vectors is the
Pearson correlation of the two coarse images. A second vector built from the
tumour footprint (label > 0) says whether the tumours also sit in the same place.

Run:
    python scripts/leakage_check.py
    python scripts/leakage_check.py analysis.leakage.out_dir=outputs/leakage/ped ...
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from neurovision.data.dataset import load_splits
from neurovision.utils.io import ensure_dir, read_json, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file so it works from any cwd (same pattern as other scripts).
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
_PAIR_COLUMNS = [
    "kind",
    "case_a",
    "split_a",
    "case_b",
    "split_b",
    "image_score",
    "tumour_score",
]


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, streamed so big files are never fully in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def block_mean(
    volume: np.ndarray, bbox: list[list[int]], full_shape: tuple[int, ...], grid: tuple[int, ...]
) -> np.ndarray:
    """Place a cropped 3D volume into the full frame and block-average it.

    Args:
        volume: Cropped array `(D, H, W)` (may be a memmap, any dtype).
        bbox: `[[lo, hi], ...]` of the crop inside the original grid.
        full_shape: Original `(D, H, W)` shape.
        grid: Coarse grid `(gD, gH, gW)` to reduce to.

    Returns:
        float32 array of shape `grid`. Voxels outside the crop count as zero.
    """
    # Pad the frame up to a multiple of the grid so blocks are equal-sized.
    padded = tuple(int(math.ceil(s / g) * g) for s, g in zip(full_shape, grid))
    frame = np.zeros(padded, dtype=np.float32)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in bbox)
    frame[slices] = np.asarray(volume, dtype=np.float32)
    block = tuple(p // g for p, g in zip(padded, grid))
    shaped = frame.reshape(grid[0], block[0], grid[1], block[1], grid[2], block[2])
    return shaped.mean(axis=(1, 3, 5), dtype=np.float32)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    """Subtract the mean and L2-normalise. A constant vector becomes all zeros (no NaN)."""
    centred = vec.astype(np.float32) - np.float32(vec.mean())
    norm = float(np.linalg.norm(centred))
    if norm < 1e-8:
        return np.zeros_like(centred)
    return centred / np.float32(norm)


def case_descriptors(
    case_dir: Path, grid: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Coarse image and tumour-footprint unit vectors for one case.

    Args:
        case_dir: Folder with image.npy `(4, D, H, W)`, label.npy `(D, H, W)`, meta.json.
        grid: Coarse grid `(gD, gH, gW)`.

    Returns:
        `(image_vec, tumour_vec, label_empty)`. `image_vec` has 4 * prod(grid)
        entries; `tumour_vec` has prod(grid) and is all zeros if the label is empty.
    """
    meta = read_json(case_dir / "meta.json")
    full_shape = tuple(int(s) for s in meta["original_shape"])
    bbox = meta["bbox"]
    image = np.load(case_dir / "image.npy", mmap_mode="r")
    # One channel at a time keeps peak memory at a single padded float32 volume.
    coarse = [block_mean(image[c], bbox, full_shape, grid).ravel() for c in range(image.shape[0])]
    image_vec = unit_vector(np.concatenate(coarse))

    label = np.load(case_dir / "label.npy", mmap_mode="r")
    footprint = np.asarray(label > 0, dtype=np.float32)
    label_empty = not bool(footprint.any())
    tumour_vec = unit_vector(block_mean(footprint, bbox, full_shape, grid).ravel())
    return image_vec, tumour_vec, label_empty


def exact_duplicate_groups(
    hashes: dict[str, str], splits_of: dict[str, str], what: str
) -> list[dict[str, Any]]:
    """Group cases sharing a hash; keep groups of size >= 2.

    Args:
        hashes: case id -> hex digest.
        splits_of: case id -> split name.
        what: `"image"` or `"label"` (which file was hashed).
    """
    by_hash: dict[str, list[str]] = {}
    for cid, h in hashes.items():
        by_hash.setdefault(h, []).append(cid)
    rows = []
    for h, ids in by_hash.items():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        sp = [splits_of[i] for i in ids]
        rows.append(
            {
                "hash_kind": what,
                "sha256": h,
                "case_ids": ";".join(ids),
                "splits": ";".join(sp),
                "cross_split": len(set(sp)) > 1,
            }
        )
    return rows


def _pair_rows(
    kind: str,
    idx: np.ndarray,
    ids: list[str],
    split_names: list[str],
    ii: np.ndarray,
    jj: np.ndarray,
    img: np.ndarray,
    tum: np.ndarray,
) -> list[dict[str, Any]]:
    """Turn selected pair positions into report rows."""
    return [
        {
            "kind": kind,
            "case_a": ids[ii[p]],
            "split_a": split_names[ii[p]],
            "case_b": ids[jj[p]],
            "split_b": split_names[jj[p]],
            "image_score": float(img[p]),
            "tumour_score": float(tum[p]),
        }
        for p in idx
    ]


def top_indices(scores: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores among entries where `mask` is True, best first."""
    cand = np.flatnonzero(mask)
    if cand.size == 0:
        return cand
    order = np.argsort(-scores[cand], kind="stable")
    return cand[order[:k]]


def run(
    prep_dir: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    grid: tuple[int, int, int] = (24, 24, 16),
    top_k: int = 50,
    flag_threshold: float = 0.99,
    flag_tumour_threshold: float = 0.9,
) -> dict[str, Any]:
    """Run the leakage scan and write the four report files.

    Args:
        prep_dir: Folder of `<case_id>/{image.npy,label.npy,meta.json}`.
        splits_path: Split YAML (keys train/val/test).
        out_dir: Where pairs_top.csv, exact_duplicates.csv, flagged.csv, summary.json go.
        grid: Coarse grid for descriptors.
        top_k: Pairs kept per kind (cross / within) in pairs_top.csv.
        flag_threshold: Minimum image score to flag a cross-split pair.
        flag_tumour_threshold: Minimum tumour score to flag a cross-split pair.

    Returns:
        The summary dict (also written to summary.json).

    Raises:
        FileNotFoundError: If split cases are missing on disk.
    """
    prep_dir, out_dir = Path(prep_dir), ensure_dir(out_dir)
    grid = tuple(int(g) for g in grid)  # type: ignore[assignment]
    splits = load_splits(splits_path)

    # A case listed under two split keys is a leak in itself; report, keep the first.
    seen: dict[str, list[str]] = {}
    for name, case_ids in splits.items():
        for cid in case_ids:
            seen.setdefault(cid, [])
            if name not in seen[cid]:
                seen[cid].append(name)
    ids = list(seen)
    split_names = [seen[c][0] for c in ids]
    splits_of = dict(zip(ids, split_names))
    split_overlap = {c: s for c, s in seen.items() if len(s) > 1}

    missing = [c for c in ids if not (prep_dir / c / "image.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} split case(s) missing under {prep_dir}: {missing[:10]}"
        )

    image_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    img_vecs: list[np.ndarray] = []
    tum_vecs: list[np.ndarray] = []
    empty_labels: list[str] = []
    for n, cid in enumerate(ids, 1):
        case_dir = prep_dir / cid
        image_hashes[cid] = file_sha256(case_dir / "image.npy")
        label_hashes[cid] = file_sha256(case_dir / "label.npy")
        iv, tv, empty = case_descriptors(case_dir, grid)
        img_vecs.append(iv)
        tum_vecs.append(tv)
        if empty:
            empty_labels.append(cid)
        if n % 100 == 0 or n == len(ids):
            logger.info("leakage: described %d / %d cases", n, len(ids))

    img_mat = np.stack(img_vecs).astype(np.float32)
    tum_mat = np.stack(tum_vecs).astype(np.float32)
    ii, jj = np.triu_indices(len(ids), k=1)  # every pair once (i < j)
    # Full N x N similarity (N ~ 1251, float32 -> a few MB); keep the i < j half.
    img = (img_mat @ img_mat.T)[ii, jj]
    tum = (tum_mat @ tum_mat.T)[ii, jj]
    img, tum = np.nan_to_num(img), np.nan_to_num(tum)

    sp = np.array(split_names)
    cross = sp[ii] != sp[jj]
    rows = _pair_rows("cross", top_indices(img, cross, top_k), ids, split_names, ii, jj, img, tum)
    rows += _pair_rows(
        "within", top_indices(img, ~cross, top_k), ids, split_names, ii, jj, img, tum
    )
    pd.DataFrame(rows, columns=_PAIR_COLUMNS).to_csv(out_dir / "pairs_top.csv", index=False)

    dup_rows = exact_duplicate_groups(image_hashes, splits_of, "image")
    dup_rows += exact_duplicate_groups(label_hashes, splits_of, "label")
    dup_cols = ["hash_kind", "sha256", "case_ids", "splits", "cross_split"]
    pd.DataFrame(dup_rows, columns=dup_cols).to_csv(out_dir / "exact_duplicates.csv", index=False)

    flag_mask = cross & (img >= flag_threshold) & (tum >= flag_tumour_threshold)
    flag_idx = top_indices(img, flag_mask, int(flag_mask.sum()))
    flag_rows = _pair_rows("cross", flag_idx, ids, split_names, ii, jj, img, tum)
    pd.DataFrame(flag_rows, columns=_PAIR_COLUMNS).to_csv(out_dir / "flagged.csv", index=False)

    cross_scores = img[cross]
    quantiles = (
        {str(q): float(np.quantile(cross_scores, q)) for q in _QUANTILES}
        if cross_scores.size
        else {}
    )
    summary: dict[str, Any] = {
        "n_cases": len(ids),
        "n_cases_per_split": {s: int((sp == s).sum()) for s in sorted(set(split_names))},
        "n_pairs_cross": int(cross.sum()),
        "n_pairs_within": int((~cross).sum()),
        "n_exact_duplicate_groups": len(dup_rows),
        "n_exact_duplicate_groups_cross_split": sum(r["cross_split"] for r in dup_rows),
        "n_flagged": int(flag_mask.sum()),
        "cross_split_image_score_quantiles": quantiles,
        "empty_label_cases": empty_labels,
        "split_overlap": split_overlap,
        "config": {
            "prep_dir": str(prep_dir),
            "splits_path": str(splits_path),
            "grid": list(grid),
            "top_k": top_k,
            "flag_threshold": flag_threshold,
            "flag_tumour_threshold": flag_tumour_threshold,
        },
    }
    write_json(summary, out_dir / "summary.json")
    logger.info(
        "leakage: %d exact groups, %d flagged pairs, %d split-overlap cases",
        len(dup_rows),
        summary["n_flagged"],
        len(split_overlap),
    )
    return summary


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; reads `cfg.analysis.leakage`."""
    setup_logging(level="INFO")
    c = cfg.analysis.leakage
    run(
        prep_dir=hydra.utils.to_absolute_path(c.prep_dir),
        splits_path=hydra.utils.to_absolute_path(c.splits_path),
        out_dir=hydra.utils.to_absolute_path(c.out_dir),
        grid=tuple(c.grid),
        top_k=int(c.top_k),
        flag_threshold=float(c.flag_threshold),
        flag_tumour_threshold=float(c.flag_tumour_threshold),
    )


if __name__ == "__main__":
    main()
