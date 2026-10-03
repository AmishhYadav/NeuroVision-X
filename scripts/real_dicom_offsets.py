"""Reproduces experiments note 57, "R3": why does the real-DICOM front end lose WT Dice?

## Why this script exists

On the original DICOM of 22 accepted BraTS test-split patients, our clinical
front end loses a median of about 0.23 WT Dice against BraTS ground truth,
scored on the BraTS SRI24 grid. The question is whether that is a
coordinate/template convention bug or plain registration disagreement.
Per accepted case this script measures two integer voxel translations:

- `t`: the shift that best aligns OUR atlas-space tumour (WT) mask onto the
  BraTS WT label.
- `b`: the shift that best aligns BraTS's OWN brain mask onto the SRI24
  template brain mask that the front end registers to.

A convention bug (flipped axis, half-voxel origin, wrong template) would give
every case the same offset. The finding to reproduce is different: `t` mirrors
`b` case by case (t ~ -k*b, Pearson r ~ -0.86 along axis 1, anterior-posterior),
i.e. BraTS's own brain sits off the template by a case-specific amount and our
registration disagrees with theirs. It also checks that the template affine
equals the BraTS affine (no grid convention differs) and that re-scored WT
Dice matches what `result.json` logged (so the case -> job mapping is right).

Reads note 55's outputs; runs nothing through the pipeline. CPU only.

## Sign convention

`best_shift(moving, fixed)` returns `s` such that `shift_mask(moving, s)`
moves `moving` onto `fixed`: `shifted[x + s] = moving[x]`.

## Outputs (in `cfg.analysis.real_dicom_offsets.out_dir`)

- `offsets_per_case.csv`, `brain_vs_atlas_test.csv`, `summary.json`,
  `real_dicom_offsets_config.yaml`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import nibabel as nib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from scipy import stats

from neurovision.analysis.real_dicom_scoring import to_reference_grid, uncrop_to_full
from neurovision.utils.io import ensure_dir
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Strict: 32 hex chars for the job, and a full `BraTS2021_NNNNN` path component
# (the trailing slash stops a case id matching as a prefix of a longer one).
_JOB_RE = re.compile(r"created job ([0-9a-f]{32}) for study_dir=.*/_extract/(BraTS2021_\d{5})/")
_MASK_REL = Path("dicom_seg_work") / "atlas_space_mask.nii.gz"


def shift_mask(mask: np.ndarray, s: Sequence[int]) -> np.ndarray:
    """Translates a mask by integer voxels with ZERO fill (no wrap-around).

    Args:
        mask: `(D, H, W)` array.
        s: Integer shift per axis; `out[x + s] = mask[x]`.

    Returns:
        `(D, H, W)` array, same dtype; voxels shifted out are dropped, new
        voxels are 0.
    """
    if len(s) != mask.ndim:
        raise ValueError(f"shift has {len(s)} entries for a {mask.ndim}-D mask")
    out = np.zeros_like(mask)
    src, dst = [], []
    for n, k in zip(mask.shape, s):
        k = int(k)
        if abs(k) >= n:
            return out
        src.append(slice(max(0, -k), n - max(0, k)))
        dst.append(slice(max(0, k), n - max(0, -k)))
    out[tuple(dst)] = mask[tuple(src)]
    return out


def best_shift_with_peak(
    moving: np.ndarray, fixed: np.ndarray, max_shift: int
) -> tuple[np.ndarray, int]:
    """Finds the integer translation maximising overlap of `moving` with `fixed`.

    Uses FFT cross-correlation on arrays zero-padded by `max_shift` on every
    side, so the circular correlation equals the linear one inside the searched
    window (no wrap-around aliasing). Ties go to the first maximum in C order
    over the window (`np.argmax`), which is deterministic. The window is ordered
    lag 0 first, then positive lags, then negative lags, so ties prefer lag 0,
    then positive lags, in C order over the three axes.

    Args:
        moving: `(D, H, W)` boolean mask.
        fixed: `(D, H, W)` boolean mask, same shape.
        max_shift: Search window; every `|s_k| <= max_shift`.

    Returns:
        `(s, peak)`: int vector of length 3, `s`, such that
        `shift_mask(moving, s)` moves `moving` onto `fixed`, and `peak`, the
        overlap voxel count at `s` (0 means no overlap anywhere in the window,
        so `s` is meaningless).
    """
    if moving.shape != fixed.shape:
        raise ValueError(f"shape mismatch: {moving.shape} vs {fixed.shape}")
    p = int(max_shift)
    pad = [(p, p)] * 3
    m = np.pad(moving.astype(np.float64), pad)
    f = np.pad(fixed.astype(np.float64), pad)
    # c[s] = sum_x m[x] * f[x + s]  ==  ifft(conj(M) * F)[s]
    corr = np.fft.irfftn(np.conj(np.fft.rfftn(m)) * np.fft.rfftn(f), s=m.shape, axes=(0, 1, 2))
    del m, f
    # Lags -p..p live at indices 0..p and L-p..L-1 (negative lags wrap).
    idx = [np.r_[0 : p + 1, n - p : n] for n in corr.shape]
    lags = [np.r_[0 : p + 1, -p:0] for _ in idx]
    window = corr[np.ix_(*idx)]
    # Corr values are integer counts; round away FFT noise so ties are exact.
    window = np.round(window)
    flat = int(np.argmax(window))
    pos = np.unravel_index(flat, window.shape)
    shift = np.array([lags[k][pos[k]] for k in range(3)], dtype=int)
    return shift, int(window.flat[flat])


def best_shift(moving: np.ndarray, fixed: np.ndarray, max_shift: int) -> np.ndarray:
    """`best_shift_with_peak` without the peak count; see that function."""
    return best_shift_with_peak(moving, fixed, max_shift)[0]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean arrays; 1.0 if both are empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def _all_job_ids(log_path: str | Path) -> dict[str, list[str]]:
    """Maps case id -> every logged job id, in log order (last = most recent run).

    Matching is by strict regex on a full `BraTS2021_NNNNN/` path component, so
    one case id can never match as a prefix of another.
    """
    out: dict[str, list[str]] = {}
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _JOB_RE.search(line)
            if m:
                out.setdefault(m.group(2), []).append(m.group(1))
    return out


def resolve_job(case_id: str, jobs: Mapping[str, list[str]], work_dir: Path) -> str:
    """Picks the last logged job for a case whose atlas mask file exists.

    Args:
        case_id: BraTS case id.
        jobs: Output of `_all_job_ids`.
        work_dir: Validation work dir containing `jobs/`.

    Returns:
        The job id.

    Raises:
        ValueError: No logged job for the case has an atlas-space mask.
    """
    ids = jobs.get(case_id, [])
    for job in reversed(ids):
        if (work_dir / "jobs" / job / _MASK_REL).exists():
            if job != ids[-1]:
                logger.warning(
                    "%s: last logged job %s has no mask; using %s", case_id, ids[-1], job
                )
            return job
    raise ValueError(f"{case_id}: no logged job has {_MASK_REL} (logged: {ids})")


def load_atlas_brain(path: str | Path, reference_affine: np.ndarray) -> np.ndarray:
    """Loads the template brain mask `(D, H, W)` and checks its affine.

    Raises:
        ValueError: Template affine differs from `reference_affine` (atol 1e-3).
    """
    img = nib.load(str(path))
    if not np.allclose(img.affine, np.asarray(reference_affine), atol=1e-3):
        raise ValueError(
            f"atlas affine differs from BraTS affine:\n{img.affine}\nvs\n{reference_affine}"
        )
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return data > 0


def load_brats_brain(case_dir: Path, meta: Mapping[str, Any]) -> np.ndarray:
    """BraTS brain mask on the full grid `(D, H, W)`: any modality nonzero."""
    img = np.load(case_dir / "image.npy", mmap_mode="r")  # (4, D, H, W)
    cropped = np.abs(np.asarray(img, dtype=np.float32)).sum(0) > 0
    return uncrop_to_full(cropped.astype(np.uint8), meta).astype(bool)


def analyse_case(
    case_id: str,
    ours_full: np.ndarray,
    gt_full: np.ndarray,
    brain: np.ndarray,
    atlas: np.ndarray,
    logged_dice_wt: float,
    max_shift: int,
    usable_threshold: float,
) -> dict[str, Any]:
    """Measures t, b and the Dice before/after each correction for one case.

    Args:
        case_id: Case id.
        ours_full: `(D, H, W)` our atlas-space labels {0..3} on the GT grid.
        gt_full: `(D, H, W)` BraTS labels {0..3}.
        brain: `(D, H, W)` bool BraTS brain.
        atlas: `(D, H, W)` bool template brain.
        logged_dice_wt: WT Dice recorded in result.json.
        max_shift: Search window in voxels.
        usable_threshold: WT and TC must both reach this for "usable".

    Returns:
        Flat dict of per-case measurements.
    """
    ours_wt, gt_wt = ours_full > 0, gt_full > 0
    ours_tc, gt_tc = np.isin(ours_full, (1, 3)), np.isin(gt_full, (1, 3))
    t, t_peak = best_shift_with_peak(ours_wt, gt_wt, max_shift)
    b = best_shift(brain, atlas, max_shift)
    nb = -b
    row: dict[str, Any] = {"case_id": case_id}
    row.update({f"t{k}": int(t[k]) for k in range(3)})
    row.update({f"b{k}": int(b[k]) for k in range(3)})
    row["t_found"] = bool(t_peak > 0)  # False: empty/non-overlapping, t meaningless
    row["t_norm_mm"] = float(np.linalg.norm(t))
    variants = {
        "raw": (ours_wt, ours_tc),
        "brain_corrected": (shift_mask(ours_wt, nb), shift_mask(ours_tc, nb)),
        "tumour_fit": (shift_mask(ours_wt, t), shift_mask(ours_tc, t)),
    }
    for name, (wt, tc) in variants.items():
        w, c = dice(wt, gt_wt), dice(tc, gt_tc)
        row[f"wt_{name}"] = w
        row[f"tc_{name}"] = c
        row[f"usable_{name}"] = bool(w >= usable_threshold and c >= usable_threshold)
    row["logged_dice_WT"] = float(logged_dice_wt)
    return row


def check_dice_match(rows: Sequence[Mapping[str, Any]], atol: float) -> None:
    """Self-check: re-scored raw WT Dice must equal result.json's.

    Raises:
        ValueError: Naming the first case that differs by more than `atol`.
    """
    for r in rows:
        if abs(r["wt_raw"] - r["logged_dice_WT"]) > atol:
            raise ValueError(
                f"{r['case_id']}: re-scored WT {r['wt_raw']:.6f} != logged "
                f"{r['logged_dice_WT']:.6f} (atol {atol}); wrong case->job or grid mapping"
            )


def _corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pearson and Spearman of x vs y (NaN if either is constant)."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        nan = float("nan")
        return {"pearson_r": nan, "pearson_p": nan, "spearman_rho": nan, "spearman_p": nan}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr[0]),
        "pearson_p": float(pr[1]),
        "spearman_rho": float(sr[0]),
        "spearman_p": float(sr[1]),
    }


def summarise(per_case: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Any]:
    """Builds summary.json content from the two tables."""
    out: dict[str, Any] = {"n_cases": int(len(per_case))}
    # Cases with no overlap have a meaningless t; keep them out of t-vs-b.
    found = per_case[per_case["t_found"].astype(bool)]
    out["n_t_excluded"] = int(len(per_case) - len(found))
    out["t_vs_b"] = {
        f"axis{k}": _corr(found[f"t{k}"].to_numpy(float), found[f"b{k}"].to_numpy(float))
        for k in range(3)
    }
    t1 = found["t1"]
    out["t1_signs"] = {
        "positive": int((t1 > 0).sum()),
        "negative": int((t1 < 0).sum()),
        "zero": int((t1 == 0).sum()),
    }
    for name in ("raw", "brain_corrected", "tumour_fit"):
        out[f"median_wt_{name}"] = float(per_case[f"wt_{name}"].median())
        out[f"median_tc_{name}"] = float(per_case[f"tc_{name}"].median())
        out[f"usable_{name}"] = int(per_case[f"usable_{name}"].sum())
    diff = per_case["wt_brain_corrected"] - per_case["wt_raw"]
    out["wilcoxon_p_brain_corrected_vs_raw"] = (
        float(stats.wilcoxon(per_case["wt_brain_corrected"], per_case["wt_raw"]).pvalue)
        if (diff != 0).any()
        else float("nan")
    )
    out["n_improved_wt_brain_corrected"] = int((diff > 0).sum())
    norm = cohort["norm"].to_numpy(float)
    out["cohort_brain_to_atlas"] = {
        "n": int(len(cohort)),
        "norm_median": float(np.median(norm)),
        "frac_ge_3mm": float((norm >= 3).mean()),
        "frac_ge_5mm": float((norm >= 5).mean()),
        "norm_max": float(norm.max()),
    }
    return out


def run(block: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Runs the R3 analysis from a resolved `real_dicom_offsets` config block.

    Args:
        block: `cfg.analysis.real_dicom_offsets`.

    Returns:
        The summary dict (also written to summary.json).
    """
    work_dir = Path(str(block["work_dir"]))
    gt_root = Path(str(block["gt_root"]))
    out_dir = ensure_dir(str(block["out_dir"]))
    max_shift = int(block["max_shift_vox"])
    thr = float(block["usable_threshold"])
    atol = float(block["dice_match_atol"])

    jobs = _all_job_ids(work_dir / "run.log")
    results = []
    for p in sorted((work_dir / "cases").glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("accepted") and r.get("scored") and not r.get("pilot"):
            results.append(r)
    logger.info("%d accepted, scored, non-pilot cases", len(results))
    if not results:
        raise ValueError("no accepted scored cases")
    expected = int(block["expected_n_cases"])
    if len(results) != expected:
        raise ValueError(f"found {len(results)} accepted scored cases, expected {expected}")

    atlas = None
    rows = []
    for r in results:
        cid = r["case_id"]
        job = resolve_job(cid, jobs, work_dir)
        meta = json.loads((gt_root / cid / "meta.json").read_text())
        affine = np.asarray(meta["affine"])
        if atlas is None:
            atlas = load_atlas_brain(str(block["atlas_brain"]), affine)
        nifti = nib.load(str(work_dir / "jobs" / job / _MASK_REL))
        mask = np.asarray(nifti.dataobj).astype(np.uint8)
        ours = to_reference_grid(mask, nifti.affine, affine)
        gt = uncrop_to_full(np.load(gt_root / cid / "label.npy"), meta)
        brain = load_brats_brain(gt_root / cid, meta)
        rows.append(analyse_case(cid, ours, gt, brain, atlas, r["dice_WT"], max_shift, thr))
        del mask, ours, gt, brain
    check_dice_match(rows, atol)
    per_case = pd.DataFrame(rows)
    per_case.to_csv(out_dir / "offsets_per_case.csv", index=False)

    test_ids = pd.read_csv(str(block["test_cases_from"]))["case_id"].tolist()
    crow = []
    for cid in test_ids:
        meta = json.loads((gt_root / cid / "meta.json").read_text())
        brain = load_brats_brain(gt_root / cid, meta)
        b = best_shift(brain, atlas, max_shift)
        crow.append(
            {
                "case_id": cid,
                "b0": int(b[0]),
                "b1": int(b[1]),
                "b2": int(b[2]),
                "norm": float(np.linalg.norm(b)),
                "dice_brain_atlas": dice(brain, atlas),
            }
        )
        del brain
    cohort = pd.DataFrame(crow)
    cohort.to_csv(out_dir / "brain_vs_atlas_test.csv", index=False)

    summary = summarise(per_case, cohort)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "real_dicom_offsets_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(block, resolve=True)))
        if isinstance(block, DictConfig)
        else OmegaConf.to_yaml(OmegaConf.create(dict(block)))
    )
    logger.info("real_dicom_offsets summary:\n%s", json.dumps(summary, indent=2))
    return summary


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point: `python scripts/real_dicom_offsets.py`.

    Args:
        cfg: Composed config; reads `cfg.analysis.real_dicom_offsets`.
    """
    setup_logging(level="INFO")
    run(cfg.analysis.real_dicom_offsets)


if __name__ == "__main__":
    main()
