"""Does the fusion gate open and close with anatomy? (prediction P1)

`docs/research/contribution.md` states P1 -- "the mechanism fires" -- as the
claim that the adaptive gate is not a decorative extra parameter block but a
spatially organised signal. Until now P1 was recorded as **undecided**: the
producer existed (`scripts/extract_gates.py` writes one tumour-centred patch
of gate maps per case) and the reducer existed
(`neurovision.metrics.boundary.distance_band_means`), but no number had ever
been written into the experiment log.

This script closes that gap. For every case in a gate-extraction directory it
bins the gate value by SIGNED distance to the ground-truth whole-tumour
surface -- negative inside the tumour, positive outside -- and reports, per
fusion level, the mean gate in each band plus two paired contrasts with
bootstrap confidence intervals.

Three decisions worth knowing before reading the output:

- **Signed bands, not absolute ones.** The interesting question is not "is the
  gate different near the surface" but "is it different INSIDE the tumour than
  OUTSIDE it", and an absolute-distance band folds those two sides together
  into one number that can hide a perfectly monotone profile.
- **Nearest-neighbour upsampling, never interpolation.** A stride-8 gate is an
  8^3 map inside a 64^3 patch; interpolating it to voxel resolution would
  invent precision the quantity does not have. Same rule the gate figure in
  `visualization/figures.py` follows.
- **The gate is the TRANSFORMER weight.** The fusion merge is
  `cnn + layer_scale * gate * attn` (see `models/fusion/adaptive_fusion.py`),
  so a gate near 1 means "admit the Swin branch's context here" and a gate
  near 0 means "this voxel is decided by the CNN branch alone". A sign
  confusion here inverts the entire mechanistic reading, which is why it is
  stated in the module docstring rather than left to the reader.

The distance field comes from the GROUND-TRUTH whole-tumour mask, never the
prediction: two levels binned by two different partitions of space would not
be comparable, which is the same reasoning `metrics/boundary.py` records for
the boundary-stratified error table.

Run:

    python scripts/gate_boundary_profile.py --gate-dir outputs/gates_test
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neurovision.analysis.statistics import paired_bootstrap_ci
from neurovision.metrics.boundary import (
    band_label,
    distance_band_means,
    signed_distance_to_boundary,
)
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Signed distance bands in mm. Negative is inside the tumour. Half-open
# [lo, hi) exactly like DEFAULT_BANDS, so every voxel lands in exactly one.
SIGNED_BANDS: tuple[tuple[float, float], ...] = (
    (-np.inf, -10.0),
    (-10.0, -5.0),
    (-5.0, -2.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, np.inf),
)

# Channel index of whole tumour in the saved label, matching
# neurovision.data.transforms.REGION_NAMES == ("ET", "TC", "WT").
WT_CHANNEL = 2


def upsample_nearest(gate: np.ndarray, target: int) -> torch.Tensor:
    """Repeats a coarse gate map up to the patch resolution, without interpolating.

    Args:
        gate: One level's gate map, shape `(d, h, w)`.
        target: Patch edge length the map must reach.

    Returns:
        A `(target, target, target)` float32 tensor.

    Raises:
        ValueError: If `target` is not an integer multiple of every axis of
            `gate` -- which would mean the gate came from a different patch
            size than the label saved beside it.
    """
    factors = [target // s for s in gate.shape]
    if any(f * s != target for f, s in zip(factors, gate.shape, strict=True)):
        raise ValueError(
            f"Gate map of shape {gate.shape} does not tile a {target}^3 patch by an "
            "integer factor on every axis."
        )
    block = np.ones(tuple(factors), dtype=np.float32)
    return torch.from_numpy(np.kron(gate.astype(np.float32), block))


def profile_case(path: Path, bands: tuple[tuple[float, float], ...]) -> list[dict[str, Any]]:
    """Bins every fusion level's gate map by signed distance for one case.

    Args:
        path: A `<case_id>.npz` written by `scripts/extract_gates.py`.
        bands: Signed distance bands, in mm.

    Returns:
        One row per fusion level, or an empty list when the case's
        ground-truth whole tumour is empty or fills the patch -- neither has a
        surface, and `signed_distance_to_boundary` returns all-NaN for both.
    """
    payload = np.load(path)
    case_id = path.stem
    label = payload["label"]
    wt = torch.from_numpy(np.ascontiguousarray(label[WT_CHANNEL]).astype(np.uint8))
    if int(wt.sum()) in (0, wt.numel()):
        logger.warning("Skipping %s: whole-tumour mask has no surface in this patch.", case_id)
        return []

    sdf = signed_distance_to_boundary(wt, spacing=(1.0, 1.0, 1.0))
    edge = int(wt.shape[0])

    rows: list[dict[str, Any]] = []
    level = 0
    while f"gate_level_{level}" in payload:
        gate = payload[f"gate_level_{level}"][0]
        means = distance_band_means(upsample_nearest(gate, edge), sdf, bands=bands, signed=True)
        rows.append({"case_id": case_id, "level": level, **means})
        level += 1
    return rows


def build_profile(gate_dir: Path, bands: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    """Runs `profile_case` over every case in a gate-extraction directory.

    Args:
        gate_dir: Directory of `<case_id>.npz` files.
        bands: Signed distance bands, in mm.

    Returns:
        A long DataFrame with one row per (case, level).

    Raises:
        FileNotFoundError: If the directory holds no `.npz` file.
    """
    paths = sorted(gate_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No gate .npz files found under {gate_dir.resolve()}.")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(profile_case(path, bands))
    return pd.DataFrame(rows)


def contrast_table(
    profile: pd.DataFrame,
    bands: tuple[tuple[float, float], ...],
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Paired near-boundary-versus-elsewhere contrasts, per fusion level.

    Pairing is per case: the same case contributes both sides of every
    difference, so the bootstrap resamples CASE indices into the difference
    array rather than resampling the two band means independently -- the same
    discipline `analysis.statistics.paired_bootstrap_ci` enforces everywhere
    else in this project.

    Args:
        profile: Output of `build_profile`.
        bands: The bands used to build it.
        n_boot: Bootstrap replicate count.
        seed: Seed for the bootstrap generator.

    Returns:
        One row per (level, contrast) with the paired difference, its CI, and
        whether that CI excludes zero.
    """
    labels = ["mean_" + band_label(lo, hi) for lo, hi in bands]
    inner, interior, healthy = labels[3], labels[0], labels[-1]
    rows: list[dict[str, Any]] = []
    for level in sorted(profile["level"].unique()):
        level_rows = profile[profile["level"] == level]
        for other, name in ((interior, "tumour interior"), (healthy, "healthy tissue")):
            a = level_rows[inner].to_numpy(dtype=float)
            b = level_rows[other].to_numpy(dtype=float)
            keep = np.isfinite(a) & np.isfinite(b)
            result = paired_bootstrap_ci(
                a[keep], b[keep], n_boot=n_boot, generator=np.random.default_rng(seed)
            )
            rows.append(
                {
                    "level": int(level),
                    "contrast": f"inner margin [-2,0) mm minus {name}",
                    "n": int(keep.sum()),
                    "diff": result.point,
                    "ci_lo": result.lo,
                    "ci_hi": result.hi,
                    "excludes_zero": not result.contains_zero,
                }
            )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Argument list, or `None` to read `sys.argv`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate-dir", type=Path, default=Path("outputs/gates_test"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Writes the per-case profile and prints the band table and contrasts.

    Args:
        argv: Argument list, or `None` to read `sys.argv`.

    Returns:
        Process exit code.
    """
    setup_logging(level="INFO")
    args = parse_args(argv)

    profile = build_profile(args.gate_dir, SIGNED_BANDS)
    labels = ["mean_" + band_label(lo, hi) for lo, hi in SIGNED_BANDS]
    contrasts = contrast_table(profile, SIGNED_BANDS, n_boot=args.n_boot, seed=args.seed)

    profile_path = args.gate_dir / "gate_boundary_profile.csv"
    contrast_path = args.gate_dir / "gate_boundary_contrasts.csv"
    profile.to_csv(profile_path, index=False)
    contrasts.to_csv(contrast_path, index=False)

    print("=" * 78)
    print(f"Gate vs signed distance to the ground-truth WT surface -- {args.gate_dir}")
    print(f"cases: {profile['case_id'].nunique()}   (negative distance = inside the tumour)")
    print("=" * 78)
    print(profile.groupby("level")[labels].mean().round(4).to_string())
    print()
    print(contrasts.round(4).to_string(index=False))
    print()
    print(f"wrote {profile_path}")
    print(f"wrote {contrast_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
