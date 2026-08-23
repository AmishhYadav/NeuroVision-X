"""Is inter-branch disagreement as good an error localiser as MC-dropout, at 1/10 the cost?

**Secondary analysis, explicitly outside the pre-registered Gate 1 and Gate 2
families.** `docs/research/preregistration_gate2.md` states that the
"matches MC-dropout at 1/10 the cost" claim is NOT made, because the per-voxel
MC mutual-information maps did not exist for the external cohorts. This script
is what makes it measurable: the maps were generated on 2026-08-23
(`neurovision-mc-{ssa,ped}`, N=10, ~1.25 h and ~2.2 h of T4 time).

## The claim is an EQUIVALENCE, and is tested as one

"As good as MC-dropout" cannot be established by failing to reject a
difference -- at n=60 that is indistinguishable from an underpowered study,
and this project has six underpowered nulls already. It is tested with a
paired TOST (`neurovision.analysis.equivalence`) against a margin of **0.03
AUROC fixed in `docs/research/execution_plan.md` Phase 2 before any external
MC map existed**.

## All three quantities are read at the SAME voxels

`_process_voxel_case` draws the label-free sample; this script asks it for the
drawn indices and reads the MC map at exactly those positions. Re-deriving the
mask and redrawing would compare the three quantities on three different
samples, which would look entirely reasonable and mean nothing.

## The cost asymmetry is the point

MC-dropout at N=10 is ten stochastic sliding-window passes per case.
Disagreement is a by-product of ONE deterministic pass -- the two encoder
branches have to be run anyway. So equivalence, if it holds, is a 10x
inference saving; and if it does not hold, the honest statement is that the
cheap signal is worse, with the gap quantified.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from neurovision.analysis.equivalence import paired_tost
from neurovision.analysis.localisation import case_auroc
from neurovision.analysis.statistics import paired_bootstrap_ci
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_DETECTION_PATH = Path(__file__).resolve().parent / "detection_stats.py"

_spec = importlib.util.spec_from_file_location("detection_stats_script", _DETECTION_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - import plumbing
    raise ImportError(f"mc_comparison: cannot import {_DETECTION_PATH}.")
_DETECTION = importlib.util.module_from_spec(_spec)
sys.modules["detection_stats_script"] = _DETECTION
_spec.loader.exec_module(_DETECTION)


def load_mc_map(mc_dir: Path, case_id: str) -> np.ndarray | None:
    """Loads one case's MC-dropout mutual-information map, or None if absent.

    Returns the REGION MEAN, matching how `_process_voxel_case` builds its
    `"ANY"` row from disagreement and entropy -- three quantities compared on
    one cohort must be reduced the same way or the comparison is between
    reductions rather than between signals.
    """
    path = mc_dir / f"{case_id}.npy"
    if not path.is_file():
        return None
    return np.load(path).astype(np.float32).mean(axis=0)


def score_cohort(cohort_cfg: Any, voxel_cfg: Any) -> pd.DataFrame:
    """Per-case AUROC for entropy, disagreement and MC mutual information."""
    summary, npz_paths = _DETECTION.load_cohort(cohort_cfg)
    prep_dir = Path(str(cohort_cfg.prep_dir))
    mc_dir = Path(str(cohort_cfg.mc_dir))
    generator = np.random.default_rng(int(voxel_cfg.seed))

    rows: list[dict[str, Any]] = []
    n_missing_mc = 0
    for case_id in sorted(summary.index.astype(str)):
        result = _DETECTION._process_voxel_case(
            case_id=case_id,
            npz_path=npz_paths[case_id],
            prep_dir=prep_dir,
            mask_mode=str(voxel_cfg.mask),
            dilation_mm=float(voxel_cfg.dilation_mm),
            max_voxels=int(voxel_cfg.max_voxels_per_case),
            generator=generator,
            cohort_name=str(cohort_cfg.name),
            return_indices=True,
        )
        if result is None:
            continue
        per_region, drawn = result
        any_rows = per_region["ANY"]

        mc_map = load_mc_map(mc_dir, case_id)
        if mc_map is None:
            n_missing_mc += 1
            continue
        # The MC pass and the ambiguity pass must have run on the same geometry.
        # A shape disagreement means two different preprocessing runs, and
        # sampling one at the other's flat indices would silently scramble the
        # correspondence -- every number downstream would still look sane. The
        # label is the shared reference both passes were built against, and
        # mmap_mode reads only its header.
        label_shape = np.load(prep_dir / case_id / "label.npy", mmap_mode="r").shape
        if mc_map.shape != tuple(label_shape):
            raise ValueError(
                f"mc_comparison: cohort {cohort_cfg.name!r} case {case_id!r}: MC map is "
                f"{mc_map.shape} but the preprocessed label is {tuple(label_shape)}. The MC "
                "run and the ambiguity run used different preprocessing."
            )

        positive = np.asarray(any_rows["positive"], dtype=bool)
        rows.append(
            {
                "case_id": case_id,
                "cohort": str(cohort_cfg.name),
                "auroc_entropy": case_auroc(any_rows["control"], positive),
                "auroc_disagreement": case_auroc(any_rows["score"], positive),
                "auroc_mc": case_auroc(mc_map.reshape(-1)[drawn], positive),
            }
        )

    if n_missing_mc:
        logger.warning(
            "Cohort %r: %d case(s) had no MC map and were dropped.", cohort_cfg.name, n_missing_mc
        )
    return pd.DataFrame(rows).set_index("case_id")


def compare(
    table: pd.DataFrame, cohort: str, margin: float, stats_cfg: Any
) -> list[dict[str, Any]]:
    """Paired differences against MC-dropout, plus the TOST for disagreement."""
    out: list[dict[str, Any]] = []
    mc = table["auroc_mc"].to_numpy()
    for name, column in (("disagreement", "auroc_disagreement"), ("entropy", "auroc_entropy")):
        arm = table[column].to_numpy()
        finite = np.isfinite(arm) & np.isfinite(mc)
        ci = paired_bootstrap_ci(
            arm[finite],
            mc[finite],
            generator=np.random.default_rng(int(stats_cfg.seed)),
            n_boot=int(stats_cfg.n_boot),
            ci=float(stats_cfg.ci),
        )
        tost = paired_tost(arm[finite], mc[finite], margin=margin)
        out.append(
            {
                "cohort": cohort,
                "arm": name,
                "n": int(finite.sum()),
                "mean_arm": float(arm[finite].mean()),
                "mean_mc": float(mc[finite].mean()),
                "difference": float(ci.point),
                "ci_lo": float(ci.lo),
                "ci_hi": float(ci.hi),
                "contains_zero": bool(ci.contains_zero),
                "tost_margin": margin,
                "tost_ci_lo": tost.ci_lo,
                "tost_ci_hi": tost.ci_hi,
                "tost_p": tost.p_tost,
                "equivalent_to_mc": tost.equivalent,
            }
        )
    return out


def run_mc_comparison(cfg: DictConfig) -> Path:
    """Scores every configured cohort and writes the comparison table."""
    mc_cfg = cfg.analysis.mc_comparison
    out_dir = ensure_dir(mc_cfg.out_dir)
    margin = float(mc_cfg.tost_margin)

    per_case_frames, rows = [], []
    for cohort_cfg in mc_cfg.cohorts:
        mc_dir = Path(str(cohort_cfg.mc_dir))
        if not mc_dir.is_dir():
            logger.warning(
                "Cohort %r: no MC directory at %s; skipping. Generate it first "
                "(kernel neurovision-mc-<cohort>).",
                cohort_cfg.name,
                mc_dir,
            )
            continue
        table = score_cohort(cohort_cfg, mc_cfg.voxel)
        if table.empty:
            logger.warning("Cohort %r produced no usable case; skipping.", cohort_cfg.name)
            continue
        per_case_frames.append(table)
        rows.extend(compare(table, str(cohort_cfg.name), margin, mc_cfg.bootstrap))

    if not rows:
        raise ValueError("run_mc_comparison: no cohort had MC maps -- nothing to compute.")

    comparison = pd.DataFrame(rows)
    pd.concat(per_case_frames).to_csv(out_dir / "mc_per_case.csv")
    comparison.to_csv(out_dir / "mc_comparison.csv", index=False)
    write_json(
        {
            "tost_margin": margin,
            "margin_source": "docs/research/execution_plan.md Phase 2, fixed before any "
            "external-cohort MC map existed",
            "pre_registered": False,
            "note": "Secondary analysis, outside the Gate 1 and Gate 2 families.",
            "rows": rows,
        },
        out_dir / "mc_comparison.json",
    )

    print("=" * 70)
    print("Disagreement vs MC-dropout as a per-voxel error localiser")
    print("=" * 70)
    for row in rows:
        verdict = "EQUIVALENT" if row["equivalent_to_mc"] else "not equivalent"
        print(
            f"  [{row['cohort']}] {row['arm']:13} {row['mean_arm']:.4f} vs MC {row['mean_mc']:.4f}"
            f"  diff={row['difference']:+.4f} CI=({row['ci_lo']:+.4f}, {row['ci_hi']:+.4f})"
            f"  TOST@{row['tost_margin']}: {verdict} (p={row['tost_p']:.3g}) n={row['n']}"
        )
    return out_dir


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    setup_logging(level="INFO")
    out_dir = run_mc_comparison(cfg)
    logger.info("MC comparison written to %s", out_dir)


if __name__ == "__main__":
    main()
