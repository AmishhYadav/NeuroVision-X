"""Gate 2 — does entropy PLUS disagreement localise per-voxel error better than entropy alone?

Driver for `docs/research/preregistration_gate2.md`. **Read that file before
changing anything here**: the combiner, the fit split, the two endpoints, the
budget, the thresholds and the six-test Holm family were all fixed in writing
before a single number was computed.

Runs no model and needs no GPU. Every input is a cache already on disk: the
per-case ambiguity `.npz` written by `scripts/extract_ambiguity.py`, and the
preprocessed labels.

## Why this imports scripts/detection_stats.py

Gate 1 and Gate 2 must sample the SAME voxels under the SAME label-free mask,
or a difference between them is a difference in bookkeeping rather than in
signal. Rather than restate that logic, this script imports
`_process_voxel_case` and `load_cohort` from the Gate 1 driver by path — the
same importlib pattern `tests/test_detection_stats.py` already uses. A copy
would be free to drift; an import cannot.

## The fit split is never a reported cohort

`analysis.gate2.fit_cohort` is the frozen 187-case validation split. It is
fitted on and then never scored, so no reported number comes from data the
combiner has seen. This is the temperature-scaling rule applied to a second
fitted quantity.

## Sample caching

Extracting features costs one `.npz` load plus a full post-processing chain
per case — about 4.5 hours over 535 cases, which is most of this script's
runtime. The sampled features are therefore cached per cohort under
`<out_dir>/samples_<cohort>.npz` and reused when the cache covers every case
and was built with the same sampling parameters. Delete the cache to force a
re-extract; the parameters are stored inside it so a changed mask or budget
invalidates it automatically rather than silently reusing the wrong sample.
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

from neurovision.analysis.localisation import (
    case_auroc,
    fit_combiner,
    rank_transform,
    recall_at_budget,
)
from neurovision.analysis.statistics import holm_bonferroni
from neurovision.utils.io import ensure_dir, write_json
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_DETECTION_PATH = Path(__file__).resolve().parent / "detection_stats.py"


def _load_detection_module():
    """Imports scripts/detection_stats.py by path, so the two gates share one sampler."""
    spec = importlib.util.spec_from_file_location("detection_stats_script", _DETECTION_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"gate2: cannot import {_DETECTION_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["detection_stats_script"] = module
    spec.loader.exec_module(module)
    return module


_DETECTION = _load_detection_module()


def _sample_parameters(voxel_cfg: Any) -> dict[str, Any]:
    """The sampling parameters a cache must match to be reusable."""
    return {
        "mask": str(voxel_cfg.mask),
        "dilation_mm": float(voxel_cfg.dilation_mm),
        "max_voxels_per_case": int(voxel_cfg.max_voxels_per_case),
        "seed": int(voxel_cfg.seed),
        "region": "ANY",
    }


def collect_cohort_samples(
    cohort_cfg: Any, voxel_cfg: Any, cache_path: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Returns `{case_id: {"score", "control", "positive"}}` for one cohort's ANY rows.

    `score` is mean disagreement, `control` is single-pass predictive entropy
    and `positive` is the per-voxel error indicator — all at the same sampled
    voxel positions, drawn from the label-free predicted-dilated mask.

    Args:
        cohort_cfg: One `analysis.gate2` cohort entry.
        voxel_cfg: `analysis.gate2.voxel`.
        cache_path: Where the per-cohort sample cache lives.

    Returns:
        One entry per case that produced a usable sample. Cases with an empty
        mask or no label on disk are dropped by `_process_voxel_case` and are
        simply absent, exactly as in Gate 1.
    """
    params = _sample_parameters(voxel_cfg)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            stored = {k: str(v) for k, v in zip(data["param_keys"], data["param_values"])}
            if stored == {k: str(v) for k, v in params.items()}:
                case_ids = [str(c) for c in data["case_ids"]]
                out = {
                    cid: {
                        "score": data[f"score::{cid}"],
                        "control": data[f"control::{cid}"],
                        "positive": data[f"positive::{cid}"],
                    }
                    for cid in case_ids
                }
                logger.info(
                    "Cohort %r: reusing cached samples for %d case(s) from %s.",
                    cohort_cfg.name,
                    len(out),
                    cache_path,
                )
                return out
            logger.warning(
                "Cohort %r: sample cache at %s was built with different parameters "
                "(%s vs %s); re-extracting.",
                cohort_cfg.name,
                cache_path,
                stored,
                params,
            )

    summary, npz_paths = _DETECTION.load_cohort(cohort_cfg)
    prep_dir = Path(str(cohort_cfg.prep_dir))
    generator = np.random.default_rng(int(voxel_cfg.seed))

    out: dict[str, dict[str, np.ndarray]] = {}
    for case_id in sorted(summary.index.astype(str)):
        per_region = _DETECTION._process_voxel_case(
            case_id=case_id,
            npz_path=npz_paths[case_id],
            prep_dir=prep_dir,
            mask_mode=str(voxel_cfg.mask),
            dilation_mm=float(voxel_cfg.dilation_mm),
            max_voxels=int(voxel_cfg.max_voxels_per_case),
            generator=generator,
            cohort_name=str(cohort_cfg.name),
        )
        if per_region is None:
            continue
        any_rows = per_region["ANY"]
        out[case_id] = {
            "score": np.asarray(any_rows["score"], dtype=np.float32),
            "control": np.asarray(any_rows["control"], dtype=np.float32),
            "positive": np.asarray(any_rows["positive"], dtype=bool),
        }

    payload: dict[str, Any] = {
        "case_ids": np.array(sorted(out), dtype=object).astype(str),
        "param_keys": np.array(list(params), dtype=str),
        "param_values": np.array([str(v) for v in params.values()], dtype=str),
    }
    for cid, arrays in out.items():
        for key, array in arrays.items():
            payload[f"{key}::{cid}"] = array
    ensure_dir(cache_path.parent)
    np.savez_compressed(cache_path, **payload)
    logger.info(
        "Cohort %r: cached samples for %d case(s) to %s.", cohort_cfg.name, len(out), cache_path
    )
    return out


def fit_both_arms(samples: dict[str, dict[str, np.ndarray]]):
    """Fits the entropy-only and entropy+disagreement combiners on pooled fit voxels.

    The rank transform is applied PER CASE before pooling, so a case with an
    unusually wide entropy range cannot dominate the fit.
    """
    entropy_parts, disagreement_parts, positive_parts = [], [], []
    for arrays in samples.values():
        entropy_parts.append(rank_transform(arrays["control"]))
        disagreement_parts.append(rank_transform(arrays["score"]))
        positive_parts.append(np.asarray(arrays["positive"], dtype=bool))
    entropy = np.concatenate(entropy_parts)
    disagreement = np.concatenate(disagreement_parts)
    positive = np.concatenate(positive_parts)

    baseline = fit_combiner(entropy, disagreement, positive, mode="entropy")
    combined = fit_combiner(entropy, disagreement, positive, mode="both")
    logger.info(
        "Fit on %d voxels from %d case(s): entropy-only coefficients %s (converged=%s); "
        "entropy+disagreement coefficients %s (converged=%s).",
        positive.size,
        len(samples),
        np.round(baseline.coefficients, 4).tolist(),
        baseline.converged,
        np.round(combined.coefficients, 4).tolist(),
        combined.converged,
    )
    return baseline, combined


def score_cohort(
    samples: dict[str, dict[str, np.ndarray]], baseline, combined, budget: float
) -> pd.DataFrame:
    """Per-case AUROC and recall@budget for both arms, plus their paired differences."""
    rows = []
    for case_id, arrays in sorted(samples.items()):
        entropy_rank = rank_transform(arrays["control"])
        disagreement_rank = rank_transform(arrays["score"])
        positive = np.asarray(arrays["positive"], dtype=bool)

        score_base = baseline.score(entropy_rank, disagreement_rank)
        score_both = combined.score(entropy_rank, disagreement_rank)

        auroc_base = case_auroc(score_base, positive)
        auroc_both = case_auroc(score_both, positive)
        recall_base = recall_at_budget(score_base, positive, budget=budget)
        recall_both = recall_at_budget(score_both, positive, budget=budget)
        rows.append(
            {
                "case_id": case_id,
                "n_voxels": int(positive.size),
                "error_fraction": float(positive.mean()),
                "auroc_entropy": auroc_base,
                "auroc_both": auroc_both,
                "delta_auroc": auroc_both - auroc_base,
                "recall_entropy": recall_base,
                "recall_both": recall_both,
                "delta_recall": recall_both - recall_base,
            }
        )
    return pd.DataFrame(rows).set_index("case_id")


def _paired_bootstrap(a: np.ndarray, b: np.ndarray, *, generator, n_boot: int, ci: float):
    """Paired bootstrap over CASE indices, returning the CI and a two-sided p-value.

    `analysis.statistics.paired_bootstrap_ci` gives the interval but does not
    expose its replicates, and Holm needs a p-value. Rather than resample
    twice with two conventions, the replicates are drawn here and BOTH the
    interval and the p-value come from them; a test asserts the interval
    agrees with `paired_bootstrap_ci` on the same seed, so the canonical
    implementation stays the reference.

    Resampling is over case indices into the DIFFERENCE array, never over
    the two arms independently -- the pairing is the whole point, and
    independent resampling would measure the spread of each arm's absolute
    AUROC instead of the spread of the paired difference.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    idx = generator.integers(0, n, size=(n_boot, n))
    replicates = diff[idx].mean(axis=1)
    lo_q = (1.0 - ci) / 2.0
    lo, hi = np.quantile(replicates, [lo_q, 1.0 - lo_q])
    p_boot = _DETECTION._bootstrap_two_sided_p(replicates, n_boot)
    return float(diff.mean()), float(lo), float(hi), float(p_boot)


def _endpoint_row(
    cohort: str, endpoint: str, a: np.ndarray, b: np.ndarray, stats_cfg: Any
) -> dict[str, Any]:
    """One paired-bootstrap row: `a` is the two-feature arm, `b` the entropy-only arm.

    Cases where either arm is NaN (no errors, or entirely error -- see
    `case_auroc`) are dropped pairwise and counted, never silently imputed.
    """
    generator = np.random.default_rng(int(stats_cfg.seed))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    point, lo, hi, p_boot = _paired_bootstrap(
        a[finite],
        b[finite],
        generator=generator,
        n_boot=int(stats_cfg.n_boot),
        ci=float(stats_cfg.ci),
    )
    return {
        "cohort": cohort,
        "endpoint": endpoint,
        "n": int(finite.sum()),
        "n_dropped": int((~finite).sum()),
        "mean_entropy": float(b[finite].mean()),
        "mean_both": float(a[finite].mean()),
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "contains_zero": bool(lo <= 0.0 <= hi),
        "p_boot": p_boot,
    }


def build_verdict(family: pd.DataFrame, thresholds: Any, external: list[str]) -> dict[str, Any]:
    """Applies the pre-registered decision rule verbatim.

    From `docs/research/preregistration_gate2.md`:

    | Pass | On at least one EXTERNAL cohort: delta_auroc >= 0.01 with a CI
      excluding zero, AND delta_recall@5% >= 0.02 with a CI excluding zero |
    | Partial | A CI excludes zero on an external cohort but the magnitude
      misses its threshold |
    | Fail | No external cohort shows a CI excluding zero on either endpoint |

    Both conjuncts must hold on the SAME cohort for a pass -- the rule says
    "on at least one external cohort", not "somewhere among them".
    """
    per_cohort: dict[str, Any] = {}
    passed, partial = [], []
    for cohort in sorted(set(family["cohort"])):
        rows = family[family["cohort"] == cohort].set_index("endpoint")
        auroc_row = rows.loc["delta_auroc"]
        recall_row = rows.loc["delta_recall"]
        auroc_sig = not bool(auroc_row["contains_zero"])
        recall_sig = not bool(recall_row["contains_zero"])
        meets = (
            auroc_sig
            and recall_sig
            and float(auroc_row["delta"]) >= float(thresholds.delta_auroc)
            and float(recall_row["delta"]) >= float(thresholds.delta_recall)
        )
        per_cohort[cohort] = {
            "external": cohort in external,
            "delta_auroc": float(auroc_row["delta"]),
            "auroc_ci_excludes_zero": auroc_sig,
            "delta_recall": float(recall_row["delta"]),
            "recall_ci_excludes_zero": recall_sig,
            "meets_threshold": bool(meets),
        }
        if cohort in external:
            if meets:
                passed.append(cohort)
            elif auroc_sig or recall_sig:
                partial.append(cohort)

    if passed:
        verdict = "pass"
    elif partial:
        verdict = "partial"
    else:
        verdict = "fail"
    return {
        "verdict": verdict,
        "thresholds": {
            "delta_auroc": float(thresholds.delta_auroc),
            "delta_recall": float(thresholds.delta_recall),
        },
        "external_cohorts": external,
        "passed_cohorts": passed,
        "partial_cohorts": partial,
        "per_cohort": per_cohort,
        "preregistration": "docs/research/preregistration_gate2.md",
    }


def run_gate2(cfg: DictConfig) -> Path:
    """Fits on the val split, applies frozen to every cohort, writes the tables."""
    gate_cfg = cfg.analysis.gate2
    out_dir = ensure_dir(gate_cfg.out_dir)
    budget = float(gate_cfg.budget)

    fit_samples = collect_cohort_samples(
        gate_cfg.fit_cohort, gate_cfg.voxel, out_dir / f"samples_{gate_cfg.fit_cohort.name}.npz"
    )
    if not fit_samples:
        raise ValueError(
            f"run_gate2: the fit cohort {gate_cfg.fit_cohort.name!r} produced no usable case. "
            "Extract its ambiguity maps first (scripts/extract_ambiguity_serial.py --cohort val)."
        )
    baseline, combined = fit_both_arms(fit_samples)

    per_case_frames, endpoint_rows = [], []
    for cohort_cfg in gate_cfg.cohorts:
        name = str(cohort_cfg.name)
        samples = collect_cohort_samples(
            cohort_cfg, gate_cfg.voxel, out_dir / f"samples_{name}.npz"
        )
        if not samples:
            logger.warning("Cohort %r produced no usable case; skipping.", name)
            continue
        table = score_cohort(samples, baseline, combined, budget)
        table.insert(0, "cohort", name)
        per_case_frames.append(table)
        for endpoint, col_a, col_b in (
            ("delta_auroc", "auroc_both", "auroc_entropy"),
            ("delta_recall", "recall_both", "recall_entropy"),
        ):
            endpoint_rows.append(
                _endpoint_row(
                    name,
                    endpoint,
                    table[col_a].to_numpy(),
                    table[col_b].to_numpy(),
                    gate_cfg.bootstrap,
                )
            )

    if not endpoint_rows:
        raise ValueError("run_gate2: no cohort was ready -- nothing to compute.")

    family = pd.DataFrame(endpoint_rows)
    # Holm across the whole pre-registered family, applied ONCE. Fixed at 6
    # tests (2 endpoints x 3 cohorts) before any p-value was seen; if a cohort
    # is missing the family is smaller and that must be stated, not hidden.
    # holm_bonferroni returns (adjusted_pvalues, reject) -- both are used, so
    # the rejection flag comes from the canonical implementation rather than
    # from a re-comparison here that could drift from its alpha convention.
    p_holm, reject = holm_bonferroni(family["p_boot"].to_numpy(), alpha=float(gate_cfg.alpha))
    family["p_holm"] = p_holm
    family["reject"] = reject

    per_case = pd.concat(per_case_frames)
    per_case.to_csv(out_dir / "gate2_per_case.csv")
    family.to_csv(out_dir / "gate2_family.csv", index=False)

    external = [str(c.name) for c in gate_cfg.cohorts if bool(c.get("external", False))]
    verdict = build_verdict(family, gate_cfg.thresholds, external)
    verdict["combiner"] = {
        "entropy_only_coefficients": baseline.coefficients.tolist(),
        "entropy_only_converged": baseline.converged,
        "both_coefficients": combined.coefficients.tolist(),
        "both_converged": combined.converged,
        "fit_cohort": str(gate_cfg.fit_cohort.name),
        "n_fit_cases": len(fit_samples),
        "n_fit_voxels": int(combined.n_fit_voxels),
        "budget": budget,
    }
    write_json(verdict, out_dir / "gate2_verdict.json")

    _log_summary(family, verdict)
    return out_dir


def _log_summary(family: pd.DataFrame, verdict: dict[str, Any]) -> None:
    """Prints the family table and the verdict, so a terminal run is self-contained."""
    print("=" * 70)
    print("Gate 2 -- entropy + disagreement vs entropy alone, per-voxel error localisation")
    print("=" * 70)
    for _, row in family.iterrows():
        print(
            f"  [{row['cohort']}] {row['endpoint']}: {row['mean_entropy']:.4f} -> "
            f"{row['mean_both']:.4f}  delta={row['delta']:+.4f} "
            f"CI=({row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}) p_holm={row['p_holm']:.4g} "
            f"n={row['n']}" + (f" (dropped {row['n_dropped']})" if row["n_dropped"] else "")
        )
    print(f"  verdict: {verdict['verdict']!r}")
    print(f"  Holm family: {len(family)} test(s), {int(family['reject'].sum())} rejected")


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    setup_logging(level="INFO")
    out_dir = run_gate2(cfg)
    logger.info("Gate 2 outputs written to %s", out_dir)


if __name__ == "__main__":
    main()
