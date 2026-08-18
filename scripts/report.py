"""Hydra entry point for the Phase 4 structured-report driver.

This script runs no model, loads no checkpoint, no atlas, and no volume. Both
`scripts/burden.py` and `scripts/localize.py` have already walked a split and
written their tables to disk; this script's only job is to JOIN those two
tables, per case, and call `neurovision.reporting.report.build_report` /
`write_report`. Recomputing either table here would duplicate both drivers,
take minutes instead of seconds, and risk a report disagreeing with the CSVs
already published in `docs/experiments.md` -- the report library
(`src/neurovision/reporting/report.py`) is a pure function of already-computed
artifacts, and this driver is the thin thing that finds those artifacts on
disk and feeds them in.

Example usage:

    python scripts/report.py \\
        analysis.report.burden_dir=outputs/burden_neurovision \\
        analysis.report.localize_dir=outputs/localize_neurovision

The guard this script exists to enforce: `burden_dir` and `localize_dir` must
come from the SAME underlying segmentation. `outputs/burden_gt` and
`outputs/burden_neurovision` are sibling directories that differ only by
suffix (the same family of confusion CLAUDE.md's "Three eval directories
differ only by suffix" note already cost this project once), so pointing
`burden_dir` at ground truth while `localize_dir` holds a prediction would
silently mix a ground-truth burden profile with a prediction-derived
structure list. Nothing would fail on its own -- the join is on `case_id` and
every case is present on both sides -- and the artifact would look entirely
plausible. `load_inputs` therefore checks that `burden_config.yaml` and
`localize_config.yaml` agree on `source`, `split`, and `resolved_source_dir`
before doing anything else.
"""

from __future__ import annotations

import logging
import os
import subprocess
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from neurovision.anatomy.localize import Classification, load_classification
from neurovision.reporting.report import Provenance, build_report, write_report
from neurovision.utils.io import ensure_dir, read_yaml, write_yaml
from neurovision.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Relative to this file, so the script works from any working directory and on
# any machine -- no absolute paths. Copied from scripts/localize.py.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")

# The keys burden_config.yaml and localize_config.yaml must agree on. Both
# files are written by their respective drivers (run_burden / run_localize)
# and always carry these three -- source/split from the composed
# analysis.{burden,localize} config, resolved_source_dir added by the driver
# itself.
_PROVENANCE_KEYS: tuple[str, ...] = ("source", "split")


@dataclass(frozen=True)
class ReportInputs:
    """Everything one run needs, loaded and cross-checked once.

    Attributes:
        burden: `burden.csv`, indexed by `case_id`.
        anatomy: `anatomy.csv`, long format, WITH a `case_id` column (not
            indexed -- a case's rows are a slice, not a single row).
        anatomy_summary: `anatomy_summary.csv`, indexed by `case_id`.
        classification: The eloquence map's own metadata (name, evidence,
            citation, coverage gaps, near-eloquent threshold), read with no
            atlas via `load_classification`.
        provenance_atlas: `{"name", "version", "source", "licence"}`, read
            from `localize_config.yaml`'s own `atlas` block.
        knowledge_versions: `{"eloquence_map": <version>, "aal_lobes":
            <version>}`, each file's own `version` field.
        segmentation_source: `"prediction"` or `"label"`, from
            `localize_config.yaml`.
        segmentation_dir: `localize_config.yaml`'s `resolved_source_dir`.
        coverage_line: The knowledge-base coverage summary, read from
            `localize_config.yaml` rather than recomputed -- it belongs to
            the run that produced `anatomy.csv`, not to this one.
        split: The frozen split name both input tables were built over.
    """

    burden: pd.DataFrame
    anatomy: pd.DataFrame
    anatomy_summary: pd.DataFrame
    classification: Classification
    provenance_atlas: dict[str, str]
    knowledge_versions: dict[str, int]
    segmentation_source: str
    segmentation_dir: str
    coverage_line: str
    split: str


def git_revision(repo_root: Path) -> str | None:
    """Returns the current commit SHA, or `None` if it cannot be determined.

    A report built from uncommitted code is not reproducible from its SHA
    alone, so `"-dirty"` is appended when the working tree has uncommitted
    changes.

    Args:
        repo_root: Directory to run `git` in.

    Returns:
        The SHA (optionally suffixed `"-dirty"`), or `None` on any failure --
        `repo_root` is not a git repository, `git` is not installed, or the
        command otherwise fails. Never raises.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    if not sha:
        return None

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, FileNotFoundError):
        return sha
    if status.returncode == 0 and status.stdout.strip():
        return f"{sha}-dirty"
    return sha


def _check_provenance_agreement(
    burden_config: dict,
    localize_config: dict,
    burden_dir: Path,
    localize_dir: Path,
) -> None:
    """Raises if `burden_config.yaml` and `localize_config.yaml` disagree on their source run.

    See the module docstring for why this matters: `outputs/burden_gt` and
    `outputs/burden_neurovision` are sibling directories that differ only by
    suffix, and a mismatch here would silently mix a ground-truth burden
    profile with a prediction-derived structure list (or vice versa) with
    nothing else failing.

    Args:
        burden_config: The parsed `burden_config.yaml`.
        localize_config: The parsed `localize_config.yaml`.
        burden_dir: Where `burden_config` was read from (for the message).
        localize_dir: Where `localize_config` was read from (for the message).

    Raises:
        ValueError: Naming both directories and every disagreeing key, with
            each side's value.
    """
    mismatches: list[str] = []
    for key in _PROVENANCE_KEYS:
        b_val = burden_config.get(key)
        l_val = localize_config.get(key)
        if b_val != l_val:
            mismatches.append(f"{key} (burden={b_val!r}, localize={l_val!r})")

    b_dir = Path(str(burden_config.get("resolved_source_dir", ""))).resolve()
    l_dir = Path(str(localize_config.get("resolved_source_dir", ""))).resolve()
    if b_dir != l_dir:
        mismatches.append(f"resolved_source_dir (burden={b_dir}, localize={l_dir})")

    if mismatches:
        raise ValueError(
            f"load_inputs: burden_dir={burden_dir} and localize_dir={localize_dir} disagree on "
            f"provenance: {'; '.join(mismatches)}. Pointing burden_dir at one segmentation "
            "source (e.g. ground truth) while localize_dir holds another (e.g. a prediction) "
            "would silently mix a ground-truth burden profile with a prediction-derived "
            "structure list -- the join on case_id succeeds either way and the artifact would "
            "look entirely plausible. Point both at the same run's outputs."
        )


def load_inputs(cfg: DictConfig) -> ReportInputs:
    """Loads and cross-checks the burden table, the anatomy tables, and the knowledge metadata.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        A `ReportInputs`.

    Raises:
        FileNotFoundError: If any of the expected input files is missing.
        ValueError: If `burden_config.yaml` and `localize_config.yaml`
            disagree on provenance -- see `_check_provenance_agreement`.
        KeyError: If `localize_config.yaml` is missing an expected key
            (`source`, `split`, `resolved_source_dir`, `coverage_line`).
    """
    report_cfg = cfg.analysis.report
    burden_dir = Path(report_cfg.burden_dir)
    localize_dir = Path(report_cfg.localize_dir)

    burden_config = read_yaml(burden_dir / "burden_config.yaml")
    localize_config = read_yaml(localize_dir / "localize_config.yaml")
    _check_provenance_agreement(burden_config, localize_config, burden_dir, localize_dir)

    burden = pd.read_csv(burden_dir / "burden.csv")
    burden = burden.set_index("case_id")

    # Kept long, WITH case_id, unlike burden/anatomy_summary -- a case's rows
    # here are a slice of the table, not a single row.
    anatomy = pd.read_csv(localize_dir / "anatomy.csv")

    anatomy_summary = pd.read_csv(localize_dir / "anatomy_summary.csv")
    anatomy_summary = anatomy_summary.set_index("case_id")

    classification = load_classification(report_cfg.eloquence_map)
    lobe_doc = read_yaml(report_cfg.lobe_map)
    knowledge_versions = {
        "eloquence_map": classification.version,
        "aal_lobes": int(lobe_doc["version"]),
    }

    atlas_block = localize_config.get("atlas", {})
    provenance_atlas = {
        "name": str(atlas_block.get("name", "")),
        "version": str(atlas_block.get("version", "")),
        "source": str(atlas_block.get("source", "")),
        "licence": str(atlas_block.get("licence", "")),
    }

    return ReportInputs(
        burden=burden,
        anatomy=anatomy,
        anatomy_summary=anatomy_summary,
        classification=classification,
        provenance_atlas=provenance_atlas,
        knowledge_versions=knowledge_versions,
        segmentation_source=str(localize_config["source"]),
        segmentation_dir=str(localize_config["resolved_source_dir"]),
        coverage_line=str(localize_config["coverage_line"]),
        split=str(localize_config["split"]),
    )


def resolve_cases(inputs: ReportInputs, requested: Sequence[str] | None) -> list[str]:
    """Resolves the list of case ids to report on.

    Args:
        inputs: The loaded `ReportInputs`.
        requested: An explicit list of case ids, or `None` for "every case
            present in both `burden` and `anatomy_summary`".

    Returns:
        A sorted (when `requested` is `None`) or as-given (when explicit)
        list of case ids, guaranteed non-empty.

    Raises:
        ValueError: If `requested` names a case absent from `burden` and/or
            `anatomy_summary` (saying which table(s)), or if the resolved set
            is empty.
    """
    burden_ids = set(inputs.burden.index)
    summary_ids = set(inputs.anatomy_summary.index)

    if requested is None:
        only_burden = sorted(burden_ids - summary_ids)
        only_summary = sorted(summary_ids - burden_ids)
        if only_burden:
            logger.warning(
                "resolve_cases: %d case(s) present in burden.csv but not "
                "anatomy_summary.csv, excluded: %s",
                len(only_burden),
                only_burden,
            )
        if only_summary:
            logger.warning(
                "resolve_cases: %d case(s) present in anatomy_summary.csv but not "
                "burden.csv, excluded: %s",
                len(only_summary),
                only_summary,
            )
        cases = sorted(burden_ids & summary_ids)
    else:
        missing_burden = [c for c in requested if c not in burden_ids]
        missing_summary = [c for c in requested if c not in summary_ids]
        if missing_burden or missing_summary:
            parts = []
            if missing_burden:
                parts.append(f"missing from burden.csv: {missing_burden}")
            if missing_summary:
                parts.append(f"missing from anatomy_summary.csv: {missing_summary}")
            raise ValueError(f"resolve_cases: requested case(s) not found -- {'; '.join(parts)}.")
        cases = list(requested)

    if not cases:
        raise ValueError("resolve_cases: resolved case set is empty.")
    return cases


def report_one(case_id: str, inputs: ReportInputs, provenance: Provenance, top_n: int) -> dict:
    """Assembles one case's report dict from the already-loaded input tables.

    Args:
        case_id: The case to report on. Must be present in every table
            `inputs` carries.
        inputs: The loaded `ReportInputs`.
        provenance: This run's `Provenance` (shared by every case of the run).
        top_n: How many structure rows to keep in the anatomy block.

    Returns:
        A `build_report` output dict.

    Raises:
        KeyError: If `case_id` is absent from `inputs.burden` or
            `inputs.anatomy_summary`.
        ValueError: See `build_report`.
    """
    burden_row = inputs.burden.loc[case_id].to_dict()
    # Defensive: build_report would otherwise land this as a duplicate inside
    # the burden "other" block -- case_id is already the report's top-level
    # field. loc[case_id] on an index-set DataFrame does not itself carry a
    # case_id entry, but this guards against a burden table that was not
    # indexed the way this module expects.
    burden_row.pop("case_id", None)

    anatomy_table = inputs.anatomy[inputs.anatomy["case_id"] == case_id].drop(columns=["case_id"])

    anatomy_summary_row = inputs.anatomy_summary.loc[case_id].to_dict()

    classification = inputs.classification
    return build_report(
        case_id,
        burden_row,
        anatomy_table,
        anatomy_summary_row,
        provenance,
        evidence=classification.evidence,
        citation=classification.citation,
        classification_name=classification.name,
        coverage_line=inputs.coverage_line,
        coverage_gaps=classification.coverage_gaps,
        near_eloquent_mm=classification.near_eloquent_mm,
        top_n=top_n,
    )


def run_report(cfg: DictConfig) -> Path:
    """Builds and writes one report per resolved case, plus the manifest and config record.

    The manifest is rewritten after EVERY case (not once at the end), the
    same convention `scripts/evaluate.py`, `scripts/burden.py`, and
    `scripts/localize.py` use, so a killed run leaves usable partial output.
    A case that raises while being reported does not kill the run -- its
    traceback is logged at ERROR, it is counted as a failure, and the loop
    continues.

    Args:
        cfg: The full composed Hydra config.

    Returns:
        Path to the written `report_manifest.csv`.

    Raises:
        Whatever `load_inputs` / `resolve_cases` raises (a config problem,
        checked before any per-case work starts). Also:
        RuntimeError: If every resolved case failed to report -- a manifest
            of zero real rows would look like a successful run over an empty
            case set rather than a run where nothing actually worked.
    """
    report_cfg = cfg.analysis.report
    inputs = load_inputs(cfg)
    requested = list(report_cfg.cases) if report_cfg.cases is not None else None
    cases = resolve_cases(inputs, requested)

    out_dir = ensure_dir(Path(cfg.output_dir) / report_cfg.out_subdir)
    manifest_path = out_dir / "report_manifest.csv"

    repo_root = Path(__file__).resolve().parent.parent
    code_revision = git_revision(repo_root)
    # Computed ONCE and shared by every case of this run, so one run stamps
    # one timestamp rather than each case reporting a slightly different one.
    generated_utc = datetime.now(UTC).isoformat()

    provenance = Provenance(
        atlas_name=inputs.provenance_atlas["name"],
        atlas_version=inputs.provenance_atlas["version"],
        atlas_source=inputs.provenance_atlas["source"],
        atlas_licence=inputs.provenance_atlas["licence"],
        knowledge_versions=inputs.knowledge_versions,
        segmentation_source=inputs.segmentation_source,
        segmentation_dir=inputs.segmentation_dir,
        code_revision=code_revision,
        generated_utc=generated_utc,
    )

    top_n = int(report_cfg.top_n)
    markdown = bool(report_cfg.markdown)
    cwd = Path.cwd()

    manifest_rows: dict[str, dict[str, object]] = {}
    n_failed = 0

    for case_id in tqdm(cases, desc="Report"):
        try:
            report = report_one(case_id, inputs, provenance, top_n)
            written = write_report(report, out_dir, markdown=markdown)
        except Exception:
            n_failed += 1
            logger.error("report_one failed for case %s:\n%s", case_id, traceback.format_exc())
            continue

        markdown_path = written.get("markdown")
        eloquence = report["eloquence"]
        anatomy = report["anatomy"]
        manifest_rows[case_id] = {
            "case_id": case_id,
            "json_path": os.path.relpath(written["json"], cwd),
            "markdown_path": os.path.relpath(markdown_path, cwd) if markdown_path else "",
            "n_structures_involved": anatomy.get("n_structures_involved"),
            "n_eloquent_involved": len(eloquence.get("involved", [])),
            "distance_to_eloquent_mm": eloquence.get("distance_mm"),
            "near_eloquent": eloquence.get("near_eloquent"),
            "frac_unlabelled": anatomy.get("frac_unlabelled"),
        }

        # Rewritten after every case, not just at the end: a killed run keeps
        # every already-written case's row instead of losing all of them.
        pd.DataFrame.from_records(list(manifest_rows.values())).to_csv(manifest_path, index=False)

    n_succeeded = len(manifest_rows)
    logger.info(
        "Report complete: %d succeeded, %d failed (of %d resolved case(s)).",
        n_succeeded,
        n_failed,
        len(cases),
    )

    if n_succeeded == 0:
        raise RuntimeError(
            f"run_report: 0/{len(cases)} resolved case(s) succeeded -- see the ERROR-level log "
            "lines above for per-case tracebacks."
        )

    pd.DataFrame.from_records(list(manifest_rows.values())).to_csv(manifest_path, index=False)

    # A report artifact whose provenance is only in a terminal log nobody
    # kept cannot be traced months later -- same reasoning as
    # burden_config.yaml / localize_config.yaml / eval_config.yaml.
    config_record = OmegaConf.to_container(report_cfg, resolve=True)
    config_record["resolved_burden_dir"] = str(Path(report_cfg.burden_dir).resolve())
    config_record["resolved_localize_dir"] = str(Path(report_cfg.localize_dir).resolve())
    config_record["atlas"] = inputs.provenance_atlas
    config_record["knowledge_versions"] = inputs.knowledge_versions
    config_record["code_revision"] = code_revision
    config_record["generated_utc"] = generated_utc
    write_yaml(config_record, out_dir / "report_config.yaml")

    return manifest_path


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Build one structured report per resolved case, per the composed config.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    manifest_path = run_report(cfg)
    print(f"Report manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
