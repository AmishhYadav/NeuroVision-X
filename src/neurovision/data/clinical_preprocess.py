"""Clinical preprocessing: co-registration, atlas registration, skull-stripping.

Milestone 4, Phase E, task E2. `data/preprocessing.py`'s `preprocess_case`
(reorient to LPS, per-channel nonzero z-score, crop to the nonzero bbox) is
the RESEARCH path -- it assumes its input already lives in BraTS 2021's own
space: co-registered, atlas-aligned, skull-stripped. A real clinical scan is
none of those things. Feeding it straight into `preprocess_case` produces a
confident and completely wrong mask with nothing raising.

This module is what runs BEFORE that, not instead of it: it wraps
`brainles-preprocessing`'s `AtlasCentricPreprocessor` (co-registration ->
SRI24 atlas registration -> optional N4 -> HD-BET skull-stripping) so its
output lands in the same space BraTS 2021 already lives in, and the
unmodified research path then consumes it exactly as it always has.

**The central design constraint.** `brainles-preprocessing`, `antspyx` and
`HD-BET` live only in `.venv-clinical` (see `requirements-clinical.txt`);
the project's main test suite runs in `.venv` and must stay green there. So
this file is split cleanly in two, the same shape as `dicom_ingest.py`:

- The **planning layer** (`resolve_atlas_name`, `resolve_use_gpu`,
  `build_plan`) is pure: it validates inputs, resolves the atlas name and
  the device, decides which stages run, and computes every output path --
  all without importing `brainles_preprocessing` and without touching the
  filesystem beyond checking that input files exist. It is fully testable
  in `.venv`.
- The **execution layer** (`run_plan`, `preprocess_clinical_study`) is thin:
  it builds the brainles objects the plan describes and calls `.run()`.
  Every function here imports `brainles_preprocessing` *inside its body*,
  never at module scope, so importing this module never requires
  `.venv-clinical`.

**A dependency default this module exists to override.**
`AtlasCentricPreprocessor.__init__` defaults `use_gpu=True`, and
`HDBetExtractor.extract`'s own `device` parameter defaults to `0` (GPU 0).
Read against `brainles_preprocessing==0.6.13`'s own source
(`preprocessor/preprocessor.py`, `modality.py`): `AtlasCentricPreprocessor`
stores `self.use_gpu` and `CenterModality.extract_brain_region` computes
`device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else
"cpu")` and passes that explicit `device=` into `HDBetExtractor.extract`,
overriding its own `device=0` default. So the propagation genuinely works --
passing `use_gpu=False` does stop HD-BET from touching a GPU -- but nothing
in the dependency stops `use_gpu=True` being silently the default on a Mac
with no CUDA build. `resolve_use_gpu` exists purely to make sure this
module's caller can never inherit that default by omission (hard
constraint 3: no CUDA-only assumptions, device resolved once from config).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurovision.data.dicom_ingest import ROLES
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir

logger = logging.getLogger(__name__)

# Mirrors `brainles_preprocessing.constants.Atlas`'s member names, measured
# against brainles-preprocessing==0.6.13 on 2026-08-24. Kept as a plain
# tuple, not imported from the dependency, so `resolve_atlas_name` can run
# with no heavy import at module scope. The guarded test
# `test_atlas_member_names_match_the_dependency` is the tripwire: it asserts
# this tuple equals `[a.name for a in Atlas]` in the real installed package,
# so a brainles upgrade that adds or renames an atlas is caught immediately
# instead of `resolve_atlas_name` silently accepting or rejecting the wrong
# set of names.
_ATLAS_NAMES: tuple[str, ...] = (
    "BRATS_SRI24",
    "BRATS_SRI24_SKULLSTRIPPED",
    "SRI24",
    "SRI24_SKULLSTRIPPED",
    "BRATS_MNI152",
    "MNI152",
)

# Every stage `AtlasCentricPreprocessor.run` can save an intermediate
# directory for, mapped to the keyword argument name `run_plan` passes it
# under. Order matches the pipeline's own execution order (see that
# method's docstring in brainles' source).
_SAVE_DIR_KWARG_BY_STAGE: dict[str, str] = {
    "coregistration": "save_dir_coregistration",
    "atlas_registration": "save_dir_atlas_registration",
    "atlas_correction": "save_dir_atlas_correction",
    "n4_bias_correction": "save_dir_n4_bias_correction",
    "brain_extraction": "save_dir_brain_extraction",
    "defacing": "save_dir_defacing",
    "transformations": "save_dir_transformations",
}


@dataclass(frozen=True)
class PreprocessPlan:
    """Everything decided before any heavy work starts. Pure data.

    Attributes:
        inputs: role -> input NIfTI path, keys a subset of `ROLES`.
        center_role: The role every other modality is co-registered to.
        moving_roles: The other supplied roles, in `ROLES` order.
        atlas: A validated `Atlas` member NAME (e.g. `"BRATS_SRI24"`).
        use_gpu: Resolved GPU flag (see `resolve_use_gpu`).
        run_n4: Whether N4 bias correction runs.
        run_brain_extraction: Whether HD-BET skull-stripping runs.
        run_defacing: Whether defacing runs.
        out_dir: Root directory every output path below lives under.
        outputs: role -> final normalised volume path,
            `<out_dir>/<role>.nii.gz`. Skull-stripped when
            `run_brain_extraction` is True; otherwise normalised but still
            skulled (see `run_plan`'s docstring for exactly which brainles
            output path each case wires to).
        brain_mask_path: `<out_dir>/brain_mask.nii.gz`. Only produced when
            `run_brain_extraction` is True.
        intermediate_dirs: stage name -> directory, one entry per stage
            that will actually run, populated only when
            `cfg.clinical.preprocess.keep_intermediate` is true.
        log_file: `<out_dir>/preprocess.log`.
    """

    inputs: dict[str, Path]
    center_role: str
    moving_roles: tuple[str, ...]
    atlas: str
    use_gpu: bool
    run_n4: bool
    run_brain_extraction: bool
    run_defacing: bool
    out_dir: Path
    outputs: dict[str, Path]
    brain_mask_path: Path
    intermediate_dirs: dict[str, Path]
    log_file: Path


@dataclass(frozen=True)
class PreprocessResult:
    """The outcome of executing a `PreprocessPlan`.

    Attributes:
        plan: The plan that was executed.
        outputs: role -> the final volume path actually written (same
            paths as `plan.outputs`, returned here too so a caller does not
            have to reach back into `plan`).
        brain_mask: `plan.brain_mask_path`, whether or not it was produced
            (only produced when `plan.run_brain_extraction` is True).
        stages_run: Names of the stages that actually ran, regardless of
            whether their intermediates were kept.
        warnings: Non-fatal notes -- missing roles, no moving modalities,
            an `out_dir` that already held a previous run's output.
        transformations_dir: `plan.intermediate_dirs["transformations"]`,
            the directory `AtlasCentricPreprocessor.run` was told to write
            every modality's fitted registration transforms into (one
            subdirectory per role, named after that role's own
            `modality_name` -- see
            `neurovision.data.clinical_resample.resample_mask_to_source`,
            which reads this directory back to resample a predicted mask
            from atlas space into a modality's native geometry). `None`
            when the "transformations" stage was not kept -- today that
            only happens if `cfg.clinical.preprocess.keep_intermediate` is
            `False`, since `_stage_flags` always requests this stage, but
            the type stays honest rather than assuming the config can never
            change.
    """

    plan: PreprocessPlan
    outputs: dict[str, Path]
    brain_mask: Path
    stages_run: tuple[str, ...]
    warnings: tuple[str, ...]
    # Defaults to None (rather than being a required positional field) so
    # existing call sites that build a PreprocessResult by hand (fakes in
    # tests, e.g. test_app_clinical_jobs.py's _fake_preprocess_clinical_study)
    # do not have to be touched just because this field was added.
    transformations_dir: Path | None = None


def resolve_atlas_name(name: str) -> str:
    """Validate a configured atlas name against `brainles_preprocessing.constants.Atlas`.

    Deliberately does not import brainles to do this -- see the module
    docstring. `test_atlas_member_names_match_the_dependency` is the
    tripwire that keeps `_ATLAS_NAMES` honest against the installed
    package.

    Args:
        name: A candidate `Atlas` member name, e.g. `"BRATS_SRI24"`.

    Returns:
        `name`, unchanged, once validated.

    Raises:
        ValueError: If `name` is not one of `_ATLAS_NAMES`, listing the
            valid names.
    """
    if name not in _ATLAS_NAMES:
        raise ValueError(
            f"resolve_atlas_name: unknown atlas {name!r}; expected one of {_ATLAS_NAMES}."
        )
    return name


def resolve_use_gpu(cfg: Any) -> bool:
    """Resolve whether clinical preprocessing should use a GPU.

    Reads `cfg.clinical.preprocess.use_gpu`. When it is a real `bool`, that
    value is honoured outright. When it is `None`, the flag is derived from
    `get_device(cfg).type == "cuda"` -- the same device every other module
    in this project resolves against, so preprocessing can never disagree
    with the rest of the run.

    This function exists because `AtlasCentricPreprocessor.__init__`
    defaults `use_gpu=True` regardless of what hardware is actually
    present. A `None` config value must never fall through to that
    default -- this function is the one and only place that "not
    configured" gets turned into an actual boolean, and it never returns
    `True` unless `get_device` itself resolved to `"cuda"`.

    Args:
        cfg: The root config, exposing `cfg.clinical.preprocess.use_gpu`
            and (when that is `None`) `cfg.device`.

    Returns:
        Whether to request GPU execution from brainles-preprocessing.
    """
    configured = cfg.clinical.preprocess.use_gpu
    if isinstance(configured, bool):
        return configured

    device = get_device(cfg)
    return device.type == "cuda"


def _stage_flags(
    *,
    moving_roles: tuple[str, ...],
    run_n4: bool,
    run_brain_extraction: bool,
    run_defacing: bool,
) -> dict[str, bool]:
    """Which pipeline stages will actually run, given a plan's flags.

    Pure function of the flags rather than of a `PreprocessPlan`, so
    `build_plan` can call it before the plan itself is constructed, and
    `run_plan` can call it again from the plan's own fields -- the two are
    guaranteed to agree because they compute from the same inputs.

    Atlas registration and atlas correction always run: this preprocessor
    is atlas-CENTRIC, that is the entire point of using it. Transformations
    are always saved (when intermediates are kept) because they are the
    provenance record for how the study was moved into atlas space.
    Co-registration only genuinely runs when there is something to
    co-register.
    """
    return {
        "coregistration": bool(moving_roles),
        "atlas_registration": True,
        "atlas_correction": True,
        "n4_bias_correction": run_n4,
        "brain_extraction": run_brain_extraction,
        "defacing": run_defacing,
        "transformations": True,
    }


def _plan_warnings(plan: PreprocessPlan) -> tuple[str, ...]:
    """Structural warnings derivable purely from a built `PreprocessPlan`.

    Shared by `build_plan` (which only logs them -- `PreprocessPlan` itself
    carries no warnings field) and `run_plan` (which also returns them in
    `PreprocessResult.warnings`), so the two can never disagree about what
    was worth warning about.
    """
    warnings: list[str] = []

    missing_roles = tuple(role for role in ROLES if role not in plan.inputs)
    if missing_roles:
        warnings.append(
            f"missing modality role(s) {missing_roles}; only "
            f"{tuple(plan.inputs)} will be preprocessed. Refusing an "
            "incomplete study is a separate, later step (E4), not this one."
        )

    if not plan.moving_roles:
        warnings.append("only the center modality was supplied; there is nothing to co-register.")

    if plan.out_dir.exists() and any(plan.out_dir.iterdir()):
        warnings.append(
            f"out_dir {plan.out_dir} already contains output from a previous run; "
            "it will be overwritten."
        )

    return tuple(warnings)


def build_plan(
    cfg: Any,
    inputs: Mapping[str, Path],
    out_dir: Path | None = None,
) -> PreprocessPlan:
    """Decide every path and every stage flag, without doing any heavy work.

    Touches no heavy dependency and creates no directories -- the only
    filesystem access is checking that each input file exists (and, for
    the overwrite warning, whether `out_dir` already has contents). That is
    what makes this function fully testable in `.venv`.

    Args:
        cfg: The root config, exposing `cfg.clinical.preprocess` (see
            `configs/clinical/default.yaml`) and `cfg.device`.
        inputs: role -> input NIfTI path. Keys must be a subset of
            `ROLES`; need not be all four (see module docstring's edge
            cases -- an incomplete study is legitimate here).
        out_dir: Root output directory. Defaults to
            `cfg.clinical.preprocess.out_dir` when `None`.

    Returns:
        The fully decided `PreprocessPlan`.

    Raises:
        ValueError: If `inputs` is empty, if `cfg.clinical.preprocess
            .center_modality` is not a key of `inputs`, or if `inputs`
            contains a role outside `ROLES`.
        FileNotFoundError: If any input path does not exist -- all missing
            paths are listed together in one error, not one at a time.
    """
    preprocess_cfg = cfg.clinical.preprocess

    if not inputs:
        raise ValueError("build_plan: inputs is empty; at least the center modality is required.")

    unknown_roles = sorted(role for role in inputs if role not in ROLES)
    if unknown_roles:
        raise ValueError(
            f"build_plan: unknown role key(s) {unknown_roles} in inputs; "
            f"expected roles from {ROLES}."
        )

    center_role = preprocess_cfg.center_modality
    if center_role not in inputs:
        raise ValueError(
            f"build_plan: center_modality {center_role!r} (from config) is not a key of "
            f"the supplied inputs {sorted(inputs)}."
        )

    inputs = {role: Path(path) for role, path in inputs.items()}
    missing_files = sorted(str(path) for path in inputs.values() if not path.is_file())
    if missing_files:
        raise FileNotFoundError(f"build_plan: input file(s) not found: {missing_files}")

    # `ROLES` order, never dict-insertion order -- see the class docstring.
    moving_roles = tuple(role for role in ROLES if role in inputs and role != center_role)

    atlas = resolve_atlas_name(str(preprocess_cfg.atlas))
    use_gpu = resolve_use_gpu(cfg)
    run_n4 = bool(preprocess_cfg.n4_bias_correction)
    run_brain_extraction = bool(preprocess_cfg.brain_extraction)
    run_defacing = bool(preprocess_cfg.defacing)
    keep_intermediate = bool(preprocess_cfg.keep_intermediate)

    resolved_out_dir = Path(out_dir) if out_dir is not None else Path(preprocess_cfg.out_dir)

    outputs = {role: resolved_out_dir / f"{role}.nii.gz" for role in inputs}
    brain_mask_path = resolved_out_dir / "brain_mask.nii.gz"
    log_file = resolved_out_dir / "preprocess.log"

    stage_flags = _stage_flags(
        moving_roles=moving_roles,
        run_n4=run_n4,
        run_brain_extraction=run_brain_extraction,
        run_defacing=run_defacing,
    )
    intermediate_dirs: dict[str, Path] = {}
    if keep_intermediate:
        intermediate_dirs = {
            stage: resolved_out_dir / "intermediate" / stage
            for stage, runs in stage_flags.items()
            if runs
        }

    plan = PreprocessPlan(
        inputs=inputs,
        center_role=center_role,
        moving_roles=moving_roles,
        atlas=atlas,
        use_gpu=use_gpu,
        run_n4=run_n4,
        run_brain_extraction=run_brain_extraction,
        run_defacing=run_defacing,
        out_dir=resolved_out_dir,
        outputs=outputs,
        brain_mask_path=brain_mask_path,
        intermediate_dirs=intermediate_dirs,
        log_file=log_file,
    )

    for warning in _plan_warnings(plan):
        logger.warning(warning)

    return plan


def run_plan(plan: PreprocessPlan) -> PreprocessResult:
    """Execute a `PreprocessPlan`: co-registration, atlas registration, skull-stripping.

    Creates `plan.out_dir` and its intermediate directories, builds the
    `AtlasCentricPreprocessor` and its modalities exactly as `plan`
    describes, calls `.run(...)` with the plan's save directories, and then
    verifies every promised output file actually exists on disk --
    `brainles-preprocessing` is never trusted to have written what it was
    asked for (see `CLAUDE.md`'s trap 9: a glob that matched nothing has
    already shipped past a green test here).

    Which brainles output path each role is wired to depends on
    `plan.run_brain_extraction`: when True, `plan.outputs[role]` is
    `normalized_bet_output_path` (skull-stripped); when False, it is
    `normalized_skull_output_path` (normalised, atlas-registered, but still
    skulled) -- `plan.outputs[role]` is always written to, either way.

    Every modality is given a `PercentileNormalizer`. This is NOT about
    matching BraTS's training intensity distribution -- `preprocessing.py`'s
    `preprocess_case` does the real (nonzero z-score) normalisation
    afterwards, on the output this function writes. It is about making
    ANTs' registration cost function well-conditioned: registration
    similarity metrics assume roughly comparable intensity ranges across
    the images being aligned, and a raw clinical scan's intensities are on
    an arbitrary, scanner-dependent scale.

    Args:
        plan: A `PreprocessPlan` from `build_plan`.

    Returns:
        A `PreprocessResult` describing what was written.

    Raises:
        RuntimeError: If, after `.run()` returns, any promised output file
            (or the brain mask, when `plan.run_brain_extraction` is True)
            is missing -- names every missing path.
    """
    from brainles_preprocessing.brain_extraction import HDBetExtractor
    from brainles_preprocessing.constants import Atlas
    from brainles_preprocessing.modality import CenterModality, Modality
    from brainles_preprocessing.n4_bias_correction import SitkN4BiasCorrector
    from brainles_preprocessing.normalization import PercentileNormalizer
    from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
    from brainles_preprocessing.registration import ANTsRegistrator

    # The tripwire: a brainles upgrade that adds or renames an Atlas member
    # must fail loudly here, not let `resolve_atlas_name` silently validate
    # against a stale hardcoded tuple.
    live_atlas_names = tuple(member.name for member in Atlas)
    assert live_atlas_names == _ATLAS_NAMES, (
        "clinical_preprocess._ATLAS_NAMES is out of sync with the installed "
        f"brainles_preprocessing.constants.Atlas: this module has {_ATLAS_NAMES}, "
        f"the installed package has {live_atlas_names}. Update _ATLAS_NAMES."
    )

    warnings = _plan_warnings(plan)
    for warning in warnings:
        logger.warning(warning)

    ensure_dir(plan.out_dir)
    for stage_dir in plan.intermediate_dirs.values():
        ensure_dir(stage_dir)

    normalizer = PercentileNormalizer()

    def _output_kwargs(role: str) -> dict[str, Path]:
        if plan.run_brain_extraction:
            return {"normalized_bet_output_path": plan.outputs[role]}
        return {"normalized_skull_output_path": plan.outputs[role]}

    center_kwargs: dict[str, Any] = {
        "modality_name": plan.center_role,
        "input_path": plan.inputs[plan.center_role],
        "normalizer": normalizer,
        "n4_bias_correction": plan.run_n4,
        **_output_kwargs(plan.center_role),
    }
    if plan.run_brain_extraction:
        center_kwargs["bet_mask_output_path"] = plan.brain_mask_path
    center_modality = CenterModality(**center_kwargs)

    moving_modalities = [
        Modality(
            modality_name=role,
            input_path=plan.inputs[role],
            normalizer=normalizer,
            n4_bias_correction=plan.run_n4,
            **_output_kwargs(role),
        )
        for role in plan.moving_roles
    ]

    brain_extractor = HDBetExtractor() if plan.run_brain_extraction else None

    preprocessor = AtlasCentricPreprocessor(
        center_modality=center_modality,
        moving_modalities=moving_modalities,
        registrator=ANTsRegistrator(),
        brain_extractor=brain_extractor,
        atlas_image_path=Atlas[plan.atlas],
        n4_bias_corrector=SitkN4BiasCorrector() if plan.run_n4 else None,
        temp_folder=plan.out_dir / "_brainles_tmp",
        # This is exactly the guard resolve_use_gpu exists for --
        # AtlasCentricPreprocessor's own default is use_gpu=True.
        use_gpu=plan.use_gpu,
    )

    run_kwargs: dict[str, Path] = {"log_file": plan.log_file}
    for stage, save_dir in plan.intermediate_dirs.items():
        run_kwargs[_SAVE_DIR_KWARG_BY_STAGE[stage]] = save_dir
    preprocessor.run(**run_kwargs)

    expected_files = list(plan.outputs.values())
    if plan.run_brain_extraction:
        expected_files.append(plan.brain_mask_path)

    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"run_plan: brainles-preprocessing did not produce the following expected "
            f"output(s): {missing}. Check {plan.log_file} for what actually ran."
        )

    stage_flags = _stage_flags(
        moving_roles=plan.moving_roles,
        run_n4=plan.run_n4,
        run_brain_extraction=plan.run_brain_extraction,
        run_defacing=plan.run_defacing,
    )
    stages_run = tuple(stage for stage, runs in stage_flags.items() if runs)

    return PreprocessResult(
        plan=plan,
        outputs=dict(plan.outputs),
        brain_mask=plan.brain_mask_path,
        stages_run=stages_run,
        warnings=warnings,
        transformations_dir=plan.intermediate_dirs.get("transformations"),
    )


def preprocess_clinical_study(
    cfg: Any,
    inputs: Mapping[str, Path],
    out_dir: Path | None = None,
) -> PreprocessResult:
    """Plan and run clinical preprocessing for one study in a single call.

    Args:
        cfg: The root config, exposing `cfg.clinical.preprocess` and
            `cfg.device`.
        inputs: role -> input NIfTI path, keys a subset of `ROLES`.
        out_dir: Root output directory. Defaults to
            `cfg.clinical.preprocess.out_dir` when `None`.

    Returns:
        The `PreprocessResult` from executing the built plan.
    """
    plan = build_plan(cfg, inputs, out_dir=out_dir)
    return run_plan(plan)
