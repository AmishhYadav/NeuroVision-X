"""Severity-ladder driver (MasterPlan W3.2): graded image corruptions, three models, small outputs.

Protocol: `docs/research_docs/preregistrations/preregistration_severity_ladder.md` (binding; read it
first). The 189 BraTS test cases get one of five deterministic corruptions (`thick_slice`,
`rician`, `bias`, `gamma`, `motion`) at four severities, plus the clean image (level 0). Each
(model, corruption, level) condition is segmented and only SMALL per-case results are saved --
never a volume. Corrupted volumes are regenerated from `case_seed(base_seed, case_id, kind,
level)` every time; the only corrupted voxels that ever touch disk are the transient inputs
nnU-Net needs, deleted before the shard ends.

Work unit = SHARD = (model, kind, level). Level 0 (clean) is run once per model as kind "none".
Shard order: every model's level-0 shard first (the bridge), then by kind, level, model. A shard
writes, under `<out_dir>/<model>/<kind>/level<k>/`:

    per_case.csv         metrics, volumes, entropy, gate signals, seconds per case
    curves.npz           per-case |G \\ M(tau)|, |G|, |M(tau)| on the conformal v2 grid, in
    curves_manifest.json exactly the format `conformal.load_curves_npz` reads
    ood_features.csv     the 46 input-statistics features of the (corrupted) image
    shard_meta.json      what was run (case ids, seed, severity)
    DONE                 written last

A shard is built in `<shard>.partial/` and renamed into place only when complete, so a session
kill at any moment leaves either a finished shard or no shard. A shard with `DONE` is skipped on
the next run (resume). `max_hours` is checked between chunks of cases; when it is exhausted the
shard in progress is discarded (never half-written), the run logs it and exits 0.

Bridge: after a model's level-0 shard, its per-case Dice is compared with the reference per-case
metrics of the clean test evaluation. A mean absolute difference above `bridge_tol` in any region
logs `NVX_BRIDGE: FAIL` and no corrupted shard runs for that model.

Gate signals saved per case (the gate's DECISION is applied offline, never here):
`predicted_dice_<R>` (QC model, on the model's own deployed mask and the CORRUPTED image channel),
`conformal_band_<R>` (mask inflation at the model's fitted conformal threshold), and mean
foreground entropy. Not computed here: `input_qc` (needs DICOM-stage volume info), `intended_use`
(needs patient age), and the OOD score (needs the frozen OOD model; features are saved instead).

Machine-greppable log lines: `NVX_SHARD_DONE model kind level n s_per_case`,
`NVX_BRIDGE: PASS|FAIL`, `NVX_BUDGET_EXHAUSTED`, `NVX_RUN_COMPLETE`.

Example:

    python scripts/severity_ladder.py analysis.severity_ladder.probe=true
    python scripts/severity_ladder.py analysis.severity_ladder.models_to_run=[neurovision]
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from neurovision.analysis.detection import case_entropy_scalars
from neurovision.analysis.gatekeeper_calibration import (
    case_conformal_band_widths,
    resolve_fitted_thresholds,
)
from neurovision.analysis.ood import extract_features
from neurovision.analysis.qc_inference import CaseArrays, entropy_from_logits, pack_sample
from neurovision.data.corruptions import KINDS, case_seed, corrupt_image
from neurovision.data.dataset import load_splits
from neurovision.data.transforms import REGION_NAMES
from neurovision.inference.postprocess import postprocess_logits
from neurovision.inference.sliding_window import sliding_window_predict
from neurovision.metrics.segmentation import classes_to_regions, compute_case_metrics
from neurovision.models.qc import build_segqc, predicted_dice
from neurovision.training.checkpoint import load_checkpoint
from neurovision.uncertainty.conformal import EXTENDED_THRESHOLDS, case_loss_curve
from neurovision.utils.device import get_device
from neurovision.utils.io import ensure_dir, read_json, write_json
from neurovision.utils.logging import setup_logging
from neurovision.utils.seed import set_seed

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
# Relative to this file so the script works from any cwd and on any machine.
_CONFIG_DIR = str(_SCRIPTS_DIR.parent / "configs")

_N_CHANNELS = 4  # image.npy channels: t1, t1ce, t2, flair
DONE_MARKER = "DONE"
_PARTIAL_SUFFIX = ".partial"


def _load_sibling(name: str) -> ModuleType:
    """Loads `scripts/<name>.py` by path (scripts/ is not a package), reusing a loaded copy.

    Registered as `<name>_script`, the same key `tests/script_loader.load_script` uses, so the
    script and its tests share ONE module object.
    """
    key = f"{name}_script"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _SCRIPTS_DIR / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"severity_ladder: cannot import scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


# Reused, not re-implemented: model building + checkpoint loading (evaluate.py), the curves.npz
# writer / manifest (conformal.py), and the nnU-Net glue (nnunet_predict_probs.py).
_EVAL = _load_sibling("evaluate")
_CONFORMAL = _load_sibling("conformal")
_NNUNET = _load_sibling("nnunet_predict_probs")


class BudgetExhausted(Exception):
    """Raised between chunks when the wall-clock budget (`max_hours`) has run out."""


# ---------------------------------------------------------------------------
# Plan: which shards, in which order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shard:
    """One (model, kind, level) work unit.

    Attributes:
        model: Model name (key of `severity_ladder.models`).
        kind: Corruption kind, or `"none"` for the clean level 0.
        level_index: 0 for clean, else 1..N (position in the kind's severity list; this is the
            number fed to `case_seed`, so it must not change when a level is skipped).
        severity: Kind-specific severity (see `corrupt_image`); 0.0 for clean.
    """

    model: str
    kind: str
    level_index: int
    severity: float

    @property
    def rel_dir(self) -> Path:
        """`<model>/<kind>/level<k>`, relative to the output root."""
        return Path(self.model) / self.kind / f"level{self.level_index}"


def plan_shards(sl_cfg: Mapping[str, Any], models: Sequence[str]) -> list[Shard]:
    """Lists every shard in run order: all level-0 shards first, then kind -> level -> model.

    Level 0 goes first for every model because it is the bridge: a model whose clean numbers do
    not reproduce must be caught before any corrupted number is produced.

    Args:
        sl_cfg: The `severity_ladder` config block (reads `kinds`, `skip_levels`,
            `skip_model_kinds`).
        models: Model names to plan for, in order.

    Returns:
        Shards in run order.

    Raises:
        ValueError: A configured kind is not a known corruption.
    """
    shards = [Shard(m, "none", 0, 0.0) for m in models]
    skip_levels = {int(k) for k in sl_cfg.get("skip_levels") or []}
    skip_model_kinds = {m: set(v) for m, v in (sl_cfg.get("skip_model_kinds") or {}).items()}
    for kind, severities in sl_cfg["kinds"].items():
        if kind not in KINDS or kind == "none":
            raise ValueError(f"severity_ladder.kinds has unknown kind {kind!r}; valid: {KINDS}.")
        for i, severity in enumerate(severities, start=1):
            if i in skip_levels:
                continue
            for m in models:
                if kind in skip_model_kinds.get(m, set()):
                    continue
                shards.append(Shard(m, kind, i, float(severity)))
    return shards


# ---------------------------------------------------------------------------
# Backends: corrupted image(s) in, logits out
# ---------------------------------------------------------------------------


def build_inhouse_cfg(
    cfg: DictConfig, model_config: Path, cfg_overrides: Mapping[str, Any] | None = None
) -> DictConfig:
    """Returns a copy of `cfg` whose `model` group is replaced by the YAML at `model_config`.

    `cfg_overrides` is merged on top (e.g. the `training.loss` block of an experiment file):
    `build_model` cross-checks `model.deep_supervision_levels` against `training.loss`, so a
    model built outside its experiment config must carry the experiment's loss settings too.

    The two in-house models differ in nothing the evaluation reads except `cfg.model`
    (sliding-window and post-processing settings are shared), so swapping that one group is
    enough to build either from the same composed config.
    """
    # Merge UNRESOLVED, then resolve: model YAMLs interpolate from the rest of the config
    # (e.g. `in_channels: ${data.in_channels}`), so they can only be resolved in place.
    container = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(container, dict)
    container["model"] = OmegaConf.to_container(OmegaConf.load(model_config), resolve=False)
    merged = OmegaConf.create(container)
    if cfg_overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(dict(cfg_overrides)))
    OmegaConf.resolve(merged)
    return merged


class InhouseBackend:
    """An in-house model (`neurovision`, `baseline_unet3d`) run with the evaluation path."""

    chunk_size = 1  # whole volumes are large; one case at a time on the GPU

    def __init__(
        self,
        cfg: DictConfig,
        model_config: Path,
        checkpoint: Path,
        device: torch.device,
        cfg_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Builds the model and loads its checkpoint via `scripts/evaluate.py::load_eval_model`.

        Args:
            cfg: The full composed config.
            model_config: YAML of the model group (e.g. `configs/model/neurovision.yaml`).
            checkpoint: Checkpoint file.
            device: Resolved device.
            cfg_overrides: Extra config merged before building (see `build_inhouse_cfg`).
        """
        self.cfg = build_inhouse_cfg(cfg, model_config, cfg_overrides)
        self.device = device
        if not checkpoint.is_file():
            raise FileNotFoundError(f"severity_ladder: no checkpoint at {checkpoint}.")
        self.model, _ = _EVAL.load_eval_model(self.cfg, checkpoint, device)

    def predict(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Sliding-window logits for each `(4, D, H, W)` image.

        Returns:
            `{case_id: float32 logits (3, D, H, W)}` on CPU. AMP follows the device (on for CUDA,
            off for CPU), exactly as in `scripts/evaluate.py`.
        """
        out: dict[str, np.ndarray] = {}
        for case_id, image in images.items():
            batch = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
            # sliding_window_predict calls model.eval() and torch.no_grad() itself.
            logits = sliding_window_predict(self.model, batch, self.cfg, self.device)
            out[case_id] = logits[0].float().cpu().numpy()
        return out


def resolve_nnunet_model(nn_cfg: Mapping[str, Any], device: str) -> tuple[Path, str]:
    """Runs the same preflight checks as `scripts/nnunet_predict_probs.py` on the model folder.

    Returns:
        `(model_folder, file_ending)`.
    """
    root = nn_cfg.get("model_search_root")
    if root is None:
        raise ValueError(
            "severity_ladder.nnunet.model_search_root is not set. Example: '/kaggle/input'."
        )
    checkpoint_name = str(nn_cfg["checkpoint_name"])
    if checkpoint_name != _NNUNET._ALLOWED_CHECKPOINT:
        raise ValueError(f"nnunet.checkpoint_name must be {_NNUNET._ALLOWED_CHECKPOINT!r}.")
    model_folder = _NNUNET.find_model_folder(Path(str(root)))
    meta = _NNUNET.read_checkpoint_meta(model_folder / "fold_all" / checkpoint_name)
    _NNUNET.check_checkpoint_identity(meta, int(nn_cfg["expect_epoch"]))
    file_ending = _NNUNET.read_file_ending(model_folder)
    _NNUNET.check_region_mode(read_json(model_folder / "dataset.json"))
    _NNUNET.check_reader_writer(model_folder)
    if device == "cuda":
        _NNUNET._require_cuda_or_raise()
    return model_folder, file_ending


class NnunetBackend:
    """The Gate A nnU-Net, TTA off (note 61's `--disable_tta` arm), via the existing glue."""

    def __init__(
        self,
        model_folder: Path,
        file_ending: str,
        prep_dir: Path,
        work_dir: Path,
        device: str,
        chunk_size: int,
        label_check_tol: float,
    ) -> None:
        """Stores what one batched prediction needs.

        Args:
            model_folder: nnU-Net trainer folder (`-m`).
            file_ending: `.nii` or `.nii.gz`, from the model's dataset.json.
            prep_dir: Preprocessed tree (only `meta.json` is read from it here).
            work_dir: Scratch root for the transient inputs and raw outputs.
            device: `"cuda"` or `"cpu"` for nnU-Net's `-device`.
            chunk_size: Cases per nnU-Net call (bounds transient disk: ~50 MB/case/copy).
            label_check_tol: Largest allowed per-region mismatch with nnU-Net's own label map.
        """
        self.model_folder = model_folder
        self.file_ending = file_ending
        self.prep_dir = prep_dir
        self.work_dir = work_dir
        self.device = device
        self.chunk_size = chunk_size
        self.label_check_tol = label_check_tol

    def predict(self, images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One batched nnU-Net call over this chunk's corrupted images.

        The corrupted images are written to a TEMPORARY prep tree (image.npy + a copy of
        meta.json) so `write_nnunet_inputs` can be reused unchanged; that tree, the NIfTI inputs
        and nnU-Net's raw outputs are all deleted before returning.

        Returns:
            `{case_id: float32 logits (3, D, H, W)}`; fp16-rounded probabilities converted by
            `convert_case` (channel order and axis order handled there).
        """
        case_ids = list(images)
        tmp_prep = self.work_dir / "tmp_prep"
        input_dir = self.work_dir / "input"
        raw_dir = self.work_dir / "raw_out"
        for scratch in (tmp_prep, input_dir, raw_dir):
            shutil.rmtree(scratch, ignore_errors=True)
            ensure_dir(scratch)
        try:
            for case_id in case_ids:
                case_dir = ensure_dir(tmp_prep / case_id)
                np.save(case_dir / "image.npy", np.asarray(images[case_id], dtype=np.float32))
                shutil.copyfile(
                    self.prep_dir / case_id / "meta.json", case_dir / "meta.json"
                )  # affine + bbox for the NIfTI header
            _NNUNET.write_nnunet_inputs(tmp_prep, case_ids, input_dir, self.file_ending)
            shutil.rmtree(tmp_prep)  # corrupted volumes live on disk only until here

            cmd = _NNUNET.build_predict_command(
                input_dir, raw_dir, self.model_folder, self.device, True  # disable_tta=True
            )
            cmd.append("--save_probabilities")
            _NNUNET.run_prediction(cmd)

            out: dict[str, np.ndarray] = {}
            for case_id in case_ids:
                shape = tuple(images[case_id].shape[1:])
                logits, _ = _NNUNET.convert_case(
                    case_id, raw_dir, shape, self.file_ending, self.label_check_tol
                )
                out[case_id] = logits.astype(np.float32)
            return out
        finally:
            for scratch in (tmp_prep, input_dir, raw_dir):
                shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# Gate signals computed from logits
# ---------------------------------------------------------------------------


class GateSignalComputer:
    """Computes the deployed gate's logit-derived signals for one case.

    The QC model and the conformal-band calculation are the project's own functions; this class
    only wires them to in-memory arrays instead of saved eval directories.
    """

    def __init__(
        self,
        regions: Sequence[str],
        fitted_thresholds: Mapping[str, float] | None,
        qc_model: torch.nn.Module | None,
        qc_target_shape: tuple[int, int, int],
        qc_modality_index: int,
        device: torch.device,
    ) -> None:
        """Args:
        regions: Gate regions (`cfg.clinical.gatekeeper.regions`), e.g. `("WT", "TC")`.
        fitted_thresholds: `{region: tau}` conformal thresholds for the deployed alpha, or
            `None` (the band columns are then NaN).
        qc_model: Loaded `SegQC` in eval mode, or `None` (predicted-Dice columns are NaN).
        qc_target_shape: Packed volume size the QC model was trained at.
        qc_modality_index: Which image channel the QC model reads.
        device: Device for the QC forward pass.
        """
        self.regions = tuple(regions)
        self.fitted = fitted_thresholds
        self.qc_model = qc_model
        self.qc_target_shape = qc_target_shape
        self.qc_modality_index = qc_modality_index
        self.device = device

    def predicted_dice(
        self, image: np.ndarray, mask: np.ndarray, label: np.ndarray, entropy: np.ndarray
    ) -> dict[str, float]:
        """QC-model Dice estimate per gate region for the model's own deployed mask.

        Args:
            image: The (corrupted) `(4, D, H, W)` image the segmentation model saw.
            mask: Deployed post-processed mask `(3, D, H, W)`.
            label: Class map `(D, H, W)` (carried by `CaseArrays`, never read by the QC model).
            entropy: Per-voxel entropy `(3, D, H, W)` in nats.
        """
        if self.qc_model is None:
            return {f"predicted_dice_{r}": float("nan") for r in self.regions}
        arrays = CaseArrays(
            pred_mask=mask.astype(np.uint8),
            label=label,
            image_modality=image[self.qc_modality_index].astype(np.float32),
            entropy=entropy,
        )
        out: dict[str, float] = {}
        with torch.no_grad():
            for region in self.regions:
                channel = REGION_NAMES.index(region)
                sample = pack_sample(arrays, arrays.pred_mask, channel, self.qc_target_shape)
                logit = self.qc_model(sample.unsqueeze(0).to(self.device))
                out[f"predicted_dice_{region}"] = float(predicted_dice(logit).item())
        return out

    def conformal_band(self, curves: Mapping[str, Any]) -> dict[str, float]:
        """Mask inflation at each region's fitted conformal threshold (the gate's band width)."""
        if self.fitted is None:
            return {f"conformal_band_{r}": float("nan") for r in self.regions}
        widths = case_conformal_band_widths(
            {r: [curves[r]] for r in self.regions}, self.fitted, reference_threshold=0.5
        )
        return {f"conformal_band_{r}": next(iter(widths[r].values())) for r in self.regions}


def build_gate_signals(
    sl_cfg: Mapping[str, Any],
    cfg: DictConfig,
    model_entry: Mapping[str, Any],
    repo_root: Path,
    device: torch.device,
) -> GateSignalComputer:
    """Builds the gate-signal computer for one model from config (missing pieces become NaN)."""
    regions = [str(r) for r in sl_cfg["gate"]["regions"]]
    alpha = float(sl_cfg["gate"]["conformal_alpha"])

    fitted: dict[str, float] | None = None
    fit_path = model_entry.get("conformal_fit")
    if fit_path is None:
        logger.warning("severity_ladder: no conformal_fit configured; conformal_band is NaN.")
    else:
        fitted = resolve_fitted_thresholds(read_json(_resolve(repo_root, fit_path)), regions, alpha)

    qc_model: torch.nn.Module | None = None
    qc_cfg = sl_cfg["qc"]
    if qc_cfg.get("enabled", False):
        if qc_cfg.get("checkpoint") is None:
            logger.warning("severity_ladder: qc.enabled but no checkpoint; predicted_dice is NaN.")
        else:
            model_cfg = OmegaConf.create(
                {"model": OmegaConf.load(_resolve(repo_root, qc_cfg["model_config"]))}
            )
            qc_model = build_segqc(model_cfg).to(device)
            load_checkpoint(
                _resolve(repo_root, qc_cfg["checkpoint"]),
                qc_model,
                optimizer=None,
                map_location=str(device),
                restore_rng=False,
            )
            qc_model.eval()
    else:
        logger.warning("severity_ladder: qc.enabled is false; predicted_dice is NaN.")

    return GateSignalComputer(
        regions,
        fitted,
        qc_model,
        tuple(int(v) for v in cfg.analysis.qc.target_shape),
        int(cfg.analysis.qc.modality_index),
        device,
    )


# ---------------------------------------------------------------------------
# Scoring one case
# ---------------------------------------------------------------------------


def score_case(
    case_id: str,
    logits: np.ndarray,
    image: np.ndarray,
    label: np.ndarray,
    spacing: Sequence[float],
    cfg: DictConfig,
    gate: GateSignalComputer,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Turns one case's logits into a metrics row and per-region conformal loss curves.

    Metrics are computed on the deployed post-processed mask (`postprocess_logits`). Curves,
    entropy and the QC input use logits rounded through fp16, the precision every saved-logit
    analysis in this project works at, so numbers are comparable with the clean results.

    Args:
        case_id: Case identifier.
        logits: Raw logits `(3, D, H, W)` float32, channels (ET, TC, WT).
        image: The corrupted `(4, D, H, W)` image the model saw.
        label: Class map `(D, H, W)` with values 0..3.
        spacing: Voxel spacing in mm (for HD95).
        cfg: Full config (`inference.postprocess` is read).
        gate: Gate-signal computer.

    Returns:
        `(row, curves)`: a flat dict of scalars and `{region: CaseLossCurve}`.
    """
    logits_t = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32)).unsqueeze(0)
    regions = postprocess_logits(logits_t, cfg)  # (1, 3, D, H, W), CPU
    target = classes_to_regions(torch.from_numpy(label))  # (1, 3, D, H, W)
    # Everything is on CPU: never hand CUDA tensors to MONAI's HD95 (it routes through CuPy).
    metrics = compute_case_metrics(regions, target, spacing=spacing)

    row: dict[str, float] = {}
    for name in REGION_NAMES:
        row[f"dice_{name}"] = float(metrics[f"dice_{name}"])
        row[f"hd95_{name}"] = float(metrics[f"hd95_{name}"])
        row[f"gt_empty_{name}"] = float(metrics[f"gt_empty_{name}"])
    voxel_mm3 = float(np.prod([float(s) for s in spacing]))
    row["voxel_mm3"] = voxel_mm3
    for i, name in enumerate(REGION_NAMES):
        row[f"pred_voxels_{name}"] = float(regions[0, i].sum().item())

    logits_q = logits_t.half().float()  # fp16 round trip (saved-logit precision), done in fp32
    prob = torch.sigmoid(logits_q)[0].numpy()
    target_np = target[0].numpy()
    curves = {
        name: case_loss_curve(
            prob[i],
            target_np[i],
            case_id=case_id,
            region=name,
            thresholds=EXTENDED_THRESHOLDS,
        )
        for i, name in enumerate(REGION_NAMES)
    }
    for name in REGION_NAMES:
        row[f"gt_voxels_{name}"] = float(curves[name].gt_voxels)

    entropy_scalars = case_entropy_scalars(logits_q[0].numpy())
    for name in REGION_NAMES:
        row[f"ent_mean_fg_{name}"] = entropy_scalars[f"ent_mean_fg_{name}"]
    row["ent_mean_fg_mean"] = entropy_scalars["ent_mean_fg_mean"]

    entropy = entropy_from_logits(logits_q[0]).numpy()
    row.update(
        gate.predicted_dice(image, regions[0].numpy(), label, entropy)  # the QC model's estimate
    )
    row.update(gate.conformal_band(curves))
    return row, curves


# ---------------------------------------------------------------------------
# One shard
# ---------------------------------------------------------------------------


def _resolve(repo_root: Path, path: str | Path) -> Path:
    """Relative config paths resolve against the repo root; absolute ones are kept."""
    p = Path(str(path))
    return p if p.is_absolute() else Path(repo_root) / p


def _check_deadline(deadline: float | None) -> None:
    """Raises `BudgetExhausted` if the wall-clock deadline has passed."""
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExhausted()


def run_shard(
    shard: Shard,
    *,
    out_dir: Path,
    case_ids: Sequence[str],
    prep_dir: Path,
    cfg: DictConfig,
    backend: Any,
    gate: GateSignalComputer,
    base_seed: int,
    deadline: float | None,
) -> float | None:
    """Runs one shard, or skips it if it is already DONE.

    Args:
        shard: The work unit.
        out_dir: Output root.
        case_ids: Cases to run, in order.
        prep_dir: Preprocessed tree.
        cfg: Full config.
        backend: `InhouseBackend` or `NnunetBackend` (anything with `chunk_size` and `predict`).
        gate: Gate-signal computer for this model.
        base_seed: Base seed for `case_seed`.
        deadline: `time.monotonic()` value after which to stop, or `None`.

    Returns:
        Mean seconds per case, or `None` if the shard was already DONE (skipped).

    Raises:
        BudgetExhausted: Deadline passed mid-shard; the partial shard has been discarded.
        RuntimeError: An existing DONE shard was made for a different case list.
    """
    final_dir = Path(out_dir) / shard.rel_dir
    partial_dir = final_dir.with_name(final_dir.name + _PARTIAL_SUFFIX)

    if (final_dir / DONE_MARKER).is_file():
        recorded = read_json(final_dir / "shard_meta.json")["case_ids"]
        if list(recorded) != list(case_ids):
            raise RuntimeError(
                f"{final_dir} is DONE but for a different case list ({len(recorded)} cases vs "
                f"{len(case_ids)} now). Probe/limited runs belong in their own out_dir."
            )
        logger.info("severity_ladder: skipping DONE shard %s", shard.rel_dir)
        return None

    _check_deadline(deadline)
    shutil.rmtree(partial_dir, ignore_errors=True)  # debris from a killed run
    shutil.rmtree(final_dir, ignore_errors=True)  # a final dir without DONE is not a result
    ensure_dir(partial_dir)

    rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    curves_by_region: dict[str, list[Any]] = {name: [] for name in REGION_NAMES}
    chunk_size = max(1, int(backend.chunk_size))
    chunks = [list(case_ids[i : i + chunk_size]) for i in range(0, len(case_ids), chunk_size)]

    try:
        for chunk in chunks:
            _check_deadline(deadline)
            start = time.monotonic()
            images: dict[str, np.ndarray] = {}
            seeds: dict[str, int] = {}
            for case_id in chunk:
                clean = np.load(prep_dir / case_id / "image.npy").astype(np.float32)
                # Level 0 is "none": corrupt_image returns the image unchanged.
                seeds[case_id] = case_seed(base_seed, case_id, shard.kind, shard.level_index)
                images[case_id] = corrupt_image(clean, shard.kind, shard.severity, seeds[case_id])
            logits_by_case = backend.predict(images)

            scored: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
            for case_id in chunk:
                label = np.load(prep_dir / case_id / "label.npy")
                spacing = read_json(prep_dir / case_id / "meta.json")["spacing"]
                row, curves = score_case(
                    case_id, logits_by_case[case_id], images[case_id], label, spacing, cfg, gate
                )
                row = {
                    "case_id": case_id,
                    "model": shard.model,
                    "kind": shard.kind,
                    "level_index": shard.level_index,
                    "severity": shard.severity,
                    "seed": seeds[case_id],
                    **row,
                }
                ood = {"case_id": case_id, **extract_features(images[case_id])}
                scored.append((case_id, row, ood))
                for name in REGION_NAMES:
                    curves_by_region[name].append(curves[name])
            per_case_s = (time.monotonic() - start) / len(chunk)
            for _, row, ood in scored:
                row["seconds"] = per_case_s
                rows.append(row)
                ood_rows.append(ood)
            logger.info("severity_ladder: %s %d/%d cases", shard.rel_dir, len(rows), len(case_ids))
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)  # never leave a half-written shard
        raise

    per_case = pd.DataFrame(rows)
    per_case.to_csv(partial_dir / "per_case.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(partial_dir / "ood_features.csv", index=False)
    _CONFORMAL._write_curves_npz(curves_by_region, EXTENDED_THRESHOLDS, partial_dir / "curves.npz")
    manifest = {
        **_CONFORMAL._manifest_request(
            final_dir, prep_dir, list(REGION_NAMES), EXTENDED_THRESHOLDS
        ),
        "n_cases": len(case_ids),
        "gt_voxels": {
            c.case_id: {name: curves_by_region[name][i].gt_voxels for name in REGION_NAMES}
            for i, c in enumerate(curves_by_region[REGION_NAMES[0]])
        },
    }
    write_json(manifest, partial_dir / "curves_manifest.json")
    s_per_case = float(per_case["seconds"].mean())
    write_json(
        {
            "model": shard.model,
            "kind": shard.kind,
            "level_index": shard.level_index,
            "severity": shard.severity,
            "base_seed": base_seed,
            "case_ids": list(case_ids),
            "s_per_case": s_per_case,
        },
        partial_dir / "shard_meta.json",
    )
    (partial_dir / DONE_MARKER).write_text("done\n", encoding="utf-8")
    partial_dir.rename(final_dir)  # the atomic step: DONE only ever appears with a full shard

    logger.info(
        "NVX_SHARD_DONE %s %s %d %d %.3f",
        shard.model,
        shard.kind,
        shard.level_index,
        len(case_ids),
        s_per_case,
    )
    return s_per_case


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


def bridge_check(
    level0_csv: Path, reference_csv: Path, tol: float
) -> tuple[bool, dict[str, float]]:
    """Compares re-inferred clean per-case Dice with the saved clean evaluation.

    Args:
        level0_csv: This run's level-0 `per_case.csv`.
        reference_csv: The saved clean test `per_case_metrics.csv` (indexed by `case_id`).
        tol: Largest allowed mean absolute difference per region.

    Returns:
        `(passed, {region: mean absolute Dice difference})`. A case missing from the reference,
        or a NaN difference, fails the bridge (never a silent pass).
    """
    ours = pd.read_csv(level0_csv).set_index("case_id")
    ref = pd.read_csv(reference_csv).set_index("case_id")
    missing = [c for c in ours.index if c not in ref.index]
    diffs: dict[str, float] = {}
    passed = not missing
    for name in REGION_NAMES:
        if missing:
            diffs[name] = float("nan")
            continue
        d = (ours[f"dice_{name}"] - ref.loc[ours.index, f"dice_{name}"]).abs()
        diffs[name] = float(d.mean())
        passed = passed and bool(np.isfinite(diffs[name])) and diffs[name] <= tol
    return passed, diffs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def select_case_ids(sl_cfg: Mapping[str, Any], splits_path: Path) -> list[str]:
    """The case list: the configured split, cut to `probe_cases` (probe) or `case_limit`."""
    splits = load_splits(str(splits_path))
    split = str(sl_cfg["split"])
    if split not in splits:
        raise ValueError(f"severity_ladder.split={split!r} not in {sorted(splits)}.")
    case_ids = list(splits[split])
    limit = int(sl_cfg["probe_cases"]) if sl_cfg["probe"] else sl_cfg.get("case_limit")
    if limit is not None:
        case_ids = case_ids[: int(limit)]
    if not case_ids:
        raise ValueError(f"Split {split!r} has no cases.")
    return case_ids


def build_backend(
    name: str,
    entry: Mapping[str, Any],
    sl_cfg: Mapping[str, Any],
    cfg: DictConfig,
    prep_dir: Path,
    repo_root: Path,
    device: torch.device,
) -> Any:
    """Builds the backend named by `entry["backend"]` (`inhouse` or `nnunet`)."""
    backend = entry["backend"]
    if backend == "inhouse":
        return InhouseBackend(
            cfg,
            _resolve(repo_root, entry["model_config"]),
            _resolve(repo_root, entry["checkpoint"]),
            device,
            OmegaConf.to_container(entry["cfg_overrides"]) if entry.get("cfg_overrides") else None,
        )
    if backend == "nnunet":
        nn_cfg = sl_cfg["nnunet"]
        if nn_cfg.get("work_dir") is None:
            raise ValueError("severity_ladder.nnunet.work_dir is not set.")
        device_str = device.type if device.type in ("cuda", "cpu") else "cpu"
        model_folder, file_ending = resolve_nnunet_model(nn_cfg, device_str)
        return NnunetBackend(
            model_folder,
            file_ending,
            prep_dir,
            Path(str(nn_cfg["work_dir"])),
            device_str,
            int(nn_cfg["chunk_size"]),
            float(nn_cfg["label_check_tol"]),
        )
    raise ValueError(f"model {name!r}: backend must be 'inhouse' or 'nnunet', got {backend!r}.")


def run_severity_ladder(cfg: DictConfig, repo_root: Path) -> dict[str, Any]:
    """Runs every planned shard (resuming finished ones) within the time budget.

    Args:
        cfg: The full composed config (`cfg.analysis.severity_ladder` is the block).
        repo_root: Root against which relative config paths resolve.

    Returns:
        `{"completed": [...], "skipped": [...], "bridge": {model: bool}, "stopped_models":
        [...], "budget_exhausted": bool}` (shard names are relative dir strings).
    """
    sl_cfg = cfg.analysis.severity_ladder
    out_root = _resolve(repo_root, sl_cfg["out_dir"])
    if sl_cfg["probe"]:
        # A 5-case probe must never leave DONE markers the main run would mistake for real shards.
        out_root = out_root / "probe"
    ensure_dir(out_root)

    prep_dir = _resolve(repo_root, sl_cfg["prep_dir"])
    case_ids = select_case_ids(sl_cfg, _resolve(repo_root, sl_cfg["splits_path"]))
    models = [str(m) for m in sl_cfg["models_to_run"]]
    unknown = [m for m in models if m not in sl_cfg["models"]]
    if unknown:
        raise ValueError(f"models_to_run names unconfigured model(s) {unknown}.")
    for m in models:  # fail early, before any shard: a bridge needs its reference
        ref = sl_cfg["models"][m].get("reference_metrics")
        if ref is None or not _resolve(repo_root, ref).is_file():
            raise FileNotFoundError(f"model {m!r}: reference_metrics {ref!r} not found.")

    device = get_device(cfg)
    max_hours = sl_cfg.get("max_hours")
    deadline = None if max_hours is None else time.monotonic() + float(max_hours) * 3600.0
    base_seed = int(sl_cfg["base_seed"])
    tol = float(sl_cfg["bridge_tol"])
    shards = plan_shards(OmegaConf.to_container(sl_cfg, resolve=True), models)
    logger.info(
        "severity_ladder: %d shard(s), %d case(s), device=%s, probe=%s",
        len(shards),
        len(case_ids),
        device,
        bool(sl_cfg["probe"]),
    )

    backends: dict[str, Any] = {}
    gates: dict[str, GateSignalComputer] = {}
    bridge: dict[str, bool] = {}
    stopped: set[str] = set()
    completed: list[str] = []
    skipped: list[str] = []
    timings: dict[str, float] = {}
    exhausted = False

    for shard in shards:
        if shard.model in stopped:
            continue
        if shard.kind != "none" and shard.model not in bridge:
            continue  # unreachable by plan order; guards a model whose level 0 never ran
        entry = sl_cfg["models"][shard.model]
        try:
            _check_deadline(deadline)
            if shard.model not in backends:
                backends[shard.model] = build_backend(
                    shard.model, entry, sl_cfg, cfg, prep_dir, repo_root, device
                )
                gates[shard.model] = build_gate_signals(sl_cfg, cfg, entry, repo_root, device)
            s_per_case = run_shard(
                shard,
                out_dir=out_root,
                case_ids=case_ids,
                prep_dir=prep_dir,
                cfg=cfg,
                backend=backends[shard.model],
                gate=gates[shard.model],
                base_seed=base_seed,
                deadline=deadline,
            )
        except BudgetExhausted:
            logger.warning(
                "NVX_BUDGET_EXHAUSTED: max_hours=%s reached before finishing %s; nothing partial "
                "was kept. Re-run to resume.",
                max_hours,
                shard.rel_dir,
            )
            exhausted = True
            break
        (skipped if s_per_case is None else completed).append(str(shard.rel_dir))
        if s_per_case is not None:
            timings[str(shard.rel_dir)] = s_per_case

        if shard.kind == "none":
            passed, diffs = bridge_check(
                out_root / shard.rel_dir / "per_case.csv",
                _resolve(repo_root, entry["reference_metrics"]),
                tol,
            )
            bridge[shard.model] = passed
            detail = " ".join(f"{k}={v:.5f}" for k, v in diffs.items())
            write_json(
                {"passed": passed, "tol": tol, "mean_abs_dice_diff": diffs},
                out_root / shard.model / "bridge.json",
            )
            if passed:
                logger.info("NVX_BRIDGE: PASS model=%s tol=%s %s", shard.model, tol, detail)
            else:
                logger.error(
                    "NVX_BRIDGE: FAIL model=%s tol=%s %s -- no corrupted shard will run for "
                    "this model.",
                    shard.model,
                    tol,
                    detail,
                )
                stopped.add(shard.model)

    write_json({"s_per_case": timings}, out_root / "run_summary.json")
    logger.info("NVX_RUN_COMPLETE completed=%d skipped=%d", len(completed), len(skipped))
    return {
        "completed": completed,
        "skipped": skipped,
        "bridge": bridge,
        "stopped_models": sorted(stopped),
        "budget_exhausted": exhausted,
    }


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs the severity ladder per `cfg.analysis.severity_ladder`.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    setup_logging(level="INFO")
    set_seed(cfg.seed)
    run_severity_ladder(cfg, _SCRIPTS_DIR.parent)


if __name__ == "__main__":
    main()
