"""SRI24 atlas loading, put into the BraTS voxel index frame.

This module loads the SRI24 anatomical atlas (parcellation + tissue maps) and
reorients it so its voxel indices line up exactly with a preprocessed BraTS
case's voxel indices. Downstream code can then intersect a tumour mask with
an anatomical structure by plain boolean indexing -- no registration and no
resampling.

The reorientation is a **pure index transform**: an axis permutation with,
per axis, an optional exact reversal. `docs/research/phase0_atlas_findings.md`
measured this by solving each SRI24 file's own affine against the BraTS
target affine and confirmed the required transform is always one of these --
never a rotation, shear, or non-unit scale. That measurement is why
`solve_index_transform` raises loudly on anything else instead of silently
interpolating: this pipeline's "registration-free by construction" property
depends on every file actually satisfying that.

Two orientation facts measured there are the whole reason the transform is
solved **per file**, never once for the whole atlas distribution:
`tzo116plus.nii` / `lpba40.nii` / `tissues.nii` / `suptent.nii` / `spgr.nii`
are anterior-posterior mirrored relative to BraTS, while `pbmap_GM/WM/CSF.nii`
are ALSO left-right mirrored relative to every other file in the very same
SRI24 distribution.

This module is pure array + text arithmetic: no model, no checkpoint, and no
dependency on the deep-learning stack, so it (and anything that imports it)
stays importable in an environment with none of that installed -- see
`tests/test_atlas.py::test_atlas_module_does_not_import_the_deep_learning_stack`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from omegaconf import DictConfig

__all__ = [
    "AtlasStructure",
    "AtlasLabels",
    "Atlas",
    "IndexTransform",
    "parse_lut",
    "solve_index_transform",
    "apply_index_transform",
    "reorient_to_target",
    "load_nifti",
    "load_atlas",
]

logger = logging.getLogger(__name__)

_VALID_SIGN_VALUES = (-1.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Index transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IndexTransform:
    """A pure axis permutation with optional per-axis reversal. No interpolation.

    Attributes:
        perm: `perm[b] = a` means output axis `b` reads from input axis `a`.
        flip: `flip[b]` is whether input axis `perm[b]` is reversed before
            being placed at output axis `b`.
    """

    perm: tuple[int, int, int]
    flip: tuple[bool, bool, bool]


def solve_index_transform(
    src_affine: np.ndarray,
    dst_affine: np.ndarray,
    src_shape: Sequence[int],
    dst_shape: Sequence[int],
) -> IndexTransform:
    """Solves the pure-flip index transform that maps `src_affine` onto `dst_affine`.

    The world position of a voxel is `affine @ [i, j, k, 1]`. Setting the two
    world positions equal gives, for every destination index `i_dst`, the
    source index `i_src` whose world position matches:

        M = inv(src_affine) @ dst_affine
        i_src = M[:3, :3] @ i_dst + M[:3, 3]

    `M` is rounded to 6 decimals before inspection, because these affines are
    exact but arrive through float32 NIfTI headers.

    Args:
        src_affine: Source 4x4 affine.
        dst_affine: Destination 4x4 affine.
        src_shape: Source spatial shape, `(D, H, W)`.
        dst_shape: Destination spatial shape, `(D, H, W)`.

    Returns:
        The `IndexTransform` that reproduces `M` exactly via a transpose plus
        per-axis flips.

    Raises:
        ValueError: If `M[:3, :3]` is not a signed permutation matrix (every
            entry in `{-1, 0, +1}`, exactly one non-zero per row and column)
            -- i.e. the affine pair requires real resampling (a rotation,
            shear, or non-unit scale) rather than a pure index flip. Also
            raised if `M[:3, 3]` is not all-integer, if a `+1` axis has a
            non-zero offset, if a `-1` axis's offset is not
            `src_shape[a] - 1`, or if the two axes paired by the permutation
            disagree in size.
    """
    src_affine = np.asarray(src_affine, dtype=np.float64)
    dst_affine = np.asarray(dst_affine, dtype=np.float64)
    m = np.round(np.linalg.inv(src_affine) @ dst_affine, 6)

    rot = m[:3, :3]
    trans = m[:3, 3]

    is_valid_sign = np.zeros_like(rot, dtype=bool)
    for value in _VALID_SIGN_VALUES:
        is_valid_sign |= np.isclose(rot, value, atol=1e-6)
    if not is_valid_sign.all():
        bad_entries = rot[~is_valid_sign].tolist()
        raise ValueError(
            f"solve_index_transform: M[:3, :3] has entries outside {{-1, 0, 1}}: "
            f"{bad_entries}. This affine pair requires real resampling (a rotation, shear, "
            "or non-unit scale), which this registration-free pipeline refuses to perform "
            "silently."
        )

    nonzero = ~np.isclose(rot, 0.0, atol=1e-6)
    row_counts = nonzero.sum(axis=1)
    col_counts = nonzero.sum(axis=0)
    if not (np.all(row_counts == 1) and np.all(col_counts == 1)):
        raise ValueError(
            f"solve_index_transform: M[:3, :3] is not a signed permutation matrix (row "
            f"non-zero counts {row_counts.tolist()}, column non-zero counts "
            f"{col_counts.tolist()}); expected exactly one non-zero entry per row and per "
            "column."
        )

    if not np.allclose(trans, np.round(trans), atol=1e-6):
        raise ValueError(
            f"solve_index_transform: M[:3, 3] is not integer-valued: {trans.tolist()}."
        )
    trans_int = np.round(trans).astype(np.int64)

    perm = [0, 0, 0]
    flip = [False, False, False]
    for a in range(3):
        b = int(np.nonzero(nonzero[a])[0][0])
        sign = rot[a, b]
        offset = int(trans_int[a])

        if np.isclose(sign, 1.0, atol=1e-6):
            if offset != 0:
                raise ValueError(
                    f"solve_index_transform: source axis {a} maps to destination axis {b} "
                    f"with scale +1, but its offset is {offset} != 0."
                )
            axis_flip = False
        else:
            expected_offset = int(src_shape[a]) - 1
            if offset != expected_offset:
                raise ValueError(
                    f"solve_index_transform: source axis {a} maps to destination axis {b} "
                    f"with scale -1, but its offset is {offset} != "
                    f"src_shape[{a}] - 1 = {expected_offset}."
                )
            axis_flip = True

        if int(src_shape[a]) != int(dst_shape[b]):
            raise ValueError(
                f"solve_index_transform: source axis {a} (size {src_shape[a]}) is paired "
                f"with destination axis {b} (size {dst_shape[b]}), but the sizes disagree."
            )

        perm[b] = a
        flip[b] = axis_flip

    return IndexTransform(perm=tuple(perm), flip=tuple(flip))


def apply_index_transform(array: np.ndarray, transform: IndexTransform) -> np.ndarray:
    """Applies a solved `IndexTransform` to an array. No interpolation anywhere.

    Args:
        array: `(D, H, W)` array in the source frame.
        transform: An `IndexTransform` from `solve_index_transform`.

    Returns:
        A C-contiguous `(D, H, W)` array in the destination frame.
    """
    out = np.transpose(array, transform.perm)
    for axis, do_flip in enumerate(transform.flip):
        if do_flip:
            out = np.flip(out, axis=axis)
    # np.flip returns a negative-stride VIEW; downstream indexing (np.isin,
    # boolean masking, saving) should not silently inherit that.
    return np.ascontiguousarray(out)


def reorient_to_target(
    array: np.ndarray,
    src_affine: np.ndarray,
    dst_affine: np.ndarray,
    dst_shape: Sequence[int],
) -> np.ndarray:
    """Solves and applies the index transform from `src_affine` to `dst_affine` in one call.

    Args:
        array: `(D, H, W)` array in the source frame.
        src_affine: Source 4x4 affine.
        dst_affine: Destination 4x4 affine.
        dst_shape: Expected destination spatial shape, `(D, H, W)`.

    Returns:
        A C-contiguous `(D, H, W)` array in the destination frame.

    Raises:
        ValueError: From `solve_index_transform`, or if the result's shape
            does not equal `dst_shape` (which would mean a bug in the solver
            rather than a bad affine, since the solver already checks
            per-axis sizes).
    """
    transform = solve_index_transform(src_affine, dst_affine, array.shape, dst_shape)
    out = apply_index_transform(array, transform)
    if out.shape != tuple(int(s) for s in dst_shape):
        raise ValueError(
            f"reorient_to_target: result shape {out.shape} != expected dst_shape "
            f"{tuple(dst_shape)}."
        )
    return out


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads a NIfTI file, squeezing the trailing singleton axis SRI24 files carry.

    Args:
        path: Path to a `.nii` (or `.nii.gz`) file.

    Returns:
        `(array, affine)`. `array` is 3-D, `dtype` unchanged from the file
        (never cast). `affine` is the 4x4 `numpy` array from the header.

    Raises:
        ValueError: If the loaded array is 4-D with a non-singleton trailing
            axis (unexpected for this atlas -- every SRI24 file measured is
            `(240, 240, 155, 1)`).
    """
    img = nib.load(str(path))
    array = np.asanyarray(img.dataobj)
    if array.ndim == 4:
        if array.shape[3] != 1:
            raise ValueError(
                f"load_nifti: {path} is 4-D with a non-singleton trailing axis "
                f"(shape {array.shape}); expected a squeezable (D, H, W, 1) volume."
            )
        array = array[..., 0]
    return array, np.asarray(img.affine)


# --------------------------------------------------------------------------- #
# LUT parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AtlasStructure:
    """One anatomical structure, after merging per-plane sub-labels to a parent.

    Attributes:
        name: Merged parent name, e.g. `"LateralVentricle_L"`.
        label_ids: Every raw label id merged into this structure, ascending.
        laterality: `"L"`, `"R"`, or `"midline"`, derived from `name`.
    """

    name: str
    label_ids: tuple[int, ...]
    laterality: str


@dataclass(frozen=True)
class AtlasLabels:
    """The parsed and merged SRI24 label table.

    Attributes:
        structures: Merged structures in LUT order of first appearance.
            Background (label 0) is excluded.
        unmapped_name: The name assigned by `name_for_id` to a label value
            present in a volume but absent from every structure's
            `label_ids`.
    """

    structures: tuple[AtlasStructure, ...]
    unmapped_name: str

    @property
    def names(self) -> tuple[str, ...]:
        """The merged structure names, in the same order as `structures`."""
        return tuple(s.name for s in self.structures)

    def by_name(self, name: str) -> AtlasStructure:
        """Looks up a structure by its merged name.

        Args:
            name: A merged structure name, e.g. `"LateralVentricle_L"`.

        Returns:
            The matching `AtlasStructure`.

        Raises:
            ValueError: If `name` is not a known structure.
        """
        for structure in self.structures:
            if structure.name == name:
                return structure
        example_names = list(self.names[:5])
        raise ValueError(
            f"AtlasLabels.by_name: unknown structure '{name}'. A few valid names: "
            f"{example_names}."
        )

    def name_for_id(self, label_id: int) -> str:
        """Merged structure name for one raw label id.

        Args:
            label_id: A raw SRI24 label value.

        Returns:
            The merged structure name, `""` for background (`label_id == 0`),
            or `unmapped_name` if `label_id` matches no structure.
        """
        if label_id == 0:
            return ""
        for structure in self.structures:
            if label_id in structure.label_ids:
                return structure.name
        return self.unmapped_name

    def lookup_array(self, max_id: int) -> np.ndarray:
        """Builds a dense label-id -> structure-index lookup table.

        Args:
            max_id: The highest raw label id the table must cover.

        Returns:
            `int32` array of shape `(max_id + 1,)`. Index 0 (background) and
            any id belonging to no structure hold `-1`; every other index
            `i` holds the position of its structure within `structures`.

        Raises:
            ValueError: If any structure's `label_ids` exceeds `max_id`.
        """
        table = np.full(max_id + 1, -1, dtype=np.int32)
        for structure_index, structure in enumerate(self.structures):
            for label_id in structure.label_ids:
                if label_id > max_id:
                    raise ValueError(
                        f"AtlasLabels.lookup_array: label id {label_id} (structure "
                        f"'{structure.name}') exceeds max_id={max_id}."
                    )
                table[label_id] = structure_index
        return table


def parse_lut(
    path: str | Path,
    merge_patterns: Sequence[str],
    *,
    unmapped_name: str,
) -> AtlasLabels:
    """Parses an SRI24 LUT text file and merges per-plane sub-labels to parents.

    Lines are split on generic whitespace (never `"\\t"` alone): the real
    `SRI24-tzo116plus.txt` uses tabs for labels 1-116 and spaces for labels
    201+, so a tab-only parser silently stops mapping at 116. Blank lines and
    lines whose first whitespace token is not purely digits (e.g. a header
    row) are skipped. Label id 0 (background) is skipped.

    Each row's raw name is merged to a parent name by applying every pattern
    in `merge_patterns`, in order, as `re.sub(pattern, "", name)` -- this
    strips the per-anatomical-plane suffixes (`_y48`, `_x111`, `_AP_0`, ...)
    that the "plus" labels (201+) carry.

    Args:
        path: Path to the LUT text file.
        merge_patterns: Regex patterns applied in order to strip per-plane
            suffixes from a raw name, giving its merged parent name.
        unmapped_name: Stored on the returned `AtlasLabels` for later use by
            `name_for_id`.

    Returns:
        An `AtlasLabels` with one `AtlasStructure` per merged parent name, in
        order of first appearance.

    Raises:
        ValueError: If the file yields zero structures, or if two different
            parent names both claim the same raw label id.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    order: list[str] = []
    ids_by_parent: dict[str, list[int]] = {}
    parent_by_id: dict[int, str] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 2 or not tokens[0].isdigit():
            continue

        label_id = int(tokens[0])
        if label_id == 0:
            continue
        raw_name = tokens[1]

        parent_name = raw_name
        for pattern in merge_patterns:
            parent_name = re.sub(pattern, "", parent_name)

        if label_id in parent_by_id and parent_by_id[label_id] != parent_name:
            raise ValueError(
                f"parse_lut: label id {label_id} is claimed by both "
                f"'{parent_by_id[label_id]}' and '{parent_name}'."
            )
        parent_by_id[label_id] = parent_name

        if parent_name not in ids_by_parent:
            ids_by_parent[parent_name] = []
            order.append(parent_name)
        ids_by_parent[parent_name].append(label_id)

    if not order:
        raise ValueError(f"parse_lut: {path} yielded zero structures.")

    structures = []
    for name in order:
        label_ids = tuple(sorted(set(ids_by_parent[name])))
        if name.endswith("_L"):
            laterality = "L"
        elif name.endswith("_R"):
            laterality = "R"
        else:
            laterality = "midline"
        structures.append(AtlasStructure(name=name, label_ids=label_ids, laterality=laterality))

    return AtlasLabels(structures=tuple(structures), unmapped_name=unmapped_name)


# --------------------------------------------------------------------------- #
# Atlas
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Atlas:
    """A loaded SRI24 atlas, already reoriented into the BraTS index frame.

    Attributes:
        parcellation: `(D, H, W)` `int16` raw label values, in the BraTS
            index frame.
        labels: The parsed and merged label table.
        tissue: `(D, H, W)` `uint8` hard tissue codes, already reoriented,
            or `None` if no tissue map was loaded.
        tissue_codes: Tissue name -> code, e.g. `{"CSF": 1, "GM": 2, "WM": 3}`.
        name: The parcellation variant name, e.g. `"tzo116plus"`.
        version: Atlas version string, from config.
        source: Atlas provenance string, from config.
        unmapped_ids: Raw label values present in `parcellation` (non-zero)
            with no row in `labels`.
    """

    parcellation: np.ndarray
    labels: AtlasLabels
    tissue: np.ndarray | None
    tissue_codes: dict[str, int]
    name: str
    version: str
    source: str
    unmapped_ids: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        """The `(D, H, W)` spatial shape of `parcellation`."""
        d, h, w = self.parcellation.shape
        return (int(d), int(h), int(w))

    def structure_mask(self, name: str) -> np.ndarray:
        """Boolean mask of one anatomical structure.

        Args:
            name: A merged structure name, e.g. `"LateralVentricle_L"`.

        Returns:
            `(D, H, W)` boolean array: the union of every raw label id
            merged into that structure.

        Raises:
            ValueError: If `name` is not a known structure.
        """
        structure = self.labels.by_name(name)
        return np.isin(self.parcellation, structure.label_ids)

    def tissue_mask(self, tissue: str) -> np.ndarray:
        """Boolean mask of one tissue class.

        Args:
            tissue: A tissue name, e.g. `"GM"`.

        Returns:
            `(D, H, W)` boolean array.

        Raises:
            ValueError: If no tissue map was loaded, or `tissue` is not a
                known tissue name.
        """
        if self.tissue is None:
            raise ValueError(
                "Atlas.tissue_mask: no tissue map was loaded for this atlas (tissue=None)."
            )
        if tissue not in self.tissue_codes:
            raise ValueError(
                f"Atlas.tissue_mask: unknown tissue '{tissue}'; valid tissues are "
                f"{list(self.tissue_codes)}."
            )
        return self.tissue == self.tissue_codes[tissue]

    def coverage(self) -> dict[str, int]:
        """Coverage summary backing the report's "N of M structures classified" line.

        Returns:
            Dict with `n_structures` (structures in `labels`),
            `n_labelled_voxels` (non-zero voxels assigned to a known
            structure), `n_unmapped_voxels` (non-zero voxels with no LUT
            row), and `n_unmapped_ids` (`len(unmapped_ids)`).
        """
        n_unmapped_voxels = int(np.isin(self.parcellation, self.unmapped_ids).sum())
        n_labelled_voxels = int((self.parcellation != 0).sum()) - n_unmapped_voxels
        return {
            "n_structures": len(self.labels.structures),
            "n_labelled_voxels": n_labelled_voxels,
            "n_unmapped_voxels": n_unmapped_voxels,
            "n_unmapped_ids": len(self.unmapped_ids),
        }


def _require_file(path: Path) -> None:
    """Raises `FileNotFoundError` naming `scripts/fetch_atlas.py` if `path` is missing."""
    if not path.is_file():
        raise FileNotFoundError(
            f"load_atlas: required atlas file {path} is missing. Run "
            "`python scripts/fetch_atlas.py` to download and extract SRI24 first."
        )


def load_atlas(cfg: DictConfig) -> Atlas:
    """Loads the SRI24 atlas per `configs/anatomy/sri24.yaml` and reorients it to BraTS.

    Every path is resolved from `cfg`; nothing here is hardcoded. See
    `docs/research/phase0_atlas_findings.md` for the measurements this
    function implements against.

    Args:
        cfg: The `anatomy` config node (mirroring
            `configs/anatomy/sri24.yaml`).

    Returns:
        A fully loaded, reoriented `Atlas`.

    Raises:
        FileNotFoundError: If the atlas root directory, or any required
            file, is absent -- naming `scripts/fetch_atlas.py`.
        ValueError: If `cfg.tissue.source` is neither `"tissues"` nor
            `"pbmap"`, or (via `reorient_to_target`) if any file's affine
            does not solve to a pure index flip.
    """
    root = Path(cfg.dir) / cfg.subdir
    if not root.is_dir():
        raise FileNotFoundError(
            f"load_atlas: atlas directory {root} does not exist. Run "
            "`python scripts/fetch_atlas.py` to download and extract SRI24 first."
        )

    dst_shape = tuple(int(s) for s in cfg.target.shape)
    dst_affine = np.array(cfg.target.affine, dtype=np.float64)

    # --- Parcellation --------------------------------------------------- #
    parc_path = root / cfg.parcellation.image
    _require_file(parc_path)
    parc_array, parc_affine = load_nifti(parc_path)
    parc_transform = solve_index_transform(parc_affine, dst_affine, parc_array.shape, dst_shape)
    logger.info(
        "load_atlas: %s solved to perm=%s flip=%s",
        parc_path.name,
        parc_transform.perm,
        parc_transform.flip,
    )
    parcellation = apply_index_transform(parc_array, parc_transform).astype(np.int16)

    lut_path = root / cfg.parcellation.lut
    _require_file(lut_path)
    labels = parse_lut(
        lut_path,
        cfg.parcellation.merge_patterns,
        unmapped_name=cfg.parcellation.unmapped_name,
    )

    present_ids = {int(v) for v in np.unique(parcellation) if v != 0}
    known_ids: set[int] = set()
    for structure in labels.structures:
        known_ids.update(structure.label_ids)
    unmapped_ids = tuple(sorted(present_ids - known_ids))
    if unmapped_ids:
        n_unmapped_voxels = int(np.isin(parcellation, unmapped_ids).sum())
        logger.warning(
            "load_atlas: %d label value(s) with no LUT row %s, covering %d voxels; mapped "
            "to '%s'.",
            len(unmapped_ids),
            unmapped_ids,
            n_unmapped_voxels,
            cfg.parcellation.unmapped_name,
        )

    # --- Tissue ----------------------------------------------------------- #
    tissue_codes = {str(k): int(v) for k, v in cfg.tissue.codes.items()}
    tissue_source = cfg.tissue.source

    if tissue_source == "tissues":
        tissue_path = root / cfg.tissue.image
        _require_file(tissue_path)
        tissue_array, tissue_affine = load_nifti(tissue_path)
        tissue_transform = solve_index_transform(
            tissue_affine, dst_affine, tissue_array.shape, dst_shape
        )
        logger.info(
            "load_atlas: %s solved to perm=%s flip=%s",
            tissue_path.name,
            tissue_transform.perm,
            tissue_transform.flip,
        )
        tissue = apply_index_transform(tissue_array, tissue_transform).astype(np.uint8)

    elif tissue_source == "pbmap":
        prob_by_name: dict[str, np.ndarray] = {}
        for tissue_name, filename in cfg.tissue.pbmap.items():
            prob_path = root / filename
            _require_file(prob_path)
            prob_array, prob_affine = load_nifti(prob_path)
            prob_transform = solve_index_transform(
                prob_affine, dst_affine, prob_array.shape, dst_shape
            )
            logger.info(
                "load_atlas: %s solved to perm=%s flip=%s",
                prob_path.name,
                prob_transform.perm,
                prob_transform.flip,
            )
            prob_by_name[str(tissue_name)] = apply_index_transform(
                prob_array, prob_transform
            ).astype(np.float32)

        ordered_names = list(prob_by_name)
        stacked = np.stack([prob_by_name[n] for n in ordered_names], axis=0)
        best_index = np.argmax(stacked, axis=0)
        best_prob = np.max(stacked, axis=0)

        tissue = np.zeros(dst_shape, dtype=np.uint8)
        confident = best_prob >= 0.5
        for i, tissue_name in enumerate(ordered_names):
            code = tissue_codes[tissue_name]
            tissue[confident & (best_index == i)] = code

    else:
        raise ValueError(
            f"load_atlas: unknown cfg.tissue.source '{tissue_source}'; expected 'tissues' or "
            "'pbmap'."
        )

    return Atlas(
        parcellation=parcellation,
        labels=labels,
        tissue=tissue,
        tissue_codes=tissue_codes,
        name=str(cfg.parcellation.name),
        version=str(cfg.version),
        source=str(cfg.source),
        unmapped_ids=unmapped_ids,
    )
