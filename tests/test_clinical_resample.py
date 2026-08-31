"""Tests for `neurovision.data.clinical_resample`.

Split the same way `tests/test_clinical_preprocess.py` is split, and following
that file's exact `pytest.importorskip("brainles_preprocessing")` idiom
inside each test body (never at module scope), so every test still gets
collected -- and skips cleanly, one at a time -- in the main `.venv`.

Part 1 (fast, mocked): `brainles_preprocessing.transform.Transform.apply` is
monkeypatched so these tests never call real ANTs. They check
`resample_mask_to_source`'s own argument validation and output validation
logic in well under a second total.

Part 2 (slow, real): ONE test that runs a genuine ANTs registration end to
end -- forward-warping a known synthetic lesion from native space into atlas
space, then calling `resample_mask_to_source` for real to warp it back, and
checking the round trip with a Dice score. Per this project's own testing
philosophy (CLAUDE.md: "an analysis fix is not verified by its unit tests"),
Part 1's mocks only prove the function's validation branches; only a real
ANTs round trip proves the geometry hazard `clinical_resample.py`'s module
docstring names is actually closed. Gated the same way as
`test_clinical_preprocess.py`'s execution-layer tests, plus an additional
`pytest.importorskip("ants")` -- `antspyx` is the PyPI package name, but the
importable module is `ants` (confirmed against the installed
`.venv-clinical/lib/python3.11/site-packages/ants/` -- `import antspyx`
raises `ModuleNotFoundError` there).

No real patient data, no real BraTS data. Every volume here is small and
synthetic, built with `nibabel`/`numpy`.
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    """Writes a small NIfTI volume, for tests that only need a file to exist."""
    nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), str(path))
    return path


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap between two boolean masks. 1.0 when both are empty."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


# ---------------------------------------------------------------------------
# Part 1 -- fast unit tests, mocked, no real registration.
# ---------------------------------------------------------------------------


def test_resample_mask_to_source_rejects_non_3d_mask(tmp_path: Path) -> None:
    """The ndim check runs before any filesystem work, so bogus paths are fine here."""
    pytest.importorskip("brainles_preprocessing")
    from neurovision.data.clinical_resample import resample_mask_to_source

    mask_4d = np.zeros((2, 4, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="D, H, W"):
        resample_mask_to_source(
            mask=mask_4d,
            atlas_affine=np.eye(4),
            transformations_dir=tmp_path / "does_not_matter",
            target_role="t1ce",
            target_native_path=tmp_path / "does_not_matter.nii.gz",
            out_dir=tmp_path / "out",
        )


def test_resample_mask_to_source_raises_when_role_transformations_missing(
    tmp_path: Path,
) -> None:
    """Missing role dir: the error must name the missing role and list what IS there.

    Checks for `'t1'` (quoted) rather than a bare substring match, because `'t1'` is
    also a substring of `'t1ce'` -- exactly the trap CLAUDE.md's traps list warns
    about ("any short token you match against a path or prose is a substring of
    some longer word there").
    """
    pytest.importorskip("brainles_preprocessing")
    from neurovision.data.clinical_resample import resample_mask_to_source

    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1").mkdir(parents=True)
    (transformations_dir / "flair").mkdir(parents=True)

    target_native_path = _write_nifti(tmp_path / "native.nii.gz", np.zeros((4, 4, 4), np.float32))

    with pytest.raises(FileNotFoundError) as excinfo:
        resample_mask_to_source(
            mask=np.zeros((4, 4, 4), dtype=np.uint8),
            atlas_affine=np.eye(4),
            transformations_dir=transformations_dir,
            target_role="t1ce",
            target_native_path=target_native_path,
            out_dir=tmp_path / "out",
        )

    message = str(excinfo.value)
    assert "'t1ce'" in message  # the missing role, named
    assert "'t1'" in message  # an available role, listed -- not a substring hit off 't1ce'
    assert "'flair'" in message


def test_resample_mask_to_source_raises_on_shape_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resampled output landing on the wrong grid must raise, not be trusted."""
    pytest.importorskip("brainles_preprocessing")
    from neurovision.data.clinical_resample import resample_mask_to_source

    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1ce").mkdir(parents=True)

    target_shape = (6, 6, 6)
    target_native_path = _write_nifti(
        tmp_path / "native.nii.gz", np.zeros(target_shape, dtype=np.float32)
    )

    def _fake_apply(
        self,
        target_modality_name,
        target_modality_img,
        moving_image,
        output_img_path,
        log_file_path,
        interpolator=None,
        inverse=False,
    ) -> None:
        wrong_shape = (4, 4, 4)  # deliberately NOT target_shape
        nib.save(
            nib.Nifti1Image(np.zeros(wrong_shape, dtype=np.float32), np.eye(4)),
            str(output_img_path),
        )
        Path(log_file_path).write_text("fake transform log\n")

    monkeypatch.setattr("brainles_preprocessing.transform.Transform.apply", _fake_apply)

    with pytest.raises(RuntimeError, match="shape"):
        resample_mask_to_source(
            mask=np.zeros((5, 5, 5), dtype=np.uint8),
            atlas_affine=np.eye(4),
            transformations_dir=transformations_dir,
            target_role="t1ce",
            target_native_path=target_native_path,
            out_dir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    ("bad_value", "expected_match"),
    [
        (1.5, "round cleanly"),  # non-integer: genericLabel is supposed to never blend
        (7.0, "outside"),  # integer, but not one of {0, 1, 2, 3}
    ],
)
def test_resample_mask_to_source_raises_on_invalid_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
    expected_match: str,
) -> None:
    """Values that don't round cleanly, or that round to an unknown class, both raise."""
    pytest.importorskip("brainles_preprocessing")
    from neurovision.data.clinical_resample import resample_mask_to_source

    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1ce").mkdir(parents=True)

    target_shape = (6, 6, 6)
    target_native_path = _write_nifti(
        tmp_path / "native.nii.gz", np.zeros(target_shape, dtype=np.float32)
    )

    def _fake_apply(
        self,
        target_modality_name,
        target_modality_img,
        moving_image,
        output_img_path,
        log_file_path,
        interpolator=None,
        inverse=False,
    ) -> None:
        data = np.zeros(target_shape, dtype=np.float32)
        data[1, 1, 1] = bad_value
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(output_img_path))
        Path(log_file_path).write_text("fake transform log\n")

    monkeypatch.setattr("brainles_preprocessing.transform.Transform.apply", _fake_apply)

    with pytest.raises(RuntimeError, match=expected_match):
        resample_mask_to_source(
            mask=np.zeros((5, 5, 5), dtype=np.uint8),
            atlas_affine=np.eye(4),
            transformations_dir=transformations_dir,
            target_role="t1ce",
            target_native_path=target_native_path,
            out_dir=tmp_path / "out",
        )


def test_resample_mask_to_source_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correctly-shaped, valid-values result is returned, and re-saved as clean uint8."""
    pytest.importorskip("brainles_preprocessing")
    from neurovision.data.clinical_resample import resample_mask_to_source

    transformations_dir = tmp_path / "transformations"
    (transformations_dir / "t1ce").mkdir(parents=True)

    target_shape = (6, 6, 6)
    target_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    target_native_path = _write_nifti(
        tmp_path / "native.nii.gz", np.zeros(target_shape, dtype=np.float32), target_affine
    )

    valid_data = np.zeros(target_shape, dtype=np.float32)
    valid_data[2:4, 2:4, 2:4] = 2.0  # a small block of class 2 (edema)

    def _fake_apply(
        self,
        target_modality_name,
        target_modality_img,
        moving_image,
        output_img_path,
        log_file_path,
        interpolator=None,
        inverse=False,
    ) -> None:
        nib.save(nib.Nifti1Image(valid_data.copy(), target_affine), str(output_img_path))
        Path(log_file_path).write_text("fake transform log\n")

    monkeypatch.setattr("brainles_preprocessing.transform.Transform.apply", _fake_apply)

    out_dir = tmp_path / "out"
    result_path = resample_mask_to_source(
        mask=np.zeros((5, 5, 5), dtype=np.uint8),  # atlas-space shape is irrelevant here: mocked
        atlas_affine=np.eye(4),
        transformations_dir=transformations_dir,
        target_role="t1ce",
        target_native_path=target_native_path,
        out_dir=out_dir,
    )

    assert result_path == out_dir / "t1ce_native_mask.nii.gz"
    # The atlas-space mask this function writes for Transform.apply to consume
    # is a real side effect, not mocked -- confirm it actually happened.
    assert (out_dir / "atlas_space_mask.nii.gz").is_file()

    result_img = nib.load(str(result_path))
    assert result_img.get_data_dtype() == np.uint8
    result_data = np.asarray(result_img.dataobj)
    assert result_data.shape == target_shape
    assert set(np.unique(result_data).tolist()) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(result_data, valid_data.astype(np.uint8))


# ---------------------------------------------------------------------------
# Part 2 -- ONE real, slow, ANTs-backed round-trip test.
# ---------------------------------------------------------------------------


def _affine(spacing: tuple[float, float, float], origin: np.ndarray) -> np.ndarray:
    """Builds a diagonal NIfTI affine: no rotation of its own, real spacing/origin."""
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = spacing
    aff[:3, 3] = origin
    return aff


def _centered_origin(
    shape: tuple[int, int, int], spacing: tuple[float, float, float]
) -> np.ndarray:
    """Origin that puts the volume's physical center at (0, 0, 0)."""
    return -(np.array(shape, dtype=np.float64) * np.array(spacing, dtype=np.float64)) / 2.0


def _physical_coords(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    """Returns `(D, H, W, 3)`: the physical (x, y, z) mm coordinate of every voxel."""
    ii, jj, kk = np.indices(shape, dtype=np.float64)
    ones = np.ones_like(ii)
    homogeneous = np.stack([ii, jj, kk, ones], axis=-1)
    physical = homogeneous @ affine.T
    return physical[..., :3]


# Six Gaussian "hotspots" (some positive, some negative, varied width) at fixed
# physical (x, y, z) mm locations. This is deliberately NOT one smooth blob or a
# flat gradient: an earlier version of this test used one broad, smooth field and
# real ANTs Rigid registration converged to a wrong, spurious alignment (measured
# post-registration correlation -0.23, i.e. worse than no registration at all) --
# there was not enough distinct local structure for the similarity metric to find
# the true optimum. This sharper, multi-blob texture measured a post-registration
# correlation of 1.000 against the (independently constructed) atlas image in
# manual testing, so it is what both the native and atlas images below are built
# from.
_ANATOMY_BLOBS: tuple[tuple[float, float, float, float, float], ...] = (
    (-14.0, 8.0, -6.0, 80.0, 3.0),
    (10.0, -10.0, 5.0, -60.0, 2.5),
    (2.0, 14.0, -10.0, 70.0, 3.5),
    (-6.0, -12.0, 8.0, -50.0, 2.0),
    (12.0, 6.0, 10.0, 90.0, 2.8),
    (-16.0, -4.0, -12.0, 55.0, 2.2),
)


def _synthetic_anatomy(xyz: np.ndarray) -> np.ndarray:
    """Evaluates the fixed hotspot field at physical coordinates `xyz` (`(..., 3)`)."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    field = np.zeros_like(x)
    for cx, cy, cz, amplitude, sigma in _ANATOMY_BLOBS:
        squared_distance = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        field = field + amplitude * np.exp(-squared_distance / (2 * sigma**2))
    return field.astype(np.float32)


def _sphere_mask(xyz: np.ndarray, center: tuple[float, float, float], radius: float) -> np.ndarray:
    """Boolean sphere of physical `radius` mm centered at `center` (mm), in `xyz`'s grid."""
    squared_distance = sum((xyz[..., i] - c) ** 2 for i, c in enumerate(center))
    return squared_distance <= radius**2


def test_resample_mask_to_source_real_round_trip_native_atlas_native(tmp_path: Path) -> None:
    """Real ANTs round trip: native -> atlas -> back to native, checked with Dice.

    This is the test that actually proves the geometry hazard `clinical_resample.py`
    exists to close. Part 1's mocked tests only prove this function's own
    validation branches; they would all still pass if the real inverse-warp call
    silently did the wrong thing. Nothing here is mocked: a real `ANTsRegistrator`
    fits a real Rigid transform between two small synthetic volumes, and the real
    `Transform.apply` (both directions) does the actual resampling.

    Uses `AtlasCentricPreprocessor` directly rather than
    `clinical_preprocess.run_plan`: `run_plan` always resolves `atlas_image_path`
    to a real `brainles_preprocessing.constants.Atlas` member, which triggers a
    real SRI24 download over the network on first use. Calling
    `AtlasCentricPreprocessor` directly lets a plain synthetic `Path` stand in for
    the atlas image instead, while still exercising the exact transformation-
    saving code path `clinical_preprocess.run_plan` relies on (`.run(
    save_dir_transformations=...)`, one subdirectory per modality role -- see
    that module's own docstring).

    After the correct round trip is checked, this test also runs a NEGATIVE
    CONTROL: the single most plausible regression this function exists to
    prevent -- forgetting to invert, i.e. calling `Transform.apply` a second
    time with `inverse=False` instead of `inverse=True`. This reuses the same
    fitted `transformations_dir` (the expensive real-ANTs-registration call
    above runs exactly once) and the same atlas-space mask and native ground
    truth already in scope; only the cheap resample step itself is repeated,
    with the wrong direction. Without this, nothing in the test suite would
    fail if `resample_mask_to_source`'s `inverse=True` call were ever flipped
    -- the Dice threshold below would be an unenforced claim in a comment.
    """
    pytest.importorskip("brainles_preprocessing")
    # antspyx is the PyPI package name (see requirements-clinical.txt); the
    # importable module is "ants" (confirmed against the installed
    # .venv-clinical/lib/python3.11/site-packages/ants/ -- "import antspyx" is a
    # ModuleNotFoundError there).
    pytest.importorskip("ants")

    from brainles_preprocessing.modality import CenterModality
    from brainles_preprocessing.normalization import PercentileNormalizer
    from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
    from brainles_preprocessing.registration import ANTsRegistrator
    from brainles_preprocessing.transform import Transform

    from neurovision.data.clinical_resample import resample_mask_to_source

    start = time.perf_counter()

    # --- Native-space volume: a real, non-trivial (anisotropic, non-identity)
    # affine -- an identity affine would not exercise any real resampling math.
    native_shape = (56, 56, 56)
    native_spacing = (1.2, 1.3, 1.1)
    native_affine = _affine(native_spacing, _centered_origin(native_shape, native_spacing))
    native_phys = _physical_coords(native_shape, native_affine)
    native_image = _synthetic_anatomy(native_phys)

    native_path = tmp_path / "native_t1ce.nii.gz"
    nib.save(nib.Nifti1Image(native_image, native_affine), str(native_path))

    # The KNOWN ground-truth lesion, in NATIVE space, kept entirely separate from
    # the intensity image ANTs registers on -- class value 2 ("edema"), a valid
    # class per neurovision.reporting.dicom_seg.SEGMENT_DEFINITIONS.
    lesion_center_mm = (3.0, -2.0, 4.0)
    lesion_radius_mm = 6.0
    native_lesion = (_sphere_mask(native_phys, lesion_center_mm, lesion_radius_mm) * 2).astype(
        np.uint8
    )
    native_lesion_path = tmp_path / "native_lesion.nii.gz"
    nib.save(nib.Nifti1Image(native_lesion, native_affine), str(native_lesion_path))

    # --- A synthetic "atlas" grid, related to native space by a KNOWN, non-trivial
    # rigid pose (12 degree rotation about z, plus a translation) that real ANTs
    # Rigid registration below must recover on its own -- nothing here tells it
    # what the pose is.
    atlas_shape = (56, 56, 56)
    atlas_spacing = (1.0, 1.0, 1.0)
    atlas_affine = _affine(atlas_spacing, _centered_origin(atlas_shape, atlas_spacing))
    atlas_phys = _physical_coords(atlas_shape, atlas_affine)

    theta = np.deg2rad(12.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([4.0, -3.0, 2.0])
    # Where each atlas voxel's physical location falls IN THE NATIVE IMAGE'S OWN
    # FRAME, under the hidden ground-truth pose -- evaluating the same anatomy
    # field there is what makes the two images genuinely correspond, rather than
    # being unrelated synthetic noise.
    native_frame_phys = atlas_phys @ rotation.T + translation
    atlas_image = _synthetic_anatomy(native_frame_phys)

    atlas_path = tmp_path / "synthetic_atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas_image, atlas_affine), str(atlas_path))

    # --- Real registration: fit native -> atlas with ANTs (Rigid, the same
    # default AtlasCentricPreprocessor.run_atlas_registration uses for the center
    # modality), saving the transform the same way clinical_preprocess.run_plan
    # does. No moving modalities, no brain extraction, no N4, no defacing -- none
    # of that is what this test exists to check, and skipping them avoids a real
    # HD-BET weight download.
    out_dir = tmp_path / "preprocess_out"
    center_modality = CenterModality(
        modality_name="t1ce",
        input_path=native_path,
        normalizer=PercentileNormalizer(),
        normalized_skull_output_path=out_dir / "t1ce.nii.gz",
    )
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center_modality,
        moving_modalities=[],
        registrator=ANTsRegistrator(),
        brain_extractor=None,
        atlas_image_path=atlas_path,  # a plain Path -- never triggers fetch_atlases()
        n4_bias_corrector=None,
        temp_folder=tmp_path / "_brainles_tmp",
        use_gpu=False,
    )
    transformations_dir = tmp_path / "transformations"
    preprocessor.run(
        save_dir_transformations=transformations_dir,
        log_file=tmp_path / "preprocess.log",
    )
    assert (transformations_dir / "t1ce").is_dir()

    # --- Simulate "what a real predicted mask looks like": forward-warp the
    # KNOWN native-space lesion into atlas space using the transform just fitted
    # -- this is exactly the atlas-space mask this project's segmentation model
    # would hand to resample_mask_to_source in production.
    atlas_space_lesion_path = tmp_path / "atlas_space_lesion.nii.gz"
    Transform(transformations_dir).apply(
        target_modality_name="t1ce",
        target_modality_img=str(atlas_path),
        moving_image=str(native_lesion_path),
        output_img_path=str(atlas_space_lesion_path),
        log_file_path=str(tmp_path / "forward_warp.log"),
        interpolator="genericLabel",
        inverse=False,
    )
    atlas_space_img = nib.load(str(atlas_space_lesion_path))
    # Rounded before being handed to the function under test: this forward-warp
    # step (part of THIS TEST's setup, not of resample_mask_to_source) can leave
    # the same float64 NIfTI round-trip noise resample_mask_to_source's own
    # validation is designed to tolerate. A real caller's mask (straight off
    # classes_from_regions) is already clean integers.
    atlas_space_mask = np.rint(np.asarray(atlas_space_img.dataobj)).astype(np.uint8)

    # --- The function under test: warp the atlas-space mask back to native. ---
    result_path = resample_mask_to_source(
        mask=atlas_space_mask,
        atlas_affine=atlas_space_img.affine,
        transformations_dir=transformations_dir,
        target_role="t1ce",
        target_native_path=native_path,
        out_dir=tmp_path / "resampled",
    )
    round_tripped = np.asarray(nib.load(str(result_path)).dataobj)

    elapsed_s = time.perf_counter() - start

    dice = _dice(round_tripped == 2, native_lesion == 2)
    print(
        f"[test_resample_mask_to_source_real_round_trip_native_atlas_native] "
        f"Dice={dice:.4f}, wall clock={elapsed_s:.2f}s "
        f"(native {native_shape} @ {native_spacing} mm -> "
        f"atlas {atlas_shape} @ {atlas_spacing} mm, "
        f"true rigid offset {translation.tolist()} mm / {np.rad2deg(theta):.0f} deg)"
    )

    # Threshold justified by two actual measurements taken while writing this
    # test (three repeated runs each, same setup, `.venv-clinical`, this Mac):
    #   - This correct code path: Dice 0.9952-0.9962 every time (elapsed ~0.25s).
    #   - A deliberately broken variant (forgetting to invert -- calling
    #     Transform.apply a second time with inverse=False instead of True, the
    #     single-most-plausible bug this function exists to prevent): Dice
    #     0.0173-0.0193 every time, on the identical setup.
    # 0.9 sits far below the correct path's observed range and far above the
    # broken path's, so it catches a real directional/geometry bug (which shows
    # near-zero overlap) while leaving headroom for the small interpolation loss
    # any double-resample genuinely incurs, even with a perfect implementation.
    # The broken-variant number above is now also EXECUTED, not just claimed in
    # this comment -- see the negative control immediately below.
    assert dice > 0.9, (
        f"round-tripped lesion Dice {dice:.4f} is far below the ~0.995 this exact synthetic "
        "setup measures for a correct implementation, and close to the ~0.02 measured for a "
        "deliberately broken one (forgetting to invert) -- treat this as a real regression, "
        "not resampling noise."
    )

    # --- Negative control: prove the dice > 0.9 threshold above would actually catch
    # the single most plausible regression in this function -- forgetting to invert.
    # Reuses the transformations_dir already fitted above (no second real ANTs
    # registration) and the same atlas-space mask / native ground truth already in
    # scope; only the cheap resample step itself is repeated, this time deliberately
    # calling Transform.apply with inverse=False instead of inverse=True.
    broken_atlas_mask_path = tmp_path / "atlas_space_mask_for_negative_control.nii.gz"
    nib.save(nib.Nifti1Image(atlas_space_mask, atlas_space_img.affine), str(broken_atlas_mask_path))

    broken_native_path = tmp_path / "broken_native_mask.nii.gz"
    Transform(transformations_dir).apply(
        target_modality_name="t1ce",
        target_modality_img=str(native_path),
        moving_image=str(broken_atlas_mask_path),
        output_img_path=str(broken_native_path),
        log_file_path=str(tmp_path / "broken_resample.log"),
        interpolator="genericLabel",
        inverse=False,  # deliberately wrong direction -- the regression under test
    )
    broken_data = np.rint(np.asarray(nib.load(str(broken_native_path)).dataobj)).astype(np.uint8)
    dice_broken = _dice(broken_data == 2, native_lesion == 2)

    print(
        f"[test_resample_mask_to_source_real_round_trip_native_atlas_native] "
        f"negative control (inverse=False): Dice={dice_broken:.4f}"
    )

    # Threshold set well above the ~0.0173-0.0193 measured for this exact broken
    # variant (three repeated runs while writing this test, see the comment above)
    # and well below the correct path's own 0.9 threshold, so a regression that
    # accidentally narrows this gap (in either direction) is still caught.
    assert dice_broken < 0.3, (
        f"negative-control (inverse=False) Dice {dice_broken:.4f} is not clearly separated "
        "from the correct-direction result -- this defeats the point of the negative control, "
        "which is to prove the dice > 0.9 threshold above would actually catch this exact "
        "regression class (forgetting to invert) if it ever happened for real."
    )
