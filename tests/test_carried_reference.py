"""Tests for neurovision.analysis.carried_reference (synthetic data only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neurovision.analysis import carried_reference as cr

N = 48


def _brain(shift=(0, 0, 0)):
    """Smooth ellipsoid 'brain' with internal texture and a labelled blob, then shifted."""
    z, y, x = np.mgrid[:N, :N, :N].astype(np.float32)
    c = N / 2
    ell = ((z - c) / 16) ** 2 + ((y - c) / 14) ** 2 + ((x - c) / 12) ** 2 < 1
    texture = 1.0 + 0.5 * np.sin(x / 3.0) * np.cos(y / 4.0) + 0.3 * np.sin(z / 5.0 + x / 7.0)
    img = np.where(ell, texture + 2.0, 0.0).astype(np.float32)
    blob = (((z - c) / 5) ** 2 + ((y - c - 3) / 5) ** 2 + ((x - c + 2) / 4) ** 2 < 1) & ell
    label = np.where(blob, 3, 0).astype(np.int16)
    label[blob & (z < c)] = 1
    img = np.roll(img, shift, axis=(0, 1, 2))
    label = np.roll(label, shift, axis=(0, 1, 2))
    return img, label


def _dice(a, b):
    return cr.brain_overlap_dice(a > 0, b > 0)


def test_translation_recovered():
    pytest.importorskip("ants")
    moving, label = _brain()
    fixed, true_label = _brain(shift=(3, -2, 1))
    res = cr.carry_label(moving, label, fixed)
    assert _dice(res.carried_label, true_label) >= 0.95
    # fixed point p maps to moving point p - t, so the displacement is -(3, -2, 1)
    assert np.allclose(res.translation_mm, (-3, 2, -1), atol=1.0)


def test_identity():
    pytest.importorskip("ants")
    img, label = _brain()
    res = cr.carry_label(img, label, img)
    assert _dice(res.carried_label, label) >= 0.99
    assert np.allclose(res.translation_mm, 0, atol=0.5)
    assert res.rotation_deg < 1.0


def test_deterministic_and_labels_preserved():
    pytest.importorskip("ants")
    moving, label = _brain()
    fixed, _ = _brain(shift=(2, 1, -1))
    a = cr.carry_label(moving, label, fixed, seed=7)
    b = cr.carry_label(moving, label, fixed, seed=7)
    assert np.array_equal(a.carried_label, b.carried_label)
    assert set(np.unique(a.carried_label)) <= set(np.unique(label))
    assert a.carried_brain_mask.dtype == bool


def test_brain_overlap_dice_hand_values():
    a = np.zeros((4, 4, 4), bool)
    a[:2] = True  # 32 voxels
    b = np.zeros_like(a)
    b[1:3] = True  # 32 voxels, 16 shared
    assert cr.brain_overlap_dice(a, a) == 1.0
    assert cr.brain_overlap_dice(a, ~a) == 0.0
    assert cr.brain_overlap_dice(a, b) == pytest.approx(0.5)


def test_carry_qc_boundary():
    assert cr.carry_qc_passes(0.90, 0.90)
    assert not cr.carry_qc_passes(0.8999, 0.90)


def test_score_against_carried():
    lab = np.zeros((6, 6, 6), np.int16)
    lab[1:4, 1:4, 1:4] = 2
    lab[2, 2, 2] = 3
    scores = cr.score_against_carried(lab, lab)
    assert scores == {"dice_ET": 1.0, "dice_TC": 1.0, "dice_WT": 1.0}
    assert cr.score_against_carried(np.zeros_like(lab), lab)["dice_WT"] == 0.0


def _write_case(root: Path, channel_value=(0.0, 5.0, 0.0, 0.0)) -> Path:
    case = root / "BraTS_fake"
    case.mkdir()
    cropped = (6, 7, 5)
    bbox = [[2, 8], [3, 10], [0, 5]]
    image = np.zeros((4,) + cropped, np.float32)
    for c, v in enumerate(channel_value):
        image[c] = v
    label = np.zeros(cropped, np.int16)
    label[2:4, 2:5, 1:3] = 3
    np.save(case / "image.npy", image)
    np.save(case / "label.npy", label)
    meta = {
        "case_id": "fake",
        "original_shape": [12, 12, 6],
        "cropped_shape": list(cropped),
        "bbox": bbox,
        "affine": np.diag([-1.0, -1.0, 1.0, 1.0]).tolist(),
        "spacing": [1.0, 1.0, 1.0],
        "has_label": True,
    }
    (case / "meta.json").write_text(json.dumps(meta))
    return case


def test_brats_reference_on_grid(tmp_path):
    case = _write_case(tmp_path)
    img, lab = cr.brats_reference_on_grid(case)
    assert img.shape == lab.shape == (12, 12, 6)
    assert img.dtype == np.float32
    assert lab.max() == 3
    assert (img[lab > 0] != 0).all()  # label sits inside the image's nonzero region
    assert img.max() == 5.0  # channel 1 was picked
    img0, _ = cr.brats_reference_on_grid(case, modality_index=0)
    assert img0.max() == 0.0


def test_brats_reference_errors(tmp_path):
    case = _write_case(tmp_path)
    np.save(case / "label.npy", np.zeros((2, 2, 2), np.int16))
    with pytest.raises(ValueError):
        cr.brats_reference_on_grid(case)
    (case / "meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        cr.brats_reference_on_grid(case)


def test_import_has_no_toplevel_ants():
    code = (
        "import sys; sys.modules['ants'] = None; "
        "import neurovision.analysis.carried_reference as m; "
        "import numpy as np\n"
        "try:\n"
        "    m.carry_label(np.zeros((2,2,2)), np.zeros((2,2,2)), np.zeros((2,2,2)))\n"
        "except ImportError as e:\n"
        "    assert '.venv-clinical' in str(e)\n"
        "else:\n"
        "    raise SystemExit('no ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
