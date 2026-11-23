"""Deterministic synthetic image corruptions for the severity ladder (W3.2).

Each corruption degrades an already-preprocessed BraTS image in a controlled,
reproducible way, so a model's accuracy and conformal risk can be measured as
a function of corruption severity. Labels are never touched: the truth does
not change when the picture gets worse.

Image layout: `(C, H, W, D)` with channels T1, T1CE, T2, FLAIR. Axis 3 of the
4-D array (axis 2 of one channel) is S, the axial through-plane axis. Every
channel is z-scored over its nonzero (brain) voxels; background is exactly 0.

Randomness comes only from an explicit integer seed (never global RNG state,
never Python's salted ``hash()``).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from numpy.polynomial import legendre

logger = logging.getLogger(__name__)

KINDS: tuple[str, ...] = ("thick_slice", "rician", "bias", "gamma", "motion", "none")


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Voxels that are nonzero in any channel of the ORIGINAL image.

    Args:
        image: Array `(C, H, W, D)`.

    Returns:
        Boolean array `(H, W, D)`.
    """
    return np.any(np.asarray(image) != 0, axis=0)


def case_seed(base_seed: int, case_id: str, kind: str, level: int) -> int:
    """Stable 32-bit seed for one (case, corruption, level) cell.

    Uses sha256 because Python's built-in ``hash()`` is salted per process and
    would give different noise on every run.

    Returns:
        Integer in `[0, 2**32)`.
    """
    text = f"{base_seed}|{case_id}|{kind}|{level}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------
# Individual corruptions. Each takes a non-negative "magnitude" channel `m`
# of shape `(H, W, D)` (air = 0, darkest brain voxel = 0) and returns an array
# of the same shape.
# --------------------------------------------------------------------------


def thick_slice(m: np.ndarray, slice_mm: int, axis: int = -1) -> np.ndarray:
    """Simulate thick-slice acquisition along the through-plane axis.

    Real thick-slice scans average the signal over the slab thickness. We
    block-average groups of `slice_mm` consecutive slices (the last group may
    be shorter), then linearly interpolate back to the original length with
    each block's mean placed at the block centre. Linear interpolation is done
    with explicit index/weight arrays in numpy: it is exact, deterministic and
    works on one axis without building a 3-D coordinate grid.

    Args:
        m: Magnitude channel `(H, W, D)`.
        slice_mm: Slab thickness in voxels (1 mm each). 1 returns `m` unchanged.
        axis: Through-plane axis.

    Returns:
        Float32 array, same shape as `m`.
    """
    if slice_mm < 1:
        raise ValueError(f"slice_mm must be >= 1, got {slice_mm}")
    m = np.asarray(m, dtype=np.float32)
    if slice_mm == 1:
        return m.copy()

    moved = np.moveaxis(m, axis, -1)
    n = moved.shape[-1]
    starts = np.arange(0, n, slice_mm)
    stops = np.minimum(starts + slice_mm, n)
    # Mean of each block (reduceat sums, divide by true block length).
    sums = np.add.reduceat(moved, starts, axis=-1)
    means = sums / (stops - starts).astype(np.float32)
    centres = (starts + stops - 1) / 2.0  # block centre in voxel coordinates

    if len(starts) == 1:
        out = np.broadcast_to(means, moved.shape).copy()
    else:
        j = np.arange(n, dtype=np.float64)
        lo = np.clip(np.searchsorted(centres, j, side="right") - 1, 0, len(starts) - 2)
        hi = lo + 1
        # Weight of the upper block; clipping holds the edge value constant
        # before the first / after the last block centre.
        w = np.clip((j - centres[lo]) / (centres[hi] - centres[lo]), 0.0, 1.0)
        w = w.astype(np.float32)
        out = means[..., lo] * (1.0 - w) + means[..., hi] * w
    return np.moveaxis(out.astype(np.float32), -1, axis)


def rician_noise(
    m: np.ndarray,
    sigma_rel: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Add Rician noise inside the brain mask.

    MRI scanners record complex signal (real + imaginary) with Gaussian noise
    on both parts; the image we see is the magnitude. So noise is
    ``sqrt((m + n1)^2 + n2^2)``, which is Rician, not Gaussian: it is
    non-negative and biases dark regions upward.

    Args:
        m: Magnitude channel `(H, W, D)`.
        sigma_rel: Noise sigma as a fraction of std of `m` inside the mask.
        rng: Seeded generator.
        mask: Boolean `(H, W, D)`; defaults to `m > 0`.

    Returns:
        Float32 array `(H, W, D)`; unchanged outside the mask.
    """
    m = np.asarray(m, dtype=np.float32)
    if mask is None:
        mask = m > 0
    sigma = float(sigma_rel) * float(m[mask].std()) if mask.any() else 0.0
    n1 = rng.normal(0.0, sigma, size=m.shape).astype(np.float32)
    n2 = rng.normal(0.0, sigma, size=m.shape).astype(np.float32)
    noisy = np.sqrt((m + n1) ** 2 + n2**2)
    return np.where(mask, noisy, m).astype(np.float32)


def bias_field(
    m: np.ndarray, coeff: float, rng: np.random.Generator, degree: int = 3
) -> np.ndarray:
    """Multiply by a smooth random bias field (coil sensitivity inhomogeneity).

    The field is ``exp(P(x, y, z))`` where P is a sum of products of Legendre
    polynomials of total degree <= `degree` on coordinates scaled to [-1, 1].
    Exponentiating keeps the field positive. The constant term is 0 (a global
    gain is removed by z-scoring anyway). `coeff=0` gives field == 1.

    Args:
        m: Magnitude channel `(H, W, D)`.
        coeff: Coefficients are drawn from U(-coeff, coeff).
        rng: Seeded generator.
        degree: Maximum total polynomial degree.

    Returns:
        Float32 array `(H, W, D)`.
    """
    m = np.asarray(m, dtype=np.float32)
    orders = [
        (i, j, k)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
        for k in range(degree + 1 - i - j)
        if (i, j, k) != (0, 0, 0)
    ]
    coefs = rng.uniform(-coeff, coeff, size=len(orders))

    def basis(n: int) -> np.ndarray:
        # Rows: Legendre polynomial of order 0..degree evaluated on [-1, 1].
        x = np.linspace(-1.0, 1.0, n)
        return np.stack([legendre.legval(x, np.eye(degree + 1)[o]) for o in range(degree + 1)])

    bx, by, bz = (basis(n) for n in m.shape)
    poly = np.zeros(m.shape, dtype=np.float64)
    for (i, j, k), c in zip(orders, coefs):
        poly += c * bx[i][:, None, None] * by[j][None, :, None] * bz[k][None, None, :]
    return (m * np.exp(poly)).astype(np.float32)


def gamma_contrast(
    m: np.ndarray,
    log_gamma: float,
    rng: np.random.Generator,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Non-linear contrast change ``u ** gamma`` with random direction.

    `m` is scaled to [0, 1] by its max inside the mask, raised to
    ``gamma = exp(+/- log_gamma)``, and scaled back. The sign is drawn from
    `rng`, so severity is ``|log gamma|`` and direction varies per case.
    The sign is always drawn, even at `log_gamma=0`, to keep RNG use fixed.

    Args:
        m: Magnitude channel `(H, W, D)`.
        log_gamma: Severity, >= 0. 0 leaves `m` unchanged.
        rng: Seeded generator.
        mask: Boolean `(H, W, D)`; defaults to `m > 0`.

    Returns:
        Float32 array `(H, W, D)`.
    """
    m = np.asarray(m, dtype=np.float32)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    if mask is None:
        mask = m > 0
    if not mask.any():
        return m.copy()
    peak = float(m[mask].max())
    if peak <= 0:
        return m.copy()
    gamma = float(np.exp(sign * log_gamma))
    u = np.clip(m / peak, 0.0, 1.0)
    return (u**gamma * peak).astype(np.float32)


def motion_line_block(n: int, fraction: float, rng: np.random.Generator) -> tuple[int, int]:
    """Choose the corrupted block of phase-encode lines, never touching DC.

    Indices are in centred (fftshifted) order, where the central (DC) line sits
    at index ``n // 2``. DC carries the overall brightness and coarse contrast;
    leaving it clean keeps the image recognisable and makes the artefact a
    ghosting effect rather than a total wipe-out.

    Args:
        n: Number of phase-encode lines.
        fraction: Fraction of lines to corrupt.
        rng: Seeded generator.

    Returns:
        `(start, stop)` half-open range in centred order, excluding `n // 2`.
    """
    length = max(1, int(round(fraction * n)))
    centre = n // 2
    # Valid starts: block ends before the centre, or begins after it.
    valid = list(range(0, centre - length + 1)) + list(range(centre + 1, n - length + 1))
    if not valid:
        raise ValueError(f"fraction={fraction} too large to avoid the DC line (n={n})")
    start = int(valid[int(rng.integers(len(valid)))])
    return start, start + length


def kspace_motion(
    m: np.ndarray, shift_mm: float, rng: np.random.Generator, fraction: float = 0.4
) -> np.ndarray:
    """Simulate in-plane rigid motion during acquisition (ghosting).

    MRI measures the Fourier transform ("k-space") of the image. By the shift
    theorem, translating an image by d multiplies its k-space by the phase
    ramp ``exp(-2 pi i k.d)``. If the patient moves part-way through the scan,
    only the k-space lines acquired after the move carry that ramp; the other
    lines are from the un-moved head. Inverse-transforming this mixture gives
    ghost copies of edges rather than a clean shifted image. We ramp a
    contiguous block of phase-encode lines (axis 1) and take the magnitude.

    Args:
        m: Magnitude channel `(H, W, D)`.
        shift_mm: Translation length in voxels (1 mm each); 0 is identity.
        rng: Seeded generator.
        fraction: Fraction of phase-encode lines that see the motion.

    Returns:
        Float32 array `(H, W, D)`.
    """
    m = np.asarray(m, dtype=np.float32)
    h, w, _ = m.shape
    start, stop = motion_line_block(w, fraction, rng)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    dx, dy = shift_mm * np.cos(angle), shift_mm * np.sin(angle)

    # Centred k-space along axis 1 so the block is contiguous in frequency.
    kspace = np.fft.fftshift(np.fft.fftn(m.astype(np.float64)), axes=1)
    kx = np.fft.fftfreq(h)[:, None, None]  # cycles per voxel, axis 0
    ky = np.fft.fftshift(np.fft.fftfreq(w))[None, start:stop, None]  # axis 1 block
    kspace[:, start:stop, :] *= np.exp(-2j * np.pi * (kx * dx + ky * dy))
    out = np.fft.ifftn(np.fft.ifftshift(kspace, axes=1))
    return np.abs(out).astype(np.float32)


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


def _zscore_in_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score over mask voxels; zero elsewhere (same semantics as normalize_nonzero)."""
    out = np.zeros(x.shape, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    std = vals.std()
    if std == 0:
        return out
    out[mask] = ((vals - vals.mean()) / std).astype(np.float32)
    return out


def corrupt_image(image: np.ndarray, kind: str, severity: float, seed: int) -> np.ndarray:
    """Apply one corruption to a preprocessed image.

    Severity meaning per kind: thick_slice = slab thickness in voxels,
    rician = sigma_rel, bias = coeff, gamma = log_gamma, motion = shift in
    voxels. "none" returns the image as float32 unchanged.

    Per channel: cast to float32, shift to a non-negative "magnitude" (darkest
    brain voxel = 0, air = 0), corrupt, zero outside the ORIGINAL mask, then
    re-z-score over that mask. The deployed pipeline z-scores each scan, so a
    corrupted scan must go through the same normalisation the model would see.
    The mask stays fixed to the original so skull-stripping is not what is
    being tested. Each channel gets an independent child generator (each MRI
    sequence is a separate acquisition).

    Args:
        image: Array `(C, H, W, D)`.
        kind: One of thick_slice, rician, bias, gamma, motion, none.
        severity: Kind-specific severity (see above).
        seed: Integer seed (see `case_seed`).

    Returns:
        Float32 array `(C, H, W, D)`.

    Raises:
        ValueError: If `kind` is not a valid kind.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown corruption kind {kind!r}; valid kinds: {', '.join(KINDS)}")
    image = np.asarray(image)
    if kind == "none":
        return image.astype(np.float32)

    mask = brain_mask(image)
    children = np.random.SeedSequence(seed).spawn(image.shape[0])
    out = np.zeros(image.shape, dtype=np.float32)
    for c, child in enumerate(children):
        rng = np.random.default_rng(child)
        x = image[c].astype(np.float32)
        m = np.zeros_like(x)
        if mask.any():
            m[mask] = x[mask] - x[mask].min()

        if kind == "thick_slice":
            corrupted = thick_slice(m, int(round(severity)))
        elif kind == "rician":
            corrupted = rician_noise(m, severity, rng, mask)
        elif kind == "bias":
            corrupted = bias_field(m, severity, rng)
        elif kind == "gamma":
            corrupted = gamma_contrast(m, severity, rng, mask)
        else:  # motion
            corrupted = kspace_motion(m, severity, rng)

        out[c] = _zscore_in_mask(np.where(mask, corrupted, 0.0).astype(np.float32), mask)
    return out
