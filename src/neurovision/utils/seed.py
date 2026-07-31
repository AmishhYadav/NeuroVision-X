"""Central seeding utility.

Every source of randomness in NeuroVision-X — Python, NumPy, PyTorch, and MONAI's
own transform RNGs — is seeded from this one function, so that experiments start
from a known state and any explicit sampling (e.g. DataLoader workers) can take
the returned generator instead of touching global state.
"""

import logging
import random

import numpy as np
import torch
from monai.utils import set_determinism

logger = logging.getLogger(__name__)

# NumPy's legacy global RNG only accepts seeds in this range.
_MAX_NUMPY_SEED = 2**32


def set_seed(seed: int, cudnn_benchmark: bool = True) -> torch.Generator:
    """Seed every RNG used in this project and return a CPU generator.

    Seeds, in order: Python's `random`, NumPy's legacy global RNG,
    `torch.manual_seed` (CPU), `torch.cuda.manual_seed_all` (skipped if no
    CUDA device is present, so this is safe to call on a CPU-only machine),
    and MONAI's `set_determinism`, which controls the RNG state MONAI's
    random transforms carry internally — seeding torch alone does not
    reach those.

    Note on determinism: some 3D CUDA convolution and pooling backward
    kernels have no deterministic implementation at all, so two GPU runs
    with identical seeds can still differ slightly no matter what flags are
    set. The project therefore reports mean ± std across seeds rather than
    claiming bit-exact reproducibility from a single run. What seeding here
    *does* guarantee is reproducible data order, augmentation choices, and
    weight initialization.

    Given that, we do not chase full determinism. `set_determinism` sets
    `cudnn.deterministic = True` and `cudnn.benchmark = False`; we leave the
    first alone and restore the second (see `cudnn_benchmark`), because
    disabling autotuning costs throughput without delivering the bit-exact
    guarantee we never claimed. `torch.use_deterministic_algorithms(True)`
    is likewise not called — it would raise on the nondeterministic 3D
    kernels above rather than make them deterministic.

    Args:
        seed: Non-negative integer seed, must satisfy `0 <= seed < 2**32`
            (the range NumPy's legacy seeder accepts).
        cudnn_benchmark: Whether to leave cuDNN kernel autotuning on. MONAI's
            `set_determinism` turns it off; we turn it back on by default
            because training uses fixed-size patches, where autotuning pays
            for itself. Set False for variable-shape workloads (e.g. sliding
            -window inference with ragged edge patches), where re-tuning on
            every new shape costs more than it saves, or when debugging a
            suspected nondeterminism bug.

    Returns:
        A CPU `torch.Generator` seeded with `seed`, for callers (e.g.
        DataLoaders) that want an explicit generator instead of relying
        on global RNG state.

    Raises:
        TypeError: If `seed` is not an `int`.
        ValueError: If `seed` is negative or `>= 2**32`.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0 or seed >= _MAX_NUMPY_SEED:
        raise ValueError(f"seed must satisfy 0 <= seed < 2**32, got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeds MONAI's own random transforms; without this, transforms like
    # RandCropByPosNegLabeld keep their own unseeded RNG state.
    set_determinism(seed=seed)

    # set_determinism sets cudnn.benchmark = False. Restore it, because that
    # flag is not buying us what it looks like it buys: 3D conv backward
    # kernels are nondeterministic regardless, so we already report mean +/-
    # std across seeds instead of claiming bit-exact runs. Meanwhile
    # benchmark=False disables cuDNN kernel autotuning, which is worth real
    # throughput on the fixed 96^3 patches training uses -- and throughput is
    # scarce inside a rationed 12-hour Kaggle session.
    torch.backends.cudnn.benchmark = cudnn_benchmark

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    logger.info("Seed set to %d (cudnn.benchmark=%s)", seed, cudnn_benchmark)
    return generator
