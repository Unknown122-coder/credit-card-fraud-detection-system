"""Global random-seed pinning for reproducible experiments.

Sets seeds for Python's built-in ``random`` module, NumPy, and the
environment variables consumed by XGBoost and LightGBM to guarantee
deterministic behaviour across training runs.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Pin every source of randomness to *seed*.

    Covers:
    - ``random.seed``
    - ``numpy.random.seed`` + ``numpy.random.default_rng``
    - ``PYTHONHASHSEED`` environment variable
    - ``OMP_NUM_THREADS`` = 1 (for thread-determinism)
    - XGBoost / LightGBM deterministic environment variables

    Parameters
    ----------
    seed : int, optional
        The seed value to use everywhere. Defaults to ``42``.
    """
    # Python built-in random
    random.seed(seed)

    # NumPy legacy + new-style generator
    np.random.seed(seed)

    # Hash seed — must be set before any hashing occurs (best-effort here)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Thread determinism for OpenMP-backed libraries
    os.environ["OMP_NUM_THREADS"] = "1"

    # XGBoost deterministic settings
    os.environ["XGB_NTHREAD"] = "1"

    # LightGBM deterministic settings
    os.environ["OMP_NUM_THREADS"] = "1"

    logger.info("Global random seed set to %d", seed)
