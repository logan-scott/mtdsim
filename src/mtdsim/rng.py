"""Centralized, reproducible randomness.

All stochasticity in the simulation flows through ``numpy.random.Generator``
instances created here. The contract: a given ``(master_seed, trial_index)``
pair always yields the same ``Generator`` state, so a full run is bit-for-bit
reproducible and **independent of the number of parallel workers** — the trial
seeds are materialised in the parent process and handed to workers.
"""

from __future__ import annotations

import numpy as np


def trial_seeds(master_seed: int, n_trials: int) -> list[int]:
    """Deterministically derive ``n_trials`` independent integer seeds.

    Uses :class:`numpy.random.SeedSequence` spawning, which is designed to
    produce statistically independent, high-quality child streams. Returning
    plain ``int`` seeds keeps them trivially picklable for multiprocessing.
    """
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_trials)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def generator(seed: int) -> np.random.Generator:
    """Create a fresh PCG64 generator for a single trial seed."""
    return np.random.default_rng(seed)
