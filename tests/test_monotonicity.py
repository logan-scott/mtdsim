"""Statistical sanity: ASP is non-increasing in mutation frequency.

These are seeded but slow (they run many trials), so they carry the ``slow``
marker and can be skipped with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import pytest

from mtdsim.engine import run_config
from mtdsim.metrics import aggregate


@pytest.mark.slow
def test_asp_non_increasing_in_frequency(fast_config):
    cfg0 = fast_config.with_overrides(engine={"n_trials": 600})
    freqs = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    asps = []
    for f in freqs:
        cfg = cfg0.with_overrides(defender={"mutation_frequency": f})
        asps.append(aggregate(run_config(cfg))["asp"])

    # Static defense yields the highest ASP.
    assert asps[0] == max(asps)
    assert asps[0] > 0.9
    # Monotone non-increasing within Monte Carlo tolerance.
    tol = 0.04
    for lo, hi in zip(asps[1:], asps[:-1], strict=True):
        assert lo <= hi + tol, f"ASP rose with frequency beyond tolerance: {asps}"
    # And the effect is real: high frequency strongly suppresses success.
    assert asps[-1] < 0.2


@pytest.mark.slow
def test_overhead_increases_with_frequency(fast_config):
    cfg0 = fast_config.with_overrides(engine={"n_trials": 300})
    prev = -1.0
    for f in [0.0, 0.05, 0.15, 0.3]:
        cfg = cfg0.with_overrides(defender={"mutation_frequency": f})
        overhead = aggregate(run_config(cfg))["overhead_mean"]
        assert overhead >= prev, "overhead should be non-decreasing in frequency"
        prev = overhead
