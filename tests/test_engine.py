"""Engine: termination semantics, determinism, parallel-equivalence."""

from __future__ import annotations

import pytest

from mtdsim.engine import run_config, run_trial
from mtdsim.metrics import aggregate


def test_static_defense_compromises_and_records_ttc(fast_config):
    cfg = fast_config.with_overrides(defender={"mutation_frequency": 0.0})
    res = run_trial(cfg, seed=1)
    assert res.compromised is True
    assert res.ttc is not None and res.ttc > 0
    assert res.overhead == 0.0  # f=0 => no reconfiguration cost


def test_horizon_timeout_yields_no_compromise(fast_config):
    # Tiny horizon: the attacker cannot finish a single kill chain.
    cfg = fast_config.with_overrides(engine={"horizon": 2})
    res = run_trial(cfg, seed=1)
    assert res.compromised is False
    assert res.ttc is None
    assert res.trial_ticks == 2


def test_asp_in_unit_interval(fast_config):
    cfg = fast_config.with_overrides(
        defender={"mutation_frequency": 0.15}, engine={"n_trials": 100}
    )
    row = aggregate(run_config(cfg))
    assert 0.0 <= row["asp"] <= 1.0
    # TTC stats are defined only over successful trials.
    assert row["compromises"] <= row["n_trials"]


def test_timeout_trials_excluded_from_ttc(fast_config):
    cfg = fast_config.with_overrides(defender={"mutation_frequency": 0.5}, engine={"n_trials": 60})
    results = run_config(cfg)
    for r in results:
        if not r.compromised:
            assert r.ttc is None
        else:
            assert r.ttc is not None


def test_determinism_same_seed_same_aggregate(fast_config):
    cfg = fast_config.with_overrides(
        defender={"mutation_frequency": 0.08}, engine={"n_trials": 120}
    )
    a = aggregate(run_config(cfg), bootstrap_seed=1)
    b = aggregate(run_config(cfg), bootstrap_seed=1)
    for key in ("asp", "ttc_median", "ttc_mean", "overhead_mean", "forced_recons_mean"):
        assert a[key] == b[key], f"{key} differs across identical runs"


@pytest.mark.slow
def test_parallel_matches_serial(fast_config):
    cfg = fast_config.with_overrides(defender={"mutation_frequency": 0.08}, engine={"n_trials": 96})
    serial = run_config(cfg, parallel=False)
    parallel = run_config(cfg, parallel=True, processes=2)
    assert [r.as_dict() for r in serial] == [r.as_dict() for r in parallel]
