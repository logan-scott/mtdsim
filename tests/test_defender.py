"""Defender mutation behavior and overhead accounting."""

from __future__ import annotations

from mtdsim.config import DefenderConfig, SystemConfig
from mtdsim.defender import Defender
from mtdsim.rng import generator
from mtdsim.system import build_system


def test_static_defender_never_mutates_or_charges():
    cfg = DefenderConfig(mutation_frequency=0.0)
    sys = build_system(SystemConfig(n_assets=5, decoy_ratio=0.0), False, generator(1))
    d = Defender(cfg)
    for _ in range(50):
        res = d.step(sys, generator(99))
        assert res.fired is False
        assert res.overhead == 0.0
    assert d.total_overhead == 0.0
    assert d.mutation_events == 0


def test_overhead_equals_cost_times_attributes_changed_multitechnique():
    # f=1.0 => every enabled attribute technique fires this tick.
    cfg = DefenderConfig(
        mutation_frequency=1.0,
        mutation_coverage=1.0,
        enabled_techniques=("port_rotation", "shuffling"),
        per_mutation_cost=2.0,
        technique_cost_weights={"port_rotation": 1.0, "shuffling": 3.0},
    )
    sys = build_system(SystemConfig(n_assets=4, decoy_ratio=0.0), False, generator(1))
    d = Defender(cfg)
    res = d.step(sys, generator(5))
    # 2 techniques x 4 assets = 8 attribute-cells changed.
    assert res.attributes_changed == 8
    # overhead = per_mutation_cost * (w_port*4 + w_shuffle*4) = 2 * (1*4 + 3*4) = 32.
    assert res.overhead == 2.0 * (1.0 * 4 + 3.0 * 4)
    assert d.total_overhead == res.overhead


def test_deception_overhead_counts_decoys():
    cfg = DefenderConfig(
        mutation_frequency=1.0,
        enabled_techniques=("deception",),
        per_mutation_cost=1.0,
        technique_cost_weights={"deception": 2.5},
    )
    sys = build_system(SystemConfig(n_assets=4, decoy_ratio=0.5), True, generator(1))
    assert sys.n_decoys == 2
    d = Defender(cfg)
    res = d.step(sys, generator(5))
    assert res.attributes_changed == 2  # #decoys
    assert res.overhead == 1.0 * 2.5 * 2


def test_coverage_limits_assets_changed():
    cfg = DefenderConfig(
        mutation_frequency=1.0,
        mutation_coverage=0.5,
        enabled_techniques=("port_rotation",),
    )
    sys = build_system(SystemConfig(n_assets=8, decoy_ratio=0.0), False, generator(1))
    d = Defender(cfg)
    res = d.step(sys, generator(2))
    assert res.attributes_changed == 4  # round(0.5 * 8)
