"""System construction and attribute mutation."""

from __future__ import annotations

import numpy as np

from mtdsim import ATTRIBUTES
from mtdsim.config import SystemConfig
from mtdsim.rng import generator
from mtdsim.system import build_system


def test_decoys_present_only_with_deception():
    cfg = SystemConfig(n_assets=8, decoy_ratio=0.5)
    rng = generator(1)
    with_dec = build_system(cfg, deception_enabled=True, rng=rng)
    without = build_system(cfg, deception_enabled=False, rng=generator(1))
    assert with_dec.n_decoys == 4
    assert with_dec.n_total == 12
    assert without.n_decoys == 0
    assert without.n_total == 8


def test_real_target_is_a_real_asset():
    cfg = SystemConfig(n_assets=5, decoy_ratio=1.0)
    sys = build_system(cfg, deception_enabled=True, rng=generator(7))
    assert 0 <= sys.real_target_idx < sys.n_real
    assert not sys.is_decoy(sys.real_target_idx)
    assert sys.is_real_target(sys.real_target_idx)


def test_mutate_attribute_changes_only_targeted_assets():
    cfg = SystemConfig(n_assets=6, decoy_ratio=0.0)
    rng = generator(3)
    sys = build_system(cfg, deception_enabled=False, rng=rng)
    before = sys.attributes["port"].copy()
    changed = sys.mutate_attribute("port", np.array([1, 3]), rng)
    assert changed == 2
    after = sys.attributes["port"]
    # Targeted indices (almost surely) changed; others identical.
    assert after[1] != before[1]
    assert after[3] != before[3]
    untouched = [0, 2, 4, 5]
    assert np.array_equal(after[untouched], before[untouched])


def test_snapshot_returns_all_attributes():
    cfg = SystemConfig(n_assets=4, decoy_ratio=0.0)
    sys = build_system(cfg, deception_enabled=False, rng=generator(2))
    snap = sys.snapshot(0)
    assert set(snap) == set(ATTRIBUTES)
