"""Attacker kill-chain: knowledge invalidation, staleness, adaptivity."""

from __future__ import annotations

from mtdsim.attacker import Attacker, Stage
from mtdsim.config import AttackerConfig, SystemConfig
from mtdsim.rng import generator
from mtdsim.system import build_system


def _committed_attacker(adaptive=True, relied_idx=2):
    """An attacker placed into a committed (EXPLOIT_DEV) stage on a fresh system."""
    sys = build_system(SystemConfig(n_assets=6, decoy_ratio=0.0), False, generator(1))
    att = Attacker(AttackerConfig(adaptive=adaptive, replan_penalty=4), False)
    att.stage = Stage.EXPLOIT_DEV
    att.stage_remaining = 5
    att.relied_idx = relied_idx
    att.relied_recorded = sys.snapshot(relied_idx)
    att.relied_is_real = True
    return sys, att


def test_mutation_to_recorded_attribute_forces_recon():
    sys, att = _committed_attacker()
    # Mutate an attribute of the *relied* asset -> stale knowledge.
    sys.attributes["port"][att.relied_idx] += 1
    compromised = att.step(sys, generator(10))
    assert compromised is False
    assert att.forced_recons == 1
    assert att.stage == Stage.RECON
    assert att.relied_idx is None  # committed knowledge dropped


def test_unrelated_mutation_does_not_force_recon():
    sys, att = _committed_attacker(relied_idx=2)
    # Mutate a *different* asset (asset 4) the attacker is not relying on.
    sys.attributes["port"][4] += 1
    compromised = att.step(sys, generator(10))
    assert compromised is False
    assert att.forced_recons == 0
    assert att.stage == Stage.EXPLOIT_DEV  # still progressing
    assert att.stage_remaining == 4  # one tick consumed


def test_nonadaptive_incurs_replan_penalty_on_failure():
    sys, att = _committed_attacker(adaptive=False)
    sys.attributes["endpoint_path"][att.relied_idx] += 1
    att.step(sys, generator(10))
    assert att.forced_recons == 1
    assert att.penalty_remaining == att.cfg.replan_penalty
    # The next steps just burn the penalty without acquiring knowledge.
    before = att.penalty_remaining
    att.step(sys, generator(11))
    assert att.penalty_remaining == before - 1


def test_execute_completion_on_real_target_compromises():
    sys, att = _committed_attacker()
    att.stage = Stage.EXECUTE
    att.stage_remaining = 1
    att.relied_is_real = True
    compromised = att.step(sys, generator(3))
    assert compromised is True
    assert att.stage == Stage.COMPROMISED


def test_execute_completion_on_wrong_target_fails_to_recon():
    sys, att = _committed_attacker()
    att.stage = Stage.EXECUTE
    att.stage_remaining = 1
    att.relied_is_real = False
    compromised = att.step(sys, generator(3))
    assert compromised is False
    assert att.wrong_target_failures == 1
    assert att.stage == Stage.RECON


def test_identification_accuracy_drops_with_decoys():
    sys_dec = build_system(SystemConfig(n_assets=8, decoy_ratio=0.5), True, generator(1))
    sys_plain = build_system(SystemConfig(n_assets=8, decoy_ratio=0.0), False, generator(1))
    att = Attacker(AttackerConfig(adaptive=False), service_diversity_enabled=False)
    assert att._p_correct(sys_dec) < att._p_correct(sys_plain)


def test_learning_uses_decoy_encounters_not_rounds_by_default():
    """C2a: the default signal learns from decoy encounters, not raw failures."""
    sys = build_system(SystemConfig(n_assets=8, decoy_ratio=0.5), True, generator(1))
    att = Attacker(AttackerConfig(adaptive=True, learning_rate=0.05), False)
    p0 = att._p_correct(sys)
    # Pure failure/round count (e.g. from MTD-forced re-recons) must NOT help.
    att.rounds_completed = 5
    assert att._p_correct(sys) == p0
    # Only informative decoy encounters raise accuracy.
    att.decoys_encountered = 5
    assert att._p_correct(sys) > p0


def test_learning_invariant_no_decoys_no_diversity_is_flat():
    """C2a invariant: with no decoys and diversity off, mutation pressure cannot
    change identification accuracy under the honest learning signals."""
    sys = build_system(SystemConfig(n_assets=8, decoy_ratio=0.0), False, generator(1))
    for signal in ("decoy_encounters", "none"):
        att = Attacker(AttackerConfig(adaptive=True, learning_signal=signal), False)
        base = att._p_correct(sys)
        # Simulate heavy MTD pressure: many forced re-recons / rounds.
        att.rounds_completed = 50
        att.forced_recons = 50
        assert att._p_correct(sys) == base == att.cfg.identify_base_accuracy

    # The legacy "rounds" signal VIOLATES the invariant (this is exactly the C2a
    # bug): raw MTD-forced rounds perversely raise accuracy.
    legacy = Attacker(AttackerConfig(adaptive=True, learning_signal="rounds"), False)
    base = legacy._p_correct(sys)
    legacy.rounds_completed = 50
    assert legacy._p_correct(sys) > base


def test_legacy_rounds_signal_reproduces_old_behavior():
    sys = build_system(SystemConfig(n_assets=8, decoy_ratio=0.5), True, generator(1))
    att = Attacker(
        AttackerConfig(adaptive=True, learning_rate=0.05, learning_signal="rounds"), False
    )
    p0 = att._p_correct(sys)
    att.rounds_completed = 5
    assert att._p_correct(sys) > p0


def test_committing_to_decoy_increments_encounter_counter():
    """A forced re-recon while committed to a decoy counts as an informative encounter."""
    sys = build_system(SystemConfig(n_assets=4, decoy_ratio=1.0), True, generator(1))
    decoy_idx = sys.n_real  # first decoy
    assert sys.is_decoy(decoy_idx)
    att = Attacker(AttackerConfig(adaptive=True), False)
    att.stage = Stage.EXECUTE
    att.stage_remaining = 1
    att.relied_idx = decoy_idx
    att.relied_recorded = sys.snapshot(decoy_idx)
    att.relied_is_real = False  # decoy is never the real target
    att.step(sys, generator(3))
    assert att.decoys_encountered == 1
    # Discarding a non-decoy must NOT increment the counter.
    att2 = Attacker(AttackerConfig(adaptive=True), False)
    att2.stage = Stage.EXECUTE
    att2.stage_remaining = 1
    att2.relied_idx = 0  # a real (non-target) asset
    att2.relied_recorded = sys.snapshot(0)
    att2.relied_is_real = False
    att2.step(sys, generator(3))
    assert att2.decoys_encountered == 0
