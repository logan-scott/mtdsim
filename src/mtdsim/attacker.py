"""Attacker — staged AI-driven kill chain with knowledge, staleness, adaptivity.

The attacker advances through a four-stage kill chain, each stage taking a
stochastic number of ticks:

1. ``RECON``       — learn current observable attributes of probed assets.
2. ``IDENTIFY``    — decide which probed asset is the real vulnerable target.
3. ``EXPLOIT_DEV`` — develop/select an exploit for the believed target.
4. ``EXECUTE``     — run the exploit; succeeds iff the relied knowledge is still
                     valid *and* the believed target is the real target.

**Knowledge & staleness (the core dynamic).** RECON ends by snapshotting live
attribute values, so freshly acquired knowledge is always valid. On entering
IDENTIFY the attacker *commits* to one relied asset and its recorded attributes.
Thereafter, on every tick, if any relied attribute's current value differs from
the recorded value (because the defender mutated it), the in-progress action
fails and the attacker is forced back to RECON — a *forced re-recon*. Because
re-recon re-reads live state, any knowledge that survives to EXECUTE completion
is necessarily valid, which implicitly enforces "succeeds only if knowledge
valid at execution."

**Identification error.** Decoys and service diversity reduce the probability
``p_correct`` that the committed asset is the real target; a wrong commitment
leads to a wasted exploit cycle ending in failure.

**Adaptivity.** An *adaptive* attacker resumes from RECON on failure. Its
identification accuracy improves only from *informative decoy encounters* —
committing to and then discarding an actual decoy (``learning_signal =
decoy_encounters``, the default) — so mutation pressure alone never changes
accuracy. The ``rounds`` signal, which lets any forced re-recon raise accuracy,
is an alternative kept for comparison. A *non-adaptive* attacker restarts with
a fixed replanning penalty and
never learns; with no decoys to disambiguate, the adaptive advantage reduces to
avoiding that penalty.

These per-trial signals feed the metrics in :mod:`mtdsim.metrics`:
``forced_recons`` and the time-averaged ``stale_fraction`` (attacker
uncertainty), the compromise flag and tick (ASP / TTC).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from . import ATTRIBUTES
from .config import AttackerConfig
from .system import System


class Stage(StrEnum):
    """Kill-chain stages plus the terminal compromised state."""

    RECON = "recon"
    IDENTIFY = "identify"
    EXPLOIT_DEV = "exploit_dev"
    EXECUTE = "execute"
    COMPROMISED = "compromised"


# Stages during which the attacker relies on committed knowledge (staleness applies).
_COMMITTED_STAGES = (Stage.IDENTIFY, Stage.EXPLOIT_DEV, Stage.EXECUTE)


class Attacker:
    """Stateful attacker agent for one trial."""

    def __init__(self, cfg: AttackerConfig, service_diversity_enabled: bool) -> None:
        self.cfg = cfg
        self.service_diversity_enabled = service_diversity_enabled

        # Kill-chain state
        self.stage: Stage = Stage.RECON
        self.stage_remaining: int = 0
        self.penalty_remaining: int = 0  # non-adaptive replanning idle ticks

        # Committed knowledge (set at IDENTIFY entry)
        self.relied_idx: int | None = None
        self.relied_recorded: dict[str, int] | None = None
        self.relied_is_real: bool = False

        # Experience / uncertainty counters
        self.rounds_completed: int = 0  # failed cycles total ("rounds" learning signal)
        self.forced_recons: int = 0  # staleness-induced re-recons
        self.wrong_target_failures: int = 0  # identification-error failures
        self.decoys_encountered: int = 0  # decoys committed-to-and-discarded (learning signal)

        # Metric accumulators
        self.recon_ticks: int = 0
        self.post_recon_ticks: int = 0
        self._stale_fraction_accum: float = 0.0

    # -- duration sampling --------------------------------------------------
    def _sample_duration(self, mean: float, rng: np.random.Generator) -> int:
        """Sample a positive integer stage duration (mean-preserving, light-tailed)."""
        if self.cfg.stage_time_cv <= 0:
            return max(1, int(round(mean)))
        val = mean * (1.0 + self.cfg.stage_time_cv * rng.standard_normal())
        return max(1, int(round(val)))

    def _stage_mean(self, stage: Stage) -> float:
        return {
            Stage.RECON: self.cfg.recon_time,
            Stage.IDENTIFY: self.cfg.identify_time,
            Stage.EXPLOIT_DEV: self.cfg.exploit_dev_time,
            Stage.EXECUTE: self.cfg.exploit_exec_time,
        }[stage]

    def _enter(self, stage: Stage, rng: np.random.Generator) -> None:
        self.stage = stage
        self.stage_remaining = self._sample_duration(self._stage_mean(stage), rng)

    def start(self, rng: np.random.Generator) -> None:
        """Initialise the attacker into its first RECON stage."""
        self._enter(Stage.RECON, rng)

    # -- identification -----------------------------------------------------
    def _p_correct(self, system: System) -> float:
        """Probability the committed asset is the real target this round.

        Adaptive learning raises base accuracy only via the
        configured ``learning_signal``. With the default ``decoy_encounters``,
        accuracy is invariant to pure mutation pressure: with no decoys the
        signal stays 0, so ``f`` cannot change identification accuracy.
        """
        denom = 1.0 + self.cfg.decoy_confusion_weight * system.n_decoys
        if self.service_diversity_enabled:
            denom += self.cfg.diversity_confusion_weight
        acc = self.cfg.identify_base_accuracy
        if self.cfg.adaptive and self.cfg.learning_signal != "none":
            signal = (
                self.decoys_encountered
                if self.cfg.learning_signal == "decoy_encounters"
                else self.rounds_completed
            )
            acc = min(self.cfg.max_identify_accuracy, acc + self.cfg.learning_rate * signal)
        return acc / denom

    def _probe(self, system: System, rng: np.random.Generator) -> np.ndarray:
        """Return the asset indices probed during reconnaissance."""
        if self.cfg.parallelism is None or self.cfg.parallelism >= system.n_total:
            return np.arange(system.n_total)
        return rng.choice(system.n_total, size=self.cfg.parallelism, replace=False)

    def _recon_complete(self, system: System, rng: np.random.Generator) -> None:
        """Snapshot probed assets, commit to a believed target, enter IDENTIFY."""
        probed = self._probe(system, rng)
        p = self._p_correct(system)
        target_probed = bool(np.any(probed == system.real_target_idx))
        self.relied_is_real = target_probed and (rng.random() < p)
        if self.relied_is_real:
            self.relied_idx = system.real_target_idx
        else:
            others = probed[probed != system.real_target_idx]
            self.relied_idx = int(rng.choice(others)) if others.size > 0 else int(probed[0])
        self.relied_recorded = system.snapshot(self.relied_idx)
        self._enter(Stage.IDENTIFY, rng)

    # -- staleness ----------------------------------------------------------
    def _count_stale(self, system: System) -> int:
        """Number of committed attributes whose live value no longer matches."""
        assert self.relied_idx is not None and self.relied_recorded is not None
        return sum(
            1
            for attr in ATTRIBUTES
            if int(system.attributes[attr][self.relied_idx]) != self.relied_recorded[attr]
        )

    def _force_recon(self, system: System, rng: np.random.Generator, *, staleness: bool) -> None:
        """Abort the current cycle and return to reconnaissance."""
        if staleness:
            self.forced_recons += 1
        else:
            self.wrong_target_failures += 1
        self.rounds_completed += 1
        # Informative decoy encounter: the attacker committed to an asset that is
        # actually a decoy and is now discarding it. This is the only thing that
        # teaches an adaptive attacker to disambiguate (learning_signal default).
        if self.relied_idx is not None and system.is_decoy(self.relied_idx):
            self.decoys_encountered += 1
        self.relied_idx = None
        self.relied_recorded = None
        self.relied_is_real = False
        if not self.cfg.adaptive:
            self.penalty_remaining = self.cfg.replan_penalty
        self._enter(Stage.RECON, rng)

    # -- per-tick step ------------------------------------------------------
    def step(self, system: System, rng: np.random.Generator) -> bool:
        """Advance the attacker one tick. Returns True iff a compromise occurs."""
        if self.stage == Stage.COMPROMISED:
            return True

        # Non-adaptive replanning idle: burn penalty ticks before re-acquiring.
        if self.penalty_remaining > 0:
            self.penalty_remaining -= 1
            return False

        # Staleness check for committed stages (runs before any stage completion,
        # so an invalidation on the EXECUTE-completing tick still aborts).
        if self.stage in _COMMITTED_STAGES:
            stale = self._count_stale(system)
            self.post_recon_ticks += 1
            self._stale_fraction_accum += stale / len(ATTRIBUTES)
            if stale > 0:
                self._force_recon(system, rng, staleness=True)
                return False
        elif self.stage == Stage.RECON:
            self.recon_ticks += 1

        # Spend one tick in the current stage.
        self.stage_remaining -= 1
        if self.stage_remaining > 0:
            return False

        # Stage complete -> transition.
        if self.stage == Stage.RECON:
            self._recon_complete(system, rng)
            return False
        if self.stage == Stage.IDENTIFY:
            self._enter(Stage.EXPLOIT_DEV, rng)
            return False
        if self.stage == Stage.EXPLOIT_DEV:
            self._enter(Stage.EXECUTE, rng)
            return False
        # EXECUTE complete.
        if self.relied_is_real:
            self.stage = Stage.COMPROMISED
            return True
        self._force_recon(system, rng, staleness=False)
        return False

    @property
    def stale_fraction(self) -> float:
        """Time-averaged fraction of committed knowledge found stale (post-recon)."""
        if self.post_recon_ticks == 0:
            return 0.0
        return self._stale_fraction_accum / self.post_recon_ticks
