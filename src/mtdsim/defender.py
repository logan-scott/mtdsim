"""Defender — Moving Target Defense techniques and overhead accounting.

Each enabled attribute-mutating technique fires **independently** with per-tick
probability ``mutation_frequency`` (``f``). When it fires it reconfigures its
attribute on a ``mutation_coverage`` fraction of assets, drawing fresh values
from the large attribute space. The ``deception`` technique, when enabled, fires
at the same rate and re-randomises every decoy's attributes.

Operational overhead accrues per firing as::

    overhead += per_mutation_cost * weight[technique] * (#attributes changed)

with ``#attributes changed`` equal to ``round(coverage * n_total)`` for an
attribute technique and ``n_decoys`` for deception. Heavier weights for
``shuffling`` (connection resets) and ``deception`` (decoy upkeep) encode that
those reconfigurations cost more than a port swap. See ``DECISIONS.md`` for the
rationale behind independent per-technique firing (it is what makes the
technique ablation produce a monotone ASP gradient).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ALL_TECHNIQUES, TECHNIQUE_ATTRIBUTE
from .config import DefenderConfig
from .system import System


@dataclass
class StepResult:
    """What happened on a single defender tick."""

    overhead: float
    attributes_changed: int
    fired: bool


class Defender:
    """Stateful MTD controller for one trial; accumulates operational overhead."""

    def __init__(self, cfg: DefenderConfig) -> None:
        self.cfg = cfg
        self.total_overhead: float = 0.0
        self.total_attributes_changed: int = 0
        self.mutation_events: int = 0  # ticks on which at least one technique fired
        # Per-technique cumulative attribute-cells changed. Because
        # total_overhead == per_mutation_cost * sum_t weight[t] * technique_changes[t],
        # these counts let overhead be recomputed under alternate cost weights
        # WITHOUT re-running the simulation (reviewer item Ma). Tracking them does
        # not consume RNG or alter overhead, so all existing results are unchanged.
        self.technique_changes: dict[str, int] = dict.fromkeys(ALL_TECHNIQUES, 0)

    def _coverage_indices(self, system: System, rng: np.random.Generator) -> np.ndarray:
        """Pick the asset indices reconfigured by a firing attribute technique."""
        if self.cfg.mutation_coverage >= 1.0:
            return np.arange(system.n_total)
        k = max(1, round(self.cfg.mutation_coverage * system.n_total))
        return rng.choice(system.n_total, size=k, replace=False)

    def step(self, system: System, rng: np.random.Generator) -> StepResult:
        """Advance the defender one tick, mutating attributes and charging cost."""
        f = self.cfg.mutation_frequency
        overhead = 0.0
        changed = 0
        fired = False
        if f <= 0.0:
            return StepResult(0.0, 0, False)

        for tech in self.cfg.enabled_attribute_techniques:
            if rng.random() < f:
                fired = True
                attr = TECHNIQUE_ATTRIBUTE[tech]
                idx = self._coverage_indices(system, rng)
                n = system.mutate_attribute(attr, idx, rng)
                changed += n
                self.technique_changes[tech] += n
                overhead += self.cfg.per_mutation_cost * self.cfg.weight(tech) * n

        if self.cfg.deception_enabled and system.n_decoys > 0 and rng.random() < f:
            fired = True
            n = system.rotate_decoys(rng)
            changed += n
            self.technique_changes["deception"] += n
            overhead += self.cfg.per_mutation_cost * self.cfg.weight("deception") * n

        self.total_overhead += overhead
        self.total_attributes_changed += changed
        if fired:
            self.mutation_events += 1
        return StepResult(overhead, changed, fired)
