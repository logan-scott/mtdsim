"""Engine — the per-trial discrete-time loop, termination, and parallel trials.

Each tick the defender acts first (possibly mutating attributes, which may
invalidate the attacker's committed knowledge), then the attacker acts. A trial
terminates the moment the attacker compromises the real target (record TTC) or
when the horizon ``T`` is reached (no compromise). ASP is computed downstream as
compromises / trials.

Reproducibility: per-trial seeds come from :func:`mtdsim.rng.trial_seeds`, are
materialised in the parent process, and are mapped in order, so the aggregate
results are identical whether trials run serially or across worker processes.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from .attacker import Attacker
from .config import Config
from .defender import Defender
from .rng import generator, trial_seeds
from .system import build_system


@dataclass(frozen=True)
class TrialResult:
    """Per-trial outcome and instrumentation."""

    seed: int
    compromised: bool
    ttc: int | None  # ticks to compromise; None on timeout
    trial_ticks: int  # ticks the trial actually ran
    overhead: float  # cumulative operational overhead
    forced_recons: int  # staleness-induced re-recons (attacker uncertainty)
    wrong_target_failures: int  # identification-error failures
    stale_fraction: float  # time-averaged stale-knowledge fraction
    rounds_completed: int  # failed cycles total
    decoys_encountered: int  # decoys committed-to-and-discarded (adaptive learning signal)
    recon_ticks: int  # ticks spent (re-)acquiring knowledge
    mutation_events: int  # ticks on which the defender fired >=1 technique
    total_assets: int  # n_real + n_decoys (for per-asset overhead normalization)
    # Per-technique cumulative attribute-cells changed (reviewer item Ma). Lets
    # overhead be recomputed under alternate cost weights without re-running.
    chg_port_rotation: int
    chg_endpoint_mutation: int
    chg_shuffling: int
    chg_service_diversity: int
    chg_deception: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_trial(cfg: Config, seed: int) -> TrialResult:
    """Run a single seeded trial to compromise or horizon."""
    rng = generator(seed)
    system = build_system(cfg.system, cfg.defender.deception_enabled, rng)
    defender = Defender(cfg.defender)
    # Whether the attacker experiences service-diversity confusion is normally
    # derived from the mutating set, but can be overridden to decouple diversity's
    # two channels (reviewer item M6).
    sdiv = "service_diversity" in cfg.defender.enabled_techniques
    if cfg.attacker.diversity_confusion_override is not None:
        sdiv = cfg.attacker.diversity_confusion_override
    attacker = Attacker(cfg.attacker, service_diversity_enabled=sdiv)
    attacker.start(rng)

    horizon = cfg.engine.horizon
    compromised = False
    last_tick = horizon
    for tick in range(1, horizon + 1):
        defender.step(system, rng)
        if attacker.step(system, rng):
            compromised = True
            last_tick = tick
            break

    return TrialResult(
        seed=seed,
        compromised=compromised,
        ttc=last_tick if compromised else None,
        trial_ticks=last_tick,
        overhead=defender.total_overhead,
        forced_recons=attacker.forced_recons,
        wrong_target_failures=attacker.wrong_target_failures,
        stale_fraction=attacker.stale_fraction,
        rounds_completed=attacker.rounds_completed,
        decoys_encountered=attacker.decoys_encountered,
        recon_ticks=attacker.recon_ticks,
        mutation_events=defender.mutation_events,
        total_assets=system.n_total,
        chg_port_rotation=defender.technique_changes["port_rotation"],
        chg_endpoint_mutation=defender.technique_changes["endpoint_mutation"],
        chg_shuffling=defender.technique_changes["shuffling"],
        chg_service_diversity=defender.technique_changes["service_diversity"],
        chg_deception=defender.technique_changes["deception"],
    )


def run_config(
    cfg: Config,
    *,
    parallel: bool = False,
    processes: int | None = None,
) -> list[TrialResult]:
    """Run ``cfg.engine.n_trials`` independent trials for one configuration.

    Results are deterministic in ``(cfg.seed, cfg)`` and identical whether
    ``parallel`` is True or False.
    """
    seeds = trial_seeds(cfg.seed, cfg.engine.n_trials)
    worker = partial(run_trial, cfg)
    if parallel and cfg.engine.n_trials > 1:
        with mp.Pool(processes=processes) as pool:
            # chunk to amortise per-task pickling of cfg
            chunk = max(1, cfg.engine.n_trials // (4 * (processes or mp.cpu_count())))
            return list(pool.imap(worker, seeds, chunksize=chunk))
    return [worker(s) for s in seeds]
