"""mtdsim — Monte Carlo simulation of Moving Target Defense (MTD) against an
adaptive AI-driven attacker.

The package quantifies the central trade-off: as the defender mutates observable
system attributes more frequently (``mutation_frequency``), the attacker's
**attack success probability (ASP)** and progress fall while **operational
overhead** rises. The headline output is the Pareto frontier (overhead vs ASP)
and its knee.

Module map (see each module's docstring for the model→metric mapping):

- :mod:`mtdsim.config`    — typed, validated configuration loaded from YAML.
- :mod:`mtdsim.rng`       — centralized, reproducible seeding (SeedSequence).
- :mod:`mtdsim.system`    — assets and their observable attributes.
- :mod:`mtdsim.defender`  — MTD techniques, mutation, and overhead accounting.
- :mod:`mtdsim.attacker`  — staged kill-chain, knowledge/staleness, adaptivity.
- :mod:`mtdsim.engine`    — per-trial loop, termination, parallel trials.
- :mod:`mtdsim.metrics`   — ASP / TTC / uncertainty aggregation, CIs, knee.
- :mod:`mtdsim.experiments` — the five regenerable experiments + run_all.
- :mod:`mtdsim.viz`       — grayscale, print-safe figures.
- :mod:`mtdsim.tables`    — CSV + LaTeX (booktabs) summary tables.
- :mod:`mtdsim.manifest`  — run manifest (config, seeds, git commit, hashes).
"""

from __future__ import annotations

__version__ = "1.0.0"

ATTRIBUTES = ("port", "endpoint_path", "service_fingerprint", "ip_binding")
"""The observable attributes an attacker can learn and MTD can mutate."""

# Maps each attribute-mutating technique to the attribute it reconfigures.
# ``deception`` is special: it manages decoy assets rather than an attribute.
TECHNIQUE_ATTRIBUTE = {
    "port_rotation": "port",
    "endpoint_mutation": "endpoint_path",
    "shuffling": "ip_binding",
    "service_diversity": "service_fingerprint",
}
ALL_TECHNIQUES = (*TECHNIQUE_ATTRIBUTE.keys(), "deception")
