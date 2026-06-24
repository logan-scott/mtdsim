"""The five regenerable experiments plus the one-shot ``run_all`` driver.

Each experiment module exposes ``run(cfg, layout, *, parallel) -> ExperimentOutput``
and is self-contained: it builds its configuration cells from the ``experiments``
block of the config, runs the trials, aggregates metrics, writes raw + summary
data, and renders its figures/tables. ``run_all`` orchestrates all five and emits
the run manifest.

Experiments (build spec §3):

1. :mod:`mtdsim.experiments.baseline`     — static vs MTD.
2. :mod:`mtdsim.experiments.frequency_sweep` — headline ASP/TTC/overhead/Pareto.
3. :mod:`mtdsim.experiments.ablation`     — cumulative technique ablation.
4. :mod:`mtdsim.experiments.adaptive`     — adaptive vs non-adaptive attacker.
5. :mod:`mtdsim.experiments.sensitivity`  — attacker speed and system size.
"""

from .common import ExperimentOutput, OutputLayout

__all__ = ["ExperimentOutput", "OutputLayout"]
