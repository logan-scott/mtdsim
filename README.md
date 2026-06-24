# MTD-vs-AI Attacker Simulation (`mtdsim`)

A reproducible, well-tested Monte Carlo simulation that quantifies how **Moving
Target Defense (MTD)** degrades an **adaptive AI-driven attacker**. It produces
the publication-ready figures and tables for the IEEE/INTCEC 2026 paper and the
PhD dissertation *Moving Target Defense Against AI Exploits*.

The headline result is the **mutation-frequency vs. operational-overhead
trade-off**: as the defender mutates observable system attributes more often,
the attacker's success probability and progress fall while operational cost
rises. The simulation locates the Pareto frontier and its **knee**.

> **Defensive research only.** This is a pure, in-memory mathematical model. It
> contains **no exploit code, malware, scanning, network access, or LLM calls**.
> The "attacker" is an abstract staged process governed by timers, knowledge
> validity, and probabilities. See [Ethics & scope](#ethics--scope).

---

## Metric definitions

These map one-to-one to the dissertation's *Definitions* section.

| Term | Symbol | Definition (as implemented) |
|---|---|---|
| **Attack success probability** | ASP | Fraction of trials in which the attacker compromises the *real* target within the time horizon `T`. Reported with a 95% **Wilson** interval. |
| **Time to compromise** | TTC | Elapsed simulated ticks from attack start to compromise, **recorded only for successful trials**. Mean (normal CI) and median (bootstrap CI). |
| **Attacker uncertainty** | — | Degree to which attacker knowledge is invalidated: **forced re-reconnaissance** count and **time-averaged stale-knowledge fraction**. |
| **Mutation frequency** | `f` | Per-tick probability that each enabled defender technique reconfigures its attribute. `f = 0` is static perimeter defense. |
| **Operational overhead** | — | Cumulative defender cost from reconfiguration: `per_mutation_cost · weight · (#attributes changed)`, summed over the engagement. |

---

## The model in one paragraph

A discrete-time contest between one defender and one attacker over a system of
`N` assets (exactly one is the real vulnerable target; optional **decoys** are
added when deception is enabled). Each asset exposes four observable attributes
(`port`, `endpoint_path`, `service_fingerprint`, `ip_binding`). Each tick, every
enabled MTD technique independently fires with probability `f`, re-randomising
its attribute (and accruing overhead). The attacker runs a four-stage kill chain
— **recon → identify → exploit-dev → execute** — over a knowledge base it builds
during recon. If the defender mutates an attribute the attacker has committed
to, the in-progress action fails and the attacker is forced back to recon (a
*forced re-recon*). A trial ends at compromise (record TTC) or at the horizon
`T`. Run many seeded trials per configuration and aggregate. Full modeling
rationale is in [`DECISIONS.md`](DECISIONS.md).

Module map (each module's docstring explains its model→metric mapping):

```
src/mtdsim/
  config.py      typed, validated YAML configuration
  rng.py         centralized reproducible seeding (SeedSequence)
  system.py      assets + observable attributes
  defender.py    MTD techniques + overhead accounting
  attacker.py    staged kill-chain, knowledge/staleness, adaptivity
  engine.py      per-trial loop, termination, parallel trials
  metrics.py     ASP / TTC / uncertainty aggregation, CIs, Pareto knee
  viz.py         grayscale, print-safe figures
  tables.py      CSV + LaTeX (booktabs) tables
  manifest.py    run manifest (config, seeds, git commit, hashes)
  experiments/   the five experiments + run_all driver
```

---

## Install

Requires Python 3.11+ (developed/tested on 3.13).

```bash
python3 -m venv .venv && source .venv/bin/activate
make install          # pip install -r requirements.txt && pip install -e .
# or manually:
pip install -r requirements.txt
pip install -e .
```

## Run everything (one command)

```bash
make all
# equivalently:
python -m mtdsim.experiments.run_all --config configs/paper.yaml
```

This regenerates **every** figure (`figures/`, grayscale `.pdf`+`.png`), table
(`tables/`, `.csv`+`.tex`), raw and summary data (`results/`), figure captions
(`figures/CAPTIONS.md`), and a JSON **run manifest** (`results/manifest.json`)
recording the config, seed, git commit, environment, and SHA-256 of every output
— with no manual steps. Add `--parallel` to spread trials across CPU cores.

Run a single experiment or a quick smoke run:

```bash
mtdsim list                                   # list experiments
mtdsim run frequency_sweep --config configs/paper.yaml --parallel
mtdsim run-all --trials 100 --seed 7          # fast low-trial sanity run
```

## Test

```bash
make test            # full suite (includes slow statistical + parallel tests)
make test-fast       # pytest -m "not slow"
```

---

## Experiments → figures → paper claims

| # | Experiment | Command | Figures / Tables | Paper claim it supports |
|---|---|---|---|---|
| 1 | **Baseline** static vs MTD | `mtdsim run baseline` | `tables/baseline.*` | MTD lowers ASP and raises TTC vs. a static perimeter. |
| 2 | **Frequency sweep** (headline) | `mtdsim run frequency_sweep` | `asp_vs_frequency`, `ttc_vs_frequency`, `overhead_vs_frequency`, `attacker_uncertainty_vs_frequency`, `pareto_overhead_vs_asp`; `tables/frequency_sweep.*` | ASP/progress fall and overhead rises with `f`; a Pareto **knee** identifies the best balance. |
| 3 | **Technique ablation** | `mtdsim run ablation` | `technique_ablation`; `tables/ablation.*` | Each added MTD/deception mechanism further degrades attacker success. |
| 4 | **Adaptive vs non-adaptive** | `mtdsim run adaptive_vs_nonadaptive` | `adaptive_vs_nonadaptive`; `tables/adaptive_vs_nonadaptive.*` | Adaptive attackers are more resilient, but MTD degrades both. |
| 5 | **Sensitivity** | `mtdsim run sensitivity` | `sensitivity_asp_heatmaps`; `tables/sensitivity_*.*` | The effect is robust to attacker speed and system size `N`. |

All figures are grayscale and print-safe (distinct line styles + markers, no
color dependence), 300 dpi. Captions are auto-written to `figures/CAPTIONS.md`.

### Revision experiments (address Reviewer 2)

These were added to support an honest rewrite; see **[`RESULTS-DELTA.md`](RESULTS-DELTA.md)**
for the verdicts and **[`DECISIONS.md`](DECISIONS.md)** for the mechanics.

| Reviewer item | Experiment | Command | Figures / Tables | What it answers |
|---|---|---|---|---|
| **C1** (blocking) | **Frequency × technique count** | `mtdsim run frequency_by_technique_count` | `frontier_by_technique_count`; `tables/frequency_by_technique_count_matched_{overhead,asp}.*` | Is stacking techniques (esp. deception) ever cheaper than raising `f`? Builds the real diversity-vs-frequency frontier (round 4 adds non-cumulative **lean-deception** candidates `{port,deception}`, `{port,endpoint,deception}`). |
| **Ma** | **Cost-weight sensitivity** | `mtdsim run cost_weight_sensitivity` | `cost_weight_sensitivity`; `tables/cost_weight_sensitivity.*` | Is the lever ranking an artifact of the asserted cost weights? Recomputes the frontier under a 5-way factorial over **all four mutating** weights (port/endpoint/shuffling/diversity, rounds 3–4) and the decoy weight + all-equal control, from one run. |
| **R3** | **Decoy-ratio sweep** | `mtdsim run decoy_ratio_sweep` | `decoy_ratio_sweep`; `tables/decoy_ratio_sweep.*` | How does the structural deception penalty scale with decoy provisioning? Real frontier re-runs per `decoy_ratio`. |
| **C2a** (blocking) | (mechanic, no new experiment) | — | — | Identification "learning" now comes only from genuine decoy encounters (`learning_signal`), not from MTD-forced re-recons. Re-run `adaptive_vs_nonadaptive`. |
| **C2b** (blocking) | **Parallelism sweep** | `mtdsim run parallelism_sweep` | `parallelism_sweep`; `tables/parallelism_sweep.*` | Activates a genuine agent capability (concurrent multi-target probing) and reports how it shifts the MTD trade-off. |
| **M6** | **Diversity channel decomposition** | `mtdsim run diversity_channel_decomposition` | `diversity_channel_decomposition`; `tables/diversity_channel_decomposition.*` | Splits `service_diversity` into its mutation vs. identification-confusion channels. |
| **M3** | (extends frequency sweep) | `mtdsim run frequency_sweep` | `tables/frequency_sweep_knee_robustness.*` | How the Pareto knee moves as the sweep's max frequency is truncated; overhead now also reported per tick and per asset. |

**Reproduce the pre-revision ("legacy") numbers** for auditing the deltas:

```bash
python -m mtdsim.experiments.run_all --config configs/legacy.yaml --outdir results_legacy \
    --figures figures_legacy --tables tables_legacy
```

The only behavioral difference is `attacker.learning_signal: rounds` (the perverse
coupling flagged as C2a).

---

## Reproducibility & determinism

- A single master `seed` plus `numpy.random.SeedSequence(seed).spawn(n)` gives
  one independent RNG stream per trial. There is **no** wall-clock or unseeded
  randomness.
- Trial seeds are materialised in the parent process, so results are **identical
  whether trials run serially or in parallel** (verified by a test).
- Re-running `run_all` with the same config reproduces identical aggregate
  numbers (verified by a determinism test).
- `requirements.txt` pins exact versions; the manifest records the resolved
  environment and git commit.

---

## Configuration

All parameters live in YAML (no magic numbers in code), validated on load.

- `configs/default.yaml` — documented defaults for every knob.
- `configs/paper.yaml` — the publication run, including the `experiments:` grids
  (sweep frequencies, ablation steps, sensitivity ranges).

Override at the CLI with `--seed` and `--trials`, or point `--config` at your own
file. Programmatically, `Config.with_overrides(...)` returns a validated copy.

---

## How to add a new MTD technique

The model is structured so a new attribute-mutating technique is a few lines:

1. **Register the attribute & mapping.** In `src/mtdsim/__init__.py`, add the new
   observable attribute to `ATTRIBUTES` and add `"<technique>": "<attribute>"` to
   `TECHNIQUE_ATTRIBUTE` (which extends `ALL_TECHNIQUES` automatically). If the
   technique manages decoys rather than an attribute (like `deception`), instead
   add special handling in `defender.Defender.step`.
2. **Give it a cost weight.** Add a default weight in
   `config.DefenderConfig.technique_cost_weights` and in `configs/default.yaml`.
3. **Enable it.** Add the technique name to `defender.enabled_techniques` in your
   config (and to an `experiments.ablation.steps` entry if you want it in the
   ablation).
4. **(Optional) identification effect.** If the technique should also raise
   identification error (like `service_diversity`), thread a flag into
   `attacker.Attacker._p_correct`.
5. **Test it.** Add a `defender` overhead test and, if it affects identification,
   an `attacker` test. Existing tests already cover the mutation/staleness path
   generically, so a new attribute technique is exercised by the sweep.

No engine changes are needed: `system.System` stores attributes by name, the
defender iterates over `enabled_attribute_techniques`, and the attacker's
staleness check compares all recorded attributes against live values.

---

## Ethics & scope

This project is **defensive security research**: an abstract simulation for
quantitative evaluation. It does not generate exploit code, malware, or attack
tooling, performs no reconnaissance/scanning, and never touches real systems,
networks, or models. The attacker is modeled with stage timers, knowledge
validity, and probabilities only. Every modeling assumption that could affect
results is documented in [`DECISIONS.md`](DECISIONS.md) so the work is auditable
and defensible for committee review.

## License

MIT (see `pyproject.toml`).


---

## Citation

If you use this code or data, please cite:

```bibtex
@misc{scott2026,
    author = {Logan Scott},
    year = {2026},
    month = {June},
    title = {mtdsim},
    url = {https://github.com/logan-scott/mtdsim},
    howpublished= {https://github.com/logan-scott/mtdsim},
}
```
