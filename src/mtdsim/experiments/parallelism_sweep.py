"""Parallel multi-target probing as a genuine agent capability.

The attacker's ``parallelism`` (assets probed concurrently per reconnaissance
round) is activated here as a study: an AI agent that scans many targets at once
gathers a wider snapshot and is likelier to include the real target in its
candidate set. We sweep ``parallelism in {1,2,4,8}`` against a representative
mutation-frequency grid and report ASP/TTC, so the paper can state how an
agent-like capability shifts the MTD trade-off without changing the other
experiments' baselines (which keep ``parallelism = None`` / probe-all).
"""

from __future__ import annotations

from .. import tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "parallelism_sweep"


def build_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get(NAME, {})
    parallelisms = spec.get("parallelisms", [1, 2, 4, 8])
    freqs = spec.get("frequencies")
    if not freqs:
        raise ValueError(f"experiments.{NAME}.frequencies is required")
    cells: list[Cell] = []
    for par in parallelisms:
        for f in freqs:
            cells.append(
                Cell(
                    {"parallelism": int(par), "frequency": float(f)},
                    cfg.with_overrides(
                        defender={"mutation_frequency": float(f)},
                        attacker={"parallelism": int(par)},
                    ),
                )
            )
    return cells


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="cell")
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.parallelism_sweep(summary, layout.figures)]

    cols = [
        ("parallelism", "probes", ".0f"),
        ("frequency", "$f$", ".4g"),
        ("asp", "ASP", ".3f"),
        ("ttc_median", "med.\\ TTC", ".1f"),
        ("n_success", "$n_{succ}$", ".0f"),
        ("overhead_mean", "overhead", ".1f"),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary.sort_values(["parallelism", "frequency"]),
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Parallel multi-target probing: ASP/TTC as the agent probes more assets "
            "per reconnaissance round, across mutation frequencies. More concurrent "
            "probing raises success but does not remove the ASP decline with $f$."
        ),
        label="tab:parallelism_sweep",
        path=layout.tables / f"{NAME}.tex",
    )
    return ExperimentOutput(
        name=NAME, summary=summary, figures=figures, tables=[{"csv": csv, "tex": tex}]
    )
