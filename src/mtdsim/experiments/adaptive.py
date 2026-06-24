"""Experiment 4: adaptive vs non-adaptive attacker across the frequency sweep.

Runs the mutation-frequency sweep twice — once with an adaptive attacker, once
with a non-adaptive attacker — and overlays the two ASP curves.
"""

from __future__ import annotations

from .. import tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "adaptive_vs_nonadaptive"


def build_cells(cfg: Config) -> list[Cell]:
    freqs = cfg.experiments.get(NAME, {}).get("frequencies")
    if not freqs:
        raise ValueError(f"experiments.{NAME}.frequencies is required")
    cells: list[Cell] = []
    for adaptive in (True, False):
        mode = "adaptive" if adaptive else "non-adaptive"
        for f in freqs:
            cells.append(
                Cell(
                    {"attacker_mode": mode, "adaptive": adaptive, "frequency": float(f)},
                    cfg.with_overrides(
                        defender={"mutation_frequency": float(f)},
                        attacker={"adaptive": adaptive},
                    ),
                )
            )
    return cells


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="cell")
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.adaptive_vs_nonadaptive(summary, layout.figures)]

    cols = [
        ("attacker_mode", "Attacker", None),
        ("frequency", "$f$", ".4g"),
        ("asp", "ASP", ".3f"),
        ("ttc_median", "med.\\ TTC", ".1f"),
        ("forced_recons_mean", "re-recons", ".2f"),
        ("overhead_mean", "overhead", ".1f"),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary.sort_values(["attacker_mode", "frequency"]),
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Adaptive versus non-adaptive attacker across the mutation-frequency "
            "sweep. The adaptive attacker re-plans efficiently and learns to "
            "disambiguate decoys, sustaining higher ASP, but MTD degrades both."
        ),
        label="tab:adaptive_vs_nonadaptive",
        path=layout.tables / f"{NAME}.tex",
    )
    return ExperimentOutput(
        name=NAME, summary=summary, figures=figures, tables=[{"csv": csv, "tex": tex}]
    )
