"""Experiment 1: static perimeter defense (f=0) vs a representative MTD config.

Reports ASP and TTC for the two conditions as a CSV and a LaTeX comparison
table. This is the headline "does MTD help at all?" result.
"""

from __future__ import annotations

from .. import tables
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "baseline"


def build_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get("baseline", {})
    static_f = float(spec.get("static_frequency", 0.0))
    mtd_f = float(spec.get("mtd_frequency", 0.1))
    return [
        Cell(
            {"condition": "Static (f=0)", "frequency": static_f},
            cfg.with_overrides(defender={"mutation_frequency": static_f}),
        ),
        Cell(
            {"condition": f"MTD (f={mtd_f:g})", "frequency": mtd_f},
            cfg.with_overrides(defender={"mutation_frequency": mtd_f}),
        ),
    ]


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="cond")
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    cols = [
        ("condition", "Condition", None),
        ("asp", "ASP", ".3f"),
        ("asp_ci_lo", "ASP$_{lo}$", ".3f"),
        ("asp_ci_hi", "ASP$_{hi}$", ".3f"),
        ("ttc_median", "med.\\ TTC", ".1f"),
        ("ttc_mean", "mean TTC", ".1f"),
        ("overhead_mean", "overhead", ".1f"),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary,
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Static perimeter defense versus a representative MTD configuration. "
            "MTD reduces attack success probability and increases time to "
            "compromise."
        ),
        label="tab:baseline",
        path=layout.tables / f"{NAME}.tex",
    )
    return ExperimentOutput(name=NAME, summary=summary, tables=[{"csv": csv, "tex": tex}])
