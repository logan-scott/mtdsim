"""Experiment 3: cumulative technique ablation.

Enables MTD techniques incrementally (port only -> +endpoint -> +shuffling ->
+diversity -> +deception) at a fixed representative mutation frequency and
reports ASP per configuration as a bar chart and table.
"""

from __future__ import annotations

from .. import tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "ablation"


def build_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get("ablation", {})
    steps = spec.get("steps")
    freq = float(spec.get("frequency", 0.1))
    if not steps:
        raise ValueError("experiments.ablation.steps is required")
    cells: list[Cell] = []
    for i, step in enumerate(steps):
        techniques = list(step.get("techniques", []))
        cells.append(
            Cell(
                {
                    "step": step["name"],
                    "step_order": i,
                    "techniques": ",".join(techniques) or "(none)",
                    "frequency": freq,
                },
                cfg.with_overrides(
                    defender={"enabled_techniques": techniques, "mutation_frequency": freq}
                ),
            )
        )
    return cells


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="step")
    summary = summary.sort_values("step_order").reset_index(drop=True)
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.technique_ablation(summary, layout.figures)]

    cols = [
        ("step", "Configuration", None),
        ("techniques", "Techniques enabled", None),
        ("asp", "ASP", ".3f"),
        ("asp_ci_lo", "ASP$_{lo}$", ".3f"),
        ("asp_ci_hi", "ASP$_{hi}$", ".3f"),
        ("ttc_median", "med.\\ TTC", ".1f"),
        ("overhead_mean", "overhead", ".1f"),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary,
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Cumulative technique ablation at a fixed mutation frequency. Adding "
            "each MTD/deception mechanism further reduces attack success "
            "probability."
        ),
        label="tab:ablation",
        path=layout.tables / f"{NAME}.tex",
        align="llrrrrr",
    )
    return ExperimentOutput(
        name=NAME, summary=summary, figures=figures, tables=[{"csv": csv, "tex": tex}]
    )
