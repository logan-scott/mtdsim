"""Experiment 2 (headline): mutation-frequency sweep.

Sweeps ``mutation_frequency`` across the configured grid with the full technique
set and produces the headline figures (ASP, median TTC, overhead, and attacker
uncertainty vs. ``f``) plus the overhead-vs-ASP Pareto frontier with its knee
annotated, and the master summary table (CSV + LaTeX).
"""

from __future__ import annotations

import pandas as pd

from .. import metrics, tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "frequency_sweep"
DEFAULT_KNEE_MAX_FREQS = [0.20, 0.30, 0.50]


def _knee_robustness(cfg: Config, summary: pd.DataFrame) -> pd.DataFrame:
    """How the kneedle knee moves as the sweep's maximum frequency is restricted.

    The kneedle chord anchors on the sweep's max-frequency endpoint, so the knee
    is sensitive to where the sweep stops (reviewer item M3). This reports the
    knee over several truncations of the frequency range.
    """
    maxes = cfg.experiments.get(NAME, {}).get("knee_max_frequencies", DEFAULT_KNEE_MAX_FREQS)
    rows = []
    for fmax in maxes:
        sub = summary[summary["frequency"] <= float(fmax)].reset_index(drop=True)
        if len(sub) < 2:
            continue
        k = metrics.pareto_knee(sub["overhead_mean"], sub["asp"])
        rows.append(
            {
                "max_frequency": float(fmax),
                "n_points": int(len(sub)),
                "knee_frequency": float(sub.iloc[k["index"]]["frequency"]),
                "knee_overhead": k["overhead"],
                "knee_asp": k["asp"],
            }
        )
    return pd.DataFrame(rows)


def build_cells(cfg: Config) -> list[Cell]:
    freqs = cfg.experiments.get("frequency_sweep", {}).get("frequencies")
    if not freqs:
        raise ValueError("experiments.frequency_sweep.frequencies is required")
    return [
        Cell({"frequency": float(f)}, cfg.with_overrides(defender={"mutation_frequency": float(f)}))
        for f in freqs
    ]


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="freq")
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    knee = metrics.pareto_knee(summary["overhead_mean"], summary["asp"])
    knee["frequency"] = float(summary.iloc[knee["index"]]["frequency"])

    figures = [
        viz.asp_vs_frequency(summary, layout.figures),
        viz.ttc_vs_frequency(summary, layout.figures),
        viz.overhead_vs_frequency(summary, layout.figures),
        viz.attacker_uncertainty_vs_frequency(summary, layout.figures),
        viz.pareto_overhead_vs_asp(summary, knee, layout.figures),
    ]
    table = tables.write_sweep_table(
        summary,
        caption=(
            "Mutation-frequency sweep: attack success probability (ASP), time to "
            "compromise (TTC), operational overhead, and attacker-uncertainty "
            "signals for the full MTD technique set."
        ),
        label="tab:frequency_sweep",
        csv_path=layout.tables / f"{NAME}.csv",
        tex_path=layout.tables / f"{NAME}.tex",
    )

    # M3: knee robustness vs. the sweep's maximum frequency.
    knee_df = _knee_robustness(cfg, summary)
    knee_df.to_csv(layout.summary / f"{NAME}_knee_robustness.csv", index=False)
    knee_tab = {
        "csv": tables.write_csv(knee_df, layout.tables / f"{NAME}_knee_robustness.csv"),
        "tex": tables.to_booktabs(
            knee_df,
            columns=["max_frequency", "n_points", "knee_frequency", "knee_overhead", "knee_asp"],
            headers=["max $f$", "points", "knee $f$", "knee overhead", "knee ASP"],
            formats=[".2g", ".0f", ".4g", ".1f", ".3f"],
            caption=(
                "Pareto-knee robustness: the kneedle knee as the sweep's maximum "
                "frequency is truncated. Because the chord anchors on the "
                "max-frequency endpoint, the knee shifts with the sweep range."
            ),
            label="tab:knee_robustness",
            path=layout.tables / f"{NAME}_knee_robustness.tex",
        ),
    }
    return ExperimentOutput(
        name=NAME,
        summary=summary,
        figures=figures,
        tables=[table, knee_tab],
        extra={"pareto_knee": knee, "knee_robustness": knee_df.to_dict(orient="records")},
    )
