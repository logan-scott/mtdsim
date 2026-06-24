"""Experiment 5: sensitivity analysis.

Two crossed grids show how the headline result responds to nuisance parameters:

* **Attacker speed** — all kill-chain stage means are scaled by a multiplier
  (``<1`` = faster attacker) and crossed with a few mutation frequencies.
* **System size** — the number of assets ``N`` is varied and crossed with the
  same frequencies.

Both are rendered as grayscale ASP heatmaps and written as CSV + LaTeX tables.
"""

from __future__ import annotations

import pandas as pd

from .. import tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "sensitivity"


def build_speed_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get(NAME, {})
    freqs = [float(f) for f in spec.get("frequencies", [])]
    mults = [float(m) for m in spec.get("attacker_speed_multipliers", [])]
    a = cfg.attacker
    cells: list[Cell] = []
    for m in mults:
        scaled = {
            "recon_time": a.recon_time * m,
            "identify_time": a.identify_time * m,
            "exploit_dev_time": a.exploit_dev_time * m,
            "exploit_exec_time": a.exploit_exec_time * m,
        }
        for f in freqs:
            cells.append(
                Cell(
                    {"speed_multiplier": m, "frequency": f},
                    cfg.with_overrides(defender={"mutation_frequency": f}, attacker=scaled),
                )
            )
    return cells


def build_size_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get(NAME, {})
    freqs = [float(f) for f in spec.get("frequencies", [])]
    sizes = [int(n) for n in spec.get("n_assets", [])]
    cells: list[Cell] = []
    for n in sizes:
        for f in freqs:
            cells.append(
                Cell(
                    {"n_assets": n, "frequency": f},
                    cfg.with_overrides(defender={"mutation_frequency": f}, system={"n_assets": n}),
                )
            )
    return cells


def _flat_table(
    df: pd.DataFrame, key: str, key_head: str, caption: str, label: str, csv_path, tex_path
) -> dict[str, str]:
    cols = [
        (key, key_head, ".4g"),
        ("frequency", "$f$", ".4g"),
        ("asp", "ASP", ".3f"),
        ("ttc_median", "med.\\ TTC", ".1f"),
        ("overhead_mean", "overhead", ".1f"),
    ]
    csv = tables.write_csv(df, csv_path)
    tex = tables.to_booktabs(
        df.sort_values([key, "frequency"]),
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=caption,
        label=label,
        path=tex_path,
    )
    return {"csv": csv, "tex": tex}


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    speed = run_cells(
        build_speed_cells(cfg), parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="speed"
    )
    size = run_cells(
        build_size_cells(cfg), parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="size"
    )
    speed.to_csv(layout.summary / f"{NAME}_speed.csv", index=False)
    size.to_csv(layout.summary / f"{NAME}_size.csv", index=False)

    figures = [viz.sensitivity_heatmaps(speed, size, layout.figures)]
    tabs = [
        _flat_table(
            speed,
            "speed_multiplier",
            "speed$\\times$",
            "Sensitivity to attacker speed (stage-time multiplier) across mutation " "frequencies.",
            "tab:sensitivity_speed",
            layout.tables / f"{NAME}_speed.csv",
            layout.tables / f"{NAME}_speed.tex",
        ),
        _flat_table(
            size,
            "n_assets",
            "$N$",
            "Sensitivity to system size (number of assets $N$) across mutation " "frequencies.",
            "tab:sensitivity_size",
            layout.tables / f"{NAME}_size.csv",
            layout.tables / f"{NAME}_size.tex",
        ),
    ]
    combined = pd.concat([speed.assign(grid="speed"), size.assign(grid="size")], ignore_index=True)
    return ExperimentOutput(name=NAME, summary=combined, figures=figures, tables=tabs)
