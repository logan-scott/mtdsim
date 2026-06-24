"""Reviewer item C1: a real diversity-vs-frequency frontier.

Runs the full mutation-frequency sweep *separately* for each cumulative
technique set (1->5 techniques, derived from the ablation steps), so the
overhead-vs-ASP trade-off of "raise f on N techniques" can be compared directly
against "stack more techniques." Produces:

* one overhead-vs-ASP curve per technique set, with the global Pareto frontier
  (non-dominated points) highlighted;
* a matched-overhead table (ASP each set achieves at the 4-technique set's
  overhead levels) and a matched-ASP table (overhead each set needs to reach
  target ASP levels);
* a machine-checked verdict (in ``extra``) on whether any technique-stacked
  configuration is ever Pareto-superior to the 4-technique frequency sweep.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import metrics, tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "frequency_by_technique_count"
REFERENCE_K = 4  # the "pure-frequency" reference set has 4 techniques


def _technique_sets(cfg: Config) -> list[dict[str, Any]]:
    """Cumulative non-empty technique sets, from the ablation steps by default.

    This is the *original* cumulative frontier ({port} ⊂ {port,endpoint} ⊂ …).
    Kept stable so experiments that depend only on it (e.g. ``decoy_ratio_sweep``)
    stay byte-identical.
    """
    spec = cfg.experiments.get(NAME, {})
    if spec.get("technique_sets"):
        steps = spec["technique_sets"]
    else:
        steps = cfg.experiments.get("ablation", {}).get("steps", [])
    sets = []
    for step in steps:
        techs = list(step.get("techniques", []))
        if techs:  # skip the empty "static" step
            sets.append({"label": step["name"], "techniques": techs, "n": len(techs)})
    return sets


def _extra_technique_sets(cfg: Config) -> list[dict[str, Any]]:
    """Additional *non-cumulative* candidate sets (reviewer round 4, item a).

    The cumulative frontier never tests a lean deception config (cheap invalidation
    + decoy confusion), so ``deception`` only ever appears on the full 5-stack. These
    added sets — e.g. ``{port, deception}`` and ``{port, endpoint, deception}`` — are
    exactly where decoys could be cost-effective, so they are the only candidates
    that can *falsify* "deception is never the cheapest lever." Config-driven via
    ``experiments.frequency_by_technique_count.extra_technique_sets``; absent =>
    none (so other configs are unaffected).
    """
    steps = cfg.experiments.get(NAME, {}).get("extra_technique_sets", [])
    sets = []
    for step in steps:
        techs = list(step.get("techniques", []))
        if techs:
            sets.append({"label": step["name"], "techniques": techs, "n": len(techs)})
    return sets


def _frontier_sets(cfg: Config) -> list[dict[str, Any]]:
    """Full candidate set for the frontier: cumulative sets + any extra sets."""
    return _technique_sets(cfg) + _extra_technique_sets(cfg)


def _frequencies(cfg: Config) -> list[float]:
    spec = cfg.experiments.get(NAME, {})
    freqs = spec.get("frequencies") or cfg.experiments.get("frequency_sweep", {}).get("frequencies")
    if not freqs:
        raise ValueError(f"experiments.{NAME}.frequencies (or frequency_sweep) is required")
    return [float(f) for f in freqs]


def build_cells(cfg: Config) -> list[Cell]:
    cells: list[Cell] = []
    for s in _frontier_sets(cfg):
        for f in _frequencies(cfg):
            cells.append(
                Cell(
                    {
                        "set_label": s["label"],
                        "n_techniques": s["n"],
                        "techniques": ",".join(s["techniques"]),
                        "frequency": f,
                    },
                    cfg.with_overrides(
                        defender={"enabled_techniques": s["techniques"], "mutation_frequency": f}
                    ),
                )
            )
    return cells


def _matched_overhead_table(summary: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    """ASP each technique set achieves at the reference set's overhead levels."""
    ref = summary[summary["set_label"] == ref_label].sort_values("overhead_mean")
    rows = []
    set_labels = list(summary.sort_values("n_techniques")["set_label"].unique())
    for _, rr in ref.iterrows():
        o = rr["overhead_mean"]
        row: dict[str, Any] = {"overhead": o, "ref_f": rr["frequency"]}
        for lbl in set_labels:
            g = summary[summary["set_label"] == lbl]
            asp = metrics.interp_monotone([o], g["overhead_mean"], g["asp"])[0]
            row[lbl] = asp
        rows.append(row)
    return pd.DataFrame(rows)


def _matched_asp_table(summary: pd.DataFrame, asp_targets: list[float]) -> pd.DataFrame:
    """Overhead each technique set needs to reach target ASP levels."""
    set_labels = list(summary.sort_values("n_techniques")["set_label"].unique())
    rows = []
    for a in asp_targets:
        row: dict[str, Any] = {"asp_target": a}
        for lbl in set_labels:
            g = summary[summary["set_label"] == lbl].sort_values("asp")
            # overhead as a function of asp (asp ascending for interp)
            oh = metrics.interp_monotone([a], g["asp"], g["overhead_mean"])[0]
            row[lbl] = oh
        rows.append(row)
    return pd.DataFrame(rows)


def _verdict(summary: pd.DataFrame, ref_label: str) -> dict[str, Any]:
    """Answer the reviewer's C1 question robustly.

    Two distinct questions are reported separately:

    1. Does the *deception stack* (the set with the most techniques) sit below the
       4-technique frequency sweep? This is the paper's "diversity/deception is
       cheaper than frequency" claim.
    2. Is the 4-technique set even Pareto-optimal across technique counts, or does
       raising ``f`` on a cheaper, smaller set dominate it?

    The ``f=0`` static point ``(overhead=0, ASP=1)`` is shared by every set, so it
    is excluded from the frontier composition (it is a trivial tie).
    """
    a_grid = [0.75, 0.5, 0.25, 0.1]
    stacked_label = summary.sort_values("n_techniques")["set_label"].iloc[-1]
    ref = summary[summary["set_label"] == ref_label]
    stacked = summary[summary["set_label"] == stacked_label]

    # (1) deception stack vs 4-technique reference, at matched ASP and matched overhead.
    ref_oh_at_a = metrics.interp_monotone(a_grid, ref["asp"], ref["overhead_mean"])
    stk_oh_at_a = metrics.interp_monotone(a_grid, stacked["asp"], stacked["overhead_mean"])
    oh_gap = stk_oh_at_a - ref_oh_at_a  # >0 => stacking costs MORE overhead for same ASP
    o_grid = ref.loc[ref["overhead_mean"] > 0, "overhead_mean"].to_numpy()
    ref_asp_at_o = metrics.interp_monotone(o_grid, ref["overhead_mean"], ref["asp"])
    stk_asp_at_o = metrics.interp_monotone(o_grid, stacked["overhead_mean"], stacked["asp"])
    asp_gap = stk_asp_at_o - ref_asp_at_o  # <0 => stacking achieves LOWER ASP at same overhead

    # Robust verdict: the matched-ASP overhead comparison. Stacking is "cheaper"
    # only if, to reach the SAME ASP, it needs LESS overhead on average. The
    # matched-overhead ASP gap is reported too but kept out of the boolean because
    # at fine overhead resolution a single point can dip within Monte Carlo noise.
    finite_oh_gap = oh_gap[np.isfinite(oh_gap)]
    mean_oh_gap = float(np.mean(finite_oh_gap)) if finite_oh_gap.size else float("nan")
    stacking_cheaper = bool(mean_oh_gap < 0.0)

    # (2) cheapest technique set to reach a representative ASP target.
    target = 0.25
    cheapest_overhead = np.inf
    cheapest_set = None
    per_set: dict[str, Any] = {}
    for lbl in summary["set_label"].unique():
        g = summary[summary["set_label"] == lbl]
        oh = float(metrics.interp_monotone([target], g["asp"], g["overhead_mean"])[0])
        per_set[lbl] = {"overhead_to_reach_asp_0.25": oh}
        if np.isfinite(oh) and oh < cheapest_overhead:
            cheapest_overhead, cheapest_set = oh, lbl

    # Global frontier composition, excluding the shared static (overhead==0) point.
    nontrivial = summary[(summary["on_frontier"]) & (summary["overhead_mean"] > 0)]
    frontier_counts = nontrivial.groupby("set_label").size().sort_index().to_dict()

    return {
        "reference_set": ref_label,
        "stacked_set": stacked_label,
        "stacking_cheaper_than_frequency": stacking_cheaper,
        "stacked_vs_ref_mean_overhead_gap_at_matched_asp": mean_oh_gap,
        "stacked_vs_ref_overhead_gap_at_matched_asp": {
            str(a): float(g) for a, g in zip(a_grid, oh_gap, strict=True)
        },
        "stacked_vs_ref_min_asp_gap_at_matched_overhead": float(np.nanmin(asp_gap)),
        "cheapest_set_to_reach_asp_0.25": cheapest_set,
        "cheapest_overhead_to_reach_asp_0.25": float(cheapest_overhead),
        "overhead_to_reach_asp_0.25_per_set": per_set,
        "global_frontier_counts_excl_static": frontier_counts,
    }


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="cell")
    mask = metrics.pareto_frontier(summary["overhead_mean"], summary["asp"])
    summary = summary.assign(on_frontier=mask)
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    # Reference = the 4-technique set (falls back to the largest set < deception).
    ref_rows = summary[summary["n_techniques"] == REFERENCE_K]
    ref_label = (
        ref_rows["set_label"].iloc[0]
        if len(ref_rows)
        else summary.sort_values("n_techniques")["set_label"].iloc[-2]
    )

    figures = [viz.frontier_by_technique_count(summary, layout.figures)]

    matched_oh = _matched_overhead_table(summary, ref_label)
    matched_asp = _matched_asp_table(summary, [0.75, 0.5, 0.25, 0.1])
    set_labels = list(summary.sort_values("n_techniques")["set_label"].unique())

    tabs = []
    # Matched-overhead table.
    cols = ["overhead", "ref_f", *set_labels]
    heads = ["overhead", "ref.\\ $f$", *[lbl.replace("_", "\\_") for lbl in set_labels]]
    fmts = [".1f", ".4g", *[".3f"] * len(set_labels)]
    tabs.append(
        {
            "csv": tables.write_csv(matched_oh, layout.tables / f"{NAME}_matched_overhead.csv"),
            "tex": tables.to_booktabs(
                matched_oh,
                columns=cols,
                headers=heads,
                formats=fmts,
                caption=(
                    "ASP achieved by each cumulative technique set at the "
                    "4-technique set's overhead levels (matched overhead). Lower "
                    "ASP is better for the defender; if frequency-only is cheaper, "
                    "the 4-technique column is lowest in each row."
                ),
                label="tab:c1_matched_overhead",
                path=layout.tables / f"{NAME}_matched_overhead.tex",
            ),
        }
    )
    # Matched-ASP table.
    cols2 = ["asp_target", *set_labels]
    heads2 = ["ASP target", *[lbl.replace("_", "\\_") for lbl in set_labels]]
    fmts2 = [".2f", *[".0f"] * len(set_labels)]
    tabs.append(
        {
            "csv": tables.write_csv(matched_asp, layout.tables / f"{NAME}_matched_asp.csv"),
            "tex": tables.to_booktabs(
                matched_asp,
                columns=cols2,
                headers=heads2,
                formats=fmts2,
                caption=(
                    "Operational overhead each cumulative technique set needs to "
                    "reach a target ASP (matched ASP; '--' = unreachable in range). "
                    "Lower overhead is cheaper; frequency-only on 4 techniques is "
                    "cheaper wherever its entry is smallest."
                ),
                label="tab:c1_matched_asp",
                path=layout.tables / f"{NAME}_matched_asp.tex",
            ),
        }
    )

    verdict = _verdict(summary, ref_label)
    return ExperimentOutput(name=NAME, summary=summary, figures=figures, tables=tabs, extra=verdict)
