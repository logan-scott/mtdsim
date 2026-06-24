"""Reviewer round 3 (minor): decoy-ratio sensitivity of the deception penalty.

The "structural" deception cost (decoys enlarge the reconfigured surface) is set
by `system.decoy_ratio`, which the cost-weight study held fixed at 0.5. Unlike
the weight grid, decoys change ASP and per-technique change counts, so this
**requires real frontier re-runs per ratio** (the analytic overhead shortcut does
not apply).

For each ``decoy_ratio`` it measures the overhead the 5-technique deception stack
needs over the cheapest (decoy-invariant) technique set to reach ASP 0.25. The
non-deception cumulative sets contain no decoys, so their frontier is run once;
only the deception set is re-run per ratio. ``decoy_ratio = 0`` is handled
gracefully (deception enabled but with zero decoys).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import tables, viz
from ..config import Config
from ..engine import run_config
from ..metrics import interp_monotone
from .common import ExperimentOutput, OutputLayout
from .frequency_by_technique_count import _frequencies, _technique_sets

NAME = "decoy_ratio_sweep"
TARGET_ASP = 0.25


def _set_frontier(cfg: Config, techniques: list[str], *, parallel: bool) -> pd.DataFrame:
    """Per-frequency ASP and mean actual overhead for one technique set."""
    rows: list[dict[str, Any]] = []
    for f in _frequencies(cfg):
        cfg_cell = cfg.with_overrides(
            defender={"enabled_techniques": techniques, "mutation_frequency": f}
        )
        results = run_config(cfg_cell, parallel=parallel)
        rows.append(
            {
                "frequency": f,
                "asp": float(np.mean([r.compromised for r in results])),
                "overhead_mean": float(np.mean([r.overhead for r in results])),
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    ratios = [
        float(r) for r in cfg.experiments.get(NAME, {}).get("decoy_ratios", [0.0, 0.25, 0.5, 1.0])
    ]
    sets = _technique_sets(cfg)
    deception_set = next(s for s in sets if "deception" in s["techniques"])
    nondeception = [s for s in sets if "deception" not in s["techniques"]]

    # Cheapest non-deception set to reach the target ASP (decoy-invariant -> run once).
    cheapest_overhead = np.inf
    cheapest_label = None
    for s in nondeception:
        fr = _set_frontier(cfg, s["techniques"], parallel=parallel)
        oh = float(interp_monotone([TARGET_ASP], fr["asp"], fr["overhead_mean"])[0])
        if np.isfinite(oh) and oh < cheapest_overhead:
            cheapest_overhead, cheapest_label = oh, s["label"]

    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        cfg_r = cfg.with_overrides(system={"decoy_ratio": ratio})
        fr = _set_frontier(cfg_r, deception_set["techniques"], parallel=parallel)
        dec_oh = float(interp_monotone([TARGET_ASP], fr["asp"], fr["overhead_mean"])[0])
        rows.append(
            {
                "decoy_ratio": ratio,
                "n_decoys": int(round(ratio * cfg.system.n_assets)),
                "deception_overhead_at_asp_0.25": dec_oh,
                "cheapest_set": cheapest_label,
                "cheapest_overhead_at_asp_0.25": cheapest_overhead,
                "gap_at_asp_0.25": dec_oh - cheapest_overhead,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.decoy_ratio_sweep(summary, layout.figures)]

    cols = [
        ("decoy_ratio", "decoy ratio", ".2g"),
        ("n_decoys", "\\#decoys", ".0f"),
        ("deception_overhead_at_asp_0.25", "deception overhead", ".1f"),
        ("cheapest_set", "cheapest set", None),
        ("cheapest_overhead_at_asp_0.25", "cheapest overhead", ".1f"),
        ("gap_at_asp_0.25", "gap", ".1f"),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary,
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Decoy-ratio sensitivity of the deception penalty. Overhead the "
            "5-technique deception stack needs over the cheapest (decoy-invariant) "
            "technique set to reach ASP 0.25, as decoy provisioning varies. The gap "
            "grows with the decoy ratio, bounding the ``structural'' deception cost "
            "to the provisioning level."
        ),
        label="tab:decoy_ratio_sweep",
        path=layout.tables / f"{NAME}.tex",
        align="lrrlrr",
    )

    gaps = summary["gap_at_asp_0.25"].to_numpy()
    extra = {
        "target_asp": TARGET_ASP,
        "cheapest_set": cheapest_label,
        "gap_by_decoy_ratio": {
            str(r): float(g) for r, g in zip(summary["decoy_ratio"], gaps, strict=True)
        },
        "gap_positive_at_all_ratios": bool(np.all(gaps > 0)),
        "gap_monotone_increasing": bool(np.all(np.diff(gaps) >= -1e-9)),
    }
    return ExperimentOutput(
        name=NAME, summary=summary, figures=figures, tables=[{"csv": csv, "tex": tex}], extra=extra
    )
