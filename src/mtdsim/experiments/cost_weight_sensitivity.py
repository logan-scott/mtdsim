"""Cost-weight sensitivity of the lever ranking.

The headline cost-effectiveness claim — *deception is never the cheapest lever* —
is computed from `defender.technique_cost_weights`, which are asserted. This
experiment stress-tests it against a factorial of weight settings over the candidate
frontier sets (the cumulative ablation sets **plus** the lean-deception sets
`{port,deception}` and `{port,endpoint,deception}`).

The grid varies all four **mutating** techniques
(`port`, `endpoint`, `shuffling`, `service_diversity` ∈ {1,3}) and the **decoy**
technique (`deception` ∈ {1,2.5,5}); deception is not a mutating technique (it rotates
decoys).

**Key efficiency/correctness point.** ASP and the per-technique attribute-change counts
are *independent of cost weights* (weights only scale overhead). So the frontier is run
**once** (capturing per-technique change counts via the instrumented defender), and
overhead is recomputed **analytically** for every weight setting — exact, identical-ASP
across settings, and 1x cost. The all-equal control (every weight = 1) isolates the
decoy-count / trial-length channel from the assigned weights: any residual deception
penalty there is structural, not a weight artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import ALL_TECHNIQUES, tables, viz
from ..config import Config
from ..engine import run_config
from ..metrics import interp_monotone
from .common import ExperimentOutput, OutputLayout
from .frequency_by_technique_count import _frequencies, _frontier_sets

NAME = "cost_weight_sensitivity"
ASP_TARGETS = [0.50, 0.25, 0.10]


def _winner_expensive(weights: dict[str, float]) -> bool:
    """True if any *winning* (frontier-competitive) technique costs more than a port swap.

    The frontier winners are the cheap mutating techniques port/endpoint and — as the
    cost-effective layered set in port-expensive cells — service_diversity.
    """
    return bool(
        weights.get("port_rotation", 1.0) > 1.0
        or weights.get("endpoint_mutation", 1.0) > 1.0
        or weights.get("service_diversity", 1.0) > 1.0
    )


def weight_settings(cfg: Config) -> list[dict[str, Any]]:
    """All-equal control + a factorial over the cost weights.

    The factorial varies all four **mutating** techniques
    (``port``, ``endpoint``, ``shuffling``, ``service_diversity`` ∈ {1,3}) and the
    **decoy** technique (``deception`` ∈ {1,2.5,5}); ``deception`` is not a mutating
    technique (it rotates decoys). The factorial also varies ``service_diversity`` —
    itself the cheapest set in several port-expensive cells — so every frontier
    *winner* is stress-tested.
    ``varies_winners`` flags settings where a winning technique (port/endpoint/diversity)
    costs more than a port swap.
    """
    spec = cfg.experiments.get(NAME, {})
    if spec.get("weight_settings"):
        return [
            {
                "label": w["label"],
                "is_control": bool(w.get("is_control", False)),
                "varies_winners": _winner_expensive(w["weights"]),
                "weights": {t: float(w["weights"].get(t, 1.0)) for t in ALL_TECHNIQUES},
            }
            for w in spec["weight_settings"]
        ]
    ports = [float(p) for p in spec.get("port_weights", [1.0, 3.0])]
    endps = [float(e) for e in spec.get("endpoint_weights", [1.0, 3.0])]
    shus = [float(s) for s in spec.get("shuffling_weights", [1.0, 3.0])]
    divs = [float(v) for v in spec.get("diversity_weights", [1.0, 3.0])]
    decs = [float(d) for d in spec.get("deception_weights", [1.0, 2.5, 5.0])]

    settings: list[dict[str, Any]] = [
        {
            "label": "all-equal (control)",
            "is_control": True,
            "varies_winners": False,
            "weights": dict.fromkeys(ALL_TECHNIQUES, 1.0),
        }
    ]
    for p in ports:
        for e in endps:
            for shu in shus:
                for div in divs:
                    for dec in decs:
                        weights = {
                            "port_rotation": p,
                            "endpoint_mutation": e,
                            "shuffling": shu,
                            "service_diversity": div,
                            "deception": dec,
                        }
                        settings.append(
                            {
                                "label": f"p{p:g} e{e:g} s{shu:g} v{div:g} d{dec:g}",
                                "is_control": False,
                                "varies_winners": _winner_expensive(weights),
                                "weights": weights,
                            }
                        )
    return settings


def _run_frontier_once(cfg: Config, *, parallel: bool) -> pd.DataFrame:
    """Run each (technique-set, frequency) cell once; capture ASP + mean per-technique
    change counts (weight-independent), so overhead can be recomputed analytically."""
    rows: list[dict[str, Any]] = []
    pmc = cfg.defender.per_mutation_cost
    for s in _frontier_sets(cfg):
        for f in _frequencies(cfg):
            cfg_cell = cfg.with_overrides(
                defender={"enabled_techniques": s["techniques"], "mutation_frequency": f}
            )
            results = run_config(cfg_cell, parallel=parallel)
            comp = np.fromiter((r.compromised for r in results), dtype=float)
            row: dict[str, Any] = {
                "set_label": s["label"],
                "n_techniques": s["n"],
                "has_deception": "deception" in s["techniques"],
                "frequency": f,
                "asp": float(comp.mean()),
                "per_mutation_cost": pmc,
                "actual_overhead_mean": float(np.mean([r.overhead for r in results])),
            }
            for t in ALL_TECHNIQUES:
                row[f"mean_chg_{t}"] = float(np.mean([getattr(r, f"chg_{t}") for r in results]))
            rows.append(row)
    return pd.DataFrame(rows)


def _overhead_under(base: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Mean overhead per cell under an arbitrary weight vector (analytic)."""
    oh = np.zeros(len(base))
    for t in ALL_TECHNIQUES:
        oh = oh + weights[t] * base[f"mean_chg_{t}"].to_numpy()
    return base["per_mutation_cost"].to_numpy() * oh


def _ranking_for_setting(base: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
    """Compute the cheapest technique set at each ASP target and the verdict
    booleans under a given weight setting."""
    df = base.assign(overhead=_overhead_under(base, weights))
    set_labels = list(df.sort_values("n_techniques")["set_label"].unique())
    n_tech = {
        lbl: int(df.loc[df["set_label"] == lbl, "n_techniques"].iloc[0]) for lbl in set_labels
    }
    deception_labels = set(df.loc[df["has_deception"], "set_label"].unique())
    stacked = max(set_labels, key=lambda x: n_tech[x])

    out: dict[str, Any] = {}
    lean_all = True
    decep_max_all = True
    for tgt in ASP_TARGETS:
        oh_by_set: dict[str, float] = {}
        for lbl in set_labels:
            g = df[df["set_label"] == lbl]
            oh = float(interp_monotone([tgt], g["asp"], g["overhead"])[0])
            oh_by_set[lbl] = oh
        reachable = {k: v for k, v in oh_by_set.items() if np.isfinite(v)}
        if not reachable:
            out[f"cheapest@{tgt}"] = "--"
            continue
        cheapest = min(reachable, key=reachable.get)
        out[f"cheapest@{tgt}"] = cheapest
        out[f"cheapest_overhead@{tgt}"] = reachable[cheapest]
        out[f"deception_overhead@{tgt}"] = oh_by_set.get(stacked, float("nan"))
        # Cheapest deception-containing set (not just the 5-stack).
        dec_reach = {k: v for k, v in reachable.items() if k in deception_labels}
        out[f"cheapest_deception_overhead@{tgt}"] = (
            min(dec_reach.values()) if dec_reach else float("nan")
        )
        if n_tech[cheapest] > 2:
            lean_all = False
        # deception least cost-effective => stacked set is the most expensive reaching set
        if stacked in reachable and reachable[stacked] < max(reachable.values()):
            decep_max_all = False
    out["lean_dominates"] = bool(lean_all)
    out["deception_least_cost_effective"] = bool(decep_max_all)
    out["stacked_set"] = stacked
    return out


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    base = _run_frontier_once(cfg, parallel=parallel)
    base.to_csv(layout.summary / f"{NAME}_frontier_base.csv", index=False)

    settings = weight_settings(cfg)
    rows: list[dict[str, Any]] = []
    for s in settings:
        rank = _ranking_for_setting(base, s["weights"])
        # Gap = (cheapest deception-containing set) - (overall cheapest set),
        # so gap > 0 means no deception set is the cheapest lever at ASP 0.25.
        gap_025 = rank.get("cheapest_deception_overhead@0.25", float("nan")) - rank.get(
            "cheapest_overhead@0.25", float("nan")
        )
        rows.append(
            {
                "setting_label": s["label"],
                "is_control": s["is_control"],
                "varies_winners": s["varies_winners"],
                "port_weight": s["weights"]["port_rotation"],
                "endpoint_weight": s["weights"]["endpoint_mutation"],
                "shuffling_weight": s["weights"]["shuffling"],
                "diversity_weight": s["weights"]["service_diversity"],
                "deception_weight": s["weights"]["deception"],
                "cheapest@0.50": rank.get("cheapest@0.5", "--"),
                "cheapest@0.25": rank.get("cheapest@0.25", "--"),
                "cheapest@0.10": rank.get("cheapest@0.1", "--"),
                "lean_dominates": rank["lean_dominates"],
                "deception_least_cost_effective": rank["deception_least_cost_effective"],
                "gap_at_asp_0.25": float(gap_025),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.cost_weight_sensitivity(summary, layout.figures)]

    cols = [
        ("setting_label", "p e s v d weights", None),
        ("varies_winners", "var.\\ win.", None),
        ("cheapest@0.50", "cheapest @0.50", None),
        ("cheapest@0.25", "cheapest @0.25", None),
        ("cheapest@0.10", "cheapest @0.10", None),
        ("lean_dominates", "lean dom.", None),
        ("deception_least_cost_effective", "decep.\\ worst", None),
    ]
    csv = tables.write_csv(summary, layout.tables / f"{NAME}.csv")
    tex = tables.to_booktabs(
        summary,
        columns=[c for c, _, _ in cols],
        headers=[h for _, h, _ in cols],
        formats=[f for _, _, f in cols],
        caption=(
            "Cost-weight sensitivity of the lever ranking over a factorial of the four "
            "mutating-technique weights (port p, endpoint e, shuffling s, diversity v "
            "$\\in\\{1,3\\}$) and the decoy technique (deception d $\\in\\{1,2.5,5\\}$), "
            "plus an all-equal control, with lean deception candidate sets included. "
            "`var.\\ win.' marks settings where a winning technique "
            "(port/endpoint/diversity) costs more than a port swap. For each setting: the "
            "cheapest technique set at each ASP target, whether a 1--2-technique set "
            "dominates, and whether the deception stack is the most expensive. Overhead "
            "recomputed analytically from one frontier run."
        ),
        label="tab:cost_weight_sensitivity",
        path=layout.tables / f"{NAME}.tex",
        align="lllllll",
    )

    control = summary[summary["is_control"]].iloc[0]
    winners = summary[summary["varies_winners"]]
    cheapest_cols = ["cheapest@0.50", "cheapest@0.25", "cheapest@0.10"]

    # Generalize "never cheapest" from the 5-stack to ANY deception-containing
    # set (incl. the lean-deception candidates {port,deception}, {port,endpoint,
    # deception}). This is the only test that could falsify "deception never cheapest".
    deception_labels = set(base.loc[base["has_deception"], "set_label"].unique())
    cheapest_seen = set(summary[cheapest_cols].to_numpy().ravel().tolist())
    deception_sets_ever_cheapest = sorted(deception_labels & cheapest_seen)

    # Does "deception never cheapest" survive an expensive diversity?
    div_expensive = summary[summary["diversity_weight"] > 1.0]
    div_exp_cheapest = set(div_expensive[cheapest_cols].to_numpy().ravel().tolist())
    deception_cheapest_when_diversity_expensive = sorted(deception_labels & div_exp_cheapest)

    # At the literal baseline operating weights (not necessarily a grid cell, since the
    # grid sweeps diversity over {1,3} not its baseline 2).
    base_rank = _ranking_for_setting(base, cfg.defender.technique_cost_weights)
    baseline_cheapest = {t: base_rank.get(f"cheapest@{t}", "--") for t in (0.5, 0.25, 0.1)}

    extra = {
        "n_weight_settings": int(len(summary)),
        "n_candidate_sets": int(base["set_label"].nunique()),
        "deception_containing_sets": sorted(deception_labels),
        "lean_dominates_all_settings": bool(summary["lean_dominates"].all()),
        "lean_dominates_count": f"{int(summary['lean_dominates'].sum())}/{len(summary)}",
        "deception_least_cost_effective_all_settings": bool(
            summary["deception_least_cost_effective"].all()
        ),
        "all_equal_control": {
            "cheapest@0.25": str(control["cheapest@0.25"]),
            "lean_dominates": bool(control["lean_dominates"]),
            "deception_least_cost_effective": bool(control["deception_least_cost_effective"]),
            "gap_at_asp_0.25": float(control["gap_at_asp_0.25"]),
        },
        "deception_least_cost_effective_at_dec_weight_1": bool(
            summary.loc[summary["deception_weight"] == 1.0, "deception_least_cost_effective"].all()
        ),
        # Does "lean wins" survive when the winning techniques are expensive?
        "n_winner_expensive_settings": int(len(winners)),
        "lean_dominates_when_winners_expensive": f"{int(winners['lean_dominates'].sum())}/{len(winners)}",
        "cheapest_at_0.25_when_winners_expensive": sorted(
            winners["cheapest@0.25"].unique().tolist()
        ),
        # Is ANY deception-containing set ever the cheapest lever?
        "deception_sets_ever_cheapest": deception_sets_ever_cheapest,
        "any_deception_set_ever_cheapest": bool(deception_sets_ever_cheapest),
        "deception_never_cheapest_all_settings": bool(not deception_sets_ever_cheapest),
        "min_gap_at_asp_0.25": float(summary["gap_at_asp_0.25"].min()),
        # Survival when diversity (a frontier winner) is also expensive.
        "n_diversity_expensive_settings": int(len(div_expensive)),
        "deception_cheapest_when_diversity_expensive": deception_cheapest_when_diversity_expensive,
        "baseline_weights_cheapest": baseline_cheapest,
    }
    return ExperimentOutput(
        name=NAME, summary=summary, figures=figures, tables=[{"csv": csv, "tex": tex}], extra=extra
    )
