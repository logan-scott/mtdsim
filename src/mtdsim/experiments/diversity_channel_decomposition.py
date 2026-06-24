"""Reviewer item M6: separate service_diversity's two channels.

``service_diversity`` does two things at once: it mutates a 4th observable
attribute (feeding the invalidation hazard) *and* it adds to the
identification-confusion denominator. This experiment isolates the two channels
at a fixed mutation frequency, relative to a 3-technique base:

* **base** — {port, endpoint, shuffling}; no diversity at all.
* **mutation-only** — add service_diversity to the mutating set but disable its
  confusion (``diversity_confusion_override=False``).
* **confusion-only** — keep 3 mutating techniques but enable diversity confusion
  (``diversity_confusion_override=True``).
* **both** — the standard "+diversity" configuration.

The decomposition (base ASP minus each variant's ASP) shows how much of the
``+diversity`` ablation drop comes from mutation vs. confusion.
"""

from __future__ import annotations

from .. import tables, viz
from ..config import Config
from .common import Cell, ExperimentOutput, OutputLayout, run_cells

NAME = "diversity_channel_decomposition"
BASE3 = ["port_rotation", "endpoint_mutation", "shuffling"]
WITH_DIV = [*BASE3, "service_diversity"]


def _variants(freq: float) -> list[dict]:
    return [
        {
            "variant": "base",
            "variant_label": "base (3 tech)",
            "variant_order": 0,
            "defender": {"enabled_techniques": BASE3, "mutation_frequency": freq},
            "attacker": {"diversity_confusion_override": False},
        },
        {
            "variant": "mutation_only",
            "variant_label": "+mutation only",
            "variant_order": 1,
            "defender": {"enabled_techniques": WITH_DIV, "mutation_frequency": freq},
            "attacker": {"diversity_confusion_override": False},
        },
        {
            "variant": "confusion_only",
            "variant_label": "+confusion only",
            "variant_order": 2,
            "defender": {"enabled_techniques": BASE3, "mutation_frequency": freq},
            "attacker": {"diversity_confusion_override": True},
        },
        {
            "variant": "both",
            "variant_label": "+both (=+diversity)",
            "variant_order": 3,
            "defender": {"enabled_techniques": WITH_DIV, "mutation_frequency": freq},
            "attacker": {"diversity_confusion_override": True},
        },
    ]


def build_cells(cfg: Config) -> list[Cell]:
    spec = cfg.experiments.get(NAME, {})
    freq = float(spec.get("frequency", cfg.experiments.get("ablation", {}).get("frequency", 0.1)))
    cells: list[Cell] = []
    for v in _variants(freq):
        cells.append(
            Cell(
                {
                    "variant": v["variant"],
                    "variant_label": v["variant_label"],
                    "variant_order": v["variant_order"],
                    "frequency": freq,
                },
                cfg.with_overrides(defender=v["defender"], attacker=v["attacker"]),
            )
        )
    return cells


def run(cfg: Config, layout: OutputLayout, *, parallel: bool = False) -> ExperimentOutput:
    cells = build_cells(cfg)
    summary = run_cells(cells, parallel=parallel, raw_dir=layout.raw / NAME, raw_prefix="variant")
    summary = summary.sort_values("variant_order").reset_index(drop=True)

    # Decompose: ASP reduction from base attributable to each channel.
    base_asp = float(summary.loc[summary["variant"] == "base", "asp"].iloc[0])
    summary["asp_drop_vs_base"] = base_asp - summary["asp"]
    summary.to_csv(layout.summary / f"{NAME}.csv", index=False)

    figures = [viz.diversity_channel_decomposition(summary, layout.figures)]

    cols = [
        ("variant_label", "Variant", None),
        ("asp", "ASP", ".3f"),
        ("asp_drop_vs_base", "$\\Delta$ASP vs.\\ base", ".3f"),
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
            "Decomposition of service\\_diversity at a fixed mutation frequency. "
            "$\\Delta$ASP vs.\\ base attributes the ``+diversity'' effect to its "
            "mutation channel, its identification-confusion channel, and both "
            "combined."
        ),
        label="tab:diversity_decomposition",
        path=layout.tables / f"{NAME}.tex",
        align="lrrrr",
    )
    return ExperimentOutput(
        name=NAME,
        summary=summary,
        figures=figures,
        tables=[{"csv": csv, "tex": tex}],
        extra={
            "base_asp": base_asp,
            "mutation_only_drop": float(
                summary.loc[summary["variant"] == "mutation_only", "asp_drop_vs_base"].iloc[0]
            ),
            "confusion_only_drop": float(
                summary.loc[summary["variant"] == "confusion_only", "asp_drop_vs_base"].iloc[0]
            ),
            "both_drop": float(
                summary.loc[summary["variant"] == "both", "asp_drop_vs_base"].iloc[0]
            ),
        },
    )
