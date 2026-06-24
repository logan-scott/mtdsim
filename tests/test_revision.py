"""Reviewer-2 revision mechanics: C2a learning, C2b parallelism, M6 channels,
and the new metric columns."""

from __future__ import annotations

import numpy as np
import pytest

from mtdsim.config import load_config
from mtdsim.engine import run_config
from mtdsim.experiments import (
    diversity_channel_decomposition,
    frequency_by_technique_count,
    parallelism_sweep,
)
from mtdsim.experiments.common import OutputLayout
from mtdsim.metrics import aggregate


@pytest.fixture
def paper_small():
    return load_config("configs/paper.yaml").with_overrides(engine={"n_trials": 60})


# -- C2a: learning-signal default + no perverse coupling -------------------
def test_learning_signal_default_is_decoy_encounters(paper_small):
    assert paper_small.attacker.learning_signal == "decoy_encounters"


def test_no_decoy_run_records_zero_decoy_encounters(paper_small):
    # Default technique set has no deception => no decoys => no decoy encounters,
    # so adaptive learning is inert regardless of MTD pressure.
    cfg = paper_small.with_overrides(defender={"mutation_frequency": 0.2}, engine={"n_trials": 80})
    results = run_config(cfg)
    assert all(r.decoys_encountered == 0 for r in results)


# -- New aggregate columns -------------------------------------------------
def test_aggregate_has_revision_columns(paper_small):
    cfg = paper_small.with_overrides(defender={"mutation_frequency": 0.1})
    row = aggregate(run_config(cfg))
    for col in (
        "n_success",
        "ttc_small_n",
        "overhead_per_asset_mean",
        "overhead_per_tick_mean",
        "decoys_encountered_mean",
    ):
        assert col in row


def test_small_n_ttc_is_flagged(paper_small):
    # Very high f => almost no compromises => few successful trials => flagged.
    cfg = paper_small.with_overrides(defender={"mutation_frequency": 0.5}, engine={"n_trials": 60})
    row = aggregate(run_config(cfg))
    assert row["n_success"] < 30
    assert row["ttc_small_n"] is True


# -- C2b: parallelism is a real capability ---------------------------------
def test_more_parallel_probing_raises_asp(paper_small):
    """At a fixed moderate frequency, probing more assets per recon should not
    hurt and generally helps the attacker (a genuine agent capability)."""
    cfg1 = paper_small.with_overrides(
        defender={"mutation_frequency": 0.04},
        attacker={"parallelism": 1},
        engine={"n_trials": 200},
    )
    cfg8 = paper_small.with_overrides(
        defender={"mutation_frequency": 0.04},
        attacker={"parallelism": 8},
        engine={"n_trials": 200},
    )
    asp1 = aggregate(run_config(cfg1))["asp"]
    asp8 = aggregate(run_config(cfg8))["asp"]
    assert asp8 >= asp1


def test_parallelism_sweep_builds(paper_small):
    cells = parallelism_sweep.build_cells(paper_small)
    pars = {c.labels["parallelism"] for c in cells}
    assert pars == {1, 2, 4, 8}


# -- M6: service_diversity channel decomposition ---------------------------
def test_diversity_decomposition_isolates_channels(paper_small, tmp_path):
    layout = OutputLayout.create(tmp_path / "r", tmp_path / "f", tmp_path / "t")
    cfg = paper_small.with_overrides(engine={"n_trials": 300})
    out = diversity_channel_decomposition.run(cfg, layout, parallel=False)
    s = out.summary.set_index("variant")
    # Both channels combined should suppress ASP at least as much as either alone.
    assert s.loc["both", "asp"] <= s.loc["mutation_only", "asp"] + 0.05
    assert s.loc["both", "asp"] <= s.loc["confusion_only", "asp"] + 0.05
    # Base (no diversity) has the highest ASP of the four.
    assert s.loc["base", "asp"] >= s.loc["both", "asp"]
    # Each channel contributes a non-negative drop from base (within noise).
    assert out.extra["mutation_only_drop"] >= -0.05
    assert out.extra["confusion_only_drop"] >= -0.05


def test_diversity_confusion_override_decouples_from_mutating_set(paper_small):
    """confusion-only (override=True, diversity NOT in mutating set) must lower
    ASP vs. an identical run with confusion disabled."""
    base = paper_small.with_overrides(
        defender={
            "enabled_techniques": ["port_rotation", "endpoint_mutation", "shuffling"],
            "mutation_frequency": 0.06,
        },
        attacker={"diversity_confusion_override": False},
        engine={"n_trials": 300},
    )
    confused = base.with_overrides(attacker={"diversity_confusion_override": True})
    asp_base = aggregate(run_config(base))["asp"]
    asp_conf = aggregate(run_config(confused))["asp"]
    assert asp_conf < asp_base


# -- C1: frontier experiment ----------------------------------------------
def test_c1_frontier_runs_and_reports_verdict(paper_small, tmp_path):
    layout = OutputLayout.create(tmp_path / "r", tmp_path / "f", tmp_path / "t")
    cfg = paper_small.with_overrides(engine={"n_trials": 60})
    out = frequency_by_technique_count.run(cfg, layout, parallel=False)
    assert "on_frontier" in out.summary.columns
    for key in (
        "stacking_cheaper_than_frequency",
        "stacked_vs_ref_overhead_gap_at_matched_asp",
        "cheapest_set_to_reach_asp_0.25",
        "global_frontier_counts_excl_static",
    ):
        assert key in out.extra


# -- Ma: cost-weight sensitivity (round 4: + diversity weight, + lean-deception) -----
def test_cost_weight_settings_vary_all_mutating_weights(paper_small):
    from mtdsim.experiments import cost_weight_sensitivity as cws

    settings = cws.weight_settings(paper_small)
    assert sum(s["is_control"] for s in settings) == 1  # exactly one all-equal control
    # all-equal control + 5-way factorial (port x endpoint x shuffling x diversity x deception).
    assert len(settings) == 1 + 2 * 2 * 2 * 2 * 3
    control = next(s for s in settings if s["is_control"])
    assert set(control["weights"].values()) == {1.0}
    # Round 4 (item b): service_diversity (a frontier winner) is now varied too.
    assert any(s["weights"]["service_diversity"] == 3.0 for s in settings)
    assert any(s["weights"]["port_rotation"] == 3.0 for s in settings)
    assert any(s["weights"]["endpoint_mutation"] == 3.0 for s in settings)


def test_lean_deception_sets_in_frontier(paper_small):
    """Round 4 (item a): the non-cumulative lean-deception sets are frontier candidates."""
    from mtdsim.experiments import frequency_by_technique_count as fbt

    labels = {s["label"] for s in fbt._frontier_sets(paper_small)}
    assert {"port+decep", "port+endp+decep"} <= labels
    # ...but the original cumulative sets are unchanged (additive).
    cumulative = {s["label"] for s in fbt._technique_sets(paper_small)}
    assert "port+decep" not in cumulative and "+deception" in cumulative


def test_analytic_overhead_matches_actual(paper_small):
    """The core correctness invariant (retained): overhead recomputed from
    per-technique change counts under the baseline weights equals the defender's
    actual overhead -- including the added lean-deception sets."""
    from mtdsim.experiments import cost_weight_sensitivity as cws

    cfg = paper_small.with_overrides(engine={"n_trials": 80})
    base = cws._run_frontier_once(cfg, parallel=False)
    assert {"port+decep", "port+endp+decep"} <= set(base["set_label"].unique())
    recomputed = cws._overhead_under(base, cfg.defender.technique_cost_weights)
    actual = base["actual_overhead_mean"].to_numpy()
    assert float(np.max(np.abs(recomputed - actual))) < 1e-6


def test_no_deception_set_cheapest_at_moderate_asp(paper_small, tmp_path):
    """Round-4 verdict: no deception-containing set is the cheapest lever at the
    moderate ASP targets (0.50, 0.25) in any weight setting, and never at the
    model's baseline weights. (A lean-deception set CAN win at the noisy ASP-0.10
    target in a few expensive-endpoint cells; that boundary is reported, not asserted
    here, to avoid flakiness at low trial counts.)"""
    from mtdsim.experiments import cost_weight_sensitivity as cws
    from mtdsim.experiments.common import OutputLayout

    layout = OutputLayout.create(tmp_path / "r", tmp_path / "f", tmp_path / "t")
    out = cws.run(paper_small.with_overrides(engine={"n_trials": 200}), layout, parallel=False)
    # No deception set is cheapest at ASP 0.25 in any setting.
    assert out.summary["gap_at_asp_0.25"].min() > 0
    assert not (out.summary["cheapest@0.50"].isin(out.extra["deception_containing_sets"]).any())
    assert not (out.summary["cheapest@0.25"].isin(out.extra["deception_containing_sets"]).any())
    # At baseline weights, the cheapest lever is never a deception set.
    assert (
        out.extra["baseline_weights_cheapest"][0.25] not in out.extra["deception_containing_sets"]
    )
    assert out.extra["baseline_weights_cheapest"][0.5] not in out.extra["deception_containing_sets"]
    # Sanity: the grid includes the lean-deception candidates.
    assert {"port+decep", "port+endp+decep"} <= set(out.extra["deception_containing_sets"])


# -- Round 3: decoy-ratio sweep -------------------------------------------
def test_decoy_ratio_sweep_gap_grows_and_stays_positive(paper_small, tmp_path):
    from mtdsim.experiments import decoy_ratio_sweep as drs
    from mtdsim.experiments.common import OutputLayout

    layout = OutputLayout.create(tmp_path / "r", tmp_path / "f", tmp_path / "t")
    cfg = paper_small.with_overrides(engine={"n_trials": 120})
    out = drs.run(cfg, layout, parallel=False)
    ratios = out.summary["decoy_ratio"].tolist()
    assert ratios == sorted(ratios) and ratios[0] == 0.0  # includes the 0.0 edge case
    # Deception always costs more than the cheapest set, and the gap grows with decoys.
    assert out.extra["gap_positive_at_all_ratios"] is True
    assert out.extra["gap_monotone_increasing"] is True
    # decoy_ratio=0 yields zero decoys (handled gracefully).
    assert int(out.summary.loc[out.summary["decoy_ratio"] == 0.0, "n_decoys"].iloc[0]) == 0
