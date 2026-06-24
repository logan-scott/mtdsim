"""End-to-end experiment drivers and the run_all orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtdsim.config import load_config
from mtdsim.experiments import (
    ablation,
    adaptive,
    baseline,
    diversity_channel_decomposition,
    frequency_by_technique_count,
    frequency_sweep,
    parallelism_sweep,
    sensitivity,
)
from mtdsim.experiments.common import OutputLayout
from mtdsim.experiments.run_all import EXPERIMENTS, run_all


@pytest.fixture
def paper_small():
    return load_config("configs/paper.yaml").with_overrides(engine={"n_trials": 40})


def test_all_experiments_build_nonempty_cells(paper_small):
    for builder in (
        baseline.build_cells,
        frequency_sweep.build_cells,
        ablation.build_cells,
        adaptive.build_cells,
        sensitivity.build_speed_cells,
        sensitivity.build_size_cells,
        frequency_by_technique_count.build_cells,
        parallelism_sweep.build_cells,
        diversity_channel_decomposition.build_cells,
    ):
        assert len(builder(paper_small)) > 0


def _layout(tmp_path: Path) -> OutputLayout:
    return OutputLayout.create(tmp_path / "results", tmp_path / "figures", tmp_path / "tables")


def test_run_all_subset_writes_artifacts(paper_small, tmp_path):
    layout = _layout(tmp_path)
    manifest = run_all(
        paper_small,
        layout,
        only=["baseline", "frequency_sweep"],
        config_path="configs/paper.yaml",
        verbose=False,
    )
    # Headline figures + tables exist as both formats.
    for stem in ("asp_vs_frequency", "pareto_overhead_vs_asp"):
        assert (layout.figures / f"{stem}.pdf").exists()
        assert (layout.figures / f"{stem}.png").exists()
    assert (layout.tables / "frequency_sweep.tex").exists()
    assert (layout.tables / "baseline.csv").exists()
    # Manifest records hashes + environment + git block.
    assert (layout.results / "manifest.json").exists()
    assert manifest["output_files"]
    assert "git" in manifest and "packages" in manifest["environment"]
    # Manifest on disk parses.
    json.loads((layout.results / "manifest.json").read_text())


def test_frequency_sweep_static_has_highest_asp(paper_small):
    """Acceptance criterion: static defense yields the highest ASP; a knee exists."""
    layout_cfg = paper_small.with_overrides(engine={"n_trials": 200})
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        layout = OutputLayout.create(Path(d) / "r", Path(d) / "f", Path(d) / "t")
        out = frequency_sweep.run(layout_cfg, layout, parallel=False)
    summary = out.summary.sort_values("frequency").reset_index(drop=True)
    asp = summary["asp"].to_numpy()
    assert asp[0] == asp.max()  # f=0 is the maximum
    assert summary.loc[summary["frequency"] == 0.0, "asp"].iloc[0] > 0.9
    # A knee was located and maps to a swept frequency.
    knee = out.extra["pareto_knee"]
    assert knee["frequency"] in set(summary["frequency"])


def test_registry_matches_modules():
    assert set(EXPERIMENTS) == {
        "baseline",
        "frequency_sweep",
        "frequency_by_technique_count",
        "cost_weight_sensitivity",
        "decoy_ratio_sweep",
        "ablation",
        "diversity_channel_decomposition",
        "adaptive_vs_nonadaptive",
        "parallelism_sweep",
        "sensitivity",
    }


@pytest.mark.slow
def test_run_all_full(tmp_path):
    cfg = load_config("configs/paper.yaml").with_overrides(engine={"n_trials": 60})
    layout = _layout(tmp_path)
    run_all(cfg, layout, config_path="configs/paper.yaml", verbose=False)
    expected = [
        "asp_vs_frequency",
        "ttc_vs_frequency",
        "overhead_vs_frequency",
        "pareto_overhead_vs_asp",
        "technique_ablation",
        "adaptive_vs_nonadaptive",
        "attacker_uncertainty_vs_frequency",
        "sensitivity_asp_heatmaps",
        "frontier_by_technique_count",
        "parallelism_sweep",
        "diversity_channel_decomposition",
        "cost_weight_sensitivity",
        "decoy_ratio_sweep",
    ]
    for stem in expected:
        assert (layout.figures / f"{stem}.pdf").exists(), stem
