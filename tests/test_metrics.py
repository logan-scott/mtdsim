"""Metrics: confidence intervals and Pareto-knee detection."""

from __future__ import annotations

import numpy as np

from mtdsim.metrics import (
    bootstrap_median_ci,
    interp_monotone,
    mean_ci,
    pareto_frontier,
    pareto_knee,
    wilson_interval,
)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # Reviewer minor: degenerate endpoints are pinned to exactly 0.0 / 1.0 (no
    # spurious values like 2.17e-19 in the released CSVs).
    lo0, hi0 = wilson_interval(0, 100)
    lo1, hi1 = wilson_interval(100, 100)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.1
    assert hi1 == 1.0 and 0.9 < lo1 < 1.0


def test_wilson_empty():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_mean_ci_brackets_mean():
    vals = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    mean, lo, hi = mean_ci(vals)
    assert lo < mean < hi
    assert abs(mean - 14.0) < 1e-9


def test_bootstrap_median_ci_is_seeded():
    vals = np.arange(1, 101, dtype=float)
    a = bootstrap_median_ci(vals, seed=7)
    b = bootstrap_median_ci(vals, seed=7)
    assert a == b
    assert a[1] <= a[0] <= a[2]


def test_pareto_knee_detects_elbow():
    # Decreasing frontier whose sharpest bend is at overhead=20 (the big ASP drop
    # happens between 10 and 20, leaving overhead=20 farthest from the endpoint chord).
    overhead = [0.0, 10.0, 20.0, 30.0, 40.0]
    asp = [1.0, 0.9, 0.3, 0.25, 0.2]
    knee = pareto_knee(overhead, asp)
    assert knee["index"] == 2
    assert knee["overhead"] == 20.0


def test_pareto_knee_handles_few_points():
    knee = pareto_knee([1.0, 2.0], [0.9, 0.2])
    assert knee["index"] in (0, 1)


def test_pareto_frontier_identifies_nondominated():
    # Points: A(0,1) B(1,0.9) C(2,0.5) D(2,0.6) E(3,0.5)
    # D is dominated by C (same overhead, higher ASP). E dominated by C (more
    # overhead, equal ASP). A,B,C are non-dominated.
    overhead = [0.0, 1.0, 2.0, 2.0, 3.0]
    asp = [1.0, 0.9, 0.5, 0.6, 0.5]
    mask = pareto_frontier(overhead, asp)
    assert mask.tolist() == [True, True, True, False, False]


def test_pareto_frontier_keeps_ties():
    # Two identical points are mutually non-dominated; both kept.
    mask = pareto_frontier([1.0, 1.0], [0.5, 0.5])
    assert mask.tolist() == [True, True]


def test_interp_monotone_no_extrapolation():
    y = interp_monotone([0.5, 5.0], x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0])
    assert np.isnan(y[0])  # below range
    assert np.isnan(y[1])  # above range
    mid = interp_monotone([2.5], x=[3.0, 1.0, 2.0], y=[30.0, 10.0, 20.0])  # unsorted input
    assert abs(mid[0] - 25.0) < 1e-9
