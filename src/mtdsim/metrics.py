"""Metrics — aggregation of trial outcomes into the paper's constructs.

Maps raw :class:`~mtdsim.engine.TrialResult` records to the quantities defined
in the build spec / paper:

- **ASP** (attack success probability) = compromises / trials, with a Wilson
  score confidence interval (well-behaved near 0 and 1).
- **TTC** (time to compromise) summarised over successful trials only: mean
  (normal-approx CI) and median (seeded percentile bootstrap CI).
- **Attacker uncertainty**: mean forced re-recons and mean time-averaged
  stale-knowledge fraction.
- **Operational overhead**: mean cumulative overhead per trial and per tick.

Also provides :func:`pareto_knee`, a dependency-free "kneedle"-style detector of
the knee of the overhead-vs-ASP frontier.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .engine import TrialResult

Z_95 = 1.959963984540054  # standard-normal 97.5th percentile
TTC_SMALL_N = 30  # below this many successful trials, TTC statistics are flagged as low-n


def to_dataframe(results: Sequence[TrialResult]) -> pd.DataFrame:
    """Flatten per-trial results into a tidy DataFrame (one row per trial)."""
    return pd.DataFrame([r.as_dict() for r in results])


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    The bounds are clamped to ``[0, 1]``; in the degenerate cases of 0 or all
    successes the corresponding bound is pinned to exactly ``0.0``/``1.0`` so the
    released CSVs never show spurious values like ``2.17e-19``.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - margin)
    hi = 1.0 if successes == n else min(1.0, center + margin)
    return (lo, hi)


def mean_ci(values: np.ndarray, z: float = Z_95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using a normal-approximation interval.

    For the publication's ~1000-trial subsets the normal approximation is
    indistinguishable from a t-interval; this keeps the dependency surface small.
    """
    n = values.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(np.mean(values))
    if n == 1:
        return (mean, mean, mean)
    se = float(np.std(values, ddof=1)) / math.sqrt(n)
    return (mean, mean - z * se, mean + z * se)


def bootstrap_median_ci(
    values: np.ndarray, *, seed: int, n_boot: int = 2000, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (median, lo, hi) via a seeded percentile bootstrap."""
    n = values.size
    if n == 0:
        return (math.nan, math.nan, math.nan)
    median = float(np.median(values))
    if n == 1:
        return (median, median, median)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(values[idx], axis=1)
    lo = float(np.quantile(boot_medians, alpha / 2))
    hi = float(np.quantile(boot_medians, 1 - alpha / 2))
    return (median, lo, hi)


def aggregate(
    results: Sequence[TrialResult],
    *,
    labels: dict[str, Any] | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate one configuration's trials into a single summary row."""
    df = to_dataframe(results)
    n = len(df)
    compromises = int(df["compromised"].sum())
    asp = compromises / n if n else math.nan
    asp_lo, asp_hi = wilson_interval(compromises, n)

    ttc_vals = df.loc[df["compromised"], "ttc"].to_numpy(dtype=float)
    ttc_mean, ttc_mean_lo, ttc_mean_hi = mean_ci(ttc_vals)
    ttc_median, ttc_med_lo, ttc_med_hi = bootstrap_median_ci(ttc_vals, seed=bootstrap_seed)

    overhead = df["overhead"].to_numpy(dtype=float)
    overhead_mean, overhead_lo, overhead_hi = mean_ci(overhead)
    overhead_per_tick = (df["overhead"] / df["trial_ticks"]).to_numpy(dtype=float)
    overhead_per_asset = (df["overhead"] / df["total_assets"]).to_numpy(dtype=float)

    row: dict[str, Any] = dict(labels or {})
    row.update(
        n_trials=n,
        compromises=compromises,
        asp=asp,
        asp_ci_lo=asp_lo,
        asp_ci_hi=asp_hi,
        n_success=int(ttc_vals.size),  # successful trials behind the TTC statistics
        ttc_small_n=bool(ttc_vals.size < TTC_SMALL_N),
        ttc_mean=ttc_mean,
        ttc_mean_ci_lo=ttc_mean_lo,
        ttc_mean_ci_hi=ttc_mean_hi,
        ttc_median=ttc_median,
        ttc_median_ci_lo=ttc_med_lo,
        ttc_median_ci_hi=ttc_med_hi,
        overhead_mean=overhead_mean,
        overhead_ci_lo=overhead_lo,
        overhead_ci_hi=overhead_hi,
        # One overhead unit == one weight-1 attribute reconfiguration; also report
        # the cost normalized per tick and per asset for interpretability (M3).
        overhead_per_tick_mean=float(np.mean(overhead_per_tick)),
        overhead_per_asset_mean=float(np.mean(overhead_per_asset)),
        forced_recons_mean=float(df["forced_recons"].mean()),
        wrong_target_failures_mean=float(df["wrong_target_failures"].mean()),
        decoys_encountered_mean=float(df["decoys_encountered"].mean()),
        stale_fraction_mean=float(df["stale_fraction"].mean()),
        rounds_completed_mean=float(df["rounds_completed"].mean()),
        trial_ticks_mean=float(df["trial_ticks"].mean()),
        mutation_events_mean=float(df["mutation_events"].mean()),
    )
    return row


def pareto_knee(overhead: Sequence[float], asp: Sequence[float]) -> dict[str, Any]:
    """Locate the knee of the overhead-vs-ASP frontier (kneedle-style).

    Returns a dict with the original-array ``index`` of the knee plus its
    ``overhead`` and ``asp``. Points are sorted by overhead internally; the knee
    is the frontier point of maximum perpendicular distance from the chord
    joining the two extreme points. Falls back gracefully for <3 points.
    """
    x = np.asarray(overhead, dtype=float)
    y = np.asarray(asp, dtype=float)
    if x.size < 3:
        i = int(np.argmin(y)) if y.size else 0
        return {
            "index": i,
            "overhead": float(x[i]) if x.size else math.nan,
            "asp": float(y[i]) if y.size else math.nan,
        }

    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xr = xs.max() - xs.min() or 1.0
    yr = ys.max() - ys.min() or 1.0
    xn = (xs - xs.min()) / xr
    yn = (ys - ys.min()) / yr

    p1 = np.array([xn[0], yn[0]])
    p2 = np.array([xn[-1], yn[-1]])
    span = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) or 1.0
    dist = (
        np.abs((p2[1] - p1[1]) * xn - (p2[0] - p1[0]) * yn + p2[0] * p1[1] - p2[1] * p1[0]) / span
    )
    knee_sorted = int(np.argmax(dist))
    orig = int(order[knee_sorted])
    return {"index": orig, "overhead": float(x[orig]), "asp": float(y[orig])}


def pareto_frontier(overhead: Sequence[float], asp: Sequence[float]) -> np.ndarray:
    """Boolean mask of non-dominated points in (overhead, ASP) space.

    Both objectives are minimized (the defender wants *low* ASP at *low*
    overhead). Point ``i`` is dominated if some other point is ``<=`` on both
    axes and strictly ``<`` on at least one. Ties are both kept. O(n^2); the grid
    here is small. Used to highlight the global frontier across technique sets
    (experiment 6: frequency × technique count).
    """
    x = np.asarray(overhead, dtype=float)
    y = np.asarray(asp, dtype=float)
    n = x.size
    idx = np.arange(n)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        leq = (x <= x[i]) & (y <= y[i])
        strict = (x < x[i]) | (y < y[i])
        if np.any(leq & strict & (idx != i)):
            dominated[i] = True
    return ~dominated


def interp_monotone(xq: Sequence[float], x: Sequence[float], y: Sequence[float]) -> np.ndarray:
    """Linearly interpolate ``y(xq)`` after sorting samples by ``x``.

    Returns ``nan`` outside the sampled ``x`` range (no extrapolation), so that
    matched-overhead / matched-ASP comparisons only report where curves overlap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xq = np.asarray(xq, dtype=float)
    out = np.interp(xq, xs, ys, left=np.nan, right=np.nan)
    return out
