"""Grayscale, print-safe figures for the paper and dissertation.

Every figure is rendered without color dependence (distinct line styles and
markers, grayscale fills), at 300 dpi, and saved as both ``.pdf`` (vector, for
LaTeX) and ``.png`` (raster, for quick viewing). Each plotting function returns
the saved file paths and a documented caption so the run driver can record them.

The functions consume the tidy summary DataFrames produced by
:mod:`mtdsim.experiments`. They never run the simulation themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Grayscale-safe cycles for distinguishing multiple series in print.
_GRAYS = ["0.0", "0.40", "0.60", "0.25", "0.75"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_style() -> None:
    """Apply a compact, grayscale, IEEE-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (5.0, 3.4),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "image.cmap": "Greys",
        }
    )


def _save(fig: plt.Figure, outdir: Path, stem: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for ext in ("pdf", "png"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(str(p))
    plt.close(fig)
    return paths


def _series_style(i: int) -> dict[str, Any]:
    return {
        "color": _GRAYS[i % len(_GRAYS)],
        "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
        "marker": _MARKERS[i % len(_MARKERS)],
    }


def _line_with_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    lo: np.ndarray | None,
    hi: np.ndarray | None,
    *,
    style_idx: int = 0,
    label: str | None = None,
) -> None:
    st = _series_style(style_idx)
    ax.plot(x, y, label=label, **st)
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, linewidth=0)


# --------------------------------------------------------------------------
# Frequency-sweep figures
# --------------------------------------------------------------------------
def asp_vs_frequency(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.sort_values("frequency")
    fig, ax = plt.subplots()
    _line_with_band(
        ax,
        d["frequency"].to_numpy(),
        d["asp"].to_numpy(),
        d["asp_ci_lo"].to_numpy(),
        d["asp_ci_hi"].to_numpy(),
    )
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ASP vs. mutation frequency")
    paths = _save(fig, outdir, "asp_vs_frequency")
    return {
        "paths": paths,
        "caption": (
            "Attack success probability (ASP) as a function of MTD mutation "
            "frequency $f$. Shaded band is the 95\\% Wilson interval. ASP is "
            "highest under static defense ($f=0$) and falls monotonically as the "
            "defender mutates more frequently."
        ),
    }


def ttc_vs_frequency(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.sort_values("frequency")
    fig, ax = plt.subplots()
    _line_with_band(
        ax,
        d["frequency"].to_numpy(),
        d["ttc_median"].to_numpy(),
        d["ttc_median_ci_lo"].to_numpy(),
        d["ttc_median_ci_hi"].to_numpy(),
    )
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Median time to compromise (ticks)")
    ax.set_title("Median TTC vs. mutation frequency")
    paths = _save(fig, outdir, "ttc_vs_frequency")
    return {
        "paths": paths,
        "caption": (
            "Median time to compromise (TTC) over successful trials vs. mutation "
            "frequency $f$ (95\\% bootstrap interval). TTC rises with $f$; at high "
            "$f$ few trials succeed, so the median is defined over a shrinking "
            "successful subset."
        ),
    }


def overhead_vs_frequency(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.sort_values("frequency")
    fig, ax = plt.subplots()
    _line_with_band(
        ax,
        d["frequency"].to_numpy(),
        d["overhead_mean"].to_numpy(),
        d["overhead_ci_lo"].to_numpy(),
        d["overhead_ci_hi"].to_numpy(),
    )
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Mean operational overhead (cost units)")
    ax.set_title("Operational overhead vs. mutation frequency")
    paths = _save(fig, outdir, "overhead_vs_frequency")
    return {
        "paths": paths,
        "caption": (
            "Mean cumulative operational overhead per trial vs. mutation "
            "frequency $f$ (95\\% normal interval). Overhead grows with $f$ as "
            "reconfigurations accumulate over the engagement."
        ),
    }


def attacker_uncertainty_vs_frequency(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.sort_values("frequency")
    fig, ax = plt.subplots()
    x = d["frequency"].to_numpy()
    ax.plot(x, d["forced_recons_mean"].to_numpy(), label="Forced re-recons", **_series_style(0))
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Mean forced re-recons per trial")
    ax2 = ax.twinx()
    ax2.plot(
        x,
        d["stale_fraction_mean"].to_numpy(),
        label="Stale-knowledge fraction",
        **_series_style(1),
    )
    ax2.set_ylabel("Time-avg. stale-knowledge fraction")
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_title("Attacker uncertainty vs. mutation frequency")
    paths = _save(fig, outdir, "attacker_uncertainty_vs_frequency")
    return {
        "paths": paths,
        "caption": (
            "Attacker uncertainty vs. mutation frequency $f$: mean forced "
            "re-reconnaissance events (left axis) and time-averaged "
            "stale-knowledge fraction (right axis). Both rise with $f$, "
            "quantifying the planning disruption MTD imposes."
        ),
    }


def pareto_overhead_vs_asp(df: pd.DataFrame, knee: dict[str, Any], outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.sort_values("overhead_mean")
    fig, ax = plt.subplots()
    x = d["overhead_mean"].to_numpy()
    y = d["asp"].to_numpy()
    ax.plot(x, y, color="0.0", linestyle="-", marker="o", label="Frontier")
    # Thin labels by minimum spacing so the dense low-overhead cluster stays legible.
    xr = (x.max() - x.min()) or 1.0
    yr = (y.max() - y.min()) or 1.0
    last_x = last_y = -1e18
    for _, r in d.iterrows():
        ox, oy = r["overhead_mean"], r["asp"]
        if abs(ox - last_x) / xr < 0.07 and abs(oy - last_y) / yr < 0.07:
            continue
        ax.annotate(
            f"f={r['frequency']:g}",
            (ox, oy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=6.5,
            color="0.4",
        )
        last_x, last_y = ox, oy
    ax.scatter(
        [knee["overhead"]],
        [knee["asp"]],
        s=90,
        facecolors="none",
        edgecolors="0.0",
        linewidths=1.6,
        zorder=5,
    )
    ax.annotate(
        f"knee (f={knee['frequency']:g})",
        (knee["overhead"], knee["asp"]),
        textcoords="offset points",
        xytext=(10, 12),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.0", lw=1.0),
    )
    ax.set_xlabel("Mean operational overhead (cost units)")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Overhead-vs-ASP Pareto frontier")
    paths = _save(fig, outdir, "pareto_overhead_vs_asp")
    return {
        "paths": paths,
        "caption": (
            "Pareto frontier of operational overhead vs. attack success "
            "probability across the mutation-frequency sweep; each point is "
            "labeled with its $f$. The circled point marks the knee "
            f"($f={knee['frequency']:g}$): the configuration beyond which further "
            "overhead yields diminishing ASP reduction."
        ),
    }


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------
def technique_ablation(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    d = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(d))
    grays = [str(0.15 + 0.6 * i / max(1, len(d) - 1)) for i in range(len(d))]
    # Clamp to >=0: the Wilson interval can land a bound on the far side of the
    # point estimate at extreme proportions (asp near 0/1), which matplotlib's
    # asymmetric yerr rejects as "negative".
    err_lo = np.clip((d["asp"] - d["asp_ci_lo"]).to_numpy(), 0.0, None)
    err_hi = np.clip((d["asp_ci_hi"] - d["asp"]).to_numpy(), 0.0, None)
    ax.bar(
        x,
        d["asp"].to_numpy(),
        color=grays,
        edgecolor="0.0",
        linewidth=0.8,
        yerr=[err_lo, err_hi],
        capsize=3,
        ecolor="0.0",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(d["step"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Technique ablation (cumulative)")
    paths = _save(fig, outdir, "technique_ablation")
    return {
        "paths": paths,
        "caption": (
            "Attack success probability as MTD techniques are enabled "
            "cumulatively at a fixed mutation frequency (error bars: 95\\% Wilson "
            "interval). Each added mechanism further degrades attacker success."
        ),
    }


# --------------------------------------------------------------------------
# Adaptive vs non-adaptive
# --------------------------------------------------------------------------
def adaptive_vs_nonadaptive(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    apply_style()
    fig, ax = plt.subplots()
    for i, (mode, g) in enumerate(df.groupby("attacker_mode")):
        g = g.sort_values("frequency")
        _line_with_band(
            ax,
            g["frequency"].to_numpy(),
            g["asp"].to_numpy(),
            g["asp_ci_lo"].to_numpy(),
            g["asp_ci_hi"].to_numpy(),
            style_idx=i,
            label=str(mode),
        )
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Attacker")
    ax.set_title("Adaptive vs. non-adaptive attacker")
    paths = _save(fig, outdir, "adaptive_vs_nonadaptive")
    return {
        "paths": paths,
        "caption": (
            "ASP vs. mutation frequency for an adaptive attacker (re-plans from "
            "recon; learns to disambiguate decoys only from genuine decoy "
            "encounters) and a non-adaptive attacker (fixed replanning penalty). "
            "In the default no-decoy configuration the adaptive advantage reduces "
            "to avoiding the replanning penalty; MTD degrades both."
        ),
    }


# --------------------------------------------------------------------------
# Sensitivity heatmaps
# --------------------------------------------------------------------------
def _heatmap(
    ax: plt.Axes, pivot: pd.DataFrame, *, title: str, xlabel: str, ylabel: str, fmt: str
) -> Any:
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(
        data,
        cmap="Greys",
        aspect="auto",
        origin="lower",
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate cells with contrasting text for readability.
    rng = (np.nanmax(data) - np.nanmin(data)) or 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            shade = (v - np.nanmin(data)) / rng
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if shade > 0.55 else "black",
            )
    return im


def sensitivity_heatmaps(
    speed_df: pd.DataFrame, size_df: pd.DataFrame, outdir: Path
) -> dict[str, Any]:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    p1 = speed_df.pivot(index="speed_multiplier", columns="frequency", values="asp")
    _heatmap(
        axes[0],
        p1,
        title="ASP vs. attacker speed",
        xlabel="Mutation frequency $f$",
        ylabel="Attacker speed multiplier",
        fmt=".2f",
    )
    p2 = size_df.pivot(index="n_assets", columns="frequency", values="asp")
    im = _heatmap(
        axes[1],
        p2,
        title="ASP vs. system size",
        xlabel="Mutation frequency $f$",
        ylabel="Number of assets $N$",
        fmt=".2f",
    )
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="ASP")
    paths = _save(fig, outdir, "sensitivity_asp_heatmaps")
    return {
        "paths": paths,
        "caption": (
            "Sensitivity of ASP. Left: attacker speed multiplier (>1 = slower "
            "stages) vs. mutation frequency. Right: system size $N$ vs. mutation "
            "frequency. Darker cells indicate higher ASP. MTD's effect strengthens "
            "as the attacker slows and persists across system sizes."
        ),
    }


# --------------------------------------------------------------------------
# C1: frequency-vs-ASP frontier by technique count
# --------------------------------------------------------------------------
def frontier_by_technique_count(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    """Overlay one overhead-vs-ASP curve per cumulative technique set and
    highlight the global Pareto frontier (lower envelope across all configs)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    order = df[["n_techniques", "set_label"]].drop_duplicates().sort_values("n_techniques")
    for i, (_, row) in enumerate(order.iterrows()):
        g = df[df["set_label"] == row["set_label"]].sort_values("overhead_mean")
        ax.plot(
            g["overhead_mean"].to_numpy(),
            g["asp"].to_numpy(),
            label=str(row["set_label"]),
            **_series_style(i),
        )
    front = df[df["on_frontier"]]
    ax.scatter(
        front["overhead_mean"].to_numpy(),
        front["asp"].to_numpy(),
        s=80,
        facecolors="none",
        edgecolors="0.0",
        linewidths=1.4,
        zorder=5,
        label="Global Pareto frontier",
    )
    ax.set_xlabel("Mean operational overhead (cost units)")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Overhead vs. ASP by technique count")
    paths = _save(fig, outdir, "frontier_by_technique_count")
    return {
        "paths": paths,
        "caption": (
            "Operational overhead vs. attack success probability for the full "
            "mutation-frequency sweep run separately at each cumulative technique "
            "count (1--5 techniques). Open circles mark the global Pareto frontier "
            "(non-dominated configurations). If stacking techniques were cheaper "
            "than raising $f$, stacked curves would lie below the 4-technique "
            "curve; the frontier shows which configurations actually dominate."
        ),
    }


# --------------------------------------------------------------------------
# C2b: parallelism sweep
# --------------------------------------------------------------------------
def parallelism_sweep(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    """ASP vs. mutation frequency, one line per probing parallelism."""
    apply_style()
    fig, ax = plt.subplots()
    for i, (par, g) in enumerate(df.groupby("parallelism")):
        g = g.sort_values("frequency")
        _line_with_band(
            ax,
            g["frequency"].to_numpy(),
            g["asp"].to_numpy(),
            g["asp_ci_lo"].to_numpy(),
            g["asp_ci_hi"].to_numpy(),
            style_idx=i,
            label=f"{int(par)}",
        )
    ax.set_xlabel("Mutation frequency $f$ (per tick)")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Probes / recon")
    ax.set_title("Parallel multi-target probing vs. MTD")
    paths = _save(fig, outdir, "parallelism_sweep")
    return {
        "paths": paths,
        "caption": (
            "ASP vs. mutation frequency as the AI agent probes more assets "
            "concurrently per reconnaissance round (an agent capability). More "
            "parallel probing raises baseline success and shifts the MTD trade-off, "
            "but does not eliminate the decline in ASP as $f$ rises."
        ),
    }


# --------------------------------------------------------------------------
# M6: service_diversity channel decomposition
# --------------------------------------------------------------------------
def diversity_channel_decomposition(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    """Bar chart decomposing service_diversity into its mutation and confusion
    channels at a fixed mutation frequency."""
    apply_style()
    d = df.sort_values("variant_order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(d))
    grays = [str(0.15 + 0.6 * i / max(1, len(d) - 1)) for i in range(len(d))]
    err_lo = np.clip((d["asp"] - d["asp_ci_lo"]).to_numpy(), 0.0, None)
    err_hi = np.clip((d["asp_ci_hi"] - d["asp"]).to_numpy(), 0.0, None)
    ax.bar(
        x,
        d["asp"].to_numpy(),
        color=grays,
        edgecolor="0.0",
        linewidth=0.8,
        yerr=[err_lo, err_hi],
        capsize=3,
        ecolor="0.0",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(d["variant_label"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("Attack success probability (ASP)")
    ax.set_ylim(0, 1.02)
    ax.set_title("service_diversity: mutation vs. confusion channel")
    paths = _save(fig, outdir, "diversity_channel_decomposition")
    return {
        "paths": paths,
        "caption": (
            "Decomposition of the service\\_diversity technique at a fixed mutation "
            "frequency. Relative to a 3-technique base, the mutation channel "
            "(mutating a 4th attribute) and the identification-confusion channel "
            "are isolated and combined, showing how much of the ``+diversity'' "
            "ablation effect comes from each mechanism."
        ),
    }


# --------------------------------------------------------------------------
# Ma: cost-weight sensitivity of the lever ranking
# --------------------------------------------------------------------------
def cost_weight_sensitivity(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    """Strip plot of the deception-vs-cheapest overhead gap at ASP 0.25 across the
    cost-weight factorial, grouped into the all-equal control, settings where the
    winning techniques are cheap, and settings where a winner is expensive. Every
    point lying right of zero means no deception-containing set is ever the cheapest
    lever; the design scales to the larger round-4 grid (49 settings)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    def _category(row: Any) -> int:
        if row["is_control"]:
            return 0
        return 2 if row["varies_winners"] else 1

    cats = df.apply(_category, axis=1).to_numpy()
    bands = {
        0: ("all-equal control", "o", "0.0"),
        1: ("winners cheap", "s", "0.6"),
        2: ("winners expensive (p/e/v $>$ 1)", "^", "0.3"),
    }
    rng = np.random.default_rng(0)  # deterministic jitter
    for c, (label, marker, color) in bands.items():
        sel = cats == c
        if not sel.any():
            continue
        x = df.loc[sel, "gap_at_asp_0.25"].to_numpy()
        y = c + (rng.random(sel.sum()) - 0.5) * 0.5
        ax.scatter(
            x,
            y,
            marker=marker,
            s=26,
            facecolors="none",
            edgecolors=color,
            linewidths=1.0,
            label=f"{label} (n={int(sel.sum())})",
        )
    ax.axvline(0.0, color="0.0", linewidth=1.0)
    gmin = float(df["gap_at_asp_0.25"].min())
    ax.annotate(
        f"min gap = +{gmin:.0f}\n(no deception set cheapest at ASP 0.25)",
        xy=(gmin, 0.0),
        xytext=(gmin + 0.18 * (df["gap_at_asp_0.25"].max() - gmin), 1.0),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", color="0.0", lw=0.9),
    )
    ax.set_yticks(list(bands))
    ax.set_yticklabels([bands[c][0] for c in bands], fontsize=7)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Overhead gap at ASP 0.25 (cheapest deception set $-$ cheapest set)")
    ax.set_title("Lever ranking vs. cost weights (ASP 0.25)")
    ax.legend(fontsize=6.5, loc="lower right")
    paths = _save(fig, outdir, "cost_weight_sensitivity")
    return {
        "paths": paths,
        "caption": (
            "Robustness of the lever ranking to the cost model at the ASP 0.25 operating "
            "target. Each point is the extra overhead the cheapest deception-containing set "
            "needs over the overall cheapest technique set, across a factorial of the four "
            "mutating weights (port, endpoint, shuffling, diversity $\\in\\{1,3\\}$) and "
            "deception $\\in\\{1,2.5,5\\}$, plus an all-equal control, with lean-deception "
            "candidate sets ($\\{$port, deception$\\}$, $\\{$port, endpoint, deception$\\}$) "
            "included. Points are grouped by whether a winning technique "
            "(port/endpoint/diversity) is expensive. Every point lies right of zero, so no "
            "deception set is ever the cheapest lever at ASP 0.25 -- even when the lean "
            "winners are made as expensive as a shuffle. (At the deeper ASP 0.10 target the "
            "lean $\\{$port, deception$\\}$ set is cheapest in a few expensive-endpoint "
            "cells; see the table and RESULTS-DELTA.)"
        ),
    }


# --------------------------------------------------------------------------
# Round 3: decoy-ratio sensitivity of the deception penalty
# --------------------------------------------------------------------------
def decoy_ratio_sweep(df: pd.DataFrame, outdir: Path) -> dict[str, Any]:
    """Deception-vs-cheapest overhead gap at ASP 0.25 as a function of decoy ratio."""
    apply_style()
    d = df.sort_values("decoy_ratio")
    fig, ax = plt.subplots()
    ax.plot(
        d["decoy_ratio"].to_numpy(),
        d["gap_at_asp_0.25"].to_numpy(),
        **_series_style(0),
    )
    ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Decoy ratio (decoys $= $ ratio $\\times N$)")
    ax.set_ylabel("Overhead gap at ASP 0.25\n(deception $-$ cheapest set)")
    ax.set_title("Deception penalty vs. decoy provisioning")
    paths = _save(fig, outdir, "decoy_ratio_sweep")
    return {
        "paths": paths,
        "caption": (
            "Decoy-ratio sensitivity of the structural deception penalty: the extra "
            "overhead the deception stack needs over the cheapest technique set to "
            "reach ASP 0.25, as decoy provisioning varies (real frontier re-runs per "
            "ratio). The gap stays positive at every ratio and grows with the decoy "
            "count, so deception is never the cheapest lever but its cost scales with "
            "provisioning."
        ),
    }
