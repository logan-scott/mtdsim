"""Command-line interface for the simulation.

Subcommands::

    mtdsim run-all  --config configs/paper.yaml [--parallel] [--seed N] [--trials N]
    mtdsim run      <experiment> --config configs/paper.yaml [...]
    mtdsim list                     # list available experiments

``run-all`` reproduces every paper artifact; ``run`` executes a single
experiment (useful while iterating). Both honor ``--seed`` and ``--trials``
overrides so a quick low-trial smoke run is one flag away.
"""

from __future__ import annotations

import argparse
from typing import Any

from .config import load_config
from .experiments import run_all as _run_all
from .experiments.common import OutputLayout


def _common_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/paper.yaml", help="path to YAML config")
    p.add_argument("--outdir", default="results", help="directory for raw/summary/manifest")
    p.add_argument("--figures", default="figures", help="directory for figures")
    p.add_argument("--tables", default="tables", help="directory for tables")
    p.add_argument("--seed", type=int, default=None, help="override master seed")
    p.add_argument("--trials", type=int, default=None, help="override engine.n_trials")
    p.add_argument("--parallel", action="store_true", help="run trials across processes")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtdsim", description="MTD-vs-AI attacker simulation.")
    sub = p.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser("run-all", help="regenerate every figure, table, and the manifest")
    _common_opts(p_all)

    p_run = sub.add_parser("run", help="run a single experiment")
    p_run.add_argument("experiment", choices=list(_run_all.EXPERIMENTS))
    _common_opts(p_run)

    sub.add_parser("list", help="list available experiments")
    return p


def _load_with_overrides(args: argparse.Namespace) -> Any:
    cfg = load_config(args.config)
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.trials is not None:
        overrides["engine"] = {"n_trials": args.trials}
    return cfg.with_overrides(**overrides) if overrides else cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for name in _run_all.EXPERIMENTS:
            print(name)
        return 0

    cfg = _load_with_overrides(args)
    layout = OutputLayout.create(args.outdir, args.figures, args.tables)
    verbose = not args.quiet

    if args.command == "run-all":
        _run_all.run_all(
            cfg, layout, parallel=args.parallel, config_path=args.config, verbose=verbose
        )
    elif args.command == "run":
        _run_all.run_all(
            cfg,
            layout,
            parallel=args.parallel,
            only=[args.experiment],
            config_path=args.config,
            verbose=verbose,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
