"""Shared scaffolding for experiments: output layout and cell execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..engine import run_config
from ..metrics import aggregate, to_dataframe


@dataclass
class OutputLayout:
    """Resolved output directories for a run."""

    results: Path
    figures: Path
    tables: Path

    @property
    def raw(self) -> Path:
        return self.results / "raw"

    @property
    def summary(self) -> Path:
        return self.results / "summary"

    def ensure(self) -> OutputLayout:
        for d in (self.results, self.figures, self.tables, self.raw, self.summary):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def create(cls, results: str | Path, figures: str | Path, tables: str | Path) -> OutputLayout:
        return cls(Path(results), Path(figures), Path(tables)).ensure()


@dataclass
class ExperimentOutput:
    """Everything one experiment produced."""

    name: str
    summary: pd.DataFrame
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def output_files(self) -> list[str]:
        files: list[str] = []
        for fig in self.figures:
            files.extend(fig.get("paths", []))
        for tab in self.tables:
            files.extend(v for v in tab.values())
        return files

    def captions(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for fig in self.figures:
            for p in fig.get("paths", []):
                if p.endswith(".pdf"):
                    out[Path(p).name] = fig.get("caption", "")
        return out


@dataclass
class Cell:
    """A single configuration to run, with its descriptive labels."""

    labels: dict[str, Any]
    config: Config


def run_cells(
    cells: Sequence[Cell],
    *,
    parallel: bool,
    raw_dir: Path | None = None,
    raw_prefix: str = "cell",
) -> pd.DataFrame:
    """Run every cell, aggregate to one summary row each, optionally save raw CSVs.

    The bootstrap seed for TTC-median CIs is derived deterministically from the
    cell's config seed and index so summaries are reproducible.
    """
    rows: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        results = run_config(cell.config, parallel=parallel)
        boot_seed = (cell.config.seed * 1_000_003 + i) & 0x7FFFFFFF
        rows.append(aggregate(results, labels=cell.labels, bootstrap_seed=boot_seed))
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            df = to_dataframe(results)
            for k, v in cell.labels.items():
                df[k] = v
            df.to_csv(raw_dir / f"{raw_prefix}_{i:03d}.csv", index=False)
    return pd.DataFrame(rows)
