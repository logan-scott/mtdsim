"""Summary tables as CSV (analysis) and LaTeX booktabs (direct paper inclusion).

The LaTeX output is hand-rolled rather than delegated to ``DataFrame.to_latex``
so that we control escaping, column formatting, and booktabs rules exactly, and
so the generated ``.tex`` compiles with only ``\\usepackage{booktabs}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)


def _fmt(value: Any, spec: str | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    if spec is None:
        return _escape(str(value))
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return _escape(str(value))


def _escape(text: str) -> str:
    """Escape LaTeX-special characters in plain text cells."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def to_booktabs(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    headers: Sequence[str],
    formats: Sequence[str | None],
    caption: str,
    label: str,
    path: str | Path,
    align: str | None = None,
) -> str:
    """Write a booktabs LaTeX table for the selected columns.

    ``formats`` is a per-column Python format spec (e.g. ``".3f"``) or ``None``
    for verbatim text (which is LaTeX-escaped).
    """
    assert len(columns) == len(headers) == len(formats)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    colspec = align or ("l" + "r" * (len(columns) - 1))
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        # Headers are author-controlled LaTeX (may contain math like $f$); kept raw.
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], fmt) for c, fmt in zip(columns, formats, strict=True)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


# Column presets reused by several experiments.
SWEEP_COLUMNS = [
    ("frequency", "$f$", ".4g"),
    ("asp", "ASP", ".3f"),
    ("asp_ci_lo", "ASP$_{lo}$", ".3f"),
    ("asp_ci_hi", "ASP$_{hi}$", ".3f"),
    ("ttc_median", "med.\\ TTC", ".1f"),
    ("ttc_mean", "mean TTC", ".1f"),
    ("overhead_mean", "overhead", ".1f"),
    ("forced_recons_mean", "re-recons", ".2f"),
    ("stale_fraction_mean", "stale frac.", ".3f"),
]


def write_sweep_table(
    df: pd.DataFrame, *, caption: str, label: str, csv_path: str | Path, tex_path: str | Path
) -> dict[str, str]:
    cols = [c for c, _, _ in SWEEP_COLUMNS]
    heads = [h for _, h, _ in SWEEP_COLUMNS]
    fmts = [f for _, _, f in SWEEP_COLUMNS]
    csv = write_csv(df, csv_path)
    tex = to_booktabs(
        df.sort_values("frequency"),
        columns=cols,
        headers=heads,
        formats=fmts,
        caption=caption,
        label=label,
        path=tex_path,
    )
    return {"csv": csv, "tex": tex}
