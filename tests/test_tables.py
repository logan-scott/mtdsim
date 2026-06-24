"""LaTeX/CSV table generation: escaping contract and structure."""

from __future__ import annotations

import pandas as pd

from mtdsim.tables import to_booktabs, write_csv


def test_booktabs_headers_raw_body_escaped(tmp_path):
    df = pd.DataFrame({"name": ["port_rotation"], "asp": [0.5]})
    path = to_booktabs(
        df,
        columns=["name", "asp"],
        headers=["$f$ technique", "ASP"],  # author LaTeX, must stay raw
        formats=[None, ".2f"],
        caption="A 95\\% caption with $math$",
        label="tab:t",
        path=tmp_path / "t.tex",
    )
    text = open(path).read()
    assert "\\toprule" in text and "\\bottomrule" in text and "\\midrule" in text
    # Header math is preserved (not escaped).
    assert "$f$ technique" in text
    # Caption is kept raw.
    assert "95\\% caption with $math$" in text
    # Body text with an underscore IS escaped.
    assert "port\\_rotation" in text
    # Numeric body cell is formatted.
    assert "0.50" in text


def test_write_csv_roundtrips(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    p = write_csv(df, tmp_path / "x.csv")
    back = pd.read_csv(p)
    assert list(back.columns) == ["a", "b"]
    assert back.shape == (2, 2)
