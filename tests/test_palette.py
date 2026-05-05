"""Tests for ``spatialcellchat._palette`` against the R reference.

Reference CSVs are bundled under ``tests/data/r_dumps/sccp_slice1/``,
originally produced by ``dump_palette.R``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialcellchat._palette import colorRamp3, ggPalette, scPalette

GOLD = Path(__file__).parent / "data" / "r_dumps" / "sccp_slice1"


def _read_gold(name: str) -> pd.DataFrame:
    return pd.read_csv(GOLD / name)


# ----------------------------------------------------------------------
# ggPalette
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,name", [(1, "ggpal_1.csv"), (2, "ggpal_2.csv"),
                                    (5, "ggpal_5.csv"), (10, "ggpal_10.csv"),
                                    (30, "ggpal_30.csv")])
def test_ggPalette_matches_R(n, name):
    gold = _read_gold(name)["color"].to_list()
    out = list(ggPalette(n))
    assert out == gold, f"ggPalette({n}) mismatch.\nR:  {gold}\nPy: {out}"


# ----------------------------------------------------------------------
# scPalette
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n,name", [(1, "scpal_1.csv"), (5, "scpal_5.csv"),
                                    (26, "scpal_26.csv"), (30, "scpal_30.csv"),
                                    (50, "scpal_50.csv")])
def test_scPalette_matches_R(n, name):
    gold = _read_gold(name)["color"].to_list()
    out = list(scPalette(n))
    assert out == gold, f"scPalette({n}) mismatch.\nR:  {gold}\nPy: {out}"


# ----------------------------------------------------------------------
# colorRamp3
# ----------------------------------------------------------------------
def test_colorRamp3_basic():
    gold = _read_gold("cr3_basic.csv")
    fn = colorRamp3([-1, 0, 1], ["green", "white", "red"], space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_two_breaks():
    gold = _read_gold("cr3_two.csv")
    fn = colorRamp3([0, 1], ["#000000", "#ffffff"], space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_multi_break():
    gold = _read_gold("cr3_multi.csv")
    fn = colorRamp3([0, 5, 10, 20], ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a"],
                    space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_with_NaN():
    gold = _read_gold("cr3_na.csv")
    fn = colorRamp3([0, 1], ["#000000", "#ffffff"], space="RGB")
    xs = gold["x"].to_numpy()  # has NaN
    out = fn(xs)
    # R returns NA which we serialize as None / NaN; gold csv has missing.
    for k, exp in enumerate(gold["color"].to_list()):
        if isinstance(exp, float) and np.isnan(exp):
            assert out[k] is None
        else:
            assert out[k] == exp


def test_colorRamp3_transparency():
    gold = _read_gold("cr3_alpha.csv")
    fn = colorRamp3([0, 1], ["#ff0000", "#0000ff"], transparency=0.5, space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_clipping():
    gold = _read_gold("cr3_clip.csv")
    fn = colorRamp3([-1, 0, 1], ["#000000", "#ffffff", "#ff0000"], space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_unsorted_breaks():
    gold = _read_gold("cr3_unsorted.csv")
    fn = colorRamp3([1, -1, 0], ["red", "green", "white"], space="RGB")
    out = list(fn(gold["x"].to_numpy()))
    assert out == gold["color"].to_list()


def test_colorRamp3_validation():
    with pytest.raises(ValueError):
        colorRamp3([0, 1], ["red"], space="RGB")
    with pytest.raises(ValueError):
        colorRamp3([0, 0], ["red", "blue"], space="RGB")
    with pytest.raises(NotImplementedError):
        colorRamp3([0, 1], ["red", "blue"], space="LAB")
