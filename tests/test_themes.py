"""Tests for ``spatialcellchat._themes``.

Theme functions (``CellChat_theme_opts`` / ``themeFeaturePlot``) return
matplotlib ``rcParams``-style dicts. R's ggplot equivalents are not bytewise
comparable, so we validate the *contract*: required keys are present and the
returned values are the documented matplotlib equivalents (no axis grids, no
spines, white panel, etc.).

Colormap functions (``generate_custom_colorscale`` /
``generate_RedsBlues_colormap``) ARE compared against the R reference CSVs
bundled under ``tests/data/r_dumps/sccp_slice1/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from spatialcellchat._themes import (
    CellChat_theme_opts,
    generate_custom_colorscale,
    generate_RedsBlues_colormap,
    themeFeaturePlot,
)

GOLD = Path(__file__).parent / "data" / "r_dumps" / "sccp_slice1"


def _read_gold(name: str) -> pd.DataFrame:
    return pd.read_csv(GOLD / name)


# ----------------------------------------------------------------------
# CellChat_theme_opts
# ----------------------------------------------------------------------
def test_CellChat_theme_opts_keys_and_values():
    opts = CellChat_theme_opts()
    # Spec features port: white panel, no grid, axis lines, bold centred title
    assert opts["axes.facecolor"] == "white"
    assert opts["figure.facecolor"] == "white"
    assert opts["axes.edgecolor"] == "black"
    assert opts["axes.grid"] is False
    assert opts["axes.spines.top"] is False
    assert opts["axes.spines.right"] is False
    assert opts["axes.spines.left"] is True
    assert opts["axes.spines.bottom"] is True
    assert opts["legend.frameon"] is False
    assert opts["axes.titlesize"] == 10
    assert opts["axes.titleweight"] == "bold"
    assert opts["axes.titlelocation"] == "center"


def test_CellChat_theme_opts_can_be_applied_to_rcParams():
    opts = CellChat_theme_opts()
    with matplotlib.rc_context(opts):
        # If keys are invalid this raises; sanity test
        assert matplotlib.rcParams["axes.titleweight"] == "bold"


# ----------------------------------------------------------------------
# themeFeaturePlot
# ----------------------------------------------------------------------
def test_themeFeaturePlot_default():
    opts = themeFeaturePlot()
    assert opts["legend.fontsize"] == 8
    assert opts["xtick.bottom"] is False
    assert opts["ytick.left"] is False
    assert opts["axes.facecolor"] == "white"
    assert opts["legend.frameon"] is False


def test_themeFeaturePlot_custom_size():
    opts = themeFeaturePlot(legend_text_size=12)
    assert opts["legend.fontsize"] == 12


def test_themeFeaturePlot_can_apply():
    opts = themeFeaturePlot(legend_text_size=10)
    with matplotlib.rc_context(opts):
        assert matplotlib.rcParams["legend.fontsize"] == 10


# ----------------------------------------------------------------------
# generate_custom_colorscale (against R)
# ----------------------------------------------------------------------
def test_generate_custom_colorscale_spectral():
    gold = _read_gold("ccs_spectral.csv")["color"].to_list()
    out = generate_custom_colorscale(
        "Spectral", n_colors=8, reversescale=False, plotly_format=False,
    )
    assert out == gold


def test_generate_custom_colorscale_reverse():
    gold = _read_gold("ccs_spectral_rev.csv")["color"].to_list()
    out = generate_custom_colorscale(
        "Spectral", n_colors=8, reversescale=True, plotly_format=False,
    )
    assert out == gold


def test_generate_custom_colorscale_ylorred():
    gold = _read_gold("ccs_yorred.csv")["color"].to_list()
    out = generate_custom_colorscale(
        "YlOrRd", n_colors=5, reversescale=False, plotly_format=False,
    )
    assert out == gold


def test_generate_custom_colorscale_vector_input():
    gold = _read_gold("ccs_vec.csv")["color"].to_list()
    out = generate_custom_colorscale(
        ["#000000", "#ffffff"], n_colors=8, reversescale=False, plotly_format=False,
    )
    # When color_heatmap is a vector, R returns the vector exactly (case
    # preserved) — so compare case-insensitively to the gold dump.
    assert [c.lower() for c in out] == [c.lower() for c in gold]


def test_generate_custom_colorscale_plotly_format():
    gold = _read_gold("ccs_plotly.csv")
    out = generate_custom_colorscale(
        "Spectral", n_colors=8, reversescale=False, plotly_format=True,
    )
    assert len(out) == len(gold)
    for k, (idx, color) in enumerate(out):
        np.testing.assert_allclose(idx, gold["idx"].iloc[k], rtol=1e-12, atol=1e-12)
        assert color == gold["color"].iloc[k]


def test_generate_custom_colorscale_n_colors_validation():
    with pytest.raises(ValueError):
        # vector branch still triggers the n_colors < 2 stop
        generate_custom_colorscale(["#000", "#fff"], n_colors=1, plotly_format=False)


# ----------------------------------------------------------------------
# generate_RedsBlues_colormap (against R)
# ----------------------------------------------------------------------
def test_generate_RedsBlues_pos_only():
    gold = _read_gold("rb_pos.csv")["color"].to_list()
    out = list(generate_RedsBlues_colormap([0, 0.5, 1, 2, 5]))
    assert out == gold


def test_generate_RedsBlues_neg_only():
    gold = _read_gold("rb_neg.csv")["color"].to_list()
    out = list(generate_RedsBlues_colormap([-5, -2, -1, -0.5, 0]))
    assert out == gold


def test_generate_RedsBlues_both():
    gold = _read_gold("rb_both.csv")["color"].to_list()
    out = list(generate_RedsBlues_colormap([-5, -2, 0, 2, 5]))
    assert out == gold


def test_generate_RedsBlues_switch():
    gold = _read_gold("rb_switch.csv")["color"].to_list()
    out = list(generate_RedsBlues_colormap([-3, -1, 0, 1, 3], switch_RedsBlues=True))
    assert out == gold


def test_generate_RedsBlues_mid_white():
    gold = _read_gold("rb_midwhite.csv")["color"].to_list()
    out = list(generate_RedsBlues_colormap([-2, 0, 2], mid_color="#ffffff"))
    assert out == gold
