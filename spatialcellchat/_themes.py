"""Theme + colormap helpers ported from ``visualization.R``.

Implements:
- :func:`CellChat_theme_opts` — matplotlib ``rcParams``-style override dict
  reproducing CellChat's ggplot2 ``theme_classic`` customisation.
- :func:`themeFeaturePlot` — feature-plot theme dict (axes hidden, legend
  on right with adjustable text size).
- :func:`generate_custom_colorscale` — port of
  ``visualization.R::generate_custom_colorscale`` (RColorBrewer/viridis →
  99-step ramp; optional plotly-style ``[index, color]`` pairs).
- :func:`generate_RedsBlues_colormap` — port of
  ``visualization.R::generate_RedsBlues_colormap`` (returns 99 hex strings
  spanning the input vector's range).
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import matplotlib as mpl
from matplotlib import colors as mcolors

from ._palette import (
    colorRamp3,
    _hex_from_rgb,
    _hex_to_rgb01,
    _r_color_ramp_palette,
)

__all__ = [
    "CellChat_theme_opts",
    "themeFeaturePlot",
    "generate_custom_colorscale",
    "generate_RedsBlues_colormap",
]


# ---------------------------------------------------------------------------
# CellChat_theme_opts (R: visualization.R:759)
# ---------------------------------------------------------------------------
def CellChat_theme_opts() -> dict:
    """ggplot-style theme overrides as a matplotlib ``rcParams`` dict.

    Mirrors :func:`CellChat_theme_opts` from ``visualization.R``: removes
    panel background and minor/major gridlines, draws black axis lines,
    centres the title in bold size 10, and keeps a clean white background.

    Returns
    -------
    dict
        Mapping of matplotlib rc keys to override values.
    """
    return {
        # strip background = white
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        # panel.border = blank, axis lines black
        "axes.edgecolor": "black",
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        # remove grid (major + minor)
        "axes.grid": False,
        "axes.grid.which": "both",
        # legend key blank (no border / facecolor)
        "legend.frameon": False,
        # plot.title: size 10, bold, centred (hjust = 0.5)
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
    }


# ---------------------------------------------------------------------------
# themeFeaturePlot (R: visualization.R:8083)
# ---------------------------------------------------------------------------
def themeFeaturePlot(legend_text_size: int = 8) -> dict:
    """Theme overrides for feature plots.

    Mirrors :func:`themeFeaturePlot` from ``visualization.R`` — places the
    legend on the right, removes the legend title, hides axis text/ticks,
    and sets the legend key size to ~0.15 inches.

    Parameters
    ----------
    legend_text_size : int, default ``8``
        Font size of legend text.

    Returns
    -------
    dict
        Mapping of matplotlib rc keys to override values.
    """
    return {
        # legend on right with no title, custom text size
        "legend.loc": "right",
        "legend.title_fontsize": 0,  # blank title
        "legend.fontsize": legend_text_size,
        # legend.key.size = unit(0.15, "inches") -> handlelength/height
        "legend.handlelength": 0.15 * 72.0 / 12.0,
        "legend.handleheight": 0.15 * 72.0 / 12.0,
        # panel.background = blank, axis ticks/text removed
        "axes.facecolor": "white",
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.labelleft": False,
        # xlab / ylab = NULL
        "axes.labelpad": 0,
        # legend.key blank
        "legend.frameon": False,
    }


# ---------------------------------------------------------------------------
# generate_custom_colorscale (R: visualization.R:9593)
# ---------------------------------------------------------------------------
# R RColorBrewer hard-coded palettes (subset commonly referenced).
# Source: RColorBrewer::brewer.pal.info.
# We mirror only the maximum-n entry for each palette and slice via the same
# rules brewer.pal uses (return n colours from the palette, with min n = 3).
_BREWER_PALETTES: dict = {
    "Spectral": (
        "#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
        "#FFFFBF", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD",
        "#5E4FA2",
    ),
    "RdBu": (
        "#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
        "#F7F7F7", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC",
        "#053061",
    ),
    "RdYlBu": (
        "#A50026", "#D73027", "#F46D43", "#FDAE61", "#FEE090",
        "#FFFFBF", "#E0F3F8", "#ABD9E9", "#74ADD1", "#4575B4",
        "#313695",
    ),
    "RdYlGn": (
        "#A50026", "#D73027", "#F46D43", "#FDAE61", "#FEE08B",
        "#FFFFBF", "#D9EF8B", "#A6D96A", "#66BD63", "#1A9850",
        "#006837",
    ),
    "PiYG": (
        "#8E0152", "#C51B7D", "#DE77AE", "#F1B6DA", "#FDE0EF",
        "#F7F7F7", "#E6F5D0", "#B8E186", "#7FBC41", "#4D9221",
        "#276419",
    ),
    "BrBG": (
        "#543005", "#8C510A", "#BF812D", "#DFC27D", "#F6E8C3",
        "#F5F5F5", "#C7EAE5", "#80CDC1", "#35978F", "#01665E",
        "#003C30",
    ),
    "PuOr": (
        "#7F3B08", "#B35806", "#E08214", "#FDB863", "#FEE0B6",
        "#F7F7F7", "#D8DAEB", "#B2ABD2", "#8073AC", "#542788",
        "#2D004B",
    ),
    "PRGn": (
        "#40004B", "#762A83", "#9970AB", "#C2A5CF", "#E7D4E8",
        "#F7F7F7", "#D9F0D3", "#A6DBA0", "#5AAE61", "#1B7837",
        "#00441B",
    ),
    # Sequential
    "YlOrRd": ("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C",
               "#FC4E2A", "#E31A1C", "#BD0026", "#800026"),
    "YlOrBr": ("#FFFFE5", "#FFF7BC", "#FEE391", "#FEC44F", "#FE9929",
               "#EC7014", "#CC4C02", "#993404", "#662506"),
    "YlGn":   ("#FFFFE5", "#F7FCB9", "#D9F0A3", "#ADDD8E", "#78C679",
               "#41AB5D", "#238443", "#006837", "#004529"),
    "YlGnBu": ("#FFFFD9", "#EDF8B1", "#C7E9B4", "#7FCDBB", "#41B6C4",
               "#1D91C0", "#225EA8", "#253494", "#081D58"),
    "Reds":   ("#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A",
               "#EF3B2C", "#CB181D", "#A50F15", "#67000D"),
    "Blues":  ("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
               "#4292C6", "#2171B5", "#08519C", "#08306B"),
    "Greens": ("#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B", "#74C476",
               "#41AB5D", "#238B45", "#006D2C", "#00441B"),
    "Oranges":("#FFF5EB", "#FEE6CE", "#FDD0A2", "#FDAE6B", "#FD8D3C",
               "#F16913", "#D94801", "#A63603", "#7F2704"),
    "Purples":("#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC", "#9E9AC8",
               "#807DBA", "#6A51A3", "#54278F", "#3F007D"),
    "Greys":  ("#FFFFFF", "#F0F0F0", "#D9D9D9", "#BDBDBD", "#969696",
               "#737373", "#525252", "#252525", "#000000"),
}

# RColorBrewer's brewer.pal subsets each palette differently for n < max.
# We mirror brewer.pal.info's index tables (from the R source).
_BREWER_INDEX = {
    # Diverging palettes share this index map (max 11 colors).
    "diverging": {
        3: (5, 7, 4),  # 0-based: middle, right side (one), left side (one)
    },
    # Sequential palettes have their own (handled inline).
}


def _brewer_pal(name: str, n: int) -> list:
    """Mirror ``RColorBrewer::brewer.pal(n, name)`` for a subset of palettes.

    The palettes carried here are the diverging set (max n = 11) and the
    nine common sequential palettes (max n = 9). Selection indices match
    RColorBrewer's tables — for n in [3 .. max] the colours are evenly
    distributed across the full palette.
    """
    if name not in _BREWER_PALETTES:
        raise ValueError(f"palette {name!r} not in supported brewer.pal set")
    full = _BREWER_PALETTES[name]
    full_n = len(full)
    if n < 3:
        # brewer.pal warns and returns n=3 for n<3 in R; replicate.
        n = 3
    if n > full_n:
        raise ValueError(
            f"n={n} exceeds maximum {full_n} colours for palette {name!r}"
        )
    if n == full_n:
        return list(full)
    # RColorBrewer uses fixed index tables. The exact subsets for diverging
    # palettes (max 11) are listed below; sequential palettes (max 9) are
    # similar. Rather than encode every table, we replicate the observable
    # behaviour of ``RColorBrewer::brewer.pal`` by indexing R's selection
    # rules; for our use we rely on ``n.colors`` defaulting to 8 in
    # ``generate_custom_colorscale`` and handle that case explicitly.
    return list(_BREWER_SUBSET[name][n])


# Pre-computed brewer.pal subsets matching R's table (n = 3..max).
# Reference: RColorBrewer source (data-raw/brewer.pal.info).
_BREWER_SUBSET: dict = {
    "Spectral": {
        3: ("#FC8D59", "#FFFFBF", "#99D594"),
        4: ("#D7191C", "#FDAE61", "#ABDDA4", "#2B83BA"),
        5: ("#D7191C", "#FDAE61", "#FFFFBF", "#ABDDA4", "#2B83BA"),
        6: ("#D53E4F", "#FC8D59", "#FEE08B", "#E6F598", "#99D594", "#3288BD"),
        7: ("#D53E4F", "#FC8D59", "#FEE08B", "#FFFFBF", "#E6F598", "#99D594", "#3288BD"),
        8: ("#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD"),
        9: ("#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#FFFFBF", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD"),
        10: ("#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2"),
        11: ("#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B", "#FFFFBF", "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2"),
    },
    "YlOrRd": {
        3: ("#FFEDA0", "#FEB24C", "#F03B20"),
        4: ("#FFFFB2", "#FECC5C", "#FD8D3C", "#E31A1C"),
        5: ("#FFFFB2", "#FECC5C", "#FD8D3C", "#F03B20", "#BD0026"),
        6: ("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C", "#F03B20", "#BD0026"),
        7: ("#FFFFB2", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"),
        8: ("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"),
        9: ("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#BD0026", "#800026"),
    },
    "Reds": {
        3: ("#FEE0D2", "#FC9272", "#DE2D26"),
        4: ("#FEE5D9", "#FCAE91", "#FB6A4A", "#CB181D"),
        5: ("#FEE5D9", "#FCAE91", "#FB6A4A", "#DE2D26", "#A50F15"),
        6: ("#FEE5D9", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26", "#A50F15"),
        7: ("#FEE5D9", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D"),
        8: ("#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D"),
        9: ("#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"),
    },
    "Blues": {
        3: ("#DEEBF7", "#9ECAE1", "#3182BD"),
        4: ("#EFF3FF", "#BDD7E7", "#6BAED6", "#2171B5"),
        5: ("#EFF3FF", "#BDD7E7", "#6BAED6", "#3182BD", "#08519C"),
        6: ("#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#3182BD", "#08519C"),
        7: ("#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"),
        8: ("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"),
        9: ("#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"),
    },
}


def generate_custom_colorscale(
    color_heatmap: Union[str, Sequence[str]],
    n_colors: int = 8,
    reversescale: bool = False,
    plotly_format: bool = True,
) -> list:
    """Build a 99-step colour scale from an RColorBrewer palette name.

    Mirrors :func:`generate_custom_colorscale` from ``visualization.R``.
    When ``color_heatmap`` is a length-one palette name (e.g. ``"Spectral"``)
    the routine pulls ``n_colors`` colours from RColorBrewer (or falls back
    to ``viridis`` if the palette is unknown), reverses if requested,
    expands to 99 steps via R-style ``colorRampPalette``, replaces the first
    colour with ``"#E5E5E5"``, and (optionally) wraps in plotly's
    ``[[index, colour], ...]`` format. When ``color_heatmap`` is already a
    vector of colours, that vector is used as-is.

    Parameters
    ----------
    color_heatmap : str or sequence of str
        Palette name or a vector of hex colours.
    n_colors : int, default ``8``
        Number of palette anchors before expansion. Must be ≥ 2.
    reversescale : bool, default ``False``
        Reverse the colour order before expansion.
    plotly_format : bool, default ``True``
        If ``True``, return ``[[idx, hex], ...]`` plotly-compatible pairs.

    Returns
    -------
    list
        Either a list of 99 hex strings, or a list of length 99 plotly
        ``[index, colour]`` pairs. (When ``color_heatmap`` is itself a
        vector, the length matches the input vector.)
    """
    if isinstance(color_heatmap, str):
        if color_heatmap in _BREWER_SUBSET:
            colormap = list(_brewer_pal(color_heatmap, n_colors))
        else:
            # Fallback: viridis-style with direction = -1 (R: scales::viridis_pal)
            cmap = mpl.colormaps[color_heatmap].resampled(n_colors)
            samples = np.linspace(1.0, 0.0, n_colors)  # direction = -1
            cols = cmap(samples)[:, :3]
            colormap = list(_hex_from_rgb(cols, alpha=""))
        if reversescale:
            colormap = list(reversed(colormap))
        # Expand to 99 colours with R's colorRampPalette
        colormap = list(_r_color_ramp_palette(colormap, 99))
        colormap[0] = "#E5E5E5"
    else:
        colormap = list(color_heatmap)

    if n_colors < 2:
        raise ValueError("`n.colors` must be greater than 1.")

    if plotly_format:
        len_cm = len(colormap)
        index = np.linspace(0.0, 1.0, len_cm)
        return [[float(index[k]), str(colormap[k])] for k in range(len_cm)]
    return [str(c) for c in colormap]


# ---------------------------------------------------------------------------
# generate_RedsBlues_colormap (R: visualization.R:9642)
# ---------------------------------------------------------------------------
_REDS = ("#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15")
_BLUES = ("#eff3ff", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c")


def generate_RedsBlues_colormap(
    vec: Sequence[float],
    mid_color: str = "#E5E5E5",
    switch_RedsBlues: bool = False,
) -> np.ndarray:
    """99-colour Reds/Blues ramp anchored at zero.

    Mirrors :func:`generate_RedsBlues_colormap` from ``visualization.R``.
    Returns 99 ``"#RRGGBBAA"`` strings sampling :func:`colorRamp3` between
    ``min(vec)`` and ``max(vec)``. When the data is sign-mixed both Blues
    and Reds anchor either side of zero (with ``mid_color`` exactly at 0).

    Parameters
    ----------
    vec : sequence of float
        Numeric vector. NaNs are dropped before computing the range.
    mid_color : str, default ``"#E5E5E5"``
        Colour anchored at zero (or at ``max`` / ``min`` for sign-uniform vectors).
    switch_RedsBlues : bool, default ``False``
        Swap the Reds and Blues anchors.

    Returns
    -------
    numpy.ndarray
        Array of 99 ``"#RRGGBBAA"`` strings.
    """
    arr = np.asarray(vec, dtype=float)
    arr = arr[~np.isnan(arr)]
    min_vec = float(np.min(arr))
    max_vec = float(np.max(arr))
    Colors_length = 6

    Reds = list(_REDS)
    Blues = list(_BLUES)
    if switch_RedsBlues:
        Blues = list(_REDS)
        Reds = list(_BLUES)

    if 0 <= min_vec:
        breaks = np.linspace(0.0, max_vec, Colors_length + 1)
        col_fun = colorRamp3(breaks.tolist(), [mid_color] + Reds)
    elif 0 >= max_vec:
        breaks = np.linspace(min_vec, 0.0, Colors_length + 1)
        col_fun = colorRamp3(breaks.tolist(), list(reversed(Blues)) + [mid_color])
    else:
        Blues_breaks = np.linspace(min_vec, 0.0, Colors_length + 1)
        Reds_breaks = np.linspace(0.0, max_vec, Colors_length + 1)
        breaks = np.concatenate([Blues_breaks, Reds_breaks[1:]])
        col_fun = colorRamp3(
            breaks.tolist(),
            list(reversed(Blues)) + [mid_color] + Reds,
        )

    samples = np.linspace(min_vec, max_vec, 99)
    return col_fun(samples)
