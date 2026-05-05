"""Network-graph visualisations for the cell-cell communication network.

Direct port of the network-graph exports in
``SpatialCellChat::visualization.R`` lines 866-2324 (top dispatchers and
graph plots), lines 2387-2507 (``netVisual_diffInteraction``) and lines
3262-3752 (chord-based exports).

NOTE: in R, every export in this module renders via
:func:`igraph::plot.igraph` on top of base R graphics. The Python port is
a deliberate redesign — vertices, edges and labels are drawn on a
:class:`matplotlib.axes.Axes` instance and the layout is computed via
:mod:`igraph` (Python igraph). The chord exports use :mod:`pycirclize`
in lieu of ``circlize::chordDiagram``. R outputs are
``recordedplot``/``ggplot``; here every public function returns a
:class:`matplotlib.figure.Figure`.

Everything else (data flow, defaults, edge filtering rules, sort orders,
colour usage, scale factors, label positions, NA handling) follows the
R reference line-by-line.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import igraph
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ._centrality import centr_value
from ._palette import scPalette
from .modeling import _ensure_uns, _get_idents
from .database import searchPair

__all__ = [
    "netVisual_circle",
    "netVisual_hierarchy1",
    "netVisual_hierarchy2",
    "netVisual_diffInteraction",
    "netVisual_spatial",
    "netVisual_chord_cell",
    "netVisual_chord_cell_internal",
    "netVisual_chord_gene",
    "netVisual",
    "netVisual_aggregate",
    "netVisual_individual",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_named_matrix(
    net: Union[np.ndarray, pd.DataFrame],
    cell_levels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return ``(matrix_array, names)`` from a 2-D ``net`` argument.

    Mirrors the R idiom where ``rownames(net)`` is used to derive cell
    levels. ``net`` may be a :class:`pandas.DataFrame`, a :class:`numpy.ndarray`
    or a list-of-lists. If ``cell_levels`` is provided, it overrides the
    DataFrame index.
    """
    if isinstance(net, pd.DataFrame):
        levels = list(net.index) if cell_levels is None else list(cell_levels)
        arr = net.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(net, dtype=float).copy()
        if cell_levels is None:
            levels = [str(i) for i in range(arr.shape[0])]
        else:
            levels = list(cell_levels)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("net must be a square 2-D matrix")
    if len(levels) != arr.shape[0]:
        raise ValueError("cell_levels length must match net dim")
    return arr, levels


def _resolve_indices(
    use: Sequence,
    cell_levels: Sequence[str],
) -> List[str]:
    """Return resolved cell names from numeric or character indices."""
    out: List[str] = []
    for v in use:
        if isinstance(v, (int, np.integer)):
            # R uses 1-based indices.
            out.append(cell_levels[int(v) - 1])
        else:
            out.append(str(v))
    return out


def _filter_by_sources_targets(
    net: np.ndarray,
    cell_levels: List[str],
    sources_use: Optional[Sequence],
    targets_use: Optional[Sequence],
    idents_use: Optional[Sequence] = None,
) -> np.ndarray:
    """Apply ``sources.use`` / ``targets.use`` / ``idents.use`` filters.

    Matches the R idiom: melt → filter → tapply (``sum``) → fill NA.
    """
    if sources_use is None and targets_use is None and idents_use is None:
        return net
    K = len(cell_levels)
    df = pd.DataFrame(
        {
            "source": np.repeat(cell_levels, K),
            "target": np.tile(cell_levels, K),
            "value": net.T.ravel(order="F"),  # melt is column-major in R
        }
    )
    # Note: using row-major ordering for clarity since net is 2D-numpy
    df = pd.DataFrame(
        {
            "source": np.repeat(cell_levels, K),
            "target": np.tile(cell_levels, K),
            "value": net.ravel(order="C"),
        }
    )
    if sources_use is not None:
        sources_use = _resolve_indices(list(sources_use), cell_levels)
        df = df[df["source"].isin(sources_use)]
    if targets_use is not None:
        targets_use = _resolve_indices(list(targets_use), cell_levels)
        df = df[df["target"].isin(targets_use)]
    if idents_use is not None:
        idents_use = _resolve_indices(list(idents_use), cell_levels)
        df = df[df["source"].isin(idents_use) | df["target"].isin(idents_use)]
    df["source"] = pd.Categorical(df["source"], categories=cell_levels)
    df["target"] = pd.Categorical(df["target"], categories=cell_levels)
    df["value"] = df["value"].fillna(0.0)
    out = (
        df.groupby(["source", "target"], observed=False)["value"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(index=cell_levels, columns=cell_levels)
        .to_numpy(dtype=float)
    )
    out = np.where(np.isnan(out), 0.0, out)
    return out


def _remove_isolate(
    net: np.ndarray,
    cell_levels: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Drop rows + cols that are all-zero (R's ``remove.isolate`` branch)."""
    row_zero = net.sum(axis=1) == 0
    col_zero = net.sum(axis=0) == 0
    drop = row_zero & col_zero
    keep = ~drop
    levels_new = [cell_levels[i] for i in np.flatnonzero(keep)]
    return net[np.ix_(keep, keep)], levels_new


def _adjust_color_alpha(color: str, alpha: float) -> Tuple[float, float, float, float]:
    """Convert ``color`` to RGBA with the given alpha (R: ``adjustcolor``)."""
    rgb = mcolors.to_rgb(color)
    return (rgb[0], rgb[1], rgb[2], float(alpha))


def _quantile_threshold(net: np.ndarray, top: float) -> float:
    """R ``stats::quantile(net, probs = 1-top)`` with default type=7."""
    return float(np.quantile(net, 1.0 - top))


def _abs_quantile_threshold(net: np.ndarray, top: float) -> float:
    """R ``stats::quantile(abs(net), probs = 1-top)``."""
    return float(np.quantile(np.abs(net), 1.0 - top))


def _circle_layout(n: int) -> np.ndarray:
    """Return ``n × 2`` circle layout coordinates (igraph ``in_circle()``).

    Matches igraph's R/Python implementation: equally spaced points on a
    unit circle starting at angle ``π/2`` (top), going counter-clockwise.
    """
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    g = igraph.Graph(n=n, directed=False)
    coords = np.array(g.layout_circle().coords, dtype=float)
    return coords


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    """R ``base::scale`` — centre and divide by sample SD."""
    if coords.shape[0] <= 1:
        return coords
    mean = coords.mean(axis=0)
    centred = coords - mean
    # R uses sample SD (ddof=1) by default
    sd = centred.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return centred / sd


def _radian_rescale(n: int, start: float = 0.0, direction: int = -1) -> np.ndarray:
    """Mirror R's local ``radian.rescale`` from ``netVisual_circle``."""
    if n <= 1:
        return np.zeros(n)
    x = np.arange(1, n + 1, dtype=float)
    rng = x.max() - x.min()
    rescaled = (x - x.min()) / rng * (2.0 * np.pi)
    return ((rescaled + start) % (2.0 * np.pi)) * direction


def _draw_curved_edge(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    width: float,
    color: Tuple[float, float, float, float],
    curvature: float = 0.2,
    arrow: bool = True,
    arrow_size: float = 0.2,
    arrow_width: float = 1.0,
) -> None:
    """Draw a curved edge between ``p1`` → ``p2`` with optional arrowhead.

    Mirrors igraph's ``edge.curved`` rendering: a quadratic Bezier whose
    control point is offset perpendicular to the chord by ``curvature *
    chord_length``.
    """
    if p1 == p2:
        # self-loop — draw as a small circle anchored offset from the vertex
        return
    p1a = np.asarray(p1, dtype=float)
    p2a = np.asarray(p2, dtype=float)
    if arrow:
        style = "-|>" if arrow else "-"
        connectionstyle = f"arc3,rad={curvature}"
        arrow_obj = mpatches.FancyArrowPatch(
            posA=p1a,
            posB=p2a,
            connectionstyle=connectionstyle,
            arrowstyle=style,
            mutation_scale=10.0 * arrow_size + 5.0,
            linewidth=width,
            color=color,
            shrinkA=8.0,
            shrinkB=8.0,
        )
        ax.add_patch(arrow_obj)
    else:
        connectionstyle = f"arc3,rad={curvature}"
        line = mpatches.FancyArrowPatch(
            posA=p1a,
            posB=p2a,
            connectionstyle=connectionstyle,
            arrowstyle="-",
            linewidth=width,
            color=color,
        )
        ax.add_patch(line)


def _network_render(
    layout_coords: np.ndarray,
    cell_levels: List[str],
    net: np.ndarray,
    color_use: List[str],
    vertex_weight: np.ndarray,
    *,
    edge_weight_max: float,
    edge_width_max: float,
    weight_scale: bool,
    alpha_edge: float,
    edge_curved: float,
    arrow_size: float,
    arrow_width: float,
    label_cex: float,
    label_color: str,
    label_dist: np.ndarray,
    title_name: Optional[str],
    label_locs_rad: Optional[np.ndarray],
    figsize: Tuple[float, float] = (6.0, 6.0),
    margin: float = 0.2,
    fig_ax: Optional[Tuple[Figure, plt.Axes]] = None,
    edge_labels: Optional[List[str]] = None,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.5,
    color_use_frame: Optional[List[str]] = None,
    arrow: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Render a directed weighted graph onto a matplotlib Figure/Axes.

    Internal H3 helper. The visual aesthetic is a redesign rather than
    a drift; see module docstring.
    """
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = fig_ax
    n = len(cell_levels)
    if color_use_frame is None:
        color_use_frame = list(color_use)

    # Edge list (from non-zero entries of net)
    rows, cols = np.nonzero(net)
    weights = net[rows, cols]
    for r, c, w in zip(rows, cols, weights):
        if weight_scale:
            ew = 0.3 + (w / edge_weight_max) * edge_width_max
        else:
            ew = 0.3 + edge_width_max * w
        col_i = mcolors.to_rgb(color_use[r])
        edge_color = (col_i[0], col_i[1], col_i[2], float(alpha_edge))
        p1 = (float(layout_coords[r, 0]), float(layout_coords[r, 1]))
        p2 = (float(layout_coords[c, 0]), float(layout_coords[c, 1]))
        if r == c:
            # Self-loop: draw a small offset circle
            cx, cy = p1
            radius = 0.05 + 0.05 * (w / max(edge_weight_max, 1e-12))
            loop = mpatches.Circle(
                (cx + radius, cy + radius), radius=radius,
                fill=False, edgecolor=edge_color, linewidth=ew,
            )
            ax.add_patch(loop)
            continue
        _draw_curved_edge(
            ax, p1, p2, width=float(ew), color=edge_color,
            curvature=float(edge_curved),
            arrow=arrow, arrow_size=arrow_size, arrow_width=arrow_width,
        )

    # Vertices
    for i, name in enumerate(cell_levels):
        sz = float(vertex_weight[i]) if hasattr(vertex_weight, "__len__") else float(vertex_weight)
        ax.scatter(
            layout_coords[i, 0], layout_coords[i, 1],
            s=(sz * 8.0) ** 2 / 4.0,  # roughly map weight units → marker area
            c=[color_use[i]], edgecolors=color_use_frame[i], linewidths=1.0, zorder=3,
        )
        if label_cex > 0:
            # Place the label outside the vertex along ``label_locs_rad``
            if label_locs_rad is not None and i < len(label_locs_rad):
                ang = float(label_locs_rad[i])
                ld = float(label_dist[i]) if hasattr(label_dist, "__len__") else float(label_dist)
                lx = layout_coords[i, 0] + ld * 0.04 * math.cos(ang)
                ly = layout_coords[i, 1] + ld * 0.04 * math.sin(ang)
            else:
                lx = layout_coords[i, 0]
                ly = layout_coords[i, 1]
            if name:
                ax.text(
                    lx, ly, name, ha="center", va="center",
                    fontsize=10.0 * float(label_cex), color=label_color, zorder=4,
                )

    # Edge labels
    if edge_labels is not None:
        for r, c, lbl in zip(rows, cols, edge_labels):
            mx = (layout_coords[r, 0] + layout_coords[c, 0]) / 2.0
            my = (layout_coords[r, 1] + layout_coords[c, 1]) / 2.0
            ax.text(
                mx, my, lbl, fontsize=10.0 * float(edge_label_cex),
                color=edge_label_color, zorder=5,
            )

    # Title
    if title_name is not None:
        ax.set_title(title_name, fontsize=11)

    # Aesthetic clean-up
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if layout_coords.shape[0] > 0:
        xs = layout_coords[:, 0]
        ys = layout_coords[:, 1]
        rng = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
        cx = 0.5 * (xs.max() + xs.min())
        cy = 0.5 * (ys.max() + ys.min())
        pad = (1.0 + float(margin)) * rng / 2.0
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
    return fig, ax


# ---------------------------------------------------------------------------
# netVisual_circle (R: visualization.R:1991)
# ---------------------------------------------------------------------------
def netVisual_circle(
    net: Union[np.ndarray, pd.DataFrame],
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    idents_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    vertex_label_cex: float = 1.0,
    vertex_label_color: str = "black",
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    alpha_edge: float = 0.6,
    label_edge: bool = False,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.8,
    edge_curved: float = 0.2,
    shape: str = "circle",
    margin: float = 0.2,
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> Figure:
    """Circle plot of cell-cell communication.

    Direct port of R's :func:`netVisual_circle` (``visualization.R:1991``).
    The width of edges represents the strength of communication.

    Parameters
    ----------
    net : ndarray or DataFrame
        Square weighted adjacency matrix; row/col names define cell groups.
    color_use, sources_use, targets_use, idents_use, remove_isolate, top,
    weight_scale, vertex_weight, vertex_weight_max, vertex_size_max,
    vertex_label_cex, vertex_label_color, edge_weight_max, edge_width_max,
    alpha_edge, label_edge, edge_label_color, edge_label_cex, edge_curved,
    shape, margin, arrow_width, arrow_size :
        See R reference.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered Figure.
    """
    if vertex_size_max is None:
        if hasattr(vertex_weight, "__len__") and len(set(np.atleast_1d(vertex_weight).tolist())) == 1:
            vertex_size_max = 5.0
        elif not hasattr(vertex_weight, "__len__"):
            vertex_size_max = 5.0
        else:
            vertex_size_max = 15.0

    arr, levels = _ensure_named_matrix(net)
    thresh = _quantile_threshold(arr, top)
    arr = np.where(arr < thresh, 0.0, arr)
    arr = _filter_by_sources_targets(arr, levels, sources_use, targets_use, idents_use)
    arr = np.where(np.isnan(arr), 0.0, arr)
    if remove_isolate:
        arr, levels = _remove_isolate(arr, levels)
    n = len(levels)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        if title_name is not None:
            ax.set_title(title_name)
        return fig

    if color_use is None:
        color_use = list(scPalette(n))
    else:
        color_use = list(color_use)[:n] if len(color_use) >= n else list(color_use) + list(scPalette(n - len(color_use)))

    coords = _circle_layout(n)
    coords_scale = _scale_coords(coords) if n > 1 else coords

    if vertex_weight_max is None:
        vw_max = float(np.max(vertex_weight))
    else:
        vw_max = float(vertex_weight_max)
    if hasattr(vertex_weight, "__len__"):
        vw = np.asarray(vertex_weight, dtype=float) / vw_max * float(vertex_size_max) + 5.0
    else:
        vw = np.full(n, float(vertex_weight) / vw_max * float(vertex_size_max) + 5.0)

    if edge_weight_max is None:
        edge_weight_max = float(arr.max()) if arr.size > 0 and arr.max() > 0 else 1.0

    label_locs = _radian_rescale(n, start=0.0, direction=-1)
    label_dist = vw / vw.max() + 2.0 if vw.size > 0 else np.full(n, 2.0)

    edge_labels: Optional[List[str]] = None
    if label_edge:
        rows, cols = np.nonzero(arr)
        edge_labels = [f"{round(float(arr[r, c]), 1)}" for r, c in zip(rows, cols)]

    fig, ax = _network_render(
        coords_scale, levels, arr, color_use, vw,
        edge_weight_max=float(edge_weight_max),
        edge_width_max=float(edge_width_max),
        weight_scale=bool(weight_scale),
        alpha_edge=float(alpha_edge),
        edge_curved=float(edge_curved),
        arrow_size=float(arrow_size),
        arrow_width=float(arrow_width),
        label_cex=float(vertex_label_cex),
        label_color=str(vertex_label_color),
        label_dist=label_dist,
        title_name=title_name,
        label_locs_rad=label_locs,
        figsize=figsize,
        margin=float(margin),
        edge_labels=edge_labels,
        edge_label_color=str(edge_label_color),
        edge_label_cex=float(edge_label_cex),
    )
    return fig


# ---------------------------------------------------------------------------
# netVisual_diffInteraction (R: visualization.R:2387)
# ---------------------------------------------------------------------------
def netVisual_diffInteraction(
    adata_list: Sequence[ad.AnnData],
    comparison: Sequence[int] = (1, 2),
    measure: str = "count",
    color_use: Optional[Sequence[str]] = None,
    color_edge: Sequence[str] = ("#b2182b", "#2166ac"),
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: float = 15.0,
    vertex_label_cex: float = 1.0,
    vertex_label_color: str = "black",
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    alpha_edge: float = 0.6,
    label_edge: bool = False,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.8,
    edge_curved: float = 0.2,
    shape: str = "circle",
    margin: float = 0.2,
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> Figure:
    """Differential cell-cell communication network between two datasets.

    Direct port of R's :func:`netVisual_diffInteraction`
    (``visualization.R:2387``). R parity: takes a merged CellChat object
    whose ``@net`` slots are per-dataset lists; in Python the natural
    equivalent is a ``Sequence[ad.AnnData]`` (R's
    ``mergeCellChat(list(c1, c2))`` is a no-op in this design — just pass
    ``[c1, c2]`` directly).

    Parameters
    ----------
    adata_list : sequence of anndata.AnnData
        Per-dataset CellChat-state AnnDatas (``len ≥ 2`` required;
        single-dataset input raises, mirroring R's
        ``stop("...single dataset!")`` discriminator).
    comparison : sequence of int
        1-based indices into ``adata_list`` (e.g. ``(1, 2)`` selects the
        first two datasets, computing ``net2 - net1``).
    measure : str
        ``"count"``, ``"weight"``, ``"count.merged"`` or ``"weight.merged"`` —
        which ``adata.uns['cellchat']['net'][<measure>]`` matrix to diff.

    Notes
    -----
    See the R reference (``SpatialCellChat::netVisual_*``) for all other parameters.
    """
    if measure not in ("count", "weight", "count.merged", "weight.merged"):
        raise ValueError(f"Unknown measure: {measure}")
    if len(adata_list) < 2:
        raise ValueError(
            "netVisual_diffInteraction requires at least 2 AnnDatas (R "
            "parity: `is.list(object@net[[1]])` discriminator at "
            "visualization.R:2387). Pass `[adata_ctrl, adata_treat]`."
        )
    # R parity: ``obj1 <- object@net[[comparison[1]]][[measure]]``
    #           ``obj2 <- object@net[[comparison[2]]][[measure]]``
    i1 = int(comparison[0]) - 1
    i2 = int(comparison[1]) - 1
    cc1 = adata_list[i1].uns.get("cellchat", {})
    cc2 = adata_list[i2].uns.get("cellchat", {})
    if "net" not in cc1 or measure not in cc1["net"]:
        raise ValueError(
            f"adata_list[{i1}].uns['cellchat']['net'][{measure!r}] is missing — "
            "run aggregateNet first."
        )
    if "net" not in cc2 or measure not in cc2["net"]:
        raise ValueError(
            f"adata_list[{i2}].uns['cellchat']['net'][{measure!r}] is missing — "
            "run aggregateNet first."
        )
    obj1 = cc1["net"][measure]
    obj2 = cc2["net"][measure]
    obj1_arr, levels1 = _ensure_named_matrix(obj1)
    obj2_arr, levels2 = _ensure_named_matrix(obj2)
    # R parity (visualization.R:2396): ``net.diff <- obj2 - obj1`` —
    # both matrices must share row/col labels (i.e. cell-group axis is
    # the joint factor). When the two datasets have differently-ordered
    # idents, broadcast via the joint level set and reindex.
    if levels1 != levels2:
        joint = list(dict.fromkeys(list(levels1) + list(levels2)))
        df1 = pd.DataFrame(obj1_arr, index=levels1, columns=levels1).reindex(
            index=joint, columns=joint, fill_value=0.0
        )
        df2 = pd.DataFrame(obj2_arr, index=levels2, columns=levels2).reindex(
            index=joint, columns=joint, fill_value=0.0
        )
        obj1_arr = df1.to_numpy(dtype=float)
        obj2_arr = df2.to_numpy(dtype=float)
        levels = joint
    else:
        levels = list(levels1)
    net_diff = obj2_arr - obj1_arr

    if title_name is None:
        if measure in ("count", "count.merged"):
            title_name = "Differential number of interactions"
        else:
            title_name = "Differential interaction strength"

    arr = net_diff.copy()
    arr = _filter_by_sources_targets(arr, levels, sources_use, targets_use)
    if remove_isolate:
        arr, levels = _remove_isolate(arr, levels)

    thresh = _abs_quantile_threshold(arr, top)
    arr = np.where(np.abs(arr) < thresh, 0.0, arr)

    n = len(levels)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_title(title_name)
        return fig

    if color_use is None:
        color_use = list(scPalette(n))
    else:
        color_use = list(color_use)[:n] if len(color_use) >= n else list(color_use) + list(scPalette(n - len(color_use)))

    coords = _circle_layout(n)
    coords_scale = _scale_coords(coords) if n > 1 else coords

    if vertex_weight_max is None:
        vw_max = float(np.max(vertex_weight))
    else:
        vw_max = float(vertex_weight_max)
    if hasattr(vertex_weight, "__len__"):
        vw = np.asarray(vertex_weight, dtype=float) / vw_max * float(vertex_size_max) + 5.0
    else:
        vw = np.full(n, float(vertex_weight) / vw_max * float(vertex_size_max) + 5.0)

    # Edge colour: red for positive, blue for negative
    fig, ax = plt.subplots(figsize=figsize)
    label_locs = _radian_rescale(n, start=0.0, direction=-1)
    label_dist = vw / vw.max() + 2.0 if vw.size > 0 else np.full(n, 2.0)

    rows, cols = np.nonzero(arr)
    weights = arr[rows, cols]
    abs_max = float(np.abs(arr).max()) if arr.size > 0 and np.abs(arr).max() > 0 else 1.0
    if edge_weight_max is None:
        edge_weight_max = abs_max

    for r, c, w in zip(rows, cols, weights):
        absw = abs(w)
        if weight_scale:
            ew = 0.3 + (absw / edge_weight_max) * edge_width_max
        else:
            ew = 0.3 + edge_width_max * absw
        edge_col = color_edge[0] if w > 0 else color_edge[1]
        rgb = mcolors.to_rgb(edge_col)
        edge_color = (rgb[0], rgb[1], rgb[2], float(alpha_edge))
        p1 = (float(coords_scale[r, 0]), float(coords_scale[r, 1]))
        p2 = (float(coords_scale[c, 0]), float(coords_scale[c, 1]))
        if r == c:
            cx, cy = p1
            radius = 0.05 + 0.05 * (absw / max(edge_weight_max, 1e-12))
            loop = mpatches.Circle(
                (cx + radius, cy + radius), radius=radius,
                fill=False, edgecolor=edge_color, linewidth=ew,
            )
            ax.add_patch(loop)
        else:
            _draw_curved_edge(
                ax, p1, p2, width=float(ew), color=edge_color,
                curvature=float(edge_curved),
                arrow=True, arrow_size=arrow_size, arrow_width=arrow_width,
            )
    # vertices and labels
    for i, name in enumerate(levels):
        ax.scatter(
            coords_scale[i, 0], coords_scale[i, 1],
            s=(vw[i] * 8.0) ** 2 / 4.0,
            c=[color_use[i]], edgecolors=color_use[i], linewidths=1.0, zorder=3,
        )
        if vertex_label_cex > 0 and name:
            ang = float(label_locs[i])
            lx = coords_scale[i, 0] + label_dist[i] * 0.04 * math.cos(ang)
            ly = coords_scale[i, 1] + label_dist[i] * 0.04 * math.sin(ang)
            ax.text(
                lx, ly, name, ha="center", va="center",
                fontsize=10.0 * float(vertex_label_cex), color=vertex_label_color, zorder=4,
            )
    if title_name:
        ax.set_title(title_name, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    xs = coords_scale[:, 0]
    ys = coords_scale[:, 1]
    rng = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    pad = (1.0 + float(margin)) * rng / 2.0
    ax.set_xlim(cx - pad, cx + pad)
    ax.set_ylim(cy - pad, cy + pad)
    return fig


# ---------------------------------------------------------------------------
# Hierarchy plots (R: visualization.R:1610, 1800)
# ---------------------------------------------------------------------------
def _hierarchy_plot(
    *,
    side: int,  # 1 = "hierarchy1", 2 = "hierarchy2"
    net: Union[np.ndarray, pd.DataFrame],
    vertex_receiver: Sequence[int],
    color_use: Optional[Sequence[str]],
    title_name: Optional[str],
    sources_use: Optional[Sequence],
    targets_use: Optional[Sequence],
    remove_isolate: bool,
    top: float,
    weight_scale: bool,
    vertex_weight: Union[float, Sequence[float]],
    vertex_weight_max: Optional[float],
    vertex_size_max: Optional[float],
    edge_weight_max: Optional[float],
    edge_width_max: float,
    alpha_edge: float,
    label_dist: float,
    space_v: float,
    space_h: float,
    label_edge: bool,
    edge_curved: float,
    margin: float,
    vertex_label_cex: float,
    vertex_label_color: str,
    arrow_width: float,
    arrow_size: float,
    edge_label_color: str,
    edge_label_cex: float,
    figsize: Tuple[float, float],
) -> Figure:
    """Internal: shared logic for hierarchy1/hierarchy2."""
    if vertex_size_max is None:
        if hasattr(vertex_weight, "__len__") and len(set(np.atleast_1d(vertex_weight).tolist())) == 1:
            vertex_size_max = 5.0
        elif not hasattr(vertex_weight, "__len__"):
            vertex_size_max = 5.0
        else:
            vertex_size_max = 15.0

    arr, levels = _ensure_named_matrix(net)
    thresh = _quantile_threshold(arr, top)
    arr = np.where(arr < thresh, 0.0, arr)
    arr = _filter_by_sources_targets(arr, levels, sources_use, targets_use)
    arr = np.where(np.isnan(arr), 0.0, arr)
    if remove_isolate:
        arr, levels = _remove_isolate(arr, levels)
    n = len(levels)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        if title_name is not None:
            ax.set_title(title_name)
        return fig

    if color_use is None:
        color_use = list(scPalette(n))
    else:
        color_use = list(color_use)[:n] if len(color_use) >= n else list(color_use) + list(scPalette(n - len(color_use)))

    if vertex_weight_max is None:
        vw_max = float(np.max(vertex_weight))
    else:
        vw_max = float(vertex_weight_max)
    if hasattr(vertex_weight, "__len__"):
        vw = np.asarray(vertex_weight, dtype=float) / vw_max * float(vertex_size_max) + 6.0
    else:
        vw = np.full(n, float(vertex_weight) / vw_max * float(vertex_size_max) + 6.0)

    # Convert vertex_receiver to 0-based indices
    vr_set = set(int(v) - 1 for v in vertex_receiver)
    vr_idx = sorted(vr_set)
    other_idx = [i for i in range(n) if i not in vr_set]

    if side == 1:
        # hierarchy1: m = len(vr); reorder.row = c(vr, others)
        m = len(vr_idx)
        reorder_row = vr_idx + other_idx
        col_idx = vr_idx  # cols = vr
        # left column = vr (m vertices), right column = others (m1-m), targets = m
        net2 = arr[np.ix_(reorder_row, col_idx)]
        m1 = net2.shape[0]
        n1 = net2.shape[1]
        # Build expanded matrix: rbind(cbind(zeros(m1,m1), net2), zeros(n1, m1+n1))
        net3 = np.zeros((m1 + n1, m1 + n1), dtype=float)
        net3[:m1, m1:] = net2
        # row.names: c(vr names, others names, rep("", m))
        names_left = [levels[i] for i in vr_idx]
        names_right = [levels[i] for i in other_idx]
        names_target = [""] * m
        levels_new = names_left + names_right + names_target
        # color.use3
        color_use3 = (
            [color_use[i] for i in vr_idx]
            + [color_use[i] for i in other_idx]
            + ["#FFFFFF"] * len(vr_idx)
        )
        color_use3_frame = (
            [color_use[i] for i in vr_idx]
            + [color_use[i] for i in other_idx]
            + [color_use[i] for i in vr_idx]
        )
        # Coordinates
        coords = np.full((m1 + n1, 2), np.nan)
        if m > 0:
            coords[:m, 0] = 0.0
            if m == 1:
                coords[:m, 1] = space_v
            else:
                coords[:m, 1] = np.linspace(space_v, 0.0, m)
        if m1 > m:
            coords[m:m1, 0] = space_h
            if m1 - m == 1:
                coords[m:m1, 1] = space_v
            else:
                coords[m:m1, 1] = np.linspace(space_v, 0.0, m1 - m)
        if n1 > 0:
            coords[m1:, 0] = space_h / 2.0
            if n1 == 1:
                coords[m1:, 1] = space_v
            else:
                coords[m1:, 1] = np.linspace(space_v, 0.0, n1)
        if hasattr(vertex_weight, "__len__"):
            vw_full = np.concatenate(
                [vw[vr_idx], vw[other_idx], vw[vr_idx]]
            )
        else:
            vw_full = np.full(m1 + n1, vw[0])
    else:
        # hierarchy2: reorder.row = c(others, vr); cols = vr
        m = len(vr_idx)
        m0 = n - m
        reorder_row = other_idx + vr_idx
        col_idx = vr_idx
        net2 = arr[np.ix_(reorder_row, col_idx)]
        m1 = net2.shape[0]
        n1 = net2.shape[1]
        net3 = np.zeros((m1 + n1, m1 + n1), dtype=float)
        net3[:m1, m1:] = net2
        names_left = [levels[i] for i in other_idx]
        names_right = [levels[i] for i in vr_idx]
        names_target = [""] * m
        levels_new = names_left + names_right + names_target
        color_use3 = (
            [color_use[i] for i in other_idx]
            + [color_use[i] for i in vr_idx]
            + ["#FFFFFF"] * len(vr_idx)
        )
        color_use3_frame = (
            [color_use[i] for i in other_idx]
            + [color_use[i] for i in vr_idx]
            + [color_use[i] for i in vr_idx]
        )
        coords = np.full((m1 + n1, 2), np.nan)
        if m0 > 0:
            coords[:m0, 0] = 0.0
            if m0 == 1:
                coords[:m0, 1] = space_v
            else:
                coords[:m0, 1] = np.linspace(space_v, 0.0, m0)
        if m1 > m0:
            coords[m0:m1, 0] = space_h
            if m1 - m0 == 1:
                coords[m0:m1, 1] = space_v
            else:
                coords[m0:m1, 1] = np.linspace(space_v, 0.0, m1 - m0)
        if n1 > 0:
            coords[m1:, 0] = space_h / 2.0
            if n1 == 1:
                coords[m1:, 1] = space_v
            else:
                coords[m1:, 1] = np.linspace(space_v, 0.0, n1)
        if hasattr(vertex_weight, "__len__"):
            vw_full = np.concatenate([vw[other_idx], vw[vr_idx], vw[vr_idx]])
        else:
            vw_full = np.full(m1 + n1, vw[0])

    if edge_weight_max is None:
        edge_weight_max = float(net3.max()) if net3.size > 0 and net3.max() > 0 else 1.0

    # R parity (visualization.R:1722):
    # ``label.dist <- c(rep(space.h*label.dist, m1), rep(0, nrow(net3) - m1))``.
    # First m1 vertices (left + right cell-group columns) get the same
    # offset; the remaining vertices belong to the central target column
    # whose labels render in-place (no offset).
    label_dist_v = np.concatenate([
        np.full(m1, space_h * label_dist),
        np.zeros(net3.shape[0] - m1),
    ])

    edge_labels: Optional[List[str]] = None
    if label_edge:
        rows, cols = np.nonzero(net3)
        edge_labels = [f"{round(float(net3[r, c]), 1)}" for r, c in zip(rows, cols)]

    fig, ax = plt.subplots(figsize=figsize)
    rows, cols = np.nonzero(net3)
    weights = net3[rows, cols]
    for r, c, w in zip(rows, cols, weights):
        if weight_scale:
            ew = 0.3 + (w / edge_weight_max) * edge_width_max
        else:
            ew = 0.3 + edge_width_max * w
        col_i = mcolors.to_rgb(color_use3[r])
        edge_color = (col_i[0], col_i[1], col_i[2], float(alpha_edge))
        p1 = (float(coords[r, 0]), float(coords[r, 1]))
        p2 = (float(coords[c, 0]), float(coords[c, 1]))
        _draw_curved_edge(
            ax, p1, p2, width=float(ew), color=edge_color,
            curvature=float(edge_curved),
            arrow=True, arrow_size=float(arrow_size), arrow_width=float(arrow_width),
        )

    # Vertices — R parity (visualization.R:1727):
    # ``vertex.frame.width = c(rep(1, m1), rep(2, nrow(net3) - m1))`` —
    # the duplicated source-on-right column gets a thicker frame.
    for i, name in enumerate(levels_new):
        frame_lw = 2.0 if i >= m1 else 1.0
        ax.scatter(
            coords[i, 0], coords[i, 1],
            s=(vw_full[i] * 8.0) ** 2 / 4.0,
            c=[color_use3[i]], edgecolors=color_use3_frame[i],
            linewidths=frame_lw, zorder=3,
        )
        if vertex_label_cex > 0 and name:
            # R parity (visualization.R:1722): per-vertex label distance is
            # ``space.h * label.dist`` for the cell-group columns (left + right
            # source) and ``0`` for the central target column. Use that as
            # the x-offset; right-aligned for left/right-source, left-aligned
            # for the target column.
            ld = float(label_dist_v[i])
            if i < (m if side == 1 else m0):
                tx = coords[i, 0] - ld
                ha = "right"
            elif i < m1:
                tx = coords[i, 0] + ld
                ha = "left"
            else:
                tx = coords[i, 0] - max(ld, 0.15)
                ha = "right"
            ax.text(
                tx, coords[i, 1], name, ha=ha, va="center",
                fontsize=10.0 * float(vertex_label_cex), color=vertex_label_color, zorder=4,
            )

    if edge_labels is not None:
        for r, c, lbl in zip(rows, cols, edge_labels):
            mx = (coords[r, 0] + coords[c, 0]) / 2.0
            my = (coords[r, 1] + coords[c, 1]) / 2.0
            ax.text(
                mx, my, lbl, fontsize=10.0 * float(edge_label_cex),
                color=edge_label_color, zorder=5,
            )

    # Source/Target column header text (R lines 1729, 1920)
    if side == 1:
        head_colors = ("#c51b7d", "#c51b7d", "#2f6661")
    else:
        head_colors = ("#c51b7d", "#2f6661", "#2f6661")
    head_y = space_v - space_v / 7.0
    ax.text(-space_h / 1.5, head_y, "Source", fontsize=8, color=head_colors[0], ha="center")
    ax.text(space_h / 22.0, head_y, "Target", fontsize=8, color=head_colors[1], ha="center")
    ax.text(space_h / 1.5, head_y, "Source", fontsize=8, color=head_colors[2], ha="center")

    # R parity (visualization.R:1730-1733): two arrow indicators between
    # the column headers showing source → target flow direction. R uses
    # ``shape::Arrows`` with a triangle head; matplotlib's ``annotate``
    # with ``arrowstyle='->,head_length=...'`` is the equivalent.
    arrow_y = space_v - space_v / 4.0
    ax.annotate(
        "",
        xy=(space_h / 100000.0, arrow_y),
        xytext=(-space_h / 1.5, arrow_y),
        arrowprops=dict(arrowstyle="->", color="#c51b7d", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(space_h / 20.0, arrow_y),
        xytext=(space_h / 1.5, arrow_y),
        arrowprops=dict(arrowstyle="->", color="#2f6661", lw=0.8),
    )

    if title_name:
        ax.set_title(f"{title_name} signaling network", fontsize=11)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlim(-space_h, 2.0 * space_h)
    ax.set_ylim(-space_v * 0.2, space_v * 1.2)
    return fig


def _replay_axes(src_ax: "plt.Axes", dest_ax: "plt.Axes") -> None:
    """Copy artists from ``src_ax`` to ``dest_ax`` (best-effort)."""
    if src_ax is None:
        dest_ax.set_axis_off()
        return
    for ln in src_ax.lines:
        dest_ax.add_line(mlines.Line2D(
            ln.get_xdata(), ln.get_ydata(),
            color=ln.get_color(), linewidth=ln.get_linewidth(),
            linestyle=ln.get_linestyle(), alpha=ln.get_alpha(),
        ))
    for patch in src_ax.patches:
        if hasattr(patch, "get_path"):
            p_copy = mpatches.PathPatch(
                patch.get_path(),
                facecolor=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                alpha=patch.get_alpha(),
            )
            dest_ax.add_patch(p_copy)
    for coll in src_ax.collections:
        offsets = coll.get_offsets()
        sizes = coll.get_sizes() if hasattr(coll, "get_sizes") else None
        faces = coll.get_facecolors()
        edges = coll.get_edgecolors()
        dest_ax.scatter(
            offsets[:, 0], offsets[:, 1],
            s=sizes if sizes is not None else 50,
            c=faces if len(faces) > 0 else None,
            edgecolors=edges if len(edges) > 0 else None,
            zorder=coll.get_zorder(),
        )
    for txt in src_ax.texts:
        dest_ax.text(
            txt.get_position()[0], txt.get_position()[1], txt.get_text(),
            fontsize=txt.get_fontsize(), color=txt.get_color(),
            ha=txt.get_horizontalalignment(),
            va=txt.get_verticalalignment(),
            rotation=txt.get_rotation(), zorder=txt.get_zorder(),
        )
    dest_ax.set_xlim(src_ax.get_xlim())
    dest_ax.set_ylim(src_ax.get_ylim())
    dest_ax.set_aspect("equal")
    dest_ax.set_xticks([])
    dest_ax.set_yticks([])
    for sp in dest_ax.spines.values():
        sp.set_visible(False)
    if src_ax.get_title():
        dest_ax.set_title(src_ax.get_title(), fontsize=10)


def _hierarchy_combined(
    *,
    net: Union[np.ndarray, pd.DataFrame],
    vertex_receiver: Sequence[int],
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    alpha_edge: float = 0.6,
    label_dist: float = 2.8,
    space_v: float = 1.5,
    space_h: float = 1.6,
    label_edge: bool = False,
    edge_curved: float = 0.0,
    margin: float = 0.2,
    vertex_label_cex: float = 0.6,
    vertex_label_color: str = "black",
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.5,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> Figure:
    """Render hierarchy1 + hierarchy2 side-by-side as a single composite Figure.

    Mirrors R: under ``layout='hierarchy'`` ``netVisual`` /
    ``netVisual_individual`` render BOTH hierarchy panels per call.
    """
    f1 = _hierarchy_plot(
        side=1, net=net, vertex_receiver=vertex_receiver, color_use=color_use,
        title_name=title_name, sources_use=sources_use, targets_use=targets_use,
        remove_isolate=remove_isolate, top=top, weight_scale=weight_scale,
        vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
        vertex_size_max=vertex_size_max, edge_weight_max=edge_weight_max,
        edge_width_max=edge_width_max, alpha_edge=alpha_edge,
        label_dist=label_dist, space_v=space_v, space_h=space_h,
        label_edge=label_edge, edge_curved=edge_curved, margin=margin,
        vertex_label_cex=vertex_label_cex, vertex_label_color=vertex_label_color,
        arrow_width=arrow_width, arrow_size=arrow_size,
        edge_label_color=edge_label_color, edge_label_cex=edge_label_cex,
        figsize=(figsize[0] / 2.0, figsize[1]),
    )
    f2 = _hierarchy_plot(
        side=2, net=net, vertex_receiver=vertex_receiver, color_use=color_use,
        title_name=title_name, sources_use=sources_use, targets_use=targets_use,
        remove_isolate=remove_isolate, top=top, weight_scale=weight_scale,
        vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
        vertex_size_max=vertex_size_max, edge_weight_max=edge_weight_max,
        edge_width_max=edge_width_max, alpha_edge=alpha_edge,
        label_dist=label_dist, space_v=space_v, space_h=space_h,
        label_edge=label_edge, edge_curved=edge_curved, margin=margin,
        vertex_label_cex=vertex_label_cex, vertex_label_color=vertex_label_color,
        arrow_width=arrow_width, arrow_size=arrow_size,
        edge_label_color=edge_label_color, edge_label_cex=edge_label_cex,
        figsize=(figsize[0] / 2.0, figsize[1]),
    )

    # Composite the two figures into a single 1x2 figure.
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    _replay_axes(f1.axes[0] if f1.axes else None, ax1)
    _replay_axes(f2.axes[0] if f2.axes else None, ax2)
    plt.close(f1)
    plt.close(f2)
    if title_name:
        fig.suptitle(title_name, fontsize=12)
    return fig


def netVisual_hierarchy1(
    net: Union[np.ndarray, pd.DataFrame],
    vertex_receiver: Sequence[int],
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    alpha_edge: float = 0.6,
    label_dist: float = 2.8,
    space_v: float = 1.5,
    space_h: float = 1.6,
    shape: Optional[str] = None,
    label_edge: bool = False,
    edge_curved: float = 0.0,
    margin: float = 0.2,
    vertex_label_cex: float = 0.6,
    vertex_label_color: str = "black",
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.5,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> Figure:
    """Hierarchy plot (sources -> ``vertex_receiver`` targets).

    Direct port of R's :func:`netVisual_hierarchy1` (``visualization.R:1610``).

    Parameters
    ----------
    net : ndarray or DataFrame
        Square weighted matrix.
    vertex_receiver : sequence of int
        1-based indices of cell groups acting as targets in the first
        hierarchy plot.
    Notes
    -----
    See the R reference (``SpatialCellChat::netVisual_*``) for all other parameters.
    """
    return _hierarchy_plot(
        side=1, net=net, vertex_receiver=vertex_receiver, color_use=color_use,
        title_name=title_name, sources_use=sources_use, targets_use=targets_use,
        remove_isolate=remove_isolate, top=top, weight_scale=weight_scale,
        vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
        vertex_size_max=vertex_size_max, edge_weight_max=edge_weight_max,
        edge_width_max=edge_width_max, alpha_edge=alpha_edge,
        label_dist=label_dist, space_v=space_v, space_h=space_h,
        label_edge=label_edge, edge_curved=edge_curved, margin=margin,
        vertex_label_cex=vertex_label_cex, vertex_label_color=vertex_label_color,
        arrow_width=arrow_width, arrow_size=arrow_size,
        edge_label_color=edge_label_color, edge_label_cex=edge_label_cex,
        figsize=figsize,
    )


def netVisual_hierarchy2(
    net: Union[np.ndarray, pd.DataFrame],
    vertex_receiver: Sequence[int],
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    alpha_edge: float = 0.6,
    label_dist: float = 2.8,
    space_v: float = 1.5,
    space_h: float = 1.6,
    shape: Optional[str] = None,
    label_edge: bool = False,
    edge_curved: float = 0.0,
    margin: float = 0.2,
    vertex_label_cex: float = 0.6,
    vertex_label_color: str = "black",
    arrow_width: float = 1.0,
    arrow_size: float = 0.2,
    edge_label_color: str = "black",
    edge_label_cex: float = 0.5,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> Figure:
    """Hierarchy plot (sources -> non-``vertex_receiver`` targets).

    Direct port of R's :func:`netVisual_hierarchy2` (``visualization.R:1800``).
    """
    return _hierarchy_plot(
        side=2, net=net, vertex_receiver=vertex_receiver, color_use=color_use,
        title_name=title_name, sources_use=sources_use, targets_use=targets_use,
        remove_isolate=remove_isolate, top=top, weight_scale=weight_scale,
        vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
        vertex_size_max=vertex_size_max, edge_weight_max=edge_weight_max,
        edge_width_max=edge_width_max, alpha_edge=alpha_edge,
        label_dist=label_dist, space_v=space_v, space_h=space_h,
        label_edge=label_edge, edge_curved=edge_curved, margin=margin,
        vertex_label_cex=vertex_label_cex, vertex_label_color=vertex_label_color,
        arrow_width=arrow_width, arrow_size=arrow_size,
        edge_label_color=edge_label_color, edge_label_cex=edge_label_cex,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# netVisual_spatial (R: visualization.R:2195)
# ---------------------------------------------------------------------------
def netVisual_spatial(
    net: Union[np.ndarray, pd.DataFrame],
    coordinates: Union[np.ndarray, pd.DataFrame],
    labels: Sequence,
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    idents_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    remove_loop: bool = True,
    top: float = 1.0,
    weight_scale: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 20,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    vertex_label_cex: float = 3.0,
    vertex_label_color: str = "black",
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    edge_curved: float = 0.2,
    alpha_edge: float = 0.6,
    arrow_angle: float = 5.0,
    arrow_size: float = 0.2,
    alpha_image: float = 0.15,
    point_size: float = 1.0,
    legend_size: float = 3.0,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Figure:
    """Spatial cell-cell communication network.

    Direct port of R's :func:`netVisual_spatial` (``visualization.R:2195``).

    Parameters
    ----------
    net : ndarray or DataFrame
        Square weighted matrix.
    coordinates : ndarray or DataFrame
        ``N × 2`` matrix of spot coordinates.
    labels : sequence
        Group labels for each spot (length ``N``).
    Notes
    -----
    See the R reference (``SpatialCellChat::netVisual_*``) for all other parameters.
    """
    arr, levels = _ensure_named_matrix(net)
    coords_arr = np.asarray(coordinates, dtype=float)
    if coords_arr.shape[1] != 2:
        raise ValueError(
            "Please check the input 'coordinates' and make sure it is a "
            "two column matrix."
        )
    # R swaps columns because R ingests ``object@images$coordinates`` as
    # ``(imagerow, imagecol)`` from Seurat ``GetTissueCoordinates`` and
    # re-labels to ``(x_cent=col, y_cent=row)``. Python receives
    # ``adata.obsm['spatial']`` already in scanpy/squidpy convention
    # ``(pxl_col_in_fullres, pxl_row_in_fullres) = (x, y)`` — no swap is
    # needed (canonical convention pinned by ``viz_spatial._xy_from_adata``).
    # Audit M-VIZ-6 flagged the previous swap for misaligning vertices
    # against the rest of the package.
    labels_arr = np.asarray(labels, dtype=object)

    num_cluster = len(levels)
    node_coords = np.zeros((num_cluster, 2), dtype=float)
    for i, lvl in enumerate(levels):
        mask = labels_arr.astype(str) == str(lvl)
        if mask.any():
            node_coords[i, 0] = float(np.median(coords_arr[mask, 0]))
            node_coords[i, 1] = float(np.median(coords_arr[mask, 1]))
        else:
            node_coords[i, 0] = float(np.median(coords_arr[:, 0]))
            node_coords[i, 1] = float(np.median(coords_arr[:, 1]))

    if vertex_size_max is None:
        if hasattr(vertex_weight, "__len__") and len(set(np.atleast_1d(vertex_weight).tolist())) == 1:
            vertex_size_max = 5.0
        elif not hasattr(vertex_weight, "__len__"):
            vertex_size_max = 5.0
        else:
            vertex_size_max = 15.0

    thresh = _quantile_threshold(arr, top)
    arr = np.where(arr < thresh, 0.0, arr)
    arr = _filter_by_sources_targets(arr, levels, sources_use, targets_use, idents_use)
    arr = np.where(np.isnan(arr), 0.0, arr)

    if remove_loop:
        np.fill_diagonal(arr, 0.0)
    if remove_isolate:
        arr, kept_levels = _remove_isolate(arr, levels)
        keep_mask = np.array([lv in set(kept_levels) for lv in levels], dtype=bool)
        node_coords = node_coords[keep_mask]
        levels = kept_levels

    n = len(levels)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        if title_name is not None:
            ax.set_title(title_name)
        return fig

    if color_use is None:
        color_use = list(scPalette(n))
    else:
        color_use = list(color_use)[:n] if len(color_use) >= n else list(color_use) + list(scPalette(n - len(color_use)))
    color_map = {lv: color_use[i] for i, lv in enumerate(levels)}

    if vertex_weight_max is None:
        vw_max = float(np.max(vertex_weight))
    else:
        vw_max = float(vertex_weight_max)
    if hasattr(vertex_weight, "__len__"):
        vw = np.asarray(vertex_weight, dtype=float) / vw_max * float(vertex_size_max) + 1.0
    else:
        vw = np.full(n, float(vertex_weight) / vw_max * float(vertex_size_max) + 1.0)

    if edge_weight_max is None:
        edge_weight_max = float(arr.max()) if arr.size > 0 and arr.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # background spots
    for i in range(coords_arr.shape[0]):
        lab = str(labels_arr[i])
        col = color_map.get(lab, "#888888")
        ax.scatter(
            coords_arr[i, 0], coords_arr[i, 1],
            s=(point_size * 6.0) ** 2 / 4.0, c=[col], alpha=alpha_image,
            edgecolors="none", zorder=1,
        )

    # edges. R keeps self-loops (the loop_curve removal block at
    # visualization.R:1556-1562 is commented out): geom_curve renders a
    # small arc when X1==X2 and Y1==Y2. We mirror this with a tiny
    # circular arc patch above the node centre. Honour `remove_loop=True`
    # only via the diagonal-zeroing performed earlier.
    rows, cols = np.nonzero(arr)
    for r, c in zip(rows, cols):
        w = arr[r, c]
        if weight_scale:
            ew = 0.3 + (w / edge_weight_max) * edge_width_max
        else:
            ew = 0.3 + edge_width_max * w
        col_i = mcolors.to_rgb(color_use[r])
        edge_color = (col_i[0], col_i[1], col_i[2], float(alpha_edge))
        p1 = (float(node_coords[r, 0]), float(node_coords[r, 1]))
        p2 = (float(node_coords[c, 0]), float(node_coords[c, 1]))
        if r == c:
            # Draw a small self-loop arc just above the node
            radius = max(float(vw[r]) * 0.3, 1e-3)
            cx, cy = p1
            loop_centre = (cx, cy + radius)
            arc = matplotlib.patches.Arc(
                xy=loop_centre, width=2 * radius, height=2 * radius,
                angle=0.0, theta1=0.0, theta2=360.0,
                color=edge_color, linewidth=float(ew),
            )
            ax.add_patch(arc)
            continue
        _draw_curved_edge(
            ax, p1, p2, width=float(ew), color=edge_color,
            curvature=float(edge_curved),
            arrow=True, arrow_size=arrow_size, arrow_width=1.0,
        )

    # vertices (centroids)
    for i, name in enumerate(levels):
        ax.scatter(
            node_coords[i, 0], node_coords[i, 1],
            s=(vw[i] * 8.0) ** 2 / 4.0,
            c=[color_use[i]], edgecolors=color_use[i], linewidths=1.0, zorder=3,
            label=name,
        )
        if vertex_label_cex > 0 and name:
            ax.text(
                node_coords[i, 0], node_coords[i, 1] + vw[i] * 0.5,
                name, ha="center", va="bottom",
                fontsize=float(vertex_label_cex) * 3.0, color=vertex_label_color, zorder=4,
            )

    ax.invert_yaxis()  # R: scale_y_reverse()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title_name:
        ax.set_title(title_name, fontsize=11)
    ax.legend(loc="best", fontsize=int(legend_size * 3))
    return fig


# ---------------------------------------------------------------------------
# Chord exports (R: visualization.R:3262, 3382, 3553)
# ---------------------------------------------------------------------------
def _chord_diagram(
    df: pd.DataFrame,
    cell_levels: List[str],
    color_use: Dict[str, str],
    *,
    transparency: float,
    title_name: Optional[str],
    figsize: Tuple[float, float] = (7.0, 7.0),
    show_legend: bool = False,
    lab_cex: float = 0.8,
    directional: int = 1,
    link_target_prop: bool = True,
) -> Figure:
    """Render a chord diagram via :mod:`pycirclize`.

    Parameters
    ----------
    df : DataFrame
        Three-column df: ``source``, ``target``, ``prob``.
    cell_levels : list of str
        Sector order.
    color_use : dict
        Mapping ``cell_level -> hex colour``.
    transparency : float
        R parity: ``transparency`` (alpha = 1 - transparency).
    directional : int
        R parity: ``directional`` ∈ {-1, 0, 1, 2}. Controls link coloring:
        ``1`` (default) → color by SOURCE (flow ``source → target``);
        ``-1`` → color by TARGET (flow ``target → source``);
        ``0`` → neutral grey (no direction);
        ``2`` → bidirectional, falls back to source-coloring with a
        runtime warning because pycirclize cannot draw bi-directional
        gradients in :func:`Circos.chord_diagram`.
    link_target_prop : bool
        R parity: ``link.target.prop``. circlize draws thin proportion bars
        at each source sector when ``True``. pycirclize does not support
        these mini-bars; chord widths already encode flow magnitude. When
        ``False`` (non-default) is passed, we emit an explicit warning
        rather than silently ignore the request — the visual outcome is
        unchanged either way under pycirclize.
    """
    # Build adjacency matrix in the order of cell_levels
    K = len(cell_levels)
    matrix = np.zeros((K, K), dtype=float)
    idx = {lv: i for i, lv in enumerate(cell_levels)}
    for _, row in df.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        p = float(row["prob"])
        if s in idx and t in idx and p > 0:
            matrix[idx[s], idx[t]] += p

    from pycirclize import Circos

    if matrix.sum() <= 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        if title_name:
            ax.set_title(title_name)
        return fig

    df_mat = pd.DataFrame(matrix, index=cell_levels, columns=cell_levels)
    sector_colors = {lv: color_use.get(lv, "#888888") for lv in cell_levels}

    # R parity for ``directional``: pick the colour source per-link.
    if directional == 2:
        warnings.warn(
            "pycirclize cannot render bidirectional chord links with a "
            "two-colour gradient; falling back to source-colour rendering. "
            "If you need the dual-colour effect, use circlize via R or a "
            "richer chord library.",
            stacklevel=2,
        )
        coloured_by_source = True
    elif directional == -1:
        coloured_by_source = False
    elif directional == 0:
        coloured_by_source = None  # neutral grey
    else:
        coloured_by_source = True

    if not link_target_prop:
        warnings.warn(
            "`link_target_prop=False` is not honoured by pycirclize; the "
            "circlize-style per-source proportion mini-bars are unavailable. "
            "Chord widths already encode source→target flow magnitude.",
            stacklevel=2,
        )

    def link_kws_handler(row_label: str, col_label: str) -> Dict[str, Any]:
        if coloured_by_source is True:
            fc = sector_colors[row_label]
        elif coloured_by_source is False:
            fc = sector_colors[col_label]
        else:  # directional == 0 → neutral grey
            fc = "#7f7f7f"
        return {"fc": fc, "alpha": 1.0 - float(transparency)}

    circos = Circos.chord_diagram(
        df_mat,
        space=3.0,
        cmap=sector_colors,
        label_kws=dict(size=int(lab_cex * 10)),
        link_kws_handler=link_kws_handler,
        ticks_interval=None,
    )
    fig = circos.plotfig(figsize=figsize)
    if title_name:
        fig.suptitle(title_name, fontsize=11)
    if show_legend:
        ax = fig.gca()
        handles = [
            mpatches.Patch(color=sector_colors[lv], label=lv) for lv in cell_levels
        ]
        ax.legend(handles=handles, loc="lower right", title="Cell State", fontsize=8)
    return fig


def netVisual_chord_cell_internal(
    net: Union[np.ndarray, pd.DataFrame],
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    group: Optional[Dict[str, str]] = None,
    cell_order: Optional[Sequence[str]] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    lab_cex: float = 0.8,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    annotation_track_height: float = 0.03,
    remove_isolate: bool = False,
    link_visible: bool = True,
    scale: bool = False,
    directional: int = 1,
    link_target_prop: bool = True,
    reduce: float = -1.0,
    transparency: float = 0.4,
    link_border: Optional[str] = None,
    title_name: Optional[str] = None,
    show_legend: bool = False,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Figure:
    """Chord diagram of cell-cell communication (internal).

    Direct port of R's :func:`netVisual_chord_cell_internal`
    (``visualization.R:3382``). Uses :mod:`pycirclize` in lieu of
    :func:`circlize::chordDiagram`.
    """
    if isinstance(net, pd.DataFrame) and {"source", "target", "prob"}.issubset(net.columns):
        df = net[["source", "target", "prob"]].copy()
        cell_levels = list(pd.unique(pd.concat([df["source"], df["target"]]).astype(str)))
    else:
        arr, levels = _ensure_named_matrix(net)
        cell_levels = list(set(levels))
        cell_levels = list(levels)
        K = len(levels)
        df = pd.DataFrame(
            {
                "source": np.repeat(levels, K),
                "target": np.tile(levels, K),
                "prob": arr.ravel(order="C"),
            }
        )

    if cell_order is not None:
        cell_levels = list(cell_order)

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    if sources_use is not None:
        sources_use = _resolve_indices(list(sources_use), cell_levels)
        df = df[df["source"].isin(sources_use)]
    if targets_use is not None:
        targets_use = _resolve_indices(list(targets_use), cell_levels)
        df = df[df["target"].isin(targets_use)]
    df = df[df["prob"] > 0]

    if not remove_isolate:
        cells_present = set(df["source"]).union(df["target"])
        cells_removed = [lv for lv in cell_levels if lv not in cells_present]
        if len(cells_removed) > 0:
            # Add "fake" entries with negligible prob to keep sectors visible
            fake = pd.DataFrame({
                "source": cells_removed,
                "target": cells_removed,
                "prob": [1e-10 * (i + 1) for i in range(len(cells_removed))],
            })
            df = pd.concat([df, fake], ignore_index=True)
            scale = True

    cells_use = set(df["source"]).union(df["target"])
    order_sector = [lv for lv in cell_levels if lv in cells_use]

    if isinstance(color_use, dict):
        color_use_d = dict(color_use)
    else:
        if color_use is None:
            cu = list(scPalette(len(cell_levels)))
        else:
            cu = list(color_use)
        color_use_d = {lv: cu[i % len(cu)] for i, lv in enumerate(cell_levels)}

    return _chord_diagram(
        df, order_sector, color_use_d,
        transparency=transparency, title_name=title_name,
        figsize=figsize, show_legend=show_legend, lab_cex=lab_cex,
        directional=directional, link_target_prop=link_target_prop,
    )


def netVisual_chord_cell(
    adata: ad.AnnData,
    signaling: Optional[Union[str, Sequence[str]]] = None,
    net: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    slot_name: str = "netP",
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    group: Optional[Dict[str, str]] = None,
    cell_order: Optional[Sequence[str]] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    lab_cex: float = 0.8,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    annotation_track_height: float = 0.03,
    remove_isolate: bool = False,
    link_visible: bool = True,
    scale: bool = False,
    directional: int = 1,
    link_target_prop: bool = True,
    reduce: float = -1.0,
    transparency: float = 0.4,
    link_border: Optional[str] = None,
    title_name: Optional[str] = None,
    show_legend: bool = False,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
    n_col: Optional[int] = None,
    thresh: float = 0.05,
    figsize: Tuple[float, float] = (7.0, 7.0),
) -> Union[Figure, List[Figure]]:
    """Chord diagram for a signaling pathway or arbitrary ``net``.

    Direct port of R's :func:`netVisual_chord_cell` (``visualization.R:3262``).
    """
    if signaling is not None:
        cc = _ensure_uns(adata)
        LRsig = cc["LR"]["LRsig"]
        if isinstance(signaling, str):
            signaling_list = [signaling]
        else:
            signaling_list = list(signaling)
        # match all pathway names; for each signaling we run searchPair
        pairLR_all = pd.concat(
            [
                searchPair(s, LRsig, key="pathway_name", matching_exact=True, pair_only=False)
                for s in signaling_list
            ],
            ignore_index=False,
        )
        pairLR = pairLR_all
        net_dict = cc["net"]
        prob = np.asarray(net_dict["prob"], dtype=float).copy()
        pval_raw = net_dict.get("pval", None)
        if pval_raw is None:
            pval = np.ones_like(prob)
        else:
            pval = np.asarray(pval_raw, dtype=float)
        prob[pval > thresh] = 0.0
        # Mirror the canonical fallback chain used elsewhere in this
        # file: ``modeling.py`` writes the third-dim names under
        # ``"interaction_name"`` while ``modeling_avg.py`` mirrors both
        # ``"interaction_name"`` and ``"pairs"`` for back-compat.  Read
        # in the same order everywhere to keep the pair-index map valid
        # regardless of which producer wrote ``cc["net"]``.
        _dimnames = cc["net"].get("dimnames", {})
        pair_axis = (
            _dimnames.get("interaction_name")
            or _dimnames.get("pairs")
            or [str(i) for i in range(prob.shape[2])]
        )
        pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
        pair_names = [p for p in pairLR.index if p in pair_idx_map]
        if len(pair_names) == 0:
            raise ValueError(f"There is no significant communication of {signaling}")
        idx = [pair_idx_map[p] for p in pair_names]
        sums = prob[:, :, idx].sum(axis=(0, 1))
        keep_mask = sums > 0
        if not np.any(keep_mask):
            raise ValueError(f"There is no significant communication of {signaling}")
        pair_names_use = [pair_names[k] for k in np.flatnonzero(keep_mask)]
        idx_use = [pair_idx_map[p] for p in pair_names_use]
        prob_use = prob[:, :, idx_use]

        # "cells" (computeCommunProb) vs "source"/"target" (computeAvgCommunProb)
        _dn3 = cc["net"].get("dimnames", {})
        cells = _dn3.get("cells") or _dn3.get("source")
        if cells is None:
            idents = _get_idents(adata, None)
            cells = list(idents.categories)

        if slot_name == "netP":
            net_2d = prob_use.sum(axis=2)
            df_net = pd.DataFrame(net_2d, index=cells, columns=cells)
            if title_name is None:
                title_name = f"{signaling_list[0]} signaling pathway network"
            return netVisual_chord_cell_internal(
                df_net, color_use=color_use, group=group, cell_order=cell_order,
                sources_use=sources_use, targets_use=targets_use,
                lab_cex=lab_cex, small_gap=small_gap, big_gap=big_gap,
                annotation_track_height=annotation_track_height,
                remove_isolate=remove_isolate, link_visible=link_visible,
                scale=scale, directional=directional,
                link_target_prop=link_target_prop, reduce=reduce,
                transparency=transparency, link_border=link_border,
                title_name=title_name, show_legend=show_legend,
                legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
                figsize=figsize,
            )
        elif slot_name == "net":
            n_col_resolved = min(len(pair_names_use), 2) if n_col is None else int(n_col)
            figs: List[Figure] = []
            for i, p in enumerate(pair_names_use):
                df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
                tn = pairLR.loc[p, "interaction_name_2"] if p in pairLR.index and "interaction_name_2" in pairLR.columns else p
                figs.append(netVisual_chord_cell_internal(
                    df_i, color_use=color_use, group=group, cell_order=cell_order,
                    sources_use=sources_use, targets_use=targets_use,
                    lab_cex=lab_cex, small_gap=small_gap, big_gap=big_gap,
                    annotation_track_height=annotation_track_height,
                    remove_isolate=remove_isolate, link_visible=link_visible,
                    scale=scale, directional=directional,
                    link_target_prop=link_target_prop, reduce=reduce,
                    transparency=transparency, link_border=link_border,
                    title_name=tn, show_legend=show_legend,
                    legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
                    figsize=figsize,
                ))
            return figs
        else:
            raise ValueError(f"Unknown slot_name: {slot_name}")
    elif net is not None:
        return netVisual_chord_cell_internal(
            net, color_use=color_use, group=group, cell_order=cell_order,
            sources_use=sources_use, targets_use=targets_use,
            lab_cex=lab_cex, small_gap=small_gap, big_gap=big_gap,
            annotation_track_height=annotation_track_height,
            remove_isolate=remove_isolate, link_visible=link_visible,
            scale=scale, directional=directional,
            link_target_prop=link_target_prop, reduce=reduce,
            transparency=transparency, link_border=link_border,
            title_name=title_name, show_legend=show_legend,
            legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
            figsize=figsize,
        )
    else:
        raise ValueError("Please assign values to either `signaling` or `net`")


def netVisual_chord_gene(
    adata: ad.AnnData,
    slot_name: str = "net",
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR_use: Optional[pd.DataFrame] = None,
    net: Optional[pd.DataFrame] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    lab_cex: float = 0.8,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    annotation_track_height: float = 0.03,
    link_visible: bool = True,
    scale: bool = False,
    directional: int = 1,
    link_target_prop: bool = True,
    reduce: float = -1.0,
    transparency: float = 0.4,
    link_border: Optional[str] = None,
    title_name: Optional[str] = None,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
    show_legend: bool = True,
    thresh: float = 0.05,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> Figure:
    """Chord diagram showing ligand → receptor (or pathway) connections.

    Direct port of R's :func:`netVisual_chord_gene` (``visualization.R:3553``).
    """
    if pairLR_use is not None:
        if not isinstance(pairLR_use, pd.DataFrame):
            raise ValueError(
                "pairLR.use should be a data frame with a single column named "
                "either 'interaction_name' or 'pathway_name'"
            )
        if "pathway_name" in pairLR_use.columns:
            slot_name = "netP"

    if pairLR_use is not None and signaling is not None:
        raise ValueError("Please do not assign values to 'signaling' when using 'pairLR.use'")

    cc = _ensure_uns(adata)
    LRsig = cc["LR"]["LRsig"]
    # "cells" (computeCommunProb) vs "source"/"target" (computeAvgCommunProb)
    _dn3 = cc["net"].get("dimnames", {})
    cells = _dn3.get("cells") or _dn3.get("source")
    if cells is None:
        idents = _get_idents(adata, None)
        cells = list(idents.categories)

    if net is None:
        prob = np.asarray(cc["net"]["prob"], dtype=float).copy()
        pval = cc["net"].get("pval", None)
        if pval is not None:
            prob[np.asarray(pval, dtype=float) > thresh] = 0.0
        # Mirror the canonical fallback chain used elsewhere in this
        # file: ``modeling.py`` writes the third-dim names under
        # ``"interaction_name"`` while ``modeling_avg.py`` mirrors both
        # ``"interaction_name"`` and ``"pairs"`` for back-compat.  Read
        # in the same order everywhere to keep the pair-index map valid
        # regardless of which producer wrote ``cc["net"]``.
        _dimnames = cc["net"].get("dimnames", {})
        pair_axis = (
            _dimnames.get("interaction_name")
            or _dimnames.get("pairs")
            or [str(i) for i in range(prob.shape[2])]
        )
        # melt into long
        K = prob.shape[0]
        rows: List[Dict[str, Any]] = []
        for k, pn in enumerate(pair_axis):
            sub = prob[:, :, k]
            r_idx, c_idx = np.nonzero(sub)
            for ri, ci in zip(r_idx, c_idx):
                rows.append({
                    "source": cells[ri],
                    "target": cells[ci],
                    "interaction_name": pn,
                    "prob": float(sub[ri, ci]),
                })
        net_df = pd.DataFrame(rows)
        if "interaction_name_2" in LRsig.columns:
            extra_cols = [c for c in ["interaction_name_2", "pathway_name", "ligand", "receptor", "annotation", "evidence"] if c in LRsig.columns]
            sub = LRsig[extra_cols].copy()
            net_df = net_df.merge(sub, left_on="interaction_name", right_index=True, how="left")
    else:
        net_df = net.copy()

    if signaling is not None:
        sig_list = [signaling] if isinstance(signaling, str) else list(signaling)
        pairLR_use_local = pd.concat(
            [
                searchPair(s, LRsig, key="pathway_name", matching_exact=True, pair_only=True)
                for s in sig_list
            ],
            ignore_index=False,
        )
        pairLR_use = pairLR_use_local

    if pairLR_use is not None:
        if "interaction_name" in pairLR_use.columns:
            net_df = net_df[net_df["interaction_name"].isin(pairLR_use["interaction_name"])]
        elif "pathway_name" in pairLR_use.columns:
            net_df = net_df[net_df["pathway_name"].astype(str).isin(pairLR_use["pathway_name"].astype(str))]

    if slot_name == "netP":
        cols = [c for c in ["source", "target", "pathway_name", "prob"] if c in net_df.columns]
        net_df = net_df[cols].copy()
        net_df["source_target"] = (
            net_df["source"].astype(str) + "sourceTotarget" + net_df["target"].astype(str)
        )
        agg = (
            net_df.groupby(["source_target", "pathway_name"], observed=False, sort=False)["prob"]
            .sum().reset_index()
        )
        st = agg["source_target"].str.split("sourceTotarget", n=1, expand=True)
        agg["source"] = st[0].astype(str)
        agg["target"] = st[1].astype(str)
        agg["ligand"] = agg["pathway_name"].astype(str)
        agg["receptor"] = " "
        net_df = agg

    if sources_use is not None:
        sources_use = _resolve_indices(list(sources_use), cells)
        net_df = net_df[net_df["source"].astype(str).isin(sources_use)]
    else:
        sources_use = list(cells)
    if targets_use is not None:
        targets_use = _resolve_indices(list(targets_use), cells)
        net_df = net_df[net_df["target"].astype(str).isin(targets_use)]
    else:
        targets_use = list(cells)

    df = net_df[net_df["prob"] > 0].copy()
    if df.shape[0] == 0:
        raise ValueError("No signaling links are inferred!")

    df = df.reset_index(drop=True)
    df["id"] = np.arange(df.shape[0]) + 1
    # Disambiguate duplicated ligand/receptor by appending spaces
    if "ligand" in df.columns:
        for ln in df["ligand"].unique():
            sub = df[df["ligand"] == ln]
            for j, src in enumerate(sub["source"].unique()):
                ids = sub[sub["source"] == src]["id"].tolist()
                df.loc[df["id"].isin(ids), "ligand"] = ln + " " * j
    if "receptor" in df.columns:
        for rn in df["receptor"].unique():
            sub = df[df["receptor"] == rn]
            for j, tgt in enumerate(sub["target"].unique()):
                ids = sub[sub["target"] == tgt]["id"].tolist()
                df.loc[df["id"].isin(ids), "receptor"] = rn + " " * j

    cell_order_sources = [c for c in cells if c in set(sources_use)]
    cell_order_targets = [c for c in cells if c in set(targets_use)]

    df["source"] = pd.Categorical(df["source"], categories=cell_order_sources)
    df["target"] = pd.Categorical(df["target"], categories=cell_order_targets)
    df_ordered_source = df.sort_values(["source", "prob"], ascending=[True, False])
    df_ordered_target = df.sort_values(["target", "prob"], ascending=[True, False])

    order_source = df_ordered_source[["ligand", "source"]].drop_duplicates()
    order_target = df_ordered_target[["receptor", "target"]].drop_duplicates()
    order_sector = list(order_source["ligand"]) + list(order_target["receptor"])

    if color_use is None:
        cu_arr = list(scPalette(len(cells)))
        cu_full = {c: cu_arr[i] for i, c in enumerate(cells)}
        color_use_d = {c: cu_full[c] for c in cells if c in set(df["source"].astype(str)).union(set(df["target"].astype(str)))}
    elif isinstance(color_use, dict):
        color_use_d = dict(color_use)
    else:
        cu = list(color_use)
        color_use_d = {c: cu[i % len(cu)] for i, c in enumerate(cells) if c in set(df["source"].astype(str)).union(set(df["target"].astype(str)))}

    # grid colours for ligand/receptor sectors derive from the cell palette
    grid_col_ligand = {row["ligand"]: color_use_d.get(str(row["source"]), "#888888") for _, row in order_source.iterrows()}
    grid_col_receptor = {row["receptor"]: color_use_d.get(str(row["target"]), "#888888") for _, row in order_target.iterrows()}
    grid_col = {**grid_col_ligand, **grid_col_receptor}

    df_plot = df_ordered_source[["ligand", "receptor", "prob"]].copy()
    df_plot.columns = ["source", "target", "prob"]

    fig = _chord_diagram(
        df_plot, order_sector, grid_col,
        transparency=transparency, title_name=title_name,
        figsize=figsize, show_legend=show_legend, lab_cex=lab_cex,
        directional=directional, link_target_prop=link_target_prop,
    )
    return fig


# ---------------------------------------------------------------------------
# Dispatcher (R: visualization.R:866, 1255, 1419)
# ---------------------------------------------------------------------------
def _resolve_signaling(
    adata: ad.AnnData,
    signaling: str,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], List[str]]:
    """Internal: resolve signaling -> (prob, prob_sum, pairLR, pair_names_use, cells)."""
    cc = _ensure_uns(adata)
    LRsig = cc["LR"]["LRsig"]
    pairLR = searchPair(signaling, LRsig, key="pathway_name", matching_exact=True, pair_only=False)
    net_dict = cc["net"]
    prob = np.asarray(net_dict["prob"], dtype=float).copy()
    pval = net_dict.get("pval", None)
    if pval is not None:
        prob[np.asarray(pval, dtype=float) > thresh] = 0.0
    # The 3rd-dim names key was originally "pairs" (set by computeCommunProb)
    # but computeAvgCommunProb / *_Visium overwrites the dimnames dict with
    # R-style "interaction_name" keys; check both before falling back to
    # integer placeholders.
    _dimnames = net_dict.get("dimnames", {})
    pair_axis = (
        _dimnames.get("interaction_name")
        or _dimnames.get("pairs")
        or [str(i) for i in range(prob.shape[2])]
    )
    pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
    pair_names = [p for p in pairLR.index if p in pair_idx_map]
    if len(pair_names) == 0:
        raise ValueError(f"There is no significant communication of {signaling}")
    idx = [pair_idx_map[p] for p in pair_names]
    sums = prob[:, :, idx].sum(axis=(0, 1))
    keep = sums > 0
    if not np.any(keep):
        raise ValueError(f"There is no significant communication of {signaling}")
    pair_names_use = [pair_names[k] for k in np.flatnonzero(keep)]
    idx_use = [pair_idx_map[p] for p in pair_names_use]
    prob_use = prob[:, :, idx_use]
    # "cells" (computeCommunProb) vs "source"/"target" (computeAvgCommunProb)
    _dn2 = net_dict.get("dimnames", {})
    cells = _dn2.get("cells") or _dn2.get("source")
    if cells is None:
        idents = _get_idents(adata, None)
        cells = list(idents.categories)
    return prob_use, prob_use.sum(axis=2), pairLR.loc[pair_names_use], pair_names_use, cells


def _persist_per_pair_files(
    per_pair_figs: List[Figure],
    agg_fig: Figure,
    *,
    signaling_name: str,
    layout: str,
    out_format: Union[str, Sequence[str]],
    out_dir: str,
    height: float,
    n_col: int,
) -> None:
    """Write per-pair-stacked + aggregate files to ``out_dir``.

    R parity (``visualization.R:1001-1057`` for ``circle``, similar for
    ``spatial`` / ``chord``): each requested format produces:

    - ``<signaling>_<layout>_individual.<fmt>`` — grid of per-LR-pair
      panels (R uses ``par(mfrow = c(ceiling(nPair/nCol), nCol))``).
    - ``<signaling>_<layout>_aggregate.<fmt>`` — single panel of the
      ``prob.sum`` aggregate.

    For ``pdf`` we use a multi-page :class:`PdfPages` document (one page
    per pair) so vector output stays sharp; for raster formats we
    composite the per-pair Figures into a single grid via
    :func:`_replay_axes`. The aggregate file is always a single panel.
    """
    import os
    from io import BytesIO
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)
    fmts = [out_format] if isinstance(out_format, str) else list(out_format)
    n_pairs = len(per_pair_figs)
    n_cols_eff = max(int(n_col), 1)
    n_rows = max(1, int(np.ceil(n_pairs / n_cols_eff)))
    for fmt in fmts:
        ind_path = os.path.join(
            out_dir, f"{signaling_name}_{layout}_individual.{fmt}"
        )
        if fmt == "pdf":
            # Multi-page PDF: one page per pair. Mirrors
            # ``cairo_pdf(..., width=height, height=nRow*height)``.
            with PdfPages(ind_path) as pdf:
                for f in per_pair_figs:
                    pdf.savefig(f, bbox_inches="tight")
        elif layout == "hierarchy":
            # Hierarchy panels are pure matplotlib subplots — re-host
            # them in a 2-column grid via ``_replay_axes`` (R: par(mfrow=c(nRow,2))).
            if n_pairs == 0:
                stacked = plt.figure()
            else:
                stacked = plt.figure(figsize=(8.0, height * n_pairs))
                for k, f_pair in enumerate(per_pair_figs):
                    sub_axes = list(f_pair.axes)
                    ax_left = stacked.add_subplot(n_pairs, 2, k * 2 + 1)
                    ax_right = stacked.add_subplot(n_pairs, 2, k * 2 + 2)
                    for src_ax, dest_ax in zip(
                        sub_axes[:2], (ax_left, ax_right)
                    ):
                        _replay_axes(src_ax, dest_ax)
            stacked.savefig(ind_path, bbox_inches="tight")
            plt.close(stacked)
        else:
            # circle / spatial / chord: rasterise each per-pair Figure to
            # an in-memory PNG buffer (avoids fragility of ``_replay_axes``
            # for pycirclize / spatial overlays), then composite into a
            # single grid image via PIL. Mirrors R's single
            # ``<sig>_<layout>_individual.<fmt>`` output (R uses
            # ``par(mfrow = c(ceiling(nPair/nCol), nCol))``).
            from PIL import Image

            if n_pairs == 0:
                stacked = plt.figure()
                stacked.savefig(ind_path, bbox_inches="tight")
                plt.close(stacked)
            else:
                tile_buffers = []
                for f_pair in per_pair_figs:
                    buf = BytesIO()
                    f_pair.savefig(buf, format="png", bbox_inches="tight",
                                   dpi=150)
                    buf.seek(0)
                    tile_buffers.append(Image.open(buf).convert("RGBA"))
                tile_w = max(t.width for t in tile_buffers)
                tile_h = max(t.height for t in tile_buffers)
                # Pad each tile up to (tile_w, tile_h) to keep the grid aligned
                grid = Image.new(
                    "RGBA",
                    (tile_w * n_cols_eff, tile_h * n_rows),
                    (255, 255, 255, 0),
                )
                for k, t in enumerate(tile_buffers):
                    rr, cc = divmod(k, n_cols_eff)
                    grid.paste(t, (cc * tile_w, rr * tile_h), mask=t)
                if fmt == "svg":
                    # PIL can't write SVG directly — fall back to embedding the
                    # composite PNG inside an SVG wrapper (still a valid SVG file).
                    png_buf = BytesIO()
                    grid.save(png_buf, format="PNG")
                    png_buf.seek(0)
                    import base64
                    b64 = base64.b64encode(png_buf.read()).decode("ascii")
                    svg = (
                        f'<svg xmlns="http://www.w3.org/2000/svg" '
                        f'width="{grid.width}" height="{grid.height}">'
                        f'<image href="data:image/png;base64,{b64}" '
                        f'width="{grid.width}" height="{grid.height}"/>'
                        f'</svg>'
                    )
                    with open(ind_path, "w") as fh:
                        fh.write(svg)
                else:
                    grid.save(ind_path)
        agg_path = os.path.join(
            out_dir, f"{signaling_name}_{layout}_aggregate.{fmt}"
        )
        agg_fig.savefig(agg_path, bbox_inches="tight")


def netVisual(
    adata: ad.AnnData,
    signaling: str,
    signaling_name: Optional[str] = None,
    color_use: Optional[Sequence[str]] = None,
    vertex_receiver: Optional[Sequence[int]] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    top: float = 1.0,
    remove_isolate: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 1.0,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    weight_scale: bool = True,
    edge_weight_max_individual: Optional[float] = None,
    edge_weight_max_aggregate: Optional[float] = None,
    edge_width_max: float = 8.0,
    layout: str = "circle",
    height: float = 5.0,
    thresh: float = 0.05,
    pt_title: float = 12.0,
    title_space: float = 6.0,
    vertex_label_cex: float = 0.8,
    alpha_image: float = 0.15,
    point_size: float = 1.5,
    group: Optional[Dict[str, str]] = None,
    cell_order: Optional[Sequence[str]] = None,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    scale: bool = False,
    reduce: float = -1.0,
    show_legend: bool = False,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
    n_col: Optional[int] = None,
    out_format: Optional[Union[str, Sequence[str]]] = None,
    out_dir: Optional[str] = None,
) -> Figure:
    """Visualize the inferred signaling network.

    Direct port of R's :func:`netVisual` (``visualization.R:866``). Routes
    to the appropriate per-layout function based on ``layout``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"]["net"]`` populated.
    signaling : str
        Pathway name.
    layout : {"circle", "hierarchy", "spatial", "chord", "heatmap"}
        Plot layout.
    out_format : str or sequence of str, optional
        When set together with ``out_dir`` and ``layout='hierarchy'``,
        two files per format are written under ``out_dir``:
        ``<signaling>_hierarchy_individual.<fmt>`` (multi-page / stacked
        per-pair panels) and ``<signaling>_hierarchy_aggregate.<fmt>``
        (the prob.sum aggregate panel). Mirrors R's vector ``out.format``
        argument (``visualization.R:937-998``).
    out_dir : str, optional
        Output directory for the hierarchy-mode files.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if layout not in ("circle", "hierarchy", "spatial", "chord", "heatmap"):
        raise ValueError(f"Unknown layout: {layout}")
    if signaling_name is None:
        signaling_name = signaling
    if vertex_weight is None:
        idents = _get_idents(adata, None)
        vertex_weight = pd.Series(idents).value_counts().reindex(idents.categories, fill_value=0).to_numpy(dtype=float)
    if vertex_size_max is None:
        if hasattr(vertex_weight, "__len__") and len(set(np.atleast_1d(vertex_weight).tolist())) == 1:
            vertex_size_max = 5.0
        elif not hasattr(vertex_weight, "__len__"):
            vertex_size_max = 5.0
        else:
            vertex_size_max = 15.0

    prob_use, prob_sum, pairLR, pair_names_use, cells = _resolve_signaling(adata, signaling, thresh)
    if edge_weight_max_individual is None:
        edge_weight_max_individual = float(prob_use.max()) if prob_use.size > 0 else 1.0
    if edge_weight_max_aggregate is None:
        edge_weight_max_aggregate = float(prob_sum.max()) if prob_sum.size > 0 else 1.0

    n_col_resolved = (
        int(n_col) if n_col is not None else min(max(len(pair_names_use), 1), 2)
    )

    if layout == "circle":
        df_sum = pd.DataFrame(prob_sum, index=cells, columns=cells)
        agg = netVisual_circle(
            df_sum, color_use=color_use, sources_use=sources_use, targets_use=targets_use,
            remove_isolate=remove_isolate, top=top, vertex_weight=vertex_weight,
            vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
            weight_scale=weight_scale, edge_weight_max=edge_weight_max_aggregate,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            title_name=f"{signaling_name} signaling pathway network",
        )
        # R parity (visualization.R:1001-1057): when out_format is given,
        # write per-pair grid + aggregate file per format.
        if out_format is not None and out_dir is not None:
            per_pair_figs: List[Figure] = []
            for i, p in enumerate(pair_names_use):
                df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
                tn = pairLR.loc[p, "interaction_name_2"] if (
                    p in pairLR.index and "interaction_name_2" in pairLR.columns
                ) else p
                f_pair = netVisual_circle(
                    df_i, color_use=color_use, sources_use=sources_use,
                    targets_use=targets_use, remove_isolate=remove_isolate,
                    top=top, vertex_weight=vertex_weight,
                    vertex_weight_max=vertex_weight_max,
                    vertex_size_max=vertex_size_max, weight_scale=weight_scale,
                    edge_weight_max=edge_weight_max_individual,
                    edge_width_max=edge_width_max,
                    vertex_label_cex=vertex_label_cex, title_name=str(tn),
                )
                per_pair_figs.append(f_pair)
            _persist_per_pair_files(
                per_pair_figs, agg,
                signaling_name=str(signaling_name), layout="circle",
                out_format=out_format, out_dir=out_dir,
                height=height, n_col=n_col_resolved,
            )
            for f in per_pair_figs:
                plt.close(f)
        return agg
    elif layout == "hierarchy":
        if vertex_receiver is None:
            raise ValueError("vertex_receiver is required for hierarchy layout")
        vr = list(vertex_receiver)
        # R: iterate per-pair, draw both hierarchy panels per pair, then write
        # ONE multi-page individual file plus ONE aggregate file per requested
        # output format (R: `out.format` is a vector;
        # ``visualization.R:937-998``). Produce per-pair Figures first.
        per_pair_figs: List[Figure] = []
        for i, p in enumerate(pair_names_use):
            df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
            tn = pairLR.loc[p, "interaction_name_2"] if (
                p in pairLR.index and "interaction_name_2" in pairLR.columns
            ) else p
            f_pair = _hierarchy_combined(
                net=df_i, vertex_receiver=vr, sources_use=sources_use,
                targets_use=targets_use, remove_isolate=remove_isolate,
                top=top, color_use=color_use, vertex_weight=vertex_weight,
                vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
                weight_scale=weight_scale, edge_weight_max=edge_weight_max_individual,
                edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
                title_name=str(tn),
            )
            per_pair_figs.append(f_pair)
        # Aggregate prob.sum hierarchy panel for the pathway
        df_sum = pd.DataFrame(prob_sum, index=cells, columns=cells)
        f_agg = _hierarchy_combined(
            net=df_sum, vertex_receiver=vr, sources_use=sources_use,
            targets_use=targets_use, remove_isolate=remove_isolate,
            top=top, color_use=color_use, vertex_weight=vertex_weight,
            vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
            weight_scale=weight_scale, edge_weight_max=edge_weight_max_aggregate,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            title_name=str(signaling_name),
        )
        # R parity (visualization.R:937-998): write per-pair (multi-row,
        # 2 columns of source/target panels) + aggregate file per format.
        if out_format is not None and out_dir is not None:
            _persist_per_pair_files(
                per_pair_figs, f_agg,
                signaling_name=str(signaling_name), layout="hierarchy",
                out_format=out_format, out_dir=out_dir,
                height=height, n_col=2,
            )
        # Close per-pair sub figures (kept on disk if requested) and return the
        # aggregate figure; R returns the recordedplot of the aggregate pane.
        for f in per_pair_figs:
            plt.close(f)
        return f_agg
    elif layout == "spatial":
        coords_obs = adata.obsm["spatial"]
        idents = _get_idents(adata, None)
        df_sum = pd.DataFrame(prob_sum, index=cells, columns=cells)
        if isinstance(vertex_weight, str) and vertex_weight in ("incoming", "outgoing"):
            centr = adata.uns["cellchat"].get("netP", {}).get("centr", None)
            if centr is None:
                raise ValueError(
                    "Please run `netAnalysis_computeCentrality` to compute the network centrality scores!"
                )
            measure = "indeg" if vertex_weight == "incoming" else "outdeg"
            vertex_weight = centr_value(centr, measure, signaling)
        agg = netVisual_spatial(
            df_sum, coordinates=coords_obs, labels=list(idents),
            color_use=color_use, sources_use=sources_use, targets_use=targets_use,
            remove_isolate=remove_isolate, top=top, vertex_weight=vertex_weight,
            vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
            weight_scale=weight_scale, edge_weight_max=edge_weight_max_aggregate,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            alpha_image=alpha_image, point_size=point_size,
            title_name=f"{signaling_name} signaling pathway network",
        )
        # R parity (visualization.R:1058-1124): per-pair grid + aggregate.
        if out_format is not None and out_dir is not None:
            per_pair_figs: List[Figure] = []
            for i, p in enumerate(pair_names_use):
                df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
                tn = pairLR.loc[p, "interaction_name_2"] if (
                    p in pairLR.index and "interaction_name_2" in pairLR.columns
                ) else p
                f_pair = netVisual_spatial(
                    df_i, coordinates=coords_obs, labels=list(idents),
                    color_use=color_use, sources_use=sources_use,
                    targets_use=targets_use, remove_isolate=remove_isolate,
                    top=top, vertex_weight=vertex_weight,
                    vertex_weight_max=vertex_weight_max,
                    vertex_size_max=vertex_size_max, weight_scale=weight_scale,
                    edge_weight_max=edge_weight_max_individual,
                    edge_width_max=edge_width_max,
                    vertex_label_cex=vertex_label_cex,
                    alpha_image=alpha_image, point_size=point_size,
                    title_name=str(tn),
                )
                per_pair_figs.append(f_pair)
            _persist_per_pair_files(
                per_pair_figs, agg,
                signaling_name=str(signaling_name), layout="spatial",
                out_format=out_format, out_dir=out_dir,
                height=height, n_col=n_col_resolved,
            )
            for f in per_pair_figs:
                plt.close(f)
        return agg
    elif layout == "chord":
        df_sum = pd.DataFrame(prob_sum, index=cells, columns=cells)
        agg = netVisual_chord_cell_internal(
            df_sum, color_use=color_use, sources_use=sources_use,
            targets_use=targets_use, remove_isolate=remove_isolate, group=group,
            cell_order=cell_order, lab_cex=vertex_label_cex,
            small_gap=small_gap, big_gap=big_gap, scale=scale, reduce=reduce,
            title_name=f"{signaling_name} signaling pathway network",
            show_legend=show_legend, legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
        )
        # R parity (visualization.R:1125-1183): per-pair grid + aggregate.
        if out_format is not None and out_dir is not None:
            per_pair_figs: List[Figure] = []
            for i, p in enumerate(pair_names_use):
                df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
                tn = pairLR.loc[p, "interaction_name_2"] if (
                    p in pairLR.index and "interaction_name_2" in pairLR.columns
                ) else p
                f_pair = netVisual_chord_cell_internal(
                    df_i, color_use=color_use, sources_use=sources_use,
                    targets_use=targets_use, remove_isolate=remove_isolate,
                    group=group, cell_order=cell_order,
                    lab_cex=vertex_label_cex, small_gap=small_gap,
                    big_gap=big_gap, scale=scale, reduce=reduce,
                    title_name=str(tn), show_legend=show_legend,
                    legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
                )
                per_pair_figs.append(f_pair)
            _persist_per_pair_files(
                per_pair_figs, agg,
                signaling_name=str(signaling_name), layout="chord",
                out_format=out_format, out_dir=out_dir,
                height=height, n_col=n_col_resolved,
            )
            for f in per_pair_figs:
                plt.close(f)
        return agg
    elif layout == "heatmap":
        # Lazy-import sub-agent 6b's module. R doesn't list "heatmap" as a
        # valid `netVisual` layout (line 869), but the porting contract
        # routes it via the dispatcher for convenience.
        try:
            from .viz_heatmap import netVisual_heatmap  # type: ignore
        except ImportError:
            raise NotImplementedError(
                "layout='heatmap' requires viz_heatmap.netVisual_heatmap "
                "(sub-agent 6b's module). Re-run after that slice is "
                "merged."
            ) from None
        return netVisual_heatmap(
            adata, signaling=signaling, slot_name="net", thresh=thresh,
            color_use=color_use, sources_use=sources_use,
            targets_use=targets_use, remove_isolate=remove_isolate,
        )
    raise ValueError(f"Unknown layout: {layout}")


def netVisual_aggregate(
    adata: ad.AnnData,
    signaling: str,
    signaling_name: Optional[str] = None,
    color_use: Optional[Sequence[str]] = None,
    thresh: float = 0.05,
    vertex_receiver: Optional[Sequence[int]] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    idents_use: Optional[Sequence] = None,
    top: float = 1.0,
    remove_isolate: bool = False,
    vertex_weight: Union[float, Sequence[float], str] = 1.0,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    weight_scale: bool = True,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    layout: str = "circle",
    pt_title: float = 12.0,
    title_space: float = 6.0,
    vertex_label_cex: float = 0.8,
    alpha_image: float = 0.15,
    point_size: float = 1.5,
    group: Optional[Dict[str, str]] = None,
    cell_order: Optional[Sequence[str]] = None,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    scale: bool = False,
    reduce: float = -1.0,
    show_legend: bool = False,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
) -> Figure:
    """Aggregate L-R pairs of a signaling pathway then render.

    Direct port of R's :func:`netVisual_aggregate` (``visualization.R:1255``).
    """
    if layout not in ("circle", "hierarchy", "spatial", "chord"):
        raise ValueError(f"Unknown layout: {layout}")
    if signaling_name is None:
        signaling_name = signaling
    if vertex_weight is None:
        idents = _get_idents(adata, None)
        vertex_weight = pd.Series(idents).value_counts().reindex(idents.categories, fill_value=0).to_numpy(dtype=float)

    cc = _ensure_uns(adata)
    LRsig = cc["LR"]["LRsig"]
    pairLR = searchPair(signaling, LRsig, key="pathway_name", matching_exact=True, pair_only=True)
    net_dict = cc["net"]
    prob = np.asarray(net_dict["prob"], dtype=float).copy()
    pval = net_dict.get("pval", None)
    if pval is not None:
        prob[np.asarray(pval, dtype=float) > thresh] = 0.0
    # The 3rd-dim names key was originally "pairs" (set by computeCommunProb)
    # but computeAvgCommunProb / *_Visium overwrites the dimnames dict with
    # R-style "interaction_name" keys; check both before falling back to
    # integer placeholders.
    _dimnames = net_dict.get("dimnames", {})
    pair_axis = (
        _dimnames.get("interaction_name")
        or _dimnames.get("pairs")
        or [str(i) for i in range(prob.shape[2])]
    )
    pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
    pair_names = [p for p in pairLR.index if p in pair_idx_map]
    if len(pair_names) == 0:
        raise ValueError(f"There is no significant communication of {signaling}")
    idx = [pair_idx_map[p] for p in pair_names]
    sums = prob[:, :, idx].sum(axis=(0, 1))
    keep = sums > 0
    if not np.any(keep):
        raise ValueError(f"There is no significant communication of {signaling}")
    pair_names_use = [pair_names[k] for k in np.flatnonzero(keep)]
    idx_use = [pair_idx_map[p] for p in pair_names_use]
    prob_use = prob[:, :, idx_use]
    # "cells" (computeCommunProb) vs "source"/"target" (computeAvgCommunProb)
    _dn2 = net_dict.get("dimnames", {})
    cells = _dn2.get("cells") or _dn2.get("source")
    if cells is None:
        idents = _get_idents(adata, None)
        cells = list(idents.categories)

    prob_sum = prob_use.sum(axis=2)
    if edge_weight_max is None:
        edge_weight_max = float(prob_sum.max()) if prob_sum.size > 0 else 1.0

    df_sum = pd.DataFrame(prob_sum, index=cells, columns=cells)

    if layout == "circle":
        return netVisual_circle(
            df_sum, color_use=color_use, sources_use=sources_use, targets_use=targets_use,
            idents_use=idents_use, remove_isolate=remove_isolate,
            top=top, vertex_weight=vertex_weight,
            vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
            weight_scale=weight_scale, edge_weight_max=edge_weight_max,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            title_name=f"{signaling_name} signaling pathway network",
        )
    elif layout == "hierarchy":
        if vertex_receiver is None:
            raise ValueError("vertex_receiver required for hierarchy layout")
        return netVisual_hierarchy1(
            df_sum, vertex_receiver=vertex_receiver,
            sources_use=sources_use, targets_use=targets_use,
            remove_isolate=remove_isolate, top=top, color_use=color_use,
            vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
            vertex_size_max=vertex_size_max, weight_scale=weight_scale,
            edge_weight_max=edge_weight_max, edge_width_max=edge_width_max,
            vertex_label_cex=vertex_label_cex, title_name=signaling_name,
        )
    elif layout == "spatial":
        coords_obs = adata.obsm["spatial"]
        idents = _get_idents(adata, None)
        if isinstance(vertex_weight, str) and vertex_weight in ("incoming", "outgoing"):
            centr = adata.uns["cellchat"].get("netP", {}).get("centr", None)
            if centr is None:
                raise ValueError(
                    "Please run `netAnalysis_computeCentrality` to compute the network centrality scores!"
                )
            measure = "indeg" if vertex_weight == "incoming" else "outdeg"
            vertex_weight = centr_value(centr, measure, signaling)
        return netVisual_spatial(
            df_sum, coordinates=coords_obs, labels=list(idents),
            color_use=color_use, sources_use=sources_use, targets_use=targets_use,
            idents_use=idents_use, remove_isolate=remove_isolate,
            top=top, vertex_weight=vertex_weight,
            vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
            weight_scale=weight_scale, edge_weight_max=edge_weight_max,
            edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
            alpha_image=alpha_image, point_size=point_size,
            title_name=f"{signaling_name} signaling pathway network",
        )
    elif layout == "chord":
        return netVisual_chord_cell_internal(
            df_sum, color_use=color_use, sources_use=sources_use,
            targets_use=targets_use, remove_isolate=remove_isolate, group=group,
            cell_order=cell_order, lab_cex=vertex_label_cex,
            small_gap=small_gap, big_gap=big_gap, scale=scale, reduce=reduce,
            title_name=f"{signaling_name} signaling pathway network",
            show_legend=show_legend, legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
        )
    raise ValueError(f"Unknown layout: {layout}")


def netVisual_individual(
    adata: ad.AnnData,
    signaling: str,
    signaling_name: Optional[str] = None,
    pairLR_use: Optional[Union[pd.DataFrame, Sequence[str]]] = None,
    color_use: Optional[Sequence[str]] = None,
    vertex_receiver: Optional[Sequence[int]] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    top: float = 1.0,
    remove_isolate: bool = False,
    vertex_weight: Union[float, Sequence[float]] = 1.0,
    vertex_weight_max: Optional[float] = None,
    vertex_size_max: Optional[float] = None,
    vertex_label_cex: float = 0.8,
    weight_scale: bool = True,
    edge_weight_max: Optional[float] = None,
    edge_width_max: float = 8.0,
    layout: str = "circle",
    height: float = 5.0,
    thresh: float = 0.05,
    alpha_image: float = 0.15,
    point_size: float = 1.5,
    group: Optional[Dict[str, str]] = None,
    cell_order: Optional[Sequence[str]] = None,
    small_gap: float = 1.0,
    big_gap: float = 10.0,
    scale: bool = False,
    reduce: float = -1.0,
    show_legend: bool = False,
    legend_pos_x: float = 20.0,
    legend_pos_y: float = 20.0,
    n_col: Optional[int] = None,
) -> Union[List[Figure], Figure]:
    """Render one Figure per L-R pair within a signaling pathway.

    Direct port of R's :func:`netVisual_individual` (``visualization.R:1419``).

    Returns a list of figures for ``layout`` in
    ``{'circle', 'spatial', 'chord'}``. For ``layout='hierarchy'`` returns
    a single composite Figure (R returns one ``recordedplot``).
    """
    if layout not in ("circle", "hierarchy", "spatial", "chord"):
        raise ValueError(f"Unknown layout: {layout}")
    if signaling_name is None:
        signaling_name = signaling
    if vertex_weight is None:
        idents = _get_idents(adata, None)
        vertex_weight = pd.Series(idents).value_counts().reindex(idents.categories, fill_value=0).to_numpy(dtype=float)

    cc = _ensure_uns(adata)
    LRsig = cc["LR"]["LRsig"]
    pairLR = searchPair(signaling, LRsig, key="pathway_name", matching_exact=True, pair_only=False)
    net_dict = cc["net"]
    prob = np.asarray(net_dict["prob"], dtype=float).copy()
    pval = net_dict.get("pval", None)
    if pval is not None:
        prob[np.asarray(pval, dtype=float) > thresh] = 0.0
    # The 3rd-dim names key was originally "pairs" (set by computeCommunProb)
    # but computeAvgCommunProb / *_Visium overwrites the dimnames dict with
    # R-style "interaction_name" keys; check both before falling back to
    # integer placeholders.
    _dimnames = net_dict.get("dimnames", {})
    pair_axis = (
        _dimnames.get("interaction_name")
        or _dimnames.get("pairs")
        or [str(i) for i in range(prob.shape[2])]
    )
    pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
    pair_names = [p for p in pairLR.index if p in pair_idx_map]
    if pairLR_use is not None:
        if isinstance(pairLR_use, pd.DataFrame):
            user_names = [str(x) for x in pairLR_use["interaction_name"]]
        else:
            user_names = [str(x) for x in pairLR_use]
        pair_names = [p for p in pair_names if p in set(user_names)]
        if len(pair_names) == 0:
            raise ValueError("There is no significant communication for the input L-R pairs!")
    if len(pair_names) == 0:
        raise ValueError(f"There is no significant communication of {signaling}")
    idx = [pair_idx_map[p] for p in pair_names]
    sums = prob[:, :, idx].sum(axis=(0, 1))
    keep = sums > 0
    if not np.any(keep):
        raise ValueError(f"There is no significant communication of {signaling}")
    pair_names_use = [pair_names[k] for k in np.flatnonzero(keep)]
    idx_use = [pair_idx_map[p] for p in pair_names_use]
    prob_use = prob[:, :, idx_use]
    # "cells" (computeCommunProb) vs "source"/"target" (computeAvgCommunProb)
    _dn2 = net_dict.get("dimnames", {})
    cells = _dn2.get("cells") or _dn2.get("source")
    if cells is None:
        idents = _get_idents(adata, None)
        cells = list(idents.categories)

    if edge_weight_max is None:
        edge_weight_max = float(prob_use.max()) if prob_use.size > 0 else 1.0

    figs: List[Figure] = []
    for i, p in enumerate(pair_names_use):
        df_i = pd.DataFrame(prob_use[:, :, i], index=cells, columns=cells)
        tn = pairLR.loc[p, "interaction_name_2"] if p in pairLR.index and "interaction_name_2" in pairLR.columns else p
        if layout == "circle":
            figs.append(netVisual_circle(
                df_i, color_use=color_use, sources_use=sources_use,
                targets_use=targets_use, remove_isolate=remove_isolate,
                top=top, vertex_weight=vertex_weight,
                vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
                weight_scale=weight_scale, edge_weight_max=edge_weight_max,
                edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
                title_name=tn,
            ))
        elif layout == "hierarchy":
            if vertex_receiver is None:
                raise ValueError("vertex_receiver required for hierarchy layout")
            # R renders BOTH hierarchy1 and hierarchy2 per L-R pair.
            figs.append(_hierarchy_combined(
                net=df_i, vertex_receiver=vertex_receiver,
                sources_use=sources_use, targets_use=targets_use,
                remove_isolate=remove_isolate, top=top, color_use=color_use,
                vertex_weight=vertex_weight, vertex_weight_max=vertex_weight_max,
                vertex_size_max=vertex_size_max, weight_scale=weight_scale,
                edge_weight_max=edge_weight_max, edge_width_max=edge_width_max,
                vertex_label_cex=vertex_label_cex, title_name=str(tn),
            ))
        elif layout == "spatial":
            coords_obs = adata.obsm["spatial"]
            idents = _get_idents(adata, None)
            figs.append(netVisual_spatial(
                df_i, coordinates=coords_obs, labels=list(idents),
                color_use=color_use, sources_use=sources_use, targets_use=targets_use,
                remove_isolate=remove_isolate, top=top, vertex_weight=vertex_weight,
                vertex_weight_max=vertex_weight_max, vertex_size_max=vertex_size_max,
                weight_scale=weight_scale, edge_weight_max=edge_weight_max,
                edge_width_max=edge_width_max, vertex_label_cex=vertex_label_cex,
                alpha_image=alpha_image, point_size=point_size, title_name=tn,
            ))
        elif layout == "chord":
            figs.append(netVisual_chord_cell_internal(
                df_i, color_use=color_use, sources_use=sources_use,
                targets_use=targets_use, remove_isolate=remove_isolate, group=group,
                cell_order=cell_order, lab_cex=vertex_label_cex,
                small_gap=small_gap, big_gap=big_gap, scale=scale, reduce=reduce,
                title_name=tn, show_legend=show_legend,
                legend_pos_x=legend_pos_x, legend_pos_y=legend_pos_y,
            ))
    if layout == "hierarchy":
        # R returns a single recordedplot for hierarchy. Composite per-pair
        # figures into one tall Figure (one row per L-R pair).
        if not figs:
            return plt.figure()
        if len(figs) == 1:
            return figs[0]
        nrows = len(figs)
        composite = plt.figure(figsize=(12.0, 5.0 * nrows))
        for k, f in enumerate(figs):
            # Each per-pair figure has 2 axes (h1, h2). Replay them.
            sub_axes = list(f.axes)
            ax_left = composite.add_subplot(nrows, 2, k * 2 + 1)
            ax_right = composite.add_subplot(nrows, 2, k * 2 + 2)
            for src_ax, dest_ax in zip(sub_axes[:2], (ax_left, ax_right)):
                _replay_axes(src_ax, dest_ax)
            plt.close(f)
        return composite
    return figs
