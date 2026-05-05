"""Basic plotting helpers ported from ``visualization.R`` / ``spatial.R``.

Implements the slice-6a public surface (matplotlib + seaborn). The R
reference is the ground truth for the data flow; the visual aesthetic is
matched as closely as possible without depending on the (unavailable)
ggplot2-Python port.

Public exports
--------------
- :func:`barPlot`              (R: visualization.R:6533)
- :func:`barplot_internal`     (R: visualization.R:6617)
- :func:`netVisual_barplot`    (R: visualization.R:2746)
- :func:`netAnalysis_dot`      (R: visualization.R:4044)
- :func:`vizDimLoadings`       (R: visualization.R:5138)
- :func:`plotFeatures`         (R: visualization.R:7105)
- :func:`plotStatistics_Lee`   (R: visualization.R:8321)
- :func:`plotStatistics_Gi`    (R: visualization.R:8828)
- :func:`savePlotWithCow`      (R: utilities.R:10)
- :func:`communicationDistPlot`  (R: spatial.R:95)
- :func:`communicationDistPlot2` (R: spatial.R:11)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from scipy import sparse

from ._anndata import cellchat_dict as _ensure_uns
from ._palette import scPalette
from ._themes import _BREWER_SUBSET, generate_RedsBlues_colormap
from ._utils import triMean

__all__ = [
    "barPlot",
    "barplot_internal",
    "netVisual_barplot",
    "netAnalysis_dot",
    "pieChart",
    "showDatabaseCategory",
    "vizDimLoadings",
    "plotFeatures",
    "plotStatistics_Lee",
    "plotStatistics_Gi",
    "savePlotWithCow",
    "communicationDistPlot",
    "communicationDistPlot2",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_brewer(name: str, n_colors: int) -> List[str]:
    """Return ``n_colors`` from an RColorBrewer palette, falling back to viridis.

    Mirrors R's ``tryCatch(RColorBrewer::brewer.pal(n, name), ...)`` —
    when the palette name is not in :data:`_BREWER_SUBSET` we fall back to
    sampling :mod:`matplotlib`'s named colormap (mapped to viridis-style).
    """
    if name in _BREWER_SUBSET:
        # _BREWER_SUBSET indexes by the palette anchor count; clamp to range
        max_n = max(_BREWER_SUBSET[name].keys())
        use_n = max(3, min(n_colors, max_n))
        return list(_BREWER_SUBSET[name][use_n])
    cmap = matplotlib.colormaps[name].resampled(n_colors)
    samples = np.linspace(0.0, 1.0, n_colors)
    return [matplotlib.colors.to_hex(cmap(s)) for s in samples]


def _color_ramp_palette(colors: Sequence[str], n: int) -> List[str]:
    """R-style ``colorRampPalette(colors)(n)`` — linear sRGB interpolation."""
    rgb = np.array([to_rgb(c) for c in colors], dtype=float)
    if rgb.shape[0] == 1:
        rgb = np.vstack([rgb, rgb])
    anchors = np.linspace(0.0, 1.0, rgb.shape[0])
    samples = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=float)
    for j in range(3):
        out[:, j] = np.interp(samples, anchors, rgb[:, j])
    return [matplotlib.colors.to_hex(out[i]) for i in range(n)]


def _factor_levels(s: Union[pd.Series, pd.Categorical, Sequence]) -> List[str]:
    """Return categorical levels (mirrors R's ``levels(x)``)."""
    if isinstance(s, pd.Categorical):
        return list(s.categories.astype(str))
    if isinstance(s, pd.Series) and isinstance(s.dtype, pd.CategoricalDtype):
        return list(s.cat.categories.astype(str))
    arr = pd.unique(pd.Series(s))
    return [str(x) for x in arr]


def _apply_xlabel_rot(ax: Axes, x_lab_rot: Union[bool, float]) -> None:
    """Mirror R's ``axis.text.x = element_text(angle=45, hjust=1)``."""
    angle = 0.0
    if isinstance(x_lab_rot, bool):
        if x_lab_rot:
            angle = 45.0
    else:
        angle = float(x_lab_rot)
    if angle != 0.0:
        for lab in ax.get_xticklabels():
            lab.set_rotation(angle)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")


# ---------------------------------------------------------------------------
# barplot_internal (R: visualization.R:6617)
# ---------------------------------------------------------------------------
def barplot_internal(
    df: pd.DataFrame,
    x: str = "cellType",
    y: str = "value",
    fill: str = "condition",
    legend_title: Optional[str] = None,
    width: float = 0.6,
    title_name: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    remove_xtick: bool = False,
    stat_add: bool = False,
    stat_method: str = "wilcox.test",
    percent_y: bool = False,
    label_x: float = 1.5,
    show_legend: bool = True,
    x_lab_rot: Union[bool, float] = False,
    size_text: int = 10,
    ax: Optional[Axes] = None,
) -> Axes:
    """Bar plot for a tidy data frame.

    Direct port of R's :func:`barplot_internal` (``visualization.R:6617``).
    Uses :mod:`matplotlib` to draw side-by-side bars when ``fill`` differs
    from ``x``; otherwise the bars share their fill with the x-axis class.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy data frame with at least three columns (``x``, ``y``, ``fill``).
    x, y, fill : str
        Column names (mirror the R formulae arguments).
    legend_title : str, optional
        Legend title; ``None`` removes it (mirrors R's
        ``theme(legend.title = element_blank())``).
    width : float, default ``0.6``
        Bar width.
    title_name : str, optional
        Plot title (centred, bold, size 10).
    xlabel, ylabel : str, optional
        Axis labels.
    color_use : sequence of str or dict, optional
        Colour palette. If a dict, maps ``fill`` levels to colour. If a
        sequence, paired with ``fill`` levels in order.
    remove_xtick : bool, default ``False``
        If ``True``, hide the x ticks / tick labels / axis title.
    stat_add : bool, default ``False``
        Whether to overlay a Wilcoxon p-value (mirrors R's
        ``ggpubr::stat_compare_means``).
    stat_method : str, default ``"wilcox.test"``
        Statistical test name (only ``"wilcox.test"`` and ``"t.test"`` honoured).
    percent_y : bool, default ``False``
        Whether to format the y-axis as percentages.
    label_x : float, default ``1.5``
        X-axis position used to draw the p-value annotation.
    show_legend : bool, default ``True``
    x_lab_rot : bool or float, default ``False``
        ``True`` rotates by 45 degrees; numeric values rotate by that angle.
    size_text : int, default ``10``
        Text/legend font size.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into. A new figure is created when ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bar plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x_levels = _factor_levels(df[x])
    fill_levels = _factor_levels(df[fill])

    # Build a colour map from `fill_levels` → hex
    color_map: Dict[str, str] = {}
    if color_use is None:
        palette = scPalette(len(fill_levels))
        for k, lvl in enumerate(fill_levels):
            color_map[str(lvl)] = str(palette[k])
    elif isinstance(color_use, dict):
        for lvl in fill_levels:
            color_map[str(lvl)] = str(color_use.get(lvl, "#777777"))
    else:
        cu = list(color_use)
        for k, lvl in enumerate(fill_levels):
            color_map[str(lvl)] = str(cu[k % len(cu)])

    # ``geom_bar(stat="identity", position=position_dodge())`` — side-by-side
    n_fill = max(1, len(fill_levels))
    # R parity (visualization.R:6622):
    #   ``geom_bar(stat="identity", width=width, position=position_dodge())``
    # ggplot2's ``position_dodge()`` with default arguments dodges by
    # ``width`` and lets each bar retain its full ``width`` — bars within
    # one dodge group abut their fill-neighbours rather than auto-shrinking.
    # Audit G7-1 flagged the previous ``width / n_fill`` shrink as a
    # divergence; restore R's no-shrink semantics.
    bar_w = float(width)
    # Dodge stride is ``width`` per ggplot2 default; with N fills the
    # group spans ``N * width`` centred on each x.
    dodge_stride = float(width)
    x_idx = np.arange(len(x_levels), dtype=float)
    for k, fl in enumerate(fill_levels):
        sub = df[df[fill] == fl]
        if sub.empty:
            heights = np.zeros(len(x_levels))
        else:
            agg = sub.groupby(x, observed=False)[y].sum().reindex(x_levels, fill_value=0.0)
            heights = agg.to_numpy(dtype=float)
        offset = (k - (n_fill - 1) / 2.0) * dodge_stride
        ax.bar(
            x_idx + offset,
            heights,
            width=bar_w,
            color=color_map[str(fl)],
            edgecolor=color_map[str(fl)],
            label=str(fl),
        )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([str(v) for v in x_levels])
    ax.set_xlabel(xlabel if xlabel is not None else "")
    ax.set_ylabel(ylabel if ylabel is not None else "")
    if title_name is not None:
        ax.set_title(str(title_name), fontsize=10, fontweight="bold", loc="center")
    ax.tick_params(axis="both", labelsize=size_text)

    # theme_classic
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")

    if remove_xtick:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False)

    if percent_y:
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    if x_lab_rot:
        _apply_xlabel_rot(ax, x_lab_rot)

    if show_legend:
        leg = ax.legend(title=legend_title if legend_title is not None else None,
                        frameon=False, fontsize=size_text)
        if legend_title is None and leg is not None:
            leg.set_title("")
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    if stat_add:
        # Light-weight Wilcoxon (or t-test) annotation across the first two fill levels
        from scipy import stats as _stats

        if len(fill_levels) >= 2:
            a = df[df[fill] == fill_levels[0]][y].to_numpy(dtype=float)
            b = df[df[fill] == fill_levels[1]][y].to_numpy(dtype=float)
            if a.size > 0 and b.size > 0:
                if stat_method == "t.test":
                    pval = float(_stats.ttest_ind(a, b, equal_var=False).pvalue)
                else:
                    pval = float(_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                ax.annotate(f"p={pval:.2g}",
                            xy=(label_x, ax.get_ylim()[1] * 0.95),
                            ha="left", va="top", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# barPlot (R: visualization.R:6533)
# ---------------------------------------------------------------------------
def barPlot(
    adata: ad.AnnData,
    features: Sequence[str],
    group_by: Optional[str] = None,
    split_by: Optional[str] = None,
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    method: Literal["truncatedMean", "triMean", "median"] = "truncatedMean",
    trim: float = 0.1,
    layer: Optional[str] = None,
    x_lab_rot: Union[bool, float] = False,
    ncol: int = 1,
    size_text: int = 10,
    **kwargs: Any,
) -> Figure:
    """Bar plot of average gene expression per cell group.

    Direct port of R's :func:`barPlot` (``visualization.R:6533``). Reads
    counts from ``adata.X`` (or ``adata.layers[layer]``), groups cells by
    ``group_by`` (defaults to ``adata.obs['active_ident']`` or
    ``'cellchat_idents'``), optionally splits by ``split_by``, and for
    each gene in ``features`` plots one bar per group (or per
    group × condition when ``split_by`` is given).

    Parameters
    ----------
    adata : anndata.AnnData
    features : sequence of str
        Gene names to plot. Each gene gets its own subplot.
    group_by, split_by : str, optional
        Column names in ``adata.obs``. ``group_by`` determines the x-axis
        groups; ``split_by`` determines the bar fill (condition).
    color_use : sequence or dict, optional
        Colour palette for ``split_by`` levels (or for ``group_by`` when
        ``split_by`` is ``None``).
    method : {"truncatedMean", "triMean", "median"}, default ``"truncatedMean"``
        Aggregation function applied per gene per group.
    trim : float, default ``0.1``
        Trim parameter for ``truncatedMean``.
    layer : str, optional
        Layer in ``adata.layers`` to read instead of ``adata.X``.
    x_lab_rot : bool or float, default ``False``
    ncol : int, default ``1``
        Number of subplot columns.
    size_text : int, default ``10``
    **kwargs
        Forwarded to :func:`barplot_internal`.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one axis per feature.
    """
    if method not in ("truncatedMean", "triMean", "median"):
        raise ValueError(f"`method` must be truncatedMean / triMean / median, got {method!r}")

    if group_by is None:
        for k in ("cellchat_idents", "active_ident", "cell.type"):
            if k in adata.obs.columns:
                group_by = k
                break
    if group_by is None or group_by not in adata.obs.columns:
        raise ValueError("Could not infer group_by; pass an obs column name explicitly.")

    labels_full = adata.obs[group_by]
    if not isinstance(labels_full.dtype, pd.CategoricalDtype):
        labels_full = pd.Categorical(labels_full)
    else:
        labels_full = labels_full.values  # underlying Categorical

    if method == "truncatedMean":
        from scipy.stats import trim_mean

        def fun_mean(v: np.ndarray) -> float:
            v = v[~np.isnan(v)]
            if v.size == 0:
                return float("nan")
            return float(trim_mean(v, proportiontocut=float(trim)))

    elif method == "triMean":
        def fun_mean(v: np.ndarray) -> float:
            v = v[~np.isnan(v)]
            if v.size == 0:
                return float("nan")
            return float(triMean(v))
    else:
        def fun_mean(v: np.ndarray) -> float:
            v = v[~np.isnan(v)]
            if v.size == 0:
                return float("nan")
            return float(np.median(v))

    var_idx = pd.Index(adata.var_names.astype(str))
    feat_idx = [var_idx.get_loc(f) for f in features]
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    if sparse.issparse(X):
        data_full = np.asarray(X[:, feat_idx].toarray(), dtype=float)
    else:
        data_full = np.asarray(X[:, feat_idx], dtype=float)

    # Build the long-form ``df``
    if split_by is not None:
        if split_by not in adata.obs.columns:
            raise ValueError(f"split_by={split_by!r} not in adata.obs")
        cond_full = adata.obs[split_by]
        if not isinstance(cond_full.dtype, pd.CategoricalDtype):
            cond_full = pd.Categorical(cond_full)
        else:
            cond_full = cond_full.values
        cond_levels = list(cond_full.categories.astype(str))
        rows: List[Dict[str, Any]] = []
        labels_levels = list(labels_full.categories.astype(str))
        for cond in cond_levels:
            mask = (np.asarray(cond_full) == cond)
            for fi, fname in enumerate(features):
                col = data_full[mask, fi]
                lab_sub = np.asarray(labels_full)[mask]
                for lvl in labels_levels:
                    sub = col[lab_sub == lvl]
                    rows.append({
                        "gene": fname, "labels": lvl,
                        "value": fun_mean(sub) if sub.size > 0 else 0.0,
                        "condition": cond,
                    })
        df = pd.DataFrame(rows)
        df["labels"] = pd.Categorical(df["labels"], categories=labels_levels)
        df["condition"] = pd.Categorical(df["condition"], categories=cond_levels)
    else:
        labels_levels = list(labels_full.categories.astype(str))
        rows = []
        for fi, fname in enumerate(features):
            col = data_full[:, fi]
            for lvl in labels_levels:
                sub = col[np.asarray(labels_full) == lvl]
                rows.append({
                    "gene": fname, "labels": lvl,
                    "value": fun_mean(sub) if sub.size > 0 else 0.0,
                    "condition": lvl,
                })
        df = pd.DataFrame(rows)
        df["labels"] = pd.Categorical(df["labels"], categories=labels_levels)
        df["condition"] = pd.Categorical(df["condition"], categories=labels_levels)

    nrow = math.ceil(len(features) / max(int(ncol), 1))
    fig, axes = plt.subplots(nrow, int(ncol), figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
    axes_flat = axes.flatten()
    for i, fname in enumerate(features):
        ax = axes_flat[i]
        df_use = df[df["gene"] == fname]
        remove_xtick = (i < len(features) - 1)
        barplot_internal(
            df_use, x="labels", y="value", fill="condition",
            color_use=color_use, ylabel=str(fname),
            remove_xtick=remove_xtick, x_lab_rot=x_lab_rot,
            size_text=size_text, ax=ax, **kwargs,
        )
    # Hide spare axes
    for j in range(len(features), nrow * int(ncol)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netVisual_barplot (R: visualization.R:2746)
# ---------------------------------------------------------------------------
def netVisual_barplot(
    adata: ad.AnnData,
    comparison: Sequence[int] = (1, 2),
    measure: Literal["count", "weight"] = "count",
    sources_use: Optional[Union[Sequence[str], Sequence[int]]] = None,
    targets_use: Optional[Union[Sequence[str], Sequence[int]]] = None,
    invert_source: bool = False,
    invert_target: bool = False,
    signaling: Optional[str] = None,
    slot_name: Literal["netP", "net"] = "netP",
    color_use: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    x_lab_rot: Union[bool, float] = False,
    **kwargs: Any,
) -> Figure:
    """Bar plot of inferred number / strength of interactions.

    Direct port of R's :func:`netVisual_barplot` (``visualization.R:2746``).
    For a single object this draws ``net['count']`` (or ``net['weight']``,
    or one slice of ``netP['prob'][:, :, signaling]``) summed over
    ``sources_use``. For a "merged" object (where ``net[k]`` is itself a
    list of two named comparisons), the differential is plotted instead.

    Parameters
    ----------
    adata : anndata.AnnData
    comparison : sequence of int, default ``(1, 2)``
        For merged objects, the two indices to subtract.
    measure : {"count", "weight"}
    sources_use, targets_use : sequence, optional
        Subset of source / target groups (names or 1-based positional ints).
    invert_source, invert_target : bool, default ``False``
        When ``True``, take the complement of the subset.
    signaling : str, optional
        For single objects, plot the per-signaling slice of
        ``netP['prob']`` (or ``net['prob']``).
    slot_name : {"netP", "net"}, default ``"netP"``
    color_use : sequence of str, optional
    title_name : str, optional
    x_lab_rot : bool or float
    **kwargs
        Forwarded to :func:`barplot_internal`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_uns(adata)
    if "net" not in cc:
        raise ValueError("netVisual_barplot requires uns['cellchat']['net']")
    net = cc["net"]

    # Detect merged vs. single object: the R guard is
    # ``if (is.list(object@net[[1]]))``. In Python, when net values are
    # themselves dicts indexed by the comparison int, treat as merged.
    merged = isinstance(next(iter(net.values()), None), dict) and \
        all(isinstance(v, dict) for v in net.values())
    if merged:
        idx1, idx2 = int(comparison[0]), int(comparison[1])
        obj1 = net[idx1][measure]
        obj2 = net[idx2][measure]
        net_diff = np.asarray(obj2, dtype=float) - np.asarray(obj1, dtype=float)
        if title_name is None:
            title_name = ("Differential number of interactions" if measure == "count"
                          else "Differential interaction strength")
    else:
        if signaling is not None:
            slot = cc[slot_name]
            prob = np.asarray(slot["prob"], dtype=float)
            pair_names = list(slot.get("dimnames", [None, None, list(range(prob.shape[2]))])[2])
            if signaling not in pair_names:
                raise ValueError(f"signaling {signaling!r} not in slot {slot_name!r} dimnames")
            net_diff = prob[:, :, pair_names.index(signaling)]
            if title_name is None:
                title_name = f"{signaling} signaling network"
        else:
            net_diff = np.asarray(net[measure], dtype=float)
            if title_name is None:
                title_name = ("Number of interactions" if measure == "count"
                              else "Interaction strength")

    # Resolve cell-group level names
    meta = net.get("meta", {})
    cells_level = list(meta.get("groups", None) or [str(k) for k in range(net_diff.shape[0])])

    # Build the long form
    df_net = pd.DataFrame(
        net_diff,
        index=pd.Index(cells_level, name="source"),
        columns=pd.Index(cells_level, name="target"),
    ).reset_index().melt(id_vars="source", var_name="target", value_name="value")

    if sources_use is not None:
        srcs = list(sources_use)
        if all(isinstance(x, (int, np.integer)) for x in srcs):
            srcs = [cells_level[int(i) - 1] for i in srcs]
        if invert_source:
            srcs = [c for c in cells_level if c not in srcs]
        df_net = df_net[df_net["source"].isin(srcs)]
    if targets_use is not None:
        tgts = list(targets_use)
        if all(isinstance(x, (int, np.integer)) for x in tgts):
            tgts = [cells_level[int(i) - 1] for i in tgts]
        if invert_target:
            tgts = [c for c in cells_level if c not in tgts]
        df_net = df_net[df_net["target"].isin(tgts)]

    src_present = [c for c in cells_level if c in set(df_net["source"])]
    tgt_present = [c for c in cells_level if c in set(df_net["target"])]
    df_net["source"] = pd.Categorical(df_net["source"], categories=src_present)
    df_net["target"] = pd.Categorical(df_net["target"], categories=tgt_present)

    if color_use is None:
        full = scPalette(len(cells_level))
        color_use = {cells_level[k]: str(full[k]) for k in range(len(cells_level))}
        color_use = {c: color_use[c] for c in tgt_present if c in color_use}

    fig, ax = plt.subplots()
    barplot_internal(
        df_net, x="target", y="value", fill="target",
        color_use=color_use, title_name=title_name,
        x_lab_rot=x_lab_rot, ax=ax, **kwargs,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_dot (R: visualization.R:4044)
# ---------------------------------------------------------------------------
def netAnalysis_dot(
    adata: ad.AnnData,
    slot_name: str = "netP",
    pattern: Literal["outgoing", "incoming"] = "outgoing",
    cutoff: Optional[float] = None,
    color_use: Optional[Sequence[str]] = None,
    pathway_show: Optional[Sequence[str]] = None,
    group_show: Optional[Sequence[str]] = None,
    shape: str = "o",
    dot_size: Tuple[float, float] = (1.0, 3.0),
    dot_alpha: float = 1.0,
    main_title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 12,
) -> Figure:
    """Dot plot of NMF communication patterns.

    Direct port of R's :func:`netAnalysis_dot` (``visualization.R:4044``).
    Reads ``adata.uns['cellchat'][slot_name]['pattern'][pattern]`` written
    by :func:`identifyCommunicationPatterns`. The plot has cell groups on
    the y-axis and signaling pathways on the x-axis; dot size encodes the
    product ``W[cell, pattern] * H[pattern, signaling]`` for the highest-
    contributing pattern per (cell, signaling) cell.

    Parameters
    ----------
    adata : anndata.AnnData
    slot_name : str, default ``"netP"``
    pattern : {"outgoing", "incoming"}, default ``"outgoing"``
    cutoff : float, optional
        Minimum contribution for a dot to be shown. Default
        ``1 / nPatterns`` (mirrors R).
    color_use : sequence of str, optional
        One colour per cell group.
    pathway_show, group_show : sequence of str, optional
        Subset / order x-axis pathways or y-axis cell groups.
    shape : str, default ``"o"``
        Marker shape (matplotlib spec; mirrors R's ``shape = 21``).
    dot_size : tuple of (min, max), default ``(1, 3)``
        Min/max dot size in points² mapping (passed to ``scatter``).
    dot_alpha : float, default ``1.0``
    main_title : str, optional
    font_size, font_size_title : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pattern not in ("outgoing", "incoming"):
        raise ValueError("`pattern` must be 'outgoing' or 'incoming'")
    cc = _ensure_uns(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat'][{slot_name!r}] missing")
    slot = cc[slot_name]
    if "pattern" not in slot or pattern not in slot["pattern"]:
        raise ValueError("Run identifyCommunicationPatterns first to populate the slot")
    patternSignaling = slot["pattern"][pattern]
    data1: pd.DataFrame = patternSignaling["pattern"]["cell"].copy()
    data2: pd.DataFrame = patternSignaling["pattern"]["signaling"].copy()
    data: np.ndarray = np.asarray(patternSignaling["data"], dtype=float)

    if main_title is None:
        if pattern == "outgoing":
            main_title = "Outgoing communication patterns of secreting cells"
        else:
            main_title = "Incoming communication patterns of target cells"

    # cell-group levels = unique data1.CellGroup in original order
    cg_levels = list(pd.Categorical(data1["CellGroup"]).categories.astype(str))
    if color_use is None:
        color_use = list(scPalette(len(cg_levels)))

    n_pat = data1["Pattern"].nunique()
    if cutoff is None:
        cutoff = 1.0 / max(int(n_pat), 1)

    data1.loc[data1["Contribution"] < cutoff, "Contribution"] = 0.0
    data2.loc[data2["Contribution"] < cutoff, "Contribution"] = 0.0

    data3 = data1.merge(data2, on="Pattern", suffixes=(".x", ".y"))
    data3["Contribution"] = data3["Contribution.x"] * data3["Contribution.y"]
    data3 = data3[["CellGroup", "Signaling", "Contribution"]]

    if pathway_show is not None:
        data3 = data3[data3["Signaling"].isin(pathway_show)].copy()
        pathway_add = [p for p in pathway_show if p not in set(data3["Signaling"])]
        if len(pathway_add) > 1:
            extra_rows = []
            for c in cg_levels:
                for p in pathway_add:
                    extra_rows.append({"CellGroup": c, "Signaling": p, "Contribution": 0.0})
            data3 = pd.concat([data3, pd.DataFrame(extra_rows)], ignore_index=True)
        data3["Signaling"] = pd.Categorical(data3["Signaling"], categories=list(pathway_show))
    if group_show is not None:
        data3 = data3[data3["CellGroup"].astype(str).isin(group_show)].copy()
        data3["CellGroup"] = pd.Categorical(data3["CellGroup"], categories=list(group_show))
        cg_levels = list(group_show)
    else:
        data3["CellGroup"] = pd.Categorical(data3["CellGroup"], categories=cg_levels)

    # zero out (cell, signaling) pairs where the underlying `data` is zero
    cells_kept = patternSignaling.get("cells_kept", cg_levels)
    pathway_cols = list(_factor_levels(data2["Signaling"]))
    nz_pairs: set = set()
    for i, ckept in enumerate(cells_kept):
        for j, pname in enumerate(pathway_cols):
            if i < data.shape[0] and j < data.shape[1] and data[i, j] != 0:
                nz_pairs.add((str(ckept), str(pname)))
    data3["id"] = data3["CellGroup"].astype(str) + "|" + data3["Signaling"].astype(str)
    keep_mask = data3.apply(
        lambda r: (str(r["CellGroup"]), str(r["Signaling"])) in nz_pairs,
        axis=1,
    )
    data3.loc[~keep_mask, "Contribution"] = 0.0
    # collapse to top-1 per id (R: ``group_by(id) %>% top_n(1, Contribution)``)
    data3 = data3.sort_values("Contribution", ascending=False).drop_duplicates(subset="id")
    data3.loc[data3["Contribution"] == 0.0, "Contribution"] = float("nan")

    # Build axes
    fig, ax = plt.subplots()
    sig_levels = list(_factor_levels(data3["Signaling"]))
    cg_levels_rev = list(reversed(cg_levels))  # R: ``scale_y_discrete(limits = rev(...))``
    cg_to_y = {c: idx for idx, c in enumerate(cg_levels_rev)}
    sig_to_x = {s: idx for idx, s in enumerate(sig_levels)}
    color_map = {c: color_use[i % len(color_use)] for i, c in enumerate(cg_levels)}

    valid = data3.dropna(subset=["Contribution"])
    if not valid.empty:
        contrib = valid["Contribution"].astype(float).to_numpy()
        c_min, c_max = float(contrib.min()), float(contrib.max())
        if c_max == c_min:
            sizes = np.full_like(contrib, fill_value=(dot_size[0] + dot_size[1]) / 2.0 * 20.0)
        else:
            sizes = (
                dot_size[0] + (dot_size[1] - dot_size[0]) * (contrib - c_min) / (c_max - c_min)
            ) * 20.0
        xs = [sig_to_x[str(s)] for s in valid["Signaling"]]
        ys = [cg_to_y[str(c)] for c in valid["CellGroup"]]
        cols = [color_map.get(str(c), "#777777") for c in valid["CellGroup"]]
        ax.scatter(xs, ys, s=sizes, c=cols, marker=shape, alpha=float(dot_alpha),
                   edgecolors=cols)

    ax.set_xticks(np.arange(len(sig_levels)))
    ax.set_xticklabels(sig_levels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(cg_levels_rev)))
    ax.set_yticklabels(cg_levels_rev)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(str(main_title), fontsize=font_size_title)
    ax.tick_params(axis="both", labelsize=font_size)
    ax.grid(True, color="grey", alpha=0.3, linewidth=0.1)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# vizDimLoadings (R: visualization.R:5138)
# ---------------------------------------------------------------------------
def vizDimLoadings(
    adata: ad.AnnData,
    dims: Sequence[int] = tuple(range(1, 6)),
    nfeatures: int = 30,
    nTop: int = 10,
    feature_cat: Literal["LR-pair", "ligand", "signaling"] = "LR-pair",
    col: Optional[Dict[str, str]] = None,
    reduction: str = "pca",
    projected: bool = False,
    balanced: bool = False,
    species: Literal["mouse", "human"] = "mouse",
    db: Optional[Dict[str, pd.DataFrame]] = None,
    ncol: Optional[int] = None,
    combine: bool = True,
    ylabel: str = "Signaling score",
    text_nudge_x: float = -8.0,
    text_size: float = 3.0,
    point_size: float = 2.0,
    line_size: float = 1.0,
) -> Figure:
    """Plot top features per PC dimension.

    Direct port of R's :func:`vizDimLoadings` (``visualization.R:5138``).
    Reads PCA loadings from ``adata.varm[f"{reduction.upper()}s"]`` (or
    ``adata.uns['cellchat']['pca']['loadings']`` if present) and ranks
    features by absolute loading; the top ``nTop`` are labelled.

    Parameters
    ----------
    adata : anndata.AnnData
    dims : sequence of int, default ``(1, 2, 3, 4, 5)``
        1-based PC indices.
    nfeatures, nTop : int
    feature_cat : {"LR-pair", "ligand", "signaling"}
    col : dict, optional
        Mapping ``{"neg": "#377EB8", "pos": "#E41A1C"}``.
    reduction : str, default ``"pca"``
    projected, balanced : bool
    species : {"mouse", "human"}
    db : dict, optional
        CellChat database (with key ``"interaction"``).
    ncol : int, optional
        Number of subplot columns (auto-derived from ``len(dims)`` per R).
    combine : bool, default ``True``
    ylabel : str
    text_nudge_x, text_size, point_size, line_size : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    if col is None:
        col = {"neg": "#377EB8", "pos": "#E41A1C"}
    if feature_cat not in ("LR-pair", "ligand", "signaling"):
        raise ValueError("invalid feature_cat")

    if ncol is None:
        ncol = 2
        if len(dims) == 1:
            ncol = 1
        if len(dims) > 6:
            ncol = 3
        if len(dims) > 9:
            ncol = 4

    # Pull loadings (var x PC).  Fall back to a flat ``uns`` slot when the
    # AnnData has no ``varm`` PCA results.  When ``projected=True``,
    # mirror Seurat's ``Loadings(object, projected=TRUE)`` by reading from
    # a parallel slot — ``varm[f"{key}_projected"]`` (parallels scanpy's
    # ``X_pca`` / ``X_pca_projected`` convention) or
    # ``cc[reduction]["loadings_projected"]`` for the uns-fallback path.
    loadings: Optional[np.ndarray] = None
    var_names: Optional[List[str]] = None
    # scanpy convention: varm['PCs']; some R-style ports use 'PCAs'.
    key_candidates = [
        f"{reduction.upper()}s",   # 'PCAs' for reduction='pca'
        f"{reduction[:-1].upper()}s" if reduction.endswith("a") else None,
        "PCs",
    ]
    key = None
    for cand in key_candidates:
        if cand is not None and cand in adata.varm:
            key = cand
            break
    if key is not None:
        proj_key = f"{key}_projected"
        if projected:
            if proj_key not in adata.varm:
                raise ValueError(
                    f"vizDimLoadings(projected=True): expected projected "
                    f"loadings at adata.varm[{proj_key!r}] (parallels "
                    f"adata.varm[{key!r}]); not found."
                )
            loadings_full = np.asarray(adata.varm[proj_key], dtype=float)
        else:
            loadings_full = np.asarray(adata.varm[key], dtype=float)
        var_names = list(adata.var_names.astype(str))
    else:
        cc = _ensure_uns(adata)
        slot = cc.get(reduction)
        if slot is None or "loadings" not in slot:
            raise ValueError(
                "vizDimLoadings: no loadings found at adata.varm or "
                f"adata.uns['cellchat'][{reduction!r}]['loadings']"
            )
        if projected:
            if "loadings_projected" not in slot:
                raise ValueError(
                    f"vizDimLoadings(projected=True): expected "
                    f"adata.uns['cellchat'][{reduction!r}]"
                    f"['loadings_projected']; not found."
                )
            loadings_full = np.asarray(slot["loadings_projected"], dtype=float)
        else:
            loadings_full = np.asarray(slot["loadings"], dtype=float)
        var_names = list(slot.get(
            "feature_names", [f"f{i}" for i in range(loadings_full.shape[0])]
        ))

    # Build per-dim feature lists.  Mirror Seurat::TopFeatures:
    #   * balanced=FALSE: top ``nfeatures`` by ``|score|`` (current default).
    #   * balanced=TRUE : top ``nfeatures`` by +score AND top ``nfeatures``
    #                     by -score concatenated (R returns 2*nfeatures
    #                     features per PC).
    dim_idx = [int(d) - 1 for d in dims]
    if any(d < 0 or d >= loadings_full.shape[1] for d in dim_idx):
        raise ValueError("`dims` out of range of available loadings")

    db_use = None
    if feature_cat in ("ligand", "signaling"):
        if db is None:
            from .database import load_db

            db_full = load_db(species)
        else:
            db_full = db
        db_use = db_full["interaction"]

    nrow = math.ceil(len(dims) / max(int(ncol), 1))
    fig, axes = plt.subplots(nrow, int(ncol), figsize=(4.0 * ncol, 3.0 * nrow), squeeze=False)
    axes_flat = axes.flatten()

    for plot_i, d_i in enumerate(dim_idx):
        scores = loadings_full[:, d_i]
        if balanced:
            # Top ``nfeatures`` by +score AND top ``nfeatures`` by -score
            # (R: TopFeatures(..., balanced=TRUE) → list(positive,negative)
            #  → unlist → 2*nfeatures features per PC).
            asc = np.argsort(scores)               # most negative first
            desc = np.argsort(-scores)             # most positive first
            n_each = int(nfeatures)
            negative = list(asc[:n_each])
            positive = list(desc[:n_each])
            # De-duplicate while preserving order: positive first, then
            # negatives that aren't already present (Seurat returns the
            # union for 2-colour scoring).
            seen: set = set()
            kept_list: list = []
            for j in positive + negative:
                if j not in seen:
                    seen.add(j)
                    kept_list.append(j)
            kept = np.asarray(kept_list, dtype=int)
        else:
            kept = np.argsort(-np.abs(scores))[: int(nfeatures)]
        df = pd.DataFrame({
            "feature": [var_names[i].replace("-", "_") for i in kept],
            "score": scores[kept],
        })
        if feature_cat in ("ligand", "signaling") and db_use is not None:
            df_join = db_use.reindex(df["feature"].values)
            df = df.assign(
                ligand=df_join["ligand"].values if "ligand" in df_join.columns else None,
                receptor=df_join["receptor"].values if "receptor" in df_join.columns else None,
                pathway_name=df_join["pathway_name"].values if "pathway_name" in df_join.columns else None,
            )
            if feature_cat == "ligand":
                data_plot = df.groupby("ligand", as_index=False)["score"].mean()
                data_plot.columns = ["feature", "score"]
            else:
                data_plot = df.groupby("pathway_name", as_index=False)["score"].mean()
                data_plot.columns = ["feature", "score"]
        else:
            data_plot = df

        data_plot = data_plot.assign(
            positive=np.where(data_plot["score"].values > 0, "pos", "neg"),
            absolute_v=np.abs(data_plot["score"].values),
        )
        data_plot = data_plot.sort_values("absolute_v", ascending=False).reset_index(drop=True)
        data_plot["rank"] = np.arange(1, len(data_plot) + 1)
        data_top = data_plot.head(int(nTop))

        ax = axes_flat[plot_i]
        ax.plot(data_plot["rank"], data_plot["absolute_v"],
                color="grey", alpha=0.6, linewidth=line_size)
        for _, row in data_top.iterrows():
            color = col["pos"] if row["positive"] == "pos" else col["neg"]
            ax.scatter(row["rank"], row["absolute_v"],
                       s=point_size * 8, marker="o",
                       facecolors="none", edgecolors=color)
            ax.annotate(
                str(row["feature"]),
                xy=(row["rank"], row["absolute_v"]),
                xytext=(row["rank"] + text_nudge_x, row["absolute_v"]),
                fontsize=int(text_size) + 4,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.3),
            )
        ax.set_xlabel("Ranking")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{reduction.upper()}_{int(dims[plot_i])}", fontsize=10)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(dims), nrow * int(ncol)):
        axes_flat[j].set_visible(False)
    if combine:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# pieChart (R: database.R:49) + showDatabaseCategory (R: database.R:10)
# ---------------------------------------------------------------------------
def pieChart(
    df: pd.DataFrame,
    label_size: float = 2.5,
    color_use: Optional[Sequence[str]] = None,
    title: str = "",
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot a pie chart with percent labels.

    Direct port of R's :func:`pieChart` (``database.R:49``). The R version
    uses ``ggplot2`` ``geom_bar(stat="identity") + coord_polar()`` with
    ``ggrepel`` percent labels; the Python port renders an equivalent
    matplotlib pie via :func:`matplotlib.pyplot.pie`.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least the columns ``group`` and ``value``. The
        wedge order matches ``df`` after sorting by descending ``value``
        (mirrors R ``arrange(desc(value))``).
    label_size : float, default ``2.5``
        Font size for the percent labels (R's ``label.size``).
    color_use : sequence of str, optional
        Colours to override the default matplotlib palette (R's
        ``color.use``).
    title : str, default ``""``
        Legend title (R's ``title``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto; a new figure is created when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the pie.
    """
    df = df.copy()
    # arrange(desc(value))
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    total = float(df["value"].sum())
    # scales::percent — R defaults to 1% accuracy.
    df["prop"] = [f"{(v / total * 100):.0f}%" for v in df["value"].astype(float)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
    else:
        fig = ax.figure

    groups = df["group"].astype(str).tolist()
    values = df["value"].astype(float).tolist()
    labels = df["prop"].tolist()

    if color_use is not None:
        colors = list(color_use)[: len(groups)]
    else:
        colors = scPalette(len(groups))

    # R parity (visualization.R:6242): ``ggrepel::geom_text_repel(aes(label
    # = prop), size = label.size, position = position_stack(vjust = 0.5))``.
    # Compute the wedge midpoint (R: ``position_stack(vjust=0.5)``) for
    # each label, then route through :func:`spatialcellchat._text_repel.repel_text`
    # so labels avoid each other (R parity for the ggrepel call).
    wedges, _texts = ax.pie(
        values,
        colors=colors,
        labels=None,         # we draw labels via repel_text below
        startangle=90,
        counterclock=False,
    )
    # Wedge mid-angle (radians, mathematical convention from matplotlib).
    angles_deg = []
    cum = 0.0
    total = float(sum(values)) or 1.0
    for v in values:
        # matplotlib startangle=90, counterclock=False → wedges go clockwise
        # from the top. Mid-angle = startangle - (cum + v/2)/total * 360.
        mid = 90.0 - (cum + v / 2.0) / total * 360.0
        angles_deg.append(mid)
        cum += v
    angles_rad = np.deg2rad(angles_deg)
    # Stack midpoint at 0.5 of radius (R: ``position_stack(vjust=0.5)``).
    r_label = 0.5
    label_x = r_label * np.cos(angles_rad)
    label_y = r_label * np.sin(angles_rad)
    from ._text_repel import repel_text

    repel_text(
        ax,
        label_x, label_y,
        [str(s) for s in labels],
        fontsize=float(label_size) * 4.0,
        color="black",
    )
    ax.set_aspect("equal")
    # legend at the bottom (R: legend.position="bottom", direction="vertical")
    ax.legend(
        wedges,
        groups,
        title=title if title else None,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
    )
    return fig


def showDatabaseCategory(
    CellChatDB: Dict[str, pd.DataFrame],
    nrow: int = 1,
) -> Figure:
    """Show the description of a CellChatDB database as three pies.

    Direct port of R's :func:`showDatabaseCategory` (``database.R:10``).
    Renders three pies ((i) annotation breakdown, (ii) heterodimer split,
    (iii) KEGG vs Literature evidence) arranged via
    :func:`matplotlib.pyplot.subplots`.

    Parameters
    ----------
    CellChatDB : dict
        CellChatDB dict with at least ``"interaction"`` and ``"geneInfo"``.
    nrow : int, default ``1``
        Number of rows for the subplot grid (R's ``nrow``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    interaction_input = CellChatDB["interaction"]
    geneInfo = CellChatDB["geneInfo"]

    # Pie 1: annotation — R parity (visualization.R:6201):
    #   ``factor(df$annotation, levels = c("Secreted Signaling",
    #            "ECM-Receptor", "Cell-Cell Contact"))``
    # R only enumerates these three levels; "Non-protein Signaling" rows
    # are not shown in the pie even when present in the DB. Audit G7-4
    # flagged the previous 4-category enumeration as a divergence.
    ann_levels = (
        "Secreted Signaling",
        "ECM-Receptor",
        "Cell-Cell Contact",
    )
    ann_counts = interaction_input.groupby("annotation").size().to_dict()
    df1 = pd.DataFrame(
        {
            "group": [lv for lv in ann_levels if lv in ann_counts],
            "value": [ann_counts[lv] for lv in ann_levels if lv in ann_counts],
        }
    )

    # Pie 2: heterodimers (R: ligand AND receptor are both single Symbol entries)
    sym_set = set(geneInfo["Symbol"].astype(str).tolist())
    binary = interaction_input["ligand"].astype(str).isin(sym_set) & (
        interaction_input["receptor"].astype(str).isin(sym_set)
    )
    # R rule: default = "Heterodimers"; flip to "Others" when binary == TRUE
    grp2 = np.where(binary.to_numpy(), "Others", "Heterodimers")
    g2_counts = pd.Series(grp2).value_counts().to_dict()
    df2 = pd.DataFrame(
        {
            "group": [lv for lv in ("Heterodimers", "Others") if lv in g2_counts],
            "value": [g2_counts[lv] for lv in ("Heterodimers", "Others") if lv in g2_counts],
        }
    )

    # Pie 3: evidence (R: grepl("KEGG", evidence))
    evidence = interaction_input.get("evidence", pd.Series([""] * len(interaction_input)))
    kegg_mask = evidence.astype(str).str.contains("KEGG", na=False)
    grp3 = np.where(kegg_mask.to_numpy(), "KEGG", "Literature")
    g3_counts = pd.Series(grp3).value_counts().to_dict()
    df3 = pd.DataFrame(
        {
            "group": [lv for lv in ("KEGG", "Literature") if lv in g3_counts],
            "value": [g3_counts[lv] for lv in ("KEGG", "Literature") if lv in g3_counts],
        }
    )

    ncol = math.ceil(3 / max(int(nrow), 1))
    fig, axes = plt.subplots(int(nrow), ncol, figsize=(4.0 * ncol, 4.0 * nrow), squeeze=False)
    axes_flat = axes.flatten()
    pieChart(df1, ax=axes_flat[0])
    pieChart(df2, ax=axes_flat[1])
    pieChart(df3, ax=axes_flat[2])
    for j in range(3, nrow * ncol):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plotFeatures (R: visualization.R:7105)
# ---------------------------------------------------------------------------
def plotFeatures(
    df_score: pd.DataFrame,
    group_by: Optional[Sequence] = None,
    group_highlight: Optional[Any] = None,
    show_group: bool = False,
    cells_highlight: Optional[Union[Sequence[str], Sequence[int]]] = None,
    method: Literal["normal", "3d", "2d"] = "normal",
    plot_title: Optional[str] = None,
    plot_ncol: int = 2,
    point_size: float = 3.0,
    highlight_alpha: float = 1.0,
    nohighlight_alpha: float = 0.65,
    show_colorscale: bool = True,
    color_heatmap: Union[str, Sequence[str]] = "Blues",
    color_0: Optional[str] = None,
    color_NA: str = "#E5E5E5",
    n_colors: int = 8,
    direction: int = 1,
    color_use: Optional[Union[Sequence[str], Dict[str, str]]] = None,
    legend_group: str = "score",
    legend_group_title: str = "score's value",
    scene_name: str = "scene1",
    colorbar_y: float = 0.5,
    three_dim_z: float = 0.0,
) -> Figure:
    """Spot-level scatter plot coloured by a continuous score.

    Direct port of R's :func:`plotFeatures` (``visualization.R:7105``).
    The R reference dispatches on ``method`` to either ggplot2 (default,
    ``"normal"``) or ``plotly`` (``"2d"`` / ``"3d"``). The Python port
    renders all three with matplotlib (the ``"3d"`` branch uses a
    flat-Z 2-D projection, mirroring how ``plot_ly``'s scatter3d looks
    when the z-coordinate is constant).

    Parameters
    ----------
    df_score : pandas.DataFrame
        Three-column data frame ``(x, y, score)``. The index gives cell ids.
    group_by : sequence, optional
        Cell-group labels (one per row) for the dim-plot panel.
    group_highlight : Any, optional
        Group level to highlight when ``group_by`` is provided.
    show_group : bool, default ``False``
        If ``True`` returns a side-by-side dim-plot + score-plot Figure.
    cells_highlight : sequence of str / int, optional
    method : {"normal", "3d", "2d"}, default ``"normal"``
    plot_title : str, optional
    plot_ncol : int, default ``2``
    point_size, highlight_alpha, nohighlight_alpha : float
    show_colorscale : bool
    color_heatmap : str or sequence of str
    color_0, color_NA : str, optional
    n_colors : int
    direction : int (-1 or 1)
    color_use : sequence or dict, optional
        Cell-group palette for the dim-plot panel.
    legend_group, legend_group_title, scene_name : str
        Carried for parity with the R signature.
    colorbar_y, three_dim_z : float

    Returns
    -------
    matplotlib.figure.Figure
    """
    if method not in ("normal", "3d", "2d"):
        raise ValueError("`method` must be 'normal', '3d', or '2d'")

    df = df_score.copy()
    df.columns = ["x", "y", "score"]

    # cells_highlight branch
    spot_highlight: Optional[np.ndarray] = None
    if cells_highlight is not None:
        if isinstance(cells_highlight[0], (str, np.str_)) and df.index is not None:
            mask = df.index.isin(list(cells_highlight))
        else:
            sel = set(int(i) for i in cells_highlight)
            mask = np.array([(i in sel) for i in range(len(df))])
        if method == "normal":
            spot_highlight = np.where(mask, highlight_alpha, nohighlight_alpha)
        else:
            spot_highlight = mask
    if (group_highlight is not None) and (group_by is not None):
        gb = np.asarray(group_by)
        mask = (gb == group_highlight)
        if method == "normal":
            spot_highlight = np.where(mask, highlight_alpha, nohighlight_alpha)
        else:
            spot_highlight = mask

    # Resolve the colormap
    if isinstance(color_heatmap, str):
        if color_heatmap == "RedsBlues":
            switch_RB = (direction == -1)
            mid = "#E5E5E5" if (color_0 is None or color_0 == "default") else color_0
            colormap_hex = generate_RedsBlues_colormap(
                df["score"].to_numpy(dtype=float),
                mid_color=mid, switch_RedsBlues=switch_RB,
            )
            colormap_list = [str(c)[:7] for c in colormap_hex]  # strip alpha if present
        else:
            anchors = _resolve_brewer(color_heatmap, n_colors)
            if direction == -1:
                anchors = list(reversed(anchors))
            colormap_list = _color_ramp_palette(anchors, 99)
            if color_0 is not None:
                colormap_list[0] = "#E5E5E5" if color_0 == "default" else color_0
    else:
        colormap_list = list(color_heatmap)
        if direction == -1:
            colormap_list = list(reversed(colormap_list))

    cmap = LinearSegmentedColormap.from_list("plotFeatures", colormap_list, N=99)

    # Score colour normalisation
    score_arr = df["score"].to_numpy(dtype=float)
    finite_mask = np.isfinite(score_arr) & ~np.isnan(score_arr)
    if finite_mask.sum() == 0:
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        v_min = float(score_arr[finite_mask].min())
        v_max = float(score_arr[finite_mask].max())
        if v_min == v_max:
            v_max = v_min + 1.0
        norm = Normalize(vmin=v_min, vmax=v_max)

    if not show_group:
        fig, ax = plt.subplots()
        # alpha handling
        alphas = (
            spot_highlight if (spot_highlight is not None and method == "normal")
            else np.ones(len(df))
        )
        # NA mask
        score_face = np.where(finite_mask, score_arr, np.nan)
        sc = ax.scatter(
            df["x"], df["y"], c=score_face, cmap=cmap, norm=norm,
            s=point_size * 10, alpha=alphas,
        )
        if (~finite_mask).any():
            ax.scatter(df["x"][~finite_mask], df["y"][~finite_mask],
                       c=color_NA, s=point_size * 10)
        if show_colorscale:
            cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
            cb.outline.set_visible(False)
        if plot_title is not None:
            ax.set_title(str(plot_title), fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        return fig

    # show_group branch — two side-by-side panels
    if group_by is None:
        raise ValueError("show_group=True requires group_by")
    gb = pd.Categorical(group_by)
    cg_levels = list(gb.categories.astype(str))
    if color_use is None:
        cm = scPalette(len(cg_levels))
        color_use = {cg_levels[k]: str(cm[k]) for k in range(len(cg_levels))}
    elif isinstance(color_use, dict):
        color_use = {str(k): str(v) for k, v in color_use.items()}
    else:
        cu = list(color_use)
        color_use = {cg_levels[k]: str(cu[k % len(cu)]) for k in range(len(cg_levels))}

    fig, (ax_dim, ax_feat) = plt.subplots(1, max(int(plot_ncol), 2),
                                          figsize=(4.0 * max(int(plot_ncol), 2), 4.0))
    alphas = (spot_highlight if (spot_highlight is not None and method == "normal")
              else np.ones(len(df)))
    # Dim plot
    for lvl in cg_levels:
        sel = (np.asarray(gb) == lvl)
        if not sel.any():
            continue
        ax_dim.scatter(df["x"][sel], df["y"][sel],
                       c=color_use.get(str(lvl), "#777777"),
                       s=point_size * 10,
                       alpha=alphas[sel])
    ax_dim.set_title("Cell annotation", fontsize=10, fontweight="bold")
    ax_dim.invert_yaxis()
    ax_dim.set_aspect("equal")
    ax_dim.set_xticks([])
    ax_dim.set_yticks([])
    for s in ax_dim.spines.values():
        s.set_visible(False)
    handles = [
        mpatches.Patch(color=color_use.get(str(lvl), "#777777"), label=str(lvl))
        for lvl in cg_levels
    ]
    ax_dim.legend(handles=handles, loc="best", frameon=False, fontsize=8)

    # Feature plot
    score_face = np.where(finite_mask, score_arr, np.nan)
    sc = ax_feat.scatter(df["x"], df["y"], c=score_face,
                         cmap=cmap, norm=norm,
                         s=point_size * 10, alpha=alphas)
    if (~finite_mask).any():
        ax_feat.scatter(df["x"][~finite_mask], df["y"][~finite_mask],
                        c=color_NA, s=point_size * 10)
    if show_colorscale:
        cb = fig.colorbar(sc, ax=ax_feat, fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
    if plot_title is not None:
        ax_feat.set_title(str(plot_title), fontsize=10, fontweight="bold")
    ax_feat.invert_yaxis()
    ax_feat.set_aspect("equal")
    ax_feat.set_xticks([])
    ax_feat.set_yticks([])
    for s in ax_feat.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plotStatistics_Lee (R: visualization.R:8321)
# ---------------------------------------------------------------------------
def _coerce_label_matrix(
    labels: Union[np.ndarray, pd.Series, sparse.spmatrix],
    fallback_name: str,
) -> Tuple[np.ndarray, pd.Categorical, str]:
    """Mirror R's branch: matrix -> as-is; numeric -> 1-col; else model.matrix.

    Returns ``(label_mat, label_levels, mode)`` where ``mode`` is either
    ``"heatmap"`` (the input was already categorical / matrix-like and
    therefore the function should default to a heatmap) or ``"dot"``
    (the input was a numeric vector — R forces ``type = "dot"``).
    """
    if isinstance(labels, sparse.spmatrix):
        mat = labels.toarray().astype(float)
        cat = pd.Categorical([f"col{i}" for i in range(mat.shape[1])])
        return mat, cat, "heatmap"
    if isinstance(labels, pd.DataFrame):
        cat = pd.Categorical(list(labels.columns.astype(str)),
                             categories=list(labels.columns.astype(str)))
        return labels.to_numpy(dtype=float), cat, "heatmap"
    arr = np.asarray(labels)
    if arr.ndim == 2:
        # already a matrix
        names = [f"col{i}" for i in range(arr.shape[1])]
        cat = pd.Categorical(names, categories=names)
        return arr.astype(float), cat, "heatmap"
    if np.issubdtype(arr.dtype, np.number):
        names = [fallback_name]
        cat = pd.Categorical(names, categories=names)
        return arr.astype(float).reshape(-1, 1), cat, "dot"
    # categorical / string vector → one-hot model.matrix(~x-1)
    if isinstance(labels, pd.Series) and isinstance(labels.dtype, pd.CategoricalDtype):
        cat = labels.values
    else:
        cat = pd.Categorical(arr)
    levels = list(cat.categories.astype(str))
    mat = np.zeros((len(cat), len(levels)), dtype=float)
    for j, lvl in enumerate(levels):
        mat[:, j] = (np.asarray(cat) == lvl).astype(float)
    return mat, pd.Categorical(levels, categories=levels), "heatmap"


def plotStatistics_Lee(
    x_labels: Union[np.ndarray, pd.Series, sparse.spmatrix],
    y_labels: Union[np.ndarray, pd.Series, sparse.spmatrix],
    coords: Optional[np.ndarray] = None,
    lw: Optional[Any] = None,
    x_levels: Optional[Sequence[str]] = None,
    y_levels: Optional[Sequence[str]] = None,
    cutoff: bool = True,
    cutoff_Lee: float = 0.0,
    type: Literal["heatmap", "dot", "data"] = "heatmap",
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "Reds",
    reverse_colorscale: bool = False,
    n_colors: int = 6,
    dot_size: Tuple[float, float] = (1.0, 6.0),
    angle_x: float = 45.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title_name: Optional[str] = None,
    legend_title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 10,
    x_lab_rot: float = 45.0,
    row_show: Optional[Sequence[str]] = None,
    col_show: Optional[Sequence[str]] = None,
) -> Union[Figure, pd.DataFrame]:
    """Spatial co-occurrence (Lee statistic) plot.

    Direct port of R's :func:`plotStatistics_Lee` (``visualization.R:8321``).
    Computes the Lee L statistic between every column of ``x_labels`` and
    every column of ``y_labels`` using :class:`esda.lee.Spatial_Pearson`,
    then renders either a heatmap (default) or a dot plot. When
    ``type='data'`` the raw cooccurrence DataFrame is returned.

    Parameters
    ----------
    x_labels, y_labels : ndarray, pandas.Series, or scipy.sparse
        Either a categorical vector (one-hot encoded internally), a numeric
        vector (becomes a single column), or a (n × k) matrix of features.
    coords : ndarray, optional
        ``(n, 2)`` cell coordinates. Either ``coords`` or ``lw`` must be set.
    lw : libpysal.weights.W, optional
        Pre-computed spatial-weights object.
    x_levels, y_levels : sequence of str, optional
        Re-order columns / rows of the cooccurrence matrix.
    cutoff : bool, default ``True``
        If ``True``, set values ``<= cutoff_Lee`` to NaN.
    cutoff_Lee : float, default ``0.0``
    type : {"heatmap", "dot", "data"}, default ``"heatmap"``
    color_use : sequence of str, optional
        Per-column annotation strip colours.
    color_heatmap : str, default ``"Reds"``
    reverse_colorscale : bool
    n_colors : int, default ``6``
    dot_size : (min, max), default ``(1, 6)``
    angle_x : float
    xlabel, ylabel, title_name, legend_title : str, optional
    font_size, font_size_title : int
    x_lab_rot : float, default ``45``
    row_show, col_show : sequence of str, optional

    Returns
    -------
    matplotlib.figure.Figure or pandas.DataFrame
        Figure (heatmap / dot) or the raw DataFrame when ``type='data'``.
    """
    labels1_mat, x_cat, mode1 = _coerce_label_matrix(x_labels, xlabel or "variable")
    labels2_mat, y_cat, mode2 = _coerce_label_matrix(y_labels, ylabel or "variable")
    if mode1 == "dot" or mode2 == "dot":
        type = "dot"

    if coords is None and lw is None:
        raise ValueError("Please provide either coords or lw for Lee statistics.")
    if coords is not None and lw is not None:
        raise ValueError("Please don't provide both coords and lw at the same time.")

    if coords is not None:
        coords_arr = np.asarray(coords, dtype=float)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError("Please check your coordinates. It must be 2 columns.")
        # R uses ``mat2listw(diag(n))`` — i.e. a self-loop weight matrix.
        # Translate to an identity sparse connectivity.
        n = coords_arr.shape[0]
        W = sparse.eye(n, format="csr")
    else:
        # Convert pysal ``W`` (object with ``.sparse``) into a scipy sparse
        # matrix; some versions of pysal already return a sparse matrix.
        W = lw.sparse if hasattr(lw, "sparse") else lw

    from esda.lee import Spatial_Pearson

    # Compute Lee L for every (x col, y col). Spatial_Pearson.fit expects
    # 2-D ``(n, 1)`` columns even for univariate inputs.
    n_x = labels1_mat.shape[1]
    n_y = labels2_mat.shape[1]
    df_lee = np.zeros((n_y, n_x), dtype=float)
    for L1 in range(n_x):
        a = labels1_mat[:, L1].reshape(-1, 1)
        for L2 in range(n_y):
            b = labels2_mat[:, L2].reshape(-1, 1)
            est = Spatial_Pearson(connectivity=W, permutations=0).fit(a, b)
            df_lee[L2, L1] = float(est.association_[0, 1])

    # Convert to (rows = x cols, cols = y cols), like R's transpose.
    mat = df_lee.T  # shape (n_x, n_y)
    x_lvls = list(x_cat.categories.astype(str))
    y_lvls = list(y_cat.categories.astype(str))

    if cutoff:
        mat = np.where(mat <= cutoff_Lee, np.nan, mat)

    if x_levels is not None:
        keep = [lv for lv in x_levels if lv in y_lvls]
        keep_idx = [y_lvls.index(lv) for lv in keep]
        mat = mat[:, keep_idx]
        y_lvls = keep
    if y_levels is not None:
        keep = [lv for lv in y_levels if lv in x_lvls]
        keep_idx = [x_lvls.index(lv) for lv in keep]
        mat = mat[keep_idx, :]
        x_lvls = keep

    if color_use is None:
        color_use = list(scPalette(mat.shape[1]))

    if type == "data":
        return pd.DataFrame(mat.T, index=y_lvls, columns=x_lvls)

    if type == "dot":
        # Long-form ``(x, y, proportion)`` mirroring R; mat[nrow(mat):1,]
        df_long = pd.DataFrame(mat[::-1, :].T, index=y_lvls, columns=list(reversed(x_lvls)))
        df_long_resh = df_long.stack(future_stack=True).reset_index()
        df_long_resh.columns = ["x", "y", "proportion"]
        df_long_resh = df_long_resh.dropna(subset=["proportion"])

        anchors = _resolve_brewer(color_heatmap, n_colors)
        cm_list = _color_ramp_palette(anchors, 100)
        if reverse_colorscale:
            cm_list = list(reversed(cm_list))
        cmap = LinearSegmentedColormap.from_list("Lee", cm_list, N=100)

        fig, ax = plt.subplots()
        if df_long_resh.empty:
            ax.text(0.5, 0.5, "no above-cutoff cooccurrences",
                    ha="center", va="center", transform=ax.transAxes)
            return fig
        x_codes = pd.Categorical(df_long_resh["x"]).codes
        y_codes = pd.Categorical(df_long_resh["y"]).codes
        prop = df_long_resh["proportion"].to_numpy(dtype=float)
        p_min, p_max = float(prop.min()), float(prop.max())
        if p_max == p_min:
            sizes = np.full_like(prop, fill_value=(dot_size[0] + dot_size[1]) / 2.0 * 20.0)
        else:
            sizes = (
                dot_size[0] + (dot_size[1] - dot_size[0]) * (prop - p_min) / (p_max - p_min)
            ) * 20.0
        sc = ax.scatter(x_codes, y_codes, s=sizes, c=prop, cmap=cmap)
        ax.set_xticks(np.arange(len(pd.Categorical(df_long_resh["x"]).categories)))
        ax.set_xticklabels(list(pd.Categorical(df_long_resh["x"]).categories),
                           rotation=x_lab_rot, ha="right", rotation_mode="anchor",
                           fontsize=font_size - 2)
        ax.set_yticks(np.arange(len(pd.Categorical(df_long_resh["y"]).categories)))
        ax.set_yticklabels(list(pd.Categorical(df_long_resh["y"]).categories),
                           fontsize=font_size)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, color="grey", alpha=0.3, linewidth=0.1)
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="Cooccurrence")
        cb.outline.set_visible(False)
        fig.tight_layout()
        return fig

    # heatmap branch
    if row_show is not None:
        keep = [lv for lv in row_show if lv in x_lvls]
        keep_idx = [x_lvls.index(lv) for lv in keep]
        mat = mat[keep_idx, :]
        x_lvls = keep
    if col_show is not None:
        keep = [lv for lv in col_show if lv in y_lvls]
        keep_idx = [y_lvls.index(lv) for lv in keep]
        mat = mat[:, keep_idx]
        y_lvls = keep
        color_use = [color_use[i] for i in keep_idx]

    anchors = _resolve_brewer(color_heatmap, n_colors)
    cm_list = _color_ramp_palette(anchors, 100)
    if reverse_colorscale:
        cm_list = list(reversed(cm_list))
    cmap = LinearSegmentedColormap.from_list("Lee", cm_list, N=100)
    cmap.set_bad("white")

    fig, ax = plt.subplots()
    sns.heatmap(
        pd.DataFrame(mat, index=x_lvls, columns=y_lvls),
        cmap=cmap, ax=ax, cbar_kws={"label": "Cooccurrence"},
        annot=True, fmt=".2f", annot_kws={"fontsize": 7, "color": "black"},
        linewidths=0.0, linecolor="grey",
    )
    ax.set_xlabel("")
    ax.set_ylabel(xlabel if xlabel is not None else "")
    title = (title_name or "") + "Spatial co-occurrence (Lee statistic)"
    ax.set_title(title, fontsize=font_size_title)
    ax.tick_params(axis="x", labelrotation=x_lab_rot, labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# plotStatistics_Gi (R: visualization.R:8828)
# ---------------------------------------------------------------------------
def plotStatistics_Gi(
    df_score: pd.DataFrame,
    group_by: Optional[Sequence] = None,
    group_highlight: Optional[Sequence[bool]] = None,
    n: int = 12,
    method: Literal["normal", "scoring", "3d"] = "normal",
    gstar: bool = False,
    p_value: Optional[float] = None,
    plot_title: Optional[str] = None,
    point_size: float = 2.0,
    nohighlight_alpha: float = 0.65,
    show_colorscale: bool = True,
    legend_group: str = "Gi",
    legend_group_title: str = "Gi's value",
    scene_name: str = "scene1",
    colorbar_y: float = 0.5,
    three_dim_z: float = 0.0,
) -> Figure:
    """Plot the Getis-Ord ``Gi`` (or ``Gi*``) statistic over a spatial layout.

    Direct port of R's :func:`plotStatistics_Gi` (``visualization.R:8828``).
    Reads the score column from ``df_score`` (with x/y in the first two
    columns), builds a k-nearest-neighbours weights object, computes the
    local-G statistic via :class:`esda.getisord.G_Local`, and returns a
    matplotlib figure with a centered RedsBlues colour scale.

    Parameters
    ----------
    df_score : pandas.DataFrame
        Three-column data frame with columns ``(x, y, score)``.
    group_by : sequence, optional
        Cell-group labels (carried for parity with R; not drawn).
    group_highlight : sequence of bool, optional
        Per-cell mask to mark highlighted cells (drawn at full alpha).
    n : int, default ``12``
        Number of nearest neighbours.
    method : {"normal", "scoring", "3d"}, default ``"normal"``
    gstar : bool, default ``False``
    p_value : float, optional
        p-value cutoff. Cells failing the cutoff are shown as outlined
        (mirrors R's ``point_shape = if_else(... 19, 1)``).
    plot_title : str, optional
    point_size, nohighlight_alpha : float
    show_colorscale : bool
    legend_group, legend_group_title, scene_name, colorbar_y, three_dim_z :
        Carried for parity with the R signature.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df_score.copy()
    df.columns = ["x", "y", "score"]
    n_obs = df.shape[0]

    # Build the kNN weights using libpysal
    from libpysal.weights import KNN

    coords = df[["x", "y"]].to_numpy(dtype=float)
    W = KNN.from_array(coords, k=int(n))
    W.transform = "B"

    from esda.getisord import G_Local

    # G_Local interprets weights as binary; we use star=gstar (include self).
    g_local = G_Local(df["score"].to_numpy(dtype=float), W,
                      transform="B", star=bool(gstar),
                      permutations=0)
    df["statistic_G"] = np.asarray(g_local.Zs, dtype=float)
    # The "p-value" column R reads is ``Pr(z != E(Gi))``; G_Local stores
    # that as ``p_norm`` (normal-distribution p-value).
    df["p_norm"] = np.asarray(g_local.p_norm, dtype=float)

    if method == "scoring":
        # Spectral palette (reversed) — mirrors R's
        # ``generate_custom_colorscale("Spectral", reversescale=T)``.
        anchors = _resolve_brewer("Spectral", 8)
        cm_list = _color_ramp_palette(list(reversed(anchors)), 99)
        cmap = LinearSegmentedColormap.from_list("Gi-scoring", cm_list, N=99)
        fig, ax = plt.subplots()
        sc = ax.scatter(df["x"], df["y"], c=df["score"], cmap=cmap, s=point_size * 10)
        if show_colorscale:
            cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="score")
            cb.outline.set_visible(False)
        if plot_title is not None:
            ax.set_title(str(plot_title), fontsize=10, fontweight="bold")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        return fig

    # "normal" / "3d" — both render to matplotlib (Z is constant, so 3D is
    # visually identical to the normal panel here).
    cm_hex = generate_RedsBlues_colormap(df["statistic_G"].to_numpy(dtype=float))
    cm_list = [str(c)[:7] for c in cm_hex]
    cmap = LinearSegmentedColormap.from_list("Gi", cm_list, N=99)

    fig, ax = plt.subplots()
    if p_value is None:
        sc = ax.scatter(df["x"], df["y"], c=df["statistic_G"], cmap=cmap,
                        s=point_size * 10, marker="o")
    else:
        sig = df["p_norm"] < float(p_value)
        sc = ax.scatter(df["x"][sig], df["y"][sig],
                        c=df["statistic_G"][sig], cmap=cmap,
                        s=point_size * 10, marker="o")
        ax.scatter(df["x"][~sig], df["y"][~sig],
                   c="white", edgecolors="grey",
                   s=point_size * 10, marker="o")
    if show_colorscale:
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label="Gi/Gi*")
        cb.outline.set_visible(False)
    gstar_label = "Gi*" if gstar else "Gi"
    title = f"Spatial statistics with {gstar_label}"
    if p_value is not None:
        title += f" (Normal, p.value={p_value})"
    if plot_title is not None:
        title = str(plot_title)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# savePlotWithCow (R: utilities.R:10)
# ---------------------------------------------------------------------------
def savePlotWithCow(
    filename: str,
    plot: Optional[Figure] = None,
    ncol: int = 1,
    nrow: int = 1,
    base_height: float = 3.71,
    base_asp: float = 1.618,
    base_width: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Save a matplotlib figure to disk.

    Direct port of R's :func:`savePlotWithCow` (``utilities.R:10``).
    Mirrors :func:`cowplot::save_plot` — given a width-aspect-height
    triple, derives the missing dimension and forwards to
    :meth:`Figure.savefig`. When ``plot`` is ``None`` the current figure
    is used (mirrors R's ``grDevices::recordPlot()``).

    Parameters
    ----------
    filename : str
        Output path.
    plot : matplotlib.figure.Figure, optional
        Figure to save. If ``None``, uses :func:`plt.gcf`.
    ncol, nrow : int, default ``1``
    base_height : float, default ``3.71``
    base_asp : float, default ``1.618``
        Aspect ratio used when ``base_width`` is ``None``.
    base_width : float, optional
    **kwargs
        Forwarded to :meth:`Figure.savefig` (``width``, ``height``,
        ``base_aspect_ratio``, ``cols``, ``rows`` are R-only aliases and
        are stripped before forwarding).
    """
    if plot is None:
        plot = plt.gcf()
    # Aliases that R supports for backwards compat
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    base_aspect_ratio = kwargs.pop("base_aspect_ratio", None)
    kwargs.pop("cols", None)
    kwargs.pop("rows", None)
    if base_aspect_ratio is not None:
        base_asp = float(base_aspect_ratio)
    if width is None:
        if base_width is None:
            base_width = base_height * base_asp
        width = base_width * ncol
    if height is None:
        height = base_height * nrow
    plot.set_size_inches(float(width), float(height))
    plot.savefig(filename, **kwargs)


# ---------------------------------------------------------------------------
# communicationDistPlot* (R: spatial.R)
# ---------------------------------------------------------------------------
def _density_panel(
    ax: Axes,
    xs: np.ndarray,
    fill: str,
    title: str,
    interaction_range: float,
    density_alpha: float,
) -> None:
    """Internal: draw one density panel mirroring R's ggplot helper."""
    xs = xs[~np.isnan(xs)]
    if xs.size > 1:
        sns.kdeplot(x=xs, ax=ax, fill=True, color=fill, alpha=density_alpha)
    elif xs.size == 1:
        ax.axvline(float(xs[0]), color=fill, alpha=density_alpha)
    ax.set_xlim(0.0, float(interaction_range) + 10.0)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, interaction_range + 11.0, 20.0))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Distance between cell pairs (um)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def communicationDistPlot2(
    adata: ad.AnnData,
    signaling_type: Literal["All", "Secreted", "Contact"] = "All",
    enriched_only: bool = True,
    density_alpha: float = 0.5,
) -> Figure:
    """Cell-cell communication distance density (Secreted vs. Contact).

    Direct port of R's :func:`communicationDistPlot2` (``spatial.R:11``).

    Parameters
    ----------
    adata : anndata.AnnData
    signaling_type : {"All", "Secreted", "Contact"}, default ``"All"``
    enriched_only : bool, default ``True``
        If ``True``, restrict to interactions in
        ``net["prob.cell"]`` (significantly enriched only).
    density_alpha : float, default ``0.5``

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_uns(adata)
    params = cc.get("options", {}).get("parameter", {})
    interaction_range = float(params.get("interaction.range", 250.0))
    res = cc.get("images", {}).get("result.computeCellDistance", None)
    if res is None:
        raise ValueError("communicationDistPlot2 requires uns['cellchat']['images']"
                         "['result.computeCellDistance'] (set by computeCommunProb)")

    d_spatial = res["d.spatial"]  # sparse
    adj_contact_sparse = res["adj.contact"]  # sparse
    # adj.contact * d.spatial elementwise (R: ``d.spatial * res$adj.contact``)
    adj_contact = sparse.csr_matrix(d_spatial).multiply(sparse.csr_matrix(adj_contact_sparse))

    if enriched_only:
        net = cc.get("net", {})
        prob_cell = net.get("prob.cell", None)
        if prob_cell is None:
            raise ValueError("enriched_only=True requires net['prob.cell']")
        # iterate over (i, j, k) coords; collect d.spatial[i,j], adj_contact[i,j]
        if hasattr(prob_cell, "coords"):
            ii = prob_cell.coords[0]
            jj = prob_cell.coords[1]
        else:
            ii = np.asarray(prob_cell["i"])
            jj = np.asarray(prob_cell["j"])
        d_arr = np.asarray(d_spatial.tocsr()[ii, jj]).ravel().astype(float)
        c_arr = np.asarray(adj_contact[ii, jj]).ravel().astype(float)
        d_arr = np.where(d_arr == 0, np.nan, d_arr)
        c_arr = np.where(c_arr == 0, np.nan, c_arr)
        df_secreted = d_arr
        df_contact = c_arr
    else:
        df_secreted = d_spatial.tocsr().data.astype(float)
        df_contact = adj_contact.tocsr().data.astype(float)

    if signaling_type == "Secreted":
        fig, ax = plt.subplots()
        _density_panel(ax, df_secreted, "#69b3a2",
                       "Secreting-dependent signaling",
                       interaction_range, density_alpha)
        fig.tight_layout()
        return fig
    if signaling_type == "Contact":
        fig, ax = plt.subplots()
        _density_panel(ax, df_contact, "#404080",
                       "Contact-dependent signaling",
                       interaction_range, density_alpha)
        fig.tight_layout()
        return fig
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.0))
    _density_panel(axes[0], df_secreted, "#69b3a2",
                   "Secreting-dependent signaling",
                   interaction_range, density_alpha)
    _density_panel(axes[1], df_contact, "#404080",
                   "Contact-dependent signaling",
                   interaction_range, density_alpha)
    fig.tight_layout()
    return fig


def communicationDistPlot(
    adata: ad.AnnData,
    enriched_only: bool = True,
    density_alpha: float = 0.5,
) -> Figure:
    """Cell-cell communication distance density (Diffusible / Contact).

    Direct port of R's :func:`communicationDistPlot` (``spatial.R:95``).
    The two-panel layout depends on ``options['parameter']['all.diffusible']``
    / ``['all.contact.dependent']`` flags written by
    :func:`computeCommunProb`.

    Parameters
    ----------
    adata : anndata.AnnData
    enriched_only : bool, default ``True``
    density_alpha : float, default ``0.5``

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_uns(adata)
    params = cc.get("options", {}).get("parameter", {})
    interaction_range = float(params.get("interaction.range", 250.0))
    res = cc.get("images", {}).get("result.computeCellDistance", None)
    if res is None:
        raise ValueError("communicationDistPlot requires uns['cellchat']['images']"
                         "['result.computeCellDistance'] (set by computeCommunProb)")

    d_spatial = res["d.spatial"]
    adj_contact_sparse = res["adj.contact"]
    adj_contact = sparse.csr_matrix(d_spatial).multiply(sparse.csr_matrix(adj_contact_sparse))

    df_spatial: List[Optional[np.ndarray]] = [None, None]
    if enriched_only:
        net = cc.get("net", {})
        prob_cell = net.get("prob.cell", None)
        if prob_cell is None:
            raise ValueError("enriched_only=True requires net['prob.cell']")
        if hasattr(prob_cell, "coords"):
            ii = prob_cell.coords[0]
            jj = prob_cell.coords[1]
            kk = np.asarray(prob_cell.coords[2], dtype=int)
        else:
            ii = np.asarray(prob_cell["i"])
            jj = np.asarray(prob_cell["j"])
            kk = np.asarray(prob_cell["k"], dtype=int)
        # If kk happens to be R-stored 1-based, normalize at the top.
        if kk.size > 0 and int(kk.min()) >= 1:
            kk = kk - 1
        d_arr = np.asarray(d_spatial.tocsr()[ii, jj]).ravel().astype(float)
        d_arr = np.where(d_arr == 0, np.nan, d_arr)
        if bool(params.get("all.diffusible", False)):
            df_spatial[0] = d_arr
        elif bool(params.get("all.contact.dependent", False)):
            df_spatial[1] = d_arr
        else:
            cc_LR = cc.get("LR", {}).get("LRsig", None)
            if cc_LR is not None and "annotation" in cc_LR.columns:
                ann = cc_LR["annotation"].astype(str).values
                # R: nLR1 = max(which(annotation %in% diff_set)) (1-based).
                # Python equivalent for the "diffusible-rows-first" layout:
                # nLR1 = number of diffusible rows.  Split by 0-based k:
                # kk < nLR1 → diffusible, kk >= nLR1 → contact.
                diff_set = {"Secreted Signaling", "ECM-Receptor", "Non-protein Signaling"}
                diff_mask = np.isin(ann, list(diff_set))
                if not diff_mask.any():
                    raise ValueError(
                        "communicationDistPlot: no diffusible-signaling annotation "
                        "found in uns['cellchat']['LR']['LRsig']['annotation'] — "
                        "expected at least one of "
                        "{'Secreted Signaling','ECM-Receptor','Non-protein Signaling'}."
                    )
                nLR1 = int(diff_mask.sum())
            else:
                raise ValueError(
                    "communicationDistPlot: cannot split diffusible vs "
                    "contact-dependent rows because uns['cellchat']['LR']['LRsig']"
                    " or its 'annotation' column is missing."
                )
            mask1 = (kk < nLR1)
            mask2 = (kk >= nLR1)
            df_spatial[0] = d_arr[mask1]
            df_spatial[1] = d_arr[mask2]
    else:
        df_spatial[0] = d_spatial.tocsr().data.astype(float)
        df_spatial[1] = adj_contact.tocsr().data.astype(float)

    panels: List[Tuple[np.ndarray, str, str]] = []
    if df_spatial[0] is not None:
        panels.append((df_spatial[0], "#69b3a2", "Diffusible signaling"))
    if df_spatial[1] is not None:
        panels.append((df_spatial[1], "#404080", "Contact-dependent signaling"))
    if not panels:
        panels.append((np.array([]), "#69b3a2", "Diffusible signaling"))

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(5.0, 3.0 * len(panels)),
                             squeeze=False)
    for i, (xs, fill, title) in enumerate(panels):
        _density_panel(axes[i, 0], xs, fill, title, interaction_range, density_alpha)
    fig.tight_layout()
    return fig
