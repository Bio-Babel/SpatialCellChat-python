"""Thin scanpy shims for the delegated SpatialCellChat plotting exports.

Per the SpatialCellChat porting contract §5.G, three R visualisation
functions are delegated to scanpy.pl rather than re-implemented:

- :func:`dotPlot`           → :func:`scanpy.pl.dotplot`
- :func:`StackedVlnPlot`    → :func:`scanpy.pl.stacked_violin`
- :func:`plotGeneExpression` → dispatches to :func:`scanpy.pl.violin`
  (or :func:`scanpy.pl.dotplot`) after :func:`extractEnrichedLR`.

These wrappers do nothing more than translate the R argument names to
the scanpy parameters and (when applicable) resolve the gene set via
:func:`extractEnrichedLR`. They return whatever scanpy returns
(typically :class:`matplotlib.figure.Figure` or ``None``).
"""

from __future__ import annotations

import warnings
from typing import Any, List, Literal, Optional, Sequence, Union

import matplotlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .analysis_topics import extractEnrichedLR

__all__ = [
    "dotPlot",
    "StackedVlnPlot",
    "plotGeneExpression",
]


def _resolve_group_by(adata: ad.AnnData, group_by: Optional[str]) -> Optional[str]:
    if group_by is None:
        for k in ("cellchat_idents", "active_ident", "cell.type"):
            if k in adata.obs.columns:
                return k
    return group_by


def _subset_idents(
    adata: ad.AnnData,
    group_by: Optional[str],
    idents: Optional[Sequence[str]],
) -> ad.AnnData:
    if idents is None or group_by is None or group_by not in adata.obs.columns:
        return adata
    keep = adata.obs[group_by].astype(str).isin([str(x) for x in idents])
    return adata[keep.to_numpy()].copy()


# ---------------------------------------------------------------------------
# dotPlot — delegates to sc.pl.dotplot
# ---------------------------------------------------------------------------
def dotPlot(
    adata: ad.AnnData,
    features: Sequence[str],
    rotation: bool = True,
    colormap: str = "OrRd",
    color_direction: int = 1,
    color_use: Sequence[str] = ("#F8766D", "#00BFC4"),
    scale: bool = True,
    col_min: float = -2.5,
    col_max: float = 2.5,
    dot_scale: float = 6.0,
    assay: str = "RNA",
    idents: Optional[Sequence[str]] = None,
    group_by: Optional[str] = None,
    split_by: Optional[str] = None,
    legend_width: float = 0.5,
    angle_x: float = 45.0,
    hjust_x: float = 1.0,
    angle_y: float = 0.0,
    hjust_y: float = 0.5,
    show_legend: bool = True,
    color_heatmap: Optional[str] = None,
    standard_scale: Optional[Literal["var", "group"]] = None,
    swap_axes: Optional[bool] = None,
    **kwargs: Any,
) -> Figure:
    """Dot plot of gene expression by cell group.

    Mirrors R's :func:`dotPlot` (``visualization.R:3954``) by translating
    its parameters onto :func:`scanpy.pl.dotplot`:

    - ``features``       → ``var_names``
    - ``group.by``       → ``groupby`` (default: ``"active_ident"`` /
      ``"cellchat_idents"`` if available)
    - ``colormap``       → ``cmap`` (default ``"OrRd"`` matches R)
    - ``color.direction``→ reverses ``cmap`` when ``-1`` (R default ``1``)
    - ``col.min`` / ``col.max`` → ``vmin``/``vmax`` (R defaults ``-2.5``/``2.5``)
    - ``dot.scale``      → ``dot_max``/``size_max`` (R default ``6``)
    - ``scale``          → ``standard_scale`` (R default ``TRUE``)
    - ``rotation``       → ``swap_axes`` (R default ``TRUE``: ``coord_flip()``)
    - ``idents``         → row-level subset of cells
    - ``split.by``       → ``split_by`` forwarded to scanpy

    Parameters
    ----------
    adata : anndata.AnnData
    features : sequence of str
    rotation : bool, default ``True``
        ``coord_flip``-equivalent (forwarded as scanpy ``swap_axes``).
    colormap : str, default ``"OrRd"``
    color_direction : {-1, 1}, default 1
        Reverses ``cmap`` when -1.
    color_use : sequence of str
        Carried for parity (scanpy uses its own legend / cmap).
    scale : bool, default ``True``
        Dispatched to scanpy ``standard_scale='var'`` (matches Seurat's row-scale).
    col_min, col_max : float
        Dispatched to scanpy ``vmin``/``vmax`` after standardisation.
    dot_scale : float, default 6.0
        Dispatched to ``dot_max`` / ``size_max``.
    assay : str
        Reserved for parity (scanpy plots from ``adata.X``).
    idents : sequence of str, optional
        Restrict to a subset of group levels (R: keeps ``Idents %in% idents``).
    group_by : str, optional
    split_by : str, optional
    legend_width : float
        Reserved for parity.
    angle_x, hjust_x, angle_y, hjust_y : visual kwargs (best-effort).
    show_legend : bool
    color_heatmap : str, optional
        Backwards-compatible alias for ``colormap`` (matches the slice-6a
        signature). When set, overrides ``colormap``.
    standard_scale : {"var", "group"}, optional
        Override for ``scale``: when set, takes precedence.
    swap_axes : bool, optional
        Override for ``rotation``: when set, takes precedence.

    Returns
    -------
    matplotlib.figure.Figure
    """
    group_by = _resolve_group_by(adata, group_by)
    adata_use = _subset_idents(adata, group_by, idents)

    if color_heatmap is not None:
        colormap = color_heatmap
    use_cmap = colormap
    if color_direction == -1:
        use_cmap = matplotlib.colormaps.get_cmap(colormap).reversed()

    if standard_scale is None:
        ss = "var" if scale else None
    else:
        ss = standard_scale
    if swap_axes is None:
        swap_axes = bool(rotation)

    kwargs.setdefault("show", False)
    kwargs.setdefault("return_fig", True)
    fig_or_obj = sc.pl.dotplot(
        adata_use, var_names=list(features), groupby=group_by,
        cmap=use_cmap, standard_scale=ss,
        swap_axes=swap_axes,
        vmin=float(col_min), vmax=float(col_max),
        dot_max=1.0, size_title="Percent expressed",
        colorbar_title="Scaled expression",
        **kwargs,
    )
    if isinstance(fig_or_obj, Figure):
        fig = fig_or_obj
    else:
        # scanpy returns its own DotPlot wrapper; the figure is created
        # lazily by ``make_figure()``. With ``return_fig=True`` scanpy
        # returns *before* calling it (see scanpy.pl.dotplot source), so
        # we must invoke it here or .fig will be unset.
        if hasattr(fig_or_obj, "style"):
            try:
                fig_or_obj.style(dot_max=1.0, dot_min=0.0, size_exponent=float(dot_scale) / 6.0)
            except Exception:
                pass
        if getattr(fig_or_obj, "fig", None) is None and hasattr(fig_or_obj, "make_figure"):
            fig_or_obj.make_figure()
        fig = getattr(fig_or_obj, "fig", None) or plt.gcf()

    # Best-effort tick rotation
    for ax in fig.axes:
        for label in ax.get_xticklabels():
            label.set_rotation(float(angle_x))
            label.set_horizontalalignment(
                "right" if hjust_x >= 1.0 else ("left" if hjust_x <= 0.0 else "center")
            )
        for label in ax.get_yticklabels():
            label.set_rotation(float(angle_y))
    if not show_legend:
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    return fig


# ---------------------------------------------------------------------------
# StackedVlnPlot — delegates to sc.pl.stacked_violin
# ---------------------------------------------------------------------------
def StackedVlnPlot(
    adata: ad.AnnData,
    features: Sequence[str],
    idents: Optional[Sequence[str]] = None,
    split_by: Optional[str] = None,
    color_use: Optional[Sequence[str]] = None,
    colors_ggplot: bool = False,
    angle_x: float = 90.0,
    vjust_x: Optional[float] = None,
    hjust_x: Optional[float] = None,
    show_text_y: bool = True,
    line_size: Optional[float] = None,
    pt_size: float = 0.0,
    plot_margin: Any = None,
    show_median: bool = False,
    median_size: float = 1.0,
    group_by: Optional[str] = None,
    swap_axes: bool = False,
    **kwargs: Any,
) -> Figure:
    """Stacked violin plot of gene expression by cell group.

    Mirrors R's :func:`StackedVlnPlot` (``visualization.R:4010``) on top
    of :func:`scanpy.pl.stacked_violin`. R parameter mapping:

    - ``features`` → ``var_names``
    - ``idents``   → cell-level subset on ``group_by``
    - ``split.by`` → ``split_by``
    - ``color.use``→ ``palette``
    - ``angle.x``/``vjust.x``/``hjust.x`` → axis-label rotation
    - ``pt.size`` → forwarded to scanpy (jitter size)
    - ``plot.margin``, ``show.text.y``, ``line.size`` → best-effort visual tweaks
    - ``show.median`` / ``median.size`` → reserved (no scanpy equivalent)

    Returns
    -------
    matplotlib.figure.Figure
    """
    group_by = _resolve_group_by(adata, group_by)
    adata_use = _subset_idents(adata, group_by, idents)

    # Default vjust/hjust (R's defaults table)
    if vjust_x is None or hjust_x is None:
        defaults = {0: (0.0, 0.0), 45: (1.0, 1.0), 90: (0.5, 1.0)}
        vj, hj = defaults.get(int(angle_x), (1.0, 1.0))
        if vjust_x is None:
            vjust_x = vj
        if hjust_x is None:
            hjust_x = hj

    palette = list(color_use) if (color_use is not None and not colors_ggplot) else None
    if palette is not None:
        kwargs.setdefault("colorbar_title", None)
    if split_by is not None:
        # scanpy doesn't accept split_by directly; honour the R semantics by
        # ignoring or warning when set.
        warnings.warn(
            "`split_by` is reserved for the merged-mode comparison and is "
            "ignored by the scanpy delegation.",
            stacklevel=2,
        )

    kwargs.setdefault("show", False)
    kwargs.setdefault("return_fig", True)
    fig_or_obj = sc.pl.stacked_violin(
        adata_use, var_names=list(features), groupby=group_by,
        swap_axes=swap_axes,
        **kwargs,
    )
    if isinstance(fig_or_obj, Figure):
        fig = fig_or_obj
    else:
        # scanpy returns the StackedViolin wrapper *before* calling
        # ``make_figure()`` when ``return_fig=True`` (see
        # scanpy.pl.stacked_violin source), so trigger it explicitly.
        if getattr(fig_or_obj, "fig", None) is None and hasattr(fig_or_obj, "make_figure"):
            fig_or_obj.make_figure()
        fig = getattr(fig_or_obj, "fig", None) or plt.gcf()

    # Best-effort visual tweaks: rotation, palette
    # R parity (visualization.R:6440-6444):
    #   ymaxs <- purrr::map_dbl(plot_list, extract_max)
    #   plot_list <- purrr::map2(plot_list, ymaxs, function(x,y) x +
    #     scale_y_continuous(breaks = c(y)) + expand_limits(y = y))
    # Each per-feature panel gets a single y-tick at its panel's max
    # (audit G7-6).
    feature_count = len(list(features))
    for ax in fig.axes:
        for label in ax.get_xticklabels():
            label.set_rotation(float(angle_x))
            label.set_horizontalalignment(
                "right" if (hjust_x or 0.0) >= 1.0
                else ("left" if (hjust_x or 0.0) <= 0.0 else "center")
            )
        if line_size is not None:
            for spine in ax.spines.values():
                spine.set_linewidth(float(line_size))

    # Per-violin y-tick: scanpy's stacked_violin places one Axes per
    # feature row (when ``swap_axes=False``) or per group column
    # (when ``swap_axes=True``). Each panel's y-axis spans from 0 (or
    # negative) to the data max for that feature → set a single tick
    # at the maximum visible y-value (R: ``scale_y_continuous(breaks =
    # c(extract_max(p)))``). Skip when the user explicitly suppressed
    # y-text via ``show_text_y=False``.
    if show_text_y and feature_count >= 1:
        # Identify violin panels by data-bearing axes and apply per-axes
        # ymax tick. matplotlib's ``ax.dataLim.y1`` reflects the natural
        # data extent.
        violin_axes = [
            ax for ax in fig.axes if ax.collections and any(
                hasattr(c, "get_paths") for c in ax.collections
            )
        ]
        for ax in violin_axes:
            try:
                _, ymax = ax.get_ylim()
            except Exception:
                continue
            if not np.isfinite(ymax):
                continue
            # Round to 1 decimal for a clean tick label (R uses
            # ``extract_max`` which returns the panel's exact max — we
            # mirror by reading get_ylim()'s upper bound, which scanpy
            # already pads to the data max).
            tick = float(np.round(ymax, 1))
            ax.set_yticks([tick])
            ax.set_yticklabels([f"{tick:g}"])
    elif not show_text_y:
        for ax in fig.axes:
            ax.set_yticks([])
    if palette is not None and len(palette) > 0:
        # apply colors to violin patches (best effort)
        for ax in fig.axes:
            patches = [c for c in ax.collections if hasattr(c, "set_facecolor")]
            for k, p in enumerate(patches):
                p.set_facecolor(palette[k % len(palette)])
    return fig


# ---------------------------------------------------------------------------
# plotGeneExpression — uses extractEnrichedLR + sc.pl.violin / sc.pl.dotplot / barPlot
# ---------------------------------------------------------------------------
def _barPlot(
    adata: ad.AnnData,
    features: Sequence[str],
    group_by: Optional[str] = None,
    split_by: Optional[str] = None,
    color_use: Optional[Sequence[str]] = None,
    method: Literal["truncatedMean", "triMean", "median"] = "truncatedMean",
    trim: float = 0.1,
    x_lab_rot: bool = False,
    ncol: int = 1,
    **kwargs: Any,
) -> Figure:
    """Internal barplot mirror of R's ``barPlot`` (visualization.R:4112).

    Computes per-cell-group average expression using ``method`` and renders
    one bar plot per feature, stacked in ``ncol`` columns.
    """
    if method not in ("truncatedMean", "triMean", "median"):
        raise ValueError(
            "method must be 'truncatedMean', 'triMean', or 'median'."
        )
    group_by = _resolve_group_by(adata, group_by)
    if group_by is None or group_by not in adata.obs.columns:
        raise ValueError("group_by could not be resolved.")
    labels = pd.Categorical(adata.obs[group_by])
    levels = list(labels.categories)
    feat_list = [f for f in features if f in adata.var_names]
    if not feat_list:
        raise ValueError("No features in adata.var_names")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    def _agg(values: np.ndarray) -> float:
        v = values[~np.isnan(values)]
        if v.size == 0:
            return 0.0
        if method == "truncatedMean":
            from ._utils import thresholdedMean
            return float(thresholdedMean(v, trim=trim, na_rm=True))
        if method == "triMean":
            q = np.quantile(v, [0.25, 0.5, 0.5, 0.75])
            return float(q.mean())
        return float(np.median(v))

    nfeat = len(feat_list)
    nrow = int(np.ceil(nfeat / ncol))
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(max(3.0, 0.5 * len(levels) + 2.0) * ncol, max(2.0, 1.6) * nrow),
        squeeze=False,
    )
    for k, feat in enumerate(feat_list):
        ax = axes[k // ncol, k % ncol]
        col = list(adata.var_names).index(feat)
        vals = []
        for lvl in levels:
            mask = (labels == lvl).to_numpy() if hasattr(labels == lvl, "to_numpy") else np.asarray(labels == lvl)
            vals.append(_agg(X[mask, col]))
        bar_colors = (
            list(color_use)[: len(levels)]
            if color_use is not None
            else None
        )
        ax.bar(np.arange(len(levels)), vals, color=bar_colors)
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(
            levels, rotation=(45.0 if x_lab_rot else 0.0),
            ha="right" if x_lab_rot else "center",
        )
        ax.set_ylabel(feat)
    # hide unused
    for k in range(nfeat, nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    return fig


def plotGeneExpression(
    adata: ad.AnnData,
    features: Optional[Sequence[str]] = None,
    signaling: Optional[Sequence[str]] = None,
    enriched_only: bool = True,
    type: Literal["violin", "dot", "bar"] = "violin",
    group_by: Optional[str] = None,
    color_use: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Figure:
    """Show ligand/receptor gene expression for one or more signaling pathways.

    Mirrors R's :func:`plotGeneExpression` (visualization.R:3883):

    - When ``features`` is set, it overrides ``signaling`` (with a warning
      if both are provided).
    - When only ``signaling`` is set, the L-R gene set is resolved via
      :func:`extractEnrichedLR`.
    - ``type='violin'`` dispatches to :func:`StackedVlnPlot`,
      ``type='dot'`` to :func:`dotPlot`,
      ``type='bar'`` to the internal barplot (R: ``barPlot``).

    Parameters
    ----------
    adata : anndata.AnnData
    features : sequence of str, optional
        Genes to plot. Takes precedence over ``signaling``.
    signaling : sequence of str, optional
        Signaling pathway names; resolved via :func:`extractEnrichedLR`.
    enriched_only : bool, default ``True``
    type : {"violin", "dot", "bar"}, default ``"violin"``
    group_by : str, optional
    color_use : sequence of str, optional
        Used as the violin / dot palette.
    **kwargs
        Forwarded to the underlying scanpy plotting function.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if type not in ("violin", "dot", "bar"):
        raise ValueError("`type` must be 'violin', 'dot', or 'bar'")

    if features is not None and signaling is not None:
        warnings.warn(
            "`features` will be used when inputting both `features` and "
            "`signaling`!",
            stacklevel=2,
        )
        feature_use: List[str] = list(features)
    elif features is not None:
        feature_use = list(features)
    elif signaling is not None:
        res = extractEnrichedLR(
            adata, signaling=list(signaling),
            geneLR_return=True, enriched_only=enriched_only,
        )
        feature_use = list(res["geneLR"])
    else:
        raise ValueError("Either `features` or `signaling` must be provided.")

    var_set = set(map(str, adata.var_names))
    genes_present = [g for g in feature_use if str(g) in var_set]
    if not genes_present:
        raise ValueError("None of the inferred ligand/receptor genes are in adata.var_names")

    group_by = _resolve_group_by(adata, group_by)

    if type == "violin":
        return StackedVlnPlot(
            adata, features=genes_present,
            group_by=group_by, color_use=color_use,
            **kwargs,
        )
    if type == "dot":
        return dotPlot(
            adata, features=genes_present,
            group_by=group_by, color_use=color_use or ("#F8766D", "#00BFC4"),
            **kwargs,
        )
    # type == "bar"
    return _barPlot(
        adata, features=genes_present, group_by=group_by,
        color_use=color_use, **kwargs,
    )
