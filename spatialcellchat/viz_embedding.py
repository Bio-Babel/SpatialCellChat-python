"""Embedding, signaling-role, contribution and river plots (slice 6b).

Direct ports of the manifold / signaling-role / river exports in
``visualization.R`` and ``analysis.R``:

- :func:`netVisual_embedding`                  (R: ``visualization.R:4152``)
- :func:`netVisual_embeddingZoomIn`            (R: ``visualization.R:4246``)
- :func:`netVisual_embeddingPairwise`          (R: ``visualization.R:4333``)
- :func:`netVisual_embeddingPairwiseZoomIn`    (R: ``visualization.R:4448``)
- :func:`netAnalysis_signalingRole_scatter`    (R: ``analysis.R:2642``)
- :func:`netAnalysis_diff_signalingRole_scatter`
                                                (R: ``analysis.R:2773``)
- :func:`netAnalysis_signalingChanges_scatter` (R: ``analysis.R:2937``)
- :func:`netAnalysis_signalingRole_network`    (R: ``analysis.R:2559``)
- :func:`netAnalysis_contribution`             (R: ``analysis.R:229``)
- :func:`netAnalysis_river`                    (R: ``visualization.R:3793``)

All public functions accept :class:`anndata.AnnData` first and return a
:class:`matplotlib.figure.Figure`. :func:`spatialcellchat._text_repel.repel_text`
is used to space labels.
"""

from __future__ import annotations

import matplotlib

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from natsort import natsorted

from ._anndata import (
    get_idents_levels as _get_idents_levels,
    joint_idents_levels,
    resolve_dataset_names,
)
from ._palette import scPalette, ggPalette
from ._text_repel import repel_text
from ._themes import _BREWER_PALETTES, _BREWER_SUBSET, _brewer_pal


_R_GREY_MAP = {
    "grey0": "0.0", "grey10": "0.1", "grey20": "0.2", "grey30": "0.3",
    "grey40": "0.4", "grey50": "0.5", "grey60": "0.6", "grey70": "0.7",
    "grey80": "0.8", "grey90": "0.9", "grey100": "1.0",
    "gray0": "0.0", "gray10": "0.1", "gray20": "0.2", "gray30": "0.3",
    "gray40": "0.4", "gray50": "0.5", "gray60": "0.6", "gray70": "0.7",
    "gray80": "0.8", "gray90": "0.9", "gray100": "1.0",
}


def _safe_color(c: str) -> str:
    """Translate R-style ``"grey90"`` colour names to matplotlib equivalents."""
    if isinstance(c, str) and c.lower() in _R_GREY_MAP:
        return _R_GREY_MAP[c.lower()]
    return c


def _safe_brewer(name: str, n: int) -> List[str]:
    """``_brewer_pal`` with a matplotlib fallback for unsupported palettes."""
    if name in _BREWER_SUBSET:
        return list(_brewer_pal(name, n))
    if name in _BREWER_PALETTES:
        full = list(_BREWER_PALETTES[name])
        if n <= len(full):
            idxs = np.linspace(0, len(full) - 1, n).round().astype(int)
            return [full[i] for i in idxs]
    cmap = matplotlib.colormaps.get_cmap(name)
    samples = np.linspace(0.0, 1.0, n)
    return [matplotlib.colors.to_hex(cmap(s)) for s in samples]

__all__ = [
    "netVisual_embedding",
    "netVisual_embeddingZoomIn",
    "netVisual_embeddingPairwise",
    "netVisual_embeddingPairwiseZoomIn",
    "netAnalysis_signalingRole_scatter",
    "netAnalysis_diff_signalingRole_scatter",
    "netAnalysis_signalingChanges_scatter",
    "netAnalysis_signalingRole_network",
    "netAnalysis_contribution",
    "netAnalysis_river",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_cellchat(adata: ad.AnnData) -> Dict[str, Any]:
    """Read-only access to ``adata.uns['cellchat']``; raise if missing.

    Local reader helper — distinct from :func:`spatialcellchat._anndata.cellchat_dict`
    (writer / get-or-create) because viz functions should error out clearly
    when the upstream ``computeCommunProb`` etc. has not been run.
    """
    if "cellchat" not in adata.uns:
        raise ValueError("adata.uns['cellchat'] is missing.")
    return adata.uns["cellchat"]


# ---------------------------------------------------------------------------
# Common embedding plot helper
# ---------------------------------------------------------------------------
def _plot_embedding(
    ax,
    df: pd.DataFrame,
    color_use: Sequence[str],
    dot_size: Tuple[float, float],
    dot_alpha: float,
    label_size: float,
    do_label: bool,
    label_subset: Optional[pd.DataFrame] = None,
) -> None:
    """Scatter for embeddings (used by all four netVisual_embedding* funcs)."""
    grp_col = "Groups" if "Groups" in df.columns else "clusters"
    groups = df[grp_col]
    if isinstance(groups.dtype, pd.CategoricalDtype):
        levels = list(groups.cat.categories)
    else:
        levels = sorted(set(groups.astype(str).tolist()))
    color_use = list(color_use)
    if len(color_use) < len(levels):
        # cycle the palette to cover all levels
        color_use = (color_use * ((len(levels) // len(color_use)) + 1))[: len(levels)]

    size_min, size_max = dot_size
    sizes = (
        np.clip(df["Commun.Prob."].to_numpy(dtype=float), 0.0, 1.0)
        * (size_max - size_min) + size_min
    )

    for k, lvl in enumerate(levels):
        sel = df[grp_col].astype(str) == str(lvl)
        ax.scatter(
            df.loc[sel, "x"], df.loc[sel, "y"],
            s=(sizes[sel] ** 2) * 8.0,
            facecolors=matplotlib.colors.to_rgba(color_use[k], alpha=dot_alpha),
            edgecolors=color_use[k],
            linewidth=0.5,
            label=str(lvl),
        )
    if do_label and label_subset is not None and len(label_subset) > 0:
        clrs: List[str] = []
        for _, row in label_subset.iterrows():
            lvl_str = str(row[grp_col])
            # ``label_subset`` is a row-subset of ``df`` whose ``grp_col``
            # values are by construction in ``levels``; the only way an
            # entry can be absent is a NaN group label, in which case fall
            # back to the first colour (R's ggplot uses grey, but absent a
            # neutral colour parameter we keep the prior behaviour).
            idx_lvl = levels.index(lvl_str) if lvl_str in levels else 0
            clrs.append(color_use[idx_lvl])
        repel_text(
            ax,
            label_subset["x"].to_numpy(dtype=float),
            label_subset["y"].to_numpy(dtype=float),
            label_subset["labels"].astype(str).tolist(),
            fontsize=label_size * 2.4,
            color=clrs,
        )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _build_embedding_df(
    adata: ad.AnnData, slot_name: str, type_: str, comparison_name: str = "single",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Reconstruct the (x, y, Commun.Prob., labels, Groups) frame.

    Returns ``(df, prob, pathway_remove)``.
    """
    cc = _ensure_cellchat(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] missing.")
    slot = cc[slot_name]
    sim = slot.get("similarity", {})
    typ = sim.get(type_)
    if typ is None:
        raise ValueError(f"similarity[{type_!r}] missing; run computeNetSimilarity.")
    dr = typ.get("dr", {}).get(comparison_name)
    if dr is None:
        raise ValueError(
            f"similarity[{type_!r}]['dr'][{comparison_name!r}] missing; run netEmbedding."
        )
    grp = typ.get("group", {}).get(comparison_name)
    if grp is None:
        raise ValueError(
            f"similarity[{type_!r}]['group'][{comparison_name!r}] missing; run netClustering."
        )
    if isinstance(dr, pd.DataFrame):
        Y = dr.values
        names = list(dr.index)
    else:
        Y = np.asarray(dr, dtype=float)
        names = [str(i) for i in range(Y.shape[0])]
    if isinstance(grp, pd.Series):
        groups = grp.values
    else:
        groups = np.asarray(grp)

    sim_mat = typ.get("matrix", {}).get(comparison_name)
    pathway_remove: List[str] = []
    if sim_mat is not None:
        if isinstance(sim_mat, pd.DataFrame):
            csum = sim_mat.values.sum(axis=0)
            row_names = list(sim_mat.index)
        else:
            csum = np.asarray(sim_mat, dtype=float).sum(axis=0)
            row_names = [str(i) for i in range(sim_mat.shape[0])]
        pathway_remove = [row_names[i] for i in np.flatnonzero(csum == 1.0)]

    prob = np.asarray(slot.get("prob"), dtype=float)
    pathway_names = list(slot.get("pathways", []))
    if len(pathway_names) == 0:
        pathway_names = [str(i) for i in range(prob.shape[2])]
    if pathway_remove:
        keep_idx = [i for i, p in enumerate(pathway_names) if p not in pathway_remove]
        prob_kept = prob[:, :, keep_idx]
        names_kept = [pathway_names[i] for i in keep_idx]
    else:
        prob_kept = prob
        names_kept = list(pathway_names)

    prob_sum = prob_kept.sum(axis=(0, 1))
    if prob_sum.max() == 0:
        commun = np.zeros_like(prob_sum)
    else:
        commun = prob_sum / prob_sum.max()

    # Y rows align with names_kept; if not, reindex
    n_use = min(len(names_kept), Y.shape[0], len(commun))
    df = pd.DataFrame({
        "x": Y[:n_use, 0],
        "y": Y[:n_use, 1],
        "Commun.Prob.": commun[:n_use],
        "labels": names_kept[:n_use],
        "Groups": pd.Categorical([str(g) for g in groups[:n_use]]),
    })
    return df, prob, pathway_remove


# ---------------------------------------------------------------------------
# netVisual_embedding (R: visualization.R:4152)
# ---------------------------------------------------------------------------
def netVisual_embedding(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    color_use: Optional[Sequence[str]] = None,
    pathway_labeled: Optional[Sequence[str]] = None,
    top_label: float = 1.0,
    pathway_remove: Optional[Sequence[str]] = None,
    pathway_remove_show: bool = True,
    dot_size: Tuple[float, float] = (2.0, 6.0),
    label_size: float = 2.0,
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    font_size: float = 10.0,
    font_size_title: float = 12.0,
    do_label: bool = True,
    show_legend: bool = True,
    show_axes: bool = True,
) -> Figure:
    """2-D manifold visualisation of signaling networks.

    Mirrors :func:`netVisual_embedding` (R: ``visualization.R:4152``).

    Parameters
    ----------
    adata : anndata.AnnData
    slot_name, type_ : as in R.
    color_use, pathway_labeled, pathway_remove, pathway_remove_show : as in R.
    top_label : float (0, 1]
        Reserved (used only for merged objects in R).
    dot_size : (min, max) tuple.
    label_size, dot_alpha : visual kwargs.
    xlabel, ylabel, title : axis labels and main title.
    font_size, font_size_title : visual kwargs.
    do_label, show_legend, show_axes : visual flags.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if type_ not in ("functional", "structural"):
        raise ValueError("type_ must be 'functional' or 'structural'.")
    df, _, auto_remove = _build_embedding_df(adata, slot_name, type_, "single")
    if pathway_remove is None:
        pathway_remove = auto_remove
    pathway_remove = list(pathway_remove)

    if color_use is None:
        n_groups = len(set(df["Groups"].astype(str)))
        color_use = list(ggPalette(n_groups))

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    if pathway_labeled is None:
        label_subset = df.copy() if do_label else None
    else:
        label_subset = df[df["labels"].isin(set(pathway_labeled))].copy()

    _plot_embedding(
        ax, df, color_use, dot_size, dot_alpha, label_size,
        do_label=do_label, label_subset=label_subset,
    )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size_title)
    if pathway_remove and pathway_remove_show:
        ax.text(
            0.01, 0.99,
            f"Isolate pathways: {', '.join(pathway_remove)}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=label_size * 2.4, fontstyle="italic",
        )
    if show_legend:
        ax.legend(loc="best", fontsize=font_size - 2)
    if not show_axes:
        ax.set_axis_off()
    return fig


# ---------------------------------------------------------------------------
# netVisual_embeddingZoomIn (R: visualization.R:4246)
# ---------------------------------------------------------------------------
def netVisual_embeddingZoomIn(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    color_use: Optional[Sequence[str]] = None,
    pathway_remove: Optional[Sequence[str]] = None,
    nCol: int = 1,
    dot_size: Tuple[float, float] = (2.0, 6.0),
    label_size: float = 2.8,
    dot_alpha: float = 0.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    do_label: bool = True,
    show_legend: bool = False,
    show_axes: bool = True,
) -> Figure:
    """Per-cluster zoom-in of the signaling-network manifold.

    Mirrors :func:`netVisual_embeddingZoomIn` (R: ``visualization.R:4246``).
    Splits the manifold into one subplot per cluster.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df, _, auto_remove = _build_embedding_df(adata, slot_name, type_, "single")
    if pathway_remove is None:
        pathway_remove = auto_remove
    df = df.copy()
    df["clusters"] = df["Groups"].astype(str)

    # R parity: ``unique(clusters)`` returns first-appearance order, but the
    # plotting loop is ``for i in 1:length(unique(clusters))`` which assumes
    # cluster IDs are integers 1..N and renders them in numeric order. For
    # ≥10 clusters lexicographic ``sorted()`` would put ``"10"`` between
    # ``"1"`` and ``"2"``; ``natsorted`` mirrors R's numeric-aware ordering
    # for the common integer-coded case while gracefully handling
    # string-labelled clusters too.
    cluster_ids = list(natsorted(pd.unique(df["clusters"])))
    if color_use is None:
        color_use = list(ggPalette(len(cluster_ids)))
    color_use = list(color_use)

    nrow = int(np.ceil(len(cluster_ids) / nCol))
    fig, axes = plt.subplots(
        nrows=nrow, ncols=nCol,
        figsize=(4.0 * nCol, 3.5 * nrow), squeeze=False,
    )
    for k, cl in enumerate(cluster_ids):
        ax = axes[k // nCol, k % nCol]
        sub = df[df["clusters"] == cl]
        sizes = np.clip(sub["Commun.Prob."].to_numpy(dtype=float), 0.0, 1.0)
        size_min, size_max = dot_size
        sizes_pt = (sizes * (size_max - size_min) + size_min) ** 2 * 8.0
        ax.scatter(
            sub["x"], sub["y"], s=sizes_pt,
            facecolors=matplotlib.colors.to_rgba(color_use[k], alpha=dot_alpha),
            edgecolors=color_use[k], linewidth=0.5,
        )
        if do_label and len(sub) > 0:
            repel_text(
                ax,
                sub["x"].to_numpy(dtype=float),
                sub["y"].to_numpy(dtype=float),
                sub["labels"].astype(str).tolist(),
                fontsize=label_size * 2.6,
                color=[color_use[k]] * len(sub),
            )
        ax.set_title(f"Group {k + 1}", fontsize=12)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if not show_axes:
            ax.set_axis_off()
    # Hide empty subplots
    for k in range(len(cluster_ids), nrow * nCol):
        axes[k // nCol, k % nCol].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netVisual_embeddingPairwise (R: visualization.R:4333)
# ---------------------------------------------------------------------------
def netVisual_embeddingPairwise(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    comparison: Optional[Sequence[int]] = None,
    color_use: Optional[Sequence[str]] = None,
    point_shape: Optional[Sequence[str]] = None,
    pathway_labeled: Optional[Sequence[str]] = None,
    top_label: float = 1.0,
    pathway_remove: Optional[Sequence[str]] = None,
    pathway_remove_show: bool = True,
    dot_size: Tuple[float, float] = (2.0, 6.0),
    label_size: float = 2.5,
    dot_alpha: float = 0.5,
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    title: Optional[str] = None,
    do_label: bool = True,
    show_legend: bool = True,
    show_axes: bool = True,
) -> Figure:
    """Joint manifold visualisation across two datasets.

    Mirrors :func:`netVisual_embeddingPairwise` (R: ``visualization.R:4333``).
    For a single-dataset object the dataset list collapses to one
    element so this still produces a sensible plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if comparison is None:
        comparison = (1,)
    comparison = list(comparison)
    comparison_name = "-".join(str(c) for c in comparison)
    df, _, auto_remove = _build_embedding_df(adata, slot_name, type_, comparison_name)
    if pathway_remove is None:
        pathway_remove = auto_remove
    pathway_remove = list(pathway_remove)

    df = df.copy()
    df["clusters"] = df["Groups"].astype(str)
    if "group" not in df.columns:
        df["group"] = "1"

    # R parity: ``unique(clusters)`` returns first-appearance order, but the
    # plotting loop is ``for i in 1:length(unique(clusters))`` which assumes
    # cluster IDs are integers 1..N and renders them in numeric order. For
    # ≥10 clusters lexicographic ``sorted()`` would put ``"10"`` between
    # ``"1"`` and ``"2"``; ``natsorted`` mirrors R's numeric-aware ordering
    # for the common integer-coded case while gracefully handling
    # string-labelled clusters too.
    cluster_ids = list(natsorted(pd.unique(df["clusters"])))
    if color_use is None:
        color_use = list(ggPalette(len(cluster_ids)))
    color_use = list(color_use)

    # R parity (visualization.R:4392): ``geom_point(aes(fill = clusters,
    # colour = clusters, shape = group))``. Color encodes CLUSTER, shape
    # encodes GROUP (which dataset the pathway came from). The previous
    # implementation varied shape by cluster — that's the bug M-VIZ-5.
    point_shape = list(point_shape or ["o", "s", "^", "D", "v", "P", "X"])
    groups_unique = list(pd.unique(df["group"].astype(str)))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    size_min, size_max = dot_size
    sizes_pt = (
        np.clip(df["Commun.Prob."].to_numpy(dtype=float), 0.0, 1.0)
        * (size_max - size_min) + size_min
    ) ** 2 * 8.0
    for gi, gp in enumerate(groups_unique):
        marker = point_shape[gi % len(point_shape)]
        for k, cl in enumerate(cluster_ids):
            sel = (df["clusters"] == cl) & (df["group"].astype(str) == gp)
            if not sel.any():
                continue
            label = cl if len(groups_unique) == 1 else f"{cl} | {gp}"
            ax.scatter(
                df.loc[sel, "x"], df.loc[sel, "y"],
                s=sizes_pt[sel.to_numpy()],
                facecolors=matplotlib.colors.to_rgba(color_use[k], alpha=dot_alpha),
                edgecolors=color_use[k], linewidth=0.5,
                marker=marker,
                label=label,
            )
    if do_label:
        sub = df if pathway_labeled is None else df[df["labels"].isin(set(pathway_labeled))]
        if len(sub) > 0:
            clrs = [
                color_use[cluster_ids.index(c)] for c in sub["clusters"].astype(str)
            ]
            repel_text(
                ax,
                sub["x"].to_numpy(dtype=float),
                sub["y"].to_numpy(dtype=float),
                sub["labels"].astype(str).tolist(),
                fontsize=label_size * 2.4,
                color=clrs,
            )
    if pathway_remove and pathway_remove_show:
        ax.text(
            0.01, 0.99,
            f"Isolate pathways: {', '.join(pathway_remove)}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=label_size * 2.4, fontstyle="italic",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    if not show_axes:
        ax.set_axis_off()
    return fig


# ---------------------------------------------------------------------------
# netVisual_embeddingPairwiseZoomIn (R: visualization.R:4448)
# ---------------------------------------------------------------------------
def netVisual_embeddingPairwiseZoomIn(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    comparison: Optional[Sequence[int]] = None,
    color_use: Optional[Sequence[str]] = None,
    nCol: int = 1,
    point_shape: Optional[Sequence[str]] = None,
    pathway_remove: Optional[Sequence[str]] = None,
    dot_size: Tuple[float, float] = (2.0, 6.0),
    label_size: float = 2.8,
    dot_alpha: float = 0.5,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    do_label: bool = True,
    show_legend: bool = False,
    show_axes: bool = True,
) -> Figure:
    """Per-cluster zoom-in of the joint manifold across two datasets.

    Mirrors :func:`netVisual_embeddingPairwiseZoomIn`
    (R: ``visualization.R:4448``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if comparison is None:
        comparison = (1,)
    comparison = list(comparison)
    comparison_name = "-".join(str(c) for c in comparison)
    df, _, auto_remove = _build_embedding_df(adata, slot_name, type_, comparison_name)
    if pathway_remove is None:
        pathway_remove = auto_remove
    df = df.copy()
    df["clusters"] = df["Groups"].astype(str)
    if "group" not in df.columns:
        df["group"] = "1"

    # R parity: ``unique(clusters)`` returns first-appearance order, but the
    # plotting loop is ``for i in 1:length(unique(clusters))`` which assumes
    # cluster IDs are integers 1..N and renders them in numeric order. For
    # ≥10 clusters lexicographic ``sorted()`` would put ``"10"`` between
    # ``"1"`` and ``"2"``; ``natsorted`` mirrors R's numeric-aware ordering
    # for the common integer-coded case while gracefully handling
    # string-labelled clusters too.
    cluster_ids = list(natsorted(pd.unique(df["clusters"])))
    if color_use is None:
        color_use = list(ggPalette(len(cluster_ids)))
    color_use = list(color_use)
    point_shape = list(point_shape or ["o", "s", "^", "D", "v", "P", "X"])

    nrow = int(np.ceil(len(cluster_ids) / nCol))
    fig, axes = plt.subplots(
        nrows=nrow, ncols=nCol,
        figsize=(4.0 * nCol, 3.5 * nrow), squeeze=False,
    )
    size_min, size_max = dot_size
    for k, cl in enumerate(cluster_ids):
        ax = axes[k // nCol, k % nCol]
        sub = df[df["clusters"] == cl]
        sizes_pt = (
            np.clip(sub["Commun.Prob."].to_numpy(dtype=float), 0.0, 1.0)
            * (size_max - size_min) + size_min
        ) ** 2 * 8.0
        for j, grp in enumerate(sub["group"].astype(str).unique()):
            sel = sub["group"].astype(str) == grp
            ax.scatter(
                sub.loc[sel, "x"], sub.loc[sel, "y"],
                s=sizes_pt[sel.to_numpy()],
                facecolors=matplotlib.colors.to_rgba(color_use[k], alpha=dot_alpha),
                edgecolors=color_use[k], linewidth=0.5,
                marker=point_shape[j % len(point_shape)],
            )
        if do_label and len(sub) > 0:
            repel_text(
                ax,
                sub["x"].to_numpy(dtype=float),
                sub["y"].to_numpy(dtype=float),
                sub["labels"].astype(str).tolist(),
                fontsize=label_size * 2.6,
                color=[color_use[k]] * len(sub),
            )
        ax.set_title(f"Cluster {k + 1}", fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if not show_axes:
            ax.set_axis_off()
    for k in range(len(cluster_ids), nrow * nCol):
        axes[k // nCol, k % nCol].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_scatter (R: analysis.R:2642)
# ---------------------------------------------------------------------------
def netAnalysis_signalingRole_scatter(
    adata: ad.AnnData,
    signaling: Optional[Sequence[str]] = None,
    color_use: Optional[Sequence[str]] = None,
    slot_name: str = "netP",
    group: Optional[Sequence[str]] = None,
    weight_MinMax: Optional[Tuple[float, float]] = None,
    dot_size: Tuple[float, float] = (2.0, 6.0),
    point_shape: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X"),
    label_size: float = 3.0,
    dot_alpha: float = 0.6,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: str = "Outgoing interaction strength",
    ylabel: str = "Incoming interaction strength",
    title: Optional[str] = None,
    font_size: float = 10.0,
    font_size_title: float = 10.0,
    do_label: bool = True,
    show_legend: bool = True,
    show_axes: bool = True,
) -> Figure:
    """Dominant senders / receivers 2-D scatter.

    Mirrors :func:`netAnalysis_signalingRole_scatter` (R: ``analysis.R:2642``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_cellchat(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] missing.")
    slot = cc[slot_name]
    centr = slot.get("centr")
    if not centr:
        raise ValueError(
            "Please run `netAnalysis_computeCentrality` to compute the network "
            "centrality scores!"
        )
    centr_arr = np.asarray(centr["values"], dtype=float)
    centr_names = list(centr["centr_names"])
    node_names = list(centr["node_names"])
    pathway_names = list(centr["signaling_names"])
    if x_measure not in centr_names or y_measure not in centr_names:
        raise ValueError(
            f"x.measure / y.measure must be one of {centr_names!r}"
        )
    out_idx = centr_names.index(x_measure)
    in_idx = centr_names.index(y_measure)
    outgoing = centr_arr[out_idx, :, :]  # (K, N)
    incoming = centr_arr[in_idx, :, :]

    if signaling is not None:
        keep = [i for i, p in enumerate(pathway_names) if p in set(signaling)]
        if not keep:
            raise ValueError(
                "There is no significant communication for the input signaling."
            )
        outgoing = outgoing[:, keep]
        incoming = incoming[:, keep]

    out_cells = outgoing.sum(axis=1)
    in_cells = incoming.sum(axis=1)

    # num.link = aggregateNet(...)$count rowSums + colSums - diag
    from .modeling import aggregateNet

    net_dict = aggregateNet(
        adata, signaling=list(signaling) if signaling else None,
        return_object=False, remove_isolate=False,
    )
    count_mat = np.asarray(net_dict["count"], dtype=float)
    num_link = count_mat.sum(axis=1) + count_mat.sum(axis=0) - np.diag(count_mat)

    df = pd.DataFrame({
        "x": out_cells,
        "y": in_cells,
        "labels": node_names,
        "Count": num_link,
    })
    if group is not None:
        df["Group"] = list(group)

    if color_use is None:
        color_use = list(scPalette(len(node_names)))
    color_use = list(color_use)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    if weight_MinMax is None:
        size_min, size_max = float(df["Count"].min()), float(df["Count"].max())
    else:
        size_min, size_max = float(weight_MinMax[0]), float(weight_MinMax[1])

    if size_max == size_min:
        sizes_pt = np.full(len(df), float(dot_size[1])) ** 2 * 8.0
    else:
        norm = (df["Count"].to_numpy() - size_min) / (size_max - size_min)
        norm = np.clip(norm, 0.0, 1.0)
        sizes_pt = (norm * (dot_size[1] - dot_size[0]) + dot_size[0]) ** 2 * 8.0

    for k, lbl in enumerate(df["labels"]):
        marker = "o"
        if "Group" in df.columns:
            grp_levels = list(pd.Categorical(df["Group"]).categories)
            marker = point_shape[grp_levels.index(df["Group"].iloc[k]) % len(point_shape)]
        ax.scatter(
            df["x"].iloc[k], df["y"].iloc[k],
            s=sizes_pt[k],
            facecolors=matplotlib.colors.to_rgba(color_use[k % len(color_use)], alpha=dot_alpha),
            edgecolors=color_use[k % len(color_use)],
            linewidth=0.5,
            marker=marker,
            label=lbl,
        )

    if do_label:
        repel_text(
            ax,
            df["x"].to_numpy(dtype=float),
            df["y"].to_numpy(dtype=float),
            df["labels"].astype(str).tolist(),
            fontsize=label_size * 2.4,
            color=[color_use[k % len(color_use)] for k in range(len(df))],
        )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size_title)
    if show_legend:
        ax.legend(loc="best", fontsize=font_size - 2)
    if not show_axes:
        ax.set_axis_off()
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_diff_signalingRole_scatter (R: analysis.R:2773)
# ---------------------------------------------------------------------------
def _build_mat_all_per_dataset(
    adata_list: Sequence[ad.AnnData],
    *,
    comparison: Sequence[int],
    signaling: Sequence[str],
    cell_levels: Sequence[str],
    x_measure: str,
    y_measure: str,
    slot_name: str,
) -> List[np.ndarray]:
    """Build the per-dataset ``(n_signaling, n_cells, 2)`` mat.all arrays.

    R parity (``analysis.R:2799-2832``): for each dataset in ``comparison``,
    pull ``centr`` from ``adata.uns['cellchat'][slot_name]['centr']``,
    extract outgoing/incoming per-cell × per-signaling, and pad to the
    joint ``(signaling, cell_levels)`` index. Returns one
    ``(n_sig, n_cell, 2)`` array per dataset.

    Layer 3 channels: index ``0`` = outgoing (``x_measure``);
    index ``1`` = incoming (``y_measure``). Mirrors
    ``dimnames(mat.all) <- list(..., c("outgoing", "incoming"))``.
    """
    cell_pos = {c: k for k, c in enumerate(cell_levels)}
    sig_pos = {s: k for k, s in enumerate(signaling)}
    out_list: List[np.ndarray] = []
    for ds_idx in comparison:
        cc_ds = adata_list[ds_idx].uns.get("cellchat", {})
        slot_ds = cc_ds.get(slot_name, {})
        centr = slot_ds.get("centr")
        if not isinstance(centr, dict) or "values" not in centr:
            raise ValueError(
                f"adata_list[{ds_idx}].uns['cellchat']['{slot_name}']['centr'] "
                "is missing — run `netAnalysis_computeCentrality` for each "
                "dataset separately (R: analysis.R:2802)."
            )
        centr_names = list(centr["centr_names"])
        for m in (x_measure, y_measure):
            if m not in centr_names:
                raise ValueError(
                    f"`x_measure, y_measure` should be one of "
                    f"{centr_names!r}; got {m!r} (R: analysis.R:2805)."
                )
        node_names = list(centr["node_names"])
        sig_names_ds = list(centr["signaling_names"])
        values = np.asarray(centr["values"], dtype=float)  # (centr, node, sig)
        x_idx = centr_names.index(x_measure)
        y_idx = centr_names.index(y_measure)
        outgoing_ds = values[x_idx, :, :]
        incoming_ds = values[y_idx, :, :]

        mat_out = np.zeros((len(signaling), len(cell_levels)), dtype=float)
        mat_in = np.zeros((len(signaling), len(cell_levels)), dtype=float)
        for sj, sname in enumerate(sig_names_ds):
            if sname not in sig_pos:
                continue
            tj = sig_pos[sname]
            for ci, cname in enumerate(node_names):
                if cname not in cell_pos:
                    continue
                ti = cell_pos[cname]
                mat_out[tj, ti] = outgoing_ds[ci, sj]
                mat_in[tj, ti] = incoming_ds[ci, sj]
        out_list.append(np.stack([mat_out, mat_in], axis=2))
    return out_list


def _resolve_default_signaling(
    adata_list: Sequence[ad.AnnData],
    *,
    comparison: Sequence[int],
    slot_name: str,
    signaling: Optional[Sequence[str]],
    signaling_exclude: Optional[Sequence[str]],
) -> List[str]:
    """R parity (``analysis.R:2792-2796``): default signaling = union of
    per-dataset pathway lists, optionally minus ``signaling_exclude``.
    Uses base R's ``union()`` semantics (first-appearance order).
    """
    if signaling is None:
        sig_set: Dict[str, None] = {}
        for ds_idx in comparison:
            slot_ds = adata_list[ds_idx].uns.get("cellchat", {}).get(slot_name, {})
            for s in slot_ds.get("pathways", []):
                sig_set.setdefault(str(s), None)
        result = list(sig_set)
    else:
        result = [str(s) for s in signaling]
    if signaling_exclude is not None:
        excl = set(map(str, signaling_exclude))
        result = [s for s in result if s not in excl]
    if len(result) == 0:
        raise ValueError(
            "No signaling pathways remain after applying `signaling_exclude`."
        )
    return result


def compute_diff_signalingRole_data(
    adata_list: Sequence[ad.AnnData],
    *,
    comparison: Sequence[int] = (1, 2),
    signaling: Optional[Sequence[str]] = None,
    signaling_exclude: Optional[Sequence[str]] = None,
    slot_name: str = "netP",
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
) -> Dict[str, Any]:
    """Pure data-side port of :func:`netAnalysis_diff_signalingRole_scatter`.

    Returns the intermediate values that R produces inside the function,
    so a downstream test (or alternative renderer) can pin them
    independently of matplotlib output.

    R parity references:
        - ``mat.diff <- mat.all.merged[[2]] - mat.all.merged[[1]]``  (analysis.R:2836)
        - ``outgoing.diff <- colSums(mat.diff[ , , 1])``             (analysis.R:2838)
        - ``incoming.diff <- colSums(mat.diff[ , , 2])``             (analysis.R:2839)

    Returns
    -------
    dict with keys
        ``cell_levels``    : list[str]   — joint cell-group levels
        ``signaling``      : list[str]   — pathway list (after exclude)
        ``mat_all_merged`` : list[ndarray] — per-dataset (n_sig, n_cell, 2)
        ``mat_diff``       : ndarray (n_sig, n_cell, 2)
        ``outgoing_diff``  : ndarray (n_cell,)
        ``incoming_diff``  : ndarray (n_cell,)
    """
    if len(adata_list) < 2:
        raise ValueError(
            "At least 2 AnnDatas required (R parity: analysis.R:2779)."
        )
    cell_levels = joint_idents_levels(adata_list)
    sig_list = _resolve_default_signaling(
        adata_list, comparison=[int(c) - 1 for c in comparison],
        slot_name=slot_name, signaling=signaling,
        signaling_exclude=signaling_exclude,
    )
    cmp_idx = [int(c) - 1 for c in comparison]
    mat_all_merged = _build_mat_all_per_dataset(
        adata_list, comparison=cmp_idx, signaling=sig_list,
        cell_levels=cell_levels, x_measure=x_measure, y_measure=y_measure,
        slot_name=slot_name,
    )
    mat_diff = mat_all_merged[1] - mat_all_merged[0]
    outgoing_diff = mat_diff[:, :, 0].sum(axis=0)
    incoming_diff = mat_diff[:, :, 1].sum(axis=0)
    return {
        "cell_levels": list(cell_levels),
        "signaling": list(sig_list),
        "mat_all_merged": mat_all_merged,
        "mat_diff": mat_diff,
        "outgoing_diff": outgoing_diff,
        "incoming_diff": incoming_diff,
    }


def compute_signalingChanges_data(
    adata_list: Sequence[ad.AnnData],
    *,
    idents_use: str,
    comparison: Sequence[int] = (1, 2),
    signaling: Optional[Sequence[str]] = None,
    signaling_exclude: Optional[Sequence[str]] = None,
    slot_name: str = "netP",
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    dataset_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Pure data-side port of :func:`netAnalysis_signalingChanges_scatter`.

    Returns the classification intermediates so they can be pinned to R
    references regardless of plot rendering.

    R parity references:
        - idx.specific  = use1 * use2                              (analysis.R:3008)
        - mat.sum       = use2 + use1                              (analysis.R:3009)
        - mat.diff      = use2 - use1                              (analysis.R:3013)
        - keep          = rowSums(mat.diff) != 0                   (analysis.R:3014)
        - specificity.out.in ∈ {0,-1,1,2}                          (analysis.R:3019-3022)
        - specificity ∈ {0,-1,1}                                   (analysis.R:3027-3029)

    Returns
    -------
    dict with keys
        ``cell_levels``               : list[str]
        ``signaling``                 : list[str]   (after exclude, BEFORE keep filter)
        ``names``                     : list[str]   (R: ``add.names``)
        ``use1`` / ``use2``           : ndarray (n_sig, 2)
        ``idx_specific`` / ``mat_sum`` : ndarray (n_sig, 2)
        ``mat_diff_kept``             : ndarray (n_kept, 2)
        ``signaling_kept``            : list[str]
        ``out_specific_signaling``    : list[str]   (any-row-spec, BEFORE keep)
        ``in_specific_signaling``     : list[str]
        ``specificity_out_in``        : ndarray (n_kept,) int  ∈ {0,-1,1,2}
        ``specificity``               : ndarray (n_kept,) int  ∈ {0,-1,1}
    """
    if len(adata_list) < 2:
        raise ValueError(
            "At least 2 AnnDatas required (R parity: analysis.R:2940)."
        )
    cell_levels = joint_idents_levels(adata_list)
    if idents_use not in cell_levels:
        raise ValueError(
            f"idents_use={idents_use!r} not in joint cell levels "
            f"{cell_levels!r} (R: analysis.R:2962)."
        )
    names = resolve_dataset_names(adata_list, dataset_names)
    sig_list = _resolve_default_signaling(
        adata_list, comparison=[int(c) - 1 for c in comparison],
        slot_name=slot_name, signaling=signaling,
        signaling_exclude=signaling_exclude,
    )
    cmp_idx = [int(c) - 1 for c in comparison]
    mat_all_merged = _build_mat_all_per_dataset(
        adata_list, comparison=cmp_idx, signaling=sig_list,
        cell_levels=cell_levels, x_measure=x_measure, y_measure=y_measure,
        slot_name=slot_name,
    )
    cell_idx = cell_levels.index(idents_use)
    use1 = mat_all_merged[0][:, cell_idx, :]
    use2 = mat_all_merged[1][:, cell_idx, :]

    idx_specific = use1 * use2
    mat_sum = use2 + use1
    out_specific_signaling = [
        sig_list[i] for i in range(len(sig_list))
        if mat_sum[i, 0] != 0 and idx_specific[i, 0] == 0
    ]
    in_specific_signaling = [
        sig_list[i] for i in range(len(sig_list))
        if mat_sum[i, 1] != 0 and idx_specific[i, 1] == 0
    ]

    mat_diff = use2 - use1
    keep = mat_diff.sum(axis=1) != 0
    mat_diff_kept = mat_diff[keep, :]
    sig_kept = [sig_list[k] for k in np.flatnonzero(keep)]
    out_kept = np.array(
        [s in set(out_specific_signaling) for s in sig_kept], dtype=bool,
    )
    in_kept = np.array(
        [s in set(in_specific_signaling) for s in sig_kept], dtype=bool,
    )
    out_in_specific = out_kept & in_kept

    spec_out_in = np.zeros(mat_diff_kept.shape[0], dtype=int)
    spec_out_in[out_in_specific] = 2
    only_out = out_kept & ~out_in_specific
    only_in = in_kept & ~out_in_specific
    spec_out_in[only_out] = 1
    spec_out_in[only_in] = -1

    nonzero = spec_out_in != 0
    all_pos = (mat_diff_kept >= 0).sum(axis=1) == 2
    all_neg = (mat_diff_kept <= 0).sum(axis=1) == 2
    specificity = np.zeros(mat_diff_kept.shape[0], dtype=int)
    specificity[nonzero & all_pos] = 1
    specificity[nonzero & all_neg] = -1

    return {
        "cell_levels": list(cell_levels),
        "signaling": list(sig_list),
        "names": list(names),
        "use1": use1,
        "use2": use2,
        "idx_specific": idx_specific,
        "mat_sum": mat_sum,
        "mat_diff_kept": mat_diff_kept,
        "signaling_kept": sig_kept,
        "out_specific_signaling": list(out_specific_signaling),
        "in_specific_signaling": list(in_specific_signaling),
        "specificity_out_in": spec_out_in,
        "specificity": specificity,
    }


def netAnalysis_diff_signalingRole_scatter(
    adata_list: Sequence[ad.AnnData],
    color_use: Optional[Sequence[str]] = None,
    comparison: Sequence[int] = (1, 2),
    signaling: Optional[Sequence[str]] = None,
    signaling_exclude: Optional[Sequence[str]] = None,
    idents_exclude: Optional[Sequence[str]] = None,
    slot_name: str = "netP",
    group: Optional[Sequence] = None,
    dataset_names: Optional[Sequence[str]] = None,
    dot_size: float = 2.5,
    point_shape: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X"),
    label_size: float = 3.0,
    dot_alpha: float = 0.6,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: str = "Differential outgoing interaction strength",
    ylabel: str = "Differential incoming interaction strength",
    title: Optional[str] = None,
    font_size: float = 10.0,
    font_size_title: float = 10.0,
    do_label: bool = True,
    show_legend: bool = True,
    show_axes: bool = True,
) -> Figure:
    """Differential signaling-role 2-D scatter across two datasets.

    Direct port of R's :func:`netAnalysis_diff_signalingRole_scatter`
    (``analysis.R:2773``). R requires a merged CellChat object built via
    :func:`mergeCellChat`; in the Python port the natural equivalent is
    a ``Sequence[ad.AnnData]`` (R's ``mergeCellChat(list(c1, c2))`` is a
    no-op in this design — just pass ``[c1, c2]`` directly).

    Each cell group is plotted as **one point** whose
    :math:`x` = sum-over-signaling of differential outgoing strength
    (``mat.diff[, , 'outgoing']``) and :math:`y` = sum-over-signaling of
    differential incoming strength (``mat.diff[, , 'incoming']``), where
    ``mat.diff = mat.all.merged[[2]] - mat.all.merged[[1]]`` per R.

    Parameters
    ----------
    adata_list
        Per-dataset CellChat-state AnnDatas (``len ≥ 2`` required;
        single-dataset input raises, mirroring R's
        ``stop("...single dataset!")`` discriminator at
        ``analysis.R:2779``).
    comparison
        1-based dataset indices into ``adata_list`` (R parity:
        ``object@net[[comparison[1]]]`` etc).
    dataset_names
        Optional per-dataset labels (R parity: ``add.names``); defaults
        to ``["d1", "d2", ...]``.
    """
    if len(adata_list) < 2:
        raise ValueError(
            "netAnalysis_diff_signalingRole_scatter requires at least 2 "
            "AnnDatas (R parity: `is.list(object@net[[1]])` discriminator "
            "at analysis.R:2779). Pass `[adata_ctrl, adata_treat]`."
        )
    names = resolve_dataset_names(adata_list, dataset_names)
    i1 = int(comparison[0]) - 1
    i2 = int(comparison[1]) - 1

    # Delegate the math to the pure helper so tests can pin every R
    # intermediate (mat.diff, outgoing.diff, incoming.diff) without
    # going through matplotlib.
    data = compute_diff_signalingRole_data(
        adata_list,
        comparison=comparison,
        signaling=signaling,
        signaling_exclude=signaling_exclude,
        slot_name=slot_name,
        x_measure=x_measure,
        y_measure=y_measure,
    )
    cell_levels = data["cell_levels"]
    outgoing_diff = data["outgoing_diff"]
    incoming_diff = data["incoming_diff"]

    df = pd.DataFrame({
        "x": outgoing_diff,
        "y": incoming_diff,
        "labels": pd.Categorical(cell_levels, categories=cell_levels),
    })
    if group is not None:
        df["Group"] = list(group)

    if color_use is None:
        color_use = list(scPalette(len(cell_levels)))
    color_use = list(color_use)

    # R parity (analysis.R:2850-2853): drop excluded idents from BOTH
    # the data and the colour palette in lock-step.
    if idents_exclude is not None:
        keep_mask = ~df["labels"].astype(str).isin(set(map(str, idents_exclude)))
        df = df[keep_mask].reset_index(drop=True)
        color_use = [
            c for c, lv in zip(color_use, cell_levels)
            if lv not in set(map(str, idents_exclude))
        ]
        cell_levels = [lv for lv in cell_levels if lv not in set(map(str, idents_exclude))]

    if title is None:
        title = (
            f"Signaling changes  ({names[i1]} vs. {names[i2]})"
        )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cell_to_color = {lv: color_use[i] for i, lv in enumerate(cell_levels)}
    if "Group" in df.columns:
        # R: scale_shape_manual(values = point.shape[1:length(unique(df$Group))])
        groups_unique = list(pd.unique(df["Group"].astype(str)))
        shapes = list(point_shape)[: len(groups_unique)] or ["o"]
        for gi, gp in enumerate(groups_unique):
            sub = df[df["Group"].astype(str) == gp]
            ax.scatter(
                sub["x"], sub["y"],
                s=(dot_size ** 2) * 18,
                facecolors=[matplotlib.colors.to_rgba(cell_to_color[lv], dot_alpha)
                            for lv in sub["labels"].astype(str)],
                edgecolors=[cell_to_color[lv] for lv in sub["labels"].astype(str)],
                marker=shapes[gi % len(shapes)],
                linewidth=0.5,
            )
    else:
        ax.scatter(
            df["x"], df["y"],
            s=(dot_size ** 2) * 18,
            facecolors=[matplotlib.colors.to_rgba(cell_to_color[lv], dot_alpha)
                        for lv in df["labels"].astype(str)],
            edgecolors=[cell_to_color[lv] for lv in df["labels"].astype(str)],
            linewidth=0.5,
        )
    # R parity (analysis.R:2865): hline at y=0, vline at x=0 dashed grey50.
    ax.axhline(0.0, linestyle="--", color="0.5", linewidth=0.25)
    ax.axvline(0.0, linestyle="--", color="0.5", linewidth=0.25)

    if do_label and len(df) > 0:
        repel_text(
            ax,
            df["x"].to_numpy(dtype=float),
            df["y"].to_numpy(dtype=float),
            df["labels"].astype(str).tolist(),
            fontsize=label_size * 2.4,
            color=[cell_to_color[lv] for lv in df["labels"].astype(str)],
        )
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size_title)
    if not show_legend:
        for handle in ax.get_legend_handles_labels()[0]:
            handle.set_visible(False)
    if not show_axes:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_signalingChanges_scatter (R: analysis.R:2937)
# ---------------------------------------------------------------------------
def netAnalysis_signalingChanges_scatter(
    adata_list: Sequence[ad.AnnData],
    idents_use: str,
    color_use: Sequence[str] = ("grey10", "#F8766D", "#00BFC4"),
    comparison: Sequence[int] = (1, 2),
    signaling: Optional[Sequence[str]] = None,
    signaling_label: Optional[Sequence[str]] = None,
    top_label: float = 1.0,
    signaling_exclude: Optional[Sequence[str]] = None,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    slot_name: str = "netP",
    dataset_names: Optional[Sequence[str]] = None,
    dot_size: float = 2.5,
    point_shape: Sequence[str] = ("o", "s", "^", "D"),
    label_size: float = 3.0,
    dot_alpha: float = 0.6,
    x_measure: str = "outdeg",
    y_measure: str = "indeg",
    xlabel: str = "Differential outgoing interaction strength",
    ylabel: str = "Differential incoming interaction strength",
    title: Optional[str] = None,
    font_size: float = 10.0,
    font_size_title: float = 10.0,
    do_label: bool = True,
    show_legend: bool = True,
    show_axes: bool = True,
) -> Figure:
    """Per-cell-group differential outgoing / incoming signaling scatter.

    Direct port of R's :func:`netAnalysis_signalingChanges_scatter`
    (``analysis.R:2937``). For one ``idents_use`` cell group, classifies
    each pathway by **specificity** (3-way: Shared / dataset-A-specific /
    dataset-B-specific) and **specificity.out.in** (4-way: Shared /
    Outgoing-specific / Incoming-specific / Outgoing & Incoming-specific),
    mirroring R's idx.specific / mat.sum / mat.diff math at
    ``analysis.R:3007-3037``.

    Parameters
    ----------
    adata_list
        Per-dataset CellChat-state AnnDatas (``len ≥ 2`` required;
        single-dataset input raises, mirroring R's
        ``stop("...single dataset!")`` discriminator).
    idents_use
        Cell-group label whose per-pathway differential is plotted.
    comparison
        1-based dataset indices into ``adata_list`` (R parity).
    dataset_names
        Optional per-dataset labels (R: ``add.names``); defaults to
        ``["d1", "d2", ...]``.
    """
    if len(adata_list) < 2:
        raise ValueError(
            "netAnalysis_signalingChanges_scatter requires at least 2 "
            "AnnDatas (R parity: `is.list(object@net[[1]])` discriminator "
            "at analysis.R:2940). Pass `[adata_ctrl, adata_treat]`."
        )
    names = resolve_dataset_names(adata_list, dataset_names)
    i1 = int(comparison[0]) - 1
    i2 = int(comparison[1]) - 1

    # Delegate the math to the pure helper so tests can pin every R
    # intermediate (use1, use2, idx.specific, mat.sum, mat.diff,
    # specificity.out.in, specificity) without going through matplotlib.
    data = compute_signalingChanges_data(
        adata_list,
        idents_use=idents_use,
        comparison=comparison,
        signaling=signaling,
        signaling_exclude=signaling_exclude,
        slot_name=slot_name,
        x_measure=x_measure,
        y_measure=y_measure,
        dataset_names=dataset_names,
    )
    names = data["names"]
    cell_levels = data["cell_levels"]
    mat_diff = data["mat_diff_kept"]
    sig_kept = data["signaling_kept"]
    spec_out_in = data["specificity_out_in"]
    specificity = data["specificity"]

    # R parity (analysis.R:3032-3037): integer codes → categorical labels.
    out_in_category = (
        "Shared", "Incoming specific", "Outgoing specific",
        "Incoming & Outgoing specific",
    )
    specificity_category = (
        "Shared", f"{names[i1]} specific", f"{names[i2]} specific",
    )
    out_in_map = {0: out_in_category[0], -1: out_in_category[1],
                  1: out_in_category[2], 2: out_in_category[3]}
    spec_map = {0: specificity_category[0], -1: specificity_category[1],
                1: specificity_category[2]}
    df = pd.DataFrame({
        "outgoing": mat_diff[:, 0],
        "incoming": mat_diff[:, 1],
        "labels": sig_kept,
    })
    df["specificity.out.in"] = pd.Categorical(
        [out_in_map[v] for v in spec_out_in],
        categories=out_in_category, ordered=True,
    )
    df["specificity"] = pd.Categorical(
        [spec_map[v] for v in specificity],
        categories=specificity_category, ordered=True,
    )

    if title is None:
        title = (
            f"Signaling changes of {idents_use} "
            f"({names[i1]} vs. {names[i2]})"
        )

    # R parity (analysis.R:3039-3042): drop unused factor levels for the
    # palette + shape arrays so legends stay tight.
    out_in_used = list(df["specificity.out.in"].cat.remove_unused_categories().cat.categories)
    spec_used = list(df["specificity"].cat.remove_unused_categories().cat.categories)
    point_shape = list(point_shape)
    shape_used = [point_shape[out_in_category.index(c) % len(point_shape)]
                  for c in out_in_used]
    color_used = [color_use[specificity_category.index(c) % len(color_use)]
                  for c in spec_used]
    spec_to_color = dict(zip(spec_used, color_used))
    out_in_to_shape = dict(zip(out_in_used, shape_used))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for shp_lvl in out_in_used:
        for spc_lvl in spec_used:
            sel = (df["specificity.out.in"] == shp_lvl) & (df["specificity"] == spc_lvl)
            if not sel.any():
                continue
            ax.scatter(
                df.loc[sel, "outgoing"], df.loc[sel, "incoming"],
                s=(dot_size ** 2) * 18,
                facecolors=matplotlib.colors.to_rgba(spec_to_color[spc_lvl], dot_alpha),
                edgecolors=spec_to_color[spc_lvl],
                marker=out_in_to_shape[shp_lvl],
                linewidth=0.5,
                label=f"{spc_lvl} | {shp_lvl}",
            )
    # R parity (analysis.R:3049): hline at 0, vline at 0 dashed grey50.
    ax.axhline(0.0, linestyle="--", color="0.5", linewidth=0.25)
    ax.axvline(0.0, linestyle="--", color="0.5", linewidth=0.25)

    if do_label:
        # R parity (analysis.R:3066-3076): label top |x|/|y| pathways
        # (top.label fraction) or use signaling_label whitelist.
        if signaling_label is None:
            xy = df[["outgoing", "incoming"]].abs().to_numpy(dtype=float)
            if xy.size > 0:
                thresh = float(np.quantile(xy, 1.0 - float(top_label)))
                lbl_mask = (df["outgoing"].abs() > thresh) | (df["incoming"].abs() > thresh)
            else:
                lbl_mask = pd.Series([False] * len(df))
        else:
            lbl_mask = df["labels"].isin(set(map(str, signaling_label)))
        sub = df[lbl_mask]
        if len(sub) > 0:
            repel_text(
                ax,
                sub["outgoing"].to_numpy(dtype=float),
                sub["incoming"].to_numpy(dtype=float),
                sub["labels"].astype(str).tolist(),
                fontsize=label_size * 2.4,
                color=[spec_to_color[s] for s in sub["specificity"].astype(str)],
            )

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size_title)
    if xlims is not None:
        ax.set_xlim(*xlims)
    if ylims is not None:
        ax.set_ylim(*ylims)
    if show_legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="best")
    if not show_axes:
        ax.set_axis_off()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_network (R: analysis.R:2559)
# ---------------------------------------------------------------------------
def netAnalysis_signalingRole_network(
    adata: ad.AnnData,
    signaling: Sequence[str],
    slot_name: str = "netP",
    measure: Sequence[str] = ("outdeg", "indeg", "flowbet", "info"),
    measure_name: Sequence[str] = ("Sender", "Receiver", "Mediator", "Influencer"),
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "BuGn",
    width: float = 6.5,
    height: float = 1.4,
    font_size: float = 8.0,
    font_size_title: float = 10.0,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
) -> Figure:
    """Per-pathway signaling-role heatmap.

    Mirrors :func:`netAnalysis_signalingRole_network`
    (R: ``analysis.R:2559``). Draws one mini-heatmap per signaling
    pathway showing measure (rows) × cell-group (columns) row-normalised
    centrality.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_cellchat(adata)
    slot = cc.get(slot_name, {})
    centr = slot.get("centr")
    if not centr:
        raise ValueError(
            "Please run `netAnalysis_computeCentrality` to compute the network "
            "centrality scores!"
        )
    centr_arr = np.asarray(centr["values"], dtype=float)
    centr_names = list(centr["centr_names"])
    node_names = list(centr["node_names"])
    pathway_names = list(centr["signaling_names"])
    measure = list(measure)
    measure_name = list(measure_name)
    available = [m for m in measure if m in centr_names]
    if not available:
        raise ValueError(
            f"None of measure {measure!r} are in centr names {centr_names!r}"
        )
    sig_kept = [s for s in signaling if s in pathway_names]
    if not sig_kept:
        raise ValueError(f"signaling {signaling!r} not in pathways.")

    if color_use is None:
        color_use = list(scPalette(len(node_names)))
    color_use = list(color_use)
    cols = list(_safe_brewer(color_heatmap, 9))
    cmap = LinearSegmentedColormap.from_list("nsr_brew", cols, N=256)

    # R: ``width`` and ``height`` are in centimetres for the heatmap body.
    # Convert to inches for the matplotlib figure, leaving a small slot for
    # the bottom colour strip, the column-rotated tick labels, the title
    # and the right-side legend (legend_height = unit(20, "mm") in R).
    cm_to_in = 1.0 / 2.54
    body_w_in = float(width) * cm_to_in
    body_h_in = float(height) * cm_to_in
    legend_w_in = 1.0  # ~25 mm — vertical bar + rotated title slot + min/max ticks
    bottom_strip_in = 0.10  # 0.2 cm color strip
    label_pad_in = 0.9       # rotated x-tick labels + xlabel area
    title_pad_in = 0.45
    left_pad_in = 1.1        # row labels (measure names)

    nrow = len(sig_kept)
    fig_w = left_pad_in + body_w_in + 0.05 + legend_w_in + 0.20
    per_panel_h = title_pad_in + body_h_in + bottom_strip_in + label_pad_in + 0.15
    fig_h = per_panel_h * nrow
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Outer gridspec stacks one heatmap per signaling pathway.
    outer = fig.add_gridspec(
        nrow, 1,
        hspace=(0.50 / max(per_panel_h, 1e-3)),
    )

    for i, sig in enumerate(sig_kept):
        sig_idx = pathway_names.index(sig)
        mat = np.zeros((len(available), len(node_names)), dtype=float)
        for r, m in enumerate(available):
            mat[r, :] = centr_arr[centr_names.index(m), :, sig_idx]
        # R: ``mat <- sweep(mat, 1L, apply(mat, 1, max), '/')`` — row-max normalise
        rmax = mat.max(axis=1, keepdims=True)
        rmax = np.where(rmax == 0, 1.0, rmax)
        mat_norm = mat / rmax

        # Inner gridspec: dedicated columns/rows for body, bottom strip, legend.
        # Heights / widths are in inches, mapped through height_ratios /
        # width_ratios so the body is rendered at exactly (width × height) cm.
        height_ratios = [
            title_pad_in,           # 0: title slot
            body_h_in,              # 1: heatmap body
            bottom_strip_in,        # 2: cell-group color strip (R: bottom_annotation)
            label_pad_in,           # 3: rotated column labels
        ]
        # Legend column is split: a slim ``cax`` for the bar and a slot for
        # the rotated "Importance" title to its left, so the title can never
        # overlap the bar (matplotlib's ``set_ylabel`` falls back inside a
        # narrow cax in some backends).
        width_ratios = [
            left_pad_in,            # 0: row labels
            body_w_in,              # 1: heatmap body
            0.20,                   # 2: gap between body and legend block
            legend_w_in * 0.45,     # 3: rotated title slot
            legend_w_in * 0.30,     # 4: colorbar
            legend_w_in * 0.25,     # 5: tick label slot
        ]
        inner = outer[i, 0].subgridspec(
            4, 6,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.0, wspace=0.0,
        )
        ax_main = fig.add_subplot(inner[1, 1])
        ax_strip = fig.add_subplot(inner[2, 1], sharex=ax_main)
        ax_cbar = fig.add_subplot(inner[1, 4])

        im = ax_main.imshow(
            mat_norm, cmap=cmap, aspect="auto", interpolation="nearest",
        )
        # R: ``row_names_side = "left"``, ``row_names_rot = 0``
        ax_main.set_yticks(np.arange(len(available)))
        ax_main.set_yticklabels(
            [measure_name[measure.index(m)] for m in available],
            fontsize=font_size, rotation=0,
        )
        # Tick *positions* on the x axis are shared with ax_strip (sharex);
        # call ``set_xticks([])`` on a sharex axis would clear the strip's
        # ticks too, so suppress the labels here via tick_params instead.
        ax_main.set_xticks(np.arange(len(node_names)))
        ax_main.tick_params(left=True, right=False, top=False, bottom=False,
                            labelbottom=False)
        for spine in ax_main.spines.values():
            spine.set_visible(False)
        ax_main.set_title(
            f"{sig} signaling pathway network",
            fontsize=font_size_title,
            pad=4.0,
        )

        # Bottom colour strip — one cell per cell-group, coloured by ``color_use``.
        strip_rgb = np.array(
            [[matplotlib.colors.to_rgb(c) for c in color_use[: len(node_names)]]]
        )
        ax_strip.imshow(strip_rgb, aspect="auto", interpolation="nearest")
        # ``set_xticks`` MUST precede ``set_xticklabels`` — otherwise
        # matplotlib emits ``UserWarning: FixedFormatter should only be used
        # together with FixedLocator`` and the labels can drift on resize.
        ax_strip.set_xticks(np.arange(len(node_names)))
        ax_strip.set_xticklabels(
            node_names, rotation=45, ha="right", fontsize=font_size,
        )
        ax_strip.tick_params(top=False, bottom=True, left=False, right=False,
                             labelleft=False)
        for spine in ax_strip.spines.values():
            spine.set_visible(False)

        # Legend (R: ``heatmap_legend_param`` → leftcenter-rot title, height 20mm,
        # ticks at min/max of ``mat_norm`` rounded to 1 digit). Title is placed
        # outside the colorbar via an axes-level text artist so it doesn't
        # overlap the bar (matplotlib's ``set_ylabel`` with a narrow cax falls
        # back inside the bar).
        cbar = fig.colorbar(im, cax=ax_cbar)
        vmin = float(np.nanmin(mat_norm))
        vmax = float(np.nanmax(mat_norm))
        cbar.set_ticks([round(vmin, 1), round(vmax, 1)])
        cbar.ax.tick_params(labelsize=font_size - 1)
        cbar.outline.set_visible(False)
        # Title sits in its own gridspec slot (inner[1, 3]) so it is
        # guaranteed to land between the heatmap body and the colorbar.
        ax_cbar_title = fig.add_subplot(inner[1, 3])
        ax_cbar_title.set_axis_off()
        ax_cbar_title.text(
            0.5, 0.5, "Importance",
            transform=ax_cbar_title.transAxes,
            rotation=90, va="center", ha="center",
            fontsize=font_size,
        )
    return fig


# ---------------------------------------------------------------------------
# netAnalysis_contribution (R: analysis.R:229)
# ---------------------------------------------------------------------------
def netAnalysis_contribution(
    adata: ad.AnnData,
    signaling: str,
    signaling_name: Optional[str] = None,
    do_group: bool = True,
    width: float = 0.1,
    vertex_receiver: Optional[Sequence[int]] = None,
    thresh: float = 0.05,
    return_data: bool = False,
    x_rotation: float = 0.0,
    title: str = "Contribution of each L-R pair",
    font_size: float = 10.0,
    font_size_title: float = 10.0,
) -> Figure:
    """Per L-R-pair relative contribution to a signaling pathway.

    Mirrors :func:`netAnalysis_contribution` (R: ``analysis.R:229``).
    Reads ``adata.uns['cellchat']['net']['prob']`` and the LR table to
    build the per-LR-pair relative contribution bar plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_cellchat(adata)
    if "net" not in cc:
        raise ValueError("adata.uns['cellchat']['net'] is missing.")
    net = cc["net"]
    LR = cc.get("LR", {}).get("LRsig")
    if LR is None or not isinstance(LR, pd.DataFrame):
        raise ValueError(
            "adata.uns['cellchat']['LR']['LRsig'] is required (run "
            "identifyOverExpressedInteractions first)."
        )
    pair_in_signaling = LR.index[LR["pathway_name"].astype(str) == str(signaling)]
    if pair_in_signaling.empty:
        raise ValueError(f"signaling {signaling!r} not found in LR table.")
    if signaling_name is None:
        signaling_name = signaling

    if do_group:
        prob = np.asarray(net["prob"], dtype=float).copy()
        pval = np.asarray(net["pval"], dtype=float)
        prob[pval > thresh] = 0.0
    else:
        if "prob.cell" not in net:
            raise ValueError("net['prob.cell'] missing for do_group=False.")
        prob = np.asarray(net["prob.cell"].todense() if hasattr(net["prob.cell"], "todense") else net["prob.cell"])
    pair_names = list(net.get("dimnames", {}).get("interaction_name",
                                                  list(LR.index.astype(str))))
    pair_in_data = [p for p in pair_in_signaling.tolist() if p in pair_names]
    if not pair_in_data:
        raise ValueError(f"There is no significant communication of {signaling_name}")

    idx = [pair_names.index(p) for p in pair_in_data]
    sub = prob[:, :, idx]
    sums3 = sub.sum(axis=(0, 1))
    keep = np.flatnonzero(sums3 > 0)
    if keep.size == 0:
        raise ValueError(f"There is no significant communication of {signaling_name}")
    sub = sub[:, :, keep]
    pair_in_data = [pair_in_data[i] for i in keep]

    if sub.size > 0:
        sub_norm = (sub - sub.min()) / max(sub.max() - sub.min(), 1e-12)
    else:
        sub_norm = sub

    # Pretty names
    interaction = cc.get("DB", {}).get("interaction")
    if interaction is not None and "interaction_name_2" in interaction.columns:
        labels = [
            interaction.loc[p, "interaction_name_2"] if p in interaction.index else p
            for p in pair_in_data
        ]
    else:
        labels = list(pair_in_data)

    if vertex_receiver is None:
        pSum = sub_norm.sum(axis=(0, 1))
        pSum_max = sub_norm.sum() or 1.0
        pSum = pSum / pSum_max
        pSum[np.isnan(pSum)] = 0.0
        y_lim = max(float(pSum.max()), 1e-6)

        df = pd.DataFrame({"name": labels, "contribution": pSum})
        if len(df) < 10:
            pad_n = 10 - len(df)
            df = pd.concat(
                [df, pd.DataFrame({"name": [""] * pad_n, "contribution": [0.0] * pad_n})],
                ignore_index=True,
            )
        df = df.sort_values("contribution", ascending=False, kind="stable").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.3 * len(df) + 2.0)))
        _barh_pane(
            ax, df, y_lim, width=width, font_size=font_size, x_rotation=x_rotation,
        )
        ax.set_title(title, fontsize=font_size_title, ha="center")
        if return_data:
            fig._netAnalysis_contribution_df = df  # type: ignore[attr-defined]
        return fig

    # ---- 3-panel All / Hierarchy1 / Hierarchy2 (R: analysis.R:309-359) ----
    vr = np.asarray(list(vertex_receiver), dtype=int)
    n_src = sub_norm.shape[0]
    complement = np.array([i for i in range(n_src) if i not in set(vr.tolist())], dtype=int)

    # Panel "All": pSum over full normalized prob.
    pSum_all = sub_norm.sum(axis=(0, 1))
    pSum_max = sub_norm.sum() or 1.0
    pSum_all = pSum_all / pSum_max
    pSum_all[np.isnan(pSum_all)] = 0.0
    y_lim = max(float(pSum_all.max()), 1e-6)  # R reuses "All" y_lim across panels.

    # Panel "Hierarchy1": pSum restricted to vertex_receiver targets.
    pSum_h1 = sub_norm[:, vr, :].sum(axis=(0, 1)) / pSum_max
    pSum_h1[np.isnan(pSum_h1)] = 0.0

    # Panel "Hierarchy2": pSum restricted to complement (non-vertex_receiver) targets.
    if complement.size > 0:
        pSum_h2 = sub_norm[:, complement, :].sum(axis=(0, 1)) / pSum_max
    else:
        pSum_h2 = np.zeros_like(pSum_all)
    pSum_h2[np.isnan(pSum_h2)] = 0.0

    df_all = pd.DataFrame({"name": labels, "contribution": pSum_all})
    df_h1 = pd.DataFrame({"name": labels, "contribution": pSum_h1})
    df_h2 = pd.DataFrame({"name": labels, "contribution": pSum_h2})

    fig = plt.figure(figsize=(12.0, max(3.0, 0.3 * len(df_all) + 2.5)))
    gs = fig.add_gridspec(nrows=2, ncols=3, height_ratios=[0.1, 1.0])
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.5,
        f"Contribution of each signaling in {signaling_name} pathway",
        ha="center", va="center", fontsize=font_size_title, fontweight="bold",
    )
    ax_all = fig.add_subplot(gs[1, 0])
    ax_h1 = fig.add_subplot(gs[1, 1], sharey=ax_all)
    ax_h2 = fig.add_subplot(gs[1, 2], sharey=ax_all)
    for sub_ax, sub_df, sub_title in (
        (ax_all, df_all, "All"),
        (ax_h1, df_h1, "Hierarchy1"),
        (ax_h2, df_h2, "Hierarchy2"),
    ):
        _barh_pane(
            sub_ax, sub_df, y_lim, width=width, font_size=font_size, x_rotation=x_rotation,
        )
        sub_ax.set_title(sub_title, fontsize=font_size_title, ha="center")
    # Hide y-axis labels on shared panels for visual cleanliness.
    for sub_ax in (ax_h1, ax_h2):
        plt.setp(sub_ax.get_yticklabels(), visible=False)
        sub_ax.set_ylabel("")
    if return_data:
        fig._netAnalysis_contribution_df = df_all  # type: ignore[attr-defined]
        fig._netAnalysis_contribution_df_h1 = df_h1  # type: ignore[attr-defined]
        fig._netAnalysis_contribution_df_h2 = df_h2  # type: ignore[attr-defined]
    return fig


def _barh_pane(
    ax,
    df: pd.DataFrame,
    y_lim: float,
    *,
    width: float,
    font_size: float,
    x_rotation: float,
) -> None:
    """Render a single horizontal-bar contribution panel onto ``ax``.

    Mirrors the per-panel ``geom_bar(stat='identity') + coord_flip()`` portion
    of R ``netAnalysis_contribution`` so all three panels render identically.
    """
    ax.barh(
        np.arange(len(df))[::-1], df["contribution"], color=_safe_color("grey50"),
        height=max(0.3, 1.0 - 6.0 * width),
    )
    ax.set_yticks(np.arange(len(df))[::-1])
    ax.set_yticklabels(df["name"], fontsize=font_size, rotation=x_rotation)
    ax.set_xlabel("Relative contribution", fontsize=font_size)
    ax.set_xlim(0.0, y_lim)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# ---------------------------------------------------------------------------
# netAnalysis_river (R: visualization.R:3793)
# ---------------------------------------------------------------------------
def _alluvial_polygon(
    x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float,
    color: str, alpha: float = 0.4,
) -> PathPatch:
    """Smooth alluvial flow polygon between two stack columns."""
    # Cubic Bezier flow with control points at the column midpoints.
    mx = 0.5 * (x0 + x1)
    verts = [
        (x0, y0a),
        (mx, y0a), (mx, y1a), (x1, y1a),  # top curve
        (x1, y1b),
        (mx, y1b), (mx, y0b), (x0, y0b),  # bottom curve
        (x0, y0a),  # close
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    path = Path(verts, codes)
    return PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha)


def _draw_alluvial_pair(
    ax,
    df_left_right: pd.DataFrame,
    left_col: str,
    right_col: str,
    value_col: str,
    color_left: Mapping[str, str],
    color_right: Mapping[str, str],
    bar_width: float = 0.05,
    flow_color: Optional[str] = None,
    font_size: float = 8.0,
    label_left: str = "left",
    label_right: str = "right",
) -> None:
    """Draw a 2-column alluvial diagram on ``ax``."""
    color_left = {k: _safe_color(v) for k, v in dict(color_left).items()}
    color_right = {k: _safe_color(v) for k, v in dict(color_right).items()}
    if flow_color is not None:
        flow_color = _safe_color(flow_color)
    left_levels = list(dict.fromkeys(df_left_right[left_col].astype(str).tolist()))
    right_levels = list(dict.fromkeys(df_left_right[right_col].astype(str).tolist()))

    # Compute totals per level
    left_totals = (
        df_left_right.groupby(left_col)[value_col].sum()
        .reindex(left_levels).fillna(0).to_numpy(dtype=float)
    )
    right_totals = (
        df_left_right.groupby(right_col)[value_col].sum()
        .reindex(right_levels).fillna(0).to_numpy(dtype=float)
    )

    L_total = float(left_totals.sum()) or 1.0
    R_total = float(right_totals.sum()) or 1.0
    scale_y = 1.0
    # Use separate left / right cumulative trackers
    left_cum = np.cumsum(np.concatenate([[0.0], left_totals / L_total]))
    right_cum = np.cumsum(np.concatenate([[0.0], right_totals / R_total]))

    # Per-cell trackers for stacked sub-flows
    left_used = left_cum[:-1].copy()
    right_used = right_cum[:-1].copy()

    # Pivot to (left x right) value matrix
    piv = (
        df_left_right.pivot_table(
            index=left_col, columns=right_col, values=value_col,
            aggfunc="sum", fill_value=0.0,
        ).reindex(index=left_levels, columns=right_levels, fill_value=0.0)
    )
    M = piv.to_numpy(dtype=float)

    x_left, x_right = 0.0, 1.0
    # Draw flow polygons
    for i, lvl_l in enumerate(left_levels):
        for j, lvl_r in enumerate(right_levels):
            v = M[i, j]
            if v <= 0.0:
                continue
            h_l = (v / L_total) * scale_y
            h_r = (v / R_total) * scale_y
            y0a = left_used[i]
            y0b = y0a + h_l
            y1a = right_used[j]
            y1b = y1a + h_r
            colour = flow_color or color_left.get(lvl_l, "#888888")
            poly = _alluvial_polygon(
                x_left + bar_width / 2, x_right - bar_width / 2,
                y0a, y0b, y1a, y1b,
                color=colour, alpha=0.4,
            )
            ax.add_patch(poly)
            left_used[i] += h_l
            right_used[j] += h_r

    # Draw stratum bars
    for i, lvl in enumerate(left_levels):
        ax.add_patch(plt.Rectangle(
            (x_left - bar_width / 2, left_cum[i]),
            bar_width, left_cum[i + 1] - left_cum[i],
            color=color_left.get(lvl, "#888888"),
            ec="black", linewidth=0.1, alpha=0.8,
        ))
        ax.text(
            x_left, 0.5 * (left_cum[i] + left_cum[i + 1]),
            lvl, ha="center", va="center", fontsize=font_size,
        )
    for j, lvl in enumerate(right_levels):
        ax.add_patch(plt.Rectangle(
            (x_right - bar_width / 2, right_cum[j]),
            bar_width, right_cum[j + 1] - right_cum[j],
            color=color_right.get(lvl, "#888888"),
            ec="black", linewidth=0.1, alpha=0.8,
        ))
        ax.text(
            x_right, 0.5 * (right_cum[j] + right_cum[j + 1]),
            lvl, ha="center", va="center", fontsize=font_size,
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([label_left, label_right], fontsize=font_size + 1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def netAnalysis_river(
    adata: ad.AnnData,
    slot_name: str = "netP",
    pattern: str = "outgoing",
    cutoff: float = 0.5,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    signaling: Optional[Sequence[str]] = None,
    color_use: Optional[Sequence[str]] = None,
    color_use_pattern: Optional[Sequence[str]] = None,
    color_use_signaling: Union[str, Sequence[str]] = "grey50",
    do_order: bool = False,
    main_title: Optional[str] = None,
    font_size: float = 2.5,
    font_size_title: float = 12.0,
) -> Figure:
    """Alluvial / river plot for communication patterns.

    Mirrors :func:`netAnalysis_river` (R: ``visualization.R:3793``).
    Reads the pattern data from
    ``adata.uns['cellchat'][slot_name]['pattern'][pattern]`` (written by
    :func:`identifyCommunicationPatterns`) and produces:

    - a single CellGroup-Pattern alluvial when only ``cell`` data exists,
    - a CellGroup-Pattern + Pattern-Signaling pair when both exist.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pattern not in ("outgoing", "incoming"):
        raise ValueError("pattern must be 'outgoing' or 'incoming'.")
    cc = _ensure_cellchat(adata)
    slot = cc.get(slot_name, {})
    res_pattern = slot.get("pattern", {}).get(pattern)
    if res_pattern is None:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['pattern'][{pattern!r}] missing."
        )
    data1 = res_pattern["pattern"]["cell"].copy()
    data2 = res_pattern["pattern"].get("signaling")
    if data2 is not None:
        data2 = data2.copy()

    if main_title is None:
        main_title = (
            "Outgoing communication patterns of secreting cells"
            if pattern == "outgoing"
            else "Incoming communication patterns of target cells"
        )

    nPatterns = len(data1["Pattern"].unique())
    if color_use_pattern is None:
        # ggPalette stride-2 trick
        gp = list(ggPalette(nPatterns * 2))
        if pattern == "outgoing":
            color_use_pattern = gp[0::2]
        else:
            color_use_pattern = gp[1::2]
    color_use_pattern = list(color_use_pattern)

    # R parity (visualization.R:3867-3885): subset by sources_use/targets_use
    # against ``levels(object@idents)``. Both filters operate on the
    # ``CellGroup`` column; numeric inputs index into the idents levels.
    cells_level = _get_idents_levels(adata, slot)
    if sources_use is not None:
        sources_use_list = list(sources_use)
        if all(isinstance(s, (int, np.integer)) for s in sources_use_list):
            sources_use_list = [cells_level[int(s) - 1] for s in sources_use_list]
        data1 = data1[data1["CellGroup"].isin(set(sources_use_list))].copy()
    if targets_use is not None:
        targets_use_list = list(targets_use)
        if all(isinstance(t, (int, np.integer)) for t in targets_use_list):
            targets_use_list = [cells_level[int(t) - 1] for t in targets_use_list]
        data1 = data1[data1["CellGroup"].isin(set(targets_use_list))].copy()

    cells_in_data = list(data1["CellGroup"].unique())
    if color_use is None:
        # R: ``color.use <- scPalette(length(cells.level))[cells.level %in% unique(plot.data$CellGroup)]``
        full_palette = list(scPalette(len(cells_level)))
        color_use = [
            full_palette[cells_level.index(c)] for c in cells_in_data if c in cells_level
        ]
        # Fallback for cell groups that aren't in cells_level (defensive — should not happen
        # because data1's CellGroup is sourced from the same idents factor).
        if len(color_use) < len(cells_in_data):
            color_use += list(scPalette(len(cells_in_data) - len(color_use)))
    color_use = list(color_use)

    plot1 = data1.copy()
    plot1.loc[plot1["Contribution"] < cutoff, "Contribution"] = 0.0

    # R parity (visualization.R:3833-3842 / 3888-3896): when do_order=TRUE,
    # reorder cell groups by hierarchical clustering of the
    # CellGroup × Pattern contribution matrix (avg linkage, cut into k = #
    # patterns with non-zero contribution).
    if do_order and len(cells_in_data) >= 2:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        mat = (
            plot1.pivot_table(
                index="CellGroup", columns="Pattern", values="Contribution",
                aggfunc="sum", fill_value=0.0,
            ).reindex(index=cells_in_data, fill_value=0.0)
        )
        if mat.shape[0] >= 2:
            dist_vec = pdist(mat.to_numpy(dtype=float), metric="euclidean")
            Z = linkage(dist_vec, method="average")
            k = max(1, plot1.loc[plot1["Contribution"] > 0, "Pattern"].nunique())
            cluster = fcluster(Z, t=k, criterion="maxclust")
            order = np.argsort(cluster, kind="stable")
            cells_in_data = [cells_in_data[i] for i in order]
            color_use = [color_use[i] for i in order]
            plot1["CellGroup"] = pd.Categorical(
                plot1["CellGroup"], categories=cells_in_data, ordered=True
            )
            plot1 = plot1.sort_values(by=["CellGroup", "Pattern"]).reset_index(drop=True)

    cell_color_map: Dict[str, str] = {
        c: color_use[i % len(color_use)] for i, c in enumerate(cells_in_data)
    }
    pattern_color_map: Dict[str, str] = {
        p: color_use_pattern[i % len(color_use_pattern)]
        for i, p in enumerate(sorted(data1["Pattern"].unique()))
    }

    if data2 is None:
        # single CellGroup x Pattern alluvial
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        _draw_alluvial_pair(
            ax, plot1, "CellGroup", "Pattern", "Contribution",
            cell_color_map, pattern_color_map,
            font_size=font_size * 3.0,
            label_left="Cell groups", label_right="Patterns",
        )
        fig.suptitle(main_title, fontsize=font_size_title)
        return fig

    plot2 = data2.copy()
    plot2.loc[plot2["Contribution"] < cutoff, "Contribution"] = 0.0
    if signaling is not None:
        plot2 = plot2[plot2["Signaling"].isin(set(signaling))]
    sig_levels = list(plot2["Signaling"].unique()) if len(plot2) > 0 else []
    if isinstance(color_use_signaling, str):
        sig_color_map = {s: color_use_signaling for s in sig_levels}
    else:
        sig_color_map = {
            s: list(color_use_signaling)[i % len(list(color_use_signaling))]
            for i, s in enumerate(sig_levels)
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    _draw_alluvial_pair(
        axes[0], plot1, "CellGroup", "Pattern", "Contribution",
        cell_color_map, pattern_color_map,
        font_size=font_size * 3.0,
        label_left="Cell groups", label_right="Patterns",
    )
    _draw_alluvial_pair(
        axes[1], plot2, "Pattern", "Signaling", "Contribution",
        pattern_color_map, sig_color_map,
        font_size=font_size * 3.0,
        label_left="Patterns", label_right="Signaling",
    )
    fig.suptitle(main_title, fontsize=font_size_title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig
