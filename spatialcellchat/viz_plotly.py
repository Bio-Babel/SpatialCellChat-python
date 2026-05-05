"""Interactive Plotly visualisations.

Direct port of the ``plotly_*`` exports in
``SpatialCellChat::visualization.R`` lines 6937-7071 (LR pair plot) and
7787-8074 (spatial dim plot).

These functions produce :class:`plotly.graph_objects.Figure` instances
suitable for use in Jupyter notebooks or Dash / Shiny dashboards.
The R reference renders via :func:`plotly::plot_ly`; the Python port
uses :mod:`plotly.graph_objects` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import sparse

from ._palette import scPalette
from .modeling import _ensure_uns, _get_idents

__all__ = [
    "plotly_spatialDimPlot",
    "plotly_spatialLRpairPlot",
]


def _get_coords_swapped(adata: ad.AnnData) -> pd.DataFrame:
    """Return ``obsm['spatial']`` as a labeled ``(x_cent, y_cent)`` frame.

    The function name is historical: R does swap because R ingests
    ``object@images$coordinates`` from Seurat's
    ``GetTissueCoordinates`` which yields ``(imagerow, imagecol)`` and
    R re-labels it to ``(x_cent=col, y_cent=row)``. The Python port
    receives ``adata.obsm['spatial']`` already in scanpy/squidpy
    convention ``(pxl_col_in_fullres, pxl_row_in_fullres) = (x, y)`` —
    no swap is needed (canonical convention pinned by
    :func:`spatialcellchat.viz_spatial._xy_from_adata`). Audit M-VIZ-6
    flagged the previous swap behaviour for misaligning plotly plots
    against ``viz_spatial`` matplotlib plots.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.shape[1] != 2:
        raise ValueError(
            "Please check the input 'coordinates' and make sure it is a two column matrix."
        )
    df = pd.DataFrame(
        {"x_cent": coords[:, 0], "y_cent": coords[:, 1]},
        index=list(adata.obs_names),
    )
    return df


def plotly_spatialDimPlot(
    adata: ad.AnnData,
    group_by: Optional[str] = None,
    group_highlight: Optional[Sequence[str]] = None,
    cells_highlight: Optional[Sequence] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    idents_use: Optional[Sequence] = None,
    method: str = "2d",
    color_use: Optional[Sequence[str]] = None,
    nohighlight_alpha: float = 0.65,
    title_name: Optional[str] = None,
    scene_name: str = "scene",
    point_size: float = 2.0,
    show_legend: bool = True,
    legend_group_title: str = "labels",
    three_dim_z: float = 0.0,
) -> go.Figure:
    """Spatial dim-plot using Plotly.

    Direct port of R's :func:`plotly_spatialDimPlot`
    (``visualization.R:7787``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``obsm['spatial']`` populated.
    group_by : str, optional
        Column in ``adata.obs`` to colour points by; falls back to the
        ident factor.
    group_highlight : sequence, optional
        Cell groups (categories) to highlight; non-highlighted spots are
        drawn at ``nohighlight_alpha`` opacity.
    cells_highlight : sequence, optional
        Cell IDs (or 1-based indices) to highlight.
    sources_use, targets_use, idents_use : sequence, optional
        Filter spots before plotting.
    method : {"2d", "3d"}
    color_use : sequence of str, optional
    nohighlight_alpha : float
    title_name, scene_name, point_size, show_legend, legend_group_title,
    three_dim_z :
        See R reference.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if method not in ("2d", "3d"):
        raise ValueError(
            'The parameter "method" in this function only supports for "2d","3d"!'
        )

    coords = _get_coords_swapped(adata)
    if group_by is None:
        labels = pd.Series(_get_idents(adata, None), index=list(adata.obs_names))
    else:
        labels = adata.obs[group_by].astype("category")
        labels = pd.Series(labels.values, index=list(adata.obs_names))
    cells_level = list(labels.cat.categories) if isinstance(labels.dtype, pd.CategoricalDtype) else sorted(set(labels))
    coords["spot_labels"] = labels.values
    coords["cell_id"] = list(adata.obs_names)

    if idents_use is not None:
        idents_use_resolved = [
            cells_level[int(x) - 1] if isinstance(x, (int, np.integer)) else str(x)
            for x in idents_use
        ]
        group = pd.Series(["Others"] * len(labels), index=labels.index)
        mask = labels.isin(idents_use_resolved)
        group[mask] = labels[mask].astype(str)
        group = pd.Categorical(group, categories=list(idents_use_resolved) + ["Others"])
        if color_use is None:
            cu_all = list(scPalette(len(cells_level)))
            color_use = [cu_all[cells_level.index(x)] for x in idents_use_resolved]
            color_use.append("#E5E5E5")  # grey90
        labels = pd.Series(group, index=coords.index)

    if sources_use is None and targets_use is None:
        if color_use is None:
            color_use = list(scPalette(len(set(labels))))
    else:
        sources_resolved = (
            [cells_level[int(x) - 1] if isinstance(x, (int, np.integer)) else str(x) for x in sources_use]
            if sources_use is not None else []
        )
        targets_resolved = (
            [cells_level[int(x) - 1] if isinstance(x, (int, np.integer)) else str(x) for x in targets_use]
            if targets_use is not None else []
        )
        group = pd.Series(["Others"] * len(labels), index=labels.index)
        mask_s = labels.isin(sources_resolved)
        mask_t = labels.isin(targets_resolved)
        group[mask_s] = labels[mask_s].astype(str)
        group[mask_t] = labels[mask_t].astype(str)
        cats = list(dict.fromkeys(sources_resolved + targets_resolved + ["Others"]))
        group = pd.Categorical(group, categories=cats)
        if color_use is None:
            cu_all = list(scPalette(len(cells_level)))
            color_use = [cu_all[cells_level.index(x)] for x in sources_resolved + targets_resolved]
            color_use.append("#E5E5E5")
        labels = pd.Series(group, index=coords.index)

    coords["color_labels"] = labels.values

    # Highlighting
    if group_highlight is not None and cells_highlight is not None:
        raise ValueError("Please don't input `group.highlight` or `cells.highlight` simultaneously.")

    if group_highlight is not None:
        labels_set = set(map(str, group_highlight))
        coords["spot_highlight"] = ~coords["spot_labels"].astype(str).isin(labels_set)
    elif cells_highlight is not None:
        cells_h = list(cells_highlight)
        if all(isinstance(c, (int, np.integer)) for c in cells_h):
            mask = np.zeros(len(coords), dtype=bool)
            for c in cells_h:
                if 1 <= int(c) <= len(coords):
                    mask[int(c) - 1] = True
            coords["spot_highlight"] = ~mask
        else:
            cells_set = set(map(str, cells_h))
            coords["spot_highlight"] = ~coords["cell_id"].astype(str).isin(cells_set)
    else:
        coords["spot_highlight"] = False

    # Resolve color_use into a per-category mapping
    label_categories = (
        list(coords["color_labels"].cat.categories)
        if isinstance(coords["color_labels"].dtype, pd.CategoricalDtype)
        else list(pd.unique(coords["color_labels"].astype(str)))
    )
    if color_use is None:
        color_use = list(scPalette(len(label_categories)))
    cu_list = list(color_use)
    color_map: Dict[str, str] = {}
    for i, c in enumerate(label_categories):
        color_map[str(c)] = cu_list[i % len(cu_list)]

    fig = go.Figure()

    def _hover_text(df: pd.DataFrame) -> List[str]:
        return [
            f"<i><b>cell type: {row['spot_labels']}</b></i><br> x: {row['x_cent']}, y: {row['y_cent']}<br> cell id: {row['cell_id']}"
            for _, row in df.iterrows()
        ]

    df_main = coords[~coords["spot_highlight"]]
    df_dim = coords[coords["spot_highlight"]]

    for cat in label_categories:
        sub_main = df_main[df_main["color_labels"].astype(str) == str(cat)]
        if not sub_main.empty:
            if method == "3d":
                fig.add_trace(go.Scatter3d(
                    x=sub_main["x_cent"], y=sub_main["y_cent"], z=[three_dim_z] * len(sub_main),
                    mode="markers",
                    marker=dict(size=point_size, color=color_map.get(str(cat), "#888888"), symbol="circle"),
                    name=str(cat), legendgroup=legend_group_title,
                    hoverinfo="text", text=_hover_text(sub_main),
                    customdata=sub_main["cell_id"].tolist(),
                    opacity=1.0,
                    scene=scene_name,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=sub_main["x_cent"], y=sub_main["y_cent"],
                    mode="markers",
                    marker=dict(size=point_size + 4, color=color_map.get(str(cat), "#888888"), symbol="circle"),
                    name=str(cat), legendgroup=legend_group_title,
                    hoverinfo="text", text=_hover_text(sub_main),
                    customdata=sub_main["cell_id"].tolist(),
                    opacity=1.0,
                ))
        sub_dim = df_dim[df_dim["color_labels"].astype(str) == str(cat)]
        if not sub_dim.empty:
            if method == "3d":
                fig.add_trace(go.Scatter3d(
                    x=sub_dim["x_cent"], y=sub_dim["y_cent"], z=[three_dim_z] * len(sub_dim),
                    mode="markers",
                    marker=dict(size=point_size, color=color_map.get(str(cat), "#888888"), symbol="circle"),
                    name=str(cat), legendgroup=legend_group_title, showlegend=False,
                    hoverinfo="text", text=_hover_text(sub_dim),
                    customdata=sub_dim["cell_id"].tolist(),
                    opacity=float(nohighlight_alpha),
                    scene=scene_name,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=sub_dim["x_cent"], y=sub_dim["y_cent"],
                    mode="markers",
                    marker=dict(size=point_size + 4, color=color_map.get(str(cat), "#888888"), symbol="circle"),
                    name=str(cat), legendgroup=legend_group_title, showlegend=False,
                    hoverinfo="text", text=_hover_text(sub_dim),
                    customdata=sub_dim["cell_id"].tolist(),
                    opacity=float(nohighlight_alpha),
                ))

    if method == "2d":
        fig.update_layout(
            title=title_name,
            yaxis=dict(autorange="reversed", title="", showgrid=False, ticks="",
                       tickvals=[], scaleanchor="x", showline=False, visible=False),
            xaxis=dict(title="", showgrid=False, ticks="",
                       tickvals=[], showline=False, visible=False),
            showlegend=show_legend,
        )
    else:
        fig.update_layout(
            title=title_name, showlegend=show_legend,
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
        )
    return fig


def plotly_spatialLRpairPlot(
    adata: ad.AnnData,
    features: Optional[Sequence[str]] = None,
    pairLR_use: Optional[Union[str, Sequence[str], pd.DataFrame]] = None,
    cells_highlight: Optional[Sequence] = None,
    enriched_only: bool = False,
    thresh: float = 0.05,
    do_group: bool = False,
    colormap: str = "Spectral",
    reversescale: bool = True,
    point_size: float = 4.0,
    nohighlight_alpha: float = 0.3,
    do_plot: bool = True,
) -> Union[List[go.Figure], int]:
    """Spatial LR pair / feature plot via Plotly.

    Direct port of R's :func:`plotly_spatialLRpairPlot`
    (``visualization.R:6937``).

    Parameters
    ----------
    adata : anndata.AnnData
        With ``obsm['spatial']`` populated and gene expression in ``adata.X``.
    features : sequence of str, optional
        Gene symbols (≤ 2 allowed if non-None).
    pairLR_use : str / sequence of str / DataFrame, optional
        Single LR pair name (or DataFrame with column ``interaction_name``).
    cells_highlight, enriched_only, thresh, do_group, colormap, reversescale,
    point_size, nohighlight_alpha, do_plot :
        See R reference.

    Returns
    -------
    list of plotly.graph_objects.Figure or int
        If ``do_plot`` is True returns one Figure per feature. Otherwise
        returns the number of features.
    """
    coords = _get_coords_swapped(adata)
    cc = _ensure_uns(adata)

    if features is not None and pairLR_use is not None:
        raise ValueError("Please don't input features or pairLR.use simultaneously.")
    if features is not None and len(features) > 2:
        raise ValueError("Please don't input more than 2 features.")
    if pairLR_use is not None and not isinstance(pairLR_use, (pd.DataFrame, str)):
        if len(list(pairLR_use)) > 1:
            raise ValueError("Please don't input more than 1 LR pair.")

    if pairLR_use is not None:
        if isinstance(pairLR_use, str):
            pairLR_use_df = pd.DataFrame({"interaction_name": [pairLR_use]})
        elif isinstance(pairLR_use, (list, tuple)):
            pairLR_use_df = pd.DataFrame({"interaction_name": list(pairLR_use)})
        else:
            pairLR_use_df = pairLR_use.copy()

        if enriched_only:
            net_d = cc["net"]
            if do_group:
                prob = np.asarray(net_d["prob"], dtype=float).copy()
                pval = net_d.get("pval", None)
                if pval is not None:
                    prob[np.asarray(pval, dtype=float) > thresh] = 0.0
                pair_axis = net_d.get("dimnames", {}).get("pairs", [])
                pair_names = list(pairLR_use_df["interaction_name"])
                pair_names_use = [p for p in pair_names if p in set(pair_axis)]
                if len(pair_names_use) == 0:
                    raise ValueError(
                        "There is no significant communication related with the input "
                        "`pairLR.use`. Set `enriched.only = FALSE` to show non-significant signaling."
                    )
                pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
                idx = [pair_idx_map[p] for p in pair_names_use]
                prob_sub = prob[:, :, idx]
                sums = (prob_sub > 0).sum(axis=(0, 1))
                signaling_includes = [pair_names_use[k] for k in np.flatnonzero(sums > 0)]
                pairLR_use_df = pairLR_use_df[pairLR_use_df["interaction_name"].isin(signaling_includes)]
            else:
                # prob.cell branch (use if available)
                if "prob.cell" not in net_d:
                    raise ValueError(
                        "enriched_only=True requires net['prob.cell'] when do_group=False"
                    )
                prob_cell = net_d["prob.cell"]  # sparse 3-D
                pair_axis = net_d.get("dimnames", {}).get("pairs", [])
                pair_names = list(pairLR_use_df["interaction_name"])
                pair_names_use = [p for p in pair_names if p in set(pair_axis)]
                if len(pair_names_use) == 0:
                    raise ValueError(
                        "There is no significant communication related with the input "
                        "`pairLR.use`. Set `enriched.only = FALSE` to show non-significant signaling."
                    )
                pair_idx_map = {p: i for i, p in enumerate(pair_axis)}
                idx = [pair_idx_map[p] for p in pair_names_use]
                # count non-zero per layer
                signaling_includes: List[str] = []
                for k_axis, k_layer in zip(pair_names_use, idx):
                    layer = prob_cell[:, :, k_layer]
                    nnz = int((layer > 0).sum()) if hasattr(layer, "sum") else 0
                    if nnz > 0:
                        signaling_includes.append(k_axis)
                pairLR_use_df = pairLR_use_df[pairLR_use_df["interaction_name"].isin(signaling_includes)]
            if len(pairLR_use_df) == 0:
                raise ValueError(
                    "There is no significant communication related with the input "
                    "`pairLR.use`. Set `enriched.only = FALSE` to show non-significant signaling."
                )

        LRsig = cc["LR"]["LRsig"]
        sub = LRsig.loc[pairLR_use_df["interaction_name"], ["ligand", "receptor"]]
        gene_L = list(dict.fromkeys(sub["ligand"].tolist()))
        gene_R = list(dict.fromkeys(sub["receptor"].tolist()))
        feature_use = list(gene_L) + list(gene_R)
    else:
        feature_use = list(features) if features is not None else []

    # data matrix: prefer adata.X if features in var_names; else fall back to obs columns
    var_names = list(adata.var_names)
    obs_cols = list(adata.obs.columns)
    feat_in_var = [f for f in feature_use if f in set(var_names)]
    feat_in_obs = [f for f in feature_use if f in set(obs_cols)]
    if len(feat_in_var) > 0:
        feature_use = feat_in_var
        if sparse.issparse(adata.X):
            sub_X = adata[:, feature_use].X.toarray()
        else:
            sub_X = np.asarray(adata[:, feature_use].X)
        data_use = sub_X.T  # features x cells
    elif len(feat_in_obs) > 0:
        feature_use = feat_in_obs
        data_use = adata.obs[feature_use].T.to_numpy(dtype=float)
    else:
        raise ValueError("Please check your input!")

    num_feature = len(feature_use)
    if not do_plot:
        return num_feature

    figs: List[go.Figure] = []
    for i in range(num_feature):
        df_score = coords.copy()
        df_score["feature.data"] = data_use[i, :]
        figs.append(_plotly_feature(
            df_score=df_score,
            cells_highlight=cells_highlight,
            method="2d",
            plot_title=feature_use[i],
            point_size=point_size,
            nohighlight_alpha=nohighlight_alpha,
            color_heatmap=colormap,
            direction=-1 if reversescale else 1,
        ))
    return figs


def _plotly_feature(
    df_score: pd.DataFrame,
    cells_highlight: Optional[Sequence],
    method: str,
    plot_title: str,
    point_size: float,
    nohighlight_alpha: float,
    color_heatmap: str,
    direction: int,
) -> go.Figure:
    """Internal: render a single feature plot via plotly.

    A trimmed Python equivalent of R's :func:`plotFeatures`
    (``visualization.R:7105``).
    """
    df = df_score.copy()
    if "feature.data" in df.columns:
        df["score"] = df["feature.data"]
    elif "score" not in df.columns:
        # fall through — assume the third column is the score
        df.columns = list(df.columns[:-1]) + ["score"]

    df["cell_id"] = df.index.astype(str)

    if cells_highlight is not None:
        cells_h = list(cells_highlight)
        if all(isinstance(c, (int, np.integer)) for c in cells_h):
            mask = np.zeros(len(df), dtype=bool)
            for c in cells_h:
                if 1 <= int(c) <= len(df):
                    mask[int(c) - 1] = True
        else:
            cells_set = set(map(str, cells_h))
            mask = df["cell_id"].isin(cells_set).to_numpy()
        df["spot_highlight"] = mask

    fig = go.Figure()

    # Choose colorscale
    cs_name = str(color_heatmap)
    # Plotly accepts a name for built-in scales; reverse if direction == -1
    cs = cs_name + ("_r" if direction == -1 and not cs_name.endswith("_r") else "")

    if "spot_highlight" not in df.columns:
        fig.add_trace(go.Scatter(
            x=df["x_cent"], y=df["y_cent"], mode="markers",
            marker=dict(size=point_size, color=df["score"], colorscale=cs, showscale=True,
                        colorbar=dict(title="")),
            text=df["cell_id"], hoverinfo="text+x+y",
            opacity=1.0, name="all",
        ))
    else:
        df_keep = df[df["spot_highlight"]]
        df_dim = df[~df["spot_highlight"]]
        if len(df_keep) > 0:
            fig.add_trace(go.Scatter(
                x=df_keep["x_cent"], y=df_keep["y_cent"], mode="markers",
                marker=dict(size=point_size, color=df_keep["score"], colorscale=cs,
                            showscale=True, colorbar=dict(title="")),
                text=df_keep["cell_id"], hoverinfo="text+x+y",
                opacity=1.0, name="highlight",
            ))
        if len(df_dim) > 0:
            fig.add_trace(go.Scatter(
                x=df_dim["x_cent"], y=df_dim["y_cent"], mode="markers",
                marker=dict(size=point_size, color=df_dim["score"], colorscale=cs,
                            showscale=False),
                text=df_dim["cell_id"], hoverinfo="text+x+y",
                opacity=float(nohighlight_alpha), name="other", showlegend=False,
            ))

    fig.update_layout(
        title=plot_title,
        yaxis=dict(autorange="reversed", title="", showgrid=False, ticks="",
                   tickvals=[], scaleanchor="x", showline=False, visible=False),
        xaxis=dict(title="", showgrid=False, ticks="", tickvals=[], showline=False, visible=False),
        showlegend=False,
    )
    return fig
