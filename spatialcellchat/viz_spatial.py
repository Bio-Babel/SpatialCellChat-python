"""Spatial visualization (matplotlib backend).

Direct ports of the spatial-plot exports in ``visualization.R`` (sub-agent
6c scope, slice 6c). All public functions take ``adata`` as the first
argument and return a :class:`matplotlib.figure.Figure`.

Public exports
--------------
Spatial-on-image scatter:

- :func:`spatialDimPlot`             (R ``visualization.R:114``)
- :func:`spatialFeaturePlot`         (R ``visualization.R:252``)
- :func:`spatialLeePlot`             (R ``visualization.R:8118``)
- :func:`spatialGiPlot`              (R ``visualization.R:8582``)
- :func:`spatialCCCDistPlot`         (R ``visualization.R:21``)
- :func:`spatialVisual_scoring`      (R ``visualization.R:539``)
- :func:`spatialTopicPlot`           (R ``visualization.R:6771``)

Communication-field arrows:

- :func:`netVisual_CommunField`      (R ``visualization.R:5292``)
- :func:`netVisual_CommunFieldGrid`  (R ``visualization.R:5584``)
- :func:`netVisual_CommunFieldZoomIn`(R ``visualization.R:6108``)
- :func:`netVisual_CommunFlow`       (R ``visualization.R:5855``)

Pie-on-spatial:

- :func:`visualizePie`               (R ``visualization.R:6677``)

Per the porting contract, the AnnData layout follows the convention from
slices 0-5: ``obsm["spatial"]`` holds full-resolution pixel coordinates;
``uns["spatial"][lib_id]`` (scanpy/squidpy schema) holds the tissue image
and scale factors; ``uns["cellchat"]`` mirrors R's ``object@`` slots
(``net``, ``netP``, ``DB``, ``LR``, ``meta`` and friends).
"""

from __future__ import annotations

import matplotlib

from typing import Any, Mapping, Optional, Sequence, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from ._centrality import centr_signaling_names, centr_value
from ._palette import scPalette, ggPalette
from ._plot_helpers import _scatterpie_layer, _spatial_layer
from ._themes import (
    _brewer_pal,
    _BREWER_SUBSET,
    generate_custom_colorscale,
    generate_RedsBlues_colormap,
)


__all__ = [
    "spatialDimPlot",
    "spatialFeaturePlot",
    "spatialLeePlot",
    "spatialGiPlot",
    "spatialCCCDistPlot",
    "spatialVisual_scoring",
    "spatialTopicPlot",
    "netVisual_CommunField",
    "netVisual_CommunFieldGrid",
    "netVisual_CommunFieldZoomIn",
    "netVisual_CommunFlow",
    "visualizePie",
]


# ---------------------------------------------------------------------------
# Internal — colormap construction (mirrors R's brewer.pal/viridis cascade)
# ---------------------------------------------------------------------------
def _build_colormap(
    color_heatmap: Union[str, Sequence[str], None],
    n_colors: int,
    direction: int,
    slice_first: bool = False,
    replace_grey: bool = True,
) -> list:
    """Build the 99-step colour ramp used throughout R's spatial plots.

    Mirrors the inline colormap construction in
    ``spatialFeaturePlot`` / ``spatialTopicPlot`` / ``spatialVisual_scoring``:
    pull ``n_colors`` colours from RColorBrewer (or fall back to viridis
    with direction = -1), reverse if ``direction == -1``, expand to 99
    steps via R's :func:`grDevices::colorRampPalette`, and (for the spatial
    feature plot family) replace index 0 with ``"#E5E5E5"``.

    Parameters
    ----------
    color_heatmap : str, sequence of str, or None
    n_colors : int
    direction : int
    slice_first : bool, default ``False``
        If ``True``, drop the first brewer anchor (``colormap[2:length(...)]``
        in R's ``spatialVisual_scoring`` merge branch).
    replace_grey : bool, default ``True``
        If ``True``, swap index 0 of the ramp for ``"#E5E5E5"``.  R's
        ``spatialVisual_scoring`` merge branch deliberately does NOT do
        this swap (the per-channel ramps stay anchored on the brewer
        colour after slicing).
    """
    if color_heatmap is None:
        # spatialFeaturePlot uses a hand-tuned 6-anchor ramp when None.
        anchors = ["#FFFFEF", "#FFFF3E", "#FF9F00", "#FF1A00", "#930000", "#050000"]
        from ._palette import _r_color_ramp_palette

        out = list(_r_color_ramp_palette(anchors, 64))
        if replace_grey:
            out[0] = "#E5E5E5"
        return out

    if isinstance(color_heatmap, str):
        if color_heatmap in _BREWER_SUBSET:
            base = list(_brewer_pal(color_heatmap, n_colors))
        else:
            cmap = matplotlib.colormaps[color_heatmap].resampled(n_colors)
            samples = np.linspace(1.0, 0.0, n_colors)  # direction = -1
            cols = cmap(samples)[:, :3]
            base = [
                "#%02X%02X%02X"
                % tuple((np.clip(c, 0.0, 1.0) * 255).astype(int))
                for c in cols
            ]
        if direction == -1:
            base = list(reversed(base))
        if slice_first and len(base) > 1:
            base = base[1:]
        from ._palette import _r_color_ramp_palette

        out = list(_r_color_ramp_palette(base, 99))
        if replace_grey:
            out[0] = "#E5E5E5"
        return out

    return list(color_heatmap)


def _gradient_cmap(colors: Sequence[str]) -> LinearSegmentedColormap:
    """Build a matplotlib :class:`LinearSegmentedColormap` from a hex list."""
    return LinearSegmentedColormap.from_list("scc", list(colors), N=256)


# ---------------------------------------------------------------------------
# Internal — coordinate handling
# ---------------------------------------------------------------------------
def _xy_from_adata(adata: ad.AnnData) -> np.ndarray:
    """Pull the 2-D ``(x, y)`` plotting coordinates from ``adata.obsm['spatial']``.

    AnnData stores Visium coordinates in scanpy/squidpy convention:
    ``obsm['spatial'][:, 0] = pxl_col_in_fullres`` (x), ``[:, 1] = pxl_row_in_fullres`` (y).
    No transformation is needed for plotting on top of the tissue image at
    its native pixel extent — the same contract consumed by ``sc.pl.spatial``
    and ``sq.pl.spatial_scatter``. R's ``visualization.R`` swaps two columns
    before plotting because it ingests ``(imagerow, imagecol)`` from
    Seurat's ``GetTissueCoordinates``; that adapter is unnecessary here.
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "adata.obsm['spatial'] must be a 2-column (x, y) matrix."
        )
    return coords.copy()


def _idents_from_adata(adata: ad.AnnData, group_by: Optional[str]) -> pd.Categorical:
    """Resolve cell-group labels from ``obs`` (mirrors ``object@idents``)."""
    if group_by is None:
        if "active_ident" in adata.obs.columns:
            labels = adata.obs["active_ident"]
        else:
            raise ValueError(
                "adata.obs has no 'active_ident' column; please pass group_by."
            )
    else:
        labels = adata.obs[group_by]
    if not isinstance(labels.dtype, pd.CategoricalDtype):
        labels = pd.Categorical(labels)
    else:
        labels = labels.values
    return labels


def _ensure_cellchat(adata: ad.AnnData) -> Mapping:
    """Return ``adata.uns['cellchat']`` (raise if missing)."""
    if "cellchat" not in adata.uns:
        raise ValueError(
            "adata.uns['cellchat'] is missing — run the upstream CellChat "
            "pipeline first."
        )
    return adata.uns["cellchat"]


# ---------------------------------------------------------------------------
# spatialCCCDistPlot (R: visualization.R:21)
# ---------------------------------------------------------------------------
def spatialCCCDistPlot(
    adata: ad.AnnData,
    signaling_type: str = "both",
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    tol: Optional[float] = None,
    density_alpha: float = 0.5,
) -> Figure:
    """Density of cell-cell distances used as a communication constraint.

    Mirrors :func:`spatialCCCDistPlot` (``visualization.R:21``). When the
    upstream :func:`computeCellDistance` result is cached in
    ``adata.uns['cellchat']['spatial']['result.computeCellDistance']`` we
    read it; otherwise we recompute on the fly.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with spatial coordinates in ``obsm['spatial']`` and (optionally)
        a cached ``computeCellDistance`` result under ``uns['cellchat']``.
    signaling_type : {"both", "Secreted", "Contact"}, default ``"both"``
        Which density to draw.
    interaction_range : float, default ``250``
        Maximum interaction distance in microns.
    contact_range : float, default ``10``
        Maximum contact distance in microns.
    tol : float or None, default ``None``
        Distance tolerance; defaults to ``contact_range / 2``.
    density_alpha : float, default ``0.5``
        Fill opacity for density polygons.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one (single mode) or two (``"both"``) axes.
    """
    if signaling_type not in ("both", "Secreted", "Contact"):
        raise ValueError("signaling_type must be one of 'both','Secreted','Contact'.")

    cc = adata.uns.get("cellchat", {}) if "cellchat" in adata.uns else {}
    cached = None
    spatial_block = cc.get("spatial", {}) if isinstance(cc, Mapping) else {}
    if "result.computeCellDistance" in spatial_block:
        cached = spatial_block["result.computeCellDistance"]
    if cached is None:
        from .spatial import computeCellDistance

        # ratio: prefer cached spatial.factors then default 1.0.
        spatial_factors = spatial_block.get("spatial.factors", {})
        ratio = float(spatial_factors.get("ratio", 1.0))
        if tol is None:
            tol = float(spatial_factors.get("tol", contact_range / 2.0))
        cached = computeCellDistance(
            coordinates=adata,
            interaction_range=interaction_range,
            contact_range=contact_range,
            ratio=ratio,
            tol=tol,
        )

    d_spatial = cached["d.spatial"]
    if sparse.issparse(d_spatial):
        secreted = d_spatial.tocoo().data.astype(float)
    else:
        secreted = np.asarray(d_spatial)
        secreted = secreted[secreted != 0]

    adj_contact = cached["adj.contact"]
    if sparse.issparse(d_spatial) and sparse.issparse(adj_contact):
        contact_mat = d_spatial.multiply(adj_contact)
        contact_mat = contact_mat.tocoo()
        contact = contact_mat.data.astype(float)
    else:
        d_dense = np.asarray(d_spatial)
        a_dense = np.asarray(adj_contact)
        prod = d_dense * a_dense
        contact = prod[prod != 0].astype(float)

    n_panels = 2 if signaling_type == "both" else 1
    fig, axes = plt.subplots(nrows=n_panels, ncols=1, figsize=(6, 3 * n_panels))
    if n_panels == 1:
        axes = [axes]
    panels: list = []
    if signaling_type in ("both", "Secreted"):
        panels.append(("Secreted", secreted, "#69b3a2", "interaction range (um)"))
    if signaling_type in ("both", "Contact"):
        panels.append(("Contact", contact, "#404080", "contact range (um)"))

    for ax, (name, vec, color, xlab) in zip(axes, panels):
        if vec.size > 1 and np.std(vec) > 0:
            kde = gaussian_kde(vec)
            xs = np.linspace(0.0, interaction_range + 10.0, 200)
            ys = kde(xs)
            ax.fill_between(xs, ys, color=color, alpha=density_alpha)
            ax.plot(xs, ys, color=color)
        elif vec.size > 0:
            # Degenerate: show a single bar at the unique value.
            ax.axvline(float(vec[0]), color=color, alpha=density_alpha)
        ax.set_xlim(0.0, interaction_range + 10.0)
        ax.set_xticks(np.arange(0, interaction_range + 11, 20))
        ax.set_yticks([])
        ax.set_xlabel(xlab)
        ax.set_ylabel("")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# spatialDimPlot (R: visualization.R:114)
# ---------------------------------------------------------------------------
def spatialDimPlot(
    adata: ad.AnnData,
    color_use: Optional[Sequence[str]] = None,
    group_by: Optional[str] = None,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    idents_use: Optional[Sequence] = None,
    proportion: Optional[pd.DataFrame] = None,
    radius: Optional[float] = None,
    radius_size: float = 0.55,
    alpha: float = 1.0,
    shape_by: Union[int, str] = "o",
    title_name: Optional[str] = None,
    point_size: float = 2.4,
    legend_size: float = 3.0,
    legend_text_size: int = 8,
    legend_position: str = "right",
    legend_spacing: Sequence[float] = (0.0, -8.0),
    ncol: int = 1,
    byrow: bool = False,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Spatial scatter plot of cell groups (with optional cell-type pies).

    Mirrors :func:`spatialDimPlot` (``visualization.R:114``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``obsm['spatial']``, an ``obs`` group column and
        (optionally) the scanpy/squidpy ``uns['spatial']`` block.
    color_use : sequence of str or None
        Colour per cell group. If ``None`` we fall back to :func:`scPalette`.
    group_by : str or None
        Column name in ``adata.obs`` with cell-group labels. Defaults to
        ``"active_ident"``.
    sources_use, targets_use, idents_use : sequences or None
        Subsetting / colouring controls (mirror R argument names).
    proportion : pandas.DataFrame or None
        Per-spot cell-type proportions (rows match ``adata.obs_names``,
        columns are the cell types). When supplied, draws scatter pies via
        :func:`_scatterpie_layer`.
    radius, radius_size : float or None / float
        Per-pie radius. ``None`` triggers R's heuristic computation.
    alpha : float
        Per-spot opacity.
    shape_by : str or int
        Marker style.
    title_name : str or None
    point_size : float
    legend_size, legend_text_size, legend_position : float / int / str
    legend_spacing : sequence of two floats
        ``(x, y)`` legend key spacings.
    ncol, byrow : int / bool
    lib_id, image_key : str
        Pulled by :func:`_spatial_layer` for the background image.

    Returns
    -------
    matplotlib.figure.Figure
    """
    coords = _xy_from_adata(adata)
    labels = _idents_from_adata(adata, group_by)
    cells_level = list(labels.categories)

    if idents_use is not None:
        if all(isinstance(x, (int, np.integer)) for x in idents_use):
            idents_use = [cells_level[i - 1] for i in idents_use]
        idents_use = list(idents_use)
        group = np.array(["Others"] * len(labels), dtype=object)
        in_use = np.isin(labels.astype(str), idents_use)
        group[in_use] = labels[in_use].astype(str)
        group = pd.Categorical(group, categories=list(idents_use) + ["Others"])
        if color_use is None:
            color_all = scPalette(len(cells_level))
            color_use = [
                color_all[cells_level.index(x)] for x in idents_use
            ] + ["#E5E5E5"]
        labels = group
        cells_level = list(labels.categories)

    if sources_use is not None or targets_use is not None:
        if sources_use is not None and all(
            isinstance(x, (int, np.integer)) for x in sources_use
        ):
            sources_use = [cells_level[i - 1] for i in sources_use]
        if targets_use is not None and all(
            isinstance(x, (int, np.integer)) for x in targets_use
        ):
            targets_use = [cells_level[i - 1] for i in targets_use]
        sources_use = list(sources_use) if sources_use is not None else []
        targets_use = list(targets_use) if targets_use is not None else []

        group = np.array(["Others"] * len(labels), dtype=object)
        for label_set in (sources_use, targets_use):
            in_use = np.isin(labels.astype(str), label_set)
            group[in_use] = labels[in_use].astype(str)
        cats = list(sources_use) + list(targets_use) + ["Others"]
        group = pd.Categorical(group, categories=cats)
        if color_use is None:
            color_all = scPalette(len(cells_level))
            color_use = [
                color_all[cells_level.index(x)] for x in sources_use + targets_use
            ] + ["#E5E5E5"]
        labels = group
        cells_level = list(labels.categories)

    if color_use is None:
        color_use = list(scPalette(len(cells_level)))
    color_map = dict(zip(cells_level, list(color_use)))

    # Background image (optional)
    img, scalef, scaled = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    # When the scanpy uns block is present, swap-and-scale matches R's pipeline.
    if img is not None:
        # _xy_from_adata above operated on the un-scaled coords; rescale.
        coords = coords * scalef

    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(img, origin="upper")

    if proportion is None:
        for cat in cells_level:
            sel = (np.asarray(labels) == cat)
            if not np.any(sel):
                continue
            ax.scatter(
                coords[sel, 0],
                coords[sel, 1],
                s=(point_size ** 2) * 6,
                c=color_map[cat],
                alpha=alpha,
                marker=shape_by,
                edgecolors="none",
                label=str(cat),
            )
        leg = ax.legend(
            loc=legend_position,
            fontsize=legend_text_size,
            ncol=ncol,
            markerscale=max(legend_size / max(point_size, 1e-6), 0.1),
            handletextpad=0.4,
        )
        if leg is not None:
            leg.set_title("")
    else:
        prop = pd.DataFrame(proportion)
        ct_select = list(prop.columns)
        if radius is None:
            x_range = float(coords[:, 0].max() - coords[:, 0].min())
            y_range = float(coords[:, 1].max() - coords[:, 1].min())
            area = x_range * y_range
            r2 = area / float(prop.shape[0])
            r2 = r2 / np.pi
            radius = float(np.sqrt(r2) * radius_size)
        df = prop.copy()
        df["x_cent"] = coords[:, 0]
        df["y_cent"] = coords[:, 1]
        if color_use is None or len(color_use) < len(ct_select):
            colors_pie = list(scPalette(len(ct_select)))
        else:
            colors_pie = list(color_use[: len(ct_select)])
        _scatterpie_layer(
            df,
            x="x_cent",
            y="y_cent",
            r=radius,
            cols=ct_select,
            ax=ax,
            colors=colors_pie,
        )
        # Legend handles for pie colors
        for k, cat in enumerate(ct_select):
            ax.plot([], [], "s", color=colors_pie[k], label=str(cat))
        leg = ax.legend(
            loc=legend_position,
            fontsize=legend_text_size,
            ncol=ncol,
            handletextpad=0.4,
        )
        if leg is not None:
            leg.set_title("")

    ax.set_aspect("equal")
    if img is None:
        ax.invert_yaxis()  # R's `scale_y_reverse()`
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title_name is not None:
        ax.set_title(title_name, fontsize=10, loc="center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# spatialFeaturePlot (R: visualization.R:252)
# ---------------------------------------------------------------------------
def spatialFeaturePlot(
    adata: ad.AnnData,
    features: Optional[Sequence[str]] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR_use: Optional[Union[Sequence[str], pd.DataFrame]] = None,
    slot_name: str = "data.signaling",
    enriched_only: bool = True,
    thresh: float = 0.05,
    do_group: bool = True,
    color_heatmap: Optional[Union[str, Sequence[str]]] = None,
    n_colors: int = 8,
    direction: int = -1,
    do_binary: bool = False,
    cutoff: Optional[float] = None,
    color_use: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
    point_size: float = 1.0,
    legend_size: float = 3.0,
    legend_text_size: int = 8,
    shape_by: Union[int, str] = "o",
    ncol: Optional[int] = None,
    show_legend: bool = True,
    show_legend_combined: bool = False,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Spatial feature (gene/metadata) plot.

    Mirrors :func:`spatialFeaturePlot` (``visualization.R:252``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``obsm['spatial']`` and a counts/data matrix in ``X``.
    features : sequence of str, optional
        Gene names (rows of ``adata.var``) or metadata column names
        (``adata.obs``). Mutually exclusive with ``signaling`` /
        ``pairLR_use``.
    signaling : sequence of str, optional
        Pathway names to expand into ligand/receptor genes via
        :func:`extractEnrichedLR`.
    pairLR_use : sequence of str, pandas.DataFrame, optional
        Specific L-R pair names (or a DataFrame with an
        ``interaction_name`` column).
    slot_name : {"data.signaling","data"}, default ``"data.signaling"``
        Which expression slot to read from. ``data.signaling`` reads
        ``uns['cellchat']['data.signaling']``; ``data`` falls back to
        ``adata.X``.
    enriched_only : bool, default ``True``
        Restrict ``signaling`` / ``pairLR_use`` to enriched pairs.
    thresh : float, default ``0.05``
        p-value threshold (mirrors R).
    do_group : bool, default ``True``
        Group-level vs. cell-level enrichment.
    color_heatmap : str, sequence of str, or None
        RColorBrewer / viridis palette name or a vector of hex colours.
    n_colors, direction : int / int
        Anchor count and ``-1`` reverses the colour order.
    do_binary, cutoff : bool, float or None
        ``do_binary=True`` requires ``signaling`` or ``pairLR_use`` plus a
        ``cutoff`` value; renders 4-class (Ligand-only / Receptor-only /
        Both / None) per spot.  Otherwise ``cutoff`` masks values
        ``<= cutoff`` to NaN.
    color_use : sequence of str or None
    alpha, point_size : float
    legend_size, legend_text_size : float / int
    shape_by : str / int
    ncol : int or None
        Columns when plotting multiple features.
    show_legend, show_legend_combined : bool
    lib_id, image_key : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_inputs = sum(x is not None for x in (features, signaling, pairLR_use))
    if n_inputs == 0:
        raise ValueError("Please input either features, signaling or pairLR.use.")
    if n_inputs > 1:
        raise ValueError(
            "Please don't input features / signaling / pairLR.use simultaneously."
        )

    coords = _xy_from_adata(adata)
    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    var_names = list(adata.var_names)
    obs_cols = list(adata.obs.columns)

    # Resolve the expression matrix backing ``features`` lookup.  R: ``data
    # <- methods::slot(object, slot.name)``; Python prefers
    # ``uns['cellchat'][slot_name]`` then falls back to ``adata.X``.
    cc = adata.uns.get("cellchat", {}) if "cellchat" in adata.uns else {}
    data_block = None
    if slot_name in cc:
        data_block = cc[slot_name]
    if data_block is not None:
        # Expect (genes, cells) DataFrame; coerce.
        if isinstance(data_block, pd.DataFrame):
            data_full = data_block
        else:
            data_full = pd.DataFrame(np.asarray(data_block))
            data_full.index = pd.Index(var_names[: data_full.shape[0]])
    else:
        X = adata.X
        if sparse.issparse(X):
            data_full = pd.DataFrame(np.asarray(X.todense()).T,
                                     index=pd.Index(var_names))
        else:
            data_full = pd.DataFrame(np.asarray(X).T, index=pd.Index(var_names))

    # ---------- resolve feature names / expression matrix ----------
    if signaling is not None:
        from .analysis_topics import extractEnrichedLR

        if not do_binary:
            res = extractEnrichedLR(
                adata, signaling=list(signaling), geneLR_return=True,
                enriched_only=enriched_only, thresh=thresh, do_group=do_group,
            )
            feature_use = list(res["geneLR"])
        else:
            res = extractEnrichedLR(
                adata, signaling=list(signaling), geneLR_return=False,
                enriched_only=enriched_only, thresh=thresh, do_group=do_group,
            )
            pairLR_names = list(res["interaction_name"])
            LR_pair = cc["LR"]["LRsig"].loc[pairLR_names, ["ligand", "receptor"]]
    elif pairLR_use is not None:
        if isinstance(pairLR_use, str):
            pairLR_use = [pairLR_use]
        if not isinstance(pairLR_use, pd.DataFrame):
            pairLR_use = pd.DataFrame({"interaction_name": list(pairLR_use)})
        if enriched_only:
            net = cc.get("net", {})
            if do_group:
                prob_raw = net.get("prob")
                if prob_raw is None:
                    raise ValueError("net['prob'] required for enriched_only=True.")
                prob = np.asarray(prob_raw, dtype=float).copy()
                pval_raw = net.get("pval")
                if pval_raw is not None:
                    prob[np.asarray(pval_raw, dtype=float) > thresh] = 0.0
                pair_names_dim = list(net.get("dimnames", [None, None, []])[2])
            else:
                prob_raw = net.get("prob.cell")
                if prob_raw is None:
                    raise ValueError("net['prob.cell'] required for enriched_only=True.")
                prob = np.asarray(prob_raw, dtype=float)
                pair_names_dim = list(net.get("dimnames.cell", [None, None, []])[2])
            keep_names = [n for n in pairLR_use["interaction_name"] if n in pair_names_dim]
            keep_idx = [pair_names_dim.index(n) for n in keep_names]
            prob_sub = prob[:, :, keep_idx] if len(keep_idx) > 0 else prob[:, :, []]
            prob_sum = (prob_sub > 0).sum(axis=(0, 1))
            keep_names = [n for n, s in zip(keep_names, prob_sum) if s > 0]
            pairLR_use = pairLR_use[pairLR_use["interaction_name"].isin(keep_names)]
            if len(pairLR_use) == 0:
                raise ValueError(
                    "There is no significant communication related with the input "
                    "`pairLR.use`. Set `enriched.only=False` to show non-significant "
                    "signaling."
                )
        LR_pair = cc["LR"]["LRsig"].loc[
            pairLR_use["interaction_name"].astype(str).tolist(),
            ["ligand", "receptor"],
        ]
        if not do_binary:
            from .database import extractGeneSubset

            geneL = list(LR_pair["ligand"].unique())
            geneR = list(LR_pair["receptor"].unique())
            geneL = list(extractGeneSubset(geneL, cc["DB"]["complex"], cc["DB"]["geneInfo"]))
            geneR = list(extractGeneSubset(geneR, cc["DB"]["complex"], cc["DB"]["geneInfo"]))
            feature_use = list(geneL) + list(geneR)
    else:
        feature_use = list(features)

    if not do_binary:
        in_var = [f for f in feature_use if f in data_full.index]
        in_obs = [f for f in feature_use if f in obs_cols]
        if in_var:
            data_use = data_full.loc[in_var, :].to_numpy(dtype=float)
            feature_use = in_var
        elif in_obs:
            data_use = np.asarray(adata.obs[in_obs].T)
            feature_use = in_obs
        else:
            raise ValueError("Please check your input! ")

        if cutoff is not None:
            data_use = np.where(data_use <= cutoff, np.nan, data_use)

        n_feature = len(feature_use)
        if ncol is None:
            if n_feature > 9:
                ncol = 5
            else:
                ncol = min(n_feature, 5)
        nrow = int(np.ceil(n_feature / max(ncol, 1)))

        cmap_colors = _build_colormap(
            color_heatmap, n_colors=n_colors, direction=direction
        )
        cmap = _gradient_cmap(cmap_colors)

        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False,
        )
        for i in range(n_feature):
            r, c = divmod(i, ncol)
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img, origin="upper")
            vals = data_use[i, :].astype(float)
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                cmap=cmap,
                s=(point_size ** 2) * 6,
                alpha=alpha,
                marker=shape_by,
                edgecolors="none",
            )
            if show_legend or (show_legend_combined and i == n_feature - 1):
                cb = fig.colorbar(sc, ax=ax, fraction=0.05, shrink=0.4)
                cb.ax.tick_params(labelsize=legend_text_size)
            ax.set_aspect("equal")
            if img is None:
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(feature_use[i], fontsize=10, loc="center")
        for j in range(n_feature, nrow * ncol):
            r, c = divmod(j, ncol)
            axes[r, c].set_visible(False)
        fig.tight_layout()
        return fig

    # do_binary=True branch (R: visualization.R:395-507): four-class palette
    # of (Ligand only / Receptor only / Both / None).
    if cutoff is None:
        raise ValueError(
            "A `cutoff` must be provided when plotting expression in binary mode!"
        )
    if color_use is None:
        # ggPalette(4) then [4] <- "grey90" (R)
        from ._palette import ggPalette

        color_use = list(ggPalette(4))
        color_use[3] = "#E5E5E5"
    color_use1 = list(color_use)

    geneL_arr = LR_pair["ligand"].astype(str).tolist()
    geneR_arr = LR_pair["receptor"].astype(str).tolist()
    from ._expr import computeExpr_LR

    complex_input = cc["DB"]["complex"]
    dataL = computeExpr_LR(geneL_arr, data_full, complex_input)
    dataR = computeExpr_LR(geneR_arr, data_full, complex_input)

    feature_names = list(LR_pair.index)
    n_feature = len(feature_names)
    if ncol is None:
        ncol = 4 if n_feature > 9 else min(n_feature, 4)
    nrow = int(np.ceil(n_feature / max(ncol, 1)))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    for i in range(n_feature):
        r, c = divmod(i, ncol)
        ax = axes[r, c]
        if img is not None:
            ax.imshow(img, origin="upper")
        idx1 = dataL[i, :] > cutoff
        idx2 = dataR[i, :] > cutoff
        idx_both = idx1 & idx2
        # group: "None" / geneL / geneR / "Both"
        groups = np.full(dataL.shape[1], "None", dtype=object)
        groups[idx1] = geneL_arr[i]
        groups[idx2] = geneR_arr[i]
        groups[idx_both] = "Both"
        levels = [geneL_arr[i], geneR_arr[i], "Both", "None"]
        cmap_dict = dict(zip(levels, color_use1))
        for lvl in levels:
            sel = groups == lvl
            if not np.any(sel):
                continue
            ax.scatter(
                coords[sel, 0], coords[sel, 1],
                c=cmap_dict[lvl],
                s=(point_size ** 2) * 6,
                alpha=alpha,
                marker=shape_by,
                edgecolors="none",
                label=str(lvl),
            )
        ax.legend(fontsize=legend_text_size, markerscale=max(legend_size, 0.1))
        ax.set_aspect("equal")
        if img is None:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(feature_names[i], fontsize=10, loc="center")
    for j in range(n_feature, nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r, c].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# spatialVisual_scoring (R: visualization.R:539)
# ---------------------------------------------------------------------------
def spatialVisual_scoring(
    adata: ad.AnnData,
    signaling: str,
    method: str = "centrality",
    measure: Sequence[str] = ("outdeg", "indeg"),
    measure_name: Sequence[str] = ("outgoing", "incoming"),
    merge: bool = False,
    slot_name: str = "netP",
    do_group: bool = False,
    color_heatmap: Union[str, Sequence[str]] = "BuGn",
    n_colors: int = 9,
    direction: int = 1,
    alpha: float = 1.0,
    do_binary: bool = False,
    color_use: Optional[Sequence[str]] = None,
    point_size: float = 0.8,
    legend_size: float = 3.0,
    legend_text_size: int = 8,
    legend_spacing: Sequence[float] = (0.0, -8.0),
    shape_by: Union[int, str] = "o",
    ncol: Optional[int] = None,
    combined: bool = True,
    show_legend: bool = True,
    show_legend_combined: bool = False,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Spatial scatter plot of network centrality scores.

    Mirrors :func:`spatialVisual_scoring` (``visualization.R:539``). The
    centrality scores are pulled from
    ``adata.uns['cellchat'][slot_name]['centr.cell']`` (when
    ``do_group=False``) or ``[...][slot_name]['centr']`` (when
    ``do_group=True``).
    """
    if isinstance(signaling, (list, tuple, np.ndarray)) and len(signaling) > 1:
        raise ValueError("Only one signaling is allowed as an input.")
    measure = list(measure)
    measure_name = list(measure_name)
    for i, m in enumerate(measure):
        if m == "outdeg":
            measure_name[i] = "Outgoing"
        elif m == "indeg":
            measure_name[i] = "Incoming"

    coords = _xy_from_adata(adata)
    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    cc = _ensure_cellchat(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] is missing.")

    if method == "centrality":
        if do_group:
            centr_block = cc[slot_name].get("centr")
            if centr_block is None or len(centr_block) == 0:
                raise ValueError(
                    "Please run `netAnalysis_computeCentrality` to compute the network centrality scores!"
                )
            labels = _idents_from_adata(adata, None)
            cell_levels = list(labels.categories)
            value = np.full((len(measure), coords.shape[0]), np.nan)
            # Per-cell-group centrality: ``centr_value(...)`` returns a vector
            # of length ``n_cell_groups`` (because do_group=True processes
            # group-level networks); broadcast each group's value to its cells.
            for k, m in enumerate(measure):
                per_group = centr_value(centr_block, m, signaling)
                for i, lvl in enumerate(cell_levels):
                    mask = np.asarray(labels) == lvl
                    value[k, mask] = float(per_group[i])
        else:
            centr_cell = cc[slot_name].get("centr.cell")
            if centr_cell is None or len(centr_cell) == 0:
                raise ValueError(
                    "Please run `netAnalysis_computeCentrality` to compute the network centrality scores!"
                )
            value = np.stack(
                [centr_value(centr_cell, m, signaling) for m in measure],
                axis=0,
            )
    else:
        raise NotImplementedError("Only method='centrality' is supported.")

    n_feature = len(measure)
    if not merge:
        if not do_binary:
            if isinstance(color_heatmap, str):
                cmap_colors = _build_colormap(color_heatmap, n_colors, direction)
            else:
                cmap_colors = list(color_heatmap)
            cmap = _gradient_cmap(cmap_colors)
        else:
            if color_use is None:
                color_use = ["#2171b5"]
            color_use = ["#E5E5E5"] + list(color_use)
        if ncol is None:
            if n_feature > 9:
                ncol = 4
            else:
                ncol = min(n_feature, 2)
        nrow = int(np.ceil(n_feature / max(ncol, 1)))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
        for i in range(n_feature):
            r, c = divmod(i, ncol)
            ax = axes[r, c]
            if img is not None:
                ax.imshow(img, origin="upper")
            if not do_binary:
                sc = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=value[i, :], cmap=cmap, s=(point_size ** 2) * 6,
                    alpha=alpha, marker=shape_by, edgecolors="none",
                )
                if show_legend:
                    cb = fig.colorbar(sc, ax=ax, fraction=0.04, shrink=0.4)
                    cb.ax.tick_params(labelsize=legend_text_size)
            else:
                bin_vals = (value[i, :] > 0).astype(int)
                colors_arr = np.where(bin_vals == 0, color_use[0], color_use[1])
                ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=colors_arr, s=(point_size ** 2) * 6,
                    alpha=alpha, marker=shape_by, edgecolors="none",
                )
            ax.set_aspect("equal")
            if img is None:
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(
                f"{measure_name[i]} scores of {signaling} signaling",
                fontsize=10, loc="center",
            )
        for j in range(n_feature, nrow * ncol):
            r, c = divmod(j, ncol)
            axes[r, c].set_visible(False)
    else:
        if n_feature != 2:
            raise ValueError("`merge` only works when visualizing two measurements!")
        if not do_binary:
            if not isinstance(color_heatmap, (list, tuple)) or len(color_heatmap) != 3:
                color_heatmap = ("Blues", "Reds", "Purples")
            # R: ``colormap[2:length(colormap)]`` drops the first (lightest)
            # brewer anchor in the merge branch and DOES NOT swap index 0
            # with grey90 — see visualization.R:667-672.
            cmaps = [
                _gradient_cmap(_build_colormap(
                    c, n_colors, direction,
                    slice_first=True, replace_grey=False,
                ))
                for c in color_heatmap
            ]
        else:
            if color_use is None:
                color_use = ["#2171b5", "#cb181d", "#6a51a3"]
            color_use = ["#E5E5E5"] + list(color_use)
        fig, ax = plt.subplots(figsize=(5, 5))
        if img is not None:
            ax.imshow(img, origin="upper")
        v1 = value[0, :]
        v2 = value[1, :]
        m0 = (v1 == 0) & (v2 == 0)
        m1 = (v1 > 0) & (v2 == 0)
        m2 = (v1 == 0) & (v2 > 0)
        m3 = (v1 > 0) & (v2 > 0)
        if not do_binary:
            ax.scatter(coords[m0, 0], coords[m0, 1], c="#E5E5E5",
                       s=(point_size ** 2) * 6, alpha=alpha, marker=shape_by)
            ax.scatter(coords[m1, 0], coords[m1, 1], c=v1[m1], cmap=cmaps[0],
                       s=(point_size ** 2) * 6, alpha=alpha, marker=shape_by)
            ax.scatter(coords[m2, 0], coords[m2, 1], c=v2[m2], cmap=cmaps[1],
                       s=(point_size ** 2) * 6, alpha=alpha, marker=shape_by)
            v3 = (v1[m3] + v2[m3]) / 2.0
            ax.scatter(coords[m3, 0], coords[m3, 1], c=v3, cmap=cmaps[2],
                       s=(point_size ** 2) * 6, alpha=alpha, marker=shape_by)
        else:
            for mask, color in zip(
                (m0, m1, m2, m3), color_use,
            ):
                ax.scatter(coords[mask, 0], coords[mask, 1], c=color,
                           s=(point_size ** 2) * 6, alpha=alpha, marker=shape_by)
        ax.set_aspect("equal")
        if img is None:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"Merged scores of {signaling} signaling",
            fontsize=10, loc="center",
        )
        combined = False
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# spatialTopicPlot (R: visualization.R:6771)
# ---------------------------------------------------------------------------
def spatialTopicPlot(
    adata: ad.AnnData,
    topics: Optional[Sequence] = None,
    slot_name: str = "net",
    pattern: str = "incoming",
    topics_key: str = "Topic_",
    color_heatmap: Union[str, Sequence[str]] = "Reds",
    n_colors: int = 8,
    direction: int = 1,
    cutoff: Optional[float] = None,
    color_use: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
    point_size: float = 1.0,
    legend_size: float = 3.0,
    legend_text_size: int = 8,
    shape_by: Union[int, str] = "o",
    ncol: Optional[int] = None,
    combine: bool = True,
    show_legend: bool = True,
    show_legend_combined: bool = False,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Spatial scatter plot of cell-topic embeddings.

    Mirrors :func:`spatialTopicPlot` (``visualization.R:6771``). Reads the
    cell embeddings from
    ``adata.uns['cellchat'][slot_name]['topic'][pattern]['cell']`` (a
    ``(n_cells, K)`` matrix; we transpose to ``(K, n_cells)`` to match R).
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if ncol is None:
        ncol = 2
        if topics is not None and len(topics) == 1:
            ncol = 1
        if topics is not None and len(topics) > 6:
            ncol = 3
        if topics is not None and len(topics) > 9:
            ncol = 4

    coords = _xy_from_adata(adata)
    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    cc = _ensure_cellchat(adata)
    pattern_block = cc[slot_name]["topic"][pattern]
    cell = np.asarray(pattern_block["cell"], dtype=float)
    cell_emb = cell.T  # (K, n_cells) to match R
    K = cell_emb.shape[0]
    topic_names = [f"{topics_key}{i + 1}" for i in range(K)]

    if topics is None:
        feature_use = topic_names
    else:
        feature_use = []
        for t in topics:
            if isinstance(t, (int, np.integer)):
                feature_use.append(topic_names[int(t) - 1])
            else:
                feature_use.append(str(t))

    cmap_colors = _build_colormap(color_heatmap, n_colors=n_colors, direction=direction)
    cmap = _gradient_cmap(cmap_colors)

    feature_idx = [topic_names.index(t) for t in feature_use]
    data_use = cell_emb[feature_idx, :].astype(float)
    if cutoff is not None:
        data_use = np.where(data_use <= cutoff, np.nan, data_use)

    n_feature = len(feature_use)
    nrow = int(np.ceil(n_feature / max(ncol, 1)))

    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    for i in range(n_feature):
        r, c = divmod(i, ncol)
        ax = axes[r, c]
        if img is not None:
            ax.imshow(img, origin="upper")
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=data_use[i, :], cmap=cmap, s=(point_size ** 2) * 6,
            alpha=alpha, marker=shape_by, edgecolors="none",
        )
        if show_legend:
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, shrink=0.4)
            cb.ax.tick_params(labelsize=legend_text_size)
        ax.set_aspect("equal")
        if img is None:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(feature_use[i], fontsize=10, loc="center")
    for j in range(n_feature, nrow * ncol):
        r, c = divmod(j, ncol)
        axes[r, c].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal — Lee statistic
# ---------------------------------------------------------------------------
def _lee_pair(label1: np.ndarray, label2: np.ndarray, W: sparse.csr_matrix) -> float:
    """Lee's L statistic for a pair of label vectors with weight matrix ``W``.

    Bit-exact mirror of ``spdep::lee`` (R 4.x):

        ``L = (n / S2) * sum_i (Wz1)_i (Wz2)_i / sqrt(SSz1 * SSz2)``

    where the normaliser is the sum of *squared row sums* (NOT the sum of
    all weights):

        ``S2 = sum_i ( sum_j W_ij ) ^ 2``

    For the W = I (``lw_type="Itself"``) case this reduces to the Pearson
    correlation; for any other W the previous ``sum_W`` denominator over-
    counts and produces values 1/(mean-row-sum) too large.
    """
    z1 = np.asarray(label1, dtype=float)
    z2 = np.asarray(label2, dtype=float)
    n = z1.shape[0]
    z1c = z1 - z1.mean()
    z2c = z2 - z2.mean()
    # spatially-lagged centred values (spdep::lag.listw)
    Wz1 = np.asarray(W @ z1c).ravel()
    Wz2 = np.asarray(W @ z2c).ravel()
    # spdep::lee normaliser: sum of squared row sums.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    S2 = float((row_sums ** 2).sum())
    if S2 == 0:
        return 0.0
    num = float(np.sum(Wz1 * Wz2))
    den = float(np.sqrt(np.sum(z1c ** 2) * np.sum(z2c ** 2)))
    if den == 0:
        return 0.0
    return (n / S2) * num / den


# ---------------------------------------------------------------------------
# spatialLeePlot (R: visualization.R:8118)
# ---------------------------------------------------------------------------
def spatialLeePlot(
    adata: ad.AnnData,
    group_by2: Union[str, Sequence],
    group_by1: Optional[Union[str, Sequence]] = None,
    x_levels: Optional[Sequence[str]] = None,
    y_levels: Optional[Sequence[str]] = None,
    lw_type: str = "Itself",
    lw: Optional[sparse.spmatrix] = None,
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    cutoff: bool = True,
    cutoff_lee: float = 0.0,
    type: str = "heatmap",
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "Reds",
    reverse_colorscale: bool = False,
    n_colors: int = 6,
    dot_size: Sequence[float] = (1.0, 6.0),
    angle_x: float = 45.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title_name: Optional[str] = None,
    legend_title: Optional[str] = None,
    font_size: int = 10,
    font_size_title: int = 10,
    x_lab_rot: float = 45.0,
    row_show: Optional[Sequence] = None,
    col_show: Optional[Sequence] = None,
) -> Figure:
    """Lee's spatial co-occurrence statistic — heatmap or dot plot.

    Mirrors :func:`spatialLeePlot` (``visualization.R:8118``). The Python
    port computes Lee's L (Lee 2001) on a one-hot encoding of the two
    label vectors, using the spatial weight matrix selected by ``lw_type``.

    Parameters
    ----------
    adata : anndata.AnnData
    group_by2 : str or sequence
        Column name in ``adata.obs`` (or vector of labels) → x axis.
    group_by1 : str, sequence or None
        Same for the y axis. Defaults to ``"active_ident"``.
    x_levels, y_levels : sequence or None
        Optional ordering.
    lw_type : {"Itself","contact.range","interaction.range"}
        Selects the spatial weight matrix when ``lw`` is not supplied.
    lw : scipy.sparse matrix or None
        Pre-computed spatial weight matrix.
    interaction_range, contact_range : float
        Passed to :func:`computeCellDistance` when needed.
    cutoff, cutoff_lee : bool / float
        ``cutoff=True`` masks ``L <= cutoff_lee`` entries to NaN.
    type : {"data","dot","heatmap"}
        Output mode. ``"data"`` returns the matrix wrapped as a 1-axis
        figure (with the values shown as a table-like text), ``"dot"``
        returns a dot plot, ``"heatmap"`` an annotated heatmap.
    color_use : sequence of str or None
    color_heatmap : str
    reverse_colorscale : bool
    n_colors : int
    dot_size : sequence of two floats
    angle_x, x_lab_rot : float
    xlabel, ylabel, title_name, legend_title : str or None
    font_size, font_size_title : int
    row_show, col_show : sequence or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    coords = _xy_from_adata(adata)

    # Detect matrix inputs (R: ``inherits(x, c("matrix","Matrix","dgCMatrix"))``)
    # Each "label" can therefore be (a) a column name in obs, (b) a categorical
    # vector, or (c) a numeric matrix/DataFrame (rows = cells, cols = features).
    def _resolve_label(arg, default_label_name="variable"):
        """Return ``(labels_or_None, mat_or_None, cats)``.

        - Categorical / vector inputs → ``(pd.Categorical, None, list_of_cats)``
        - Matrix inputs               → ``(None, np.ndarray, col_names)``
        """
        if arg is None:
            cat_labels = _idents_from_adata(adata, None)
            return cat_labels, None, list(cat_labels.categories)
        if isinstance(arg, str):
            ser = adata.obs[arg]
            if not isinstance(ser.dtype, pd.CategoricalDtype):
                cat_labels = pd.Categorical(ser)
            else:
                cat_labels = ser.values
            return cat_labels, None, list(cat_labels.categories)
        if isinstance(arg, pd.DataFrame):
            return None, np.asarray(arg.values, dtype=float), list(arg.columns)
        if sparse.issparse(arg):
            return None, np.asarray(arg.todense(), dtype=float), [
                f"col{i + 1}" for i in range(arg.shape[1])
            ]
        arr = np.asarray(arg)
        if arr.ndim == 2:
            cols = [f"col{i + 1}" for i in range(arr.shape[1])]
            return None, arr.astype(float), cols
        cat_labels = pd.Categorical(arg)
        return cat_labels, None, list(cat_labels.categories)

    y_labels, y_mat, y_cats_default = _resolve_label(group_by1)
    x_labels, x_mat, x_cats_default = _resolve_label(group_by2)

    # Build the spatial weight matrix.
    if lw is None:
        if lw_type == "Itself":
            W = sparse.eye(coords.shape[0], format="csr")
        else:
            from .spatial import computeCellDistance, createCellCellContactMatrixFrom_dspatial

            cc_dist = computeCellDistance(
                coordinates=coords,
                interaction_range=interaction_range,
                contact_range=contact_range,
                ratio=1.0,
                tol=contact_range / 2.0,
            )
            if lw_type == "contact.range":
                W = sparse.csr_matrix(cc_dist["adj.contact"]).astype(float)
            elif lw_type == "interaction.range":
                d_spatial = cc_dist["d.spatial"]
                # Mirror R: P.spatial = createPspatialFrom_dspatial(d.spatial,
                # distance.use=T). i.e. invert d off-diagonal, set diagonal
                # to max(1/d.spatial). Then ``Matrix::diag(P.spatial) <- 1``
                # (overwrites again to 1) before ``mat2listw(P, style="W")``
                # row-standardises the entire matrix (including the diagonal).
                d_arr = sparse.csr_matrix(d_spatial).astype(float)
                P = d_arr.copy()
                P.data = 1.0 / P.data
                # Set diagonal = 1 (matches R order: P after the override)
                P = P.tolil()
                for k_idx in range(P.shape[0]):
                    P[k_idx, k_idx] = 1.0
                P = P.tocsr()
                # mat2listw(., style="W") row-standardises the WHOLE matrix
                row_sums = np.asarray(P.sum(axis=1)).ravel()
                row_sums[row_sums == 0] = 1.0
                D = sparse.diags(1.0 / row_sums)
                W = (D @ P).tocsr()
            else:
                raise ValueError(
                    "lw_type must be one of 'Itself', 'contact.range', 'interaction.range'."
                )
    else:
        W = sparse.csr_matrix(lw)

    # Build the per-feature matrix (rows = cells, cols = features).
    # Matrix branch: use the provided values directly.
    # Categorical branch: model.matrix(~x-1) → one-hot encoding.
    def _onehot(labels: pd.Categorical) -> tuple:
        cats = list(labels.categories)
        codes = np.asarray(labels.codes if hasattr(labels, "codes") else
                           pd.Categorical(labels).codes)
        onehot = np.zeros((len(codes), len(cats)), dtype=float)
        for k in range(len(cats)):
            onehot[codes == k, k] = 1.0
        return onehot, cats

    if x_mat is not None:
        L1 = x_mat
        x_cats = x_cats_default
    else:
        L1, x_cats = _onehot(x_labels)
    if y_mat is not None:
        L2 = y_mat
        y_cats = y_cats_default
    else:
        L2, y_cats = _onehot(y_labels)

    n_x = L1.shape[1]
    n_y = L2.shape[1]
    mat = np.zeros((n_y, n_x), dtype=float)
    for i in range(n_y):
        for j in range(n_x):
            mat[i, j] = _lee_pair(L1[:, j], L2[:, i], W)
    mat_df = pd.DataFrame(mat, index=y_cats, columns=x_cats)

    if cutoff:
        mat_df[mat_df <= cutoff_lee] = np.nan

    # Mirror R: ``mat <- t(dfLee)``.  After the transpose:
    #   * rownames(mat) = ``group_by2`` levels  (visual y-axis)
    #   * colnames(mat) = ``group_by1`` levels  (visual x-axis)
    # R's ``x.levels`` reorders/filters ``colnames(mat)`` (group_by1) and
    # ``y.levels`` reorders/filters ``rownames(mat)`` (group_by2) — see
    # visualization.R:8434-8442.  The Python port previously had these
    # two filters swapped (against R), which silently produced an
    # empty heatmap whenever the caller correctly pointed
    # ``y_levels`` at group_by2 levels.
    mat_t = mat_df.T  # rows = group_by2 (= mat_df.columns); cols = group_by1

    if x_levels is not None:
        x_levels = [v for v in x_levels if v in mat_t.columns]
        mat_t = mat_t[x_levels]
    if y_levels is not None:
        y_levels = [v for v in y_levels if v in mat_t.index]
        mat_t = mat_t.loc[y_levels]

    if row_show is not None:
        mat_t = mat_t.loc[row_show]
    if col_show is not None:
        mat_t = mat_t[col_show]

    cmap_colors = _build_colormap(color_heatmap, n_colors=n_colors, direction=1)
    if reverse_colorscale:
        cmap_colors = list(reversed(cmap_colors))
    cmap = _gradient_cmap(cmap_colors)

    if type == "data":
        # Wrap the data as a one-axis textual figure for completeness.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.table(
            cellText=mat_t.round(3).values.astype(str),
            rowLabels=mat_t.index,
            colLabels=mat_t.columns,
            loc="center",
        )
        return fig

    if type == "dot":
        fig, ax = plt.subplots(figsize=(max(4, len(mat_t.index) * 0.7),
                                        max(4, len(mat_t.columns) * 0.5)))
        # build long form
        records = []
        for ix in mat_t.index:
            for jy in mat_t.columns:
                v = mat_t.loc[ix, jy]
                if not np.isnan(v):
                    records.append((ix, jy, v))
        if records:
            xs = [r[0] for r in records]
            ys = [r[1] for r in records]
            vs = np.array([r[2] for r in records], dtype=float)
            xpos = [list(mat_t.index).index(x) for x in xs]
            ypos = [list(mat_t.columns).index(y) for y in ys]
            sizes = np.interp(vs, (vs.min(), vs.max()), dot_size)
            sc = ax.scatter(xpos, ypos, s=(sizes ** 2) * 5, c=vs, cmap=cmap)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, shrink=0.6)
            cb.set_label("Cooccurrence")
        ax.set_xticks(range(len(mat_t.index)))
        ax.set_xticklabels(list(mat_t.index), rotation=x_lab_rot,
                           ha="right", fontsize=font_size - 2)
        ax.set_yticks(range(len(mat_t.columns)))
        ax.set_yticklabels(list(mat_t.columns), fontsize=font_size)
        ax.grid(True, color="grey", alpha=0.3, linewidth=0.1)
        if title_name is not None:
            ax.set_title(title_name, fontsize=font_size_title)
        fig.tight_layout()
        return fig

    if type == "heatmap":
        fig, ax = plt.subplots(figsize=(max(5, len(mat_t.columns) * 0.6),
                                        max(4, len(mat_t.index) * 0.5)))
        # mat_t: rows = x, cols = y. We want columns on x-axis like R.
        plot_mat = mat_t.values
        im = ax.imshow(plot_mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(mat_t.columns)))
        ax.set_xticklabels(list(mat_t.columns),
                           rotation=x_lab_rot, ha="right", fontsize=font_size)
        ax.set_yticks(range(len(mat_t.index)))
        ax.set_yticklabels(list(mat_t.index), fontsize=font_size)
        for i in range(plot_mat.shape[0]):
            for j in range(plot_mat.shape[1]):
                v = plot_mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="black", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, shrink=0.6)
        cb.set_label("Cooccurrence")
        title = (title_name or "") + "Spatial co-occurrence (Lee statistic)"
        ax.set_title(title, fontsize=font_size_title)
        fig.tight_layout()
        return fig

    raise ValueError("type must be one of 'data', 'dot', 'heatmap'.")


# ---------------------------------------------------------------------------
# Internal — Getis-Ord Gi
# ---------------------------------------------------------------------------
def _local_gi(
    score: np.ndarray, coords: np.ndarray, n: int, gstar: bool
) -> tuple:
    """Compute the Getis-Ord Gi (or Gi*) statistic and an analytical p-value.

    Mirrors ``spdep::knearneigh + spdep::localG`` (binary weights).

    Parameters
    ----------
    score : (N,) ndarray
        Numeric values per spot.
    coords : (N, 2) ndarray
        Spot coordinates.
    n : int
        Number of nearest neighbours.
    gstar : bool
        If True, include the spot itself in the neighbourhood (Gi*).

    Returns
    -------
    G : (N,) ndarray
        Local Gi statistic.
    pvals : (N,) ndarray
        Two-sided analytical p-values.
    """
    N = coords.shape[0]
    tree = cKDTree(coords)
    # k+1 because the first neighbour is the point itself
    kuse = n + 1 if not gstar else n + 1
    _, idx = tree.query(coords, k=kuse)
    # Build sparse W (binary)
    rows = np.repeat(np.arange(N), n if not gstar else n + 1)
    if not gstar:
        cols = idx[:, 1:].ravel()  # exclude self
    else:
        cols = idx.ravel()
    data = np.ones_like(rows, dtype=float)
    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

    x = score.astype(float)
    # x_bar excluding (or including) self
    if gstar:
        x_bar_i = np.full(N, x.mean())
        s_i_sq = ((x ** 2).sum() / N) - x_bar_i ** 2
    else:
        # mean of x excluding i (R: spdep::localG default)
        x_bar_i = (x.sum() - x) / max(N - 1, 1)
        s_i_sq = ((x ** 2).sum() - x ** 2) / max(N - 1, 1) - x_bar_i ** 2

    # G_i numerator: sum_j W_ij * x_j   (no self)
    Wx = np.asarray(W @ x).ravel()
    Wi = np.asarray(W.sum(axis=1)).ravel()

    num = Wx - Wi * x_bar_i
    if not gstar:
        denom = np.sqrt(s_i_sq * (((N - 1) * Wi - Wi ** 2) / max(N - 2, 1)))
    else:
        denom = np.sqrt(s_i_sq * ((N * Wi - Wi ** 2) / max(N - 1, 1)))
    denom[denom == 0] = np.nan
    G = num / denom
    # Two-sided p-value from standard normal
    from scipy.stats import norm

    pvals = 2.0 * (1.0 - norm.cdf(np.abs(G)))
    return G, pvals


# ---------------------------------------------------------------------------
# spatialGiPlot (R: visualization.R:8582)
# ---------------------------------------------------------------------------
def spatialGiPlot(
    adata: ad.AnnData,
    slot_name: str = "netP",
    signaling_name: Optional[str] = None,
    do_binary: bool = True,
    group_highlight: Optional[Sequence] = None,
    measure: Sequence[str] = ("indeg", "outdeg"),
    measure_name: Sequence[str] = ("incoming", "outgoing"),
    plot_title: Optional[str] = None,
    n: int = 12,
    method: str = "normal",
    gstar: bool = False,
    p_value: Optional[float] = None,
    point_size: float = 2.0,
    nohighlight_alpha: float = 0.65,
    show_colorscale: bool = True,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Union[Figure, pd.DataFrame]:
    """Hotspots (Getis-Ord Gi/Gi*) of in/out-degree centrality.

    Mirrors :func:`spatialGiPlot` (``visualization.R:8582``).  Supported
    ``method`` modes: ``'normal'``, ``'scoring'``, ``'computation'``,
    ``'density'``.

    Tier-2 deviation: R's ``'3d'`` and ``'contour'`` modes both rely on
    :mod:`plotly` (R ``visualization.R:8951-9148``); rather than ship a
    plotly dependency in this matplotlib port we raise
    :class:`NotImplementedError` for those two modes.  All other modes
    are fully supported.

    Returns
    -------
    matplotlib.figure.Figure or pandas.DataFrame
        ``method='computation'`` returns a long-form DataFrame of
        ``statistic_G`` and ``Pr(z != E(Gi))`` values *without* plotting.
        Other modes return a Figure.  In every mode, the per-spot Gi and
        Gi-p-value are persisted into ``adata.obs`` (mirrors R's
        ``object@meta`` mutation at ``visualization.R:8741-8744``).
    """
    # R reference: ``visualization.R:9109-9147`` (contour) and
    # ``visualization.R:8951-9099`` (3d). Both supported here:
    # ``contour`` via :func:`scipy.interpolate.griddata` + matplotlib
    # ``contourf``; ``3d`` returns a :class:`plotly.graph_objects.Figure`
    # because matplotlib's 3D scatter does not match R's interactive
    # plotly behaviour. ``computation`` is a non-plot data return.
    valid_methods = {"normal", "scoring", "computation", "density", "contour", "3d"}
    if method not in valid_methods:
        raise ValueError(
            "spatialGiPlot supports method ∈ "
            "{'normal','scoring','computation','density','contour','3d'} "
            "(got %r). Mirrors R `plotStatistics_Gi` in visualization.R." % method
        )
    measure = list(measure)
    measure_name = list(measure_name)
    for i, m in enumerate(measure):
        if m == "outdeg":
            measure_name[i] = "outgoing"
        elif m == "indeg":
            measure_name[i] = "incoming"

    coords = _xy_from_adata(adata)
    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    cc = _ensure_cellchat(adata)
    if slot_name not in cc or cc[slot_name].get("centr.cell") is None:
        raise ValueError(
            "Please run `netAnalysis_computeCentrality` to compute the network centrality scores!"
        )
    centr_cell = cc[slot_name]["centr.cell"]
    # Flat schema: {values, centr_names, node_names, signaling_names}.
    sig_names_all = centr_signaling_names(centr_cell)
    if signaling_name is None:
        signaling_keys = list(sig_names_all)
    elif isinstance(signaling_name, (int, np.integer)):
        signaling_keys = [sig_names_all[int(signaling_name) - 1]]
    else:
        signaling_keys = [signaling_name]

    # Sum scores across signalings (mirrors apply(centr,c(1,2),sum))
    n_cells = coords.shape[0]
    centr_sum = np.zeros((len(measure), n_cells), dtype=float)
    for s in signaling_keys:
        for k, m in enumerate(measure):
            centr_sum[k, :] += centr_value(centr_cell, m, s)

    if do_binary:
        centr_sum = (centr_sum > 0).astype(float)

    # Pre-compute Gi statistic + p-values for every measure (always
    # needed: even ``method='normal'`` persists them into ``adata.obs``).
    G_all = []
    pvals_all = []
    for i, m in enumerate(measure):
        score = centr_sum[i, :]
        G, pvals = _local_gi(score, coords, n=n, gstar=gstar)
        G_all.append(G)
        pvals_all.append(pvals)

    # ``method='computation'`` mirrors R's plotStatistics_Gi(method='computation')
    # → returns a long-form DataFrame and skips plotting entirely.  We
    # additionally persist the columns into ``adata.obs`` to mirror the
    # outer ``object@meta`` mutation in R (visualization.R:8741-8744).
    if method == "computation":
        df_records = []
        for i, m in enumerate(measure):
            feature_name = (
                plot_title if plot_title is not None
                else f"Hotspots of {signaling_name}'s {measure_name[i]} scoring"
            )
            for cell_idx in range(coords.shape[0]):
                df_records.append({
                    "x": coords[cell_idx, 0],
                    "y": coords[cell_idx, 1],
                    "score": centr_sum[i, cell_idx],
                    "statistic_G": G_all[i][cell_idx],
                    "Pr(z != E(Gi))": pvals_all[i][cell_idx],
                    "measure": m,
                    "feature_name": feature_name,
                })
            # Persist into adata.obs (R's make.names() prefix).
            base = (f"Getis.Ord.Gi.{measure_name[i]}.{signaling_name}").replace(" ", ".")
            adata.obs[base] = G_all[i]
            adata.obs[base + ".p"] = pvals_all[i]
        return pd.DataFrame.from_records(df_records)

    # Plotting modes (normal / scoring / density).  Persist obs columns
    # first — mirrors R's outer ``object@meta`` mutation at
    # visualization.R:8741-8744 (applies in *every* mode, including
    # 'scoring' which colours by raw score instead of Gi).
    for i, m in enumerate(measure):
        base = (f"Getis.Ord.Gi.{measure_name[i]}.{signaling_name}").replace(" ", ".")
        adata.obs[base] = G_all[i]
        adata.obs[base + ".p"] = pvals_all[i]

    # 3D mode (R: visualization.R:8951-9099) returns a plotly Figure rather
    # than a matplotlib Figure — branch out before building the mpl grid.
    if method == "3d":
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plot_ncol_3d = min(len(measure), 2)
        specs = [[{"type": "scene"}] * plot_ncol_3d]
        plotly_fig = make_subplots(
            rows=1, cols=plot_ncol_3d, specs=specs,
            subplot_titles=[
                plot_title or f"Hotspots of {signaling_name}'s {measure_name[i]} scoring"
                for i in range(len(measure))
            ],
        )
        for i, m in enumerate(measure):
            G = G_all[i]
            pvals = pvals_all[i]
            cmap_colors = list(generate_RedsBlues_colormap(G[~np.isnan(G)]))
            cmap_colors = [
                c[:7] if c is not None and len(c) >= 7 else "#FFFFFF"
                for c in cmap_colors
            ]
            # Significance shape: filled marker for p < threshold, open otherwise
            # — mirrors R's ``mutate(point_shape = if_else(p < p.value, I(19), I(1)))``.
            if p_value is None:
                marker_symbol = ["circle"] * len(coords)
            else:
                marker_symbol = [
                    "circle" if (not np.isnan(pv) and pv < p_value) else "circle-open"
                    for pv in pvals
                ]
            plotly_fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1],
                    z=np.full(coords.shape[0], 0.0),
                    mode="markers",
                    marker=dict(
                        size=point_size * 2,
                        color=G, colorscale=[[k / (len(cmap_colors) - 1), col]
                                              for k, col in enumerate(cmap_colors)],
                        symbol=marker_symbol,
                        showscale=show_colorscale,
                        colorbar=dict(title="Gi/Gi*", thickness=10) if show_colorscale and i == 0 else None,
                    ),
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )
        plotly_fig.update_layout(showlegend=False)
        return plotly_fig

    plot_ncol = min(len(measure), 2)
    nrow = int(np.ceil(len(measure) / plot_ncol))
    fig, axes = plt.subplots(nrow, plot_ncol, figsize=(5 * plot_ncol, 5 * nrow), squeeze=False)
    for i, m in enumerate(measure):
        r, c = divmod(i, plot_ncol)
        ax = axes[r, c]
        G = G_all[i]
        pvals = pvals_all[i]

        # Choose colormap + values per method.
        if method == "scoring":
            # R (visualization.R:8886-8908): colour by raw per-spot score
            # (centrality sum) using a reversed Spectral palette
            # ``generate_custom_colorscale("Spectral", n.colors=8,
            # reversescale=T, plotly.format=F)``.
            cmap_colors = list(
                generate_custom_colorscale(
                    "Spectral", n_colors=8, reversescale=True, plotly_format=False,
                )
            )
            color_label = "score"
            color_vals = centr_sum[i, :]
        else:
            cmap_colors = list(generate_RedsBlues_colormap(G[~np.isnan(G)]))
            # strip the "AA" alpha suffix if present
            cmap_colors = [
                c[:7] if c is not None and len(c) >= 7 else "#FFFFFF"
                for c in cmap_colors
            ]
            color_label = "Gi/Gi*"
            if method == "density":
                # R: ``calculate_density(statistic_G, df_score[c('x','y')], 'wkde')``
                from ._kde import weighted_kde_2d

                valid = ~np.isnan(G)
                xs = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
                ys = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
                # Use absolute Gi as KDE weight (positive weights only).
                w = np.abs(G[valid])
                if w.sum() == 0:
                    w = np.ones_like(w)
                dens_grid = weighted_kde_2d(coords[valid], w, grid=(xs, ys))
                # Re-sample density at each cell coordinate via nearest grid cell.
                ix = np.clip(np.searchsorted(xs, coords[:, 0]) - 1, 0, len(xs) - 1)
                iy = np.clip(np.searchsorted(ys, coords[:, 1]) - 1, 0, len(ys) - 1)
                color_vals = dens_grid[iy, ix]
            else:  # method == "normal" or "contour"
                color_vals = G

        cmap = _gradient_cmap(cmap_colors)

        if img is not None:
            ax.imshow(img, origin="upper")

        if method == "contour":
            # R parity (visualization.R:9109-9147): ``metR::geom_contour_fill``
            # with kriging interpolation. Python uses scipy ``griddata``
            # (cubic) for the gridded interpolation — visually equivalent
            # for a smooth Gi statistic field.
            from scipy.interpolate import griddata

            valid = ~np.isnan(G)
            if valid.sum() >= 4:
                gx_min, gx_max = coords[:, 0].min(), coords[:, 0].max()
                gy_min, gy_max = coords[:, 1].min(), coords[:, 1].max()
                xs = np.linspace(gx_min, gx_max, 120)
                ys = np.linspace(gy_min, gy_max, 120)
                XX, YY = np.meshgrid(xs, ys)
                ZZ = griddata(
                    coords[valid], G[valid], (XX, YY),
                    method="cubic", fill_value=np.nan,
                )
                cs = ax.contourf(XX, YY, ZZ, levels=10, cmap=cmap, alpha=0.85)
                if show_colorscale:
                    cb = fig.colorbar(cs, ax=ax, fraction=0.04, shrink=0.5)
                    cb.set_label(color_label)
            # Overlay points: R uses ``color = "grey10"`` (visualization.R:9126)
            # for the foreground markers. Filled circle for significant
            # spots (R: ``I(19)``), open circle otherwise (R: ``I(1)``).
            grey_fg = "0.1"  # matplotlib grey ↔ R `grey10`
            if p_value is None:
                ax.scatter(coords[:, 0], coords[:, 1],
                           s=(point_size ** 2) * 6,
                           c=grey_fg, marker="o", edgecolors="none")
            else:
                sig = (~np.isnan(pvals)) & (pvals < p_value)
                if sig.any():
                    ax.scatter(coords[sig, 0], coords[sig, 1],
                               s=(point_size ** 2) * 6,
                               c=grey_fg, marker="o", edgecolors="none")
                if (~sig).any():
                    ax.scatter(coords[~sig, 0], coords[~sig, 1],
                               s=(point_size ** 2) * 6,
                               facecolors="none", edgecolors=grey_fg,
                               marker="o")
        else:
            sc = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=color_vals, cmap=cmap, s=(point_size ** 2) * 6,
                marker="o", edgecolors="none",
            )
            if show_colorscale:
                cb = fig.colorbar(sc, ax=ax, fraction=0.04, shrink=0.5)
                cb.set_label(color_label)
        ax.set_aspect("equal")
        if img is None:
            ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if plot_title is None:
            title = f"Hotspots of {signaling_name}'s {measure_name[i]} scoring"
        else:
            title = plot_title
        ax.set_title(title, fontsize=10, loc="center")

    for j in range(len(measure), nrow * plot_ncol):
        r, c = divmod(j, plot_ncol)
        axes[r, c].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal — pull a 3-D field array from uns["cellchat"][slot_name]
# ---------------------------------------------------------------------------
def _resolve_field(
    adata: ad.AnnData, slot_name: str, pattern: str, signaling: str
) -> np.ndarray:
    """Return the ``(n_cells, 2)`` per-spot vector field for ``signaling``."""
    cc = _ensure_cellchat(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] is missing.")
    field_block = cc[slot_name].get("field")
    if field_block is None or pattern not in field_block:
        raise ValueError(
            "Please check the input 'signaling' and make sure it has been computed via `computeCommunField`."
        )
    field_pattern = field_block[pattern]
    # Two supported formats: dict {signaling: (n,2)} or 3-D array {signaling -> indexable}
    if isinstance(field_pattern, Mapping):
        if signaling not in field_pattern:
            raise ValueError(
                "Please check the input 'signaling' and make sure it has been computed via `computeCommunField`."
            )
        return np.asarray(field_pattern[signaling], dtype=float)
    arr = np.asarray(field_pattern)
    # Expect (n, 2, K) with names along last dim — fall back to assuming index 0.
    return arr[:, :, 0].astype(float)


def _swap_field(field_use: np.ndarray) -> np.ndarray:
    """Pass-through: align field vectors with the coordinate convention.

    R does ``temp_field.use = field.use; field.use[,1]=temp[,2];
    field.use[,2]=temp[,1]`` because R also swaps the spatial coordinates
    (``object@images$coordinates`` is ``(imagerow, imagecol)`` from Seurat
    and R re-labels it to ``(x_cent=col, y_cent=row)`` before plotting —
    so both coords AND fields swap, keeping (dx, dy) aligned with (x, y)).

    The Python port consumes ``adata.obsm['spatial']`` in scanpy/squidpy
    convention ``(pxl_col_in_fullres, pxl_row_in_fullres) = (x, y)`` —
    NO swap on coordinates (see :func:`_xy_from_adata`). The
    ``computeCommunField`` writer also returns vectors "in the same XY
    coordinate frame as ``obsm['spatial']``" (analysis_network.py:1436-1437).
    Therefore the field needs **no swap** either; doing so would
    misalign arrows with spots (audit M-VIZ-6). Kept as a typed
    pass-through so the call sites in ``netVisual_CommunField`` /
    ``netVisual_CommunFieldGrid`` / ``netVisual_CommunFlow`` stay
    self-documenting; the function's existence is a paper-trail of the
    R↔Python convention bridge.
    """
    return np.asarray(field_use, dtype=float)


# ---------------------------------------------------------------------------
# netVisual_CommunField (R: visualization.R:5292)
# ---------------------------------------------------------------------------
def netVisual_CommunField(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "incoming",
    zoom_to: int = 0,
    zoom_mode: bool = True,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
    color_use: Optional[Sequence[str]] = None,
    point_size: float = 2.0,
    min_mag: float = 0.0,
    alpha_image: float = 0.2,
    arrow_angle: float = 15.0,
    arrow_length: float = 0.6,
    arrow_size: float = 1.0,
    arrow_color: Optional[str] = None,
    legend_size: float = 0.5,
    legend_text_size: int = 8,
    plot_title_name: Optional[str] = None,
    plot_title_text_size: int = 8,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Per-spot communication-field arrows over a spatial scatter.

    Mirrors :func:`netVisual_CommunField` (``visualization.R:5292``).
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if zoom_to < 0:
        raise ValueError("Please check your `zoom.to`!")

    field_raw = _resolve_field(adata, slot_name, pattern, signaling)
    field_use = _swap_field(field_raw)
    coords = _xy_from_adata(adata)

    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    dx = field_use[:, 0]
    dy = field_use[:, 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)

    labels = _idents_from_adata(adata, None)
    cells_level = list(labels.categories)
    if color_use is None:
        color_use = list(scPalette(len(cells_level)))
    color_map = dict(zip(cells_level, color_use))

    # Zoom logic
    alpha_arr = np.full(coords.shape[0], alpha_image, dtype=float)
    if zoom_to >= 1:
        n_breaks = zoom_to + 1
        x_range = (coords[:, 0].min(), coords[:, 0].max())
        y_range = (coords[:, 1].min(), coords[:, 1].max())
        x_breaks = np.linspace(x_range[0], x_range[1], n_breaks + 1)
        y_breaks = np.linspace(y_range[0], y_range[1], n_breaks + 1)
        xb_idx = np.searchsorted(x_breaks, coords[:, 0], side="right") - 1
        yb_idx = np.searchsorted(y_breaks, coords[:, 1], side="right") - 1
        xb_idx = np.clip(xb_idx, 0, n_breaks - 1)
        yb_idx = np.clip(yb_idx, 0, n_breaks - 1)
        # R: ``findInterval(anchor.x, anchor.Interval)`` where
        # anchor.Interval are the *interior* breakpoints (the leading 0
        # and trailing 1 are dropped before findInterval).
        anchor_breaks = np.linspace(0.0, 1.0, n_breaks + 1)[1:-1]
        anchor_int_x = int(np.searchsorted(anchor_breaks, anchor_x, side="right"))
        anchor_int_y = int(np.searchsorted(anchor_breaks, anchor_y, side="right"))
        in_zoom = (xb_idx == anchor_int_x) & (yb_idx == anchor_int_y)
        if not zoom_mode:
            # Drop spots outside zoom
            keep = in_zoom
            coords = coords[keep]
            dx = dx[keep]
            dy = dy[keep]
            mag = mag[keep]
            labels = pd.Categorical(np.asarray(labels)[keep], categories=cells_level)
            alpha_arr = np.full(coords.shape[0], alpha_image, dtype=float)
        else:
            alpha_arr[in_zoom] = min(alpha_image + 0.1, 1.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(img, origin="upper")

    # Plot points
    for cat in cells_level:
        sel = (np.asarray(labels) == cat)
        if not np.any(sel):
            continue
        ax.scatter(
            coords[sel, 0], coords[sel, 1],
            s=(point_size ** 2) * 6, c=color_map[cat],
            alpha=float(alpha_arr[sel].mean()),
            edgecolors="none", label=str(cat),
        )

    # Plot arrows where mag > min_mag
    keep = mag > min_mag
    if np.any(keep):
        if arrow_color is None:
            arrow_colors = [color_map[c] for c in np.asarray(labels)[keep]]
        else:
            arrow_colors = arrow_color
        # R uses dy = -dy after swap because the y-axis is reversed.
        ax.quiver(
            coords[keep, 0], coords[keep, 1],
            dx[keep], -dy[keep],
            color=arrow_colors,
            angles="xy", scale_units="xy", scale=1.0 / max(arrow_size, 1e-6),
            width=0.003, headwidth=arrow_angle / 5.0,
            headlength=arrow_length * 5.0,
        )

    if zoom_mode and zoom_to > 0:
        for x_b in x_breaks[1:-1]:
            ax.axvline(x_b, color="gray", linestyle="--", linewidth=0.5)
        for y_b in y_breaks[1:-1]:
            ax.axhline(y_b, color="gray", linestyle="--", linewidth=0.5)

    ax.set_aspect("equal")
    if img is None:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_title = "Sources" if pattern == "outgoing" else "Targets"
    leg = ax.legend(
        title=legend_title, loc="right",
        fontsize=legend_text_size, markerscale=max(legend_size, 0.1),
    )
    if plot_title_name is None:
        plot_title_name = (
            f"Outgoing communication field of {signaling}"
            if pattern == "outgoing"
            else f"Incoming communication field of {signaling}"
        )
    ax.set_title(plot_title_name, fontsize=plot_title_text_size, loc="center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal — uniform 2D grid aggregation (replaces sf::st_make_grid)
# ---------------------------------------------------------------------------
def _grid_aggregate(
    coords: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    cellsize: float, agg: str,
) -> tuple:
    """Aggregate vector field on a uniform 2-D grid.

    Returns ``(centroids, u, v)`` where each returned array is one entry
    per non-empty grid cell.
    """
    if agg not in ("median", "sum"):
        raise ValueError("agg must be 'median' or 'sum'.")
    x0, x1 = coords[:, 0].min(), coords[:, 0].max()
    y0, y1 = coords[:, 1].min(), coords[:, 1].max()
    nx = max(1, int(np.ceil((x1 - x0) / cellsize)))
    ny = max(1, int(np.ceil((y1 - y0) / cellsize)))
    ix = np.clip(((coords[:, 0] - x0) / cellsize).astype(int), 0, nx - 1)
    iy = np.clip(((coords[:, 1] - y0) / cellsize).astype(int), 0, ny - 1)
    bin_id = ix + nx * iy
    centroids = []
    us = []
    vs = []
    for b in np.unique(bin_id):
        mask = bin_id == b
        if not np.any(mask):
            continue
        bx = b % nx
        by = b // nx
        cx = x0 + (bx + 0.5) * cellsize
        cy = y0 + (by + 0.5) * cellsize
        if agg == "median":
            us.append(np.nanmedian(dx[mask]))
            vs.append(np.nanmedian(dy[mask]))
        else:
            us.append(np.nansum(dx[mask]))
            vs.append(np.nansum(dy[mask]))
        centroids.append((cx, cy))
    return np.asarray(centroids), np.asarray(us), np.asarray(vs)


# ---------------------------------------------------------------------------
# Internal — Barnes-style 2-D interpolation (mirrors oce::interpBarnes)
# ---------------------------------------------------------------------------
def _barnes_interp(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    xg: np.ndarray, yg: np.ndarray,
    n_passes: int = 2,
    gamma: float = 0.2,
) -> np.ndarray:
    """Barnes (1964) objective-analysis interpolation onto a regular grid.

    Mirrors :func:`oce::interpBarnes` (default ``n.passes=2``, ``gamma=0.2``).
    Each pass updates the gridded field by adding a Gaussian-weighted mean
    of residuals between the data and the previous gridded estimate at the
    data points.

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Length-N data point coordinates and values.
    xg, yg : numpy.ndarray
        1-D grid coordinate vectors.
    n_passes : int, default ``2``
        Number of refinement passes.
    gamma : float, default ``0.2``
        Length-scale shrink factor between passes.

    Returns
    -------
    numpy.ndarray
        ``(len(yg), len(xg))`` interpolated field.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    XG, YG = np.meshgrid(xg, yg)
    # Initial scale length: average grid spacing.
    if xg.size > 1 and yg.size > 1:
        xrange = float(xg.max() - xg.min())
        yrange = float(yg.max() - yg.min())
        L = max(xrange / max(xg.size - 1, 1), yrange / max(yg.size - 1, 1))
    else:
        L = float(np.abs(x.max() - x.min()) / max(x.size, 1)) or 1.0
    # Convert L → kappa (Gaussian length-scale parameter).
    # oce default sets kappa = (5.052 * 2 * L / pi)^2; we use an
    # equivalent simple form.
    kappa = (5.052 * 2.0 * L / np.pi) ** 2

    # Initial pass: weighted mean of z at every grid point.
    def _pass(field: np.ndarray, residuals: np.ndarray, kappa_use: float) -> np.ndarray:
        out = np.empty_like(field)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                d2 = (x - XG[i, j]) ** 2 + (y - YG[i, j]) ** 2
                w = np.exp(-d2 / kappa_use)
                w_sum = w.sum()
                if w_sum > 0:
                    out[i, j] = field[i, j] + (w * residuals).sum() / w_sum
                else:
                    out[i, j] = field[i, j]
        return out

    field = np.zeros_like(XG)
    field = _pass(field, z, kappa)
    for _ in range(max(n_passes - 1, 0)):
        # Residuals at data points: bilinear interpolate field at data
        # points, then z - field_at_data.
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (yg, xg), field, bounds_error=False, fill_value=None,
        )
        z_at_data = interp(np.column_stack([y, x]))
        residuals = z - z_at_data
        kappa = kappa * gamma
        field = _pass(field, residuals, kappa)
    return field


# ---------------------------------------------------------------------------
# netVisual_CommunFieldGrid (R: visualization.R:5584)
# ---------------------------------------------------------------------------
def netVisual_CommunFieldGrid(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "incoming",
    cellsize: Optional[float] = None,
    grid_resolution: Optional[float] = None,
    grid_field_mean: str = "median",
    color_use: Optional[Sequence[str]] = None,
    point_size: float = 2.0,
    image_alpha: float = 0.3,
    min_mag: float = 0.0,
    arrow_line_color: str = "black",
    arrow_line_alpha: float = 0.9,
    arrow_angle: float = 22.5,
    arrow_length: float = 0.5,
    legend_size: float = 2.0,
    legend_text_size: int = 8,
    title_name: Optional[str] = None,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Gridded communication-flow arrows over the spatial scatter.

    Mirrors :func:`netVisual_CommunFieldGrid` (``visualization.R:5584``).
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if grid_field_mean not in ("median", "sum"):
        raise ValueError("grid_field_mean must be 'median' or 'sum'.")

    field_raw = _resolve_field(adata, slot_name, pattern, signaling)
    field_use = _swap_field(field_raw)
    coords = _xy_from_adata(adata)

    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    dx = field_use[:, 0]
    dy = field_use[:, 1]

    if cellsize is None:
        # min positive nearest-neighbour distance
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        cellsize = float(dists[:, 1].min())
    if grid_resolution is None:
        grid_resolution = 2.0
    newcellsize = float(cellsize) * float(grid_resolution)

    centroids, us, vs = _grid_aggregate(coords, dx, dy, newcellsize, grid_field_mean)

    labels = _idents_from_adata(adata, None)
    cells_level = list(labels.categories)
    if color_use is None:
        color_use = list(scPalette(len(cells_level)))
    color_map = dict(zip(cells_level, color_use))

    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(img, origin="upper")
    for cat in cells_level:
        sel = (np.asarray(labels) == cat)
        if not np.any(sel):
            continue
        ax.scatter(
            coords[sel, 0], coords[sel, 1],
            s=(point_size ** 2) * 6, c=color_map[cat],
            alpha=image_alpha, edgecolors="none", label=str(cat),
        )

    # R uses ``oce::interpBarnes`` to smooth U/V before plotting arrows.
    # We therefore Barnes-interpolate (us, vs) onto a regular grid spanning
    # the centroids' bounding box and mask out grid cells whose nearest
    # data centroid is more than ``newcellsize`` away.
    if centroids.size > 0:
        x0, x1 = float(centroids[:, 0].min()), float(centroids[:, 0].max())
        y0, y1 = float(centroids[:, 1].min()), float(centroids[:, 1].max())
        if x1 > x0 and y1 > y0:
            n_xi = max(int(np.ceil((x1 - x0) / newcellsize)) + 1, 4)
            n_yi = max(int(np.ceil((y1 - y0) / newcellsize)) + 1, 4)
            xg = np.linspace(x0, x1, n_xi)
            yg = np.linspace(y0, y1, n_yi)
            U_grid = _barnes_interp(centroids[:, 0], centroids[:, 1], us, xg, yg)
            V_grid = _barnes_interp(centroids[:, 0], centroids[:, 1], vs, xg, yg)
            # Mask cells whose nearest data point is farther than 1 cell.
            grid_tree = cKDTree(centroids)
            XG, YG = np.meshgrid(xg, yg)
            grid_pts = np.column_stack([XG.ravel(), YG.ravel()])
            nearest, _ = grid_tree.query(grid_pts, k=1)
            mask_far = (nearest > newcellsize).reshape(XG.shape)
            U_grid[mask_far] = 0.0
            V_grid[mask_far] = 0.0
            mag_grid = np.sqrt(U_grid ** 2 + V_grid ** 2)
            keep_grid = mag_grid > min_mag
            if np.any(keep_grid):
                ax.quiver(
                    XG[keep_grid], YG[keep_grid],
                    U_grid[keep_grid], -V_grid[keep_grid],
                    color=arrow_line_color,
                    alpha=arrow_line_alpha,
                    angles="xy", scale_units="xy", scale=1.0,
                    width=0.004, headwidth=arrow_angle / 4.0,
                    headlength=arrow_length * 5.0,
                )
        else:
            # degenerate: fall back to raw centroid quiver
            mag = np.sqrt(us ** 2 + vs ** 2)
            keep = mag > min_mag
            if np.any(keep):
                ax.quiver(
                    centroids[keep, 0], centroids[keep, 1],
                    us[keep], -vs[keep],
                    color=arrow_line_color,
                    alpha=arrow_line_alpha,
                    angles="xy", scale_units="xy", scale=1.0,
                    width=0.004, headwidth=arrow_angle / 4.0,
                    headlength=arrow_length * 5.0,
                )

    ax.set_aspect("equal")
    if img is None:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_title = "Sources" if pattern == "outgoing" else "Targets"
    ax.legend(title=legend_title, loc="right",
              fontsize=legend_text_size, markerscale=max(legend_size, 0.1))
    if title_name is None:
        title_name = (
            f"Outgoing communication flow of {signaling}"
            if pattern == "outgoing"
            else f"Incoming communication flow of {signaling}"
        )
    ax.set_title(title_name, fontsize=10, loc="center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netVisual_CommunFlow (R: visualization.R:5855)
# ---------------------------------------------------------------------------
def _integrate_streamlines(
    XX: np.ndarray,
    YY: np.ndarray,
    UU: np.ndarray,
    VV: np.ndarray,
    *,
    L: float,
    min_L: float,
    res: float,
    n: int,
    jitter: float,
    seed: int = 0,
) -> List[np.ndarray]:
    """RK4 streamline integration on a regular grid (R: ``metR::geom_streamline``).

    Parameters
    ----------
    XX, YY : 2-D ndarray
        Mesh grid coordinates produced by :func:`numpy.meshgrid`.
    UU, VV : 2-D ndarray
        Velocity field components on the same grid.
    L : float
        Maximum integration length (R parity: ``L``).
    min_L : float
        Streamlines shorter than this are dropped (R parity: ``min.L``).
    res : float
        Step size as a fraction of the cell spacing (R parity: ``res``).
    n : int
        Seed-point density per axis (R parity: ``n``); the seed grid has
        approximately ``n × n`` points.
    jitter : float
        Random uniform jitter added to seed positions, in cell-spacing
        units (R parity: ``jitter``).
    seed : int
        RNG seed for the jitter (deterministic by default).

    Returns
    -------
    list of (k, 2) ndarray
        One polyline per surviving streamline.
    """
    rng = np.random.default_rng(seed)
    xs = XX[0, :]
    ys = YY[:, 0]
    dx_cell = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    dy_cell = float(ys[1] - ys[0]) if ys.size > 1 else 1.0
    x_min, x_max = float(xs[0]), float(xs[-1])
    y_min, y_max = float(ys[0]), float(ys[-1])

    def sample(x: float, y: float) -> Tuple[float, float]:
        """Bilinear interpolation of (UU, VV) at (x, y); zero outside grid."""
        if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
            return 0.0, 0.0
        ix = (x - x_min) / dx_cell
        iy = (y - y_min) / dy_cell
        i0 = int(np.floor(ix))
        j0 = int(np.floor(iy))
        i1 = min(i0 + 1, UU.shape[1] - 1)
        j1 = min(j0 + 1, UU.shape[0] - 1)
        i0 = max(min(i0, UU.shape[1] - 1), 0)
        j0 = max(min(j0, UU.shape[0] - 1), 0)
        a = ix - i0
        b = iy - j0
        u = (
            (1 - a) * (1 - b) * UU[j0, i0] + a * (1 - b) * UU[j0, i1]
            + (1 - a) * b * UU[j1, i0] + a * b * UU[j1, i1]
        )
        v = (
            (1 - a) * (1 - b) * VV[j0, i0] + a * (1 - b) * VV[j0, i1]
            + (1 - a) * b * VV[j1, i0] + a * b * VV[j1, i1]
        )
        return float(u), float(v)

    # Build the n×n seed grid with optional jitter.
    seed_xs = np.linspace(x_min, x_max, max(n, 2))
    seed_ys = np.linspace(y_min, y_max, max(n, 2))
    SX, SY = np.meshgrid(seed_xs, seed_ys)
    if jitter > 0:
        SX = SX + rng.uniform(-jitter * dx_cell, jitter * dx_cell, SX.shape)
        SY = SY + rng.uniform(-jitter * dy_cell, jitter * dy_cell, SY.shape)
    seeds = np.column_stack([SX.ravel(), SY.ravel()])

    step_h = float(res) * float(min(dx_cell, dy_cell))
    max_steps = int(np.ceil(L / max(step_h, 1e-12))) + 1

    lines: List[np.ndarray] = []
    for sx, sy in seeds:
        x, y = float(sx), float(sy)
        u0, v0 = sample(x, y)
        if u0 == 0.0 and v0 == 0.0:
            continue
        pts = [(x, y)]
        length = 0.0
        for _ in range(max_steps):
            # RK4 step
            k1u, k1v = sample(x, y)
            k2u, k2v = sample(x + 0.5 * step_h * k1u, y + 0.5 * step_h * k1v)
            k3u, k3v = sample(x + 0.5 * step_h * k2u, y + 0.5 * step_h * k2v)
            k4u, k4v = sample(x + step_h * k3u, y + step_h * k3v)
            du = (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
            dv = (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            mag = float(np.hypot(du, dv))
            if mag < 1e-12:
                break
            # Normalise so the step covers ``step_h`` regardless of vector
            # magnitude (R's metR::geom_streamline integrates by arc
            # length when ``res`` controls step size, not by raw dt).
            x_new = x + step_h * du / mag
            y_new = y + step_h * dv / mag
            if not (x_min <= x_new <= x_max and y_min <= y_new <= y_max):
                break
            pts.append((x_new, y_new))
            length += step_h
            x, y = x_new, y_new
            if length >= L:
                break
        if length >= max(min_L, 0.0) and len(pts) >= 2:
            lines.append(np.asarray(pts, dtype=float))
    return lines


def netVisual_CommunFlow(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "incoming",
    cellsize: Optional[float] = None,
    grid_resolution: Optional[float] = None,
    grid_field_mean: str = "median",
    color_use: Optional[Sequence[str]] = None,
    point_size: float = 2.0,
    image_alpha: float = 0.9,
    L: float = 300.0,
    min_L: float = 0.0,
    res: float = 0.5,
    n: int = 30,
    jitter: float = 3.0,
    arrow_line_color: str = "#666666",
    arrow_line_width: float = 0.7,
    arrow_line_alpha: float = 0.6,
    arrow_angle: float = 12.0,
    arrow_length: float = 0.3,
    legend_size: float = 2.0,
    legend_text_size: int = 8,
    title_name: Optional[str] = None,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Streamlines of the gridded communication field.

    Mirrors :func:`netVisual_CommunFlow` (``visualization.R:5855``). R uses
    :func:`metR::geom_streamline` which seed-point integrates the vector
    field; we port that algorithm directly via :func:`_integrate_streamlines`
    (RK4 on bilinearly-interpolated grid). All R-side parameters
    (``L``, ``min_L``, ``res``, ``n``, ``jitter``, ``arrow_*``) are honoured.
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if grid_field_mean not in ("median", "sum"):
        raise ValueError("grid_field_mean must be 'median' or 'sum'.")

    field_raw = _resolve_field(adata, slot_name, pattern, signaling)
    field_use = _swap_field(field_raw)
    coords = _xy_from_adata(adata)

    img, scalef, _ = _spatial_layer(adata, lib_id=lib_id, image_key=image_key)
    if img is not None:
        coords = coords * scalef

    dx = field_use[:, 0]
    dy = field_use[:, 1]

    if cellsize is None:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        cellsize = float(dists[:, 1].min())
    if grid_resolution is None:
        grid_resolution = 2.0
    newcellsize = float(cellsize) * float(grid_resolution)

    centroids, us, vs = _grid_aggregate(coords, dx, dy, newcellsize, grid_field_mean)

    labels = _idents_from_adata(adata, None)
    cells_level = list(labels.categories)
    if color_use is None:
        color_use = list(scPalette(len(cells_level)))
    color_map = dict(zip(cells_level, color_use))

    fig, ax = plt.subplots(figsize=(6, 6))
    if img is not None:
        ax.imshow(img, origin="upper")
    for cat in cells_level:
        sel = (np.asarray(labels) == cat)
        if not np.any(sel):
            continue
        ax.scatter(
            coords[sel, 0], coords[sel, 1],
            s=(point_size ** 2) * 6, c=color_map[cat],
            alpha=image_alpha, edgecolors="none", label=str(cat),
        )

    if centroids.shape[0] >= 4:
        # Build a rectangular grid and place per-cell vectors on it via
        # nearest-cell lookup. We then call :func:`_integrate_streamlines`
        # which mirrors R's :func:`metR::geom_streamline` semantics
        # (seed-point integration honouring ``L`` / ``min_L`` / ``n`` /
        # ``jitter`` / ``res``). matplotlib's :func:`streamplot` is
        # *density*-based and cannot honour those R parameters.
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()
        n_x = max(int(np.ceil((x1 - x0) / newcellsize)) + 1, 2)
        n_y = max(int(np.ceil((y1 - y0) / newcellsize)) + 1, 2)
        xs = np.linspace(x0, x1, n_x)
        ys = np.linspace(y0, y1, n_y)
        XX, YY = np.meshgrid(xs, ys)
        UU = np.zeros_like(XX, dtype=float)
        VV = np.zeros_like(XX, dtype=float)
        for k in range(centroids.shape[0]):
            cx, cy = centroids[k]
            ix = int(np.clip(np.searchsorted(xs, cx) - 1, 0, len(xs) - 1))
            iy = int(np.clip(np.searchsorted(ys, cy) - 1, 0, len(ys) - 1))
            UU[iy, ix] = us[k]
            # R's ggplot uses ``y = -lat, dy = -v`` (flipped y axis) at
            # visualization.R:6056-6058 so the streamline integrator sees
            # ``v -> -v``; we mirror that here by negating VV.
            VV[iy, ix] = -vs[k]
        polylines = _integrate_streamlines(
            XX, YY, UU, VV,
            L=float(L), min_L=float(min_L), res=float(res),
            n=int(n), jitter=float(jitter),
        )
        # Each polyline → a coloured polyline + a small arrowhead at the tip.
        # R's ``arrow.angle`` (default 12°) controls the arrowhead opening
        # angle; ``arrow.length`` (default 0.3) is in matplotlib points.
        for poly in polylines:
            ax.plot(
                poly[:, 0], poly[:, 1],
                color=arrow_line_color,
                linewidth=arrow_line_width,
                alpha=arrow_line_alpha,
                solid_capstyle="round",
            )
            if len(poly) >= 2:
                tip = poly[-1]
                base = poly[-2]
                dx_arrow = tip[0] - base[0]
                dy_arrow = tip[1] - base[1]
                if dx_arrow == 0.0 and dy_arrow == 0.0:
                    continue
                ax.annotate(
                    "",
                    xy=(tip[0], tip[1]),
                    xytext=(base[0], base[1]),
                    arrowprops=dict(
                        arrowstyle=f"->,head_length={arrow_length},head_width={arrow_length * np.tan(np.radians(arrow_angle))}",
                        color=arrow_line_color,
                        alpha=arrow_line_alpha,
                        lw=arrow_line_width,
                    ),
                )
    ax.set_aspect("equal")
    if img is None:
        ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend_title = "Sources" if pattern == "outgoing" else "Targets"
    ax.legend(title=legend_title, loc="right",
              fontsize=legend_text_size, markerscale=max(legend_size, 0.1))
    if title_name is None:
        title_name = (
            f"Outgoing communication flow of {signaling}"
            if pattern == "outgoing"
            else f"Incoming communication flow of {signaling}"
        )
    ax.set_title(title_name, fontsize=10, loc="center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# netVisual_CommunFieldZoomIn (R: visualization.R:6108)
# ---------------------------------------------------------------------------
def netVisual_CommunFieldZoomIn(
    adata: ad.AnnData,
    signaling: str,
    slot_name: str = "netP",
    pattern: str = "incoming",
    color_use: Optional[Sequence[str]] = None,
    point_size: float = 2.0,
    zoom_to: int = 1,
    anchor_x: float = 0.5,
    anchor_y: float = 0.5,
    zoom_pt_size: Optional[float] = None,
    alpha_image: float = 0.2,
    arrow_angle: float = 15.0,
    arrow_length: float = 0.6,
    arrow_size: float = 1.0,
    arrow_color: Optional[str] = None,
    legend_size: float = 0.5,
    legend_text_size: int = 8,
    plot_title_name: Optional[str] = None,
    plot_title_text_size: int = 8,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> Figure:
    """Side-by-side full + zoomed-in communication-field view.

    Mirrors :func:`netVisual_CommunFieldZoomIn` (``visualization.R:6108``)
    by stitching two :func:`netVisual_CommunField` panels together.
    """
    if zoom_pt_size is None:
        zoom_pt_size = point_size + zoom_to * 0.5

    fig1 = netVisual_CommunField(
        adata=adata, signaling=signaling, slot_name=slot_name, pattern=pattern,
        zoom_to=zoom_to, zoom_mode=True, anchor_x=anchor_x, anchor_y=anchor_y,
        color_use=color_use, point_size=point_size, alpha_image=alpha_image,
        arrow_angle=arrow_angle, arrow_length=arrow_length, arrow_size=arrow_size,
        arrow_color=arrow_color, legend_size=legend_size,
        legend_text_size=legend_text_size, plot_title_name=plot_title_name,
        plot_title_text_size=plot_title_text_size, lib_id=lib_id, image_key=image_key,
    )
    title2 = (
        (plot_title_name or "")
        + "\n"
        + f"zoom.to={zoom_to}x,anchor.x={anchor_x},anchor.y={anchor_y}".replace(" ", "")
    )
    fig2 = netVisual_CommunField(
        adata=adata, signaling=signaling, slot_name=slot_name, pattern=pattern,
        zoom_to=zoom_to, zoom_mode=False, anchor_x=anchor_x, anchor_y=anchor_y,
        color_use=color_use, point_size=zoom_pt_size, alpha_image=alpha_image,
        arrow_angle=arrow_angle, arrow_length=arrow_length, arrow_size=arrow_size,
        arrow_color=arrow_color, legend_size=legend_size,
        legend_text_size=legend_text_size, plot_title_name=title2,
        plot_title_text_size=plot_title_text_size, lib_id=lib_id, image_key=image_key,
    )

    # Combine the two figures into a single side-by-side figure.
    combined = plt.figure(figsize=(12, 6))
    canvas1 = fig1.canvas
    canvas2 = fig2.canvas
    canvas1.draw()
    canvas2.draw()
    img1 = np.asarray(canvas1.buffer_rgba())
    img2 = np.asarray(canvas2.buffer_rgba())
    ax1 = combined.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    ax1.axis("off")
    ax2 = combined.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    ax2.axis("off")
    combined.tight_layout()
    plt.close(fig1)
    plt.close(fig2)
    return combined


# ---------------------------------------------------------------------------
# visualizePie (R: visualization.R:6677)
# ---------------------------------------------------------------------------
def visualizePie(
    proportion: pd.DataFrame,
    spatial_location: pd.DataFrame,
    colors: Optional[Sequence[str]] = None,
    radius: Optional[float] = None,
    seed: Optional[int] = None,
) -> Figure:
    """Spatial scatter-pie of cell-type proportions.

    Mirrors :func:`visualizePie` (``visualization.R:6677``). Adapted from
    the CARD package — the rownames of ``proportion`` and
    ``spatial_location`` must match.

    Parameters
    ----------
    proportion : pandas.DataFrame
        Rows = spots, columns = cell types.
    spatial_location : pandas.DataFrame
        Must contain ``"x"`` and ``"y"`` (or ``"x_cent"`` / ``"y_cent"``)
        columns; rows match ``proportion``.
    colors : sequence of str or None
        Per-cell-type colours. If ``None``, the CARD palette is used.
    radius : float or None
        Pie radius (data units). ``None`` triggers the CARD heuristic.
    seed : int or None
        RNG seed used when sampling colours from the candidate palette.

    Returns
    -------
    matplotlib.figure.Figure
    """
    res = pd.DataFrame(proportion).copy()
    # gtools::mixedsort → natural sort (e.g. "T2" < "T10").
    from natsort import natsorted

    res = res.reindex(columns=list(natsorted(list(res.columns))))
    location = pd.DataFrame(spatial_location).copy()
    if not list(res.index) == list(location.index):
        raise ValueError(
            "The rownames of proportion data does not match with the rownames of spatial location data"
        )

    # Resolve x/y columns (R uses x_cent/y_cent in the call, but the docstring
    # mentions $x, $y for the radius heuristic — accept both).
    if "x_cent" in location.columns and "y_cent" in location.columns:
        x_col, y_col = "x_cent", "y_cent"
    elif "x" in location.columns and "y" in location.columns:
        x_col, y_col = "x", "y"
    else:
        raise ValueError(
            "spatial_location must contain either ('x','y') or ('x_cent','y_cent') columns."
        )

    color_candidate = (
        "#1e77b4", "#ff7d0b", "#ceaaa3", "#2c9f2c", "#babc22", "#d52828",
        "#9267bc", "#8b544c", "#e277c1", "#d42728", "#adc6e8", "#97df89",
        "#fe9795", "#4381bd", "#f2941f", "#5aa43a", "#cc4d2e", "#9f83c8",
        "#91675a", "#da8ec8", "#929292", "#c3c237", "#b4e0ea", "#bacceb",
        "#f7c685", "#dcf0d0", "#f4a99f", "#c8bad8", "#F56867", "#FEB915",
        "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E",
        "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236",
        "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370", "#877F6C", "#268785",
        "#f4f1de", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#a8dadc",
        "#f1faee", "#f08080",
    )
    n_ct = res.shape[1]
    if colors is None:
        if n_ct > len(color_candidate):
            from ._palette import _r_color_ramp_palette

            colors = list(_r_color_ramp_palette(color_candidate, n_ct))
        else:
            # R parity: ``set.seed(iseed); colorCandidate[sample(1:N, n_ct)]``
            # uses R's Mersenne-Twister + Rejection sample-without-replacement
            # (R 3.6+ default).  Bundled here as :class:`RMT` so legend
            # colours are bit-identical to R's output for the same seed.
            from ._rng_r import RMT

            iseed = 12345 if seed is None else int(seed)
            chosen_idx_1based = RMT(iseed).sample_int_no_replace(
                len(color_candidate), n_ct
            )
            colors = [color_candidate[i - 1] for i in chosen_idx_1based]

    df = pd.concat([res, location], axis=1)
    if radius is None:
        x_range = float(df[x_col].max() - df[x_col].min())
        y_range = float(df[y_col].max() - df[y_col].min())
        area = x_range * y_range
        r2 = area / float(df.shape[0])
        r2 = r2 / np.pi
        radius = float(np.sqrt(r2) * 0.85)

    fig, ax = plt.subplots(figsize=(6, 6))
    _scatterpie_layer(
        df, x=x_col, y=y_col, r=radius,
        cols=list(res.columns), ax=ax, colors=list(colors),
    )
    # Manually set limits because Wedge patches don't auto-extend axis.
    margin = radius * 1.5
    ax.set_xlim(df[x_col].min() - margin, df[x_col].max() + margin)
    ax.set_ylim(df[y_col].min() - margin, df[y_col].max() + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#E3E3E3")
        ax.spines[spine].set_linewidth(0.5)
    # Legend handles
    for k, cat in enumerate(res.columns):
        ax.plot([], [], "s", color=colors[k], label=str(cat))
    ax.legend(title="Cell Type", loc="lower center", bbox_to_anchor=(0.5, -0.1),
              ncol=min(len(res.columns), 4))
    fig.tight_layout()
    return fig
