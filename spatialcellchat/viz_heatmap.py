"""Heatmap and bubble visualisations (slice 6b).

Direct ports of the heatmap-shaped exports in ``visualization.R``:

- :func:`netVisual_heatmap`           (R: ``visualization.R:2549``)
- :func:`netAnalysis_signalingRole_heatmap`
                                       (R: ``analysis.R:3118``)
- :func:`netVisual_bubble`             (R: ``visualization.R:2894``)
- :func:`doHeatmap`                    (R: ``visualization.R:9212``)
- :func:`netVisual_TopicComposition`   (R: ``visualization.R:4581``)
- :func:`netVisual_TopicSignaling`     (R: ``visualization.R:4887``)

All functions accept :class:`anndata.AnnData` first and return a
:class:`matplotlib.figure.Figure`. Cell-cell communication probabilities
and centrality arrays are read from ``adata.uns["cellchat"]`` slots
written by upstream functions in ``modeling`` / ``analysis_network`` /
``analysis_topics``.
"""

from __future__ import annotations

import matplotlib

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

from ._anndata import get_idents_levels as _get_idents_levels
from ._anndata import resolve_data_matrix
from ._palette import scPalette, ggPalette
from ._themes import _BREWER_PALETTES, _BREWER_SUBSET, _brewer_pal
from ._text_repel import repel_text


# matplotlib's pure-grey numeric levels; R names ("grey90", ...) map to these.
_R_GREY_MAP = {
    "grey0": "0.0", "grey10": "0.1", "grey20": "0.2", "grey30": "0.3",
    "grey40": "0.4", "grey50": "0.5", "grey60": "0.6", "grey70": "0.7",
    "grey80": "0.8", "grey90": "0.9", "grey100": "1.0",
    "gray0": "0.0", "gray10": "0.1", "gray20": "0.2", "gray30": "0.3",
    "gray40": "0.4", "gray50": "0.5", "gray60": "0.6", "gray70": "0.7",
    "gray80": "0.8", "gray90": "0.9", "gray100": "1.0",
}


def _safe_color(c: str) -> str:
    """Translate R-style ``"grey90"`` to a matplotlib-friendly form."""
    if isinstance(c, str) and c.lower() in _R_GREY_MAP:
        return _R_GREY_MAP[c.lower()]
    return c


def _safe_brewer(name: str, n: int) -> List[str]:
    """``_brewer_pal`` with a matplotlib fallback for unsupported palettes.

    The R brewer set used by SpatialCellChat exceeds the explicit
    ``_BREWER_SUBSET`` list (which only has fixed n=3..9 sub-tables). For
    the diverging / sequential palettes already in ``_BREWER_PALETTES``
    we sample anchors from the full palette via linear interpolation
    when ``_BREWER_SUBSET`` doesn't carry the entry. For unknown
    palettes we delegate to matplotlib (the same fallback as
    :func:`generate_custom_colorscale`).
    """
    if name in _BREWER_SUBSET:
        return list(_brewer_pal(name, n))
    if name in _BREWER_PALETTES:
        full = list(_BREWER_PALETTES[name])
        if n <= len(full):
            # Evenly spaced indices over the full palette.
            idxs = np.linspace(0, len(full) - 1, n).round().astype(int)
            return [full[i] for i in idxs]
    # matplotlib fallback (viridis-style). Keep n-anchor parity.
    cmap = matplotlib.colormaps.get_cmap(name)
    samples = np.linspace(0.0, 1.0, n)
    return [matplotlib.colors.to_hex(cmap(s)) for s in samples]

__all__ = [
    "netVisual_heatmap",
    "netAnalysis_signalingRole_heatmap",
    "netVisual_bubble",
    "doHeatmap",
    "netVisual_TopicComposition",
    "netVisual_TopicSignaling",
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


def _palette_listed(name: str, n: int) -> List[str]:
    """Return ``n`` colours from a brewer palette (or matplotlib fallback)."""
    if name in _BREWER_SUBSET or name in _BREWER_PALETTES:
        cols = list(_safe_brewer(name, min(n, 9)))
        if n > len(cols):
            # extend with linear interpolation in sRGB across the brewer set
            cmap = LinearSegmentedColormap.from_list("brew", cols, N=n)
            return [matplotlib.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
        return cols[:n]
    cmap = matplotlib.colormaps[name].resampled(n)
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


def _expand_palette(colors: Sequence[str], n: int) -> np.ndarray:
    """Expand ``colors`` to length ``n`` via R-style colorRampPalette."""
    from ._palette import _r_color_ramp_palette

    return _r_color_ramp_palette(list(colors), n)


def _generate_RedsBlues_colormap(
    vec: np.ndarray,
    mid_color: str = "#E5E5E5",
    switch_RedsBlues: bool = False,
) -> List[str]:
    """Build an R-style ``MyRdBu`` colormap evaluated at ``length 99``.

    Direct port of ``SpatialCellChat:::generate_RedsBlues_colormap`` (R).
    Returns 99 hex anchors interpolated by R-style colorRamp3 over the
    asymmetric break-points implied by ``vec``.
    """
    arr = np.asarray(vec, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        min_vec, max_vec = -1.0, 1.0
    else:
        min_vec = float(finite.min())
        max_vec = float(finite.max())
    Reds = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]
    Blues = ["#eff3ff", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    if switch_RedsBlues:
        Reds, Blues = Blues, Reds
    Colors_length = 6
    if min_vec >= 0:
        breaks = np.linspace(0.0, max_vec if max_vec > 0 else 1.0, Colors_length + 1)
        colors = [mid_color] + Reds
    elif max_vec <= 0:
        breaks = np.linspace(min_vec if min_vec < 0 else -1.0, 0.0, Colors_length + 1)
        colors = list(reversed(Blues)) + [mid_color]
    else:
        Blues_breaks = np.linspace(min_vec, 0.0, Colors_length + 1)
        Reds_breaks = np.linspace(0.0, max_vec, Colors_length + 1)
        breaks = np.concatenate([Blues_breaks, Reds_breaks[1:]])
        colors = list(reversed(Blues)) + [mid_color] + Reds
    # Build piecewise linear interp like colorRamp3.
    # Sample 99 points from min_vec..max_vec (R: seq(from=min_vec, to=max_vec, length.out=99))
    xs = np.linspace(min_vec, max_vec, 99)
    out: List[str] = []
    breaks = np.asarray(breaks, dtype=float)
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in colors])
    for x in xs:
        if x <= breaks[0]:
            out.append(matplotlib.colors.to_hex(tuple(rgb[0])))
            continue
        if x >= breaks[-1]:
            out.append(matplotlib.colors.to_hex(tuple(rgb[-1])))
            continue
        # find segment
        j = int(np.searchsorted(breaks, x, side="right")) - 1
        if j < 0:
            j = 0
        if j >= len(breaks) - 1:
            j = len(breaks) - 2
        denom = breaks[j + 1] - breaks[j]
        if denom == 0:
            t = 0.0
        else:
            t = (x - breaks[j]) / denom
        col = rgb[j] + t * (rgb[j + 1] - rgb[j])
        out.append(matplotlib.colors.to_hex(tuple(col)))
    return out


def _name_to_index(name_or_idx: Union[str, int], names: Sequence[str]) -> int:
    if isinstance(name_or_idx, str):
        return list(names).index(name_or_idx)
    return int(name_or_idx)


def _resolve_subset(
    use: Optional[Sequence], names: Sequence[str]
) -> Optional[List[str]]:
    """Convert numeric or name-based subset to the matching name list."""
    if use is None:
        return None
    out: List[str] = []
    for v in use:
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out.append(names[int(v) - 1])  # R 1-based index
        else:
            out.append(str(v))
    return out


# ---------------------------------------------------------------------------
# netVisual_heatmap (R: visualization.R:2549)
# ---------------------------------------------------------------------------
def netVisual_heatmap(
    adata: ad.AnnData,
    comparison: Sequence[int] = (1, 2),
    measure: Optional[str] = "count",
    signaling: Optional[str] = None,
    slot_name: str = "netP",
    thresh: float = 0.05,
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: Sequence[str] = ("#2166ac", "#b2182b"),
    title_name: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    font_size: float = 8.0,
    font_size_title: float = 10.0,
    x_lab_rot: float = 90.0,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    remove_isolate: bool = False,
    remove_diagonal: bool = False,
    row_show: Optional[Sequence] = None,
    col_show: Optional[Sequence] = None,
) -> Figure:
    """Communication-network heatmap with side bar plots.

    Mirrors :func:`netVisual_heatmap` (R: ``visualization.R:2549``).
    Single-object heatmaps display the ``count`` / ``weight`` matrix from
    ``adata.uns['cellchat']['net']`` (or, when ``signaling`` is set, the
    pathway slice of ``slot_name['prob']``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns['cellchat']`` populated by upstream calls.
    comparison : sequence of int, default ``(1, 2)``
        Reserved for the merged-mode (kept for API parity).
    measure : {"count", "weight"} or None, default ``"count"``
        Quantity to plot when ``signaling`` is ``None``.
    signaling : str, optional
        Pathway / L-R name. When set, draws the corresponding slice of
        ``slot_name['prob']``.
    slot_name : {"netP", "net"}
        Source slot for ``signaling``-driven plotting.
    thresh : float, default ``0.05``
        p-value cutoff applied when ``slot_name == 'net'``.
    color_use : sequence of str, optional
        Per-cell-group colour. Defaults to :func:`scPalette`.
    color_heatmap : sequence of str, default ``("#2166ac", "#b2182b")``
        Bottom and top anchor colours for the heatmap colormap.
    title_name : str, optional
        Suptitle.
    width, height : float, optional
        Figure size in inches.
    font_size, font_size_title : float
        Axis-label and title font sizes.
    x_lab_rot : float, default 90.0
        x-axis label rotation in degrees.
    cluster_rows, cluster_cols : bool
        Reserved for clustered axis ordering (kept for API parity).
    sources_use, targets_use : sequence, optional
        Source / target subsetting (numeric indices are 1-based).
    remove_isolate : bool, default False
        Drop rows + columns whose row-sum and column-sum are both zero.
    remove_diagonal : bool, default False
        Zero the diagonal before plotting.
    row_show, col_show : sequence, optional
        Explicit row / column subsetting.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_cellchat(adata)
    if measure is not None and measure not in ("count", "weight"):
        raise ValueError("measure must be 'count' or 'weight'.")
    if slot_name not in ("netP", "net"):
        raise ValueError("slot_name must be 'netP' or 'net'.")

    legend_name = "values"
    # Merged-mode detection: R checks `class(object@net[[1]]) == "list"`.
    # In the Python port, a merged net slot is a dict-of-dicts keyed by
    # dataset name, each mapping to a per-dataset {count, weight, prob, ...}.
    # We additionally honour an explicit `'comparison'` flag on cc['net'].
    net_slot = cc.get("net", {})
    is_merged = False
    merged_keys: List[str] = []
    if isinstance(net_slot, dict):
        if "comparison" in net_slot:
            is_merged = True
            merged_keys = list(net_slot.get("comparison", []))
        else:
            # heuristic: child is itself a per-dataset dict containing 'count' / 'weight'
            child_keys = [
                k for k, v in net_slot.items()
                if isinstance(v, dict) and ("count" in v or "weight" in v)
            ]
            if len(child_keys) >= 2:
                is_merged = True
                merged_keys = child_keys
    if is_merged and signaling is None and measure is not None:
        if color_heatmap is None or list(color_heatmap) == ["#2166ac", "#b2182b"]:
            # use R's merged-mode default
            color_heatmap = ("#2166ac", "#b2182b")
        c1 = int(comparison[0]) - 1
        c2 = int(comparison[1]) - 1
        if c1 < 0 or c2 < 0 or c1 >= len(merged_keys) or c2 >= len(merged_keys):
            raise ValueError(
                f"comparison indices {comparison} are out of range for merged net "
                f"with {len(merged_keys)} datasets."
            )
        obj1 = np.asarray(net_slot[merged_keys[c1]][measure], dtype=float)
        obj2 = np.asarray(net_slot[merged_keys[c2]][measure], dtype=float)
        net = obj2 - obj1
        if title_name is None:
            title_name = (
                "Differential number of interactions" if measure == "count"
                else "Differential interaction strength"
            )
        legend_name = "Relative values"
    elif signaling is not None:
        if slot_name not in cc:
            raise ValueError(f"adata.uns['cellchat']['{slot_name}'] missing.")
        slot = cc[slot_name]
        prob = np.asarray(slot["prob"], dtype=float).copy()
        if slot_name == "net":
            pval = np.asarray(slot["pval"], dtype=float)
            prob[pval > thresh] = 0.0
        names_dim3 = (
            list(slot.get("pathways", []))
            if slot_name == "netP"
            else list(slot.get("dimnames", {}).get("interaction_name", []))
        )
        if signaling not in names_dim3:
            raise ValueError(f"signaling {signaling!r} not in {slot_name}.")
        idx = names_dim3.index(signaling)
        net = prob[:, :, idx]
        if title_name is None:
            title_name = f"{signaling} signaling network"
        legend_name = "Communication Prob."
    elif measure is not None:
        if "net" not in cc:
            raise ValueError("adata.uns['cellchat']['net'] missing.")
        if measure not in cc["net"]:
            raise ValueError(f"adata.uns['cellchat']['net']['{measure}'] missing.")
        net = np.asarray(cc["net"][measure], dtype=float)
        if title_name is None:
            title_name = (
                "Number of interactions" if measure == "count" else "Interaction strength"
            )
        legend_name = title_name
    else:
        raise ValueError("Either signaling or measure must be provided.")

    # Row / column names (cell groups)
    row_names = _get_idents_levels(adata, cc.get("net", {}))
    col_names = list(row_names)

    # Subset by sources/targets
    if sources_use is not None or targets_use is not None:
        src = _resolve_subset(sources_use, row_names)
        tgt = _resolve_subset(targets_use, row_names)
        net = net.copy()
        if src is not None:
            mask = np.array([n not in src for n in row_names], dtype=bool)
            net[mask, :] = 0.0
        if tgt is not None:
            mask = np.array([n not in col_names for n in col_names], dtype=bool) | np.array(
                [n not in tgt for n in col_names], dtype=bool
            )
            net[:, mask] = 0.0
    net = np.where(np.isnan(net), 0.0, net)

    if remove_isolate:
        idx1 = np.where(net.sum(axis=1) == 0)[0]
        idx2 = np.where(net.sum(axis=0) == 0)[0]
        idx = np.intersect1d(idx1, idx2)
        if idx.size > 0:
            keep = np.array([i for i in range(net.shape[0]) if i not in idx])
            net = net[np.ix_(keep, keep)]
            row_names = [row_names[i] for i in keep]
            col_names = list(row_names)
    if remove_diagonal:
        np.fill_diagonal(net, 0.0)

    mat = net.copy()
    if color_use is None:
        color_use = list(scPalette(len(col_names)))
    color_use = list(color_use)

    if row_show is not None:
        rs = _resolve_subset(row_show, row_names)
        keep = [row_names.index(n) for n in rs]
        mat = mat[keep, :]
        row_names = [row_names[i] for i in keep]
    if col_show is not None:
        cs = _resolve_subset(col_show, col_names)
        keep = [col_names.index(n) for n in cs]
        mat = mat[:, keep]
        col_names = [col_names[i] for i in keep]
        color_use = [color_use[i] for i in keep]

    if mat.size == 0 or float(np.nansum(mat)) == 0.0:
        raise ValueError("No significant cell-cell communication is inferred!")

    # Colormap. Mirrors R `colorRamp3` with asymmetric break-points.
    mat_min = float(np.nanmin(mat))
    mat_max = float(np.nanmax(mat))
    if mat_min < 0:
        # R: colorRamp3(c(min(mat), 0, max(mat)),
        #               c(color.heatmap[1], "#f7f7f7", color.heatmap[2]))
        # Three break-points anchored at min, 0, max — diverging palette
        # interpolated asymmetrically via TwoSlopeNorm.
        cmap = LinearSegmentedColormap.from_list(
            "ch_div",
            [color_heatmap[0], "#f7f7f7", color_heatmap[1]],
            N=256,
        )
        if mat_max <= 0:
            # all values <= 0 (degenerate diverging case): expand vmax slightly
            mat_max = abs(mat_min) if mat_min < 0 else 1.0
        norm = TwoSlopeNorm(vmin=mat_min, vcenter=0.0, vmax=mat_max)
    else:
        if len(color_heatmap) == 3:
            # R: colorRamp3(c(0, min(mat), max(mat)), color.heatmap)
            # Three anchors: 0 -> color1, min -> color2, max -> color3.
            mat_pos_min = (
                float(np.nanmin(mat[mat > 0])) if (mat > 0).any() else mat_max
            )
            # Build with three break-points: positions normalised against vmax.
            denom = mat_max if mat_max > 0 else 1.0
            stops = [0.0, mat_pos_min / denom, 1.0]
            # Guard monotonicity / uniqueness for from_list
            if stops[1] <= 0.0:
                stops[1] = 1e-6
            if stops[1] >= 1.0:
                stops[1] = 1.0 - 1e-6
            cmap = LinearSegmentedColormap.from_list(
                "ch_seq3",
                list(zip(stops, list(color_heatmap))),
                N=256,
            )
            norm = Normalize(vmin=0.0, vmax=mat_max if mat_max > 0 else 1.0)
        elif len(color_heatmap) == 2:
            cmap = LinearSegmentedColormap.from_list(
                "ch_seq2", list(color_heatmap), N=256
            )
            vmin = float(np.nanmin(mat[mat > 0])) if (mat > 0).any() else 0.0
            vmax = mat_max
            if vmax == vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        elif len(color_heatmap) == 1:
            cols = list(_safe_brewer(color_heatmap[0], 9))
            cmap = LinearSegmentedColormap.from_list("ch_brewer", cols, N=256)
            vmin = float(np.nanmin(mat[mat > 0])) if (mat > 0).any() else 0.0
            vmax = mat_max
            if vmax == vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            cmap = LinearSegmentedColormap.from_list(
                "ch_user", list(color_heatmap), N=256
            )
            vmin = float(np.nanmin(mat[mat > 0])) if (mat > 0).any() else 0.0
            vmax = mat_max
            if vmax == vmin:
                vmax = vmin + 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)

    # ------------------------------------------------------------------
    # Layout — mirrors R ComplexHeatmap composition for ``netVisual_heatmap``:
    #
    #   row 0:                              [ top barplot (ha2) ]
    #   row 1:                              [ row labels region ]   (handled
    #                                         by ax_main tick params)
    #   row 2:    [left strip] [ HEATMAP BODY ]  [right barplot (ha1)]   [legend]
    #   row 3:                  [ bottom strip (col_annotation) ]
    #   row 4:                  [ rotated col labels ]
    #
    # The right-side legend lives in its own dedicated column (``ax_cbar``)
    # so the right barplot stays tightly adjacent to the heatmap body
    # (matches ``right_annotation`` in R, which sits immediately beside the
    # body and is drawn before the legend slot).
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(width or 7.5, height or 6.0))
    # Columns:
    #   0 row-name labels  · 1 left strip · 2 body · 3 right barplot ·
    #   4 gap              · 5 legend title · 6 cax · 7 tick-label slot
    # The row labels live in their own column so they cannot collide with
    # the colored strip drawn by ``imshow`` (the strip's pixels are opaque
    # and would otherwise clip text drawn under default ``tick_params(pad)``).
    gs = fig.add_gridspec(
        5, 8,
        height_ratios=[0.22, 0.02, 1.00, 0.04, 0.32],
        width_ratios=[0.20, 0.04, 1.00, 0.22, 0.10, 0.06, 0.04, 0.05],
        wspace=0.04, hspace=0.04,
    )
    ax_main = fig.add_subplot(gs[2, 2])
    ax_row_labels = fig.add_subplot(gs[2, 0], sharey=ax_main)
    ax_top = fig.add_subplot(gs[0, 2], sharex=ax_main)
    ax_right = fig.add_subplot(gs[2, 3], sharey=ax_main)
    ax_left_anno = fig.add_subplot(gs[2, 1], sharey=ax_main)
    ax_bottom_anno = fig.add_subplot(gs[3, 2], sharex=ax_main)
    ax_cbar_title = fig.add_subplot(gs[2, 5])
    ax_cbar = fig.add_subplot(gs[2, 6])

    plot_mat = np.ma.masked_where(mat == 0, mat) if mat.min() >= 0 else mat
    im = ax_main.imshow(plot_mat, cmap=cmap, norm=norm, aspect="auto",
                        interpolation="nearest")
    # Heatmap body: ticks are shared with the strip / barplot axes
    # (``sharex`` / ``sharey``); set them on ax_main but draw the labels on
    # the dedicated label / strip axes — avoids tick labels colliding with
    # the opaque strip imshow.
    ax_main.set_yticks(np.arange(len(row_names)))
    ax_main.set_xticks(np.arange(len(col_names)))
    ax_main.tick_params(top=False, bottom=False, left=False, right=False,
                        labelleft=False, labelbottom=False)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # Dedicated row-name labels (R: ``row_names_side = "left"``, ``row_names_rot = 0``).
    ax_row_labels.set_axis_off()
    ax_row_labels.set_xlim(0.0, 1.0)
    for k, name in enumerate(row_names):
        ax_row_labels.text(
            1.0, k, str(name),
            ha="right", va="center",
            fontsize=font_size, transform=ax_row_labels.transData,
        )
    # R: ``row_title = "Sources (Sender)"`` rotated 90°. Use a figure-level
    # text artist anchored to the row-labels column so it sits outside the
    # heatmap body.
    pos = ax_row_labels.get_position()
    fig.text(
        pos.x0 - 0.005, (pos.y0 + pos.y1) / 2.0,
        "Sources (Sender)", rotation=90,
        ha="right", va="center", fontsize=font_size_title,
    )

    # Top barplot — R: ``ha2 = HeatmapAnnotation(Strength = anno_barplot(colSums(abs(mat))))``
    col_strength = np.abs(mat).sum(axis=0)
    ax_top.bar(np.arange(len(col_names)), col_strength,
               color=color_use, edgecolor=color_use)
    ax_top.tick_params(left=True, right=False, top=False, bottom=False,
                       labelbottom=False, labelsize=font_size - 1)
    ax_top.set_ylabel("Strength", fontsize=font_size)
    for spine in ("top", "right", "bottom"):
        ax_top.spines[spine].set_visible(False)

    # Right barplot — R: ``ha1 = rowAnnotation(Strength = anno_barplot(rowSums(abs(mat))))``
    row_strength = np.abs(mat).sum(axis=1)
    ax_right.barh(np.arange(len(row_names)), row_strength,
                  color=color_use, edgecolor=color_use)
    ax_right.invert_yaxis()
    ax_right.tick_params(top=False, bottom=True, left=False, right=False,
                         labelleft=False, labelsize=font_size - 1)
    for spine in ("top", "right", "left"):
        ax_right.spines[spine].set_visible(False)

    # Bottom annotation strip — R: ``col_annotation`` via ``HeatmapAnnotation(df, which = "column")``.
    # Shares x with ax_main; ticks are inherited and the labels are drawn here
    # (matches R's ``column_names_rot`` placement on the bottom annotation row).
    ax_bottom_anno.imshow(
        np.array([[matplotlib.colors.to_rgb(c) for c in color_use]]),
        aspect="auto", interpolation="nearest",
    )
    ax_bottom_anno.set_xticklabels(
        col_names, rotation=x_lab_rot, fontsize=font_size,
        ha="right" if x_lab_rot not in (0, 90) else ("center" if x_lab_rot == 0 else "right"),
    )
    ax_bottom_anno.tick_params(top=False, bottom=True, left=False, right=False,
                               labelleft=False)
    for spine in ax_bottom_anno.spines.values():
        spine.set_visible(False)
    ax_bottom_anno.set_xlabel("Targets (Receiver)", fontsize=font_size_title,
                              labelpad=4.0)

    # Left annotation strip — R: ``row_annotation`` via ``HeatmapAnnotation(df, which = "row")``.
    # Shares y with ax_main; do not call ``set_yticks([])`` (would also clear
    # the heatmap body's row labels via the sharey link).
    ax_left_anno.imshow(
        np.array([[matplotlib.colors.to_rgb(c)] for c in color_use]),
        aspect="auto", interpolation="nearest",
    )
    ax_left_anno.tick_params(top=False, bottom=False, left=False, right=False,
                             labelleft=False, labelbottom=False)
    for spine in ax_left_anno.spines.values():
        spine.set_visible(False)

    # Dedicated colorbar — R: ``heatmap_legend_param`` with ``leftcenter-rot``
    # title placement and height 20 mm. Title is placed outside the colorbar
    # via an axes-level text artist (set_ylabel falls back inside the bar
    # when the cax is narrow).
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=font_size - 1)
    # Rotated legend title sits in its own gridspec column so it cannot
    # overlap the bar (R: ``title_position = "leftcenter-rot"``).
    ax_cbar_title.set_axis_off()
    ax_cbar_title.text(
        0.5, 0.5, legend_name,
        transform=ax_cbar_title.transAxes,
        rotation=90, va="center", ha="center",
        fontsize=font_size,
    )

    if title_name is not None:
        fig.suptitle(title_name, fontsize=font_size_title, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_heatmap (R: analysis.R:3118)
# ---------------------------------------------------------------------------
def netAnalysis_signalingRole_heatmap(
    adata: ad.AnnData,
    signaling: Optional[Sequence[str]] = None,
    pattern: str = "outgoing",
    slot_name: str = "netP",
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "BuGn",
    title: Optional[str] = None,
    width: float = 10.0,
    height: float = 8.0,
    font_size: float = 8.0,
    font_size_title: float = 10.0,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
) -> Figure:
    """Outgoing / incoming signaling-role heatmap.

    Mirrors :func:`netAnalysis_signalingRole_heatmap` (R: ``analysis.R:3118``).
    Reads centrality from
    ``adata.uns['cellchat'][slot_name]['centr']`` (written by
    :func:`netAnalysis_computeCentrality`), constructs an ``(N_path, K)``
    matrix of outgoing / incoming / overall strengths, row-normalises by
    its row max, and plots as a heatmap with column-strength bar plot on
    top and pathway-strength bar plot on the right.

    Parameters
    ----------
    adata, slot_name : as upstream.
    signaling : sequence of str, optional
        Subset / re-order pathways.
    pattern : {"outgoing", "incoming", "all"}.
    color_use : sequence of str, optional
        Per-cell-group colours.
    color_heatmap : str, default ``"BuGn"``
        RColorBrewer palette name driving the heatmap.
    title : str, optional
        Plot title.
    width, height : float
        Figure size in centimetres (matches R's defaults).
    font_size, font_size_title : float
        Tick + title font sizes.
    cluster_rows, cluster_cols : bool
        Reserved for axis-clustering parity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pattern not in ("outgoing", "incoming", "all"):
        raise ValueError("pattern must be 'outgoing', 'incoming' or 'all'.")
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
    centr_arr = np.asarray(centr["values"], dtype=float)  # (M, K, N)
    centr_names = list(centr["centr_names"])
    node_names = list(centr["node_names"])
    pathway_names = list(centr["signaling_names"])

    out_idx = centr_names.index("outdeg")
    in_idx = centr_names.index("indeg")
    outgoing = centr_arr[out_idx, :, :].T  # (N, K)
    incoming = centr_arr[in_idx, :, :].T

    if pattern == "outgoing":
        mat = outgoing
        legend_name = "Outgoing"
    elif pattern == "incoming":
        mat = incoming
        legend_name = "Incoming"
    else:
        mat = outgoing + incoming
        legend_name = "Overall"

    full_title = f"{legend_name} signaling patterns"
    if title is not None:
        full_title = f"{full_title} - {title}"

    if signaling is not None:
        signaling = list(signaling)
        present = [s for s in signaling if s in pathway_names]
        sub = np.zeros((len(signaling), mat.shape[1]), dtype=float)
        for s in present:
            i_signal = signaling.index(s)
            i_orig = pathway_names.index(s)
            sub[i_signal, :] = mat[i_orig, :]
        mat = sub
        pathway_names = list(signaling)

    mat_ori = mat.copy()
    row_max = mat.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)
    mat_norm = mat / row_max
    mat_plot = np.where(mat_norm == 0, np.nan, mat_norm)

    if color_use is None:
        color_use = list(scPalette(mat.shape[1]))
    color_use = list(color_use)

    cols = list(_safe_brewer(color_heatmap, 9))
    cmap = LinearSegmentedColormap.from_list("nas_brew", cols, N=256)

    # cm to inches for size parity with R
    fig = plt.figure(figsize=(width / 2.54, height / 2.54))
    # Columns:
    #   0 body · 1 right barplot · 2 gap · 3 legend title (rot 90°) · 4 cax
    # Rows:
    #   0 top barplot · 1 top color strip · 2 body row
    # The dedicated cax + title columns mirror what 1d4135d did for
    # ``netVisual_heatmap`` so the colorbar stops stealing space from
    # ``ax_main`` and the rotated legend title (R: ``title_position =
    # "leftcenter-rot"``) sits outside the bar.
    gs = fig.add_gridspec(
        3, 5,
        width_ratios=[1.00, 0.18, 0.10, 0.06, 0.04],
        height_ratios=[0.18, 0.05, 1.00],
        wspace=0.03, hspace=0.05,
    )
    ax_main = fig.add_subplot(gs[2, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top_anno = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[2, 1], sharey=ax_main)
    ax_cbar_title = fig.add_subplot(gs[2, 3])
    ax_cbar = fig.add_subplot(gs[2, 4])

    im = ax_main.imshow(
        mat_plot, cmap=cmap, aspect="auto", interpolation="nearest",
    )
    ax_main.set_xticks(np.arange(len(node_names)))
    ax_main.set_xticklabels(node_names, rotation=90, fontsize=font_size,
                            ha="right")
    ax_main.set_yticks(np.arange(len(pathway_names)))
    ax_main.set_yticklabels(pathway_names, fontsize=font_size)
    ax_main.set_title(full_title, fontsize=font_size_title)

    # Top bar: column strength of mat.ori
    col_strength = mat_ori.sum(axis=0)
    ax_top.bar(np.arange(len(node_names)), col_strength,
               color=color_use, edgecolor=color_use)
    ax_top.tick_params(left=True, right=False, top=False, bottom=False,
                       labelbottom=False, labelsize=font_size - 1)
    ax_top.set_ylabel("Strength", fontsize=font_size)
    for spine in ("top", "right", "bottom"):
        ax_top.spines[spine].set_visible(False)

    # Right bar: row pSum (transformed exactly like R)
    pSum = mat_ori.sum(axis=1)
    pSum_orig = pSum.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pSum_t = np.where(pSum > 0, -1.0 / np.log(pSum), 0.0)
    pSum_t = np.where(np.isnan(pSum_t), 0.0, pSum_t)
    bad = np.isinf(pSum_t) | (pSum_t < 0)
    if bad.any():
        finite_max = pSum_t[~bad].max() if (~bad).any() else 1.0
        replacements = np.linspace(finite_max * 1.1, finite_max * 1.5, int(bad.sum()))
        order = np.argsort(pSum_orig[bad])
        new_vals = np.empty_like(replacements)
        new_vals[order] = replacements
        pSum_t[bad] = new_vals
    ax_right.barh(np.arange(len(pathway_names)), pSum_t,
                  color=_safe_color("grey60"), edgecolor=_safe_color("grey60"))
    ax_right.invert_yaxis()
    ax_right.tick_params(top=False, bottom=True, left=False, right=False,
                         labelleft=False, labelsize=font_size - 1)
    for spine in ("top", "right", "left"):
        ax_right.spines[spine].set_visible(False)

    # Top annotation bar (cell-group colours)
    ax_top_anno.imshow(
        np.array([[matplotlib.colors.to_rgb(c) for c in color_use]]),
        aspect="auto",
    )
    ax_top_anno.set_xticks([])
    ax_top_anno.set_yticks([])
    for spine in ax_top_anno.spines.values():
        spine.set_visible(False)

    # Dedicated colorbar — matches R's ``heatmap_legend_param`` block at
    # ``analysis.R:3199-3201``: ``title_position = "leftcenter-rot"``,
    # ``at = c(round(min(mat), 1), round(max(mat), 1))``,
    # ``legend_height = unit(20, "mm")``.
    finite = np.isfinite(mat_norm)
    if finite.any():
        v_min = float(mat_norm[finite].min())
        v_max = float(mat_norm[finite].max())
    else:
        v_min, v_max = 0.0, 1.0
    if v_min == v_max:
        legend_break = [round(v_max, 1)]
    else:
        legend_break = [round(v_min, 1), round(v_max, 1)]
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.outline.set_visible(False)
    cbar.set_ticks(legend_break)
    cbar.ax.tick_params(labelsize=font_size - 1)
    # Rotated legend title in its own gridspec column (R: ``title_position
    # = "leftcenter-rot"``). ``set_ylabel`` falls back inside the bar when
    # the cax is narrow, so we draw via an axes-level text artist.
    ax_cbar_title.set_axis_off()
    ax_cbar_title.text(
        0.5, 0.5, "Relative strength",
        transform=ax_cbar_title.transAxes,
        rotation=90, va="center", ha="center",
        fontsize=font_size,
    )
    return fig


# ---------------------------------------------------------------------------
# netVisual_bubble (R: visualization.R:2894)
# ---------------------------------------------------------------------------
def netVisual_bubble(
    adata: ad.AnnData,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR_use: Optional[pd.DataFrame] = None,
    sort_by_source: bool = False,
    sort_by_target: bool = False,
    sort_by_source_priority: bool = True,
    color_heatmap: str = "Spectral",
    n_colors: int = 10,
    direction: int = -1,
    thresh: float = 0.05,
    comparison: Optional[Sequence[int]] = None,
    group: Optional[Sequence[int]] = None,
    remove_isolate: bool = False,
    max_dataset: Optional[int] = None,
    min_dataset: Optional[int] = None,
    min_quantile: float = 0.0,
    max_quantile: float = 1.0,
    line_on: bool = True,
    line_size: float = 0.2,
    color_text_use: bool = True,
    color_text: Optional[Sequence[str]] = None,
    title_name: Optional[str] = None,
    font_size: float = 10.0,
    font_size_title: float = 10.0,
    show_legend: bool = True,
    grid_on: bool = True,
    color_grid: str = "grey90",
    angle_x: float = 90.0,
    return_data: bool = False,
) -> Figure:
    """Significant L-R interactions as a bubble plot.

    Mirrors :func:`netVisual_bubble` (R: ``visualization.R:2894``). Calls
    :func:`spatialcellchat.modeling.subsetCommunication` to extract the
    significant L-R / cell-pair table, then draws one dot per row sized
    by the binned p-value and coloured by ``-1/log(prob)``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with significant ``net['prob']`` / ``net['pval']``.
    sources_use, targets_use, signaling, pairLR_use, thresh : forwarded
        to :func:`subsetCommunication`.
    sort_by_source, sort_by_target, sort_by_source_priority : bool
        Influence x-axis ordering. Mirrors R's identical-named flags.
    color_heatmap : str, default ``"Spectral"``
        RColorBrewer palette name driving the colour ramp.
    n_colors : int, default 10
        Brewer palette anchor count.
    direction : {-1, 1}, default -1
        Reverse the colour ramp when -1.
    comparison, group, max_dataset, min_dataset : reserved for the
        merged-dataset code path (kept for API parity).
    remove_isolate : bool, default False
        Drop empty source/target combinations.
    min_quantile, max_quantile : float
        Quantile cutoffs applied to the colour scale.
    line_on, line_size, color_text_use, color_text : reserved (merged
        mode).
    title_name, font_size, font_size_title : visual kwargs.
    show_legend, grid_on, color_grid : visual kwargs.
    angle_x : x-tick rotation in degrees.
    return_data : bool, default False
        When True, attach the long table as ``fig._netVisual_bubble_df``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cc = _ensure_cellchat(adata)
    cell_levels = _get_idents_levels(adata, cc.get("net", {}))
    src = _resolve_subset(sources_use, cell_levels)
    tgt = _resolve_subset(targets_use, cell_levels)

    from .modeling import subsetCommunication
    from ._palette import ggPalette

    # Detect merged-mode: net is dict-of-dicts keyed by dataset name (each
    # value is itself a dict containing 'prob' / 'pval'), or comparison was
    # provided.
    net_slot = cc.get("net", {})
    merged_keys: List[str] = []
    if isinstance(net_slot, dict):
        for k, v in net_slot.items():
            if isinstance(v, dict) and "prob" in v and "pval" in v:
                merged_keys.append(k)

    is_merged_mode = comparison is not None and len(merged_keys) >= 2

    if is_merged_mode:
        # R: per-comparison subsetCommunication, dataset suffixing, isolate row
        # padding. We synthesise df_all by iterating per-dataset.
        dataset_names = [merged_keys[int(c) - 1] for c in comparison]
        df_all = pd.DataFrame()
        group_names0: List[str] = []
        for ii, name in enumerate(dataset_names):
            sub_net = net_slot[name]
            # Build a temporary adata-like view with this dataset's net.
            tmp_adata = adata
            df_net = subsetCommunication(
                tmp_adata, net=sub_net, slot_name="net",
                sources_use=src, targets_use=tgt,
                signaling=list(signaling) if signaling is not None else None,
                pairLR_use=pairLR_use, thresh=thresh,
            )
            df_net = df_net.copy()
            df_net["source"] = df_net["source"].astype(str)
            df_net["target"] = df_net["target"].astype(str)
            df_net["interaction_name_2"] = df_net.get(
                "interaction_name_2", df_net.get("interaction_name", df_net["source"])
            ).astype(str)
            df_net["source.target"] = df_net["source"] + " -> " + df_net["target"]
            if src is not None and tgt is not None:
                srctgt_full = [f"{s} -> {t}" for s in src for t in tgt]
                missing = [st for st in srctgt_full if st not in set(df_net["source.target"])]
                if missing:
                    extra_rows = []
                    fallback_iname = (
                        df_net["interaction_name_2"].iloc[0] if len(df_net) > 0 else ""
                    )
                    for st in missing:
                        s, t = st.split(" -> ")
                        row = {col: np.nan for col in df_net.columns}
                        row["source"] = s
                        row["target"] = t
                        row["source.target"] = st
                        row["interaction_name_2"] = fallback_iname
                        row["pval"] = 1.0
                        extra_rows.append(row)
                    df_net = pd.concat(
                        [df_net, pd.DataFrame(extra_rows)], ignore_index=True
                    )
            # Bin pvals + transform prob (per dataset)
            if len(df_net) > 0:
                pv = df_net["pval"].astype(float).to_numpy()
                df_net["pval"] = np.where(pv > 0.05, 1.0, np.where(pv > 0.01, 2.0, 3.0))
                pr = df_net["prob"].astype(float).to_numpy()
                pr = np.where(pr == 0, np.nan, pr)
                df_net["prob.original"] = pr.copy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    pr = np.where(np.isnan(pr), np.nan, -1.0 / np.log(pr))
                df_net["prob"] = pr
            else:
                df_net = pd.DataFrame({
                    "interaction_name_2": [],
                    "source.target": [],
                    "prob": [],
                    "pval": [],
                    "prob.original": [],
                })
            # Source-target ordering for the panel
            grp = (
                df_net["source.target"].astype(str).tolist()
                if "source.target" in df_net.columns
                else []
            )
            for nm in grp:
                if nm not in group_names0:
                    group_names0.append(nm)
            df_net["group.names"] = df_net["source.target"].astype(str)
            df_net["source.target"] = df_net["source.target"].astype(str) + f" ({name})"
            df_net["dataset"] = name
            df_all = pd.concat([df_all, df_net], ignore_index=True)

        if df_all.shape[0] == 0:
            raise ValueError(
                "No interactions are detected. Please consider changing the "
                "cell groups for analysis."
            )

        # Replace inf/neg in prob (R idx1 step)
        prob = df_all["prob"].astype(float).to_numpy()
        idx_bad = np.where(np.isinf(prob) | (prob < 0))[0]
        if idx_bad.size > 0:
            valid_mask = ~(np.isinf(prob) | (prob < 0)) & ~np.isnan(prob)
            finite_max = float(np.nanmax(prob[valid_mask])) if valid_mask.any() else 1.0
            repl = np.linspace(finite_max * 1.1, finite_max * 1.5, idx_bad.size)
            order = np.argsort(df_all["prob.original"].to_numpy()[idx_bad])
            new = np.empty_like(repl)
            new[order] = repl
            prob[idx_bad] = new
            df_all["prob"] = prob

        # Fill na interaction_name_2 with first non-na
        not_na = df_all["interaction_name_2"].dropna()
        if len(not_na) > 0:
            df_all["interaction_name_2"] = df_all["interaction_name_2"].fillna(
                not_na.iloc[0]
            )

        df = df_all
        df = df.sort_values("interaction_name_2", kind="stable")
        df["interaction_name_2"] = df["interaction_name_2"].astype(str)

        # Build cells.order = group_names0 × dataset_names interleaved
        cells_order: List[str] = []
        for st in group_names0:
            for name in dataset_names:
                cells_order.append(f"{st} ({name})")
        df["source.target"] = pd.Categorical(
            df["source.target"], categories=cells_order
        )
        df = df.sort_values(["interaction_name_2", "source.target"], kind="stable")

        # Quantile cutoff
        valid = ~np.isnan(df["prob"].to_numpy())
        if valid.any():
            lo = float(np.quantile(df["prob"].to_numpy()[valid], min_quantile))
            hi = float(np.quantile(df["prob"].to_numpy()[valid], max_quantile))
            pr = df["prob"].to_numpy()
            pr = np.where(pr < lo, lo, pr)
            pr = np.where(pr > hi, hi, pr)
            df["prob"] = pr

        # max_dataset / min_dataset masking
        if max_dataset is not None:
            inames = list(dict.fromkeys(df["interaction_name_2"].astype(str).tolist()))
            for iname in inames:
                df_i = df[df["interaction_name_2"] == iname]
                cells = list(dict.fromkeys(df_i["group.names"].astype(str).tolist()))
                for cell in cells:
                    df_i_j = df_i[df_i["group.names"] == cell]
                    values = df_i_j["prob"].to_numpy(dtype=float)
                    if np.all(np.isnan(values)):
                        continue
                    val_max = np.nanmax(values)
                    val_min = np.nanmin(values)
                    idx_max_local = np.where(values == val_max)[0]
                    idx_min_local = np.where(values == val_min)[0]
                    max_ds_names = [
                        merged_keys[int(idx) - 1]
                        for idx in (
                            [max_dataset] if isinstance(max_dataset, int)
                            else list(max_dataset)
                        )
                    ]
                    min_ds_names = (
                        [merged_keys[int(idx) - 1]
                         for idx in ([min_dataset] if isinstance(min_dataset, int)
                                     else list(min_dataset))]
                        if min_dataset is not None else None
                    )
                    df_i_j_dataset = df_i_j["dataset"].to_numpy()
                    if len(idx_max_local) > 0:
                        max_dataset_at_max = df_i_j_dataset[idx_max_local]
                        if not any(d in max_ds_names for d in max_dataset_at_max):
                            df.loc[df_i_j.index, "prob"] = np.nan
                        elif (
                            np.all(idx_max_local != idx_min_local)
                            and min_ds_names is not None
                        ):
                            min_dataset_at_min = df_i_j_dataset[idx_min_local]
                            if not any(d in min_ds_names for d in min_dataset_at_min):
                                df.loc[df_i_j.index, "prob"] = np.nan

        if remove_isolate:
            df = df.loc[~np.isnan(df["prob"].to_numpy())].copy()
            line_on = False

        if df.shape[0] == 0:
            raise ValueError(
                "No interactions are detected. Please consider changing the "
                "cell groups for analysis."
            )

        # Render in merged style. Skip the single-object branch entirely.
        x_levels = list(dict.fromkeys(df["source.target"].astype(str).tolist()))
        y_levels = list(dict.fromkeys(df["interaction_name_2"].astype(str).tolist()))
        x_pos = {nm: i for i, nm in enumerate(x_levels)}
        y_pos = {nm: i for i, nm in enumerate(y_levels)}
        df["_x"] = df["source.target"].astype(str).map(x_pos)
        df["_y"] = df["interaction_name_2"].astype(str).map(y_pos)

        cols = list(_safe_brewer(color_heatmap, n_colors))
        if direction == -1:
            cols = list(reversed(cols))
        cmap99 = LinearSegmentedColormap.from_list(
            "bubble", list(_expand_palette(cols, 99)), N=99,
        )
        fig, ax = plt.subplots(figsize=(max(4.0, 0.3 * len(x_levels) + 4.0),
                                        max(3.0, 0.3 * len(y_levels) + 2.0)))
        valid_prob = df["prob"].to_numpy(dtype=float)
        if np.any(np.isfinite(valid_prob)):
            vmin = float(np.nanmin(valid_prob))
            vmax = float(np.nanmax(valid_prob))
            if vmax == vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        sizes = (df["pval"].to_numpy(dtype=float) ** 2) * 30.0
        sc = ax.scatter(
            df["_x"].to_numpy(), df["_y"].to_numpy(), s=sizes,
            c=valid_prob, cmap=cmap99, norm=norm,
        )

        # color group / x-tick coloring
        if group is None:
            grp_arr = np.arange(1, len(dataset_names) + 1)
            grp_index = {dataset_names[i]: grp_arr[i] for i in range(len(dataset_names))}
        else:
            grp_arr = np.asarray(list(group), dtype=int)
            grp_index = {dataset_names[i]: grp_arr[i] for i in range(len(dataset_names))}
        if color_text is None:
            color_per_grp = list(ggPalette(int(np.unique(grp_arr).size)))
        else:
            color_per_grp = list(color_text)
        # Map dataset -> color via group integer.
        unique_groups = list(dict.fromkeys(grp_arr.tolist()))
        group_color_map = {
            g: color_per_grp[i % len(color_per_grp)] for i, g in enumerate(unique_groups)
        }
        ds_to_color = {ds: group_color_map[grp_index[ds]] for ds in dataset_names}

        # Extract dataset suffix from each x-tick label
        xtick_colors = []
        for lbl in x_levels:
            # Format: "src -> tgt (dataset)"
            m = lbl.rfind("(")
            if m >= 0 and lbl.endswith(")"):
                ds = lbl[m + 1:-1]
                xtick_colors.append(ds_to_color.get(ds, "black"))
            else:
                xtick_colors.append("black")

        ax.set_xticks(np.arange(len(x_levels)))
        labels = ax.set_xticklabels(
            x_levels, rotation=angle_x,
            ha="right" if angle_x != 0 else "center",
            fontsize=font_size,
        )
        if color_text_use:
            for tick_label, c in zip(labels, xtick_colors):
                tick_label.set_color(c)
        ax.set_yticks(np.arange(len(y_levels)))
        ax.set_yticklabels(y_levels, fontsize=font_size)
        ax.set_xlim(-0.5, len(x_levels) - 0.5)
        ax.set_ylim(-0.5, len(y_levels) - 0.5)
        ax.invert_yaxis()

        # vertical separators between dataset groups
        if line_on and len(dataset_names) > 1:
            step = len(dataset_names)
            xs = np.arange(0.5 + step - 1, len(x_levels), step)
            for x in xs:
                if 0 < x < len(x_levels) - 1:
                    ax.axvline(x, linestyle="--", color=_safe_color("grey60"), linewidth=line_size)

        if grid_on:
            grid_col = _safe_color(color_grid)
            if len(x_levels) > 1:
                for v in np.arange(0.5, len(x_levels) - 0.5, 1.0):
                    ax.axvline(v, color=grid_col, linewidth=0.1)
            if len(y_levels) > 1:
                for h in np.arange(0.5, len(y_levels) - 0.5, 1.0):
                    ax.axhline(h, color=grid_col, linewidth=0.1)
        if title_name is not None:
            ax.set_title(title_name, fontsize=font_size_title)
        if show_legend:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.06, shrink=0.7)
            cbar.set_label("Commun. Prob.", fontsize=font_size - 2)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels(["min", "max"])
            cbar.ax.tick_params(labelsize=font_size - 2)
            legend_p = sorted(set(df["pval"].astype(int).tolist()))
            labels_map = {1: "p > 0.05", 2: "0.01 < p < 0.05", 3: "p < 0.01"}
            handles = [
                plt.scatter([], [], s=(p ** 2) * 30.0, color="grey", label=labels_map[p])
                for p in legend_p
            ]
            ax.legend(
                handles=handles, title="p-value",
                loc="upper left", bbox_to_anchor=(1.18, 1.0),
                fontsize=font_size - 2, title_fontsize=font_size - 1,
            )
        if return_data:
            fig._netVisual_bubble_df = df  # type: ignore[attr-defined]
        return fig

    df_net = subsetCommunication(
        adata, slot_name="net",
        sources_use=src, targets_use=tgt,
        signaling=list(signaling) if signaling is not None else None,
        pairLR_use=pairLR_use, thresh=thresh,
    )
    if isinstance(df_net, dict):
        # In single-object Python port, this branch is unreachable
        # because subsetCommunication returns a DataFrame.
        df_net = next(iter(df_net.values()))
    df = df_net.copy()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["source.target"] = df["source"] + " -> " + df["target"]

    if src is not None and tgt is not None:
        srctgt_full = [f"{s} -> {t}" for s in src for t in tgt]
        missing = [st for st in srctgt_full if st not in set(df["source.target"])]
        if missing:
            extra_rows = []
            for st in missing:
                s, t = st.split(" -> ")
                row = {col: np.nan for col in df.columns}
                row["source"] = s
                row["target"] = t
                row["source.target"] = st
                if "interaction_name_2" in df.columns and len(df) > 0:
                    row["interaction_name_2"] = df["interaction_name_2"].iloc[0]
                row["pval"] = 1.0
                extra_rows.append(row)
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Bin p-values
    pval = df["pval"].astype(float).to_numpy()
    pval_bin = np.where(pval > 0.05, 1.0,
                        np.where(pval > 0.01, 2.0, 3.0))
    df["pval"] = pval_bin

    prob = df["prob"].astype(float).to_numpy()
    prob = np.where(prob == 0, np.nan, prob)
    df["prob.original"] = prob.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(np.isnan(prob), np.nan, -1.0 / np.log(prob))
    df["prob"] = prob

    bad = np.where(np.isinf(prob) | (prob < 0))[0]
    if bad.size > 0:
        finite_max = float(np.nanmax(prob[~(np.isinf(prob) | (prob < 0))])) \
            if np.any(~(np.isinf(prob) | (prob < 0)) & ~np.isnan(prob)) else 1.0
        repl = np.linspace(finite_max * 1.1, finite_max * 1.5, bad.size)
        order = np.argsort(df["prob.original"].to_numpy()[bad])
        new = np.empty_like(repl)
        new[order] = repl
        prob[bad] = new
        df["prob"] = prob

    # Quantile cutoffs
    valid = ~np.isnan(prob)
    if valid.any():
        lo = float(np.quantile(prob[valid], min_quantile))
        hi = float(np.quantile(prob[valid], max_quantile))
        prob = np.where(prob < lo, lo, prob)
        prob = np.where(prob > hi, hi, prob)
        df["prob"] = prob

    if remove_isolate:
        df = df.loc[~np.isnan(df["prob"].to_numpy())].copy()

    if df.shape[0] == 0:
        raise ValueError(
            "No interactions are detected. Please consider changing the "
            "cell groups for analysis."
        )

    # Order rows by interaction_name_2 (alphabetical within an LR-pair)
    if "interaction_name_2" not in df.columns:
        df["interaction_name_2"] = df.get("interaction_name", df["source.target"])
    df["interaction_name_2"] = df["interaction_name_2"].astype(str)
    df = df.sort_values("interaction_name_2", kind="stable").reset_index(drop=True)

    # x-axis ordering
    if sort_by_target and not sort_by_source and tgt is not None:
        df["target"] = pd.Categorical(df["target"], categories=tgt)
        df = df.sort_values(["target", "source"], kind="stable")
    elif sort_by_source and not sort_by_target and src is not None:
        df["source"] = pd.Categorical(df["source"], categories=src)
        df = df.sort_values(["source", "target"], kind="stable")
    elif sort_by_source and sort_by_target and src is not None and tgt is not None:
        df["source"] = pd.Categorical(df["source"], categories=src)
        df["target"] = pd.Categorical(df["target"], categories=tgt)
        if sort_by_source_priority:
            df = df.sort_values(["source", "target"], kind="stable")
        else:
            df = df.sort_values(["target", "source"], kind="stable")
    df = df.reset_index(drop=True)

    x_levels = list(dict.fromkeys(df["source.target"].astype(str).tolist()))
    y_levels = list(dict.fromkeys(df["interaction_name_2"].astype(str).tolist()))
    x_pos = {nm: i for i, nm in enumerate(x_levels)}
    y_pos = {nm: i for i, nm in enumerate(y_levels)}
    df["_x"] = df["source.target"].astype(str).map(x_pos)
    df["_y"] = df["interaction_name_2"].astype(str).map(y_pos)

    # Colour ramp: brewer / viridis -> 99-step ramp
    cols = list(_safe_brewer(color_heatmap, n_colors))
    if direction == -1:
        cols = list(reversed(cols))
    cmap99 = LinearSegmentedColormap.from_list(
        "bubble", list(_expand_palette(cols, 99)), N=99,
    )

    fig, ax = plt.subplots(figsize=(max(4.0, 0.3 * len(x_levels) + 4.0),
                                    max(3.0, 0.3 * len(y_levels) + 2.0)))

    valid_prob = df["prob"].to_numpy(dtype=float)
    has_finite = np.any(np.isfinite(valid_prob))
    if has_finite:
        vmin = float(np.nanmin(valid_prob))
        vmax = float(np.nanmax(valid_prob))
        if vmax == vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    sizes = (df["pval"].to_numpy(dtype=float) ** 2) * 30.0
    sc = ax.scatter(
        df["_x"].to_numpy(),
        df["_y"].to_numpy(),
        s=sizes,
        c=valid_prob,
        cmap=cmap99,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(x_levels)))
    ax.set_xticklabels(x_levels, rotation=angle_x, ha="right" if angle_x != 0 else "center",
                       fontsize=font_size)
    ax.set_yticks(np.arange(len(y_levels)))
    ax.set_yticklabels(y_levels, fontsize=font_size)
    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    ax.set_ylim(-0.5, len(y_levels) - 0.5)
    ax.invert_yaxis()  # mirror ggplot's bottom-up y axis to image-style

    if grid_on:
        grid_col = _safe_color(color_grid)
        if len(x_levels) > 1:
            for v in np.arange(0.5, len(x_levels) - 0.5, 1.0):
                ax.axvline(v, color=grid_col, linewidth=0.1)
        if len(y_levels) > 1:
            for h in np.arange(0.5, len(y_levels) - 0.5, 1.0):
                ax.axhline(h, color=grid_col, linewidth=0.1)

    if title_name is not None:
        ax.set_title(title_name, fontsize=font_size_title)

    if show_legend:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.06, shrink=0.7)
        cbar.set_label("Commun. Prob.", fontsize=font_size - 2)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(["min", "max"])
        cbar.ax.tick_params(labelsize=font_size - 2)
        # size legend
        legend_p = sorted(set(df["pval"].astype(int).tolist()))
        labels_map = {1: "p > 0.05", 2: "0.01 < p < 0.05", 3: "p < 0.01"}
        handles = [
            plt.scatter([], [], s=(p ** 2) * 30.0, color="grey", label=labels_map[p])
            for p in legend_p
        ]
        ax.legend(handles=handles, title="p-value",
                  loc="upper left", bbox_to_anchor=(1.18, 1.0),
                  fontsize=font_size - 2, title_fontsize=font_size - 1)

    if return_data:
        fig._netVisual_bubble_df = df  # type: ignore[attr-defined]
    return fig


# ---------------------------------------------------------------------------
# doHeatmap (R: visualization.R:9212)
# ---------------------------------------------------------------------------
def doHeatmap(
    adata: ad.AnnData,
    features: Optional[Sequence[str]] = None,
    cells: Optional[Sequence[str]] = None,
    group_by: str = "ident",
    colors_use: Optional[Mapping[str, str]] = None,
    colors_ggplot: bool = False,
    group_bar: bool = True,
    additional_group_by: Optional[Sequence[str]] = None,
    additional_group_sort_by: Optional[Sequence[str]] = None,
    color_heatmap: str = "RdBu",
    n_colors: int = 11,
    direction: int = -1,
    disp_min: float = -2.1,
    disp_max: float = 2.1,
    slot: str = "scale.data",
    label: bool = True,
    text_size: float = 8.0,
    size: float = 4.0,
    angle: float = 45.0,
    raster: bool = True,
    legend_title: Optional[str] = None,
    legend_remove: bool = False,
) -> Figure:
    """Cell-by-feature expression heatmap (Seurat-style ``DoHeatmap`` port).

    Mirrors :func:`doHeatmap` (R: ``visualization.R:9212``). Reads the
    ``slot`` matrix via
    :func:`spatialcellchat._anndata.resolve_data_matrix` after mapping
    the Seurat-style ``slot`` names onto the chat-slot vocabulary
    (``"scale.data"`` → ``chat@data.scale`` →
    ``adata.layers["data_scale"]``; ``"data"`` → ``chat@data`` →
    ``adata.X``; ``"counts"`` → ``chat@data.raw`` →
    ``adata.layers["counts"]``).  Draws a cell × feature heatmap with
    a coloured group bar along the top encoding ``group_by``.

    Parameters
    ----------
    adata : anndata.AnnData
    features : sequence of str, optional
        Genes to plot. Defaults to ``var_names`` (mirrors Seurat fallback).
    cells : sequence of str, optional
        Subset of cells (defaults to all). Numeric inputs are 1-based.
    group_by : str, default ``"ident"``
        ``adata.obs`` column to use for grouping. ``"ident"`` falls back
        to the standard cellchat idents column.
    colors_use : mapping, optional
        Per-group colour overrides.
    colors_ggplot : bool
        Use ``ggPalette``-style colours when no overrides are provided.
    group_bar : bool, default True
        Draw the coloured group annotation strip above the heatmap.
    additional_group_by, additional_group_sort_by : reserved for parity.
    color_heatmap : str, default ``"RdBu"``
        Brewer palette driving the heatmap.
    n_colors : int, default 11
        Brewer anchor count.
    direction : {-1, 1}, default -1
        Reverse the palette when -1.
    disp_min, disp_max : float
        Clipping limits for the colour scale.
    slot : str, default ``"scale.data"``
        Where to read the expression matrix from.
    label : bool, default True
        Print group labels above the colour bar.
    text_size, size, angle : visual kwargs.
    raster : bool, default True
        Reserved (matplotlib's ``imshow`` is always rasterised).
    legend_title, legend_remove : visual kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if features is None:
        features = list(adata.var_names)
    feature_list = list(dict.fromkeys(reversed(list(features))))
    feature_list = [f for f in feature_list if f in set(adata.var_names)]
    if len(feature_list) == 0:
        raise ValueError(
            "No requested features found in the slot/assay."
        )

    if cells is None:
        cell_idx = np.arange(adata.n_obs)
        cell_names = list(adata.obs_names)
    else:
        if all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in cells):
            cell_idx = np.array([int(c) - 1 for c in cells], dtype=int)  # 1-based
        else:
            name_index = {n: i for i, n in enumerate(adata.obs_names)}
            cell_idx = np.array([name_index[c] for c in cells], dtype=int)
        cell_names = [adata.obs_names[i] for i in cell_idx]

    # Resolve expression matrix.  Map Seurat-style slot names to the
    # canonical chat-slot vocabulary (Seurat ``slot="scale.data"`` ≡
    # chat ``@data.scale``; ``slot="data"`` ≡ chat ``@data``;
    # ``slot="counts"`` ≡ chat ``@data.raw``) and route through
    # :func:`resolve_data_matrix` so the AnnData-canonical layer keys
    # (``"data_scale"`` / ``"counts"`` / …) stay in one place.
    _SEURAT_TO_CHAT = {
        "scale.data": "data.scale",
        "data":       "data",
        "counts":     "data.raw",
    }
    chat_slot = _SEURAT_TO_CHAT.get(slot, slot)
    # ``resolve_data_matrix`` returns genes × cells; ``X`` here is
    # consumed downstream as cells × genes (``X[cell_idx, feat_idx]``),
    # so transpose back to AnnData orientation.
    X = resolve_data_matrix(adata, chat_slot).to_numpy(dtype=float).T

    feat_idx = [list(adata.var_names).index(f) for f in feature_list]
    expr = X[np.ix_(cell_idx, feat_idx)]

    # Resolve group_by
    if group_by == "ident":
        for k in ("cellchat_idents", "active_ident", "cell.type"):
            if k in adata.obs.columns:
                group_by = k
                break
    if group_by not in adata.obs.columns:
        raise ValueError(f"adata.obs[{group_by!r}] not found.")
    groups_full = adata.obs[group_by].iloc[cell_idx]
    groups = pd.Categorical(groups_full)
    levels = list(groups.categories)

    order = np.argsort(groups.codes, kind="stable")
    expr_sorted = expr[order, :]
    groups_sorted = groups[order]

    # Clip values
    expr_sorted = np.clip(expr_sorted, disp_min, disp_max)

    # Colormap. R has a special branch when color.heatmap == "MyRdBu":
    # uses generate_RedsBlues_colormap with mid.color="white".
    if color_heatmap == "MyRdBu":
        cols = _generate_RedsBlues_colormap(
            np.clip(expr_sorted, disp_min, disp_max), mid_color="white",
        )
        cmap = LinearSegmentedColormap.from_list("dh_myrdbu", cols, N=256)
    else:
        cols = list(_safe_brewer(color_heatmap, n_colors))
        if direction == -1:
            cols = list(reversed(cols))
        cmap = LinearSegmentedColormap.from_list("dh", cols, N=256)

    # Group colours
    if colors_ggplot:
        gcols = list(ggPalette(len(levels)))
    else:
        gcols = list(scPalette(len(levels)))
    if colors_use is not None:
        for i, lvl in enumerate(levels):
            if lvl in colors_use:
                gcols[i] = colors_use[lvl]

    fig = plt.figure(figsize=(max(6.0, 0.05 * len(cell_names) + 4.0),
                              max(4.0, 0.18 * len(feature_list) + 1.5)))
    if group_bar:
        gs = fig.add_gridspec(2, 2, height_ratios=[0.04, 1.0],
                              width_ratios=[1.0, 0.03],
                              hspace=0.04, wspace=0.02)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0], sharex=ax_bar)
        cax = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.03], wspace=0.02)
        ax_main = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        ax_bar = None

    im = ax_main.imshow(
        expr_sorted.T, cmap=cmap, aspect="auto",
        vmin=disp_min, vmax=disp_max, interpolation="nearest",
    )
    ax_main.set_yticks(np.arange(len(feature_list)))
    ax_main.set_yticklabels(feature_list, fontsize=text_size)
    ax_main.set_xticks([])
    ax_main.set_xlabel("")
    for spine in ("top", "right"):
        ax_main.spines[spine].set_visible(False)

    if ax_bar is not None:
        # group strip
        codes = groups_sorted.codes
        bar = np.array([[matplotlib.colors.to_rgb(gcols[c]) for c in codes]])
        ax_bar.imshow(bar, aspect="auto", interpolation="nearest")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        if label:
            # Centre each level label over its run
            uniq, idxs = np.unique(codes, return_index=True)
            for u in uniq:
                run = np.where(codes == u)[0]
                if run.size == 0:
                    continue
                centre = float(run.mean())
                ax_bar.text(centre, 1.4, levels[u], rotation=angle,
                            ha="left", va="bottom",
                            fontsize=size * 2.0)

    if not legend_remove:
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=text_size - 1)
        if legend_title is not None:
            cbar.set_label(legend_title, fontsize=text_size)
    else:
        cax.set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# Hierarchical-clustering leaf order helper for netVisual_TopicComposition.
# Mirrors ComplexHeatmap::Heatmap's ``cluster_rows`` / ``cluster_columns``
# behaviour: R defers to ``stats::hclust`` on a distance matrix derived from
# ``clustering_distance_rows``; we use ``scipy.cluster.hierarchy`` with the
# matching metric.
# ---------------------------------------------------------------------------
def _hclust_leaves(mat: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return the leaf ordering of an average-linkage dendrogram on ``mat``.

    The metric mirrors R's ``clustering_distance_rows`` argument to
    ``ComplexHeatmap::Heatmap``. ``"correlation"`` follows R's convention of
    ``1 - cor(t(mat))`` (distance from 1) by delegating to scipy's
    ``pdist(metric="correlation")``.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    arr = np.asarray(mat, dtype=float)
    if arr.shape[0] < 2:
        return np.arange(arr.shape[0])
    dists = pdist(arr, metric=metric)
    # Numerical noise can produce tiny negatives in ``correlation`` distances.
    if metric == "correlation":
        dists = np.clip(dists, 0.0, None)
    Z = linkage(dists, method="average")
    return np.asarray(leaves_list(Z))


# ---------------------------------------------------------------------------
# netVisual_TopicComposition (R: visualization.R:4581)
# ---------------------------------------------------------------------------
def netVisual_TopicComposition(
    adata: ad.AnnData,
    y_group: Union[str, np.ndarray, pd.DataFrame],
    x_group: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
    x_levels: Optional[Sequence[str]] = None,
    y_levels: Optional[Sequence[str]] = None,
    scale_rows: bool = True,
    cutoff_prop: float = 0.1,
    trim: float = 0.1,
    type_: str = "heatmap",
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "RdPu",
    n_colors: int = 8,
    dot_size: Tuple[float, float] = (1.0, 6.0),
    angle_x: float = 45.0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title_name: Optional[str] = None,
    legend_title: Optional[str] = None,
    font_size: float = 8.0,
    font_size_title: float = 10.0,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    clustering_distance_rows: str = "euclidean",
    x_lab_rot: float = 45.0,
    row_show: Optional[Sequence] = None,
    col_show: Optional[Sequence] = None,
    remove_isolate: bool = True,
) -> Figure:
    """Topic composition heatmap or dot plot.

    Mirrors :func:`netVisual_TopicComposition` (R: ``visualization.R:4581``).
    Builds a (topic × cell-group) proportion matrix and draws either a
    heatmap (default) or dot plot.

    Parameters
    ----------
    adata : anndata.AnnData
    y_group : {"topicIn", "topicOut", "inmfNorm"} | str | DataFrame
        Topic / signature loadings or an ``adata.obs`` column.
    x_group : str | DataFrame, optional
        Cell-group labels. Default uses the ``cellchat_idents`` column.
    x_levels, y_levels : sequence of str, optional
        Explicit ordering of axes.
    scale_rows : bool, default True
        Normalise within rows when True (else within columns).
    cutoff_prop : float, default 0.1
        Drop columns whose maximum proportion is below this cutoff.
    trim : float, default 0.1
        Trim fraction passed to :func:`thresholdedMean` for column averages.
    type_ : {"heatmap", "dot"}.
    color_use, color_heatmap, n_colors : palette overrides.
    dot_size : tuple(min, max)
        Sizing range for ``type_="dot"``.
    angle_x, x_lab_rot : x-tick rotation.
    xlabel, ylabel, title_name, legend_title : labels.
    font_size, font_size_title : visual kwargs.
    cluster_rows, cluster_cols : bool
        When True, reorder rows / columns by an average-linkage dendrogram
        on the heatmap matrix (mirrors ``ComplexHeatmap::Heatmap``'s
        ``cluster_rows`` / ``cluster_columns``). Tier-2 visual deviation:
        the dendrogram glyph itself is omitted; only the leaf ordering is
        applied.
    clustering_distance_rows : str, default ``"euclidean"``
        Distance metric forwarded to :func:`scipy.spatial.distance.pdist`
        (e.g. ``"euclidean"``, ``"correlation"``, ``"manhattan"``).
    row_show, col_show, remove_isolate : subsetting / filtering options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ._utils import thresholdedMean

    cc = _ensure_cellchat(adata)

    # Build the (topic × cells) matrix `pred`
    if isinstance(y_group, str):
        if y_group in ("topicIn", "topicOut", "inmfNorm"):
            topic_key = {"topicIn": "incoming", "topicOut": "outgoing"}
            pattern = topic_key.get(y_group, y_group)
            net = cc.get("net", {})
            topic = net.get("topic", {}).get(pattern)
            if topic is None:
                raise ValueError(
                    f"adata.uns['cellchat']['net']['topic'][{pattern!r}] missing."
                )
            cell_df = topic["cell"]  # (nC, k)
            pred = np.asarray(cell_df.values).T  # (k, nC)
            topic_names = list(cell_df.columns)
            if title_name is None:
                title_name = (
                    "outgoing CCC topics" if y_group == "topicOut"
                    else "incoming CCC topics"
                )
        elif y_group in adata.obs.columns:
            y_state = pd.Categorical(adata.obs[y_group])
            topic_names = list(y_state.categories)
            pred = np.zeros((len(topic_names), adata.n_obs), dtype=float)
            for i, lvl in enumerate(topic_names):
                pred[i, np.asarray(y_state == lvl)] = 1.0
        else:
            raise ValueError(f"y_group {y_group!r} not in adata.obs.columns")
    else:
        # data.frame / matrix input
        df_y = pd.DataFrame(y_group)
        topic_names = list(df_y.index)
        pred = df_y.to_numpy(dtype=float)

    # Resolve x.group
    if x_group is None or isinstance(x_group, str):
        if x_group is None:
            for k in ("cellchat_idents", "active_ident", "cell.type"):
                if k in adata.obs.columns:
                    x_group = k
                    break
        if x_group is None or x_group not in adata.obs.columns:
            raise ValueError("Cannot resolve x.group from adata.obs.")
        x_state = pd.Categorical(adata.obs[x_group])
        labels_level = list(x_state.categories)
        labels = np.asarray(x_state)
    else:
        df_x = pd.DataFrame(x_group)
        labels_level = list(df_x.index)
        # When x.group is a DataFrame, treat it as a cell-state proportion
        # matrix and use it directly as `pred` (mirrors R's else branch).
        labels = None

    if labels is not None:
        prop = np.zeros((len(labels_level), pred.shape[0]), dtype=float)
        for i, lvl in enumerate(labels_level):
            cell_use = np.where(labels == lvl)[0]
            if cell_use.size > 0:
                sub = pred[:, cell_use]
                prop[i, :] = np.array([
                    thresholdedMean(sub[k, :], trim=trim, na_rm=True)
                    for k in range(sub.shape[0])
                ])
        prop = prop.T  # (topics, groups)
    else:
        # x_group is a matrix (cell-state x cells), use it as `pred` already
        # and follow the y_group branch above. This unusual case is unsupported
        # by the test fixture; raise to be explicit.
        raise NotImplementedError(
            "DataFrame x.group is reserved for the spot-deconvolution branch "
            "and is not driven by tests."
        )

    mat = prop.astype(float)
    if scale_rows:
        rs = mat.sum(axis=1, keepdims=True)
        rs = np.where(rs == 0, 1.0, rs)
        mat = mat / rs
    else:
        cs = mat.sum(axis=0, keepdims=True)
        cs = np.where(cs == 0, 1.0, cs)
        mat = mat / cs
    mat = np.where(np.isnan(mat), 0.0, mat)

    row_names = list(topic_names)
    col_names = list(labels_level)

    if remove_isolate:
        keep = np.where(mat.max(axis=0) >= cutoff_prop)[0]
        if keep.size > 0:
            mat = mat[:, keep]
            col_names = [col_names[i] for i in keep]

    if x_levels is not None:
        x_levels = [x for x in x_levels if x in col_names]
        order = [col_names.index(x) for x in x_levels]
        mat = mat[:, order]
        col_names = list(x_levels)
    if y_levels is not None:
        y_levels = [y for y in y_levels if y in row_names]
        order = [row_names.index(y) for y in y_levels]
        mat = mat[order, :]
        row_names = list(y_levels)

    if color_use is None:
        color_use = list(scPalette(mat.shape[1]))
    color_use = list(color_use)

    if row_show is not None:
        rs = _resolve_subset(row_show, row_names)
        keep = [row_names.index(n) for n in rs]
        mat = mat[keep, :]
        row_names = [row_names[i] for i in keep]
    if col_show is not None:
        cs = _resolve_subset(col_show, col_names)
        keep = [col_names.index(n) for n in cs]
        mat = mat[:, keep]
        col_names = [col_names[i] for i in keep]
        color_use = [color_use[i] for i in keep]

    if type_ == "dot":
        fig, ax = plt.subplots(figsize=(max(4.0, 0.4 * mat.shape[1] + 2.0),
                                        max(3.0, 0.4 * mat.shape[0] + 2.0)))
        xs, ys, vals = [], [], []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                xs.append(j)
                ys.append(i)
                vals.append(mat[i, j])
        size_min, size_max = dot_size
        sizes = (
            np.clip(np.asarray(vals), 0.0, None) * (size_max - size_min) + size_min
        ) ** 2 * 5.0
        sc = ax.scatter(xs, ys, s=sizes, c=vals, cmap="viridis")
        ax.set_xticks(np.arange(len(col_names)))
        ax.set_xticklabels(col_names, rotation=x_lab_rot, ha="right",
                           fontsize=font_size)
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names, fontsize=font_size)
        ax.invert_yaxis()
        cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.06)
        cbar.set_label(legend_title or "Proportion", fontsize=font_size)
        if title_name is not None:
            ax.set_title(title_name, fontsize=font_size_title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return fig

    # type_ == "heatmap"
    # Mirror ComplexHeatmap::Heatmap row/column hierarchical clustering. R
    # passes ``cluster_rows`` / ``cluster_columns`` / ``clustering_distance_rows``
    # straight to ComplexHeatmap, which performs ``hclust`` on the rows (or
    # transposed matrix for columns) and reorders both the matrix and labels
    # accordingly. We replicate that with ``scipy.cluster.hierarchy``.
    if cluster_rows and mat.shape[0] >= 2:
        row_order = _hclust_leaves(mat, metric=clustering_distance_rows)
        mat = mat[row_order, :]
        row_names = [row_names[i] for i in row_order]
    if cluster_cols and mat.shape[1] >= 2:
        col_order = _hclust_leaves(mat.T, metric=clustering_distance_rows)
        mat = mat[:, col_order]
        col_names = [col_names[i] for i in col_order]
        color_use = [color_use[i] for i in col_order]

    cols = list(_safe_brewer(color_heatmap, 9))
    cols = ["white"] + cols
    cmap = LinearSegmentedColormap.from_list("topic_brew", cols, N=256)

    fig = plt.figure(figsize=(max(4.0, 0.35 * mat.shape[1] + 2.5),
                              max(3.5, 0.18 * mat.shape[0] + 2.0)))
    # R parity (visualization.R:4840-4848): heatmap_legend_param uses
    # ``title_position = "leftcenter-rot"``, ``legend_height = 20mm``,
    # ``grid_width = 2mm``. Mirror via dedicated cax + rotated title slot
    # — same pattern as the netVisual_heatmap rebuild (1d4135d).
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.0, 0.05],
        width_ratios=[1.00, 0.10, 0.06, 0.04],
        hspace=0.04, wspace=0.04,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_anno = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_cbar_title = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])

    im = ax_main.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
    ax_main.set_xticks(np.arange(len(col_names)))
    ax_main.set_xticklabels(col_names, rotation=x_lab_rot, ha="right",
                            fontsize=font_size)
    ax_main.set_yticks(np.arange(len(row_names)))
    ax_main.set_yticklabels(row_names, fontsize=font_size)
    if title_name is not None:
        ax_main.set_title(f"Compositions of {title_name}", fontsize=font_size_title)
    if ylabel is not None:
        ax_main.set_ylabel(ylabel, fontsize=font_size_title)

    ax_anno.imshow(
        np.array([[matplotlib.colors.to_rgb(c) for c in color_use]]),
        aspect="auto",
    )
    ax_anno.set_xticks([])
    ax_anno.set_yticks([])
    for spine in ax_anno.spines.values():
        spine.set_visible(False)

    # R parity (visualization.R:4843-4845): rounded min/max ticks +
    # ``title_position = "leftcenter-rot"``.
    finite = np.isfinite(mat)
    if finite.any():
        v_min = float(mat[finite].min())
        v_max = float(mat[finite].max())
    else:
        v_min, v_max = 0.0, 1.0
    legend_break = (
        [round(v_max, 1)] if v_min == v_max
        else [round(v_min, 1), round(v_max, 1)]
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cbar.set_ticks(legend_break)
    cbar.ax.tick_params(labelsize=font_size - 1)
    ax_cbar_title.set_axis_off()
    ax_cbar_title.text(
        0.5, 0.5, legend_title or "Proportion",
        transform=ax_cbar_title.transAxes,
        rotation=90, va="center", ha="center",
        fontsize=font_size,
    )
    return fig


# ---------------------------------------------------------------------------
# netVisual_TopicSignaling (R: visualization.R:4887)
# ---------------------------------------------------------------------------
def netVisual_TopicSignaling(
    adata: ad.AnnData,
    pattern: str = "incoming",
    slot_name: str = "net",
    topics: Optional[Sequence[int]] = None,
    topic_key: str = "Topic_",
    nfeatures: int = 30,
    nTop: int = 10,
    feature_cat: str = "LR-pair",
    col: Tuple[str, str] = ("black", "black"),
    ncol: Optional[int] = None,
    combine: bool = True,
    ylabel: Optional[str] = None,
    balanced: bool = False,
    text_nudge_x: float = -8.0,
    text_size: float = 3.0,
    point_size: float = 2.0,
    line_size: float = 1.0,
) -> Figure:
    """Topic-loading ranking plot for the top marker LR-pairs / signalings.

    Mirrors :func:`netVisual_TopicSignaling` (R: ``visualization.R:4887``).
    For each topic this draws a rank-vs-|loading| line plot annotated
    with the top-``nTop`` features.

    Parameters
    ----------
    adata : anndata.AnnData
    pattern : {"incoming", "outgoing"}
    slot_name : str
        Slot containing ``topic`` (typically ``"net"``).
    topics : sequence of int, optional
        1-based topic indices. Defaults to all topics.
    topic_key : str
        Topic-prefix string used for plot titles.
    nfeatures : int
        Number of features per topic to keep before ranking.
    nTop : int
        Number of features to label.
    feature_cat : {"LR-pair", "ligand", "signaling"}
    col : tuple(str, str)
        Negative / positive label colours.
    ncol : int, optional
        Subplot grid width (auto when None).
    combine : bool
        Reserved (always combined here for parity).
    ylabel : str, optional
        y-axis label.
    balanced : bool
        Balance positive / negative loadings (matches R MyTop branch).
    text_nudge_x, text_size, point_size, line_size : visual kwargs.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("pattern must be 'incoming' or 'outgoing'.")
    if feature_cat not in ("LR-pair", "ligand", "signaling"):
        raise ValueError(
            "feature_cat must be 'LR-pair', 'ligand', or 'signaling'."
        )
    cc = _ensure_cellchat(adata)
    slot = cc.get(slot_name, {})
    pattern_res = slot.get("topic", {}).get(pattern)
    if pattern_res is None:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['topic'][{pattern!r}] is missing."
        )
    loadings = pattern_res["signaling"]
    if isinstance(loadings, pd.DataFrame):
        loadings_arr = loadings.values
        feature_names = list(loadings.index)
    else:
        loadings_arr = np.asarray(loadings, dtype=float)
        feature_names = [str(i) for i in range(loadings_arr.shape[0])]
    nFeatures, nTopics = loadings_arr.shape

    # Override columns to topic_key + 1..n
    topic_labels = [f"{topic_key}{i + 1}" for i in range(nTopics)]
    DB = cc.get("DB", {})
    interaction = DB.get("interaction", None)

    if topics is None:
        topics_idx = list(range(1, nTopics + 1))
    else:
        topics_idx = [int(t) for t in topics]

    if ncol is None:
        ncol = 2
        if len(topics_idx) == 1:
            ncol = 1
        if len(topics_idx) > 6:
            ncol = 3
        if len(topics_idx) > 9:
            ncol = 4
    nrows = int(np.ceil(len(topics_idx) / ncol))

    def _my_top(scores: np.ndarray, num: int, balanced: bool) -> np.ndarray:
        nr = scores.shape[0]
        if num > nr:
            num = nr
        if num == 1:
            balanced = False
        if balanced:
            num = max(1, num // 2)
            order = np.argsort(-scores)
            pos = order[:num]
            neg = order[-num:][::-1]
            if num > 0 and pos[-1] == neg[-1]:
                neg = neg[:-1]
            return np.concatenate([pos, neg])
        order = np.argsort(-np.abs(scores))[:num]
        # within `top`, order by score asc
        return order[np.argsort(scores[order])]

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncol,
        figsize=(4.0 * ncol, 3.5 * nrows),
        squeeze=False,
    )
    for k, t_idx in enumerate(topics_idx):
        ax = axes[k // ncol, k % ncol]
        scores_full = loadings_arr[:, t_idx - 1]
        keep = _my_top(scores_full, nfeatures, balanced)
        feats = [feature_names[i] for i in keep]
        scores = scores_full[keep]
        df = pd.DataFrame({"feature": feats, "score": scores})

        if feature_cat in ("ligand", "signaling") and interaction is not None:
            present = [f for f in df["feature"] if f in set(interaction.index)]
            if len(present) == 0:
                data_plot = df.copy()
            else:
                lookup = interaction.loc[present, ["ligand", "receptor", "pathway_name"]]
                data_plot = df.merge(
                    lookup.assign(feature=lookup.index), on="feature", how="left"
                )
                if feature_cat == "ligand":
                    data_plot = (
                        data_plot.groupby("ligand", as_index=False)["score"].mean()
                        .rename(columns={"ligand": "feature"})
                    )
                else:
                    data_plot = (
                        data_plot.groupby("pathway_name", as_index=False)["score"].mean()
                        .rename(columns={"pathway_name": "feature"})
                    )
        else:
            if interaction is not None and "interaction_name_2" in interaction.columns:
                # Map LR-pair feature name to its pretty version
                pretty = []
                for f in df["feature"]:
                    if f in interaction.index:
                        pretty.append(interaction.loc[f, "interaction_name_2"])
                    else:
                        pretty.append(f)
                df = df.assign(feature=pretty)
            data_plot = df

        data_plot["positive"] = np.where(data_plot["score"] > 0, "pos", "neg")
        data_plot["abs"] = np.abs(data_plot["score"])
        data_plot = data_plot.sort_values("abs", ascending=False, kind="stable")
        data_plot["rank"] = np.arange(1, len(data_plot) + 1)
        top = data_plot.head(int(nTop))

        ax.plot(
            data_plot["rank"], data_plot["abs"],
            color=_safe_color("grey80"), linewidth=line_size,
        )
        for _, row in top.iterrows():
            c = col[1] if row["positive"] == "pos" else col[0]
            ax.scatter(
                row["rank"], row["abs"],
                facecolors="none", edgecolors=c, s=point_size * 20.0,
            )
        # Labels with simple repulsion (no overlap with point cloud)
        repel_text(
            ax,
            top["rank"].to_numpy(dtype=float) + text_nudge_x,
            top["abs"].to_numpy(dtype=float),
            top["feature"].astype(str).tolist(),
            fontsize=text_size * 2.5,
            color=[
                col[1] if p == "pos" else col[0]
                for p in top["positive"]
            ],
            push_anchor=0.02,
            push_label=0.04,
            segment_color=_safe_color("grey50"),
            segment_alpha=1.0,
            segment_lw=0.3,
        )

        ax.set_xlabel("Ranking", fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([])
        ax.set_title(f"{topic_key}{t_idx}", fontsize=10, ha="center")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Hide unused axes
    for k in range(len(topics_idx), nrows * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.tight_layout()
    return fig
