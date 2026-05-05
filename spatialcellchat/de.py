"""Differential-expression / spatially-variable feature identifiers.

Ports the three R exports:

- :func:`identifyOverExpressedGenes` (R: utilities.R:1095)
- :func:`identifyOverExpressedLigandReceptor` (R: utilities.R:1417)
- :func:`identifyOverExpressedInteractions` (R: utilities.R:1496)

The R code branches on ``selection.method ∈ {"wilcox", "moransi", "meringue"}``.
The wilcox branch reproduces ``Seurat::FindAllMarkers``-style logic with a
fixed Wilcoxon rank-sum test and a Bonferroni correction over the *full*
gene universe. The moransi branch reproduces ``Seurat::RunMoransI`` (per-gene
Moran's I with inverse-square distance weights). The meringue branch
reproduces the authors' ``callSVG.MERINGUE`` helper using a Delaunay-based
binary neighbor graph and a per-gene normal-approximation p-value.

All three exports take an :class:`anndata.AnnData` (already log-normalised —
the R reference assumes ``object@data.signaling`` is normalised) and write
their outputs into ``adata.uns["cellchat"]``.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.spatial.distance import squareform, pdist
from scipy.stats import mannwhitneyu, norm

import anndata as ad

from ._anndata import cellchat_dict as _ensure_uns

__all__ = [
    "identifyOverExpressedGenes",
    "identifyOverExpressedLigandReceptor",
    "identifyOverExpressedInteractions",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_dense_matrix(X: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Materialise a (cells × genes) matrix as a dense float ``ndarray``."""
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=float)
    return np.asarray(X, dtype=float)


def _mean_log1p_expm1(arr: np.ndarray) -> np.ndarray:
    """``log(mean(expm1(x)) + 1)`` along the last axis (R's ``mean.fxn``)."""
    return np.log(np.mean(np.expm1(arr), axis=-1) + 1.0)


def _bonferroni(pvals: np.ndarray, n: int) -> np.ndarray:
    """Bonferroni correction over an explicit family size ``n`` (R's
    ``stats::p.adjust(method = "bonferroni", n = nrow(X))``).
    """
    return np.clip(pvals * n, a_min=None, a_max=1.0)


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg correction matching ``stats::p.adjust(method = "BH")``.

    Mirrors R's behaviour for ``NA`` p-values: positions with ``NaN`` are
    skipped (their family size shrinks correspondingly) and the output keeps
    ``NaN`` at those positions, exactly like ``p.adjust(NA, method="BH")``.
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    finite = ~np.isnan(pvals)
    if not finite.any():
        return out
    finite_idx = np.flatnonzero(finite)
    p = pvals[finite_idx]
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # cumulative-min from the right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, a_min=None, a_max=1.0)
    finite_out = np.empty_like(adj)
    finite_out[order] = adj
    out[finite_idx] = finite_out
    return out


def _ape_moran_i(x: np.ndarray, weight: np.ndarray) -> tuple[float, float, float, float]:
    """Direct port of ``ape::Moran.I(x, weight)`` (defaults: scaled = FALSE,
    na.rm = FALSE, alternative = "two.sided"). Returns
    ``(observed, expected, sd, p.value)``.
    """
    n = x.size
    ei = -1.0 / (n - 1)
    rowsum = weight.sum(axis=1)
    rowsum = np.where(rowsum == 0, 1.0, rowsum)
    weight = weight / rowsum[:, None]
    s = float(weight.sum())
    m = float(x.mean())
    y = x - m
    cv = float(np.sum(weight * np.outer(y, y)))
    v = float(np.sum(y * y))
    if v == 0:
        return float("nan"), ei, float("nan"), float("nan")
    obs = (n / s) * (cv / v)
    S1 = 0.5 * float(np.sum((weight + weight.T) ** 2))
    S2 = float(np.sum((weight.sum(axis=1) + weight.sum(axis=0)) ** 2))
    s_sq = s * s
    k = (np.sum(y ** 4) / n) / (v / n) ** 2
    sdi_sq = (
        n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * s_sq)
        - k * (n * (n - 1) * S1 - 2 * n * S2 + 6 * s_sq)
    ) / ((n - 1) * (n - 2) * (n - 3) * s_sq) - 1.0 / ((n - 1) ** 2)
    sdi = float(np.sqrt(sdi_sq))
    pv = norm.cdf(obs, loc=ei, scale=sdi)
    if obs <= ei:
        p = 2.0 * pv
    else:
        p = 2.0 * (1.0 - pv)
    return float(obs), float(ei), sdi, float(p)


def _meringue_neighbor_weights(coords: np.ndarray) -> np.ndarray:
    """Replicate ``MERINGUE::getSpatialNeighbors(pos, filterDist = NA,
    binary = TRUE)``: build a Delaunay triangulation, derive an undirected
    edge set, and return a 0/1 adjacency matrix (the MERINGUE default).
    """
    pos = coords.astype(float).copy()
    # Mirror R's perturb-on-duplicate behaviour
    seen: set[tuple[float, ...]] = set()
    for i in range(pos.shape[0]):
        key = tuple(pos[i])
        if key in seen:
            pos[i] = pos[i] + 1e-6
        seen.add(tuple(pos[i]))
    tri = Delaunay(pos)
    n = pos.shape[0]
    D = np.zeros((n, n), dtype=float)
    # Each simplex contributes the three pairwise edges (in 2-D)
    for simplex in tri.simplices:
        a, b, c = simplex
        for u, v in ((a, b), (a, c), (b, c)):
            d = float(np.linalg.norm(pos[u] - pos[v]))
            D[u, v] = d
            D[v, u] = d
    # binary = TRUE
    D[D > 0] = 1.0
    return D


def _meringue_spatial_patterns(
    mat: np.ndarray, weight: np.ndarray
) -> pd.DataFrame:
    """Replicate ``MERINGUE::getSpatialPatterns(mat, weight,
    alternative = "greater")``. ``mat`` is genes × cells. The MERINGUE C
    backend row-normalises the binary adjacency matrix (matching ape's
    convention) before running the Moran's-I formulas with the
    ``alternative = "greater"`` p-value and a final BH correction.
    """
    n_genes, n_cells = mat.shape
    n = n_cells
    rowsum = weight.sum(axis=1)
    rowsum = np.where(rowsum == 0, 1.0, rowsum)
    weight = weight / rowsum[:, None]
    s = float(weight.sum())
    s_sq = s * s
    S1 = 0.5 * float(np.sum((weight + weight.T) ** 2))
    S2 = float(np.sum((weight.sum(axis=1) + weight.sum(axis=0)) ** 2))
    ei = -1.0 / (n - 1)

    observed = np.empty(n_genes)
    expected = np.full(n_genes, ei)
    sd = np.empty(n_genes)
    pv = np.empty(n_genes)
    for i in range(n_genes):
        x = mat[i]
        m = float(x.mean())
        y = x - m
        cv = float(np.sum(weight * np.outer(y, y)))
        v = float(np.sum(y * y))
        if v == 0:
            observed[i] = float("nan")
            sd[i] = float("nan")
            pv[i] = float("nan")
            continue
        observed[i] = (n / s) * (cv / v)
        k = (np.sum(y ** 4) / n) / (v / n) ** 2
        sd_sq = (
            n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * s_sq)
            - k * (n * (n - 1) * S1 - 2 * n * S2 + 6 * s_sq)
        ) / ((n - 1) * (n - 2) * (n - 3) * s_sq) - 1.0 / ((n - 1) ** 2)
        sd[i] = float(np.sqrt(sd_sq))
        # alternative = "greater" → p = 1 - pnorm(obs, ei, sd)
        pv[i] = float(1.0 - norm.cdf(observed[i], loc=ei, scale=sd[i]))

    p_adj = _bh(pv)
    return pd.DataFrame(
        {
            "observed": observed,
            "expected": expected,
            "sd": sd,
            "p.value": pv,
            "p.adj": p_adj,
        }
    )


# ---------------------------------------------------------------------------
# identifyOverExpressedGenes (R: utilities.R:1095)
# ---------------------------------------------------------------------------
def identifyOverExpressedGenes(
    adata: ad.AnnData,
    *,
    do_grid: bool = False,
    cellsize: Sequence[float] = (5.0, 5.0),
    what: str = "polygons",
    square: bool = True,
    data_use: Optional[pd.DataFrame] = None,
    selection_method: Literal["wilcox", "moransi", "meringue"] = "wilcox",
    features_name: str = "features",
    group_by: Optional[str] = None,
    idents_use: Optional[Sequence] = None,
    invert: bool = False,
    group_dataset: Optional[str] = None,
    pos_dataset: Optional[str] = None,
    only_pos: bool = True,
    features: Optional[Sequence[str]] = None,
    thresh_pc: float = 0.0,
    thresh_fc: float = 0.0,
    thresh_p: float = 0.05,
) -> pd.DataFrame:
    """Identify over-expressed / spatially-variable genes for downstream LR
    inference.

    Parameters
    ----------
    adata : anndata.AnnData
        Cells × genes AnnData with already-normalised expression in ``adata.X``
        (matches the R reference's ``object@data.signaling``). For ``moransi``
        and ``meringue`` branches, ``adata.obsm["spatial"]`` must hold 2-D
        spot coordinates. For ``wilcox`` the ident vector is taken from
        ``adata.obs["active_ident"]`` when ``group_by`` is ``None``.
    do_grid : bool, default ``False``
        Whether to first run :func:`makeGridSpatialCellChat` to coarse-grain
        the cells into a regular grid before performing DE (R's ``do.grid``).
        Mirrors R ``utilities.R:1137-1177``: builds a grid AnnData, runs
        :func:`subsetData` on it (with the parent's CellChat DB), recurses
        with ``do_grid=False``, then propagates the resulting variable
        features back onto the parent ``adata`` and stashes the grid
        AnnData under ``adata.uns['cellchat']['misc']['grid.object']``.
    cellsize : sequence of float, default ``(5.0, 5.0)``
        Size of grid cells for the grid-DE branch (R's ``cellsize``).
    what : str, default ``"polygons"``
        Geometry kind passed to :func:`makeGridSpatialCellChat` (R's ``what``).
    square : bool, default ``True``
        Whether to use square grid cells (R's ``square``).
    data_use : pandas.DataFrame, optional
        Pre-prepared genes × cells expression matrix overriding ``adata.X``
        (mirrors R's ``data.use`` argument).
    selection_method : {"wilcox", "moransi", "meringue"}
        Selection algorithm (R's ``selection.method``).
    features_name : str, default ``"features"``
        Key under ``adata.uns["cellchat"]["DE"]`` (and prefix for ``.info``)
        that the results are stored at.
    group_by : str, optional
        ``adata.obs`` column to use as ident labels (default
        ``"active_ident"``).
    idents_use : sequence, optional
        Restrict comparisons to these idents (R's ``idents.use``).
    invert : bool, default ``False``
        When ``True`` and ``idents_use`` is set, exclude those idents.
    group_dataset, pos_dataset : str, optional
        Dataset/condition column and positive level for two-condition DE.
    only_pos : bool, default ``True``
        Keep only positive log-fold-changes after ranking.
    features : sequence of str, optional
        Restrict to these genes (intersect with ``adata.var_names``).
    thresh_pc, thresh_fc, thresh_p : float
        Minimum-percent-expressing, log2-FC, and p-value thresholds (R defaults
        ``0, 0, 0.05``).

    Returns
    -------
    pandas.DataFrame
        ``markers.all`` table for ``wilcox``; the per-gene Moran's-I result
        for ``moransi`` / ``meringue``. Also written into
        ``adata.uns["cellchat"]["DE"]`` and ``adata.var["highly_variable_signaling"]``.
    """
    if do_grid:
        # R utilities.R:1137-1177 — recursive grid-DE branch.
        from .preprocessing import subsetData
        from .spatial import makeGridSpatialCellChat

        cc_main = adata.uns.get("cellchat") or {}
        db = cc_main.get("DB")
        # R: ``grid.object@DB <- object@DB`` happens inline after the
        # ``makeGridSpatialCellChat`` call.  In Python ``subsetData``
        # accepts ``db`` as a kwarg, so we don't need to mutate
        # ``grid_adata.uns`` first — we just pass ``db=`` directly.

        grid_adata = makeGridSpatialCellChat(
            adata, cellsize=tuple(cellsize), what=what, square=square,
        )

        # Mirror R: ``subsetData(grid.object)`` with the parent's DB.
        # When ``db`` is None (caller hasn't loaded a CellChatDB yet) the
        # grid recursion would fail at the ``data.signaling`` precondition
        # — let R-style error propagate.
        grid_adata = subsetData(grid_adata, db=db)

        # Recursive call (R: identifyOverExpressedGenes(grid.object,
        # do.grid=F, data.use=NULL, …)).  Forward every other R argument
        # verbatim; ``data_use`` is reset to None per R line 1146.
        markers_all = identifyOverExpressedGenes(
            grid_adata,
            do_grid=False,
            cellsize=cellsize,
            what=what,
            square=square,
            data_use=None,
            selection_method=selection_method,
            features_name=features_name,
            group_by=group_by,
            idents_use=idents_use,
            invert=invert,
            group_dataset=group_dataset,
            pos_dataset=pos_dataset,
            only_pos=only_pos,
            features=features,
            thresh_pc=thresh_pc,
            thresh_fc=thresh_fc,
            thresh_p=thresh_p,
        )

        # Propagate `grid.object@var.features` back to the parent —
        # mirrors R: ``object@var.features <- grid.object@var.features``.
        cc_grid = grid_adata.uns.get("cellchat") or {}
        de_grid = cc_grid.get("DE", {})
        cc_main = adata.uns.setdefault("cellchat", {})
        de_main = cc_main.setdefault("DE", {})
        info_key = f"{features_name}.info"
        if features_name in de_grid:
            de_main[features_name] = list(de_grid[features_name])
        if info_key in de_grid:
            de_main[info_key] = de_grid[info_key]
        # Per-gene flag on parent.var (analogue of R's ``object@var.features``
        # being the canonical store of variable-feature membership).
        sig_set = set(de_main.get(features_name, []))
        adata.var["highly_variable_signaling"] = adata.var_names.isin(sig_set)

        # R: ``misc(object,"grid.object") <- grid.object``.
        misc = cc_main.setdefault("misc", {})
        misc["grid.object"] = grid_adata

        return markers_all
    if selection_method not in ("wilcox", "moransi", "meringue"):
        raise ValueError(
            "selection_method must be one of 'wilcox', 'moransi', 'meringue'"
        )

    # Build the genes × cells data matrix the R branch operates on
    if data_use is None:
        X = adata.X
        if sp.issparse(X):
            data_full = pd.DataFrame(
                np.asarray(X.todense(), dtype=float).T,
                index=list(adata.var_names),
                columns=list(adata.obs_names),
            )
        else:
            data_full = pd.DataFrame(
                np.asarray(X, dtype=float).T,
                index=list(adata.var_names),
                columns=list(adata.obs_names),
            )
    else:
        data_full = data_use

    if features is None:
        features_use = list(data_full.index)
    else:
        features_use = [g for g in features if g in data_full.index]
    data_use_mat = data_full.loc[features_use]
    full_n_genes = data_full.shape[0]

    if selection_method == "wilcox":
        markers_all = _wilcox_branch(
            data_use=data_use_mat,
            adata=adata,
            full_n_genes=full_n_genes,
            group_by=group_by,
            idents_use=idents_use,
            invert=invert,
            group_dataset=group_dataset,
            pos_dataset=pos_dataset,
            only_pos=only_pos,
            thresh_pc=thresh_pc,
            thresh_fc=thresh_fc,
            thresh_p=thresh_p,
        )
        features_sig = list(markers_all["features"].astype(str))
    elif selection_method == "moransi":
        coords = _get_coords(adata)
        markers_all = _moransi_branch(data_use_mat, coords)
        keep = (markers_all["p.value"] < thresh_p) & (~markers_all["observed"].isna())
        features_sig = list(markers_all.index[keep])
    else:  # meringue
        coords = _get_coords(adata)
        markers_all = _meringue_branch(data_use_mat, coords)
        keep = (markers_all["p.adj"] < thresh_p) & (~markers_all["observed"].isna())
        features_sig = list(markers_all.index[keep])

    cc = _ensure_uns(adata)
    de = cc.setdefault("DE", {})
    de[features_name] = features_sig
    de[f"{features_name}.info"] = markers_all
    sig_set = set(features_sig)
    adata.var["highly_variable_signaling"] = adata.var_names.isin(sig_set)
    return markers_all


def _get_coords(adata: ad.AnnData) -> np.ndarray:
    if "spatial" not in adata.obsm:
        raise ValueError(
            "selection_method requires spatial coordinates in adata.obsm['spatial']"
        )
    return np.asarray(adata.obsm["spatial"], dtype=float)


def _wilcox_branch(
    *,
    data_use: pd.DataFrame,
    adata: ad.AnnData,
    full_n_genes: int,
    group_by: Optional[str],
    idents_use: Optional[Sequence],
    invert: bool,
    group_dataset: Optional[str],
    pos_dataset: Optional[str],
    only_pos: bool,
    thresh_pc: float,
    thresh_fc: float,
    thresh_p: float,
) -> pd.DataFrame:
    """R: utilities.R:1198-1326 — Wilcoxon-test selection branch."""
    if group_by is None:
        if "active_ident" not in adata.obs.columns:
            raise ValueError(
                "group_by is None and adata.obs has no 'active_ident' column"
            )
        labels_raw = adata.obs["active_ident"]
    else:
        labels_raw = adata.obs[group_by]

    if isinstance(labels_raw.dtype, pd.CategoricalDtype):
        level_use = [
            lvl for lvl in labels_raw.cat.categories if lvl in set(labels_raw)
        ]
    else:
        level_use = sorted(pd.unique(labels_raw))
    if idents_use is not None:
        idents_set = set(idents_use)
        if invert:
            level_use = [lvl for lvl in level_use if lvl not in idents_set]
        else:
            level_use = [lvl for lvl in level_use if lvl in idents_set]
    num_cluster = len(level_use)

    labels = np.asarray(labels_raw.astype(str))
    if group_dataset is not None:
        labels_dataset = np.asarray(adata.obs[group_dataset].astype(str))
        if pos_dataset is None or pos_dataset not in set(labels_dataset):
            raise ValueError(
                "pos_dataset must be one of: " + ", ".join(sorted(set(labels_dataset)))
            )
    else:
        labels_dataset = None

    data_arr = data_use.to_numpy(dtype=float)  # genes × cells
    feature_names = np.asarray(data_use.index)

    genes_de: list[Optional[pd.DataFrame]] = [None] * num_cluster
    for i, lvl in enumerate(level_use):
        if labels_dataset is None:
            cell_use1 = np.where(labels == str(lvl))[0]
            cell_use2 = np.setdiff1d(np.arange(labels.size), cell_use1)
        else:
            cell_use1 = np.where(
                (labels == str(lvl)) & (labels_dataset == str(pos_dataset))
            )[0]
            cell_use2 = np.where(
                (labels == str(lvl)) & (labels_dataset != str(pos_dataset))
            )[0]
        if cell_use1.size == 0 or cell_use2.size == 0:
            continue
        sub1 = data_arr[:, cell_use1]
        sub2 = data_arr[:, cell_use2]

        thresh_min = 0.0
        pct1 = np.round((sub1 > thresh_min).sum(axis=1) / cell_use1.size, 3)
        pct2 = np.round((sub2 > thresh_min).sum(axis=1) / cell_use2.size, 3)
        alpha_min = np.maximum(pct1, pct2)
        keep_mask = alpha_min > thresh_pc
        if not keep_mask.any():
            continue
        feat_idx = np.where(keep_mask)[0]
        sub1 = sub1[feat_idx]
        sub2 = sub2[feat_idx]
        feats_after_pc = feature_names[feat_idx]
        pct1 = pct1[feat_idx]
        pct2 = pct2[feat_idx]

        data1_mean = _mean_log1p_expm1(sub1)
        data2_mean = _mean_log1p_expm1(sub2)
        FC = data1_mean - data2_mean
        if only_pos:
            diff_mask = FC > thresh_fc
        else:
            diff_mask = np.abs(FC) > thresh_fc
        if not diff_mask.any():
            continue
        sub1 = sub1[diff_mask]
        sub2 = sub2[diff_mask]
        feats_final = feats_after_pc[diff_mask]
        pct1 = pct1[diff_mask]
        pct2 = pct2[diff_mask]
        FC = FC[diff_mask]

        # wilcox.test(x, y) — two-sided, with default exact for small n,
        # normal approximation otherwise. R uses ``exact = NULL`` which means
        # exact when both samples < 50 and no ties. SciPy's mannwhitneyu with
        # ``method="auto"`` matches that behaviour.
        pvals = np.empty(len(feats_final), dtype=float)
        for j in range(len(feats_final)):
            res = mannwhitneyu(
                sub1[j], sub2[j], alternative="two-sided", method="auto"
            )
            pvals[j] = res.pvalue
        pval_adj = _bonferroni(pvals, full_n_genes)
        genes_de[i] = pd.DataFrame(
            {
                "clusters": str(lvl),
                "features": feats_final.astype(str),
                "pvalues": pvals,
                "logFC": FC,
                "pct.1": pct1,
                "pct.2": pct2,
                "pvalues.adj": pval_adj,
            }
        )

    parts: list[pd.DataFrame] = []
    for gde in genes_de:
        if gde is None:
            continue
        # R: gde[order(gde$pvalues, -gde$logFC), ]
        gde = gde.sort_values(["pvalues", "logFC"], ascending=[True, False])
        gde = gde[gde["pvalues"] < thresh_p]
        if not gde.empty:
            parts.append(gde)
    if parts:
        markers_all = pd.concat(parts, ignore_index=True)
    else:
        markers_all = pd.DataFrame(
            columns=[
                "clusters",
                "features",
                "pvalues",
                "logFC",
                "pct.1",
                "pct.2",
                "pvalues.adj",
            ]
        )
    if only_pos and len(markers_all) > 0:
        markers_all = markers_all[markers_all["logFC"] > 0].reset_index(drop=True)
    if group_dataset is not None and len(markers_all) > 0:
        # R parity (utilities.R:1319-1323):
        #   markers.all$datasets[markers.all$logFC > 0] <- pos.dataset
        #   markers.all$datasets[markers.all$logFC < 0] <-
        #       setdiff(unique(labels.dataset), pos.dataset)
        #
        # The factor *levels* are ``levels(factor(meta[[group.dataset]]))``
        # — alphabetical (R's ``levels()`` default ordering).
        # The *recycling source* for negative-FC rows is
        # ``setdiff(unique(labels.dataset), pos.dataset)`` — preserves
        # first-appearance order in ``meta[[group.dataset]]``. R then
        # recycles that vector into the logical-mask assignment, so for
        # >2 datasets each negative-FC row cycles through the available
        # "other" datasets.
        #
        # Rows with ``logFC == 0`` are NEITHER condition — R leaves them
        # as ``NA`` (factor was initialised with NAs). Python uses
        # ``pd.NA`` in an object array → ``Categorical`` will store NaN,
        # mirroring R bit-for-bit (audit M1).
        labels_ds_arr = adata.obs[group_dataset].astype(str).to_numpy()
        labels_ds_unique = sorted(pd.unique(labels_ds_arr))   # factor levels (alphabetical)
        # Appearance-ordered ``setdiff`` for recycling (audit M2).
        others_in_appearance = [
            d for d in pd.unique(labels_ds_arr) if d != str(pos_dataset)
        ]
        logFC = markers_all["logFC"].to_numpy()
        ds: np.ndarray = np.full(len(markers_all), None, dtype=object)
        ds[logFC > 0] = str(pos_dataset)
        neg_idx = np.flatnonzero(logFC < 0)
        if neg_idx.size > 0 and len(others_in_appearance) > 0:
            # R recycles ``setdiff`` through the negative-FC rows.
            M = len(others_in_appearance)
            ds[neg_idx] = [others_in_appearance[k % M] for k in range(neg_idx.size)]
        markers_all["datasets"] = pd.Categorical(ds, categories=labels_ds_unique)
        markers_all = markers_all.sort_values(
            ["datasets", "pvalues", "logFC"], ascending=[True, True, False]
        ).reset_index(drop=True)
    markers_all["features"] = markers_all["features"].astype(str)
    return markers_all


def _moransi_branch(data_use: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """R: utilities.R:1328-1339 — Seurat::RunMoransI per gene."""
    pos_dist = squareform(pdist(coords))
    with np.errstate(divide="ignore"):
        weights = 1.0 / (pos_dist ** 2)
    np.fill_diagonal(weights, 0.0)

    arr = data_use.to_numpy(dtype=float)
    n_genes = arr.shape[0]
    obs = np.empty(n_genes)
    pvals = np.empty(n_genes)
    for i in range(n_genes):
        observed, _ei, _sd, p = _ape_moran_i(arr[i], weights)
        obs[i] = observed
        pvals[i] = p
    return pd.DataFrame(
        {"observed": obs, "p.value": pvals},
        index=list(data_use.index),
    )


def _meringue_branch(data_use: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
    """R: utilities.R:1340-1357 — MERINGUE callSVG.MERINGUE."""
    weight = _meringue_neighbor_weights(coords)
    res = _meringue_spatial_patterns(data_use.to_numpy(dtype=float), weight)
    res.index = list(data_use.index)
    return res


# ---------------------------------------------------------------------------
# identifyOverExpressedLigandReceptor (R: utilities.R:1417)
# ---------------------------------------------------------------------------
def identifyOverExpressedLigandReceptor(
    adata: ad.AnnData,
    db: dict[str, pd.DataFrame],
    *,
    features_name: str = "features",
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Identify over-expressed ligands / (complex) receptors per cell group.

    Mirrors R's :func:`identifyOverExpressedLigandReceptor` (utilities.R:1417).
    Reads the per-cluster ``markers.all`` table (written by
    :func:`identifyOverExpressedGenes` under ``adata.uns["cellchat"]["DE"]
    [<features_name>.info]``) and rewrites each row as either:
    (a) the raw ligand/receptor symbol if it appears directly in the
    interaction table's ``ligand`` or ``receptor`` columns, or
    (b) one row per *complex* whose subunit table contains that gene
    symbol.

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``uns["cellchat"]["DE"][<features_name>.info]`` table was
        populated by :func:`identifyOverExpressedGenes`.
    db : dict
        CellChatDB dict (must contain ``"interaction"`` and ``"complex"``
        keys), as returned by :func:`spatialcellchat.load_db`.
    features_name : str, default ``"features"``
        Key under ``adata.uns["cellchat"]["DE"]`` whose ``.info`` table is
        consumed.
    features : sequence of str, optional
        Restrict the input markers table to these gene symbols (R's
        ``features`` argument).

    Returns
    -------
    pandas.DataFrame
        Filtered/expanded markers table. Also written to
        ``adata.uns["cellchat"]["DE"][<features_name>.LR]``.
    """
    cc = _ensure_uns(adata)
    de = cc.setdefault("DE", {})
    info_key = f"{features_name}.info"
    if info_key not in de:
        raise ValueError(
            f"adata.uns['cellchat']['DE']['{info_key}'] missing — run "
            "identifyOverExpressedGenes() first"
        )
    markers_all = de[info_key]
    if features is not None:
        feats = set(features)
        markers_all = markers_all[markers_all["features"].isin(feats)]

    interaction_input = db["interaction"]
    complex_input = db["complex"]
    pair_lr = interaction_input[["ligand", "receptor"]]
    lr_use = set(pd.unique(pd.concat([pair_lr["ligand"], pair_lr["receptor"]])))

    complex_subunits = complex_input.loc[
        :, [c for c in complex_input.columns if "subunit" in c]
    ]
    complex_index = list(complex_subunits.index)
    # Pre-compute per-complex non-empty subunit lists
    subunits_per_complex: list[list[str]] = []
    for _, row in complex_subunits.iterrows():
        vals = [str(v) for v in row.values if pd.notna(v) and str(v) != ""]
        subunits_per_complex.append(vals)

    new_rows: list[pd.DataFrame] = []
    for i in range(len(markers_all)):
        row = markers_all.iloc[[i]]
        feat = str(row["features"].iloc[0])
        if feat in lr_use:
            new_rows.append(row)
            continue
        # find every complex whose subunit list contains this gene
        complex_hits = [
            complex_index[x]
            for x, subs in enumerate(subunits_per_complex)
            if feat in subs
        ]
        if not complex_hits:
            continue
        for cname in complex_hits:
            r = row.copy()
            r.iloc[0, r.columns.get_loc("features")] = cname
            new_rows.append(r)

    if new_rows:
        markers_new = pd.concat(new_rows, ignore_index=True)
    else:
        markers_new = markers_all.iloc[0:0].copy()

    de[f"{features_name}.LR"] = markers_new
    return markers_new


# ---------------------------------------------------------------------------
# identifyOverExpressedInteractions (R: utilities.R:1496)
# ---------------------------------------------------------------------------
def identifyOverExpressedInteractions(
    adata: ad.AnnData,
    db: dict[str, pd.DataFrame],
    *,
    features_name: str = "features",
    variable_both: bool = True,
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter the interaction table down to over-expressed ligand-receptor
    pairs.

    Mirrors R's :func:`identifyOverExpressedInteractions` (utilities.R:1496).
    A complex (rowname of ``db["complex"]``) qualifies as ``"sig"`` when at
    least one of its subunits appears in the over-expressed gene set *and*
    every subunit is present in the data universe. A pair ``(ligand, receptor)``
    is retained when:

    - ``variable_both = True``: both ligand and receptor are in the union of
      the over-expressed gene set and the sig-complex names; or
    - ``variable_both = False``: every member is present in the data universe
      *and* the pair has at least one over-expressed member (gene or complex).

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``uns["cellchat"]["DE"][<features_name>]`` lists the
        over-expressed genes (written by :func:`identifyOverExpressedGenes`).
    db : dict
        CellChatDB dict.
    features_name : str, default ``"features"``
        Key under ``adata.uns["cellchat"]["DE"]`` containing the
        over-expressed gene list.
    variable_both : bool, default ``True``
        See above.
    features : sequence of str, optional
        Override the over-expressed gene set with this list.

    Returns
    -------
    pandas.DataFrame
        Filtered interaction table. Also written to
        ``adata.uns["cellchat"]["LR"]["LRsig"]``.
    """
    gene_use = list(adata.var_names)
    cc = _ensure_uns(adata)
    de = cc.setdefault("DE", {})
    if features is None:
        if features_name not in de:
            raise ValueError(
                f"adata.uns['cellchat']['DE']['{features_name}'] missing — run "
                "identifyOverExpressedGenes() first"
            )
        features_sig = list(de[features_name])
    else:
        features_sig = list(features)

    interaction_input = db["interaction"]
    complex_input = db["complex"]
    complex_subunits = complex_input.loc[
        :, [c for c in complex_input.columns if "subunit" in c]
    ]
    complex_index = list(complex_subunits.index)
    subunits_per_complex: list[list[str]] = []
    for _, row in complex_subunits.iterrows():
        vals = [str(v) for v in row.values if pd.notna(v) and str(v) != ""]
        subunits_per_complex.append(vals)

    feats_set = set(features_sig)
    gene_use_set = set(gene_use)

    # complexSubunits.sig — subunits intersect features.sig AND all subunits in data
    complex_sig_idx = [
        x
        for x, subs in enumerate(subunits_per_complex)
        if (len(set(subs) & feats_set) > 0) and all(s in gene_use_set for s in subs)
    ]
    complex_sig_names = {complex_index[x] for x in complex_sig_idx}

    # complexSubunits.use — all subunits in data
    complex_use_idx = [
        x
        for x, subs in enumerate(subunits_per_complex)
        if all(s in gene_use_set for s in subs)
    ]
    complex_use_names = {complex_index[x] for x in complex_use_idx}

    pair_lr = interaction_input[["ligand", "receptor"]]
    keep_idx: list[int] = []
    sig_universe = feats_set | complex_sig_names
    use_universe = gene_use_set | complex_use_names
    for x in range(len(pair_lr)):
        members = [
            str(pair_lr["ligand"].iloc[x]),
            str(pair_lr["receptor"].iloc[x]),
        ]
        if variable_both:
            if all(m in sig_universe for m in members):
                keep_idx.append(x)
        else:
            if all(m in use_universe for m in members) and (
                len(set(members) & sig_universe) > 0
            ):
                keep_idx.append(x)
    pair_lr_sig = interaction_input.iloc[keep_idx]
    lr_dict = cc.setdefault("LR", {})
    lr_dict["LRsig"] = pair_lr_sig
    return pair_lr_sig
