"""Cell-cell communication modeling: ``computeCommunProb`` + filters.

Direct port of the cell-level / pathway-level / filter exports in
``SpatialCellChat::modeling.R`` lines 29-336, 1517-1645, 1709-1871,
1932-1961, 1976-2054, 2077-2146, and 2613-2683.

The public surface accepts an :class:`anndata.AnnData` first argument and
writes results into ``adata.uns["cellchat"]["net"]``:

- ``net["prob.cell"]``: dense list-of-CSC equivalent. Stored as
  :class:`sparse.COO` (pydata sparse 0.18) of shape ``(nC, nC, nLR)``,
  matching R's ``spatstat.sparse::sparse3Darray``.
- ``net["tmp"]``: dict with ``"prob.cell"`` (a list of CSC matrices, one
  per LR pair, mirroring R's ``net$tmp$prob.cell``), ``"Lavg"``,
  ``"Ravg"``.

The R reference (``modeling.R`` lines noted in each docstring) is the
ground truth. The expression helpers used here live in :mod:`._expr`,
the Hill / spatial probability primitives in :mod:`._hill`, and the
spatial-distance computation in :mod:`.spatial`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import sparse as _sparse
from scipy import sparse

from ._anndata import cellchat_dict as _ensure_uns
from ._anndata import resolve_data_matrix
from ._expr import (
    computeExpr_LR,
    computeExpr_agonist,
    computeExpr_antagonist,
    computeExpr_coreceptor,
)
from ._hill import HillFunctionFordataLR, createPspatialFrom_dspatial
from ._utils import myElementwiseProduct
from .database import searchPair
from .spatial import computeCellDistance

__all__ = [
    "computeCommunProb",
    "computeCommunProbPathway",
    "aggregateNet",
    "filterCommunication",
    "filterProbability",
    "scMatrixTruncation",
    "subsetCommunication",
    "identifyEnrichedInteractions",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_data_use(adata: ad.AnnData, raw_use: bool, layer: Optional[str]) -> pd.DataFrame:
    """Return a (genes x cells) DataFrame with row-named gene index.

    Direct port of R's :func:`computeCommunProb` data-loading branch::

        if (raw.use) {
            data <- object@data.signaling
            data@x <- data@x/max(data@x)
            data.use <- as.matrix(data)
        } else {
            data <- object@data.project
            data.use <- data/max(data)
        }

    The Python port routes both branches through
    :func:`resolve_data_matrix` so the user-facing slot names map onto the
    canonical AnnData layers (``"data_signaling"`` / ``"data_project"``)
    without silent fall-through to ``adata.X``.  ``layer`` overrides the
    canonical lookup when the caller wants a non-default store.
    """
    if layer is not None:
        # Caller-supplied explicit override (e.g. a dataset-specific layer).
        block = adata.layers[layer]
        if sparse.issparse(block):
            arr = np.asarray(block.todense(), dtype=float)
        else:
            arr = np.asarray(block, dtype=float)
        if arr.shape == (adata.n_obs, adata.n_vars):
            arr = arr.T
        df = pd.DataFrame(
            arr,
            index=pd.Index(adata.var_names.astype(str)),
            columns=pd.Index(adata.obs_names.astype(str)),
        )
    else:
        slot = "data.signaling" if raw_use else "data.project"
        # ``raw_use=False`` requires the projection layer to have been
        # written by :func:`projectData` — mirror R's ``object@data.project``
        # precondition by using ``require_layer=True``.
        df = resolve_data_matrix(adata, slot, require_layer=not raw_use)

    # R rescales by the global maximum so the per-cell expression sits in
    # ``[0, 1]`` before the Hill function is applied.
    arr = df.to_numpy(dtype=float, copy=True)
    max_val = float(arr.max()) if arr.size > 0 else 1.0
    if max_val > 0:
        arr = arr / max_val
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def _get_idents(adata: ad.AnnData, group_by: Optional[str]) -> pd.Categorical:
    """Return the ``object@idents`` factor.

    Default key is ``"cellchat_idents"`` (per the porting contract).
    """
    if group_by is None:
        for k in ("cellchat_idents", "active_ident", "cell.type"):
            if k in adata.obs.columns:
                group_by = k
                break
    if group_by is None or group_by not in adata.obs.columns:
        raise ValueError(
            "computeCommunProb requires an idents column. Pass group_by= or "
            "set adata.obs['cellchat_idents']."
        )
    s = adata.obs[group_by]
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.values  # type: ignore[return-value]
    return pd.Categorical(s)


def _get_LR(adata: ad.AnnData, LR_use: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``pairLR.use`` mirroring the R top-of-function branch.

    When ``LR_use`` is ``None`` we fall back to
    ``adata.uns["cellchat"]["LR"]["LRsig"]`` (the table written by
    :func:`identifyOverExpressedInteractions`).
    """
    if LR_use is None:
        cc = _ensure_uns(adata)
        if "LR" not in cc or "LRsig" not in cc["LR"]:
            raise ValueError(
                "computeCommunProb requires uns['cellchat']['LR']['LRsig']; "
                "run identifyOverExpressedInteractions() first or pass LR_use="
            )
        return cc["LR"]["LRsig"]
    return LR_use


def _get_DB(adata: ad.AnnData, db: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Return the CellChatDB dict.

    Looks first for ``adata.uns["cellchat"]["DB"]``, then falls back to
    the explicit ``db`` argument.
    """
    if db is not None:
        return db
    cc = _ensure_uns(adata)
    if "DB" in cc:
        return cc["DB"]
    raise ValueError(
        "computeCommunProb requires the CellChatDB dict. Pass db= or set "
        "adata.uns['cellchat']['DB']."
    )


def _sort_LR_by_annotation(LR_use: pd.DataFrame) -> pd.DataFrame:
    """Replicate R's ordering of ``pairLR.use`` by annotation when more than
    one annotation type is present.

    Mirrors modeling.R lines 60-73::

        if (length(unique(LR.use$annotation)) > 1) {
            LR.use$annotation <- factor(LR.use$annotation, levels = c(
                "Secreted Signaling", "ECM-Receptor",
                "Non-protein Signaling", "Cell-Cell Contact"))
            LR.use <- LR.use[order(LR.use$annotation), , drop = FALSE]
            LR.use$annotation <- as.character(LR.use$annotation)
        }
    """
    if "annotation" not in LR_use.columns:
        return LR_use
    if LR_use["annotation"].nunique() <= 1:
        return LR_use
    levels = ["Secreted Signaling", "ECM-Receptor", "Non-protein Signaling", "Cell-Cell Contact"]
    df = LR_use.copy()
    cat = pd.Categorical(df["annotation"], categories=levels, ordered=True)
    df["annotation"] = cat
    df = df.sort_values("annotation", kind="stable")
    df["annotation"] = df["annotation"].astype(str)
    return df


def _crossprod_outer_sparse(L_row: np.ndarray, R_row: np.ndarray) -> sparse.csc_matrix:
    """Compute ``crossprod(L,R)`` for two ``(1, n)`` vectors as a sparse matrix.

    Mirrors R::

        x_ <- as(matrix(dataLavg[i, ], nrow = 1), "TsparseMatrix")
        y_ <- as(matrix(dataRavg[i, ], nrow = 1), "TsparseMatrix")
        dataLR <- Matrix::crossprod(x_, y_)

    Result is the outer product L^T R as an ``(n, n)`` sparse CSC matrix
    storing only entries where both ``L_row[i]`` and ``R_row[j]`` were
    non-zero in the original (sparse) row-vectors.
    """
    L = np.asarray(L_row, dtype=float).ravel()
    R = np.asarray(R_row, dtype=float).ravel()
    nL = L.size
    nR = R.size
    # Sparse rows: keep only positions where the corresponding L / R is
    # non-zero (mirrors the conversion-to-TsparseMatrix step).
    Lnz = np.flatnonzero(L != 0.0)
    Rnz = np.flatnonzero(R != 0.0)
    if Lnz.size == 0 or Rnz.size == 0:
        return sparse.csc_matrix((nL, nR), dtype=float)
    # Build outer product on the active sub-grid only.
    rows = np.repeat(Lnz, Rnz.size)
    cols = np.tile(Rnz, Lnz.size)
    vals = np.outer(L[Lnz], R[Rnz]).ravel()
    return sparse.coo_matrix(
        (vals, (rows, cols)), shape=(nL, nR)
    ).tocsc()


def _list_to_sparse3D(prob_list: List[sparse.spmatrix], shape: tuple) -> _sparse.COO:
    """Stack a list of ``(nC, nC)`` sparse matrices into a ``sparse.COO``
    of shape ``(nC, nC, nLR)``.

    Mirrors R's ``my_as_sparse3Darray`` (binds a list of 2-D sparse
    matrices into a 3-D ``sparse3Darray``).
    """
    nC1, nC2, nLR = shape
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    layers: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for k, M in enumerate(prob_list):
        coo = M.tocoo()
        if coo.nnz == 0:
            continue
        rows.append(coo.row)
        cols.append(coo.col)
        layers.append(np.full(coo.nnz, k, dtype=int))
        vals.append(coo.data)
    if len(vals) == 0:
        return _sparse.COO(
            np.zeros((3, 0), dtype=int), np.array([], dtype=float),
            shape=(nC1, nC2, nLR),
        )
    coords = np.vstack([
        np.concatenate(rows), np.concatenate(cols), np.concatenate(layers)
    ])
    data = np.concatenate(vals)
    return _sparse.COO(coords, data, shape=(nC1, nC2, nLR))


# ---------------------------------------------------------------------------
# computeCommunProb (R: modeling.R:29-319)
# ---------------------------------------------------------------------------
def computeCommunProb(
    adata: ad.AnnData,
    LR_use: Optional[pd.DataFrame] = None,
    db: Optional[Dict[str, pd.DataFrame]] = None,
    raw_use: bool = True,
    Kh: float = 0.5,
    n: float = 1.0,
    distance_use: bool = True,
    tol: Optional[float] = None,
    interaction_range: float = 250.0,
    scale_distance: float = 0.01,
    use_AGAN: bool = True,
    contact_dependent: bool = True,
    contact_range: float = 10.0,
    contact_dependent_forced: bool = False,
    *,
    group_by: Optional[str] = None,
    layer: Optional[str] = None,
) -> ad.AnnData:
    """Compute the cell-level communication probability for every L-R pair.

    Direct port of R's :func:`computeCommunProb` (``modeling.R:29-319``).
    Per LR pair, the probability is

    .. math:: P_{ij} = \\text{Hill}(L_i \\cdot R_j; K_h, n) \\cdot
        P_{\\text{spatial}}(d_{ij}) \\cdot
        f_{\\text{ago}}(\\text{cofactor}; K_h, n) \\cdot
        f_{\\text{ant}}(\\text{cofactor}; K_h, n)

    where ``L_i`` is the (geometric-mean-aggregated) ligand expression in
    cell ``i``, ``R_j`` is the (geometric-mean-aggregated) receptor
    expression in cell ``j``, ``P_{spatial}`` is the inverse-distance
    weight from :func:`createPspatialFrom_dspatial`, and
    ``f_{ago}`` / ``f_{ant}`` are the agonist / antagonist Hill
    multipliers from :func:`computeExpr_agonist` /
    :func:`computeExpr_antagonist`. Pairs annotated as ``Cell-Cell
    Contact`` are additionally multiplied by the binary
    ``adj.contact`` mask (R lines 216-218, 228-230).

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData. Must have ``obsm["spatial"]`` (the spatial
        coordinates) and ``uns["cellchat"]["spatial.factors"]`` with
        keys ``"ratio"`` and ``"tol"``.
    LR_use : pandas.DataFrame, optional
        Subset of L-R pairs to use. Defaults to the
        ``uns["cellchat"]["LR"]["LRsig"]`` table.
    db : dict, optional
        CellChatDB dict (``"interaction"``, ``"complex"``, ``"cofactor"``,
        ``"geneInfo"``). Defaults to ``adata.uns["cellchat"]["DB"]``.
    raw_use : bool, default ``True``
        Use the raw (``data.signaling``) layer instead of
        the projected (``data.project``) one.
    Kh : float, default ``0.5``
        Hill EC50 (R: ``Kh``).
    n : float, default ``1``
        Hill coefficient (R: ``n``).
    distance_use : bool, default ``True``
        Whether to weight communications by ``1/d``. When ``False`` the
        result is a binary connectivity weight bounded by
        ``interaction_range``.
    tol : float, optional
        Distance tolerance, in microns (R: ``tol``). Defaults to the
        ``"tol"`` field of ``adata.uns["cellchat"]["spatial.factors"]``.
    interaction_range : float, default ``250``
        Maximum interaction range, in microns (R: ``interaction.range``).
    scale_distance : float, default ``0.01``
        Distance scaling factor (R: ``scale.distance``); chosen so the
        minimum scaled distance falls in ``[1, 2]``.
    use_AGAN : bool, default ``True``
        Whether to multiply by agonist / antagonist factors.
    contact_dependent : bool, default ``True``
        See R docs.
    contact_range : float, default ``10``
        Cell-cell contact threshold, in microns.
    contact_dependent_forced : bool, default ``False``
        When ``True`` every L-R pair is treated as contact-dependent.
    group_by : str, optional
        Column of ``adata.obs`` used as the ident factor. Defaults to
        ``"cellchat_idents"`` if present, else ``"active_ident"``,
        else ``"cell.type"``.
    layer : str, optional
        Layer name to use as the expression matrix.  Defaults to the
        canonical R-mirrored store: ``adata.layers["data_signaling"]``
        (R's ``chat@data.signaling`` — ALRA-imputed by
        :func:`preProcessing`) when ``raw_use=True``, or
        ``adata.layers["data_project"]`` (R's ``chat@data.project`` —
        written by :func:`projectData`) when ``raw_use=False``.  Pass
        an explicit layer name to override.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``. Results are written to
        ``adata.uns["cellchat"]["net"]``:

        - ``"prob.cell"``: :class:`sparse.COO` of shape ``(nC, nC, nLR)``.
        - ``"tmp"``: dict with ``"prob.cell"`` (list of CSC, one per LR
          pair), ``"Lavg"`` (dense ``(nLR, nC)``), ``"Ravg"`` (same).

        and to ``adata.uns["cellchat"]["options"]``:

        - ``"parameter"``: a dict capturing all call parameters and
          internally-derived ``nLR``, ``nLR1``, ``all.contact.dependent``,
          ``all.diffusible``.
    """
    cc = _ensure_uns(adata)
    DB = _get_DB(adata, db)

    # R lines 45-55: build data.use (genes x cells, scaled by global max).
    data_use_df = _get_data_use(adata, raw_use=raw_use, layer=layer)

    # R lines 57-75: pairLR.use; annotation-sorted only when the caller
    # passed LR.use (R: ``if (is.null(LR.use)) pairLR.use <- object@LR$LRsig``
    # — no resort; otherwise build a factor on annotation and sort).
    if LR_use is None:
        pairLR_use = _get_LR(adata, None)
    else:
        pairLR_use = _sort_LR_by_annotation(LR_use)

    # R lines 77-78
    complex_input = DB["complex"]
    cofactor_input = DB["cofactor"]

    # R lines 82-90: cell idents
    pairLRsig = pairLR_use
    group = _get_idents(adata, group_by)
    geneL = pairLRsig["ligand"].astype(str).tolist()
    geneR = pairLRsig["receptor"].astype(str).tolist()
    nLR = len(pairLRsig)
    numCluster = len(group.categories)
    if numCluster != len(set(group)):
        raise ValueError(
            "Please check `unique(adata.obs[idents])` and ensure that the "
            "factor levels are correct! You may need to drop unused levels."
        )

    nC = data_use_df.shape[1]

    # ---------------------------------------------------------------------
    # R lines 98-118: spatial coordinates -> d.spatial / adj.contact
    # ---------------------------------------------------------------------
    if "spatial" not in adata.obsm:
        raise ValueError(
            "computeCommunProb requires adata.obsm['spatial']"
        )
    if "spatial.factors" not in cc:
        raise ValueError(
            "computeCommunProb requires uns['cellchat']['spatial.factors'] "
            "with keys 'ratio' and 'tol'"
        )
    sf = cc["spatial.factors"]
    ratio = float(sf["ratio"])
    if tol is None:
        tol = float(sf["tol"])

    res = computeCellDistance(
        adata.obsm["spatial"],
        ratio=ratio,
        interaction_range=float(interaction_range),
        contact_range=float(contact_range),
        tol=float(tol),
    )
    d_spatial = res["d.spatial"]
    adj_contact = res["adj.contact"]

    # ---------------------------------------------------------------------
    # R lines 120-137: scale d.spatial -> P.spatial
    # ---------------------------------------------------------------------
    if distance_use:
        # Scale stored values by scale.distance (in-place mimic)
        d_spatial_scaled = d_spatial.copy()
        d_spatial_scaled.data = d_spatial_scaled.data * float(scale_distance)
        if d_spatial_scaled.nnz == 0:
            d_min = float("inf")
        else:
            d_min = float(np.min(d_spatial_scaled.data))
        if d_min < 1.0 and d_spatial_scaled.nnz > 0:
            raise ValueError(
                "Please increase the value of `scale_distance` and use a value "
                f"that is slightly smaller than {1.0 / d_min:.2g}"
            )
        P_spatial = createPspatialFrom_dspatial(d_spatial_scaled, distance_use=True)
    else:
        P_spatial = createPspatialFrom_dspatial(d_spatial, distance_use=False)

    # ---------------------------------------------------------------------
    # R lines 139-172: nLR1 / all.contact.dependent / all.diffusible
    # ---------------------------------------------------------------------
    all_contact_dependent = False
    all_diffusible = False
    if contact_dependent_forced:
        # Force every pair to be contact-dependent: P.spatial = P.spatial * adj.contact
        P_spatial = P_spatial.multiply(adj_contact)
        nLR1 = nLR
        all_contact_dependent = True
    else:
        if contact_dependent and "annotation" in pairLRsig.columns and \
                pairLRsig["annotation"].nunique() > 0:
            ann_unique = set(pairLRsig["annotation"].astype(str).unique())
            if ann_unique == {"Cell-Cell Contact"}:
                P_spatial = P_spatial.multiply(adj_contact)
                nLR1 = nLR
                all_contact_dependent = True
            elif ann_unique.issubset({
                "Secreted Signaling", "ECM-Receptor", "Non-protein Signaling"
            }):
                nLR1 = nLR
                all_diffusible = True
            else:
                # Mixed: nLR1 is the LAST index (0-based: last+1) of the
                # diffusible block, since LR has been sorted by annotation
                # so diffusible pairs come first and Cell-Cell Contact pairs
                # come last (R levels order).
                ann = pairLRsig["annotation"].astype(str).values
                diffusible_mask = np.isin(
                    ann, ["Secreted Signaling", "ECM-Receptor", "Non-protein Signaling"]
                )
                if diffusible_mask.any():
                    nLR1 = int(np.flatnonzero(diffusible_mask).max()) + 1  # 1-based count
                else:
                    nLR1 = 0
        else:
            nLR1 = nLR

    # ---------------------------------------------------------------------
    # R lines 175-185: ligand / receptor expression + coreceptor adjustments
    # ---------------------------------------------------------------------
    dataLavg = computeExpr_LR(geneL, data_use_df, complex_input)
    dataRavg = computeExpr_LR(geneR, data_use_df, complex_input)

    dataR_co_A = computeExpr_coreceptor(cofactor_input, data_use_df, pairLRsig, type="A")
    dataR_co_I = computeExpr_coreceptor(cofactor_input, data_use_df, pairLRsig, type="I")
    dataRavg = dataRavg * dataR_co_A / dataR_co_I

    # ---------------------------------------------------------------------
    # R lines 192-275: per-LR-pair sparse cell-cell probability
    # ---------------------------------------------------------------------
    P_spatial_csc = P_spatial.tocsc()
    prob_cell_list: List[sparse.csc_matrix] = []
    for i in range(nLR):
        # crossprod(L_row, R_row)
        dataLR = _crossprod_outer_sparse(dataLavg[i, :], dataRavg[i, :])

        # Hill function
        P1 = HillFunctionFordataLR(dataLR, Kh=Kh, n=n)

        # P1 * P.spatial (elementwise sparse multiply)
        P1_Pspatial = P1.multiply(P_spatial_csc)

        if not use_AGAN:
            # R lines 211-225 — still apply contact mask for cell-cell
            # contact pairs (i.e. i > nLR1 in 1-based; i >= nLR1 in 0-based).
            if i >= nLR1:
                P1_Pspatial = P1_Pspatial.multiply(adj_contact)
            prob_cell_list.append(P1_Pspatial.tocsc())
            continue

        # use_AGAN = True
        if i >= nLR1:
            P1_Pspatial = P1_Pspatial.multiply(adj_contact)

        # If P1_Pspatial is all-zero, terminate early.
        P1_Pspatial = P1_Pspatial.tocsc()
        if P1_Pspatial.nnz == 0 or float(P1_Pspatial.sum()) == 0.0:
            prob_cell_list.append(P1_Pspatial)
            continue

        # data.agonist -> P2
        data_agonist = computeExpr_agonist(
            data_use_df, pairLRsig, cofactor_input,
            index_agonist=i, Kh=Kh, n=n,
        )
        P_ = myElementwiseProduct(P1_Pspatial, data_agonist.ravel())

        # data.antagonist -> P3
        data_antag = computeExpr_antagonist(
            data_use_df, pairLRsig, cofactor_input,
            index_antagonist=i, Kh=Kh, n=n,
        )
        Pnull_cell = myElementwiseProduct(P_, data_antag.ravel())
        prob_cell_list.append(Pnull_cell.tocsc())

    # ---------------------------------------------------------------------
    # R lines 277-318: assemble the 3-D sparse array and write back.
    # ---------------------------------------------------------------------
    prob_cell_3d = _list_to_sparse3D(prob_cell_list, (nC, nC, nLR))

    pair_names = list(pairLRsig.index.astype(str))
    cell_names = list(adata.obs_names.astype(str))

    cc.setdefault("net", {})
    cc["net"]["prob.cell"] = prob_cell_3d
    cc["net"]["tmp"] = {
        "prob.cell": dict(zip(pair_names, prob_cell_list)),
        "Lavg": dataLavg,
        "Ravg": dataRavg,
    }
    cc["net"]["dimnames"] = {
        "cells": cell_names,
        # Mirror both keys so downstream readers may probe either
        # ``dimnames['pairs']`` (legacy) or ``dimnames['interaction_name']``
        # (computeAvgCommunProb-style) without branching.
        "pairs": pair_names,
        "interaction_name": pair_names,
    }
    cc["net"]["meta"] = {
        "cells": cell_names,
        "pairs": pair_names,
        "interaction_name": pair_names,
        "groups": list(group.categories),
        "idents": np.asarray(group),
    }

    cc.setdefault("options", {})
    cc["options"]["parameter"] = {
        "raw.use": bool(raw_use),
        "ratio": float(ratio),
        "tol": float(tol),
        "Kh": float(Kh),
        "n": float(n),
        "nLR": int(nLR),
        "nLR1": int(nLR1),
        "scale.distance": float(scale_distance),
        "use.AGAN": bool(use_AGAN),
        "distance.use": bool(distance_use),
        "interaction.range": float(interaction_range),
        "contact.dependent": bool(contact_dependent),
        "contact.range": float(contact_range),
        "contact.dependent.forced": bool(contact_dependent_forced),
        "all.contact.dependent": bool(all_contact_dependent),
        "all.diffusible": bool(all_diffusible),
    }
    cc.setdefault("images", {})
    cc["images"]["result.computeCellDistance"] = {
        "d.spatial": d_spatial,
        "adj.contact": adj_contact,
    }
    return adata


# ---------------------------------------------------------------------------
# computeCommunProbPathway (R: modeling.R:1517-1645)
# ---------------------------------------------------------------------------
def computeCommunProbPathway(
    adata: ad.AnnData,
    pairLR_use: Optional[pd.DataFrame] = None,
    thresh: float = 0.05,
    do_group: bool = True,
    do_cell: bool = True,
) -> ad.AnnData:
    """Aggregate the L-R-pair communication net to pathway-level.

    Direct port of R's :func:`computeCommunProbPathway`
    (``modeling.R:1517-1645``). The function reads
    ``adata.uns["cellchat"]["net"]`` and writes the pathway-level result
    to ``adata.uns["cellchat"]["netP"]``.

    Two parallel pathways exist:

    - ``do_group=True``: aggregate the dense ``net["prob"]`` (and use
      ``net["pval"]`` as the significance mask) by pathway. Sub-agent B
      writes ``net["prob"]`` and ``net["pval"]``; this branch is a no-op
      when those keys are missing.
    - ``do_cell=True``: aggregate the sparse ``net["prob.cell"]`` /
      ``net["tmp"]["prob.cell"]`` (set by :func:`computeCommunProb`) by
      pathway, sum across L-R pairs that share a pathway, and order by
      total cell-level signaling.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"]["net"]`` populated.
    pairLR_use : pandas.DataFrame, optional
        L-R table with ``pathway_name`` column. Defaults to
        ``uns["cellchat"]["LR"]["LRsig"]``.
    thresh : float, default ``0.05``
        Significance threshold (used only by the ``do_group`` branch).
    do_group : bool, default ``True``
    do_cell : bool, default ``True``

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    cc = _ensure_uns(adata)
    if "net" not in cc:
        raise ValueError(
            "computeCommunProbPathway requires uns['cellchat']['net']; run "
            "computeCommunProb first"
        )
    net = cc["net"]
    if pairLR_use is None:
        pairLR_use = _get_LR(adata, None)

    netP: Dict[str, Any] = {}

    # -------------------- group branch -----------------------------------
    if do_group:
        if "prob" not in net:
            raise ValueError(
                "computeCommunProbPathway(do_group=True) requires net['prob']; "
                "run computeAvgCommunProb first"
            )
        prob = np.asarray(net["prob"], dtype=float).copy()
        pval = np.asarray(net["pval"], dtype=float)
        prob[pval >= thresh] = 0.0

        pathways_unique = list(dict.fromkeys(pairLR_use["pathway_name"].astype(str).tolist()))
        # group factor by pathway
        ann = pairLR_use["pathway_name"].astype(str).values
        # Sum prob across LR-pairs sharing the same pathway -> shape
        # (nGroups, nGroups, nPathways)
        nGroups = prob.shape[0]
        prob_pathways = np.zeros((nGroups, nGroups, len(pathways_unique)), dtype=float)
        for k, pw in enumerate(pathways_unique):
            idx = np.flatnonzero(ann == pw)
            if idx.size > 0:
                prob_pathways[:, :, k] = prob[:, :, idx].sum(axis=2)

        # Pathways with non-zero total
        sums = prob_pathways.sum(axis=(0, 1))
        sig_mask = sums != 0.0
        pathways_sig = [pathways_unique[k] for k in np.flatnonzero(sig_mask)]
        prob_sig = prob_pathways[:, :, sig_mask]
        # Sort decreasing by total signaling
        sums_sig = prob_sig.sum(axis=(0, 1))
        sort_idx = np.argsort(-sums_sig, kind="stable")
        pathways_sig = [pathways_sig[i] for i in sort_idx]
        prob_sig = prob_sig[:, :, sort_idx]

        netP["pathways"] = pathways_sig
        netP["prob"] = prob_sig
    else:
        netP["pathways"] = None
        netP["prob"] = None

    # -------------------- cell branch ------------------------------------
    if do_cell and "prob.cell" in net:
        prob_cell = net["prob.cell"]
        prob_cell_dict: Dict[str, sparse.csc_matrix] = net["tmp"]["prob.cell"]
        pair_names = list(prob_cell_dict.keys())

        pathways_unique = list(dict.fromkeys(pairLR_use["pathway_name"].astype(str).tolist()))
        ann = pairLR_use["pathway_name"].astype(str).values

        nC = prob_cell.shape[0]
        # For each pathway, sum CSC matrices across LR pairs sharing it.
        prob_all: Dict[str, sparse.csc_matrix] = {}
        for pw in pathways_unique:
            idx = np.flatnonzero(ann == pw)
            if idx.size == 0:
                prob_all[pw] = sparse.csc_matrix((nC, nC), dtype=float)
                continue
            # marginSumsSparse(arr, MARGIN = c(1,2)) on the (nC, nC, len(idx))
            # subarray. We sum the matched CSC matrices.
            acc = sparse.csc_matrix((nC, nC), dtype=float)
            for k in idx:
                # Use the pair name to look up the dict (matches R's named-list indexing)
                acc = acc + prob_cell_dict[pair_names[k]]
            prob_all[pw] = acc.tocsc()

        # prob.sum: per pathway, sum of all entries
        prob_sum = np.array([float(M.sum()) for M in prob_all.values()])
        keep_idx = np.flatnonzero(prob_sum > 0)
        sort_idx = keep_idx[np.argsort(-prob_sum[keep_idx], kind="stable")]
        pathways_sig_cell = [pathways_unique[k] for k in sort_idx]
        prob_cell_sig_dict = {pw: prob_all[pw] for pw in pathways_sig_cell}

        # Stack
        nLR_path = len(pathways_sig_cell)
        prob_cell_sig = _list_to_sparse3D(
            list(prob_cell_sig_dict.values()), (nC, nC, nLR_path)
        )
        netP["pathways.cell"] = pathways_sig_cell
        netP["prob.cell"] = prob_cell_sig
        netP["tmp"] = {"prob.cell": prob_cell_sig_dict}

    # R: ``cellchat@netP$prob`` and ``cellchat@netP$prob.cell`` inherit
    # dimnames from ``cellchat@net$prob`` / ``cellchat@net$prob.cell``.
    # In this Python port the arrays carry no intrinsic dimnames, so mirror
    # the relevant keys onto ``cc['netP']['dimnames']`` and the centrality
    # / heatmap visualisations can recover real cell-group / cell-barcode
    # labels without falling back to integer placeholders.
    net_dimnames = net.get("dimnames", {}) if isinstance(net, dict) else {}
    netP_dimnames: Dict[str, Any] = {}
    for k in ("source", "target", "cells"):
        if k in net_dimnames:
            netP_dimnames[k] = list(net_dimnames[k])
    if do_group and netP.get("pathways") is not None:
        netP_dimnames["pathways"] = list(netP["pathways"])
    if do_cell and "prob.cell" in net and netP.get("pathways.cell") is not None:
        netP_dimnames["pathways.cell"] = list(netP["pathways.cell"])
    if netP_dimnames:
        netP["dimnames"] = netP_dimnames

    cc["netP"] = netP
    return adata


# ---------------------------------------------------------------------------
# aggregateNet (R: modeling.R:2077-2146)
# ---------------------------------------------------------------------------
def aggregateNet(
    adata: ad.AnnData,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR_use: Optional[pd.DataFrame] = None,
    remove_isolate: bool = True,
    thresh: float = 0.05,
    return_object: bool = True,
) -> Union[ad.AnnData, Dict[str, Any]]:
    """Aggregate the L-R-pair net into per-cell-pair count / weight matrices.

    Direct port of R's :func:`aggregateNet` (``modeling.R:2077-2146``).
    Writes ``net["count"]``, ``net["weight"]``, ``net["LR.sig"]`` and (if
    ``net["prob.cell"]`` is present) ``net["count.cell"]``,
    ``net["weight.cell"]``, ``net["LR.sig.cell"]`` into
    ``adata.uns["cellchat"]["net"]``.
    """
    cc = _ensure_uns(adata)
    if "net" not in cc:
        raise ValueError("aggregateNet requires uns['cellchat']['net']")
    net = cc["net"]

    # ------------ no-filter branch (R lines 2079-2088) -------------------
    if sources_use is None and targets_use is None and signaling is None and pairLR_use is None:
        if "prob" not in net:
            raise ValueError(
                "aggregateNet (no-filter branch) requires net['prob']; run "
                "computeAvgCommunProb first or pass sources_use/targets_use/"
                "signaling/pairLR_use to take the long-table branch"
            )
        prob = np.asarray(net["prob"], dtype=float).copy()
        # When `computeAvgCommunProb` was run with `do_permutation=False`,
        # `net["pval"]` is `None` (mirrors R's "no permutation = no p-values")
        # — treat as "all 1" (no significance gating applied).
        pval_raw = net.get("pval", None)
        if pval_raw is None:
            pval = np.ones_like(prob)
        else:
            pval = np.asarray(pval_raw, dtype=float).copy()
        pval[prob == 0] = 1.0
        prob[pval >= thresh] = 0.0

        net["count"] = np.sum(prob > 0, axis=2).astype(float)
        weight = np.sum(prob, axis=2)
        weight[np.isnan(weight)] = 0.0
        net["weight"] = weight
        net["count"][np.isnan(net["count"])] = 0.0
        # LR.sig: dim-3 names whose total prob > 0
        # The 3rd-dim names key was originally "pairs" (set by
        # computeCommunProb), but computeAvgCommunProb / *_Visium overwrites
        # the dimnames dict with R-style "interaction_name" keys; check both.
        meta = cc.get("net", {}).get("dimnames", {})
        pair_names = meta.get("interaction_name", meta.get("pairs", None))
        if pair_names is None:
            pair_names = [str(k) for k in range(prob.shape[2])]
        sums3 = prob.sum(axis=(0, 1))
        net["LR.sig"] = [pair_names[k] for k in np.flatnonzero(sums3 > 0)]
    else:
        # ------------ filter / long-table branch (R lines 2090-2118) -----
        df_net = subsetCommunication(
            adata, slot_name="net",
            sources_use=sources_use, targets_use=targets_use,
            signaling=signaling, pairLR_use=pairLR_use, thresh=thresh,
        )
        df_net = df_net.copy()
        df_net["source_target"] = df_net["source"].astype(str) + "_" + df_net["target"].astype(str)
        # ``df.net %>% group_by(source_target) %>% summarize(count = n())``
        grp_count = df_net.groupby("source_target", sort=False).size().rename("count")
        grp_prob = df_net.groupby("source_target", sort=False)["prob"].sum().rename("prob")
        df_net2 = pd.concat([grp_count, grp_prob], axis=1).reset_index()
        a = df_net2["source_target"].str.split("_", n=1, expand=True)
        df_net2["source"] = a.iloc[:, 0]
        df_net2["target"] = a.iloc[:, 1]
        cells_level = list(_get_idents(adata, None).categories)
        if remove_isolate:
            present_src = [c for c in cells_level if c in set(df_net2["source"])]
            present_tgt = [c for c in cells_level if c in set(df_net2["target"])]
        else:
            present_src = cells_level
            present_tgt = cells_level
        df_net2["source"] = pd.Categorical(df_net2["source"], categories=present_src)
        df_net2["target"] = pd.Categorical(df_net2["target"], categories=present_tgt)

        count = pd.crosstab(
            df_net2["source"], df_net2["target"],
            values=df_net2["count"], aggfunc="sum",
        ).reindex(index=present_src, columns=present_tgt, fill_value=np.nan)
        prob = pd.crosstab(
            df_net2["source"], df_net2["target"],
            values=df_net2["prob"], aggfunc="sum",
        ).reindex(index=present_src, columns=present_tgt, fill_value=np.nan)
        count_arr = count.fillna(0).to_numpy(dtype=float)
        prob_arr = prob.fillna(0).to_numpy(dtype=float)
        net["count"] = count_arr
        net["weight"] = prob_arr

    # ------------ prob.cell branch (R lines 2120-2136) -------------------
    if "prob.cell" in net:
        prob_cell = net["prob.cell"]  # sparse.COO

        # binary -> count.cell = sum over dim-3 of indicator
        nz = prob_cell.data != 0
        coords = prob_cell.coords[:, nz]
        # collapse over dim 3: sum 1 per non-zero entry (i, j) regardless of k
        # build per-(i, j) count
        nC = prob_cell.shape[0]
        idx = coords[0] * nC + coords[1]
        bins = np.bincount(idx, minlength=nC * nC)
        count_cell = bins.reshape(nC, nC).astype(float)
        net["count.cell"] = sparse.csc_matrix(count_cell)

        # weight.cell = sum of prob over dim-3
        weight_cell_dense = np.zeros((nC, nC), dtype=float)
        for k_layer in range(prob_cell.shape[2]):
            mask = coords[2] == k_layer  # not used; we already have all data
        # simpler: marginal sum over axis=2
        if prob_cell.nnz > 0:
            weight_cell_arr = prob_cell.sum(axis=2).todense()
            weight_cell_dense = np.asarray(weight_cell_arr, dtype=float)
        net["weight.cell"] = sparse.csc_matrix(weight_cell_dense)

        # LR.sig.cell — pair names where dim-3 sum > 0
        if prob_cell.nnz > 0:
            sums3 = prob_cell.sum(axis=(0, 1)).todense()
            sums3 = np.asarray(sums3).ravel()
        else:
            sums3 = np.zeros(prob_cell.shape[2], dtype=float)
        meta = net.get("dimnames", {})
        pair_names = meta.get(
            "interaction_name",
            meta.get("pairs", [str(k) for k in range(prob_cell.shape[2])]),
        )
        net["LR.sig.cell"] = [pair_names[k] for k in np.flatnonzero(sums3 > 0)]

    if return_object:
        return adata
    return net


# ---------------------------------------------------------------------------
# filterCommunication (R: modeling.R:1709-1871)
# ---------------------------------------------------------------------------
def filterCommunication(
    adata: ad.AnnData,
    min_cells: Optional[int] = 10,
    min_links: Optional[int] = 5,
    min_cells_sr: Optional[int] = 5,
) -> ad.AnnData:
    """In-place: zero-out cell-cell communications failing min-count thresholds.

    Direct port of R's :func:`filterCommunication` (``modeling.R:1709``).

    Parameters
    ----------
    adata : anndata.AnnData
    min_cells : int, optional, default ``10``
        If not ``None``, cell-group level communications involving any
        group with fewer than ``min_cells`` cells are zeroed
        (``net["prob"][cell.excludes, ...] <- 0`` and same for the second
        axis). When ``net["prob"]`` is missing this branch is silently
        skipped; the cell-level branch is still run.
    min_links : int, optional, default ``5``
        Drop L-R pairs whose ``prob.cell`` matrix has fewer than
        ``min_links`` non-zero entries (replaced by an all-zero CSC).
    min_cells_sr : int, optional, default ``5``
        Drop L-R pairs whose ``prob.cell`` matrix has fewer than
        ``min_cells_sr`` cells acting as senders or receivers.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    cc = _ensure_uns(adata)
    net = cc.get("net", None)
    if net is None:
        raise ValueError("filterCommunication requires uns['cellchat']['net']")

    # -------------- min.cells branch (R lines 1711-1727) -----------------
    if min_cells is not None and "prob" in net:
        idents = _get_idents(adata, None)
        counts_per_group = pd.Series(idents).value_counts().reindex(idents.categories, fill_value=0)
        cell_excludes = np.flatnonzero(counts_per_group.to_numpy() < int(min_cells))
        if cell_excludes.size > 0:
            net["prob"][cell_excludes, :, :] = 0.0
            net["prob"][:, cell_excludes, :] = 0.0
            if "pval" in net and net["pval"] is not None:
                net["pval"][net["prob"] == 0] = 1.0

    # -------------- prob.cell branch (R lines 1729-1869) -----------------
    if "prob.cell" not in net:
        raise ValueError(
            "filterCommunication: please run computeCommunProb to compute the "
            "communication probability/strength between any interacting individual cells"
        )
    prob_cell_dict: Dict[str, sparse.csc_matrix] = net["tmp"]["prob.cell"]
    pair_names = list(prob_cell_dict.keys())

    if min_links is None and min_cells_sr is None:
        return adata

    # prob.sum: number of stored values per matrix
    prob_sum = np.array([M.nnz for M in prob_cell_dict.values()])
    nC = next(iter(prob_cell_dict.values())).shape[0] if prob_cell_dict else 0
    AllzeroMat = sparse.csc_matrix((nC, nC), dtype=float)

    if min_links is not None:
        idx_excludes = np.flatnonzero((prob_sum < int(min_links)) & (prob_sum > 0))
        for x in idx_excludes:
            prob_cell_dict[pair_names[x]] = AllzeroMat.copy()
        # update prob_sum
        prob_sum = np.array([M.nnz for M in prob_cell_dict.values()])

    if min_cells_sr is not None:
        if min_links is None:
            mask_use = prob_sum > 0
        else:
            mask_use = prob_sum >= max(1, int(min_links))
        pathways_idx = np.flatnonzero(mask_use)
        pathways_remove_idx: List[int] = []
        for x in pathways_idx:
            M = prob_cell_dict[pair_names[x]]
            # logical sparse: M > 0
            bin_M = (M > 0)
            row_sum = np.asarray(bin_M.sum(axis=1)).ravel()
            col_sum = np.asarray(bin_M.sum(axis=0)).ravel()
            n_senders = int(np.sum(row_sum > 0))
            n_receivers = int(np.sum(col_sum > 0))
            if n_senders < int(min_cells_sr) or n_receivers < int(min_cells_sr):
                pathways_remove_idx.append(int(x))
        for x in pathways_remove_idx:
            prob_cell_dict[pair_names[x]] = AllzeroMat.copy()

    # Rebuild the 3-D sparse array
    nLR = len(pair_names)
    net["prob.cell"] = _list_to_sparse3D(list(prob_cell_dict.values()), (nC, nC, nLR))
    net["tmp"]["prob.cell"] = prob_cell_dict
    return adata


# ---------------------------------------------------------------------------
# scMatrixTruncation (R: modeling.R:1932-1961)
# ---------------------------------------------------------------------------
def scMatrixTruncation(
    Mat: sparse.spmatrix,
    cutoff: Optional[float] = None,
    remain_cutoff_v: bool = True,
    repr: Literal["C", "T"] = "C",
) -> sparse.spmatrix:
    """Truncate a sparse expression matrix to keep ``v >= cutoff``.

    Mirrors R's :func:`scMatrixTruncation` (``modeling.R:1932``). When
    ``cutoff`` is ``None`` the matrix is returned (re-)cast to the
    requested storage format unchanged.

    Parameters
    ----------
    Mat : scipy.sparse.spmatrix
        Sparse matrix (R's ``dgCMatrix`` / ``dgTMatrix``).
    cutoff : float, optional
    remain_cutoff_v : bool, default ``True``
        ``True`` -> keep entries with value ``>=`` cutoff;
        ``False`` -> keep entries with value ``>`` cutoff.
    repr : {"C", "T"}, default ``"C"``

    Returns
    -------
    scipy.sparse.spmatrix
        The truncated sparse matrix in the requested storage format
        (``csc_matrix`` for ``"C"``, ``coo_matrix`` for ``"T"``).
    """
    if not sparse.issparse(Mat):
        raise TypeError(
            "Please use `dgCMatrix` or `dgTMatrix` as a Matrix input!"
        )
    out: sparse.spmatrix = Mat
    if cutoff is not None and isinstance(cutoff, (int, float)) and Mat.nnz > 0:
        coo = Mat.tocoo()
        if remain_cutoff_v:
            keep = coo.data >= float(cutoff)
        else:
            keep = coo.data > float(cutoff)
        coo = sparse.coo_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=coo.shape,
        )
        out = coo

    if repr == "C":
        return out.tocsc()
    if repr == "T":
        return out.tocoo()
    raise ValueError("repr must be 'C' or 'T'")


# ---------------------------------------------------------------------------
# filterProbability (R: modeling.R:1976-2054)
# ---------------------------------------------------------------------------
def filterProbability(
    adata: ad.AnnData,
    nboot: int = 100,
    seed_use: int = 666,
    thresh: float = 0.05,
) -> ad.AnnData:
    """Filter individual cell-cell communications via a per-LR null quantile.

    Direct port of R's :func:`filterProbability` (``modeling.R:1976-2054``).
    For every L-R pair: draw ``nboot`` cells without replacement, gather
    the per-cell ``Prob.cell.i`` row at columns where ``d.spatial[cell, ] > 0``,
    take the ``(1 - thresh)`` quantile of the pooled values, then truncate
    ``Prob.cell.i`` so that only entries above the quantile survive.

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``uns["cellchat"]["net"]["tmp"]["prob.cell"]`` list
        was written by :func:`computeCommunProb`.
    nboot : int, default ``100``
    seed_use : int, default ``666``
    thresh : float, default ``0.05``

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    from ._bootstrap import make_cell_permutation

    quantile_prob = 1.0 - float(thresh)
    if quantile_prob == 0.0:
        return adata
    cc = _ensure_uns(adata)
    net = cc.get("net", None)
    if net is None or "tmp" not in net or "prob.cell" not in net["tmp"]:
        raise ValueError(
            "filterProbability: please run computeCommunProb to compute the "
            "communication probability/strength between any interacting individual cells"
        )
    prob_cell_dict: Dict[str, sparse.csc_matrix] = net["tmp"]["prob.cell"]
    pair_names = list(prob_cell_dict.keys())
    cell_names = net.get("dimnames", {}).get("cells", list(adata.obs_names.astype(str)))

    images = cc.get("images", {})
    res = images.get("result.computeCellDistance", None)
    if res is None:
        raise ValueError(
            "filterProbability requires uns['cellchat']['images']"
            "['result.computeCellDistance'] (set by computeCommunProb)"
        )
    d_spatial = res["d.spatial"].copy().tolil()
    # Matrix::diag(d_spatial) <- 1
    nC = d_spatial.shape[0]
    for k in range(nC):
        d_spatial[k, k] = 1.0
    d_spatial = d_spatial.tocsc()
    adj_contact = res["adj.contact"]

    params = cc.get("options", {}).get("parameter", {})
    nLR = int(params.get("nLR", len(pair_names)))
    nLR1 = int(params.get("nLR1", nLR))
    all_contact_dependent = bool(params.get("all.contact.dependent", False))

    if all_contact_dependent:
        d_spatial = adj_contact.copy().tocsc()

    perm = make_cell_permutation(nC=nC, nLR=nLR, nboot=int(nboot), seed_use=int(seed_use))

    new_dict: Dict[str, sparse.csc_matrix] = {}
    for i in range(nLR):
        if i < nLR1:
            d_use = d_spatial
        else:
            d_use = adj_contact.tocsc()
        sample_cells = perm[:, i]
        Prob_cell_i = prob_cell_dict[pair_names[i]]
        # Gather Prob.cell.i[cell.index, prob.index] for each sample cell
        sample_vals: List[np.ndarray] = []
        Prob_csr = Prob_cell_i.tocsr()
        d_use_csr = d_use.tocsr()
        for cell_index in sample_cells:
            row = d_use_csr.getrow(int(cell_index))
            prob_index = np.asarray(row.toarray()).ravel() > 0
            cols = np.flatnonzero(prob_index)
            if cols.size == 0:
                continue
            row_vals = np.asarray(Prob_csr.getrow(int(cell_index)).toarray()).ravel()
            sample_vals.append(row_vals[cols])
        if not sample_vals:
            new_dict[pair_names[i]] = Prob_cell_i
            continue
        pooled = np.concatenate(sample_vals)
        nboot_quantile = float(np.quantile(pooled, quantile_prob))
        if nboot_quantile == 0.0:
            new_dict[pair_names[i]] = Prob_cell_i
        else:
            new_dict[pair_names[i]] = scMatrixTruncation(
                Prob_cell_i, cutoff=nboot_quantile, remain_cutoff_v=True, repr="C"
            )

    net["tmp"]["prob.cell"] = new_dict
    net["prob.cell"] = _list_to_sparse3D(list(new_dict.values()), (nC, nC, len(pair_names)))
    return adata


# ---------------------------------------------------------------------------
# subsetCommunication (R: analysis.R:2266-2532)
# ---------------------------------------------------------------------------
def subsetCommunication(
    adata: ad.AnnData,
    net: Optional[Dict[str, Any]] = None,
    slot_name: Literal["net", "netP"] = "net",
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR_use: Optional[pd.DataFrame] = None,
    thresh: float = 0.05,
) -> pd.DataFrame:
    """Convert ``net`` / ``netP`` into a tidy long-table data frame.

    Direct port of R's :func:`subsetCommunication` (``analysis.R:2266``)
    for the ``mode == "single"`` branch — the only branch SpatialCellChat
    Python supports (single AnnData; merged objects are out-of-scope per
    the porting contract).

    Parameters
    ----------
    adata : anndata.AnnData
    net : dict, optional
        If supplied, used directly; otherwise read from
        ``adata.uns["cellchat"]["net"]`` (or ``["netP"]`` when
        ``slot_name="netP"``).
    slot_name : {"net", "netP"}
    sources_use, targets_use : sequence, optional
        Cell-group restrictions (R: ``sources.use`` / ``targets.use``).
    signaling : sequence of str, optional
        Pathway names to filter on (R: ``signaling``).
    pairLR_use : pandas.DataFrame, optional
        Direct LR pair restriction (R: ``pairLR.use``).
    thresh : float, default ``0.05``

    Returns
    -------
    pandas.DataFrame
        Long-table of inferred communications. Columns include
        ``source``, ``target``, ``ligand``, ``receptor``, ``prob``,
        ``pval``, ``interaction_name``, ``interaction_name_2``,
        ``pathway_name``, ``annotation``, ``evidence``.
    """
    cc = _ensure_uns(adata)
    if pairLR_use is not None:
        if not isinstance(pairLR_use, pd.DataFrame):
            raise ValueError(
                "pairLR.use should be a data frame with a single column named "
                "either 'interaction_name' or 'pathway_name'"
            )
        if "pathway_name" in pairLR_use.columns:
            slot_name = "netP"

    if pairLR_use is not None and signaling is not None:
        raise ValueError(
            "Please do not assign values to 'signaling' when using 'pairLR.use'"
        )

    if net is None:
        if slot_name not in cc:
            raise ValueError(f"adata.uns['cellchat']['{slot_name}'] missing")
        net = cc[slot_name]
    LR_table = _get_LR(adata, None)
    cells_level = list(_get_idents(adata, None).categories)

    # Reshape prob (and pval) into a long DataFrame.
    # R parity (analysis.R:2378-2387):
    #   df.net <- reshape2::melt(prob, value.name = "prob")
    #   colnames(df.net)[1:3] <- c("source", "target", "interaction_name")
    #   df.net$pval <- c(pval)
    #   df.net <- subset(df.net, prob > 0)
    # i.e. fully vectorised melt + subset. Audit G4 flagged the prior
    # ``for k: for i: for j:`` triple loop as a performance defect
    # (correct output, but n_cell² × n_LR Python-loop iterations on real
    # data take seconds vs. ms in R). Replace with a numpy ``np.where``
    # + frame-from-arrays construction; ordering is k-major (R's melt
    # default), which downstream groupby / merge is invariant to.
    prob = np.asarray(net["prob"], dtype=float).copy()
    pval = np.asarray(net["pval"], dtype=float).copy()
    prob[pval >= thresh] = 0.0
    nGroups = prob.shape[0]
    nLR_table = prob.shape[2]
    pair_names = list(LR_table.index.astype(str))

    cell_level_arr = np.asarray(cells_level, dtype=object)
    pair_name_arr = np.asarray(pair_names, dtype=object)

    # ``np.transpose(prob, (2, 0, 1))`` gives k-major indexing matching
    # R's ``reshape2::melt`` (which iterates the FIRST axis fastest;
    # after melt, source then target then LR). For our purposes the
    # downstream pipeline (groupby on source_target × pathway_name,
    # merge on interaction_name) is invariant to row order, so we only
    # need to make sure (i, j, k) entries with prob>0 land in df.
    nonzero = np.asarray(prob > 0).nonzero()
    if nonzero[0].size == 0:
        df = pd.DataFrame(columns=["source", "target", "interaction_name",
                                   "prob", "pval"])
    else:
        i_idx, j_idx, k_idx = nonzero
        df = pd.DataFrame({
            "source": cell_level_arr[i_idx],
            "target": cell_level_arr[j_idx],
            "interaction_name": pair_name_arr[k_idx],
            "prob": prob[i_idx, j_idx, k_idx],
            "pval": pval[i_idx, j_idx, k_idx],
        })
    if df.empty:
        raise ValueError(
            "No significant signaling interactions are inferred based on the input!"
        )

    # Join LR_table extra columns
    if "ligand" not in df.columns:
        extra_cols = [
            c for c in ("interaction_name_2", "pathway_name", "ligand", "receptor",
                        "annotation", "evidence")
            if c in LR_table.columns
        ]
        extras = LR_table.loc[:, extra_cols].copy()
        extras.index = pd.Index(extras.index.astype(str))
        df = df.merge(extras, left_on="interaction_name", right_index=True, how="left")

    # Apply signaling / pairLR.use filter
    if signaling is not None:
        sig_pairs: List[pd.DataFrame] = []
        for sig in signaling:
            sig_pairs.append(searchPair(
                signaling=sig, pairLR_use=LR_table,
                key="pathway_name", matching_exact=True, pair_only=True,
            ))
        pairLR_use = pd.concat(sig_pairs, axis=0)

    if pairLR_use is not None:
        # Mirror R's ``tryCatch(subset(net, interaction_name %in%
        # pairLR.use$interaction_name), error = subset(net, pathway_name %in%
        # pairLR.use$pathway_name))`` (analysis.R:2402-2408).
        #
        # In R, ``pairLR.use$interaction_name`` returns NULL when the column
        # is absent — ``%in% NULL`` evaluates to ``logical(0)`` and ``subset``
        # then returns an empty frame WITHOUT raising.  The pathway_name
        # fallback only fires on a real error (e.g., ``net`` itself lacks
        # ``interaction_name``).  Mirror that with explicit NULL semantics:
        # if the result df has the canonical column but pairLR_use lacks
        # it, return an empty frame; let the alternative column fallback
        # only fire when the result df itself has neither.
        if "interaction_name" in df.columns:
            if "interaction_name" in pairLR_use.columns:
                keep = df["interaction_name"].isin(pairLR_use["interaction_name"])
            else:
                # R: ``%in% NULL`` → all-FALSE → empty subset (no error).
                keep = pd.Series(False, index=df.index)
        elif "pathway_name" in df.columns:
            if "pathway_name" in pairLR_use.columns:
                keep = df["pathway_name"].isin(pairLR_use["pathway_name"])
            else:
                keep = pd.Series(False, index=df.index)
        else:
            # Neither key in df → R's outer ``tryCatch`` error path.
            raise ValueError(
                "subsetCommunication: result frame has neither "
                "'interaction_name' nor 'pathway_name' column"
            )
        df = df.loc[keep]

    # Drop all-NA rows
    df = df.loc[~df.isna().all(axis=1)]
    if df.empty:
        raise ValueError(
            "No significant signaling interactions are inferred based on the input!"
        )

    if slot_name == "netP":
        df = df.loc[:, ["source", "target", "pathway_name", "prob", "pval", "annotation"]]
        df["source_target"] = df["source"].astype(str) + "sourceTotarget" + df["target"].astype(str)
        gpval = df.groupby(["source_target", "pathway_name"], sort=False)["pval"].mean().reset_index()
        gprob = df.groupby(["source_target", "pathway_name"], sort=False)["prob"].sum().reset_index()
        df = gprob.copy()
        a = df["source_target"].str.split("sourceTotarget", n=1, expand=True)
        df["source"] = a.iloc[:, 0]
        df["target"] = a.iloc[:, 1]
        df = df.drop(columns="source_target")
        df["pval"] = gpval["pval"].values

    if sources_use is not None:
        if all(isinstance(s, (int, np.integer)) for s in sources_use):
            sources_use = [cells_level[int(s)] for s in sources_use]
        df = df.loc[df["source"].isin(list(sources_use))]
    if targets_use is not None:
        if all(isinstance(t, (int, np.integer)) for t in targets_use):
            targets_use = [cells_level[int(t)] for t in targets_use]
        df = df.loc[df["target"].isin(list(targets_use))]

    df = df.reset_index(drop=True)
    if df.empty:
        # R warns instead of erroring here; we follow R.
        return df

    if slot_name == "net":
        cols = [
            "source", "target", "ligand", "receptor", "prob", "pval",
            "interaction_name", "interaction_name_2", "pathway_name",
            "annotation", "evidence",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df.loc[:, cols]
    elif slot_name == "netP":
        df = df.loc[:, ["source", "target", "pathway_name", "prob", "pval"]]

    return df


# ---------------------------------------------------------------------------
# identifyEnrichedInteractions (R: modeling.R:2613-2683)
# ---------------------------------------------------------------------------
def identifyEnrichedInteractions(
    adata: ad.AnnData,
    from_: Sequence,
    to: Sequence,
    bidirection: bool = False,
    pair_only: bool = True,
    pairLR_use0: Optional[pd.DataFrame] = None,
    thresh: float = 0.05,
) -> pd.DataFrame:
    """Identify all significant interactions between specified cell groups.

    Direct port of R's :func:`identifyEnrichedInteractions`
    (``modeling.R:2613-2683``). Reads
    ``adata.uns["cellchat"]["net"]["pairwiseRank"]`` (set by
    :func:`rankNetPairwise`).

    Parameters
    ----------
    adata : anndata.AnnData
    from_, to : sequence
        Source / target cell-group indices (1-based int) or names. Must be
        the same length.
    bidirection : bool, default ``False``
        When ``True`` the bidirectional pairs are added.
    pair_only : bool, default ``True``
        Return only ``ligand``/``receptor`` columns.
    pairLR_use0 : pandas.DataFrame, optional
        Pre-computed LR pair table to use; otherwise built from
        ``net["pairwiseRank"][from][to]``.
    thresh : float, default ``0.05``
    """
    cc = _ensure_uns(adata)
    net = cc.get("net", None)
    if net is None or "pairwiseRank" not in net:
        raise ValueError(
            "The interactions between pairwise cell groups have not been "
            "extracted! Please first run rankNetPairwise(adata)"
        )
    pairwiseLR = net["pairwiseRank"]
    group_names_all = list(pairwiseLR.keys())

    def _to_int_indices(x: Sequence, all_names: Sequence[str]) -> List[int]:
        x_list = list(x)
        if all(isinstance(v, (int, np.integer)) for v in x_list):
            return [int(v) - 1 for v in x_list]  # R 1-based -> 0-based
        # Treat as names
        out = []
        for v in x_list:
            if v in all_names:
                out.append(all_names.index(v))
        return out

    if len(list(from_)) != len(list(to)):
        raise ValueError("The length of 'from' and 'to' must be the same!")
    from_idx = _to_int_indices(from_, group_names_all)
    to_idx = _to_int_indices(to, group_names_all)

    # R only emits a soft message when names are missing; we follow it.
    if len(from_idx) != len(to_idx):
        # name-resolution dropped some entries; fall through to R's
        # subsequent comparison. R also issues a message here.
        pass

    if bidirection:
        from2 = list(from_idx) + list(to_idx)
        to_idx = list(to_idx) + list(from_idx)
        from_idx = from2

    if pairLR_use0 is None:
        parts: List[pd.DataFrame] = []
        for i in range(len(from_idx)):
            pairwiseLR_ij = pairwiseLR[group_names_all[from_idx[i]]][group_names_all[to_idx[i]]]
            idx = pairwiseLR_ij["pval"] < float(thresh)
            if idx.any():
                parts.append(pairwiseLR_ij.loc[idx])
        if parts:
            pairLR_use0 = pd.concat(parts, axis=0)
        else:
            pairLR_use0 = pd.DataFrame()

    pval = np.full((pairLR_use0.shape[0], len(from_idx)), np.nan)
    prob = np.full((pairLR_use0.shape[0], len(from_idx)), np.nan)
    group_names: List[str] = []
    for k, (fi, ti) in enumerate(zip(from_idx, to_idx)):
        pairwiseLR_ij = pairwiseLR[group_names_all[fi]][group_names_all[ti]].loc[pairLR_use0.index]
        pval_ij = pairwiseLR_ij["pval"].astype(float).values.copy()
        prob_ij = pairwiseLR_ij["prob"].astype(float).values.copy()
        # discretise pval (R lines 2666-2668)
        pval_ij_d = np.zeros_like(pval_ij)
        pval_ij_d[pval_ij > 0.05] = 1
        pval_ij_d[(pval_ij > 0.01) & (pval_ij <= 0.05)] = 2
        pval_ij_d[pval_ij <= 0.01] = 3
        prob_ij[pval_ij_d == 1] = 0.0
        pval[:, k] = pval_ij_d
        prob[:, k] = prob_ij
        group_names.append(f"{group_names_all[fi]} - {group_names_all[ti]}")

    prob[prob == 0] = np.nan
    keep_rows = ~np.all(np.isnan(prob), axis=1)
    pval = pval[keep_rows, :]
    pairLR_use0 = pairLR_use0.iloc[keep_rows]
    prob = prob[keep_rows, :]
    if pair_only:
        pairLR_use0 = pairLR_use0.loc[:, ["ligand", "receptor"]]
    return pairLR_use0
