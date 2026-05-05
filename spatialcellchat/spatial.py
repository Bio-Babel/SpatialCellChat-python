"""Spatial geometry routines for SpatialCellChat.

Direct ports of the spatial-geometry exports in ``spatial.R`` (lines 200–744)
and ``computeRegionDistance`` (modeling.R:2700). Per the SpatialCellChat
porting contract §3.E, the contact-matrix construction uses
``squidpy.gr.spatial_neighbors`` and the grid construction uses
``geopandas`` + ``shapely`` instead of R's ``sf::st_make_grid`` /
``sf::st_intersects``.

Public functions take an :class:`anndata.AnnData` (or the bare ``coordinates``
matrix when the R reference does so) and write spatial-related results into
``obsp["spatial_distances"]``, ``obsp["spatial_connectivities"]``,
``obsm["X_grid"]`` and ``uns["cellchat"]["spatial"]``.

The R reference (``spatial.R`` lines noted in each docstring) is the
ground truth; correctness has been verified row-by-row against the R
implementation through the harness scripts under ``/tmp/spatial_harness``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as _sg
import squidpy as sq
from scipy import sparse
from sklearn.neighbors import KDTree

from ._anndata import cellchat_dict

__all__ = [
    "computeCellDistance",
    "createCellCellContactMatrixFrom_dspatial",
    "computeRegionDistance",
    "computeColocalization",
    "computeGridSize",
    "makeGridSpatialCellChat",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _coords_to_array(coordinates: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """Coerce ``coordinates`` to a 2-D ``ndarray`` of shape ``(n, 2)``.

    R guards: ``if (ncol(coordinates) == 2) {...} else { stop(...) }``. We
    raise the same error for any input whose second dimension is not 2.
    """
    if isinstance(coordinates, pd.DataFrame):
        arr = coordinates.to_numpy(dtype=float)
    else:
        arr = np.asarray(coordinates, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "Please check the input 'coordinates' and make sure it is a two column matrix."
        )
    return arr


def _ensure_uns(adata: ad.AnnData) -> Dict:
    """Return ``adata.uns['cellchat']`` with the ``spatial`` sub-dict ready.

    Routes through the shared :func:`cellchat_dict` helper for the outer
    slot; ensures the ``spatial`` sub-dict exists for ``computeCellDistance`` /
    ``computeRegionDistance`` writes.
    """
    cc = cellchat_dict(adata)
    if "spatial" not in cc:
        cc["spatial"] = {}
    return cc


# ---------------------------------------------------------------------------
# computeCellDistance (R: spatial.R:541)
# ---------------------------------------------------------------------------
def computeCellDistance(
    coordinates: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    interaction_range: float = 250.0,
    contact_range: float = 10.0,
    ratio: Optional[float] = None,
    tol: float = 10.0 / 2,
) -> Dict[str, sparse.csc_matrix]:
    """Compute pairwise cell-cell distance and the contact adjacency matrix.

    Mirrors R's :func:`computeCellDistance` (``spatial.R:541``). For each
    column of the dense Euclidean distance matrix produced by
    ``Rfast::Dist`` we keep the strictly-positive entries that are at most
    ``(interaction.range + tol) / ratio`` (the `interaction.range` argument
    in R is in microns; ``ratio`` converts to the unit used by the input
    coordinates). The result is then optionally re-scaled by ``ratio`` so
    the stored values are in microns again, before
    :func:`createCellCellContactMatrixFrom_dspatial` derives the binary
    contact mask.

    Parameters
    ----------
    coordinates : numpy.ndarray, pandas.DataFrame or anndata.AnnData
        Either a ``(n, 2)`` matrix of cell coordinates (rows are cells) or
        an ``AnnData`` whose ``obsm["spatial"]`` holds those coordinates.
    interaction_range : float, default ``250``
        Maximum interaction range, in microns (R: ``interaction.range``).
    contact_range : float, default ``10``
        Cell-cell contact threshold, in microns (R: ``contact.range``).
    ratio : float, optional
        Pixel-to-micron conversion factor (R: ``ratio``). When ``None``
        — matching R's default — the threshold ``(interaction.range +
        tol) / ratio`` becomes ill-defined (R returns ``numeric(0)``)
        and the resulting ``d.spatial`` matrix is empty.
    tol : float, default ``5``
        Distance tolerance, in microns (R: ``tol``).

    Returns
    -------
    dict
        Two-key dict ``{"d.spatial": csc_matrix, "adj.contact":
        csc_matrix}`` matching the R return value. When the first
        argument is an ``AnnData`` the same matrices are also written
        to ``obsp["spatial_distances"]`` and ``obsp["spatial_connectivities"]``,
        and the call parameters are stashed under
        ``uns["cellchat"]["spatial"]["computeCellDistance"]``.
    """
    adata: Optional[ad.AnnData]
    if isinstance(coordinates, ad.AnnData):
        adata = coordinates
        if "spatial" not in adata.obsm:
            raise ValueError(
                "computeCellDistance requires spatial coordinates in adata.obsm['spatial']"
            )
        coords = _coords_to_array(adata.obsm["spatial"])
    else:
        adata = None
        coords = _coords_to_array(coordinates)

    n = coords.shape[0]

    # ``Rfast::Dist`` -> dense Euclidean distance matrix (zero diagonal)
    diffs = coords[:, None, :] - coords[None, :, :]
    d_dense = np.sqrt(np.sum(diffs * diffs, axis=2))

    # threshold: ((interaction.range + tol) / ratio). R's behaviour for
    # ``ratio = NULL`` is that ``x / NULL`` returns ``numeric(0)``, hence
    # the comparison ``d.spatial[, j] <= numeric(0)`` is always FALSE and
    # the resulting sparse matrix is empty. Replicate that exactly.
    if ratio is None:
        threshold: Optional[float] = None
    else:
        threshold = float(interaction_range + tol) / float(ratio)

    # Build COO triplets. R iterates by column, keeping i where
    # d[i, j] <= threshold AND d[i, j] > 0.
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    if threshold is not None:
        for j in range(n):
            col = d_dense[:, j]
            mask = (col <= threshold) & (col > 0)
            i_idx = np.flatnonzero(mask)
            if i_idx.size == 0:
                continue
            rows.extend(i_idx.tolist())
            cols.extend([j] * i_idx.size)
            vals.extend(col[i_idx].tolist())

    d_spatial = sparse.coo_matrix(
        (np.asarray(vals, dtype=float),
         (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))),
        shape=(n, n),
    ).tocsc()

    # ``if (!is.null(ratio)) { d.spatial@x <- d.spatial@x * ratio }``
    if ratio is not None:
        d_spatial.data = d_spatial.data * float(ratio)

    adj_contact = createCellCellContactMatrixFrom_dspatial(
        d_spatial, contact_threshold=float(contact_range), tol=float(tol)
    )

    if adata is not None:
        adata.obsp["spatial_distances"] = d_spatial
        # Per the SpatialCellChat porting contract §3.E, write the
        # spatial-connectivities adjacency via squidpy (radius =
        # contact.range + tol, in microns). This produces the same
        # off-diagonal pattern as R's
        # ``createCellCellContactMatrixFrom_dspatial`` *before* the diagonal
        # write; we then add the unit diagonal to match R exactly.
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            radius=float(contact_range) + float(tol),
            delaunay=False,
        )
        sc_mat = adata.obsp["spatial_connectivities"]
        sc_lil = sc_mat.tolil()
        for k in range(sc_mat.shape[0]):
            sc_lil[k, k] = 1
        adata.obsp["spatial_connectivities"] = sc_lil.tocsr()
        cc = _ensure_uns(adata)
        cc["spatial"]["computeCellDistance"] = {
            "interaction.range": float(interaction_range),
            "contact.range": float(contact_range),
            "ratio": None if ratio is None else float(ratio),
            "tol": float(tol),
        }

    return {"d.spatial": d_spatial, "adj.contact": adj_contact}


# ---------------------------------------------------------------------------
# createCellCellContactMatrixFrom_dspatial (R: spatial.R:620)
# ---------------------------------------------------------------------------
def createCellCellContactMatrixFrom_dspatial(
    d_spatial: sparse.spmatrix,
    contact_threshold: float,
    tol: float,
) -> sparse.csc_matrix:
    """Build the binary contact-adjacency matrix from a sparse distance matrix.

    Mirrors R's :func:`createCellCellContactMatrixFrom_dspatial`
    (``spatial.R:620``): it keeps only the entries with
    ``distance <= contact_threshold + tol``, sets them to ``1``, then sets
    the diagonal to ``1``.

    Per the SpatialCellChat porting contract §3.E, when the caller has an
    :class:`anndata.AnnData` it should instead use
    :func:`computeCellDistance` (which routes through the standard
    ``squidpy.gr.spatial_neighbors``-equivalent contact mask under
    ``obsp["spatial_connectivities"]``). This standalone function preserves
    parity with R for ad-hoc use and is exercised directly by
    :func:`computeCellDistance`.

    Parameters
    ----------
    d_spatial : scipy.sparse.spmatrix
        Sparse cell-cell distance matrix (R's ``d.spatial``).
    contact_threshold : float
        Contact-distance threshold in microns (R's ``contact.threshold``).
    tol : float
        Distance tolerance in microns (R's ``tol``).

    Returns
    -------
    scipy.sparse.csc_matrix
        The binary contact adjacency matrix, with a unit diagonal.
    """
    adj = d_spatial.tocoo()
    contact_range = float(contact_threshold)
    threshold = contact_range + float(tol)

    # ``which(adj.contact@x <= (contact.range + tol))``
    keep = adj.data <= threshold
    rows = adj.row[keep]
    cols = adj.col[keep]
    new_data = np.ones(int(np.sum(keep)), dtype=float)

    n_rows, n_cols = adj.shape
    # ``Matrix::diag(adj.contact) <- 1`` — fills positions ``(k, k)`` for
    # ``k = 0..min(n_rows, n_cols)-1``. Note that some of these may
    # already be present in the kept entries (defensive: dedupe later).
    diag_n = min(n_rows, n_cols)
    diag_idx = np.arange(diag_n, dtype=int)
    rows_full = np.concatenate([rows, diag_idx])
    cols_full = np.concatenate([cols, diag_idx])
    data_full = np.concatenate([new_data, np.ones(diag_n, dtype=float)])

    # Build CSC; if (i,i) was already kept, R's ``Matrix::diag<-`` would
    # overwrite it with 1, which is what coo->csc summation gives (1 vs.
    # 1+1=2). Use a max-merge by collapsing duplicates: take the last
    # value per coordinate (matching R's overwrite semantics).
    coo = sparse.coo_matrix(
        (data_full, (rows_full, cols_full)), shape=(n_rows, n_cols)
    )
    # Collapse duplicates by mapping all duplicate coordinates to value 1
    # (the R diag overwrite + binary mask both produce 1).
    coo.sum_duplicates()
    coo.data = np.ones_like(coo.data)
    return coo.tocsc()


# ---------------------------------------------------------------------------
# computeRegionDistance (R: modeling.R:2700)
# ---------------------------------------------------------------------------
def computeRegionDistance(
    coordinates: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    group: Union[Sequence, pd.Categorical],
    trim: float = 0.1,
    interaction_length: Optional[float] = None,
    spot_size: Optional[float] = None,
    spot_size_fullres: Optional[float] = None,
    k_min: int = 10,
) -> pd.DataFrame:
    """Compute pairwise region-to-region distance from spatial coordinates.

    Mirrors R's :func:`computeRegionDistance` (``modeling.R:2700``). For
    each (i, j) pair of region labels we compute the nearest-neighbour
    distance from each cell of region ``i`` to its closest cell of region
    ``j`` (k=1 KNN), then average those distances using the truncated
    mean (``trim`` proportion trimmed from each end). The resulting matrix
    is symmetrised by ``(D + D.T) / 2``.

    When both ``spot_size`` and ``spot_size_fullres`` are given, distances
    are first re-scaled by ``spot_size / spot_size_fullres``, an
    ``interaction.length`` filter is applied per-pair, and pairs with
    fewer than ``k_min`` interacting neighbours are masked out (set to
    ``NaN``).

    Parameters
    ----------
    coordinates : array-like or anndata.AnnData
        ``(n, 2)`` coordinate matrix or an AnnData with
        ``obsm["spatial"]``.
    group : sequence
        Per-cell categorical group labels (R's ``group`` factor).
    trim : float, default ``0.1``
        Truncated-mean trim fraction (R: ``trim``).
    interaction_length, spot_size, spot_size_fullres : float, optional
        Adjacency mask parameters (R: ``interaction.length``,
        ``spot.size``, ``spot.size.fullres``).
    k_min : int, default ``10``
        Minimum interacting-neighbour count per region pair (R:
        ``k.min``).

    Returns
    -------
    pandas.DataFrame
        Square ``DataFrame`` of region-to-region distances indexed by
        the levels of ``group``.

    Notes
    -----
    *Tier-2 deviation (per the porting contract).* R's
    ``computeRegionDistance`` errors at ``BiocNeighbors::queryKNN`` when
    a level present in ``levels(group)`` has zero cells (k must be
    positive on a 0-row matrix).  This Python port keeps the result's
    ``nlevels × nlevels`` shape and silently fills NaN for those entries
    instead — strictly safer; the unused-level error is itself an R
    artefact that should not propagate to the user.
    """
    if isinstance(coordinates, ad.AnnData):
        adata: Optional[ad.AnnData] = coordinates
        coords = _coords_to_array(adata.obsm["spatial"])
    else:
        adata = None
        coords = _coords_to_array(coordinates)

    if not isinstance(group, pd.Categorical):
        group = pd.Categorical(group)
    if not (0.0 <= trim <= 0.25):
        # R's `mean(x, trim, ...)` accepts trim in [0, 0.5] but the doc says
        # 0..0.25; we keep the broader range silently to match R.
        pass

    # R sets ``numCluster <- nlevels(group)`` so the result is always
    # nlevels × nlevels indexed by *all* levels (including any absent ones).
    # Iterate over every level pair; if either side has zero observations
    # the corresponding entry stays NaN (R's loop returns NaN here too via
    # an empty-vector ``mean``).
    levels_all = list(group.categories)
    n = len(levels_all)

    d_spatial = np.full((n, n), np.nan, dtype=float)
    adj_spatial = np.zeros((n, n), dtype=float)

    use_adj = spot_size is not None and spot_size_fullres is not None
    if use_adj and interaction_length is None:
        raise ValueError(
            "computeRegionDistance requires `interaction_length` when both "
            "`spot_size` and `spot_size_fullres` are provided."
        )

    grp_arr = np.asarray(group)
    for i, lvl_i in enumerate(levels_all):
        idx_i = np.flatnonzero(grp_arr == lvl_i)
        if idx_i.size == 0:
            continue
        data_i = coords[idx_i]
        for j, lvl_j in enumerate(levels_all):
            idx_j = np.flatnonzero(grp_arr == lvl_j)
            if idx_j.size == 0:
                continue
            data_j = coords[idx_j]

            # ``BiocNeighbors::queryKNN(data_j, data_i, k=1)`` finds, for
            # each row of ``data_i``, the nearest row of ``data_j``.
            tree = KDTree(data_j)
            dist_ij, idx_ij = tree.query(data_i, k=1)
            dist_ij = np.asarray(dist_ij, dtype=float).ravel()
            idx_ij = np.asarray(idx_ij, dtype=int).ravel()

            if use_adj:
                scaled = dist_ij * float(spot_size) / float(spot_size_fullres)
                mask = scaled - float(interaction_length) < float(spot_size) / 2
                adj_spatial[i, j] = float(len(np.unique(idx_ij[mask])) >= int(k_min))
                d_spatial[i, j] = _truncated_mean(scaled, trim=trim)
            else:
                d_spatial[i, j] = _truncated_mean(dist_ij, trim=trim)

    # Symmetrise
    d_spatial = (d_spatial + d_spatial.T) / 2.0
    if use_adj:
        adj = adj_spatial * adj_spatial.T
        adj_nan = np.where(adj == 0, np.nan, adj)
        d_spatial = d_spatial * adj_nan

    out = pd.DataFrame(d_spatial, index=levels_all, columns=levels_all)

    if adata is not None:
        cc = _ensure_uns(adata)
        cc["spatial"]["computeRegionDistance"] = out.copy()

    return out


def _truncated_mean(x: np.ndarray, trim: float) -> float:
    """Truncated mean matching R's ``mean(x, trim = trim, na.rm = TRUE)``.

    R's ``base::mean.default`` with ``trim`` removes ``floor(trim * n)``
    observations from each end of the sorted vector, then averages the rest.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return float("nan")
    if trim <= 0.0:
        return float(np.mean(arr))
    if trim >= 0.5:
        return float(np.median(arr))
    arr = np.sort(arr)
    lo = int(np.floor(trim * n))
    hi = n - lo
    return float(np.mean(arr[lo:hi]))


# ---------------------------------------------------------------------------
# computeColocalization (R: spatial.R:660)
# ---------------------------------------------------------------------------
def computeColocalization(
    coordinates: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    group: Union[Sequence, pd.Categorical, None] = None,
    idents_use: Optional[Sequence] = None,
    symmetric: bool = True,
    nboot: int = 100,
    seed_use: int = 1,
    permutations: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> pd.DataFrame:
    """Permutation-based colocalisation matrix between cell-group labels.

    Mirrors R's :func:`computeColocalization` (``spatial.R:660``). For
    every (i, j) pair of group labels we

    1. compute the per-cell distance matrix from cells in ``label_i`` to
       cells in ``label_j`` (Euclidean, via a k-d tree look-up but
       comparing every pair like R does),
    2. compute the median over rows of the row-mean distances (the
       "observed" statistic),
    3. shuffle the row-set ``data_label_1`` ``nboot`` times by sampling
       (without replacement) ``nrow(data_label_1)`` indices from the
       cells **outside** both labels (R's ``data_nont``), and recompute
       the same statistic for each draw,
    4. return ``fdr[i, j] = (#permutations <= observed) / nboot``.

    Parameters
    ----------
    coordinates : array-like or anndata.AnnData
    group : sequence
        Per-cell categorical group labels (must be a factor).
    idents_use : sequence, optional
        Restrict the matrix to these idents (R: ``idents.use``).
    symmetric : bool, default ``True``
        Symmetrise by taking the per-pair minimum (R: ``symmetric``).
    nboot : int, default ``100``
        Number of permutations (R: ``nboot``).
    seed_use : int, default ``1``
        RNG seed (R: ``seed.use``). Note that R's ``set.seed`` and the
        Mersenne-Twister stream are non-portable; for bit-exact parity
        with R use the ``permutations`` argument below.
    permutations : dict[(i, j), array], optional
        Pre-generated permutation matrix (shape ``(len(label_1), nboot)``,
        1-based indices into ``data_nont``). When supplied, the function
        uses these instead of drawing its own (lets test suites compare
        bit-for-bit against R-dumped permutations).

    Returns
    -------
    pandas.DataFrame
        ``(numCluster, numCluster)`` colocalisation matrix indexed by the
        considered idents.
    """
    if isinstance(coordinates, ad.AnnData):
        adata: Optional[ad.AnnData] = coordinates
        coords = _coords_to_array(adata.obsm["spatial"])
    else:
        adata = None
        coords = _coords_to_array(coordinates)

    if group is None:
        raise ValueError("Please input the `group` as a factor!")
    if not isinstance(group, pd.Categorical):
        group = pd.Categorical(group)

    levels_all = list(group.categories)
    level_use0 = list(levels_all)

    if idents_use is None:
        present = set(group.unique())
        level_use = [lvl for lvl in levels_all if lvl in present]
        numCluster = len(levels_all)
    else:
        idents_use = list(idents_use)
        level_use = [lvl for lvl in levels_all if lvl in idents_use]
        numCluster = len(idents_use)
        level_use0 = list(level_use)

    grp_arr = np.asarray(group)
    rng = np.random.default_rng(int(seed_use))

    fdr = np.full((numCluster, numCluster), np.nan, dtype=float)
    for i in range(numCluster):
        for j in range(numCluster):
            label_1 = level_use[i]
            label_2 = level_use[j]
            data_label_1 = coords[grp_arr == label_1]
            data_label_2 = coords[grp_arr == label_2]
            shuffle_sel = ~np.isin(grp_arr, [label_1, label_2])
            data_nont = coords[shuffle_sel]

            n1 = data_label_1.shape[0]
            n2 = data_label_2.shape[0]

            # observed coord2_dist: pairwise Euclidean from data_label_1 to
            # each data_label_2 point (n1 x n2). R loops over each j-cell
            # and calls raster::pointDistance(coord1, coord2_j, lonlat=F).
            diffs = data_label_1[:, None, :] - data_label_2[None, :, :]
            coord2_dist = np.sqrt(np.sum(diffs * diffs, axis=2))

            # Permutation matrix: in R, ``permutation`` is a (n1, nboot)
            # matrix of 1-based indices into ``data_nont`` (sampled w/o
            # replacement per column).
            if permutations is not None and (i + 1, j + 1) in permutations:
                perm = np.asarray(permutations[(i + 1, j + 1)], dtype=int) - 1
            else:
                # Fall back to numpy RNG (bit-for-bit parity with R is
                # impossible without a Mersenne-Twister port). R uses
                # ``set.seed(seed.use)`` once *per pair* inside the (i, j)
                # loop, so we re-seed here too.
                rng = np.random.default_rng(int(seed_use))
                if data_nont.shape[0] > 0:
                    perm = np.column_stack([
                        rng.choice(data_nont.shape[0], size=n1, replace=False)
                        for _ in range(nboot)
                    ])
                else:
                    perm = np.zeros((n1, nboot), dtype=int)

            coord2_shuf_dist = np.empty((n1, n2, nboot), dtype=float)
            for idx3 in range(nboot):
                coord1_shuf = data_nont[perm[:, idx3]]
                d = coord1_shuf[:, None, :] - data_label_2[None, :, :]
                coord2_shuf_dist[:, :, idx3] = np.sqrt(np.sum(d * d, axis=2))

            label2_mean_dists = np.mean(coord2_dist, axis=1)
            label2_obs_median = float(np.median(label2_mean_dists))

            fdr_count = [0, 0]
            for idx4 in range(nboot):
                shuf_mean = np.mean(coord2_shuf_dist[:, :, idx4], axis=1)
                shuf_med = float(np.median(shuf_mean))
                if shuf_med > label2_obs_median:
                    fdr_count[0] += 1
                else:
                    fdr_count[1] += 1
            fdr[i, j] = fdr_count[1] / float(nboot)

    if symmetric and numCluster >= 2:
        for i in range(numCluster - 1):
            for j in range(i + 1, numCluster):
                if fdr[i, j] > fdr[j, i]:
                    fdr[i, j] = fdr[j, i]
                fdr[j, i] = fdr[i, j]

    out = pd.DataFrame(fdr, index=level_use0, columns=level_use0)

    if adata is not None:
        cc = _ensure_uns(adata)
        cc["spatial"]["computeColocalization"] = out.copy()

    return out


# ---------------------------------------------------------------------------
# computeGridSize (R: spatial.R:200)
# ---------------------------------------------------------------------------
def computeGridSize(
    coordinates: Union[np.ndarray, pd.DataFrame, ad.AnnData],
    cellsize: Optional[float] = None,
    grid_resolution: Optional[float] = None,
    spatial_factors: Optional[Dict[str, float]] = None,
    do_plot: bool = False,
    what: Literal["polygons", "corners", "centers"] = "polygons",
    square: bool = True,
) -> Dict[str, Union[float, int]]:
    """Auto-pick a grid cell-size from the cell-cell minimum distance.

    Mirrors R's :func:`computeGridSize` (``spatial.R:200``). When
    ``cellsize`` is ``None`` the cell-cell minimum distance (with the
    diagonal masked) is used as the default; the proposed new cell-size
    is then ``cellsize * grid_resolution`` (default ``2``).

    The R reference also draws a ``ggplot`` summary when ``do_plot=TRUE``.
    The Python port skips plotting (``do_plot`` is accepted for
    signature parity) and instead returns the numeric report as a dict.

    Parameters
    ----------
    coordinates : array-like or anndata.AnnData
    cellsize : float, optional
        Cell-size in coordinate units (R: ``cellsize``). Defaults to the
        minimum pairwise distance.
    grid_resolution : float, optional
        Multiplier applied to ``cellsize`` (R: ``grid.resolution``);
        defaults to ``2``.
    spatial_factors : dict, optional
        Mapping with key ``"ratio"`` used to convert the cell-size to
        microns. Pulled from ``adata.uns['cellchat']['spatial.factors']``
        when an AnnData is passed and the dict is otherwise ``None``.
    do_plot : bool, default ``False``
        Plotting is skipped in Python (always returns the numeric report).
    what : {"polygons", "corners", "centers"}, default ``"polygons"``
    square : bool, default ``True``

    Returns
    -------
    dict
        ``{"cellsize": <input or default cell-size>, "newcellsize":
        cellsize * grid_resolution, "spot.size": newcellsize * ratio,
        "n_grids": <number of generated grid cells (when do_plot)>,
        "n_assigned_grids": <number of grids containing at least one
        spot>, "all_spots_assigned": bool}``.
    """
    if isinstance(coordinates, ad.AnnData):
        adata: Optional[ad.AnnData] = coordinates
        coords = _coords_to_array(adata.obsm["spatial"])
        if spatial_factors is None:
            cc = adata.uns.get("cellchat", {})
            spatial_factors = cc.get("spatial.factors", None)
    else:
        adata = None
        coords = _coords_to_array(coordinates)

    sf = spatial_factors if spatial_factors is not None else {}
    ratio = float(sf.get("ratio", 1.0))

    if cellsize is None:
        # min cell-cell distance excluding the diagonal
        diffs = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt(np.sum(diffs * diffs, axis=2))
        np.fill_diagonal(d, np.nan)
        cellsize_used = float(np.nanmin(d))
    else:
        cellsize_used = float(cellsize)

    grid_res = 2.0 if grid_resolution is None else float(grid_resolution)
    newcellsize = cellsize_used * grid_res
    spot_size = round(newcellsize * ratio, 4)

    out: Dict[str, Union[float, int, bool]] = {
        "cellsize": cellsize_used,
        "newcellsize": float(newcellsize),
        "spot.size": float(spot_size),
    }

    if do_plot:
        # construct the grid only to derive the summary (mirrors the R
        # branch's ``cat(... will generate about ", sum(grid_nspots != 0))``
        # message). The plot itself is intentionally skipped — Python
        # tutorials use scanpy / matplotlib elsewhere.
        within = _make_grid_within(coords, cellsize=newcellsize, square=square)
        within_nGrid = within.sum(axis=1)
        grid_nspots = within.sum(axis=0)
        out["n_grids"] = int(within.shape[1])
        out["n_assigned_grids"] = int(np.sum(grid_nspots != 0))
        out["all_spots_assigned"] = bool(np.all(within_nGrid > 0))

    if adata is not None:
        cc = _ensure_uns(adata)
        cc["spatial"]["computeGridSize"] = dict(out)

    return out


def _grid_origin(coords: np.ndarray, cellsize: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """Bounding box (xmin, ymin, xmax, ymax) of a grid covering ``coords``.

    Mirrors ``sf::st_make_grid`` semantics: the grid origin is the point's
    bbox lower-left corner, and the grid extends so that every cell-size
    step contains at least one full cell. The R harness shows that
    bbox extent for square grids is exactly the bbox of the input
    points, padded so that ``ceil((max - min) / cellsize) * cellsize``.
    """
    xmin, ymin = float(coords[:, 0].min()), float(coords[:, 1].min())
    xmax_raw, ymax_raw = float(coords[:, 0].max()), float(coords[:, 1].max())
    cx, cy = float(cellsize[0]), float(cellsize[1])
    nx = int(np.ceil((xmax_raw - xmin) / cx))
    ny = int(np.ceil((ymax_raw - ymin) / cy))
    if nx == 0:
        nx = 1
    if ny == 0:
        ny = 1
    xmax = xmin + nx * cx
    ymax = ymin + ny * cy
    return xmin, ymin, xmax, ymax


def _make_grid_within(
    coords: np.ndarray,
    cellsize: Union[float, Tuple[float, float]],
    square: bool = True,
) -> np.ndarray:
    """Build the (n_spots x n_grids) intersects-Boolean matrix.

    Replaces R's ``sf::st_make_grid(...) %>% sf::st_intersects(df.sf, ...)``.
    Uses ``geopandas`` + ``shapely.geometry.box`` and ``gpd.sjoin`` with
    the ``intersects`` predicate so points on cell boundaries are counted
    in every adjacent cell — matching ``sf::st_intersects`` semantics
    (the R harness explicitly reproduces this multi-assignment for points
    on a grid vertex).
    """
    if np.isscalar(cellsize):
        cs = (float(cellsize), float(cellsize))
    else:
        cs = (float(cellsize[0]), float(cellsize[1]))
    cx, cy = cs

    xmin, ymin, xmax, ymax = _grid_origin(coords, cs)

    nx = int(round((xmax - xmin) / cx))
    ny = int(round((ymax - ymin) / cy))

    polys: List[_sg.Polygon] = []
    grid_ids: List[int] = []
    grid_id = 1  # 1-based to match R
    if square:
        for j in range(ny):  # column-major: matches sf::st_make_grid order
            for i in range(nx):
                x0 = xmin + i * cx
                y0 = ymin + j * cy
                polys.append(_sg.box(x0, y0, x0 + cx, y0 + cy))
                grid_ids.append(grid_id)
                grid_id += 1
    else:
        # hexagonal grid — sf uses "edge length = cellsize / sqrt(3)" with
        # flat-top orientation and ``st_make_grid``'s exact tessellation.
        # Replicate that here via shapely Polygons.
        edge = cx / np.sqrt(3.0)
        dx = cx
        dy = 1.5 * edge
        ny_h = int(np.ceil((ymax - ymin) / dy)) + 1
        nx_h = int(np.ceil((xmax - xmin) / dx)) + 1
        for j in range(ny_h):
            for i in range(nx_h):
                x_c = xmin + i * dx + (dx / 2.0 if j % 2 == 1 else 0.0)
                y_c = ymin + j * dy
                pts = [
                    (x_c, y_c + edge),
                    (x_c + cx / 2.0, y_c + edge / 2.0),
                    (x_c + cx / 2.0, y_c - edge / 2.0),
                    (x_c, y_c - edge),
                    (x_c - cx / 2.0, y_c - edge / 2.0),
                    (x_c - cx / 2.0, y_c + edge / 2.0),
                ]
                polys.append(_sg.Polygon(pts))
                grid_ids.append(grid_id)
                grid_id += 1

    grid_gdf = gpd.GeoDataFrame({"grid_id": grid_ids}, geometry=polys, crs=3857)
    pt_gdf = gpd.GeoDataFrame(
        {"point_id": np.arange(coords.shape[0])},
        geometry=[_sg.Point(p) for p in coords],
        crs=3857,
    )
    joined = gpd.sjoin(pt_gdf, grid_gdf, how="left", predicate="intersects")

    n_spots = coords.shape[0]
    n_grids = len(polys)
    within = np.zeros((n_spots, n_grids), dtype=bool)
    for _, row in joined.iterrows():
        if pd.isna(row["index_right"]):
            continue
        within[int(row["point_id"]), int(row["index_right"])] = True
    return within


# ---------------------------------------------------------------------------
# makeGridSpatialCellChat (R: spatial.R:326)
# ---------------------------------------------------------------------------
def makeGridSpatialCellChat(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    cellsize: Union[float, Tuple[float, float]] = (5.0, 5.0),
    what: Literal["polygons", "corners", "centers"] = "polygons",
    square: bool = True,
    idents_ties_method: Literal["first", "last", "random"] = "first",
    group_by: str = "active_ident",
    spatial_factors: Optional[Dict[str, float]] = None,
) -> ad.AnnData:
    """Aggregate an AnnData into a regular grid of square (or hex) cells.

    Mirrors R's :func:`makeGridSpatialCellChat` (``spatial.R:326``).
    Builds a ``cellsize``-sized grid covering the bounding box of
    ``adata.obsm['spatial']``, assigns each spot to one or more grid
    cells via the ``intersects`` predicate, then aggregates the data
    matrix by per-grid row-mean (R: ``Matrix::rowMeans``). Grids
    containing zero spots are dropped. The dominant ident per grid is
    taken via majority vote (R: ``base::max.col(..., ties.method)``).

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData. Must have ``obsm["spatial"]`` and (when
        ``group_by="active_ident"``) an ``"active_ident"`` column.
    layer : str, optional
        Layer name to aggregate from (R: ``data.slot``). Defaults to
        ``adata.X``.
    cellsize : float or (float, float), default ``(5, 5)``
        Cell-size in coordinate units (R: ``cellsize``).
    what, square : str, bool
        Grid generation parameters (R: ``what``, ``square``).
    idents_ties_method : str, default ``"first"``
        Tie-breaker passed to majority vote (R: ``ties.method``).
    group_by : str, default ``"active_ident"``
        ``adata.obs`` column to use as ident labels.
    spatial_factors : dict, optional
        Mapping with ``"ratio"`` and (optional) ``"tol"``. The new
        ``tol`` is set to ``spot_size / 2`` where ``spot_size = cellsize
        * ratio``.

    Returns
    -------
    anndata.AnnData
        New grid-aggregated AnnData with the same gene set, the new
        ``X`` (per-grid means), per-grid coordinates in ``obsm["spatial"]``
        and ``obsm["X_grid"]``, the dominant ``cell.type`` factor in
        ``obs``, and the per-ident counts in ``obs`` columns.
    """
    if not isinstance(adata, ad.AnnData):
        raise ValueError("makeGridSpatialCellChat expects an AnnData object")
    if "spatial" not in adata.obsm:
        raise ValueError(
            "makeGridSpatialCellChat requires adata.obsm['spatial']"
        )
    if group_by not in adata.obs.columns:
        raise ValueError(f"group_by={group_by!r} not found in adata.obs")

    coords = _coords_to_array(adata.obsm["spatial"])
    coord_df = pd.DataFrame(coords, columns=["x_cent", "y_cent"], index=adata.obs_names)

    idents = adata.obs[group_by]
    if not isinstance(idents.dtype, pd.CategoricalDtype):
        idents = pd.Categorical(idents)
    else:
        idents = idents.values
    levels_idents = list(idents.categories)

    # spatial.factors defaults
    if spatial_factors is None:
        cc = adata.uns.get("cellchat", {})
        sf = cc.get("spatial.factors", {})
    else:
        sf = dict(spatial_factors)
    sf = dict(sf)  # copy

    if np.isscalar(cellsize):
        cs0 = float(cellsize)
        cellsize_t: Tuple[float, float] = (cs0, cs0)
    else:
        cellsize_t = (float(cellsize[0]), float(cellsize[1]))

    ratio = float(sf.get("ratio", 1.0))
    spot_size = cellsize_t[0] * ratio
    sf["tol"] = spot_size / 2.0

    within = _make_grid_within(coords, cellsize=cellsize_t, square=square)
    within_nGrid = within.sum(axis=1)
    spotsCounts_perGrid = within.sum(axis=0)

    # Filter to non-empty grids
    keep_grid_idx = np.flatnonzero(spotsCounts_perGrid > 0)

    # Per-grid ident counts (NewMeta) — for each grid, count cells per
    # ident among the spots in it.
    ident_codes = np.asarray(idents.codes, dtype=int)
    n_idents = len(levels_idents)
    new_meta = np.zeros((within.shape[1], n_idents), dtype=int)
    for g in range(within.shape[1]):
        spots = np.flatnonzero(within[:, g])
        if spots.size == 0:
            continue
        codes = ident_codes[spots]
        # remove -1 (R's ``NA`` -> dplyr::if_else NA)
        codes = codes[codes >= 0]
        if codes.size:
            counts = np.bincount(codes, minlength=n_idents)
            new_meta[g, :] = counts

    # max.col with ties.method
    if idents_ties_method == "first":
        present_idents_idx = np.argmax(new_meta, axis=1)
    elif idents_ties_method == "last":
        # numpy argmax returns first; reverse arrays to get "last"
        rev = new_meta[:, ::-1]
        present_idents_idx = (n_idents - 1) - np.argmax(rev, axis=1)
    elif idents_ties_method == "random":
        rng = np.random.default_rng(0)
        present_idents_idx = np.empty(new_meta.shape[0], dtype=int)
        for k in range(new_meta.shape[0]):
            row = new_meta[k]
            best = np.flatnonzero(row == row.max())
            present_idents_idx[k] = int(rng.choice(best))
    else:
        raise ValueError(
            f"idents_ties_method must be one of 'first', 'last', 'random' (got {idents_ties_method!r})"
        )
    present_idents = np.array(levels_idents)[present_idents_idx]

    new_meta_df = pd.DataFrame(new_meta, columns=levels_idents)
    new_meta_df["cell.type"] = present_idents
    new_meta_df["spots.counts"] = spotsCounts_perGrid
    new_meta_df = new_meta_df.iloc[keep_grid_idx].reset_index(drop=True)

    # New cell names "Grid<id>" — grid_id is 1-based (R: ``square_grid_sf$grid_id``)
    new_spot_names = [f"Grid{i + 1}" for i in keep_grid_idx]

    # New coordinates (per-grid mean)
    new_coords = np.zeros((keep_grid_idx.size, 2), dtype=float)
    for k_out, g in enumerate(keep_grid_idx):
        spots = np.flatnonzero(within[:, g])
        new_coords[k_out] = coords[spots].mean(axis=0)

    # New data matrix (genes x grids — but AnnData wants cells x genes)
    if layer is None:
        X_in = adata.X
    else:
        X_in = adata.layers[layer]
    if sparse.issparse(X_in):
        X_dense = X_in.toarray()
    else:
        X_dense = np.asarray(X_in)
    # adata is (cells x genes); we want a per-grid mean across the cells
    # that intersect each grid -> result is (grids x genes)
    new_X = np.zeros((keep_grid_idx.size, X_dense.shape[1]), dtype=float)
    for k_out, g in enumerate(keep_grid_idx):
        spots = np.flatnonzero(within[:, g])
        new_X[k_out] = X_dense[spots].mean(axis=0)

    new_meta_df["cell.type"] = pd.Categorical(
        new_meta_df["cell.type"], categories=levels_idents
    )
    new_meta_df.index = pd.Index(new_spot_names, name="obs")

    obs_out = new_meta_df
    var_out = adata.var.copy()

    # Build the new AnnData
    adata_grid = ad.AnnData(
        X=sparse.csr_matrix(new_X.astype(np.float32)),
        obs=obs_out,
        var=var_out,
    )
    adata_grid.obsm["spatial"] = new_coords
    adata_grid.obsm["X_grid"] = new_coords.copy()
    adata_grid.obs["active_ident"] = obs_out["cell.type"].values
    adata_grid.uns["cellchat"] = {
        "spatial.factors": sf,
        "spatial": {
            "within_nGrid": within_nGrid,
            "recommended.contact.range": float(spot_size),
            "makeGridSpatialCellChat": {
                "cellsize": list(cellsize_t),
                "what": what,
                "square": bool(square),
                "idents.ties.method": idents_ties_method,
                "n_input_spots": int(coords.shape[0]),
                "n_grids_total": int(within.shape[1]),
                "n_grids_kept": int(keep_grid_idx.size),
                "all_spots_assigned": bool(np.all(within_nGrid > 0)),
            },
        },
    }
    return adata_grid
