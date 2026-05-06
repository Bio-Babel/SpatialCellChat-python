"""Preprocessing routines for SpatialCellChat.

Direct port of the matching block in ``utilities.R`` (preProcessing,
normalizeData, scaleData, runPCA, runUMAP, subsetData, updateClusterLabels,
sketchData, projectData and its helpers). The public surface accepts an
:class:`anndata.AnnData` first argument and writes results in-place, except
for :func:`subsetData` and :func:`sketchData` which return new AnnData
objects.

The R reference (``utilities.R`` lines 744–1734 and 1750–1853) is the
ground truth; correctness has been verified column-by-column against the R
implementation through ``r_dumps/`` golden CSVs.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
import umap as _umap
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.utils.extmath import randomized_svd

from . import _alra
from .database import extractGene

__all__ = [
    "preProcessing",
    "normalizeData",
    "scaleData",
    "runPCA",
    "runUMAP",
    "subsetData",
    "updateClusterLabels",
    "sketchData",
    "sketchDataAdata",
    "projectData",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_dense(M: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    """Materialise a (potentially sparse) matrix as a dense ``ndarray``."""
    if sparse.issparse(M):
        return np.asarray(M.toarray(), dtype=float)
    return np.asarray(M, dtype=float)


def _genes_by_cells(adata: ad.AnnData, layer: Optional[str] = None) -> np.ndarray:
    """Return a dense (genes x cells) view of a layer of ``adata``.

    The R code expects expression matrices with genes in rows and cells in
    columns; AnnData stores them transposed.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    return _to_dense(X).T


# ---------------------------------------------------------------------------
# normalizeData (R: utilities.R:787)
# ---------------------------------------------------------------------------
def normalizeData(
    adata: ad.AnnData,
    scale_factor: float = 10000.0,
    do_log: bool = True,
    layer_in: Optional[str] = None,
    layer_out: Optional[str] = None,
) -> ad.AnnData:
    """Library-size normalise an AnnData expression matrix.

    Mirrors R's ``normalizeData`` (``utilities.R:787``). Each cell is divided
    by its library size, multiplied by ``scale_factor``, and (optionally)
    passed through ``log1p``. The original counts (when present in
    ``adata.X``) are first stashed under ``adata.layers["counts"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    scale_factor : float, default 10000
        The ``scale.factor`` argument from R.
    do_log : bool, default True
        Whether to apply ``log1p``. The R reference contains a typo in the
        ``do.log = FALSE`` branch (``data.norm`` is undefined); the Python
        port returns the un-logged ``expr`` matrix in that branch, which is
        what the R parameter documentation describes.
    layer_in : str, optional
        If set, normalise the named layer instead of ``adata.X``.
    layer_out : str, optional
        If set, write the normalised matrix to this layer; otherwise
        overwrite ``adata.X``.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    if layer_in is None:
        X = adata.X
    else:
        X = adata.layers[layer_in]

    # cache raw counts when overwriting X for the first time
    if layer_in is None and layer_out is None and "counts" not in adata.layers:
        adata.layers["counts"] = X.copy()

    # AnnData stores cells in rows; the R formula is colSums on a (genes x
    # cells) matrix. Per cell -> sum across var axis (axis=1).
    if sparse.issparse(X):
        lib = np.asarray(X.sum(axis=1)).ravel().astype(float)
        # divide rows by lib, then multiply by scale_factor
        inv = sparse.diags(scale_factor / lib)
        expr = inv @ X
        if do_log:
            out = expr.copy()
            out.data = np.log1p(out.data)
        else:
            out = expr
    else:
        lib = X.sum(axis=1).astype(float)
        expr = (X / lib[:, np.newaxis]) * scale_factor
        out = np.log1p(expr) if do_log else expr

    if layer_out is None:
        adata.X = out
    else:
        adata.layers[layer_out] = out
    return adata


# ---------------------------------------------------------------------------
# scaleData (R: utilities.R:805)
# ---------------------------------------------------------------------------
def scaleData(
    adata: ad.AnnData,
    do_center: bool = True,
    layer_in: Optional[str] = None,
    layer_out: str = "data_scale",
) -> ad.AnnData:
    """Z-score standardisation across cells, gene-by-gene.

    Mirrors R's ``scaleData`` (``utilities.R:805``). The R version transposes
    twice around :func:`scale`, which scales by column. For an AnnData
    (cells x genes), this is equivalent to scaling each gene (axis=0) over
    the cells.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    do_center : bool, default True
        Mirrors R's ``do.center``.
    layer_in : str, optional
        Layer to read; defaults to ``adata.X``.
    layer_out : str, default ``"data_scale"``
        Name of the layer that will receive the scaled matrix.  This is
        the canonical key for R's ``chat@data.scale`` slot
        (see :data:`spatialcellchat._anndata.DATA_SLOT_TO_LAYER`).

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    if layer_in is None:
        X = _to_dense(adata.X)
    else:
        X = _to_dense(adata.layers[layer_in])

    # axis=0 -> per-gene over cells (R: column-wise scale on cells x genes)
    if do_center:
        mu = X.mean(axis=0)
        Xc = X - mu
    else:
        Xc = X
    # R uses sd() with (n - 1) denominator and divides centred values by sd
    # whether or not ``center=TRUE``. ``scale=TRUE`` always divides.
    if do_center:
        sd = X.std(axis=0, ddof=1)
    else:
        # When center=FALSE, R's scale() uses sqrt(sum(x^2) / (n-1))
        sd = np.sqrt((X ** 2).sum(axis=0) / max(X.shape[0] - 1, 1))
    # avoid division by zero like R (gives NaN)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = Xc / sd[np.newaxis, :]
    adata.layers[layer_out] = scaled
    return adata


# ---------------------------------------------------------------------------
# runPCA (R: utilities.R:1750)
# ---------------------------------------------------------------------------
def runPCA(
    adata: ad.AnnData,
    do_fast: bool = True,
    dimPC: int = 50,
    seed_use: int = 42,
    weight_by_var: bool = True,
    layer_in: Optional[str] = "data_scale",
    obsm_key: str = "X_pca",
) -> ad.AnnData:
    """Principal components on (cells x features).

    Mirrors R's ``runPCA`` (``utilities.R:1750``). The R signature takes
    samples in rows and features in columns; this Python port reads cells
    from ``adata`` so the orientation is consistent with the rest of the
    package.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    do_fast : bool, default True
        Truncated SVD (mirrors R's ``irlba::irlba``) instead of full PCA.
    dimPC : int, default 50
        Number of components to keep. Capped at ``ncol(data) - 1`` like R.
    seed_use : int, default 42
        Random seed for the truncated SVD path.
    weight_by_var : bool, default True
        When ``True``, scores are multiplied by the singular values
        (fast path) or the squared singular values (full path) — exactly
        what the R source does.
    layer_in : str, optional
        Layer to read. Defaults to ``"data_scale"`` (R's
        ``chat@data.scale`` slot, populated by :func:`scaleData`); pass
        ``None`` to use ``adata.X``.
    obsm_key : str, default ``"X_pca"``
        Where to store the resulting cell embeddings.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata`` with ``obsm[obsm_key]``, ``varm["PCs"]``, and
        ``uns["pca"]`` (containing ``"variance"`` and ``"variance_ratio"``).
    """
    if layer_in is None:
        X = _to_dense(adata.X)
    else:
        X = _to_dense(adata.layers[layer_in])

    n_obs, n_var = X.shape
    dimPC = int(min(dimPC, n_var - 1))

    if do_fast:
        # irlba::irlba(data, nv = dimPC) on cells x genes; matches randomized SVD.
        u, d, vt = randomized_svd(
            X, n_components=dimPC, n_iter=10, random_state=seed_use
        )
        sdev = d / np.sqrt(max(1, n_obs - 1))
        if weight_by_var:
            pc_scores = u * d
        else:
            pc_scores = u
        components = vt  # (dimPC, n_var)
    else:
        # stats::prcomp centers by default and applies SVD on the centred X.
        mu = X.mean(axis=0)
        Xc = X - mu
        u, d, vt = np.linalg.svd(Xc, full_matrices=False)
        u = u[:, :dimPC]
        d = d[:dimPC]
        vt = vt[:dimPC, :]
        sdev = d / np.sqrt(max(1, n_obs - 1))
        if weight_by_var:
            pc_scores = (u * d) * (sdev ** 2)
        else:
            pc_scores = u * d
        components = vt

    adata.obsm[obsm_key] = np.asarray(pc_scores)
    adata.varm["PCs"] = np.asarray(components.T)
    adata.uns["pca"] = {
        "variance": sdev ** 2,
        "variance_ratio": (sdev ** 2) / float(np.sum(sdev ** 2)),
    }
    return adata


# ---------------------------------------------------------------------------
# runUMAP (R: utilities.R:1805)
# ---------------------------------------------------------------------------
def runUMAP(
    adata: ad.AnnData,
    n_neighbors: int = 30,
    n_components: int = 2,
    metric: str = "correlation",
    n_epochs: Optional[int] = None,
    learning_rate: float = 1.0,
    min_dist: float = 0.3,
    spread: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    a: Optional[float] = None,
    b: Optional[float] = None,
    seed_use: int = 42,
    metric_kwds: Optional[dict] = None,
    angular_rp_forest: bool = False,
    verbose: bool = False,
    use_rep: str = "X_pca",
    obsm_key: str = "X_umap",
) -> ad.AnnData:
    """Run UMAP on cell embeddings.

    Mirrors R's ``runUMAP`` (``utilities.R:1805``). The R version applies
    ``umap$fit_transform(t(data.use))`` — i.e. it transposes a
    ``(features × cells)`` matrix to ``(cells × features)`` before
    fitting. This Python port consumes ``adata.obsm[use_rep]`` directly,
    which is **already cells × features** per scverse / AnnData
    convention (rows = observations / cells, cols = the embedding's
    components). **No transpose is applied.**

    Audit G2 / runUMAP orientation note: when porting an R script
    line-by-line, callers who feed a ``(features × cells)`` matrix will
    get the wrong orientation. Always pass a representation whose first
    axis is cells — typically the output of :func:`runPCA`
    (``adata.obsm["X_pca"]``) or :func:`projectData`. If you must use a
    raw ``adata.X``-style matrix, ensure it is in the standard
    ``(n_obs, n_vars)`` AnnData layout (cells along axis 0). This is
    different from R, which operates on the Seurat-default
    ``(features × cells)`` layout and transposes internally.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    n_neighbors, n_components, metric, n_epochs, learning_rate, min_dist,
    spread, set_op_mix_ratio, local_connectivity, repulsion_strength,
    negative_sample_rate, a, b, metric_kwds, angular_rp_forest, verbose
        Passed straight through to :class:`umap.UMAP`.
    seed_use : int, default 42
        Random seed (R's ``seed.use``); becomes ``random_state``.
    use_rep : str, default ``"X_pca"``
        Embedding to embed further. Must be a key in ``adata.obsm`` whose
        value is a ``(n_obs, n_dims)`` matrix (cells × features). Set to
        ``None`` to consume ``adata.X`` (also in cells × features layout).
    obsm_key : str, default ``"X_umap"``
        Where to store the result.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    if use_rep is None:
        data = _to_dense(adata.X)
    else:
        data = np.asarray(adata.obsm[use_rep])

    np.random.seed(seed_use)
    reducer = _umap.UMAP(
        n_neighbors=int(n_neighbors),
        n_components=int(n_components),
        metric=metric,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        min_dist=min_dist,
        spread=spread,
        set_op_mix_ratio=set_op_mix_ratio,
        local_connectivity=local_connectivity,
        repulsion_strength=repulsion_strength,
        negative_sample_rate=negative_sample_rate,
        a=a,
        b=b,
        metric_kwds=metric_kwds,
        angular_rp_forest=angular_rp_forest,
        verbose=verbose,
        random_state=seed_use,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(data)
    adata.obsm[obsm_key] = np.asarray(embedding)
    return adata


# ---------------------------------------------------------------------------
# subsetData (R: utilities.R:1035)
# ---------------------------------------------------------------------------
def subsetData(
    adata: ad.AnnData,
    features: Optional[Sequence[str]] = None,
    db: Optional[dict] = None,
    layer_out: str = "data_signaling",
) -> ad.AnnData:
    """Subset the expression matrix to signaling-relevant genes.

    Mirrors R's ``subsetData`` (``utilities.R:1035``). When ``features`` is
    ``None`` the gene set is the union of all cellchat-database genes
    (``extractGene(db)``), intersected with the genes already present in
    ``adata``.

    Parameters
    ----------
    adata : anndata.AnnData
        The input expression object.
    features : sequence of str, optional
        User-supplied gene set; bypasses the database.
    db : dict, optional
        CellChat database (returned by :func:`spatialcellchat.load_db`). Must
        be supplied when ``features is None``.
    layer_out : str, default ``"data_signaling"``
        Name of the layer in the returned AnnData that holds the same
        sub-matrix (mirrors R's ``object@data.signaling`` slot semantics).

    Returns
    -------
    anndata.AnnData
        A new AnnData restricted to the chosen genes.
    """
    # Mirror R: utilities.R:1037-1050 — annotation handling on
    # ``object@DB$interaction`` happens *before* the gene subset, including
    # auto-adding "Secreted Signaling" for non-RNA datatypes and re-ordering
    # the interaction table by the canonical annotation factor levels.
    if db is not None and "interaction" in db:
        interaction_input = db["interaction"]
        cc_uns = adata.uns.get("cellchat", {})
        opts = cc_uns.get("options", {}) if isinstance(cc_uns, dict) else {}
        datatype = opts.get("datatype", "RNA")
        if datatype != "RNA":
            if "annotation" not in interaction_input.columns:
                warnings.warn(
                    "A column named `annotation` is required in "
                    "`object@DB$interaction` when running CellChat on spatial "
                    "transcriptomics! The `annotation` column is now "
                    "automatically added and all L-R pairs are assigned as "
                    "`Secreted Signaling`, which means that these L-R pairs "
                    "are assumed to mediate diffusion-based cellular "
                    "communication.",
                    stacklevel=2,
                )
                interaction_input = interaction_input.copy()
                interaction_input["annotation"] = "Secreted Signaling"
        if "annotation" in interaction_input.columns:
            uniq_ann = pd.unique(interaction_input["annotation"].astype(str))
            if len(uniq_ann) > 1:
                ann_levels = [
                    "Secreted Signaling",
                    "ECM-Receptor",
                    "Non-protein Signaling",
                    "Cell-Cell Contact",
                ]
                interaction_input = interaction_input.copy()
                interaction_input["annotation"] = pd.Categorical(
                    interaction_input["annotation"].astype(str),
                    categories=ann_levels,
                    ordered=True,
                )
                interaction_input = interaction_input.sort_values("annotation")
                interaction_input["annotation"] = interaction_input[
                    "annotation"
                ].astype(str)
            db["interaction"] = interaction_input

    if features is None:
        if db is None:
            raise ValueError(
                "Either `features` or `db` must be provided to subsetData."
            )
        gene_use_input = extractGene(db)
        gene_use = [g for g in gene_use_input if g in adata.var_names]
    else:
        gene_use = [g for g in features if g in adata.var_names]

    sub = adata[:, gene_use].copy()
    sub.layers[layer_out] = sub.X.copy()
    return sub


# ---------------------------------------------------------------------------
# updateClusterLabels (R: utilities.R:995)
# ---------------------------------------------------------------------------
def updateClusterLabels(
    adata: ad.AnnData,
    old_cluster_name: Optional[Sequence[str]] = None,
    new_cluster_name: Optional[Sequence[str]] = None,
    new_order: Optional[Sequence[str]] = None,
    new_cluster_metaname: str = "new.labels",
    ident_key: str = "cellchat_idents",
) -> ad.AnnData:
    """Rename and/or reorder cell-group labels.

    Mirrors R's ``updateClusterLabels`` (``utilities.R:995``).  After
    relabeling / reordering, R re-runs ``computeCommunProbPathway``,
    ``aggregateNet`` and ``netAnalysis_computeCentrality`` so the
    ``net`` / ``netP`` / centrality slots stay consistent with the new
    labels; this Python port mirrors that behaviour when the relevant
    slots are present (``adata.uns['cellchat']['net']`` exists).  When
    the cellchat pipeline has not been run yet (no ``net`` slot), the
    downstream re-runs are skipped — only the relabel / reorder takes
    effect.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    old_cluster_name : sequence of str, optional
        Current labels; defaults to the existing categories in
        ``adata.obs[ident_key]``.
    new_cluster_name : sequence of str, optional
        Replacement labels matching ``old_cluster_name`` element-wise.
    new_order : sequence of str, optional
        Target category order after renaming.
    new_cluster_metaname : str, default ``"new.labels"``
        Name under which the renamed series is also kept in ``adata.obs``
        (matches R's behaviour of writing back to the meta slot).
    ident_key : str, default ``"cellchat_idents"``
        Column in ``adata.obs`` that holds the active cell identities.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    if ident_key not in adata.obs.columns:
        raise KeyError(
            f"adata.obs is missing the active identity column '{ident_key}'."
        )
    current = adata.obs[ident_key]
    if not isinstance(current.dtype, pd.CategoricalDtype):
        current = pd.Categorical(current)
        adata.obs[ident_key] = current

    if old_cluster_name is None:
        old_cluster_name = list(adata.obs[ident_key].cat.categories)

    if new_cluster_metaname in adata.obs.columns:
        raise ValueError(
            f"Please define another `new_cluster_metaname` as '"
            f"{new_cluster_metaname}' already exists in adata.obs."
        )

    if new_cluster_name is not None:
        if len(new_cluster_name) != len(old_cluster_name):
            raise ValueError(
                "`new_cluster_name` and `old_cluster_name` must have the "
                "same length."
            )
        mapping = dict(zip(old_cluster_name, new_cluster_name))
        relabeled = adata.obs[ident_key].map(lambda x: mapping.get(x, x))
        relabeled = pd.Categorical(relabeled)
        adata.obs[new_cluster_metaname] = relabeled
        adata.obs[ident_key] = relabeled

    if new_order is not None:
        cur = adata.obs[ident_key]
        adata.obs[ident_key] = pd.Categorical(
            cur, categories=list(new_order), ordered=False
        )

    if "0" in [str(c) for c in adata.obs[ident_key].cat.categories]:
        raise ValueError("Cell labels cannot contain `0`!")

    # ── R parity: re-run the three downstream stages so net / netP /
    #    centrality slots stay coherent with the new labels.
    #    R: utilities.R:1014-1019.
    cc_existing = adata.uns.get("cellchat") or {}
    if "net" in cc_existing:
        # Local imports avoid a top-level cycle: modeling imports from
        # preprocessing for ``_get_idents``.
        from .modeling import aggregateNet, computeCommunProbPathway
        from .analysis_network import netAnalysis_computeCentrality

        computeCommunProbPathway(adata)
        aggregateNet(adata)
        if "netP" in adata.uns["cellchat"]:
            netAnalysis_computeCentrality(adata, slot_name="netP")

    return adata


# ---------------------------------------------------------------------------
# sketchData (R: utilities.R:867)
# ---------------------------------------------------------------------------
def sketchData(
    adata: ad.AnnData,
    percent: float,
    idents: Optional[Sequence[str]] = None,
    do_PCA: bool = True,
    dimPC: int = 30,
    ident_key: str = "cellchat_idents",
) -> List[str]:
    """Geometric-sketch downsampling — return the kept cell names.

    Mirrors R's ``sketchData`` (``utilities.R:867``), which returns a
    character vector of selected cell names (``sketch.cells``). Requires
    the ``geosketch`` Python package, which is exactly what the R version
    imports through reticulate.

    R-bug fix (intentional deviation per the project's R-bug-handling
    decision rule — "honour R's documented intent, not R's typos"):
    R's ``sketchData`` non-Seurat branch reads
    ``sketch.size <- as.integer(percent * nrow(X))`` —
    upstream ``CellChat::utilities.R:111`` and fork
    ``SpatialCellChat::utilities.R:898`` — but the variable ``X`` is
    **undefined** in that scope (only ``X.pcs`` and ``object`` exist).
    Calling that R branch raises ``Error: object 'X' not found``. The
    intent (per the comment ``# Sketch percent of data.``) is clearly to
    sketch a fraction of the input matrix's rows, i.e. ``nrow(object)``.
    The Python port reads ``sub.n_obs`` directly — equivalent to the
    intended ``nrow(object)`` — so the function actually runs.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object.
    percent : float
        Fraction of cells to keep.
    idents : sequence of str, optional
        Identity classes to retain prior to sketching.
    do_PCA : bool, default True
        Whether to run :func:`runPCA` before sketching.
    dimPC : int, default 30
        Number of PCs to use when ``do_PCA`` is True.
    ident_key : str, default ``"cellchat_idents"``
        Identity column in ``adata.obs`` consulted when ``idents`` is set.

    Returns
    -------
    list[str]
        Cell names selected by the geometric sketch (R parity).
    """
    import geosketch as _geosketch

    if idents is not None:
        mask = adata.obs[ident_key].isin(list(idents)).values
        sub = adata[mask].copy()
    else:
        sub = adata

    sketch_size = int(percent * sub.n_obs)

    if do_PCA:
        runPCA(sub, dimPC=dimPC, layer_in=None)
        X_pcs = np.asarray(sub.obsm["X_pca"])
    else:
        X_pcs = _to_dense(sub.X)

    sketch_index = _geosketch.gs(X_pcs, sketch_size)
    sketch_index = np.asarray(sketch_index, dtype=int).ravel()
    return list(sub.obs_names[sketch_index].tolist())


def sketchDataAdata(
    adata: ad.AnnData,
    percent: float,
    idents: Optional[Sequence[str]] = None,
    do_PCA: bool = True,
    dimPC: int = 30,
    ident_key: str = "cellchat_idents",
) -> ad.AnnData:
    """Convenience wrapper around :func:`sketchData` returning the
    sketched sub-AnnData.

    Use this when downstream code expects an :class:`anndata.AnnData`
    rather than a list of cell names. Behaviour matches the previous
    Python sketchData; the new R-faithful :func:`sketchData` returns the
    list of names instead (mirroring R's ``sketch.cells`` return).
    """
    import geosketch as _geosketch

    if idents is not None:
        mask = adata.obs[ident_key].isin(list(idents)).values
        sub = adata[mask].copy()
    else:
        sub = adata

    sketch_size = int(percent * sub.n_obs)

    if do_PCA:
        runPCA(sub, dimPC=dimPC, layer_in=None)
        X_pcs = np.asarray(sub.obsm["X_pca"])
    else:
        X_pcs = _to_dense(sub.X)

    sketch_index = _geosketch.gs(X_pcs, sketch_size)
    sketch_index = np.asarray(sketch_index, dtype=int).ravel()
    return sub[sketch_index].copy()


# ---------------------------------------------------------------------------
# preProcessing (R: utilities.R:744)
# ---------------------------------------------------------------------------
def preProcessing(
    adata: ad.AnnData,
    layer_in: Optional[str] = None,
    layer_out: str = "data_signaling",
    K: Optional[int] = None,
    noise_start: Optional[int] = None,
    quantile_prob: float = 1e-5,
) -> ad.AnnData:
    """Run the ALRA-based imputation step from the R reference.

    Mirrors R's ``preProcessing`` (``utilities.R:744``). The function
    bootstraps :func:`_alra.choose_k` ten times, takes the median ``k``,
    then calls :func:`_alra.alra` once and writes the imputed
    ``A_norm_rank_k_cor_sc`` (cells x genes) into ``adata.layers[layer_out]``.

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    layer_in : str, optional
        Layer to read. ``None`` uses ``adata.X``.
    layer_out : str, default ``"data_signaling"``
        Layer name to overwrite with the ALRA-imputed values.
    K, noise_start : int, optional
        Override the values that would otherwise be derived from the input
        shape (``round(0.9 * min(dim(data)))`` etc., as in R).
    quantile_prob : float, default 1e-5
        Per-gene mirror-trick threshold quantile (R: ``quantile.prob``).

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``. ``adata.uns["alra_k"]`` holds the chosen rank.
    """
    if layer_in is None:
        X = _to_dense(adata.X)
    else:
        X = _to_dense(adata.layers[layer_in])

    # AnnData stores cells x genes — exactly what ALRA expects.
    A = np.asarray(X, dtype=float)

    # Mirror R: K defaults to formals(ALRA::choose_k)$K (=100) unless
    # min(dim(data)) is too small; noise_start defaults to 80 capped by 0.8K.
    default_K = 100
    default_noise = 80
    if K is None:
        K = (
            round(0.9 * min(A.shape))
            if min(A.shape) < default_K
            else default_K
        )
    if noise_start is None:
        noise_start = int(min(round(0.8 * K), default_noise))

    # R errors when K is too small to do the random-projection rank check;
    # we let ``_alra.choose_k`` raise the same way rather than silently
    # papering over the failure with ``k_med = K``.
    k_boot: List[int] = []
    for _ in range(10):
        k_boot.append(
            _alra.choose_k(A, K=int(K), noise_start=int(noise_start))["k"]
        )
    k_med = int(np.median(k_boot))

    res = _alra.alra(A, k=k_med, q=10, quantile_prob=quantile_prob)
    adata.layers[layer_out] = res["A_norm_rank_k_cor_sc"]
    adata.uns["alra_k"] = k_med
    return adata


# ---------------------------------------------------------------------------
# projectData and helpers (R: utilities.R:1603..1734)
# ---------------------------------------------------------------------------
def _l1NormalizeRows(A: np.ndarray) -> np.ndarray:
    """Row-normalise ``A`` to unit L1 norm.

    Mirrors R's private ``l1NormalizeRows``.
    """
    rs = A.sum(axis=1)
    return A / rs[:, np.newaxis]


def _l1NormalizeColumns(A: np.ndarray) -> np.ndarray:
    """Column-normalise ``A`` to unit L1 norm.

    Mirrors R's private ``l1NormalizeColumns``.
    """
    cs = A.sum(axis=0)
    return A / cs[np.newaxis, :]


def _projectOnNetwork(
    gene_expression: np.ndarray,
    new_features: Sequence[str],
    row_index: Sequence[str],
    missing_value: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """Project gene expression matrix onto the row space of the network.

    Returns the projected matrix and the new row names (``new_features``).
    """
    df = pd.DataFrame(gene_expression, index=list(row_index))
    out = pd.DataFrame(
        np.full((len(new_features), gene_expression.shape[1]), missing_value),
        index=list(new_features),
        columns=df.columns,
    )
    common = [g for g in new_features if g in df.index]
    if common:
        out.loc[common] = df.loc[common].values
    return out.values, list(new_features)


def _randomWalkBySolve(
    E: np.ndarray,
    A: np.ndarray,
    alpha: float,
    normalize_adj: str = "rows",
) -> np.ndarray:
    """Solve ``(I - alpha A_norm) x = (1 - alpha) E`` for ``x``.

    Mirrors R's private ``randomWalkBySolve``. Uses
    :func:`scipy.sparse.linalg.spsolve` when the LHS is sparse, falling back
    to :func:`numpy.linalg.solve` otherwise.
    """
    if normalize_adj == "rows":
        A_norm = _l1NormalizeRows(A)
    elif normalize_adj == "columns":
        A_norm = _l1NormalizeColumns(A)
    else:
        raise ValueError("normalize_adj must be 'rows' or 'columns'")

    n = A.shape[0]
    LHS = np.eye(n) - alpha * A_norm
    RHS = (1.0 - alpha) * E
    if sparse.issparse(LHS):
        return np.asarray(spsolve(LHS, RHS))
    return np.linalg.solve(LHS, RHS)


def _projectFromNetworkRecombine(
    original_expression: np.ndarray,
    projected_expression: np.ndarray,
    original_index: Sequence[str],
    projected_index: Sequence[str],
) -> np.ndarray:
    """Combine projected rows with un-projected rows.

    Mirrors R's private ``projectFromNetworkRecombine``.
    """
    out = original_expression.copy().astype(float)
    orig_pos = {g: i for i, g in enumerate(original_index)}
    for j, g in enumerate(projected_index):
        if g in orig_pos:
            out[orig_pos[g], :] = projected_expression[j, :]
    return out


def _projectAndRecombine(
    gene_expression: np.ndarray,
    adj_matrix: np.ndarray,
    alpha: float,
    gene_names: Sequence[str],
    adj_names: Sequence[str],
    normalize_adj: str = "rows",
) -> np.ndarray:
    """Glue the three private helpers above.

    Mirrors R's private ``projectAndRecombine``.
    """
    proj_in_A, new_index = _projectOnNetwork(
        gene_expression, list(adj_names), list(gene_names)
    )
    proj_done = _randomWalkBySolve(
        proj_in_A, adj_matrix, alpha, normalize_adj=normalize_adj
    )
    return _projectFromNetworkRecombine(
        gene_expression, proj_done, list(gene_names), new_index
    )


def projectData(
    adata: ad.AnnData,
    adjMatrix: Union[np.ndarray, sparse.spmatrix, pd.DataFrame],
    alpha: float = 0.5,
    normalizeAdjMatrix: str = "rows",
    layer_in: str = "data_signaling",
    layer_out: str = "data_project",
    adj_gene_names: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """Project the signaling-gene expression matrix on a PPI graph.

    Mirrors R's ``projectData`` (``utilities.R:1603``). Solves the
    network-smoothing system ``(I - α A_norm) x = (1 - α) E`` using
    :func:`scipy.sparse.linalg.spsolve` (the linear-algebra equivalent of
    R's ``solve``).

    Parameters
    ----------
    adata : anndata.AnnData
        Input object, modified in-place.
    adjMatrix : ndarray, sparse matrix, or DataFrame
        PPI adjacency matrix. Must be square and have non-zero row and
        column sums. When a :class:`pandas.DataFrame` is supplied, its index
        is used as ``adj_gene_names``.
    alpha : float, default 0.5
        Restart probability complement (R's ``alpha``). Must lie in (0, 1).
    normalizeAdjMatrix : str, default ``"rows"``
        ``"rows"`` or ``"columns"``.
    layer_in : str, default ``"data_signaling"``
        Layer that holds the (cells x genes) expression to project.
    layer_out : str, default ``"data_project"``
        Layer name for the projected matrix (cells x genes).  Canonical
        store for R's ``chat@data.project`` slot
        (see :data:`spatialcellchat._anndata.DATA_SLOT_TO_LAYER`).
    adj_gene_names : sequence of str, optional
        Required when ``adjMatrix`` is a plain ``ndarray``; ignored for
        DataFrames.

    Returns
    -------
    anndata.AnnData
        The mutated ``adata``.
    """
    if not (isinstance(alpha, (int, float)) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be a number in (0, 1).")
    if normalizeAdjMatrix not in ("rows", "columns"):
        raise ValueError("normalizeAdjMatrix must be 'rows' or 'columns'.")

    # Resolve adjacency to a dense ndarray and obtain its gene labels.
    if isinstance(adjMatrix, pd.DataFrame):
        adj = adjMatrix.values.astype(float)
        adj_names = list(adjMatrix.index)
    else:
        adj = _to_dense(adjMatrix)
        if adj_gene_names is None:
            raise ValueError(
                "adj_gene_names is required when adjMatrix is a plain ndarray."
            )
        adj_names = list(adj_gene_names)

    if adj.shape[0] != adj.shape[1]:
        raise ValueError("adjMatrix must be square.")
    if (adj.sum(axis=1) == 0).any() or (adj.sum(axis=0) == 0).any():
        raise ValueError("PPI cannot have zero rows/columns.")

    # AnnData stores cells x genes; the R helper expects genes x cells.
    E_cells_genes = _to_dense(adata.layers[layer_in])
    E_genes_cells = E_cells_genes.T
    gene_names = list(adata.var_names)

    projected = _projectAndRecombine(
        E_genes_cells, adj, alpha, gene_names, adj_names,
        normalize_adj=normalizeAdjMatrix,
    )
    adata.layers[layer_out] = projected.T  # back to cells x genes
    return adata
