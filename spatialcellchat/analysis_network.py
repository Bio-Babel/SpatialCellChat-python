"""Network analysis exports (signaling network similarity, embedding, ...).

Direct port of the public network-analysis API in
``SpatialCellChat::analysis.R``:

- ``netAnalysis_computeCentrality`` (R: ``analysis.R:82``)
- ``computeLaplacian`` (R: ``analysis.R:1720``)
- ``computeEigengap`` (R: ``analysis.R:1669``)
- ``computeNetSimilarity`` (R: ``analysis.R:1315``)
- ``computeNetSimilarityPairwise`` (R: ``analysis.R:1391``)
- ``netEmbedding`` (R: ``analysis.R:1503``)
- ``netClustering`` (R: ``analysis.R:1564``)
- ``netMappingDEG`` (R: ``analysis.R:3222``)

The public surface accepts an :class:`anndata.AnnData` first argument and
writes results into ``adata.uns["cellchat"]``. Slot keys mirror the R
``cellchat@netP$...`` layout so that downstream readers stay portable.

Helper :func:`computeNetD_structure` (R: ``analysis.R:2051``) is also
exported because both ``computeNetSimilarity`` and
``computeNetSimilarityPairwise`` consume it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
import umap as _umap
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from ._anndata import cellchat_dict as _ensure_uns
from ._centrality import (
    CENTRALITY_NAMES_DEGREE,
    CENTRALITY_NAMES_FULL,
    computeCentralityLocal,
)
from ._snn import buildSNN


def _idents_levels(adata: ad.AnnData) -> Optional[List[str]]:
    """Return ``levels(cellchat@idents)`` from adata.obs categoricals.

    Mirrors R's ``levels(cellchat@idents)``. Returns ``None`` if no idents
    column is found (callers fall back to slot dimnames or integer indices).
    """
    if adata is None:
        return None
    for k in ("cellchat_idents", "active_ident", "cell.type"):
        if k in adata.obs.columns:
            s = adata.obs[k]
            if isinstance(s.dtype, pd.CategoricalDtype):
                return list(s.cat.categories.astype(str))
            return list(pd.Categorical(s).categories.astype(str))
    return None

__all__ = [
    "netAnalysis_computeCentrality",
    "computeLaplacian",
    "computeEigengap",
    "computeNetSimilarity",
    "computeNetSimilarityPairwise",
    "netEmbedding",
    "netClustering",
    "netMappingDEG",
    "computeNetD_structure",
    "computeCommunField",
    "get_dens",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_slot(adata: ad.AnnData, slot_name: str) -> Dict[str, Any]:
    """Return ``adata.uns["cellchat"][slot_name]`` (creating empty dict on miss)."""
    cc = _ensure_uns(adata)
    if slot_name not in cc:
        cc[slot_name] = {}
    return cc[slot_name]


def _entropia(a: np.ndarray) -> float:
    """Direct port of R's ``entropia`` (``R/analysis.R:1966``)."""
    a = np.asarray(a, dtype=float).ravel()
    a = a[a > 0]
    if a.size == 0:
        return 0.0
    return float(-np.sum(a * np.log(a)))


def _node_distance(g: ig.Graph) -> np.ndarray:
    """Direct port of R's ``node_distance`` (``R/analysis.R:1976``).

    Returns an ``n x n`` matrix where ``a[i, q]`` (0-based ``i``,
    1-based ``q``) is the count of nodes at distance ``q`` from node
    ``i+1``, divided by ``n - 1``. Unreachable vertices are mapped to
    distance ``n`` (R's ``m[which(m=="Inf")] <- n``). The diagonal
    (distance 0) is not represented.
    """
    n = g.vcount()
    if n == 1:
        return np.array([[1.0]])
    # R: igraph::shortest.paths(g, algorithm="unweighted") defaults to
    # mode="all" (treat as undirected). python-igraph defaults to "out"
    # so we must pass mode="all" explicitly for parity.
    sp = np.asarray(g.distances(mode="all"), dtype=float)  # n x n
    sp[np.isinf(sp)] = float(n)
    a = np.zeros((n, n))
    # Iterate over distinct non-zero values present in sp.
    quem_vals = np.unique(sp[sp > 0]).astype(int)
    for q in quem_vals:
        # R logic: linhas = column index of every cell where m == q (1-based);
        # then hist(linhas, breaks=0..n)$counts gives the per-column count.
        # That's exactly: how many rows in column c have value q.
        # NOTE: R's ``which(m == quem[j])`` reads the matrix in column-major
        # order, so ``linhas`` ends up enumerating COLUMN indices. But the
        # assignment is ``a[1:n, quem[j]] <- counts``, populating the q-th
        # column with a histogram over the n column indices. Hence:
        col_counts = (sp == q).sum(axis=0)
        a[:, q - 1] = col_counts
    a = a / float(n - 1)
    return a


def _nnd(g: ig.Graph) -> np.ndarray:
    """Direct port of R's ``nnd`` (``R/analysis.R:2009``).

    Returns a length-(N+1) vector. The first ``N`` entries are the
    column means of :func:`_node_distance`'s per-column distance
    histograms; the last entry is a normalised entropy term (R's
    ``max(0, entropia(pdfm) - entropia(as.matrix(nd))/N)/norm``).
    """
    N = g.vcount()
    if N == 1:
        return np.array([1.0, 0.0])

    nd = _node_distance(g)
    pdfm = nd.mean(axis=0)  # length N
    nonzero = int(np.sum(pdfm[: N - 1] > 0))
    norm = np.log(max(2.0, nonzero + 1.0))
    # entropia(as.matrix(nd)) treats the whole matrix as a vector
    flat_entropia = _entropia(nd)
    val = max(0.0, _entropia(pdfm) - flat_entropia / float(N)) / norm
    return np.concatenate([pdfm, [val]])


def _alpha_centrality(g: ig.Graph) -> np.ndarray:
    """Direct port of R's ``alpha_centrality`` (``R/analysis.R:2027``)."""
    n = g.vcount()
    if n == 1:
        return np.array([1.0])
    # alpha = 1/N, exo = degree(g)/(N-1)
    deg = np.asarray(g.degree(mode="all"), dtype=float)
    exo = deg / float(n - 1)
    # python-igraph does not expose alpha_centrality directly; use numpy linear solve
    # alpha-centrality: x = (I - alpha A^T)^-1 exo
    A = np.asarray(g.get_adjacency().data, dtype=float)
    M = np.eye(n) - (1.0 / float(n)) * A.T
    x = np.linalg.solve(M, exo)
    x = np.sort(x) / float(n * n)
    last = max(0.0, 1.0 - x.sum())
    return np.concatenate([x, [last]])


def computeNetD_structure(
    g: np.ndarray,
    h: np.ndarray,
    w1: float = 0.45,
    w2: float = 0.45,
    w3: float = 0.1,
) -> float:
    """Compute structural distance between two signaling networks.

    Mirrors ``SpatialCellChat::computeNetD_structure``
    (``R/analysis.R:2051``). Used internally by
    :func:`computeNetSimilarity` and :func:`computeNetSimilarityPairwise`.
    """
    g_arr = np.asarray(g, dtype=float)
    h_arr = np.asarray(h, dtype=float)

    G = ig.Graph.Adjacency((g_arr > 0).tolist(), mode="directed")
    H = ig.Graph.Adjacency((h_arr > 0).tolist(), mode="directed")

    N = G.vcount()
    M = H.vcount()

    PM = np.zeros(max(M, N))
    first = 0.0
    second = 0.0
    third = 0.0

    if w1 + w2 > 0:
        pg = _nnd(G)
        ph = _nnd(H)

        # Replicate R indexing: PM[1:(N-1)] = pg[1:(N-1)] (1-based) ->
        # PM[0:N-1] = pg[0:N-1] in 0-based.
        PM[: N - 1] = pg[: N - 1]
        PM[-1] = pg[N - 1] if N - 1 < len(pg) else 0.0

        if M >= 1:
            PM[: M - 1] = PM[: M - 1] + ph[: M - 1]
        PM[-1] = PM[-1] + (ph[M - 1] if M - 1 < len(ph) else 0.0)

        PM = PM / 2.0

        rhs = (
            _entropia(PM)
            - (_entropia(pg[:N]) + _entropia(ph[:M])) / 2.0
        ) / np.log(2.0)
        first = float(np.sqrt(max(rhs, 0.0)))
        # second uses pg[N+1] == pg[N] (0-based) and ph[M+1] == ph[M]
        pg_last = pg[N] if N < len(pg) else 0.0
        ph_last = ph[M] if M < len(ph) else 0.0
        second = float(abs(np.sqrt(max(pg_last, 0.0)) - np.sqrt(max(ph_last, 0.0))))

    if w3 > 0:
        pg = _alpha_centrality(G)
        ph = _alpha_centrality(H)
        m = max(len(pg), len(ph))

        Pg = np.zeros(m)
        Ph = np.zeros(m)
        Pg[m - len(pg) :] = pg
        Ph[m - len(ph) :] = ph

        third = (
            np.sqrt(
                max(
                    (
                        _entropia((Pg + Ph) / 2.0)
                        - (_entropia(pg) + _entropia(ph)) / 2.0
                    )
                    / np.log(2.0),
                    0.0,
                )
            )
            / 2.0
        )

        Gc = G.complementer(loops=False)
        Hc = H.complementer(loops=False)
        pg2 = _alpha_centrality(Gc)
        ph2 = _alpha_centrality(Hc)
        m2 = max(len(pg2), len(ph2))
        Pg2 = np.zeros(m2)
        Ph2 = np.zeros(m2)
        Pg2[m2 - len(pg2) :] = pg2
        Ph2[m2 - len(ph2) :] = ph2
        third = third + np.sqrt(
            max(
                (
                    _entropia((Pg2 + Ph2) / 2.0)
                    - (_entropia(pg2) + _entropia(ph2)) / 2.0
                )
                / np.log(2.0),
                0.0,
            )
        ) / 2.0

    return float(w1 * first + w2 * second + w3 * third)


# ---------------------------------------------------------------------------
# computeLaplacian (R: analysis.R:1720)
# ---------------------------------------------------------------------------
def computeLaplacian(
    CM: Union[np.ndarray, sparse.spmatrix],
    tol: float = 0.01,
) -> Dict[str, Any]:
    """Compute eigenvalues of the normalised graph Laplacian of ``CM``.

    Mirrors ``SpatialCellChat::computeLaplacian``
    (``R/analysis.R:1720``).

    Parameters
    ----------
    CM : numpy.ndarray | scipy.sparse.spmatrix
        Square consensus / similarity matrix.
    tol : float, default 0.01
        Tolerance below which an eigenvalue is considered zero (used by
        ``computeEigengap`` to estimate the lower bound on the cluster
        count).

    Returns
    -------
    dict
        ``{"val": np.ndarray, "n_zeros": int}`` where ``val`` is the
        sorted vector of eigen-magnitudes and ``n_zeros`` is the number
        of values ``<= tol``.
    """
    if sparse.issparse(CM):
        CM = np.asarray(CM.toarray(), dtype=float)
    else:
        CM = np.asarray(CM, dtype=float)
    n = CM.shape[0]
    Dsq = np.sqrt(CM.sum(axis=0))
    # Match R: L = -t(CM / Dsq) / Dsq  (broadcast: row-divide then col-divide)
    L = -((CM / Dsq).T) / Dsq
    L = L + np.diag(np.ones(n))  # 1 + diag(L) per R
    # NOTE: R does NOT symmetrise here; eigs_sym tolerates numerical
    # asymmetry. computeEigengap symmetrises post-truncation, so a
    # second forced-symmetrise here would be redundant. We keep parity
    # with R by leaving L unmodified.

    numEigs = int(min(100, n))
    if numEigs == n:
        # R: when k == nrow it falls back to base eigen()
        ev = np.linalg.eigvalsh(L)
        eigs = np.abs(ev.real)
    else:
        # SciPy eigsh with which="SM" matches R RSpectra::eigs_sym
        ev = eigsh(
            sparse.csr_matrix(L),
            k=numEigs,
            which="SM",
            tol=1e-4,
            return_eigenvectors=False,
        )
        eigs = np.abs(ev.real)
    eigs_sorted = np.sort(eigs)
    n_zeros = int((eigs_sorted <= tol).sum())
    return {"val": eigs_sorted, "n_zeros": n_zeros}


# ---------------------------------------------------------------------------
# computeEigengap (R: analysis.R:1669)
# ---------------------------------------------------------------------------
def computeEigengap(
    CM: Union[np.ndarray, sparse.spmatrix],
    tau: Optional[float] = None,
    tol: float = 0.01,
) -> Dict[str, Any]:
    """Estimate the number of clusters via Laplacian eigengap.

    Mirrors ``SpatialCellChat::computeEigengap``
    (``R/analysis.R:1669``).

    Parameters
    ----------
    CM : numpy.ndarray | scipy.sparse.spmatrix
        Consensus matrix.
    tau : float, optional
        Truncation threshold; estimated from ``K.init`` when missing
        (R defaults: 0.3 / 0.4 / 0.5 for ``K.init <= 5 / 10 / >10``).
    tol : float, default 0.01
        Tolerance for counting near-zero eigenvalues.

    Returns
    -------
    dict
        ``{"upper_bound": int, "lower_bound": int, "eigs": dict,
        "upper_bound_ties": numpy.ndarray}``. ``eigs`` is the dict
        returned by :func:`computeLaplacian`. ``upper_bound`` is the
        first 1-based index where ``diff(eigs)`` attains its maximum.
        ``upper_bound_ties`` lists every 1-based index that ties for the
        maximum (R's ``which(gaps == max(gaps))``); downstream callers
        should consume the first or all per their need. The ``"gg.obj"``
        slot in the R return is dropped — plotting is delegated to the
        visualisation slice.
    """
    if sparse.issparse(CM):
        CM = np.asarray(CM.toarray(), dtype=float)
    else:
        CM = np.asarray(CM, dtype=float).copy()

    K_init = computeLaplacian(CM, tol=tol)["n_zeros"]
    if tau is None:
        if K_init <= 5:
            tau = 0.3
        elif K_init <= 10:
            tau = 0.4
        else:
            tau = 0.5

    # truncate the ensemble consensus matrix
    CM[CM <= tau] = 0.0
    # normalise + symmetrise
    CM = (CM + CM.T) / 2.0

    eigs = computeLaplacian(CM, tol=tol)
    gaps = np.diff(eigs["val"])
    # R: which(gaps == max(gaps)) returns ALL tied indices in 1-based.
    # Expose every tie under upper_bound_ties so callers can choose
    # first / all per need; upper_bound mirrors R's downstream
    # `min(which(...))` consumers (first tie).
    tie_idx_1based = np.flatnonzero(gaps == gaps.max()) + 1
    upper_bound = int(tie_idx_1based[0])
    lower_bound = int(eigs["n_zeros"])
    return {
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "eigs": eigs,
        "upper_bound_ties": tie_idx_1based.astype(int),
    }


# ---------------------------------------------------------------------------
# computeNetSimilarity (R: analysis.R:1315)
# ---------------------------------------------------------------------------
def _build_signaling_similarity_matrix(
    prob: np.ndarray,
    type_: str,
    thresh: Optional[float] = None,
) -> np.ndarray:
    """Construct the signaling similarity matrix for a single dataset.

    Implements the R loop body in ``computeNetSimilarity`` (lines
    1331-1359) for ``type %in% {"functional", "structural"}``.
    """
    prob = np.asarray(prob, dtype=float).copy()
    n_signal = prob.shape[2]

    if thresh is not None:
        nonzero = prob[prob != 0]
        if nonzero.size:
            cutoff = float(np.quantile(nonzero, float(thresh)))
            prob[prob < cutoff] = 0.0

    S = np.zeros((n_signal, n_signal))

    if type_ == "functional":
        for i in range(n_signal - 1):
            Gi = (prob[:, :, i] > 0).astype(float)
            for j in range(i + 1, n_signal):
                Gj = (prob[:, :, j] > 0).astype(float)
                num = float((Gi * Gj).sum())
                denom = float((Gi + Gj - Gi * Gj).sum())
                if denom == 0:
                    S[i, j] = 0.0
                else:
                    S[i, j] = num / denom
        S = S + S.T
        np.fill_diagonal(S, 1.0)
    elif type_ == "structural":
        D = np.zeros((n_signal, n_signal))
        for i in range(n_signal - 1):
            Gi = (prob[:, :, i] > 0).astype(float)
            for j in range(i + 1, n_signal):
                Gj = (prob[:, :, j] > 0).astype(float)
                D[i, j] = computeNetD_structure(Gi, Gj)
        D[~np.isfinite(D)] = 0.0
        D = D + D.T
        S = 1.0 - D
    else:
        raise ValueError(f"Unknown type {type_!r}; expected 'functional' or 'structural'.")
    return S


def computeNetSimilarity(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    k: Optional[int] = None,
    thresh: Optional[float] = None,
) -> ad.AnnData:
    """Compute the signaling-network similarity matrix.

    Mirrors ``SpatialCellChat::computeNetSimilarity``
    (``R/analysis.R:1315``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose ``adata.uns["cellchat"][slot_name]["prob"]`` array
        of shape ``(K, K, N)`` is consumed.
    slot_name : str, default ``"netP"``
        Slot to read ``prob`` from (``"netP"`` for pathway-level data
        or ``"net"`` for L-R-pair-level data).
    type_ : {"functional", "structural"}, default "functional"
        Which similarity to compute. Mirrors R's ``type`` argument.
    k : int, optional
        Number of nearest neighbours when smoothing via SNN. Default
        formula matches R: ``ceil(sqrt(N))`` for ``N <= 25``,
        ``ceil(sqrt(N)) + 1`` otherwise.
    thresh : float, optional
        Quantile in ``[0, 0.25]`` used to trim weak interactions before
        the per-pair similarity computation.

    Returns
    -------
    anndata.AnnData
        ``adata`` augmented with
        ``uns["cellchat"][slot_name]["similarity"][type_]["matrix"]["single"]``.
    """
    slot = _get_slot(adata, slot_name)
    if "prob" not in slot:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['prob'] is required (run "
            "computeCommunProb / computeCommunProbPathway first)."
        )
    prob = np.asarray(slot["prob"], dtype=float)
    if prob.ndim != 3:
        raise ValueError("`prob` must be a 3-D array of shape (K, K, N).")

    n_signal = prob.shape[2]
    if k is None:
        k = int(np.ceil(np.sqrt(n_signal)))
        if n_signal > 25:
            k = k + 1

    S = _build_signaling_similarity_matrix(prob, type_, thresh=thresh)
    # Smooth via SNN — R passes S transposed under the hood: buildSNN treats
    # columns as samples, S is symmetric square so it doesn't matter.
    SNN = buildSNN(S, k=int(k), k_scale=10, prune_SNN=1.0 / 15.0).toarray()
    Similarity = S * SNN

    pathway_names = list(slot.get("pathway_names", [str(i) for i in range(n_signal)]))
    if len(pathway_names) != n_signal:
        pathway_names = [str(i) for i in range(n_signal)]
    Similarity_df = pd.DataFrame(Similarity, index=pathway_names, columns=pathway_names)

    if "similarity" not in slot or not isinstance(slot["similarity"], dict):
        slot["similarity"] = {}
    if type_ not in slot["similarity"] or not isinstance(slot["similarity"][type_], dict):
        slot["similarity"][type_] = {}
    if "matrix" not in slot["similarity"][type_] or not isinstance(
        slot["similarity"][type_]["matrix"], dict
    ):
        slot["similarity"][type_]["matrix"] = {}
    slot["similarity"][type_]["matrix"]["single"] = Similarity_df
    return adata


# ---------------------------------------------------------------------------
# computeNetSimilarityPairwise (R: analysis.R:1391)
# ---------------------------------------------------------------------------
def computeNetSimilarityPairwise(
    adata_list: Sequence[ad.AnnData],
    slot_name: str = "netP",
    type_: str = "functional",
    comparison: Optional[Sequence[int]] = None,
    k: Optional[int] = None,
    thresh: Optional[float] = None,
    dataset_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute the joint signaling-network similarity across datasets.

    Mirrors ``SpatialCellChat::computeNetSimilarityPairwise``
    (``R/analysis.R:1391``). Because the Python port has no merged
    AnnData object, the API takes a list of AnnData objects together
    with optional ``dataset_names``.

    Parameters
    ----------
    adata_list : sequence of anndata.AnnData
        One per dataset to compare.
    slot_name : str, default ``"netP"``
        Slot in ``uns["cellchat"]`` to read ``prob`` from.
    type_ : {"functional", "structural"}, default "functional"
        Which similarity to compute.
    comparison : sequence of int, optional
        1-based dataset indices (mirrors R semantics). Defaults to all
        datasets.
    k : int, optional
        Number of nearest neighbours used for SNN smoothing.
    thresh : float, optional
        Trim quantile threshold in ``[0, 0.25]``.
    dataset_names : sequence of str, optional
        Used to build pathway labels of the form
        ``"<pathway>--<dataset_name>"``. Defaults to ``["dataset1",
        "dataset2", ...]``.

    Returns
    -------
    dict
        ``{"matrix": pandas.DataFrame, "comparison_name": str}``.
    """
    if comparison is None:
        comparison = list(range(1, len(adata_list) + 1))
    comparison = [int(c) for c in comparison]
    comparison_name = "-".join(str(c) for c in comparison)

    if dataset_names is None:
        dataset_names = [f"dataset{i}" for i in comparison]
    elif len(dataset_names) != len(comparison):
        raise ValueError("len(dataset_names) must match len(comparison).")

    nets: List[np.ndarray] = []
    signaling_all: List[str] = []
    for ds_idx, ds_name in zip(comparison, dataset_names):
        adata = adata_list[ds_idx - 1]
        slot = _get_slot(adata, slot_name)
        if "prob" not in slot:
            raise ValueError(
                f"adata_list[{ds_idx - 1}].uns['cellchat']['{slot_name}']['prob'] is required."
            )
        prob = np.asarray(slot["prob"], dtype=float)
        if prob.ndim != 3:
            raise ValueError("`prob` must be 3-D.")
        nets.append(prob)
        pathway_names = list(slot.get("pathway_names", [str(i) for i in range(prob.shape[2])]))
        if len(pathway_names) != prob.shape[2]:
            pathway_names = [str(i) for i in range(prob.shape[2])]
        for pn in pathway_names:
            signaling_all.append(f"{pn}--{ds_name}")

    net_dim = np.array([n.shape[2] for n in nets])
    n_total = int(net_dim.sum())
    position = np.concatenate([[0], np.cumsum(net_dim)])  # length len(nets)+1

    if k is None:
        k = int(np.ceil(np.sqrt(n_total)))
        if n_total > 25:
            k = k + 1

    if thresh is not None:
        new_nets = []
        for net_arr in nets:
            arr = net_arr.copy()
            nz = arr[arr != 0]
            if nz.size:
                cutoff = float(np.quantile(nz, float(thresh)))
                arr[arr < cutoff] = 0.0
            new_nets.append(arr)
        nets = new_nets

    if type_ == "functional":
        S = np.zeros((n_total, n_total))
        for i in range(n_total):
            # Locate the dataset and inner index for column i in concat ordering
            idx_i = int(np.searchsorted(position[1:], i, side="right"))
            inner_i = i - int(position[idx_i])
            net_i = nets[idx_i]
            Gi = (net_i[:, :, inner_i] > 0).astype(float)
            for j in range(n_total):
                idx_j = int(np.searchsorted(position[1:], j, side="right"))
                inner_j = j - int(position[idx_j])
                net_j = nets[idx_j]
                Gj = (net_j[:, :, inner_j] > 0).astype(float)
                num = float((Gi * Gj).sum())
                denom = float((Gi + Gj - Gi * Gj).sum())
                S[i, j] = (num / denom) if denom else 0.0
        np.fill_diagonal(S, 1.0)
    elif type_ == "structural":
        D = np.zeros((n_total, n_total))
        for i in range(n_total):
            idx_i = int(np.searchsorted(position[1:], i, side="right"))
            inner_i = i - int(position[idx_i])
            net_i = nets[idx_i]
            Gi = (net_i[:, :, inner_i] > 0).astype(float)
            for j in range(n_total):
                idx_j = int(np.searchsorted(position[1:], j, side="right"))
                inner_j = j - int(position[idx_j])
                net_j = nets[idx_j]
                Gj = (net_j[:, :, inner_j] > 0).astype(float)
                D[i, j] = computeNetD_structure(Gi, Gj)
        D[~np.isfinite(D)] = 0.0
        S = 1.0 - D
    else:
        raise ValueError(f"Unknown type {type_!r}.")

    SNN = buildSNN(S, k=int(k), k_scale=10, prune_SNN=1.0 / 15.0).toarray()
    Similarity = S * SNN
    df = pd.DataFrame(Similarity, index=signaling_all, columns=signaling_all)

    # Persist into the FIRST adata's slot; mirror R behaviour of saving
    # under matrix[[comparison_name]]
    first_slot = _get_slot(adata_list[0], slot_name)
    if "similarity" not in first_slot or not isinstance(first_slot["similarity"], dict):
        first_slot["similarity"] = {}
    if type_ not in first_slot["similarity"] or not isinstance(
        first_slot["similarity"][type_], dict
    ):
        first_slot["similarity"][type_] = {}
    if "matrix" not in first_slot["similarity"][type_] or not isinstance(
        first_slot["similarity"][type_]["matrix"], dict
    ):
        first_slot["similarity"][type_]["matrix"] = {}
    first_slot["similarity"][type_]["matrix"][comparison_name] = df
    return {"matrix": df, "comparison_name": comparison_name}


def _resolve_comparison_name(
    comparison: Optional[Sequence[int]],
    matrix_dict: Optional[Mapping[str, Any]],
    mode: Optional[str] = None,
) -> str:
    """Compute the per-mode default comparison key (R: analysis.R:1506-1515).

    R behaviour:

    - ``object@options$mode == "single"`` -> ``comparison <- "single"``
    - ``object@options$mode == "merged"`` and ``comparison`` NULL ->
      ``comparison <- 1:length(unique(object@meta$datasets))`` (joined
      with ``"-"`` to form e.g. ``"1-2"``).

    Python persists matrices under
    ``slot['similarity'][type_]['matrix']`` keyed by their
    ``comparison.name``. Merged-mode storage typically holds a single
    entry like ``"1-2"`` (sometimes more if the user has run multiple
    comparisons). Detection rules:

    - ``mode='single'`` (or auto-detect with ``"single"`` key present
      or no merged-style keys): return ``"single"``.
    - ``mode='merged'`` (or auto-detect with one or more merged-style
      keys present and no ``"single"`` key): if exactly one key exists,
      return that key as-is (matches R's existing comparison name); if
      multiple, fall back to ``"-".join("1".."n")`` where ``n`` is the
      number of keys.
    """
    if comparison is not None:
        return "-".join(str(c) for c in comparison)
    is_dict = isinstance(matrix_dict, dict)
    if mode == "single":
        return "single"
    if mode == "merged":
        if not is_dict or len(matrix_dict) == 0:
            raise ValueError(
                "mode='merged' requires at least one matrix stored under "
                "slot['similarity'][type_]['matrix']."
            )
        if len(matrix_dict) == 1:
            return next(iter(matrix_dict.keys()))
        n_datasets = len(matrix_dict)
        return "-".join(str(c) for c in range(1, n_datasets + 1))
    # Auto-detect: merged when 'single' is absent and at least one
    # merged-style key (containing '-') is present.
    if is_dict and "single" not in matrix_dict:
        merged_keys = [k for k in matrix_dict.keys() if "-" in str(k)]
        if len(merged_keys) == 1:
            return str(merged_keys[0])
        if len(merged_keys) > 1:
            n_datasets = len(matrix_dict)
            return "-".join(str(c) for c in range(1, n_datasets + 1))
    return "single"


# ---------------------------------------------------------------------------
# netEmbedding (R: analysis.R:1503)
# ---------------------------------------------------------------------------
def netEmbedding(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    comparison: Optional[Sequence[int]] = None,
    pathway_remove: Optional[Sequence[str]] = None,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.3,
    n_components: int = 2,
    seed_use: int = 42,
    umap_method: str = "umap-learn",
    mode: Optional[str] = None,
    **umap_kwargs: Any,
) -> ad.AnnData:
    """Manifold learning of signaling networks (UMAP).

    Mirrors ``SpatialCellChat::netEmbedding`` (``R/analysis.R:1503``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose
        ``uns["cellchat"][slot_name]["similarity"][type_]["matrix"][comparison_name]``
        is consumed.
    slot_name : str, default ``"netP"``
        Slot to read from.
    type_ : {"functional", "structural"}, default "functional"
        Type of similarity matrix.
    comparison : sequence of int, optional
        Defaults to ``"single"`` for single-dataset mode and to
        ``range(1, n_datasets+1)`` for merged mode (when multiple
        per-dataset matrices are stored under
        ``slot['similarity'][type_]['matrix']``).
    pathway_remove : sequence of str, optional
        Pathway names to drop from the embedding. Defaults to those
        with ``colSums == 1`` (i.e. only the diagonal entry is non-zero).
    n_neighbors : int, optional
        UMAP ``n_neighbors``. Defaults to ``ceil(sqrt(N)) + 1`` per R.
    min_dist : float, default 0.3
        UMAP ``min_dist``.
    n_components : int, default 2
        UMAP output dimensionality.
    seed_use : int, default 42
        UMAP ``random_state``.
    umap_method : {"umap-learn", "uwot"}, default ``"umap-learn"``
        R surfaces ``umap.method``: ``"umap-learn"`` calls Python's
        ``umap.UMAP(metric='correlation')`` (R's ``runUMAP`` default);
        ``"uwot"`` calls ``uwot::umap`` whose default metric is
        ``"euclidean"``. Both pass the similarity matrix as features
        (rows = signaling pathways) — neither uses precomputed distance.
    mode : {"single", "merged"}, optional
        R inspects ``object@options$mode``. When provided, this overrides
        merged-mode auto-detection: ``"single"`` forces ``comparison_name``
        to ``"single"``; ``"merged"`` forces auto-detection of dataset
        count from the stored matrix dict.
    **umap_kwargs : Any
        Additional kwargs forwarded to :class:`umap.UMAP`.

    Returns
    -------
    anndata.AnnData
        ``adata`` augmented with
        ``uns["cellchat"][slot_name]["similarity"][type_]["dr"][comparison_name]``.
    """
    if umap_method not in ("umap-learn", "uwot"):
        raise ValueError(
            f"umap_method must be 'umap-learn' or 'uwot'; got {umap_method!r}"
        )
    slot = _get_slot(adata, slot_name)
    if "similarity" not in slot or type_ not in slot["similarity"]:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['similarity']['{type_}'] missing; "
            "run computeNetSimilarity first."
        )
    matrix_dict = slot["similarity"][type_].get("matrix")
    comparison_name = _resolve_comparison_name(comparison, matrix_dict, mode)

    if not isinstance(matrix_dict, dict) or comparison_name not in matrix_dict:
        raise ValueError(
            f"similarity['{type_}']['matrix']['{comparison_name}'] not found."
        )
    Sim = matrix_dict[comparison_name]
    if isinstance(Sim, pd.DataFrame):
        sim_arr = Sim.values
        names = list(Sim.index)
    else:
        sim_arr = np.asarray(Sim, dtype=float)
        names = [str(i) for i in range(sim_arr.shape[0])]

    if pathway_remove is None:
        col_sums = sim_arr.sum(axis=0)
        pathway_remove = [names[i] for i in range(len(names)) if col_sums[i] == 1.0]
    pathway_remove = list(pathway_remove)
    if len(pathway_remove) > 0:
        keep = [i for i, nm in enumerate(names) if nm not in pathway_remove]
        sim_arr = sim_arr[np.ix_(keep, keep)]
        names = [names[i] for i in keep]

    if n_neighbors is None:
        n_neighbors = int(np.ceil(np.sqrt(sim_arr.shape[0]))) + 1
    n_neighbors = int(min(n_neighbors, max(2, sim_arr.shape[0] - 1)))

    # R passes the similarity matrix directly as data.use to runUMAP /
    # uwot::umap (rows = signaling pathways, cols = features). Neither
    # uses precomputed distance: runUMAP defaults to metric='correlation'
    # and uwot defaults to metric='euclidean'.
    metric = "correlation" if umap_method == "umap-learn" else "euclidean"
    np.random.seed(seed_use)
    reducer = _umap.UMAP(
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        n_components=int(n_components),
        random_state=int(seed_use),
        n_jobs=1,
        **umap_kwargs,
    )
    Y = reducer.fit_transform(sim_arr)
    Y_df = pd.DataFrame(
        Y,
        index=names,
        columns=[f"UMAP{i + 1}" for i in range(Y.shape[1])],
    )
    if "dr" not in slot["similarity"][type_] or not isinstance(
        slot["similarity"][type_]["dr"], dict
    ):
        slot["similarity"][type_]["dr"] = {}
    slot["similarity"][type_]["dr"][comparison_name] = Y_df
    return adata


# ---------------------------------------------------------------------------
# netClustering (R: analysis.R:1564)
# ---------------------------------------------------------------------------
def netClustering(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type_: str = "functional",
    comparison: Optional[Sequence[int]] = None,
    k: Optional[int] = None,
    methods: str = "kmeans",
    k_eigen: Optional[int] = None,
    n_init: int = 10,
    seed_use: int = 42,
    mode: Optional[str] = None,
) -> ad.AnnData:
    """Clustering of signaling networks based on UMAP embedding.

    Mirrors ``SpatialCellChat::netClustering``
    (``R/analysis.R:1564``).

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing the embedding written by :func:`netEmbedding`.
    slot_name : str, default ``"netP"``
        Slot to read / write.
    type_ : {"functional", "structural"}, default "functional"
        Similarity type.
    comparison : sequence of int, optional
        Defaults to ``"single"`` for single-dataset mode and to
        ``range(1, n_datasets+1)`` for merged mode (auto-detected from the
        per-dataset matrices stored under
        ``slot['similarity'][type_]['matrix']``).
    k : int, optional
        Number of clusters. When missing, the cluster count is inferred
        via :func:`computeEigengap` (mirrors R's consensus-clustering loop
        over ``kRange = 2 .. min(N-1, 10)``).
    methods : {"kmeans", "spectral"}, default "kmeans"
        Clustering algorithm. ``"spectral"`` requires ``k_eigen`` and
        ``k`` to both be set.
    k_eigen : int, optional
        Number of eigenvectors used by spectral clustering.
    n_init : int, default 10
        ``KMeans(n_init=...)`` seed restarts (R uses ``nstart=10``).
    seed_use : int, default 42
        ``KMeans(random_state=...)``.
    mode : {"single", "merged"}, optional
        Mirrors R's ``object@options$mode``. When provided, overrides
        merged-mode auto-detection. See :func:`netEmbedding` for details.

    Returns
    -------
    anndata.AnnData
        ``adata`` augmented with
        ``uns["cellchat"][slot_name]["similarity"][type_]["group"][comparison_name]``.
    """
    slot = _get_slot(adata, slot_name)
    matrix_dict = slot.get("similarity", {}).get(type_, {}).get("matrix")
    comparison_name = _resolve_comparison_name(comparison, matrix_dict, mode)
    dr_dict = slot.get("similarity", {}).get(type_, {}).get("dr")
    if not isinstance(dr_dict, dict) or comparison_name not in dr_dict:
        raise ValueError(
            f"similarity['{type_}']['dr']['{comparison_name}'] not found; "
            "run netEmbedding first."
        )
    Y = dr_dict[comparison_name]
    if isinstance(Y, pd.DataFrame):
        data_use = Y.values
        names = list(Y.index)
    else:
        data_use = np.asarray(Y, dtype=float)
        names = [str(i) for i in range(data_use.shape[0])]

    N = data_use.shape[0]

    if methods == "kmeans":
        if k is None:
            kRange = list(range(2, min(N - 1, 10) + 1))
            CM = np.zeros((N, N))
            for kx in kRange:
                km = KMeans(
                    n_clusters=kx,
                    n_init=int(n_init),
                    random_state=int(seed_use),
                ).fit(data_use)
                idents = km.labels_
                adjMat0 = (idents[:, None] == idents[None, :]).astype(float)
                CM += adjMat0
            CM /= float(len(kRange))
            res = computeEigengap(CM, tol=0.01)
            num_cluster = int(res["upper_bound"])
            num_cluster = max(num_cluster, 1)
            km = KMeans(
                n_clusters=num_cluster,
                n_init=int(n_init),
                random_state=int(seed_use),
            ).fit(data_use)
            clusters = km.labels_
        else:
            km = KMeans(
                n_clusters=int(k),
                n_init=int(n_init),
                random_state=int(seed_use),
            ).fit(data_use)
            clusters = km.labels_
    elif methods == "spectral":
        if k is None or k_eigen is None:
            raise ValueError("methods='spectral' requires both `k` and `k_eigen`.")
        A = data_use
        D = A.sum(axis=1)
        D_inv_sqrt = np.diag(np.where(D > 0, 1.0 / np.sqrt(D), 0.0))
        L = np.diag(D) - A
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
        ev_vals, ev_vecs = np.linalg.eigh(L_norm)
        # ev_vals is increasing; pick the k_eigen LARGEST eigenvectors as in R
        # R: Z <- evL$vectors[,(ncol-k.eigen+1):ncol] when symmetric=TRUE returns
        # eigenvalues in DECREASING order; here we pick the top-k_eigen
        # corresponding to the largest eigenvalues (last columns of ev_vecs).
        Z = ev_vecs[:, -int(k_eigen):]
        km = KMeans(
            n_clusters=int(k), n_init=20, random_state=int(seed_use)
        ).fit(Z)
        clusters = km.labels_
    else:
        raise ValueError(f"Unknown clustering method {methods!r}.")

    clusters_series = pd.Series(clusters + 1, index=names)  # R is 1-based
    if "group" not in slot["similarity"][type_] or not isinstance(
        slot["similarity"][type_]["group"], dict
    ):
        slot["similarity"][type_]["group"] = {}
    slot["similarity"][type_]["group"][comparison_name] = clusters_series
    return adata


# ---------------------------------------------------------------------------
# netAnalysis_computeCentrality (R: analysis.R:82)
# ---------------------------------------------------------------------------
def netAnalysis_computeCentrality(
    adata: Optional[ad.AnnData] = None,
    net: Union[None, np.ndarray, Mapping[str, np.ndarray]] = None,
    slot_name: str = "net",
    signaling_name: Optional[Sequence[str]] = None,
    do_group: bool = False,
    thresh: float = 0.05,
    degree_only: bool = True,
) -> Union[np.ndarray, ad.AnnData]:
    """Compute centrality scores across all inferred signaling networks.

    Mirrors ``SpatialCellChat::netAnalysis_computeCentrality``
    (``R/analysis.R:82``).

    Parameters
    ----------
    adata : anndata.AnnData, optional
        Object whose ``uns["cellchat"][slot_name]`` is consumed.
        When ``None`` the raw ``net`` argument must be supplied.
    net : numpy.ndarray | Mapping[str, numpy.ndarray] | None, optional
        A 2-D array (single signaling network), a 3-D array
        ``(K, K, N)`` (with ``names`` taken from a leading dimension if
        provided as a ``dict``), or a mapping of name -> 2-D array.
    slot_name : str, default ``"net"``
        Slot to read from when ``net`` is None.
    signaling_name : sequence of str, optional
        Subset of signaling names to keep. When provided, all entries
        must be present in ``net`` (or the underlying mapping).
    do_group : bool, default False
        When True, read the K x K x N group-level ``prob``/``pval``
        arrays from ``adata.uns["cellchat"][slot_name]`` and write the
        result back to ``[slot_name]["centr"]``. When False, read the
        cell-level ``tmp["prob.cell"]`` mapping and write back to
        ``[slot_name]["centr.cell"]``.
    thresh : float, default 0.05
        p-value threshold; entries with ``pval >= thresh`` (and the
        corresponding ``prob == 0`` entries) are zeroed before centrality
        computation.
    degree_only : bool, default True
        Forward to :func:`computeCentralityLocal`.

    Returns
    -------
    numpy.ndarray | anndata.AnnData
        When ``adata`` is None, returns the dense centrality array of
        shape ``(len(centr.name), N_nodes, n_signaling)``. Otherwise,
        the input ``adata`` is augmented and returned.
    """
    node_names: Optional[List[str]] = None

    if net is None:
        if adata is None:
            raise ValueError("Either `adata` or `net` must be provided.")
        slot = _get_slot(adata, slot_name)
        if do_group:
            if "prob" not in slot or slot["prob"] is None:
                raise ValueError(
                    "do_group=True requires a `prob` array in "
                    f"adata.uns['cellchat']['{slot_name}']."
                )
            prob = np.asarray(slot["prob"], dtype=float).copy()
            # Mirror R's NULL-tolerant behaviour: when pval is absent
            # (e.g. on the netP slot, or after computeAvgCommunProb with
            # do.permutation = FALSE), skip the significance gate.
            pval_raw = slot.get("pval", None)
            if pval_raw is not None:
                pval = np.asarray(pval_raw, dtype=float).copy()
                pval[prob == 0] = 1.0
                prob[pval >= thresh] = 0.0
            net_dict = {}
            # Pull dim-3 names: prefer the slot's own canonical key
            # (``pathways`` for netP, ``interaction_name`` for net) before
            # falling back to integer placeholders. ``pathway_names`` is the
            # legacy key still used by some tests.
            names_candidate = (
                slot.get("pathways", None)
                or slot.get("pathway_names", None)
                or slot.get("dimnames", {}).get("interaction_name", None)
                or [str(i) for i in range(prob.shape[2])]
            )
            names = list(names_candidate)
            if len(names) != prob.shape[2]:
                names = [str(i) for i in range(prob.shape[2])]
            for i, nm in enumerate(names):
                net_dict[nm] = prob[:, :, i]
            net = net_dict
            # R: ``node.names <- dimnames(prob)[[1]]`` — the first dim of
            # ``prob`` carries cell-group levels, which in R come from
            # ``levels(cellchat@idents)``. In this Python port the prob
            # matrix has no embedded dimnames, so resolve in this order:
            #   1. ``levels(cellchat@idents)`` via adata.obs categoricals
            #   2. slot's mirrored ``dimnames["source"]`` (set by
            #      :func:`computeAvgCommunProb` / :func:`computeCommunProbPathway`)
            #   3. integer-index placeholders (last resort)
            node_names = _idents_levels(adata)
            if node_names is None or len(node_names) != prob.shape[0]:
                node_names = list(
                    slot.get("dimnames", {}).get(
                        "source",
                        slot.get("dimnames_K", []),
                    )
                )
            if len(node_names) != prob.shape[0]:
                node_names = [str(i) for i in range(prob.shape[0])]
        else:
            tmp = slot.get("tmp", {})
            if "prob.cell" not in tmp:
                if slot_name == "net":
                    raise ValueError(
                        "Please run `computeCommunProb` to compute the "
                        "communication probability/strength between any "
                        "interacting individual cells!"
                    )
                elif slot_name == "netP":
                    raise ValueError(
                        "Please run `computeCommunProbPathway` to compute "
                        "the communication probability/strength between "
                        "any interacting individual cells!"
                    )
                else:
                    raise ValueError(
                        f"adata.uns['cellchat']['{slot_name}']['tmp']['prob.cell'] missing."
                    )
            net = tmp["prob.cell"]
            # R: ``node.names <- dimnames.sparse3Darray(prob.cell)[[1]]`` —
            # the row-dim is single-cell barcodes, not cell-group levels.
            # The canonical key written by ``computeCommunProb`` is
            # ``slot["dimnames"]["cells"]`` (see modeling.py). Fall back to
            # adata.obs_names when the slot lacks dimnames, and only to
            # integer placeholders as a last resort.
            sample_mat = next(iter(net.values()))
            n_nodes = (
                sample_mat.shape[0]
                if hasattr(sample_mat, "shape")
                else np.asarray(sample_mat).shape[0]
            )
            node_names = list(
                slot.get("dimnames", {}).get(
                    "cells",
                    list(adata.obs_names.astype(str)) if adata is not None else [],
                )
            )
            if len(node_names) != n_nodes:
                node_names = [str(i) for i in range(n_nodes)]
    else:
        if isinstance(net, np.ndarray):
            if net.ndim == 2:
                net = {"net_temp": net}
                node_names = [str(i) for i in range(next(iter(net.values())).shape[0])]
            elif net.ndim == 3:
                tmp_names = [str(i) for i in range(net.shape[2])]
                tmp_node_names = [str(i) for i in range(net.shape[0])]
                net = {nm: net[:, :, i] for i, nm in enumerate(tmp_names)}
                node_names = tmp_node_names
            else:
                raise ValueError("`net` ndarray must be 2-D or 3-D.")
        elif isinstance(net, Mapping):
            sample_mat = np.asarray(next(iter(net.values())))
            node_names = [str(i) for i in range(sample_mat.shape[0])]
        else:
            raise TypeError("`net` must be ndarray or mapping.")

    if signaling_name is not None:
        signaling_name = list(signaling_name)
        missing = [s for s in signaling_name if s not in net]
        if missing:
            raise ValueError(
                "Please check the input `signaling.name` because some are "
                f"not the significant signaling: {missing!r}"
            )
        net = {s: net[s] for s in signaling_name}

    sig_names = list(net.keys())
    n_signal = len(sig_names)
    if n_signal == 0:
        raise ValueError("No signaling networks to process.")

    # net values may be scipy sparse matrices; ``np.asarray`` on those
    # returns a 0-d object array (not what we want). Use the matrix's own
    # ``shape`` attribute instead.
    sample_mat = next(iter(net.values()))
    n_nodes = sample_mat.shape[0]

    centr_names = (
        CENTRALITY_NAMES_DEGREE if degree_only else CENTRALITY_NAMES_FULL
    )
    centr_all = np.zeros((len(centr_names), n_nodes, n_signal))
    for i, nm in enumerate(sig_names):
        net_i = net[nm]
        if hasattr(net_i, "toarray"):
            net_i = net_i.toarray()
        net_i = np.asarray(net_i, dtype=float)
        centr_x = computeCentralityLocal(net_i, degree_only=degree_only)
        centr_all[:, :, i] = centr_x

    if adata is None:
        return centr_all

    slot = _get_slot(adata, slot_name)
    target_key = "centr" if do_group else "centr.cell"
    slot[target_key] = {
        "values": centr_all,
        "centr_names": list(centr_names),
        "node_names": list(node_names) if node_names is not None else [
            str(i) for i in range(n_nodes)
        ],
        "signaling_names": list(sig_names),
    }
    return adata


# ---------------------------------------------------------------------------
# netMappingDEG (R: analysis.R:3222)
# ---------------------------------------------------------------------------
def netMappingDEG(
    adata: ad.AnnData,
    features_name: str,
    thresh: float = 0.05,
) -> pd.DataFrame:
    """Map differentially-expressed-gene info onto inferred communications.

    Mirrors ``SpatialCellChat::netMappingDEG``
    (``R/analysis.R:3222``).

    Parameters
    ----------
    adata : anndata.AnnData
        Must contain ``uns["cellchat"]["var.features"][f"{features_name}.info"]``
        (written by :func:`identifyOverExpressedGenes`) and
        ``uns["cellchat"]["DB"]`` for ``geneInfo`` + ``complex_input``.
    features_name : str
        Used to look up ``var.features[features_name + ".info"]``.
    thresh : float, default 0.05
        p-value threshold passed through to :func:`subsetCommunication`.

    Returns
    -------
    pandas.DataFrame
        The inferred-communication data frame augmented with eight DEG
        columns: ``ligand.pvalues``, ``ligand.logFC``, ``ligand.pct.1``,
        ``ligand.pct.2``, plus the equivalent ``receptor.*``.
    """
    cc = _ensure_uns(adata)
    full_features_name = f"{features_name}.info"
    var_features = cc.get("var.features", {})
    if full_features_name not in var_features:
        raise ValueError(
            f"The input features.name does not exist in `names(var.features)`."
            " Please first run `identifyOverExpressedGenes`!"
        )
    DEG = pd.DataFrame(var_features[full_features_name]).copy()
    DB = cc.get("DB", {})
    geneInfo = DB.get("geneInfo")
    complex_input = DB.get("complex")
    if geneInfo is None or complex_input is None:
        raise ValueError(
            "adata.uns['cellchat']['DB']['geneInfo'] and "
            "adata.uns['cellchat']['DB']['complex'] are required."
        )

    # Late import to avoid circular dependency
    from .modeling import subsetCommunication

    df_net = subsetCommunication(adata, thresh=thresh)
    if isinstance(df_net, dict):
        # Merged-mode: dict of dataset_name -> DataFrame
        net_pieces = []
        for ds_name, df_i in df_net.items():
            df_i = df_i.copy()
            df_i["datasets"] = ds_name
            net_pieces.append(df_i)
        net = pd.concat(net_pieces, axis=0, ignore_index=True)
    else:
        net = df_net.copy()

    net["source.ligand"] = net["source"].astype(str) + "." + net["ligand"].astype(str)
    net["target.receptor"] = (
        net["target"].astype(str) + "." + net["receptor"].astype(str)
    )

    DEG["clusters.features"] = DEG["clusters"].astype(str) + "." + DEG["features"].astype(str)

    new_cols = [
        "ligand.pvalues",
        "ligand.logFC",
        "ligand.pct.1",
        "ligand.pct.2",
        "receptor.pvalues",
        "receptor.logFC",
        "receptor.pct.1",
        "receptor.pct.2",
    ]
    for col in new_cols:
        net[col] = np.nan

    geneInfo_symbols = set(geneInfo["Symbol"].astype(str).tolist())

    # Ligand
    idx1_ligand = net["ligand"].astype(str).isin(geneInfo_symbols)
    idx2_ligand = np.where(~idx1_ligand)[0]
    deg_clusters_features = DEG["clusters.features"].astype(str).tolist()
    deg_index = pd.Index(deg_clusters_features)
    pos_ligand = deg_index.get_indexer(net["source.ligand"].astype(str))
    matched_ligand_rows = np.where(pos_ligand != -1)[0]
    matched_deg_rows_l = pos_ligand[pos_ligand != -1]
    cols_src = ["pvalues", "logFC", "pct.1", "pct.2"]
    cols_dst = ["ligand.pvalues", "ligand.logFC", "ligand.pct.1", "ligand.pct.2"]
    if matched_ligand_rows.size > 0:
        for s, d in zip(cols_src, cols_dst):
            net.iloc[
                matched_ligand_rows, net.columns.get_loc(d)
            ] = DEG.iloc[matched_deg_rows_l][s].values

    if idx2_ligand.size > 0:
        net_temp_all = []
        for ix in idx2_ligand:
            complex_name = str(net.iloc[ix]["ligand"])
            if complex_name in complex_input.index:
                row = complex_input.loc[complex_name]
                subunits = [
                    str(row[c])
                    for c in complex_input.columns
                    if c.startswith("subunit") and pd.notna(row[c]) and str(row[c]).strip() != ""
                ]
                subunits = list(dict.fromkeys(subunits))
            else:
                subunits = []
            source_complex = [f"{net.iloc[ix]['source']}.{su}" for su in subunits]
            pos = deg_index.get_indexer(source_complex)
            matched = pos[pos != -1]
            if matched.size > 0:
                sub = DEG.iloc[matched][cols_src].mean(axis=0, skipna=True)
                net_temp_all.append(sub.values)
            else:
                net_temp_all.append([np.nan] * 4)
        if net_temp_all:
            net_temp_all = np.asarray(net_temp_all)
            for col_i, d in enumerate(cols_dst):
                net.iloc[idx2_ligand, net.columns.get_loc(d)] = net_temp_all[:, col_i]

    # Receptor
    idx1_receptor = net["receptor"].astype(str).isin(geneInfo_symbols)
    idx2_receptor = np.where(~idx1_receptor)[0]
    pos_receptor = deg_index.get_indexer(net["target.receptor"].astype(str))
    matched_receptor_rows = np.where(pos_receptor != -1)[0]
    matched_deg_rows_r = pos_receptor[pos_receptor != -1]
    cols_dst_r = [
        "receptor.pvalues",
        "receptor.logFC",
        "receptor.pct.1",
        "receptor.pct.2",
    ]
    if matched_receptor_rows.size > 0:
        for s, d in zip(cols_src, cols_dst_r):
            net.iloc[
                matched_receptor_rows, net.columns.get_loc(d)
            ] = DEG.iloc[matched_deg_rows_r][s].values

    if idx2_receptor.size > 0:
        net_temp_all = []
        for ix in idx2_receptor:
            complex_name = str(net.iloc[ix]["receptor"])
            if complex_name in complex_input.index:
                row = complex_input.loc[complex_name]
                subunits = [
                    str(row[c])
                    for c in complex_input.columns
                    if c.startswith("subunit") and pd.notna(row[c]) and str(row[c]).strip() != ""
                ]
                subunits = list(dict.fromkeys(subunits))
            else:
                subunits = []
            target_complex = [f"{net.iloc[ix]['target']}.{su}" for su in subunits]
            pos = deg_index.get_indexer(target_complex)
            matched = pos[pos != -1]
            if matched.size > 0:
                sub = DEG.iloc[matched][cols_src].mean(axis=0, skipna=True)
                net_temp_all.append(sub.values)
            else:
                net_temp_all.append([np.nan] * 4)
        if net_temp_all:
            net_temp_all = np.asarray(net_temp_all)
            for col_i, d in enumerate(cols_dst_r):
                net.iloc[idx2_receptor, net.columns.get_loc(d)] = net_temp_all[:, col_i]

    return net


# ---------------------------------------------------------------------------
# computeCommunField + get_dens (R: analysis.R:3952 / 4259)
# ---------------------------------------------------------------------------
def computeCommunField(
    adata: ad.AnnData,
    *,
    slot_name: str = "netP",
    signaling_name: Optional[Union[str, Sequence[str], int, Sequence[int]]] = None,
    top: float = 0.8,
    sparse_top: bool = True,
) -> ad.AnnData:
    """Per-cell outgoing/incoming communication direction field.

    Direct port of ``SpatialCellChat::computeCommunField`` (R: analysis.R:3952).
    For each pathway / LR pair in ``net['tmp']['prob.cell']``, computes a
    weighted-mean direction vector for each sender (outgoing) and each
    receiver (incoming). The top ``top`` fraction (cumulative) of partner
    interactions is kept; per-edge direction is the unit vector to the
    partner spot scaled by the edge probability.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``uns['cellchat'][slot_name]['tmp']['prob.cell']`` (a dict
        of per-name sparse matrices) and ``obsm['spatial']``.
    slot_name : str, default ``"netP"``
        Either ``"net"`` (LR-level) or ``"netP"`` (pathway-level).
    signaling_name : str | int | sequence | None, optional
        Filter to a subset of pathways / LR pairs. Default uses all available.
    top : float, default ``0.8``
        Cumulative-probability threshold for retained partners (R parity).
    sparse_top : bool, default ``True``
        If ``True``, retain partners while ``cumsum < top * total`` (matches
        R's ``sparse=T`` branch). If ``False``, fallback uses the R
        ``cumsum <= top * total`` branch with the small-N ties special case.

    Returns
    -------
    anndata.AnnData
        Same object; writes ``uns['cellchat'][slot_name]['field']`` with keys
        ``"outgoing"`` and ``"incoming"`` — each is a dict of (signaling_name
        -> ``(nC, 2)`` dense ndarray of direction vectors in the same XY
        coordinate frame as ``obsm['spatial']``).
    """
    cc = adata.uns.setdefault("cellchat", {})
    slot = cc.get(slot_name, {})
    tmp = slot.get("tmp", {})
    if "prob.cell" not in tmp:
        raise RuntimeError(
            "Please run `computeCommunProb` (slot='net') / "
            "`computeCommunProbPathway` (slot='netP') first."
        )

    net_dict = tmp["prob.cell"]
    if isinstance(net_dict, dict):
        signaling_use = list(net_dict.keys())
    else:
        signaling_use = [f"item{i + 1}" for i in range(len(net_dict))]
        net_dict = dict(zip(signaling_use, list(net_dict)))

    if slot_name == "net" and "tmp" in slot and "LRsig.use.idx" in slot["tmp"]:
        idx = list(slot["tmp"]["LRsig.use.idx"])
        signaling_use = [signaling_use[i] for i in idx]
        net_dict = {k: net_dict[k] for k in signaling_use}

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    nC = coords.shape[0]

    if signaling_name is None:
        names_used = signaling_use
    elif isinstance(signaling_name, (int, np.integer)):
        names_used = [signaling_use[int(signaling_name) - 1]]
    elif isinstance(signaling_name, str):
        names_used = [signaling_name]
    else:
        names_used = []
        for s in signaling_name:
            if isinstance(s, (int, np.integer)):
                names_used.append(signaling_use[int(s) - 1])
            else:
                names_used.append(str(s))
    nets = [net_dict[name] for name in names_used]

    def _direction_field(net0: sparse.spmatrix, axis: int) -> np.ndarray:
        """axis=0 -> outgoing (sum across cols), axis=1 -> incoming (sum across rows).

        Returns an (nC, 2) dense matrix.
        """
        result = np.zeros((nC, 2), dtype=float)
        sums = np.asarray(net0.sum(axis=1 - axis)).ravel()
        active_idx = np.flatnonzero(sums > 0)
        for i in active_idx:
            row = net0[i, :].toarray().ravel() if axis == 0 else net0[:, i].toarray().ravel()
            order = np.argsort(-row)  # decreasing
            sorted_vals = row[order]
            total = sorted_vals.sum()
            cumsum = np.cumsum(sorted_vals)
            if sparse_top:
                keep_mask = cumsum < top * total
            else:
                if len(cumsum) >= 3 and cumsum[1] == cumsum[0]:
                    keep_mask = np.array([True] + [False] * (len(cumsum) - 1))
                elif len(cumsum) >= 3 and cumsum[2] == cumsum[1]:
                    keep_mask = np.array([True, True] + [False] * (len(cumsum) - 2))
                else:
                    keep_mask = cumsum <= top * total
            if not keep_mask.any():
                continue
            kept = order[keep_mask]
            probs = sorted_vals[keep_mask]
            if axis == 0:
                # outgoing: vectors from i -> kept partners
                vecs = coords[kept] - coords[i][None, :]
            else:
                # incoming: vectors from kept partners -> i
                vecs = coords[i][None, :] - coords[kept]
            norms = np.linalg.norm(vecs, axis=1)
            scale = np.divide(
                probs, norms, out=np.zeros_like(probs), where=norms > 0
            )
            scaled_vecs = vecs * scale[:, None]
            # Where norm == 0, fall back to unweighted vector (matches R)
            zero_norm = norms == 0
            if zero_norm.any():
                scaled_vecs[zero_norm] = vecs[zero_norm]
            result[i] = scaled_vecs.sum(axis=0)
        return result

    outgoing = {
        name: _direction_field(net0, axis=0) for name, net0 in zip(names_used, nets)
    }
    incoming = {
        name: _direction_field(net0, axis=1) for name, net0 in zip(names_used, nets)
    }

    slot.setdefault("field", {})
    slot["field"]["outgoing"] = outgoing
    slot["field"]["incoming"] = incoming
    cc[slot_name] = slot
    return adata


def get_dens(
    data: np.ndarray,
    dens: Mapping[str, Any],
    method: str = "wkde",
) -> np.ndarray:
    """Look up density values at the rows of ``data`` from a precomputed grid.

    Direct port of ``SpatialCellChat::get_dens`` (R: analysis.R:4259).

    Parameters
    ----------
    data : numpy.ndarray
        ``(N, 2)`` query coordinates (only the first two columns are used).
    dens : Mapping[str, Any]
        Density estimate dict. For ``method="wkde"``: keys ``"x"``, ``"y"``,
        ``"z"`` (z is a (len(y), len(x)) array of grid densities). For
        ``method="ks"``: keys ``"eval.points"`` (list of 1-D vectors) and
        ``"estimate"`` (matching grid array).
    method : {"wkde", "ks"}, default ``"wkde"``
        Selector for the bandwidth-selection / grid-format used by ``dens``.

    Returns
    -------
    numpy.ndarray
        Length-``N`` density values evaluated at each row of ``data``.
    """
    data = np.asarray(data, dtype=float)
    if method == "ks":
        xs = np.asarray(dens["eval.points"][0])
        ys = np.asarray(dens["eval.points"][1])
        z = np.asarray(dens["estimate"])
    elif method == "wkde":
        xs = np.asarray(dens["x"])
        ys = np.asarray(dens["y"])
        z = np.asarray(dens["z"])
    else:
        raise ValueError(f"unknown method {method!r}; expected 'wkde' or 'ks'")
    # Mirror R's findInterval + matrix-indexing semantics:
    #
    #   * x < breaks[1]      : findInterval returns 0; matrix indexing with 0
    #                          DROPS the row (length shrinks).  Python keeps
    #                          length intact and yields NaN as the closest
    #                          length-preserving analogue.
    #   * x >= max(breaks)   : findInterval returns length(breaks) which is
    #                          the LAST 1-based row index; R uses the last
    #                          bin (NOT NaN).  We mirror that by clipping
    #                          the upper side rather than masking it.
    ix = np.searchsorted(xs, data[:, 0], side="right") - 1
    iy = np.searchsorted(ys, data[:, 1], side="right") - 1
    below_grid = (
        (data[:, 0] < xs[0])
        | (data[:, 1] < ys[0])
    )
    ix_safe = np.clip(ix, 0, xs.size - 1)
    iy_safe = np.clip(iy, 0, ys.size - 1)
    res = z[ix_safe, iy_safe].astype(float, copy=True)
    res[below_grid] = np.nan
    return res
