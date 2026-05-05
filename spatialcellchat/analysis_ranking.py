"""Ranking and comparison helpers for cross-dataset SpatialCellChat analysis.

Direct port of the ranking / comparison exports in
``SpatialCellChat::analysis.R``:

- :func:`compareInteractions`  (R: analysis.R:1846)
- :func:`rankNet`              (R: analysis.R:415)
- :func:`rankNetPairwise`      (R: analysis.R:1932)
- :func:`rankSimilarity`       (R: analysis.R:1755)
- :func:`getMaxWeight`         (R: analysis.R:2159)
- :func:`computeNetD_structure` (R: analysis.R:2051)
- :func:`mergeInteractions`    (R: analysis.R:2192)

The R↔Py harness lives at ``/tmp/slice5_harness/harness.R``.

Per the contract the user-facing CellChat S4 object is replaced by either a
single :class:`anndata.AnnData` (when the original R function consumed a
single CellChat object) or a list of AnnDatas (when the function consumed
a *merged* CellChat object). The numerical core of
:func:`mergeInteractions` operates on a freshly constructed
``net.merged`` dict that lives alongside the existing
``adata.uns["cellchat"]["net"]`` slot.

The helpers ``entropia``, ``_node_distance``, ``_nnd`` and
``_alpha_centrality`` are kept private to this module — they're not part
of the public API but are required by :func:`computeNetD_structure`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd

from ._anndata import cellchat_dict as _ensure_uns

__all__ = [
    "compareInteractions",
    "rankNet",
    "rankNetPairwise",
    "rankSimilarity",
    "getMaxWeight",
    "computeNetD_structure",
    "mergeInteractions",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _entropia(a: np.ndarray) -> float:
    """Shannon entropy ``-sum(a * log(a))``, taken only over ``a > 0``.

    Direct port of R::

        entropia<-function(a){
          a<-a[which(a>0)]
          return(-sum(a*log(a)))
        }

    Parameters
    ----------
    a : numpy.ndarray
        1-D non-negative numeric vector. Negative entries are dropped
        prior to summation.

    Returns
    -------
    float
        ``-sum(a[a>0] * log(a[a>0]))``. Returns ``0`` when no positive
        entries exist.
    """
    a = np.asarray(a, dtype=float).ravel()
    pos = a[a > 0]
    if pos.size == 0:
        return 0.0
    return float(-np.sum(pos * np.log(pos)))


def _node_distance(g: ig.Graph) -> np.ndarray:
    """Compute the per-vertex node-distance distribution matrix.

    Direct port of R::

        node_distance<-function(g){
          n<-length(V(g))
          if(n==1){retorno=1}
          if(n>1){
            a<-Matrix::Matrix(0,nrow=n,ncol=n,sparse=TRUE)
            m<-igraph::shortest.paths(g,algorithm=c("unweighted"))
            m[which(m=="Inf")]<-n
            quem<-setdiff(intersect(m,m),0)
            for(j in (1:length(quem))){
              l<-which(m==quem[j])/n
              linhas<-floor(l)+1
              posicoesm1<-which(l==floor(l))
              if(length(posicoesm1)>0){
                linhas[posicoesm1]<-linhas[posicoesm1]-1
              }
              a[1:n,quem[j]]<-hist(linhas,plot=FALSE,breaks=(0:n))$counts
            }
            retorno=(a/(n-1))
          }
        }

    Parameters
    ----------
    g : igraph.Graph
        Directed igraph graph.

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` matrix where column ``q`` is the histogram (per
        source vertex) of how many destinations are exactly ``q`` hops
        away, divided by ``n-1``. Column 0 is always 0.
    """
    n = g.vcount()
    if n == 1:
        return np.array([[1.0]])
    # R: m <- shortest.paths(g, algorithm="unweighted") — default mode="all"
    # python-igraph: distances(mode="all")
    m = np.asarray(g.distances(mode="all"), dtype=float)
    # Replace Inf with n (R: m[which(m=="Inf")] <- n)
    m = np.where(np.isinf(m), float(n), m)

    a = np.zeros((n, n), dtype=float)
    # quem <- setdiff(intersect(m, m), 0)
    # intersect(m, m) is unique(m); setdiff drops 0
    unique_vals = np.unique(m)
    quem = unique_vals[unique_vals != 0]
    # R uses column-major flattening for which(m == quem[j])
    # which() returns linear indices (column-major) of TRUE entries
    # In R: which(m==v)/n -> floor + 1 gives column index, and posicoesm1
    # marks rows that are *integer multiples of n* — those are bottom-row
    # positions whose column index is one less.
    # Equivalent: column index col = (idx-1) // n  (0-based), row = (idx-1) % n
    # but in R: l = idx / n, linhas = floor(l)+1, posicoesm1 picks
    # idx where l is integer (i.e., bottom row), and decrements linhas.
    # In the end "linhas" is the column index (1-based) of the (row,col)
    # pair. Column index in 1-based R is sub-array index 1..n. The hist
    # then bins these column indices into (0..n).
    for q in quem:
        # which(m == q) — R returns column-major linear indices (1-based)
        # Use Fortran order to mirror R column-major.
        flat = np.flatnonzero(m.flatten(order="F") == q) + 1  # 1-based
        l = flat / float(n)
        linhas = np.floor(l).astype(int) + 1  # 1-based column index
        # posicoesm1: indices where l == floor(l) (i.e., l is integer)
        posicoesm1 = np.where(l == np.floor(l))[0]
        if posicoesm1.size > 0:
            linhas[posicoesm1] = linhas[posicoesm1] - 1
        # Now hist(linhas, breaks=(0:n))$counts — counts in bins (i, i+1] for i=0..n-1
        # NB: R's hist with right-closed bins; values at bin boundary go to lower bin
        # except first bin which is closed-closed.
        # With 1-based linhas in 1..n and breaks 0..n, each linhas value v counts
        # as 1 in bin v.
        counts = np.bincount(linhas, minlength=n + 1)[1 : n + 1]  # bins 1..n
        # a[1:n, q] = counts (R uses column q which is 1-based)
        a[:, int(q) - 1] = counts.astype(float)

    return a / float(n - 1)


def _nnd(g: ig.Graph) -> np.ndarray:
    """Compute the network node distribution (NND) descriptor.

    Direct port of R::

        nnd<-function(g){
          N<-length(V(g))
          nd<-node_distance(g)
          pdfm<-Matrix::colMeans(nd)
          norm<-log(max(c(2,length(which(pdfm[1:(N-1)]>0))+1)))
          return(c(pdfm,max(c(0,entropia(pdfm)-entropia(as.matrix(nd))/N))/norm))
        }

    Parameters
    ----------
    g : igraph.Graph
        Directed graph with at least 2 vertices.

    Returns
    -------
    numpy.ndarray
        Length ``N+1`` vector: the first ``N`` entries are
        ``colMeans(nd)`` and the last is the normalized entropy
        difference (NND).
    """
    N = g.vcount()
    nd = _node_distance(g)
    pdfm = nd.mean(axis=0)
    # R: pdfm[1:(N-1)] — first N-1 entries (1-based)
    # length(which(pdfm[1:(N-1)] > 0)) + 1
    if N >= 2:
        prefix = pdfm[: N - 1]
    else:
        prefix = pdfm
    inner_count = int(np.sum(prefix > 0)) + 1
    norm = np.log(max(2.0, float(inner_count)))
    # entropia(pdfm) — R's entropia drops zeros automatically
    e1 = _entropia(pdfm)
    # entropia(as.matrix(nd)) over flattened nd — R applies entropia to the
    # whole matrix as a vector
    e2 = _entropia(nd.ravel())
    val = max(0.0, e1 - e2 / float(N))
    if norm == 0:
        last = 0.0
    else:
        last = val / norm
    return np.concatenate([pdfm, [last]])


def _alpha_centrality(g: ig.Graph) -> np.ndarray:
    """Sorted, scaled alpha centrality + remainder mass.

    Direct port of R::

        alpha_centrality<-function(g){
          N<-length(igraph::V(g))
          r<-sort(igraph::alpha.centrality(g,exo=igraph::degree(g)/(N-1),alpha=1/N))/((N^2))
          return(c(r,max(c(0,1-sum(r)))))
        }

    igraph::alpha.centrality(g, exo, alpha) computes
    ``x = (I - alpha * t(A))^{-1} * exo`` (Bonacich alpha centrality).

    Parameters
    ----------
    g : igraph.Graph
        Directed graph with ``N >= 2`` vertices.

    Returns
    -------
    numpy.ndarray
        Length ``N+1`` vector: sorted alpha centralities scaled by
        ``1/N^2`` plus a tail entry ``max(0, 1 - sum(r))``.
    """
    N = g.vcount()
    if N == 0:
        return np.array([1.0])
    # R: degree(g) — default mode="all" (in + out)
    deg = np.asarray(g.degree(mode="all"), dtype=float)
    exo = deg / float(N - 1) if N > 1 else deg
    alpha = 1.0 / float(N) if N > 0 else 0.0
    # Adjacency (rows = source, cols = target — same as R's get.adjacency)
    A = np.asarray(list(g.get_adjacency()), dtype=float)
    # alpha.centrality: x = (I - alpha * A^T)^-1 * exo
    M = np.eye(N) - alpha * A.T
    x = np.linalg.solve(M, exo)
    r = np.sort(x) / (float(N) ** 2)
    return np.concatenate([r, [max(0.0, 1.0 - float(np.sum(r)))]])


# ---------------------------------------------------------------------------
# computeNetD_structure (R: analysis.R:2051)
# ---------------------------------------------------------------------------
def computeNetD_structure(
    g: np.ndarray,
    h: np.ndarray,
    w1: float = 0.45,
    w2: float = 0.45,
    w3: float = 0.1,
) -> float:
    """Structural distance between two signaling networks.

    Direct port of R's :func:`computeNetD_structure`
    (``analysis.R:2051``). The function combines three components:

    - ``first``: square root of the (clipped, log2-normalised)
      Jensen–Shannon-like difference between the two NND vectors.
    - ``second``: absolute difference of the square-root NND values
      (last entry of :func:`_nnd`).
    - ``third``: the alpha-centrality JS-distance averaged over the
      original graphs and their complementers.

    Parameters
    ----------
    g, h : numpy.ndarray
        Square adjacency matrices (need not have the same shape).
    w1, w2, w3 : float, default 0.45, 0.45, 0.1
        Weights of the three components.

    Returns
    -------
    float
        ``w1 * first + w2 * second + w3 * third``.
    """
    g_arr = np.asarray(g, dtype=float)
    h_arr = np.asarray(h, dtype=float)
    g_ig = ig.Graph.Adjacency(g_arr.tolist(), mode="directed")
    h_ig = ig.Graph.Adjacency(h_arr.tolist(), mode="directed")

    N = g_ig.vcount()
    M_v = h_ig.vcount()

    # PM: 1-D vector of length max(M, N), zeros initially.
    pm_len = max(M_v, N)
    PM = np.zeros(pm_len, dtype=float)

    first = 0.0
    second = 0.0
    third = 0.0

    if w1 + w2 > 0:
        pg = _nnd(g_ig)
        # R: PM[1:(N-1)] = pg[1:(N-1)]; PM[length(PM)] <- pg[N]
        if N >= 2:
            PM[: N - 1] = pg[: N - 1]
        PM[-1] = pg[N - 1] if N >= 1 else 0.0
        # Actually re-read R: PM[length(PM)] <- pg[N]
        # pg has length N+1; pg[N] is the N-th element (1-based) i.e. pg[N-1] (0-based)
        # which is the last entry of colMeans (NOT the entropy).
        # R index: pg[N] is element at position N in 1-based, which is pg[N-1] 0-based.
        # OK, that's correct.

        ph = _nnd(h_ig)
        if M_v >= 2:
            PM[: M_v - 1] = PM[: M_v - 1] + ph[: M_v - 1]
        PM[-1] = PM[-1] + (ph[M_v - 1] if M_v >= 1 else 0.0)

        PM = PM / 2.0

        # first <- sqrt(max((entropia(PM) - (entropia(pg[1:N]) + entropia(ph[1:M]))/2)/log(2), 0))
        e_pm = _entropia(PM)
        # entropia(pg[1:N]) — first N entries of pg (the "colMeans" part)
        e_pg = _entropia(pg[:N])
        e_ph = _entropia(ph[:M_v])
        diff = (e_pm - (e_pg + e_ph) / 2.0) / np.log(2.0)
        first = float(np.sqrt(max(diff, 0.0)))

        # second <- abs(sqrt(pg[N+1]) - sqrt(ph[M+1]))
        # 1-based: pg[N+1] = pg[N] in 0-based (the last entry, which is the entropy)
        v_g = pg[N] if pg.size > N else 0.0
        v_h = ph[M_v] if ph.size > M_v else 0.0
        second = float(abs(np.sqrt(max(v_g, 0.0)) - np.sqrt(max(v_h, 0.0))))

    if w3 > 0:
        pg = _alpha_centrality(g_ig)
        ph = _alpha_centrality(h_ig)
        m = max(len(pg), len(ph))
        Pg = np.zeros(m, dtype=float)
        Ph = np.zeros(m, dtype=float)
        # R: Pg[(m - length(pg) + 1):m] <- pg
        # 1-based -> 0-based: indices (m - len(pg)) .. m-1
        Pg[m - len(pg) :] = pg
        Ph[m - len(ph) :] = ph

        e_PgPh = _entropia((Pg + Ph) / 2.0)
        e_pg = _entropia(pg)
        e_ph = _entropia(ph)
        diff = (e_PgPh - (e_pg + e_ph) / 2.0) / np.log(2.0)
        third += float(np.sqrt(max(diff, 0.0))) / 2.0

        # Complementer step
        g_ig2 = g_ig.complementer(loops=False)
        h_ig2 = h_ig.complementer(loops=False)

        pg = _alpha_centrality(g_ig2)
        ph = _alpha_centrality(h_ig2)
        m = max(len(pg), len(ph))
        Pg = np.zeros(m, dtype=float)
        Ph = np.zeros(m, dtype=float)
        Pg[m - len(pg) :] = pg
        Ph[m - len(ph) :] = ph

        e_PgPh = _entropia((Pg + Ph) / 2.0)
        e_pg = _entropia(pg)
        e_ph = _entropia(ph)
        diff = (e_PgPh - (e_pg + e_ph) / 2.0) / np.log(2.0)
        third += float(np.sqrt(max(diff, 0.0))) / 2.0

    return float(w1 * first + w2 * second + w3 * third)


# ---------------------------------------------------------------------------
# mergeInteractions (R: analysis.R:2192)
# ---------------------------------------------------------------------------
def mergeInteractions(
    adata: ad.AnnData,
    group_merged: Sequence,
) -> ad.AnnData:
    """Merge per-cell-group count/weight matrices into per-cell-type matrices.

    Direct port of R's :func:`mergeInteractions` (``analysis.R:2192``).

    The function reads ``adata.uns["cellchat"]["net"]["count"]`` and
    ``["weight"]`` (both are ``(K, K)`` matrices with one entry per
    cell-group pair) and aggregates them into ``(L, L)`` matrices where
    ``L`` is the number of unique values in ``group_merged``.

    Three new keys are written into ``adata.uns["cellchat"]["net"]``:

    - ``count.merged`` — ``(L, L)`` aggregated count matrix.
    - ``weight.merged`` — ``(L, L)`` aggregated weight matrix.
    - ``group.merged`` — the resolved ``pandas.Categorical`` group factor
      (length ``K``).

    Both merged matrices are :class:`pandas.DataFrame` instances with
    rows / columns indexed by the merged group levels.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"]["net"]["count"]`` and ``["weight"]``
        already populated by :func:`spatialcellchat.aggregateNet`.
    group_merged : sequence of length K
        Mapping from each of the K cell groups to a merged cell-type
        label. Will be coerced to a :class:`pandas.Categorical` (the R
        ``factor()`` analogue).

    Returns
    -------
    anndata.AnnData
        The same ``adata`` object, modified in-place.
    """
    cc = _ensure_uns(adata)
    if "net" not in cc:
        raise ValueError("mergeInteractions requires uns['cellchat']['net']")
    net = cc["net"]
    if "count" not in net or "weight" not in net:
        raise ValueError(
            "mergeInteractions requires net['count'] and net['weight']; "
            "run aggregateNet first"
        )

    count = np.asarray(net["count"], dtype=float)
    weight = np.asarray(net["weight"], dtype=float)
    if count.shape != weight.shape:
        raise ValueError(
            "net['count'] and net['weight'] must have the same shape"
        )
    K = count.shape[0]
    if count.shape[1] != K:
        raise ValueError("net['count'] must be square (K, K)")

    if not isinstance(group_merged, pd.Categorical):
        gm = pd.Categorical(list(group_merged))
    else:
        gm = group_merged
    if len(gm) != K:
        raise ValueError(
            f"group_merged has length {len(gm)}, expected {K} (matching net['count'])"
        )
    levels = list(gm.categories)
    nlev = len(levels)

    count_merged = np.zeros((nlev, nlev), dtype=float)
    weight_merged = np.zeros((nlev, nlev), dtype=float)
    codes = np.asarray(gm.codes)
    for i, lvi in enumerate(levels):
        idx_i = np.flatnonzero(codes == i)
        for j, lvj in enumerate(levels):
            idx_j = np.flatnonzero(codes == j)
            if idx_i.size == 0 or idx_j.size == 0:
                continue
            count_merged[i, j] = float(count[np.ix_(idx_i, idx_j)].sum())
            weight_merged[i, j] = float(weight[np.ix_(idx_i, idx_j)].sum())

    net["count.merged"] = pd.DataFrame(count_merged, index=levels, columns=levels)
    net["weight.merged"] = pd.DataFrame(weight_merged, index=levels, columns=levels)
    net["group.merged"] = gm
    return adata


# ---------------------------------------------------------------------------
# getMaxWeight (R: analysis.R:2159)
# ---------------------------------------------------------------------------
def getMaxWeight(
    adata_list: Sequence[ad.AnnData],
    slot_name: Union[str, Sequence[str]] = ("idents", "net"),
    attribute: Union[str, Sequence[str]] = ("idents", "count"),
) -> np.ndarray:
    """Maximum value of selected measures across one or more datasets.

    Direct port of R's :func:`getMaxWeight` (``analysis.R:2159``). For
    each ``(slot, attr)`` pair the function returns the max across
    datasets of:

    - ``slot="idents"`` — the largest cluster size
      (``max(table(idents))`` per dataset).
    - ``slot="net"`` and ``attr in {"count", "weight", "count.merged",
      "weight.merged"}`` — ``max(net[attr])`` per dataset.
    - ``slot in {"net", "netP"}`` and ``attr`` is a pathway name or
      L-R-pair name — ``max(slot$prob[, , attr])`` per dataset.

    Parameters
    ----------
    adata_list : sequence of anndata.AnnData
        Already-populated CellChat AnnData objects. The "idents" slot is
        resolved from ``adata.uns["cellchat"]["net"]["meta"]["idents"]``
        when present, falling back to ``adata.obs["cellchat_idents"]``
        and finally the legacy ``adata.obs["idents"]``. The "net" slot
        is read from ``adata.uns["cellchat"]["net"]``, and "netP" from
        ``adata.uns["cellchat"]["netP"]``. The "DB" slot is sourced from
        ``adata.uns["cellchat"]["DB"]``.
    slot_name : str or sequence of str
        Which slot to read for each measure. Must align in length with
        ``attribute``.
    attribute : str or sequence of str
        Specific measure to read (see above).

    Returns
    -------
    numpy.ndarray
        Length-``len(attribute)`` numeric array, named (in R) by
        ``attribute``. The Python return is a plain :class:`numpy.ndarray`
        for consistency; callers can wrap with ``pd.Series`` if needed.
    """
    # Coerce both to lists
    if isinstance(slot_name, str):
        slot_name = [slot_name]
    else:
        slot_name = list(slot_name)
    if isinstance(attribute, str):
        attribute = [attribute]
    else:
        attribute = list(attribute)
    if len(slot_name) != len(attribute):
        raise ValueError(
            f"slot_name and attribute must have the same length, "
            f"got {len(slot_name)} and {len(attribute)}"
        )

    weight = np.zeros(len(slot_name), dtype=float)
    for i, (sn, attr) in enumerate(zip(slot_name, attribute)):
        if sn == "idents":
            vals = []
            for x in adata_list:
                idents = _resolve_idents(x)
                # max(table(slot(x, slot.name[i])))
                # table(...) tallies counts per category; we want the largest
                cats, counts = np.unique(np.asarray(idents), return_counts=True)
                vals.append(float(counts.max()) if counts.size > 0 else 0.0)
            weight_all = np.asarray(vals, dtype=float)
        elif sn == "net" and attr in ("count", "weight", "count.merged", "weight.merged"):
            vals = []
            for x in adata_list:
                cc = _ensure_uns(x)
                net = cc.get("net", {})
                if attr not in net:
                    raise ValueError(
                        f"net['{attr}'] missing in one of the AnnData objects"
                    )
                arr = np.asarray(net[attr], dtype=float)
                vals.append(float(arr.max()))
            weight_all = np.asarray(vals, dtype=float)
        else:
            # Pathway / L-R name branch: requires slot in {"net", "netP"} and
            # attr matches one of object_list[0]@DB$interaction$pathway_name
            # or interaction_name.
            cc0 = _ensure_uns(adata_list[0])
            DB = cc0.get("DB", None)
            if DB is None or "interaction" not in DB:
                raise ValueError(
                    f"attribute '{attr}' could not be resolved against "
                    f"DB['interaction']; first AnnData lacks uns['cellchat']['DB']"
                )
            interaction = DB["interaction"]
            valid_names = set(interaction["pathway_name"].astype(str)) | set(
                interaction["interaction_name"].astype(str)
            )
            if attr not in valid_names:
                raise ValueError(
                    f"attribute '{attr}' is not a known pathway_name or "
                    f"interaction_name in DB['interaction']"
                )
            vals = []
            for x in adata_list:
                cc = _ensure_uns(x)
                src = cc.get(sn, None)
                if src is None:
                    raise ValueError(
                        f"slot '{sn}' missing in one of the AnnData objects"
                    )
                # For netP, prob is (K, K, nPathways) with names = pathways.
                # For net, prob is (K, K, nLR) with names = LR pair ids.
                prob = src.get("prob", None)
                if prob is None:
                    raise ValueError(
                        f"slot['{sn}']['prob'] missing in one of the AnnData objects"
                    )
                prob_arr = np.asarray(prob, dtype=float)
                # Resolve the third-dim name to an index
                if sn == "netP":
                    # netP['pathways'] holds the list of names.
                    pathways = src.get("pathways", None)
                    if pathways is None:
                        # fallback to dimnames stored elsewhere
                        names_list = list(map(str, range(prob_arr.shape[2])))
                    else:
                        names_list = list(map(str, pathways))
                else:
                    dn = cc.get("net", {}).get("dimnames", {})
                    names_list = list(map(str, dn.get("interaction_name", dn.get("pairs", []))))
                    if not names_list:
                        names_list = list(map(str, range(prob_arr.shape[2])))
                if attr not in names_list:
                    # Pathway not present in this dataset — R errors out;
                    # we mirror that.
                    raise ValueError(
                        f"attribute '{attr}' not found in dataset's {sn}['prob'] dim names"
                    )
                k = names_list.index(attr)
                vals.append(float(prob_arr[:, :, k].max()))
            weight_all = np.asarray(vals, dtype=float)

        weight[i] = float(weight_all.max())

    return weight


def _resolve_idents(adata: ad.AnnData) -> pd.Categorical:
    """Pick the ident factor. Mirrors how SpatialCellChat stores idents."""
    cc = _ensure_uns(adata)
    net = cc.get("net", {})
    meta = net.get("meta", {})
    if "idents" in meta:
        idents = meta["idents"]
        if isinstance(idents, pd.Categorical):
            return idents
        return pd.Categorical(idents)
    for col in ("cellchat_idents", "idents"):
        if col in adata.obs.columns:
            v = adata.obs[col]
            if isinstance(v.dtype, pd.CategoricalDtype):
                return pd.Categorical(v)
            return pd.Categorical(v)
    raise ValueError(
        "Cannot resolve idents factor; expected uns['cellchat']['net']['meta']['idents']"
        " or obs['cellchat_idents'] / obs['idents']"
    )


# ---------------------------------------------------------------------------
# compareInteractions (R: analysis.R:1846)
# ---------------------------------------------------------------------------
def compareInteractions(
    adata_list: Sequence[ad.AnnData],
    measure: str = "count",
    group: Optional[Sequence] = None,
    group_levels: Optional[Sequence] = None,
    digits: int = 3,
    dataset_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Comparison data frame of total counts/weights across datasets.

    Direct port of the data-construction part of R's
    :func:`compareInteractions` (``analysis.R:1846``). The plot-only
    arguments (``color.use``, ``width``, ``title.name``, etc.) are
    omitted: this function returns the underlying ``ggplot$data`` data
    frame, not the ggplot object.

    Parameters
    ----------
    adata_list : sequence of anndata.AnnData
        Each AnnData must have ``uns["cellchat"]["net"]["count"]``
        (when ``measure="count"``) or ``["weight"]`` (when
        ``measure="weight"``).
    measure : {"count", "weight"}, default ``"count"``
        Aggregated quantity to compare.
    group : sequence, optional
        Per-dataset group label. Default ``1`` for every dataset (one
        group, single bar color in the original R plot).
    group_levels : sequence, optional
        Optional ordering of the categorical group factor.
    digits : int, default ``3``
        Round the per-dataset weight totals to this many decimals
        (R behaviour for ``measure="weight"``).
    dataset_names : sequence of str, optional
        Override the dataset labels used for the ``dataset`` column.
        When ``None`` we generate ``"d1", "d2", ...`` (matching the
        behaviour of an unnamed CellChat ``@net`` list).

    Returns
    -------
    pandas.DataFrame
        Tidy ``(n_datasets, 3)`` data frame with columns:

        - ``"count"`` (always called this; the value is "weight" when
          ``measure="weight"``, mirroring R's ``colnames(df) <- "count"``)
        - ``"dataset"`` (categorical, ordered as ``adata_list``)
        - ``"group"`` (categorical, default ``"1"``)
    """
    if measure not in ("count", "weight"):
        raise ValueError(f"measure must be 'count' or 'weight', got {measure!r}")

    if dataset_names is None:
        names = [f"d{i + 1}" for i in range(len(adata_list))]
    else:
        names = list(dataset_names)
        if len(names) != len(adata_list):
            raise ValueError(
                f"dataset_names length {len(names)} != adata_list length "
                f"{len(adata_list)}"
            )

    vals: List[float] = []
    for x in adata_list:
        cc = _ensure_uns(x)
        net = cc.get("net", {})
        if measure not in net:
            raise ValueError(
                f"net['{measure}'] missing; run aggregateNet first"
            )
        arr = np.asarray(net[measure], dtype=float)
        vals.append(float(arr.sum()))

    if measure == "weight":
        vals = [float(np.round(v, digits)) for v in vals]

    if group is None:
        group_resolved: List = [1] * len(adata_list)
    else:
        group_resolved = list(group)
        if len(group_resolved) != len(adata_list):
            raise ValueError(
                f"group length {len(group_resolved)} != adata_list length "
                f"{len(adata_list)}"
            )

    df = pd.DataFrame(
        {
            "count": vals,  # R: colnames(df) <- "count" regardless of measure
            "dataset": pd.Categorical(names, categories=names),
            "group": (
                pd.Categorical(group_resolved, categories=list(group_levels))
                if group_levels is not None
                else pd.Categorical(group_resolved)
            ),
        },
        index=names,
    )
    return df


# ---------------------------------------------------------------------------
# rankNet (R: analysis.R:415)
# ---------------------------------------------------------------------------
def rankNet(
    adata: Union[ad.AnnData, Sequence[ad.AnnData]],
    slot_name: str = "netP",
    measure: str = "weight",
    mode: str = "comparison",
    comparison: Optional[Sequence[int]] = None,
    do_group: bool = True,
    sources_use: Optional[Sequence] = None,
    targets_use: Optional[Sequence] = None,
    signaling: Optional[Sequence[str]] = None,
    pairLR: Optional[Sequence[str]] = None,
    signaling_type: Optional[Sequence[str]] = None,
    do_stat: bool = False,
    cutoff_pvalue: float = 0.05,
    tol: float = 0.05,
    thresh: float = 0.05,
    show_raw: bool = False,
    stacked: bool = False,
    dataset_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rank pathways or L-R pairs by total information flow / count.

    Direct port of R's :func:`rankNet` (``analysis.R:415``). The plot-
    constructing portion is omitted; the function returns the data frame
    that R would have used for plotting (or, in single mode, that the R
    function would have used after sorting and zero-removal).

    Parameters
    ----------
    adata : anndata.AnnData or list of anndata.AnnData
        In ``mode="single"`` a single AnnData with the requested slot
        populated. In ``mode="comparison"`` a *list* of AnnData (each
        playing the role of one element of R's named ``netP`` list).
    slot_name : {"netP", "net"}, default ``"netP"``
        Which slot to rank. ``"netP"`` ranks pathways
        (``netP$prob``/``pval``); ``"net"`` ranks L-R pairs
        (``net$prob``/``pval``).
    measure : {"weight", "count"}, default ``"weight"``
        - ``"weight"``: information-flow / interaction-strength ranking
          (sum of probability across cell-pair entries; rescaled with
          ``-1/log(x)`` and tied entries de-tied with a synthetic ramp
          when ``do_group=True``).
        - ``"count"``: number of significant interactions per pathway.
    mode : {"comparison", "single"}, default ``"comparison"``
        See ``adata`` description.
    comparison : sequence of int, optional
        1-based indices into the AnnData list. Defaults to
        ``[1, 2, ..., len(adata_list)]`` for ``"comparison"`` mode and
        ``[1]`` for ``"single"`` mode.
    do_group : bool, default ``True``
        Use ``prob`` (group-level) when ``True``; ``prob.cell`` when
        ``False``.
    sources_use, targets_use : sequence of int or str, optional
        Restrict the rank to interactions originating from / arriving at
        these cell groups. Strings must match the ident-factor levels.
    signaling : sequence of str, optional
        Restrict to these signaling pathway names (only valid for
        ``slot_name="netP"``).
    pairLR : sequence of str, optional
        Restrict to these L-R pair names (only valid for
        ``slot_name="net"``).
    signaling_type : sequence of str, optional
        Restrict via the interaction-table ``annotation`` column; mirrors R's
        ``signaling.type`` argument (e.g. ``"Secreted Signaling"``).
    do_stat : bool, default ``False``
        Add a ``pvalues`` column (Wilcoxon paired test) when the number
        of compared datasets is exactly 2 and rows are not all NA.
    cutoff_pvalue, tol : float, defaults ``0.05`` and ``0.05``
        Used in the (plot-only) text-color logic — kept for parity but
        unused in the returned data frame.
    thresh : float, default ``0.05``
        ``prob[pval > thresh] <- 0`` masking threshold.
    show_raw : bool, default ``False``
        Unused; kept for API parity.
    stacked : bool, default ``False``
        Unused; kept for API parity (R uses it for plot-stacking only).
    dataset_names : sequence of str, optional
        Override dataset labels (passed to the ``group`` column in
        comparison mode).

    Returns
    -------
    pandas.DataFrame
        Long-form data frame with one row per (pathway / L-R pair,
        dataset). Columns: ``name``, ``contribution``,
        ``contribution.scaled``, ``group``, plus
        ``contribution.relative.<i>`` (for ``i = 1..len(comparison)-1``)
        in comparison mode.
    """
    if measure not in ("weight", "count"):
        raise ValueError(f"measure must be 'weight' or 'count', got {measure!r}")
    if mode not in ("single", "comparison"):
        raise ValueError(f"mode must be 'single' or 'comparison', got {mode!r}")
    if slot_name not in ("netP", "net"):
        raise ValueError(f"slot_name must be 'netP' or 'net', got {slot_name!r}")

    if mode == "single":
        if isinstance(adata, (list, tuple)):
            if len(adata) != 1 and (comparison is None or len(comparison) != 1):
                raise ValueError(
                    "rankNet(mode='single') expects a single AnnData "
                    "or a list of length 1"
                )
            adata_one = adata[0] if len(adata) > 0 else None
        else:
            adata_one = adata
        return _rank_net_single(
            adata_one,
            slot_name=slot_name,
            measure=measure,
            do_group=do_group,
            sources_use=sources_use,
            targets_use=targets_use,
            signaling=signaling,
            pairLR=pairLR,
            signaling_type=signaling_type,
            thresh=thresh,
            comparison=comparison,
            dataset_names=dataset_names,
        )

    # comparison mode
    if not isinstance(adata, (list, tuple)):
        raise ValueError(
            "rankNet(mode='comparison') requires a list of AnnData objects"
        )
    adata_list = list(adata)
    if dataset_names is None:
        names = [f"d{i + 1}" for i in range(len(adata_list))]
    else:
        names = list(dataset_names)
        if len(names) != len(adata_list):
            raise ValueError(
                f"dataset_names length {len(names)} != adata_list length "
                f"{len(adata_list)}"
            )
    if comparison is None:
        comparison = list(range(1, len(adata_list) + 1))
    comparison = list(comparison)
    return _rank_net_comparison(
        adata_list,
        names=names,
        slot_name=slot_name,
        measure=measure,
        do_group=do_group,
        sources_use=sources_use,
        targets_use=targets_use,
        signaling=signaling,
        pairLR=pairLR,
        signaling_type=signaling_type,
        thresh=thresh,
        do_stat=do_stat,
        comparison=comparison,
    )


def _get_slot(adata: ad.AnnData, slot_name: str) -> Dict[str, Any]:
    cc = _ensure_uns(adata)
    if slot_name not in cc:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}'] missing"
        )
    return cc[slot_name]


def _slot_pair_names(
    adata: ad.AnnData, slot_name: str, prob_arr: np.ndarray
) -> List[str]:
    """Resolve the third-dim name vector for the prob array."""
    if slot_name == "netP":
        slot = _get_slot(adata, "netP")
        pathways = slot.get("pathways", None)
        if pathways is None:
            return [str(k) for k in range(prob_arr.shape[2])]
        return list(map(str, pathways))
    # net
    cc = _ensure_uns(adata)
    dn = cc.get("net", {}).get("dimnames", {})
    names = dn.get("interaction_name", dn.get("pairs", None))
    if names is None:
        return [str(k) for k in range(prob_arr.shape[2])]
    return list(map(str, names))


def _slot_cell_names(adata: ad.AnnData, slot_name: str) -> List[str]:
    cc = _ensure_uns(adata)
    dn = cc.get("net", {}).get("dimnames", {})
    src = dn.get("source", None)
    if src is not None:
        return list(map(str, src))
    cells = dn.get("cells", None)
    if cells is not None:
        return list(map(str, cells))
    return []


def _resolve_uses(
    use: Optional[Sequence], dim_names: List[str], n: int, what: str
) -> Optional[List[int]]:
    """Resolve ``sources.use`` / ``targets.use`` to 1-based int indices."""
    if use is None:
        return None
    use_arr = list(use)
    # If all are strings, convert to 1-based indices via dim_names
    if all(isinstance(u, str) for u in use_arr):
        if not all(u in dim_names for u in use_arr):
            raise ValueError(
                f"The input `{what}` should be cell group names or a "
                f"numerical vector!"
            )
        return [dim_names.index(u) + 1 for u in use_arr]
    # Numeric
    try:
        return [int(u) for u in use_arr]
    except (TypeError, ValueError):
        raise ValueError(
            f"The input `{what}` should be cell group names or a "
            f"numerical vector!"
        )


def _apply_use_mask(
    prob: np.ndarray,
    sources_use: Optional[List[int]],
    targets_use: Optional[List[int]],
) -> np.ndarray:
    """Zero-out rows / cols outside the use indices (1-based)."""
    out = prob.copy()
    n = out.shape[0]
    if sources_use is not None:
        # idx.t = setdiff(1:n, sources.use)
        keep = set(sources_use)
        idx_t = [i for i in range(1, n + 1) if i not in keep]
        if idx_t:
            out[[i - 1 for i in idx_t], :, :] = 0.0
    if targets_use is not None:
        keep = set(targets_use)
        idx_t = [i for i in range(1, n + 1) if i not in keep]
        if idx_t:
            out[:, [i - 1 for i in idx_t], :] = 0.0
    return out


def _rank_net_single(
    adata: ad.AnnData,
    slot_name: str,
    measure: str,
    do_group: bool,
    sources_use: Optional[Sequence],
    targets_use: Optional[Sequence],
    signaling: Optional[Sequence[str]],
    pairLR: Optional[Sequence[str]],
    signaling_type: Optional[Sequence[str]],
    thresh: float,
    comparison: Optional[Sequence[int]],
    dataset_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Single-mode rankNet branch (R: analysis.R:429-523)."""
    slot = _get_slot(adata, slot_name)
    if do_group:
        prob = np.asarray(slot["prob"], dtype=float).copy()
        # Mirror R's NULL-tolerant behaviour: when pval is absent (e.g. on
        # the netP slot, populated by computeCommunProbPathway, which never
        # writes a pval), skip the significance gate.
        pval_raw = slot.get("pval", None)
        if pval_raw is not None:
            pval = np.asarray(pval_raw, dtype=float)
            prob[pval > thresh] = 0.0
    else:
        prob = np.asarray(slot["prob.cell"], dtype=float).copy()
    if measure == "count":
        prob = (prob > 0).astype(float)

    cell_names = _slot_cell_names(adata, slot_name)
    src_idx = _resolve_uses(sources_use, cell_names, prob.shape[0], "sources.use")
    tgt_idx = _resolve_uses(targets_use, cell_names, prob.shape[0], "targets.use")
    prob = _apply_use_mask(prob, src_idx, tgt_idx)

    if prob.sum() == 0:
        raise ValueError("No inferred communications for the input!")

    pSum_orig = prob.sum(axis=(0, 1))
    if measure == "weight":
        if do_group:
            with np.errstate(divide="ignore", invalid="ignore"):
                pSum = -1.0 / np.log(pSum_orig)
            pSum = np.where(np.isnan(pSum), 0.0, pSum)
            idx1 = np.flatnonzero(np.isinf(pSum) | (pSum < 0))
            if idx1.size > 0:
                # values.assign <- seq(max(pSum)*1.1, max(pSum)*1.5, length.out = length(idx1))
                pmax = float(pSum.max())
                values_assign = np.linspace(
                    pmax * 1.1, pmax * 1.5, num=idx1.size
                )
                # position <- sort(pSum.original[idx1], index.return = TRUE)$ix
                # match(1:length(idx1), position)
                pos = np.argsort(pSum_orig[idx1], kind="stable") + 1  # 1-based
                # match(1:k, pos) — for each i in 1:k, find the position in pos
                k = idx1.size
                lookup = np.empty(k, dtype=int)
                for ii in range(1, k + 1):
                    lookup[ii - 1] = int(np.flatnonzero(pos == ii)[0])
                pSum[idx1] = values_assign[lookup]
        else:
            pSum = pSum_orig.copy()
    else:
        pSum = pSum_orig.copy()

    pair_names = _slot_pair_names(adata, slot_name, prob)
    if comparison is None:
        comparison_use = [1]
    else:
        comparison_use = list(comparison)
    if dataset_names is None:
        # Mirror R: object.names <- names(slot(object, slot.name)); the slot
        # is a dict with keys like "prob", "pval", ... — index by
        # comparison[1] (1-based). Most callers will see "prob" for
        # comparison=1, "pval" for comparison=2, etc.
        slot = _get_slot(adata, slot_name)
        slot_keys = list(slot.keys())
        idx = comparison_use[0] - 1
        if 0 <= idx < len(slot_keys):
            group_label = str(slot_keys[idx])
        else:
            group_label = f"d{comparison_use[0]}"
    else:
        names_list = list(dataset_names)
        idx = comparison_use[0] - 1
        if idx < 0 or idx >= len(names_list):
            group_label = f"d{comparison_use[0]}"
        else:
            group_label = names_list[idx]

    df = pd.DataFrame(
        {
            "name": pair_names,
            "contribution": pSum_orig,
            "contribution.scaled": pSum,
            "group": [group_label] * len(pair_names),
        },
        index=pair_names,
    )
    # order(df$contribution) — ascending stable sort
    order_idx = np.argsort(df["contribution"].to_numpy(), kind="stable")
    df = df.iloc[order_idx]
    df["name"] = pd.Categorical(df["name"], categories=list(df["name"]))
    # Drop entries with zero contribution
    keep = df["contribution"].to_numpy() != 0
    df = df.loc[keep]

    if signaling_type is not None:
        cc = _ensure_uns(adata)
        DB = cc.get("DB", None)
        if DB is None or "interaction" not in DB:
            raise ValueError(
                "signaling_type filter requires uns['cellchat']['DB']['interaction']"
            )
        LR = DB["interaction"]
        LR_filt = LR[LR["annotation"].isin(list(signaling_type))]
        if slot_name == "netP":
            signaling = list(np.unique(LR_filt["pathway_name"].astype(str)))
        else:
            pairLR = list(LR_filt["interaction_name"].astype(str))

    if slot_name == "netP" and signaling is not None:
        df = df[df["name"].isin(list(signaling))]
    elif slot_name == "netP" and pairLR is not None:
        raise ValueError(
            "You need to set `slot.name == 'net'` if showing specific L-R pairs "
        )
    if slot_name == "net" and pairLR is not None:
        df = df[df["name"].isin(list(pairLR))]
    elif slot_name == "net" and signaling is not None:
        raise ValueError(
            "You need to set `slot.name == 'netP'` if showing specific signaling pathways "
        )

    return df.reset_index(drop=True)


def _rank_net_comparison(
    adata_list: Sequence[ad.AnnData],
    names: List[str],
    slot_name: str,
    measure: str,
    do_group: bool,
    sources_use: Optional[Sequence],
    targets_use: Optional[Sequence],
    signaling: Optional[Sequence[str]],
    pairLR: Optional[Sequence[str]],
    signaling_type: Optional[Sequence[str]],
    thresh: float,
    do_stat: bool,
    comparison: List[int],
) -> pd.DataFrame:
    """Comparison-mode rankNet branch (R: analysis.R:525-802)."""
    prob_list: List[np.ndarray] = []
    pSum_list: List[np.ndarray] = []
    pSum_orig_list: List[np.ndarray] = []
    pair_name_lists: List[List[str]] = []
    idx_list: List[np.ndarray] = []
    pSum_orig_all: List[float] = []
    object_names_comparison: List[str] = []

    for c in comparison:
        # 1-based index into adata_list
        if c < 1 or c > len(adata_list):
            raise ValueError(
                f"comparison index {c} out of range 1..{len(adata_list)}"
            )
        adata = adata_list[c - 1]
        slot = _get_slot(adata, slot_name)
        if do_group:
            prob = np.asarray(slot["prob"], dtype=float).copy()
            # Mirror R's NULL-tolerant behaviour (also applied in
            # _rank_net_single): when pval is absent (e.g. on the netP
            # slot, populated by computeCommunProbPathway, which never
            # writes a pval), skip the significance gate.
            pval_raw = slot.get("pval", None)
            if pval_raw is not None:
                pval = np.asarray(pval_raw, dtype=float)
                prob[pval > thresh] = 0.0
        else:
            prob = np.asarray(slot["prob.cell"], dtype=float).copy()
        if measure == "count":
            prob = (prob > 0).astype(float)
        prob_list.append(prob)
        cell_names = _slot_cell_names(adata, slot_name)
        src_idx = _resolve_uses(sources_use, cell_names, prob.shape[0], "sources.use")
        tgt_idx = _resolve_uses(targets_use, cell_names, prob.shape[0], "targets.use")
        prob = _apply_use_mask(prob, src_idx, tgt_idx)

        if prob.sum() == 0:
            raise ValueError("No inferred communications for the input!")

        pSum_orig_i = prob.sum(axis=(0, 1))
        pSum_orig_list.append(pSum_orig_i)
        if measure == "weight":
            if do_group:
                with np.errstate(divide="ignore", invalid="ignore"):
                    pSum_i = -1.0 / np.log(pSum_orig_i)
                pSum_i = np.where(np.isnan(pSum_i), 0.0, pSum_i)
                idx_i = np.flatnonzero(np.isinf(pSum_i) | (pSum_i < 0))
                idx_list.append(idx_i)
                pSum_orig_all.extend(pSum_orig_i[idx_i].tolist())
            else:
                pSum_i = pSum_orig_i.copy()
                idx_list.append(np.array([], dtype=int))
                pSum_orig_all.extend(pSum_orig_i.tolist())
        else:  # count
            pSum_i = pSum_orig_i.copy()
            idx_list.append(np.array([], dtype=int))
        pSum_list.append(pSum_i)
        pair_name_lists.append(_slot_pair_names(adata, slot_name, prob))

        # object_names[comparison[i]]
        # In R, names(slot(object, slot.name)) gives the dataset names.
        # Here we use the pre-resolved `names` list (one per AnnData),
        # indexed by (c-1) since c is 1-based.
        object_names_comparison.append(names[c - 1])

    # Detie weight values when they are -inf / NA (R lines 588-598)
    if measure == "weight" and do_group:
        flat_idx_concat = np.concatenate(idx_list) if any(i.size > 0 for i in idx_list) else np.array([], dtype=int)
        total_idx = sum(int(i.size) for i in idx_list)
        if total_idx > 0:
            max_pSum = max(float(p.max()) for p in pSum_list)
            values_assign = np.linspace(max_pSum * 1.1, max_pSum * 1.5, num=total_idx)
            position = np.argsort(np.asarray(pSum_orig_all), kind="stable") + 1  # 1-based
            for i in range(len(comparison)):
                if idx_list[i].size == 0:
                    continue
                k = idx_list[i].size
                if i == 0:
                    # match(1:k, position)
                    lookup = np.empty(k, dtype=int)
                    for ii in range(1, k + 1):
                        lookup[ii - 1] = int(np.flatnonzero(position == ii)[0])
                else:
                    prior = sum(int(idx_list[j].size) for j in range(i))
                    rng = np.arange(prior + 1, prior + k + 1)
                    lookup = np.empty(k, dtype=int)
                    for ii_idx, val in enumerate(rng):
                        lookup[ii_idx] = int(np.flatnonzero(position == val)[0])
                pSum_list[i][idx_list[i]] = values_assign[lookup]

    pair_name_all = list(dict.fromkeys([n for lst in pair_name_lists for n in lst]))

    df_list: List[pd.DataFrame] = []
    for i in range(len(comparison)):
        df_i = pd.DataFrame(
            {
                "name": pair_name_all,
                "contribution": 0.0,
                "contribution.scaled": 0.0,
                "group": [object_names_comparison[i]] * len(pair_name_all),
            },
            index=pair_name_all,
        )
        # df[[i]][pair.name[[i]], 3] <- pSum[[i]] (col 3 = contribution.scaled)
        for j, nm in enumerate(pair_name_lists[i]):
            df_i.at[nm, "contribution.scaled"] = float(pSum_list[i][j])
            df_i.at[nm, "contribution"] = float(pSum_orig_list[i][j])
        df_list.append(df_i)

    # contribution.relative — R lines 622-632
    contribution_relative: List[np.ndarray] = []
    for i in range(1, len(comparison)):
        # df[[length(comparison) - i + 1]]$contribution / df[[1]]$contribution
        # Then format(..., digits=1), then NA -> 0
        denom = df_list[0]["contribution"].to_numpy()
        num = df_list[len(comparison) - i]["contribution"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / denom
        ratio_fmt = _r_format_digits1(ratio)
        ratio_fmt = np.where(np.isnan(ratio_fmt), 0.0, ratio_fmt)
        contribution_relative.append(ratio_fmt)
    cr_names = [
        f"contribution.relative.{j + 1}" for j in range(len(contribution_relative))
    ]
    for i in range(len(comparison)):
        for j, nm in enumerate(cr_names):
            df_list[i][nm] = contribution_relative[j]

    # Sort according to R lines 633-642 — use df[[1]] keys
    df1 = df_list[0].copy()
    df1["contribution.data2"] = df_list[len(comparison) - 1]["contribution"].to_numpy()
    if len(comparison) == 2:
        keys = ["contribution.relative.1", "contribution", "contribution.data2"]
        ascending = [False, True, False]
    elif len(comparison) == 3:
        keys = [
            "contribution.relative.1",
            "contribution.relative.2",
            "contribution",
            "contribution.data2",
        ]
        ascending = [False, False, True, False]
    elif len(comparison) == 4:
        keys = [
            "contribution.relative.1",
            "contribution.relative.2",
            "contribution.relative.3",
            "contribution",
            "contribution.data2",
        ]
        ascending = [False, False, False, True, False]
    else:
        keys = [
            f"contribution.relative.{j}" for j in range(1, len(comparison))
        ] + ["contribution", "contribution.data2"]
        ascending = [False] * (len(comparison) - 1) + [True, False]
    order_idx = df1.sort_values(by=keys, ascending=ascending, kind="stable").index
    # Apply identical reordering to all df_list[i]
    df_list_sorted: List[pd.DataFrame] = []
    for i in range(len(comparison)):
        df_i = df_list[i].loc[order_idx].copy()
        df_i["name"] = pd.Categorical(df_i["name"], categories=list(df_i["name"]))
        df_list_sorted.append(df_i)

    df_combined = pd.concat(df_list_sorted, axis=0)
    df_combined["group"] = pd.Categorical(
        df_combined["group"], categories=object_names_comparison
    )

    # Drop rows for any pair whose total contribution across datasets is 0
    # (R lines 721-726)
    name_total = (
        df_combined.groupby("name", observed=False)["contribution"].sum()
    )
    keep_names = list(name_total[name_total != 0].index)
    df_combined = df_combined[df_combined["name"].isin(keep_names)]

    # Slot-specific signaling/pairLR filter (R lines 728-737)
    if slot_name == "netP" and signaling is not None:
        df_combined = df_combined[df_combined["name"].isin(list(signaling))]
    elif slot_name == "netP" and pairLR is not None:
        raise ValueError(
            "You need to set `slot.name == 'net'` if showing specific L-R pairs "
        )
    if slot_name == "net" and pairLR is not None:
        df_combined = df_combined[df_combined["name"].isin(list(pairLR))]
    elif slot_name == "net" and signaling is not None:
        raise ValueError(
            "You need to set `slot.name == 'netP'` if showing specific signaling pathways "
        )

    if signaling_type is not None:
        # Mirror R lines 498-505 (single mode) — comparison mode in R doesn't
        # explicitly invoke this, but for parity we apply the same logic
        # before the signaling/pairLR filter when signaling_type is set.
        cc = _ensure_uns(adata_list[comparison[0] - 1])
        DB = cc.get("DB", None)
        if DB is None or "interaction" not in DB:
            raise ValueError(
                "signaling_type filter requires uns['cellchat']['DB']['interaction']"
            )
        LR = DB["interaction"]
        LR_filt = LR[LR["annotation"].isin(list(signaling_type))]
        if slot_name == "netP":
            sig_keep = list(np.unique(LR_filt["pathway_name"].astype(str)))
            df_combined = df_combined[df_combined["name"].isin(sig_keep)]
        else:
            pair_keep = list(LR_filt["interaction_name"].astype(str))
            df_combined = df_combined[df_combined["name"].isin(pair_keep)]

    if do_stat and len(comparison) == 2:
        # Wilcoxon paired test per pair (R lines 685-707)
        from scipy import stats

        pvals_per_name: Dict[str, float] = {}
        for nm in pair_name_all:
            n_rows = prob_list[0].shape[0]
            if any(p.shape[0] != n_rows for p in prob_list):
                raise ValueError(
                    "Statistical test is not applicable to datasets with different "
                    "cellular compositions! Please set `do_stat=False`"
                )
            prob_values = np.full((n_rows * n_rows, len(comparison)), np.nan)
            for j in range(len(comparison)):
                if nm in pair_name_lists[j]:
                    k = pair_name_lists[j].index(nm)
                    prob_values[:, j] = prob_list[j][:, :, k].flatten(order="F")
                else:
                    prob_values[:, j] = np.nan
            mask = np.nansum(prob_values, axis=1) != 0
            prob_values = prob_values[mask, :]
            if prob_values.shape[0] > 3 and not np.isnan(prob_values).any():
                # Wilcoxon paired
                res = stats.wilcoxon(prob_values[:, 0], prob_values[:, 1])
                pv = float(res.pvalue) if not np.isnan(res.pvalue) else 0.0
            else:
                pv = 0.0
            pvals_per_name[nm] = pv
        df_combined["pvalues"] = df_combined["name"].astype(str).map(pvals_per_name).astype(float)

    return df_combined.reset_index(drop=True)


def _r_format_digits1(x: np.ndarray, digits: int = 1) -> np.ndarray:
    """Mimic R's ``as.numeric(format(x, digits=N))``.

    R's :func:`format` aligns *all* elements to a uniform number of
    decimal places: the smallest number of decimals that gives every
    finite, non-zero element at least ``digits`` significant digits.
    Inf / NaN entries are preserved.

    Parameters
    ----------
    x : numpy.ndarray
        Numeric array.
    digits : int, default ``1``
        Minimum significant digits per element.

    Returns
    -------
    numpy.ndarray
        Array with the same shape; finite elements are rounded to a
        common decimal precision; Inf / NaN preserved.
    """
    x = np.asarray(x, dtype=float)
    out = np.copy(x)
    finite_nz = np.isfinite(x) & (x != 0)
    if not finite_nz.any():
        return out
    log_abs = np.log10(np.abs(x[finite_nz]))
    decimals_needed = digits - 1 - np.floor(log_abs).astype(int)
    target_decimals = max(0, int(np.max(decimals_needed)))
    finite_mask = np.isfinite(x)
    out[finite_mask] = np.round(x[finite_mask], target_decimals)
    return out


# ---------------------------------------------------------------------------
# rankNetPairwise (R: analysis.R:1932)
# ---------------------------------------------------------------------------
def rankNetPairwise(
    adata: ad.AnnData,
    LR_use: Optional[pd.DataFrame] = None,
) -> ad.AnnData:
    """Rank L-R interactions for every pair of cell groups.

    Direct port of R's :func:`rankNetPairwise` (``analysis.R:1932``). The
    function reads ``adata.uns["cellchat"]["net"]["prob"]`` and
    ``["pval"]`` (as written by :func:`computeAvgCommunProb`) and the
    significant-LR table ``adata.uns["cellchat"]["LR"]["LRsig"]`` (or
    ``LR_use`` when provided), then writes a nested dict at
    ``adata.uns["cellchat"]["net"]["pairwiseRank"]`` with the structure::

        pairwiseRank[source_group][target_group] = pandas.DataFrame

    Each per-(source, target) data frame has one row per L-R pair, sorted
    by ``pval`` ascending then ``-prob`` (i.e. ``prob`` descending). The
    columns mirror R's: ``pathway_index``, ``interaction_name``,
    ``interaction_name_2``, ``pathway_name``, ``ligand``, ``receptor``,
    ``prob``, ``pval``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData populated by ``computeAvgCommunProb`` (so ``net['prob']``
        and ``net['pval']`` are dense ``(K, K, nLR)`` arrays).
    LR_use : pandas.DataFrame, optional
        Override for ``adata.uns["cellchat"]["LR"]["LRsig"]``. The frame
        must have columns ``interaction_name``, ``interaction_name_2``,
        ``pathway_name``, ``ligand`` and ``receptor``; its row index is
        carried into the per-(i, j) data frame.

    Returns
    -------
    anndata.AnnData
        ``adata`` with ``adata.uns["cellchat"]["net"]["pairwiseRank"]``
        populated.
    """
    cc = _ensure_uns(adata)
    if "net" not in cc or "prob" not in cc["net"] or "pval" not in cc["net"]:
        raise ValueError(
            "rankNetPairwise requires uns['cellchat']['net']['prob'] and ['pval']"
        )
    if LR_use is None:
        if "LR" not in cc or "LRsig" not in cc["LR"]:
            raise ValueError(
                "rankNetPairwise requires uns['cellchat']['LR']['LRsig'] "
                "or an explicit LR_use DataFrame"
            )
        pairLR_use = cc["LR"]["LRsig"]
    else:
        pairLR_use = LR_use

    net = cc["net"]
    prob = np.asarray(net["prob"], dtype=float)
    pval = np.asarray(net["pval"], dtype=float)
    K = prob.shape[0]
    nLR = prob.shape[2]
    if pval.shape != prob.shape:
        raise ValueError("net['prob'] and net['pval'] must have the same shape")
    if len(pairLR_use) != nLR:
        raise ValueError(
            f"LR_use length {len(pairLR_use)} != net['prob'].shape[2] {nLR}"
        )

    # Resolve cell-group names (rownames of prob[,,k])
    dn = net.get("dimnames", {})
    src_names = list(map(str, dn.get("source", [str(k) for k in range(K)])))
    tgt_names = list(map(str, dn.get("target", src_names)))

    pathway_index = np.arange(1, nLR + 1)  # 1-based, mirrors R
    pairwise_rank: Dict[str, Dict[str, pd.DataFrame]] = {}
    for i in range(K):
        per_src: Dict[str, pd.DataFrame] = {}
        for j in range(K):
            pvalij = pval[i, j, :].astype(float)
            probij = prob[i, j, :].astype(float)
            data = pd.DataFrame(
                {
                    "pathway_index": pathway_index,
                    "interaction_name": pairLR_use["interaction_name"].to_numpy(),
                    "interaction_name_2": pairLR_use["interaction_name_2"].to_numpy(),
                    "pathway_name": pairLR_use["pathway_name"].to_numpy(),
                    "ligand": pairLR_use["ligand"].to_numpy(),
                    "receptor": pairLR_use["receptor"].to_numpy(),
                    "prob": probij,
                    "pval": pvalij,
                },
                index=pairLR_use.index,
            )
            # order(pval, -prob)
            order_idx = np.lexsort((-data["prob"].to_numpy(), data["pval"].to_numpy()))
            per_src[tgt_names[j]] = data.iloc[order_idx]
        pairwise_rank[src_names[i]] = per_src

    net["pairwiseRank"] = pairwise_rank
    return adata


# ---------------------------------------------------------------------------
# rankSimilarity (R: analysis.R:1755)
# ---------------------------------------------------------------------------
def rankSimilarity(
    adata: ad.AnnData,
    slot_name: str = "netP",
    type: str = "functional",
    comparison1: Optional[Sequence[int]] = None,
    comparison2: Sequence[int] = (1, 2),
    dataset_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rank pathways by Euclidean distance in the embedding space.

    Direct port of the data-construction part of R's
    :func:`rankSimilarity` (``analysis.R:1755``). The function reads the
    similarity-embedding matrix at::

        adata.uns["cellchat"][slot_name]["similarity"][type]["dr"][comparison_name]

    and computes the per-pathway Euclidean distance between the two
    datasets named in ``comparison2``. Plot construction is omitted; we
    return the underlying data frame.

    Parameters
    ----------
    adata : anndata.AnnData
        Holding the merged similarity object. The "dataset" labels are
        the suffix after ``--`` in the row names of the embedding
        matrix (e.g. ``"WNT--d1"`` -> dataset ``"d1"``).
    slot_name : {"netP", "net"}, default ``"netP"``
        Which slot's similarity embedding to read.
    type : {"functional", "structural"}, default ``"functional"``
        Which similarity type to use.
    comparison1 : sequence of int, optional
        1-based indices of datasets contributing to the embedding. The
        embedding key is ``"-".join(comparison1)`` (e.g. ``"1-2"``).
        Default is ``[1, 2, ..., K]`` where K is the number of unique
        datasets in ``adata.uns["cellchat"]["meta"]["datasets"]``.
    comparison2 : sequence of int, default ``(1, 2)``
        1-based indices into ``comparison1`` selecting which two
        datasets to compare.
    dataset_names : sequence of str, optional
        Override the dataset name vector that the function uses to slice
        embedding rows.

    Returns
    -------
    pandas.DataFrame
        Columns ``name`` and ``dist``, one row per pathway shared between
        the two datasets, sorted ascending by ``dist``.
    """
    cc = _ensure_uns(adata)
    slot = cc.get(slot_name, None)
    if slot is None:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] missing")
    if "similarity" not in slot:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['similarity'] missing"
        )
    if type not in slot["similarity"]:
        raise ValueError(
            f"similarity['{type}'] missing"
        )
    if "dr" not in slot["similarity"][type]:
        raise ValueError(
            f"similarity['{type}']['dr'] missing"
        )

    if comparison1 is None:
        # length(unique(object@meta$datasets))
        meta = cc.get("meta", {})
        if "datasets" in meta:
            n_datasets = len(np.unique(np.asarray(meta["datasets"])))
        else:
            n_datasets = len(slot["similarity"][type]["dr"])
        comparison1 = list(range(1, n_datasets + 1))
    comparison1 = list(comparison1)
    comparison2 = list(comparison2)

    comparison_name = "-".join(str(c) for c in comparison1)
    if comparison_name not in slot["similarity"][type]["dr"]:
        raise ValueError(
            f"similarity['{type}']['dr']['{comparison_name}'] missing"
        )

    # comparison2.name <- names(slot(object, slot.name))[comparison1[comparison2]]
    if dataset_names is None:
        # Mirror R: names(netP)[comparison1[comparison2]]. In our Python
        # storage we provide a special list at slot["__dataset_names__"]
        # if available; otherwise fall back to the keys of the slot
        # filtered to the dataset entries.
        if "__dataset_names__" in slot:
            slot_names = list(slot["__dataset_names__"])
        else:
            slot_names = [k for k in slot.keys() if k != "similarity"]
    else:
        slot_names = list(dataset_names)
    # 1-based indexing: comparison1[comparison2[i]] -> Python: comparison1[c2_i - 1]
    comparison2_name: List[str] = []
    for c in comparison2:
        idx = comparison1[c - 1] - 1
        if idx < 0 or idx >= len(slot_names):
            raise ValueError(
                f"comparison resolves to dataset index {idx + 1} which is "
                f"out of range 1..{len(slot_names)}"
            )
        comparison2_name.append(slot_names[idx])

    Y = slot["similarity"][type]["dr"][comparison_name]
    if isinstance(Y, pd.DataFrame):
        Y_arr = Y.to_numpy()
        rownames = list(map(str, Y.index))
    else:
        Y_arr = np.asarray(Y)
        # Stored alongside in slot["similarity"][type]["dr_rownames"][comparison_name]?
        rownames = list(map(str, slot["similarity"][type].get("dr_rownames", {}).get(comparison_name, [])))
        if not rownames:
            raise ValueError(
                "Embedding matrix lacks row names; either provide a DataFrame or "
                "store rownames at similarity[type]['dr_rownames'][comparison_name]"
            )

    # group <- sub(".*--", "", rownames(Y))
    groups = np.array([rn.split("--")[-1] for rn in rownames])
    mask1 = groups == comparison2_name[0]
    mask2 = groups == comparison2_name[1]
    data1 = Y_arr[mask1, :]
    data2 = Y_arr[mask2, :]
    rn1 = [rn.split("--")[0] for rn, m in zip(rownames, mask1) if m]
    rn2 = [rn.split("--")[0] for rn, m in zip(rownames, mask2) if m]

    # pathway.show = intersect(rn1, rn2). intersect preserves the order of
    # the first argument.
    in_set2 = set(rn2)
    pathway_show = [n for n in rn1 if n in in_set2]
    # subset to pathway_show — preserve pathway_show order
    data1_sub = np.array([data1[rn1.index(n)] for n in pathway_show])
    data2_sub = np.array([data2[rn2.index(n)] for n in pathway_show])

    dists = np.sqrt(np.sum((data1_sub - data2_sub) ** 2, axis=1))
    df = pd.DataFrame({"name": pathway_show, "dist": dists}, index=pathway_show)
    df = df.sort_values(by="dist", ascending=True, kind="stable")
    df["name"] = pd.Categorical(df["name"], categories=list(df["name"]))
    return df
