"""Network centrality measures (``computeCentralityLocal``).

Direct port of ``SpatialCellChat::computeCentralityLocal``
(R: ``analysis.R:9``).  Eleven measures are computed when
``degree_only=False``; five when ``degree_only=True``:

- Always (5): ``outdeg_unweighted``, ``indeg_unweighted``, ``outdeg``,
  ``indeg``, ``page_rank``.
- Full set adds (6): ``hub``, ``authority``, ``eigen``, ``betweenness``,
  ``flowbet``, ``info``.

Per porting contract §3.C the underlying R packages map to:

- ``igraph::strength/hub_score/authority_score/eigen_centrality/page_rank/betweenness``
  → :mod:`igraph` (python-igraph).
- ``sna::flowbet`` → :func:`networkx.current_flow_betweenness_centrality`
  (random-walk current-flow betweenness on the symmetrised graph).
- ``sna::infocent`` → :func:`networkx.current_flow_closeness_centrality`
  (information centrality on the symmetrised graph).

The first nine measures are bit-identical to R; the two
network-flow-based measures are conceptually equivalent but not
bit-identical due to the algorithm-substitution above.
"""

from __future__ import annotations

from typing import Optional, Union

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "computeCentralityLocal",
    "CENTRALITY_NAMES_FULL",
    "CENTRALITY_NAMES_DEGREE",
    "centr_value",
    "centr_signaling_names",
]


CENTRALITY_NAMES_DEGREE = (
    "outdeg_unweighted",
    "indeg_unweighted",
    "outdeg",
    "indeg",
    "page_rank",
)

CENTRALITY_NAMES_FULL = (
    "outdeg_unweighted",
    "indeg_unweighted",
    "outdeg",
    "indeg",
    "hub",
    "authority",
    "eigen",
    "page_rank",
    "betweenness",
    "flowbet",
    "info",
)


def _flowbet_via_networkx(net: np.ndarray) -> np.ndarray:
    """Replacement for ``sna::flowbet``.

    Maps to :func:`networkx.current_flow_betweenness_centrality` on the
    symmetrised undirected graph (per slice5 contract). Returns zeros if
    the graph is not connected (mirrors R's ``tryCatch(..., error = ...,
    matrix(0, n, 1))``).
    """
    n = net.shape[0]
    sym = (np.asarray(net, dtype=float) + np.asarray(net, dtype=float).T) / 2.0
    np.fill_diagonal(sym, 0.0)
    if (sym <= 0).all():
        return np.zeros(n)
    G = nx.from_numpy_array(sym, create_using=nx.Graph)
    if not nx.is_connected(G):
        return np.zeros(n)
    res = nx.current_flow_betweenness_centrality(G, normalized=False, weight="weight")
    out = np.zeros(n)
    for i, v in res.items():
        out[i] = float(v)
    return out


def _info_via_networkx(net: np.ndarray) -> np.ndarray:
    """Replacement for ``sna::infocent`` with ``cmode="lower"``.

    Maps to :func:`networkx.current_flow_closeness_centrality` on the
    symmetrised undirected graph (per slice5 contract). ``rescale=True``
    is mirrored by dividing through by the sum (the R behaviour for
    ``rescale=TRUE``). Returns zeros if the graph is not connected.
    """
    n = net.shape[0]
    arr = np.asarray(net, dtype=float)
    # R's symmetrize(..., rule="lower") = pmin(m, t(m))
    sym = np.minimum(arr, arr.T)
    np.fill_diagonal(sym, 0.0)
    if (sym <= 0).all():
        return np.zeros(n)
    G = nx.from_numpy_array(sym, create_using=nx.Graph)
    if not nx.is_connected(G):
        return np.zeros(n)
    res = nx.current_flow_closeness_centrality(G, weight="weight")
    out = np.zeros(n)
    for i, v in res.items():
        out[i] = float(v)
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def computeCentralityLocal(
    net: Union[np.ndarray, pd.DataFrame],
    degree_only: bool = False,
) -> np.ndarray:
    """Compute centrality measures for a single signaling network.

    Mirrors ``SpatialCellChat::computeCentralityLocal``
    (``R/analysis.R:9``).

    Parameters
    ----------
    net : numpy.ndarray | pandas.DataFrame
        Square (n_nodes x n_nodes) directed weighted adjacency matrix.
    degree_only : bool, default False
        When True, compute only the five degree-style measures
        (``outdeg_unweighted``, ``indeg_unweighted``, ``outdeg``,
        ``indeg``, ``page_rank``). When False, compute all eleven.

    Returns
    -------
    numpy.ndarray
        A ``(len(centr.name), n_nodes)`` matrix in the same row order
        used by the R reference. Row names are not embedded; the caller
        should consult :data:`CENTRALITY_NAMES_FULL` /
        :data:`CENTRALITY_NAMES_DEGREE`.
    """
    if isinstance(net, pd.DataFrame):
        net_arr = np.asarray(net.values, dtype=float)
    else:
        net_arr = np.asarray(net, dtype=float)

    if net_arr.ndim != 2 or net_arr.shape[0] != net_arr.shape[1]:
        raise ValueError("`net` must be a square 2-D matrix.")

    n_nodes = net_arr.shape[0]

    # 1. Build directed weighted igraph (loops allowed for parity with R)
    G = ig.Graph.Weighted_Adjacency(
        net_arr.tolist(), mode="directed", attr="weight", loops=True
    )

    # 2. Always compute degree-style measures
    outdeg_unw = (net_arr > 0).sum(axis=1).astype(float)
    indeg_unw = (net_arr > 0).sum(axis=0).astype(float)
    outdeg = np.asarray(G.strength(mode="out", weights="weight"), dtype=float)
    indeg = np.asarray(G.strength(mode="in", weights="weight"), dtype=float)
    page_rank = np.asarray(G.pagerank(weights="weight"), dtype=float)

    if degree_only:
        rows = np.vstack([outdeg_unw, indeg_unw, outdeg, indeg, page_rank])
        return rows

    # 3. Hub / authority — both expect weighted directed
    hub = np.asarray(G.hub_score(weights="weight"), dtype=float)
    authority = np.asarray(G.authority_score(weights="weight"), dtype=float)

    # 4. Eigen centrality. R's `igraph::eigen_centrality(G)` defaults
    #    to `directed=FALSE`. python-igraph rejects calling
    #    `eigenvector_centrality(directed=False)` on a directed graph,
    #    so we collapse the graph to undirected with `combine_edges="sum"`.
    G_und = G.as_undirected(combine_edges="sum")
    eigen = np.asarray(
        G_und.eigenvector_centrality(weights="weight", scale=True), dtype=float
    )

    # 5. Betweenness with inverted weights (R: ``E(G)$weight <- 1/E(G)$weight``)
    G_inv = G.copy()
    G_inv.es["weight"] = [1.0 / w for w in G_inv.es["weight"]]
    betweenness = np.asarray(
        G_inv.betweenness(weights="weight", directed=True), dtype=float
    )

    # 6. flowbet via networkx (per slice5 mapping)
    flowbet = _flowbet_via_networkx(net_arr)
    # 7. info via networkx (per slice5 mapping)
    info = _info_via_networkx(net_arr)

    rows = np.vstack(
        [
            outdeg_unw,
            indeg_unw,
            outdeg,
            indeg,
            hub,
            authority,
            eigen,
            page_rank,
            betweenness,
            flowbet,
            info,
        ]
    )
    return rows


# ---------------------------------------------------------------------------
# Centrality slot accessors
# ---------------------------------------------------------------------------
# Mirror R's `obj@netP$centr["indeg", , signaling]` 3-D indexing on the flat
# schema written by `netAnalysis_computeCentrality`:
#
#     {"values": (n_centr, n_nodes, n_signal) ndarray,
#      "centr_names": [...],
#      "node_names": [...],
#      "signaling_names": [...]}
#
# Both ``centr`` (do_group=True) and ``centr.cell`` (do_group=False) use this
# schema; consumers across viz_network / viz_spatial / viz_embedding /
# viz_heatmap / analysis_topics route through the helpers below.


def centr_value(
    centr_dict: dict,
    measure: str,
    signaling: Optional[str] = None,
) -> np.ndarray:
    """Look up centrality values for ``(measure, signaling)``.

    Mirrors R's ``obj@netP$centr["indeg", , signaling]`` indexing.

    Parameters
    ----------
    centr_dict : dict
        Flat schema written by :func:`netAnalysis_computeCentrality`
        (keys ``values``, ``centr_names``, ``node_names``,
        ``signaling_names``).
    measure : str
        One of the centrality measure names (e.g. ``"indeg"``,
        ``"outdeg"``, ``"flowbet"``, ``"info"``); must be present in
        ``centr_dict["centr_names"]``.
    signaling : str, optional
        Signaling pathway name. When ``None``, returns the
        ``(n_nodes, n_signal)`` matrix for ``measure``.

    Returns
    -------
    numpy.ndarray
        1-D ``(n_nodes,)`` array when ``signaling`` is supplied,
        else 2-D ``(n_nodes, n_signal)``.
    """
    if not isinstance(centr_dict, dict):
        raise TypeError(
            "centr_value expects a dict with the flat schema "
            "{values, centr_names, node_names, signaling_names}; got "
            f"{type(centr_dict).__name__}"
        )
    if "values" not in centr_dict or "centr_names" not in centr_dict:
        # Truncate the key dump: the most common failure mode here is the
        # legacy nested ``{signaling: {measure: array}}`` schema where the
        # key list is hundreds of LR-pair names — printing them all is not
        # informative.
        keys = list(centr_dict.keys())
        if len(keys) > 8:
            keys_repr = repr(keys[:5])[:-1] + f", ... and {len(keys) - 5} more]"
        else:
            keys_repr = repr(keys)
        raise TypeError(
            "centr_value expects the flat schema "
            "{values, centr_names, node_names, signaling_names} produced by "
            "`netAnalysis_computeCentrality`; got dict with "
            f"{len(keys)} keys: {keys_repr}.  If this is the legacy nested "
            "{signaling: {measure: array}} layout, replace the lookup with "
            "`centr_value(centr, measure, signaling)` on the flat schema."
        )
    names = list(centr_dict["centr_names"])
    if measure not in names:
        raise KeyError(
            f"Centrality measure {measure!r} not in centr_names={names!r}"
        )
    ci = names.index(measure)
    values = np.asarray(centr_dict["values"], dtype=float)
    if signaling is None:
        return values[ci]
    sig_names = list(centr_dict.get("signaling_names", []))
    if signaling not in sig_names:
        raise KeyError(
            f"Signaling {signaling!r} not in signaling_names={sig_names!r}"
        )
    si = sig_names.index(signaling)
    return values[ci, :, si]


def centr_signaling_names(centr_dict: dict) -> list:
    """Return the signaling pathway names recorded in ``centr_dict``.

    Mirrors R's ``dimnames(obj@netP$centr)[[3]]`` on the flat schema.
    """
    if not isinstance(centr_dict, dict):
        raise TypeError("centr_dict must be a dict")
    if "signaling_names" in centr_dict:
        return list(centr_dict["signaling_names"])
    raise KeyError(
        "centr_dict missing 'signaling_names' key — expected flat schema "
        "{values, centr_names, node_names, signaling_names}"
    )
