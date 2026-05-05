"""Smoke tests for ``spatialcellchat.viz_heatmap`` — slice 6b."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import sparse as _sparse
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import sparse


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")

from spatialcellchat.viz_heatmap import (
    doHeatmap,
    netAnalysis_signalingRole_heatmap,
    netVisual_TopicComposition,
    netVisual_TopicSignaling,
    netVisual_bubble,
    netVisual_heatmap,
)


# ---------------------------------------------------------------------------
# Helper: build a small AnnData with cellchat slots
# ---------------------------------------------------------------------------
def _make_adata_with_net(n_cells: int = 9, n_genes: int = 12, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype("float32"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    obs["active_ident"] = pd.Categorical(
        [f"G{i % 3}" for i in range(n_cells)], categories=["G0", "G1", "G2"]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    K = 3
    nLR = 4
    prob_grp = rng.uniform(0.0, 1.0, size=(K, K, nLR))
    pval_grp = rng.uniform(0.0, 0.05, size=(K, K, nLR))
    pair_names = [f"pair{k}" for k in range(nLR)]
    cell_names = ["G0", "G1", "G2"]
    LRsig = pd.DataFrame({
        "interaction_name": pair_names,
        "interaction_name_2": [f"L{i} - R{i}" for i in range(nLR)],
        "ligand": [f"L{i}" for i in range(nLR)],
        "receptor": [f"R{i}" for i in range(nLR)],
        "pathway_name": [f"PW{i % 2}" for i in range(nLR)],
        "annotation": ["Secreted Signaling"] * nLR,
        "evidence": ["KEGG"] * nLR,
    }, index=pair_names)
    adata.uns["cellchat"] = {
        "DB": {"interaction": LRsig.copy()},
        "LR": {"LRsig": LRsig.copy()},
        "net": {
            "prob": prob_grp,
            "pval": pval_grp,
            "count": np.sum(prob_grp > 0, axis=2).astype(float),
            "weight": np.sum(prob_grp, axis=2).astype(float),
            "LR.sig": pair_names,
            "dimnames": {
                "source": cell_names,
                "target": cell_names,
                "interaction_name": pair_names,
            },
            "meta": {"cells": list(obs.index), "pairs": pair_names,
                     "groups": cell_names, "idents": np.asarray(obs["active_ident"])},
        },
        "netP": {
            "prob": prob_grp,
            "pval": pval_grp,
            "pathways": ["PW0", "PW1"],
            "dimnames": [cell_names, cell_names, ["PW0", "PW1"]],
        },
    }
    return adata


def _add_centr_slot(adata: ad.AnnData, slot_name: str = "netP", n_pathways: int = 3) -> None:
    """Populate ``slot_name['centr']`` with a synthetic centrality dict."""
    K = 3
    rng = np.random.default_rng(7)
    centr = np.zeros((11, K, n_pathways))
    for i in range(n_pathways):
        centr[2, :, i] = rng.uniform(0.0, 1.0, size=K)  # outdeg
        centr[3, :, i] = rng.uniform(0.0, 1.0, size=K)  # indeg
        centr[9, :, i] = rng.uniform(0.0, 1.0, size=K)  # flowbet
        centr[10, :, i] = rng.uniform(0.0, 1.0, size=K)  # info
    centr_names = [
        "outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg",
        "hub", "authority", "eigen", "page_rank", "betweenness",
        "flowbet", "info",
    ]
    pathway_names = [f"PW{i}" for i in range(n_pathways)]
    adata.uns["cellchat"][slot_name]["centr"] = {
        "values": centr,
        "centr_names": centr_names,
        "node_names": ["G0", "G1", "G2"],
        "signaling_names": pathway_names,
    }


def _add_topic_slot(adata: ad.AnnData) -> None:
    """Populate ``net['topic']`` with synthetic outgoing + incoming dicts."""
    K_topics = 3
    n_cells = adata.n_obs
    nLR = 4
    rng = np.random.default_rng(11)
    cell_emb = pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(n_cells, K_topics)),
        index=adata.obs_names,
        columns=[f"Topic_{i + 1}" for i in range(K_topics)],
    )
    sig_emb = pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(nLR, K_topics)),
        index=[f"pair{k}" for k in range(nLR)],
        columns=[f"Topic_{i + 1}" for i in range(K_topics)],
    )
    adata.uns["cellchat"]["net"]["topic"] = {
        "outgoing": {"cell": cell_emb.copy(), "signaling": sig_emb.copy()},
        "incoming": {"cell": cell_emb.copy(), "signaling": sig_emb.copy()},
    }


# ---------------------------------------------------------------------------
# netVisual_heatmap
# ---------------------------------------------------------------------------
def test_netVisual_heatmap_count_returns_figure():
    adata = _make_adata_with_net()
    fig = netVisual_heatmap(adata, measure="count")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_netVisual_heatmap_weight_returns_figure():
    adata = _make_adata_with_net()
    fig = netVisual_heatmap(adata, measure="weight")
    assert isinstance(fig, Figure)


def test_netVisual_heatmap_signaling_branch():
    adata = _make_adata_with_net()
    fig = netVisual_heatmap(adata, measure=None, signaling="PW0", slot_name="netP")
    assert isinstance(fig, Figure)


def test_netVisual_heatmap_invalid_measure():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="measure"):
        netVisual_heatmap(adata, measure="bogus")


def test_netVisual_heatmap_remove_diagonal_zeros_diag():
    adata = _make_adata_with_net()
    fig = netVisual_heatmap(adata, measure="count", remove_diagonal=True)
    assert isinstance(fig, Figure)


def test_netVisual_heatmap_remove_isolate():
    adata = _make_adata_with_net()
    # Force a row+col of all zeros for the isolate detection path
    adata.uns["cellchat"]["net"]["count"][2, :] = 0.0
    adata.uns["cellchat"]["net"]["count"][:, 2] = 0.0
    fig = netVisual_heatmap(adata, measure="count", remove_isolate=True)
    assert isinstance(fig, Figure)


def test_netVisual_heatmap_zero_matrix_raises():
    adata = _make_adata_with_net()
    adata.uns["cellchat"]["net"]["count"][:, :] = 0.0
    with pytest.raises(ValueError, match="No significant"):
        netVisual_heatmap(adata, measure="count")


def test_netVisual_heatmap_subset_sources_targets():
    adata = _make_adata_with_net()
    fig = netVisual_heatmap(adata, measure="count",
                            sources_use=["G0"], targets_use=["G1"])
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_heatmap
# ---------------------------------------------------------------------------
def test_signalingRole_heatmap_outgoing_returns_figure():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, "netP", n_pathways=3)
    fig = netAnalysis_signalingRole_heatmap(adata, pattern="outgoing")
    assert isinstance(fig, Figure)


def test_signalingRole_heatmap_incoming_returns_figure():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, "netP", n_pathways=3)
    fig = netAnalysis_signalingRole_heatmap(adata, pattern="incoming")
    assert isinstance(fig, Figure)


def test_signalingRole_heatmap_all_returns_figure():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, "netP", n_pathways=3)
    fig = netAnalysis_signalingRole_heatmap(adata, pattern="all", title="extra")
    assert isinstance(fig, Figure)


def test_signalingRole_heatmap_signaling_subset():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, "netP", n_pathways=4)
    fig = netAnalysis_signalingRole_heatmap(
        adata, pattern="outgoing", signaling=["PW0", "PW2"],
    )
    assert isinstance(fig, Figure)


def test_signalingRole_heatmap_missing_centr_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        netAnalysis_signalingRole_heatmap(adata, pattern="outgoing")


def test_signalingRole_heatmap_invalid_pattern_raises():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, "netP", n_pathways=2)
    with pytest.raises(ValueError, match="pattern"):
        netAnalysis_signalingRole_heatmap(adata, pattern="bogus")


# ---------------------------------------------------------------------------
# netVisual_bubble
# ---------------------------------------------------------------------------
def test_netVisual_bubble_returns_figure():
    adata = _make_adata_with_net()
    fig = netVisual_bubble(
        adata, sources_use=["G0", "G1"], targets_use=["G1", "G2"],
        thresh=0.1,
    )
    assert isinstance(fig, Figure)


def test_netVisual_bubble_signaling_branch():
    adata = _make_adata_with_net()
    fig = netVisual_bubble(
        adata, sources_use=["G0"], targets_use=["G1", "G2"],
        signaling=["PW0"], thresh=0.5,
    )
    assert isinstance(fig, Figure)


def test_netVisual_bubble_no_significant_raises():
    adata = _make_adata_with_net()
    # zero out probabilities so subsetCommunication finds no rows
    adata.uns["cellchat"]["net"]["prob"][:] = 0.0
    adata.uns["cellchat"]["net"]["pval"][:] = 1.0
    with pytest.raises(ValueError, match="No (significant|interactions)"):
        netVisual_bubble(adata, sources_use=["G0"], targets_use=["G1"])


def test_netVisual_bubble_remove_isolate_keeps_significant():
    adata = _make_adata_with_net(seed=2)
    fig = netVisual_bubble(
        adata, sources_use=["G0", "G1"], targets_use=["G2"],
        thresh=0.1, remove_isolate=True,
    )
    assert isinstance(fig, Figure)


def test_netVisual_bubble_return_data_attaches_df():
    adata = _make_adata_with_net()
    fig = netVisual_bubble(
        adata, sources_use=["G0"], targets_use=["G1"],
        thresh=0.1, return_data=True,
    )
    assert hasattr(fig, "_netVisual_bubble_df")


# ---------------------------------------------------------------------------
# doHeatmap
# ---------------------------------------------------------------------------
def test_doHeatmap_default_features_returns_figure():
    adata = _make_adata_with_net(n_cells=12, n_genes=8)
    fig = doHeatmap(adata, group_by="active_ident")
    assert isinstance(fig, Figure)


def test_doHeatmap_subset_features():
    adata = _make_adata_with_net(n_cells=12, n_genes=8)
    fig = doHeatmap(adata, features=["g0", "g3", "g5"], group_by="active_ident")
    assert isinstance(fig, Figure)


def test_doHeatmap_no_features_in_assay_raises():
    adata = _make_adata_with_net(n_cells=12, n_genes=8)
    with pytest.raises(ValueError, match="No requested features"):
        doHeatmap(adata, features=["totally_unknown"], group_by="active_ident")


def test_doHeatmap_no_group_bar():
    adata = _make_adata_with_net(n_cells=12, n_genes=8)
    fig = doHeatmap(adata, group_by="active_ident", group_bar=False)
    assert isinstance(fig, Figure)


def test_doHeatmap_legend_remove():
    adata = _make_adata_with_net(n_cells=12, n_genes=8)
    fig = doHeatmap(adata, group_by="active_ident", legend_remove=True)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_TopicComposition
# ---------------------------------------------------------------------------
def test_TopicComposition_topicOut_returns_figure():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=False, cutoff_prop=0.0,
    )
    assert isinstance(fig, Figure)


def test_TopicComposition_topicIn_returns_figure():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicComposition(
        adata, y_group="topicIn", x_group="active_ident",
        cluster_rows=False, cutoff_prop=0.0,
    )
    assert isinstance(fig, Figure)


def test_TopicComposition_dot_type():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        type_="dot", cutoff_prop=0.0,
    )
    assert isinstance(fig, Figure)


def test_TopicComposition_missing_topic_raises():
    adata = _make_adata_with_net(n_cells=15)
    with pytest.raises(ValueError, match="topic"):
        netVisual_TopicComposition(
            adata, y_group="topicOut", x_group="active_ident",
        )


def _adata_for_topic_clustering() -> ad.AnnData:
    """Build an adata + structured topic matrix with non-trivial row/col clustering.

    Topics A and C are perfectly correlated across cell groups, while topic B
    is distinct, so any reasonable hierarchical clustering pairs A with C
    rather than leaving the rows in input order ``[A, B, C]``.
    """
    adata = _make_adata_with_net(n_cells=9)
    # 4 topics × n_cells. Build per-cell loadings so that group-averaged
    # rows have a clear A~C, B~D structure.
    n_cells = adata.n_obs
    rows: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    grp = np.asarray(adata.obs["active_ident"])
    base_AC = np.where(grp == "G0", 0.9, np.where(grp == "G1", 0.1, 0.05))
    base_BD = np.where(grp == "G0", 0.05, np.where(grp == "G1", 0.1, 0.9))
    rows.append(base_AC + rng.normal(0, 1e-4, n_cells))  # A
    rows.append(base_BD + rng.normal(0, 1e-4, n_cells))  # B
    rows.append(base_AC + rng.normal(0, 1e-4, n_cells))  # C ~ A
    rows.append(base_BD + rng.normal(0, 1e-4, n_cells))  # D ~ B
    cell_emb = pd.DataFrame(
        np.vstack(rows).T,
        index=adata.obs_names,
        columns=["TopicA", "TopicB", "TopicC", "TopicD"],
    )
    sig_emb = pd.DataFrame(
        np.eye(4),
        index=[f"pair{k}" for k in range(4)],
        columns=cell_emb.columns,
    )
    adata.uns["cellchat"]["net"]["topic"] = {
        "outgoing": {"cell": cell_emb.copy(), "signaling": sig_emb.copy()},
        "incoming": {"cell": cell_emb.copy(), "signaling": sig_emb.copy()},
    }
    return adata


def _yticklabels(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.axes[0].get_yticklabels()]


def _xticklabels(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.axes[0].get_xticklabels()]


def test_TopicComposition_cluster_rows_reorders():
    """Gap #57 regression: ``cluster_rows=True`` must reorder rows."""
    adata = _adata_for_topic_clustering()

    fig_no_clust = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=False, cluster_cols=False, cutoff_prop=0.0,
    )
    fig_clust = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=True, cluster_cols=False, cutoff_prop=0.0,
    )

    rows_input = _yticklabels(fig_no_clust)
    rows_clustered = _yticklabels(fig_clust)

    # Input order is the topic columns from the embedding DataFrame.
    assert rows_input == ["TopicA", "TopicB", "TopicC", "TopicD"]
    # Clustering must permute rows so that the A~C pair and B~D pair are
    # adjacent (i.e. not the input order).
    assert rows_clustered != rows_input
    assert sorted(rows_clustered) == sorted(rows_input)
    pos = {n: i for i, n in enumerate(rows_clustered)}
    assert abs(pos["TopicA"] - pos["TopicC"]) == 1
    assert abs(pos["TopicB"] - pos["TopicD"]) == 1


def test_TopicComposition_cluster_cols_reorders():
    """Gap #57: ``cluster_cols=True`` reorders columns by the column dendrogram."""
    adata = _adata_for_topic_clustering()

    fig_no_clust = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=False, cluster_cols=False, cutoff_prop=0.0,
    )
    fig_clust = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=False, cluster_cols=True, cutoff_prop=0.0,
    )

    # The shared-x annotation strip wipes ax_main's x-tick locator, so we
    # compare the rendered heatmap arrays themselves instead of tick labels.
    mat_input = np.asarray(fig_no_clust.axes[0].get_images()[0].get_array())
    mat_clustered = np.asarray(fig_clust.axes[0].get_images()[0].get_array())
    # Same shape, same multiset of columns, but column order must differ.
    assert mat_input.shape == mat_clustered.shape
    sorted_input = np.sort(mat_input, axis=1)
    sorted_clust = np.sort(mat_clustered, axis=1)
    np.testing.assert_allclose(sorted_input, sorted_clust)
    # If ordering changed, at least one column position must hold a
    # different vector after clustering.
    assert not np.allclose(mat_input, mat_clustered)


def test_TopicComposition_cluster_rows_correlation_metric():
    """Gap #57: ``clustering_distance_rows='correlation'`` is honoured."""
    adata = _adata_for_topic_clustering()
    fig = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=True, clustering_distance_rows="correlation",
        cutoff_prop=0.0,
    )
    rows = _yticklabels(fig)
    # A and C are perfectly correlated, as are B and D, so correlation
    # distance must place each correlated pair adjacent.
    pos = {n: i for i, n in enumerate(rows)}
    assert abs(pos["TopicA"] - pos["TopicC"]) == 1
    assert abs(pos["TopicB"] - pos["TopicD"]) == 1


def test_TopicComposition_cluster_rows_false_preserves_order():
    """Gap #57: ``cluster_rows=False`` keeps the input row order untouched."""
    adata = _adata_for_topic_clustering()
    fig = netVisual_TopicComposition(
        adata, y_group="topicOut", x_group="active_ident",
        cluster_rows=False, cluster_cols=False, cutoff_prop=0.0,
    )
    assert _yticklabels(fig) == ["TopicA", "TopicB", "TopicC", "TopicD"]


# ---------------------------------------------------------------------------
# netVisual_TopicSignaling
# ---------------------------------------------------------------------------
def test_TopicSignaling_returns_figure():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicSignaling(
        adata, pattern="incoming", slot_name="net",
        nfeatures=4, nTop=2, feature_cat="LR-pair",
    )
    assert isinstance(fig, Figure)


def test_TopicSignaling_topic_subset():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicSignaling(
        adata, pattern="outgoing", slot_name="net",
        topics=[1, 2], nfeatures=4, nTop=2,
    )
    assert isinstance(fig, Figure)


def test_TopicSignaling_missing_pattern_raises():
    adata = _make_adata_with_net(n_cells=15)
    with pytest.raises(ValueError, match="topic"):
        netVisual_TopicSignaling(adata, pattern="incoming", slot_name="net")


def test_TopicSignaling_invalid_pattern_raises():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    with pytest.raises(ValueError, match="pattern"):
        netVisual_TopicSignaling(adata, pattern="weird", slot_name="net")


def test_TopicSignaling_invalid_feature_cat_raises():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    with pytest.raises(ValueError, match="feature_cat"):
        netVisual_TopicSignaling(
            adata, pattern="incoming", slot_name="net", feature_cat="bogus"
        )


def test_TopicSignaling_balanced_branch():
    adata = _make_adata_with_net(n_cells=15)
    _add_topic_slot(adata)
    fig = netVisual_TopicSignaling(
        adata, pattern="incoming", slot_name="net",
        nfeatures=4, nTop=2, balanced=True,
        col=("blue", "red"),
    )
    assert isinstance(fig, Figure)
