"""Smoke tests for ``spatialcellchat.viz_embedding`` — slice 6b."""

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

from spatialcellchat.viz_embedding import (
    netAnalysis_contribution,
    netAnalysis_diff_signalingRole_scatter,
    netAnalysis_river,
    netAnalysis_signalingChanges_scatter,
    netAnalysis_signalingRole_network,
    netAnalysis_signalingRole_scatter,
    netVisual_embedding,
    netVisual_embeddingPairwise,
    netVisual_embeddingPairwiseZoomIn,
    netVisual_embeddingZoomIn,
)
from spatialcellchat._text_repel import repel_text


# ---------------------------------------------------------------------------
# Shared fixture builders
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
        "pathway_name": [f"PW_{k}" for k in range(nLR)],
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
            "pathways": [f"PW_{k}" for k in range(nLR)],
            "dimnames": [cell_names, cell_names, [f"PW_{k}" for k in range(nLR)]],
        },
    }
    return adata


def _add_centr_slot(adata: ad.AnnData, slot_name: str = "netP", n_pathways: int = 4) -> None:
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
    pathway_names = [f"PW_{i}" for i in range(n_pathways)]
    adata.uns["cellchat"][slot_name]["centr"] = {
        "values": centr,
        "centr_names": centr_names,
        "node_names": ["G0", "G1", "G2"],
        "signaling_names": pathway_names,
    }


def _add_similarity_slot(adata: ad.AnnData, n_pathways: int = 4) -> None:
    """Populate netP['similarity']['functional'] with synthetic dr/group/matrix."""
    rng = np.random.default_rng(13)
    pathway_names = [f"PW_{i}" for i in range(n_pathways)]
    Y = pd.DataFrame(
        rng.uniform(-1.0, 1.0, size=(n_pathways, 2)),
        index=pathway_names,
        columns=["UMAP1", "UMAP2"],
    )
    groups = pd.Series([1, 1, 2, 2][:n_pathways], index=pathway_names)
    sim_mat = pd.DataFrame(
        np.eye(n_pathways) + 0.2 * rng.uniform(0.0, 1.0, size=(n_pathways, n_pathways)),
        index=pathway_names, columns=pathway_names,
    )
    adata.uns["cellchat"]["netP"]["similarity"] = {
        "functional": {
            "dr": {"single": Y, "1": Y.copy()},
            "group": {"single": groups, "1": groups.copy()},
            "matrix": {"single": sim_mat, "1": sim_mat.copy()},
        }
    }


def _add_pattern_slot(adata: ad.AnnData) -> None:
    cell_names = ["G0", "G1", "G2"]
    pair_names = ["PW_0", "PW_1", "PW_2", "PW_3"]
    n_pat = 2
    df_W = pd.DataFrame([
        {"CellGroup": cn, "Pattern": f"Pattern {p + 1}",
         "Contribution": 0.5 + 0.1 * (i + p)}
        for i, cn in enumerate(cell_names) for p in range(n_pat)
    ])
    df_H = pd.DataFrame([
        {"Pattern": f"Pattern {p + 1}", "Signaling": pn,
         "Contribution": 0.5 + 0.05 * (j + p)}
        for p in range(n_pat) for j, pn in enumerate(pair_names)
    ])
    adata.uns["cellchat"]["netP"]["pattern"] = {
        "outgoing": {
            "data": np.zeros((3, 4)),
            "pattern": {"cell": df_W, "signaling": df_H},
            "W": np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]),
            "H": np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.4, 0.3, 0.1]]),
            "rows_kept": np.array([0, 1, 2]),
            "cells_kept": cell_names,
            "pattern_labels": [f"Pattern {p + 1}" for p in range(n_pat)],
        },
        "incoming": {
            "data": np.zeros((3, 4)),
            "pattern": {"cell": df_W.copy(), "signaling": df_H.copy()},
            "W": np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]),
            "H": np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.4, 0.3, 0.1]]),
            "rows_kept": np.array([0, 1, 2]),
            "cells_kept": cell_names,
            "pattern_labels": [f"Pattern {p + 1}" for p in range(n_pat)],
        },
    }


# ---------------------------------------------------------------------------
# repel_text helper
# ---------------------------------------------------------------------------
def test_repel_text_returns_text_objects():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    texts = repel_text(ax, [0.0, 1.0, 2.0], [0.0, 1.0, 0.5], ["a", "b", "c"])
    assert len(texts) == 3
    assert all(t.get_text() in {"a", "b", "c"} for t in texts)
    plt.close(fig)


def test_repel_text_empty_returns_empty():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    out = repel_text(ax, [], [], [])
    assert out == []
    plt.close(fig)


# ---------------------------------------------------------------------------
# netVisual_embedding
# ---------------------------------------------------------------------------
def test_netVisual_embedding_returns_figure():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embedding(adata, type_="functional")
    assert isinstance(fig, Figure)


def test_netVisual_embedding_pathway_remove_show():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embedding(
        adata, type_="functional", pathway_remove=["PW_3"], pathway_remove_show=True,
    )
    assert isinstance(fig, Figure)


def test_netVisual_embedding_invalid_type_raises():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="type_"):
        netVisual_embedding(adata, type_="bogus")


def test_netVisual_embedding_pathway_labeled_subset():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embedding(
        adata, type_="functional", pathway_labeled=["PW_0", "PW_2"],
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_embeddingZoomIn
# ---------------------------------------------------------------------------
def test_netVisual_embeddingZoomIn_returns_figure():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embeddingZoomIn(adata, type_="functional", nCol=2)
    assert isinstance(fig, Figure)


def test_netVisual_embeddingZoomIn_no_label():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embeddingZoomIn(
        adata, type_="functional", do_label=False, show_axes=False,
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_embeddingPairwise
# ---------------------------------------------------------------------------
def test_netVisual_embeddingPairwise_returns_figure():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embeddingPairwise(
        adata, type_="functional", comparison=[1],
    )
    assert isinstance(fig, Figure)


def test_netVisual_embeddingPairwise_pathway_remove_show():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embeddingPairwise(
        adata, type_="functional", comparison=[1],
        pathway_remove=["PW_3"], pathway_remove_show=True,
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_embeddingPairwiseZoomIn
# ---------------------------------------------------------------------------
def test_netVisual_embeddingPairwiseZoomIn_returns_figure():
    adata = _make_adata_with_net()
    _add_similarity_slot(adata, n_pathways=4)
    fig = netVisual_embeddingPairwiseZoomIn(
        adata, type_="functional", comparison=[1], nCol=2,
    )
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_scatter
# ---------------------------------------------------------------------------
def test_signalingRole_scatter_returns_figure():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    fig = netAnalysis_signalingRole_scatter(adata)
    assert isinstance(fig, Figure)


def test_signalingRole_scatter_signaling_subset():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    fig = netAnalysis_signalingRole_scatter(adata, signaling=["PW_0", "PW_1"])
    assert isinstance(fig, Figure)


def test_signalingRole_scatter_signaling_unknown_raises():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="significant"):
        netAnalysis_signalingRole_scatter(adata, signaling=["not_a_path"])


def test_signalingRole_scatter_invalid_measure_raises():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="measure"):
        netAnalysis_signalingRole_scatter(adata, x_measure="bogus")


def test_signalingRole_scatter_missing_centr_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        netAnalysis_signalingRole_scatter(adata)


# ---------------------------------------------------------------------------
# netAnalysis_diff_signalingRole_scatter — Sequence[AnnData] API (R parity)
# ---------------------------------------------------------------------------
# R's ``mergeCellChat(list(c1, c2))`` is a no-op in this port: the Python
# equivalent is just passing ``[c1, c2]`` to the comparison function. The
# discriminator (R: ``is.list(object@net[[1]])``) becomes
# ``len(adata_list) >= 2``; single-AnnData input raises eagerly.
def _merged_pair(seed_a: int = 0, seed_b: int = 1):
    a = _make_adata_with_net(seed=seed_a)
    _add_centr_slot(a, n_pathways=4)
    b = _make_adata_with_net(seed=seed_b)
    _add_centr_slot(b, n_pathways=4)
    return a, b


def test_diff_signalingRole_scatter_single_adata_raises():
    """R parity (analysis.R:2779): single-AnnData input raises with the
    discriminator message."""
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="2 AnnDatas"):
        netAnalysis_diff_signalingRole_scatter([adata], comparison=[1, 2])


def test_diff_signalingRole_scatter_two_adatas_returns_figure():
    """R parity (analysis.R:2773): per-cell-group differential scatter from
    two CellChat AnnDatas."""
    a, b = _merged_pair()
    fig = netAnalysis_diff_signalingRole_scatter([a, b], comparison=[1, 2])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_diff_signalingRole_scatter_dataset_names():
    """``dataset_names`` flows into the title (R parity: ``add.names``)."""
    a, b = _merged_pair()
    fig = netAnalysis_diff_signalingRole_scatter(
        [a, b], comparison=[1, 2],
        dataset_names=["control", "treatment"],
    )
    assert "control" in fig.axes[0].get_title()
    assert "treatment" in fig.axes[0].get_title()
    plt.close(fig)


def test_diff_signalingRole_scatter_idents_exclude():
    """``idents_exclude`` drops cell groups from data and palette in lock-step
    (R parity: analysis.R:2850-2853)."""
    a, b = _merged_pair()
    fig = netAnalysis_diff_signalingRole_scatter(
        [a, b], comparison=[1, 2], idents_exclude=["G0"],
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_diff_signalingRole_scatter_missing_centr_raises():
    """If a dataset has no centr slot, raise with an upstream hint."""
    a, b = _merged_pair()
    a.uns["cellchat"]["netP"].pop("centr", None)
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        netAnalysis_diff_signalingRole_scatter([a, b], comparison=[1, 2])


# ---------------------------------------------------------------------------
# netAnalysis_signalingChanges_scatter — Sequence[AnnData] API (R parity)
# ---------------------------------------------------------------------------
def test_signalingChanges_scatter_single_adata_raises():
    """R parity (analysis.R:2940): single-AnnData input raises."""
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="2 AnnDatas"):
        netAnalysis_signalingChanges_scatter([adata], idents_use="G0")


def test_signalingChanges_scatter_two_adatas_returns_figure():
    """R parity (analysis.R:2937): pathway-level differential scatter for
    a single ``idents_use`` cell group."""
    a, b = _merged_pair()
    fig = netAnalysis_signalingChanges_scatter(
        [a, b], idents_use="G0", comparison=[1, 2],
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_signalingChanges_scatter_invalid_idents_use_raises():
    """R parity (analysis.R:2962): ``stop("Please check the input cell group
    names!")`` when ``idents_use`` is not in joint cell.levels."""
    a, b = _merged_pair()
    with pytest.raises(ValueError, match="not in joint cell levels"):
        netAnalysis_signalingChanges_scatter(
            [a, b], idents_use="NoSuchCell", comparison=[1, 2],
        )


def test_signalingChanges_scatter_classification_invariants():
    """Pathways with mat.diff = 0 must be filtered out (R: analysis.R:3014).

    Construct identical centrality across two datasets → all pathways have
    zero diff → the resulting df should be empty (rendered as an empty
    scatter, not crash).
    """
    a, b = _merged_pair()
    # Force B's centr to equal A's so the diff is zero everywhere.
    b.uns["cellchat"]["netP"]["centr"]["values"] = (
        a.uns["cellchat"]["netP"]["centr"]["values"].copy()
    )
    fig = netAnalysis_signalingChanges_scatter(
        [a, b], idents_use="G0", comparison=[1, 2],
    )
    # Empty data still renders a Figure (with 0 scatter points).
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_signalingChanges_scatter_dataset_names_in_title():
    """``dataset_names`` must flow through to the title and to the
    ``specificity`` factor labels (R: analysis.R:3033)."""
    a, b = _merged_pair()
    fig = netAnalysis_signalingChanges_scatter(
        [a, b], idents_use="G0", comparison=[1, 2],
        dataset_names=["pre", "post"],
    )
    assert "pre" in fig.axes[0].get_title()
    assert "post" in fig.axes[0].get_title()
    plt.close(fig)


# ---------------------------------------------------------------------------
# netAnalysis_signalingRole_network
# ---------------------------------------------------------------------------
def test_signalingRole_network_returns_figure():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    fig = netAnalysis_signalingRole_network(adata, signaling=["PW_0"])
    assert isinstance(fig, Figure)


def test_signalingRole_network_multi_signaling():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    fig = netAnalysis_signalingRole_network(adata, signaling=["PW_0", "PW_1"])
    assert isinstance(fig, Figure)


def test_signalingRole_network_invalid_signaling_raises():
    adata = _make_adata_with_net()
    _add_centr_slot(adata, n_pathways=4)
    with pytest.raises(ValueError, match="not in pathways"):
        netAnalysis_signalingRole_network(adata, signaling=["not_a_path"])


def test_signalingRole_network_missing_centr_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        netAnalysis_signalingRole_network(adata, signaling=["PW_0"])


# ---------------------------------------------------------------------------
# netAnalysis_contribution
# ---------------------------------------------------------------------------
def test_netAnalysis_contribution_returns_figure():
    adata = _make_adata_with_net()
    fig = netAnalysis_contribution(adata, signaling="PW_0", thresh=0.1)
    assert isinstance(fig, Figure)


def test_netAnalysis_contribution_signaling_unknown_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="not found"):
        netAnalysis_contribution(adata, signaling="ZZZ")


def test_netAnalysis_contribution_no_significant_raises():
    adata = _make_adata_with_net()
    # Force no significant comm by setting all pvals high
    adata.uns["cellchat"]["net"]["pval"][:] = 0.9
    with pytest.raises(ValueError, match="no significant"):
        netAnalysis_contribution(adata, signaling="PW_0", thresh=0.05)


def test_netAnalysis_contribution_return_data_attaches_df():
    adata = _make_adata_with_net()
    fig = netAnalysis_contribution(
        adata, signaling="PW_0", thresh=0.1, return_data=True,
    )
    assert hasattr(fig, "_netAnalysis_contribution_df")


def test_netAnalysis_contribution_no_vertex_receiver_single_panel():
    """When vertex_receiver is None the figure has a single barh axes."""
    adata = _make_adata_with_net()
    fig = netAnalysis_contribution(adata, signaling="PW_0", thresh=0.1)
    # Single panel layout = exactly one Axes.
    assert len([a for a in fig.axes if a.has_data() or a.get_yticklabels()]) == 1
    assert len(fig.axes) == 1


def test_netAnalysis_contribution_vertex_receiver_three_panel():
    """vertex_receiver triggers 3-panel All / Hierarchy1 / Hierarchy2 layout."""
    adata = _make_adata_with_net()
    fig = netAnalysis_contribution(
        adata, signaling="PW_0", thresh=0.1, vertex_receiver=[0, 1],
    )
    assert isinstance(fig, Figure)
    # Layout = 1 title axis + 3 panel axes.
    assert len(fig.axes) == 4
    titles = sorted(a.get_title() for a in fig.axes if a.get_title())
    assert titles == ["All", "Hierarchy1", "Hierarchy2"]
    # Top-of-figure descriptive title text is present.
    fig_texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("Contribution of each signaling in" in t for t in fig_texts)
    # All three panels share the same x-axis range so they are comparable.
    panel_axes = [a for a in fig.axes if a.get_title() in {"All", "Hierarchy1", "Hierarchy2"}]
    xlims = {a.get_xlim() for a in panel_axes}
    assert len(xlims) == 1


def test_netAnalysis_contribution_vertex_receiver_with_do_group_false():
    """do_group=False + vertex_receiver also yields a 3-panel figure."""
    adata = _make_adata_with_net()
    # Provide prob.cell at the per-cell granularity for do_group=False (R:
    # net$prob.cell branch). Use the same pair-name dimnames so signaling
    # filtering still finds the LR pairs.
    rng = np.random.default_rng(11)
    n_cells = adata.n_obs
    nLR = 4
    prob_cell = rng.uniform(0.0, 1.0, size=(n_cells, n_cells, nLR))
    adata.uns["cellchat"]["net"]["prob.cell"] = prob_cell
    fig = netAnalysis_contribution(
        adata, signaling="PW_0", do_group=False, thresh=0.1,
        vertex_receiver=[0, 1, 2],
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


# ---------------------------------------------------------------------------
# netAnalysis_river
# ---------------------------------------------------------------------------
def test_netAnalysis_river_outgoing_returns_figure():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    fig = netAnalysis_river(adata, pattern="outgoing", cutoff=0.1)
    assert isinstance(fig, Figure)


def test_netAnalysis_river_incoming_returns_figure():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    fig = netAnalysis_river(adata, pattern="incoming", cutoff=0.1)
    assert isinstance(fig, Figure)


def test_netAnalysis_river_signaling_subset():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    fig = netAnalysis_river(
        adata, pattern="outgoing", cutoff=0.1, signaling=["PW_0", "PW_1"],
    )
    assert isinstance(fig, Figure)


def test_netAnalysis_river_signaling_only_no_data2():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    # remove signaling df to test the single-alluvial branch
    adata.uns["cellchat"]["netP"]["pattern"]["outgoing"]["pattern"]["signaling"] = None
    fig = netAnalysis_river(adata, pattern="outgoing", cutoff=0.1)
    assert isinstance(fig, Figure)


def test_netAnalysis_river_invalid_pattern_raises():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    with pytest.raises(ValueError, match="pattern"):
        netAnalysis_river(adata, pattern="bogus")


def test_netAnalysis_river_missing_pattern_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="missing"):
        netAnalysis_river(adata, pattern="outgoing")
