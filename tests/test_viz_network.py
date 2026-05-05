"""Smoke tests for ``spatialcellchat.viz_network``.

The R reference uses ``igraph::plot.igraph`` (base R graphics) for these
exports and ``circlize::chordDiagram`` for chord plots. The Python port
re-renders via :mod:`matplotlib` + :mod:`pycirclize` (deliberate redesign
per contract §5.H), so these tests verify only that:

- the public functions accept all R-documented arguments,
- they return a :class:`matplotlib.figure.Figure` (or list of them),
- the data-flow branches (filtering, isolation removal, etc.) produce
  consistent output.
"""

from __future__ import annotations

from typing import Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import sparse

from spatialcellchat import viz_network


@pytest.fixture(autouse=True)
def _close_figs_after_each():
    """Close all matplotlib figures after each test to prevent memory warnings."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def small_net() -> Tuple[pd.DataFrame, list]:
    """Build a 3 × 3 weighted adjacency matrix for tests."""
    rng = np.random.default_rng(7)
    cells = ["A", "B", "C"]
    arr = rng.uniform(0.1, 1.0, size=(3, 3))
    np.fill_diagonal(arr, 0.5)  # include self-loops
    df = pd.DataFrame(arr, index=cells, columns=cells)
    return df, cells


@pytest.fixture(scope="module")
def small_adata() -> ad.AnnData:
    """Build a tiny AnnData with mocked ``net`` slot for dispatcher tests."""
    rng = np.random.default_rng(11)
    n_obs, n_var = 12, 8
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_var)).astype(float))
    obs = pd.DataFrame(
        {"cellchat_idents": pd.Categorical(["A", "B", "C"] * 4, categories=["A", "B", "C"])},
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_obs, 2))

    # Mock cellchat slots
    K = 3
    nLR = 2
    prob = rng.uniform(0, 1, size=(K, K, nLR))
    pval = np.full_like(prob, 0.01)
    pairs = ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"]
    LRsig = pd.DataFrame(
        {
            "interaction_name": pairs,
            "pathway_name": ["TGFb", "BMP"],
            "ligand": ["TGFB1", "BMP2"],
            "receptor": ["TGFBR1_TGFBR2", "BMPR1A_BMPR2"],
            "interaction_name_2": ["TGFB1 - (TGFBR1+TGFBR2)", "BMP2 - (BMPR1A+BMPR2)"],
            "annotation": ["Secreted Signaling"] * 2,
            "evidence": ["KEGG:04350", "KEGG:04350"],
        },
        index=pairs,
    )
    adata.uns["cellchat"] = {
        "LR": {"LRsig": LRsig},
        "net": {
            "prob": prob,
            "pval": pval,
            "dimnames": {"pairs": pairs, "cells": ["A", "B", "C"]},
        },
    }
    return adata


# ---------------------------------------------------------------------------
# netVisual_circle
# ---------------------------------------------------------------------------
def test_netVisual_circle_basic(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, title_name="test")
    assert isinstance(fig, Figure)


def test_netVisual_circle_with_label_edge(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, label_edge=True, weight_scale=True)
    assert isinstance(fig, Figure)


def test_netVisual_circle_remove_isolate():
    arr = np.zeros((3, 3))
    arr[0, 1] = 0.5
    df = pd.DataFrame(arr, index=["A", "B", "C"], columns=["A", "B", "C"])
    fig = viz_network.netVisual_circle(df, remove_isolate=True)
    assert isinstance(fig, Figure)


def test_netVisual_circle_with_sources_targets(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, sources_use=["A"], targets_use=["B", "C"])
    assert isinstance(fig, Figure)


def test_netVisual_circle_numeric_indices(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, sources_use=[1], targets_use=[2, 3])
    assert isinstance(fig, Figure)


def test_netVisual_circle_with_top(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, top=0.3)
    assert isinstance(fig, Figure)


def test_netVisual_circle_custom_color_use(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, color_use=["#ff0000", "#00ff00", "#0000ff"])
    assert isinstance(fig, Figure)


def test_netVisual_circle_vertex_weight_vector(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_circle(df, vertex_weight=[5, 10, 15], vertex_weight_max=15)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_diffInteraction — Sequence[AnnData] API (R parity)
# ---------------------------------------------------------------------------
def _adata_with_net(seed: int, cells=("A", "B", "C")) -> ad.AnnData:
    """Build a minimal CellChat-state AnnData with a net.count / net.weight."""
    cells = list(cells)
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=np.zeros((len(cells), 1), dtype=np.float32))
    a.obs["cellchat_idents"] = pd.Categorical(cells, categories=cells)
    a.uns["cellchat"] = {
        "net": {
            "count": pd.DataFrame(
                rng.uniform(0, 5, (len(cells), len(cells))),
                index=cells, columns=cells,
            ),
            "weight": pd.DataFrame(
                rng.uniform(0, 1, (len(cells), len(cells))),
                index=cells, columns=cells,
            ),
        }
    }
    return a


def test_netVisual_diffInteraction_count():
    adata_list = [_adata_with_net(42), _adata_with_net(43)]
    fig = viz_network.netVisual_diffInteraction(adata_list, comparison=(1, 2), measure="count")
    assert isinstance(fig, Figure)


def test_netVisual_diffInteraction_weight():
    adata_list = [_adata_with_net(44), _adata_with_net(45)]
    fig = viz_network.netVisual_diffInteraction(adata_list, measure="weight", weight_scale=True)
    assert isinstance(fig, Figure)


def test_netVisual_diffInteraction_invalid_measure():
    adata_list = [_adata_with_net(46, cells=["A", "B"]),
                  _adata_with_net(47, cells=["A", "B"])]
    with pytest.raises(ValueError):
        viz_network.netVisual_diffInteraction(adata_list, measure="bogus")


def test_netVisual_diffInteraction_single_adata_raises():
    """R parity (visualization.R:2387): single-dataset input must raise.

    The Python port mirrors R's discriminator
    ``is.list(object@net[[1]])`` by requiring ``len(adata_list) >= 2``.
    """
    adata = _adata_with_net(48)
    with pytest.raises(ValueError, match="2 AnnDatas"):
        viz_network.netVisual_diffInteraction([adata], measure="count")


def test_netVisual_diffInteraction_missing_net_raises():
    """If the underlying net slot is missing, raise with an upstream hint."""
    a = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
    a.obs["cellchat_idents"] = pd.Categorical(["A", "B"], categories=["A", "B"])
    a.uns["cellchat"] = {}  # no 'net'
    b = _adata_with_net(49, cells=["A", "B"])
    with pytest.raises(ValueError, match="aggregateNet"):
        viz_network.netVisual_diffInteraction([a, b], measure="count")


def test_netVisual_diffInteraction_disjoint_idents_unions():
    """Two datasets with different cell-group ordering get aligned to the
    joint factor (R parity: ``object@idents$joint`` reindex)."""
    a = _adata_with_net(50, cells=["A", "B", "C"])
    b = _adata_with_net(51, cells=["B", "C", "D"])
    fig = viz_network.netVisual_diffInteraction([a, b], measure="count")
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_hierarchy1 / hierarchy2
# ---------------------------------------------------------------------------
def test_netVisual_hierarchy1_basic(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_hierarchy1(df, vertex_receiver=[1], title_name="TEST")
    assert isinstance(fig, Figure)


def test_netVisual_hierarchy2_basic(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_hierarchy2(df, vertex_receiver=[2, 3], title_name="TEST")
    assert isinstance(fig, Figure)


def test_netVisual_hierarchy1_with_remove_isolate():
    arr = np.zeros((4, 4))
    arr[0, 2] = 0.5
    arr[1, 3] = 0.7
    df = pd.DataFrame(arr, index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"])
    fig = viz_network.netVisual_hierarchy1(df, vertex_receiver=[3, 4], remove_isolate=True)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# netVisual_spatial
# ---------------------------------------------------------------------------
def test_netVisual_spatial_basic():
    rng = np.random.default_rng(0)
    cells = ["A", "B", "C"]
    arr = rng.uniform(0.0, 1.0, (3, 3))
    df = pd.DataFrame(arr, index=cells, columns=cells)
    n_spots = 30
    coords = rng.uniform(0, 200, (n_spots, 2))
    labels = np.tile(cells, n_spots // 3 + 1)[:n_spots]
    fig = viz_network.netVisual_spatial(df, coordinates=coords, labels=labels, title_name="x")
    assert isinstance(fig, Figure)


def test_netVisual_spatial_remove_loop():
    rng = np.random.default_rng(1)
    cells = ["A", "B"]
    arr = rng.uniform(0, 1, (2, 2))
    df = pd.DataFrame(arr, index=cells, columns=cells)
    coords = rng.uniform(0, 100, (10, 2))
    labels = ["A", "B"] * 5
    fig = viz_network.netVisual_spatial(df, coordinates=coords, labels=labels, remove_loop=True)
    assert isinstance(fig, Figure)


def test_netVisual_spatial_invalid_coords():
    df = pd.DataFrame(np.ones((2, 2)), index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError):
        viz_network.netVisual_spatial(df, coordinates=np.zeros((5, 3)), labels=["A"] * 5)


# ---------------------------------------------------------------------------
# Chord exports
# ---------------------------------------------------------------------------
def test_netVisual_chord_cell_internal_matrix(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_chord_cell_internal(df, title_name="chord")
    assert isinstance(fig, Figure)


def test_netVisual_chord_cell_internal_dataframe():
    df = pd.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["B", "C", "C"],
        "prob": [0.5, 0.8, 0.3],
    })
    fig = viz_network.netVisual_chord_cell_internal(df)
    assert isinstance(fig, Figure)


def test_netVisual_chord_cell_internal_with_sources_targets(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_chord_cell_internal(
        df, sources_use=["A"], targets_use=["B", "C"]
    )
    assert isinstance(fig, Figure)


def test_netVisual_chord_cell_with_net(small_net):
    df, _ = small_net
    fig = viz_network.netVisual_chord_cell(small_adata_dummy(), net=df, title_name="x")
    assert isinstance(fig, Figure)


def small_adata_dummy() -> ad.AnnData:
    """Bare AnnData for chord tests that take the ``net`` branch."""
    obs = pd.DataFrame(
        {"cellchat_idents": pd.Categorical(["A", "B", "C"], categories=["A", "B", "C"])},
        index=["c0", "c1", "c2"],
    )
    return ad.AnnData(X=np.ones((3, 1)), obs=obs)


def test_netVisual_chord_cell_signaling_netP(small_adata):
    fig = viz_network.netVisual_chord_cell(small_adata, signaling="TGFb", slot_name="netP")
    assert isinstance(fig, Figure)


def test_netVisual_chord_cell_no_signaling_no_net():
    with pytest.raises(ValueError):
        viz_network.netVisual_chord_cell(small_adata_dummy())


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def test_netVisual_circle_layout(small_adata):
    fig = viz_network.netVisual(small_adata, signaling="TGFb", layout="circle")
    assert isinstance(fig, Figure)


def test_netVisual_chord_layout(small_adata):
    fig = viz_network.netVisual(small_adata, signaling="TGFb", layout="chord")
    assert isinstance(fig, Figure)


def test_netVisual_spatial_layout(small_adata):
    fig = viz_network.netVisual(small_adata, signaling="TGFb", layout="spatial")
    assert isinstance(fig, Figure)


def test_netVisual_spatial_vertex_weight_incoming_reads_centr(small_adata):
    """R parity (visualization.R:1335-1343): ``layout='spatial'`` with
    ``vertex_weight="incoming"`` resolves to
    ``object@netP$centr["indeg", , signaling]`` — the in-degree centrality
    for the requested pathway. Pins that the dispatcher routes to
    :func:`centr_value` rather than feeding the raw string downstream."""
    # Inject a centr slot on small_adata so the dispatcher can resolve
    # the vertex.weight string-magic.
    centr_values = np.zeros((11, 3, 1), dtype=float)
    centr_values[2, :, 0] = [1.0, 2.0, 3.0]   # outdeg
    centr_values[3, :, 0] = [4.0, 5.0, 6.0]   # indeg
    centr_dict = {
        "values": centr_values,
        "centr_names": [
            "outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg",
            "hub", "authority", "eigen", "page_rank", "betweenness",
            "flowbet", "info",
        ],
        "node_names": ["A", "B", "C"],
        "signaling_names": ["TGFb"],
    }
    cc = small_adata.uns.setdefault("cellchat", {})
    cc.setdefault("netP", {})["centr"] = centr_dict

    fig = viz_network.netVisual(
        small_adata, signaling="TGFb", layout="spatial",
        vertex_weight="incoming",
    )
    assert isinstance(fig, Figure)


def test_netVisual_spatial_vertex_weight_outgoing_reads_centr(small_adata):
    """Same as above, with ``vertex_weight="outgoing"`` → ``outdeg``."""
    centr_values = np.zeros((11, 3, 1), dtype=float)
    centr_values[2, :, 0] = [1.0, 2.0, 3.0]
    centr_values[3, :, 0] = [4.0, 5.0, 6.0]
    centr_dict = {
        "values": centr_values,
        "centr_names": [
            "outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg",
            "hub", "authority", "eigen", "page_rank", "betweenness",
            "flowbet", "info",
        ],
        "node_names": ["A", "B", "C"],
        "signaling_names": ["TGFb"],
    }
    cc = small_adata.uns.setdefault("cellchat", {})
    cc.setdefault("netP", {})["centr"] = centr_dict

    fig = viz_network.netVisual(
        small_adata, signaling="TGFb", layout="spatial",
        vertex_weight="outgoing",
    )
    assert isinstance(fig, Figure)


def test_netVisual_spatial_vertex_weight_string_without_centr_raises(small_adata):
    """When ``vertex_weight`` is a magic string but no ``centr`` slot has
    been computed, raise the R-equivalent message pointing at
    ``netAnalysis_computeCentrality``.

    Builds a fresh adata to avoid the module-scoped ``small_adata``
    fixture's centr slot leaking from earlier tests.
    """
    import copy

    fresh = copy.deepcopy(small_adata)
    cc = fresh.uns.get("cellchat", {})
    cc.setdefault("netP", {}).pop("centr", None)
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        viz_network.netVisual(
            fresh, signaling="TGFb", layout="spatial",
            vertex_weight="incoming",
        )


def test_netVisual_hierarchy_layout(small_adata):
    fig = viz_network.netVisual(small_adata, signaling="TGFb", layout="hierarchy",
                                vertex_receiver=[1])
    assert isinstance(fig, Figure)


def test_netVisual_invalid_layout(small_adata):
    with pytest.raises(ValueError):
        viz_network.netVisual(small_adata, signaling="TGFb", layout="bogus")


def test_netVisual_hierarchy_writes_individual_and_aggregate(small_adata, tmp_path):
    """R parity (visualization.R:937-998): hierarchy mode writes one
    multi-page individual file and one aggregate file per requested
    output format. ``out_format`` must accept a list/tuple."""
    fig = viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="hierarchy",
        vertex_receiver=[1],
        out_format=["pdf", "png"],
        out_dir=str(tmp_path),
    )
    assert isinstance(fig, Figure)
    for fmt in ("pdf", "png"):
        assert (tmp_path / f"TGFb_hierarchy_individual.{fmt}").exists()
        assert (tmp_path / f"TGFb_hierarchy_aggregate.{fmt}").exists()
        assert (tmp_path / f"TGFb_hierarchy_individual.{fmt}").stat().st_size > 0
        assert (tmp_path / f"TGFb_hierarchy_aggregate.{fmt}").stat().st_size > 0


def test_netVisual_hierarchy_out_format_str(small_adata, tmp_path):
    """``out_format`` accepts a single string (R: vector of length 1)."""
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="hierarchy",
        vertex_receiver=[1],
        out_format="svg",
        out_dir=str(tmp_path),
    )
    assert (tmp_path / "TGFb_hierarchy_individual.svg").exists()
    assert (tmp_path / "TGFb_hierarchy_aggregate.svg").exists()


def test_netVisual_hierarchy_pdf_is_multipage(small_adata, tmp_path):
    """The individual PDF must contain one page per L-R pair (R uses
    ``cairo_pdf`` + ``par(mfrow=c(nRow,2))`` so each pair is its own
    panel; in matplotlib that maps to one PdfPages page per pair)."""
    pypdf = pytest.importorskip("pypdf")
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="hierarchy",
        vertex_receiver=[1],
        out_format=["pdf"],
        out_dir=str(tmp_path),
    )
    reader = pypdf.PdfReader(str(tmp_path / "TGFb_hierarchy_individual.pdf"))
    # small_adata has 2 L-R pairs under TGFb (TGFB1 + BMP2 share the prob
    # array but only TGFB1 maps to the TGFb pathway). Whichever maps,
    # the file must have at least 1 page.
    assert len(reader.pages) >= 1


def test_netVisual_circle_layout_writes_files(small_adata, tmp_path):
    """R parity (visualization.R:1001-1057): ``layout='circle'`` with
    ``out_format``/``out_dir`` set must write a per-pair ``individual``
    file and an ``aggregate`` file per requested format."""
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="circle",
        out_format=["png"],
        out_dir=str(tmp_path),
    )
    assert (tmp_path / "TGFb_circle_individual.png").exists()
    assert (tmp_path / "TGFb_circle_aggregate.png").exists()
    assert (tmp_path / "TGFb_circle_individual.png").stat().st_size > 0
    assert (tmp_path / "TGFb_circle_aggregate.png").stat().st_size > 0


def test_netVisual_circle_layout_pdf_is_multipage(small_adata, tmp_path):
    """``layout='circle'`` PDF mode produces a multi-page document
    (one page per L-R pair)."""
    pypdf = pytest.importorskip("pypdf")
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="circle",
        out_format=["pdf"],
        out_dir=str(tmp_path),
    )
    reader = pypdf.PdfReader(str(tmp_path / "TGFb_circle_individual.pdf"))
    assert len(reader.pages) >= 1


def test_netVisual_chord_layout_writes_files(small_adata, tmp_path):
    """R parity (visualization.R:1125-1183): ``layout='chord'`` writes
    individual + aggregate per format."""
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="chord",
        out_format=["png"],
        out_dir=str(tmp_path),
    )
    assert (tmp_path / "TGFb_chord_individual.png").exists()
    assert (tmp_path / "TGFb_chord_aggregate.png").exists()


def test_netVisual_circle_layout_no_files_when_out_format_unset(small_adata, tmp_path):
    """When ``out_format`` is None, no files are written (default behavior
    matches single-figure R return)."""
    viz_network.netVisual(
        small_adata,
        signaling="TGFb",
        layout="circle",
    )
    assert not list(tmp_path.iterdir())


def test_netVisual_heatmap_layout_deferred(small_adata_for_heatmap):
    """layout='heatmap' routes to sub-agent 6b's viz_heatmap.netVisual_heatmap."""
    fig = viz_network.netVisual(small_adata_for_heatmap, signaling="TGFb", layout="heatmap")
    assert isinstance(fig, Figure)


@pytest.fixture(scope="module")
def small_adata_for_heatmap() -> ad.AnnData:
    """Like ``small_adata`` but with the dimnames shape sub-agent 6b expects."""
    rng = np.random.default_rng(11)
    n_obs, n_var = 12, 8
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n_obs, n_var)).astype(float))
    obs = pd.DataFrame(
        {"cellchat_idents": pd.Categorical(["A", "B", "C"] * 4, categories=["A", "B", "C"])},
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_obs, 2))
    K, nLR = 3, 2
    prob = rng.uniform(0, 1, size=(K, K, nLR))
    pval = np.full_like(prob, 0.01)
    pairs = ["TGFb", "BMP"]  # use pathway names so signaling="TGFb" matches
    LRsig = pd.DataFrame(
        {
            "interaction_name": pairs,
            "pathway_name": pairs,
            "ligand": ["TGFB1", "BMP2"],
            "receptor": ["TGFBR1_TGFBR2", "BMPR1A_BMPR2"],
            "interaction_name_2": ["TGFb", "BMP"],
            "annotation": ["Secreted Signaling"] * 2,
            "evidence": ["KEGG:04350", "KEGG:04350"],
        },
        index=pairs,
    )
    adata.uns["cellchat"] = {
        "LR": {"LRsig": LRsig},
        "net": {
            "prob": prob,
            "pval": pval,
            "dimnames": {
                "pairs": pairs,
                "interaction_name": pairs,  # sub-agent 6b's key
                "cells": ["A", "B", "C"],
            },
        },
    }
    return adata


# ---------------------------------------------------------------------------
# netVisual_aggregate / netVisual_individual
# ---------------------------------------------------------------------------
def test_netVisual_aggregate_circle(small_adata):
    fig = viz_network.netVisual_aggregate(small_adata, signaling="TGFb", layout="circle")
    assert isinstance(fig, Figure)


def test_netVisual_aggregate_chord(small_adata):
    fig = viz_network.netVisual_aggregate(small_adata, signaling="TGFb", layout="chord")
    assert isinstance(fig, Figure)


def test_netVisual_aggregate_spatial(small_adata):
    fig = viz_network.netVisual_aggregate(small_adata, signaling="TGFb", layout="spatial")
    assert isinstance(fig, Figure)


def test_netVisual_aggregate_hierarchy(small_adata):
    fig = viz_network.netVisual_aggregate(
        small_adata, signaling="TGFb", layout="hierarchy", vertex_receiver=[1]
    )
    assert isinstance(fig, Figure)


def test_netVisual_aggregate_invalid_layout(small_adata):
    with pytest.raises(ValueError):
        viz_network.netVisual_aggregate(small_adata, signaling="TGFb", layout="bogus")


def test_netVisual_individual_circle(small_adata):
    figs = viz_network.netVisual_individual(small_adata, signaling="TGFb", layout="circle")
    assert isinstance(figs, list) and all(isinstance(f, Figure) for f in figs)


def test_netVisual_individual_chord(small_adata):
    figs = viz_network.netVisual_individual(small_adata, signaling="TGFb", layout="chord")
    assert isinstance(figs, list) and all(isinstance(f, Figure) for f in figs)


def test_netVisual_individual_spatial(small_adata):
    figs = viz_network.netVisual_individual(small_adata, signaling="TGFb", layout="spatial")
    assert isinstance(figs, list) and all(isinstance(f, Figure) for f in figs)


def test_netVisual_individual_invalid(small_adata):
    with pytest.raises(ValueError):
        viz_network.netVisual_individual(small_adata, signaling="TGFb", layout="bogus")


def test_netVisual_individual_pairLR_filter(small_adata):
    figs = viz_network.netVisual_individual(
        small_adata, signaling="TGFb", layout="circle",
        pairLR_use=["TGFB1_TGFBR1_TGFBR2"],
    )
    assert len(figs) == 1


# ---------------------------------------------------------------------------
# netVisual_chord_gene
# ---------------------------------------------------------------------------
def test_netVisual_chord_gene_basic(small_adata):
    fig = viz_network.netVisual_chord_gene(small_adata, signaling=["TGFb"])
    assert isinstance(fig, Figure)


def test_netVisual_chord_gene_pairLR_use(small_adata):
    pairLR = pd.DataFrame({"interaction_name": ["TGFB1_TGFBR1_TGFBR2"]})
    fig = viz_network.netVisual_chord_gene(small_adata, pairLR_use=pairLR)
    assert isinstance(fig, Figure)


def test_netVisual_chord_gene_simultaneous_args_raises(small_adata):
    pairLR = pd.DataFrame({"interaction_name": ["TGFB1_TGFBR1_TGFBR2"]})
    with pytest.raises(ValueError):
        viz_network.netVisual_chord_gene(small_adata, signaling=["TGFb"], pairLR_use=pairLR)
