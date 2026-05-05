"""Smoke tests for ``spatialcellchat.viz_basic`` — slice 6a."""

from __future__ import annotations

from pathlib import Path

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

from spatialcellchat.viz_basic import (
    barPlot,
    barplot_internal,
    communicationDistPlot,
    communicationDistPlot2,
    netAnalysis_dot,
    netVisual_barplot,
    plotFeatures,
    plotStatistics_Gi,
    plotStatistics_Lee,
    savePlotWithCow,
    vizDimLoadings,
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
    coords = rng.uniform(0, 100, size=(n_cells, 2))
    adata.obsm["spatial"] = coords

    # Build a tiny prob tensor and net dict
    K = 3
    nLR = 4
    prob_grp = rng.uniform(0.0, 1.0, size=(K, K, nLR))
    pval_grp = rng.uniform(0.0, 0.05, size=(K, K, nLR))
    pair_names = [f"pair{k}" for k in range(nLR)]
    cell_names = ["G0", "G1", "G2"]
    adata.uns["cellchat"] = {
        "net": {
            "prob": prob_grp,
            "pval": pval_grp,
            "count": np.sum(prob_grp > 0, axis=2).astype(float),
            "weight": np.sum(prob_grp, axis=2).astype(float),
            "LR.sig": pair_names,
            "dimnames": {"cells": cell_names, "pairs": pair_names},
            "meta": {"cells": list(obs.index), "pairs": pair_names,
                     "groups": cell_names, "idents": np.asarray(obs["active_ident"])},
        },
        "netP": {
            "prob": prob_grp,
            "pval": pval_grp,
            "pathways": pair_names,
            "dimnames": [cell_names, cell_names, pair_names],
        },
    }
    return adata


def _add_pattern_slot(adata: ad.AnnData) -> None:
    """Populate ``netP['pattern']['outgoing']`` with a synthetic pattern dict."""
    cell_names = ["G0", "G1", "G2"]
    pair_names = ["pair0", "pair1", "pair2", "pair3"]
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
    data = np.array([[1.0, 0.5, 0.0, 0.7], [0.0, 1.0, 0.4, 0.0], [0.3, 0.0, 0.5, 0.6]])
    adata.uns["cellchat"]["netP"]["pattern"] = {
        "outgoing": {
            "data": data,
            "pattern": {"cell": df_W, "signaling": df_H},
            "W": np.array([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]),
            "H": np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.4, 0.3, 0.1]]),
            "rows_kept": np.array([0, 1, 2]),
            "cells_kept": cell_names,
        }
    }


# ---------------------------------------------------------------------------
# barplot_internal
# ---------------------------------------------------------------------------
def test_barplot_internal_returns_axes():
    df = pd.DataFrame({
        "labels": pd.Categorical(["A", "B", "A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "condition": pd.Categorical(["X", "X", "Y", "Y"], categories=["X", "Y"]),
    })
    ax = barplot_internal(df, x="labels", y="value", fill="condition",
                          color_use=["#FF0000", "#0000FF"])
    assert len(ax.patches) >= 4  # 2 fill levels × 2 x levels
    assert ax.get_xticklabels()[0].get_text() in {"A", "B"}
    plt.close("all")


def test_barplot_internal_with_color_dict():
    df = pd.DataFrame({
        "labels": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0],
        "condition": pd.Categorical(["A", "B"], categories=["A", "B"]),
    })
    ax = barplot_internal(df, x="labels", y="value", fill="condition",
                          color_use={"A": "#FF0000", "B": "#0000FF"},
                          x_lab_rot=True)
    assert len(ax.patches) >= 2
    plt.close("all")


def test_barplot_internal_remove_xtick():
    df = pd.DataFrame({
        "labels": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "value": [1.0, 2.0],
        "condition": pd.Categorical(["A", "B"], categories=["A", "B"]),
    })
    ax = barplot_internal(df, x="labels", y="value", fill="condition",
                          remove_xtick=True, show_legend=False,
                          percent_y=True, title_name="My title")
    assert ax.get_title() == "My title"
    plt.close("all")


def test_barplot_internal_stat_add():
    df = pd.DataFrame({
        "labels": pd.Categorical(["A", "B", "A", "B"] * 4, categories=["A", "B"]),
        "value": list(np.arange(16, dtype=float)),
        "condition": pd.Categorical(["X"] * 8 + ["Y"] * 8, categories=["X", "Y"]),
    })
    ax = barplot_internal(df, x="labels", y="value", fill="condition",
                          stat_add=True, stat_method="wilcox.test")
    # one annotation should have been added
    assert any(isinstance(c, matplotlib.text.Annotation) for c in ax.get_children())
    plt.close("all")


# ---------------------------------------------------------------------------
# barPlot
# ---------------------------------------------------------------------------
def test_barPlot_smoke_truncatedMean():
    adata = _make_adata_with_net()
    fig = barPlot(adata, features=["g0", "g1"], method="truncatedMean", trim=0.1)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_barPlot_method_triMean_and_median():
    adata = _make_adata_with_net()
    fig1 = barPlot(adata, features=["g0"], method="triMean")
    fig2 = barPlot(adata, features=["g0"], method="median")
    assert isinstance(fig1, Figure) and isinstance(fig2, Figure)
    plt.close("all")


def test_barPlot_with_split_by():
    adata = _make_adata_with_net()
    adata.obs["condition"] = pd.Categorical(
        ["A", "B"] * (adata.n_obs // 2) + ["A"] * (adata.n_obs % 2),
        categories=["A", "B"],
    )
    fig = barPlot(adata, features=["g0", "g1"], split_by="condition", ncol=2)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_barPlot_invalid_method_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="method"):
        barPlot(adata, features=["g0"], method="badmethod")  # type: ignore[arg-type]


def test_barPlot_unknown_groupby_raises():
    adata = _make_adata_with_net()
    del adata.obs["active_ident"]
    with pytest.raises(ValueError, match="group_by"):
        barPlot(adata, features=["g0"])


# ---------------------------------------------------------------------------
# netVisual_barplot
# ---------------------------------------------------------------------------
def test_netVisual_barplot_count_default():
    adata = _make_adata_with_net()
    fig = netVisual_barplot(adata, measure="count")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_barplot_weight_with_subsets():
    adata = _make_adata_with_net()
    fig = netVisual_barplot(adata, measure="weight",
                            sources_use=["G0"], targets_use=[1, 2],
                            color_use=["#FF0000", "#00FF00", "#0000FF"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_barplot_signaling():
    adata = _make_adata_with_net()
    fig = netVisual_barplot(adata, signaling="pair0", title_name="custom",
                            x_lab_rot=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_barplot_invert_source():
    adata = _make_adata_with_net()
    fig = netVisual_barplot(adata, sources_use=["G0"], invert_source=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_barplot_no_net_raises():
    adata = ad.AnnData(X=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="net"):
        netVisual_barplot(adata)


# ---------------------------------------------------------------------------
# netAnalysis_dot
# ---------------------------------------------------------------------------
def test_netAnalysis_dot_runs_outgoing():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    fig = netAnalysis_dot(adata, pattern="outgoing")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netAnalysis_dot_pathway_show_and_group_show():
    adata = _make_adata_with_net()
    _add_pattern_slot(adata)
    fig = netAnalysis_dot(adata, pattern="outgoing",
                          pathway_show=["pair0", "pair1"], group_show=["G0", "G1"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netAnalysis_dot_invalid_pattern():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="pattern"):
        netAnalysis_dot(adata, pattern="bogus")  # type: ignore[arg-type]


def test_netAnalysis_dot_no_pattern_raises():
    adata = _make_adata_with_net()
    with pytest.raises(ValueError, match="identifyCommunicationPatterns"):
        netAnalysis_dot(adata, pattern="outgoing")


# ---------------------------------------------------------------------------
# vizDimLoadings
# ---------------------------------------------------------------------------
def test_vizDimLoadings_with_varm_pca():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"f{i}" for i in range(15)])
    X = rng.uniform(0.0, 1.0, size=(10, 15))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.varm["PCs"] = rng.normal(size=(15, 5))
    fig = vizDimLoadings(adata, dims=[1, 2], nfeatures=10, nTop=3,
                         feature_cat="LR-pair", reduction="pca")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_vizDimLoadings_with_uns_loadings_fallback():
    rng = np.random.default_rng(1)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(8)])
    var = pd.DataFrame(index=[f"f{i}" for i in range(10)])
    X = rng.uniform(0.0, 1.0, size=(8, 10))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["cellchat"] = {
        "pca": {"loadings": rng.normal(size=(10, 4)),
                "feature_names": [f"f{i}" for i in range(10)]}
    }
    fig = vizDimLoadings(adata, dims=[1, 2, 3], nTop=2, ncol=3)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_vizDimLoadings_missing_raises():
    rng = np.random.default_rng(2)
    adata = ad.AnnData(X=rng.uniform(0.0, 1.0, size=(5, 6)))
    with pytest.raises(ValueError, match="loadings"):
        vizDimLoadings(adata, dims=[1])


def test_vizDimLoadings_invalid_feature_cat_raises():
    adata = ad.AnnData(X=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="feature_cat"):
        vizDimLoadings(adata, dims=[1], feature_cat="bogus")  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:Tight layout not applied:UserWarning")
def test_vizDimLoadings_projected_uses_separate_varm_slot():
    """R parity: ``projected=True`` reads ``varm[f"{key}_projected"]``."""
    rng = np.random.default_rng(11)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(8)])
    var = pd.DataFrame(index=[f"f{i}" for i in range(12)])
    X = rng.uniform(0.0, 1.0, size=(8, 12))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.varm["PCs"] = rng.normal(size=(12, 4))
    adata.varm["PCs_projected"] = rng.normal(size=(12, 4)) * 10.0  # distinct
    # Without projected — should consume the un-projected slot.
    fig1 = vizDimLoadings(adata, dims=[1], nfeatures=4, nTop=2,
                          projected=False)
    plt.close(fig1)
    # With projected — should consume the projected slot (different scale).
    fig2 = vizDimLoadings(adata, dims=[1], nfeatures=4, nTop=2,
                          projected=True)
    plt.close(fig2)
    # When projected slot is missing, raise.
    del adata.varm["PCs_projected"]
    with pytest.raises(ValueError, match="projected"):
        vizDimLoadings(adata, dims=[1], projected=True)


@pytest.mark.filterwarnings("ignore:Tight layout not applied:UserWarning")
def test_vizDimLoadings_balanced_picks_pos_and_neg():
    """R parity: ``balanced=True`` returns top-k positive AND top-k negative
    features per PC (Seurat::TopFeatures convention)."""
    obs = pd.DataFrame(index=[f"c{i}" for i in range(5)])
    var = pd.DataFrame(index=[f"f{i}" for i in range(8)])
    X = np.zeros((5, 8))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    # Crafted loadings: f0..f3 strongly positive, f4..f7 strongly negative.
    adata.varm["PCs"] = np.array([
        [4.0], [3.0], [2.0], [1.0],
        [-1.0], [-2.0], [-3.0], [-4.0],
    ])
    fig = vizDimLoadings(adata, dims=[1], nfeatures=2, nTop=4,
                         balanced=True, feature_cat="LR-pair")
    # Inspect the rendered annotations to confirm both ends are present.
    # Capture ax annotations: balanced=True with nfeatures=2 picks
    # [f0, f1] (top +) ∪ [f7, f6] (top -) → 4 unique features rendered.
    ax = fig.axes[0]
    annotated = sorted(t.get_text() for t in ax.texts)
    assert "f0" in annotated and "f1" in annotated
    assert "f6" in annotated and "f7" in annotated
    plt.close(fig)


# ---------------------------------------------------------------------------
# plotFeatures
# ---------------------------------------------------------------------------
def test_plotFeatures_normal_no_group():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    }, index=[f"c{i}" for i in range(n)])
    fig = plotFeatures(df, method="normal", color_heatmap="Blues",
                       point_size=2, plot_title="A title")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotFeatures_show_group_two_panels():
    rng = np.random.default_rng(1)
    n = 25
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    }, index=[f"c{i}" for i in range(n)])
    gb = pd.Categorical([f"G{i % 3}" for i in range(n)], categories=["G0", "G1", "G2"])
    fig = plotFeatures(df, group_by=gb, show_group=True, plot_ncol=2)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plotFeatures_cells_highlight_str():
    rng = np.random.default_rng(2)
    n = 20
    idx = [f"c{i}" for i in range(n)]
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    }, index=idx)
    fig = plotFeatures(df, cells_highlight=["c1", "c3", "c5"], method="normal")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotFeatures_color_heatmap_RedsBlues():
    rng = np.random.default_rng(3)
    n = 20
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.normal(0, 1, size=n),
    }, index=[f"c{i}" for i in range(n)])
    fig = plotFeatures(df, color_heatmap="RedsBlues", direction=-1)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotFeatures_color_heatmap_list_input():
    rng = np.random.default_rng(4)
    n = 15
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    }, index=[f"c{i}" for i in range(n)])
    fig = plotFeatures(df, color_heatmap=["#FF0000", "#FFFF00", "#0000FF"],
                       n_colors=3)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotFeatures_invalid_method_raises():
    df = pd.DataFrame({"x": [1.0], "y": [2.0], "score": [0.5]})
    with pytest.raises(ValueError, match="method"):
        plotFeatures(df, method="badbad")  # type: ignore[arg-type]


def test_plotFeatures_show_group_requires_group_by():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "score": [0.5, 0.7]})
    with pytest.raises(ValueError, match="group_by"):
        plotFeatures(df, show_group=True)


# ---------------------------------------------------------------------------
# plotStatistics_Lee
# ---------------------------------------------------------------------------
def test_plotStatistics_Lee_heatmap_with_categorical():
    rng = np.random.default_rng(0)
    n = 20
    coords = rng.uniform(0, 100, size=(n, 2))
    x_lab = pd.Categorical([f"X{i % 3}" for i in range(n)],
                           categories=["X0", "X1", "X2"])
    y_lab = pd.Categorical([f"Y{i % 2}" for i in range(n)],
                           categories=["Y0", "Y1"])
    fig = plotStatistics_Lee(x_lab, y_lab, coords=coords, type="heatmap")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotStatistics_Lee_dot_with_numeric_x():
    rng = np.random.default_rng(1)
    n = 25
    coords = rng.uniform(0, 100, size=(n, 2))
    x_num = rng.uniform(0, 1, size=n)
    y_lab = pd.Categorical([f"Y{i % 3}" for i in range(n)],
                           categories=["Y0", "Y1", "Y2"])
    fig = plotStatistics_Lee(x_num, y_lab, coords=coords)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotStatistics_Lee_data_returns_dataframe():
    rng = np.random.default_rng(2)
    n = 15
    coords = rng.uniform(0, 100, size=(n, 2))
    x_lab = pd.Categorical([f"X{i % 2}" for i in range(n)],
                           categories=["X0", "X1"])
    y_lab = pd.Categorical([f"Y{i % 2}" for i in range(n)],
                           categories=["Y0", "Y1"])
    out = plotStatistics_Lee(x_lab, y_lab, coords=coords, type="data",
                             cutoff=False)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 2)


def test_plotStatistics_Lee_no_coords_or_lw_raises():
    x_lab = pd.Categorical(["A", "B", "A"])
    y_lab = pd.Categorical(["X", "Y", "X"])
    with pytest.raises(ValueError, match="coords or lw"):
        plotStatistics_Lee(x_lab, y_lab)


def test_plotStatistics_Lee_both_coords_and_lw_raises():
    rng = np.random.default_rng(3)
    n = 5
    x_lab = pd.Categorical([f"X{i % 2}" for i in range(n)])
    y_lab = pd.Categorical([f"Y{i % 2}" for i in range(n)])
    coords = rng.uniform(0, 100, size=(n, 2))
    fake_lw = sparse.eye(n)
    with pytest.raises(ValueError, match="don't provide both"):
        plotStatistics_Lee(x_lab, y_lab, coords=coords, lw=fake_lw)


# ---------------------------------------------------------------------------
# plotStatistics_Gi
# ---------------------------------------------------------------------------
def test_plotStatistics_Gi_normal_runs():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    })
    fig = plotStatistics_Gi(df, n=5, method="normal")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotStatistics_Gi_with_pvalue_filter():
    rng = np.random.default_rng(1)
    n = 30
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    })
    fig = plotStatistics_Gi(df, n=5, method="normal", p_value=0.5,
                            plot_title="Hot spots")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotStatistics_Gi_scoring_method():
    rng = np.random.default_rng(2)
    n = 25
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    })
    fig = plotStatistics_Gi(df, n=4, method="scoring")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plotStatistics_Gi_gstar_3d():
    rng = np.random.default_rng(3)
    n = 20
    df = pd.DataFrame({
        "x": rng.uniform(0, 100, size=n),
        "y": rng.uniform(0, 100, size=n),
        "score": rng.uniform(0, 1, size=n),
    })
    fig = plotStatistics_Gi(df, n=4, method="3d", gstar=True, three_dim_z=2.0)
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# savePlotWithCow
# ---------------------------------------------------------------------------
def test_savePlotWithCow_uses_provided_figure(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    out = tmp_path / "out.png"
    savePlotWithCow(str(out), plot=fig, ncol=2, nrow=1, base_height=2.0)
    assert out.exists()
    plt.close(fig)


def test_savePlotWithCow_default_uses_current_figure(tmp_path: Path):
    plt.figure()
    plt.plot([0, 1, 2], [0, 1, 0])
    out = tmp_path / "out2.png"
    savePlotWithCow(str(out))
    assert out.exists()
    plt.close("all")


def test_savePlotWithCow_explicit_width_height(tmp_path: Path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = tmp_path / "out3.png"
    savePlotWithCow(str(out), plot=fig, width=4.0, height=3.0,
                    base_aspect_ratio=2.0)
    assert out.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# communicationDistPlot{,2}
# ---------------------------------------------------------------------------
def _make_adata_with_dist(seed: int = 0) -> ad.AnnData:
    """Build a tiny AnnData with d.spatial / adj.contact / prob.cell."""
    rng = np.random.default_rng(seed)
    n = 6
    X = sparse.csr_matrix(rng.poisson(2.0, size=(n, 4)).astype("float32"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["active_ident"] = pd.Categorical(
        [f"G{i % 2}" for i in range(n)], categories=["G0", "G1"]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n, 2))

    # Sparse distance matrix
    d_dense = np.zeros((n, n), dtype=float)
    rng2 = np.random.default_rng(seed + 1)
    pairs = [(0, 1, 30.0), (1, 2, 50.0), (2, 3, 80.0), (3, 4, 20.0), (4, 5, 60.0)]
    for i, j, d in pairs:
        d_dense[i, j] = d
        d_dense[j, i] = d
    d_spatial = sparse.csc_matrix(d_dense)
    adj_dense = (d_dense > 0).astype(float)
    adj_contact = sparse.csc_matrix(adj_dense)

    # Make a tiny prob.cell tensor (sparse.COO of shape (n, n, 2))
    coords = np.array([[0, 1, 2, 3, 4],
                       [1, 2, 3, 4, 5],
                       [0, 1, 0, 1, 0]])
    data = np.array([0.4, 0.5, 0.3, 0.6, 0.7])
    prob_cell = _sparse.COO(coords=coords, data=data, shape=(n, n, 2))

    adata.uns["cellchat"] = {
        "options": {"parameter": {
            "interaction.range": 100.0,
            "all.diffusible": False,
            "all.contact.dependent": False,
        }},
        "images": {
            "result.computeCellDistance": {
                "d.spatial": d_spatial,
                "adj.contact": adj_contact,
            }
        },
        "net": {"prob.cell": prob_cell},
        "LR": {"LRsig": pd.DataFrame({
            "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
        }, index=["pair0", "pair1"])},
    }
    return adata


def test_communicationDistPlot2_all():
    adata = _make_adata_with_dist()
    fig = communicationDistPlot2(adata, signaling_type="All",
                                 enriched_only=True, density_alpha=0.4)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_communicationDistPlot2_secreted_only():
    adata = _make_adata_with_dist()
    fig = communicationDistPlot2(adata, signaling_type="Secreted")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_communicationDistPlot2_contact_only():
    adata = _make_adata_with_dist()
    fig = communicationDistPlot2(adata, signaling_type="Contact",
                                 enriched_only=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_communicationDistPlot2_no_dist_raises():
    adata = ad.AnnData(X=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="result.computeCellDistance"):
        communicationDistPlot2(adata)


def test_communicationDistPlot_smoke():
    adata = _make_adata_with_dist()
    fig = communicationDistPlot(adata, enriched_only=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_communicationDistPlot_all_diffusible_branch():
    adata = _make_adata_with_dist()
    adata.uns["cellchat"]["options"]["parameter"]["all.diffusible"] = True
    fig = communicationDistPlot(adata)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_communicationDistPlot_all_contact_branch():
    adata = _make_adata_with_dist()
    adata.uns["cellchat"]["options"]["parameter"]["all.contact.dependent"] = True
    fig = communicationDistPlot(adata)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_communicationDistPlot_enriched_only_false():
    adata = _make_adata_with_dist()
    fig = communicationDistPlot(adata, enriched_only=False)
    assert isinstance(fig, Figure)
    plt.close(fig)
