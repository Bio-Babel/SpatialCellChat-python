"""Smoke tests for ``spatialcellchat.viz_plotly``.

These verify each public Plotly export returns a
:class:`plotly.graph_objects.Figure` and that data-flow branches
(highlight, sources/targets, group_by) are reached without error.
"""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy import sparse

from spatialcellchat import viz_plotly


@pytest.fixture(scope="module")
def small_spatial_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    n_obs, n_var = 12, 6
    X = sparse.csr_matrix(rng.poisson(2, size=(n_obs, n_var)).astype(float))
    obs = pd.DataFrame(
        {
            "cellchat_idents": pd.Categorical(["A", "B", "C"] * 4, categories=["A", "B", "C"]),
            "celltype2": pd.Categorical(["X", "Y"] * 6, categories=["X", "Y"]),
        },
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=["TGFB1", "TGFBR1", "TGFBR2", "BMP2", "BMPR1A", "BMPR2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_obs, 2))

    # cellchat slot — only LR + net needed for LR-pair plot
    K = 3
    nLR = 2
    prob = rng.uniform(0, 1, size=(K, K, nLR))
    pval = np.full_like(prob, 0.01)
    pairs = ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"]
    LRsig = pd.DataFrame(
        {
            "interaction_name": pairs,
            "ligand": ["TGFB1", "BMP2"],
            "receptor": ["TGFBR1", "BMPR1A"],
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
# plotly_spatialDimPlot
# ---------------------------------------------------------------------------
def test_get_coords_swapped_no_longer_swaps(small_spatial_adata):
    """Audit M-VIZ-6: ``_get_coords_swapped`` must mirror
    :func:`viz_spatial._xy_from_adata`'s no-swap convention. The function
    name is historical; in the Python port it is a labeled pass-through.
    """
    coords = small_spatial_adata.obsm["spatial"]
    df = viz_plotly._get_coords_swapped(small_spatial_adata)
    assert (df["x_cent"].to_numpy() == coords[:, 0]).all()
    assert (df["y_cent"].to_numpy() == coords[:, 1]).all()


def test_plotly_spatialDimPlot_2d(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, method="2d")
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_3d(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, method="3d")
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_group_by(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, group_by="celltype2")
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_idents_use(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, idents_use=["A"])
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_sources_targets(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata,
                                            sources_use=["A"], targets_use=["B"])
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_cells_highlight(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(
        small_spatial_adata, cells_highlight=["cell0", "cell1"]
    )
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_group_highlight(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, group_highlight=["A"])
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_both_highlight_raises(small_spatial_adata):
    with pytest.raises(ValueError):
        viz_plotly.plotly_spatialDimPlot(
            small_spatial_adata, group_highlight=["A"], cells_highlight=["cell0"]
        )


def test_plotly_spatialDimPlot_invalid_method(small_spatial_adata):
    with pytest.raises(ValueError):
        viz_plotly.plotly_spatialDimPlot(small_spatial_adata, method="bogus")


def test_plotly_spatialDimPlot_with_color_use(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(
        small_spatial_adata, color_use=["#ff0000", "#00ff00", "#0000ff"]
    )
    assert isinstance(fig, go.Figure)


def test_plotly_spatialDimPlot_numeric_idents(small_spatial_adata):
    fig = viz_plotly.plotly_spatialDimPlot(small_spatial_adata, idents_use=[1])
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plotly_spatialLRpairPlot
# ---------------------------------------------------------------------------
def test_plotly_spatialLRpairPlot_features(small_spatial_adata):
    figs = viz_plotly.plotly_spatialLRpairPlot(small_spatial_adata, features=["TGFB1"])
    assert isinstance(figs, list)
    assert all(isinstance(f, go.Figure) for f in figs)


def test_plotly_spatialLRpairPlot_features_two(small_spatial_adata):
    figs = viz_plotly.plotly_spatialLRpairPlot(
        small_spatial_adata, features=["TGFB1", "TGFBR1"]
    )
    assert len(figs) == 2


def test_plotly_spatialLRpairPlot_features_three_raises(small_spatial_adata):
    with pytest.raises(ValueError):
        viz_plotly.plotly_spatialLRpairPlot(
            small_spatial_adata, features=["TGFB1", "TGFBR1", "BMP2"]
        )


def test_plotly_spatialLRpairPlot_pairLR_str(small_spatial_adata):
    figs = viz_plotly.plotly_spatialLRpairPlot(
        small_spatial_adata, pairLR_use="TGFB1_TGFBR1_TGFBR2"
    )
    assert isinstance(figs, list)
    assert len(figs) >= 1


def test_plotly_spatialLRpairPlot_pairLR_df(small_spatial_adata):
    pairLR = pd.DataFrame({"interaction_name": ["TGFB1_TGFBR1_TGFBR2"]})
    figs = viz_plotly.plotly_spatialLRpairPlot(small_spatial_adata, pairLR_use=pairLR)
    assert isinstance(figs, list)


def test_plotly_spatialLRpairPlot_features_and_pairLR_raises(small_spatial_adata):
    with pytest.raises(ValueError):
        viz_plotly.plotly_spatialLRpairPlot(
            small_spatial_adata, features=["TGFB1"], pairLR_use="TGFB1_TGFBR1_TGFBR2"
        )


def test_plotly_spatialLRpairPlot_cells_highlight(small_spatial_adata):
    figs = viz_plotly.plotly_spatialLRpairPlot(
        small_spatial_adata, features=["TGFB1"], cells_highlight=["cell0", "cell1"]
    )
    assert isinstance(figs, list)


def test_plotly_spatialLRpairPlot_do_plot_false(small_spatial_adata):
    n = viz_plotly.plotly_spatialLRpairPlot(
        small_spatial_adata, features=["TGFB1"], do_plot=False
    )
    assert isinstance(n, int)
    assert n == 1


def test_plotly_spatialLRpairPlot_invalid_feature_raises(small_spatial_adata):
    with pytest.raises(ValueError):
        viz_plotly.plotly_spatialLRpairPlot(small_spatial_adata, features=["UNKNOWN_GENE"])
