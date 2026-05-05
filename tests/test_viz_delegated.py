"""Smoke tests for ``spatialcellchat.viz_delegated`` — slice 6a."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import sparse

from spatialcellchat.viz_delegated import (
    StackedVlnPlot,
    dotPlot,
    plotGeneExpression,
)


# ---------------------------------------------------------------------------
# Helper: build a tiny AnnData
# ---------------------------------------------------------------------------
def _make_simple_adata(n: int = 20, n_genes: int = 8, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.poisson(2.0, size=(n, n_genes)).astype("float32"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["active_ident"] = pd.Categorical(
        [f"G{i % 3}" for i in range(n)], categories=["G0", "G1", "G2"]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_adata_for_extractEnrichedLR(seed: int = 1) -> ad.AnnData:
    """Build an AnnData with enough cellchat slots for ``extractEnrichedLR``.

    Uses only a small fake DB / LR / net so we don't depend on the real
    ``CellChatDB`` — this is sufficient for the shim's smoke test.
    """
    rng = np.random.default_rng(seed)
    n = 12
    var = ["L1", "L2", "R1", "R2", "filler"]
    X = sparse.csr_matrix(rng.poisson(1.0, size=(n, len(var))).astype("float32"))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["active_ident"] = pd.Categorical(
        [f"G{i % 2}" for i in range(n)], categories=["G0", "G1"]
    )
    var_df = pd.DataFrame(index=var)
    adata = ad.AnnData(X=X, obs=obs, var=var_df)

    # Minimal DB — only `complex` and `geneInfo` are read by extractGeneSubset
    DB = {
        "complex": pd.DataFrame(index=pd.Index([], name="symbol")),
        "geneInfo": pd.DataFrame({"Symbol": var}, index=var),
        "interaction": pd.DataFrame({
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "pathway_name": ["PW_A", "PW_B"],
        }, index=["L1_R1", "L2_R2"]),
    }
    LR = {
        "LRsig": pd.DataFrame({
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "pathway_name": ["PW_A", "PW_B"],
            "interaction_name": ["L1_R1", "L2_R2"],
        }, index=["L1_R1", "L2_R2"])
    }
    K = 2
    nLR = 2
    prob = rng.uniform(0.0, 1.0, size=(K, K, nLR))
    pval = rng.uniform(0.0, 0.04, size=(K, K, nLR))
    adata.uns["cellchat"] = {
        "DB": DB, "LR": LR,
        "net": {
            "prob": prob, "pval": pval,
            "LR.sig": ["L1_R1", "L2_R2"],
            "LR.sig.dimnames3": ["L1_R1", "L2_R2"],
        },
    }
    return adata


# ---------------------------------------------------------------------------
# dotPlot
# ---------------------------------------------------------------------------
def test_dotPlot_smoke():
    adata = _make_simple_adata()
    fig = dotPlot(adata, features=["g0", "g1", "g2"])
    assert isinstance(fig, Figure)
    plt.close("all")


def test_dotPlot_explicit_groupby():
    adata = _make_simple_adata()
    fig = dotPlot(adata, features=["g0", "g1"], group_by="active_ident",
                  color_heatmap="Blues", swap_axes=True)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_dotPlot_standard_scale():
    adata = _make_simple_adata()
    fig = dotPlot(adata, features=["g0", "g1", "g2"], standard_scale="var")
    assert isinstance(fig, Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# StackedVlnPlot
# ---------------------------------------------------------------------------
def test_StackedVlnPlot_smoke():
    adata = _make_simple_adata()
    fig = StackedVlnPlot(adata, features=["g0", "g1", "g2"])
    assert isinstance(fig, Figure)
    plt.close("all")


def test_StackedVlnPlot_with_color_use_and_swap_axes():
    adata = _make_simple_adata()
    fig = StackedVlnPlot(adata, features=["g0", "g1"],
                         color_use=["#FF0000", "#0000FF", "#00FF00"],
                         swap_axes=True)
    assert isinstance(fig, Figure)
    plt.close("all")


# ---------------------------------------------------------------------------
# plotGeneExpression
# ---------------------------------------------------------------------------
def test_plotGeneExpression_violin_smoke():
    adata = _make_adata_for_extractEnrichedLR()
    fig = plotGeneExpression(adata, signaling=["PW_A"], type="violin",
                             enriched_only=False)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plotGeneExpression_dot_smoke():
    adata = _make_adata_for_extractEnrichedLR()
    fig = plotGeneExpression(adata, signaling=["PW_A", "PW_B"], type="dot",
                             enriched_only=False)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plotGeneExpression_invalid_type_raises():
    adata = _make_adata_for_extractEnrichedLR()
    with pytest.raises(ValueError, match="type"):
        plotGeneExpression(adata, signaling=["PW_A"], type="bogus")  # type: ignore[arg-type]


def test_plotGeneExpression_no_overlap_raises():
    adata = _make_adata_for_extractEnrichedLR()
    # Drop all genes so the L/R set has nothing to plot
    adata = adata[:, ["filler"]].copy()
    with pytest.raises(ValueError, match="None of the inferred"):
        plotGeneExpression(adata, signaling=["PW_A"], type="violin",
                           enriched_only=False)
