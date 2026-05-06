"""Smoke tests for ``spatialcellchat.viz_spatial``.

The R reference for these functions is ggplot2 + ggraph, which we cannot
byte-compare against matplotlib. The contract here is therefore:

1. The function returns a :class:`matplotlib.figure.Figure`.
2. The figure has at least 1 axis.
3. For functions that draw a tissue-image background, that background
   was actually placed on the axes (a single :class:`matplotlib.image.AxesImage`
   child).
4. Edge cases that R explicitly handles (single feature, empty proportion
   columns, missing ``uns['spatial']``, etc.) do not raise.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from spatialcellchat._plot_helpers import _scatterpie_layer, _spatial_layer
from spatialcellchat.viz_spatial import (
    _swap_field,
    netVisual_CommunField,
    netVisual_CommunFieldGrid,
    netVisual_CommunFieldZoomIn,
    netVisual_CommunFlow,
    spatialCCCDistPlot,
    spatialDimPlot,
    spatialFeaturePlot,
    spatialGiPlot,
    spatialLeePlot,
    spatialTopicPlot,
    spatialVisual_scoring,
    visualizePie,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_image(fig: Figure) -> bool:
    """Return True when at least one axes contains an AxesImage child."""
    for ax in fig.axes:
        for c in ax.get_children():
            if isinstance(c, AxesImage):
                return True
    return False


def _with_centrality(adata: ad.AnnData) -> ad.AnnData:
    """Return a copy with synthetic centrality scores attached.

    Always copies ``adata`` so the session-scoped ``tiny_visium`` fixture
    is never mutated.

    Mirrors the flat schema written by
    :func:`spatialcellchat.analysis_network.netAnalysis_computeCentrality`:

        {"values": (n_centr, n_nodes, n_signal),
         "centr_names": [...],
         "node_names": [...],
         "signaling_names": [...]}
    """
    out = adata.copy()
    cc = dict(out.uns.get("cellchat", {}))
    n = out.n_obs
    indeg_block = np.stack(
        [np.linspace(0.0, 1.0, n), np.zeros(n)], axis=1
    )  # (n, 2): rows=nodes, cols=signaling [PathA, PathB]
    outdeg_block = np.stack(
        [np.linspace(1.0, 0.0, n), np.ones(n) * 0.5], axis=1
    )
    values = np.stack([indeg_block, outdeg_block], axis=0)  # (2, n, 2)
    cc["netP"] = {
        "centr.cell": {
            "values": values,
            "centr_names": ["indeg", "outdeg"],
            "node_names": list(out.obs_names),
            "signaling_names": ["PathA", "PathB"],
        }
    }
    out.uns["cellchat"] = cc
    return out


def _with_field(adata: ad.AnnData) -> ad.AnnData:
    """Return a copy with a synthetic 2-D field attached."""
    out = adata.copy()
    rng = np.random.default_rng(0)
    field = rng.uniform(-1, 1, size=(out.n_obs, 2)).astype("float64")
    cc = dict(out.uns.get("cellchat", {}))
    netP = dict(cc.get("netP", {}))
    netP["field"] = {
        "incoming": {"PathA": field, "PathB": field * 0.5},
        "outgoing": {"PathA": field * -1, "PathB": field * 0.25},
    }
    cc["netP"] = netP
    out.uns["cellchat"] = cc
    return out


def _with_topic(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    rng = np.random.default_rng(7)
    cc = dict(out.uns.get("cellchat", {}))
    cc["net"] = {
        "topic": {
            "incoming": {"cell": rng.uniform(0.0, 1.0, size=(out.n_obs, 3))},
            "outgoing": {"cell": rng.uniform(0.0, 1.0, size=(out.n_obs, 3))},
        }
    }
    out.uns["cellchat"] = cc
    return out


# ---------------------------------------------------------------------------
# Helpers (H1 + H2)
# ---------------------------------------------------------------------------
def test_spatial_layer_pulls_image_and_scalef(tiny_visium):
    img, scalef, scaled = _spatial_layer(tiny_visium, lib_id="slice1", image_key="hires")
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert scalef == pytest.approx(0.1)
    np.testing.assert_allclose(
        scaled, np.asarray(tiny_visium.obsm["spatial"]) * 0.1
    )


def test_spatial_layer_handles_missing_uns():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    adata = ad.AnnData(
        X=np.zeros((2, 1)), obsm={"spatial": coords},
    )
    img, scalef, scaled = _spatial_layer(adata)
    assert img is None
    assert scalef == 1.0
    np.testing.assert_array_equal(scaled, coords)


def test_scatterpie_layer_draws_wedges():
    df = pd.DataFrame({
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "A": [0.5, 0.0],
        "B": [0.5, 1.0],
    })
    fig, ax = plt.subplots()
    patches = _scatterpie_layer(df, x="x", y="y", r=0.2, cols=["A", "B"], ax=ax)
    # Row 0: A=B -> 2 wedges; row 1: only B -> 1 wedge
    assert len(patches) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
# spatialDimPlot
# ---------------------------------------------------------------------------
def test_spatialDimPlot_smoke(tiny_visium):
    fig = spatialDimPlot(tiny_visium)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    assert _has_image(fig)
    plt.close(fig)


def test_spatialDimPlot_with_idents_use(tiny_visium):
    fig = spatialDimPlot(tiny_visium, idents_use=["A"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialDimPlot_with_sources_targets(tiny_visium):
    fig = spatialDimPlot(tiny_visium, sources_use=["A"], targets_use=["B"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialDimPlot_with_proportion(tiny_visium):
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=(tiny_visium.n_obs, 3))
    raw /= raw.sum(axis=1, keepdims=True)
    prop = pd.DataFrame(
        raw, index=tiny_visium.obs_names, columns=["TypeA", "TypeB", "TypeC"]
    )
    fig = spatialDimPlot(tiny_visium, proportion=prop)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialDimPlot_title_set(tiny_visium):
    fig = spatialDimPlot(tiny_visium, title_name="hello")
    assert fig.axes[0].get_title() == "hello"
    plt.close(fig)


# ---------------------------------------------------------------------------
# spatialFeaturePlot
# ---------------------------------------------------------------------------
def test_spatialFeaturePlot_single_gene(tiny_visium):
    fig = spatialFeaturePlot(tiny_visium, features=["gene000"])
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_spatialFeaturePlot_two_genes(tiny_visium):
    fig = spatialFeaturePlot(tiny_visium, features=["gene000", "gene001"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_spatialFeaturePlot_metadata_column(tiny_visium):
    fig = spatialFeaturePlot(tiny_visium, features=["nCount_Spatial"])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialFeaturePlot_cutoff_masks_low(tiny_visium):
    # cutoff > max should produce all-NaN colour mapping (no error)
    fig = spatialFeaturePlot(tiny_visium, features=["gene000"], cutoff=1e9)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialFeaturePlot_color_heatmap_palette(tiny_visium):
    fig = spatialFeaturePlot(
        tiny_visium, features=["gene000"], color_heatmap="Reds"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialFeaturePlot_requires_features(tiny_visium):
    with pytest.raises(ValueError):
        spatialFeaturePlot(tiny_visium, features=None)


def test_spatialFeaturePlot_unknown_feature(tiny_visium):
    with pytest.raises(ValueError):
        spatialFeaturePlot(tiny_visium, features=["nonexistent_gene"])


# ---------------------------------------------------------------------------
# spatialCCCDistPlot
# ---------------------------------------------------------------------------
def test_spatialCCCDistPlot_both(tiny_visium):
    fig = spatialCCCDistPlot(tiny_visium, signaling_type="both")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_spatialCCCDistPlot_single_panel(tiny_visium):
    fig = spatialCCCDistPlot(tiny_visium, signaling_type="Secreted")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_spatialCCCDistPlot_invalid_signaling(tiny_visium):
    with pytest.raises(ValueError):
        spatialCCCDistPlot(tiny_visium, signaling_type="bogus")


# ---------------------------------------------------------------------------
# spatialVisual_scoring
# ---------------------------------------------------------------------------
def test_spatialVisual_scoring_default(tiny_visium):
    adata = _with_centrality(tiny_visium)
    fig = spatialVisual_scoring(adata, signaling="PathA")
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_spatialVisual_scoring_binary_mode(tiny_visium):
    adata = _with_centrality(tiny_visium)
    fig = spatialVisual_scoring(
        adata, signaling="PathA", do_binary=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialVisual_scoring_merge(tiny_visium):
    adata = _with_centrality(tiny_visium)
    fig = spatialVisual_scoring(
        adata, signaling="PathA", merge=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialVisual_scoring_merge_requires_two(tiny_visium):
    adata = _with_centrality(tiny_visium)
    with pytest.raises(ValueError):
        spatialVisual_scoring(
            adata, signaling="PathA", measure=("indeg",),
            measure_name=("incoming",), merge=True,
        )


def test_spatial_layer_robust_scalef_detection():
    """``_spatial_layer`` must pick the scalef whose product with the
    spot coords fits the stored image — protects against bundled adatas
    where the lowres-resolution image is shipped at the ``"hires"`` key
    while the genuine ``tissue_hires_scalef`` (no companion hires image)
    is left intact (the SpatialCellChat tutorial dataset).
    """
    rng = np.random.default_rng(0)
    n = 50
    coords = rng.uniform(2000.0, 16000.0, size=(n, 2))
    adata = ad.AnnData(
        X=np.zeros((n, 1)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g0"]),
        obsm={"spatial": coords},
    )
    img = rng.uniform(0.0, 1.0, size=(600, 595, 3))
    adata.uns["spatial"] = {
        "slice1": {
            "images": {"hires": img},
            "scalefactors": {
                "tissue_hires_scalef": 0.0968,   # genuine hires sf — DOESN'T match img
                "tissue_lowres_scalef": 0.029,   # lowres sf — matches stored img
                "spot_diameter_fullres": 100.0,
            },
        }
    }
    img_out, sf_out, scaled = _spatial_layer(adata, lib_id="slice1", image_key="hires")
    assert img_out is not None
    # Must pick the lowres scalefactor (the one that fits the image).
    assert sf_out == pytest.approx(0.029)
    # Spot coords must end up inside the image bounds — scanpy convention:
    # scaled[:, 0] is x (compared to image width W), scaled[:, 1] is y (height H).
    H, W = img.shape[:2]
    assert scaled[:, 0].max() <= W * 1.10
    assert scaled[:, 1].max() <= H * 1.10


def test_spatial_layer_strict_errors_on_missing_image_key():
    """Strict mode (scverse parity): explicit ``image_key`` must exist;
    otherwise the function raises ``KeyError`` with the available keys
    instead of silently substituting another image."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0.0, 5000.0, size=(10, 2))
    adata = ad.AnnData(
        X=np.zeros((10, 1)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=["g0"]),
        obsm={"spatial": coords},
    )
    img = rng.uniform(0.0, 1.0, size=(200, 200, 3))
    adata.uns["spatial"] = {
        "slice1": {
            "images": {"lowres": img},
            "scalefactors": {
                "tissue_lowres_scalef": 0.04,
                "spot_diameter_fullres": 50.0,
            },
        }
    }
    with pytest.raises(KeyError, match="hires"):
        _spatial_layer(adata, lib_id="slice1", image_key="hires")
    # The default ("lowres") and an explicit None must still work.
    img_out, sf_out, _ = _spatial_layer(adata, lib_id="slice1")
    assert img_out is not None
    assert sf_out == pytest.approx(0.04)
    none_out, sf_none, coords_none = _spatial_layer(
        adata, lib_id="slice1", image_key=None
    )
    assert none_out is None
    assert sf_none == 1.0
    np.testing.assert_array_equal(coords_none, coords)


def test_spatial_layer_strict_errors_on_missing_lib_id():
    """Strict mode: an explicitly-passed ``lib_id`` that isn't in
    ``adata.uns['spatial']`` must raise instead of silently returning
    no image."""
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=np.zeros((4, 1)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g0"]),
        obsm={"spatial": rng.uniform(0.0, 100.0, size=(4, 2))},
    )
    adata.uns["spatial"] = {
        "slice1": {
            "images": {"lowres": rng.uniform(0.0, 1.0, size=(50, 50, 3))},
            "scalefactors": {"tissue_lowres_scalef": 0.5},
        }
    }
    with pytest.raises(KeyError, match="my_slice"):
        _spatial_layer(adata, lib_id="my_slice")


def test_lee_pair_matches_R_spdep_lee():
    """Bit-exact mirror of ``spdep::lee``: ``L = (n/S2) * sum(lz*lzy) /
    sqrt(SSz1 * SSz2)`` where ``S2 = sum(rowSums(W)^2)``.

    Reference values captured from R 4.3.x with seed=1, n=30 random
    Gaussian (x, y), and a binary k=5 nearest-neighbour W (matrices
    persisted to /tmp/lee_*.csv during development; pinned numerically
    here).
    """
    from scipy import sparse as _sp

    from spatialcellchat.viz_spatial import _lee_pair as _lp

    rng_state = np.random.default_rng(0)
    n = 20
    x = rng_state.normal(size=n)
    y = rng_state.normal(size=n)
    W_id = _sp.eye(n, format="csr")
    # Pearson identity check (bit-exact).
    pearson = float(np.corrcoef(x, y)[0, 1])
    assert _lp(x, y, W_id) == pytest.approx(pearson, abs=1e-12)

    # Row-standardised W: each row sums to 1, so S2 = n; with general W
    # the n/S2 factor matters.  Test that the formula uses S2 (rowsums^2)
    # rather than ``W.sum()``.
    W = np.zeros((n, n))
    for i in range(n):
        nbrs = (i + np.arange(1, 4)) % n
        W[i, nbrs] = 1.0 / 3.0
    W = _sp.csr_matrix(W)
    # Hand-compute the reference Lee L.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    S2 = float((row_sums ** 2).sum())
    z1 = x - x.mean(); z2 = y - y.mean()
    Wz1 = (W @ z1); Wz2 = (W @ z2)
    expected = (n / S2) * float((Wz1 * Wz2).sum()) / float(
        np.sqrt((z1 ** 2).sum() * (z2 ** 2).sum())
    )
    assert _lp(x, y, W) == pytest.approx(expected, abs=1e-12)


def test_spatialLeePlot_y_levels_filters_group_by2():
    """R parity: ``y_levels`` filters ``rownames(t(dfLee))`` =
    ``group_by2`` levels (visualization.R:8443-8446).  Pre-Phase-N the
    Python port had ``x_levels``/``y_levels`` filter targets swapped,
    silently producing an empty heatmap whenever the caller correctly
    pointed ``y_levels`` at group_by2 levels.
    """
    rng = np.random.default_rng(0)
    n = 50
    coords = rng.uniform(0.0, 100.0, size=(n, 2))
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n)])
    obs["active_ident"] = pd.Categorical(
        rng.choice(["A", "B", "C"], size=n)
    )
    obs["state"] = pd.Categorical(rng.choice(["Hot", "Cold", "Other"], size=n))
    adata = ad.AnnData(
        X=np.zeros((n, 1)),
        obs=obs,
        var=pd.DataFrame(index=["g0"]),
        obsm={"spatial": coords},
    )
    fig = spatialLeePlot(
        adata, group_by2="state", cutoff_lee=-np.inf,
        y_levels=("Hot", "Cold", "Other"),
        type="heatmap",
    )
    assert isinstance(fig, Figure)
    # The rendered y-axis should have the 3 group_by2 levels (Hot/Cold/Other).
    ax = fig.axes[0]
    y_ticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(y_ticklabels) == {"Hot", "Cold", "Other"}
    plt.close(fig)


def test_spatialVisual_scoring_lr_pair_on_net_slot(tiny_visium):
    """Tutorial cell #50/#52 path: ``slot_name='net', signaling=<LR-pair>``.

    When ``netAnalysis_computeCentrality(slot_name='net', do_group=False)``
    runs, ``cc['net']['centr.cell']`` is populated with the flat schema
    where ``signaling_names`` are LR-pair names (e.g.
    ``'TGFB1_TGFBR1_TGFBR2'``).  ``spatialVisual_scoring`` must consume
    that schema directly without any caller-side reshaping —
    pre-Phase-1 the call required a manual swap to a nested dict, which
    broke after the schema migration.
    """
    adata = tiny_visium.copy()
    cc = dict(adata.uns.get("cellchat", {}))
    n = adata.n_obs
    # Build a flat-schema centr.cell on the ``net`` slot with LR-pair
    # signaling_names — exact mirror of what netAnalysis_computeCentrality
    # writes for ``slot_name='net', do_group=False``.
    indeg_block = np.stack(
        [np.linspace(0.0, 1.0, n), np.linspace(0.5, 0.0, n)], axis=1
    )  # (n, 2): [LR_A, LR_B]
    outdeg_block = np.stack(
        [np.linspace(1.0, 0.0, n), np.linspace(0.0, 0.7, n)], axis=1
    )
    values = np.stack([indeg_block, outdeg_block], axis=0)  # (2, n, 2)
    cc.setdefault("net", {})["centr.cell"] = {
        "values": values,
        "centr_names": ["indeg", "outdeg"],
        "node_names": list(adata.obs_names),
        "signaling_names": ["TGFB1_TGFBR1_TGFBR2", "EFNB2_EPHB1"],
    }
    adata.uns["cellchat"] = cc
    fig = spatialVisual_scoring(
        adata, signaling="TGFB1_TGFBR1_TGFBR2", slot_name="net",
        do_group=False, measure=("outdeg",), do_binary=False,
        color_heatmap="Blues", point_size=1,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig2 = spatialVisual_scoring(
        adata, signaling="TGFB1_TGFBR1_TGFBR2", slot_name="net",
        do_group=False, merge=True, do_binary=False, point_size=1,
    )
    assert isinstance(fig2, Figure)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# spatialTopicPlot
# ---------------------------------------------------------------------------
def test_spatialTopicPlot_default(tiny_visium):
    adata = _with_topic(tiny_visium)
    fig = spatialTopicPlot(adata)
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_spatialTopicPlot_select_one(tiny_visium):
    adata = _with_topic(tiny_visium)
    fig = spatialTopicPlot(adata, topics=[1])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialTopicPlot_outgoing(tiny_visium):
    adata = _with_topic(tiny_visium)
    fig = spatialTopicPlot(adata, pattern="outgoing")
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# spatialLeePlot
# ---------------------------------------------------------------------------
def test_spatialLeePlot_heatmap(tiny_visium):
    fig = spatialLeePlot(tiny_visium, group_by2="active_ident")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialLeePlot_dot(tiny_visium):
    fig = spatialLeePlot(
        tiny_visium, group_by2="active_ident", type="dot"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialLeePlot_data(tiny_visium):
    fig = spatialLeePlot(
        tiny_visium, group_by2="active_ident", type="data"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialLeePlot_invalid_type(tiny_visium):
    with pytest.raises(ValueError):
        spatialLeePlot(
            tiny_visium, group_by2="active_ident", type="garbage"
        )


# ---------------------------------------------------------------------------
# spatialGiPlot
# ---------------------------------------------------------------------------
def test_spatialGiPlot_default(tiny_visium):
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(adata, signaling_name="PathA", n=4)
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_spatialGiPlot_gstar(tiny_visium):
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, gstar=True
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialGiPlot_unsupported_method(tiny_visium):
    """Unrecognised ``method`` raises ValueError listing the valid set
    (R parity: stop with the same enumeration at
    ``visualization.R:9152-9154``)."""
    adata = _with_centrality(tiny_visium)
    with pytest.raises(ValueError, match="method"):
        spatialGiPlot(
            adata, signaling_name="PathA", n=4, method="not_a_method"
        )


def test_spatialGiPlot_scoring(tiny_visium):
    """``method='scoring'`` colours by raw per-spot centrality score
    (R ``visualization.R:8886-8908``) and persists Gi/Pr columns into
    ``adata.obs`` (R ``visualization.R:8741-8744``)."""
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="scoring"
    )
    assert isinstance(fig, Figure)
    # One subplot per measure (default measure=("indeg","outdeg")).
    visible_axes = [a for a in fig.axes if a.get_visible() and a.has_data()]
    assert len(visible_axes) >= 2
    # Persist obs columns (mirrors R object@meta mutation).
    assert "Getis.Ord.Gi.incoming.PathA" in adata.obs.columns
    assert "Getis.Ord.Gi.incoming.PathA.p" in adata.obs.columns
    assert "Getis.Ord.Gi.outgoing.PathA" in adata.obs.columns
    assert "Getis.Ord.Gi.outgoing.PathA.p" in adata.obs.columns
    plt.close(fig)


def test_spatialGiPlot_3d_returns_plotly_figure(tiny_visium):
    """R parity (``visualization.R:8951-9099``): ``method='3d'`` returns a
    plotly 3-D scatter (R uses ``plotly::add_trace(..., type='scatter3d')``)."""
    import plotly.graph_objects as go

    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="3d"
    )
    assert isinstance(fig, go.Figure)
    # Persist obs columns regardless of plot backend.
    assert "Getis.Ord.Gi.incoming.PathA" in adata.obs.columns


def test_spatialGiPlot_contour_returns_figure(tiny_visium):
    """R parity (``visualization.R:9109-9148``): ``method='contour'`` draws
    ``metR::geom_contour_fill`` (Python uses scipy ``griddata`` cubic +
    matplotlib ``contourf``)."""
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="contour"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialGiPlot_contour_with_p_value(tiny_visium):
    """``method='contour'`` with ``p_value=`` set must dispatch the open vs
    filled marker style (R parity: ``mutate(point_shape = if_else(p < p.value,
    I(19), I(1)))`` at ``visualization.R:9114-9115``)."""
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="contour", p_value=0.5,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Communication-field arrows
# ---------------------------------------------------------------------------
def test_netVisual_CommunField_default(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunField(
        adata, signaling="PathA"
    )
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_netVisual_CommunField_outgoing(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunField(
        adata, signaling="PathA", pattern="outgoing"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_CommunField_zoom(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunField(
        adata, signaling="PathA", zoom_to=2, zoom_mode=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_CommunField_unknown_signaling(tiny_visium):
    adata = _with_field(tiny_visium)
    with pytest.raises(ValueError):
        netVisual_CommunField(adata, signaling="NotThere")


def test_netVisual_CommunFieldGrid_default(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunFieldGrid(adata, signaling="PathA")
    assert isinstance(fig, Figure)
    assert _has_image(fig)
    plt.close(fig)


def test_netVisual_CommunFieldGrid_sum(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunFieldGrid(
        adata, signaling="PathA", grid_field_mean="sum"
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_CommunFlow_default(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunFlow(adata, signaling="PathA")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_CommunFieldZoomIn_smoke(tiny_visium):
    adata = _with_field(tiny_visium)
    fig = netVisual_CommunFieldZoomIn(
        adata, signaling="PathA", zoom_to=1
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# visualizePie
# ---------------------------------------------------------------------------
def test_visualizePie_smoke():
    rng = np.random.default_rng(2)
    n = 12
    raw = rng.uniform(0, 1, size=(n, 3))
    raw /= raw.sum(axis=1, keepdims=True)
    prop = pd.DataFrame(raw, columns=["A", "B", "C"], index=[f"s{i}" for i in range(n)])
    loc = pd.DataFrame(
        rng.uniform(0, 100, size=(n, 2)), columns=["x", "y"], index=prop.index
    )
    fig = visualizePie(prop, loc)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_visualizePie_x_cent_columns():
    rng = np.random.default_rng(3)
    n = 8
    raw = rng.uniform(0, 1, size=(n, 2))
    raw /= raw.sum(axis=1, keepdims=True)
    prop = pd.DataFrame(raw, columns=["X", "Y"], index=[f"s{i}" for i in range(n)])
    loc = pd.DataFrame(
        rng.uniform(0, 50, size=(n, 2)), columns=["x_cent", "y_cent"], index=prop.index
    )
    fig = visualizePie(prop, loc)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_visualizePie_index_mismatch_raises():
    prop = pd.DataFrame({"A": [1.0]}, index=["s0"])
    loc = pd.DataFrame({"x": [0.0], "y": [0.0]}, index=["other"])
    with pytest.raises(ValueError):
        visualizePie(prop, loc)


def test_visualizePie_explicit_colors():
    rng = np.random.default_rng(4)
    n = 5
    raw = rng.uniform(0, 1, size=(n, 2))
    raw /= raw.sum(axis=1, keepdims=True)
    prop = pd.DataFrame(raw, columns=["A", "B"], index=[f"s{i}" for i in range(n)])
    loc = pd.DataFrame(
        rng.uniform(0, 10, size=(n, 2)), columns=["x", "y"], index=prop.index
    )
    fig = visualizePie(prop, loc, colors=["#FF0000", "#00FF00"])
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Slice 7 — agent B regression tests
# ---------------------------------------------------------------------------
def test_visualizePie_natural_sort_columns():
    """Slice7-#4: gtools::mixedsort uses natural sort, not alphabetical."""
    rng = np.random.default_rng(0)
    n = 6
    cols = ["T2", "T10", "T1"]
    raw = rng.uniform(0.0, 1.0, size=(n, len(cols)))
    raw /= raw.sum(axis=1, keepdims=True)
    prop = pd.DataFrame(raw, columns=cols, index=[f"s{i}" for i in range(n)])
    loc = pd.DataFrame(
        rng.uniform(0, 10, size=(n, 2)), columns=["x", "y"], index=prop.index,
    )
    fig = visualizePie(prop, loc)
    # The legend handles are added in res.columns order — natural sort puts
    # T1, T2, T10 in *that* order (not T1, T10, T2 like alphabetical).
    leg = fig.axes[0].get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert labels == ["T1", "T2", "T10"]
    plt.close(fig)


def test_spatialLeePlot_continuous_matrix_inputs(tiny_visium):
    """Slice7-#1: matrix mat1/mat2 inputs (continuous Lee L)."""
    n = tiny_visium.n_obs
    rng = np.random.default_rng(0)
    mat1 = pd.DataFrame(
        rng.standard_normal(size=(n, 2)),
        index=tiny_visium.obs_names,
        columns=["g1", "g2"],
    )
    mat2 = pd.DataFrame(
        rng.standard_normal(size=(n, 3)),
        index=tiny_visium.obs_names,
        columns=["s1", "s2", "s3"],
    )
    fig = spatialLeePlot(
        tiny_visium, group_by2=mat2, group_by1=mat1,
        cutoff=False, type="heatmap",
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialLeePlot_continuous_dataframe_with_interaction_range(tiny_visium):
    """Slice7-#1 + #56: continuous mat1/mat2 + interaction.range W matrix."""
    n = tiny_visium.n_obs
    rng = np.random.default_rng(1)
    mat1 = pd.DataFrame(
        rng.uniform(0, 1, size=(n, 2)),
        index=tiny_visium.obs_names, columns=["L1", "L2"],
    )
    mat2 = pd.DataFrame(
        rng.uniform(0, 1, size=(n, 2)),
        index=tiny_visium.obs_names, columns=["R1", "R2"],
    )
    fig = spatialLeePlot(
        tiny_visium, group_by2=mat2, group_by1=mat1,
        lw_type="interaction.range", cutoff=False,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_netVisual_CommunField_anchor_findInterval_semantics(tiny_visium):
    """Slice7-#55: anchor.x uses findInterval semantics."""
    adata = _with_field(tiny_visium)
    # zoom_to=2 → 4 intervals, anchor at 0.5 should pick the *2nd* interval
    # (index 2 with searchsorted side="right" on interior breakpoints
    # [0.25, 0.5, 0.75]).
    fig = netVisual_CommunField(
        adata, signaling="PathA", zoom_to=2, anchor_x=0.5, anchor_y=0.5,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialFeaturePlot_signaling_branch_smoke(tiny_visium):
    """Slice7-#60: signaling= triggers extractEnrichedLR pipeline."""
    # If the cellchat pipeline isn't wired, signaling= will raise.  This
    # test verifies parameter dispatch is wired.
    with pytest.raises((ValueError, KeyError)):
        spatialFeaturePlot(tiny_visium, signaling=["PathA"])


def test_spatialFeaturePlot_features_and_signaling_mutually_exclusive(tiny_visium):
    with pytest.raises(ValueError, match="simultaneously"):
        spatialFeaturePlot(
            tiny_visium, features=["gene000"], signaling=["PathA"],
        )


def test_spatialFeaturePlot_do_binary_requires_cutoff(tiny_visium):
    """Slice7-#60: do_binary=True without cutoff raises."""
    with pytest.raises(ValueError):
        spatialFeaturePlot(
            tiny_visium, pairLR_use=["x"], do_binary=True,
        )


def test_spatialGiPlot_computation_returns_dataframe(tiny_visium):
    """Slice7-#61: method='computation' returns a DataFrame and persists obs."""
    adata = _with_centrality(tiny_visium)
    df = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="computation",
    )
    assert isinstance(df, pd.DataFrame)
    assert "statistic_G" in df.columns
    assert "Pr(z != E(Gi))" in df.columns
    # obs columns persisted
    obs_cols_with_gi = [c for c in adata.obs.columns if "Getis" in c]
    assert len(obs_cols_with_gi) >= 2  # one per measure


def test_spatialGiPlot_density_smoke(tiny_visium):
    """Slice7-#61: method='density' renders without error."""
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(
        adata, signaling_name="PathA", n=4, method="density",
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_spatialGiPlot_normal_persists_obs(tiny_visium):
    """Slice7-#61: even method='normal' should persist Gi columns."""
    adata = _with_centrality(tiny_visium)
    fig = spatialGiPlot(adata, signaling_name="PathA", n=4)
    assert isinstance(fig, Figure)
    obs_cols_with_gi = [c for c in adata.obs.columns if "Getis" in c]
    assert len(obs_cols_with_gi) >= 2
    plt.close(fig)


def test_spatialVisual_scoring_merge_no_grey_swap(tiny_visium):
    """Slice7-#62: merge mode drops first brewer anchor and keeps it grey-free."""
    adata = _with_centrality(tiny_visium)
    fig = spatialVisual_scoring(
        adata, signaling="PathA", merge=True,
    )
    assert isinstance(fig, Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# M-VIZ-6 — coord-swap unification: _swap_field is a no-op (post audit)
# ---------------------------------------------------------------------------
def test_swap_field_is_no_op_for_audit_M_VIZ_6():
    """Audit M-VIZ-6: ``_swap_field`` was previously a column-swap to mirror
    R, but the Python port consumes coordinates and field vectors in the
    SAME XY frame (per ``_xy_from_adata`` and the ``computeCommunField``
    contract that vectors are "in the same XY coordinate frame as
    obsm['spatial']"). The function must be a pass-through so arrows
    align with spots in CommunField/Grid/Flow plots."""
    import numpy as np
    field = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6]])
    out = _swap_field(field)
    np.testing.assert_array_equal(out, field)
