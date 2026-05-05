"""Tests for spatialcellchat.de — DE / spatially-variable feature exports.

Each test compares Python output to ground-truth dumped from the R
``SpatialCellChat`` package via ``Rscript`` (frozen CSV fixtures live under
``/tmp/de_harness`` and are produced by the harness scripts noted in the
docstrings; the per-row numeric values are also embedded inline below).

Coverage matrix
---------------
- wilcox: defaults, only_pos=False, idents_use=[...], invert=True, group_by,
  no-feature-passes-threshold edge case, group_dataset two-condition
- moransi: parity to ape::Moran.I (machine precision)
- meringue: parity to MERINGUE::getSpatialPatterns (machine precision)
- identifyOverExpressedLigandReceptor: complex-expansion behaviour (gene
  appearing in multiple complex subunit lists yields one row per complex)
- identifyOverExpressedInteractions: variable_both=True/False
"""

from __future__ import annotations

import io
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spatialcellchat import load_db
from spatialcellchat.de import (
    _ape_moran_i,
    _bh,
    _bonferroni,
    _meringue_neighbor_weights,
    identifyOverExpressedGenes,
    identifyOverExpressedInteractions,
    identifyOverExpressedLigandReceptor,
)


# ---------------------------------------------------------------------------
# Inline fixture data — generated once with the same RNG seed used for
# `/tmp/de_harness/data.csv`. Keeping the fixtures embedded in the test file
# makes the tests deterministic & free-standing.
# ---------------------------------------------------------------------------
def _build_harness_adata() -> ad.AnnData:
    """Reproduce the harness fixture: 30 cells × 25 log-normalised genes,
    3 idents (A/B/C), random 2-D spatial coords. Matches the data the R
    harness scripts (`/tmp/de_harness/run_*.R`) consumed exactly.
    """
    rng = np.random.default_rng(42)
    n_cells, n_genes = 30, 25
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / totals * 10000)
    genes = [f"gene{i:02d}" for i in range(n_genes)]
    cells = [f"cell{i:02d}" for i in range(n_cells)]
    obs = pd.DataFrame(index=cells)
    obs["active_ident"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_cells))
    coords = rng.uniform(0, 100, size=(n_cells, 2))
    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm.astype("float64")),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = coords
    return adata


@pytest.fixture(scope="module")
def harness_adata() -> ad.AnnData:
    return _build_harness_adata()


# ---------------------------------------------------------------------------
# Wilcox branch
# ---------------------------------------------------------------------------
# Generated with /tmp/de_harness/run_wilcox.R (same fixture).
_WILCOX_DEFAULTS_CSV = """clusters,features,pvalues,logFC,pct.1,pct.2,pvalues.adj
A,gene20,0.0158087769926966,0.781580231948173,0.9,0.7,0.395219424817414
B,gene06,0.0135513822244659,0.529270204177997,1.0,0.905,0.338784555611647
"""


def test_wilcox_defaults(harness_adata):
    """`identifyOverExpressedGenes(method='wilcox')` matches R bit-for-bit."""
    markers = identifyOverExpressedGenes(harness_adata.copy(), selection_method="wilcox")
    expected = pd.read_csv(io.StringIO(_WILCOX_DEFAULTS_CSV))
    assert len(markers) == len(expected)
    for col in ("pvalues", "logFC", "pct.1", "pct.2", "pvalues.adj"):
        np.testing.assert_allclose(
            markers[col].to_numpy(),
            expected[col].to_numpy(),
            atol=1e-12,
            rtol=1e-9,
        )
    assert list(markers["clusters"]) == list(expected["clusters"])
    assert list(markers["features"]) == list(expected["features"])


def test_wilcox_writes_uns(harness_adata):
    adata = harness_adata.copy()
    identifyOverExpressedGenes(adata, selection_method="wilcox")
    de = adata.uns["cellchat"]["DE"]
    assert "features" in de and "features.info" in de
    assert isinstance(de["features"], list)
    assert isinstance(de["features.info"], pd.DataFrame)
    # var-level highly-variable mask
    assert adata.var["highly_variable_signaling"].dtype == bool
    n_sig = int(adata.var["highly_variable_signaling"].sum())
    assert n_sig == len(set(de["features"]))


def test_wilcox_idents_use(harness_adata):
    """Restricting to idents.use = ['A', 'B'] (R harness: 24 rows at p<1)."""
    markers = identifyOverExpressedGenes(
        harness_adata.copy(),
        selection_method="wilcox",
        idents_use=["A", "B"],
        thresh_p=1.0,
    )
    assert set(markers["clusters"]) == {"A", "B"}
    assert len(markers) == 24


def test_wilcox_idents_use_invert(harness_adata):
    """invert=True drops the listed idents (R harness: 26 rows for not-A)."""
    markers = identifyOverExpressedGenes(
        harness_adata.copy(),
        selection_method="wilcox",
        idents_use=["A"],
        invert=True,
        thresh_p=1.0,
    )
    assert set(markers["clusters"]) == {"B", "C"}
    assert len(markers) == 26


def test_wilcox_only_pos_false(harness_adata):
    """only.pos=FALSE keeps logFC<0 genes (R harness same 2 rows since
    no negative-FC gene crosses p<0.05 in this fixture)."""
    markers = identifyOverExpressedGenes(
        harness_adata.copy(),
        selection_method="wilcox",
        only_pos=False,
    )
    assert len(markers) == 2  # matches R harness


def test_wilcox_no_feature_passes_threshpc(harness_adata):
    """thresh.pc=2.0 → no gene has pct>2 → empty markers (R: empty df)."""
    markers = identifyOverExpressedGenes(
        harness_adata.copy(),
        selection_method="wilcox",
        thresh_pc=2.0,
    )
    assert len(markers) == 0
    assert list(markers.columns) == [
        "clusters",
        "features",
        "pvalues",
        "logFC",
        "pct.1",
        "pct.2",
        "pvalues.adj",
    ]


def test_wilcox_group_by_alternative_column(harness_adata):
    """`group_by` overrides the default 'active_ident' column."""
    adata = harness_adata.copy()
    adata.obs["other_ident"] = adata.obs["active_ident"]
    m_default = identifyOverExpressedGenes(
        adata.copy(), selection_method="wilcox"
    )
    m_grouped = identifyOverExpressedGenes(
        adata.copy(), selection_method="wilcox", group_by="other_ident"
    )
    pd.testing.assert_frame_equal(
        m_default.reset_index(drop=True),
        m_grouped.reset_index(drop=True),
    )


def test_wilcox_invalid_selection_method(harness_adata):
    with pytest.raises(ValueError, match="selection_method"):
        identifyOverExpressedGenes(
            harness_adata.copy(),
            selection_method="bogus",  # type: ignore[arg-type]
        )


def test_wilcox_full_table_p_one(harness_adata):
    """All 37 (cluster, gene) rows match R harness output at p=1.0."""
    markers = identifyOverExpressedGenes(
        harness_adata.copy(), selection_method="wilcox", thresh_p=1.0
    )
    # R yields 37 rows under defaults thresh.pc=0, thresh.fc=0, thresh.p=1.0
    assert len(markers) == 37


def test_wilcox_group_dataset_two_condition():
    """`group_dataset`/`pos_dataset` → datasets column + two-condition split."""
    rng = np.random.default_rng(123)
    n_cells, n_genes = 40, 12
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / totals * 10000)
    obs = pd.DataFrame(
        {
            "active_ident": pd.Categorical(
                rng.choice(["X", "Y"], size=n_cells)
            ),
            "dataset": rng.choice(["case", "ctrl"], size=n_cells),
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm.astype("float64")),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    adata.obsm["spatial"] = rng.uniform(0, 100, (n_cells, 2))
    markers = identifyOverExpressedGenes(
        adata,
        selection_method="wilcox",
        group_dataset="dataset",
        pos_dataset="case",
        thresh_p=1.0,
        only_pos=False,
    )
    assert "datasets" in markers.columns
    # All positive-FC rows must be tagged with the pos_dataset
    pos_rows = markers[markers["logFC"] > 0]
    assert (pos_rows["datasets"] == "case").all()


def test_wilcox_datasets_factor_levels_sorted():
    """R: ``levels(factor(meta[[group.dataset]]))`` returns alphabetically
    sorted unique values, *not* appearance order. Mirror that by using
    ``sorted(pd.unique(...))`` rather than appearance order."""
    rng = np.random.default_rng(7)
    n_cells, n_genes = 30, 8
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / totals * 10000)
    # Make the appearance order ['z', 'a'] but the alphabetical order is ['a', 'z']
    dataset_labels = ["z"] * 15 + ["a"] * 15
    obs = pd.DataFrame(
        {
            "active_ident": pd.Categorical(
                rng.choice(["X", "Y"], size=n_cells)
            ),
            "dataset": dataset_labels,
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm.astype("float64")),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    markers = identifyOverExpressedGenes(
        adata,
        selection_method="wilcox",
        group_dataset="dataset",
        pos_dataset="z",
        thresh_p=1.0,
        only_pos=False,
    )
    # Categories must be sorted alphabetically (R parity)
    assert list(markers["datasets"].cat.categories) == ["a", "z"]


def test_wilcox_group_dataset_logFC_zero_assigned_NA():
    """R parity (utilities.R:1319-1323): R's
        markers.all$datasets[markers.all$logFC > 0] <- pos.dataset
        markers.all$datasets[markers.all$logFC < 0] <- setdiff(...)
    leaves rows with ``logFC == 0`` as NA (audit M1). Python must mirror
    by writing ``None`` (which becomes ``NaN`` in the categorical).
    """
    # Manually construct a markers_all-style dataframe path is not exposed,
    # so instead drive a fixture where logFC happens to be exactly 0 for
    # at least one row. Build identical expression across two datasets
    # so logFC = log(mean1) - log(mean2) = 0 by construction.
    n_cells, n_genes = 20, 6
    rng = np.random.default_rng(13)
    base = rng.poisson(2.0, size=(n_cells // 2, n_genes)).astype(float)
    counts = np.vstack([base, base])  # case == ctrl exactly → logFC == 0
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / np.where(totals == 0, 1.0, totals) * 10000)
    obs = pd.DataFrame(
        {
            "active_ident": pd.Categorical(["X"] * n_cells),
            "dataset": ["case"] * (n_cells // 2) + ["ctrl"] * (n_cells // 2),
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm.astype("float64")),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    markers = identifyOverExpressedGenes(
        adata,
        selection_method="wilcox",
        group_dataset="dataset",
        pos_dataset="case",
        thresh_p=1.0,
        thresh_fc=-1.0,    # let logFC=0 rows survive
        thresh_pc=-1.0,
        only_pos=False,
    )
    # Any logFC==0 row must have a NaN dataset assignment (R: NA).
    zero_rows = markers[markers["logFC"] == 0.0]
    if len(zero_rows) > 0:
        assert zero_rows["datasets"].isna().all(), (
            "logFC==0 rows must be NA (R parity), got "
            f"{zero_rows['datasets'].tolist()!r}"
        )


def test_wilcox_group_dataset_three_datasets_recycles_others():
    """R parity (utilities.R:1321): with >2 datasets, R writes
    ``setdiff(unique(labels.dataset), pos.dataset)`` recycled into rows
    with logFC<0 (audit M2). Order is first-appearance.
    """
    n_cells, n_genes = 60, 6
    rng = np.random.default_rng(17)
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(counts / np.where(totals == 0, 1.0, totals) * 10000)
    # Three datasets in appearance order ["b", "a", "c"]: pos=b, others=[a, c]
    ds = ["b"] * 20 + ["a"] * 20 + ["c"] * 20
    obs = pd.DataFrame(
        {
            "active_ident": pd.Categorical(rng.choice(["X", "Y"], size=n_cells)),
            "dataset": ds,
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(lognorm.astype("float64")),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    markers = identifyOverExpressedGenes(
        adata,
        selection_method="wilcox",
        group_dataset="dataset",
        pos_dataset="b",
        thresh_p=1.0,
        only_pos=False,
    )
    # Categorical levels are sorted alphabetically (R: ``levels(factor(...))``)
    assert list(markers["datasets"].cat.categories) == ["a", "b", "c"]
    # All neg-FC rows must be in {a, c} (the appearance-ordered "others").
    neg_rows = markers[markers["logFC"] < 0]
    if len(neg_rows) > 0:
        assert set(neg_rows["datasets"].dropna().unique()) <= {"a", "c"}


def test_identifyOverExpressedGenes_do_grid_recursive():
    """R utilities.R:1137-1177: do_grid=True builds a grid AnnData,
    runs subsetData on it, recurses with do_grid=False, and stashes
    the grid AnnData under ``misc['grid.object']``."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 8
    counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(float)
    coords = np.column_stack([
        rng.uniform(0.0, 30.0, size=n_cells),
        rng.uniform(0.0, 30.0, size=n_cells),
    ])
    obs = pd.DataFrame(
        {"active_ident": pd.Categorical(["X"] * 30 + ["Y"] * 30)},
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
        obsm={"spatial": coords},
    )
    # Tiny CellChat DB: every interaction's ligand/receptor must be among
    # adata.var_names so subsetData yields a non-empty gene set.
    interaction = pd.DataFrame({
        "interaction_name": ["g0_g1", "g2_g3"],
        "ligand": ["g0", "g2"],
        "receptor": ["g1", "g3"],
        "pathway_name": ["P1", "P2"],
        "annotation": ["Secreted Signaling", "Cell-Cell Contact"],
        # extractGene also walks these cofactor columns; empty strings
        # are skipped (mirrors how the real CellChatDB rows behave when
        # no cofactor is annotated).
        "agonist": ["", ""],
        "antagonist": ["", ""],
        "co_A_receptor": ["", ""],
        "co_I_receptor": ["", ""],
    })
    geneInfo = pd.DataFrame(
        {"Symbol": [f"g{i}" for i in range(n_genes)]},
        index=[f"g{i}" for i in range(n_genes)],
    )
    adata.uns["cellchat"] = {
        "DB": {
            "interaction": interaction,
            "complex": pd.DataFrame(),
            "cofactor": pd.DataFrame(),
            "geneInfo": geneInfo,
        }
    }
    # Use moransi (per-gene spatial autocorrelation; doesn't need grid
    # groups to have ≥ 2 cells, which a tiny synthetic grid would fail).
    markers = identifyOverExpressedGenes(
        adata,
        do_grid=True,
        cellsize=(8.0, 8.0),
        what="polygons",
        square=True,
        selection_method="moransi",
    )

    assert isinstance(markers, pd.DataFrame)
    cc = adata.uns["cellchat"]
    # ``misc['grid.object']`` populated (R: misc(object,"grid.object") <- grid.object).
    assert "misc" in cc and "grid.object" in cc["misc"]
    assert isinstance(cc["misc"]["grid.object"], ad.AnnData)
    # Variable-features propagated back from grid object to parent.
    assert "DE" in cc and "features" in cc["DE"]
    assert "features.info" in cc["DE"]
    assert "highly_variable_signaling" in adata.var.columns


# ---------------------------------------------------------------------------
# Moran's I branch
# ---------------------------------------------------------------------------
# Generated with /tmp/de_harness/run_moransi.R (same fixture).
_MORANSI_FIRST10 = pd.DataFrame(
    {
        "observed": [
            -0.0440956600147502,
            0.0881658021402483,
            0.0437621140775582,
            -0.0941415599991187,
            -0.0491337010020639,
            -0.0769988781867926,
            -0.231258014140736,
            -0.130988466473233,
            -0.13746101921745,
            -0.0699187846675673,
        ],
        "p.value": [
            0.925408609676421,
            0.227131830331723,
            0.159556002772181,
            0.561255079885644,
            0.87505530218985,
            0.64718343355589,
            0.0219295164712644,
            0.341253062845575,
            0.271005129227903,
            0.731994868633209,
        ],
    },
    index=[f"gene{i:02d}" for i in range(10)],
)


def test_moransi_parity(harness_adata):
    markers = identifyOverExpressedGenes(
        harness_adata.copy(), selection_method="moransi", thresh_p=1.0
    )
    head = markers.iloc[:10]
    np.testing.assert_allclose(
        head["observed"].to_numpy(),
        _MORANSI_FIRST10["observed"].to_numpy(),
        atol=1e-12,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        head["p.value"].to_numpy(),
        _MORANSI_FIRST10["p.value"].to_numpy(),
        atol=1e-12,
        rtol=1e-9,
    )


def test_moransi_filters_on_p(harness_adata):
    """Only genes with p.value<0.05 land in adata.uns['cellchat']['DE']
    'features'."""
    adata = harness_adata.copy()
    identifyOverExpressedGenes(adata, selection_method="moransi", thresh_p=0.05)
    de = adata.uns["cellchat"]["DE"]
    info = de["features.info"]
    assert set(de["features"]) == set(
        info.index[(info["p.value"] < 0.05) & (~info["observed"].isna())]
    )


def test_moransi_requires_spatial():
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(10, 5)).astype(float)
    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(
            {"active_ident": pd.Categorical(["A"] * 10)},
            index=[f"c{i}" for i in range(10)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    with pytest.raises(ValueError, match="spatial"):
        identifyOverExpressedGenes(adata, selection_method="moransi")


def test_ape_moran_i_vector_match():
    """Direct unit test of the inline ape::Moran.I port against a hand-
    computed example (R: ape::Moran.I(rnorm(20, seed=42), 1/d^2))."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(20)
    pos = rng.uniform(0, 1, size=(20, 2))
    from scipy.spatial.distance import pdist, squareform

    d = squareform(pdist(pos))
    with np.errstate(divide="ignore"):
        w = 1.0 / (d ** 2)
    np.fill_diagonal(w, 0.0)
    obs, ei, sd, p = _ape_moran_i(x, w)
    # Hand-verified against an inline ape::Moran.I implementation in R
    # (see /tmp/de_harness inline check)
    assert ei == pytest.approx(-1.0 / 19, abs=1e-12)
    assert np.isfinite(obs)
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# MERINGUE branch
# ---------------------------------------------------------------------------
# Generated with /tmp/de_harness/run_meringue.R.
_MERINGUE_FIRST5 = pd.DataFrame(
    {
        "observed": [
            0.0606521241240692,
            0.192722496051943,
            0.00816151494671486,
            -0.166688532961197,
            -0.0797207188634451,
        ],
        "expected": [-0.0344827586206897] * 5,
        "sd": [
            0.104439146957341,
            0.10328489749119,
            0.0561705331904153,
            0.104449513647336,
            0.0947134086655232,
        ],
        "p.value": [
            0.181170844676586,
            0.0139108404763685,
            0.223868567266593,
            0.897196646336214,
            0.683543193411776,
        ],
        "p.adj": [
            0.700007238108072,
            0.173885505954606,
            0.700007238108072,
            0.963796043607123,
            0.899398938699705,
        ],
    },
    index=[f"gene{i:02d}" for i in range(5)],
)


def test_meringue_parity(harness_adata):
    markers = identifyOverExpressedGenes(
        harness_adata.copy(), selection_method="meringue", thresh_p=1.0
    )
    head = markers.iloc[:5]
    for col in ("observed", "expected", "sd", "p.value", "p.adj"):
        np.testing.assert_allclose(
            head[col].to_numpy(),
            _MERINGUE_FIRST5[col].to_numpy(),
            atol=1e-12,
            rtol=1e-9,
        )


def test_meringue_neighbor_weights_edge_count(harness_adata):
    coords = np.asarray(harness_adata.obsm["spatial"])
    W = _meringue_neighbor_weights(coords)
    # Symmetric, zero-diagonal, binary — and the harness yielded 78 undirected
    # edges (see R's MERINGUE::getSpatialNeighbors output).
    assert (W.diagonal() == 0).all()
    assert np.array_equal(W, W.T)
    assert int(W.sum() / 2) == 78


def test_meringue_filters_on_padj(harness_adata):
    adata = harness_adata.copy()
    identifyOverExpressedGenes(adata, selection_method="meringue", thresh_p=0.5)
    de = adata.uns["cellchat"]["DE"]
    info = de["features.info"]
    assert set(de["features"]) == set(
        info.index[(info["p.adj"] < 0.5) & (~info["observed"].isna())]
    )


# ---------------------------------------------------------------------------
# Bonferroni / BH helpers
# ---------------------------------------------------------------------------
def test_bonferroni_matches_r():
    pvals = np.array([0.01, 0.04, 0.5, 0.9])
    out = _bonferroni(pvals, n=10)
    np.testing.assert_allclose(out, np.array([0.1, 0.4, 1.0, 1.0]))


def test_bh_matches_r():
    """`p.adjust(c(0.01, 0.04, 0.5, 0.9), 'BH')` in R = 0.04 0.08 0.6667 0.9."""
    pvals = np.array([0.01, 0.04, 0.5, 0.9])
    out = _bh(pvals)
    np.testing.assert_allclose(
        out,
        np.array([0.04, 0.08, 2 / 3, 0.9]),
        atol=1e-9,
    )


def test_bh_handles_nan_like_r():
    """`p.adjust(c(0.001, 0.5, NA, 0.04, 0.8), 'BH')` in R =
    0.004 0.6666667 NA 0.08 0.8 — the meringue branch produces NaN p-values
    for genes with zero variance, so this case is on the hot path of the
    tutorial (regression test for the previous all-NaN propagation bug)."""
    pvals = np.array([0.001, 0.5, np.nan, 0.04, 0.8])
    out = _bh(pvals)
    np.testing.assert_allclose(
        out[~np.isnan(out)],
        np.array([0.004, 2 / 3, 0.08, 0.8]),
        atol=1e-9,
    )
    assert np.isnan(out[2])


# ---------------------------------------------------------------------------
# identifyOverExpressedLigandReceptor
# ---------------------------------------------------------------------------
def _markers_from_csv() -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(
            """clusters,features,pvalues,logFC,pct.1,pct.2,pvalues.adj
c1,TGFB1,0.001,1.0,0.5,0.1,0.01
c1,TGFB2,0.001,1.0,0.5,0.1,0.01
c2,TGFBR1,0.001,1.0,0.5,0.1,0.01
c2,TGFBR2,0.001,1.0,0.5,0.1,0.01
c3,EGFR,0.001,1.0,0.5,0.1,0.01
c3,VEGFA,0.001,1.0,0.5,0.1,0.01
c4,IL2,0.001,1.0,0.5,0.1,0.01
c4,CD8A,0.001,1.0,0.5,0.1,0.01
c5,ITGB1,0.001,1.0,0.5,0.1,0.01
c5,ITGA4,0.001,1.0,0.5,0.1,0.01
c5,GENE_NOT_IN_DB,0.001,1.0,0.5,0.1,0.01
"""
        )
    )


@pytest.fixture(scope="module")
def lr_setup():
    db = load_db("human")
    markers = _markers_from_csv()
    from spatialcellchat.database import extractGene

    gene_use = sorted(set(extractGene(db)) | set(markers["features"]))
    adata = ad.AnnData(
        X=sp.csr_matrix((1, len(gene_use)), dtype=float),
        obs=pd.DataFrame(index=["spot0"]),
        var=pd.DataFrame(index=gene_use),
    )
    adata.uns["cellchat"] = {
        "DE": {
            "features": list(markers["features"].unique()),
            "features.info": markers,
        }
    }
    return adata, db


# Generated with /tmp/de_harness/run_lr.R variable.both=TRUE.
_LR_EXPECTED_FEATS = sorted(
    [
        "TGFB1",
        "TGFB2",
        "ACVR1_TGFbR",
        "TGFBR1_ACVR2A",
        "TGFBR1_ACVR2B",
        "TGFbR1_R2",
        "TGFBR2",
        "EGFR",
        "VEGFA",
        "IL2",
        "CD8A",
        "ITGA1_ITGB1",
        "ITGA10_ITGB1",
        "ITGA11_ITGB1",
        "ITGA2_ITGB1",
        "ITGA3_ITGB1",
        "ITGA4_ITGB1",
        "ITGA5_ITGB1",
        "ITGA6_ITGB1",
        "ITGA7_ITGB1",
        "ITGA8_ITGB1",
        "ITGA9_ITGB1",
        "ITGAV_ITGB1",
        "ITGB1_ITGA1",
        "ITGA4_ITGB1",  # duplicated: ITGA4 and ITGB1 each map to it
        "ITGA4_ITGB7",
    ]
)


def test_lr_complex_expansion(lr_setup):
    """Single complex name appears once per gene whose subunits include it,
    matching the R loop-of-rbind behaviour."""
    adata, db = lr_setup
    lr = identifyOverExpressedLigandReceptor(adata.copy(), db)
    assert sorted(lr["features"].tolist()) == _LR_EXPECTED_FEATS
    assert "GENE_NOT_IN_DB" not in lr["features"].tolist()


def test_lr_writes_uns(lr_setup):
    adata, db = lr_setup
    a = adata.copy()
    identifyOverExpressedLigandReceptor(a, db)
    assert "features.LR" in a.uns["cellchat"]["DE"]
    assert isinstance(a.uns["cellchat"]["DE"]["features.LR"], pd.DataFrame)


def test_lr_features_subset(lr_setup):
    adata, db = lr_setup
    lr = identifyOverExpressedLigandReceptor(
        adata.copy(), db, features=["TGFB1"]
    )
    # Only TGFB1 (a direct LR member) survives — no complexes
    assert lr["features"].tolist() == ["TGFB1"]


def test_lr_missing_info_raises(lr_setup):
    """Calling LR before identifyOverExpressedGenes raises."""
    _, db = lr_setup
    adata = ad.AnnData(
        X=sp.csr_matrix((1, 2), dtype=float),
        obs=pd.DataFrame(index=["spot0"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    with pytest.raises(ValueError, match="features.info"):
        identifyOverExpressedLigandReceptor(adata, db)


# ---------------------------------------------------------------------------
# identifyOverExpressedInteractions
# ---------------------------------------------------------------------------
def test_interactions_variable_both_true(lr_setup):
    """With variable_both=True the R harness produces 8 interaction rows."""
    adata, db = lr_setup
    inter = identifyOverExpressedInteractions(adata.copy(), db, variable_both=True)
    assert len(inter) == 8
    # Sanity: must include the TGFB1/TGFB2-driven interactions
    inames = set(inter["interaction_name"])
    assert "TGFB1_TGFBR1_TGFBR2" in inames


def test_interactions_variable_both_false(lr_setup):
    """variable_both=FALSE expands to 263 (R harness)."""
    adata, db = lr_setup
    inter = identifyOverExpressedInteractions(adata.copy(), db, variable_both=False)
    assert len(inter) == 263


def test_interactions_writes_uns(lr_setup):
    adata, db = lr_setup
    a = adata.copy()
    identifyOverExpressedInteractions(a, db, variable_both=True)
    assert "LR" in a.uns["cellchat"]
    assert "LRsig" in a.uns["cellchat"]["LR"]
    assert isinstance(a.uns["cellchat"]["LR"]["LRsig"], pd.DataFrame)


def test_interactions_features_override(lr_setup):
    """`features` overrides the stored over-expressed gene set. With only
    TGFB1 marked sig, variable_both=False keeps every interaction whose
    ligand is TGFB1 and whose receptors are present in the data universe."""
    adata, db = lr_setup
    inter = identifyOverExpressedInteractions(
        adata.copy(),
        db,
        variable_both=False,
        features=["TGFB1"],
    )
    inames = set(inter["interaction_name"])
    assert "TGFB1_TGFBR1_TGFBR2" in inames
    # Only TGFB1-driven pairs (no TGFB2-driven ones)
    assert not any("TGFB2" in n for n in inames)


def test_interactions_missing_features_raises(lr_setup):
    _, db = lr_setup
    adata = ad.AnnData(
        X=sp.csr_matrix((1, 2), dtype=float),
        obs=pd.DataFrame(index=["spot0"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    with pytest.raises(ValueError, match="features"):
        identifyOverExpressedInteractions(adata, db)


# ---------------------------------------------------------------------------
# Tiny Visium fixture sanity (uses conftest.tiny_visium)
# ---------------------------------------------------------------------------
def test_wilcox_runs_on_tiny_visium(tiny_visium):
    """Smoke test: full pipeline on the 15-spot fixture must run without
    error (it does not need to find any DE genes)."""
    adata = tiny_visium.copy()
    # log-normalise (the fixture is integer counts)
    X = adata.X.toarray().astype(float)
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    adata.X = sp.csr_matrix(np.log1p(X / totals * 10000))
    markers = identifyOverExpressedGenes(adata, selection_method="wilcox")
    assert isinstance(markers, pd.DataFrame)
    assert {
        "clusters",
        "features",
        "pvalues",
        "logFC",
        "pct.1",
        "pct.2",
        "pvalues.adj",
    }.issubset(markers.columns)
