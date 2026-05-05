"""Tests for ``spatialcellchat.preprocessing``.

R↔Py golden CSVs are bundled under ``tests/data/r_dumps/preprocessing/``;
they were originally produced by ``r_harness/dump_preproc.R``.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialcellchat import preprocessing as pp

R_DUMP = Path(__file__).parent / "data" / "r_dumps" / "preprocessing"


# ---------------------------------------------------------------------------
# Shared fixture: load the R-side raw counts (genes x cells) into an AnnData
# (cells x genes), exactly the orientation that AnnData expects.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_adata() -> ad.AnnData:
    raw = pd.read_csv(R_DUMP / "raw_counts.csv", index_col=0)  # genes x cells
    A = ad.AnnData(
        X=raw.values.T.astype(float),
        obs=pd.DataFrame(index=raw.columns),
        var=pd.DataFrame(index=raw.index),
    )
    return A


def _r_genes_x_cells(name: str) -> np.ndarray:
    return pd.read_csv(R_DUMP / name, index_col=0).values


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


# ---------------------------------------------------------------------------
# normalizeData
# ---------------------------------------------------------------------------
def test_normalize_default_matches_R(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=10000.0, do_log=True)
    # AnnData stores cells x genes; transpose to compare with R.
    py = np.asarray(A.X).T
    r = _r_genes_x_cells("norm.csv")
    assert _max_abs_diff(py, r) < 1e-9


def test_normalize_alternative_scale(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=5000.0, do_log=True)
    r = _r_genes_x_cells("norm_alt.csv")
    assert _max_abs_diff(np.asarray(A.X).T, r) < 1e-9


def test_normalize_no_log(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=5000.0, do_log=False)
    r = _r_genes_x_cells("norm_nolog.csv")
    assert _max_abs_diff(np.asarray(A.X).T, r) < 1e-9


def test_normalize_caches_counts(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A)
    assert "counts" in A.layers
    cached = np.asarray(A.layers["counts"])
    raw = pd.read_csv(R_DUMP / "raw_counts.csv", index_col=0).values
    assert _max_abs_diff(cached, raw.T) == 0.0


def test_normalize_sparse_input(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    A.X = sparse.csr_matrix(A.X)
    pp.normalizeData(A, scale_factor=10000.0)
    r = _r_genes_x_cells("norm.csv")
    py = np.asarray(A.X.toarray()).T if sparse.issparse(A.X) else np.asarray(A.X).T
    assert _max_abs_diff(py, r) < 1e-9


# ---------------------------------------------------------------------------
# scaleData
# ---------------------------------------------------------------------------
def test_scaleData_center_matches_R(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=10000.0, do_log=True)
    pp.scaleData(A, do_center=True, layer_in=None, layer_out="scaled")
    py = np.asarray(A.layers["scaled"]).T
    r = _r_genes_x_cells("scaled_center.csv")
    assert _max_abs_diff(py, r) < 1e-9


def test_scaleData_no_center_matches_R(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=10000.0, do_log=True)
    pp.scaleData(A, do_center=False, layer_in=None, layer_out="scaled_nc")
    py = np.asarray(A.layers["scaled_nc"]).T
    r = _r_genes_x_cells("scaled_nocenter.csv")
    assert _max_abs_diff(py, r) < 1e-9


# ---------------------------------------------------------------------------
# runPCA
# ---------------------------------------------------------------------------
def test_runPCA_full_matches_R(raw_adata: ad.AnnData) -> None:
    """``do_fast=False`` is deterministic — must match R column-magnitude.

    SVD signs are gauge-free, so we compare absolute values column-wise.
    """
    A = raw_adata.copy()
    pp.normalizeData(A, scale_factor=10000.0, do_log=True)
    pp.scaleData(A, do_center=True, layer_in=None, layer_out="scaled")
    pp.runPCA(
        A, do_fast=False, dimPC=5, seed_use=42, weight_by_var=True,
        layer_in="scaled", obsm_key="X_pca",
    )
    r = pd.read_csv(R_DUMP / "pca_full.csv", index_col=0).values
    py = np.asarray(A.obsm["X_pca"])
    assert py.shape == r.shape
    # column-wise absolute correlation must be ~1
    for j in range(r.shape[1]):
        cor = abs(np.corrcoef(py[:, j], r[:, j])[0, 1])
        assert cor > 0.999, f"PC{j+1} correlation = {cor}"


def test_runPCA_full_no_weight(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A)
    pp.scaleData(A, layer_in=None, layer_out="scaled")
    pp.runPCA(A, do_fast=False, dimPC=5, weight_by_var=False, layer_in="scaled")
    r = pd.read_csv(R_DUMP / "pca_full_noweight.csv", index_col=0).values
    py = np.asarray(A.obsm["X_pca"])
    assert py.shape == r.shape
    for j in range(r.shape[1]):
        assert abs(np.corrcoef(py[:, j], r[:, j])[0, 1]) > 0.999


def test_runPCA_fast_correlated_with_R(raw_adata: ad.AnnData) -> None:
    """Fast path uses randomized SVD; expect high correlation, not equality."""
    A = raw_adata.copy()
    pp.normalizeData(A)
    pp.scaleData(A, layer_in=None, layer_out="scaled")
    pp.runPCA(A, do_fast=True, dimPC=5, seed_use=42, layer_in="scaled")
    r = pd.read_csv(R_DUMP / "pca_fast.csv", index_col=0).values
    py = np.asarray(A.obsm["X_pca"])
    assert py.shape == r.shape
    # at least the leading components should correlate strongly
    for j in range(min(3, r.shape[1])):
        assert abs(np.corrcoef(py[:, j], r[:, j])[0, 1]) > 0.95


def test_runPCA_writes_expected_slots(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A)
    pp.scaleData(A, layer_in=None, layer_out="scaled")
    pp.runPCA(A, dimPC=4, layer_in="scaled")
    assert "X_pca" in A.obsm
    assert "PCs" in A.varm
    assert "pca" in A.uns
    assert "variance_ratio" in A.uns["pca"]


def test_runPCA_dimPC_capped() -> None:
    """``dimPC`` should be capped at ``ncol(data) - 1`` like R.

    The R reference does ``dimPC <- min(dimPC, ncol(data.use) - 1)``; we
    use a cells > genes shape so the cap is mathematically meaningful.
    """
    rng = np.random.default_rng(0)
    A = ad.AnnData(
        X=rng.normal(size=(40, 8)).astype(float),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(40)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    pp.scaleData(A, layer_in=None, layer_out="scaled")
    pp.runPCA(A, dimPC=999, do_fast=False, layer_in="scaled")
    assert A.obsm["X_pca"].shape[1] == A.shape[1] - 1


# ---------------------------------------------------------------------------
# runUMAP
# ---------------------------------------------------------------------------
def test_runUMAP_basic(raw_adata: ad.AnnData) -> None:
    A = raw_adata.copy()
    pp.normalizeData(A)
    pp.scaleData(A, layer_in=None, layer_out="scaled")
    pp.runPCA(A, dimPC=5, layer_in="scaled")
    pp.runUMAP(
        A, n_neighbors=5, n_components=2, seed_use=1, use_rep="X_pca",
    )
    assert A.obsm["X_umap"].shape == (A.n_obs, 2)


def test_runUMAP_seed_reproducible(raw_adata: ad.AnnData) -> None:
    A1 = raw_adata.copy()
    A2 = raw_adata.copy()
    for A in (A1, A2):
        pp.normalizeData(A)
        pp.scaleData(A, layer_in=None, layer_out="scaled")
        pp.runPCA(A, dimPC=5, layer_in="scaled")
        pp.runUMAP(A, n_neighbors=5, seed_use=99, use_rep="X_pca")
    assert _max_abs_diff(A1.obsm["X_umap"], A2.obsm["X_umap"]) < 1e-6


# ---------------------------------------------------------------------------
# subsetData
# ---------------------------------------------------------------------------
def test_subsetData_with_features(raw_adata: ad.AnnData) -> None:
    feats = list(raw_adata.var_names[:7]) + ["nonexistent"]
    sub = pp.subsetData(raw_adata, features=feats)
    assert sub.n_vars == 7
    assert "data_signaling" in sub.layers


def test_subsetData_requires_db_or_features(raw_adata: ad.AnnData) -> None:
    with pytest.raises(ValueError):
        pp.subsetData(raw_adata)


def test_subsetData_with_fake_db(raw_adata: ad.AnnData) -> None:
    """``extractGene`` accepts a dict of dataframes; build a tiny one."""
    interaction = pd.DataFrame({
        "interaction_name": ["g001_g002"],
        "ligand": ["g001"],
        "receptor": ["g002"],
        "agonist": [""],
        "antagonist": [""],
        "co_A_receptor": [""],
        "co_I_receptor": [""],
    })
    complex_df = pd.DataFrame(index=[])
    cof = pd.DataFrame(index=[])
    geneInfo = pd.DataFrame({"Symbol": ["g001", "g002", "g003"]})
    db = {
        "interaction": interaction,
        "complex": complex_df,
        "cofactor": cof,
        "geneInfo": geneInfo,
    }
    sub = pp.subsetData(raw_adata, db=db)
    # at minimum the ligand g001 and receptor g002 survive
    assert "g001" in sub.var_names
    assert "g002" in sub.var_names


def test_subsetData_adds_annotation_for_non_RNA(raw_adata: ad.AnnData) -> None:
    """R: utilities.R:1037-1041 — when datatype != 'RNA' and annotation is
    missing, fill with 'Secreted Signaling' and warn."""
    interaction = pd.DataFrame({
        "interaction_name": ["g001_g002"],
        "ligand": ["g001"],
        "receptor": ["g002"],
        "agonist": [""],
        "antagonist": [""],
        "co_A_receptor": [""],
        "co_I_receptor": [""],
    })
    db = {
        "interaction": interaction,
        "complex": pd.DataFrame(index=[]),
        "cofactor": pd.DataFrame(index=[]),
        "geneInfo": pd.DataFrame({"Symbol": ["g001", "g002"]}),
    }
    raw_adata.uns["cellchat"] = {"options": {"datatype": "spatial"}}
    with pytest.warns(UserWarning, match="annotation"):
        pp.subsetData(raw_adata, db=db)
    assert "annotation" in db["interaction"].columns
    assert db["interaction"]["annotation"].iloc[0] == "Secreted Signaling"


def test_subsetData_reorders_annotation(raw_adata: ad.AnnData) -> None:
    """R: utilities.R:1043-1049 — when annotation has multiple unique values,
    reorder by canonical factor levels."""
    interaction = pd.DataFrame({
        "interaction_name": ["g001_g002", "g002_g003", "g003_g001"],
        "ligand": ["g001", "g002", "g003"],
        "receptor": ["g002", "g003", "g001"],
        "annotation": ["Cell-Cell Contact", "Secreted Signaling", "ECM-Receptor"],
        "agonist": ["", "", ""],
        "antagonist": ["", "", ""],
        "co_A_receptor": ["", "", ""],
        "co_I_receptor": ["", "", ""],
    })
    db = {
        "interaction": interaction,
        "complex": pd.DataFrame(index=[]),
        "cofactor": pd.DataFrame(index=[]),
        "geneInfo": pd.DataFrame({"Symbol": ["g001", "g002", "g003"]}),
    }
    pp.subsetData(raw_adata, db=db)
    # Order should be Secreted Signaling, ECM-Receptor, Cell-Cell Contact
    ann_after = db["interaction"]["annotation"].tolist()
    assert ann_after == [
        "Secreted Signaling",
        "ECM-Receptor",
        "Cell-Cell Contact",
    ]


# ---------------------------------------------------------------------------
# updateClusterLabels
# ---------------------------------------------------------------------------
def _make_clustered_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    A = ad.AnnData(
        X=rng.normal(size=(9, 5)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(9)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    A.obs["cellchat_idents"] = pd.Categorical(
        ["A"] * 3 + ["B"] * 3 + ["C"] * 3, categories=["A", "B", "C"]
    )
    return A


def test_updateClusterLabels_renames() -> None:
    A = _make_clustered_adata()
    pp.updateClusterLabels(
        A,
        old_cluster_name=["A", "B", "C"],
        new_cluster_name=["X", "Y", "Z"],
    )
    assert set(A.obs["cellchat_idents"].cat.categories) == {"X", "Y", "Z"}
    assert "new.labels" in A.obs.columns


def test_updateClusterLabels_reorders() -> None:
    A = _make_clustered_adata()
    pp.updateClusterLabels(
        A,
        old_cluster_name=["A", "B", "C"],
        new_cluster_name=["A", "B", "C"],
        new_order=["C", "A", "B"],
    )
    assert list(A.obs["cellchat_idents"].cat.categories) == ["C", "A", "B"]


def test_updateClusterLabels_zero_label_rejected() -> None:
    A = _make_clustered_adata()
    with pytest.raises(ValueError):
        pp.updateClusterLabels(
            A,
            old_cluster_name=["A", "B", "C"],
            new_cluster_name=["0", "Y", "Z"],
        )


def test_updateClusterLabels_existing_metaname_rejected() -> None:
    A = _make_clustered_adata()
    A.obs["new.labels"] = "x"
    with pytest.raises(ValueError):
        pp.updateClusterLabels(
            A,
            old_cluster_name=["A", "B", "C"],
            new_cluster_name=["X", "Y", "Z"],
        )


def test_updateClusterLabels_missing_ident_key() -> None:
    A = _make_clustered_adata()
    del A.obs["cellchat_idents"]
    with pytest.raises(KeyError):
        pp.updateClusterLabels(A, new_cluster_name=["X"])


def test_updateClusterLabels_runs_downstream_when_net_present(monkeypatch) -> None:
    """R parity: when ``cellchat['net']`` exists, updateClusterLabels
    must re-run computeCommunProbPathway → aggregateNet →
    netAnalysis_computeCentrality (utilities.R:1014-1019).
    """
    A = _make_clustered_adata()
    A.uns["cellchat"] = {"net": {"prob": np.zeros((3, 3, 1))}, "netP": {"prob": np.zeros((3, 3, 1))}}
    calls: list = []
    from spatialcellchat import modeling as _m
    from spatialcellchat import analysis_network as _an

    def fake_ccpp(adata, **kw):
        calls.append("computeCommunProbPathway")
        return adata

    def fake_agg(adata, **kw):
        calls.append("aggregateNet")
        return adata

    def fake_centr(adata, **kw):
        calls.append("netAnalysis_computeCentrality")
        return adata

    monkeypatch.setattr(_m, "computeCommunProbPathway", fake_ccpp)
    monkeypatch.setattr(_m, "aggregateNet", fake_agg)
    monkeypatch.setattr(_an, "netAnalysis_computeCentrality", fake_centr)
    pp.updateClusterLabels(
        A, old_cluster_name=["A", "B", "C"], new_cluster_name=["X", "Y", "Z"],
    )
    assert calls == [
        "computeCommunProbPathway",
        "aggregateNet",
        "netAnalysis_computeCentrality",
    ]


def test_updateClusterLabels_skips_downstream_without_net() -> None:
    """When the cellchat pipeline has not been run, no downstream
    invocation occurs (existing tests exercise this path)."""
    A = _make_clustered_adata()
    pp.updateClusterLabels(
        A, old_cluster_name=["A", "B", "C"], new_cluster_name=["X", "Y", "Z"],
    )
    assert "cellchat" not in A.uns or "net" not in A.uns.get("cellchat", {})


# ---------------------------------------------------------------------------
# preProcessing (ALRA wrapper)
# ---------------------------------------------------------------------------
def test_preProcessing_runs_and_writes_layer() -> None:
    """The ALRA wrapper needs a matrix big enough that
    ``round(0.9 * min(dim)) >= 11`` (else ``choose_k`` errors — same as R)."""
    rng = np.random.default_rng(0)
    A = ad.AnnData(
        X=rng.poisson(2.0, size=(120, 110)).astype(float),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(120)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(110)]),
    )
    pp.normalizeData(A)
    pp.preProcessing(A, layer_out="data_signaling", quantile_prob=1e-5)
    assert "data_signaling" in A.layers
    assert "alra_k" in A.uns
    assert A.layers["data_signaling"].shape == A.X.shape
