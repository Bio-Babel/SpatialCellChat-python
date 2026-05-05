"""Tests for the ``projectData`` family in ``preprocessing.py``.

The R reference (``utilities.R:1603..1734``) is exercised through golden
files bundled under ``tests/data/r_dumps/projectdata/``."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialcellchat import preprocessing as pp
from spatialcellchat.preprocessing import (
    _l1NormalizeColumns,
    _l1NormalizeRows,
    _projectAndRecombine,
    _projectFromNetworkRecombine,
    _projectOnNetwork,
    _randomWalkBySolve,
)

R_DUMP = Path(__file__).parent / "data" / "r_dumps" / "projectdata"


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


# ---------------------------------------------------------------------------
# l1 normalisation
# ---------------------------------------------------------------------------
def test_l1NormalizeRows_matches_R() -> None:
    adj = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0).values
    r = pd.read_csv(R_DUMP / "proj_l1r.csv", index_col=0).values
    assert _diff(_l1NormalizeRows(adj), r) < 1e-12


def test_l1NormalizeColumns_matches_R() -> None:
    adj = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0).values
    r = pd.read_csv(R_DUMP / "proj_l1c.csv", index_col=0).values
    assert _diff(_l1NormalizeColumns(adj), r) < 1e-12


# ---------------------------------------------------------------------------
# randomWalkBySolve
# ---------------------------------------------------------------------------
def test_randomWalkBySolve_rows_matches_R() -> None:
    E = pd.read_csv(R_DUMP / "proj_E.csv", index_col=0).values
    adj = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0).values
    r = pd.read_csv(R_DUMP / "proj_rw_rows.csv", index_col=0).values
    py = _randomWalkBySolve(E, adj, alpha=0.5, normalize_adj="rows")
    assert _diff(py, r) < 1e-9


def test_randomWalkBySolve_columns_matches_R() -> None:
    E = pd.read_csv(R_DUMP / "proj_E.csv", index_col=0).values
    adj = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0).values
    r = pd.read_csv(R_DUMP / "proj_rw_cols.csv", index_col=0).values
    py = _randomWalkBySolve(E, adj, alpha=0.5, normalize_adj="columns")
    assert _diff(py, r) < 1e-9


def test_randomWalkBySolve_invalid_normalize() -> None:
    E = np.eye(3)
    A = np.eye(3)
    with pytest.raises(ValueError):
        _randomWalkBySolve(E, A, alpha=0.5, normalize_adj="bogus")


# ---------------------------------------------------------------------------
# projectOnNetwork & projectFromNetworkRecombine
# ---------------------------------------------------------------------------
def test_projectOnNetwork_handles_missing_genes() -> None:
    rng = np.random.default_rng(0)
    E = rng.normal(size=(3, 4))
    rows = ["a", "b", "c"]
    new_features = ["a", "z", "b"]
    out, idx = _projectOnNetwork(E, new_features, rows)
    assert idx == new_features
    assert out.shape == (3, 4)
    assert (out[idx.index("z")] == 0).all()
    assert _diff(out[idx.index("a")], E[0]) == 0.0


def test_projectFromNetworkRecombine_overlay() -> None:
    rng = np.random.default_rng(1)
    orig = rng.normal(size=(4, 3))
    proj = rng.normal(size=(2, 3))
    out = _projectFromNetworkRecombine(
        orig, proj, ["a", "b", "c", "d"], ["a", "c"]
    )
    assert _diff(out[0], proj[0]) == 0.0
    assert _diff(out[2], proj[1]) == 0.0
    assert _diff(out[1], orig[1]) == 0.0
    assert _diff(out[3], orig[3]) == 0.0


# ---------------------------------------------------------------------------
# projectAndRecombine
# ---------------------------------------------------------------------------
def test_projectAndRecombine_matches_R() -> None:
    gE2 = pd.read_csv(R_DUMP / "proj_E2.csv", index_col=0)   # 12 x 6
    adj_df = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0)
    r = pd.read_csv(R_DUMP / "proj_full.csv", index_col=0).values
    out = _projectAndRecombine(
        gE2.values, adj_df.values, alpha=0.5,
        gene_names=list(gE2.index), adj_names=list(adj_df.index),
        normalize_adj="rows",
    )
    assert _diff(out, r) < 1e-9


# ---------------------------------------------------------------------------
# projectData (public)
# ---------------------------------------------------------------------------
def test_projectData_writes_layer() -> None:
    gE2 = pd.read_csv(R_DUMP / "proj_E2.csv", index_col=0)   # genes x cells
    adj_df = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0)
    A = ad.AnnData(
        X=gE2.values.T.astype(float),
        obs=pd.DataFrame(index=gE2.columns),
        var=pd.DataFrame(index=gE2.index),
    )
    A.layers["data_signaling"] = A.X.copy()
    pp.projectData(A, adj_df, alpha=0.5, normalizeAdjMatrix="rows")
    assert "projected" in A.layers
    r = pd.read_csv(R_DUMP / "proj_full.csv", index_col=0).values
    py = np.asarray(A.layers["projected"]).T
    assert _diff(py, r) < 1e-9


def test_projectData_alpha_validation() -> None:
    A = ad.AnnData(X=np.eye(3))
    A.layers["data_signaling"] = A.X.copy()
    with pytest.raises(ValueError):
        pp.projectData(A, np.eye(3), alpha=-0.1, adj_gene_names=list(A.var_names))
    with pytest.raises(ValueError):
        pp.projectData(A, np.eye(3), alpha=1.5, adj_gene_names=list(A.var_names))


def test_projectData_zero_row_rejected() -> None:
    A = ad.AnnData(X=np.eye(3))
    A.var_names = ["a", "b", "c"]
    A.layers["data_signaling"] = A.X.copy()
    bad = np.array([[1, 2, 0], [0, 0, 0], [3, 4, 5]], dtype=float)
    with pytest.raises(ValueError):
        pp.projectData(A, bad, alpha=0.5, adj_gene_names=["a", "b", "c"])


def test_projectData_requires_gene_names_for_ndarray() -> None:
    A = ad.AnnData(X=np.eye(3))
    A.layers["data_signaling"] = A.X.copy()
    with pytest.raises(ValueError):
        pp.projectData(A, np.eye(3), alpha=0.5)


def test_projectData_dataframe_adj_picks_index() -> None:
    gE2 = pd.read_csv(R_DUMP / "proj_E2.csv", index_col=0)
    adj_df = pd.read_csv(R_DUMP / "proj_adj.csv", index_col=0)
    A = ad.AnnData(
        X=gE2.values.T.astype(float),
        obs=pd.DataFrame(index=gE2.columns),
        var=pd.DataFrame(index=gE2.index),
    )
    A.layers["data_signaling"] = A.X.copy()
    # Force rename the var to ensure DataFrame index is used
    pp.projectData(A, adj_df, alpha=0.5, normalizeAdjMatrix="columns")
    assert A.layers["projected"].shape == (A.n_obs, A.n_vars)
