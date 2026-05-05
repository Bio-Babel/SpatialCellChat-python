"""Tests for computeCommunField and get_dens.

These are the two analysis-side exports that were not covered by the wave-5
sub-agents (kde stub vs full implementation gap). R↔Py harness uses
synthetic per-cell sparse matrices (no need to invoke computeCommunProb).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spatialcellchat.analysis_network import computeCommunField, get_dens


def _toy_adata(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = 12
    coords = rng.uniform(0, 100, size=(n, 2))
    obs = pd.DataFrame(index=[f"c{i:02d}" for i in range(n)])
    var = pd.DataFrame(index=[f"g{i:02d}" for i in range(5)])
    X = sp.csr_matrix(rng.poisson(2, size=(n, 5)))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    # Build two synthetic per-LR sparse matrices (n x n) with ~30% density.
    def _mat(seed_):
        rs = np.random.default_rng(seed_)
        m = rs.uniform(0, 1, size=(n, n)) * (rs.uniform(0, 1, size=(n, n)) < 0.3)
        return sp.csr_matrix(m)
    adata.uns["cellchat"] = {
        "netP": {
            "tmp": {
                "prob.cell": {"PWAY1": _mat(11), "PWAY2": _mat(22)},
            },
            "prob.cell": None,
        }
    }
    return adata


def test_computeCommunField_writes_field_slot():
    a = _toy_adata(seed=42)
    out = computeCommunField(a, slot_name="netP")
    field = out.uns["cellchat"]["netP"]["field"]
    assert "outgoing" in field and "incoming" in field
    assert set(field["outgoing"]) == {"PWAY1", "PWAY2"}
    for name, arr in field["outgoing"].items():
        assert arr.shape == (12, 2)
        assert np.isfinite(arr).all()


def test_computeCommunField_signaling_subset_str():
    a = _toy_adata(seed=42)
    out = computeCommunField(a, slot_name="netP", signaling_name="PWAY2")
    assert set(out.uns["cellchat"]["netP"]["field"]["outgoing"]) == {"PWAY2"}


def test_computeCommunField_signaling_subset_int():
    a = _toy_adata(seed=42)
    out = computeCommunField(a, slot_name="netP", signaling_name=1)
    assert set(out.uns["cellchat"]["netP"]["field"]["outgoing"]) == {"PWAY1"}


def test_computeCommunField_no_prob_cell_raises():
    a = _toy_adata(seed=42)
    a.uns["cellchat"]["netP"]["tmp"] = {}
    with pytest.raises(RuntimeError, match="run `computeCommunProb"):
        computeCommunField(a, slot_name="netP")


def test_get_dens_wkde_lookup():
    """Round-trip: build a 2D grid, evaluate via wkde-style mapping."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([0.0, 1.0, 2.0])
    z = np.outer(np.arange(xs.size), np.arange(ys.size)).astype(float)
    dens = {"x": xs, "y": ys, "z": z}
    data = np.array([[0.5, 0.0], [2.5, 1.5], [4.0, 2.0]])
    out = get_dens(data, dens, method="wkde")
    # Index lookup: (ix, iy) for each row
    # row0: x=0.5 -> ix=0, y=0.0 -> iy=0 -> z[0,0]=0
    # row1: x=2.5 -> ix=2, y=1.5 -> iy=1 -> z[2,1]=2
    # row2: x=4.0 -> ix=4, y=2.0 -> iy=2 -> z[4,2]=8
    np.testing.assert_array_equal(out, np.array([0.0, 2.0, 8.0]))


def test_get_dens_ks_lookup():
    """Round-trip: ks-style dens dict with eval.points + estimate matrix."""
    ep = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])]
    estimate = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    dens = {"eval.points": ep, "estimate": estimate}
    data = np.array([[0.5, 1.5], [2.0, 0.0]])
    out = get_dens(data, dens, method="ks")
    np.testing.assert_array_equal(out, np.array([estimate[0, 1], estimate[2, 0]]))


def test_get_dens_invalid_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        get_dens(np.zeros((1, 2)), {}, method="bogus")
