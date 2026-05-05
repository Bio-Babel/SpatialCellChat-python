"""Tests for ``spatialcellchat.modeling_avg`` against the R reference.

Gold values come from running the corresponding R functions via Rscript on
synthetic inputs. The R harness scripts live under ``/tmp/agentB_slice4/``
during development and are reproduced here as inline gold values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from spatialcellchat.modeling_avg import (
    _r_format_digits1_compare,
    _signif,
    computeAvgCommunProb,
    computeAvgCommunProb_LR_Avg,
    computeAvgCommunProb_LR_Sum,
    computeAvgCommunProb_LR_Visium_Avg,
    computeAvgCommunProb_LR_Visium_Sum,
    computeAvgCommunProb_Visium,
)


# ---------------------------------------------------------------------------
# Shared fixture inputs (matches the R harness in /tmp/agentB_slice4/)
# ---------------------------------------------------------------------------
def _build_prob_8x8() -> sparse.csc_matrix:
    """Build the same 8x8 sparse probability matrix as the R harness."""
    rows = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7,  0, 1, 2, 3, 4, 5, 6, 7,  0, 1, 2, 3, 4, 5, 6, 7]
    )
    cols = np.array(
        [1, 2, 3, 4, 5, 6, 7, 0,  0, 0, 0, 0, 0, 0, 0, 0,  7, 7, 7, 7, 7, 7, 7, 7]
    )
    vals = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    ])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(8, 8)).tocsc()


def _build_dataLR_8() -> np.ndarray:
    return np.column_stack([
        [0.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.4, 0.5],
        [1.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5],
    ])


_GROUP_8 = pd.Categorical(
    ["A", "A", "A", "B", "B", "C", "C", "C"], categories=["A", "B", "C"]
)


# ---------------------------------------------------------------------------
# _signif and _r_format_digits1_compare (internal helpers)
# ---------------------------------------------------------------------------
def test_signif_zero_and_typical():
    np.testing.assert_allclose(
        _signif(np.array([0.0, 0.05, 0.06, 0.094, 0.0951, 0.6666667, 1.0]), 1),
        np.array([0.0, 0.05, 0.06, 0.09, 0.1, 0.7, 1.0]),
        rtol=1e-12,
    )


def test_signif_handles_negative_and_large():
    # signif(-0.05, 1) = -0.05; signif(123.4, 1) = 100
    np.testing.assert_allclose(
        _signif(np.array([-0.05, -0.149, 123.4, 0.001234]), 1),
        np.array([-0.05, -0.1, 100.0, 0.001]),
        rtol=1e-12,
    )


def test_signif_keeps_inf_and_nan():
    out = _signif(np.array([np.inf, -np.inf, np.nan]), 1)
    assert np.isposinf(out[0]) and np.isneginf(out[1]) and np.isnan(out[2])


def test_format_compare_default_threshold():
    # Match R harness exactly:
    # format(c(0.6666667, 0, 1), digits=1) >= 0.1 -> TRUE FALSE TRUE
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.6666667, 0, 1]), 0.1),
        np.array([True, False, True]),
    )
    # format(c(0.6666667, 1, 0.3333333), digits=1) >= 0.1 -> TRUE TRUE TRUE
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.6666667, 1, 0.3333333]), 0.1),
        np.array([True, True, True]),
    )


def test_format_compare_other_thresholds():
    # min_percent=0.5: format(c(0.4, 0.5, 0.6, 1.0), digits=1) >= 0.5 -> F T T T
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.4, 0.5, 0.6, 1.0]), 0.5),
        np.array([False, True, True, True]),
    )
    # min_percent=0.05: format(c(0.04, 0.05, 0.06, 1.0), digits=1) >= 0.05 -> F T T T
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.04, 0.05, 0.06, 1.0]), 0.05),
        np.array([False, True, True, True]),
    )
    # min_percent=0.2: format(c(0.1, 0.2, 0.3), digits=1) >= 0.2 -> F T T
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.1, 0.2, 0.3]), 0.2),
        np.array([False, True, True]),
    )


def test_format_compare_all_zero_vector():
    # format(c(0, 0, 0), digits=1) >= 0.1 -> all FALSE
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.0, 0.0, 0.0]), 0.1),
        np.array([False, False, False]),
    )


def test_format_compare_mixed_vector_with_signif_boundary():
    # format(c(0.094, 0.095, 0.099), digits=1) >= 0.1
    # signif: c(0.09, 0.10, 0.10), max decimals = 2, format = c("0.09","0.10","0.10")
    # >= "0.1" lex: F T T
    np.testing.assert_array_equal(
        _r_format_digits1_compare(np.array([0.094, 0.095, 0.099]), 0.1),
        np.array([False, True, True]),
    )


def test_format_compare_empty():
    out = _r_format_digits1_compare(np.array([]), 0.1)
    assert out.shape == (0,)
    assert out.dtype == bool


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Sum
# ---------------------------------------------------------------------------
def test_LR_Sum_matches_R_default():
    """Gold values from R harness (/tmp/agentB_slice4/harness_format.R):
       dput(res_sum) -> structure(c(0.48, 0, 1.13, 0.3, 0, 0, 0.06, 0, 1.51), ...)
    """
    prob = _build_prob_8x8()
    dataLR = _build_dataLR_8()
    out = computeAvgCommunProb_LR_Sum(
        prob, _GROUP_8, dataLR, min_percent=0.1, min_cells_sr=1
    )
    expected = np.array(
        [
            [0.48, 0.30, 0.06],
            [0.00, 0.00, 0.00],
            [1.13, 0.00, 1.51],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


def test_LR_Sum_min_cells_sr_filters():
    """min_cells_sr=5 filters out the (A, B) and (B, *) entries that don't
    have enough sender/receiver cells in the binarised prob.

    Gold (R harness): structure(c(0.48, 0, 1.13, 0, 0, 0, 0.06, 0, 1.51), ...)
    """
    prob = _build_prob_8x8()
    dataLR = _build_dataLR_8()
    out = computeAvgCommunProb_LR_Sum(
        prob, _GROUP_8, dataLR, min_percent=0.1, min_cells_sr=5
    )
    expected = np.array(
        [
            [0.48, 0.00, 0.06],
            [0.00, 0.00, 0.00],
            [1.13, 0.00, 1.51],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


def test_LR_Sum_short_circuit_returns_zero_matrix():
    """When the per-group expression filter yields all-zero Prob_percent,
    R returns ``Prob_percent`` directly (an NGroup x NGroup zero matrix)
    without computing the costly crossprod.
    """
    prob = _build_prob_8x8()
    # All-zero R column -> R_percent all zero -> Prob_percent all zero
    dataLR = np.column_stack([
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    out = computeAvgCommunProb_LR_Sum(
        prob, _GROUP_8, dataLR, min_percent=0.1, min_cells_sr=1
    )
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, np.zeros((3, 3)))


def test_LR_Sum_high_min_percent_short_circuit():
    """min_percent=0.95 forces the short-circuit path because no group has
    100% expression.
    """
    prob = _build_prob_8x8()
    dataLR = _build_dataLR_8()
    out = computeAvgCommunProb_LR_Sum(
        prob, _GROUP_8, dataLR, min_percent=0.95, min_cells_sr=1
    )
    np.testing.assert_array_equal(out, np.zeros((3, 3)))


def test_LR_Sum_single_group():
    """With only one group, the result is a single 1x1 sum value."""
    nC = 5
    prob = sparse.csc_matrix(
        (np.array([0.1, 0.2, 0.3]),
         (np.array([0, 1, 2]), np.array([1, 2, 3]))),
        shape=(nC, nC),
    )
    group = pd.Categorical(["X"] * nC, categories=["X"])
    dataLR = np.column_stack([
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ])
    out = computeAvgCommunProb_LR_Sum(prob, group, dataLR, min_cells_sr=1)
    assert out.shape == (1, 1)
    # Sum-of-prob across all cells = 0.6
    np.testing.assert_allclose(out, np.array([[0.6]]), rtol=1e-12)


def test_LR_Sum_accepts_dense_prob():
    """Sparse and dense prob inputs must give identical results."""
    prob_sparse = _build_prob_8x8()
    prob_dense = prob_sparse.toarray()
    dataLR = _build_dataLR_8()
    out_sparse = computeAvgCommunProb_LR_Sum(
        prob_sparse, _GROUP_8, dataLR, min_cells_sr=1
    )
    out_dense = computeAvgCommunProb_LR_Sum(
        prob_dense, _GROUP_8, dataLR, min_cells_sr=1
    )
    np.testing.assert_allclose(out_sparse, out_dense, rtol=1e-12)


def test_LR_Sum_does_not_mutate_prob():
    prob = _build_prob_8x8()
    original = prob.data.copy()
    dataLR = _build_dataLR_8()
    computeAvgCommunProb_LR_Sum(prob, _GROUP_8, dataLR, min_cells_sr=1)
    np.testing.assert_array_equal(prob.data, original)


def test_LR_Sum_dataLR_shape_validation():
    prob = _build_prob_8x8()
    bad_dataLR = np.zeros((5, 2))
    with pytest.raises(ValueError, match="dataLR must be shape"):
        computeAvgCommunProb_LR_Sum(prob, _GROUP_8, bad_dataLR, min_cells_sr=1)


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Avg
# ---------------------------------------------------------------------------
def test_LR_Avg_matches_R_default():
    """Gold (R harness): dput ->
       structure(c(0.096, 0, 0.376666666666667, 0.3, 0, 0, 0.02, 0, 0.3775), ...)
    """
    prob = _build_prob_8x8()
    dataLR = _build_dataLR_8()
    out = computeAvgCommunProb_LR_Avg(
        prob, _GROUP_8, dataLR, min_percent=0.1, min_cells_sr=1
    )
    expected = np.array(
        [
            [0.0960, 0.30, 0.0200],
            [0.0000, 0.00, 0.0000],
            [0.376666666666667, 0.00, 0.3775],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


def test_LR_Avg_matches_LR_Sum_when_links_are_one_per_pair():
    """When the binarised prob has exactly one link per (sender_group,
    receiver_group) pair, AVG should equal SUM (divisor = 1).
    """
    nC = 4
    prob = sparse.csc_matrix(
        (np.array([0.4, 0.6]),
         (np.array([0, 2]), np.array([1, 3]))),
        shape=(nC, nC),
    )
    group = pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"])
    dataLR = np.column_stack([[0.5] * 4, [0.5] * 4])
    out_sum = computeAvgCommunProb_LR_Sum(prob, group, dataLR, min_cells_sr=1)
    out_avg = computeAvgCommunProb_LR_Avg(prob, group, dataLR, min_cells_sr=1)
    # Each (sender_group, receiver_group) has exactly 1 cell-link in the
    # binarised prob, so divisor = 1 and AVG == SUM.
    np.testing.assert_allclose(out_avg, out_sum, rtol=1e-12)


def test_LR_Avg_zero_division_to_zero():
    """When a sender_group has no outgoing links to a receiver_group, the
    divisor is 0; the result is set to 0 (R: ``Prob.avg[is.nan(.)] <- 0``).
    """
    nC = 4
    prob = sparse.csc_matrix(
        (np.array([0.5]), (np.array([0]), np.array([1]))),
        shape=(nC, nC),
    )
    group = pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"])
    dataLR = np.column_stack([[0.5] * 4, [0.5] * 4])
    out = computeAvgCommunProb_LR_Avg(prob, group, dataLR, min_cells_sr=1)
    # Only (A, A) link — all other entries are 0/0 -> 0.
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1, :], np.zeros(2))


def test_LR_Avg_short_circuit():
    prob = _build_prob_8x8()
    dataLR = np.column_stack([
        [0.0] * 8, [0.5] * 8,  # all-zero L column -> short-circuit
    ])
    out = computeAvgCommunProb_LR_Avg(
        prob, _GROUP_8, dataLR, min_percent=0.1, min_cells_sr=1
    )
    np.testing.assert_array_equal(out, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Visium_Sum
# ---------------------------------------------------------------------------
def _build_decomp_8x3() -> np.ndarray:
    return np.array([
        [0.7, 0.3, 0.0],
        [0.5, 0.5, 0.0],
        [0.3, 0.7, 0.0],
        [0.0, 0.4, 0.6],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])


def test_LR_Visium_Sum_matches_R():
    """Gold (R harness): dput ->
       structure(c(0.1252, 0.1647, 0.8561, 0.1668, 0.3003, 0.4869, 0.08, 0.531, 1.929), ...)
    """
    prob = _build_prob_8x8()
    decomp = _build_decomp_8x3()
    onehot = (decomp > 0).astype(float)
    out = computeAvgCommunProb_LR_Visium_Sum(prob, decomp, onehot)
    expected = np.array(
        [
            [0.1252, 0.1668, 0.080],
            [0.1647, 0.3003, 0.531],
            [0.8561, 0.4869, 1.929],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)


def test_LR_Visium_Sum_onehot_unused():
    """The Sum variant ignores ``cell_type_onehot`` (placeholder)."""
    prob = _build_prob_8x8()
    decomp = _build_decomp_8x3()
    out_with_onehot = computeAvgCommunProb_LR_Visium_Sum(
        prob, decomp, (decomp > 0).astype(float)
    )
    out_without = computeAvgCommunProb_LR_Visium_Sum(prob, decomp, None)
    np.testing.assert_array_equal(out_with_onehot, out_without)


def test_LR_Visium_Sum_dense_prob():
    prob_sparse = _build_prob_8x8()
    prob_dense = prob_sparse.toarray()
    decomp = _build_decomp_8x3()
    out_s = computeAvgCommunProb_LR_Visium_Sum(prob_sparse, decomp, None)
    out_d = computeAvgCommunProb_LR_Visium_Sum(prob_dense, decomp, None)
    np.testing.assert_allclose(out_s, out_d, rtol=1e-12)


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Visium_Avg
# ---------------------------------------------------------------------------
def test_LR_Visium_Avg_matches_R():
    """Gold (R harness): dput ->
       structure(c(0.02504, 0.0235285714285714, 0.17122, 0.0278, 0.0333666666666667,
                   0.08115, 0.02, 0.066375, 0.241125), ...)
    """
    prob = _build_prob_8x8()
    decomp = _build_decomp_8x3()
    onehot = (decomp > 0).astype(float)
    out = computeAvgCommunProb_LR_Visium_Avg(prob, decomp, onehot)
    expected = np.array(
        [
            [0.02504, 0.0278, 0.020],
            [0.0235285714285714, 0.0333666666666667, 0.066375],
            [0.17122, 0.08115, 0.241125],
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)


def test_LR_Visium_Avg_zero_link_yields_zero():
    """A pair of cell types with NO overlapping spots in the binarised prob
    yields a 0/0 division — set to 0 by ``[is.nan(.)] <- 0``.
    """
    nC = 4
    nT = 2
    # Only (spot 0 -> spot 1) link in prob
    prob = sparse.csc_matrix(
        (np.array([0.5]), (np.array([0]), np.array([1]))),
        shape=(nC, nC),
    )
    # Decomp: spot 0 is 100% type A; spots 1-3 are 100% type B
    decomp = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    onehot = (decomp > 0).astype(float)
    out = computeAvgCommunProb_LR_Visium_Avg(prob, decomp, onehot)
    assert out.shape == (nT, nT)
    assert np.all(np.isfinite(out))
    # Only (A->B) entry should be 0.5 (single link with proportion 1*1)
    np.testing.assert_allclose(out[0, 1], 0.5, rtol=1e-12)
    # No links from (A, A), (B, A), (B, B) -> all 0
    np.testing.assert_allclose(out[0, 0], 0.0)
    np.testing.assert_allclose(out[1, 0], 0.0)
    np.testing.assert_allclose(out[1, 1], 0.0)


def test_LR_Visium_Avg_dense_and_sparse_agree():
    prob_s = _build_prob_8x8()
    decomp = _build_decomp_8x3()
    onehot = (decomp > 0).astype(float)
    a = computeAvgCommunProb_LR_Visium_Avg(prob_s, decomp, onehot)
    b = computeAvgCommunProb_LR_Visium_Avg(prob_s.toarray(), decomp, onehot)
    np.testing.assert_allclose(a, b, rtol=1e-12)


# ---------------------------------------------------------------------------
# Top-level computeAvgCommunProb (with mocked AnnData inputs)
# ---------------------------------------------------------------------------
def _build_mock_adata(
    nC: int = 8,
    nLR: int = 3,
    levels: tuple = ("A", "B", "C"),
    seed: int = 0,
) -> ad.AnnData:
    """Build an AnnData with the slots that ``computeCommunProb`` would write."""
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.random((nC, 5)))
    # Distribute cells across levels evenly
    labels = []
    for i in range(nC):
        labels.append(levels[i % len(levels)])
    obs = pd.DataFrame(
        {"active_ident": pd.Categorical(labels, categories=list(levels))},
        index=[f"cell{i}" for i in range(nC)],
    )
    adata = ad.AnnData(X=X, obs=obs)

    prob_cell_list = []
    for k in range(nLR):
        rng_k = np.random.default_rng(k)
        i = rng_k.integers(0, nC, size=8)
        j = rng_k.integers(0, nC, size=8)
        v = rng_k.random(8) * 0.5
        prob_cell_list.append(sparse.csc_matrix((v, (i, j)), shape=(nC, nC)))

    prob_cell_arr = np.zeros((nC, nC, nLR))
    for k in range(nLR):
        prob_cell_arr[:, :, k] = prob_cell_list[k].toarray()

    rng2 = np.random.default_rng(42)
    Lavg = rng2.random((nLR, nC))
    Ravg = rng2.random((nLR, nC))

    adata.uns["cellchat"] = {
        "net": {
            "prob.cell": prob_cell_arr,
            "tmp": {
                "prob.cell": prob_cell_list,
                "Lavg": Lavg,
                "Ravg": Ravg,
            },
            "LR.names": [f"LR{k + 1}" for k in range(nLR)],
        },
        "options": {"parameter": {}},
    }
    return adata


def test_computeAvgCommunProb_writes_expected_slots():
    adata = _build_mock_adata(nC=8, nLR=3)
    out = computeAvgCommunProb(
        adata, avg_type="avg", do_permutation=False, min_cells_sr=1
    )
    cc = out.uns["cellchat"]
    assert cc["net"]["prob"].shape == (3, 3, 3)
    assert cc["net"]["pval"] is None
    assert cc["net"]["tmp"]["LRsig.CCC.counts"].shape == (3,)
    assert cc["net"]["tmp"]["LRsig.GGC.counts"].shape == (3,)
    # Parameter dict mirrors R's storage
    p = cc["options"]["parameter"]
    assert p["min.percent"] == 0.1
    assert p["min.cells.sr"] == 1
    assert p["avg.type"] == "avg"
    assert p["do.permutation"] is False
    assert p["nboot"] == 100
    assert p["seed.use"] == 1
    assert p["colocalization.use"] is False
    assert p["thresh.colo"] == 0.05


def test_computeAvgCommunProb_avg_vs_sum():
    adata_avg = _build_mock_adata(nC=8, nLR=2)
    adata_sum = _build_mock_adata(nC=8, nLR=2)
    out_avg = computeAvgCommunProb(
        adata_avg, avg_type="avg", do_permutation=False, min_cells_sr=1
    )
    out_sum = computeAvgCommunProb(
        adata_sum, avg_type="sum", do_permutation=False, min_cells_sr=1
    )
    p_avg = out_avg.uns["cellchat"]["net"]["prob"]
    p_sum = out_sum.uns["cellchat"]["net"]["prob"]
    # AVG ≤ SUM elementwise (when both are ≥ 0)
    assert p_avg.shape == p_sum.shape
    assert np.all(p_avg <= p_sum + 1e-12)


def test_computeAvgCommunProb_permutation_pval_shape_and_bounds():
    adata = _build_mock_adata(nC=8, nLR=2)
    out = computeAvgCommunProb(
        adata,
        avg_type="sum",
        do_permutation=True,
        nboot=10,
        seed_use=1,
        min_cells_sr=1,
    )
    pval = out.uns["cellchat"]["net"]["pval"]
    prob = out.uns["cellchat"]["net"]["prob"]
    assert pval.shape == (3, 3, 2)
    assert np.all(pval >= 0.0) and np.all(pval <= 1.0)
    # Where prob == 0, pval is forced to 1
    np.testing.assert_array_equal(pval[prob == 0], 1.0)


def test_computeAvgCommunProb_invalid_avg_type():
    adata = _build_mock_adata()
    with pytest.raises(ValueError, match="avg_type must be one of"):
        computeAvgCommunProb(adata, avg_type="median", do_permutation=False)


def test_computeAvgCommunProb_missing_prob_cell():
    adata = _build_mock_adata()
    del adata.uns["cellchat"]["net"]["tmp"]["prob.cell"]
    with pytest.raises(RuntimeError, match="Please run `computeCommunProb`"):
        computeAvgCommunProb(adata, do_permutation=False)


def test_computeAvgCommunProb_missing_Lavg_Ravg():
    adata = _build_mock_adata()
    del adata.uns["cellchat"]["net"]["tmp"]["Lavg"]
    with pytest.raises(RuntimeError, match="Lavg.*Ravg"):
        computeAvgCommunProb(adata, do_permutation=False)


def test_computeAvgCommunProb_factor_levels_unique_check():
    """R guard: numCluster != length(unique(group)) -> stop."""
    adata = _build_mock_adata(nC=8)
    # Add a level that has no cells
    cats = list(adata.obs["active_ident"].cat.categories) + ["D"]
    adata.obs["active_ident"] = pd.Categorical(
        adata.obs["active_ident"].astype(str), categories=cats
    )
    with pytest.raises(ValueError, match="ensure that the factor levels"):
        computeAvgCommunProb(adata, do_permutation=False)


def test_computeAvgCommunProb_LRsig_use_idx_when_some_LRs_empty():
    """LRs whose prob.cell is empty are skipped; their slice of Prob stays 0."""
    adata = _build_mock_adata(nC=8, nLR=3)
    # Zero out the second LR's prob.cell
    adata.uns["cellchat"]["net"]["tmp"]["prob.cell"][1] = sparse.csc_matrix(
        (8, 8), dtype=float
    )
    adata.uns["cellchat"]["net"]["prob.cell"][:, :, 1] = 0
    out = computeAvgCommunProb(
        adata, avg_type="sum", do_permutation=False, min_cells_sr=1
    )
    cc = out.uns["cellchat"]
    assert cc["net"]["tmp"]["LRsig.CCC.counts"][1] == 0
    np.testing.assert_array_equal(
        cc["net"]["prob"][:, :, 1], np.zeros((3, 3))
    )


def test_computeAvgCommunProb_all_empty_LRs_raises():
    adata = _build_mock_adata(nC=8, nLR=2)
    for k in range(2):
        adata.uns["cellchat"]["net"]["tmp"]["prob.cell"][k] = sparse.csc_matrix(
            (8, 8), dtype=float
        )
    with pytest.raises(ValueError, match="does not have any cell-level"):
        computeAvgCommunProb(adata, do_permutation=False)


# ---------------------------------------------------------------------------
# Top-level computeAvgCommunProb_Visium
# ---------------------------------------------------------------------------
def _build_mock_visium_adata(
    nC: int = 8,
    nLR: int = 2,
    levels: tuple = ("A", "B", "C"),
    seed: int = 100,
) -> tuple[ad.AnnData, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.random((nC, 5)))
    labels = [levels[i % len(levels)] for i in range(nC)]
    obs = pd.DataFrame(
        {"active_ident": pd.Categorical(labels, categories=list(levels))},
        index=[f"spot{i}" for i in range(nC)],
    )
    adata = ad.AnnData(X=X, obs=obs)

    prob_cell_list = []
    for k in range(nLR):
        rng_k = np.random.default_rng(seed + k)
        i = rng_k.integers(0, nC, size=8)
        j = rng_k.integers(0, nC, size=8)
        v = rng_k.random(8) * 0.5
        prob_cell_list.append(sparse.csc_matrix((v, (i, j)), shape=(nC, nC)))

    prob_cell_arr = np.zeros((nC, nC, nLR))
    for k in range(nLR):
        prob_cell_arr[:, :, k] = prob_cell_list[k].toarray()

    decomp = pd.DataFrame(
        rng.random((nC, len(levels))),
        columns=list(levels),
        index=adata.obs_names,
    )
    # Make some sparsity in decomp (zero out some entries) so onehot is non-trivial
    decomp_arr = decomp.to_numpy()
    decomp_arr[decomp_arr < 0.2] = 0
    decomp = pd.DataFrame(decomp_arr, columns=list(levels), index=adata.obs_names)

    adata.uns["cellchat"] = {
        "net": {
            "prob.cell": prob_cell_arr,
            "tmp": {"prob.cell": prob_cell_list},
            "LR.names": [f"LR{k + 1}" for k in range(nLR)],
        },
        "options": {"parameter": {}},
    }
    return adata, decomp


def test_computeAvgCommunProb_Visium_writes_expected_slots():
    adata, decomp = _build_mock_visium_adata(nC=8, nLR=2)
    out = computeAvgCommunProb_Visium(
        adata, decomp, avg_type="avg", do_permutation=False
    )
    cc = out.uns["cellchat"]
    assert cc["net"]["prob"].shape == (3, 3, 2)
    assert cc["net"]["pval"] is None
    assert "cell.type.decomposition" in cc["net"]["tmp"]
    np.testing.assert_array_equal(
        cc["net"]["tmp"]["cell.type.decomposition"], decomp.to_numpy()
    )
    p = cc["options"]["parameter"]
    assert p["avg.type"] == "avg"
    assert p["do.permutation"] is False


def test_computeAvgCommunProb_Visium_decomp_columns_must_match_levels():
    adata, decomp = _build_mock_visium_adata(nC=8, nLR=2)
    decomp_bad = decomp.rename(columns={"A": "X"})
    with pytest.raises(ValueError, match="cell groups' names"):
        computeAvgCommunProb_Visium(adata, decomp_bad, do_permutation=False)


def test_computeAvgCommunProb_Visium_decomp_rows_must_match_obs():
    adata, decomp = _build_mock_visium_adata(nC=8, nLR=2)
    decomp_bad = decomp.copy()
    decomp_bad.index = [f"other{i}" for i in range(decomp_bad.shape[0])]
    with pytest.raises(ValueError, match="cells' names"):
        computeAvgCommunProb_Visium(adata, decomp_bad, do_permutation=False)


def test_computeAvgCommunProb_Visium_numpy_decomp_shape_check():
    adata, _ = _build_mock_visium_adata(nC=8, nLR=2)
    bad = np.zeros((8, 5))  # wrong number of columns
    with pytest.raises(ValueError, match="expected 3"):
        computeAvgCommunProb_Visium(adata, bad, do_permutation=False)


def test_computeAvgCommunProb_Visium_permutation_pval_bounds():
    adata, decomp = _build_mock_visium_adata(nC=10, nLR=2)
    out = computeAvgCommunProb_Visium(
        adata, decomp, avg_type="sum", do_permutation=True, nboot=5, seed_use=42
    )
    pval = out.uns["cellchat"]["net"]["pval"]
    prob = out.uns["cellchat"]["net"]["prob"]
    assert pval.shape == (3, 3, 2)
    assert np.all(pval >= 0.0) and np.all(pval <= 1.0)
    np.testing.assert_array_equal(pval[prob == 0], 1.0)


def test_computeAvgCommunProb_Visium_avg_short_circuit():
    """If a sender cell-type has no spots with prob > 0, AVG output for that
    row is 0 (zero-by-zero short-circuit handled inside the LR_Visium_Avg)."""
    nC = 6
    nT = 2
    prob = sparse.csc_matrix(
        (np.array([0.5]), (np.array([0]), np.array([1]))),
        shape=(nC, nC),
    )
    # Decomp: spot 0 is 100% A; spots 1-5 are 100% B
    decomp = np.array(
        [[1.0, 0.0]] + [[0.0, 1.0]] * 5, dtype=float
    )
    onehot = (decomp > 0).astype(float)
    out = computeAvgCommunProb_LR_Visium_Avg(prob, decomp, onehot)
    # Only (A->B) link
    np.testing.assert_allclose(out, np.array([[0.0, 0.5], [0.0, 0.0]]),
                               rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Random R parity regression tests (gold values from /tmp/agentB_slice4/parity_LR.out)
# ---------------------------------------------------------------------------
def test_LR_variants_random_R_parity_case2():
    """R harness CASE 2: nC=24, ngroup=5, min_pct=0.15, min_sr=4, density~0.10."""
    nC = 24
    ngroup = 5
    grp_str = "A B C D E E C A A C C A A E C E D C D E D A A D"
    group = pd.Categorical(grp_str.split(), categories=list("ABCDE"))

    # Gold inputs
    prob_i = np.array(
        [10, 12, 14, 18, 20, 21, 23, 14, 15, 10, 1, 3, 5, 12, 15, 18, 10,
         2, 16, 1, 11, 16, 0, 4, 21, 23, 1, 15, 3, 4, 9, 12, 13, 21, 1,
         17, 9, 12, 8, 8, 7, 13, 22, 1, 20, 1, 8, 17, 20, 1, 16, 6, 3,
         16, 2, 6, 20]
    )
    prob_p = np.array(
        [0, 7, 7, 9, 10, 16, 17, 19, 22, 26, 28, 29, 34, 36, 38, 39, 40,
         43, 45, 49, 51, 52, 52, 54, 57]
    )
    prob_x = np.array([
        0.2324929636, 0.03096607642, 0.4932805626, 0.0594073059, 0.1766830175,
        0.0285348983, 0.2494870723, 0.06383873022, 0.4031100769, 0.560792588,
        0.3154356716, 0.2491095011, 0.1064421947, 0.2823046709, 0.1975260469,
        0.4958282252, 0.4187143021, 0.1899465424, 0.4796818569, 0.4006433277,
        0.1910616257, 0.4466655951, 0.3145062767, 0.1190334898, 0.4810881276,
        0.4793221753, 0.1764378732, 0.05314472294, 0.1823853396, 0.4963251322,
        0.2772460803, 0.4888941883, 0.4042057617, 0.1365273492, 0.1107387104,
        0.09548939567, 0.3511529363, 0.01919373195, 0.4704275971, 0.293895659,
        0.241294428, 0.3006049029, 0.3229058144, 0.3139211459, 0.3074757092,
        0.02397579525, 0.4265990618, 0.3506386688, 0.4576317031, 0.1317020564,
        0.4794887616, 0.2775144193, 0.368117678, 0.4940556905, 0.4291326825,
        0.3838363421, 0.09443349135,
    ])
    prob = sparse.csc_matrix((prob_x, prob_i, prob_p), shape=(nC, nC))

    dataLR = np.column_stack([
        [0, 0.04223553138, 0, 0, 0, 0.9859554549, 0, 0.356241531, 0,
         0.8227537966, 0.9211417169, 0, 0.3952802736, 0, 0.3818383149,
         0.6547320059, 0.5343504981, 0.8761670894, 0, 0, 0.7750025427,
         0, 0, 0.7627627309],
        [0, 0, 0, 0.7106694179, 0.4212004612, 0, 0.6323596181, 0, 0, 0,
         0, 0.8969085531, 0.1296318695, 0.6873720777, 0.8668531801,
         0.9891777674, 0.8948460841, 0.03490370302, 0, 0, 0,
         0.6827363421, 0, 0],
    ])

    # Expected outputs (R col-major order, reshape with order='F')
    expected_sum = np.array(
        [1.671578542, 0.5113820381, 1.098509002, 2.273738535, 1.019564384,
         0, 0, 0, 0, 0,
         0.4704275971, 0.490359019, 0.2537852726, 0.9695429058, 0.4562547999,
         0.9907993042, 0.02397579525, 2.001914701, 0.5520651945, 0.3006049029,
         0.5953940619, 0.447137728, 0.7698672385, 1.224426488, 0.3039682416]
    ).reshape((5, 5), order="F")
    expected_avg = np.array(
        [0.2387969346, 0.2556910191, 0.2746272505, 0.3248197906, 0.3398547946,
         0, 0, 0, 0, 0,
         0.4704275971, 0.2451795095, 0.1268926363, 0.3231809686, 0.2281273999,
         0.3302664347, 0.02397579525, 0.4003829401, 0.2760325972, 0.3006049029,
         0.1984646873, 0.223568864, 0.3849336192, 0.4081421626, 0.1519841208]
    ).reshape((5, 5), order="F")

    py_sum = computeAvgCommunProb_LR_Sum(
        prob, group, dataLR, min_percent=0.15, min_cells_sr=4
    )
    py_avg = computeAvgCommunProb_LR_Avg(
        prob, group, dataLR, min_percent=0.15, min_cells_sr=4
    )
    np.testing.assert_allclose(py_sum, expected_sum, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(py_avg, expected_avg, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# Cross-variant consistency: the four LR variants form a coherent family
# ---------------------------------------------------------------------------
def test_LR_variants_agree_on_simple_case():
    """When using one-hot decomposition (each spot belongs to exactly one
    cell type), the Visium variants must match the regular variants
    (for the SUM rule); for AVG the variants differ in their divisor
    (Visium AVG divides by per-cell-type-pair LINK count via one-hot,
    while the regular AVG also includes the min.cells.sr filter).
    """
    nC = 6
    prob = sparse.csc_matrix(
        (np.array([0.1, 0.2, 0.3, 0.4]),
         (np.array([0, 1, 2, 3]), np.array([2, 3, 4, 5]))),
        shape=(nC, nC),
    )
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    # Hard one-hot decomposition
    decomp = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=float)
    onehot = decomp.copy()

    # All-on dataLR (no min.percent / min.cells.sr filter) so SUM variants agree
    dataLR = np.column_stack([[1.0] * 6, [1.0] * 6])

    out_sum = computeAvgCommunProb_LR_Sum(
        prob, group, dataLR, min_percent=0.1, min_cells_sr=1
    )
    out_v_sum = computeAvgCommunProb_LR_Visium_Sum(prob, decomp, onehot)
    np.testing.assert_allclose(out_sum, out_v_sum, rtol=1e-12)

    # And the Visium AVG matches the Avg for one-hot decomposition + no extra filter
    out_avg = computeAvgCommunProb_LR_Avg(
        prob, group, dataLR, min_percent=0.1, min_cells_sr=1
    )
    out_v_avg = computeAvgCommunProb_LR_Visium_Avg(prob, decomp, onehot)
    np.testing.assert_allclose(out_avg, out_v_avg, rtol=1e-12)
