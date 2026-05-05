"""Tests for ``spatialcellchat._hill`` against the R reference implementation.

Gold values are taken directly from running ``SpatialCellChat::HillFunctionFordataLR``
and ``SpatialCellChat::createPspatialFrom_dspatial`` in R via Rscript.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from spatialcellchat._hill import HillFunctionFordataLR, createPspatialFrom_dspatial


# ---------------------------------------------------------------------------
# HillFunctionFordataLR
# ---------------------------------------------------------------------------
def _make_csc(rows, cols, vals, shape):
    """Build a CSC matrix from explicit (row, col, value) triplets."""
    return sparse.coo_matrix(
        (np.asarray(vals, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=shape,
    ).tocsc()


def test_HillFunctionFordataLR_default_kh_n():
    # R Test 1: Kh=0.5, n=1 on 3x3 sparse matrix
    # i = c(1,2,3,1,3), j = c(1,1,2,3,3), x = c(0.1, 0.5, 1.0, 0.25, 2.0)
    # 1-indexed in R; convert to 0-indexed
    m = _make_csc(
        rows=[0, 1, 2, 0, 2],
        cols=[0, 0, 1, 2, 2],
        vals=[0.1, 0.5, 1.0, 0.25, 2.0],
        shape=(3, 3),
    )
    out = HillFunctionFordataLR(m, Kh=0.5, n=1)
    # Reference dense form from R
    expected = np.array(
        [
            [0.1666666666666667, 0.0, 0.3333333333333333],
            [0.5, 0.0, 0.0],
            [0.0, 0.6666666666666666, 0.8],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)


def test_HillFunctionFordataLR_kh_03_n_2():
    # R Test 2: Kh=0.3, n=2 on the same matrix
    m = _make_csc(
        rows=[0, 1, 2, 0, 2],
        cols=[0, 0, 1, 2, 2],
        vals=[0.1, 0.5, 1.0, 0.25, 2.0],
        shape=(3, 3),
    )
    out = HillFunctionFordataLR(m, Kh=0.3, n=2)
    # Gold @x in CSC ordering (col-major):
    #   col 0 -> rows [0, 1] -> vals [0.1, 0.5]^2 / (0.3^2 + ...)
    # We compare full dense
    Kh, n = 0.3, 2
    Kh_n = Kh**n
    expected_dense = np.zeros((3, 3))
    for r, c, v in [(0, 0, 0.1), (1, 0, 0.5), (2, 1, 1.0), (0, 2, 0.25), (2, 2, 2.0)]:
        expected_dense[r, c] = v**n / (Kh_n + v**n)
    np.testing.assert_allclose(out.toarray(), expected_dense, rtol=1e-12, atol=1e-12)
    # Also compare to literal R values
    # R: 0.1000000 0.7352941 0.9174312 0.4098361 0.9779951 (in @x storage order)
    # CSC stored order: col-major so:
    #  col 0: (0,0)=0.1, (1,0)=0.5
    #  col 1: (2,1)=1.0
    #  col 2: (0,2)=0.25, (2,2)=2.0
    np.testing.assert_allclose(
        out.data,
        [0.1000000, 0.7352941, 0.9174312, 0.4098361, 0.9779951],
        rtol=1e-6,
        atol=1e-6,
    )


def test_HillFunctionFordataLR_with_explicit_zero():
    # R Test 3: an explicit-zero stored entry stays zero (0^1 / (Kh + 0) = 0)
    m = _make_csc(
        rows=[0, 1],
        cols=[0, 1],
        vals=[0.0, 0.5],
        shape=(3, 3),
    )
    out = HillFunctionFordataLR(m, Kh=0.5, n=1)
    # The explicit zero must remain in the data array (R behavior)
    assert out.nnz == 2
    np.testing.assert_allclose(np.sort(out.data), np.array([0.0, 0.5]),
                               rtol=1e-12, atol=1e-12)


def test_HillFunctionFordataLR_empty():
    # R Test 4: empty sparse matrix -> empty data, no error
    m = sparse.csc_matrix((3, 3), dtype=float)
    out = HillFunctionFordataLR(m, Kh=0.5, n=1)
    assert out.shape == (3, 3)
    assert out.nnz == 0
    assert out.data.size == 0


def test_HillFunctionFordataLR_single_element():
    # R Test 5: 1x1 with value 0.5 -> 0.5
    m = _make_csc(rows=[0], cols=[0], vals=[0.5], shape=(1, 1))
    out = HillFunctionFordataLR(m, Kh=0.5, n=1)
    assert out.shape == (1, 1)
    np.testing.assert_allclose(out.toarray(), [[0.5]], rtol=1e-12, atol=1e-12)


def test_HillFunctionFordataLR_wide_range_kh07_n3():
    # R Test 6: Kh=0.7, n=3 on a wide range of inputs
    # Input @x: [0.01, 0.1, 0.5, 1.0, 5.0, 100.0]
    # Triplets (row, col): (0,0)=0.01, (1,0)=0.1, (2,0)=0.5, (3,0)=1.0,
    #                      (4,0)=5.0,  (0,1)=100.0
    m = _make_csc(
        rows=[0, 1, 2, 3, 4, 0],
        cols=[0, 0, 0, 0, 0, 1],
        vals=[0.01, 0.1, 0.5, 1.0, 5.0, 100.0],
        shape=(5, 2),
    )
    out = HillFunctionFordataLR(m, Kh=0.7, n=3)
    # Gold from R (in @x storage order, col-major):
    # col 0: vals [0.01, 0.1, 0.5, 1.0, 5.0]
    # col 1: vals [100.0]
    expected = [
        2.915443e-06, 2.906977e-03, 2.670940e-01, 7.446016e-01, 9.972635e-01,
        9.999997e-01,
    ]
    np.testing.assert_allclose(out.data, expected, rtol=1e-6, atol=1e-9)


def test_HillFunctionFordataLR_preserves_shape_and_pattern():
    # Sparsity pattern of the output equals the sparsity pattern of the input
    m = _make_csc(
        rows=[0, 1, 2, 0, 2],
        cols=[0, 0, 1, 2, 2],
        vals=[0.1, 0.5, 1.0, 0.25, 2.0],
        shape=(3, 3),
    )
    out = HillFunctionFordataLR(m, Kh=0.5, n=1)
    np.testing.assert_array_equal(out.indices, m.indices)
    np.testing.assert_array_equal(out.indptr, m.indptr)
    assert out.shape == m.shape


def test_HillFunctionFordataLR_does_not_mutate_input():
    # Caller's matrix is not modified in place
    m = _make_csc(
        rows=[0, 1, 2, 0, 2],
        cols=[0, 0, 1, 2, 2],
        vals=[0.1, 0.5, 1.0, 0.25, 2.0],
        shape=(3, 3),
    )
    original = m.data.copy()
    HillFunctionFordataLR(m, Kh=0.5, n=1)
    np.testing.assert_array_equal(m.data, original)


# ---------------------------------------------------------------------------
# createPspatialFrom_dspatial
# ---------------------------------------------------------------------------
def test_createPspatialFrom_dspatial_distance_use_true_3x3():
    # R Test A: distance.use=TRUE on 3x3 distance matrix
    # i = c(1,2,3,1,3,2), j = c(1,1,1,2,2,3), x = c(2,5,10,4,8,3)
    d = _make_csc(
        rows=[0, 1, 2, 0, 2, 1],
        cols=[0, 0, 0, 1, 1, 2],
        vals=[2.0, 5.0, 10.0, 4.0, 8.0, 3.0],
        shape=(3, 3),
    )
    out = createPspatialFrom_dspatial(d, distance_use=True)
    # Reciprocate -> max=0.5 -> diag set to 0.5
    expected = np.array(
        [
            [0.5, 0.25, 0.0],
            [0.2, 0.5, 1.0 / 3.0],
            [0.1, 0.125, 0.5],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)
    # Diagonal must be exactly the max of the reciprocated values
    np.testing.assert_allclose(out.diagonal(), [0.5, 0.5, 0.5],
                               rtol=1e-12, atol=1e-12)


def test_createPspatialFrom_dspatial_distance_use_false_3x3():
    # R Test B: distance.use=FALSE on the same matrix
    d = _make_csc(
        rows=[0, 1, 2, 0, 2, 1],
        cols=[0, 0, 0, 1, 1, 2],
        vals=[2.0, 5.0, 10.0, 4.0, 8.0, 3.0],
        shape=(3, 3),
    )
    out = createPspatialFrom_dspatial(d, distance_use=False)
    # All stored values become 1; sparsity pattern is the post-diag-fill pattern
    # Original positions: (0,0),(1,0),(2,0),(0,1),(2,1),(1,2)
    # Diag fill adds (1,1) and (2,2) (since (0,0) already exists)
    # so 8 entries total, all == 1
    expected = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out.data, np.ones(8))
    np.testing.assert_allclose(out.diagonal(), [1.0, 1.0, 1.0],
                               rtol=1e-12, atol=1e-12)


def test_createPspatialFrom_dspatial_diag_only():
    # R Test C: diagonal-only input. Reciprocate -> max==first entry's recip.
    # Input diag = [2, 4, 8] -> reciprocate -> [0.5, 0.25, 0.125], max=0.5.
    # Setting diag to 0.5 overwrites all of them with 0.5.
    d = _make_csc(
        rows=[0, 1, 2],
        cols=[0, 1, 2],
        vals=[2.0, 4.0, 8.0],
        shape=(3, 3),
    )
    out = createPspatialFrom_dspatial(d, distance_use=True)
    expected = np.diag([0.5, 0.5, 0.5])
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)


def test_createPspatialFrom_dspatial_off_diagonal_only_4x4():
    # R Test D: 4x4 with no diagonal entries, distance.use=TRUE
    d = _make_csc(
        rows=[0, 1, 2, 3, 0, 2],
        cols=[1, 2, 3, 0, 3, 1],
        vals=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=(4, 4),
    )
    out = createPspatialFrom_dspatial(d, distance_use=True)
    # Reciprocate values: (0,1)=2/3, (1,2)=0.5, (2,3)=1/3, (3,0)=0.25,
    #                     (0,3)=0.2, (2,1)=1/6
    # Max = 2/3 (from 1/1.5)
    # Diag fill adds (0,0), (1,1), (2,2), (3,3) all = 2/3
    expected = np.array(
        [
            [2.0 / 3.0, 2.0 / 3.0, 0.0, 0.2],
            [0.0, 2.0 / 3.0, 0.5, 0.0],
            [0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0],
            [0.25, 0.0, 0.0, 2.0 / 3.0],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)


def test_createPspatialFrom_dspatial_off_diagonal_only_4x4_binary():
    # R Test E: same as D but distance.use=FALSE
    d = _make_csc(
        rows=[0, 1, 2, 3, 0, 2],
        cols=[1, 2, 3, 0, 3, 1],
        vals=[1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape=(4, 4),
    )
    out = createPspatialFrom_dspatial(d, distance_use=False)
    expected = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(out.data, np.ones(10))


def test_createPspatialFrom_dspatial_with_unit_distance():
    # R Test F: distances containing 1.0 -> reciprocal 1.0 -> max=1.
    # Diag fill makes everything in-pattern equal to 1.
    d = _make_csc(
        rows=[0, 1, 2, 0],
        cols=[0, 1, 2, 1],
        vals=[1.0, 2.0, 4.0, 1.0],
        shape=(3, 3),
    )
    out = createPspatialFrom_dspatial(d, distance_use=True)
    expected = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)
    # Exactly 4 stored entries (no new ones added because diag entries pre-existed)
    assert out.nnz == 4


def test_createPspatialFrom_dspatial_max_uses_pre_diag_state():
    # Mixed off-diag value larger than reciprocal of diag entries: still
    # max is computed over POST-reciprocation, PRE-diag-fill values.
    # Input: (0,0)=0.5, (1,1)=0.25, (2,2)=4, (0,1)=100
    # Reciprocate: (0,0)=2, (0,1)=0.01, (1,1)=4, (2,2)=0.25
    # Max = 4 (from (1,1)=1/0.25)
    # Diag fill -> (0,0)=4, (1,1)=4, (2,2)=4 (overwriting (0,0)=2 and (2,2)=0.25)
    d = _make_csc(
        rows=[0, 1, 2, 0],
        cols=[0, 1, 2, 1],
        vals=[0.5, 0.25, 4.0, 100.0],
        shape=(3, 3),
    )
    out = createPspatialFrom_dspatial(d, distance_use=True)
    expected = np.array(
        [
            [4.0, 0.01, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    np.testing.assert_allclose(out.toarray(), expected, rtol=1e-12, atol=1e-12)


def test_createPspatialFrom_dspatial_does_not_mutate_input():
    d = _make_csc(
        rows=[0, 1, 2, 0, 2, 1],
        cols=[0, 0, 0, 1, 1, 2],
        vals=[2.0, 5.0, 10.0, 4.0, 8.0, 3.0],
        shape=(3, 3),
    )
    original = d.data.copy()
    createPspatialFrom_dspatial(d, distance_use=True)
    np.testing.assert_array_equal(d.data, original)


def test_createPspatialFrom_dspatial_default_distance_use_true():
    # Default value of distance_use is True (matching R's distance.use=T)
    d = _make_csc(
        rows=[0, 1],
        cols=[1, 0],
        vals=[2.0, 4.0],
        shape=(2, 2),
    )
    out_default = createPspatialFrom_dspatial(d)
    out_true = createPspatialFrom_dspatial(d, distance_use=True)
    np.testing.assert_allclose(out_default.toarray(), out_true.toarray(),
                               rtol=1e-12, atol=1e-12)
