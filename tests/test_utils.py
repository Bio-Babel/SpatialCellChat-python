"""Tests for ``spatialcellchat._utils`` against the R reference implementation.

The reference CSV files are bundled under
``tests/data/r_dumps/sccp_slice1/``, originally produced by ``dump_utils.R``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialcellchat._utils import (
    elementwiseProductForSparseMatandDenseMat,
    geometricMean,
    myElementwiseProduct,
    scaleMat,
    thresholdedMean,
    triMean,
)

GOLD = Path(__file__).parent / "data" / "r_dumps" / "sccp_slice1"


def _read_gold(name: str) -> pd.DataFrame:
    return pd.read_csv(GOLD / name)


# ----------------------------------------------------------------------
# geometricMean
# ----------------------------------------------------------------------
def test_geometricMean_vector_typical():
    gold = _read_gold("gm_scalar.csv")
    g1 = geometricMean(np.array([1.0, 2.0, 4.0, 8.0]))
    g2 = geometricMean(np.array([1.0, np.nan, 2.0, 4.0]))
    g3 = geometricMean(np.array([5.0]))
    np.testing.assert_allclose(
        [g1, g2, g3], gold["value"].to_numpy(), rtol=1e-12, atol=1e-12
    )


def test_geometricMean_matrix():
    gold1 = _read_gold("gm_mat1.csv")
    # R: matrix(c(1,2,4,3,6,12), nrow=2, byrow=FALSE) fills column-major →
    # col1=[1,2], col2=[4,3], col3=[6,12]
    mat1 = np.array(
        [[1, 4, 6],
         [2, 3, 12]],
        dtype=float,
    )
    out = geometricMean(mat1)
    np.testing.assert_allclose(out, gold1["g4"].to_numpy(), rtol=1e-12, atol=1e-12)


def test_geometricMean_matrix_with_NA():
    gold2 = _read_gold("gm_mat2.csv")
    # R: matrix(c(1, NA, 4, 3, 6, NA), nrow=2, byrow=FALSE)
    # col1=[1,NA], col2=[4,3], col3=[6,NA]
    mat2 = np.array(
        [[1.0, 4.0, 6.0],
         [np.nan, 3.0, np.nan]],
        dtype=float,
    )
    out = geometricMean(mat2)
    np.testing.assert_allclose(out, gold2["g5"].to_numpy(), rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# triMean
# ----------------------------------------------------------------------
def test_triMean_against_R():
    gold = _read_gold("trimean.csv")
    inputs = [
        np.arange(1, 11, dtype=float),
        np.concatenate([[np.nan], np.arange(1, 11, dtype=float)]),
        np.array([0.0]),
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
        np.linspace(0, 1, 11),
    ]
    out = [triMean(x) for x in inputs]
    np.testing.assert_allclose(out, gold["value"].to_numpy(), rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# thresholdedMean
# ----------------------------------------------------------------------
def test_thresholdedMean_against_R():
    gold = _read_gold("thresholdedmean.csv")
    cases = [
        (np.array([0, 0, 0, 0, 0, 1.0]), 0.1),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]), 0.1),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]), 0.1),
        (np.array([0.0]), 0.1),
        (np.array([2.0, 3.0, 4.0, 5.0, 0.0]), 0.5),
    ]
    out = [thresholdedMean(x, trim=t) for (x, t) in cases]
    np.testing.assert_allclose(out, gold["value"].to_numpy(), rtol=1e-12, atol=1e-12)


def test_thresholdedMean_NaN_raises():
    with pytest.raises(ValueError):
        thresholdedMean(np.array([2.0, 3.0, np.nan]), trim=0.1)


# ----------------------------------------------------------------------
# myElementwiseProduct
# ----------------------------------------------------------------------
def _build_sparse_from_R_ij(rows_1based, cols_1based, vals, shape):
    rows = np.asarray(rows_1based, dtype=int) - 1
    cols = np.asarray(cols_1based, dtype=int) - 1
    return sparse.csc_matrix(
        sparse.coo_matrix((vals, (rows, cols)), shape=shape)
    )


def test_myElementwiseProduct_basic():
    sm = _build_sparse_from_R_ij([1, 2, 3, 1], [1, 2, 3, 3], [2.0, 3.0, 4.0, 5.0], (3, 3))
    dv = np.array([0.5, 1.0, 2.0])
    out = myElementwiseProduct(sm, dv).toarray()
    gold = _read_gold("my_elemprod.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_myElementwiseProduct_4x4():
    sm = _build_sparse_from_R_ij([1, 3, 4], [2, 4, 1], [7.0, 8.0, 9.0], (4, 4))
    dv = np.array([2.0, 3.0, 4.0, 5.0])
    out = myElementwiseProduct(sm, dv).toarray()
    gold = _read_gold("my_elemprod2.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_myElementwiseProduct_single_nonzero():
    sm = _build_sparse_from_R_ij([2], [3], [7.5], (3, 3))
    dv = np.array([1.0, 2.0, 3.0])
    out = myElementwiseProduct(sm, dv).toarray()
    gold = _read_gold("my_elemprod3.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_myElementwiseProduct_1x1():
    sm = _build_sparse_from_R_ij([1], [1], [3.0], (1, 1))
    dv = np.array([7.0])
    out = myElementwiseProduct(sm, dv).toarray()
    gold = _read_gold("my_elemprod4.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_myElementwiseProduct_negative():
    sm = _build_sparse_from_R_ij(
        [1, 2, 2, 3], [1, 2, 3, 1], [-1.0, 2.0, -3.0, 4.0], (3, 3)
    )
    dv = np.array([-2.0, 1.0, 0.5])
    out = myElementwiseProduct(sm, dv).toarray()
    gold = _read_gold("my_elemprod5.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# elementwiseProductForSparseMatandDenseMat
# ----------------------------------------------------------------------
def test_elemprod_sd_basic():
    sm = _build_sparse_from_R_ij([1, 2, 3, 1], [1, 2, 3, 3], [2.0, 3.0, 4.0, 5.0], (3, 3))
    dm = np.arange(1, 10, dtype=float).reshape(3, 3, order="F")
    out = elementwiseProductForSparseMatandDenseMat(sm, dm).toarray()
    gold = _read_gold("elemprod_sd_a.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_elemprod_sd_rectangular():
    sm = _build_sparse_from_R_ij([1, 3], [2, 4], [7.0, 8.0], (3, 4))
    dm = np.arange(1, 13, dtype=float).reshape(3, 4, order="F")
    out = elementwiseProductForSparseMatandDenseMat(sm, dm).toarray()
    gold = _read_gold("elemprod_sd_b.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_elemprod_sd_single_nonzero():
    sm = _build_sparse_from_R_ij([2], [3], [11.0], (3, 3))
    dm = np.arange(1, 10, dtype=float).reshape(3, 3, order="F")
    out = elementwiseProductForSparseMatandDenseMat(sm, dm).toarray()
    gold = _read_gold("elemprod_sd_c.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_elemprod_sd_1x1():
    sm = _build_sparse_from_R_ij([1], [1], [3.0], (1, 1))
    dm = np.array([[2.0]])
    out = elementwiseProductForSparseMatandDenseMat(sm, dm).toarray()
    gold = _read_gold("elemprod_sd_d.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_elemprod_sd_negative():
    sm = _build_sparse_from_R_ij(
        [1, 2, 2, 3], [1, 2, 3, 1], [-1.0, 2.0, -3.0, 4.0], (3, 3)
    )
    dm = np.array(
        [[-1.0, 4.0, 7.0],
         [2.0, -5.0, 8.0],
         [3.0, 6.0, -9.0]],
    )
    out = elementwiseProductForSparseMatandDenseMat(sm, dm).toarray()
    gold = _read_gold("elemprod_sd_e.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# scaleMat
# ----------------------------------------------------------------------
def _r_matrix_3x4():
    return np.arange(1, 13, dtype=float).reshape(3, 4, order="F")


def test_scaleMat_none():
    out = scaleMat(_r_matrix_3x4(), "none")
    gold = _read_gold("scalemat_none.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_scaleMat_row():
    out = scaleMat(_r_matrix_3x4(), "row")
    gold = _read_gold("scalemat_row.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_scaleMat_col():
    out = scaleMat(_r_matrix_3x4(), "column")
    gold = _read_gold("scalemat_col.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_scaleMat_r1():
    out = scaleMat(_r_matrix_3x4(), "r1")
    gold = _read_gold("scalemat_r1.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_scaleMat_c1():
    out = scaleMat(_r_matrix_3x4(), "c1")
    gold = _read_gold("scalemat_c1.csv").to_numpy()
    np.testing.assert_allclose(out, gold, rtol=1e-12, atol=1e-12)


def test_scaleMat_pmatch_partial():
    # "ro" -> "row", "co" -> "column"
    out_ro = scaleMat(_r_matrix_3x4(), "ro")
    out_co = scaleMat(_r_matrix_3x4(), "co")
    gold_ro = _read_gold("scalemat_pmatch_ro.csv").to_numpy()
    gold_co = _read_gold("scalemat_pmatch_co.csv").to_numpy()
    np.testing.assert_allclose(out_ro, gold_ro, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_co, gold_co, rtol=1e-12, atol=1e-12)


def test_scaleMat_NA():
    mx = _r_matrix_3x4()
    mx[1, 1] = np.nan
    out_row = scaleMat(mx, "row")
    out_col = scaleMat(mx, "column")
    gold_row = _read_gold("scalemat_row_na.csv").to_numpy()
    gold_col = _read_gold("scalemat_col_na.csv").to_numpy()
    np.testing.assert_allclose(out_row, gold_row, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out_col, gold_col, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_scaleMat_invalid_argument():
    with pytest.raises(ValueError):
        scaleMat(_r_matrix_3x4(), "xyz")
    # "c" alone is ambiguous between "column" and "c1" → R returns NA → stop
    with pytest.raises(ValueError):
        scaleMat(_r_matrix_3x4(), "c")
