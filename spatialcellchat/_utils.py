"""Math primitives mirroring R helpers in ``modeling.R`` and ``utilities.R``.

These routines are direct ports of the R reference implementations and
preserve the same edge-case behaviour (NA handling, ``pmatch`` semantics,
sparse-matrix algorithms).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "geometricMean",
    "triMean",
    "thresholdedMean",
    "myElementwiseProduct",
    "elementwiseProductForSparseMatandDenseMat",
    "scaleMat",
]


ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list, tuple]


# ---------------------------------------------------------------------------
# geometricMean (R: modeling.R:2556)
# ---------------------------------------------------------------------------
def geometricMean(
    x: ArrayLike,
    na_rm: bool = True,
) -> Union[float, np.ndarray]:
    """Compute the geometric mean.

    Mirrors R's :func:`geometricMean`. For a 1-D vector the function returns
    a single number; for a 2-D matrix it returns column-wise geometric means.

    Parameters
    ----------
    x : array-like
        Numeric vector or 2-D matrix. ``NaN`` is treated as R's ``NA``.
    na_rm : bool, default ``True``
        Whether to drop NaN before averaging logs.

    Returns
    -------
    float or numpy.ndarray
        The geometric mean. A scalar for vectors; a 1-D ``ndarray`` of
        column means for matrices.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim <= 1:
        # vector branch — corresponds to R's ``is.null(nrow(x))``
        log_arr = np.log(arr)
        if na_rm:
            return float(np.exp(np.nanmean(log_arr)))
        return float(np.exp(np.mean(log_arr)))
    # matrix branch: column-wise mean of logs
    log_arr = np.log(arr)
    if na_rm:
        return np.exp(np.nanmean(log_arr, axis=0))
    return np.exp(np.mean(log_arr, axis=0))


# ---------------------------------------------------------------------------
# triMean (R: modeling.R:2580)
# ---------------------------------------------------------------------------
def triMean(x: ArrayLike, na_rm: bool = True) -> float:
    """Compute Tukey's trimean.

    Implements R's :func:`triMean`: the mean of the quantiles at probabilities
    ``(0.25, 0.50, 0.50, 0.75)``. Quantiles are computed with R's default
    interpolation (type 7), which is also numpy's default.

    Parameters
    ----------
    x : array-like
        Numeric vector.
    na_rm : bool, default ``True``
        Whether to drop ``NaN`` values before computing quantiles.

    Returns
    -------
    float
        The trimean.
    """
    arr = np.asarray(x, dtype=float)
    probs = np.array([0.25, 0.50, 0.50, 0.75])
    if na_rm:
        q = np.nanquantile(arr, probs)
    else:
        q = np.quantile(arr, probs)
    return float(np.mean(q))


# ---------------------------------------------------------------------------
# thresholdedMean (R: modeling.R:2591)
# ---------------------------------------------------------------------------
def thresholdedMean(x: ArrayLike, trim: float = 0.1, na_rm: bool = True) -> float:
    """Mean of ``x`` if the fraction of non-zero elements is at least ``trim``.

    Mirrors R's :func:`thresholdedMean`. Returns ``0`` when the proportion of
    expressing cells (non-zero entries) is below ``trim``.

    Parameters
    ----------
    x : array-like
        Numeric vector.
    trim : float, default ``0.1``
        Threshold on the proportion of non-zero entries.
    na_rm : bool, default ``True``
        Whether to drop ``NaN`` values when computing the mean.

    Returns
    -------
    float
        Either ``0`` or the (NaN-aware) mean of ``x``.
    """
    arr = np.asarray(x, dtype=float)
    # R's Matrix::nnzero returns NA when ``x`` contains any NA, so
    # ``percent`` is NA and the ``if (percent < trim)`` test errors with
    # "missing value where TRUE/FALSE needed". Mirror that here.
    if np.any(np.isnan(arr)):
        raise ValueError("missing value where TRUE/FALSE needed")
    percent = float(np.count_nonzero(arr)) / float(arr.size)
    if percent < trim:
        return 0.0
    if na_rm:
        return float(np.nanmean(arr))
    return float(np.mean(arr))


# ---------------------------------------------------------------------------
# myElementwiseProduct (R: modeling.R:419)
# ---------------------------------------------------------------------------
def myElementwiseProduct(
    SparseMat: sparse.spmatrix,
    DenseVec: ArrayLike,
) -> sparse.csc_matrix:
    """Multiply a sparse matrix elementwise by ``outer(DenseVec, DenseVec)``.

    Mirrors R's :func:`myElementwiseProduct`. For each non-zero entry
    ``(i, j)`` of ``SparseMat`` the value is multiplied by
    ``DenseVec[i] * DenseVec[j]``.

    Parameters
    ----------
    SparseMat : scipy.sparse.spmatrix
        Sparse matrix; must be square.
    DenseVec : array-like
        Numeric vector with length matching the side of ``SparseMat``.

    Returns
    -------
    scipy.sparse.csc_matrix
        The elementwise product as a CSC sparse matrix.
    """
    coo = SparseMat.tocoo()
    vec = np.asarray(DenseVec, dtype=float)
    new_x = coo.data * vec[coo.row] * vec[coo.col]
    out = sparse.coo_matrix((new_x, (coo.row, coo.col)), shape=coo.shape)
    return out.tocsc()


# ---------------------------------------------------------------------------
# elementwiseProductForSparseMatandDenseMat (R: modeling.R:390)
# ---------------------------------------------------------------------------
def elementwiseProductForSparseMatandDenseMat(
    SparseMat: sparse.spmatrix,
    DenseMat: ArrayLike,
) -> sparse.csc_matrix:
    """Elementwise product of a sparse matrix and a dense matrix.

    Mirrors R's :func:`elementwiseProductForSparseMatandDenseMat`. Visits each
    non-zero ``(i, j)`` of ``SparseMat`` and multiplies by ``DenseMat[i, j]``.

    Parameters
    ----------
    SparseMat : scipy.sparse.spmatrix
        Sparse matrix.
    DenseMat : array-like
        Dense matrix with the same shape as ``SparseMat``.

    Returns
    -------
    scipy.sparse.csc_matrix
        The elementwise product as a CSC sparse matrix.
    """
    coo = SparseMat.tocoo()
    dense = np.asarray(DenseMat, dtype=float)
    if dense.shape != coo.shape:
        raise ValueError(
            f"DenseMat shape {dense.shape} does not match SparseMat shape {coo.shape}"
        )
    new_x = coo.data * dense[coo.row, coo.col]
    out = sparse.coo_matrix((new_x, (coo.row, coo.col)), shape=coo.shape)
    return out.tocsc()


# ---------------------------------------------------------------------------
# scaleMat (R: utilities.R:821)
# ---------------------------------------------------------------------------
_SCALE_OPTIONS = ("none", "row", "column", "r1", "c1")


def _r_pmatch(arg: str, choices: tuple) -> str:
    """Replicate R's ``pmatch``: exact match wins; otherwise unique prefix.

    Returns the matched element of ``choices``. Raises ``ValueError`` when
    no exact match exists and the prefix match is either absent or
    ambiguous — mirroring R's NA-then-stop in :func:`scaleMat`.
    """
    if arg in choices:
        return arg
    matches = [c for c in choices if c.startswith(arg)]
    if len(matches) != 1:
        raise ValueError(
            "scale argument shoud take values: 'none', 'row' or 'column'"
        )
    return matches[0]


def scaleMat(
    x: ArrayLike,
    scale: str,
    na_rm: bool = True,
) -> np.ndarray:
    """Scale a data matrix.

    Mirrors R's :func:`scaleMat`. ``scale`` accepts (with ``pmatch``) one of
    ``"none"``, ``"row"``, ``"column"``, ``"r1"``, or ``"c1"``.

    Row/column modes z-score along the chosen axis (subtract mean, divide by
    sample standard deviation with ``ddof = 1``). ``r1`` divides by the row
    sum and ``c1`` by the column sum.

    Parameters
    ----------
    x : array-like
        Two-dimensional numeric matrix.
    scale : str
        One of ``"none"``, ``"row"``, ``"column"``, ``"r1"``, ``"c1"``;
        partial matches accepted.
    na_rm : bool, default ``True``
        Whether to ignore ``NaN`` when computing means / sums / sds.

    Returns
    -------
    numpy.ndarray
        The scaled matrix.
    """
    matched = _r_pmatch(scale, _SCALE_OPTIONS)
    arr = np.asarray(x, dtype=float)

    if matched == "none":
        return arr

    if matched == "row":
        if na_rm:
            row_mean = np.nanmean(arr, axis=1, keepdims=True)
        else:
            row_mean = np.mean(arr, axis=1, keepdims=True)
        centered = arr - row_mean
        if na_rm:
            row_sd = np.nanstd(centered, axis=1, ddof=1, keepdims=True)
        else:
            row_sd = np.std(centered, axis=1, ddof=1, keepdims=True)
        return centered / row_sd

    if matched == "column":
        if na_rm:
            col_mean = np.nanmean(arr, axis=0, keepdims=True)
        else:
            col_mean = np.mean(arr, axis=0, keepdims=True)
        centered = arr - col_mean
        if na_rm:
            col_sd = np.nanstd(centered, axis=0, ddof=1, keepdims=True)
        else:
            col_sd = np.std(centered, axis=0, ddof=1, keepdims=True)
        return centered / col_sd

    if matched == "r1":
        if na_rm:
            row_sum = np.nansum(arr, axis=1, keepdims=True)
        else:
            row_sum = np.sum(arr, axis=1, keepdims=True)
        return arr / row_sum

    # c1
    if na_rm:
        col_sum = np.nansum(arr, axis=0, keepdims=True)
    else:
        col_sum = np.sum(arr, axis=0, keepdims=True)
    return arr / col_sum
