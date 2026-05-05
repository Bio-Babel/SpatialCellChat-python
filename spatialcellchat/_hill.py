"""Hill function and spatial-probability primitives from ``modeling.R``.

These are pure-numerical helpers used by :func:`computeCommunProb`. Both are
direct ports of R's :func:`HillFunctionFordataLR` (modeling.R:369) and
:func:`createPspatialFrom_dspatial` (modeling.R:337) and operate on
``scipy.sparse`` CSC matrices --- the Python equivalent of R's ``dgCMatrix``
/ ``CsparseMatrix``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

__all__ = [
    "HillFunctionFordataLR",
    "createPspatialFrom_dspatial",
]


# ---------------------------------------------------------------------------
# HillFunctionFordataLR (R: modeling.R:369)
# ---------------------------------------------------------------------------
def HillFunctionFordataLR(
    dataLR: sparse.spmatrix,
    Kh: float,
    n: float,
) -> sparse.csc_matrix:
    """Apply a Hill function to every non-zero entry of a sparse L-R matrix.

    Mirrors R's :func:`HillFunctionFordataLR`. For each non-zero entry
    ``x`` of ``dataLR``, the value is replaced by ``x**n / (Kh**n + x**n)``.
    The sparsity pattern is preserved exactly --- explicit zeros that exist
    in the input ``@x`` slot remain explicit zeros (mirroring R's
    ``purrr::map_dbl`` over ``@x``).

    Parameters
    ----------
    dataLR : scipy.sparse.spmatrix
        Sparse ligand-receptor matrix; treated as CSC (R's ``dgCMatrix``).
    Kh : float
        EC50 parameter of the Hill function. Default in R is ``0.5``;
        this port has no default --- callers (e.g. ``computeCommunProb``)
        pass it explicitly to mirror R's behaviour.
    n : float
        Hill coefficient. Default in R is ``1``; passed explicitly here.

    Returns
    -------
    scipy.sparse.csc_matrix
        New CSC matrix with the same shape and sparsity pattern as
        ``dataLR`` but with the Hill function applied element-wise to the
        stored values.
    """
    csc = dataLR.tocsc(copy=True)
    x = csc.data.astype(float, copy=False)
    # x^n / (Kh^n + x^n) --- applied element-wise to the stored values
    Kh_n = Kh**n
    x_n = np.power(x, n)
    csc.data = x_n / (Kh_n + x_n)
    return csc


# ---------------------------------------------------------------------------
# createPspatialFrom_dspatial (R: modeling.R:337)
# ---------------------------------------------------------------------------
def createPspatialFrom_dspatial(
    d_spatial: sparse.spmatrix,
    distance_use: bool = True,
) -> sparse.csc_matrix:
    """Convert a scaled cell-cell distance matrix to a spatial probability.

    Mirrors R's :func:`createPspatialFrom_dspatial`. The input ``d_spatial``
    is a sparse cell-cell distance matrix that has already been scaled
    (typically by :func:`computeCommunProb` so that the minimum distance
    is in ``[1, 2]``). The transformation is

    1. Take the reciprocal of every non-zero entry: ``P@x <- 1 / d@x``.
    2. Fill the diagonal with ``max(P@x)`` --- the largest reciprocated
       distance --- so autocrine pairs receive the strongest spatial
       weight. The maximum is taken **before** the diagonal is filled
       (matching R's evaluation order in
       ``Matrix::diag(P) <- max(P@x)``).
    3. If ``distance_use`` is ``False``, every stored entry is then set
       to ``1`` to give a binary connectivity matrix bounded by the
       interaction range.

    Parameters
    ----------
    d_spatial : scipy.sparse.spmatrix
        Sparse cell-cell distance matrix (R's ``CsparseMatrix``); must
        be square. Stored values must already be scaled to be greater
        than or equal to one for the resulting reciprocals to fall in
        ``(0, 1]``.
    distance_use : bool, default ``True``
        Whether the cell-cell communication probability should remain
        inversely proportional to distance (``True``) or be made binary
        (``False``).

    Returns
    -------
    scipy.sparse.csc_matrix
        The spatial probability matrix, returned in CSC form.
    """
    csc = d_spatial.tocsc(copy=True)
    # Reciprocate stored values: P@x <- 1 / d@x
    csc.data = 1.0 / csc.data.astype(float, copy=False)
    # max(P@x) is evaluated BEFORE the diagonal assignment in R
    diag_value = float(np.max(csc.data))
    # Fill the diagonal --- adds entries that did not previously exist
    csc = csc.tolil()
    csc.setdiag(diag_value)
    csc = csc.tocsc()
    if distance_use:
        return csc
    # Binary mode: set every stored value to 1 (preserving sparsity pattern)
    csc.data = np.ones_like(csc.data)
    return csc
