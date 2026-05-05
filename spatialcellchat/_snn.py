"""Shared-Nearest-Neighbour graph (``buildSNN``).

Direct port of ``SpatialCellChat::buildSNN`` (R: ``analysis.R::1641``).
The R reference dispatches into the C++ ``ComputeSNN`` routine; per the
porting contract Step 5 strategy (``port_reports/SpatialCellChat/04``)
the body becomes a small NumPy/SciPy block built around an
``S @ S.T`` co-membership product.

The R code uses ``FNN::get.knn`` with the default Euclidean kd-tree
search; SciKit-Learn's :class:`sklearn.neighbors.NearestNeighbors` with
``algorithm='kd_tree'`` produces bit-identical neighbours on the
SpatialCellChat reference tests (verified against R via
``/tmp/slice5_harness``).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

__all__ = ["buildSNN"]


def buildSNN(
    data_use: Union[np.ndarray, pd.DataFrame, sparse.spmatrix],
    k: int = 10,
    k_scale: int = 10,
    prune_SNN: float = 1.0 / 15.0,
) -> sparse.csr_matrix:
    """Build the symmetric SNN matrix from a feature x sample matrix.

    Mirrors ``SpatialCellChat::buildSNN`` (``R/analysis.R:1641``).

    Parameters
    ----------
    data_use : numpy.ndarray | pandas.DataFrame | scipy.sparse.spmatrix
        Features-by-samples matrix. The columns are the samples whose SNN
        graph is built. Sparse inputs are densified internally to mirror
        ``as.matrix(data.use)`` in the R code.
    k : int, default 10
        Number of nearest neighbours used when constructing the SNN.
        Equivalent to R's ``k`` argument.
    k_scale : int, default 10
        Granularity option for ``k`` (R's ``k.scale``). The kd-tree is
        queried for ``min(k.scale * k, n.cells - 1)`` neighbours; only
        the first ``k - 1`` are used to build the SNN, matching the R
        ``cbind(1:n, my.knn$nn.index[, 1:(k-1)])`` snippet.
    prune_SNN : float, default ``1/15``
        Cut-off applied to the Jaccard-style similarities; entries below
        this threshold are zeroed out.

    Returns
    -------
    scipy.sparse.csr_matrix
        The symmetric SNN matrix (n_samples x n_samples) with 1 on the
        diagonal, mirroring R's ``Matrix::diag(w) <- 1``.
    """
    # 1. Materialise data_use as a dense (features x samples) ndarray
    if isinstance(data_use, pd.DataFrame):
        sample_names = list(data_use.columns)
        arr = np.asarray(data_use.values, dtype=float)
    elif sparse.issparse(data_use):
        sample_names = None
        arr = np.asarray(data_use.toarray(), dtype=float)
    else:
        sample_names = None
        arr = np.asarray(data_use, dtype=float)

    n_cells = arr.shape[1]
    if n_cells < int(k):
        raise ValueError("k cannot be greater than the number of samples")

    # 2. Build the kd-tree on samples x features (R's t(as.matrix(data.use)))
    n_query = int(min(int(k_scale) * int(k), n_cells - 1))
    # NearestNeighbors returns the query point itself as the first match
    nn = NearestNeighbors(n_neighbors=n_query + 1, algorithm="kd_tree").fit(arr.T)
    indices = nn.kneighbors(arr.T, return_distance=False)[:, 1:]  # drop self

    # 3. R: nn.ranked = cbind(1:n, my.knn$nn.index[, 1:(k-1)])
    #    We are 0-indexed end-to-end so the prepended column is np.arange(n).
    nn_ranked = np.column_stack([np.arange(n_cells), indices[:, : int(k) - 1]])

    # 4. Build the boolean co-membership sparse matrix S of shape (n, n).
    rows = np.repeat(np.arange(n_cells), int(k))
    cols = nn_ranked.ravel()
    data_ones = np.ones(n_cells * int(k), dtype=float)
    S = sparse.coo_matrix((data_ones, (rows, cols)), shape=(n_cells, n_cells)).tocsr()

    # 5. SNN = S @ S.T; nnz entries are co-membership counts.
    SNN = (S @ S.T).tocsr()

    # 6. Jaccard-style normalisation: e_ij / (k + (k - e_ij))
    SNN.data = SNN.data / (int(k) + (int(k) - SNN.data))

    # 7. Prune entries below threshold
    SNN.data[SNN.data < float(prune_SNN)] = 0.0
    SNN.eliminate_zeros()

    # 8. Force the diagonal to 1 (R: ``Matrix::diag(w) <- 1``)
    SNN = SNN.tolil()
    SNN.setdiag(1.0)
    SNN = SNN.tocsr()

    if sample_names is not None:
        # We do not embed dimnames in scipy.sparse; pass back as-is.
        pass
    return SNN
