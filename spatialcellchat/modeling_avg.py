"""Average-communication-probability variants from ``modeling.R``.

Direct ports of the six ``computeAvgCommunProb*`` exports defined in
``modeling.R``. These functions take a precomputed cell-cell communication
probability (per ligand-receptor pair, as a sparse matrix or list of
sparse matrices) and aggregate it to the cell-group level under several
averaging rules.

The four ``_LR_*`` helpers operate on a single LR's sparse probability
matrix and return a dense ``(nGroup, nGroup)`` matrix:

- :func:`computeAvgCommunProb_LR_Sum`
- :func:`computeAvgCommunProb_LR_Avg`
- :func:`computeAvgCommunProb_LR_Visium_Sum`
- :func:`computeAvgCommunProb_LR_Visium_Avg`

The two top-level orchestrators iterate over LR pairs, optionally run a
permutation test, and write results into ``adata.uns['cellchat']['net']``:

- :func:`computeAvgCommunProb`        (single-cell resolution)
- :func:`computeAvgCommunProb_Visium` (low-resolution Visium with
  cell-type decomposition)

The R reference (``modeling.R`` lines noted in each docstring) is the
ground truth; correctness has been verified element-wise against R via
the harness scripts under ``/tmp/agentB_slice4``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from ._anndata import cellchat_dict

__all__ = [
    "computeAvgCommunProb_LR_Sum",
    "computeAvgCommunProb_LR_Avg",
    "computeAvgCommunProb_LR_Visium_Sum",
    "computeAvgCommunProb_LR_Visium_Avg",
    "computeAvgCommunProb",
    "computeAvgCommunProb_Visium",
]


# ---------------------------------------------------------------------------
# R format(x, digits=1) >= min.percent — faithful port
# ---------------------------------------------------------------------------
def _signif(x: np.ndarray, digits: int) -> np.ndarray:
    """Mirror R's :func:`signif` — round to ``digits`` significant figures.

    R's ``signif`` uses round-half-to-even at the specified significance
    level. ``signif(0, n) = 0``.
    """
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    for i, v in enumerate(arr.ravel()):
        if v == 0.0 or not np.isfinite(v):
            out.flat[i] = v
        else:
            d = int(np.ceil(np.log10(abs(v))))
            power = digits - d
            factor = 10.0 ** power
            # numpy uses banker's rounding by default — same as R's signif
            out.flat[i] = np.round(v * factor) / factor
    return out


def _natural_decimals(v: float) -> int:
    """Return the smallest non-negative ``d`` such that ``v == round(v, d)``.

    For ``signif``-rounded values in our range (e.g. ``0.06 -> 2``,
    ``0.1 -> 1``, ``0 -> 0``) this gives the natural decimal width R needs
    to print the value without trailing zeros.
    """
    if v == 0.0 or not np.isfinite(v):
        return 0
    # Try increasing decimals until exact representation
    for d in range(0, 16):
        if np.isclose(v, np.round(v, d), rtol=0.0, atol=1e-12):
            return d
    return 15


def _r_format_digits1_compare(x: np.ndarray, min_percent: float) -> np.ndarray:
    """Mirror R's ``format(x, digits=1) >= min.percent``.

    R's ``format(x, digits=1)``:

    1. Computes ``signif(xi, 1)`` for each element ``xi``.
    2. Determines the maximum decimal width ``D`` needed to represent
       any of the signif values exactly.
    3. Formats each ``xi`` rounded to ``D`` decimals as a fixed-width
       string (``%.Df``), padded.

    The comparison ``>= min.percent`` then coerces ``min.percent`` to
    its R default representation (which is ``as.character(min.percent)``)
    and does **lexicographic string comparison**.

    For non-negative inputs in ``[0, 1]`` and the typical ``min.percent``
    of ``0.1`` (or other values >= 0.05), this lexicographic comparison
    is equivalent to ``signif(x, 1) >= min.percent`` numerically. We
    implement it the literal way to preserve parity for unusual
    ``min.percent`` values too.
    """
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)

    signif_vals = _signif(arr, 1)
    decimals_needed = np.array(
        [_natural_decimals(v) for v in signif_vals.ravel()],
        dtype=int,
    )
    max_dec = int(np.max(decimals_needed))

    # Round originals to max_dec decimals and format
    rounded = np.round(arr, decimals=max_dec)
    fmt = f"%.{max_dec}f" if max_dec > 0 else "%g"
    if max_dec == 0:
        # R prints the integer-valued elements as e.g. "0" or "1"
        formatted = np.array([str(int(v)) if v == int(v) else f"{v}" for v in rounded])
    else:
        formatted = np.array([fmt % v for v in rounded])

    # Coerce ``min_percent`` to R's default character representation
    min_str = _r_default_str(float(min_percent))

    return formatted >= min_str


def _r_default_str(x: float) -> str:
    """Reproduce R's default ``as.character(numeric)`` representation.

    R uses 15 significant digits but trims trailing zeros and does not
    print decimal point if the value is an integer. For our compare
    threshold (typically ``0.1``), the result is exactly ``"0.1"``.
    """
    if x == int(x):
        return str(int(x))
    # R uses up to 15 sig figs by default; we use the shortest repr
    # that round-trips, matching R's typical print.
    s = ("%.15g" % x)
    return s


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Sum (R: modeling.R:1247)
# ---------------------------------------------------------------------------
def computeAvgCommunProb_LR_Sum(
    prob: sparse.spmatrix,
    group: Union[Sequence, pd.Categorical],
    dataLR: np.ndarray,
    min_percent: float = 0.1,
    min_cells_sr: int = 5,
) -> np.ndarray:
    """Compute per-group SUM of communication probabilities for one LR pair.

    Mirrors R's :func:`computeAvgCommunProb_LR_Sum` (``modeling.R:1247``):

    1. ``cell.type.mat <- model.matrix(~group-1)`` — one-hot per cell.
    2. ``Prob.avg <- crossprod(cell.type.mat, prob %*% cell.type.mat)``
       — the per-(sender_group, receiver_group) sum of probabilities.
    3. Filter by per-group ligand/receptor expression frequency
       (``format(..., digits=1) >= min.percent``).
    4. Filter by per-group sender/receiver cell counts in the binarised
       ``prob`` (``aggregate(..., sum) >= min.cells.sr``).
    5. Multiply elementwise by the two filter masks.

    Parameters
    ----------
    prob : scipy.sparse.spmatrix
        ``(nC, nC)`` cell-cell communication probability matrix for one
        ligand-receptor pair (R's ``prob.cell_[[i]]``).
    group : sequence or pandas.Categorical
        Per-cell group labels. Coerced to a ``pandas.Categorical`` if
        not already one. The categories define the row/column order of
        the output.
    dataLR : numpy.ndarray
        ``(nC, 2)`` matrix; column 0 is the per-cell ligand expression,
        column 1 is the per-cell receptor expression (R's
        ``cbind(dataLavg[i, ], dataRavg[i, ])``).
    min_percent : float, default ``0.1``
        Minimum fraction of expressing cells per group (R: ``min.percent``).
    min_cells_sr : int, default ``5``
        Minimum sender/receiver cell count per group in the binarised
        ``prob`` (R: ``min.cells.sr``).

    Returns
    -------
    numpy.ndarray
        ``(nGroup, nGroup)`` dense matrix of group-level communication
        probability sums. Returned as an all-zero matrix when the
        per-group expression filter eliminates every (sender, receiver)
        pair (R returns ``Prob_percent`` directly in that branch).
    """
    if not isinstance(group, pd.Categorical):
        group = pd.Categorical(group)
    levels = list(group.categories)
    n_groups = len(levels)

    # cell.type.mat: model.matrix(~group-1) — one-hot per cell, columns in
    # the order of the factor levels.
    codes = np.asarray(group.codes, dtype=int)
    nC = codes.size
    cell_type_mat = np.zeros((nC, n_groups), dtype=float)
    for i, c in enumerate(codes):
        if c >= 0:
            cell_type_mat[i, c] = 1.0

    # min.percent filter
    dataLR_arr = np.asarray(dataLR, dtype=float)
    if dataLR_arr.shape != (nC, 2):
        raise ValueError(
            f"dataLR must be shape (nC, 2)=({nC}, 2), got {dataLR_arr.shape}"
        )
    dataLR_indicator = (dataLR_arr > 0).astype(float)

    # aggregate by group (mean per group of indicator)
    dataLR_means = np.zeros((n_groups, 2), dtype=float)
    for g in range(n_groups):
        mask = codes == g
        if mask.any():
            dataLR_means[g, :] = dataLR_indicator[mask].mean(axis=0)
        # else: row stays at 0

    # format(...) >= min_percent  — column-wise per R's data.frame semantics
    L_percent = _r_format_digits1_compare(dataLR_means[:, 0], min_percent).astype(int)
    R_percent = _r_format_digits1_compare(dataLR_means[:, 1], min_percent).astype(int)

    # Prob_percent = crossprod(matrix(L, nrow=1), matrix(R, nrow=1)) -> outer(L, R)
    Prob_percent = np.outer(L_percent, R_percent).astype(float)

    if Prob_percent.sum() == 0:
        return Prob_percent  # already all-zero NGroup x NGroup matrix

    # Sum-aggregation: cell.type.mat^T @ prob @ cell.type.mat
    P = prob @ cell_type_mat
    if sparse.issparse(P):
        P = np.asarray(P.todense())
    Prob_avg = cell_type_mat.T @ P

    # min.cells.sr filter — uses BINARISED prob
    prob_bin = prob.copy()
    if sparse.issparse(prob_bin):
        prob_bin = prob_bin.tocsr()
        prob_bin.data = np.ones_like(prob_bin.data, dtype=float)
        sender_counts = np.asarray(prob_bin.sum(axis=1)).ravel()
        receiver_counts = np.asarray(prob_bin.sum(axis=0)).ravel()
    else:
        prob_bin = (np.asarray(prob_bin, dtype=float) != 0).astype(float)
        sender_counts = prob_bin.sum(axis=1)
        receiver_counts = prob_bin.sum(axis=0)

    # aggregate sender / receiver per group, sum
    cells_sr = np.zeros((n_groups, 2), dtype=float)
    for g in range(n_groups):
        mask = codes == g
        if mask.any():
            cells_sr[g, 0] = sender_counts[mask].sum()
            cells_sr[g, 1] = receiver_counts[mask].sum()

    sr_pass = (cells_sr >= float(min_cells_sr)).astype(int)
    cells_sr_outer = np.outer(sr_pass[:, 0], sr_pass[:, 1]).astype(float)

    return Prob_avg * Prob_percent * cells_sr_outer


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Avg (R: modeling.R:1318)
# ---------------------------------------------------------------------------
def computeAvgCommunProb_LR_Avg(
    prob: sparse.spmatrix,
    group: Union[Sequence, pd.Categorical],
    dataLR: np.ndarray,
    min_percent: float = 0.1,
    min_cells_sr: int = 5,
) -> np.ndarray:
    """Compute per-group AVERAGE of communication probabilities for one LR.

    Mirrors R's :func:`computeAvgCommunProb_LR_Avg` (``modeling.R:1318``):
    same as :func:`computeAvgCommunProb_LR_Sum` except the per-group sum
    is divided element-wise by the per-group **link count** (the same
    crossprod with the binarised ``prob``). ``NaN`` entries (zero-by-zero)
    are replaced by ``0``.

    Parameters
    ----------
    prob, group, dataLR, min_percent, min_cells_sr
        See :func:`computeAvgCommunProb_LR_Sum`.

    Returns
    -------
    numpy.ndarray
        ``(nGroup, nGroup)`` dense matrix of group-level communication
        probability averages.
    """
    if not isinstance(group, pd.Categorical):
        group = pd.Categorical(group)
    levels = list(group.categories)
    n_groups = len(levels)

    codes = np.asarray(group.codes, dtype=int)
    nC = codes.size
    cell_type_mat = np.zeros((nC, n_groups), dtype=float)
    for i, c in enumerate(codes):
        if c >= 0:
            cell_type_mat[i, c] = 1.0

    dataLR_arr = np.asarray(dataLR, dtype=float)
    if dataLR_arr.shape != (nC, 2):
        raise ValueError(
            f"dataLR must be shape (nC, 2)=({nC}, 2), got {dataLR_arr.shape}"
        )
    dataLR_indicator = (dataLR_arr > 0).astype(float)

    dataLR_means = np.zeros((n_groups, 2), dtype=float)
    for g in range(n_groups):
        mask = codes == g
        if mask.any():
            dataLR_means[g, :] = dataLR_indicator[mask].mean(axis=0)

    L_percent = _r_format_digits1_compare(dataLR_means[:, 0], min_percent).astype(int)
    R_percent = _r_format_digits1_compare(dataLR_means[:, 1], min_percent).astype(int)
    Prob_percent = np.outer(L_percent, R_percent).astype(float)

    if Prob_percent.sum() == 0:
        return Prob_percent

    # Sum across cells -> Prob.avg before scaling
    P = prob @ cell_type_mat
    if sparse.issparse(P):
        P = np.asarray(P.todense())
    Prob_avg = cell_type_mat.T @ P

    # Binarise prob and compute the per-group LINK-COUNT scale factor
    prob_bin = prob.copy()
    if sparse.issparse(prob_bin):
        prob_bin = prob_bin.tocsr()
        prob_bin.data = np.ones_like(prob_bin.data, dtype=float)
        sender_counts = np.asarray(prob_bin.sum(axis=1)).ravel()
        receiver_counts = np.asarray(prob_bin.sum(axis=0)).ravel()
        Pscale = prob_bin @ cell_type_mat
        if sparse.issparse(Pscale):
            Pscale = np.asarray(Pscale.todense())
        Prob_scale = cell_type_mat.T @ Pscale
    else:
        prob_bin_dense = (np.asarray(prob_bin, dtype=float) != 0).astype(float)
        sender_counts = prob_bin_dense.sum(axis=1)
        receiver_counts = prob_bin_dense.sum(axis=0)
        Prob_scale = cell_type_mat.T @ prob_bin_dense @ cell_type_mat

    with np.errstate(divide="ignore", invalid="ignore"):
        Prob_avg = Prob_avg / Prob_scale
    Prob_avg = np.where(np.isnan(Prob_avg), 0.0, Prob_avg)

    # min.cells.sr filter using the un-aggregated binarised counts
    cells_sr = np.zeros((n_groups, 2), dtype=float)
    for g in range(n_groups):
        mask = codes == g
        if mask.any():
            cells_sr[g, 0] = sender_counts[mask].sum()
            cells_sr[g, 1] = receiver_counts[mask].sum()

    sr_pass = (cells_sr >= float(min_cells_sr)).astype(int)
    cells_sr_outer = np.outer(sr_pass[:, 0], sr_pass[:, 1]).astype(float)

    return Prob_avg * Prob_percent * cells_sr_outer


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Visium_Sum (R: modeling.R:823)
# ---------------------------------------------------------------------------
def computeAvgCommunProb_LR_Visium_Sum(
    prob: sparse.spmatrix,
    cell_type_proportion: np.ndarray,
    cell_type_onehot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute Visium-LR sum of communication probabilities (cell-type level).

    Mirrors R's :func:`computeAvgCommunProb_LR_Visium_Sum`
    (``modeling.R:823``). For low-resolution Visium spots, each spot has
    a *proportion* of each cell type (from cell-type decomposition).
    The cell-type-pair communication strength is

    .. math::

        \\mathrm{Prob.avg} =
            \\mathrm{cell.type.proportion}^T
            \\, \\mathrm{prob} \\,
            \\mathrm{cell.type.proportion}.

    Parameters
    ----------
    prob : scipy.sparse.spmatrix
        ``(nC, nC)`` spot-level probability matrix for one LR pair
        (R's ``prob.cell_[[i]]``).
    cell_type_proportion : numpy.ndarray
        ``(nC, nTypes)`` matrix of cell-type proportions per spot
        (R's ``cell.type.decomposition``).
    cell_type_onehot : numpy.ndarray, optional
        Placeholder for signature parity with the Avg variant; unused.

    Returns
    -------
    numpy.ndarray
        ``(nTypes, nTypes)`` dense matrix of cell-type-level
        communication probability sums.
    """
    prop = np.asarray(cell_type_proportion, dtype=float)
    P = prob @ prop
    if sparse.issparse(P):
        P = np.asarray(P.todense())
    return prop.T @ P


# ---------------------------------------------------------------------------
# computeAvgCommunProb_LR_Visium_Avg (R: modeling.R:848)
# ---------------------------------------------------------------------------
def computeAvgCommunProb_LR_Visium_Avg(
    prob: sparse.spmatrix,
    cell_type_proportion: np.ndarray,
    cell_type_onehot: np.ndarray,
) -> np.ndarray:
    """Compute Visium-LR average of communication probabilities (cell-type level).

    Mirrors R's :func:`computeAvgCommunProb_LR_Visium_Avg`
    (``modeling.R:848``). Same as the ``_Sum`` variant divided by the
    per-cell-type-pair count of communication LINKS (computed using the
    binarised one-hot decomposition):

    .. math::

        \\mathrm{Prob.avg} =
            \\frac{
                \\mathrm{cell.type.proportion}^T \\, \\mathrm{prob} \\,
                \\mathrm{cell.type.proportion}
            }{
                \\mathrm{cell.type.onehot}^T \\, \\mathrm{1}(\\mathrm{prob}) \\,
                \\mathrm{cell.type.onehot}
            }.

    ``NaN`` (zero-by-zero) entries are set to ``0``.

    Parameters
    ----------
    prob : scipy.sparse.spmatrix
        ``(nC, nC)`` spot-level probability matrix for one LR pair.
    cell_type_proportion : numpy.ndarray
        ``(nC, nTypes)`` matrix of cell-type proportions per spot.
    cell_type_onehot : numpy.ndarray
        ``(nC, nTypes)`` one-hot matrix derived from
        ``cell.type.proportion > 0`` (R: ``cell.type.onehot``).

    Returns
    -------
    numpy.ndarray
        ``(nTypes, nTypes)`` dense matrix of cell-type-level
        communication probability averages.
    """
    prop = np.asarray(cell_type_proportion, dtype=float)
    onehot = np.asarray(cell_type_onehot, dtype=float)

    P = prob @ prop
    if sparse.issparse(P):
        P = np.asarray(P.todense())
    Prob_avg = prop.T @ P

    # Binarise prob and compute the per-cell-type-pair scale factor
    prob_bin = prob.copy()
    if sparse.issparse(prob_bin):
        prob_bin = prob_bin.tocsr()
        prob_bin.data = np.ones_like(prob_bin.data, dtype=float)
        Pscale = prob_bin @ onehot
        if sparse.issparse(Pscale):
            Pscale = np.asarray(Pscale.todense())
        Prob_scale = onehot.T @ Pscale
    else:
        prob_bin_dense = (np.asarray(prob_bin, dtype=float) != 0).astype(float)
        Prob_scale = onehot.T @ prob_bin_dense @ onehot

    with np.errstate(divide="ignore", invalid="ignore"):
        Prob_avg = Prob_avg / Prob_scale
    Prob_avg = np.where(np.isnan(Prob_avg), 0.0, Prob_avg)
    return Prob_avg


# ---------------------------------------------------------------------------
# Internal AnnData helpers
# ---------------------------------------------------------------------------
def _ensure_uns(adata: ad.AnnData) -> Dict:
    """Return ``adata.uns['cellchat']`` with the writable scaffolds populated.

    Routes through the shared :func:`cellchat_dict` helper for the outer
    slot, then ensures the per-call scaffolds (``net.tmp`` for permutation
    state, ``options.parameter`` for run-time arguments) exist.
    """
    cc = cellchat_dict(adata)
    if "net" not in cc:
        cc["net"] = {}
    if "tmp" not in cc["net"]:
        cc["net"]["tmp"] = {}
    if "options" not in cc:
        cc["options"] = {}
    if "parameter" not in cc["options"]:
        cc["options"]["parameter"] = {}
    return cc


def _get_group(adata: ad.AnnData, group_by: Optional[str]) -> pd.Categorical:
    """Resolve the per-cell group factor from ``group_by`` or ``active_ident``."""
    if group_by is None:
        if "active_ident" not in adata.obs.columns:
            raise ValueError(
                "computeAvgCommunProb requires `adata.obs['active_ident']` "
                "when `group_by` is None"
            )
        col = adata.obs["active_ident"]
    else:
        if group_by not in adata.obs.columns:
            raise ValueError(
                f"The 'group_by'={group_by!r} is not a column name in adata.obs"
            )
        col = adata.obs[group_by]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return pd.Categorical(col)
    return pd.Categorical(col)


def _get_prob_cell_list(adata: ad.AnnData) -> List[sparse.spmatrix]:
    """Pull the per-LR sparse-matrix list ``prob.cell_`` from ``adata.uns``.

    Mirrors R's ``object@net$tmp$prob.cell``. Sub-agent A's ``computeCommunProb``
    writes this slot as a ``dict[str, csc_matrix]`` keyed by LR name; older
    callers may pass a plain list of sparse matrices.
    """
    cc = adata.uns.get("cellchat", {})
    net = cc.get("net", {})
    tmp = net.get("tmp", {})
    if "prob.cell" not in tmp:
        raise RuntimeError(
            "Please run `computeCommunProb` to compute the communication "
            "probability/strength between any interacting individual cells!"
        )
    payload = tmp["prob.cell"]
    if isinstance(payload, dict):
        return list(payload.values())
    return list(payload)


def _get_prob_cell_array(adata: ad.AnnData) -> np.ndarray:
    """Pull the dense ``prob.cell`` 3-D array (nC, nC, nLR) from ``adata.uns``.

    Mirrors R's ``object@net$prob.cell``. Sub-agent A writes a sparse.COO
    here; densify before returning.
    """
    cc = adata.uns.get("cellchat", {})
    net = cc.get("net", {})
    if "prob.cell" not in net:
        raise RuntimeError(
            "Please run `computeCommunProb` to compute the communication "
            "probability/strength between any interacting individual cells!"
        )
    arr = net["prob.cell"]
    if hasattr(arr, "todense"):
        return np.asarray(arr.todense())
    return np.asarray(arr)


def _get_LR_names(adata: ad.AnnData) -> List[str]:
    """Pull the LR-pair names (third dim of ``prob.cell``)."""
    cc = adata.uns.get("cellchat", {})
    net = cc.get("net", {})
    if "LR.names" in net:
        return list(net["LR.names"])
    # Sub-agent A keys the per-LR dict by LR name — preserve that order.
    tmp = net.get("tmp", {})
    payload = tmp.get("prob.cell")
    if isinstance(payload, dict):
        return list(payload.keys())
    # Fall back to indices: read .shape[2] directly (works on both sparse.COO
    # and ndarray; np.asarray would error on sparse.COO without auto-densify).
    if "prob.cell" in net:
        n = int(net["prob.cell"].shape[2])
        return [f"LR{i + 1}" for i in range(n)]
    return []


# ---------------------------------------------------------------------------
# computeAvgCommunProb (R: modeling.R:989)
# ---------------------------------------------------------------------------
def computeAvgCommunProb(
    adata: ad.AnnData,
    group_by: Optional[str] = None,
    avg_type: Literal["avg", "sum"] = "avg",
    min_percent: float = 0.1,
    min_cells_sr: int = 5,
    do_permutation: bool = True,
    nboot: int = 100,
    seed_use: int = 1,
    colocalization_use: bool = False,
    thresh_colo: float = 0.05,
) -> ad.AnnData:
    """Compute group-level cell-cell communication.

    Mirrors R's :func:`computeAvgCommunProb` (``modeling.R:989``). Reads
    the precomputed per-LR sparse cell-cell probability matrices from
    ``adata.uns['cellchat']['net']['tmp']['prob.cell']``, aggregates them
    to the cell-group level using one of two rules (``"avg"`` or
    ``"sum"``), and writes the result into
    ``adata.uns['cellchat']['net']['prob']`` (and ``['pval']`` when
    ``do_permutation=True``).

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData with ``computeCommunProb`` results stored in
        ``uns['cellchat']['net']['tmp']['prob.cell']`` (list of sparse
        ``(nC, nC)`` matrices, one per LR pair) and the dense 3-D array
        in ``uns['cellchat']['net']['prob.cell']``.
    group_by : str, optional
        ``adata.obs`` column to use as the per-cell group factor. When
        ``None``, defaults to ``adata.obs['active_ident']`` (R's
        ``object@idents``).
    avg_type : {"avg", "sum"}, default ``"avg"``
        Aggregation rule. ``"avg"`` divides the per-group sum by the
        per-group communication-link count; ``"sum"`` returns the raw
        per-group sum.
    min_percent : float, default ``0.1``
        Minimum fraction of expressing cells per group required for the
        per-group filter (R: ``min.percent``).
    min_cells_sr : int, default ``5``
        Minimum sender/receiver cell count per group required for the
        per-group filter (R: ``min.cells.sr``).
    do_permutation : bool, default ``True``
        Whether to compute permutation-test p-values (R: ``do.permutation``).
    nboot : int, default ``100``
        Number of permutations (R: ``nboot``).
    seed_use : int, default ``1``
        RNG seed for the permutation test (R: ``seed.use``). Note that R's
        Mersenne-Twister stream is non-portable; the resulting p-values
        will not be bit-for-bit identical to R's.
    colocalization_use : bool, default ``False``
        Whether to mask out spatially-distant cell groups based on
        :func:`spatialcellchat.spatial.computeColocalization` (R:
        ``colocalization.use``).
    thresh_colo : float, default ``0.05``
        Colocalization p-value threshold (R: ``thresh.colo``).

    Returns
    -------
    anndata.AnnData
        The same ``adata`` (mutated) with the following entries written:

        - ``uns['cellchat']['net']['prob']`` — ``(nGroup, nGroup, nLR)``
          dense array of group-level communication probabilities.
        - ``uns['cellchat']['net']['pval']`` — same shape, permutation
          p-values (``None`` when ``do_permutation=False``).
        - ``uns['cellchat']['net']['tmp']['LRsig.CCC.counts']`` —
          per-LR cell-level link count.
        - ``uns['cellchat']['net']['tmp']['LRsig.GGC.counts']`` —
          per-LR group-level link count after aggregation.
        - ``uns['cellchat']['net']['tmp']['LRsig.use.idx']`` —
          0-based indices of LRs with at least one cell-level link.
        - ``uns['cellchat']['options']['parameter']`` — the call
          parameters.
    """
    cc = _ensure_uns(adata)
    group = _get_group(adata, group_by)
    levels = list(group.categories)
    n_clusters = len(levels)

    # The R guard: numCluster != length(unique(group)) -> stop
    present = set(group.dropna().unique())
    if n_clusters != len(present):
        raise ValueError(
            "Please check `unique(adata.obs[group_by])` and ensure that the "
            "factor levels are correct! You may need to drop unused levels."
        )

    prob_cell_list = _get_prob_cell_list(adata)
    LR_names = _get_LR_names(adata)
    nLR = len(prob_cell_list)
    if nLR != len(LR_names):
        # Use indices when names are not aligned
        LR_names = [f"LR{i + 1}" for i in range(nLR)]

    # data.use scaling: in R, data.signaling/data.project is scaled by max,
    # then dataLavg/dataRavg are read out of net$tmp. We rely on the caller
    # (sub-agent A's `computeCommunProb`) having written these.
    net_tmp = cc["net"]["tmp"]
    if "Lavg" not in net_tmp or "Ravg" not in net_tmp:
        raise RuntimeError(
            "computeAvgCommunProb requires `adata.uns['cellchat']['net']"
            "['tmp']['Lavg']` and `['Ravg']` (per-LR x per-cell ligand and "
            "receptor average expression) — produced by computeCommunProb."
        )
    dataLavg = np.asarray(net_tmp["Lavg"], dtype=float)  # (nLR, nC)
    dataRavg = np.asarray(net_tmp["Ravg"], dtype=float)  # (nLR, nC)

    if avg_type not in ("avg", "sum"):
        raise ValueError(f"avg_type must be one of 'avg', 'sum' (got {avg_type!r})")
    LR_func = computeAvgCommunProb_LR_Avg if avg_type == "avg" else computeAvgCommunProb_LR_Sum

    if colocalization_use:
        from .spatial import computeColocalization  # local import to avoid cycle
        if "spatial" not in adata.obsm:
            raise ValueError(
                "colocalization_use=True requires adata.obsm['spatial']"
            )
        pval_colo = computeColocalization(
            adata.obsm["spatial"], group=group, nboot=nboot, seed_use=seed_use
        ).to_numpy()
    else:
        pval_colo = np.zeros((n_clusters, n_clusters), dtype=float)

    Prob = np.zeros((n_clusters, n_clusters, nLR), dtype=float)
    Pval = np.ones((n_clusters, n_clusters, nLR), dtype=float)

    rng = np.random.default_rng(int(seed_use))

    # per-LR cell-level link count = nnz of each sparse matrix
    prob_sum = np.array([m.nnz if sparse.issparse(m) else int(np.count_nonzero(m))
                         for m in prob_cell_list], dtype=int)
    LR_use_idx = np.flatnonzero(prob_sum > 0)
    cc["net"]["tmp"]["LRsig.CCC.counts"] = prob_sum
    cc["net"]["tmp"]["LRsig.use.idx"] = LR_use_idx

    if LR_use_idx.size == 0:
        raise ValueError("Each LR pair does not have any cell-level links/interactions.")

    for x in range(LR_use_idx.size):
        i = int(LR_use_idx[x])
        prob_i = prob_cell_list[i]
        dataLR_i = np.column_stack([dataLavg[i, :], dataRavg[i, :]])
        Prob_avg = LR_func(
            prob_i,
            group=group,
            dataLR=dataLR_i,
            min_percent=min_percent,
            min_cells_sr=min_cells_sr,
        )
        Prob_avg = np.where(pval_colo > thresh_colo, 0.0, Prob_avg)
        Prob[:, :, i] = Prob_avg

    # Per-LR group-level link count after aggregation
    prob_sum_group = np.sum(Prob > 0, axis=(0, 1))
    cc["net"]["tmp"]["LRsig.GGC.counts"] = prob_sum_group
    LR_use_idx_perm = np.flatnonzero(prob_sum_group > 0)

    if do_permutation and LR_use_idx_perm.size > 0:
        nC = int(prob_cell_list[0].shape[0])
        permutation = np.column_stack([
            rng.permutation(nC) for _ in range(int(nboot))
        ])  # nC x nboot
        for x in range(LR_use_idx_perm.size):
            i = int(LR_use_idx_perm[x])
            prob_i = prob_cell_list[i]
            dataLR_i = np.column_stack([dataLavg[i, :], dataRavg[i, :]])
            Pnull = Prob[:, :, i].ravel(order="F")
            Pboot = np.empty((Pnull.size, int(nboot)), dtype=float)
            for nE in range(int(nboot)):
                group_boot = pd.Categorical(
                    np.asarray(group)[permutation[:, nE]],
                    categories=group.categories,
                )
                Pboot_avg = LR_func(
                    prob_i,
                    group=group_boot,
                    dataLR=dataLR_i,
                    min_percent=min_percent,
                    min_cells_sr=min_cells_sr,
                )
                Pboot[:, nE] = Pboot_avg.ravel(order="F")
            n_reject = np.sum(Pboot - Pnull[:, None] > 0, axis=1)
            p = n_reject.astype(float) / float(nboot)
            Pval[:, :, i] = p.reshape((n_clusters, n_clusters), order="F")

        # P-values for zero-prob entries are forced to 1
        Pval = np.where(Prob == 0, 1.0, Pval)
    elif not do_permutation:
        Pval = None  # type: ignore[assignment]

    cc["net"]["prob"] = Prob
    cc["net"]["pval"] = Pval
    cc["net"]["dimnames"] = {
        "source": list(levels),
        "target": list(levels),
        # Mirror both 'pairs' and 'interaction_name' for cross-module
        # compatibility (some viz functions read 'pairs', others read
        # 'interaction_name'; see slice-7 #50-53).
        "interaction_name": list(LR_names),
        "pairs": list(LR_names),
    }

    params = cc["options"]["parameter"]
    params["min.percent"] = float(min_percent)
    params["min.cells.sr"] = int(min_cells_sr)
    params["do.permutation"] = bool(do_permutation)
    params["nboot"] = int(nboot)
    params["avg.type"] = str(avg_type)
    params["seed.use"] = int(seed_use)
    params["colocalization.use"] = bool(colocalization_use)
    params["thresh.colo"] = float(thresh_colo)

    if colocalization_use:
        if "images" not in cc:
            cc["images"] = {}
        cc["images"]["colocalization"] = pval_colo

    return adata


# ---------------------------------------------------------------------------
# computeAvgCommunProb_Visium (R: modeling.R:603)
# ---------------------------------------------------------------------------
def computeAvgCommunProb_Visium(
    adata: ad.AnnData,
    cell_type_decomposition: Union[np.ndarray, pd.DataFrame],
    avg_type: Literal["avg", "sum"] = "avg",
    group_by: Optional[str] = None,
    do_permutation: bool = True,
    nboot: int = 100,
    seed_use: int = 1,
    colocalization_use: bool = False,
    thresh_colo: float = 0.05,
) -> ad.AnnData:
    """Compute group-level communication for low-resolution Visium data.

    Mirrors R's :func:`computeAvgCommunProb_Visium` (``modeling.R:603``).
    Differs from :func:`computeAvgCommunProb` in that the per-cell
    aggregation uses a *cell-type decomposition* matrix (rows are spots,
    columns are cell types, entries are proportions) instead of a hard
    per-cell group assignment. Permutation shuffles the *columns* of the
    decomposition matrix (R: ``cell.type.decomposition[, permutation[, nE]]``).

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial AnnData with ``computeCommunProb`` results.
    cell_type_decomposition : numpy.ndarray or pandas.DataFrame
        ``(nC, nTypes)`` cell-type proportion matrix. Column names must
        match the levels of the resolved group factor (R guard at
        modeling.R:631-632); row names must match the cells of ``adata``
        (R guard at modeling.R:634-635).
    avg_type : {"avg", "sum"}, default ``"avg"``
        Aggregation rule for the LR-Visium variants.
    group_by : str, optional
        ``adata.obs`` column for the group factor (used only to derive
        the level order; the actual aggregation is column-wise on
        ``cell_type_decomposition``).
    do_permutation, nboot, seed_use, colocalization_use, thresh_colo
        See :func:`computeAvgCommunProb`.

    Returns
    -------
    anndata.AnnData
        The same ``adata`` (mutated) with the same entries as written by
        :func:`computeAvgCommunProb`, plus
        ``uns['cellchat']['net']['tmp']['cell.type.decomposition']``.
    """
    cc = _ensure_uns(adata)
    group = _get_group(adata, group_by)
    levels = list(group.categories)
    n_clusters = len(levels)

    # cell.type.decomposition validation
    if isinstance(cell_type_decomposition, pd.DataFrame):
        if list(cell_type_decomposition.columns) != levels:
            raise ValueError(
                "Please check your `cell_type_decomposition`'s columns, and "
                "make sure they are the same as the cell groups' names "
                "in adata."
            )
        if list(cell_type_decomposition.index) != list(adata.obs_names):
            raise ValueError(
                "Please check your `cell_type_decomposition`'s rows, and "
                "make sure they are the same as the cells' names in adata."
            )
        decomp = cell_type_decomposition.to_numpy(dtype=float)
    else:
        decomp = np.asarray(cell_type_decomposition, dtype=float)
        if decomp.shape[1] != n_clusters:
            raise ValueError(
                f"cell_type_decomposition has {decomp.shape[1]} columns; "
                f"expected {n_clusters} (matching the number of group levels)."
            )
        if decomp.shape[0] != adata.n_obs:
            raise ValueError(
                f"cell_type_decomposition has {decomp.shape[0]} rows; "
                f"expected {adata.n_obs} (matching the number of cells)."
            )

    decomp_onehot = (decomp > 0).astype(float)

    prob_cell_list = _get_prob_cell_list(adata)
    LR_names = _get_LR_names(adata)
    nLR = len(prob_cell_list)
    if nLR != len(LR_names):
        LR_names = [f"LR{i + 1}" for i in range(nLR)]

    if avg_type not in ("avg", "sum"):
        raise ValueError(f"avg_type must be one of 'avg', 'sum' (got {avg_type!r})")
    Visium_func = (
        computeAvgCommunProb_LR_Visium_Avg
        if avg_type == "avg" else computeAvgCommunProb_LR_Visium_Sum
    )

    if colocalization_use:
        from .spatial import computeColocalization
        if "spatial" not in adata.obsm:
            raise ValueError(
                "colocalization_use=True requires adata.obsm['spatial']"
            )
        pval_colo = computeColocalization(
            adata.obsm["spatial"], group=group, nboot=nboot, seed_use=seed_use
        ).to_numpy()
    else:
        pval_colo = np.zeros((n_clusters, n_clusters), dtype=float)

    Prob = np.zeros((n_clusters, n_clusters, nLR), dtype=float)
    Pval = np.ones((n_clusters, n_clusters, nLR), dtype=float)

    rng = np.random.default_rng(int(seed_use))

    prob_sum = np.array([m.nnz if sparse.issparse(m) else int(np.count_nonzero(m))
                         for m in prob_cell_list], dtype=int)
    LR_use_idx = np.flatnonzero(prob_sum > 0)
    cc["net"]["tmp"]["LRsig.CCC.counts"] = prob_sum
    cc["net"]["tmp"]["LRsig.use.idx"] = LR_use_idx

    if LR_use_idx.size == 0:
        raise ValueError("Each LR pair does not have any spot-level links/interactions.")

    for x in range(LR_use_idx.size):
        i = int(LR_use_idx[x])
        prob_i = prob_cell_list[i]
        Prob_avg = Visium_func(
            prob_i,
            cell_type_proportion=decomp,
            cell_type_onehot=decomp_onehot,
        )
        Prob_avg = np.where(pval_colo > thresh_colo, 0.0, Prob_avg)
        Prob[:, :, i] = Prob_avg

    prob_sum_group = np.sum(Prob > 0, axis=(0, 1))
    cc["net"]["tmp"]["LRsig.GGC.counts"] = prob_sum_group
    LR_use_idx_perm = np.flatnonzero(prob_sum_group > 0)

    if do_permutation and LR_use_idx_perm.size > 0:
        # permutation: numCluster x nboot — permutes the COLUMNS of decomp
        permutation = np.column_stack([
            rng.permutation(n_clusters) for _ in range(int(nboot))
        ])
        for x in range(LR_use_idx_perm.size):
            i = int(LR_use_idx_perm[x])
            prob_i = prob_cell_list[i]
            Pnull = Prob[:, :, i].ravel(order="F")
            Pboot = np.empty((Pnull.size, int(nboot)), dtype=float)
            for nE in range(int(nboot)):
                decomp_boot = decomp[:, permutation[:, nE]]
                onehot_boot = (decomp_boot > 0).astype(float)
                Pboot_avg = Visium_func(
                    prob_i,
                    cell_type_proportion=decomp_boot,
                    cell_type_onehot=onehot_boot,
                )
                Pboot[:, nE] = Pboot_avg.ravel(order="F")
            n_reject = np.sum(Pboot - Pnull[:, None] > 0, axis=1)
            p = n_reject.astype(float) / float(nboot)
            Pval[:, :, i] = p.reshape((n_clusters, n_clusters), order="F")

        Pval = np.where(Prob == 0, 1.0, Pval)
    elif not do_permutation:
        Pval = None  # type: ignore[assignment]

    cc["net"]["prob"] = Prob
    cc["net"]["pval"] = Pval
    cc["net"]["dimnames"] = {
        "source": list(levels),
        "target": list(levels),
        # Mirror both 'pairs' and 'interaction_name' for cross-module
        # compatibility (some viz functions read 'pairs', others read
        # 'interaction_name'; see slice-7 #50-53).
        "interaction_name": list(LR_names),
        "pairs": list(LR_names),
    }
    cc["net"]["tmp"]["cell.type.decomposition"] = decomp

    params = cc["options"]["parameter"]
    params["do.permutation"] = bool(do_permutation)
    params["nboot"] = int(nboot)
    params["avg.type"] = str(avg_type)
    params["seed.use"] = int(seed_use)
    params["colocalization.use"] = bool(colocalization_use)
    params["thresh.colo"] = float(thresh_colo)

    if colocalization_use:
        if "images" not in cc:
            cc["images"] = {}
        cc["images"]["colocalization"] = pval_colo

    return adata
