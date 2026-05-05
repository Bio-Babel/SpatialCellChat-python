"""Expression helpers for the cell-cell communication model.

Direct ports of the eight ``computeExpr*`` / ``computeAveExpr`` functions in
``SpatialCellChat::modeling.R`` (lines 2170--2548). They are the per-LR-pair
expression aggregators consumed by :func:`computeCommunProb` (in
``modeling.py``) and :func:`computeAvgCommunProb` (in ``modeling_avg.py``).

The R reference uses gene-by-cell matrices; AnnData stores them transposed.
Every helper here therefore expects a ``data_use`` matrix in
**genes-by-cells** orientation, matching R exactly. Callers (the public
:func:`computeCommunProb`) take care of the transpose.

Functions
---------
- :func:`computeAveExpr` (R: modeling.R:2170)
- :func:`computeExpr_complex` (R: modeling.R:2222)
- :func:`computeExpr_LR` (R: modeling.R:2288)
- :func:`computeExpr_coreceptor` (R: modeling.R:2316)
- :func:`computeExpr_agonist` (R: modeling.R:2491)
- :func:`computeExpr_antagonist` (R: modeling.R:2527)
- :func:`computeExprGroup_agonist` (R: modeling.R:2425)
- :func:`computeExprGroup_antagonist` (R: modeling.R:2458)
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._utils import geometricMean, thresholdedMean, triMean
from .database import _is_blank_or_na

__all__ = [
    "computeAveExpr",
    "computeExpr_complex",
    "computeExpr_LR",
    "computeExpr_coreceptor",
    "computeExpr_agonist",
    "computeExpr_antagonist",
    "computeExprGroup_agonist",
    "computeExprGroup_antagonist",
]


ArrayLike = Union[np.ndarray, pd.DataFrame]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _make_FunMean(
    type: Literal["triMean", "truncatedMean", "median"],
    trim: Optional[float],
) -> Callable[[np.ndarray], float]:
    """Return the per-group reduction matching R's ``FunMean`` switch.

    Mirrors the ``switch(type, ...)`` block at the top of every
    ``computeExpr*Group`` / ``computeAveExpr`` function in modeling.R.
    """
    if type == "triMean":
        return lambda x: triMean(x, na_rm=True)
    if type == "truncatedMean":
        if trim is None:
            # R's ``mean(x, trim = NULL)`` errors with "invalid 'trim' argument".
            raise ValueError("invalid 'trim' argument")
        t = float(trim)

        def truncated_mean(x: np.ndarray) -> float:
            arr = np.asarray(x, dtype=float)
            arr = arr[~np.isnan(arr)]
            n = arr.size
            if n == 0:
                return float("nan")
            if t <= 0.0:
                return float(np.mean(arr))
            if t >= 0.5:
                return float(np.median(arr))
            arr = np.sort(arr)
            lo = int(np.floor(t * n))
            hi = n - lo
            return float(np.mean(arr[lo:hi]))

        return truncated_mean
    if type == "median":
        return lambda x: float(np.nanmedian(np.asarray(x, dtype=float)))
    # R's ``match.arg`` would error here.
    raise ValueError(
        "'arg' should be one of 'triMean', 'truncatedMean', 'median'"
    )


def _aggregate_by_group(
    data_use: np.ndarray,
    group: pd.Categorical,
    FunMean: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, list]:
    """Return the (genes x nlevels) matrix of per-group ``FunMean`` reductions.

    Mirrors R's idiom::

        data.use.avg <- aggregate(t(data.use), list(labels), FUN = FunMean)
        data.use.avg <- t(data.use.avg[,-1])

    ``data_use`` is genes-by-cells; the result is genes-by-groups in the
    factor-level order (R's ``levels(group)``). Following R's
    ``aggregate``, levels that are absent from ``group`` are silently
    dropped — the column order is the present levels in factor order.

    Returns
    -------
    (numpy.ndarray, list)
        - The aggregated matrix, shape ``(n_genes, n_groups_present)``.
        - The present level names in factor order.
    """
    if data_use.ndim == 1:
        data_use = data_use[None, :]
    n_genes = data_use.shape[0]
    levels_all = list(group.categories)
    grp_codes = np.asarray(group.codes, dtype=int)
    # R's aggregate keeps groups in factor-level order; absent levels are
    # silently dropped. Determine the present levels in factor order.
    present_codes = sorted(set(int(c) for c in grp_codes if c >= 0))
    present_levels = [levels_all[c] for c in present_codes]

    out = np.empty((n_genes, len(present_codes)), dtype=float)
    for col_idx, code in enumerate(present_codes):
        mask = grp_codes == code
        if not np.any(mask):
            out[:, col_idx] = np.nan
            continue
        for row_idx in range(n_genes):
            out[row_idx, col_idx] = FunMean(data_use[row_idx, mask])
    return out, present_levels


# ---------------------------------------------------------------------------
# computeAveExpr (R: modeling.R:2170)
# ---------------------------------------------------------------------------
def computeAveExpr(
    data_use: ArrayLike,
    group: Union[Sequence, pd.Categorical],
    features: Optional[Sequence[str]] = None,
    type: Literal["triMean", "truncatedMean", "median"] = "triMean",
    trim: Optional[float] = None,
) -> pd.DataFrame:
    """Compute average expression per cell group.

    Mirrors R's :func:`computeAveExpr` (``modeling.R:2170``). The R reference
    accepts a CellChat object and pulls ``data.use`` from a slot; the Python
    port skips that indirection and accepts a ready-made
    ``data_use`` matrix (genes-by-cells) plus a ``group`` factor.

    Parameters
    ----------
    data_use : numpy.ndarray or pandas.DataFrame
        Genes-by-cells expression matrix. If a ``DataFrame``, its index is
        used as the gene names; otherwise a generic integer index is used.
    group : sequence
        Per-cell group labels (R's ``object@idents``). Coerced to
        :class:`pandas.Categorical` if not already.
    features : sequence of str, optional
        Restrict the output to these features (R: ``features``). Order
        follows ``rownames(data.use)`` (the index of ``data_use``).
    type : {"triMean", "truncatedMean", "median"}, default ``"triMean"``
        Per-group reduction method.
    trim : float, optional
        Truncation fraction passed to the truncated mean. Required when
        ``type="truncatedMean"``.

    Returns
    -------
    pandas.DataFrame
        Genes-by-groups average expression matrix, indexed by the chosen
        features and with columns in factor-level order (present levels).
    """
    if isinstance(data_use, pd.DataFrame):
        gene_names = list(data_use.index.astype(str))
        mat = data_use.to_numpy(dtype=float)
    else:
        mat = np.asarray(data_use, dtype=float)
        gene_names = [str(i) for i in range(mat.shape[0])]

    if features is None:
        features_use = list(gene_names)
    else:
        features_use = [g for g in features if g in set(gene_names)]
    idx = [gene_names.index(g) for g in features_use]
    mat = mat[idx, :]

    if not isinstance(group, pd.Categorical):
        group_cat = pd.Categorical(group)
    else:
        group_cat = group

    FunMean = _make_FunMean(type, trim)
    avg, present_levels = _aggregate_by_group(mat, group_cat, FunMean)

    return pd.DataFrame(
        np.asarray(avg, dtype=float),
        index=pd.Index(features_use),
        columns=pd.Index(present_levels),
    )


# ---------------------------------------------------------------------------
# computeExpr_complex (R: modeling.R:2222)
# ---------------------------------------------------------------------------
def computeExpr_complex(
    complex_input: pd.DataFrame,
    data_use: np.ndarray,
    complex: Sequence[str],
) -> np.ndarray:
    """Compute per-cell complex expression as the geometric mean of subunits.

    Mirrors R's :func:`computeExpr_complex` (``modeling.R:2222``). For each
    requested complex, fetch its subunit gene names from
    ``complex_input[complex, subunit_*]``, drop empty entries (R's
    ``RsubunitsV[RsubunitsV != ""]``), and compute the column-wise
    geometric mean of the corresponding rows of ``data_use``.

    Parameters
    ----------
    complex_input : pandas.DataFrame
        CellChatDB complex frame indexed by complex name with
        ``subunit_*`` columns.
    data_use : numpy.ndarray
        Genes-by-cells expression matrix. Row labels are not stored
        explicitly — the caller must align ``complex_input``'s subunit
        symbols with ``data_use``'s row index. This matches R, where the
        helper does ``data.use[RsubunitsV, , drop = F]`` directly.
    complex : sequence of str
        Names of complexes to compute (must be present in
        ``complex_input.index``).

    Returns
    -------
    numpy.ndarray
        Shape ``(len(complex), n_cells)``. Row ``i`` is the column-wise
        geometric mean of ``data_use`` over the (non-empty) subunits of
        ``complex[i]``. The R helper additionally requires the subunit
        symbols to exist in ``rownames(data.use)`` — if they do not, the
        downstream lookup raises ``"subscript out of bounds"``.
    """
    Rsubunits = complex_input.loc[list(complex)]
    sub_cols = [c for c in Rsubunits.columns if c.startswith("subunit")]
    Rsubunits = Rsubunits[sub_cols]

    n_cells = data_use.shape[1] if data_use.ndim > 1 else 1
    n_complex = Rsubunits.shape[0]
    data_complex = np.zeros((n_complex, n_cells), dtype=float)

    # Need a row-name lookup to mirror R's ``data.use[RsubunitsV, ]``.
    # data_use is anonymous here; the *caller* in R guarantees the subunits
    # are in rownames(data.use). To support that in Python we assume the
    # caller has supplied a ``data_use`` whose row index matches the gene
    # symbols. We expose that index via a wrapper from ``computeCommunProb``;
    # in the standalone unit-test path, callers pass a DataFrame.
    # If data_use is a DataFrame, use its index for the row lookup.
    if isinstance(data_use, pd.DataFrame):
        gene_index = data_use.index
        data_arr = data_use.to_numpy(dtype=float)
    else:
        # Anonymous arrays cannot be subset by name; require a gene_index
        # attribute set by the caller — see ``computeExpr_LR``.
        gene_index = getattr(data_use, "_gene_index", None)
        if gene_index is None:
            raise ValueError(
                "computeExpr_complex requires a row-named data_use; either "
                "pass a DataFrame or attach a `_gene_index` attribute"
            )
        data_arr = np.asarray(data_use, dtype=float)
    pos = pd.Index(gene_index).get_indexer

    for i in range(n_complex):
        # ``unlist(Rsubunits[i, ], use.names = F)`` then drop empties
        row = Rsubunits.iloc[i].tolist()
        # Audit G1: pd.NA-aware filter (was: ``isinstance(g, float) and
        # np.isnan(g)`` which misses ``pd.NA`` from nullable-string dtypes).
        RsubunitsV = [str(g) for g in row if not _is_blank_or_na(g)]
        # R does ``data.use[RsubunitsV, , drop = F]`` then ``geometricMean``
        # which takes column-wise geometric mean of a matrix.
        sub_idx = pos(RsubunitsV)
        if (sub_idx < 0).any():
            missing = [s for s, i_ in zip(RsubunitsV, sub_idx) if i_ < 0]
            raise KeyError(
                f"computeExpr_complex: subunit symbols not in data_use rows: {missing}"
            )
        sub_mat = data_arr[sub_idx, :]
        if sub_mat.shape[0] == 1:
            # Single-subunit branch: ``geometricMean`` on a 1xN matrix is
            # ``exp(log(x)/1) == x``. R's ``geometricMean`` checks
            # ``is.null(nrow(x))``; a 1-row matrix has nrow=1 (not null),
            # so it goes through the matrix branch and returns the row.
            data_complex[i, :] = sub_mat[0, :]
        else:
            data_complex[i, :] = geometricMean(sub_mat, na_rm=True)
    return data_complex


# ---------------------------------------------------------------------------
# computeExpr_LR (R: modeling.R:2288)
# ---------------------------------------------------------------------------
def computeExpr_LR(
    geneLR: Sequence[str],
    data_use: ArrayLike,
    complex_input: pd.DataFrame,
) -> np.ndarray:
    """Compute per-cell ligand-or-receptor expression for an L-R pair table.

    Mirrors R's :func:`computeExpr_LR` (``modeling.R:2288``). For each
    entry of ``geneLR``: if it is present in ``rownames(data.use)``, copy
    that row directly; otherwise treat it as a complex name and dispatch
    to :func:`computeExpr_complex`.

    Parameters
    ----------
    geneLR : sequence of str
        Ligand or receptor names from a ``pairLR`` table; can be either a
        gene symbol or a complex name.
    data_use : numpy.ndarray or pandas.DataFrame
        Genes-by-cells expression matrix. Row labels (gene symbols) are
        required: pass a ``DataFrame``.
    complex_input : pandas.DataFrame
        CellChatDB complex frame.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(geneLR), n_cells)``. Each row is the per-cell
        expression of the corresponding ligand / receptor / complex.
    """
    geneLR_list = list(geneLR)
    nLR = len(geneLR_list)

    if isinstance(data_use, pd.DataFrame):
        gene_index = list(data_use.index.astype(str))
        data_arr = data_use.to_numpy(dtype=float)
    else:
        gene_index = list(getattr(data_use, "_gene_index", []))
        if not gene_index:
            raise ValueError(
                "computeExpr_LR requires a row-named data_use; either pass "
                "a DataFrame or attach a `_gene_index` attribute"
            )
        data_arr = np.asarray(data_use, dtype=float)

    n_cells = data_arr.shape[1] if data_arr.ndim > 1 else 1
    gene_set = set(gene_index)

    # ``index.singleL <- which(geneLR %in% rownames(data.use))`` (1-based)
    is_single = np.array([g in gene_set for g in geneLR_list], dtype=bool)
    dataLavg = np.empty((nLR, n_cells), dtype=float)
    dataLavg.fill(np.nan)

    if np.any(is_single):
        single_genes = [geneLR_list[i] for i in np.flatnonzero(is_single)]
        # ``data.use[geneLR[index.singleL], ]``
        gene_pos = pd.Index(gene_index).get_indexer(single_genes)
        dataLavg[is_single, :] = data_arr[gene_pos, :]

    is_complex = ~is_single
    if np.any(is_complex):
        complex_names = [geneLR_list[i] for i in np.flatnonzero(is_complex)]
        # Dispatch to computeExpr_complex
        # Wrap `data_arr` as a DataFrame so computeExpr_complex can index
        # by gene symbols.
        df = pd.DataFrame(data_arr, index=pd.Index(gene_index))
        sub = computeExpr_complex(complex_input, df, complex_names)
        dataLavg[is_complex, :] = sub

    return dataLavg


# ---------------------------------------------------------------------------
# computeExpr_coreceptor (R: modeling.R:2316)
# ---------------------------------------------------------------------------
def computeExpr_coreceptor(
    cofactor_input: pd.DataFrame,
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    type: Literal["A", "I"] = "A",
) -> np.ndarray:
    """Per-cell co-receptor multiplicative factor for each L-R pair.

    Mirrors R's :func:`computeExpr_coreceptor` (``modeling.R:2316``). Returns
    the per-cell expression product ``prod(1 + data[coreceptor.indV, ])``
    for each L-R pair's co-A or co-I receptor; pairs with no co-receptor
    return a constant ``1`` row.

    Parameters
    ----------
    cofactor_input : pandas.DataFrame
        CellChatDB cofactor frame, indexed by cofactor name with
        ``cofactor*`` columns.
    data_use : numpy.ndarray or pandas.DataFrame
        Genes-by-cells expression matrix (row-named).
    pairLRsig : pandas.DataFrame
        L-R interaction table with ``co_A_receptor`` / ``co_I_receptor``
        string columns (empty / ``NaN`` cells = no co-receptor).
    type : {"A", "I"}, default ``"A"``
        ``"A"`` -> use ``co_A_receptor`` (co-activation); ``"I"`` ->
        ``co_I_receptor`` (co-inhibition).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_LR_pairs, n_cells)``. All-ones rows for pairs without
        a co-receptor.
    """
    if type == "A":
        coreceptor_all = pairLRsig["co_A_receptor"].astype(object).where(
            pairLRsig["co_A_receptor"].notna(), ""
        ).astype(str).tolist()
    elif type == "I":
        coreceptor_all = pairLRsig["co_I_receptor"].astype(object).where(
            pairLRsig["co_I_receptor"].notna(), ""
        ).astype(str).tolist()
    else:
        raise ValueError("'arg' should be one of 'A', 'I'")

    if isinstance(data_use, pd.DataFrame):
        gene_index = list(data_use.index.astype(str))
        data_arr = data_use.to_numpy(dtype=float)
    else:
        gene_index = list(getattr(data_use, "_gene_index", []))
        if not gene_index:
            raise ValueError(
                "computeExpr_coreceptor requires a row-named data_use"
            )
        data_arr = np.asarray(data_use, dtype=float)

    n_cells = data_arr.shape[1] if data_arr.ndim > 1 else 1
    nLR = len(coreceptor_all)

    # ``index.coreceptor <- which(!is.na(coreceptor.all) & coreceptor.all != "")``
    index_cc = [i for i, s in enumerate(coreceptor_all) if s and s != "nan"]

    data_coreceptor = np.ones((nLR, n_cells), dtype=float)

    if not index_cc:
        return data_coreceptor

    # Each coreceptor name indexes a row of cofactor_input
    coreceptor = [coreceptor_all[i] for i in index_cc]
    cofactor_cols = [c for c in cofactor_input.columns if c.startswith("cofactor")]
    coreceptor_ind = cofactor_input.loc[coreceptor, cofactor_cols]

    gene_idx_lookup = pd.Index(gene_index).get_indexer

    rows_cc = np.ones((len(index_cc), n_cells), dtype=float)
    for k in range(len(index_cc)):
        # ``coreceptor.indV`` -> drop empties, intersect with rownames(data.use)
        row = coreceptor_ind.iloc[k].tolist()
        # Audit G1: pd.NA-aware filter via the shared predicate.
        cleaned = [str(g) for g in row if not _is_blank_or_na(g)]
        present_idx = gene_idx_lookup(cleaned)
        present_vals = present_idx[present_idx >= 0]
        if present_vals.size == 1:
            rows_cc[k, :] = 1.0 + data_arr[present_vals[0], :]
        elif present_vals.size > 1:
            sub = 1.0 + data_arr[present_vals, :]
            rows_cc[k, :] = np.prod(sub, axis=0)
        else:
            rows_cc[k, :] = 1.0

    for j, i in enumerate(index_cc):
        data_coreceptor[i, :] = rows_cc[j, :]
    return data_coreceptor


# ---------------------------------------------------------------------------
# computeExpr_agonist (R: modeling.R:2491)
# ---------------------------------------------------------------------------
def computeExpr_agonist(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_agonist: int,
    Kh: float,
    n: float,
) -> np.ndarray:
    """Hill-aggregated per-cell agonist factor for **one** L-R pair.

    Mirrors R's :func:`computeExpr_agonist` (``modeling.R:2491``). Looks up
    ``pairLRsig$agonist[index.agonist]`` -> ``cofactor_input``'s
    ``cofactor*`` row -> drops empties / intersects with
    ``rownames(data.use)``. Three cases:

    - 1 cofactor: ``1 + x^n / (Kh^n + x^n)``.
    - >1 cofactor: ``apply(1 + x^n / (Kh^n + x^n), 2, prod)``.
    - 0 cofactor: a 1xN ones row.

    Parameters
    ----------
    data_use : numpy.ndarray or pandas.DataFrame
        Genes-by-cells matrix (row-named).
    pairLRsig : pandas.DataFrame
        L-R interaction table; the ``agonist`` column references
        ``cofactor_input``'s row index.
    cofactor_input : pandas.DataFrame
        CellChatDB cofactor table.
    index_agonist : int
        0-based row index into ``pairLRsig`` (the Python equivalent of
        R's 1-based ``index.agonist``).
    Kh : float
        Hill EC50 parameter (R default ``0.5``; passed explicitly here).
    n : float
        Hill coefficient (R default ``1``; passed explicitly here).

    Returns
    -------
    numpy.ndarray
        Shape ``(1, n_cells)``. The per-cell agonist multiplier.
    """
    return _hill_cofactor_one(
        data_use, pairLRsig, cofactor_input, index_agonist, Kh, n,
        column="agonist", mode="agonist",
    )


# ---------------------------------------------------------------------------
# computeExpr_antagonist (R: modeling.R:2527)
# ---------------------------------------------------------------------------
def computeExpr_antagonist(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_antagonist: int,
    Kh: float,
    n: float,
) -> np.ndarray:
    """Hill-aggregated per-cell antagonist factor for **one** L-R pair.

    Mirrors R's :func:`computeExpr_antagonist` (``modeling.R:2527``). Same
    plumbing as :func:`computeExpr_agonist` but uses
    ``Kh^n / (Kh^n + x^n)`` (suppressive Hill).

    Parameters
    ----------
    data_use, pairLRsig, cofactor_input, Kh, n
        See :func:`computeExpr_agonist`.
    index_antagonist : int
        0-based row index into ``pairLRsig`` (mirrors R's
        ``index.antagonist``).

    Returns
    -------
    numpy.ndarray
        Shape ``(1, n_cells)``.
    """
    return _hill_cofactor_one(
        data_use, pairLRsig, cofactor_input, index_antagonist, Kh, n,
        column="antagonist", mode="antagonist",
    )


def _hill_cofactor_one(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    index_lr: int,
    Kh: float,
    n: float,
    column: str,
    mode: Literal["agonist", "antagonist"],
) -> np.ndarray:
    """Shared implementation of :func:`computeExpr_agonist` / :func:`computeExpr_antagonist`.

    Direct port of the body shared between modeling.R lines 2491-2548.
    """
    if isinstance(data_use, pd.DataFrame):
        gene_index = list(data_use.index.astype(str))
        data_arr = data_use.to_numpy(dtype=float)
    else:
        gene_index = list(getattr(data_use, "_gene_index", []))
        if not gene_index:
            raise ValueError(
                "_hill_cofactor_one requires a row-named data_use"
            )
        data_arr = np.asarray(data_use, dtype=float)

    n_cells = data_arr.shape[1] if data_arr.ndim > 1 else 1

    # ``agonist <- pairLRsig$agonist[index.agonist]``
    # Audit G1: pd.NA-aware predicate replaces the prior 3-clause check.
    name_val = pairLRsig[column].iloc[index_lr]
    if _is_blank_or_na(name_val):
        # ``cofactor_input[NA, ...]`` returns a row of NAs -> all empty -> empty intersection
        # So fall through to the empty-cleaned branch below; agonist=ones, antagonist=ones.
        # R's ``cofactor_input[NA, ]`` returns a row whose values are all NA. Then
        # ``unlist`` -> ``[!= ""]`` -> ``intersect(., rownames(data.use))`` is empty.
        cleaned: list[str] = []
    else:
        # ``cofactor_input[agonist, grepl("cofactor", colnames(cofactor_input))]``
        cofactor_cols = [c for c in cofactor_input.columns if c.startswith("cofactor")]
        if str(name_val) not in cofactor_input.index:
            cleaned = []
        else:
            row = cofactor_input.loc[str(name_val), cofactor_cols]
            row_vals = row.tolist() if hasattr(row, "tolist") else [row]
            cleaned = [str(g) for g in row_vals if not _is_blank_or_na(g)]

    # ``intersect(agonist.indV, rownames(data.use))``
    pos = pd.Index(gene_index).get_indexer(cleaned)
    present = pos[pos >= 0]

    Kh_n = float(Kh) ** float(n)

    if present.size == 0:
        return np.ones((1, n_cells), dtype=float)

    data_avg = data_arr[present, :]  # shape (k, n_cells)
    if mode == "agonist":
        # ``1 + x^n / (Kh^n + x^n)``
        x_n = np.power(data_avg, float(n))
        per_gene = 1.0 + x_n / (Kh_n + x_n)
    else:
        # ``Kh^n / (Kh^n + x^n)``
        x_n = np.power(data_avg, float(n))
        per_gene = Kh_n / (Kh_n + x_n)

    if data_avg.shape[0] == 1:
        # single-cofactor branch: shape (1, n_cells)
        return per_gene
    # multi-cofactor branch: ``apply(., 2, prod)`` -> shape (n_cells,)
    return np.prod(per_gene, axis=0).reshape(1, -1)


# ---------------------------------------------------------------------------
# computeExprGroup_agonist (R: modeling.R:2425)
# ---------------------------------------------------------------------------
def computeExprGroup_agonist(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    group: Union[Sequence, pd.Categorical],
    index_agonist: int,
    Kh: float,
    FunMean: Callable[[np.ndarray], float],
    n: float,
) -> np.ndarray:
    """Hill-aggregated per-cell-group agonist factor for **one** L-R pair.

    Mirrors R's :func:`computeExprGroup_agonist` (``modeling.R:2425``).
    Equivalent to :func:`computeExpr_agonist` but applied to the per-group
    averaged expression matrix (``aggregate(t(data.use[..., ]), list(group),
    FUN = FunMean)``).

    Parameters
    ----------
    data_use : numpy.ndarray or pandas.DataFrame
        Genes-by-cells matrix (row-named).
    pairLRsig, cofactor_input, index_agonist, Kh, n
        See :func:`computeExpr_agonist`.
    group : sequence
        Per-cell group labels (R's ``group`` factor).
    FunMean : callable
        Per-group reduction function (e.g. result of :func:`_make_FunMean`).

    Returns
    -------
    numpy.ndarray
        Shape ``(1, n_groups_present)``.
    """
    return _hill_cofactor_group_one(
        data_use, pairLRsig, cofactor_input, group, index_agonist, Kh, FunMean, n,
        column="agonist", mode="agonist",
    )


# ---------------------------------------------------------------------------
# computeExprGroup_antagonist (R: modeling.R:2458)
# ---------------------------------------------------------------------------
def computeExprGroup_antagonist(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    group: Union[Sequence, pd.Categorical],
    index_antagonist: int,
    Kh: float,
    FunMean: Callable[[np.ndarray], float],
    n: float,
) -> np.ndarray:
    """Hill-aggregated per-cell-group antagonist factor for **one** L-R pair.

    Mirrors R's :func:`computeExprGroup_antagonist` (``modeling.R:2458``).
    Same plumbing as :func:`computeExprGroup_agonist` with the antagonist
    Hill kernel ``Kh^n / (Kh^n + x^n)``.

    See :func:`computeExprGroup_agonist` for the parameter list.

    Returns
    -------
    numpy.ndarray
        Shape ``(1, n_groups_present)``.
    """
    return _hill_cofactor_group_one(
        data_use, pairLRsig, cofactor_input, group, index_antagonist, Kh, FunMean, n,
        column="antagonist", mode="antagonist",
    )


def _hill_cofactor_group_one(
    data_use: ArrayLike,
    pairLRsig: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    group: Union[Sequence, pd.Categorical],
    index_lr: int,
    Kh: float,
    FunMean: Callable[[np.ndarray], float],
    n: float,
    column: str,
    mode: Literal["agonist", "antagonist"],
) -> np.ndarray:
    """Shared implementation of group-level agonist / antagonist modulation.

    Direct port of modeling.R lines 2425-2476.
    """
    if isinstance(data_use, pd.DataFrame):
        gene_index = list(data_use.index.astype(str))
        data_arr = data_use.to_numpy(dtype=float)
    else:
        gene_index = list(getattr(data_use, "_gene_index", []))
        if not gene_index:
            raise ValueError(
                "_hill_cofactor_group_one requires a row-named data_use"
            )
        data_arr = np.asarray(data_use, dtype=float)

    if not isinstance(group, pd.Categorical):
        group_cat = pd.Categorical(group)
    else:
        group_cat = group
    n_groups_present = len(set(int(c) for c in group_cat.codes if c >= 0))

    # Audit G1: pd.NA-aware predicate replaces the prior 3-clause check.
    name_val = pairLRsig[column].iloc[index_lr]
    if _is_blank_or_na(name_val):
        cleaned: list[str] = []
    else:
        cofactor_cols = [c for c in cofactor_input.columns if c.startswith("cofactor")]
        if str(name_val) not in cofactor_input.index:
            cleaned = []
        else:
            row = cofactor_input.loc[str(name_val), cofactor_cols]
            row_vals = row.tolist() if hasattr(row, "tolist") else [row]
            cleaned = [str(g) for g in row_vals if not _is_blank_or_na(g)]

    pos = pd.Index(gene_index).get_indexer(cleaned)
    present = pos[pos >= 0]

    Kh_n = float(Kh) ** float(n)

    if present.size == 0:
        return np.ones((1, n_groups_present), dtype=float)

    sub = data_arr[present, :]
    # aggregate per group
    avg, _ = _aggregate_by_group(sub, group_cat, FunMean)
    # avg shape: (k, n_groups_present)
    if mode == "agonist":
        x_n = np.power(avg, float(n))
        per_gene = 1.0 + x_n / (Kh_n + x_n)
    else:
        x_n = np.power(avg, float(n))
        per_gene = Kh_n / (Kh_n + x_n)

    if avg.shape[0] == 1:
        return per_gene
    return np.prod(per_gene, axis=0).reshape(1, -1)
