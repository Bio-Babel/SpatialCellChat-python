"""NMF-based topic / pattern analysis exports for SpatialCellChat.

Direct ports of the nine ``analysis.R`` exports under sub-agent B's
slice-5 scope:

- :func:`extractEnrichedLR`              (R ``analysis.R:826``)
- :func:`selectK`                        (R ``analysis.R:941``)
- :func:`suggestK`                       (R ``analysis.R:1014``)
- :func:`identifyCommunicationPatterns`  (R ``analysis.R:1219``)
- :func:`computeEnrichmentScore`         (R ``analysis.R:3337``)
- :func:`identifyCellTopics`             (R ``analysis.R:3393``)
- :func:`identifyTopicCommunities`       (R ``analysis.R:3521``)
- :func:`suggestK1`                      (R ``analysis.R:3738``)
- :func:`extractTopicSignaling`          (R ``analysis.R:3869``)

The R reference is the ground truth; deviations from R behaviour follow
the design contract under §3.B (Tier-2 NMF substitutions).

Functions take :class:`anndata.AnnData` first and read/write
``adata.uns["cellchat"]`` slots that mirror R's
``cellchat@netP$pattern`` / ``cellchat@net$topic`` layout. Python-only
slot names use snake_case where R uses dollar-name access; the slot map
is documented in :mod:`spatialcellchat.analysis_topics` test fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd

from ._anndata import cellchat_dict as _ensure_uns
from ._nmf import _consensus_nmf, _l1_nmf, _lee_nmf, _redistribute_d
from ._utils import scaleMat
from .database import extractGeneSubset, load_db, searchPair

__all__ = [
    "extractEnrichedLR",
    "selectK",
    "suggestK",
    "suggestK1",
    "identifyCommunicationPatterns",
    "computeEnrichmentScore",
    "identifyCellTopics",
    "identifyTopicCommunities",
    "extractTopicSignaling",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_slot(adata: ad.AnnData, slot_name: str) -> Dict[str, Any]:
    """Fetch the named ``net`` / ``netP`` slot, raising if missing."""
    cc = _ensure_uns(adata)
    if slot_name not in cc:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}'] is missing; run the "
            "appropriate upstream function first."
        )
    return cc[slot_name]


def _prepare_pattern_data(
    prob: np.ndarray, pattern: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a ``(K, K, nLR)`` prob array into the pattern-input matrix.

    Mirrors R's
    ``apply(prob, c(1,3), sum)`` → ``sweep(., 2L, max(., na.rm=TRUE), '/')``
    and the subsequent ``data <- data[rowSums(data) != 0, ]`` filter.

    R parity (``CellChat::analysis.R:305-313``): R names the unfiltered
    column-max-normalised matrix ``data0`` and the row-zero-filtered
    matrix ``data``; the slot stores ``data0`` under the ``"data"`` key
    (line 445). We return both so the caller can persist ``data0`` while
    still using the filtered matrix as NMF input.

    Parameters
    ----------
    prob : numpy.ndarray
        ``(K, K, nLR)`` cell-group / pathway probability tensor.
    pattern : {"outgoing", "incoming"}
        Whether to sum over receivers (``"outgoing"``) or senders
        (``"incoming"``).

    Returns
    -------
    data : numpy.ndarray
        Filtered ``(K_keep, nLR)`` matrix used as NMF input (R: ``data``).
    data0 : numpy.ndarray
        Unfiltered ``(K, nLR)`` matrix (R: ``data0``) — the value R
        persists into ``slot$pattern[[pattern]]$data``.
    keep_idx : numpy.ndarray
        Indices into the original cell-group axis kept after filtering.
    """
    if prob.ndim != 3:
        raise ValueError("`prob` must be a 3-D array (K, K, nLR).")
    if pattern not in ("outgoing", "incoming"):
        raise ValueError("`pattern` must be 'outgoing' or 'incoming'.")
    if pattern == "outgoing":
        # apply(prob, c(1,3), sum) — sum over axis=1 (receivers)
        data0 = np.nansum(prob, axis=1)
    else:
        # apply(prob, c(2,3), sum) — sum over axis=0 (senders)
        data0 = np.nansum(prob, axis=0)
    # column-max normalisation (sweep(., 2L, max, '/'))
    col_max = np.nanmax(data0, axis=0, keepdims=True)
    safe_max = np.where(col_max == 0.0, 1.0, col_max)
    data0 = data0 / safe_max
    # drop rows that are entirely zero (NMF input)
    row_sums = np.nansum(data0, axis=1)
    keep_idx = np.flatnonzero(row_sums != 0.0)
    return data0[keep_idx, :], data0, keep_idx


def _resolve_db(species: str, db: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Resolve the CellChat database for a (species, override) pair."""
    if db is not None:
        return db
    if species not in ("mouse", "human"):
        raise ValueError(
            "Only 'mouse' and 'human' are supported by default. "
            "Pass `db=` explicitly otherwise."
        )
    return load_db(species)


# ---------------------------------------------------------------------------
# extractEnrichedLR  (R: analysis.R:826)
# ---------------------------------------------------------------------------
def _extract_enriched_lr_internal(
    net: Dict[str, Any],
    LR: Dict[str, pd.DataFrame],
    DB: Dict[str, pd.DataFrame],
    signaling: str,
    enriched_only: bool = True,
    thresh: float = 0.05,
    do_group: bool = True,
) -> Tuple[List[str], List[str]]:
    """R's ``extractEnrichedLR_internal`` (analysis.R:883).

    Returns ``(geneLR, pairLR.name.use)`` for one signaling pathway.
    """
    pairLR = searchPair(
        signaling=signaling,
        pairLR_use=LR["LRsig"],
        key="pathway_name",
        matching_exact=True,
        pair_only=True,
    )
    if enriched_only:
        if do_group:
            if "prob" not in net:
                raise ValueError("net['prob'] is missing; run computeAvgCommunProb first.")
            if "LR.sig" not in net:
                raise ValueError("Please run the `aggregateNet` function!")
            prob = np.asarray(net["prob"], dtype=float).copy()
            pval_obj = net.get("pval")
            if pval_obj is not None:
                pval = np.asarray(pval_obj, dtype=float)
                prob[pval > thresh] = 0.0
            # dim-3 names — R uses dimnames(net$prob)[[3]]
            pair_use_names = list(net.get("LR.sig.dimnames3", net["LR.sig"]))
            # pair_use_names is the universe of pair names that align with prob's 3rd axis;
            # if "LR.sig.dimnames3" was not stored, fall back to LR.sig (which is always
            # a subset of pair names).
            pair_names_keep = [n for n in pairLR.index if n in set(pair_use_names)]
            LR_sig = list(net["LR.sig"])
        else:
            if "prob.cell" not in net:
                raise ValueError(
                    "net['prob.cell'] is missing; run computeCommunProb first."
                )
            if "LR.sig.cell" not in net:
                raise ValueError("Please run the `aggregateNet` function!")
            pair_use_names = list(net.get(
                "LR.sig.cell.dimnames3", net["LR.sig.cell"]
            ))
            pair_names_keep = [n for n in pairLR.index if n in set(pair_use_names)]
            LR_sig = list(net["LR.sig.cell"])
        # Intersect with significant
        sig_set = set(LR_sig)
        pair_use = [n for n in pair_names_keep if n in sig_set]
        if len(pair_use) > 0:
            pairLR = pairLR.loc[pair_use, :]
        else:
            # mirror R's empty-result branch: pairLR is left untouched
            # (just an informational message), but pairLR.name.use is empty
            pairLR = pairLR.iloc[0:0]
        pair_names_keep = pair_use
    else:
        pair_names_keep = pairLR.index.tolist()

    # geneL = unique(pairLR$ligand); geneR similarly
    geneL = list(dict.fromkeys(pairLR["ligand"].astype(str).tolist())) if len(pairLR) else []
    geneR = list(dict.fromkeys(pairLR["receptor"].astype(str).tolist())) if len(pairLR) else []
    geneL_expanded = extractGeneSubset(geneL, DB["complex"], DB["geneInfo"])
    geneR_expanded = extractGeneSubset(geneR, DB["complex"], DB["geneInfo"])
    geneLR = list(geneL_expanded) + list(geneR_expanded)
    return geneLR, pair_names_keep


def extractEnrichedLR(
    adata: Union[ad.AnnData, Sequence[ad.AnnData]],
    signaling: Sequence[str],
    geneLR_return: bool = False,
    enriched_only: bool = True,
    thresh: float = 0.05,
    do_group: bool = True,
    geneInfo: Optional[pd.DataFrame] = None,
    complex_input: Optional[pd.DataFrame] = None,
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """Identify all enriched L-R pairs for a list of signaling pathways.

    Direct port of R's :func:`extractEnrichedLR` (analysis.R:826). The
    function returns a single-column data frame ``interaction_name`` of
    enriched pair names; with ``geneLR_return=True`` it also returns the
    expanded ligand/receptor gene set.

    Parameters
    ----------
    adata : anndata.AnnData or sequence of anndata.AnnData
        Single-mode: pass one AnnData with ``uns["cellchat"]["net"]``,
        ``uns["cellchat"]["LR"]`` and ``uns["cellchat"]["DB"]``
        populated. Merged-mode (R: ``object@options$mode == "merged"``):
        pass a sequence of per-dataset AnnDatas; this branch unions
        ``geneLR``/``pairLR`` across datasets per signaling pathway,
        mirroring R ``analysis.R:842-862``.
    signaling : sequence of str
        Signaling pathway names to query.
    geneLR_return : bool, default ``False``
        If ``True`` return a dict ``{"pairLR": DataFrame, "geneLR": list}``.
    enriched_only : bool, default ``True``
        If ``True`` filter to L-R pairs marked significant by
        :func:`aggregateNet`. If ``False`` return every pair in the
        signaling pathway from the database.
    thresh : float, default ``0.05``
        p-value threshold (used only when ``enriched_only=True`` and
        ``do_group=True``).
    do_group : bool, default ``True``
        ``True`` reads ``net["LR.sig"]`` (group-level); ``False`` reads
        ``net["LR.sig.cell"]`` (cell-level).
    geneInfo : pandas.DataFrame, optional
        Override ``DB["geneInfo"]``.
    complex_input : pandas.DataFrame, optional
        Override ``DB["complex"]``.

    Returns
    -------
    pandas.DataFrame or dict
        If ``geneLR_return=False``, a one-column ``interaction_name``
        data frame. Otherwise a dict with keys ``pairLR`` (data frame)
        and ``geneLR`` (list of gene symbols).
    """
    # Detect merged mode: a sequence of AnnDatas (R's net0[[i]] /
    # object@LR[[i]] iteration). Single mode: one AnnData.
    if isinstance(adata, ad.AnnData):
        adata_list: List[ad.AnnData] = [adata]
        merged = False
    else:
        adata_list = list(adata)
        if len(adata_list) == 0:
            raise ValueError("`adata` must be a non-empty sequence of AnnData.")
        for a in adata_list:
            if not isinstance(a, ad.AnnData):
                raise TypeError(
                    "all elements of `adata` must be AnnData objects."
                )
        merged = len(adata_list) > 1

    # Resolve DB/LR/net per dataset (R: DB <- object@DB; net0 <-
    # slot(object,'net'); LR <- object@LR — single is a single dict;
    # merged is a list-of-dicts indexed by dataset).
    per_dataset: List[Dict[str, Any]] = []
    for a in adata_list:
        cc = _ensure_uns(a)
        if "DB" not in cc:
            raise ValueError("adata.uns['cellchat']['DB'] is missing.")
        if "LR" not in cc:
            raise ValueError("adata.uns['cellchat']['LR'] is missing.")
        if "net" not in cc:
            raise ValueError("adata.uns['cellchat']['net'] is missing.")
        DB_i = dict(cc["DB"])
        if geneInfo is not None:
            DB_i["geneInfo"] = geneInfo
        if complex_input is not None:
            DB_i["complex"] = complex_input
        per_dataset.append({"DB": DB_i, "LR": cc["LR"], "net": cc["net"]})

    if isinstance(signaling, str):
        signaling_list: List[str] = [signaling]
    else:
        signaling_list = list(signaling)

    pairLR_all: List[str] = []
    geneLR_all: List[str] = []
    for sig in signaling_list:
        if not merged:
            d = per_dataset[0]
            geneLR_i, pair_use_i = _extract_enriched_lr_internal(
                net=d["net"], LR=d["LR"], DB=d["DB"], signaling=sig,
                enriched_only=enriched_only, thresh=thresh, do_group=do_group,
            )
        else:
            # Merged mode: union geneLR / pairLR across datasets per sig
            # (R uses BiocGenerics::union which preserves first-occurrence order).
            geneLR_t: List[str] = []
            pair_use_t: List[str] = []
            for d in per_dataset:
                geneLR_d, pair_use_d = _extract_enriched_lr_internal(
                    net=d["net"], LR=d["LR"], DB=d["DB"], signaling=sig,
                    enriched_only=enriched_only, thresh=thresh, do_group=do_group,
                )
                for g in geneLR_d:
                    if g not in geneLR_t:
                        geneLR_t.append(str(g))
                for p in pair_use_d:
                    if p not in pair_use_t:
                        pair_use_t.append(str(p))
            geneLR_i, pair_use_i = geneLR_t, pair_use_t
        geneLR_all.extend(geneLR_i)
        pairLR_all.extend(pair_use_i)

    pairLR_df = pd.DataFrame({"interaction_name": pairLR_all})
    if geneLR_return:
        return {"pairLR": pairLR_df, "geneLR": geneLR_all}
    return pairLR_df


# ---------------------------------------------------------------------------
# selectK  (R: analysis.R:941)
# ---------------------------------------------------------------------------
def selectK(
    adata: ad.AnnData,
    slot_name: str = "netP",
    pattern: Literal["outgoing", "incoming"] = "outgoing",
    title_name: Optional[str] = None,
    do_facet: bool = True,
    k_range: Sequence[int] = tuple(range(2, 11)),
    nrun: int = 30,
    seed_use: int = 10,
) -> Dict[str, Any]:
    """Select the number of NMF patterns using cophenetic + silhouette.

    Direct port of R's :func:`selectK` (analysis.R:941). The R version
    returns a ggplot object; the Python port returns a dict containing
    the diagnostic statistics — a separate viz helper (slice 6a) wraps
    the plot.

    Per design contract §3.B, ``NMF::nmfEstimateRank`` is approximated by
    :func:`spatialcellchat._nmf._consensus_nmf` (Tier-2 deviation).

    R-bug fix (intentional deviation per the project's R-bug-handling
    decision rule — "honour R's documented intent, not R's typos"):
    SpatialCellChat ``analysis.R:962`` writes::

        res <- NMF::nmfEstimateRank(data_sr, range = 20:30,
                                    nrun = 30L, seed = seed.use)

    Three typos relative to the function's documented intent:
    (1) ``data_sr`` is undefined in scope (only ``data`` and ``data0``
    exist) — running the function raises ``Error: object 'data_sr' not
    found`` and aborts;
    (2) ``range = 20:30`` ignores the user's ``k.range`` argument;
    (3) ``nrun = 30L`` ignores the user's ``nrun`` argument.

    The upstream ``CellChat::analysis.R:320`` is correct — it reads
    ``NMF::nmfEstimateRank(data, range = k.range, method = 'lee',
    nrun = nrun, seed = seed.use)`` — so this is a fork-side regression.
    The Python port honours all three intended arguments, pinned by
    ``test_selectK_honours_k_range``.

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"][slot_name]["prob"]`` populated.
    slot_name : str, default ``"netP"``
        Which slot stores the input ``prob`` tensor.
    pattern : {"outgoing", "incoming"}, default ``"outgoing"``
        Pattern direction.
    title_name : str, optional
        Title to be attached to the returned dict (used by the plot
        helper). Defaults to ``f"{pattern} signaling \\n"``.
    do_facet : bool, default ``True``
        Carried through to the plot helper.
    k_range : sequence of int, default ``range(2, 11)``
        Candidate NMF ranks.
    nrun : int, default ``30``
        Number of NMF restarts per ``k``.
    seed_use : int, default ``10``
        Base seed.

    Returns
    -------
    dict
        ``{"data": numpy.ndarray, "k": list[int],
        "cophenetic": numpy.ndarray, "silhouette": numpy.ndarray,
        "k_hat": int, "title": str, "do_facet": bool, "pattern": str}``.
        ``data`` is the ``(rows_kept, nLR)`` matrix that was actually fed
        to NMF (after the row-zero filter).
    """
    slot = _get_slot(adata, slot_name)
    if "prob" not in slot:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['prob'] is missing."
        )
    prob = np.asarray(slot["prob"], dtype=float)
    data, _data0, _ = _prepare_pattern_data(prob, pattern)

    if title_name is None:
        title_name = f"{pattern} signaling \n"

    k_arr = list(k_range)
    k_hat, coph, silh = _consensus_nmf(
        data=data,
        k_range=k_arr,
        n_runs=nrun,
        seed=seed_use,
        method="lee",
    )
    return {
        "data": data,
        "k": k_arr,
        "cophenetic": coph,
        "silhouette": silh,
        "k_hat": k_hat,
        "title": title_name,
        "do_facet": do_facet,
        "pattern": pattern,
    }


# ---------------------------------------------------------------------------
# suggestK / suggestK1  (R: analysis.R:1014, 3738)
# ---------------------------------------------------------------------------
def _suggestK_stats(
    Mat: np.ndarray,
    k_test: Sequence[int],
    reps: int = 10,
    p: float = 0.3,
    tol: float = 1e-6,
    maxit: int = 100,
    L1: Sequence[float] = (0.0, 0.0),
    seed_use: int = 666,
) -> pd.DataFrame:
    """Compute the cross-validation statistics underlying ``suggestK1``.

    Mirrors R's ``suggestK1`` evaluation loop (analysis.R:3738):

    1. Set seed; sample ``reps`` random integers in ``[1, 10000]`` to
       seed each NMF run.
    2. If ``p > 0``, reserve a sub-block of rows / columns of ``Mat`` to
       evaluate Frobenius reconstruction error; otherwise score on the
       full matrix.
    3. For each ``(k, run_seed)`` pair, fit NMF with the chosen seed and
       record ``objErr = ||Mat_sub - W_sub @ H_sub.T||_F /
       (rowSampleNum * colSampleNum)`` (or full size when ``p == 0``).

    Returns
    -------
    pandas.DataFrame
        Long-format ``{k, seed_use, objErr, iter}`` for each run.
    """
    Mat = np.asarray(Mat, dtype=float)
    if Mat.ndim != 2:
        raise ValueError("`Mat` must be a 2D matrix.")
    if not (0.0 <= p < 1.0):
        raise ValueError("`p` must be in [0, 1).")
    nrow, ncol = Mat.shape

    rng = np.random.default_rng(int(seed_use))
    # R: sample(1:10000, size=reps) — without replacement.
    seedSample = rng.choice(np.arange(1, 10001), size=int(reps), replace=False)
    seedSample = seedSample.astype(int)

    if p == 0.0:
        rowSampleNum = nrow
        colSampleNum = ncol
        rowSample = np.arange(nrow)
        colSample = np.arange(ncol)
    else:
        rowSampleNum = int(np.ceil(p * nrow))
        colSampleNum = int(np.ceil(p * ncol))
        rowSample = rng.choice(nrow, size=rowSampleNum, replace=False)
        colSample = rng.choice(ncol, size=colSampleNum, replace=False)

    # R-bug fix (intentional deviation from SpatialCellChat's analysis.R:
    # 3772-3779 / 3815-3822). The R function signature documents
    # ``@param L1 L1/LASSO penalties between 0 and 1, array of length two
    # for c(w, h)`` (analysis.R:3733), but the function body calls
    # ``RcppML::nmf(Mat, k=k, tol=tol, maxit=maxit, seed=seedSample[[i]],
    # verbose=F)`` *without* passing ``L1`` — the parameter is silently
    # ignored despite the signature + docstring promise. This is a typo
    # in the R fork (no ``suggestK1`` exists in upstream jinworks/CellChat
    # to compare against, so we cannot blame an upstream change). Per the
    # project's R-bug-handling decision rule, when R's *intent* (signature
    # + docstring) and R's *implementation* disagree, Python honours the
    # intent. So we pass L1 through to the underlying NMF.
    L1_forward = tuple(float(x) for x in L1)

    rows: List[Dict[str, Any]] = []
    for k in k_test:
        for seed_i in seedSample:
            W, H, n_iter = _l1_nmf(
                data=Mat, rank=int(k), L1=L1_forward,
                seed=int(seed_i), max_iter=int(maxit), tol=float(tol),
            )
            # R: W <- w %*% diag(sqrt(d)); H <- diag(sqrt(d)) %*% h |> t().
            # sklearn has no `d`; redistribute the per-factor scale so the
            # reconstruction matches R's symmetric W / H norms.
            W, H = _redistribute_d(W, H)
            # Recovery: Mat_sub_hat = W_sub @ H_sub.T (sklearn returns H of
            # shape (k, ncol); transpose semantics align with R's "W %*% t(H)").
            W_sub = W[rowSample, :]
            H_sub_T = H[:, colSample]  # shape (k, colSampleNum)
            Mat_sub_hat = W_sub @ H_sub_T
            Mat_sub = Mat[np.ix_(rowSample, colSample)]
            sub_F = float(np.linalg.norm(Mat_sub - Mat_sub_hat, ord="fro"))
            denom = float(rowSampleNum) * float(colSampleNum)
            rows.append({
                "k": int(k),
                "seed_use": int(seed_i),
                "objErr": sub_F / denom,
                "iter": int(n_iter),
            })
    return pd.DataFrame(rows)


def suggestK1(
    Mat: np.ndarray,
    k_test: Sequence[int] = tuple(range(5, 51, 5)),
    reps: int = 10,
    p: float = 0.3,
    tol: float = 1e-6,
    maxit: int = 100,
    L1: Sequence[float] = (0.0, 0.0),
    seed_use: int = 666,
) -> Dict[str, Any]:
    """Cross-validate ``RcppML::nmf`` rank choice (R: analysis.R:3738).

    Per design contract §3.B, R's ``RcppML::nmf`` is approximated by
    sklearn's ``solver='cd'`` with L1 regularisation (Tier-2). The
    cross-validation block-sample protocol mirrors R exactly.

    Parameters
    ----------
    Mat : numpy.ndarray
        ``(features x samples)`` non-negative matrix.
    k_test : sequence of int, default ``(5, 10, 15, ..., 50)``
        Candidate ranks. Must contain at least two values.
    reps : int, default ``10``
        Random restarts per ``k``.
    p : float, default ``0.3``
        Proportion of rows / columns sampled for the held-out block. Use
        ``p == 0`` to score on the full matrix.
    tol, maxit : float, int
        Stopping criteria forwarded to NMF.
    L1 : sequence of two floats, default ``(0, 0)``
        L1 strengths for ``W`` and ``H``.
    seed_use : int, default ``666``
        RNG seed.

    Returns
    -------
    dict
        ``{"stats": DataFrame, "figure": None}`` — mirrors R's
        ``list(stats=..., figure=...)``. The Python port leaves
        ``figure`` as ``None`` (the matplotlib renderer lives in
        slice 6a's ``.plotSuggestK``); call that helper on
        ``result['stats']`` to materialise the diagnostic figure.
    """
    if not isinstance(Mat, (np.ndarray, pd.DataFrame)):
        raise ValueError("Please check your input. It must be a matrix.")
    if len(list(k_test)) <= 1:
        raise ValueError("Please provide more than one `k` for cross validate.")
    if p < 0 or p >= 1:
        raise ValueError("Please check your `p`. It must be in [0,1).")
    df = _suggestK_stats(
        Mat=Mat, k_test=k_test, reps=reps, p=p,
        tol=tol, maxit=maxit, L1=L1, seed_use=seed_use,
    )
    return {"stats": df, "figure": None}


def suggestK(
    adata: ad.AnnData,
    kTest: Sequence[int] = tuple(range(5, 51, 5)),
    nRandomStart: int = 10,
    lambda_: float = 5.0,
    nIteration: int = 30,
    nCores: int = 1,
    verbose: bool = True,
    layer: str = "X",
    L1: Sequence[float] = (0.0, 0.0),
    seed_use: int = 666,
) -> Dict[str, Any]:
    """Sweep candidate ranks for an iNMF-style factorisation (R: analysis.R:1014).

    R's :func:`suggestK` calls ``rliger::runINMF`` for integrative NMF on
    a Liger object. There is no pure-Python iNMF in the SpatialCellChat
    pipeline (it is only used in tutorial code), so this Python port runs
    a plain L1-regularised NMF on ``adata.X`` (or the requested
    ``layer``) and returns the same ``{stats, best_k}`` shape as
    :func:`suggestK1`.

    Parameters
    ----------
    adata : anndata.AnnData
        Object whose matrix is factorised.
    kTest : sequence of int, default ``(5, 10, ..., 50)``
        Candidate ranks.
    nRandomStart : int, default ``10``
        Restarts per ``k``.
    lambda_ : float, default ``5.0``
        Forwarded as L1 strength when both entries of ``L1`` are zero.
    nIteration : int, default ``30``
        Maximum iterations per NMF run.
    nCores : int, default ``1``
        Reserved for future parallelism; currently unused (each NMF call
        is sequential).
    verbose : bool, default ``True``
        Whether to print progress (currently unused — kept for API
        parity).
    layer : str, default ``"X"``
        Which layer of ``adata`` to use.
    L1 : sequence of two floats, default ``(0, 0)``
    seed_use : int, default ``666``

    Returns
    -------
    dict
        ``{"stats": DataFrame, "figure": None}`` — matches R's
        ``list(stats=..., figure=...)``. Pass ``stats`` to slice 6a's
        plotting helper to materialise the figure.
    """
    if layer == "X":
        X = adata.X
    else:
        X = adata.layers[layer]
    from scipy.sparse import issparse as _issparse
    if _issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    # iNMF takes (cells x features); we transpose to (features x cells)
    # so the L1 wrapper sees the conventional NMF orientation.
    Mat = X.T
    L1_arr = np.asarray(L1, dtype=float)
    if float(L1_arr[0]) == 0.0 and float(L1_arr[1]) == 0.0:
        # Use the lambda regularisation argument as a soft-equivalent.
        L1_arr = np.array([float(lambda_) / 100.0, float(lambda_) / 100.0])
    return suggestK1(
        Mat=Mat,
        k_test=kTest,
        reps=int(nRandomStart),
        p=0.3,
        tol=1e-6,
        maxit=int(nIteration),
        L1=L1_arr,
        seed_use=int(seed_use),
    )


# ---------------------------------------------------------------------------
# identifyCommunicationPatterns (R: analysis.R:1219)
# ---------------------------------------------------------------------------
def identifyCommunicationPatterns(
    adata: ad.AnnData,
    slot_name: str = "netP",
    pattern: Literal["outgoing", "incoming"] = "outgoing",
    k: Optional[int] = None,
    k_range: Sequence[int] = tuple(range(2, 11)),
    heatmap_show: bool = True,
    color_use: Optional[Sequence[str]] = None,
    color_heatmap: str = "Spectral",
    title_legend: str = "Contributions",
    width: float = 4.0,
    height: float = 6.0,
    font_size: int = 8,
    seed: int = 10,
) -> ad.AnnData:
    """Identify major signals + general communication patterns via NMF.

    Direct port of R's :func:`identifyCommunicationPatterns`
    (analysis.R:1219). Per design contract §3.B the underlying
    ``NMF::nmf(method='lee', seed='nndsvd')`` call is replaced by
    sklearn's ``solver='mu'`` + NNDSVD initialisation (Tier-2 deviation).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"][slot_name]["prob"]``.
    slot_name : str, default ``"netP"``
    pattern : {"outgoing", "incoming"}, default ``"outgoing"``
    k : int, optional
        Number of patterns. ``None`` raises (matches R).
    k_range : sequence of int
        Reserved (matches R API; not used when ``k`` is provided).
    heatmap_show, color_use, color_heatmap, title_legend, width,
    height, font_size : visual parameters carried through to the slot
        for downstream plotting helpers (slice 6a).
    seed : int, default ``10``
        Random seed.

    Returns
    -------
    anndata.AnnData
        ``adata`` with
        ``uns["cellchat"][slot_name]["pattern"][pattern]`` populated as
        ``{"data": ndarray, "pattern": {"cell": DataFrame,
        "signaling": DataFrame}, "W": ndarray, "H": ndarray,
        "rows_kept": ndarray}``.
    """
    cc = _ensure_uns(adata)
    if slot_name not in cc:
        raise ValueError(f"adata.uns['cellchat']['{slot_name}'] is missing.")
    slot = cc[slot_name]
    if "prob" not in slot:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['prob'] is missing."
        )
    if k is None:
        raise ValueError("Please run the function `selectK` for selecting a suitable k!")

    prob = np.asarray(slot["prob"], dtype=float)
    data, data0, keep_idx = _prepare_pattern_data(prob, pattern)

    # Need cell-group + pathway names. R reads dimnames(prob)[[1]] / [[3]].
    # We rely on the upstream slot to provide them or fall back to indices.
    cell_names: List[str] = list(slot.get("dimnames", [None, None, None])[0]) \
        if isinstance(slot.get("dimnames", None), list) else []
    pathway_names: List[str] = []
    if isinstance(slot.get("pathways", None), list):
        pathway_names = list(slot["pathways"])
    if len(cell_names) == 0:
        K = prob.shape[0]
        cell_names = [str(i + 1) for i in range(K)]
    if len(pathway_names) == 0:
        nLR = prob.shape[2]
        pathway_names = [str(i + 1) for i in range(nLR)]

    # Run NMF (Tier-2 sklearn substitute for NMF::nmf(method='lee'))
    W, H = _lee_nmf(data, rank=int(k), seed=int(seed))
    # R: scaleMat(W, 'r1') and scaleMat(H, 'c1')
    W_scaled = scaleMat(W, "r1")
    H_scaled = scaleMat(H, "c1")
    pattern_labels = [f"Pattern {i + 1}" for i in range(int(k))]

    # Build the long-form data frames (mirrors R's
    # as.data.frame(as.table(.)) which is column-major: the first
    # dimension varies fastest within each level of the second).
    # For W (cells x patterns): outer = Pattern, inner = CellGroup.
    # For H (patterns x signalings): outer = Signaling, inner = Pattern.
    cells_kept = [cell_names[i] for i in keep_idx]
    df_W_rows: List[Dict[str, Any]] = []
    for j, pat in enumerate(pattern_labels):
        for i, cell in enumerate(cells_kept):
            df_W_rows.append({
                "CellGroup": cell, "Pattern": pat,
                "Contribution": float(W_scaled[i, j]),
            })
    df_W = pd.DataFrame(df_W_rows)
    df_H_rows: List[Dict[str, Any]] = []
    for j, sig in enumerate(pathway_names):
        for i, pat in enumerate(pattern_labels):
            df_H_rows.append({
                "Pattern": pat, "Signaling": sig,
                "Contribution": float(H_scaled[i, j]),
            })
    df_H = pd.DataFrame(df_H_rows)

    if "pattern" not in slot:
        slot["pattern"] = {}
    slot["pattern"][pattern] = {
        # R parity (CellChat::analysis.R:445):
        # ``slot$pattern[[pattern]] <- list(data = data0, ...)`` — R stores
        # the **unfiltered** column-max-normalised matrix under the
        # ``"data"`` key. ``data_filtered`` exposes the filtered NMF input
        # for callers that want exactly what was fed to NMF.
        "data": data0,
        "data_filtered": data,
        "pattern": {"cell": df_W, "signaling": df_H},
        "W": W_scaled,
        "H": H_scaled,
        "rows_kept": keep_idx,
        "cells_kept": cells_kept,
        "pattern_labels": pattern_labels,
        "options": {
            "heatmap_show": heatmap_show,
            "color_use": color_use,
            "color_heatmap": color_heatmap,
            "title_legend": title_legend,
            "width": width,
            "height": height,
            "font_size": font_size,
        },
    }
    return adata


# ---------------------------------------------------------------------------
# computeEnrichmentScore (R: analysis.R:3337)
# ---------------------------------------------------------------------------
def computeEnrichmentScore(
    df: pd.DataFrame,
    measure: Literal["ligand", "signaling", "LR-pair"] = "ligand",
    species: Literal["mouse", "human"] = "mouse",
    color_use: Optional[Sequence[str]] = None,
    color_name: str = "Dark2",
    n_color: int = 8,
    scale: Tuple[float, float] = (4.0, 0.8),
    min_freq: float = 0.0,
    max_words: int = 200,
    random_order: bool = False,
    rot_per: float = 0.0,
    return_data: bool = False,
    seed: int = 1,
    db: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Compute (and optionally return) per-ligand enrichment scores.

    Direct port of R's :func:`computeEnrichmentScore` (analysis.R:3337).
    R also wraps this in a ``wordcloud::wordcloud`` plot — the visual
    rendering lives in slice 6a. This Python port mirrors R's contract:
    ``return None`` when ``return_data=False`` (R returns
    ``invisible(NULL)``); return the per-ligand summary
    :class:`pandas.DataFrame` when ``return_data=True``.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table from :func:`netMappingDEG` with columns
        ``interaction_name``, ``ligand.logFC``, ``receptor.logFC``,
        ``ligand.pct.1``, ``ligand.pct.2``, ``receptor.pct.1``,
        ``receptor.pct.2``.
    measure : {"ligand", "signaling", "LR-pair"}
    species : {"mouse", "human"}
    color_use, color_name, n_color, scale, min_freq, max_words,
    random_order, rot_per, seed : visual parameters carried through
        for the slice-6a plot helper. They do not affect the returned
        data frame.
    return_data : bool, default ``False``
    db : dict, optional
        Override CellChatDB (must have an ``interaction`` data frame
        indexed by ``interaction_name``).

    Returns
    -------
    pandas.DataFrame
        ``{"ligand": str, "total": float}`` (the per-ligand summary).
    """
    if measure not in ("ligand", "signaling", "LR-pair"):
        raise ValueError("`measure` must be one of 'ligand', 'signaling', 'LR-pair'.")
    if species not in ("mouse", "human"):
        raise ValueError("`species` must be 'mouse' or 'human'.")
    LRpairs = list(dict.fromkeys(df["interaction_name"].astype(str).tolist()))
    ES = np.zeros(len(LRpairs), dtype=float)
    for i, lr in enumerate(LRpairs):
        df_i = df.loc[df["interaction_name"].astype(str) == lr]
        # drop rows with any NA
        keep = ~df_i.isna().any(axis=1)
        df_i = df_i.loc[keep]
        if len(df_i) == 0:
            ES[i] = 0.0
            continue
        ES[i] = float(np.mean(
            np.abs(df_i["ligand.logFC"].to_numpy(dtype=float)) *
            np.abs(df_i["receptor.logFC"].to_numpy(dtype=float)) *
            np.abs(df_i["ligand.pct.2"].to_numpy(dtype=float) - df_i["ligand.pct.1"].to_numpy(dtype=float)) *
            np.abs(df_i["receptor.pct.2"].to_numpy(dtype=float) - df_i["receptor.pct.1"].to_numpy(dtype=float))
        ))
    DB = _resolve_db(species, db)
    interaction = DB["interaction"].loc[LRpairs, ["ligand", "receptor", "pathway_name"]].copy()
    interaction["score"] = ES
    df_es_ensemble = (
        interaction.groupby("ligand", as_index=False)["score"].sum()
        .rename(columns={"score": "total"})
    )
    if return_data:
        return df_es_ensemble
    return None


# ---------------------------------------------------------------------------
# identifyCellTopics (R: analysis.R:3393)
# ---------------------------------------------------------------------------
def _select_centr_cell(
    centr_cell: Dict[str, Any] | np.ndarray,
    measure: str,
) -> np.ndarray:
    """Slice the ``centr.cell`` 3-D array (``measure`` x ``cell`` x ``LR``).

    :func:`netAnalysis_computeCentrality` (sub-agent A) writes
    ``centr.cell`` as a dict with keys ``{'values', 'centr_names',
    'node_names', 'signaling_names'}``: ``values`` is a
    ``(n_centr, nC, nLR)`` ndarray and ``centr_names`` is the parallel
    measure-name list. This helper accepts that layout and returns the
    ``(nC, nLR)`` slice for the requested measure. For backward
    compatibility the function also accepts:

    - a dict keyed directly by measure name (each value ``(nC, nLR)``),
    - a 3-D ndarray when the caller has no name list.
    """
    if isinstance(centr_cell, dict):
        # Sub-agent A's layout: {'values': (n_centr, nC, nLR), 'centr_names': [...]}
        if "values" in centr_cell and "centr_names" in centr_cell:
            names = list(centr_cell["centr_names"])
            if measure not in names:
                raise ValueError(
                    f"centr.cell missing measure {measure!r}; available: {names}"
                )
            idx = names.index(measure)
            arr = np.asarray(centr_cell["values"], dtype=float)
            return arr[idx, :, :]
        # Legacy layout: dict keyed directly by measure
        if measure not in centr_cell:
            raise ValueError(
                f"centr.cell missing measure {measure!r}; available: "
                f"{list(centr_cell.keys())}"
            )
        return np.asarray(centr_cell[measure], dtype=float)
    arr = np.asarray(centr_cell, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            "centr.cell must be a 3-D array or dict keyed by measure."
        )
    # R packs the measure axis first: slice arr[centr_idx, :, :].
    raise ValueError(
        "centr.cell stored as ndarray requires a parallel name list; "
        "store as dict keyed by measure name (preferred)."
    )


def identifyCellTopics(
    adata: ad.AnnData,
    slot_name: str = "net",
    pattern: Literal["incoming", "outgoing"] = "incoming",
    do_scale: bool = False,
    topic_k: Sequence[int] = tuple(range(5, 51, 5)),
    n_reps: int = 10,
    prop: float = 0.33,
    tol: float = 1e-6,
    maxit: int = 600,
    L1: Sequence[float] = (0.0, 0.0),
    verbose: bool = True,
    seed_use: int = 666,
) -> ad.AnnData:
    """Identify cell topics via L1-regularised NMF.

    Direct port of R's :func:`identifyCellTopics` (analysis.R:3393). Per
    design contract §3.B, R's ``RcppML::nmf(L1=...)`` is replaced by
    sklearn's ``solver='cd'`` with ``alpha_W=L1[0]``, ``alpha_H=L1[1]``,
    ``l1_ratio=1.0``, ``init='nndsvd'`` (Tier-2 deviation).

    Parameters
    ----------
    adata : anndata.AnnData
        Object with ``uns["cellchat"][slot_name]["centr.cell"]`` populated by
        :func:`netAnalysis_computeCentrality(do_group=False)`.
    slot_name : str, default ``"net"``
    pattern : {"incoming", "outgoing"}, default ``"incoming"``
    do_scale : bool, default ``False``
        If ``True``, divide each row of the input matrix by its standard
        deviation (mirrors Seurat's ``ScaleData(do.center=FALSE)``).
    topic_k : sequence of int
        Candidate ranks. If ``len > 1`` the function runs cross-validation
        (:func:`suggestK1`) and stores the result; if ``len == 1`` the
        function fits a single NMF and writes the topic embedding.
    n_reps, prop, tol, maxit, L1, verbose, seed_use :
        Forwarded to NMF and ``suggestK1``.

    Returns
    -------
    anndata.AnnData
        ``adata`` with one of the following written:

        - ``uns["cellchat"][slot_name]["tmp"]["suggestK"][pattern]`` =
          ``suggestK1(...)`` output (``len(topic_k) > 1`` branch).
        - ``uns["cellchat"][slot_name]["topic"][pattern]`` =
          ``{"cell": H_t, "signaling": W}`` where ``H_t`` is
          ``(nC, k)`` and ``W`` is ``(nLR, k)`` (``len(topic_k) == 1``
          branch).
    """
    if pattern not in ("incoming", "outgoing"):
        raise ValueError("`pattern` must be 'incoming' or 'outgoing'.")
    slot = _get_slot(adata, slot_name)
    if "centr.cell" not in slot or len(slot["centr.cell"]) == 0:
        raise ValueError(
            "Please run `netAnalysis_computeCentrality` with `do.group=F` to "
            "compute the network centrality scores!"
        )
    centr_cell = slot["centr.cell"]
    measure = "outdeg" if pattern == "outgoing" else "indeg"
    data_pat = _select_centr_cell(centr_cell, measure)  # shape (nC, nLR)
    # Matrix::t -> features x samples = nLR x nC
    data_sr = data_pat.T

    # Feature names: R reads dimnames(centr.cell)[[3]], which maps to
    # slot['centr.cell']['signaling_names']. Fall back to LR.sig.cell
    # then integer placeholders.
    nLR, nC = data_sr.shape
    if (
        isinstance(centr_cell, dict)
        and "signaling_names" in centr_cell
        and centr_cell["signaling_names"] is not None
        and len(list(centr_cell["signaling_names"])) == nLR
    ):
        feature_names = [str(s) for s in centr_cell["signaling_names"]]
    else:
        feature_names = list(slot.get("LR.sig.cell", [str(i + 1) for i in range(nLR)]))
        if len(feature_names) != nLR:
            feature_names = [str(i + 1) for i in range(nLR)]
    cell_names = list(adata.obs_names) if adata.n_obs == nC else [str(i + 1) for i in range(nC)]

    if do_scale:
        # Mirror Seurat::ScaleData(do.center=FALSE): divide each row by
        # sqrt(sum(x_i^2) / (n-1)) — RMS without centering — NOT by
        # numpy's std (which divides by sqrt(sum((x_i - mean)^2)/(n-1))).
        n_obs = data_sr.shape[1]
        denom_n = max(n_obs - 1, 1)
        # nan-aware sum-of-squares per row
        sumsq = np.nansum(data_sr ** 2, axis=1, keepdims=True)
        rms = np.sqrt(sumsq / float(denom_n))
        rms_safe = np.where(rms == 0.0, 1.0, rms)
        data_sr = data_sr / rms_safe

    if len(list(topic_k)) > 1:
        res_suggest = suggestK1(
            Mat=data_sr,
            k_test=topic_k,
            reps=int(n_reps),
            p=float(prop),
            tol=float(tol),
            maxit=int(maxit),
            L1=L1,
            seed_use=int(seed_use),
        )
        if "tmp" not in slot:
            slot["tmp"] = {}
        if "suggestK" not in slot["tmp"]:
            slot["tmp"]["suggestK"] = {}
        slot["tmp"]["suggestK"][pattern] = res_suggest
    else:
        k = int(list(topic_k)[0])
        W, H, n_iter = _l1_nmf(
            data=data_sr, rank=k, L1=L1,
            seed=int(seed_use), max_iter=int(maxit), tol=float(tol),
        )
        # R: W <- outs_NMF$w %*% diag(sqrt(outs_NMF$d));
        #    H <- diag(sqrt(d)) %*% h |> t()
        # sklearn has no `d`; redistribute the per-factor scale so that
        # ||W[:, j]|| == ||H[j, :]|| post-rescale (matches R's symmetric
        # layout used downstream by Seurat's Loadings()).
        W, H = _redistribute_d(W, H)
        H_t = H.T  # shape (nC, k)
        topic_labels = [f"Topic_{i + 1}" for i in range(k)]
        # store as DataFrames to preserve names (mirrors R's row/col-named
        # sparse matrices). Keep the underlying ndarray accessible via
        # `.values`.
        df_H = pd.DataFrame(H_t, index=cell_names, columns=topic_labels)
        df_W = pd.DataFrame(W, index=feature_names, columns=topic_labels)
        if "topic" not in slot:
            slot["topic"] = {}
        slot["topic"][pattern] = {
            "cell": df_H,
            "signaling": df_W,
            "iter": int(n_iter),
        }
    return adata


# ---------------------------------------------------------------------------
# identifyTopicCommunities (R: analysis.R:3521)
# ---------------------------------------------------------------------------
def identifyTopicCommunities(
    adata: ad.AnnData,
    slot_name: str = "net",
    pattern: Literal["incoming", "outgoing"] = "incoming",
    resolution: Optional[float] = None,
    clustering_method: Literal["seurat", "autoStardust", "weightStardust", "fusion"] = "seurat",
    nPrograms: Optional[int] = None,
    k_signaling: int = 20,
    k_spatial: int = 20,
    method: str = "matrix",
    algorithm: int = 1,
    prune_SNN: float = 1.0 / 15.0,
    spatialWeight: float = 0.5,
    k_fusion: int = 20,
    t_fusion: int = 20,
    truncated: bool = False,
    resRange: Optional[Sequence[float]] = None,
    eigengap: bool = True,
    tol: float = 0.01,
    min_pct: float = 0.1,
    logfc_threshold: float = 0.25,
    do_plot: bool = False,
    filename: str = "",
    seed_use: int = 666,
    return_seurat: bool = True,
) -> ad.AnnData:
    """Identify topic communities (R: analysis.R:3521).

    Direct port. The branch logic mirrors R:

    - ``"seurat"``: build SNN on the topic embedding via
      :func:`spatialcellchat._snn.buildSNN`.
    - ``"autoStardust"`` / ``"weightStardust"``: combine PCA-distance with
      a normalised spatial-distance term, then build SNN on the fused
      distance.
    - ``"fusion"``: SNF (Wang et al., 2014) over per-modality SNNs.
      Requires :func:`spatialcellchat._snf.snf` from sub-agent A. If
      that helper is a stub, this branch raises ``NotImplementedError``.

    Clustering is run by Leiden (``leidenalg``), sweeping
    ``resRange`` if ``resolution is None`` and selecting the resolution
    whose cluster count best matches the eigengap-derived ``K``.

    Parameters
    ----------
    adata : anndata.AnnData
    slot_name : str, default ``"net"``
    pattern : {"incoming", "outgoing"}, default ``"incoming"``
    resolution : float, optional
    clustering_method : str
    nPrograms : int, optional
        Number of topics to use; defaults to ``W.shape[1]``.
    k_signaling, k_spatial, k_fusion : int
    t_fusion : int
    truncated : bool
    method : str
        Reserved (R's igraph/matrix selector). Ignored here — Leiden runs
        over igraph regardless.
    algorithm : int
        ``1``-``4``: 1 = Louvain (igraph::community_multilevel),
        2 = Louvain w/ multilevel refinement, 3 = SLM, 4 = Leiden
        (leidenalg). 2/3 dispatch to Louvain (closest python-igraph
        analogue). Only algorithm != 4 triggers label renumbering.
    prune_SNN : float
    spatialWeight : float
    resRange : sequence of float, optional
    eigengap : bool
        When ``True``, K is inferred via :func:`computeEigengap` over
        the consensus matrix (R parity). When ``False``, K uses the
        truncated-SVD rule on CM (R: ``min(which(s/sum(s) < tol)) - 1``).
    tol : float
    min_pct, logfc_threshold : float
        Marker-gene filter thresholds. After clustering, a per-cluster
        Wilcoxon rank-sum DE pass runs on the centrality matrix
        (``data_sr``) — mirroring R's
        ``Seurat::FindAllMarkers(only.pos=T, min.pct=min_pct,
        logfc.threshold=logfc_threshold)``. Results are stored under
        ``slot['topic'][pattern]['markers']``.

        *Tier-2 deviation:* Seurat's ``Seurat::FindAllMarkers`` log2FC
        formula is
        ``log2(mean(expm1(x_in)) + 1) - log2(mean(expm1(x_out)) + 1)``,
        designed for log-normalised expression.  The Python port uses
        the simpler ``log2(mean_in / mean_out)`` because the input
        ``data_sr`` is the *centrality* matrix (not log-normalised
        expression), so even R's Seurat formula is applied
        off-distribution; both paths approximate, and Python's choice
        is a documented proxy.
    do_plot : bool, default ``False``
    filename : str
    seed_use : int
    return_seurat : bool
        If ``True`` store the embedding + cluster labels under
        ``uns["cellchat"][slot_name]["topic"][pattern]["seurat_obj"]``.

    Returns
    -------
    anndata.AnnData
        ``adata`` augmented with a per-cell label column under
        ``adata.obs[label_col]``. ``label_col`` is ``"OCR"`` (outgoing)
        or ``"ICR"`` (incoming). Labels follow ``f"{prefix}{cluster_id}"``.

    Raises
    ------
    NotImplementedError
        For ``clustering_method='fusion'`` when :mod:`spatialcellchat._snf`
        is not yet implemented (sub-agent A's territory in slice 5).
    """
    # local imports keep the module loadable when sub-agent A's modules
    # are not yet present.
    from ._snn import buildSNN as _buildSNN  # type: ignore
    from scipy import sparse as _sp

    def _build_snn_from_distance(
        D: np.ndarray, k_param: int, prune_SNN: float,
    ) -> np.ndarray:
        """SNN from a precomputed distance matrix.

        Mirrors :func:`spatialcellchat._snn.buildSNN` but takes the
        kNN graph from the distance matrix's row-wise argpartition. Used
        for the ``autoStardust``/``weightStardust``/``fusion`` branches
        which build an SNN over a pre-computed distance.
        """
        n = D.shape[0]
        k = int(k_param)
        if n < k:
            raise ValueError("k cannot be greater than the number of samples")
        # nearest-neighbour indices: drop the diagonal (self-match) and
        # take the k-1 closest (mirrors R's `nn.index[, 1:(k-1)]`)
        D_for_nn = D.copy()
        np.fill_diagonal(D_for_nn, np.inf)
        nn_idx = np.argsort(D_for_nn, axis=1, kind="stable")[:, : k - 1]
        nn_ranked = np.column_stack([np.arange(n), nn_idx])
        rows = np.repeat(np.arange(n), k)
        cols = nn_ranked.ravel()
        data_ones = np.ones(n * k, dtype=float)
        S = _sp.coo_matrix((data_ones, (rows, cols)), shape=(n, n)).tocsr()
        SNN = (S @ S.T).tocsr()
        SNN.data = SNN.data / (k + (k - SNN.data))
        SNN.data[SNN.data < float(prune_SNN)] = 0.0
        SNN.eliminate_zeros()
        SNN = SNN.tolil()
        SNN.setdiag(1.0)
        return np.asarray(SNN.tocsr().toarray(), dtype=float)

    slot = _get_slot(adata, slot_name)
    if "centr.cell" not in slot or len(slot["centr.cell"]) == 0:
        raise ValueError(
            "Please run `netAnalysis_computeCentrality` with `do.group=F` to "
            "compute the network centrality scores!"
        )
    if "topic" not in slot or pattern not in slot["topic"]:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['topic'][{pattern!r}] is "
            "missing; run `identifyCellTopics` with a single `topic_k` first."
        )
    topic = slot["topic"][pattern]
    cell_loadings = topic["cell"]
    sig_loadings = topic["signaling"]
    if isinstance(cell_loadings, pd.DataFrame):
        cell_loadings = cell_loadings.values
    if isinstance(sig_loadings, pd.DataFrame):
        sig_loadings = sig_loadings.values
    H = np.asarray(cell_loadings, dtype=float)  # (nC, k)
    W = np.asarray(sig_loadings, dtype=float)  # (nLR, k)
    n_cells, k_topics = H.shape

    # Re-compute data_sr from centr.cell for downstream marker DE (R:
    # data_sr <- Matrix::t(centr.cell[c('outdeg'/'indeg'),,,drop=T])).
    # This mirrors the same data Seurat::FindAllMarkers consumes inside
    # the R reference function.
    centr_cell_dict = slot["centr.cell"]
    measure = "outdeg" if pattern == "outgoing" else "indeg"
    data_pat = _select_centr_cell(centr_cell_dict, measure)  # (nC, nLR)
    data_sr = data_pat.T  # (nLR, nC) features-by-cells
    nLR_centr = data_sr.shape[0]
    if (
        isinstance(centr_cell_dict, dict)
        and "signaling_names" in centr_cell_dict
        and centr_cell_dict["signaling_names"] is not None
        and len(list(centr_cell_dict["signaling_names"])) == nLR_centr
    ):
        feature_names = [str(s) for s in centr_cell_dict["signaling_names"]]
    else:
        feature_names = list(slot.get("LR.sig.cell", [str(i + 1) for i in range(nLR_centr)]))
        if len(feature_names) != nLR_centr:
            feature_names = [str(i + 1) for i in range(nLR_centr)]

    if nPrograms is None:
        nPrograms = int(W.shape[1])
    nPrograms = int(min(nPrograms, k_topics))

    # SNN over topic embedding (gene/topic space)
    # buildSNN expects features-by-samples; transpose H (samples-by-features).
    snn_gene = np.asarray(
        _buildSNN(H[:, :nPrograms].T, k=int(k_signaling),
                  prune_SNN=float(prune_SNN)).toarray(),
        dtype=float,
    )

    coords = None
    coord_key_candidates = ("spatial", "X_spatial")
    for key in coord_key_candidates:
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key], dtype=float)
            break

    if clustering_method == "seurat":
        snn_use = snn_gene
    elif clustering_method == "autoStardust":
        if coords is None:
            raise ValueError(
                "autoStardust requires spatial coordinates in adata.obsm['spatial']."
            )
        from scipy.spatial.distance import pdist as _pdist, squareform as _sq
        distPCA = _pdist(H[:, :nPrograms], metric="minkowski", p=2)
        distCoord = _pdist(coords, metric="minkowski", p=2)
        # rescale spatial distance to [0, max(distPCA)]
        distCoord = distCoord * (np.nanmax(distPCA) / np.nanmax(distCoord))
        expr_norm = (distPCA - distPCA.min()) / (distPCA.max() - distPCA.min())
        distCoord = distCoord * expr_norm  # element-wise
        finalDistance = _sq(distPCA + distCoord)
        snn_use = _build_snn_from_distance(
            finalDistance, k_param=int(k_spatial), prune_SNN=float(prune_SNN),
        )
    elif clustering_method == "weightStardust":
        if coords is None:
            raise ValueError(
                "weightStardust requires spatial coordinates in adata.obsm['spatial']."
            )
        from scipy.spatial.distance import pdist as _pdist, squareform as _sq
        distPCA = _pdist(H[:, :nPrograms], metric="minkowski", p=2)
        distCoord = _pdist(coords, metric="minkowski", p=2)
        distCoord = distCoord * (
            (np.nanmax(distPCA) * float(spatialWeight)) / np.nanmax(distCoord)
        )
        finalDistance = _sq(distPCA + distCoord)
        snn_use = _build_snn_from_distance(
            finalDistance, k_param=int(k_spatial), prune_SNN=float(prune_SNN),
        )
    elif clustering_method == "fusion":
        # SNF (Wang et al., 2014) — sub-agent A's territory. The import
        # raises ImportError naturally when the helper is absent; we
        # surface that as NotImplementedError per the slice-5 contract.
        import importlib as _il
        if _il.util.find_spec("spatialcellchat._snf") is None:
            raise NotImplementedError(
                "clustering_method='fusion' requires spatialcellchat._snf "
                "(sub-agent A's territory in slice 5)."
            )
        from ._snf import snf as _snf  # type: ignore
        if coords is None:
            raise ValueError(
                "fusion requires spatial coordinates in adata.obsm['spatial']."
            )
        snn_spatial = np.asarray(
            _buildSNN(coords.T, k=int(k_spatial), prune_SNN=float(prune_SNN)).toarray(),
            dtype=float,
        )
        snn_fusion = _snf([snn_gene, snn_spatial], k=int(k_fusion), t=int(t_fusion))
        snn_fusion = snn_fusion / np.max(np.diag(snn_fusion))
        if truncated:
            n_spot = snn_fusion.shape[1]
            for j in range(n_spot):
                col = snn_fusion[:, j]
                drop_idx = np.argsort(-col)[(int(k_fusion) + 1):]
                snn_fusion[drop_idx, j] = 0.0
            snn_use = snn_fusion
        else:
            finalDistance = 1.0 - snn_fusion
            snn_use = _build_snn_from_distance(
                finalDistance, k_param=int(k_spatial), prune_SNN=float(prune_SNN),
            )
    else:
        raise ValueError(f"Unknown clustering_method: {clustering_method!r}")

    # Build the SNN igraph once; the algorithm dispatcher consumes it.
    import igraph as _ig
    import leidenalg as _la
    from .analysis_network import computeEigengap as _compute_eigengap

    snn_arr = np.asarray(snn_use, dtype=float)
    n = snn_arr.shape[0]
    src, dst = np.nonzero(snn_arr)
    weights = snn_arr[src, dst]
    g = _ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())),
                  directed=False, edge_attrs={"weight": weights.tolist()})
    g.simplify(combine_edges="max", multiple=True, loops=True)

    def _cluster_at_resolution(res: float) -> List[int]:
        """Dispatch to the requested community-detection algorithm.

        R supports algorithm in {1, 2, 3, 4} via Seurat::FindClusters:
        1 = Louvain (igraph::community_multilevel),
        2 = Louvain with multilevel refinement,
        3 = SLM,
        4 = Leiden (leidenalg).

        For 1-3 we use python-igraph's community detection; only
        algorithm 4 invokes leidenalg.
        """
        algo = int(algorithm)
        if algo == 4:
            partition = _la.find_partition(
                g, _la.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=float(res),
                seed=int(seed_use),
            )
            return list(partition.membership)
        if algo == 1:
            # Louvain with resolution
            comm = g.community_multilevel(
                weights="weight", resolution=float(res),
            )
            return list(comm.membership)
        if algo in (2, 3):
            # python-igraph does not expose multilevel-refinement /
            # SLM separately; fall back to Louvain + resolution as the
            # closest analogue.
            comm = g.community_multilevel(
                weights="weight", resolution=float(res),
            )
            return list(comm.membership)
        raise ValueError(
            f"algorithm must be in {{1, 2, 3, 4}}; got {algorithm!r}"
        )

    if resolution is not None:
        labels = _cluster_at_resolution(resolution)
    else:
        if resRange is None:
            if n < 50:
                resRange = list((0.02, 0.06, 0.08)) + list(np.arange(0.1, 1.11, 0.2).tolist())
            else:
                resRange = list((0.02, 0.06, 0.08)) + list(np.arange(0.1, 1.51, 0.2).tolist())
        # consensus matrix
        CM = np.zeros((n, n), dtype=float)
        ncluster: List[int] = []
        cluster_runs: List[List[int]] = []
        for r in resRange:
            lbl = _cluster_at_resolution(float(r))
            cluster_runs.append(lbl)
            ncluster.append(len(set(lbl)))
            arr = np.asarray(lbl)
            CM = CM + (arr[:, None] == arr[None, :]).astype(float)
        CM = CM / float(len(resRange))
        # Determine K: R uses computeEigengap(CM, tol) when eigengap=True,
        # else irlba-SVD on CM with K = min(which((s/sum(s) < tol))) - 1.
        if eigengap:
            res_eigengap = _compute_eigengap(CM, tol=float(tol))
            K = int(res_eigengap["upper_bound"])
        else:
            from numpy.linalg import svd as _svd
            n_cell = CM.shape[0]
            n_dims_consensus = 30
            n_pc = int(min(n_dims_consensus, n_cell))
            _, s, _ = _svd(CM, full_matrices=False)
            s = s[:n_pc]
            below = np.flatnonzero((s / s.sum()) < float(tol))
            K = int(below[0]) - 1 if below.size > 0 else int(len(s)) - 1
            if K < 1:
                K = 1
        # pick the resolution whose cluster count is closest to K (R:
        # min(which(abs(ncluster - K) == min(abs(ncluster - K)))))
        ncluster_arr = np.asarray(ncluster, dtype=int)
        diffs = np.abs(ncluster_arr - int(K))
        best_idx = int(np.argmin(diffs))
        resolution = float(resRange[best_idx])
        labels = cluster_runs[best_idx]

        # R's resolution loop: while smallest cluster < 10, walk resRange
        # backwards and re-cluster until min cluster size >= 10 (or no
        # more lower resolutions remain).
        def _min_cluster_size(lbl_seq: Sequence[int]) -> int:
            cs = np.unique(np.asarray(lbl_seq), return_counts=True)[1]
            return int(cs.min()) if cs.size else 0

        while _min_cluster_size(labels) < 10 and best_idx > 0:
            best_idx -= 1
            resolution = float(resRange[best_idx])
            labels = _cluster_at_resolution(resolution)

    # Convert labels to consecutive 1..K with sorted order (R:
    # levels <- sort(levels); when algorithm != 4, R relabels via
    # plyr::mapvalues to 1..nlevels). Algorithm 4 (Leiden) keeps the
    # leidenalg output labels untouched.
    labels_arr = np.asarray(labels, dtype=int)
    unique_labels = sorted(set(labels_arr.tolist()))
    if int(algorithm) != 4:
        # Renumber to consecutive 1..K
        relabel = {old: new + 1 for new, old in enumerate(unique_labels)}
        labels_1based = np.array(
            [relabel[v] for v in labels_arr.tolist()], dtype=int,
        )
    else:
        # Leiden — keep label values as-is, but shift to 1-based for the
        # downstream string label format ("ICR1" / "OCR1" etc.).
        labels_1based = labels_arr + 1

    prefix = "OCR" if pattern == "outgoing" else "ICR"
    label_col = f"label.{prefix}"
    str_labels = [f"{prefix}{v}" for v in labels_1based]
    adata.obs[label_col] = pd.Categorical(
        str_labels,
        categories=[f"{prefix}{v}" for v in sorted(set(labels_1based.tolist()))],
        ordered=False,
    )

    # Per-cluster Wilcoxon DE (mirror R's
    # Seurat::FindAllMarkers(only.pos=T, min.pct, logfc.threshold)). R
    # runs DE on the *centrality* matrix (data_sr), not the gene matrix:
    # use that exact matrix here to match. Skip when only one cluster.
    markers_df: Optional[pd.DataFrame] = None
    if len(set(labels_1based.tolist())) > 1:
        # data_sr is shape (nLR, nC) — features-by-cells.
        # Build a per-cluster Wilcoxon table compatible with the
        # FindAllMarkers schema R produces (cluster, gene, p_val,
        # avg_log2FC, pct.1, pct.2, p_val_adj).
        from scipy import stats as _stats
        nLR_, nC_ = data_sr.shape
        clusters = np.asarray(labels_1based, dtype=int)
        unique_clusters = sorted(set(clusters.tolist()))
        rows_dfm: List[Dict[str, Any]] = []
        for cl in unique_clusters:
            in_mask = (clusters == cl)
            out_mask = ~in_mask
            n_in = int(in_mask.sum())
            n_out = int(out_mask.sum())
            if n_in == 0 or n_out == 0:
                continue
            for f_idx in range(nLR_):
                x_in = data_sr[f_idx, in_mask]
                x_out = data_sr[f_idx, out_mask]
                pct1 = float(np.mean(x_in > 0))
                pct2 = float(np.mean(x_out > 0))
                if max(pct1, pct2) < float(min_pct):
                    continue
                # log2FC of mean+1 (Seurat-style for non-log data)
                mean_in = float(np.mean(x_in))
                mean_out = float(np.mean(x_out))
                log2fc = float(np.log2(mean_in + 1e-9) - np.log2(mean_out + 1e-9))
                if log2fc < float(logfc_threshold):
                    continue
                _, pval = _stats.ranksums(x_in, x_out)
                rows_dfm.append({
                    "cluster": f"{prefix}{cl}",
                    "gene": feature_names[f_idx],
                    "p_val": float(pval),
                    "avg_log2FC": float(log2fc),
                    "pct.1": pct1,
                    "pct.2": pct2,
                })
        if rows_dfm:
            markers_df = pd.DataFrame(rows_dfm)
            # Bonferroni-corrected p_val_adj per cluster
            markers_df["p_val_adj"] = (
                markers_df["p_val"] * float(nLR_)
            ).clip(upper=1.0)

    if "topic" not in slot:
        slot["topic"] = {}
    if pattern not in slot["topic"]:
        slot["topic"][pattern] = {}
    slot["topic"][pattern]["clusters"] = labels_1based
    slot["topic"][pattern]["resolution"] = float(resolution)
    slot["topic"][pattern]["label_col"] = label_col
    if markers_df is not None:
        slot["topic"][pattern]["markers"] = markers_df
    if return_seurat:
        slot["topic"][pattern]["seurat_obj"] = {
            "cell_embeddings": H,
            "loadings": W,
            "labels": labels_1based,
            "label_col": label_col,
        }
    return adata


# ---------------------------------------------------------------------------
# extractTopicSignaling (R: analysis.R:3869)
# ---------------------------------------------------------------------------
def _top_features(
    loadings: pd.DataFrame, dim: int, nfeatures: int, balanced: bool,
) -> List[str]:
    """Mirror Seurat's ``Top()`` selector for a single dim."""
    col = loadings.iloc[:, dim - 1]
    nr = len(col)
    if nfeatures > nr:
        nfeatures = nr
    if nfeatures == 1:
        balanced = False
    if balanced:
        num = int(round(nfeatures / 2))
        sorted_desc = col.sort_values(ascending=False)
        positive = sorted_desc.index[:num].tolist()
        negative = sorted_desc.index[-num:][::-1].tolist()
        if num > 0 and positive[num - 1] == negative[num - 1]:
            negative = negative[:-1]
        return list(positive) + list(negative)
    abs_sorted = col.abs().sort_values(ascending=False)
    top = abs_sorted.index[:nfeatures].tolist()
    # R reorders these by ascending value of `data[top, ]`
    sub = col.loc[top].sort_values(ascending=True)
    return sub.index.tolist()


def extractTopicSignaling(
    adata: ad.AnnData,
    dims: Sequence[int] = (1, 2, 3, 4, 5),
    nfeatures: int = 30,
    nTop: int = 10,
    feature_cat: Literal["LR-pair", "ligand", "signaling"] = "LR-pair",
    reduction: str = "topic",
    pattern: Literal["incoming", "outgoing"] = "incoming",
    slot_name: str = "net",
    projected: bool = False,
    balanced: bool = False,
    species: Literal["mouse", "human"] = "mouse",
    db: Optional[Dict[str, pd.DataFrame]] = None,
    simplify: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Extract per-topic loading rankings (R: analysis.R:3869).

    Direct port. R operates on a Seurat ``DimReduc`` object via
    :func:`Seurat::Loadings` + :func:`Seurat::TopFeatures`. The Python
    port reads the corresponding ``loadings`` matrix from
    ``adata.uns["cellchat"][slot_name]["topic"][pattern]["signaling"]``
    (a ``(nLR, k)`` :class:`pandas.DataFrame` written by
    :func:`identifyCellTopics`).

    Parameters
    ----------
    adata : anndata.AnnData
    dims : sequence of int
        1-based topic indices (matches R API).
    nfeatures : int
        Number of features to extract per topic before optional grouping.
    nTop : int
        Final number of rows kept per topic after grouping.
    feature_cat : {"LR-pair", "ligand", "signaling"}
        Aggregation level. ``"ligand"`` summarises by ligand,
        ``"signaling"`` by pathway, ``"LR-pair"`` keeps individual pairs.
    reduction : str, default ``"topic"``
        Logical reduction name (used only as the prefix for the output
        ``reduction`` column, mirrors R's ``Key(object[[reduction]])``
        which yields e.g. ``"topic_"``).
    pattern : {"incoming", "outgoing"}, default ``"incoming"``
        Which pattern's loadings to read from
        ``slot["topic"][pattern]["signaling"]``.
    slot_name : str, default ``"net"``
    projected : bool, default ``False``
        If ``True`` use the ``signaling.projected`` slot if present
        (otherwise falls back to ``signaling``).
    balanced : bool
    species : {"mouse", "human"}
    db : dict, optional
        Override CellChatDB.
    simplify : bool
        ``True`` returns a single concatenated data frame; ``False``
        returns a ``dict[str, DataFrame]`` keyed by ``"<reduction>_<dim>"``.

    Returns
    -------
    pandas.DataFrame or dict
    """
    if feature_cat not in ("LR-pair", "ligand", "signaling"):
        raise ValueError("`feature_cat` must be 'LR-pair', 'ligand', or 'signaling'.")
    slot = _get_slot(adata, slot_name)
    if "topic" not in slot or pattern not in slot["topic"]:
        raise ValueError(
            f"adata.uns['cellchat']['{slot_name}']['topic'][{pattern!r}] is "
            "missing; run `identifyCellTopics` first."
        )
    topic = slot["topic"][pattern]
    loadings_key = "signaling.projected" if projected and "signaling.projected" in topic else "signaling"
    loadings_df = topic[loadings_key]
    if not isinstance(loadings_df, pd.DataFrame):
        raise TypeError(
            f"topic[{pattern!r}][{loadings_key!r}] must be a pandas DataFrame."
        )

    DB = _resolve_db(species, db)
    interaction = DB["interaction"]

    dims_list = list(int(d) for d in dims)
    features_per_dim: Dict[int, List[str]] = {}
    for d in dims_list:
        features_per_dim[d] = _top_features(loadings_df, d, int(nfeatures), bool(balanced))

    # union of features kept across all dims (mirrors R's
    # `loadings[unlist(features), dims]` slice)
    union_feats: List[str] = []
    seen: set = set()
    for d in dims_list:
        for f in features_per_dim[d]:
            if f not in seen:
                seen.add(f)
                union_feats.append(f)

    df_loadings_per_dim: Dict[str, pd.DataFrame] = {}
    reduction_key = f"{reduction}_"
    for d in dims_list:
        feats = features_per_dim[d]
        scores = loadings_df.loc[feats, loadings_df.columns[d - 1]].astype(float)
        df = pd.DataFrame({
            "feature": [f.replace("-", "_") for f in feats],
            "score": scores.values,
        })
        # join on interaction db
        ix = df["feature"].tolist()
        # If feature names are not in the DB index, leave NaN (mirrors R's
        # NA propagation when interaction[df$feature, ...] misses a row).
        valid = [f for f in ix if f in interaction.index]
        sub = interaction.loc[valid, ["ligand", "receptor", "pathway_name"]] if len(valid) else \
            pd.DataFrame(columns=["ligand", "receptor", "pathway_name"])
        # reindex to keep df ordering
        sub = sub.reindex(ix)
        df = pd.concat([df.reset_index(drop=True), sub.reset_index(drop=True)], axis=1)

        if feature_cat == "ligand":
            df_g = df.groupby("ligand", as_index=False)["score"].mean()
            df_g.columns = ["feature", "score"]
        elif feature_cat == "signaling":
            df_g = df.groupby("pathway_name", as_index=False)["score"].mean()
            df_g.columns = ["feature", "score"]
        else:
            df_g = df

        df_g = df_g.copy()
        df_g["absolute.v"] = df_g["score"].abs()
        df_g = df_g.sort_values("absolute.v", ascending=False).reset_index(drop=True)
        df_g["rank"] = np.arange(1, len(df_g) + 1)
        df_g["reduction"] = f"{reduction_key}{d}"
        df_g = df_g.head(int(nTop))
        df_loadings_per_dim[f"{reduction_key}{d}"] = df_g

    if simplify:
        out = pd.concat(list(df_loadings_per_dim.values()), axis=0, ignore_index=True)
        return out
    return df_loadings_per_dim
