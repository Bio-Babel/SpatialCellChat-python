"""Adaptive Low-Rank Approximation (ALRA) — direct port of the R ``ALRA``
package, kept private and trimmed to only what ``preProcessing`` calls.

ALRA (Linderman et al., *Nature Communications* 2022) imputes the dropped
zeros in single-cell RNA-seq counts via a randomized SVD of the
(log-normalised) expression matrix, followed by a per-gene threshold derived
from the symmetric "mirror trick" and a column-wise rescaling that restores
the original mean / variance of the truly-expressed entries.

This file follows the R reference (``ALRA::alra`` and ``ALRA::choose_k``)
line for line so that the public ``preProcessing`` function in
``preprocessing.py`` can be validated cell-by-cell against R.

Reference
---------
KrishnaswamyLab / ALRA — https://github.com/KlugerLab/ALRA
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.utils.extmath import randomized_svd

__all__ = ["alra", "choose_k"]


# ---------------------------------------------------------------------------
# choose_k (R: ALRA::choose_k)
# ---------------------------------------------------------------------------
def choose_k(
    A_norm: np.ndarray,
    K: int = 100,
    thresh: float = 6.0,
    noise_start: int = 80,
    q: int = 2,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Choose the truncation rank ``k`` by spectral-gap detection.

    Mirrors R's ``ALRA::choose_k``. The largest ``K`` singular values are
    computed via randomized SVD; the per-pair differences are split into a
    putative "noise" tail (``noise_start:K``) whose mean and standard
    deviation are used to z-score every gap.  The largest gap with z-score
    above ``thresh`` defines the rank ``k``.

    Parameters
    ----------
    A_norm : numpy.ndarray
        Cells x genes log-normalised expression matrix.
    K : int, default 100
        Number of singular values to compute.
    thresh : float, default 6.0
        Standard-deviation threshold for the noise z-score.
    noise_start : int, default 80
        Index (1-based, R-style) of the first singular gap considered noise.
    q : int, default 2
        Number of power iterations passed to the randomized SVD.
    random_state : int, optional
        Seed forwarded to ``randomized_svd``.

    Returns
    -------
    dict
        ``{"k": int, "num_of_sds": ndarray, "d": ndarray}``.
    """
    if K > min(A_norm.shape):
        raise ValueError(
            "For an m by n matrix, K must be smaller than the min(m,n)."
        )
    if noise_start > K - 5:
        raise ValueError(
            "There need to be at least 5 singular values considered noise."
        )
    _, d, _ = randomized_svd(
        A_norm, n_components=K, n_iter=q, random_state=random_state
    )
    diffs = d[:-1] - d[1:]
    # noise_svals in R is noise_start:K, of length (K - noise_start + 1)
    # diffs is 1-indexed in R; index (i) corresponds to d[i] - d[i+1].
    # diffs[noise_svals - 1] in R == diffs[noise_start - 1 : K] in 0-based py
    noise = diffs[noise_start - 1 : K - 1]
    mu = float(np.mean(noise))
    sigma = float(np.std(noise, ddof=1))
    num_of_sds = (diffs - mu) / sigma
    # R: max(which(num_of_sds > thresh))  — 1-based last index above thresh
    above = np.flatnonzero(num_of_sds > thresh)
    if above.size == 0:
        raise ValueError("No singular value exceeds the noise threshold.")
    k = int(above[-1] + 1)  # back to 1-based to match R semantics
    return {"k": k, "num_of_sds": num_of_sds, "d": d}


# ---------------------------------------------------------------------------
# alra (R: ALRA::alra)
# ---------------------------------------------------------------------------
def alra(
    A_norm: np.ndarray,
    k: int = 0,
    q: int = 10,
    quantile_prob: float = 0.001,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Compute the rank-``k`` low-rank reconstruction with the ALRA threshold.

    Mirrors R's ``ALRA::alra``. Returns the un-thresholded reconstruction
    (``A_norm_rank_k``), the threshold-zeroed matrix
    (``A_norm_rank_k_cor``) and the rescaled imputed matrix
    (``A_norm_rank_k_cor_sc``).

    Parameters
    ----------
    A_norm : numpy.ndarray
        Cells x genes log-normalised expression matrix.
    k : int, default 0
        Truncation rank. ``0`` triggers automatic selection via
        :func:`choose_k`.
    q : int, default 10
        Power iterations for the randomized SVD.
    quantile_prob : float, default 0.001
        Per-gene mirror-trick quantile probability used as the threshold.
    random_state : int, optional
        Seed for the randomized SVD.

    Returns
    -------
    dict
        ``{"A_norm_rank_k": ..., "A_norm_rank_k_cor": ...,
        "A_norm_rank_k_cor_sc": ...}``.
    """
    if not isinstance(A_norm, np.ndarray):
        raise TypeError(
            "A_norm must be a numpy.ndarray (cells x genes); call np.asarray()."
        )

    if k == 0:
        chosen = choose_k(A_norm, random_state=random_state)
        k = chosen["k"]

    originally_nonzero = A_norm > 0

    # randomized SVD — sklearn's interface mirrors ``rsvd::rsvd``
    u, d, vt = randomized_svd(
        A_norm, n_components=k, n_iter=q, random_state=random_state
    )
    # rank-k reconstruction
    A_norm_rank_k = (u * d) @ vt

    # per-column quantile — np.quantile linear interp matches R's default
    # type 7. abs() mirrors the R source.
    A_norm_rank_k_mins = np.abs(
        np.quantile(A_norm_rank_k, quantile_prob, axis=0, method="linear")
    )

    # threshold: zero entries that are <= per-column threshold
    A_norm_rank_k_cor = np.where(
        A_norm_rank_k <= A_norm_rank_k_mins[np.newaxis, :],
        0.0,
        A_norm_rank_k,
    )

    # column statistics over non-zero entries only
    sigma_1 = _col_sd_nonzero(A_norm_rank_k_cor)
    sigma_2 = _col_sd_nonzero(A_norm)
    nz_cor = (A_norm_rank_k_cor != 0).sum(axis=0)
    nz_norm = (A_norm != 0).sum(axis=0)
    # R does colSums(!!A_norm_rank_k_cor); when nz==0 this is 0/0 -> NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        mu_1 = A_norm_rank_k_cor.sum(axis=0) / nz_cor
        mu_2 = A_norm.sum(axis=0) / nz_norm

    # decide which columns can be rescaled (R's `toscale` mask)
    toscale = (
        ~np.isnan(sigma_1)
        & ~np.isnan(sigma_2)
        & ~((sigma_1 == 0) & (sigma_2 == 0))
        & (sigma_1 != 0)
    )

    sigma_1_2 = np.zeros_like(sigma_2)
    toadd = np.zeros_like(sigma_2)
    sigma_1_2[toscale] = sigma_2[toscale] / sigma_1[toscale]
    toadd[toscale] = -mu_1[toscale] * sigma_2[toscale] / sigma_1[toscale] + mu_2[toscale]

    A_norm_rank_k_cor_sc = A_norm_rank_k_cor.copy()
    cols = np.where(toscale)[0]
    if cols.size:
        sub = A_norm_rank_k_cor[:, cols] * sigma_1_2[cols][np.newaxis, :]
        sub = sub + toadd[cols][np.newaxis, :]
        A_norm_rank_k_cor_sc[:, cols] = sub

    # zeros stay zeros (R: A_norm_rank_k_cor_sc[A_norm_rank_k_cor == 0] = 0)
    A_norm_rank_k_cor_sc[A_norm_rank_k_cor == 0] = 0.0

    # clip negatives to zero
    A_norm_rank_k_cor_sc[A_norm_rank_k_cor_sc < 0] = 0.0

    # restore originally non-zero values that got zeroed
    mask = originally_nonzero & (A_norm_rank_k_cor_sc == 0)
    A_norm_rank_k_cor_sc[mask] = A_norm[mask]

    return {
        "A_norm_rank_k": A_norm_rank_k,
        "A_norm_rank_k_cor": A_norm_rank_k_cor,
        "A_norm_rank_k_cor_sc": A_norm_rank_k_cor_sc,
    }


# ---------------------------------------------------------------------------
# private helper
# ---------------------------------------------------------------------------
def _col_sd_nonzero(M: np.ndarray) -> np.ndarray:
    """Per-column sample standard deviation over the non-zero entries.

    Mirrors R's ``apply(M, 2, sd_nonzero)`` where
    ``sd_nonzero <- function(x) sd(x[!x == 0])``. R's :func:`sd` returns
    ``NA`` when the column has fewer than two non-zeros, which we match
    with :data:`numpy.nan`.
    """
    out = np.empty(M.shape[1], dtype=float)
    for j in range(M.shape[1]):
        col = M[:, j]
        nz = col[col != 0]
        if nz.size < 2:
            out[j] = np.nan
        else:
            out[j] = float(np.std(nz, ddof=1))
    return out
