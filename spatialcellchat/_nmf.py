"""NMF helpers for SpatialCellChat (consensus K-selection + L1-regularised wrappers).

This module centralises the two NMF flavours required by the analysis layer:

- :func:`_consensus_nmf` reproduces ``NMF::nmfEstimateRank`` (used by
  :func:`spatialcellchat.analysis_topics.selectK` and indirectly by
  :func:`identifyCommunicationPatterns` when ``k`` is unspecified). For
  each candidate ``k`` we run ``n_runs`` independent factorisations, build
  the cluster co-occurrence consensus matrix, hierarchically cluster it,
  and report the cophenetic correlation. The optimal ``k`` is the value
  preceding the largest decrease in cophenetic correlation (the "knee"
  rule used by Brunet et al., 2004 — same heuristic as ``NMF::nmf.fit``).

- :func:`_lee_nmf` wraps ``sklearn.decomposition.NMF`` with the
  ``solver='mu'`` Frobenius update that approximates R's ``NMF::nmf
  (method='lee')`` (Tier-2 deviation per design contract §3.B).

- :func:`_l1_nmf` wraps ``sklearn.decomposition.NMF`` with the
  ``solver='cd'`` coordinate-descent update with L1 regularisation that
  approximates R's ``RcppML::nmf(L1=...)`` (Tier-2 deviation per design
  contract §3.B).

References
----------
Brunet et al., 2004, "Metagenes and molecular pattern discovery using
matrix factorization", PNAS 101(12):4164-4169 — Section "Estimating the
number of clusters" defines the cophenetic correlation criterion that
:func:`_consensus_nmf` reproduces.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Tuple

import numpy as np
from scipy.cluster import hierarchy as _hier
from scipy.spatial.distance import squareform as _squareform

__all__ = [
    "_consensus_nmf",
    "_lee_nmf",
    "_l1_nmf",
    "_redistribute_d",
]


def _redistribute_d(W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror R ``RcppML::nmf``'s ``d`` redistribution onto sklearn output.

    R's ``RcppML::nmf`` returns ``$w``, ``$h``, ``$d`` with ``W = w *
    diag(d)``-style scaling. The canonical post-processing is::

        W <- w %*% diag(sqrt(d))
        H <- diag(sqrt(d)) %*% h

    sklearn's NMF returns only ``W``, ``H`` (the scale is folded into
    ``W``). Decompose each factor as ``W[:, j] = n_W_j * w_j`` and
    ``H[j, :] = n_H_j * h_j`` with ``w_j`` / ``h_j`` unit-norm; then the
    factor's true scale is ``d_j = n_W_j * n_H_j``, and redistributing
    that scale equally produces::

        n_W_j = ||W[:, j]||_2,  n_H_j = ||H[j, :]||_2
        d_j   = n_W_j * n_H_j                       # NOT sqrt(n_W*n_H)
        W_new[:, j] = (W[:, j] / n_W_j) * sqrt(d_j)
        H_new[j, :] = (H[j, :] / n_H_j) * sqrt(d_j)

    The product ``W_new @ H_new`` equals ``W @ H`` exactly (because the
    per-factor scale ``d_j / (n_W_j * n_H_j) = 1``), but each factor
    now has the symmetric scaling R produces — both
    ``||W_new[:, j]||`` and ``||H_new[j, :]||`` equal ``sqrt(d_j)``,
    which downstream loadings consume.

    Parameters
    ----------
    W : numpy.ndarray
        ``(n_features, rank)``.
    H : numpy.ndarray
        ``(rank, n_samples)``.

    Returns
    -------
    W_redist : numpy.ndarray
    H_redist : numpy.ndarray
    """
    W = np.asarray(W, dtype=float)
    H = np.asarray(H, dtype=float)
    n_W = np.linalg.norm(W, axis=0)
    n_H = np.linalg.norm(H, axis=1)
    d = n_W * n_H  # per-factor scale (R's `d` from RcppML::nmf)
    # Avoid div-by-zero on degenerate factors (zero column / row).
    n_W_safe = np.where(n_W > 0.0, n_W, 1.0)
    n_H_safe = np.where(n_H > 0.0, n_H, 1.0)
    sqrt_d = np.sqrt(d)
    W_redist = (W / n_W_safe) * sqrt_d[None, :]
    H_redist = (H / n_H_safe[:, None]) * sqrt_d[:, None]
    return W_redist, H_redist


# ---------------------------------------------------------------------------
# Lee MU NMF — proxy for R `NMF::nmf(method='lee')`
# ---------------------------------------------------------------------------
def _lee_nmf(
    data: np.ndarray,
    rank: int,
    seed: int = 10,
    init: str = "nndsvd",
    max_iter: int = 2000,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate R's ``NMF::nmf(method='lee', seed='nndsvd')``.

    Per design contract §3.B this is a Tier-2 deviation: sklearn's
    ``solver='mu'`` minimises the squared Frobenius loss with the same
    multiplicative update form as Lee (2001). The ``init`` strategy
    mirrors NMF's ``"nndsvd"`` seed.

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_features, n_samples)`` non-negative matrix.
    rank : int
        Number of factors.
    seed : int, optional
        Random seed for fallback random initialisations (NMF's NNDSVD is
        deterministic so ``seed`` only matters when ``init`` is randomised).
    init : str, optional
        sklearn NMF init strategy. Default ``"nndsvd"`` matches R.
    max_iter : int, optional
        Maximum iterations (R's NMF default is 2000).
    tol : float, optional
        Stopping tolerance (R's NMF default is 1e-4 on the connectivity
        matrix; here we pass to sklearn's reconstruction-error tolerance).

    Returns
    -------
    W : numpy.ndarray
        ``(n_features, rank)`` basis.
    H : numpy.ndarray
        ``(rank, n_samples)`` coefficients.
    """
    from sklearn.decomposition import NMF as _SkNMF

    # NNDSVD with the MU solver leaves zeros frozen at zero. sklearn warns
    # about this; we silence the warning since the deviation is documented.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _SkNMF(
            n_components=rank,
            init=init,
            solver="mu",
            beta_loss="frobenius",
            max_iter=max_iter,
            tol=tol,
            random_state=int(seed),
        )
        W = model.fit_transform(data)
        H = model.components_
    return W, H


# ---------------------------------------------------------------------------
# L1 (CD) NMF — proxy for R `RcppML::nmf(L1=...)`
# ---------------------------------------------------------------------------
def _l1_nmf(
    data: np.ndarray,
    rank: int,
    L1: Sequence[float] = (0.0, 0.0),
    seed: int = 666,
    init: str = "nndsvd",
    max_iter: int = 600,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Approximate R's ``RcppML::nmf(L1=c(L1[0], L1[1]))``.

    Per design contract §3.B this is a Tier-2 deviation: sklearn's
    ``solver='cd'`` performs coordinate descent on the squared Frobenius
    loss with elastic-net regularisation. Setting ``l1_ratio=1.0`` and
    ``alpha_W=L1[0]``, ``alpha_H=L1[1]`` reproduces RcppML's L1 penalty
    form (in expectation — convergence trajectory differs slightly).

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_features, n_samples)`` non-negative matrix.
    rank : int
        Number of factors.
    L1 : sequence of two floats, optional
        ``(L1_W, L1_H)`` regularisation strengths in ``[0, 1]``.
    seed : int, optional
        Random seed.
    init : str, optional
        sklearn NMF init strategy. Default ``"nndsvd"``.
    max_iter : int, optional
        Maximum iterations.
    tol : float, optional
        Stopping tolerance.

    Returns
    -------
    W : numpy.ndarray
        ``(n_features, rank)`` basis.
    H : numpy.ndarray
        ``(rank, n_samples)`` coefficients.
    iters : int
        Number of iterations actually run (sklearn's ``n_iter_``).
    """
    from sklearn.decomposition import NMF as _SkNMF

    L1_arr = np.asarray(L1, dtype=float)
    if L1_arr.size != 2:
        raise ValueError("`L1` must be a sequence of length 2: (L1_W, L1_H)")
    alpha_W = float(L1_arr[0])
    alpha_H = float(L1_arr[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _SkNMF(
            n_components=rank,
            init=init,
            solver="cd",
            beta_loss="frobenius",
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=1.0,
            max_iter=max_iter,
            tol=tol,
            random_state=int(seed),
        )
        W = model.fit_transform(data)
        H = model.components_
    return W, H, int(model.n_iter_)


# ---------------------------------------------------------------------------
# Consensus + cophenetic — proxy for R `NMF::nmfEstimateRank`
# ---------------------------------------------------------------------------
def _max_factor_clustering(H: np.ndarray) -> np.ndarray:
    """Argmax-based clustering: assign each sample to its dominant factor.

    Mirrors R's ``NMF`` package convention of clustering samples by the
    row of ``H`` (factors x samples) with the largest loading.

    Parameters
    ----------
    H : numpy.ndarray
        ``(rank, n_samples)`` coefficient matrix.

    Returns
    -------
    numpy.ndarray of int
        Length ``n_samples`` cluster labels in ``[0, rank)``.
    """
    return np.argmax(H, axis=0).astype(int)


def _connectivity_matrix(labels: np.ndarray) -> np.ndarray:
    """Build a binary co-occurrence matrix from cluster labels.

    ``C[i, j] = 1`` iff samples ``i`` and ``j`` share a label.

    Parameters
    ----------
    labels : numpy.ndarray of int
        Length ``n`` cluster labels.

    Returns
    -------
    numpy.ndarray of float
        ``(n, n)`` symmetric binary connectivity matrix with ``1`` on the
        diagonal.
    """
    labels = np.asarray(labels).reshape(-1)
    return (labels[:, None] == labels[None, :]).astype(float)


def _cophenetic_correlation(consensus: np.ndarray) -> float:
    """Cophenetic correlation of a consensus matrix.

    Mirrors NMF's ``copheneticCorrelation``: hierarchical clustering with
    average linkage on ``1 - consensus`` produces a cophenetic distance
    that is correlated against the original ``1 - consensus`` distance
    (Pearson). For a perfectly stable factorisation across runs the
    consensus is exactly binary and the correlation is 1.0.

    Parameters
    ----------
    consensus : numpy.ndarray
        ``(n, n)`` consensus matrix in ``[0, 1]`` with 1 on the diagonal.

    Returns
    -------
    float
        Cophenetic correlation in ``[-1, 1]``.
    """
    n = consensus.shape[0]
    if n < 2:
        return 1.0
    # Ensure symmetry & zero diagonal in the distance space.
    dist = 1.0 - consensus
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    condensed = _squareform(dist, checks=False)
    if condensed.size == 0 or np.all(condensed == 0.0):
        return 1.0
    Z = _hier.linkage(condensed, method="average")
    coph = _hier.cophenet(Z)
    # Pearson correlation between condensed distance and cophenetic distance
    if np.std(condensed) == 0.0 or np.std(coph) == 0.0:
        return 1.0
    return float(np.corrcoef(condensed, coph)[0, 1])


def _consensus_nmf(
    data: np.ndarray,
    k_range: Sequence[int],
    n_runs: int = 30,
    seed: int = 10,
    method: str = "lee",
    L1: Sequence[float] = (0.0, 0.0),
    max_iter: int = 2000,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Estimate the optimal NMF rank by consensus + cophenetic correlation.

    For each ``k`` in ``k_range`` we run ``n_runs`` independent NMFs (each
    seeded by a distinct value), cluster each run by argmax over factors,
    accumulate the binary co-occurrence into a consensus matrix, and
    compute the cophenetic correlation of the consensus's hierarchical
    clustering. The recommended ``k_hat`` is the value preceding the
    largest drop in cophenetic correlation (Brunet et al., 2004).

    Parameters
    ----------
    data : numpy.ndarray
        ``(n_features, n_samples)`` non-negative matrix.
    k_range : sequence of int
        Candidate ranks (e.g. ``range(2, 11)``).
    n_runs : int, optional
        Number of NMF restarts per ``k`` (default 30, matching R's
        ``nmfEstimateRank`` default for the analysis pipeline).
    seed : int, optional
        Base seed; each run uses ``seed + run_idx``.
    method : {"lee", "l1"}, optional
        Underlying NMF flavour. ``"lee"`` calls :func:`_lee_nmf`, ``"l1"``
        calls :func:`_l1_nmf` with ``L1`` strength.
    L1 : sequence of float, optional
        L1 strengths (only used when ``method='l1'``).
    max_iter : int, optional
        Maximum iterations per NMF run.

    Returns
    -------
    k_hat : int
        Recommended rank.
    cophenetic_curve : numpy.ndarray
        Cophenetic correlation per ``k`` (same length as ``k_range``).
    silhouette_curve : numpy.ndarray
        Silhouette score of the consensus matrix's hierarchical
        clustering at the corresponding ``k`` (length-aligned).

    Notes
    -----
    Silhouette is computed against the predicted ``k`` clusters (cut tree
    at the candidate ``k``). When ``n_samples < k`` we report ``NaN``.
    """
    from sklearn.metrics import silhouette_score as _silh

    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2:
        raise ValueError("`data` must be a 2D matrix (features x samples).")
    n_samples = data_arr.shape[1]
    k_arr = np.asarray(list(k_range), dtype=int)
    if k_arr.size == 0:
        raise ValueError("`k_range` must contain at least one candidate.")

    coph = np.zeros(k_arr.size, dtype=float)
    silh = np.full(k_arr.size, np.nan, dtype=float)

    for ki, k in enumerate(k_arr):
        consensus = np.zeros((n_samples, n_samples), dtype=float)
        for r in range(n_runs):
            run_seed = int(seed) + r
            if method == "lee":
                _, H = _lee_nmf(data_arr, rank=int(k), seed=run_seed,
                                max_iter=max_iter)
            elif method == "l1":
                _, H, _ = _l1_nmf(data_arr, rank=int(k), L1=L1,
                                  seed=run_seed, max_iter=max_iter)
            else:
                raise ValueError(f"Unknown method: {method!r}")
            labels = _max_factor_clustering(H)
            consensus = consensus + _connectivity_matrix(labels)
        consensus = consensus / float(n_runs)
        coph[ki] = _cophenetic_correlation(consensus)

        # silhouette on the cluster cut at this k
        if n_samples > int(k) >= 2:
            dist = 1.0 - consensus
            np.fill_diagonal(dist, 0.0)
            dist = 0.5 * (dist + dist.T)
            condensed = _squareform(dist, checks=False)
            if np.any(condensed != 0.0):
                Z = _hier.linkage(condensed, method="average")
                labels_consensus = _hier.fcluster(Z, t=int(k), criterion="maxclust")
                if len(set(labels_consensus.tolist())) > 1:
                    silh[ki] = float(_silh(dist, labels_consensus, metric="precomputed"))

    # knee detection: largest drop from k[i] to k[i+1] => optimal at k[i]
    if k_arr.size == 1:
        k_hat = int(k_arr[0])
    else:
        diffs = coph[:-1] - coph[1:]
        if np.all(diffs <= 0.0):
            # no drop ever; pick the k with maximum cophenetic
            k_hat = int(k_arr[int(np.argmax(coph))])
        else:
            k_hat = int(k_arr[int(np.argmax(diffs))])

    return k_hat, coph, silh
