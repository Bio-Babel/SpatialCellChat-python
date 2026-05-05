"""Similarity Network Fusion (Wang et al. 2014).

Lightweight implementation of the SNF algorithm used by the
``clustering="fusion"`` branch of :func:`identifyTopicCommunities` (sub-agent
B's territory). Per the slice5 plan this module is a thin scaffold; the
tutorial path uses ``clustering="seurat"`` so a full implementation is
not required for the published tests. The placeholder keeps the public
API stable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["snf"]


def _normalize_kernel(W: np.ndarray) -> np.ndarray:
    """Normalise a similarity matrix as in Wang et al. 2014 §3.1."""
    W = np.asarray(W, dtype=float).copy()
    W[~np.isfinite(W)] = 0.0
    np.fill_diagonal(W, 0.0)
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P = W / (2.0 * row_sum)
    np.fill_diagonal(P, 0.5)
    return P


def _local_kernel(W: np.ndarray, k: int) -> np.ndarray:
    """Mask all but the top-``k`` neighbours of each row (R: ``buildSNN`` analog)."""
    W = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    out = np.zeros_like(W)
    if k <= 0:
        return out
    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        return out
    for i in range(n):
        idx = np.argsort(-W[i])[:k_eff]
        denom = W[i, idx].sum()
        if denom > 0:
            out[i, idx] = W[i, idx] / denom
        else:
            out[i, idx] = 0.0
    return out


def snf(
    W_list: Sequence[np.ndarray],
    k: int = 20,
    t: int = 20,
) -> np.ndarray:
    """Fuse a list of similarity matrices via SNF.

    Mirrors the algorithm of Wang et al. 2014 (a reference Python port
    of Carlucci/SNFpy). All input matrices must share the same shape.

    Parameters
    ----------
    W_list : sequence of numpy.ndarray
        Similarity matrices (n x n) to fuse.
    k : int, default 20
        Local-affinity neighbourhood size.
    t : int, default 20
        Number of fusion iterations.

    Returns
    -------
    numpy.ndarray
        The fused similarity matrix of shape (n, n).
    """
    if len(W_list) == 0:
        raise ValueError("`W_list` must contain at least one matrix.")
    Ps = [_normalize_kernel(W) for W in W_list]
    Ss = [_local_kernel(W, k=k) for W in W_list]
    L = len(Ps)
    if L == 1:
        return Ps[0]
    P_iter = [P.copy() for P in Ps]
    for _ in range(int(t)):
        new_P = []
        for v in range(L):
            other_avg = np.mean(
                [P_iter[u] for u in range(L) if u != v], axis=0
            )
            new_P.append(Ss[v] @ other_avg @ Ss[v].T)
        P_iter = [_normalize_kernel(P) for P in new_P]
    return np.mean(P_iter, axis=0)
