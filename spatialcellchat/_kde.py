"""2D weighted kernel-density estimate helper for ``computeCommunField``.

This module is a stub for slice 5 (sub-agent B's NMF/topics scope). The
full implementation lands in slice 6a (``computeCommunField``).

Per design contract §3.E, the production helper will use
``scipy.stats.gaussian_kde(weights=...)`` with Silverman bandwidth as a
Tier-2 deviation from R's ``ks::hpi`` plug-in bandwidth (the latter
typically differs by 10-20% in absolute scale).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import gaussian_kde

__all__ = ["weighted_kde_2d"]


def weighted_kde_2d(
    coords: np.ndarray,
    weights: np.ndarray,
    grid: Tuple[np.ndarray, np.ndarray] | None = None,
    bw_method: str | float = "silverman",
) -> np.ndarray:
    """Weighted 2-D KDE on a regular grid (Silverman bandwidth).

    Parameters
    ----------
    coords : numpy.ndarray
        ``(N, 2)`` coordinates.
    weights : numpy.ndarray
        Length ``N`` non-negative weights.
    grid : tuple of (numpy.ndarray, numpy.ndarray), optional
        ``(xs, ys)`` 1-D grid arrays. If ``None``, a 100x100 grid spanning
        the data bounding box is generated.
    bw_method : str or float, optional
        Passed through to :class:`scipy.stats.gaussian_kde`.

    Returns
    -------
    numpy.ndarray
        Density evaluated on the grid. Shape ``(len(ys), len(xs))``.
    """
    coords = np.asarray(coords, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have shape (N, 2).")
    if weights.shape != (coords.shape[0],):
        raise ValueError("`weights` must have shape (N,).")
    if grid is None:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        xs = np.linspace(xmin, xmax, 100)
        ys = np.linspace(ymin, ymax, 100)
    else:
        xs, ys = grid
    kde = gaussian_kde(coords.T, weights=weights, bw_method=bw_method)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel()])
    return kde(pts).reshape(XX.shape)
