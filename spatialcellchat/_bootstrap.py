"""Permutation helpers for the cell-cell communication model.

Houses the cell-sampling permutation used by :func:`filterProbability` (R:
``modeling.R:1976``). The R reference uses ``base::sample(x = 1:nC, size =
nboot, replace = F)`` to draw, for each L-R pair, a sample of ``nboot``
cell indices without replacement. We faithfully reproduce that semantics
through :class:`numpy.random.Generator.choice` with ``replace=False``.

Bit-exact parity with R's Mersenne-Twister stream is **not** possible from
pure NumPy. Tests therefore inject pre-generated ``permutations`` arrays
(dumped from R) when bit-for-bit comparison is required.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "make_cell_permutation",
]


def make_cell_permutation(
    nC: int,
    nLR: int,
    nboot: int,
    seed_use: int,
    permutations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Replicate R's ``replicate(nLR, base::sample(x = 1:nC, size = nboot, replace = F))``.

    Mirrors the per-LR permutation matrix built at the top of
    :func:`filterProbability` (R: ``modeling.R:2014``)::

        set.seed(seed.use)
        permutation <- replicate(nLR, base::sample(x = 1:nC, size = nboot,
                                                   replace = F))

    The result is a ``(nboot, nLR)`` array of **0-based** cell indices
    (R returns 1-based indices, but the Python caller indexes a NumPy
    array directly so 0-based is more convenient).

    Parameters
    ----------
    nC : int
        Number of cells (R's ``nC``).
    nLR : int
        Number of ligand-receptor pairs.
    nboot : int
        Number of bootstrap draws per LR pair.
    seed_use : int
        RNG seed (R's ``seed.use``).
    permutations : numpy.ndarray, optional
        If supplied, used directly (must have shape ``(nboot, nLR)``);
        useful for bit-for-bit parity tests against R-dumped indices.
        The array may be 1-based (R-style) or 0-based — we infer
        from the maximum value: if ``permutations.max() == nC`` we
        subtract one.

    Returns
    -------
    numpy.ndarray
        ``(nboot, nLR)`` integer array of 0-based cell indices.
    """
    if permutations is not None:
        perm = np.asarray(permutations, dtype=int)
        if perm.shape != (nboot, nLR):
            raise ValueError(
                f"permutations shape {perm.shape} does not match (nboot={nboot}, nLR={nLR})"
            )
        # Convert from 1-based if needed
        if perm.max() == nC:
            perm = perm - 1
        return perm

    rng = np.random.default_rng(int(seed_use))
    out = np.empty((nboot, nLR), dtype=int)
    for j in range(nLR):
        out[:, j] = rng.choice(nC, size=nboot, replace=False)
    return out
