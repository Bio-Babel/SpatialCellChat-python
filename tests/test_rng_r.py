"""Bit-exact R 4.x parity tests for :mod:`spatialcellchat._rng_r`.

These tests pin the public surface of ``RMT`` (``unif_rand``,
``sample_int_no_replace``) against fixed R 4.3.x ``set.seed`` reference
values.  If pytest fails here after a code change, the migration broke
R parity — investigate before merging.

Reference values were generated with R 4.3.3 + ``RNGkind("Mersenne-
Twister", "Inversion", "Rejection")`` and copied verbatim from a live
``Rscript`` harness.  See ``_rng_r.RMT.__doc__`` for the algorithm
mirror.
"""
from __future__ import annotations

import math

import pytest

from spatialcellchat._rng_r import RMT, r_sample_int_no_replace


# ── Reference data: R 4.3.3 ``runif(3)`` for several seeds ───────────
# Captured with ``sprintf("%.17g", ...)`` so the literals round-trip
# through ``float(...)`` to the exact bit pattern R produced.
_RUNIF_REFERENCE = {
    1:     [0.26550866314209998, 0.37212389963679016, 0.57285336335189641],
    12345: [0.72090389626100659, 0.87577319308184087, 0.76098232832737267],
    42:    [0.91480604349635541, 0.93707541329786181, 0.28613953478634357],
    0:     [0.8966972001362592,  0.26550866314209998, 0.37212389963679016],
    7:     [0.98890929785557091, 0.39774545328691602, 0.11569777876138687],
}


# ── Reference data: R ``set.seed(s); sample(1:n, size, FALSE)`` ──────
_SAMPLE_REFERENCE = {
    # (seed, n, size): expected 1-based output
    (1, 55, 5):       [4, 39, 1, 34, 23],
    (1, 10, 10):      [9, 4, 7, 1, 2, 5, 3, 10, 6, 8],
    (1, 100, 20):     [68, 39, 1, 34, 87, 43, 14, 82, 59, 51,
                       85, 21, 54, 74, 7, 73, 79, 37, 83, 97],
    (12345, 55, 5):   [14, 51, 16, 26, 28],
    (12345, 10, 10):  [3, 8, 2, 5, 10, 9, 7, 6, 4, 1],
    (12345, 100, 20): [14, 51, 80, 90, 92, 24, 58, 93, 75, 88,
                       2, 86, 96, 38, 10, 81, 32, 40, 39, 87],
    (42, 55, 5):      [49, 37, 1, 25, 10],
    (42, 10, 10):     [1, 5, 10, 8, 2, 4, 6, 9, 7, 3],
    (42, 100, 20):    [49, 65, 25, 74, 18, 100, 47, 24, 71, 89,
                       37, 20, 26, 3, 41, 27, 36, 5, 34, 87],
    (0, 55, 5):       [14, 4, 39, 1, 34],
    (7, 55, 5):       [42, 19, 31, 28, 39],
}


@pytest.mark.parametrize("seed,expected", _RUNIF_REFERENCE.items())
def test_unif_rand_matches_R_bit_exact(seed: int, expected: list) -> None:
    """``RMT(seed).unif_rand()`` must match R's ``set.seed(seed); runif(3)``."""
    rmt = RMT(seed)
    got = [rmt.unif_rand() for _ in range(len(expected))]
    for g, r in zip(got, expected):
        # Equality up to double round-trip (the values originate as
        # ``y * (1/2^32)`` so there's no representation slack).
        assert g == pytest.approx(r, abs=0.0, rel=0.0), (
            f"seed={seed}: Py={g!r} R={r!r}"
        )


@pytest.mark.parametrize("key,expected", _SAMPLE_REFERENCE.items())
def test_sample_int_no_replace_matches_R(key, expected: list) -> None:
    """``RMT.sample_int_no_replace`` must mirror R's Rejection sampler."""
    seed, n, size = key
    got = RMT(seed).sample_int_no_replace(n, size)
    assert got == expected, (
        f"seed={seed} n={n} size={size}: Py={got} R={expected}"
    )


def test_top_level_helper_matches_class() -> None:
    """``r_sample_int_no_replace(seed, n, size)`` is a thin wrapper."""
    assert r_sample_int_no_replace(42, 55, 5) == [49, 37, 1, 25, 10]


def test_sample_full_population_is_permutation() -> None:
    """``sample(1:n, n, replace=FALSE)`` must yield a permutation of 1..n."""
    for n in (10, 50, 100):
        out = RMT(7).sample_int_no_replace(n, n)
        assert sorted(out) == list(range(1, n + 1))


def test_sample_size_zero_returns_empty() -> None:
    assert RMT(1).sample_int_no_replace(10, 0) == []


def test_sample_size_larger_than_n_raises() -> None:
    with pytest.raises(ValueError, match="cannot take a sample"):
        RMT(1).sample_int_no_replace(5, 6)


def test_sample_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RMT(1).sample_int_no_replace(-1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        RMT(1).sample_int_no_replace(5, -1)


def test_unif_rand_in_unit_interval() -> None:
    """``unif_rand`` outputs ∈ [0, 1) for arbitrary seeds."""
    for seed in (1, 12345, 99999):
        rmt = RMT(seed)
        for _ in range(1000):
            u = rmt.unif_rand()
            assert 0.0 <= u < 1.0


def test_independent_instances_dont_interfere() -> None:
    """Two RMT instances seeded identically produce the same stream."""
    a = RMT(2026)
    b = RMT(2026)
    for _ in range(100):
        assert a.unif_rand() == b.unif_rand()
