"""R-compatible Mersenne-Twister RNG.

Bit-exact mirror of R 4.x's ``set.seed(seed)`` initialization and
``unif_rand()`` / ``sample(n, size, replace=FALSE)`` algorithms.

References (R 4.x source):

* Seeding & MT_genrand : ``src/main/RNG.c`` (``RNG_Init`` / ``MT_genrand``)
* sample(no replace)   : ``src/main/random.c``
                         (``R_unif_index`` / ``SampleNoReplace``)

R uses the "Mersenne-Twister" generator with a Knuth-TAOCP-2002
"initial scrambling" of the user seed (50 iterations of
``s <- 69069*s + 1``), followed by 625 additional iterations that
populate the position counter and the 624-word state.  The position
counter is then overwritten to 624 (forcing block regeneration on the
first call).  Combined with the standard MT19937 recurrence and
tempering, the resulting ``unif_rand()`` stream is bit-identical to
R's.

For ``sample(n, size, replace=FALSE)`` we use R 3.6+'s default
"Rejection" sampler (the historical "Rounding" sampler had a known bias
for n > 2^25; R 3.6+ deprecated it as the default).  ``RNGkind(...,
sample.kind="Rejection")`` is the documented default.

Verified bit-exact against ``Rscript`` (R 4.x) for a range of seeds —
see ``tests/test_rng_r.py``.
"""
from __future__ import annotations

from typing import List

__all__ = ["RMT", "r_sample_int_no_replace"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000
_KNUTH = 69069  # Knuth-TAOCP-2002 multiplier (R uses this for the seed scramble)


class RMT:
    """Bit-exact R 4.x Mersenne-Twister.

    After ``RMT(seed)``, calls to :meth:`unif_rand` produce the same
    values as R's ``set.seed(seed); runif(...)`` for the same seed.

    The R 3.6+ default no-replace sampler ``sample(n, size,
    replace=FALSE)`` is exposed as :meth:`sample_int_no_replace` and
    returns 1-based indices to match R's semantics.

    Notes
    -----
    The state is the 624-word MT19937 vector plus a position counter
    ``mti``.  The counter starts at 624 (matching R's FixupSeeds), so
    the first call to :meth:`_next_uint32` triggers a regeneration
    block before returning the first tempered word.
    """

    __slots__ = ("mt", "mti")

    def __init__(self, seed: int) -> None:
        s = int(seed) & 0xFFFFFFFF
        # ── "initial scrambling" — 50 Knuth-TAOCP-2002 iterations ─────
        # (RNG.c::ScrambleSeed)
        for _ in range(50):
            s = (_KNUTH * s + 1) & 0xFFFFFFFF
        # ── Populate 625-word i_seed array ────────────────────────────
        # Slot 0 (the position counter) is overwritten to 624 by
        # FixupSeeds, so we step the recurrence once and discard.
        s = (_KNUTH * s + 1) & 0xFFFFFFFF  # i_seed[0] — discarded
        mt: List[int] = [0] * _N
        for j in range(_N):
            s = (_KNUTH * s + 1) & 0xFFFFFFFF
            mt[j] = s
        self.mt: List[int] = mt
        self.mti: int = _N  # FixupSeeds → forces regeneration on first call

    # ------------------------------------------------------------------
    # Internal MT19937 generation (mirrors R's MT_genrand).
    # ------------------------------------------------------------------
    def _generate_block(self) -> None:
        """Refill the 624-word state in place (RNG.c::MT_genrand block)."""
        mt = self.mt
        mag01 = (0, _MATRIX_A)
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + _M] ^ (y >> 1) ^ mag01[y & 1]
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ mag01[y & 1]
        y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
        mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ mag01[y & 1]
        self.mti = 0

    def _next_uint32(self) -> int:
        if self.mti >= _N:
            self._generate_block()
        y = self.mt[self.mti]
        self.mti += 1
        # Tempering (mirrors R's MT_genrand).
        y ^= (y >> 11)
        y ^= (y << 7) & _TEMPERING_MASK_B
        y ^= (y << 15) & _TEMPERING_MASK_C
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # Public R surface.
    # ------------------------------------------------------------------
    def unif_rand(self) -> float:
        """Mirror R's ``unif_rand()`` for kind="Mersenne-Twister".

        Returns a double in ``[0, 1)`` derived from a tempered uint32 by
        scaling with ``1 / 2**32``.  R applies a tiny ``fixup`` to
        clamp 0 / 1 boundary — we omit it because consumers that care
        always feed the value into ``floor(x * n)`` where the boundary
        is irrelevant (R's R_unif_index ignores fixup too).
        """
        return self._next_uint32() * (1.0 / 4294967296.0)

    def _rbits(self, bits: int) -> int:
        """Mirror R's ``rbits(bits)`` (random.c).

        Walks the MT stream in 16-bit chunks (one chunk per loop
        iteration where ``n <= bits``), accumulates them little-end-first
        as ``v = 65536*v + v1`` where ``v1 = floor(unif_rand()*65536)``,
        and returns ``v & ((1<<bits)-1)``.  For ``bits <= 16`` this is
        ``(u32 >> 16) & ((1<<bits)-1)`` — i.e. bits 16..(15+bits) of the
        tempered uint32, not the top bits.
        """
        v = 0
        n_acc = 0
        # ``for (int n = 0; n <= bits; n += 16)`` in C.
        while n_acc <= bits:
            v1 = self._next_uint32() >> 16  # floor(unif_rand()*65536)
            v = 65536 * v + v1
            n_acc += 16
        return v & ((1 << bits) - 1)

    def _r_unif_index(self, n: int) -> int:
        """Uniform integer in ``[0, n)`` via R 3.6+'s Rejection sampler.

        Mirrors ``R_unif_index(n)`` (random.c): pick ``bits = ceil(log2(n))``,
        draw ``rbits(bits)``, reject when ``>= n``.
        """
        if n < 2:
            return 0
        bits = (n - 1).bit_length()  # smallest b with 2**b >= n
        while True:
            v = self._rbits(bits)
            if v < n:
                return v

    def sample_int_no_replace(self, n: int, size: int) -> List[int]:
        """R's ``sample(n, size, replace=FALSE)`` — 1-based indices.

        Mirrors ``random.c::SampleNoReplace`` exactly: maintain a
        workspace ``x = [0..n-1]``, draw ``j ~ R_unif_index(cur)``,
        emit ``x[j] + 1``, then swap-and-shrink with ``x[cur-1]``.
        """
        if size < 0 or n < 0:
            raise ValueError("n and size must be non-negative")
        if size > n:
            raise ValueError(
                f"cannot take a sample of size {size} from a population of size {n}"
            )
        x = list(range(n))
        out = [0] * size
        cur = n
        for i in range(size):
            j = self._r_unif_index(cur)
            out[i] = x[j] + 1  # R 1-indexed
            cur -= 1
            x[j] = x[cur]
        return out


def r_sample_int_no_replace(seed: int, n: int, size: int) -> List[int]:
    """``set.seed(seed); sample(1:n, size, FALSE)`` in R, bit-exact."""
    return RMT(seed).sample_int_no_replace(n, size)
