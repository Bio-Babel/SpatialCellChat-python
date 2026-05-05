"""Tests for the private ``_alra`` module — bit-for-bit compared to R."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialcellchat import _alra

R_DUMP = Path(__file__).parent / "data" / "r_dumps" / "alra"


@pytest.fixture(scope="module")
def alra_input() -> np.ndarray:
    """The exact (cells x genes) matrix R was given (12 x 30)."""
    return pd.read_csv(R_DUMP / "alra_input.csv", index_col=0).values


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


# ---------------------------------------------------------------------------
# choose_k
# ---------------------------------------------------------------------------
def test_choose_k_argument_validation() -> None:
    A = np.random.default_rng(0).normal(size=(15, 20))
    with pytest.raises(ValueError):
        _alra.choose_k(A, K=25)            # K too big for shape
    with pytest.raises(ValueError):
        _alra.choose_k(A, K=10, noise_start=8)  # too few noise svals


def test_choose_k_basic_shape() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(size=(40, 50))
    out = _alra.choose_k(A, K=20, noise_start=10, q=2, random_state=42)
    assert set(out.keys()) == {"k", "num_of_sds", "d"}
    assert out["d"].shape == (20,)
    assert out["num_of_sds"].shape == (19,)
    assert isinstance(out["k"], int)


# ---------------------------------------------------------------------------
# alra rank-k reconstruction matches R
# ---------------------------------------------------------------------------
def test_alra_matches_R(alra_input: np.ndarray) -> None:
    res = _alra.alra(alra_input, k=5, q=10, quantile_prob=1e-5, random_state=7)
    r_rk = pd.read_csv(R_DUMP / "alra_rank_k.csv", index_col=0).values
    r_cor = pd.read_csv(R_DUMP / "alra_rank_k_cor.csv", index_col=0).values
    r_sc = pd.read_csv(R_DUMP / "alra_rank_k_cor_sc.csv", index_col=0).values
    assert _diff(res["A_norm_rank_k"], r_rk) < 1e-10
    assert _diff(res["A_norm_rank_k_cor"], r_cor) < 1e-10
    assert _diff(res["A_norm_rank_k_cor_sc"], r_sc) < 1e-10


def test_alra_returns_keys(alra_input: np.ndarray) -> None:
    res = _alra.alra(alra_input, k=5, random_state=7)
    assert set(res.keys()) == {
        "A_norm_rank_k",
        "A_norm_rank_k_cor",
        "A_norm_rank_k_cor_sc",
    }
    for v in res.values():
        assert v.shape == alra_input.shape


def test_alra_type_check() -> None:
    with pytest.raises(TypeError):
        _alra.alra("not an array")  # type: ignore[arg-type]


def test_alra_originally_nonzero_preserved(alra_input: np.ndarray) -> None:
    """ALRA must never zero out an originally non-zero count.

    From the R source: ``A_norm_rank_k_cor_sc[originally_nonzero & == 0] <- A_norm[...]``.
    """
    res = _alra.alra(alra_input, k=5, random_state=7)
    nz = alra_input > 0
    assert (res["A_norm_rank_k_cor_sc"][nz] > 0).all()
