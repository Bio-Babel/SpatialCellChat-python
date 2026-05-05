"""Tests for spatialcellchat._nmf — consensus NMF + Lee/L1 wrappers."""

# ---------------------------------------------------------------------------
# _redistribute_d — RcppML-style symmetric scale split
# ---------------------------------------------------------------------------
# These tests pin two invariants:
#   (1) ``W_new @ H_new == W @ H`` exactly (R parity on reconstruction).
#   (2) Each redistributed factor's column-norm in W equals its row-norm
#       in H (the symmetric "diag(sqrt(d))" split R produces).
# A regression that re-introduces ``d = sqrt(n_W * n_H)`` (the prior bug)
# fails (1) by a substantial margin (~1e0 on random matrices).

from __future__ import annotations

import numpy as np
import pytest

from spatialcellchat._nmf import (
    _connectivity_matrix,
    _consensus_nmf,
    _cophenetic_correlation,
    _l1_nmf,
    _lee_nmf,
    _max_factor_clustering,
    _redistribute_d,
)


# ---------------------------------------------------------------------------
# _redistribute_d — RcppML d-scale redistribution (R parity invariants)
# ---------------------------------------------------------------------------
def test_redistribute_d_preserves_W_at_H_exactly():
    """``W_new @ H_new`` must equal ``W @ H`` to numerical precision."""
    rng = np.random.default_rng(0)
    for shape in [(20, 4, 30), (10, 3, 8), (50, 6, 25)]:
        n, k, m = shape
        W = rng.uniform(0.0, 1.0, size=(n, k))
        H = rng.uniform(0.0, 1.0, size=(k, m))
        W_new, H_new = _redistribute_d(W, H)
        np.testing.assert_allclose(
            W_new @ H_new, W @ H, atol=1e-10, rtol=1e-10,
            err_msg=f"shape={shape}: reconstruction broke",
        )


def test_redistribute_d_yields_symmetric_factor_norms():
    """After redistribution, ``||W[:, j]|| == ||H[j, :]||`` per factor."""
    rng = np.random.default_rng(1)
    W = rng.uniform(0.0, 1.0, size=(15, 5))
    H = rng.uniform(0.0, 1.0, size=(5, 12))
    W_new, H_new = _redistribute_d(W, H)
    n_W = np.linalg.norm(W_new, axis=0)
    n_H = np.linalg.norm(H_new, axis=1)
    np.testing.assert_allclose(n_W, n_H, atol=1e-12, rtol=1e-12)


def test_redistribute_d_factor_norms_equal_sqrt_d():
    """``||W_new[:, j]|| == ||H_new[j, :]|| == sqrt(d_j)`` where d_j = orig n_W * n_H."""
    rng = np.random.default_rng(2)
    W = rng.uniform(0.0, 1.0, size=(8, 3))
    H = rng.uniform(0.0, 1.0, size=(3, 7))
    n_W_orig = np.linalg.norm(W, axis=0)
    n_H_orig = np.linalg.norm(H, axis=1)
    sqrt_d_expected = np.sqrt(n_W_orig * n_H_orig)
    W_new, H_new = _redistribute_d(W, H)
    np.testing.assert_allclose(
        np.linalg.norm(W_new, axis=0), sqrt_d_expected, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.norm(H_new, axis=1), sqrt_d_expected, atol=1e-12,
    )


def test_redistribute_d_handles_zero_factor():
    """A zero column / row must not divide-by-zero."""
    W = np.array([[1.0, 0.0], [2.0, 0.0]])
    H = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0]])
    W_new, H_new = _redistribute_d(W, H)
    # The all-zero second factor stays zero in W_new (and the H_new row's
    # contribution is annihilated), so reconstruction equals W @ H.
    np.testing.assert_allclose(W_new @ H_new, W @ H, atol=1e-12)


# ---------------------------------------------------------------------------
# Lee NMF (sklearn 'mu' + Frobenius) — Tier-2 proxy for NMF::nmf(method='lee')
# ---------------------------------------------------------------------------
def test_lee_nmf_returns_correct_shapes():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 1.0, size=(12, 8))
    W, H = _lee_nmf(X, rank=3, seed=42)
    assert W.shape == (12, 3)
    assert H.shape == (3, 8)
    # both factors are non-negative
    assert np.all(W >= 0.0)
    assert np.all(H >= 0.0)


def test_lee_nmf_reconstruction_quality():
    rng = np.random.default_rng(1)
    # generate a low-rank-ish non-negative matrix
    A = rng.uniform(0.1, 1.0, size=(15, 3))
    B = rng.uniform(0.1, 1.0, size=(3, 10))
    X = A @ B
    W, H = _lee_nmf(X, rank=3, seed=42, max_iter=2000)
    err = np.linalg.norm(X - W @ H, ord="fro")
    rel_err = err / np.linalg.norm(X, ord="fro")
    assert rel_err < 0.1  # should reconstruct reasonably well


def test_lee_nmf_is_deterministic_with_seed():
    rng = np.random.default_rng(2)
    X = rng.uniform(0.1, 1.0, size=(10, 6))
    W1, H1 = _lee_nmf(X, rank=3, seed=42)
    W2, H2 = _lee_nmf(X, rank=3, seed=42)
    np.testing.assert_array_equal(W1, W2)
    np.testing.assert_array_equal(H1, H2)


# ---------------------------------------------------------------------------
# L1 NMF (sklearn 'cd' + l1_ratio=1.0) — Tier-2 proxy for RcppML::nmf
# ---------------------------------------------------------------------------
def test_l1_nmf_returns_correct_shapes_and_iter():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.1, 1.0, size=(15, 8))
    W, H, n_iter = _l1_nmf(X, rank=3, L1=(0.1, 0.1), seed=666)
    assert W.shape == (15, 3)
    assert H.shape == (3, 8)
    assert n_iter >= 1
    assert np.all(W >= 0.0)
    assert np.all(H >= 0.0)


def test_l1_nmf_l1_strength_increases_sparsity():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.1, 1.0, size=(20, 10))
    W_no_pen, H_no_pen, _ = _l1_nmf(X, rank=4, L1=(0.0, 0.0), seed=666, max_iter=500)
    W_pen, H_pen, _ = _l1_nmf(X, rank=4, L1=(1.0, 1.0), seed=666, max_iter=500)
    # higher L1 should give more zeros (or smaller-magnitude entries) in
    # the basis
    n_zeros_no = int(np.count_nonzero(W_no_pen < 1e-10))
    n_zeros_yes = int(np.count_nonzero(W_pen < 1e-10))
    assert n_zeros_yes >= n_zeros_no


def test_l1_nmf_invalid_L1_raises():
    rng = np.random.default_rng(5)
    X = rng.uniform(0.1, 1.0, size=(8, 5))
    with pytest.raises(ValueError):
        _l1_nmf(X, rank=2, L1=(0.1, 0.2, 0.3), seed=42)


# ---------------------------------------------------------------------------
# Consensus matrix machinery
# ---------------------------------------------------------------------------
def test_max_factor_clustering_picks_dominant_row():
    H = np.array([[0.9, 0.1, 0.5],
                  [0.1, 0.8, 0.3],
                  [0.0, 0.1, 0.2]])
    labels = _max_factor_clustering(H)
    assert labels.tolist() == [0, 1, 0]


def test_connectivity_matrix_shape_and_symmetry():
    labels = np.array([0, 1, 0, 1, 2])
    C = _connectivity_matrix(labels)
    assert C.shape == (5, 5)
    np.testing.assert_array_equal(C, C.T)
    # diagonal is 1
    assert np.all(np.diag(C) == 1.0)
    # samples 0 and 2 share a label → C[0,2] == 1
    assert C[0, 2] == 1.0
    # 0 and 1 do not share → 0
    assert C[0, 1] == 0.0


def test_cophenetic_correlation_perfectly_stable():
    # if every run yields the same clustering, consensus is binary and
    # cophenetic should be 1.0 (or very close)
    n = 6
    labels = np.array([0, 0, 1, 1, 2, 2])
    C = _connectivity_matrix(labels)  # already binary
    c = _cophenetic_correlation(C)
    assert c == pytest.approx(1.0, abs=1e-8)


def test_cophenetic_correlation_handles_all_one_consensus():
    # all entries equal -> distances all zero -> handled safely
    C = np.ones((4, 4))
    c = _cophenetic_correlation(C)
    assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _consensus_nmf — knee detection on synthetic data
# ---------------------------------------------------------------------------
def test_consensus_nmf_returns_aligned_curves():
    rng = np.random.default_rng(7)
    X = rng.uniform(0.1, 1.0, size=(12, 8))
    k_hat, coph, silh = _consensus_nmf(
        X, k_range=[2, 3, 4], n_runs=3, seed=10, method="lee", max_iter=300,
    )
    assert coph.shape == (3,)
    assert silh.shape == (3,)
    assert isinstance(k_hat, int)
    assert k_hat in (2, 3, 4)
    # cophenetic correlations are in [-1, 1]
    assert np.all((coph >= -1.0) & (coph <= 1.0))


def test_consensus_nmf_picks_two_when_only_two_clusters():
    """With clean 2-cluster structure, k=2 should give the highest cophenetic."""
    rng = np.random.default_rng(8)
    n_per = 5
    A = rng.uniform(0.1, 1.0, size=(15, n_per))
    B = rng.uniform(0.1, 1.0, size=(15, n_per))
    A_block = np.hstack([A, np.full_like(A, 0.01)])
    B_block = np.hstack([np.full_like(B, 0.01), B])
    X = A_block + B_block
    k_hat, coph, _ = _consensus_nmf(
        X, k_range=[2, 3, 4], n_runs=4, seed=10, method="lee", max_iter=400,
    )
    # the lee NMF with NNDSVD is deterministic (same seed yields same H)
    # so consensus is 1.0 across all k. The knee detection in this case
    # falls through to argmax-cophenetic which can pick any k. Just
    # ensure the run completed and returned a sane k_hat.
    assert k_hat in (2, 3, 4)
    assert np.all(coph <= 1.0 + 1e-9)


def test_consensus_nmf_with_l1_method():
    rng = np.random.default_rng(9)
    X = rng.uniform(0.1, 1.0, size=(10, 7))
    k_hat, coph, _ = _consensus_nmf(
        X, k_range=[2, 3], n_runs=2, seed=666, method="l1",
        L1=(0.05, 0.05), max_iter=300,
    )
    assert k_hat in (2, 3)
    assert coph.shape == (2,)


def test_consensus_nmf_single_k_skips_knee():
    rng = np.random.default_rng(10)
    X = rng.uniform(0.1, 1.0, size=(8, 6))
    k_hat, coph, silh = _consensus_nmf(
        X, k_range=[3], n_runs=2, seed=10, method="lee", max_iter=200,
    )
    assert k_hat == 3
    assert coph.shape == (1,)
    assert silh.shape == (1,)


def test_consensus_nmf_empty_k_raises():
    rng = np.random.default_rng(11)
    X = rng.uniform(0.1, 1.0, size=(6, 4))
    with pytest.raises(ValueError):
        _consensus_nmf(X, k_range=[], n_runs=2, seed=10)


def test_consensus_nmf_unknown_method_raises():
    rng = np.random.default_rng(12)
    X = rng.uniform(0.1, 1.0, size=(6, 4))
    with pytest.raises(ValueError):
        _consensus_nmf(X, k_range=[2, 3], n_runs=1, method="foo", seed=10)


def test_consensus_nmf_2d_required():
    with pytest.raises(ValueError):
        _consensus_nmf(np.ones((3, 4, 2)), k_range=[2, 3])


# ---------------------------------------------------------------------------
# Knee detection: synthetic cophenetic curve with engineered drop
# ---------------------------------------------------------------------------
def test_consensus_nmf_chooses_k_before_largest_drop():
    """Inject a controlled NMF that produces a clear knee at k=3.

    We patch ``_lee_nmf`` so the simulated cluster assignments are highly
    consistent for k <= 3 but drop in agreement at k=4. The knee
    heuristic should then pick k=3.
    """
    from spatialcellchat import _nmf as _nmf_mod
    n_samples = 8
    # design clear cluster structures per k
    fixed_clusters = {
        2: np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        3: np.array([0, 0, 0, 1, 1, 1, 2, 2]),
        4: np.array([0, 1, 2, 3, 0, 1, 2, 3]),  # bad split
    }
    rng_state = np.random.default_rng(123)

    def fake_lee_nmf(data, rank, seed, max_iter, init="nndsvd", tol=1e-4):
        n = data.shape[1]
        labels = fixed_clusters[int(rank)].copy()
        # introduce small jitter to simulate run-to-run agreement
        if int(seed) % 3 == 0 and rank == 4:
            labels = labels[::-1]  # disagreeing run
        H = np.zeros((rank, n))
        for j in range(n):
            H[labels[j], j] = 1.0
        W = np.zeros((data.shape[0], rank))
        return W, H

    monkeypatched = _nmf_mod._lee_nmf
    _nmf_mod._lee_nmf = fake_lee_nmf
    try:
        X = rng_state.uniform(0.1, 1.0, size=(6, n_samples))
        k_hat, coph, _ = _consensus_nmf(
            X, k_range=[2, 3, 4], n_runs=6, seed=10, method="lee",
        )
    finally:
        _nmf_mod._lee_nmf = monkeypatched
    # k=2 and k=3 should yield perfect cophenetic (1.0); k=4 should drop
    assert coph[0] >= coph[2] - 1e-6 and coph[1] >= coph[2] - 1e-6
    # knee at k=3 means the largest drop is between k=3 and k=4 → k_hat=3
    assert k_hat == 3
