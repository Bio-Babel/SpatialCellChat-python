"""Unit tests for ``spatialcellchat._snn.buildSNN``.

The SNN graph builder is a direct port of R's ``buildSNN`` (which
delegates to the ``ComputeSNN`` C++ routine). Reference outputs are
bundled under ``tests/data/r_dumps/snn/`` (~180 KB), produced once by
``r_harness/snn_harness.R`` (lives outside the package tree) and
compared bit-for-bit at runtime (atol 1e-12). To regenerate the
fixtures after intentional changes, re-run the R harness and copy
outputs into this directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from spatialcellchat._snn import buildSNN

FIXTURE_DIR = Path(__file__).parent / "data" / "r_dumps" / "snn"


# ---------------------------------------------------------------------------
# R-parity tests on prebuilt harness datasets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,input_file,kwargs",
    [
        ("default", "snn_input_default.tsv", dict(k=10, k_scale=10, prune_SNN=1.0 / 15.0)),
        ("k3", "snn_input_default.tsv", dict(k=3, k_scale=10, prune_SNN=1.0 / 15.0)),
        ("prune_high", "snn_input_default.tsv", dict(k=10, k_scale=10, prune_SNN=0.5)),
        ("size50", "snn_input_50.tsv", dict(k=5, k_scale=10, prune_SNN=1.0 / 15.0)),
        ("size15_k3", "snn_input_15.tsv", dict(k=3, k_scale=2, prune_SNN=1.0 / 15.0)),
    ],
)
def test_buildSNN_matches_R(name: str, input_file: str, kwargs: dict) -> None:
    """``buildSNN`` is bit-identical to R's ``ComputeSNN`` for 5 inputs."""
    data = np.loadtxt(str(FIXTURE_DIR / input_file))
    py_snn = buildSNN(data, **kwargs).toarray()
    r_snn = np.loadtxt(str(FIXTURE_DIR / f"snn_{name}.tsv"))
    np.testing.assert_allclose(py_snn, r_snn, atol=1e-12)


# ---------------------------------------------------------------------------
# Behaviour / edge-case tests
# ---------------------------------------------------------------------------
def test_buildSNN_shape_and_diagonal() -> None:
    """Output has shape (n, n) with ones on the diagonal."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 30))
    snn = buildSNN(data, k=5)
    assert snn.shape == (30, 30)
    np.testing.assert_array_equal(snn.diagonal(), np.ones(30))


def test_buildSNN_returns_sparse() -> None:
    """Return type is a scipy sparse matrix in CSR format."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((10, 20))
    out = buildSNN(data, k=5)
    assert sparse.issparse(out)
    assert out.format == "csr"


def test_buildSNN_dataframe_input() -> None:
    """DataFrame input is accepted (columns become the samples)."""
    import pandas as pd

    rng = np.random.default_rng(2)
    data = rng.standard_normal((15, 25))
    df = pd.DataFrame(data, columns=[f"c{i}" for i in range(25)])
    out = buildSNN(df, k=5)
    assert out.shape == (25, 25)


def test_buildSNN_k_too_large_raises() -> None:
    """When ``k > n_samples`` an error is raised, mirroring R."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((10, 5))
    with pytest.raises(ValueError, match="k cannot be greater"):
        buildSNN(data, k=10)


def test_buildSNN_prune_zeroes_below_threshold() -> None:
    """Entries below ``prune_SNN`` are zeroed out."""
    rng = np.random.default_rng(4)
    data = rng.standard_normal((10, 30))
    out = buildSNN(data, k=5, prune_SNN=0.5).toarray()
    # Off-diagonal nonzeros must all be >= 0.5
    off = out.copy()
    np.fill_diagonal(off, 0.0)
    nonzero_vals = off[off > 0]
    assert (nonzero_vals >= 0.5 - 1e-12).all()


def test_buildSNN_symmetric() -> None:
    """SNN is symmetric (since it is `S @ S.T`)."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal((15, 30))
    out = buildSNN(data, k=5).toarray()
    np.testing.assert_allclose(out, out.T, atol=1e-12)
