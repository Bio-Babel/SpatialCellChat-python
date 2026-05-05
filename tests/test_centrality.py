"""Unit tests for ``spatialcellchat._centrality.computeCentralityLocal``.

The five degree-style measures (``outdeg_unweighted``, ``indeg_unweighted``,
``outdeg``, ``indeg``, ``page_rank``) and the four spectral measures
(``hub``, ``authority``, ``eigen``, ``betweenness``) match R bit-for-bit.
The two flow-based measures (``flowbet``, ``info``) use networkx
substitutions per the slice5 mapping; we only check structural
properties (length, finiteness, non-negativity) for those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialcellchat._centrality import (
    CENTRALITY_NAMES_DEGREE,
    CENTRALITY_NAMES_FULL,
    computeCentralityLocal,
)

# R-side reference dumps bundled under tests/data/r_dumps/centrality/
# (~9 KB, 9 files). Produced once by ``r_harness/centrality_harness.R``
# (lives outside the package tree). To regenerate after intentional
# centrality changes, re-run the R harness and copy outputs here.
FIXTURE_DIR = Path(__file__).parent / "data" / "r_dumps" / "centrality"


# ---------------------------------------------------------------------------
# R-parity tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["1", "2", "3"])
def test_centrality_degree_only_matches_R(case: str) -> None:
    """The five degree-only measures match R within 1e-12."""
    net = np.loadtxt(str(FIXTURE_DIR / f"centr_net{case}.tsv"))
    py = computeCentralityLocal(net, degree_only=True)
    r = pd.read_csv(
        str(FIXTURE_DIR / f"centr{case}_degonly.tsv"), sep="\t", index_col=0
    )
    for i, nm in enumerate(CENTRALITY_NAMES_DEGREE):
        np.testing.assert_allclose(py[i], r.loc[nm].values, atol=1e-12)


@pytest.mark.parametrize("case", ["1", "2", "3"])
def test_centrality_spectral_match_R(case: str) -> None:
    """``hub``, ``authority``, ``eigen``, ``betweenness`` match R within 1e-12."""
    net = np.loadtxt(str(FIXTURE_DIR / f"centr_net{case}.tsv"))
    py = computeCentralityLocal(net, degree_only=False)
    r = pd.read_csv(
        str(FIXTURE_DIR / f"centr{case}_all.tsv"), sep="\t", index_col=0
    )
    for nm in ("hub", "authority", "eigen", "betweenness"):
        i = CENTRALITY_NAMES_FULL.index(nm)
        np.testing.assert_allclose(py[i], r.loc[nm].values, atol=1e-12)


@pytest.mark.parametrize("case", ["1", "2"])
def test_centrality_flow_measures_structural(case: str) -> None:
    """``flowbet`` and ``info`` are non-negative finite vectors of length N."""
    net = np.loadtxt(str(FIXTURE_DIR / f"centr_net{case}.tsv"))
    py = computeCentralityLocal(net, degree_only=False)
    for nm in ("flowbet", "info"):
        i = CENTRALITY_NAMES_FULL.index(nm)
        vals = py[i]
        assert vals.shape == (net.shape[0],)
        assert np.all(np.isfinite(vals))
        assert (vals >= -1e-12).all()


# ---------------------------------------------------------------------------
# Behaviour tests
# ---------------------------------------------------------------------------
def test_computeCentralityLocal_shape_degree_only() -> None:
    """``degree_only=True`` returns a 5 x N matrix."""
    rng = np.random.default_rng(0)
    net = rng.uniform(0, 1, (6, 6))
    net[net < 0.5] = 0
    np.fill_diagonal(net, 0)
    out = computeCentralityLocal(net, degree_only=True)
    assert out.shape == (5, 6)
    assert (out[0] == (net > 0).sum(axis=1)).all()
    assert (out[1] == (net > 0).sum(axis=0)).all()


def test_computeCentralityLocal_shape_full() -> None:
    """``degree_only=False`` returns an 11 x N matrix."""
    rng = np.random.default_rng(1)
    net = rng.uniform(0, 1, (5, 5))
    net[net < 0.5] = 0
    np.fill_diagonal(net, 0)
    out = computeCentralityLocal(net, degree_only=False)
    assert out.shape == (11, 5)


def test_computeCentralityLocal_dataframe_input() -> None:
    """DataFrame input is accepted."""
    rng = np.random.default_rng(2)
    arr = rng.uniform(0, 1, (4, 4))
    arr[arr < 0.5] = 0
    np.fill_diagonal(arr, 0)
    df = pd.DataFrame(arr, index=[f"a{i}" for i in range(4)], columns=[f"a{i}" for i in range(4)])
    out = computeCentralityLocal(df, degree_only=True)
    assert out.shape == (5, 4)


def test_computeCentralityLocal_non_square_raises() -> None:
    """Non-square input raises ValueError."""
    arr = np.zeros((3, 5))
    with pytest.raises(ValueError, match="square"):
        computeCentralityLocal(arr)


def test_centrality_disconnected_graph_returns_zero_flow() -> None:
    """Disconnected graphs zero out ``flowbet``/``info`` (mirrors R try-catch)."""
    # Two isolated single-edge components: 0-1 and 2-3
    net = np.zeros((4, 4))
    net[0, 1] = 1.0
    net[2, 3] = 1.0
    out = computeCentralityLocal(net, degree_only=False)
    flow_idx = CENTRALITY_NAMES_FULL.index("flowbet")
    info_idx = CENTRALITY_NAMES_FULL.index("info")
    np.testing.assert_array_equal(out[flow_idx], np.zeros(4))
    np.testing.assert_array_equal(out[info_idx], np.zeros(4))
