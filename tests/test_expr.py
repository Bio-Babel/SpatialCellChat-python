"""Tests for the eight expression helpers in :mod:`spatialcellchat._expr`.

The R-side golden values are not committed; they are reproduced inline via
constant expected arrays for the smallest, most edge-case-y inputs. A
larger numerical-equivalence harness lives in the development scratch
directory (outside the package tree) and is run separately during
development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialcellchat._expr import (
    _make_FunMean,
    computeAveExpr,
    computeExpr_LR,
    computeExpr_agonist,
    computeExpr_antagonist,
    computeExpr_complex,
    computeExpr_coreceptor,
    computeExprGroup_agonist,
    computeExprGroup_antagonist,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_data() -> pd.DataFrame:
    """A 6-gene x 6-cell deterministic data matrix for unit tests."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.1, 4.0, size=(6, 6))
    return pd.DataFrame(
        arr,
        index=pd.Index(["G1", "G2", "G3", "G4", "G5", "G6"]),
        columns=pd.Index([f"c{i}" for i in range(6)]),
    )


@pytest.fixture
def complex_input() -> pd.DataFrame:
    """A small complex DataFrame with two complexes (G1+G2, G3+G4+G5)."""
    return pd.DataFrame(
        {
            "subunit_1": ["G1", "G3"],
            "subunit_2": ["G2", "G4"],
            "subunit_3": ["", "G5"],
        },
        index=pd.Index(["CPX_AB", "CPX_CDE"]),
    )


@pytest.fixture
def cofactor_input() -> pd.DataFrame:
    """A small cofactor DataFrame with two cofactor entries."""
    return pd.DataFrame(
        {
            "cofactor1": ["G6", "G1"],
            "cofactor2": ["", "G2"],
        },
        index=pd.Index(["AGO_X", "ANT_Y"]),
    )


@pytest.fixture
def pairLRsig() -> pd.DataFrame:
    """Two L-R pairs with one having an agonist+co_A receptor, the other an
    antagonist+co_I receptor."""
    return pd.DataFrame(
        {
            "ligand": ["G1", "G3"],
            "receptor": ["G2", "G4"],
            "agonist": ["AGO_X", ""],
            "antagonist": ["", "ANT_Y"],
            "co_A_receptor": ["AGO_X", ""],
            "co_I_receptor": ["", "ANT_Y"],
        }
    )


# ---------------------------------------------------------------------------
# computeAveExpr
# ---------------------------------------------------------------------------
def test_computeAveExpr_triMean_groups(small_data: pd.DataFrame) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    out = computeAveExpr(small_data, group, type="triMean")
    assert out.shape == (6, 3)
    assert list(out.columns) == ["A", "B", "C"]


def test_computeAveExpr_truncatedMean_requires_trim(small_data: pd.DataFrame) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"])
    with pytest.raises(ValueError, match="invalid 'trim'"):
        computeAveExpr(small_data, group, type="truncatedMean", trim=None)


def test_computeAveExpr_median_basic(small_data: pd.DataFrame) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    out = computeAveExpr(small_data, group, type="median")
    # Compare against direct numpy computation
    arr = small_data.to_numpy()
    expected = np.column_stack([
        np.median(arr[:, [0, 1]], axis=1),
        np.median(arr[:, [2, 3]], axis=1),
        np.median(arr[:, [4, 5]], axis=1),
    ])
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-12)


def test_computeAveExpr_features_filter(small_data: pd.DataFrame) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    out = computeAveExpr(small_data, group, features=["G1", "G3", "G5"], type="median")
    assert list(out.index) == ["G1", "G3", "G5"]


def test_make_FunMean_invalid() -> None:
    with pytest.raises(ValueError):
        _make_FunMean("notatype", trim=None)


# ---------------------------------------------------------------------------
# computeExpr_complex
# ---------------------------------------------------------------------------
def test_computeExpr_complex_pair(small_data: pd.DataFrame, complex_input: pd.DataFrame) -> None:
    out = computeExpr_complex(complex_input, small_data, ["CPX_AB", "CPX_CDE"])
    assert out.shape == (2, 6)
    arr = small_data.to_numpy()
    # Geometric mean of subunits per cell
    expected_AB = np.sqrt(arr[0, :] * arr[1, :])
    expected_CDE = np.exp(np.mean(np.log(arr[2:5, :]), axis=0))
    np.testing.assert_allclose(out[0, :], expected_AB, atol=1e-12)
    np.testing.assert_allclose(out[1, :], expected_CDE, atol=1e-12)


def test_computeExpr_complex_missing_subunit_errors(small_data: pd.DataFrame) -> None:
    """When a subunit symbol is not in data_use rows, R errors with
    'subscript out of bounds'. Python raises KeyError."""
    bad_complex = pd.DataFrame(
        {"subunit_1": ["NOT_A_GENE"], "subunit_2": ["G1"]},
        index=pd.Index(["CPX_X"]),
    )
    with pytest.raises(KeyError, match="subunit"):
        computeExpr_complex(bad_complex, small_data, ["CPX_X"])


# ---------------------------------------------------------------------------
# computeExpr_LR
# ---------------------------------------------------------------------------
def test_computeExpr_LR_mix_single_complex(small_data: pd.DataFrame, complex_input: pd.DataFrame) -> None:
    geneLR = ["G1", "CPX_AB", "G6", "CPX_CDE"]
    out = computeExpr_LR(geneLR, small_data, complex_input)
    assert out.shape == (4, 6)
    # row 0 = G1 directly
    np.testing.assert_allclose(out[0, :], small_data.loc["G1"].to_numpy(), atol=1e-12)
    # row 2 = G6 directly
    np.testing.assert_allclose(out[2, :], small_data.loc["G6"].to_numpy(), atol=1e-12)
    # row 1 = sqrt(G1*G2)
    np.testing.assert_allclose(
        out[1, :], np.sqrt(small_data.loc["G1"] * small_data.loc["G2"]), atol=1e-12,
    )


# ---------------------------------------------------------------------------
# computeExpr_coreceptor
# ---------------------------------------------------------------------------
def test_computeExpr_coreceptor_A_basic(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    out = computeExpr_coreceptor(cofactor_input, small_data, pairLRsig, type="A")
    # row 0: AGO_X has cofactor1 = G6 only -> 1 + G6
    expected = 1.0 + small_data.loc["G6"].to_numpy()
    np.testing.assert_allclose(out[0, :], expected, atol=1e-12)
    # row 1: empty co_A -> all ones
    np.testing.assert_allclose(out[1, :], np.ones(6), atol=1e-12)


def test_computeExpr_coreceptor_I_multi(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    out = computeExpr_coreceptor(cofactor_input, small_data, pairLRsig, type="I")
    # row 0: empty co_I -> all ones
    np.testing.assert_allclose(out[0, :], np.ones(6), atol=1e-12)
    # row 1: ANT_Y has cofactor1=G1, cofactor2=G2 -> (1+G1)*(1+G2)
    expected = (1.0 + small_data.loc["G1"].to_numpy()) * (1.0 + small_data.loc["G2"].to_numpy())
    np.testing.assert_allclose(out[1, :], expected, atol=1e-12)


def test_computeExpr_coreceptor_invalid_type(
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
    small_data: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        computeExpr_coreceptor(cofactor_input, small_data, pairLRsig, type="X")


# ---------------------------------------------------------------------------
# computeExpr_agonist / computeExpr_antagonist
# ---------------------------------------------------------------------------
def test_computeExpr_agonist_single_cofactor(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    out = computeExpr_agonist(small_data, pairLRsig, cofactor_input,
                              index_agonist=0, Kh=0.5, n=1)
    # AGO_X has cofactor1=G6 only -> 1 + x/(0.5+x) where x = G6
    G6 = small_data.loc["G6"].to_numpy()
    expected = 1.0 + G6 / (0.5 + G6)
    np.testing.assert_allclose(out.ravel(), expected, atol=1e-12)


def test_computeExpr_agonist_no_cofactor(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    # Pair index 1 has agonist = ""
    out = computeExpr_agonist(small_data, pairLRsig, cofactor_input,
                              index_agonist=1, Kh=0.5, n=1)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out, np.ones((1, 6)), atol=1e-12)


def test_computeExpr_antagonist_multi_cofactor(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    # Pair index 1 has antagonist = ANT_Y (G1, G2)
    out = computeExpr_antagonist(small_data, pairLRsig, cofactor_input,
                                 index_antagonist=1, Kh=0.5, n=1)
    G1 = small_data.loc["G1"].to_numpy()
    G2 = small_data.loc["G2"].to_numpy()
    expected = (0.5 / (0.5 + G1)) * (0.5 / (0.5 + G2))
    np.testing.assert_allclose(out.ravel(), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# computeExprGroup_agonist / computeExprGroup_antagonist
# ---------------------------------------------------------------------------
def test_computeExprGroup_agonist_single(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    FunMean = _make_FunMean("truncatedMean", trim=0.0)
    out = computeExprGroup_agonist(small_data, pairLRsig, cofactor_input,
                                   group=group, index_agonist=0,
                                   Kh=0.5, FunMean=FunMean, n=1)
    assert out.shape == (1, 3)
    G6 = small_data.loc["G6"].to_numpy()
    avg_per_group = np.array([
        np.mean(G6[[0, 1]]),
        np.mean(G6[[2, 3]]),
        np.mean(G6[[4, 5]]),
    ])
    expected = 1.0 + avg_per_group / (0.5 + avg_per_group)
    np.testing.assert_allclose(out.ravel(), expected, atol=1e-12)


def test_computeExprGroup_antagonist_no_cofactor(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    group = pd.Categorical(["A", "A", "B", "B", "C", "C"], categories=["A", "B", "C"])
    FunMean = _make_FunMean("truncatedMean", trim=0.0)
    out = computeExprGroup_antagonist(small_data, pairLRsig, cofactor_input,
                                      group=group, index_antagonist=0,
                                      Kh=0.5, FunMean=FunMean, n=1)
    # index 0 has antagonist = "" -> all ones, shape (1, 3)
    np.testing.assert_allclose(out, np.ones((1, 3)), atol=1e-12)


def test_computeExprGroup_agonist_truncatedMean(
    small_data: pd.DataFrame,
    cofactor_input: pd.DataFrame,
    pairLRsig: pd.DataFrame,
) -> None:
    group = pd.Categorical(["A"] * 4 + ["B", "C"], categories=["A", "B", "C"])
    FunMean = _make_FunMean("truncatedMean", trim=0.1)
    out = computeExprGroup_agonist(small_data, pairLRsig, cofactor_input,
                                   group=group, index_agonist=0,
                                   Kh=0.5, FunMean=FunMean, n=1)
    assert out.shape == (1, 3)
    # All values must be in [1, 2] (Hill-modulated)
    assert np.all(out >= 1.0) and np.all(out <= 2.0)
