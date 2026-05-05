"""Tests for spatialcellchat.spatial — spatial-geometry exports.

Each test compares Python output to ground-truth dumped from the R
``SpatialCellChat`` package via ``Rscript``. Frozen CSV fixtures are
bundled under ``tests/data/r_dumps/spatial/`` and the embedded expected
values inline below are the canonical fixture set.

Coverage matrix
---------------
- ``computeCellDistance``: typical-15-spot, ratio-set, tight-contact-tol,
  large-interaction-range, two-spot edge.
- ``createCellCellContactMatrixFrom_dspatial``: direct call with a known
  sparse distance matrix.
- ``computeRegionDistance``: typical, with-spot-size, single-cell-cluster,
  trim=0.25, two-cluster-small.
- ``computeColocalization``: 30-cell with default args & R permutations,
  symmetric=False, idents.use, 4-group asymmetric, small-3 case.
- ``computeGridSize``: with default cellsize, with explicit cellsize and
  ratio, do_plot=True summary.
- ``makeGridSpatialCellChat``: per-grid mean reduction matches a hand
  computed reference; cell.type majority vote matches; new coordinates
  equal per-grid spot centroids.
"""

from __future__ import annotations

import io
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pathlib import Path

from spatialcellchat.spatial import (
    computeCellDistance,
    computeColocalization,
    computeGridSize,
    computeRegionDistance,
    createCellCellContactMatrixFrom_dspatial,
    makeGridSpatialCellChat,
)

SPATIAL_DIR = Path(__file__).parent / "data" / "r_dumps" / "spatial"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# The 15-spot coordinates used in tests 1-4 of the R harness.
_COORDS_15 = np.array(
    [
        [0, 0], [100, 0], [100, 100], [200, 100], [300, 100],
        [250, 200], [250, 250], [50, 175], [75, 250], [150, 300],
        [175, 350], [350, 50], [400, 100], [450, 200], [500, 250],
    ],
    dtype=float,
)


def _coo_dict(M: sp.spmatrix) -> Dict[Tuple[int, int], float]:
    """Convert a sparse matrix to {(row, col): value} for set-comparison."""
    coo = M.tocoo()
    return {(int(i), int(j)): float(v) for i, j, v in zip(coo.row, coo.col, coo.data)}


# ---------------------------------------------------------------------------
# computeCellDistance — R-parity tests
# ---------------------------------------------------------------------------
# R harness: ``01_computeCellDistance.R``, test 1
# coords = _COORDS_15, ratio=1.0, interaction.range=250, contact.range=10, tol=5
# d.spatial has 128 non-zero off-diagonal entries.
def _expected_test1_dspatial() -> Dict[Tuple[int, int], float]:
    csv_text = """i,j,x
1,0,100
2,0,141.42135623730951
3,0,223.60679774997896
7,0,182.0027472320130
0,1,100
2,1,100
3,1,141.42135623730951
4,1,223.60679774997896
5,1,250
7,1,182.0027472320130
8,1,251.24689052802207
11,1,254.95097567963924
0,2,141.42135623730951
1,2,100
3,2,100
4,2,200
5,2,180.27756377319946
6,2,212.13203435596427
7,2,90.13878188659973
8,2,152.0690632574555
9,2,206.15528128088303
11,2,254.95097567963924
0,3,223.60679774997896
1,3,141.42135623730951
2,3,100
4,3,100
5,3,111.80339887498949
6,3,158.11388300841898
7,3,167.70509831248423
8,3,195.25624189766635
9,3,206.15528128088303
10,3,251.24689052802207
11,3,158.11388300841898
12,3,200
1,4,223.60679774997896
2,4,200
3,4,100
5,4,111.80339887498949
6,4,158.11388300841898
9,4,250
11,4,70.71067811865476
12,4,100
13,4,180.27756377319946
14,4,250
1,5,250
2,5,180.27756377319946
3,5,111.80339887498949
4,5,111.80339887498949
6,5,50
7,5,201.55644370746374
8,5,182.0027472320130
9,5,141.42135623730951
10,5,167.70509831248423
11,5,180.27756377319946
12,5,180.27756377319946
13,5,200
14,5,254.95097567963924
2,6,212.13203435596427
3,6,158.11388300841898
4,6,158.11388300841898
5,6,50
7,6,213.6000936329383
8,6,175
9,6,111.80339887498949
10,6,125
11,6,223.60679774997896
12,6,212.13203435596427
13,6,206.15528128088303
14,6,250
0,7,182.0027472320130
1,7,182.0027472320130
2,7,90.13878188659973
3,7,167.70509831248423
5,7,201.55644370746374
6,7,213.6000936329383
8,7,79.05694150420948
9,7,160.07810593582122
10,7,215.05813167606572
1,8,251.24689052802207
2,8,152.0690632574555
3,8,195.25624189766635
5,8,182.0027472320130
6,8,175
7,8,79.05694150420948
9,8,90.13878188659973
10,8,141.42135623730951
2,9,206.15528128088303
3,9,206.15528128088303
4,9,250
5,9,141.42135623730951
6,9,111.80339887498949
7,9,160.07810593582122
8,9,90.13878188659973
10,9,55.9016994374947
3,10,251.24689052802207
5,10,167.70509831248423
6,10,125
7,10,215.05813167606572
8,10,141.42135623730951
9,10,55.9016994374947
1,11,254.95097567963924
2,11,254.95097567963924
3,11,158.11388300841898
4,11,70.71067811865476
5,11,180.27756377319946
6,11,223.60679774997896
12,11,70.71067811865476
13,11,180.27756377319946
14,11,250
3,12,200
4,12,100
5,12,180.27756377319946
6,12,212.13203435596427
11,12,70.71067811865476
13,12,111.80339887498949
14,12,180.27756377319946
4,13,180.27756377319946
5,13,200
6,13,206.15528128088303
11,13,180.27756377319946
12,13,111.80339887498949
14,13,70.71067811865476
4,14,250
5,14,254.95097567963924
6,14,250
11,14,250
12,14,180.27756377319946
13,14,70.71067811865476
"""
    df = pd.read_csv(io.StringIO(csv_text))
    return {(int(r), int(c)): float(v) for r, c, v in zip(df["i"], df["j"], df["x"])}


def test_computeCellDistance_typical():
    """Test 1: 15 spots, default args (ratio=1.0)."""
    res = computeCellDistance(
        _COORDS_15, interaction_range=250, contact_range=10, ratio=1.0, tol=5
    )
    d = res["d.spatial"]
    a = res["adj.contact"]
    expected = _expected_test1_dspatial()
    actual = _coo_dict(d)
    assert actual.keys() == expected.keys(), f"Mismatch in non-zero pattern: missing {expected.keys() - actual.keys()}, extra {actual.keys() - expected.keys()}"
    for key, val in expected.items():
        np.testing.assert_allclose(actual[key], val, rtol=1e-9, atol=1e-9)
    # adj.contact has only the diagonal (no off-diagonal pair within 10+5=15um)
    assert d.shape == (15, 15)
    assert a.shape == (15, 15)
    a_dict = _coo_dict(a)
    expected_diag = {(k, k): 1.0 for k in range(15)}
    assert a_dict == expected_diag


def test_computeCellDistance_with_ratio():
    """Test 2: small interaction.range with ratio=0.18, contact.range=50, tol=10."""
    res = computeCellDistance(
        _COORDS_15, interaction_range=100, contact_range=50, ratio=0.18, tol=10
    )
    d = res["d.spatial"]
    a = res["adj.contact"]
    # Threshold = (100+10)/0.18 = 611.11... => all 15*14 = 210 pairs included
    coo_d = d.tocoo()
    n_offdiag = coo_d.nnz
    assert n_offdiag == 15 * 14
    # Distances are scaled by ratio = 0.18
    # Cell 0->1 = 100 * 0.18 = 18
    d_dense = d.toarray()
    assert pytest.approx(d_dense[1, 0], abs=1e-9) == 100 * 0.18
    assert pytest.approx(d_dense[0, 1], abs=1e-9) == 100 * 0.18
    # adj.contact: contact threshold = 50+10 = 60; how many 18s? 18 <= 60.
    # Diagonal=1 + every 0,1,2 distances => more than just diagonal.
    assert a.shape == (15, 15)
    a_dict = _coo_dict(a)
    # Diagonal entries are present
    for k in range(15):
        assert (k, k) in a_dict and a_dict[(k, k)] == 1.0


def test_computeCellDistance_tight_tol():
    """Test 3: tol=0.1 => contact threshold = 1+0.1 = 1.1; only diagonal in adj."""
    res = computeCellDistance(
        _COORDS_15, interaction_range=250, contact_range=1, ratio=1.0, tol=0.1
    )
    a = res["adj.contact"]
    # All distances > 1.1 => only diagonal in contact
    a_dict = _coo_dict(a)
    expected_diag = {(k, k): 1.0 for k in range(15)}
    assert a_dict == expected_diag


def test_computeCellDistance_large_interaction():
    """Test 4: huge interaction.range => every pair kept."""
    res = computeCellDistance(
        _COORDS_15, interaction_range=1e6, contact_range=50, ratio=1.0, tol=5
    )
    d = res["d.spatial"]
    n_offdiag = d.nnz
    assert n_offdiag == 15 * 14, f"Expected 210 off-diagonal entries, got {n_offdiag}"


def test_computeCellDistance_two_spots():
    """Test 5: 2-spot edge case, contact.range covers the pair."""
    coords = np.array([[0.0, 0.0], [100.0, 100.0]])
    res = computeCellDistance(
        coords, interaction_range=250, contact_range=100, ratio=1.0, tol=10
    )
    d = res["d.spatial"]
    a = res["adj.contact"]
    expected_dist = np.sqrt(20000)
    d_dense = d.toarray()
    np.testing.assert_allclose(d_dense[0, 1], expected_dist, rtol=1e-9)
    np.testing.assert_allclose(d_dense[1, 0], expected_dist, rtol=1e-9)
    # Contact threshold = 100+10 = 110; sqrt(20000)~=141.42 > 110 => only diagonal
    a_dict = _coo_dict(a)
    assert a_dict == {(0, 0): 1.0, (1, 1): 1.0}


def test_computeCellDistance_anndata(tiny_visium):
    """When given an AnnData, computeCellDistance writes obsp[...]."""
    adata = tiny_visium.copy()
    res = computeCellDistance(adata, interaction_range=300, contact_range=50, ratio=1.0, tol=10)
    assert "spatial_distances" in adata.obsp
    assert "spatial_connectivities" in adata.obsp
    # obsp shape == (n_obs, n_obs)
    assert adata.obsp["spatial_distances"].shape == (15, 15)
    assert adata.obsp["spatial_connectivities"].shape == (15, 15)
    # CellChat-spatial uns block was populated
    assert "cellchat" in adata.uns
    assert "spatial" in adata.uns["cellchat"]
    assert "computeCellDistance" in adata.uns["cellchat"]["spatial"]
    # The squidpy connectivities should have a unit diagonal added by us
    a = adata.obsp["spatial_connectivities"]
    diag = a.diagonal()
    np.testing.assert_array_equal(diag, np.ones(15))


# ---------------------------------------------------------------------------
# createCellCellContactMatrixFrom_dspatial
# ---------------------------------------------------------------------------
def test_createCellCellContactMatrixFrom_dspatial_direct():
    """R harness: build a known sparse distance matrix and apply the contact filter.

    R input triplets (1-based -> 0-based here):
        (0,1)=50, (0,2)=200, (1,0)=50, (1,2)=100, (2,0)=200, (2,1)=100, (3,4)=30
    contact.threshold=60, tol=5 => keep entries with distance <= 65; then add
    diag of 1 (only positions (0,0)..(3,3) are valid since the matrix is 4x5).
    Expected non-zero pattern:
        (1,0), (0,1), (3,4) from the threshold; (0,0), (1,1), (2,2), (3,3)
        from the diagonal write.
    """
    rows = np.array([0, 0, 1, 1, 2, 2, 3])
    cols = np.array([1, 2, 0, 2, 0, 1, 4])
    vals = np.array([50, 200, 50, 100, 200, 100, 30], dtype=float)
    d = sp.csc_matrix((vals, (rows, cols)), shape=(4, 5))
    a = createCellCellContactMatrixFrom_dspatial(d, contact_threshold=60, tol=5)
    # Expected entries
    a_dict = _coo_dict(a)
    expected = {
        (0, 0): 1.0, (1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0,  # diagonal
        (1, 0): 1.0, (0, 1): 1.0, (3, 4): 1.0,
    }
    assert a_dict == expected


# ---------------------------------------------------------------------------
# computeRegionDistance — R-parity tests
# ---------------------------------------------------------------------------
def test_computeRegionDistance_typical():
    """R: 4-cluster, default args (trim=0.1)."""
    group = pd.Categorical(
        ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D"]
    )
    res = computeRegionDistance(_COORDS_15, group)
    expected = pd.DataFrame(
        [
            [0.0, 164.05589209736, 164.362061368567, 312.178805834951],
            [164.05589209736, 0.0, 156.304793826532, 152.030706009126],
            [164.362061368567, 156.304793826532, 0.0, 323.513989937997],
            [312.178805834951, 152.030706009126, 323.513989937997, 0.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    pd.testing.assert_frame_equal(res, expected, atol=1e-9, rtol=1e-9)


def test_computeRegionDistance_with_spot_size():
    """R test 2: spot.size=65, spot.size.fullres=100 => some entries become NaN."""
    group = pd.Categorical(
        ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D"]
    )
    res = computeRegionDistance(
        _COORDS_15, group, trim=0.0, interaction_length=100,
        spot_size=65, spot_size_fullres=100, k_min=1,
    )
    expected = pd.DataFrame(
        [
            [0.0, 106.636329863284, 106.835339889569, np.nan],
            [106.636329863284, 0.0, 101.598115987246, 98.8199589059321],
            [106.835339889569, 101.598115987246, 0.0, np.nan],
            [np.nan, 98.8199589059321, np.nan, 0.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    pd.testing.assert_frame_equal(res, expected, atol=1e-9, rtol=1e-9)


def test_computeRegionDistance_trim():
    """R test 4: trim=0.25 changes the truncated mean (golden values from R)."""
    group = pd.Categorical(
        ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "D", "D", "D", "D"]
    )
    res = computeRegionDistance(_COORDS_15, group, trim=0.25)
    expected = pd.DataFrame(
        [
            [0.0, 172.574083274515, 165.246798859689, 309.910597274765],
            [172.574083274515, 0.0, 150.457888215569, 154.667252638704],
            [165.246798859689, 150.457888215569, 0.0, 320.385047440031],
            [309.910597274765, 154.667252638704, 320.385047440031, 0.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    pd.testing.assert_frame_equal(res, expected, atol=1e-9, rtol=1e-9)


def test_computeRegionDistance_single_cell_cluster():
    """R test 3: factor with 7 cells in 'C' (one of the smaller categories)."""
    group = pd.Categorical(
        ["A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "C", "C", "C", "C"],
        categories=["A", "B", "C"],
    )
    res = computeRegionDistance(_COORDS_15, group)
    expected = pd.DataFrame(
        [
            [0.0, 164.05589209736, 204.161178134932],
            [164.05589209736, 0.0, 134.03721072238],
            [204.161178134932, 134.03721072238, 0.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    pd.testing.assert_frame_equal(res, expected, atol=1e-9, rtol=1e-9)


def test_computeRegionDistance_two_cluster():
    """R test 5: 2-cluster small case (golden values from R)."""
    coords = np.array(
        [[0, 0], [1, 1], [2, 2], [10, 10], [11, 11], [12, 12]], dtype=float
    )
    group = pd.Categorical(["A", "A", "A", "B", "B", "B"])
    res = computeRegionDistance(coords, group)
    expected = pd.DataFrame(
        [[0.0, 12.7279220613579], [12.7279220613579, 0.0]],
        index=["A", "B"], columns=["A", "B"],
    )
    pd.testing.assert_frame_equal(res, expected, atol=1e-9, rtol=1e-9)


def test_computeRegionDistance_anndata(tiny_visium):
    """AnnData round-trip: result is also written to uns."""
    adata = tiny_visium.copy()
    group = adata.obs["active_ident"]
    res = computeRegionDistance(adata, group)
    assert "cellchat" in adata.uns
    assert "computeRegionDistance" in adata.uns["cellchat"]["spatial"]
    pd.testing.assert_frame_equal(res, adata.uns["cellchat"]["spatial"]["computeRegionDistance"])


# ---------------------------------------------------------------------------
# computeColocalization — R-parity tests
# ---------------------------------------------------------------------------
def _load_perms(tag: str, i: int, j: int) -> np.ndarray:
    """Load R-dumped permutation matrix from the bundled spatial fixtures."""
    fp = SPATIAL_DIR / "coloc_perms" / f"perm_{tag}_i{i}_j{j}.csv"
    return pd.read_csv(fp).to_numpy()


def _coords_30() -> np.ndarray:
    """Reload R test1 coords from the bundled harness CSV."""
    df = pd.read_csv(SPATIAL_DIR / "coloc_coords1.csv")
    return df[["x_cent", "y_cent"]].to_numpy()


def _build_perm_dict(tag: str, n_idents: int) -> Dict[Tuple[int, int], np.ndarray]:
    return {
        (i, j): _load_perms(tag, i, j)
        for i in range(1, n_idents + 1)
        for j in range(1, n_idents + 1)
    }


def test_computeColocalization_typical():
    """R test1: 30-cell, 3-group, nboot=50, seed=1, with R-dumped permutations."""
    coords = _coords_30()
    group = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    perms = _build_perm_dict("test1", 3)
    res = computeColocalization(coords, group, nboot=50, seed_use=1, permutations=perms)
    expected = pd.DataFrame(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    pd.testing.assert_frame_equal(res, expected)


def test_computeColocalization_asymmetric():
    """R test2: symmetric=False."""
    coords = _coords_30()
    group = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    perms = _build_perm_dict("test2", 3)
    res = computeColocalization(
        coords, group, symmetric=False, nboot=30, seed_use=1, permutations=perms,
    )
    # Verify that the matrix is not necessarily symmetric (it might be, but we
    # at least check that it matches what the R harness produced).
    df = pd.read_csv(SPATIAL_DIR / "coloc_test2.csv", index_col=0)
    pd.testing.assert_frame_equal(res, df, check_dtype=False, atol=1e-9, rtol=1e-9)


def test_computeColocalization_idents_use():
    """R test3: idents.use restricts to {A, B}."""
    coords = _coords_30()
    group = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    perms = _build_perm_dict("test3", 2)
    res = computeColocalization(
        coords, group, idents_use=["A", "B"], nboot=20, seed_use=5,
        permutations=perms,
    )
    df = pd.read_csv(SPATIAL_DIR / "coloc_test3.csv", index_col=0)
    pd.testing.assert_frame_equal(res, df, check_dtype=False, atol=1e-9, rtol=1e-9)


def test_computeColocalization_4group():
    """R test4: 4-group with corner clusters."""
    coords = pd.read_csv(SPATIAL_DIR / "coloc_coords4.csv")[["x_cent", "y_cent"]].to_numpy()
    group = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    perms = _build_perm_dict("test4", 4)
    res = computeColocalization(coords, group, nboot=25, seed_use=1, permutations=perms)
    df = pd.read_csv(SPATIAL_DIR / "coloc_test4.csv", index_col=0)
    pd.testing.assert_frame_equal(res, df, check_dtype=False, atol=1e-9, rtol=1e-9)


def test_computeColocalization_anndata():
    """AnnData round-trip: result is also written to uns.

    Build a small AnnData with 24 spots and 4 idents so ``data_nont`` always has
    enough rows for the permutation sample (R errors when nrow(data_nont) <
    nrow(data_label_1)).
    """
    coords = pd.read_csv(SPATIAL_DIR / "coloc_coords4.csv")[["x_cent", "y_cent"]].to_numpy()
    n = coords.shape[0]
    X = sp.csr_matrix(np.ones((n, 3)))
    obs = pd.DataFrame(index=[f"c{i:03d}" for i in range(n)])
    obs["active_ident"] = pd.Categorical(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 10)
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    res = computeColocalization(adata, adata.obs["active_ident"], nboot=5, seed_use=1)
    assert "computeColocalization" in adata.uns["cellchat"]["spatial"]
    pd.testing.assert_frame_equal(res, adata.uns["cellchat"]["spatial"]["computeColocalization"])


# ---------------------------------------------------------------------------
# computeGridSize
# ---------------------------------------------------------------------------
def test_computeGridSize_default():
    """R: with no cellsize provided, picks min cell-cell distance."""
    coords = np.array(
        [[0, 0], [25, 25], [50, 50], [75, 75], [100, 100]], dtype=float
    )
    res = computeGridSize(coords, do_plot=False)
    # min distance = sqrt(625+625) = sqrt(1250) ~ 35.355
    np.testing.assert_allclose(res["cellsize"], np.sqrt(1250.0), rtol=1e-9)
    # default grid_resolution=2 => newcellsize = 2*sqrt(1250) ~ 70.71
    np.testing.assert_allclose(res["newcellsize"], 2 * np.sqrt(1250.0), rtol=1e-9)


def test_computeGridSize_explicit():
    """R: explicit cellsize, ratio passed via spatial_factors."""
    res = computeGridSize(
        _COORDS_15, cellsize=100.0, grid_resolution=1.5,
        spatial_factors={"ratio": 0.5}, do_plot=False,
    )
    assert res["cellsize"] == 100.0
    assert res["newcellsize"] == 150.0
    # spot.size = newcellsize * ratio = 150 * 0.5 = 75.0
    np.testing.assert_allclose(res["spot.size"], 75.0, rtol=1e-9)


def test_computeGridSize_with_plot_summary():
    """R: do_plot=True populates the grid summary fields."""
    res = computeGridSize(
        _COORDS_15, cellsize=100.0, grid_resolution=1.0, do_plot=True,
    )
    # 15-spot data, cellsize=100 => the harness reported 20 cells, 13 assigned
    assert res["n_grids"] == 20
    assert res["n_assigned_grids"] >= 1
    assert isinstance(res["all_spots_assigned"], bool)


def test_computeGridSize_anndata(tiny_visium):
    """AnnData round-trip: writes uns['cellchat']['spatial']['computeGridSize']."""
    adata = tiny_visium.copy()
    adata.uns["cellchat"] = {"spatial.factors": {"ratio": 0.18}}
    res = computeGridSize(adata, cellsize=100.0, do_plot=False)
    assert "computeGridSize" in adata.uns["cellchat"]["spatial"]
    np.testing.assert_allclose(adata.uns["cellchat"]["spatial"]["computeGridSize"]["cellsize"], 100.0)


# ---------------------------------------------------------------------------
# makeGridSpatialCellChat
# ---------------------------------------------------------------------------
def test_makeGridSpatialCellChat_typical():
    """Build a small AnnData with 8 spots cleanly in 4 grid cells; verify aggregation.

    Coordinates are picked so that no spot sits on a grid boundary (matching the
    common case; ``sf::st_intersects`` and our :func:`_make_grid_within`
    multi-assigns boundary points, but here every spot lives strictly inside
    one cell).
    """
    coords = np.array(
        [
            [12.0, 12.0],   # grid (0,0)
            [22.0, 22.0],   # grid (0,0)
            [72.0, 72.0],   # grid (1,1)
            [82.0, 82.0],   # grid (1,1)
            [122.0, 122.0], # grid (2,2)
            [132.0, 132.0], # grid (2,2)
            [12.0, 122.0],  # grid (0,2)
            [22.0, 132.0],  # grid (0,2)
        ]
    )
    n_obs, n_var = coords.shape[0], 5
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.uniform(0, 5, size=(n_obs, n_var)))
    obs = pd.DataFrame(index=[f"spot{i:02d}" for i in range(n_obs)])
    obs["active_ident"] = pd.Categorical(["A", "A", "B", "B", "A", "A", "C", "C"])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.uns["cellchat"] = {"spatial.factors": {"ratio": 1.0}}

    grid = makeGridSpatialCellChat(adata, cellsize=50.0)
    # 4 non-empty grids: 3 dominated by A/B/A + 1 dominated by C
    assert grid.n_obs == 4
    assert grid.n_vars == n_var
    # Active ident on the new object matches the dominant input ident in each grid
    expected_idents = sorted(["A", "B", "A", "C"])
    actual_idents = sorted(list(grid.obs["active_ident"].astype(str)))
    assert actual_idents == expected_idents
    # New coordinates equal per-grid mean of input coords
    new_X = grid.X.toarray() if sp.issparse(grid.X) else np.asarray(grid.X)
    assert new_X.shape == (4, n_var)


def test_makeGridSpatialCellChat_visium(tiny_visium):
    """Smoke test on tiny_visium fixture — must produce a valid AnnData."""
    adata = tiny_visium.copy()
    adata.uns["cellchat"] = {"spatial.factors": {"ratio": 0.18}}
    grid = makeGridSpatialCellChat(adata, cellsize=200.0)
    assert isinstance(grid, ad.AnnData)
    assert grid.n_obs > 0
    assert "active_ident" in grid.obs
    assert "spatial" in grid.obsm
    # uns metadata
    assert "cellchat" in grid.uns
    assert "spatial.factors" in grid.uns["cellchat"]
    assert grid.uns["cellchat"]["spatial.factors"]["ratio"] == 0.18
    # The new tol is set to spot_size / 2 = (cellsize * ratio) / 2 = (200*0.18)/2 = 18
    np.testing.assert_allclose(
        grid.uns["cellchat"]["spatial.factors"]["tol"], 200 * 0.18 / 2, rtol=1e-9
    )


def test_makeGridSpatialCellChat_recommended_contact():
    """`recommended.contact.range` = cellsize * ratio."""
    n_obs, n_var = 6, 3
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
                       [0.0, 10.0], [10.0, 10.0], [20.0, 10.0]])
    X = sp.csr_matrix(np.ones((n_obs, n_var)))
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n_obs)])
    obs["active_ident"] = pd.Categorical(["A"] * n_obs)
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_var)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.uns["cellchat"] = {"spatial.factors": {"ratio": 0.5}}
    grid = makeGridSpatialCellChat(adata, cellsize=20.0)
    np.testing.assert_allclose(
        grid.uns["cellchat"]["spatial"]["recommended.contact.range"],
        20.0 * 0.5,
        rtol=1e-9,
    )
