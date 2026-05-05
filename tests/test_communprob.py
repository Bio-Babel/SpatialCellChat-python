"""Tests for :func:`spatialcellchat.modeling.computeCommunProb` and the
pathway-level aggregator :func:`computeCommunProbPathway`.

These tests build a self-contained tiny AnnData and exercise the
key branches of ``computeCommunProb`` (default, mixed annotation,
``contact_dependent_forced``, ``use_AGAN=False``). Numerical equivalence
to R is verified at machine epsilon by
``test_computeCommunProb_matches_R_golden`` (loads bundled CSV golden
values from ``tests/data/r_dumps/communprob/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from spatialcellchat import database, modeling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db():
    return database.load_db("human")


def _build_adata(
    db: dict,
    pair_names: list[str],
    seed: int = 123,
    n_cells: int = 12,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Build a tiny SpatialCellChat-shape AnnData with the requested LR pairs.

    Returns ``(adata, LRsig_df)``.
    """
    rng = np.random.default_rng(seed)
    interaction = db["interaction"]
    LRsig = interaction.loc[pair_names].copy()

    # Collect every gene/subunit referenced by these pairs.
    from spatialcellchat.database import extractGeneSubsetFromPair
    extracted = extractGeneSubsetFromPair(
        LRsig, complex_input=db["complex"], geneInfo=db["geneInfo"], combined=True
    )
    gene_universe = list(dict.fromkeys(extracted))
    n_genes = len(gene_universe)
    raw = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
    raw = raw / max(1.0, raw.max()) + 0.01
    X = sparse.csr_matrix(raw)
    coords = rng.uniform(0, 100, size=(n_cells, 2))

    obs = pd.DataFrame(
        {"cellchat_idents": pd.Categorical(
            np.tile(["A", "B", "C"], (n_cells // 3) + 1)[:n_cells],
            categories=["A", "B", "C"],
        )},
        index=pd.Index([f"cell{i}" for i in range(n_cells)]),
    )
    var = pd.DataFrame(index=pd.Index(gene_universe))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = coords
    adata.uns["cellchat"] = {
        "spatial.factors": {"ratio": 1.0, "tol": 1.0},
        "DB": db,
        "LR": {"LRsig": LRsig},
    }
    return adata, LRsig


# ---------------------------------------------------------------------------
# Default branch — secreted-only (or mixed) annotation
# ---------------------------------------------------------------------------
def test_computeCommunProb_secreted_basic(db) -> None:
    adata, LRsig = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"])
    modeling.computeCommunProb(
        adata, Kh=0.5, n=1.0, distance_use=True,
        interaction_range=250.0, scale_distance=1.5,
        contact_range=10.0,
    )
    cc = adata.uns["cellchat"]
    assert "net" in cc
    assert "prob.cell" in cc["net"]
    pc = cc["net"]["prob.cell"]
    nC = adata.n_obs
    assert pc.shape == (nC, nC, 2)

    # Per-LR matrices should be populated as a dict.
    pc_dict = cc["net"]["tmp"]["prob.cell"]
    assert set(pc_dict.keys()) == set(LRsig.index)
    for k, M in pc_dict.items():
        assert sparse.issparse(M)
        assert M.shape == (nC, nC)


def test_computeCommunProb_writes_parameters(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["Kh"] == 0.5
    assert params["n"] == 1.0
    assert params["distance.use"] is True
    assert params["interaction.range"] == 250.0
    assert params["nLR"] == 1
    # Single secreted pair -> all_diffusible should be True
    assert params["all.diffusible"] is True
    assert params["all.contact.dependent"] is False


def test_computeCommunProb_dimnames(db) -> None:
    adata, LRsig = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    dimnames = adata.uns["cellchat"]["net"]["dimnames"]
    assert dimnames["pairs"] == list(LRsig.index)
    assert dimnames["cells"] == list(adata.obs_names)


# ---------------------------------------------------------------------------
# Mixed annotation branch — secreted + cell-cell contact
# ---------------------------------------------------------------------------
def test_computeCommunProb_mixed_annotation(db) -> None:
    interaction = db["interaction"]
    secreted = interaction.index[interaction["annotation"] == "Secreted Signaling"][:2]
    contact = interaction.index[interaction["annotation"] == "Cell-Cell Contact"][:2]
    pairs = list(secreted) + list(contact)
    adata, _ = _build_adata(db, pairs, n_cells=12, seed=456)
    modeling.computeCommunProb(adata, scale_distance=1.5)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["nLR"] == 4
    # nLR1 should be 2 (the 2 secreted ones)
    assert params["nLR1"] == 2
    assert params["all.contact.dependent"] is False
    assert params["all.diffusible"] is False


# ---------------------------------------------------------------------------
# All Cell-Cell Contact branch
# ---------------------------------------------------------------------------
def test_computeCommunProb_all_cellcell_contact(db) -> None:
    interaction = db["interaction"]
    contact = interaction.index[interaction["annotation"] == "Cell-Cell Contact"][:2]
    adata, _ = _build_adata(db, list(contact), n_cells=12, seed=789)
    modeling.computeCommunProb(adata, scale_distance=1.5)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["all.contact.dependent"] is True
    assert params["all.diffusible"] is False


# ---------------------------------------------------------------------------
# contact_dependent_forced branch
# ---------------------------------------------------------------------------
def test_computeCommunProb_contact_forced(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5, contact_dependent_forced=True)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["contact.dependent.forced"] is True
    assert params["all.contact.dependent"] is True


# ---------------------------------------------------------------------------
# use_AGAN = False branch
# ---------------------------------------------------------------------------
def test_computeCommunProb_no_AGAN(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5, use_AGAN=False)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["use.AGAN"] is False
    pc = adata.uns["cellchat"]["net"]["prob.cell"]
    # Probabilities still exist
    assert pc.shape == (adata.n_obs, adata.n_obs, 1)


# ---------------------------------------------------------------------------
# distance_use = False branch
# ---------------------------------------------------------------------------
def test_computeCommunProb_no_distance(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, distance_use=False)
    params = adata.uns["cellchat"]["options"]["parameter"]
    assert params["distance.use"] is False


# ---------------------------------------------------------------------------
# Validation against R-dumped golden bundled at tests/data/r_dumps/communprob/
# (~11 KB, 11 CSVs). Produced once by ``r_harness/communprob_harness.R``
# (lives outside the package tree). This is the Tier-1 numerical-parity test.
# ---------------------------------------------------------------------------
def test_computeCommunProb_matches_R_golden(db) -> None:
    DUMP = Path(__file__).parent / "data" / "r_dumps" / "communprob"
    data_use = pd.read_csv(DUMP / "data_use.csv", index_col=0)
    coords = pd.read_csv(DUMP / "coords.csv", index_col=0).to_numpy(dtype=float)
    meta = pd.read_csv(DUMP / "meta.csv", index_col=0)
    LRsub = pd.read_csv(DUMP / "LRsig.csv")
    LRsub.index = pd.Index(LRsub["interaction_name"].astype(str))

    n_cells = data_use.shape[1]
    X_csr = sparse.csr_matrix(data_use.to_numpy().T)
    adata = ad.AnnData(
        X=X_csr,
        obs=meta.copy(),
        var=pd.DataFrame(index=data_use.index.astype(str)),
    )
    adata.obs["cellchat_idents"] = pd.Categorical(meta["group"].values, categories=["A", "B", "C"])
    adata.obsm["spatial"] = coords
    adata.uns["cellchat"] = {
        "spatial.factors": {"ratio": 1.0, "tol": 1.0},
        "DB": db,
        "LR": {"LRsig": LRsub},
    }

    modeling.computeCommunProb(
        adata, Kh=0.5, n=1.0, distance_use=True,
        interaction_range=250.0, scale_distance=1.5,
        contact_range=10.0,
    )
    pc_dict = adata.uns["cellchat"]["net"]["tmp"]["prob.cell"]
    for i, name in enumerate(LRsub.index.astype(str)):
        py = pc_dict[name].toarray()
        r = pd.read_csv(DUMP / f"prob_cell_{i+1}.csv").to_numpy(dtype=float)
        # Tier-1 parity (Pearson correlation >= 0.99).
        py_flat = py.ravel()
        r_flat = r.ravel()
        if py_flat.std() == 0 and r_flat.std() == 0:
            corr = 1.0
        else:
            corr = float(np.corrcoef(py_flat, r_flat)[0, 1])
        assert corr >= 0.99, f"LR pair {name} pearson r={corr}"
        # Tighter parity expected (matches at machine epsilon).
        np.testing.assert_allclose(py, r, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# computeCommunProbPathway
# ---------------------------------------------------------------------------
def test_computeCommunProbPathway_cell_only(db) -> None:
    adata, LRsig = _build_adata(db, [
        "TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"
    ])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    modeling.computeCommunProbPathway(adata, do_group=False, do_cell=True)
    cc = adata.uns["cellchat"]
    assert "netP" in cc
    netP = cc["netP"]
    assert "prob.cell" in netP
    assert netP["prob.cell"].shape[0] == adata.n_obs
    # Should have at least one pathway (TGFb / BMP)
    assert len(netP["pathways.cell"]) >= 1


def test_computeCommunProbPathway_requires_prob_for_group(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    with pytest.raises(ValueError, match="net\\['prob'\\]"):
        modeling.computeCommunProbPathway(adata, do_group=True, do_cell=False)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_computeCommunProb_requires_spatial(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    del adata.obsm["spatial"]
    with pytest.raises(ValueError, match="obsm"):
        modeling.computeCommunProb(adata, scale_distance=1.5)


def test_computeCommunProb_requires_spatial_factors(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    del adata.uns["cellchat"]["spatial.factors"]
    with pytest.raises(ValueError, match="spatial.factors"):
        modeling.computeCommunProb(adata, scale_distance=1.5)


def test_computeCommunProb_requires_LR(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    del adata.uns["cellchat"]["LR"]
    with pytest.raises(ValueError, match="LRsig"):
        modeling.computeCommunProb(adata, scale_distance=1.5)


def test_computeCommunProb_scale_distance_too_small(db) -> None:
    adata, _ = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    with pytest.raises(ValueError, match="scale"):
        modeling.computeCommunProb(adata, scale_distance=0.001)
