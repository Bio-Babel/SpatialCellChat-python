"""Tests for the filter / aggregator / accessor exports in
:mod:`spatialcellchat.modeling`:

- :func:`scMatrixTruncation`
- :func:`filterCommunication`
- :func:`filterProbability`
- :func:`aggregateNet`
- :func:`subsetCommunication`
- :func:`identifyEnrichedInteractions`
"""

from __future__ import annotations

from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import sparse as _sparse
from scipy import sparse

from spatialcellchat import database, modeling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db():
    return database.load_db("human")


def _build_adata(db: dict, pair_names: list[str], seed: int = 123, n_cells: int = 12):
    """Same as the tests/test_communprob.py helper."""
    from spatialcellchat.database import extractGeneSubsetFromPair
    rng = np.random.default_rng(seed)
    interaction = db["interaction"]
    LRsig = interaction.loc[pair_names].copy()
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
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gene_universe))
    adata.obsm["spatial"] = coords
    adata.uns["cellchat"] = {
        "spatial.factors": {"ratio": 1.0, "tol": 1.0},
        "DB": db,
        "LR": {"LRsig": LRsig},
    }
    return adata


# ---------------------------------------------------------------------------
# scMatrixTruncation
# ---------------------------------------------------------------------------
def test_scMatrixTruncation_keeps_above_cutoff() -> None:
    M = sparse.csc_matrix(np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]))
    out = modeling.scMatrixTruncation(M, cutoff=2.0, remain_cutoff_v=True)
    expected = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 4.0]])
    np.testing.assert_array_equal(out.toarray(), expected)


def test_scMatrixTruncation_strict_above_cutoff() -> None:
    M = sparse.csc_matrix(np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]))
    out = modeling.scMatrixTruncation(M, cutoff=2.0, remain_cutoff_v=False)
    expected = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]])
    np.testing.assert_array_equal(out.toarray(), expected)


def test_scMatrixTruncation_no_cutoff_passthrough() -> None:
    M = sparse.csc_matrix(np.array([[1.0, 2.0]]))
    out = modeling.scMatrixTruncation(M, cutoff=None)
    np.testing.assert_array_equal(out.toarray(), M.toarray())


def test_scMatrixTruncation_repr_T() -> None:
    M = sparse.csc_matrix(np.array([[0.0, 1.0]]))
    out = modeling.scMatrixTruncation(M, cutoff=0.5, repr="T")
    assert sparse.isspmatrix_coo(out) or out.format == "coo"


def test_scMatrixTruncation_invalid_repr() -> None:
    M = sparse.csc_matrix(np.array([[1.0]]))
    with pytest.raises(ValueError):
        modeling.scMatrixTruncation(M, cutoff=0.0, repr="X")


def test_scMatrixTruncation_dense_input_raises() -> None:
    with pytest.raises(TypeError, match="dgCMatrix"):
        modeling.scMatrixTruncation(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# filterCommunication
# ---------------------------------------------------------------------------
def test_filterCommunication_min_links_zeroes_low_count_pairs(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    pc_dict = adata.uns["cellchat"]["net"]["tmp"]["prob.cell"]
    n_initial_nonzero = sum(M.nnz for M in pc_dict.values())
    # Set min_links to a value larger than current nnz, so all should be zeroed
    huge_min = max(M.nnz for M in pc_dict.values()) + 1
    modeling.filterCommunication(adata, min_cells=None, min_links=huge_min, min_cells_sr=None)
    pc_dict_after = adata.uns["cellchat"]["net"]["tmp"]["prob.cell"]
    for M in pc_dict_after.values():
        # min_links applies only when prob_sum > 0 AND prob_sum < min_links
        # -> all originally non-empty should now be empty
        assert M.nnz == 0


def test_filterCommunication_requires_prob_cell() -> None:
    adata = ad.AnnData(np.zeros((3, 3)))
    adata.uns["cellchat"] = {"net": {}}
    with pytest.raises(ValueError, match="computeCommunProb"):
        modeling.filterCommunication(adata)


def test_filterCommunication_min_cells_sr(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    # min_cells_sr above the cell count should zero the matrix
    huge_sr = adata.n_obs + 1
    modeling.filterCommunication(
        adata, min_cells=None, min_links=None, min_cells_sr=huge_sr,
    )
    pc_dict = adata.uns["cellchat"]["net"]["tmp"]["prob.cell"]
    for M in pc_dict.values():
        assert M.nnz == 0


# ---------------------------------------------------------------------------
# filterProbability
# ---------------------------------------------------------------------------
def test_filterProbability_runs(db) -> None:
    # n_cells must be >= nboot for sample-without-replacement
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"], n_cells=24)
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nnz_before = sum(M.nnz for M in adata.uns["cellchat"]["net"]["tmp"]["prob.cell"].values())
    modeling.filterProbability(adata, nboot=20, seed_use=42, thresh=0.5)
    nnz_after = sum(M.nnz for M in adata.uns["cellchat"]["net"]["tmp"]["prob.cell"].values())
    # Truncating with a quantile threshold should never grow nnz
    assert nnz_after <= nnz_before


def test_filterProbability_thresh_zero_returns_unchanged(db) -> None:
    """thresh=1 -> quantile_prob = 0 -> early return."""
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nnz_before = sum(M.nnz for M in adata.uns["cellchat"]["net"]["tmp"]["prob.cell"].values())
    modeling.filterProbability(adata, nboot=10, thresh=1.0)
    nnz_after = sum(M.nnz for M in adata.uns["cellchat"]["net"]["tmp"]["prob.cell"].values())
    assert nnz_after == nnz_before


def test_filterProbability_requires_computeCommunProb_first() -> None:
    adata = ad.AnnData(np.zeros((3, 3)))
    adata.uns["cellchat"] = {"net": {}}
    with pytest.raises(ValueError, match="computeCommunProb"):
        modeling.filterProbability(adata)


# ---------------------------------------------------------------------------
# aggregateNet — long-table branch (the no-filter branch needs net['prob']
# from sub-agent B, so it is exercised in tests/test_avg.py).
# ---------------------------------------------------------------------------
def test_aggregateNet_long_table_branch(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    # Inject a fake group-level net['prob']/net['pval'] (sub-agent B's output)
    nGroups = 3
    nLR = 1
    rng = np.random.default_rng(0)
    adata.uns["cellchat"]["net"]["prob"] = rng.uniform(0, 1, size=(nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["pval"] = rng.uniform(0, 0.05, size=(nGroups, nGroups, nLR))
    modeling.aggregateNet(adata, sources_use=["A"], targets_use=["B", "C"], thresh=0.05)
    net = adata.uns["cellchat"]["net"]
    assert "count" in net
    assert "weight" in net


def test_aggregateNet_no_filter_requires_prob(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    with pytest.raises(ValueError, match="prob"):
        modeling.aggregateNet(adata)


def test_aggregateNet_writes_count_cell(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nGroups = 3
    nLR = 1
    rng = np.random.default_rng(1)
    adata.uns["cellchat"]["net"]["prob"] = rng.uniform(0, 1, size=(nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["pval"] = rng.uniform(0, 0.05, size=(nGroups, nGroups, nLR))
    modeling.aggregateNet(adata)
    net = adata.uns["cellchat"]["net"]
    assert "count.cell" in net
    assert "weight.cell" in net


# ---------------------------------------------------------------------------
# subsetCommunication
# ---------------------------------------------------------------------------
def test_subsetCommunication_long_table(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nGroups = 3
    nLR = 2
    # Put nontrivial values
    rng = np.random.default_rng(2)
    prob = rng.uniform(0, 1, size=(nGroups, nGroups, nLR))
    pval = rng.uniform(0, 0.04, size=(nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["prob"] = prob
    adata.uns["cellchat"]["net"]["pval"] = pval
    df = modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)
    assert "source" in df.columns
    assert "target" in df.columns
    assert "ligand" in df.columns
    assert "interaction_name" in df.columns
    # Should have entries (since pvals are below 0.05)
    assert len(df) > 0


def test_subsetCommunication_filters_sources(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nGroups = 3
    nLR = 1
    rng = np.random.default_rng(3)
    adata.uns["cellchat"]["net"]["prob"] = rng.uniform(0, 1, size=(nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["pval"] = rng.uniform(0, 0.04, size=(nGroups, nGroups, nLR))
    df = modeling.subsetCommunication(adata, sources_use=["A"], thresh=0.05)
    assert df["source"].unique().tolist() == ["A"]


def test_subsetCommunication_no_results_raises(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nGroups = 3
    nLR = 1
    # All zeros -> nothing significant
    adata.uns["cellchat"]["net"]["prob"] = np.zeros((nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["pval"] = np.ones((nGroups, nGroups, nLR))
    with pytest.raises(ValueError, match="No significant"):
        modeling.subsetCommunication(adata)


# ---------------------------------------------------------------------------
# Audit G4: vectorised long-table construction must match the original
# triple-loop output element-for-element (modulo row order).
# ---------------------------------------------------------------------------
def test_subsetCommunication_vectorised_matches_triple_loop_reference(db) -> None:
    """Audit G4: the new vectorised ``np.where`` path must produce the
    SAME (source, target, interaction_name, prob, pval) rows as the
    naive triple-loop reference implementation (R parity:
    ``reshape2::melt(prob); subset(prob > 0)``).

    Pin row-by-row equality (after sorting on a stable key) so any
    future refactor that drifts numerically or topologically fails
    immediately.
    """
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    rng = np.random.default_rng(11)
    nGroups, nLR = 3, 2
    prob = rng.uniform(0.0, 1.0, size=(nGroups, nGroups, nLR))
    pval = rng.uniform(0.0, 0.04, size=(nGroups, nGroups, nLR))
    # Force some zeros to verify the prob > 0 filter.
    prob[0, 1, 0] = 0.0
    prob[2, 2, 1] = 0.0
    adata.uns["cellchat"]["net"]["prob"] = prob
    adata.uns["cellchat"]["net"]["pval"] = pval

    # Reference: the original triple-loop construction (verbatim).
    cells_level = list(
        adata.obs["cellchat_idents"].cat.categories
        if "cellchat_idents" in adata.obs.columns
        else adata.obs["active_ident"].cat.categories
    )
    pair_names = list(adata.uns["cellchat"]["LR"]["LRsig"].index.astype(str))
    prob_thresh = prob.copy()
    prob_thresh[pval >= 0.05] = 0.0
    ref_rows: list[dict] = []
    for k in range(nLR):
        for i in range(nGroups):
            for j in range(nGroups):
                p = prob_thresh[i, j, k]
                if p == 0.0:
                    continue
                ref_rows.append({
                    "source": cells_level[i],
                    "target": cells_level[j],
                    "interaction_name": pair_names[k],
                    "prob": p,
                    "pval": pval[i, j, k],
                })
    ref = pd.DataFrame(ref_rows)

    df = modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)
    keep_cols = ["source", "target", "interaction_name", "prob", "pval"]
    actual = df.loc[:, keep_cols].copy()
    expected = ref.loc[:, keep_cols].copy()

    # Sort on the lexicographic key so row order is canonical (the new
    # vectorised path uses k-major nonzero indexing; the reference uses
    # k-major triple-loop — they should agree element-wise even before
    # sorting, but pin the canonical sort to be robust.)
    sort_key = ["interaction_name", "source", "target"]
    actual = actual.sort_values(sort_key).reset_index(drop=True)
    expected = expected.sort_values(sort_key).reset_index(drop=True)

    assert actual.shape == expected.shape, (
        f"row count diverged: {actual.shape[0]} vs {expected.shape[0]}"
    )
    # String columns: exact equality.
    assert (actual["source"].values == expected["source"].values).all()
    assert (actual["target"].values == expected["target"].values).all()
    assert (actual["interaction_name"].values
            == expected["interaction_name"].values).all()
    # Numeric columns: bit-exact (pure indexing, no float arithmetic
    # diff between vectorised and looped paths — array indexing must
    # produce identical values).
    np.testing.assert_array_equal(
        actual["prob"].to_numpy(dtype=float),
        expected["prob"].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(
        actual["pval"].to_numpy(dtype=float),
        expected["pval"].to_numpy(dtype=float),
    )


def test_subsetCommunication_vectorised_idempotent(db) -> None:
    """Audit G4: the vectorised long-table construction must be
    idempotent — calling twice on the same input yields identical
    output (no hidden state in the helper)."""
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2", "BMP2_BMPR1A_BMPR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    rng = np.random.default_rng(13)
    nGroups, nLR = 3, 2
    adata.uns["cellchat"]["net"]["prob"] = rng.uniform(0.0, 1.0, size=(nGroups, nGroups, nLR))
    adata.uns["cellchat"]["net"]["pval"] = rng.uniform(0.0, 0.04, size=(nGroups, nGroups, nLR))
    df1 = modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)
    df2 = modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)
    pd.testing.assert_frame_equal(
        df1.reset_index(drop=True),
        df2.reset_index(drop=True),
    )


def test_subsetCommunication_vectorised_filters_zero_prob(db) -> None:
    """Audit G4: only entries with ``prob > 0`` survive (R parity:
    ``subset(df.net, prob > 0)`` at analysis.R:2386). Zero-prob cells
    must NOT appear in the output."""
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    nGroups, nLR = 3, 1
    prob = np.zeros((nGroups, nGroups, nLR))
    # Only one nonzero entry survives.
    prob[1, 2, 0] = 0.7
    adata.uns["cellchat"]["net"]["prob"] = prob
    adata.uns["cellchat"]["net"]["pval"] = np.full((nGroups, nGroups, nLR), 0.01)
    df = modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)
    assert len(df) == 1
    assert df.iloc[0]["prob"] == 0.7
    # And: pval < thresh enforces no extra rows when pval is high.
    adata.uns["cellchat"]["net"]["pval"] = np.full((nGroups, nGroups, nLR), 0.99)
    with pytest.raises(ValueError, match="No significant"):
        modeling.subsetCommunication(adata, slot_name="net", thresh=0.05)


# ---------------------------------------------------------------------------
# identifyEnrichedInteractions
# ---------------------------------------------------------------------------
def test_identifyEnrichedInteractions_requires_pairwiseRank(db) -> None:
    adata = _build_adata(db, ["TGFB1_TGFBR1_TGFBR2"])
    modeling.computeCommunProb(adata, scale_distance=1.5)
    with pytest.raises(ValueError, match="rankNetPairwise"):
        modeling.identifyEnrichedInteractions(adata, from_=["A"], to=["B"])


def test_identifyEnrichedInteractions_uses_pairwiseRank() -> None:
    """Build a minimal pairwiseRank dict to exercise the happy path."""
    adata = ad.AnnData(np.zeros((3, 3)))
    fake_lr = pd.DataFrame(
        {"prob": [0.8, 0.3], "pval": [0.01, 0.5], "ligand": ["L1", "L2"], "receptor": ["R1", "R2"]},
        index=pd.Index(["pair_X", "pair_Y"]),
    )
    adata.uns["cellchat"] = {
        "net": {
            "pairwiseRank": {
                "A": {"B": fake_lr},
                "B": {"A": fake_lr},
            }
        }
    }
    out = modeling.identifyEnrichedInteractions(
        adata, from_=["A"], to=["B"], thresh=0.05,
    )
    assert "ligand" in out.columns
    assert "receptor" in out.columns
    # pair_X has pval=0.01 which is < 0.05; pair_Y has pval=0.5 -> filtered
    assert "pair_X" in out.index


def test_identifyEnrichedInteractions_length_mismatch_errors() -> None:
    adata = ad.AnnData(np.zeros((3, 3)))
    fake_lr = pd.DataFrame(
        {"prob": [0.8], "pval": [0.01], "ligand": ["L1"], "receptor": ["R1"]},
        index=pd.Index(["pair_X"]),
    )
    adata.uns["cellchat"] = {
        "net": {"pairwiseRank": {"A": {"B": fake_lr}}}
    }
    with pytest.raises(ValueError, match="length"):
        modeling.identifyEnrichedInteractions(adata, from_=["A"], to=["B", "A"])
