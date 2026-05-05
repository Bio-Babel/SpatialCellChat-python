"""Tests for spatialcellchat.analysis_topics — slice 5B exports."""

from __future__ import annotations

import importlib.util as _imp_util

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialcellchat.analysis_topics import (
    _extract_enriched_lr_internal,
    _prepare_pattern_data,
    _select_centr_cell,
    _suggestK_stats,
    _top_features,
    computeEnrichmentScore,
    extractEnrichedLR,
    extractTopicSignaling,
    identifyCellTopics,
    identifyCommunicationPatterns,
    selectK,
    suggestK,
    suggestK1,
)


# ---------------------------------------------------------------------------
# _prepare_pattern_data
# ---------------------------------------------------------------------------
def test_prepare_pattern_data_outgoing_matches_R():
    """Verify R's apply(prob, c(1,3), sum) + sweep + row-filter."""
    rng = np.random.default_rng(7)
    prob = rng.uniform(0.0, 1.0, size=(4, 4, 5))
    data, _data0, keep = _prepare_pattern_data(prob, "outgoing")
    # data_sender = sum over receivers (axis=1)
    expected = prob.sum(axis=1)
    expected = expected / expected.max(axis=0, keepdims=True)
    np.testing.assert_allclose(data, expected[keep, :])


def test_prepare_pattern_data_incoming_uses_axis0():
    rng = np.random.default_rng(8)
    prob = rng.uniform(0.0, 1.0, size=(3, 3, 4))
    data, _data0, keep = _prepare_pattern_data(prob, "incoming")
    expected = prob.sum(axis=0)
    expected = expected / expected.max(axis=0, keepdims=True)
    np.testing.assert_allclose(data, expected[keep, :])


def test_prepare_pattern_data_drops_zero_rows():
    prob = np.zeros((4, 4, 3))
    prob[0, :, 0] = [1.0, 2.0, 3.0, 4.0]  # only row 0 has outgoing signal
    prob[2, :, 1] = [0.5, 0.5, 0.5, 0.5]
    data, _data0, keep = _prepare_pattern_data(prob, "outgoing")
    assert keep.tolist() == [0, 2]
    assert data.shape == (2, 3)


def test_prepare_pattern_data_returns_data_data0_keep():
    """R parity (CellChat::analysis.R:305-313): the helper exposes both
    the row-zero-filtered NMF input and the unfiltered ``data0`` (which
    R persists into ``slot$pattern[[pattern]]$data`` per
    analysis.R:445)."""
    prob = np.zeros((4, 4, 3))
    prob[0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    prob[2, :, 1] = [0.5, 0.5, 0.5, 0.5]
    data, data0, keep = _prepare_pattern_data(prob, "outgoing")
    assert data0.shape == (4, 3)             # full K rows preserved
    assert data.shape == (len(keep), 3)      # filtered to non-zero rows
    np.testing.assert_allclose(data, data0[keep, :])


def test_prepare_pattern_data_invalid_pattern_raises():
    with pytest.raises(ValueError):
        _prepare_pattern_data(np.zeros((2, 2, 2)), "foo")


def test_prepare_pattern_data_invalid_dim_raises():
    with pytest.raises(ValueError):
        _prepare_pattern_data(np.zeros((2, 2)), "outgoing")


def test_prepare_pattern_data_matches_R_harness_test1():
    """Element-wise R parity for the simple 4x4x5 case (harness Test1).

    The R harness produced::

        data_sender_swept[1,] = [1.00000, 0.96060, 0.93371, 0.82706, 0.83816]
        ...
    """
    rng = np.random.default_rng(7)
    # numpy / R RNG differ; here we just verify the math by recomputing
    # the swept matrix manually and comparing against `_prepare_pattern_data`.
    prob = rng.uniform(0.0, 1.0, size=(4, 4, 5))
    # R: apply(prob, c(1,3), sum)  -> sum over receivers (axis=1)
    expected = prob.sum(axis=1)
    expected = expected / expected.max(axis=0, keepdims=True)
    data, _data0, keep = _prepare_pattern_data(prob, "outgoing")
    np.testing.assert_allclose(data, expected[keep, :], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# selectK
# ---------------------------------------------------------------------------
def _make_adata_with_netP(prob_shape=(4, 4, 6), seed=42):
    """Build an AnnData with a synthetic netP['prob'] tensor."""
    rng = np.random.default_rng(seed)
    K, _, nLR = prob_shape
    prob = rng.uniform(0.0, 1.0, size=prob_shape)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(K)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])
    X = rng.uniform(0.0, 1.0, size=(K, 8))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["cellchat"] = {
        "netP": {
            "prob": prob,
            "pathways": [f"PW{i + 1}" for i in range(nLR)],
            "dimnames": [list(obs.index), list(obs.index), [f"PW{i + 1}" for i in range(nLR)]],
        }
    }
    return adata


def test_selectK_returns_dict_with_curves():
    adata = _make_adata_with_netP(seed=1)
    res = selectK(adata, k_range=[2, 3, 4], nrun=3, seed_use=10)
    assert "k" in res and "cophenetic" in res and "silhouette" in res and "k_hat" in res
    assert len(res["k"]) == 3
    assert len(res["cophenetic"]) == 3
    assert res["k_hat"] in (2, 3, 4)
    # data should be (rows_kept, nLR)
    assert res["data"].shape[1] == 6


def test_selectK_missing_slot_raises():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.uniform(0.0, 1.0, size=(4, 8)))
    with pytest.raises(ValueError, match="netP"):
        selectK(adata, k_range=[2, 3])


def test_selectK_incoming_pattern():
    adata = _make_adata_with_netP(seed=2)
    res = selectK(adata, pattern="incoming", k_range=[2, 3], nrun=2, seed_use=10)
    assert res["pattern"] == "incoming"


def test_selectK_honours_k_range():
    """R's selectK hard-codes range = 20:30 (R bug); Python honours k_range.

    Pin the deviation: when the caller passes k_range=[2, 3, 4], the
    returned diagnostic curves must align with those exact ranks (not
    R's leftover 20:30 default).
    """
    adata = _make_adata_with_netP(seed=42)
    res = selectK(adata, k_range=[2, 3, 4], nrun=2, seed_use=10)
    assert list(res["k"]) == [2, 3, 4]
    assert len(res["cophenetic"]) == 3
    assert len(res["silhouette"]) == 3


# ---------------------------------------------------------------------------
# identifyCommunicationPatterns
# ---------------------------------------------------------------------------
def test_identifyCommunicationPatterns_writes_slot():
    adata = _make_adata_with_netP(prob_shape=(5, 5, 6), seed=3)
    out = identifyCommunicationPatterns(adata, pattern="outgoing", k=2, seed=10)
    pat = out.uns["cellchat"]["netP"]["pattern"]["outgoing"]
    assert "data" in pat and "pattern" in pat and "W" in pat and "H" in pat
    df_W = pat["pattern"]["cell"]
    df_H = pat["pattern"]["signaling"]
    assert isinstance(df_W, pd.DataFrame)
    assert isinstance(df_H, pd.DataFrame)
    # K=5 cell groups (assuming none zeroed out) x k=2 patterns = 10 rows
    # but rows can be filtered, so check shape <= 10
    assert df_W.shape[0] <= 10
    # nLR=6 x k=2 = 12 rows
    assert df_H.shape[0] == 12
    assert set(df_W.columns) == {"CellGroup", "Pattern", "Contribution"}
    assert set(df_H.columns) == {"Pattern", "Signaling", "Contribution"}


def test_identifyCommunicationPatterns_data_is_data0_per_R():
    """R parity (CellChat::analysis.R:445):
    ``slot$pattern[[pattern]] <- list(data = data0, ...)`` — the slot's
    ``data`` key holds the **unfiltered** column-max-normalised matrix
    (R's ``data0``), NOT the row-zero-filtered NMF input.

    The Python port stores ``data_filtered`` separately for callers that
    want the NMF input view.
    """
    # Force a row-zero filter: cell-group 1 has no outgoing prob
    K, nLR = 5, 6
    rng = np.random.default_rng(11)
    prob = rng.uniform(0.0, 1.0, size=(K, K, nLR))
    prob[1, :, :] = 0.0   # cell-group #1 has zero outgoing → filtered
    obs = pd.DataFrame(index=[f"c{i}" for i in range(K)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])
    X = rng.uniform(0.0, 1.0, size=(K, 8))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["cellchat"] = {
        "netP": {
            "prob": prob,
            "pathways": [f"PW{i + 1}" for i in range(nLR)],
            "dimnames": [list(obs.index), list(obs.index),
                         [f"PW{i + 1}" for i in range(nLR)]],
        }
    }
    out = identifyCommunicationPatterns(adata, pattern="outgoing", k=2, seed=11)
    pat = out.uns["cellchat"]["netP"]["pattern"]["outgoing"]
    # data: full K rows (R's data0)
    assert pat["data"].shape == (K, nLR)
    # data_filtered: rows with non-zero sum only (NMF input view)
    assert pat["data_filtered"].shape[0] < K
    assert pat["data_filtered"].shape[1] == nLR
    # The row-zero filter is internally consistent.
    np.testing.assert_allclose(
        pat["data_filtered"], pat["data"][pat["rows_kept"], :], atol=1e-12,
    )


def test_identifyCommunicationPatterns_k_required():
    adata = _make_adata_with_netP(seed=4)
    with pytest.raises(ValueError, match="selectK"):
        identifyCommunicationPatterns(adata, k=None)


def test_identifyCommunicationPatterns_W_row_sums_one():
    """scaleMat(W, 'r1') must yield row sums of 1.0 within tol."""
    adata = _make_adata_with_netP(prob_shape=(5, 5, 6), seed=5)
    out = identifyCommunicationPatterns(adata, pattern="outgoing", k=2, seed=10)
    W = out.uns["cellchat"]["netP"]["pattern"]["outgoing"]["W"]
    # row sums = 1 (some rows may be all-zero due to NMF zeros, treat NaN -> 0)
    sums = np.nansum(W, axis=1)
    nonzero_mask = sums > 0
    np.testing.assert_allclose(sums[nonzero_mask], 1.0, atol=1e-10)


def test_identifyCommunicationPatterns_H_col_sums_one():
    adata = _make_adata_with_netP(prob_shape=(5, 5, 6), seed=6)
    out = identifyCommunicationPatterns(adata, pattern="outgoing", k=2, seed=10)
    H = out.uns["cellchat"]["netP"]["pattern"]["outgoing"]["H"]
    sums = np.nansum(H, axis=0)
    nonzero_mask = sums > 0
    np.testing.assert_allclose(sums[nonzero_mask], 1.0, atol=1e-10)


def test_identifyCommunicationPatterns_incoming_pattern():
    adata = _make_adata_with_netP(prob_shape=(4, 4, 5), seed=7)
    out = identifyCommunicationPatterns(adata, pattern="incoming", k=2, seed=10)
    assert "incoming" in out.uns["cellchat"]["netP"]["pattern"]


def test_identifyCommunicationPatterns_no_prob_raises():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.uniform(0.0, 1.0, size=(4, 8)))
    adata.uns["cellchat"] = {"netP": {}}
    with pytest.raises(ValueError, match="prob"):
        identifyCommunicationPatterns(adata, k=2)


# ---------------------------------------------------------------------------
# suggestK / suggestK1
# ---------------------------------------------------------------------------
def test_suggestK1_returns_stats_and_figure():
    rng = np.random.default_rng(9)
    Mat = rng.uniform(0.1, 1.0, size=(20, 12))
    res = suggestK1(Mat, k_test=[2, 3, 4], reps=2, p=0.3, maxit=50, seed_use=666)
    df = res["stats"]
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"k", "seed_use", "objErr", "iter"}
    assert df.shape[0] == 6  # 3 k's x 2 reps
    # R returns list(stats=..., figure=ggplot); Python returns figure=None
    # (matplotlib renderer lives in slice 6a's plotSuggestK helper).
    assert "figure" in res and res["figure"] is None


def test_suggestK1_p_zero_uses_full_matrix():
    rng = np.random.default_rng(10)
    Mat = rng.uniform(0.1, 1.0, size=(15, 10))
    res = suggestK1(Mat, k_test=[2, 3], reps=2, p=0.0, maxit=50, seed_use=42)
    # objErr should be normalised by full matrix size
    df = res["stats"]
    err_2 = df.loc[df["k"] == 2, "objErr"].mean()
    err_3 = df.loc[df["k"] == 3, "objErr"].mean()
    # higher k should fit better → err_3 <= err_2
    assert err_3 <= err_2 + 1e-3


def test_suggestK1_invalid_k_test_raises():
    rng = np.random.default_rng(11)
    Mat = rng.uniform(0.1, 1.0, size=(10, 6))
    with pytest.raises(ValueError, match="more than one"):
        suggestK1(Mat, k_test=[2], reps=2, seed_use=1)


def test_suggestK1_invalid_p_raises():
    rng = np.random.default_rng(12)
    Mat = rng.uniform(0.1, 1.0, size=(10, 6))
    with pytest.raises(ValueError, match="0,1"):
        suggestK1(Mat, k_test=[2, 3], reps=1, p=1.0)


def test_suggestK1_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a matrix"):
        suggestK1("not_a_matrix", k_test=[2, 3])


def test_suggestK1_L1_is_actually_forwarded_to_NMF():
    """R-bug fix (intentional deviation from SpatialCellChat analysis.R:
    3772-3779). The R signature documents ``L1 = c(0, 0)`` (analysis.R:
    3733-3745) but the body calls ``RcppML::nmf(...)`` without forwarding
    ``L1`` — silent typo. Python honours the signature's promise.

    Pin: if ``L1`` actually reaches the NMF, then ``L1=(0,0)`` and
    ``L1=(0.5,0.5)`` must produce DIFFERENT objErr values for the SAME
    matrix and seed (sparsity penalty changes the fit). If the prior
    bug-mirror behaviour returned, this test would fail because both
    runs would land on identical L1=(0,0) NMF outputs."""
    rng = np.random.default_rng(101)
    Mat = rng.uniform(0.1, 1.0, size=(20, 12))
    res_no_L1 = suggestK1(Mat, k_test=[3, 4], reps=2, p=0.3, maxit=50,
                          L1=(0.0, 0.0), seed_use=42)
    res_with_L1 = suggestK1(Mat, k_test=[3, 4], reps=2, p=0.3, maxit=50,
                            L1=(0.5, 0.5), seed_use=42)
    # The objErr columns should differ in at least one row when L1 is
    # actively applied. (Equality across all rows would imply L1 was
    # silently ignored — the very bug we're fixing.)
    err_no_L1 = res_no_L1["stats"]["objErr"].to_numpy(dtype=float)
    err_with_L1 = res_with_L1["stats"]["objErr"].to_numpy(dtype=float)
    assert not np.allclose(err_no_L1, err_with_L1, atol=1e-12), (
        "L1 was silently ignored (R-bug regression). Expected "
        f"differing objErr between L1=(0,0) and L1=(0.5,0.5); got "
        f"identical values: {err_no_L1.tolist()}"
    )


def test_suggestK1_seed_sample_is_without_replacement():
    """R uses sample(1:10000, size=reps) — without replacement."""
    rng = np.random.default_rng(13)
    Mat = rng.uniform(0.1, 1.0, size=(15, 10))
    res = suggestK1(Mat, k_test=[2, 3], reps=10, p=0.3, maxit=50, seed_use=666)
    df = res["stats"]
    # All seed_use values within the same k should be unique (no duplicates).
    seeds_for_k2 = df.loc[df["k"] == 2, "seed_use"].tolist()
    assert len(seeds_for_k2) == len(set(seeds_for_k2))


def test_suggestK_uses_adata_X():
    rng = np.random.default_rng(14)
    X = rng.uniform(0.1, 1.0, size=(20, 12))
    adata = ad.AnnData(X=X)
    res = suggestK(adata, kTest=[2, 3], nRandomStart=2, nIteration=50, seed_use=666)
    assert "stats" in res and "figure" in res
    assert res["figure"] is None


# ---------------------------------------------------------------------------
# computeEnrichmentScore
# ---------------------------------------------------------------------------
def test_computeEnrichmentScore_matches_R_ES_math():
    """Expected from R harness: A_X=0.08, B_Y=0.22, C_Z=0."""
    df = pd.DataFrame({
        "interaction_name": ["A_X", "A_X", "B_Y", "B_Y", "C_Z"],
        "ligand.logFC":     [1.0, 0.5, 2.0, 1.5, 0.0],
        "receptor.logFC":   [0.5, 1.0, 1.0, 0.5, 1.0],
        "ligand.pct.1":     [0.1, 0.2, 0.3, 0.4, 0.5],
        "ligand.pct.2":     [0.5, 0.6, 0.7, 0.8, 0.9],
        "receptor.pct.1":   [0.2, 0.3, 0.4, 0.5, 0.6],
        "receptor.pct.2":   [0.6, 0.7, 0.8, 0.9, 1.0],
    })
    # Build a fake DB so we don't need real CellChatDB
    fake_DB = {
        "interaction": pd.DataFrame({
            "ligand": ["A", "B", "C"],
            "receptor": ["X", "Y", "Z"],
            "pathway_name": ["PW1", "PW2", "PW3"],
        }, index=["A_X", "B_Y", "C_Z"])
    }
    out = computeEnrichmentScore(
        df, measure="ligand", species="mouse", db=fake_DB, return_data=True,
    )
    assert set(out.columns) == {"ligand", "total"}
    # Convert to dict for easy lookup
    d = dict(zip(out["ligand"], out["total"]))
    assert d["A"] == pytest.approx(0.08, abs=1e-6)
    assert d["B"] == pytest.approx(0.22, abs=1e-6)
    assert d["C"] == pytest.approx(0.0, abs=1e-12)


def test_computeEnrichmentScore_invalid_measure_raises():
    fake_DB = {"interaction": pd.DataFrame({
        "ligand": ["A"], "receptor": ["X"], "pathway_name": ["PW"]
    }, index=["A_X"])}
    df = pd.DataFrame({
        "interaction_name": ["A_X"],
        "ligand.logFC": [1.0], "receptor.logFC": [1.0],
        "ligand.pct.1": [0.0], "ligand.pct.2": [1.0],
        "receptor.pct.1": [0.0], "receptor.pct.2": [1.0],
    })
    with pytest.raises(ValueError, match="measure"):
        computeEnrichmentScore(df, measure="foo", db=fake_DB)


def test_computeEnrichmentScore_invalid_species_raises():
    df = pd.DataFrame({
        "interaction_name": ["A_X"],
        "ligand.logFC": [1.0], "receptor.logFC": [1.0],
        "ligand.pct.1": [0.0], "ligand.pct.2": [1.0],
        "receptor.pct.1": [0.0], "receptor.pct.2": [1.0],
    })
    fake_DB = {"interaction": pd.DataFrame({
        "ligand": ["A"], "receptor": ["X"], "pathway_name": ["PW"]
    }, index=["A_X"])}
    with pytest.raises(ValueError, match="species"):
        computeEnrichmentScore(df, species="zebrafish", db=fake_DB)


def test_computeEnrichmentScore_drops_NA_rows():
    df = pd.DataFrame({
        "interaction_name": ["A_X", "A_X"],
        "ligand.logFC":     [1.0, np.nan],
        "receptor.logFC":   [1.0, 2.0],
        "ligand.pct.1":     [0.1, 0.2],
        "ligand.pct.2":     [0.5, 0.6],
        "receptor.pct.1":   [0.2, 0.3],
        "receptor.pct.2":   [0.6, 0.7],
    })
    fake_DB = {"interaction": pd.DataFrame({
        "ligand": ["A"], "receptor": ["X"], "pathway_name": ["PW"]
    }, index=["A_X"])}
    out = computeEnrichmentScore(df, db=fake_DB, return_data=True)
    # only the first row survives the NA drop:
    # |1| * |1| * |0.5-0.1| * |0.6-0.2| = 0.16
    assert out.loc[out["ligand"] == "A", "total"].iloc[0] == pytest.approx(0.16, abs=1e-6)


def test_computeEnrichmentScore_returns_None_without_return_data():
    """R returns invisible(NULL) by default; Python returns None."""
    df = pd.DataFrame({
        "interaction_name": ["A_X"],
        "ligand.logFC":     [1.0],
        "receptor.logFC":   [1.0],
        "ligand.pct.1":     [0.1],
        "ligand.pct.2":     [0.5],
        "receptor.pct.1":   [0.2],
        "receptor.pct.2":   [0.6],
    })
    fake_DB = {"interaction": pd.DataFrame({
        "ligand": ["A"], "receptor": ["X"], "pathway_name": ["PW"]
    }, index=["A_X"])}
    out = computeEnrichmentScore(df, db=fake_DB)
    assert out is None


# ---------------------------------------------------------------------------
# extractEnrichedLR
# ---------------------------------------------------------------------------
def _make_adata_with_net_for_lr():
    """Mock an AnnData with a 'PW1' pathway containing two enriched LR pairs."""
    K, nLR = 3, 4
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(K)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    adata = ad.AnnData(X=rng.uniform(0, 1, (K, 5)), obs=obs, var=var)
    LRsig = pd.DataFrame({
        "interaction_name": ["A_X", "B_Y", "C_Z", "D_W"],
        "pathway_name":     ["PW1", "PW1", "PW2", "PW2"],
        "ligand":           ["A", "B", "C", "D"],
        "receptor":         ["X", "Y", "Z", "W"],
    }, index=["A_X", "B_Y", "C_Z", "D_W"])
    DB = {
        "interaction": LRsig.copy(),
        "complex": pd.DataFrame(columns=["subunit_1"]),
        "geneInfo": pd.DataFrame({"Symbol": ["A", "B", "C", "D", "X", "Y", "Z", "W"]}),
    }
    prob = rng.uniform(0, 1, (K, K, nLR))
    pval = np.full((K, K, nLR), 0.01)
    pval[:, :, 2] = 0.5
    pval[:, :, 3] = 0.5
    # mock prob.cell as a sparse 3-D placeholder (only its 3rd-axis names matter)
    import sparse as _sparse
    prob_cell = _sparse.COO.from_numpy(np.zeros((K, K, nLR)))
    adata.uns["cellchat"] = {
        "DB": DB,
        "LR": {"LRsig": LRsig},
        "net": {
            "prob": prob,
            "pval": pval,
            "prob.cell": prob_cell,
            "LR.sig": ["A_X", "B_Y"],
            "LR.sig.cell": ["A_X", "B_Y"],
            "LR.sig.dimnames3": ["A_X", "B_Y", "C_Z", "D_W"],
            "LR.sig.cell.dimnames3": ["A_X", "B_Y", "C_Z", "D_W"],
        },
    }
    return adata


def test_extractEnrichedLR_returns_pair_df():
    adata = _make_adata_with_net_for_lr()
    out = extractEnrichedLR(adata, signaling=["PW1"])
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"interaction_name"}
    assert sorted(out["interaction_name"].tolist()) == ["A_X", "B_Y"]


def test_extractEnrichedLR_with_geneLR_return():
    adata = _make_adata_with_net_for_lr()
    out = extractEnrichedLR(adata, signaling=["PW1"], geneLR_return=True)
    assert isinstance(out, dict)
    assert "pairLR" in out and "geneLR" in out
    assert sorted(out["pairLR"]["interaction_name"].tolist()) == ["A_X", "B_Y"]
    # geneLR should include all of A, B, X, Y (in some order)
    assert sorted(set(out["geneLR"])) == ["A", "B", "X", "Y"]


def test_extractEnrichedLR_pw2_has_no_significant():
    adata = _make_adata_with_net_for_lr()
    out = extractEnrichedLR(adata, signaling=["PW2"])
    assert out["interaction_name"].tolist() == []


def test_extractEnrichedLR_do_group_false_uses_LR_sig_cell():
    adata = _make_adata_with_net_for_lr()
    out = extractEnrichedLR(adata, signaling=["PW1"], do_group=False)
    assert sorted(out["interaction_name"].tolist()) == ["A_X", "B_Y"]


def test_extractEnrichedLR_internal_returns_geneLR_and_pair_use():
    adata = _make_adata_with_net_for_lr()
    cc = adata.uns["cellchat"]
    geneLR, pair_use = _extract_enriched_lr_internal(
        net=cc["net"], LR=cc["LR"], DB=cc["DB"],
        signaling="PW1", enriched_only=True, thresh=0.05, do_group=True,
    )
    assert sorted(pair_use) == ["A_X", "B_Y"]
    assert sorted(set(geneLR)) == ["A", "B", "X", "Y"]


def test_extractEnrichedLR_missing_LR_slot_raises():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.uniform(0, 1, (3, 5)))
    adata.uns["cellchat"] = {"DB": {}, "net": {"prob": np.zeros((3, 3, 1))}}
    with pytest.raises(ValueError, match="LR"):
        extractEnrichedLR(adata, signaling=["PW1"])


# ---------------------------------------------------------------------------
# identifyCellTopics
# ---------------------------------------------------------------------------
def _make_adata_with_centr_cell(nC=10, nLR=6):
    rng = np.random.default_rng(20)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(nC)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])
    adata = ad.AnnData(X=rng.uniform(0, 1, (nC, 8)), obs=obs, var=var)
    centr_cell = {
        "outdeg": rng.uniform(0.1, 1.0, size=(nC, nLR)),
        "indeg":  rng.uniform(0.1, 1.0, size=(nC, nLR)),
        "outdeg_unweighted": rng.uniform(0, 5, size=(nC, nLR)),
        "indeg_unweighted":  rng.uniform(0, 5, size=(nC, nLR)),
    }
    adata.uns["cellchat"] = {
        "net": {
            "centr.cell": centr_cell,
            "LR.sig.cell": [f"LR{i + 1}" for i in range(nLR)],
        },
    }
    return adata


def test_select_centr_cell_dict_layout():
    arr = np.arange(12).reshape(3, 4).astype(float)
    out = _select_centr_cell({"outdeg": arr}, "outdeg")
    np.testing.assert_array_equal(out, arr)


def test_select_centr_cell_dict_missing_measure_raises():
    arr = np.arange(12).reshape(3, 4).astype(float)
    with pytest.raises(ValueError, match="missing measure"):
        _select_centr_cell({"outdeg": arr}, "indeg")


def test_select_centr_cell_ndarray_raises_without_names():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    with pytest.raises(ValueError, match="dict"):
        _select_centr_cell(arr, "outdeg")


def test_select_centr_cell_values_centr_names_layout():
    """Sub-agent A's :func:`netAnalysis_computeCentrality` writes the dict
    ``{'values': (n_centr, nC, nLR), 'centr_names': [...]}``. The
    ``_select_centr_cell`` helper must look up the measure index in
    ``centr_names`` and slice ``values[idx, :, :]``."""
    n_centr, nC, nLR = 5, 3, 4
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 1, size=(n_centr, nC, nLR))
    names = ["outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg", "page_rank"]
    centr_cell = {
        "values": values,
        "centr_names": names,
        "node_names": [f"c{i}" for i in range(nC)],
        "signaling_names": [f"LR{i}" for i in range(nLR)],
    }
    out = _select_centr_cell(centr_cell, "outdeg")
    np.testing.assert_array_equal(out, values[2, :, :])

    out_in = _select_centr_cell(centr_cell, "indeg")
    np.testing.assert_array_equal(out_in, values[3, :, :])

    with pytest.raises(ValueError, match="missing measure"):
        _select_centr_cell(centr_cell, "doesnotexist")


def test_identifyCellTopics_single_k_writes_topic_slot():
    adata = _make_adata_with_centr_cell(nC=12, nLR=8)
    out = identifyCellTopics(
        adata, pattern="outgoing", topic_k=[3], maxit=200, seed_use=42,
    )
    topic = out.uns["cellchat"]["net"]["topic"]["outgoing"]
    assert "cell" in topic and "signaling" in topic
    df_cell = topic["cell"]
    df_sig = topic["signaling"]
    assert df_cell.shape == (12, 3)
    assert df_sig.shape == (8, 3)
    assert list(df_cell.columns) == ["Topic_1", "Topic_2", "Topic_3"]


def test_identifyCellTopics_multi_k_runs_suggestK1():
    adata = _make_adata_with_centr_cell(nC=20, nLR=10)
    out = identifyCellTopics(
        adata, pattern="outgoing", topic_k=[2, 3, 4],
        n_reps=2, prop=0.3, maxit=100, seed_use=42,
    )
    sk = out.uns["cellchat"]["net"]["tmp"]["suggestK"]["outgoing"]
    assert "stats" in sk and "figure" in sk
    df = sk["stats"]
    assert df.shape[0] == 6  # 3 k's x 2 reps


def test_identifyCellTopics_missing_centr_cell_raises():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.uniform(0, 1, (4, 6)))
    adata.uns["cellchat"] = {"net": {"centr.cell": {}}}
    with pytest.raises(ValueError, match="netAnalysis_computeCentrality"):
        identifyCellTopics(adata, topic_k=[2])


def test_identifyCellTopics_invalid_pattern_raises():
    adata = _make_adata_with_centr_cell()
    with pytest.raises(ValueError, match="pattern"):
        identifyCellTopics(adata, pattern="foo", topic_k=[2])


def test_identifyCellTopics_do_scale_normalises_rows():
    adata = _make_adata_with_centr_cell(nC=8, nLR=5)
    out = identifyCellTopics(
        adata, pattern="incoming", topic_k=[2], do_scale=True, maxit=100, seed_use=1,
    )
    # just verifies no crash + shape parity
    topic = out.uns["cellchat"]["net"]["topic"]["incoming"]
    assert topic["cell"].shape == (8, 2)


# ---------------------------------------------------------------------------
# extractTopicSignaling
# ---------------------------------------------------------------------------
def _make_adata_with_topic_loadings(nLR=15, k=3):
    """Build an AnnData with topic loadings + a fake DB."""
    rng = np.random.default_rng(2)
    loadings = rng.normal(loc=0.0, scale=1.0, size=(nLR, k))
    feature_names = [f"LR_{i + 1}" for i in range(nLR)]
    cols = [f"Topic_{i + 1}" for i in range(k)]
    df = pd.DataFrame(loadings, index=feature_names, columns=cols)
    cell_emb = pd.DataFrame(
        rng.normal(0, 1, (8, k)), index=[f"c{i}" for i in range(8)], columns=cols,
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(8)])
    adata = ad.AnnData(X=rng.uniform(0, 1, (8, 5)), obs=obs)
    adata.uns["cellchat"] = {
        "net": {
            "centr.cell": {"outdeg": rng.uniform(0, 1, (8, nLR))},
            "topic": {"incoming": {"cell": cell_emb, "signaling": df}},
        }
    }
    return adata, df


def test_top_features_unbalanced_matches_R():
    """Reproduce R harness Test6 (dim 1 nfeat=5 balanced=F)."""
    rng = np.random.default_rng(2)
    loadings_R = pd.DataFrame(
        # match R's matrix(rnorm(15*3), 15, 3) by replaying with rnorm seed 2
        # NOTE: R and numpy RNG are not bit-identical; we just check the logic
        # via a direct synthesis below.
        rng.normal(0, 1, (15, 3)),
        index=[f"LR_{i + 1}" for i in range(15)],
        columns=["T1", "T2", "T3"],
    )
    out = _top_features(loadings_R, dim=1, nfeatures=5, balanced=False)
    # the output must contain 5 unique LR names and they must be the 5
    # features with the largest absolute T1 score
    col = loadings_R.iloc[:, 0]
    expected_set = set(col.abs().sort_values(ascending=False).head(5).index)
    assert set(out) == expected_set
    # ordering is by ascending value of data[top, ]
    expected_order = col.loc[list(expected_set)].sort_values(ascending=True).index.tolist()
    assert out == expected_order


def test_top_features_balanced_picks_pos_and_neg():
    rng = np.random.default_rng(3)
    loadings = pd.DataFrame(
        rng.normal(0, 1, (10, 2)),
        index=[f"f{i}" for i in range(10)],
        columns=["A", "B"],
    )
    out = _top_features(loadings, dim=1, nfeatures=4, balanced=True)
    col = loadings.iloc[:, 0]
    # top 2 positive (sorted descending)
    pos = col.sort_values(ascending=False).head(2).index.tolist()
    # bottom 2 negative reversed
    neg = col.sort_values(ascending=False).tail(2).index.tolist()[::-1]
    assert out == pos + neg


def test_extractTopicSignaling_LR_pair_returns_score():
    adata, df = _make_adata_with_topic_loadings(nLR=15, k=3)
    fake_DB = {
        "interaction": pd.DataFrame({
            "ligand":   [f"L{i}" for i in range(15)],
            "receptor": [f"R{i}" for i in range(15)],
            "pathway_name": [f"PW{i % 3}" for i in range(15)],
        }, index=[f"LR_{i + 1}" for i in range(15)])
    }
    out = extractTopicSignaling(
        adata, dims=[1, 2], nfeatures=5, nTop=3,
        feature_cat="LR-pair", reduction="topic", pattern="incoming",
        species="mouse", db=fake_DB, simplify=True,
    )
    assert isinstance(out, pd.DataFrame)
    # 2 dims x 3 nTop = 6 rows max
    assert out.shape[0] <= 6
    assert "score" in out.columns and "rank" in out.columns and "reduction" in out.columns


def test_extractTopicSignaling_simplify_false_returns_dict():
    adata, _ = _make_adata_with_topic_loadings(nLR=10, k=2)
    fake_DB = {
        "interaction": pd.DataFrame({
            "ligand": [f"L{i}" for i in range(10)],
            "receptor": [f"R{i}" for i in range(10)],
            "pathway_name": [f"PW{i % 2}" for i in range(10)],
        }, index=[f"LR_{i + 1}" for i in range(10)])
    }
    out = extractTopicSignaling(
        adata, dims=[1, 2], nfeatures=4, nTop=2, feature_cat="LR-pair",
        pattern="incoming", db=fake_DB, simplify=False,
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"topic_1", "topic_2"}


def test_extractTopicSignaling_ligand_aggregation():
    adata, _ = _make_adata_with_topic_loadings(nLR=12, k=2)
    fake_DB = {
        "interaction": pd.DataFrame({
            "ligand":   ["L0", "L1", "L0", "L1"] * 3,
            "receptor": ["R0", "R1", "R2", "R3"] * 3,
            "pathway_name": ["PW0", "PW1", "PW2", "PW0"] * 3,
        }, index=[f"LR_{i + 1}" for i in range(12)])
    }
    out = extractTopicSignaling(
        adata, dims=[1], nfeatures=8, nTop=4, feature_cat="ligand",
        pattern="incoming", db=fake_DB, simplify=True,
    )
    # ligand aggregation: feature column should contain ligand names
    assert "feature" in out.columns
    # no duplicate ligands within a single dim
    assert out["feature"].is_unique


def test_extractTopicSignaling_invalid_feature_cat_raises():
    adata, _ = _make_adata_with_topic_loadings()
    with pytest.raises(ValueError, match="feature_cat"):
        extractTopicSignaling(adata, feature_cat="foo")


def test_extractTopicSignaling_missing_topic_raises():
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.uniform(0, 1, (4, 5)))
    adata.uns["cellchat"] = {"net": {"centr.cell": {"outdeg": np.ones((4, 3))}}}
    with pytest.raises(ValueError, match="identifyCellTopics"):
        extractTopicSignaling(adata, pattern="incoming")


# ---------------------------------------------------------------------------
# identifyTopicCommunities — gracefully skip if _snn.py absent
# ---------------------------------------------------------------------------
_HAS_SNN = _imp_util.find_spec("spatialcellchat._snn") is not None


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_seurat_branch():
    """Smoke test: build an AnnData with topics + run Leiden on the SNN."""
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=24, nLR=6)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[3], maxit=200, seed_use=1)
    out = identifyTopicCommunities(
        adata, pattern="incoming", clustering_method="seurat",
        nPrograms=3, k_signaling=5, resolution=0.5, do_plot=False, seed_use=1,
    )
    assert "label.ICR" in out.obs.columns


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_missing_topic_raises():
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=10, nLR=5)
    with pytest.raises(ValueError, match="identifyCellTopics"):
        identifyTopicCommunities(adata, pattern="incoming")


_HAS_SNF = _imp_util.find_spec("spatialcellchat._snf") is not None


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_autoStardust_branch():
    """Run the autoStardust branch end-to-end (uses spatial coords)."""
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=24, nLR=6)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[3], maxit=200, seed_use=1)
    rng = np.random.default_rng(0)
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(24, 2))
    out = identifyTopicCommunities(
        adata, pattern="incoming", clustering_method="autoStardust",
        nPrograms=3, k_signaling=4, k_spatial=4,
        resolution=0.5, do_plot=False, seed_use=1,
    )
    assert "label.ICR" in out.obs.columns


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_weightStardust_branch():
    """Run the weightStardust branch end-to-end."""
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=24, nLR=6)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[3], maxit=200, seed_use=1)
    rng = np.random.default_rng(0)
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(24, 2))
    out = identifyTopicCommunities(
        adata, pattern="incoming", clustering_method="weightStardust",
        nPrograms=3, k_signaling=4, k_spatial=4, spatialWeight=0.5,
        resolution=0.5, do_plot=False, seed_use=1,
    )
    assert "label.ICR" in out.obs.columns


@pytest.mark.skipif(
    not (_HAS_SNN and _HAS_SNF),
    reason="spatialcellchat._snn or _snf not yet built",
)
def test_identifyTopicCommunities_fusion_branch():
    """Run the fusion branch (requires _snf from sub-agent A)."""
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=24, nLR=6)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[3], maxit=200, seed_use=1)
    rng = np.random.default_rng(0)
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(24, 2))
    out = identifyTopicCommunities(
        adata, pattern="incoming", clustering_method="fusion",
        nPrograms=3, k_signaling=4, k_spatial=4, k_fusion=4, t_fusion=5,
        resolution=0.5, do_plot=False, seed_use=1,
    )
    assert "label.ICR" in out.obs.columns


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_unknown_method_raises():
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=12, nLR=4)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[2], maxit=100, seed_use=1)
    with pytest.raises(ValueError, match="Unknown clustering_method"):
        identifyTopicCommunities(
            adata, pattern="incoming", clustering_method="not_a_real_method",
            k_signaling=3, resolution=0.5, do_plot=False, seed_use=1,
        )


@pytest.mark.skipif(not _HAS_SNN, reason="spatialcellchat._snn (sub-agent A) not yet built")
def test_identifyTopicCommunities_resolution_sweep_picks_best():
    """When `resolution` is None, the function should sweep `resRange`."""
    from spatialcellchat.analysis_topics import identifyTopicCommunities
    adata = _make_adata_with_centr_cell(nC=20, nLR=5)
    adata = identifyCellTopics(adata, pattern="incoming", topic_k=[2], maxit=100, seed_use=1)
    out = identifyTopicCommunities(
        adata, pattern="incoming", clustering_method="seurat",
        nPrograms=2, k_signaling=4, resolution=None,
        resRange=(0.1, 0.5, 1.0), do_plot=False, seed_use=1,
    )
    assert "label.ICR" in out.obs.columns
    res = out.uns["cellchat"]["net"]["topic"]["incoming"]["resolution"]
    assert res in (0.1, 0.5, 1.0)
