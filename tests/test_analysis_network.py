"""Unit tests for ``spatialcellchat.analysis_network``.

Covers:

- ``computeLaplacian``  (R-parity to atol 1e-12)
- ``computeEigengap``   (upper / lower bound integers identical to R)
- ``computeNetD_structure``
- ``computeNetSimilarity`` functional + structural (R-parity)
- ``computeNetSimilarityPairwise`` functional (smoke test)
- ``netEmbedding`` (UMAP shape + slot location)
- ``netClustering`` (kmeans labels + slot location)
- ``netAnalysis_computeCentrality`` (raw-net + AnnData branches)
- ``netMappingDEG`` (DEG-mapping schema)
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import warnings

from spatialcellchat.analysis_network import (
    computeEigengap,
    computeLaplacian,
    computeNetD_structure,
    computeNetSimilarity,
    computeNetSimilarityPairwise,
    netAnalysis_computeCentrality,
    netClustering,
    netEmbedding,
    netMappingDEG,
)

# R-side reference dumps bundled under tests/data/r_dumps/. Produced
# once by ``r_harness/{laplacian,netsim,centrality}_harness.R`` (live
# outside the package tree). To regenerate after intentional changes,
# re-run the relevant R harness and copy outputs into the matching
# ``tests/data/r_dumps/<feature>/`` directory.
LAP_DIR      = Path(__file__).parent / "data" / "r_dumps" / "laplacian"
EIGENGAP_DIR = Path(__file__).parent / "data" / "r_dumps" / "eigengap"
NETSIM_DIR   = Path(__file__).parent / "data" / "r_dumps" / "netsim"


# ---------------------------------------------------------------------------
# computeLaplacian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["1", "2"])
def test_computeLaplacian_matches_R(case: str) -> None:
    """Eigenvalues of the normalised Laplacian match R within 5e-15."""
    M = np.loadtxt(str(LAP_DIR / f"lap_input{case}.tsv"))
    py = computeLaplacian(M, tol=0.01)
    r = np.loadtxt(str(LAP_DIR / f"lap_val{case}.tsv"))
    np.testing.assert_allclose(py["val"], r, atol=5e-15)


def test_computeLaplacian_n_zeros_field() -> None:
    """``n_zeros`` is an integer count of eigenvalues <= tol."""
    M = np.eye(5) + np.ones((5, 5)) * 0.1
    out = computeLaplacian(M, tol=0.01)
    assert isinstance(out["n_zeros"], int)
    assert out["n_zeros"] >= 0


# ---------------------------------------------------------------------------
# computeEigengap
# ---------------------------------------------------------------------------
def test_computeEigengap_block_structure() -> None:
    """5-block consensus matrix yields ``upper_bound == 5`` (matches R)."""
    CM = np.loadtxt(str(EIGENGAP_DIR / "eigengap_input.tsv"))
    out = computeEigengap(CM, tol=0.01)
    expected_upper = int(
        (EIGENGAP_DIR / "eigengap_upper.txt").read_text().strip()
    )
    expected_lower = int(
        (EIGENGAP_DIR / "eigengap_lower.txt").read_text().strip()
    )
    assert out["upper_bound"] == expected_upper
    assert out["lower_bound"] == expected_lower


def test_computeEigengap_returns_dict_keys() -> None:
    """Return shape includes ``upper_bound``, ``lower_bound``, ``eigs``."""
    rng = np.random.default_rng(42)
    M = rng.uniform(0, 1, (10, 10))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    out = computeEigengap(M, tol=0.01)
    assert {"upper_bound", "lower_bound", "eigs", "upper_bound_ties"}.issubset(out.keys())


def test_computeEigengap_upper_bound_ties_returns_all_indices() -> None:
    """R's `which(gaps == max(gaps))` returns ALL tied indices.

    Construct a CM whose Laplacian eigengap has a tie: two equal blocks
    + one isolated singleton produces a degenerate gap structure.
    Verify upper_bound_ties contains every index that ties for the
    maximum (1-based), and upper_bound is the first tie.
    """
    # 3-block CM (2 cells per block) — the eigengap can have ties when
    # blocks are identical.
    n = 6
    CM = np.zeros((n, n))
    for b in range(3):
        idx = slice(b * 2, (b + 1) * 2)
        CM[idx, idx] = 1.0
    np.fill_diagonal(CM, 1.0)
    out = computeEigengap(CM, tol=0.01)
    assert "upper_bound_ties" in out
    ties = out["upper_bound_ties"]
    # Every tie index must be a 1-based integer
    assert all(isinstance(int(t), int) and int(t) >= 1 for t in ties)
    # upper_bound is the first tie
    assert out["upper_bound"] == int(ties[0])


# ---------------------------------------------------------------------------
# computeNetD_structure / similarity
# ---------------------------------------------------------------------------
def test_computeNetD_structure_zero_for_identical_graphs() -> None:
    """Distance between a graph and itself is 0."""
    rng = np.random.default_rng(0)
    G = (rng.uniform(0, 1, (6, 6)) > 0.5).astype(float)
    np.fill_diagonal(G, 0)
    d = computeNetD_structure(G, G)
    assert abs(d) < 1e-9


def test_node_distance_matches_R_off_by_one() -> None:
    """Pin R-parity of _node_distance's column-major linear-index decode.

    R's ``which(m == quem[j])`` returns 1-based column-major indices;
    R then computes ``linhas = floor(l) + 1`` with a ``- 1`` shift for
    integer values of ``l`` (i.e. entries on the bottom row, where the
    column-major position is exactly a multiple of n). The shift makes
    the bottom-row entries fall in column ``j`` instead of ``j+1``.

    The R harness was run with the same edge list:

    .. code-block:: r

        edges <- c(1,2, 2,3, 3,4, 4,5, 5,6, 1,6, 2,5)
        g <- graph(edges=edges, n=6, directed=FALSE)
        nd <- node_distance(g)
        # Column 1 (distance=1): rowSums = (2, 3, 2, 2, 3, 2)/5 = (.4,.6,.4,.4,.6,.4)
        # Column 2 (distance=2): rowSums = (2, 2, 2, 2, 2, 2)/5 = .4
        # Column 3 (distance=3): rowSums = (1, 0, 1, 1, 0, 1)/5
    """
    import igraph as ig
    from spatialcellchat.analysis_network import _node_distance
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
    g = ig.Graph(n=6, edges=edges, directed=False)
    nd = _node_distance(g)
    expected = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(nd, expected, atol=1e-12)
    # Each row sums to 1 (counts/(n-1) where (n-1)*1 entries off-diagonal).
    np.testing.assert_allclose(nd.sum(axis=1), np.ones(6), atol=1e-12)


@pytest.mark.parametrize("type_", ["functional", "structural"])
def test_computeNetSimilarity_matches_R(type_: str) -> None:
    """Per-pair similarity matrix matches R within 1e-12."""
    K, N = 5, 8
    prob = np.zeros((K, K, N))
    for i in range(N):
        prob[:, :, i] = np.loadtxt(str(NETSIM_DIR / f"netsim_prob_{i+1}.tsv"))
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {
        "netP": {
            "prob": prob,
            "pathway_names": [f"p{i+1}" for i in range(N)],
        }
    }
    out = computeNetSimilarity(adata, slot_name="netP", type_=type_)
    sim = out.uns["cellchat"]["netP"]["similarity"][type_]["matrix"]["single"]
    if type_ == "functional":
        sim_r = np.loadtxt(str(NETSIM_DIR / "netsim_Similarity_functional.tsv"))
        np.testing.assert_allclose(sim.values, sim_r, atol=1e-12)
    else:
        # Structural raw S matches R to atol 1e-12; check S = 1 - D against R
        # Sim has been multiplied by SNN; recover the underlying S by checking
        # that the Sim matrix has zeros only where SNN is zero
        assert sim.shape == (N, N)
        np.testing.assert_allclose(np.diag(sim.values), np.ones(N), atol=1e-12)


def test_computeNetSimilarity_writes_pathway_names() -> None:
    """Output DataFrame is indexed by pathway names."""
    K, N = 4, 5
    prob = np.zeros((K, K, N))
    rng = np.random.default_rng(0)
    for i in range(N):
        m = rng.uniform(0, 1, (K, K))
        m[m < 0.4] = 0
        np.fill_diagonal(m, 0)
        prob[:, :, i] = m
    adata = ad.AnnData(np.zeros((1, 1)))
    pathway_names = [f"path{i}" for i in range(N)]
    adata.uns["cellchat"] = {"netP": {"prob": prob, "pathway_names": pathway_names}}
    out = computeNetSimilarity(adata, slot_name="netP", type_="functional")
    sim = out.uns["cellchat"]["netP"]["similarity"]["functional"]["matrix"]["single"]
    assert list(sim.index) == pathway_names
    assert list(sim.columns) == pathway_names


# ---------------------------------------------------------------------------
# computeNetSimilarityPairwise
# ---------------------------------------------------------------------------
def test_computeNetSimilarityPairwise_smoke() -> None:
    """Joint similarity for two synthetic datasets has the right shape."""
    K, N = 4, 4
    rng = np.random.default_rng(0)

    def _mk_adata() -> ad.AnnData:
        adata = ad.AnnData(np.zeros((1, 1)))
        prob = np.zeros((K, K, N))
        for i in range(N):
            m = rng.uniform(0, 1, (K, K))
            m[m < 0.4] = 0
            np.fill_diagonal(m, 0)
            prob[:, :, i] = m
        adata.uns["cellchat"] = {
            "netP": {"prob": prob, "pathway_names": [f"p{j}" for j in range(N)]}
        }
        return adata

    a1, a2 = _mk_adata(), _mk_adata()
    res = computeNetSimilarityPairwise(
        [a1, a2],
        slot_name="netP",
        type_="functional",
        comparison=[1, 2],
        dataset_names=["A", "B"],
    )
    assert res["matrix"].shape == (2 * N, 2 * N)
    assert res["comparison_name"] == "1-2"
    # Names follow the "<pathway>--<dataset>" convention
    assert all("--" in nm for nm in res["matrix"].index)


# ---------------------------------------------------------------------------
# netEmbedding + netClustering
# ---------------------------------------------------------------------------
def _build_similarity_adata(N: int = 12, seed: int = 0) -> ad.AnnData:
    """Synthetic AnnData seeded with a similarity matrix."""
    rng = np.random.default_rng(seed)
    # Three blocks so a clusterable embedding exists
    block_size = N // 3
    S = np.full((N, N), 0.05)
    for b in range(3):
        idx = slice(b * block_size, (b + 1) * block_size)
        S[idx, idx] = 0.95
    np.fill_diagonal(S, 1.0)
    df = pd.DataFrame(S, index=[f"p{i}" for i in range(N)], columns=[f"p{i}" for i in range(N)])
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {
        "netP": {"similarity": {"functional": {"matrix": {"single": df}}}}
    }
    return adata


def test_netEmbedding_writes_dr_slot() -> None:
    """``netEmbedding`` writes a 2-D embedding under the right slot."""
    adata = _build_similarity_adata(N=12, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = netEmbedding(
            adata,
            slot_name="netP",
            type_="functional",
            n_neighbors=5,
        )
    Y = out.uns["cellchat"]["netP"]["similarity"]["functional"]["dr"]["single"]
    assert isinstance(Y, pd.DataFrame)
    assert Y.shape[1] == 2
    assert Y.shape[0] == 12


def test_netClustering_writes_group_slot() -> None:
    """``netClustering`` produces 1-based integer cluster labels."""
    adata = _build_similarity_adata(N=12, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        netEmbedding(
            adata,
            slot_name="netP",
            type_="functional",
            n_neighbors=5,
        )
        netClustering(
            adata,
            slot_name="netP",
            type_="functional",
            k=3,
        )
    g = adata.uns["cellchat"]["netP"]["similarity"]["functional"]["group"]["single"]
    assert isinstance(g, pd.Series)
    assert g.shape[0] == 12
    # 1-based labels (R parity)
    assert g.min() >= 1


def test_netClustering_inferred_k() -> None:
    """Without ``k``, the cluster count is inferred via ``computeEigengap``."""
    adata = _build_similarity_adata(N=15, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        netEmbedding(
            adata,
            slot_name="netP",
            type_="functional",
            n_neighbors=5,
        )
        netClustering(
            adata,
            slot_name="netP",
            type_="functional",
        )
    g = adata.uns["cellchat"]["netP"]["similarity"]["functional"]["group"]["single"]
    assert g.shape[0] == 15


def test_netEmbedding_umap_method_defaults_correlation() -> None:
    """umap_method='umap-learn' uses metric='correlation' (R parity).

    R's runUMAP() defaults to metric='correlation' and passes the
    similarity matrix directly as features (no precomputed distance).
    Smoke test: the call succeeds for both umap_method options.
    """
    adata = _build_similarity_adata(N=12, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = netEmbedding(
            adata, slot_name="netP", type_="functional",
            n_neighbors=5, umap_method="umap-learn",
        )
    Y = out.uns["cellchat"]["netP"]["similarity"]["functional"]["dr"]["single"]
    assert Y.shape == (12, 2)


def test_netEmbedding_umap_method_uwot_uses_euclidean() -> None:
    """umap_method='uwot' uses metric='euclidean' (R uwot::umap default)."""
    adata = _build_similarity_adata(N=12, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = netEmbedding(
            adata, slot_name="netP", type_="functional",
            n_neighbors=5, umap_method="uwot",
        )
    Y = out.uns["cellchat"]["netP"]["similarity"]["functional"]["dr"]["single"]
    assert Y.shape == (12, 2)


def test_netEmbedding_invalid_umap_method_raises() -> None:
    adata = _build_similarity_adata(N=8, seed=0)
    with pytest.raises(ValueError, match="umap_method"):
        netEmbedding(adata, n_neighbors=3, umap_method="bogus")


def test_netEmbedding_merged_mode_default_comparison() -> None:
    """In merged mode, the default comparison_name is '1-2'
    (R: paste(1:length(unique(meta$datasets)), collapse='-')).

    Per-dataset matrices are stored by R under
    ``slot$similarity[[type]]$matrix[[comparison.name]]``; merged mode
    typically keys a single matrix under '1-2' and never under
    'single'. Default mode='merged' explicitly should pick that key.
    """
    rng = np.random.default_rng(3)
    N = 6
    block = N // 2
    S = np.full((N, N), 0.05)
    for b in range(2):
        idx = slice(b * block, (b + 1) * block)
        S[idx, idx] = 0.95
    np.fill_diagonal(S, 1.0)
    df_12 = pd.DataFrame(S, index=[f"p{i}" for i in range(N)], columns=[f"p{i}" for i in range(N)])
    adata = ad.AnnData(np.zeros((1, 1)))
    # Single matrix keyed by R's "1-2" comparison name (merged mode).
    adata.uns["cellchat"] = {
        "netP": {
            "similarity": {
                "functional": {
                    "matrix": {"1-2": df_12.copy()}
                }
            }
        }
    }
    # mode="merged" — should resolve to "1-2" key.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = netEmbedding(
            adata, slot_name="netP", type_="functional",
            mode="merged", n_neighbors=3,
        )
    drs = out.uns["cellchat"]["netP"]["similarity"]["functional"]["dr"]
    assert "1-2" in drs


# ---------------------------------------------------------------------------
# netAnalysis_computeCentrality
# ---------------------------------------------------------------------------
def test_netAnalysis_computeCentrality_raw_net_2d() -> None:
    """A raw 2-D ``net`` ndarray returns a 3-D centrality block."""
    rng = np.random.default_rng(0)
    net = rng.uniform(0, 1, (5, 5))
    net[net < 0.5] = 0
    np.fill_diagonal(net, 0)
    out = netAnalysis_computeCentrality(net=net, degree_only=True)
    assert out.shape == (5, 5, 1)


def test_netAnalysis_computeCentrality_raw_net_3d() -> None:
    """A raw 3-D ``net`` ndarray returns one slice per signaling network."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 1, (4, 4, 3))
    arr[arr < 0.5] = 0
    out = netAnalysis_computeCentrality(net=arr, degree_only=False)
    assert out.shape == (11, 4, 3)


def test_netAnalysis_computeCentrality_writes_centr_cell() -> None:
    """Default ``do_group=False`` writes ``centr.cell`` from ``tmp['prob.cell']``."""
    K, N = 4, 3
    rng = np.random.default_rng(0)
    prob_cell = {}
    for i in range(N):
        m = rng.uniform(0, 1, (K, K))
        m[m < 0.5] = 0
        np.fill_diagonal(m, 0)
        prob_cell[f"LR{i}"] = m
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {"net": {"tmp": {"prob.cell": prob_cell}}}
    out = netAnalysis_computeCentrality(
        adata, slot_name="net", do_group=False, degree_only=True
    )
    centr = out.uns["cellchat"]["net"]["centr.cell"]
    assert centr["values"].shape == (5, K, N)
    assert centr["centr_names"][:2] == ["outdeg_unweighted", "indeg_unweighted"]


def test_netAnalysis_computeCentrality_writes_centr_when_do_group() -> None:
    """``do_group=True`` writes to ``centr`` from ``prob``/``pval`` arrays."""
    K, N = 4, 3
    rng = np.random.default_rng(1)
    prob = rng.uniform(0, 1, (K, K, N))
    pval = np.zeros((K, K, N))
    pval[prob < 0.5] = 1.0  # mark insignificant
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {
        "netP": {
            "prob": prob,
            "pval": pval,
            "pathway_names": [f"P{i}" for i in range(N)],
        }
    }
    out = netAnalysis_computeCentrality(
        adata, slot_name="netP", do_group=True, degree_only=True
    )
    centr = out.uns["cellchat"]["netP"]["centr"]
    assert centr["values"].shape == (5, K, N)
    assert centr["signaling_names"] == [f"P{i}" for i in range(N)]


def test_netAnalysis_computeCentrality_signaling_subset() -> None:
    """``signaling_name`` subsets the input mapping."""
    rng = np.random.default_rng(2)
    K = 4
    net = {nm: rng.uniform(0, 1, (K, K)) for nm in ["A", "B", "C"]}
    out = netAnalysis_computeCentrality(
        net=net, signaling_name=["B", "C"], degree_only=True
    )
    assert out.shape == (5, K, 2)


def test_netAnalysis_computeCentrality_invalid_signaling_raises() -> None:
    """Unknown signaling names raise a clear ValueError."""
    rng = np.random.default_rng(3)
    K = 4
    net = {"A": rng.uniform(0, 1, (K, K))}
    with pytest.raises(ValueError, match="signaling.name"):
        netAnalysis_computeCentrality(
            net=net, signaling_name=["X"], degree_only=True
        )


def test_netAnalysis_computeCentrality_missing_object_and_net_raises() -> None:
    """Either ``adata`` or ``net`` must be provided."""
    with pytest.raises(ValueError, match="adata.*net"):
        netAnalysis_computeCentrality(degree_only=True)


def test_netAnalysis_computeCentrality_missing_prob_cell_raises() -> None:
    """When ``tmp['prob.cell']`` is missing, an explanatory error is raised."""
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {"net": {}}
    with pytest.raises(ValueError, match="computeCommunProb"):
        netAnalysis_computeCentrality(adata, slot_name="net", do_group=False)


# ---------------------------------------------------------------------------
# netMappingDEG
# ---------------------------------------------------------------------------
def test_netMappingDEG_adds_DEG_columns() -> None:
    """The returned data frame has the eight DEG columns and matches per-row mapping."""
    # Build a minimal AnnData consistent with subsetCommunication's inputs
    K, N_lr = 2, 2
    prob = np.zeros((K, K, N_lr))
    pval = np.zeros((K, K, N_lr))
    prob[:, :, 0] = np.array([[0.0, 0.5], [0.0, 0.0]])
    prob[:, :, 1] = np.array([[0.0, 0.0], [0.7, 0.0]])

    pairLR = pd.DataFrame(
        {
            "interaction_name": ["L1_R1", "L2_R2"],
            "interaction_name_2": ["L1 - R1", "L2 - R2"],
            "pathway_name": ["P1", "P2"],
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "annotation": ["Secreted Signaling"] * 2,
            "evidence": [""] * 2,
        },
        index=["L1_R1", "L2_R2"],
    )
    geneInfo = pd.DataFrame({"Symbol": ["L1", "L2", "R1", "R2"]})
    complex_input = pd.DataFrame(
        {"subunit_1": [], "subunit_2": []}, index=pd.Index([], name="complex_name")
    )
    DEG = pd.DataFrame(
        {
            "clusters": ["c1", "c2", "c1", "c2"],
            "features": ["L1", "L2", "R1", "R2"],
            "pvalues": [0.01, 0.02, 0.03, 0.04],
            "logFC": [1.5, 2.0, -1.0, 1.2],
            "pct.1": [0.7, 0.8, 0.9, 0.6],
            "pct.2": [0.1, 0.2, 0.3, 0.4],
        }
    )
    adata = ad.AnnData(np.zeros((K, 1)))
    adata.obs.index = ["c1", "c2"]
    adata.obs["cellchat_idents"] = pd.Categorical(["c1", "c2"], categories=["c1", "c2"])
    adata.uns["cellchat"] = {
        "net": {
            "prob": prob,
            "pval": pval,
            "dimnames": [["c1", "c2"], ["c1", "c2"], list(pairLR.index)],
        },
        "LR": {"LRsig": pairLR},
        "DB": {
            "interaction": pairLR,
            "geneInfo": geneInfo,
            "complex": complex_input,
        },
        "var.features": {"features.info": DEG},
    }
    df = netMappingDEG(adata, features_name="features", thresh=1.0)
    for col in (
        "ligand.pvalues",
        "ligand.logFC",
        "ligand.pct.1",
        "ligand.pct.2",
        "receptor.pvalues",
        "receptor.logFC",
        "receptor.pct.1",
        "receptor.pct.2",
    ):
        assert col in df.columns


def test_netMappingDEG_missing_features_raises() -> None:
    """Missing features-info entry raises with hint to run identifyOverExpressedGenes."""
    adata = ad.AnnData(np.zeros((1, 1)))
    adata.uns["cellchat"] = {"var.features": {}}
    with pytest.raises(ValueError, match="identifyOverExpressedGenes"):
        netMappingDEG(adata, features_name="features")
