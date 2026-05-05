"""Integration smoke test: computeCommunProb → computeAvgCommunProb → aggregateNet.

Verifies that the slot contract between Slice 4 sub-agent A's
``computeCommunProb`` (writes ``adata.uns['cellchat']['net']['tmp']``) and
sub-agent B's ``computeAvgCommunProb`` (reads the same slots) is honoured.

Also verifies aggregateNet's no-filter branch finds the populated
``net['prob']`` once computeAvgCommunProb has written it.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import spatialcellchat as scc


def _build_minimal_cellchat_adata(seed: int = 0) -> ad.AnnData:
    """Build a 30-cell × ~25-gene AnnData with a small CellChatDB-overlapping
    gene set, 3 cell-type idents, and minimal `uns["cellchat"]` plumbing.
    """
    rng = np.random.default_rng(seed)
    db = scc.load_db("human")
    interaction = db["interaction"]
    # Pick 3 secreted-signaling pairs whose ligand+receptor are simple genes
    secreted = interaction[interaction["annotation"] == "Secreted Signaling"]
    chosen = secreted[
        ~secreted["ligand"].str.contains("_", na=False)
        & ~secreted["receptor"].str.contains("_", na=False)
    ].head(3)
    LR_genes = pd.Series(
        chosen["ligand"].tolist() + chosen["receptor"].tolist()
    ).drop_duplicates().tolist()
    extra_genes = [f"BG{i:03d}" for i in range(20)]
    var_names = LR_genes + extra_genes
    n_genes = len(var_names)

    n_obs = 30
    obs_names = [f"cell{i:03d}" for i in range(n_obs)]
    X = rng.gamma(0.5, 1.0, size=(n_obs, n_genes)).astype("float32")
    obs = pd.DataFrame(index=obs_names)
    obs["cellchat_idents"] = pd.Categorical(
        rng.choice(["A", "B", "C"], size=n_obs),
        categories=["A", "B", "C"],
    )
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(n_obs, 2))

    adata.uns["cellchat"] = {
        "DB": {
            "interaction": chosen,
            "complex": db["complex"],
            "cofactor": db["cofactor"],
            "geneInfo": db["geneInfo"],
        },
        "LR": {"LRsig": chosen},
        "spatial.factors": {"ratio": 1.0, "tol": 1.0},
    }
    return adata


def test_communprob_then_avg_then_aggregate():
    """End-to-end: A's computeCommunProb populates the slots B's
    computeAvgCommunProb expects, and the resulting net["prob"] lets
    aggregateNet's no-filter branch run without error."""
    adata = _build_minimal_cellchat_adata(seed=42)

    scc.computeCommunProb(
        adata,
        group_by="cellchat_idents",
        scale_distance=5.0,
        contact_dependent_forced=False,
    )

    net = adata.uns["cellchat"]["net"]
    # A's writes:
    assert "tmp" in net, "Slice 4A should write net['tmp']"
    tmp = net["tmp"]
    assert "prob.cell" in tmp, "tmp['prob.cell'] missing — sub-agent A contract"
    assert "Lavg" in tmp, "tmp['Lavg'] missing"
    assert "Ravg" in tmp, "tmp['Ravg'] missing"
    n_LR = len(adata.uns["cellchat"]["LR"]["LRsig"])
    assert tmp["Lavg"].shape[0] == n_LR
    assert tmp["Ravg"].shape[0] == n_LR

    # B's read + write:
    scc.computeAvgCommunProb(
        adata,
        group_by="cellchat_idents",
        do_permutation=False,
    )
    assert "prob" in adata.uns["cellchat"]["net"], "Slice 4B should write net['prob']"
    prob = adata.uns["cellchat"]["net"]["prob"]
    assert prob.shape[0] == 3 and prob.shape[1] == 3 and prob.shape[2] == n_LR

    # A's aggregateNet no-filter branch reads net['prob']:
    scc.aggregateNet(adata)
    assert "count" in adata.uns["cellchat"]["net"]
    assert adata.uns["cellchat"]["net"]["count"].shape == (3, 3)
