"""End-to-end spatial CCC inference on a tiny synthetic Visium-like AnnData.

Maps onto workflows/spatial_ccc_inference_lowres.yaml's steps:

  1. normalizeData          -> preprocessing.normalizeData    (SKIPPED here; see note below)
  2. subsetData             -> preprocessing.subsetData       (SKIPPED here; see note below)
  3. preProcessing (ALRA)   -> preprocessing.preProcessing    (SKIPPED here; see note below)
  4. identifyOverExpressedGenes        -> de.identifyOverExpressedGenes        (SKIPPED here; see note below)
  5. identifyOverExpressedInteractions -> de.identifyOverExpressedInteractions (SKIPPED here; see note below)
  6. computeCommunProb                 -> modeling.computeCommunProb
  7. filterProbability                 -> modeling.filterProbability
  8. filterCommunication (cell-level)  -> modeling.filterCommunication
  9. computeAvgCommunProb_Visium       -> modeling_avg.computeAvgCommunProb_Visium
 10. filterCommunication (group-level) -> modeling.filterCommunication
 11. computeCommunProbPathway          -> modeling.computeCommunProbPathway
 12. aggregateNet                      -> modeling.aggregateNet
 13. netAnalysis_computeCentrality     -> analysis_network.netAnalysis_computeCentrality

Note on the skipped steps (1-5): ALRA rank selection (preProcessing) needs
enough singular values to separate signal from noise and MERINGUE's
spatial-autocorrelation test needs enough spots for a stable null — neither
is well-posed on a ~30-cell/25-gene toy matrix. This template instead
builds the post-preprocessing state directly (a small CellChatDB subset,
`LR.LRsig`, `spatial.factors`) exactly as the package's own integration
test does (tests/test_integration_communprob_avg.py), then exercises the
real, unmodified inference chain from computeCommunProb onward. On real
data, run steps 1-5 for real; only the synthetic-input shortcut is
template-specific.
"""

from __future__ import annotations


def main() -> None:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import spatialcellchat as scc

    rng = np.random.default_rng(0)

    # ---- Build a tiny synthetic Visium-like AnnData -----------------------
    db = scc.load_db("human")
    interaction = db["interaction"]
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

    n_obs = 30
    ident_levels = ["A", "B", "C"]
    obs = pd.DataFrame(index=[f"spot{i:03d}" for i in range(n_obs)])
    obs["cellchat_idents"] = pd.Categorical(
        rng.choice(ident_levels, size=n_obs), categories=ident_levels
    )
    X = rng.gamma(0.5, 1.0, size=(n_obs, len(var_names))).astype("float32")
    adata = ad.AnnData(
        X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=var_names)
    )
    adata.obsm["spatial"] = rng.uniform(0, 600, size=(n_obs, 2))

    # Per-spot cell-type proportions (rows sum ~1); this is what a real
    # deconvolution (RCTD/spotlight/cell2location) would provide.
    decomp = pd.DataFrame(
        rng.random((n_obs, len(ident_levels))),
        columns=ident_levels,
        index=adata.obs_names,
    )
    decomp = decomp.div(decomp.sum(axis=1), axis=0)

    # ---- Register CellChat state (mirrors the tutorial's manual assignment) ----
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

    # ---- Steps 6-13: the real, unmodified inference chain ------------------
    scc.computeCommunProb(
        adata,
        group_by="cellchat_idents",
        scale_distance=5.0,
        contact_dependent_forced=False,
    )
    scc.filterProbability(adata, nboot=20, thresh=0.5)
    scc.filterCommunication(adata, min_cells=None, min_links=1, min_cells_sr=1)
    scc.computeAvgCommunProb_Visium(
        adata,
        cell_type_decomposition=decomp,
        avg_type="avg",
        group_by="cellchat_idents",
        nboot=20,
        do_permutation=False,
    )
    scc.filterCommunication(adata, min_cells=1, min_links=None, min_cells_sr=None)
    scc.computeCommunProbPathway(adata)
    scc.aggregateNet(adata)
    scc.netAnalysis_computeCentrality(
        adata, slot_name="net", do_group=False, degree_only=True
    )

    net = adata.uns["cellchat"]["net"]
    print(f"net.count shape: {net['count'].shape}")
    print(f"net.weight shape: {net['weight'].shape}")
    print(
        "centr.cell measures:",
        net["centr.cell"]["centr_names"],
    )


if __name__ == "__main__":
    main()
