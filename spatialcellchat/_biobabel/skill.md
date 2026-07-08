---
name: use-spatialcellchat
description: Reach for spatialcellchat when inferring, ranking, or visualizing cell-cell communication (ligand-receptor signaling) from spatial transcriptomics AnnData (10X Visium, Slide-seqV2, Stereo-seq, MERFISH).
---

# spatialcellchat

Python port of the R `SpatialCellChat` package. It models the probability of
cell-cell communication by combining gene expression (ligand/receptor,
geometric-mean-aggregated, Hill-function transformed) with spatial distance
between cells/spots and the curated `CellChatDB` ligand-receptor database.

Unlike the R port's dedicated S4 object, this package keeps its state inside
the `AnnData` you already have: **all CellChat state (database, communication
networks, centrality scores, NMF topics, ...) lives under
`adata.uns["cellchat"]`**, a plain nested dict, while `adata.obs` / `adata.var`
/ `adata.obsm` / `adata.layers` keep their conventional scverse roles. Most
functions take `adata` as the first argument, mutate `adata.uns["cellchat"]`
in place, and return the same object — a few (`subsetData`, `sketchData`,
`load_visium_psoriasis`, `subsetCommunication`, the DB-loading functions)
instead return a fresh object/DataFrame.

## When to use / when not to use

Use it when the task is inferring or visualizing ligand-receptor signaling
from **spatial** transcriptomics data. Do not reach for it for non-spatial
single-cell CCC (use the non-spatial CellChat lineage), for general
expression analysis with no communication question, or when there are no
spatial coordinates at all.

## Entry points

- `load_db` / `subsetDB` — load and filter `CellChatDB` (human/mouse).
- `normalizeData`, `subsetData`, `preProcessing` — expression preprocessing
  down to the DB gene set, with ALRA imputation.
- `identifyOverExpressedGenes`, `identifyOverExpressedInteractions` — find
  spatially variable genes and the significant L-R pairs built from them.
- `computeCommunProb` → `filterProbability`/`filterCommunication` →
  `computeAvgCommunProb_Visium` (low-res) or `computeAvgCommunProb` (high-res)
  → `computeCommunProbPathway` → `aggregateNet` — the core inference pipeline.
- `netAnalysis_computeCentrality`, `rankNet`, `identifyCellTopics` — downstream
  network/topic analysis.
- `netVisual_aggregate`, `netVisual_heatmap`, `spatialFeaturePlot`,
  `spatialGiPlot`, `spatialLeePlot`, `spatialTopicPlot` — visualization.

## Minimal usage sketch

```python
import spatialcellchat as scc
from spatialcellchat.datasets import load_visium_psoriasis

adata = load_visium_psoriasis()
adata.obs["cellchat_idents"] = adata.obs["final.clusters"]
adata.uns.setdefault("cellchat", {})
adata.uns["cellchat"]["spatial.factors"] = {"ratio": 0.379, "tol": 32.5}

db_use = scc.subsetDB(scc.load_db("human"), search=["Secreted Signaling"])
adata.uns["cellchat"]["DB"] = db_use

scc.normalizeData(adata)
sub = scc.subsetData(adata, db=db_use)
sub.uns["cellchat"] = adata.uns["cellchat"]  # subsetData returns a fresh AnnData
scc.preProcessing(sub)
scc.identifyOverExpressedGenes(sub, selection_method="meringue")
scc.identifyOverExpressedInteractions(sub, db=db_use)

scc.computeCommunProb(sub, distance_use=True, contact_dependent=True)
scc.filterProbability(sub)
scc.filterCommunication(sub, min_links=10, min_cells_sr=10)
scc.aggregateNet(sub)
```

For more: `biobabel.describe_package(import_name="spatialcellchat")`.
