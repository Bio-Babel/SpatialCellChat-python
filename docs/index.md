# SpatialCellChat-python

Python port of the R [**SpatialCellChat**](https://github.com/jinworks/CellChat) package — inference, analysis, and visualization of cell-cell communication networks from spatial transcriptomics data.

`SpatialCellChat-python` is built on the [scverse](https://scverse.org) stack: AnnData / scanpy / squidpy / pydata sparse / esda / libpysal. CellChat IP (the `computeCommunProb` model and its expression-helper machinery) is reimplemented from R verbatim; non-CellChat math is delegated to upstream Python where parity exists, and ported inline where it does not.

## Quick start

```bash
pip install -e .
```

```python
import spatialcellchat as scc

# Load the bundled human Visium psoriatic-skin tutorial dataset
adata = scc.datasets.load_visium_psoriasis()
adata.obs["cellchat_idents"] = adata.obs["final.clusters"]

# Set the ligand-receptor database (Secreted + ECM + Cell-Cell Contact)
db = scc.load_db("human")
db_use = scc.subsetDB(db, search=["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"])
adata.uns["cellchat"] = {"DB": db_use, "spatial.factors": {"ratio": 0.18, "tol": 32.5}}

# Pre-process and identify over-expressed ligand-receptor pairs
scc.normalizeData(adata)
scc.subsetData(adata)
scc.preProcessing(adata)
scc.identifyOverExpressedGenes(adata, selection_method="meringue", do_grid=False)
scc.identifyOverExpressedInteractions(adata, variable_both=False)

# Compute cell-level and group-level communication probabilities
scc.computeCommunProb(adata, distance_use=True, scale_distance=0.2,
                      contact_dependent=True, interaction_range=250, contact_range=10)
scc.filterProbability(adata)
scc.filterCommunication(adata, min_links=10, min_cells_sr=10)
scc.computeAvgCommunProb_Visium(adata, cell_type_decomposition=adata.obsm["predictions"])
scc.aggregateNet(adata)

# Visualise — return matplotlib Figure for further composition
fig = scc.netVisual_aggregate(adata, signaling="TNF", layout="circle")
```

The [**SpatialCellChat analysis tutorial**](tutorials/SpatialCellChat_analysis_of_spatial_transcriptomics_data.ipynb) walks the full R-vignette workflow end-to-end (~6 minutes).

## What's implemented

- **CellChat IP — line-for-line.** `computeCommunProb` (Hill-transformed ligand × receptor with geometric-mean complex aggregation, agonist / antagonist / coreceptor terms, spatial-distance modulation, and `nboot` permutation-based p-values) ports verbatim from `R/modeling.R`. Validated to Pearson r = 1.0000 against R on the Visium psoriasis tutorial.
- **Full database surface.** CellChatDB.{human,mouse,zebrafish} bundled as parquet (~4.5 MB total). PPI matrices for human and mouse bundled as scipy CSR.
- **All 49 visualization exports.** Implemented via matplotlib + pycirclize + plotly. Three Seurat-thin shims (`dotPlot`, `StackedVlnPlot`, `plotGeneExpression`) delegate to `scanpy.pl.*`.
- **Tier-2 substitutions documented.** NMF (sklearn MU vs R Lee MU rule), bandwidth (Silverman vs `ks::hpi`), centrality (`networkx.current_flow_*` for `sna::flowbet` / `sna::infocent`). See [validation report](https://github.com/jinworks/CellChat).

See the [API Reference](api.md) for the full public surface.

## Container layout

The Python port replaces the R `SpatialCellChat` S4 class with `anndata.AnnData`. Everything CellChat-specific lives under `adata.uns["cellchat"]`:

| R `SpatialCellChat@` slot | AnnData |
|---|---|
| `data.raw` / `data` | `adata.layers["counts"]` / `adata.X` (normalised) |
| `data.signaling` | `adata.layers["data_signaling"]` (`subsetData` / `preProcessing`) |
| `data.scale` / `data.project` | `adata.layers["data_scale"]` / `adata.layers["data_project"]` |
| `images` | `adata.uns["spatial"][library_id]` (scanpy/squidpy Visium schema) |
| `meta` / `idents` | `adata.obs` / `adata.obs["cellchat_idents"]` (Categorical) |
| `dr[["umap"]]` / `dr[["spatial"]]` | `adata.obsm["X_umap"]` / `adata.obsm["spatial"]` |
| `var.features` | `adata.var["highly_variable_signaling"]` (bool) |
| `DB`, `LR`, `net`, `netP`, `options` | `adata.uns["cellchat"][{DB, LR, net, netP, options}]` |

Spatial graphs use `obsp["spatial_connectivities"]` + `["spatial_distances"]` (squidpy convention). `net["prob"]` and `netP["prob"]` are dense `(K, K, N)` arrays; `net["prob.cell"]` is `sparse.COO` (pydata sparse 0.18).

## Citation

If you use `SpatialCellChat-python` in your research, please cite the original SpatialCellChat paper (Cheng, Su & Jin) as well as the upstream Python libraries (anndata, scanpy, squidpy).

## License

GPL-3.0 (matches the upstream R package).
