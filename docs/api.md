# API Reference

`SpatialCellChat-python` mirrors the R package's public surface (`camelCase` / `snake.case` preserved) so that translated R tutorials remain recognisable. The full list of public exports is grouped below by R source file; full docstrings (NumPy style) are auto-rendered via `mkdocstrings`.

## Database

::: spatialcellchat.database
    options:
      members:
        - load_db
        - load_ppi
        - subsetDB
        - searchPair
        - extractGene
        - extractGeneSubset
        - extractGeneSubsetFromPair
        - checkGeneSymbol
        - updateCellChatDB

## Datasets

::: spatialcellchat.datasets

## Preprocessing

::: spatialcellchat.preprocessing

## Differential expression / over-expressed identifiers

::: spatialcellchat.de

## Spatial geometry

::: spatialcellchat.spatial

## CellChat IP — communication probability

::: spatialcellchat.modeling

::: spatialcellchat.modeling_avg

::: spatialcellchat._expr
    options:
      heading_level: 3

## Network analysis

::: spatialcellchat.analysis_network

::: spatialcellchat.analysis_topics

::: spatialcellchat.analysis_ranking

## Visualization

### Basic plots

::: spatialcellchat.viz_basic

### Heatmaps + embeddings + signaling-role

::: spatialcellchat.viz_heatmap

::: spatialcellchat.viz_embedding

### Spatial scatter / fields / pies

::: spatialcellchat.viz_spatial

### Network graphs / chord / dispatchers

::: spatialcellchat.viz_network

### Plotly (3D / interactive)

::: spatialcellchat.viz_plotly

### scverse delegated (sc.pl shims)

::: spatialcellchat.viz_delegated

## Themes & palettes

::: spatialcellchat._palette

::: spatialcellchat._themes

## Utilities

::: spatialcellchat._utils

::: spatialcellchat._hill
