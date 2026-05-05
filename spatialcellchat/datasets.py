"""Tutorial-dataset loaders.

Currently exports ``load_visium_psoriasis()``: the human psoriatic skin Visium
dataset packaged with the original R tutorial. The h5ad file (~7.8 MB) is
downloaded on first use via the standard 3-tier resolver.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._download import resolve_data_path

__all__ = ["load_visium_psoriasis"]

_VISIUM_FILENAME = "visium_human_psoriasis.h5ad"
_DECOMP_RESOURCE = (
    Path(__file__).resolve().parent
    / "resources"
    / "visium_psoriasis_cell_type_decomposition.csv"
)


def load_visium_psoriasis() -> ad.AnnData:
    """Load the Visium human psoriatic-skin tutorial dataset.

    Returns
    -------
    anndata.AnnData
        905 spots × 33538 genes. ``adata.X`` holds **raw integer counts**
        (range 0–7332). The original Seurat object stores these in its
        ``data`` slot — the same slot R's tutorial subsequently runs
        normalisation on — and the ``counts`` slot is a 2×2 placeholder that
        was discarded by the upstream tutorial author. The Python tutorial
        therefore begins with ``scanpy.pp.normalize_total`` /
        ``scanpy.pp.log1p`` (or the package's ``normalizeData``) just like
        the R workflow does.
        ``adata.obs["active_ident"]`` is a categorical with the 6 cell-type
        labels used in the tutorial:
        ``FIB, ENDO, SM, EG, Keratinocyte, Immune``.
        ``adata.obsm["X_pca"]``, ``adata.obsm["X_umap"]`` and
        ``adata.obsm["spatial"]`` hold pre-computed embeddings and image-pixel
        spot coordinates. ``adata.uns["spatial"]["slice1"]`` follows the
        scanpy/squidpy Visium schema.

        The 905×6 cell-type-deconvolution matrix used by the original
        tutorial is bundled inside the package and attached to
        ``adata.obsm["predictions"]`` (and to ``adata.uns["predictions_columns"]``
        for the column names).
    """
    adata = ad.read_h5ad(resolve_data_path(_VISIUM_FILENAME))
    if "predictions" not in adata.obsm:
        ctd = pd.read_csv(_DECOMP_RESOURCE, index_col=0)
        ctd = ctd.reindex(index=adata.obs_names)
        adata.obsm["predictions"] = np.asarray(ctd.to_numpy(), dtype=float)
        adata.uns["predictions_columns"] = list(ctd.columns)
    return adata
