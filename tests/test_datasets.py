"""Smoke tests for spatialcellchat.datasets."""

from __future__ import annotations

import anndata as ad

from spatialcellchat import datasets


def test_load_visium_psoriasis_shape_and_keys():
    adata = datasets.load_visium_psoriasis()
    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (905, 33538)
    # Required obs / obsm / uns slots for downstream cellchat work.
    assert "active_ident" in adata.obs.columns
    assert list(adata.obs["active_ident"].cat.categories) == [
        "FIB",
        "ENDO",
        "SM",
        "EG",
        "Keratinocyte",
        "Immune",
    ]
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape == (905, 2)
    assert "X_pca" in adata.obsm and adata.obsm["X_pca"].shape == (905, 50)
    assert "X_umap" in adata.obsm and adata.obsm["X_umap"].shape == (905, 2)
    spatial = adata.uns["spatial"]
    assert "slice1" in spatial
    sf = spatial["slice1"]["scalefactors"]
    for key in (
        "spot_diameter_fullres",
        "fiducial_diameter_fullres",
        "tissue_hires_scalef",
        "tissue_lowres_scalef",
    ):
        assert key in sf, key
