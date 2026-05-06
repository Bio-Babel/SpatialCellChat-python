"""Builders for pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_tiny_visium(out: Path) -> None:
    """Build a deterministic 15-spot × 50-gene Visium-shape AnnData.

    Used by tests that need a non-trivial AnnData with spatial coords, an
    image-like ``uns["spatial"]`` block, and ident labels — but not with the
    full ~1000-gene CellChat universe.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    n_obs, n_var = 15, 50
    X = sp.csr_matrix(rng.poisson(2.0, size=(n_obs, n_var)).astype("float32"))
    obs_names = [f"spot{i:03d}" for i in range(n_obs)]
    var_names = [f"gene{i:03d}" for i in range(n_var)]

    obs = pd.DataFrame(index=obs_names)
    obs["active_ident"] = pd.Categorical(
        rng.choice(["A", "B", "C"], size=n_obs),
        categories=["A", "B", "C"],
    )
    obs["nCount_Spatial"] = np.asarray(X.sum(axis=1)).ravel()
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    coords = rng.uniform(0, 600, size=(n_obs, 2)).astype("float64")
    adata.obsm["spatial"] = coords

    img = np.full((600, 595, 3), 200, dtype="uint8")
    adata.uns["spatial"] = {
        "slice1": {
            "images": {"hires": img, "lowres": img},
            "scalefactors": {
                "spot_diameter_fullres": 30.0,
                "fiducial_diameter_fullres": 50.0,
                "tissue_hires_scalef": 0.1,
                "tissue_lowres_scalef": 0.03,
            },
            "metadata": {"source": "tests fixture"},
        }
    }
    adata.write_h5ad(out, compression="gzip")
