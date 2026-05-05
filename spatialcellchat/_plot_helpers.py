"""Reusable matplotlib helpers shared by ``viz_spatial`` (and friends).

Currently exports:

- :func:`_spatial_layer` — fetch the high-resolution tissue image, the
  ``tissue_hires_scalef`` factor, and the scaled (image-pixel) coordinates
  from a Visium/scanpy-style :class:`anndata.AnnData`.
- :func:`_scatterpie_layer` — draw "scatter pie" (one mini pie chart per
  observation) on a matplotlib axes, à la R's ``scatterpie::geom_scatterpie``.

Sub-agent 6d will extend this module — see the H3 marker comment below.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge
from matplotlib.axes import Axes

import anndata as ad


__all__ = [
    "_spatial_layer",
    "_scatterpie_layer",
]


# ---------------------------------------------------------------------------
# H1: _spatial_layer
# ---------------------------------------------------------------------------
def _spatial_layer(
    adata: ad.AnnData,
    lib_id: str = "slice1",
    image_key: str = "hires",
) -> tuple:
    """Pull background image, scale factor, and image-pixel coordinates.

    Mirrors the convention used throughout R's ``visualization.R`` — the
    spatial coordinates stored in ``adata.obsm["spatial"]`` are full-resolution
    pixel positions; the high-/low-res image used for plotting is downscaled
    by ``tissue_hires_scalef`` (or ``tissue_lowres_scalef``). To overlay the
    spots on the background image we therefore multiply each spot's
    full-resolution pixel coordinate by the same scale factor.

    Parameters
    ----------
    adata : anndata.AnnData
        Object holding ``obsm["spatial"]`` and (optionally) the
        scanpy/squidpy-style ``uns["spatial"][lib_id]`` block.
    lib_id : str, default ``"slice1"``
        Library identifier — the key under ``adata.uns["spatial"]``.
    image_key : str, default ``"hires"``
        Which image to pull (``"hires"`` or ``"lowres"``). The companion
        scale factor is ``tissue_<image_key>_scalef``.

    Returns
    -------
    tuple
        ``(image, scalefactor, scaled_coords)`` where
        - ``image`` is a NumPy array (H × W × C) or ``None`` if ``uns["spatial"]``
          is missing,
        - ``scalefactor`` is the float multiplier applied to spot coordinates,
        - ``scaled_coords`` is an ``(n_obs, 2)`` array of image-pixel positions
          (= ``obsm["spatial"] * scalefactor``).
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=float)

    if "spatial" not in adata.uns:
        return None, 1.0, coords

    sp = adata.uns["spatial"]
    if lib_id not in sp:
        return None, 1.0, coords

    block = sp[lib_id]
    images = block.get("images") or {}
    sfs = block.get("scalefactors") or {}

    # ── locate the image ────────────────────────────────────────────
    img = None
    chosen_key = image_key
    if image_key in images:
        img = np.asarray(images[image_key])
    else:
        # Some bundled adatas only ship one image (e.g. lowres) — fall
        # back to whichever resolution IS present.  Mirrors the
        # forgiving R behaviour where ``object@images$image`` is
        # whatever was stored.
        for fb_key in ("hires", "lowres"):
            if fb_key in images:
                img = np.asarray(images[fb_key])
                chosen_key = fb_key
                break

    sf_named = float(sfs.get(f"tissue_{chosen_key}_scalef", 1.0))

    if img is None:
        return None, sf_named, coords * sf_named

    # ── pick a scalefactor that places the spots inside the image ───
    # Some adatas (notably the SpatialCellChat tutorial dataset) store
    # the lowres-resolution image at the "hires" key while keeping the
    # genuine ``tissue_hires_scalef``; applying that scalef leaves the
    # spots ~3x outside the actual image bounds.  Pick whichever
    # scalefactor projects ``coords`` inside the stored image's pixel
    # rectangle.  ``coords`` is in scanpy/squidpy convention —
    # ``[:, 0]`` is x (column, width), ``[:, 1]`` is y (row, height).
    H, W = img.shape[:2]
    x_ext = float(coords[:, 0].max())
    y_ext = float(coords[:, 1].max())

    def _fits(sf: float) -> bool:
        # 10% slack accommodates spots near the tissue edge.
        return (x_ext * sf) <= W * 1.10 and (y_ext * sf) <= H * 1.10

    sf_pick = sf_named
    if not _fits(sf_named):
        for cand_key in ("hires", "lowres"):
            cand_sf = sfs.get(f"tissue_{cand_key}_scalef")
            if cand_sf is None:
                continue
            cand_sf = float(cand_sf)
            if _fits(cand_sf):
                sf_pick = cand_sf
                break

    return img, sf_pick, coords * sf_pick


# ---------------------------------------------------------------------------
# H2: _scatterpie_layer
# ---------------------------------------------------------------------------
def _scatterpie_layer(
    df: pd.DataFrame,
    x: str,
    y: str,
    r: float,
    cols: Sequence[str],
    ax: Axes,
    colors: Sequence[str] | None = None,
    edgecolor: str | None = None,
) -> list:
    """Draw a scatter-pie layer on ``ax``.

    For each row of ``df``, draws a fan of :class:`matplotlib.patches.Wedge`
    centred at ``(df[x], df[y])`` with radius ``r``. The wedge angles are
    proportional to the values in ``cols`` (each row is normalised to sum 1).
    This mirrors R's ``scatterpie::geom_scatterpie``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``x``, ``y`` and all entries of ``cols``.
    x, y : str
        Column names giving spot coordinates.
    r : float
        Pie radius (in data units).
    cols : sequence of str
        Column names whose values are the per-spot category proportions.
    ax : matplotlib.axes.Axes
        Target axes.
    colors : sequence of str or None, default ``None``
        Hex / matplotlib colour for each ``cols`` entry. ``None`` falls back
        to matplotlib's default cycle.
    edgecolor : str or None, default ``None``
        Edge colour passed to each wedge. R's call uses ``color = NA``,
        i.e. no edge.

    Returns
    -------
    list of matplotlib.patches.Wedge
        Patches added to ``ax``.
    """
    cols = list(cols)
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [cycle[k % len(cycle)] for k in range(len(cols))]
    colors = list(colors)

    patches: list = []
    values_arr = df[cols].to_numpy(dtype=float)
    xs = df[x].to_numpy(dtype=float)
    ys = df[y].to_numpy(dtype=float)
    n_rows = values_arr.shape[0]

    for i in range(n_rows):
        row = values_arr[i]
        s = float(np.nansum(row))
        if s <= 0:
            continue
        proportions = row / s
        theta_start = 0.0
        for k, p in enumerate(proportions):
            if not np.isfinite(p) or p <= 0:
                continue
            theta_end = theta_start + 360.0 * float(p)
            wedge = Wedge(
                center=(xs[i], ys[i]),
                r=r,
                theta1=theta_start,
                theta2=theta_end,
                facecolor=colors[k],
                edgecolor=("none" if edgecolor is None else edgecolor),
                linewidth=0.0,
            )
            ax.add_patch(wedge)
            patches.append(wedge)
            theta_start = theta_end
    return patches


# H3 lives here — sub-agent 6d to extend.
