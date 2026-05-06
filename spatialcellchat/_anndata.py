"""Tiny AnnData state helpers shared across the package.

These are pure functions that take an :class:`anndata.AnnData` and return a
plain Python value. There is **no wrapper class** — call sites read and
write ``adata.uns / .obsm / .obs / .layers`` directly, scverse-style. These
helpers exist only to (a) avoid duplicating the get-or-create idiom for the
``cellchat`` uns slot across 13 modules, (b) centralise the
``levels(cellchat@idents)`` resolution chain that ``viz_embedding`` and
``viz_heatmap`` had each implemented separately, and (c) provide pure
helpers for cross-dataset (R: ``mergeCellChat``) operations without
introducing a wrapper type.

R parity for cross-dataset operations
-------------------------------------
R's :func:`mergeCellChat` re-bundles a ``list(cellchat1, cellchat2, ...)``
into a single S4 object whose slots become per-dataset lists. The
restructuring is forced by S4's lack of "list of S4 objects" ergonomics;
in Python the natural equivalent is a plain ``Sequence[AnnData]``. The
two helpers :func:`joint_idents_levels` and :func:`resolve_dataset_names`
provide the cross-cutting metadata that R's merged object carries
(``object@idents$joint`` and ``names(object@net)``) — purely on demand,
no caching, no wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as _sp

__all__ = [
    "cellchat_dict",
    "DATA_SLOT_TO_LAYER",
    "get_idents_levels",
    "joint_idents_levels",
    "resolve_data_matrix",
    "resolve_dataset_names",
]


# ---------------------------------------------------------------------------
# Canonical mapping between R's chat slot names and AnnData layer keys.
# ---------------------------------------------------------------------------
# R's :class:`SpatialCellChat` carries five matrix-valued slots:
#   ``@data.raw``, ``@data``, ``@data.signaling``, ``@data.scale``, ``@data.project``.
# AnnData layer keys cannot contain dots cleanly, so the Python port stores
# the four non-``data`` slots in :attr:`AnnData.layers` under
# underscore-separated keys; ``@data`` lives directly in :attr:`AnnData.X`.
# This table is the single source of truth — every reader (viz, modeling,
# de) and every writer (preprocessing) goes through it.
DATA_SLOT_TO_LAYER: Dict[str, str] = {
    "data.signaling": "data_signaling",
    "data.scale":     "data_scale",
    "data.project":   "data_project",
    "data.raw":       "counts",
}


def resolve_data_matrix(
    adata: ad.AnnData,
    slot_name: str = "data.signaling",
    *,
    require_layer: bool = False,
) -> pd.DataFrame:
    """Return the expression matrix for an R-style chat slot.

    R parity for ``methods::slot(object, slot.name)`` —
    every matrix-valued slot read in the original SpatialCellChat code
    (``visualization.R::spatialFeaturePlot``, ``utilities.R::identifyOverExpressedGenes``,
    ``modeling.R::computeCommunProb``, …) goes through this helper.

    The returned :class:`pandas.DataFrame` is **genes × cells** to match the
    R indexing convention (``X[features.use, ]``).  Callers that prefer
    cells × genes can transpose; the helper keeps a single orientation so
    downstream `gene-name → row` indexing stays uniform.

    Lookup priority (mirrors the canonical layout documented in
    ``docs/index.md``)::

        1. ``adata.uns["cellchat"][slot_name]``  — explicit override
                                                   (e.g. tests stashing a
                                                   pre-computed (genes×cells)
                                                   matrix).  Same key the
                                                   pre-fix Python code used to
                                                   look at — kept for
                                                   forward compatibility.
        2. ``slot_name == "data"`` →  ``adata.X``  (the canonical R
                                                   ``chat@data`` slot —
                                                   live normalised matrix).
        3. ``adata.layers[DATA_SLOT_TO_LAYER[slot_name]]``  — the canonical
                                                   AnnData store written by
                                                   :func:`subsetData`,
                                                   :func:`preProcessing`,
                                                   :func:`scaleData`,
                                                   :func:`projectData`,
                                                   :func:`normalizeData`.
        4. Fallback to ``adata.X`` for ``"data.signaling"`` **only when
           no layer is present** — this lets a freshly-built AnnData that
           hasn't been ``subsetData``-d (typical in tests) still feed a
           reader that defaults to ``slot_name="data.signaling"``, while
           still preferring the ALRA-imputed values once
           :func:`preProcessing` has populated the layer.

    Parameters
    ----------
    adata
        The AnnData carrying the CellChat state.
    slot_name
        One of ``{"data", "data.raw", "data.signaling", "data.scale",
        "data.project"}`` — same vocabulary R's ``slot.name`` argument
        accepts.
    require_layer
        If ``True``, the fallback at step 4 is disabled and a missing
        layer raises :class:`ValueError`.  Use this when the call site
        knows the upstream pipeline must have written the layer
        (R reference also errors in those branches).

    Returns
    -------
    pandas.DataFrame
        Genes × cells matrix with ``index`` = ``adata.var_names`` and
        ``columns`` = ``adata.obs_names``.

    Raises
    ------
    ValueError
        When the requested slot is missing and no fallback applies.
    """
    cc = adata.uns.get("cellchat")
    if isinstance(cc, dict) and slot_name in cc:
        # ``uns["cellchat"][slot_name]`` is the legacy-stash entry point.
        # By R-parity convention these are stored genes × cells (R's slot
        # layout), so trust that orientation when the block is a plain
        # ndarray.  DataFrame inputs always disambiguate via their own
        # index/columns labels.
        return _to_genes_cells_df(cc[slot_name], adata, anndata_layout=False)

    if slot_name == "data":
        return _to_genes_cells_df(adata.X, adata, anndata_layout=True)

    if slot_name not in DATA_SLOT_TO_LAYER:
        raise ValueError(
            f"Unknown slot_name {slot_name!r}; expected one of "
            f"{sorted({'data'} | DATA_SLOT_TO_LAYER.keys())}."
        )

    layer_key = DATA_SLOT_TO_LAYER[slot_name]
    if layer_key in adata.layers:
        return _to_genes_cells_df(
            adata.layers[layer_key], adata, anndata_layout=True
        )

    if not require_layer and slot_name == "data.signaling":
        # Test/legacy fall-through: a freshly-built AnnData with X already
        # carrying log-normalised values is the closest analogue to
        # ``chat@data.signaling`` when no subsetData has run.
        return _to_genes_cells_df(adata.X, adata, anndata_layout=True)

    raise ValueError(
        f"Cannot resolve slot {slot_name!r}: layer "
        f"adata.layers[{layer_key!r}] is missing.  Run the upstream "
        f"writer first ({_writer_hint(slot_name)})."
    )


def _to_genes_cells_df(
    block: Any,
    adata: ad.AnnData,
    *,
    anndata_layout: bool,
) -> pd.DataFrame:
    """Coerce ``block`` to a genes × cells :class:`pandas.DataFrame`.

    Parameters
    ----------
    block
        The matrix to coerce.  Accepts:

        * :class:`pandas.DataFrame` — orientation is read from its own
          ``index`` / ``columns`` (so ``anndata_layout`` is ignored when
          the labels disambiguate).
        * AnnData layer / ``adata.X`` payload (``ndarray`` or sparse
          matrix) — orientation determined by ``anndata_layout``.
        * Plain ``ndarray`` from an ``uns``-stashed slot.
    adata
        The owning AnnData (used for ``var_names`` / ``obs_names`` labels
        and the canonical shape).
    anndata_layout
        ``True`` if ``block`` came from ``adata.X`` or
        ``adata.layers[...]`` (cells × genes — the AnnData convention);
        ``False`` if it came from ``adata.uns["cellchat"][slot]`` where
        the R-parity layout is genes × cells.  This flag disambiguates
        the orientation when ``n_obs == n_vars`` (otherwise the shape
        alone would be insufficient).
    """
    if isinstance(block, pd.DataFrame):
        # DataFrame labels disambiguate orientation regardless of the
        # caller-declared layout — if the block's own index matches the
        # AnnData's var_names we're already in genes × cells.
        if list(block.index) == list(adata.var_names) and list(block.columns) == list(adata.obs_names):
            return block
        if list(block.index) == list(adata.obs_names) and list(block.columns) == list(adata.var_names):
            return block.T
        # Shape-based fallback (handles unlabelled DataFrames and
        # asymmetric n_obs ≠ n_vars cases).
        if block.shape == (adata.n_vars, adata.n_obs):
            return block
        if block.shape == (adata.n_obs, adata.n_vars):
            return block.T
        # Last-resort: trust the caller's labelling.
        return block

    if _sp.issparse(block):
        arr = np.asarray(block.todense(), dtype=float)
    else:
        arr = np.asarray(block, dtype=float)

    if anndata_layout:
        # Cells × genes (AnnData convention) → transpose to genes × cells.
        if arr.shape != (adata.n_obs, adata.n_vars):
            raise ValueError(
                f"AnnData-layout block shape {arr.shape} does not match "
                f"({adata.n_obs}, {adata.n_vars})."
            )
        arr = arr.T
    else:
        # Genes × cells (R-parity stash) — already in target orientation.
        if arr.shape != (adata.n_vars, adata.n_obs):
            raise ValueError(
                f"R-layout block shape {arr.shape} does not match "
                f"({adata.n_vars}, {adata.n_obs})."
            )

    return pd.DataFrame(
        arr,
        index=pd.Index(adata.var_names.astype(str)),
        columns=pd.Index(adata.obs_names.astype(str)),
    )


def _writer_hint(slot_name: str) -> str:
    """Return a one-liner naming the upstream function that writes the slot."""
    return {
        "data.raw":       "normalizeData(...) stashes raw counts to layers['counts']",
        "data.signaling": "subsetData(...) / preProcessing(...) write layers['data_signaling']",
        "data.scale":     "scaleData(...) writes layers['data_scale']",
        "data.project":   "projectData(...) writes layers['data_project']",
    }.get(slot_name, "see preprocessing.py")


def cellchat_dict(adata: ad.AnnData) -> Dict[str, Any]:
    """Get-or-create ``adata.uns['cellchat']`` as a plain dict.

    R parity: this is the Python container for the S4 slots
    ``@DB`` / ``@net`` / ``@netP`` / ``@LR`` / ``@centr`` / ``@meta`` /
    ``@options`` / ``@misc``. Writers call this to ensure the slot
    exists; readers should follow up with explicit
    ``if "<sub>" not in cc: raise ...`` so the error message can name the
    upstream function (``computeCommunProb``, ``identifyOverExpressedInteractions``, …).
    """
    if "cellchat" not in adata.uns:
        adata.uns["cellchat"] = {}
    return adata.uns["cellchat"]


def get_idents_levels(
    adata: ad.AnnData,
    slot: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Return ``levels(cellchat@idents)`` — the categorical ordering for plots.

    R parity: ``levels(cellchat@idents)``.

    Resolution chain (each step matches a specific R-side write):

    1. ``slot["dimnames"]["source"]`` — written by :func:`aggregateNet` and
       :func:`computeAvgCommunProb`. Preferred when a slot snapshot is
       available because it reflects the factor levels at the time the
       data was computed (not whatever the user later renamed).
    2. ``slot["meta"]["groups"]`` — written by older code paths.
    3. ``adata.obs[<cellchat_idents | active_ident | cell.type>].cat.categories``
       — current factor on the AnnData; matches R ``cellchat@idents``.

    Parameters
    ----------
    adata
        The :class:`anndata.AnnData` carrying the CellChat state.
    slot
        Optional snapshot dict (e.g. ``cc["net"]`` or ``cc["netP"]``).
        When ``None`` the chain skips steps 1-2 and goes straight to ``adata.obs``.
    """
    if slot is not None:
        dn = slot.get("dimnames")
        if isinstance(dn, dict) and "source" in dn and len(dn["source"]) > 0:
            return list(dn["source"])
        meta = slot.get("meta")
        if isinstance(meta, dict) and "groups" in meta:
            return list(meta["groups"])
    for k in ("cellchat_idents", "active_ident", "cell.type"):
        if k in adata.obs.columns:
            s = adata.obs[k]
            if isinstance(s.dtype, pd.CategoricalDtype):
                return list(s.cat.categories)
            return list(pd.Categorical(s).categories)
    raise ValueError(
        "Cannot recover idents levels from adata. Run aggregateNet / "
        "computeAvgCommunProb (writes 'dimnames'), or set "
        "adata.obs['cellchat_idents'] as a categorical."
    )


def joint_idents_levels(adata_list: Sequence[ad.AnnData]) -> List[str]:
    """Return the joint-idents levels for a list of CellChat AnnDatas.

    R parity: ``levels(object@idents$joint)`` constructed by R's
    :func:`mergeCellChat`. The joint factor is the **union** of per-dataset
    idents categories in **first-appearance order across the list** (i.e.
    every level from ``adata_list[0]`` first, then any new levels from
    ``adata_list[1]`` not seen yet, etc.). This matches R's behaviour
    where ``mergeCellChat`` stacks the per-dataset factor levels.

    Parameters
    ----------
    adata_list
        Sequence of CellChat-state AnnDatas, one per dataset. Equivalent
        to R's ``list(cellchat1, cellchat2, ...)``.

    Returns
    -------
    list of str
        Joint factor levels.
    """
    seen: Dict[str, None] = {}
    for a in adata_list:
        for lv in get_idents_levels(a):
            seen.setdefault(str(lv), None)
    return list(seen)


def resolve_dataset_names(
    adata_list: Sequence[ad.AnnData],
    dataset_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Default per-dataset labels matching R's ``add.names`` semantics.

    R parity (``CellChat_class.R::mergeCellChat``, line 296):

    .. code-block:: r

        if (is.null(add.names)) {
            add.names <- paste("Dataset", 1:length(object.list), sep = "_")
        }

    Defaults to ``["Dataset_1", "Dataset_2", ...]`` to match R bit-for-bit
    so any text rendered into messages, plot titles, or ``specificity``
    factor labels is identical to a parallel R run.

    Parameters
    ----------
    adata_list
        Sequence of CellChat-state AnnDatas (same length as
        ``dataset_names`` if provided).
    dataset_names
        Optional explicit names. If ``None``, defaults to
        ``["Dataset_1", "Dataset_2", ...]``.

    Returns
    -------
    list of str
        Resolved per-dataset labels.
    """
    if dataset_names is None:
        return [f"Dataset_{i + 1}" for i in range(len(adata_list))]
    names = list(dataset_names)
    if len(names) != len(adata_list):
        raise ValueError(
            f"dataset_names length {len(names)} != adata_list length "
            f"{len(adata_list)}."
        )
    return names
