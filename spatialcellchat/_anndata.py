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
import pandas as pd

__all__ = [
    "cellchat_dict",
    "get_idents_levels",
    "joint_idents_levels",
    "resolve_dataset_names",
]


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
