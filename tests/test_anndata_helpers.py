"""Tests for the cross-dataset AnnData helpers.

Pins the R parity of :func:`joint_idents_levels` and
:func:`resolve_dataset_names` against the exact R reference in
``CellChat_class.R::mergeCellChat`` (lines 294-407).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatialcellchat._anndata import (
    cellchat_dict,
    get_idents_levels,
    joint_idents_levels,
    resolve_dataset_names,
)


def _adata_with_idents(levels):
    """Build a tiny AnnData carrying ``adata.obs['cellchat_idents']`` as a
    categorical with the given factor levels (one cell per level)."""
    levels = list(levels)
    a = ad.AnnData(X=np.zeros((len(levels), 1), dtype=np.float32))
    a.obs["cellchat_idents"] = pd.Categorical(
        levels, categories=levels, ordered=True,
    )
    return a


# ---------------------------------------------------------------------------
# joint_idents_levels — R parity (CellChat_class.R:357-365)
# ---------------------------------------------------------------------------
# Reference R code:
#     idents.levels <- c()
#     for (i in 1:length(object.list)) {
#       idents.levels <- union(idents.levels, levels(object.list[[i]]@idents))
#     }
# Base R ``union(x, y)`` preserves first-appearance order: it returns
# ``x`` followed by elements of ``y`` not already in ``x``. Our Python
# helper must give the same order.
def test_joint_idents_levels_disjoint_concatenates_in_order():
    """Two datasets with disjoint level sets → union = first ⊕ new from second."""
    a = _adata_with_idents(["A", "B", "C"])
    b = _adata_with_idents(["X", "Y"])
    assert joint_idents_levels([a, b]) == ["A", "B", "C", "X", "Y"]


def test_joint_idents_levels_overlap_keeps_first_appearance():
    """Overlapping levels: only NEW elements from the second dataset get
    appended, in the second dataset's level order. R parity: ``union``."""
    a = _adata_with_idents(["A", "B", "C"])
    b = _adata_with_idents(["B", "D", "C", "E"])
    # R: union(c("A","B","C"), c("B","D","C","E")) -> c("A","B","C","D","E")
    assert joint_idents_levels([a, b]) == ["A", "B", "C", "D", "E"]


def test_joint_idents_levels_single_dataset_passthrough():
    """A single AnnData's joint = its own levels in level order."""
    a = _adata_with_idents(["G2", "G0", "G1"])
    assert joint_idents_levels([a]) == ["G2", "G0", "G1"]


def test_joint_idents_levels_three_datasets_chained_union():
    """Three datasets: each adds only its NEW levels (R: chained union)."""
    a = _adata_with_idents(["A", "B"])
    b = _adata_with_idents(["B", "C"])
    c = _adata_with_idents(["A", "D", "C", "E"])
    # R chained: union(union(c("A","B"), c("B","C")), c("A","D","C","E"))
    #         = union(c("A","B","C"), c("A","D","C","E")) = c("A","B","C","D","E")
    assert joint_idents_levels([a, b, c]) == ["A", "B", "C", "D", "E"]


def test_joint_idents_levels_preserves_original_factor_level_order():
    """``levels(adata1@idents)`` order must be honoured even when it's
    not alphabetical — R uses the factor's stored level order, not the
    alphabetical / observed order."""
    a = _adata_with_idents(["Macrophage", "Tcell", "Bcell"])  # explicit non-alpha
    # Build B with a different non-alpha order and an extra level.
    b = _adata_with_idents(["Bcell", "DC", "Tcell"])
    # R: union(c("Macrophage","Tcell","Bcell"), c("Bcell","DC","Tcell"))
    #    = c("Macrophage","Tcell","Bcell","DC")
    assert joint_idents_levels([a, b]) == ["Macrophage", "Tcell", "Bcell", "DC"]


# ---------------------------------------------------------------------------
# resolve_dataset_names — R parity (CellChat_class.R:295-297)
# ---------------------------------------------------------------------------
# Reference R code:
#     if (is.null(add.names)) {
#       add.names <- paste("Dataset", 1:length(object.list), sep = "_")
#     }
def test_resolve_dataset_names_default_is_R_default():
    """Default labels must match R's ``paste("Dataset", 1:n, sep="_")``."""
    a = _adata_with_idents(["A"])
    b = _adata_with_idents(["B"])
    c = _adata_with_idents(["C"])
    assert resolve_dataset_names([a, b]) == ["Dataset_1", "Dataset_2"]
    assert resolve_dataset_names([a, b, c]) == ["Dataset_1", "Dataset_2", "Dataset_3"]


def test_resolve_dataset_names_explicit_passthrough():
    """Explicit names are returned as-is (after length validation)."""
    a = _adata_with_idents(["A"])
    b = _adata_with_idents(["B"])
    out = resolve_dataset_names([a, b], dataset_names=["control", "treatment"])
    assert out == ["control", "treatment"]


def test_resolve_dataset_names_length_mismatch_raises():
    """Length mismatch raises ValueError (R: implicit via vector recycling
    rules; here we surface it eagerly)."""
    a = _adata_with_idents(["A"])
    b = _adata_with_idents(["B"])
    with pytest.raises(ValueError, match="length"):
        resolve_dataset_names([a, b], dataset_names=["only_one"])


# ---------------------------------------------------------------------------
# cellchat_dict — get-or-create idempotency
# ---------------------------------------------------------------------------
def test_cellchat_dict_creates_then_returns_same_dict():
    a = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    cc1 = cellchat_dict(a)
    cc1["sentinel"] = 42
    cc2 = cellchat_dict(a)
    # Same dict identity — second call must NOT recreate.
    assert cc1 is cc2
    assert cc2["sentinel"] == 42


def test_cellchat_dict_writes_into_uns():
    """Mutating the returned dict mutates ``adata.uns['cellchat']`` directly
    (no copying — pure scverse semantics)."""
    a = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    cc = cellchat_dict(a)
    cc["DB"] = {"interaction": pd.DataFrame()}
    assert "cellchat" in a.uns
    assert "DB" in a.uns["cellchat"]
    assert isinstance(a.uns["cellchat"]["DB"]["interaction"], pd.DataFrame)
