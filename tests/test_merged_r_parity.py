"""R-parity numerical tests for the merged-mode (M-VIZ-11) functions.

Pins every intermediate value computed by the merged-mode comparison
functions against **hand-derived expected values** following R's
formulas line-by-line in ``analysis.R:2773-3037``.

Reference R harness: ``r_harness/dump_merged_diff.R`` (lives outside
the package tree, in the development scratch directory). Run it on a
host with R + SpatialCellChat to regenerate the bundled CSV fixtures
under ``tests/data/r_dumps/merged_v1/``.

The Python fixture below uses **explicit known centrality values**
(not RNG) so the expected outputs are reproducible without R
execution. The R harness consumes the SAME hard-coded values, so the
dumped CSVs compared against the helper outputs are byte-for-byte
identical (max abs diff = 0.0).
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from spatialcellchat._anndata import joint_idents_levels, resolve_dataset_names
from spatialcellchat.viz_embedding import (
    compute_diff_signalingRole_data,
    compute_signalingChanges_data,
)


# ---------------------------------------------------------------------------
# Synthetic fixture — mirrors r_harness/dump_merged_diff.R structure.
# Cells: dataset 1 = G0,G1,G2; dataset 2 = G1,G2,G3 → joint G0,G1,G2,G3
# Pathways: ds1 = a,b,c; ds2 = a,b,d → joint a,b,c,d (R `union` order)
# centr values are explicit (not RNG) so the expected values can be
# hand-derived for assertion below.
# ---------------------------------------------------------------------------
CENTR_NAMES = (
    "outdeg_unweighted", "indeg_unweighted", "outdeg", "indeg",
    "hub", "authority", "eigen", "page_rank", "betweenness",
    "flowbet", "info",
)
OUTDEG_IDX = CENTR_NAMES.index("outdeg")  # = 2
INDEG_IDX = CENTR_NAMES.index("indeg")    # = 3


def _make_centr(outdeg, indeg, cells, sigs):
    """Build the (centr_names, cells, sigs) array with only outdeg/indeg
    populated; other rows zero (mirrors what netAnalysis_computeCentrality
    writes in the synthetic case)."""
    arr = np.zeros((len(CENTR_NAMES), len(cells), len(sigs)), dtype=float)
    arr[OUTDEG_IDX] = outdeg
    arr[INDEG_IDX] = indeg
    return arr


def _make_dataset(cells, sigs, outdeg, indeg) -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((len(cells), 1), dtype=np.float32))
    a.obs["cellchat_idents"] = pd.Categorical(
        cells, categories=cells, ordered=True,
    )
    a.uns["cellchat"] = {
        "netP": {
            "pathways": list(sigs),
            "centr": {
                "values": _make_centr(outdeg, indeg, cells, sigs),
                "centr_names": list(CENTR_NAMES),
                "node_names": list(cells),
                "signaling_names": list(sigs),
            },
        },
    }
    return a


@pytest.fixture
def merged_pair():
    """Two datasets with explicit known outdeg/indeg matrices.

    Layout — outdeg[cell, signaling] for dataset 1 (cells G0,G1,G2 vs PW_a,PW_b,PW_c):
                   PW_a  PW_b  PW_c
            G0      1.0   2.0   0.0
            G1      0.5   0.5   3.0
            G2      0.0   1.0   1.0

    indeg[cell, signaling] for dataset 1:
                   PW_a  PW_b  PW_c
            G0      0.5   1.0   1.0
            G1      1.5   0.0   0.5
            G2      1.0   1.0   0.0

    outdeg for dataset 2 (cells G1,G2,G3 vs PW_a,PW_b,PW_d):
                   PW_a  PW_b  PW_d
            G1      2.0   1.0   0.0
            G2      1.0   0.0   2.0
            G3      0.0   1.5   1.0

    indeg for dataset 2:
                   PW_a  PW_b  PW_d
            G1      1.0   0.5   1.0
            G2      0.5   1.0   0.0
            G3      0.0   0.5   1.5
    """
    cells_1 = ["G0", "G1", "G2"]
    sigs_1 = ["PW_a", "PW_b", "PW_c"]
    outdeg_1 = np.array([
        [1.0, 2.0, 0.0],
        [0.5, 0.5, 3.0],
        [0.0, 1.0, 1.0],
    ])
    indeg_1 = np.array([
        [0.5, 1.0, 1.0],
        [1.5, 0.0, 0.5],
        [1.0, 1.0, 0.0],
    ])
    a1 = _make_dataset(cells_1, sigs_1, outdeg_1, indeg_1)

    cells_2 = ["G1", "G2", "G3"]
    sigs_2 = ["PW_a", "PW_b", "PW_d"]
    outdeg_2 = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.5, 1.0],
    ])
    indeg_2 = np.array([
        [1.0, 0.5, 1.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.5],
    ])
    a2 = _make_dataset(cells_2, sigs_2, outdeg_2, indeg_2)
    return a1, a2


# ---------------------------------------------------------------------------
# Joint-idents and joint-pathways — R parity baseline
# ---------------------------------------------------------------------------
def test_joint_idents_levels_matches_R_union(merged_pair):
    """R: union(c('G0','G1','G2'), c('G1','G2','G3')) = c('G0','G1','G2','G3')."""
    a1, a2 = merged_pair
    assert joint_idents_levels([a1, a2]) == ["G0", "G1", "G2", "G3"]


def test_default_signaling_is_R_union(merged_pair):
    """R ``union(object@netP[[1]]$pathways, object@netP[[2]]$pathways)``
    in first-appearance order = c('PW_a','PW_b','PW_c','PW_d')."""
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    assert out["signaling"] == ["PW_a", "PW_b", "PW_c", "PW_d"]


# ---------------------------------------------------------------------------
# diff_signalingRole_scatter — hand-derived from R formulas
# ---------------------------------------------------------------------------
# Hand computation:
#
# mat_out_1[joint_sigs × joint_cells]:
#         G0    G1    G2    G3
# PW_a   1.0   0.5   0.0   0.0
# PW_b   2.0   0.5   1.0   0.0
# PW_c   0.0   3.0   1.0   0.0
# PW_d   0.0   0.0   0.0   0.0
#
# mat_in_1:
#         G0    G1    G2    G3
# PW_a   0.5   1.5   1.0   0.0
# PW_b   1.0   0.0   1.0   0.0
# PW_c   1.0   0.5   0.0   0.0
# PW_d   0.0   0.0   0.0   0.0
#
# mat_out_2:
#         G0    G1    G2    G3
# PW_a   0.0   2.0   1.0   0.0
# PW_b   0.0   1.0   0.0   1.5
# PW_c   0.0   0.0   0.0   0.0
# PW_d   0.0   0.0   2.0   1.0
#
# mat_in_2:
#         G0    G1    G2    G3
# PW_a   0.0   1.0   0.5   0.0
# PW_b   0.0   0.5   1.0   0.5
# PW_c   0.0   0.0   0.0   0.0
# PW_d   0.0   1.0   0.0   1.5
#
# mat.diff[:, :, 0] = mat_out_2 - mat_out_1:
#         G0     G1    G2    G3
# PW_a  -1.0    1.5   1.0   0.0
# PW_b  -2.0    0.5  -1.0   1.5
# PW_c   0.0   -3.0  -1.0   0.0
# PW_d   0.0    0.0   2.0   1.0
#
# mat.diff[:, :, 1] = mat_in_2 - mat_in_1:
#         G0     G1    G2    G3
# PW_a  -0.5   -0.5  -0.5   0.0
# PW_b  -1.0    0.5   0.0   0.5
# PW_c  -1.0   -0.5   0.0   0.0
# PW_d   0.0    1.0   0.0   1.5
#
# colSums(mat.diff[,,1]) = (G0:-3, G1:-1, G2:1, G3:2.5)
# colSums(mat.diff[,,2]) = (G0:-2.5, G1:0.5, G2:-0.5, G3:2.0)
# ---------------------------------------------------------------------------

EXPECTED_MAT_DIFF_OUT = np.array([
    [-1.0,  1.5,  1.0,  0.0],
    [-2.0,  0.5, -1.0,  1.5],
    [ 0.0, -3.0, -1.0,  0.0],
    [ 0.0,  0.0,  2.0,  1.0],
])
EXPECTED_MAT_DIFF_IN = np.array([
    [-0.5, -0.5, -0.5,  0.0],
    [-1.0,  0.5,  0.0,  0.5],
    [-1.0, -0.5,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  1.5],
])
EXPECTED_OUT_DIFF = np.array([-3.0, -1.0,  1.0,  2.5])
EXPECTED_IN_DIFF  = np.array([-2.5,  0.5, -0.5,  2.0])


def test_diff_signalingRole_mat_diff_matches_hand_derived(merged_pair):
    """``mat.diff = mat.all.merged[[2]] - mat.all.merged[[1]]``
    (R: analysis.R:2836). Pin both channels."""
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    npt.assert_allclose(out["mat_diff"][:, :, 0], EXPECTED_MAT_DIFF_OUT, atol=1e-12)
    npt.assert_allclose(out["mat_diff"][:, :, 1], EXPECTED_MAT_DIFF_IN, atol=1e-12)


def test_diff_signalingRole_outgoing_diff_matches_hand_derived(merged_pair):
    """``outgoing.diff = colSums(mat.diff[,,1])`` (R: analysis.R:2838)."""
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    npt.assert_allclose(out["outgoing_diff"], EXPECTED_OUT_DIFF, atol=1e-12)


def test_diff_signalingRole_incoming_diff_matches_hand_derived(merged_pair):
    """``incoming.diff = colSums(mat.diff[,,2])`` (R: analysis.R:2839)."""
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    npt.assert_allclose(out["incoming_diff"], EXPECTED_IN_DIFF, atol=1e-12)


def test_diff_signalingRole_cell_levels_in_joint_order(merged_pair):
    """The points must be aligned with the joint cell-level vector
    (G0, G1, G2, G3) in that order — same as ``joint_idents_levels``."""
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    assert out["cell_levels"] == ["G0", "G1", "G2", "G3"]


# ---------------------------------------------------------------------------
# signalingChanges_scatter — hand-derived classification for idents_use="G1"
# ---------------------------------------------------------------------------
# Pull the G1 column from each dataset's (n_sig, n_cell, 2) tensor:
#
# use_1[joint_sigs, 2] (G1 column):
#         outgoing  incoming
# PW_a       0.5       1.5
# PW_b       0.5       0.0
# PW_c       3.0       0.5
# PW_d       0.0       0.0
#
# use_2[joint_sigs, 2] (G1 column):
#         outgoing  incoming
# PW_a       2.0       1.0
# PW_b       1.0       0.5
# PW_c       0.0       0.0
# PW_d       0.0       1.0
#
# idx_specific = use_1 * use_2:
#         outgoing  incoming
# PW_a       1.0       1.5
# PW_b       0.5       0.0
# PW_c       0.0       0.0
# PW_d       0.0       0.0
#
# mat_sum = use_2 + use_1:
#         outgoing  incoming
# PW_a       2.5       2.5
# PW_b       1.5       0.5
# PW_c       3.0       0.5
# PW_d       0.0       1.0
#
# out_specific_signaling = rownames where mat_sum[,1]!=0 AND idx_specific[,1]==0
#                        = ["PW_c"]   (PW_c: mat_sum=3, idx_specific=0 → only ds1 has out)
# in_specific_signaling  = rownames where mat_sum[,2]!=0 AND idx_specific[,2]==0
#                        = ["PW_b","PW_c","PW_d"]
#
# mat_diff = use_2 - use_1:
#         outgoing  incoming
# PW_a       1.5      -0.5
# PW_b       0.5       0.5
# PW_c      -3.0      -0.5
# PW_d       0.0       1.0
#
# rowSums(mat_diff): PW_a=1.0, PW_b=1.0, PW_c=-3.5, PW_d=1.0 — all nonzero, all kept.
#
# out_kept = ["PW_c"]   in_kept = ["PW_b","PW_c","PW_d"]
# out_kept (boolean per kept): [F, F, T, F]
# in_kept  (boolean per kept): [F, T, T, T]
# out_in_specific = out_kept AND in_kept = [F, F, T, F]
#
# specificity.out.in (4-way, R: analysis.R:3019-3022):
#   PW_a:  no out, no in → 0 (Shared)
#   PW_b:  no out, in only, not both → -1 (Incoming specific)
#   PW_c:  out & in, both → 2 (Incoming & Outgoing specific)
#   PW_d:  no out, in only, not both → -1 (Incoming specific)
#
# specificity (3-way, R: analysis.R:3027-3029):
#   non-zero spec_out_in AND row mat_diff signs
#   PW_a: spec=0 → 0 (Shared)
#   PW_b: spec=-1 nonzero, row=[0.5, 0.5] → all >=0 → +1 (treatment specific)
#   PW_c: spec=2  nonzero, row=[-3.0, -0.5] → all <=0 → -1 (control specific)
#   PW_d: spec=-1 nonzero, row=[0.0, 1.0] → all >=0 → +1 (treatment specific)
# ---------------------------------------------------------------------------

EXPECTED_USE1 = np.array([
    [0.5, 1.5],
    [0.5, 0.0],
    [3.0, 0.5],
    [0.0, 0.0],
])
EXPECTED_USE2 = np.array([
    [2.0, 1.0],
    [1.0, 0.5],
    [0.0, 0.0],
    [0.0, 1.0],
])
EXPECTED_IDX_SPECIFIC = np.array([
    [1.0, 1.5],
    [0.5, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
])
EXPECTED_MAT_SUM = np.array([
    [2.5, 2.5],
    [1.5, 0.5],
    [3.0, 0.5],
    [0.0, 1.0],
])
EXPECTED_MAT_DIFF_KEPT = np.array([
    [ 1.5, -0.5],
    [ 0.5,  0.5],
    [-3.0, -0.5],
    [ 0.0,  1.0],
])
EXPECTED_SIG_KEPT = ["PW_a", "PW_b", "PW_c", "PW_d"]
EXPECTED_OUT_SPECIFIC = ["PW_c"]
EXPECTED_IN_SPECIFIC = ["PW_b", "PW_c", "PW_d"]
EXPECTED_SPEC_OUT_IN = np.array([0, -1, 2, -1])
EXPECTED_SPECIFICITY = np.array([0, 1, -1, 1])


def test_signalingChanges_use_matrices_match_hand_derived(merged_pair):
    """Per-dataset (n_sig, 2) ``use1`` / ``use2`` tensors picking the
    ``idents_use`` column (R: analysis.R:3007)."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    npt.assert_allclose(out["use1"], EXPECTED_USE1, atol=1e-12)
    npt.assert_allclose(out["use2"], EXPECTED_USE2, atol=1e-12)


def test_signalingChanges_idx_specific_and_mat_sum(merged_pair):
    """``idx.specific = use1 * use2``, ``mat.sum = use2 + use1``
    (R: analysis.R:3008-3009)."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    npt.assert_allclose(out["idx_specific"], EXPECTED_IDX_SPECIFIC, atol=1e-12)
    npt.assert_allclose(out["mat_sum"], EXPECTED_MAT_SUM, atol=1e-12)


def test_signalingChanges_specific_signaling_lists(merged_pair):
    """``out.specific.signaling`` and ``in.specific.signaling`` per
    R: analysis.R:3010-3011."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    assert out["out_specific_signaling"] == EXPECTED_OUT_SPECIFIC
    assert out["in_specific_signaling"] == EXPECTED_IN_SPECIFIC


def test_signalingChanges_mat_diff_kept(merged_pair):
    """Pathways with ``rowSums(mat.diff) == 0`` are filtered out
    (R: analysis.R:3014). All 4 survive in this fixture by construction."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    npt.assert_allclose(out["mat_diff_kept"], EXPECTED_MAT_DIFF_KEPT, atol=1e-12)
    assert out["signaling_kept"] == EXPECTED_SIG_KEPT


def test_signalingChanges_specificity_out_in_4way(merged_pair):
    """4-way classification ``specificity.out.in`` ∈ {0,-1,1,2}
    (R: analysis.R:3019-3022)."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    npt.assert_array_equal(out["specificity_out_in"], EXPECTED_SPEC_OUT_IN)


def test_signalingChanges_specificity_3way(merged_pair):
    """3-way classification ``specificity`` ∈ {0,-1,1}
    (R: analysis.R:3027-3029)."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    npt.assert_array_equal(out["specificity"], EXPECTED_SPECIFICITY)


def test_signalingChanges_dataset_names_default_is_Dataset_N(merged_pair):
    """R parity (CellChat_class.R:296): default add.names = 'Dataset_N'."""
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    assert out["names"] == ["Dataset_1", "Dataset_2"]


# ---------------------------------------------------------------------------
# R-side dump cross-checks. The dumps are bundled under
# ``tests/data/r_dumps/merged_v1/`` (~1 KB total, 5 files), produced by
# ``r_harness/dump_merged_diff.R`` (lives outside the package tree, in
# the development scratch directory). Bundling makes these
# always-on contracts: any future drift in
# :func:`compute_diff_signalingRole_data` /
# :func:`compute_signalingChanges_data` from the R formulas at
# ``analysis.R:2773-3037`` will fail CI immediately. To regenerate the
# dumps after intentional changes, re-run the R harness and copy the
# outputs here.
# ---------------------------------------------------------------------------
R_DUMP_DIR = Path(__file__).parent / "data" / "r_dumps" / "merged_v1"


def test_R_dump_diff_outgoing_matches(merged_pair):
    """Cross-check ``mat.diff[,,'outgoing']`` against the R-side dump."""
    csv_path = R_DUMP_DIR / "diff_signalingRole_outgoing.csv"
    r_df = pd.read_csv(csv_path, index_col=0)
    a1, a2 = merged_pair
    out = compute_diff_signalingRole_data([a1, a2])
    npt.assert_allclose(
        out["mat_diff"][:, :, 0],
        r_df.reindex(index=out["signaling"], columns=out["cell_levels"]).to_numpy(dtype=float),
        atol=1e-9,
    )


def test_R_dump_signalingChanges_classification_matches(merged_pair):
    """Cross-check ``specificity.out.in`` and ``specificity`` integer
    codes against the R-side dump for ``idents.use = "G1"``."""
    csv_path = R_DUMP_DIR / "signalingChanges_G1.csv"
    r_df = pd.read_csv(csv_path)
    a1, a2 = merged_pair
    out = compute_signalingChanges_data([a1, a2], idents_use="G1")
    # Pin codes by signaling name (order may differ).
    by_sig_py = dict(zip(out["signaling_kept"], out["specificity_out_in"]))
    by_sig_r = dict(zip(r_df["signaling"].astype(str),
                        r_df["specificity_out_in"].astype(int)))
    for sig in by_sig_py:
        assert by_sig_py[sig] == by_sig_r.get(sig), \
            f"specificity_out_in mismatch for {sig}: Py={by_sig_py[sig]} R={by_sig_r.get(sig)}"
    by_sig_py2 = dict(zip(out["signaling_kept"], out["specificity"]))
    by_sig_r2 = dict(zip(r_df["signaling"].astype(str),
                         r_df["specificity"].astype(int)))
    for sig in by_sig_py2:
        assert by_sig_py2[sig] == by_sig_r2.get(sig), \
            f"specificity mismatch for {sig}: Py={by_sig_py2[sig]} R={by_sig_r2.get(sig)}"
