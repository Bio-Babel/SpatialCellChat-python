"""Tests for :mod:`spatialcellchat.analysis_ranking`.

R↔Py numerical-parity tests for the seven exports in slice 5C:
``compareInteractions``, ``rankNet``, ``rankNetPairwise``,
``rankSimilarity``, ``getMaxWeight``, ``computeNetD_structure`` and
``mergeInteractions``.

Golden values are bundled under ``tests/data/r_dumps/ranking/``
(~83 KB, 94 files). Produced once by ``r_harness/harness.R`` (lives
outside the package tree). To regenerate after intentional changes,
re-run the R harness and copy outputs into this directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import igraph as ig
import numpy as np
import pandas as pd
import pytest

from spatialcellchat.analysis_ranking import (
    _alpha_centrality,
    _entropia,
    _nnd,
    _node_distance,
    _r_format_digits1,
    compareInteractions,
    computeNetD_structure,
    getMaxWeight,
    mergeInteractions,
    rankNet,
    rankNetPairwise,
    rankSimilarity,
)


DUMP = Path(__file__).parent / "data" / "r_dumps" / "ranking"
GOLDEN_AVAILABLE = DUMP.exists()


def _read_json(name: str):
    with open(DUMP / f"{name}.json") as f:
        return json.load(f)


def _read_csv(name: str, **kwargs):
    return pd.read_csv(DUMP / f"{name}.csv", **kwargs)


def _read_prob(prefix: str):
    sub_dir = DUMP / prefix
    pair_names_file = sub_dir / "_pair_names.txt"
    if pair_names_file.exists():
        names = pair_names_file.read_text().strip().split("\n")
    else:
        names = sorted(
            [f.stem for f in sub_dir.iterdir() if f.suffix == ".csv"]
        )
    slices = []
    for nm in names:
        slc = pd.read_csv(sub_dir / f"{nm}.csv").to_numpy()
        slices.append(slc)
    return np.stack(slices, axis=2), names


# ---------------------------------------------------------------------------
# Helper / private function tests
# ---------------------------------------------------------------------------
class TestPrivateHelpers:
    def test_entropia_positive(self) -> None:
        v = _entropia(np.array([0.1, 0.2, 0.3, 0.4]))
        # H = -sum(p log p)
        expected = -(0.1 * np.log(0.1) + 0.2 * np.log(0.2)
                     + 0.3 * np.log(0.3) + 0.4 * np.log(0.4))
        np.testing.assert_allclose(v, expected, atol=1e-12)

    def test_entropia_all_zero(self) -> None:
        assert _entropia(np.array([0, 0, 0])) == 0.0

    def test_entropia_drops_negatives(self) -> None:
        # Negatives are filtered out (a > 0 only)
        v = _entropia(np.array([0.0, -1.0, 0.5]))
        np.testing.assert_allclose(v, -0.5 * np.log(0.5), atol=1e-12)

    def test_node_distance_single_vertex(self) -> None:
        g = ig.Graph(n=1, directed=True)
        out = _node_distance(g)
        assert out.shape == (1, 1)
        assert out[0, 0] == 1.0

    def test_alpha_centrality_N1_returns_length_2(self) -> None:
        """R parity (CellChat_class.R alpha_centrality):
        ``c(r, max(c(0, 1-sum(r))))`` always has length ``N+1`` for ``N >= 1``.
        Audit-flagged edge case: Python must NOT short-circuit to length 1
        for ``N=1``."""
        g = ig.Graph(n=1, directed=True)
        out = _alpha_centrality(g)
        assert out.shape == (2,), f"expected length 2 for N=1, got {out.shape}"
        # No-edge → r = [0]; max(0, 1-0) = 1 → c(0, 1)
        np.testing.assert_allclose(out, np.array([0.0, 1.0]), atol=1e-12)

    def test_alpha_centrality_N0_short_circuit(self) -> None:
        """For ``N=0`` (empty graph) the R formula
        ``c(r, max(0, 1-sum(r)))`` collapses to ``c(1)`` (length 1) because
        ``r`` is empty and ``sum(numeric(0)) = 0``. Python's explicit
        ``N == 0`` early-return is functionally equivalent."""
        g = ig.Graph(n=0, directed=True)
        out = _alpha_centrality(g)
        assert out.shape == (1,)
        assert out[0] == 1.0

    def test_alpha_centrality_N_general_length_is_N_plus_1(self) -> None:
        """General invariant: length is always ``N + 1`` for ``N >= 1``
        regardless of edge structure (R: ``c(r, max(...))`` shape contract)."""
        for n in (1, 2, 3, 5):
            g = ig.Graph(n=n, directed=True)
            out = _alpha_centrality(g)
            assert out.shape == (n + 1,), (
                f"N={n}: expected length {n + 1}, got {out.shape}"
            )

    def test_r_format_digits1_basic(self) -> None:
        # Mirrors R: format(c(1.832, 1.541, 0.479, 0), digits=1)
        v = _r_format_digits1(np.array([1.832, 1.541, 0.479, 0.0]))
        np.testing.assert_allclose(v, [1.8, 1.5, 0.5, 0.0])

    def test_r_format_digits1_inf(self) -> None:
        v = _r_format_digits1(np.array([np.inf, 1.337, 0.832, 0.0]))
        assert np.isinf(v[0])
        np.testing.assert_allclose(v[1:], [1.3, 0.8, 0.0])

    def test_r_format_digits1_zero_only(self) -> None:
        v = _r_format_digits1(np.zeros(3))
        np.testing.assert_array_equal(v, np.zeros(3))


# ---------------------------------------------------------------------------
# computeNetD_structure
# ---------------------------------------------------------------------------
class TestComputeNetDStructure:
    def test_identical_graphs_zero(self) -> None:
        A = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ])
        d = computeNetD_structure(A, A)
        assert d == 0.0

    def test_default_weights(self) -> None:
        # Star vs cycle, both 5 nodes.
        A = np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        B = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ])
        d = computeNetD_structure(A, B)
        assert d > 0.0
        assert np.isfinite(d)

    def test_weights_sum_to_zero_returns_zero(self) -> None:
        A = np.eye(3, dtype=int)
        B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        d = computeNetD_structure(A, B, w1=0, w2=0, w3=0)
        assert d == 0.0

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_default(self) -> None:
        A = np.array([
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        B = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ])
        cases = [
            ("d1", dict()),
            ("d1_w", dict(w1=0.5, w2=0.5, w3=0)),
            ("d1_w3", dict(w1=0, w2=0, w3=1)),
            ("d1_w1", dict(w1=1, w2=0, w3=0)),
            ("d1_w2", dict(w1=0, w2=1, w3=0)),
        ]
        for nm, kw in cases:
            r_v = _read_json(nm)[0]
            py_v = computeNetD_structure(A, B, **kw)
            np.testing.assert_allclose(py_v, r_v, atol=1e-10)

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_different_sizes(self) -> None:
        A3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        B3 = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ])
        np.testing.assert_allclose(
            computeNetD_structure(A3, B3),
            _read_json("d3")[0],
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# mergeInteractions
# ---------------------------------------------------------------------------
class TestMergeInteractions:
    def _fixture(self, count: np.ndarray, weight: np.ndarray) -> ad.AnnData:
        n = count.shape[0]
        a = ad.AnnData(np.zeros((n, 1)))
        a.uns["cellchat"] = {"net": {"count": count, "weight": weight}}
        return a

    def test_basic_three_groups(self) -> None:
        rng = np.random.default_rng(0)
        n = 6
        count = rng.poisson(3, size=(n, n)).astype(float)
        weight = rng.random((n, n))
        a = self._fixture(count, weight)
        mergeInteractions(a, ["g1", "g1", "g2", "g2", "g3", "g3"])
        cm = a.uns["cellchat"]["net"]["count.merged"]
        wm = a.uns["cellchat"]["net"]["weight.merged"]
        # Check shape and sum invariance
        assert cm.shape == (3, 3)
        assert wm.shape == (3, 3)
        np.testing.assert_allclose(cm.values.sum(), count.sum())
        np.testing.assert_allclose(wm.values.sum(), weight.sum())

    def test_all_in_one_group(self) -> None:
        n = 4
        count = np.ones((n, n))
        weight = np.full((n, n), 0.25)
        a = self._fixture(count, weight)
        mergeInteractions(a, ["all"] * n)
        cm = a.uns["cellchat"]["net"]["count.merged"]
        assert cm.shape == (1, 1)
        np.testing.assert_allclose(cm.values, [[16.0]])

    def test_identity_grouping(self) -> None:
        n = 4
        count = np.arange(16).reshape(4, 4).astype(float)
        weight = count * 0.5
        a = self._fixture(count, weight)
        mergeInteractions(a, [chr(ord('A') + i) for i in range(n)])
        cm = a.uns["cellchat"]["net"]["count.merged"]
        np.testing.assert_array_equal(cm.values, count)

    def test_requires_count_and_weight(self) -> None:
        a = ad.AnnData(np.zeros((3, 1)))
        with pytest.raises(ValueError, match="net"):
            mergeInteractions(a, ["g1"] * 3)

    def test_length_mismatch(self) -> None:
        n = 3
        count = np.zeros((n, n))
        weight = np.zeros((n, n))
        a = self._fixture(count, weight)
        with pytest.raises(ValueError, match="length"):
            mergeInteractions(a, ["g1", "g2"])

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden(self) -> None:
        count_R = _read_csv("count_mat", index_col=0).to_numpy()
        weight_R = _read_csv("weight_mat", index_col=0).to_numpy()
        a = self._fixture(count_R, weight_R)
        mergeInteractions(a, ["g1", "g1", "g2", "g2", "g3", "g3"])
        cm = a.uns["cellchat"]["net"]["count.merged"]
        wm = a.uns["cellchat"]["net"]["weight.merged"]
        cm_R = _read_csv("merge_count", index_col=0)
        wm_R = _read_csv("merge_weight", index_col=0)
        np.testing.assert_array_equal(cm.values, cm_R.values)
        np.testing.assert_allclose(wm.values, wm_R.values, atol=1e-12)


# ---------------------------------------------------------------------------
# getMaxWeight
# ---------------------------------------------------------------------------
class TestGetMaxWeight:
    @pytest.fixture
    def two_datasets(self):
        rng = np.random.default_rng(4)
        n1, n2 = 4, 5
        count1 = rng.poisson(3, (n1, n1)).astype(float)
        weight1 = rng.random((n1, n1))
        count2 = rng.poisson(3, (n2, n2)).astype(float)
        weight2 = rng.random((n2, n2))
        prob1 = rng.random((n1, n1, 3))
        prob2 = rng.random((n2, n2, 3))
        pair_names = ["WNT", "TGFb", "BMP"]

        ad1 = ad.AnnData(np.zeros((n1, 1)))
        ad1.obs["cellchat_idents"] = pd.Categorical(["A", "A", "B", "B"])
        ad1.uns["cellchat"] = {
            "net": {
                "count": count1, "weight": weight1, "prob": prob1,
                "dimnames": {"interaction_name": pair_names},
            },
            "netP": {"prob": prob1, "pathways": pair_names},
            "DB": {"interaction": pd.DataFrame({
                "pathway_name": pair_names,
                "interaction_name": [f"{p}_A" for p in pair_names],
            })},
        }
        ad2 = ad.AnnData(np.zeros((n2, 1)))
        ad2.obs["cellchat_idents"] = pd.Categorical(["A", "B", "B", "C", "C"])
        ad2.uns["cellchat"] = {
            "net": {
                "count": count2, "weight": weight2, "prob": prob2,
                "dimnames": {"interaction_name": pair_names},
            },
            "netP": {"prob": prob2, "pathways": pair_names},
            "DB": ad1.uns["cellchat"]["DB"],
        }
        return ad1, ad2

    def test_idents(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        mw = getMaxWeight([ad1, ad2], slot_name="idents", attribute="idents")
        # ad1 max group size is 2 (A/B); ad2 max is 2 (B/C)
        assert mw[0] == 2.0

    def test_count_weight(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        mw = getMaxWeight(
            [ad1, ad2], slot_name=["net", "net"],
            attribute=["count", "weight"],
        )
        assert mw.shape == (2,)
        assert mw[0] == max(
            ad1.uns["cellchat"]["net"]["count"].max(),
            ad2.uns["cellchat"]["net"]["count"].max(),
        )

    def test_pathway_lookup(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        mw = getMaxWeight([ad1, ad2], slot_name="netP", attribute="WNT")
        assert mw.shape == (1,)
        assert mw[0] == max(
            ad1.uns["cellchat"]["netP"]["prob"][:, :, 0].max(),
            ad2.uns["cellchat"]["netP"]["prob"][:, :, 0].max(),
        )

    def test_unknown_attribute_raises(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="not a known"):
            getMaxWeight(
                [ad1, ad2], slot_name="netP", attribute="NONSUCH_PATHWAY",
            )

    def test_length_mismatch_raises(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="same length"):
            getMaxWeight(
                [ad1, ad2], slot_name=["net"], attribute=["count", "weight"],
            )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden(self) -> None:
        count1 = _read_csv("count1", index_col=0).to_numpy()
        weight1 = _read_csv("weight1", index_col=0).to_numpy()
        count2 = _read_csv("count2", index_col=0).to_numpy()
        weight2 = _read_csv("weight2", index_col=0).to_numpy()
        prob1, p1_names = _read_prob("prob1")
        prob2, p2_names = _read_prob("prob2")

        ad1 = ad.AnnData(np.zeros((4, 1)))
        ad1.obs["cellchat_idents"] = pd.Categorical(["A", "A", "B", "B"])
        ad1.uns["cellchat"] = {
            "net": {
                "count": count1, "weight": weight1, "prob": prob1,
                "dimnames": {"interaction_name": p1_names},
            },
            "netP": {"prob": prob1, "pathways": p1_names},
            "DB": {"interaction": pd.DataFrame({
                "pathway_name": ["WNT", "TGFb", "BMP"],
                "interaction_name": ["WNT_A", "TGFb_A", "BMP_A"],
            })},
        }
        ad2 = ad.AnnData(np.zeros((5, 1)))
        ad2.obs["cellchat_idents"] = pd.Categorical(["A", "B", "B", "C", "C"])
        ad2.uns["cellchat"] = {
            "net": {
                "count": count2, "weight": weight2, "prob": prob2,
                "dimnames": {"interaction_name": p2_names},
            },
            "netP": {"prob": prob2, "pathways": p2_names},
            "DB": ad1.uns["cellchat"]["DB"],
        }
        mw1 = getMaxWeight(
            [ad1, ad2], slot_name=["net", "net"],
            attribute=["count", "weight"],
        )
        mw2 = getMaxWeight([ad1, ad2], slot_name="idents", attribute="idents")
        mw3 = getMaxWeight([ad1, ad2], slot_name="netP", attribute="WNT")
        mw4 = getMaxWeight(
            [ad1, ad2],
            slot_name=["idents", "net", "netP"],
            attribute=["idents", "weight", "TGFb"],
        )
        np.testing.assert_allclose(mw1, _read_json("mw1"), atol=1e-12)
        np.testing.assert_allclose(mw2, _read_json("mw2"), atol=1e-12)
        np.testing.assert_allclose(mw3, _read_json("mw3"), atol=1e-12)
        np.testing.assert_allclose(mw4, _read_json("mw4"), atol=1e-12)


# ---------------------------------------------------------------------------
# compareInteractions
# ---------------------------------------------------------------------------
class TestCompareInteractions:
    @pytest.fixture
    def two_datasets(self):
        ad1 = ad.AnnData(np.zeros((3, 1)))
        ad2 = ad.AnnData(np.zeros((3, 1)))
        ad1.uns["cellchat"] = {"net": {
            "count": np.array([[2, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
            "weight": np.array([[0.4, 0.2, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]]),
        }}
        ad2.uns["cellchat"] = {"net": {
            "count": np.array([[3, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=float),
            "weight": np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]),
        }}
        return ad1, ad2

    def test_count(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = compareInteractions([ad1, ad2], measure="count")
        assert list(df.columns) == ["count", "dataset", "group"]
        assert df["count"].tolist() == [4.0, 4.0]
        assert list(df["dataset"].astype(str)) == ["d1", "d2"]

    def test_weight_rounding(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = compareInteractions([ad1, ad2], measure="weight", digits=2)
        # ad1: 0.4 + 0.2 + 0.1 = 0.7
        # ad2: 0.5 + 0.3 = 0.8
        assert df["count"].tolist() == [0.7, 0.8]

    def test_invalid_measure(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="measure"):
            compareInteractions([ad1, ad2], measure="badname")

    def test_dataset_names_override(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = compareInteractions(
            [ad1, ad2], measure="count", dataset_names=["control", "treated"],
        )
        assert list(df["dataset"].astype(str)) == ["control", "treated"]

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden(self) -> None:
        count1 = _read_csv("count1", index_col=0).to_numpy()
        weight1 = _read_csv("weight1", index_col=0).to_numpy()
        count2 = _read_csv("count2", index_col=0).to_numpy()
        weight2 = _read_csv("weight2", index_col=0).to_numpy()
        ad1 = ad.AnnData(np.zeros((4, 1)))
        ad2 = ad.AnnData(np.zeros((5, 1)))
        ad1.uns["cellchat"] = {"net": {"count": count1, "weight": weight1}}
        ad2.uns["cellchat"] = {"net": {"count": count2, "weight": weight2}}
        df_c = compareInteractions([ad1, ad2], measure="count")
        df_c_R = _read_csv("ci_count", index_col=0)
        np.testing.assert_allclose(df_c["count"].values, df_c_R["count"].values)
        df_w = compareInteractions([ad1, ad2], measure="weight")
        df_w_R = _read_csv("ci_weight", index_col=0)
        np.testing.assert_allclose(df_w["count"].values, df_w_R["count"].values)
        df_w_d = compareInteractions(
            [ad1, ad2], measure="weight", digits=5,
        )
        df_w_d_R = _read_csv("ci_weight_d", index_col=0)
        np.testing.assert_allclose(
            df_w_d["count"].values, df_w_d_R["count"].values,
        )


# ---------------------------------------------------------------------------
# rankNet
# ---------------------------------------------------------------------------
class TestRankNet:
    @pytest.fixture
    def two_datasets(self):
        prob_a, _ = _read_prob("prob_a") if GOLDEN_AVAILABLE else (
            np.random.default_rng(5).random((4, 4, 4)), None
        )
        pval_a, _ = _read_prob("pval_a") if GOLDEN_AVAILABLE else (
            np.full((4, 4, 4), 0.01), None
        )
        prob_b, _ = _read_prob("prob_b") if GOLDEN_AVAILABLE else (
            np.random.default_rng(6).random((4, 4, 4)), None
        )
        pval_b, _ = _read_prob("pval_b") if GOLDEN_AVAILABLE else (
            np.full((4, 4, 4), 0.01), None
        )
        names = ["WNT", "TGFb", "BMP", "FGF"]
        DB = pd.DataFrame({
            "pathway_name": names,
            "interaction_name": [f"{n}_A" for n in names],
            "annotation": ["Secreted Signaling"] * 4,
        })
        ad1 = ad.AnnData(np.zeros((4, 1)))
        ad1.obs["cellchat_idents"] = pd.Categorical(["A", "B", "C", "D"])
        ad1.uns["cellchat"] = {
            "net": {"dimnames": {
                "source": ["A", "B", "C", "D"],
                "target": ["A", "B", "C", "D"],
                "interaction_name": names,
            }},
            "netP": {"prob": prob_a, "pval": pval_a, "pathways": names},
            "DB": {"interaction": DB},
        }
        ad2 = ad.AnnData(np.zeros((4, 1)))
        ad2.obs["cellchat_idents"] = pd.Categorical(["A", "B", "C", "D"])
        ad2.uns["cellchat"] = {
            "net": {"dimnames": {
                "source": ["A", "B", "C", "D"],
                "target": ["A", "B", "C", "D"],
                "interaction_name": names,
            }},
            "netP": {"prob": prob_b, "pval": pval_b, "pathways": names},
            "DB": {"interaction": DB},
        }
        return ad1, ad2

    def test_comparison_weight_basic(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2],
        )
        assert "name" in df.columns
        assert "contribution" in df.columns
        assert "contribution.scaled" in df.columns
        assert "group" in df.columns
        assert "contribution.relative.1" in df.columns

    def test_comparison_count(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="count",
            mode="comparison", comparison=[1, 2],
        )
        # Counts should always be integer-like
        assert (df["contribution"] >= 0).all()

    def test_comparison_sources_use_string(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2], sources_use=["A", "B"],
        )
        assert "name" in df.columns

    def test_comparison_invalid_pairLR_with_netP(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="slot.name == 'net'"):
            rankNet(
                [ad1, ad2], slot_name="netP", measure="weight",
                mode="comparison", pairLR=["WNT_A"],
            )

    def test_invalid_measure(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="measure"):
            rankNet([ad1, ad2], measure="bad")

    def test_invalid_mode(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        with pytest.raises(ValueError, match="mode"):
            rankNet([ad1, ad2], mode="weird")

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_weight(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2],
        )
        df_R = _read_csv("df_rank_w", index_col=0)
        # Compare row by row by (name, group)
        Py_keys = list(zip(df["name"].astype(str), df["group"].astype(str)))
        R_keys = list(zip(df_R["name"].astype(str), df_R["group"].astype(str)))
        assert Py_keys == R_keys
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-10,
        )
        np.testing.assert_allclose(
            df["contribution.scaled"].values,
            df_R["contribution.scaled"].values,
            atol=1e-10,
        )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_count(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="count",
            mode="comparison", comparison=[1, 2],
        )
        df_R = _read_csv("df_rank_c", index_col=0)
        np.testing.assert_array_equal(
            df["name"].astype(str).values,
            df_R["name"].astype(str).values,
        )
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-10,
        )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_sources_use(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2], sources_use=[1, 2],
        )
        df_R = _read_csv("df_rank_src", index_col=0)
        np.testing.assert_array_equal(
            df["name"].astype(str).values,
            df_R["name"].astype(str).values,
        )
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-10,
        )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_targets_use(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2], targets_use=[1, 2],
        )
        df_R = _read_csv("df_rank_tgt", index_col=0)
        np.testing.assert_array_equal(
            df["name"].astype(str).values,
            df_R["name"].astype(str).values,
        )
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-10,
        )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_signaling_filter(self, two_datasets) -> None:
        ad1, ad2 = two_datasets
        df = rankNet(
            [ad1, ad2], slot_name="netP", measure="weight",
            mode="comparison", comparison=[1, 2],
            signaling=["WNT", "TGFb"],
        )
        df_R = _read_csv("df_rank_sig", index_col=0)
        assert sorted(df["name"].astype(str).unique()) == ["TGFb", "WNT"]
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-10,
        )

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden_single_mode(self) -> None:
        prob_a, names = _read_prob("prob_a")
        pval_a, _ = _read_prob("pval_a")
        ad1 = ad.AnnData(np.zeros((4, 1)))
        ad1.obs["cellchat_idents"] = pd.Categorical(["A", "B", "C", "D"])
        ad1.uns["cellchat"] = {
            "net": {"dimnames": {
                "source": ["A", "B", "C", "D"],
                "target": ["A", "B", "C", "D"],
            }},
            "netP": {"prob": prob_a, "pval": pval_a, "pathways": names},
            "DB": {"interaction": pd.DataFrame({
                "pathway_name": names,
                "interaction_name": [f"{n}_A" for n in names],
                "annotation": ["Secreted Signaling"] * 4,
            })},
        }
        df = rankNet(
            ad1, slot_name="netP", measure="weight",
            mode="single", comparison=[1],
        )
        df_R = _read_csv("df_rank_single", index_col=0)
        np.testing.assert_array_equal(
            df["name"].astype(str).values,
            df_R["name"].astype(str).values,
        )
        np.testing.assert_allclose(
            df["contribution"].values, df_R["contribution"].values, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# rankNetPairwise
# ---------------------------------------------------------------------------
class TestRankNetPairwise:
    @pytest.fixture
    def adata_with_LR(self):
        rng = np.random.default_rng(6)
        nC, nL = 3, 4
        prob_p = rng.random((nC, nC, nL))
        pval_p = rng.random((nC, nC, nL))
        names = ["L1_R1", "L2_R2", "L3_R3", "L4_R4"]
        LRsig = pd.DataFrame({
            "interaction_name": names,
            "interaction_name_2": ["L1-R1", "L2-R2", "L3-R3", "L4-R4"],
            "pathway_name": ["WNT", "TGFb", "BMP", "FGF"],
            "ligand": ["L1", "L2", "L3", "L4"],
            "receptor": ["R1", "R2", "R3", "R4"],
        }, index=names)
        a = ad.AnnData(np.zeros((nC, 1)))
        a.uns["cellchat"] = {
            "net": {
                "prob": prob_p, "pval": pval_p,
                "dimnames": {
                    "source": ["A", "B", "C"],
                    "target": ["A", "B", "C"],
                    "interaction_name": names,
                },
            },
            "LR": {"LRsig": LRsig},
        }
        return a

    def test_basic(self, adata_with_LR) -> None:
        rankNetPairwise(adata_with_LR)
        pwr = adata_with_LR.uns["cellchat"]["net"]["pairwiseRank"]
        assert set(pwr.keys()) == {"A", "B", "C"}
        for src in pwr:
            assert set(pwr[src].keys()) == {"A", "B", "C"}
            for tgt in pwr[src]:
                df = pwr[src][tgt]
                assert "interaction_name" in df.columns
                assert "prob" in df.columns
                assert "pval" in df.columns
                # Sorted by pval ascending then prob descending
                pvals = df["pval"].to_numpy()
                assert np.all(pvals[:-1] <= pvals[1:])

    def test_LR_use_override(self, adata_with_LR) -> None:
        # Truncated LR_use of length 4 still passes (matches nLR)
        names = ["L1_R1", "L2_R2", "L3_R3", "L4_R4"]
        my_lr = pd.DataFrame({
            "interaction_name": names,
            "interaction_name_2": names,
            "pathway_name": ["P1"] * 4,
            "ligand": names,
            "receptor": names,
        }, index=names)
        rankNetPairwise(adata_with_LR, LR_use=my_lr)
        pwr = adata_with_LR.uns["cellchat"]["net"]["pairwiseRank"]
        assert (pwr["A"]["A"]["pathway_name"] == "P1").all()

    def test_missing_net_raises(self) -> None:
        a = ad.AnnData(np.zeros((3, 1)))
        with pytest.raises(ValueError, match="prob"):
            rankNetPairwise(a)

    def test_LR_length_mismatch(self, adata_with_LR) -> None:
        bad_lr = adata_with_LR.uns["cellchat"]["LR"]["LRsig"].iloc[:2].copy()
        with pytest.raises(ValueError, match="length"):
            rankNetPairwise(adata_with_LR, LR_use=bad_lr)

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden(self) -> None:
        prob_p, names = _read_prob("prob_p")
        pval_p, _ = _read_prob("pval_p")
        LRsig = _read_csv("LRsig_df", index_col=0)
        a = ad.AnnData(np.zeros((3, 1)))
        a.uns["cellchat"] = {
            "net": {
                "prob": prob_p, "pval": pval_p,
                "dimnames": {
                    "source": ["A", "B", "C"],
                    "target": ["A", "B", "C"],
                    "interaction_name": names,
                },
            },
            "LR": {"LRsig": LRsig},
        }
        rankNetPairwise(a)
        pwr = a.uns["cellchat"]["net"]["pairwiseRank"]
        for src, tgt, key in [
            ("A", "A", "pwr_AA"),
            ("A", "B", "pwr_AB"),
            ("B", "C", "pwr_BC"),
        ]:
            R_df = _read_csv(key, index_col=0)
            Py_df = pwr[src][tgt]
            np.testing.assert_array_equal(
                Py_df["interaction_name"].astype(str).values,
                R_df["interaction_name"].astype(str).values,
            )
            np.testing.assert_allclose(
                Py_df["prob"].values, R_df["prob"].values, atol=1e-12,
            )
            np.testing.assert_allclose(
                Py_df["pval"].values, R_df["pval"].values, atol=1e-12,
            )


# ---------------------------------------------------------------------------
# rankSimilarity
# ---------------------------------------------------------------------------
class TestRankSimilarity:
    def _build(self, dr_func: pd.DataFrame, dr_struct: pd.DataFrame) -> ad.AnnData:
        a = ad.AnnData(np.zeros((1, 1)))
        a.uns["cellchat"] = {
            "netP": {
                "__dataset_names__": ["d1", "d2"],
                "similarity": {
                    "functional": {"dr": {"1-2": dr_func}},
                    "structural": {"dr": {"1-2": dr_struct}},
                },
            },
            "meta": {"datasets": ["d1"] * 5 + ["d2"] * 5},
        }
        return a

    def test_basic_functional(self) -> None:
        # Build a matrix where d1 and d2 entries match exactly per pathway,
        # so the per-pathway Euclidean distance is 0.
        dr_mat = pd.DataFrame(
            np.array([
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ]),
            index=[
                "WNT--d1", "TGFb--d1", "BMP--d1",
                "WNT--d2", "TGFb--d2", "BMP--d2",
            ],
        )
        a = self._build(dr_mat, dr_mat)
        df = rankSimilarity(
            a, slot_name="netP", type="functional",
            comparison1=[1, 2], comparison2=[1, 2],
        )
        assert "name" in df.columns
        assert "dist" in df.columns
        assert sorted(df["name"].astype(str).tolist()) == ["BMP", "TGFb", "WNT"]
        np.testing.assert_allclose(df["dist"].values, 0.0, atol=1e-12)

    def test_intersect_only(self) -> None:
        # Pathway only present in one dataset is dropped.
        dr_mat = pd.DataFrame(
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            index=["WNT--d1", "TGFb--d1", "WNT--d2", "BMP--d2"],
        )
        a = self._build(dr_mat, dr_mat)
        df = rankSimilarity(
            a, slot_name="netP", type="functional",
            comparison1=[1, 2], comparison2=[1, 2],
        )
        # Only WNT is shared
        assert df["name"].astype(str).tolist() == ["WNT"]

    def test_missing_slot_raises(self) -> None:
        a = ad.AnnData(np.zeros((1, 1)))
        a.uns["cellchat"] = {}
        with pytest.raises(ValueError, match="missing"):
            rankSimilarity(a, slot_name="netP", type="functional",
                           comparison1=[1, 2])

    @pytest.mark.skipif(not GOLDEN_AVAILABLE, reason="R-dumped golden absent")
    def test_R_golden(self) -> None:
        dr_func = _read_csv("dr_mat_func", index_col=0)
        dr_struct = _read_csv("dr_mat_struct", index_col=0)
        a = self._build(dr_func, dr_struct)
        df_f = rankSimilarity(
            a, slot_name="netP", type="functional",
            comparison1=[1, 2], comparison2=[1, 2],
        )
        df_f_R = _read_csv("df_sim_f", index_col=0)
        np.testing.assert_array_equal(
            df_f["name"].astype(str).values,
            df_f_R["name"].astype(str).values,
        )
        np.testing.assert_allclose(
            df_f["dist"].values, df_f_R["dist"].values, atol=1e-10,
        )
        df_s = rankSimilarity(
            a, slot_name="netP", type="structural",
            comparison1=[1, 2], comparison2=[1, 2],
        )
        df_s_R = _read_csv("df_sim_s", index_col=0)
        np.testing.assert_allclose(
            df_s["dist"].values, df_s_R["dist"].values, atol=1e-10,
        )
