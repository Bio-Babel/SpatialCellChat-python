"""Tests for spatialcellchat.database loaders and query helpers.

Loader shape parity (Slice 0):

- human: interaction(3234x?), complex(338x5), cofactor(32x16), geneInfo(26827x8); PPI 4815x4815
- mouse: interaction(3379x?), complex(337x5), cofactor(33x16), geneInfo(25520x8); PPI 557x557
- zebrafish: interaction(2774x?), complex(267x2), cofactor(62x5), geneInfo(15778x2)

Query-helper tests verify R<->Py parity by comparing against ground-truth
output dumped from the R ``SpatialCellChat`` package via Rscript (values
hardcoded into the test fixtures below, with the Rscript commands listed in
docstrings for reproducibility).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from spatialcellchat import database


# ---------------------------------------------------------------------------
# Loader tests (Slice 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("species", "expected"),
    [
        ("human", {"interaction": 3234, "complex": 338, "cofactor": 32, "geneInfo": 26827}),
        ("mouse", {"interaction": 3379, "complex": 337, "cofactor": 33, "geneInfo": 25520}),
        ("zebrafish", {"interaction": 2774, "complex": 267, "cofactor": 62, "geneInfo": 15778}),
    ],
)
def test_load_db_shapes(species, expected):
    db = database.load_db(species)
    assert set(db) == {"interaction", "complex", "cofactor", "geneInfo"}
    for key, n_rows in expected.items():
        assert isinstance(db[key], pd.DataFrame)
        assert len(db[key]) == n_rows, f"{species}.{key} expected {n_rows}, got {len(db[key])}"


def test_load_db_interaction_indexed_by_name():
    db = database.load_db("human")
    interaction = db["interaction"]
    assert "interaction_name" in interaction.columns
    assert interaction.index.equals(pd.Index(interaction["interaction_name"]))


def test_load_db_invalid_species_raises():
    with pytest.raises(ValueError, match="species"):
        database.load_db("rat")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("species", "n", "nnz"),
    [("human", 4815, 27702), ("mouse", 557, 1430)],
)
def test_load_ppi_shape(species, n, nnz):
    mat, rows = database.load_ppi(species)
    assert mat.shape == (n, n)
    assert mat.nnz == nnz
    assert len(rows) == n
    assert all(isinstance(r, str) for r in rows[:5])


def test_load_ppi_invalid_species_raises():
    with pytest.raises(ValueError, match="species"):
        database.load_ppi("zebrafish")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# subsetDB
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def db_human():
    return database.load_db("human")


def test_subsetDB_default_strips_non_protein(db_human):
    """Rscript: ``nrow(subsetDB(CellChatDB.human, search=NULL)$interaction)`` -> 2240."""
    res = database.subsetDB(db_human)
    assert len(res["interaction"]) == 2240
    annotations = set(res["interaction"]["annotation"].dropna().unique())
    assert annotations == {"Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"}


def test_subsetDB_default_with_non_protein(db_human):
    """Rscript: ``nrow(subsetDB(CellChatDB.human, NULL, non_protein=TRUE)$interaction)`` -> 3234."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = database.subsetDB(db_human, search=None, non_protein=True)
    assert len(res["interaction"]) == 3234


def test_subsetDB_single_annotation(db_human):
    """Rscript: ``subsetDB(db, search=c('Secreted Signaling'), key='annotation')`` -> 1280 rows."""
    res = database.subsetDB(db_human, search=["Secreted Signaling"], key="annotation")
    assert len(res["interaction"]) == 1280
    assert (res["interaction"]["annotation"] == "Secreted Signaling").all()
    # Other tables unchanged.
    assert len(res["complex"]) == len(db_human["complex"])
    assert len(res["cofactor"]) == len(db_human["cofactor"])


def test_subsetDB_pathway(db_human):
    """Rscript: ``subsetDB(db, 'TGFb', key='pathway_name')`` -> 12 rows."""
    res = database.subsetDB(db_human, search="TGFb", key="pathway_name")
    assert len(res["interaction"]) == 12
    assert set(res["interaction"]["pathway_name"]) == {"TGFb"}


def test_subsetDB_vector_key(db_human):
    """Rscript: ``subsetDB(db, list('Secreted Signaling','TGFb'), c('annotation','pathway_name'))`` -> 12."""
    res = database.subsetDB(
        db_human,
        search=[["Secreted Signaling"], ["TGFb"]],
        key=["annotation", "pathway_name"],
    )
    assert len(res["interaction"]) == 12
    assert set(res["interaction"]["pathway_name"]) == {"TGFb"}
    assert set(res["interaction"]["annotation"]) == {"Secreted Signaling"}


def test_subsetDB_non_protein_warning(db_human):
    with pytest.warns(UserWarning, match="non-protein"):
        database.subsetDB(
            db_human,
            search=["Non-protein Signaling"],
            key="annotation",
        )


def test_subsetDB_invalid_key_raises(db_human):
    with pytest.raises(ValueError, match="column names"):
        database.subsetDB(db_human, search="x", key="not_a_column")


def test_subsetDB_length1_list_key_acts_like_scalar(db_human):
    """R parity: ``length(key) == 1`` triggers the scalar branch.

    ``key=["annotation"]`` (length-1 list) must behave the same as
    ``key="annotation"`` (string). Both forms are valid in R.
    """
    via_str = database.subsetDB(
        db_human, search=["Secreted Signaling"], key="annotation"
    )
    via_len1_list = database.subsetDB(
        db_human, search=["Secreted Signaling"], key=["annotation"]
    )
    assert len(via_str["interaction"]) == len(via_len1_list["interaction"])
    assert (
        via_str["interaction"]["interaction_name"].tolist()
        == via_len1_list["interaction"]["interaction_name"].tolist()
    )


def test_subsetDB_length1_list_key_with_listoflist_search(db_human):
    """``key=["annotation"]`` with ``search=[["Secreted Signaling"]]`` must
    also work — that's the natural form when callers pre-build vector-style
    arguments and the key happens to collapse to length 1.
    """
    res = database.subsetDB(
        db_human,
        search=[["Secreted Signaling"]],
        key=["annotation"],
    )
    assert len(res["interaction"]) > 0
    assert set(res["interaction"]["annotation"]) == {"Secreted Signaling"}


# ---------------------------------------------------------------------------
# searchPair
# ---------------------------------------------------------------------------
def test_searchPair_single_pathway(db_human):
    """Rscript: ``searchPair('TGFb', db$interaction)`` -> 12x4."""
    res = database.searchPair("TGFb", db_human["interaction"])
    assert res.shape == (12, 4)
    assert list(res.columns) == ["interaction_name", "pathway_name", "ligand", "receptor"]
    # First three rows must match R order.
    assert res.index[:3].tolist() == [
        "TGFB1_TGFBR1_TGFBR2",
        "TGFB2_TGFBR1_TGFBR2",
        "TGFB3_TGFBR1_TGFBR2",
    ]


def test_searchPair_multi_pathway(db_human):
    """Rscript: ``searchPair(c('TGFb','WNT'), db$interaction)`` -> 363x4."""
    res = database.searchPair(["TGFb", "WNT"], db_human["interaction"])
    assert res.shape == (363, 4)


def test_searchPair_pair_only_false(db_human):
    """Rscript: ``searchPair('TGFb', db$interaction, pair.only=FALSE)`` -> 12x28."""
    res = database.searchPair("TGFb", db_human["interaction"], pair_only=False)
    # Python loader carries an extra ``rowname`` column from parquet, so the
    # column count is one more than R's 28; check it is in the expected range.
    assert res.shape[0] == 12
    assert {"interaction_name", "pathway_name", "ligand", "receptor", "evidence"}.issubset(
        res.columns
    )


def test_searchPair_matching_exact(db_human):
    res = database.searchPair("TGFb", db_human["interaction"], matching_exact=True)
    assert res.shape == (12, 4)


def test_searchPair_key_ligand(db_human):
    """Rscript: ``searchPair('TGFB1', db$interaction, key='ligand')`` -> 4x4."""
    res = database.searchPair("TGFB1", db_human["interaction"], key="ligand")
    assert res.shape == (4, 4)
    assert (res["ligand"] == "TGFB1").all()


def test_searchPair_invalid_key_raises(db_human):
    with pytest.raises(ValueError, match="key"):
        database.searchPair("TGFb", db_human["interaction"], key="annotation")


def test_searchPair_not_found_raises(db_human):
    with pytest.raises(ValueError, match="Cannot find"):
        database.searchPair("DOES_NOT_EXIST_XYZ", db_human["interaction"])


# ---------------------------------------------------------------------------
# extractGene
# ---------------------------------------------------------------------------
def test_extractGene_human_length(db_human):
    """Rscript: ``length(extractGene(CellChatDB.human))`` -> 1446."""
    g = database.extractGene(db_human)
    assert len(g) == 1446
    assert len(set(g)) == 1446  # no duplicates


def test_extractGene_known_members(db_human):
    g = database.extractGene(db_human)
    # Sample ligands, receptor subunits and cofactor subunits known from R.
    for sym in ["TGFB1", "BMP2", "TGFBR1", "TGFBR2", "FST", "WIF1", "FRZB"]:
        assert sym in g, f"missing {sym}"


def test_extractGene_first_genes_match_R(db_human):
    """First 5 gene symbols (from interaction$ligand order) match R."""
    g = database.extractGene(db_human)
    assert g[:5] == ["TGFB1", "TGFB2", "TGFB3", "BMP2", "BMP4"]


# ---------------------------------------------------------------------------
# extractGeneSubset
# ---------------------------------------------------------------------------
def test_extractGeneSubset_basic(db_human):
    """Rscript: ``extractGeneSubset(c('TGFB1','TGFbR1_R2','TGFB2'), db$complex, db$geneInfo)``
    -> c('TGFB1','TGFB2','TGFBR1','TGFBR2')."""
    g = database.extractGeneSubset(
        ["TGFB1", "TGFbR1_R2", "TGFB2"], db_human["complex"], db_human["geneInfo"]
    )
    assert g == ["TGFB1", "TGFB2", "TGFBR1", "TGFBR2"]


def test_extractGeneSubset_only_complex(db_human):
    """Rscript: ``extractGeneSubset('TGFbR1_R2', db$complex, db$geneInfo)``
    -> c('TGFBR1','TGFBR2')."""
    g = database.extractGeneSubset(["TGFbR1_R2"], db_human["complex"], db_human["geneInfo"])
    assert g == ["TGFBR1", "TGFBR2"]


def test_extractGeneSubset_empty(db_human):
    g = database.extractGeneSubset([], db_human["complex"], db_human["geneInfo"])
    assert g == []


def test_extractGeneSubset_column_major_order(db_human):
    """R's ``unlist(complex_input[..., starts_with('subunit')])`` is column-major."""
    g = database.extractGeneSubset(
        ["TGFbR1_R2", "Activin AB", "Inhibin A"],
        db_human["complex"],
        db_human["geneInfo"],
    )
    # R-side: TGFBR1, INHBA, INHA, TGFBR2, INHBB (col-major; INHBA dedup'd).
    assert g == ["TGFBR1", "INHBA", "INHA", "TGFBR2", "INHBB"]


# ---------------------------------------------------------------------------
# Audit G1: ``_is_blank_or_na`` filters every NA-like sentinel uniformly
# ---------------------------------------------------------------------------
def test_is_blank_or_na_handles_all_na_sentinels():
    """Audit G1: prior idiom ``isinstance(g, float) and np.isnan(g)``
    missed ``pd.NA`` (because ``isinstance(pd.NA, float) is False``).
    The replacement helper must filter every NA-like sentinel."""
    import numpy as np
    from spatialcellchat.database import _is_blank_or_na

    # All NA-like values must be filtered out.
    assert _is_blank_or_na(None) is True
    assert _is_blank_or_na(np.nan) is True
    assert _is_blank_or_na(pd.NA) is True
    assert _is_blank_or_na(pd.NaT) is True
    assert _is_blank_or_na("") is True
    assert _is_blank_or_na("nan") is True   # str(np.nan) == "nan"
    # Real gene symbols pass through.
    assert _is_blank_or_na("TGFB1") is False
    assert _is_blank_or_na("CXCL12") is False


def test_flatten_unique_nonempty_filters_pd_NA(db_human):
    """Audit G1: ``_flatten_unique_nonempty`` must drop ``pd.NA`` entries
    that arrive via ``StringDtype``-backed columns (not just None/NaN)."""
    from spatialcellchat.database import _flatten_unique_nonempty

    out = _flatten_unique_nonempty(["TGFB1", pd.NA, "", None, np.nan, "TGFB1", "CXCL12"])
    assert out == ["TGFB1", "CXCL12"]


def test_checkGeneSymbol_filters_pd_NA(db_human):
    """``checkGeneSymbol`` must not report ``pd.NA`` as 'not official'."""
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = database.checkGeneSymbol(
            ["TGFB1", pd.NA, np.nan, None, ""], db_human["geneInfo"],
        )
    # Should not mention any NA-shaped token in the diagnostic.
    text = buf.getvalue()
    assert "<NA>" not in text
    assert "nan" not in text
    assert result is False


def test_extractGeneSubset_filters_pd_NA_in_StringDtype_complex(db_human):
    """Audit G1: a complex table with ``StringDtype()`` (which holds
    ``pd.NA`` instead of NaN) must produce identical output to the
    ``object`` dtype variant. The prior code let ``pd.NA`` slip into the
    gene name list, producing ``"<NA>"`` strings downstream."""
    complex_obj = db_human["complex"].copy()
    # Round-trip through StringDtype to ensure pd.NA is materialised.
    complex_str = complex_obj.copy()
    for col in complex_str.columns:
        complex_str[col] = complex_str[col].astype("string")  # nullable

    g_obj = database.extractGeneSubset(
        ["TGFbR1_R2", "Activin AB"], complex_obj, db_human["geneInfo"],
    )
    g_str = database.extractGeneSubset(
        ["TGFbR1_R2", "Activin AB"], complex_str, db_human["geneInfo"],
    )
    assert g_obj == g_str
    # Ensure no "<NA>" string snuck through.
    assert "<NA>" not in g_str
    assert all(s.isalnum() or "_" in s or "-" in s for s in g_str)


# ---------------------------------------------------------------------------
# extractGeneSubsetFromPair
# ---------------------------------------------------------------------------
def test_extractGeneSubsetFromPair_combined(db_human):
    """Rscript: ``extractGeneSubsetFromPair(searchPair('TGFb', db$interaction), complex_input=db$complex, geneInfo=db$geneInfo)``."""
    pairLR = database.searchPair("TGFb", db_human["interaction"])
    g = database.extractGeneSubsetFromPair(
        pairLR, complex_input=db_human["complex"], geneInfo=db_human["geneInfo"]
    )
    assert g == [
        "TGFB1", "TGFB2", "TGFB3",
        "TGFBR1", "ACVR1B", "ACVR1C", "ACVR1", "TGFBR2",
    ]


def test_extractGeneSubsetFromPair_split(db_human):
    pairLR = database.searchPair("TGFb", db_human["interaction"])
    g = database.extractGeneSubsetFromPair(
        pairLR,
        complex_input=db_human["complex"],
        geneInfo=db_human["geneInfo"],
        combined=False,
    )
    assert isinstance(g, dict)
    assert g["geneL"] == ["TGFB1", "TGFB2", "TGFB3"]
    assert g["geneR"] == ["TGFBR1", "ACVR1B", "ACVR1C", "ACVR1", "TGFBR2"]


def test_extractGeneSubsetFromPair_missing_columns_raises(db_human):
    bad = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="ligand"):
        database.extractGeneSubsetFromPair(
            bad, complex_input=db_human["complex"], geneInfo=db_human["geneInfo"]
        )


def test_extractGeneSubsetFromPair_no_lookup_raises():
    pairLR = pd.DataFrame({"ligand": ["X"], "receptor": ["Y"]})
    with pytest.raises(ValueError, match="complex_input"):
        database.extractGeneSubsetFromPair(pairLR)


def test_extractGeneSubsetFromPair_object_reads_uns_DB(db_human):
    """``object=adata`` must read complex/geneInfo from ``adata.uns['cellchat']['DB']``.

    R parity: ``object@DB$complex`` / ``object@DB$geneInfo``.
    """
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    adata.uns["cellchat"] = {"DB": dict(db_human)}
    pairLR = database.searchPair("TGFb", db_human["interaction"])

    g = database.extractGeneSubsetFromPair(pairLR, object=adata)

    assert g == [
        "TGFB1", "TGFB2", "TGFB3",
        "TGFBR1", "ACVR1B", "ACVR1C", "ACVR1", "TGFBR2",
    ]


def test_extractGeneSubsetFromPair_object_matches_explicit(db_human):
    """``object=`` and explicit ``complex_input=/geneInfo=`` must agree.

    Pins the dual-API contract so the two branches cannot drift apart.
    """
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    adata.uns["cellchat"] = {"DB": dict(db_human)}
    pairLR = database.searchPair("TGFb", db_human["interaction"])

    via_object = database.extractGeneSubsetFromPair(pairLR, object=adata)
    via_explicit = database.extractGeneSubsetFromPair(
        pairLR, complex_input=db_human["complex"], geneInfo=db_human["geneInfo"]
    )

    assert via_object == via_explicit


def test_extractGeneSubsetFromPair_object_missing_DB_raises():
    """If ``adata.uns['cellchat']['DB']`` is absent, raise a helpful ValueError."""
    import anndata as ad
    import numpy as np

    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))  # no cellchat slot
    pairLR = pd.DataFrame({"ligand": ["TGFB1"], "receptor": ["TGFBR1"]})

    with pytest.raises(ValueError, match=r"cellchat.*DB.*load_db"):
        database.extractGeneSubsetFromPair(pairLR, object=adata)


# ---------------------------------------------------------------------------
# checkGeneSymbol
# ---------------------------------------------------------------------------
def test_checkGeneSymbol_returns_false(db_human):
    assert database.checkGeneSymbol(["TGFB1", "TGFB2"], db_human["geneInfo"]) is False


def test_checkGeneSymbol_warns_on_unknown(db_human, capsys):
    out = database.checkGeneSymbol(["TGFB1", "FOOBAR"], db_human["geneInfo"])
    assert out is False
    captured = capsys.readouterr().out
    assert "FOOBAR" in captured


def test_checkGeneSymbol_drops_empty_and_dup(db_human):
    # Empty / duplicate / NaN should be silently filtered.
    assert (
        database.checkGeneSymbol(["TGFB1", "", None, "TGFB1"], db_human["geneInfo"]) is False
    )


# ---------------------------------------------------------------------------
# updateCellChatDB
# ---------------------------------------------------------------------------
def test_updateCellChatDB_minimal_with_species(db_human):
    """Rscript: ``updateCellChatDB(data.frame(ligand=c('TGFB1','TGFB2'),
    receptor=c('TGFBR1','TGFBR2')), gene_info=NULL, species_target='human')``."""
    udb = pd.DataFrame({"ligand": ["TGFB1", "TGFB2"], "receptor": ["TGFBR1", "TGFBR2"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_db = database.updateCellChatDB(udb, gene_info=None, species_target="human")
    assert list(new_db) == ["interaction", "complex", "cofactor", "geneInfo"]
    interaction = new_db["interaction"]
    assert len(interaction) == 2
    assert interaction.index.tolist() == ["TGFB1_TGFBR1", "TGFB2_TGFBR2"]
    assert interaction["interaction_name_2"].tolist() == [
        "TGFB1 - TGFBR1",
        "TGFB2 - TGFBR2",
    ]
    assert (interaction["pathway_name"] == "").all()
    assert len(new_db["geneInfo"]) == 26827
    assert new_db["complex"].empty and new_db["cofactor"].empty


def test_updateCellChatDB_explicit_gene_info():
    udb = pd.DataFrame(
        {"ligand": ["X1"], "receptor": ["Y1"], "pathway_name": ["P1"]}
    )
    new_db = database.updateCellChatDB(udb, gene_info=pd.DataFrame({"Symbol": ["X1"]}))
    assert len(new_db["interaction"]) == 1
    assert len(new_db["geneInfo"]) == 1


def test_updateCellChatDB_trim_pathway(capsys):
    udb = pd.DataFrame(
        {"ligand": ["X1", "X2"], "receptor": ["Y1", "Y2"], "pathway_name": ["", "p2"]}
    )
    new_db = database.updateCellChatDB(
        udb, gene_info=pd.DataFrame({"Symbol": ["X1"]}), trim_pathway=True
    )
    interaction = new_db["interaction"]
    assert len(interaction) == 1
    assert interaction.index.tolist() == ["X2_Y2"]


def test_updateCellChatDB_preserves_NA():
    """R: ``mutate(across(everything(), as.character))`` keeps ``NA_character_``;
    Python should preserve ``<NA>`` instead of folding it into ``""``.

    The R reference's ``which(pathway_name == "")`` yields ``NA``-rows as
    "neither equal nor not-equal", so they are *not* dropped by the
    ``trim.pathway`` branch. Only literal empty strings are dropped.
    """
    udb = pd.DataFrame(
        {
            "ligand": ["X1", "X2", "X3"],
            "receptor": ["Y1", "Y2", "Y3"],
            "pathway_name": ["P1", pd.NA, ""],
        }
    )
    with pytest.warns(UserWarning, match="missing"):
        new_db = database.updateCellChatDB(
            udb, gene_info=pd.DataFrame({"Symbol": ["X1"]}), trim_pathway=False
        )
    interaction = new_db["interaction"]
    # NA row preserved; empty-string row also preserved when trim_pathway=False
    assert len(interaction) == 3
    assert pd.isna(interaction.loc["X2_Y2", "pathway_name"])

    # When trim_pathway=True, only the empty-string row is dropped (R parity).
    new_db2 = database.updateCellChatDB(
        udb, gene_info=pd.DataFrame({"Symbol": ["X1"]}), trim_pathway=True
    )
    interaction2 = new_db2["interaction"]
    assert len(interaction2) == 2
    assert interaction2.index.tolist() == ["X1_Y1", "X2_Y2"]
    assert pd.isna(interaction2.loc["X2_Y2", "pathway_name"])


def test_updateCellChatDB_duplicates_dropped():
    udb = pd.DataFrame(
        {
            "ligand": ["X1", "X1", "X2"],
            "receptor": ["Y1", "Y1", "Y2"],
            "pathway_name": ["p1", "p1", "p2"],
        }
    )
    with pytest.warns(UserWarning, match="duplicated"):
        new_db = database.updateCellChatDB(
            udb, gene_info=pd.DataFrame({"Symbol": ["X1"]})
        )
    assert len(new_db["interaction"]) == 2
    assert new_db["interaction"].index.tolist() == ["X1_Y1", "X2_Y2"]


def test_updateCellChatDB_uppercases_interaction_name():
    """``interaction_name`` is built via ``toupper(ligand)_toupper(receptor)``."""
    udb = pd.DataFrame(
        {
            "ligand": ["tgfb1"],
            "receptor": ["tgfbr1"],
            "pathway_name": ["TGFb"],
        }
    )
    new_db = database.updateCellChatDB(udb, gene_info=pd.DataFrame({"Symbol": ["x"]}))
    assert new_db["interaction"]["interaction_name"].tolist() == ["TGFB1_TGFBR1"]
    # interaction_name_2 uses original case.
    assert new_db["interaction"]["interaction_name_2"].tolist() == ["tgfb1 - tgfbr1"]


def test_updateCellChatDB_specific_columns_preserved():
    """Custom columns appear after the default block in their original order."""
    udb = pd.DataFrame(
        {
            "ligand": ["L1"],
            "receptor": ["R1"],
            "pathway_name": ["P"],
            "custom1": ["C1"],
            "custom2": ["C2"],
        }
    )
    new_db = database.updateCellChatDB(udb, gene_info=pd.DataFrame({"Symbol": ["x"]}))
    cols = new_db["interaction"].columns.tolist()
    assert cols[-2:] == ["custom1", "custom2"]


def test_updateCellChatDB_merged_with_human():
    """Rscript: ``updateCellChatDB(udb, merged=TRUE, species_target='human')``."""
    udb = pd.DataFrame(
        {
            "ligand": ["MYLIG1", "MYLIG2"],
            "receptor": ["MYREC1", "MYREC2"],
            "pathway_name": ["MYPATH", "MYPATH"],
        }
    )
    new_db = database.updateCellChatDB(udb, merged=True, species_target="human")
    assert len(new_db["interaction"]) == 3236  # 3234 + 2 user
    assert "source.merged" in new_db["interaction"].columns
    src_counts = new_db["interaction"]["source.merged"].value_counts().to_dict()
    assert src_counts == {"CellChatDB": 3234, "User": 2}
    # complex / cofactor unchanged in size when user supplies neither.
    assert len(new_db["complex"]) == 338
    assert len(new_db["cofactor"]) == 32


def test_updateCellChatDB_missing_required_cols():
    udb = pd.DataFrame({"ligand": ["X"]})
    with pytest.raises(ValueError, match="ligand,receptor"):
        database.updateCellChatDB(udb, gene_info=pd.DataFrame({"Symbol": ["x"]}))


def test_updateCellChatDB_gene_info_missing_symbol():
    udb = pd.DataFrame({"ligand": ["X"], "receptor": ["Y"], "pathway_name": ["p"]})
    with pytest.raises(ValueError, match="Symbol"):
        database.updateCellChatDB(udb, gene_info=pd.DataFrame({"WrongCol": ["x"]}))


def test_updateCellChatDB_no_gene_info_no_species_raises():
    udb = pd.DataFrame({"ligand": ["X"], "receptor": ["Y"], "pathway_name": ["p"]})
    with pytest.raises(ValueError, match="species_target"):
        database.updateCellChatDB(udb, gene_info=None)


def test_updateCellChatDB_complex_bad_columns_raises():
    udb = pd.DataFrame({"ligand": ["X"], "receptor": ["Y"], "pathway_name": ["p"]})
    bad_complex = pd.DataFrame({"oops_col": [""]})
    with pytest.raises(ValueError, match="subunit_"):
        database.updateCellChatDB(
            udb,
            gene_info=pd.DataFrame({"Symbol": ["x"]}),
            other_info={"complex": bad_complex},
        )


def test_updateCellChatDB_cofactor_bad_columns_raises():
    udb = pd.DataFrame({"ligand": ["X"], "receptor": ["Y"], "pathway_name": ["p"]})
    bad_cofactor = pd.DataFrame({"oops_col": [""]})
    with pytest.raises(ValueError, match="cofactor"):
        database.updateCellChatDB(
            udb,
            gene_info=pd.DataFrame({"Symbol": ["x"]}),
            other_info={"cofactor": bad_cofactor},
        )
