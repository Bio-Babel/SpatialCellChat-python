"""CellChatDB and PPI loaders + database query helpers.

Provides ``load_db(species)`` and ``load_ppi(species)`` for accessing the
bundled CellChatDB / PPI assets, plus higher-level query helpers
(``subsetDB``, ``searchPair``, ``extractGene``, ``extractGeneSubset``,
``extractGeneSubsetFromPair``, ``checkGeneSymbol``, ``updateCellChatDB``).

These mirror the corresponding R functions in ``SpatialCellChat::database.R``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "load_db",
    "load_ppi",
    "DB_SPECIES",
    "PPI_SPECIES",
    "subsetDB",
    "searchPair",
    "extractGene",
    "extractGeneSubset",
    "extractGeneSubsetFromPair",
    "checkGeneSymbol",
    "updateCellChatDB",
]

_RESOURCES = Path(__file__).resolve().parent / "resources" / "db"

DB_SPECIES = ("human", "mouse", "zebrafish")
PPI_SPECIES = ("human", "mouse")

_DF_NAMES = ("interaction", "complex", "cofactor", "geneInfo")

# Default column order applied by ``updateCellChatDB`` when assembling the
# interaction frame (matches the R ``cols.default``).
_UPDATE_DEFAULT_COLS = (
    "interaction_name",
    "pathway_name",
    "ligand",
    "receptor",
    "agonist",
    "antagonist",
    "co_A_receptor",
    "co_I_receptor",
    "annotation",
    "interaction_name_2",
)


def load_db(species: Literal["human", "mouse", "zebrafish"]) -> dict[str, pd.DataFrame]:
    """Load a CellChatDB.

    Parameters
    ----------
    species : {"human", "mouse", "zebrafish"}
        Species code.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys ``"interaction"``, ``"complex"``, ``"cofactor"``, ``"geneInfo"``;
        each value is the corresponding bundled data frame. The
        ``"interaction"`` table is indexed by ``interaction_name`` (matches the
        R object's row names).
    """
    if species not in DB_SPECIES:
        raise ValueError(f"species must be one of {DB_SPECIES!r}; got {species!r}")
    species_dir = _RESOURCES / species
    out: dict[str, pd.DataFrame] = {}
    for name in _DF_NAMES:
        df = pd.read_parquet(species_dir / f"{name}.parquet")
        if name == "interaction" and "interaction_name" in df.columns:
            df = df.set_index("interaction_name", drop=False)
            df.index.name = None
            if "rowname" in df.columns:
                df = df.drop(columns="rowname")
        if name == "complex" and "rowname" in df.columns:
            df = df.set_index("rowname", drop=True)
            df.index.name = None
        if name == "cofactor" and "rowname" in df.columns:
            df = df.set_index("rowname", drop=True)
            df.index.name = None
        out[name] = df
    return out


def load_ppi(species: Literal["human", "mouse"]) -> tuple[sp.csr_matrix, list[str]]:
    """Load a PPI sparse matrix.

    Parameters
    ----------
    species : {"human", "mouse"}
        Species code.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, list[str]]
        The square symmetric PPI adjacency matrix and the gene symbol list
        (row/column labels are identical, in matrix order).
    """
    if species not in PPI_SPECIES:
        raise ValueError(f"species must be one of {PPI_SPECIES!r}; got {species!r}")
    species_dir = _RESOURCES / species
    mat = sp.load_npz(species_dir / "PPI.npz").tocsr()
    rows = (species_dir / "PPI.rows.txt").read_text().splitlines()
    if mat.shape[0] != len(rows):
        raise RuntimeError(
            f"PPI shape mismatch: matrix {mat.shape}, rows {len(rows)}"
        )
    return mat, rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _as_str_series(s: pd.Series) -> pd.Series:
    """Return a ``str``-typed copy of *s* with ``NaN`` / ``None`` rendered as ``""``.

    The R source treats absent cells as the empty string ``""`` (because the
    bundled R data uses character vectors with empty strings instead of
    ``NA``); the Python parquet round-trip exposes those cells as ``None`` /
    ``NaN``. Normalising both to ``""`` keeps the boolean comparisons faithful
    to R.
    """
    return s.where(s.notna(), "").astype(str)


def _is_blank_or_na(v: object) -> bool:
    """Return ``True`` iff scalar ``v`` should be filtered out as missing / empty.

    Mirrors R's ``x != ""`` semantics for gene-symbol vectors. R's bundled
    CellChatDB uses character vectors with empty strings ``""`` instead of
    ``NA``; the parquet round-trip on the Python side can surface absent
    cells as any of ``None``, ``np.nan``, ``pd.NA`` (nullable string
    dtype), ``pd.NaT``, or the literal string ``"nan"`` (post-``str()``
    coercion of a NumPy NaN). All five must be uniformly filtered out.

    Why ``pd.isna`` and not ``isinstance(v, float) and np.isnan(v)``: the
    older idiom misses ``pd.NA`` (audit G1: G1-class latent bug) because
    ``isinstance(pd.NA, float) is False`` and ``str(pd.NA) == "<NA>"``.
    ``pd.isna`` handles every NA-like sentinel uniformly.

    Contract: ``v`` must be a scalar (one cell of a DataFrame). Passing
    an array would make ``pd.isna`` return an array and the truth-value
    check would raise — that's by design (callers iterate scalar values).
    """
    # ``pd.isna`` accepts ``None``, ``np.nan``, ``pd.NA``, ``pd.NaT`` and
    # returns ``True`` for all of them. Returns ``False`` for actual
    # strings, ints, floats, etc.
    if pd.isna(v):
        return True
    s = str(v)
    return s == "" or s == "nan"


def _flatten_unique_nonempty(values: Iterable) -> list[str]:
    """Flatten an iterable into a unique list of non-empty / non-NA strings (in order).

    Mirrors the R idiom ``unique(x[x != ""])`` after ``unlist(...)``.
    Uses :func:`_is_blank_or_na` so ``pd.NA``-backed nullable-string
    columns are filtered correctly (see audit G1).
    """
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if _is_blank_or_na(v):
            continue
        s = str(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# subsetDB
# ---------------------------------------------------------------------------
def subsetDB(
    CellChatDB: Mapping[str, pd.DataFrame],
    search: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    key: str | Sequence[str] = "annotation",
    non_protein: bool = False,
) -> dict[str, pd.DataFrame]:
    """Subset CellChatDB by selecting interactions matching ``search`` on ``key``.

    Parameters
    ----------
    CellChatDB : Mapping[str, pandas.DataFrame]
        A CellChatDB dict (as returned by :func:`load_db`).
    search : str | Sequence[str] | Sequence[Sequence[str]] | None, optional
        Values to match against ``key``. When ``key`` is a string, ``search``
        may be a string or a sequence of strings. When ``key`` is a vector,
        ``search`` must be a sequence of sequences with the same length as
        ``key``. ``None`` (default) selects the standard protein signaling
        annotations (or all four when ``non_protein`` is ``True``).
    key : str | Sequence[str], optional
        Column name(s) of ``CellChatDB["interaction"]`` to filter on. Default
        ``"annotation"``.
    non_protein : bool, optional
        If ``False`` (default), rows annotated as ``"Non-protein Signaling"``
        are dropped first.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A new dict with the same keys as the input and a filtered
        ``"interaction"`` table. The other tables are returned unchanged.
    """
    interaction_input = CellChatDB["interaction"]

    # R parity: dispatch on ``length(key) == 1``, NOT on whether ``key`` is a
    # string. So ``key = ["annotation"]`` (length-1 list) and
    # ``key = "annotation"`` (string) are equivalent — both follow the
    # scalar branch.
    key_list: list[str] = [key] if isinstance(key, str) else list(key)
    key_is_scalar = len(key_list) == 1

    # Default search expansion — R: ``is.null(search) & any(key == "annotation")``.
    if search is None:
        if "annotation" in key_list:
            if not non_protein:
                search = ["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"]
            else:
                search = [
                    "Secreted Signaling",
                    "ECM-Receptor",
                    "Cell-Cell Contact",
                    "Non-protein Signaling",
                ]

    # Activate non-protein flag if user explicitly asked for that annotation.
    flat_search: list[str] = []
    if isinstance(search, str):
        flat_search = [search]
    elif search is not None:
        for item in search:
            if isinstance(item, str):
                flat_search.append(item)
            else:
                flat_search.extend(list(item))
    if "Non-protein Signaling" in flat_search:
        non_protein = True
        warnings.warn(
            "The non-protein signaling is now included for CellChat analysis, "
            "which is usually used for neuron-neuron and metabolic communication!",
            stacklevel=2,
        )

    if not non_protein:
        annotation = _as_str_series(interaction_input["annotation"])
        interaction_input = interaction_input.loc[annotation != "Non-protein Signaling"]

    # Validate key.
    missing_keys = [k for k in key_list if k not in interaction_input.columns]
    if missing_keys:
        raise ValueError(
            "Each element of the `key` should be one of the column names of "
            "the interaction_input from CellChatDB"
        )

    if key_is_scalar:
        # R: ``interaction_input[interaction_input[[key]] %in% search, ]``.
        # ``search`` may be (a) a flat string/list/tuple, or (b) a length-1
        # list-of-lists when the caller used the vector-style API with
        # length-1 ``key``. Flatten one level so both forms work.
        if isinstance(search, str):
            targets = [search]
        elif search is None:
            targets = []
        elif (
            len(search) == 1
            and not isinstance(search[0], str)
            and hasattr(search[0], "__iter__")
        ):
            targets = list(search[0])
        else:
            targets = list(search)
        col = _as_str_series(interaction_input[key_list[0]])
        interaction_input = interaction_input.loc[col.isin(targets)]
    else:
        # R: ``length(key) > 1`` requires ``search`` to be a list of vectors.
        if not isinstance(search, (list, tuple)) or any(
            isinstance(s, str) for s in (search or [])
        ):
            raise TypeError(
                "When `key` is a vector, the `search` should be a list."
            )
        if len(search) != len(key_list):
            raise ValueError(
                "When `key` is a vector, the `search` should be a list with "
                "the size being `length(key)`."
            )
        idx = pd.Series(True, index=interaction_input.index)
        for k, targets in zip(key_list, search):
            col = _as_str_series(interaction_input[k])
            idx = idx & col.isin(list(targets))
        interaction_input = interaction_input.loc[idx]

    out = dict(CellChatDB)
    out["interaction"] = interaction_input
    return out


# ---------------------------------------------------------------------------
# searchPair
# ---------------------------------------------------------------------------
def searchPair(
    signaling: str | Sequence[str],
    pairLR_use: pd.DataFrame,
    key: Literal["pathway_name", "ligand"] = "pathway_name",
    matching_exact: bool = False,
    pair_only: bool = True,
) -> pd.DataFrame:
    """Subset L-R interactions for the given signals.

    Parameters
    ----------
    signaling : str | Sequence[str]
        Signal name(s) to look up in ``key`` column.
    pairLR_use : pandas.DataFrame
        L-R pair table (typically ``CellChatDB$interaction`` or output from
        :func:`extractGene*`).
    key : {"pathway_name", "ligand"}, optional
        Column to match on. Default ``"pathway_name"``.
    matching_exact : bool, optional
        If ``False`` (default), match by substring (mirrors R's ``grep``
        without anchors). If ``True``, exact equality.
    pair_only : bool, optional
        If ``True`` (default), keep only the four columns
        ``interaction_name``, ``pathway_name``, ``ligand``, ``receptor``;
        otherwise return all columns.

    Returns
    -------
    pandas.DataFrame
        A frame indexed by the rows' ``interaction_name``, containing the
        union of matches across the requested ``signaling`` entries.
    """
    if key not in ("pathway_name", "ligand"):
        raise ValueError("`key` should be one of 'pathway_name', 'ligand'.")

    if isinstance(signaling, str):
        signaling_list: list[str] = [signaling]
    else:
        signaling_list = list(signaling)

    col = _as_str_series(pairLR_use[key])
    parts: list[pd.DataFrame] = []
    for sig in signaling_list:
        if matching_exact:
            mask = col == sig
        else:
            mask = col.str.contains(sig, regex=True, na=False)
        if not mask.any():
            raise ValueError(
                f"Cannot find  {sig} . Please input a correct name!"
            )
        sub = pairLR_use.loc[mask]
        if pair_only:
            sub = sub.loc[:, ["interaction_name", "pathway_name", "ligand", "receptor"]]
        parts.append(sub)

    pairLR = pd.concat(parts, axis=0)
    # R coerces all values to character via ``matrix(unlist(...))`` and then
    # ``as.data.frame``. Mirror that by casting object/numeric columns to str.
    pairLR = pairLR.astype(object).where(pairLR.notna(), "").astype(str)
    pairLR.index = pd.Index(pairLR["interaction_name"].tolist())
    return pairLR


# ---------------------------------------------------------------------------
# extractGene + helpers
# ---------------------------------------------------------------------------
def extractGene(CellChatDB: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return all gene symbols referenced in the CellChatDB.

    Parameters
    ----------
    CellChatDB : Mapping[str, pandas.DataFrame]
        A CellChatDB dict (as returned by :func:`load_db`).

    Returns
    -------
    list[str]
        Unique union of ligand / receptor symbols, complex subunits, and
        cofactor genes referenced from agonist / antagonist /
        co_A_receptor / co_I_receptor columns.
    """
    interaction_input = CellChatDB["interaction"]
    complex_input = CellChatDB["complex"]
    cofactor_input = CellChatDB["cofactor"]
    geneInfo = CellChatDB["geneInfo"]

    # ``checkGeneSymbol`` is invoked for its diagnostic side effect on complex
    # and cofactor frames, then on the ligand/receptor symbols that are not
    # complex names. The R version returns ``FALSE`` regardless.
    checkGeneSymbol(complex_input.values.ravel().tolist(), geneInfo)
    checkGeneSymbol(cofactor_input.values.ravel().tolist(), geneInfo)

    geneL = list(dict.fromkeys(_as_str_series(interaction_input["ligand"]).tolist()))
    geneR = list(dict.fromkeys(_as_str_series(interaction_input["receptor"]).tolist()))
    geneLR = geneL + geneR
    complex_names = set(complex_input.index.tolist())
    checkGeneSymbol([g for g in geneLR if g not in complex_names], geneInfo)

    geneL = extractGeneSubset(geneL, complex_input, geneInfo)
    geneR = extractGeneSubset(geneR, complex_input, geneInfo)
    geneLR = geneL + geneR

    cofactor_cols = ["agonist", "antagonist", "co_A_receptor", "co_I_receptor"]
    cofactor_values: list[str] = []
    for c in cofactor_cols:
        cofactor_values.extend(_as_str_series(interaction_input[c]).tolist())
    cofactor = _flatten_unique_nonempty(cofactor_values)

    # Match cofactor names against cofactor_input rownames, preserving the
    # order in which they appear in ``cofactor`` (mirrors R's ``match``).
    cofactor_index_set = set(cofactor_input.index)
    matched = [g for g in cofactor if g in cofactor_index_set]
    cofactor_subset = cofactor_input.loc[matched]
    sub_cols = [c for c in cofactor_subset.columns if c.startswith("cofactor")]
    cofactor_subunits_v = (
        cofactor_subset.loc[:, sub_cols].values.ravel(order="F").tolist()
    )
    geneCofactor = _flatten_unique_nonempty(cofactor_subunits_v)

    # Final unique union.
    seen: set[str] = set()
    out: list[str] = []
    for g in geneLR + geneCofactor:
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def extractGeneSubset(
    geneSet: Sequence[str],
    complex_input: pd.DataFrame,
    geneInfo: pd.DataFrame,
) -> list[str]:
    """Expand a gene set by replacing complex names with their subunits.

    Parameters
    ----------
    geneSet : Sequence[str]
        Gene / complex names.
    complex_input : pandas.DataFrame
        CellChatDB complex frame (rows indexed by complex name, columns
        ``subunit_*``).
    geneInfo : pandas.DataFrame
        CellChatDB gene info frame (must have a ``Symbol`` column).

    Returns
    -------
    list[str]
        Gene symbols: official symbols from ``geneSet`` plus subunits of any
        complex entries in ``geneSet``. Duplicates removed, order preserved.
    """
    symbols = set(geneInfo["Symbol"].tolist())
    geneSet_list = list(geneSet)
    is_official = [g in symbols for g in geneSet_list]
    complex_candidates = [g for g, ok in zip(geneSet_list, is_official) if not ok]
    official = [g for g, ok in zip(geneSet_list, is_official) if ok]

    complex_index = complex_input.index
    matched = [g for g in complex_candidates if g in set(complex_index)]
    complex_subset = complex_input.loc[matched]
    sub_cols = [c for c in complex_subset.columns if c.startswith("subunit")]
    # R's ``unlist`` traverses a data frame column-major; mimic that with
    # ``ravel(order="F")``.
    subunit_values = complex_subset.loc[:, sub_cols].values.ravel(order="F").tolist()
    subunits = _flatten_unique_nonempty(subunit_values)

    seen: set[str] = set()
    out: list[str] = []
    for g in official + subunits:
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def extractGeneSubsetFromPair(
    pairLR: pd.DataFrame,
    object: ad.AnnData | None = None,
    complex_input: pd.DataFrame | None = None,
    geneInfo: pd.DataFrame | None = None,
    combined: bool = True,
) -> list[str] | dict[str, list[str]]:
    """Extract signaling gene names referenced by an L-R pair table.

    Parameters
    ----------
    pairLR : pandas.DataFrame
        L-R pair table containing at least ``ligand`` and ``receptor`` cols.
    object : anndata.AnnData | None, optional
        A CellChat-state AnnData. When provided, the ``complex`` and
        ``geneInfo`` tables are read from
        ``object.uns['cellchat']['DB']`` (R parity:
        ``object@DB$complex`` / ``object@DB$geneInfo``).
    complex_input : pandas.DataFrame | None, optional
        Complex table (used if ``object`` is ``None``).
    geneInfo : pandas.DataFrame | None, optional
        Gene info table (used if ``object`` is ``None``).
    combined : bool, optional
        If ``True`` (default), return a single flat list. If ``False``,
        return a dict ``{"geneL": [...], "geneR": [...]}``.

    Returns
    -------
    list[str] | dict[str, list[str]]
        Gene symbols implied by the ligand / receptor columns.
    """
    if not all(c in pairLR.columns for c in ("ligand", "receptor")):
        raise ValueError(
            "The input data frame must contain columns named `ligand` and `receptor`"
        )
    if object is None:
        if complex_input is None or geneInfo is None:
            raise ValueError(
                "Either `object` or `complex_input` and `geneInfo` should be provided!"
            )
        complex_used = complex_input
        gene_info_used = geneInfo
    else:
        # R parity: object@DB$complex / object@DB$geneInfo
        # Python port stores DB under adata.uns['cellchat']['DB'].
        if "cellchat" not in object.uns or "DB" not in object.uns["cellchat"]:
            raise ValueError(
                "object.uns['cellchat']['DB'] is missing. Pass `complex_input=` "
                "and `geneInfo=` explicitly, or attach the DB via "
                "load_db / updateCellChatDB."
            )
        db = object.uns["cellchat"]["DB"]
        complex_used = db["complex"]
        gene_info_used = db["geneInfo"]

    geneL = list(dict.fromkeys(_as_str_series(pairLR["ligand"]).tolist()))
    geneR = list(dict.fromkeys(_as_str_series(pairLR["receptor"]).tolist()))
    geneL = extractGeneSubset(geneL, complex_used, gene_info_used)
    geneR = extractGeneSubset(geneR, complex_used, gene_info_used)
    if combined:
        seen: set[str] = set()
        out: list[str] = []
        for g in geneL + geneR:
            if g and g not in seen:
                seen.add(g)
                out.append(g)
        return out
    return {"geneL": geneL, "geneR": geneR}


# ---------------------------------------------------------------------------
# checkGeneSymbol
# ---------------------------------------------------------------------------
def checkGeneSymbol(
    geneSet: Sequence[str],
    geneInfo: pd.DataFrame,
) -> bool:
    """Warn about gene symbols not present in ``geneInfo$Symbol``.

    Parameters
    ----------
    geneSet : Sequence[str]
        Gene symbols to check (may contain ``""``, ``None`` or ``NaN``).
    geneInfo : pandas.DataFrame
        Reference gene table with column ``Symbol``.

    Returns
    -------
    bool
        Always ``False`` (matches R's ``return(FALSE)``).
    """
    symbols = set(geneInfo["Symbol"].tolist())
    # Inline filter previously missed ``pd.NA`` (audit G1). Route through
    # the shared :func:`_is_blank_or_na` predicate so all NA sentinels
    # (``None``, ``np.nan``, ``pd.NA``, ``pd.NaT``) are handled uniformly.
    seen: set[str] = set()
    cleaned: list[str] = []
    for g in geneSet:
        if _is_blank_or_na(g):
            continue
        s = str(g)
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    not_official = [g for g in cleaned if g not in symbols]
    if not_official:
        # ``cat`` + ``\n`` in R produces a console message; we use stdout via
        # ``print`` to keep it visible without raising. ``warnings`` would be
        # filtered in pytest more aggressively.
        print(
            "Issue identified!! Please check the official Gene Symbol of the "
            "following genes:  \n " + " ".join(not_official) + " "
        )
    return False


# ---------------------------------------------------------------------------
# updateCellChatDB
# ---------------------------------------------------------------------------
def updateCellChatDB(
    db: pd.DataFrame,
    gene_info: pd.DataFrame | None = None,
    other_info: Mapping[str, pd.DataFrame] | None = None,
    trim_pathway: bool = False,
    merged: bool = False,
    species_target: Literal["human", "mouse"] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build (or augment) a CellChatDB from a user-supplied L-R table.

    Parameters
    ----------
    db : pandas.DataFrame
        Custom L-R pair table. Must contain ``ligand`` and ``receptor``
        columns; ``pathway_name`` is strongly recommended.
    gene_info : pandas.DataFrame | None, optional
        Gene info frame with at least a ``Symbol`` column. If ``None``,
        ``species_target`` must be supplied so the bundled CellChatDB gene
        info can be reused.
    other_info : Mapping[str, pandas.DataFrame] | None, optional
        Optional dict with optional keys ``"complex"`` and ``"cofactor"``.
        Their columns must follow ``subunit_<n>`` / ``cofactor<n>`` naming.
    trim_pathway : bool, optional
        If ``True``, drop interactions whose ``pathway_name`` is empty.
    merged : bool, optional
        If ``True``, merge the result with the bundled CellChatDB for
        ``species_target``.
    species_target : {"human", "mouse"} | None, optional
        Required when ``gene_info=None`` or ``merged=True``.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A new CellChatDB dict (``interaction``, ``complex``, ``cofactor``,
        ``geneInfo``).
    """
    db = db.copy()
    # mutate(across(everything(), as.character)) — R preserves NA_character_,
    # so we cast to pandas StringDtype to keep <NA> distinct from "" rather
    # than forcing it to "" with ``_as_str_series``.
    for c in db.columns:
        db[c] = db[c].astype("string")

    if not all(c in db.columns for c in ("ligand", "receptor")):
        raise ValueError(
            "The input `db` must contain at least two columns named as ligand,receptor"
        )

    if "pathway_name" not in db.columns:
        warnings.warn(
            "The pathway_name associated with each L-R pair is not provided "
            "in `db`. We suggest to provide this information so that the "
            "versatile functionalities of CellChat can be fully used!",
            stacklevel=2,
        )
        db["pathway_name"] = pd.array([""] * len(db), dtype="string")
    else:
        # R: ``pathway.missing <- which(db$pathway_name == "")`` — equality
        # against "" yields ``NA`` for ``NA_character_`` rows, which
        # ``which`` drops. Mirror that by treating only literal empty strings
        # as missing.
        pathway_missing = db.index[(db["pathway_name"] == "").fillna(False)]
        if len(pathway_missing) > 0:
            if trim_pathway:
                print(
                    f"The pathway names of {len(pathway_missing)} interactions "
                    "are missing and the corresponding interactions are now deleted."
                )
                db = db.drop(index=pathway_missing)
            else:
                warnings.warn(
                    f"The pathway names of {len(pathway_missing)} interactions "
                    "are missing and it may cause error in the downstream "
                    "analysis. Setting `trim.pathway = TRUE` to avoid such "
                    "possible errors.",
                    stacklevel=2,
                )

    if "interaction_name" not in db.columns:
        db["interaction_name"] = (
            db["ligand"].str.upper() + "_" + db["receptor"].str.upper()
        )
    if "interaction_name_2" not in db.columns:
        db["interaction_name_2"] = db["ligand"] + " - " + db["receptor"]
    for c in ("agonist", "antagonist", "co_A_receptor", "co_I_receptor"):
        if c not in db.columns:
            db[c] = [""] * len(db)

    # Drop duplicates by interaction_name (R uses ``duplicated`` which keeps
    # the first occurrence).
    dup_mask = db["interaction_name"].duplicated(keep="first")
    if dup_mask.any():
        warnings.warn(
            f"{int(dup_mask.sum())} duplicated interaction_names are identified "
            "and the corresponding interactions are now deleted.",
            stacklevel=2,
        )
        db = db.loc[~dup_mask]

    interaction_input = db.copy()
    interaction_input.index = pd.Index(interaction_input["interaction_name"].tolist())
    cols_default = list(_UPDATE_DEFAULT_COLS)
    cols_common = [c for c in cols_default if c in interaction_input.columns]
    cols_specific = [c for c in interaction_input.columns if c not in cols_default]
    interaction_input = interaction_input.loc[:, cols_common + cols_specific]

    # Build complex / cofactor frames.
    if other_info is not None:
        if "complex" in other_info:
            complex_input = other_info["complex"].copy()
            allowed = {f"subunit_{i}" for i in range(1, 101)}
            if not all(c in allowed for c in complex_input.columns):
                raise ValueError(
                    "The colnames of the input `other_info$complex` should be "
                    "`subunit_1`,`subunit_2`,..."
                )
        else:
            complex_input = pd.DataFrame()
        if "cofactor" in other_info:
            cofactor_input = other_info["cofactor"].copy()
            allowed = {f"cofactor{i}" for i in range(1, 101)}
            if not all(c in allowed for c in cofactor_input.columns):
                raise ValueError(
                    "The colnames of the input `other_info$cofactor` should be "
                    "`cofactor1`,`cofactor2`,..."
                )
        else:
            cofactor_input = pd.DataFrame()
    else:
        complex_input = pd.DataFrame()
        cofactor_input = pd.DataFrame()

    # geneInfo
    if gene_info is not None:
        if "Symbol" not in gene_info.columns:
            raise ValueError(
                "The input `gene_info` must contain at least one column named as `Symbol`"
            )
    else:
        if species_target is None:
            raise ValueError(
                "When setting gene_info = NULL, the input `species_target` "
                "should be provided: either `human` or `mouse`."
            )
        if species_target not in ("human", "mouse"):
            raise ValueError("`species_target` must be 'human' or 'mouse'.")
        gene_info = load_db(species_target)["geneInfo"]
    geneInfo_input = gene_info

    if merged:
        if species_target is None:
            raise ValueError(
                "When setting merged = TRUE, the input `species_target` should "
                "be provided: either `human` or `mouse`."
            )
        if species_target not in ("human", "mouse"):
            raise ValueError("`species_target` must be 'human' or 'mouse'.")
        db_cellchat = load_db(species_target)
        print(f"Starting to merge the input database with CellChatDB.{species_target}...")

        interaction_cellchat = db_cellchat["interaction"].copy()
        interaction_cellchat["source.merged"] = "CellChatDB"
        interaction_input["source.merged"] = "User"
        cols_common_merge = [
            c for c in interaction_input.columns if c in interaction_cellchat.columns
        ]
        # Preserve order from interaction_input.
        interaction_input_m = interaction_input.loc[:, cols_common_merge]
        interaction_cellchat_m = interaction_cellchat.loc[:, cols_common_merge]
        interaction_merged = pd.concat(
            [interaction_cellchat_m, interaction_input_m], axis=0
        )
        dup_mask_m = interaction_merged["interaction_name"].duplicated(keep="first")
        if dup_mask_m.any():
            interaction_merged = interaction_merged.loc[~dup_mask_m]

        # Complex merge with subunit padding.
        complex_cellchat = db_cellchat["complex"].copy()
        num_subunit = max(complex_input.shape[1], complex_cellchat.shape[1])
        if complex_input.shape[1] < num_subunit:
            n_extra = num_subunit - complex_input.shape[1]
            extra = pd.DataFrame(
                "", index=complex_input.index,
                columns=[f"subunit_{i}" for i in range(complex_input.shape[1] + 1, num_subunit + 1)],
            )
            complex_input = pd.concat([complex_input, extra], axis=1)
            complex_input.columns = [f"subunit_{i}" for i in range(1, num_subunit + 1)]
        if complex_cellchat.shape[1] < num_subunit:
            n_extra = num_subunit - complex_cellchat.shape[1]
            extra = pd.DataFrame(
                "", index=complex_cellchat.index,
                columns=[f"subunit_{i}" for i in range(complex_cellchat.shape[1] + 1, num_subunit + 1)],
            )
            complex_cellchat = pd.concat([complex_cellchat, extra], axis=1)
            complex_cellchat.columns = [f"subunit_{i}" for i in range(1, num_subunit + 1)]
        complex_merged = pd.concat([complex_cellchat, complex_input], axis=0)
        dup_idx = complex_merged.index.duplicated(keep="first")
        if dup_idx.any():
            complex_merged = complex_merged.loc[~dup_idx]

        # Cofactor merge with subunit padding.
        cofactor_cellchat = db_cellchat["cofactor"].copy()
        num_subunit = max(cofactor_input.shape[1], cofactor_cellchat.shape[1])
        if cofactor_input.shape[1] < num_subunit:
            extra = pd.DataFrame(
                "", index=cofactor_input.index,
                columns=[f"cofactor{i}" for i in range(cofactor_input.shape[1] + 1, num_subunit + 1)],
            )
            cofactor_input = pd.concat([cofactor_input, extra], axis=1)
            cofactor_input.columns = [f"cofactor{i}" for i in range(1, num_subunit + 1)]
        if cofactor_cellchat.shape[1] < num_subunit:
            extra = pd.DataFrame(
                "", index=cofactor_cellchat.index,
                columns=[f"cofactor{i}" for i in range(cofactor_cellchat.shape[1] + 1, num_subunit + 1)],
            )
            cofactor_cellchat = pd.concat([cofactor_cellchat, extra], axis=1)
            cofactor_cellchat.columns = [f"cofactor{i}" for i in range(1, num_subunit + 1)]
        cofactor_merged = pd.concat([cofactor_cellchat, cofactor_input], axis=0)
        dup_idx = cofactor_merged.index.duplicated(keep="first")
        if dup_idx.any():
            cofactor_merged = cofactor_merged.loc[~dup_idx]

        interaction_input = interaction_merged
        complex_input = complex_merged
        cofactor_input = cofactor_merged

    return {
        "interaction": interaction_input,
        "complex": complex_input,
        "cofactor": cofactor_input,
        "geneInfo": geneInfo_input,
    }
