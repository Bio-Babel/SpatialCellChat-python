"""Load and filter the CellChatDB ligand-receptor database.

Maps onto workflows/ligand_receptor_db_query.yaml's steps:

  1. load_db    -> database.load_db
  2. subsetDB   -> database.subsetDB
  3. searchPair -> database.searchPair (optional pathway lookup)
"""

from __future__ import annotations


def main() -> None:
    import spatialcellchat as scc

    # Step 1: load the full bundled database for one species.
    db = scc.load_db("human")

    # Step 2: filter to the categories used for spatial CCC.
    db_use = scc.subsetDB(
        db,
        search=["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"],
        non_protein=False,
    )
    print(
        f"db: {len(db['interaction'])} interactions -> "
        f"db_use: {len(db_use['interaction'])} after category filtering"
    )

    # Step 3 (optional): look up the L-R pairs of one pathway.
    pathway_name = db_use["interaction"]["pathway_name"].iloc[0]
    pairLR = scc.searchPair(
        pathway_name, db_use["interaction"], key="pathway_name",
        matching_exact=True, pair_only=True,
    )
    print(f"pathway {pathway_name!r}: {len(pairLR)} L-R pair(s)")

    # `db_use` is what a downstream workflow registers as
    # adata.uns["cellchat"]["DB"] = db_use before running
    # spatial_ccc_inference_lowres / _highres.


if __name__ == "__main__":
    main()
