"""Smoke test for spatialcellchat._biobabel.

Exercises a real, self-contained path through the package's public API: load
the bundled CellChatDB (a package resource — no network access) and subset it
to the interaction categories used for spatial CCC. This does not touch
AnnData or any downloaded dataset, keeping the check fast and dependency-light
while still calling real, non-trivial package code.
"""

from __future__ import annotations


def main() -> None:
    import spatialcellchat as scc

    db = scc.load_db("human")
    assert "interaction" in db and len(db["interaction"]) > 0

    db_use = scc.subsetDB(
        db, search=["Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"]
    )
    assert len(db_use["interaction"]) > 0
    assert len(db_use["interaction"]) <= len(db["interaction"])

    print(
        f"loaded CellChatDB(human): {len(db['interaction'])} interactions; "
        f"subset to {len(db_use['interaction'])} after category filtering"
    )


if __name__ == "__main__":
    main()
