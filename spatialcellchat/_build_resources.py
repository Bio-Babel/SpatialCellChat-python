"""One-shot resource builder.

Converts the R-side dump under
``port_reports/SpatialCellChat/data_exports/db/`` into the bundled package
resources directory (``spatialcellchat/resources/db/``):

- TSV.gz → parquet (one per data frame)
- MatrixMarket → scipy CSR ``.npz`` (sparse PPI matrices)
- Row labels file is copied verbatim.

Idempotent: re-running overwrites existing files.

Run via:

    python -m spatialcellchat._build_resources \
        --src port_reports/SpatialCellChat/data_exports/db \
        [--dst spatialcellchat/resources/db]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

__all__ = ["build", "main"]

_HERE = Path(__file__).resolve().parent
_DEFAULT_DST = _HERE / "resources" / "db"

_DF_NAMES = ("interaction", "complex", "cofactor", "geneInfo")
_SPECIES_WITH_PPI = ("human", "mouse")


def _convert_tsv_to_parquet(src: Path, dst: Path) -> None:
    df = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def _convert_mtx_to_npz(src: Path, dst: Path) -> None:
    mat = sio.mmread(src).tocsr()
    dst.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(dst, mat)


def build(src: Path, dst: Path = _DEFAULT_DST) -> None:
    """Build bundled resources from the R-side dump.

    Parameters
    ----------
    src : Path
        Source directory holding ``<species>/{interaction,complex,cofactor,geneInfo}.tsv.gz``
        and ``<species>/PPI.{mtx,rows.txt}``.
    dst : Path, optional
        Destination directory; defaults to ``spatialcellchat/resources/db``.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"source dir not found: {src}")
    species_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    for species_dir in species_dirs:
        species = species_dir.name
        out = dst / species
        out.mkdir(parents=True, exist_ok=True)
        for name in _DF_NAMES:
            tsv = species_dir / f"{name}.tsv.gz"
            if not tsv.exists():
                continue
            _convert_tsv_to_parquet(tsv, out / f"{name}.parquet")
        if species in _SPECIES_WITH_PPI:
            mtx = species_dir / "PPI.mtx"
            rows = species_dir / "PPI.rows.txt"
            if mtx.exists():
                _convert_mtx_to_npz(mtx, out / "PPI.npz")
            if rows.exists():
                (out / "PPI.rows.txt").write_text(rows.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spatialcellchat._build_resources")
    parser.add_argument(
        "--src",
        required=True,
        help="Source dir from R exporter (e.g. port_reports/SpatialCellChat/data_exports/db).",
    )
    parser.add_argument(
        "--dst",
        default=str(_DEFAULT_DST),
        help="Destination dir under the package; default = spatialcellchat/resources/db.",
    )
    args = parser.parse_args(argv)
    build(Path(args.src), Path(args.dst))
    print(f"built resources from {args.src} → {args.dst}")


if __name__ == "__main__":
    main()
