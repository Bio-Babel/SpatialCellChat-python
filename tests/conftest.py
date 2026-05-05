"""Shared pytest fixtures.

Provides:
- ``tiny_visium`` — a small AnnData fixture (15 spots × 50 genes) with all the
  slots cellchat downstream code expects: ``obsm["spatial"]``, an ``obs``
  cluster-id column, and ``uns["spatial"]``. Built by
  ``tests/_make_fixtures.py`` if not already present.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

from . import _make_fixtures

_FIXTURES_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def tiny_visium() -> ad.AnnData:
    fp = _FIXTURES_DIR / "tiny_visium.h5ad"
    if not fp.exists():
        _make_fixtures.build_tiny_visium(fp)
    return ad.read_h5ad(fp)
