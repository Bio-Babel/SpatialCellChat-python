"""Remote data asset registry.

Production Zenodo record: https://zenodo.org/records/20030302
(DOI: 10.5281/zenodo.20030302)
"""

CACHE_DIR_NAME = "SpatialCellChat-python"

REGISTRY: dict[str, dict[str, str]] = {
    "visium_human_psoriasis.h5ad": {
        "url": "https://zenodo.org/records/20030302/files/visium_human_psoriasis.h5ad?download=1",
        "sha256": "bdb748971f02b37191b5999063fead2b063c35964bca4ae1a5a490631523c178",
    },
}
