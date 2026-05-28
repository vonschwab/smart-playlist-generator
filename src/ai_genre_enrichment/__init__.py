"""AI-assisted genre enrichment backend.

This package is recommendation-only. It reads the library metadata database,
writes enrichment checks to a sidecar database, and never mutates existing
genre tables.
"""

from .discovery import ReleasePayload, compute_input_hash, discover_releases
from .normalization import make_release_key

__all__ = [
    "ReleasePayload",
    "compute_input_hash",
    "discover_releases",
    "make_release_key",
]
