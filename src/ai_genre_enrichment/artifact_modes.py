"""Explicit artifact genre-source modes."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


class GenreArtifactSource(str, Enum):
    LEGACY = "legacy"
    ENRICHED = "enriched"
    HYBRID_SHADOW = "hybrid_shadow"

    @classmethod
    def resolve(cls, value: str | None) -> "GenreArtifactSource":
        return cls(value or cls.LEGACY.value)


def make_resolver(mode: GenreArtifactSource, sidecar_db: str | Path) -> Any | None:
    if mode is GenreArtifactSource.LEGACY:
        return None
    from .genre_resolver import EnrichedGenreResolver
    return EnrichedGenreResolver(sidecar_db)
