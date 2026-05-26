"""Three-tier genre vocabulary for deterministic source-tag classification."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenreLookupResult:
    genre: str
    confidence: float
    tier: int
    reason: str


_NON_GENRE_CATEGORIES = ("descriptor", "instrument", "place", "format", "mood_function", "label_or_org")
_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "data" / "genre_vocabulary.yaml"


class GenreVocabulary:
    """Three-tier genre vocabulary: curated YAML → engine synonyms → library DB."""

    def __init__(
        self,
        yaml_path: str | Path = _DEFAULT_YAML_PATH,
        *,
        library_db_path: str | Path | None = None,
    ) -> None:
        self._yaml_path = Path(yaml_path)
        self._raw: dict[str, Any] = {}
        self._tier1_genres: set[str] = set()
        self._non_genre_sets: dict[str, set[str]] = {}
        self._aliases: dict[str, str] = {}
        self._tier2_genres: set[str] = set()
        self._tier3_genres: set[str] = set()

        self._load_yaml()
        self._bootstrap_engine_genres()
        if library_db_path:
            self._load_library_genres(library_db_path)

    def _load_yaml(self) -> None:
        if not self._yaml_path.exists():
            return
        with self._yaml_path.open("r", encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}
        self._tier1_genres = set(self._raw.get("genre_style", []))
        for category in _NON_GENRE_CATEGORIES:
            self._non_genre_sets[category] = set(self._raw.get(category, []))
        self._aliases = dict(self._raw.get("aliases", {}))

    def _bootstrap_engine_genres(self) -> None:
        from src.genre.normalize_unified import SYNONYM_MAP, PHRASE_MAP

        engine_genres: set[str] = set()
        for target in SYNONYM_MAP.values():
            if target:
                engine_genres.add(target)
        for outputs in PHRASE_MAP.values():
            for token in outputs:
                if token:
                    engine_genres.add(token)
        self._tier2_genres = engine_genres - self._tier1_genres - self._all_non_genre_terms()

    def _load_library_genres(self, db_path: str | Path) -> None:
        resolved = Path(db_path).resolve()
        if not resolved.exists():
            return
        uri = f"file:{resolved.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            raw_genres: set[str] = set()
            for table in ("artist_genres", "album_genres", "track_genres"):
                try:
                    for row in conn.execute(f"SELECT DISTINCT genre FROM {table}"):
                        if row["genre"]:
                            raw_genres.add(row["genre"].strip().casefold())
                except sqlite3.OperationalError:
                    continue
            known = self._tier1_genres | self._tier2_genres | self._all_non_genre_terms()
            self._tier3_genres = raw_genres - known
        finally:
            conn.close()

    def _all_non_genre_terms(self) -> set[str]:
        result: set[str] = set()
        for terms in self._non_genre_sets.values():
            result |= terms
        return result

    def resolve_alias(self, tag: str) -> str:
        return self._aliases.get(tag, tag)

    def classify_genre(self, normalized_tag: str) -> GenreLookupResult | None:
        tag = self.resolve_alias(normalized_tag)
        if tag in self._tier1_genres:
            return GenreLookupResult(tag, 0.95, 1, "Curated genre vocabulary.")
        if tag in self._tier2_genres:
            return GenreLookupResult(tag, 0.85, 2, "Engine-recognized genre token.")
        if tag in self._tier3_genres:
            return GenreLookupResult(tag, 0.80, 3, "Genre found in library metadata.")
        return None

    def classify_non_genre(self, normalized_tag: str) -> str | None:
        tag = self.resolve_alias(normalized_tag)
        for category, terms in self._non_genre_sets.items():
            if tag in terms:
                return category
        return None

    def add_term(self, category: str, term: str) -> None:
        if category == "genre_style":
            self._tier1_genres.add(term)
        elif category in self._non_genre_sets:
            self._non_genre_sets[category].add(term)
        else:
            raise ValueError(f"Unknown category: {category}")

    def save(self) -> None:
        data: dict[str, Any] = {"version": self._raw.get("version", 1)}
        data["genre_style"] = sorted(self._tier1_genres)
        for category in _NON_GENRE_CATEGORIES:
            data[category] = sorted(self._non_genre_sets.get(category, set()))
        if self._aliases:
            data["aliases"] = dict(sorted(self._aliases.items()))
        with self._yaml_path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
