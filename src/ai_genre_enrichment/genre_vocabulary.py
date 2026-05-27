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


def _collect_engine_genres() -> frozenset[str]:
    from src.genre.normalize_unified import SYNONYM_MAP, PHRASE_MAP

    genres: set[str] = set()
    for target in SYNONYM_MAP.values():
        if target:
            genres.add(target)
    for outputs in PHRASE_MAP.values():
        for token in outputs:
            if token:
                genres.add(token)
    return frozenset(genres)


_ENGINE_GENRES: frozenset[str] = _collect_engine_genres()


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
        self._decompose: dict[str, list[str]] = {}
        self._tier2_genres: set[str] = set()
        self._tier3_genres: set[str] = set()

        self._load_yaml()
        self._bootstrap_engine_genres()
        if library_db_path:
            self._load_library_genres(library_db_path)

    def _load_yaml(self) -> None:
        if not self._yaml_path.exists():
            raise FileNotFoundError(f"Genre vocabulary YAML not found: {self._yaml_path}")
        with self._yaml_path.open("r", encoding="utf-8") as fh:
            self._raw = yaml.safe_load(fh) or {}
        self._tier1_genres = set(self._raw.get("genre_style", []))
        for category in _NON_GENRE_CATEGORIES:
            self._non_genre_sets[category] = set(self._raw.get(category, []))
        self._aliases = dict(self._raw.get("aliases", {}))
        self._decompose = {k: list(v) for k, v in self._raw.get("decompose", {}).items()}

    def _bootstrap_engine_genres(self) -> None:
        self._tier2_genres = _ENGINE_GENRES - self._tier1_genres - self._all_non_genre_terms()

    def _load_library_genres(self, db_path: str | Path) -> None:
        # normalize_and_split_genre handles both single-value rows and any multi-value
        # strings (e.g. "dark ambient; drone") that may appear in the DB.
        from src.genre.normalize_unified import normalize_and_split_genre

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
                            for token in normalize_and_split_genre(row["genre"]):
                                if token:
                                    raw_genres.add(token)
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

    def decompose_tag(self, tag: str) -> list[str] | None:
        """Return decomposed genre list if a decompose rule exists, else None."""
        return self._decompose.get(tag) or None

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
        # Tier-2 (engine) and tier-3 (library DB) genres are derived at load time and not saved —
        # they're re-bootstrapped from their sources on the next instantiation.
        lines: list[str] = [f"version: {self._raw.get('version', 1)}"]

        section_order = ["genre_style"] + list(_NON_GENRE_CATEGORIES)
        section_data: dict[str, list[str]] = {"genre_style": sorted(self._tier1_genres)}
        for cat in _NON_GENRE_CATEGORIES:
            section_data[cat] = sorted(self._non_genre_sets.get(cat, set()))

        for section in section_order:
            if section_data[section]:
                lines.append(f"{section}:")
                for item in section_data[section]:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{section}: []")

        if self._aliases:
            lines.append("aliases:")
            for key, value in sorted(self._aliases.items()):
                lines.append(f"  {key}: {value}")

        if self._decompose:
            lines.append("decompose:")
            for key, values in sorted(self._decompose.items()):
                lines.append(f"  {key}:")
                for v in values:
                    lines.append(f"    - {v}")

        with self._yaml_path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
