from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "layered_genre_taxonomy.yaml"
FAMILY_KIND = "family"


def normalize_taxonomy_name(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())


@dataclass(frozen=True)
class CanonicalGenre:
    genre_id: str
    name: str
    kind: str
    specificity_score: float
    status: str
    taxonomy_version: str


@dataclass(frozen=True)
class GenreAlias:
    alias: str
    canonical_genre_id: str
    source: str
    confidence: float


@dataclass(frozen=True)
class GenreEdge:
    source_genre_id: str
    target_genre_id: str
    edge_type: str
    weight: float
    confidence: float
    source: str
    notes: str | None = None


@dataclass(frozen=True)
class CanonicalFacet:
    facet_id: str
    name: str
    facet_type: str
    status: str


@dataclass(frozen=True)
class BridgeRule:
    source_genre_id: str
    target_genre_id: str
    required_family_min: float
    required_facet_overlap: float
    required_sonic_similarity: float
    required_transition_quality: float
    mode_allowed: tuple[str, ...]
    notes: str | None = None


@dataclass(frozen=True)
class RejectedTerm:
    term: str
    reason: str


@dataclass(frozen=True)
class LayeredTaxonomy:
    version: str
    genres: tuple[CanonicalGenre, ...]
    aliases: tuple[GenreAlias, ...]
    edges: tuple[GenreEdge, ...]
    facets: tuple[CanonicalFacet, ...]
    bridge_rules: tuple[BridgeRule, ...]
    rejected_terms: tuple[RejectedTerm, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_genres_by_id", {genre.genre_id: genre for genre in self.genres})
        object.__setattr__(self, "_genres_by_name", {
            normalize_taxonomy_name(genre.name): genre
            for genre in self.genres
        })
        object.__setattr__(self, "_alias_targets_by_name", {
            normalize_taxonomy_name(alias.alias): alias.canonical_genre_id
            for alias in self.aliases
        })
        object.__setattr__(self, "_alias_targets_by_exact_name", {
            alias.alias.casefold().strip(): alias.canonical_genre_id
            for alias in self.aliases
        })
        object.__setattr__(self, "_facets_by_name", {
            normalize_taxonomy_name(facet.name): facet
            for facet in self.facets
        })
        object.__setattr__(self, "_rejected_terms_by_name", {
            normalize_taxonomy_name(term.term): term
            for term in self.rejected_terms
        })
        object.__setattr__(self, "_is_a_edges_by_source", _group_is_a_edges(self.edges))
        object.__setattr__(self, "_bridge_rules_by_pair", {
            (rule.source_genre_id, rule.target_genre_id): rule
            for rule in self.bridge_rules
        })

    def genre_by_id(self, genre_id: str) -> CanonicalGenre | None:
        return self._genres_by_id.get(genre_id)

    def genre_by_name(self, name: str) -> CanonicalGenre | None:
        normalized = normalize_taxonomy_name(name)
        genre = self._genres_by_name.get(normalized)
        if genre is not None:
            return genre
        alias_target = self._alias_targets_by_name.get(normalized)
        if alias_target is None:
            return None
        return self.genre_by_id(alias_target)

    def facet_by_name(self, name: str) -> CanonicalFacet | None:
        return self._facets_by_name.get(normalize_taxonomy_name(name))

    def alias_target_for_name(self, name: str) -> CanonicalGenre | None:
        alias_target = self._alias_targets_by_name.get(normalize_taxonomy_name(name))
        if alias_target is None:
            return None
        return self.genre_by_id(alias_target)

    def exact_alias_target_for_name(self, name: str) -> CanonicalGenre | None:
        alias_target = self._alias_targets_by_exact_name.get(str(name or "").casefold().strip())
        if alias_target is None:
            return None
        return self.genre_by_id(alias_target)

    def rejected_term_by_name(self, name: str) -> RejectedTerm | None:
        return self._rejected_terms_by_name.get(normalize_taxonomy_name(name))

    def parents_for_genre(self, genre_id: str) -> tuple[CanonicalGenre, ...]:
        return tuple(
            genre
            for genre in (
                self.genre_by_id(edge.target_genre_id)
                for edge in self._is_a_edges_by_source.get(genre_id, ())
            )
            if genre is not None and genre.kind != FAMILY_KIND
        )

    def families_for_genre(self, genre_id: str) -> tuple[CanonicalGenre, ...]:
        families: dict[str, CanonicalGenre] = {}
        visited: set[str] = set()
        stack = [genre_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for edge in self._is_a_edges_by_source.get(current, ()):
                parent = self.genre_by_id(edge.target_genre_id)
                if parent is None:
                    continue
                if parent.kind == FAMILY_KIND:
                    families[parent.genre_id] = parent
                else:
                    stack.append(parent.genre_id)
        return tuple(sorted(families.values(), key=lambda genre: genre.name))

    def bridge_rule_for(self, source_genre_id: str, target_genre_id: str) -> BridgeRule | None:
        direct = self._bridge_rules_by_pair.get((source_genre_id, target_genre_id))
        if direct is not None:
            return direct
        reverse = self._bridge_rules_by_pair.get((target_genre_id, source_genre_id))
        if reverse is None:
            return None
        return BridgeRule(
            source_genre_id=source_genre_id,
            target_genre_id=target_genre_id,
            required_family_min=reverse.required_family_min,
            required_facet_overlap=reverse.required_facet_overlap,
            required_sonic_similarity=reverse.required_sonic_similarity,
            required_transition_quality=reverse.required_transition_quality,
            mode_allowed=reverse.mode_allowed,
            notes=reverse.notes,
        )


def load_default_layered_taxonomy() -> LayeredTaxonomy:
    return load_layered_taxonomy(DEFAULT_TAXONOMY_PATH)


def load_layered_taxonomy(path: str | Path) -> LayeredTaxonomy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError("Layered taxonomy is missing version")

    genres = tuple(_genre_from_row(row, version) for row in data.get("genres", []))
    facets = tuple(_facet_from_row(row) for row in data.get("facets", []))
    aliases = tuple(_alias_from_row(row) for row in data.get("aliases", []))
    edges = tuple(_edge_from_row(row) for row in data.get("edges", []))
    bridge_rules = tuple(_bridge_rule_from_row(row) for row in data.get("bridge_rules", []))
    rejected_terms = tuple(_rejected_term_from_row(row) for row in data.get("rejected_terms", []))

    taxonomy = LayeredTaxonomy(
        version=version,
        genres=genres,
        aliases=aliases,
        edges=edges,
        facets=facets,
        bridge_rules=bridge_rules,
        rejected_terms=rejected_terms,
    )
    _validate_taxonomy(taxonomy)
    return taxonomy


def _genre_from_row(row: dict[str, Any], version: str) -> CanonicalGenre:
    return CanonicalGenre(
        genre_id=str(row["genre_id"]),
        name=str(row["name"]),
        kind=str(row["kind"]),
        specificity_score=float(row["specificity_score"]),
        status=str(row["status"]),
        taxonomy_version=version,
    )


def _alias_from_row(row: dict[str, Any]) -> GenreAlias:
    return GenreAlias(
        alias=str(row["alias"]),
        canonical_genre_id=str(row["canonical_genre_id"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
    )


def _edge_from_row(row: dict[str, Any]) -> GenreEdge:
    return GenreEdge(
        source_genre_id=str(row["source_genre_id"]),
        target_genre_id=str(row["target_genre_id"]),
        edge_type=str(row["edge_type"]),
        weight=float(row["weight"]),
        confidence=float(row["confidence"]),
        source=str(row["source"]),
        notes=row.get("notes"),
    )


def _facet_from_row(row: dict[str, Any]) -> CanonicalFacet:
    return CanonicalFacet(
        facet_id=str(row["facet_id"]),
        name=str(row["name"]),
        facet_type=str(row["facet_type"]),
        status=str(row["status"]),
    )


def _bridge_rule_from_row(row: dict[str, Any]) -> BridgeRule:
    mode_allowed = row["mode_allowed"]
    if isinstance(mode_allowed, str):
        modes = tuple(part.strip() for part in mode_allowed.split(",") if part.strip())
    else:
        modes = tuple(str(part) for part in mode_allowed)
    return BridgeRule(
        source_genre_id=str(row["source_genre_id"]),
        target_genre_id=str(row["target_genre_id"]),
        required_family_min=float(row["required_family_min"]),
        required_facet_overlap=float(row["required_facet_overlap"]),
        required_sonic_similarity=float(row["required_sonic_similarity"]),
        required_transition_quality=float(row["required_transition_quality"]),
        mode_allowed=modes,
        notes=row.get("notes"),
    )


def _rejected_term_from_row(row: dict[str, Any]) -> RejectedTerm:
    return RejectedTerm(
        term=str(row["term"]),
        reason=str(row["reason"]),
    )


def _group_is_a_edges(edges: tuple[GenreEdge, ...]) -> dict[str, tuple[GenreEdge, ...]]:
    grouped: dict[str, list[GenreEdge]] = {}
    for edge in edges:
        if edge.edge_type == "is_a":
            grouped.setdefault(edge.source_genre_id, []).append(edge)
    return {key: tuple(value) for key, value in grouped.items()}


def _validate_taxonomy(taxonomy: LayeredTaxonomy) -> None:
    genre_ids = {genre.genre_id for genre in taxonomy.genres}
    facet_ids = {facet.facet_id for facet in taxonomy.facets}
    if len(genre_ids) != len(taxonomy.genres):
        raise ValueError("Layered taxonomy has duplicate genre ids")
    if len(facet_ids) != len(taxonomy.facets):
        raise ValueError("Layered taxonomy has duplicate facet ids")

    for alias in taxonomy.aliases:
        if alias.canonical_genre_id not in genre_ids:
            raise ValueError(f"Alias points to unknown genre: {alias.alias}")
    for edge in taxonomy.edges:
        if edge.source_genre_id not in genre_ids or edge.target_genre_id not in genre_ids:
            raise ValueError(f"Edge points to unknown genre: {edge.source_genre_id}->{edge.target_genre_id}")
    for rule in taxonomy.bridge_rules:
        if rule.source_genre_id not in genre_ids or rule.target_genre_id not in genre_ids:
            raise ValueError(f"Bridge rule points to unknown genre: {rule.source_genre_id}->{rule.target_genre_id}")
