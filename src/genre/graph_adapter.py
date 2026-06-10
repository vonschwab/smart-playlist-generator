"""Read-only adapter over the SP3a layered genre taxonomy for playlist genre systems.

Stage 1 of the graph-taxonomy integration (see docs/LAYERED_GENRE_GRAPH_SPEC.md).
This module is a facade over src.ai_genre_enrichment.layered_taxonomy: it does
not parse or validate YAML itself (the loader already does both), it exposes
the playlist-facing semantics the similarity-matrix generator (Stage 2) and
artifact builder (Stage 3) will consume:

- canonicalize_tag: raw source tag -> canonical / alias / facet / rejected / unknown
- active_genre_vocabulary: the canonical genre dimensions (facets, aliases,
  rejects, and review-status nodes excluded by default)
- broad-node marking (family/umbrella) so matrix generation can damp hubs
- edges keyed by canonical names, not internal ids

Everything here is read-only. No playlist-generation behavior changes until a
later stage wires the outputs in behind config flags.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from src.ai_genre_enrichment.layered_taxonomy import (
    GenreAlias,
    LayeredTaxonomy,
    load_default_layered_taxonomy,
    load_layered_taxonomy,
    normalize_taxonomy_name,
)

BROAD_KINDS = frozenset({"family", "umbrella"})
ACTIVE_STATUSES = frozenset({"active"})
VOCABULARY_STATUSES = frozenset({"active", "review"})


@dataclass(frozen=True)
class GraphNode:
    """Canonical genre node with the metadata downstream stages need."""

    name: str
    kind: str
    status: str
    role: str
    specificity_score: float
    is_broad: bool


@dataclass(frozen=True)
class GraphEdge:
    """Taxonomy edge expressed in canonical names (not internal ids)."""

    source: str
    target: str
    edge_type: str
    weight: float
    confidence: float


@dataclass(frozen=True)
class CanonicalizationResult:
    """Verdict for one raw source tag.

    resolution is one of:
      "canonical" — the tag is itself a canonical genre name
      "alias"     — resolved through an alias to a canonical genre
      "facet"     — descriptor/facet term (or alias to one); not a genre dimension
      "rejected"  — curated noise; must never become a genre dimension
      "unknown"   — not in the taxonomy; caller decides passthrough policy
    canonical is set only for "canonical"/"alias" and always holds the
    canonical genre name — never the alias.
    """

    raw: str
    resolution: str
    canonical: Optional[str] = None
    node: Optional[GraphNode] = None
    facet_name: Optional[str] = None
    facet_type: Optional[str] = None
    reject_reason: Optional[str] = None


def _is_plain_alias(alias: GenreAlias) -> bool:
    policy = alias.alias_policy or {}
    alias_type = str(policy.get("type") or policy.get("alias_type") or "").casefold()
    if alias_type == "conditional":
        return False
    return not policy.get("requires_any_context") and not policy.get("requires_all_context")


class GenreGraphAdapter:
    """Read-only view of a LayeredTaxonomy for genre-similarity integration."""

    def __init__(self, taxonomy: LayeredTaxonomy) -> None:
        self._taxonomy = taxonomy
        self._nodes_by_name: dict[str, GraphNode] = {}
        self._names_by_id: dict[str, str] = {}
        for genre in taxonomy.genres:
            node = GraphNode(
                name=genre.name,
                kind=genre.kind,
                status=genre.status,
                role=genre.role,
                specificity_score=float(genre.specificity_score),
                is_broad=genre.kind in BROAD_KINDS,
            )
            self._nodes_by_name[normalize_taxonomy_name(genre.name)] = node
            self._names_by_id[genre.genre_id] = genre.name

    @property
    def taxonomy_version(self) -> str:
        return self._taxonomy.version

    @property
    def taxonomy(self) -> LayeredTaxonomy:
        return self._taxonomy

    def canonicalize_tag(
        self,
        raw_tag: str,
        context_terms: Sequence[str] | None = None,
    ) -> CanonicalizationResult:
        normalized = normalize_taxonomy_name(raw_tag)

        rejected = self._taxonomy.rejected_term_by_name(normalized)
        if rejected is not None:
            return CanonicalizationResult(
                raw=raw_tag, resolution="rejected", reject_reason=rejected.reason
            )

        node = self._nodes_by_name.get(normalized)
        if node is not None:
            return CanonicalizationResult(
                raw=raw_tag, resolution="canonical", canonical=node.name, node=node
            )

        alias = self._taxonomy.alias_for_name(
            raw_tag, context_terms=list(context_terms) if context_terms else None
        )
        if alias is not None:
            if alias.target_kind == "facet":
                facet = self._taxonomy.facet_by_id(alias.canonical_genre_id)
                if facet is not None:
                    return CanonicalizationResult(
                        raw=raw_tag,
                        resolution="facet",
                        facet_name=facet.name,
                        facet_type=facet.facet_type,
                    )
            else:
                target_name = self._names_by_id.get(alias.canonical_genre_id)
                target = (
                    self._nodes_by_name.get(normalize_taxonomy_name(target_name))
                    if target_name
                    else None
                )
                if target is not None:
                    return CanonicalizationResult(
                        raw=raw_tag, resolution="alias", canonical=target.name, node=target
                    )

        facet = self._taxonomy.facet_by_name(normalized)
        if facet is not None:
            return CanonicalizationResult(
                raw=raw_tag,
                resolution="facet",
                facet_name=facet.name,
                facet_type=facet.facet_type,
            )

        return CanonicalizationResult(raw=raw_tag, resolution="unknown")

    def node(self, canonical_name: str) -> Optional[GraphNode]:
        return self._nodes_by_name.get(normalize_taxonomy_name(canonical_name))

    def is_active_genre(self, name: str) -> bool:
        node = self.node(name)
        return node is not None and node.status in ACTIVE_STATUSES

    def active_genre_vocabulary(self, *, include_review: bool = False) -> list[str]:
        allowed = VOCABULARY_STATUSES if include_review else ACTIVE_STATUSES
        return sorted(
            node.name for node in self._nodes_by_name.values() if node.status in allowed
        )

    def alias_map(self) -> dict[str, str]:
        """Plain (non-conditional) genre-target aliases: normalized alias -> canonical name.

        Conditional aliases need context and resolve only through
        canonicalize_tag; facet-target aliases are not genre mappings. An alias
        whose normalized name collides with a canonical genre is excluded —
        the canonical record wins.
        """
        mapping: dict[str, str] = {}
        for alias in self._taxonomy.aliases:
            if alias.target_kind != "genre" or not _is_plain_alias(alias):
                continue
            key = normalize_taxonomy_name(alias.alias)
            if key in self._nodes_by_name:
                continue
            target_name = self._names_by_id.get(alias.canonical_genre_id)
            if target_name:
                mapping[key] = target_name
        return mapping

    def rejected_terms(self) -> set[str]:
        return {normalize_taxonomy_name(term.term) for term in self._taxonomy.rejected_terms}

    def edges(self) -> tuple[GraphEdge, ...]:
        result: list[GraphEdge] = []
        for edge in self._taxonomy.edges:
            source = self._names_by_id.get(edge.source_genre_id)
            target = self._names_by_id.get(edge.target_genre_id)
            if source is None or target is None:
                continue
            result.append(
                GraphEdge(
                    source=source,
                    target=target,
                    edge_type=edge.edge_type,
                    weight=float(edge.weight),
                    confidence=float(edge.confidence),
                )
            )
        return tuple(result)


def load_graph_adapter(path: str | Path | None = None) -> GenreGraphAdapter:
    """Load the SP3a taxonomy (default: data/layered_genre_taxonomy.yaml) read-only."""
    if path is None:
        return GenreGraphAdapter(_cached_default_taxonomy())
    return GenreGraphAdapter(load_layered_taxonomy(path))


@lru_cache(maxsize=1)
def _cached_default_taxonomy() -> LayeredTaxonomy:
    return load_default_layered_taxonomy()
