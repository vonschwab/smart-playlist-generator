"""Granularity ordering of genre tags via the SP3a layered taxonomy.

Display-path helper (spec: docs/superpowers/specs/2026-06-11-genre-chips-granularity-design.md):
canonicalize raw genre tags through the taxonomy graph and order the canonical
names most-specific first (sub-genre -> broad). Used by the GUI worker
(playlist table) and the web staged-seed endpoint. Read-only; never raises
into a generation or request path.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.genre.graph_adapter import GenreGraphAdapter, load_graph_adapter

logger = logging.getLogger(__name__)

# Log the degraded-taxonomy path once per process, not once per track.
_ADAPTER_WARNED = False


def order_genres_by_granularity(
    raw_tags: Sequence[str],
    adapter: Optional[GenreGraphAdapter] = None,
) -> list[str]:
    """Canonical node names for raw tags, most-specific first.

    - canonicalizes each tag (aliases resolve to their canonical genre)
    - drops tags that don't resolve to an ACTIVE canonical node
      (facets, rejects, unknown, review/deprecated nodes)
    - de-dups by canonical name (first occurrence wins)
    - stable-sorts by specificity_score descending; ties keep input order
    - returns [] when nothing canonicalizes
    - never raises: on taxonomy load failure, returns the raw tags unchanged
      (degraded but functional; logged once per process)
    """
    global _ADAPTER_WARNED
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    if not tags:
        return []
    try:
        if adapter is None:
            adapter = load_graph_adapter()
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for tag in tags:
            result = adapter.canonicalize_tag(tag)
            node = result.node
            if result.resolution not in ("canonical", "alias") or node is None:
                continue
            if node.status != "active":
                continue
            if node.name in seen:
                continue
            seen.add(node.name)
            ranked.append((float(node.specificity_score), node.name))
        # list.sort is stable: equal scores keep input order.
        ranked.sort(key=lambda pair: -pair[0])
        return [name for _, name in ranked]
    except Exception:
        if not _ADAPTER_WARNED:
            logger.warning(
                "Taxonomy unavailable for genre display ordering; showing raw tags",
                exc_info=True,
            )
            _ADAPTER_WARNED = True
        return tags


def order_genres_for_display(
    raw_tags: Sequence[str],
    adapter: Optional[GenreGraphAdapter] = None,
) -> list[str]:
    """order_genres_by_granularity with the display safety fallback.

    When raw tags exist but none canonicalize, return the raw tags unordered —
    a track never regresses to blank chips because the taxonomy didn't cover it.
    """
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    ordered = order_genres_by_granularity(tags, adapter=adapter)
    if not ordered and tags:
        return tags
    return ordered
