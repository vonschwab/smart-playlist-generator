from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .layered_taxonomy import LayeredTaxonomy


@dataclass(frozen=True)
class LayeredGenreMatrices:
    X_genre_leaf_idf: np.ndarray
    X_genre_family: np.ndarray
    X_genre_bridge: np.ndarray
    X_facet: np.ndarray
    genre_leaf_vocab: tuple[str, ...]
    genre_family_vocab: tuple[str, ...]
    genre_bridge_vocab: tuple[str, ...]
    facet_vocab: tuple[str, ...]
    taxonomy_version: str
    graph_fingerprint: str


def build_layered_genre_matrices(
    store: Any,
    *,
    track_release_keys: list[str],
    taxonomy: LayeredTaxonomy,
) -> LayeredGenreMatrices:
    release_keys = sorted(set(track_release_keys))
    genre_rows, facet_rows = _load_assignments(store, release_keys)

    leaf_ids = _vocab_ids(
        row["genre_id"]
        for row in genre_rows
        if row["assignment_layer"] == "observed_leaf"
    )
    family_ids = _vocab_ids(
        row["genre_id"]
        for row in genre_rows
        if row["assignment_layer"] == "inferred_family"
    )
    # Bridge affordance is scored against leaf/style vectors:
    #   seed_bridge ⟂ candidate_leaf, candidate_bridge ⟂ seed_leaf.
    # Keep bridge coordinates aligned to the leaf vocabulary so graph edges can
    # authorize movement toward observed candidate styles.
    bridge_ids = leaf_ids
    facet_ids = _vocab_ids(row["facet_id"] for row in facet_rows)

    leaf_index = {genre_id: idx for idx, genre_id in enumerate(leaf_ids)}
    family_index = {genre_id: idx for idx, genre_id in enumerate(family_ids)}
    bridge_index = {genre_id: idx for idx, genre_id in enumerate(bridge_ids)}
    facet_index = {facet_id: idx for idx, facet_id in enumerate(facet_ids)}

    n_tracks = len(track_release_keys)
    X_leaf = np.zeros((n_tracks, len(leaf_ids)), dtype=np.float32)
    X_family = np.zeros((n_tracks, len(family_ids)), dtype=np.float32)
    X_bridge = np.zeros((n_tracks, len(bridge_ids)), dtype=np.float32)
    X_facet = np.zeros((n_tracks, len(facet_ids)), dtype=np.float32)

    rows_by_release: dict[str, list[dict[str, Any]]] = {}
    for row in genre_rows:
        rows_by_release.setdefault(row["release_id"], []).append(row)
    facets_by_release: dict[str, list[dict[str, Any]]] = {}
    for row in facet_rows:
        facets_by_release.setdefault(row["release_id"], []).append(row)

    idf = _leaf_idf(leaf_ids, rows_by_release, total_releases=max(1, len(release_keys)))
    for track_idx, release_key in enumerate(track_release_keys):
        for row in rows_by_release.get(release_key, []):
            genre_id = row["genre_id"]
            confidence = float(row["confidence"])
            if row["assignment_layer"] == "observed_leaf" and genre_id in leaf_index:
                X_leaf[track_idx, leaf_index[genre_id]] = max(
                    X_leaf[track_idx, leaf_index[genre_id]],
                    np.float32(confidence * idf.get(genre_id, 1.0)),
                )
            elif row["assignment_layer"] == "inferred_family" and genre_id in family_index:
                X_family[track_idx, family_index[genre_id]] = max(
                    X_family[track_idx, family_index[genre_id]],
                    np.float32(confidence),
                )

        for bridge_id in _bridge_affordance_ids(rows_by_release.get(release_key, []), taxonomy):
            if bridge_id in bridge_index:
                X_bridge[track_idx, bridge_index[bridge_id]] = np.float32(1.0)

        for row in facets_by_release.get(release_key, []):
            facet_id = row["facet_id"]
            if facet_id in facet_index:
                X_facet[track_idx, facet_index[facet_id]] = max(
                    X_facet[track_idx, facet_index[facet_id]],
                    np.float32(float(row["confidence"])),
                )

    return LayeredGenreMatrices(
        X_genre_leaf_idf=X_leaf,
        X_genre_family=X_family,
        X_genre_bridge=X_bridge,
        X_facet=X_facet,
        genre_leaf_vocab=_genre_names(leaf_ids, taxonomy),
        genre_family_vocab=_genre_names(family_ids, taxonomy),
        genre_bridge_vocab=_genre_names(bridge_ids, taxonomy),
        facet_vocab=_facet_names(facet_ids, taxonomy),
        taxonomy_version=taxonomy.version,
        graph_fingerprint=_graph_fingerprint(
            taxonomy=taxonomy,
            genre_rows=genre_rows,
            facet_rows=facet_rows,
        ),
    )


def _load_assignments(
    store: Any,
    release_keys: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not release_keys:
        return [], []
    placeholders = ",".join("?" for _ in release_keys)
    with store.connect() as conn:
        genre_rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT release_id, genre_id, assignment_layer, confidence
                FROM genre_graph_release_genre_assignments
                WHERE release_id IN ({placeholders})
                  AND rejected_by_user = 0
                """,
                release_keys,
            )
        ]
        facet_rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT release_id, facet_id, confidence
                FROM genre_graph_release_facet_assignments
                WHERE release_id IN ({placeholders})
                """,
                release_keys,
            )
        ]
    return genre_rows, facet_rows


def _vocab_ids(values: Any) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if value}))


def _leaf_idf(
    leaf_ids: tuple[str, ...],
    rows_by_release: dict[str, list[dict[str, Any]]],
    *,
    total_releases: int,
) -> dict[str, float]:
    idf: dict[str, float] = {}
    for leaf_id in leaf_ids:
        df = sum(
            1
            for rows in rows_by_release.values()
            if any(row["genre_id"] == leaf_id and row["assignment_layer"] == "observed_leaf" for row in rows)
        )
        idf[leaf_id] = float(np.log((1.0 + total_releases) / (1.0 + df)) + 1.0)
    return idf


def _bridge_affordance_ids(rows: list[dict[str, Any]], taxonomy: LayeredTaxonomy) -> tuple[str, ...]:
    observed = {
        row["genre_id"]
        for row in rows
        if row.get("assignment_layer") == "observed_leaf"
    }
    bridge_ids: set[str] = set()
    for rule in taxonomy.bridge_rules:
        if rule.source_genre_id in observed:
            bridge_ids.add(rule.target_genre_id)
        if rule.target_genre_id in observed:
            bridge_ids.add(rule.source_genre_id)
    return tuple(sorted(bridge_ids))


def _genre_names(genre_ids: tuple[str, ...], taxonomy: LayeredTaxonomy) -> tuple[str, ...]:
    return tuple(
        genre.name
        for genre in (taxonomy.genre_by_id(genre_id) for genre_id in genre_ids)
        if genre is not None
    )


def _facet_names(facet_ids: tuple[str, ...], taxonomy: LayeredTaxonomy) -> tuple[str, ...]:
    by_id = {facet.facet_id: facet.name for facet in taxonomy.facets}
    return tuple(by_id[facet_id] for facet_id in facet_ids if facet_id in by_id)


def _graph_fingerprint(
    *,
    taxonomy: LayeredTaxonomy,
    genre_rows: list[dict[str, Any]],
    facet_rows: list[dict[str, Any]],
) -> str:
    payload = {
        "taxonomy_version": taxonomy.version,
        "genres": sorted(
            tuple(sorted({
                "genre_id": genre.genre_id,
                "name": genre.name,
                "kind": genre.kind,
                "status": genre.status,
                "specificity_score": genre.specificity_score,
            }.items()))
            for genre in taxonomy.genres
        ),
        "edges": sorted(
            tuple(sorted({
                "source_genre_id": edge.source_genre_id,
                "target_genre_id": edge.target_genre_id,
                "edge_type": edge.edge_type,
                "weight": edge.weight,
                "confidence": edge.confidence,
            }.items()))
            for edge in taxonomy.edges
        ),
        "facets": sorted(
            tuple(sorted({
                "facet_id": facet.facet_id,
                "name": facet.name,
                "facet_type": facet.facet_type,
                "status": facet.status,
            }.items()))
            for facet in taxonomy.facets
        ),
        "bridge_rules": sorted(
            tuple(sorted({
                "source_genre_id": rule.source_genre_id,
                "target_genre_id": rule.target_genre_id,
                "required_family_min": rule.required_family_min,
                "required_facet_overlap": rule.required_facet_overlap,
                "required_sonic_similarity": rule.required_sonic_similarity,
                "required_transition_quality": rule.required_transition_quality,
                "mode_allowed": sorted(rule.mode_allowed),
            }.items()))
            for rule in taxonomy.bridge_rules
        ),
        "genre_assignments": sorted(
            tuple(sorted({
                "release_id": row["release_id"],
                "genre_id": row["genre_id"],
                "assignment_layer": row["assignment_layer"],
                "confidence": round(float(row["confidence"]), 6),
            }.items()))
            for row in genre_rows
        ),
        "facet_assignments": sorted(
            tuple(sorted({
                "release_id": row["release_id"],
                "facet_id": row["facet_id"],
                "confidence": round(float(row["confidence"]), 6),
            }.items()))
            for row in facet_rows
        ),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
