#!/usr/bin/env python3
"""
Build Beat 3-Tower Artifacts
============================

Builds the genre + metadata artifact skeleton over the `sonic_features`
universe gate (tracks with beat3tower features recorded in the DB — that
gate defines the track universe, even though the tower vectors themselves
are no longer baked here). The sonic space is NOT built by this script:
`stage_artifacts` folds the active variant sidecar (X_sonic_muq*) in
immediately after, which also stamps `X_sonic_variant`. (SP-B)

Usage:
    python scripts/build_beat3tower_artifacts.py \
        --db-path data/metadata.db \
        --config config.yaml \
        --output data/artifacts/beat3tower_32k/data_matrices_step1.npz

Features:
- Loads tracks gated on beat3tower feature presence (universe gate)
- Builds genre matrices (raw + smoothed) from the configured genre source
- Saves genre + metadata matrices for playlist generation

Output NPZ contents:
- X_genre_raw, X_genre_smoothed, genre_vocab: Genre matrices
- track_ids, track_artists, track_titles, artist_keys, durations_ms: Metadata
- build_config: Build provenance
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Genre normalization (Taxonomy v1)
try:
    from src.genre.normalize import normalize_and_split_genre
    GENRE_NORMALIZATION_AVAILABLE = True
except ImportError:
    GENRE_NORMALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build genre + metadata artifacts over the beat3tower feature universe gate"
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to metadata database",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for NPZ artifact",
    )
    parser.add_argument(
        "--genre-sim-path",
        help="Path to genre similarity matrix NPZ (optional)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="Maximum tracks to include (0 = all)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--no-genre-normalization",
        action="store_true",
        help="Disable Genre Taxonomy v1 normalization (use raw genres)",
    )
    parser.add_argument(
        "--sidecar-db",
        default="data/ai_genre_enrichment.db",
        help="Path to AI genre enrichment sidecar DB (used only with explicit non-legacy genre source)",
    )
    parser.add_argument(
        "--genre-source",
        choices=["legacy", "enriched", "graph", "hybrid_shadow"],
        default=None,
        help="Genre source for artifact matrices. Defaults to config, then legacy. "
             "graph = published authority tables in metadata.db (release_effective_genres).",
    )
    return parser.parse_args()


def normalize_artist_key(raw: Optional[str], track_id: Optional[str]) -> str:
    """Normalize artist identifier with fallbacks."""
    from src.string_utils import normalize_artist_key as _normalize_artist_key

    key = _normalize_artist_key(raw or "")
    if key:
        return key
    if track_id:
        return f"unknown:{track_id}"
    return "unknown"


def load_tracks_with_beat3tower(
    db_path: str,
    max_tracks: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load tracks that have beat3tower features.

    Returns:
        Tuple of (tracks_metadata, features_list)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check for required columns
    cursor.execute("PRAGMA table_info(tracks)")
    columns = {row[1] for row in cursor.fetchall()}

    if "sonic_features" not in columns:
        raise RuntimeError("tracks table missing sonic_features column")

    has_norm_artist = "norm_artist" in columns
    norm_artist_expr = "norm_artist" if has_norm_artist else "artist"

    limit_clause = f" LIMIT {int(max_tracks)}" if max_tracks > 0 else ""

    cursor.execute(
        f"""
        SELECT track_id, artist, title, album, {norm_artist_expr} as norm_artist, sonic_features, duration_ms
        FROM tracks
        WHERE sonic_features IS NOT NULL
        {limit_clause}
        """
    )
    rows = cursor.fetchall()
    conn.close()

    tracks = []
    features_list = []
    skipped_non_beat3tower = 0
    skipped_parse_error = 0

    for row in rows:
        track_id = row["track_id"]
        raw_features = row["sonic_features"]

        try:
            features = json.loads(raw_features)
        except Exception:
            skipped_parse_error += 1
            continue

        # Check if this is a beat3tower feature set
        if not _is_beat3tower_features(features):
            skipped_non_beat3tower += 1
            continue

        tracks.append({
            "track_id": track_id,
            "artist": row["artist"] or "",
            "title": row["title"] or "",
            "album": row["album"] or "",
            "norm_artist": row["norm_artist"] or row["artist"] or "",
            "duration_ms": row["duration_ms"] if row["duration_ms"] is not None else 0,
        })
        features_list.append(features)

    logger.info(
        f"Loaded {len(tracks)} tracks with beat3tower features "
        f"(skipped {skipped_non_beat3tower} non-beat3tower, {skipped_parse_error} parse errors)"
    )

    return tracks, features_list


def _is_beat3tower_features(features: Dict[str, Any]) -> bool:
    """Check if features are from beat3tower extraction."""
    # Check for beat3tower structure
    if "full" in features and isinstance(features["full"], dict):
        full = features["full"]
        # Check for tower structure
        if all(key in full for key in ["rhythm", "timbre", "harmony"]):
            return True
        # Check for extraction_method marker
        if full.get("extraction_method") == "beat3tower":
            return True
    return False


def smooth_genres(
    X: np.ndarray,
    S: Optional[np.ndarray],
    vocab: List[str],
    sim_vocab: Optional[List[str]],
) -> np.ndarray:
    """Apply genre similarity smoothing."""
    if S is None or sim_vocab is None:
        return X.astype(np.float32)
    index = {g: i for i, g in enumerate(sim_vocab)}
    present = [i for i, g in enumerate(vocab) if g in index]
    if not present:
        return X.astype(np.float32)
    S_sub = np.eye(len(vocab), dtype=np.float32)
    sim_idx = [index[vocab[i]] for i in present]
    S_sub[np.ix_(present, present)] = S[np.ix_(sim_idx, sim_idx)].astype(np.float32)
    return X.astype(np.float32) @ S_sub


def load_genres_for_tracks(
    db_path: str,
    track_ids: List[str],
    normalize_genres: bool = True,
    tracks_metadata: Optional[List[Dict[str, Any]]] = None,
    enriched_resolver: Optional[Any] = None,
    use_graph_genres: bool = False,
) -> Tuple[List[List[Tuple[str, float]]], List[str], Dict[str, int]]:
    """
    Load weighted genres for tracks from database, including artist and album genres.

    Inherits genres with priority weights:
    - track_genres: 1.0
    - album_genres: 0.8
    - artist_genres: 0.5

    When an EnrichedGenreResolver is provided, enriched releases use the enriched
    signature as the authoritative source (replacement, not supplement). Only
    unenriched releases fall back to the raw DB lookups.

    When ``use_graph_genres`` is True, albums present in the published authority
    table (``release_effective_genres``, maintained by the publish stage) use
    those graph genres as the authoritative source — same replacement semantics.
    genre_ids are mapped to canonical display names via the published taxonomy
    copy so the vocabulary aligns with the graph similarity matrix. Weights are
    ``confidence x layer`` (observed_leaf 1.0, inferred_family 0.5 — families
    are hubs, kept for structure but damped per the rare>common principle).
    Tracks without a covered album fall back to the raw lookups.

    Args:
        db_path: Path to database
        track_ids: List of track IDs to load genres for
        normalize_genres: Apply Genre Taxonomy v1 normalization (default True)
        tracks_metadata: Optional list of track metadata dicts with 'artist' and 'album' keys
        enriched_resolver: Optional EnrichedGenreResolver for sidecar-aware genre lookups
        use_graph_genres: Source genres from the published graph authority tables

    Returns:
        Tuple of (weighted_genre_lists per track, vocabulary, stats dict)
        Each genre list contains (genre, weight) tuples
    """
    import hashlib

    # Source weights - track > album > artist
    WEIGHT_TRACK = 1.0
    WEIGHT_ALBUM = 0.8
    WEIGHT_ARTIST = 0.5

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check which tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row['name'] for row in cursor.fetchall()}
    has_track_genres = 'track_genres' in existing_tables
    has_album_genres = 'album_genres' in existing_tables
    has_artist_genres = 'artist_genres' in existing_tables

    vocab_counts: Dict[str, int] = {}
    total_raw_genres = 0
    total_normalized_tokens = 0
    apply_normalization = normalize_genres and GENRE_NORMALIZATION_AVAILABLE

    # Build track_id -> (artist, album) mapping if not provided
    track_info: Dict[str, Tuple[str, str]] = {}
    if tracks_metadata:
        for i, tid in enumerate(track_ids):
            if i < len(tracks_metadata):
                track_info[tid] = (
                    tracks_metadata[i].get('artist', ''),
                    tracks_metadata[i].get('album', ''),
                )
    else:
        # Query from database
        batch_size = 800
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"SELECT track_id, artist, album FROM tracks WHERE track_id IN ({placeholders})",
                batch,
            )
            for row in cursor.fetchall():
                track_info[row['track_id']] = (row['artist'] or '', row['album'] or '')

    # Initialize genre storage: track_id -> {genre: max_weight}
    track_genres: Dict[str, Dict[str, float]] = {tid: {} for tid in track_ids}

    # Tracks whose release has an enriched signature — populated below after
    # add_genre is defined. Raw lookups skip these so enriched is authoritative.
    enriched_tids: set = set()
    enriched_release_count = 0

    def add_genre(tid: str, raw_genre: str, weight: float) -> None:
        """Add a genre to a track with weight, keeping max weight for duplicates."""
        nonlocal total_raw_genres, total_normalized_tokens

        if not raw_genre or raw_genre == '__EMPTY__':
            return

        total_raw_genres += 1

        if apply_normalization:
            tokens = normalize_and_split_genre(raw_genre)
            # Distribute weight across split tokens
            per_token_weight = weight / len(tokens) if tokens else weight
            for token in tokens:
                if token:
                    # Keep max weight for this token
                    current = track_genres[tid].get(token, 0)
                    if per_token_weight > current:
                        if current == 0:
                            vocab_counts[token] = vocab_counts.get(token, 0) + 1
                            total_normalized_tokens += 1
                        track_genres[tid][token] = per_token_weight
        else:
            current = track_genres[tid].get(raw_genre, 0)
            if weight > current:
                if current == 0:
                    vocab_counts[raw_genre] = vocab_counts.get(raw_genre, 0) + 1
                    total_normalized_tokens += 1
                track_genres[tid][raw_genre] = weight

    def add_canonical(tid: str, token: str, weight: float) -> None:
        """Add an already-canonical graph token, bypassing legacy normalization.

        Graph display names are final vocabulary ('indie/alternative' is one
        token); running them through normalize_and_split_genre would split or
        alias-fold them away from the graph similarity matrix vocab.
        """
        nonlocal total_raw_genres, total_normalized_tokens
        if not token:
            return
        total_raw_genres += 1
        current = track_genres[tid].get(token, 0)
        if weight > current:
            if current == 0:
                vocab_counts[token] = vocab_counts.get(token, 0) + 1
                total_normalized_tokens += 1
            track_genres[tid][token] = weight

    # Per-track override deltas for override-only releases (no signature).
    # Applied after raw tiers load (step 4). Maps tid -> (add_list, remove_list).
    track_override_delta: Dict[str, Tuple[List[str], List[str]]] = {}

    # Tracks covered by the published graph authority — raw lookups skip these.
    graph_tids: set = set()
    graph_album_ids: set = set()
    inferred_only_albums: set = set()
    inferred_rows_dropped = 0

    # 0a. Published graph genres (release_effective_genres) — authoritative for
    #     covered albums, same replacement semantics as enriched signatures.
    #     User overrides are already baked in by the publish stage.
    if use_graph_genres:
        required = {"release_effective_genres", "genre_graph_canonical_genres"}
        missing = required - existing_tables
        if missing:
            raise RuntimeError(
                f"genre_source=graph but {sorted(missing)} missing from {db_path}; "
                "run the publish stage first (analyze_library --stages publish)"
            )
        from src.genre.authority import canonical_genre_names, resolved_genres_by_album

        by_album = resolved_genres_by_album(conn)
        id_to_name = canonical_genre_names(conn)
        # Similarity vectors use OBSERVED layers only. Inferred taxonomy
        # ancestors put rock/pop/indie-alternative mass on nearly every release
        # and saturate genre cosine (2026-06-12 diagnosis: random-pair p50 of
        # X_genre_smoothed hit 0.414). They stay in the authority for display;
        # related-genre mass re-enters X_genre_smoothed via the similarity
        # matrix, which is the controlled mechanism. `legacy` rows are absorbed
        # raw tags of un-enriched albums — observations, full weight.
        solid_layer_weights = {"observed_leaf": 1.0, "legacy": 1.0}
        inferred_layers = {"inferred_family", "inferred_parent"}
        inferred_fallback_weight = 0.5
        unmapped_ids: set = set()

        # track_id -> album_id (tracks table; publish keys the authority by it)
        album_of: Dict[str, str] = {}
        for i in range(0, len(track_ids), 800):
            batch = track_ids[i : i + 800]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"SELECT track_id, album_id FROM tracks WHERE track_id IN ({placeholders})",
                batch,
            )
            for row in cursor.fetchall():
                if row["album_id"]:
                    album_of[row["track_id"]] = row["album_id"]

        def _resolve_name(genre_id: str) -> str:
            name = id_to_name.get(genre_id)
            if name is None:
                if genre_id not in unmapped_ids:
                    unmapped_ids.add(genre_id)
                    logger.warning(
                        "graph genre_id %r has no canonical name in "
                        "genre_graph_canonical_genres; keeping raw id as token",
                        genre_id,
                    )
                name = genre_id
            return name

        for tid in track_ids:
            album_id = album_of.get(tid)
            rows = by_album.get(album_id) if album_id else None
            if not rows:
                continue
            unknown_layers = (
                {r.assignment_layer for r in rows}
                - set(solid_layer_weights)
                - inferred_layers
            )
            if unknown_layers:
                raise RuntimeError(
                    f"release_effective_genres has assignment_layer(s) "
                    f"{sorted(unknown_layers)} (album_id={album_id}) that this "
                    "builder does not recognize; the publish schema moved — "
                    "update solid_layer_weights/inferred_layers in "
                    "load_genres_for_tracks instead of silently weighting them"
                )
            first_seen = album_id not in graph_album_ids
            graph_tids.add(tid)
            graph_album_ids.add(album_id)
            solid = [r for r in rows if r.assignment_layer in solid_layer_weights]
            if solid:
                if first_seen:
                    inferred_rows_dropped += len(rows) - len(solid)
                for grow in solid:
                    weight = grow.confidence * solid_layer_weights[grow.assignment_layer]
                    add_canonical(tid, _resolve_name(grow.genre_id), weight)
            else:
                # Inferred-only album: keep damped ancestors so the release
                # retains a usable genre vector instead of silently zeroing
                # out of every genre gate.
                inferred_only_albums.add(album_id)
                for grow in rows:
                    weight = grow.confidence * inferred_fallback_weight
                    add_canonical(tid, _resolve_name(grow.genre_id), weight)
        if graph_tids:
            logger.info(
                "Using published graph genres for %d tracks across %d albums "
                "(observed/legacy layers only; %d inferred rows excluded; "
                "%d inferred-only albums kept at %.1fx damped fallback; "
                "raw lookups skipped for these)",
                len(graph_tids),
                len(graph_album_ids),
                inferred_rows_dropped,
                len(inferred_only_albums),
                inferred_fallback_weight,
            )

    # 0. Apply enriched signatures first — these are authoritative, raw lookups skip them.
    #    User overrides are applied here too: signature releases get the
    #    (sig - remove) + add set as a full replacement; override-only releases
    #    keep their raw tiers and have the delta applied in step 4 below.
    if enriched_resolver is not None:
        # Fetch all enrichment (signatures UNION overrides) in one query.
        all_enrichment = enriched_resolver.get_all_enrichment()
        enriched_release_keys: set = set()
        override_only_release_keys: set = set()
        for tid in track_ids:
            artist, album = track_info.get(tid, ('', ''))
            if not artist or not album:
                continue
            release_key = enriched_resolver.make_release_key(artist, album)
            rec = all_enrichment.get(release_key)
            if rec is None:
                continue
            if rec["genres"] is not None:
                # Signature (override-applied): full replacement of raw tiers.
                genres = rec["genres"]
                if not genres:
                    continue
                enriched_tids.add(tid)
                enriched_release_keys.add(release_key)
                for g in genres:
                    add_genre(tid, g, WEIGHT_TRACK)
            else:
                # Override-only: keep raw tiers; record delta for step 4.
                if rec["add"] or rec["remove"]:
                    track_override_delta[tid] = (rec["add"], rec["remove"])
                    override_only_release_keys.add(release_key)
        enriched_release_count = len(enriched_release_keys)
        if enriched_release_count:
            logger.info(
                f"Using enriched genres for {len(enriched_tids)} tracks "
                f"across {enriched_release_count} releases (raw lookups skipped for these)"
            )
        if override_only_release_keys:
            logger.info(
                f"Applying override-only deltas for {len(track_override_delta)} tracks "
                f"across {len(override_only_release_keys)} releases (raw tiers retained)"
            )

    # 1. Load track genres (highest priority - weight 1.0)
    if has_track_genres:
        batch_size = 800
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"SELECT track_id, genre FROM track_genres WHERE track_id IN ({placeholders})",
                batch,
            )
            for row in cursor.fetchall():
                if row['track_id'] in enriched_tids or row['track_id'] in graph_tids:
                    continue
                add_genre(row['track_id'], row['genre'], WEIGHT_TRACK)

    # 2. Load album genres (weight 0.8)
    if has_album_genres:
        # Build album_id -> [track_ids] mapping
        album_to_tracks: Dict[str, List[str]] = {}
        for tid in track_ids:
            if tid in enriched_tids or tid in graph_tids:
                continue
            artist, album = track_info.get(tid, ('', ''))
            if artist and album:
                # Compute album_id as md5 hash (matching schema generation)
                album_id = hashlib.md5(f"{artist}|{album}".lower().encode('utf-8')).hexdigest()[:16]
                if album_id not in album_to_tracks:
                    album_to_tracks[album_id] = []
                album_to_tracks[album_id].append(tid)

        # Query album genres
        album_ids = list(album_to_tracks.keys())
        for i in range(0, len(album_ids), batch_size):
            batch = album_ids[i : i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            cursor.execute(
                f"SELECT album_id, genre FROM album_genres WHERE album_id IN ({placeholders}) AND genre != '__EMPTY__'",
                batch,
            )
            for row in cursor.fetchall():
                for tid in album_to_tracks.get(row['album_id'], []):
                    add_genre(tid, row['genre'], WEIGHT_ALBUM)

    # 3. Load artist genres (lowest priority - weight 0.5)
    if has_artist_genres:
        # Build artist -> [track_ids] mapping
        artist_to_tracks: Dict[str, List[str]] = {}
        for tid in track_ids:
            if tid in enriched_tids or tid in graph_tids:
                continue
            artist, _ = track_info.get(tid, ('', ''))
            if artist:
                artist_key = artist.lower().strip()
                if artist_key not in artist_to_tracks:
                    artist_to_tracks[artist_key] = []
                artist_to_tracks[artist_key].append(tid)

        # Query artist genres - need to match case-insensitively
        cursor.execute("SELECT DISTINCT artist, genre FROM artist_genres WHERE genre != '__EMPTY__'")
        for row in cursor.fetchall():
            artist_key = (row['artist'] or '').lower().strip()
            if artist_key in artist_to_tracks:
                for tid in artist_to_tracks[artist_key]:
                    add_genre(tid, row['genre'], WEIGHT_ARTIST)

    # 4. Apply override-only deltas on top of the raw tiers. Removes are global
    #    (subtract from whatever tier produced the token); adds use track weight.
    #    Remove labels are normalized the same way genres are so they match the
    #    stored token form.
    for tid, (add_labels, remove_labels) in track_override_delta.items():
        if remove_labels:
            remove_tokens: set = set()
            for label in remove_labels:
                if apply_normalization:
                    remove_tokens.update(t for t in normalize_and_split_genre(label) if t)
                else:
                    token = label.strip().lower()
                    if token:
                        remove_tokens.add(token)
            for token in remove_tokens:
                track_genres[tid].pop(token, None)
        for label in add_labels:
            add_genre(tid, label, WEIGHT_TRACK)

    conn.close()

    # Convert to list of (genre, weight) tuples
    genre_lists = [
        [(g, w) for g, w in track_genres.get(tid, {}).items()]
        for tid in track_ids
    ]
    vocab = sorted(vocab_counts.keys())
    stats = {
        "raw_genres": total_raw_genres,
        "normalized_tokens": total_normalized_tokens,
        "normalization_applied": apply_normalization,
        "sources": {
            "track_genres": has_track_genres,
            "album_genres": has_album_genres,
            "artist_genres": has_artist_genres,
            "enriched_signatures": bool(enriched_resolver is not None and enriched_tids),
            "graph_authority": bool(graph_tids),
        },
        "enriched_tracks": len(enriched_tids),
        "enriched_releases": enriched_release_count,
        "graph_tracks": len(graph_tids),
        "graph_albums": len(graph_album_ids),
        "graph_inferred_only_albums": len(inferred_only_albums),
        "graph_inferred_rows_dropped": inferred_rows_dropped,
    }
    return genre_lists, vocab, stats


def build_genre_matrices(
    genre_lists: List[List[Tuple[str, float]]],
    vocab: List[str],
    genre_sim_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build genre matrices from weighted genre lists.

    Args:
        genre_lists: List of [(genre, weight), ...] per track
        vocab: Sorted vocabulary of all genres
        genre_sim_path: Optional path to genre similarity matrix

    Returns:
        Tuple of (X_genre_raw, X_genre_smoothed)
    """
    vocab_index = {g: i for i, g in enumerate(vocab)}
    n_tracks = len(genre_lists)
    n_genres = len(vocab)

    X_genre_raw = np.zeros((n_tracks, n_genres), dtype=np.float32)
    for i, genres in enumerate(genre_lists):
        for g, weight in genres:
            j = vocab_index.get(g)
            if j is not None:
                # Use the weight, keeping max if duplicates
                X_genre_raw[i, j] = max(X_genre_raw[i, j], weight)

    # Apply smoothing if similarity matrix provided
    X_genre_smoothed = X_genre_raw.copy()
    if genre_sim_path:
        try:
            sim_npz = np.load(genre_sim_path, allow_pickle=True)
            sim_vocab = sim_npz["genre_vocab"].tolist()
            S = sim_npz["S"]
            X_genre_smoothed = smooth_genres(X_genre_raw, S, vocab, sim_vocab)
            logger.info(f"Applied genre smoothing from {genre_sim_path}")
        except Exception as e:
            logger.warning(f"Failed to load genre similarity matrix: {e}")

    return X_genre_raw, X_genre_smoothed


def refresh_genre_matrices(
    artifact_path: str,
    db_path: str,
    *,
    genre_sim_path: Optional[str],
    sidecar_db: str,
    config_path: str,
) -> dict:
    """Re-bake ONLY the genre matrices in an existing artifact from the authority.

    Loads the NPZ, recomputes X_genre_raw/smoothed + genre_vocab for the
    artifact's track order using the same loaders as a full build, and re-saves
    with every other array (sonic/MERT, metadata) written back unchanged.
    """
    from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource, make_resolver
    from src.config_loader import Config

    npz = np.load(artifact_path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    track_ids = [str(t) for t in data["track_ids"].tolist()]

    config_genre_source = (
        Config(config_path).config.get("playlists", {}).get("ds_pipeline", {}).get("genre_source")
    )
    genre_source = GenreArtifactSource.resolve(config_genre_source)
    resolver = make_resolver(genre_source, sidecar_db)

    tracks_meta, _features = load_tracks_with_beat3tower(db_path)
    by_id = {t["track_id"]: t for t in tracks_meta}
    tracks_metadata = [by_id[tid] for tid in track_ids if tid in by_id]

    genre_lists, vocab, _stats = load_genres_for_tracks(
        db_path, track_ids, normalize_genres=True,
        tracks_metadata=tracks_metadata, enriched_resolver=resolver,
        use_graph_genres=(genre_source is GenreArtifactSource.GRAPH),
    )
    X_genre_raw, X_genre_smoothed = build_genre_matrices(genre_lists, vocab, genre_sim_path)

    data["X_genre_raw"] = X_genre_raw
    data["X_genre_smoothed"] = X_genre_smoothed
    data["genre_vocab"] = np.array(vocab, dtype=object)
    np.savez(artifact_path, **data)
    logger.info("Re-baked genre matrices: %d tracks, %d genres", len(track_ids), len(vocab))
    return {"n_tracks": len(track_ids), "n_genres": len(vocab)}


def build_artifacts(args: argparse.Namespace, enriched_resolver: Optional[Any] = None) -> None:
    """Main artifact building workflow.

    Args:
        args: Parsed CLI arguments
        enriched_resolver: Optional EnrichedGenreResolver for sidecar-aware genre lookups.
            When provided, enriched releases use their authoritative signatures and bypass
            raw track/album/artist genre lookups.
    """
    from src.ai_genre_enrichment.artifact_modes import GenreArtifactSource, make_resolver
    from src.config_loader import Config

    config_genre_source = (
        Config(args.config).config.get("playlists", {}).get("ds_pipeline", {}).get("genre_source")
    )
    genre_source = GenreArtifactSource.resolve(getattr(args, "genre_source", None) or config_genre_source)
    if enriched_resolver is None:
        enriched_resolver = make_resolver(
            genre_source,
            getattr(args, "sidecar_db", "data/ai_genre_enrichment.db"),
        )
    logger.info("Artifact genre source: %s", genre_source.value)

    logger.info("Loading tracks with beat3tower features...")
    tracks, _features_list = load_tracks_with_beat3tower(args.db_path, args.max_tracks)

    if not tracks:
        raise RuntimeError("No tracks with beat3tower features found")

    logger.info(f"Processing {len(tracks)} tracks...")

    # Load genres with optional normalization (includes artist/album genre inheritance)
    normalize_genres = not args.no_genre_normalization
    logger.info(f"Loading genre information (normalization={'enabled' if normalize_genres else 'disabled'})...")
    track_ids = [t["track_id"] for t in tracks]
    genre_lists, vocab, genre_stats = load_genres_for_tracks(
        args.db_path, track_ids, normalize_genres=normalize_genres,
        tracks_metadata=tracks, enriched_resolver=enriched_resolver,
        use_graph_genres=(genre_source is GenreArtifactSource.GRAPH),
    )
    X_genre_raw, X_genre_smoothed = build_genre_matrices(
        genre_lists, vocab, args.genre_sim_path
    )

    # Log genre loading and normalization statistics
    sources = genre_stats.get("sources", {})
    source_names = [k for k, v in sources.items() if v]
    logger.info(f"Genre sources: {', '.join(source_names) if source_names else 'none'}")
    if genre_stats.get("enriched_tracks"):
        logger.info(
            f"Enriched signatures applied to {genre_stats['enriched_tracks']} tracks "
            f"({genre_stats['enriched_releases']} releases)"
        )

    if genre_stats["normalization_applied"]:
        raw_count = genre_stats["raw_genres"]
        norm_count = genre_stats["normalized_tokens"]
        reduction = 100 * (1 - norm_count / raw_count) if raw_count > 0 else 0
        logger.info(
            f"Genre normalization: {raw_count} raw → {norm_count} tokens "
            f"({reduction:.1f}% reduction), vocabulary size: {len(vocab)}"
        )
    elif normalize_genres and not GENRE_NORMALIZATION_AVAILABLE:
        logger.warning("Genre normalization requested but taxonomy module not available - using raw genres")
        logger.info(f"Genre vocabulary size: {len(vocab)}")
    else:
        logger.info(f"Genre vocabulary size: {len(vocab)} (raw genres, no normalization)")

    # Log weight distribution
    logger.info("Genre weights: track=1.0, album=0.8, artist=0.5")

    # Prepare metadata
    artist_keys = [
        normalize_artist_key(t["norm_artist"], t["track_id"]) for t in tracks
    ]
    track_artists = [t["artist"] for t in tracks]
    track_titles = [t["title"] for t in tracks]
    durations_ms = np.array([t["duration_ms"] for t in tracks], dtype=np.int32)

    # Save artifact
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving artifact to {out_path}...")
    np.savez(
        out_path,
        # Genre matrices
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=np.array(vocab, dtype=object),
        # Track metadata
        track_ids=np.array(track_ids, dtype=object),
        track_artists=np.array(track_artists, dtype=object),
        track_titles=np.array(track_titles, dtype=object),
        artist_keys=np.array(artist_keys, dtype=object),
        durations_ms=durations_ms,
        # Build metadata. The sonic space is NOT baked here: stage_artifacts
        # folds the active variant sidecar (X_sonic_muq*) immediately after,
        # which also stamps X_sonic_variant. (SP-B)
        build_config={
            'random_seed': args.random_seed,
            'extraction_method': 'universe_gate_beat3tower_features',
            'genre_normalization': genre_stats["normalization_applied"],
            'genre_stats': genre_stats,
            'genre_source': genre_source.value,
        },
    )

    # Log duration stats
    valid_durations = durations_ms[durations_ms > 0]
    if len(valid_durations) > 0:
        mean_dur_sec = float(np.mean(valid_durations)) / 1000.0
        logger.info(f"Duration statistics: {len(valid_durations)}/{len(durations_ms)} tracks with duration (mean={mean_dur_sec:.1f}s)")

    logger.info(
        f"Artifact saved successfully: "
        f"{len(tracks)} tracks, "
        f"{len(vocab)} genres (sonic space folded separately by stage_artifacts)"
    )


def main() -> None:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    from src.logging_utils import configure_logging
    configure_logging(level=logging.getLevelName(log_level), force=True)

    try:
        build_artifacts(args)
    except Exception as e:
        logger.error(f"Build failed: {e}")
        raise


if __name__ == "__main__":
    main()
