"""
Album genre frequency lab (MusicBrainz + Discogs only).

Scans album-level genres/styles from MusicBrainz and Discogs, normalizes them
using the same production normalization/broad filtering, and emits per-genre
frequency statistics.

Usage example:
    python -m experiments.genre_similarity_lab.album_genre_frequency_lab \
        --db-path experiments/genre_similarity_lab/metadata_lab.db \
        --config config.yaml \
        --garbage-path data/genre_filters/garbage_hard_block.csv \
        --meta-path data/genre_filters/meta_broad.csv \
        --min-albums 1 \
        --output experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv
"""

import argparse
import csv
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from src.config_loader import Config
from src.similarity_calculator import SimilarityCalculator
from experiments.genre_similarity_lab.genre_normalization import (
    load_filter_sets,
    normalize_and_filter_genres,
)

logger = logging.getLogger(__name__)


def load_list_from_csv(path: Optional[str]) -> Set[str]:
    """Load a single-column CSV with header 'genre' into a lowercase set."""
    if not path:
        return set()
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("CSV not found at %s; ignoring.", csv_path)
        return set()
    genres = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = (row.get("genre") or "").strip().lower()
            if g:
                genres.add(g)
    return genres


def resolve_db_path(config_path: str, explicit_db: Optional[str]) -> str:
    """Resolve DB path from CLI override or config.yaml."""
    if explicit_db:
        return explicit_db
    try:
        cfg = Config(config_path)
        return cfg.get("library", "database_path", default="data/metadata.db")
    except Exception as exc:
        logger.warning("Failed to load config %s (%s); falling back to data/metadata.db", config_path, exc)
        return "data/metadata.db"


def main():
    parser = argparse.ArgumentParser(description="Album genre frequency (MB + Discogs only).")
    parser.add_argument("--db-path", help="SQLite DB path (default from config.yaml).")
    parser.add_argument("--config", default="config.yaml", help="Config path (default: config.yaml).")
    parser.add_argument(
        "--garbage-path",
        help="CSV with column 'genre' listing hard-block genres (default: none).",
    )
    parser.add_argument(
        "--meta-path",
        help="CSV with column 'genre' listing meta/broad genres to skip (default: none).",
    )
    parser.add_argument(
        "--min-albums",
        type=int,
        default=1,
        help="Min album count for diagnostics only (does not filter output).",
    )
    parser.add_argument(
        "--output",
        default="experiments/genre_similarity_lab/artifacts/album_genre_frequency.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = resolve_db_path(args.config, args.db_path)
    logger.info("Using DB: %s", db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # SimilarityCalculator provides genre normalization and broad filtering.
    try:
        cfg = Config(args.config)
        sim_calc = SimilarityCalculator(db_path=db_path, config=cfg.config)
    except Exception:
        sim_calc = SimilarityCalculator(db_path=db_path, config={})

    # Collect album -> track_ids once to avoid repeated queries.
    album_tracks: Dict[str, Set[str]] = defaultdict(set)
    cur = conn.cursor()
    cur.execute("SELECT album_id, track_id FROM tracks WHERE album_id IS NOT NULL")
    for row in cur.fetchall():
        album_id = row["album_id"]
        track_id = row["track_id"]
        if album_id and track_id:
            album_tracks[album_id].add(track_id)

    # Sources to include.
    sources = ["musicbrainz_release", "discogs_release", "discogs_master"]

    # Filter sets (broad from config, garbage/meta from CSVs)
    broad_filters = getattr(sim_calc, "broad_filters", set())
    broad_set, garbage_set, meta_set = load_filter_sets(broad_filters, args.garbage_path, args.meta_path)

    genre_albums: Dict[str, Set[str]] = defaultdict(set)
    genre_tracks: Dict[str, Set[str]] = defaultdict(set)
    genre_sources: Dict[str, Set[str]] = defaultdict(set)

    cur.execute(
        f"""
        SELECT album_id, genre, source
        FROM album_genres
        WHERE source IN ({','.join('?' for _ in sources)})
          AND genre != '__EMPTY__'
        """,
        sources,
    )

    album_rows = cur.fetchall()
    logger.info("Fetched %d album-genre rows", len(album_rows))
    raw_genres_seen = len(album_rows)

    # Accumulate raw tags per album, then normalize/filter via shared helper
    album_genre_accum: Dict[str, list] = defaultdict(list)
    album_sources_accum: Dict[str, Set[str]] = defaultdict(set)
    for row in album_rows:
        album_id = row["album_id"]
        genre_raw = row["genre"]
        source = row["source"]
        album_genre_accum[album_id].append(genre_raw)
        album_sources_accum[album_id].add("musicbrainz" if source == "musicbrainz_release" else "discogs")

    for album_id, raw_tags in album_genre_accum.items():
        normalized = normalize_and_filter_genres(
            raw_tags,
            broad_set=broad_set,
            garbage_set=garbage_set,
            meta_set=meta_set,
            canonical_set=None,
        )
        if not normalized:
            continue
        for g in normalized:
            genre_albums[g].add(album_id)
            if album_id in album_tracks:
                genre_tracks[g].update(album_tracks[album_id])
            genre_sources[g].update(album_sources_accum[album_id])

    filtered_count = len(genre_albums)
    logger.info("Raw genre rows scanned: %d", raw_genres_seen)
    logger.info("Unique genres after filtering: %d", filtered_count)

    # Diagnostics: top 20 by album_count
    top20 = sorted(((g, len(albs)) for g, albs in genre_albums.items()), key=lambda x: x[1], reverse=True)[:20]
    logger.info("Top 20 genres by album_count:")
    for g, c in top20:
        logger.info("  %-30s %6d", g, c)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_genre", "album_count", "track_count", "sources"])
        writer.writeheader()
        for genre, albums in sorted(genre_albums.items()):
            albums_count = len(albums)
            tracks_count = len(genre_tracks.get(genre, set()))
            sources_str = ";".join(sorted(genre_sources.get(genre, set())))
            writer.writerow(
                {
                    "source_genre": genre,
                    "album_count": albums_count,
                    "track_count": tracks_count,
                    "sources": sources_str,
                }
            )

    logger.info("Wrote genre frequency CSV: %s", out_path)


if __name__ == "__main__":
    main()
