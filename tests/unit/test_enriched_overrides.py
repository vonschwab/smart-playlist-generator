"""Tests for EnrichedGenreResolver.get_all_enrichment (bulk signature + override delta)."""

from __future__ import annotations

import json
import sqlite3


from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


def _make_db(tmp_path):
    db_path = tmp_path / "enrichment.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE enriched_genre_signatures(release_key TEXT, "
        "normalized_artist TEXT, normalized_album TEXT, signature_json TEXT)"
    )
    conn.execute(
        "CREATE TABLE ai_genre_user_overrides("
        "release_key TEXT, normalized_artist TEXT, normalized_album TEXT, "
        "genres_add_json TEXT, genres_remove_json TEXT, updated_at TEXT)"
    )
    # (a) signature release with an override that removes one sig genre and adds one
    conn.execute(
        "INSERT INTO enriched_genre_signatures(release_key, signature_json) VALUES (?, ?)",
        ("artist_a::album_a", json.dumps({"genres": ["shoegaze", "dreampop", "indie rock"]})),
    )
    conn.execute(
        "INSERT INTO ai_genre_user_overrides("
        "release_key, genres_add_json, genres_remove_json) VALUES (?, ?, ?)",
        ("artist_a::album_a", json.dumps(["slowcore"]), json.dumps(["indie rock"])),
    )
    # (b) override-only release (no signature)
    conn.execute(
        "INSERT INTO ai_genre_user_overrides("
        "release_key, genres_add_json, genres_remove_json) VALUES (?, ?, ?)",
        ("artist_b::album_b", json.dumps(["art punk", "minimal wave"]), json.dumps(["pop"])),
    )
    conn.commit()
    conn.close()
    return db_path


def test_get_all_enrichment_signature_applies_override_delta(tmp_path):
    resolver = EnrichedGenreResolver(_make_db(tmp_path))
    result = resolver.get_all_enrichment()

    assert "artist_a::album_a" in result
    rec = result["artist_a::album_a"]
    # genres = (sig - remove) + add
    assert rec["genres"] == ["shoegaze", "dreampop", "slowcore"]
    assert "indie rock" not in rec["genres"]
    # raw override lists carried through
    assert rec["add"] == ["slowcore"]
    assert rec["remove"] == ["indie rock"]


def test_get_all_enrichment_override_only_release(tmp_path):
    resolver = EnrichedGenreResolver(_make_db(tmp_path))
    result = resolver.get_all_enrichment()

    assert "artist_b::album_b" in result
    rec = result["artist_b::album_b"]
    assert rec["genres"] is None
    assert rec["add"] == ["art punk", "minimal wave"]
    assert rec["remove"] == ["pop"]


def test_get_all_enrichment_covers_union_of_both_tables(tmp_path):
    resolver = EnrichedGenreResolver(_make_db(tmp_path))
    result = resolver.get_all_enrichment()
    assert set(result.keys()) == {"artist_a::album_a", "artist_b::album_b"}
