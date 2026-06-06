"""Tests for source-page skip support used by resumable collection passes."""

from __future__ import annotations

from src.ai_genre_enrichment.storage import SidecarStore


def _page(store, release_key, source_url, source_type):
    store.upsert_source_page(
        release_key=release_key,
        normalized_artist=release_key.split("::")[0],
        normalized_album=release_key.split("::")[1],
        album_id=None,
        source_url=source_url,
        source_type=source_type,
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="test",
    )


def test_release_keys_with_source_type_empty(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    assert store.release_keys_with_source_type("lastfm_tags") == set()


def test_release_keys_with_source_type_filters_by_type(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _page(store, "acetone::york blvd", "lastfm://artist/acetone/album/york blvd", "lastfm_tags")
    _page(store, "caribou::swim", "https://caribou.bandcamp.com/album/swim", "bandcamp_release")

    assert store.release_keys_with_source_type("lastfm_tags") == {"acetone::york blvd"}
    assert store.release_keys_with_source_type("bandcamp_release") == {"caribou::swim"}


def test_release_keys_with_source_type_dedups_release(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    # Same release, two Last.fm URLs (artist-level + album-level) -> one release_key.
    _page(store, "a::b", "lastfm://artist/a", "lastfm_tags")
    _page(store, "a::b", "lastfm://artist/a/album/b", "lastfm_tags")
    assert store.release_keys_with_source_type("lastfm_tags") == {"a::b"}


def test_attempt_ledger_records_hits_and_misses(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    assert store.release_keys_attempted("bandcamp") == set()

    store.record_source_attempt("acetone::york blvd", "bandcamp", "miss")
    store.record_source_attempt("duster::stratosphere", "bandcamp", "hit",
                                "https://duster.bandcamp.com/album/stratosphere")
    # A miss counts as "attempted" — that's the whole point (don't re-pay).
    assert store.release_keys_attempted("bandcamp") == {
        "acetone::york blvd", "duster::stratosphere",
    }


def test_attempt_ledger_upserts_latest_status(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.record_source_attempt("a::b", "bandcamp", "miss")
    store.record_source_attempt("a::b", "bandcamp", "hit", "https://a.bandcamp.com/album/b")
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT status, detail FROM ai_genre_source_attempts "
            "WHERE release_key='a::b' AND source_type='bandcamp'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "hit"


def test_attempt_ledger_is_source_scoped(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.record_source_attempt("a::b", "bandcamp", "miss")
    assert store.release_keys_attempted("bandcamp") == {"a::b"}
    assert store.release_keys_attempted("lastfm") == set()
