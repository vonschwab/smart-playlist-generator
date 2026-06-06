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
