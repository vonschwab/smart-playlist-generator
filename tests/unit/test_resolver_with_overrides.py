"""Test EnrichedGenreResolver applies user overrides on read."""

from __future__ import annotations

import json

from src.ai_genre_enrichment.storage import SidecarStore
from src.ai_genre_enrichment.genre_resolver import EnrichedGenreResolver


def _seed_signature(store, release_key, normalized_artist, normalized_album, genres):
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (release_key, normalized_artist, normalized_album, None,
             json.dumps({"genres": genres, "sources": []}), "2026-05-28"),
        )
        conn.commit()


def test_resolver_applies_override_add(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber", ["idm", "glitch"])
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", genres_add=["modular synthesizer"], genres_remove=[],
    )
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    genres = resolver.get_enriched_genres(artist="Autechre", album="Amber")
    assert set(genres) == {"idm", "glitch", "modular synthesizer"}


def test_resolver_applies_override_remove(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber",
                    ["idm", "glitch", "warp"])
    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", genres_add=[], genres_remove=["warp"],
    )
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    genres = resolver.get_enriched_genres(artist="Autechre", album="Amber")
    assert set(genres) == {"idm", "glitch"}


def test_resolver_works_without_override(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "autechre::amber", "autechre", "amber", ["idm"])
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    assert set(resolver.get_enriched_genres(artist="Autechre", album="Amber")) == {"idm"}


def test_override_without_signature_returns_add_as_genres(tmp_path):
    """Edge case: user creates an override for an unenriched release — treat add list as signature."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="rare::release", normalized_artist="rare",
        normalized_album="release", genres_add=["field recordings"], genres_remove=[],
    )
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    assert resolver.get_enriched_genres(artist="Rare", album="Release") == ["field recordings"]


def test_resolver_remove_casefold_matches_mixed_case_sig(tmp_path):
    """Remove set (stored casefolded) must match sig genres of mixed case."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _seed_signature(store, "a::b", "a", "b", ["IDM", "Glitch"])
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=[], genres_remove=["glitch"],
    )
    resolver = EnrichedGenreResolver(str(tmp_path / "sidecar.db"))
    genres = resolver.get_enriched_genres(artist="A", album="B")
    genres_lower = {g.casefold() for g in genres}
    assert "glitch" not in genres_lower
    assert "idm" in genres_lower
