"""Tests for ai_genre_user_overrides storage and resolver integration."""

from __future__ import annotations

import json

from src.ai_genre_enrichment.storage import SidecarStore


def test_user_override_round_trip(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="autechre::amber",
        normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"],
        genres_remove=["warp"],
    )
    override = store.get_user_override("autechre::amber")
    assert override is not None
    assert set(override["genres_add"]) == {"modular synthesizer"}
    assert set(override["genres_remove"]) == {"warp"}


def test_user_override_replaces_on_repeat_set(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["x"], genres_remove=[],
    )
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["y"], genres_remove=["z"],
    )
    override = store.get_user_override("a::b")
    assert set(override["genres_add"]) == {"y"}
    assert set(override["genres_remove"]) == {"z"}


def test_user_override_delete(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["x"], genres_remove=[],
    )
    store.delete_user_override("a::b")
    assert store.get_user_override("a::b") is None


def test_get_user_override_missing_returns_none(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    assert store.get_user_override("never::seen") is None


def test_user_override_casefolds_genres(tmp_path):
    """'IDM' and 'idm' must collapse to 'idm' at write time."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.set_user_override(
        release_key="a::b", normalized_artist="a", normalized_album="b",
        genres_add=["IDM", "idm", "Glitch"],
        genres_remove=["Warp"],
    )
    override = store.get_user_override("a::b")
    assert set(override["genres_add"]) == {"idm", "glitch"}
    assert set(override["genres_remove"]) == {"warp"}


def test_build_enriched_preserves_overrides_in_signature(tmp_path):
    """After rebuild_enriched_genres_for_release, the signature must reflect overrides."""
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    # Use a seedable source (local_metadata): lastfm_tags cannot seed a signature
    # under the current model, so a lastfm-only release produces no signature row
    # for the override to attach to. This test is about override preservation, so
    # it needs a base source that actually seeds.
    page_id = store.upsert_source_page(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber", album_id=None,
        source_url="local://autechre/amber",
        source_type="local_metadata", identity_status="confirmed",
        identity_confidence=1.0, evidence_summary="local",
    )
    store.replace_source_tags(page_id, ["idm", "glitch"])
    store.classify_source_tags(page_id, adjudicate=False)

    store.set_user_override(
        release_key="autechre::amber", normalized_artist="autechre",
        normalized_album="amber",
        genres_add=["modular synthesizer"], genres_remove=["glitch"],
    )

    store.rebuild_enriched_genres_for_release("autechre::amber")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
            ("autechre::amber",),
        ).fetchone()

    payload = json.loads(row["signature_json"])
    genres = set(payload["genres"])
    assert "modular synthesizer" in genres, f"add not preserved: {genres}"
    assert "glitch" not in genres, f"remove not preserved: {genres}"
    assert "idm" in genres
