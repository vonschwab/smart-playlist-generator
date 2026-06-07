from src.ai_genre_enrichment.storage import SidecarStore


def _page_with_tags(store, release_key, artist, album, source_type, tags):
    page_id = store.upsert_source_page(
        release_key=release_key, normalized_artist=artist, normalized_album=album,
        album_id=None, source_url=f"{source_type}://{release_key}/{album}",
        source_type=source_type, identity_status="confirmed",
        identity_confidence=0.9, evidence_summary="t",
    )
    store.replace_source_tags(page_id, tags)


def test_all_collected_tags_returns_release_scoped_rows(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    _page_with_tags(store, "acetone::york blvd", "acetone", "york blvd",
                    "lastfm_tags", ["slowcore", "indie rock"])
    rows = store.all_collected_tags()
    got = {(r["release_key"], r["normalized_tag"]) for r in rows}
    assert ("acetone::york blvd", "slowcore") in got
    assert ("acetone::york blvd", "indie rock") in got
    # carries artist/album for example strings
    sample = next(r for r in rows if r["normalized_tag"] == "slowcore")
    assert sample["normalized_artist"] == "acetone"
    assert sample["normalized_album"] == "york blvd"
