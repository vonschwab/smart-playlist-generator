from src.analyze.popularity_runner import resolve_top_tracks_to_popularity


def test_resolver_prefers_studio_over_live_and_honors_remaster_and_mbid():
    # Last.fm top tracks (canonical names), ranked
    top = [
        {"name": "Smells Like Teen Spirit", "mbid": "mbid-slts", "rank": 0},
        {"name": "In Bloom", "mbid": "", "rank": 1},
        {"name": "Come as You Are", "mbid": "", "rank": 2},
    ]
    local = [
        # SLTS only as a remaster -> must still match + carry top popularity
        {"track_id": "t_slts_rem", "title": "Smells Like Teen Spirit (2021 Remaster)", "musicbrainz_id": "mbid-slts"},
        # In Bloom: studio + live -> studio must win
        {"track_id": "t_inbloom_studio", "title": "In Bloom", "musicbrainz_id": ""},
        {"track_id": "t_inbloom_live", "title": "In Bloom (Live In Seattle)", "musicbrainz_id": ""},
        # Come as You Are: only studio
        {"track_id": "t_caya", "title": "Come as You Are", "musicbrainz_id": ""},
        # an unrelated deep cut -> no popularity
        {"track_id": "t_deep", "title": "Endless, Nameless", "musicbrainz_id": ""},
    ]
    pop = resolve_top_tracks_to_popularity(top, local)
    assert pop["t_slts_rem"] == 1.0                  # rank 0, matched via mbid; remaster NOT penalized
    assert "t_inbloom_studio" in pop                  # studio In Bloom got the popularity
    assert "t_inbloom_live" not in pop                # live lost to studio
    assert pop["t_inbloom_studio"] > pop["t_caya"]    # rank 1 > rank 2
    assert "t_deep" not in pop                         # unmatched deep cut neutral


def test_resolver_keeps_higher_score_on_collision_and_handles_empty():
    top = [{"name": "Song", "mbid": "", "rank": 0}, {"name": "song", "mbid": "", "rank": 1}]
    local = [{"track_id": "t1", "title": "Song", "musicbrainz_id": ""}]
    pop = resolve_top_tracks_to_popularity(top, local)
    assert pop["t1"] == 1.0   # both ranks map to t1; keep the higher (rank 0)
    assert resolve_top_tracks_to_popularity([], local) == {}


def test_resolver_remaster_only_via_title_path_carries_popularity():
    # Remaster with NO mbid -> must still match the canonical hit via loose-title
    # normalization (which strips "(... Remaster)") and carry full popularity.
    top = [{"name": "Heart-Shaped Box", "mbid": "", "rank": 0}]
    local = [
        {"track_id": "t_hsb_remaster", "title": "Heart-Shaped Box (2021 Remaster)", "musicbrainz_id": ""},
        {"track_id": "t_other", "title": "Milk It", "musicbrainz_id": ""},
    ]
    pop = resolve_top_tracks_to_popularity(top, local)
    assert pop.get("t_hsb_remaster") == 1.0   # remaster matched via title, NOT penalized
    assert "t_other" not in pop


def test_resolver_empty_local_returns_empty():
    assert resolve_top_tracks_to_popularity([{"name": "X", "mbid": "", "rank": 0}], []) == {}
