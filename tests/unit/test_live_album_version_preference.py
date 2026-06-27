"""Live-album version-preference (Unwound 'Live Leaves' bug).

A clean track title on a live album ('Corpse Pose' on 'Live Leaves') must lose to the
studio version, in BOTH the version-preference scorer AND the Last.fm->local resolver.
"""

from src.title_dedupe import calculate_version_preference_score
from src.analyze.popularity_runner import resolve_top_tracks_to_rank


def test_leading_live_word_album_is_penalized():
    # "Live Leaves" is a live album but matches none of the phrase markers
    # ("live at", "(live", "unplugged", ...); a standalone "live" word must catch it.
    studio = calculate_version_preference_score("Corpse Pose", "New Plastic Ideas")
    live = calculate_version_preference_score("Corpse Pose", "Live Leaves")
    assert live < studio


def test_alive_is_not_a_false_positive():
    # word-boundary "live" must NOT fire on substrings like "Alive"/"Olive".
    assert calculate_version_preference_score("Song", "Still Alive") == \
        calculate_version_preference_score("Song", "Studio Record")


def test_resolver_prefers_studio_over_live_album_on_clean_title():
    # Last.fm #1 "Corpse Pose" maps to the STUDIO local file, not the live one,
    # even when the live file's track_id would win an album-blind tie-break.
    top = [{"name": "Corpse Pose", "rank": 0, "mbid": ""}]
    local = [
        {"track_id": "a_studio", "title": "Corpse Pose", "musicbrainz_id": "", "album": "New Plastic Ideas"},
        {"track_id": "z_live", "title": "Corpse Pose", "musicbrainz_id": "", "album": "Live Leaves"},
    ]
    ranks = resolve_top_tracks_to_rank(top, local)
    assert ranks.get("a_studio") == 0
    assert "z_live" not in ranks


def test_popularity_lands_on_the_dedup_canonical_studio_version(monkeypatch):
    # Two studio "Corpse Pose" copies (No Energy / Repetition) both score 100. The dedup keeps
    # the LONGER one (idx 0), but an album-blind resolver picks the max-track_id one (idx 1).
    # The popularity score MUST land on the same version the dedup/piers keep — otherwise the
    # #1 hit carries no score on the surviving version and vanishes from the 🔥 piers.
    import numpy as np
    from types import SimpleNamespace
    import src.analyze.popularity_runner as pr

    bundle = SimpleNamespace(
        track_ids=np.array(["aaa_long", "zzz_short"], dtype=object),
        track_titles=np.array(["Corpse Pose", "Corpse Pose"], dtype=object),
        artist_keys=np.array(["unwound", "unwound"], dtype=object),
        durations_ms=np.array([300000.0, 200000.0], dtype=float),
    )
    monkeypatch.setattr(
        pr, "get_artist_top_tracks_cached_or_fetch",
        lambda *a, **k: [{"name": "Corpse Pose", "rank": 0, "mbid": ""}],
    )
    out = pr.load_artist_popularity_values(
        bundle, "Unwound", client=object(), db_path=":mem:",
        limit=50, max_age_days=30, now_iso="2026-01-01T00:00:00", metadata_db_path=None,
    )
    assert out is not None
    # score lands on the dedup-canonical (idx 0, the longer cut), NOT the max-track_id one (idx 1)
    assert np.isfinite(out[0]) and np.isnan(out[1])
