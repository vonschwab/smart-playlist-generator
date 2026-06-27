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
