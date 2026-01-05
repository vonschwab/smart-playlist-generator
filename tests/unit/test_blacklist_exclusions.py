from src.playlist_generator import PlaylistGenerator


def test_apply_blacklist_to_ids_filters_allowed_and_excluded():
    gen = PlaylistGenerator.__new__(PlaylistGenerator)
    allowed = ["1", "2", "3"]
    excluded = {"9"}
    blacklist = {"2", "4"}

    new_allowed, new_excluded, removed = gen._apply_blacklist_to_ids(
        allowed_track_ids=allowed,
        excluded_track_ids=excluded,
        blacklist_ids=blacklist,
        context="test",
    )

    assert new_allowed == ["1", "3"]
    assert removed == 1
    assert new_excluded == {"9", "2", "4"}
