from src.metadata_client import MetadataClient


def test_artist_blacklist_marks_existing_and_new_tracks(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Brian Eno", "Ambient 1", 100)
    metadata.add_track("2", "B", "Brian Eno", "Another Green World", 100)
    metadata.add_track("3", "C", "Roxy Music", "For Your Pleasure", 100)

    updated = metadata.set_artist_blacklisted("Brian Eno", True)

    assert updated == 2
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2"}

    metadata.add_track("4", "D", "brian eno", "Before and After Science", 100)
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2", "4"}


def test_album_blacklist_marks_matching_artist_album_only(tmp_path):
    db_path = tmp_path / "metadata.db"
    metadata = MetadataClient(str(db_path))
    metadata.add_track("1", "A", "Wire", "Chairs Missing", 100)
    metadata.add_track("2", "B", "Wire", "Chairs Missing", 100)
    metadata.add_track("3", "C", "Other Wire", "Chairs Missing", 100)

    updated = metadata.set_album_blacklisted("Wire", "Chairs Missing", True)

    assert updated == 2
    assert metadata.fetch_blacklisted_track_ids() == {"1", "2"}

    metadata.set_album_blacklisted("Wire", "Chairs Missing", False)
    assert metadata.fetch_blacklisted_track_ids() == set()
