from src.playlist_gui.seed_parser import parse_seed_tracks


def test_parse_seed_tracks_empty():
    assert parse_seed_tracks("") == []


def test_parse_seed_tracks_single():
    assert parse_seed_tracks("pink diamond") == ["pink diamond"]


def test_parse_seed_tracks_multiple_commas():
    text = '"pink diamond", "hot girl (bodies bodies bodies)", forever, anthems, "party 4 u"'
    assert parse_seed_tracks(text) == [
        "pink diamond",
        "hot girl (bodies bodies bodies)",
        "forever",
        "anthems",
        "party 4 u",
    ]
