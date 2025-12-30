from src.string_utils import normalize_artist_key


def test_normalize_artist_key_punctuation_only():
    assert normalize_artist_key("@") == "@"
    assert normalize_artist_key("!!!") == "!!!"
