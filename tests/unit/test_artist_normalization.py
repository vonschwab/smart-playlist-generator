from src.string_utils import normalize_artist_key


def test_normalize_artist_key_diacritics():
    assert normalize_artist_key("Luiz Bonfá") == "luiz bonfa"
    assert normalize_artist_key("Luiz Bonfa") == "luiz bonfa"
    assert normalize_artist_key("João Gilberto") == "joao gilberto"
    assert normalize_artist_key("Antônio Carlos Jobim") == "antonio carlos jobim"


def test_normalize_artist_key_typography():
    assert normalize_artist_key("Guns N’ Roses") == "guns n roses"
    assert normalize_artist_key("Antonio—Carlos Jobim") == "antonio carlos jobim"
    assert normalize_artist_key("D'Angelo") == "d angelo"
