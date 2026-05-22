from src.playlist.title_quality import detect_title_artifacts


def test_detect_demo_in_parenthetical():
    flags = detect_title_artifacts("Lee #2 (8 Track Demo)")
    assert flags == {"demo"}


def test_detect_live_at_venue():
    flags = detect_title_artifacts("All Of You (Take 2 / Live At The Village Vanguard / 1961)")
    assert flags == {"live", "take"}


def test_detect_medley():
    flags = detect_title_artifacts("Rubber Ring/What She Said (Medley)")
    assert flags == {"medley"}


def test_detect_remaster_variants():
    assert detect_title_artifacts("Witchcraft (Remastered Stereo 2025)") == {"remaster", "stereo"}
    assert detect_title_artifacts("Some Day My Prince Will Come - Remastered") == {"remaster"}


def test_detect_instrumental():
    flags = detect_title_artifacts("Song Title (Instrumental)")
    assert flags == {"instrumental"}


def test_clean_title_returns_empty_set():
    assert detect_title_artifacts("Library Pictures") == set()
    assert detect_title_artifacts("Soiled Little Filly") == set()


def test_word_boundaries_only():
    # 'demolish' should NOT match 'demo'
    assert detect_title_artifacts("Demolish the Building") == set()
    # 'alternative' should NOT match 'alternate' substring; only 'alternate take' / parenthetical
    assert detect_title_artifacts("Alternative Rock Anthem") == set()


def test_empty_and_none_inputs():
    assert detect_title_artifacts("") == set()
    assert detect_title_artifacts(None) == set()
