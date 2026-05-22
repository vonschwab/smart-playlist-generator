from src.playlist.title_quality import detect_title_artifacts


def test_interlude_detected_with_word_boundary():
    assert "interlude" in detect_title_artifacts("Interlude")
    assert "interlude" in detect_title_artifacts("Track Interlude")
    assert "interlude" in detect_title_artifacts("Side B Interlude")
    assert detect_title_artifacts("Interludial Phase") == set()


def test_skit_detected_with_word_boundary():
    assert "skit" in detect_title_artifacts("Skit")
    assert "skit" in detect_title_artifacts("Skit 1")
    assert "skit" in detect_title_artifacts("Intro Skit")
    assert detect_title_artifacts("Skitter Like Light") == set()
    assert detect_title_artifacts("Skittles") == set()


def test_acapella_detected_with_spelling_variants():
    assert "acapella" in detect_title_artifacts("Song (Acapella)")
    assert "acapella" in detect_title_artifacts("Song (A Cappella)")
    assert "acapella" in detect_title_artifacts("Song (A Capella)")
    assert "acapella" in detect_title_artifacts("Acapella Version")
    assert detect_title_artifacts("Capella Star System") == set()


def test_new_flags_do_not_collide_with_existing_flags():
    assert "demo" in detect_title_artifacts("Lee #2 (8 Track Demo)")
    assert "live" in detect_title_artifacts("All Of You (Live At The Village Vanguard)")
    assert "medley" in detect_title_artifacts("Rubber Ring/What She Said (Medley)")
