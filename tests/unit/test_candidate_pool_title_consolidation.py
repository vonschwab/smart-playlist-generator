import numpy as np
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(*, hard_exclude_flags=frozenset({"interlude", "skit", "acapella"})):
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=hard_exclude_flags,
    )


def _embedding(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 8))


def test_interlude_track_is_excluded_by_default():
    embedding = _embedding()
    artist_keys = np.array(["a", "b", "c", "d", "e", "f"])
    titles = np.array(["Seed", "Real Track", "Some Interlude", "Another Real", "Live At Venue", "Demo Cut"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted_titles = {titles[int(i)] for i in result.pool_indices}
    assert "Some Interlude" not in admitted_titles
    assert "Live At Venue" in admitted_titles
    assert "Demo Cut" in admitted_titles


def test_user_can_add_medley_to_hard_exclude():
    embedding = _embedding()
    artist_keys = np.array(["a", "b", "c", "d", "e", "f"])
    titles = np.array(["Seed", "Real", "Track (Medley)", "Other", "Fine", "Fine 2"])
    cfg = _make_cfg(hard_exclude_flags=frozenset({"interlude", "skit", "acapella", "medley"}))

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted_titles = {titles[int(i)] for i in result.pool_indices}
    assert "Track (Medley)" not in admitted_titles


def test_acapella_variants_all_excluded():
    embedding = _embedding(n=5)
    artist_keys = np.array(["seed", "b", "c", "d", "e"])
    titles = np.array([
        "Seed",
        "Song (Acapella)",
        "Song (A Cappella)",
        "Song (A Capella)",
        "Normal Song",
    ])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted = {titles[int(i)] for i in result.pool_indices}
    assert "Normal Song" in admitted
    assert "Song (Acapella)" not in admitted
    assert "Song (A Cappella)" not in admitted
    assert "Song (A Capella)" not in admitted


def test_empty_flag_set_admits_everything_titled():
    embedding = _embedding(n=4)
    artist_keys = np.array(["seed", "b", "c", "d"])
    titles = np.array(["Seed", "Interlude", "Skit 1", "Acapella"])
    cfg = _make_cfg(hard_exclude_flags=frozenset())

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_titles=titles,
        cfg=cfg,
        random_seed=0,
    )

    admitted = {titles[int(i)] for i in result.pool_indices}
    assert "Interlude" in admitted
    assert "Skit 1" in admitted
    assert "Acapella" in admitted
