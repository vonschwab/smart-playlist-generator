from src.playlist.seed_eligibility import seed_recency_exclusion_for_presence


def test_no_relaxation_when_enough_fresh():
    artist = {"a", "b", "c", "d", "e"}
    excluded = {"a"}                       # 4 fresh >= target 4
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=4)
    assert out == {"a"}


def test_relax_up_to_target_only():
    artist = {"a", "b", "c", "d", "e"}     # 5 total
    excluded = {"a", "b", "c"}             # 2 fresh, target 4 -> re-admit 2
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=4)
    assert len(out) == 1                    # exactly one stays excluded
    assert out < excluded


def test_never_exceed_catalog():
    artist = {"a", "b", "c"}
    excluded = {"a", "b", "c"}             # 0 fresh, target 10 -> re-admit all 3
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=10)
    assert out == set()


def test_readmit_rank_prefers_listed_first():
    artist = {"a", "b", "c", "d"}
    excluded = {"a", "b", "c"}             # target 2, 1 fresh (d) -> re-admit 1
    out = seed_recency_exclusion_for_presence(
        artist, excluded, target_piers=2, readmit_rank=["c", "b", "a"])
    assert "c" not in out                   # highest-ranked re-admitted
    assert out == {"a", "b"}


def test_only_touches_seed_artist_tracks():
    artist = {"a", "b"}
    excluded = {"a", "x", "y"}             # x,y are other artists
    out = seed_recency_exclusion_for_presence(artist, excluded, target_piers=2)
    assert {"x", "y"} <= out                # other artists stay excluded
