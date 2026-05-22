from src.playlist.config import CandidatePoolConfig


def test_candidate_pool_config_has_pace_defaults():
    cfg = CandidatePoolConfig(
        similarity_floor=0.2,
        min_sonic_similarity=0.12,
        max_pool_size=2400,
        target_artists=22,
        candidates_per_artist=6,
        seed_artist_bonus=2,
        max_artist_fraction_final=0.2,
    )

    assert cfg.pace_admission_floor == 0.0
    assert cfg.pace_bridge_floor == 0.0
