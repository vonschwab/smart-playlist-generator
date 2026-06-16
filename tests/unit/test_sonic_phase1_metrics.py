import numpy as np

from scripts.research.sonic_phase1_metrics import cosine_spread_to_seed, per_tower_contribution


def test_cosine_spread_keys_and_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 86)).astype(np.float32)
    s = cosine_spread_to_seed(X, seed_idx=0)
    assert set(s) == {"max", "p99", "p90", "median"}
    assert -1.0 <= s["median"] <= 1.0 and s["max"] <= 1.0 + 1e-6


def test_per_tower_contribution_reflects_weights():
    # rows with per-tower norms sqrt(0.2)/sqrt(0.5)/sqrt(0.3) -> contributions 0.2/0.5/0.3
    N = 10
    r = np.full((N, 9), np.sqrt(0.2 / 9), np.float32)
    t = np.full((N, 57), np.sqrt(0.5 / 57), np.float32)
    h = np.full((N, 20), np.sqrt(0.3 / 20), np.float32)
    X = np.concatenate([r, t, h], axis=1)
    c = per_tower_contribution(X)
    assert abs(c["timbre"] - 0.5) < 1e-3
    assert c["timbre"] > c["harmony"] > c["rhythm"]
