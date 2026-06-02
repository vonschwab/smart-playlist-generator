import numpy as np

from src.features.sonic_rebuild import (
    tower_weighted_from_towers,
    build_tower_weighted_arrays,
)


def test_per_tower_block_norms_equal_sqrt_weight():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 9)).astype(np.float32)
    t = rng.normal(size=(5, 57)).astype(np.float32)
    h = rng.normal(size=(5, 20)).astype(np.float32)
    tw = tower_weighted_from_towers(r, t, h, (0.2, 0.5, 0.3))
    assert tw.shape == (5, 86)
    assert tw.dtype == np.float32
    assert np.allclose(np.linalg.norm(tw[:, 0:9], axis=1), np.sqrt(0.2), atol=1e-5)
    assert np.allclose(np.linalg.norm(tw[:, 9:66], axis=1), np.sqrt(0.5), atol=1e-5)
    assert np.allclose(np.linalg.norm(tw[:, 66:86], axis=1), np.sqrt(0.3), atol=1e-5)


def test_zero_row_does_not_nan():
    r = np.zeros((1, 9), np.float32)
    t = np.ones((1, 57), np.float32)
    h = np.ones((1, 20), np.float32)
    tw = tower_weighted_from_towers(r, t, h, (0.2, 0.5, 0.3))
    assert not np.isnan(tw).any()


def test_build_arrays_overwrites_segments_sets_variant_preserves_genre(tmp_path):
    N = 4
    d = {}
    for seg in ("", "_start", "_mid", "_end"):
        d[f"X_sonic_rhythm{seg}"] = np.full((N, 9), 2.0, np.float32)
        d[f"X_sonic_timbre{seg}"] = np.full((N, 57), 3.0, np.float32)
        d[f"X_sonic_harmony{seg}"] = np.full((N, 20), 4.0, np.float32)
    d["X_sonic"] = np.zeros((N, 86), np.float32)
    for seg in ("start", "mid", "end"):
        d[f"X_sonic_{seg}"] = np.zeros((N, 86), np.float32)
    d["X_sonic_robust_whiten"] = np.zeros((N, 86), np.float32)
    d["X_genre_raw"] = np.ones((N, 3), np.float32)
    d["track_ids"] = np.array(["a", "b", "c", "d"], dtype=object)
    p = tmp_path / "art.npz"
    np.savez(p, **d)
    data = np.load(p, allow_pickle=True)

    out = build_tower_weighted_arrays(data, (0.2, 0.5, 0.3))

    assert str(out["X_sonic_variant"]) == "tower_weighted"
    assert bool(out["X_sonic_pre_scaled"]) is True
    assert out["X_sonic_tower_weighted"].shape == (N, 86)
    # segments are now the weighted vectors (no longer all-zero)
    for seg in ("start", "mid", "end"):
        assert np.linalg.norm(out[f"X_sonic_{seg}"]) > 0
    # raw X_sonic key also updated for fallback consistency
    assert np.array_equal(out["X_sonic"], out["X_sonic_tower_weighted"])
    # genre + ids preserved byte-identical
    assert np.array_equal(out["X_genre_raw"], d["X_genre_raw"])
    assert np.array_equal(out["track_ids"], d["track_ids"])
