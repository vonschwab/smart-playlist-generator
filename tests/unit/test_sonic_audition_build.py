import numpy as np
import pytest
from scripts.sonic_audition_build import (
    compute_spaces,
    find_medoid,
    top_k_for_seed,
    build_seed_manifest,
    _slug,
)


class _Bundle:
    """Minimal bundle-like object for testing."""
    def __init__(self, N=20):
        rng = np.random.default_rng(0)
        self.X_sonic = rng.normal(size=(N, 86)).astype(np.float32)
        self.X_sonic_start = rng.normal(size=(N, 86)).astype(np.float32)
        self.X_sonic_end = rng.normal(size=(N, 86)).astype(np.float32)
        self.track_ids = np.array([f"t{i:03d}" for i in range(N)], dtype=object)
        self.track_artists = np.array(
            ["ArtistA"] * 5 + ["ArtistB"] * 5 + ["ArtistC"] * 10, dtype=object
        )
        self.track_titles = np.array([f"Track{i}" for i in range(N)], dtype=object)


def _make_per_tower(N=20):
    rng = np.random.default_rng(1)
    return {
        "X_sonic_rhythm": rng.normal(size=(N, 9)).astype(np.float32),
        "X_sonic_timbre": rng.normal(size=(N, 57)).astype(np.float32),
        "X_sonic_harmony": rng.normal(size=(N, 20)).astype(np.float32),
    }


def test_compute_spaces_returns_four_spaces():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    assert set(spaces.keys()) == {
        "full_track", "production_transition", "timbre", "harmony"
    }


def test_compute_spaces_query_rows_are_unit_norm():
    b = _Bundle(N=20)
    spaces = compute_spaces(b, _make_per_tower())
    for name, (Xq, _) in spaces.items():
        norms = np.linalg.norm(Xq, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"{name}: query rows not unit-norm"


def test_compute_spaces_transition_uses_end_start():
    b = _Bundle(N=20)
    spaces = compute_spaces(b, _make_per_tower())
    Xq, Xs = spaces["production_transition"]
    # shapes match X_sonic_end and X_sonic_start
    assert Xq.shape == b.X_sonic_end.shape
    assert Xs.shape == b.X_sonic_start.shape


def test_top_k_excludes_same_artist_and_seed():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    artist_a = set(range(5))  # ArtistA = indices 0-4
    neighbors = top_k_for_seed(seed_idx=0, spaces=spaces, exclude_indices=artist_a, k=5)
    for space, pairs in neighbors.items():
        assert len(pairs) == 5, f"{space}: expected 5 neighbors"
        for idx, _ in pairs:
            assert idx not in artist_a, f"{space}: same-artist idx {idx} in neighbors"


def test_build_seed_manifest_blinded_structure():
    b = _Bundle()
    per_tower = _make_per_tower()
    spaces = compute_spaces(b, per_tower)
    file_paths = {f"t{i:03d}": f"/music/t{i}.flac" for i in range(20)}
    manifest = build_seed_manifest("ArtistA", b, spaces, file_paths, k=3)
    assert manifest is not None
    assert manifest["slug"] == "artista"
    # space_data lives at top level, not in individual neighbor entries
    assert "space_data" in manifest
    for n in manifest["neighbors"]:
        assert "spaces" not in n
        assert "track_id" in n
        assert "artist" in n
        assert "file_path" in n


def test_build_seed_manifest_returns_none_for_unknown():
    b = _Bundle()
    spaces = compute_spaces(b, _make_per_tower())
    assert build_seed_manifest("Nobody", b, spaces, {}, k=3) is None


def test_slug():
    assert _slug("Charli XCX") == "charli_xcx"
    assert _slug("J Dilla") == "j_dilla"
    assert _slug("Green-House") == "green_house"
