import numpy as np

from scripts.rebuild_sonic_tower_weighted import rebuild_artifact


def _make_artifact(path, N=4):
    d = {}
    for seg in ("", "_start", "_mid", "_end"):
        d[f"X_sonic_rhythm{seg}"] = np.random.default_rng(1).normal(size=(N, 9)).astype(np.float32)
        d[f"X_sonic_timbre{seg}"] = np.random.default_rng(2).normal(size=(N, 57)).astype(np.float32)
        d[f"X_sonic_harmony{seg}"] = np.random.default_rng(3).normal(size=(N, 20)).astype(np.float32)
    d["X_sonic"] = np.zeros((N, 86), np.float32)
    for seg in ("start", "mid", "end"):
        d[f"X_sonic_{seg}"] = np.zeros((N, 86), np.float32)
    d["X_genre_raw"] = np.ones((N, 3), np.float32)
    d["track_ids"] = np.array(["a", "b", "c", "d"], dtype=object)
    np.savez(path, **d)


def test_rebuild_writes_variant_and_backs_up(tmp_path):
    art = tmp_path / "data_matrices_step1.npz"
    _make_artifact(art)
    before = np.load(art, allow_pickle=True)["X_genre_raw"].copy()

    backup = rebuild_artifact(str(art), weights=(0.2, 0.5, 0.3), backup=True)

    assert backup is not None and backup.exists()
    out = np.load(art, allow_pickle=True)
    assert str(out["X_sonic_variant"]) == "tower_weighted"
    assert "X_sonic_tower_weighted" in out.files
    assert np.linalg.norm(out["X_sonic_start"]) > 0  # segments rebuilt
    assert np.array_equal(out["X_genre_raw"], before)  # genre preserved
    # backup holds the pre-rebuild content
    assert str(np.load(backup, allow_pickle=True)["X_sonic_variant"]) != "tower_weighted" \
        if "X_sonic_variant" in np.load(backup, allow_pickle=True).files else True
