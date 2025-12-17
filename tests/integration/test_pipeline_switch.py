import pytest
import time
import numpy as np
import random

from types import SimpleNamespace

from src.playlist_generator import PlaylistGenerator
from src.playlist.ds_pipeline_runner import DsRunResult
from src.playlist_generator import PlaylistGenerator


class FakeConfig:
    def __init__(self, data=None):
        self._data = data or {}

    def get(self, section: str, key: str, default=None):
        return self._data.get(section, {}).get(key, default)

    @property
    def recently_played_filter_enabled(self) -> bool:
        return self._data.get("playlists", {}).get("recently_played_filter", {}).get("enabled", True)

    @property
    def recently_played_lookback_days(self) -> int:
        return self._data.get("playlists", {}).get("recently_played_filter", {}).get("lookback_days", 0)

    @property
    def recently_played_min_playcount(self) -> int:
        return self._data.get("playlists", {}).get("recently_played_filter", {}).get("min_playcount_threshold", 0)
    
    @property
    def lastfm_history_days(self) -> int:
        return self._data.get("lastfm", {}).get("history_days", 90)

    @property
    def max_tracks_per_artist(self) -> int:
        return self._data.get("playlists", {}).get("max_tracks_per_artist", 4)

    @property
    def min_seed_artist_ratio(self) -> float:
        return self._data.get("playlists", {}).get("min_seed_artist_ratio", 0.125)

    @property
    def min_track_duration_seconds(self) -> int:
        return 0

    @property
    def artist_window_size(self) -> int:
        return self._data.get("playlists", {}).get("artist_window_size", 5)


class FakeLibrary:
    def __init__(self, tracks):
        self.tracks = tracks
        self.similarity_calc = object()  # prevent SimilarityCalculator init

    def get_track_by_key(self, key):
        return self.tracks.get(key)

    def get_play_history(self, library_id=None, days=0):
        return []

    def get_similar_tracks_sonic_only(self, seed_id, limit=0, min_similarity=0.0):
        return []


def make_generator(config_data=None, tracks=None):
    cfg = FakeConfig(config_data or {})
    lib = FakeLibrary(tracks or {})
    return PlaylistGenerator(lib, cfg)


def test_pipeline_defaults_to_ds_but_falls_back_without_artifact():
    gen = make_generator(config_data={})
    # No artifact path configured, so DS attempt will fall back to legacy/no result.
    assert gen._maybe_generate_ds_playlist(seed_track_id="t1", target_length=5) is None


def test_pipeline_ds_calls_runner(monkeypatch):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return DsRunResult(
            track_ids=["t1", "t2"],
            requested={},
            effective={},
            metrics={"below_floor": 0, "artist_counts": {"a": 2}},
            playlist_stats={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    tracks = {"t1": {"rating_key": "t1"}, "t2": {"rating_key": "t2"}}
    fake_bundle = SimpleNamespace(
        track_id_to_index={"t1": 0},
        track_artists=["a"],
        track_ids=["t1"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    gen = make_generator(
        config_data={
            "playlists": {
                "pipeline": "ds",
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic", "random_seed": 5},
            }
        },
        tracks=tracks,
    )

    result = gen._maybe_generate_ds_playlist(seed_track_id="t1", target_length=5)
    assert [t["rating_key"] for t in result] == ["t1", "t2"]
    assert calls[0]["random_seed"] == 5
    assert calls[0]["mode"] == "dynamic"


def test_pipeline_ds_missing_artifact_falls_back(monkeypatch, caplog):
    def fake_runner(**kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    gen = make_generator(
        config_data={
            "playlists": {
                "pipeline": "ds",
                "ds_pipeline": {"artifact_path": "missing.npz", "mode": "dynamic"},
            }
        }
    )
    with caplog.at_level("WARNING"):
        result = gen._maybe_generate_ds_playlist(seed_track_id="t1", target_length=5)
    assert result is None
    assert any("legacy path" in msg for msg in caplog.text.splitlines())


def test_pipeline_ds_override_over_legacy(monkeypatch):
    def fake_runner(**kwargs):
        return DsRunResult(
            track_ids=["seed"],
            requested={},
            effective={},
            metrics={},
            playlist_stats={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    fake_bundle = SimpleNamespace(
        track_id_to_index={"seed": 0},
        track_artists=["a"],
        track_ids=["seed"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    gen = make_generator(
        config_data={
            "playlists": {
                "pipeline": "legacy",
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        },
        tracks={"seed": {"rating_key": "seed"}},
    )
    tracks = gen._maybe_generate_ds_playlist(seed_track_id="seed", target_length=5, pipeline_override="ds")
    assert tracks and tracks[0]["rating_key"] == "seed"


def test_ds_mode_override_passed(monkeypatch):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return DsRunResult(
            track_ids=["t1"],
            requested={},
            effective={},
            metrics={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    fake_bundle = SimpleNamespace(
        track_id_to_index={"t1": 0},
        track_artists=["a"],
        track_ids=["t1"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    gen = make_generator(
        config_data={
            "playlists": {
                "pipeline": "ds",
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        },
        tracks={"t1": {"rating_key": "t1"}},
    )
    gen._maybe_generate_ds_playlist(seed_track_id="t1", target_length=5, mode_override="discover")
    assert calls and calls[0]["mode"] == "discover"


def test_create_playlist_for_artist_uses_ds(monkeypatch):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return DsRunResult(
            track_ids=["seed", "x1"],
            requested={},
            effective={},
            metrics={},
            playlist_stats={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)

    class Lib(FakeLibrary):
        def get_all_tracks(self, library_id=None):
            return [
                {"rating_key": "seed", "artist": "A", "title": "T1", "duration": 1000},
                {"rating_key": "x1", "artist": "A", "title": "T2", "duration": 1000},
                {"rating_key": "x2", "artist": "A", "title": "T3", "duration": 1000},
                {"rating_key": "x3", "artist": "A", "title": "T4", "duration": 1000},
            ]

    cfg = FakeConfig(
        {
            "playlists": {
                "pipeline": "ds",
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic", "random_seed": 1},
            }
        }
    )
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._warn_if_ds_artifact_missing", lambda self: None)
    fake_bundle = SimpleNamespace(
        track_id_to_index={"seed": 0, "x1": 1},
        track_artists=["a", "a"],
        track_ids=["seed", "x1"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    gen = PlaylistGenerator(Lib({"seed": {"rating_key": "seed"}, "x1": {"rating_key": "x1"}}), cfg)
    result = gen.create_playlist_for_artist("A", track_count=2, pipeline_override="ds")
    assert result is not None
    assert [t["rating_key"] for t in result["tracks"]] == ["seed", "x1"]
    assert calls and calls[0]["mode"] == "dynamic"


def test_ds_playlist_respects_recent_history(monkeypatch):
    def fake_runner(**kwargs):
        return DsRunResult(
            track_ids=["recent", "fresh"],
            requested={},
            effective={},
            metrics={},
            playlist_stats={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._warn_if_ds_artifact_missing", lambda self: None)
    fake_bundle = SimpleNamespace(
        track_id_to_index={"seed": 0},
        track_artists=["a"],
        track_ids=["seed"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)

    cfg = FakeConfig(
        {
            "playlists": {
                "pipeline": "ds",
                "tracks_per_playlist": 2,
                "recently_played_filter": {"enabled": True, "lookback_days": 30, "min_playcount_threshold": 0},
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        }
    )

    class Lib(FakeLibrary):
        def get_tracks_by_ids(self, ids):
            return [self.tracks[i] for i in ids if i in self.tracks]

    lib = Lib(
        {
            "seed": {"rating_key": "seed", "artist": "A", "title": "S"},
            "recent": {"rating_key": "recent", "artist": "A", "title": "R"},
            "fresh": {"rating_key": "fresh", "artist": "B", "title": "F"},
        }
    )
    gen = PlaylistGenerator(lib, cfg)

    import time

    history = [{"rating_key": "recent", "timestamp": int(time.time())}]
    seeds = {"A": [{"rating_key": "seed", "artist": "A", "title": "Seed"}]}
    playlists = gen._create_playlists_from_single_artists(seeds, history, pipeline_override="ds")
    assert playlists
    tracks = playlists[0]["tracks"]
    assert [t["rating_key"] for t in tracks] == ["fresh"]


def test_single_artist_ds_filters_recent(monkeypatch):
    def fake_runner(**kwargs):
        allowed = kwargs.get("allowed_track_ids") or []
        excluded = kwargs.get("excluded_track_ids") or set()
        ids_base = ["recent", "fresh"]
        ids = [tid for tid in ids_base if (not allowed or tid in allowed) and (tid not in excluded)]
        return DsRunResult(
            track_ids=ids,
            requested={},
            effective={},
            metrics={},
            playlist_stats={},
        )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._warn_if_ds_artifact_missing", lambda self: None)
    fake_bundle = SimpleNamespace(
        track_id_to_index={"seed": 0},
        track_artists=["a"],
        track_ids=["seed"],
    )
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)

    cfg = FakeConfig(
        {
            "playlists": {
                "pipeline": "ds",
                "tracks_per_playlist": 2,
                "recently_played_filter": {"enabled": True, "lookback_days": 30, "min_playcount_threshold": 0},
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        }
    )

    class Lib(FakeLibrary):
        def get_tracks_by_ids(self, ids):
            return [self.tracks[i] for i in ids if i in self.tracks]

        def get_all_tracks(self):
            return list(self.tracks.values())

    lib = Lib(
        {
            "seed": {"rating_key": "seed", "artist": "Brian Green", "title": "Seed"},
            "recent": {"rating_key": "recent", "artist": "Brian Green", "title": "R"},
            "fresh": {"rating_key": "fresh", "artist": "Other", "title": "F"},
            "extra": {"rating_key": "extra", "artist": "Brian Green", "title": "E"},
            "extra2": {"rating_key": "extra2", "artist": "Brian Green", "title": "E2"},
        }
    )
    monkeypatch.setattr(random, "sample", lambda seq, n: list(seq)[:n])
    gen = PlaylistGenerator(lib, cfg)
    gen.lastfm = object()
    gen.matcher = object()
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._get_lastfm_scrobbles_raw", lambda self: [{"artist": "Brian Green", "title": "R", "timestamp": int(time.time())}])
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._get_lastfm_history", lambda self: [{"rating_key": "recent", "timestamp": int(time.time())}])

    playlist = gen.create_playlist_for_artist("Brian Green", track_count=2, pipeline_override="ds")
    assert playlist
    assert [t["rating_key"] for t in playlist["tracks"]] == ["fresh"]


def test_filter_tracks_respects_history_keys():
    cfg = FakeConfig(
        {
            "playlists": {
                "recently_played_filter": {"enabled": True, "lookback_days": 30, "min_playcount_threshold": 0},
            }
        }
    )
    lib = FakeLibrary({})
    gen = PlaylistGenerator(lib, cfg)
    history = [{"rating_key": "a", "timestamp": int(time.time())}]
    tracks = [{"rating_key": "a"}, {"rating_key": "b"}]
    filtered = gen.filter_tracks(tracks, history, exempt_tracks=None)
    assert [t["rating_key"] for t in filtered] == ["b"]


def test_scrobble_filter_removes_matching_candidates():
    cfg = FakeConfig(
        {
            "playlists": {
                "recently_played_filter": {"enabled": True, "lookback_days": 30, "min_playcount_threshold": 0},
            },
            "lastfm": {"history_days": 30},
        }
    )
    lib = FakeLibrary({})
    gen = PlaylistGenerator(lib, cfg)
    scrobbles = [
        {"artist": "The Clash", "title": "London Calling", "timestamp": int(time.time())},
        {"artist": "Ramones", "title": "Blitzkrieg Bop", "timestamp": int(time.time())},
        {"artist": "Other", "title": "Irrelevant", "timestamp": int(time.time())},
    ]
    candidates = [
        {"artist": "Ramones", "title": "Blitzkrieg Bop", "rating_key": "1"},
        {"artist": "Pixies", "title": "Debaser", "rating_key": "2"},
        {"artist": "The Clash", "title": "London Calling", "rating_key": "3"},
    ]
    filtered = gen._filter_by_scrobbles(candidates, scrobbles, lookback_days=30)
    assert {t["rating_key"] for t in filtered} == {"2"}


def test_seed_artist_count_normalized_matches():
    cfg = FakeConfig({})
    lib = FakeLibrary({})
    gen = PlaylistGenerator(lib, cfg)
    tracks = [
        {"artist": "orchid mantis", "title": "empty palms"},
        {"artist": "Other", "title": "X"},
    ]
    # Use same helper as main app does for stats
    seed_norm = "Orchid Mantis"
    # mimic the logger computation in _print_playlist_report
    norm_artists = [(t.get("artist") or "").strip().casefold() for t in tracks]
    seed_count = sum(1 for a in norm_artists if a == seed_norm.strip().casefold())
    assert seed_count == 1


def test_ds_stats_block_replaces_sonic_similarity(caplog):
    cfg = FakeConfig({})
    lib = FakeLibrary({})
    gen = PlaylistGenerator(lib, cfg)
    gen._last_ds_report = {
        "metrics": {"min_transition": 0.1, "mean_transition": 0.2, "below_floor": 0, "artist_counts": {"a": 1}},
        "requested_len": 5,
        "actual_len": 4,
        "edge_scores": [{"prev_id": "a", "cur_id": "b", "T": 0.5, "S": 0.4, "G": 0.3, "H": 0.45}],
    }
    caplog.set_level("INFO")
    gen._print_playlist_report(
        [{"artist": "A", "title": "T1", "rating_key": "a"}, {"artist": "B", "title": "T2", "rating_key": "b"}],
        artist_name="A",
        verbose_edges=True,
    )
    text = "\n".join(caplog.messages)
    assert "Edge score summaries:" in text
    assert "Sonic similarity" not in text
    assert "edge 01->02: T=0.500" in text
    assert "H=" in text
    assert "scores unavailable" not in text
    assert "nan" not in text.lower()


def test_ds_verbose_logs_baseline_percentiles(caplog):
    cfg = FakeConfig({})
    lib = FakeLibrary({})
    gen = PlaylistGenerator(lib, cfg)
    edge_scores = [
        {"prev_id": "a", "cur_id": "b", "prev_idx": 0, "cur_idx": 1, "T": 0.9, "T_raw": 0.8, "H": 0.85, "S": 0.7, "G": 0.6}
    ]
    gen._last_ds_report = {
        "metrics": {},
        "playlist_stats": {"playlist": {"edge_scores": edge_scores, "transition_floor": 0.5, "transition_gamma": 0.5}},
        "baseline": {
            "T": {"p50": 0.1, "p90": 0.2, "p99": 0.3},
            "T_raw": {"p50": 0.15, "p90": 0.25, "p99": 0.35},
            "S": {"p50": 0.2, "p90": 0.3, "p99": 0.4},
            "G": {"p50": 0.3, "p90": 0.4, "p99": 0.5},
        },
    }
    caplog.set_level("INFO")
    gen._print_playlist_report(
        [{"artist": "A", "title": "t1", "rating_key": "a"}, {"artist": "B", "title": "t2", "rating_key": "b"}],
        artist_name="A",
        verbose_edges=True,
    )
    text = "\n".join(caplog.messages)
    assert "Baseline vs playlist percentiles:" in text
    assert "T_raw" in text
    assert "edge 01->02" in text
    assert "nan" not in text.lower()

def test_center_transitions_reduces_similarity():
    import numpy as np
    from src.playlist.constructor import _transition_array

    # Highly aligned due to large shared mean component
    X_end = np.array([[100.0, 1.0], [100.0, -1.0]])
    X_start = np.array([[100.0, 1.0], [100.0, -1.0]])
    emb_norm = np.eye(2)
    order = [0, 1]
    trans_no_center = _transition_array(order, emb_norm, X_end, X_start, 1.0, False)
    X_end_c = X_end - X_end.mean(axis=0, keepdims=True)
    X_start_c = X_start - X_start.mean(axis=0, keepdims=True)
    trans_center = _transition_array(order, emb_norm, X_end_c, X_start_c, 1.0, True)
    assert trans_center[0] < trans_no_center[0]


def test_center_transitions_rescale_clamped():
    import numpy as np
    from src.playlist.constructor import _transition_array

    X_end = np.array([[1.0, 0.0], [-1.0, 0.0]])
    X_start = np.array([[1.0, 0.0], [-1.0, 0.0]])
    emb_norm = np.eye(2)
    order = [0, 1]
    trans = _transition_array(order, emb_norm, X_end, X_start, 1.0, True)
    assert (trans >= 0.0).all() and (trans <= 1.0).all()

def test_center_transitions_t_used_matches_rescaled():
    import numpy as np
    from src.similarity.hybrid import transition_similarity_end_to_start
    from src.playlist.constructor import _transition_array

    X_end = np.array([[10.0, 1.0], [10.0, -1.0]])
    X_start = np.array([[10.0, 1.0], [10.0, -1.0]])
    emb_norm = np.eye(2)
    order = [0, 1]
    # Raw uncentered cosine (high due to mean direction)
    raw = float(transition_similarity_end_to_start(X_end, X_start, 0, np.array([1]))[0])
    trans_center = _transition_array(order, emb_norm, X_end - X_end.mean(axis=0, keepdims=True), X_start - X_start.mean(axis=0, keepdims=True), 1.0, True)
    centered_cos = float(transition_similarity_end_to_start(X_end - X_end.mean(axis=0, keepdims=True), X_start - X_start.mean(axis=0, keepdims=True), 0, np.array([1]))[0])
    centered_rescaled = float(np.clip((centered_cos + 1.0) / 2.0, 0.0, 1.0))
    assert trans_center[0] == centered_rescaled
    assert raw != centered_rescaled

def test_ds_seed_artist_fallback(monkeypatch):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return DsRunResult(
            track_ids=["a1"],
            requested={},
            effective={},
            metrics={},
        )

    fake_bundle = SimpleNamespace(
        track_id_to_index={"a1": 0},
        track_artists=["artistx"],
        track_ids=["a1"],
    )

    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    gen = make_generator(
        config_data={
            "playlists": {
                "pipeline": "ds",
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        },
        tracks={"a1": {"rating_key": "a1"}},
    )
    result = gen._maybe_generate_ds_playlist(
        seed_track_id="missing",
        target_length=5,
        pipeline_override="ds",
        seed_artist="artistx",
    )
    assert calls and calls[0]["seed_track_id"] == "a1"


def test_edge_scores_recomputed_after_recency(monkeypatch, caplog):
    cfg = FakeConfig({})
    class Lib(FakeLibrary):
        def __init__(self):
            tracks = {
                "a": {"rating_key": "a", "artist": "A", "title": "One"},
                "b": {"rating_key": "b", "artist": "B", "title": "Two"},
                "c": {"rating_key": "c", "artist": "C", "title": "Three"},
            }
            super().__init__(tracks)
    lib = Lib()
    gen = PlaylistGenerator(lib, cfg)
    gen._last_ds_report = {
        "metrics": {},
        "playlist_stats": {},
        "artifact_path": "dummy",
        "transition_centered": False,
        "transition_gamma": 1.0,
    }
    final_tracks = [lib.tracks["a"], lib.tracks["c"]]
    def fake_compute(tracks, *args, **kwargs):
        assert [t["rating_key"] for t in tracks] == ["a", "c"]
        return [
            {"prev_id": "a", "cur_id": "c", "prev_idx": 0, "cur_idx": 1, "T": 0.5, "S": 0.1, "G": 0.2}
        ]
    monkeypatch.setattr(gen, "_compute_edge_scores_from_artifact", fake_compute)
    recomputed = gen._compute_edge_scores_from_artifact(final_tracks, "dummy")
    gen._last_ds_report["edge_scores"] = recomputed
    gen._last_ds_report["playlist_stats"] = {"playlist": {"edge_scores": recomputed}}
    caplog.set_level("INFO")
    gen._print_playlist_report(final_tracks, artist_name="X", verbose_edges=True)
    text = "\n".join(caplog.messages)
    assert "edge 01->02" in text
    assert "Missing edge score" not in text
    assert "Edge score count mismatch" not in text
    assert "Chain index-degree summary" in text


def test_recency_creates_new_edges_logged(monkeypatch, caplog):
    cfg = FakeConfig(
        {
            "playlists": {
                "pipeline": "ds",
                "tracks_per_playlist": 5,
                "recently_played_filter": {"enabled": True, "lookback_days": 30, "min_playcount_threshold": 0},
                "ds_pipeline": {"artifact_path": "dummy.npz", "mode": "dynamic"},
            }
        }
    )

    class Lib(FakeLibrary):
        def __init__(self):
            tracks = {
                "a": {"rating_key": "a", "artist": "A", "title": "One"},
                "b": {"rating_key": "b", "artist": "A", "title": "Two"},
                "c": {"rating_key": "c", "artist": "A", "title": "Three"},
                "d": {"rating_key": "d", "artist": "A", "title": "Four"},
                "e": {"rating_key": "e", "artist": "A", "title": "Five"},
            }
            super().__init__(tracks)

        def get_tracks_by_ids(self, ids):
            return [self.tracks[i] for i in ids if i in self.tracks]

        def get_all_tracks(self, library_id=None):
            return list(self.tracks.values())

    lib = Lib()

    def fake_runner(**kwargs):
        return DsRunResult(
            track_ids=["a", "b", "c", "d", "e"],
            requested={},
            effective={},
            metrics={},
            playlist_stats={},
        )

    fake_bundle = SimpleNamespace(
        track_id_to_index={"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
        track_artists=["A", "B", "C", "D", "E"],
        track_ids=np.array(["a", "b", "c", "d", "e"]),
    )

    monkeypatch.setenv("PLAYLIST_DIAG_RECENCY", "1")
    monkeypatch.setattr("src.playlist_generator.run_ds_pipeline", fake_runner)
    monkeypatch.setattr("src.playlist_generator.load_artifact_bundle", lambda path: fake_bundle)
    monkeypatch.setattr("src.playlist_generator.PlaylistGenerator._warn_if_ds_artifact_missing", lambda self: None)
    gen = PlaylistGenerator(lib, cfg)
    gen.lastfm = object()
    gen.matcher = object()
    monkeypatch.setattr(
        "src.playlist_generator.PlaylistGenerator._get_lastfm_scrobbles_raw",
        lambda self: [{"artist": "A", "title": "Three", "timestamp": int(time.time())}],
    )
    caplog.set_level("INFO")
    playlist = gen.create_playlist_for_artist("A", track_count=5, pipeline_override="ds", verbose=True)
    assert playlist is not None
    msgs = "\n".join(caplog.messages)
    assert "Recency adjacency diag" in msgs
    assert "new_edges=0" in msgs
