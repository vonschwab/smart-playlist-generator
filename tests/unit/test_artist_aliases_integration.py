import types
import numpy as np
from src.playlist.artist_aliases import set_artist_link_map_for_testing


def _ns_bundle(artist_keys, track_artists=None):
    return types.SimpleNamespace(
        artist_keys=np.array(artist_keys, dtype=object),
        track_artists=np.array(track_artists if track_artists is not None else artist_keys, dtype=object),
    )


def test_artist_indices_gathers_alias_members():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0, 1]
    assert _artist_indices_in_bundle(b, "(Sandy) Alex G") == [0, 1]


def test_artist_indices_unlinked_unchanged():
    from src.playlist.artist_style import _artist_indices_in_bundle
    set_artist_link_map_for_testing(None)  # empty
    b = _ns_bundle(["Alex G", "(Sandy) Alex G", "Other Band"])
    assert _artist_indices_in_bundle(b, "Alex G") == [0]


def test_normalize_primary_artist_key_merges_aliases():
    from src.playlist.identity_keys import normalize_primary_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert normalize_primary_artist_key("Alex G") == normalize_primary_artist_key("(Sandy) Alex G")


def test_identity_keys_for_index_merges_aliases():
    from src.playlist.identity_keys import identity_keys_for_index
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    b = types.SimpleNamespace(
        track_ids=np.array(["t0", "t1"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G"], dtype=object),
        track_titles=np.array(["S0", "S1"], dtype=object),
    )
    assert identity_keys_for_index(b, 0).artist_key == identity_keys_for_index(b, 1).artist_key


def test_candidate_pool_normalize_key_merges_aliases():
    from src.playlist.candidate_pool import _normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])
    assert _normalize_artist_key("Alex G") == _normalize_artist_key("(Sandy) Alex G")
    set_artist_link_map_for_testing(None)
    assert _normalize_artist_key("Alex G") != _normalize_artist_key("(Sandy) Alex G")


def test_fire_popularity_merges_alias_catalogs(tmp_path):
    from src.analyze.popularity_runner import init_top_tracks_cache, upsert_artist_top_tracks, load_artist_popularity_values
    from unittest.mock import MagicMock
    from src.string_utils import normalize_artist_key
    set_artist_link_map_for_testing([{"type": "alias", "members": ["Alex G", "(Sandy) Alex G"]}])

    b = types.SimpleNamespace(
        track_ids=np.array(["early", "late", "other"], dtype=object),
        track_titles=np.array(["Early Song", "Late Song", "Nope"], dtype=object),
        artist_keys=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        track_artists=np.array(["Alex G", "(Sandy) Alex G", "Other"], dtype=object),
        durations_ms=None,
    )
    db = str(tmp_path / "pop.db")
    init_top_tracks_cache(db)
    # Each name's catalog has its own Last.fm hit, cached under its own key.
    upsert_artist_top_tracks(db, normalize_artist_key("Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Early Song", "mbid": "early-song-mbid", "rank": 0}])
    upsert_artist_top_tracks(db, normalize_artist_key("(Sandy) Alex G"), "2026-06-24T00:00:00+00:00",
                             [{"name": "Late Song", "mbid": "late-song-mbid", "rank": 0}])
    client = MagicMock()  # fresh cache -> no network
    vec = load_artist_popularity_values(
        b, "Alex G", client=client, db_path=db, limit=50, max_age_days=30,
        now_iso="2026-06-24T00:00:00+00:00")
    assert vec is not None and vec.shape == (3,)
    assert vec[0] == 1.0 and vec[1] == 1.0   # BOTH catalogs' hits landed
    assert np.isnan(vec[2])
    client.get_artist_top_tracks.assert_not_called()


def _artist_positions(result_track_ids, bundle, artist_name):
    idx = {str(t): i for i, t in enumerate(bundle.track_ids)}
    tset = {str(bundle.track_ids[i]) for i in range(len(bundle.track_ids))
            if str(bundle.track_artists[i]) == artist_name}
    return [pos for pos, tid in enumerate(result_track_ids) if str(tid) in tset]


def _make_sibling_pressure_bundle():
    """A bundle engineered so Smog and Bill Callahan are near-identical in sonic
    space (mutual cosine ~0.997) while sitting at the interpolation midpoint
    between the two piers -- i.e. each is the beam's natural top pick for the
    slot immediately after the other. ~16 filler artists (spread evenly along
    the pier_a -> pier_b interpolation, each its own distinct artist) give the
    beam ample slack to complete the segment even when a sibling gets excluded.
    Fully deterministic (fixed seed, non-random pier/sibling vectors) so the
    adjacency pressure -- and hence the regression this guards -- is stable.
    """
    import numpy as np
    from pathlib import Path
    from src.features.artifacts import ArtifactBundle

    def norm(v):
        return v / np.linalg.norm(v)

    dim = 8
    va = np.zeros(dim)
    va[0] = 1.0
    vb = np.zeros(dim)
    vb[1] = 1.0
    mid = norm(va + vb)

    rng = np.random.default_rng(1)
    artists = ["Pier A", "Pier B"]
    vecs = [va, vb]
    titles = ["SeedA", "SeedB"]

    for k, t in enumerate(np.linspace(0.08, 0.92, 16)):
        artists.append(f"Filler_{k}")
        vecs.append(norm((1 - t) * va + t * vb + 0.03 * rng.standard_normal(dim)))
        titles.append(f"F_{k}")

    e1 = np.zeros(dim)
    e1[2] = 1.0
    e2 = np.zeros(dim)
    e2[3] = 1.0
    eps = 0.05
    artists.append("Smog")
    vecs.append(norm(mid + eps * e1))
    titles.append("SmogSong")
    artists.append("Bill Callahan")
    vecs.append(norm(mid + eps * e2))
    titles.append("BillSong")

    n = len(artists)
    return ArtifactBundle(
        artifact_path=Path("sib_test"),
        track_ids=np.array([f"t{i}" for i in range(n)]),
        artist_keys=np.array(artists, dtype=object),
        track_artists=np.array(artists, dtype=object),
        track_titles=np.array(titles, dtype=object),
        X_sonic=np.array(vecs),
        X_sonic_start=None, X_sonic_mid=None, X_sonic_end=None,
        X_genre_raw=np.zeros((n, 4)),
        X_genre_smoothed=np.zeros((n, 4)),
        genre_vocab=np.array([f"g{i}" for i in range(4)]),
        track_id_to_index={f"t{i}": i for i in range(n)},
        durations_ms=np.full(n, 200_000, dtype=np.int64),
    ), n


def test_siblings_never_within_min_gap():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    bundle, n = _make_sibling_pressure_bundle()
    min_gap = 3
    # collapse_segment_pool_by_artist=False matches the live config.yaml default
    # (CLAUDE.md: the dataclass default of True is legacy/debug-only); tail_dp and
    # progress are turned off purely to isolate the beam's own admission gate from
    # unrelated post-processing/progress-gating behavior on this synthetic bundle.
    cfg = PierBridgeConfig(bridge_floor=0.0, transition_floor=0.0, center_transitions=False,
                           variable_bridge_length=False, edge_delete_enabled=False,
                           collapse_segment_pool_by_artist=False, tail_dp_enabled=False,
                           progress_enabled=False)

    # Control: WITHOUT the sibling link, Smog and Bill Callahan's near-identical
    # embeddings land them adjacent -- proving this fixture actually exercises
    # the repulsion gate rather than trivially never colliding.
    set_artist_link_map_for_testing(None)
    baseline = build_pier_bridge_playlist(
        seed_track_ids=["t0", "t1"], total_tracks=14, bundle=bundle,
        candidate_pool_indices=[i for i in range(n) if i not in (0, 1)],
        cfg=cfg, min_gap=min_gap, min_genre_similarity=None, X_genre_smoothed=None,
    )
    base_smog = _artist_positions(baseline.track_ids, bundle, "Smog")
    base_bill = _artist_positions(baseline.track_ids, bundle, "Bill Callahan")
    assert any(
        abs(s - b) < min_gap for s in base_smog for b in base_bill
    ), f"fixture sanity check failed: expected an unlinked adjacency, got {list(baseline.track_ids)}"

    # With the sibling link declared, the same bundle must never place them
    # within min_gap of each other.
    set_artist_link_map_for_testing([{"type": "sibling", "members": ["Smog", "Bill Callahan"]}])
    result = build_pier_bridge_playlist(
        seed_track_ids=["t0", "t1"], total_tracks=14, bundle=bundle,
        candidate_pool_indices=[i for i in range(n) if i not in (0, 1)],
        cfg=cfg, min_gap=min_gap, min_genre_similarity=None, X_genre_smoothed=None,
    )
    smog = _artist_positions(result.track_ids, bundle, "Smog")
    bill = _artist_positions(result.track_ids, bundle, "Bill Callahan")
    for s in smog:
        for b in bill:
            assert abs(s - b) >= min_gap, f"sibling within min_gap: Smog@{s}, Bill@{b} -> {list(result.track_ids)}"
