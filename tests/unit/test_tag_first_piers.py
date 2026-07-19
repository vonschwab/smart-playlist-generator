import sqlite3

from src.genre.authority import on_tag_track_ids_for_artist


def _db():
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, assignment_layer TEXT);
        INSERT INTO tracks VALUES ('t1','Boards of Canada','alb_h'),
                                  ('t2','Boards of Canada','alb_h'),
                                  ('t3','Boards of Canada','alb_e'),
                                  ('t4','Aphex Twin','alb_a');
        INSERT INTO release_effective_genres VALUES
            ('alb_h','hauntology','observed_leaf'),
            ('alb_h','kosmische','observed_leaf'),
            ('alb_e','electronica','observed_leaf'),
            ('alb_e','hauntology','inferred_family'),  -- inferred: must NOT count
            ('alb_a','hauntology','observed_leaf');
        """
    )
    return c


def test_on_tag_union_and_hitcount():
    c = _db()
    m = on_tag_track_ids_for_artist(c, "boards of canada", {"hauntology", "kosmische"})
    assert m == {"t1": 2, "t2": 2}          # union; both tags hit; t3 inferred-only excluded


def test_on_tag_single_genre_and_case_insensitive_artist():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "BOARDS OF CANADA", {"hauntology"}) == {"t1": 1, "t2": 1}


def test_on_tag_empty_inputs():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "Boards of Canada", set()) == {}
    assert on_tag_track_ids_for_artist(c, "Nobody", {"hauntology"}) == {}


def _authority_db_with_taxonomy(tmp_path):
    import sqlite3
    p = tmp_path / "meta.db"
    c = sqlite3.connect(p)
    c.executescript(
        """
        CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, assignment_layer TEXT);
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        CREATE TABLE genre_graph_aliases (alias TEXT, canonical_genre_id TEXT);
        INSERT INTO tracks VALUES ('t1','Boards of Canada','alb_h'),('t2','Boards of Canada','alb_h');
        INSERT INTO release_effective_genres VALUES ('alb_h','hauntology','observed_leaf');
        INSERT INTO genre_graph_canonical_genres VALUES ('hauntology','hauntology');
        """
    )
    c.commit()
    c.close()
    return str(p)


def test_resolve_artist_on_tag_membership(tmp_path):
    from src.playlist.tag_steering import resolve_artist_on_tag_membership
    dbp = _authority_db_with_taxonomy(tmp_path)
    t2r = {"t1": 5, "t2": 9, "other": 3}
    m = resolve_artist_on_tag_membership(
        ["hauntology"], "Boards of Canada", metadata_db_path=dbp, track_id_to_row=t2r
    )
    assert m == {5: 1, 9: 1}


def test_resolve_artist_on_tag_membership_unmapped(tmp_path):
    from src.playlist.tag_steering import resolve_artist_on_tag_membership
    dbp = _authority_db_with_taxonomy(tmp_path)
    assert resolve_artist_on_tag_membership(
        ["not_a_genre"], "Boards of Canada", metadata_db_path=dbp, track_id_to_row={"t1": 0}
    ) == {}


def test_cluster_restrict_to_track_ids_subsets_members(monkeypatch):
    import numpy as np
    from src.playlist import artist_style
    from src.playlist.artist_style import cluster_artist_tracks, ArtistStyleConfig

    class B:
        track_ids = np.array([f"t{i}" for i in range(12)])
        artist_keys = np.array(["boc"] * 12)
        X_sonic = np.random.default_rng(0).normal(size=(12, 8))
        durations_ms = np.array([200000] * 12)
        track_titles = np.array([f"Title {i}" for i in range(12)])
    monkeypatch.setattr(artist_style, "_artist_indices_in_bundle",
                        lambda bundle, name, include_collaborations=False: list(range(12)))
    keep = {f"t{i}" for i in range(6)}
    cfg = ArtistStyleConfig(pier_bridgeability_enabled=False, dedupe_versions=False)
    clusters, medoids, _mbc, _xn, _support = cluster_artist_tracks(
        bundle=B(), artist_name="boc", cfg=cfg, random_seed=0, restrict_to_track_ids=keep,
    )
    picked = {str(B.track_ids[i]) for c in clusters for i in c}
    assert picked <= keep and picked  # only restricted members clustered


def test_build_members_empty_is_none():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    assert build_tag_first_pier_members(
        {}, np.zeros(10), list(range(10)), target_pier_count=4, cluster_k_min=3, topup_mult=2.0
    ) is None


def test_build_members_large_set_unchanged():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {i: 1 for i in range(18)}          # 18 on-tag >> floor
    M = build_tag_first_pier_members(
        membership, np.zeros(30), list(range(30)),
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert M == set(range(18))


def test_build_members_topup_by_affinity():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {0: 1, 1: 1}                        # 2 members, floor = max(3, ceil(2*4)) = 8
    aff = np.full(20, -9.0)
    aff[0] = aff[1] = 5.0
    aff[7] = 4.0
    aff[3] = 3.0
    aff[5] = 2.0
    aff[9] = 1.0
    aff[2] = 0.5
    aff[8] = 0.4
    M = build_tag_first_pier_members(
        membership, aff, list(range(20)),
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert {0, 1} <= M and len(M) == 8              # topped up to floor
    assert M == {0, 1, 7, 3, 5, 9, 2, 8}            # highest-affinity non-members added


def test_build_members_floor_capped_at_artist_count():
    from src.playlist.tag_steering import build_tag_first_pier_members
    import numpy as np
    membership = {0: 1}
    M = build_tag_first_pier_members(
        membership, np.array([5.0, 4.0, 3.0]), [0, 1, 2],
        target_pier_count=4, cluster_k_min=3, topup_mult=2.0,
    )
    assert M == {0, 1, 2}                            # floor capped at 3 available tracks
