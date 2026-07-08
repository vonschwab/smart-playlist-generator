"""Tag-name -> dense steering target resolution."""
import logging

import numpy as np

from src.playlist.tag_steering import resolve_tag_steering_target, sonic_prototype_from_rows, sonic_global_mean

VOCAB = ["jazz-funk", "post-bop", "soul"]
EMB = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_maps_case_insensitively_and_normalizes():
    target, mapped, unmapped = resolve_tag_steering_target(
        ["Jazz-Funk", " post-bop "], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert mapped == ["Jazz-Funk", "post-bop"]  # resolver stores stripped inputs
    assert unmapped == []
    np.testing.assert_allclose(target, np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5]))
    assert abs(np.linalg.norm(target) - 1.0) < 1e-9


def test_unmapped_tags_warn_and_are_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk", "vaporwave"], genre_vocab=VOCAB, genre_emb=EMB
        )
    assert unmapped == ["vaporwave"]
    assert target is not None
    assert any("not in the artifact genre vocabulary" in r.message for r in caplog.records)


def test_missing_genre_emb_warns_and_disables(caplog):
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["jazz-funk"], genre_vocab=VOCAB, genre_emb=None
        )
    assert target is None and mapped == [] and unmapped == ["jazz-funk"]
    assert any("genre_emb" in r.message for r in caplog.records)


def test_no_tags_is_silent_none():
    target, mapped, unmapped = resolve_tag_steering_target(
        [], genre_vocab=VOCAB, genre_emb=EMB
    )
    assert target is None and mapped == [] and unmapped == []


def test_no_tags_emits_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_tag_steering_target([], genre_vocab=VOCAB, genre_emb=EMB)
    assert caplog.records == []


def test_degenerate_zero_norm_target_returns_none_and_warns(caplog):
    # Two selected tags whose embedding rows cancel to a ~zero mean vector.
    vocab = ["cancel-a", "cancel-b"]
    emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
    with caplog.at_level(logging.WARNING):
        target, mapped, unmapped = resolve_tag_steering_target(
            ["cancel-a", "cancel-b"], genre_vocab=vocab, genre_emb=emb
        )
    assert target is None
    assert mapped == ["cancel-a", "cancel-b"]
    assert unmapped == []
    assert any(
        "degenerate" in r.message.lower() or "zero-norm" in r.message.lower()
        for r in caplog.records
    )


def test_sonic_prototype_points_at_member_mean_direction():
    A = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1)) + 1e-3
    B = np.tile(np.array([0.0, 1.0, 0.0]), (10, 1)) + 1e-3
    M = np.vstack([A, B])
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(10)))
    assert n == 10
    assert cohesion > 0.9
    assert proto @ np.array([1.0, 0.0, 0.0]) > 0.8
    assert proto @ np.array([0.0, 1.0, 0.0]) < 0.2


def test_sonic_prototype_low_cohesion_for_scattered_rows():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((50, 16))
    proto, cohesion, n = sonic_prototype_from_rows(M, list(range(50)))
    assert proto is not None
    assert cohesion < 0.5


def test_sonic_prototype_empty_rows_returns_none():
    M = np.eye(4)
    proto, cohesion, n = sonic_prototype_from_rows(M, [])
    assert proto is None and n == 0


def test_global_mean_centering_subtracts_common_component():
    common = np.array([5.0, 0.0, 0.0])
    M = np.vstack([common + np.array([0, 1.0, 0]), common + np.array([0, 0, 1.0])])
    gm = sonic_global_mean(M)
    proto_centered, _, _ = sonic_prototype_from_rows(M, [0], global_mean=gm)
    assert abs(proto_centered[0]) < abs(proto_centered[1]) + abs(proto_centered[2])


import sqlite3
from src.playlist.tag_steering import resolve_tag_sonic_prototype_rows


def _make_authority_db(tmp_path, name="m.db", artists=None, layer="observed_leaf"):
    """Build the PUBLISHED-AUTHORITY schema the resolver reads: tracks (with album_id),
    release_effective_genres (album_id -> canonical genre_id + layer), and the canonical
    name->id map. Each track ti lives on its own album alb{i} tagged 'jangle_pop'.
    `artists[i]` gives ti's artist (default: t0-t2 'Seed', rest 'Other')."""
    if artists is None:
        artists = ["Seed" if i < 3 else "Other" for i in range(40)]
    db = tmp_path / name
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT)")
    con.execute("CREATE TABLE release_effective_genres "
                "(album_id TEXT, genre_id TEXT, assignment_layer TEXT)")
    con.execute("CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT)")
    con.execute("CREATE TABLE genre_graph_aliases (alias TEXT, canonical_genre_id TEXT)")
    con.executemany("INSERT INTO tracks VALUES (?,?,?)",
                    [(f"t{i}", artists[i], f"alb{i}") for i in range(len(artists))])
    con.executemany("INSERT INTO release_effective_genres VALUES (?,?,?)",
                    [(f"alb{i}", "jangle_pop", layer) for i in range(len(artists))])
    con.execute("INSERT INTO genre_graph_canonical_genres VALUES ('jangle_pop','jangle pop')")
    con.commit()
    con.close()
    return str(db)


def test_resolver_returns_rows_excluding_seed_artist(tmp_path):
    db = _make_authority_db(tmp_path)
    t2r = {f"t{i}": i for i in range(40)}
    rows, n, used = resolve_tag_sonic_prototype_rows(
        ["Jangle Pop"], metadata_db_path=db, track_id_to_row=t2r,
        exclude_artist="Seed", min_support=25)
    assert n == 37                       # 40 tagged minus 3 Seed tracks
    assert all(r >= 3 for r in rows)     # t0..t2 excluded
    assert used == ["Jangle Pop"]


def test_resolver_excludes_multiple_seed_artists(tmp_path):
    arts = ["Seed" if i < 3 else "Second" if i < 6 else "Other" for i in range(40)]
    db = _make_authority_db(tmp_path, name="m2.db", artists=arts)
    t2r = {f"t{i}": i for i in range(40)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        ["jangle pop"], metadata_db_path=db, track_id_to_row=t2r,
        exclude_artists=["Seed", "Second"], min_support=25)
    assert n == 34                       # 40 minus 3 Seed minus 3 Second
    assert all(r >= 6 for r in rows)     # t0..t5 excluded


def test_resolver_ignores_inferred_layer(tmp_path):
    # A release tagged the genre only via graph inference must NOT count (matches the
    # authority's chip semantics: observed_leaf/legacy only). This is the fix for the
    # 2026-07-08 track_effective_genres miswiring.
    db = _make_authority_db(tmp_path, name="inf.db", layer="inferred_family")
    t2r = {f"t{i}": i for i in range(40)}
    rows, n, _ = resolve_tag_sonic_prototype_rows(
        ["jangle pop"], metadata_db_path=db, track_id_to_row=t2r, min_support=1)
    assert rows is None and n == 0


def test_resolver_warns_and_returns_none_below_support(tmp_path, caplog):
    db = _make_authority_db(tmp_path)
    t2r = {f"t{i}": i for i in range(40)}
    import logging
    with caplog.at_level(logging.WARNING):
        rows, n, used = resolve_tag_sonic_prototype_rows(
            ["jangle pop"], metadata_db_path=db, track_id_to_row=t2r,
            exclude_artist="Seed", min_support=100)   # 37 < 100
    assert rows is None
    assert any("sonic prototype" in r.message.lower() for r in caplog.records)


def test_resolver_unmapped_tag_returns_none_and_warns(tmp_path, caplog):
    db = _make_authority_db(tmp_path)
    t2r = {f"t{i}": i for i in range(40)}
    import logging
    with caplog.at_level(logging.WARNING):
        rows, n, _ = resolve_tag_sonic_prototype_rows(
            ["not a real genre"], metadata_db_path=db, track_id_to_row=t2r, min_support=1)
    assert rows is None and n == 0
    assert any("canonical genre" in r.message.lower() for r in caplog.records)


def test_resolver_empty_tags_returns_none(tmp_path):
    db = _make_authority_db(tmp_path)
    rows, n, used = resolve_tag_sonic_prototype_rows(
        [], metadata_db_path=db, track_id_to_row={}, min_support=25)
    assert rows is None and n == 0
