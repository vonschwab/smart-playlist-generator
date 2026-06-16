# tests/unit/test_genre_audition_build.py
import numpy as np

from scripts.research.genre_audition_build import (
    top_k_row,
    resolve_cooc_index,
    sample_decoys,
    union_cards,
    build_seed_manifest,
    _slug,
)


def _norm(s):
    return s.strip().lower()


# canonicalize: clean tokens map to themselves; "funk / soul" is unmappable junk
_CANON = {"indie rock": "indie rock", "shoegaze": "shoegaze", "dream pop": "dream pop"}


def _canon(token):
    return _CANON.get(_norm(token))


def test_top_k_row_excludes_self_and_returns_sorted():
    vocab = ["a", "b", "c", "d"]
    S = np.array([
        [1.0, 0.9, 0.1, 0.5],
        [0.9, 1.0, 0.2, 0.3],
        [0.1, 0.2, 1.0, 0.4],
        [0.5, 0.3, 0.4, 1.0],
    ])
    out = top_k_row(S, vocab, seed_idx=0, k=2)
    assert out == [("b", 0.9), ("d", 0.5)]


def test_top_k_row_honors_exclude():
    vocab = ["a", "b", "c"]
    S = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.1], [0.8, 0.1, 1.0]])
    out = top_k_row(S, vocab, seed_idx=0, k=2, exclude={"b"})
    assert out == [("c", 0.8)]


def test_resolve_cooc_index_exact_match():
    cooc_vocab = ["rock", "indie rock", "jazz"]
    token, idx = resolve_cooc_index("indie rock", cooc_vocab, _canon)
    assert token == "indie rock" and idx == 1


def test_resolve_cooc_index_via_canonicalization():
    # canonical seed "dream pop" is absent verbatim but a token canonicalizes to it
    cooc_vocab = ["rock", "dreampop", "jazz"]
    canon = lambda t: "dream pop" if _norm(t) == "dreampop" else None
    token, idx = resolve_cooc_index("dream pop", cooc_vocab, canon)
    assert token == "dreampop" and idx == 1


def test_resolve_cooc_index_no_match():
    token, idx = resolve_cooc_index("nonexistent", ["rock", "jazz"], _canon)
    assert token is None and idx is None


def test_sample_decoys_from_low_tail_disjoint():
    vocab = ["seed", "near", "mid", "far1", "far2", "far3"]
    # seed row: high to near, ~0 to far*
    S = np.zeros((6, 6))
    np.fill_diagonal(S, 1.0)
    S[0, 1] = S[1, 0] = 0.8
    S[0, 2] = S[2, 0] = 0.4
    rng = np.random.default_rng(0)
    decoys = sample_decoys(S, vocab, seed_idx=0, n=2, exclude={"near"}, rng=rng)
    assert len(decoys) == 2
    assert "seed" not in decoys and "near" not in decoys
    assert all(d in {"far1", "far2", "far3", "mid"} for d in decoys)


def test_union_merges_provenances_on_canonical_key():
    graph_n = [("indie rock", 0.82), ("shoegaze", 0.5)]
    cooc_n = [("indie rock", 0.41), ("funk / soul", 0.3)]  # "funk / soul" -> None
    decoys = ["polka"]
    cards = union_cards(graph_n, cooc_n, decoys, _canon, _norm)
    # indie rock card carries BOTH graph and cooccurrence
    ir = next(c for c in cards if c["name"] == "indie rock")
    assert set(ir["spaces"]) == {"graph", "cooccurrence"}
    assert ir["spaces"]["graph"]["rank"] == 1
    assert ir["spaces"]["cooccurrence"]["rank"] == 1
    # unmappable cooc token stays its own raw card
    junk = next(c for c in cards if c["name"] == "funk / soul")
    assert set(junk["spaces"]) == {"cooccurrence"}
    # decoy card
    polka = next(c for c in cards if c["name"] == "polka")
    assert set(polka["spaces"]) == {"decoy"}


def test_build_seed_manifest_blinded_structure():
    graph_vocab = ["indie rock", "shoegaze", "dream pop", "polka", "techno"]
    cooc_vocab = ["indie rock", "dream pop", "funk / soul"]
    gS = np.eye(5)
    gS[0, 1] = gS[1, 0] = 0.7   # indie rock ~ shoegaze
    gS[0, 2] = gS[2, 0] = 0.6   # indie rock ~ dream pop
    cS = np.eye(3)
    cS[0, 1] = cS[1, 0] = 0.4
    cS[0, 2] = cS[2, 0] = 0.5

    def artist_fn(name):
        return ["ArtistX", "ArtistY"]

    m = build_seed_manifest(
        seed="indie rock",
        graph=(graph_vocab, gS),
        cooc=(cooc_vocab, cS),
        canonicalize=_canon,
        normalize=_norm,
        artist_fn=artist_fn,
        k=3,
        n_decoy=1,
        rng_seed=7,
    )
    assert m is not None
    assert m["slug"] == "indie_rock"
    assert m["seed"]["genre"] == "indie rock"
    assert m["seed"]["artists"] == ["ArtistX", "ArtistY"]
    # blinded: neighbors carry NO provenance/sim/rank
    for n in m["neighbors"]:
        assert set(n.keys()) == {"name", "artists"}
    # space_data lives separately, keyed by name
    assert "space_data" in m
    assert all("graph" in v or "cooccurrence" in v or "decoy" in v
               for v in m["space_data"].values())


def test_build_seed_manifest_unknown_seed_returns_none():
    m = build_seed_manifest(
        seed="not a genre",
        graph=(["indie rock"], np.eye(1)),
        cooc=(["indie rock"], np.eye(1)),
        canonicalize=_canon,
        normalize=_norm,
        artist_fn=lambda n: [],
        k=3,
        n_decoy=1,
        rng_seed=1,
    )
    assert m is None


def test_slug():
    assert _slug("indie rock") == "indie_rock"
    assert _slug("witch house") == "witch_house"
    assert _slug("funk / soul") == "funk_soul"
