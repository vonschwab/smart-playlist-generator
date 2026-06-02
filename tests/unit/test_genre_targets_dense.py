import numpy as np
from src.playlist.pier_bridge.genre_targets import build_dense_genre_targets


def _toy():
    # dense pier vectors in a 3-dim genre embedding
    dA = np.array([1.0, 0.0, 0.0])
    dB = np.array([0.0, 0.0, 1.0])
    return dA, dB


def test_linear_dense_targets_interpolate_endpoints():
    dA, dB = _toy()
    g = build_dense_genre_targets(dA, dB, interior_length=3, route="linear",
                                  genre_emb=None, genre_vocab=None, genre_graph=None,
                                  labels_a=None, labels_b=None)
    assert len(g) == 3
    # first target leans toward A, last toward B
    assert float(g[0] @ (dA/np.linalg.norm(dA))) > float(g[0] @ (dB/np.linalg.norm(dB)))
    assert float(g[-1] @ (dB/np.linalg.norm(dB))) > float(g[-1] @ (dA/np.linalg.norm(dA)))
    # rows L2-normalized
    for v in g:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_ladder_walks_rungs_in_dense_space():
    # 4 genres; emb so that a path noiserock->nowave->collegerock->powerpop exists
    vocab = ["noise rock", "no wave", "college rock", "power pop"]
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.0, 0.0],
        [0.0, 0.7, 0.7, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    graph = {
        "noise rock": [("no wave", 0.7)],
        "no wave": [("noise rock", 0.7), ("college rock", 0.49)],
        "college rock": [("no wave", 0.49), ("power pop", 0.0)],
        "power pop": [("college rock", 0.0)],
    }
    dA, dB = emb[0], emb[3]
    g = build_dense_genre_targets(
        dA, dB, interior_length=4, route="ladder",
        genre_emb=emb, genre_vocab=vocab, genre_graph=graph,
        labels_a=["noise rock"], labels_b=["power pop"], max_steps=6,
    )
    assert len(g) == 4
    # a mid target should be closer to an intermediate rung (no wave/college rock)
    # than to either endpoint, proving it walked rather than blended directly.
    mid = g[1]
    inter = emb[1]  # no wave
    assert float(mid @ inter) > float(mid @ dA)
