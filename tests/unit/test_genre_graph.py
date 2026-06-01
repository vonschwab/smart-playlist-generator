import numpy as np
from src.playlist.pier_bridge.genre_graph import build_genre_graph


def test_graph_connects_niche_neighbors_and_excludes_hubs():
    # 5 genres: two tight niche clusters + one hub correlated with everything.
    vocab = ["noise rock", "no wave", "power pop", "college rock", "rock"]
    emb = np.array([
        [1.0, 0.0, 0.0],   # noise rock
        [0.95, 0.10, 0.0], # no wave  (near noise rock)
        [0.0, 1.0, 0.0],   # power pop
        [0.0, 0.95, 0.10], # college rock (near power pop)
        [0.6, 0.6, 0.5],   # rock (hub: correlated with all)
    ], dtype=float)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    g = build_genre_graph(emb, vocab, k=3, min_cos=0.5, hub_labels={"rock"})
    # hub excluded as a node
    assert "rock" not in g
    # niche neighbors connected
    assert any(nb == "no wave" for nb, _ in g["noise rock"])
    assert any(nb == "college rock" for nb, _ in g["power pop"])
    # hub never appears as a neighbor either
    for nbrs in g.values():
        assert all(nb != "rock" for nb, _ in nbrs)
