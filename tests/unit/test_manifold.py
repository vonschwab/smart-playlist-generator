import numpy as np
from src.playlist.pier_bridge.manifold import build_knn_graph, geodesic_from_source, mutual_proximity

def _ring():
    # 6 points on a circle: nearest neighbours are the ring neighbours.
    ang = np.linspace(0, 2*np.pi, 6, endpoint=False)
    return np.c_[np.cos(ang), np.sin(ang)].astype(np.float64)

def test_knn_graph_is_symmetric_and_sparse():
    g = build_knn_graph(_ring(), k=2, mutual_proximity_approx=False)
    assert g.shape == (6, 6)
    a = g.toarray()
    assert np.allclose(a, a.T)              # symmetrized
    assert (a > 0).sum(axis=1).min() >= 2   # each node has >= k neighbours

def test_geodesic_follows_the_ring_not_the_chord():
    g = build_knn_graph(_ring(), k=2, mutual_proximity_approx=False)
    d = geodesic_from_source(g, 0)
    # Opposite point (3) is reached by walking the ring (2 hops each way), not a chord.
    assert d[3] > d[1]            # farther around the ring than an adjacent node
    assert np.isfinite(d[3])

def test_mutual_proximity_downweights_a_hub():
    # Point 0 is a hub: close to everyone; others are far from each other.
    X = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]], dtype=np.float64)
    raw = build_knn_graph(X, k=2, mutual_proximity_approx=False).toarray()
    mp  = build_knn_graph(X, k=2, mutual_proximity_approx=True).toarray()
    # MP raises the effective distance from the hub to its asymmetric neighbours.
    assert mp[0].sum() >= raw[0].sum()
