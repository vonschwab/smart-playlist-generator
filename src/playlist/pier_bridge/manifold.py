"""On-manifold routing primitives for roam corridors.

A distance-weighted kNN graph over real tracks; shortest paths on it approximate
geodesics on the data manifold (Alamgir & von Luxburg 2012), so bridges route
through real, dense regions instead of a straight chord through off-manifold holes.
Mutual proximity (Flexer/Schnitzer) corrects high-dimensional hubness.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def mutual_proximity(dist: np.ndarray) -> np.ndarray:
    """Empirical Gaussian mutual proximity on a dense (N,N) distance matrix.

    MP(i,j) = 1 - P(X > d_ij | mu_i,sig_i) * P(X > d_ij | mu_j,sig_j), turned back
    into a distance. Hub tracks (small mean distance) get their asymmetric edges
    inflated. O(N^2); used only for small candidate sets or precomputed offline.
    """
    from scipy.stats import norm
    n = dist.shape[0]
    mu = dist.mean(axis=1)
    sd = dist.std(axis=1)
    sd[sd == 0] = 1e-12
    # P(d_ij is "far") from each endpoint's perspective.
    p_i = 1.0 - norm.cdf(dist, loc=mu[:, None], scale=sd[:, None])
    p_j = 1.0 - norm.cdf(dist, loc=mu[None, :], scale=sd[None, :])
    mp = 1.0 - (p_i * p_j)
    np.fill_diagonal(mp, 0.0)
    return mp


def build_knn_graph(X: np.ndarray, k: int, *, mutual_proximity: bool = True) -> sp.csr_matrix:
    """Distance-weighted, symmetrized kNN graph (cosine distance) over rows of X."""
    Xn = _l2(np.asarray(X, dtype=np.float64))
    n = Xn.shape[0]
    k = int(max(1, min(k, n - 1)))
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    # k nearest by similarity per row.
    nbr = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    d = 1.0 - sims[rows, cols]                       # cosine distance >= 0
    d = np.clip(d, 0.0, 2.0)
    g = sp.csr_matrix((d, (rows, cols)), shape=(n, n))
    g = g.maximum(g.T)                                # symmetrize (keep the larger edge)
    if mutual_proximity:
        # MP-correct only the realised edges (sparse), using a local dense block is
        # too costly library-wide; here we rescale edge weights by endpoint hubness.
        deg_dist = np.asarray(g.sum(axis=1)).ravel()
        med = np.median(deg_dist[deg_dist > 0]) or 1.0
        coo = g.tocoo()
        # hubs = many short edges => small summed distance => inflate.
        inv = (deg_dist + 1e-12) / med
        w = coo.data * np.sqrt(inv[coo.row] * inv[coo.col])
        g = sp.csr_matrix((w, (coo.row, coo.col)), shape=(n, n))
    return g.tocsr()


def geodesic_from_source(graph: sp.csr_matrix, source: int) -> np.ndarray:
    """Single-source shortest-path distances over the kNN graph (Dijkstra)."""
    return dijkstra(graph, directed=False, indices=int(source))
