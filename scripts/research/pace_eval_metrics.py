"""Pure metrics for the pace-representation eval (no IO)."""
from __future__ import annotations

import numpy as np


def zscore_params(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return (0.0, 1.0)
    mean = float(finite.mean())
    std = float(finite.std())
    return (mean, 1.0 if std == 0.0 else std)


def apply_zscore(x, mean: float, std: float):
    return (np.asarray(x, dtype=float) - mean) / (std if std else 1.0)


def weighted_euclidean(a, b, weights=None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    d = a - b
    if weights is not None:
        d = d * np.sqrt(np.asarray(weights, dtype=float))
    return float(np.sqrt(np.sum(d * d)))


def auc_pos_below_neg(pos, neg) -> float:
    """P(random positive distance < random negative distance); ties=0.5."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    less = float((pos[:, None] < neg[None, :]).sum())
    ties = float((pos[:, None] == neg[None, :]).sum())
    return (less + 0.5 * ties) / (pos.size * neg.size)


def distribution(distances) -> dict:
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "min": float("nan"), "p10": float("nan"),
                "p50": float("nan"), "p90": float("nan")}
    return {
        "n": int(d.size),
        "min": float(d.min()),
        "p10": float(np.percentile(d, 10)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
    }
