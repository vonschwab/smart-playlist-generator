# src/playlist/pier_bridge/var_bridge.py
"""Variable bridge length — pick each segment's interior length to maximize its
worst edge (bottleneck), flexing off the nominal only when it earns it.

The bottleneck of a bridge is the weakest edge over pier_a -> interior -> pier_b,
INCLUDING the return edge, so a shorter bridge can never hide a bad landing.
"""
from __future__ import annotations

from typing import Callable


def segment_bottleneck(nodes, edge_score: Callable[[int, int], float]) -> tuple[float, int]:
    """Min edge score over the complete bridge nodes=[pier_a, *interior, pier_b],
    and the index of the weakest edge (0 = pier_a->first)."""
    best = float("inf")
    best_i = 0
    for i in range(len(nodes) - 1):
        s = float(edge_score(int(nodes[i]), int(nodes[i + 1])))
        if s < best:
            best, best_i = s, i
    return best, best_i


def choose_segment_length(nominal: int, lo: int, hi: int,
                          build_and_score: Callable[[int], tuple], *,
                          good_enough: float, eps: float) -> tuple[int, object, bool]:
    """Choose interior length in [lo, hi] maximizing the segment bottleneck.

    Tries the nominal first; if its bottleneck >= good_enough, keeps it (no flex,
    one build). Otherwise builds the other allowed lengths and picks the best
    bottleneck, preferring the length CLOSEST to nominal among those within eps of
    the best (the prefer-N + eps anti-crutch).

    Returns (chosen_length, chosen_path, flexed) where flexed is True iff the
    nominal bottleneck was below good_enough and other lengths were evaluated."""
    nom = max(lo, min(hi, int(nominal)))
    nom_path, nom_b = build_and_score(nom)
    if nom_b >= good_enough:
        return nom, nom_path, False
    results = {nom: (nom_b, nom_path)}
    for l in range(lo, hi + 1):
        if l not in results:
            path, b = build_and_score(l)
            results[l] = (b, path)
    if len(results) == 1:
        # Only the nominal length was buildable (lo == hi == nom): no alternative
        # was evaluated, so no flex actually occurred even though nominal fell short
        # of good_enough.  Report flexed=False so the caller's flex budget/counter
        # and the "flexed" log reflect real exploration work — this is what fixes
        # the (N+1/N) over-count once the flex cap has forced lo==hi==nominal.
        return nom, nom_path, False
    best_b = max(b for b, _ in results.values())
    near = [l for l, (b, _) in results.items() if b >= best_b - eps]
    chosen = min(near, key=lambda l: (abs(l - nom), l))    # closest to nominal, then smaller
    return chosen, results[chosen][1], True
