# src/playlist/repair/edge_delete.py
"""Remove-only weak-edge fixer (repair-by-deletion) — the last resort.

Runs AFTER break-glass repair. For an edge still below floor, delete the interior
endpoint whose removal best merges the two edges, but ONLY if the merged edge
strictly beats the broken edge (never-worse). Never deletes a pier/seed. Deleting
a track compacts the list by one slot, which can shrink a BYSTANDER same-artist
pair's positional gap below the hard `min_gap` diversity constraint -- when
`artist_key_of`/`min_gap` are supplied, such a deletion is skipped. See
docs/superpowers/specs/2026-07-02-weak-edge-cascade-reorder-design.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeleteResult:
    indices: list[int]
    delete_log: list[dict] = field(default_factory=list)


def _violates_min_gap_after_delete(idx, artist_key_of, del_pos, min_gap):
    """Would deleting idx[del_pos] compress a BYSTANDER same-artist pair to <= min_gap?
    Deleting del_pos shifts everything after it left by one, so a same-artist pair
    (p1 < del_pos < p2) goes from distance (p2 - p1) to (p2 - p1 - 1). The pre-delete
    list is already min_gap-legal, so only a straddling pair at exactly distance
    min_gap+1 becomes a fresh violation. Scan the window [del_pos-min_gap, del_pos+min_gap]."""
    if not min_gap or artist_key_of is None:
        return False
    before = {}  # artist key -> nearest position < del_pos
    for p in range(max(0, del_pos - min_gap), del_pos):
        before[artist_key_of(idx[p])] = p
    for p in range(del_pos + 1, min(len(idx), del_pos + min_gap + 1)):
        a = artist_key_of(idx[p])
        p1 = before.get(a)
        if p1 is not None and (p - 1) - p1 <= min_gap:
            return True
    return False


def delete_broken_edges(
    indices: Sequence[int],
    *,
    edge_score: Callable[[int, int], float],
    floor: float,
    protected_indices: set[int],
    max_deletions: int = 4,
    artist_key_of: Callable[[int], object] | None = None,
    min_gap: int = 0,
) -> DeleteResult:
    idx = [int(x) for x in indices]
    delete_log: list[dict] = []
    try:
        for _ in range(max(0, int(max_deletions))):
            if len(idx) < 3:
                break
            # worst adjacent edge
            worst_pos, worst_t = 0, float("inf")
            for i in range(len(idx) - 1):
                t = float(edge_score(idx[i], idx[i + 1]))
                if t < worst_t:
                    worst_pos, worst_t = i, t
            if worst_t >= float(floor):
                break  # nothing broken
            # candidate deletions: the two endpoints of the worst edge
            best = None  # (merged_t, del_pos)
            for del_pos in (worst_pos, worst_pos + 1):
                if idx[del_pos] in protected_indices:
                    continue
                prev, nxt = del_pos - 1, del_pos + 1
                if prev < 0 or nxt >= len(idx):
                    continue  # boundary track has no neighbor to merge across
                if _violates_min_gap_after_delete(idx, artist_key_of, del_pos, min_gap):
                    continue  # deletion would breach the hard diversity constraint for a bystander
                merged = float(edge_score(idx[prev], idx[nxt]))
                if best is None or merged > best[0]:
                    best = (merged, del_pos)  # ties -> lower del_pos (worst_pos first)
            if best is None or best[0] <= worst_t:
                break  # nothing improves the broken edge -> leave it
            merged_t, del_pos = best
            delete_log.append({
                "position": del_pos,
                "deleted_idx": idx[del_pos],
                "old_worst_T": worst_t,
                "new_merged_T": merged_t,
            })
            logger.info(
                "Edge-delete: removed interior idx=%s at pos=%d, worst-T %.3f -> merged %.3f",
                idx[del_pos], del_pos, worst_t, merged_t,
            )
            del idx[del_pos]
    except Exception:  # never break a generation
        logger.warning("edge_delete failed; leaving playlist unchanged", exc_info=True)
        return DeleteResult(indices=[int(x) for x in indices], delete_log=[])
    return DeleteResult(indices=idx, delete_log=delete_log)
