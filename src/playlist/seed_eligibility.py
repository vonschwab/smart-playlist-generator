"""Scarcity-gated freshness for artist-mode seed piers.

Freshness normally removes recently-played tracks from seed selection. For the
SEED artist that starves the piers when the catalog is small (most-played =
most-popular = recently-played). This re-admits the seed artist's own recently-
played tracks, and ONLY as many as needed to keep >= target_piers eligible.
Other artists' recency exclusions are untouched. See
docs/superpowers/specs/2026-07-01-artist-pier-scarcity-and-spacing-design.md.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence


def seed_recency_exclusion_for_presence(
    artist_track_ids: Iterable[str],
    recency_excluded_ids: Iterable[str],
    target_piers: int,
    *,
    readmit_rank: Optional[Sequence[str]] = None,
) -> set[str]:
    """Return the recency-exclusion set to apply, relaxed only enough that at least
    ``target_piers`` of the seed artist's tracks stay eligible.

    ``readmit_rank``: seed-artist track ids in the order to re-admit first (e.g.
    most-popular first when Popular Seeds is on). Ids absent from the excluded-artist
    set are ignored; excluded artist tracks not in the rank are re-admitted last.
    """
    excluded = {str(t) for t in recency_excluded_ids}
    artist = {str(t) for t in artist_track_ids}
    excluded_artist = excluded & artist
    fresh_count = len(artist) - len(excluded_artist)
    shortfall = int(target_piers) - fresh_count
    if shortfall <= 0:
        return excluded
    if readmit_rank:
        ranked = [str(t) for t in readmit_rank if str(t) in excluded_artist]
        ranked += [t for t in excluded_artist if t not in set(ranked)]
    else:
        ranked = sorted(excluded_artist)  # deterministic
    readmit = set(ranked[:shortfall])
    return excluded - readmit
