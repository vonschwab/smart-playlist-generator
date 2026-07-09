"""User-selected genre-tag steering: tag names -> dense target vector.

Single resolver shared by the candidate-pool lever (pipeline/core.py) and the
artist pier lever (playlist_generator -> artist_style). Soft-bias only:
callers blend or re-rank with the target; nothing here gates or excludes.
A selected tag that cannot act WARNS loudly — never a silent no-op.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def resolve_tag_steering_target(
    tags: Sequence[str],
    *,
    genre_vocab: Sequence[str],
    genre_emb: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], list[str], list[str]]:
    """Map tag names to a unit-norm mean of their vocabulary embeddings.

    Returns ``(target | None, mapped_tags, unmapped_tags)``. Matching is
    case-insensitive on the artifact ``genre_vocab``. Returns ``None`` when
    nothing maps or the dense vocabulary embedding is absent.
    """
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return None, [], []
    if genre_emb is None:
        logger.warning(
            "Tag steering requested (%s) but the artifact's dense genre sidecar "
            "has no vocabulary embedding (genre_emb) — steering disabled for this run.",
            wanted,
        )
        return None, [], list(wanted)
    vocab_lower = {str(v).strip().lower(): i for i, v in enumerate(genre_vocab)}
    mapped: list[str] = []
    unmapped: list[str] = []
    rows: list[int] = []
    for tag in wanted:
        idx = vocab_lower.get(tag.lower())
        if idx is None or idx >= int(genre_emb.shape[0]):
            unmapped.append(tag)
        else:
            mapped.append(tag)
            rows.append(int(idx))
    if unmapped:
        logger.warning(
            "Tag steering: %d/%d selected tags not in the artifact genre vocabulary: %s",
            len(unmapped), len(wanted), unmapped,
        )
    if not rows:
        logger.warning("Tag steering: no selected tags mapped — steering disabled for this run.")
        return None, mapped, unmapped
    target = np.asarray(genre_emb, dtype=np.float64)[rows].mean(axis=0)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        logger.warning("Tag steering: degenerate zero-norm target — steering disabled for this run.")
        return None, mapped, unmapped
    target = target / norm
    logger.info("Tag steering target: tags=%s (mapped %d/%d)", mapped, len(rows), len(wanted))
    return target, mapped, unmapped


def sonic_global_mean(sonic_matrix: np.ndarray) -> np.ndarray:
    """Mean of the per-row L2-normalized sonic rows (the 'generic' direction)."""
    M = np.asarray(sonic_matrix, dtype=np.float64)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return Mn.mean(axis=0)


def sonic_prototype_from_rows(
    sonic_matrix: np.ndarray,
    rows: Sequence[int],
    *,
    global_mean: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], float, int]:
    """Centered, L2-normalized centroid of ``rows`` + intra-set cohesion.

    ``rows`` index into ``sonic_matrix`` (bundle-aligned). When ``global_mean`` is
    given it is subtracted from each normalized member row before averaging, to
    remove the generic-sonic component. ``cohesion`` is the mean cosine of member
    vectors to the prototype (low => sonically multimodal tag). Returns
    ``(prototype | None, cohesion, support_n)``.
    """
    idx = [int(r) for r in rows]
    if not idx:
        return None, 0.0, 0
    M = np.asarray(sonic_matrix, dtype=np.float64)[idx]
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    if global_mean is not None:
        Mn = Mn - np.asarray(global_mean, dtype=np.float64)
    proto = Mn.mean(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm <= 1e-12:
        return None, 0.0, len(idx)
    proto = proto / norm
    cohesion = float(np.mean(Mn @ proto))
    return proto, cohesion, len(idx)


def _canonical_genre_ids_for_tags(con: "sqlite3.Connection", tags: Sequence[str]) -> set:
    """Chip names/ids (space or underscore form) -> canonical genre_ids via
    genre_graph_canonical_genres + genre_graph_aliases. {} if none map."""
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return set()
    lower = [t.lower() for t in wanted]
    under = [t.lower().replace(" ", "_") for t in wanted]
    ph = ",".join("?" for _ in wanted)
    cur = con.cursor()
    gids: set = set()
    cur.execute(
        f"SELECT genre_id FROM genre_graph_canonical_genres "
        f"WHERE lower(name) IN ({ph}) OR lower(genre_id) IN ({ph})",
        lower + under,
    )
    gids.update(str(r[0]) for r in cur.fetchall())
    cur.execute(
        f"SELECT canonical_genre_id FROM genre_graph_aliases WHERE lower(alias) IN ({ph})",
        lower,
    )
    gids.update(str(r[0]) for r in cur.fetchall())
    return gids


def resolve_artist_on_tag_membership(
    tags: Sequence[str],
    artist_name: str,
    *,
    metadata_db_path: str,
    track_id_to_row: dict,
) -> dict:
    """{bundle_row: tag_hit_count} for the SEED artist's authority on-tag tracks
    (seed-included; union over the selected tags). {} when nothing maps or no on-tag
    tracks. Reads the authority via on_tag_track_ids_for_artist (One Rule)."""
    from src.genre.authority import on_tag_track_ids_for_artist
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return {}
    con = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        gids = _canonical_genre_ids_for_tags(con, wanted)
        if not gids:
            logger.warning(
                "Tag-first piers: none of %s map to a canonical genre — legacy pier "
                "selection for this run.", wanted,
            )
            return {}
        hits = on_tag_track_ids_for_artist(con, artist_name, gids)
    finally:
        con.close()
    return {track_id_to_row[t]: int(n) for t, n in hits.items() if t in track_id_to_row}


def resolve_tag_sonic_prototype_rows(
    tags: Sequence[str],
    *,
    metadata_db_path: str,
    track_id_to_row: dict,
    exclude_artist: Optional[str] = None,
    exclude_artists: Optional[Sequence[str]] = None,
    min_support: int = 25,
) -> tuple[Optional[list], int, list]:
    """Library row indices for tracks whose RELEASE carries ANY selected tag in the
    PUBLISHED GENRE AUTHORITY (``release_effective_genres``, joined via ``album_id``),
    seed-artist(s) excluded. Rows index into the caller's bundle track ordering.

    Reads the authority, NOT ``track_effective_genres`` — the latter is a raw/partial
    input layer (MusicBrainz-polluted, missing file-sourced tags) and reading it here
    was a real bug (2026-07-08: user-tagged Ghost Box 'hauntology' tracks were invisible
    while the authority had them correctly). See the ``genre-data-authority`` skill.

    Chip names (``tags``, space form) are mapped to canonical ``genre_id`` (underscore
    form) via ``genre_graph_canonical_genres`` + ``genre_graph_aliases``. Only non-inferred
    layers (``observed_leaf``/``legacy``) count — matching ``resolved_genres_for_artist``'s
    chip semantics (graph-inferred hub families are excluded by design).
    ``exclude_artist``/``exclude_artists`` are unioned and excluded so the learned prototype
    never partly describes the seed artist. Returns ``(rows | None, support_n, tags_used)``;
    ``None`` + WARN when no tag maps to a canonical genre or support < ``min_support``."""
    wanted = [str(t).strip() for t in tags if str(t).strip()]
    if not wanted:
        return None, 0, []
    _excl_artists = {str(a).strip() for a in (exclude_artists or []) if str(a).strip()}
    if exclude_artist and str(exclude_artist).strip():
        _excl_artists.add(str(exclude_artist).strip())
    con = sqlite3.connect(f"file:{metadata_db_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        gids = _canonical_genre_ids_for_tags(con, wanted)
        if not gids:
            logger.warning(
                "Tag steering sonic prototype: none of %s map to a canonical genre in "
                "the authority — sonic pool/beam levers disabled for this run.", wanted,
            )
            return None, 0, wanted
        gph = ",".join("?" for _ in gids)
        query = (
            "SELECT DISTINCT t.track_id FROM tracks t "
            "JOIN release_effective_genres reg ON reg.album_id = t.album_id "
            f"WHERE reg.genre_id IN ({gph}) AND reg.assignment_layer NOT LIKE 'inferred%'"
        )
        params = list(gids)
        if _excl_artists:
            aph = ",".join("?" for _ in _excl_artists)
            query += f" AND t.artist NOT IN ({aph})"
            params += list(_excl_artists)
        cur.execute(query, params)
        tids = [str(r[0]) for r in cur.fetchall()]
    finally:
        con.close()
    rows = [track_id_to_row[t] for t in tids if t in track_id_to_row]
    if len(rows) < int(min_support):
        logger.warning(
            "Tag steering sonic prototype: only %d library tracks carry %s in the "
            "authority (min_support=%d) — sonic prototype disabled, falling back to "
            "the genre-dense signal for this run.",
            len(rows), wanted, int(min_support),
        )
        return None, len(rows), wanted
    return rows, len(rows), wanted


def build_tag_first_pier_members(
    membership: dict,
    combined_affinity,
    artist_indices: Sequence[int],
    *,
    target_pier_count: int,
    cluster_k_min: int,
    topup_mult: float,
) -> Optional[set]:
    """On-tag pier member set M (bundle indices) or None (=> legacy fallback).

    members = keys(membership) (authority on-tag, seed-included). Empty -> None: we go
    tag-first ONLY when the artist actually has authority on-tag tracks, never fabricate
    membership from a sonic proxy. If len(members) < floor, top up with the artist's
    highest-combined-affinity NON-member tracks. floor = min(len(artist_indices),
    max(cluster_k_min, ceil(topup_mult * target_pier_count))).
    """
    members = set(int(i) for i in membership.keys())
    if not members:
        return None
    artist_set = [int(i) for i in artist_indices]
    floor = min(len(artist_set), max(int(cluster_k_min), math.ceil(float(topup_mult) * int(target_pier_count))))
    if len(members) >= floor:
        return members
    aff = np.asarray(combined_affinity, dtype=np.float64)
    candidates = [i for i in artist_set if i not in members]
    candidates.sort(key=lambda i: (-float(aff[i]), i))   # highest affinity first, index tiebreak
    for i in candidates:
        if len(members) >= floor:
            break
        members.add(i)
    return members
