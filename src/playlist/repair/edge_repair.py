from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from src.features.artifacts import ArtifactBundle
from src.playlist.artist_identity_resolver import ArtistIdentityConfig, resolve_artist_identity_keys
from src.playlist.identity_keys import TrackIdentityKeys, identity_keys_for_index
from src.playlist.title_quality import detect_title_artifacts
from src.playlist.transition_metrics import (
    TransitionMetricContext,
    is_broken_transition,
    score_transition_edge,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdgeRepairResult:
    indices: list[int]
    swap_log: list[dict] = field(default_factory=list)


def _as_int_set(values: Optional[Iterable[int]]) -> set[int]:
    return {int(v) for v in (values or [])}


def _title_for_idx(bundle: ArtifactBundle, idx: int) -> str:
    try:
        if bundle.track_titles is not None:
            return str(bundle.track_titles[int(idx)] or "")
    except Exception:
        return ""
    return ""


def _track_id_for_idx(bundle: ArtifactBundle, idx: int) -> str:
    try:
        return str(bundle.track_ids[int(idx)])
    except Exception:
        return str(idx)


def _edge(context: TransitionMetricContext, prev_idx: int, cur_idx: int) -> dict:
    return score_transition_edge(context, int(prev_idx), int(cur_idx))


def _adjacent_edges(context: TransitionMetricContext, indices: Sequence[int], position: int) -> list[dict]:
    edges: list[dict] = []
    if position > 0:
        edges.append(_edge(context, int(indices[position - 1]), int(indices[position])))
    if position < len(indices) - 1:
        edges.append(_edge(context, int(indices[position]), int(indices[position + 1])))
    return edges


def _worst_t(edges: Sequence[dict]) -> float:
    vals = [float(e.get("T")) for e in edges if isinstance(e.get("T"), (int, float))]
    return min(vals) if vals else 1.0


def _needs_repair(
    edge: dict, *, t_floor: float, transition_floor: float, centered_cos_floor: float
) -> bool:
    """Break-glass trigger: weak T (below t_floor) OR catastrophic anti-alignment.

    t_floor=0 disables the weak-T arm (T is always > 0 post-sigmoid), reverting to
    the legacy anti-alignment-only behavior. is_broken_transition itself is untouched.
    """
    t_val = edge.get("T")
    if isinstance(t_val, (int, float)) and float(t_val) < float(t_floor):
        return True
    return is_broken_transition(
        edge,
        transition_floor=float(transition_floor),
        centered_cos_floor=float(centered_cos_floor),
    )


def _all_edges_clear(
    edges: Sequence[dict],
    *,
    transition_floor: float,
    centered_cos_floor: float,
) -> bool:
    return not any(
        is_broken_transition(
            e,
            transition_floor=float(transition_floor),
            centered_cos_floor=float(centered_cos_floor),
        )
        for e in edges
    )


def _non_seed_artist_counts_after_replacement(
    candidate: int,
    current_indices: Sequence[int],
    replace_position: int,
    bundle: ArtifactBundle,
    seed_indices: set[int],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    memo: Optional[_IdentityMemo] = None,
) -> dict[str, int]:
    if memo is None:  # standalone call: local memo keeps results identical, just unshared
        memo = _IdentityMemo()
    counts: dict[str, int] = {}
    for pos, idx in enumerate(current_indices):
        track_idx = int(candidate) if int(pos) == int(replace_position) else int(idx)
        if track_idx in seed_indices:
            continue
        for artist_key in memo.cap_keys(bundle, track_idx, artist_identity_cfg):
            counts[str(artist_key)] = counts.get(str(artist_key), 0) + 1
    return counts


def _cap_artist_keys_for_idx(
    bundle: ArtifactBundle,
    idx: int,
    artist_identity_cfg: Optional[ArtistIdentityConfig],
) -> set[str]:
    use_identity = artist_identity_cfg is not None and artist_identity_cfg.enabled
    if use_identity:
        raw_artist = ""
        try:
            if bundle.track_artists is not None:
                raw_artist = str(bundle.track_artists[int(idx)] or "")
        except Exception:
            raw_artist = ""
        if raw_artist:
            try:
                return {
                    str(key)
                    for key in resolve_artist_identity_keys(raw_artist, artist_identity_cfg)
                    if str(key)
                }
            except Exception:
                return set()
    try:
        artist_key = identity_keys_for_index(bundle, int(idx)).artist_key
    except Exception:
        artist_key = ""
    return {str(artist_key)} if artist_key else set()


@dataclass
class _IdentityMemo:
    """Per-repair-pass memo for the pure-per-index identity lookups.

    Within one ``repair_playlist_edges`` call the ``bundle`` is immutable and
    ``artist_identity_cfg`` is fixed, so ``identity_keys_for_index(bundle, idx)``
    and ``_cap_artist_keys_for_idx(bundle, idx, cfg)`` are pure functions of the
    integer track index. This memoizes them by index; lifetime is one repair pass
    (a fresh instance per call), mirroring beam.py's per-segment genre_cache /
    trans_cache. Keyed by index only (never by playlist position), so an accepted
    swap never invalidates an entry — the index->keys mapping is immutable.
    """

    _ident: dict[int, TrackIdentityKeys] = field(default_factory=dict)
    _cap: dict[int, set[str]] = field(default_factory=dict)

    def keys_for_index(self, bundle: ArtifactBundle, idx: int) -> TrackIdentityKeys:
        i = int(idx)
        cached = self._ident.get(i)
        if cached is None:  # identity_keys_for_index never returns None -> safe sentinel
            cached = identity_keys_for_index(bundle, i)
            self._ident[i] = cached
        return cached

    def cap_keys(
        self,
        bundle: ArtifactBundle,
        idx: int,
        artist_identity_cfg: Optional[ArtistIdentityConfig],
    ) -> set[str]:
        i = int(idx)
        if i not in self._cap:  # membership, not truthiness: an empty set is a valid value
            self._cap[i] = _cap_artist_keys_for_idx(bundle, i, artist_identity_cfg)
        return self._cap[i]


def _candidate_refusal_reasons(
    *,
    candidate: int,
    current_indices: Sequence[int],
    replace_position: int,
    bundle: ArtifactBundle,
    seed_indices: set[int],
    pier_indices: set[int],
    allowed_indices: Optional[set[int]],
    disallowed_artist_keys: set[str],
    metric_context: TransitionMetricContext,
    variety_guard_enabled: bool,
    variety_guard_threshold: float,
    max_non_seed_tracks_per_artist: Optional[int],
    artist_identity_cfg: Optional[ArtistIdentityConfig],
    min_gap: int = 0,
    memo: Optional[_IdentityMemo] = None,
) -> list[str]:
    reasons: list[str] = []
    if memo is None:  # standalone call: local memo keeps results identical, just unshared
        memo = _IdentityMemo()
    candidate = int(candidate)
    if candidate in seed_indices:
        reasons.append("candidate_is_seed")
    if candidate in pier_indices:
        reasons.append("candidate_is_pier")
    if candidate in set(int(i) for i in current_indices):
        reasons.append("existing_playlist_index")
    if allowed_indices is not None and candidate not in allowed_indices:
        reasons.append("allowed_set")

    try:
        cand_keys = memo.keys_for_index(bundle, candidate)
    except Exception:
        cand_keys = None
    if cand_keys is not None:
        existing_track_keys = set()
        for pos, idx in enumerate(current_indices):
            if int(pos) == int(replace_position):
                continue
            try:
                existing_track_keys.add(memo.keys_for_index(bundle, int(idx)).track_key)
            except Exception:
                continue
        if cand_keys.track_key in existing_track_keys:
            reasons.append("duplicate_track_key")
        if cand_keys.artist_key and str(cand_keys.artist_key) in disallowed_artist_keys:
            reasons.append("disallowed_artist")

    if (
        max_non_seed_tracks_per_artist is not None
        and int(max_non_seed_tracks_per_artist) > 0
        and any(
            count > int(max_non_seed_tracks_per_artist)
            for count in _non_seed_artist_counts_after_replacement(
                candidate,
                current_indices,
                replace_position,
                bundle,
                seed_indices,
                artist_identity_cfg,
                memo=memo,
            ).values()
        )
    ):
        reasons.append("max_non_seed_artist_cap")

    if int(min_gap) > 0:
        cand_artist_keys = memo.cap_keys(bundle, candidate, artist_identity_cfg)
        if cand_artist_keys:
            lo = max(0, int(replace_position) - int(min_gap))
            hi = min(len(current_indices) - 1, int(replace_position) + int(min_gap))
            for pos in range(lo, hi + 1):
                if int(pos) == int(replace_position):
                    continue
                other_keys = memo.cap_keys(
                    bundle, int(current_indices[pos]), artist_identity_cfg
                )
                if cand_artist_keys & other_keys:
                    reasons.append("min_gap")
                    break

    if detect_title_artifacts(_title_for_idx(bundle, candidate)):
        reasons.append("title_artifact")
    if variety_guard_enabled and replace_position > 0:
        edge = _edge(metric_context, int(current_indices[replace_position - 1]), candidate)
        s_val = edge.get("S")
        if isinstance(s_val, (int, float)) and float(s_val) > float(variety_guard_threshold):
            reasons.append("variety_guard")
    return reasons


def repair_playlist_edges(
    *,
    final_indices: Sequence[int],
    candidate_indices: Iterable[int],
    metric_context: TransitionMetricContext,
    bundle: ArtifactBundle,
    seed_indices: Iterable[int],
    pier_positions: Iterable[int],
    transition_floor: float,
    centered_cos_floor: float = -0.5,
    margin: float = 0.05,
    allowed_indices: Optional[Iterable[int]] = None,
    disallowed_artist_keys: Optional[Iterable[str]] = None,
    repair_edge_position: Optional[int] = None,
    variety_guard_enabled: bool = False,
    variety_guard_threshold: float = 0.85,
    max_non_seed_tracks_per_artist: Optional[int] = None,
    artist_identity_cfg: Optional[ArtistIdentityConfig] = None,
    t_floor: float = 0.0,
    min_gap: int = 0,
) -> EdgeRepairResult:
    """Best-effort swap of interior tracks to lift weak/broken adjacent transitions.

    Guarantee is LOCAL-PAIR MINIMAX, not global monotonicity: an accepted swap
    improves min(T_in, T_out) for the swapped slot by >= margin; an individual
    adjacent edge may still end lower than before (bounded below by
    old_worst + margin). Edges triggered only as a side-effect of a later swap
    are not re-enqueued this pass.
    """

    indices = [int(i) for i in final_indices]
    swap_log: list[dict] = []
    if len(indices) < 2:
        return EdgeRepairResult(indices=indices, swap_log=swap_log)

    seed_set = _as_int_set(seed_indices)
    pier_pos_set = _as_int_set(pier_positions)
    pier_idx_set = {int(indices[pos]) for pos in pier_pos_set if 0 <= int(pos) < len(indices)}
    allowed_set = _as_int_set(allowed_indices) if allowed_indices is not None else None
    disallowed_artist_set = {str(v) for v in (disallowed_artist_keys or []) if str(v)}
    candidates = [int(c) for c in candidate_indices]
    identity_memo = _IdentityMemo()  # per-pass memo: pure-per-index identity lookups

    if repair_edge_position is not None:
        edge_positions = [int(repair_edge_position)]
    else:
        scored: list[tuple[float, int]] = []
        for pos in range(1, len(indices)):
            e = _edge(metric_context, indices[pos - 1], indices[pos])
            if _needs_repair(
                e, t_floor=float(t_floor),
                transition_floor=float(transition_floor),
                centered_cos_floor=float(centered_cos_floor),
            ):
                t_val = e.get("T")
                scored.append(
                    (float(t_val) if isinstance(t_val, (int, float)) else 1.0, int(pos))
                )
        scored.sort()  # worst-first; a neighboring swap may fix later entries
        edge_positions = [pos for _t, pos in scored]
    edges_triggered = len(edge_positions)
    edges_repaired = 0

    for edge_pos in edge_positions:
        if edge_pos <= 0 or edge_pos >= len(indices):
            continue
        current_edge = _edge(metric_context, indices[edge_pos - 1], indices[edge_pos])
        if not _needs_repair(
            current_edge,
            t_floor=float(t_floor),
            transition_floor=float(transition_floor),
            centered_cos_floor=float(centered_cos_floor),
        ):
            continue

        if edge_pos not in pier_pos_set:
            replace_pos = edge_pos
            replace_reason = "destination"
        elif (edge_pos - 1) not in pier_pos_set:
            replace_pos = edge_pos - 1
            replace_reason = "source_before_pier"
        else:
            swap_log.append(
                {
                    "edge_position": int(edge_pos),
                    "reason": "pier",
                    "old_idx": int(indices[edge_pos]),
                    "old_id": _track_id_for_idx(bundle, int(indices[edge_pos])),
                }
            )
            continue

        old_edges = _adjacent_edges(metric_context, indices, replace_pos)
        old_worst = _worst_t(old_edges)
        best: Optional[tuple[float, int, list[dict]]] = None

        for cand in candidates:
            reasons = _candidate_refusal_reasons(
                candidate=int(cand),
                current_indices=indices,
                replace_position=replace_pos,
                bundle=bundle,
                seed_indices=seed_set,
                pier_indices=pier_idx_set,
                allowed_indices=allowed_set,
                disallowed_artist_keys=disallowed_artist_set,
                metric_context=metric_context,
                variety_guard_enabled=bool(variety_guard_enabled),
                variety_guard_threshold=float(variety_guard_threshold),
                max_non_seed_tracks_per_artist=max_non_seed_tracks_per_artist,
                artist_identity_cfg=artist_identity_cfg,
                min_gap=int(min_gap),
                memo=identity_memo,
            )
            if reasons:
                for reason in reasons:
                    swap_log.append(
                        {
                            "edge_position": int(edge_pos),
                            "position": int(replace_pos),
                            "candidate_idx": int(cand),
                            "candidate_id": _track_id_for_idx(bundle, int(cand)),
                            "reason": reason,
                        }
                    )
                continue

            trial = list(indices)
            trial[replace_pos] = int(cand)
            new_edges = _adjacent_edges(metric_context, trial, replace_pos)
            if not _all_edges_clear(
                new_edges,
                transition_floor=float(transition_floor),
                centered_cos_floor=float(centered_cos_floor),
            ):
                continue
            new_worst = _worst_t(new_edges)
            if new_worst < old_worst + float(margin):
                continue
            if best is None or new_worst > best[0]:
                best = (float(new_worst), int(cand), new_edges)

        if best is None:
            continue

        _new_worst, new_idx, _new_edges = best
        old_idx = int(indices[replace_pos])
        indices[replace_pos] = int(new_idx)
        edges_repaired += 1
        logger.info(
            "Edge repair: pos=%d swapped %s -> %s, worst-T %.3f -> %.3f",
            int(replace_pos), _track_id_for_idx(bundle, old_idx),
            _track_id_for_idx(bundle, int(new_idx)), float(old_worst), float(_new_worst),
        )
        swap_log.append(
            {
                "edge_position": int(edge_pos),
                "position": int(replace_pos),
                "old_idx": old_idx,
                "new_idx": int(new_idx),
                "old_id": _track_id_for_idx(bundle, old_idx),
                "new_id": _track_id_for_idx(bundle, int(new_idx)),
                "reason": replace_reason,
                "old_worst_T": float(old_worst),
                "new_worst_T": float(_new_worst),
            }
        )

    if repair_edge_position is None and edges_triggered:
        logger.info(
            "Edge repair summary: triggered=%d repaired=%d left_alone=%d (t_floor=%.2f)",
            edges_triggered, edges_repaired, edges_triggered - edges_repaired, float(t_floor),
        )

    return EdgeRepairResult(indices=indices, swap_log=swap_log)
