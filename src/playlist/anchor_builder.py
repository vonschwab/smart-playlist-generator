from __future__ import annotations

import copy
import itertools
import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.playlist.candidate_pool import CandidatePoolResult
from src.similarity.hybrid import transition_similarity_end_to_start
from src.string_utils import normalize_artist_key
from src.title_dedupe import TitleDedupeTracker

logger = logging.getLogger(__name__)


@dataclass
class AnchorConstraints:
    artist_keys: Sequence[str]
    track_titles: Optional[Sequence[str]]
    track_ids: Optional[Sequence[str]]
    max_per_artist: int
    min_gap: int
    transition_floor: float
    hard_floor: bool = False
    title_dedupe: Optional[TitleDedupeTracker] = None


@dataclass
class BridgeSearchConfig:
    prefilter: int = 400
    max_prefilter: int = 1200
    prefilter_step: int = 300
    beam_width: int = 12
    max_beam_width: int = 48
    beam_width_step: int = 6
    relaxed_floor_delta: float = 0.05
    greedy_lambda: float = 0.35
    debug: bool = False


@dataclass
class TransitionScorer:
    emb_norm: np.ndarray
    transition_gamma: float
    X_end: Optional[np.ndarray] = None
    X_start: Optional[np.ndarray] = None
    rescale_transitions: bool = False

    def score(self, prev_idx: int, next_idx: int) -> float:
        base = float(self.emb_norm[next_idx] @ self.emb_norm[prev_idx])
        if self.X_end is not None and self.X_start is not None:
            seg = float(transition_similarity_end_to_start(self.X_end, self.X_start, prev_idx, np.array([next_idx]))[0])
            if self.rescale_transitions:
                seg = float(np.clip((seg + 1.0) / 2.0, 0.0, 1.0))
            base = self.transition_gamma * seg + (1 - self.transition_gamma) * base
        return base


def compute_anchor_positions(n_tracks: int, n_seeds: int) -> List[int]:
    if n_tracks <= 0:
        raise ValueError("n_tracks must be positive")
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    if n_seeds > n_tracks:
        raise ValueError("n_seeds cannot exceed n_tracks")
    if n_seeds == 1:
        return [0]
    return [int(round(i * (n_tracks - 1) / (n_seeds - 1))) for i in range(n_seeds)]


def choose_seed_order(
    seeds: Sequence[int],
    transition_scorer: TransitionScorer | Callable[[int, int], float],
) -> List[int]:
    """Brute-force the best ordering of seeds based on transition score."""
    if len(seeds) <= 1:
        return list(seeds)

    def _score(order: Sequence[int]) -> float:
        total = 0.0
        for a, b in zip(order, order[1:]):
            total += float(transition_scorer.score(a, b) if hasattr(transition_scorer, "score") else transition_scorer(a, b))  # type: ignore[attr-defined]
        return total

    base_order = list(seeds)
    best_score = _score(base_order)
    best_order = base_order
    for perm in itertools.permutations(seeds):
        score = _score(perm)
        if score > best_score + 1e-9:
            best_score = score
            best_order = list(perm)
    return best_order


def _artist_for_idx(idx: int, constraints: AnchorConstraints) -> str:
    return normalize_artist_key(str(constraints.artist_keys[idx]))


def _title_for_idx(idx: int, constraints: AnchorConstraints) -> str:
    if constraints.track_titles is None:
        return ""
    return str(constraints.track_titles[idx] or "")


def _violates_constraints(
    idx: int,
    counts: dict[str, int],
    recent_artists: Sequence[str],
    constraints: AnchorConstraints,
    *,
    allow_cap: bool = False,
) -> bool:
    artist = _artist_for_idx(idx, constraints)
    if not allow_cap and counts.get(artist, 0) >= constraints.max_per_artist:
        return True
    if constraints.min_gap > 0 and artist in recent_artists[-constraints.min_gap :]:
        return True
    return False


def _update_recency(recent: List[str], artist: str, max_len: int) -> List[str]:
    recent.append(artist)
    if len(recent) > max_len:
        return recent[-max_len:]
    return recent


def build_bridge(
    A: int,
    B: int,
    k: int,
    bridge_candidates: Sequence[int],
    transition_scorer: TransitionScorer | Callable[[int, int], float],
    constraints: AnchorConstraints,
    rng: np.random.Generator,
    *,
    used_counts: Optional[dict[str, int]] = None,
    recent_artists: Optional[List[str]] = None,
    used_track_ids: Optional[Set[int]] = None,
    search_cfg: Optional[BridgeSearchConfig] = None,
) -> List[int]:
    if k <= 0:
        return []

    cfg = search_cfg or BridgeSearchConfig()
    counts = dict(used_counts or {})
    recents = list(recent_artists or [])
    used = set(used_track_ids or set())
    disallowed = used | {A, B}
    label = lambda idx: str(constraints.track_ids[idx]) if constraints.track_ids is not None else str(idx)

    def _score_edge(x: int, y: int) -> float:
        return float(transition_scorer.score(x, y) if hasattr(transition_scorer, "score") else transition_scorer(x, y))  # type: ignore[attr-defined]

    # Precompute affinity to keep the search space bounded.
    affinities: list[tuple[float, int, float, float]] = []
    for cand in bridge_candidates:
        if cand in disallowed:
            continue
        to_a = _score_edge(A, cand)
        to_b = _score_edge(cand, B)
        aff = 0.5 * (to_a + to_b)
        affinities.append((aff, cand, to_a, to_b))
    if not affinities:
        logger.warning("Bridge skipped: no eligible candidates between %s -> %s", A, B)
        return []
    affinities.sort(key=lambda t: t[0], reverse=True)

    recent_limit = max(constraints.min_gap * 2, 8)
    attempt_logs: list[dict[str, object]] = []

    def _beam_once(prefilter: int, beam_width: int, allow_relaxed_floor: bool) -> Tuple[Optional[List[int]], Optional[float]]:
        filtered = affinities[: max(1, min(prefilter, len(affinities)))]
        candidates = [c for _, c, _, _ in filtered]
        if not candidates:
            return None, None

        @dataclass
        class State:
            path: List[int]
            score: float
            counts: dict[str, int]
            recents: List[str]
            dedupe: Optional[TitleDedupeTracker]

        initial_tracker = copy.deepcopy(constraints.title_dedupe) if constraints.title_dedupe else None
        start_state = State(path=[A], score=0.0, counts=dict(counts), recents=list(recents), dedupe=initial_tracker)
        beam: List[State] = [start_state]
        for depth in range(1, k + 1):
            expanded: List[State] = []
            for state in beam:
                prev = state.path[-1]
                for cand in candidates:
                    if cand in state.path or cand in disallowed:
                        continue
                    artist = _artist_for_idx(cand, constraints)
                    if _violates_constraints(cand, state.counts, state.recents, constraints):
                        continue
                    title = _title_for_idx(cand, constraints)
                    tracker = state.dedupe
                    if tracker:
                        is_dup, matched = tracker.is_duplicate(artist, title)
                        if is_dup:
                            continue
                    edge = _score_edge(prev, cand)
                    effective_floor = constraints.transition_floor - (cfg.relaxed_floor_delta if allow_relaxed_floor else 0.0)
                    if constraints.hard_floor and edge < effective_floor:
                        continue
                    new_counts = dict(state.counts)
                    new_counts[artist] = new_counts.get(artist, 0) + 1
                    new_recents = _update_recency(list(state.recents), artist, recent_limit)
                    new_tracker = copy.deepcopy(tracker) if tracker else None
                    if new_tracker:
                        new_tracker.add(artist, title)
                    expanded.append(
                        State(
                            path=state.path + [cand],
                            score=state.score + edge,
                            counts=new_counts,
                            recents=new_recents,
                            dedupe=new_tracker,
                        )
                    )
            if not expanded:
                return None, None
            expanded.sort(key=lambda s: s.score, reverse=True)
            beam = expanded[: max(1, beam_width)]

        best_path: Optional[List[int]] = None
        best_score = -math.inf
        for state in beam:
            last = state.path[-1]
            final_edge = _score_edge(last, B)
            if constraints.hard_floor and final_edge < constraints.transition_floor:
                continue
            # Allow seeds (B) even when at cap, but still avoid tight gaps.
            if _violates_constraints(B, state.counts, state.recents, constraints, allow_cap=True):
                continue
            total_score = state.score + final_edge
            if total_score > best_score:
                best_score = total_score
                best_path = state.path
        return best_path, best_score if best_path is not None else None

    prefilter = cfg.prefilter
    beam_width = cfg.beam_width
    allow_relaxed = False
    attempt = 0
    best_bridge: Optional[List[int]] = None
    best_score: Optional[float] = None
    while attempt < 8:
        path, score = _beam_once(prefilter, beam_width, allow_relaxed)
        attempt_logs.append(
            {
                "prefilter": prefilter,
                "beam_width": beam_width,
                "relaxed": allow_relaxed,
                "candidates": len(affinities),
                "score": score,
            }
        )
        if path is not None and score is not None:
            best_bridge = path
            best_score = score
            break
        # Relaxation schedule: expand prefilter -> expand beam -> allow relaxed floor.
        if prefilter < cfg.max_prefilter:
            prefilter = min(cfg.max_prefilter, prefilter + cfg.prefilter_step)
        elif beam_width < cfg.max_beam_width:
            beam_width = min(cfg.max_beam_width, beam_width + cfg.beam_width_step)
        elif not allow_relaxed:
            allow_relaxed = True
            prefilter = max(prefilter, cfg.prefilter)
            beam_width = max(beam_width, cfg.beam_width)
        else:
            break
        attempt += 1

    fallback_used = best_bridge is None
    logger.info("BUILD_BRIDGE: A=%s B=%s k=%d fallback_used=%s", A, B, k, fallback_used)
    if fallback_used:
        # Greedy fallback: pick best local edge with mild pull toward B.
        available = [c for _, c, _, _ in affinities]
        current = A
        path: List[int] = []
        local_counts = dict(counts)
        local_recents = list(recents)
        while len(path) < k and available:
            best_choice = None
            best_choice_score = -math.inf
            rng.shuffle(available)
            for cand in list(available):
                if cand in disallowed or cand in path:
                    continue
                artist = _artist_for_idx(cand, constraints)
                if _violates_constraints(cand, local_counts, local_recents, constraints):
                    continue
                title = _title_for_idx(cand, constraints)
                tracker = constraints.title_dedupe
                if tracker:
                    is_dup, matched = tracker.is_duplicate(artist, title)
                    if is_dup:
                        logger.info("Greedy fallback: skipping duplicate %s - %s (matches %s)", artist, title, matched)
                        continue
                    if "jens lekman" in artist.lower():
                        logger.info("Greedy considering: %s - %s", artist, title)
                score_local = _score_edge(current, cand) + cfg.greedy_lambda * _score_edge(cand, B)
                if score_local > best_choice_score:
                    best_choice_score = score_local
                    best_choice = cand
            if best_choice is None:
                break
            path.append(best_choice)
            available.remove(best_choice)
            artist = _artist_for_idx(best_choice, constraints)
            local_counts[artist] = local_counts.get(artist, 0) + 1
            local_recents = _update_recency(local_recents, artist, recent_limit)
            if constraints.title_dedupe:
                constraints.title_dedupe.add(artist, _title_for_idx(best_choice, constraints))
            current = best_choice
        # If still short, pad with remaining candidates, preferring those that respect constraints.
        for cand in available:
            if len(path) >= k:
                break
            if cand in disallowed or cand in path:
                continue
            # Check constraints before adding
            artist = _artist_for_idx(cand, constraints)
            if _violates_constraints(cand, local_counts, local_recents, constraints):
                continue
            # Check title deduplication
            title = _title_for_idx(cand, constraints)
            if constraints.title_dedupe and constraints.title_dedupe.is_duplicate(artist, title)[0]:
                continue
            path.append(cand)
            local_counts[artist] = local_counts.get(artist, 0) + 1
            local_recents = _update_recency(local_recents, artist, recent_limit)
            if constraints.title_dedupe:
                constraints.title_dedupe.add(artist, title)
        # Final fallback: if still short, add remaining candidates ignoring min_gap (but still check title_dedupe)
        if len(path) < k:
            remaining_needed = k - len(path)
            added_ignoring_constraints = 0
            for cand in available:
                if len(path) >= k:
                    break
                if cand in disallowed or cand in path:
                    continue
                # Still check title deduplication even when ignoring min_gap
                artist = _artist_for_idx(cand, constraints)
                title = _title_for_idx(cand, constraints)
                if constraints.title_dedupe and constraints.title_dedupe.is_duplicate(artist, title)[0]:
                    continue
                path.append(cand)
                added_ignoring_constraints += 1
                local_counts[artist] = local_counts.get(artist, 0) + 1
                local_recents = _update_recency(local_recents, artist, recent_limit)
                if constraints.title_dedupe:
                    constraints.title_dedupe.add(artist, title)
            if added_ignoring_constraints > 0:
                logger.warning(
                    "Bridge fallback added %d tracks ignoring min_gap constraint (needed=%d, got=%d)",
                    added_ignoring_constraints,
                    remaining_needed,
                    len(path),
                )
        best_bridge = [A] + path
        best_score = None

    assert best_bridge is not None
    bridge_tracks = best_bridge[1:]
    if len(bridge_tracks) < k:
        logger.warning(
            "Anchor bridge %s->%s UNDERFILLED: got %d/%d tracks (candidates=%d fallback=%s)",
            label(A),
            label(B),
            len(bridge_tracks),
            k,
            len(affinities),
            fallback_used,
        )
    else:
        logger.info(
            "Anchor bridge %s->%s k=%d candidates=%d beam=%d fallback=%s best_score=%s",
            label(A),
            label(B),
            k,
            len(affinities),
            beam_width,
            fallback_used,
            f"{best_score:.4f}" if best_score is not None else "n/a",
        )
    if cfg.debug and bridge_tracks:
        edges = []
        prev = A
        for t in bridge_tracks + [B]:
            edges.append((_score_edge(prev, t), label(prev), label(t)))
            prev = t
        logger.debug("Anchor bridge path: %s", edges)
    return bridge_tracks[:k]


def build_anchor_playlist(
    seeds: Sequence[int],
    n_tracks: int,
    candidate_pool: CandidatePoolResult,
    transition_scorer: TransitionScorer | Callable[[int, int], float],
    constraints: AnchorConstraints,
    rng: np.random.Generator,
    *,
    search_cfg: Optional[BridgeSearchConfig] = None,
) -> List[int]:
    if n_tracks <= 0:
        return []
    seeds_unique = []
    seen_seed = set()
    for s in seeds:
        if s in seen_seed:
            continue
        seen_seed.add(s)
        seeds_unique.append(s)
    if not seeds_unique:
        raise ValueError("At least one seed is required for anchor playlist construction.")
    if len(seeds_unique) > n_tracks:
        raise ValueError("Number of seeds exceeds requested playlist length.")

    cfg = search_cfg or BridgeSearchConfig()
    positions = compute_anchor_positions(n_tracks, len(seeds_unique))
    seed_order = choose_seed_order(seeds_unique, transition_scorer)
    order_score = 0.0
    if len(seed_order) > 1:
        order_score = sum(
            float(transition_scorer.score(a, b) if hasattr(transition_scorer, "score") else transition_scorer(a, b))  # type: ignore[attr-defined]
            for a, b in zip(seed_order, seed_order[1:])
        )
    logger.info("BUILD_ANCHOR_PLAYLIST CALLED with %d seeds", len(seed_order))
    logger.info("Anchor seeds order: %s score=%.4f", seed_order, order_score)

    used_counts: dict[str, int] = {}
    recent_artists: List[str] = []
    used_track_ids: Set[int] = set()
    playlist: List[int] = []

    # Exclude seeds from bridge candidates
    seed_set = set(seeds_unique)
    bridge_candidates = [idx for idx in candidate_pool.pool_indices if idx not in seed_set]

    # Pre-populate title_dedupe tracker with ALL seed titles
    # This prevents bridges from picking tracks with same title as future seeds
    if constraints.title_dedupe:
        for seed in seeds_unique:
            seed_artist = _artist_for_idx(seed, constraints)
            seed_title = _title_for_idx(seed, constraints)
            constraints.title_dedupe.add(seed_artist, seed_title)
        logger.debug("Pre-populated title dedupe tracker with %d seed titles", len(seeds_unique))

    for i, seed in enumerate(seed_order):
        artist = _artist_for_idx(seed, constraints)
        used_counts[artist] = used_counts.get(artist, 0) + 1
        playlist.append(seed)
        recent_artists = _update_recency(recent_artists, artist, max(constraints.min_gap * 2, 8))
        used_track_ids.add(seed)
        # Note: seed titles were pre-populated into title_dedupe tracker before the loop
        if i == len(seed_order) - 1:
            continue
        gap = positions[i + 1] - positions[i] - 1
        if gap < 0:
            gap = 0
        bridge = build_bridge(
            seed,
            seed_order[i + 1],
            gap,
            bridge_candidates,
            transition_scorer,
            constraints,
            rng,
            used_counts=used_counts,
            recent_artists=recent_artists,
            used_track_ids=used_track_ids,
            search_cfg=cfg,
        )
        for t in bridge:
            artist_t = _artist_for_idx(t, constraints)
            used_counts[artist_t] = used_counts.get(artist_t, 0) + 1
            recent_artists = _update_recency(recent_artists, artist_t, max(constraints.min_gap * 2, 8))
            used_track_ids.add(t)
            if constraints.title_dedupe:
                constraints.title_dedupe.add(artist_t, _title_for_idx(t, constraints))
        playlist.extend(bridge)

    # Fill any remaining slots with best-fit candidates.
    while len(playlist) < n_tracks:
        remaining = [idx for idx in bridge_candidates if idx not in used_track_ids]
        if not remaining:
            break
        prev = playlist[-1]
        best_idx = None
        best_score = -math.inf
        rng.shuffle(remaining)
        for cand in remaining:
            if _violates_constraints(cand, used_counts, recent_artists, constraints):
                continue
            title = _title_for_idx(cand, constraints)
            artist = _artist_for_idx(cand, constraints)
            if constraints.title_dedupe and constraints.title_dedupe.is_duplicate(artist, title)[0]:
                continue
            score = float(transition_scorer.score(prev, cand) if hasattr(transition_scorer, "score") else transition_scorer(prev, cand))  # type: ignore[attr-defined]
            if score > best_score:
                best_score = score
                best_idx = cand
        if best_idx is None:
            # Relax min_gap constraint but still respect title_dedupe to avoid duplicate songs.
            for cand in remaining:
                title = _title_for_idx(cand, constraints)
                artist = _artist_for_idx(cand, constraints)
                if constraints.title_dedupe and constraints.title_dedupe.is_duplicate(artist, title)[0]:
                    continue
                best_idx = cand
                logger.warning(
                    "Fill-remaining fallback: adding track ignoring min_gap (artist=%s, position=%d)",
                    artist,
                    len(playlist),
                )
                break
            # Final fallback if even title_dedupe can't be satisfied
            if best_idx is None:
                best_idx = remaining[0]
                artist = _artist_for_idx(best_idx, constraints)
                logger.warning(
                    "Fill-remaining fallback: adding track ignoring ALL constraints (artist=%s, position=%d)",
                    artist,
                    len(playlist),
                )
        playlist.append(best_idx)
        used_track_ids.add(best_idx)
        artist = _artist_for_idx(best_idx, constraints)
        used_counts[artist] = used_counts.get(artist, 0) + 1
        recent_artists = _update_recency(recent_artists, artist, max(constraints.min_gap * 2, 8))
        if constraints.title_dedupe:
            constraints.title_dedupe.add(artist, _title_for_idx(best_idx, constraints))

    if len(playlist) > n_tracks:
        # Drop extra non-seed tracks from the end.
        trimmed: List[int] = []
        for idx in playlist:
            if len(trimmed) >= n_tracks:
                break
            trimmed.append(idx)
        playlist = trimmed

    # Log anchor placement diagnostics
    if constraints.track_ids is not None:
        seed_labels = [str(constraints.track_ids[s]) for s in seeds_unique]
    else:
        seed_labels = [str(s) for s in seeds_unique]
    actual_positions = [playlist.index(s) for s in seed_order]
    logger.info(
        "Anchor positions target=%s actual=%s seeds=%s",
        positions,
        actual_positions,
        seed_labels,
    )
    return playlist
