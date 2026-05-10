"""Post-order validation + failure diagnostic messages.

Extracted from pipeline.core.generate_playlist_ds (Tier-1.5 split).

Two responsibilities:
  * build_failure_diagnostic: classifies a pier-bridge failure into one of
    three user-facing root-cause buckets (genre isolation, sonic isolation,
    insufficient pool) and returns the diagnostic message + raw counters
    for audit emission.
  * run_post_order_validation: enforces the post-order DS contract — length
    must match, no recency-excluded ids may appear (except piers).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.features.artifacts import ArtifactBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolDiagnostics:
    """Bottleneck classification numbers from the candidate-pool stats."""
    admitted: int
    rejected_sonic: int
    rejected_genre: int
    total_considered: int


@dataclass(frozen=True)
class FailureDiagnostic:
    """User-facing failure message + structured counters for the audit."""
    failure_reason: str
    diagnostic_msg: str
    pool_diagnostics: PoolDiagnostics


def build_failure_diagnostic(
    pool_stats: Dict[str, Any],
    pb_failure_reason: Optional[str],
    *,
    pool_indices_count: int,
    seed_track_ids_for_pier_count: int,
    cfg_mode: str,
    cfg_genre_gate_min_similarity: float,
) -> FailureDiagnostic:
    """Classify a pier-bridge failure and produce the user-facing message.

    Three root-cause buckets, in priority order:
      1. GENRE ISOLATION — admitted=0 and any tracks were rejected by genre
      2. SONIC ISOLATION — admitted=0 and any tracks were rejected by sonic
      3. INSUFFICIENT CANDIDATE POOL — 0 < admitted < 10
    Otherwise the raw failure reason is returned unmodified.
    """
    failure_reason = str(pb_failure_reason or "pier-bridge failed")

    admitted = int(pool_stats.get("pool_size", pool_indices_count))
    rejected_sonic = int(pool_stats.get("below_sonic_similarity", 0))
    rejected_genre = int(pool_stats.get("below_genre_similarity", 0))
    total_considered = int(pool_stats.get("total_candidates_considered", 0))

    diagnostic_msg = failure_reason
    if admitted == 0 and rejected_genre > 0:
        # Genre gating eliminated all candidates
        passed_sonic = total_considered - rejected_sonic if total_considered > rejected_sonic else 0
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: GENRE ISOLATION\n"
            f"   - {passed_sonic} tracks passed sonic similarity\n"
            f"   - ALL {rejected_genre} were rejected by genre filter (min_similarity={cfg_genre_gate_min_similarity:.2f})\n"
            f"   - Zero candidates remained for playlist generation\n\n"
            f"💡 This artist's genres are too isolated from your library.\n"
            f"   Suggestions:\n"
            f"   - Reduce genre mode from {cfg_mode} to 'narrow' or 'dynamic'\n"
            f"   - Add more music in similar genres to your library\n"
            f"   - Use genre mode 'discover' to disable hard genre filtering"
        )
    elif admitted == 0 and rejected_sonic > 0:
        # Sonic similarity eliminated all candidates
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: SONIC ISOLATION\n"
            f"   - {rejected_sonic} of {total_considered} tracks rejected by sonic floor\n"
            f"   - Zero candidates remained for playlist generation\n\n"
            f"💡 This artist's sound is too sonically isolated from your library.\n"
            f"   Suggestions:\n"
            f"   - Reduce sonic mode from {cfg_mode} to 'narrow' or 'dynamic'\n"
            f"   - Add more sonically similar music to your library"
        )
    elif admitted > 0 and admitted < 10:
        # Small pool passed filters but not enough for bridging
        diagnostic_msg = (
            f"{failure_reason}\n\n"
            f"🔍 Root cause: INSUFFICIENT CANDIDATE POOL\n"
            f"   - Only {admitted} candidates passed all filters\n"
            f"   - Rejected: {rejected_sonic} sonic, {rejected_genre} genre\n"
            f"   - Need more candidates to bridge between {seed_track_ids_for_pier_count} seeds\n\n"
            f"💡 The candidate pool is too small for multi-seed bridging.\n"
            f"   Suggestions:\n"
            f"   - Use fewer seeds (1-2 instead of {seed_track_ids_for_pier_count})\n"
            f"   - Reduce filtering: set genre/sonic modes to 'discover'\n"
            f"   - Choose seeds that are more sonically/genre similar to each other"
        )

    return FailureDiagnostic(
        failure_reason=failure_reason,
        diagnostic_msg=diagnostic_msg,
        pool_diagnostics=PoolDiagnostics(
            admitted=admitted,
            rejected_sonic=rejected_sonic,
            rejected_genre=rejected_genre,
            total_considered=total_considered,
        ),
    )


@dataclass(frozen=True)
class PostOrderValidation:
    """Result of post-order validation — both summary stats and any errors."""
    summary: Dict[str, int]
    errors: List[str]
    recency_overlap_ids: List[str]


def run_post_order_validation(
    *,
    bundle: ArtifactBundle,
    ordered_track_ids: List[str],
    expected_length: int,
    excluded_track_ids: Optional[set],
    seed_track_ids_for_pier: List[str],
) -> PostOrderValidation:
    """Run the post-order DS contract checks.

    Two checks:
      * Length must match ``expected_length`` (when expected_length > 0).
      * No track in ``excluded_track_ids`` may appear (piers are exempt).

    Returns a ``PostOrderValidation`` carrying both the summary dict (for
    audit emission) and the list of human-readable error strings. The
    caller is responsible for raising once errors exist.
    """
    excluded_ids_set = {str(t) for t in (excluded_track_ids or set())}
    pier_ids_set = {str(t) for t in seed_track_ids_for_pier}
    recency_overlap_ids = [
        tid for tid in ordered_track_ids if (tid in excluded_ids_set and tid not in pier_ids_set)
    ]
    summary = {
        "recency_overlap_count": int(len(recency_overlap_ids)),
        "final_size": int(len(ordered_track_ids)),
        "expected_size": int(expected_length),
    }

    errors: List[str] = []
    if expected_length > 0 and len(ordered_track_ids) != expected_length:
        errors.append(
            f"length_mismatch final={len(ordered_track_ids)} expected={expected_length}"
        )

    if recency_overlap_ids:
        offenders: List[str] = []
        for tid in recency_overlap_ids[:10]:
            idx = bundle.track_id_to_index.get(str(tid))
            if idx is None:
                offenders.append(str(tid))
                continue
            artist = ""
            title = ""
            try:
                if bundle.track_artists is not None:
                    artist = str(bundle.track_artists[int(idx)])
                elif bundle.artist_keys is not None:
                    artist = str(bundle.artist_keys[int(idx)])
                if bundle.track_titles is not None:
                    title = str(bundle.track_titles[int(idx)])
            except Exception:
                artist = ""
                title = ""
            offenders.append(f"{tid} ({artist} - {title})")
        errors.append(f"recency_overlap={len(recency_overlap_ids)} offenders={offenders}")

    return PostOrderValidation(
        summary=summary,
        errors=errors,
        recency_overlap_ids=recency_overlap_ids,
    )
