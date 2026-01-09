DJ Bridging Relaxation + Micro-Pier Audit
=========================================

Step 0: locate existing failure modes, relaxation, and micro-pier logic.

1) Segment failure modes + propagation
- Beam search failure:
  - src/playlist/pier_bridge_builder.py::_beam_search_segment returns
    (path, genre_penalty_hits, edges_scored, failure_reason).
  - Direct transition below floor returns failure_reason string.
- Segment failure handling:
  - build_pier_bridge_playlist loop sets last_failure_reason from
    _beam_search_segment or pool gating.
  - On failure, returns PierBridgeResult(success=False, failure_reason=...).
  - Run audit captures segment_success / segment_failure events with
    failure_reason and attempted bridge floors.
- "Below floor" counts:
  - Transition floor violations computed in pipeline using edge_scores
    (playlist_stats["below_floor_count"]).

2) Existing relaxation/backoff logic
- Infeasible handling (default OFF):
  - src/playlist/pier_bridge_builder.py::_bridge_floor_attempts uses
    InfeasibleHandlingConfig (from src/playlist/run_audit.py).
  - On backoff attempts, can widen search:
    - increase pool (extra_neighbors_m/extra_bridge_helpers)
    - increase beam width
    - increase max_expansion_attempts
  - Also relaxes seed-artist-in-interiors constraint when allowed in config.
- This is generic pier-bridge backoff, not DJ-specific.

3) Existing micro-pier logic
- _attempt_micro_pier_split in src/playlist/pier_bridge_builder.py:
  - Selects micro candidates by max-min bridge score (min(sim(A,C), sim(C,B))).
  - Splits into A->C and C->B subsegments.
  - Emits warning type "micro_pier_used".
- Trigger: only if dj_micro_piers_enabled AND segment_allow_detours
  (segment_far_stats threshold + dj_allow_detours_when_far).

4) Existing connector bias hooks
- DJ connectors injected into segment pool when segment_allow_detours.
- Selected via _select_connector_candidates; recorded in pool diagnostics.

Planned reuse / minimal plumbing
- Reuse:
  - Existing backoff loop in build_pier_bridge_playlist.
  - _attempt_micro_pier_split selection metric.
  - Warnings list + run audit events.
  - connector bias hooks for candidate source.
- Add:
  - New DJ-only relaxation sequence gated by
    dj_bridging.relaxation.enabled (separate from infeasible_handling).
  - Diagnostics for each relaxation attempt.
  - Micro-pier fallback only after relaxation attempts are exhausted and
    gated by new dj_bridging.micro_piers.* config (not the current far-only trigger).
  - Minimal configuration plumbing; defaults remain unchanged.
