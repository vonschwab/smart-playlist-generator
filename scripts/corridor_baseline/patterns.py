"""Category F reporting-presence patterns + per-cell topology fingerprint extraction.

Corridor Phase 0a, Task 4. Every ``F_PATTERNS`` regex below was verified against the
actual production format string with ``grep -n`` at the cited site (not trusted from
the task-4 brief's starter set as-is) -- several starter regexes were wrong or
misleadingly marked ``conditional_on: None`` for lines that are in fact gated behind a
config knob that is OFF by default in this repo's config.yaml. Those corrections are
called out inline.

Notable corrections (see per-pattern comments for detail):
  - ``segment_header``: the starter set pointed at beam.py:580's
    ``"[Phase2] Segment %d→%d: ..."`` line, which only fires under
    ``dj_bridging_enabled`` -- a dataclass default of False (pier_bridge/config.py:232)
    that turns out to be OVERRIDDEN TO TRUE in THIS repo's config.yaml
    (``playlists.ds_pipeline.pier_bridge.dj_bridging.enabled: true``, wired by
    ``pier_bridge_overrides.py:500-502``), so it is in fact expected to fire in this
    corpus. Kept as an alternation with the truly config-independent per-segment line
    (pier_bridge_builder.py:1795-1796, ``"Building segment %d: %s -> %s (interior=%d)"``,
    inside the bare ``for seg_idx in range(num_segments):`` loop) anyway, so the
    checklist entry stays correctly ``None``/always-on even if a future config
    disables dj_bridging. The gated Phase2 variant also gets its own separate pattern
    (``phase2_dj_segment_header``) so its presence is tracked explicitly.
  - ``edge_repair_swap_log`` / ``edge_repair_applied``: same trap --
    ``edge_repair_enabled`` defaults False (pier_bridge/config.py:329) but config.yaml
    sets ``playlists.ds_pipeline.pier_bridge.edge_repair.enabled: true``
    (pier_bridge_overrides.py:239-242) -- expect PRESENT (subject to >=1 accepted
    swap at runtime).
  - ``roam_segments``: ``roam_corridors_enabled`` defaults False (pier_bridge/config.py:219)
    but config.yaml sets ``playlists.ds_pipeline.pier_bridge.roam.enabled: true``
    (mapped by ``roam_kwargs_from_dict``, pier_bridge/config.py:488-489) -- expect
    PRESENT (subject to a roam detour actually applying to >=1 edge at runtime).
  - ``solo_collab_split``: the starter marked this ``conditional_on: None``, but the
    log line (artist_style.py:742) is inside ``if include_collaborations:`` -- a
    ``create_playlist_for_artist`` param that defaults False and is never passed by
    ``runner.run_cell``. Expect ABSENT in this corpus. (Verified no config.yaml
    override exists for this one -- it is a call-site kwarg, not a config knob.)
  - ``sonic_admission_pct`` / ``bpm_admission_gate`` / ``onset_admission_band``: each
    gated behind ``sonic_mode``/``pace_mode`` != off. Both DETENTS use non-off values
    for these axes, so these are expected PRESENT in this corpus despite being
    genuinely conditional in general.
  - ``recency_edge_diff``: ``log_recency_edge_diff`` (reporter.py:67) is defined but
    never called anywhere in ``src/playlist_generator.py`` -- dead code. This pattern
    can never fire regardless of config; ``conditional_on`` documents that rather than
    naming a knob.

CAUTION for future edits: several ``pier_bridge.*`` booleans that default False in the
dataclass are flipped True by NESTED blocks in config.yaml (``dj_bridging: {enabled:
true}``, ``edge_repair: {enabled: true}``, ``roam: {enabled: true}``) rather than by a
flat ``<name>_enabled: true`` key -- a naive ``grep '<name>_enabled' config.yaml`` finds
nothing and looks like "default stands," which is wrong. Always trace the override
wiring in ``src/playlist/pipeline/pier_bridge_overrides.py`` /
``src/playlist/pier_bridge/config.py`` before trusting a config.yaml grep's silence.

EMPIRICAL RESULTS from the real 12-cell capture (docs/corridor_baseline/corpus_baseline.json,
2026-07-16), read from each cell's DS-success JSON payload's ``effective.playlist.pier_config``
(the actual resolved PierBridgeConfig used for that generation -- ground truth, not a
config.yaml read):
  - ``edge_repair_enabled`` and ``roam_corridors_enabled`` DO resolve True as the
    config.yaml override predicts (``edge_repair_swap_log``/``roam_segments`` fired;
    edge_repair 9/12 swap-log, 7/12 an accepted swap; roam 12/12).
  - ``dj_bridging_enabled`` resolves FALSE in the effective pier_config for all 12
    cells, despite config.yaml's ``dj_bridging.enabled: true``. pier_bridge_builder.py:
    511-515 has an explicit supersession (``genre_steering_enabled=True supersedes
    dj_bridging_enabled``) that logs when it fires and forces the field False -- but
    that log line ("...supersedes dj_bridging_enabled...") never appears in any of the
    12 logs either, meaning ``cfg.dj_bridging_enabled`` was apparently already False
    *before* that guard ran (the config.yaml override never took effect for the
    artist-mode/DS path in the first place, rather than being silently superseded).
    Functionally benign here (taxonomy steering is the live genre-arc method per
    CLAUDE.md Layer 3 item 16), but the config.yaml knob is misleading/dead for this
    path -- worth a follow-up "configured knob that can't act" investigation.
    ``phase2_dj_segment_header`` is correctly 0/12 (ABSENT) as a result.
  - ``emit_selected_edge_audit`` resolves TRUE in the effective pier_config for all 12
    cells (verified directly in the JSON payload), yet the "Selected-edge audit (N
    edges)" line NEVER appears in any of the 12 real logs -- 0/12, confirmed via
    ``docs/corridor_baseline/corpus_baseline.json``. This is a genuine, reproducible
    "configured knob that can't act" finding in ``create_playlist_for_artist``'s
    reporting tail (src/playlist_generator.py:2571-2588): the flag is true in
    ``pier_config`` but the diagnostic never fires. Root cause not isolated here (out
    of scope for this read-only capture task; ``t_mismatch_warning``, which is only
    ever invoked from inside ``emit_selected_edge_audit``, is consequently also 0/12)
    -- flagged for a follow-up fix task.

Corridor-scoped tooling: delete this module when the corridor contract closes
(see docs/corridor_baseline/README.md).
"""
from __future__ import annotations

import re
from typing import Any

F_PATTERNS: dict[str, dict[str, Any]] = {
    # ---- DS pipeline / reporter structural lines (always fire on the DS path) ----
    "ds_success_json": {
        "regex": r'"pipeline": "ds"',
        "conditional_on": None,
    },  # src/playlist/ds_pipeline_runner.py:200 (json.dumps(payload))

    "transition_metrics_json": {
        "regex": r'"min_transition"',
        "conditional_on": None,
    },  # src/playlist/ds_pipeline_runner.py:183, metrics dict embedded in the DS-success payload

    "candidate_pool_tally": {
        "regex": r"Candidate pool: mode=",
        "conditional_on": None,
    },  # src/playlist/candidate_pool.py:1610

    "playlist_statistics_header": {
        "regex": r"PLAYLIST STATISTICS",
        "conditional_on": None,
    },  # src/playlist/reporter.py:601

    "edge_score_summaries": {
        "regex": r"Edge score summaries:",
        "conditional_on": None,
    },  # src/playlist/reporter.py:704 -- requires last_ds_report+edge_scores, always true on the DS path

    "transition_pctiles": {
        "regex": r"T transition: mean=",
        "conditional_on": None,
    },  # src/playlist/reporter.py:751 -- corrected from the starter's bare "min_transition"
        # substring (which only matched the JSON payload, now its own
        # transition_metrics_json pattern above) to the actual human-readable line

    "sonic_transition_stats": {
        "regex": r"S sonic:\s*mean=",
        "conditional_on": None,
    },  # src/playlist/reporter.py:812-813

    "genre_transition_stats": {
        "regex": r"G genre:\s*mean=",
        "conditional_on": None,
    },  # src/playlist/reporter.py:822-823

    "bpm_playlist_summary": {
        "regex": r"BPM \(perceptual\): min=",
        "conditional_on": None,
    },  # src/playlist/reporter.py:838, bpm_summary computed unconditionally in
        # pier_bridge_builder.py:3092 whenever >=1 track has BPM data

    "weakest_edges": {
        "regex": r"Weakest transitions \(bottom 3 by T\):",
        "conditional_on": None,
    },  # src/playlist/reporter.py:872 -- corrected from starter's bare "[Ww]eakest" to the
        # exact wording

    "segment_header": {
        "regex": r"(\[Phase2\] Segment \d+|Building segment \d+: .*?\(interior=\d+\))",
        "conditional_on": None,
    },  # See module docstring: alternation of the gated beam.py:580 Phase2 line (kept so
        # the brief's prescribed literal-string test still matches) and the truly
        # unconditional pier_bridge_builder.py:1795-1796 line. extract_fingerprint below
        # parses interior_lengths from the "Building segment" clause specifically, since
        # that is the one guaranteed to fire.

    "phase2_dj_segment_header": {
        "regex": r"\[Phase2\] Segment \d+",
        "conditional_on": "dj_bridging_enabled -- config.yaml sets "
                           "playlists.ds_pipeline.pier_bridge.dj_bridging.enabled: true, but the "
                           "EMPIRICAL 12-cell capture shows it resolves False in the effective "
                           "pier_config for every cell (verified in the DS-success JSON payload) "
                           "-- confirmed ABSENT (0/12); see module docstring 'EMPIRICAL RESULTS'",
    },  # src/playlist/pier_bridge/beam.py:580, gated by `if cfg.dj_bridging_enabled and interior_length > 0:`

    # ---- artist-style clustering ----
    "solo_collab_split": {
        "regex": r"solo=\d+ collab=\d+",
        "conditional_on": "include_collaborations=True (create_playlist_for_artist param, default "
                           "False, never passed by runner.run_cell -- expect ABSENT)",
    },  # src/playlist/artist_style.py:742, inside `if include_collaborations:` (line 738)
        # CORRECTED: starter marked this conditional_on: None

    "pier_bridgeability_gate": {
        "regex": r"Pier bridgeability genre gate:",
        "conditional_on": "pier_bridgeability_enabled (default True in artist_style.py, not "
                           "overridden in config.yaml -- expect PRESENT)",
    },  # src/playlist/artist_style.py:779-783

    "pier_veto": {
        "regex": r"Pier bridgeability: vetoed",
        "conditional_on": "fires only when >=1 medoid candidate fails floor_t under the "
                           "(default-True) bridgeability gate",
    },  # src/playlist/artist_style.py:805 -- CORRECTED regex from starter's bare
        # "[Bb]ridgeability" (too broad -- also matches the always-on gate header above)

    # ---- candidate-pool admission gates ----
    "sonic_admission_pct": {
        "regex": r"Sonic admission percentile active:",
        "conditional_on": "sonic_mode != off (percentile>0); both home (strict) and open "
                           "(dynamic) detents satisfy this -- expect PRESENT",
    },  # src/playlist/candidate_pool.py:765

    "bpm_admission_gate": {
        "regex": r"BPM admission gate:",
        "conditional_on": "pace_mode != off (finite bpm_admission_max_log_distance); both "
                           "detents use pace_mode=dynamic -- expect PRESENT",
    },  # src/playlist/candidate_pool.py:806

    "onset_admission_band": {
        "regex": r"Onset admission band:",
        "conditional_on": "pace_mode != off (finite onset_admission_max_log_distance); both "
                           "detents use pace_mode=dynamic -- expect PRESENT",
    },  # src/playlist/candidate_pool.py:831

    "duration_penalty": {
        "regex": r"Duration penalty applied:",
        "conditional_on": "DEBUG log level (run_cell always sets this) AND >=1 candidate "
                           "actually duration-penalized at runtime",
    },  # src/playlist/candidate_pool.py:691, gated by
        # `if duration_penalty_count and logger.isEnabledFor(logging.DEBUG):`

    "min_pool_backstop": {
        "regex": r"Min-pool backstop:",
        "conditional_on": "fires only when the admitted pool falls below min_pool_size at runtime",
    },  # src/playlist/candidate_pool.py:1375

    "instrumental_lean": {
        "regex": r"Instrumental-lean pool demotion applied:",
        "conditional_on": "instrumental_enabled (default False, not set in config.yaml -- expect ABSENT)",
    },  # src/playlist/candidate_pool.py:714

    "tag_steering_pool": {
        "regex": r"Tag steering",
        "conditional_on": "tag chips set (artist-mode single-seed runs never set tag chips -- expect ABSENT)",
    },  # src/playlist/candidate_pool.py:745/883/931/957/1413/1650

    "roam_segments": {
        "regex": r"Roam\[seg \d+\]:",
        "conditional_on": "roam_corridors_enabled (dataclass default False, but config.yaml sets "
                           "playlists.ds_pipeline.pier_bridge.roam.enabled: true -- expect PRESENT, "
                           "subject to a roam detour actually applying to >=1 edge at runtime)",
    },  # src/playlist/pier_bridge_builder.py:1573

    # ---- pier-bridge endgame passes ----
    "tail_dp_summary": {
        "regex": r"Tail-DP summary: applied=",
        "conditional_on": "tail_dp_enabled (default True in pier_bridge/config.py, not "
                           "overridden -- expect PRESENT)",
    },  # src/playlist/pier_bridge_builder.py:2877

    "tail_dp_applied": {
        "regex": r"Tail-DP seg \d+: window min",
        "conditional_on": "fires only when tail-dp actually swaps a weak landing edge "
                           "(tail_dp_enabled=True is necessary but not sufficient)",
    },  # src/playlist/pier_bridge_builder.py:2553

    "mini_pier": {
        "regex": r"Mini-piers: \d+ waypoint",
        "conditional_on": "mini_pier_enabled=true (set True in config.yaml, default False in "
                           "pier_bridge/config.py) AND >=1 waypoint actually inserted",
    },  # src/playlist/pier_bridge_builder.py:904 -- CORRECTED starter regex "[Mm]ini[- ]pier"
        # (too broad -- also matches unrelated config-key/comment mentions of "mini_pier")

    "edge_repair_swap_log": {
        "regex": r"Edge repair swap log",
        "conditional_on": "edge_repair_enabled (dataclass default False, but config.yaml sets "
                           "playlists.ds_pipeline.pier_bridge.edge_repair.enabled: true -- expect "
                           "PRESENT, subject to the swap log being non-empty at runtime)",
    },  # src/playlist/reporter.py:205, only emitted when the swap log is non-empty

    "edge_repair_applied": {
        "regex": r"Repair edge=",
        "conditional_on": "edge_repair_enabled=true (set in config.yaml) AND >=1 accepted repair "
                           "swap at runtime -- expect present in at least some cells",
    },  # src/playlist/reporter.py:211-222 -- CORRECTED starter regex "[Ee]dge repair"
        # (too broad; also matched comments/config-key mentions elsewhere in the codebase)

    # ---- opt-in edge audit (emit_selected_edge_audit: true in config.yaml:219) ----
    "selected_edge_audit": {
        "regex": r"Selected-edge audit \(\d+ edges\)",
        "conditional_on": "emit_selected_edge_audit=true -- SET True in config.yaml:219 AND "
                           "confirmed True in the effective pier_config for every cell (verified in "
                           "the DS-success JSON payload), yet EMPIRICALLY ABSENT (0/12) in the real "
                           "capture -- a genuine 'configured knob that can't act' finding in "
                           "create_playlist_for_artist's reporting tail (playlist_generator.py:2571-"
                           "2588); see module docstring 'EMPIRICAL RESULTS'. Root cause not isolated "
                           "here (read-only capture task) -- flagged for a follow-up fix.",
    },  # src/playlist/reporter.py:151

    "t_mismatch_warning": {
        "regex": r"T-mismatch edge",
        "conditional_on": "emit_selected_edge_audit=true AND a broken edge where beam T and final T "
                           "disagree beyond tolerance exists at runtime -- consequently also 0/12 "
                           "since it is only ever invoked from inside emit_selected_edge_audit (see "
                           "selected_edge_audit above)",
    },  # src/playlist/reporter.py:120-124, called from within emit_selected_edge_audit

    # ---- verbose_edges-gated diagnostics (verbose param, default False; run_cell
    # never passes verbose=True to create_playlist_for_artist) ----
    "chain_degree_summary": {
        "regex": r"Chain index-degree summary:",
        "conditional_on": "verbose_edges=True (create_playlist_for_artist verbose param, "
                           "default False, not passed by runner.run_cell -- expect ABSENT)",
    },  # src/playlist/reporter.py:696

    "baseline_artifact_percentiles": {
        "regex": r"Baseline \(artifact\) percentiles:",
        "conditional_on": "verbose_edges=True (default False, not passed by runner.run_cell -- expect ABSENT)",
    },  # src/playlist/reporter.py:675

    # ---- dead code (documented finding, not a pattern that can ever fire) ----
    "recency_edge_diff": {
        "regex": r"Recency adjacency diag:",
        "conditional_on": "DEAD CODE -- log_recency_edge_diff (reporter.py:67) is defined but "
                           "never called anywhere in src/playlist_generator.py; this pattern can "
                           "never fire regardless of config (finding, not a fix -- out of scope "
                           "for this task)",
    },  # src/playlist/reporter.py:67-98
}


# ---- auxiliary regexes for the topology fingerprint (not part of the F-checklist) ---
# These parse counts/values out of the log; they intentionally target only the
# unconditional clause of each line (documented above) so counts are reliable across
# all 12 cells regardless of which optional passes fired.
_SEGMENT_BUILD_RE = re.compile(r"Building segment (\d+): .*?\(interior=(\d+)\)")
_MINI_PIER_RE = re.compile(r"Mini-piers: (\d+) waypoint")
_TAIL_DP_APPLIED_RE = re.compile(r"Tail-DP seg \d+: window min")
_EDGE_REPAIR_APPLIED_RE = re.compile(r"Repair edge=")
_SOLO_COLLAB_RE = re.compile(r"solo=\d+ collab=\d+")


def extract_fingerprint(log_text: str, run: dict) -> dict:
    """Per-cell topology + Category F reporting-presence fingerprint.

    ``run`` is the dict returned by ``runner.run_cell`` (or a copy the caller has
    stamped with "artist"/"detent" -- run_cell itself does not know either).
    """
    seg_matches = _SEGMENT_BUILD_RE.findall(log_text)
    interior_lengths = [int(interior) for _seg_idx, interior in seg_matches]
    solo_collab_match = _SOLO_COLLAB_RE.search(log_text)

    track_ids = list(run.get("track_ids") or [])
    reporting_presence = {
        name: bool(re.search(spec["regex"], log_text))
        for name, spec in F_PATTERNS.items()
    }

    return {
        "artist": run.get("artist"),
        "detent": run.get("detent"),
        "n_tracks": len(track_ids),
        "track_ids": track_ids,
        "min_transition": run.get("min_transition"),
        "mean_transition": run.get("mean_transition"),
        "below_floor": run.get("below_floor"),
        "distinct_artists": run.get("distinct_artists"),
        "admitted": run.get("admitted"),
        "wall_s": run.get("wall"),
        "segments": len(seg_matches),
        "interior_lengths": interior_lengths,
        "mini_pier_mentions": len(_MINI_PIER_RE.findall(log_text)),
        "tail_dp_fired": len(_TAIL_DP_APPLIED_RE.findall(log_text)),
        "edge_repair_fired": len(_EDGE_REPAIR_APPLIED_RE.findall(log_text)),
        "solo_collab": solo_collab_match.group(0) if solo_collab_match else None,
        "reporting_presence": reporting_presence,
        "err": run.get("err"),
        "log_path": run.get("log_path"),
    }
