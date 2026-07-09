# Bridge-side Phase A — surface on-tag tracks as bridges (design spec)

**Date:** 2026-07-09. **Status:** approved design, ready to plan. **Branch:** feat/tag-steering-bridge-side.
**Prior art (read first):** `docs/superpowers/specs/2026-07-08-tag-steering-architecture.md` (the gate map;
this is the fix for its **stage D**). Backed by three research passes (2026-07-09): the stage-D drop point,
the waypoint/anchor machinery, and an external-literature review — findings folded in below.

## Problem

On-tag but sonically-peripheral tracks (Ghost Box for BoC+hauntology; canonical krautrock for
Bowie+krautrock) never reach the beam. **Root cause (pinned):** `src/playlist/segment_pool_builder.py`
`_compute_bridge_scores` (~line 541) gates bridge admission on `min(sim_a, sim_b) < bridge_floor`, where
`sim_a`/`sim_b` are **raw MuQ cosine to BOTH piers** (~522-523). Genre similarity
(`segment_pool_genre_weight`) is applied *after*, only to RANK survivors — never to admission. A track
smooth to *one* pier (calibrated T=0.87) but far from the other is cut by `min()`. This is why stage C
(pool guarantee) and stage E (beam term + worst-edge band) are inert: they operate on candidates that never
arrive at the segment pool.

**External framing (research):** a DJ "bridge track" is a stepping-stone that *carries the feel of the
outgoing track, then moves toward the next* — it should be scored against the **trajectory between piers**,
not required to sit near both. And genre steering should be a **weighted term, never a gate** (a KG+RL
ablation found pure-smoothness optimization degenerates into smooth-but-*irrelevant* playlists).

## Scope

**Phase A makes on-tag tracks REACHABLE as bridges** (interior fill). **Phase B** (anchor mode — force K
on-tag tracks as *piers* via the already-live, ungated `anchor_seed_ids` path) is **deferred**, to be built
only if Phase A's results show the extreme case (Ghost Box) still needs a hard guarantee.

## Design — three coordinated changes, ALL gated on tag steering active

Non-steered playlists must be **byte-identical** to today.

### Change 1 — relaxed bridge admission (`min` → `max`)
When tag steering is active, segment-pool admission uses `max(sim_a, sim_b) ≥ bridge_floor` instead of
`min(...)`: a candidate near *either* pier is an eligible stepping-stone. The beam's sequential worst-edge +
`eta_destination_pull` + progress terms enforce actual bridge quality downstream, so the pool gate must not
pre-exclude one-sided candidates.
- **Location:** `segment_pool_builder.py::_compute_bridge_scores` (~541). Add `bridge_admission_relaxed:
  bool = False` to `SegmentPoolConfig`; `min(...)` when False (default/legacy), `max(...)` when True.
- **Wired True** from `pier_bridge_builder.py` only when steering is active.
- Chosen over (a) on-tag-bypass-only (leaves the flawed gate for normal bridging) and (b) kNN-path/trajectory
  admission (more principled per the geodesic/low-density-hole finding, but bigger — a Phase-A+ refinement).

### Change 2 — on-tag membership guarantee at stage D
Force-include the top on-tag tracks per segment past the admission floor, capped + per-artist-limited,
membership-keyed. Backstops Change 1 for on-tag tracks far from *both* piers. Reuses the guarantee-id set
already computed once per run (`pipeline/core.py` ~636-641, `_on_tag_guarantee_ids`, from
`resolve_tag_sonic_prototype_rows` = authority on-tag membership). **The missing wire:** `core.py` ~992-1016
passes `allowed_track_ids_set` into `build_pier_bridge_playlist` but NOT the guarantee ids.
- Thread `on_tag_guarantee_ids: Optional[set[str]]`, `on_tag_segment_guarantee_max: int`,
  `on_tag_segment_guarantee_per_artist: int` into `build_pier_bridge_playlist`
  (`pier_bridge_builder.py:437`), passed from `core.py` reusing the existing `_on_tag_guarantee_ids` /
  `_guar_max`-style locals.
- Resolve to a per-run `Set[int]` (same pattern as `allowed_set_indices`, `pier_bridge_builder.py:806-814`);
  pass into each `SegmentPoolConfig`.
- In `_compute_bridge_scores` (~536-553): for candidates whose index ∈ guarantee set, **bypass the
  admission floor** (force into `passing`), still computing the ranking score. Cap per segment.
- In `_select_final_candidates` (~951-1043): **priority-insert** guarantee tracks ahead of the
  `segment_pool_max` truncation (mirroring `internal_connectors`), so the sonic-ranked Phase-4 cut doesn't
  re-drop them.
- **New knobs:** `tag_steering_segment_guarantee_max: 8` (per segment), `tag_steering_segment_guarantee_per_artist: 2`.
- No `dj_union` changes needed: `dj_bridging_enabled` is force-disabled whenever `genre_steering_enabled`
  (the live default) is on, so the live path is `segment_scored` only. (If `dj_union` is ever re-enabled,
  the guarantee must also be unioned into its `available` set — noted, not built.)

### Change 3 — activate the beam term + worst-edge band
`tag_steering_sonic_beam_weight` and `tag_steering_worst_edge_band` (both built, both 0.0 today) get
**calibrated nonzero defaults**. Now that on-tag tracks reach the beam (Changes 1-2), these let the beam
*prefer* them while trading only a bounded sliver of worst-edge (the band relaxes the minimax tie-break;
see architecture doc §2 stage E). They already only act when steering is active (`_sonic_tag_active` gate,
`beam.py:1027`). **Calibrate during validation** (trial weight 0.5–2.0, band 0.05–0.15); ship the value that
surfaces on-tag bridges without worst-edge dropping more than ~one notch below the pre-change playlist.
Keep genre a *weighted term*, never a gate (research; project Layer-4 discipline).

## Config (`playlists.ds_pipeline.pier_bridge`)

- NEW `tag_steering_segment_guarantee_max: 8`, `tag_steering_segment_guarantee_per_artist: 2` — stage-D guarantee.
- NEW `tag_steering_relax_bridge_admission: true` — Change 1 live default; `false` = legacy `min()` rollback.
- (existing, now set to calibrated values) `tag_steering_sonic_beam_weight`, `tag_steering_worst_edge_band`.

Every knob has a documented rollback; the missing-data path must warn/raise, never silently no-op
(project gotcha). All effects gated on tag steering active.

## Testing

- **Unit:** `_compute_bridge_scores` admits a one-pier-close candidate under `relaxed=True`, rejects it under
  `False` (byte-identical legacy); stage-D guarantee force-includes an on-tag id past the floor and it
  survives `_select_final_candidates` truncation; per-segment + per-artist caps honored.
- **Integration (real `PlaylistGenerator`, per the pier-fix precedent — `generate_like_gui` only reaches
  seeds mode):**
  - BoC + hauntology: the playlist contains ≥ N non-BoC authority-hauntology bridges (the payoff). If 0,
    assert-xfail pointing to Phase B — never a silent pass.
  - Bowie + krautrock: more authority-krautrock bridges, fewer off-genre; worst-edge ≥ pre-change − ~1 notch.
  - Non-steered artist (no tag): pool + playlist byte-identical (rollback guard on Change 1).
  - A clean case (Eno/neoclassical or Real Estate/jangle): no regression (guarantee ≈ no-op there).
- **Manual predict-then-check:** regenerate BoC/hauntology + Bowie/krautrock through the worker path; count
  on-genre bridges; quote worst-edge vs the pre-change runs.

## Acceptance

BoC+hauntology contains Ghost Box / non-BoC hauntology bridges (target ≥2–3); Bowie+krautrock more on-genre;
worst-edge within ~one notch of pre-change; non-steered byte-identical; clean cases unaffected. **If BoC
still shows ~0 on-tag bridges, that is the decisive evidence that Phase B (anchors) is required** — recorded,
not hidden.

## Out of scope

Phase B (anchor / "compose toward genre" mode — force on-tag tracks as piers via `anchor_seed_ids`).
kNN-path/trajectory admission (a refinement of Change 1). Learned per-axis MuQ decomposition (research: the
field's answer is *learned contrastive* disentanglement, not hand-built towers — which this project already
walked back; do not revisit here).
