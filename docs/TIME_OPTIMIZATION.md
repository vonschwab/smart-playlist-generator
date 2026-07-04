# Playlist generation — time optimization (DEFERRED backlog)

**Current stance (Dylan, 2026-07-02): QUALITY FIRST. Time optimization is deferred.**
We tune the generator to its highest quality first; only once quality is dialed in do we
come back and optimize for wall-clock. The **90s generation ceiling**
(`feedback_generation_time_budget`: "a playlist NEVER >90s") is **temporarily suspended**
for this quality-tuning phase — full-strength search that overruns 90s is acceptable *for now*.
This doc is the backlog of what to cut, and by how much, when we re-arm the ceiling.

Do not silently shrink search to save time during the quality phase. Record the cost here
instead, and cut it deliberately later.

---

## LOSSLESS wins landed (2026-07-03)

A separate, orthogonal track: **bit-identical** speedups (same track_ids, same order), golden-diff
gated — no search shrinkage, so these compose with the (still-suspended) ceiling. Spec/plan:
`docs/superpowers/specs/2026-07-03-lossless-generation-speedup-design.md`. Measured by
**load-independent cProfile** — single-run wall-clock is unusable here (a concurrent session's
generations swing it 60→100s; trust the profiler + the bit-diff, not wall-clock).

Porches golden replay, profiled total **377.5s → 117.5s (3.2×)**:

| Hotspot | Baseline | After | Change |
|---------|----------|-------|--------|
| `resolve_artist_identity_keys` | 2.35M calls / 230.5s | 128K / 12.0s | **T1-a** — cache the parse per candidate (16477ff) |
| transition scoring (`score_transition_edge`) | 1.22M / 24.2s | 636K / 11.1s | **T1-e** — memoize by (prev,cur); diamonds ~halve it (2394637) |
| `compute_step_log_*_target` | ~15s | out of top-30 | **T1-c** — hoist per-step out of the candidate loop (03cbe4a) |
| `compute_energy_pace_penalty` | per-candidate | skipped | **T1-d** — skip when energy weights 0 (2394637) |

Foundation (golden capture + bit-diff gate + timing/profile harness + Porches fixture + absolute-path
overrides so the gate runs from a worktree): `d087b69 d0024cc ca90825 4efaddb`. Baseline/after profiles:
`docs/run_audits/lossless_speedup_baseline_profile.txt`, `..._after_t1_profile.txt`.

**Biggest remaining lossless levers** (not yet done):
- `bpm_log_distance` still **4M calls / 31.7s** — scalar `math` fast-path (plan T2-c, Tier-2 float-reassoc,
  must pass ΔT==0). Now the #1 remaining function-time cost after the beam's own body.
- **T1-g** — hoist the flex re-run invariant work (pool / genre-route / roam): the profile shows **15
  beam calls for 9 segments** (6 flex re-runs), `choose_segment_length` at 86.8s cumulative. Touches
  `pier_bridge_builder.py` (god-class) — do in an isolated worktree, see [[feedback_worktree_data_absolute_overrides]].
- `edge_repair` (post-loop) is 21.4s — out of the beam, a separate future target.

---

## Lever #1 (headline): artist-path beam + pooling width

**What.** The pier-bridge beam explores `beam_width` candidate paths per segment, pulling from
`neighbors_m` nearest neighbours and `bridge_helpers` connector candidates. Wider = more paths
explored = (hypothesis) better worst-edge, at roughly linear-to-superlinear time cost.

**Current state (2026-07-02, branch `fix/artist-pier-config-restore-tuning`).** The artist-style
path was accidentally running the beam at HALF config width for months — the hand-built
`PierBridgeConfig` omitted the six beam/pool fields, so they fell to dataclass defaults
(`20/100` beam, `100/400` neighbours, `50/200` helpers) instead of config's `40/200`, `200/800`,
`100/400`. Fixed by threading them through `_build_artist_pier_config`
(`src/playlist_generator.py`). See `docs/CLEANUP_LIST.md` → "Artist-style path drops
resolved-tuning fields".

**Measured cost of full width.** A 50-track Porches artist run:

| Beam config | Build time | Source |
|-------------|-----------|--------|
| half (`20/100`, accidental) | ~72s | 2026-07-02_172834 (GUI) |
| **full (`40/200`, config)** | **~151s** | 2026-07-02_180247 (CLI, medoid piers) |

≈2× the wall-clock. This is the single biggest known time cost in generation, and the primary
dial when we optimize. **Quality delta of the wider beam is NOT yet measured** — the CLI run
picks medoid piers, not the GUI's popular-seed piers, so it isn't comparable to the half-width
GUI runs. Measure worst-edge (min-T) full vs half through the GUI on an identical seed before
deciding how far to walk the beam back down.

**When optimizing, the dials (roughly cheapest→most invasive):**
1. **Cheaper var-bridge flex retries (Lever #2 below) — do this first.** Much of the 151s is
   flex re-running whole segments at full beam, not the base beam itself.
2. **Decouple pool breadth from search width.** `neighbors_m` / `bridge_helpers` (how many
   candidates enter the pool) may buy quality more cheaply than `beam_width` (how many partial
   paths the beam keeps). Sweep them independently; a wide pool + moderate beam may be the
   sweet spot under a restored ceiling.
3. **Scale beam to segment difficulty.** Full width only where the segment is hard (long
   interior, sparse pool, weak landing); narrow it where the pool is easy.
4. **Early-exit on beam convergence** (stop widening once the top-k stabilises).
5. **Parallelise segment builds** (segments are largely independent given fixed piers).

---

## Lever #2: var-bridge flex retries re-run whole segments at full beam

The variable-bridge "flex" re-runs a segment's beam at a different interior length to lift a weak
landing. Each retry is a **full-beam segment build**, so at full width the retries multiply the
cost (the 2026-07-02_180247 run flexed 4 segments — a large share of its 151s). A flex-retry that
used a *narrower* beam than the base pass (it only needs to compare a few length options, not
find the globally-best path) would cut this sharply without shrinking the base beam. Highest-ROI
time lever after the base beam itself.

## Lever #3: `generation_budget_s` is inert (0.0)

The soft-deadline knob is `0.0` in the effective config → disabled. It was wired in
(`pier_bridge_overrides.py`, the `generation_budget_s` cast) so the beam *can* honour a soft
deadline + relaxation cap, but at 0.0 nothing fires. When we re-arm the 90s ceiling this is the
enforcement point — set it to the target and verify the beam actually bails/relaxes on overrun
(the "a configured knob that can't act is a startup error" gotcha applies: confirm it fires).

## Observed overruns (baseline data for later)

- **50-track Porches, full artist beam: ~151s** (2026-07-02_180247).
- **50-track multi-seed: ~127s** (2026-07-02_174010 seeds; ~10s/segment × 9 + 3× var-bridge
  re-runs on the last segment). See `docs/CLEANUP_LIST.md` → "Fixer-cascade / reporting warts".
- **51-track Herbie Hancock artist: ~85s** — just under, at the OLD half beam.

These will all rise once full-width beam lands everywhere. Re-measure after the quality phase.

---

## Cross-references
- `docs/CLEANUP_LIST.md` — the beam-width drop item (root cause + fix) and the fixer-cascade 90s items.
- Memory `feedback_generation_time_budget` — the (currently suspended) 90s hard ceiling.
- `src/playlist_generator.py::_build_artist_pier_config` — where the artist-path beam widths are set.
