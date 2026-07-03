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
