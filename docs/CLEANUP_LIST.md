# Cleanup list — parked / deferred items

Running list of built-but-parked, superseded, or minor tech-debt items to revisit.
Per CLAUDE.md Layer 4 ("activate fixes, never default to legacy"), parked items are
either revived+validated later or deleted — not left inert as the default. Newest first.

## Test-suite: synthetic-artifact golden tests fail — pool starvation from artist-style genre gating (surfaced 2026-07-04)

**6 integration tests fail on master.** Verified **NOT** a lossless-speedup regression — they fail
identically with the 2026-07-04 edge_repair memo reverted to `a8e63b6` (repro:
`git stash push -- src/playlist/repair/edge_repair.py`, re-run, `git stash pop`). Failing tests:
- `tests/integration/test_ds_pipeline_smoke.py::TestDSPipelineSmoke::test_dynamic_mode_10_tracks`
- `tests/integration/test_ds_pipeline_smoke.py::TestDSPipelineSmoke::test_narrow_mode_10_tracks`
- `tests/integration/test_playlist_golden_files.py::test_synthetic_playlist_golden` — all 4 params
  (`narrow-…0000`, `dynamic-…0001`, `dynamic-…0010`, `discover-…0020`).

**Symptom:** length shortfall on the synthetic 100-track artifact — `generated 6/8 tracks but
expected 10` (`variable_bridge_length=False`) — so the synthetic golden track_id lists no longer
match. Run log shows `BPM data missing for 100/100` and `Pier-bridge length mismatch … expected 10`.

**Root cause (hypothesis, high confidence):** the artist-style pier-bridgeability genre gating that
landed on master this window — `aa1b1f4` (gate pier-bridgeability neighbors by genre relevance),
`77f9b63` (zero-genre rows always ineligible + genre-gate fallback), `4103664` (warn on pier
shortfall after reallocation) — starves the candidate pool on the **synthetic** artifact, whose rows
carry no real genres. "Zero-genre rows always ineligible" plausibly makes most synthetic rows
ineligible → pool < interior_len → the shortfall. `edge_repair` is length-preserving and cannot
change the track count, which exonerates the memo.

**Handoff / decide before fixing** — is the new gating *correct and the fixtures merely stale*, or a
*bug on genre-sparse artifacts*?
- If intended: regenerate the synthetic goldens (`pytest tests/integration/test_playlist_golden_files.py
  --generate-golden`) and update the smoke-test length expectations — or give the synthetic artifact
  non-empty genre rows so it isn't all-zero-genre.
- If the gate over-fires when genre data is absent: it likely needs a "genre data present" guard so it
  no-ops on genre-less/synthetic artifacts (making *every* zero-genre row ineligible when the WHOLE
  artifact is zero-genre is what starves the pool). Read a synthetic run at INFO + the three commits
  above before regenerating — regenerating stale-but-correct goldens is fine; regenerating to paper
  over a real pool-starvation bug is not.

Owner: the artist-style pier-bridgeability work (not the lossless-speedup track).

## ✅ Resolved 2026-07-03 (verified-fixes cleanup pass)

Buckets A/B/C of the triage; each re-verified against master before fixing. Scoped OUT of
this pass (still open below): the 🔴 edge-repair↔reporter T-mismatch (needs instrumentation
first), the design work (energy-arc, tempo-whiplash, identity/dedup), the artist-style
partial-`PierBridgeConfig` refactor, and the `genre_compatibility` deletion.

- **`config.example.yaml`: `edge_repair:` block added (on) + `artist_style.enabled: true`** —
  both now match live `config.yaml`; a fresh clone gets break-glass repair + medoid piers.
- **Var-bridge flex-counter over-count FIXED** (`var_bridge.py`) — `choose_segment_length` now
  reports `flexed=False` when `lo==hi==nominal` (only nominal buildable, no flex work); the
  `(N+1/N)` over-count + `chosen==nominal` mislabel is gone. Regression test added.
- **`ProgressLogger` verbose periodic summaries FIXED** (`logging_utils.py`) — removed the
  `_should_emit` `verbose_each→False` short-circuit so verbose runs emit INFO summaries per
  the class docstring; test added. Existing verbose test still green.
- **`--show-run-id` FIXED** (`main_app.py`) — now forwarded to `configure_logging` (was parsed
  by `add_logging_args`, honored in `analyze_library.py`, silently dropped on the generation CLI).
- **`--beat-sync` dead flag REMOVED** from `analyze_library.py` (arg + handler). NOTE: the twin
  flag + deeper beat-sync plumbing in `scripts/update_sonic.py`/`librosa_analyzer.py` is
  entangled with Beat3Tower extraction → deferred to **SP-C**.
- **`mode_presets.py` misleading "MERT p75/p50/p25" comments FIXED** — re-annotated as inert
  legacy floors superseded at runtime by `sonic_admission_percentile` (`candidate_pool.py:658-666`).
  Values KEPT: they're a tested plumbing contract (`test_mode_threshold_resolution.py`,
  `test_single_writer_settings.py`); actually deleting them means reworking `min_sonic_similarity`
  + ~6 test assertions — deferred as a larger change.
- **`transition_weights` config.example bug — MOOT**: SP-B deleted the knob; no tower keys ship
  in `config.example.yaml`, so `validate_tower_knobs` can't raise. (Already noted in WIRING_STATUS.)
- **`sim_variant` — CORRECTED**: the cited `sonic_variant.py:390` was deleted by SP-B; the key is
  no longer a transform selector. It survives only as a display/export tag surfaced in
  `main_app.py:66` (`self.sonic_variant`, default `"raw"`, passed to m3u export) — harmless, not
  a live gate. Full removal (drop the key + the export-tag plumbing) deferred; low value.
- **`tempo_stability` — CONFIRMED inert**: read-only DB query shows **5 / 41,178** analyzed tracks
  (0.01%) below the 0.5 `bpm_stability_min` bypass (mean 0.958, 0 nulls) — the BPM-bypass
  effectively never fires (the `bpm_trust` finding). Removal belongs to **SP-C** (its origin).
- **`pace_admission_floor` — VERIFIED dead** (write-only: `config.py:61/607-608` +
  `pipeline/core.py:373` set it; nothing reads `cfg.pace_admission_floor`). The 2026-06-12 pace
  retune removed the consumer but left the field. Removal deferred — it's a frozen-dataclass field
  touching `config.py` + `pipeline/core.py` + 3 tests (incl. `test_candidate_pool_pace_floor.py`,
  whose whole purpose is documenting its inertness).
- **Stale `doctor.py` Python floor + stale CLAUDE.md gotchas — MOOT**: `doctor.py` already checks
  3.11+ (noted below); the CLAUDE.md gotchas (`genre_conflict_min_confidence`, "162-d tower blend",
  "~455 genres") are already gone from CLAUDE.md (cleaned in d708e8b/4ba42f5).

## Fixer-cascade / reporting warts (surfaced 2026-07-02, seeds + Herbie Hancock log reads)

Evidence: `logs/playlists/2026-07-02_174010_seeds_fb3bb8.log` (10-seed, 50 trk) and
`logs/playlists/2026-07-02_175339_Herbie_Hancock_8552c5.log` (artist mode, 51 trk).

- **🔴 Edge repair and the reporter disagree on T for the SAME edges — one ruler is wrong.**
  Herbie run: `Edge repair summary: triggered=2 repaired=0 left_alone=2 (t_floor=0.30)`
  on edges 44/45 (the two edges around Aretha "Respect", pos 44 — repair tried to swap her,
  1,016-entry refusal log, ~5s spent), but the final report scores those same edges
  **T=0.790 and T=0.664** — nowhere near the 0.30 trigger. Repair ran *after* tail-DP
  (17:55:02 vs 17:54:57) and `repaired=0`, so it saw the identical final playlist; the two
  components computed incompatible T values for identical edges. Either repair over-triggers
  on healthy edges (wasted budget + could someday "fix" a good edge) or the reporter inflates
  (and every min_T/weakest-edge claim we trust is wrong). Layer-3 item 18's admission/reporter
  alignment lesson resurfacing inside the fixer cascade. Note: the 07-02 break-glass validation
  saw repair agree with reporter scale (`min_T 0.003→0.216`), so this may be a regression from
  the 07-02 weak-edge cascade reorder — verify against a fresh worker first (stale-worker trap),
  then diff repair's T computation (`src/playlist/repair/edge_repair.py`) vs the reporter's
  (`edge_metric_source: final_emitted_playlist`). **Blocker for trusting quality metrics.**
- **✅ RESOLVED 2026-07-03.** `choose_segment_length` (`var_bridge.py`) now returns `flexed=False`
  when only the nominal length was buildable (`lo==hi==nominal`, `len(results)==1`); regression
  test added. Original report: — **Var-bridge flex counter runs past its cap and flags non-flexed segments as flexed.**
  Herbie run, `variable_bridge_max_flex_segments=3`: segs 4–6 each ran 3 beam attempts then
  logged `flexed=True (1/3)…(3/3)` — correct. Segs 7–8 ran ONE attempt each (no extra beam
  work — the cap does bound the cost) yet logged `flexed=True (4/3)`, `(5/3)` with
  `chosen==nominal`. Accounting/logging only, but "diagnostic logging is part of the feature";
  cheap fix near `pier_bridge_builder.py:2011`.
- **90s generation ceiling breached on multi-seed 50-track runs.** Seeds run: 127s wall-clock
  (17:40:10→17:42:17; ~10s/segment × 9 segments + 3× var-bridge beam re-runs on seg 8). Herbie:
  85s — just under. Also `generation_budget_s: 0.0` in the effective config — if that's the
  budget knob, it's configured-but-inert (the "knob that can't act" failure mode); confirm 0.0
  means "disabled by intent" or wire it. Interacts with the artist-path beam-width item below
  (restoring full widths ≈ 2× beam cost) — this is a quality/time decision, not an auto-fix.
- **Fixer deadzone 0.30–~0.75: ugly-but-legal edges get no attention (policy gap, not a bug).**
  Seeds run min edge T=0.457 (Dinosaur Jr → Springsteen "Dancing in the Dark" pier on-ramp) got
  zero fixer help — every fixer floor sits at 0.30 (`tail_dp_floor`, `edge_repair_t_floor`,
  `edge_delete_floor`) and the edge is legal, just rough. Decide deliberately: raise the trigger
  floors, make them percentile-relative to the run, or accept that outlier seeds produce one
  rough seam (and surface it in the GUI instead). Blocked on the T-mismatch item above —
  retuning floors against an untrusted ruler is wasted work.

## Parked features

### Energy arc / pace-contour — PARKED 2026-07-01
- **What:** the arousal-based energy-arc + energy-step scoring (`energy_arc_strength`,
  `energy_arc_band`, `energy_step_*` in `PierBridgeConfig`; consumed live in
  `beam.py:1206`). The "first-class scored contour" extension (2-D [arousal, log-z-onset]
  stacking, SDD Tasks 1-3) is wired-but-off on branch `worktree-sp1-collapse-harness`,
  NOT on master.
- **Why parked:** measured redundant with MuQ for smoothness. On a real MuQ playlist,
  MuQ-similarity predicts energy-closeness (corr −0.25) ~2.5× better than tempo-closeness
  (−0.10), and energy along the playlist is already fairly smooth (2/29 adjacent jumps vs
  5/29 for tempo). An energy arc would add only *coarse* (arousal is a blunt signal) polish
  to something MuQ mostly does for free. Low marginal ROI; would need calibration + a
  by-ear audition to ship safely.
- **Revive only if:** we want INTENTIONAL directional energy dynamics (a deliberate
  build-up / comedown across the whole playlist) — the one thing MuQ can't do (it smooths,
  it doesn't *direct*). That's a "playlist has an arc" north-star feature distinct from
  anti-whiplash, and it needs its own calibration + audition.
- **It is NOT the tool for tempo whiplash** — see the feature note below.

## Higher-value gap surfaced (a FEATURE to build, not cleanup)

### Tempo-whiplash smoothness (onset-based)
MuQ is nearly blind to tempo (corr −0.10), so a real MuQ playlist has jarring adjacent
tempo jumps (measured 129→83 and 86→144 BPM; 9/29 transitions exceed 1.5× BPM). The lever
is a SOFT onset-based penalty — **onset over BPM** (BPM is meaningless on beatless/drone
tracks; the `bpm_trust` finding) and **soft over hard-gate** (hard tempo gates detonate the
relaxation cascade; the onset-band lesson). The BPM/onset bridge bands + soft penalties
already exist (`*_bridge_max_log_distance`, `*_bridge_soft_penalty_strength`) but are OFF
in `pace_mode: dynamic`. Open design question: an *adjacent-step* penalty (the tempo mirror
of `energy_step`) vs the *pier-relative* onset bridge band — the whiplash measured is
adjacent-transition, which the pier-relative band does not directly target.

## Identity fragmentation + near-dup dedup (surfaced 2026-07-01, Green-House run)

A Green-House playlist exposed two related identity gaps. Neither breaks the arc, but
both degrade diversity accuracy and let a near-duplicate slip in. Evidence: the
`artist_counts` block + tracklist of the 2026-07-01 Green-House generation.

- **Same artist counted as distinct identities (undercounts the per-artist cap).**
  Hiroshi Yoshimura appears as both `hiroshi yoshimura 吉村弘` (1) and
  `hiroshi yoshimura` (2) — the CJK-name-suffixed variant is not normalized to the
  romanized key, so he's 3 tracks scored as 1+2 against `max_artist_fraction`. Same
  latent pattern on `takahiko ishikawa 石川鷹彦` / `kohsei morimoto 森本浩正`
  (1 track each, so no cap impact *yet*). Identity resolution should fold a
  `"<romanized> <CJK>"` suffix form onto the romanized identity.
- **Cross-moniker aliases not collapsed.** `leon todd johnson` and `airport people`
  are the same artist under two names (per Dylan) but resolve to two identities — the
  same class as the "Smog ≟ Bill Callahan" follow-up noted in the `playlist-testing`
  skill. No mechanism maps same-person / different-project names together.
- **Near-dup slipped past the pool dedup (edge case).** Tracks 31→32 were both titled
  `underscore dash apostrophe`, identical 2:58 duration, back-to-back — almost certainly
  the same recording, published under the two artist strings above. The candidate-pool
  dedup is **artist+title** keyed (`Pier bridge candidate pool deduped: 572 → 564`), so
  two different artist strings on the same recording are not caught. Confirm it's one
  recording (metadata) vs two genuine tracks; if the former, identity-normalized dedup
  (or the alias map below) would catch it.
- **Trailing-punctuation variant counted separately (`PORCHES.` vs `Porches`), surfaced
  2026-07-02 Porches run.** The pier track "PORCHES. - Headsgiving" (trailing period)
  resolved to a distinct identity in the *reporter* path but not the *metrics* path: the
  run's `artist_counts` merged it (`porches`=10, distinct=37) while `print_playlist_report`
  counted it apart (Porches=9 @17.3%, unique=38). Same class as the CJK/alias items above —
  two counters normalize differently, so per-artist counts disagree. Narrow fix: strip
  trailing punctuation in the reporter's artist normalization to match the metrics path
  (`src/playlist/reporter.py`); the alias tool below also covers it.

**Proposed fix — GUI context-menu artist-alias tool.** Right-click an artist →
"alias to…" → writes `alias → canonical identity` mappings that feed
`artist_identity` resolution (min_gap, per-artist cap, seed-artist exclusion) *and*
pool dedup. Solves all three above with one authoritative, user-editable source, and
is the same capability the Smog/Bill-Callahan follow-up needs. Priority: low–medium.

## Artist-style path drops resolved-tuning fields: beam widths (½) + genre knobs (surfaced 2026-07-02, Porches runs)

The artist-style / popular-seeds path hand-builds `PierBridgeConfig`
(`playlist_generator.py:1974` **and** the sibling at `:2845`) and passes it as
`pier_bridge_config=`. In `apply_pier_bridge_overrides` (`pier_bridge_overrides.py:77`),
`pb_cfg = pier_bridge_config or PierBridgeConfig(...)` means a pre-built config
**short-circuits** the resolved-tuning construction — the mode tuning is resolved, logged,
then discarded (live log line: *"pre-built pier config supplied; the resolved-tuning weights
above were NOT applied"*). The complete wiring on the non-artist path threads every genre knob
(`pier_bridge_overrides.py:89,94`); the two hand-built artist constructors **omit a whole set of
resolved-tuning fields**, so they fall to their `PierBridgeConfig` dataclass defaults:

- **🔴 Beam + pooling widths run at HALF config — the "beam widths ran at half config for months"
  incident (CLAUDE.md) recurring on the artist path.** config.yaml (`:279-292`) sets
  `initial_beam_width=40, max_beam_width=200, initial_neighbors_m=200, max_neighbors_m=800,
  initial_bridge_helpers=100, max_bridge_helpers=400` — all exactly 2× the `PierBridgeConfig`
  dataclass defaults. The artist constructors set NONE of the six, so every artist-style /
  popular-seeds playlist runs the beam at `20/100`, neighbors `100/400`, helpers `50/200` — half
  the configured exploration (confirmed in both 2026-07-02 Porches runs' effective `pier_config`).
  This is a direct bridge-QUALITY lever (wider beam explores more paths → better worst-edge), i.e.
  the exact thing the weak-edge repair work is compensating for. **Fixing it has a real time cost**
  (≈2× beam+pool work; the 17:28 run just landed at 72s under the 90s ceiling), so restoring full
  width vs. setting a deliberate artist-mode width is a quality/time decision, not an auto-fix.
  **MEASURED 2026-07-02 (branch `fix/artist-pier-config-restore-tuning`):** the fix (thread all
  nine via a shared `_build_artist_pier_config` helper; verified live through the artist path —
  effective `initial_beam_width=40` etc. + `weight_bridge=0.60`) makes a full-width artist run take
  **~151s** (2026-07-02_180247 Porches, medoid piers) — ~2× the 72s half-width run, well over the
  hard 90s ceiling. So **full config width is OUT** under `feedback_generation_time_budget`.
  Decision pending: keep the (cheap) genre guardrails ON; set the beam to a deliberate under-90s
  artist width (little headroom above 20/100 → ~72s), OR cut generation time first — the var-bridge
  *flex retries* re-run whole segments at full beam and are a big multiplier, so a cheaper
  flex-retry beam could buy headroom for a wider base. Quality delta not yet measurable via CLI
  (medoid piers ≠ the GUI's popular-seed piers); needs a GUI re-run of the exact seed.
- **`segment_pool_genre_weight`: config `0.25` → effective `0.0`.** Per-segment candidate pool
  is ranked pure-sonic (passed as `genre_bridge_weight=0` at
  `pier_bridge_builder.py:1297/1360/1434`); genre does not shape what the beam considers.
- **`genre_pair_floor`: config `0.10` (dynamic) → effective `0.0`.** The per-edge genre-pair
  soft penalty (`beam.py:854`) never fires (nothing is below a 0 floor), so a sonically-close
  but genre-clashing edge is never demoted — the exact "guardrail against false-positive sonic
  matches" this lever exists for. The `genre_pair_floor > 0` DEGRADED warning
  (`pier_bridge_builder.py:665`) is silent as a side effect.
- (`genre_pair_penalty` is dropped too, but its dataclass default `0.5` == config `0.5`, so
  currently harmless — would bite if the config value changed.)

Everything else genre threads fine on this path (tiebreak 0.05, soft_genre_penalty 0.73/0.15,
`weight_genre` 0.0, genre_arc_*, the dense candidate-pool admission gate, taxonomy steering). On
the Porches run the drop was harmless (G already high — sonic and genre agreed; all chosen edges
`baseline_only`), but the two levers are silently disarmed for **every** artist-style /
popular-seeds playlist. Not a problem the day sonic is right; a missed guardrail the day sonic
gives a false positive.

**Fix (the real one, not the two-line patch):** stop hand-building a partial `PierBridgeConfig`
in the artist path. Either (a) thread the missing fields into both constructors now, or better
(b) route the artist path through the same resolved-tuning construction as
`apply_pier_bridge_overrides` and `replace()` only the genuinely artist-specific fields
(`bridge_score_weights`, `bridge_floor`, popularity, variable-bridge). (b) kills the whole class:
any tuning field added later is otherwise silently dropped on the artist path — the "config that
looks wired but isn't" failure mode.

**Not a bug, noted for clarity:** `artist_style.bridge_score_weights.dynamic` (0.6/0.4,
`config.yaml:42`) intentionally overrides the global `pier_bridge.weight_bridge_dynamic` (0.40),
so the global per-mode weight key is dead for artist playlists (they ran bridge-heavy 0.6/0.4,
not the config-global transition-heavy 0.4/0.6). Two sources, one silently shadows the other —
worth collapsing when the refactor above happens.

## Minor tech-debt (from `docs/HANDOFF_2026-06-30_muq_collapse_merge.md`, non-blocking)
- **Analyze stage-list triple-drift (item 2):** `web/src/components/ToolsPanel.tsx`
  `ALL_STAGES` (stale — lists `enrich`, missing `adjudicate`/`apply`/`popularity`) vs the
  canonical `request_models.py:ANALYZE_LIBRARY_STAGE_ORDER` vs `analyze_library.py`
  `STAGE_FUNCS`. `_clean_stages` runs ALL stages when the result is empty (silent footgun).
- **`doctor.py` Python floor (item 4):** ✅ RESOLVED (moot) — already checks 3.11+ (see the
  "Correction to an item above" note below; verified 2026-07-01).
- **`pace_admission_floor` dead knob:** defined + threaded, never read in `candidate_pool.py`.
  ✅ VERIFIED dead 2026-07-03 (write-only: `config.py:61/607-608` + `pipeline/core.py:373` set it;
  nothing reads `cfg.pace_admission_floor` — the 2026-06-12 pace retune removed the consumer but
  left the field). Removal DEFERRED: frozen-dataclass field touching `config.py`, `pipeline/core.py`,
  and 3 tests (incl. `test_candidate_pool_pace_floor.py`, whose purpose is documenting its inertness).
- **Stale CLAUDE.md gotchas:** ✅ RESOLVED (moot) — `genre_conflict_min_confidence`,
  "162-d tower blend", and "~455 genres" are all already gone from CLAUDE.md (cleaned in
  d708e8b/4ba42f5, 2026-07-02; verified absent 2026-07-03).

## Bugs + inert knobs (from the documentation audit, 2026-07-01)

Found while auditing the codebase for the doc rewrite; verified against current master.
(The MuQ auto-fold footgun from the merge handoff is intentionally NOT listed — it was
fixed by `54d682c`.)

**Bugs**
- **✅ RESOLVED 2026-07-03 (MOOT):** SP-B deleted the `transition_weights` knob and all tower
  keys; `config.example.yaml` no longer ships them, so `validate_tower_knobs` cannot raise. The
  original report below is retained for history. — ~~**🔴 `config.example.yaml` ships
  `transition_weights` that raise on the default learned variant — fresh-clone setup breaker.**~~
  The template sets `ds_pipeline.transition_weights` to
  `0.40/0.35/0.25` (`config.example.yaml:157-160`), which is **non-default** (the default is
  `DEFAULT_TOWER_TRANSITION_WEIGHTS = 0.20/0.50/0.30`, `artifacts.py:20`). On a no-tower sonic
  variant (mert/muq), `validate_tower_knobs` (`artifacts.py:404-439`, called from
  `pipeline/core.py:736`) **raises** on non-default `transition_weights`. So a fresh clone that
  copies `config.example.yaml → config.yaml` and generates on the default MERT artifact hits a
  `ValueError` on first generation. The live `config.yaml` doesn't hit it because it aligns them
  to the default `0.20/0.50/0.30`. **Fix:** set `config.example.yaml`'s `transition_weights` to
  `0.20/0.50/0.30` (or drop the block so they default) — the shipped `0.40/0.35/0.25` is the
  stale pre-v4.1 rhythm-heavy default that predates both the alignment fix and the guard.
- **✅ RESOLVED 2026-07-03:** removed the `_should_emit` `verbose_each→False` short-circuit;
  verbose runs now emit periodic INFO summaries (matching the docstring) alongside per-item DEBUG.
  Test added. Original: — **`ProgressLogger` periodic-summary branch is dead in verbose mode:** `_should_emit()`
  returns `False` whenever `verbose_each=True` (`src/logging_utils.py:363-364`), so the
  `if self._should_emit()` inside the verbose branch of `update()` (`:401`) never fires. A
  `--verbose` run gets DEBUG-per-item plus only the final `finish()` summary — contradicting
  the class docstring's "also emit periodic summaries." Fix `_should_emit` or the docstring.
- **✅ RESOLVED 2026-07-03:** `main_app.py` now forwards `show_run_id=getattr(args, "show_run_id",
  False)` into `configure_logging`. Original: — **`--show-run-id` is inert on `main_app.py`:** the flag is parsed by `add_logging_args()`
  (`src/logging_utils.py:459`) and honored by `analyze_library.py` (`:2616`), but
  `main_app.py` never forwards `show_run_id` to `configure_logging()` — so CLI generation
  runs never stamp a run_id (unless `--debug` / json logs). Forward it, or drop the flag there.

**Inert / vestigial knobs**
- **✅ CORRECTED 2026-07-03:** the cited `sonic_variant.py:390` no longer exists (SP-B deleted
  `sonic_variant.py`); `sim_variant` is no longer a transform selector. It survives only as a
  display/export tag: `main_app.py:66` reads it into `self.sonic_variant` (default `"raw"`) and
  passes it to m3u export (`:224/:359/:441`). Harmless, not a live gate. Full removal (drop the
  key + the export-tag plumbing) deferred; low value. Original: — **`playlists.sonic.sim_variant`
  (tower_weighted / tower_pca):** vestigial on the learned sonic variants.
- **✅ CONFIRMED inert 2026-07-03:** read-only DB query — **5 / 41,178** analyzed tracks (0.01%)
  fall below the 0.5 `bpm_stability_min` bypass (mean 0.958, 0 nulls). The BPM-admission bypass
  (`candidate_pool.py:697-698`) effectively never fires (the `bpm_trust` finding). Removal belongs
  to **SP-C** (Beat3Tower extraction, its origin). Original: — **`tempo_stability`:** read only as
  a BPM-admission bypass; reads ~0.96 for ~all tracks, so the bypass ~never fires.

**Deprecated**
- **✅ RESOLVED 2026-07-03 (partial):** removed the dead `--beat-sync` arg + its `if args.beat_sync`
  handler from `analyze_library.py`. NOTE: `scripts/update_sonic.py` has a twin `--beat-sync` +
  deeper beat-sync plumbing (`librosa_analyzer.py`, `hybrid_sonic_analyzer.py`) entangled with
  Beat3Tower extraction → deferred to **SP-C**. Original: — **`analyze_library.py --beat-sync`:**
  DEPRECATED — dead CLI flag, remove.
- **`genre_compatibility` (raw-tag pool penalty) — superseded by the dense PMI-SVD genre gate; delete.**
  Set OFF in live `config.yaml` 2026-07-01 (was erroneously `true` — default is `False`, absent from
  `config.example.yaml`, CLAUDE.md says keep off). It's a pool-level "candidate-vs-seed genre
  compatibility" filter (same altitude as the dense admission gate) implemented as **exact raw-tag
  overlap on the 410-dim vocab with identity affinity** (`compute_raw_genre_compatibility`,
  `genre_compatibility.py:68` → `np.eye` when `genre_affinity=None`, which is how both call sites
  invoke it). No relatedness: for a narrow seed it flags ~everything as conflict (Green-House run
  penalized 4307/4459 ≈ 97%). The **dense PMI-SVD admission gate** (`candidate_pool.py:772-847`,
  co-occurrence embedding = relatedness-aware) does the same job strictly better, so this is a
  redundant pre-dense predecessor. **Deletion is not a one-liner — two consumers:**
  1. `candidate_pool.py:905-931` — uses the `penalty` output, gated by `genre_compatibility_enabled`
     (now off). Remove the block + the four `cfg.genre_compatibility_*` fields (`config.py:55-58`,
     `585-...`) + the `config.yaml`/`config.example.yaml`/`CONFIG.md` keys.
  2. `artist_style.py:850-859` — calls it with `penalty_strength=0.0` and uses only
     `confidence`/`missing_or_sparse` to gate via `min_confidence` (`:886`). **Currently inert**
     (live path logs `min_confidence=None`). Decide: drop this confidence path, or re-source the
     signal from the dense embedding. Only after this is `genre_compatibility.py` (+ `tests/unit/
     test_genre_compatibility.py`) deletable.
  Also fix the stale doc pointers: `CONFIG.md:218` and `PLAYLIST_ORDERING_TUNING.md:350-358` still
  present it as a live-supported lever.

**Correction to an item above**
- The `doctor.py` Python-floor item (Minor tech-debt) is **RESOLVED** — `doctor.py:76-85` now
  checks 3.11+, not 3.8 (verified 2026-07-01). The list entry is stale.

### Cascade recon addendum (2026-07-02)

Found while mapping the weak-edge recovery cascade for the tuning-doc update; verified against master.

- **✅ RESOLVED 2026-07-03:** added the `edge_repair:` block (enabled, mirroring live
  `config.yaml:204-211` incl. the off-by-default `variety_guard`) to `config.example.yaml`, in
  cascade order between `tail_dp` and `edge_delete`. Original report retained for history. —
  ~~**🔴 `config.example.yaml` is missing the entire `edge_repair:` block.**~~ Same class as the
  `transition_weights` gap above. `edge_repair` is on in the live `config.yaml` (`:204`) but the
  template has no `edge_repair:` section at all — so a fresh clone runs the break-glass repair pass
  *off* (dataclass `edge_repair_enabled=False`) while `tail_dp` and `edge_delete` ARE templated and
  on. **Fix:** add `edge_repair: {enabled: true, t_floor: 0.30, centered_cos_floor: -0.5, margin: 0.05}`
  to `config.example.yaml`. (Note: `variable_bridge_length: true` IS present in config.example at
  `:258` — a parallel recon miscalled it missing; verified present.)
- **Reporter transition-blend weights are hardcoded, not threaded from config.** The reporter's
  `build_transition_metric_context()` uses hardcoded `weight_end_start/mid_mid/full_full`
  (0.70/0.15/0.15) instead of the `PierBridgeConfig` values. Dormant today (defaults match, unset
  in both yamls) but a second source of truth — tuning those weights in `config.yaml` would silently
  desync the reporter from the beam/fixers.
- **`tail_dp.batch_T` is a hand-mirrored reimplementation of the calibration sigmoid** (not a
  delegating call to `vec._calibrate_transition_cos`). Documented maintenance hazard; guarded by
  `test_batch_T_matches_score_transition_edge_both_branches`, which only catches drift if run.
- **Addendum to the 🔴 edge-repair/reporter T-mismatch (warts section above):** root-causing is
  blocked because `edge_repair._needs_repair` has two trigger arms (`t < t_floor` OR
  `centered_cos < floor`) but the swap/summary log never records which arm fired or the raw T at
  trigger time. Log the triggering arm + raw values first, then diff — before retuning any floors.

## Deferred sub-projects

### SP-B: Remove MERT + Beat3Tower (deferred 2026-07-01, GATED on SP-A)
Once MuQ is self-sufficient in the analyze flow (SP-A: a `muq` extraction stage lands +
a real rebuild proves it), remove the deprecated MERT + tower sonic paths: the `mert`
analyze stage, the tower decomposition (X_sonic_rhythm/timbre/harmony + tower_weights/
transition_weights — already INERT under MuQ), the MERT/tower artifact bakes (X_sonic_mert*,
X_sonic_tower_weighted, X_sonic_rhythm/timbre/harmony*), the variant-switching machinery
(sonic_variant.py tower branches, the MERT side of the variant-aware transition calib), and
fold_mert / fold_2dftm. **DATA SAFETY: ARCHIVE — never delete — the MERT shards + mert_sidecar.npz
(~55h CPU, irreplaceable). Remove only code + artifact bake.** Rationale: at runtime MuQ is the
sole active sonic space; MERT/towers are unused rollback + artifact bloat + superseded fusion
research (MuQ beat MERT 86% vs 73% on trusted triplets). Dylan: no deprecated code for fallback.

**SP-B completion note (Task 9, 2026-07-02):** Tasks 1-8 landed — the MERT analyze stage, the
MERT/tower artifact bakes, and the tower/MERT variant-switching code are gone from `src/`; MuQ
(`X_sonic_muq`) is the sole live sonic space (`artifacts.sonic_variant_override: muq`). Task 9
(this sweep) retired the matching config surface — `tower_weights`/`transition_weights`/
`tower_pca_dims` (`playlists.ds_pipeline`), `playlists.sonic.sim_variant`, and `analyze.mert` —
from both `config.example.yaml` and the live `config.yaml`, and updated CLAUDE.md's key-paths/
gotchas/Layer-3 items to the MuQ-only reality. Task 10 (artifact rebuild) and Task 11 (MERT
shard archival to `data/archive/mert_2026/`) are still open — the MERT shards + sidecar remain
in place under `data/artifacts/beat3tower_32k/` until then.

### SP-C: Retire Beat3Tower extraction (proposed 2026-07-02, follow-on to SP-B)
Once SP-B lands (Task 10 rebuild + Task 11 archive), the Beat3Tower *extraction* path itself —
the `stage_sonic` analyze stage + its extractor, which populate `tracks.sonic_features` and
gate the analyzable-track "universe" — is the next thing to retire: build a dedicated,
lightweight BPM/onset/pace extractor (rhythm-only; that's the pace gate's actual dependency per
the `bpm_trust`/`pace_admission_floor` history), migrate off `tracks.sonic_features`, rewrite
the universe/coverage gate to check MuQ coverage instead of tower coverage, delete `stage_sonic`
+ the tower extractor module, and re-validate the BPM/onset pace gates (`pace_mode: strict` /
`narrow`) against the new extractor's output before calling it done. Not started — needs its own
brainstorm/spec/plan.

## Research-harness modernization (surfaced during SP-B Task 9 review, 2026-07-02)

The following `scripts/research/*.py` audition/eval harnesses hardcode or pre-flight-check a
literal `"mert"` sonic-variant label, or read tower/MERT artifact keys that SP-B has removed
from `src/` (the artifact itself still has them until the Task 10 rebuild). They will abort, or
silently degrade, against a muq-only artifact until modernized — read the live/stamped variant
or take it as an explicit CLI arg instead of hardcoding `"mert"`. Not fixed now — SP-B follow-up:
- `collapse_eval.py:414` — aborts unless the live sonic variant == `"mert"`.
- `slider_differentiation_eval.py:428` — aborts unless the live sonic variant == `"mert"`; also
  `bpm_array` silently degrades at `:61` if its expected key is absent.
- `adaptive_admission_eval.py:476` — warns but proceeds unless variant == `"mert"`.
- `energy_neighborhood_probe.py:62-64` — hardcodes `set_sonic_variant_override("mert")`.
- `energy_spread_eval.py:117` — hardcodes the `--sonic-variant` CLI default to `"mert"`.
- `collapse_rescore.py:29,149,155` — hardcodes `VARIANTS = ["mert", "muq"]` and prints
  MERT-labeled columns unconditionally.

## Dead-code sweep (2026-07-04, ruff/mypy-triggered investigation)

Kicked off from the pre-existing linter noise. **`mypy` is clean (no issues, 178 files)** — the
pyright diagnostics in-editor are stricter-than-project noise (test mocks + god-class debt), not a
gate failure. The 49 ruff violations are ~36 style nits in `scripts/research/` + a few in live
files (cosmetic). The real dead code below came from import-graph + knob-read analysis, not the
linters. All items are NEW (not already tracked above) unless noted. Spot-checked items marked ✔.

### Dead config (Tier 1 — safe, high value)

- **✅ RESOLVED 2026-07-04 (commit `229fdb7`)** — removed the dead `RepairConfig` (dataclass,
  DSPipelineConfig field, parse/override-merge/effective-dump in `config.py`+`core.py`,
  `config_loader.py` getters + tuning-dict block, the `playlist_generator.py` overrides entry, and
  both yaml `repair:` blocks). mypy clean, 1887 unit tests pass. Original report retained:
  — **🔴 `RepairConfig` is 100% dead but SHIPS a live-looking `repair:` block** in both
  `config.yaml` (~`:346-350`) and `config.example.yaml` (`:722-726`) — ✔ verified zero reads of
  any `cfg.repair.<field>` in `src/`. The real repair feature is the unrelated `edge_repair:`
  block (`config.example.yaml:304`). Classic "config that looks wired but isn't" — a user tuning
  `repair:` changes nothing. Remove: the `RepairConfig` dataclass (`config.py:100-105`, `673-678`),
  the override-merge (`pipeline/core.py:66-68`), the effective-dump line (`core.py:78`), and both
  yaml blocks. No tests reference `RepairConfig`.
- **⏸️ HELD 2026-07-04 — entangled, needs a focused reviewed pass (not a mechanical delete):**
  removing the dead scoring fields touches `policy.py`'s single-writer registry
  (`"…scoring.gamma"` at `:41`/`:299`), the yaml `scoring:` blocks, `config_loader.py` getters,
  `AlphaSchedule` (+ its `__init__` export), and MULTIPLE tests — including the currently-failing
  golden tests (`test_playlist_golden_files`, `test_ds_pipeline_smoke`) another session is actively
  working, so touching this area now risks a collision. **CORRECTION:** `transition_gamma` is NOT
  log-only/dead — it is threaded live through the whole reporting chain (`core.py`,
  `transition_metrics.py`, `reporter.py`, `ds_pipeline_runner`, `worker`, `playlist_generator`);
  KEEP it. Original finding retained:
  — **`ConstructionConfig` scoring fields — 10 dead** (`local_top_m`, `alpha_schedule`, `alpha`,
  `alpha_start/mid/end`, `arc_midpoint`, `beta`, `gamma`, `hard_floor` — `config.py:81-90`,
  `631-663`) + `transition_gamma` (log-only, embedded for the reporter). ✔ Root cause verified:
  their only consumer `construct_playlist()` is retired (`pipeline/core.py:17` "no longer calling",
  zero call sites). The parallel `ds_scoring_*` / `ds_constraints_hard_floor` props in
  `config_loader.py` are equally dead (only caller `get_ds_tuning_dict()` is used by one unit test).
  KEEP the live siblings on `ConstructionConfig`: `transition_floor`, `min_gap`,
  `max_artist_fraction_final`(see below), `center_transitions`.
- **✅ RESOLVED 2026-07-04 (commits `a5fc338`+`b52c16a`)** — deleted the module + removed the smoke
  test. Original: **`src/features/beat3tower_normalizer.py` — entire dead module.** Zero non-test refs; only
  `tests/test_smoke_imports.py:57-59` import-checks it. SP-B's design doc kept it "as used by
  extraction," but `beat3tower_extractor.py` does not import it (imports only `beat3tower_types`).
  Its artifact output key `normalizer_params` was already found zero-reader. Delete module + adjust
  the smoke test. (The rest of the `beat3tower_*` family IS live — SP-C territory.)
- **✅ RESOLVED 2026-07-04 (commit `3d8b78e`)** — removed the field, override cast, construction
  site, and the commented `config.example.yaml` line. Original:
  **`variable_bridge_band` (`pier_bridge/config.py:356`)** — unused since the 2026-07-02 add-only
  reorder (own code comment says so); already commented out in `config.example.yaml:274`, but the
  dataclass field + override cast (`pier_bridge_overrides.py:117`) + construction
  (`playlist_generator.py:256`) are still live plumbing that silently accepts/discards it.
- **✅ RESOLVED 2026-07-04 (commit `c0eec48`)** — DRY consolidation done: repointed
  `test_duration_penalty.py` at the live `candidate_pool` copy (identical math, 10/10 pass), then
  deleted `beam.py`'s duplicate `_compute_duration_penalty`, the `pier_bridge_builder` re-export,
  the dead `PierBridgeConfig.duration_penalty_*` fields, and the config-field tests for them.
  −111 lines. The tangle it required (not a blind delete):
  `beam.py:150`'s `_compute_duration_penalty` was a DUPLICATE of the live `candidate_pool.py:111`
  one, and it was the copy that `tests/unit/test_duration_penalty.py` actually
  tested via the `pier_bridge_builder` re-export — while production scoring calls the candidate_pool
  copy (`candidate_pool.py:626`). Original finding:
  **`PierBridgeConfig.duration_penalty_enabled/_weight` (`pier_bridge/config.py:169-170`)** — dead
  name-collision copies of the LIVE `CandidatePoolConfig` fields (same names). Nothing reads the
  PierBridge copies. Plus a second unused `_compute_duration_penalty()` in `beam.py:150-189`,
  re-exported `# noqa: F401` from `pier_bridge_builder.py:138` for back-compat only.

### Dead config (Tier 2 — dead but the fix is a decision, not just a delete)

**⚠️ VERIFIED 2026-07-04 — Tier 2 is NOT a clean deletion sweep; corrections below.** Investigated
each item; the production-read analysis that flagged them missed test-construction, name-collisions,
and indirect threading:
- **`PierBridgeTuning.dj_route_shape` is LIVE, not dead** — `cfg.dj_route_shape` IS read at
  `pier_bridge/genre_targets.py:78` and `pier_bridge_builder.py:762`, and `tuning.dj_route_shape` is
  threaded via `pier_bridge/config.py:403`. Struck from the cleanup list.
- **`piers_per_cluster`, `CandidatePoolConfig.max_artist_fraction_final`, `title_exclusion_enabled/_words`**
  are unread in production but **constructed in tests** (`test_artist_style.py:61/1395`,
  `test_candidate_filters.py:25/64/99`, `test_dense_genre_integration.py:62`,
  `test_tag_steering_pool_lever.py:30`) — removing them means editing tests / dropping assertions; not
  a pure delete.
- **`pool_balance_mode` is read** (passed to `build_balanced_candidate_pool` at
  `playlist_generator.py:1974/2903`; the function ignores it) — removal needs the param dropped too.
- The genuinely-dead ArtistStyleConfig fields (`bridge_floor_strict/narrow/dynamic`, `bridge_weight`,
  `transition_weight`, the ArtistStyle `genre_tiebreak_weight` copy) are best removed as part of the
  tracked **"artist-style path drops resolved-tuning fields"** refactor (consolidate the double-parse,
  option b) — piecemeal field-deletes leave a messy half-state. HELD for that refactor.

Original findings retained below (with the corrections above applied):

- **`ArtistStyleConfig` shadow-reparse — 8 fields populated but never read back**: the artist path
  builds `ArtistStyleConfig` from the raw YAML (`playlist_generator.py:~1712/2791`) then
  independently RE-PARSES the same dict (`~2075-2097/2960-2965`) for the values it actually uses,
  so these never feed anything: `piers_per_cluster`, `pool_balance_mode` (also:
  `"proportional_capped"` was never implemented — `build_balanced_candidate_pool` ignores the
  param), `bridge_floor_strict/narrow/dynamic`, `bridge_weight`, `transition_weight`,
  `genre_tiebreak_weight`. Harmless today (both parses read the same YAML) but a footgun — editing
  a field in isolation does nothing. Fix = delete the fields OR kill the double-parse (route the
  artist path through one construction). Pairs with the tracked "artist-style path drops
  resolved-tuning fields" item above — same root cause (partial hand-built config on the artist path).
- Misc dead singles: `CandidatePoolConfig.max_artist_fraction_final` (`config.py:48` — only the
  `ConstructionConfig` copy is read at `core.py:654`); `CandidatePoolConfig.title_exclusion_enabled`
  / `title_exclusion_words` (`config.py:52-53` — real behavior reads raw YAML at
  `playlist_generator.py:394-397`, bypassing the resolved fields); `PierBridgeTuning.dj_route_shape`
  (`config.py:138` — resolved by `resolve_pier_bridge_tuning()` but never threaded into any
  `PierBridgeConfig` construction; live value comes from `pier_bridge.dj_bridging.route_shape`).

### Tier 3 — needs Dylan's intent before removal

- **✅ RESOLVED 2026-07-04 (commit `76dd0ce`, Dylan-approved kill)** — deleted
  `src/playlist/popularity_loader.py` + its only-caller test. NOT a live feature's scaffolding: the
  "Oops, All Bangers" feature IS alive (substrate + beam penalty + fire mode on master), but it loads
  popularity from the Last.fm cache via `popularity_runner.py`; the sidecar-vector `load_popularity_vector`
  was a superseded early approach the shipped + planned (poolgate) implementation abandoned (the poolgate
  design/plan/handoff never reference it). Zero production callers.
- **✅ RESOLVED 2026-07-04 (Dylan decision: KEEP) — investigation opened instead.** Digging into the
  keep/kill surfaced a real finding: **Roam Corridors is LIVE** (enabled in `config.yaml:138`,
  `corridor_penalty` applied in the beam at `beam.py:1303-1309`) but running on **MERT-era params it
  was never re-tuned for after the 07-01 MERT→MuQ swap** — the "Phase-3 calibration" never happened
  (`manifold.py`/`roam.py` frozen at 2026-06-24). The dense `mutual_proximity()` (exact Flexer/
  Schnitzer hubness correction) was built but NEVER wired — only a cheap approx is live — and MuQ's
  hubness/collapse problem ([[project_muq_collapse_quiet_audio]]) makes the exact form *more* relevant
  now. So `mutual_proximity` is kept, and a **Roam Phase-3 MuQ-calibration investigation** is on the
  roadmap: **`docs/superpowers/handoffs/2026-07-04_roam-muq-calibration-handoff.md`** (measure roam
  ON-vs-OFF on MuQ first; then sweep params + A/B exact-vs-approx MP). Original keep/kill note:
  only ref is `test_manifold.py`; roam-corridors plan said "both ship."

### Tier 4 — `scripts/research/` graveyard (67 scripts; triaged 2026-07-04)

25 LIVE-HARNESS (keep — sonic/genre audition builds, `slider_differentiation_eval`,
`time_golden_replay`, `spb_artifact_checks`, `calibrate_transition_sigmoid`, the genre-adjudication
family, current energy_probe_* WIP). ~10 broken-but-worth-modernizing (extends the
"Research-harness modernization" list above: `run_admission_grid`, `pace_eval_features`+`_run`,
`sonic_audition_build`, `verify_roam_transition`). **25 removed this pass** (each ✔ verified to
have zero Python importers among kept files; doc-only mentions don't count); 7 more held below:
- **Broken-dead (removed, 13):** `energy_neighborhood_probe`, `cutoff_probe`, `genre_traversal_probe`,
  `pace_head_mert_redundancy`, `pace_head_probe_analyze`, `sonic_centering_probe`,
  `sonic_beatsync_2dftm`, `sonic_gate1_blend`, `sonic_gate2_rhythm`, `sonic_harmony_richer_probe`,
  `sonic_harmony_weight_sweep`, `sonic_tower_diagnostic`, `sonic_why_yuji` — all read removed
  tower/MERT artifact keys or import deleted modules; their investigations shipped/concluded.
- **Orphan-abandoned (removed, 12):** `energy_arc_proof`, `energy_blocker_pinpoint`,
  `energy_ceiling_probe`, `energy_room_proof`, `energy_switch_proof` (concluded energy-arc, now
  PARKED), `research_genre_similarity`, `research_sonic_hubness`+`research_sonic_transition`
  (mutually-dependent pair, no external ref — delete together), `sonic_keyinvariance_check`,
  `pace_head_probe_manifest`, `pace_head_edgetest_extract_wsl`, `pace_head_probe_extract_wsl`.
- **HELD — test-coupled (deleting the script means deleting its test too; separate decision):**
  `sonic_phase1_metrics` (unit test + imported by `scripts/research/__init__.py`),
  `measure_genre_baseline`, `pace_audition_analyze`, `pace_audition_build`, `pace_audition_serve`
  (each has a passing unit test; build↔serve import each other).
- **HELD — low confidence:** `run_model_prior_album_tests`, `import_backfill_adjudications`
  (borderline in the active adjudication family — confirm before removal).
