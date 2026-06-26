# Spec — "Oops, All Bangers": Pool-Gate Re-Architecture

**Date:** 2026-06-26
**Status:** Design approved (brainstorming), pending spec review → plan
**Supersedes:** the beam-penalty design (`docs/superpowers/specs/2026-06-25-oops-all-bangers-design.md`)
as the *primary* mechanism. The beam penalty is retained as a secondary refinement.
**Base:** local `master` @ `d882d87` (popularity substrate already merged — do not rebuild).

---

## 1. Problem

Today's Bangers mode is a **soft beam penalty**: it re-ranks candidates *within* the
sonic/genre/pace-gated candidate pool (`_popularity_factor` in `pier_bridge/beam.py`). This has a
proven ceiling. For a Pearl Jam / Nirvana seed in an indie-heavy library, the sonically-similar pool
is full of grunge/indie **deep cuts**, while the artists' actual hits (Modest Mouse *Float On*,
Sonic Youth *Kool Thing*) are **cross-genre/cross-texture and get filtered out of the candidate pool
before the penalty ever sees them**. A re-ranker cannot promote a track that was never admitted.

**Verified (prior session):** the popularity scores themselves are correct and correctly wired
(Even Flow, Kool Thing, Float On all resolve to 1.00 against the live bundle). The failure is **pool
composition**, not a wiring bug. Therefore the fix must move popularity **upstream into pool
admission**, not stay in the beam.

## 2. Decision summary

Bangers becomes a **popularity admission gate** on the candidate pool, alongside the existing
sonic / genre / pace gates in `build_candidate_pool`.

| Mode | Popularity gate | Cohesion |
|---|---|---|
| **OFF** | none | exactly the user's slider settings (today's behavior, byte-identical) |
| **ON** | admit only rank `< 50` (top-50) — kills deep cuts, deluxe/bonus, live | **unchanged** — the user's normal vibe, just no filler |
| **OOPS** | admit only rank `< 10` (top-10) — the real hits | **rests lower** on a fixed cohesion ladder; starvation pushes further down the *same* ladder |

**The cohesion ladder (give-up order).** OOPS is "FM radio of this artist's world." A station holds
its **genre umbrella** longest and lets **texture and tempo swing** within it. So both OOPS's resting
cohesion *and* its starvation fallback follow one fixed priority order:

> **sonic → pace → genre → popularity**

- **Sonic/timbre is sacrificed first.** The cross-genre hits we want (Float On next to Even Flow) are
  *same-umbrella, different-texture* — both "'90s alternative rock," but jangly-clean vs. murky-grunge.
  The axis blocking them is sonic, so loosening sonic is the **targeted** fix and it admits **more
  bangers, not fewer**. Relaxing sonic costs cohesion, never banger-purity.
- **Pace second.** Cheap to relax: the pace *gate* only changes which tempos are *eligible*; the beam
  still sequences for an energy arc, so a faster hit is never slammed next to a ballad.
- **Genre is the glue — relaxed last, and only under starvation.** It's the umbrella that makes the
  station feel like a station; relaxing it admits genuinely cross-cultural tracks (a hip-hop banger on
  a Pearl Jam seed) — correctly the most-disorienting, last-resort step.
- **Popularity is the final rung.** It is the **only** relaxation that admits a non-banger (a deeper
  cut). It loosens (top-10 → top-25 → top-50 → off) **only after sonic/pace/genre are exhausted**, and
  every notch is logged loudly.

**Control model (who drives cohesion when OOPS is on):**
- **OOPS owns sonic + pace.** It imposes a loosened "radio" baseline regardless of where the
  `sonic_mode` / `pace_mode` sliders sit. Varied texture/tempo *is* the radio character.
- **The user owns genre.** `genre_mode` sets the **umbrella width** (how far the station may wander
  culturally) and is respected as OOPS's resting genre tightness. Only starvation widens genre past it.
- **ON owns nothing.** Normal cohesion exactly as the user set it; it only adds the popularity gate.

**Starvation policy: relax-to-fill, not ship-shorter.** A thin banger pool starves the pier-bridge
beam (bridges can't form → the infeasible-handling cascade fires → risk of blowing the 90s ceiling or
failing). Bangers is a *soft* axis like sonic/genre/pace (diversity is the only hard constraint), so it
follows the project's never-fail discipline: relax until it generates. Purity stays **observable** —
`annotate_and_log_playlist_popularity` already prints each final track's Last.fm rank, so any dilution
is visible, never silent.

**Cutoffs (starting values, calibration-tunable):** ON `rank < 50`, OOPS `rank < 10`. Because Last.fm
top-track lists are fetched at `limit=50` and coverage resolves ~3.5 tracks/artist on average, ON ≈
"is in the artist's top-tracks set at all" and the cutoffs only sharply differentiate high-coverage
artists. This is expected and fine.

## 3. Architecture

All insertion points verified against the current code.

### 3.1 Popularity **rank** loader (new, small)

`load_pool_popularity_values_cached` (`popularity_runner.py:322`) returns a **score** `1 − rank/n`,
not the rank — and because `n` (the artist's cached top-track count) varies, a score threshold is *not*
a fixed rank threshold. The gate is rank-based, so add a sibling:

```
load_pool_popularity_ranks_cached(bundle, pool_indices, *, db_path) -> np.ndarray  # int, -1 = unknown
```

It mirrors the existing cached loader exactly but stores the **0-based rank** (from the already-existing
`resolve_top_tracks_to_rank`) instead of `1 − rank/n`. Uncached / sub-8-track / not-in-top-N artists →
`-1`. Cache-only (no Last.fm client), never raises. The beam keeps using the score loader unchanged.

### 3.2 The admission gate in `build_candidate_pool`

`build_candidate_pool` (`candidate_pool.py:506`) filters candidates in sequence: title-exclude →
similarity floor → sonic floor (eligible loop ~919) → genre hard gate (~999) → energy rescue (~1032) →
artist grouping/walk (~1094). The popularity gate is **one more filter, applied as the final
eligibility step — after energy rescue, immediately before artist grouping** (~line 1043). Placing it
last guarantees *every* track that reaches the pool is a banger, including energy-rescued ones.

```
# pseudo: applied when popularity_ranks is not None and rank_cutoff is not None
eligible = [i for i in eligible if 0 <= popularity_ranks[i] < rank_cutoff]
```

NaN/`-1` (uncached / not-in-top-N) → excluded (treated as non-banger). Seeds are already excluded from
`eligible` (via `seed_mask`), so **pier tracks are never gated** — the gate governs **bridge candidates
only** (see §3.8). Log `before/after/excluded` like the genre hard gate does.

### 3.3 Threading the mode + cutoff + ranks

- New direct kwargs on `build_candidate_pool`: `popularity_ranks: Optional[np.ndarray]` (aligned to
  `bundle.track_ids`) and `popularity_rank_cutoff: Optional[int]`. The gate is active iff both are set.
  Passed as kwargs (not a cfg field) because the cutoff **changes per cascade attempt** — mirrors how
  `genre_gate` is already threaded through the `_build_pool(candidate_cfg, genre_gate)` closure
  (`core.py:463`).
- The rank vector is loaded **once** over the allowed-candidate index set, before the cascade, in
  `core.generate_playlist_ds` (same place the beam's popularity is loaded today, but moved earlier — the
  gate needs it *during* pool admission, not after). Conditioned on `popularity_mode in {on, oops}`.
- `popularity_mode` ("off"|"on"|"oops") must reach `core.generate_playlist_ds` and the policy layer.
  It already flows request → worker → `create_playlist_for_artist`; extend it inward to the pool build.

### 3.4 OOPS owns sonic/pace (policy-layer override)

The mode→threshold mapping is single-sourced in the **policy layer** (`derive_runtime_config`). When
`popularity_mode == "oops"`, the policy layer resolves the **sonic and pace** admission thresholds from
the **OOPS baseline** instead of the user's `sonic_mode` / `pace_mode`; `genre_mode` is untouched.

- **Starting baseline (tunable):** OOPS sonic = `dynamic`, OOPS pace = `dynamic`. The cascade (§3.5)
  relaxes both toward `off`.
- ON and OFF apply no override.

Concretely this sets the underlying knobs (`sonic_admission_percentile`, `bpm/onset_admission_max_log_distance`)
to the OOPS baseline values. Exact knob values are pinned in the plan and exposed in config (§5).

### 3.5 The cohesion ladder / starvation cascade

**Home:** `core.generate_playlist_ds`, reusing the existing build-then-relax pattern (the `_build_pool`
closure + the One-Each relaxation loop at `core.py:765`). The banger cascade:

1. Build the banger pool at the OOPS baseline (sonic/pace overridden, genre = user, popularity = mode cutoff).
2. If `len(pool) >= min_banger_pool_size`, done.
3. Else apply the **next notch** of the relaxation ladder, rebuild, re-check. **Stop the instant the pool fills.**

**Default ladder (weighted-interleaved: sonic biggest/earliest, pace medium, genre small/late,
popularity last).** Tunable via `playlists.bangers.relax_ladder`:

```
1. sonic   → looser   (one notch)
2. pace    → looser   (one notch)
3. sonic   → looser   (two notches)
4. pace    → off
5. sonic   → off
6. genre   → looser   (one notch past the user's setting)   ← first cohesion loss the user didn't ask for; log
7. genre   → off
8. popularity → top-25                                       ← first non-banger admitted; log loudly
9. popularity → top-50
10. popularity → off (gate disabled; pure cascade fallback)  ← log loudly
```

`min_banger_pool_size` (config) is the fill target — a pool floor large enough that the pier-bridge
won't starve (default tied to target track count, refined in the plan). Each notch logs the axis, old→new
value, and resulting pool size. Rebuilding re-runs the pool matmuls; acceptable because the cascade only
fires under starvation, attempts are bounded by the ladder length, and the matmuls are fast (well within
the 90s budget). A single-pass threshold relaxation (compute distances once, lower thresholds in place)
is a noted future optimization.

### 3.6 Never-starve backstop reconciliation

`build_candidate_pool`'s existing `min_pool_size` backstop (`candidate_pool.py:~1138`) backfills from
highest-sonic-sim candidates **regardless of popularity** — it would smuggle non-bangers in. When the
popularity gate is active, the **banger cascade replaces it**: the backstop is banger-constrained (only
backfills tracks passing the active rank cutoff) or disabled, so the cascade's logged popularity-relax
rung remains the *only* path by which a non-banger enters.

### 3.7 Secondary beam penalty (unchanged)

`_popularity_factor` + `playlists.bangers.strength_on` / `strength_oops` stay as-is. Within the now
banger-only pool, the beam still prefers the bigger banger. It uses the existing **score** loader.

### 3.8 Scope: bridges vs. piers

The gate governs **bridge candidates** (the candidate pool). **Piers** are seed-artist tracks chosen by
pier selection and are never gated here. Making the *piers* hits is the **Popular Seeds** program's job
(`w_pop` / `popular_seeds`), which is orthogonal and already works. For a truly end-to-end "all bangers"
result, OOPS pairs with Popular Seeds (piers = hits, bridges = hits). Whether OOPS should *auto-enable*
Popular Seeds is raised in §10 (recommended, but a separate decision).

## 4. Data flow

```
GUI dropdown (Off/On/Oops)
  → GenerateRequestBody.popularity_mode
  → worker → create_playlist_for_artist(popularity_mode=...)
  → derive_runtime_config:  if oops → override sonic/pace thresholds to OOPS baseline (genre untouched)
  → core.generate_playlist_ds:
        ranks = load_pool_popularity_ranks_cached(bundle, allowed_indices, db_path)   # once, if on/oops
        cutoff = rank_cutoff_on (50) | rank_cutoff_oops (10)
        cascade:
          pool = _build_pool(cfg, genre_gate, popularity_ranks=ranks, popularity_rank_cutoff=cutoff)
          while len(pool) < min_banger_pool_size and ladder has next notch:
              apply next notch (loosen sonic/pace/genre cfg, or raise cutoff); rebuild; log
  → pier-bridge runs within the banger-gated universe (its infeasible-handling can only relax *within*
        the gated universe — it cannot add non-bangers)
  → beam: secondary _popularity_factor refinement (score loader)
  → annotate_and_log_playlist_popularity: per-track Last.fm rank on the final playlist (observability)
```

## 5. Config (`playlists.bangers.*`)

```yaml
playlists:
  bangers:
    rank_cutoff_on: 50          # ON: admit rank < 50
    rank_cutoff_oops: 10        # OOPS: admit rank < 10
    oops_sonic_baseline: dynamic   # OOPS overrides sonic_mode to this resting tier
    oops_pace_baseline: dynamic    # OOPS overrides pace_mode to this resting tier
    min_banger_pool_size: null     # cascade fill target; null → reuse the mode's min_pool_size,
                                   #   else fall back to 2 × target_tracks (concrete default)
    relax_ladder: null          # null → the default ladder documented in §3.5
    strength_on: 0.25           # existing secondary beam penalty (unchanged)
    strength_oops: 0.60         # existing secondary beam penalty (unchanged)
```

All knobs are **live defaults**, configurable for tuning/rollback — never legacy-defaulted off.
Adding a `PierBridgeConfig` field (if any) will drift the 4 pipeline config goldens; regenerate per the
handoff (delete the 4 `tests/unit/goldens/pipeline/*.json`, run golden test to rewrite, re-run to pass,
confirm the diff is only the new key).

## 6. Observability

- Gate: log `before/after/excluded` count + active cutoff (mode).
- Each cascade notch: axis, old→new value, resulting pool size; popularity-relax rungs logged loudly
  (these are the only purity-breaking steps).
- Final playlist: existing per-track Last.fm rank annotation (already wired) confirms how pure the result
  was. A "configured gate that can't act" must warn/raise, never silently no-op (project rule).

## 7. Never-fail & budget guarantees

- A playlist can never **fail** on bangers — the cascade always terminates at "gate off," which is OFF
  behavior. Bangers is a soft, relaxable preference.
- 90s ceiling respected: cascade is bounded (ladder length), matmuls are cheap, and it only fires under
  starvation. Non-bangers enter only via the explicit, logged popularity-relax rungs.

## 8. Testing

No real data in the worktree → unit tests only; **live generation verification is the user's** via the
GUI (rebuild `web/dist`, restart `serve_web` from the main checkout, which has real data).

- **`load_pool_popularity_ranks_cached`** — synthetic bundle + seeded cache rows: correct rank vector,
  `-1` for uncached/not-in-top-N, never raises.
- **Gate filter** — synthetic eligible set + rank vector: admits exactly `0 ≤ rank < cutoff`; seeds never
  gated; runs after energy rescue (rescued non-bangers excluded).
- **Policy override** — `popularity_mode=oops` overrides sonic/pace thresholds to the OOPS baseline and
  leaves `genre_mode` untouched; `on`/`off` override nothing.
- **Cascade ordering** — a starved synthetic pool relaxes in `sonic → pace → genre → popularity` order,
  stops as soon as the fill target is met, and logs each notch.
- **Backstop reconciliation** — with the gate active, `min_pool_size` backfill never admits a non-banger.
- **Goldens** — regenerate the 4 pipeline config goldens if a config field is added; confirm the diff is
  only the new keys.

Generation tests must route modes through the policy layer (`derive_runtime_config`) and use multi-pier
artist-mode seeds via the `gui_fidelity` harness — never hand-built single-seed overrides (per the
`playlist-testing` skill).

## 9. Out of scope / deferred

- **Pool-wide version-preference** (live/deluxe bridges in OFF mode). In ON/OOPS this comes free — a live
  cut not in the artist's top-N is gated out. The general OFF-mode pool-wide dedup benefits all modes but
  is orthogonal to this re-arch; deferred to its own follow-up.
- **Single-pass threshold relaxation** (perf optimization of §3.5) — deferred; the rebuild-per-notch
  cascade is adequate for v1.

## 10. Open calibration knobs (post-merge, "see behavior first")

- OOPS sonic/pace resting baseline (`dynamic` start) and the exact notch values per ladder step.
- `min_banger_pool_size` derivation from target length.
- Cutoffs (ON 50 / OOPS 10).
- **OOPS ↔ Popular Seeds:** recommend OOPS auto-enabling Popular-Seed pier selection so piers are hits
  too (otherwise deep-cut piers with banger bridges). Flagged for the user; not in v1 scope unless chosen.
```

