# Handoff — "Oops, All Bangers": Pool-Gate Re-Architecture

**Date:** 2026-06-26. **Goal of the next session:** make Bangers mode *actually* produce
all bangers by re-architecting it from a **beam re-ranker** into a **popularity admission
gate on the candidate pool**. Start with the pool-gate.

---

## TL;DR of the decision

The current Bangers mechanism is a *soft beam penalty* — it re-ranks candidates *within* the
sonic/genre-gated pool. That has a hard ceiling: for a Nirvana seed in an indie-heavy library,
the sonically-similar pool is full of grunge/indie **deep cuts**, and the artists' actual hits
(Float On, Kool Thing) are **cross-genre and gated out of the pool before the penalty ever
sees them**. So it can't be "all bangers."

**New architecture (agreed with the user):** Bangers becomes a **popularity admission gate**
in `build_candidate_pool`, alongside the existing sonic/genre/pace gates:
- **OFF:** unchanged (today's behavior).
- **ON:** admit only tracks in their artist's Last.fm **top-N** → kills deep cuts, deluxe/bonus
  tracks (they're not in top-N).
- **OOPS:** strict — admit only the artist's **top-K** songs (the real hits).
- **Sonic + genre still gate** the banger-only pool (the user wants cohesion *within* bangers).
- **Never-fail:** if the banger pool can't fill the target length, **relax the rank cutoff as
  the LAST cascade step** (after sonic/genre/pace), logged loudly. (User's leaning; the one open
  judgment call is relax-to-fill vs. ship-a-shorter-100%-bangers playlist.)
- The **beam penalty stays as a secondary** "prefer the bigger banger within the pool."

---

## What's already built & on master (HEAD `3011b21`)

All of this is merged. **Do not rebuild it.**

**Popularity substrate (`src/analyze/popularity_runner.py`):**
- `resolve_top_tracks_to_rank` / `_to_popularity` — Last.fm top tracks → local tracks (mbid-first,
  then loose-title + version-preference; remaster NOT penalized).
- `get_artist_top_tracks_cached_or_fetch` (lazy cache-first + TTL, never raises),
  `get_artist_top_tracks_cached` (cache-only read).
- `artist_top_tracks_cache` table in `ai_genre_enrichment.db` — **EAGER BATCH IS DONE**: 1,128
  qualifying artists cached (`analyze_library.py --stages popularity`), **52% library coverage**
  (~4k tracks score ≥0.90). Don't re-run unless adding artists.
- `load_artist_popularity_values` (one artist → bundle vector, for Popular Seeds piers).
- `load_pool_popularity_values` (client-based, many artists) + `load_pool_popularity_values_cached`
  (**cache-only, no client — the one to use in the pipeline**).
- `annotate_and_log_playlist_popularity` (per-track Last.fm rank on the final playlist + logs).
- `enrichment_db_path()` (ROOT-anchored).

**Beam penalty (current mechanism — keep as secondary):**
- `_popularity_factor(p, strength)` in `src/playlist/pier_bridge/beam.py` — graded multiplicative:
  `combined_score *= (1 - strength*(1-p))`, NaN→max (ruthless). Applied at the two genre-penalty
  sites `beam.py:1368` (path A) and `:1485` (tie-break path), keyed on loop var `cand`.
- `popularity_values` threaded: `core._run_pier_bridge` → `build_pier_bridge_playlist`
  (`pier_bridge_builder.py:1476` call) → `_beam_search_segment`; also `micro_pier.py:270,305`.
- `PierBridgeConfig.popularity_penalty_strength` (`pier_bridge/config.py`, default 0.0).
- Strength set on `pier_cfg` in `create_playlist_for_artist` (`playlist_generator.py` ~1919),
  flows to `pb_cfg`; `core._run_pier_bridge` loads cache-only popularity over the **gated pool**
  when strength>0 and passes it to the beam.
- **Config-tunable:** `playlists.bangers.strength_on` / `strength_oops` (defaults 0.25 / 0.60).

**Request/UX plumbing:**
- `popularity_mode: "off"|"on"|"oops"` through `request_models.py` (from_worker_args + sparse
  to_worker_args) → `schemas.py` (GenerateRequestBody.to_request) → `worker.py` (artist dispatch
  → `create_playlist_for_artist(popularity_mode=...)`) → `web/src/lib/types.ts`.
- **GUI:** Bangers is a **dropdown** (Off / On / Oops, All Bangers) in the mode row of
  `web/src/components/GenerateControls.tsx`. Track table shows a **Last.fm** column
  (`TrackTable.tsx`) when `popularity_rank` is present; `TrackOut.popularity_rank` in
  `schemas.py` + `types.ts`.

**Popular Seeds (earlier program, separate but related — steers the PIERS):**
- `w_pop` medoid term + per-pier rank logging (`log_seed_popularity`). `popular_seeds: bool`.
- This is orthogonal to All-Bangers (piers vs bridges) and works.

**Live-album version fix (this session, PIERS only):**
- `calculate_version_preference_score(title, album="")` in `src/title_dedupe.py` now penalizes
  **live albums** ("MTV Unplugged", "Live at Reading", etc.) even when the *track title* is clean.
- Album sourced from `metadata.db` (read-only) via `artist_style._load_albums_for_indices`,
  threaded `bundle → cluster_artist_tracks(metadata_db_path=...) → _dedupe_artist_indices(albums_by_index=...)`.
- **Gap:** bridges are NOT version-deduped → **live bridges still slip in** (e.g. Sonic Youth
  *Skip Tracer* from "Live in Austin"). The pool-gate work should apply version-preference
  pool-wide (needs album for pool tracks — same metadata.db sourcing, generalized).

---

## Verified facts (don't re-investigate)

- **The cache-only loader WORKS — key-matching is correct.** `ab_probe.py` (scratchpad) loaded the
  live bundle and confirmed: Pearl Jam Even Flow=1.00, Sonic Youth Kool Thing=1.00, Modest Mouse
  Float On=1.00, Ovlov Strokes=1.00. `bundle.artist_keys` == the cache's `artist_key`. So
  `popularity_values` carries the right scores. The "not all bangers" problem is **the pool
  composition (sonic gate excludes cross-genre hits)**, not a wiring bug.
- **The eager popularity batch is complete** (1,128 artists, 52% coverage). The pool-gate can run
  cache-only and most pool artists will be covered; sub-8-track artists are NaN (treat as
  non-bangers / not in top-N).
- The beam `universe` is the fixed closure of selectable tracks (`pier_bridge_builder.py:514`;
  all expansion draws from it, `:1166-1383`). Popularity (soft, beam-only) never reshapes it —
  but the **pool gate WILL** reshape it (that's the point), so the never-fail relaxation matters.

---

## Hypothesized implementation (pool-gate) — for the next session to validate

1. **Load popularity for the pool BEFORE the gate.** Today popularity is loaded in
   `core._run_pier_bridge` (after `build_candidate_pool`). The gate needs it *during* pool
   admission. Trace `core.generate_playlist_ds`: find the `build_candidate_pool(...)` call and
   load `load_pool_popularity_values_cached(bundle, <pool-or-allowed-indices>, db_path=enrichment_db_path())`
   before/inside it, conditioned on the bangers mode being on.
2. **Add a popularity admission gate in `src/playlist/candidate_pool.py`** (`build_candidate_pool`),
   parallel to the sonic floor / genre gate / pace bands (around the per-candidate admission loop;
   the title hard-exclude is at `candidate_pool.py:~925`). Exclude candidates whose per-artist
   popularity **rank** is worse than the mode's cutoff (NaN / not-in-top-N = excluded).
3. **Thread the mode + cutoffs to the pool builder.** `popularity_penalty_strength>0` currently
   signals "bangers on" but the gate needs the *mode* (off/on/oops) and the *rank cutoffs*. Add
   config: `playlists.bangers.rank_cutoff_on` (e.g. 50 = top-50/in-top-N) and `rank_cutoff_oops`
   (e.g. 10-20). Decide the carrier (a `PierBridgeConfig`/pool-config field, or thread through the
   pool-builder signature like other admission params).
4. **Never-fail relaxation.** If the admitted banger pool < what's needed for the target length,
   relax the rank cutoff progressively (last cascade step, after sonic/genre/pace). Log each
   relaxation. (Confirm relax-to-fill vs ship-shorter with the user first.)
5. **Version-preference pool-wide** (fixes live bridges): either dedup the pool by song (keep the
   album-aware canonical version) or apply `calculate_version_preference_score(title, album)` in
   pool admission. Needs album for pool tracks — generalize `_load_albums_for_indices` to the pool.
6. **Keep the beam penalty as secondary** (within the banger pool, prefer the bigger banger).
7. **Calibrate** on a diverse artist panel (legacy/active, niche/popular) — `slider_differentiation_eval.py`.

**Open design questions:** rank cutoffs (ON vs OOPS); does the gate apply to the artist-style
allowed pool (~8-9k) or the post-sonic/genre gated pool, or both; relax-to-fill vs ship-shorter;
how strict OOPS should be (top-10? top-20?). The user is the authority on "how aggressive."

---

## Traps & environment gotchas (learned this session)

- **Always work in a worktree, branch FRESH from local `master`** (not from `origin/master`, which
  is ~340 commits stale, and not from this session's `worktree-oops-all-bangers` branch). Use
  `git worktree add <path> -b <branch> master`. (EnterWorktree's `fresh`/`head` defaults don't give
  local master.) The current master HAS the cwd-independent-hooks fix (commit `1c75077`), so a
  fresh branch avoids the hook trap below.
- **Subagents launch in the MAIN checkout, not the worktree** → their commits leak to master. Do
  implementation **inline in the worktree** (this session did), or cd-guard + verify-branch every
  subagent. Read-only Explore subagents are fine (no commits).
- **`pytest` piped through `tail`/`head` is blocked by a hook.** Run pytest directly with `-q` and
  bound it with the tool's timeout; the tool truncates long output for you.
- **Windows pytest temp trap:** `pytest-of-Dylan` `PermissionError (WinError 5)` cascades to all
  tmp_path tests. Always pass a fresh `--basetemp=<scratchpad>/...`.
- **npm build from `web/` can trip the cwd hook** on a stale branch; run `npm --prefix web run build`
  from the worktree root. `web/node_modules` isn't created by `git worktree add` — junction it from
  the main checkout (PowerShell `New-Item -ItemType Junction`).
- **No real data in the worktree** (`data/` is a placeholder). Integration/golden tests skip; unit
  tests only. **The real generation must be verified by the user via the GUI** (rebuild `web/dist`,
  restart `serve_web` from the MAIN checkout, which has real data). Plan for user-in-the-loop verify.
- **Master moves under you** (other active sessions commit/merge). Before merging: `git -C <main>
  merge-tree --write-tree --name-only master <branch>` dry-run; after committing, verify your
  commits are NOT in master (`git merge-base --is-ancestor <yourcommit> master` should be false).
  Merges are 3-way and have been clean so far.
- **Adding a `PierBridgeConfig` field breaks `test_pipeline_smoke_golden`** (4 config goldens drift).
  Regenerate: delete the 4 `tests/unit/goldens/pipeline/*.json`, run the golden test (writes
  baselines + skips), re-run (passes); confirm the diff is *only* your new key.
- **Don't push to origin** (push-readiness gate; local master is intentionally far ahead).
- **Never re-run the analyze pipeline / re-fold MERT / touch `metadata.db` or the MERT shards.**
  Irreplaceable. `metadata.db` read-only here (the album loader opens it `?mode=ro`).

---

## Key file map

| File | Role |
|---|---|
| `src/analyze/popularity_runner.py` | resolver, cache, `load_pool_popularity_values_cached`, annotate |
| `src/playlist/candidate_pool.py` | `build_candidate_pool` — **where the gate goes** |
| `src/playlist/pipeline/core.py` | `generate_playlist_ds`, `build_candidate_pool` call, `_run_pier_bridge`, popularity load |
| `src/playlist/pier_bridge/beam.py` | `_popularity_factor`, beam penalty `:1368/:1485` |
| `src/playlist/pier_bridge/config.py` | `PierBridgeConfig.popularity_penalty_strength` |
| `src/playlist_generator.py` | `create_playlist_for_artist`, `_pop_strength`/config, `cluster_artist_tracks` call `:1745` |
| `src/title_dedupe.py` | `calculate_version_preference_score(title, album)` (album-aware) |
| `src/playlist/artist_style.py` | `_dedupe_artist_indices`, `_load_albums_for_indices`, `cluster_artist_tracks` |
| `web/src/components/GenerateControls.tsx` / `TrackTable.tsx`, `web/src/lib/types.ts`, `schemas.py` | GUI + API |

Prior design docs (describe the **beam-penalty** design — now being superseded by the pool-gate):
`docs/superpowers/specs/2026-06-25-oops-all-bangers-design.md`,
`docs/superpowers/plans/2026-06-25-oops-all-bangers.md`. The next session should write a **new
spec/plan for the pool-gate** (or revise these). SDD ledger of prior work:
`.superpowers/sdd/progress.md` (in the `worktree-oops-all-bangers` worktree).

## Suggested first moves for the new session

1. Re-read this doc + skim `core.generate_playlist_ds` and `candidate_pool.build_candidate_pool`
   to confirm the gate insertion point and how sonic/genre/pace admission is structured (mirror it).
2. Confirm the relax-to-fill vs ship-shorter call with the user, and the OOPS rank cutoff.
3. Brainstorm → spec → plan the pool-gate (it's a real re-architecture; treat it properly).
4. Build inline-in-worktree, unit-test what you can, hand the live GUI verification to the user.
