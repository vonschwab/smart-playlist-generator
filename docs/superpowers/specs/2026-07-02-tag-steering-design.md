# Genre-tag steering (artist mode) — design spec

**Date:** 2026-07-02
**Status:** Stage 1 shipped (Tasks 1–8 committed, local master, unpushed); Task 9
(docs/skill sync) in progress; Task 10 (live GUI verification) pending. Stage 2 (beam
lever) not built — gated on stage-1 log evidence per §3.
**Feature group:** waypoint-steering primitive (shared with future A→B journey + era steering)

## Goal

In artist mode: after picking an artist, the GUI shows that artist's genre tags; the user
selects up to 3; the generated playlist **leans** toward those flavors — pool, piers, and
(stage 2) bridge routing all tilt the same direction. Soft bias only: a great off-tag
neighbor still gets in, and generation can never fail on steering.

## Decisions (locked with Dylan, 2026-07-02)

| Decision | Choice | Why |
|---|---|---|
| Steering semantics | **Lean** (soft bias), not lane (filter) or journey (arc) | Never-fail on soft axes; hard genre gates historically detonate generation time |
| Tag menu contents | **Artist's own published-authority tags only** | Honest, small menu; dense genre space generalizes the lean to adjacent flavors anyway |
| Menu granularity | **observed-leaf tags only** (no inferred hub families) | Hub families carry no steering signal (hub-saturation incident, 2026-06-12) |
| Scope of lean | **Piers AND bridges** | Piers dominate feel; anchors and bridge material must pull the same direction |
| Strength control | Config knobs only, **no GUI slider v1** | Deterministic-great first, knobs later |
| Mode scope | **Artist mode only v1** | YAGNI; seeds-mode steering is a later extension of the same seam |
| No tags selected | **Exactly today's behavior** | Feature is naturally inert when unused |

## 1. UX flow

1. User picks an artist (existing autocomplete in `web/src/components/GenerateControls.tsx`).
2. GUI calls new `GET /api/genres/for_artist?artist=<name>` → `{"genres": [{name,
   release_count, confidence}, ...]}`.
3. Chips render under the artist field: server orders by (release_count desc, confidence
   desc) and returns the top 12; multi-select capped at 3 client-side.
4. Selections travel as `steering_tags: list[str]` (default `[]`; canonical display names)
   on `GenerateRequestBody` (`src/playlist_web/schemas.py`).
5. Artist with no published genres → empty chip row + hint text ("no published genres —
   run enrichment publish").

## 2. Data path (authority-compliant)

Per the genre-data-authority skill: consumers read `release_effective_genres` via
`src/genre/authority.py`. Raw `artist_genres` is forbidden.

- **New:** `authority.py::resolved_genres_for_artist(conn, artist_name)` — as shipped: exact
  case-insensitive match on `tracks.artist` (`LOWER(TRIM(artist)) = LOWER(TRIM(?))`, no
  substring match, no `artist_key`/`norm_artist` normalization — the artist autocomplete
  that feeds this call also reads plain `tracks.artist`, so the same column is the correct
  key), via each matched track's `album_id`, aggregate `release_effective_genres` rows per
  `genre_id` (release count, max confidence), filter to `observed_leaf` (+`legacy`),
  map ids→names via `genre_graph_canonical_genres`, order by (release_count desc,
  max_confidence desc).
- **New:** `GET /api/genres/for_artist` in `src/playlist_web/app.py`, following the existing
  `/api/genres/for_album` pattern (`app.py:502-521`, direct read-only authority query in the
  web process).
- **Maintenance:** add the new function to the genre-data-authority skill's recipes table
  (its maintenance protocol requires it).

## 3. Engine: one target, three levers, two stages

### Target construction

- Selected tag names → indices in artifact `genre_vocab`.
- **Dense target** (pool + pier levers): `normalize(mean(genre_emb[tag_indices]))` from the
  genre-embedding sidecar (`src/features/artifacts.py:260-277`).
- **Taxonomy target** (beam lever, stage 2): tags in vocab/taxonomy space for the live
  taxonomy-steering path.
- A selected tag that fails to map **logs a WARNING naming the tag** and is dropped from the
  target — never a silent no-op ("a configured knob that can't act" rule). All tags failing →
  WARNING + steering disabled for the run.

### Stage 1 — pool + pier levers (ship first)

Note: bridges DO lean in stage 1 — bridge material is drawn from the steered pool — so the
"piers AND bridges" decision is satisfied by composition; stage 2 adds *route-level*
steering only if the logs show composition alone under-delivers.

- **Pool lever** (`src/playlist/candidate_pool.py`, dense admission ~`:810-833`): when
  steering is active, force `genre_admission_aggregate="centroid"` and replace the centroid
  with `normalize((1−λ)·seed_centroid + λ·target)`, `λ = tag_steering.pool_blend`
  (default 0.5). Percentile floor logic unchanged → pool size stays healthy; nothing is
  hard-excluded.
- **Pier lever** (artist pier candidate scoring in `src/playlist_generator.py`
  `create_playlist_for_artist` pier planning, where freshness/scarcity compose): add bonus
  term `tag_steering.pier_weight × cos(X_genre_dense[track], target)` (initial weight 0.3,
  tuning-recipe candidate).

### Stage 2 — beam lever (gated, not scaffolded)

- Blend the taxonomy target into the live steering targets
  (`pier_bridge_builder.py:1853-1895` `build_taxonomy_genre_targets` → beam `arc_g_targets`,
  `beam.py:501-512`).
- **Gate:** build stage 2 only if stage-1 diagnostic logs show the lean under-delivers on
  real generations (prior finding: genre is a minor nudge at edge level — pool composition
  may carry the whole effect). If stage 1 suffices, stage 2 is **not built** — no inert
  scaffold, no `beam_blend` knob shipped in stage 1.
- **Warning for the implementer:** the artist path hand-builds `PierBridgeConfig` and drops
  resolved-tuning fields (see CLEANUP_LIST "Artist-style path drops resolved-tuning fields").
  Stage 2 must not thread its knob through that constructor without addressing that wart.

### Diagnostic logging (part of the feature, Layer 4)

Each lever logs firing + effect size at INFO in the per-playlist log (as shipped, in
`src/playlist/tag_steering.py`, `src/playlist/candidate_pool.py`,
`src/playlist_generator.py`, `src/playlist_gui/policy.py`):
- `Tag steering: <tags>` — emitted as a policy note (`derive_runtime_config`), surfaces as
  `Policy: Tag steering: <tags>` once tags are selected.
- `Tag steering target: tags=[...] (mapped k/n)` — target resolution outcome (or a WARNING
  naming any unmapped tag / missing `genre_emb` / degenerate target).
- `Tag steering pool lever: blend=0.50 applied to admission centroid` — pool lever fired.
- `Tag steering pool affinity (genre-admitted set): p10=... p50=... p90=... n=...` —
  admitted-pool tag-affinity distribution, so the composition shift is visible.
- `Tag steering piers: affinity per selected pier = [...]` — per-selected-pier tag affinity.
- No lines at all = the knob didn't act (a bug, not a tuning problem) — see the deferred
  tuning recipe, `docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md`.

## 4. Config & wiring

- `config.yaml` + `config.example.yaml`: `playlists.ds_pipeline.pier_bridge.
  tag_steering_pool_blend: 0.5` and `tag_steering_pier_weight: 0.3` (stage 2 adds a
  `tag_steering_beam_blend` only if built). Home chosen for wiring reality: the
  `pier_bridge` dict passes wholesale through `build_ds_overrides` →
  `pb_overrides` (`playlist_generator.py:58`, `pipeline/core.py:304`) and is
  equally readable from the artist path's `ds_cfg` — one home, both consumers,
  zero new config plumbing. Tuning recipe drafted as
  `docs/HANDOFF_2026-07-02_tag_steering_tuning_recipe.md`, pending merge into
  `docs/PLAYLIST_ORDERING_TUNING.md` (deferred to avoid colliding with a concurrent
  docs-rewrite project that owns that file).
- `steering_tags` flows `GenerateRequestBody` → `UIStateModel` →
  `policy.derive_runtime_config` → overrides → worker `create_playlist_for_artist(...)`.
  Never a hand-carried config string (playlist-testing trap: knobs bypassing policy go
  inert in tests).
- Live by default (activate-fixes principle); inert with no tags selected, so no
  legacy-default risk.

## 5. Never-fail & edge cases

- No hard gates anywhere in the feature → no new generation-failure modes.
- 0 tags selected → all levers off, output byte-identical to today.
- Unmappable tag → loud WARNING + dropped (see Target construction).
- No published genres for artist → empty menu, generation unaffected.
- Deterministic: same artist + tags + seed → same target vector → same playlist.

## 6. Validation

- **Unit:** authority aggregation (multi-release artist, observed-leaf filter, no raw-table
  reads); vocab mapping incl. unmappable-tag warning; blended-centroid math (λ=0 ≡ seed
  centroid; λ=1 ≡ pure target; unit-norm).
- **Harness:** `tests/support/gui_fidelity.py::generate_like_gui` grows `steering_tags`,
  wired through `UIStateModel` + policy (never a hand-built override).
- **Behavioral (integration, slow):** Herbie Hancock ± `jazz-funk` tags — assert the
  admitted pool's tag-affinity distribution shifts and the pier set changes; read the gate
  tally + steering log lines per the playlist-testing skill (metric alone is not evidence).
- **Acceptance:** Dylan's ears on a real GUI generation, worker restarted, per-playlist log
  reviewed.

## Out of scope (explicitly)

- A→B journey mode and era steering (same waypoint primitive, separate designs).
- Seeds-mode steering; GUI strength slider; taxonomy-neighbor tag suggestions
  ("+ more like this" expander).
- Any hard genre gating (rejected as "lane" semantics).

## Key references (scouted 2026-07-02)

- Steering live-path: `pier_bridge_builder.py:504-508` (steering supersedes dj_bridging),
  `:1853-1895` (taxonomy targets); `beam.py:501-512`, `:846-880`.
- Pool admission: `candidate_pool.py:425`, `:771-843` (dense gate; centroid at `:813`).
- Artifact genre assets: `artifacts.py:56-58` (`X_genre_raw`, `genre_vocab`), `:260-277`
  (dense sidecar + `genre_emb`).
- Web flow: `GenerateControls.tsx:97`, `api.ts:46`, `app.py:166-210` (generate),
  `:523-561` (autocomplete pattern), `schemas.py:12-57`, `worker.py:1292-1305`,
  `playlist_generator.py:1339`.
- Authority: `authority.py:22-53` (existing per-album readers; artist aggregation is new).
