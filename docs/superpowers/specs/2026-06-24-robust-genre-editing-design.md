# Robust genre editing — design

**Date:** 2026-06-24
**Status:** Approved (design); implementation pending
**Branch:** `worktree-robust-genre-editing`

## Summary

Rebuild the GUI "Edit genres for album" feature so a manual genre edit is **authoritative immediately** — it corrects the published authority (`release_effective_genres` in metadata.db), shows instantly in display/export, persists across reloads/regenerations, survives a future full re-publish, and (after a one-click re-bake) steers generation. The current feature stages a write into a pipeline-internal table that nothing the user looks at ever reads, so edits silently do nothing.

This is **not** Genre Review (the 34k-term hybrid-fusion adjudication queue). This is direct, per-release manual correction.

## Problem / root cause (measured 2026-06-24)

Saving an edit writes a *user override* into `ai_genre_user_overrides` (enrichment DB) via `worker.handle_edit_genres` → `SidecarStore.set_user_override`. The edit then dies at up to four independent points before becoming visible. Editing **The Radio Dept. – Pet Grief** hits all four:

- **A. Architectural gap (every edit).** Display/export read the **authority** `release_effective_genres` (metadata.db); generation reads the artifact (`X_genre_*` in `data_matrices_step1.npz`) baked from that authority. The override only reaches the authority when the **publish** stage runs, and reaches generation only after the **artifacts** stage rebuilds the NPZ. The GUI edit triggers neither. `applyGenreEdit` updates React state optimistically, masking the no-op until reload/regenerate.
- **B. Orphaned album (Pet-Grief-specific, fatal).** Pet Grief's tracks carry `album_id = 1e37941b7875c46c`, but there is **no row** for it in the `albums` table (its artist is stored as `The␠␠Radio Dept.` — a double space — unlike its six sibling albums; the album row was never created). Publish maps `release_key → album_id` and iterates album_ids **only from `albums`**, so an orphaned album can never receive a `release_effective_genres` row. It is one of **82 orphaned album_ids** in the library.
- **C. Empty override (UX trap).** The dialog only commits a genre chip on **Enter**; there is no Add button, and `Save` sends committed chips, not the text still in the input box. Type a genre, click Save without Enter → empty override. **76 of 2010** stored overrides are empty.
- **D. Free-text drop at publish.** Even a correctly-saved add survives publish only if the term resolves to a taxonomy `genre_id` (`classify_override_terms`); a term outside the 455-genre graph is silently discarded.

### Authority chain (verified)

Generation never reads enrichment.db. `config.yaml: genre_source: graph`; `src/playlist/` has zero references to `EnrichedGenreResolver` / `enriched_genre_signatures` / the sidecar. The real chain:

```
release_effective_genres  (metadata.db, written by publish)
   └─► X_genre_raw/smoothed in data_matrices_step1.npz  (artifact, genre_source=graph)
        └─► generation bundle (embedding_setup.py)  ──►  the engine
```

`ai_genre_enrichment.db` is pipeline-internal: collection evidence, the deprecated `enriched_genre_signatures`, and `ai_genre_user_overrides`. It is an *input to publish*, not the authority generation consumes.

## Goals

1. A genre edit corrects the authority for that release **immediately** (display + export reflect it without a full publish).
2. The edit **persists** across reloads, regenerations, and a future *full* publish/re-enrichment.
3. The edit can **steer generation** after one explicit, fast step (genre-only artifact re-bake — seconds, sonic untouched).
4. Works for **orphaned albums** (no `albums` row) without writing to scan data.
5. Only genres that exist in the canonical vocabulary can be saved (no silent no-ops); unknown terms get clear feedback.
6. No more empty-override saves.

## Non-goals

- Genre Review / hybrid-fusion adjudication (separate system).
- Adding new genres to the taxonomy graph (that is the SP3a taxonomy-growth process).
- Repairing the 82 orphaned `albums` rows as scan data (optional separate maintenance; this feature works without it).
- Track-level or artist-level genre editing (stays album-level, matching the authority's album_id key).
- Automatic re-bake on every edit (re-bake is an explicit one-click action).

## Decisions (resolved during brainstorming)

| Decision | Choice |
|---|---|
| Edit depth | **Authority correction (full):** write `release_effective_genres` directly; generation picks up on next artifact re-bake. |
| Vocabulary | **Constrain to taxonomy** via autocomplete; unknown terms rejected with feedback. |
| Generation pickup | **One-click fast genre re-bake** (re-bake only `X_genre_*` from the authority). |
| Backup policy | **Rely on regenerability** — no per-edit metadata.db backup (authority is derived; override is durable; NPZ backed up on re-bake). |
| Orphan handling | **Tracks-derived album_id**; never write the `albums` table; publish augmented to survive orphans. |

## Architecture & data flow

On **Save** the worker performs one logical operation:

1. **Resolve names → canonical `genre_id`s** via the taxonomy graph. Split into `resolved` and `unknown`. Unknown terms are **not** saved; they are returned to the GUI as a warning. (Kills D.)
2. **Compute diff** of resolved target vs the **current authoritative** genres for the album (`add`, `remove` in genre-id space). If the diff is empty → no-op, write nothing, report "no changes." (Kills the empty-override accumulation, C-at-storage.)
3. **Write the durable override** to `ai_genre_user_overrides` (add/remove as canonical genre **names**, so the existing publish path re-resolves them). Keyed by `make_release_key(artist, album)`. This is the replay instruction that survives a future full publish.
4. **Resolve `album_id` from `tracks`** (orphan-safe; see below). (Kills B.)
5. **Surgically materialize** `release_effective_genres` for that one album_id using the **same single-album materializer the full publish uses** (extracted from `build_resolved_table`). Result is byte-identical to a later full publish for that album. Rows from the user carry `source='user'`, `assignment_layer='observed_leaf'`, `confidence=1.0` (mirrors current publish behavior). (Kills A for display/export.)
6. **Return** `resolved` (canonical names saved) + `unknown` to the GUI; the playlist table updates from this authoritative response, not raw optimistic input.

**Generation pickup** — the one-click **"Refresh genres for generation"** action runs a new worker command that backs up the NPZ (timestamped), loads it, recomputes only `X_genre_raw/smoothed` + `genre_vocab` from the authority, and re-saves with all sonic/MERT arrays written back unchanged. The next generation reflects the edits.

**Durability fix (one-time)** — augment `resolve_release_key_to_album_id` and `build_resolved_table` so a full publish also derives album_ids from `tracks` for album_ids absent from `albums`. Without this, a future full publish would delete an orphan's authority row and never recreate it, silently wiping the edit.

```
Save ─► worker.edit_genres
          ├─ resolve names→genre_ids (taxonomy)              [reject unknowns → warning]
          ├─ diff vs current authority; empty ⇒ no-op
          ├─ ai_genre_user_overrides (durable replay diff)    [enrichment.db]
          └─ materialize_album_genres(album_id)               [metadata.db: release_effective_genres, source='user']
                 │  (shared with full publish → identical output)
        display / export read authority  ◄── instant
                 │
   "Refresh genres for generation" ─► re-bake X_genre_* in NPZ ─► generation reflects edit
```

## Components

### Shared genre layer

- **`src/genre/genre_publish.py` (refactor):**
  - Extract the per-album body of `build_resolved_table` into `materialize_album_genres(conn, album_id, *, graph_rows, legacy_rows, override, taxonomy)`. Both the full publish loop and the edit handler call it → no drift.
  - Durability fix: add a "recompute from `tracks`" step to `resolve_release_key_to_album_id` for keys not already mapped from `albums`/signatures; UNION track-referenced album_ids into the album set `build_resolved_table` iterates.
- **`src/genre/genre_edit.py` (new):**
  - `resolve_terms_to_genre_ids(taxonomy, names) -> (resolved: dict[name,genre_id], unknown: list[name])`.
  - `album_id_for_release(conn, artist, album) -> str | None` — exact (artist, album) over `tracks`, falling back to normalized `make_release_key` grouped over tracks; deterministic pick on collision (most tracks, then min id), logged.
  - `current_authoritative_genres(conn, album_id) -> list[(genre_id, name)]`.
  - `apply_user_genre_edit(metadata_conn, sidecar_store, *, artist, album, target_names) -> EditResult` — orchestrates steps 1–6; `EditResult` carries `resolved`, `unknown`, `added`, `removed`, `no_change`. **Base for the diff is read server-side** from the live authority (`current_authoritative_genres`), not trusted from the client, to avoid races. The client's `base_genres` field (if sent) is advisory/ignored; the dialog still fetches current genres on open purely to render editable chips.
- **`src/genre/authority.py`:** add `canonical_genre_search(conn, q, limit) -> list[(genre_id, name)]` (read-only) over `genre_graph_canonical_genres`.
- **`scripts/build_beat3tower_artifacts.py` (refactor):** extract `refresh_genre_matrices(artifact_path, db_path, config)` — load existing NPZ, recompute only `X_genre_raw/smoothed` + `genre_vocab` from the authority (reusing the existing graph-authority load + `build_genre_matrices`), re-save with other arrays untouched.

### Worker (`src/playlist_gui/worker.py`)

- Rewrite `handle_edit_genres` to call `genre_edit.apply_user_genre_edit`; emit `resolved` + `unknown` (+ `no_change`) in `result`.
- New `handle_refresh_genre_artifact`: timestamped NPZ backup → `refresh_genre_matrices`; register in `TRACKED_COMMAND_HANDLERS` (it writes; single-flight). Emit `result` + `done` on all paths.

### API (`src/playlist_web/app.py`, `schemas.py`)

- `GET /api/genres/search?q=&limit=` → `canonical_genre_search` (read-only, direct DB, like the existing `/api/autocomplete`). Returns `[{genre_id, name}]`.
- `GET /api/genres/for_album?artist=&album=` → current authoritative genres for the dialog's base/diff (read-only).
- `POST /api/refresh_genre_artifact` → `bridge.submit({"cmd":"refresh_genre_artifact", ...})`; `BridgeBusy → 409`; returns `{job_id}`; WS progress.
- `/api/edit_genres` response gains `resolved: string[]`, `unknown: string[]`, `no_change: bool`.

### Frontend (`web/src/`)

- **`components/EditGenresDialog.tsx`:**
  - On open, fetch current authoritative genres (`api.albumGenres`) → accurate base for the diff (not stale generation genres).
  - **Autocomplete dropdown** over `/api/genres/search` (debounced); selecting commits a chip.
  - **Commit pending input on Save** + an explicit Add affordance. If pending input is a non-empty unknown term, show the unknown-term warning rather than silently dropping it. (Kills C.)
  - Render server-returned `unknown` as a warning; if `no_change`, say so.
- **`App.tsx`:** `applyGenreEdit` updates from the authoritative `resolved` list; add a **"Refresh genres for generation"** button (post-save) → `api.refreshGenreArtifact()`, progress via `useWorkerEvents`.
- **`lib/api.ts` / `lib/types.ts`:** `genresSearch`, `albumGenres`, `refreshGenreArtifact`; extend the edit-response type with `resolved` / `unknown` / `no_change`.

## Data model

- **`release_effective_genres`** (metadata.db) PK `(album_id, genre_id, assignment_layer)`. User edits write `source='user'`, `assignment_layer='observed_leaf'`, `confidence=1.0`. No schema change.
- **`ai_genre_user_overrides`** (enrichment.db) — unchanged schema; the durable add/remove diff (canonical names, casefolded as today).
- No new tables. No scan-data (`tracks`, `albums`, audio/MERT) writes.

## Edge cases

- **Orphaned album:** album_id resolved from `tracks`; collisions resolved deterministically and logged.
- **Remove a graph leaf with inferred parents:** the shared materializer re-derives inferred rows from remaining leaves exactly as publish does → no stranded `inferred_family` rows.
- **No-op edit:** target == current authority → write nothing, report "no changes."
- **Edit / refresh while generating:** both tracked → single-flight `409`; dialog surfaces "a generation is in progress."
- **Re-bake with no artifact present:** clear error ("build artifacts first"); no partial write.
- **Unknown terms:** partial save of known genres; unknowns returned as a warning (never block the whole edit).

## Error handling

- Write order: **override first** (durable intent), then surgical authority write (live view).
  - Authority write fails after override succeeds → edit is still durable; report "saved; will apply on refresh."
  - Override write fails after authority succeeds → live now but not yet durable across a *full* publish; warn.
- Worker handlers emit `error` + `done(success=False)` on all failure paths (a missing `done` wedges the bridge). API maps `BridgeBusy → 409`, worker timeout/down → 504/503 via existing global handlers.

## Data safety

- Only metadata.db write is to `release_effective_genres`, a **derived, regenerable** table publish rebuilds wholesale. The durable truth is the override + the graph; the authority is always reconstructable via publish. **No per-edit metadata.db backup** (full-file copies of an ~800 MB DB per click are too heavy and unnecessary).
- `refresh_genre_artifact` backs up the NPZ (timestamped) before overwrite and rewrites only the genre arrays; sonic/MERT arrays are read and written back byte-for-byte, never recomputed — the irreplaceable MERT-derived sonic data is never at risk.
- Never write `tracks`, `albums`, or audio-analysis tables.

## Testing

- **Unit (pytest, fast):**
  - *Materializer parity:* `materialize_album_genres(album_id)` output == a full `build_resolved_table` run for that album (golden).
  - *Orphan durability:* seed an orphaned album (tracks, no `albums` row) + an override; run a full publish; assert genres survive (proves the tracks-derived album_id fix).
  - *Term resolution:* known→id, unknown→reported, casefold/dedupe.
  - *Diff + no-op:* add/remove vs current authority; identical target writes nothing.
- **Integration (web, Windows-safe via `asyncio.run` driving `WorkerBridge` — not TestClient for real-worker, per the web-gui skill), on temp copies of both DBs:**
  - `edit_genres` end-to-end → `release_effective_genres` rows with `source='user'`.
  - `refresh_genre_artifact` re-bakes `X_genre_*`; edited tracks' genre vector changes (asserted at the artifact-vector level — no full generation run needed).
- **API (TestClient + fake worker):** fake-worker branches for `edit_genres`, `refresh_genre_artifact`, `/api/genres/search`, `/api/genres/for_album`.
- **Playwright:** open dialog → autocomplete pick → save (incl. pending-input commit) → chips/warning assertions → refresh button fires a job.
- **DB safety in tests:** always operate on **temp copies** of metadata.db / ai_genre_enrichment.db — never symlink the real SQLite DBs into the worktree (dual-WAL corruption).
- **Manual end-to-end (from the real checkout, post-review):** rebuild `web/dist` + restart `serve_web.py`; edit Pet Grief's genres; confirm chips display; click refresh; regenerate; confirm persistence.

## Rollout / sequencing

1. Shared genre layer: `materialize_album_genres` extraction + publish orphan-durability fix (with parity + durability tests).
2. `genre_edit.py` + worker `edit_genres` rewrite + `canonical_genre_search`.
3. `refresh_genre_matrices` + worker `refresh_genre_artifact`.
4. API endpoints + schemas.
5. Frontend dialog (autocomplete, pending-input fix) + refresh button + `App.tsx` wiring.
6. Rebuild `web/dist`, restart `serve_web.py`, manual e2e on Pet Grief.

Backward-compatible and opt-in by construction: no behavior changes until a user edits genres; existing overrides remain valid; the publish path is a strict superset of today's.
