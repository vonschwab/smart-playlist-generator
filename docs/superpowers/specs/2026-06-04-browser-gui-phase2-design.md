# Browser GUI — Phase 2: Track Interactions

**Status:** Design approved (brainstorm 2026-06-04)
**Supersedes:** nothing — extends Phase 1 (`2026-06-04-browser-gui-phase1-design.md`)

---

## 1. Why

Phase 1 delivered the core Generate loop. Phase 2 makes the resulting playlist *actionable*: you can listen to it in the browser, replace tracks that don't fit, blacklist things you never want in playlists, fix genre tags, and export the result. Two write paths begin here — blacklist (the `is_blacklisted` flag in `metadata.db`) and edit-genres (a user override in `ai_genre_enrichment.db`) — both through existing worker commands. See §10 for the safety analysis.

## 2. Goals

- **In-browser audio playback** — full song streaming, floating mini-player, play through the entire playlist
- **Context menu** — right-click or kebab hover on any track row: Replace / Blacklist (track+album+artist) / Edit genres
- **Replace track** — modal with ranked candidates from the engine's replacement algorithm
- **Blacklist** — persistent blacklist via existing worker commands; blacklisted rows dim in-place
- **Edit genres** — album-scoped genre override written to `ai_genre_enrichment.db` user override table
- **Export M3U8** — client-side download from current playlist
- **Export to Plex** — server-side via `PlexExporter`; requires `plex.*` configured in `config.yaml`

## 3. Non-goals

- Pin to position (deferred — introduce later)
- Genre Enrichment slide-over (Phase 3)
- Blacklist management UI / slide-over (Phase 3)
- Advanced settings panel content (Phase 3)
- Any change to the engine, metadata.db schema, or artifact files

## 4. Backend architecture — extended WorkerBridge

The current `WorkerBridge` handles only `generate_playlist`. Phase 2 adds a generic `command()` method:

```python
async def command(self, cmd: dict) -> dict:
    """Submit any worker command and await its done event.

    Raises BridgeBusy if a generate or another command is running.
    Returns the payload from the matching done event.
    """
```

All Phase 2 mutations (blacklist, edit_genres, find_replacement_suggestions) go through this method. The worker is the single authority for all writes; no direct DB access from FastAPI for mutations.

**Busy behaviour for non-generate commands:** if the worker is busy (generate running), the API returns HTTP 409 with `{"detail": "A generation is in progress — try again when it finishes."}`. The frontend shows this message inline without crashing.

### New API routes

| Method | Path | Body | Worker command |
|--------|------|------|----------------|
| `GET` | `/api/audio/{track_id}` | — | (no worker — reads DB directly) |
| `POST` | `/api/replace_suggestions` | `{job_id, position, top_k=10}` | `find_replacement_suggestions` |
| `POST` | `/api/blacklist` | `{track_ids?, scope?, value?, name?}` | `blacklist_set` or `blacklist_scope_set` |
| `POST` | `/api/edit_genres` | `{artist, album, genres}` | `edit_genres` |
| `POST` | `/api/export/plex` | `{title, tracks}` | (no worker — runs PlexExporter directly) |

**Audio route** (`/api/audio/{track_id}`): looks up `file_path` from `metadata.db` (read-only), then streams the file with full HTTP Range / 206 Partial Content support so the browser's native `<audio>` element can seek. No worker involvement.

**Plex export route**: reads Plex config from `config.yaml` (`plex.base_url`, `plex.token`, `plex.music_section`, `plex.path_map`), instantiates `PlexExporter`, and calls `export_playlist(title, tracks)`. Returns `{"ok": true, "playlist_key": "..."}` on success or `{"ok": false, "error": "..."}` on failure. Returns HTTP 503 if Plex is not configured.

### New pydantic schemas (`schemas.py`)

```python
class ReplaceSuggestionsRequest(BaseModel):
    # job_id is unused by the worker (it reads from its own _LAST_GENERATION_CACHE)
    # but kept for client correlation and future multi-job support
    job_id: str
    position: int
    top_k: int = 10

class CandidateOut(BaseModel):
    track_id: str                     # from candidate "rating_key"/"track_id"
    title: str
    artist: str
    album: str = ""
    genres: list[str] = Field(default_factory=list)
    fit_score: float                  # mapped from worker candidate "mean_t"

class ReplaceSuggestionsResponse(BaseModel):
    position: int
    candidates: list[CandidateOut]

class BlacklistRequest(BaseModel):
    track_ids: list[str] = []         # for blacklist_set
    scope: Optional[str] = None       # "album" | "artist" for blacklist_scope_set
    value: str = ""                   # album title (album scope) or artist name (artist scope)
    artist: str = ""                  # REQUIRED for album scope: set_album_blacklisted(artist, album, enabled)
    enabled: bool = True

class EditGenresRequest(BaseModel):
    artist: str
    album: str
    genres: list[str]

class PlexExportRequest(BaseModel):
    title: str
    tracks: list[dict]                # [{rating_key, title, artist, file_path}]
```

### New Python files

- `src/playlist_web/audio.py` — `stream_audio(track_id, db_path, request)` helper; handles Range header, MIME detection, 404 for missing files
- `src/playlist_web/plex_export.py` — `run_plex_export(title, tracks, config_path)` helper; wraps PlexExporter import error gracefully

## 5. Frontend architecture

### New packages

```
@radix-ui/react-context-menu   # right-click menus
@radix-ui/react-dialog         # modals (replace, edit genres, plex export)
```

No full shadcn bootstrap needed — install these Radix primitives directly.

### Player state — `PlayerContext`

```tsx
// web/src/contexts/PlayerContext.tsx
interface PlayerState {
  playlist: TrackOut[];          // current playlist tracks (in order)
  currentIndex: number;          // -1 = nothing loaded
  playing: boolean;
}
interface PlayerActions {
  load(playlist: TrackOut[], index: number): void;
  play(): void;
  pause(): void;
  next(): void;
  prev(): void;
}
```

`PlayerContext` wraps `App`. The single `<audio>` element lives inside `MiniPlayer`; `PlayerContext` drives it via a ref. When `audio.ended` fires, `next()` is called automatically (wraps around at end of playlist).

### New frontend files

| File | Responsibility |
|------|---------------|
| `web/src/contexts/PlayerContext.tsx` | Player state + actions; `usePlayer()` hook |
| `web/src/components/MiniPlayer.tsx` | Floating bottom-right player pill |
| `web/src/components/ContextMenu.tsx` | Radix ContextMenu wrapper with dynamic labels |
| `web/src/components/ReplaceDialog.tsx` | Replace track modal |
| `web/src/components/EditGenresDialog.tsx` | Edit genres modal |
| `web/src/components/ExportPlexDialog.tsx` | Plex playlist name + export modal |

### Modified frontend files

| File | Changes |
|------|---------|
| `web/src/components/TrackTable.tsx` | Play button column; kebab (⋯) hover button; `onPlay`, `onContextAction` props; blacklist dim state; active-row highlight |
| `web/src/components/QualityStats.tsx` | M3U8 download button + Plex export button (right-aligned, disabled when no playlist) |
| `web/src/App.tsx` | `PlayerContext` provider; dialog state (replace, editGenres, plexExport open/closed); handlers for all context menu actions |
| `web/src/lib/types.ts` | `CandidateOut`, `BlacklistRequest`, `EditGenresRequest`, `PlexExportRequest` |
| `web/src/lib/api.ts` | `api.replaceSuggestions()`, `api.blacklist()`, `api.editGenres()`, `api.exportPlex()` |

## 6. UI design

### Track row

Every row has two new interactive elements that appear on hover:

- **Play button** (left column, always faintly visible, brightens on hover): clicking loads that track into the player at its playlist index. If that row is the currently playing track, shows ❚❚ (pause) instead of ▶.
- **Kebab (⋯)** (right edge, only on hover): clicking opens the context menu at that position. Right-clicking anywhere on the row also opens it.

**Active playing row**: `bg-[#15202b]`, mint-coloured index number, ❚❚ icon in play column.

**Blacklisted rows**: dimmed to 45% opacity, title gets `line-through`, a danger chip `blacklisted` appears. The row stays in the list for the current session — it does not disappear.

### Context menu

Matches the existing PySide6 context menu exactly. Labels are dynamic:

```
Replace this track…          (disabled on pier tracks — position 0 and last)
Blacklist 1 Track(s)
Blacklist Album: [album]
Blacklist Artist: [artist]
────────────────────────
Edit genres for album: [album]
```

Multi-select is not supported in Phase 2 — the menu always operates on the right-clicked row.

### Floating mini-player (`MiniPlayer`)

Fixed position `bottom-4 right-4`, `z-50`. Hidden until first track is loaded.

```
[⏮] [▶/❚❚] [⏭]   Track Title — Artist        1:43 / 4:29
                  ████████░░░░░░░░░░░░░░░░░░░░
```

- Border: `border-accent` (mint)
- Background: `bg-panel`
- Seek bar: click or drag to seek (uses `<audio>.currentTime`)
- `⏮` / `⏭` navigate prev/next in the current playlist; wraps at ends

### Replace track dialog

Centered modal with dimmed backdrop (`bg-black/60`).

- Header: "Replace track" + "Position N · between [prev title] and [next title]"
- Sub-header: "N candidates · ranked by transition fit to neighbors"
- Scrollable candidate table: index · title + genre chips + artist/album · fit score
- Candidates fade in opacity as fit score decreases
- Click to select (highlighted row); click Replace button (or press Enter) to confirm
- Pier positions (0 and last): "Replace this track…" menu item is greyed out; clicking does nothing

### Edit genres dialog

Centered modal.

- Album + artist in header
- Tag pill editor: existing genres as removable chips (`×`); inline text input to add new genre (Enter to confirm)
- Source genres section (read-only, below): shows genres from enrichment sources with their source label
- Footer copy: "Saves a user override · does not affect source tags"
- On Save: `POST /api/edit_genres` → worker `edit_genres` → writes `ai_genre_enrichment.db` user override table

### Export controls

Right-aligned in the `QualityStats` bar, disabled when no playlist loaded:

```
TRACKS 20  MEAN T 0.55  MIN T 0.48  DISTINCT ARTISTS 15      [↓ M3U8]  [→ Plex]
```

**M3U8 export**: pure client-side. Generates `#EXTM3U\n` + one `#EXTINF:{duration_s},{artist} - {title}\n{file_path}\n` per track, creates a `Blob`, triggers `<a download="playlist.m3u8">` click.

**Plex export**: opens `ExportPlexDialog` with playlist name input (default: `"{artist} — {date}"`). On confirm: `POST /api/export/plex`. If Plex not configured in `config.yaml`: dialog shows an error message instead of the name input.

## 7. New TypeScript types

```typescript
export interface CandidateOut {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  fit_score: number;   // mapped from the worker candidate's `mean_t` field
}

export interface ReplaceSuggestionsResponse {
  position: number;
  candidates: CandidateOut[];
}
```

## 8. Data flow — key sequences

### Play a track
1. User clicks ▶ on row N
2. `usePlayer().load(playlist.tracks, N)` — sets `currentIndex = N`
3. `MiniPlayer` sets `audio.src = /api/audio/{track_id}` and calls `audio.play()`
4. `audio.ended` → `next()` → loads track N+1

### Replace a track
1. Right-click row → "Replace this track…"
2. `App` opens `ReplaceDialog` with `{jobId, position}`
3. `ReplaceDialog` mounts → `api.replaceSuggestions({job_id, position})` → `POST /api/replace_suggestions`
4. Backend: `bridge.command({cmd: "find_replacement_suggestions", position, ...})` → result event → returns candidates
5. Dialog shows candidate list
6. User selects candidate + clicks Replace
7. `App` swaps the track in `playlist.tracks[position]` (local state mutation — no generate re-run). Quality stats (MEAN T, MIN T, etc.) are **not** recalculated after replacement — they reflect the original generation and are shown as-is.
8. Dialog closes; row updates in table

### Blacklist a track
1. Right-click row → "Blacklist 1 Track(s)"
2. `api.blacklist({track_ids: [track_id], enabled: true})` → `POST /api/blacklist`
3. Backend: `bridge.command({cmd: "blacklist_set", track_ids, value: true, ...})`
4. On success: `App` marks that track_id as blacklisted in local state → row dims

### Edit genres
1. Right-click row → "Edit genres for album: [album]"
2. `App` opens `EditGenresDialog` with `{artist, album}`
3. Dialog fetches current genres from the job result (already in the track's `genres[]`)
4. Source genres fetched via `GET /api/jobs/{job_id}` (already cached) or a new `GET /api/genres/{track_id}` endpoint — **use the genres already in the track data**, no extra fetch needed for current genres
5. User edits tag pills + clicks Save
6. `api.editGenres({artist, album, genres})` → `POST /api/edit_genres` → worker `edit_genres`
7. On success: genre chips on all rows for that album update in local state

## 9. Testing

### Backend (`tests/integration/test_web_api_phase2.py`)
- `GET /api/audio/{track_id}` — 200 with correct audio content-type (`.mp3`→`audio/mpeg`, `.flac`→`audio/flac`, `.m4a`→`audio/mp4`, `.ogg`→`audio/ogg`); 206 with Range header; 404 for unknown track_id or missing file on disk
- `POST /api/replace_suggestions` — returns candidates for valid non-pier position; 422 for pier position; 409 when worker busy
- `POST /api/blacklist` — track blacklist returns success; scope blacklist (album/artist) returns success; 409 when busy
- `POST /api/edit_genres` — returns success when worker accepts; 409 when busy
- `POST /api/export/plex` — 503 when Plex not configured

### Frontend (`web/tests/interactions.spec.ts`)
- Generate playlist → right-click row 5 → context menu appears with expected items
- Click "Replace this track…" → dialog opens, candidate list loads, click candidate → row updates
- Click "Blacklist 1 Track(s)" → row dims with strikethrough
- Click play button on row → mini-player appears with track info
- Mini-player ⏭ → next track loads
- M3U8 download → `<a download>` triggered (check element created)

## 10. Safety constraints

Phase 2 introduces two write paths. Both go exclusively through existing worker commands — the web layer never opens either database for writing directly.

- **`data/metadata.db` — blacklist writes the `is_blacklisted` flag.** The blacklist feature calls `MetadataClient.set_blacklisted` / `set_artist_blacklisted` / `set_album_blacklisted`, which run `UPDATE tracks SET is_blacklisted = ?` (and maintain the `artist_blacklist` / album-scope tables). This is the **same code path the existing PySide6 GUI already uses** — a bounded, reversible boolean-flag toggle, not a schema migration or re-analysis. It is the kind of write CLAUDE.md permits ("explicit user instruction" — the user requested the blacklist feature). It is **not** the kind of destructive write the backup-first rule targets. No new tables, no destructive statements, no migrations.
- **`ai_genre_enrichment.db` — edit-genres writes a user override.** Only via the worker `edit_genres` command (`SidecarStore.set_user_override`). The worker is the authority; the web layer never writes this DB directly.
- **`data/metadata.db` reads** — the audio route reads `file_path`; autocomplete reads `artists`. Read-only.
- **Audio files on disk are streamed read-only.** Never written, moved, or renamed.

**One-time precaution:** before the blacklist task lands, take a timestamped backup of `data/metadata.db` (`metadata.db.bak.<timestamp>`), per the CLAUDE.md data-safety rule, even though the write is a bounded flag toggle.
