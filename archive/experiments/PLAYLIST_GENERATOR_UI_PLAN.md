# Playlist Generator GUI — Formal MVP Plan (Local‑First)

Last updated: 2025-12-12  
Target OS: **Windows first** (expand later)

This document is the reference plan for designing and building a GUI for the local playlist generator. The goal is an **MVP that works end-to-end** (seed → generate → edit → export) with **clean UX** and **approachable implementation**, then iterative expansion.

---

## 0) Product goals and principles

### Goals
- Make the generator usable without CLI: **pick a seed, pick a mode, generate, tweak, export**.
- Keep it **local-first / privacy-friendly**:
  - UI + API run on the user’s machine.
  - Local SQLite database is the primary source of truth.
- Keep the MVP simple, but not limiting:
  - Provide an **Advanced Settings page** (not just a hidden drawer) to support power users.

### Non-goals (for MVP)
- Full “music player” UI.
- Full explainability/attribution graphs (post-MVP).
- Multi-user, cloud sync, remote access.

---

## 1) Chosen technical architecture

### Option A (selected): **React/Vite/TypeScript UI** + **FastAPI local backend**
**Why:**
- Fast iteration loop (hot reload) with straightforward component model.
- Local API layer treats the existing Python generator as a **black box** (no rewrite).
- Easy to add packaging later (Tauri/Electron) once UX + API stabilize.

**Core runtime shape**
- `ui/` (Vite dev server) → calls → `api/` (FastAPI on `127.0.0.1`)
- `api/` imports existing Python modules and uses existing DB/config locations

### Later: Desktop wrapper (Tauri/Electron)
- **Not MVP.**
- Re-evaluate after MVP proves UX + API.  
  - Tauri for smaller footprint, Electron for easiest packaging.

---

## 2) MVP scope (what “done” means)

### MVP capabilities (must ship)
1. **Seed selection**
   - Search and choose seed by **track OR artist**.
   - Track seed: pick a track row → generate.
   - Artist seed: choose artist → system chooses a “smart” seed set (see below).

2. **Mode + length**
   - Mode: `narrow | dynamic | discover`
   - Length: numeric (e.g., 10–200; MVP default 30)

3. **Generate playlist**
   - Generate and display results list:
     - Track title / artist / album
     - Optional similarity score (if available)

4. **Basic playlist editing**
   - Regenerate playlist (“Try again” variation)
   - Lock a track (exact semantics TBD; implement useful default now, refine later)
   - Remove a track
   - Reorder tracks (drag and drop)

5. **Export**
   - Export to `.m3u`
   - Show output path + success/failure
   - Default naming: **`Auto - <Band Name>.m3u`**
   - Provide **Save As** to override name

6. **Library status panel**
   - Total tracks
   - Sonic-analysis coverage
   - Genre coverage
   - (Simple, lightweight stats; no heavy graphs)

### Must-have UX rules
- Keep MVP “clean”: only mode + length + seed visible on main screen.
- Any extra knobs live in **Advanced Settings**.
- Generation must show **loading state** (spinner/progress text).

---

## 3) Seed selection behavior (using your answers)

### Seed types
- **Track seed**: user chooses a specific track.
- **Artist seed**: user chooses an artist; system selects one or more seed tracks.

### Artist seed selection algorithm (MVP)
When an artist is selected as the seed:
1. Choose the **most played track** by that artist as the primary seed track.
2. Add additional seed tracks sampled **randomly** from the **Top 10** tracks for that artist derived from **Last.fm listening history**.

**Notes & implementation tradeoffs**
- This requires access to “top tracks by artist from Last.fm history.” Options:
  1. **Preferred**: cached locally in SQLite (stored via your existing metadata workflow).
  2. **Fallback**: call Last.fm API (requires user API key + username in config).
  3. **Fallback fallback**: if Last.fm data unavailable, use local playcount (if present) or choose highest-rated/most-recent/random track(s).

We will implement the algorithm with graceful degradation:
- If Last.fm top tracks are available → use them.
- If not → still allow artist seed generation with a “best effort” seed set.

---

## 4) Determinism + variations

### Requirements
- **Deterministic by default**: same inputs should yield the same playlist.
- Provide a **Try Again** action that yields a **different variation**.

### Approach
- Every generation takes a `random_seed`:
  - Default seed = stable hash of `(seed_selection, mode, length, advanced_settings_version)`
  - Try Again = same request but with `random_seed` incremented (or new random seed)

This preserves reproducibility while enabling exploration.

---

## 5) Export behavior

### Default export name
- `Auto - <Band Name>.m3u`
  - For track seed: band name = seed track artist
  - For artist seed: band name = chosen artist

### Save As + output directory
- The export UI provides:
  - A default directory (from config)
  - An override directory (chosen in UI)
  - A **Save As** filename input
- “Remember this folder” toggle:
  - If enabled, API writes back to config (or stores UI prefs locally)

---

## 6) Information architecture & screens

### Screens (MVP + needed for your requirements)
1. **Generate (Main Screen)**
   - Seed picker (track/artist)
   - Mode + length
   - Generate / Try again
   - Playlist results (edit controls)
   - Export block
   - Library status panel

2. **Advanced Settings (Page)**
   - All advanced knobs grouped by category
   - “Reset to defaults”
   - “Restore mode defaults”
   - Persist to config (and/or per-user UI prefs)

3. (Modal) **Seed Picker**
   - Tabs or segmented control: Tracks | Artists
   - Search box + results list
   - Selection action

---

## 7) Wireframes (ASCII)

### Main Screen
```
┌────────────────────────────────────────────────────────────────────────────┐
│ Playlist Generator                                                         │
│  Seed: [ Search tracks/artists… ______________________ ] [Pick]            │
│  Mode: (•) Narrow  ( ) Dynamic  ( ) Discover    Length: [ 30 ▼ ]           │
│  [Generate] [Try again]      [Advanced Settings →]                         │
├────────────────────────────────────────────────────────────────────────────┤
│ Playlist (N)                                              Library Status   │
│ ┌───────────────────────────────────────────────────────┐  ┌────────────┐ │
│ │ ⠿  Title — Artist                 Album           score│  │ Tracks:    │ │
│ │ ⠿  …………………………………………         ……………………       0.83 │  │ Sonic cov: │ │
│ │ ⠿  …………………………………………         ……………………       0.81 │  │ Genre cov: │ │
│ └───────────────────────────────────────────────────────┘  └────────────┘ │
│  Row actions: [Lock] [Remove]                                                │
├────────────────────────────────────────────────────────────────────────────┤
│ Export                                                                       │
│  Folder:   [ <from config or chosen>____________________ ] [Browse…]        │
│  Name:     [ Auto - <Band Name>.m3u ____________________ ] [Save As…]       │
│  [Export M3U]   Status: ✅ Saved to … / ❌ Error: …                          │
└────────────────────────────────────────────────────────────────────────────┘
```

### Seed Picker Modal
```
┌──────────────────────────── Pick a seed ─────────────────────────────┐
│ Search: [ __________________________ ]   ( Tracks | Artists )         │
│                                                                      │
│ Tracks                                                              │
│  • “Track” — Artist (Album)                         [Select]        │
│  • …                                                                  │
│                                                                      │
│ Artists                                                             │
│  • Artist Name (track count)                       [Use artist]     │
└──────────────────────────────────────────────────────────────────────┘
```

### Advanced Settings Page (outline)
```
Advanced Settings
- Generation
  - Mode defaults (narrow/dynamic/discover)
  - Deterministic seed (toggle) / seed value (optional)
  - Candidate pool / beam / floor mode (if supported by backend)
- Similarity weights
  - sonic_weight, genre_weight (and/or lambdas)
- Genre constraints
  - min genre coverage, smoothing toggles (if applicable)
- Export defaults
  - default export folder
- Integrations
  - Last.fm username + API key (optional)
[Save] [Reset to defaults]
```

---

## 8) Design system starter (minimal + utilitarian)

### Layout + spacing
- 8px spacing scale: `4, 8, 12, 16, 24, 32`
- Page max width: ~1100–1200px; comfortable scan lines
- Left-to-right: Primary workflow on left; status on right

### Typography
- System font stack (Windows-friendly)
- Title: 20–24px
- Section headers: 14–16px, semi-bold
- Body: 13–14px
- Monospace for paths / ids

### Components (MVP)
- Search input + results list
- Segmented control (Tracks/Artists, and mode selection)
- Number input / select for length
- Button (primary/secondary/danger)
- Table/list row with:
  - drag handle
  - row actions (lock/remove)
- Toast or inline status messages
- Modal dialog (seed picker)
- Spinner/progress indicator

### Interaction conventions
- Debounce search requests (e.g., 150–250ms) — “instant” not required, but avoid spam.
- Always show empty states:
  - “Pick a seed to generate a playlist.”
- Errors are readable and actionable:
  - export permission error → suggest changing folder

---

## 9) API contract (initial)

All endpoints are local-only; return JSON; errors use `{"detail": "..."} `.

### Search
**GET** `/api/search/tracks?q=...&limit=20`
```json
{
  "results": [
    { "track_id": "…", "title": "…", "artist": "…", "album": "…", "duration_s": 212 }
  ]
}
```

**GET** `/api/search/artists?q=...&limit=20`
```json
{
  "results": [
    { "artist_id": "…", "name": "…", "track_count": 1234 }
  ]
}
```

### Seed resolution (artist → seed tracks)
**GET** `/api/seed/artist/{artist_id}`
Returns the chosen primary seed + additional seeds and their provenance.
```json
{
  "artist_id": "…",
  "artist_name": "…",
  "primary_seed_track_id": "…",
  "seed_track_ids": ["…", "…", "…"],
  "source": "lastfm_top10|local_cache|fallback"
}
```

### Generate playlist
**POST** `/api/playlist/generate`
```json
{
  "seed": { "type": "track", "track_id": "…" },
  "mode": "dynamic",
  "length": 30,
  "random_seed": 12345,
  "advanced": { }
}
```
Response:
```json
{
  "playlist_id": "uuid",
  "seed": { "type": "track", "track_id": "…" },
  "mode": "dynamic",
  "length": 30,
  "random_seed": 12345,
  "tracks": [
    { "track_id": "…", "title": "…", "artist": "…", "album": "…", "score": 0.83 }
  ]
}
```

### Edit playlist (MVP: UI-local, but API-ready)
We’ll do **UI-only editing** first (lock/remove/reorder in the browser state).  
Post-MVP we can persist edits server-side with:
- `POST /api/playlist/{playlist_id}/apply_edits`

### Export
**POST** `/api/playlist/export`
```json
{
  "playlist_id": "uuid",
  "output_dir": "E:\\PLAYLISTS",
  "filename": "Auto - Slowdive.m3u"
}
```
Response:
```json
{ "ok": true, "path": "E:\\PLAYLISTS\\Auto - Slowdive.m3u" }
```

### Library status
**GET** `/api/library/status`
```json
{
  "total_tracks": 33636,
  "sonic_analyzed_tracks": 12000,
  "sonic_coverage_pct": 35.7,
  "genre_covered_tracks": 29600,
  "genre_coverage_pct": 88.0
}
```

### Advanced settings
**GET** `/api/settings`
**PUT** `/api/settings`
- Reads/writes config defaults (plus optional UI prefs).

---

## 10) Step-by-step implementation plan (milestones)

Each milestone ends with **something runnable**.

### Milestone 0 — Repo prep (30–60 min)
- Create top-level folders:
  - `api/` FastAPI app
  - `ui/` React app
- Add a `DEV.md` with commands and ports.

**Done when:** `python -m api` runs and `npm run dev` runs.

### Milestone 1 — “Hello library” slice
- API: `/api/library/status`
- UI: show status panel with real numbers.

**Done when:** opening UI shows track counts.

### Milestone 2 — Seed search (tracks + artists)
- API: `/api/search/tracks`, `/api/search/artists`
- UI: Seed Picker modal with tabs, search, selection

**Done when:** you can pick either a track or artist seed.

### Milestone 3 — Generate playlist (deterministic)
- API: `/api/playlist/generate` (track seed only first)
- UI: list results

**Done when:** seed track → playlist appears.

### Milestone 4 — Artist seed resolution (Last.fm-informed)
- API: `/api/seed/artist/{id}`
- Add Last.fm integration + caching strategy (graceful fallback).
- UI: artist seed now produces playlist via resolved seed tracks.

**Done when:** artist → generates playlist using “most played + sampled from top10”.

### Milestone 5 — Try Again variations
- UI button “Try again” calls generate with new `random_seed`.
- Ensure deterministic default for first generate.

**Done when:** Try again produces different playlist.

### Milestone 6 — Playlist editing (UI-local)
- Lock / remove / reorder within UI state.
- Export uses current edited ordering.

**Done when:** edits reflect in exported M3U ordering.

### Milestone 7 — Export
- API: `/api/playlist/export`
- UI: folder pick + Save As + status messaging

**Done when:** exported file opens correctly in a player.

### Milestone 8 — Advanced Settings page
- UI: settings page with categories + save/reset
- API: read/write settings (config + UI prefs)
- Wire advanced settings into generate request

**Done when:** settings persist and influence generation.

---

## 11) Scaffolding & file structure (planned)

```
repo/
  api/
    main.py
    routers/
      search.py
      playlist.py
      status.py
      settings.py
      seed.py
    services/
      generator_service.py
      export_service.py
      lastfm_service.py
      db_service.py
    models/
      schemas.py
  ui/
    src/
      app/
      components/
      pages/
        GeneratePage.tsx
        AdvancedSettingsPage.tsx
      api/
        client.ts
        types.ts
      state/
        playlistStore.ts
``

---

## 12) Open decisions (tracked)
- **Lock semantics**:
  - Default MVP: lock means “keep this track in the playlist and keep its position during Try Again.”
  - We can refine to “lock in place vs lock anywhere” later.
- Where Last.fm history lives:
  - cached in SQLite vs live API calls
- Persist playlist edits server-side (optional post-MVP)

---

## 13) Immediate next steps (what you do next)
1. Create `api/` + `ui/` folders and commit the empty skeleton.
2. Implement Milestone 1 (`/api/library/status` + UI display).
3. Implement Milestone 2 (seed search + modal).
4. Implement Milestone 3 (track seed generate + show results).
