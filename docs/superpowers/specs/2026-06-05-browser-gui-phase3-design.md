# Browser GUI — Phase 3: Diagnostics, Blacklist Management, Jobs, Decommission

**Status:** Design approved (brainstorm 2026-06-05)
**Supersedes:** nothing — extends Phase 2 (`2026-06-04-browser-gui-phase2-design.md`)
**Excluded from this phase:** Genre Enrichment slide-over (separate initiative in progress)

---

## 1. Scope

Four self-contained subsystems, executed in order:

| # | Subsystem | Files touched |
|---|-----------|---------------|
| 1 | Right panel: Diagnostics tab | New `DiagnosticsPanel.tsx`, `AdvancedPanel.tsx`, `schemas.py`, `worker.py` |
| 2 | Right panel: Blacklist tab | New `BlacklistPanel.tsx`, `app.py`, `schemas.py`, `api.ts`, `types.ts` |
| 3 | Jobs panel enhancements | `JobsPanel.tsx`, `jobs.py`, `app.py`, `schemas.py`, `worker_bridge.py`, `App.tsx` |
| 4 | PySide6 decommission | `pyproject.toml`, `playlist_gui/app.py`, docs |

---

## 2. Subsystem 1 — Diagnostics Tab

### 2.1 What it shows

The right panel gains a **Diagnostics** tab (alongside the existing Genre Review stub and the new Blacklist tab). Content is read-only, derived from the most recent generation. Empty state when no playlist has been generated yet.

**Layout (top → bottom, scrollable):**

```
SUMMARY                        
Mean   0.61   Min    0.44
P10    0.49   P90    0.74
Artists  14   Tracks   20

⚠ WEAKEST EDGE
track 08 → 09   T = 0.41

TRANSITIONS
1→2  ████████████░░░░  0.74
2→3  ██████████░░░░░░  0.68
3→4  ██████░░░░░░░░░░  0.55   ← orange
4→5  ████░░░░░░░░░░░░  0.41   ← red
...
```

Color thresholds for transition bars:
- T ≥ 0.60 → mint (`#5eead4`)
- 0.40 ≤ T < 0.60 → orange (`#f97316`)
- T < 0.40 → red (`#ef4444`)

### 2.2 Data changes required

**Worker (`src/playlist_gui/worker.py`):**
- Add `"transition_score"` to each formatted track from the `edge_map` (already built at line 1283). The edge dict contains `S` (sonic) and `G` (genre) component scores — check whether a combined `T` key exists; if not, compute it as `round(0.5 * S + 0.5 * G, 4)` as a fallback (or read the actual tower weights from config). Null for the last track, which has no outgoing edge.
- Add `"p10_transition"` and `"p90_transition"` to `playlist_result["metrics"]`. These exist in `ds_report["metrics"]` but are currently omitted.

**Schemas (`src/playlist_web/schemas.py`):**
- Add `transition_score: Optional[float] = None` to `TrackOut`.
- Add `p10_transition` and `p90_transition` to `MetricsOut.from_worker()` mapping (already in the model, just not populated).
- Add `transition_score` to `PlaylistOut.from_worker()` track mapping.

**Frontend (`web/src/lib/types.ts`):**
- Add `transition_score?: number | null` to `TrackOut`.

### 2.3 New component: `DiagnosticsPanel.tsx`

Props: `playlist: PlaylistOut | null`

- Renders empty state ("Generate a playlist to see diagnostics") when `playlist` is null.
- Summary grid: 2-column, 3 rows — Mean/Min, P10/P90, Artists/Tracks.
- Weakest edge box: finds the track with the lowest `transition_score`, shows position label (`track N → N+1`) and score. Styled with red border and background.
- Transition bar list: one row per track except the last. Bar width = `transition_score * 100%`, color per threshold above.

### 2.4 Modified: `AdvancedPanel.tsx`

Replace the two placeholder tabs with three real tabs:
- **Diagnostics** → `<DiagnosticsPanel playlist={playlist} />`
- **Blacklist** → `<BlacklistPanel />` (see §3)
- **Genre Review** → keep existing stub ("lands in a later phase")

`AdvancedPanel` receives `playlist: PlaylistOut | null` as a prop. `App.tsx` passes it down.

---

## 3. Subsystem 2 — Blacklist Tab

### 3.1 What it shows

Three sections: **Artists**, **Albums**, **Tracks** — each with a count badge. Each entry has a × remove button. At the top: a search input + **Add** button to blacklist an artist directly (autocomplete against `/api/autocomplete`). Albums and tracks continue to be blacklisted via the track table context menu.

```
[artist name search…]  [+ Add]

ARTISTS  2
● Nick Drake            ×
● Coldplay              ×

ALBUMS   1
◆ Pink Moon             ×

TRACKS   3
▸ Harvest               ×
▸ Hotel California      ×
```

### 3.2 New API route: `GET /api/blacklist`

Calls worker `blacklist_fetch` command (already implemented in `worker.py`). Returns:

```python
class BlacklistEntryOut(BaseModel):
    track_id: Optional[str] = None   # for track-scope entries
    artist: Optional[str] = None
    album: Optional[str] = None
    scope: str                        # "track" | "artist" | "album"
    display_name: str                 # human-readable label

class BlacklistFetchResponse(BaseModel):
    artists: list[BlacklistEntryOut]
    albums: list[BlacklistEntryOut]
    tracks: list[BlacklistEntryOut]
    total: int
```

The worker's `blacklist_fetch` returns raw track data; the route groups and formats it into this response.

### 3.3 New API route: `POST /api/blacklist/artist`

Simplified route for adding an artist by name from the Blacklist tab search box. Body: `{"artist": str}`. Internally calls the existing `POST /api/blacklist` with `scope="artist", value=artist`.

### 3.4 New component: `BlacklistPanel.tsx`

- On mount: fetches `GET /api/blacklist` and stores result in local state.
- Search input: debounced autocomplete against `/api/autocomplete`. Dropdown shows artist name options. Selecting one populates the input.
- **Add** button (or Enter): calls `POST /api/blacklist/artist`, then refetches.
- Remove (×): calls existing `POST /api/blacklist` with `enabled: false` for the appropriate scope/value, then refetches.
- Shows a "Refreshing…" indicator during any mutation.
- Empty state: "Nothing blacklisted yet. Use the track table context menu or search above."

---

## 4. Subsystem 3 — Jobs Panel Enhancements

### 4.1 Per-job additions

**Running jobs:**
- Thin animated progress bar (driven by existing `stage` field).
- **✕ Cancel** button → calls `POST /api/jobs/{job_id}/cancel`.

**Completed jobs:**
- Timestamp (time of completion, formatted as `HH:MM`).
- Mean T score next to track count: `20 tracks · T̄ 0.61 · 14:32`.
- **↺ Re-run** button → loads that job's original generation params back into the form and resets to "ready to generate".
- **Restore** button (existing behavior) → loads the job's playlist result into the center panel.

**Failed / cancelled jobs:**
- **↺ Re-run** button.
- Timestamp.

### 4.2 Job registry changes (`src/playlist_web/jobs.py`)

Add two fields to the internal job record:
- `created_at: float` — `time.time()` at job creation.
- `request_params: dict` — serialized `GenerateRequestBody` stored at creation time.

`JobOut` schema gains:
- `created_at: Optional[float] = None`
- `request_params: Optional[dict] = None`

`JobRegistry.create(request_params: dict | None = None)` stores params at creation.

The `/api/generate` route passes `body.model_dump()` as `request_params` to `registry.create()`.

### 4.3 Cancel route: `POST /api/jobs/{job_id}/cancel`

Adds a `WorkerBridge.cancel()` method:
```python
async def cancel(self) -> None:
    """Fire-and-forget cancel for the currently running request."""
    if self._active_request_id and self._proc and self._proc.stdin:
        import json, uuid
        cmd = {"cmd": "cancel", "request_id": self._active_request_id}
        line = json.dumps(cmd) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()
```

The route:
1. Looks up the job — 404 if not found.
2. Returns 409 if job is not in `"running"` status.
3. Calls `bridge.cancel()`.
4. Returns `{"ok": True}`.

The job's status transitions from `"running"` to `"cancelled"` when the worker emits its done event with `cancelled=True` (handled by existing `registry.apply_event()`).

### 4.4 Re-run flow

`JobsPanel` receives an `onRerun: (params: GenerateRequestBody) => void` prop. When clicked, it calls this with `job.request_params` cast to `GenerateRequestBody`.

`App.tsx`:
- Adds `rerunKey: number` state (increments on each re-run to force `GenerateControls` remount).
- Adds `rerunValues: GenerateRequestBody | null` state.
- Handler: sets `rerunValues` + increments `rerunKey`.
- `GenerateControls` receives optional `initialValues?: Partial<GenerateRequestBody>` prop. On mount, if `initialValues` is present, it overrides the localStorage-seeded defaults for that session. This avoids writing to localStorage directly.

---

## 5. Subsystem 4 — PySide6 Decommission

### 5.1 Scope

The Qt GUI files (`main_window.py`, widgets/, etc.) remain on disk but PySide6 is removed from the install requirements so it is no longer needed to run the app. The worker subprocess (`worker.py`) is not Qt-bound and continues unchanged.

### 5.2 Changes

**`pyproject.toml`:**
- Remove `PySide6>=6.6` from `[project.optional-dependencies] gui`.
- Rename the extra from `[gui]` to `[web]` (contains only `uvicorn`, `fastapi`, `pydantic`).
- Keep a `[legacy-gui]` extra for anyone who still wants to run the Qt app: `PySide6>=6.6`.
- Install instructions in README become: `pip install -e .[web]`.

**`playlist_gui/app.py`:**
- Already guards PySide6 with a try/import check. Add a clear deprecation notice at the top: the Qt GUI is no longer actively maintained; use `python tools/serve_web.py`.

**Docs:**
- Update `CLAUDE.md` GUI line: `python tools/serve_web.py` (port 8770).
- Update `README.md` install and launch instructions.
- Update `docs/GOLDEN_COMMANDS.md` if it references `python -m playlist_gui.app`.

**Verification:** After these changes, `pip install -e .[web]` completes without installing PySide6, and `python tools/serve_web.py` works.

---

## 6. API surface summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/blacklist` | Fetch full blacklist, grouped by scope |
| `POST` | `/api/blacklist/artist` | Add an artist to the blacklist by name |
| `POST` | `/api/jobs/{job_id}/cancel` | Cancel a running generation |

Existing routes unchanged. `TrackOut` and `MetricsOut` gain new optional fields (backward-compatible).

---

## 7. File map

**New files:**
- `web/src/components/DiagnosticsPanel.tsx`
- `web/src/components/BlacklistPanel.tsx`

**Modified files:**
- `src/playlist_gui/worker.py` — add `transition_score`, `p10_transition`, `p90_transition` to output
- `src/playlist_web/schemas.py` — `TrackOut.transition_score`, `MetricsOut` p10/p90 mapping, `BlacklistEntryOut`, `BlacklistFetchResponse`, `JobOut` new fields
- `src/playlist_web/jobs.py` — `created_at`, `request_params` per job
- `src/playlist_web/app.py` — new routes + cancel logic
- `src/playlist_web/worker_bridge.py` — `cancel()` method
- `web/src/components/AdvancedPanel.tsx` — real tabs, accepts `playlist` prop
- `web/src/components/JobsPanel.tsx` — cancel, re-run, quality score, timestamp
- `web/src/lib/types.ts` — `TrackOut.transition_score`, `JobOut` new fields, `BlacklistEntryOut`, `BlacklistFetchResponse`
- `web/src/lib/api.ts` — new endpoints
- `web/src/App.tsx` — plumb `playlist` to `AdvancedPanel`, re-run handler, cancel handler
- `web/src/components/GenerateControls.tsx` — `initialValues` prop for re-run
- `pyproject.toml` — dependency restructure
- `CLAUDE.md`, `README.md`, `docs/GOLDEN_COMMANDS.md` — doc updates

---

## 8. Testing

**Backend:**
- `tests/integration/test_web_api_phase3.py`:
  - `GET /api/blacklist` returns grouped structure with correct fields
  - `POST /api/blacklist/artist` adds entry and fetch reflects it
  - `POST /api/jobs/{job_id}/cancel` returns 404 for unknown job, 409 for non-running job, 200 for running job
  - `TrackOut` includes `transition_score` when worker provides edge data
  - `MetricsOut` includes `p10_transition` and `p90_transition`
  - `JobOut` includes `created_at` and `request_params`

**Frontend (Playwright `web/tests/phase3.spec.ts`):**
  - Diagnostics tab shows "Generate a playlist" empty state
  - After generation, transition bars appear and weakest edge is highlighted
  - Blacklist tab shows empty state, artist search adds an entry, × removes it
  - Running job shows cancel button; completed job shows re-run + T̄ score
