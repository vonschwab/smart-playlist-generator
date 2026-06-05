# Browser GUI Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Diagnostics tab, a Blacklist management tab, Jobs-panel cancel/re-run/quality-score, and deprecate the PySide6 GUI in favor of the browser app.

**Architecture:** Three new feature areas plus a decommission pass. Backend data flows through the existing single-active-request worker (NDJSON over stdio) via `WorkerBridge`; the FastAPI layer adds three routes and never touches the databases directly for mutations. Frontend adds two React components (`DiagnosticsPanel`, `BlacklistPanel`) into the existing right-panel tab host, and enriches `JobsPanel`. All new track/metric/job fields are additive and backward-compatible.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, pytest (`asyncio_mode=auto`); React 18 + TypeScript + Vite 6 + Tailwind v4; Playwright for e2e. Tests inject a fake worker via the `worker_cmd` argument to `create_app` (see `tests/fixtures/fake_worker.py`).

**Spec:** `docs/superpowers/specs/2026-06-05-browser-gui-phase3-design.md`

**Branch:** Work continues on `browser-gui-phase1` (where Phase 2 landed).

---

## Reference: key facts verified against the codebase

- **Edge transition score:** The worker builds `edge_map` at `src/playlist_gui/worker.py:1283` from `edge_scores`; each edge dict has a combined `"T"` key (plus `"S"`, `"G"`). Use `edge.get("T")`.
- **Metrics:** `MetricsOut.from_worker()` (`src/playlist_web/schemas.py`) already maps `p10_transition`/`p90_transition`. The worker currently only emits `mean_transition`/`min_transition`/`distinct_artists` at `worker.py:1364-1368`.
- **Blacklist storage:** three tables — `artist_blacklist(artist_key, artist_name, last_updated)`, `album_blacklist(artist_key, album_key, artist_name, album_name, last_updated)`, `track_blacklist(track_id, title, artist, album, last_updated)`. The `tracks.is_blacklisted` flag is *derived* from these. `fetch_blacklisted_tracks()` returns derived-flag rows (wrong for a grouped view) — read the scope tables instead.
- **Worker command registration:** add tracked commands to `TRACKED_COMMAND_HANDLERS` (`worker.py:2287`). `emit_result(result_type, data_dict)` and `emit_done(cmd, ok, detail)` are the output helpers; the worker injects `request_id` automatically.
- **Bridge:** `WorkerBridge.command(cmd, timeout=60.0)` returns the captured `result` event dict (data fields spread at top level). `WorkerBridge` has `_active_request_id` and `_proc`. `submit()` is fire-and-forget for generation.
- **Job registry:** `JobRegistry.create()` (`src/playlist_web/jobs.py:38`) currently takes no args and sets status `running`. `JobOut` (`schemas.py:117`) has `job_id, status, stage, error, playlist`.
- **Right panel host:** `AdvancedPanel` (`web/src/components/AdvancedPanel.tsx`) renders into `Shell`'s `right` slot from `App.tsx:169`.
- **Tests:** backend uses `from fastapi.testclient import TestClient` + `create_app(worker_cmd=FAKE)` where `FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]`. Frontend e2e uses Playwright (`npm --prefix web run test:e2e`) and `data-testid` selectors. Build gate: `npm --prefix web run build`.

---

## File Structure

**New files:**
- `web/src/components/DiagnosticsPanel.tsx` — read-only metrics + per-edge transition bars.
- `web/src/components/BlacklistPanel.tsx` — grouped blacklist view with add-artist + remove.
- `tests/integration/test_web_api_phase3.py` — backend route/schema tests.
- `web/tests/phase3.spec.ts` — Playwright e2e for the new UI.

**Modified files:**
- `src/metadata_client.py` — `fetch_artist_blacklist()`, `fetch_album_blacklist()`, `fetch_track_blacklist()`.
- `src/playlist_gui/worker.py` — emit `transition_score`/p10/p90; new `blacklist_fetch_scopes` command.
- `src/playlist_web/schemas.py` — `TrackOut.transition_score`; `BlacklistEntryOut`/`BlacklistFetchResponse`/`BlacklistArtistRequest`; `JobOut.created_at`/`request_params`.
- `src/playlist_web/jobs.py` — store `created_at`/`request_params`.
- `src/playlist_web/worker_bridge.py` — `cancel()` method.
- `src/playlist_web/app.py` — `GET /api/blacklist`, `POST /api/blacklist/artist`, `POST /api/jobs/{job_id}/cancel`; pass `request_params` to `registry.create()`.
- `tests/fixtures/fake_worker.py` — handle `blacklist_fetch_scopes`.
- `web/src/lib/types.ts` — `TrackOut.transition_score`; `JobOut` fields; blacklist types.
- `web/src/lib/api.ts` — `getBlacklist()`, `blacklistArtist()`, `cancelJob()`.
- `web/src/components/AdvancedPanel.tsx` — three real tabs, accepts `playlist`.
- `web/src/components/JobsPanel.tsx` — cancel/re-run/score/timestamp.
- `web/src/components/GenerateControls.tsx` — `initialValues` prop.
- `web/src/App.tsx` — plumb `playlist` to `AdvancedPanel`; cancel + re-run handlers.
- `pyproject.toml`, `CLAUDE.md`, `README.md`, `docs/GOLDEN_COMMANDS.md` — decommission docs.

---

## SUBSYSTEM 1 — Diagnostics Tab

### Task 1: Worker emits transition_score per track + p10/p90 metrics

**Files:**
- Modify: `src/playlist_gui/worker.py` (track formatting ~1340-1353, metrics ~1362-1368)

- [ ] **Step 1: Add `transition_score` to the formatted track dict**

In `src/playlist_gui/worker.py`, locate the `formatted_tracks.append({...})` block (around line 1340). The `edge` variable is already in scope from `edge = edge_map.get(str(rating_key), {})` (line 1314). Add a `transition_score` key after `"genres": genres,`:

```python
                formatted_tracks.append({
                    "position": i,
                    "rating_key": rating_key,
                    "artist": track.get('artist', 'Unknown'),
                    "title": track.get('title', 'Unknown'),
                    "album": track.get('album', ''),
                    "duration_ms": track.get('duration', 0),
                    "file_path": track.get('file_path', ''),
                    "sonic_similarity": sonic_sim,
                    "genre_similarity": genre_sim,
                    "sonic_similarity_components": sonic_comp,
                    "genre_similarity_components": genre_comp,
                    "genres": genres,
                    "transition_score": edge.get("T"),
                })
```

- [ ] **Step 2: Add p10/p90 to the metrics dict**

Locate the metrics block (around line 1362):

```python
            if ds_report:
                metrics = ds_report.get('metrics', {})
                playlist_result["metrics"] = {
                    "mean_transition": metrics.get('mean_transition'),
                    "min_transition": metrics.get('min_transition'),
                    "p10_transition": metrics.get('p10_transition'),
                    "p90_transition": metrics.get('p90_transition'),
                    "distinct_artists": metrics.get('distinct_artists'),
                }
```

- [ ] **Step 3: Update the fake worker so e2e/integration sees the new fields**

In `tests/fixtures/fake_worker.py`, update the `generate_playlist` playlist payload: add `"transition_score"` to each track and `p10_transition`/`p90_transition` to metrics. Replace the two track dicts and the metrics dict inside the `generate_playlist` branch:

```python
                "tracks": [
                    {"position": 0, "rating_key": "k0", "artist": "Acetone", "title": "Sundown",
                     "album": "Cindy", "duration_ms": 200000, "file_path": "/0.flac",
                     "sonic_similarity": 0.91, "genre_similarity": 0.8, "genres": ["slowcore"],
                     "transition_score": 0.62},
                    {"position": 1, "rating_key": "k1", "artist": "Mazzy Star", "title": "Taxi",
                     "album": "So Tonight", "duration_ms": 210000, "file_path": "/1.flac",
                     "sonic_similarity": 0.87, "genre_similarity": 0.7, "genres": ["dreampop"],
                     "transition_score": None},
                ],
                "metrics": {"mean_transition": 0.89, "min_transition": 0.87,
                            "p10_transition": 0.62, "p90_transition": 0.91, "distinct_artists": 2},
```

- [ ] **Step 4: Commit**

```bash
git add src/playlist_gui/worker.py tests/fixtures/fake_worker.py
git commit -m "feat(worker): emit per-track transition_score and p10/p90 metrics"
```

---

### Task 2: Schema + types for transition_score

**Files:**
- Modify: `src/playlist_web/schemas.py` (`TrackOut`, `PlaylistOut.from_worker`)
- Modify: `web/src/lib/types.ts` (`TrackOut`)
- Test: `tests/integration/test_web_api_phase3.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_web_api_phase3.py`:

```python
import sys

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


def test_playlist_out_maps_transition_score_and_percentiles():
    from src.playlist_web.schemas import PlaylistOut

    raw = {
        "name": "X", "track_count": 2,
        "tracks": [
            {"position": 0, "rating_key": "k0", "artist": "A", "title": "T0",
             "album": "Al", "duration_ms": 1, "file_path": "/0", "genres": [],
             "transition_score": 0.62},
            {"position": 1, "rating_key": "k1", "artist": "B", "title": "T1",
             "album": "Al", "duration_ms": 1, "file_path": "/1", "genres": [],
             "transition_score": None},
        ],
        "metrics": {"mean_transition": 0.7, "min_transition": 0.5,
                    "p10_transition": 0.55, "p90_transition": 0.8, "distinct_artists": 2},
    }
    pl = PlaylistOut.from_worker(raw)
    assert pl.tracks[0].transition_score == 0.62
    assert pl.tracks[1].transition_score is None
    assert pl.metrics.p10_transition == 0.55
    assert pl.metrics.p90_transition == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase3.py::test_playlist_out_maps_transition_score_and_percentiles -v`
Expected: FAIL with `AttributeError` / validation error — `TrackOut` has no `transition_score`.

- [ ] **Step 3: Add the field and mapping**

In `src/playlist_web/schemas.py`, add to `TrackOut` (after `genre_similarity`):

```python
    transition_score: Optional[float] = None
```

In `PlaylistOut.from_worker`, add to each `TrackOut(...)` built in the list comprehension (after `genres=t.get("genres", []),`):

```python
                transition_score=t.get("transition_score"),
```

(`MetricsOut.from_worker` already maps p10/p90 — no change needed there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_web_api_phase3.py::test_playlist_out_maps_transition_score_and_percentiles -v`
Expected: PASS

- [ ] **Step 5: Add the TS type**

In `web/src/lib/types.ts`, add to `TrackOut` (after `genre_similarity?`):

```typescript
  transition_score?: number | null;
```

- [ ] **Step 6: Verify the frontend still builds**

Run: `npm --prefix web run build`
Expected: build succeeds, no TS errors.

- [ ] **Step 7: Commit**

```bash
git add src/playlist_web/schemas.py web/src/lib/types.ts tests/integration/test_web_api_phase3.py
git commit -m "feat(web): surface transition_score through schema and TS types"
```

---

### Task 3: DiagnosticsPanel component

**Files:**
- Create: `web/src/components/DiagnosticsPanel.tsx`

- [ ] **Step 1: Create the component**

Create `web/src/components/DiagnosticsPanel.tsx`:

```tsx
import type { PlaylistOut } from "../lib/types";

function barColor(t: number): string {
  if (t >= 0.6) return "#5eead4";
  if (t >= 0.4) return "#f97316";
  return "#ef4444";
}

function fmt(n: number | null | undefined): string {
  return typeof n === "number" ? n.toFixed(2) : "—";
}

export function DiagnosticsPanel({ playlist }: { playlist: PlaylistOut | null }) {
  if (!playlist || playlist.tracks.length === 0) {
    return (
      <div className="p-4 text-xs text-[#3a3f4b]" data-testid="diagnostics-empty">
        Generate a playlist to see diagnostics.
      </div>
    );
  }

  const m = playlist.metrics;
  // Edges = every track that has a transition_score (last track is null).
  const edges = playlist.tracks
    .map((t, i) => ({ i, score: t.transition_score }))
    .filter((e): e is { i: number; score: number } => typeof e.score === "number");

  const weakest = edges.length
    ? edges.reduce((min, e) => (e.score < min.score ? e : min), edges[0])
    : null;

  const stat = (label: string, value: string) => (
    <div className="flex justify-between items-baseline py-1 border-b border-[#1a1c21]">
      <span className="text-[9px] uppercase tracking-[.06em] text-[#5b6470]">{label}</span>
      <span className="text-[11px] font-bold font-mono text-[#5eead4]">{value}</span>
    </div>
  );

  return (
    <div className="p-3 overflow-y-auto text-xs" data-testid="diagnostics-content">
      <div className="text-[9px] uppercase tracking-[.08em] text-[#3a3f4b] mb-1">Summary</div>
      <div className="grid grid-cols-2 gap-x-3">
        {stat("Mean", fmt(m?.mean_transition))}
        {stat("Min", fmt(m?.min_transition))}
        {stat("P10", fmt(m?.p10_transition))}
        {stat("P90", fmt(m?.p90_transition))}
        {stat("Artists", m?.distinct_artists != null ? String(m.distinct_artists) : "—")}
        {stat("Tracks", String(playlist.track_count))}
      </div>

      {weakest && (
        <div className="mt-3 bg-[#1a1015] border border-[#3a1a1a] rounded p-2" data-testid="weakest-edge">
          <div className="text-[8px] uppercase tracking-[.06em] text-[#ef4444] mb-0.5">⚠ Weakest edge</div>
          <div className="text-[10px] text-[#e6e9ec]">track {weakest.i + 1} → {weakest.i + 2}</div>
          <div className="text-[8px] font-mono text-[#ef4444]">T = {weakest.score.toFixed(2)}</div>
        </div>
      )}

      <div className="text-[8px] uppercase tracking-[.08em] text-[#3a3f4b] mt-3 mb-1">Transitions</div>
      <div className="flex flex-col gap-0.5" data-testid="transition-bars">
        {edges.map((e) => (
          <div key={e.i} className="flex items-center gap-1.5">
            <span className="text-[8px] text-[#3a3f4b] w-7 text-right shrink-0">{e.i + 1}→{e.i + 2}</span>
            <div className="flex-1 h-1.5 bg-[#1a1c21] rounded-sm overflow-hidden">
              <div className="h-full rounded-sm" style={{ width: `${e.score * 100}%`, background: barColor(e.score) }} />
            </div>
            <span className="text-[8px] font-mono text-[#5b6470] w-8 text-right shrink-0">{e.score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it builds (component is imported in Task 4; build now to catch syntax errors early)**

Run: `npm --prefix web run build`
Expected: build succeeds. (TypeScript will report the file as unused-but-valid; that's fine — Vite only fails on type errors, not unused modules.)

- [ ] **Step 3: Commit**

```bash
git add web/src/components/DiagnosticsPanel.tsx
git commit -m "feat(web): add DiagnosticsPanel (summary + per-edge transition bars)"
```

---

### Task 4: Right-panel tabs + wire playlist into AdvancedPanel

**Files:**
- Modify: `web/src/components/AdvancedPanel.tsx`
- Modify: `web/src/App.tsx`

> Note: `BlacklistPanel` is built in Task 9. To keep this task self-contained and the build green, this task adds the **Diagnostics** and **Genre Review** tabs now and a placeholder Blacklist tab body ("Loading…"); Task 9 replaces the placeholder with `<BlacklistPanel />`.

- [ ] **Step 1: Rewrite AdvancedPanel with three tabs**

Replace the entire contents of `web/src/components/AdvancedPanel.tsx`:

```tsx
import { useState } from "react";
import type { PlaylistOut } from "../lib/types";
import { DiagnosticsPanel } from "./DiagnosticsPanel";

type Tab = "diagnostics" | "blacklist" | "review";

export function AdvancedPanel({ playlist }: { playlist: PlaylistOut | null }) {
  const [tab, setTab] = useState<Tab>("diagnostics");

  const tabBtn = (t: Tab, label: string) => (
    <button
      key={t}
      data-testid={`tab-${t}`}
      onClick={() => setTab(t)}
      className={`text-[11px] px-2.5 py-1.5 rounded-t ${tab === t ? "text-accent bg-bg" : "text-muted"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="h-full flex flex-col">
      <div className="flex gap-1 px-2 pt-2 bg-panel2">
        {tabBtn("diagnostics", "Diagnostics")}
        {tabBtn("blacklist", "Blacklist")}
        {tabBtn("review", "Genre Review")}
      </div>
      <div className="flex-1 overflow-hidden">
        {tab === "diagnostics" && <DiagnosticsPanel playlist={playlist} />}
        {tab === "blacklist" && (
          <div className="p-3 text-xs text-muted" data-testid="blacklist-placeholder">Loading…</div>
        )}
        {tab === "review" && (
          <div className="p-3 text-xs text-muted">Genre review lands in a later phase.</div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Pass `playlist` from App**

In `web/src/App.tsx`, find the `right={<AdvancedPanel />}` prop in the `<Shell>` element (line ~169) and replace with:

```tsx
        right={<AdvancedPanel playlist={playlist} />}
```

- [ ] **Step 3: Verify build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/AdvancedPanel.tsx web/src/App.tsx
git commit -m "feat(web): three-tab right panel with live Diagnostics"
```

---

## SUBSYSTEM 2 — Blacklist Tab

### Task 5: MetadataClient scope-fetch methods

**Files:**
- Modify: `src/metadata_client.py` (add three methods near `fetch_blacklisted_tracks`, ~line 576)
- Test: `tests/integration/test_metadata_blacklist_fetch.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_metadata_blacklist_fetch.py`:

```python
import tempfile
from pathlib import Path

from src.metadata_client import MetadataClient


def _client(tmp: Path) -> MetadataClient:
    return MetadataClient(str(tmp / "metadata.db"))


def test_fetch_scope_blacklists_round_trip():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        mc = _client(tmp)
        # Seed a couple of tracks so artist/album scope has something to flag.
        mc.add_track("t1", "Pink Moon", "Nick Drake", "Pink Moon")
        mc.add_track("t2", "Road", "Nick Drake", "Pink Moon")
        mc.add_track("t3", "Yellow", "Coldplay", "Parachutes")

        mc.set_artist_blacklisted("Coldplay", True)
        mc.set_album_blacklisted("Nick Drake", "Pink Moon", True)
        mc.set_blacklisted(["t1"], True)  # also individually blacklist a track

        artists = mc.fetch_artist_blacklist()
        albums = mc.fetch_album_blacklist()
        tracks = mc.fetch_track_blacklist()

        assert any(a["artist_name"] == "Coldplay" for a in artists)
        assert any(al["album_name"] == "Pink Moon" and al["artist_name"] == "Nick Drake" for al in albums)
        assert any(t["track_id"] == "t1" for t in tracks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_metadata_blacklist_fetch.py -v`
Expected: FAIL — `AttributeError: 'MetadataClient' object has no attribute 'fetch_artist_blacklist'`.

- [ ] **Step 3: Add the three methods**

In `src/metadata_client.py`, immediately before `def fetch_blacklisted_tracks` (line ~576), add:

```python
    def fetch_artist_blacklist(self) -> List[Dict[str, Any]]:
        """Return artist-scope blacklist entries."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT artist_key, artist_name FROM artist_blacklist ORDER BY artist_name"
        )
        return [
            {"artist_key": str(r["artist_key"] or ""), "artist_name": str(r["artist_name"] or "")}
            for r in cursor.fetchall()
        ]

    def fetch_album_blacklist(self) -> List[Dict[str, Any]]:
        """Return album-scope blacklist entries."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT artist_key, album_key, artist_name, album_name
            FROM album_blacklist
            ORDER BY artist_name, album_name
            """
        )
        return [
            {
                "artist_key": str(r["artist_key"] or ""),
                "album_key": str(r["album_key"] or ""),
                "artist_name": str(r["artist_name"] or ""),
                "album_name": str(r["album_name"] or ""),
            }
            for r in cursor.fetchall()
        ]

    def fetch_track_blacklist(self) -> List[Dict[str, Any]]:
        """Return individually blacklisted tracks (track-scope only)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT track_id, title, artist, album FROM track_blacklist ORDER BY artist, title"
        )
        return [
            {
                "track_id": str(r["track_id"] or ""),
                "title": str(r["title"] or ""),
                "artist": str(r["artist"] or ""),
                "album": str(r["album"] or ""),
            }
            for r in cursor.fetchall()
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_metadata_blacklist_fetch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metadata_client.py tests/integration/test_metadata_blacklist_fetch.py
git commit -m "feat(metadata): fetch_artist/album/track_blacklist scope readers"
```

---

### Task 6: Worker `blacklist_fetch_scopes` command

**Files:**
- Modify: `src/playlist_gui/worker.py` (new handler near `handle_blacklist_fetch` ~1810; register at `TRACKED_COMMAND_HANDLERS` ~2297)
- Modify: `tests/fixtures/fake_worker.py`

- [ ] **Step 1: Add the handler**

In `src/playlist_gui/worker.py`, immediately after `handle_blacklist_fetch` (ends ~line 1826), add:

```python
def handle_blacklist_fetch_scopes(cmd_data: Dict[str, Any]) -> None:
    """Fetch artist/album/track scope blacklists (grouped, for the web UI)."""
    base_path = cmd_data.get("base_config_path", "config.yaml")
    overrides = cmd_data.get("overrides", {})
    try:
        config = load_config_with_overrides(base_path, overrides)
        db_path = config.get('library', {}).get('database_path', 'data/metadata.db')
        from src.metadata_client import MetadataClient

        metadata = MetadataClient(db_path)
        artists = metadata.fetch_artist_blacklist()
        albums = metadata.fetch_album_blacklist()
        tracks = metadata.fetch_track_blacklist()
        emit_result(
            "blacklist_scopes",
            {"artists": artists, "albums": albums, "tracks": tracks},
        )
        total = len(artists) + len(albums) + len(tracks)
        emit_done("blacklist_fetch_scopes", True, f"Fetched {total} blacklist entries")
    except Exception as e:
        tb = traceback.format_exc()
        emit_error(str(e), tb)
        emit_done("blacklist_fetch_scopes", False, str(e))
```

- [ ] **Step 2: Register the command**

In `TRACKED_COMMAND_HANDLERS` (line ~2297), add after `"blacklist_fetch": handle_blacklist_fetch,`:

```python
    "blacklist_fetch_scopes": handle_blacklist_fetch_scopes,
```

- [ ] **Step 3: Teach the fake worker the command**

In `tests/fixtures/fake_worker.py`, add a new branch before the final `else:`:

```python
        elif name == "blacklist_fetch_scopes":
            emit({"type": "result", "result_type": "blacklist_scopes", "request_id": rid, "job_id": jid,
                  "artists": [{"artist_key": "nick drake", "artist_name": "Nick Drake"}],
                  "albums": [{"artist_key": "nick drake", "album_key": "pink moon",
                              "artist_name": "Nick Drake", "album_name": "Pink Moon"}],
                  "tracks": [{"track_id": "t1", "title": "Harvest", "artist": "Neil Young", "album": "Harvest"}]})
            emit({"type": "done", "cmd": name, "ok": True, "detail": "3 entries", "request_id": rid, "job_id": jid})
```

- [ ] **Step 4: Commit**

```bash
git add src/playlist_gui/worker.py tests/fixtures/fake_worker.py
git commit -m "feat(worker): blacklist_fetch_scopes command for grouped blacklist view"
```

---

### Task 7: Blacklist schemas + GET/POST routes

**Files:**
- Modify: `src/playlist_web/schemas.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_web_api_phase3.py`:

```python
from fastapi.testclient import TestClient
from src.playlist_web.app import create_app


def test_get_blacklist_groups_entries():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.get("/api/blacklist")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["artists"][0]["display_name"] == "Nick Drake"
        assert body["artists"][0]["scope"] == "artist"
        assert body["albums"][0]["album"] == "Pink Moon"
        assert body["albums"][0]["artist"] == "Nick Drake"
        assert body["albums"][0]["scope"] == "album"
        assert body["tracks"][0]["track_id"] == "t1"
        assert body["tracks"][0]["scope"] == "track"


def test_post_blacklist_artist_ok():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/blacklist/artist", json={"artist": "Coldplay"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase3.py::test_get_blacklist_groups_entries -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add the schemas**

In `src/playlist_web/schemas.py`, append:

```python
class BlacklistEntryOut(BaseModel):
    scope: str                              # "artist" | "album" | "track"
    display_name: str
    track_id: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None


class BlacklistFetchResponse(BaseModel):
    artists: list[BlacklistEntryOut] = Field(default_factory=list)
    albums: list[BlacklistEntryOut] = Field(default_factory=list)
    tracks: list[BlacklistEntryOut] = Field(default_factory=list)
    total: int = 0

    @classmethod
    def from_worker(cls, raw: dict) -> "BlacklistFetchResponse":
        artists = [
            BlacklistEntryOut(scope="artist", display_name=a.get("artist_name", ""),
                              artist=a.get("artist_name", ""))
            for a in raw.get("artists", [])
        ]
        albums = [
            BlacklistEntryOut(scope="album", display_name=al.get("album_name", ""),
                              artist=al.get("artist_name", ""), album=al.get("album_name", ""))
            for al in raw.get("albums", [])
        ]
        tracks = [
            BlacklistEntryOut(scope="track",
                              display_name=t.get("title", "") or t.get("track_id", ""),
                              track_id=t.get("track_id", ""), artist=t.get("artist", ""),
                              album=t.get("album", ""))
            for t in raw.get("tracks", [])
        ]
        return cls(artists=artists, albums=albums, tracks=tracks,
                   total=len(artists) + len(albums) + len(tracks))


class BlacklistArtistRequest(BaseModel):
    artist: str
```

- [ ] **Step 4: Add the routes**

In `src/playlist_web/app.py`, update the schema import block to include the new names:

```python
from .schemas import (
    BlacklistArtistRequest,
    BlacklistFetchResponse,
    BlacklistRequest,
    EditGenresRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsRequest,
    ReplaceSuggestionsResponse,
)
```

Then add these routes after the existing `blacklist` POST route (after the block ending ~line 168):

```python
    @app.get("/api/blacklist")
    async def get_blacklist() -> BlacklistFetchResponse:
        try:
            result = await bridge.command({
                "cmd": "blacklist_fetch_scopes",
                "base_config_path": config_path,
                "overrides": {},
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        return BlacklistFetchResponse.from_worker(result)

    @app.post("/api/blacklist/artist")
    async def blacklist_artist(body: BlacklistArtistRequest) -> dict:
        if not body.artist.strip():
            raise HTTPException(status_code=422, detail="artist is required")
        try:
            result = await bridge.command({
                "cmd": "blacklist_scope_set",
                "base_config_path": config_path,
                "overrides": {},
                "scope": "artist",
                "value": body.artist,
                "artist": body.artist,
                "enabled": True,
            })
        except BridgeBusy:
            raise HTTPException(status_code=409, detail="A generation is in progress — try again when it finishes.")
        except WorkerCommandError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"ok": True, **result}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase3.py -v`
Expected: PASS (both new tests).

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/schemas.py src/playlist_web/app.py tests/integration/test_web_api_phase3.py
git commit -m "feat(web): GET /api/blacklist + POST /api/blacklist/artist"
```

---

### Task 8: Frontend blacklist types + api client

**Files:**
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/lib/api.ts`

- [ ] **Step 1: Add the types**

In `web/src/lib/types.ts`, append:

```typescript
export interface BlacklistEntry {
  scope: "artist" | "album" | "track";
  display_name: string;
  track_id?: string | null;
  artist?: string | null;
  album?: string | null;
}

export interface BlacklistFetchResponse {
  artists: BlacklistEntry[];
  albums: BlacklistEntry[];
  tracks: BlacklistEntry[];
  total: number;
}
```

- [ ] **Step 2: Add api methods**

In `web/src/lib/api.ts`, add `BlacklistFetchResponse` to the type import block, then add inside the `api` object (after `blacklist(...)`):

```typescript
  async getBlacklist(): Promise<BlacklistFetchResponse> {
    return jsonOrThrow(await fetch("/api/blacklist"));
  },
  async blacklistArtist(artist: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch("/api/blacklist/artist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artist }),
    }));
  },
```

- [ ] **Step 3: Verify build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web): blacklist API client + types"
```

---

### Task 9: BlacklistPanel component + wire into tab

**Files:**
- Create: `web/src/components/BlacklistPanel.tsx`
- Modify: `web/src/components/AdvancedPanel.tsx`

- [ ] **Step 1: Create the component**

Create `web/src/components/BlacklistPanel.tsx`:

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { BlacklistEntry, BlacklistFetchResponse } from "../lib/types";

const DOT: Record<string, string> = { artist: "#ef4444", album: "#f97316", track: "#a855f7" };

export function BlacklistPanel() {
  const [data, setData] = useState<BlacklistFetchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const timer = useRef<number | undefined>(undefined);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      setData(await api.getBlacklist());
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (q.length < 2) { setSuggestions([]); return; }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      setSuggestions(await api.autocomplete(q).catch(() => []));
    }, 180);
  }, [q]);

  useEffect(() => {
    function onOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) setSuggestions([]);
    }
    document.addEventListener("mousedown", onOutside);
    return () => document.removeEventListener("mousedown", onOutside);
  }, []);

  async function addArtist() {
    const name = q.trim();
    if (!name) return;
    setBusy(true);
    try {
      await api.blacklistArtist(name);
      setQ("");
      setSuggestions([]);
      await refresh();
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  async function remove(entry: BlacklistEntry) {
    setBusy(true);
    try {
      if (entry.scope === "artist") {
        await api.blacklist({ scope: "artist", value: entry.artist ?? "", enabled: false });
      } else if (entry.scope === "album") {
        await api.blacklist({ scope: "album", value: entry.album ?? "", artist: entry.artist ?? "", enabled: false });
      } else {
        await api.blacklist({ track_ids: [entry.track_id ?? ""], enabled: false });
      }
      await refresh();
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }

  const section = (title: string, entries: BlacklistEntry[]) => (
    <div>
      <div className="flex justify-between text-[8px] uppercase tracking-[.08em] text-[#5b6470] mt-2 mb-1">
        <span>{title}</span><span className="text-[#3a3f4b]">{entries.length}</span>
      </div>
      {entries.map((e, i) => (
        <div key={`${e.scope}-${i}`} className="flex items-center gap-1.5 py-0.5 border-b border-[#1a1c21]">
          <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: DOT[e.scope] }} />
          <span className="text-[10px] text-[#8b939d] flex-1 truncate">{e.display_name}</span>
          <button onClick={() => remove(e)} className="text-[#3a3f4b] hover:text-[#ef4444] text-sm leading-none">×</button>
        </div>
      ))}
    </div>
  );

  const empty = data && data.total === 0;

  return (
    <div className="p-3 overflow-y-auto text-xs" data-testid="blacklist-panel">
      <div ref={dropdownRef} className="relative flex gap-1 mb-2">
        <input
          data-testid="blacklist-search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addArtist()}
          placeholder="Artist to blacklist…"
          className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded text-[10px] text-[#e6e9ec] px-2 py-1"
        />
        <button data-testid="blacklist-add" onClick={addArtist} disabled={busy}
          className="text-[10px] bg-[#1d3a35] text-[#5eead4] rounded px-2 disabled:opacity-50">+ Add</button>
        {suggestions.length > 0 && (
          <ul className="absolute z-20 top-8 left-0 right-0 bg-[#16181d] border border-[#23262d] rounded shadow-xl max-h-40 overflow-auto">
            {suggestions.map((s) => (
              <li key={s} onClick={() => { setQ(s); setSuggestions([]); }}
                className="px-2 py-1 text-[10px] text-[#e6e9ec] hover:bg-[#1e2229] cursor-pointer">{s}</li>
            ))}
          </ul>
        )}
      </div>

      {error && <div className="text-[#ef4444] text-[10px] mb-2">{error}</div>}
      {busy && <div className="text-[#3a3f4b] text-[9px] mb-1">Refreshing…</div>}

      {empty ? (
        <div className="text-[#3a3f4b] text-[10px]">Nothing blacklisted yet. Use the track table context menu or search above.</div>
      ) : data ? (
        <>
          {section("Artists", data.artists)}
          {section("Albums", data.albums)}
          {section("Tracks", data.tracks)}
        </>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 2: Wire it into the tab**

In `web/src/components/AdvancedPanel.tsx`, add the import at the top:

```tsx
import { BlacklistPanel } from "./BlacklistPanel";
```

Replace the placeholder blacklist body:

```tsx
        {tab === "blacklist" && <BlacklistPanel />}
```

- [ ] **Step 3: Verify build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/BlacklistPanel.tsx web/src/components/AdvancedPanel.tsx
git commit -m "feat(web): BlacklistPanel with grouped view, add-artist, remove"
```

---

## SUBSYSTEM 3 — Jobs Panel

### Task 10: Job registry stores created_at + request_params

**Files:**
- Modify: `src/playlist_web/jobs.py`
- Modify: `src/playlist_web/schemas.py` (`JobOut`)
- Modify: `src/playlist_web/app.py` (pass params to `create`)
- Test: `tests/integration/test_web_api_phase3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_web_api_phase3.py`:

```python
def test_generate_records_created_at_and_params():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone", "tracks": 5})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        detail = client.get(f"/api/jobs/{job_id}").json()
        assert detail["created_at"] is not None
        assert detail["request_params"]["artist"] == "Acetone"
        assert detail["request_params"]["tracks"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase3.py::test_generate_records_created_at_and_params -v`
Expected: FAIL — `created_at` key missing / is None.

- [ ] **Step 3: Update `_JobState` and `JobRegistry.create`**

In `src/playlist_web/jobs.py`, add `import time` at the top (after `import uuid`). Extend `_JobState.__init__`:

```python
    def __init__(self, job_id: str, max_log_lines: int, request_params: Optional[dict] = None):
        self.job_id = job_id
        self.status = "pending"
        self.stage = ""
        self.error: Optional[str] = None
        self.playlist: Optional[PlaylistOut] = None
        self.logs: deque[str] = deque(maxlen=max_log_lines)
        self.created_at: float = time.time()
        self.request_params: dict = request_params or {}
```

Update `create`:

```python
    def create(self, request_params: Optional[dict] = None) -> str:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = _JobState(job_id, self._max_log_lines, request_params)
        self._jobs[job_id].status = "running"
        while len(self._jobs) > self._max_jobs:
            self._jobs.popitem(last=False)
        return job_id
```

Update `_to_out`:

```python
    def _to_out(self, job: _JobState) -> JobOut:
        """Convert internal state to response model."""
        return JobOut(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            error=job.error,
            playlist=job.playlist,
            created_at=job.created_at,
            request_params=job.request_params,
        )
```

- [ ] **Step 4: Extend `JobOut`**

In `src/playlist_web/schemas.py`, add to `JobOut` (after `playlist`):

```python
    created_at: Optional[float] = None
    request_params: Optional[dict] = None
```

- [ ] **Step 5: Pass params at creation**

In `src/playlist_web/app.py`, in the `/api/generate` route, change `job_id = registry.create()` to:

```python
        job_id = registry.create(request_params=body.model_dump())
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/integration/test_web_api_phase3.py::test_generate_records_created_at_and_params -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/playlist_web/jobs.py src/playlist_web/schemas.py src/playlist_web/app.py tests/integration/test_web_api_phase3.py
git commit -m "feat(web): jobs record created_at and request_params"
```

---

### Task 11: Cancel — bridge method + route

**Files:**
- Modify: `src/playlist_web/worker_bridge.py`
- Modify: `src/playlist_web/app.py`
- Test: `tests/integration/test_web_api_phase3.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_web_api_phase3.py`:

```python
def test_cancel_unknown_job_404():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        resp = client.post("/api/jobs/does-not-exist/cancel")
        assert resp.status_code == 404


def test_cancel_completed_job_409():
    with TestClient(create_app(worker_cmd=FAKE)) as client:
        # Generate against the fake worker; it completes near-instantly.
        job_id = client.post("/api/generate", json={"mode": "artist", "artist": "Acetone"}).json()["job_id"]
        # Poll until the job is no longer running.
        import time as _t
        for _ in range(50):
            status = client.get(f"/api/jobs/{job_id}").json()["status"]
            if status != "running":
                break
            _t.sleep(0.05)
        resp = client.post(f"/api/jobs/{job_id}/cancel")
        assert resp.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_web_api_phase3.py::test_cancel_unknown_job_404 -v`
Expected: FAIL — 404 not produced because route is undefined (FastAPI returns 405/404 for missing route; assertion on 404 may pass incidentally for unknown path — to be safe the route must exist). Expected real failure: `test_cancel_completed_job_409` returns 404 (route missing).

- [ ] **Step 3: Add `cancel()` to the bridge**

In `src/playlist_web/worker_bridge.py`, add a method after `command(...)`:

```python
    async def cancel(self) -> bool:
        """Fire-and-forget cancel for the currently running request.

        Returns True if a cancel was dispatched, False if nothing was active.
        """
        if not (self._active_request_id and self._proc and self._proc.stdin):
            return False
        cmd = {"cmd": "cancel", "request_id": self._active_request_id}
        line = (json.dumps(cmd) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()
        return True
```

- [ ] **Step 4: Add the route**

In `src/playlist_web/app.py`, add after the `job_logs` route (~line 95):

```python
    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict:
        job = registry.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "running":
            raise HTTPException(status_code=409, detail="Job is not running")
        await bridge.cancel()
        return {"ok": True}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/integration/test_web_api_phase3.py::test_cancel_unknown_job_404 tests/integration/test_web_api_phase3.py::test_cancel_completed_job_409 -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_web/worker_bridge.py src/playlist_web/app.py tests/integration/test_web_api_phase3.py
git commit -m "feat(web): cancel running generation via POST /api/jobs/{id}/cancel"
```

---

### Task 12: Frontend JobOut fields + api.cancelJob

**Files:**
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/lib/api.ts`

- [ ] **Step 1: Extend `JobOut`**

In `web/src/lib/types.ts`, add to `JobOut` (after `playlist?`):

```typescript
  created_at?: number | null;
  request_params?: Record<string, unknown> | null;
```

- [ ] **Step 2: Add `cancelJob`**

In `web/src/lib/api.ts`, add inside the `api` object:

```typescript
  async cancelJob(jobId: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }));
  },
```

- [ ] **Step 3: Verify build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web): JobOut created_at/request_params + cancelJob client"
```

---

### Task 13: JobsPanel UI + App wiring + GenerateControls initialValues

**Files:**
- Modify: `web/src/components/JobsPanel.tsx`
- Modify: `web/src/components/GenerateControls.tsx`
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Add `initialValues` support to GenerateControls**

In `web/src/components/GenerateControls.tsx`, extend the props type and consume `initialValues`. Change the component signature:

```tsx
export function GenerateControls({
  mode,
  onModeChange,
  seedTrackIds,
  onSubmit,
  busy,
  initialValues,
}: {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  seedTrackIds: string[];
  onSubmit: (body: GenerateRequestBody) => void;
  busy: boolean;
  initialValues?: Partial<GenerateRequestBody>;
}) {
```

Immediately after all the `useLocalStorage`/`useState` declarations (just before the autocomplete `useEffect`s), add an effect that applies `initialValues` once per change:

```tsx
  useEffect(() => {
    if (!initialValues) return;
    if (initialValues.cohesion_mode) setCohesion(initialValues.cohesion_mode);
    if (initialValues.genre_mode || initialValues.sonic_mode || initialValues.pace_mode) {
      setAxes({
        genre_mode: initialValues.genre_mode ?? axes.genre_mode,
        sonic_mode: initialValues.sonic_mode ?? axes.sonic_mode,
        pace_mode: initialValues.pace_mode ?? axes.pace_mode,
      });
    }
    if (typeof initialValues.tracks === "number") setTracks(initialValues.tracks);
    if (typeof initialValues.artist === "string") setSeed(initialValues.artist);
    else if (typeof initialValues.genre === "string") setSeed(initialValues.genre);
    if (initialValues.artist_spacing) setArtistSpacing(initialValues.artist_spacing);
    if (initialValues.artist_presence) setArtistPresence(initialValues.artist_presence);
    if (initialValues.artist_variety) setArtistVariety(initialValues.artist_variety);
    if (typeof initialValues.include_collaborations === "boolean") setIncludeCollabs(initialValues.include_collaborations);
    if (typeof initialValues.recency_enabled === "boolean") setRecencyEnabled(initialValues.recency_enabled);
    if (typeof initialValues.recency_days === "number") setRecencyDays(initialValues.recency_days);
    if (typeof initialValues.recency_plays_threshold === "number") setRecencyPlays(initialValues.recency_plays_threshold);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialValues]);
```

- [ ] **Step 2: Rewrite JobsPanel**

Replace the entire contents of `web/src/components/JobsPanel.tsx`:

```tsx
import type { GenerateRequestBody, JobOut } from "../lib/types";

const dot: Record<string, string> = {
  success: "text-accent", running: "text-warn", failed: "text-danger",
  cancelled: "text-muted", pending: "text-muted",
};

function clock(ts?: number | null): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function JobsPanel({
  jobs,
  onSelect,
  onCancel,
  onRerun,
}: {
  jobs: JobOut[];
  onSelect: (j: JobOut) => void;
  onCancel: (j: JobOut) => void;
  onRerun: (params: GenerateRequestBody) => void;
}) {
  return (
    <div className="h-full overflow-auto" data-testid="jobs-panel">
      <div className="px-3 py-2 text-[10px] uppercase tracking-wide text-faint border-b border-border">Jobs</div>
      {jobs.map((j) => {
        const meanT = (j.playlist?.metrics?.mean_transition);
        const tracks = j.playlist?.track_count;
        return (
          <div key={j.job_id} className="px-3 py-2 border-b border-[#181b21]">
            <div className="flex items-center justify-between gap-2">
              <div className="text-[11px] text-text truncate">{j.playlist?.name ?? j.stage ?? "Playlist"}</div>
              <span className={`text-[9px] ${dot[j.status] ?? "text-muted"}`}>{j.status}</span>
            </div>
            <div className="text-[9px] text-faint mt-0.5">
              {tracks != null ? `${tracks} tracks` : "—"}
              {typeof meanT === "number" ? ` · T̄ ${meanT.toFixed(2)}` : ""}
              {clock(j.created_at) ? ` · ${clock(j.created_at)}` : ""}
            </div>
            <div className="flex gap-1.5 mt-1.5">
              {j.status === "running" && (
                <button data-testid="job-cancel" onClick={() => onCancel(j)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-[#3a1a1a] text-danger">✕ cancel</button>
              )}
              {j.status !== "running" && j.request_params && (
                <button data-testid="job-rerun" onClick={() => onRerun(j.request_params as GenerateRequestBody)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-[#1d3a35] text-accent">↺ re-run</button>
              )}
              {j.playlist && (
                <button onClick={() => onSelect(j)}
                  className="text-[9px] px-1.5 py-0.5 rounded border border-border text-muted">restore</button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 3: Wire handlers in App**

In `web/src/App.tsx`:

(a) Add state near the other `useState` declarations:

```tsx
  const [rerunValues, setRerunValues] = useState<GenerateRequestBody | null>(null);
```

(b) Add handlers (near the other `useCallback`s):

```tsx
  const handleCancel = useCallback(async (j: JobOut) => {
    try { await api.cancelJob(j.job_id); refreshJobs(); }
    catch (e) { setError(String(e)); }
  }, [refreshJobs]);

  const handleRerun = useCallback((params: GenerateRequestBody) => {
    setMode((params.mode as Mode) ?? "artist");
    setRerunValues(params);
  }, [setMode]);
```

(c) Pass the new props to `GenerateControls`:

```tsx
            <GenerateControls
              mode={mode}
              onModeChange={setMode}
              seedTrackIds={seedTracks.map((t) => t.track_id)}
              onSubmit={submit}
              busy={busy}
              initialValues={rerunValues ?? undefined}
            />
```

(d) Update the `JobsPanel` usage (in the `jobs={...}` Shell prop):

```tsx
        jobs={<JobsPanel jobs={jobs} onSelect={(j) => setPlaylist(j.playlist ?? null)} onCancel={handleCancel} onRerun={handleRerun} />}
```

- [ ] **Step 4: Verify build**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/JobsPanel.tsx web/src/components/GenerateControls.tsx web/src/App.tsx
git commit -m "feat(web): jobs panel cancel/re-run/quality-score + re-run prefill"
```

---

## SUBSYSTEM 4 — Decommission PySide6

### Task 14: Deprecate Qt GUI; make browser the default surface

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/playlist_gui/app.py`
- Modify: `CLAUDE.md`, `README.md`, `docs/GOLDEN_COMMANDS.md`

- [ ] **Step 1: Update pyproject metadata + extras**

In `pyproject.toml`:

Change the `description`:

```toml
description = "Local music library playlist generator with Beat3Tower sonic embeddings, DJ Bridge multi-seed pier playlists, and a browser GUI."
```

Mark `[gui]` deprecated and move `pytest-qt` into it:

```toml
# DEPRECATED: legacy PySide6 desktop GUI. The browser GUI (`pip install -e .[web]`,
# `python tools/serve_web.py`) is the maintained surface. Kept for fallback only.
gui = [
    "PySide6>=6.5.0",
    "pytest-qt>=4.4",
]
```

Remove `pytest-qt>=4.4` from the `dev` extra so `dev` becomes:

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0",
    "ruff>=0.15",
    "mypy>=1.10",
    "pre-commit>=4.0",
]
```

- [ ] **Step 2: Add a deprecation notice to the Qt entrypoint**

In `src/playlist_gui/app.py`, find the `main()` entry (or the top of the module's `__main__` path) and add a stderr notice. Locate the existing PySide6 import guard near the top; immediately after a successful import / at the start of `main()`, add:

```python
    import sys as _sys
    print(
        "[DEPRECATED] The PySide6 desktop GUI is no longer actively maintained. "
        "Use the browser GUI instead:\n    python tools/serve_web.py\n",
        file=_sys.stderr,
    )
```

(Place it so it runs once at startup, before the Qt application loop. If `main()` is not the entry, add it at the top of whatever function `python -m playlist_gui.app` invokes.)

- [ ] **Step 3: Update CLAUDE.md**

In `CLAUDE.md`, in the **Environment** section, replace the GUI line:

```markdown
- **GUI:** `python tools/serve_web.py` (browser GUI, default port 8770) — the maintained front-end. The legacy PySide6 app (`python -m playlist_gui.app`, install `pip install -e .[gui]`) is deprecated.
```

- [ ] **Step 4: Update README.md**

In `README.md`, update the install and launch instructions to lead with the browser GUI. Replace the GUI install/launch lines with:

```markdown
## Install

    pip install -e .[web]          # browser GUI (recommended)
    pip install -e .[web,dev]      # contributors (adds pytest, ruff, mypy)

## Launch

    python tools/serve_web.py      # opens the browser GUI on http://127.0.0.1:8770

The legacy PySide6 desktop GUI is deprecated but still available:

    pip install -e .[gui]
    python -m playlist_gui.app
```

(If the README's existing headings differ, edit the corresponding install/launch lines in place rather than duplicating sections.)

- [ ] **Step 5: Update GOLDEN_COMMANDS.md if needed**

Search `docs/GOLDEN_COMMANDS.md` for `playlist_gui.app`. If present, add a note next to it: `# DEPRECATED — use: python tools/serve_web.py`. If absent, no change.

Run: `grep -n "playlist_gui.app" docs/GOLDEN_COMMANDS.md` (skip edit if no matches).

- [ ] **Step 6: Verify clean install + non-gui tests pass without Qt**

Run:
```bash
pip install -e .[web,dev]
pytest -m "not gui" -q
```
Expected: install completes without pulling PySide6; the non-gui suite passes.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/playlist_gui/app.py CLAUDE.md README.md docs/GOLDEN_COMMANDS.md
git commit -m "chore: deprecate PySide6 GUI; browser GUI is the default surface"
```

---

## FINAL — End-to-end verification

### Task 15: Playwright e2e for Phase 3 UI

**Files:**
- Create: `web/tests/phase3.spec.ts`

- [ ] **Step 1: Write the e2e tests**

Create `web/tests/phase3.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

async function generate(page: Page) {
  await page.goto("/");
  await page.getByTestId("seed-input").fill("Acetone");
  await page.getByRole("button", { name: /Generate/ }).click();
  await expect(page.getByTestId("track-table")).toBeVisible({ timeout: 15000 });
}

test("diagnostics empty state then content after generate", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-diagnostics").click();
  await expect(page.getByTestId("diagnostics-empty")).toBeVisible();
  await generate(page);
  await page.getByTestId("tab-diagnostics").click();
  await expect(page.getByTestId("diagnostics-content")).toBeVisible();
  await expect(page.getByTestId("transition-bars")).toBeVisible();
});

test("blacklist tab lists entries and can add an artist", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("tab-blacklist").click();
  await expect(page.getByTestId("blacklist-panel")).toBeVisible();
  // Fake worker returns Nick Drake as a blacklisted artist.
  await expect(page.getByText("Nick Drake")).toBeVisible();
  await page.getByTestId("blacklist-search").fill("Coldplay");
  await page.getByTestId("blacklist-add").click();
  // Refetch happens; panel stays visible (no crash).
  await expect(page.getByTestId("blacklist-panel")).toBeVisible();
});

test("completed job shows re-run button", async ({ page }) => {
  await generate(page);
  await expect(page.getByTestId("job-rerun").first()).toBeVisible({ timeout: 10000 });
});
```

- [ ] **Step 2: Build the frontend so Playwright serves the latest bundle**

Run: `npm --prefix web run build`
Expected: build succeeds.

- [ ] **Step 3: Run the e2e suite**

Run: `npm --prefix web run test:e2e`
Expected: all Phase 3 specs pass (plus the existing Phase 2 specs).

- [ ] **Step 4: Commit**

```bash
git add web/tests/phase3.spec.ts
git commit -m "test(web): Playwright e2e for diagnostics, blacklist, jobs"
```

---

### Task 16: Full backend suite + manual smoke

- [ ] **Step 1: Run the full backend web suite**

Run: `pytest tests/integration/test_web_api.py tests/integration/test_web_api_phase2.py tests/integration/test_web_api_phase3.py tests/integration/test_metadata_blacklist_fetch.py tests/unit/test_web_worker_bridge.py -v`
Expected: all pass.

- [ ] **Step 2: Manual smoke (real worker)**

Run `python tools/serve_web.py`, open `http://127.0.0.1:8770`, then verify:
1. Generate a playlist → **Diagnostics** tab shows summary stats, weakest-edge box, and transition bars (colors vary).
2. **Blacklist** tab lists current entries grouped by Artists/Albums/Tracks; search an artist + **Add** → it appears under Artists; click × → it disappears.
3. Right-click a track → Blacklist track → the **Blacklist** tab (after switching to it) shows it under Tracks.
4. Start a generation → the running job shows **✕ cancel**; click it → job transitions to cancelled.
5. A completed job shows `N tracks · T̄ 0.xx · HH:MM` and a **↺ re-run** button → clicking re-run prefills the form with that job's artist/settings.

- [ ] **Step 3: Final commit (if any manual fixups were needed)**

```bash
git add -A
git commit -m "chore: Phase 3 manual smoke fixups"
```

(Skip if nothing changed.)

---

## Self-Review Notes

- **Spec coverage:** §2 Diagnostics → Tasks 1-4; §3 Blacklist → Tasks 5-9; §4 Jobs → Tasks 10-13; §5 Decommission → Task 14; §8 Testing → Tasks 15-16 (plus per-task backend tests). All covered.
- **Type consistency:** `BlacklistEntryOut` (backend) ↔ `BlacklistEntry` (frontend) carry the same fields (`scope`, `display_name`, `track_id`, `artist`, `album`). `transition_score` is `Optional[float]`/`number | null` throughout. `JobOut.created_at`/`request_params` match between Python and TS. `blacklist_fetch_scopes` worker command name matches the route call.
- **Removal mapping:** artist → `blacklist_scope_set(scope=artist, enabled=False)`; album → `blacklist_scope_set(scope=album, artist, enabled=False)`; track → `blacklist_set(value=False)`. All map to existing worker commands and the existing `POST /api/blacklist` route — no new removal endpoints needed.
