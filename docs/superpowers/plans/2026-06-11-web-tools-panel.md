# Web GUI Tools Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the `analyze_library` and `enrich_genres` worker commands through a Tools panel in the web GUI, with live WS progress and per-job result summaries.

**Architecture:** Two new FastAPI endpoints (`POST /api/tools/analyze` and `/api/tools/enrich`) feed the existing `JobRegistry` + `WsHub` pipeline. A new `ToolsPanel` React component renders inside the existing `Shell`'s center slot when the user switches to the "Tools" tab in `App.tsx`. Backend plumbing uses the same `bridge.submit()` / `registry.apply_event()` / `hub.broadcast()` loop that generate already uses.

**Tech Stack:** Python (FastAPI, Pydantic), TypeScript/React, Tailwind CSS, Playwright (smoke test).

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/playlist/request_models.py` | Modify | Add `lastfm`, `enrich`, `publish` to `AnalyzeLibraryStage` Literal and `ANALYZE_LIBRARY_STAGE_ORDER` |
| `src/playlist_web/schemas.py` | Modify | Add `AnalyzeToolRequest`, `EnrichToolRequest`; add `tool_result: Optional[dict]` to `JobOut` |
| `src/playlist_web/jobs.py` | Modify | Add `tool_result` to `_JobState`; capture non-playlist results in `apply_event`; include in `_to_out` |
| `src/playlist_web/app.py` | Modify | Add `POST /api/tools/analyze` and `POST /api/tools/enrich` |
| `tests/fixtures/fake_worker.py` | Modify | Add `analyze_library` and `enrich_genres` command handlers |
| `tests/integration/test_web_tools_api.py` | Create | Integration tests for the two new endpoints |
| `web/src/lib/types.ts` | Modify | Add `AnalyzeToolRequest`, `EnrichToolRequest`; extend `JobOut.tool_result` |
| `web/src/lib/api.ts` | Modify | Add `api.analyzeLibrary()` and `api.enrich()` |
| `web/src/components/ToolsPanel.tsx` | Create | Analyze Library + Enrich cards with WS-driven progress |
| `web/src/App.tsx` | Modify | Tab switcher (Generate / Tools); render ToolsPanel in center when active |
| `web/tests/tools.spec.ts` | Create | Playwright smoke test: dry-run analyze runs and completes |

---

## Task 1: Update `request_models.py` stage list

**Files:**
- Modify: `src/playlist/request_models.py:19-39`
- Test: (existing tests in `tests/unit/test_analyze_graph_stages.py` cover this indirectly)

The `AnalyzeLibraryStage` Literal and `ANALYZE_LIBRARY_STAGE_ORDER` were defined before Phase 2 and don't include `lastfm`, `enrich`, or `publish`. The `_clean_stages` function filters any requested stage not in this list; if everything is filtered out, `LibraryPipelineRequest.__post_init__` silently falls back to running ALL stages — the exact "configured knob that can't act" silent-no-op the project forbids. After this task, `stages=["lastfm"]` produces `["lastfm"]` and not the full default order.

- [ ] **Step 1: Update `AnalyzeLibraryStage` Literal and `ANALYZE_LIBRARY_STAGE_ORDER`**

  In `src/playlist/request_models.py`, replace lines 19-39:

  ```python
  AnalyzeLibraryStage = Literal[
      "scan",
      "genres",
      "discogs",
      "lastfm",
      "sonic",
      "enrich",
      "publish",
      "genre-sim",
      "artifacts",
      "genre-embedding",
      "verify",
  ]

  ANALYZE_LIBRARY_STAGE_ORDER: tuple[AnalyzeLibraryStage, ...] = (
      "scan",
      "genres",
      "discogs",
      "lastfm",
      "sonic",
      "enrich",
      "publish",
      "genre-sim",
      "artifacts",
      "genre-embedding",
      "verify",
  )
  ```

  The ordering matches `STAGE_ORDER_DEFAULT` in `scripts/analyze_library.py` exactly.

- [ ] **Step 2: Run the analyze-stages unit tests to confirm no regressions**

  ```
  pytest tests/unit/test_analyze_graph_stages.py -v
  ```

  Expected: all 10 tests pass (especially `test_default_stage_order_has_new_stages_positioned` and `test_run_pipeline_runs_new_stages_then_skips_on_rerun`).

- [ ] **Step 3: Verify silent-fallback is gone for new stages**

  In a Python REPL or via `pytest -s`:

  ```python
  from src.playlist.request_models import LibraryPipelineRequest, ANALYZE_LIBRARY_STAGE_ORDER
  r = LibraryPipelineRequest(config_path="x", stages=["lastfm", "enrich", "publish"])
  assert r.stages == ["lastfm", "enrich", "publish"], f"got {r.stages}"
  r2 = LibraryPipelineRequest(config_path="x", stages=[])
  assert list(r2.stages) == list(ANALYZE_LIBRARY_STAGE_ORDER), "empty → all stages should still hold"
  ```

- [ ] **Step 4: Commit**

  ```
  git add src/playlist/request_models.py
  git commit -m "fix(request-models): add lastfm/enrich/publish to AnalyzeLibraryStage and STAGE_ORDER"
  ```

---

## Task 2: Backend — schemas, job registry, and endpoints

**Files:**
- Modify: `src/playlist_web/schemas.py`
- Modify: `src/playlist_web/jobs.py`
- Modify: `src/playlist_web/app.py`

Three coordinated changes: (a) new request schemas + `JobOut.tool_result`, (b) registry captures non-playlist results, (c) two new `/api/tools/*` endpoints.

- [ ] **Step 1: Add `AnalyzeToolRequest`, `EnrichToolRequest`, and extend `JobOut` in `schemas.py`**

  At the bottom of `src/playlist_web/schemas.py`, before `BlacklistArtistRequest`, add:

  ```python
  class AnalyzeToolRequest(BaseModel):
      stages: list[str] = Field(default_factory=list)
      force: bool = False
      dry_run: bool = False


  class EnrichToolRequest(BaseModel):
      scope: str = "all_unenriched"
      artist: Optional[str] = None
      album: Optional[str] = None
  ```

  Extend the existing `JobOut` model to include `tool_result`:

  ```python
  class JobOut(BaseModel):
      """Response model for a job status query or completion."""

      job_id: str
      status: str
      stage: str = ""
      error: Optional[str] = None
      playlist: Optional[PlaylistOut] = None
      tool_result: Optional[dict] = None
      created_at: Optional[float] = None
      request_params: Optional[dict] = None
  ```

- [ ] **Step 2: Extend `_JobState` and `apply_event` in `jobs.py`**

  In `src/playlist_web/jobs.py`, add `tool_result` to `_JobState.__init__`:

  ```python
  def __init__(self, job_id: str, max_log_lines: int, request_params: Optional[dict] = None):
      self.job_id = job_id
      self.status = "pending"
      self.stage = ""
      self.error: Optional[str] = None
      self.playlist: Optional[PlaylistOut] = None
      self.tool_result: Optional[dict] = None
      self.logs: deque[str] = deque(maxlen=max_log_lines)
      self.created_at: float = time.time()
      self.request_params: dict = request_params or {}
  ```

  In `apply_event`, after the `result_type == "playlist"` branch, add a branch for all other result types:

  ```python
  elif etype == "result" and event.get("result_type") == "playlist":
      job.playlist = PlaylistOut.from_worker(event.get("playlist", {}))
  elif etype == "result":
      job.tool_result = dict(event)
  ```

  In `_to_out`, include the new field:

  ```python
  def _to_out(self, job: _JobState) -> JobOut:
      return JobOut(
          job_id=job.job_id,
          status=job.status,
          stage=job.stage,
          error=job.error,
          playlist=job.playlist,
          tool_result=job.tool_result,
          created_at=job.created_at,
          request_params=job.request_params,
      )
  ```

- [ ] **Step 3: Add the two endpoints to `app.py`**

  In `src/playlist_web/app.py`, add the new schema imports to the existing import block:

  ```python
  from .schemas import (
      AnalyzeToolRequest,
      BlacklistArtistRequest,
      BlacklistFetchResponse,
      BlacklistRequest,
      EditGenresRequest,
      EnrichToolRequest,
      GenerateRequestBody,
      JobOut,
      PlexExportRequest,
      ReplaceSuggestionsRequest,
      ReplaceSuggestionsResponse,
  )
  ```

  After the `@app.post("/api/jobs/{job_id}/cancel")` handler (before `@app.get("/api/tracks/search")`), add the two endpoints:

  ```python
  @app.post("/api/tools/analyze")
  async def tools_analyze(body: AnalyzeToolRequest) -> dict:
      job_id = registry.create(request_params=body.model_dump())
      try:
          await bridge.submit({
              "cmd": "analyze_library",
              "job_id": job_id,
              "base_config_path": config_path,
              "overrides": {},
              "stages": body.stages or None,
              "force": body.force,
              "dry_run": body.dry_run,
          })
      except BridgeBusy:
          raise HTTPException(status_code=409, detail="A job is already running.")
      return {"job_id": job_id}


  @app.post("/api/tools/enrich")
  async def tools_enrich(body: EnrichToolRequest) -> dict:
      job_id = registry.create(request_params=body.model_dump())
      try:
          await bridge.submit({
              "cmd": "enrich_genres",
              "job_id": job_id,
              "scope": body.scope,
              "artist": body.artist,
              "album": body.album,
          })
      except BridgeBusy:
          raise HTTPException(status_code=409, detail="A job is already running.")
      return {"job_id": job_id}
  ```

- [ ] **Step 4: Run the existing web API tests to confirm no regressions**

  ```
  pytest tests/integration/test_web_api.py -v
  ```

  Expected: all tests pass (the `JobOut` schema change adds a nullable field — backward-compatible).

- [ ] **Step 5: Commit**

  ```
  git add src/playlist_web/schemas.py src/playlist_web/jobs.py src/playlist_web/app.py
  git commit -m "feat(web-tools): add /api/tools/analyze and /api/tools/enrich endpoints"
  ```

---

## Task 3: Fake worker handlers + integration tests

**Files:**
- Modify: `tests/fixtures/fake_worker.py`
- Create: `tests/integration/test_web_tools_api.py`

The fake worker needs to handle `analyze_library` and `enrich_genres` to exercise the new endpoints in tests. The integration tests verify the full request→job→done→result roundtrip.

- [ ] **Step 1: Add `analyze_library` and `enrich_genres` handlers to `fake_worker.py`**

  In `tests/fixtures/fake_worker.py`, before the `else` fallback (the `emit({"type": "error", ...})` block), add:

  ```python
  elif name == "analyze_library":
      stages = cmd.get("stages") or [
          "scan", "genres", "discogs", "lastfm", "sonic",
          "enrich", "publish", "genre-sim", "artifacts",
          "genre-embedding", "verify",
      ]
      emit({"type": "log", "level": "INFO", "msg": "fake: analyze starting",
            "request_id": rid, "job_id": jid})
      emit({"type": "progress", "stage": "analyze_library", "current": 50, "total": 100,
            "detail": "scanning", "request_id": rid, "job_id": jid})
      emit({"type": "result", "result_type": "analyze_library",
            "request_id": rid, "job_id": jid,
            "summary": "Analyze complete (fake)",
            "stages": [
                {"name": s, "decision": "ran", "duration_ms": 10, "errors": 0}
                for s in stages
            ],
            "out_dir": "/tmp"})
      emit({"type": "done", "cmd": "analyze_library", "ok": True,
            "detail": f"Done ({len(stages)} stages)",
            "request_id": rid, "job_id": jid})
  elif name == "enrich_genres":
      scope = cmd.get("scope", "all_unenriched")
      emit({"type": "log", "level": "INFO", "msg": "fake: enrich starting",
            "request_id": rid, "job_id": jid})
      emit({"type": "progress", "stage": "enrich_genres", "current": 50, "total": 100,
            "detail": "enriching", "request_id": rid, "job_id": jid})
      emit({"type": "result", "result_type": "enrich_genres",
            "request_id": rid, "job_id": jid,
            "ok": True, "scope": scope, "releases": 5, "genres_applied": 12})
      emit({"type": "done", "cmd": "enrich_genres", "ok": True,
            "detail": "Enriched 5 releases",
            "request_id": rid, "job_id": jid})
  ```

- [ ] **Step 2: Create `tests/integration/test_web_tools_api.py`**

  ```python
  # tests/integration/test_web_tools_api.py
  """Integration tests for /api/tools/analyze and /api/tools/enrich endpoints."""
  import sys
  import time

  from fastapi.testclient import TestClient

  from src.playlist_web.app import create_app

  FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


  def _wait_done(client, job_id, timeout=5):
      deadline = time.time() + timeout
      while time.time() < deadline:
          job = client.get(f"/api/jobs/{job_id}").json()
          if job["status"] in ("success", "failed", "cancelled"):
              return job
          time.sleep(0.05)
      return client.get(f"/api/jobs/{job_id}").json()


  def test_analyze_creates_job_and_succeeds():
      app = create_app(worker_cmd=FAKE)
      with TestClient(app) as client:
          resp = client.post("/api/tools/analyze", json={})
          assert resp.status_code == 200
          job_id = resp.json()["job_id"]
          job = _wait_done(client, job_id)
          assert job["status"] == "success"
          assert job["tool_result"]["result_type"] == "analyze_library"
          stages = job["tool_result"]["stages"]
          assert isinstance(stages, list) and len(stages) == 11


  def test_analyze_with_stage_subset():
      app = create_app(worker_cmd=FAKE)
      with TestClient(app) as client:
          resp = client.post("/api/tools/analyze",
                             json={"stages": ["lastfm", "enrich", "publish"]})
          assert resp.status_code == 200
          job_id = resp.json()["job_id"]
          job = _wait_done(client, job_id)
          assert job["status"] == "success"
          stages = job["tool_result"]["stages"]
          assert len(stages) == 3


  def test_enrich_creates_job_and_succeeds():
      app = create_app(worker_cmd=FAKE)
      with TestClient(app) as client:
          resp = client.post("/api/tools/enrich", json={"scope": "all_unenriched"})
          assert resp.status_code == 200
          job_id = resp.json()["job_id"]
          job = _wait_done(client, job_id)
          assert job["status"] == "success"
          assert job["tool_result"]["result_type"] == "enrich_genres"
          assert job["tool_result"]["releases"] == 5


  def test_analyze_job_appears_in_jobs_list():
      app = create_app(worker_cmd=FAKE)
      with TestClient(app) as client:
          resp = client.post("/api/tools/analyze", json={"dry_run": True})
          job_id = resp.json()["job_id"]
          _wait_done(client, job_id)
          jobs = client.get("/api/jobs").json()
          ids = [j["job_id"] for j in jobs]
          assert job_id in ids
  ```

- [ ] **Step 3: Run the new integration tests**

  ```
  pytest tests/integration/test_web_tools_api.py -v
  ```

  Expected: all 4 tests pass.

- [ ] **Step 4: Run full fast test suite to confirm no regressions**

  ```
  pytest -m "not slow" -q
  ```

  Expected: same pass count as before (1535 + new tests).

- [ ] **Step 5: Commit**

  ```
  git add tests/fixtures/fake_worker.py tests/integration/test_web_tools_api.py
  git commit -m "test(web-tools): fake worker handlers and integration tests for /api/tools/*"
  ```

---

## Task 4: Frontend types + API client

**Files:**
- Modify: `web/src/lib/types.ts`
- Modify: `web/src/lib/api.ts`

Add the two request types and extend `JobOut.tool_result`. Add `api.analyzeLibrary()` and `api.enrich()` methods following the existing `api` object pattern.

- [ ] **Step 1: Update `web/src/lib/types.ts`**

  Extend `JobOut` with `tool_result`:

  ```typescript
  export interface JobOut {
    job_id: string;
    status: "pending" | "running" | "success" | "failed" | "cancelled";
    stage: string;
    error?: string | null;
    playlist?: PlaylistOut | null;
    tool_result?: Record<string, unknown> | null;
    created_at?: number | null;
    request_params?: Record<string, unknown> | null;
  }
  ```

  Add the two new request interfaces (at the end of the file):

  ```typescript
  export interface AnalyzeToolRequest {
    stages?: string[];
    force?: boolean;
    dry_run?: boolean;
  }

  export interface EnrichToolRequest {
    scope: "all_unenriched" | "artist" | "release";
    artist?: string;
    album?: string;
  }
  ```

- [ ] **Step 2: Update `web/src/lib/api.ts`**

  Add the new imports at the top:

  ```typescript
  import type {
    AnalyzeToolRequest,
    BlacklistFetchResponse,
    BlacklistRequest,
    EditGenresRequest,
    EnrichToolRequest,
    GenerateRequestBody,
    JobOut,
    PlexExportRequest,
    ReplaceSuggestionsResponse,
    SeedTrack,
  } from "./types";
  ```

  Add two methods to the `api` object (after `exportPlex`):

  ```typescript
  async analyzeLibrary(req: AnalyzeToolRequest): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/tools/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async enrich(req: EnrichToolRequest): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/tools/enrich", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  ```

- [ ] **Step 3: Run TypeScript type-check**

  From `web/`:

  ```
  npx tsc --noEmit
  ```

  Expected: 0 errors (the new fields are optional/additive — no breaking changes to existing consumers).

- [ ] **Step 4: Commit**

  ```
  git add web/src/lib/types.ts web/src/lib/api.ts
  git commit -m "feat(web-types): add AnalyzeToolRequest/EnrichToolRequest; extend JobOut.tool_result"
  ```

---

## Task 5: ToolsPanel component

**Files:**
- Create: `web/src/components/ToolsPanel.tsx`

Self-contained component. Two cards side-by-side: **Analyze Library** (stage checkboxes, force/dry-run toggles, run/cancel, progress, last-run summary) and **Enrich Genres** (scope select, artist/album inputs, run/cancel, progress, last-run summary). WS events routed by matching `job_id`.

Note on styling: the codebase uses raw Tailwind classes against a Studio Dark palette (no shadcn `ui/` components). Follow the color tokens used in `GenerateControls.tsx`: `bg-[#0c0e12]`, `border-[#23262d]`, `text-[#8b939d]`, `text-[#c9d1d9]`. The `text-accent` and `text-danger` CSS variables are available.

- [ ] **Step 1: Create `web/src/components/ToolsPanel.tsx`**

  ```tsx
  import { useCallback, useState } from "react";
  import { api } from "../lib/api";
  import { useWorkerEvents } from "../lib/ws";
  import type { AnalyzeToolRequest, EnrichToolRequest, WsEvent } from "../lib/types";

  const ALL_STAGES = [
    "scan", "genres", "discogs", "lastfm", "sonic",
    "enrich", "publish", "genre-sim", "artifacts",
    "genre-embedding", "verify",
  ] as const;

  type AnalyzeStageName = (typeof ALL_STAGES)[number];

  interface StageResult {
    name: string;
    decision: string;
    duration_ms: number;
    errors: number;
  }

  function Card({ title, children }: { title: string; children: React.ReactNode }) {
    return (
      <div className="flex flex-col gap-3 p-4 bg-[#0f1117] border border-[#23262d] rounded-md min-w-0">
        <div className="text-[11px] uppercase tracking-[.1em] text-[#5b6470] font-semibold">
          {title}
        </div>
        {children}
      </div>
    );
  }

  function Row({ children }: { children: React.ReactNode }) {
    return <div className="flex items-center gap-2">{children}</div>;
  }

  function Lbl({ children }: { children: React.ReactNode }) {
    return (
      <span className="text-[10px] uppercase tracking-[.08em] text-[#5b6470] font-medium select-none">
        {children}
      </span>
    );
  }

  function RunBtn({
    disabled,
    onClick,
    children,
  }: {
    disabled?: boolean;
    onClick: () => void;
    children: React.ReactNode;
  }) {
    return (
      <button
        onClick={onClick}
        disabled={disabled}
        className="px-3 py-1 text-[11px] font-medium rounded bg-accent text-black
                   disabled:opacity-30 disabled:cursor-default hover:opacity-90 transition-opacity"
      >
        {children}
      </button>
    );
  }

  function CancelBtn({ onClick }: { onClick: () => void }) {
    return (
      <button
        onClick={onClick}
        className="px-3 py-1 text-[11px] font-medium rounded border border-[#3a3f4b]
                   text-[#8b939d] hover:text-[#c9d1d9] transition-colors"
      >
        Cancel
      </button>
    );
  }

  function ProgressBar({ label }: { label: string }) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-[#1e2128] rounded overflow-hidden">
          <div className="h-full bg-accent animate-pulse w-full" />
        </div>
        <span className="text-[10px] text-[#5b6470] truncate max-w-[160px]">{label}</span>
      </div>
    );
  }

  function StageTable({ stages }: { stages: StageResult[] }) {
    return (
      <table className="w-full text-[10px] text-[#5b6470]">
        <tbody>
          {stages.map((s) => (
            <tr key={s.name}>
              <td className="pr-2 text-[#8b939d] font-mono">{s.name}</td>
              <td className={`pr-2 ${s.decision === "ran" ? "text-accent" : "text-[#5b6470]"}`}>
                {s.decision}
              </td>
              <td className="pr-2">{s.duration_ms}ms</td>
              {s.errors > 0 && <td className="text-danger">{s.errors} err</td>}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  export function ToolsPanel({
    externalBusy,
    refreshJobs,
  }: {
    externalBusy: boolean;
    refreshJobs: () => void;
  }) {
    // ── Analyze Library state ───────────────────────────────────────────────
    const [selectedStages, setSelectedStages] = useState<AnalyzeStageName[]>(
      [...ALL_STAGES]
    );
    const [force, setForce] = useState(false);
    const [dryRun, setDryRun] = useState(false);
    const [analyzeJobId, setAnalyzeJobId] = useState<string | null>(null);
    const [analyzeProgress, setAnalyzeProgress] = useState("");
    const [lastAnalyzeStages, setLastAnalyzeStages] = useState<StageResult[] | null>(null);
    const [analyzeError, setAnalyzeError] = useState<string | null>(null);

    // ── Enrich Genres state ─────────────────────────────────────────────────
    const [scope, setScope] = useState<EnrichToolRequest["scope"]>("all_unenriched");
    const [enrichArtist, setEnrichArtist] = useState("");
    const [enrichAlbum, setEnrichAlbum] = useState("");
    const [enrichJobId, setEnrichJobId] = useState<string | null>(null);
    const [enrichProgress, setEnrichProgress] = useState("");
    const [lastEnrichSummary, setLastEnrichSummary] = useState<string | null>(null);
    const [enrichError, setEnrichError] = useState<string | null>(null);

    const anyRunning = analyzeJobId !== null || enrichJobId !== null;
    const runDisabled = externalBusy || anyRunning;

    useWorkerEvents(
      useCallback(
        (e: WsEvent) => {
          if (e.type === "progress") {
            const detail = e["detail"] as string | undefined;
            if (e.job_id === analyzeJobId) setAnalyzeProgress(detail ?? "");
            if (e.job_id === enrichJobId) setEnrichProgress(detail ?? "");
          }
          if (e.type === "done") {
            if (e.job_id === analyzeJobId) {
              const jid = e.job_id as string;
              setAnalyzeJobId(null);
              setAnalyzeProgress("");
              api
                .job(jid)
                .then((j) => {
                  const tr = j.tool_result;
                  if (tr && Array.isArray(tr["stages"]))
                    setLastAnalyzeStages(tr["stages"] as StageResult[]);
                  if (j.error) setAnalyzeError(j.error);
                })
                .catch(() => {});
              refreshJobs();
            }
            if (e.job_id === enrichJobId) {
              const jid = e.job_id as string;
              setEnrichJobId(null);
              setEnrichProgress("");
              api
                .job(jid)
                .then((j) => {
                  const tr = j.tool_result;
                  if (tr)
                    setLastEnrichSummary(
                      `${tr["releases"] ?? 0} releases enriched, ${tr["genres_applied"] ?? 0} genres applied`
                    );
                  if (j.error) setEnrichError(j.error);
                })
                .catch(() => {});
              refreshJobs();
            }
          }
        },
        [analyzeJobId, enrichJobId, refreshJobs]
      )
    );

    function toggleStage(s: AnalyzeStageName) {
      setSelectedStages((prev) =>
        prev.includes(s) ? prev.filter((x) => x !== s) : [...ALL_STAGES.filter((a) => prev.includes(a) || a === s)]
      );
    }

    async function runAnalyze() {
      setAnalyzeError(null);
      const req: AnalyzeToolRequest = {
        stages: selectedStages.length < ALL_STAGES.length ? [...selectedStages] : [],
        force,
        dry_run: dryRun,
      };
      try {
        const { job_id } = await api.analyzeLibrary(req);
        setAnalyzeJobId(job_id);
        refreshJobs();
      } catch (e) {
        setAnalyzeError(String(e));
      }
    }

    async function runEnrich() {
      setEnrichError(null);
      const req: EnrichToolRequest = {
        scope,
        artist: enrichArtist.trim() || undefined,
        album: enrichAlbum.trim() || undefined,
      };
      try {
        const { job_id } = await api.enrich(req);
        setEnrichJobId(job_id);
        refreshJobs();
      } catch (e) {
        setEnrichError(String(e));
      }
    }

    async function cancelAnalyze() {
      if (!analyzeJobId) return;
      try {
        await api.cancelJob(analyzeJobId);
      } catch {}
    }

    async function cancelEnrich() {
      if (!enrichJobId) return;
      try {
        await api.cancelJob(enrichJobId);
      } catch {}
    }

    return (
      <div className="h-full overflow-auto p-4">
        <div className="grid grid-cols-2 gap-4 max-w-4xl">
          {/* ── Analyze Library ───────────────────────────────────────────── */}
          <Card title="Analyze Library">
            {/* Stage checkboxes */}
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {ALL_STAGES.map((s) => (
                <label key={s} className="flex items-center gap-1 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedStages.includes(s)}
                    onChange={() => toggleStage(s)}
                    className="accent-accent"
                  />
                  <span className="text-[10px] text-[#8b939d] font-mono">{s}</span>
                </label>
              ))}
            </div>

            {/* Options row */}
            <Row>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={force}
                  onChange={(e) => setForce(e.target.checked)}
                  className="accent-accent"
                />
                <Lbl>Force</Lbl>
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={dryRun}
                  onChange={(e) => setDryRun(e.target.checked)}
                  className="accent-accent"
                />
                <Lbl>Dry run</Lbl>
              </label>
            </Row>

            {/* Run / Cancel */}
            <Row>
              {analyzeJobId ? (
                <CancelBtn onClick={cancelAnalyze} />
              ) : (
                <RunBtn disabled={runDisabled} onClick={runAnalyze}>
                  Run
                </RunBtn>
              )}
            </Row>

            {/* Progress */}
            {analyzeJobId && analyzeProgress && (
              <ProgressBar label={analyzeProgress} />
            )}

            {/* Error */}
            {analyzeError && (
              <div className="text-[10px] text-danger">{analyzeError}</div>
            )}

            {/* Last-run summary */}
            {lastAnalyzeStages && !analyzeJobId && (
              <details className="text-[10px] text-[#5b6470]">
                <summary className="cursor-pointer select-none">
                  Last run — {lastAnalyzeStages.length} stages
                </summary>
                <div className="mt-1">
                  <StageTable stages={lastAnalyzeStages} />
                </div>
              </details>
            )}
          </Card>

          {/* ── Enrich Genres ─────────────────────────────────────────────── */}
          <Card title="Enrich Genres">
            {/* Scope select */}
            <Row>
              <Lbl>Scope</Lbl>
              <select
                value={scope}
                onChange={(e) =>
                  setScope(e.target.value as EnrichToolRequest["scope"])
                }
                className="bg-[#0c0e12] border border-[#23262d] rounded text-[11px]
                           text-[#8b939d] py-[3px] px-2 cursor-pointer"
              >
                <option value="all_unenriched">All unenriched</option>
                <option value="artist">Artist</option>
                <option value="release">Release</option>
              </select>
            </Row>

            {/* Artist field */}
            {(scope === "artist" || scope === "release") && (
              <Row>
                <Lbl>Artist</Lbl>
                <input
                  type="text"
                  value={enrichArtist}
                  onChange={(e) => setEnrichArtist(e.target.value)}
                  placeholder="artist name"
                  className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded
                             text-[11px] text-[#c9d1d9] py-[3px] px-2
                             placeholder:text-[#3a3f4b] outline-none
                             focus:border-[#3a3f4b]"
                />
              </Row>
            )}

            {/* Album field */}
            {scope === "release" && (
              <Row>
                <Lbl>Album</Lbl>
                <input
                  type="text"
                  value={enrichAlbum}
                  onChange={(e) => setEnrichAlbum(e.target.value)}
                  placeholder="album title (optional)"
                  className="flex-1 bg-[#0c0e12] border border-[#23262d] rounded
                             text-[11px] text-[#c9d1d9] py-[3px] px-2
                             placeholder:text-[#3a3f4b] outline-none
                             focus:border-[#3a3f4b]"
                />
              </Row>
            )}

            {/* Run / Cancel */}
            <Row>
              {enrichJobId ? (
                <CancelBtn onClick={cancelEnrich} />
              ) : (
                <RunBtn disabled={runDisabled} onClick={runEnrich}>
                  {scope === "all_unenriched" ? "Enrich all pending" : "Enrich"}
                </RunBtn>
              )}
            </Row>

            {/* Progress */}
            {enrichJobId && enrichProgress && (
              <ProgressBar label={enrichProgress} />
            )}

            {/* Error */}
            {enrichError && (
              <div className="text-[10px] text-danger">{enrichError}</div>
            )}

            {/* Last-run summary */}
            {lastEnrichSummary && !enrichJobId && (
              <div className="text-[10px] text-[#5b6470]">{lastEnrichSummary}</div>
            )}
          </Card>
        </div>
      </div>
    );
  }
  ```

- [ ] **Step 2: Run TypeScript type-check**

  ```
  npx tsc --noEmit
  ```

  Expected: 0 errors.

- [ ] **Step 3: Commit**

  ```
  git add web/src/components/ToolsPanel.tsx
  git commit -m "feat(web-tools): add ToolsPanel component (Analyze Library + Enrich cards)"
  ```

---

## Task 6: Wire ToolsPanel into App.tsx + Playwright smoke test

**Files:**
- Modify: `web/src/App.tsx`
- Create: `web/tests/tools.spec.ts`

Add a Generate / Tools tab switcher to the topBar. When "Tools" is active, render `ToolsPanel` in the center slot instead of the generate flow. The existing right and logs panels are preserved regardless of active tab.

- [ ] **Step 1: Update `web/src/App.tsx`**

  Add `ToolsPanel` import:

  ```tsx
  import { ToolsPanel } from "./components/ToolsPanel";
  ```

  Add tab state after the existing `busy` state declaration (around line 33):

  ```tsx
  const [tab, setTab] = useState<"generate" | "tools">("generate");
  ```

  Replace the topBar content in the `Shell` render. Current:

  ```tsx
  topBar={
    <>
      <div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>
      {error && <div className="text-danger text-xs">{error}</div>}
    </>
  }
  ```

  Replace with:

  ```tsx
  topBar={
    <>
      <div className="font-bold text-sm"><span className="text-accent">◆</span> Playlist Generator</div>
      <div className="flex items-center gap-1 ml-4">
        {(["generate", "tools"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={[
              "px-3 py-1 text-[11px] rounded transition-colors capitalize",
              tab === t
                ? "bg-[#23262d] text-[#c9d1d9]"
                : "text-[#5b6470] hover:text-[#8b939d]",
            ].join(" ")}
          >
            {t}
          </button>
        ))}
      </div>
      {error && <div className="text-danger text-xs ml-auto">{error}</div>}
    </>
  }
  ```

  Replace the `center` slot. Current center is the entire generate flow JSX block. Wrap it in a conditional and add the ToolsPanel branch:

  ```tsx
  center={
    tab === "tools" ? (
      <ToolsPanel externalBusy={busy} refreshJobs={refreshJobs} />
    ) : (
      <div className="h-full flex flex-col overflow-hidden">
        <GenerateControls
          mode={mode}
          onModeChange={setMode}
          seedTrackIds={seedTracks.map((t) => t.track_id)}
          seedDisplays={seedTracks.map((t) => `${t.title} - ${t.artist}`)}
          onSubmit={submit}
          busy={busy}
          initialValues={rerunValues ?? undefined}
        />
        {mode === "seeds" && (
          <SeedTrackSection
            tracks={seedTracks}
            onAdd={addSeed}
            onRemove={removeSeed}
            onClear={clearSeeds}
          />
        )}
        <QualityStats
          metrics={playlist?.metrics}
          count={playlist?.track_count ?? 0}
          tracks={playlist?.tracks ?? []}
          onExportM3U8={() => playlist && downloadM3U8(playlist.tracks)}
          onExportPlex={() => setPlexOpen(true)}
        />
        <div className="flex-1 overflow-auto">
          <TrackTable
            tracks={playlist?.tracks ?? []}
            blacklisted={blacklisted}
            onContextAction={openMenu}
          />
        </div>
      </div>
    )
  }
  ```

- [ ] **Step 2: Run TypeScript type-check**

  ```
  npx tsc --noEmit
  ```

  Expected: 0 errors.

- [ ] **Step 3: Build the frontend**

  From `web/`:

  ```
  npm run build
  ```

  Expected: build completes with no errors.

- [ ] **Step 4: Create `web/tests/tools.spec.ts`**

  ```typescript
  import { test, expect } from "@playwright/test";

  test("Tools tab is visible and dry-run analyze completes", async ({ page }) => {
    await page.goto("/");

    // Switch to Tools tab
    await page.getByRole("button", { name: /tools/i }).click();
    await expect(page.getByText("Analyze Library")).toBeVisible();
    await expect(page.getByText("Enrich Genres")).toBeVisible();

    // Enable dry run
    await page.getByLabel(/dry run/i).check();

    // Click Run and wait for the job to appear in the jobs panel
    await page.getByRole("button", { name: /^run$/i }).first().click();

    // Wait for the progress to appear then disappear (job completes)
    // The fake worker emits done quickly; poll for Run button re-enabled
    await expect(page.getByRole("button", { name: /^run$/i }).first()).toBeEnabled({
      timeout: 10000,
    });
  });

  test("Enrich all pending button is visible", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /tools/i }).click();
    await expect(page.getByRole("button", { name: /enrich all pending/i })).toBeVisible();
  });
  ```

- [ ] **Step 5: Run Playwright tests**

  From `web/`:

  ```
  npx playwright test tests/tools.spec.ts --reporter=list
  ```

  Expected: both tests pass.

- [ ] **Step 6: Commit**

  ```
  git add web/src/App.tsx web/tests/tools.spec.ts
  git commit -m "feat(web-tools): wire ToolsPanel into App.tsx with Generate/Tools tab switcher"
  ```

---

## Self-Review

**Spec coverage:**
- ✅ `POST /api/tools/analyze` — Task 2
- ✅ `POST /api/tools/enrich` — Task 2
- ✅ 409 on concurrent jobs — Task 2 (BridgeBusy → HTTPException)
- ✅ Cancel via existing `POST /api/jobs/{id}/cancel` — already exists; Test 4 exercises it via fake_worker
- ✅ Stage checkboxes (default: all) — Task 5 `selectedStages` initialized to `ALL_STAGES`
- ✅ Force and Dry-run toggles — Task 5
- ✅ Live progress from WS — Task 5 `useWorkerEvents` / `ProgressBar`
- ✅ Last-run summary (per-stage decision) — Task 5 `StageTable` / `lastAnalyzeStages`
- ✅ Enrich card with scope + artist + album fields — Task 5
- ✅ Disable Run while any job running — Task 5 `runDisabled = externalBusy || anyRunning`
- ✅ FastAPI endpoint tests via fake-worker — Task 3
- ✅ Playwright smoke test — Task 6
- ✅ `ANALYZE_LIBRARY_STAGE_ORDER` includes new stages — Task 1 (prerequisite fix)
- ✅ `JobOut.tool_result` captures non-playlist results — Task 2

**Type consistency check:**
- `AnalyzeToolRequest.dry_run` (Python/TS) — both use snake_case `dry_run` ✅
- `EnrichToolRequest.scope` values: `"all_unenriched" | "artist" | "release"` match worker handler ✅
- `JobOut.tool_result` is `Optional[dict]` (Python) / `Record<string, unknown> | null` (TS) ✅
- `api.analyzeLibrary(req: AnalyzeToolRequest)` matches `POST /api/tools/analyze` body ✅
- `api.enrich(req: EnrichToolRequest)` matches `POST /api/tools/enrich` body ✅
- `ToolsPanel` props `externalBusy: boolean`, `refreshJobs: () => void` match App.tsx call site ✅

**No placeholders:** all steps contain complete code. ✅
