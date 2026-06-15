---
name: web-gui
description: How the browser GUI is wired (React + FastAPI + NDJSON worker) and the traps of changing it. Use whenever editing web/src, src/playlist_web, or worker command handlers; adding a Tools/worker-backed feature; wiring a new API endpoint or worker command; or debugging "the GUI doesn't show/do X". Encodes the stale-dist, worker-restart, end-to-end-wiring, and silently-dropped-result traps that each cost a debugging cycle.
---

# Working on the browser GUI

The web GUI is the **only** front-end (the PySide6 desktop GUI was deleted 2026-06-10). Read the Core Rules and the Trap Catalog before editing the front-end or any worker command — every trap below cost a real debugging cycle.

## Architecture (where things live, and the event path)

```
React (web/src, Vite+TS+Tailwind)
  └─ fetch /api/...            ─────────────►  FastAPI  (src/playlist_web/app.py, create_app factory)
  └─ WebSocket /ws  ◄── broadcast ───  WsHub          │ submit(cmd) → BridgeBusy(409) if busy
                                       JobRegistry     │
                                       (jobs.py)       ▼
                                   apply_event ◄── WorkerBridge (worker_bridge.py, NDJSON over stdin/stdout)
                                                          │
                                                          ▼
                                       worker subprocess (src/playlist_gui/worker.py)
                                         handlers in TRACKED_COMMAND_HANDLERS
```

- **Serve it:** `python tools/serve_web.py` (default port 8770). The worker is spawned as `python -m src.playlist_gui.worker`.
- **Event flow:** a handler calls `emit_log/emit_progress/emit_result/emit_done` → worker stdout (NDJSON) → bridge `_read_loop` → `registry.apply_event(event)` + `hub.broadcast(event)` → WS → React `useWorkerEvents` handler. `job_id`/`request_id` are auto-injected by `emit_event` from worker state.
- **The policy layer** (`src/playlist_gui/policy.py::derive_runtime_config`) translates UI sliders → config overrides. `worker.py`/`policy.py`/`ui_state.py` survived the desktop-GUI deletion *because the web app shares them*. For generation-fidelity specifics see the **playlist-testing** skill.

## Core Rules

1. **The served GUI runs `web/dist`, not your source.** After editing anything under `web/src/`, run `npm --prefix web run build` (or use the Vite dev server) or the change is invisible. Symptom we hit: a new stage checkbox didn't appear because `dist` predated the edit. `grep -rl <token> web/dist/assets/*.js` to confirm a build is current.

2. **Worker code changes need a `serve_web.py` restart.** The running worker has whatever code was loaded when it was spawned. Editing `worker.py` (or anything it imports) does nothing for an already-running server — restart it. This is why a freshly-fixed GUI cancel "still doesn't work" until restart.

3. **A worker-backed feature must be wired through every layer, or it silently no-ops.** See the checklist below. The recurring failure mode of this codebase is config/wiring that *looks* connected but isn't (CLAUDE.md "a configured knob that can't act is a startup error").

## Adding a worker-backed feature end-to-end (the checklist)

Miss any link and it breaks silently. For a new command `foo`:

1. **Worker handler** — `def handle_foo(cmd_data)` in `worker.py`; register in `TRACKED_COMMAND_HANDLERS` (or `UNTRACKED_COMMAND_HANDLERS` for fast, non-tracked ops like cancel). Emit `result` then `done`. A missing `done` hangs the bridge.
2. **FastAPI endpoint** — in `app.py`: `registry.create(...)` → `await bridge.submit({"cmd": "foo", "job_id": job_id, ...})` → return `{"job_id": ...}`; wrap submit in `try/except BridgeBusy → HTTPException(409)`.
3. **Result capture** — `JobRegistry.apply_event` only routes `result_type=="playlist"` into `job.playlist`; **every other result type needs the generic `tool_result` path** (added 2026-06-11). New non-playlist result? Confirm it lands in `tool_result` and is exposed in `_to_out`/`JobOut`.
4. **Schemas** — request + `JobOut` fields in `src/playlist_web/schemas.py` (Pydantic). Additive/optional fields are backward-compatible.
5. **Front-end types + client** — `web/src/lib/types.ts` interface + `web/src/lib/api.ts` method.
6. **Component + wiring** — the React component, mounted in `App.tsx`. Read WS events via `useWorkerEvents` keyed by `job_id`.
7. **Fake worker + tests** — add a `foo` branch to `tests/fixtures/fake_worker.py`; integration test with `TestClient(create_app(worker_cmd=FAKE))`; optional Playwright (`web/tests/*.spec.ts`).
8. **Rebuild `web/dist`** (Core Rule 1) and restart the server (Core Rule 2).

## Cancellation model (read before touching it)

- The worker `main()` runs **tracked commands on a worker thread** so the reader stays free to dispatch `cancel` inline. Do **not** reintroduce a single blocking `for line in sys.stdin: process_command(line)` loop — that was the bug: cancel sat unread until the command finished. `emit_event` is guarded by `_stdout_lock` because two threads emit.
- **Cancellation is honored only at safe, committed checkpoints** (data-safety rule): stage boundaries in `analyze_library.run_pipeline` (`_check_cancelled()`), and *between* MERT tracks (`run_extraction(cancellation_check=…)` flushes completed shards in a `finally` first). **Never add a cancel check mid-DB-write.** A long stage (enrich, mert) won't stop instantly — by design.
- Single-flight: the bridge runs one tracked command at a time (`BridgeBusy`→409). Cancel is the exception (untracked, inline).

## Trap Catalog (each cost a debugging cycle)

| Trap | Symptom | Fix |
|------|---------|-----|
| Stale `web/dist` | UI change (new control, label) doesn't appear in the served GUI | `npm --prefix web run build`; the dev server hot-reloads but `serve_web.py` serves `dist`. |
| Worker edit, no restart | bug "still happens" after a worker.py fix; cancel still dead | Restart `serve_web.py` — the running worker has old code. |
| Stale stage/enum list in `request_models.py` | a stage checkbox does nothing; `LibraryPipelineRequest` runs ALL stages | `ANALYZE_LIBRARY_STAGE_ORDER` + `AnalyzeLibraryStage` must mirror `analyze_library.STAGE_ORDER_DEFAULT`; `_clean_stages` drops unknown stages then falls back to all. |
| Non-playlist result dropped | `job.tool_result` is null; analyze/enrich summary never shows | `apply_event` had only the `result_type=="playlist"` branch; use the generic `elif etype=="result"` → `tool_result`. |
| Policy disables a feature silently | dj_bridging off on every web run | web app must resolve seed artist keys so `derive_runtime_config` can enable it; missing keys → conservative off. |
| Long command "hangs" | quiet terminal for minutes during enrich/mert | not hung — enrich = chunked Claude calls; mert ≈ 50 h CPU. GUI has no `--limit`, so a GUI mert run does the full sweep. Use the CLI with `--limit N` for smoke tests. |
| Missing `done` event | bridge stays busy; next request → 409 forever | every handler must `emit_done(...)` on all paths (success, error, cancel, pause). |
| Inline (untracked) handler does a WRITE | review panel 500s during a scan; cancel starves too | untracked handlers run on the READER thread; a write/DDL there blocks behind a tracked job's write txn for `busy_timeout`, wedging the reader. Read-only + WAL only. Sidecar is WAL (`SidecarStore.connect`); `get_review_queue_page(readonly=True)` uses `connect_readonly` + no `initialize()`. (2026-06-12) |
| Uncaught `asyncio.TimeoutError`/worker-down in an endpoint | bare HTTP 500 + scary traceback, no signal | `bridge.command` raises `WorkerTimeout`/`WorkerUnavailable`; global handlers in `create_app` map them to 504/503. New `bridge.command`/`submit` caller? It's covered automatically — don't let a raw timeout escape. |
| **Slow WS client wedges the WHOLE worker** | unrelated commands (review queue) 60s-timeout; jobs stuck "running"; `/api/blacklist` 409 spam | The bridge's single `_read_loop` does `await on_event` → `await hub.broadcast` per worker event. If `broadcast` blocks on a backpressured browser's `send_json`, the loop stops draining the worker's stdout → worker blocks on `print()` → everything wedges. **WS delivery must NEVER block the read loop.** `WsHub` uses a per-client bounded queue + background sender (`offer` drops oldest on overflow, never awaits the socket). Don't reintroduce `await ws.send_json` into `broadcast`. (2026-06-12) |
| **Worker result line > 64KB kills the bridge** | command 60s-timeout; jobs stuck "running"; 409 spam — same symptom set as above | The worker emits each result as ONE NDJSON line; the bridge reads with `asyncio.StreamReader.readline()` (default limit **65,536 B**). The review `limit=50` page is ~192KB → `LimitOverrunError` → read loop dies → all futures time out. `create_subprocess_exec(..., limit=2**24)` raises it; `_read_loop` fails pending futures on exit so a future overrun is instant, not a 60s hang. **When you add an endpoint/handler that returns a big payload, mind the line size.** (2026-06-12, the actual multi-day Genre Review root cause) |
| **Harness passes, real GUI fails** | every test/repro green, user still hits the bug | The harness used toy params (`limit=3`) while the GUI sends `limit=50`. **Reproduce through the REAL HTTP server (`serve_web.py` + curl) with the EXACT params the client sends** — not a bridge/asyncio harness with convenient values. This single discipline would have caught the 64KB bug on day one. |

## Testing the web layer

**Windows trap (cost a full debugging cycle, 2026-06-12):** `TestClient` runs the app on an anyio portal loop that does NOT faithfully read the *real* worker subprocess's stdout — a real-worker endpoint call **times out under TestClient even when the code is correct** (the FAKE worker stub works fine; the discrepancy is the trap). Two reliable ways to verify a real-worker endpoint on Windows:
- Pure `asyncio.run` driving `WorkerBridge` directly (Proactor loop — subprocess pipes work).
- A live `python tools/serve_web.py --port <spare> --no-browser` + `curl` (this is what uvicorn actually does in prod; the only faithful end-to-end check). Read-only endpoints are safe against the real DB.
Do NOT conclude "the endpoint is broken" from a TestClient timeout alone.


- **API:** `TestClient(create_app(worker_cmd=FAKE))` where `FAKE=[sys.executable, "tests/fixtures/fake_worker.py"]`. The fake worker is an NDJSON stub — **add a branch for any new command before the `else` fallback**. Poll `/api/jobs/{id}` until `status in (success, failed, cancelled)`.
- **Real worker:** `tests/integration/test_web_api.py` drives the *actual* worker subprocess through the bridge — the best guard that a `main()`/protocol change didn't break dispatch.
- **Playwright:** `web/tests/*.spec.ts`; `web/playwright.config.ts` builds + serves on 8771 and points the worker at the fake via `PG_WEB_WORKER_CMD`.
- **Worker concurrency:** `tests/unit/test_worker_cancel_concurrency.py` drives `main()` with a monkeypatched `sys.stdin` generator + a fake slow handler; coordinate threads with `threading.Event`, not sleeps, to stay deterministic.

## Maintenance protocol

Keep this skill current — it's the index of how the GUI is wired, not a one-time doc. When you:
- add/change a worker command or endpoint → update the checklist if the wiring shape changed.
- hit a new GUI trap (a layer that silently dropped data, a build/serve surprise) → add a Trap Catalog row.
- change the cancellation or threading model → update the Cancellation section (it is load-bearing and easy to regress).
