# Browser GUI — Phase 1: Foundation + Core Generate Loop

**Status:** Design approved (brainstorm), pending spec review
**Date:** 2026-06-04
**Supersedes:** nothing yet — coexists with the PySide6 GUI until parity is reached

---

## 1. Why

The PySide6 desktop GUI isn't responsive enough, is hard to style to a professional bar, and — critically — is not a standard enough front-end for the AI to iterate on effectively. We're replacing the **front-end only**. The playlist engine, the `JobManager`/worker subprocess, `request_models`, and `policy.derive_runtime_config` are *not* part of this rewrite; they are wrapped, not rebuilt.

This document specifies **Phase 1** of a three-phase migration. Phase 1 stands up the foundation (stack, backend adapter, app shell, visual system) and delivers the **core Generate loop** end-to-end. Phases 2–3 are roadmapped in §11 but out of scope here.

The PySide6 GUI keeps working as the fallback throughout all three phases. Nothing is decommissioned until the browser version reaches parity (end of Phase 3).

## 2. Goals (Phase 1)

- A browser app that generates a playlist end-to-end: enter seeds + options → generate → see the ranked track table, quality stats, and a live log stream.
- Prove the architecture: React front-end ⟷ FastAPI ⟷ existing worker subprocess.
- Establish the durable foundation later phases build on: the component stack, the Studio Dark visual system, and the resizable panel shell.

## 3. Non-goals (Phase 1)

- Track-row interactions: context menus, replace track, edit genres, pin/exclude, preview clips, export. (**Phase 2**)
- Heavy tools: Genre Enrichment, Blacklist, Jobs management UI, Diagnostics, Library analysis. (**Phase 3**)
- Decommissioning the PySide6 GUI. (**after Phase 3**)
- Any change to the engine, the job system, or any database schema.
- Authentication / multi-user. The server binds `127.0.0.1`; it is single-user local-first by design (CLAUDE.md Layer 2 §14).

## 4. Tech stack (foundation — durable across all phases)

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | **React + TypeScript** | Best AI-legibility and ecosystem; TS gives compile-time safety and self-documenting component contracts. |
| Build/dev | **Vite** | Instant HMR dev server; zero-config start; fast iteration. |
| Styling | **Tailwind CSS** | Utility-first; pairs with shadcn; design tokens drive theming. |
| Components | **shadcn/ui** (on Radix primitives) | Copy-in source we *own* (lives in `web/src/components/ui/`), fully themeable, non-generic, Radix handles a11y/keyboard for menus & dialogs. |
| Backend adapter | **FastAPI** | Async; native WebSocket; request/response schemas align with TS types. |
| Resizable panels | **react-resizable-panels** | Best-in-class drag-to-resize split panes; persists layout. |
| Data table | **TanStack Table** | Sorting, row selection, virtualization for large track lists. |
| Context menus / dialogs | **Radix UI** (via shadcn) | Accessible, keyboard-navigable. (Wired in Phase 2.) |

Libraries named for later phases (dnd-kit for drag-reorder, etc.) are noted but not installed in Phase 1.

## 5. Architecture

```
React (browser)  ⟷  FastAPI  ⟷  existing JobManager / worker subprocess  ⟷  engine
   HTTP:  submit generate request, fetch result, health
   WS:    stream job progress, log lines, status transitions
```

- FastAPI is **serialization glue**, not logic. It translates HTTP/WS ⟷ the existing job system.
- **Generation runs in the worker subprocess** exactly as today (confirmed decision: it's already wired, gives crash isolation, matches the job model). FastAPI submits to `JobManager` and relays events.
- Request bodies map ~1:1 to `GeneratePlaylistRequest`. Runtime config is built with the existing `policy.derive_runtime_config` / `merge_overrides` — the web layer does not reimplement config derivation.
- The bottom **Logs** panel is fed by the WebSocket log stream (the same events the Qt log panel consumes today).

### Coexistence & launch
- New entry point (e.g. `python -m playlist_web` / `tools/serve_web.py`) starts FastAPI, which owns the worker subprocess, then opens the browser — mirroring the proven `sonic_audition_serve.py` pattern.
- Default port fixed and configurable (proposed `8770`); bind `127.0.0.1`.
- Dev: Vite dev server (HMR) proxies `/api` and `/ws` to FastAPI. Prod: FastAPI serves the built static bundle from `web/dist`.

### Project structure
```
web/                      # Vite React app (new)
  src/
    components/ui/        # shadcn components (owned source)
    components/           # app components (Shell, TrackTable, GenerateControls, LogPanel, ...)
    lib/                  # api client, ws client, types
    theme/               # Studio Dark tokens
  index.html
  package.json
src/playlist_web/         # FastAPI adapter (new Python package)
  app.py                 # FastAPI app + routes
  ws.py                  # WebSocket hub bridging JobManager events
  schemas.py             # pydantic <-> request_models mapping
tools/serve_web.py       # launcher (starts server, opens browser)
```

## 6. Visual system — "Studio Dark"

DAW / pro-tool aesthetic: dark, dense, low-fatigue for long sessions.

- **Surfaces:** app bg `#0f1115`, panels `#16181d`, borders `#23262d`.
- **Text:** primary `#e6e9ec`, muted `#8b939d`, faint `#5b6470`.
- **Accent:** mint `#5eead4` (single accent; final hue is a one-token change).
- **Status:** running/warn `#fbbf24`, danger `#fb7185`.
- **Type:** UI sans (Spline Sans or Inter — finalize at build); **monospace for all numerics** (transition scores, indices, counts) via JetBrains Mono.
- All values live as CSS variables / Tailwind theme tokens so a light mode (and accent swaps) come essentially free later.

## 7. App shell & layout (foundation)

Resizable, collapsible panels mirroring today's structure:

- **Top bar:** app title · the four mode chips (cohesion / genre / sonic / pace) · Generate button.
- **Left:** Jobs list (read-only in Phase 1 — full management is Phase 3).
- **Center:** seed/options strip + the track table.
- **Right:** Advanced Settings + Genre Review as **tabs** in one panel (consolidates today's two right docks).
- **Bottom:** Logs (live WS stream).

Behavior:
- Every region divider is a drag handle (mint on hover); panels collapse via an `×` in their header; double-click a divider resets it.
- Layout sizes persist (react-resizable-panels → localStorage).

### Established patterns for later phases (decided now, built later)
- **Standalone windows → slide-over panels.** Genre Enrichment / Blacklist (Phase 3) slide in from the right, resizable by the left edge, workspace dimmed behind. One at a time.
- **Context menus → right-click + a `⋯` kebab on row hover** (Phase 2). Browser's native menu suppressed on app surfaces. Multi-select acts on the whole selection. Planned row actions: Preview clip · Replace track · Edit genres · "Why this track?" (submenu: transition + genre diagnostics) · Pin to position · Exclude · Blacklist track / artist.
- **Transactional actions → centered modals** (export, replace, edit genres — Phase 2).

## 8. Phase 1 scope, precisely

The **Generate workspace**, end-to-end:

1. **Generate controls:** generation mode (artist / seeds / history), seed input with DB-backed autocomplete, track count, and **all four mode-axis selectors (cohesion / genre / sonic / pace)** — including genre mode (see §10).
2. **Submit** → FastAPI → worker; UI shows progress via WS.
3. **Track table:** ranked tracks with index, title, artist, genre chips, transition score (`T`); sortable; row selection (selection wired now, actions in Phase 2).
4. **Quality stats:** min / mean / p10 / p90 transition, weakest-edge callout, distinct-artist count (CLAUDE.md Layer 4 §21).
5. **Live logs:** WS-streamed log lines in the bottom panel.
6. **Jobs (read-only):** list recent jobs with status; selecting one loads its result into the table.

Generation in Phase 1 is **read-only** against `data/metadata.db` (no writes; the metadata.db safety rule is not at risk here).

## 9. Backend API surface (Phase 1)

HTTP (JSON):
- `POST /api/generate` — body ≈ `GeneratePlaylistRequest`; returns `{ job_id }`.
- `GET  /api/jobs` — recent jobs + status.
- `GET  /api/jobs/{id}` — job detail + result (track list, stats) when complete.
- `GET  /api/autocomplete?q=` — artist/seed autocomplete from the DB.
- `GET  /api/health` — server + worker liveness.

WebSocket:
- `WS /ws` — subscribe to `{progress, log, status}` events keyed by `job_id`.

Schemas are validated with pydantic and exported to the front-end as TS types (shared contract).

## 10. Open questions / deferred decisions

- **Genre-mode exposure — DECIDED: expose it in Phase 1.** Genre mode is CLI-only in the current GUI (roadmap Tier-2.4) and is *currently broken* upstream, but a fix is planned soon; the web GUI exposes all four axes from the start so the control is ready when the engine fix lands. The selector ships in Phase 1; correctness of the underlying mode is tracked separately, not a Phase 1 blocker.
- **UI sans face:** Spline Sans vs Inter — decide at build (both loaded in mockups; trivial swap).
- **Default port** `8770` — confirm no collision.
- **Worker lifecycle on server shutdown** — ensure the subprocess is reaped cleanly when FastAPI stops (mirror audition-serve teardown).

## 11. Roadmap (out of scope here)

- **Phase 2 — Track interactions:** context menus, replace track, edit genres, pin/exclude, preview clips, export (local + Plex). Writes to the enrichment DB begin here → the metadata/enrichment write-safety rules apply.
- **Phase 3 — Heavy tools:** Genre Enrichment slide-over, Blacklist slide-over, full Jobs management, Diagnostics, Library analysis. Then **decommission PySide6**.

## 12. Testing

- **FastAPI adapter:** pytest against the routes and the JobManager bridge (mock/real worker). Markers per repo convention.
- **Front-end:** Playwright via the `webapp-testing` toolkit — exercise the real Generate loop against a running server. (Note: the `qtbot` no-op caveat that plagues the PySide6 tests does **not** apply here; browser tests actually drive the UI.)
- **Contract:** assert pydantic schemas and exported TS types stay in sync.
- Verify the live path by actually generating a playlist in the browser before claiming completion (CLAUDE.md rigor; global "verify before claiming success").
