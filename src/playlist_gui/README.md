# playlist_gui — worker process + shared policy layer

The PySide6 desktop GUI that used to live in this package was removed on
2026-06-10 (evidence: `docs/DEAD_CODE_AUDIT_2026-06-10.md`; restore point:
git tag `pre-cleanup-20260610`). The browser GUI (`src/playlist_web` +
`web/`, served by `python tools/serve_web.py`) is the only front-end.

## What remains here

| Module | Role |
|---|---|
| `worker.py` | The generation worker. Spawned by the web bridge as `python -m src.playlist_gui.worker`; speaks NDJSON over stdin/stdout. |
| `policy.py` | `derive_runtime_config(ui, seed_artist_keys=...)` — translates UI state into config overrides. Policy-owned keys always win (see `POLICY_OWNED_KEYS`). Used by the web app. |
| `ui_state.py` | `UIStateModel` — the UI-state dataclass `policy.py` consumes. Used by the web app. |
| `utils/redaction.py` | Secret redaction for worker log output. |

## Worker NDJSON protocol

Commands and events are newline-delimited JSON, correlated by `request_id`.
Single active job at a time; `cancel` is cooperative (checked at stage
boundaries).

**Commands (client → worker):**

```json
{"cmd": "ping", "request_id": "<uuid>", "protocol_version": 1}
{"cmd": "generate_playlist", "request_id": "<uuid>", "protocol_version": 1, "base_config_path": "config.yaml", "overrides": {...}, "args": {"mode": "seeds", "tracks": 50, "seed_tracks": [...], "seed_track_ids": [...]}}
{"cmd": "scan_library", "request_id": "<uuid>", "protocol_version": 1, "base_config_path": "config.yaml", "overrides": {}}
{"cmd": "cancel", "request_id": "<uuid-to-cancel>"}
```

**Events (worker → client):**

```json
{"type": "log", "request_id": "<uuid>", "level": "INFO", "msg": "..."}
{"type": "progress", "request_id": "<uuid>", "stage": "generate", "current": 60, "total": 100, "detail": "..."}
{"type": "result", "request_id": "<uuid>", "result_type": "playlist", "playlist": {"name": "...", "tracks": [...]}}
{"type": "error", "request_id": "<uuid>", "message": "...", "traceback": "..."}
{"type": "done", "request_id": "<uuid>", "cmd": "generate_playlist", "ok": true, "detail": "Generated 30 tracks"}
{"type": "done", "request_id": "<uuid>", "cmd": "generate_playlist", "ok": false, "cancelled": true, "detail": "Cancelled by user"}
```

The full command list lives in `TRACKED_COMMAND_HANDLERS` /
`UNTRACKED_COMMAND_HANDLERS` at the bottom of `worker.py`.
