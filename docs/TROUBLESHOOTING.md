# Troubleshooting

Problem → fix reference for common failures. For *why* the system is shaped this way, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for a code-level walkthrough of a generation,
[`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`](TECHNICAL_PLAYLIST_GENERATION_FLOW.md); for the full
config key reference, [`CONFIG.md`](CONFIG.md); for playlist-quality knob tuning,
[`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md).

> **Start here.** Run `python tools/doctor.py --verbose` before digging further — it checks
> Python version, core dependencies, module imports, `config.yaml`, the database + `tracks`
> table, and artifact `.npz` files in one pass.

---

## Environment / install

### "ModuleNotFoundError: No module named '…'"

**Cause:** The package isn't installed into the active environment.

**Fix:**
```bash
# From the repo root (Python 3.11+ required — pyproject.toml pins it):
pip install -e .[web]        # users — FastAPI/uvicorn/httpx for the Browser GUI
pip install -e .[web,dev]    # contributors — adds pytest, ruff, mypy, pre-commit
```
Scripts add the repo root to `sys.path` themselves when run directly, so once the editable
install is in place both `python main_app.py …` and `python -m pytest` resolve `src/` and
`scripts/`.

> **Genre enrichment needs a second extra.** `adjudicate`/`apply` (and the legacy `enrich`
> stage) call Claude via the Agent SDK, which lives in the `ai` extra, not `web`:
> `pip install -e .[ai]` (installs `claude-agent-sdk` + `openai`). Symptom if missing: the
> `adjudicate` analyze stage fails loudly on import rather than silently skipping — that's
> intentional (a configured stage that can't act is an error, not a silent no-op).

### "config.yaml not found"

**Cause:** `config.yaml` is gitignored on purpose (it can carry API tokens) — a fresh clone
never has one.

**Fix:**
```bash
cp config.example.yaml config.yaml
# then edit library paths, discogs.token, lastfm.api_key, plex block, etc.
```

### `tools/doctor.py` disagrees with what I expect

If `doctor.py` reports something that looks wrong (e.g. a Python-version floor that doesn't
match `pyproject.toml`'s `requires-python`), trust `pyproject.toml` — `doctor.py` is a
convenience check, occasionally lags the real requirement, and is not the source of truth.

---

## Database

### "no such table: tracks"

**Cause:** The database file doesn't exist yet or was created empty.

**Fix:** Run a scan — `LibraryScanner` initializes the schema on first run:
```bash
python scripts/analyze_library.py --stages scan
# or, standalone:
python scripts/scan_library.py
```

### "database is locked"

**Cause:** Another process holds an open connection — most often the Browser GUI's worker
process (`serve_web.py` spawns one long-lived worker at startup) or a concurrent
`analyze_library.py` run.

**Fix:**
- Close other scripts/GUI instances accessing `data/metadata.db`.
- Wait for a running scan/analyze/generation to finish.
- If a worker is wedged, stop `serve_web.py` (which kills its worker child) before retrying.

> **Never work around a lock by copying or symlinking the DB into another checkout.**
> SQLite's WAL mode does not tolerate two independent WAL journals aliasing the same file —
> this has caused real corruption before. Run data-writing stages from one checkout only.

### Slow queries

**Cause:** Missing indexes, usually after a manual schema edit outside the normal pipeline.

**Fix:** A full re-scan recreates indexes:
```bash
python scripts/analyze_library.py --stages scan --force
```

---

## Genre issues

See the `genre-data-authority` skill for the full read/write model. Quick pointers:

### Genres look sparse, stale, or wrong for an artist/album

**Cause:** The **authority** for playlist-facing genres is `release_effective_genres` in
`metadata.db`, written only by the `publish` analyze stage and read only through
`src/genre/authority.py`. Raw `track_genres`/`album_genres`/`artist_genres` tables are inputs,
not the final answer — editing them directly does not change what generation sees until
`publish` runs again.

**Fix:** Re-run the genre-producing stages, ending in `publish`:
```bash
python scripts/analyze_library.py --stages genres,discogs,lastfm,adjudicate,apply,publish
```
For a one-off correction, use the GUI's "Edit genres for album" (writes through the authority
path) rather than hand-editing a table.

### "No tracks need genre updates" / genres seem frozen

**Cause:** The pending-work check thinks everything is already covered (fingerprint/resumability
gate). Verify with:
```bash
python scripts/update_genres_v3_normalized.py --stats
```
Force a re-run if you know source data changed: `--force` on the relevant analyze stage.

### API rate limiting (MusicBrainz / Discogs / Last.fm)

**Cause:** Too many requests in a short window.

**Fix:** Built-in rate limiting should absorb this. If a stage errors out instead of
throttling, wait and re-run — the stage is fingerprint-gated, so already-fetched
artists/albums are skipped on retry.

### Missing genres for many artists

**Notes:**
- Obscure artists may genuinely have no MusicBrainz/Discogs/Last.fm data.
- File tags (`source='file'`) are used as a fallback and are **never dropped** even when
  enrichment disagrees — a specific user file tag missing from adjudication output is treated
  as a bug (it escalates to human review), not silently discarded.
- `discogs` and `lastfm` stages need `DISCOGS_TOKEN`/`LASTFM_API_KEY` (env var or
  `config.yaml`) — they raise loudly if missing rather than skipping quietly.

---

## Artifact issues

The generation artifact is `data/artifacts/beat3tower_32k/data_matrices_step1.npz` — the
pre-computed sonic + genre + energy matrices every generation loads. Keep any troubleshooting
here **generic**: which sonic embedding variant is active is a separate, evolving topic (see
`artifacts.sonic_variant_override` in `ARCHITECTURE.md`'s "Sonic feature space" section) —
don't chase variant-specific fixes here.

### "Artifact missing required keys: […]" / "Artifact file not found"

**Cause:** Artifacts were never built, the path in `config.yaml` doesn't match, or a partial
build was interrupted.

**Fix:**
```bash
# Verify the configured path
grep artifact_path config.yaml

# Rebuild via the analyze pipeline's artifacts stage (recommended — also auto-folds
# any sidecar embeddings back into the artifact)
python scripts/analyze_library.py --stages artifacts,verify

# Or the standalone builder:
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz
```

### "… first dimension … does not match expected …" (shape mismatch)

**Cause:** The artifact was built against a different track set than the current database
(tracks added/removed since the last build).

**Fix:** Rebuild with `--force` so every matrix is regenerated against the current DB:
```bash
python scripts/analyze_library.py --stages artifacts,verify --force
```

### "artifacts.sonic_variant_override='…' is configured but artifact … has no '…' key"

**Cause:** `config.yaml` points at a sonic embedding variant that was never folded into the
artifact (e.g. the override was set before the corresponding fold step ran). This is
deliberate — **a configured knob that can't act raises at load rather than silently falling
back** to a different variant.

**Fix:** Either fold the missing variant into the artifact first (re-run the `artifacts` stage
so its auto-fold step runs), or remove/comment out `artifacts.sonic_variant_override` to use
whatever the artifact already declares.

### `verify` stage fails after a rebuild

**Cause:** The `verify` stage checks manifest fingerprint, row-count parity, and that
`X_sonic_variant` in the artifact matches the configured variant — by design, it errors loudly
rather than shipping a silently-wrong artifact.

**Fix:** Read the specific verify failure message; it names the mismatched field. Usually a
`--force` full rebuild resolves it.

---

## Playlist generation issues

### "Could not create playlist for artist" / generation returns nothing

**Causes:**
- Artist not in your library (name mismatch — check exact spelling/casing).
- Artist (including recognized collaborations) has fewer than 4 tracks — the pipeline requires
  a minimum of 4 to build piers and refuses below that rather than producing a degenerate
  playlist.
- Candidate pool too narrow for the requested mode (rare — the system relaxes progressively
  before giving up; see "never fails on the soft axes" below).

**Diagnose:**
```bash
# Confirm the artist exists and how many tracks
sqlite3 data/metadata.db "SELECT COUNT(*) FROM tracks WHERE artist LIKE '%Artist Name%'"

# Re-run with --verbose to see DS transition metrics and constraint enforcement
python main_app.py --artist "Artist Name" --tracks 30 --verbose
```
> **Read the log, don't trust a summary metric.** A relaxed mode "not helping" can mean a true
> null, a starved pool where the beam never got a shot, or a knob silently not applying — only
> the per-segment pool/gate-tally lines in the log distinguish them (see the `playlist-testing`
> skill's "Diagnosing a generation outcome").

### Playlists are too similar / too narrow

**Cause:** `cohesion_mode` (or `genre_mode`/`sonic_mode`) is set tighter than the seed's actual
neighborhood supports.

**Fix:**
```bash
python main_app.py --artist "Artist" --cohesion-mode dynamic
# or, for more discovery:
python main_app.py --artist "Artist" --cohesion-mode discover --genre-mode discover
```
Sonic, genre, and pace each have their own independent axis (`sonic_mode`, `genre_mode`,
`pace_mode`) — loosening `cohesion_mode` alone doesn't touch pool gating on the other three.
See `ARCHITECTURE.md`'s "four mode axes" table.

### Playlists have jarring transitions

**Cause:** The beam admitted a weak edge — usually because the per-mode transition floor was
low enough (or was relaxed down through infeasibility tiers) to accept it.

**Fix:**
```bash
# Tighten the beam
python main_app.py --artist "Artist" --cohesion-mode narrow

# Engage pace gating so BPM/onset jumps are penalized too
python main_app.py --artist "Artist" --pace-mode narrow
```
Or raise the per-mode transition floor directly (keyed by `cohesion_mode`, shipped defaults
shown):
```yaml
playlists:
  ds_pipeline:
    constraints:
      transition_floor_dynamic: 0.35   # raise for a stricter floor
      transition_floor_narrow: 0.45
```
This is a target, not an absolute wall — if no path meets it, the beam relaxes it down through
several tiers (and widens the beam / lowers the bridge floor) rather than failing the whole
playlist. To see exactly which edge is weakest and why, enable
`pier_bridge.emit_selected_edge_audit: true` and compare the reported `T` per edge.

### Same artist appears too often

**Fix in `config.yaml`:**
```yaml
playlists:
  max_tracks_per_artist: 3   # shipped default; lower to cap repeats harder
  ds_pipeline:
    constraints:
      min_gap: 6             # shipped default; raise for more spacing between repeats
```
`min_gap` is enforced live during beam search (including across segment boundaries), not as a
post-hoc filter — so it can't be defeated by a downstream repair pass.

---

## Export issues

### M3U paths don't resolve in my player

**Cause:** Path format mismatch between how the library was scanned and what the target player
expects.

**Fix:**
- Confirm `playlists.m3u_export_path` in `config.yaml` points where you expect.
- Ensure `playlists.export_m3u: true` is set.
- Windows players sometimes need backslash paths — check the specific player's M3U path
  convention if tracks show as missing despite existing on disk.

### Plex export fails

**Causes:** invalid token, music section name mismatch, or a path-mapping gap between how Plex
sees the library and how this app sees it.

**Fix:**
```yaml
plex:
  enabled: true
  base_url: http://localhost:32400
  token: "your_token_here"       # Plex web UI → an item → Get Info → View XML → token in URL
  music_section: "Music"         # must match your Plex library section name exactly
  path_map: []                   # add entries if Plex's file paths differ from this app's
```
Verify the section name in Plex's own library list if the export reports "section not found."

---

## Browser GUI issues

The GUI is React (`web/`) → FastAPI (`src/playlist_web/app.py`) → NDJSON worker
(`src/playlist_gui/worker.py`), launched by `python tools/serve_web.py` (default port 8770).
**Most "the GUI doesn't show / do X" reports are stale process or build state, not logic
bugs.** Walk these three traps first; the full catalog is in the `web-gui` skill.

### A front-end change doesn't appear

**Cause:** `web/dist` wasn't rebuilt, so the server is still serving the old bundle.
`serve_web.py` rebuilds `web/dist` on every launch **unless** started with `--no-build` — if
you started it that way, or the build failed silently, you're on stale JS.

**Fix:**
```bash
npm --prefix web run build     # rebuild the bundle
# then hard-reload the browser (bypass cache)
```
To confirm which build is actually being served: `grep -rl <a token from your edit>
web/dist/assets/*.js`.

### A worker/backend change doesn't take effect

**Cause:** `serve_web.py` spawns one long-lived worker subprocess at startup with no hot-reload
— editing `worker.py`, the policy layer, or an analyze stage module doesn't affect the already-
running worker.

**Fix:** Stop and restart `python tools/serve_web.py`. Also watch for `@lru_cache`d artifact
loads inside the worker process — those persist across requests but not across a restart.

### A button runs but nothing comes back (or generation modes seem inert)

**Cause:** Usually an end-to-end wiring gap — a new API endpoint or worker command not wired
through every layer (React → FastAPI route → worker command handler → streamed NDJSON result),
or a result the bridge silently dropped. A related trap: **only the web path runs UI slider
modes through the policy layer** (`derive_runtime_config`) — a test harness or script that
bypasses policy and sets mode strings directly will see every mode as inert, which looks like a
bug but is a wiring gap in the test, not the product.

**Fix:** Trace the request through all four layers. The `web-gui` skill documents the
stale-dist, worker-restart, end-to-end-wiring, and silently-dropped-result traps in detail.

---

## Logs

Default log destinations (override any of them with `--log-file PATH`):

| Entry point | Default log file |
|---|---|
| `main_app.py` | `playlist_generator.log` (repo root) |
| `scripts/analyze_library.py` | `logs/analyze/<timestamp>_<run_id>.log` (one rotated file per run; tune via `logging.analyze_logs.*`) |
| `scripts/scan_library.py` | `logs/scan_library.log` |
| `scripts/update_sonic.py` | `logs/sonic_analysis.log` |
| `scripts/update_genres_v3_normalized.py` | `logs/genre_update_v3.log` |

Analyze Library writes one rotated DEBUG log per run under `logs/analyze/` (old runs are pruned after `logging.analyze_logs.retention_days`, default 30). Follow the newest:

```bash
tail -f "$(ls -t logs/analyze/*.log | head -1)"
```
`--verbose` (CLI) raises the level to `DEBUG` and adds DS transition/constraint detail —
reach for it before concluding a generation outcome is a bug (see the note under "Could not
create playlist" above).

---

## Getting help

1. Run the doctor: `python tools/doctor.py --verbose`
2. Re-run the failing step with `--verbose` / `--log-level DEBUG` and read the log, not just
   the final summary line.
3. Diff your `config.yaml` against `config.example.yaml` — a missing or misspelled key is a
   more common cause than a real bug.
