# Analyze Library Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the MuQ analyze stage a live progress heartbeat (so a slow run is never mistaken for a hung one) and give every Analyze Library run its own rotated log file, mirroring per-playlist logging.

**Architecture:** MuQ is the one per-item analyze stage that never adopted the existing `ProgressLogger` heartbeat contract used by scan/sonic/mbid/discogs/lastfm/enrich; Tasks 1–2 wire it in. Analyze already routes file logging through `configure_logging(log_file=…)` fresh per `run_pipeline` call (a per-run-lifecycle, tagged file handler) — only its *path* is fixed — so Tasks 3–4 compute a per-run path + add age-based rotation instead of adding a parallel handler. Task 5 guarantees the two heavy builder stages also emit a "starting" line.

**Tech Stack:** Python 3.11, stdlib `logging`, pytest, numpy. Design spec: `docs/superpowers/specs/2026-07-06-analyze-library-logging-design.md`.

## Global Constraints

- Python 3.11+. No new third-party dependencies.
- Shared working checkout: stage/commit **explicit paths only** — `git add <paths>` then `git commit -m "…" --only -- <paths>`. Never `git add -A/-u/.`, never a bare `git commit`. Verify with `git diff --cached --name-only` before committing.
- Run tests bounded, never piped through `tail`/`head`: `python -m pytest -q <targets>` with the tool timeout.
- Do not touch `data/metadata.db`, the MERT/MuQ archives, or the progress-bar bridge (`parse_analyze_library_stage_progress`).
- `run_muq_extraction` with `progress=None` must stay byte-identical to today (existing unit tests must not change).

---

### Task 1: MuQ runner emits progress via an optional reporter

**Files:**
- Modify: `src/analyze/muq_runner.py:129-160` (`run_muq_extraction`)
- Test: `tests/unit/test_muq_runner.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_muq_extraction(items, embed_fn, sidecar_path, *, backup_stamp=None, save_every=SAVE_EVERY, progress=None)`. `progress` is any object exposing `.update(n=1, detail=<str|None>)` and `.finish()` (a `ProgressLogger`, or `None` for no-op). Called once per item (success, bad-file, or no-path) and `.finish()` once after the loop.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_muq_runner.py`:

```python
class _FakeProgress:
    def __init__(self):
        self.updates = 0
        self.finished = 0
        self.details = []

    def update(self, n=1, detail=None):
        self.updates += 1
        self.details.append(detail)

    def finish(self, detail=None):
        self.finished += 1


def test_run_extraction_reports_progress_for_every_item(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"

    def stub(path):
        if path == "/bad":
            raise RuntimeError("decode fail")
        return np.ones(4, np.float32)

    fp = _FakeProgress()
    # good (ok), bad (embed raises), none (no_path) -> all three must tick progress
    res = run_muq_extraction(
        [("g", "/good"), ("b", "/bad"), ("n", None)], stub, sc, progress=fp
    )
    assert res["ok"] == 1 and res["failed"] == 2
    assert fp.updates == 3           # one tick per item, including failures
    assert fp.finished == 1          # exactly one finish
    assert fp.details == ["/good", "/bad", None]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_muq_runner.py::test_run_extraction_reports_progress_for_every_item`
Expected: FAIL — `run_muq_extraction() got an unexpected keyword argument 'progress'`.

- [ ] **Step 3: Write minimal implementation**

In `src/analyze/muq_runner.py`, change the signature and loop of `run_muq_extraction`. Replace the body from `def run_muq_extraction(` through the final `return {...}`:

```python
def run_muq_extraction(
    items: Sequence[Tuple[str, Optional[str]]],
    embed_fn: Callable[[str], np.ndarray],
    sidecar_path,
    *,
    backup_stamp: Optional[str] = None,
    save_every: int = SAVE_EVERY,
    progress=None,
) -> Dict[str, object]:
    """Embed each (track_id, path); append into the sidecar (atomic, resumable). Backs up
    the existing sidecar once (when backup_stamp given) before the first write. A bad file's
    failure is recorded in the returned `fails` list, never fatal. Returns {ok, failed, fails}.

    ``progress`` (optional) is any object with ``.update(n, detail)`` / ``.finish()`` — a
    ``ProgressLogger`` from the caller. ``None`` is a no-op (byte-identical to before)."""
    done = _load_existing(sidecar_path)
    if backup_stamp is not None:
        _backup(sidecar_path, backup_stamp)
    ok = 0
    fails: List[Tuple[str, str]] = []
    succeeded: List[str] = []
    for k, (tid, path) in enumerate(items, 1):
        if not path:
            fails.append((tid, "no_path"))
        else:
            try:
                done[tid] = np.asarray(embed_fn(path), dtype=np.float32)
                ok += 1
                succeeded.append(tid)
            except Exception as exc:  # one bad file must not kill the scan
                fails.append((tid, type(exc).__name__))
        if progress is not None:
            progress.update(1, detail=path)
        if k % save_every == 0:
            _atomic_save(sidecar_path, done)
    _atomic_save(sidecar_path, done)
    if progress is not None:
        progress.finish()
    _update_failures(sidecar_path, fails, succeeded)
    return {"ok": ok, "failed": len(fails), "fails": fails}
```

(The no-path branch changed from `continue` to `if/else` so every item ticks progress; sidecar contents are unaffected.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_muq_runner.py`
Expected: PASS — the new test plus all 8 pre-existing tests (the `progress=None` default keeps them unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/analyze/muq_runner.py tests/unit/test_muq_runner.py
git commit -m "feat(analyze): MuQ extraction reports progress via optional reporter" --only -- src/analyze/muq_runner.py tests/unit/test_muq_runner.py
```

---

### Task 2: `stage_muq` builds the reporter, passes it down, times the model load

**Files:**
- Modify: `scripts/analyze_library.py:2289-2296` (`stage_muq`)
- Test: `tests/unit/test_analyze_library_logging.py`

**Interfaces:**
- Consumes: `run_muq_extraction(..., progress=…)` from Task 1; `ProgressLogger` from `src.logging_utils`.
- Produces: no new callable — `stage_muq` now emits `stage_muq: MuQ-MuLan loaded in <s>` and, via the reporter, `muq: N/total (%) | rate tracks/s | ETA …` lines at INFO.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_analyze_library_logging.py`:

```python
def test_stage_muq_logs_model_load_and_heartbeat(tmp_path, monkeypatch, caplog):
    import numpy as np
    import src.analyze.muq_runner as muq_runner
    import src.analyze.track_paths as track_paths

    config_path = _write_config(tmp_path, tmp_path / "metadata.db")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        track_paths, "load_paths",
        lambda db_path: {"t1": "/a", "t2": "/b", "t3": "/c"},
    )
    monkeypatch.setattr(
        muq_runner, "build_muq_embedder",
        lambda device="cpu", torch_threads=0: (lambda p: np.ones(4, np.float32)),
    )

    args = analyze.parse_args(["--config", str(config_path)])
    args.force = False
    args.limit = None
    args.progress = True
    args.progress_interval = 0.0   # force a heartbeat on every item
    args.progress_every = 1
    args.verbose = False

    ctx = {
        "config_path": str(config_path),
        "args": args,
        "out_dir": out_dir,
        "db_path": str(tmp_path / "metadata.db"),
    }

    caplog.set_level(logging.INFO, logger="analyze_library")
    result = analyze.stage_muq(ctx)

    assert result["ok"] == 3
    assert "MuQ-MuLan loaded in" in caplog.text
    assert "muq:" in caplog.text          # per-item heartbeat summary
    assert "muq complete" in caplog.text  # ProgressLogger.finish() line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_analyze_library_logging.py::test_stage_muq_logs_model_load_and_heartbeat`
Expected: FAIL — `"MuQ-MuLan loaded in"` / `"muq:"` absent from `caplog.text`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/analyze_library.py`, replace the tail of `stage_muq` (the block starting at the `logger.info("stage_muq: %d track(s) pending …")` call through `return {...}`):

```python
    logger.info("stage_muq: %d track(s) pending (device=%s); loading MuQ-MuLan "
                "(cold cache can take a minute)...", len(pending), device)
    _load_t0 = time.time()
    embed_fn = build_muq_embedder(device, torch_threads)
    logger.info("stage_muq: MuQ-MuLan loaded in %.1fs", time.time() - _load_t0)
    items = [(t, db_paths.get(t)) for t in pending]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress = None
    if getattr(args, "progress", True):
        from src.logging_utils import ProgressLogger
        progress = ProgressLogger(
            logger,
            total=len(pending),
            label="muq",
            unit="tracks",
            interval_s=float(getattr(args, "progress_interval", 15.0)),
            every_n=int(getattr(args, "progress_every", 500)),
            verbose_each=bool(getattr(args, "verbose", False)),
        )
    result = run_muq_extraction(items, embed_fn, sidecar_path, backup_stamp=stamp, progress=progress)
    logger.info("stage_muq: embedded ok=%d failed=%d -> %s",
                result["ok"], result["failed"], sidecar_path)
    return {"skipped": False, "pending": len(pending), **result}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_analyze_library_logging.py`
Expected: PASS — new test plus the 5 existing analyze-logging tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_library_logging.py
git commit -m "feat(analyze): stage_muq heartbeat + model-load timing via ProgressLogger" --only -- scripts/analyze_library.py tests/unit/test_analyze_library_logging.py
```

---

### Task 3: Per-run analyze-log path + rotation helpers (DRY with playlist cleanup)

**Files:**
- Modify: `src/logging_utils.py` (add helpers near the playlist trio at `:551-682`)
- Test: `tests/unit/test_analyze_logging_files.py` (create)

**Interfaces:**
- Produces:
  - `analyze_log_dir() -> Path` → `<repo_root>/logs/analyze`.
  - `make_analyze_log_path(run_id, *, dir=None) -> Path` → `<dir>/<YYYY-MM-DD_HHMMSS>_<run_id[:6]>.log`.
  - `cleanup_old_analyze_logs(dir=None, retention_days=30) -> int` and `cleanup_old_analyze_logs_async(dir=None, retention_days=30) -> None`.
  - Private `_cleanup_logs_older_than(base_dir: Path, retention_days: int) -> int` (shared by analyze + playlist cleanup).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_analyze_logging_files.py`:

```python
import os
import time
from pathlib import Path

from src.logging_utils import (
    make_analyze_log_path,
    cleanup_old_analyze_logs,
)


def test_make_analyze_log_path_shape(tmp_path):
    p = make_analyze_log_path("abcdef123456", dir=tmp_path)
    assert p.parent == Path(tmp_path)
    assert p.suffix == ".log"
    assert p.name.endswith("_abcdef.log")   # run_id truncated to 6


def test_cleanup_deletes_old_keeps_recent(tmp_path):
    old = tmp_path / "2020-01-01_000000_aaaaaa.log"
    new = tmp_path / "2026-07-06_000000_bbbbbb.log"
    old.write_text("x", encoding="utf-8")
    new.write_text("y", encoding="utf-8")
    old_mtime = time.time() - (40 * 86400)      # 40 days old
    os.utime(old, (old_mtime, old_mtime))

    deleted = cleanup_old_analyze_logs(dir=tmp_path, retention_days=30)
    assert deleted == 1
    assert not old.exists()
    assert new.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_analyze_logging_files.py`
Expected: FAIL — `ImportError: cannot import name 'make_analyze_log_path'`.

- [ ] **Step 3: Write minimal implementation**

In `src/logging_utils.py`, add after `cleanup_old_playlist_logs_async` (after `:682`):

```python
def _cleanup_logs_older_than(base_dir: Path, retention_days: int) -> int:
    """Delete *.log files under base_dir older than retention_days. Never raises."""
    try:
        if not base_dir.exists():
            return 0
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0
        for log_path in base_dir.glob("*.log"):
            try:
                if log_path.is_file() and log_path.stat().st_mtime < cutoff:
                    log_path.unlink()
                    deleted += 1
            except OSError:
                continue
        return deleted
    except Exception:
        return 0


def analyze_log_dir() -> Path:
    """ROOT-anchored directory for per-run Analyze Library log files."""
    return Path(__file__).resolve().parents[1] / "logs" / "analyze"


def make_analyze_log_path(run_id, *, dir: Optional[Union[str, Path]] = None) -> Path:
    """Build a unique, sortable per-run analyze log path.

    Shape: <dir>/<YYYY-MM-DD_HHMMSS>_<run_id[:6]>.log
    """
    base_dir = Path(dir) if dir is not None else analyze_log_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    shortid = str(run_id)[:6] if run_id else "000000"
    return base_dir / f"{timestamp}_{shortid}.log"


def cleanup_old_analyze_logs(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> int:
    """Delete analyze *.log files older than retention_days. Never raises."""
    base_dir = Path(dir) if dir is not None else analyze_log_dir()
    return _cleanup_logs_older_than(base_dir, retention_days)


def cleanup_old_analyze_logs_async(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> None:
    """Run cleanup_old_analyze_logs in a daemon thread. Never raises."""
    try:
        thread = threading.Thread(
            target=cleanup_old_analyze_logs,
            args=(dir, retention_days),
            daemon=True,
            name="analyze-log-cleanup",
        )
        thread.start()
    except Exception:
        pass
```

Then make the existing playlist cleanup delegate to the shared helper (DRY). Replace the body of `cleanup_old_playlist_logs` (`:643-664`) with:

```python
def cleanup_old_playlist_logs(
    dir: Optional[Union[str, Path]] = None,
    retention_days: int = 30,
) -> int:
    """Delete *.log files under dir older than retention_days. Never raises."""
    base_dir = Path(dir) if dir is not None else playlist_log_dir()
    return _cleanup_logs_older_than(base_dir, retention_days)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_analyze_logging_files.py tests/unit/test_playlist_logging.py`
Expected: PASS — new analyze tests plus the pre-existing playlist-logging tests (behavior of `cleanup_old_playlist_logs` is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/logging_utils.py tests/unit/test_analyze_logging_files.py
git commit -m "feat(logging): per-run analyze log path + shared age-based rotation" --only -- src/logging_utils.py tests/unit/test_analyze_logging_files.py
```

---

### Task 4: `run_pipeline` writes a per-run rotated log; drop the fixed file

**Files:**
- Modify: `scripts/analyze_library.py:2547-2565` (`run_pipeline` logging setup) and add `_analyze_log_settings` helper
- Modify: `config.example.yaml` (document `logging.analyze_logs.*`)
- Test: `tests/unit/test_analyze_library_logging.py`

**Interfaces:**
- Consumes: `make_analyze_log_path`, `cleanup_old_analyze_logs_async` (Task 3); `Config.get('logging','analyze_logs',default={})`.
- Produces: `_analyze_log_settings(config) -> Tuple[bool, str, int, str]` = `(enabled, dir, retention_days, level_name)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_analyze_library_logging.py`. First extend `_write_config` to accept an analyze-log dir (add a keyword param; default keeps existing callers working):

```python
def _write_config(tmp_path: Path, db_path: Path, analyze_log_dir: Path | None = None) -> Path:
    music_dir = tmp_path / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    text = (
        "library:\n"
        f"  database_path: {db_path}\n"
        f"  music_directory: {music_dir}\n"
        "openai:\n"
        "  api_key: test-key\n"
    )
    if analyze_log_dir is not None:
        text += (
            "logging:\n"
            "  analyze_logs:\n"
            "    enabled: true\n"
            f"    dir: {analyze_log_dir.as_posix()}\n"
            "    retention_days: 30\n"
            "    level: DEBUG\n"
        )
    config_path.write_text(text, encoding="utf-8")
    return config_path
```

Then the new test:

```python
def test_run_pipeline_writes_per_run_analyze_log(tmp_path, monkeypatch):
    db_path = tmp_path / "metadata.db"
    _make_db(db_path)
    alog_dir = tmp_path / "logs_analyze"
    config_path = _write_config(tmp_path, db_path, analyze_log_dir=alog_dir)
    out_dir = tmp_path / "artifacts"

    def _stub_scan(ctx):
        return {"total": 1, "scan_total": 1, "orphaned": {}}

    monkeypatch.setattr(analyze, "STAGE_FUNCS", {"scan": _stub_scan})

    args = analyze.parse_args(
        [
            "--config", str(config_path),
            "--db-path", str(db_path),
            "--stages", "scan",
            "--out-dir", str(out_dir),
        ]
    )  # NOTE: no --log-file -> per-run path is used

    analyze.run_pipeline(args, console_logging=False)

    logs = list(alog_dir.glob("*.log"))
    assert len(logs) == 1
    assert logs[0].name.endswith(".log")
    assert "Analyze run start" in logs[0].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/unit/test_analyze_library_logging.py::test_run_pipeline_writes_per_run_analyze_log`
Expected: FAIL — no `*.log` under `alog_dir` (run still writes the fixed `logs/analyze_library.log`).

- [ ] **Step 3: Write minimal implementation**

In `scripts/analyze_library.py`, add the settings helper (place it just above `def run_pipeline`):

```python
def _analyze_log_settings(config) -> Tuple[bool, str, int, str]:
    """Resolve logging.analyze_logs.* with documented defaults (mirrors
    _playlist_log_settings in main_app.py)."""
    cfg = config.get('logging', 'analyze_logs', default={}) or {}
    enabled = bool(cfg.get('enabled', True))
    directory = str(cfg.get('dir', 'logs/analyze'))
    retention_days = int(cfg.get('retention_days', 30))
    level_name = str(cfg.get('level', 'DEBUG')).upper()
    return enabled, directory, retention_days, level_name
```

Then replace the logging-setup block (currently `:2547-2565`, from the `from src.logging_utils import …` line through `cfg = Config(args.config)`):

```python
    from src.logging_utils import (
        configure_logging, resolve_log_level,
        make_analyze_log_path, cleanup_old_analyze_logs_async,
    )
    run_id = str(uuid.uuid4())
    log_level = resolve_log_level(args)
    if getattr(args, "verbose", False) and not getattr(args, "debug", False) and not getattr(args, "quiet", False) and getattr(args, "log_level", "INFO").upper() == "INFO":
        log_level = "DEBUG"

    cfg = Config(args.config)   # loaded early so the per-run log path can honor config
    a_enabled, a_dir, a_retention, a_level = _analyze_log_settings(cfg)
    explicit_log_file = getattr(args, 'log_file', None)
    if explicit_log_file:
        log_file = explicit_log_file          # honor --log-file exactly as before
        file_level = "DEBUG"
    elif a_enabled:
        log_file = str(make_analyze_log_path(run_id, dir=a_dir))
        file_level = a_level
    else:
        log_file = None
        file_level = "DEBUG"
    configure_logging(
        level=log_level,
        log_file=log_file,
        file_level=file_level,
        run_id=run_id,
        show_run_id=getattr(args, "show_run_id", False),
        console=console_logging,
        force=not console_logging,
    )
    if log_file and not explicit_log_file and a_enabled:
        cleanup_old_analyze_logs_async(dir=a_dir, retention_days=a_retention)

    # Re-get logger after configuration
    logger = logging.getLogger("analyze_library")
```

Delete the now-duplicate `cfg = Config(args.config)` that previously sat right after `logger = logging.getLogger("analyze_library")` (it moved up). Leave `db_path = args.db_path or cfg.library_database_path` and everything below unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_analyze_library_logging.py`
Expected: PASS — the new per-run-log test plus all existing analyze-logging tests (those pass `--log-file`, so they take the `explicit_log_file` branch and behave exactly as before).

- [ ] **Step 5: Document the config knobs**

In `config.example.yaml`, find the `logging:` block containing `playlist_logs:` and add a sibling:

```yaml
  # Per-run Analyze Library logs (one rotated file per run, mirrors playlist_logs).
  analyze_logs:
    enabled: true
    dir: logs/analyze
    retention_days: 30
    level: DEBUG
```

(If `logging:` has no `playlist_logs` block to sit beside, add the `analyze_logs` block under the existing `logging:` mapping.)

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_library.py config.example.yaml tests/unit/test_analyze_library_logging.py
git commit -m "feat(analyze): per-run rotated analyze logs; drop fixed analyze_library.log" --only -- scripts/analyze_library.py config.example.yaml tests/unit/test_analyze_library_logging.py
```

---

### Task 5: No builder stage goes dark — "starting" lines for genre-sim & artifacts (Part B)

**Files:**
- Modify: `scripts/analyze_library.py` — `stage_genre_sim` (before `:2006`) and `stage_artifacts` (before the build call at `:2081`)
- Test: `tests/unit/test_analyze_library_logging.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `stage_genre_sim` emits `genre-sim: building similarity matrix (source=…)…`; `stage_artifacts` emits `artifacts: building DS data matrices…` before their multi-second work.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_analyze_library_logging.py`. `stage_genre_sim` and `stage_artifacts` resolve their heavy deps via `from … import …` *inside* the function, so patch them on their **source** modules (that is where the call-time lookup lands):

```python
def test_genre_sim_logs_starting_line(tmp_path, monkeypatch, caplog):
    import scripts.analyze_library as a
    import src.config_loader as config_loader
    import src.genre.graph_adapter as graph_adapter
    import src.genre.graph_similarity as graph_similarity

    config_path = _write_config(tmp_path, tmp_path / "metadata.db")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config_loader.Config, "get_ds_genre_similarity_source", lambda self: "graph")
    monkeypatch.setattr(graph_adapter, "load_graph_adapter", lambda: object())

    class _GraphResult:
        stats = {}

    monkeypatch.setattr(graph_similarity, "build_graph_similarity", lambda adapter: _GraphResult())
    monkeypatch.setattr(
        graph_similarity, "save_graph_similarity_npz",
        lambda result, path: Path(path).write_text("x", encoding="utf-8"),
    )

    args = a.parse_args(["--config", str(config_path)])
    args.force = True
    ctx = {"out_dir": out_dir, "config_path": str(config_path),
           "db_path": str(tmp_path / "metadata.db"), "args": args}

    caplog.set_level(logging.INFO, logger="analyze_library")
    a.stage_genre_sim(ctx)
    assert "genre-sim: building similarity matrix" in caplog.text


def test_artifacts_logs_starting_line(tmp_path, monkeypatch, caplog):
    import scripts.analyze_library as a
    import scripts.build_beat3tower_artifacts as bba

    config_path = _write_config(tmp_path, tmp_path / "metadata.db")
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(bba, "build_artifacts", lambda args_ns: None)

    args = a.parse_args(["--config", str(config_path)])
    args.force = True
    args.max_tracks = 0
    args.limit = None
    args.verbose = False
    ctx = {"out_dir": out_dir, "config_path": str(config_path),
           "db_path": str(tmp_path / "metadata.db"), "args": args}

    caplog.set_level(logging.INFO, logger="analyze_library")
    a.stage_artifacts(ctx)
    assert "artifacts: building DS data matrices" in caplog.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest -q "tests/unit/test_analyze_library_logging.py::test_genre_sim_logs_starting_line" "tests/unit/test_analyze_library_logging.py::test_artifacts_logs_starting_line"`
Expected: FAIL — the `"… building …"` starting lines are absent from `caplog.text`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/analyze_library.py`, in `stage_genre_sim`, immediately before `if sim_source == "graph":` (`:2006`):

```python
    logger.info("genre-sim: building similarity matrix (source=%s)...", sim_source)
```

In `stage_artifacts`, immediately before `try:` / `build_beat3tower_artifacts(args_ns)` (`:2080`):

```python
    logger.info("artifacts: building DS data matrices (this can take a few minutes)...")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -q tests/unit/test_analyze_library_logging.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_analyze_library_logging.py
git commit -m "feat(analyze): genre-sim and artifacts log a starting line (no dark stages)" --only -- scripts/analyze_library.py tests/unit/test_analyze_library_logging.py
```

---

## Final verification (after all tasks)

- [ ] **Full fast suite:** `python -m pytest -q -m "not slow"` — quote real pass/fail counts.
- [ ] **Lint / types:** `ruff check src/analyze/muq_runner.py src/logging_utils.py scripts/analyze_library.py` and `mypy src/logging_utils.py src/analyze/muq_runner.py`.
- [ ] **Live GUI path (per CLAUDE.md — exercise the real path):** restart `python tools/serve_web.py`, run Analyze Library, and confirm in the GUI LogPanel: (a) a `stage_muq: MuQ-MuLan loaded in …s` line, (b) a `muq: N/total (…) | … tracks/s | ETA …` line roughly every 15s during MuQ, and (c) a new `logs/analyze/<ts>_<run_id6>.log` file for the run. If MuQ has nothing pending, force it (`analyze.muq` with `--force`, or a small `--limit`) so the heartbeat path actually runs.

---

## Self-review notes

- **Spec coverage:** Part A → Tasks 1–2; Part B → Task 5; Part C → Tasks 3–4 (per-run path + rotation + config + wiring). Non-goals respected (no bridge/parser change, no `metadata.db` writes, no new dependency).
- **`progress=None` invariant:** Task 1 keeps all 8 existing `test_muq_runner.py` tests unchanged.
- **`--log-file` override:** Task 4 honors an explicit `--log-file` (the branch all existing analyze-logging tests exercise), so they stay green; only the no-flag default becomes the per-run path.
- **GUI forwarding intact:** `configure_logging(force=True)` only strips `_HANDLER_TAG` handlers; the worker's untagged `WorkerLogHandler` survives, so heartbeat INFO lines reach the LogPanel.
