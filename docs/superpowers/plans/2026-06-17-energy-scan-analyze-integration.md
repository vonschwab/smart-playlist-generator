# Energy scan → Analyze Library integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `energy` stage to the `analyze_library` pipeline (CLI + GUI Tools panel) that runs the WSL-only Essentia extractor to produce `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz` (emoMusic arousal p10/p50/p90 + danceability).

**Architecture:** A new in-process `stage_energy(ctx)` in `scripts/analyze_library.py` shells out to WSL (the only stage that does; even MERT runs in-process) to drive the already-validated `scripts/extract_energy_sidecar.py`. The WSL-invocation logic lives in a focused, testable module `src/analyze/energy_runner.py` (config load, path translation, WSL preflight, pending calc, subprocess stream/cancel/parse). The stage is registered in the three mirrored stage lists and flows to the CLI and GUI automatically. Energy is a standalone **pace-axis sidecar** — never folded into the sonic blend, never written to metadata.db.

**Tech Stack:** Python 3.11+ (pipeline, Windows), WSL2 Ubuntu-22.04 + `essentia-tensorflow` venv at `/opt/ess` (extractor only), React+TS (`web/src`), pytest/ruff/mypy.

## Global Constraints

- **Python 3.11+.** The Windows runtime/pipeline MUST NOT import `essentia` — it is WSL-only at `/opt/ess`. `stage_energy`/`energy_runner` shell out via `wsl.exe`.
- **Energy is a standalone pace-axis sidecar.** Never fold it into the sonic similarity blend; never write `metadata.db` or the sonic/MERT artifacts. Output is confined to `data/artifacts/beat3tower_32k/energy/`.
- **Default-on, hard-fail if WSL missing.** `energy` is in the default stage set; if WSL/venv/models are unreachable, raise `RuntimeError` with remediation (no silent skip).
- **Stage list is mirrored in THREE places — keep in sync:** `scripts/analyze_library.py` (`STAGE_ORDER_DEFAULT` + `STAGE_FUNCS`), `src/playlist/request_models.py` (`AnalyzeLibraryStage` + `ANALYZE_LIBRARY_STAGE_ORDER`), `web/src/components/ToolsPanel.tsx` (`ALL_STAGES`). Placement: immediately **after `artifacts`**.
- **After any `web/src` edit:** `npm --prefix web run build` (rebuild `web/dist`) and restart `serve_web.py` (Core Rules 1 & 2 of the web-gui skill).
- **Config defaults (`analyze.energy`):** `distro: Ubuntu-22.04`, `python: /opt/ess/bin/python`, `models_dir: /opt/ess/models`, `workers: 14`.
- **Gates before done:** `ruff check` (E,F) clean, `mypy` clean, `python -m pytest -q -m "not slow"` green. Never pipe pytest through tail/head.
- **Worktree:** all work on branch `worktree-energy-analyze-integration`. Commit per task.

## File Structure

- **Create** `scripts/extract_energy_sidecar.py` — the validated WSL-side scanner (resumable, parallel, merges sidecar). [Task 1]
- **Create** `src/analyze/energy_runner.py` — config/paths/preflight/pending/subprocess driver. One responsibility: drive the WSL extractor and report. [Tasks 2–3]
- **Create** `tests/unit/test_energy_runner.py` — unit tests (subprocess injected). [Tasks 2–3]
- **Modify** `scripts/analyze_library.py` — `stage_energy`, `STAGE_FUNCS`, `STAGE_ORDER_DEFAULT`, `compute_stage_fingerprint`, `estimate_stage_units`, `run_pipeline` (expose `cancellation_check` in `ctx`), `parse_args` (`--energy-workers`). [Task 4]
- **Create** `tests/unit/test_stage_energy.py` — `stage_energy` unit tests (`energy_runner` monkeypatched). [Task 4]
- **Modify** `src/playlist/request_models.py` — add `"energy"` to enum + order. [Task 5]
- **Modify** `src/playlist/analyze_library_results.py` — add `"energy"` action label. [Task 5]
- **Modify** `config.example.yaml` — `analyze.energy` block. [Task 5]
- **Create** `tests/unit/test_request_models_energy.py` — stage-order/clean test. [Task 5]
- **Modify** `web/src/components/ToolsPanel.tsx` — add `"energy"` to `ALL_STAGES`; rebuild `web/dist`. [Task 6]
- **Verify** `tests/fixtures/fake_worker.py` still drives `analyze_library` with the new stage. [Task 6]

---

### Task 1: Add the validated WSL-side extractor to the branch

The extractor is already authored and exercised end-to-end (30-track smoke: `ok=30 missing=40363 error=0 total=40393`, sidecar aligned to artifact, Windows-readable). This task brings it into the feature branch verbatim and adds a unit test for its one pure, Windows-safe helper.

**Files:**
- Create: `scripts/extract_energy_sidecar.py`
- Test: `tests/unit/test_extract_energy_sidecar.py`

**Interfaces:**
- Produces: CLI `--workers N | --limit N | --merge-only`; writes `data/artifacts/beat3tower_32k/energy/{checkpoint.jsonl,energy_sidecar.npz}`. Module-level pure helper `_win_to_wsl(p:str)->str`. `essentia` is imported ONLY inside worker functions (`_init`/`_process`), never at module top — so the module is importable on Windows for testing the pure helper and `_merge`.

- [ ] **Step 1: Create `scripts/extract_energy_sidecar.py`** with this exact content:

```python
"""Extract Essentia energy descriptors (arousal distribution + danceability) for
the whole library into a sidecar npz aligned to the beat3tower artifact.

RUNS UNDER WSL ONLY (Essentia lives in the WSL venv at /opt/ess; the Windows
runtime never imports this). Invoke with /opt/ess/bin/python.

- Scope = the artifact's track_ids (the exact generation set); paths from
  metadata.db (READ-ONLY). Energy is a PACE-axis sidecar, never a metadata.db
  write and never folded into the sonic-similarity blend.
- Per track: emoMusic arousal (p10/p50/p90 -- distribution, the mean masks
  dynamics), valence (p50), danceability P. msd-musicnn embeddings @16kHz.
- Resumable: append-only JSONL checkpoint; re-running skips done track_ids.
- Parallel: spawn pool (TF-safe -- essentia imported only inside workers,
  AFTER thread env is pinned to 1, never at module top).
- Writes ONLY to <artifact>/energy/; backs up an existing sidecar timestamped
  before overwrite. Touches nothing else in the artifact dir.

Usage (from WSL, repo on /mnt/c):
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --workers 14
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --merge-only
    /opt/ess/bin/python scripts/extract_energy_sidecar.py --limit 50   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "data", "artifacts", "beat3tower_32k")
NPZ = os.path.join(ART, "data_matrices_step1.npz")
DB = os.path.join(ROOT, "data", "metadata.db")
OUTDIR = os.path.join(ART, "energy")
CKPT = os.path.join(OUTDIR, "checkpoint.jsonl")
SIDECAR = os.path.join(OUTDIR, "energy_sidecar.npz")
MODELS = "/opt/ess/models"

EMB_PB = f"{MODELS}/msd-musicnn-1.pb"
AV_PB = f"{MODELS}/emomusic-msd-musicnn-2.pb"
DANCE_PB = f"{MODELS}/danceability-msd-musicnn-1.pb"

_emb = _av = _dc = None


def _win_to_wsl(p: str) -> str:
    p = p.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def _artifact_track_ids() -> list[str]:
    z = np.load(NPZ, allow_pickle=True)
    return [str(t) for t in z["track_ids"]]


def _paths_for(track_ids: list[str]) -> dict[str, str]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    out: dict[str, str] = {}
    B = 900
    try:
        for i in range(0, len(track_ids), B):
            batch = track_ids[i : i + B]
            ph = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT track_id, file_path FROM tracks WHERE track_id IN ({ph})",
                tuple(batch),
            )
            for r in cur.fetchall():
                if r["file_path"]:
                    out[str(r["track_id"])] = _win_to_wsl(r["file_path"])
    finally:
        con.close()
    return out


def _done_ids() -> set[str]:
    done: set[str] = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["track_id"])
                except Exception:
                    continue
    return done


def _init() -> None:
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings

    warnings.filterwarnings("ignore")
    import essentia

    essentia.log.warningActive = False
    import essentia.standard as es

    global _emb, _av, _dc
    _emb = es.TensorflowPredictMusiCNN(graphFilename=EMB_PB, output="model/dense/BiasAdd")
    _av = es.TensorflowPredict2D(graphFilename=AV_PB, output="model/Identity")
    _dc = es.TensorflowPredict2D(graphFilename=DANCE_PB, output="model/Softmax")


def _process(item: tuple[str, str | None]) -> dict:
    tid, path = item
    if not path or not os.path.exists(path):
        return {"track_id": tid, "missing": True}
    try:
        import essentia.standard as es

        audio = es.MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
        if len(audio) == 0:
            return {"track_id": tid, "error": "empty_audio"}
        emb = _emb(audio)
        av = _av(emb)
        dc = _dc(emb)
        aro = av[:, 1]
        return {
            "track_id": tid,
            "valence": round(float(np.mean(av[:, 0])), 4),
            "arousal_p10": round(float(np.percentile(aro, 10)), 4),
            "arousal_p50": round(float(np.percentile(aro, 50)), 4),
            "arousal_p90": round(float(np.percentile(aro, 90)), 4),
            "danceability": round(float(np.mean(dc[:, 0])), 4),
            "frames": int(emb.shape[0]),
        }
    except Exception as ex:  # noqa: BLE001
        return {"track_id": tid, "error": repr(ex)[:200]}


def _merge() -> None:
    tids = _artifact_track_ids()
    rec: dict[str, dict] = {}
    with open(CKPT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec[d["track_id"]] = d
    n = len(tids)
    val = np.full(n, np.nan, np.float32)
    p10 = np.full(n, np.nan, np.float32)
    p50 = np.full(n, np.nan, np.float32)
    p90 = np.full(n, np.nan, np.float32)
    dance = np.full(n, np.nan, np.float32)
    frames = np.zeros(n, np.int32)
    ok = miss = err = 0
    for i, tid in enumerate(tids):
        d = rec.get(tid)
        if not d or "arousal_p50" not in d:
            if d and d.get("error"):
                err += 1
            else:
                miss += 1
            continue
        val[i] = d["valence"]
        p10[i] = d["arousal_p10"]
        p50[i] = d["arousal_p50"]
        p90[i] = d["arousal_p90"]
        dance[i] = d["danceability"]
        frames[i] = d["frames"]
        ok += 1
    if os.path.exists(SIDECAR):
        bak = SIDECAR + "." + time.strftime("%Y%m%d_%H%M%S") + ".bak"
        os.rename(SIDECAR, bak)
        print(f"backed up existing sidecar -> {bak}")
    np.savez_compressed(
        SIDECAR,
        track_ids=np.array(tids, dtype=object),
        valence=val,
        arousal_p10=p10,
        arousal_p50=p50,
        arousal_p90=p90,
        danceability=dance,
        frames=frames,
        model=np.array("emomusic-msd-musicnn-2 + danceability-msd-musicnn-1", dtype=object),
    )
    print(f"wrote {SIDECAR}: ok={ok} missing={miss} error={err} total={n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (smoke test)")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    if args.merge_only:
        _merge()
        return

    tids = _artifact_track_ids()
    paths = _paths_for(tids)
    done = _done_ids()
    todo = [(t, paths.get(t)) for t in tids if t not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"artifact={len(tids)} done={len(done)} todo={len(todo)} workers={args.workers}",
        flush=True,
    )
    if not todo:
        print("nothing to do; merging.")
        _merge()
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t0 = time.time()
    n = 0
    with open(CKPT, "a", encoding="utf-8") as f:
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for d in pool.imap_unordered(_process, todo, chunksize=4):
                f.write(json.dumps(d) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0:
                    rate = n / (time.time() - t0)
                    eta_h = (len(todo) - n) / rate / 3600 if rate else float("inf")
                    print(f"  {n}/{len(todo)}  {rate:.2f} trk/s  ETA {eta_h:.1f}h", flush=True)
    print(f"scan pass done: {n} tracks in {(time.time()-t0)/3600:.2f}h", flush=True)
    _merge()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_extract_energy_sidecar.py`:

```python
from scripts.extract_energy_sidecar import _win_to_wsl


def test_win_to_wsl_drive_letter():
    assert _win_to_wsl(r"E:\MUSIC\a.flac") == "/mnt/e/MUSIC/a.flac"


def test_win_to_wsl_already_posix():
    assert _win_to_wsl("/mnt/e/x.flac") == "/mnt/e/x.flac"
```

- [ ] **Step 3: Run test to verify it passes** (module imports cleanly on Windows — essentia is not imported at module top):

Run: `python -m pytest tests/unit/test_extract_energy_sidecar.py -q`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_energy_sidecar.py tests/unit/test_extract_energy_sidecar.py
git commit -m "feat(energy): add validated WSL-side energy extractor"
```

---

### Task 2: `energy_runner` — config, paths, preflight, pending

**Files:**
- Create: `src/analyze/energy_runner.py`
- Test: `tests/unit/test_energy_runner.py`

**Interfaces:**
- Produces:
  - `@dataclass EnergyConfig(distro:str="Ubuntu-22.04", python:str="/opt/ess/bin/python", models_dir:str="/opt/ess/models", workers:int=14)`
  - `load_energy_config(config_path:str) -> EnergyConfig`
  - `win_path_to_wsl(p:str) -> str`
  - `energy_paths(out_dir:Path) -> tuple[Path, Path, Path]` → `(artifact_npz, checkpoint_jsonl, sidecar_npz)`
  - `pending_energy(out_dir:Path) -> tuple[int,int]` → `(pending, total)`; `total` = artifact track_id count, `pending` = those not in `checkpoint.jsonl`. Returns `(0,0)` if the artifact npz is absent.
  - `preflight_wsl(cfg:EnergyConfig, *, runner=subprocess.run) -> None` — raises `RuntimeError` if WSL/venv/models unreachable.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_energy_runner.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.analyze import energy_runner as er


def test_win_path_to_wsl():
    assert er.win_path_to_wsl(r"C:\Users\Dylan\proj") == "/mnt/c/Users/Dylan/proj"


def test_load_energy_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("library:\n  music_directory: /x\n", encoding="utf-8")
    cfg = er.load_energy_config(str(cfg_file))
    assert cfg.distro == "Ubuntu-22.04"
    assert cfg.python == "/opt/ess/bin/python"
    assert cfg.workers == 14


def test_load_energy_config_overrides(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "analyze:\n  energy:\n    distro: Ubuntu-24.04\n    workers: 8\n",
        encoding="utf-8",
    )
    cfg = er.load_energy_config(str(cfg_file))
    assert cfg.distro == "Ubuntu-24.04"
    assert cfg.workers == 8
    assert cfg.python == "/opt/ess/bin/python"  # untouched default


def test_pending_energy(tmp_path):
    out = tmp_path
    np.savez(
        out / "data_matrices_step1.npz",
        track_ids=np.array(["a", "b", "c"], dtype=object),
    )
    (out / "energy").mkdir()
    (out / "energy" / "checkpoint.jsonl").write_text(
        '{"track_id": "a", "arousal_p50": 4.5}\n', encoding="utf-8"
    )
    pending, total = er.pending_energy(out)
    assert (pending, total) == (2, 3)


def test_pending_energy_no_artifact(tmp_path):
    assert er.pending_energy(tmp_path) == (0, 0)


def test_preflight_wsl_ok():
    def fake_runner(cmd, **kw):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    er.preflight_wsl(er.EnergyConfig(), runner=fake_runner)  # no raise


def test_preflight_wsl_missing_raises():
    def fake_runner(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
            stderr = "not found"
        return R()

    with pytest.raises(RuntimeError, match="WSL"):
        er.preflight_wsl(er.EnergyConfig(), runner=fake_runner)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_energy_runner.py -q`
Expected: FAIL (`ModuleNotFoundError: src.analyze.energy_runner`).

- [ ] **Step 3: Create `src/analyze/energy_runner.py`** (Task-2 portion):

```python
"""Drive the WSL-only Essentia energy extractor from the Windows pipeline.

The Windows runtime never imports essentia; this module shells out to
`wsl.exe ... /opt/ess/bin/python scripts/extract_energy_sidecar.py`.
Energy is a standalone pace-axis sidecar under <artifact>/energy/.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml


@dataclass
class EnergyConfig:
    distro: str = "Ubuntu-22.04"
    python: str = "/opt/ess/bin/python"
    models_dir: str = "/opt/ess/models"
    workers: int = 14


def load_energy_config(config_path: str) -> EnergyConfig:
    """Read analyze.energy from config.yaml; defaults on any miss."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    block = ((cfg.get("analyze") or {}).get("energy") or {})
    default = EnergyConfig()
    return EnergyConfig(
        distro=str(block.get("distro", default.distro)),
        python=str(block.get("python", default.python)),
        models_dir=str(block.get("models_dir", default.models_dir)),
        workers=int(block.get("workers", default.workers)),
    )


def win_path_to_wsl(p: str) -> str:
    p = str(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def energy_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    return (
        out_dir / "data_matrices_step1.npz",
        out_dir / "energy" / "checkpoint.jsonl",
        out_dir / "energy" / "energy_sidecar.npz",
    )


def pending_energy(out_dir: Path) -> tuple[int, int]:
    artifact_npz, ckpt, _ = energy_paths(out_dir)
    if not artifact_npz.exists():
        return (0, 0)
    track_ids = [str(t) for t in np.load(artifact_npz, allow_pickle=True)["track_ids"]]
    done: set[str] = set()
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["track_id"])
                except Exception:
                    continue
    pending = sum(1 for t in track_ids if t not in done)
    return (pending, len(track_ids))


def preflight_wsl(cfg: EnergyConfig, *, runner: Callable = subprocess.run) -> None:
    """Raise RuntimeError if the WSL distro, venv, or models are unreachable."""
    probe = f"test -x {cfg.python} && test -f {cfg.models_dir}/msd-musicnn-1.pb"
    cmd = ["wsl.exe", "-d", cfg.distro, "-u", "root", "--", "bash", "-c", probe]
    try:
        res = runner(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "WSL not available (wsl.exe not found). The energy stage needs WSL2 + "
            "the Essentia venv at /opt/ess. See project_energy_feature_exploration."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WSL preflight failed to run: {exc!r}") from exc
    if getattr(res, "returncode", 1) != 0:
        raise RuntimeError(
            f"WSL energy environment not ready (distro={cfg.distro}, python={cfg.python}, "
            f"models={cfg.models_dir}). Set up the /opt/ess venv + models, or fix "
            f"analyze.energy in config.yaml. stderr: {getattr(res, 'stderr', '')!r}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_energy_runner.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analyze/energy_runner.py tests/unit/test_energy_runner.py
git commit -m "feat(energy): energy_runner config/paths/preflight/pending"
```

---

### Task 3: `energy_runner.run_energy_scan` — subprocess stream, cancel, parse

**Files:**
- Modify: `src/analyze/energy_runner.py`
- Test: `tests/unit/test_energy_runner.py` (append)

**Interfaces:**
- Produces: `run_energy_scan(cfg:EnergyConfig, *, repo_root:Path, force:bool, logger, cancellation_check:Optional[Callable[[],None]]=None, popen=subprocess.Popen) -> dict` returning `{"ok":int,"missing":int,"error":int,"total":int,"sidecar":Optional[str]}` parsed from the extractor's final `wrote ... ok=.. missing=.. error=.. total=..` line. Streams stdout to `logger.info`. Raises `RuntimeError` on non-zero exit. On `cancellation_check()` raising, terminates the subprocess and re-raises.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_energy_runner.py`):

```python
import logging
import re


class _FakeProc:
    def __init__(self, lines, returncode=0, raise_on_line=None):
        self._lines = list(lines)
        self.returncode = returncode
        self.stdout = iter(self._lines)
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_run_energy_scan_parses_result():
    lines = [
        "artifact=3 done=0 todo=3 workers=2\n",
        "  100/3  1.0 trk/s  ETA 0.0h\n",
        "scan pass done: 3 tracks in 0.00h\n",
        "wrote /x/energy/energy_sidecar.npz: ok=2 missing=1 error=0 total=3\n",
    ]
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc(lines, returncode=0)

    res = er.run_energy_scan(
        er.EnergyConfig(workers=2),
        repo_root=Path(r"C:\repo"),
        force=False,
        logger=logging.getLogger("test"),
        popen=fake_popen,
    )
    assert res == {"ok": 2, "missing": 1, "error": 0, "total": 3,
                   "sidecar": "/x/energy/energy_sidecar.npz"}
    # command shells to wsl with the translated repo path
    joined = " ".join(captured["cmd"])
    assert "wsl.exe" in captured["cmd"][0]
    assert "/mnt/c/repo" in joined
    assert "--workers 2" in joined
    assert "--force" not in joined


def test_run_energy_scan_force_flag():
    def fake_popen(cmd, **kw):
        assert "--force" in " ".join(cmd)
        return _FakeProc(["wrote x: ok=0 missing=0 error=0 total=0\n"], 0)

    er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=True,
                       logger=logging.getLogger("test"), popen=fake_popen)


def test_run_energy_scan_nonzero_raises():
    def fake_popen(cmd, **kw):
        return _FakeProc(["boom\n"], returncode=2)

    with pytest.raises(RuntimeError, match="exit"):
        er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=False,
                           logger=logging.getLogger("test"), popen=fake_popen)


def test_run_energy_scan_cancel_terminates():
    proc_holder = {}

    def fake_popen(cmd, **kw):
        p = _FakeProc(["line1\n", "line2\n"], returncode=0)
        proc_holder["p"] = p
        return p

    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=False,
                           logger=logging.getLogger("test"),
                           cancellation_check=cancel, popen=fake_popen)
    assert proc_holder["p"].terminated is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_energy_runner.py -q -k run_energy_scan`
Expected: FAIL (`AttributeError: module ... has no attribute 'run_energy_scan'`).

- [ ] **Step 3: Append `run_energy_scan` to `src/analyze/energy_runner.py`:**

```python
import re

_RESULT_RE = re.compile(
    r"ok=(\d+)\s+missing=(\d+)\s+error=(\d+)\s+total=(\d+)"
)
_SIDECAR_RE = re.compile(r"^wrote\s+(\S+):")


def run_energy_scan(
    cfg: EnergyConfig,
    *,
    repo_root: Path,
    force: bool,
    logger,
    cancellation_check: Optional[Callable[[], None]] = None,
    popen: Callable = subprocess.Popen,
) -> dict:
    """Run the WSL extractor, stream progress to logger, return parsed counts."""
    wsl_repo = win_path_to_wsl(str(repo_root))
    inner = (
        f"cd '{wsl_repo}' && {cfg.python} scripts/extract_energy_sidecar.py "
        f"--workers {int(cfg.workers)}"
    )
    if force:
        inner += " --force"
    cmd = ["wsl.exe", "-d", cfg.distro, "-u", "root", "--", "bash", "-c", inner]

    proc = popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    result = {"ok": 0, "missing": 0, "error": 0, "total": 0, "sidecar": None}
    try:
        for raw in proc.stdout:  # type: ignore[union-attr]
            if cancellation_check is not None:
                cancellation_check()  # raises on cancel
            line = raw.rstrip()
            if not line:
                continue
            logger.info("energy: %s", line)
            m = _RESULT_RE.search(line)
            if m:
                result["ok"], result["missing"], result["error"], result["total"] = (
                    int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                )
            sm = _SIDECAR_RE.match(line)
            if sm:
                result["sidecar"] = sm.group(1)
        rc = proc.wait()
    except BaseException:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        raise
    if rc != 0:
        raise RuntimeError(f"energy extractor exited with code {rc}")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_energy_runner.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/analyze/energy_runner.py tests/unit/test_energy_runner.py
git commit -m "feat(energy): run_energy_scan WSL stream/cancel/parse"
```

---

### Task 4: `stage_energy` + pipeline registration

**Files:**
- Modify: `scripts/analyze_library.py`
- Test: `tests/unit/test_stage_energy.py`

**Interfaces:**
- Consumes: `src.analyze.energy_runner.{load_energy_config, pending_energy, preflight_wsl, run_energy_scan, energy_paths}`; `ROOT_DIR`, `logger`, `STAGE_FUNCS`, `STAGE_ORDER_DEFAULT` from `analyze_library`.
- Produces: `stage_energy(ctx) -> dict`; `STAGE_FUNCS["energy"]`; `"energy"` in `STAGE_ORDER_DEFAULT` after `"artifacts"`; `compute_stage_fingerprint(ctx,"energy")`; `estimate_stage_units(ctx,"energy")`; `ctx["cancellation_check"]` set in `run_pipeline`; `--energy-workers` arg.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_stage_energy.py`:

```python
import types
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_library as al


def _ctx(tmp_path, force=False, energy_workers=None):
    args = types.SimpleNamespace(force=force, energy_workers=energy_workers)
    return {
        "args": args,
        "config_path": str(tmp_path / "config.yaml"),
        "out_dir": tmp_path,
        "cancellation_check": None,
    }


def _make_artifact(tmp_path, ids=("a", "b")):
    np.savez(tmp_path / "data_matrices_step1.npz",
             track_ids=np.array(list(ids), dtype=object))


def test_stage_energy_skips_when_no_artifact(tmp_path):
    res = al.stage_energy(_ctx(tmp_path))
    assert res["skipped"] is True
    assert res.get("reason") == "no_artifact"


def test_stage_energy_skips_when_nothing_pending(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (0, 2))
    called = {"preflight": False}
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: called.__setitem__("preflight", True))
    res = al.stage_energy(_ctx(tmp_path, force=False))
    assert res["skipped"] is True
    assert called["preflight"] is False  # never touches WSL when up-to-date


def test_stage_energy_runs_and_returns_counts(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))
    monkeypatch.setattr(al, "_energy_preflight", lambda cfg: None)
    monkeypatch.setattr(
        al, "_energy_run",
        lambda cfg, force, cancellation_check: {
            "ok": 2, "missing": 0, "error": 0, "total": 2, "sidecar": "/x.npz"},
    )
    res = al.stage_energy(_ctx(tmp_path, force=False))
    assert res["skipped"] is False
    assert res["ok"] == 2 and res["pending"] == 2


def test_stage_energy_preflight_failure_propagates(tmp_path, monkeypatch):
    _make_artifact(tmp_path)
    monkeypatch.setattr(al, "_energy_pending", lambda out_dir: (2, 2))

    def boom(cfg):
        raise RuntimeError("WSL not available")

    monkeypatch.setattr(al, "_energy_preflight", boom)
    with pytest.raises(RuntimeError, match="WSL"):
        al.stage_energy(_ctx(tmp_path))


def test_energy_registered_and_ordered():
    assert "energy" in al.STAGE_FUNCS
    assert "energy" in al.STAGE_ORDER_DEFAULT
    assert al.STAGE_ORDER_DEFAULT.index("energy") == al.STAGE_ORDER_DEFAULT.index("artifacts") + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_stage_energy.py -q`
Expected: FAIL (`AttributeError: ... 'stage_energy'` / `'energy' not in STAGE_FUNCS`).

- [ ] **Step 3a: Add the stage + thin seams to `scripts/analyze_library.py`** — insert after `stage_mert` (before the `STAGE_FUNCS = {` block at ~line 1982). The seam wrappers (`_energy_*`) exist so tests monkeypatch one module:

```python
def _energy_pending(out_dir):
    from src.analyze.energy_runner import pending_energy
    return pending_energy(out_dir)


def _energy_preflight(cfg):
    from src.analyze.energy_runner import preflight_wsl
    preflight_wsl(cfg)


def _energy_run(cfg, *, force, cancellation_check):
    from src.analyze.energy_runner import run_energy_scan
    return run_energy_scan(
        cfg, repo_root=ROOT_DIR, force=force, logger=logger,
        cancellation_check=cancellation_check,
    )


def stage_energy(ctx: Dict) -> Dict:
    """Run the WSL-only Essentia energy scan into <artifact>/energy/energy_sidecar.npz.

    Default-on; skip-fast when up-to-date (like MERT). Hard-fail (RuntimeError)
    if WSL/Essentia is unreachable. Standalone pace-axis sidecar — never folded
    into the sonic blend, never writes metadata.db.
    """
    from src.analyze.energy_runner import load_energy_config, energy_paths

    args = ctx["args"]
    out_dir = Path(ctx["out_dir"])
    cfg = load_energy_config(ctx["config_path"])
    energy_workers = getattr(args, "energy_workers", None)
    if energy_workers:
        cfg.workers = int(energy_workers)

    artifact_npz, _ckpt, _sidecar = energy_paths(out_dir)
    if not artifact_npz.exists():
        logger.info("stage_energy: artifact missing; skipping (build artifacts first)")
        return {"skipped": True, "pending": 0, "reason": "no_artifact"}

    pending, total = _energy_pending(out_dir)
    if pending == 0 and not args.force:
        logger.info("stage_energy: nothing pending (sidecar complete); skipping")
        return {"skipped": True, "pending": 0}

    logger.info("stage_energy: %d/%d track(s) pending (workers=%d, distro=%s)",
                pending, total, cfg.workers, cfg.distro)
    _energy_preflight(cfg)  # raises RuntimeError if WSL/venv/models missing
    res = _energy_run(cfg, force=bool(args.force),
                      cancellation_check=ctx.get("cancellation_check"))
    return {"skipped": False, "pending": pending, **res}
```

- [ ] **Step 3b: Register the stage.** In `STAGE_FUNCS` (~line 1982) add `"energy": stage_energy,` after the `"artifacts"` entry. In `STAGE_ORDER_DEFAULT` (~line 48) insert `"energy"` right after `"artifacts"`:

```python
STAGE_ORDER_DEFAULT = ["scan", "genres", "discogs", "lastfm", "sonic", "mert", "enrich", "publish", "genre-sim", "artifacts", "energy", "genre-embedding", "verify"]
```

```python
STAGE_FUNCS = {
    "scan": stage_scan,
    "mbid": stage_mbid,
    "genres": stage_genres,
    "discogs": stage_discogs,
    "lastfm": stage_lastfm,
    "enrich": stage_enrich,
    "publish": stage_publish,
    "sonic": stage_sonic,
    "mert": stage_mert,
    "genre-sim": stage_genre_sim,
    "artifacts": stage_artifacts,
    "energy": stage_energy,
    "genre-embedding": stage_genre_embedding,
    "verify": stage_verify,
}
```

- [ ] **Step 3c: Add the fingerprint branch** in `compute_stage_fingerprint` (after the `mert` branch, ~line 318), so the stage re-runs when new tracks appear / config changes:

```python
    if stage == "energy":
        from src.analyze.energy_runner import pending_energy, load_energy_config
        pending, total = pending_energy(Path(ctx["out_dir"]))
        cfg = load_energy_config(ctx["config_path"])
        key = {"stage": stage, "pending": pending, "total": total,
               "workers": cfg.workers, "distro": cfg.distro, "python": cfg.python}
        return _hash_obj(key)
```

- [ ] **Step 3d: Add the unit estimate** in `estimate_stage_units` (inside the `try`, after the `artifacts` branch, ~line 495):

```python
        if stage == "energy":
            from src.analyze.energy_runner import pending_energy
            pending, _total = pending_energy(Path(ctx["out_dir"]))
            return pending, "tracks needing energy descriptors"
```

- [ ] **Step 3e: Expose cancellation to stages.** In `run_pipeline`, immediately after the `ctx = {...}` dict is built (~line 2115) and after `_check_cancelled` is defined (~line 2120), add:

```python
    ctx["cancellation_check"] = _check_cancelled
```

- [ ] **Step 3f: Add the CLI arg** in `parse_args`, after the `--workers` argument (~line 2014):

```python
    parser.add_argument(
        "--energy-workers",
        type=int,
        default=None,
        help="Workers for the WSL energy stage (overrides analyze.energy.workers)",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_stage_energy.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_stage_energy.py
git commit -m "feat(energy): stage_energy + pipeline registration (CLI)"
```

---

### Task 5: Mirror the stage in request_models, label, and config

**Files:**
- Modify: `src/playlist/request_models.py`
- Modify: `src/playlist/analyze_library_results.py`
- Modify: `config.example.yaml`
- Test: `tests/unit/test_request_models_energy.py`

**Interfaces:**
- Produces: `"energy"` in `AnalyzeLibraryStage` (Literal) and `ANALYZE_LIBRARY_STAGE_ORDER` (after `"artifacts"`); `"energy": "Energy scan"` in `ANALYZE_LIBRARY_ACTION_LABELS`; `analyze.energy` block in `config.example.yaml`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_request_models_energy.py`:

```python
from src.playlist.request_models import (
    ANALYZE_LIBRARY_STAGE_ORDER,
    LibraryPipelineRequest,
)


def test_energy_in_stage_order_after_artifacts():
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "energy" in order
    assert order.index("energy") == order.index("artifacts") + 1


def test_clean_stages_keeps_energy():
    req = LibraryPipelineRequest(stages=["energy"])
    assert req.stages == ["energy"]


def test_default_run_includes_energy():
    req = LibraryPipelineRequest()
    assert "energy" in req.stages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_request_models_energy.py -q`
Expected: FAIL (`assert 'energy' in [...]`).

- [ ] **Step 3a: Edit `src/playlist/request_models.py`.** In the `AnalyzeLibraryStage = Literal[...]` block (lines 19–32) and the `ANALYZE_LIBRARY_STAGE_ORDER = (...)` tuple (lines 34–47), insert `"energy"` right after `"artifacts"` in BOTH:

```python
AnalyzeLibraryStage = Literal[
    "scan", "genres", "discogs", "lastfm", "sonic", "mert",
    "enrich", "publish", "genre-sim", "artifacts", "energy",
    "genre-embedding", "verify",
]

ANALYZE_LIBRARY_STAGE_ORDER: tuple[AnalyzeLibraryStage, ...] = (
    "scan", "genres", "discogs", "lastfm", "sonic", "mert",
    "enrich", "publish", "genre-sim", "artifacts", "energy",
    "genre-embedding", "verify",
)
```

- [ ] **Step 3b: Edit `src/playlist/analyze_library_results.py`.** Add an `"energy"` entry to `ANALYZE_LIBRARY_ACTION_LABELS` (after `"artifacts"`... the dict is at lines 10–19; insert):

```python
    "artifacts": "Build DS artifacts",
    "energy": "Energy scan",
```

- [ ] **Step 3c: Edit `config.example.yaml`.** Under the existing `analyze:` → `mert:` block (lines 4–8), add a sibling `energy:` block:

```yaml
analyze:
  mert:
    device: cpu        # cpu|cuda
    torch_threads: 0   # 0 = all available (--workers N overrides)
    shard_size: 200    # tracks per resumable shard npz
  energy:
    distro: "Ubuntu-22.04"        # WSL distro hosting the Essentia venv
    python: "/opt/ess/bin/python" # interpreter inside that distro
    models_dir: "/opt/ess/models" # Essentia model dir (preflight checks it)
    workers: 14                   # parallel decode workers (--energy-workers overrides)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_request_models_energy.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/request_models.py src/playlist/analyze_library_results.py config.example.yaml tests/unit/test_request_models_energy.py
git commit -m "feat(energy): mirror energy stage in request_models, label, config"
```

---

### Task 6: GUI stage checkbox + fake-worker compatibility + rebuild

**Files:**
- Modify: `web/src/components/ToolsPanel.tsx`
- Verify: `tests/fixtures/fake_worker.py`

**Interfaces:**
- Consumes: the worker bridges `analyze_library` stage progress generically; the GUI sends the selected `stages` list (already wired). Adding `"energy"` to `ALL_STAGES` renders the checkbox and includes it in the default run.

- [ ] **Step 1: Edit `web/src/components/ToolsPanel.tsx`** — add `"energy"` to `ALL_STAGES` (lines 7–11), after `"artifacts"`:

```tsx
const ALL_STAGES = [
  "scan", "genres", "discogs", "lastfm", "sonic", "mert",
  "enrich", "publish", "genre-sim", "artifacts", "energy",
  "genre-embedding", "verify",
] as const;
```

- [ ] **Step 2: Confirm the fake worker handles `analyze_library` generically.** Inspect `tests/fixtures/fake_worker.py` for the `analyze_library` branch.

Run: `grep -n "analyze_library" tests/fixtures/fake_worker.py`
Expected: an `analyze_library` branch that emits result+done independent of which stages are listed (it does not switch per-stage). If it enumerates stages explicitly, add `"energy"` there; otherwise no change.

- [ ] **Step 3: Rebuild the front-end** (Core Rule 1 — the served GUI runs `web/dist`, not source):

Run: `npm --prefix web run build`
Expected: build succeeds; `grep -rl energy web/dist/assets/*.js` finds the new token.

- [ ] **Step 4: Run the web integration test** (real-worker dispatch unaffected; confirms protocol intact):

Run: `python -m pytest tests/integration/test_web_api.py -q -m "not slow"`
Expected: PASS (no regression).

- [ ] **Step 5: Commit**

```bash
git add web/src/components/ToolsPanel.tsx web/dist
git commit -m "feat(energy): energy stage checkbox in Tools panel (+rebuild dist)"
```

---

### Task 7: Full-suite gates + real-WSL smoke + memory note

**Files:** none (verification) ; **Modify** memory index only.

- [ ] **Step 1: Lint + types**

Run: `ruff check src/analyze/energy_runner.py scripts/analyze_library.py src/playlist/request_models.py src/playlist/analyze_library_results.py && mypy src/analyze/energy_runner.py`
Expected: no errors. (Fix any `E,F`/type issues inline.)

- [ ] **Step 2: Fast suite**

Run: `python -m pytest -q -m "not slow"`
Expected: green (quote real pass/skip counts from the output you see).

- [ ] **Step 3: Real-WSL smoke (manual, requires the live /opt/ess venv).** Confirms the end-to-end path the unit tests mock:

Run: `python scripts/analyze_library.py --stages energy --limit-energy-smoke` — i.e. run the standalone extractor through one small pass:
`wsl -d Ubuntu-22.04 -u root -- bash -c "cd /mnt/c/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3 && /opt/ess/bin/python scripts/extract_energy_sidecar.py --limit 5"`
Expected: a `wrote ... ok=.. total=..` line; `data/artifacts/beat3tower_32k/energy/energy_sidecar.npz` present. (The pipeline `--stages energy` path will do the full pending set; use the direct `--limit 5` for a bounded check.)

- [ ] **Step 4: GUI smoke (manual).** `python tools/serve_web.py --port 8770`; in the Tools panel confirm the **energy** checkbox appears and a dry-run/limited run shows `Stage N/M - Energy scan`.

- [ ] **Step 5: Update memory index.** Append to `MEMORY.md` and to `project_energy_feature_exploration.md` that the energy scan is now an `analyze_library` stage (CLI `--stages energy` + GUI checkbox), default-on, hard-fail if WSL missing; consumption (loader + pace-gate term) remains the next sub-project.

- [ ] **Step 6: Commit**

```bash
git add MEMORY.md   # plus the memory file under the projects dir if tracked
git commit -m "docs(energy): note analyze_library energy stage is live"
```

---

## Self-Review

**Spec coverage:**
- `energy` stage + WSL shell-out → Tasks 2–4. ✓
- `analyze.energy` config block → Task 5. ✓
- Default-on + hard-fail → `stage_energy` (Task 4) + `preflight_wsl` (Task 2). ✓
- Skip-fast like MERT → Task 4 (`pending`/no-artifact). ✓
- Three-place stage mirroring → Tasks 4 (analyze_library), 5 (request_models), 6 (ToolsPanel). ✓
- Fingerprint + units → Task 4 (3c/3d). ✓
- `ctx["cancellation_check"]` additive change → Task 4 (3e). ✓
- Progress to CLI + GUI → `run_energy_scan` logs (Task 3) → CLI; GUI stage-level via existing decision-line parsing (no extra work). ✓
- Error-handling table (WSL missing raise / non-zero raise / missing-artifact skip / empty track recorded / cancel terminates) → Tasks 2–4. ✓
- Tests (stage_energy mocked, request_models order, fake-worker) → Tasks 2–6. ✓
- Out-of-scope (loader + pace-gate term) → not implemented, by design. ✓

**Placeholder scan:** no TBD/TODO; every code step shows full code; commands have expected output.

**Type consistency:** `EnergyConfig`, `load_energy_config`, `pending_energy`, `preflight_wsl(runner=)`, `run_energy_scan(popen=, cancellation_check=)` names/signatures match across Tasks 2–4; the `_energy_pending`/`_energy_preflight`/`_energy_run` seams in `analyze_library.py` are the exact names the Task-4 tests monkeypatch.

> Note for the implementer: `scripts/extract_energy_sidecar.py` (Task 1) already exists, untracked, in the **main** checkout and may be running a full library scan there; do not disturb that process. The worktree copy is created fresh from the content above.
