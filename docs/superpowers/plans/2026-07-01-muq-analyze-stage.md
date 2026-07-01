# MuQ Analyze Stage (SP-A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class `muq` analyze stage so a rebuild under `variant=muq` (re)produces `X_sonic_muq` end-to-end, making MuQ self-sufficient in the pipeline.

**Architecture:** A new extraction module (`src/analyze/muq_runner.py`, productionized from `scripts/research/embed_muq_full.py`) + a `stage_muq(ctx)` mirroring `stage_mert`, gated so only the *active* sonic variant's extraction runs (Approach 1: `stage_mert` no-ops under `variant=muq` and vice-versa). The auto-fold is already wired (`54d682c`).

**Tech Stack:** Python 3.11, numpy, librosa, torch, the `muq` package (`MuQ-MuLan-large`), pytest.

## Global Constraints

- **Additive + the mert gate only.** Do NOT touch the `sonic`/tower stage or remove any MERT/tower code — that's SP-B (deferred). Spec: `docs/superpowers/specs/2026-07-01-muq-analyze-stage-design.md`.
- **Data safety:** `muq_sidecar.npz` is ~16–29h CPU — **timestamped backup before any overwrite, never delete**. Audio files are read-only. No DB writes.
- **Variant gate (Approach 1):** the active variant is `artifacts.sonic_variant_override` (via the existing `_mert_fold_settings(config_path) -> (enabled, active_variant)`). In a default run only the active variant's extraction runs; the other no-ops with a loud log. Rollback = set `variant: mert`.
- **1-window middle-10s** extraction (sr=24000, offset `max(0, mid-5)`, duration 10) — the validated choice; do not add 3-window.
- Run pytest DIRECTLY, never piped (piping has hung sessions); bound with the tool timeout. TDD, frequent commits.

---

### Task 1: `muq_runner.py` — extraction module (stub-testable)

**Files:**
- Create: `src/analyze/muq_runner.py`
- Test: `tests/unit/test_muq_runner.py`

**Interfaces:**
- Produces: `sidecar_ids(sidecar_path) -> set[str]`; `pending_muq(sidecar_path, universe_ids) -> tuple[list[str], int]`; `build_muq_embedder(device="cpu", torch_threads=0) -> Callable[[str], np.ndarray]`; `run_muq_extraction(items, embed_fn, sidecar_path, *, backup_stamp=None, save_every=500) -> dict` (keys `ok`, `failed`, `fails`). Module const `MODEL_NAME = "OpenMuQ/MuQ-MuLan-large"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_muq_runner.py
import numpy as np
from src.analyze.muq_runner import (
    MODEL_NAME, sidecar_ids, pending_muq, run_muq_extraction,
)

def _write_sidecar(p, ids):
    np.savez(str(p), track_ids=np.array(ids, dtype=object),
             embeddings=np.zeros((len(ids), 4), np.float32), model=MODEL_NAME)

def test_pending_is_universe_minus_sidecar(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    _write_sidecar(sc, ["a", "b"])
    assert sidecar_ids(sc) == {"a", "b"}
    pend, done = pending_muq(sc, ["a", "b", "c", "d"])
    assert pend == ["c", "d"] and done == 2

def test_pending_all_when_no_sidecar(tmp_path):
    pend, done = pending_muq(tmp_path / "absent.npz", ["a", "b"])
    assert pend == ["a", "b"] and done == 0

def test_run_extraction_appends_backs_up_and_is_resumable(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    _write_sidecar(sc, ["a"])                      # pre-existing vector
    stub = lambda path: np.ones(4, np.float32)    # deterministic, no model
    res = run_muq_extraction([("b", "/x.flac"), ("c", None)], stub, sc, backup_stamp="20260701_000000")
    assert res["ok"] == 1 and res["failed"] == 1  # b embedded, c skipped (no path)
    assert res["fails"] == [("c", "no_path")]
    assert sidecar_ids(sc) == {"a", "b"}           # additive — 'a' preserved
    assert (tmp_path / "muq_sidecar.bak_20260701_000000.npz").exists()  # backup written

def test_run_extraction_survives_a_bad_file(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    def stub(path):
        if path == "/bad": raise RuntimeError("decode fail")
        return np.ones(4, np.float32)
    res = run_muq_extraction([("g", "/good"), ("b", "/bad")], stub, sc)
    assert res["ok"] == 1 and res["failed"] == 1 and sidecar_ids(sc) == {"g"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest -q -x tests/unit/test_muq_runner.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analyze.muq_runner'`.

- [ ] **Step 3: Write the module**

```python
# src/analyze/muq_runner.py
"""MuQ-MuLan sonic embedding extraction for the analyze `muq` stage.

Incremental + resumable: embeds only tracks lacking a MuQ vector into muq_sidecar.npz
(single npz, atomic save). Backs up the existing sidecar before the first overwrite —
it is a ~16-29h CPU artifact, treated like the irreplaceable MERT data. Productionized
from scripts/research/embed_muq_full.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = "OpenMuQ/MuQ-MuLan-large"
SAVE_EVERY = 500


def sidecar_ids(sidecar_path) -> set:
    p = Path(sidecar_path)
    if not p.exists():
        return set()
    with np.load(str(p), allow_pickle=True) as z:
        return {str(t) for t in z["track_ids"]}


def pending_muq(sidecar_path, universe_ids: Sequence[str]) -> Tuple[List[str], int]:
    have = sidecar_ids(sidecar_path)
    return [t for t in universe_ids if t not in have], len(have)


def _load_existing(sidecar_path) -> Dict[str, np.ndarray]:
    p = Path(sidecar_path)
    if not p.exists():
        return {}
    out: Dict[str, np.ndarray] = {}
    with np.load(str(p), allow_pickle=True) as z:
        for tid, emb in zip(z["track_ids"], z["embeddings"]):
            out[str(tid)] = np.asarray(emb, dtype=np.float32)
    return out


def _atomic_save(sidecar_path, done: Dict[str, np.ndarray]) -> None:
    p = Path(sidecar_path)
    tids = list(done.keys())
    embs = np.stack([done[t] for t in tids]).astype(np.float32)
    tmp = p.with_name(p.stem + ".tmp.npz")   # must end in .npz (savez appends otherwise)
    np.savez(str(tmp), track_ids=np.array(tids, dtype=object), embeddings=embs, model=MODEL_NAME)
    tmp.replace(p)


def _backup(sidecar_path, stamp: str) -> Optional[Path]:
    p = Path(sidecar_path)
    if not p.exists():
        return None
    bak = p.with_name(f"{p.stem}.bak_{stamp}.npz")
    shutil.copy2(str(p), str(bak))
    return bak


def build_muq_embedder(device: str = "cpu", torch_threads: int = 0) -> Callable[[str], np.ndarray]:
    """Load MuQ-MuLan once; return embed(path) -> unit-norm float32 vector (middle 10s)."""
    import librosa
    import torch
    from muq import MuQMuLan
    if torch_threads:
        torch.set_num_threads(int(torch_threads))
    model = MuQMuLan.from_pretrained(MODEL_NAME).to(device).eval()

    def embed(path: str) -> np.ndarray:
        d = librosa.get_duration(path=path)
        y, _ = librosa.load(path, sr=24000, mono=True, offset=max(0.0, d * 0.5 - 5), duration=10.0)
        with torch.no_grad():
            wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            v = model(wavs=wav)[0].detach().cpu().numpy()
        return (v / np.linalg.norm(v)).astype(np.float32)

    return embed


def run_muq_extraction(
    items: Sequence[Tuple[str, Optional[str]]],
    embed_fn: Callable[[str], np.ndarray],
    sidecar_path,
    *,
    backup_stamp: Optional[str] = None,
    save_every: int = SAVE_EVERY,
) -> Dict[str, object]:
    """Embed each (track_id, path); append into the sidecar (atomic, resumable). Backs up
    the existing sidecar once (when backup_stamp given) before the first write. A bad file
    is logged and skipped, never fatal. Returns {ok, failed, fails}."""
    done = _load_existing(sidecar_path)
    if backup_stamp is not None:
        _backup(sidecar_path, backup_stamp)
    ok = 0
    fails: List[Tuple[str, str]] = []
    for k, (tid, path) in enumerate(items, 1):
        if not path:
            fails.append((tid, "no_path")); continue
        try:
            done[tid] = np.asarray(embed_fn(path), dtype=np.float32); ok += 1
        except Exception as exc:  # noqa: BLE001 — one bad file must not kill the scan
            fails.append((tid, type(exc).__name__))
        if k % save_every == 0:
            _atomic_save(sidecar_path, done)
    _atomic_save(sidecar_path, done)
    return {"ok": ok, "failed": len(fails), "fails": fails}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest -q -x tests/unit/test_muq_runner.py`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/analyze/muq_runner.py tests/unit/test_muq_runner.py
git commit -m "feat(muq): muq_runner — incremental resumable MuQ sidecar extraction"
```

---

### Task 2: `stage_muq` + the variant gate (in `analyze_library.py`)

**Files:**
- Modify: `scripts/analyze_library.py` (new `_variant_gate` helper; new `stage_muq`; gate `stage_mert`; register in `STAGE_FUNCS`; add `muq` fingerprint branch)
- Test: `tests/unit/test_variant_gate.py`

**Interfaces:**
- Consumes: Task 1's `build_muq_embedder`, `pending_muq`, `run_muq_extraction`; existing `_mert_fold_settings`, `load_paths`.
- Produces: `_variant_gate(config_path, stage_variant) -> Optional[str]`; `stage_muq(ctx) -> dict`.

- [ ] **Step 1: Write the failing test** (the gate is the unit-testable seam)

```python
# tests/unit/test_variant_gate.py
import yaml
from scripts.analyze_library import _variant_gate

def _cfg(tmp_path, variant):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"artifacts": {"sonic_variant_override": variant}}), encoding="utf-8")
    return str(p)

def test_active_variant_runs_inactive_skips(tmp_path):
    muq_cfg = _cfg(tmp_path, "muq")
    assert _variant_gate(muq_cfg, "muq") is None                # muq active -> muq runs
    assert _variant_gate(muq_cfg, "mert") is not None           # muq active -> mert skips
    mert_cfg = _cfg(tmp_path, "mert")
    assert _variant_gate(mert_cfg, "mert") is None
    assert _variant_gate(mert_cfg, "muq") is not None

def test_default_variant_is_mert(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text("{}", encoding="utf-8")   # no override -> 'mert'
    assert _variant_gate(str(p), "mert") is None
    assert _variant_gate(str(p), "muq") is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q -x tests/unit/test_variant_gate.py`
Expected: FAIL — `cannot import name '_variant_gate'`.

- [ ] **Step 3a: Add `_variant_gate`** near `_mert_fold_settings` (~line 150) in `analyze_library.py`:

```python
def _variant_gate(config_path: str, stage_variant: str) -> Optional[str]:
    """Approach-1 gate: return a skip-reason if the ACTIVE sonic variant
    (artifacts.sonic_variant_override) is not this stage's variant, else None. Only the
    active variant's extraction runs in a default rebuild; the other is reached by
    switching the variant."""
    _, active = _mert_fold_settings(config_path)
    if active != stage_variant:
        return f"active sonic variant is {active!r}; skipping {stage_variant} extraction"
    return None
```

- [ ] **Step 3b: Gate `stage_mert`** — add at the very top of `stage_mert(ctx)` (right after the docstring, before the `import yaml`):

```python
    _gate = _variant_gate(ctx["config_path"], "mert")
    if _gate:
        logger.info("stage_mert: %s", _gate)
        return {"skipped": True, "pending": 0, "reason": _gate}
```

- [ ] **Step 3c: Add `stage_muq`** immediately after `stage_mert`:

```python
def stage_muq(ctx: Dict) -> Dict:
    """Extract MuQ-MuLan sonic embeddings into muq_sidecar.npz (incremental, resumable).
    No-ops when muq is not the active sonic variant (Approach-1 variant gate)."""
    _gate = _variant_gate(ctx["config_path"], "muq")
    if _gate:
        logger.info("stage_muq: %s", _gate)
        return {"skipped": True, "pending": 0, "reason": _gate}

    import yaml
    from datetime import datetime
    from scripts.extract_mert_sidecar import load_paths
    from src.analyze.muq_runner import build_muq_embedder, pending_muq, run_muq_extraction

    args = ctx["args"]
    force = bool(args.force)
    limit: Optional[int] = args.limit if args.limit else None
    out_dir = Path(ctx["out_dir"])
    sidecar_path = out_dir / "muq_sidecar.npz"

    device, torch_threads = "cpu", 0
    try:
        _cfg = yaml.safe_load(open(ctx["config_path"], "r", encoding="utf-8")) or {}
        _mcfg = (_cfg.get("analyze") or {}).get("muq") or {}
        device = str(_mcfg.get("device", "cpu"))
        torch_threads = int(_mcfg.get("torch_threads", 0))
    except Exception:
        pass

    try:
        db_paths = load_paths(ctx["db_path"])
    except Exception as exc:
        logger.warning("stage_muq: cannot load file paths from db: %s", exc)
        return {"skipped": True, "pending": 0, "reason": str(exc)}

    universe = list(db_paths.keys())
    if force:
        pending = universe
    else:
        pending, _done = pending_muq(sidecar_path, universe)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        logger.info("stage_muq: nothing pending (sidecar complete); skipping")
        return {"skipped": True, "pending": 0}

    logger.info("stage_muq: %d track(s) pending (device=%s); loading MuQ-MuLan "
                "(cold cache can take a minute)...", len(pending), device)
    embed_fn = build_muq_embedder(device, torch_threads)
    items = [(t, db_paths.get(t)) for t in pending]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_muq_extraction(items, embed_fn, sidecar_path, backup_stamp=stamp)
    logger.info("stage_muq: embedded ok=%d failed=%d -> %s",
                result["ok"], result["failed"], sidecar_path)
    return {"skipped": False, "pending": len(pending), **result}
```

- [ ] **Step 3d: Register** in `STAGE_FUNCS` (add after the `"mert": stage_mert,` line):

```python
    "muq": stage_muq,
```

- [ ] **Step 3e: Add the `muq` fingerprint branch** — in the stage-fingerprint dispatcher, immediately after the `if stage == "mert":` block (~line 410), add the identical branch for `muq` (a newly-scanned track must re-trigger it):

```python
    if stage == "muq":
        try:
            track_ids = sorted(
                str(r[0]) for r in conn.execute(
                    "SELECT track_id FROM tracks "
                    "WHERE file_path IS NOT NULL AND file_path != ''"
                ).fetchall()
            )
        except Exception:
            track_ids = []
        return _hash_obj({"stage": stage, "track_ids": track_ids})
```

- [ ] **Step 4: Run the gate test + confirm the module imports**

Run: `python -m pytest -q -x tests/unit/test_variant_gate.py` then `python -c "import scripts.analyze_library as a; assert 'muq' in a.STAGE_FUNCS and a.STAGE_FUNCS['muq'] is a.stage_muq; print('registered')"`
Expected: gate test PASS (2 passed); prints `registered`.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_library.py tests/unit/test_variant_gate.py
git commit -m "feat(muq): stage_muq + Approach-1 variant gate (mert no-ops under muq)"
```

---

### Task 3: Register the stage everywhere (order, type, GUI, config, dep)

**Files:**
- Modify: `src/playlist/request_models.py` (`AnalyzeLibraryStage` Literal + `ANALYZE_LIBRARY_STAGE_ORDER`)
- Modify: `web/src/components/ToolsPanel.tsx` (`ALL_STAGES`)
- Modify: `config.example.yaml` (`analyze.muq` block)
- Modify: `pyproject.toml` (`[project.optional-dependencies].muq`)
- Test: `tests/unit/test_muq_stage_registration.py`

**Interfaces:**
- Consumes: Task 2's `stage_muq` registered in `STAGE_FUNCS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_muq_stage_registration.py
from src.playlist.request_models import ANALYZE_LIBRARY_STAGE_ORDER, LibraryPipelineRequest

def test_muq_is_in_order_right_after_mert():
    order = list(ANALYZE_LIBRARY_STAGE_ORDER)
    assert "muq" in order
    assert order.index("muq") == order.index("mert") + 1

def test_request_accepts_muq_stage():
    req = LibraryPipelineRequest(stages=["muq"])
    assert "muq" in req.stages
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest -q -x tests/unit/test_muq_stage_registration.py`
Expected: FAIL — `"muq" in order` assertion / `LibraryPipelineRequest` rejects `"muq"`.

- [ ] **Step 3a: `request_models.py`** — add `"muq"` to the `AnalyzeLibraryStage` Literal (right after `"mert"`, ~line 25) AND to `ANALYZE_LIBRARY_STAGE_ORDER` (right after `"mert",`, ~line 51):

```python
    "mert",
    "muq",
```
(both places; the Literal list keeps `"enrich"` etc.; the ORDER tuple does not.)

- [ ] **Step 3b: `web/src/components/ToolsPanel.tsx`** — add `"muq"` after `"mert"` in `ALL_STAGES`:

```javascript
  "scan", "genres", "discogs", "lastfm", "sonic", "mert", "muq",
  "adjudicate", "apply", "publish", "genre-sim", "artifacts", "energy",
  "popularity", "genre-embedding", "verify",
```

- [ ] **Step 3c: `config.example.yaml`** — add a `muq` block under `analyze:` mirroring `mert`:

```yaml
analyze:
  mert:
    device: cpu
    torch_threads: 0
    shard_size: 200
  muq:
    device: cpu        # cpu|cuda — MuQ-MuLan extraction device
    torch_threads: 0   # 0 = all available
```
(If `analyze.mert` already exists in the file, add only the `muq:` sub-block beside it.)

- [ ] **Step 3d: `pyproject.toml`** — add an optional extra:

```toml
[project.optional-dependencies]
muq = ["muq", "torch"]
```
(Add to the existing `[project.optional-dependencies]` table; do not duplicate the header.)

- [ ] **Step 4: Run tests + rebuild the GUI bundle**

Run: `python -m pytest -q -x tests/unit/test_muq_stage_registration.py tests/unit/test_pipeline_smoke_golden.py` then `npm --prefix web run build`
Expected: tests PASS; `tsc -b && vite build` clean (`web/dist` is gitignored — restart `serve_web.py` to serve it).

- [ ] **Step 5: Commit**

```bash
git add src/playlist/request_models.py web/src/components/ToolsPanel.tsx config.example.yaml pyproject.toml tests/unit/test_muq_stage_registration.py
git commit -m "feat(muq): register the muq stage (order, type, GUI list, config, extra)"
```

---

### Task 4: Validation — prove a real incremental rebuild (manual, non-CI)

**Files:** none (validation run). Requires the live artifact + DB + `muq` package installed.

- [ ] **Step 1:** Confirm the gate + skip path with the live config (variant=muq) without embedding — nothing pending on the full library:

Run (from repo root): `python scripts/analyze_library.py --stages muq --limit 0` (or a dry probe)
Expected log: `stage_muq: nothing pending (sidecar complete); skipping` — proves the gate passes (muq active) and pending-detection sees the existing sidecar.

- [ ] **Step 2:** Prove the mert gate: with `variant=muq`, a default-order run logs `stage_mert: active sonic variant is 'muq'; skipping mert extraction` and does NOT load the MERT model.

- [ ] **Step 3:** Real incremental extraction — pick 1–2 tracks NOT yet in the sidecar (or temporarily point at a scratch sidecar copy) and run `--stages muq --limit 2`. Confirm: a `muq_sidecar.bak_<stamp>.npz` backup appears, the sidecar gains exactly those ids (`sidecar_ids` before/after), the `artifacts` stage's `fold_muq` refreshes `X_sonic_muq`, `verify` passes, and a generation runs on the fresh vectors.

- [ ] **Step 4:** Report the run (ids added, backup path, fold/verify/generation result). This is the acceptance gate for SP-B to proceed.

---

## Self-Review

- **Spec coverage:** muq_runner extraction (Task 1) ✓; stage_muq + variant gate (Task 2) ✓; registration in STAGE_FUNCS/ORDER/Literal/ToolsPanel/fingerprint (Tasks 2–3) ✓; fold already wired (noted) ✓; config `analyze.muq` + pyproject extra (Task 3) ✓; data-safety backup (Task 1 `_backup`, Task 2 stamp) ✓; testing seams + real validation (all tasks + Task 4) ✓. Scope guard (no sonic/tower/MERT removal) honored — SP-A is additive.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `pending_muq -> (list, int)`, `run_muq_extraction -> {ok, failed, fails}`, `_variant_gate -> Optional[str]`, `build_muq_embedder -> Callable[[str], np.ndarray]` — consistent across Tasks 1→2. `MODEL_NAME` shared. `stage_muq` registered as `"muq"` everywhere (STAGE_FUNCS, ORDER, Literal, ToolsPanel, fingerprint).
