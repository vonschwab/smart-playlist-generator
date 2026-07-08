# Instrumental Lean Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Instrumental" GUI checkbox that applies a continuous, never-failing soft penalty (proportional to a per-track `voice_prob`) to demote vocal-classified bridge candidates — a guard against poetry/spoken-word over ambient. Not a hard filter.

**Architecture:** A new isolated Essentia sidecar (`instrumental_sidecar.npz`) supplies a per-track `voice_prob ∈ [0,1]`. A 1-D `voice_prob` array is threaded through the generation pipeline exactly like `onset_rate`/`energy_matrix`, and an additive penalty `= weight × voice_prob` rides the existing beam `_pace_penalty` accumulator and the `candidate_pool` `seed_sim_all` demotion. Enablement flows through the policy layer (`UIStateModel → derive_runtime_config → overrides → self.config`) — the only channel the beam/pool actually reads. Seeds/piers are structurally exempt; missing data warns loudly and the run proceeds.

**Tech Stack:** Python 3.11, NumPy, Essentia (TensorFlow models, run under WSL), FastAPI + Pydantic (`src/playlist_web`), React + TypeScript + Vite + Tailwind (`web/`), pytest.

## Global Constraints

- **Soft lean only.** Continuous on `voice_prob`; never a hard filter, never hard-fails generation (Layer 4 #19, `feedback_never_fail_three_axes`). A thin instrumental library yields a best-effort playlist, never an error.
- **Seeds/piers exempt.** Only bridge-interior candidates are penalized. Never demote a user-seeded/pier track.
- **Isolation.** Write to a **separate** `<artifact>/instrumental/instrumental_sidecar.npz`. Do **not** re-run or `--force` the energy sidecar; do **not** modify `scripts/extract_energy_sidecar.py`. Zero regression risk to the live pace axis.
- **Data-write = canonical-only.** The 42k Essentia extraction is a data write; `satellite_data_write_guard` blocks it in this satellite (PG3_SAT1). All *code + tests* are authored here; the actual sidecar build (Checkpoint B) runs in the canonical checkout.
- **Validate-first.** A `--limit` smoke on a curated label set (Black Moth vocoder + poetry-over-ambient + pure-instrumental) MUST pass and yield a real ETA before any full-library pass.
- **Route through policy or it goes inert.** Enablement MUST reach the beam/pool via `derive_runtime_config` → `overrides` → `self.config`. A flag that bypasses policy is silently inert (`project_slider_calibration`).
- **Never guess the model's class-column order.** Read `[instrumental, voice]` vs reverse from the model's companion `.json`. `voice_prob` must be the *voice* column.
- **Aggregation across frames = mean** of the per-frame voice softmax (Open item #4, decided).
- **Tests mirror production.** The generation test uses the `gui_fidelity` multi-pier harness (`generate_like_gui`), never hand-built `overrides` or single-seed topology (`playlist-testing` skill).
- **Git (canonical + satellite).** `git fetch origin` before branching. Commit explicit paths only: `git add <paths>` then `git commit --only -- <paths>`; never `git add -A`/`-u`/`.` or a bare `git commit`.
- **Config field naming.** New pier-bridge config keys: `instrumental_penalty_weight` (float, static in `config.yaml`) and `instrumental_enabled` (bool, per-request, set by policy). Both live under `playlists.ds_pipeline.pier_bridge`.

---

## File Structure

**Create:**
- `scripts/ess_sidecar_common.py` — pure, parameterized scaffold helpers shared by the new extractor (win→WSL path, checkpoint read/write, backup+merge). Energy extractor keeps its inline copies (isolation); de-dup deferred to housekeeping.
- `scripts/extract_instrumental_sidecar.py` — the isolated extractor (one `TensorflowPredict2D` voice/instrumental head on the shared `msd-musicnn` embedding).
- `src/playlist/instrumental_loader.py` — `load_voice_prob(...)` read path.
- `tests/unit/test_ess_sidecar_common.py` — scaffold unit tests.
- `tests/unit/test_instrumental_extractor.py` — class-column-order parsing test.
- `tests/unit/test_instrumental_loader.py` — loader alignment + missing-sidecar behavior.
- `tests/unit/test_instrumental_penalty.py` — penalty helper unit test.
- `tests/integration/test_gui_fidelity_instrumental.py` — generation-fidelity multi-pier test.

**Modify:**
- `src/analyze/energy_runner.py` — extend `preflight_wsl`.
- `src/playlist/pier_bridge/pace_gate.py` — add `compute_instrumental_penalty`.
- `src/playlist/pier_bridge/config.py` — add two `PierBridgeConfig` fields.
- `src/playlist/pipeline/pier_bridge_overrides.py` — parse the two new fields.
- `src/playlist/pier_bridge/beam.py` — add `voice_prob` param + apply penalty into `_pace_penalty`.
- `src/playlist/pier_bridge_builder.py` — thread `voice_prob` through `build_pier_bridge_playlist`.
- `src/playlist/pipeline/core.py` — load sidecar when enabled (warn-to-None), pass `voice_prob`, compute confession count.
- `src/playlist/candidate_pool.py` — `seed_sim_all` demotion block.
- `src/playlist_gui/ui_state.py` — `instrumental` field on `UIStateModel`.
- `src/playlist_web/app.py` — read `body.instrumental` into `UIStateModel`.
- `src/playlist_web/schemas.py` — `instrumental` field on `GenerateRequestBody`.
- `src/playlist_gui/policy.py` — register key + set `instrumental_enabled` override.
- `src/playlist_gui/receipt.py` — two confession notes.
- `config.yaml` + `config.example.yaml` — `instrumental_penalty_weight`.
- `web/src/lib/types.ts` — `instrumental?: boolean` on `GenerateRequestBody`.
- `web/src/components/GenerateControls.tsx` — checkbox + state + payload.
- `tests/test_gui_fidelity.py` — fast policy-routing guard.
- `tests/unit/test_receipt_compose.py` — confession-note cases.

**Do NOT touch:** `scripts/extract_energy_sidecar.py`, `src/playlist/energy_loader.py`, `web/src/components/AdvancedPanel.tsx`, `src/playlist/request_models.py` (Track A skipped — see plan header rationale).

---

## Task 1: Shared sidecar scaffold module

**Files:**
- Create: `scripts/ess_sidecar_common.py`
- Test: `tests/unit/test_ess_sidecar_common.py`

**Interfaces:**
- Produces:
  - `win_to_wsl_path(path: str) -> str` — `C:\X\Y` → `/mnt/c/X/Y`.
  - `read_checkpoint_ids(ckpt_path: str) -> set[str]` — resumable set of processed `track_id`s from an append-only JSONL.
  - `append_checkpoint(fh, record: dict) -> None` — write one JSON line + flush.
  - `merge_sidecar_npz(sidecar_path: str, ckpt_path: str, columns: dict[str, str]) -> str` — read the checkpoint JSONL, assemble aligned arrays for each requested column, back up any existing sidecar with a timestamp, and write `np.savez_compressed`. Returns the sidecar path. `columns` maps output-array-name → checkpoint-record-key; missing/None values become `np.nan` (float columns) so alignment is preserved.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_ess_sidecar_common.py
import json
import numpy as np
from scripts.ess_sidecar_common import (
    win_to_wsl_path,
    read_checkpoint_ids,
    merge_sidecar_npz,
)


def test_win_to_wsl_path():
    assert win_to_wsl_path(r"C:\Users\Dylan\Desktop\x.py") == "/mnt/c/Users/Dylan/Desktop/x.py"


def test_read_checkpoint_ids_skips_blank_and_bad_lines(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"track_id": "a", "voice_prob": 0.9}) + "\n"
        + "\n"
        + "not-json\n"
        + json.dumps({"track_id": "b", "voice_prob": 0.1}) + "\n",
        encoding="utf-8",
    )
    assert read_checkpoint_ids(str(ckpt)) == {"a", "b"}


def test_merge_sidecar_npz_aligns_and_nan_fills(tmp_path):
    ckpt = tmp_path / "checkpoint.jsonl"
    ckpt.write_text(
        json.dumps({"track_id": "a", "voice_prob": 0.9}) + "\n"
        + json.dumps({"track_id": "b", "missing": True}) + "\n",
        encoding="utf-8",
    )
    sidecar = tmp_path / "instrumental_sidecar.npz"
    merge_sidecar_npz(str(sidecar), str(ckpt), columns={"voice_prob": "voice_prob"})
    data = np.load(str(sidecar), allow_pickle=True)
    ids = list(data["track_ids"])
    vp = data["voice_prob"]
    by_id = {t: vp[i] for i, t in enumerate(ids)}
    assert abs(float(by_id["a"]) - 0.9) < 1e-6
    assert np.isnan(float(by_id["b"]))  # missing track -> NaN, still present in alignment
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_ess_sidecar_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.ess_sidecar_common'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ess_sidecar_common.py
"""Pure, side-effect-free scaffold shared by Essentia sidecar extractors.

Intentionally does NOT import essentia — safe to import under plain pytest.
Mirrors (does not modify) the inline helpers in scripts/extract_energy_sidecar.py;
de-duplicating the energy script is deferred housekeeping (isolation of the live
pace axis takes priority).
"""
from __future__ import annotations

import json
import os
import time
from typing import IO, Iterable

import numpy as np


def win_to_wsl_path(path: str) -> str:
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def read_checkpoint_ids(ckpt_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(ckpt_path):
        return done
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["track_id"])
            except (ValueError, KeyError):
                continue
    return done


def append_checkpoint(fh: IO[str], record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def _iter_records(ckpt_path: str) -> Iterable[dict]:
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def merge_sidecar_npz(sidecar_path: str, ckpt_path: str, columns: dict[str, str]) -> str:
    """Assemble aligned float32 columns from the checkpoint JSONL and write the sidecar.

    columns: {output_array_name: checkpoint_record_key}. Records lacking a key
    (missing/errored tracks) get np.nan so the track_id row is still present.
    """
    tids: list[str] = []
    cols: dict[str, list[float]] = {name: [] for name in columns}
    seen: set[str] = set()
    for rec in _iter_records(ckpt_path):
        tid = rec.get("track_id")
        if tid is None or tid in seen:
            continue
        seen.add(tid)
        tids.append(str(tid))
        for name, key in columns.items():
            val = rec.get(key)
            cols[name].append(float(val) if isinstance(val, (int, float)) else float("nan"))

    if os.path.exists(sidecar_path):
        bak = sidecar_path + "." + time.strftime("%Y%m%d_%H%M%S") + ".bak"
        os.rename(sidecar_path, bak)
        print(f"backed up existing sidecar -> {bak}")

    arrays = {name: np.asarray(vals, dtype=np.float32) for name, vals in cols.items()}
    np.savez_compressed(
        sidecar_path,
        track_ids=np.array(tids, dtype=object),
        **arrays,
    )
    return sidecar_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_ess_sidecar_common.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/ess_sidecar_common.py tests/unit/test_ess_sidecar_common.py
git commit --only -m "feat(instrumental): shared Essentia sidecar scaffold helpers" -- scripts/ess_sidecar_common.py tests/unit/test_ess_sidecar_common.py
```

---

## Task 2: Instrumental extractor script

**Files:**
- Create: `scripts/extract_instrumental_sidecar.py`
- Test: `tests/unit/test_instrumental_extractor.py`

**Interfaces:**
- Consumes: `scripts.ess_sidecar_common.{win_to_wsl_path, read_checkpoint_ids, append_checkpoint, merge_sidecar_npz}`.
- Produces:
  - `voice_column_index(model_json_path: str) -> int` — parse the model's companion `.json` and return the column index whose class label is voice (raises `ValueError` if the two classes can't be disambiguated). Runtime guard against guessing column order.
  - CLI: `python scripts/extract_instrumental_sidecar.py [--workers N] [--limit N] [--merge-only] [--force]`.
  - Output: `<artifact>/instrumental/instrumental_sidecar.npz` with keys `track_ids` (object) and `voice_prob` (float32; NaN for missing/errored), plus a `model` string.

**Note on the model file:** the spec names `voice_instrumental-musicnn-msd-2.pb`. Confirm the exact filename, TF output node (`model/Softmax` for the 2-class head, matching the danceability head), and the companion `.json` name from the Essentia model zoo at implementation time. The class-label order is resolved at runtime by `voice_column_index` — never hardcoded.

- [ ] **Step 1: Write the failing test** (the class-column parser is the only pure-Python unit; the full extraction runs under WSL at Checkpoint B)

```python
# tests/unit/test_instrumental_extractor.py
import json
import pytest
from scripts.extract_instrumental_sidecar import voice_column_index


def _write_json(tmp_path, classes):
    p = tmp_path / "voice_instrumental.json"
    p.write_text(json.dumps({"classes": classes}), encoding="utf-8")
    return str(p)


def test_voice_column_index_voice_second(tmp_path):
    # Essentia zoo order is typically ["instrumental", "voice"]
    assert voice_column_index(_write_json(tmp_path, ["instrumental", "voice"])) == 1


def test_voice_column_index_voice_first(tmp_path):
    assert voice_column_index(_write_json(tmp_path, ["voice", "instrumental"])) == 0


def test_voice_column_index_ambiguous_raises(tmp_path):
    with pytest.raises(ValueError):
        voice_column_index(_write_json(tmp_path, ["classA", "classB"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_instrumental_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.extract_instrumental_sidecar'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/extract_instrumental_sidecar.py
"""Isolated Essentia voice/instrumental sidecar extractor.

Runs under WSL (invoked by src/analyze/energy_runner-style plumbing or directly).
Shares the msd-musicnn embedding with the energy pass but writes a SEPARATE
sidecar under <artifact>/instrumental/ — it never touches the energy/pace path.

voice_prob = mean over frames of the *voice* softmax column (column order read
from the model .json, never guessed).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from scripts.ess_sidecar_common import (
    append_checkpoint,
    merge_sidecar_npz,
    read_checkpoint_ids,
)

# --- paths (mirror extract_energy_sidecar.py constants; ART resolved the same way) ---
ART = os.environ.get("PLAYLIST_ARTIFACT_DIR", "data/artifacts/beat3tower_32k")
MODELS = os.environ.get("ESS_MODELS", "/opt/ess/models")
OUTDIR = os.path.join(ART, "instrumental")
CKPT = os.path.join(OUTDIR, "checkpoint.jsonl")
SIDECAR = os.path.join(OUTDIR, "instrumental_sidecar.npz")

EMB_PB = f"{MODELS}/msd-musicnn-1.pb"
VI_PB = f"{MODELS}/voice_instrumental-musicnn-msd-2.pb"     # confirm exact name at impl time
VI_JSON = f"{MODELS}/voice_instrumental-musicnn-msd-2.json"  # confirm exact name at impl time

_emb = None
_vi = None
_VOICE_COL = 1  # overwritten in _init() from the model .json


def voice_column_index(model_json_path: str) -> int:
    """Return the softmax column index for the *voice* class, from the model .json."""
    with open(model_json_path, encoding="utf-8") as f:
        meta = json.load(f)
    classes = [str(c).strip().lower() for c in meta.get("classes", [])]
    if len(classes) != 2:
        raise ValueError(f"expected 2 classes, got {classes!r}")
    for i, c in enumerate(classes):
        if "voc" in c or c == "voice" or "vocal" in c:
            return i
    raise ValueError(f"cannot identify a voice column in classes {classes!r}")


def _init() -> None:
    import essentia.standard as es

    global _emb, _vi, _VOICE_COL
    _emb = es.TensorflowPredictMusiCNN(graphFilename=EMB_PB, output="model/dense/BiasAdd")
    _vi = es.TensorflowPredict2D(graphFilename=VI_PB, output="model/Softmax")
    _VOICE_COL = voice_column_index(VI_JSON)


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
        vi = _vi(emb)  # (frames, 2)
        voice_prob = float(np.mean(vi[:, _VOICE_COL]))
        return {"track_id": tid, "voice_prob": round(voice_prob, 4), "frames": int(emb.shape[0])}
    except Exception as exc:  # never crash the pool on one track
        return {"track_id": tid, "error": str(exc)[:200]}


def _load_todo(force: bool, limit: int) -> list[tuple[str, str | None]]:
    """Track_id + file_path list from metadata.db, minus already-checkpointed ids."""
    import sqlite3

    from src.config_loader import resolve_database_path, load_config

    db_path = resolve_database_path(load_config("config.yaml"))
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute("SELECT track_id, file_path FROM tracks").fetchall()
    con.close()
    done = set() if force else read_checkpoint_ids(CKPT)
    todo = [(str(t), p) for (t, p) in rows if str(t) not in done]
    if limit > 0:
        todo = todo[:limit]
    return todo


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="process at most N (smoke test)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-process all, ignoring checkpoint")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.merge_only:
        print(f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'})}")
        return

    import multiprocessing as mp

    todo = _load_todo(args.force, args.limit)
    total = len(todo)
    ok = missing = error = 0
    ctx = mp.get_context("spawn")
    with open(CKPT, "a", encoding="utf-8") as f:
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for d in pool.imap_unordered(_process, todo, chunksize=4):
                append_checkpoint(f, d)
                if d.get("missing"):
                    missing += 1
                elif d.get("error"):
                    error += 1
                else:
                    ok += 1
    print(f"RESULT ok={ok} missing={missing} error={error} total={total}")
    print(f"SIDECAR: {merge_sidecar_npz(SIDECAR, CKPT, columns={'voice_prob': 'voice_prob'})}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_instrumental_extractor.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_instrumental_sidecar.py tests/unit/test_instrumental_extractor.py
git commit --only -m "feat(instrumental): isolated voice/instrumental sidecar extractor" -- scripts/extract_instrumental_sidecar.py tests/unit/test_instrumental_extractor.py
```

---

## Task 3: Extend WSL preflight to require the voice_instrumental model

**Files:**
- Modify: `src/analyze/energy_runner.py:83-106` (`preflight_wsl`)
- Test: reuse/extend an existing `energy_runner` preflight test if present; otherwise add `tests/unit/test_energy_runner_preflight.py`.

**Interfaces:**
- Produces: `preflight_wsl` additionally `test -f {models_dir}/voice_instrumental-musicnn-msd-2.pb`, so a missing model fails the preflight loudly instead of mid-run.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_energy_runner_preflight.py
from unittest.mock import MagicMock
from src.analyze.energy_runner import preflight_wsl, EnergyConfig


def _cfg():
    return EnergyConfig()  # defaults are fine; we only inspect the probe string


def test_preflight_probe_includes_voice_instrumental_model():
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        m = MagicMock()
        m.returncode = 0
        return m

    preflight_wsl(_cfg(), runner=fake_runner)
    probe = " ".join(captured["cmd"])
    assert "voice_instrumental-musicnn-msd-2.pb" in probe
```

Note: if `EnergyConfig()` requires args, mirror the construction used by the existing energy_runner tests (grep `EnergyConfig(` under `tests/`). Adjust the fixture accordingly — the assertion on the probe string is the invariant.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_energy_runner_preflight.py -v`
Expected: FAIL — assertion error, `voice_instrumental-...pb` not in probe.

- [ ] **Step 3: Write minimal implementation** — add one line to the probe chain

```python
# src/analyze/energy_runner.py  (inside preflight_wsl, appending to the probe chain ~line 89)
    probe = (
        f"test -x {cfg.python} "
        f"&& test -f {cfg.models_dir}/msd-musicnn-1.pb "
        f"&& test -f {cfg.models_dir}/emomusic-msd-musicnn-2.pb "
        f"&& test -f {cfg.models_dir}/danceability-msd-musicnn-1.pb "
        f"&& test -f {cfg.models_dir}/voice_instrumental-musicnn-msd-2.pb"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_energy_runner_preflight.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/analyze/energy_runner.py tests/unit/test_energy_runner_preflight.py
git commit --only -m "feat(instrumental): require voice_instrumental model in WSL preflight" -- src/analyze/energy_runner.py tests/unit/test_energy_runner_preflight.py
```

---

## ⛔ Checkpoint B: Canonical-only sidecar build (NOT a satellite code task)

This is a data-write gate. It runs in the **canonical checkout** (`satellite_data_write_guard` blocks it here). Document completion in the PR; the integration test in Task 13 stays skipped until this is done.

- [ ] **B1.** Fetch `voice_instrumental-musicnn-msd-2.pb` **and** its companion `.json` from the Essentia model zoo into `/opt/ess/models` (WSL). Confirm the exact filenames and TF output node; update the four constants in `scripts/extract_instrumental_sidecar.py` (`VI_PB`, `VI_JSON`, `output=`) and the preflight line in Task 3 if the real names differ.
- [ ] **B2.** Verify the extended preflight passes: it must now require the voice model.
- [ ] **B3.** **Validate-first `--limit` smoke** on a curated label set — a few pure-instrumental ambient tracks, a few spoken-word / poetry-over-ambient, and the Black Moth Super Rainbow vocoder case. Run `python scripts/extract_instrumental_sidecar.py --limit <N>` (via the WSL python). Confirm `voice_prob` separates poetry (high) from pure-instrumental (low), and record the real `trk/s` → ETA. **If it does not separate cleanly, STOP and rethink before the full pass** (see `evaluation-methodology`).
- [ ] **B4.** If validated, run the full library pass: `python scripts/extract_instrumental_sidecar.py --workers 14` (resumable; append-only checkpoint; read-only on audio + `metadata.db`). Produces `<canonical>/data/artifacts/beat3tower_32k/instrumental/instrumental_sidecar.npz`.
- [ ] **B5.** Restart any running GUI worker so the `@lru_cache` on the artifact bundle doesn't hide the new sidecar (`web-gui`/`playlist-testing` trap catalog).

Satellite PG3_SAT1 reads canonical artifacts via absolute paths in its gitignored `config.yaml`, so once B4 lands, Task 13's integration test can run green here.

---

## Task 4: Instrumental read path (`load_voice_prob`)

**Files:**
- Create: `src/playlist/instrumental_loader.py`
- Test: `tests/unit/test_instrumental_loader.py`

**Interfaces:**
- Produces: `load_voice_prob(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray` — returns a 1-D `float64` array of length `len(track_ids)`, aligned by index-map to the sidecar's `track_ids`, `np.nan` for any track absent from the sidecar. **On a missing/unreadable sidecar file, returns an all-NaN array and logs a WARNING** (degrade to inert, never raise — differs from `energy_loader.load_energy_matrix`, which raises).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_instrumental_loader.py
import numpy as np
from src.playlist.instrumental_loader import load_voice_prob


def _write_sidecar(tmp_path, ids, probs):
    p = tmp_path / "instrumental_sidecar.npz"
    np.savez_compressed(p, track_ids=np.array(ids, dtype=object),
                        voice_prob=np.asarray(probs, dtype=np.float32))
    return str(p)


def test_load_voice_prob_aligns_to_requested_order(tmp_path):
    side = _write_sidecar(tmp_path, ["a", "b", "c"], [0.9, 0.1, 0.5])
    out = load_voice_prob(["c", "a", "zzz"], sidecar_path=side)
    assert abs(out[0] - 0.5) < 1e-6   # c
    assert abs(out[1] - 0.9) < 1e-6   # a
    assert np.isnan(out[2])           # unknown track -> NaN


def test_load_voice_prob_missing_sidecar_returns_all_nan(tmp_path, caplog):
    out = load_voice_prob(["a", "b"], sidecar_path=str(tmp_path / "nope.npz"))
    assert out.shape == (2,)
    assert np.isnan(out).all()
    assert any("instrumental" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_instrumental_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.playlist.instrumental_loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/playlist/instrumental_loader.py
"""Read path for the isolated instrumental sidecar (per-track voice_prob).

Mirrors src/playlist/energy_loader.py's index-map alignment, but degrades to an
all-NaN array (+ WARNING) on a missing sidecar instead of raising — the
Instrumental guard must be inert-but-safe when the data is absent, never fatal.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_voice_prob(track_ids: Sequence[str], *, sidecar_path: str) -> np.ndarray:
    n = len(track_ids)
    out = np.full(n, np.nan, dtype=np.float64)
    if not os.path.exists(sidecar_path):
        logger.warning(
            "instrumental_loader: sidecar %s absent; voice_prob all-NaN (Instrumental guard inert)",
            sidecar_path,
        )
        return out
    try:
        data = np.load(sidecar_path, allow_pickle=True)
    except Exception as exc:
        logger.warning(
            "instrumental_loader: failed to read sidecar %s (%s); voice_prob all-NaN", sidecar_path, exc
        )
        return out
    if "voice_prob" not in data or "track_ids" not in data:
        logger.warning("instrumental_loader: sidecar %s missing keys; voice_prob all-NaN", sidecar_path)
        return out
    side_ids = [str(t) for t in data["track_ids"]]
    side_vp = np.asarray(data["voice_prob"], dtype=np.float64)
    pos = {t: i for i, t in enumerate(side_ids)}
    for i, t in enumerate(track_ids):
        j = pos.get(str(t))
        if j is not None:
            out[i] = side_vp[j]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_instrumental_loader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/instrumental_loader.py tests/unit/test_instrumental_loader.py
git commit --only -m "feat(instrumental): voice_prob read path (inert-on-missing)" -- src/playlist/instrumental_loader.py tests/unit/test_instrumental_loader.py
```

---

## Task 5: Penalty helper `compute_instrumental_penalty`

**Files:**
- Modify: `src/playlist/pier_bridge/pace_gate.py` (add function beside `compute_energy_pace_penalty` at ~:152)
- Test: `tests/unit/test_instrumental_penalty.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `compute_instrumental_penalty(voice_prob: Optional[np.ndarray], *, cand: int, weight: float) -> float` — returns `weight * voice_prob[cand]`, clamped `>= 0`; returns `0.0` when `voice_prob is None`, `weight <= 0`, or `voice_prob[cand]` is NaN. NEVER raises, NEVER signals exclusion (callers only subtract it). Used by both the beam (Task 7) and the pool (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_instrumental_penalty.py
import numpy as np
from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty


def test_penalty_proportional_to_voice_prob():
    vp = np.array([0.95, 0.55, 0.0])
    assert abs(compute_instrumental_penalty(vp, cand=0, weight=0.6) - 0.57) < 1e-6
    assert abs(compute_instrumental_penalty(vp, cand=1, weight=0.6) - 0.33) < 1e-6
    assert compute_instrumental_penalty(vp, cand=2, weight=0.6) == 0.0


def test_penalty_nan_and_disabled_are_zero():
    vp = np.array([np.nan, 0.9])
    assert compute_instrumental_penalty(vp, cand=0, weight=0.6) == 0.0   # NaN -> unpunished
    assert compute_instrumental_penalty(vp, cand=1, weight=0.0) == 0.0   # weight 0 -> off
    assert compute_instrumental_penalty(None, cand=1, weight=0.6) == 0.0  # no data -> off
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_instrumental_penalty.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_instrumental_penalty'`

- [ ] **Step 3: Write minimal implementation** (append to `pace_gate.py`)

```python
# src/playlist/pier_bridge/pace_gate.py  (add near compute_energy_pace_penalty)
from typing import Optional
import numpy as np


def compute_instrumental_penalty(
    voice_prob: Optional[np.ndarray],
    *,
    cand: int,
    weight: float,
) -> float:
    """SOFT instrumental-lean penalty (>= 0). Additive; callers subtract it.

    penalty = weight * voice_prob[cand]. NEVER raises, NEVER signals exclusion.
    voice_prob is None / weight <= 0 / NaN prob -> 0.0 (unknown is never punished).
    """
    if voice_prob is None or weight <= 0.0:
        return 0.0
    vp = float(voice_prob[int(cand)])
    if not np.isfinite(vp):
        return 0.0
    return weight * max(0.0, vp)
```

(If `pace_gate.py` already imports `numpy as np` / `Optional`, don't duplicate the imports — reuse the existing ones.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_instrumental_penalty.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/pace_gate.py tests/unit/test_instrumental_penalty.py
git commit --only -m "feat(instrumental): additive voice_prob penalty helper" -- src/playlist/pier_bridge/pace_gate.py tests/unit/test_instrumental_penalty.py
```

---

## Task 6: Config fields on `PierBridgeConfig` + override parsing + `config.yaml`

**Files:**
- Modify: `src/playlist/pier_bridge/config.py` (add two fields near the pace-penalty fields ~:60-61)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (generic override loop ~:119-132 + a bool line)
- Modify: `config.yaml` and `config.example.yaml` (under `playlists.ds_pipeline.pier_bridge`)
- Test: `tests/unit/test_pier_bridge_overrides_instrumental.py`

**Interfaces:**
- Produces: `PierBridgeConfig.instrumental_enabled: bool = False`, `PierBridgeConfig.instrumental_penalty_weight: float = 0.0`. `apply_pier_bridge_overrides` reads `instrumental_penalty_weight` (float) and `instrumental_enabled` (bool) from the pier-bridge overrides dict onto the config. Consumed by Task 7 (beam) and Task 8 (pool via the pool cfg mirror).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pier_bridge_overrides_instrumental.py
from dataclasses import replace
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides


def test_instrumental_fields_default_off():
    cfg = PierBridgeConfig()
    assert cfg.instrumental_enabled is False
    assert cfg.instrumental_penalty_weight == 0.0


def test_overrides_apply_instrumental_fields():
    cfg = PierBridgeConfig()
    # apply_pier_bridge_overrides signature: confirm how the reference call passes the
    # overrides dict (grep an existing call, e.g. for "seed_character_strength").
    out = apply_pier_bridge_overrides(
        cfg,
        pb_overrides={"instrumental_enabled": True, "instrumental_penalty_weight": 0.6},
    )
    assert out.instrumental_enabled is True
    assert abs(out.instrumental_penalty_weight - 0.6) < 1e-9
```

Note: `apply_pier_bridge_overrides`'s real parameter name/shape (it takes a `PierBridgeTuning` plus a free-form overrides dict per the mapping) must match the existing signature at `src/playlist/pipeline/pier_bridge_overrides.py:36`. Grep an existing call site and mirror it in the test; the assertion (fields set from the dict) is the invariant.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_pier_bridge_overrides_instrumental.py -v`
Expected: FAIL — `AttributeError: 'PierBridgeConfig' object has no attribute 'instrumental_enabled'`

- [ ] **Step 3: Write minimal implementation**

In `src/playlist/pier_bridge/config.py`, add beside the pace-penalty fields (~:60-61):

```python
    # Instrumental lean (soft, continuous on voice_prob; never a hard gate).
    # enabled is per-request (policy override); penalty_weight is static (config.yaml).
    instrumental_enabled: bool = False
    instrumental_penalty_weight: float = 0.0
```

In `src/playlist/pipeline/pier_bridge_overrides.py`, add `instrumental_penalty_weight` to the generic scalar tuple (~:119) and a bool line right after the loop:

```python
    for _k, _cast in (("variable_bridge_flex", int),
                      ("variable_bridge_min_edge", float), ("variable_bridge_epsilon", float),
                      ("variable_bridge_max_flex_segments", int),
                      ("generation_budget_s", float),
                      ("seed_character_strength", float),
                      ("mini_pier_max_interior", int), ("mini_pier_smoothness_margin", float),
                      ("instrumental_penalty_weight", float)):   # <-- add
        _v = pb_overrides.get(_k)
        if isinstance(_v, (int, float)) and not isinstance(_v, bool):
            pb_cfg = replace(pb_cfg, **{_k: _cast(_v)})

    # instrumental_enabled is a bool (the generic loop above skips bools)
    _ie = pb_overrides.get("instrumental_enabled")
    if isinstance(_ie, bool):
        pb_cfg = replace(pb_cfg, instrumental_enabled=_ie)
```

In `config.yaml` and `config.example.yaml`, under `playlists.ds_pipeline.pier_bridge:` (beside `soft_genre_penalty_*`, 6-space indent), add:

```yaml
      # Instrumental lean: additive beam+pool demotion = weight * voice_prob when the
      # "Instrumental" checkbox is on. Steep enough to be decisive without a de-facto
      # hard gate. Tune via tests/integration/test_gui_fidelity_instrumental.py (Task 13).
      instrumental_penalty_weight: 0.6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_pier_bridge_overrides_instrumental.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py config.example.yaml tests/unit/test_pier_bridge_overrides_instrumental.py
git commit --only -m "feat(instrumental): pier-bridge config fields + override parsing" -- src/playlist/pier_bridge/config.py src/playlist/pipeline/pier_bridge_overrides.py config.example.yaml tests/unit/test_pier_bridge_overrides_instrumental.py
```

Note: `config.yaml` is gitignored — edit it locally (it must carry the same key for live runs) but do not stage it. Only `config.example.yaml` is committed.

---

## Task 7: Thread `voice_prob` into the beam + apply the penalty

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` — add `voice_prob` param to `_beam_search_segment` (~:192-200); accumulate the penalty into `_pace_penalty` (~:1173, beside the energy call).
- Modify: `src/playlist/pier_bridge_builder.py` — add `voice_prob` param to `build_pier_bridge_playlist` (~:457) and pass it to `_beam_search_segment` (~:1519-1526).
- Modify: `src/playlist/pipeline/core.py` — load the sidecar when `pb_cfg.instrumental_enabled` (warn-to-None), pass `voice_prob=` into `build_pier_bridge_playlist` (~:898-902).
- Test: covered end-to-end by Task 13 (integration). Add a fast, artifact-free beam smoke here only if a `_beam_search_segment` unit fixture already exists; otherwise rely on Task 5's helper test + Task 13.

**Interfaces:**
- Consumes: `compute_instrumental_penalty` (Task 5); `load_voice_prob` (Task 4); `pb_cfg.instrumental_enabled`, `pb_cfg.instrumental_penalty_weight` (Task 6).
- Produces: a demotion of `combined_score` for bridge-interior candidates by `weight * voice_prob[cand]`, applied via the existing `combined_score -= _pace_penalty` at `beam.py:1256`. Piers are never touched (guarded by `if cand in state.used: continue` at `beam.py:1112`).

- [ ] **Step 1: Add the `voice_prob` parameter to `_beam_search_segment`**

In `beam.py`, extend the signature (~:192-200, beside `energy_matrix`):

```python
    energy_matrix: Optional[np.ndarray] = None,
    voice_prob: Optional[np.ndarray] = None,   # <-- add
    popularity_values: Optional[np.ndarray] = None,
```

- [ ] **Step 2: Accumulate the penalty into `_pace_penalty`**

In `beam.py`, right after the energy-penalty block (~:1173-1183, inside the `for cand in candidates:` loop, after the `if cand in state.used: continue` guard), add:

```python
                # Instrumental lean: additive demotion of vocal-classified candidates.
                # Rides combined_score -= _pace_penalty (beam.py:1256); bridge-only by the
                # state.used guard above; NaN/None/weight<=0 -> 0.0.
                _instr_w = float(getattr(cfg, "instrumental_penalty_weight", 0.0))
                if voice_prob is not None and _instr_w > 0.0:
                    from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
                    _pace_penalty += compute_instrumental_penalty(
                        voice_prob, cand=int(cand), weight=_instr_w
                    )
```

- [ ] **Step 3: Thread `voice_prob` through `build_pier_bridge_playlist`**

In `pier_bridge_builder.py`, add the param to the signature (~:457, beside `onset_rate`/`energy_matrix`):

```python
    energy_matrix: Optional[np.ndarray] = None,
    voice_prob: Optional[np.ndarray] = None,   # <-- add
```

and pass it at the `_beam_search_segment(...)` call site (~:1519-1526):

```python
        energy_matrix=energy_matrix,
        voice_prob=voice_prob,   # <-- add
        popularity_values=popularity_values,
```

- [ ] **Step 4: Load the sidecar + pass it in `pipeline/core.py`**

In `pipeline/core.py`, after `pb_cfg` is finalized (after the pace `replace(...)` at ~:716-728) and near the popularity load (~:881-883), add:

```python
    # Instrumental lean: load voice_prob only when enabled; degrade to inert + WARN.
    voice_prob = None
    _instr_active = bool(getattr(pb_cfg, "instrumental_enabled", False))
    if _instr_active:
        from pathlib import Path
        from src.playlist.instrumental_loader import load_voice_prob
        _instr_sidecar = Path(artifact_path).parent / "instrumental" / "instrumental_sidecar.npz"
        voice_prob = load_voice_prob(bundle.track_ids, sidecar_path=str(_instr_sidecar))
        if voice_prob is None or not np.isfinite(voice_prob).any():
            logger.warning(
                "Instrumental lean ON but voice_prob absent/all-NaN at %s — guard inert this run",
                _instr_sidecar,
            )
```

and pass it into the `build_pier_bridge_playlist(...)` call (~:898-902):

```python
        energy_matrix=energy_matrix,
        voice_prob=voice_prob,   # <-- add
```

(Confirm `artifact_path`, `bundle`, `np`, and `logger` are the in-scope names at this point — grep the surrounding energy-load block ~:424-441, which uses the same names.)

- [ ] **Step 5: Run the fast suite to confirm no import/wiring breakage**

Run: `python -m pytest tests/unit/test_instrumental_penalty.py tests/unit/test_pier_bridge_overrides_instrumental.py -q -m "not slow"`
Expected: PASS. (Behavioral proof is Task 13.)

- [ ] **Step 6: Commit**

```bash
git add src/playlist/pier_bridge/beam.py src/playlist/pier_bridge_builder.py src/playlist/pipeline/core.py
git commit --only -m "feat(instrumental): thread voice_prob into beam + apply soft penalty" -- src/playlist/pier_bridge/beam.py src/playlist/pier_bridge_builder.py src/playlist/pipeline/core.py
```

---

## Task 8: Candidate-pool demotion

**Files:**
- Modify: `src/playlist/candidate_pool.py` — add a `seed_sim_all` demotion block modeled on the duration penalty (~:640, after the duration block at :609-633, before the sonic seed-sim at :643); thread `voice_prob` + weight + enabled into `build_candidate_pool` via its config object (mirror `duration_penalty_weight`).
- Modify: `src/playlist/pipeline/core.py` — pass `voice_prob` (and ensure the pool cfg carries `instrumental_penalty_weight`/`instrumental_enabled`) into the `build_candidate_pool(...)` call.
- Test: `tests/unit/test_candidate_pool_instrumental.py` if `build_candidate_pool` has an existing unit-fixture path; otherwise assert the demotion via Task 13.

**Interfaces:**
- Consumes: `compute_instrumental_penalty` (Task 5); `voice_prob` (Task 4); the pool config's `instrumental_penalty_weight` / `instrumental_enabled`.
- Produces: `seed_sim_all[i] -= weight * voice_prob[i]` for non-seed candidates, reducing their admission ranking so a demoted vocal track cannot re-enter via a strong edge (the tag-steering "beam was blind → too weak" lesson). Seeds (`seed_mask[i]`) are skipped.

- [ ] **Step 1: Locate the pool-config analog for `duration_penalty_weight`**

Grep for where `duration_penalty_weight` is declared on the pool config and populated from `config.yaml`:

Run: `grep -rn "duration_penalty_weight" src/playlist/candidate_pool.py src/playlist/pipeline/`
Add `instrumental_penalty_weight: float = 0.0` and `instrumental_enabled: bool = False` to the same pool-config dataclass, and populate them from the same override path (`playlists.ds_pipeline.pier_bridge.instrumental_*`) that Task 6 wired for the beam — so a single config source feeds both.

- [ ] **Step 2: Write the demotion block**

In `candidate_pool.py`, after the duration-penalty block (~:633) and before `sonic_seed_sim` (~:643):

```python
    # Instrumental lean: demote vocal-classified candidates in the admission ranking so a
    # demoted-but-present track can't win a pool slot via a strong edge. Soft: a thin pool
    # still admits them (never-fail). Seeds are exempt.
    instrumental_penalty_count = 0
    _instr_w = float(getattr(cfg, "instrumental_penalty_weight", 0.0))
    if getattr(cfg, "instrumental_enabled", False) and _instr_w > 0.0 and voice_prob is not None:
        from src.playlist.pier_bridge.pace_gate import compute_instrumental_penalty
        seed_sim_all = seed_sim_all.copy()
        for i in range(len(seed_sim_all)):
            if seed_mask[i]:
                continue
            penalty = compute_instrumental_penalty(voice_prob, cand=i, weight=_instr_w)
            if penalty > 0:
                seed_sim_all[i] -= penalty
                instrumental_penalty_count += 1
```

- [ ] **Step 3: Add `voice_prob` to `build_candidate_pool` signature + pass it from core.py**

Add `voice_prob: Optional[np.ndarray] = None` to `build_candidate_pool(...)` (~:518) and pass `voice_prob=voice_prob` at its call site in `pipeline/core.py`. Ensure the pool cfg built in core.py carries `instrumental_penalty_weight` + `instrumental_enabled` (Step 1).

- [ ] **Step 4: Run the fast suite for import/wiring sanity**

Run: `python -m pytest tests/unit/test_instrumental_penalty.py -q`
Expected: PASS. (Behavioral proof is Task 13; the pool demotion is visible there as a stronger drop than beam-only.)

- [ ] **Step 5: Commit**

```bash
git add src/playlist/candidate_pool.py src/playlist/pipeline/core.py
git commit --only -m "feat(instrumental): candidate-pool voice_prob demotion" -- src/playlist/candidate_pool.py src/playlist/pipeline/core.py
```

---

## Task 9: Policy routing + typed API field + `UIStateModel`

**Files:**
- Modify: `src/playlist_gui/ui_state.py:68` — `instrumental` field on `UIStateModel`.
- Modify: `src/playlist_web/schemas.py:27` — `instrumental` field on `GenerateRequestBody`.
- Modify: `src/playlist_web/app.py:175-193` — `instrumental=body.instrumental` in the `UIStateModel(...)` construction.
- Modify: `src/playlist_gui/policy.py` — register the config key in `POLICY_OWNED_KEYS` (~:29-54); set the override in `derive_runtime_config` (~:368, beside the recency block).
- Modify: `web/src/lib/types.ts:17` — `instrumental?: boolean` on `GenerateRequestBody`.
- Test: `tests/test_gui_fidelity.py` (fast) — added in Task 12.

**Interfaces:**
- Consumes: nothing new.
- Produces: `ui.instrumental` (bool) → `overrides["playlists.ds_pipeline.pier_bridge.instrumental_enabled"]`. This is the load-bearing channel the beam/pool read (Track B).

- [ ] **Step 1: Add the field to `UIStateModel`**

`src/playlist_gui/ui_state.py` (~:68, beside `recency_enabled`):

```python
    instrumental: bool = False
```

- [ ] **Step 2: Add the field to the Pydantic request schema**

`src/playlist_web/schemas.py` (~:27, beside `exclude_seed_tracks_from_recency`):

```python
    instrumental: bool = False
```

- [ ] **Step 3: Read it into `UIStateModel` in the web route**

`src/playlist_web/app.py` (~:175-193, inside the `UIStateModel(...)` construction, beside `recency_enabled=body.recency_enabled`):

```python
        instrumental=body.instrumental,
```

- [ ] **Step 4: Register the policy-owned key + set the override**

`src/playlist_gui/policy.py`, add to `POLICY_OWNED_KEYS` (~:29-54):

```python
    "playlists.ds_pipeline.pier_bridge.instrumental_enabled",
```

and in `derive_runtime_config` (~:368, beside the recency block):

```python
    _set_nested(overrides, "playlists.ds_pipeline.pier_bridge.instrumental_enabled", ui.instrumental)
    if ui.instrumental:
        notes.append("Instrumental lean: demote vocal-classified tracks")
```

- [ ] **Step 5: Add the TypeScript field**

`web/src/lib/types.ts` (~:17, beside `exclude_seed_tracks_from_recency?: boolean;`):

```ts
  instrumental?: boolean;
```

- [ ] **Step 6: Run the fast fidelity suite**

Run: `python -m pytest tests/test_gui_fidelity.py -q`
Expected: PASS (existing tests still green; the new routing assertion is added in Task 12).

- [ ] **Step 7: Commit**

```bash
git add src/playlist_gui/ui_state.py src/playlist_web/schemas.py src/playlist_web/app.py src/playlist_gui/policy.py web/src/lib/types.ts
git commit --only -m "feat(instrumental): route flag through policy to beam/pool config" -- src/playlist_gui/ui_state.py src/playlist_web/schemas.py src/playlist_web/app.py src/playlist_gui/policy.py web/src/lib/types.ts
```

---

## Task 10: GUI checkbox (Row 3)

**Files:**
- Modify: `web/src/components/GenerateControls.tsx` — state hook (~:105-111), Row 3 checkbox (~:471-538), request-body key (~:206-233).
- **Do NOT touch** `web/src/components/AdvancedPanel.tsx`.
- Verify: `npm --prefix web run build` succeeds; then rebuild `web/dist` so the served GUI picks it up (`web-gui` stale-dist trap).

**Interfaces:**
- Consumes: `GenerateRequestBody.instrumental` (Task 9, `types.ts`).
- Produces: a persisted checkbox that sends `instrumental: <bool>` on generate.

- [ ] **Step 1: Add the state hook**

In `GenerateControls.tsx`, in the Row 3 state block (~:111, beside `excludeRecentSeeds`):

```tsx
  const [instrumental, setInstrumental] = useLocalStorage("pg_instrumental", false);
```

- [ ] **Step 2: Add the checkbox to Row 3**

Insert a `<Cell>` in Row 3 (between ~:471 and ~:538), mirroring the "skip recent seeds" pattern:

```tsx
              <Cell>
                <label
                  className="flex items-center gap-1.5 cursor-pointer select-none"
                  title="Demote vocal-classified tracks (spoken-word / poetry) from bridges. Soft, never a hard filter. Note: heavily-processed vocals (vocoder/talkbox) may read as instrumental and slip through."
                >
                  <input
                    type="checkbox"
                    checked={instrumental}
                    onChange={(e) => setInstrumental(e.target.checked)}
                    className="accent-[#5eead4] cursor-pointer"
                  />
                  <Lbl>instrumental</Lbl>
                </label>
              </Cell>
```

- [ ] **Step 3: Include it in the request body**

In `submit()` (~:206-233), inside the `body` object literal (beside `exclude_seed_tracks_from_recency: excludeRecentSeeds,` at ~:220):

```tsx
      instrumental: instrumental,
```

- [ ] **Step 4: Build the frontend**

Run: `npm --prefix web run build`
Expected: build succeeds, no TypeScript errors. (Restart `serve_web.py` / rebuild `web/dist` before manual GUI checks — `web-gui` trap.)

- [ ] **Step 5: Commit**

```bash
git add web/src/components/GenerateControls.tsx
git commit --only -m "feat(instrumental): Row 3 Instrumental checkbox" -- web/src/components/GenerateControls.tsx
```

---

## Task 11: Confession / warning receipt

**Files:**
- Modify: `src/playlist/pipeline/core.py` (or wherever `playlist_stats["playlist"]` is assembled — grep) — after the playlist is built, write `playlist_stats["playlist"]["instrumental"] = {"enabled": bool, "admitted_count": int, "threshold": 0.5}` from the final non-seed track indices + `voice_prob`.
- Modify: `src/playlist_gui/receipt.py:31-68` (`compose_receipt`) — append the two notes.
- Test: `tests/unit/test_receipt_compose.py` (add cases, mirror :26-30 / :50-62).

**Interfaces:**
- Consumes: `voice_prob` (Task 4); `pb_cfg.instrumental_enabled`.
- Produces: two `notes[]` entries surfaced by the existing `QualityStats.tsx::ReceiptLine` (no frontend change). Notes must use listener-facing language only — never engine vocabulary (enforced by `test_receipt_compose.py:31-35`, `:57-59`).

- [ ] **Step 1: Write the failing receipt test**

```python
# tests/unit/test_receipt_compose.py  (add)
from src.playlist_gui.receipt import compose_receipt


def test_instrumental_admitted_note_fires_and_is_listener_facing():
    pstats = {"instrumental": {"enabled": True, "admitted_count": 3, "threshold": 0.5}}
    out = compose_receipt(pstats, {})
    joined = " ".join(out["notes"]).lower()
    assert "vocal" in joined                 # count-driven note present
    assert "3" in " ".join(out["notes"])
    # never leak engine vocabulary
    for banned in ("voice_prob", "penalty", "beam", "candidate_pool", "sidecar"):
        assert banned not in joined


def test_instrumental_caveat_present_when_enabled_zero_admitted():
    pstats = {"instrumental": {"enabled": True, "admitted_count": 0, "threshold": 0.5}}
    out = compose_receipt(pstats, {})
    joined = " ".join(out["notes"]).lower()
    assert "processed" in joined or "vocoder" in joined or "talkbox" in joined


def test_no_instrumental_notes_when_disabled():
    out = compose_receipt({}, {})
    joined = " ".join(out["notes"]).lower()
    assert "vocal" not in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_receipt_compose.py -v -k instrumental`
Expected: FAIL — notes don't yet mention vocals.

- [ ] **Step 3: Add the notes in `compose_receipt`**

In `src/playlist_gui/receipt.py`, inside `compose_receipt` before `return {...}` (~:61):

```python
    instr = playlist_stats.get("instrumental") or {}
    if instr.get("enabled"):
        admitted = _i(instr.get("admitted_count")) or 0
        if admitted:
            notes.append(
                f"kept {admitted} vocal track{'s' if admitted != 1 else ''} to fill out the playlist"
            )
        # one-time caveat (always when enabled): the classifier's known blind spots
        notes.append(
            "heavily-processed vocals may slip through, and some wordless textures may be held back"
        )
```

(`_i` is the existing int-coercion helper used by the surrounding notes.)

- [ ] **Step 4: Write the confession stats in the generation path**

Where `playlist_stats["playlist"]` is assembled (grep `playlist_stats` in `pipeline/core.py` / the ds runner), after the final tracklist is known:

```python
    if _instr_active:
        _thr = 0.5
        _admitted = 0
        for t_idx in final_nonseed_indices:   # non-seed bridge track global indices
            if voice_prob is not None and np.isfinite(voice_prob[t_idx]) and voice_prob[t_idx] > _thr:
                _admitted += 1
        playlist_stats.setdefault("playlist", {})["instrumental"] = {
            "enabled": True, "admitted_count": _admitted, "threshold": _thr,
        }
```

Confirm the exact variable holding non-seed final indices and the `playlist_stats` dict name at the assembly site (grep). If the built playlist is a list of track_ids, map them to indices via `bundle.track_id_to_index` and exclude the seed set.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_receipt_compose.py -v`
Expected: PASS (all, including the 3 new cases)

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/receipt.py src/playlist/pipeline/core.py tests/unit/test_receipt_compose.py
git commit --only -m "feat(instrumental): confession notes for admitted vocals + caveat" -- src/playlist_gui/receipt.py src/playlist/pipeline/core.py tests/unit/test_receipt_compose.py
```

---

## Task 12: Fast policy-routing guard test

**Files:**
- Modify: `tests/test_gui_fidelity.py` (fast tier — config-resolution only, no artifact)

**Interfaces:**
- Consumes: `tests/support/gui_fidelity.py::{gui_ui_state, resolve_gui_overrides}` (:48-75); `UIStateModel.instrumental` (Task 9).
- Produces: a guard that the flag survives `derive_runtime_config` into the resolved ds-overrides (the inert-knob trap).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui_fidelity.py  (add)
from support.gui_fidelity import gui_ui_state, resolve_gui_overrides


def test_instrumental_flag_reaches_pier_bridge_overrides():
    ov_on = resolve_gui_overrides(gui_ui_state(instrumental=True))
    ov_off = resolve_gui_overrides(gui_ui_state(instrumental=False))
    # resolve_gui_overrides returns build_ds_overrides output; the pier-bridge block
    # carries instrumental_enabled. Assert on whatever nesting build_ds_overrides yields
    # (grep an existing pier_bridge assertion in this file for the exact access path).
    assert _pier_bridge(ov_on).get("instrumental_enabled") is True
    assert _pier_bridge(ov_off).get("instrumental_enabled") in (False, None)


def _pier_bridge(ov):
    # mirror how existing tests reach pier_bridge overrides in this file
    return ov.get("pier_bridge", ov)
```

Note: adjust `_pier_bridge` to match how `build_ds_overrides` nests the pier-bridge block (grep an existing `pier_bridge` assertion in `tests/test_gui_fidelity.py`). The invariant: `instrumental_enabled` is `True` when the flag is on and falsy when off — proving Track B routing.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gui_fidelity.py -v -k instrumental`
Expected: FAIL initially only if a nesting-path mismatch — fix `_pier_bridge` to the real path, then it proves routing. (If Task 9 were skipped it would fail on `instrumental_enabled` being absent.)

- [ ] **Step 3: (no new impl)** — routing was implemented in Task 9. This task only adds the guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gui_fidelity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_gui_fidelity.py
git commit --only -m "test(instrumental): guard policy routing of the flag (inert-knob trap)" -- tests/test_gui_fidelity.py
```

---

## Task 13: Generation-fidelity integration test (multi-pier)

**Files:**
- Create: `tests/integration/test_gui_fidelity_instrumental.py`
- Depends on: Checkpoint B complete (real `instrumental_sidecar.npz` reachable via config's artifact path). Skips otherwise.

**Interfaces:**
- Consumes: `tests/support/gui_fidelity.py::{generate_like_gui, gui_ui_state}` (:48-136); `src/playlist/instrumental_loader.load_voice_prob`; the live artifact bundle.
- Produces: proof that `instrumental=True` lowers the playlist's mean `voice_prob` vs `instrumental=False`, through the real GUI config chain (never hand-built overrides, never single-seed).

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_gui_fidelity_instrumental.py
import os
from pathlib import Path

import numpy as np
import pytest

from support.gui_fidelity import generate_like_gui, gui_ui_state
from src.playlist.pipeline.artifact_loader import load_artifact_bundle  # match real import path
from src.playlist.instrumental_loader import load_voice_prob

ART = os.environ.get("PLAYLIST_GOLDEN_ARTIFACT", "data/artifacts/beat3tower_32k")
SIDECAR = str(Path(ART) / "instrumental" / "instrumental_sidecar.npz")

# Multi-pier artist-mode seeds: >=2 distinct artists (mirror SEEDS in
# tests/integration/test_gui_fidelity_regressions.py — reuse that constant if importable).
SEEDS = [
    # fill with 3-5 real bundle track_ids from >=2 artists, per the regressions test
]

_requires_artifact = pytest.mark.skipif(
    not Path(ART).exists() or not Path(SIDECAR).exists(),
    reason="live artifact + instrumental sidecar required (Checkpoint B)",
)


def _mean_voice_prob(bundle, track_ids):
    vp = load_voice_prob(bundle.track_ids, sidecar_path=SIDECAR)
    idx = bundle.track_id_to_index
    vals = [vp[idx[str(t)]] for t in track_ids if str(t) in idx]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_instrumental_flag_lowers_mean_voice_prob():
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))

    off = generate_like_gui(
        seeds=SEEDS, cohesion_mode="narrow", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
        instrumental=False, length=30, random_seed=0,
    )
    on = generate_like_gui(
        seeds=SEEDS, cohesion_mode="narrow", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
        instrumental=True, length=30, random_seed=0,
    )

    mv_off = _mean_voice_prob(bundle, off.track_ids)
    mv_on = _mean_voice_prob(bundle, on.track_ids)
    assert mv_on < mv_off, f"instrumental lean should lower mean voice_prob: on={mv_on} off={mv_off}"
```

Notes:
- Fill `SEEDS` with real bundle track_ids from ≥2 artists — reuse the `SEEDS` constant in `tests/integration/test_gui_fidelity_regressions.py` if importable (it is already a validated multi-pier set).
- Confirm the real `load_artifact_bundle` import path (grep — it's referenced in `test_gui_fidelity_regressions.py`).
- The mean-shift assertion is deliberately robust to pool nondeterminism (per `test_tag_steering_behavioral.py`'s pattern), rather than asserting a single track's presence/absence.

- [ ] **Step 2: Run it (skips if the sidecar isn't built yet)**

Run: `python -m pytest tests/integration/test_gui_fidelity_instrumental.py -v`
Expected before Checkpoint B: SKIPPED ("live artifact + instrumental sidecar required"). After Checkpoint B (in canonical, or in this satellite via absolute config paths): PASS with `mv_on < mv_off`.

- [ ] **Step 3: If it fails after Checkpoint B — diagnose via logs, not the metric**

Per `playlist-testing` "Diagnosing a generation outcome": run one generation at INFO, grep the gate tally + `pool_after_gate`. A flat result can be a starved pool (beam never ran), not a weak penalty. Only after confirming the pool has room and the penalty fires (monkeypatch-count `compute_instrumental_penalty`) should `instrumental_penalty_weight` be raised. Tune the weight in `config.yaml` and re-run.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_gui_fidelity_instrumental.py
git commit --only -m "test(instrumental): multi-pier gui_fidelity demotion test" -- tests/integration/test_gui_fidelity_instrumental.py
```

---

## Task 14: End-to-end activation + full-suite verification

**Files:** none (verification + activation).

- [ ] **Step 1: Run the full fast suite** (never pipe through `tail`/`head`; bound with the tool timeout)

Run: `python -m pytest -q -m "not slow"`
Expected: all pass. Quote the real pass/fail counts.

- [ ] **Step 2: Lint + types**

Run: `ruff check src/playlist/instrumental_loader.py src/playlist/pier_bridge/pace_gate.py scripts/extract_instrumental_sidecar.py scripts/ess_sidecar_common.py && mypy src/playlist/instrumental_loader.py`
Expected: clean.

- [ ] **Step 3: Exercise the real GUI path** (after Checkpoint B + `web/dist` rebuilt + worker restarted)

Toggle the **instrumental** checkbox on a poetry-over-ambient-prone seed set; confirm (a) vocal tracks are demoted vs the box off, (b) the receipt shows the "kept N vocal tracks" + caveat notes when relevant, (c) with the box on but the sidecar absent, the run still completes and logs the loud inert warning (never-fail). This is the "activate the fix; exercise the real path" gate — default off is correct (a user choice, not a withheld fix).

- [ ] **Step 4: Finish the branch** — use `superpowers:finishing-a-development-branch` (push `feat/instrumental-lean`, open PR to `master`; the canonical checkout performs the merge). Note Checkpoint B (canonical sidecar build) as a prerequisite in the PR body.

---

## Self-Review

**Spec coverage:**
- Component 1 (extraction, isolated sidecar) → Tasks 1, 2, 3, Checkpoint B. ✅ (isolation: energy script untouched; validate-first in B3.)
- Component 2 (read path) → Task 4. ✅ (missing → all-NaN + warn.)
- Component 3 (beam + pool penalty, continuous, seeds/piers exempt, never-fail) → Tasks 5, 7, 8. ✅
- Component 4 (config; missing-data warns loudly) → Task 6 (`instrumental_penalty_weight`) + Task 7 Step 4 (loud inert warning). ✅
- Component 5 (policy routing) → Task 9. ✅ (Track B load-bearing; request_models.py Track A intentionally skipped — rationale in header; nothing load-bearing lost.)
- Component 6 (GUI checkbox, Row 3, do-not-touch AdvancedPanel) → Task 10. ✅
- Component 7 (confession/warning) → Task 11. ✅
- Testing (gui_fidelity multi-pier; loader; extraction smoke; policy routing) → Tasks 4, 2, 12, 13. ✅
- Open items 1 (class-column order), 3 (weight calibration), 4 (mean aggregation) → Task 2 (`voice_column_index`, runtime-validated), Task 6 + Task 13 Step 3 (weight recipe), Task 2 (mean). Open item 2 (insertion points) → resolved by the mapped line anchors. ✅

**Placeholder scan:** every code step carries real code; the few "grep to confirm the exact line" steps (Task 3 fixture, Task 6 signature, Task 8 pool-cfg analog, Task 11 stats-assembly site, Task 12 nesting path, Task 13 seeds/import) are concrete locate-then-apply actions against a shown template, not deferred work.

**Type consistency:** `voice_prob: Optional[np.ndarray]` and `compute_instrumental_penalty(voice_prob, *, cand, weight)` are used identically in Tasks 5, 7, 8. `instrumental_enabled: bool` / `instrumental_penalty_weight: float` names are consistent across config (Task 6), policy override key (Task 9), and consumers (Tasks 7, 8). `ui.instrumental` / `body.instrumental` / `GenerateRequestBody.instrumental` consistent across Tasks 9, 10.
