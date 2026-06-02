# Sonic Tower-Weighted Fix (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the production sonic space actually apply the intended 0.20/0.50/0.30 tower weighting by rebuilding the artifact in a `tower_weighted` variant and cleaning the runtime path, then validate before/after on reference seeds.

**Architecture:** A surgical artifact rebuild recomputes only the sonic matrices (full + start/mid/end) as per-tower `√weight · L2(tower)` from per-tower matrices already in the npz — genre matrices and track order stay byte-identical, so the dense-genre sidecar remains valid and no DB/audio is read. A one-line-of-logic runtime fix in `embedding_setup.py` makes the pipeline use any pre-scaled artifact directly instead of mislabeling it and re-standardizing. Baseline metrics are captured on unmodified code first; the fix + new artifact ship together and are validated against that baseline.

**Tech Stack:** Python 3.11, NumPy, scikit-learn (existing), PySide6 GUI config chain (read-only via the GUI-fidelity harness), pytest.

**Spec:** `docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md`
**Memory:** `project-sonic-tower-weights-inert`

**Invariants (do not violate):**
- `data/metadata.db` and audio files are read-only. This plan only reads the npz + DB and writes a *new* npz (plus a backup of the old one). No DB writes, no audio access.
- Tower layout in the 86-dim space: **rhythm `[0:9]`, timbre `[9:66]`, harmony `[66:86]`** (9 + 57 + 20).
- Weights come from `config.yaml` `playlists.ds_pipeline.tower_weights` = `{rhythm: 0.2, timbre: 0.5, harmony: 0.3}`. Never hardcode them in library code.

---

### Task 1: Pure tower-weighted helpers

**Files:**
- Create: `src/features/sonic_rebuild.py`
- Test: `tests/unit/test_sonic_rebuild.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sonic_rebuild.py
import numpy as np
import pytest

from src.features.sonic_rebuild import (
    tower_weighted_from_towers,
    build_tower_weighted_arrays,
)


def test_per_tower_block_norms_equal_sqrt_weight():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 9)).astype(np.float32)
    t = rng.normal(size=(5, 57)).astype(np.float32)
    h = rng.normal(size=(5, 20)).astype(np.float32)
    tw = tower_weighted_from_towers(r, t, h, (0.2, 0.5, 0.3))
    assert tw.shape == (5, 86)
    assert tw.dtype == np.float32
    assert np.allclose(np.linalg.norm(tw[:, 0:9], axis=1), np.sqrt(0.2), atol=1e-5)
    assert np.allclose(np.linalg.norm(tw[:, 9:66], axis=1), np.sqrt(0.5), atol=1e-5)
    assert np.allclose(np.linalg.norm(tw[:, 66:86], axis=1), np.sqrt(0.3), atol=1e-5)


def test_zero_row_does_not_nan():
    r = np.zeros((1, 9), np.float32)
    t = np.ones((1, 57), np.float32)
    h = np.ones((1, 20), np.float32)
    tw = tower_weighted_from_towers(r, t, h, (0.2, 0.5, 0.3))
    assert not np.isnan(tw).any()


def test_build_arrays_overwrites_segments_sets_variant_preserves_genre(tmp_path):
    N = 4
    d = {}
    for seg in ("", "_start", "_mid", "_end"):
        d[f"X_sonic_rhythm{seg}"] = np.full((N, 9), 2.0, np.float32)
        d[f"X_sonic_timbre{seg}"] = np.full((N, 57), 3.0, np.float32)
        d[f"X_sonic_harmony{seg}"] = np.full((N, 20), 4.0, np.float32)
    d["X_sonic"] = np.zeros((N, 86), np.float32)
    for seg in ("start", "mid", "end"):
        d[f"X_sonic_{seg}"] = np.zeros((N, 86), np.float32)
    d["X_sonic_robust_whiten"] = np.zeros((N, 86), np.float32)
    d["X_genre_raw"] = np.ones((N, 3), np.float32)
    d["track_ids"] = np.array(["a", "b", "c", "d"], dtype=object)
    p = tmp_path / "art.npz"
    np.savez(p, **d)
    data = np.load(p, allow_pickle=True)

    out = build_tower_weighted_arrays(data, (0.2, 0.5, 0.3))

    assert str(out["X_sonic_variant"]) == "tower_weighted"
    assert bool(out["X_sonic_pre_scaled"]) is True
    assert out["X_sonic_tower_weighted"].shape == (N, 86)
    # segments are now the weighted vectors (no longer all-zero)
    for seg in ("start", "mid", "end"):
        assert np.linalg.norm(out[f"X_sonic_{seg}"]) > 0
    # genre + ids preserved byte-identical
    assert np.array_equal(out["X_genre_raw"], d["X_genre_raw"])
    assert np.array_equal(out["track_ids"], d["track_ids"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sonic_rebuild.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.features.sonic_rebuild'`.

- [ ] **Step 3: Write the implementation**

```python
# src/features/sonic_rebuild.py
"""Surgical tower-weighted sonic rebuild.

Recompute ONLY the sonic matrices of an existing artifact as per-tower
`sqrt(weight) * L2(tower)`, from per-tower matrices already stored in the npz.
Genre matrices and track order are copied byte-identical, so the dense-genre
sidecar stays valid and no DB / audio is read. See
docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(np.float64)
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    return mat / norms


def tower_weighted_from_towers(
    rhythm: np.ndarray,
    timbre: np.ndarray,
    harmony: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    """Per-tower L2-normalise, scale each by sqrt(weight), concatenate.

    Each output row's per-tower sub-vector has norm sqrt(weight), so the tower
    weighting is applied exactly and is invariant to the towers' raw scales.
    """
    w_r, w_t, w_h = (float(w) for w in weights)
    scales = np.sqrt(np.array([w_r, w_t, w_h], dtype=np.float64))
    out = np.concatenate(
        [
            scales[0] * _l2_rows(rhythm),
            scales[1] * _l2_rows(timbre),
            scales[2] * _l2_rows(harmony),
        ],
        axis=1,
    )
    return out.astype(np.float32)


def build_tower_weighted_arrays(
    data: Mapping[str, np.ndarray],
    weights: Tuple[float, float, float],
) -> dict:
    """Return a full set of npz arrays with sonic matrices rebuilt tower_weighted.

    Copies every key from ``data`` unchanged, then overwrites:
      - adds ``X_sonic_tower_weighted`` (full) — the variant key the loader selects
      - overwrites ``X_sonic_start`` / ``_mid`` / ``_end`` (loader reads these directly)
      - sets ``X_sonic_variant`` = "tower_weighted", ``X_sonic_pre_scaled`` = True
    The raw ``X_sonic`` key and all genre/track/per-tower keys are left intact.
    """
    out = {k: data[k] for k in data.files}  # type: ignore[attr-defined]

    out["X_sonic_tower_weighted"] = tower_weighted_from_towers(
        data["X_sonic_rhythm"], data["X_sonic_timbre"], data["X_sonic_harmony"], weights
    )
    for seg in ("start", "mid", "end"):
        out[f"X_sonic_{seg}"] = tower_weighted_from_towers(
            data[f"X_sonic_rhythm_{seg}"],
            data[f"X_sonic_timbre_{seg}"],
            data[f"X_sonic_harmony_{seg}"],
            weights,
        )
    out["X_sonic_variant"] = np.array("tower_weighted")
    out["X_sonic_pre_scaled"] = np.array(True)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sonic_rebuild.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/features/sonic_rebuild.py tests/unit/test_sonic_rebuild.py
git commit -m "feat(sonic): tower_weighted rebuild helpers"
```

---

### Task 2: Rebuild CLI script

**Files:**
- Create: `scripts/rebuild_sonic_tower_weighted.py`
- Test: `tests/unit/test_rebuild_sonic_cli.py`

- [ ] **Step 1: Write the failing test** (round-trip through a temp npz)

```python
# tests/unit/test_rebuild_sonic_cli.py
import numpy as np

from scripts.rebuild_sonic_tower_weighted import rebuild_artifact


def _make_artifact(path, N=4):
    d = {}
    for seg in ("", "_start", "_mid", "_end"):
        d[f"X_sonic_rhythm{seg}"] = np.random.default_rng(1).normal(size=(N, 9)).astype(np.float32)
        d[f"X_sonic_timbre{seg}"] = np.random.default_rng(2).normal(size=(N, 57)).astype(np.float32)
        d[f"X_sonic_harmony{seg}"] = np.random.default_rng(3).normal(size=(N, 20)).astype(np.float32)
    d["X_sonic"] = np.zeros((N, 86), np.float32)
    for seg in ("start", "mid", "end"):
        d[f"X_sonic_{seg}"] = np.zeros((N, 86), np.float32)
    d["X_genre_raw"] = np.ones((N, 3), np.float32)
    d["track_ids"] = np.array(["a", "b", "c", "d"], dtype=object)
    np.savez(path, **d)


def test_rebuild_writes_variant_and_backs_up(tmp_path):
    art = tmp_path / "data_matrices_step1.npz"
    _make_artifact(art)
    before = np.load(art, allow_pickle=True)["X_genre_raw"].copy()

    backup = rebuild_artifact(str(art), weights=(0.2, 0.5, 0.3), backup=True)

    assert backup is not None and backup.exists()
    out = np.load(art, allow_pickle=True)
    assert str(out["X_sonic_variant"]) == "tower_weighted"
    assert "X_sonic_tower_weighted" in out.files
    assert np.linalg.norm(out["X_sonic_start"]) > 0  # segments rebuilt
    assert np.array_equal(out["X_genre_raw"], before)  # genre preserved
    # backup holds the pre-rebuild content
    assert str(np.load(backup, allow_pickle=True)["X_sonic_variant"]) != "tower_weighted" \
        if "X_sonic_variant" in np.load(backup, allow_pickle=True).files else True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_rebuild_sonic_cli.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'rebuild_artifact'`.

- [ ] **Step 3: Write the implementation**

```python
# scripts/rebuild_sonic_tower_weighted.py
"""Surgically rebuild an artifact's sonic matrices as the tower_weighted variant.

Reads the existing npz, recomputes sonic matrices (full + start/mid/end) via
src.features.sonic_rebuild, backs up the original, and writes the rebuilt npz
in place (same stem -> dense-genre sidecar stays valid). No DB / audio access.

Usage:
    python scripts/rebuild_sonic_tower_weighted.py \
        --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.sonic_rebuild import build_tower_weighted_arrays  # noqa: E402


def _weights_from_config(config_path: str) -> Tuple[float, float, float]:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    tw = cfg["playlists"]["ds_pipeline"]["tower_weights"]
    return (float(tw["rhythm"]), float(tw["timbre"]), float(tw["harmony"]))


def rebuild_artifact(
    artifact: str,
    weights: Tuple[float, float, float],
    *,
    backup: bool = True,
) -> Optional[Path]:
    """Rebuild ``artifact`` in place as tower_weighted. Returns backup path or None."""
    path = Path(artifact)
    data = np.load(path, allow_pickle=True)
    out = build_tower_weighted_arrays(data, weights)

    backup_path: Optional[Path] = None
    if backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(path.name + f".bak_{ts}")
        shutil.copy2(path, backup_path)

    tmp = path.with_name(path.stem + ".rebuild.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        default="data/artifacts/beat3tower_32k/data_matrices_step1.npz",
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    weights = _weights_from_config(args.config)
    backup = rebuild_artifact(args.artifact, weights, backup=not args.no_backup)
    print(f"Rebuilt {args.artifact} as tower_weighted weights={weights}")
    if backup:
        print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_rebuild_sonic_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/rebuild_sonic_tower_weighted.py tests/unit/test_rebuild_sonic_cli.py
git commit -m "feat(sonic): CLI to rebuild artifact as tower_weighted (surgical, backup-first)"
```

---

### Task 3: Confirm the loader selects the tower_weighted variant

The loader (`src/features/artifacts.py:104-125`) is already generic: it reads `X_sonic_variant`, looks up `X_sonic_<variant>`, and sets `pre_scaled=True`. This task adds a regression test proving a tower_weighted artifact loads correctly. **No production code change is expected; if the test fails, fix the loader.**

**Files:**
- Test: `tests/unit/test_artifact_tower_weighted_load.py`
- (Modify only if test fails: `src/features/artifacts.py`)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_artifact_tower_weighted_load.py
import numpy as np

from src.features.artifacts import load_artifact_bundle


def test_tower_weighted_artifact_loads_pre_scaled(tmp_path):
    N = 4
    tw_full = np.random.default_rng(0).normal(size=(N, 86)).astype(np.float32)
    tw_start = np.random.default_rng(1).normal(size=(N, 86)).astype(np.float32)
    d = {
        "track_ids": np.array(["a", "b", "c", "d"], dtype=object),
        "artist_keys": np.array(["a", "b", "c", "d"], dtype=object),
        "track_artists": np.array(["A", "B", "C", "D"], dtype=object),
        "X_sonic": np.zeros((N, 86), np.float32),       # raw key (unselected)
        "X_sonic_tower_weighted": tw_full,              # variant key (selected)
        "X_sonic_start": tw_start,
        "X_sonic_mid": tw_start,
        "X_sonic_end": tw_start,
        "X_sonic_variant": np.array("tower_weighted"),
        "X_sonic_pre_scaled": np.array(True),
        "X_genre_raw": np.ones((N, 3), np.float32),
        "X_genre_smoothed": np.ones((N, 3), np.float32),
        "genre_vocab": np.array(["x", "y", "z"], dtype=object),
    }
    p = tmp_path / "twart.npz"
    np.savez(p, **d)

    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(p)

    assert b.sonic_variant == "tower_weighted"
    assert b.sonic_pre_scaled is True
    assert np.array_equal(b.X_sonic, tw_full)        # selected the variant key
    assert np.array_equal(b.X_sonic_start, tw_start)  # segment key used directly
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/unit/test_artifact_tower_weighted_load.py -v`
Expected: PASS. If it FAILS, read `src/features/artifacts.py:104-125`, make the variant selection set `X_sonic = data["X_sonic_tower_weighted"]` and `sonic_pre_scaled = True`, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_artifact_tower_weighted_load.py
git commit -m "test(sonic): loader selects tower_weighted variant + pre_scaled"
```

---

### Task 4: Phase-1 metrics script (tooling, used for baseline + after)

**Files:**
- Create: `scripts/sonic_phase1_metrics.py`
- Test: `tests/unit/test_sonic_phase1_metrics.py`

- [ ] **Step 1: Write the failing test** (pure metric on a synthetic matrix)

```python
# tests/unit/test_sonic_phase1_metrics.py
import numpy as np

from scripts.sonic_phase1_metrics import cosine_spread_to_seed, per_tower_contribution


def test_cosine_spread_keys_and_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 86)).astype(np.float32)
    s = cosine_spread_to_seed(X, seed_idx=0)
    assert set(s) == {"max", "p99", "p90", "median"}
    assert -1.0 <= s["median"] <= 1.0 and s["max"] <= 1.0 + 1e-6


def test_per_tower_contribution_reflects_weights():
    # rows with per-tower norms sqrt(0.2)/sqrt(0.5)/sqrt(0.3) -> contributions 0.2/0.5/0.3
    N = 10
    r = np.full((N, 9), np.sqrt(0.2 / 9), np.float32)
    t = np.full((N, 57), np.sqrt(0.5 / 57), np.float32)
    h = np.full((N, 20), np.sqrt(0.3 / 20), np.float32)
    X = np.concatenate([r, t, h], axis=1)
    c = per_tower_contribution(X)
    assert abs(c["timbre"] - 0.5) < 1e-3
    assert c["timbre"] > c["harmony"] > c["rhythm"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sonic_phase1_metrics.py -v`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Write the implementation**

```python
# scripts/sonic_phase1_metrics.py
"""Phase-1 sonic metrics: cosine spread + per-tower contribution + generation smoke.

Run on the CURRENT artifact (baseline) BEFORE editing runtime code, then again on
the rebuilt artifact (after). Writes JSON under docs/run_audits/sonic_phase1/.

    python scripts/sonic_phase1_metrics.py --artifact <path> --label baseline
    python scripts/sonic_phase1_metrics.py --artifact <path> --label after --generate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 86-dim tower layout
R, T, H = slice(0, 9), slice(9, 66), slice(66, 86)

SEEDS = [
    "Green-House", "Boards of Canada", "Autechre", "Charli XCX",
    "Bill Evans", "Jean-Yves Thibaudet", "William Tyler", "Elliott Smith",
    "Duster", "Real Estate", "Slowdive", "Sonic Youth", "Minor Threat",
    "James Brown", "J Dilla", "Beyoncé", "Grouper",
]


def _l2(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def cosine_spread_to_seed(X: np.ndarray, seed_idx: int) -> Dict[str, float]:
    Xn = _l2(X.astype(np.float64))
    sims = np.sort(Xn @ Xn[seed_idx])[::-1][1:]  # drop self
    return {
        "max": float(sims[0]),
        "p99": float(np.percentile(sims, 99)),
        "p90": float(np.percentile(sims, 90)),
        "median": float(np.percentile(sims, 50)),
    }


def per_tower_contribution(X: np.ndarray) -> Dict[str, float]:
    """Mean fraction of squared row norm carried by each tower block."""
    Xs = X.astype(np.float64)
    e_r = (Xs[:, R] ** 2).sum(axis=1)
    e_t = (Xs[:, T] ** 2).sum(axis=1)
    e_h = (Xs[:, H] ** 2).sum(axis=1)
    tot = np.maximum(e_r + e_t + e_h, 1e-12)
    return {
        "rhythm": float(np.mean(e_r / tot)),
        "timbre": float(np.mean(e_t / tot)),
        "harmony": float(np.mean(e_h / tot)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="data/artifacts/beat3tower_32k/data_matrices_step1.npz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--generate", action="store_true",
                    help="also run a generation smoke for the 5 core seeds (uses current code)")
    args = ap.parse_args()

    from src.features.artifacts import load_artifact_bundle
    load_artifact_bundle.cache_clear()
    b = load_artifact_bundle(args.artifact)
    X = b.X_sonic
    artists = np.array([str(a) for a in b.track_artists])

    def medoid(name: str):
        idx = np.where(np.char.lower(artists) == name.lower())[0]
        if len(idx) == 0:
            return None
        sub = _l2(X[idx].astype(np.float64))
        c = sub.mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return int(idx[int(np.argmax(sub @ c))])

    report: Dict[str, object] = {
        "label": args.label,
        "artifact": args.artifact,
        "sonic_variant": str(b.sonic_variant),
        "sonic_pre_scaled": bool(b.sonic_pre_scaled),
        "per_tower_contribution": per_tower_contribution(X),
        "seeds": {},
    }
    for nm in SEEDS:
        s = medoid(nm)
        if s is None:
            report["seeds"][nm] = {"missing": True}  # type: ignore[index]
            continue
        report["seeds"][nm] = {  # type: ignore[index]
            "track_id": str(b.track_ids[s]),
            "title": str(b.track_titles[s]) if b.track_titles is not None else None,
            "cosine_spread": cosine_spread_to_seed(X, s),
        }

    if args.generate:
        from tests.support.gui_fidelity import generate_like_gui
        gen: Dict[str, object] = {}
        for nm in ["Charli XCX", "Real Estate", "Bill Evans", "Beach House", "Minor Threat"]:
            s = medoid(nm)
            if s is None:
                gen[nm] = {"missing": True}
                continue
            try:
                res = generate_like_gui(
                    seeds=[str(b.track_ids[s])],
                    cohesion_mode="narrow", genre_mode="narrow",
                    sonic_mode="narrow", pace_mode="narrow",
                    artifact_path=args.artifact, length=20,
                )
                tids = [str(t) for t in res.track_ids]
                arts = [str(b.track_artists[b.track_id_to_index[t]]) for t in tids
                        if t in b.track_id_to_index]
                gen[nm] = {"length": len(tids), "distinct_artists": len(set(arts))}
            except Exception as exc:  # record, don't crash the metrics run
                gen[nm] = {"error": repr(exc)}
        report["generation_smoke"] = gen

    out_dir = ROOT / "docs" / "run_audits" / "sonic_phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(report["per_tower_contribution"], indent=2))


if __name__ == "__main__":
    main()
```

> NOTE on `generate_like_gui`: confirm its return object exposes `track_ids` (read
> `tests/support/gui_fidelity.py` + the `DsRunResult` dataclass). If the attribute name
> differs, adjust the two `res.track_ids` lines only. The generation block is wrapped in
> try/except so a signature mismatch records an error string instead of aborting the metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_sonic_phase1_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/sonic_phase1_metrics.py tests/unit/test_sonic_phase1_metrics.py
git commit -m "feat(sonic): Phase-1 metrics (cosine spread + per-tower contribution + gen smoke)"
```

---

### Task 5: Capture BASELINE metrics (unmodified code + current artifact)

This MUST run before Task 6 (runtime edit) and Task 7 (artifact swap) so the generation
smoke reflects true current production.

**Files:** none (produces `docs/run_audits/sonic_phase1/baseline.json`)

- [ ] **Step 1: Run baseline metrics with generation smoke**

Run:
```bash
python scripts/sonic_phase1_metrics.py \
  --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz \
  --label baseline --generate
```
Expected: writes `docs/run_audits/sonic_phase1/baseline.json`; printed
`per_tower_contribution` shows timbre ≈ its dim-share (NOT 0.5) — i.e. weighting not applied;
`sonic_variant` = `robust_whiten`. Generation smoke records non-zero `length` for the 5 seeds.

- [ ] **Step 2: Commit the baseline artifact-of-record**

```bash
git add docs/run_audits/sonic_phase1/baseline.json
git commit -m "chore(sonic): capture Phase-1 baseline metrics (pre-fix)"
```

---

### Task 6: Runtime fix — use pre-scaled artifacts directly

**Files:**
- Modify: `src/playlist/pipeline/embedding_setup.py` (the `X_sonic_for_embed` selection block, ~lines 73-86, and the `build_hybrid_embedding(...)` call, ~lines 203-204)

- [ ] **Step 1: Replace the variant-selection block**

Find (around lines 73-88):

```python
    variant_stats: Dict[str, Any] = {"variant": resolved_variant}
    X_sonic_for_embed = bundle.X_sonic
    pre_scaled_sonic = False
    if bundle.X_sonic is not None:
        if (
            getattr(bundle, "sonic_variant", None) == resolved_variant
            and getattr(bundle, "sonic_pre_scaled", False)
        ):
            X_sonic_for_embed = bundle.X_sonic
            variant_stats = {"variant": resolved_variant, "pre_scaled": True}
        else:
            X_sonic_for_embed, variant_stats = compute_sonic_variant_matrix(
                bundle.X_sonic, resolved_variant, l2=False
            )
    else:
        raise ValueError("Artifact missing X_sonic matrix.")
```

Replace with:

```python
    variant_stats: Dict[str, Any] = {"variant": resolved_variant}
    X_sonic_for_embed = bundle.X_sonic
    pre_scaled_sonic = False
    if bundle.X_sonic is not None:
        if getattr(bundle, "sonic_pre_scaled", False):
            # The artifact is the source of truth for its own preprocessing
            # (e.g. tower_weighted baked at build time). Use it directly; do NOT
            # re-apply a variant transform — that path no-ops via a dim-mismatch
            # fallback AND mislabels the space as un-scaled, causing a spurious
            # StandardScaler before the hybrid PCA. See
            # docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md.
            X_sonic_for_embed = bundle.X_sonic
            variant_stats = {
                "variant": getattr(bundle, "sonic_variant", resolved_variant),
                "pre_scaled": True,
            }
        else:
            X_sonic_for_embed, variant_stats = compute_sonic_variant_matrix(
                bundle.X_sonic, resolved_variant, l2=False
            )
    else:
        raise ValueError("Artifact missing X_sonic matrix.")
```

- [ ] **Step 2: Keep the hybrid sonic block PCA'd (balanced 32+32)**

Find (around lines 203-204):

```python
        pre_scaled_sonic=pre_scaled_sonic,
        use_pca_sonic=not pre_scaled_sonic,
```

Replace with:

```python
        pre_scaled_sonic=pre_scaled_sonic,
        # Always PCA the sonic block to 32 dims to keep the hybrid balanced with
        # the 32-dim genre block; pre_scaled_sonic only skips the redundant
        # StandardScaler inside _fit_pca when the space is already scaled.
        use_pca_sonic=True,
```

- [ ] **Step 3: Run the fast test suite**

Run: `pytest -m "not slow" -q`
Expected: green. If any test asserts the OLD behavior (e.g. expects a StandardScaler on a
pre-scaled artifact, or `use_pca_sonic=False`), read it: if it encodes the drift bug, update
the test to the corrected behavior and note why in the commit; if it encodes a real contract,
stop and reconsider.

- [ ] **Step 4: Commit**

```bash
git add src/playlist/pipeline/embedding_setup.py
git commit -m "fix(sonic): use pre-scaled artifact preprocessing directly; keep hybrid sonic PCA balanced"
```

---

### Task 7: Execute the artifact rebuild (backup + in-place swap)

**Files:** none (rewrites the production artifact; writes a `.bak_<ts>` alongside)

- [ ] **Step 1: Rebuild**

Run:
```bash
python scripts/rebuild_sonic_tower_weighted.py \
  --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz
```
Expected: prints `Rebuilt ... as tower_weighted weights=(0.2, 0.5, 0.3)` and a `Backup:` line.

- [ ] **Step 2: Sanity-check the rebuilt artifact loads as tower_weighted**

Run:
```bash
python -c "from src.features.artifacts import load_artifact_bundle as L; L.cache_clear(); b=L('data/artifacts/beat3tower_32k/data_matrices_step1.npz'); print(b.sonic_variant, b.sonic_pre_scaled, b.X_sonic.shape, b.X_sonic_start.shape)"
```
Expected: `tower_weighted True (39887, 86) (39887, 86)`.

- [ ] **Step 3: Verify the dense-genre sidecar still validates** (track_ids unchanged → no warning)

Run:
```bash
python -c "import logging; logging.basicConfig(level=logging.INFO); from src.features.artifacts import load_artifact_bundle as L; L.cache_clear(); b=L('data/artifacts/beat3tower_32k/data_matrices_step1.npz'); print('dense:', None if b.X_genre_dense is None else b.X_genre_dense.shape)"
```
Expected: `dense: (39887, 64)` and NO "sidecar track_ids mismatch" warning.

---

### Task 8: Capture AFTER metrics + validate against baseline

**Files:** produces `docs/run_audits/sonic_phase1/after.json` and `findings.md`

- [ ] **Step 1: Run after-metrics with generation smoke**

Run:
```bash
python scripts/sonic_phase1_metrics.py \
  --artifact data/artifacts/beat3tower_32k/data_matrices_step1.npz \
  --label after --generate
```
Expected: `sonic_variant` = `tower_weighted`; `per_tower_contribution` timbre ≈ 0.50,
harmony ≈ 0.30, rhythm ≈ 0.20 (weighting now applied); cosine `max` for Real Estate ≈ 0.57
and James Brown ≈ 0.58 (vs baseline ≈ 0.47 / 0.46); generation smoke `length` = 20 for all 5
seeds with no `error` key.

- [ ] **Step 2: Run the fast suite again on the swapped artifact**

Run: `pytest -m "not slow" -q`
Expected: green.

- [ ] **Step 3: Write the findings doc comparing baseline vs after**

Create `docs/run_audits/sonic_phase1/findings.md` with: the per-tower contribution
before/after (proving weighting is now active), the cosine-spread table (max/p99/p90 per seed,
baseline vs after), the random-pair trade-off note, and the generation-smoke distinct-artist
counts before/after. State plainly whether each validation criterion in the spec passed.
If any seed became infeasible or distinct-artist counts collapsed, STOP and report — do not
declare success.

- [ ] **Step 4: Commit metrics + findings + the rebuilt artifact**

```bash
git add docs/run_audits/sonic_phase1/after.json docs/run_audits/sonic_phase1/findings.md
git add data/artifacts/beat3tower_32k/data_matrices_step1.npz
git commit -m "feat(sonic): rebuild production artifact as tower_weighted + Phase-1 after metrics"
```

> The `.bak_<ts>` backup is intentionally NOT committed (large, redundant with git history of
> the artifact). Keep it on disk until Phase 2 confirms the fix; it is the one-step rollback.

---

### Task 9: Update config + docs to match reality

**Files:**
- Modify: `config.yaml` (`playlists.sonic.sim_variant`)
- Modify: `docs/CONFIG.md` (variant list)
- Modify: `CLAUDE.md` (sonic gotcha / principle note)
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md` ("Knob 0")

- [ ] **Step 1: Point config at the shipped variant**

In `config.yaml`, change:
```yaml
    sim_variant: tower_pca           # Preprocessing variant
```
to:
```yaml
    # Production artifact bakes tower weighting at build time (tower_weighted variant);
    # the pipeline uses the artifact's pre-scaled space directly. See
    # docs/superpowers/specs/2026-06-01-sonic-tower-weighted-fix-design.md.
    sim_variant: tower_weighted
```

- [ ] **Step 2: Correct `docs/CONFIG.md`**

Change the variant list so `tower_weighted` is described as the shipped default
("per-tower L2 + √weight, baked at build time, no global whitening") and `tower_pca`/
`robust_whiten` are listed as alternatives. Remove the stale "(default)" from `tower_pca`.

- [ ] **Step 3: Note the change in `CLAUDE.md`**

Under the sonic gotchas, add a bullet: the 0.20/0.50/0.30 tower weighting is **baked into the
`tower_weighted` artifact at build time** (was previously inert under the shipped
`robust_whiten` artifact — see the Phase-1 fix); rebuild via
`scripts/rebuild_sonic_tower_weighted.py` after any artifact rebuild that resets the variant.

- [ ] **Step 4: Note in `docs/PLAYLIST_ORDERING_TUNING.md` "Knob 0"**

Add that the invariant `transition_weights == tower_weights` is now satisfied **structurally**:
both full and start/end vectors are tower-weighted at build time, so `apply_transition_weights`
is a legitimate no-op rather than a silent fallback.

- [ ] **Step 5: Run the fast suite (catch any doc-driven config test)**

Run: `pytest -m "not slow" -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add config.yaml docs/CONFIG.md CLAUDE.md docs/PLAYLIST_ORDERING_TUNING.md
git commit -m "docs(sonic): config + docs reflect tower_weighted artifact (weighting now active)"
```

---

### Task 10: Update memory + hand off to Phase 2

**Files:**
- Modify: `~/.claude/projects/.../memory/project_sonic_tower_weights_inert.md` (mark Phase 1 done)

- [ ] **Step 1: Update the memory note**

Append a "RESOLVED (Phase 1, 2026-06-01)" section: the artifact is now `tower_weighted`,
runtime uses pre-scaled directly, weighting verified active (per-tower contribution
0.20/0.50/0.30), cosine top-end widened per `docs/run_audits/sonic_phase1/findings.md`.
Phase 2 (audition) validates by ear and decides whether to pursue raw-137.

- [ ] **Step 2: Final verification statement**

Confirm: (a) `pytest -m "not slow"` green; (b) `after.json` shows weighting active and seeds
feasible; (c) backup `.bak_<ts>` exists on disk. Report results plainly (with numbers) — do
not assert success beyond what the metrics show. Then proceed to the Phase 2 plan
(`docs/superpowers/specs/2026-06-01-sonic-neighborhood-validation-design.md`).

---

## Self-Review

**Spec coverage:** chosen variant → Task 1; surgical rebuild (full+segments, genre preserved,
sidecar valid) → Tasks 2,7; runtime pre-scaled-direct + hybrid balance decision → Task 6;
baseline-before-edit → Task 5; validation (smoke, cosine spread, weighting active, no ordering
regressions, suite green) → Tasks 4,8; config+docs → Task 9; memory/handoff → Task 10. All spec
sections mapped.

**Placeholder scan:** no TBD/TODO; the one external-shape dependency (`DsRunResult.track_ids`)
is flagged with a concrete fallback (try/except records an error string) rather than left vague.

**Type consistency:** `tower_weighted_from_towers(rhythm, timbre, harmony, weights)` and
`build_tower_weighted_arrays(data, weights)` are used with identical signatures across Tasks 1,2;
`rebuild_artifact(artifact, weights, *, backup)` consistent across script + test; metrics
`cosine_spread_to_seed(X, seed_idx)` / `per_tower_contribution(X)` consistent across script +
test. Tower slices `[0:9]/[9:66]/[66:86]` used identically in Tasks 1 and 4.
