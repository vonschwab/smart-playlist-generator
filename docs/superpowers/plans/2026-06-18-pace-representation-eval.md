# Pace-representation eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research harness that ranks candidate pace representations by how well their pairwise distance separates continuously-mixed-album adjacency (pace-compatible) from cross-register pairs (big swings), and run Pass 1 (the automated screen).

**Architecture:** Pure-Python harness in `scripts/research/pace_eval_*`, no production code touched. Four focused modules — metrics (z-score/distance/AUC), corpus (flow-tagged album list → resolved tracks + pair sets), features (candidate vectors from the energy sidecar + artifact + metadata.db, reusing `bpm_loader`), and run (orchestrator → `results_pass1.tsv` + findings). TDD on the pure cores with synthetic inputs; the real-data Pass-1 run is a controller step pointing at the main checkout's `data/`.

**Tech Stack:** Python 3.11+, numpy, sqlite3 (stdlib), pytest. No essentia (energy is read from the prebuilt sidecar). No scipy (AUC computed directly).

## Global Constraints

- **Research-only:** touches nothing in production code paths; output confined to `docs/run_audits/pace_axis_eval/` (gitignored). metadata.db opened **read-only**.
- **Non-circular:** candidate distances are scored against album-adjacency / register truth, never against another candidate's space. `register` labels are coarse human/genre buckets (high/mid/low), NOT derived from any candidate feature.
- **Distributions, not means:** every result reports n + min/p10/p50/p90 per pair-set tier, plus AUC. State N.
- **Library-wide z-score:** each scalar feature is z-scored over ALL artifact tracks (not just the corpus), so the scale matches production; rhythm_tower z-scored per-dim.
- **Determinism:** all randomness via a seeded `random.Random(seed)`; reruns reproduce `results_pass1.tsv`.
- **Ratified candidate menu (exact names):** `rhythm_tower`, `perceptual_bpm`, `onset_rate`, `beat_strength`, `arousal_p50`, `danceability`, `energy_pair`, `energy_dist`, `energy_onset`, `pace_full`. (`pace_tuned` is Pass 3, NOT in this plan.)
- **Discriminator:** the hard test is `adjacent` < `non_adjacent_same_album` (gradient albums only); `random_cross` (cross-register) is the easy floor.
- **pytest:** `python -m pytest <path> -q`; NEVER pipe through tail/head.
- **Worktree:** branch `worktree-pace-representation-eval`. Commit per task.

## File Structure

- **Create** `scripts/research/pace_eval_metrics.py` — pure math: z-score params/apply, weighted Euclidean, AUC, distribution. [Task 1]
- **Create** `scripts/research/pace_eval_corpus.py` — `AlbumSpec`, the `CORPUS` list, `resolve_corpus`, `build_pairs`, `write_corpus_tsv`. [Task 2]
- **Create** `scripts/research/pace_eval_features.py` — `CANDIDATES` registry, `load_raw_features`, `zscore_features`, `candidate_vectors`. [Task 3]
- **Create** `scripts/research/pace_eval_run.py` — pure `score_candidates` + IO `run_pass1` (writes results + findings). [Task 4]
- **Create** tests: `tests/unit/test_pace_eval_metrics.py`, `test_pace_eval_corpus.py`, `test_pace_eval_features.py`, `test_pace_eval_run.py`. [Tasks 1–4]
- **Task 5:** controller real-data run + gates + learnings-log update (no new files).

---

### Task 1: `pace_eval_metrics.py` — pure math

**Files:**
- Create: `scripts/research/pace_eval_metrics.py`
- Test: `tests/unit/test_pace_eval_metrics.py`

**Interfaces:**
- Produces: `zscore_params(values)->(mean,std)`; `apply_zscore(x,mean,std)`; `weighted_euclidean(a,b,weights=None)->float`; `auc_pos_below_neg(pos,neg)->float` (P(pos<neg), higher=better, NaN-safe, tie=0.5); `distribution(distances)->dict(n,min,p10,p50,p90)`.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_pace_eval_metrics.py`:

```python
import numpy as np
from scripts.research import pace_eval_metrics as m


def test_zscore_params_ignores_nan_and_zero_std():
    assert m.zscore_params(np.array([1.0, 1.0, 1.0])) == (1.0, 1.0)  # std 0 -> 1
    mean, std = m.zscore_params(np.array([0.0, 2.0, np.nan, 4.0]))
    assert mean == 2.0 and std > 0


def test_weighted_euclidean_nan_propagates():
    assert np.isnan(m.weighted_euclidean(np.array([np.nan]), np.array([0.0])))
    assert m.weighted_euclidean(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_auc_perfect_and_random():
    # all pos distances below all neg -> AUC 1.0
    assert m.auc_pos_below_neg(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
    # interleaved -> 0.5
    assert m.auc_pos_below_neg(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5


def test_distribution_percentiles():
    d = m.distribution(np.array([0.0, 1.0, 2.0, 3.0, 4.0, np.nan]))
    assert d["n"] == 5 and d["min"] == 0.0 and d["p50"] == 2.0
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_eval_metrics.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** `scripts/research/pace_eval_metrics.py`:

```python
"""Pure metrics for the pace-representation eval (no IO)."""
from __future__ import annotations

import numpy as np


def zscore_params(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return (0.0, 1.0)
    mean = float(finite.mean())
    std = float(finite.std())
    return (mean, 1.0 if std == 0.0 else std)


def apply_zscore(x, mean: float, std: float):
    return (np.asarray(x, dtype=float) - mean) / (std if std else 1.0)


def weighted_euclidean(a, b, weights=None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    d = a - b
    if weights is not None:
        d = d * np.sqrt(np.asarray(weights, dtype=float))
    return float(np.sqrt(np.sum(d * d)))


def auc_pos_below_neg(pos, neg) -> float:
    """P(random positive distance < random negative distance); ties=0.5."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    less = float((pos[:, None] < neg[None, :]).sum())
    ties = float((pos[:, None] == neg[None, :]).sum())
    return (less + 0.5 * ties) / (pos.size * neg.size)


def distribution(distances) -> dict:
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "min": float("nan"), "p10": float("nan"),
                "p50": float("nan"), "p90": float("nan")}
    return {
        "n": int(d.size),
        "min": float(d.min()),
        "p10": float(np.percentile(d, 10)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_eval_metrics.py -q` → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/research/pace_eval_metrics.py tests/unit/test_pace_eval_metrics.py
git commit -m "feat(pace-eval): pure metrics (zscore/distance/auc/distribution)"
```

---

### Task 2: `pace_eval_corpus.py` — corpus + pairs

**Files:**
- Create: `scripts/research/pace_eval_corpus.py`
- Test: `tests/unit/test_pace_eval_corpus.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `AlbumSpec` dataclass; `CORPUS: list[AlbumSpec]`; `CorpusTrack(track_id, album_key, track_no, flow_type, register)`; `resolve_corpus(conn, specs=CORPUS)->(list[CorpusTrack], dict[album_key,int])`; `build_pairs(tracks, *, seed=13, n_random=2000)->dict[str,list[tuple[str,str]]]` keys `adjacent`/`non_adjacent_same_album`/`random_cross`; `write_corpus_tsv(path, tracks)`.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_pace_eval_corpus.py`:

```python
import sqlite3
from scripts.research import pace_eval_corpus as c


def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, file_path TEXT)")
    conn.executemany("INSERT INTO tracks VALUES (?,?,?,?)", rows)
    return conn


def test_resolve_orders_by_filename_trackno_and_usable_range():
    conn = _db([
        ("t2", "Daft Punk", "Discovery", "/m/02 - Aerodynamic.flac"),
        ("t1", "Daft Punk", "Discovery", "/m/01 - One More Time.flac"),
        ("t3", "Daft Punk", "Discovery", "/m/03 - Digital Love.flac"),
    ])
    spec = c.AlbumSpec(key="disc", artist_like="%Daft Punk%", album_like="%Discovery%",
                       flow_type="gradient_flow", register="high", usable_first=1, usable_last=2)
    tracks, counts = c.resolve_corpus(conn, [spec])
    assert [t.track_id for t in tracks] == ["t1", "t2"]  # ordered, t3 outside usable range
    assert counts["disc"] == 2


def test_build_pairs_adjacent_nonadjacent_and_cross_register():
    tracks = [
        c.CorpusTrack("a1", "A", 1, "gradient_flow", "high"),
        c.CorpusTrack("a2", "A", 2, "gradient_flow", "high"),
        c.CorpusTrack("a3", "A", 3, "gradient_flow", "high"),
        c.CorpusTrack("b1", "B", 1, "tight_continuous", "low"),
        c.CorpusTrack("b2", "B", 2, "tight_continuous", "low"),
    ]
    pairs = c.build_pairs(tracks, seed=1, n_random=50)
    assert ("a1", "a2") in pairs["adjacent"] and ("a2", "a3") in pairs["adjacent"]
    assert ("b1", "b2") in pairs["adjacent"]
    # non-adjacent only from gradient album A (a1,a3); none from tight album B
    assert ("a1", "a3") in pairs["non_adjacent_same_album"]
    assert all(p not in pairs["non_adjacent_same_album"] for p in [("b1", "b2")])
    # random_cross pairs are cross-register only
    reg = {t.track_id: t.register for t in tracks}
    assert all(reg[x] != reg[y] for x, y in pairs["random_cross"])
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_eval_corpus.py -q` → FAIL.

- [ ] **Step 3: Implement** `scripts/research/pace_eval_corpus.py`:

```python
"""Flow-tagged gold corpus + pair-set construction for the pace eval."""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AlbumSpec:
    key: str
    artist_like: str
    album_like: str
    flow_type: str          # tight_continuous | gradient_flow | flat_uniform_mix
    register: str           # high | mid | low
    path_contains: str | None = None
    usable_first: int | None = None
    usable_last: int | None = None
    expected: int | None = None


# Register-balanced, flow_type-tagged. LIKE patterns + expected counts are
# verified against the live DB in Task 5 (adjust patterns/path_contains there).
CORPUS: list[AlbumSpec] = [
    AlbumSpec("renaissance", "%Beyonc%", "%RENAISSANCE%", "flat_uniform_mix", "high", path_contains="Explicit", expected=16),
    AlbumSpec("discovery", "%Daft Punk%", "%Discovery%", "gradient_flow", "high", expected=14),
    AlbumSpec("avalanches", "%Avalanches%", "%Since I Left You%", "tight_continuous", "high", path_contains="Disc 1", usable_first=1, usable_last=18, expected=18),
    AlbumSpec("lcd_tih", "%LCD Soundsystem%", "%This Is Happening%", "gradient_flow", "high", usable_first=1, usable_last=9, expected=9),
    AlbumSpec("caribou_swim", "%Caribou%", "%Swim%", "gradient_flow", "high", usable_first=1, usable_last=9, expected=9),
    AlbumSpec("donuts", "%Dilla%", "%Donuts%", "tight_continuous", "mid", expected=31),
    AlbumSpec("voodoo", "%D'Angelo%", "%Voodoo%", "gradient_flow", "mid", expected=13),
    AlbumSpec("dub_roots", "%King Tubby%", "%Dub From the Roots%", "tight_continuous", "mid", expected=16),
    AlbumSpec("boc_mhtrtc", "%Boards of Canada%", "%Right To Children%", "gradient_flow", "mid", expected=18),
    AlbumSpec("beach_bloom", "%Beach House%", "%Bloom%", "gradient_flow", "mid", expected=10),
    AlbumSpec("eno_onland", "%Brian Eno%", "%On Land%", "tight_continuous", "low", expected=8),
    AlbumSpec("hiroshi_green", "%Yoshimura%", "%Green%", "tight_continuous", "low", expected=8),
    AlbumSpec("sotl_aspera", "%Stars of the Lid%", "%Per Aspera%", "tight_continuous", "low", expected=6),
]

_TRACKNO = re.compile(r"^\s*(\d+)")


@dataclass(frozen=True)
class CorpusTrack:
    track_id: str
    album_key: str
    track_no: int
    flow_type: str
    register: str


def _track_no(file_path: str | None) -> int | None:
    base = os.path.basename(file_path or "")
    match = _TRACKNO.match(base)
    return int(match.group(1)) if match else None


def resolve_corpus(conn, specs: list[AlbumSpec] = CORPUS):
    out: list[CorpusTrack] = []
    counts: dict[str, int] = {}
    for spec in specs:
        rows = conn.execute(
            "SELECT track_id, file_path FROM tracks "
            "WHERE artist LIKE ? AND album LIKE ? AND file_path IS NOT NULL",
            (spec.artist_like, spec.album_like),
        ).fetchall()
        items: list[tuple[int, str]] = []
        for tid, fp in rows:
            if spec.path_contains and spec.path_contains.lower() not in (fp or "").lower():
                continue
            tn = _track_no(fp)
            if tn is None:
                continue
            if spec.usable_first is not None and tn < spec.usable_first:
                continue
            if spec.usable_last is not None and tn > spec.usable_last:
                continue
            items.append((tn, str(tid)))
        items.sort()
        seen: set[int] = set()
        ordered: list[tuple[int, str]] = []
        for tn, tid in items:
            if tn in seen:
                continue  # dedup duplicate rips at same track number
            seen.add(tn)
            ordered.append((tn, tid))
        counts[spec.key] = len(ordered)
        for tn, tid in ordered:
            out.append(CorpusTrack(tid, spec.key, tn, spec.flow_type, spec.register))
    return out, counts


def build_pairs(tracks: list[CorpusTrack], *, seed: int = 13, n_random: int = 2000):
    by_album: dict[str, list[CorpusTrack]] = {}
    for t in tracks:
        by_album.setdefault(t.album_key, []).append(t)
    for k in by_album:
        by_album[k].sort(key=lambda t: t.track_no)

    adjacent: list[tuple[str, str]] = []
    non_adjacent: list[tuple[str, str]] = []
    for ts in by_album.values():
        ids = [t.track_id for t in ts]
        for i in range(len(ids) - 1):
            adjacent.append((ids[i], ids[i + 1]))
        if ts and ts[0].flow_type == "gradient_flow":
            for i in range(len(ids)):
                for j in range(i + 2, len(ids)):
                    non_adjacent.append((ids[i], ids[j]))

    register_of = {t.track_id: t.register for t in tracks}
    all_ids = [t.track_id for t in tracks]
    rng = random.Random(seed)
    random_cross: list[tuple[str, str]] = []
    tries = 0
    seen_pairs: set[tuple[str, str]] = set()
    while len(random_cross) < n_random and tries < n_random * 40:
        tries += 1
        a, b = rng.choice(all_ids), rng.choice(all_ids)
        if a == b or register_of[a] == register_of[b]:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        random_cross.append((a, b))
    return {"adjacent": adjacent, "non_adjacent_same_album": non_adjacent,
            "random_cross": random_cross}


def write_corpus_tsv(path: str, tracks: list[CorpusTrack]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("track_id\talbum_key\ttrack_no\tflow_type\tregister\n")
        for t in tracks:
            f.write(f"{t.track_id}\t{t.album_key}\t{t.track_no}\t{t.flow_type}\t{t.register}\n")
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_eval_corpus.py -q` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/research/pace_eval_corpus.py tests/unit/test_pace_eval_corpus.py
git commit -m "feat(pace-eval): flow-tagged corpus + pair-set construction"
```

---

### Task 3: `pace_eval_features.py` — candidate vectors

**Files:**
- Create: `scripts/research/pace_eval_features.py`
- Test: `tests/unit/test_pace_eval_features.py`

**Interfaces:**
- Consumes: `pace_eval_metrics.zscore_params`/`apply_zscore`.
- Produces:
  - `SCALAR_KEYS: list[str]` = `["log_bpm_trusted","onset_rate","beat_strength","arousal_p10","arousal_p50","arousal_p90","danceability"]`
  - `CANDIDATES: dict[str, list[str]]` (the 10 ratified names → feature keys; `rhythm_tower` uses the special key `"rhythm_tower"`).
  - `zscore_features(raw_scalars: dict[str, np.ndarray], raw_tower: np.ndarray) -> (zscored_scalars, ztower)` — library-wide.
  - `candidate_vector(name, idx, zscored_scalars, ztower) -> np.ndarray`.
  - `load_raw_features(track_ids, *, db_path, artifact_path, sidecar_path, bpm_trust_min_onset=0.5) -> (index, raw_scalars, raw_tower)` — reads sidecar + artifact + DB (reusing `bpm_loader`). `index: dict[track_id,int]`.

- [ ] **Step 1: Write the failing tests** `tests/unit/test_pace_eval_features.py`:

```python
import numpy as np
from scripts.research import pace_eval_features as fe


def test_candidates_cover_ratified_menu():
    assert set(fe.CANDIDATES) == {
        "rhythm_tower", "perceptual_bpm", "onset_rate", "beat_strength",
        "arousal_p50", "danceability", "energy_pair", "energy_dist",
        "energy_onset", "pace_full",
    }


def test_zscore_and_candidate_vector():
    raw_scalars = {k: np.array([0.0, 2.0, 4.0]) for k in fe.SCALAR_KEYS}
    raw_tower = np.array([[0.0] * 9, [2.0] * 9, [4.0] * 9])
    zs, zt = fe.zscore_features(raw_scalars, raw_tower)
    # idx 1 is the mean -> zscore 0 for every scalar
    v = fe.candidate_vector("pace_full", 1, zs, zt)
    assert v.shape == (5,) and np.allclose(v, 0.0)
    assert fe.candidate_vector("energy_pair", 0, zs, zt).shape == (2,)
    assert fe.candidate_vector("rhythm_tower", 0, zs, zt).shape == (9,)


def test_candidate_vector_nan_when_feature_missing():
    raw_scalars = {k: np.array([np.nan, 1.0]) for k in fe.SCALAR_KEYS}
    raw_tower = np.zeros((2, 9))
    zs, zt = fe.zscore_features(raw_scalars, raw_tower)
    v = fe.candidate_vector("arousal_p50", 0, zs, zt)
    assert np.isnan(v).any()
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_eval_features.py -q` → FAIL.

- [ ] **Step 3: Implement** `scripts/research/pace_eval_features.py`:

```python
"""Candidate pace-representation feature vectors for the eval.

Reads the prebuilt energy sidecar + the beat3tower artifact + metadata.db.
Does NOT import essentia. Reuses bpm_loader for perceptual_bpm + onset_rate.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from scripts.research.pace_eval_metrics import apply_zscore, zscore_params

SCALAR_KEYS = [
    "log_bpm_trusted", "onset_rate", "beat_strength",
    "arousal_p10", "arousal_p50", "arousal_p90", "danceability",
]

CANDIDATES: dict[str, list[str]] = {
    "rhythm_tower": ["rhythm_tower"],
    "perceptual_bpm": ["log_bpm_trusted"],
    "onset_rate": ["onset_rate"],
    "beat_strength": ["beat_strength"],
    "arousal_p50": ["arousal_p50"],
    "danceability": ["danceability"],
    "energy_pair": ["arousal_p50", "danceability"],
    "energy_dist": ["arousal_p10", "arousal_p50", "arousal_p90", "danceability"],
    "energy_onset": ["arousal_p50", "danceability", "onset_rate"],
    "pace_full": ["arousal_p50", "danceability", "onset_rate", "log_bpm_trusted", "beat_strength"],
}


def zscore_features(raw_scalars: dict[str, np.ndarray], raw_tower: np.ndarray):
    zscored = {}
    for key, arr in raw_scalars.items():
        mean, std = zscore_params(arr)
        zscored[key] = apply_zscore(arr, mean, std)
    ztower = np.full_like(raw_tower, np.nan, dtype=float)
    for d in range(raw_tower.shape[1]):
        mean, std = zscore_params(raw_tower[:, d])
        ztower[:, d] = apply_zscore(raw_tower[:, d], mean, std)
    return zscored, ztower


def candidate_vector(name: str, idx: int, zscored_scalars: dict[str, np.ndarray],
                     ztower: np.ndarray) -> np.ndarray:
    if name == "rhythm_tower":
        return np.asarray(ztower[idx], dtype=float)
    keys = CANDIDATES[name]
    return np.array([float(zscored_scalars[k][idx]) for k in keys], dtype=float)


def load_raw_features(track_ids, *, db_path: str, artifact_path: str,
                      sidecar_path: str, bpm_trust_min_onset: float = 0.5):
    """Return (index, raw_scalars, raw_tower) aligned to track_ids order."""
    track_ids = [str(t) for t in track_ids]
    index = {tid: i for i, tid in enumerate(track_ids)}
    n = len(track_ids)

    # perceptual_bpm + onset via the production loader (reuse, DRY).
    from src.playlist.bpm_loader import load_bpm_arrays
    bpm = load_bpm_arrays(np.array(track_ids, dtype=object), db_path=db_path)
    perceptual_bpm = np.asarray(bpm["perceptual_bpm"], dtype=float)
    onset_rate = np.asarray(bpm["onset_rate"], dtype=float)
    log_bpm = np.log(np.where(perceptual_bpm > 0, perceptual_bpm, np.nan))
    untrusted = ~(onset_rate >= float(bpm_trust_min_onset))
    log_bpm_trusted = np.where(untrusted, np.nan, log_bpm)

    # beat_strength_median from DB.
    beat_strength = np.full(n, np.nan, dtype=float)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for start in range(0, n, 900):
            batch = track_ids[start:start + 900]
            ph = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"SELECT track_id, json_extract(sonic_features,"
                f"'$.full.rhythm.beat_strength_median') AS bs "
                f"FROM tracks WHERE track_id IN ({ph})", tuple(batch)
            ):
                pos = index.get(str(row["track_id"]))
                if pos is not None and row["bs"] is not None:
                    beat_strength[pos] = float(row["bs"])
    finally:
        conn.close()

    # rhythm_tower (9-dim) from the artifact, mapped to track_ids order.
    art = np.load(artifact_path, allow_pickle=True)
    art_ids = [str(t) for t in art["track_ids"]]
    art_pos = {tid: i for i, tid in enumerate(art_ids)}
    tower_src = np.asarray(art["X_sonic_rhythm"], dtype=float)
    raw_tower = np.full((n, tower_src.shape[1]), np.nan, dtype=float)
    for tid, i in index.items():
        j = art_pos.get(tid)
        if j is not None:
            raw_tower[i] = tower_src[j]

    # arousal/danceability from the sidecar, mapped to track_ids order.
    side = np.load(sidecar_path, allow_pickle=True)
    side_ids = [str(t) for t in side["track_ids"]]
    side_pos = {tid: i for i, tid in enumerate(side_ids)}

    def _side(col):
        src = np.asarray(side[col], dtype=float)
        out = np.full(n, np.nan, dtype=float)
        for tid, i in index.items():
            j = side_pos.get(tid)
            if j is not None:
                out[i] = src[j]
        return out

    raw_scalars = {
        "log_bpm_trusted": log_bpm_trusted,
        "onset_rate": onset_rate,
        "beat_strength": beat_strength,
        "arousal_p10": _side("arousal_p10"),
        "arousal_p50": _side("arousal_p50"),
        "arousal_p90": _side("arousal_p90"),
        "danceability": _side("danceability"),
    }
    return index, raw_scalars, raw_tower
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_eval_features.py -q` → 3 passed. (`load_raw_features` is exercised by Task 5's real run, not unit tests — it needs the real artifact/sidecar/DB.)

- [ ] **Step 5: Commit**

```bash
git add scripts/research/pace_eval_features.py tests/unit/test_pace_eval_features.py
git commit -m "feat(pace-eval): candidate feature vectors (reuse bpm_loader)"
```

---

### Task 4: `pace_eval_run.py` — scoring orchestrator

**Files:**
- Create: `scripts/research/pace_eval_run.py`
- Test: `tests/unit/test_pace_eval_run.py`

**Interfaces:**
- Consumes: `pace_eval_metrics` (auc/distribution/weighted_euclidean), `pace_eval_features` (CANDIDATES, candidate_vector), `pace_eval_corpus` (build_pairs).
- Produces:
  - `score_candidates(corpus_index, pairs, zscored_scalars, ztower) -> dict[name, dict]` (pure; per candidate: AUC adjacent-vs-random, AUC adjacent-vs-nonadjacent, distribution per tier).
  - `run_pass1(*, db_path, artifact_path, sidecar_path, out_dir, seed=13) -> dict` (IO: resolve corpus, load+zscore features, build pairs, score, write `results_pass1.tsv` + `findings_pass1.md`).

- [ ] **Step 1: Write the failing test** `tests/unit/test_pace_eval_run.py`:

```python
import numpy as np
from scripts.research import pace_eval_run as run


def test_score_candidates_ranks_separating_feature_high():
    # 4 corpus tracks; arousal_p50 cleanly separates adjacent (close) from random (far)
    ids = ["a1", "a2", "b1", "b2"]
    corpus_index = {t: i for i, t in enumerate(ids)}
    pairs = {
        "adjacent": [("a1", "a2"), ("b1", "b2")],
        "non_adjacent_same_album": [],
        "random_cross": [("a1", "b1"), ("a2", "b2")],
    }
    # arousal_p50: a-cluster ~0, b-cluster ~10 -> adjacent close, cross far
    zs = {k: np.zeros(4) for k in __import__("scripts.research.pace_eval_features",
                                             fromlist=["SCALAR_KEYS"]).SCALAR_KEYS}
    zs["arousal_p50"] = np.array([0.0, 0.1, 10.0, 10.1])
    zs["danceability"] = np.array([0.0, 0.1, 10.0, 10.1])
    zt = np.zeros((4, 9))
    res = run.score_candidates(corpus_index, pairs, zs, zt)
    assert res["arousal_p50"]["auc_adj_vs_random"] == 1.0
    assert res["arousal_p50"]["adjacent"]["n"] == 2
```

- [ ] **Step 2: Run to verify fail** — `python -m pytest tests/unit/test_pace_eval_run.py -q` → FAIL.

- [ ] **Step 3: Implement** `scripts/research/pace_eval_run.py`:

```python
"""Pass-1 scoring orchestrator for the pace-representation eval."""
from __future__ import annotations

import os

import numpy as np

from scripts.research.pace_eval_features import CANDIDATES, candidate_vector
from scripts.research.pace_eval_metrics import (
    auc_pos_below_neg,
    distribution,
    weighted_euclidean,
)


def _pair_distances(name, pairs_list, corpus_index, zs, zt) -> np.ndarray:
    out = []
    for a, b in pairs_list:
        ia, ib = corpus_index.get(a), corpus_index.get(b)
        if ia is None or ib is None:
            continue
        va = candidate_vector(name, ia, zs, zt)
        vb = candidate_vector(name, ib, zs, zt)
        out.append(weighted_euclidean(va, vb))
    return np.asarray(out, dtype=float)


def score_candidates(corpus_index, pairs, zscored_scalars, ztower) -> dict:
    results = {}
    for name in CANDIDATES:
        adj = _pair_distances(name, pairs["adjacent"], corpus_index, zscored_scalars, ztower)
        non = _pair_distances(name, pairs["non_adjacent_same_album"], corpus_index, zscored_scalars, ztower)
        rnd = _pair_distances(name, pairs["random_cross"], corpus_index, zscored_scalars, ztower)
        results[name] = {
            "auc_adj_vs_random": auc_pos_below_neg(adj, rnd),
            "auc_adj_vs_nonadj": auc_pos_below_neg(adj, non),
            "adjacent": distribution(adj),
            "non_adjacent_same_album": distribution(non),
            "random_cross": distribution(rnd),
        }
    return results


def run_pass1(*, db_path, artifact_path, sidecar_path, out_dir, seed=13) -> dict:
    import sqlite3

    from scripts.research.pace_eval_corpus import build_pairs, resolve_corpus, write_corpus_tsv
    from scripts.research.pace_eval_features import load_raw_features, zscore_features

    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        corpus_tracks, counts = resolve_corpus(conn)
    finally:
        conn.close()
    write_corpus_tsv(os.path.join(out_dir, "corpus.tsv"), corpus_tracks)

    # z-score over ALL artifact tracks (library-wide), then index the corpus.
    art = np.load(artifact_path, allow_pickle=True)
    all_ids = [str(t) for t in art["track_ids"]]
    index, raw_scalars, raw_tower = load_raw_features(
        all_ids, db_path=db_path, artifact_path=artifact_path, sidecar_path=sidecar_path)
    zs, zt = zscore_features(raw_scalars, raw_tower)

    corpus_index = {t.track_id: index[t.track_id] for t in corpus_tracks if t.track_id in index}
    pairs = build_pairs(corpus_tracks, seed=seed)
    results = score_candidates(corpus_index, pairs, zs, zt)

    # results_pass1.tsv
    with open(os.path.join(out_dir, "results_pass1.tsv"), "w", encoding="utf-8") as f:
        f.write("candidate\tauc_adj_vs_random\tauc_adj_vs_nonadj\t"
                "adj_p50\tnonadj_p50\trandom_p50\tadj_n\tnonadj_n\trandom_n\n")
        for name, r in sorted(results.items(), key=lambda kv: -(kv[1]["auc_adj_vs_random"] or 0)):
            f.write(f"{name}\t{r['auc_adj_vs_random']:.4f}\t{r['auc_adj_vs_nonadj']:.4f}\t"
                    f"{r['adjacent']['p50']:.4f}\t{r['non_adjacent_same_album']['p50']:.4f}\t"
                    f"{r['random_cross']['p50']:.4f}\t{r['adjacent']['n']}\t"
                    f"{r['non_adjacent_same_album']['n']}\t{r['random_cross']['n']}\n")

    return {"counts": counts, "n_corpus": len(corpus_tracks),
            "pairs": {k: len(v) for k, v in pairs.items()}, "results": results}
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/unit/test_pace_eval_run.py -q` → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/research/pace_eval_run.py tests/unit/test_pace_eval_run.py
git commit -m "feat(pace-eval): Pass-1 scoring orchestrator + results writer"
```

---

### Task 5: Real-data Pass-1 run + gates + learnings update (controller)

**Files:** none new (verification + report).

- [ ] **Step 1: Lint + types + full new tests**

Run: `ruff check scripts/research/pace_eval_*.py && mypy scripts/research/pace_eval_metrics.py scripts/research/pace_eval_run.py && python -m pytest tests/unit/test_pace_eval_*.py -q`
Expected: ruff clean; mypy clean; 10 tests pass. (Fix inline if not.)

- [ ] **Step 2: Run Pass 1 on real data** (point at the MAIN checkout's `data/`, which has metadata.db + artifact + energy sidecar; the worktree's `data/` lacks the gitignored artifacts):

```bash
python -c "
from scripts.research.pace_eval_run import run_pass1
import json
ROOT='C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3'
r=run_pass1(
  db_path=ROOT+'/data/metadata.db',
  artifact_path=ROOT+'/data/artifacts/beat3tower_32k/data_matrices_step1.npz',
  sidecar_path=ROOT+'/data/artifacts/beat3tower_32k/energy/energy_sidecar.npz',
  out_dir=ROOT+'/docs/run_audits/pace_axis_eval',
)
print('resolved counts:', json.dumps(r['counts']))
print('pairs:', r['pairs'])
"
```
Expected: per-album resolved counts roughly match `AlbumSpec.expected` (±a few). **If an album resolves to 0 or wildly off, the LIKE/path_contains pattern is wrong — fix that `AlbumSpec` and re-run.** Then read `docs/run_audits/pace_axis_eval/results_pass1.tsv`.

- [ ] **Step 3: Verify corpus resolution** — open `docs/run_audits/pace_axis_eval/corpus.tsv`; confirm all 13 albums present with sane track counts and that `register`/`flow_type` look right. Adjust `CORPUS` patterns if any album is missing/duplicated; re-run Step 2.

- [ ] **Step 4: Write findings** to `docs/run_audits/pace_axis_eval/findings_pass1.md`: the `results_pass1.tsv` ranked table, which candidates clearly lead/lag on `auc_adj_vs_random` AND `auc_adj_vs_nonadj`, the per-tier p50 separations, the resolved corpus counts (state N), and the 2–4 candidates that advance to Pass 2. Note any album whose pattern needed fixing.

- [ ] **Step 5: Update the learnings log** `docs/PACE_AXIS_LEARNINGS.md` (Decisions section): Pass-1 ranking result, the advancing candidates, any corpus surprises, and that Pass 2 (3-tier on gradient albums + per-register + blind human) is next.

- [ ] **Step 6: Commit**

```bash
git add scripts/research/pace_eval_metrics.py scripts/research/pace_eval_corpus.py scripts/research/pace_eval_features.py scripts/research/pace_eval_run.py docs/PACE_AXIS_LEARNINGS.md
git commit -m "feat(pace-eval): Pass-1 harness complete + results"
```
(`docs/run_audits/` is gitignored — corpus.tsv/results/findings stay local; the learnings log carries the narrative.)

---

## Self-Review

**Spec coverage:**
- Candidate menu (10 fixed) → Task 3 `CANDIDATES`. ✓
- flow_type-tagged register-balanced corpus → Task 2 `CORPUS`. ✓
- Pair sets (adjacent / non_adjacent gradient-only / random cross-register) → Task 2 `build_pairs`. ✓
- Library-wide z-score, weighted-Euclidean distance → Tasks 1+3. ✓
- Graded metric: AUC + per-tier distributions (not means) → Tasks 1+4. ✓
- Discriminator = adjacent<non_adjacent (gradient); random_cross floor → Task 4 `auc_adj_vs_nonadj` + `auc_adj_vs_random`. ✓
- Non-circular (register = human label; truth = album adjacency) → Task 2. ✓
- Data access via main-checkout paths (no junction) → Task 5 Step 2. ✓
- Determinism (seeded RNG) → Task 2 `build_pairs(seed=)`. ✓
- Research-only, run_audits output, DB read-only → Tasks 4/5. ✓
- Pass 2 (3-tier focus + blind human) / Pass 3 (tuned weights) → intentionally NOT in this plan (designed after Pass 1 narrows); `pace_tuned` excluded. ✓ (matches spec's coarse→fine, eval narrows over passes)

**Placeholder scan:** every code step has complete code; commands have expected output; corpus LIKE patterns are concrete with a Task-5 verification step for the inevitable pattern mismatches. No TBD.

**Type consistency:** `candidate_vector(name, idx, zscored_scalars, ztower)`, `score_candidates(corpus_index, pairs, zscored_scalars, ztower)`, `load_raw_features(...) -> (index, raw_scalars, raw_tower)`, `zscore_features(raw_scalars, raw_tower) -> (zscored, ztower)`, `build_pairs(...) -> {adjacent,non_adjacent_same_album,random_cross}` — names/shapes match across Tasks 1–4 and the tests.
