# Lossless Generation Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce pier-bridge playlist generation wall-clock with **bit-identical output** — same track IDs in the same order — proven per change by a golden replay diff.

**Architecture:** Build a deterministic replay + bit-diff harness first (capture the exact `generate_playlist_ds` inputs of real runs, freeze the output track IDs). Then apply a tiered catalog of compute-only speedups to `beam.py` and `pier_bridge_builder.py`, gating every one on "golden track IDs unchanged on all fixtures." Speed comes from computing the same result fewer times, never from searching less.

**Tech Stack:** Python 3.11, numpy, pytest (markers: `smoke`/`integration`/`slow`), cProfile.

## Global Constraints

- **Bit-identical output is the hard gate.** After every change, the ordered `track_ids` from all golden fixtures must be **exactly equal** to the frozen golden, and `min_transition`/`mean_transition` must match to full float precision (ΔT == 0). Any difference → revert the change or keep an exact-op variant. (Spec: `docs/superpowers/specs/2026-07-03-lossless-generation-speedup-design.md`.)
- **No lossy levers.** Do not change any live search knob (beam width, pool sizes, flex count, floors, weights). Removing genuinely dead / zero-weight work is allowed; changing a live weight is not.
- **Two-gate ship rule.** A change ships only if it passes the bit-diff gate AND either saves measurable time or is a net simplification. Drop changes that add complexity for no measured gain.
- **Confirm line anchors at edit time.** All `file:line` references are from static analysis and may have drifted; read the function before editing.
- **Worktree discipline.** All implementation happens in a dedicated worktree on its own branch (simultaneous sessions are the norm). Do not commit into the shared main checkout; stage explicit paths only.
- **Never touch `data/metadata.db` or the artifact.** This work is read-only against both.

---

## Universal verification block (referenced by every optimization task)

Every Tier 1/2/3 task ends with this exact cycle instead of a bespoke test — the golden harness (Task 1) is the test, written once:

```bash
# 1. Bit-diff gate — MUST pass on every fixture
python -m pytest tests/integration/test_lossless_speedup_golden.py -v -m integration
#    Expected: all fixtures PASS (track_ids identical, ΔT == 0)

# 2. Fast suite stays green
python -m pytest -q -m "not slow"

# 3. Timing (record before/after in the commit message)
python scripts/research/time_golden_replay.py --fixture herbie   # prints wall-clock
```

If step 1 shows any track_ids diff or ΔT != 0, the change is **not lossless** — revert it (or, for a Tier-2 float-reassociation, keep the exact-op variant) before committing.

---

## Task 1: Golden capture hook + replay/bit-diff harness

**Files:**
- Modify: `src/playlist/ds_pipeline_runner.py` (add env-gated capture at the top of `generate_playlist_ds`, ~line 115)
- Create: `tests/support/lossless_golden.py` (capture encoder + replay loader)
- Create: `tests/integration/test_lossless_speedup_golden.py` (the bit-diff test)
- Create: `tests/support/__init__.py` if absent (it exists — `tests/support/gui_fidelity.py` is there)

**Interfaces:**
- Produces: `dump_golden_inputs(kwargs: dict, path: str) -> None`, `load_golden(path: str) -> dict` (returns `{"kwargs": {...}, "track_ids": [...], "min_transition": float, "mean_transition": float}`), and `replay_golden(golden: dict) -> DsRunResult`.
- Consumes: `src.playlist.ds_pipeline_runner.generate_playlist_ds`.

- [ ] **Step 1: Write the capture encoder/loader with a unit test first**

Create `tests/support/lossless_golden.py`:

```python
"""Golden-fixture capture + replay for the lossless-speedup work.

Captures the EXACT kwargs entering ds_pipeline_runner.generate_playlist_ds
(the outermost deterministic seam: candidate-pool RNG is seeded downstream,
the beam is RNG-free) so a replay reproduces the identical playlist.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return {"__set__": sorted(str(x) for x in obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {"__dataclass__": type(obj).__name__, "fields": dataclasses.asdict(obj)}
    raise TypeError(f"Golden capture: non-serializable arg of type {type(obj)!r}; extend _json_default")


def _decode(obj: Dict[str, Any]) -> Any:
    if "__set__" in obj:
        return set(obj["__set__"])
    return obj  # __dataclass__ reconstruction handled in replay if ever needed


def dump_golden_inputs(kwargs: Dict[str, Any], track_ids, min_transition, mean_transition, path: str) -> None:
    payload = {
        "kwargs": kwargs,
        "track_ids": list(track_ids),
        "min_transition": None if min_transition is None else float(min_transition),
        "mean_transition": None if mean_transition is None else float(mean_transition),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default, indent=2)


def load_golden(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh, object_hook=_decode)


def replay_golden(golden: Dict[str, Any]):
    from src.playlist.ds_pipeline_runner import generate_playlist_ds
    return generate_playlist_ds(**golden["kwargs"])
```

Create `tests/test_lossless_golden_unit.py`:

```python
from tests.support.lossless_golden import _json_default, _decode
import json


def test_set_roundtrips_deterministically():
    encoded = json.dumps({"excluded": {"b", "a", "c"}}, default=_json_default)
    decoded = json.loads(encoded, object_hook=_decode)
    assert decoded["excluded"] == {"a", "b", "c"}


def test_non_serializable_raises():
    import pytest
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=_json_default)
```

- [ ] **Step 2: Run the unit test to verify it passes**

Run: `python -m pytest tests/test_lossless_golden_unit.py -v`
Expected: 2 passed.

- [ ] **Step 3: Add the env-gated capture hook in the runner**

In `src/playlist/ds_pipeline_runner.py`, at the very top of `generate_playlist_ds` (after the signature, before the existing `logger.info(...)` at ~line 115), insert:

```python
    _golden_dir = os.environ.get("PLAYLIST_GOLDEN_CAPTURE")
    _golden_kwargs = None
    if _golden_dir:
        _golden_kwargs = dict(
            artifact_path=artifact_path, seed_track_id=seed_track_id, mode=mode,
            length=length, random_seed=random_seed, pace_mode=pace_mode,
            overrides=overrides, allowed_track_ids=allowed_track_ids,
            excluded_track_ids=excluded_track_ids, single_artist=single_artist,
            anchor_seed_ids=anchor_seed_ids, pool_source=pool_source,
            artist_style_enabled=artist_style_enabled, artist_playlist=artist_playlist,
            sonic_weight=sonic_weight, genre_weight=genre_weight,
            min_genre_similarity=min_genre_similarity, genre_method=genre_method,
            internal_connector_ids=internal_connector_ids,
            internal_connector_max_per_segment=internal_connector_max_per_segment,
            internal_connector_priority=internal_connector_priority,
        )
```

Then, immediately before the final `return DsRunResult(...)` (~line 194), insert:

```python
    if _golden_kwargs is not None:
        import os as _os
        from tests.support.lossless_golden import dump_golden_inputs
        _os.makedirs(_golden_dir, exist_ok=True)
        _label = _os.environ.get("PLAYLIST_GOLDEN_LABEL", "capture")
        dump_golden_inputs(
            _golden_kwargs, result.track_ids,
            metrics.get("min_transition"), metrics.get("mean_transition"),
            _os.path.join(_golden_dir, f"{_label}.json"),
        )
```

Add `import os` to the module imports at the top if not present (it is not currently — add it beside `import json`).

**Note on `pier_bridge_config`:** it is intentionally omitted from the captured kwargs — in the artist/seeds paths the pier-bridge tuning rides in `overrides` (confirmed by the run log, which serializes all pier config under `overrides`). If a captured run passes a non-None `pier_bridge_config`, the replay would diverge: guard it — after building `_golden_kwargs`, add `assert pier_bridge_config is None, "golden capture does not yet support pier_bridge_config"`. If that assertion ever fires, extend capture to serialize it via the `__dataclass__` path.

- [ ] **Step 4: Write the replay bit-diff integration test**

Create `tests/integration/test_lossless_speedup_golden.py`:

```python
"""Bit-identical replay gate for the lossless-speedup work.

Loads each frozen golden fixture, replays generate_playlist_ds, and asserts
the ordered track_ids are IDENTICAL and min/mean transition match exactly.
This is THE regression gate for every optimization in the plan.
"""
import glob
import os
import pytest

from tests.support.lossless_golden import load_golden, replay_golden

FIXTURE_DIR = os.path.join("tests", "fixtures", "lossless_speedup")
FIXTURES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not FIXTURES, reason="no golden fixtures captured yet")
@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: os.path.basename(p))
def test_golden_bit_identical(fixture_path):
    golden = load_golden(fixture_path)
    artifact = golden["kwargs"]["artifact_path"]
    if not os.path.exists(artifact):
        pytest.skip(f"artifact missing: {artifact}")

    result = replay_golden(golden)

    assert list(result.track_ids) == list(golden["track_ids"]), (
        f"track_ids diverged for {os.path.basename(fixture_path)} — NOT lossless"
    )
    for key in ("min_transition", "mean_transition"):
        got, want = result.metrics.get(key), golden[key]
        if want is not None:
            assert got == want, f"{key} drift: {got!r} != {want!r} (ΔT != 0)"
```

- [ ] **Step 5: Verify the test is collectable and skips cleanly (no fixtures yet)**

Run: `python -m pytest tests/integration/test_lossless_speedup_golden.py -v`
Expected: skipped ("no golden fixtures captured yet"). Confirms wiring without needing fixtures.

- [ ] **Step 6: Commit**

```bash
git add src/playlist/ds_pipeline_runner.py tests/support/lossless_golden.py \
        tests/test_lossless_golden_unit.py tests/integration/test_lossless_speedup_golden.py
git commit -m "test(perf): golden capture hook + bit-identical replay gate"
```

---

## Task 2: Capture the three golden fixtures

**Files:**
- Create: `tests/fixtures/lossless_speedup/herbie.json`, `.../multiseed.json`, `.../porches.json` (generated, then committed)

**Interfaces:**
- Consumes: the capture hook from Task 1.
- Produces: the fixtures the gate in Task 1 (and every later task) runs against.

This task is operational, not TDD — it runs three real generations with capture on. Each needs the full stack (DB + Last.fm), so run them from the **main checkout** (not a stub worktree), matching how the GUI runs.

- [ ] **Step 1: Capture the Herbie artist + tag-steering fixture**

Reproduce the baseline run with capture on. From the main checkout, with the venv active:

```bash
PLAYLIST_GOLDEN_CAPTURE=tests/fixtures/lossless_speedup PLAYLIST_GOLDEN_LABEL=herbie \
  python main_app.py --artist "Herbie Hancock" --tracks 50 --tag "soul jazz"
```

(Confirm the exact CLI flags for tag steering via `docs/GOLDEN_COMMANDS.md`; if tag steering is GUI-only, capture it by driving `tools/serve_web.py` once with the same knobs — the hook fires regardless of entry point.)
Expected: `tests/fixtures/lossless_speedup/herbie.json` written, containing `track_ids` of length ~51.

- [ ] **Step 2: Capture a multi-seed (non-artist) fixture**

```bash
PLAYLIST_GOLDEN_CAPTURE=tests/fixtures/lossless_speedup PLAYLIST_GOLDEN_LABEL=multiseed \
  python main_app.py --seeds "<trackA>,<trackB>,<trackC>" --tracks 50
```

Pick any 3 seeds that produce a normal multi-pier run (see `docs/GOLDEN_COMMANDS.md` for a known-good seed set).

- [ ] **Step 3: Capture a second artist fixture with a different pool shape**

```bash
PLAYLIST_GOLDEN_CAPTURE=tests/fixtures/lossless_speedup PLAYLIST_GOLDEN_LABEL=porches \
  python main_app.py --artist "Porches" --tracks 50
```

- [ ] **Step 4: Verify all three replay bit-identically against themselves**

Run: `python -m pytest tests/integration/test_lossless_speedup_golden.py -v -m integration`
Expected: 3 passed (each fixture replays to its own captured output — this proves determinism of the seam before any optimization).
If any FAILS here, the seam is not deterministic for that run type — STOP and diagnose (likely a non-`overrides` input, e.g. `pier_bridge_config` or an unseeded path) before proceeding.

- [ ] **Step 5: Commit the fixtures**

```bash
git add tests/fixtures/lossless_speedup/herbie.json \
        tests/fixtures/lossless_speedup/multiseed.json \
        tests/fixtures/lossless_speedup/porches.json
git commit -m "test(perf): freeze 3 golden fixtures for lossless speedup gate"
```

---

## Task 3: cProfile baseline

**Files:**
- Create: `scripts/research/time_golden_replay.py` (timing + optional cProfile over a fixture replay)

**Interfaces:**
- Consumes: `replay_golden`, `load_golden` from `tests/support/lossless_golden.py`.
- Produces: `time_golden_replay.py --fixture <label> [--profile]` — prints wall-clock; with `--profile` writes cumulative-time hotspots. Used by the Universal verification block's timing step.

- [ ] **Step 1: Write the timing/profile script**

Create `scripts/research/time_golden_replay.py`:

```python
"""Time (and optionally profile) a golden-fixture replay — the timing gate
and the baseline hotspot ranking for the lossless-speedup work."""
import argparse
import cProfile
import os
import pstats
import time

from tests.support.lossless_golden import load_golden, replay_golden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="label, e.g. herbie")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sort", default="cumulative")
    args = ap.parse_args()

    path = os.path.join("tests", "fixtures", "lossless_speedup", f"{args.fixture}.json")
    golden = load_golden(path)

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        result = replay_golden(golden)
        prof.disable()
        pstats.Stats(prof).sort_stats(args.sort).print_stats(40)
    else:
        t0 = time.perf_counter()
        result = replay_golden(golden)
        dt = time.perf_counter() - t0
        print(f"[{args.fixture}] replay wall-clock: {dt:.1f}s  tracks={len(result.track_ids)}")

    assert list(result.track_ids) == list(golden["track_ids"]), "replay diverged from golden"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the profiler to capture the baseline hotspot ranking**

Run: `python scripts/research/time_golden_replay.py --fixture herbie --profile > docs/run_audits/lossless_speedup_baseline_profile.txt`
Expected: a cumulative-time table. Confirm the top entries match the static analysis (artist-identity resolution, `score_transition_edge`, the flex re-builds). **Use this ranking to reprioritize Tier 1 tasks if it disagrees with the estimates.**

- [ ] **Step 3: Commit the script and baseline profile**

```bash
git add scripts/research/time_golden_replay.py docs/run_audits/lossless_speedup_baseline_profile.txt
git commit -m "perf: golden-replay timing/profile harness + baseline profile"
```

---

## Task 4 (T1-a): Cache `resolve_artist_identity_keys` per candidate

The single biggest confirmed redundancy: an 11-delimiter regex parse, uncached, called ~2× per (beam-state × candidate). Pure function of the artist string → cache once per segment.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (`_beam_search_segment`; call sites ~`:1220,1446,1589`; caller of the parse)
- Read first: `src/playlist/artist_identity_resolver.py:137` (confirm `resolve_artist_identity_keys` signature + that it depends only on the artist string)
- Test: covered by the Universal verification block; add one micro-test below.

**Interfaces:**
- Produces: a `identity_keys_by_cand: dict[int, <return type>]` precompute local to `_beam_search_segment`, reused at every call site.

- [ ] **Step 1: Read the parse and its call sites**

Read `artist_identity_resolver.py` around line 137 and `beam.py` around 1200-1260, 1440-1460, 1580-1600. Confirm the argument is the candidate's artist string (indexed by candidate track index) and the return is hashable/reusable. Record the exact current call form.

- [ ] **Step 2: Add a micro-test that the cache equals the uncached result**

Create `tests/test_beam_identity_cache.py`:

```python
from src.playlist.artist_identity_resolver import resolve_artist_identity_keys


def test_identity_parse_is_pure_function_of_artist_string():
    a = resolve_artist_identity_keys("Miles Davis Quintet")
    b = resolve_artist_identity_keys("Miles Davis Quintet")
    assert a == b  # deterministic → safe to memoize by artist string
```

Run: `python -m pytest tests/test_beam_identity_cache.py -v` → PASS.

- [ ] **Step 3: Precompute the cache once per segment and reuse**

At the top of `_beam_search_segment`, after `candidates` is finalized, build:

```python
    identity_keys_by_cand = {
        c: resolve_artist_identity_keys(<artist-string-for-index-c>)
        for c in candidates
    }
```

Replace each of the ~3 call sites (`beam.py:1220,1446,1589`) that call `resolve_artist_identity_keys(...)` for a candidate with a lookup `identity_keys_by_cand[cand]`. **Do not** change the values used — same keys, just fetched once. If a call site resolves the *current path's last track* (a pier or already-chosen track, not a pool candidate), leave that one as a direct call (it is not in `candidates`); only the per-candidate calls are cached.

- [ ] **Step 4: Verify bit-identical + timing**

Run the Universal verification block. Expected: 3 fixtures PASS, fast suite green, `herbie` replay wall-clock lower than the Task 3 baseline. Record both numbers.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/beam.py tests/test_beam_identity_cache.py
git commit -m "perf(beam): cache artist-identity parse per candidate (bit-identical)

Golden: 3/3 fixtures identical track_ids, ΔT==0. herbie replay <before>s -> <after>s."
```

---

## Task 5 (T1-b): Reuse the already-computed sonic cosine

`edge_metric["S"]` (computed in `score_transition_edge`, `transition_metrics.py:219`) is the same value the local-sonic policy and diagnostics recompute via `np.dot(X_full_norm[current], X_full_norm[cand])` — `X_sonic_norm is X_full_norm` (`pier_bridge_builder.py:699/554`).

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (`:350` local-sonic policy; diagnostic recomputes `:1458,1600,1865`)

- [ ] **Step 1: Read `_apply_local_sonic_edge_policy` (`beam.py:336-377`) and the diagnostic sites (`:1458,1600,1865`)**

Confirm each recomputes the sonic cosine for the *same* `(current, cand)` pair whose `edge_metric` already holds `S`. Confirm `context.X_sonic_norm is X_full_norm` at the call site.

- [ ] **Step 2: Thread `edge_metric["S"]` into the recompute sites**

Where the code currently computes `np.dot(X_full_norm[current], X_full_norm[cand])` for the local-sonic penalty and the three diagnostics, pass in / reuse the `S` already present in the edge metric for that pair instead. Where a diagnostic is built for an edge that has no computed `edge_metric` yet (confirm during the read), leave it — only substitute where the identical `S` is already in hand.

- [ ] **Step 3: Verify bit-identical + timing**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0. (Same float value reused, so ΔT must be exactly 0 — if not, the two were not actually the same computation; investigate before committing.)

- [ ] **Step 4: Commit**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): reuse edge_metric S instead of recomputing sonic cos (bit-identical)"
```

---

## Task 6 (T1-c, T1-f): Hoist per-step BPM/onset targets and loop-local imports

`compute_step_log_bpm_target`/`compute_step_log_onset_target` (`beam.py:1134-1139,1174-1179`) depend only on `(pier_a, pier_b, step, interior_length)` (`pace_gate.py:22-36,67-82`) but run inside `for state in beam: for cand in candidates:`. The `from ... import ...` at `beam.py:1130-1133,1171-1172` also re-execute per candidate.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py`

- [ ] **Step 1: Read the step loop (`beam.py:1044` outer, `:1108` per-state, `:1117` per-candidate)**

Confirm the two target computations use no `cand`/`state` value. Confirm the local imports bind names used inside the loop.

- [ ] **Step 2: Move the imports to module scope**

Cut the `from ... import compute_step_log_bpm_target` / `compute_step_log_onset_target` (and any siblings on those lines) to the top-of-module imports in `beam.py`.

- [ ] **Step 3: Hoist the target computations to per-step**

Compute both targets once at the top of the `for step in range(interior_length):` body (before `for state in beam:`), into locals (e.g. `_bpm_target_this_step`, `_onset_target_this_step`), and reference those locals inside the candidate loop.

- [ ] **Step 4: Verify bit-identical + timing**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): hoist per-step bpm/onset targets + loop imports (bit-identical)"
```

---

## Task 7 (T1-d): Skip the guaranteed-zero energy penalty

`compute_energy_pace_penalty` (`beam.py:1191-1202`) is called per candidate whenever `energy_matrix is not None`, but returns exactly `0.0` when `energy_step_strength<=0 and energy_arc_strength<=0` (`pace_gate.py:179,186`) — the effective config state.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py`

- [ ] **Step 1: Read `beam.py:1191-1202` and `pace_gate.py:170-190`**

Confirm the function short-circuits to `0.0` when both strengths are ≤0, and that the strengths are available in scope once per segment (not per candidate).

- [ ] **Step 2: Guard the call with a once-per-segment strength check**

Compute `_energy_active = (energy_step_strength > 0.0) or (energy_arc_strength > 0.0)` once (near the other per-segment setup). Wrap the per-candidate call so that when `not _energy_active`, the penalty is taken as `0.0` without calling the function or indexing `energy_matrix`. Keep the exact `0.0` contribution so the score is unchanged.

- [ ] **Step 3: Verify bit-identical + timing**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0.

- [ ] **Step 4: Commit**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): skip zero-weight energy penalty call (bit-identical)"
```

---

## Task 8 (T1-e): Memoize the transition score by (prev_idx, cur_idx)

`_score_shared_transition` (`beam.py:658`, invoked `:1245`) → `score_transition_edge` is a pure function of the two track indices + fixed context within one `_beam_search_segment` call. Beam "diamonds" (multiple states landing on the same current track at a step) re-score identical pairs. Mirror the existing `genre_cache` (`beam.py:838`).

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py`

- [ ] **Step 1: Read `_score_shared_transition` (`beam.py:658-672`) and the `genre_cache` pattern (`:838-927`)**

Confirm the transition score depends only on `(prev_idx, cur_idx)` and objects fixed for the segment call (context, cfg, weights). Confirm it does NOT depend on the partial path / step / used-artist state. If it does (e.g. a path-dependent term folded in here), the pair-key memo is unsafe — STOP and leave this task out.

- [ ] **Step 2: Add a per-call `trans_cache`**

Alongside `genre_cache`, add `trans_cache: dict[tuple[int, int], <return type>] = {}`. In `_score_shared_transition`, key on `(prev_idx, cur_idx)`; return the cached value on hit, else compute, store, return. Scope: one `_beam_search_segment` call (reset per call, like `genre_cache`).

- [ ] **Step 3: Verify bit-identical + timing**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0. If the timing gate shows no improvement (diamonds rare at this beam width), keep only if it is a clean simplification; otherwise drop per the two-gate rule.

- [ ] **Step 4: Commit**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): memoize transition score by (prev,cur) within a segment (bit-identical)"
```

---

## Task 9 (T1-g): Hoist length-invariant work out of the flex/backoff loops

The big Tier-1 refactor. On the live path the segment candidate pool, the taxonomy genre-arc routing, and the roam corridor graph do **not** depend on interior length, yet `_build_segment_at` rebuilds them on every flex length (up to 3) and every backoff/expansion retry. Hoist them to per-segment scope and reuse.

**Files:**
- Modify: `src/playlist/pier_bridge_builder.py` (flex loop `:1980-2015`; `_build_segment_at` `~:1809`; `_run_segment_backoff_attempts` `~:1943`)
- Read: `src/playlist/pier_bridge/segment_pool_builder.py:189-309`; `src/playlist/pier_bridge/taxonomy_steering.py:293-353`; `src/playlist/pier_bridge/roam.py:24-60`; `src/playlist/pier_bridge/var_bridge.py:37-49`
- Follow the existing hoist precedent already in the file: `pair_sim_provider` (`:661-672`), `_segment_far_stats`/`relaxation_attempts` (`:1767-1805`).

**Interfaces:**
- Produces: a per-segment cache (computed once, before the flex/length loop) for `(segment_pool, genre_targets_prefix, roam_graph)` keyed by `(pier_a, pier_b, bridge_floor)`; `_build_segment_at(length)` reads from it instead of rebuilding.

- [ ] **Step 1: Prove invariance for each of the three, in code**

Read each builder and confirm on the LIVE path:
- **Pool** (`segment_pool_builder.py`): `interior_length` is referenced only inside `_build_dj_union_pool` (`:650,656,687,700,707`), gated on `pool_strategy=="dj_union"`; the live default is `segment_scored` (`config.yaml:299`) and `dj_bridging_enabled` is forced False when `genre_steering_enabled` (`pier_bridge_builder.py:504-508`). So the pool is length-invariant on the live path. **Confirm both conditions hold in the effective config** — if a run uses `dj_union`, this hoist is unsafe for that run; gate the hoist on `pool_strategy != "dj_union"`.
- **Genre routing** (`taxonomy_steering.py`): `canon_a/canon_b` (`:293-300`), shortest path (`:304-312`), mass filter (`:317`), waypoint vectors (`:322-333`) depend only on the piers; only the interpolation loop (`:342-353`) is `O(interior_length)`. Split the function so the pier-only part is reusable.
- **Roam** (`roam.py`): pure function of `(pier_a, pier_b, candidates, k, mutual_proximity)`; invariant once the pool is invariant.

- [ ] **Step 2: Add a golden fixture that exercises flex (guard against a narrow gate)**

Confirm at least one existing fixture (herbie) actually flexes a segment (the log shows `flexed=True (1/3)`); if not, capture a fixture that does (Task 2 pattern). The flex path must be covered by the gate before refactoring it.

- [ ] **Step 3: Hoist the pool build to per-segment scope**

In `build_pier_bridge_playlist`, before the flex-length loop (and before the backoff loop inside `_build_segment_at` re-enters), compute the segment pool once per `(pier_a, pier_b, bridge_floor)` and pass it into `_build_segment_at`. Because backoff *changes* `bridge_floor`, key the cache by `(pier_a, pier_b, bridge_floor)` so a floor-relaxation retry still rebuilds (that IS length-independent but floor-dependent). Reuse across the flex lengths at the same floor. Follow the `_segment_far_stats` hoist shape at `:1767-1805`.

- [ ] **Step 4: Split and hoist the genre-routing prefix**

Extract the pier-only prefix of `build_taxonomy_genre_targets` (canon + path + waypoint vectors) into a helper computed once per segment; keep only the `O(interior_length)` interpolation inside the per-length path.

- [ ] **Step 5: Cache the roam graph per segment**

Compute `segment_sonic_detour` once per segment (keyed like the pool) and reuse across lengths/retries. Gate on `roam_corridors_enabled`.

- [ ] **Step 6: Verify bit-identical + timing (the decisive one)**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0, and a **large** wall-clock drop on the flex fixtures (this targets the ~2/3 of build time in the flex re-runs). Record before/after per fixture.

- [ ] **Step 7: Commit**

```bash
git add src/playlist/pier_bridge_builder.py src/playlist/pier_bridge/taxonomy_steering.py
git commit -m "perf(pier-bridge): hoist length-invariant pool/genre-route/roam out of flex+backoff loops (bit-identical)

Golden 3/3 identical. Removes redundant per-length/per-retry rebuilds on the segment_scored live path."
```

---

## Task 10 (T2-a): Restrict pier similarity to candidate rows

`sim_to_a`/`sim_to_b` (`beam.py:820-821`) dot the pier against the full 41,179-track matrix, but only pool rows are read (`:1230,1256-1257`). Restrict the multiply. **Tier 2: must pass the bit-diff, or keep an exact-op variant.**

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py:820-821` and the read sites `:1230,1256-1257`

- [ ] **Step 1: Read `beam.py:815-825` and the two read sites**

Confirm `sim_to_a`/`sim_to_b` are indexed only by candidate/pool indices downstream. Determine whether indices are absolute (into the full matrix) or pool-relative — this dictates whether you restrict-and-reindex or keep absolute indexing.

- [ ] **Step 2: Compute the dot over candidate rows only**

Replace `np.dot(X_full_norm, X_full_norm[pier_a])` (full) with a product over just the candidate rows: `X_full_norm[cand_idx_array] @ X_full_norm[pier_a]`, and index the result by candidate position. Keep the exact same float dtype and operation order per row.

- [ ] **Step 3: Verify bit-identical (critical) + timing**

Run the Universal verification block. Expected: 3/3 identical, **ΔT == 0**.
- If ΔT == 0 on all fixtures → keep it (big win, ~98% off that op).
- If ΔT != 0 (BLAS blocked the submatrix gemv differently) → **revert** to full-matrix and instead index the full result (no FLOP saving) OR drop the task. Do NOT ship a bit-changing variant.

- [ ] **Step 4: Commit (only if ΔT==0)**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): restrict pier-sim dot to candidate rows (verified bit-identical)"
```

---

## Task 11 (T2-b, T2-c): Batch per-step sim loops + scalar BPM-distance fast path

Two more Tier-2 reassociation candidates: batch the per-step arc/waypoint similarity loops (`beam.py:1069-1078,1086-1102`) into one matrix-vector product, and give `bpm_log_distance` (`bpm_axis.py:36-44`, called `beam.py:1157,1182`) a scalar `math`-based fast path. Each **must pass the bit-diff**.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py` (`:1069-1078,1086-1102`); `src/playlist/pier_bridge/bpm_axis.py:36-44`

- [ ] **Step 1: Batch the per-step similarity loops**

Read `beam.py:1069-1102`. Where a per-candidate `np.dot(X_genre_for_sim[cand], target)` runs in a Python loop over the pool at a fixed step, replace with one `X_genre_for_sim[cand_idx_array] @ target` computed once per step and indexed per candidate.

- [ ] **Step 2: Verify bit-identical for Step 1; keep or revert**

Run the Universal verification block. If ΔT != 0 on any fixture, revert this sub-change (keep the loop) and note it. Commit only the batching that stays ΔT==0.

- [ ] **Step 3: Add a scalar fast path to `bpm_log_distance`**

In `bpm_axis.py:36-44`, add a scalar branch: when both args are plain floats, compute `abs(math.log2(a / b))` (mirroring `_calibrate_transition_cos`'s `math` usage at `vec.py:35-61`) instead of routing 0-d arrays through numpy.

- [ ] **Step 4: Verify bit-identical for Step 3; keep or revert**

Run the Universal verification block. `math.log2` vs `np.log2` may differ at ULP — if ΔT != 0 on any fixture, revert the scalar path (keep numpy) and drop this sub-change.

- [ ] **Step 5: Commit (only the sub-changes that stayed ΔT==0)**

```bash
git add src/playlist/pier_bridge/beam.py src/playlist/pier_bridge/bpm_axis.py
git commit -m "perf(beam): batch per-step sim loops + scalar bpm-distance where verified bit-identical"
```

---

## Task 12 (T3-a): Defer edge_component diagnostics to beam survivors only

Structural, highest-ceiling, highest-risk. Today the `edge_component` diagnostic dict + `list(state.edge_components) + [edge_component]` history copy is built for every gated successor (~`beam_width × pool`), but only `beam_width` survive `next_beam.sort(...)[:beam_width]` (`beam.py:1653-1656`). Build diagnostics for survivors only.

**Files:**
- Modify: `src/playlist/pier_bridge/beam.py:1472-1501,1614-1643,1653-1656`; the beam state representation

- [ ] **Step 1: Read the full successor-construction + selection region (`beam.py:1440-1660`)**

Map exactly what `next_beam.sort` reads: `combined_score`, and for the minimax objective a running min edge-`T` (`_state_min_edge`). Confirm the `edge_components` list is NOT read by selection — only by post-hoc reporting. Identify every downstream consumer of `edge_components` to guarantee survivors still carry a complete, correct history.

- [ ] **Step 2: Track the minimax key as a scalar on the state**

Instead of deriving the running min edge-`T` from the (to-be-deferred) `edge_components` list, carry it as a scalar updated incrementally when a successor is formed. Verify it reproduces `_state_min_edge` exactly for every state (add a temporary assert during development).

- [ ] **Step 3: Defer diagnostic dict + history copy until after truncation**

Restructure so the per-successor loop stores only what selection needs (score + scalar min-T + the chosen candidate + a back-pointer). After `next_beam.sort(...)[:beam_width]`, build the `edge_component` dict and extend `edge_components` for the surviving `beam_width` states only.

- [ ] **Step 4: Verify bit-identical (critical) + timing**

Run the Universal verification block. Expected: 3/3 identical, ΔT==0, and a build-time drop from `pool/beam_width` fewer dict builds. Because this touches state, keep the temporary `_state_min_edge` assert enabled for this run; remove it before commit once green.

- [ ] **Step 5: Commit**

```bash
git add src/playlist/pier_bridge/beam.py
git commit -m "perf(beam): build edge_component diagnostics for survivors only (verified bit-identical)"
```

---

## Task 13: Re-profile, document results, update backlog

**Files:**
- Modify: `docs/TIME_OPTIMIZATION.md` (record the lossless wins landed + new baseline)
- Create: `docs/run_audits/lossless_speedup_after_profile.txt`

- [ ] **Step 1: Re-run the profiler and timing on all fixtures**

Run `python scripts/research/time_golden_replay.py --fixture herbie --profile > docs/run_audits/lossless_speedup_after_profile.txt` and record before/after wall-clock for all 3 fixtures.

- [ ] **Step 2: Update `docs/TIME_OPTIMIZATION.md`**

Add a "Lossless wins landed (2026-07-03)" section: the per-fixture before/after, which tasks shipped, which Tier-2/3 items were dropped for failing the bit-diff, and the new baseline the (still-suspended) 90s ceiling will start from.

- [ ] **Step 3: Full slow suite + commit**

Run `python -m pytest -q` (include slow, so the golden gate runs) — expect green.

```bash
git add docs/TIME_OPTIMIZATION.md docs/run_audits/lossless_speedup_after_profile.txt
git commit -m "docs(perf): record lossless-speedup results + new baseline"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** Tier 0 → Tasks 1-3; Tier 1 → Tasks 4-9; Tier 2 → Tasks 10-11; Tier 3 → Task 12; results/backlog → Task 13. Negative result (no segment parallelism) is honored — no task attempts it.
- **The bit-diff gate is the universal test.** Perf tasks don't "write a failing test first"; the golden harness (Task 1) is written once and every task runs it. Micro-tests are added only where a cache/memo needs an independent correctness check (Tasks 1, 4).
- **Fail-closed on ΔT != 0.** Tasks 10-12 explicitly revert-or-keep-exact-op rather than ship a bit-changing variant — this is the strict bit-identical bar.
- **Line anchors are approximate** (Global Constraints) — every optimization task starts by reading the region to confirm.
