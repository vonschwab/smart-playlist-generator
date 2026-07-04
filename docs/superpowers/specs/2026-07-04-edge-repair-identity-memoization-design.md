# Edge-repair identity-key memoization — design

**Date:** 2026-07-04
**Status:** Design — approved in-session, awaiting spec review, then implementation plan
**Author:** pairing session (Dylan + Claude)
**Parent effort:** `docs/superpowers/specs/2026-07-03-lossless-generation-speedup-design.md` (this is a re-prioritized continuation — see "Why this instead of T1-g")

## Goal

Reduce pier-bridge playlist **generation wall-clock** with **bit-identical output** (same `track_ids`, same order, ΔT == 0), by removing redundant recomputation inside the post-order `edge_repair` pass. Speed comes only from computing the *same* values *fewer times* — never from repairing fewer edges or changing any acceptance decision.

This is the same discipline and the same golden bit-diff gate as the parent lossless-speedup effort; edge_repair was flagged there as "a separate future target" and the fresh profile promotes it to the top of the queue.

## Why this instead of T1-g (the evidence)

The parent plan's next task was T1-g (hoist length-invariant pool/genre-route/roam out of the variable-bridge flex + backoff loops). A fresh `cProfile` of the `porches` golden replay at current HEAD (`docs/run_audits/` — 96.66 s total, load-independent) refutes T1-g's "large win" estimate:

| Region | Cumulative | Share | Notes |
|---|---|---|---|
| `choose_segment_length` (flex loop) | 61.9 s | 64% | — |
| ↳ `_beam_search_segment` | **59.1 s** | 61% | **length-dependent → cannot be hoisted losslessly** |
| ↳ pool builds + roam + overhead (the flex/beam gap) | **2.67 s** | 2.8% | T1-g's real hoistable target |
| ↳ taxonomy genre-routing | **0.06 s** | 0.06% | effectively free |
| **`repair_playlist_edges` (post-loop)** | **24.9 s** | **25.8%** | this design |

T1-g targets the 2.67 s of pool/roam rebuilds (roam is OFF by default — `roam_corridors_enabled` absent from `config.yaml` — so a third of it is dead work); reusing pool builds across the 15→9 flex/backoff rebuilds saves **~1 s (≈1%)**, at the cost of the riskiest refactor in the plan (restructuring the ~5.3k-LOC `pier_bridge_builder.py` flex+backoff loop, with a subtle "reuse across flex AND backoff-floors" cache-keying burden, and an unproven length-invariance assumption on the pool under `progress_arc`).

`edge_repair` is **~26%** and is cured by the *same memoization pattern already accepted as T1-a*, at ~15–20× the payoff and far lower risk. The parent plan explicitly authorizes this: "Use this ranking to reprioritize Tier 1 tasks if it disagrees with the estimates."

## Root cause (the redundancy)

In `src/playlist/repair/edge_repair.py`, `_candidate_refusal_reasons` is called once per `(edge_position × candidate)` — 7,881 times on the porches fixture. Each call recomputes **pure functions of a track index**:

- **`identity_keys_for_index(bundle, idx)`** (`identity_keys.py:61`) → frozen `TrackIdentityKeys(artist_key, title_key)`. The block at `edge_repair.py:178-185` rebuilds the identity keys of *every* current playlist track (~54) on *every* candidate check → ≈ 7,881 × 54 ≈ 425k of the profiled 470,188 calls (13.3 s cumulative). Uncached.
- **`_cap_artist_keys_for_idx(bundle, idx, cfg)`** (`edge_repair.py:116`) → `set[str]`. Called per-track in `_non_seed_artist_counts_after_replacement` (artist-cap count) and per-neighbor in the `min_gap` sub-loop (`:209,216`). 99,746 calls, 10.7 s cumulative. Internally calls `resolve_artist_identity_keys` (128k calls, 13.6 s — the T1-a function).

`artist_identity_cfg` is a parameter of `repair_playlist_edges`, fixed for the whole call, and the `bundle` is immutable. Therefore, **within one `repair_playlist_edges` pass, both functions are pure functions of the integer track index.** They are recomputed hundreds of thousands of times for a few thousand distinct indices.

## Design

Single file: `src/playlist/repair/edge_repair.py`. No changes elsewhere.

### Core — per-repair-pass memo (the primary win)

1. At the top of `repair_playlist_edges`, create two caches scoped to the call:
   - `ident_cache: dict[int, TrackIdentityKeys] = {}`
   - `cap_cache: dict[int, set[str]] = {}`
2. Thread them (as an optional `cache` argument, defaulting to `None`) into `_candidate_refusal_reasons`, `_non_seed_artist_counts_after_replacement`, and `_cap_artist_keys_for_idx`. When `cache is None`, the helpers compute directly (existing unit tests and any other callers stay byte-for-byte unchanged).
3. Replace the direct calls with cached lookups keyed by `int(idx)`:
   - `identity_keys_for_index(bundle, idx)` → `ident_cache` lookup (populate on miss).
   - `_cap_artist_keys_for_idx(bundle, idx, cfg)` → `cap_cache` lookup (populate on miss).
   - Guard on membership (`idx in cache`), not truthiness — `_cap_artist_keys_for_idx` can legitimately return an empty set.

Cache lifetime = one `repair_playlist_edges` call, mirroring `beam.py`'s `genre_cache` / `trans_cache` (per-segment) and `segment_pool_cache`. Fresh dicts per call ⇒ no cross-generation or cross-bundle leakage. **Do not** use a module-level `functools.lru_cache`: its key would have to include the unhashable `bundle` and would persist/leak across generations — rejected.

### Cache-key safety

Keyed by the integer track index only. `bundle` and `artist_identity_cfg` are invariant within the call. The cache is **not** keyed by playlist position, so an accepted swap (which changes which index sits at which position) never invalidates an entry — the index→keys mapping is immutable for the pass.

### Bit-identity argument (stronger than Tier-2)

The memoized functions are pure functions of `idx` given the fixed `(bundle, cfg)`; the memo returns the identical object/value recomputation would. All operations are string/set — **no floating point, no reassociation** — so there is not even a ULP risk. Output is bit-identical by construction, and the golden gate proves it empirically.

### Optional stretch (separate, independently gated commits — only if the core underdelivers)

These remove the per-candidate *set/dict rebuilds* that survive the per-index memo. Higher value, more structural, land only with their own bit-diff pass:

- **Hoist `existing_track_keys`** (`edge_repair.py:178-185`) to once per `edge_position`: the set of current tracks' `track_key`s (excluding `replace_position`) is identical for all candidates at that position.
- **Incremental artist-cap counts:** precompute the base non-seed artist counts per `edge_position` once, then per candidate only add the candidate's keys at `replace_position`, instead of re-summing all `current_indices`.

## Expected result

`identity_keys_for_index` (13.3 s / 470k) and `_cap_artist_keys_for_idx` (10.7 s / 99k) collapse to distinct-index counts (a few thousand calls). Realistic: **`edge_repair` 24.9 s → single digits, ≈ 15–20 s off the 96.66 s total (~16–20%).** Recorded against the profiler per the parent's two-gate rule (ships only if bit-diff passes AND it measurably saves time or is a net simplification).

## Testing

- **Golden bit-diff gate (primary):** `python -m pytest tests/integration/test_lossless_speedup_golden.py -v` — porches replays with identical `track_ids` and ΔT == 0. Run after the core memo and after each optional stretch commit.
- **Micro-test** (`tests/test_edge_repair_identity_cache.py`): for a handful of indices, the cached lookup equals the uncached `identity_keys_for_index` / `_cap_artist_keys_for_idx` result — catches a broken cache independent of the golden run.
- **Fast suite:** `python -m pytest -q -m "not slow"` stays green.
- **Timing:** `scripts/research/time_golden_replay.py --fixture porches --profile` before/after; quote load-independent cumulative numbers, never a single wall-clock run.

## Risks & open items

- **Fixture coverage.** Only `porches.json` exists (herbie/multiseed still TODO — they need the GUI path). Porches exercises both the flex path and a heavy edge_repair pass, so it gates this change, but single-fixture coverage is thin; note it and add fixtures as other paths surface.
- **Missed call site.** The one real correctness risk is routing the cache through *every* path (including the `min_gap` and artist-cap sub-loops). A missed site is only slower, never wrong — but defeats the purpose; verify all `identity_keys_for_index` / `_cap_artist_keys_for_idx` sites in the file consult the cache.
- **God-class caution.** `edge_repair.py` is smaller than the `pier_bridge_builder.py` / `beam.py` hotspots; keep the change to memoization + optional local hoists, don't inflate it.

## Environment & workflow constraints

- **Golden gate needs absolute-path overrides** (worktree/data caveat, memory `feedback_worktree_data_absolute_overrides`):
  - `PLAYLIST_GOLDEN_ARTIFACT="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"`
  - `PLAYLIST_GOLDEN_DB="C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"`
- **Shared master checkout** (a second session is live): commit with `git commit --only <path>` and check the full staged set with `git diff --cached --name-only` first (memory `feedback_shared_checkout_commit_only`). Never a bare `git commit`; never `git add -A`/`-u`.
- **Bit-identical is the hard gate.** Any `track_ids` diff or ΔT != 0 ⇒ revert before committing.

## Cross-references

- Parent effort: `docs/superpowers/specs/2026-07-03-lossless-generation-speedup-design.md`, plan `docs/superpowers/plans/2026-07-03-lossless-generation-speedup.md` (Task 4 = T1-a, the accepted memoization precedent).
- Memory `project_lossless_generation_speedup` (foundation, landed wins, profiler-trust rule).
- `docs/TIME_OPTIMIZATION.md` — the *lossy* backlog (orthogonal).
