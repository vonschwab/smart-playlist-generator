# Lossless generation speedup — design

**Date:** 2026-07-03
**Status:** Design — awaiting spec review, then implementation plan
**Author:** pairing session (Dylan + Claude)

## Goal

Reduce pier-bridge playlist **generation wall-clock** as far as possible **without changing the output** — the generated playlist must be **bit-identical** (same track IDs, same order, same per-edge T values) to today's. Speed comes only from computing the *same* result *faster*, never from searching less.

Baseline (the run this design was built from): 51-track Herbie Hancock artist run, Soul-Jazz tag steering, `2026-07-02_233514_Herbie_Hancock_49ac9d.log` — ~3m10s total, of which the pier-bridge segment build is ~178s. The three flex segments (variable-bridge) consume ~2/3 of that build.

## Non-goals (explicitly out of scope)

- **No lossy levers.** Everything in `docs/TIME_OPTIMIZATION.md` (narrow the beam, cheaper flex with a narrower beam, scale beam to segment difficulty, early-exit on convergence) *changes which tracks are chosen*. All of it is out of scope here. When we later re-arm the 90s ceiling, that doc is the backlog; this one is orthogonal.
- **No quality/behavior change.** Not even a one-ULP change to a score that could flip a beam tie. The bar is bit-identical, verified, not "no worse."
- **No config/knob changes** that alter search. (Removing genuinely dead/zero-weight work is in scope; changing a live weight is not.)

## Definition of "lossless" and how we enforce it

**Lossless = bit-identical output**, proven per-change by a **golden bit-diff**, not asserted.

Generation is deterministic at the pier-bridge boundary: the beam search is RNG-free, and the one seeded RNG (`candidate_pool.py:577`, `np.random.default_rng(random_seed)`, `random_seed=0` in production) is consumed once upstream, before segment building. Tie-breaks are explicit and index-anchored (`beam.py:1652-1655` stable sort; `segment_pool_builder.py:556-558,934` secondary tie-break on integer track index), so there is no set/dict-iteration nondeterminism in selection. Therefore `build_pier_bridge_playlist` is a pure function of its inputs.

The full artist path is **not** reproducible end-to-end (Last.fm recency is run-to-run variable — see the `playlist-testing` skill trap catalog), so the golden fixture is snapshotted at the **deterministic seam**: the exact arguments entering `generate_playlist_ds` (equivalently `build_pier_bridge_playlist`) — `anchor_seed_ids`, the restricted bundle / `allowed_ids`, the full resolved `overrides`, `genre_params`, `random_seed`. All of these are already emitted in the run's JSON line (`ds_pipeline_runner.generate_playlist_ds:183`). Replay those frozen inputs → assert identical output.

**Verification protocol (applied to every change):**
1. **Bit-diff gate:** replay each golden fixture; assert output `track_ids` (order-sensitive) and per-edge `T` array are identical to the frozen golden. Any diff → the change is not lossless → revert or keep the exact-op variant. A float-reassociating change that happens to change one bit fails this gate, by design.
2. **Timing gate:** the change must help wall-clock (profile or wall-clock on the fixtures). A change that *adds* complexity but doesn't measurably save time isn't worth it — drop it. A change that is bit-identical *and* a net simplification (e.g. removing a redundant recompute, hoisting an import) is kept even if its individual timing delta is below measurement noise — it can't hurt and it de-clutters a hotspot.

A change ships only if it passes the bit-diff gate; among passing changes, keep the ones that either save measurable time or simplify the code (drop the ones that add complexity for no measured gain).

## Negative result (documented so it is not re-proposed)

**Segment builds cannot be parallelized losslessly.** Each segment's candidate pool and diversity gating are filtered by the *actual tracks the prior segments chose* — the global dedup set (`global_used`, `pier_bridge_builder.py:898`, written `:2792-2793`), the artist-identity dedup set (`used_track_keys`, `:913` / `:2794-2797`), per-artist caps (`global_non_seed_artist_counts`, `:937` / `:2798-2801`), and the cross-segment `min_gap` boundary artists (`recent_boundary_artists`, `:936` / `:2806-2842`, fed into the beam at `beam.py:951-955`). This is a genuine sequential *information* dependency and it is the diversity/dedup guarantee (CLAUDE.md Layer-2 item 11), not an accident. Running segments concurrently against the seed-only state would let two segments pick the same track or violate cross-segment gap/caps — a different, likely invalid playlist. **Concurrency across segments is therefore off the table for a bit-identical result.** (The beam *within* a segment is fully deterministic and RNG-free, which is what makes all the per-segment compute wins below safe.)

The wall-clock **budget coupling** (`deadline` / `_pb_build_start`, `pier_bridge_builder.py:1713-1718`, checked `:2099-2121,2186,2250-2251`) is also a latent ordering dependency (a segment's relaxation depth depends on cumulative elapsed time). It is inert today (`generation_budget_s <= 0` ⇒ `deadline=None`), but any future change here must keep it inert or anchor it to segment-relative time, or timing-dependent output creeps in.

## Architecture: verification foundation + a tiered optimization catalog

Two parts: (A) the harness that makes "lossless" checkable, built first; (B) a catalog of compute wins, tiered by how they preserve output.

### Part A — Verification foundation (Tier 0, built first, nothing ships without it)

1. **Golden fixtures.** Capture the `generate_playlist_ds` input tuple for 3 representative runs and freeze them plus their output (`track_ids` + per-edge `T`): (a) this Herbie artist + tag-steering run; (b) a multi-seed non-artist run; (c) a second artist run with a different pool shape (e.g. Porches, from `TIME_OPTIMIZATION.md`). Storage: serialized inputs under `tests/fixtures/lossless_speedup/` (indices + the restricted bundle slice, not the whole 453MB artifact).
2. **Replay + bit-diff harness.** A test that loads a fixture, calls the real pier-bridge entry, and asserts bit-identical `track_ids` and `T`. This is the gate for every subsequent change. Mark `integration`/`slow`; skip if the artifact is absent.
3. **Profiler pass.** One `cProfile` of a real generation (or a fixture replay that exercises the beam) to get *measured* hotspot ranking. The win estimates in Part B are from static analysis; the profile confirms/reorders priorities before we spend effort. This is also the honest baseline the timing gate compares against.

### Part B — Optimization catalog

Line anchors are from static analysis and had minor cross-agent drift; **confirm exact lines at edit time**. Each item lists its bit-identity argument.

#### Tier 1 — Provably bit-identical by construction (compute the identical FP ops, fewer times)

| ID | Change | Anchor | Bit-identity argument | Est. win |
|----|--------|--------|-----------------------|----------|
| T1-a | **Cache `resolve_artist_identity_keys` per candidate** once at the top of `_beam_search_segment`; reuse at all call sites | `artist_identity_resolver.py:137`; callers `beam.py:1220,1446,1589` | Pure function of the candidate's artist string; currently uncached and called ~2×/(state×candidate). Same inputs → same output, memoized. | **Largest single win** (agent est. seconds off ~13s/segment) |
| T1-b | **Reuse `edge_metric["S"]`** instead of recomputing the sonic cosine in the local-sonic policy + diagnostics | computed `transition_metrics.py:219`; recomputed `beam.py:350,1458,1600,1865`; `X_sonic_norm is X_full_norm` (`pier_bridge_builder.py:699/554`) | The recomputed dot is the *same value* on the *same arrays*; reuse the already-computed scalar. | Moderate (1–2 `np.dot`/edge × ~20k) |
| T1-c | **Hoist BPM/onset step-targets** out of the per-candidate loop to per-step | `beam.py:1134-1139,1174-1179`; no candidate dependence (`pace_gate.py:22-36,67-82`) | Value depends only on `(pier_a,pier_b,step,interior_len)`; identical across all candidates at a step. | Modest, free |
| T1-d | **Skip `compute_energy_pace_penalty`** when `energy_step_strength<=0 and energy_arc_strength<=0` (checked once/segment) | `beam.py:1191-1202`; early-return `pace_gate.py:179,186` | Function returns exactly `0.0` in that case; skipping the call+index is identical. Effective config has energy weights at 0. | Small, matches a known dead-work case |
| T1-e | **Memoize the transition score** by `(prev_idx,cur_idx)` within a segment, mirroring the existing `genre_cache` | add `trans_cache` at `beam.py:658-672` (pattern at `:838`) | `score_transition_edge` is a pure function of the two indices + fixed context for one segment call; beam "diamonds" re-score identical pairs. | NEEDS-VERIFICATION on magnitude; mechanism safe |
| T1-f | **Move per-candidate local `import`s** to module scope | `beam.py:1130-1133,1171-1172` | Same bound names, resolved once. | Small, free |
| T1-g | **Hoist length-invariant work out of the flex/backoff loops** — segment candidate pool, taxonomy genre-arc routing (canon + shortest-path + waypoint vectors), roam corridor graph — compute once per segment, reuse across flex lengths and backoff/expansion retries | pool `segment_pool_builder.py:189-309` (`interior_length` only used in dead `dj_union` branch; live path `segment_scored`); taxonomy `taxonomy_steering.py:293-336` (only interpolation `:342-353` is length-dependent); roam `roam.py:24-60` (called `pier_bridge_builder.py:1473-1478`) | On the live path these are pure functions of `(pier_a,pier_b,bridge_floor,pool state)` and do **not** depend on interior length; rebuilding per length/retry recomputes identical results. Follows the pattern already applied to `pair_sim_provider` (`:661-672`) and `_segment_far_stats` (`:1767-1805`). | **Large on flex segments** — this is where the 3× (and compounding backoff) cost lives |

#### Tier 2 — Numerically equivalent, must PASS the golden bit-diff (reassociates FP; keep exact-op variant if it fails)

| ID | Change | Anchor | Risk | Est. win |
|----|--------|--------|------|----------|
| T2-a | **Restrict `sim_to_a`/`sim_to_b` to candidate rows** instead of the full 41,179-track matrix | `beam.py:820-821`; only pool rows read `:1230,1256-1257` | gemv on a submatrix *should* be per-row identical, but BLAS blocking could differ at ULP — must bit-diff. If it changes a bit, index the rows from the full result instead (keeps exact op, still skips no FLOPs — or accept only if diff is empty). | ~98% cut on that op |
| T2-b | **Batch per-step arc/waypoint similarity loops** into one matrix-vector product | `beam.py:1069-1078,1086-1102` | Batched `X@v` reassociates sums vs per-row `np.dot`; small ULP risk → must bit-diff. | Modest (per-step) |
| T2-c | **Scalar math fast-path for `bpm_log_distance`** (avoid routing 0-d arrays through numpy ufuncs) | `bpm_axis.py:36-44`; called `beam.py:1157,1182` | `math.log2` vs `np.log2` may differ at ULP → must bit-diff. Precedent: `_calibrate_transition_cos` already uses `math.exp` (`vec.py:35-61`). | Removes numpy dispatch × ~20-40k calls |

#### Tier 3 — Structural (biggest ceiling, highest effort, lands last, prototype-and-diff)

| ID | Change | Anchor | Note |
|----|--------|--------|------|
| T3-a | **Defer the `edge_component` diagnostic dict + `edge_components` history-list copy until after beam truncation** — build for the ~`beam_width` survivors only, not every ~`beam_width×pool` gated successor | `beam.py:1472-1501,1614-1643`; truncation `:1653-1656` | Selection needs only `combined_score` and (for minimax) a running min edge-T, which can be tracked as a scalar on the state instead of read from the deferred list. Touches the beam state representation → careful correctness pass; prototype and bit-diff before landing. Roughly `pool/beam_width` (~35–70×) fewer dict builds. |

## Build sequence

1. **Tier 0** — golden fixtures + replay/bit-diff harness + cProfile baseline. (Gate for everything else.)
2. **Tier 1 leaf wins** (T1-a … T1-f) — smallest, safest, biggest confirmed single win (T1-a). Bit-diff + time after each.
3. **Tier 1 flex hoisting** (T1-g) — the real refactor of `_build_segment_at`; largest Tier-1 win on flex segments. Bit-diff + time.
4. **Tier 2** (T2-a … T2-c) — land only those that pass the bit-diff; keep exact-op variants for any that don't.
5. **Tier 3** (T3-a) — prototype, bit-diff carefully, land last.

Land incrementally (one item ≈ one commit), each with its bit-diff + timing evidence in the message. Re-profile after Tier 1 to re-rank the remainder against measured numbers.

## Testing

- The **replay/bit-diff harness** (Tier 0) is the primary regression: every change re-runs all 3 golden fixtures and asserts identical `track_ids` + `T`.
- Add a fast unit test per new cache/memo (e.g., `resolve_artist_identity_keys` cache returns values equal to uncached) to catch a broken cache independent of the golden run.
- Full `pytest -m "not slow"` stays green.
- Per the `playlist-testing` maintenance protocol: if any change exposes a fidelity trap, add a Trap Catalog row.

## Risks & open items

- **ULP flips (Tier 2).** The whole reason for the bit-diff gate. Mitigated by construction: fail-closed (revert/keep exact-op) on any diff.
- **Line-number drift.** Anchors are approximate; confirm at edit time. The god-class (`pier_bridge_builder.py` ~5.3k LOC, `beam.py`) is a hotspot — prefer extracting helpers over inflating it.
- **Fixture representativeness.** Three fixtures may miss a code path (e.g., a run that hits micro-pier fallback or heavy relaxation). Add fixtures as new paths surface.
- **Roam on/off.** Roam corridors are active in the baseline run; confirm the live default config state so T1-g's roam-cache work targets the real path.
- **Worktree discipline.** Implementation must happen in a dedicated worktree on its own branch (simultaneous sessions are the norm here). This session launched in the shared checkout, and mid-session `EnterWorktree` is known-risky here (subagents/hooks anchor to the launch dir) — the implementation phase should be a session launched with cwd set to the worktree.

## Cross-references

- `docs/TIME_OPTIMIZATION.md` — the *lossy* backlog (orthogonal to this).
- `playlist-testing` skill — deterministic-harness rules; Last.fm nondeterminism trap.
- Memory `feedback_generation_time_budget` — the (suspended) 90s ceiling.
- Baseline log: `logs/playlists/2026-07-02_233514_Herbie_Hancock_49ac9d.log`.
