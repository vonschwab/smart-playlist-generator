# Three-Axis Wiring & Calibration Status

**Living tracker** of what is actually *wired and live* vs *inert / broken / mis-calibrated* across the three matching axes — **sonic ⊗ genre ⊗ pace** — plus the pier-bridge infra they run on. Update this doc whenever a wiring state changes. Verified by reading real generation logs, not by reading config.

Last updated: 2026-06-21.

## Governing principle (paid for in full)

**Data-first, calibrate-last.** Calibration fits numbers to a *fixed* feature geometry. If a wiring change moves the geometry afterward, the numbers are invalid. We learned this the expensive way: the entire pace co-equal-axis calibration (strict k=20 / narrow k=5 / arc strengths / worst-edge eval-gate) was fit against the **legacy `robust_whiten` sonic space** because MERT had been silently reverted — all of it is provenance-invalid and must be re-run against MERT. Do **not** calibrate any axis until the wiring that changes what that axis sees has landed.

**Verification is mandatory and specific.** Every wiring change is proven on a **real playlist with real logs** (see Verification Protocol). "It should work" / "config looks right" / a green metric is not proof — read the log lines that show the feature firing.

---

## Status table

States: ✅ LIVE · 🟢 FIXED (this session, verified) · 🔴 BROKEN (configured but cannot act) · ⚪ INERT (no-op / cosmetic) · 🟡 DESIGN-NEEDED · ⏳ DEFERRED/BLOCKED · 🤝 IN-FLIGHT (another session)

### Sonic axis
| Component | State | Evidence / Notes | Action |
|---|---|---|---|
| MERT embedding (`X_sonic_variant=mert`, 768-d) | 🟢 FIXED | Re-folded 2026-06-21; log `Using precomputed sonic variant 'mert' from artifact key X_sonic_mert`, X_sonic (40393,768). Was 🔴 (06-18 rebuild reverted to `robust_whiten`). | After ANY analyze/rebuild, re-fold + verify variant in log (see [project_mert_migration]). |
| 2DFTM key-invariant harmony (96-d) | 🟢 FIXED | Re-folded 2026-06-21; `X_sonic_harmony=(N,96)`, tower_dims `[9,57,96]`. Was 20-d absolute-key legacy. | Same re-fold discipline. |
| Tower blend rollback (`X_sonic_tower_weighted`, 162-d) | ✅ LIVE | Present as rollback; loader pre-scaled path uses baked variant directly (`embedding_setup.py:77`). | — |
| Sonic admission floors (MERT-recal: strict 0.28 / narrow 0.18 / dynamic 0.08) | 🟡 PARTIAL | Committed & MERT-scaled, but narrow 0.18 + genre hard gate starved Charli XCX to 47 candidates. Floor may be right; the *combination* with genre is the problem. | Revisit in three-axis calibration (after genre wiring). |
| `playlists.sonic.sim_variant: tower_weighted` | ⚪ INERT | Stale config key; inert under the pre-scaled path (`embedding_setup.py` ignores it when a variant is baked). | Align to `mert` or remove (cosmetic). |

### Genre axis
| Component | State | Evidence / Notes | Action |
|---|---|---|---|
| Graph-sourced genre vectors (`X_genre_raw/smoothed` from `release_effective_genres`) | ✅ LIVE | `build_config.genre_source=graph`, `graph_authority=True`, 39,104 graph tracks. Preserved through the re-fold. | — |
| Genre **arc steering** (`genre_steering_source=dense`) | 🔴 BROKEN | Dense sidecar `data_matrices_step1_genre_emb_dim64.npz` (built 06-12) has a vocab mismatch → silently ignored → `X_genre_dense=None` → `no usable g_targets` on **every** segment → genre arc INACTIVE. | **Decide:** switch source → `taxonomy` (uses in-artifact `X_genre_raw`, rebuild-robust) **or** rebuild dense sidecar. Recommend taxonomy. Part of #3. |
| Genre **hard admission gate** (`min_genre_sim=0.4`, ensemble) | 🟡 DESIGN-NEEDED | Hard gate rejected 310 + sonic floor → 47-candidate pool → 14/30 one-artist playlist + cascade grind. | **#3 brainstorm:** hard → soft/relaxable? Changes the pool all axes see. |
| Genre compatibility soft penalty | ✅ LIVE | `Genre compatibility penalty applied: penalized=6721 strength=0.200`. | — |
| Genre runtime **layered admission** flip (SP4 item E) | ⏳ KNOWN-PENDING | Artifact genre is graph-sourced; the *layered runtime admission* enhancement is not flipped on. | Decide in #3 whether needed. |
| Dense genre sidecar (dim64) | 🔴 BROKEN | Vocab mismatch, ignored at load. Root cause of dead arc steering above. | Resolve with steering-source decision. |

### Pace axis
| Component | State | Evidence / Notes | Action |
|---|---|---|---|
| Energy soft penalty in beam (`energy_arc_*`, `energy_step_*`) | ⚪/⏳ | Wired (merged b337835). strict/narrow values are **legacy-space, invalid**; dynamic=0.0 (reverted); off=0.0. | Recalibrate vs MERT (last). |
| Energy admission rescue (`pace_rescue_k_energy`) | ⚪/⏳ | Wired. strict k=20 / narrow k=5 are **legacy-space, invalid**; dynamic/off=0. Mechanism is space-independent and sound. | Recalibrate vs MERT (last). |
| BPM / onset admission bands | ✅ LIVE | `BPM admission gate ... rejected=N`, `Onset admission band ... rejected=N`. | — |
| `bpm_trust_min_onset_rate` (beatless) | ✅ LIVE | Shipped master `ad9403e`. | — |
| GUI pace `off` option | ✅ LIVE | Added this session (b337835). | — |

### Pier-bridge / infra
| Component | State | Evidence / Notes | Action |
|---|---|---|---|
| Relaxation-cascade bound (empty-pool short-circuit + 40s wall-clock budget) | 🟢 FIXED | `bc942d5`. Charli XCX repro 274s→58s; budget bails logged; suite green. | — |
| `no usable g_targets` warning flood | 🟢 FIXED | Demoted to debug in beam; logged once/segment in builder (`bc942d5`). Confirm on real artist-mode run. | Confirm in GUI. |
| Generation **cancellation** | 🟢 INTEGRATED | Cherry-picked `87401b9` (from `fix/generation-cancellation`) 2026-06-21. Process-global hook + `OperationCancelled(BaseException)` (not swallowed by `except Exception`); checkpoints at segment boundary / expansion attempt / beam step (verified in valid loops); 6 unit tests pass; full suite green. Composes with the cascade budget. | Confirm click-cancel end-to-end in the GUI. |
| Never-fail greedy fallback (term-pool) | ✅ LIVE / 🤝 | Fills segments when the beam can't; genre-aware version in-flight `worktree-genre-aware-greedy-fallback` (`0b028d4`). | Coordinate before editing the fallback. |
| dj_bridging | ✅ LIVE (when enabled) | Ladder route + waypoints fire when `dj_bridging_enabled`. | — |

### Calibration (depends on ALL wiring above)
| Item | State | Action |
|---|---|---|
| Three-axis calibration (sonic floors + genre gate + pace rescue/arc), eval-gated on worst-edge | ⏳ BLOCKED | Run **once**, on the stable post-wiring foundation, against MERT. Diverse seeds, real playlists, `BPM loaded` verified. |

---

## Other sessions' in-flight branches (coordinate — do not merge blindly)
| Branch | Commit | Overlaps our scope? |
|---|---|---|
| `fix/generation-cancellation` | `6a8bd28` | ✅ INTEGRATED to master (`87401b9`, 2026-06-21). Owner can delete the branch. |
| `worktree-genre-aware-greedy-fallback` | `0b028d4` | YES — pier-bridge fallback quality (#3). |
| `wip-gui-logging-genre-vocab` | `f9ec957` | Maybe — genre vocab/logging. |
| `worktree-phase1-album-adjudicator` | — | Genre adjudication (longer-horizon). |
| `codex/ai-genre-model-prior` | — | Genre model prior (longer-horizon). |

---

## Wiring sequence (dependency-ordered)

1. **Sonic geometry** — ✅ DONE (MERT + 2DFTM re-folded, verified live).
2. **Genre arc steering source** — reactivate dead arc (`dense → taxonomy`, or rebuild dense). 🟡 part of #3.
3. **Genre admission policy** — hard gate → soft/relaxable (the starvation fix). 🟡 #3 design decision.
4. **(Optional) Genre layered admission flip** — decide in #3.
5. **Coordinate in-flight branches** — integrate `fix/generation-cancellation`; align with `worktree-genre-aware-greedy-fallback`.
6. **Cosmetic** — `sim_variant` key cleanup.
7. **THEN: three-axis calibration vs MERT** — pace last (it acts on the pool genre+sonic admission produces).

Steps 2–4 are the **#3 brainstorm** (genre steering + admission are entangled — one design pass).

---

## Verification protocol (every change)

Per change, generate a **real playlist** and **read the log** (CLAUDE.md + `playlist-testing` skill). Confirm:

1. **Data present:** `BPM loaded: N/N`, `Using precomputed sonic variant 'mert'`, no `vocabulary mismatch` for the sidecar you rely on.
2. **The change fired:** the specific log line for the feature (e.g. genre arc targets built, gate tally shifted, rescue `admitted=N`).
3. **Gate tally / pool health:** `admission gate ... rejected`, `Candidate pool: ... admitted=N`, `pool_after_gate`, `pool too small` / `budget exceeded` (should be rare).
4. **Completion + budget:** `Pier+Bridge complete`, wall-time < 90s (hard ceiling).
5. **No regression adjacent to the change.**

Artist-mode (typed-artist) paths go through the **real worker** (DB clustering is not covered by `generate_like_gui`); seeds-mode goes through `generate_like_gui`. Use the path that matches what you changed.
