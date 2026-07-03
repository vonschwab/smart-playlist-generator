# Wiring Status

**Living tracker** of what is actually *wired and live* vs *shipped-default* vs *off* vs
*known-broken*, across the sonic ⊗ genre ⊗ pace axes and the pier-bridge infrastructure. Verified
against `config.yaml` + code, **not** intentions. Update this doc whenever a wiring state changes.

Last verified: **2026-07-03** (post-SP-B, post-tag-steering).

## Governing principle

Behavior resolves through three layers, and they deliberately differ (see
[`ARCHITECTURE.md`](ARCHITECTURE.md) "Reading the defaults"):

1. **Dataclass defaults** (`src/playlist/pier_bridge/config.py`) — rollback baseline, every
   experimental lever `off`.
2. **`config.example.yaml`** — the *shipped* template a fresh clone copies.
3. **`config.yaml`** (gitignored) — the *live* config on one machine.

This doc's job is to flag where **shipped ≠ live** — those are the drifts that bite. Verification
is by reading real config + generation logs, not "the config looks right."

**States:** ✅ LIVE (shipped on) · 🟡 LIVE-ONLY (on in `config.yaml`, absent from
`config.example.yaml` — a shipped gap) · ⚪ OFF (default off) · 🔴 KNOWN BUG · 📦 REMOVED/ARCHIVED

---

## Sonic

| Component | State | Notes |
|---|---|---|
| **MuQ embedding** (`X_sonic_muq`, 512-d, `MuQ-MuLan-large`, `center_l2`) | ✅ LIVE | The **sole** sonic space. `sonic_variant_override` resolves to `muq` (shipped, live, artifact all agree). Reproducible: `muq` stage → `muq_runner.py` → `muq_sidecar.npz` → `fold_muq`. |
| MERT (768-d) + tower blend (163-d) + `transition_weights`/`tower_weights` | 📦 REMOVED | Deleted by SP-B. Data archived at `data/archive/mert_2026/`. No runtime path, no config keys. |
| Transition calibration (`TRANSITION_CALIB_BY_VARIANT`) | ✅ LIVE | `muq` centered at 0.594; single-sourced logistic; unresolvable variant raises. |

## Selection / pier-bridge

| Component | State | Notes |
|---|---|---|
| Pier-bridge beam search | ✅ LIVE | Sole topology; legacy greedy constructor is dead code. |
| **`artist_style.enabled`** (medoid-clustered piers) | 🟡 **LIVE-ONLY** | `config.yaml: true`, **`config.example.yaml: false`** → the shipped template runs the *legacy per-seed* pier path, not medoid clustering. Big shipped-vs-live divergence; likely a template gap. |
| Cohesion / genre / sonic / pace mode axes | ✅ LIVE | Default `dynamic`. `pace_mode` has no `discover` level (the other three do). |

## Collapse prevention (anti-sag scoring)

| Component | State | Notes |
|---|---|---|
| Anti-center (SP2, `seed_character_mode: anti_center` @ 2.0) | ✅ LIVE | Scoring anti-sag; partial (dreampop plateaus ~101%). "hubness" variant deleted. |
| Mini-piers (SP3, `mini_pier_enabled`) | ✅ LIVE | Structural anti-sag; closes the residual (dreampop 103%→63%). |

## Weak-edge recovery cascade (post-beam)

| Pass | State | Notes |
|---|---|---|
| Variable bridge length (add-only) | ✅ LIVE | `variable_bridge_length: true` shipped. |
| tail-DP | ✅ LIVE | `tail_dp` shipped on. |
| **Edge repair (break-glass)** | 🟡 **LIVE-ONLY** | `config.yaml` has `edge_repair: {enabled: true, ...}`; **`config.example.yaml` has no `edge_repair:` block** → a fresh clone runs it OFF. Shipped gap (`CLEANUP_LIST.md`). |
| Edge delete (remove-only) | ✅ LIVE | `edge_delete` shipped on. |
| Roam corridors | 🟡 LIVE-ONLY | On in `config.yaml`, absent from `config.example.yaml`. Advanced/opt-in. |
| `generation_budget_s` | ✅ LIVE (=0) | Shipped `0` = time limit disabled (quality-first). 90 s ceiling is a design target. |

## Genre

| Component | State | Notes |
|---|---|---|
| Authority (`release_effective_genres` via `authority.py`) | ✅ LIVE | Sole writer = `publish`; sole reader = `authority.py`. Artifact bakes it (`genre_source: graph`). |
| Taxonomy graph (`layered_genre_taxonomy.yaml`, ~v0.26) | ✅ LIVE | Living, GUI-grown; hub-guarded similarity. |
| Genre metric = `max` (soft penalty, never a hard gate) | ✅ LIVE | Soft-cosine alternative rejected, never merged. Two soft demotions (pool compatibility + beam pair-floor). |
| Genre arc steering (`genre_steering_source: taxonomy`) | ✅ LIVE | Rebuild-robust; `dense` source raises if its sidecar is unusable. |

## Pace / energy

| Component | State | Notes |
|---|---|---|
| BPM + onset-rate hard bands + soft rhythm penalty | ✅ LIVE | Embedding-independent (DB features); survived the MuQ migration. Beatless pier disables its own BPM band. |
| Energy arc / pace-contour (`energy_*_strength`) | ⚪ OFF | Parked — measured redundant with MuQ for smoothness; revive only for *intentional* directional arcs (`CLEANUP_LIST.md`). |

## Features

| Component | State | Notes |
|---|---|---|
| Tag-steering — **pool lever** (`tag_steering_pool_blend` 0.5) | ✅ LIVE | Blends the tag target into the dense admission centroid; mode-agnostic. Inert with no tags. |
| Tag-steering — **pier lever** (`tag_steering_pier_weight` 0.3) | 🟡 partial | Gated by `artist_style.enabled` — so dormant in the shipped template (see above), live in `config.yaml`. |
| Tag-steering stage-2 (beam lever) | 📦 not built | Designed; gate never tripped. |
| Popular-seeds (`popular_seeds_mode`) / Oops-All-Bangers (`popularity_mode`) | ⚪ OFF | Both default off. |

---

## Known open gaps / bugs (see `CLEANUP_LIST.md` for detail)

- 🟡 **`edge_repair:` absent from `config.example.yaml`** — fresh clone runs break-glass repair off while live runs it on.
- 🟡 **`artist_style.enabled: false` in `config.example.yaml`** — fresh clone runs the legacy per-seed pier path (no medoid clustering; tag-steering pier lever dormant).
- 🔴 **Edge-repair vs reporter T-mismatch** — repair has flagged edges the final reporter scores healthy (T ≈ 0.66–0.79 vs a 0.30 floor); root-cause blocked until `edge_repair` logs which trigger arm fired. Do not retune floors against the reporter until resolved.
- ⚪ **Fixer deadzone (0.30–~0.75)** — ugly-but-legal edges above every trigger floor get no attention; a deliberate policy question, not a bug.
- 📝 **`CLAUDE.md:115` stale** — "until the Task-10 rebuild" describes a rebuild that already happened (the live artifact has only `X_sonic_muq*`).
- 🔴 **`sonic_mode` admission floors are MERT-calibrated, not MuQ.** `min_sonic_similarity` in `mode_presets.py` (strict 0.28 / narrow 0.18 / dynamic 0.08) are MERT cosine percentiles (p75/p50/p25/p10, "recalibrated 2026-06" — for MERT). SP-B removed MERT but left these absolute floors; MuQ's cosine distribution differs (its transition calibration centers at 0.594 vs MERT's 0.32), so `sonic_mode` strict/narrow may over- or under-gate on MuQ. **Needs recalibration against the MuQ distribution** (cf. `FLOOR_RECALIBRATION_DISTRIBUTIONS.md`). The blend weights are embedding-agnostic and fine; only the absolute floors are suspect.
- ✅ **RESOLVED (moot):** the earlier `config.example` `transition_weights` mismatch bug — SP-B removed the knob entirely.
