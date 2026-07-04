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
| **`artist_style.enabled`** (medoid-clustered piers) | ✅ LIVE | `config.yaml: true` and **`config.example.yaml: true`** (aligned 2026-07-03) → both shipped template and live run medoid-clustered piers. |
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
| **Edge repair (break-glass)** | ✅ LIVE | `edge_repair: {enabled: true, ...}` in both `config.yaml` and `config.example.yaml` (block added 2026-07-03). |
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
| Tag-steering — **pier lever** (`tag_steering_pier_weight` 0.3) | ✅ LIVE | Gated by `artist_style.enabled`, now `true` in both `config.yaml` and `config.example.yaml` (aligned 2026-07-03) — no longer dormant in the template. |
| Tag-steering stage-2 (beam lever) | 📦 not built | Designed; gate never tripped. |
| Popular-seeds (`popular_seeds_mode`) / Oops-All-Bangers (`popularity_mode`) | ⚪ OFF | Both default off. |

---

## Known open gaps / bugs (see `CLEANUP_LIST.md` for detail)

- ✅ **RESOLVED 2026-07-03:** `edge_repair:` block added to `config.example.yaml` (mirrors live, on) — no longer a shipped gap.
- ✅ **RESOLVED 2026-07-03:** `artist_style.enabled: true` set in `config.example.yaml` (matches live) — fresh clone now runs medoid piers + the tag-steering pier lever.
- 🔴 **Edge-repair vs reporter T-mismatch** — repair has flagged edges the final reporter scores healthy (T ≈ 0.66–0.79 vs a 0.30 floor); root-cause blocked until `edge_repair` logs which trigger arm fired. Do not retune floors against the reporter until resolved.
- ⚪ **Fixer deadzone (0.30–~0.75)** — ugly-but-legal edges above every trigger floor get no attention; a deliberate policy question, not a bug.
- 🟡 **`sonic_mode`'s legacy absolute floors are INERT (cosmetic tech-debt, not a live bug).** Every active mode sets `sonic_admission_percentile > 0` (0.75/0.60/0.40/0.20), and `candidate_pool.py:658-666` **replaces** the absolute floor (`min_sonic_similarity`) with an adaptive percentile of the seed's *own* sonic-similarity distribution — embedding-agnostic, self-calibrating on MuQ; the absolute floors apply only if `sonic_admission_percentile == 0` (no shipped preset does). The **misleading "MERT p75/p50/…" comments were fixed 2026-07-03** (re-annotated as inert legacy in `mode_presets.py` + the tests). Residual = actually deleting the dead values (deferred: they're a tested plumbing contract — see `CLEANUP_LIST.md`).
- ✅ **RESOLVED (moot):** the earlier `config.example` `transition_weights` mismatch bug — SP-B removed the knob entirely.
