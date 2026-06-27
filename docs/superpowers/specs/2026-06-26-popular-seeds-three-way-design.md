# Spec — Popular Seeds Three-Way (OFF / ON / 🔥) + popularity-first piers

**Date:** 2026-06-26
**Status:** Design approved (brainstorming), pending spec review → plan
**Branch:** `worktree-oops-bangers-poolgate` (same program as the pool-gate; the OOPS→🔥 coupling
modifies the pool-gate's `_resolve_popular_seeds`).
**Related:** `docs/superpowers/specs/2026-06-26-oops-all-bangers-poolgate-design.md` (the bridge gate;
this spec is the **pier** side).

---

## 1. Problem (diagnosed)

Popular-seed pier selection "reaches deep." For a Stereolab artist-mode run the 6 piers landed at
Last.fm ranks **#5, #7, #8, #20, #28, #29** — three are genuine deep cuts. Root cause, traced through
`artist_style._medoids_for_cluster:447,465`:

- Pier selection is **one medoid per sonic cluster** (k clusters → k piers). Each medoid maximizes
  `0.7·sonic_centrality + 0.3·duration_typicality + popular_seeds_weight·popularity`.
- With `popular_seeds_weight = 0.5` (default), popularity is a **minority additive bias** (max 0.5)
  outweighed by sonic-centrality + duration (max ~0.72). So each pier is "the sonically-central track
  of its cluster, nudged toward popularity" — not "the hit." Hit-less clusters yield deep cuts.
- (Secondary, working as intended: the recency/freshness filter removes the user's most-played tracks —
  which are the biggest hits — before clustering. This is why the piers start at #5 with #1–#4 absent.
  The user confirmed recency should stay exactly as-is.)

## 2. Decision

**Popular Seeds becomes a three-way control** parallel to Bangers (OFF / ON / OOPS):

| Mode | Pier selection |
|---|---|
| **OFF** | sonic clustering only — one medoid per cluster, no popularity (today's behavior, byte-identical) |
| **ON** | clustered medoids + popularity bias, weight **raised 0.5 → 1.0** (hits-leaning, sonic spread retained) |
| **🔥** | **pure top-N popularity** — piers = the artist's top `target_pier_count` Last.fm tracks; **no sonic-diversity constraint on *selection*** ("the hits are the hits") |

**🔥 constrains *selection*, not *arrangement*.** It only decides *which* tracks are piers. The
pier-bridge still optimizes their **order** (seed-order permutation search) and beam-searches the
bridges between them for sonic flow — the playlist is sequenced for cohesion exactly like any other.
"Collapse" means the *arc* may flatten because the hits sit close in sonic space, not that ordering is
skipped.

- **🔥 fallback is *count* only:** if the artist has fewer resolved hits than `target_pier_count` (after
  recency + version-dedup), 🔥 uses however many hits exist — it **never reaches into non-hits to pad**.
  If *zero* hits resolve (uncached artist), 🔥 falls back to OFF clustering, logged loudly (never fail).
- **Recency/freshness unchanged** — applies in all three modes exactly as today.

**Coupling: Bangers OOPS forces `popular_seeds_mode = 🔥`.** "Oops, All Bangers" is the no-compromise
mode — banger-gated bridges deserve unambiguous-hit piers, and 🔥's hits-over-cohesion philosophy is
OOPS's. Consistent with OOPS already commandeering sonic/pace: at the extreme, Bangers owns everything
but genre. When OOPS is on, the Popular Seeds dropdown defers to it (forced 🔥). Bangers ON/OFF leave
`popular_seeds_mode` as the user set it.

## 3. Architecture

### 3.1 Threading: `popular_seeds: bool` → `popular_seeds_mode: str`

Replace the boolean through the stack (single-user app — replace, don't deprecate, per the project's
"activate fixes" rule). Values: `"off" | "on" | "fire"`.
- `request_models.py` (`popular_seeds` → `popular_seeds_mode`, from_worker_args + to_worker_args)
- `schemas.py` (`GenerateRequestBody`) + `web/src/lib/types.ts`
- `worker.py:1312` (artist dispatch) — pass `popular_seeds_mode`
- `playlist_generator.py::create_playlist_for_artist` — param `popular_seeds_mode: str = "off"`

### 3.2 OOPS coupling resolver (replaces pool-gate Task 3's `_resolve_popular_seeds`)

```
_resolve_popular_seeds_mode(popular_seeds_mode: str, popularity_mode: str) -> str
    # OOPS forces "fire"; else the user's mode. Artist-mode-only by construction.
    return "fire" if str(popularity_mode).lower() == "oops" else str(popular_seeds_mode or "off").lower()
```
The pool-gate's `_resolve_popular_seeds(...) -> bool` is superseded by this `-> str` resolver. The
existing call site in `create_playlist_for_artist` switches from the bool to the mode string.

### 3.3 ON path (existing mechanism, weight 1.0)

Unchanged except the default: when mode is `"on"` (or `"fire"`), load `popularity_values`
(`load_artist_popularity_values`, as today) and set `medoid_popularity_weight`. Config default
`popular_seeds_weight` **0.5 → 1.0** (`config.example.yaml` + the `style_cfg_raw.get(..., 0.5)` fallback
at `playlist_generator.py:1755`).

### 3.4 🔥 path: popularity-first pier selection (new)

New selector in `src/playlist/artist_style.py`:
```
select_popular_piers(indices, popularity_values, target_pier_count) -> list[int]
    # Rank the artist's candidate tracks by popularity (NaN/unknown excluded), take the top
    # target_pier_count by 0-based Last.fm rank. No sonic-diversity constraint. Returns fewer
    # than target_pier_count when fewer hits exist; returns [] when no track has a popularity score.
```
Wiring (`playlist_generator.py` ~1766, the `cluster_artist_tracks(...)` call): clustering still runs
(it seeds the **bridge candidate pool** — the per-cluster external candidates), but for 🔥 the **piers
(medoids) are overridden** with `select_popular_piers(...)`. If it returns `[]` (uncached artist),
fall back to the clustered medoids (OFF behavior) with a warning. The cleanest carrier is a
`pier_selection: str` argument threaded into `cluster_artist_tracks` (or applied at the call site to
the returned medoid list) — pinned in the plan. The candidate pool, anchor_seed_ids, and downstream
pier-bridge are unchanged; only *which track ids are piers* differs.

### 3.5 GUI

`web/src/components/GenerateControls.tsx`: the Popular Seeds **checkbox → dropdown** (Off / On /
🔥 — label e.g. "🔥 Pure hits"), in the mode row near the Bangers dropdown. `web/src/lib/types.ts`
gains `popular_seeds_mode`. When Bangers = OOPS, the dropdown shows forced-🔥 (disabled/deferred,
mirroring how sonic/pace defer to OOPS).

## 4. Data flow

```
GUI dropdown (Off / On / 🔥)
  → GenerateRequestBody.popular_seeds_mode
  → worker → create_playlist_for_artist(popular_seeds_mode=...)        # ARTIST MODE only
        mode = _resolve_popular_seeds_mode(popular_seeds_mode, popularity_mode)   # OOPS -> "fire"
        if mode in {"on","fire"}: load popularity_values; medoid_popularity_weight = popular_seeds_weight (1.0)
        cluster_artist_tracks(...)            # clusters seed the bridge pool
        if mode == "fire": piers = select_popular_piers(...) or <fallback to medoids, logged>
  → pier-bridge optimizes pier ORDER (seed-order search) + bridges for cohesion AS USUAL
        (🔥 only fixed *which* tracks are piers; the arc may flatten if the hits cluster sonically,
         but the sequence is still optimized — no pier-bridge change for 🔥)
```

Seed mode is untouched (it never calls `create_playlist_for_artist`; user-chosen seeds stay).

## 5. Config

```yaml
playlists:
  ds_pipeline:
    artist_style:
      popular_seeds_weight: 1.0   # was 0.5; ON-mode medoid popularity weight (tunable)
```

## 6. Testing

Unit (no real data in the worktree; live verification by the user):
- **`select_popular_piers`** — synthetic indices + popularity vector: returns top-N by rank; excludes
  NaN/unknown; returns fewer than target when hits are scarce; returns `[]` when none have scores.
- **`_resolve_popular_seeds_mode`** — `oops` → `"fire"` regardless of the user's mode; else passthrough;
  `off`/`None` → `"off"`.
- **Threading** — `popular_seeds_mode` survives `from_worker_args`/`to_worker_args` round-trip.
- Live (user): Stereolab at ON (piers shallower than the #20/#28/#29 baseline) and 🔥 (piers = the top
  ~6 hits, arc may flatten); OOPS auto-runs 🔥; OFF unchanged; budget < 90s.

## 7. Out of scope / deferred

- ON's spread-vs-popularity balance beyond `popular_seeds_weight = 1.0` (calibrate by ear later).
- 🔥 sonic-spread constraint — explicitly rejected by design ("the hits are the hits").
- Seed-mode popularity (piers are user-chosen there).
